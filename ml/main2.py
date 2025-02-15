import asyncio
import websockets
import torch
import cv2
import numpy as np
import os
import json

from dotenv import load_dotenv
from PIL import Image
from matplotlib import colormaps

# -------------------------------
# MoGE Dependencies
# -------------------------------
import trimesh
import utils3d
from moge.model import MoGeModel
from moge.utils.io import save_glb

# -------------------------------
# DepthAnythingv2 Dependencies
# -------------------------------
from depth_anything_v2.dpt import DepthAnythingV2

# -------------------------------
# Gemini Dependencies
# -------------------------------
from google import genai
from google.genai import types
# import google.generativeai as genai

# -------------------------------
# SAM Dependencies
# -------------------------------
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

# A set to keep track of connected clients (if needed)
connected_clients = set()

# A helper function to parse Gemini JSON output
def parse_json(json_output: str) -> str:
    """
    Remove markdown fences from a JSON output if present.
    """
    if json_output.startswith("```json"):
        json_output = json_output.replace("```json", "", 1)
    if json_output.endswith("```"):
        json_output = json_output[:-3]
    return json_output.strip()

async def ai_server(websocket, path = None):
    connected_clients.add(websocket)
    try:
        print("Initializing AI models upon connection...")

        # -------------------------------
        # Inference Optimization Settings
        # -------------------------------
        torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

        # -------------------------------
        # MoGE Initialization (on cuda:0)
        # -------------------------------
        moge_model_device = torch.device("cuda:0")
        moge_model = MoGeModel.from_pretrained("Ruicheng/moge-vitl").to(moge_model_device)

        # -------------------------------
        # DepthAnythingv2 Initialization (on cuda:1)
        # -------------------------------
        cmap = colormaps.get_cmap('Spectral')
        depth_device = torch.device("cuda:1")
        depth_anything = DepthAnythingV2(
            encoder='vitl',
            features=256,
            out_channels=[256, 512, 1024, 1024],
            max_depth=20
        )
        depth_checkpoint_path = 'checkpoints/depth_anything_v2_metric_hypersim_vitl.pth'
        depth_anything.load_state_dict(torch.load(depth_checkpoint_path, map_location='cpu'))
        depth_anything = depth_anything.to(depth_device).eval()

        # -------------------------------
        # Gemini Initialization
        # -------------------------------
        load_dotenv()
        gemini_client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
        model_name = "gemini-2.0-flash"
        bounding_box_system_instructions = (
            "Return bounding boxes as a JSON array with labels. "
            "Never return masks or code fencing. Limit to 1 object."
        )
        safety_settings = [
            types.SafetySetting(
                category="HARM_CATEGORY_DANGEROUS_CONTENT",
                threshold="BLOCK_ONLY_HIGH",
            ),
        ]
        # We'll use a default object name; this could be dynamic.
        object_name = "window"

        # -------------------------------
        # SAM Initialization (on cuda:2)
        # -------------------------------
        sam_device = torch.device("cuda:2")
        sam2_checkpoint = "checkpoints/sam2.1_hiera_large.pt"
        model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
        sam2_model = build_sam2(model_cfg, sam2_checkpoint, device=sam_device)
        predictor = SAM2ImagePredictor(sam2_model)
        
        print("All AI models initialized. Ready to process images.")

        async for data in websocket:
            try:
                # -------------------------------
                # 1. Decode the received image bytes
                # -------------------------------
                nparr = np.frombuffer(data, np.uint8)
                cv2_image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                if cv2_image is None:
                    await websocket.send("error:Invalid image data".encode())
                    continue

                height, width, _ = cv2_image.shape
                # Convert to RGB for models that expect it.
                np_image = cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGB)

                # -------------------------------
                # 2. Update SAM predictor with the new image
                # -------------------------------
                predictor.set_image(np_image)

                # -------------------------------
                # 3. Depth Inference with DepthAnythingv2
                # -------------------------------
                with torch.cuda.device(depth_device):
                    metric_depth = depth_anything.infer_image(cv2_image)

                # Normalize the depth output to [0, 255]
                depth_min = metric_depth.min()
                depth_max = metric_depth.max()
                norm_depth = (metric_depth - depth_min) / (depth_max - depth_min + 1e-6) * 255.0
                norm_depth = norm_depth.astype(np.uint8)
                # Apply colormap (RGB) and then convert to BGR for JPEG encoding
                colored_depth = (cmap(norm_depth/255.0)[:, :, :3] * 255).astype(np.uint8)
                colored_depth = cv2.cvtColor(colored_depth, cv2.COLOR_RGB2BGR)
                # JPEG compression (aggressive quality setting)
                encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), 30]
                success, depth_bytes = cv2.imencode('.jpg', colored_depth, encode_params)
                if not success:
                    await websocket.send("error:Failed to encode depth image".encode())
                    continue
                await websocket.send(b"depth:" + depth_bytes.tobytes())
                print("Depth image sent.")

                # -------------------------------
                # 4. MoGE Inference & Raw Mesh Generation
                # -------------------------------
                # Prepare the image tensor for MoGE (expects RGB and normalized to [0,1])
                moge_image = np_image.copy()  # Already in RGB
                moge_image_tensor = torch.tensor(moge_image / 255.0, dtype=torch.float32, device=moge_model_device).permute(2, 0, 1)
                output = moge_model.infer(moge_image_tensor, apply_mask=True)
                # Extract output data
                points = output['points'].cpu().numpy()
                depth_moge = output['depth'].cpu().numpy()
                points_mask = output['mask'].cpu().numpy()
                intrinsics = output['intrinsics'].cpu().numpy()
                # Compute normals
                normals, normals_mask = utils3d.numpy.points_to_normals(points, mask=points_mask)
                # A threshold for edge detection (as in the original code)
                threshold = 0.03
                # Build raw mesh
                faces, vertices, vertex_colors, vertex_uvs = utils3d.numpy.image_mesh(
                    points,
                    moge_image.astype(np.float32) / 255.0,
                    utils3d.numpy.image_uv(width=width, height=height),
                    mask=points_mask & ~(utils3d.numpy.depth_edge(depth_moge, rtol=threshold, mask=points_mask) &
                                          utils3d.numpy.normals_edge(normals, tol=5, mask=normals_mask)),
                    tri=True
                )
                # Adjust vertices and UVs
                vertices = vertices * [1, -1, -1]
                vertex_uvs = vertex_uvs * [1, -1] + [0, 1]
                # Save the mesh to a GLB file and then read its bytes
                raw_mesh_filename = 'mesh.glb'
                save_glb(raw_mesh_filename, vertices, faces, vertex_uvs, moge_image)
                with open(raw_mesh_filename, "rb") as f:
                    mesh_bytes = f.read()
                await websocket.send(b"mesh:" + mesh_bytes)
                print("Raw mesh sent.")

                # -------------------------------
                # 5. Gemini API Call to get bounding box
                # -------------------------------
                # Convert cv2 image to PIL Image (RGB)
                PIL_image = Image.fromarray(np_image)
                prompt = f"Detect the 2d bounding box of the {object_name}."
                response = gemini_client.models.generate_content(
                    model=model_name,
                    contents=[prompt, PIL_image],
                    config=types.GenerateContentConfig(
                        system_instruction=bounding_box_system_instructions,
                        temperature=0.5,
                        safety_settings=safety_settings,
                    )
                )
                if not response.text.strip().startswith("```json"):
                    raise ValueError("Gemini API did not return valid JSON output")
                bb_json = parse_json(response.text)
                bounding_box = json.loads(bb_json)[0]
                # The Gemini output is expected to have "box_2d" with four values scaled by 1000.
                y1 = int(bounding_box["box_2d"][0] / 1000 * height)
                x1 = int(bounding_box["box_2d"][1] / 1000 * width)
                y2 = int(bounding_box["box_2d"][2] / 1000 * height)
                x2 = int(bounding_box["box_2d"][3] / 1000 * width)
                # Draw bounding box on a copy of the original image
                bbox_image = cv2_image.copy()
                cv2.rectangle(bbox_image, (x1, y1), (x2, y2), (0, 0, 255), 2)
                success, bbox_bytes = cv2.imencode('.jpg', bbox_image, [int(cv2.IMWRITE_JPEG_QUALITY), 30])
                if success:
                    await websocket.send(b"bbox:" + bbox_bytes.tobytes())
                    print("Bounding box image sent.")

                # -------------------------------
                # 6. SAM Inference & Segmentation Overlay
                # -------------------------------
                input_box = np.array([x1, y1, x2, y2])
                masks, scores, _ = predictor.predict(
                    point_coords=None,
                    point_labels=None,
                    box=input_box[None, :],
                    multimask_output=False,
                )
                print(f"SAM segmentation score: {scores}")
                # Create segmentation overlay (red tint for the segmented object)
                mask = masks[0].astype(np.uint8)
                red_mask = np.zeros_like(cv2_image)
                red_mask[:, :, 2] = 255  # Red channel
                alpha = 0.5
                overlay = cv2.addWeighted(cv2_image, 1, red_mask, alpha, 0)
                segmented_image = np.where(mask[:, :, None] == 1, overlay, cv2_image)
                success, seg_bytes = cv2.imencode('.jpg', segmented_image, [int(cv2.IMWRITE_JPEG_QUALITY), 30])
                if success:
                    await websocket.send(b"seg:" + seg_bytes.tobytes())
                    print("Segmentation overlay sent.")

                # -------------------------------
                # 7. Build Segmented Mesh from SAM Segmentation
                # -------------------------------
                # Convert the segmented image to RGB for MoGE mesh generation
                segmented_moge_image = cv2.cvtColor(segmented_image, cv2.COLOR_BGR2RGB)
                faces_seg, vertices_seg, vertex_colors_seg, vertex_uvs_seg = utils3d.numpy.image_mesh(
                    points,
                    segmented_moge_image.astype(np.float32) / 255.0,
                    utils3d.numpy.image_uv(width=width, height=height),
                    mask=points_mask & ~(utils3d.numpy.depth_edge(depth_moge, rtol=threshold, mask=points_mask) &
                                         utils3d.numpy.normals_edge(normals, tol=5, mask=normals_mask)),
                    tri=True
                )
                vertices_seg = vertices_seg * [1, -1, -1]
                vertex_uvs_seg = vertex_uvs_seg * [1, -1] + [0, 1]
                seg_mesh_filename = 'segmented_mesh.glb'
                save_glb(seg_mesh_filename, vertices_seg, faces_seg, vertex_uvs_seg, segmented_moge_image)
                with open(seg_mesh_filename, "rb") as f:
                    seg_mesh_bytes = f.read()
                await websocket.send(b"segmesh:" + seg_mesh_bytes)
                print("Segmented mesh sent.")

            except Exception as e:
                error_msg = f"error:{str(e)}"
                print("Error during message handling:", error_msg)
                await websocket.send(error_msg.encode())
    except Exception as e:
        print("Connection handler error:", e)
    finally:
        connected_clients.remove(websocket)

async def main():
    try:
        # Increase max_size to handle larger image messages (e.g., up to 100MB)
        server = await websockets.serve(
            ai_server,
            "0.0.0.0",
            8000,
            max_size=100 * 1024 * 1024,
            max_queue=None
        )
        print("WebSocket Server Running on port 8000")
        await server.wait_closed()
    except Exception as e:
        print("Failed to start server:", e)

if __name__ == "__main__":
    asyncio.run(main())
