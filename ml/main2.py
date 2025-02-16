import asyncio
import websockets
import aiohttp
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

# -------------------------------
# SAM Dependencies
# -------------------------------
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

# A set to keep track of connected clients (if needed)
connected_clients = set()

# Global variables for models and related configurations.
moge_model = None
moge_model_device = None
depth_anything = None
depth_device = None
cmap = None
gemini_client = None
model_name = None
bounding_box_system_instructions = None
safety_settings = None
object_name = None
sam2_model = None
predictor = None
sam_device = None

def initialize_models():
    """Initialize all AI models once when the server starts."""
    global moge_model, moge_model_device
    global depth_anything, depth_device, cmap
    global gemini_client, model_name, bounding_box_system_instructions, safety_settings, object_name
    global sam2_model, predictor, sam_device

    # -------------------------------
    # Inference Optimization Settings
    # -------------------------------
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

def parse_json(json_output: str) -> str:
    """
    Remove markdown fences from a JSON output if present.
    """
    if json_output.startswith("```json"):
        json_output = json_output.replace("```json", "", 1)
    if json_output.endswith("```"):
        json_output = json_output[:-3]
    return json_output.strip()

async def send_post_request(depth_bytes, bbox_bytes, seg_bytes, mesh_bytes, seg_mesh_bytes):
    """
    Send the processed files to the web app via an HTTP POST request.
    Adjust the URL to match your Next.js endpoint.
    """
    url = "http://10.32.72.225:3000/api/upload"  # Update with your actual endpoint URL.
    
    form = aiohttp.FormData()
    # Add image files
    form.add_field("images", depth_bytes, filename="depth.jpg", content_type="image/jpeg")
    form.add_field("images", bbox_bytes, filename="bbox.jpg", content_type="image/jpeg")
    form.add_field("images", seg_bytes, filename="seg.jpg", content_type="image/jpeg")
    # Add GLB files
    form.add_field("glbFiles", mesh_bytes, filename="mesh.glb", content_type="model/gltf-binary")
    form.add_field("glbFiles", seg_mesh_bytes, filename="segmesh.glb", content_type="model/gltf-binary")
    
    async with aiohttp.ClientSession() as session:
        async with session.post(url, data=form) as response:
            try:
                response_json = await response.json()
                print("Response from web app:", response_json)
            except Exception as e:
                print("Error parsing JSON response:", e)

async def ai_server(websocket, path=None):
    print("New Client:", websocket.remote_address)
    connected_clients.add(websocket)
    try:
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
                depth_min = metric_depth.min()
                depth_max = metric_depth.max()
                norm_depth = (metric_depth - depth_min) / (depth_max - depth_min + 1e-6) * 255.0
                norm_depth = norm_depth.astype(np.uint8)
                colored_depth = (cmap(norm_depth/255.0)[:, :, :3] * 255).astype(np.uint8)
                colored_depth = cv2.cvtColor(colored_depth, cv2.COLOR_RGB2BGR)
                encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), 30]
                success, depth_encoded = cv2.imencode('.jpg', colored_depth, encode_params)
                if not success:
                    await websocket.send("error:Failed to encode depth image".encode())
                    continue

                # -------------------------------
                # 4. MoGE Inference & Raw Mesh Generation
                # -------------------------------
                moge_image = np_image.copy()  # RGB image expected by MoGE
                moge_image_tensor = torch.tensor(
                    moge_image / 255.0, dtype=torch.float32, device=moge_model_device
                ).permute(2, 0, 1)
                output = moge_model.infer(moge_image_tensor, apply_mask=True)
                points = output['points'].cpu().numpy()
                depth_moge = output['depth'].cpu().numpy()
                points_mask = output['mask'].cpu().numpy()
                intrinsics = output['intrinsics'].cpu().numpy()
                normals, normals_mask = utils3d.numpy.points_to_normals(points, mask=points_mask)
                threshold = 0.03
                faces, vertices, vertex_colors, vertex_uvs = utils3d.numpy.image_mesh(
                    points,
                    moge_image.astype(np.float32) / 255.0,
                    utils3d.numpy.image_uv(width=width, height=height),
                    mask=points_mask & ~(utils3d.numpy.depth_edge(depth_moge, rtol=threshold, mask=points_mask) &
                                          utils3d.numpy.normals_edge(normals, tol=5, mask=normals_mask)),
                    tri=True
                )
                vertices = vertices * [1, -1, -1]
                vertex_uvs = vertex_uvs * [1, -1] + [0, 1]
                raw_mesh_filename = 'mesh.glb'
                save_glb(raw_mesh_filename, vertices, faces, vertex_uvs, moge_image)
                with open(raw_mesh_filename, "rb") as f:
                    mesh_bytes = f.read()

                # -------------------------------
                # 5. Gemini API Call to get bounding box
                # -------------------------------
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
                # The bounding box values are scaled by 1000.
                y1 = int(bounding_box["box_2d"][0] / 1000 * height)
                x1 = int(bounding_box["box_2d"][1] / 1000 * width)
                y2 = int(bounding_box["box_2d"][2] / 1000 * height)
                x2 = int(bounding_box["box_2d"][3] / 1000 * width)
                bbox_image = cv2_image.copy()
                cv2.rectangle(bbox_image, (x1, y1), (x2, y2), (0, 0, 255), 2)
                success, bbox_encoded = cv2.imencode('.jpg', bbox_image, encode_params)
                if not success:
                    await websocket.send("error:Failed to encode bounding box image".encode())
                    continue

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
                mask = masks[0].astype(np.uint8)
                red_mask = np.zeros_like(cv2_image)
                red_mask[:, :, 2] = 255  # Red channel
                alpha = 0.5
                overlay = cv2.addWeighted(cv2_image, 1, red_mask, alpha, 0)
                segmented_image = np.where(mask[:, :, None] == 1, overlay, cv2_image)
                success, seg_encoded = cv2.imencode('.jpg', segmented_image, encode_params)
                if not success:
                    await websocket.send("error:Failed to encode segmentation image".encode())
                    continue

                # -------------------------------
                # 7. Build Segmented Mesh from SAM Segmentation
                # -------------------------------
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

                # -------------------------------
                # Prepare bytes for the POST request
                # -------------------------------
                depth_bytes_data = depth_encoded.tobytes()
                bbox_bytes_data = bbox_encoded.tobytes()
                seg_bytes_data = seg_encoded.tobytes()

                # -------------------------------
                # Send the POST request with all files
                # -------------------------------
                await send_post_request(
                    depth_bytes_data,
                    bbox_bytes_data,
                    seg_bytes_data,
                    mesh_bytes,
                    seg_mesh_bytes
                )

                # Optionally notify the client that the POST was sent.
                await websocket.send("POST request sent successfully".encode())

            except Exception as e:
                error_msg = f"error:{str(e)}"
                print("Error during processing:", error_msg)
                await websocket.send(error_msg.encode())
    except Exception as e:
        print("Connection handler error:", e)
    finally:
        connected_clients.remove(websocket)

async def main():
    initialize_models()
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