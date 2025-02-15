import time
import os
import json

from PIL import Image
import cv2
import numpy as np
import torch

# Global imports for models and utilities
import trimesh
import utils3d
from moge.model import MoGeModel
from moge.utils.io import save_glb

from depth_anything_v2.dpt import DepthAnythingV2
from matplotlib import colormaps

from google import genai
from google.genai import types
from dotenv import load_dotenv
import google.generativeai as genai_api

from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor


def load_inputs(image_path, object_name):
    """Load image in multiple formats and extract dimensions."""
    start = time.time()
    PIL_image = Image.open(image_path)
    width, height = PIL_image.size

    np_image = np.array(PIL_image)
    cv2_image = cv2.imread(image_path)

    print(f"Inputs section took {time.time() - start:.2f} seconds")
    return PIL_image, cv2_image, np_image, width, height, object_name


def optimize_inference():
    """Enable CUDA autocast and TF32 optimization for faster inference."""
    start = time.time()
    torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    print(f"Inference Optimization section took {time.time() - start:.2f} seconds")


def initialize_moge(cv2_image, device_str="cuda:0"):
    """Initialize the MoGE model and prepare the image tensor."""
    start = time.time()
    device = torch.device(device_str)
    moge_model = MoGeModel.from_pretrained("Ruicheng/moge-vitl").to(device)

    # Convert BGR image to RGB and prepare tensor
    moge_image = cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGB)
    moge_image_tensor = (
        torch.tensor(moge_image / 255, dtype=torch.float32, device=device)
        .permute(2, 0, 1)
    )
    threshold = 0.03
    print(f"MoGE Initialization section took {time.time() - start:.2f} seconds")
    return moge_model, moge_image, moge_image_tensor, threshold, device


def initialize_depth_anything(checkpoint_path="checkpoints/depth_anything_v2_metric_hypersim_vitl.pth", device_str="cuda:1"):
    """Initialize DepthAnythingV2 model and get the colormap."""
    start = time.time()
    cmap = colormaps.get_cmap('Spectral')
    depth_device = torch.device(device_str)
    depth_anything = DepthAnythingV2(
        encoder='vitl',
        features=256,
        out_channels=[256, 512, 1024, 1024],
        max_depth=20
    )
    depth_anything.load_state_dict(torch.load(checkpoint_path, map_location='cpu'))
    depth_anything = depth_anything.to(depth_device).eval()
    print(f"DepthAnythingv2 Initialization section took {time.time() - start:.2f} seconds")
    return depth_anything, depth_device, cmap


def initialize_gemini():
    """Initialize Gemini API client and configuration settings."""
    start = time.time()
    load_dotenv()  # load GEMINI_API_KEY from .env

    client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
    model_name = "gemini-2.0-flash"  # adjust as needed

    bounding_box_system_instructions = (
        "Return bounding boxes as a JSON array with labels. Never return masks or code fencing. Limit to 1 object."
    )
    safety_settings = [
        types.SafetySetting(
            category="HARM_CATEGORY_DANGEROUS_CONTENT",
            threshold="BLOCK_ONLY_HIGH",
        ),
    ]
    print(f"Gemini Initialization section took {time.time() - start:.2f} seconds")
    return client, model_name, bounding_box_system_instructions, safety_settings


def parse_json(json_output):
    """Remove markdown code fencing from the response text."""
    lines = json_output.splitlines()
    for i, line in enumerate(lines):
        if line.strip() == "```json":
            json_output = "\n".join(lines[i+1:])
            json_output = json_output.split("```")[0]
            break
    return json_output


def initialize_sam(np_image, device_str="cuda:2"):
    """Initialize SAM model and set the image for prediction."""
    start = time.time()
    sam_device = torch.device(device_str)
    sam2_checkpoint = "checkpoints/sam2.1_hiera_large.pt"
    model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
    sam2_model = build_sam2(model_cfg, sam2_checkpoint, device=sam_device)
    predictor = SAM2ImagePredictor(sam2_model)
    predictor.set_image(np_image)
    print(f"SAM Initialization section took {time.time() - start:.2f} seconds")
    return predictor


def perform_moge_inference(moge_model, moge_image_tensor):
    """Perform MoGE inference on the image tensor."""
    start = time.time()
    output = moge_model.infer(moge_image_tensor, apply_mask=True)
    print(f"MoGE Inference section took {time.time() - start:.2f} seconds")
    return output


def build_raw_mesh(output, width, height, moge_image, threshold, filename="mesh.glb"):
    """Build and save the raw mesh from MoGE output."""
    start = time.time()
    points = output['points'].cpu().numpy()
    depth = output['depth'].cpu().numpy()
    points_mask = output['mask'].cpu().numpy()
    intrinsics = output['intrinsics'].cpu().numpy()

    normals, normals_mask = utils3d.numpy.points_to_normals(points, mask=points_mask)
    faces, vertices, vertex_colors, vertex_uvs = utils3d.numpy.image_mesh(
        points,
        moge_image.astype(np.float32) / 255,
        utils3d.numpy.image_uv(width=width, height=height),
        mask=points_mask & ~(utils3d.numpy.depth_edge(depth, rtol=threshold, mask=points_mask) &
                             utils3d.numpy.normals_edge(normals, tol=5, mask=normals_mask)),
        tri=True
    )
    vertices, vertex_uvs = vertices * [1, -1, -1], vertex_uvs * [1, -1] + [0, 1]
    save_glb(filename, vertices, faces, vertex_uvs, moge_image)
    print(f"Build Raw Mesh section took {time.time() - start:.2f} seconds")


def perform_depth_inference(depth_anything, cv2_image, depth_device):
    """Perform depth inference on the image using DepthAnythingV2."""
    start = time.time()
    with torch.cuda.device(depth_device):
        metric_depth = depth_anything.infer_image(cv2_image)
    print(f"Depth Inference section took {time.time() - start:.2f} seconds")
    return metric_depth


def draw_depth(metric_depth, cmap, output_path="depth.jpg"):
    """Normalize, colorize, and save the depth image."""
    start = time.time()
    metric_depth_norm = (metric_depth - metric_depth.min()) / (metric_depth.max() - metric_depth.min()) * 255.0
    metric_depth_norm = metric_depth_norm.astype(np.uint8)
    metric_depth_colored = (cmap(metric_depth_norm)[:, :, :3] * 255)[:, :, ::-1].astype(np.uint8)
    cv2.imwrite(output_path, metric_depth_colored)
    print(f"Draw Depth section took {time.time() - start:.2f} seconds")


def call_gemini_api(client, model_name, bounding_box_system_instructions, safety_settings, object_name, PIL_image):
    """Call the Gemini API to get the bounding box for the object."""
    start = time.time()
    prompt = f"Detect the 2d bounding box of the {object_name}."
    config = types.GenerateContentConfig(
        system_instruction=bounding_box_system_instructions,
        temperature=0.5,
        safety_settings=safety_settings,
    )
    response = client.models.generate_content(
        model=model_name,
        contents=[prompt, PIL_image],
        config=config,
    )
    if not response.text.startswith("```json"):
        raise ValueError("Object not found in image")
    print(f'{object_name} Gemini bounding box: {response.text}')
    print(f"Gemini API Call section took {time.time() - start:.2f} seconds")
    return response.text


def draw_bounding_box(cv2_image, bounding_box, width, height, output_path="bounding_box.jpg"):
    """Draw the bounding box on the image and save it."""
    start = time.time()
    # Convert the normalized box values (assumed to be in 0-1000 scale) to pixel values.
    y1 = int(bounding_box["box_2d"][0] / 1000 * height)
    x1 = int(bounding_box["box_2d"][1] / 1000 * width)
    y2 = int(bounding_box["box_2d"][2] / 1000 * height)
    x2 = int(bounding_box["box_2d"][3] / 1000 * width)

    bounding_box_color = (0, 0, 255)  # BGR
    bounding_box_thickness = 2

    bounding_box_image = cv2_image.copy()
    cv2.rectangle(bounding_box_image, (x1, y1), (x2, y2), bounding_box_color, bounding_box_thickness)
    cv2.imwrite(output_path, bounding_box_image)
    print(f"Draw Bounding Boxes section took {time.time() - start:.2f} seconds")
    return np.array([x1, y1, x2, y2])


def perform_sam_inference(predictor, input_box):
    """Perform SAM segmentation given a bounding box."""
    start = time.time()
    masks, scores, _ = predictor.predict(
        point_coords=None,
        point_labels=None,
        box=input_box[None, :],
        multimask_output=False,
    )
    print(f"SAM Inference section took {time.time() - start:.2f} seconds")
    print(f"Segmentation SAM score: {scores}")
    return masks, scores


def draw_segmentation(cv2_image, mask, output_path="segmentation.jpg"):
    """Overlay segmentation mask on the original image and save."""
    start = time.time()
    red_mask = np.zeros_like(cv2_image)
    red_mask[:, :, 2] = 255  # Set red channel (BGR)
    alpha = 0.5
    overlay = cv2.addWeighted(cv2_image, 1, red_mask, alpha, 0, dtype=cv2.CV_8U)
    segmented_image = np.where(mask[:, :, None] == 1, overlay, cv2_image)
    cv2.imwrite(output_path, segmented_image)
    print(f"Draw Segmentation section took {time.time() - start:.2f} seconds")
    return segmented_image


def build_segmented_mesh(output, width, height, segmented_image, threshold, filename="segmented_mesh.glb"):
    """Build and save the mesh using the segmented image colors."""
    start = time.time()
    points = output['points'].cpu().numpy()
    depth = output['depth'].cpu().numpy()
    points_mask = output['mask'].cpu().numpy()
    normals, normals_mask = utils3d.numpy.points_to_normals(points, mask=points_mask)

    # Convert segmented image from BGR to RGB
    segmented_moge_image = cv2.cvtColor(segmented_image, cv2.COLOR_BGR2RGB)
    faces, vertices, vertex_colors, vertex_uvs = utils3d.numpy.image_mesh(
        points,
        segmented_moge_image.astype(np.float32) / 255,
        utils3d.numpy.image_uv(width=width, height=height),
        mask=points_mask & ~(utils3d.numpy.depth_edge(depth, rtol=threshold, mask=points_mask) &
                             utils3d.numpy.normals_edge(normals, tol=5, mask=normals_mask)),
        tri=True
    )
    vertices, vertex_uvs = vertices * [1, -1, -1], vertex_uvs * [1, -1] + [0, 1]
    save_glb(filename, vertices, faces, vertex_uvs, segmented_moge_image)
    print(f"Build Segmented Mesh section took {time.time() - start:.2f} seconds")


def main():
    # --- Inputs ---
    image_path = "/lfs/ampere9/0/krishpar/Edith/ml/notebooks/images/dorm.jpg"
    object_name = "red water bottle"
    PIL_image, cv2_image, np_image, width, height, object_name = load_inputs(image_path, object_name)

    # --- Inference Optimizations ---
    optimize_inference()

    # --- MoGE Initialization ---
    moge_model, moge_image, moge_image_tensor, threshold, _ = initialize_moge(cv2_image)

    # --- DepthAnythingv2 Initialization ---
    depth_anything, depth_device, cmap = initialize_depth_anything()

    # --- Gemini Initialization ---
    client, model_name, bb_system_instructions, safety_settings = initialize_gemini()

    # --- SAM Initialization ---
    predictor = initialize_sam(np_image)

    # --- MoGE Inference ---
    output = perform_moge_inference(moge_model, moge_image_tensor)

    # --- Build Raw Mesh ---
    build_raw_mesh(output, width, height, moge_image, threshold, filename="mesh.glb")

    # --- Depth Inference ---
    metric_depth = perform_depth_inference(depth_anything, cv2_image, depth_device)

    # --- Draw Depth ---
    draw_depth(metric_depth, cmap, output_path="depth.jpg")

    # --- Gemini API Call ---
    response_text = call_gemini_api(client, model_name, bb_system_instructions, safety_settings, object_name, PIL_image)
    bounding_box = json.loads(parse_json(response_text))[0]

    # --- Draw Bounding Boxes ---
    input_box = draw_bounding_box(cv2_image, bounding_box, width, height, output_path="bounding_box.jpg")

    # --- SAM Inference ---
    masks, scores = perform_sam_inference(predictor, input_box)

    # --- Draw Segmentation ---
    segmented_image = draw_segmentation(cv2_image, masks[0], output_path="segmentation.jpg")

    # --- Build Segmented Mesh ---
    build_segmented_mesh(output, width, height, segmented_image, threshold, filename="segmented_mesh.glb")


if __name__ == '__main__':
    main()
