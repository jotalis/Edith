import time

from PIL import Image
import cv2
import numpy as np

# Inference Optimization
start_time = time.time()

import torch

torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

print(f"Inference Optimization section took {time.time() - start_time:.2f} seconds")

# MoGE Initialization
start_time = time.time()

import trimesh
import utils3d
from moge.model import MoGeModel
from moge.utils.io import save_glb

moge_model_device = torch.device("cuda:0")
moge_model = MoGeModel.from_pretrained("Ruicheng/moge-vitl").to(moge_model_device)

moge_image = cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGB)
moge_image_tensor = torch.tensor(moge_image / 255, dtype=torch.float32, device=moge_model_device).permute(2, 0, 1)

threshold = 0.03

print(f"MoGE Initialization section took {time.time() - start_time:.2f} seconds")

# DepthAnythingv2 Initialization
start_time = time.time()

from depth_anything_v2.dpt import DepthAnythingV2

from matplotlib import colormaps
cmap = colormaps.get_cmap('Spectral')

depth_device = torch.device("cuda:1")

depth_anything = DepthAnythingV2(
    encoder='vitl',
    features=256,
    out_channels=[256, 512, 1024, 1024],
    max_depth=20
)
depth_anything.load_state_dict(torch.load('checkpoints/depth_anything_v2_metric_hypersim_vitl.pth', map_location='cpu'))
depth_anything = depth_anything.to(depth_device).eval()

print(f"DepthAnythingv2 Initialization section took {time.time() - start_time:.2f} seconds")

# Gemini Initialization
start_time = time.time()

import os
from google import genai
from google.genai import types

from dotenv import load_dotenv
load_dotenv()


client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

model_name = "gemini-2.0-flash" # @param ["gemini-1.5-flash-latest","gemini-2.0-flash-lite-preview-02-05","gemini-2.0-flash","gemini-2.0-pro-preview-02-05"] {"allow-input":true}

bounding_box_system_instructions = """
Return bounding boxes as a JSON array with labels. Never return masks or code fencing. Limit to 1 object.
"""
safety_settings = [
    types.SafetySetting(
        category="HARM_CATEGORY_DANGEROUS_CONTENT",
        threshold="BLOCK_ONLY_HIGH",
    ),
]

import google.generativeai as genai

import io
import os
import json
import requests
from io import BytesIO

def parse_json(json_output):
    # Parsing out the markdown fencing
    lines = json_output.splitlines()
    for i, line in enumerate(lines):
        if line == "```json":
            json_output = "\n".join(lines[i+1:])  # Remove everything before "```json"
            json_output = json_output.split("```")[0]  # Remove everything after the closing "```"
            break  # Exit the loop once "```json" is found
    return json_output

print(f"Gemini Initialization section took {time.time() - start_time:.2f} seconds")

# SAM Initialization
start_time = time.time()

from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

sam_device = torch.device("cuda:2")
sam2_checkpoint = "checkpoints/sam2.1_hiera_large.pt"
model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
sam2_model = build_sam2(model_cfg, sam2_checkpoint, device=sam_device)
predictor = SAM2ImagePredictor(sam2_model)

print(f"SAM Initialization section took {time.time() - start_time:.2f} seconds")

from flask import Flask, request, jsonify
import time
from PIL import Image
import io

app = Flask(__name__)

def compute_average_heading_angle_and_depth(depth_map, mask, hfov, image_width):
    """
    Compute the average heading angle of an object relative to the camera and the average depth.
    
    Parameters:
        depth_map (np.ndarray): 2D array representing the depth values (meters).
        mask (np.ndarray or list of tuples): 
            - A boolean mask (same size as depth_map), where True indicates object pixels.
            - OR a list of (x, y) pixel coordinates of the object.
        hfov (float): Horizontal field of view of the camera (default: 60 degrees).
        image_width (int): Width of the depth map in pixels (default: 1000).
        
    Returns:
        tuple: (average_heading, average_depth)
            - average_heading (float): Average heading angle in degrees (negative = left, positive = right).
            - average_depth (float): Average depth of the object in meters.
    """
    # Compute focal length in pixels using HFoV
    fx = (image_width / 2) / np.tan(np.radians(hfov / 2))

    # Principal point (image center)
    cx = image_width / 2

    # Extract object pixels from the mask
    if isinstance(mask, np.ndarray) and mask.shape == depth_map.shape:
        # If the mask is a boolean array
        y_indices, x_indices = np.where(mask)  # Get pixel coordinates from the mask
    else:
        return jsonify({'error': 'Mask should be a boolean array or a list of (x, y) tuples.'}), 400

    # Convert to numpy arrays for vectorized operations
    x_indices = np.array(x_indices)
    y_indices = np.array(y_indices)

    # Get corresponding depth values
    Z_values = depth_map[y_indices, x_indices]  # Depth values in meters

    # Filter out invalid depth values (e.g., zero or negative depths)
    valid_mask = Z_values > 0
    x_indices = x_indices[valid_mask]
    Z_values = Z_values[valid_mask]

    if len(Z_values) == 0:
        return jsonify({'error': 'No valid depth values in the object mask.'}), 400

    # Compute real-world X coordinates (left-right positions)
    X_values = (x_indices - cx) * Z_values / fx

    # Compute heading angles for all object pixels
    theta_values = np.degrees(np.arctan(X_values / Z_values))

    # Compute the average heading angle of the object
    average_heading = np.mean(theta_values)

    # Compute the average depth of the object
    average_depth = np.mean(Z_values)

    return average_heading, average_depth

@app.route('/process', methods=['POST'])
def process():

    image_file = request.files.get('image', None)
    if not image_file:
        return jsonify({'error': 'No image file provided'}), 400
    
    # Read the image if you need to process it
    PIL_image = Image.open(image_file)
    
    # Retrieve string label
    object_name = request.form.get('name', None)
    if object_name is None:
        return jsonify({'error': 'No label provided'}), 400
    
    # Retrieve angle (float)
    angle = request.form.get('angle', None)
    if angle is None:
        return jsonify({'error': 'No angle provided'}), 400
    
    try:
        angle = float(angle)
    except ValueError:
        return jsonify({'error': 'Angle must be a float'}), 40

    # Inputs

    start_time = time.time()

    width, height = PIL_image.size

    np_image = np.array(PIL_image)
    cv2_image = cv2.cvtColor(np_image, cv2.COLOR_RGB2BGR)

    print(f"Inputs section took {time.time() - start_time:.2f} seconds")

    # MoGE Inference
    start_time = time.time()

    output = moge_model.infer(moge_image_tensor, apply_mask=True)

    print(f"MoGE Inference section took {time.time() - start_time:.2f} seconds")

    # Build Raw Mesh
    start_time = time.time()

    points, depth, points_mask, intrinsics = output['points'].cpu().numpy(), output['depth'].cpu().numpy(), output['mask'].cpu().numpy(), output['intrinsics'].cpu().numpy()
    normals, normals_mask = utils3d.numpy.points_to_normals(points, mask=points_mask)

    faces, vertices, vertex_colors, vertex_uvs = utils3d.numpy.image_mesh(
        points,
        moge_image.astype(np.float32) / 255,
        utils3d.numpy.image_uv(width=width, height=height),
        mask=points_mask & ~(utils3d.numpy.depth_edge(depth, rtol=threshold, mask=points_mask) & utils3d.numpy.normals_edge(normals, tol=5, mask=normals_mask)),
        tri=True
    )
    vertices, vertex_uvs = vertices * [1, -1, -1], vertex_uvs * [1, -1] + [0, 1]

    save_glb('mesh.glb', vertices, faces, vertex_uvs, moge_image)

    print(f"Build Raw Mesh section took {time.time() - start_time:.2f} seconds")

    # Depth Inference
    start_time = time.time()

    with torch.cuda.device(depth_device):
        metric_depth = depth_anything.infer_image(cv2_image)

    print(f"Depth Inference section took {time.time() - start_time:.2f} seconds")

    # Draw Depth 
    start_time = time.time()

    # TODO: pick nicer color maps

    metric_depth = (metric_depth - metric_depth.min()) / (metric_depth.max() - metric_depth.min()) * 255.0
    metric_depth = metric_depth.astype(np.uint8)
    metric_depth = (cmap(metric_depth)[:, :, :3] * 255)[:, :, ::-1].astype(np.uint8)
    cv2.imwrite("depth.jpg", metric_depth)

    print(f"Draw Depth section took {time.time() - start_time:.2f} seconds")

    # Gemini API Call
    start_time = time.time()

    prompt = f"Detect the 2d bounding box of the {object_name}."
    response = client.models.generate_content(
        model=model_name,
        contents=[prompt, PIL_image],
        config = types.GenerateContentConfig(
            system_instruction=bounding_box_system_instructions,
            temperature=0.5,
            safety_settings=safety_settings,
        )
    )
    if (response.text[:7] != "```json"):
        return jsonify({'error': 'Object not found in image'}), 400
    print(f'{object_name} Gemini bounding box: {response.text}')

    print(f"Gemini API Call section took {time.time() - start_time:.2f} seconds")

    # Draw Bounding Boxes
    start_time = time.time()

    bounding_box = json.loads(parse_json(response.text))[0]
    y1 = int(bounding_box["box_2d"][0]/1000 * height)
    x1 = int(bounding_box["box_2d"][1]/1000 * width)
    y2 = int(bounding_box["box_2d"][2]/1000 * height)
    x2 = int(bounding_box["box_2d"][3]/1000 * width)

    bounding_box_color = (0, 0, 255) # BGR
    bounding_box_thickness = 2

    bounding_box_image = cv2_image.copy()
    cv2.rectangle(bounding_box_image, (x1, y1), (x2, y2), bounding_box_color, bounding_box_thickness)
    cv2.imwrite("bounding_box.jpg", bounding_box_image)

    print(f"Draw Bounding Boxes section took {time.time() - start_time:.2f} seconds")

    # SAM Inference
    start_time = time.time()

    predictor.set_image(np_image)

    input_box = np.array([x1, y1, x2, y2])
    masks, scores, _ = predictor.predict(
        point_coords=None,
        point_labels=None,
        box=input_box[None, :],
        multimask_output=False,
    )
    print(f'{object_name} segmentation SAM score: {scores}')

    print(f"SAM Inference section took {time.time() - start_time:.2f} seconds")

    # Draw Segmentation
    start_time = time.time()

    # TODO: make other parts of image duller for mesh (object of interest very red, everything else quite white)

    mask = masks[0].astype(np.uint8)

    red_mask = np.zeros_like(cv2_image)
    red_mask[:, :, 2] = 255 # BGR

    alpha = 0.5
    overlay = cv2.addWeighted(cv2_image, 1, red_mask, alpha, 0, dtype=cv2.CV_8U)

    segmented_image = np.where(mask[:, :, None] == 1, overlay, cv2_image)
    cv2.imwrite("segmentation.jpg", segmented_image)

    print(f"Draw Segmentation section took {time.time() - start_time:.2f} seconds")

    # Build Segmented Mesh
    start_time = time.time()

    segmented_moge_image = cv2.cvtColor(segmented_image, cv2.COLOR_BGR2RGB)

    faces, vertices, vertex_colors, vertex_uvs = utils3d.numpy.image_mesh(
        points,
        segmented_moge_image.astype(np.float32) / 255,
        utils3d.numpy.image_uv(width=width, height=height),
        mask=points_mask & ~(utils3d.numpy.depth_edge(depth, rtol=threshold, mask=points_mask) & utils3d.numpy.normals_edge(normals, tol=5, mask=normals_mask)),
        tri=True
    )
    vertices, vertex_uvs = vertices * [1, -1, -1], vertex_uvs * [1, -1] + [0, 1]

    save_glb('segmented_mesh.glb', vertices, faces, vertex_uvs, segmented_moge_image)

    print(f"Build Segmented Mesh section took {time.time() - start_time:.2f} seconds")

    average_heading, average_depth = compute_average_heading_angle_and_depth(depth, mask, 75, width)

    print(f"Average heading angle: {average_heading}")

    return jsonify({'average_heading': average_heading, 'average_depth': average_depth})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8404)