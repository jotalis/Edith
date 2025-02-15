# IDEAS
# - multiple bounding boxes multiple objects?

import time

from PIL import Image
import cv2
import numpy as np

# Inputs
image_path = "/lfs/ampere9/0/krishpar/Edith/ml/notebooks/images/dorm.jpg"
object_name = "red water bottle"

start_time = time.time()

PIL_image = Image.open(image_path)
width, height = PIL_image.size

np_image = np.array(PIL_image)
cv2_image = cv2.imread(image_path)

object_name = "red water bottle"

print(f"Inputs section took {time.time() - start_time:.2f} seconds")

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

predictor.set_image(np_image)

print(f"SAM Initialization section took {time.time() - start_time:.2f} seconds")

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
    raise ValueError("Object not found in image")
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

# Outputs
# Save the raw mesh as a GLB file
# save_glb('mesh.glb', vertices, faces, vertex_uvs, moge_image)

# # Save the depth image as a JPEG file
# cv2.imwrite("depth.jpg", metric_depth)

# # Save the image with the bounding box as a JPEG file
# cv2.imwrite("bounding_box.jpg", bounding_box_image)

# # Save the segmented image as a JPEG file
# cv2.imwrite("segmentation.jpg", segmented_image)

# # Save the segmented mesh as a GLB file
# save_glb('segmented_mesh.glb', vertices, faces, vertex_uvs, segmented_moge_image)
