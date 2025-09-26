import os
import torch
import torch.nn as nn
import torch.onnx
import tensorrt as trt
from polygraphy.backend.trt import (
    CreateConfig,
    Profile,
    engine_from_network,
    network_from_onnx_path,
    save_engine,
)
from transformers import DPTForDepthEstimation

# Wrapper class to handle model output for ONNX export
class DPTDepthModelWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, pixel_values):
        outputs = self.model(pixel_values)
        return outputs.predicted_depth

def adjust_image_size(image_size):
    patch_size = 14
    adjusted_size = (image_size // patch_size) * patch_size
    if image_size % patch_size != 0:
        adjusted_size += patch_size
    return adjusted_size

os.makedirs("onnx_models", exist_ok=True)
os.makedirs("engines", exist_ok=True)

# Model selection menu
print("Select Model to accelerate:")
print("Depth Anything v1:")
print("  1. Small (vits) - LiheYoung/depth-anything-vits-hf")
print("  2. Base (vitb) - LiheYoung/depth-anything-vitb-hf")
print("  3. Large (vitl) - LiheYoung/depth-anything-vitl-hf")
print("Depth Anything v2:")
print("  4. Small (vits) - depth-anything/Depth-Anything-V2-Small-hf")
print("  5. Base (vitb) - depth-anything/Depth-Anything-V2-Base-hf")
print("  6. Large (vitl) - depth-anything/Depth-Anything-V2-Large-hf")
print("  7. Giant (vitg) - depth-anything/Depth-Anything-V2-Giant-hf")

model_map = {
    1: ("LiheYoung/depth-anything-vits-hf", "depth_anything_v1_vits"),
    2: ("LiheYoung/depth-anything-vitb-hf", "depth_anything_v1_vitb"),
    3: ("LiheYoung/depth-anything-vitl-hf", "depth_anything_v1_vitl"),
    4: ("depth-anything/Depth-Anything-V2-Small-hf", "depth_anything_v2_small"),
    5: ("depth-anything/Depth-Anything-V2-Base-hf", "depth_anything_v2_base"),
    6: ("depth-anything/Depth-Anything-V2-Large-hf", "depth_anything_v2_large"),
    7: ("depth-anything/Depth-Anything-V2-Giant-hf", "depth_anything_v2_giant"),
}

while True:
    try:
        choice = int(input("Enter choice (1-7): "))
        if choice in model_map:
            model_id, model_short_name = model_map[choice]
            break
        else:
            print("Invalid choice. Please enter a number between 1 and 7.")
    except ValueError:
        print("Invalid input. Please enter a number.")

while True:
    try:
        width_str = input("Enter the width of the input (default 518): ")
        width = int(width_str) if width_str else 518
        height_str = input("Enter the height of the input (default 518): ")
        height = int(height_str) if height_str else 518
        break
    except ValueError:
        print("Invalid input. Please enter a valid integer for width and height.")

# Adjust size to be divisible by patch size (14)
width = adjust_image_size(width)
height = adjust_image_size(height)
image_shape = (3, height, width)
print(f'Using adjusted image shape: {width}x{height}')

# Define output paths
onnx_path = f"onnx_models/{model_short_name}_{width}x{height}.onnx"
engine_path = f"engines/{model_short_name}_{width}x{height}.engine"

# Load model from Hugging Face
print(f"Loading model: {model_id}")
base_model = DPTForDepthEstimation.from_pretrained(model_id)
model_to_export = DPTDepthModelWrapper(base_model)
model_to_export.eval()

# Create dummy input
dummy_input = torch.ones(1, *image_shape)

# Export to ONNX
print(f"Exporting model to {onnx_path}...")
torch.onnx.export(
    model_to_export,
    dummy_input,
    onnx_path,
    opset_version=14,
    input_names=["input"],
    output_names=["output"],
    verbose=False
)
print(f"Model exported successfully to {onnx_path}")

# Build TensorRT engine
print(f"Building TensorRT engine for {onnx_path}: {engine_path}")
p = Profile()
p.add("input", min=(1, *image_shape), opt=(1, *image_shape), max=(1, *image_shape))
config_kwargs = {}

engine = engine_from_network(
    network_from_onnx_path(onnx_path, flags=[trt.OnnxParserFlag.NATIVE_INSTANCENORM]),
    config=CreateConfig(
        fp16=True, refittable=False, profiles=[p], load_timing_cache=None, **config_kwargs
    ),
    save_timing_cache=None,
)
save_engine(engine, path=engine_path)

print(f"Finished building TensorRT engine: {engine_path}")