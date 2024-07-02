import os
import torch
import torch.onnx
import tensorrt as trt
from polygraphy.backend.trt import (
    CreateConfig,
    Profile,
    engine_from_network,
    network_from_onnx_path,
    save_engine,
)
torch.hub.set_dir('torchhub')
from depth_anything.dpt import DPT_DINOv2

def adjust_image_size(image_size):
    patch_size = 14
    # Calculate the nearest multiple of patch_size that is greater than or equal to image_size
    adjusted_size = (image_size // patch_size) * patch_size
    if image_size % patch_size != 0:
        adjusted_size += patch_size
    return adjusted_size

os.makedirs("onnx_models", exist_ok=True)
os.makedirs("engines", exist_ok=True)

while True:
    model_version = int(input("Enter 1 for DepthAnything v1 or 2 for DepthAnything v2: ").lower())

    if model_version in [1,2]:
        break
    else:
        print("Invalid input. Please enter '1' or '2'")

while True:
    if model_version == 1:
        model_size = input("Enter 's' for small, 'b' for base, or 'l' for large: ").lower()

        if model_size in ['s', 'b', 'l']:
            break
        else:
            print("Invalid input. Please enter 's', 'b', or 'l'.")
    else:
        if model_version == 2:
            model_size = input("Enter 's' for small, 'b' for base, 'l' for large, or 'g' for giant: ").lower()

            if model_size in ['s', 'b', 'l', 'g']:
                break
            else:
                print("Invalid input. Please enter 's', 'b', 'l', or 'g'.")

while True:
    try:
        width = int(input("Enter the width of the input: "))
        height = int(input("Enter the height of the input: "))
        break
    except ValueError:
        print("Invalid input. Please enter a valid integer for width and height.")


encoder = f'vit{model_size}'
if model_version == 1:
    load_from = f'./checkpoints/depth_anything_vit{model_size}14.pth'
else:
    load_from = f'./checkpoints/depth_anything_v2_vit{model_size}.pth'

width = adjust_image_size(width)
height = adjust_image_size(height)
image_shape = (3, height, width)
print(f'Image shape is {width}x{height}')

outputs = f"{load_from.split('/')[-1].split('.pth')[0]}"
onnx_path = f"onnx_models/{outputs}_{width}x{height}.onnx"
engine_path = f"engines/{outputs}_{width}x{height}.engine"

# build onnx
# Initializing model
#assert encoder in ['vits', 'vitb', 'vitl']

if encoder == 'vits':
    depth_anything = DPT_DINOv2(encoder='vits', features=64, out_channels=[48, 96, 192, 384], localhub=False)
elif encoder == 'vitb':
    depth_anything = DPT_DINOv2(encoder='vitb', features=128, out_channels=[96, 192, 384, 768], localhub=False)
elif encoder == 'vitl':
    depth_anything = DPT_DINOv2(encoder='vitl', features=256, out_channels=[256, 512, 1024, 1024], localhub=False)
else:
    depth_anything = DPT_DINOv2(encoder='vitg', features=384, out_channels=[1536, 1536, 1536, 1536], localhub=False)

if model_version == 2:
    from depth_anything_v2.dpt import DepthAnythingV2

    model_configs = {
    'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
    'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
    'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
    'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
    }

    depth_anything = DepthAnythingV2(**model_configs[encoder])


total_params = sum(param.numel() for param in depth_anything.parameters())
print('Total parameters: {:.2f}M'.format(total_params / 1e6))

# Loading model weight
depth_anything.load_state_dict(torch.load(load_from, map_location='cpu'), strict=True)

depth_anything.eval()

# Define dummy input data
dummy_input = torch.ones(image_shape).unsqueeze(0)

# Provide an example input to the model, this is necessary for exporting to ONNX
example_output = depth_anything(dummy_input)

# Export the PyTorch model to ONNX format
torch.onnx.export(depth_anything, dummy_input, onnx_path, opset_version=11, input_names=["input"], output_names=["output"], verbose=True)

print(f"Model exported to {onnx_path}")

# build engine 
print(f"Building TensorRT engine for {onnx_path}: {engine_path}")

p = Profile()
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