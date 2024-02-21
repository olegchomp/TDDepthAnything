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

os.makedirs("onnx_models", exist_ok=True)
os.makedirs("engines", exist_ok=True)

while True:
    model_size = input("Enter 's' for small, 'b' for base, or 'l' for large: ").lower()

    if model_size in ['s', 'b', 'l']:
        break
    else:
        print("Invalid input. Please enter 's', 'b', or 'l'.")

encoder = f'vit{model_size}'
load_from = f'./checkpoints/depth_anything_vit{model_size}14.pth'
image_shape = (3, 518, 518)

outputs = f"{load_from.split('/')[-1].split('.pth')[0]}"
onnx_path = f"onnx_models/{outputs}.onnx"
engine_path = f"engines/{outputs}.engine"

# build onnx
# Initializing model
assert encoder in ['vits', 'vitb', 'vitl']
if encoder == 'vits':
    depth_anything = DPT_DINOv2(encoder='vits', features=64, out_channels=[48, 96, 192, 384], localhub=False)
elif encoder == 'vitb':
    depth_anything = DPT_DINOv2(encoder='vitb', features=128, out_channels=[96, 192, 384, 768], localhub=False)
else:
    depth_anything = DPT_DINOv2(encoder='vitl', features=256, out_channels=[256, 512, 1024, 1024], localhub=False)

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