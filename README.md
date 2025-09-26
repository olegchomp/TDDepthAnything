# Depth-Anything TensorRT in TouchDesigner
<a href="https://discord.com/invite/wNW8xkEjrf"><img src="https://discord.com/api/guilds/838923088997122100/widget.png?style=shield" alt="Discord Shield"/></a>

TouchDesigner implementation for Depth Anything and Depth Anything v2 with TensorRT monocular depth estimation. 

![Screenshot_68](https://github.com/olegchomp/TDDepthAnything/assets/11017531/fa457aa2-d10a-4f54-a93a-27d672501f16)

## Features
* One click install and run script
* In-TouchDesigner inference
  
## Usage
Tested with TouchDesigner 2023.11340, Python 3.11, PyTorch 2.5.1, and CUDA 12.1.

#### Install:
1. Install a compatible Python version (3.11-3.13) and ensure it is added to your system's PATH.
2. Install [CUDA Toolkit 12.1](https://developer.nvidia.com/cuda-12-1-0-download-archive).
3. Download this repository.
4. Run `accelerate.bat`. This will create a virtual environment and install all required dependencies.

> [!TIP]
> You can use the same `.venv` for TDDepthAnything & [TouchDiffusion](https://github.com/olegchomp/TouchDiffusion). Copy all files from the TDDepthAnything folder to your TouchDiffusion folder and run `accelerate.bat`. In the TouchDesigner extension, on the settings tab, select the TouchDiffusion folder as well.

#### Accelerate models:
1. Run `accelerate.bat` again after the initial setup.
2. A menu will appear. Select the model you want to accelerate (v1, v2, and different sizes).
3. Enter the desired input width and height (default is 518x518). The script will automatically adjust the dimensions to be compatible with the model.
4. Wait for the acceleration process to complete. The script will first download the model from Hugging Face, convert it to ONNX, and then build the TensorRT engine.

#### TouchDesigner inference:
1. Add `TDDepthAnything.tox` to your project.
2. On the `Settings` page, change the path to the `TDDepthAnything` folder and click Re-init.
3. On the `Depth Anything` page, select the path to the generated engine file (e.g., `engines/depth_anything_v2_small_518x518.engine`) and click Load Engine.

## Acknowledgement
Based on the following projects:
* [Depth-Anything](https://github.com/LiheYoung/Depth-Anything) - Unleashing the Power of Large-Scale Unlabeled Data.
* [Depth-Anything TensorRT C++](https://github.com/spacewalk01/depth-anything-tensorrt) - Leveraging the TensorRT API for efficient real-time inference.
* [TopArray](https://github.com/IntentDev/TopArray) - Interaction between Python/PyTorch tensor operations and TouchDesigner TOPs.
