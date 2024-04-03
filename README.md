# Depth-Anything TensorRT in TouchDesigner
<a href="https://discord.com/invite/wNW8xkEjrf"><img src="https://discord.com/api/guilds/838923088997122100/widget.png?style=shield" alt="Discord Shield"/></a>

TouchDesigner implementation for Depth Anything and TensorRT monocular depth estimation. 

![Screenshot_68](https://github.com/olegchomp/TDDepthAnything/assets/11017531/fa457aa2-d10a-4f54-a93a-27d672501f16)

## Features
* One click install and run script
* In-TouchDesigner inference
  
## Usage
Tested with TouchDesigner 2023.11340 & Python 3.11

#### Install:
1. Install [Python 3.11](https://www.python.org/downloads/release/python-3118/)
2. Install [CUDA Toolkit](https://developer.nvidia.com/cuda-downloads)
3. Download TDDepthAnything repository
4. Run ```accelerate.bat```

#### Accelerate models:
1. Download [Depth-Anything model](https://huggingface.co/spaces/LiheYoung/Depth-Anything/tree/main/checkpoints)
2. Copy model to ```checkpoints``` folder
3. Run ```accelerate.bat```
4. Select model size (s - small, b - base, l - large)
5. Select width & height (default is 518x518)
6. Wait for acceleration to finish
#### TouchDesigner inference:
1. Add TDDepthAnything.tox to project
2. On ```Settings``` page change path to ```TDDepthAnything``` folder and click Re-init
3. On ```Depth Anything``` page select path to engine file (for ex. ```engines/depth_anything_vits14.engine```) and click Load Engine

## Acknowledgement
Based on the following projects:
* [Depth-Anything](https://github.com/LiheYoung/Depth-Anything) - Unleashing the Power of Large-Scale Unlabeled Data.
* [Depth-Anything TensorRT C++](https://github.com/spacewalk01/depth-anything-tensorrt) - Leveraging the TensorRT API for efficient real-time inference.
