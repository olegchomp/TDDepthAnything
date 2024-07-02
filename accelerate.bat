@echo off

set PYTHON_PATH=

if not exist .venv (
    echo Creating .venv directory...
    %PYTHON_PATH% -m venv ".venv"

    echo Activating virtual environment...
    call .venv\Scripts\activate
    REM Replace '.venv\Scripts\activate' with path to your TouchDiffusion .venv. 
    REM for ex. 'C:\User\TouchDiffusion\venv\Scripts\activate'

    echo Installing dependencies
    pip install nvidia-cudnn-cu11==8.9.4.25 --no-cache-dir
    pip install --pre --extra-index-url https://pypi.nvidia.com tensorrt==9.0.1.post11.dev4 --no-cache-dir
    pip install torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cu118
    echo Installing dependencies from requirements.txt...
    pip install -r requirements.txt
    pip uninstall -y nvidia-cudnn-cu11

    if not exist checkpoints (
        echo Creating 'checkpoints' folder...
        mkdir checkpoints
    )
    echo Installation complete. 
) else (
    echo Activating virtual environment...
    call .venv\Scripts\activate.bat
    REM Replace '.venv\Scripts\activate' with path to your TouchDiffusion .venv. 
    REM for ex. 'C:\User\TouchDiffusion\venv\Scripts\activate'
    echo Preparing for model acceleration...
    python accelerate_model.py
)
pause
