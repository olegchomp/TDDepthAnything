@echo off

if not exist .venv (
    echo Creating .venv directory...
    python -m venv ".venv"

    echo Activating virtual environment...
    call .venv\Scripts\activate
    echo Installing dependencies
    pip install nvidia-cudnn-cu12==9.1.0.70 --no-cache-dir
    pip install tensorrt==10.0.1 --no-cache-dir
    pip install torch==2.5.1 torchvision --index-url https://download.pytorch.org/whl/cu121
    echo Installing dependencies from requirements.txt...
    pip install -r requirements.txt

    if not exist checkpoints (
        echo Creating 'checkpoints' folder...
        mkdir checkpoints
    )
    echo Installation complete.
) else (
    echo Activating virtual environment...
    call .venv\Scripts\activate.bat
    echo Preparing for model acceleration...
    python accelerate_model.py
)
pause