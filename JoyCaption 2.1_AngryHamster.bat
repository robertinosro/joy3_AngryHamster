@echo off
echo Starting JoyCaption with maximum quality settings...
echo Using full GPU capabilities for best results.

:: Check for required dependencies and install if missing
python -c "import typing_extensions" 2>NUL
if %errorlevel% neq 0 (
    echo Installing missing dependency: typing_extensions...
    pip install typing_extensions -q
)

:: Set environment variables for better GPU performance
set CUDA_DEVICE_ORDER=PCI_BUS_ID
set CUDA_LAUNCH_BLOCKING=1

:: Run the application
python dark_mode_gui.py

:: If there was an error, pause to show the message
if %errorlevel% neq 0 (
    echo.
    echo An error occurred while running JoyCaption.
    pause
)
