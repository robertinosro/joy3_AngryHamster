@echo off
setlocal enabledelayedexpansion
title JoyCaption Installer - by Angry Hamster

echo ===================================================
echo   JoyCaption Installer - Angry Hamster Edition
echo ===================================================
echo.
echo This installer will:
echo  1. Check and install Python if needed
echo  2. Install required Python packages
echo  3. Download necessary model files
echo  4. Set up the correct directory structure
echo.
echo Press any key to begin installation...
pause > nul

:: Create log file
set "logfile=%~dp0install_log.txt"
echo JoyCaption Installation Log > %logfile%
echo Installation started at %date% %time% >> %logfile%

:: Check if Python is installed
echo.
echo Checking for Python installation...
python --version > nul 2>&1
if %errorlevel% neq 0 (
    echo Python is not installed. Downloading and installing Python...
    echo Python not found. Attempting to download and install... >> %logfile%
    
    :: Download Python installer
    echo Downloading Python 3.10.11 installer...
    powershell -Command "& {Invoke-WebRequest -Uri 'https://www.python.org/ftp/python/3.10.11/python-3.10.11-amd64.exe' -OutFile 'python_installer.exe'}"
    
    if exist python_installer.exe (
        echo Installing Python 3.10.11 (this may take a few minutes)...
        :: Install Python with pip and add to PATH
        start /wait python_installer.exe /quiet InstallAllUsers=1 PrependPath=1 Include_pip=1
        
        :: Verify installation
        python --version > nul 2>&1
        if !errorlevel! neq 0 (
            echo Python installation failed. Please install Python 3.10+ manually.
            echo Python installation failed. >> %logfile%
            goto :error
        ) else (
            echo Python installed successfully.
            echo Python installed successfully. >> %logfile%
            del python_installer.exe
        )
    ) else (
        echo Failed to download Python installer.
        echo Failed to download Python installer. >> %logfile%
        goto :error
    )
) else (
    for /f "tokens=*" %%i in ('python --version') do set pyversion=%%i
    echo %pyversion% is already installed.
    echo %pyversion% found. >> %logfile%
)

:: Ensure pip is installed and updated
echo.
echo Updating pip...
python -m pip install --upgrade pip >> %logfile% 2>&1
if %errorlevel% neq 0 (
    echo Failed to update pip.
    echo Failed to update pip. >> %logfile%
    goto :error
)

:: Install required packages
echo.
echo Installing required Python packages...
echo Installing required Python packages... >> %logfile%

:: Navigate to the parent directory where requirements.txt is located
cd ..

:: Install packages from requirements.txt
python -m pip install -r requirements.txt >> %logfile% 2>&1
if %errorlevel% neq 0 (
    echo Failed to install required packages.
    echo Failed to install required packages. >> %logfile%
    goto :error
)

:: Install additional dependencies that might be needed
echo Installing additional dependencies...
python -m pip install typing_extensions >> %logfile% 2>&1
python -m pip install tqdm >> %logfile% 2>&1
python -m pip install pillow >> %logfile% 2>&1
python -m pip install numpy >> %logfile% 2>&1
python -m pip install requests >> %logfile% 2>&1

:: Create necessary directories
echo.
echo Creating necessary directories...
if not exist "cgrkzexw-599808" mkdir "cgrkzexw-599808"
if not exist "cgrkzexw-599808\text_model" mkdir "cgrkzexw-599808\text_model"

:: Download model files
echo.
echo Downloading model files from Hugging Face (this may take some time)...
echo Downloading model files... >> %logfile%

:: Download image adapter
echo Downloading image_adapter.pt...
powershell -Command "& {Invoke-WebRequest -Uri 'https://huggingface.co/spaces/fancyfeast/joy-caption-alpha-two/resolve/main/cgrkzexw-599808/image_adapter.pt' -OutFile 'cgrkzexw-599808\image_adapter.pt'}"
if %errorlevel% neq 0 (
    echo Failed to download image_adapter.pt
    echo Failed to download image_adapter.pt >> %logfile%
    goto :error
)

:: Download text model
echo Downloading adapter_model.safetensors...
powershell -Command "& {Invoke-WebRequest -Uri 'https://huggingface.co/spaces/fancyfeast/joy-caption-alpha-two/resolve/main/cgrkzexw-599808/text_model/adapter_model.safetensors' -OutFile 'cgrkzexw-599808\text_model\adapter_model.safetensors'}"
if %errorlevel% neq 0 (
    echo Failed to download adapter_model.safetensors
    echo Failed to download adapter_model.safetensors >> %logfile%
    goto :error
)

:: Download clip model
echo Downloading clip_model.pt...
powershell -Command "& {Invoke-WebRequest -Uri 'https://huggingface.co/spaces/fancyfeast/joy-caption-alpha-two/resolve/main/cgrkzexw-599808/clip_model.pt' -OutFile 'cgrkzexw-599808\clip_model.pt'}"
if %errorlevel% neq 0 (
    echo Failed to download clip_model.pt
    echo Failed to download clip_model.pt >> %logfile%
    goto :error
)

:: Check if all files were downloaded successfully
echo.
echo Verifying downloaded files...
set "all_files_exist=true"

if not exist "cgrkzexw-599808\image_adapter.pt" (
    echo image_adapter.pt is missing.
    echo image_adapter.pt is missing. >> %logfile%
    set "all_files_exist=false"
)

if not exist "cgrkzexw-599808\text_model\adapter_model.safetensors" (
    echo adapter_model.safetensors is missing.
    echo adapter_model.safetensors is missing. >> %logfile%
    set "all_files_exist=false"
)

if not exist "cgrkzexw-599808\clip_model.pt" (
    echo clip_model.pt is missing.
    echo clip_model.pt is missing. >> %logfile%
    set "all_files_exist=false"
)

if "%all_files_exist%"=="false" (
    echo Some model files are missing. Installation may be incomplete.
    echo Some model files are missing. Installation may be incomplete. >> %logfile%
    goto :error
)

:: Create a shortcut to run JoyCaption
echo.
echo Creating desktop shortcut...
powershell -ExecutionPolicy Bypass -File "CreateShortcut.ps1" >> %logfile% 2>&1

:: Installation complete
echo.
echo ===================================================
echo   Installation Complete!
echo ===================================================
echo.
echo JoyCaption has been successfully installed.
echo You can now run JoyCaption by using the desktop shortcut
echo or by running the "JoyCaption 2.1_AngryHamster.bat" file.
echo.
echo Installation completed at %date% %time% >> %logfile%
echo Installation completed successfully. >> %logfile%
goto :end

:error
echo.
echo ===================================================
echo   Installation Error
echo ===================================================
echo.
echo An error occurred during installation.
echo Please check the install_log.txt file for details.
echo.
echo Installation failed at %date% %time% >> %logfile%

:end
echo.
echo Press any key to exit...
pause > nul
exit /b
