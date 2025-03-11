# JoyCaption Automated Installer

This automated installer is designed to help users who are experiencing installation issues with JoyCaption due to different software versions or missing dependencies.

## What Does This Installer Do?

The installer will automatically:

1. **Check and install Python** if it's not already installed on your system
2. **Install all required Python packages** including:
   - huggingface_hub
   - accelerate
   - torch
   - transformers
   - sentencepiece
   - peft
   - torchvision
   - typing_extensions
   - and other dependencies
3. **Download all necessary model files** from Hugging Face:
   - image_adapter.pt
   - adapter_model.safetensors
   - clip_model.pt
4. **Set up the correct directory structure**
5. **Create a desktop shortcut** for easy access

## How to Use

1. Download the entire `installer` folder from the GitHub repository
2. Run `installer.bat` by double-clicking it
3. Follow the on-screen instructions
4. Wait for the installation to complete (this may take some time, especially when downloading model files)

## Troubleshooting

If you encounter any issues during installation:

1. Check the `install_log.txt` file (created in the same directory as the installer) for detailed error information
2. Make sure your internet connection is stable, especially during model downloads
3. Try running the installer as administrator if you encounter permission issues
4. If the Python installation fails, try installing Python 3.10+ manually from [python.org](https://www.python.org/downloads/)

## System Requirements

- Windows 10 or later
- At least 4GB of free disk space
- Internet connection for downloading Python and model files
- Administrator privileges may be required for Python installation

## Credits

- Original JoyCaption by [devajyoti151](https://civitai.com/user/devajyoti151) and [fancyfeast](https://huggingface.co/fancyfeast)
- JoyCaption Alpha Two 3.1 improvements by Angry Hamster
- Automated installer created to help users with installation issues
