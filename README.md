# Joy Caption Alpha Two 3.1 - Updated & Improved by Angry Hamster

This version is not the 4bit. It's the full GPU edited version, fine-tuned for full GPU performance, enhancing captions quality and generation speed!

> I am not taking credit for building this from scratch. I only improved this as many been having issues with it (from installing to lack of options), so I edited this and made it more accessible to everyone.
> 
> Credits for the first version go to [devajyoti151](https://civitai.com/user/devajyoti151) and [fancyfeast](https://huggingface.co/fancyfeast)

## What's New?

- **Added**: Loaded Model Status Check and progress bar
  - This was an issue as often the GUI was not showing properly if model is loaded or not, causing users to start captioning with model incompletely loaded
- **Added**: Captioning generation progress bar
- **Added**: Generated Caption preview & edit for single file captions
  - When captioning a single file, now you can preview the generated caption directly inside the GUI, edit the text and copy to clipboard
- **Added**: Custom Camera Angles option
  - Now you can insert directly inside the generated Caption preview the desired camera angle and use it for your AI image gen prompt or for lora training
  - The full list with description can be found inside the ress folder
- **Added**: Save edited caption
  - It will overwrite the already generated caption inside the images directory where it was initially saved

## Installation

Full working version, no need to install anything! You only need to download the base model files:

1. Download image adapter from [here](https://huggingface.co/spaces/fancyfeast/joy-caption-alpha-two/tree/main/cgrkzexw-599808) and place it inside folder:
   ```
   cgrkzexw-599808\image_adapter.pt
   ```

2. Download text model from [here](https://huggingface.co/spaces/fancyfeast/joy-caption-alpha-two/tree/main/cgrkzexw-599808/text_model) and place inside folder:
   ```
   cgrkzexw-599808\text_model\adapter_model.safetensors
   ```

3. Download clip model from [here](https://huggingface.co/spaces/fancyfeast/joy-caption-alpha-two/tree/main/cgrkzexw-599808) and place inside folder:
   ```
   cgrkzexw-599808\clip_model.pt
   ```

## Troubleshooting

If you encounter the error `ModuleNotFoundError: No module named 'typing_extensions'`, please run this command in your command prompt or terminal:

```
pip install typing_extensions
```

This is a dependency required by PyTorch that might be missing on some systems.

## Usage

Once files are downloaded, run the `.bat` file and happy captioning!
 
