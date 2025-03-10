import sys
import os
import torch
import logging
import numpy as np
from torch import nn
from transformers import (
    AutoModel,
    AutoProcessor,
    AutoTokenizer,
    PreTrainedTokenizer,
    PreTrainedTokenizerFast,
    AutoModelForCausalLM,
)
from PIL import Image
import torchvision.transforms.functional as TVF
import contextlib
from typing import Union, List, Callable
from pathlib import Path
import re
import sys
from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
    QLabel, QLineEdit, QTextEdit, QProgressBar, QFileDialog,
    QListWidget, QListWidgetItem, QComboBox, QCheckBox, QMessageBox,
    QSplitter
)
from PyQt5.QtGui import QPixmap, QImage, QIcon, QClipboard
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QSize, QObject

# Set environment variables for better GPU performance
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # Arrange GPUs in order
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"  # More synchronous CUDA operations for stability

# Configure PyTorch for maximum performance
torch.backends.cudnn.benchmark = True  # Use cudnn auto-tuner to find the best algorithm
torch.backends.cuda.matmul.allow_tf32 = True  # Allow TF32 on Ampere GPUs
torch.backends.cudnn.allow_tf32 = True  # Allow TF32 on Ampere GPUs

CLIP_PATH = "google/siglip-so400m-patch14-384"
CAPTION_TYPE_MAP = {
    "Descriptive": [
        "Write a descriptive caption for this image in a formal tone.",
        "Write a descriptive caption for this image in a formal tone within {word_count} words.",
        "Write a {length} descriptive caption for this image in a formal tone.",
    ],
    "Descriptive (Informal)": [
        "Write a descriptive caption for this image in a casual tone.",
        "Write a descriptive caption for this image in a casual tone within {word_count} words.",
        "Write a {length} descriptive caption for this image in a casual tone.",
    ],
    "Training Prompt": [
        "Write a stable diffusion prompt for this image.",
        "Write a stable diffusion prompt for this image within {word_count} words.",
        "Write a {length} stable diffusion prompt for this image.",
    ],
    "MidJourney": [
        "Write a MidJourney prompt for this image.",
        "Write a MidJourney prompt for this image within {word_count} words.",
        "Write a {length} MidJourney prompt for this image.",
    ],
    "Booru tag list": [
        "Write a list of Booru tags for this image.",
        "Write a list of Booru tags for this image within {word_count} words.",
        "Write a {length} list of Booru tags for this image.",
    ],
    "Booru-like tag list": [
        "Write a list of Booru-like tags for this image.",
        "Write a list of Booru-like tags for this image within {word_count} words.",
        "Write a {length} list of Booru-like tags for this image.",
    ],
    "Art Critic": [
        "Analyze this image like an art critic would with information about its composition, style, symbolism, the use of color, light, any artistic movement it might belong to, etc.",
        "Analyze this image like an art critic would with information about its composition, style, symbolism, the use of color, light, any artistic movement it might belong to, etc. Keep it within {word_count} words.",
        "Analyze this image like an art critic would with information about its composition, style, symbolism, the use of color, light, any artistic movement it might belong to, etc. Keep it {length}.",
    ],
    "Product Listing": [
        "Write a caption for this image as though it were a product listing.",
        "Write a caption for this image as though it were a product listing. Keep it under {word_count} words.",
        "Write a {length} caption for this image as though it were a product listing.",
    ],
    "Social Media Post": [
        "Write a caption for this image as if it were being used for a social media post.",
        "Write a caption for this image as if it were being used for a social media post. Limit the caption to {word_count} words.",
        "Write a {length} caption for this image as if it were being used for a social media post.",
    ],
}

EXTRA_OPTIONS_LIST = [
    "If there is a person/character in the image you must refer to them as {name}.",
    "Do NOT include information about people/characters that cannot be changed (like ethnicity, gender, etc), but do still include changeable attributes (like hair style).",
    "Include information about lighting.",
    "Include information about camera angle.",
    "Include information about whether there is a watermark or not.",
    "Include information about whether there are JPEG artifacts or not.",
    "If it is a photo you MUST include information about what camera was likely used and details such as aperture, shutter speed, ISO, etc.",
    "Do NOT include anything sexual; keep it PG.",
    "Do NOT mention the image's resolution.",
    "You MUST include information about the subjective aesthetic quality of the image from low to very high.",
    "Include information on the image's composition style, such as leading lines, rule of thirds, or symmetry.",
    "Do NOT mention any text that is in the image.",
    "Specify the depth of field and whether the background is in focus or blurred.",
    "If applicable, mention the likely use of artificial or natural lighting sources.",
    "Do NOT use any ambiguous language.",
    "Include whether the image is sfw, suggestive, or nsfw.",
    "ONLY describe the most important elements of the image.",
]

CAPTION_LENGTH_CHOICES = (
    ["any", "very short", "short", "medium-length", "long", "very long"]
    + [str(i) for i in range(20, 261, 10)]
)

HF_TOKEN = os.environ.get("HF_TOKEN", None)

class ImageAdapter(nn.Module):
    def __init__(
        self,
        input_features: int,
        output_features: int,
        ln1: bool,
        pos_emb: bool,
        num_image_tokens: int,
        deep_extract: bool,
    ):
        super().__init__()
        self.deep_extract = deep_extract

        if self.deep_extract:
            input_features = input_features * 5

        self.linear1 = nn.Linear(input_features, output_features)
        self.activation = nn.GELU()
        self.linear2 = nn.Linear(output_features, output_features)
        self.ln1 = nn.Identity() if not ln1 else nn.LayerNorm(input_features)
        self.pos_emb = (
            None if not pos_emb else nn.Parameter(torch.zeros(num_image_tokens, input_features))
        )

        # Other tokens (<|image_start|>, <|image_end|>, <|eot_id|>)
        self.other_tokens = nn.Embedding(3, output_features)
        self.other_tokens.weight.data.normal_(
            mean=0.0, std=0.02
        )  # Matches HF's implementation of llama3

    def forward(self, vision_outputs: torch.Tensor):
        if self.deep_extract:
            x = torch.concat(
                (
                    vision_outputs[-2],
                    vision_outputs[3],
                    vision_outputs[7],
                    vision_outputs[13],
                    vision_outputs[20],
                ),
                dim=-1,
            )
            assert len(x.shape) == 3, f"Expected 3, got {len(x.shape)}"  # batch, tokens, features
            assert (
                x.shape[-1] == vision_outputs[-2].shape[-1] * 5
            ), f"Expected {vision_outputs[-2].shape[-1] * 5}, got {x.shape[-1]}"
        else:
            x = vision_outputs[-2]

        x = self.ln1(x)

        if self.pos_emb is not None:
            assert x.shape[-2:] == self.pos_emb.shape, f"Expected {self.pos_emb.shape}, got {x.shape[-2:]}"
            x = x + self.pos_emb

        x = self.linear1(x)
        x = self.activation(x)
        x = self.linear2(x)

        # <|image_start|>, IMAGE, <|image_end|>
        other_tokens = self.other_tokens(
            torch.tensor([0, 1], device=self.other_tokens.weight.device).expand(x.shape[0], -1)
        )
        assert other_tokens.shape == (
            x.shape[0],
            2,
            x.shape[2],
        ), f"Expected {(x.shape[0], 2, x.shape[2])}, got {other_tokens.shape}"
        x = torch.cat((other_tokens[:, 0:1], x, other_tokens[:, 1:2]), dim=1)

        return x

    def get_eot_embedding(self):
        return self.other_tokens(torch.tensor([2], device=self.other_tokens.weight.device)).squeeze(0)

# Set device with optimal settings
if torch.cuda.is_available():
    device = torch.device("cuda")
    # Print GPU info for debugging
    print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    
    # Set optimal CUDA settings for maximum performance
    torch.cuda.empty_cache()  # Clear cache before starting
    
    # Use mixed precision where appropriate
    torch_dtype = torch.float16  # Use half precision for better performance while maintaining quality
else:
    device = torch.device("cpu")
    torch_dtype = torch.float32
    print("CUDA is not available. Using CPU.")

# Create a progress signal class for model loading
class ProgressSignal(QObject):
    progress_updated = pyqtSignal(int, str)

# Global progress signal instance
progress_signal = ProgressSignal()

def load_models(CHECKPOINT_PATH, progress_callback=None):
    total_steps = 5
    current_step = 0
    
    # Helper function to update progress
    def update_progress(message):
        nonlocal current_step
        current_step += 1
        if progress_callback:
            progress_percent = int((current_step / total_steps) * 100)
            progress_callback(progress_percent, message)
        print(message)
    
    # Load CLIP
    update_progress("Loading CLIP model...")
    clip_processor = AutoProcessor.from_pretrained(CLIP_PATH)
    clip_model = AutoModel.from_pretrained(CLIP_PATH)
    clip_model = clip_model.vision_model

    assert (
        CHECKPOINT_PATH / "clip_model.pt"
    ).exists(), f"clip_model.pt not found in {CHECKPOINT_PATH}"
    update_progress("Loading VLM's custom vision model...")
    checkpoint = torch.load(CHECKPOINT_PATH / "clip_model.pt", map_location="cpu")
    checkpoint = {k.replace("_orig_mod.module.", ""): v for k, v in checkpoint.items()}
    clip_model.load_state_dict(checkpoint)
    del checkpoint

    clip_model.eval()
    clip_model.requires_grad_(False)
    clip_model.to(device)

    # Tokenizer
    update_progress("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        CHECKPOINT_PATH / "text_model", use_fast=True
    )
    assert isinstance(tokenizer, (PreTrainedTokenizer, PreTrainedTokenizerFast)), f"Tokenizer is of type {type(tokenizer)}"

    # Add special tokens to the tokenizer
    special_tokens_dict = {'additional_special_tokens': ['<|system|>', '<|image_start|>', '<|image_end|>', '<|eot_id|>', '<|end|>']}
    num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)
    print(f"Added {num_added_toks} special tokens.")

    # LLM
    update_progress("Loading LLM...")
    print("Loading VLM's custom text model")
    
    # Explicitly set model loading parameters for maximum quality
    model_loading_kwargs = {
        "torch_dtype": torch_dtype,
        "device_map": "auto",
        "low_cpu_mem_usage": True,  # Reduces CPU memory usage during loading
    }
    
    text_model = AutoModelForCausalLM.from_pretrained(
        CHECKPOINT_PATH / "text_model", 
        **model_loading_kwargs
    )
    text_model.eval()
    
    # Resize token embeddings if new tokens were added
    if num_added_toks > 0:
        text_model.resize_token_embeddings(len(tokenizer))

    # Image Adapter
    update_progress("Loading image adapter...")
    image_adapter = ImageAdapter(
        clip_model.config.hidden_size, text_model.config.hidden_size, False, False, 38, False
    )
    image_adapter.load_state_dict(
        torch.load(CHECKPOINT_PATH / "image_adapter.pt", map_location="cpu")
    )
    image_adapter.eval()
    image_adapter.to(device)

    update_progress("All models loaded successfully!")
    return clip_processor, clip_model, tokenizer, text_model, image_adapter

@torch.no_grad()
def generate_caption(
    input_image: Image.Image,
    caption_type: str,
    caption_length: Union[str, int],
    extra_options: List[str],
    name_input: str,
    custom_prompt: str,
    clip_model,
    tokenizer,
    text_model,
    image_adapter,
    progress_callback: Callable = None,
) -> tuple:
    # Clear CUDA cache before processing
    if device.type == "cuda":
        torch.cuda.empty_cache()
    
    # Define total steps for progress tracking
    total_steps = 5
    current_step = 0
    
    # Helper function to update progress
    def update_progress(message, step_increment=1):
        nonlocal current_step
        current_step += step_increment
        if progress_callback:
            progress_percent = int((current_step / total_steps) * 100)
            progress_callback(progress_percent, message)
        print(message)

    update_progress("Processing image...")
    # Ensure the image is in the correct format for highest quality
    if input_image.mode != "RGB":
        input_image = input_image.convert("RGB")
    
    # Process the image at the model's expected resolution
    # The CLIP model expects 384x384 images
    image = input_image.resize((384, 384), Image.LANCZOS)

    # If a custom prompt is provided, use it directly
    if custom_prompt.strip() != "":
        prompt_str = custom_prompt.strip()
    else:
        # 'any' means no length specified
        length = None if caption_length == "any" else caption_length

        if isinstance(length, str):
            try:
                length = int(length)
            except ValueError:
                pass

        # Build prompt
        if length is None:
            map_idx = 0
        elif isinstance(length, int):
            map_idx = 1
        elif isinstance(length, str):
            map_idx = 2
        else:
            raise ValueError(f"Invalid caption length: {length}")

        prompt_str = CAPTION_TYPE_MAP[caption_type][map_idx]

        # Add extra options
        if len(extra_options) > 0:
            prompt_str += " " + " ".join(extra_options)

        # Add name, length, word_count
        prompt_str = prompt_str.format(name=name_input, length=caption_length, word_count=caption_length)

    # For debugging
    print(f"Prompt: {prompt_str}")

    update_progress("Embedding image...")
    # Use high-quality image preprocessing
    # Normalize the image
    pixel_values = TVF.pil_to_tensor(image).unsqueeze(0) / 255.0
    pixel_values = TVF.normalize(pixel_values, [0.5], [0.5])
    pixel_values = pixel_values.to(device)

    # Embed image
    # This results in Batch x Image Tokens x Features
    with torch.autocast(device_type='cuda', dtype=torch_dtype):
        vision_outputs = clip_model(pixel_values=pixel_values, output_hidden_states=True)
        embedded_images = image_adapter(vision_outputs.hidden_states)
        embedded_images = embedded_images.to(device)

    update_progress("Preparing prompt...")
    # Build the conversation
    convo = [
        {
            "role": "system",
            "content": "You are a helpful image captioner.",
        },
        {
            "role": "user",
            "content": prompt_str,
        },
    ]

    # Format the conversation
    # The apply_chat_template method might not be available; handle accordingly
    if hasattr(tokenizer, "apply_chat_template"):
        convo_string = tokenizer.apply_chat_template(
            convo, tokenize=False, add_generation_prompt=True
        )
    else:
        # Simple concatenation if apply_chat_template is not available
        convo_string = (
            "<|system|>\n" + convo[0]["content"] + "\n<|end|>\n\n" + convo[1]["content"] + "\n<|end|>\n"
        )

    assert isinstance(convo_string, str)

    # Tokenize the conversation
    # prompt_str is tokenized separately so we can do the calculations below
    convo_tokens = tokenizer.encode(
        convo_string, return_tensors="pt", add_special_tokens=False, truncation=False
    ).to(device)
    prompt_tokens = tokenizer.encode(
        prompt_str, return_tensors="pt", add_special_tokens=False, truncation=False
    ).to(device)
    assert isinstance(convo_tokens, torch.Tensor) and isinstance(prompt_tokens, torch.Tensor)
    convo_tokens = convo_tokens.squeeze(0)  # Squeeze just to make the following easier
    prompt_tokens = prompt_tokens.squeeze(0)

    # Calculate where to inject the image
    # Use the indices of the special tokens
    end_token_id = tokenizer.convert_tokens_to_ids("<|end|>")

    # Ensure end_token_id is valid
    if end_token_id is None:
        raise ValueError("The tokenizer does not recognize the '<|end|>' token. Please ensure special tokens are added.")

    end_token_indices = (convo_tokens == end_token_id).nonzero(as_tuple=True)[0].tolist()
    if len(end_token_indices) >= 2:
        # The image is to be injected between the system message and the user prompt
        preamble_len = end_token_indices[0] + 1  # Position after the first <|end|>
    else:
        preamble_len = 0  # Fallback to the start if tokens are missing

    # Embed the tokens
    convo_embeds = text_model.model.embed_tokens(convo_tokens.unsqueeze(0).to(device))

    # Construct the input
    input_embeds = torch.cat(
        [
            convo_embeds[:, :preamble_len],  # Part before the prompt
            embedded_images.to(dtype=convo_embeds.dtype),  # Image embeddings
            convo_embeds[:, preamble_len:],  # The prompt and anything after it
        ],
        dim=1,
    ).to(device)

    input_ids = torch.cat(
        [
            convo_tokens[:preamble_len].unsqueeze(0),
            torch.full((1, embedded_images.shape[1]), tokenizer.pad_token_id, dtype=torch.long, device=device),  # Dummy tokens for the image
            convo_tokens[preamble_len:].unsqueeze(0),
        ],
        dim=1,
    ).to(device)
    attention_mask = torch.ones_like(input_ids).to(device)

    # Debugging
    print(f"Input to model: {repr(tokenizer.decode(input_ids[0]))}")

    update_progress("Generating caption...", 0)
    # Generate the caption
    generate_ids = text_model.generate(
        input_ids=input_ids,
        inputs_embeds=input_embeds,
        attention_mask=attention_mask,
        max_new_tokens=300,
        do_sample=True,
        temperature=0.65,  # Slightly lower temperature for more coherent outputs
        top_p=0.92,        # Slightly higher top_p for more diverse vocabulary
        top_k=60,          # Higher top_k for better quality
        repetition_penalty=1.15,  # Adjusted for better flow
        no_repeat_ngram_size=3,   # Prevent repeating the same 3-grams
        use_cache=True,    # Enable KV caching for faster generation
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

    update_progress("Processing caption...")
    # Trim off the prompt
    generate_ids = generate_ids[:, input_ids.shape[1]:]
    if generate_ids[0][-1] in [tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<|end|>")]:
        generate_ids = generate_ids[:, :-1]

    # Debug the generated tokens
    print(f"Generated token IDs: {generate_ids[0][:10]}...")
    
    # Use a simpler decoding approach to avoid issues
    caption = tokenizer.decode(generate_ids[0], skip_special_tokens=True)
    print(f"Raw caption after decoding: '{caption[:100]}...'")
    
    # Clean up any remaining special tokens or artifacts that might have been missed
    caption = caption.replace("<|eot_id|>", "").replace("<|finetune_right_pad_id|>", "")
    
    # More targeted cleanup of special tokens and artifacts
    special_token_patterns = [
        r"<\|.*?\|>",  # Any token in the format <|token|>
        r"_pad_id",
        r"finetune_right_pad_id",
        r"finetune_left_pad_id",
        r"start_header_id",
        r"end_header_id"
    ]
    
    # Store the original caption in case our cleaning is too aggressive
    original_caption = caption
    
    for pattern in special_token_patterns:
        caption = re.sub(pattern, "", caption)
    
    caption = caption.strip()
    
    # If our cleaning resulted in an empty string, revert to the original
    if not caption:
        caption = original_caption.strip()
        # Just do basic cleaning instead
        caption = caption.replace("<|", "").replace("|>", "")
    
    # Remove any repeated phrases that might occur in low-quality outputs
    words = caption.split()
    if len(words) > 20:  # Only apply to longer captions
        # Check for repetitive patterns
        for i in range(1, min(5, len(words) // 2)):  # Check patterns up to 5 words long
            for j in range(len(words) - 2*i):
                if words[j:j+i] == words[j+i:j+2*i]:
                    # Found a repetition, keep only one instance
                    words = words[:j+i] + words[j+2*i:]
                    break
        caption = " ".join(words)

    update_progress("Caption complete!", 1)
    return prompt_str, caption

class CaptionApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("JoyCaption 2.1 by AngryHamster")
        self.setGeometry(100, 100, 1200, 800)
        
        # Set application icon
        app_icon = QIcon("ico.ico")
        self.setWindowIcon(app_icon)
        
        # Load camera angles from file
        self.camera_angles = self.load_camera_angles()
        
        self.initUI()

        # Initialize model variables
        self.clip_processor = None
        self.clip_model = None
        self.tokenizer = None
        self.text_model = None
        self.image_adapter = None

        # Initialize variables for selected images
        self.input_dir = None
        self.single_image_path = None
        self.selected_image_path = None

        # Theme variables
        self.dark_mode = True  # Set dark mode to True by default
        
        # Apply dark theme stylesheet by default
        self.setStyleSheet("""
            QWidget {
                background-color: #2E2E2E;
                color: #FFFFFF;
            }
            QPushButton {
                background-color: #3A3A3A;
                color: #FFFFFF;
                border: none;
                padding: 5px;
            }
            QPushButton:hover {
                background-color: #555555;
            }
            QLabel {
                color: #FFFFFF;
            }
            QLineEdit, QTextEdit {
                background-color: #3A3A3A;
                color: #FFFFFF;
                border: 1px solid #555555;
            }
            QComboBox {
                background-color: #3A3A3A;
                color: #FFFFFF;
                border: 1px solid #555555;
            }
            QListWidget {
                background-color: #3A3A3A;
                color: #FFFFFF;
                border: 1px solid #555555;
            }
            QCheckBox {
                color: #FFFFFF;
            }
        """)

    def load_camera_angles(self):
        """Load camera angles from the text file in the ress folder"""
        camera_angles = []
        try:
            camera_angles_file = Path("ress/Common Camera Angles.txt")
            if camera_angles_file.exists():
                with open(camera_angles_file, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
                    for line in lines:
                        line = line.strip()
                        # Extract camera angle descriptions (lines that start with a number followed by a period)
                        if line and line[0].isdigit() and ". " in line:
                            angle = line.split(". ", 1)[1].split(":-")[0].strip()
                            camera_angles.append(angle)
            else:
                print(f"Camera angles file not found: {camera_angles_file}")
        except Exception as e:
            print(f"Error loading camera angles: {e}")
        
        return camera_angles

    def initUI(self):
        main_layout = QHBoxLayout()

        # Left panel for parameters
        left_panel = QVBoxLayout()

        # Model Loading Section with Progress Bar
        model_section = QVBoxLayout()
        model_section.addWidget(QLabel("Model Status:"))
        
        # Model loading progress bar
        self.model_progress_bar = QProgressBar()
        self.model_progress_bar.setRange(0, 100)
        self.model_progress_bar.setValue(0)
        model_section.addWidget(self.model_progress_bar)
        
        # Model status label
        self.model_status_label = QLabel("Model not loaded")
        model_section.addWidget(self.model_status_label)
        
        # Checkpoint Path
        self.checkpoint_path_line = QLineEdit()
        self.checkpoint_path_line.setText("cgrkzexw-599808")
        model_section.addWidget(QLabel("Checkpoint Path:"))
        model_section.addWidget(self.checkpoint_path_line)
        
        # Load Models Button
        self.load_models_button = QPushButton("Load Models")
        self.load_models_button.clicked.connect(self.load_models)
        model_section.addWidget(self.load_models_button)
        
        left_panel.addLayout(model_section)
        
        # Add a separator
        separator = QLabel("")
        separator.setStyleSheet("border-bottom: 1px solid #555555; margin: 10px 0px;")
        left_panel.addWidget(separator)

        # Input directory selection
        self.input_dir_button = QPushButton("Select Input Directory")
        self.input_dir_button.clicked.connect(self.select_input_directory)
        self.input_dir_label = QLabel("No directory selected")
        left_panel.addWidget(self.input_dir_button)
        left_panel.addWidget(self.input_dir_label)

        # Single image selection
        self.single_image_button = QPushButton("Select Single Image")
        self.single_image_button.clicked.connect(self.select_single_image)
        self.single_image_label = QLabel("No image selected")
        left_panel.addWidget(self.single_image_button)
        left_panel.addWidget(self.single_image_label)

        # Caption Type
        self.caption_type_combo = QComboBox()
        self.caption_type_combo.addItems(CAPTION_TYPE_MAP.keys())
        self.caption_type_combo.setCurrentText("Descriptive")
        left_panel.addWidget(QLabel("Caption Type:"))
        left_panel.addWidget(self.caption_type_combo)

        # Caption Length
        self.caption_length_combo = QComboBox()
        self.caption_length_combo.addItems(CAPTION_LENGTH_CHOICES)
        self.caption_length_combo.setCurrentText("long")
        left_panel.addWidget(QLabel("Caption Length:"))
        left_panel.addWidget(self.caption_length_combo)

        # Extra Options
        left_panel.addWidget(QLabel("Extra Options:"))
        self.extra_options_checkboxes = []
        for option in EXTRA_OPTIONS_LIST:
            checkbox = QCheckBox(option)
            self.extra_options_checkboxes.append(checkbox)
            left_panel.addWidget(checkbox)

        # Name Input
        self.name_input_line = QLineEdit()
        left_panel.addWidget(QLabel("Person/Character Name (if applicable):"))
        left_panel.addWidget(self.name_input_line)

        # Generated Caption Display and Edit Field
        self.generated_caption_text = QTextEdit()
        self.generated_caption_text.setPlaceholderText("Generated caption will appear here. You can edit it if needed.")
        left_panel.addWidget(QLabel("Generated Caption:"))
        left_panel.addWidget(self.generated_caption_text)

        # Camera Angles Dropdown
        camera_angles_layout = QHBoxLayout()
        camera_angles_layout.addWidget(QLabel("Add Camera Angle:"))
        self.camera_angle_combo = QComboBox()
        self.camera_angle_combo.addItems(self.camera_angles)
        camera_angles_layout.addWidget(self.camera_angle_combo)
        
        # Add Camera Angle Button
        self.add_camera_angle_button = QPushButton("Insert")
        self.add_camera_angle_button.clicked.connect(self.insert_camera_angle)
        camera_angles_layout.addWidget(self.add_camera_angle_button)
        
        left_panel.addLayout(camera_angles_layout)

        # Custom Prompt
        self.custom_prompt_text = QTextEdit()
        left_panel.addWidget(QLabel("Custom Prompt (optional):"))
        left_panel.addWidget(self.custom_prompt_text)

        # Run Buttons
        self.run_button = QPushButton("Generate Captions for All Images")
        self.run_button.clicked.connect(self.generate_captions)
        left_panel.addWidget(self.run_button)

        self.caption_selected_button = QPushButton("Caption Selected Image")
        self.caption_selected_button.clicked.connect(self.caption_selected_image)
        self.caption_selected_button.setEnabled(False)  # Disabled until an image is selected
        left_panel.addWidget(self.caption_selected_button)

        self.caption_single_button = QPushButton("Caption Single Image")
        self.caption_single_button.clicked.connect(self.caption_single_image)
        self.caption_single_button.setEnabled(False)  # Disabled until a single image is selected
        left_panel.addWidget(self.caption_single_button)

        # Save Edited Caption Button
        self.save_edited_caption_button = QPushButton("Save Edited Caption")
        self.save_edited_caption_button.clicked.connect(self.save_edited_caption)
        left_panel.addWidget(self.save_edited_caption_button)
        
        # Copy Prompt to Clipboard Button
        self.copy_prompt_button = QPushButton("Copy Caption")
        self.copy_prompt_button.clicked.connect(self.copy_prompt_to_clipboard)
        left_panel.addWidget(self.copy_prompt_button)

        # Add spacer to bottom of left panel
        left_panel.addStretch()

        # Right panel for image display and captions
        right_panel = QVBoxLayout()

        # List widget for images
        self.image_list_widget = QListWidget()
        self.image_list_widget.itemClicked.connect(self.display_selected_image)
        right_panel.addWidget(QLabel("Images:"))
        right_panel.addWidget(self.image_list_widget)

        # Label to display the selected image
        self.selected_image_label = QLabel()
        self.selected_image_label.setAlignment(Qt.AlignCenter)
        right_panel.addWidget(QLabel("Selected Image:"))
        right_panel.addWidget(self.selected_image_label)

        # Caption Generation Progress Section
        caption_progress_section = QVBoxLayout()
        caption_progress_section.addWidget(QLabel("Caption Generation Progress:"))
        
        # Caption generation progress bar
        self.caption_progress_bar = QProgressBar()
        self.caption_progress_bar.setRange(0, 100)
        self.caption_progress_bar.setValue(0)
        caption_progress_section.addWidget(self.caption_progress_bar)
        
        # Caption status label
        self.caption_status_label = QLabel("Ready")
        caption_progress_section.addWidget(self.caption_status_label)
        
        right_panel.addLayout(caption_progress_section)

        main_layout.addLayout(left_panel, 2)
        main_layout.addLayout(right_panel, 3)
        self.setLayout(main_layout)

    def select_input_directory(self):
        directory = QFileDialog.getExistingDirectory(self, "Select Input Directory")
        if directory:
            self.input_dir = Path(directory)
            self.input_dir_label.setText(str(self.input_dir))
            self.load_images()
        else:
            self.input_dir_label.setText("No directory selected")
            self.input_dir = None

    def select_single_image(self):
        file_filter = "Image Files (*.jpg *.jpeg *.png *.bmp *.gif *.tiff)"
        file_path, _ = QFileDialog.getOpenFileName(self, "Select Single Image", "", file_filter)
        if file_path:
            self.single_image_path = Path(file_path)
            self.single_image_label.setText(str(self.single_image_path.name))
            self.display_image(self.single_image_path)
            self.caption_single_button.setEnabled(True)
        else:
            self.single_image_label.setText("No image selected")
            self.single_image_path = None
            self.caption_single_button.setEnabled(False)

    def load_images(self):
        # List of image file extensions
        image_extensions = [".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tiff"]

        # Collect all image files in the directory
        self.image_files = [f for f in self.input_dir.iterdir() if f.suffix.lower() in image_extensions]

        if not self.image_files:
            QMessageBox.warning(self, "No Images", "No image files found in the selected directory.")
            return

        self.image_list_widget.clear()
        for image_path in self.image_files:
            item = QListWidgetItem(str(image_path.name))
            pixmap = QPixmap(str(image_path))
            if not pixmap.isNull():
                # Increase thumbnail size to 150x150
                scaled_pixmap = pixmap.scaled(150, 150, Qt.KeepAspectRatio, Qt.SmoothTransformation)
                icon = QIcon(scaled_pixmap)
                item.setIcon(icon)
            self.image_list_widget.addItem(item)

    def display_selected_image(self, item):
        # Find the image path corresponding to the clicked item
        image_name = item.text()
        if self.input_dir:
            image_path = self.input_dir / image_name
            if image_path.exists():
                self.selected_image_path = image_path
                self.display_image(image_path)
                self.caption_selected_button.setEnabled(True)
        else:
            self.selected_image_path = None
            self.caption_selected_button.setEnabled(False)

    def display_image(self, image_path):
        pixmap = QPixmap(str(image_path))
        if not pixmap.isNull():
            # Scale the image to fit the label
            scaled_pixmap = pixmap.scaled(400, 400, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.selected_image_label.setPixmap(scaled_pixmap)
        else:
            self.selected_image_label.clear()

    def load_models(self):
        checkpoint_path = Path(self.checkpoint_path_line.text())
        if not checkpoint_path.exists():
            QMessageBox.warning(self, "Checkpoint Error", f"Checkpoint path does not exist: {checkpoint_path}")
            return

        def progress_callback(progress_percent, message):
            self.model_progress_bar.setValue(progress_percent)
            self.model_status_label.setText(message)
            
            # Change progress bar color based on progress
            if progress_percent < 30:
                self.model_progress_bar.setStyleSheet("QProgressBar::chunk { background-color: #FF5733; }")
            elif progress_percent < 70:
                self.model_progress_bar.setStyleSheet("QProgressBar::chunk { background-color: #FFC300; }")
            elif progress_percent < 100:
                self.model_progress_bar.setStyleSheet("QProgressBar::chunk { background-color: #DAF7A6; }")
            else:
                # When fully loaded, set to green
                self.model_progress_bar.setStyleSheet("QProgressBar::chunk { background-color: #2ECC71; }")
                self.model_status_label.setText("Model Loaded")
                
            print(message)

        try:
            (
                self.clip_processor,
                self.clip_model,
                self.tokenizer,
                self.text_model,
                self.image_adapter,
            ) = load_models(checkpoint_path, progress_callback)
            
            # Set progress bar to 100% and green when done
            self.model_progress_bar.setValue(100)
            self.model_progress_bar.setStyleSheet("QProgressBar::chunk { background-color: #2ECC71; }")
            self.model_status_label.setText("Model Loaded")
            
            QMessageBox.information(self, "Models Loaded", "Models have been loaded successfully.")
        except Exception as e:
            # Set progress bar to red on error
            self.model_progress_bar.setStyleSheet("QProgressBar::chunk { background-color: #FF0000; }")
            self.model_status_label.setText(f"Error: {str(e)[:50]}...")
            QMessageBox.critical(self, "Model Loading Error", f"An error occurred while loading models: {e}")

    def collect_parameters(self):
        # Collect parameters for caption generation
        caption_type = self.caption_type_combo.currentText()
        caption_length = self.caption_length_combo.currentText()
        extra_options = [checkbox.text() for checkbox in self.extra_options_checkboxes if checkbox.isChecked()]
        name_input = self.name_input_line.text()
        custom_prompt = self.custom_prompt_text.toPlainText()

        return caption_type, caption_length, extra_options, name_input, custom_prompt

    def generate_captions(self):
        if not hasattr(self, 'image_files') or not self.image_files:
            QMessageBox.warning(self, "No Images", "Please select a directory containing images.")
            return

        if not all([self.clip_processor, self.clip_model, self.tokenizer, self.text_model, self.image_adapter]):
            QMessageBox.warning(self, "Models Not Loaded", "Please load the models before generating captions.")
            return

        caption_type, caption_length, extra_options, name_input, custom_prompt = self.collect_parameters()

        # Process each image
        for i, image_path in enumerate(self.image_files):
            print(f"\nProcessing image: {image_path}")
            input_image = Image.open(image_path)

            try:
                def progress_callback(progress_percent, message):
                    self.caption_progress_bar.setValue(progress_percent)
                    self.caption_status_label.setText(message)
                    
                    # Change progress bar color based on progress
                    if progress_percent < 30:
                        self.caption_progress_bar.setStyleSheet("QProgressBar::chunk { background-color: #FF5733; }")
                    elif progress_percent < 70:
                        self.caption_progress_bar.setStyleSheet("QProgressBar::chunk { background-color: #FFC300; }")
                    elif progress_percent < 100:
                        self.caption_progress_bar.setStyleSheet("QProgressBar::chunk { background-color: #DAF7A6; }")
                    else:
                        # When fully complete, set to green
                        self.caption_progress_bar.setStyleSheet("QProgressBar::chunk { background-color: #2ECC71; }")
                    
                    print(message)

                prompt_str, caption = generate_caption(
                    input_image,
                    caption_type,
                    caption_length,
                    extra_options,
                    name_input,
                    custom_prompt,
                    self.clip_model,
                    self.tokenizer,
                    self.text_model,
                    self.image_adapter,
                    progress_callback,
                )

                caption_file = image_path.with_suffix('.txt')
                
                # Debug output to help diagnose issues
                print(f"Caption before saving: '{caption[:100]}...'")
                
                with open(caption_file, 'w', encoding='utf-8') as f:
                    # Ensure the caption is clean but not empty
                    clean_caption = caption.strip()
                    if not clean_caption:
                        print("WARNING: Caption was empty after cleaning, using original output")
                        clean_caption = "Caption generation produced empty result. Please try again."
                    f.write(f"{clean_caption}\n")

                print(f"Caption saved to {caption_file}")

                # Update progress bar
                self.caption_progress_bar.setValue((i + 1) / len(self.image_files) * 100)
                
                # Change progress bar color based on overall progress
                if (i + 1) / len(self.image_files) * 100 < 30:
                    self.caption_progress_bar.setStyleSheet("QProgressBar::chunk { background-color: #FF5733; }")
                elif (i + 1) / len(self.image_files) * 100 < 70:
                    self.caption_progress_bar.setStyleSheet("QProgressBar::chunk { background-color: #FFC300; }")
                elif (i + 1) / len(self.image_files) * 100 < 100:
                    self.caption_progress_bar.setStyleSheet("QProgressBar::chunk { background-color: #DAF7A6; }")
                else:
                    # When fully complete, set to green
                    self.caption_progress_bar.setStyleSheet("QProgressBar::chunk { background-color: #2ECC71; }")
                    self.caption_status_label.setText("Captions Complete")

            except Exception as e:
                # Set progress bar to red on error
                self.caption_progress_bar.setStyleSheet("QProgressBar::chunk { background-color: #FF0000; }")
                self.caption_status_label.setText(f"Error: {str(e)[:50]}...")
                print(f"Error processing image {image_path}: {e}")
                continue

        QMessageBox.information(self, "Captions Generated", "Captions have been generated and saved.")

    def caption_selected_image(self):
        if not self.selected_image_path:
            QMessageBox.warning(self, "No Image Selected", "Please select an image from the list.")
            return

        if not all([self.clip_processor, self.clip_model, self.tokenizer, self.text_model, self.image_adapter]):
            QMessageBox.warning(self, "Models Not Loaded", "Please load the models before generating captions.")
            return

        caption_type, caption_length, extra_options, name_input, custom_prompt = self.collect_parameters()

        print(f"\nProcessing image: {self.selected_image_path}")
        input_image = Image.open(self.selected_image_path)

        try:
            def progress_callback(progress_percent, message):
                self.caption_progress_bar.setValue(progress_percent)
                self.caption_status_label.setText(message)
                
                # Change progress bar color based on progress
                if progress_percent < 30:
                    self.caption_progress_bar.setStyleSheet("QProgressBar::chunk { background-color: #FF5733; }")
                elif progress_percent < 70:
                    self.caption_progress_bar.setStyleSheet("QProgressBar::chunk { background-color: #FFC300; }")
                elif progress_percent < 100:
                    self.caption_progress_bar.setStyleSheet("QProgressBar::chunk { background-color: #DAF7A6; }")
                else:
                    # When fully complete, set to green
                    self.caption_progress_bar.setStyleSheet("QProgressBar::chunk { background-color: #2ECC71; }")
                    self.caption_status_label.setText("Caption Complete")
                
                print(message)

            prompt_str, caption = generate_caption(
                input_image,
                caption_type,
                caption_length,
                extra_options,
                name_input,
                custom_prompt,
                self.clip_model,
                self.tokenizer,
                self.text_model,
                self.image_adapter,
                progress_callback,
            )

            caption_file = self.selected_image_path.with_suffix('.txt')
                
            # Debug output to help diagnose issues
            print(f"Caption before saving: '{caption[:100]}...'")
            
            # Display the caption in the GUI text field
            clean_caption = caption.strip()
            if not clean_caption:
                print("WARNING: Caption was empty after cleaning, using original output")
                clean_caption = "Caption generation produced empty result. Please try again."
            
            # Update the generated caption text field
            self.generated_caption_text.setText(clean_caption)
                
            with open(caption_file, 'w', encoding='utf-8') as f:
                # Use the potentially edited caption from the text field
                f.write(f"{clean_caption}\n")

            print(f"Caption saved to {caption_file}")

        except Exception as e:
            # Set progress bar to red on error
            self.caption_progress_bar.setStyleSheet("QProgressBar::chunk { background-color: #FF0000; }")
            self.caption_status_label.setText(f"Error: {str(e)[:50]}...")
            print(f"Error processing image {self.selected_image_path}: {e}")
            QMessageBox.critical(self, "Error", f"An error occurred: {e}")
            return

        QMessageBox.information(self, "Caption Generated", f"Caption has been generated and saved for {self.selected_image_path.name}.")

    def caption_single_image(self):
        if not self.single_image_path:
            QMessageBox.warning(self, "No Image Selected", "Please select a single image.")
            return

        if not all([self.clip_processor, self.clip_model, self.tokenizer, self.text_model, self.image_adapter]):
            QMessageBox.warning(self, "Models Not Loaded", "Please load the models before generating captions.")
            return

        caption_type, caption_length, extra_options, name_input, custom_prompt = self.collect_parameters()

        print(f"\nProcessing image: {self.single_image_path}")
        input_image = Image.open(self.single_image_path)

        try:
            def progress_callback(progress_percent, message):
                self.caption_progress_bar.setValue(progress_percent)
                self.caption_status_label.setText(message)
                
                # Change progress bar color based on progress
                if progress_percent < 30:
                    self.caption_progress_bar.setStyleSheet("QProgressBar::chunk { background-color: #FF5733; }")
                elif progress_percent < 70:
                    self.caption_progress_bar.setStyleSheet("QProgressBar::chunk { background-color: #FFC300; }")
                elif progress_percent < 100:
                    self.caption_progress_bar.setStyleSheet("QProgressBar::chunk { background-color: #DAF7A6; }")
                else:
                    # When fully complete, set to green
                    self.caption_progress_bar.setStyleSheet("QProgressBar::chunk { background-color: #2ECC71; }")
                    self.caption_status_label.setText("Caption Complete")
                
                print(message)

            prompt_str, caption = generate_caption(
                input_image,
                caption_type,
                caption_length,
                extra_options,
                name_input,
                custom_prompt,
                self.clip_model,
                self.tokenizer,
                self.text_model,
                self.image_adapter,
                progress_callback,
            )

            caption_file = self.single_image_path.with_suffix('.txt')
                
            # Debug output to help diagnose issues
            print(f"Caption before saving: '{caption[:100]}...'")
            
            # Display the caption in the GUI text field
            clean_caption = caption.strip()
            if not clean_caption:
                print("WARNING: Caption was empty after cleaning, using original output")
                clean_caption = "Caption generation produced empty result. Please try again."
            
            # Update the generated caption text field
            self.generated_caption_text.setText(clean_caption)
                
            with open(caption_file, 'w', encoding='utf-8') as f:
                # Use the potentially edited caption from the text field
                f.write(f"{clean_caption}\n")

            print(f"Caption saved to {caption_file}")

        except Exception as e:
            # Set progress bar to red on error
            self.caption_progress_bar.setStyleSheet("QProgressBar::chunk { background-color: #FF0000; }")
            self.caption_status_label.setText(f"Error: {str(e)[:50]}...")
            print(f"Error processing image {self.single_image_path}: {e}")
            QMessageBox.critical(self, "Error", f"An error occurred: {e}")
            return

        QMessageBox.information(self, "Caption Generated", f"Caption has been generated and saved for {self.single_image_path.name}.")

    def save_edited_caption(self):
        if not self.selected_image_path and not self.single_image_path:
            QMessageBox.warning(self, "No Image Selected", "Please select an image first.")
            return

        # Determine which image path to use
        image_path = self.selected_image_path if self.selected_image_path else self.single_image_path
        
        caption_file = image_path.with_suffix('.txt')
        edited_caption = self.generated_caption_text.toPlainText()
        with open(caption_file, 'w', encoding='utf-8') as f:
            f.write(f"{edited_caption}\n")
        print(f"Edited caption saved to {caption_file}")
        QMessageBox.information(self, "Caption Saved", f"Edited caption has been saved to {caption_file.name}")

    def insert_camera_angle(self):
        """Insert the selected camera angle at the current cursor position in the caption text"""
        if not self.camera_angles:
            QMessageBox.warning(self, "No Camera Angles", "No camera angles available to insert.")
            return
            
        selected_angle = self.camera_angle_combo.currentText()
        if not selected_angle:
            return
            
        # Get the current text and cursor position
        cursor = self.generated_caption_text.textCursor()
        
        # Insert the camera angle at the cursor position with parentheses and comma
        cursor.insertText(f" ({selected_angle}), ")
        
        # Update the cursor in the text edit
        self.generated_caption_text.setTextCursor(cursor)
        
        # Focus back on the text edit
        self.generated_caption_text.setFocus()

    def copy_prompt_to_clipboard(self):
        """Copy the generated caption to the clipboard"""
        caption = self.generated_caption_text.toPlainText()
        
        if not caption:
            QMessageBox.warning(self, "No Caption", "There is no caption to copy. Please generate a caption first.")
            return
            
        clipboard = QApplication.clipboard()
        clipboard.setText(caption)
        QMessageBox.information(self, "Caption Copied", "The caption has been copied to the clipboard.")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    
    # Set application icon for taskbar
    app_icon = QIcon("ico.ico")
    app.setWindowIcon(app_icon)
    
    window = CaptionApp()
    window.show()
    sys.exit(app.exec_())
