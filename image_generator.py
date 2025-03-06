from transformers import MarianMTModel, MarianTokenizer, CLIPTokenizer
from diffusers import StableDiffusionPipeline
import torch
import os
import logging
from config_loader import device

# Configure logger (No emojis, clean format)
logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s: %(message)s")
logger = logging.getLogger(__name__)

# Function to load translation model dynamically
def get_translation_models(model_name="Helsinki-NLP/opus-mt-tr-en"):
    """Load the translation model and tokenizer dynamically."""
    try:
        logger.info(f"Loading translation model: {model_name}")
        translation_model = MarianMTModel.from_pretrained(model_name).to(device)
        translation_tokenizer = MarianTokenizer.from_pretrained(model_name)
        logger.info(f"Translation model loaded successfully: {model_name}")
        return translation_model, translation_tokenizer
    except Exception as e:
        logger.error(f"Error loading translation model '{model_name}': {e}")
        return None, None

# Function to load image generation pipeline dynamically
def get_image_pipeline(model_name="runwayml/stable-diffusion-v1-5"):
    """Returns the image generation pipeline dynamically."""
    try:
        logger.info(f"Loading image generation model: {model_name}")
        pipe = StableDiffusionPipeline.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if device.type == "cuda" else torch.float32
        ).to(device)
        logger.info(f"Image generation model loaded: {model_name}")
        return pipe
    except Exception as e:
        logger.error(f"Error loading image generation model '{model_name}': {e}")
        return None

# Load CLIP tokenizer dynamically
def get_clip_tokenizer():
    """Loads the CLIP tokenizer."""
    try:
        logger.info("Loading CLIP Tokenizer...")
        return CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
    except Exception as e:
        logger.error(f"Error loading CLIP Tokenizer: {e}")
        return None

# Function to translate text using a model
def translate_text(text, translation_model, translation_tokenizer):
    """Translates a given text from Turkish to English."""
    if not translation_model or not translation_tokenizer:
        logger.error("Translation model or tokenizer is not available.")
        return text

    try:
        inputs = translation_tokenizer(
            text,
            return_tensors='pt',
            max_length=512,
            truncation=True,
            padding='max_length'
        ).to(device)

        outputs = translation_model.generate(
            input_ids=inputs['input_ids'],
            attention_mask=inputs['attention_mask'],
            max_length=512,  
            num_beams=5,
            early_stopping=True
        )

        english_summary = translation_tokenizer.decode(
            outputs[0],
            skip_special_tokens=True
        )
        return english_summary

    except Exception as e:
        logger.error(f"Error during translation: {e}")
        return text  # Return original text if translation fails
