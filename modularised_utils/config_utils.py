# Centralizes model options and Vosk path.
# Makes it easy to add new models by updating get_model_options.
from typing import Dict

def get_model_options() -> Dict[str, str]:
    """Return available model options."""
    return {
        "DialoGPT Large (355M)": "microsoft/DialoGPT-large",
        "BlenderBot 3B": "facebook/blenderbot-3B",
        "BlenderBot 1B": "facebook/blenderbot-1B-distill",
        "GPT-2 Large (774M)": "gpt2-large",
        "GPT-2 XL (1.5B)": "gpt2-xl"
    }

def get_vosk_model_path() -> str:
    """Return path to Vosk model."""
    return "vosk-model-en-us-0.42-gigaspeech"