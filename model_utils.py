"""
Model Utilities for Multi-Model Pipeline
=========================================
Provides config loading, chat template handling, and response cleaning.
"""

import os
import re
import yaml
from pathlib import Path

CONFIG_PATH = Path(__file__).parent / "config.yaml"


def load_config():
    """Load the pipeline configuration."""
    with open(CONFIG_PATH, "r") as f:
        return yaml.safe_load(f)


def get_model_config(model_key: str) -> dict:
    """Get configuration for a specific model."""
    config = load_config()
    if model_key not in config["models"]:
        raise ValueError(f"Unknown model key: {model_key}. Available: {list(config['models'].keys())}")
    return config["models"][model_key]


def get_dataset_config() -> dict:
    """Get dataset configuration."""
    config = load_config()
    return config.get("dataset", {"use_full": False})


def get_output_dir(model_key: str) -> Path:
    """Get output directory for a model."""
    output_dir = Path(__file__).parent / "results" / model_key
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def apply_chat_template(tokenizer, messages: list, model_config: dict) -> str:
    """
    Apply chat template with model-specific handling.

    - Qwen3 (thinking_mode="disable"): use enable_thinking=False
    - DeepSeek R1 (thinking_mode="strip"): let it think, strip later
    """
    thinking_mode = model_config.get("thinking_mode", "disable")

    if thinking_mode == "disable":
        # Qwen3 style - disable thinking
        try:
            return tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False
            )
        except TypeError:
            # Fallback if enable_thinking not supported
            return tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
    else:
        # DeepSeek R1 or others - standard template, will strip thinking later
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )


def clean_response(text: str, model_config: dict) -> str:
    """
    Clean model response, stripping thinking tags if needed.

    - thinking_mode="strip": remove <think>...</think> content
    - thinking_mode="disable": no cleaning needed
    """
    thinking_mode = model_config.get("thinking_mode", "disable")

    if thinking_mode == "strip":
        # Remove everything between <think> and </think> including tags
        text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)

    return text.strip()


def get_model_key() -> str:
    """Get current model key from environment."""
    return os.getenv("MODEL_KEY", "qwen3_0.6b")
