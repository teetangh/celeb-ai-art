"""Model configurations with best settings for each model type."""

from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass
class InferenceConfig:
    """Configuration for image generation inference."""
    base_model: str
    resolution: int
    default_steps: int
    default_guidance: float
    supports_negative_prompt: bool
    vram_required: str
    lora_suffix: str
    description: str


@dataclass
class TrainingConfig:
    """Configuration for LoRA training."""
    resolution: int
    lora_rank: int
    lora_alpha: int
    learning_rate: float
    max_train_steps: int
    batch_size: int
    gradient_accumulation: int
    optimizer: str
    scheduler: str
    target_modules: List[str]
    mixed_precision: str
    gradient_checkpointing: bool


# Inference configurations for each model
MODEL_CONFIGS: Dict[str, InferenceConfig] = {
    "SD 1.5": InferenceConfig(
        base_model="runwayml/stable-diffusion-v1-5",
        resolution=512,
        default_steps=30,
        default_guidance=7.5,
        supports_negative_prompt=True,
        vram_required="4GB+",
        lora_suffix="_lora",
        description="Fast, lower quality. Good for testing.",
    ),
    "SDXL": InferenceConfig(
        base_model="stabilityai/stable-diffusion-xl-base-1.0",
        resolution=1024,
        default_steps=30,
        default_guidance=7.5,
        supports_negative_prompt=True,
        vram_required="8GB+",
        lora_suffix="_sdxl_lora",
        description="Better anatomy, recommended for production.",
    ),
    "Flux Dev": InferenceConfig(
        base_model="black-forest-labs/FLUX.1-dev",
        resolution=1024,
        default_steps=20,
        default_guidance=3.5,
        supports_negative_prompt=False,
        vram_required="12GB+",
        lora_suffix="_flux_lora",
        description="Best quality, excellent anatomy. High VRAM.",
    ),
    "Flux Schnell": InferenceConfig(
        base_model="black-forest-labs/FLUX.1-schnell",
        resolution=1024,
        default_steps=4,
        default_guidance=0.0,  # Schnell doesn't use guidance
        supports_negative_prompt=False,
        vram_required="12GB+",
        lora_suffix="_flux_schnell_lora",
        description="Fastest Flux variant. Good for previews.",
    ),
}


# Training configurations for each model
TRAINING_CONFIGS: Dict[str, TrainingConfig] = {
    "SD 1.5": TrainingConfig(
        resolution=512,
        lora_rank=16,
        lora_alpha=32,
        learning_rate=5e-5,
        max_train_steps=1500,
        batch_size=1,
        gradient_accumulation=4,
        optimizer="adamw",
        scheduler="cosine",
        target_modules=["to_k", "to_q", "to_v", "to_out.0"],
        mixed_precision="fp16",
        gradient_checkpointing=True,
    ),
    "SDXL": TrainingConfig(
        resolution=1024,
        lora_rank=64,
        lora_alpha=32,
        learning_rate=1e-4,
        max_train_steps=2000,
        batch_size=1,
        gradient_accumulation=4,
        optimizer="adamw",
        scheduler="cosine",
        target_modules=[
            "to_k", "to_q", "to_v", "to_out.0",
            "proj_in", "proj_out",
            "ff.net.0.proj", "ff.net.2",
        ],
        mixed_precision="fp16",
        gradient_checkpointing=True,
    ),
    "Flux": TrainingConfig(
        resolution=1024,
        lora_rank=16,
        lora_alpha=16,
        learning_rate=1e-4,
        max_train_steps=1000,
        batch_size=1,
        gradient_accumulation=4,
        optimizer="prodigy",  # Recommended for Flux
        scheduler="constant",
        target_modules=[
            "to_k", "to_q", "to_v", "to_out.0",
            # Flux uses transformer blocks
            "single_transformer_blocks",
            "transformer_blocks",
        ],
        mixed_precision="bf16",  # Flux prefers bf16
        gradient_checkpointing=True,
    ),
}


# Default negative prompt for models that support it
DEFAULT_NEGATIVE_PROMPT = (
    "blurry, low quality, distorted, deformed, ugly, bad anatomy, "
    "bad hands, missing fingers, extra fingers, fused fingers, too many fingers, "
    "mutated hands, malformed limbs, extra limbs, missing limbs, "
    "disfigured, gross proportions, long neck, duplicate, "
    "morbid, mutilated, poorly drawn hands, poorly drawn face, "
    "mutation, ugly, bad proportions, cloned face, "
    "extra arms, extra legs, fused limbs, too many limbs, "
    "wrong anatomy, liquid fingers, missing arms, missing legs"
)


# Example prompts for UI
EXAMPLE_PROMPTS = [
    "brad pitt portrait, professional headshot, studio lighting, sharp focus",
    "brad pitt walking on the beach, sunny day, casual outfit, professional photo",
    "brad pitt at a coffee shop, natural lighting, bokeh background",
    "brad pitt in a garden, flowers, spring, bright colors, detailed",
    "brad pitt fitness, gym, athletic wear, professional photography",
]


def get_model_config(model_type: str) -> InferenceConfig:
    """Get inference configuration for a model type."""
    if model_type not in MODEL_CONFIGS:
        raise ValueError(f"Unknown model type: {model_type}. Available: {list(MODEL_CONFIGS.keys())}")
    return MODEL_CONFIGS[model_type]


def get_training_config(model_type: str) -> TrainingConfig:
    """Get training configuration for a model type."""
    # Map model types to training configs
    if model_type in ["SD 1.5"]:
        return TRAINING_CONFIGS["SD 1.5"]
    elif model_type in ["SDXL"]:
        return TRAINING_CONFIGS["SDXL"]
    elif model_type in ["Flux Dev", "Flux Schnell"]:
        return TRAINING_CONFIGS["Flux"]
    else:
        raise ValueError(f"Unknown model type: {model_type}")
