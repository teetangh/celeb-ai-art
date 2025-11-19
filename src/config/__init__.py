"""Configuration module for model settings."""

from .model_configs import (
    MODEL_CONFIGS,
    TRAINING_CONFIGS,
    get_model_config,
    get_training_config,
)

__all__ = [
    "MODEL_CONFIGS",
    "TRAINING_CONFIGS",
    "get_model_config",
    "get_training_config",
]
