"""Flux LoRA training script for celebrity fine-tuning with best-in-class anatomy."""

import argparse
import os
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F
from accelerate import Accelerator
from accelerate.utils import set_seed
from diffusers import (
    FluxPipeline,
    FluxTransformer2DModel,
    FlowMatchEulerDiscreteScheduler,
)
from diffusers.optimization import get_scheduler
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTokenizer, T5EncoderModel, T5TokenizerFast
from peft import LoraConfig, get_peft_model
import json

# Import config
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.config.model_configs import TRAINING_CONFIGS, get_training_config


class FluxCelebrityDataset(Dataset):
    """Dataset for loading celebrity images with captions for Flux training."""

    def __init__(
        self,
        data_root: str,
        celebrity_id: str,
        tokenizer_clip: CLIPTokenizer,
        tokenizer_t5: T5TokenizerFast,
        size: int = 1024,
        trigger_word: str = "ohwx",
    ):
        self.data_root = Path(data_root)
        self.celebrity_id = celebrity_id
        self.tokenizer_clip = tokenizer_clip
        self.tokenizer_t5 = tokenizer_t5
        self.size = size
        self.trigger_word = trigger_word

        # Load images from processed folder
        celeb_path = self.data_root / "celebrities" / celebrity_id
        self.image_paths = list((celeb_path / "processed" / "face_crops").glob("*.jpg"))

        # Load metadata for captions
        metadata_path = celeb_path / "metadata.json"
        self.captions = {}
        if metadata_path.exists():
            with open(metadata_path) as f:
                metadata = json.load(f)
                for img in metadata.get("images", []):
                    self.captions[img["filename"]] = img.get("caption", f"{trigger_word} person")

        # Image transforms
        self.transforms = transforms.Compose([
            transforms.Resize(size, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(size),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert("RGB")
        image = self.transforms(image)

        # Get caption - Flux works better with natural language
        filename = image_path.name
        base_caption = self.captions.get(filename, f"{self.trigger_word} person")

        # Enhance caption for Flux (prefers descriptive natural language)
        caption = f"A high quality photograph of {base_caption}, professional photography, sharp focus, detailed"

        # Tokenize for CLIP
        tokens_clip = self.tokenizer_clip(
            caption,
            max_length=77,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        # Tokenize for T5
        tokens_t5 = self.tokenizer_t5(
            caption,
            max_length=512,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        return {
            "pixel_values": image,
            "input_ids_clip": tokens_clip.input_ids[0],
            "input_ids_t5": tokens_t5.input_ids[0],
            "attention_mask_t5": tokens_t5.attention_mask[0],
        }


class FluxLoRATrainer:
    """Trainer for Flux LoRA fine-tuning with superior anatomy."""

    def __init__(
        self,
        pretrained_model: str = "black-forest-labs/FLUX.1-dev",
        output_dir: str = "./outputs",
        learning_rate: float = 1e-4,
        train_batch_size: int = 1,
        gradient_accumulation_steps: int = 4,
        max_train_steps: int = 1000,
        lora_rank: int = 16,
        lora_alpha: int = 16,
        seed: int = 42,
        gradient_checkpointing: bool = True,
        mixed_precision: str = "bf16",
        use_8bit: bool = True,  # For low VRAM
    ):
        self.pretrained_model = pretrained_model
        self.output_dir = Path(output_dir)
        self.learning_rate = learning_rate
        self.train_batch_size = train_batch_size
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.max_train_steps = max_train_steps
        self.lora_rank = lora_rank
        self.lora_alpha = lora_alpha
        self.seed = seed
        self.gradient_checkpointing = gradient_checkpointing
        self.mixed_precision = mixed_precision
        self.use_8bit = use_8bit

        self.output_dir.mkdir(parents=True, exist_ok=True)

    def train(
        self,
        data_root: str,
        celebrity_id: str,
        trigger_word: str = "ohwx",
        resume_from: Optional[str] = None,
    ):
        """Train Flux LoRA on celebrity dataset."""

        # Initialize accelerator
        accelerator = Accelerator(
            gradient_accumulation_steps=self.gradient_accumulation_steps,
            mixed_precision=self.mixed_precision,
        )

        set_seed(self.seed)

        # Load Flux models
        print("Loading Flux models (this may take a while)...")
        print("Note: Flux requires significant VRAM. Using optimizations for 8GB cards.")

        # Load tokenizers
        tokenizer_clip = CLIPTokenizer.from_pretrained(
            self.pretrained_model, subfolder="tokenizer"
        )
        tokenizer_t5 = T5TokenizerFast.from_pretrained(
            self.pretrained_model, subfolder="tokenizer_2"
        )

        # Load text encoders
        text_encoder_clip = CLIPTextModel.from_pretrained(
            self.pretrained_model, subfolder="text_encoder"
        )

        # Load T5 with 8-bit quantization for memory efficiency
        if self.use_8bit:
            try:
                from transformers import BitsAndBytesConfig
                quantization_config = BitsAndBytesConfig(load_in_8bit=True)
                text_encoder_t5 = T5EncoderModel.from_pretrained(
                    self.pretrained_model,
                    subfolder="text_encoder_2",
                    quantization_config=quantization_config,
                )
                print("T5 encoder loaded with 8-bit quantization")
            except ImportError:
                print("bitsandbytes not available, loading T5 in full precision")
                text_encoder_t5 = T5EncoderModel.from_pretrained(
                    self.pretrained_model, subfolder="text_encoder_2"
                )
        else:
            text_encoder_t5 = T5EncoderModel.from_pretrained(
                self.pretrained_model, subfolder="text_encoder_2"
            )

        # Load transformer (main model)
        transformer = FluxTransformer2DModel.from_pretrained(
            self.pretrained_model, subfolder="transformer"
        )

        # Load VAE
        from diffusers import AutoencoderKL
        vae = AutoencoderKL.from_pretrained(
            self.pretrained_model, subfolder="vae"
        )

        # Scheduler
        scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
            self.pretrained_model, subfolder="scheduler"
        )

        # Freeze encoders and VAE
        text_encoder_clip.requires_grad_(False)
        text_encoder_t5.requires_grad_(False)
        vae.requires_grad_(False)

        # Enable gradient checkpointing
        if self.gradient_checkpointing:
            transformer.enable_gradient_checkpointing()

        # Add LoRA to transformer
        print("Adding LoRA layers to Flux transformer...")
        lora_config = LoraConfig(
            r=self.lora_rank,
            lora_alpha=self.lora_alpha,
            init_lora_weights="gaussian",
            target_modules=[
                "to_k", "to_q", "to_v", "to_out.0",
                "proj_mlp",
            ],
        )
        transformer = get_peft_model(transformer, lora_config)
        transformer.print_trainable_parameters()

        # Create dataset
        print("Loading dataset...")
        dataset = FluxCelebrityDataset(
            data_root=data_root,
            celebrity_id=celebrity_id,
            tokenizer_clip=tokenizer_clip,
            tokenizer_t5=tokenizer_t5,
            size=1024,
            trigger_word=trigger_word,
        )

        dataloader = DataLoader(
            dataset,
            batch_size=self.train_batch_size,
            shuffle=True,
            num_workers=0,
        )

        print(f"Dataset size: {len(dataset)} images")

        # Optimizer - Prodigy recommended for Flux
        try:
            from prodigyopt import Prodigy
            optimizer = Prodigy(
                transformer.parameters(),
                lr=1.0,  # Prodigy adjusts automatically
                weight_decay=0.01,
            )
            print("Using Prodigy optimizer")
        except ImportError:
            print("Prodigy not available, using AdamW")
            optimizer = torch.optim.AdamW(
                transformer.parameters(),
                lr=self.learning_rate,
                weight_decay=0.01,
            )

        # Learning rate scheduler
        lr_scheduler = get_scheduler(
            "constant",
            optimizer=optimizer,
            num_warmup_steps=100,
            num_training_steps=self.max_train_steps,
        )

        # Prepare for distributed training
        transformer, optimizer, dataloader, lr_scheduler = accelerator.prepare(
            transformer, optimizer, dataloader, lr_scheduler
        )

        # Move to device
        weight_dtype = torch.bfloat16 if self.mixed_precision == "bf16" else torch.float16
        vae.to(accelerator.device, dtype=weight_dtype)
        text_encoder_clip.to(accelerator.device, dtype=weight_dtype)
        if not self.use_8bit:
            text_encoder_t5.to(accelerator.device, dtype=weight_dtype)

        # Training loop
        print(f"Starting Flux training for {self.max_train_steps} steps...")
        global_step = 0
        progress_bar = tqdm(
            range(self.max_train_steps),
            desc="Training Flux",
            disable=not accelerator.is_local_main_process,
        )

        transformer.train()

        while global_step < self.max_train_steps:
            for batch in dataloader:
                with accelerator.accumulate(transformer):
                    # Encode images
                    latents = vae.encode(
                        batch["pixel_values"].to(dtype=weight_dtype)
                    ).latent_dist.sample()
                    latents = (latents - vae.config.shift_factor) * vae.config.scaling_factor

                    # Sample noise and timesteps
                    noise = torch.randn_like(latents)
                    bsz = latents.shape[0]

                    # Flux uses flow matching
                    timesteps = torch.rand(bsz, device=latents.device)

                    # Interpolate between noise and latents
                    noisy_latents = (1 - timesteps.view(-1, 1, 1, 1)) * latents + timesteps.view(-1, 1, 1, 1) * noise

                    # Get text embeddings
                    prompt_embeds_clip = text_encoder_clip(
                        batch["input_ids_clip"],
                        output_hidden_states=False,
                    ).pooler_output

                    prompt_embeds_t5 = text_encoder_t5(
                        batch["input_ids_t5"],
                        attention_mask=batch["attention_mask_t5"],
                    ).last_hidden_state

                    # Predict velocity
                    model_pred = transformer(
                        noisy_latents.to(dtype=weight_dtype),
                        timestep=timesteps,
                        encoder_hidden_states=prompt_embeds_t5.to(dtype=weight_dtype),
                        pooled_projections=prompt_embeds_clip.to(dtype=weight_dtype),
                        return_dict=False,
                    )[0]

                    # Target is velocity (noise - latents)
                    target = noise - latents

                    # Calculate loss
                    loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")

                    accelerator.backward(loss)
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()

                # Update progress
                if accelerator.sync_gradients:
                    progress_bar.update(1)
                    global_step += 1

                    if global_step % 50 == 0:
                        print(f"Step {global_step}, Loss: {loss.item():.4f}")

                    if global_step % 250 == 0:
                        self._save_checkpoint(
                            accelerator, transformer, global_step, celebrity_id
                        )

                if global_step >= self.max_train_steps:
                    break

        # Save final model
        print("Saving final Flux LoRA model...")
        self._save_lora_weights(accelerator, transformer, celebrity_id)

        print(f"Training complete! Flux LoRA weights saved to {self.output_dir}")

        return str(self.output_dir / f"{celebrity_id}_flux_lora")

    def _save_checkpoint(self, accelerator, transformer, step, celebrity_id):
        """Save training checkpoint."""
        if accelerator.is_main_process:
            checkpoint_dir = self.output_dir / "checkpoints_flux" / f"step_{step}"
            checkpoint_dir.mkdir(parents=True, exist_ok=True)

            unwrapped = accelerator.unwrap_model(transformer)
            unwrapped.save_pretrained(checkpoint_dir)
            print(f"Checkpoint saved at step {step}")

    def _save_lora_weights(self, accelerator, transformer, celebrity_id):
        """Save final LoRA weights."""
        if accelerator.is_main_process:
            lora_dir = self.output_dir / f"{celebrity_id}_flux_lora"
            lora_dir.mkdir(parents=True, exist_ok=True)

            unwrapped = accelerator.unwrap_model(transformer)
            unwrapped.save_pretrained(lora_dir)

            # Save config
            config = {
                "base_model": self.pretrained_model,
                "model_type": "flux",
                "celebrity_id": celebrity_id,
                "lora_rank": self.lora_rank,
                "lora_alpha": self.lora_alpha,
                "learning_rate": self.learning_rate,
                "max_train_steps": self.max_train_steps,
                "resolution": 1024,
            }

            with open(lora_dir / "training_config.json", "w") as f:
                json.dump(config, f, indent=2)


def main():
    parser = argparse.ArgumentParser(description="Train Flux LoRA for celebrity")
    parser.add_argument("--dataset", type=str, required=True, help="Dataset root directory")
    parser.add_argument("--celebrity-id", type=str, required=True, help="Celebrity ID")
    parser.add_argument("--output-dir", type=str, default="./outputs", help="Output directory")
    parser.add_argument("--trigger-word", type=str, default="ohwx", help="Trigger word")
    parser.add_argument("--learning-rate", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--max-train-steps", type=int, default=1000, help="Max training steps")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size")
    parser.add_argument("--lora-rank", type=int, default=16, help="LoRA rank")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--use-8bit", action="store_true", default=True, help="Use 8-bit quantization")

    args = parser.parse_args()

    trainer = FluxLoRATrainer(
        output_dir=args.output_dir,
        learning_rate=args.learning_rate,
        train_batch_size=args.batch_size,
        max_train_steps=args.max_train_steps,
        lora_rank=args.lora_rank,
        seed=args.seed,
        use_8bit=args.use_8bit,
    )

    trainer.train(
        data_root=args.dataset,
        celebrity_id=args.celebrity_id,
        trigger_word=args.trigger_word,
    )


if __name__ == "__main__":
    main()
