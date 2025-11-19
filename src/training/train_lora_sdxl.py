"""SDXL LoRA training script for celebrity fine-tuning with improved anatomy."""

import argparse
import math
import os
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F
from accelerate import Accelerator
from accelerate.utils import set_seed
from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    StableDiffusionXLPipeline,
    UNet2DConditionModel,
)
from diffusers.optimization import get_scheduler
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTextModelWithProjection, CLIPTokenizer
from peft import LoraConfig, get_peft_model
import json


class SDXLCelebrityDataset(Dataset):
    """Dataset for loading celebrity images with captions for SDXL training."""

    def __init__(
        self,
        data_root: str,
        celebrity_id: str,
        tokenizer_one: CLIPTokenizer,
        tokenizer_two: CLIPTokenizer,
        size: int = 1024,  # SDXL native resolution
        trigger_word: str = "ohwx",
    ):
        self.data_root = Path(data_root)
        self.celebrity_id = celebrity_id
        self.tokenizer_one = tokenizer_one
        self.tokenizer_two = tokenizer_two
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

        # Image transforms for SDXL (1024x1024)
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

        # Get caption
        filename = image_path.name
        caption = self.captions.get(filename, f"{self.trigger_word} person")

        # Tokenize caption for both text encoders
        tokens_one = self.tokenizer_one(
            caption,
            max_length=self.tokenizer_one.model_max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        tokens_two = self.tokenizer_two(
            caption,
            max_length=self.tokenizer_two.model_max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        return {
            "pixel_values": image,
            "input_ids_one": tokens_one.input_ids[0],
            "input_ids_two": tokens_two.input_ids[0],
        }


class SDXLLoRATrainer:
    """Trainer for SDXL LoRA fine-tuning with better anatomy."""

    def __init__(
        self,
        pretrained_model: str = "stabilityai/stable-diffusion-xl-base-1.0",
        output_dir: str = "./outputs",
        learning_rate: float = 1e-4,
        train_batch_size: int = 1,
        gradient_accumulation_steps: int = 4,
        max_train_steps: int = 2000,
        lora_rank: int = 64,  # Higher rank for SDXL
        lora_alpha: int = 64,
        seed: int = 42,
        gradient_checkpointing: bool = True,
        mixed_precision: str = "fp16",
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

        self.output_dir.mkdir(parents=True, exist_ok=True)

    def train(
        self,
        data_root: str,
        celebrity_id: str,
        trigger_word: str = "ohwx",
        resume_from: Optional[str] = None,
    ):
        """Train SDXL LoRA on celebrity dataset."""

        # Initialize accelerator
        accelerator = Accelerator(
            gradient_accumulation_steps=self.gradient_accumulation_steps,
            mixed_precision=self.mixed_precision,
        )

        set_seed(self.seed)

        # Load SDXL models
        print("Loading SDXL models...")

        # Load tokenizers
        tokenizer_one = CLIPTokenizer.from_pretrained(
            self.pretrained_model, subfolder="tokenizer"
        )
        tokenizer_two = CLIPTokenizer.from_pretrained(
            self.pretrained_model, subfolder="tokenizer_2"
        )

        # Load text encoders
        text_encoder_one = CLIPTextModel.from_pretrained(
            self.pretrained_model, subfolder="text_encoder"
        )
        text_encoder_two = CLIPTextModelWithProjection.from_pretrained(
            self.pretrained_model, subfolder="text_encoder_2"
        )

        # Load VAE and UNet
        vae = AutoencoderKL.from_pretrained(
            self.pretrained_model, subfolder="vae"
        )
        unet = UNet2DConditionModel.from_pretrained(
            self.pretrained_model, subfolder="unet"
        )
        noise_scheduler = DDPMScheduler.from_pretrained(
            self.pretrained_model, subfolder="scheduler"
        )

        # Freeze VAE and text encoders
        vae.requires_grad_(False)
        text_encoder_one.requires_grad_(False)
        text_encoder_two.requires_grad_(False)

        # Enable gradient checkpointing for memory efficiency
        if self.gradient_checkpointing:
            unet.enable_gradient_checkpointing()

        # Add LoRA to UNet with SDXL-optimized settings
        print("Adding LoRA layers...")
        lora_config = LoraConfig(
            r=self.lora_rank,
            lora_alpha=self.lora_alpha,
            init_lora_weights="gaussian",
            target_modules=[
                "to_k", "to_q", "to_v", "to_out.0",
                "proj_in", "proj_out",
                "ff.net.0.proj", "ff.net.2",
            ],
        )
        unet = get_peft_model(unet, lora_config)
        unet.print_trainable_parameters()

        # Create dataset and dataloader
        print("Loading dataset...")
        dataset = SDXLCelebrityDataset(
            data_root=data_root,
            celebrity_id=celebrity_id,
            tokenizer_one=tokenizer_one,
            tokenizer_two=tokenizer_two,
            size=1024,  # SDXL native resolution
            trigger_word=trigger_word,
        )

        dataloader = DataLoader(
            dataset,
            batch_size=self.train_batch_size,
            shuffle=True,
            num_workers=0,
        )

        print(f"Dataset size: {len(dataset)} images")

        # Optimizer with weight decay
        optimizer = torch.optim.AdamW(
            unet.parameters(),
            lr=self.learning_rate,
            betas=(0.9, 0.999),
            weight_decay=1e-2,
            eps=1e-8,
        )

        # Learning rate scheduler
        lr_scheduler = get_scheduler(
            "cosine",
            optimizer=optimizer,
            num_warmup_steps=100,
            num_training_steps=self.max_train_steps,
        )

        # Prepare for distributed training
        unet, optimizer, dataloader, lr_scheduler = accelerator.prepare(
            unet, optimizer, dataloader, lr_scheduler
        )

        # Move to device
        weight_dtype = torch.float16 if self.mixed_precision == "fp16" else torch.float32
        vae.to(accelerator.device, dtype=weight_dtype)
        text_encoder_one.to(accelerator.device, dtype=weight_dtype)
        text_encoder_two.to(accelerator.device, dtype=weight_dtype)

        # Training loop
        print(f"Starting SDXL training for {self.max_train_steps} steps...")
        global_step = 0
        progress_bar = tqdm(
            range(self.max_train_steps),
            desc="Training SDXL",
            disable=not accelerator.is_local_main_process,
        )

        unet.train()

        while global_step < self.max_train_steps:
            for batch in dataloader:
                with accelerator.accumulate(unet):
                    # Convert images to latent space
                    latents = vae.encode(
                        batch["pixel_values"].to(dtype=weight_dtype)
                    ).latent_dist.sample()
                    latents = latents * vae.config.scaling_factor

                    # Sample noise
                    noise = torch.randn_like(latents)
                    bsz = latents.shape[0]

                    # Sample timesteps
                    timesteps = torch.randint(
                        0,
                        noise_scheduler.config.num_train_timesteps,
                        (bsz,),
                        device=latents.device,
                    ).long()

                    # Add noise to latents
                    noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                    # Get text embeddings from both encoders
                    prompt_embeds_one = text_encoder_one(
                        batch["input_ids_one"],
                        output_hidden_states=True,
                    ).hidden_states[-2]

                    prompt_embeds_two = text_encoder_two(
                        batch["input_ids_two"],
                        output_hidden_states=True,
                    )
                    pooled_prompt_embeds = prompt_embeds_two[0]
                    prompt_embeds_two = prompt_embeds_two.hidden_states[-2]

                    # Concatenate embeddings
                    prompt_embeds = torch.concat([prompt_embeds_one, prompt_embeds_two], dim=-1)
                    prompt_embeds = prompt_embeds.to(dtype=weight_dtype)
                    pooled_prompt_embeds = pooled_prompt_embeds.to(dtype=weight_dtype)

                    # Time ids for SDXL
                    add_time_ids = torch.tensor([[
                        1024, 1024,  # Original size
                        0, 0,        # Crop top-left
                        1024, 1024,  # Target size
                    ]], device=latents.device).repeat(bsz, 1)

                    # Predict noise
                    added_cond_kwargs = {
                        "text_embeds": pooled_prompt_embeds,
                        "time_ids": add_time_ids,
                    }

                    model_pred = unet(
                        noisy_latents.to(dtype=weight_dtype),
                        timesteps,
                        prompt_embeds,
                        added_cond_kwargs=added_cond_kwargs,
                    ).sample

                    # Calculate loss
                    loss = F.mse_loss(model_pred.float(), noise.float(), reduction="mean")

                    accelerator.backward(loss)
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()

                # Update progress
                if accelerator.sync_gradients:
                    progress_bar.update(1)
                    global_step += 1

                    # Log every 50 steps
                    if global_step % 50 == 0:
                        print(f"Step {global_step}, Loss: {loss.item():.4f}")

                    # Save checkpoint every 500 steps
                    if global_step % 500 == 0:
                        self._save_checkpoint(
                            accelerator, unet, global_step, celebrity_id
                        )

                if global_step >= self.max_train_steps:
                    break

        # Save final model
        print("Saving final SDXL LoRA model...")
        self._save_lora_weights(accelerator, unet, celebrity_id)

        print(f"Training complete! SDXL LoRA weights saved to {self.output_dir}")

        return str(self.output_dir / f"{celebrity_id}_sdxl_lora")

    def _save_checkpoint(self, accelerator, unet, step, celebrity_id):
        """Save training checkpoint."""
        if accelerator.is_main_process:
            checkpoint_dir = self.output_dir / "checkpoints_sdxl" / f"step_{step}"
            checkpoint_dir.mkdir(parents=True, exist_ok=True)

            unwrapped_unet = accelerator.unwrap_model(unet)
            unwrapped_unet.save_pretrained(checkpoint_dir)
            print(f"Checkpoint saved at step {step}")

    def _save_lora_weights(self, accelerator, unet, celebrity_id):
        """Save final LoRA weights."""
        if accelerator.is_main_process:
            lora_dir = self.output_dir / f"{celebrity_id}_sdxl_lora"
            lora_dir.mkdir(parents=True, exist_ok=True)

            unwrapped_unet = accelerator.unwrap_model(unet)
            unwrapped_unet.save_pretrained(lora_dir)

            # Save config
            config = {
                "base_model": self.pretrained_model,
                "model_type": "sdxl",
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
    parser = argparse.ArgumentParser(description="Train SDXL LoRA for celebrity")
    parser.add_argument("--dataset", type=str, required=True, help="Dataset root directory")
    parser.add_argument("--celebrity-id", type=str, required=True, help="Celebrity ID")
    parser.add_argument("--output-dir", type=str, default="./outputs", help="Output directory")
    parser.add_argument("--trigger-word", type=str, default="ohwx", help="Trigger word")
    parser.add_argument("--learning-rate", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--max-train-steps", type=int, default=2000, help="Max training steps")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size")
    parser.add_argument("--lora-rank", type=int, default=64, help="LoRA rank (default: 64)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--gradient-checkpointing", action="store_true", default=True,
                        help="Enable gradient checkpointing")

    args = parser.parse_args()

    trainer = SDXLLoRATrainer(
        output_dir=args.output_dir,
        learning_rate=args.learning_rate,
        train_batch_size=args.batch_size,
        max_train_steps=args.max_train_steps,
        lora_rank=args.lora_rank,
        seed=args.seed,
        gradient_checkpointing=args.gradient_checkpointing,
    )

    trainer.train(
        data_root=args.dataset,
        celebrity_id=args.celebrity_id,
        trigger_word=args.trigger_word,
    )


if __name__ == "__main__":
    main()
