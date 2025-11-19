"""Gradio UI for celebrity image generation with multi-model support."""

import argparse
from pathlib import Path
from typing import Optional
import json

import gradio as gr
import numpy as np
import torch
from diffusers import (
    StableDiffusionPipeline,
    StableDiffusionXLPipeline,
    DPMSolverMultistepScheduler,
)
from peft import PeftModel
from PIL import Image

# Import configurations
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.config.model_configs import (
    MODEL_CONFIGS,
    DEFAULT_NEGATIVE_PROMPT,
    EXAMPLE_PROMPTS,
    get_model_config,
)

# Try to import face restoration
try:
    from gfpgan import GFPGANer
    GFPGAN_AVAILABLE = True
except ImportError:
    GFPGAN_AVAILABLE = False
    print("GFPGAN not available - face restoration disabled")

# Try to import ControlNet
try:
    from diffusers import ControlNetModel, StableDiffusionXLControlNetPipeline
    from controlnet_aux import OpenposeDetector
    CONTROLNET_AVAILABLE = True
except ImportError:
    CONTROLNET_AVAILABLE = False
    print("ControlNet not available - pose guidance disabled")

# Try to import Flux
try:
    from diffusers import FluxPipeline
    FLUX_AVAILABLE = True
except ImportError:
    FLUX_AVAILABLE = False
    print("Flux not available - install diffusers>=0.30.0")


class MultiModelGenerator:
    """Image generator with multi-model and ControlNet support."""

    def __init__(
        self,
        lora_dir: str = "./outputs",
        celebrity_id: str = "celebrity_001",
        trigger_word: str = "ohwx",
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        self.lora_dir = Path(lora_dir)
        self.celebrity_id = celebrity_id
        self.trigger_word = trigger_word
        self.device = device

        self.current_model = None
        self.pipe = None
        self.face_enhancer = None
        self.openpose = None

    def load_model(self, model_type: str = "SDXL"):
        """Load the specified model with LoRA."""

        # Check if already loaded
        if self.current_model == model_type and self.pipe is not None:
            return

        # Clear previous model
        if self.pipe is not None:
            del self.pipe
            torch.cuda.empty_cache()

        config = get_model_config(model_type)
        print(f"Loading {model_type} model: {config.base_model}")

        # Select pipeline class based on model type
        if model_type in ["Flux Dev", "Flux Schnell"]:
            if not FLUX_AVAILABLE:
                raise RuntimeError("Flux not available. Install diffusers>=0.30.0")
            self.pipe = FluxPipeline.from_pretrained(
                config.base_model,
                torch_dtype=torch.bfloat16 if self.device == "cuda" else torch.float32,
            )
        elif model_type == "SDXL":
            self.pipe = StableDiffusionXLPipeline.from_pretrained(
                config.base_model,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                safety_checker=None,
                requires_safety_checker=False,
                use_safetensors=True,
            )
        else:  # SD 1.5
            self.pipe = StableDiffusionPipeline.from_pretrained(
                config.base_model,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                safety_checker=None,
                requires_safety_checker=False,
            )

        # Use faster scheduler (not for Flux)
        if model_type not in ["Flux Dev", "Flux Schnell"]:
            self.pipe.scheduler = DPMSolverMultistepScheduler.from_config(
                self.pipe.scheduler.config
            )

        # Load LoRA weights if available
        lora_path = self.lora_dir / f"{self.celebrity_id}{config.lora_suffix}"
        if lora_path.exists():
            print(f"Loading LoRA weights from: {lora_path}")
            if model_type in ["Flux Dev", "Flux Schnell"]:
                # Flux uses transformer instead of unet
                self.pipe.transformer = PeftModel.from_pretrained(
                    self.pipe.transformer,
                    lora_path,
                )
            else:
                self.pipe.unet = PeftModel.from_pretrained(
                    self.pipe.unet,
                    lora_path,
                )
            print("LoRA weights loaded successfully!")
        else:
            print(f"No LoRA weights found at {lora_path} - using base model")

        # Move to device
        self.pipe = self.pipe.to(self.device)

        # Enable memory optimizations
        if self.device == "cuda":
            self.pipe.enable_attention_slicing()
            try:
                self.pipe.enable_xformers_memory_efficient_attention()
            except Exception:
                pass

        self.current_model = model_type
        print(f"{model_type} model loaded successfully!")

        # Load face enhancer if needed
        if GFPGAN_AVAILABLE and self.face_enhancer is None:
            try:
                self.face_enhancer = GFPGANer(
                    model_path='https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.3.pth',
                    upscale=1,
                    arch='clean',
                    channel_multiplier=2,
                    bg_upsampler=None
                )
                print("Face enhancer loaded!")
            except Exception as e:
                print(f"Failed to load face enhancer: {e}")

    def enhance_face(self, image: Image.Image) -> Image.Image:
        """Apply face restoration using GFPGAN."""
        if self.face_enhancer is None:
            return image

        try:
            img_array = np.array(image)
            _, _, output = self.face_enhancer.enhance(
                img_array,
                has_aligned=False,
                only_center_face=False,
                paste_back=True
            )
            return Image.fromarray(output)
        except Exception as e:
            print(f"Face enhancement failed: {e}")
            return image

    def upscale_image(self, image: Image.Image, scale: int = 2) -> Image.Image:
        """Simple upscaling using Lanczos interpolation."""
        new_size = (image.width * scale, image.height * scale)
        return image.resize(new_size, Image.LANCZOS)

    def generate(
        self,
        prompt: str,
        model_type: str = "SDXL",
        negative_prompt: str = "",
        num_inference_steps: int = 30,
        guidance_scale: float = 7.5,
        seed: int = -1,
        enhance_face: bool = True,
        upscale: bool = False,
    ):
        """Generate an image from prompt with specified model."""

        # Load model if needed
        self.load_model(model_type)

        config = get_model_config(model_type)
        width = height = config.resolution

        # Add trigger word to prompt
        if self.trigger_word and self.trigger_word not in prompt.lower():
            prompt = f"{self.trigger_word} {prompt}"

        # Add quality boosters
        if "professional" not in prompt.lower() and "high quality" not in prompt.lower():
            prompt = f"{prompt}, high quality, detailed, sharp focus, professional photography"

        print(f"Generating with {model_type}: {prompt}")

        # Set seed
        generator = None
        if seed >= 0:
            generator = torch.Generator(device=self.device).manual_seed(seed)

        # Prepare generation kwargs
        gen_kwargs = {
            "prompt": prompt,
            "num_inference_steps": num_inference_steps,
            "guidance_scale": guidance_scale,
            "width": width,
            "height": height,
            "generator": generator,
        }

        # Add negative prompt only for models that support it
        if config.supports_negative_prompt:
            gen_kwargs["negative_prompt"] = negative_prompt or DEFAULT_NEGATIVE_PROMPT

        # Generate image
        result = self.pipe(**gen_kwargs)
        image = result.images[0]

        # Apply face enhancement
        if enhance_face and GFPGAN_AVAILABLE:
            print("Enhancing face...")
            image = self.enhance_face(image)

        # Apply upscaling
        if upscale:
            print("Upscaling image...")
            image = self.upscale_image(image, scale=2)

        return image


def create_ui(generator: MultiModelGenerator):
    """Create the Gradio interface with model selection and dynamic configs."""

    def update_config(model_type):
        """Update UI elements when model changes."""
        config = get_model_config(model_type)
        return (
            config.default_steps,
            config.default_guidance,
            gr.update(visible=config.supports_negative_prompt),
            f"Resolution: {config.resolution}x{config.resolution} | VRAM: {config.vram_required} | {config.description}"
        )

    def generate_image(
        prompt,
        model_type,
        negative_prompt,
        steps,
        guidance,
        seed,
        enhance_face,
        upscale,
    ):
        """Wrapper for image generation."""
        try:
            image = generator.generate(
                prompt=prompt,
                model_type=model_type,
                negative_prompt=negative_prompt,
                num_inference_steps=int(steps),
                guidance_scale=float(guidance),
                seed=int(seed),
                enhance_face=enhance_face,
                upscale=upscale,
            )
            return image
        except Exception as e:
            raise gr.Error(f"Generation failed: {str(e)}")

    # Get available models (filter out Flux if not available)
    available_models = list(MODEL_CONFIGS.keys())
    if not FLUX_AVAILABLE:
        available_models = [m for m in available_models if "Flux" not in m]

    # Create interface
    with gr.Blocks(title="Celebrity AI Art Generator", theme=gr.themes.Soft()) as demo:
        gr.Markdown("# Celebrity AI Art Generator")
        gr.Markdown(f"Trigger word: **{generator.trigger_word}** | Select model for best results")

        with gr.Row():
            with gr.Column(scale=1):
                # Model selector
                model_type = gr.Dropdown(
                    choices=available_models,
                    value="SDXL",
                    label="Model",
                )

                # Model info display
                model_info = gr.Markdown(
                    f"Resolution: 1024x1024 | VRAM: 8GB+ | Better anatomy, recommended for production."
                )

                prompt = gr.Textbox(
                    label="Prompt",
                    placeholder="brad pitt walking on the beach, sunny day, professional photo",
                    lines=3,
                )
                negative_prompt = gr.Textbox(
                    label="Negative Prompt",
                    value=DEFAULT_NEGATIVE_PROMPT,
                    lines=3,
                    visible=True,
                )

                with gr.Row():
                    steps = gr.Slider(
                        minimum=1,
                        maximum=50,
                        value=30,
                        step=1,
                        label="Steps",
                    )
                    guidance = gr.Slider(
                        minimum=0,
                        maximum=20,
                        value=7.5,
                        step=0.5,
                        label="Guidance Scale",
                    )

                seed = gr.Number(
                    label="Seed (-1 for random)",
                    value=-1,
                )

                with gr.Row():
                    enhance_face = gr.Checkbox(
                        label="Face Enhancement",
                        value=GFPGAN_AVAILABLE,
                        interactive=GFPGAN_AVAILABLE,
                    )
                    upscale = gr.Checkbox(
                        label="2x Upscale",
                        value=False,
                    )

                generate_btn = gr.Button("Generate", variant="primary", size="lg")

            with gr.Column(scale=1):
                output_image = gr.Image(label="Generated Image", type="pil")

        # Update configs when model changes
        model_type.change(
            fn=update_config,
            inputs=[model_type],
            outputs=[steps, guidance, negative_prompt, model_info],
        )

        # Model comparison info
        with gr.Accordion("Model Information", open=False):
            gr.Markdown("""
            ### Model Comparison

            | Model | Resolution | Anatomy | Speed | VRAM | Notes |
            |-------|-----------|---------|-------|------|-------|
            | SD 1.5 | 512x512 | Poor | Fast | 4GB+ | Legacy, testing only |
            | SDXL | 1024x1024 | Good | Medium | 8GB+ | Recommended |
            | Flux Dev | 1024x1024 | Excellent | Slow | 12GB+ | Best quality |
            | Flux Schnell | 1024x1024 | Good | Fast | 12GB+ | Quick previews |

            **Note**: Flux models don't support negative prompts. Describe what you want positively.
            """)

        # Tips
        with gr.Accordion("Tips for Better Results", open=False):
            gr.Markdown("""
            **For better anatomy:**
            - Use SDXL or Flux (much better anatomy understanding)
            - Use specific poses: "standing", "sitting", "portrait"
            - Add quality terms: "professional photo", "sharp focus"

            **For Flux models:**
            - Use natural language descriptions
            - No negative prompts - describe what you want
            - Lower guidance (3-4) works best

            **For consistent results:**
            - Set a specific seed number
            - Use 20-30 steps for quality
            """)

        # Examples
        gr.Examples(
            examples=EXAMPLE_PROMPTS,
            inputs=[prompt],
        )

        # Connect button
        generate_btn.click(
            fn=generate_image,
            inputs=[prompt, model_type, negative_prompt, steps, guidance, seed, enhance_face, upscale],
            outputs=[output_image],
        )

    return demo


def main():
    parser = argparse.ArgumentParser(description="Celebrity AI Art Generator UI")
    parser.add_argument(
        "--lora-dir",
        type=str,
        default="./outputs",
        help="Directory containing LoRA weights",
    )
    parser.add_argument(
        "--celebrity-id",
        type=str,
        default="celebrity_001",
        help="Celebrity ID",
    )
    parser.add_argument(
        "--trigger-word",
        type=str,
        default="ohwx",
        help="Trigger word",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=7860,
        help="Port to run on",
    )
    parser.add_argument(
        "--share",
        action="store_true",
        help="Create public link",
    )

    args = parser.parse_args()

    # Create generator
    generator = MultiModelGenerator(
        lora_dir=args.lora_dir,
        celebrity_id=args.celebrity_id,
        trigger_word=args.trigger_word,
    )

    # Create and launch UI
    demo = create_ui(generator)
    demo.launch(
        server_port=args.port,
        share=args.share,
        inbrowser=True,
    )


if __name__ == "__main__":
    main()
