"""Gradio UI for celebrity image generation."""

import argparse
from pathlib import Path
from typing import Optional

import gradio as gr
import torch
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
from peft import PeftModel


class ImageGenerator:
    """Image generator with LoRA support."""

    def __init__(
        self,
        base_model: str = "runwayml/stable-diffusion-v1-5",
        lora_path: Optional[str] = None,
        trigger_word: str = "ohwx",
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        self.base_model = base_model
        self.lora_path = lora_path
        self.trigger_word = trigger_word
        self.device = device
        self.pipe = None

    def load_model(self):
        """Load the Stable Diffusion model with LoRA."""
        print(f"Loading base model: {self.base_model}")

        # Load base pipeline
        self.pipe = StableDiffusionPipeline.from_pretrained(
            self.base_model,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            safety_checker=None,
            requires_safety_checker=False,
        )

        # Use faster scheduler
        self.pipe.scheduler = DPMSolverMultistepScheduler.from_config(
            self.pipe.scheduler.config
        )

        # Load LoRA weights if provided
        if self.lora_path and Path(self.lora_path).exists():
            print(f"Loading LoRA weights from: {self.lora_path}")
            self.pipe.unet = PeftModel.from_pretrained(
                self.pipe.unet,
                self.lora_path,
            )
            print("LoRA weights loaded successfully!")
        else:
            print("No LoRA weights loaded - using base model")

        # Move to device
        self.pipe = self.pipe.to(self.device)

        # Enable memory optimizations
        if self.device == "cuda":
            self.pipe.enable_attention_slicing()
            try:
                self.pipe.enable_xformers_memory_efficient_attention()
            except Exception:
                pass  # xformers not available

        print("Model loaded successfully!")

    def generate(
        self,
        prompt: str,
        negative_prompt: str = "",
        num_inference_steps: int = 30,
        guidance_scale: float = 7.5,
        width: int = 512,
        height: int = 512,
        seed: int = -1,
    ):
        """Generate an image from prompt."""
        if self.pipe is None:
            self.load_model()

        # Add trigger word to prompt
        if self.trigger_word and self.trigger_word not in prompt.lower():
            prompt = f"{self.trigger_word} {prompt}"

        print(f"Generating: {prompt}")

        # Set seed
        generator = None
        if seed >= 0:
            generator = torch.Generator(device=self.device).manual_seed(seed)

        # Generate image
        result = self.pipe(
            prompt=prompt,
            negative_prompt=negative_prompt or "blurry, low quality, distorted, deformed",
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            width=width,
            height=height,
            generator=generator,
        )

        return result.images[0]


def create_ui(generator: ImageGenerator):
    """Create the Gradio interface."""

    def generate_image(
        prompt,
        negative_prompt,
        steps,
        guidance,
        width,
        height,
        seed,
    ):
        """Wrapper for image generation."""
        try:
            image = generator.generate(
                prompt=prompt,
                negative_prompt=negative_prompt,
                num_inference_steps=int(steps),
                guidance_scale=float(guidance),
                width=int(width),
                height=int(height),
                seed=int(seed),
            )
            return image
        except Exception as e:
            raise gr.Error(f"Generation failed: {str(e)}")

    # Create interface
    with gr.Blocks(title="Celebrity AI Art Generator") as demo:
        gr.Markdown("# Celebrity AI Art Generator")
        gr.Markdown(f"Use the trigger word **{generator.trigger_word}** in your prompts for best results.")

        with gr.Row():
            with gr.Column(scale=1):
                prompt = gr.Textbox(
                    label="Prompt",
                    placeholder="sara jay swimming in the ocean, sunny day, beach",
                    lines=3,
                )
                negative_prompt = gr.Textbox(
                    label="Negative Prompt",
                    value="blurry, low quality, distorted, deformed, ugly",
                    lines=2,
                )

                with gr.Row():
                    steps = gr.Slider(
                        minimum=10,
                        maximum=50,
                        value=30,
                        step=1,
                        label="Steps",
                    )
                    guidance = gr.Slider(
                        minimum=1,
                        maximum=15,
                        value=7.5,
                        step=0.5,
                        label="Guidance Scale",
                    )

                with gr.Row():
                    width = gr.Slider(
                        minimum=256,
                        maximum=768,
                        value=512,
                        step=64,
                        label="Width",
                    )
                    height = gr.Slider(
                        minimum=256,
                        maximum=768,
                        value=512,
                        step=64,
                        label="Height",
                    )

                seed = gr.Number(
                    label="Seed (-1 for random)",
                    value=-1,
                )

                generate_btn = gr.Button("Generate", variant="primary")

            with gr.Column(scale=1):
                output_image = gr.Image(label="Generated Image", type="pil")

        # Examples
        gr.Examples(
            examples=[
                ["sara jay swimming in the ocean, sunny day, beach, professional photo"],
                ["sara jay at a coffee shop, casual outfit, natural lighting"],
                ["sara jay portrait, studio lighting, professional headshot"],
                ["sara jay in a garden, flowers, spring, bright colors"],
            ],
            inputs=[prompt],
        )

        # Connect button
        generate_btn.click(
            fn=generate_image,
            inputs=[prompt, negative_prompt, steps, guidance, width, height, seed],
            outputs=[output_image],
        )

    return demo


def main():
    parser = argparse.ArgumentParser(description="Celebrity AI Art Generator UI")
    parser.add_argument(
        "--lora-path",
        type=str,
        default="./outputs/sara_jay_001_lora",
        help="Path to LoRA weights",
    )
    parser.add_argument(
        "--base-model",
        type=str,
        default="runwayml/stable-diffusion-v1-5",
        help="Base model",
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
    generator = ImageGenerator(
        base_model=args.base_model,
        lora_path=args.lora_path,
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
