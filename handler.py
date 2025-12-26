import runpod
import os
from diffusers import QwenImageEditPlusPipeline
from diffusers.utils import load_image
import torch
from io import BytesIO
import base64
from PIL import Image


# Load model on startup using the official Qwen pipeline
# Hugging Face automatically checks RunPod's cache and downloads if needed
model_name = "Qwen/Qwen-Image-Edit-2509"
pipeline = QwenImageEditPlusPipeline.from_pretrained(
    model_name,
    trust_remote_code=True,  # Required for Qwen models
    torch_dtype=torch.bfloat16
)
pipeline.to("cuda")
pipeline.set_progress_bar_config(disable=None)
print("✅ Pipeline loaded")


# Load LoRA once on startup (from guaranteed fixed location in Docker image)
LORA_PATH = "/models/lora/transformer_lora"  # Guaranteed location from Dockerfile
LORA_SCALE = float(os.getenv('LORA_SCALE', 1.0))
LORA_FILENAME = "pytorch_lora_weights.safetensors"

# Load LoRA from fixed path
print(f"🔄 Loading LoRA from {LORA_PATH} @ scale {LORA_SCALE}")
pipeline.load_lora_weights(
    LORA_PATH,
    weight_name=LORA_FILENAME,
    adapter_name="custom_lora"
)

pipeline.set_adapters(["custom_lora"], adapter_weights=[LORA_SCALE])
print(f"✅ LoRA loaded from {LORA_PATH}/{LORA_FILENAME}")

def handler(event):
    """
    Runpod handler function. Receives job input and returns output.
    """
    try:
        input_data = event["input"]
        image_url = input_data.get("image_url")

        if not image_url:
            return {"error": "Missing 'image_url' parameter."}

        # Fixed prompt for reflection removal
        prompt = "remove reflections from the image"

        # Load input image
        input_image = load_image(image_url)
        
        # Use torch.inference_mode() for optimal performance
        with torch.inference_mode():
            output = pipeline(
                image=[input_image],
                prompt=prompt,
                negative_prompt=" ",
                num_inference_steps=9,
                guidance_scale=0.9,
                true_cfg_scale=4.0,
                num_images_per_prompt=1,
                generator=torch.Generator(device="cuda").manual_seed(0)
            )
            output_image = output.images[0]

        buffered = BytesIO()
        output_image.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")

        return {"output_image_base64": img_str, "prompt": prompt}
    except Exception as e:
        return {"error": str(e)}

# Required by Runpod
runpod.serverless.start({"handler": handler})
