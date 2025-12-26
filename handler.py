import runpod
import os
from diffusers import QwenImageEditPlusPipeline
from diffusers.utils import load_image
import torch
from io import BytesIO
import base64
from PIL import Image


# Load model on startup - use network volume for persistent caching
# CRITICAL: Container filesystem is ephemeral and gets wiped on pod stop
# Network volume (/runpod-volume) is the ONLY way to persist models across cold starts
model_name = "Qwen/Qwen-Image-Edit-2509"

# Check if network volume is available for model caching
MODELS_CACHE_DIR = "/runpod-volume"
if not os.path.exists(MODELS_CACHE_DIR):
    raise RuntimeError(
        f"❌ FATAL: Network volume not found at {MODELS_CACHE_DIR}!\n"
        f"You MUST attach a network volume to your RunPod endpoint.\n"
        f"Without it, models will re-download on every cold start (11.7GB).\n"
        f"Go to: Endpoint Settings → Network Volume → Attach volume at /runpod-volume"
    )

print(f"📁 Using network volume for model cache: {MODELS_CACHE_DIR}")

pipeline = QwenImageEditPlusPipeline.from_pretrained(
    model_name,
    cache_dir=MODELS_CACHE_DIR,
    torch_dtype=torch.bfloat16
)
pipeline.to("cuda")
pipeline.set_progress_bar_config(disable=None)
print("✅ Pipeline loaded")


# Load LoRA once on startup - download to network volume for persistence
# CRITICAL: LoRA must also be on network volume, not in container filesystem
LORA_REPO = "huawei-bayerlab/windowseat-reflection-removal-v1-0"
LORA_SCALE = float(os.getenv('LORA_SCALE', 1.0))

from huggingface_hub import hf_hub_download

print(f"🔄 Loading LoRA from network volume @ scale {LORA_SCALE}...")

# Download LoRA if not cached (first run only)
lora_file = hf_hub_download(
    repo_id=LORA_REPO,
    filename="transformer_lora/pytorch_lora_weights.safetensors",
    cache_dir=os.path.join(MODELS_CACHE_DIR, "lora"),
)

# Load LoRA from network volume
pipeline.load_lora_weights(
    os.path.dirname(lora_file),
    weight_name="pytorch_lora_weights.safetensors",
    adapter_name="custom_lora"
)

pipeline.set_adapters(["custom_lora"], adapter_weights=[LORA_SCALE])
print(f"✅ LoRA loaded from network volume: {lora_file}")

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
