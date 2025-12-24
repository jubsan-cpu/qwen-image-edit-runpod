import runpod
import os
from diffusers import DiffusionPipeline
from diffusers.utils import load_image
import torch
from io import BytesIO
import base64
from PIL import Image

# Load model on startup
pipe = DiffusionPipeline.from_pretrained(
    "Qwen/Qwen-Image-Edit-2509", 
    torch_dtype=torch.float16,
    trust_remote_code=True  # Required for Qwen models
).to("cuda")

# Load LoRA once on startup (if specified)
LORA_URL = os.getenv('LORA_URL', 'huawei-bayerlab/windowseat-reflection-removal-v1-0')
LORA_SCALE = float(os.getenv('LORA_SCALE', 1.0))

if LORA_URL:
    print(f"🔄 Loading LoRA on startup: {LORA_URL} @ scale {LORA_SCALE}")
    if LORA_URL.startswith('http'):
        pipe.load_lora_weights(LORA_URL, weight_name="pytorch_lora_weights.safetensors", adapter_name="custom_lora")
    else:
        pipe.load_lora_weights(
            LORA_URL, 
            weight_name="pytorch_lora_weights.safetensors",
            subfolder="transformer_lora",
            adapter_name="custom_lora"
        )
    pipe.set_adapters(["custom_lora"], adapter_weights=[LORA_SCALE])
    print(f"✅ LoRA loaded permanently: {LORA_URL}")

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

        # Fixed inference steps for optimal speed
        num_steps = 1
        
        input_image = load_image(image_url)
        output_image = pipe(
            image=input_image, 
            prompt=prompt,
            num_inference_steps=num_steps
        ).images[0]

        buffered = BytesIO()
        output_image.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")

        return {"output_image_base64": img_str, "prompt": prompt}
    except Exception as e:
        return {"error": str(e)}

# Required by Runpod
runpod.serverless.start({"handler": handler})
