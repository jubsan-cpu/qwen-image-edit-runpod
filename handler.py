import runpod
import os
import torch
import base64
from io import BytesIO
from diffusers import QwenImageEditPlusPipeline
from diffusers.utils import load_image
from huggingface_hub import hf_hub_download

# --- 1. Startup: Official Model Loading Pattern ---
model_id = "Qwen/Qwen-Image-Edit-2509"
cache_dir = "/runpod-volume"

if not os.path.exists(cache_dir):
    raise RuntimeError(f"❌ Network volume not found at {cache_dir}!")

print(f"🚀 Loading {model_id} in float16...")
# Matching official pattern: float16/bfloat16 loading then .to('cuda')
pipe = QwenImageEditPlusPipeline.from_pretrained(
    model_id, 
    cache_dir=cache_dir, 
    torch_dtype=torch.float16
)
pipe.to("cuda")
pipe.set_progress_bar_config(disable=None)

# Load LoRA (Huawei Reflection Removal)
print("🔄 Loading LoRA...")
lora_file = hf_hub_download(
    repo_id="huawei-bayerlab/windowseat-reflection-removal-v1-0",
    filename="transformer_lora/pytorch_lora_weights.safetensors",
    cache_dir=os.path.join(cache_dir, "lora")
)
pipe.load_lora_weights(os.path.dirname(lora_file), weight_name="pytorch_lora_weights.safetensors", adapter_name="reflection")
pipe.set_adapters(["reflection"], adapter_weights=[1.0])

print("✅ Worker Aligned and Ready")

# --- 2. Handler: Official Inference Pattern ---
def handler(event):
    try:
        data = event["input"]
        image_url = data.get("image_url")
        prompt = "remove reflections from the image"

        if not image_url:
            return {"error": "Missing 'image_url'"}

        input_image = load_image(image_url)
        
        # Using official inference mode and parameter set
        with torch.inference_mode():
            output = pipe(
                image=[input_image],
                prompt=prompt,
                negative_prompt=" ",
                num_inference_steps=9, # Kept at 9 for speed, you can increase to 40 for quality
                guidance_scale=0.9,
                true_cfg_scale=4.0,
                num_images_per_prompt=1,
                generator=torch.Generator(device="cuda").manual_seed(0)
            )
            output_image = output.images[0]

        buffered = BytesIO()
        output_image.save(buffered, format="PNG")
        return {
            "output_image_base64": base64.b64encode(buffered.getvalue()).decode("utf-8"),
            "prompt": prompt
        }

    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return {"error": str(e)}

# --- 3. Start ---
runpod.serverless.start({"handler": handler})
