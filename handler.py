from fastapi import FastAPI, Response, status
from pydantic import BaseModel
import os
import torch
from diffusers import QwenImageEditPlusPipeline
from diffusers.utils import load_image
from io import BytesIO
import base64
from PIL import Image
from huggingface_hub import hf_hub_download
import uvicorn
import threading
from contextlib import asynccontextmanager

# --- STATE MANAGEMENT ---
class AppState:
    def __init__(self):
        self.pipeline = None
        self.is_ready = False
        self.error = None

state = AppState()

def load_models():
    """Background task to load models without blocking server startup."""
    try:
        print("🚀 Starting model loading in background...")
        
        model_name = "Qwen/Qwen-Image-Edit-2509"
        MODELS_CACHE_DIR = "/runpod-volume"

        if not os.path.exists(MODELS_CACHE_DIR):
            raise RuntimeError(
                f"❌ FATAL: Network volume not found at {MODELS_CACHE_DIR}!\n"
                f"You MUST attach a network volume to your RunPod endpoint."
            )

        print(f"📁 Using cache directory: {MODELS_CACHE_DIR}")

        # Load pipeline
        state.pipeline = QwenImageEditPlusPipeline.from_pretrained(
            model_name,
            cache_dir=MODELS_CACHE_DIR,
            torch_dtype=torch.bfloat16,
            device_map="balanced"
        )
        state.pipeline.set_progress_bar_config(disable=None)
        
        # Load LoRA
        LORA_REPO = "huawei-bayerlab/windowseat-reflection-removal-v1-0"
        LORA_SCALE = float(os.getenv('LORA_SCALE', 1.0))

        print(f"🔄 Loading LoRA @ scale {LORA_SCALE}...")
        lora_file = hf_hub_download(
            repo_id=LORA_REPO,
            filename="transformer_lora/pytorch_lora_weights.safetensors",
            cache_dir=os.path.join(MODELS_CACHE_DIR, "lora"),
        )

        state.pipeline.load_lora_weights(
            os.path.dirname(lora_file),
            weight_name="pytorch_lora_weights.safetensors",
            adapter_name="custom_lora"
        )
        state.pipeline.set_adapters(["custom_lora"], adapter_weights=[LORA_SCALE])
        
        state.is_ready = True
        print("✅ Models loaded and worker is READY")
    except Exception as e:
        state.error = str(e)
        print(f"❌ Critical Error during loading: {state.error}")

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Start loading in a separate thread so the server starts immediately
    thread = threading.Thread(target=load_models)
    thread.start()
    yield

app = FastAPI(lifespan=lifespan)

# --- MODELS ---
class InputData(BaseModel):
    image_url: str

class RunPodRequest(BaseModel):
    input: InputData

# --- ENDPOINTS ---

@app.get("/ping")
async def health_check(response: Response):
    """
    RunPod Health Check:
    - 200: Healthy (Ready to work)
    - 204: Initializing (Cold start in progress)
    - 500: Error
    """
    if state.error:
        response.status_code = status.HTTP_500_INTERNAL_SERVER_ERROR
        return {"status": "error", "message": state.error}
    
    if not state.is_ready:
        response.status_code = status.HTTP_204_NO_CONTENT
        return None # 204 has no body
    
    return {"status": "healthy"}

@app.post("/")
@app.post("/generate")
async def generate(request: RunPodRequest, response: Response):
    if not state.is_ready:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
        return {"error": "Worker is still initializing", "success": False}
    
    if state.error:
        response.status_code = status.HTTP_500_INTERNAL_SERVER_ERROR
        return {"error": state.error, "success": False}

    try:
        image_url = request.input.image_url
        prompt = "remove reflections from the image"

        print(f"📸 Processing image: {image_url}")
        input_image = load_image(image_url)
        
        with torch.inference_mode():
            output = state.pipeline(
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

        return {
            "output_image_base64": img_str, 
            "prompt": prompt,
            "success": True
        }
    except Exception as e:
        print(f"❌ Inference Error: {str(e)}")
        response.status_code = status.HTTP_500_INTERNAL_SERVER_ERROR
        return {"error": str(e), "success": False}

if __name__ == "__main__":
    port = int(os.getenv("PORT", 80))
    print(f"📡 Serving on port {port}")
    uvicorn.run(app, host="0.0.0.0", port=port)
