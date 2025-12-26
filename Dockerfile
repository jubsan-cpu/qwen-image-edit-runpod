# Use a valid RunPod PyTorch base image
FROM runpod/pytorch:2.9.1-py3.11-cuda12.6-devel-ubuntu22.04

# Set environment variable for faster downloads
ENV HF_HUB_ENABLE_HF_TRANSFER=1

# Copy requirements file
WORKDIR /app
COPY requirements.txt .

# Install dependencies from requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download the base model and LoRA to eliminate cold start downloads
RUN python -c "from diffusers import DiffusionPipeline; DiffusionPipeline.from_pretrained('Qwen/Qwen-Image-Edit-2509', trust_remote_code=True)"
RUN python -c "from huggingface_hub import hf_hub_download; hf_hub_download('huawei-bayerlab/windowseat-reflection-removal-v1-0', filename='pytorch_lora_weights.safetensors')"

# Copy handler file
COPY handler.py .

# Set entrypoint
CMD ["python", "handler.py"]