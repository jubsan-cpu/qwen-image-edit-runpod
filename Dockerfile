# Use a modern Runpod PyTorch base image
FROM runpod/pytorch:1.0.2-cu1281-torch271-ubuntu2204

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