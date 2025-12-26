# Use a valid RunPod PyTorch base image
FROM runpod/pytorch:latest-py3.11-cuda12.4-devel-ubuntu22.04

# Set environment variables for faster downloads and cache location
ENV HF_HUB_ENABLE_HF_TRANSFER=1
ENV HF_HOME=/app/.cache/huggingface

# Copy requirements file
WORKDIR /app
COPY requirements.txt .

# Install dependencies from requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Download LoRA to guaranteed fixed location (not cache)
RUN mkdir -p /models/lora && \
    python -c "from huggingface_hub import hf_hub_download; \
    hf_hub_download( \
        repo_id='huawei-bayerlab/windowseat-reflection-removal-v1-0', \
        filename='pytorch_lora_weights.safetensors', \
        local_dir='/models/lora', \
        local_dir_use_symlinks=False \
    )"

# Copy handler file
COPY handler.py .

# Set entrypoint
CMD ["python", "handler.py"]