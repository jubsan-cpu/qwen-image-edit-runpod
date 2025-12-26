# Use a valid RunPod PyTorch base image
FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

# Copy requirements file
WORKDIR /app
COPY requirements.txt .

# Install dependencies from requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy handler file
COPY handler.py .

# Set entrypoint
CMD ["python", "handler.py"]