# Use a valid RunPod PyTorch base image (Upgraded to 2.5.1 for GQA support)
FROM mnb3000/runpod-pytorch:2.5.1-py3.11-cuda12.4.1-devel-ubuntu22.04

# Copy requirements file (!)
WORKDIR /app
COPY requirements.txt .

# Install dependencies from requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy handler file
COPY handler.py .

# Expose the API port
EXPOSE 80

# Load Balancing workers start the FastAPI server directly
CMD ["python", "handler.py"]