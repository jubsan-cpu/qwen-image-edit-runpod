# Use a valid RunPod PyTorch base image (Upgraded to 2.5.1 for GQA support)
FROM runpod/pytorch:1.0.3-cu1300-torch291-ubuntu2404
# Setup working directory
WORKDIR /app
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy only the handler script
COPY handler.py .

# Run the handler directly
CMD ["python", "handler.py"]