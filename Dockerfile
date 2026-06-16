# Use a lightweight Python base image
FROM python:3.10-slim

# Install build dependencies required for tree-sitter
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Upgrade pip to fix metadata naming bugs
RUN pip install --upgrade pip

# Copy requirements first
COPY requirements.txt .
COPY siamese_best.pt . 


# Install safetensors from the normal Python index
RUN pip install safetensors

# Force PyTorch CPU (Using extra-index-url + cpu tag prevents CUDA downloads while allowing normal dependencies)
RUN pip install torch==2.6.0+cpu --extra-index-url https://download.pytorch.org/whl/cpu

# Install the rest of the requirements
RUN pip install -r requirements.txt

# Pre-download the HuggingFace model during the build step
RUN python -c "from transformers import AutoTokenizer, AutoModel; \
    AutoTokenizer.from_pretrained('microsoft/unixcoder-base'); \
    AutoModel.from_pretrained('microsoft/unixcoder-base')"

# Copy the rest of the application, including siamese_best.pt
COPY . .

# Expose the port FastAPI runs on
EXPOSE 8000

# Start the application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]