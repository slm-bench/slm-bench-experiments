FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV TRANSFORMERS_CACHE=/app/cache
ENV HF_HOME=/app/cache

# Install Python and basic dependencies
RUN apt-get update && apt-get install -y \
    python3.8 \
    python3-pip \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements first to leverage Docker cache
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy project files
COPY train.py .
COPY models.json .
COPY datasets.json .

# Create directories for results and cache
RUN mkdir -p experiment_results cache logs

# Set permissions
RUN chmod -R 777 /app

# Default command
ENTRYPOINT ["python3", "train.py"]