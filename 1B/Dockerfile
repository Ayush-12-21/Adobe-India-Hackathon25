# Use official Python base image with PyTorch and Transformers support
FROM python:3.10-slim

# Avoid prompts during build
ENV DEBIAN_FRONTEND=noninteractive

# Set working directory
WORKDIR /app

# Copy files into container
COPY . .

# Install system dependencies
RUN apt-get update && \
    apt-get install -y gcc libglib2.0-0 libsm6 libxrender1 libxext6 poppler-utils && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Default command (overridden during run)
CMD ["python", "chunk.py"]
