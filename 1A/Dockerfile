# Use official Python base image
FROM python:3.10-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set working directory
WORKDIR /app

# Copy your script and any related files
COPY . /app

# Install system dependencies for fitz (PyMuPDF)
RUN apt-get update && \
    apt-get install -y libglib2.0-0 libgl1-mesa-glx && \
    rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN pip install --no-cache-dir \
    pymupdf \
    onnxruntime \
    torch \
    joblib

# Run script by default
ENTRYPOINT ["python", "main.py"]