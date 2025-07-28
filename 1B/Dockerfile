# Use a lightweight Python base image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download the model
RUN python -c "from transformers import AutoTokenizer, AutoModel; \
               AutoTokenizer.from_pretrained('intfloat/e5-small-v2'); \
               AutoModel.from_pretrained('intfloat/e5-small-v2')"

# Copy rest of the application
COPY . .

# Set default command (overridden at runtime if needed)
CMD ["python", "chunk.py"]
