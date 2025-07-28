# Use AMD64 Python slim image
FROM --platform=linux/amd64 python:3.10-slim

# Prevent Python from writing .pyc files and buffer stdout/stderr
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Set working dir
WORKDIR /app

# Install system deps for PyMuPDF
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
         build-essential libmupdf-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy and install Python dependencies
COPY requirement.txt .
RUN pip install --no-cache-dir -r requirement.txt

# Copy your parsing code into the container
COPY main.py .

# By default, process everything under /app/input and emit JSONs to /app/output
# Assumes your script’s main() uses argparse as in your code
ENTRYPOINT ["python", "main.py", "/app/input", "-o", "/app/output"]
