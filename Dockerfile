# Dockerfile (CPU-friendly)
FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Install system deps required by opencv and ffmpeg
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    ffmpeg \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY requirements.txt /app/requirements.txt
RUN pip install --upgrade pip setuptools wheel
RUN pip install -r /app/requirements.txt

# Create models and outputs directories (optional)
RUN mkdir -p /app/models /app/outputs

# Copy application code
COPY main.py /app/main.py

# If you want to embed model weights as part of the image, uncomment and copy them:
# COPY models/*.pt /app/models/

EXPOSE 7860

# Use a simple command to run uvicorn on port 7860 (required by HF Spaces Docker)
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860", "--workers", "1"]
