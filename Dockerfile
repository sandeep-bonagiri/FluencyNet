# Use an official Python runtime as a parent image
FROM python:3.10-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV DOCKER_CONTAINER=true

# Install system dependencies
# espeak-ng: required for phonemizer (used by Kokoro/TTS)
# libportaudio2: required for sounddevice
# ffmpeg: useful for audio processing
# curl: for utilities
RUN apt-get update && apt-get install -y \
    espeak-ng \
    libportaudio2 \
    ffmpeg \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements first to leverage Docker cache
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Download Kokoro TTS models during build
RUN python -c "import urllib.request; \
    print('Downloading Kokoro models...'); \
    urllib.request.urlretrieve('https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/kokoro-v1.0.onnx', 'kokoro-v1.0.onnx'); \
    urllib.request.urlretrieve('https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/voices-v1.0.bin', 'voices-v1.0.bin'); \
    print('Download complete.')"

# Copy the rest of the application code
COPY . .

# Expose the port
EXPOSE 8000

# Run the application
CMD ["python", "main.py"]