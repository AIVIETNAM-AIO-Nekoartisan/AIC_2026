# Backend API (FastAPI + các model GPU). Cần NVIDIA Container Toolkit khi chạy.
FROM pytorch/pytorch:2.5.1-cuda12.1-cudnn9-runtime

WORKDIR /app

# opencv cần libgl; librosa/whisper cần ffmpeg
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 libglib2.0-0 ffmpeg \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV HF_HOME=/app/temp_cache
EXPOSE 8000
CMD ["python", "-m", "online.backend.api_server"]
