FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# --------------------------------------------------------------------------
# System dependencies
# Required by OpenCV, librosa, SoundFile, FFmpeg and audio processing.
# --------------------------------------------------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libgl1 \
    libglib2.0-0 \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# --------------------------------------------------------------------------
# Python dependencies
# Kept in a separate Docker layer so Railway can cache the installation.
# --------------------------------------------------------------------------
COPY requirements-deploy.txt /app/requirements-deploy.txt

RUN python -m pip install --upgrade pip && \
    python -m pip install --no-cache-dir -r /app/requirements-deploy.txt

# --------------------------------------------------------------------------
# SenseFuzeAI application
# Copies source code, trained pipelines, model configurations and tokenizers.
#
# Large pretrained neural-network weights (*.safetensors) remain excluded
# from Git and are downloaded separately below.
# --------------------------------------------------------------------------
COPY . /app

# --------------------------------------------------------------------------
# Pretrained neural model weights
# --------------------------------------------------------------------------

# MPNet text encoder:
# sentence-transformers/all-mpnet-base-v2
RUN python -c "from huggingface_hub import hf_hub_download; hf_hub_download(repo_id='sentence-transformers/all-mpnet-base-v2', filename='model.safetensors', local_dir='/app/models/all-mpnet-base-v2')"

# WavLM audio encoder:
# microsoft/wavlm-base-plus
#
# IMPORTANT:
# The upstream Hugging Face repository provides pytorch_model.bin rather
# than model.safetensors.
RUN python -c "from huggingface_hub import hf_hub_download; hf_hub_download(repo_id='microsoft/wavlm-base-plus', filename='pytorch_model.bin', local_dir='/app/models/wavlm-base-plus')"

# CLIP visual encoder:
# openai/clip-vit-large-patch14
RUN python -c "from huggingface_hub import hf_hub_download; hf_hub_download(repo_id='openai/clip-vit-large-patch14', filename='model.safetensors', local_dir='/app/models/clip-vit-large-patch14')"

# --------------------------------------------------------------------------
# Build-time model verification
#
# Fail the Docker build immediately if any required neural-model weight
# was not downloaded successfully.
# --------------------------------------------------------------------------
RUN test -f /app/models/all-mpnet-base-v2/model.safetensors && \
    test -f /app/models/wavlm-base-plus/pytorch_model.bin && \
    test -f /app/models/clip-vit-large-patch14/model.safetensors

# --------------------------------------------------------------------------
# Runtime directories
# --------------------------------------------------------------------------
RUN mkdir -p /app/web_app/uploads /app/web_app/output

# Railway supplies PORT at runtime.
# 8000 remains the local/default fallback.
EXPOSE 8000

# --------------------------------------------------------------------------
# Start FastAPI/Uvicorn
#
# Proxy headers are required because Railway terminates HTTPS at its reverse
# proxy. Without these flags, FastAPI/Starlette can generate http:// URLs,
# causing browsers to block CSS/JavaScript as mixed content.
# --------------------------------------------------------------------------
CMD ["sh", "-c", "python -m uvicorn web_app.app:app --host 0.0.0.0 --port ${PORT:-8000} --proxy-headers --forwarded-allow-ips='*'"]
