FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Native libraries required by OpenCV, librosa/soundfile and audio processing
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libgl1 \
    libglib2.0-0 \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies separately so this layer can be cached
COPY requirements-deploy.txt /app/requirements-deploy.txt

RUN python -m pip install --upgrade pip && \
    python -m pip install --no-cache-dir -r requirements-deploy.txt

# Copy SenseFuzeAI source code and tracked model configuration files
COPY . /app

# --------------------------------------------------------------------------
# Download pretrained model weights that are intentionally excluded from Git.
#
# The configuration/tokenizer files already exist under /app/models.
# Only the large safetensors weights are downloaded here.
# --------------------------------------------------------------------------

RUN python -c "from huggingface_hub import hf_hub_download; hf_hub_download(repo_id='sentence-transformers/all-mpnet-base-v2', filename='model.safetensors', local_dir='/app/models/all-mpnet-base-v2')"

RUN python -c "from huggingface_hub import hf_hub_download; hf_hub_download(repo_id='microsoft/wavlm-base-plus', filename='model.safetensors', local_dir='/app/models/wavlm-base-plus')"

RUN python -c "from huggingface_hub import hf_hub_download; hf_hub_download(repo_id='openai/clip-vit-large-patch14', filename='model.safetensors', local_dir='/app/models/clip-vit-large-patch14')"

# Verify that all three required neural-model weights exist.
RUN test -f /app/models/all-mpnet-base-v2/model.safetensors && \
    test -f /app/models/wavlm-base-plus/model.safetensors && \
    test -f /app/models/clip-vit-large-patch14/model.safetensors

RUN mkdir -p /app/web_app/uploads /app/web_app/output

EXPOSE 8000

CMD ["sh", "-c", "python -m uvicorn web_app.app:app --host 0.0.0.0 --port ${PORT:-8000} --proxy-headers --forwarded-allow-ips='*'"]
