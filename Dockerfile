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

# Copy SenseFuzeAI source code and local model artifacts
COPY . /app

RUN mkdir -p /app/web_app/uploads /app/web_app/output

EXPOSE 8000

CMD ["sh", "-c", "python -m uvicorn web_app.app:app --host 0.0.0.0 --port ${PORT:-8000} --proxy-headers --forwarded-allow-ips='*'"]
