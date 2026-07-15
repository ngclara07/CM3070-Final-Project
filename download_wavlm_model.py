# download_wavlm_model.py

from pathlib import Path
from transformers import Wav2Vec2FeatureExtractor, WavLMModel

MODEL_NAME = "microsoft/wavlm-base-plus"
SAVE_DIR = Path("models/wavlm-base-plus")

SAVE_DIR.mkdir(parents=True, exist_ok=True)

print(f"Downloading model: {MODEL_NAME}")

feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(MODEL_NAME)
model = WavLMModel.from_pretrained(MODEL_NAME)

feature_extractor.save_pretrained(SAVE_DIR)
model.save_pretrained(SAVE_DIR)

print(f"Model saved locally to: {SAVE_DIR}")
