# download_mpnet_model.py

from pathlib import Path
from sentence_transformers import SentenceTransformer

MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"
SAVE_DIR = Path("models/all-mpnet-base-v2")

SAVE_DIR.mkdir(parents=True, exist_ok=True)

print(f"Downloading model: {MODEL_NAME}")

model = SentenceTransformer(MODEL_NAME)

model.save(str(SAVE_DIR))

print(f"Model saved to: {SAVE_DIR}")
