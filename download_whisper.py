# download_whisper.py

from pathlib import Path
from transformers import WhisperProcessor, WhisperForConditionalGeneration

MODEL_NAME = "openai/whisper-large-v3"
SAVE_DIR = Path("models/whisper-large-v3")

SAVE_DIR.mkdir(parents=True, exist_ok=True)

print(f"Downloading {MODEL_NAME} ...")

processor = WhisperProcessor.from_pretrained(MODEL_NAME)

model = WhisperForConditionalGeneration.from_pretrained(
    MODEL_NAME
)

processor.save_pretrained(SAVE_DIR)
model.save_pretrained(SAVE_DIR)

print(f"Saved to: {SAVE_DIR}")
