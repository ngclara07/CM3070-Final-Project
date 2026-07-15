# === download_image_model.py ===

from transformers import (
    CLIPModel,
    CLIPProcessor,
    BlipProcessor,
    BlipForConditionalGeneration,
)


# ============================================================
# MODEL CONFIGURATION
# ============================================================

CLIP_MODEL_NAME = "openai/clip-vit-large-patch14"
BLIP_MODEL_NAME = "Salesforce/blip-image-captioning-large"

CLIP_SAVE_DIR = "models/clip-vit-large-patch14"
BLIP_SAVE_DIR = "models/blip-image-captioning-large"


# ============================================================
# DOWNLOAD CLIP
# ============================================================

print("\nDownloading CLIP model...")

clip_processor = CLIPProcessor.from_pretrained(CLIP_MODEL_NAME)
clip_model = CLIPModel.from_pretrained(CLIP_MODEL_NAME)

clip_processor.save_pretrained(CLIP_SAVE_DIR)
clip_model.save_pretrained(CLIP_SAVE_DIR)

print(f"CLIP model saved to: {CLIP_SAVE_DIR}")


# ============================================================
# DOWNLOAD BLIP
# ============================================================

print("\nDownloading BLIP image captioning model...")

blip_processor = BlipProcessor.from_pretrained(BLIP_MODEL_NAME)
blip_model = BlipForConditionalGeneration.from_pretrained(BLIP_MODEL_NAME)

blip_processor.save_pretrained(BLIP_SAVE_DIR)
blip_model.save_pretrained(BLIP_SAVE_DIR)

print(f"BLIP model saved to: {BLIP_SAVE_DIR}")


# ============================================================
# DONE
# ============================================================

print("\n========================================")
print("Image models downloaded successfully.")
print("========================================")
