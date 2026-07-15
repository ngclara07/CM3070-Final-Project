# extract_image_features.py
# CLIP-based image feature extraction pipeline

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor


DATA_PATH = Path("data/processed/master_sessions_clean_scaled.csv")
OUTPUT_PATH = Path("data/processed/image_features.csv")

SESSION_COL = "session_id"
LABEL_COL = "label"
IMAGE_PATH_COL = "image_path"

# Local CLIP model
MODEL_NAME = "models/clip-vit-large-patch14"


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_model(device: torch.device):
    processor = CLIPProcessor.from_pretrained(MODEL_NAME)
    model = CLIPModel.from_pretrained(MODEL_NAME)
    model.to(device)
    model.eval()
    return model, processor


def extract_image_embedding(
    image_path: Path,
    model: CLIPModel,
    processor: CLIPProcessor,
    device: torch.device,
) -> np.ndarray:
    image = Image.open(image_path).convert("RGB")

    inputs = processor(
        images=image,
        return_tensors="pt",
    )

    pixel_values = inputs["pixel_values"].to(device)

    with torch.no_grad():
        output = model.get_image_features(pixel_values=pixel_values)

        if isinstance(output, torch.Tensor):
            image_features = output
        elif hasattr(output, "pooler_output"):
            image_features = output.pooler_output
        else:
            raise TypeError(f"Unexpected CLIP output type: {type(output)}")

    image_features = image_features / image_features.norm(
        p=2,
        dim=-1,
        keepdim=True,
    )

    return image_features.squeeze(0).cpu().numpy()


def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)

    required_cols = {SESSION_COL, LABEL_COL, IMAGE_PATH_COL}
    missing_cols = required_cols - set(df.columns)

    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    device = get_device()
    print(f"Using device: {device}")
    print(f"Using CLIP model: {MODEL_NAME}")

    model, processor = build_model(device)

    rows = []

    for index, row in df.iterrows():
        image_path = Path(row[IMAGE_PATH_COL])

        if not image_path.exists():
            raise FileNotFoundError(f"Missing image file: {image_path}")

        embedding = extract_image_embedding(
            image_path=image_path,
            model=model,
            processor=processor,
            device=device,
        )

        feature_values = {
            f"image_clip_emb_{i}": float(value)
            for i, value in enumerate(embedding)
        }

        rows.append({
            SESSION_COL: row[SESSION_COL],
            LABEL_COL: row[LABEL_COL],
            **feature_values,
        })

        if (index + 1) % 25 == 0:
            print(f"Processed {index + 1}/{len(df)} images")

    output_df = pd.DataFrame(rows)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    output_df.to_csv(OUTPUT_PATH, index=False)

    print("\nCLIP image feature extraction complete.")
    print(f"Samples: {len(output_df)}")
    print(f"CLIP image embedding features: {output_df.shape[1] - 2}")
    print(f"Saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
