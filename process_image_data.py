# === process_image_data.py ===
# SenseFuzeAI - Image Dataset Processing Pipeline
#
# Processes labelled image folders into:
#   data/processed/processed_image_dataset.csv
#   data/processed/image_dataset_summary.json
#
# The pipeline generates:
#   - BLIP generic caption
#   - behaviour-aware caption
#   - structured behavioural cue evidence
#   - CLIP behavioural prompt scores
#   - CLIP scene prompt scores
#   - optional CLIP face/body prompt scores
#   - MediaPipe face/body/posture metadata
#   - visual fused behaviour scores
#   - train/val/test split
#
# Required synchronised functions from app/models/image_model.py:
#   - infer_visible_behavioural_cues
#   - build_behavioural_cue_caption
#   - build_visual_cue_summary

from __future__ import annotations

import hashlib
import json
import os
import random
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
from sklearn.model_selection import train_test_split

from app.models.image_model import (
    CLASS_LABELS,
    BEHAVIOUR_PROMPTS,
    FACE_PROMPTS,
    BODY_PROMPTS,
    SCENE_PROMPTS,
    load_image,
    detect_face,
    detect_pose_and_body_crop,
    classify_with_prompts,
    generate_caption,
    compute_visual_evidence,
    compute_reliability_score,
    build_quality_flag,
    infer_visible_behavioural_cues,
    build_behavioural_cue_caption,
    build_visual_cue_summary,
)


# ============================================================
# Configuration
# ============================================================

IMAGE_ROOT = Path("data/images")

OUTPUT_DIR = Path("data/processed")
OUTPUT_CSV = OUTPUT_DIR / "processed_image_dataset.csv"
SUMMARY_JSON = OUTPUT_DIR / "image_dataset_summary.json"

PREPROCESS_ROOT = Path("data/images/preprocess")
SAVE_CROPS = True

SUPPORTED_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".webp",
}

RANDOM_SEED = 42

TRAIN_SIZE = 0.70
VAL_SIZE = 0.15
TEST_SIZE = 0.15


# ============================================================
# Utility functions
# ============================================================

def make_safe_crop_filename(
    image_path: Path,
    label: str,
    crop_type: str,
) -> str:
    """
    Create a deterministic crop filename.

    This prevents name collisions when different class folders contain
    images with similar filenames.
    """
    unique_source = f"{label}/{image_path.as_posix()}/{crop_type}"
    digest = hashlib.sha1(unique_source.encode("utf-8")).hexdigest()[:10]

    return f"{image_path.stem}_{crop_type}_{digest}.jpg"


def save_crop_image(
    crop: Optional[Image.Image],
    image_path: Path,
    label: str,
    crop_type: str,
) -> str:
    """
    Save an optional face/body crop.

    Returns an empty string if the crop is unavailable or cannot be saved.
    """
    if crop is None:
        return ""

    output_dir = PREPROCESS_ROOT / label
    output_dir.mkdir(parents=True, exist_ok=True)

    save_path = output_dir / make_safe_crop_filename(
        image_path=image_path,
        label=label,
        crop_type=crop_type,
    )

    try:
        crop.convert("RGB").save(save_path, format="JPEG", quality=95)
        return str(save_path)

    except Exception as error:
        warnings.warn(
            f"Failed to save {crop_type} crop for {image_path}: {error}"
        )
        return ""


def label_scores_to_columns(
    scores: Dict[str, float],
    prefix: str,
) -> Dict[str, float]:
    """
    Convert a class-score mapping into flat CSV columns.

    Example:
        {"focused": 0.7}
    becomes:
        {"behaviour_focused_score": 0.7}
    """
    return {
        f"{prefix}_{label}_score": round(float(scores.get(label, 0.0)), 4)
        for label in CLASS_LABELS
    }


def json_dumps_safe(value) -> str:
    """
    Safely serialise values for CSV storage.
    """
    try:
        return json.dumps(value, ensure_ascii=False)
    except Exception:
        return json.dumps(str(value), ensure_ascii=False)


def compute_label_consistency(
    true_label: str,
    behaviour_pred: Dict,
    scene_pred: Dict,
    face_pred: Optional[Dict],
    body_pred: Optional[Dict],
    fused_pred: Dict,
) -> Dict:
    """
    Compare visual pseudo-cues against the folder label.

    This does not replace the ground-truth folder label. It gives useful
    diagnostic information about whether CLIP/MediaPipe evidence agrees with
    the labelled folder.
    """
    votes = [
        behaviour_pred["best_label"],
        scene_pred["best_label"],
        fused_pred["visual_fused_label"],
    ]

    if face_pred is not None:
        votes.append(face_pred["best_label"])

    if body_pred is not None:
        votes.append(body_pred["best_label"])

    agreement_with_folder = sum(
        1 for vote in votes
        if vote == true_label
    )

    total_votes = len(votes)

    majority_label = max(set(votes), key=votes.count)
    majority_count = votes.count(majority_label)

    agreement_ratio = (
        agreement_with_folder / total_votes
        if total_votes > 0
        else 0.0
    )

    return {
        "visual_votes": ";".join(votes),
        "majority_visual_label": majority_label,
        "majority_vote_count": int(majority_count),
        "agreement_with_folder_label": int(agreement_with_folder),
        "total_votes": int(total_votes),
        "agreement_ratio": round(float(agreement_ratio), 4),
    }


# ============================================================
# Single-image processing
# ============================================================

def process_single_image(
    image_path: Path,
    label: str,
) -> Dict:
    """
    Process one labelled image into a model-ready dataset row.

    Processing stages:
        1. Load image.
        2. Detect face and body/posture.
        3. Run CLIP prompt scoring.
        4. Fuse visual evidence.
        5. Generate BLIP generic caption.
        6. Generate behaviour-aware caption from visual cues.
        7. Save all numeric and textual outputs into one CSV row.
    """
    image = load_image(image_path)
    width, height = image.size

    # --------------------------------------------------------
    # Face/body detection
    # --------------------------------------------------------

    face_info = detect_face(image)
    pose_info = detect_pose_and_body_crop(image)

    face_crop_path = ""
    body_crop_path = ""

    if SAVE_CROPS:
        face_crop_path = save_crop_image(
            crop=face_info.get("face_crop"),
            image_path=image_path,
            label=label,
            crop_type="face",
        )

        body_crop_path = save_crop_image(
            crop=pose_info.get("body_crop"),
            image_path=image_path,
            label=label,
            crop_type="body",
        )

    # --------------------------------------------------------
    # CLIP zero-shot visual cue classification
    # --------------------------------------------------------

    behaviour_pred = classify_with_prompts(
        image=image,
        prompt_dict=BEHAVIOUR_PROMPTS,
    )

    scene_pred = classify_with_prompts(
        image=image,
        prompt_dict=SCENE_PROMPTS,
    )

    face_pred = (
        classify_with_prompts(
            image=face_info["face_crop"],
            prompt_dict=FACE_PROMPTS,
        )
        if face_info.get("face_crop") is not None
        else None
    )

    body_pred = (
        classify_with_prompts(
            image=pose_info["body_crop"],
            prompt_dict=BODY_PROMPTS,
        )
        if pose_info.get("body_crop") is not None
        else None
    )

    # --------------------------------------------------------
    # Visual fusion
    # --------------------------------------------------------

    fused_pred = compute_visual_evidence(
        behaviour_pred=behaviour_pred,
        scene_pred=scene_pred,
        face_pred=face_pred,
        body_pred=body_pred,
    )

    reliability_score = compute_reliability_score(
        fused_pred=fused_pred,
        face_info=face_info,
        pose_info=pose_info,
    )

    quality_flag = build_quality_flag(reliability_score)

    # --------------------------------------------------------
    # Label consistency diagnostics
    # --------------------------------------------------------

    consistency = compute_label_consistency(
        true_label=label,
        behaviour_pred=behaviour_pred,
        scene_pred=scene_pred,
        face_pred=face_pred,
        body_pred=body_pred,
        fused_pred=fused_pred,
    )

    # --------------------------------------------------------
    # Captioning and behavioural cue interpretation
    # --------------------------------------------------------

    generic_caption = generate_caption(image)

    cue_evidence = infer_visible_behavioural_cues(
        final_label=fused_pred["visual_fused_label"],
        generic_caption=generic_caption,
        behaviour_pred=behaviour_pred,
        scene_pred=scene_pred,
        face_pred=face_pred,
        body_pred=body_pred,
        face_info=face_info,
        pose_info=pose_info,
    )

    behaviour_caption = build_behavioural_cue_caption(
        final_label=fused_pred["visual_fused_label"],
        generic_caption=generic_caption,
        cue_evidence=cue_evidence,
        fused_pred=fused_pred,
        reliability_score=reliability_score,
        quality_flag=quality_flag,
    )

    visual_cue_summary = build_visual_cue_summary(
        behaviour_pred=behaviour_pred,
        scene_pred=scene_pred,
        face_pred=face_pred,
        body_pred=body_pred,
        fused_pred=fused_pred,
        face_info=face_info,
        pose_info=pose_info,
        reliability_score=reliability_score,
        quality_flag=quality_flag,
        cue_evidence=cue_evidence,
    )

    # --------------------------------------------------------
    # CSV row construction
    # --------------------------------------------------------

    row = {
        "filepath": str(image_path),
        "filename": image_path.name,
        "label": label,

        "width": int(width),
        "height": int(height),

        # Face metadata
        "face_detected": bool(face_info.get("face_detected", False)),
        "face_confidence": float(face_info.get("face_confidence", 0.0)),
        "face_box": face_info.get("face_box", ""),
        "face_crop_path": face_crop_path,

        # Body/posture metadata
        "body_detected": bool(pose_info.get("body_detected", False)),
        "body_box": pose_info.get("body_box", ""),
        "body_crop_path": body_crop_path,
        "pose_visibility": float(pose_info.get("pose_visibility", 0.0)),
        "shoulder_visibility": float(pose_info.get("shoulder_visibility", 0.0)),
        "posture_cue": pose_info.get("posture_cue", "unknown"),
        "head_position_cue": pose_info.get("head_position_cue", "unknown"),

        # Caption fields
        "generic_caption": generic_caption,
        "behaviour_caption": behaviour_caption,
        "visual_cue_summary": json_dumps_safe(visual_cue_summary),

        # Additional structured cue columns
        "supporting_visual_cues": json_dumps_safe(
            cue_evidence.get("supporting_cues", [])
        ),
        "possible_conflicting_cues": json_dumps_safe(
            cue_evidence.get("possible_conflicting_cues", [])
        ),
        "absent_or_uncertain_cues": json_dumps_safe(
            cue_evidence.get("absent_or_uncertain_cues", [])
        ),

        # Backward-compatible description columns
        "scene_description": generic_caption,
        "behaviour_description": behaviour_caption,

        # CLIP full-image behaviour cue
        "predicted_behaviour_cue": behaviour_pred["best_label"],
        "predicted_behaviour_score": round(
            float(behaviour_pred["best_score"]),
            4,
        ),
        "predicted_behaviour_margin": round(
            float(behaviour_pred["margin"]),
            4,
        ),

        # CLIP scene cue
        "predicted_scene_cue": scene_pred["best_label"],
        "predicted_scene_score": round(
            float(scene_pred["best_score"]),
            4,
        ),
        "predicted_scene_margin": round(
            float(scene_pred["margin"]),
            4,
        ),

        # Optional face cue
        "predicted_face_cue": (
            face_pred["best_label"]
            if face_pred is not None
            else "not_detected"
        ),
        "predicted_face_score": round(
            float(face_pred["best_score"]),
            4,
        ) if face_pred is not None else 0.0,
        "predicted_face_margin": round(
            float(face_pred["margin"]),
            4,
        ) if face_pred is not None else 0.0,

        # Optional body cue
        "predicted_body_cue": (
            body_pred["best_label"]
            if body_pred is not None
            else "not_detected"
        ),
        "predicted_body_score": round(
            float(body_pred["best_score"]),
            4,
        ) if body_pred is not None else 0.0,
        "predicted_body_margin": round(
            float(body_pred["margin"]),
            4,
        ) if body_pred is not None else 0.0,

        # Fused visual prediction
        "visual_fused_label": fused_pred["visual_fused_label"],
        "visual_fused_score": round(
            float(fused_pred["visual_fused_score"]),
            4,
        ),
        "visual_fused_second_label": fused_pred["visual_fused_second_label"],
        "visual_fused_second_score": round(
            float(fused_pred["visual_fused_second_score"]),
            4,
        ),
        "visual_fused_margin": round(
            float(fused_pred["visual_fused_margin"]),
            4,
        ),

        # Label/prediction consistency diagnostics
        "visual_votes": consistency["visual_votes"],
        "majority_visual_label": consistency["majority_visual_label"],
        "majority_vote_count": consistency["majority_vote_count"],
        "agreement_with_folder_label": consistency["agreement_with_folder_label"],
        "total_votes": consistency["total_votes"],
        "agreement_ratio": consistency["agreement_ratio"],

        # Reliability
        "reliability_score": float(reliability_score),
        "quality_flag": quality_flag,

        # Final fused class scores
        "focused_score": round(
            float(fused_pred["visual_fused_scores"].get("focused", 0.0)),
            4,
        ),
        "distracted_score": round(
            float(fused_pred["visual_fused_scores"].get("distracted", 0.0)),
            4,
        ),
        "fatigued_score": round(
            float(fused_pred["visual_fused_scores"].get("fatigued", 0.0)),
            4,
        ),
        "overloaded_score": round(
            float(fused_pred["visual_fused_scores"].get("overloaded", 0.0)),
            4,
        ),
    }

    # Add full prompt-score columns for each visual subcomponent.
    row.update(
        label_scores_to_columns(
            behaviour_pred["scores"],
            "behaviour",
        )
    )

    row.update(
        label_scores_to_columns(
            scene_pred["scores"],
            "scene",
        )
    )

    if face_pred is not None:
        row.update(
            label_scores_to_columns(
                face_pred["scores"],
                "face",
            )
        )
    else:
        row.update(
            {
                f"face_{label_name}_score": 0.0
                for label_name in CLASS_LABELS
            }
        )

    if body_pred is not None:
        row.update(
            label_scores_to_columns(
                body_pred["scores"],
                "body",
            )
        )
    else:
        row.update(
            {
                f"body_{label_name}_score": 0.0
                for label_name in CLASS_LABELS
            }
        )

    return row


# ============================================================
# Data collection
# ============================================================

def collect_image_paths() -> List[Tuple[Path, str]]:
    """
    Collect image files from class-labelled folders.

    Expected structure:
        data/images/focused/*.jpg
        data/images/distracted/*.jpg
        data/images/fatigued/*.jpg
        data/images/overloaded/*.jpg

    The preprocess folder is skipped to avoid re-processing generated crops.
    """
    if not IMAGE_ROOT.exists():
        raise FileNotFoundError(f"Image root does not exist: {IMAGE_ROOT}")

    image_paths: List[Tuple[Path, str]] = []

    preprocess_root_resolved = PREPROCESS_ROOT.resolve()

    for label_dir in IMAGE_ROOT.iterdir():
        if not label_dir.is_dir():
            continue

        try:
            if label_dir.resolve() == preprocess_root_resolved:
                continue
        except Exception:
            pass

        label = label_dir.name.strip().lower()

        if label not in CLASS_LABELS:
            continue

        for image_path in label_dir.rglob("*"):
            if not image_path.is_file():
                continue

            try:
                if preprocess_root_resolved in image_path.resolve().parents:
                    continue
            except Exception:
                pass

            if image_path.suffix.lower() in SUPPORTED_EXTENSIONS:
                image_paths.append((image_path, label))

    return image_paths


# ============================================================
# Split creation
# ============================================================

def add_dataset_split(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add stratified train/validation/test split.

    If stratification fails because the dataset is too small, the script falls
    back to split='unsplit' rather than crashing.
    """
    if df.empty or "label" not in df.columns:
        df = df.copy()
        df["split"] = "unknown"
        return df

    df = df.copy()

    valid_df = df[df["label"].isin(CLASS_LABELS)].copy()
    invalid_df = df[~df.index.isin(valid_df.index)].copy()

    if valid_df.empty:
        df["split"] = "unknown"
        return df

    try:
        train_df, temp_df = train_test_split(
            valid_df,
            train_size=TRAIN_SIZE,
            stratify=valid_df["label"],
            random_state=RANDOM_SEED,
        )

        relative_test_size = TEST_SIZE / (VAL_SIZE + TEST_SIZE)

        val_df, test_df = train_test_split(
            temp_df,
            test_size=relative_test_size,
            stratify=temp_df["label"],
            random_state=RANDOM_SEED,
        )

        train_df["split"] = "train"
        val_df["split"] = "val"
        test_df["split"] = "test"

        if not invalid_df.empty:
            invalid_df["split"] = "invalid"

        output_df = pd.concat(
            [train_df, val_df, test_df, invalid_df],
            axis=0,
        ).sort_index()

        return output_df

    except Exception as error:
        warnings.warn(f"Could not create stratified split: {error}")
        df["split"] = "unsplit"
        return df


# ============================================================
# Summary
# ============================================================

def save_summary(df: pd.DataFrame) -> None:
    """
    Save image processing summary JSON.
    """
    summary = {
        "total_images": int(len(df)),
        "successful_rows": int(
            len(df[df.get("error", pd.Series(index=df.index, dtype=str)).isna()])
            if "error" in df.columns
            else len(df)
        ),
        "error_rows": int(
            df["error"].notna().sum()
            if "error" in df.columns
            else 0
        ),
        "class_distribution": (
            df["label"].value_counts().to_dict()
            if "label" in df.columns
            else {}
        ),
        "split_distribution": (
            df["split"].value_counts().to_dict()
            if "split" in df.columns
            else {}
        ),
        "quality_distribution": (
            df["quality_flag"].value_counts().to_dict()
            if "quality_flag" in df.columns
            else {}
        ),
        "visual_fused_distribution": (
            df["visual_fused_label"].value_counts().to_dict()
            if "visual_fused_label" in df.columns
            else {}
        ),
        "mean_reliability_score": (
            float(df["reliability_score"].mean())
            if "reliability_score" in df.columns and not df.empty
            else None
        ),
        "face_detection_rate": (
            float(df["face_detected"].mean())
            if "face_detected" in df.columns and not df.empty
            else None
        ),
        "body_detection_rate": (
            float(df["body_detected"].mean())
            if "body_detected" in df.columns and not df.empty
            else None
        ),
        "caption_fields": {
            "generic_caption": "BLIP-generated generic image caption",
            "behaviour_caption": (
                "Behaviour-aware cue caption generated from BLIP, CLIP, "
                "MediaPipe, fused visual prediction, and reliability metadata"
            ),
            "visual_cue_summary": (
                "Structured JSON summary of visual behavioural evidence"
            ),
            "supporting_visual_cues": (
                "Visual cues supporting the final predicted behaviour"
            ),
            "possible_conflicting_cues": (
                "Visual cues that may conflict with or weaken the final prediction"
            ),
            "absent_or_uncertain_cues": (
                "Behavioural cues that were not detected or were uncertain"
            ),
        },
        "supported_extensions": sorted(list(SUPPORTED_EXTENSIONS)),
        "save_crops": SAVE_CROPS,
        "preprocess_root": str(PREPROCESS_ROOT),
        "random_seed": RANDOM_SEED,
        "train_size": TRAIN_SIZE,
        "val_size": VAL_SIZE,
        "test_size": TEST_SIZE,
        "methodological_note": (
            "Images are organised by behavioural-state folders. Each image is "
            "processed using CLIP zero-shot prompt scoring, BLIP generic caption "
            "generation, MediaPipe face/body analysis, visual score fusion, and "
            "a dedicated behaviour-aware cue caption layer. The output CSV "
            "contains folder labels, visual pseudo-predictions, reliability "
            "metadata, structured cue evidence, captions, and numeric features "
            "for downstream image model training."
        ),
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    with open(SUMMARY_JSON, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=4)

    print(f"Summary saved to: {SUMMARY_JSON}")


# ============================================================
# Main
# ============================================================

def main() -> None:
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if SAVE_CROPS:
        PREPROCESS_ROOT.mkdir(parents=True, exist_ok=True)

    image_paths = collect_image_paths()

    print("==========================================")
    print("SenseFuzeAI Image Data Processing Pipeline")
    print("==========================================")
    print(f"Image root:       {IMAGE_ROOT}")
    print(f"Output CSV:       {OUTPUT_CSV}")
    print(f"Summary JSON:     {SUMMARY_JSON}")
    print(f"Save crops:       {SAVE_CROPS}")
    print(f"Found images:     {len(image_paths)}")
    print("==========================================")

    rows: List[Dict] = []

    for image_path, label in tqdm(image_paths, desc="Processing images"):
        try:
            rows.append(
                process_single_image(
                    image_path=image_path,
                    label=label,
                )
            )

        except Exception as error:
            rows.append(
                {
                    "filepath": str(image_path),
                    "filename": image_path.name,
                    "label": label,
                    "error": str(error),
                }
            )

    df = pd.DataFrame(rows)
    df = add_dataset_split(df)

    df.to_csv(OUTPUT_CSV, index=False)
    save_summary(df)

    print("\nProcessing complete.")
    print(f"CSV saved to: {OUTPUT_CSV}")
    print(f"Summary saved to: {SUMMARY_JSON}")

    if SAVE_CROPS:
        print(f"Crops saved to: {PREPROCESS_ROOT}")

    if not df.empty:
        print("\nClass distribution:")
        if "label" in df.columns:
            print(df["label"].value_counts())

        if "split" in df.columns:
            print("\nSplit distribution:")
            print(df["split"].value_counts())

        if "quality_flag" in df.columns:
            print("\nQuality distribution:")
            print(df["quality_flag"].value_counts())

        if "visual_fused_label" in df.columns:
            print("\nVisual fused prediction distribution:")
            print(df["visual_fused_label"].value_counts())

        if "reliability_score" in df.columns:
            print("\nMean reliability score:")
            print(round(float(df["reliability_score"].mean()), 4))

        generated_fields = [
            "generic_caption",
            "behaviour_caption",
            "visual_cue_summary",
            "supporting_visual_cues",
            "possible_conflicting_cues",
            "absent_or_uncertain_cues",
        ]

        print("\nBehaviour-aware caption fields:")
        for field in generated_fields:
            status = "present" if field in df.columns else "missing"
            print(f"  {field}: {status}")

        if "error" in df.columns:
            error_count = int(df["error"].notna().sum())
            print(f"\nRows with processing errors: {error_count}")


if __name__ == "__main__":
    main()
