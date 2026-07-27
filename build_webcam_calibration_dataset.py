# build_webcam_calibration_dataset.py

from __future__ import annotations

import csv
import json
import math
import sys
import time
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import CLIPModel, CLIPProcessor


# =============================================================================
# PROJECT PATHS
# =============================================================================

ROOT_DIR = Path(__file__).resolve().parent

CALIBRATION_DIR = ROOT_DIR / "data" / "webcam_calibration"

OUTPUT_DIR = ROOT_DIR / "data" / "processed"

OUTPUT_CSV = (
    OUTPUT_DIR
    / "webcam_calibration_clip_features.csv"
)

OUTPUT_SUMMARY_JSON = (
    OUTPUT_DIR
    / "webcam_calibration_clip_features_summary.json"
)

EXTRACTED_FRAMES_DIR = (
    OUTPUT_DIR
    / "webcam_calibration_frames"
)

CLIP_MODEL_PATH = (
    ROOT_DIR
    / "models"
    / "clip-vit-large-patch14"
)


# =============================================================================
# DATASET CONFIGURATION
# =============================================================================

LABELS = [
    "focused",
    "distracted",
    "fatigued",
    "overloaded",
]

VIDEO_EXTENSIONS = {
    ".mp4",
    ".avi",
    ".mov",
    ".mkv",
    ".webm",
    ".m4v",
}

# How many frames to extract from each video.
#
# With:
#   20 videos/class
#   4 frames/video
#
# You get approximately:
#   80 frames/class
#   320 frames total
#
# This is a useful calibration size without creating excessive near-duplicates.
FRAMES_PER_VIDEO = 4

# Ignore the very beginning/end of a clip when sampling.
#
# Example:
#   0.10 means do not sample from the first 10% or last 10%.
EDGE_MARGIN_RATIO = 0.10

# Minimum valid video duration.
MIN_VIDEO_DURATION_SECONDS = 0.5

# Output JPEG quality for extracted audit/debug frames.
JPEG_QUALITY = 95

# Reduce CPU thread pressure.
torch.set_num_threads(2)


# =============================================================================
# DEVICE
# =============================================================================

def get_device() -> torch.device:
    return torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )


# =============================================================================
# CLIP MODEL
# =============================================================================

def build_clip_model(
    device: torch.device,
):
    """
    Load the exact same CLIP model family used by the live image GUI.
    """

    if not CLIP_MODEL_PATH.exists():
        raise FileNotFoundError(
            f"CLIP model directory not found: {CLIP_MODEL_PATH}"
        )

    print(
        f"Loading CLIP processor from:\n"
        f"  {CLIP_MODEL_PATH}"
    )

    processor = CLIPProcessor.from_pretrained(
        str(CLIP_MODEL_PATH)
    )

    print(
        f"Loading CLIP model from:\n"
        f"  {CLIP_MODEL_PATH}"
    )

    model = CLIPModel.from_pretrained(
        str(CLIP_MODEL_PATH)
    )

    model.to(device)
    model.eval()

    return model, processor


# =============================================================================
# CLIP FEATURE EXTRACTION
# =============================================================================

def extract_clip_embedding(
    image: Image.Image,
    model: CLIPModel,
    processor: CLIPProcessor,
    device: torch.device,
) -> np.ndarray:
    """
    Extract the exact normalized CLIP image representation expected by
    image_live_gui.py.

    Output:
        shape = (768,)
    """

    image = image.convert("RGB")

    inputs = processor(
        images=image,
        return_tensors="pt",
    )

    pixel_values = inputs[
        "pixel_values"
    ].to(device)

    with torch.inference_mode():
        try:
            output = model.get_image_features(
                pixel_values=pixel_values
            )

            if isinstance(
                output,
                torch.Tensor,
            ):
                image_features = output

            elif hasattr(
                output,
                "image_embeds",
            ):
                image_features = (
                    output.image_embeds
                )

            elif hasattr(
                output,
                "pooler_output",
            ):
                image_features = (
                    output.pooler_output
                )

            elif hasattr(
                output,
                "last_hidden_state",
            ):
                image_features = (
                    output.last_hidden_state
                    .mean(dim=1)
                )

            else:
                raise TypeError(
                    "Unsupported CLIP output type: "
                    f"{type(output)}"
                )

        except Exception:
            # Compatibility fallback, matching the live GUI logic.
            output = model.vision_model(
                pixel_values=pixel_values
            )

            if hasattr(
                output,
                "pooler_output",
            ):
                image_features = (
                    output.pooler_output
                )

            elif hasattr(
                output,
                "last_hidden_state",
            ):
                image_features = (
                    output.last_hidden_state
                    .mean(dim=1)
                )

            else:
                raise TypeError(
                    "Unsupported CLIP vision output type: "
                    f"{type(output)}"
                )

    image_features = F.normalize(
        image_features,
        p=2,
        dim=-1,
    )

    embedding = (
        image_features
        .squeeze(0)
        .detach()
        .cpu()
        .numpy()
        .astype(np.float32)
    )

    if embedding.ndim != 1:
        raise ValueError(
            "Expected 1-dimensional CLIP embedding, "
            f"received shape {embedding.shape}"
        )

    if embedding.shape[0] != 768:
        raise ValueError(
            "Expected 768-dimensional CLIP embedding, "
            f"received {embedding.shape[0]}"
        )

    if not np.all(
        np.isfinite(embedding)
    ):
        raise ValueError(
            "CLIP embedding contains NaN or infinity values."
        )

    return embedding


# =============================================================================
# VIDEO HELPERS
# =============================================================================

def find_video_files(
    folder: Path,
) -> list[Path]:
    """
    Return all supported video files recursively.
    """

    video_files = []

    for path in folder.rglob("*"):
        if (
            path.is_file()
            and path.suffix.lower()
            in VIDEO_EXTENSIONS
        ):
            video_files.append(path)

    return sorted(
        video_files
    )


def get_video_metadata(
    video_path: Path,
) -> dict:
    """
    Read basic video metadata using OpenCV.
    """

    capture = cv2.VideoCapture(
        str(video_path)
    )

    if not capture.isOpened():
        raise ValueError(
            f"Could not open video: {video_path}"
        )

    fps = float(
        capture.get(
            cv2.CAP_PROP_FPS
        )
    )

    frame_count = int(
        capture.get(
            cv2.CAP_PROP_FRAME_COUNT
        )
    )

    width = int(
        capture.get(
            cv2.CAP_PROP_FRAME_WIDTH
        )
    )

    height = int(
        capture.get(
            cv2.CAP_PROP_FRAME_HEIGHT
        )
    )

    capture.release()

    if (
        fps <= 0
        or frame_count <= 0
    ):
        duration = 0.0

    else:
        duration = (
            frame_count
            / fps
        )

    return {
        "fps": fps,
        "frame_count": frame_count,
        "duration_seconds": duration,
        "width": width,
        "height": height,
    }


def compute_sample_indices(
    frame_count: int,
    frames_per_video: int,
    edge_margin_ratio: float,
) -> list[int]:
    """
    Select evenly spaced frame indices while avoiding the first and final
    portions of the video.

    Example:
        frames_per_video = 4

        samples might be approximately:
        20%, 40%, 60%, 80%
    """

    if frame_count <= 0:
        return []

    if frames_per_video <= 0:
        return []

    margin = int(
        frame_count
        * edge_margin_ratio
    )

    start = max(
        0,
        margin,
    )

    end = min(
        frame_count - 1,
        frame_count - margin - 1,
    )

    if end <= start:
        start = 0
        end = frame_count - 1

    if end < 0:
        return []

    if frame_count <= frames_per_video:
        return list(
            range(frame_count)
        )

    sampled = np.linspace(
        start,
        end,
        num=frames_per_video,
        dtype=int,
    )

    # Remove accidental duplicates while preserving order.
    unique_indices = []

    seen = set()

    for index in sampled.tolist():
        if index not in seen:
            unique_indices.append(
                int(index)
            )
            seen.add(index)

    return unique_indices


def read_frame_at_index(
    capture: cv2.VideoCapture,
    frame_index: int,
) -> np.ndarray:
    """
    Seek to a frame and return the BGR frame.
    """

    capture.set(
        cv2.CAP_PROP_POS_FRAMES,
        frame_index,
    )

    success, frame = capture.read()

    if (
        not success
        or frame is None
    ):
        raise ValueError(
            f"Could not read frame index {frame_index}"
        )

    return frame


# =============================================================================
# FRAME OUTPUT
# =============================================================================

def save_debug_frame(
    frame_bgr: np.ndarray,
    label: str,
    video_stem: str,
    sample_number: int,
    frame_index: int,
) -> Path:
    """
    Save the exact sampled frame so you can inspect the calibration dataset.
    """

    class_dir = (
        EXTRACTED_FRAMES_DIR
        / label
    )

    class_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    filename = (
        f"{video_stem}"
        f"_sample_{sample_number:02d}"
        f"_frame_{frame_index:06d}.jpg"
    )

    output_path = (
        class_dir
        / filename
    )

    success = cv2.imwrite(
        str(output_path),
        frame_bgr,
        [
            cv2.IMWRITE_JPEG_QUALITY,
            JPEG_QUALITY,
        ],
    )

    if not success:
        raise IOError(
            f"Could not save extracted frame: {output_path}"
        )

    return output_path


# =============================================================================
# SINGLE VIDEO PROCESSING
# =============================================================================

def process_video(
    video_path: Path,
    label: str,
    model: CLIPModel,
    processor: CLIPProcessor,
    device: torch.device,
) -> tuple[list[dict], dict]:
    """
    Extract spaced frames and CLIP features from one video.
    """

    metadata = get_video_metadata(
        video_path
    )

    frame_count = metadata[
        "frame_count"
    ]

    fps = metadata[
        "fps"
    ]

    duration = metadata[
        "duration_seconds"
    ]

    if duration < MIN_VIDEO_DURATION_SECONDS:
        raise ValueError(
            f"Video too short ({duration:.3f} sec): {video_path}"
        )

    sample_indices = compute_sample_indices(
        frame_count=frame_count,
        frames_per_video=FRAMES_PER_VIDEO,
        edge_margin_ratio=EDGE_MARGIN_RATIO,
    )

    if not sample_indices:
        raise ValueError(
            f"No valid frame indices generated for: {video_path}"
        )

    capture = cv2.VideoCapture(
        str(video_path)
    )

    if not capture.isOpened():
        raise ValueError(
            f"Could not open video: {video_path}"
        )

    rows = []

    for sample_number, frame_index in enumerate(
        sample_indices,
        start=1,
    ):
        frame_bgr = read_frame_at_index(
            capture,
            frame_index,
        )

        frame_rgb = cv2.cvtColor(
            frame_bgr,
            cv2.COLOR_BGR2RGB,
        )

        pil_image = Image.fromarray(
            frame_rgb
        )

        embedding = extract_clip_embedding(
            image=pil_image,
            model=model,
            processor=processor,
            device=device,
        )

        saved_frame_path = save_debug_frame(
            frame_bgr=frame_bgr,
            label=label,
            video_stem=video_path.stem,
            sample_number=sample_number,
            frame_index=frame_index,
        )

        timestamp_seconds = (
            frame_index / fps
            if fps > 0
            else 0.0
        )

        row = {
            "label": label,
            "source_video": str(
                video_path.relative_to(
                    ROOT_DIR
                )
            ),
            "source_video_name": (
                video_path.name
            ),
            "saved_frame": str(
                saved_frame_path.relative_to(
                    ROOT_DIR
                )
            ),
            "frame_index": int(
                frame_index
            ),
            "frame_timestamp_seconds": float(
                timestamp_seconds
            ),
            "video_fps": float(
                fps
            ),
            "video_frame_count": int(
                frame_count
            ),
            "video_duration_seconds": float(
                duration
            ),
            "original_width": int(
                metadata["width"]
            ),
            "original_height": int(
                metadata["height"]
            ),
        }

        for index, value in enumerate(
            embedding
        ):
            row[
                f"image_clip_emb_{index}"
            ] = float(
                value
            )

        rows.append(
            row
        )

    capture.release()

    summary = {
        "video": str(video_path),
        "label": label,
        "duration_seconds": float(
            duration
        ),
        "frames_requested": int(
            FRAMES_PER_VIDEO
        ),
        "frames_extracted": int(
            len(rows)
        ),
    }

    return rows, summary


# =============================================================================
# DATASET BUILDING
# =============================================================================

def validate_calibration_structure() -> dict[str, list[Path]]:
    """
    Validate label folders and discover videos.
    """

    if not CALIBRATION_DIR.exists():
        raise FileNotFoundError(
            "Webcam calibration directory does not exist:\n"
            f"{CALIBRATION_DIR}"
        )

    videos_by_label = {}

    for label in LABELS:
        label_dir = (
            CALIBRATION_DIR
            / label
        )

        if not label_dir.exists():
            raise FileNotFoundError(
                f"Missing class directory: {label_dir}"
            )

        videos = find_video_files(
            label_dir
        )

        videos_by_label[
            label
        ] = videos

    return videos_by_label


def print_dataset_inventory(
    videos_by_label: dict[str, list[Path]],
) -> None:
    print()
    print("=" * 80)
    print("WEBCAM CALIBRATION DATASET INVENTORY")
    print("=" * 80)

    total = 0

    for label in LABELS:
        count = len(
            videos_by_label[label]
        )

        total += count

        print(
            f"{label:12s}: "
            f"{count:3d} video(s)"
        )

    print("-" * 80)
    print(
        f"{'TOTAL':12s}: "
        f"{total:3d} video(s)"
    )
    print()


def build_dataset() -> pd.DataFrame:
    videos_by_label = (
        validate_calibration_structure()
    )

    print_dataset_inventory(
        videos_by_label
    )

    total_videos = sum(
        len(videos)
        for videos
        in videos_by_label.values()
    )

    if total_videos == 0:
        raise RuntimeError(
            "No calibration videos were found."
        )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    EXTRACTED_FRAMES_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    device = get_device()

    print(
        f"PyTorch device: {device}"
    )
    print()

    model, processor = build_clip_model(
        device
    )

    all_rows = []
    video_summaries = []

    successful_videos = 0
    failed_videos = 0

    global_video_index = 0

    start_time = time.perf_counter()

    print()
    print("=" * 80)
    print("EXTRACTING WEBCAM CALIBRATION FEATURES")
    print("=" * 80)

    for label in LABELS:
        label_videos = (
            videos_by_label[label]
        )

        print()
        print(
            f"[{label.upper()}] "
            f"{len(label_videos)} video(s)"
        )
        print("-" * 80)

        for video_path in label_videos:
            global_video_index += 1

            print(
                f"[{global_video_index}/{total_videos}] "
                f"{video_path.name}"
            )

            try:
                rows, summary = (
                    process_video(
                        video_path=video_path,
                        label=label,
                        model=model,
                        processor=processor,
                        device=device,
                    )
                )

                all_rows.extend(
                    rows
                )

                video_summaries.append(
                    {
                        **summary,
                        "status": "success",
                    }
                )

                successful_videos += 1

                print(
                    f"    extracted "
                    f"{len(rows)} frame(s)"
                )

            except Exception as exc:
                failed_videos += 1

                video_summaries.append(
                    {
                        "video": str(
                            video_path
                        ),
                        "label": label,
                        "status": "failed",
                        "error": str(
                            exc
                        ),
                    }
                )

                print(
                    f"    ERROR: {exc}"
                )

    elapsed = (
        time.perf_counter()
        - start_time
    )

    if not all_rows:
        raise RuntimeError(
            "No calibration frame features were extracted successfully."
        )

    dataframe = pd.DataFrame(
        all_rows
    )

    # -------------------------------------------------------------------------
    # Validate CLIP columns
    # -------------------------------------------------------------------------

    feature_columns = [
        f"image_clip_emb_{index}"
        for index in range(768)
    ]

    missing_columns = [
        column
        for column in feature_columns
        if column not in dataframe.columns
    ]

    if missing_columns:
        raise RuntimeError(
            "Generated dataset is missing CLIP columns:\n"
            f"{missing_columns[:20]}"
        )

    # -------------------------------------------------------------------------
    # Validate values
    # -------------------------------------------------------------------------

    feature_matrix = dataframe[
        feature_columns
    ].to_numpy(
        dtype=np.float32
    )

    if not np.all(
        np.isfinite(feature_matrix)
    ):
        raise RuntimeError(
            "Generated CLIP feature matrix contains NaN or infinite values."
        )

    # -------------------------------------------------------------------------
    # Save CSV
    # -------------------------------------------------------------------------

    dataframe.to_csv(
        OUTPUT_CSV,
        index=False,
        encoding="utf-8",
    )

    label_counts = (
        dataframe[
            "label"
        ]
        .value_counts()
        .to_dict()
    )

    summary_payload = {
        "generated_at": (
            datetime.now()
            .astimezone()
            .isoformat()
        ),
        "calibration_directory": str(
            CALIBRATION_DIR
        ),
        "output_csv": str(
            OUTPUT_CSV
        ),
        "extracted_frames_directory": str(
            EXTRACTED_FRAMES_DIR
        ),
        "clip_model_path": str(
            CLIP_MODEL_PATH
        ),
        "device": str(
            device
        ),
        "frames_per_video": int(
            FRAMES_PER_VIDEO
        ),
        "edge_margin_ratio": float(
            EDGE_MARGIN_RATIO
        ),
        "total_videos": int(
            total_videos
        ),
        "successful_videos": int(
            successful_videos
        ),
        "failed_videos": int(
            failed_videos
        ),
        "total_extracted_frames": int(
            len(dataframe)
        ),
        "feature_dimension": 768,
        "label_counts": {
            str(key): int(value)
            for key, value
            in label_counts.items()
        },
        "processing_seconds": float(
            elapsed
        ),
        "videos": video_summaries,
    }

    OUTPUT_SUMMARY_JSON.write_text(
        json.dumps(
            summary_payload,
            indent=4,
        ),
        encoding="utf-8",
    )

    # -------------------------------------------------------------------------
    # Console summary
    # -------------------------------------------------------------------------

    print()
    print("=" * 80)
    print("WEBCAM CALIBRATION DATASET BUILD COMPLETE")
    print("=" * 80)

    print(
        f"Videos processed successfully : "
        f"{successful_videos}"
    )

    print(
        f"Videos failed                 : "
        f"{failed_videos}"
    )

    print(
        f"Extracted frame samples       : "
        f"{len(dataframe)}"
    )

    print(
        f"CLIP feature dimension        : "
        f"{len(feature_columns)}"
    )

    print(
        f"Processing time               : "
        f"{elapsed:.2f} sec"
    )

    print()
    print("Samples by behavioural class")
    print("-" * 80)

    for label in LABELS:
        count = int(
            label_counts.get(
                label,
                0,
            )
        )

        print(
            f"{label:12s}: "
            f"{count:4d}"
        )

    print()
    print(
        "Feature CSV:"
    )
    print(
        f"  {OUTPUT_CSV}"
    )

    print()
    print(
        "Extracted audit/debug frames:"
    )
    print(
        f"  {EXTRACTED_FRAMES_DIR}"
    )

    print()
    print(
        "Summary JSON:"
    )
    print(
        f"  {OUTPUT_SUMMARY_JSON}"
    )

    print()
    print(
        "NEXT STEP:"
    )
    print(
        "  Run the webcam-calibrated image retraining script "
        "using this CSV."
    )

    print("=" * 80)

    return dataframe


# =============================================================================
# MAIN
# =============================================================================

def main() -> int:
    print()
    print("=" * 80)
    print("SenseFuzeAI Webcam Calibration Dataset Builder")
    print("=" * 80)

    print(
        f"Project root:\n"
        f"  {ROOT_DIR}"
    )

    print(
        f"\nCalibration videos:\n"
        f"  {CALIBRATION_DIR}"
    )

    print(
        f"\nFrames per video:\n"
        f"  {FRAMES_PER_VIDEO}"
    )

    print(
        f"\nExpected CLIP dimension:\n"
        f"  768"
    )

    try:
        dataframe = build_dataset()

        print()
        print(
            f"Final dataframe shape: "
            f"{dataframe.shape}"
        )

        return 0

    except KeyboardInterrupt:
        print()
        print(
            "Processing interrupted by user."
        )

        return 130

    except Exception as exc:
        print()
        print("=" * 80)
        print("DATASET BUILD FAILED")
        print("=" * 80)
        print(
            str(exc)
        )

        return 1


if __name__ == "__main__":
    raise SystemExit(
        main()
    )
