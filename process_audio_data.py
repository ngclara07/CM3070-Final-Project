# === process_audio_data.py ===

from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Dict, List, Tuple

import librosa
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import train_test_split


# ============================================================
# CONFIGURATION
# ============================================================

AUDIO_ROOT = Path("data/audios")
OUTPUT_DIR = Path("data/processed")
OUTPUT_CSV = OUTPUT_DIR / "processed_audio_dataset.csv"
SUMMARY_JSON = OUTPUT_DIR / "audio_dataset_summary.json"

CLASS_LABELS = {
    "focused",
    "distracted",
    "fatigued",
    "overloaded",
}

SUPPORTED_EXTENSIONS = {
    ".wav", ".mp3", ".m4a", ".flac", ".ogg", ".aac"
}

TARGET_SAMPLE_RATE = 16000
MAX_DURATION_SECONDS = 30.0

RANDOM_SEED = 42
TRAIN_SIZE = 0.70
VAL_SIZE = 0.15
TEST_SIZE = 0.15


# ============================================================
# AUDIO LOADING
# ============================================================

def load_audio(audio_path: Path) -> Tuple[np.ndarray, int]:
    y, sr = librosa.load(
        audio_path,
        sr=TARGET_SAMPLE_RATE,
        mono=True,
        duration=MAX_DURATION_SECONDS
    )

    if len(y) == 0:
        raise ValueError("Empty audio signal")

    return y, sr


# ============================================================
# FEATURE EXTRACTION
# ============================================================

def safe_mean(x: np.ndarray) -> float:
    return float(np.nanmean(x)) if x.size > 0 else 0.0


def safe_std(x: np.ndarray) -> float:
    return float(np.nanstd(x)) if x.size > 0 else 0.0


def extract_audio_features(audio_path: Path) -> Dict[str, float]:
    y, sr = load_audio(audio_path)

    duration = librosa.get_duration(y=y, sr=sr)

    rms = librosa.feature.rms(y=y)[0]
    zcr = librosa.feature.zero_crossing_rate(y=y)[0]

    spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]
    spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
    spectral_flatness = librosa.feature.spectral_flatness(y=y)[0]

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)

    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    contrast = librosa.feature.spectral_contrast(y=y, sr=sr)

    tempo = 0.0
    try:
        tempo_value, _ = librosa.beat.beat_track(y=y, sr=sr)
        tempo = float(np.asarray(tempo_value).item())
    except Exception:
        tempo = 0.0

    features = {
        "duration_seconds": float(duration),
        "sample_rate": int(sr),

        "rms_mean": safe_mean(rms),
        "rms_std": safe_std(rms),

        "zero_crossing_rate_mean": safe_mean(zcr),
        "zero_crossing_rate_std": safe_std(zcr),

        "spectral_centroid_mean": safe_mean(spectral_centroid),
        "spectral_centroid_std": safe_std(spectral_centroid),

        "spectral_bandwidth_mean": safe_mean(spectral_bandwidth),
        "spectral_bandwidth_std": safe_std(spectral_bandwidth),

        "spectral_rolloff_mean": safe_mean(spectral_rolloff),
        "spectral_rolloff_std": safe_std(spectral_rolloff),

        "spectral_flatness_mean": safe_mean(spectral_flatness),
        "spectral_flatness_std": safe_std(spectral_flatness),

        "tempo": tempo,

        "chroma_mean": safe_mean(chroma),
        "chroma_std": safe_std(chroma),

        "spectral_contrast_mean": safe_mean(contrast),
        "spectral_contrast_std": safe_std(contrast),
    }

    for i in range(13):
        features[f"mfcc_{i + 1}_mean"] = safe_mean(mfcc[i])
        features[f"mfcc_{i + 1}_std"] = safe_std(mfcc[i])

    return features


# ============================================================
# DATASET COLLECTION
# ============================================================

def collect_audio_paths() -> List[Tuple[Path, str]]:
    if not AUDIO_ROOT.exists():
        raise FileNotFoundError(f"Audio root folder not found: {AUDIO_ROOT}")

    audio_paths = []

    for label_dir in AUDIO_ROOT.iterdir():
        if not label_dir.is_dir():
            continue

        label = label_dir.name.lower()

        if label not in CLASS_LABELS:
            continue

        for audio_path in label_dir.rglob("*"):
            if audio_path.suffix.lower() in SUPPORTED_EXTENSIONS:
                audio_paths.append((audio_path, label))

    return audio_paths


# ============================================================
# SPLIT CREATION
# ============================================================

def add_dataset_split(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        df["split"] = "unknown"
        return df

    try:
        train_df, temp_df = train_test_split(
            df,
            train_size=TRAIN_SIZE,
            stratify=df["label"],
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

        return pd.concat([train_df, val_df, test_df]).sort_index()

    except Exception as error:
        warnings.warn(f"Could not create stratified split: {error}")
        df["split"] = "unsplit"
        return df


# ============================================================
# PROCESS SINGLE AUDIO
# ============================================================

def process_single_audio(audio_path: Path, label: str) -> Dict:
    features = extract_audio_features(audio_path)

    return {
        "filepath": str(audio_path),
        "filename": audio_path.name,
        "label": label,
        "file_extension": audio_path.suffix.lower(),
        **features,
    }


# ============================================================
# SUMMARY
# ============================================================

def save_summary(df: pd.DataFrame) -> None:
    summary = {
        "total_audio_files": int(len(df)),
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
        "mean_duration_seconds": (
            float(df["duration_seconds"].mean())
            if "duration_seconds" in df.columns and not df.empty
            else None
        ),
        "supported_extensions": sorted(list(SUPPORTED_EXTENSIONS)),
        "target_sample_rate": TARGET_SAMPLE_RATE,
        "max_duration_seconds": MAX_DURATION_SECONDS,
        "methodological_note": (
            "Audio files are organised by folder labels. Extracted features include "
            "duration, energy, zero-crossing rate, spectral descriptors, MFCCs, "
            "chroma, spectral contrast, and tempo. Folder names are treated as labels."
        ),
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    with open(SUMMARY_JSON, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=4)

    print(f"Summary saved to: {SUMMARY_JSON}")


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    audio_paths = collect_audio_paths()

    print("==========================================")
    print("Audio data processing pipeline")
    print("==========================================")
    print(f"Audio root: {AUDIO_ROOT}")
    print(f"Found audio files: {len(audio_paths)}")

    rows = []

    for audio_path, label in tqdm(audio_paths, desc="Processing audio files"):
        try:
            row = process_single_audio(audio_path, label)
            rows.append(row)

        except Exception as error:
            rows.append({
                "filepath": str(audio_path),
                "filename": audio_path.name,
                "label": label,
                "file_extension": audio_path.suffix.lower(),
                "error": str(error),
            })

    df = pd.DataFrame(rows)
    df = add_dataset_split(df)

    df.to_csv(OUTPUT_CSV, index=False)
    save_summary(df)

    print("\nProcessing complete.")
    print(f"CSV saved to: {OUTPUT_CSV}")
    print(f"Summary saved to: {SUMMARY_JSON}")

    if not df.empty:
        print("\nClass distribution:")
        print(df["label"].value_counts())

        if "split" in df.columns:
            print("\nSplit distribution:")
            print(df["split"].value_counts())


if __name__ == "__main__":
    main()
