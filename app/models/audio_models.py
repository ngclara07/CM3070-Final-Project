# === app/models/audio_models.py ===
# SenseFuzeAI - Audio Model Runtime Pipeline
#
# Supports:
#   - local YAMNet loading from models/yamnet/
#   - TensorFlow Hub fallback loading
#   - trained librosa-feature audio model
#   - YAMNet acoustic-context labels
#   - heuristic audio behaviour scoring
#   - fused audio behaviour scores
#
# Local YAMNet directory expected:
#   models/yamnet/
#       saved_model.pb
#       variables/
#       assets/
#
# Recommended setup:
#   python download_audio_model.py

from __future__ import annotations

import csv
import json
import os
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import joblib
import librosa
import numpy as np
import pandas as pd
import tensorflow_hub as hub


# ============================================================
# Paths and configuration
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[2]

MODEL_DIR = BASE_DIR / "model_artifacts"
MODEL_PATH = MODEL_DIR / "audio_model.joblib"
META_PATH = MODEL_DIR / "audio_model_meta.json"

LOCAL_YAMNET_DIR = BASE_DIR / "models" / "yamnet"
YAMNET_TFHUB_URL = "https://tfhub.dev/google/yamnet/1"

TARGET_SAMPLE_RATE = 16000
MAX_DURATION_SECONDS = 30.0

BEHAVIOUR_LABELS = [
    "focused",
    "distracted",
    "fatigued",
    "overloaded",
]


# ============================================================
# Runtime caches
# ============================================================

_YAMNET = None
_CLASS_NAMES: Optional[List[str]] = None
_YAMNET_SOURCE = "not_loaded"

_AUDIO_MODEL = None
_AUDIO_META: Optional[Dict[str, Any]] = None


# ============================================================
# General utilities
# ============================================================

def safe_mean(x: np.ndarray) -> float:
    """
    Return nan-safe mean.
    """
    return float(np.nanmean(x)) if x.size > 0 else 0.0


def safe_std(x: np.ndarray) -> float:
    """
    Return nan-safe standard deviation.
    """
    return float(np.nanstd(x)) if x.size > 0 else 0.0


def normalise_scores(scores: Dict[str, float] | None) -> Dict[str, float]:
    """
    Normalise behaviour scores so that all expected labels exist and sum to 1.
    """
    if not scores:
        return {
            label: 1.0 / len(BEHAVIOUR_LABELS)
            for label in BEHAVIOUR_LABELS
        }

    clean = {
        label: max(float(scores.get(label, 0.0)), 0.0)
        for label in BEHAVIOUR_LABELS
    }

    total = sum(clean.values())

    if total <= 0:
        return {
            label: 1.0 / len(BEHAVIOUR_LABELS)
            for label in BEHAVIOUR_LABELS
        }

    return {
        label: value / total
        for label, value in clean.items()
    }


def is_valid_saved_model_dir(path: Path) -> bool:
    """
    Check whether a directory looks like a valid TensorFlow SavedModel.
    """
    if not path.exists() or not path.is_dir():
        return False

    return (
        (path / "saved_model.pb").exists()
        or (path / "saved_model.pbtxt").exists()
    )


def get_yamnet_source() -> str:
    """
    Return where YAMNet was loaded from.
    """
    return _YAMNET_SOURCE


def load_audio(path: str) -> Tuple[np.ndarray, int]:
    """
    Load audio as mono waveform at TARGET_SAMPLE_RATE.
    """
    y, sr = librosa.load(
        path,
        sr=TARGET_SAMPLE_RATE,
        mono=True,
        duration=MAX_DURATION_SECONDS,
    )

    if len(y) == 0:
        raise ValueError(f"Empty audio signal: {path}")

    return y, sr


# ============================================================
# Feature extraction
# ============================================================

def extract_audio_features(path: str) -> Dict[str, float]:
    """
    Extract deterministic librosa features used by the trained audio model.
    """
    y, sr = load_audio(path)

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
# Trained audio model loading
# ============================================================

def load_audio_artifacts() -> Tuple[Any, Dict[str, Any]]:
    """
    Load trained audio model and metadata.
    """
    global _AUDIO_MODEL
    global _AUDIO_META

    if _AUDIO_MODEL is not None and _AUDIO_META is not None:
        return _AUDIO_MODEL, _AUDIO_META

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Audio model not found: {MODEL_PATH}. "
            "Run train_audio_model.py first."
        )

    if not META_PATH.exists():
        raise FileNotFoundError(
            f"Audio metadata not found: {META_PATH}. "
            "Run train_audio_model.py first."
        )

    _AUDIO_MODEL = joblib.load(MODEL_PATH)

    with open(META_PATH, "r", encoding="utf-8") as f:
        _AUDIO_META = json.load(f)

    return _AUDIO_MODEL, _AUDIO_META


def predict_audio_model_scores(
    features: Dict[str, float],
) -> Tuple[str, Dict[str, float]]:
    """
    Predict behaviour using the trained audio classifier.
    """
    model, meta = load_audio_artifacts()

    feature_cols = meta.get("features", [])
    classes = meta.get("classes", BEHAVIOUR_LABELS)

    X = pd.DataFrame([features])
    X = X.reindex(columns=feature_cols, fill_value=0.0)

    prediction = str(model.predict(X)[0])

    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X)[0]

        if hasattr(model, "named_steps") and "model" in model.named_steps:
            model_classes = model.named_steps["model"].classes_
        else:
            model_classes = getattr(model, "classes_", classes)

        scores = {
            str(label): float(prob)
            for label, prob in zip(model_classes, probs)
        }
    else:
        scores = {label: 0.0 for label in classes}
        scores[prediction] = 1.0

    return prediction, normalise_scores(scores)


# ============================================================
# YAMNet loading
# ============================================================

def load_yamnet_from_local() -> Any:
    """
    Load YAMNet from local models/yamnet directory.
    """
    if not is_valid_saved_model_dir(LOCAL_YAMNET_DIR):
        raise FileNotFoundError(
            "Local YAMNet model directory is missing or invalid.\n"
            f"Expected valid SavedModel at: {LOCAL_YAMNET_DIR}\n"
            "Run: python download_audio_model.py"
        )

    return hub.load(str(LOCAL_YAMNET_DIR))


def load_yamnet_from_tfhub() -> Any:
    """
    Load YAMNet from TensorFlow Hub online.
    """
    return hub.load(YAMNET_TFHUB_URL)


def load_yamnet_model() -> Tuple[Any, str]:
    """
    Load YAMNet using local-first strategy.

    Priority:
        1. Local project model: models/yamnet/
        2. Online TensorFlow Hub fallback
    """
    try:
        model = load_yamnet_from_local()
        return model, "local_models_yamnet"

    except Exception as local_error:
        warnings.warn(
            "Could not load local YAMNet model. "
            "Falling back to TensorFlow Hub online loading.\n"
            f"Local path: {LOCAL_YAMNET_DIR}\n"
            f"Local error: {local_error}"
        )

    try:
        model = load_yamnet_from_tfhub()
        return model, "tensorflow_hub_online"

    except Exception as online_error:
        raise RuntimeError(
            "Failed to load YAMNet from both local model directory and "
            "TensorFlow Hub online fallback.\n\n"
            f"Local model path: {LOCAL_YAMNET_DIR}\n"
            f"TensorFlow Hub URL: {YAMNET_TFHUB_URL}\n\n"
            "Recommended fix:\n"
            "  1. Run: python download_audio_model.py --force\n"
            "  2. Confirm models/yamnet/saved_model.pb exists\n"
            "  3. Re-run the audio test\n\n"
            f"Online fallback error: {online_error}"
        )


def load_class_names_from_yamnet(model: Any) -> List[str]:
    """
    Load class names from YAMNet class_map_path().
    """
    try:
        class_map_path = model.class_map_path().numpy().decode("utf-8")

        with open(class_map_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            class_names = [row["display_name"] for row in reader]

        if not class_names:
            raise ValueError("YAMNet class map is empty.")

        return class_names

    except Exception as error:
        raise RuntimeError(
            "Failed to load YAMNet class names from class_map_path(). "
            "The YAMNet model may be incomplete or incompatible."
        ) from error


def get_yamnet() -> Tuple[Any, List[str]]:
    """
    Return cached YAMNet model and class names.
    """
    global _YAMNET
    global _CLASS_NAMES
    global _YAMNET_SOURCE

    if _YAMNET is not None and _CLASS_NAMES is not None:
        return _YAMNET, _CLASS_NAMES

    _YAMNET, _YAMNET_SOURCE = load_yamnet_model()
    _CLASS_NAMES = load_class_names_from_yamnet(_YAMNET)

    return _YAMNET, _CLASS_NAMES


# ============================================================
# YAMNet inference
# ============================================================

def yamnet_audio_labels(path: str, top_k: int = 5) -> List[Tuple[str, float]]:
    """
    Run YAMNet on an audio file and return top acoustic labels.
    """
    model, class_names = get_yamnet()

    wav, _ = librosa.load(
        path,
        sr=TARGET_SAMPLE_RATE,
        mono=True,
        duration=MAX_DURATION_SECONDS,
    )

    if len(wav) == 0:
        return []

    wav = wav.astype(np.float32)

    max_abs = float(np.max(np.abs(wav)))
    if max_abs > 0:
        wav = wav / max_abs

    scores, _, _ = model(wav)
    scores_np = scores.numpy()

    if scores_np.size == 0:
        return []

    mean_scores = scores_np.mean(axis=0)
    top_idx = np.argsort(mean_scores)[::-1][:top_k]

    return [
        (class_names[i], float(mean_scores[i]))
        for i in top_idx
        if i < len(class_names)
    ]


# ============================================================
# Heuristic audio behaviour scoring
# ============================================================

def yamnet_to_behaviour_scores(
    yamnet_labels: Optional[List[Tuple[str, float]]] = None,
) -> Dict[str, float]:
    """
    Map YAMNet acoustic labels to behaviour-state scores.
    """
    yamnet_labels = yamnet_labels or []

    scores = {
        "focused": 0.25,
        "distracted": 0.25,
        "fatigued": 0.25,
        "overloaded": 0.25,
    }

    for label, confidence in yamnet_labels:
        text = str(label).lower()
        weight = float(confidence)

        if any(
            term in text
            for term in [
                "silence",
                "inside",
                "room",
                "quiet",
                "ambient",
                "white noise",
            ]
        ):
            scores["focused"] += 0.50 * weight

        if any(
            term in text
            for term in [
                "speech",
                "conversation",
                "crowd",
                "traffic",
                "vehicle",
                "phone",
                "ringtone",
                "music",
                "dog",
                "bark",
                "television",
                "cacophony",
            ]
        ):
            scores["distracted"] += 0.80 * weight

        if any(
            term in text
            for term in [
                "breathing",
                "snoring",
                "snore",
                "yawn",
                "sigh",
                "sleep",
            ]
        ):
            scores["fatigued"] += 0.90 * weight

        if any(
            term in text
            for term in [
                "alarm",
                "siren",
                "shout",
                "scream",
                "bang",
                "crash",
                "slam",
                "explosion",
                "construction",
                "tools",
                "power tool",
                "machinery",
                "typing",
                "keyboard",
            ]
        ):
            scores["overloaded"] += 0.90 * weight

    return normalise_scores(scores)


def heuristic_audio_scores(
    features: Dict[str, float],
    yamnet_labels: Optional[List[Tuple[str, float]]] = None,
) -> Dict[str, float]:
    """
    Combine YAMNet semantic labels with basic acoustic feature heuristics.
    """
    yamnet_scores = yamnet_to_behaviour_scores(yamnet_labels)

    rms = float(features.get("rms_mean", 0.0))
    zcr = float(features.get("zero_crossing_rate_mean", 0.0))
    centroid = float(features.get("spectral_centroid_mean", 0.0))
    flatness = float(features.get("spectral_flatness_mean", 0.0))

    scores = dict(yamnet_scores)

    # Quiet, stable, low-energy context.
    if rms < 0.025 and centroid < 2200:
        scores["focused"] += 0.20

    # High-frequency/noisy or busy context.
    if centroid > 2500 or zcr > 0.12:
        scores["distracted"] += 0.15

    # Low energy and low spectral activity can indicate low arousal context.
    if rms < 0.04 and centroid < 1800:
        scores["fatigued"] += 0.10

    # Loud/noisy/flat spectral content may indicate overload context.
    if rms > 0.08 or flatness > 0.25:
        scores["overloaded"] += 0.20

    return normalise_scores(scores)


def fuse_audio_scores(
    model_scores: Dict[str, float],
    heuristic_scores: Dict[str, float],
    model_weight: float = 0.65,
    heuristic_weight: float = 0.35,
) -> Dict[str, float]:
    """
    Fuse trained audio model probabilities with YAMNet/heuristic scores.
    """
    total_weight = model_weight + heuristic_weight

    if total_weight <= 0:
        model_weight = 0.65
        heuristic_weight = 0.35
        total_weight = 1.0

    model_weight = model_weight / total_weight
    heuristic_weight = heuristic_weight / total_weight

    fused = {}

    for label in BEHAVIOUR_LABELS:
        fused[label] = (
            model_weight * float(model_scores.get(label, 0.0))
            + heuristic_weight * float(heuristic_scores.get(label, 0.0))
        )

    return normalise_scores(fused)


# ============================================================
# Public inference API
# ============================================================

def analyze_audio_file(path: str, top_k: int = 5) -> Dict[str, Any]:
    """
    Analyse an audio file and return behaviour prediction plus diagnostics.
    """
    audio_path = Path(path)

    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {path}")

    features = extract_audio_features(str(audio_path))
    yamnet_labels = yamnet_audio_labels(str(audio_path), top_k=top_k)

    model_error = ""

    try:
        model_prediction, model_scores = predict_audio_model_scores(features)
        model_available = True

    except Exception as error:
        model_prediction = "model_unavailable"
        model_scores = {
            label: 1.0 / len(BEHAVIOUR_LABELS)
            for label in BEHAVIOUR_LABELS
        }
        model_available = False
        model_error = str(error)

    heuristic_scores = heuristic_audio_scores(
        features=features,
        yamnet_labels=yamnet_labels,
    )

    if model_available:
        behaviour_scores = fuse_audio_scores(
            model_scores=model_scores,
            heuristic_scores=heuristic_scores,
            model_weight=0.65,
            heuristic_weight=0.35,
        )
        score_source = "trained_audio_model_plus_yamnet_heuristic"
    else:
        behaviour_scores = heuristic_scores
        score_source = "yamnet_heuristic_only"

    predicted_label = max(behaviour_scores, key=behaviour_scores.get)
    confidence = float(behaviour_scores[predicted_label])

    result: Dict[str, Any] = {
        "filepath": str(audio_path),
        "filename": audio_path.name,

        "predicted_label": predicted_label,
        "predicted_behaviour": predicted_label,
        "prediction_score": confidence,
        "behaviour_scores": behaviour_scores,

        "model_prediction": model_prediction,
        "model_scores": model_scores,
        "model_available": model_available,
        "model_path": str(MODEL_PATH),
        "metadata_path": str(META_PATH),

        "heuristic_scores": heuristic_scores,
        "yamnet_labels": yamnet_labels,
        "yamnet_source": get_yamnet_source(),
        "yamnet_local_path": str(LOCAL_YAMNET_DIR),
        "yamnet_tfhub_url": YAMNET_TFHUB_URL,

        "audio_features": features,
        "score_source": score_source,
        "supported_classes": BEHAVIOUR_LABELS,

        "method": "trained_librosa_audio_model_plus_local_yamnet_context",
        "methodological_note": (
            "This audio pipeline combines a trained librosa-feature classifier "
            "with YAMNet acoustic-context labels and transparent heuristic "
            "behaviour mapping. The audio model estimates environmental context "
            "and should be interpreted as supporting behavioural evidence, not "
            "as a standalone diagnosis."
        ),
    }

    if not model_available:
        result["model_error"] = model_error

    return result
