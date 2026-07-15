# === app/models/text_model.py ===
# SenseFuzeAI - Text Model Runtime Module
#
# This module synchronises with:
#   - train_text_model.py
#   - model_artifacts/live_text_test.py
#
# It loads the trained text model artifact:
#   model_artifacts/text_model.joblib
#
# The trained model expects raw text input and predicts one of:
#   focused, distracted, fatigued, overloaded
#
# It also provides lightweight sentiment estimation:
#   positive, neutral, negative
#
# Main public function:
#   analyze_text(text: str) -> dict

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict

import joblib


# ============================================================
# Project paths
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[2]

MODEL_DIR = BASE_DIR / "model_artifacts"
TEXT_MODEL_PATH = MODEL_DIR / "text_model.joblib"
TEXT_META_PATH = MODEL_DIR / "text_model_meta.json"


# ============================================================
# SenseFuzeAI behaviour classes
# ============================================================

BEHAVIOUR_CLASSES = [
    "focused",
    "distracted",
    "fatigued",
    "overloaded",
]


# ============================================================
# Lightweight sentiment lexicons
# ============================================================

POSITIVE_WORDS = {
    "good",
    "great",
    "calm",
    "clear",
    "steady",
    "confident",
    "comfortable",
    "productive",
    "focused",
    "focus",
    "motivated",
    "ready",
    "alert",
    "positive",
    "relaxed",
    "fine",
    "well",
    "happy",
    "satisfied",
    "organised",
    "organized",
    "controlled",
    "engaged",
    "stable",
    "progress",
    "smoothly",
    "effective",
    "efficient",
    "capable",
}

NEGATIVE_WORDS = {
    "bad",
    "tired",
    "fatigued",
    "drained",
    "exhausted",
    "sleepy",
    "slow",
    "overwhelmed",
    "overloaded",
    "stressed",
    "stress",
    "pressure",
    "anxious",
    "worried",
    "distracted",
    "unfocused",
    "confused",
    "frustrated",
    "difficult",
    "hard",
    "struggling",
    "cannot",
    "can't",
    "unable",
    "urgent",
    "heavy",
    "weak",
    "lost",
    "scattered",
    "panic",
    "burnt",
    "burned",
    "deadline",
    "deadlines",
}


# ============================================================
# Cached model objects
# ============================================================

_TEXT_MODEL: Any | None = None
_TEXT_META: dict[str, Any] | None = None


# ============================================================
# Loading utilities
# ============================================================

def load_json(path: Path) -> dict[str, Any]:
    """
    Safely load a JSON file.
    """
    if not path.exists():
        return {}

    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def get_text_model() -> Any:
    """
    Load and cache the trained text model.

    The model should be created by running:
        python train_text_model.py
    """
    global _TEXT_MODEL

    if _TEXT_MODEL is None:
        if not TEXT_MODEL_PATH.exists():
            raise FileNotFoundError(
                f"Trained text model not found: {TEXT_MODEL_PATH}\n\n"
                "Run the text training pipeline first:\n"
                "python train_text_model.py"
            )

        _TEXT_MODEL = joblib.load(TEXT_MODEL_PATH)

    return _TEXT_MODEL


def get_text_metadata() -> dict[str, Any]:
    """
    Load and cache text model metadata.
    """
    global _TEXT_META

    if _TEXT_META is None:
        _TEXT_META = load_json(TEXT_META_PATH)

    return _TEXT_META


# ============================================================
# General utilities
# ============================================================

def tokenize_text(text: str) -> list[str]:
    """
    Tokenise text for lightweight lexical sentiment scoring.
    """
    return re.findall(r"[a-zA-Z']+", str(text).lower())


def normalise_scores(scores: Dict[str, float]) -> Dict[str, float]:
    """
    Ensure behaviour scores are non-negative and sum to 1.
    """
    clean = {
        label: max(float(scores.get(label, 0.0)), 0.0)
        for label in BEHAVIOUR_CLASSES
    }

    total = sum(clean.values())

    if total <= 0:
        return {
            label: 1.0 / len(BEHAVIOUR_CLASSES)
            for label in BEHAVIOUR_CLASSES
        }

    return {
        label: value / total
        for label, value in clean.items()
    }


def get_model_classes(model: Any) -> list[str]:
    """
    Extract class ordering from a scikit-learn pipeline or estimator.
    """
    classes = list(getattr(model, "classes_", []))

    if not classes and hasattr(model, "named_steps"):
        final_estimator = list(model.named_steps.values())[-1]
        classes = list(getattr(final_estimator, "classes_", []))

    if not classes:
        return BEHAVIOUR_CLASSES

    return [str(label) for label in classes]


def get_best_label(scores: Dict[str, float]) -> str:
    """
    Return the label with the highest score.
    """
    return max(scores, key=scores.get)


# ============================================================
# Sentiment estimation
# ============================================================

def estimate_sentiment(text: str) -> dict[str, Any]:
    """
    Estimate sentiment using lightweight lexical scoring.

    This keeps runtime simple and avoids requiring an additional Hugging Face
    model just to run the application. The output is intended as supporting
    evidence, not as the primary behaviour classifier.
    """
    tokens = tokenize_text(text)

    positive_hits = [token for token in tokens if token in POSITIVE_WORDS]
    negative_hits = [token for token in tokens if token in NEGATIVE_WORDS]

    positive_count = len(positive_hits)
    negative_count = len(negative_hits)
    total_hits = positive_count + negative_count

    if total_hits == 0:
        return {
            "sentiment_label": "neutral",
            "sentiment_score": 1.0,
            "positive_count": 0,
            "negative_count": 0,
            "positive_hits": [],
            "negative_hits": [],
            "method": "rule_based_lexical_sentiment",
        }

    raw_score = (positive_count - negative_count) / total_hits

    if raw_score > 0.15:
        sentiment_label = "positive"
        sentiment_score = abs(raw_score)
    elif raw_score < -0.15:
        sentiment_label = "negative"
        sentiment_score = abs(raw_score)
    else:
        sentiment_label = "neutral"
        sentiment_score = 1.0 - abs(raw_score)

    return {
        "sentiment_label": sentiment_label,
        "sentiment_score": float(sentiment_score),
        "positive_count": int(positive_count),
        "negative_count": int(negative_count),
        "positive_hits": positive_hits,
        "negative_hits": negative_hits,
        "method": "rule_based_lexical_sentiment",
    }


# ============================================================
# Behaviour prediction
# ============================================================

def predict_behaviour_scores(text: str) -> dict[str, float]:
    """
    Predict behaviour probabilities/scores using text_model.joblib.
    """
    model = get_text_model()

    safe_text = str(text).strip()

    if not safe_text:
        safe_text = "neutral"

    scores: dict[str, float] = {}

    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba([safe_text])[0]
        model_classes = get_model_classes(model)

        for class_name, probability in zip(model_classes, probabilities):
            scores[str(class_name)] = float(probability)

    else:
        prediction = str(model.predict([safe_text])[0])

        scores = {
            label: 1.0 if label == prediction else 0.0
            for label in BEHAVIOUR_CLASSES
        }

    return normalise_scores(scores)


def predict_behaviour_label(text: str) -> str:
    """
    Predict the most likely behaviour label.
    """
    model = get_text_model()

    safe_text = str(text).strip()

    if not safe_text:
        safe_text = "neutral"

    prediction = str(model.predict([safe_text])[0])

    if prediction not in BEHAVIOUR_CLASSES:
        behaviour_scores = predict_behaviour_scores(safe_text)
        return get_best_label(behaviour_scores)

    return prediction


# ============================================================
# Main public API
# ============================================================

def analyze_text(text: str) -> Dict[str, Any]:
    """
    Analyse text using the trained SenseFuzeAI text model and sentiment scoring.

    Returns a dictionary compatible with earlier versions of the application,
    but the behavioural prediction now comes from the trained text_model.joblib
    pipeline rather than from keyword rules.
    """
    safe_text = str(text).strip() if text and str(text).strip() else "neutral"

    meta = get_text_metadata()

    behaviour_scores = predict_behaviour_scores(safe_text)
    predicted_behaviour = predict_behaviour_label(safe_text)

    if predicted_behaviour not in behaviour_scores:
        predicted_behaviour = get_best_label(behaviour_scores)

    behaviour_confidence = float(behaviour_scores.get(predicted_behaviour, 0.0))

    sentiment = estimate_sentiment(safe_text)

    sentiment_label = sentiment["sentiment_label"]
    sentiment_score = float(sentiment["sentiment_score"])

    return {
        # Input
        "text": text,
        "safe_text": safe_text,

        # Backward-compatible generic fields
        "label": sentiment_label,
        "score": sentiment_score,

        # Sentiment fields
        "sentiment_label": sentiment_label,
        "sentiment_score": sentiment_score,
        "sentiment_method": sentiment["method"],
        "sentiment_positive_count": sentiment["positive_count"],
        "sentiment_negative_count": sentiment["negative_count"],
        "sentiment_positive_hits": sentiment["positive_hits"],
        "sentiment_negative_hits": sentiment["negative_hits"],

        # Behaviour fields
        "predicted_behaviour": predicted_behaviour,
        "behaviour_confidence": behaviour_confidence,
        "behaviour_scores": behaviour_scores,

        # Model metadata
        "method": meta.get("method", "trained_text_model_joblib"),
        "model_name": meta.get("model_name", "text_model.joblib"),
        "model_path": str(TEXT_MODEL_PATH),
        "metadata_path": str(TEXT_META_PATH),
        "classes": meta.get("classes", BEHAVIOUR_CLASSES),
        "training_accuracy": meta.get("accuracy"),
        "training_macro_f1": meta.get("macro_f1"),
        "training_weighted_f1": meta.get("weighted_f1"),
        "selected_classifier": meta.get("selected_classifier"),
        "selection_metric": meta.get("selection_metric"),
    }


# ============================================================
# Optional command-line smoke test
# ============================================================

if __name__ == "__main__":
    sample_text = "I am overwhelmed by too many deadlines and too much information."
    result = analyze_text(sample_text)

    print("SenseFuzeAI text model smoke test")
    print("=================================")
    print(f"Input: {sample_text}")
    print(f"Predicted behaviour: {result['predicted_behaviour']}")
    print(f"Behaviour confidence: {result['behaviour_confidence']:.4f}")
    print(f"Sentiment: {result['sentiment_label']} ({result['sentiment_score']:.4f})")
    print("Behaviour scores:")
    for label, score in result["behaviour_scores"].items():
        print(f"  {label}: {score:.4f}")
