# === app/fusion.py ===
# SenseFuzeAI - Multimodal Fusion Logic
#
# Provides:
#   - class-score normalisation
#   - modality confidence calibration / score softening
#   - audio/image heuristic score mapping
#   - explainable dynamic weighted late fusion
#   - missing-modality handling
#   - effective fusion-weight reporting
#
# Core principles:
#   1. Missing optional modalities must not contribute fusion weight.
#   2. Available modality weights are re-normalised dynamically.
#   3. Overconfident single-modality predictions are softened before fusion.
#
# Rationale:
#   Live behavioural inference is noisy. A short keystroke sequence, a weak
#   caption, or a narrow model prediction should not be allowed to override
#   several other evidence sources with unrealistic certainty.

from __future__ import annotations

from typing import Any, Dict, Iterable, Mapping


# ============================================================
# Behaviour classes
# ============================================================

BEHAVIOUR_CLASSES = [
    "focused",
    "distracted",
    "fatigued",
    "overloaded",
]


# ============================================================
# Base fusion weights
# ============================================================
# These are the intended full-modality weights.
# When a modality is missing, the available weights are normalised again.

BASE_FUSION_WEIGHTS: Dict[str, float] = {
    "keystroke": 0.30,
    "text": 0.30,
    "audio": 0.20,
    "image": 0.20,
}


# ============================================================
# Modality score calibration settings
# ============================================================
# Smoothing means:
#   calibrated_score = (1 - smoothing) * model_score + smoothing * uniform_prior
#
# A higher value makes the modality less able to produce overconfident
# one-hot predictions.
#
# Keystroke is intentionally softened more strongly because short live typing
# samples are noisy and session-dependent.

MODALITY_SMOOTHING: Dict[str, float] = {
    "keystroke": 0.30,
    "text": 0.10,
    "audio": 0.10,
    "image": 0.10,
}


# ============================================================
# General utilities
# ============================================================

def normalize_label(label: Any) -> str:
    """
    Normalise behaviour labels to lowercase strings.
    """
    return str(label or "").strip().lower()


def uniform_scores() -> Dict[str, float]:
    """
    Return a uniform distribution across behaviour classes.
    """
    value = 1.0 / len(BEHAVIOUR_CLASSES)

    return {
        label: value
        for label in BEHAVIOUR_CLASSES
    }


def normalize_scores(scores: Mapping[str, Any] | None) -> Dict[str, float]:
    """
    Normalise a score dictionary so that:
      - all expected behaviour classes exist
      - values are non-negative
      - values sum to 1

    If the input is missing or invalid, a uniform distribution is returned.
    """
    if not scores:
        return uniform_scores()

    clean: Dict[str, float] = {}

    for label in BEHAVIOUR_CLASSES:
        try:
            value = float(scores.get(label, 0.0))
        except Exception:
            value = 0.0

        clean[label] = max(value, 0.0)

    total = sum(clean.values())

    if total <= 0:
        return uniform_scores()

    return {
        label: value / total
        for label, value in clean.items()
    }


def soften_scores(
    scores: Mapping[str, Any] | None,
    smoothing: float,
) -> Dict[str, float]:
    """
    Apply probability smoothing to prevent a modality from producing
    overconfident one-hot predictions.

    Example:
        raw:
            focused:    0.000
            distracted: 1.000
            fatigued:   0.000
            overloaded: 0.000

        smoothing = 0.30

        softened:
            focused:    0.075
            distracted: 0.775
            fatigued:   0.075
            overloaded: 0.075

    This keeps the model's preference but adds an uncertainty prior.
    """
    smoothing = max(0.0, min(1.0, float(smoothing)))

    normalised = normalize_scores(scores)
    uniform = uniform_scores()

    softened = {
        label: (1.0 - smoothing) * normalised[label]
        + smoothing * uniform[label]
        for label in BEHAVIOUR_CLASSES
    }

    return normalize_scores(softened)


def calibrate_modality_scores(
    modality: str,
    scores: Mapping[str, Any] | None,
) -> Dict[str, float]:
    """
    Apply modality-specific score calibration.

    This function should be used before fusion, not before standalone display.
    Standalone model cards may still show the raw model probabilities, while
    fusion uses calibrated probabilities for robustness.
    """
    clean_modality = str(modality or "").strip().lower()
    smoothing = MODALITY_SMOOTHING.get(clean_modality, 0.10)

    return soften_scores(scores, smoothing=smoothing)


def prediction_from_scores(scores: Mapping[str, Any] | None) -> str:
    """
    Select the highest-scoring behaviour label.
    """
    normalised = normalize_scores(scores)

    return max(normalised, key=normalised.get)


def confidence_from_scores(scores: Mapping[str, Any] | None) -> float:
    """
    Return the highest score from a normalised score distribution.
    """
    normalised = normalize_scores(scores)

    return float(max(normalised.values()))


def get_fusion_weights() -> Dict[str, float]:
    """
    Return the base fusion weights.
    """
    return dict(BASE_FUSION_WEIGHTS)


def get_modality_smoothing() -> Dict[str, float]:
    """
    Return modality-specific calibration smoothing values.
    """
    return dict(MODALITY_SMOOTHING)


def get_available_modalities(
    *,
    keystroke_available: bool = True,
    text_available: bool = True,
    audio_available: bool = False,
    image_available: bool = False,
) -> list[str]:
    """
    Return a list of modality names that are available for fusion.
    """
    available: list[str] = []

    if keystroke_available:
        available.append("keystroke")

    if text_available:
        available.append("text")

    if audio_available:
        available.append("audio")

    if image_available:
        available.append("image")

    return available


def get_missing_modalities(available_modalities: Iterable[str]) -> list[str]:
    """
    Return missing modality names.
    """
    available_set = {
        str(modality).strip().lower()
        for modality in available_modalities
    }

    return [
        modality
        for modality in BASE_FUSION_WEIGHTS
        if modality not in available_set
    ]


def compute_effective_fusion_weights(
    available_modalities: Iterable[str],
    base_weights: Mapping[str, float] | None = None,
) -> Dict[str, float]:
    """
    Compute dynamic effective weights.

    Missing modalities receive weight 0.0.
    Available modalities are re-normalised to sum to 1.

    Example:
        available = ["keystroke", "text"]

        base:
            keystroke 0.30
            text      0.30
            audio     0.20
            image     0.20

        effective:
            keystroke 0.50
            text      0.50
            audio     0.00
            image     0.00
    """
    weights = dict(base_weights or BASE_FUSION_WEIGHTS)

    available_set = {
        str(modality).strip().lower()
        for modality in available_modalities
    }

    raw_effective = {
        modality: float(weights.get(modality, 0.0))
        if modality in available_set
        else 0.0
        for modality in BASE_FUSION_WEIGHTS
    }

    total = sum(raw_effective.values())

    if total <= 0:
        # Fallback: use text + keystroke as minimum baseline.
        return {
            "keystroke": 0.5,
            "text": 0.5,
            "audio": 0.0,
            "image": 0.0,
        }

    return {
        modality: value / total
        for modality, value in raw_effective.items()
    }


# ============================================================
# Audio heuristic mapping
# ============================================================

AUDIO_KEYWORD_MAP: Dict[str, Dict[str, float]] = {
    "focused": {
        "silence": 1.0,
        "quiet": 0.9,
        "inside": 0.4,
        "room": 0.4,
        "typing": 0.5,
        "keyboard": 0.5,
    },
    "distracted": {
        "speech": 0.9,
        "conversation": 1.0,
        "talking": 0.9,
        "crowd": 0.8,
        "chatter": 0.9,
        "phone": 0.8,
        "ringtone": 0.8,
        "music": 0.5,
        "television": 0.7,
        "cacophony": 0.7,
    },
    "fatigued": {
        "yawn": 1.0,
        "breathing": 0.7,
        "snoring": 1.0,
        "sigh": 0.7,
        "sleep": 0.8,
        "silence": 0.25,
        "quiet": 0.25,
    },
    "overloaded": {
        "alarm": 1.0,
        "siren": 1.0,
        "shout": 0.9,
        "shouting": 0.9,
        "traffic": 0.8,
        "construction": 0.9,
        "bang": 0.8,
        "noise": 0.7,
        "chaos": 0.8,
        "tools": 0.6,
        "power tool": 0.8,
        "machinery": 0.8,
    },
}


def audio_to_behaviour_scores(
    audio_labels: Iterable[Any] | None,
) -> Dict[str, float]:
    """
    Convert YAMNet/audio labels into behaviour scores using a transparent
    keyword mapping.

    Expected audio_labels examples:
        [("Speech", 0.82), ("Inside, small room", 0.41)]
        [{"label": "Speech", "score": 0.82}]
    """
    scores = {
        label: 0.05
        for label in BEHAVIOUR_CLASSES
    }

    if not audio_labels:
        return normalize_scores(scores)

    for item in audio_labels:
        label_text = ""
        confidence = 1.0

        if isinstance(item, dict):
            label_text = str(
                item.get("label")
                or item.get("name")
                or item.get("class")
                or ""
            )

            try:
                confidence = float(
                    item.get("score", item.get("confidence", 1.0))
                )
            except Exception:
                confidence = 1.0

        elif isinstance(item, (list, tuple)) and len(item) >= 2:
            label_text = str(item[0])

            try:
                confidence = float(item[1])
            except Exception:
                confidence = 1.0

        else:
            label_text = str(item)
            confidence = 1.0

        lower_label = label_text.lower()

        for behaviour, keyword_weights in AUDIO_KEYWORD_MAP.items():
            for keyword, weight in keyword_weights.items():
                if keyword in lower_label:
                    scores[behaviour] += float(weight) * max(confidence, 0.0)

    return normalize_scores(scores)


# ============================================================
# Image caption heuristic fallback
# ============================================================

IMAGE_KEYWORD_MAP: Dict[str, Dict[str, float]] = {
    "focused": {
        "laptop": 0.6,
        "computer": 0.6,
        "desk": 0.4,
        "working": 0.7,
        "typing": 0.6,
        "focused": 1.0,
        "upright": 0.8,
        "attentive": 0.9,
        "organized": 0.7,
        "organised": 0.7,
        "tidy": 0.7,
        "task-oriented": 0.9,
        "stable body posture": 0.8,
        "normal head position": 0.7,
    },
    "distracted": {
        "phone": 1.0,
        "cell phone": 1.0,
        "mobile": 1.0,
        "looking away": 0.9,
        "distracted": 1.0,
        "multitasking": 0.8,
        "interruption": 0.8,
        "interruptions": 0.8,
        "television": 0.7,
        "off-task": 0.8,
    },
    "fatigued": {
        "tired": 1.0,
        "sleepy": 1.0,
        "yawning": 1.0,
        "head down": 0.9,
        "low head": 0.8,
        "slouched": 0.8,
        "dim": 0.6,
        "night": 0.5,
        "resting": 0.6,
        "reduced alertness": 0.8,
    },
    "overloaded": {
        "overwhelmed": 1.0,
        "overloaded": 1.0,
        "stressed": 1.0,
        "cluttered": 0.9,
        "messy": 0.9,
        "many papers": 0.9,
        "documents": 0.7,
        "deadline": 0.9,
        "chaotic": 0.9,
        "frustrated": 0.8,
        "many visible work demands": 0.9,
        "high task pressure": 0.9,
    },
}


def image_to_behaviour_scores(caption_or_text: Any) -> Dict[str, float]:
    """
    Fallback conversion from caption/description text to behaviour scores.

    This is used only when the image model does not directly return
    behaviour_scores.
    """
    text = str(caption_or_text or "").lower()

    scores = {
        label: 0.05
        for label in BEHAVIOUR_CLASSES
    }

    if not text.strip():
        return normalize_scores(scores)

    for behaviour, keyword_weights in IMAGE_KEYWORD_MAP.items():
        for keyword, weight in keyword_weights.items():
            if keyword in text:
                scores[behaviour] += float(weight)

    return normalize_scores(scores)


# ============================================================
# Explainability helpers
# ============================================================

def explain_fusion_inputs(
    *,
    keystroke_scores: Mapping[str, Any] | None = None,
    text_scores: Mapping[str, Any] | None = None,
    audio_scores: Mapping[str, Any] | None = None,
    image_scores: Mapping[str, Any] | None = None,
) -> Dict[str, Dict[str, float]]:
    """
    Return raw-normalised modality score dictionaries.

    These are useful for display because they show what each model originally
    contributed before fusion calibration.
    """
    return {
        "keystroke": normalize_scores(keystroke_scores),
        "text": normalize_scores(text_scores),
        "audio": normalize_scores(audio_scores),
        "image": normalize_scores(image_scores),
    }


def explain_calibrated_fusion_inputs(
    *,
    keystroke_scores: Mapping[str, Any] | None = None,
    text_scores: Mapping[str, Any] | None = None,
    audio_scores: Mapping[str, Any] | None = None,
    image_scores: Mapping[str, Any] | None = None,
) -> Dict[str, Dict[str, float]]:
    """
    Return calibrated modality score dictionaries used internally by fusion.
    """
    return {
        "keystroke": calibrate_modality_scores("keystroke", keystroke_scores),
        "text": calibrate_modality_scores("text", text_scores),
        "audio": calibrate_modality_scores("audio", audio_scores),
        "image": calibrate_modality_scores("image", image_scores),
    }


def build_fusion_metadata(
    *,
    keystroke_available: bool = True,
    text_available: bool = True,
    audio_available: bool = False,
    image_available: bool = False,
) -> Dict[str, Any]:
    """
    Build transparent metadata for UI/reporting.
    """
    uploaded_modalities = get_available_modalities(
        keystroke_available=keystroke_available,
        text_available=text_available,
        audio_available=audio_available,
        image_available=image_available,
    )

    missing_modalities = get_missing_modalities(uploaded_modalities)

    effective_fusion_weights = compute_effective_fusion_weights(
        uploaded_modalities
    )

    return {
        "uploaded_modalities": uploaded_modalities,
        "missing_modalities": missing_modalities,
        "base_fusion_weights": get_fusion_weights(),
        "effective_fusion_weights": effective_fusion_weights,
        "modality_smoothing": get_modality_smoothing(),
        "fusion_method": "calibrated_dynamic_weighted_late_fusion",
        "fusion_note": (
            "Missing optional modalities are assigned zero effective weight. "
            "Available modality weights are re-normalised before score fusion. "
            "Each modality score distribution is softly calibrated before fusion "
            "to reduce unrealistic overconfidence from any single model."
        ),
    }


# ============================================================
# Main dynamic late fusion
# ============================================================

def fuse_predictions(
    *,
    keystroke_scores: Mapping[str, Any] | None = None,
    text_scores: Mapping[str, Any] | None = None,
    audio_scores: Mapping[str, Any] | None = None,
    image_scores: Mapping[str, Any] | None = None,
    keystroke_available: bool = True,
    text_available: bool = True,
    audio_available: bool = False,
    image_available: bool = False,
    use_trained_fusion: bool = False,
) -> Dict[str, float]:
    """
    Fuse modality score distributions using calibrated dynamic weighted
    late fusion.

    Important:
        - Missing modalities do not contribute.
        - Effective weights are normalised over available modalities.
        - Each available modality is softly calibrated before fusion.
        - The trained fusion option is reserved for future use with true
          session-aligned multimodal training data.

    Args:
        keystroke_scores: Behaviour scores from keystroke model.
        text_scores: Behaviour scores from text model.
        audio_scores: Behaviour scores from audio model or audio heuristic.
        image_scores: Behaviour scores from vision model.
        keystroke_available: Whether keystroke evidence should contribute.
        text_available: Whether text evidence should contribute.
        audio_available: Whether audio evidence should contribute.
        image_available: Whether image evidence should contribute.
        use_trained_fusion: Reserved for experimental trained fusion model.

    Returns:
        Final normalised behaviour scores.
    """
    _ = use_trained_fusion

    available_modalities = get_available_modalities(
        keystroke_available=keystroke_available,
        text_available=text_available,
        audio_available=audio_available,
        image_available=image_available,
    )

    effective_weights = compute_effective_fusion_weights(
        available_modalities
    )

    modality_scores = {
        "keystroke": calibrate_modality_scores("keystroke", keystroke_scores),
        "text": calibrate_modality_scores("text", text_scores),
        "audio": calibrate_modality_scores("audio", audio_scores),
        "image": calibrate_modality_scores("image", image_scores),
    }

    fused_scores = {
        label: 0.0
        for label in BEHAVIOUR_CLASSES
    }

    for modality, weight in effective_weights.items():
        if weight <= 0.0:
            continue

        scores = modality_scores[modality]

        for label in BEHAVIOUR_CLASSES:
            fused_scores[label] += weight * float(scores.get(label, 0.0))

    return normalize_scores(fused_scores)


def fuse_predictions_with_metadata(
    *,
    keystroke_scores: Mapping[str, Any] | None = None,
    text_scores: Mapping[str, Any] | None = None,
    audio_scores: Mapping[str, Any] | None = None,
    image_scores: Mapping[str, Any] | None = None,
    keystroke_available: bool = True,
    text_available: bool = True,
    audio_available: bool = False,
    image_available: bool = False,
    use_trained_fusion: bool = False,
) -> Dict[str, Any]:
    """
    Fuse predictions and return both final scores and transparent metadata.

    This is the preferred function for app/main.py because it gives the web UI:
      - uploaded_modalities
      - missing_modalities
      - effective_fusion_weights
      - modality_smoothing
      - raw modality scores
      - calibrated modality scores
      - final_scores
      - final_prediction
      - final_confidence
    """
    final_scores = fuse_predictions(
        keystroke_scores=keystroke_scores,
        text_scores=text_scores,
        audio_scores=audio_scores,
        image_scores=image_scores,
        keystroke_available=keystroke_available,
        text_available=text_available,
        audio_available=audio_available,
        image_available=image_available,
        use_trained_fusion=use_trained_fusion,
    )

    metadata = build_fusion_metadata(
        keystroke_available=keystroke_available,
        text_available=text_available,
        audio_available=audio_available,
        image_available=image_available,
    )

    final_prediction = prediction_from_scores(final_scores)
    final_confidence = confidence_from_scores(final_scores)

    modality_scores = explain_fusion_inputs(
        keystroke_scores=keystroke_scores,
        text_scores=text_scores,
        audio_scores=audio_scores,
        image_scores=image_scores,
    )

    calibrated_modality_scores = explain_calibrated_fusion_inputs(
        keystroke_scores=keystroke_scores,
        text_scores=text_scores,
        audio_scores=audio_scores,
        image_scores=image_scores,
    )

    return {
        "final_scores": final_scores,
        "final_prediction": final_prediction,
        "final_confidence": final_confidence,
        "modality_scores": modality_scores,
        "calibrated_modality_scores": calibrated_modality_scores,
        **metadata,
    }
