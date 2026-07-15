# === model_artifacts/live_fusion_test.py ===
# SenseFuzeAI - Live Multimodal Fusion Test GUI
#
# Presentation-first version:
#   - main output shows concise prototype-friendly fusion result
#   - technical details are available through "Show Technical Details"
#   - exports include full diagnostic information
#
# Runtime modalities:
#   - keystroke
#   - text
#   - audio
#   - image
#
# Fusion method:
#   weighted late fusion
#
# Export support:
#   - JSON full diagnostic result
#   - CSV compact summary
#   - TXT human-readable report

from __future__ import annotations

import csv
import json
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext


# ============================================================
# Project path setup
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))


from app.fusion import (
    BEHAVIOUR_CLASSES,
    audio_to_behaviour_scores,
    explain_fusion_inputs,
    fuse_predictions,
    get_fusion_weights,
    image_to_behaviour_scores,
    normalize_scores,
    prediction_from_scores,
    uniform_scores,
)

from app.models.audio_models import analyze_audio_file
from app.models.image_model import analyze_image_file
from app.models.keystroke_model import predict_keystroke_behaviour
from app.models.text_model import analyze_text


# ============================================================
# Configuration
# ============================================================

UPLOAD_DIR = PROJECT_ROOT / "data" / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

EXPORT_DIR = PROJECT_ROOT / "model_artifacts" / "exported_results"
EXPORT_DIR.mkdir(parents=True, exist_ok=True)

SUPPORTED_AUDIO = {".wav", ".mp3", ".m4a", ".flac", ".ogg", ".aac"}
SUPPORTED_IMAGES = {".jpg", ".jpeg", ".png", ".webp"}

MIN_KEYDOWNS = 20


# ============================================================
# Runtime state
# ============================================================

events: List[Dict[str, Any]] = []
active_keys = set()

selected_audio_path: Optional[Path] = None
selected_image_path: Optional[Path] = None

latest_fusion_result: Optional[dict[str, Any]] = None
latest_summary_text: str = ""
latest_technical_text: str = ""


# ============================================================
# General utilities
# ============================================================

def timestamp_for_filename() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def format_percent(value: float) -> str:
    safe_value = max(0.0, min(1.0, safe_float(value)))
    return f"{safe_value * 100:.2f}%"


def safe_json_dump(data: Dict[str, Any]) -> str:
    return json.dumps(data, indent=2, default=str, ensure_ascii=False)


def normalise_key(event) -> str:
    if event.keysym == "BackSpace":
        return "backspace"

    if event.keysym == "Delete":
        return "delete"

    if event.keysym == "space":
        return "space"

    if len(event.char) == 1:
        return event.char.lower()

    return event.keysym.lower()


def count_keydowns(source_events: List[Dict[str, Any]]) -> int:
    return sum(1 for event in source_events if event.get("type") == "down")


def copy_to_uploads(path: Path) -> Path:
    safe_name = path.name.replace(" ", "_")
    destination = UPLOAD_DIR / safe_name

    if destination.resolve() != path.resolve():
        shutil.copy2(path, destination)

    return destination


def clean_scores(scores: Optional[Dict[str, float]]) -> Dict[str, float]:
    if not scores:
        return uniform_scores()

    return normalize_scores(scores)


def best_score(scores: Dict[str, float]) -> float:
    scores = clean_scores(scores)
    return float(max(scores.values()))


def require_latest_result() -> Optional[dict[str, Any]]:
    if latest_fusion_result is None:
        messagebox.showwarning(
            "No Result Available",
            "No fusion prediction result is available yet.\n\n"
            "Please run fusion prediction first.",
        )
        return None

    return latest_fusion_result


def flatten_scores(
    scores: Dict[str, float],
    prefix: str,
) -> dict[str, float]:
    scores = clean_scores(scores)

    return {
        f"{prefix}_{label}_prob": float(scores.get(label, 0.0))
        for label in BEHAVIOUR_CLASSES
    }


def compact_list(value: Any) -> str:
    if isinstance(value, list):
        return "; ".join(str(item) for item in value)

    if value is None:
        return ""

    return str(value)


def format_scores_decimal(scores: Dict[str, float]) -> str:
    scores = clean_scores(scores)
    lines = []

    for label, score in sorted(
        scores.items(),
        key=lambda item: item[1],
        reverse=True,
    ):
        lines.append(f"  {label}: {score:.4f}")

    return "\n".join(lines)


def format_scores_percent(scores: Dict[str, float]) -> str:
    scores = clean_scores(scores)
    label_width = max(len(label) for label in BEHAVIOUR_CLASSES)

    lines = []

    for label, score in sorted(
        scores.items(),
        key=lambda item: item[1],
        reverse=True,
    ):
        display_label = label.capitalize()
        lines.append(f"{display_label:<{label_width + 2}} {format_percent(score):>8}")

    return "\n".join(lines)


def format_visual_cue_summary(summary: Any) -> str:
    if not isinstance(summary, dict) or not summary:
        return "  No visual cue summary returned."

    lines = []

    for key, value in summary.items():
        lines.append(f"  {key}: {value}")

    return "\n".join(lines)


# ============================================================
# Score extraction helpers
# ============================================================

def extract_audio_scores(audio_result: Dict[str, Any]) -> Dict[str, float]:
    if not audio_result:
        return uniform_scores()

    for key in [
        "behaviour_scores",
        "prediction_scores",
        "scores",
        "class_probabilities",
    ]:
        value = audio_result.get(key)

        if isinstance(value, dict):
            return clean_scores(value)

    labels = audio_result.get("yamnet_labels") or audio_result.get("labels") or []

    if isinstance(labels, list):
        return audio_to_behaviour_scores(labels)

    return uniform_scores()


def extract_image_scores(image_result: Dict[str, Any]) -> Dict[str, float]:
    if not image_result:
        return uniform_scores()

    for key in [
        "behaviour_scores",
        "prediction_scores",
        "scores",
        "class_probabilities",
    ]:
        value = image_result.get(key)

        if isinstance(value, dict):
            return clean_scores(value)

    fused = image_result.get("fused_prediction", {})
    fused_scores = fused.get("visual_fused_scores") if isinstance(fused, dict) else None

    if isinstance(fused_scores, dict):
        return clean_scores(fused_scores)

    caption = (
        image_result.get("behaviour_caption")
        or image_result.get("generic_caption")
        or image_result.get("caption")
        or ""
    )

    return image_to_behaviour_scores(caption)


def extract_text_scores(text_result: Dict[str, Any]) -> Dict[str, float]:
    if not text_result:
        return uniform_scores()

    scores = text_result.get("behaviour_scores")

    if isinstance(scores, dict):
        return clean_scores(scores)

    return uniform_scores()


def get_generic_caption(image_result: Dict[str, Any]) -> str:
    return (
        image_result.get("generic_caption")
        or image_result.get("caption")
        or image_result.get("scene_description")
        or ""
    )


def get_behaviour_caption(image_result: Dict[str, Any]) -> str:
    return (
        image_result.get("behaviour_caption")
        or image_result.get("behaviour_aware_caption")
        or image_result.get("behaviour_description")
        or ""
    )


def get_visual_cue_summary(image_result: Dict[str, Any]) -> Dict[str, Any]:
    summary = image_result.get("visual_cue_summary", {})

    if isinstance(summary, dict):
        return summary

    if isinstance(summary, str):
        try:
            return json.loads(summary)
        except Exception:
            return {}

    return {}


# ============================================================
# Keystroke capture
# ============================================================

def on_key_press(event) -> None:
    key = normalise_key(event)

    if key in active_keys:
        return

    active_keys.add(key)

    events.append(
        {
            "type": "down",
            "key": key,
            "ts": time.perf_counter(),
        }
    )


def on_key_release(event) -> None:
    key = normalise_key(event)
    active_keys.discard(key)

    events.append(
        {
            "type": "up",
            "key": key,
            "ts": time.perf_counter(),
        }
    )


# ============================================================
# File selection
# ============================================================

def choose_audio_file() -> None:
    global selected_audio_path

    file_path = filedialog.askopenfilename(
        title="Select audio file",
        filetypes=[
            ("Audio files", "*.wav *.mp3 *.m4a *.flac *.ogg *.aac"),
            ("All files", "*.*"),
        ],
    )

    if not file_path:
        return

    selected_audio_path = Path(file_path)
    audio_label.config(text=str(selected_audio_path))


def choose_image_file() -> None:
    global selected_image_path

    file_path = filedialog.askopenfilename(
        title="Select image file",
        filetypes=[
            ("Image files", "*.jpg *.jpeg *.png *.webp"),
            ("All files", "*.*"),
        ],
    )

    if not file_path:
        return

    selected_image_path = Path(file_path)
    image_label.config(text=str(selected_image_path))


# ============================================================
# Presentation builders
# ============================================================

def build_brief_fusion_interpretation(result: Dict[str, Any]) -> str:
    final_prediction = str(result.get("final_prediction", "unknown")).lower()

    modality_predictions = {
        "keystroke": result.get("keystroke_prediction"),
        "text": result.get("text_prediction"),
        "audio": result.get("audio_prediction"),
        "image": result.get("image_prediction"),
    }

    agreeing_modalities = [
        name for name, prediction in modality_predictions.items()
        if str(prediction).lower() == final_prediction
    ]

    available_modalities = [
        name for name, prediction in modality_predictions.items()
        if prediction not in {None, "not_provided", "unavailable", "insufficient_data"}
    ]

    if agreeing_modalities:
        agreement_text = (
            "The final decision is supported by "
            + ", ".join(agreeing_modalities)
            + "."
        )
    else:
        agreement_text = (
            "The modalities show mixed evidence, so the final decision is based "
            "on weighted late fusion of all available scores."
        )

    if final_prediction == "focused":
        base = (
            "The fused system classifies the current state as focused because "
            "the combined behavioural evidence is strongest for task engagement "
            "and stable work behaviour."
        )
    elif final_prediction == "distracted":
        base = (
            "The fused system classifies the current state as distracted because "
            "the combined evidence suggests attention diversion or interruption."
        )
    elif final_prediction == "fatigued":
        base = (
            "The fused system classifies the current state as fatigued because "
            "the combined evidence suggests reduced alertness, slower rhythm, "
            "or fatigue-related signals."
        )
    elif final_prediction == "overloaded":
        base = (
            "The fused system classifies the current state as overloaded because "
            "the combined evidence suggests stress, high task demand, or workload pressure."
        )
    else:
        base = (
            "The fused system could not identify a strong behavioural state. "
            "The result should be treated as uncertain."
        )

    image_caption = result.get("image_behaviour_caption") or ""
    if image_caption:
        image_text = "\n\nImage evidence summary: " + image_caption
    else:
        image_text = ""

    return (
        base
        + "\n\n"
        + agreement_text
        + f"\n\nAvailable modalities: {', '.join(available_modalities) if available_modalities else 'none'}."
        + image_text
    )


def build_summary_text(result: Dict[str, Any]) -> str:
    final_prediction = str(result.get("final_prediction", "unknown"))
    final_confidence = safe_float(result.get("final_confidence"))
    final_scores = clean_scores(result.get("final_scores", {}))

    lines = [
        "Live Multimodal Fusion Test",
        "=" * 50,
        "",
        f"Final Prediction: {final_prediction.upper()}",
        f"Confidence: {format_percent(final_confidence)}",
        "Fusion Method: weighted late fusion",
        "",
        "Final Behaviour Probabilities",
        "-" * 50,
        format_scores_percent(final_scores),
        "",
        "Modality Predictions",
        "-" * 50,
        f"Keystroke: {result.get('keystroke_prediction')} "
        f"({format_percent(best_score(result.get('keystroke_scores', {})))})",
        f"Text:      {result.get('text_prediction')} "
        f"({format_percent(safe_float(result.get('text_confidence')))})",
        f"Audio:     {result.get('audio_prediction')} "
        f"({format_percent(best_score(result.get('audio_scores', {})))})",
        f"Image:     {result.get('image_prediction')} "
        f"({format_percent(best_score(result.get('image_scores', {})))})",
        "",
        "Brief Interpretation",
        "-" * 50,
        result.get("brief_interpretation", ""),
        "",
        "Evidence Summary",
        "-" * 50,
        f"Typed text length:     {result.get('typed_text_length')} characters",
        f"Keystroke key presses: {result.get('keystroke_keydown_count')}",
        f"Audio file:            {'provided' if result.get('selected_audio_path') else 'not provided'}",
        f"Image file:            {'provided' if result.get('selected_image_path') else 'not provided'}",
        "",
        "Technical Details",
        "-" * 50,
        "Full modality scores, text sentiment, image cue summary,",
        "keystroke features, and raw diagnostic JSON are available",
        "through the 'Show Technical Details' button and export functions.",
        "",
        "Report Export",
        "-" * 50,
        "Full result available as JSON, CSV, or TXT report.",
    ]

    return "\n".join(lines)


def build_technical_text(result: Dict[str, Any]) -> str:
    text_sentiment = result.get("text_sentiment", {})
    keystroke_features = result.get("keystroke_features", {})
    visual_summary = result.get("image_visual_cue_summary", {})

    lines = [
        "Live Multimodal Fusion Test - Technical Details",
        "=" * 72,
        "",
        f"Created at:                {result.get('created_at')}",
        f"Fusion method:             {result.get('fusion_method')}",
        f"Fusion weights:            {result.get('fusion_weights')}",
        f"Trained fusion model used: {result.get('trained_fusion_model_used')}",
        "",
        "Final Fusion Result",
        "-" * 72,
        f"Final prediction:          {result.get('final_prediction')}",
        f"Final confidence:          {safe_float(result.get('final_confidence')):.4f}",
        "",
        "Final fused scores",
        "-" * 72,
        format_scores_decimal(result.get("final_scores", {})),
        "",
        "Modality predictions",
        "-" * 72,
        f"Keystroke prediction:      {result.get('keystroke_prediction')}",
        f"Text prediction:           {result.get('text_prediction')}",
        f"Text confidence:           {safe_float(result.get('text_confidence')):.4f}",
        f"Audio prediction:          {result.get('audio_prediction')}",
        f"Image prediction:          {result.get('image_prediction')}",
        f"Image reliability score:   {result.get('image_reliability_score')}",
        f"Image quality flag:        {result.get('image_quality_flag')}",
        f"Image prediction margin:   {result.get('image_prediction_margin')}",
        "",
        "Text sentiment evidence",
        "-" * 72,
        f"Sentiment label:           {text_sentiment.get('sentiment_label')}",
        f"Sentiment score:           {safe_float(text_sentiment.get('sentiment_score')):.4f}",
        f"Sentiment method:          {text_sentiment.get('sentiment_method')}",
        f"Positive hits:             {text_sentiment.get('positive_count')}",
        f"Negative hits:             {text_sentiment.get('negative_count')}",
        f"Positive words:            {', '.join(text_sentiment.get('positive_hits', []))}",
        f"Negative words:            {', '.join(text_sentiment.get('negative_hits', []))}",
        "",
        "Keystroke scores",
        "-" * 72,
        format_scores_decimal(result.get("keystroke_scores", {})),
        "",
        "Text scores",
        "-" * 72,
        format_scores_decimal(result.get("text_scores", {})),
        "",
        "Audio scores",
        "-" * 72,
        format_scores_decimal(result.get("audio_scores", {})),
        "",
        "Image scores",
        "-" * 72,
        format_scores_decimal(result.get("image_scores", {})),
        "",
        "Image visual interpretation",
        "-" * 72,
    ]

    if result.get("image_generic_caption"):
        lines.extend(
            [
                "Generic image caption:",
                str(result.get("image_generic_caption")),
                "",
            ]
        )

    if result.get("image_behaviour_caption"):
        lines.extend(
            [
                "Behaviour-aware image interpretation:",
                str(result.get("image_behaviour_caption")),
                "",
            ]
        )

    lines.extend(
        [
            "Image visual cue summary:",
            format_visual_cue_summary(visual_summary),
            "",
            "Keystroke features",
            "-" * 72,
        ]
    )

    for name, value in keystroke_features.items():
        lines.append(f"  {name}: {value}")

    lines.extend(
        [
            "",
            "Evidence summary",
            "-" * 72,
            f"Typed text length:         {result.get('typed_text_length')} characters",
            f"Keystroke key presses:     {result.get('keystroke_keydown_count')}",
            f"Audio file:                {result.get('selected_audio_path') or 'not provided'}",
            f"Image file:                {result.get('selected_image_path') or 'not provided'}",
            "",
            "Raw diagnostic JSON",
            "-" * 72,
            safe_json_dump(result),
            "",
            "Methodological note",
            "-" * 72,
            str(result.get("methodological_note", "")),
        ]
    )

    return "\n".join(lines)


# ============================================================
# Prediction workflow
# ============================================================

def run_fusion_prediction() -> None:
    global latest_fusion_result
    global latest_summary_text
    global latest_technical_text

    typed_text = text_box.get("1.0", tk.END).strip()
    keydown_count = count_keydowns(events)

    output_box.delete("1.0", tk.END)
    output_box.insert(tk.END, "Running multimodal fusion analysis...\n\n")
    root.update_idletasks()

    if not typed_text:
        latest_fusion_result = None
        latest_summary_text = ""
        latest_technical_text = ""

        output_box.delete("1.0", tk.END)
        output_box.insert(tk.END, "Please type text before running fusion.\n")
        return

    # ========================================================
    # 1. Keystroke modality
    # ========================================================

    if keydown_count >= MIN_KEYDOWNS:
        try:
            keystroke_prediction, keystroke_scores, keystroke_features = (
                predict_keystroke_behaviour(events)
            )
            keystroke_scores = clean_scores(keystroke_scores)

        except Exception as error:
            keystroke_prediction = "unavailable"
            keystroke_scores = uniform_scores()
            keystroke_features = {"error": str(error)}
    else:
        keystroke_prediction = "insufficient_data"
        keystroke_scores = uniform_scores()
        keystroke_features = {
            "note": f"Need at least {MIN_KEYDOWNS} key presses.",
            "current_key_presses": keydown_count,
        }

    # ========================================================
    # 2. Text modality
    # ========================================================

    try:
        text_result = analyze_text(typed_text)
        text_scores = extract_text_scores(text_result)

        text_prediction = text_result.get(
            "predicted_behaviour",
            prediction_from_scores(text_scores),
        )

        text_confidence = float(
            text_result.get(
                "behaviour_confidence",
                best_score(text_scores),
            )
        )

        text_sentiment = {
            "sentiment_label": text_result.get("sentiment_label", "unknown"),
            "sentiment_score": text_result.get("sentiment_score", 0.0),
            "sentiment_method": text_result.get("sentiment_method", "unknown"),
            "positive_count": text_result.get("sentiment_positive_count", 0),
            "negative_count": text_result.get("sentiment_negative_count", 0),
            "positive_hits": text_result.get("sentiment_positive_hits", []),
            "negative_hits": text_result.get("sentiment_negative_hits", []),
        }

    except Exception as error:
        text_result = {
            "status": "error",
            "error": str(error),
        }
        text_scores = uniform_scores()
        text_prediction = "unavailable"
        text_confidence = 0.0
        text_sentiment = {
            "sentiment_label": "unavailable",
            "sentiment_score": 0.0,
            "sentiment_method": "unavailable",
            "positive_count": 0,
            "negative_count": 0,
            "positive_hits": [],
            "negative_hits": [],
        }

    # ========================================================
    # 3. Audio modality
    # ========================================================

    audio_result = {"status": "not_provided"}
    audio_scores = uniform_scores()
    audio_prediction = "not_provided"

    if selected_audio_path is not None:
        try:
            if not selected_audio_path.exists():
                raise FileNotFoundError(f"Audio file not found: {selected_audio_path}")

            if selected_audio_path.suffix.lower() not in SUPPORTED_AUDIO:
                raise ValueError(
                    f"Unsupported audio format: {selected_audio_path.suffix}"
                )

            audio_path = copy_to_uploads(selected_audio_path)
            audio_result = analyze_audio_file(str(audio_path))
            audio_result["status"] = "analyzed"
            audio_result["uploaded_path"] = str(audio_path)

            audio_scores = extract_audio_scores(audio_result)
            audio_prediction = audio_result.get(
                "predicted_label",
                audio_result.get(
                    "predicted_behaviour",
                    prediction_from_scores(audio_scores),
                ),
            )

        except Exception as error:
            audio_result = {
                "status": "error",
                "error": str(error),
            }
            audio_scores = uniform_scores()
            audio_prediction = "unavailable"

    # ========================================================
    # 4. Image modality
    # ========================================================

    image_result = {"status": "not_provided"}
    image_scores = uniform_scores()
    image_prediction = "not_provided"

    image_caption = ""
    image_generic_caption = ""
    image_behaviour_caption = ""
    image_visual_cue_summary: dict[str, Any] = {}
    image_reliability_score: Any = None
    image_quality_flag: Any = None
    image_prediction_margin: Any = None

    if selected_image_path is not None:
        try:
            if not selected_image_path.exists():
                raise FileNotFoundError(f"Image file not found: {selected_image_path}")

            if selected_image_path.suffix.lower() not in SUPPORTED_IMAGES:
                raise ValueError(
                    f"Unsupported image format: {selected_image_path.suffix}"
                )

            image_path = copy_to_uploads(selected_image_path)
            image_result = analyze_image_file(str(image_path))
            image_result["status"] = "analyzed"
            image_result["uploaded_path"] = str(image_path)

            image_caption = image_result.get("caption", "")
            image_generic_caption = get_generic_caption(image_result)
            image_behaviour_caption = get_behaviour_caption(image_result)
            image_visual_cue_summary = get_visual_cue_summary(image_result)

            image_reliability_score = image_result.get("reliability_score")
            image_quality_flag = image_result.get("quality_flag")
            image_prediction_margin = image_result.get("prediction_margin")

            image_scores = extract_image_scores(image_result)

            image_prediction = image_result.get(
                "predicted_label",
                image_result.get(
                    "predicted_behaviour",
                    prediction_from_scores(image_scores),
                ),
            )

        except Exception as error:
            image_result = {
                "status": "error",
                "error": str(error),
            }
            image_scores = uniform_scores()
            image_prediction = "unavailable"

    # ========================================================
    # 5. Fusion
    # ========================================================

    final_scores = fuse_predictions(
        keystroke_scores=keystroke_scores,
        text_scores=text_scores,
        audio_scores=audio_scores,
        image_scores=image_scores,
        use_trained_fusion=False,
    )

    final_scores = clean_scores(final_scores)
    final_prediction = prediction_from_scores(final_scores)
    final_confidence = best_score(final_scores)

    modality_scores = explain_fusion_inputs(
        keystroke_scores=keystroke_scores,
        text_scores=text_scores,
        audio_scores=audio_scores,
        image_scores=image_scores,
    )

    fusion_weights = get_fusion_weights()

    latest_fusion_result = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "modality": "multimodal_fusion",

        "final_prediction": final_prediction,
        "final_confidence": final_confidence,
        "final_scores": final_scores,

        "fusion_method": "weighted_late_fusion",
        "fusion_weights": fusion_weights,
        "trained_fusion_model_used": False,
        "modality_scores": modality_scores,

        "typed_text": typed_text,
        "typed_text_length": len(typed_text),
        "keystroke_keydown_count": keydown_count,

        "keystroke_prediction": keystroke_prediction,
        "keystroke_scores": keystroke_scores,
        "keystroke_features": keystroke_features,

        "text_prediction": text_prediction,
        "text_scores": text_scores,
        "text_confidence": text_confidence,
        "text_sentiment": text_sentiment,
        "text_result": text_result,

        "audio_prediction": audio_prediction,
        "audio_scores": audio_scores,
        "audio_result": audio_result,
        "selected_audio_path": str(selected_audio_path) if selected_audio_path else None,

        "image_prediction": image_prediction,
        "image_scores": image_scores,
        "image_result": image_result,
        "selected_image_path": str(selected_image_path) if selected_image_path else None,

        "image_caption": image_caption,
        "image_generic_caption": image_generic_caption,
        "image_behaviour_caption": image_behaviour_caption,
        "image_visual_cue_summary": image_visual_cue_summary,
        "image_reliability_score": image_reliability_score,
        "image_quality_flag": image_quality_flag,
        "image_prediction_margin": image_prediction_margin,

        "methodological_note": (
            "This GUI performs weighted late fusion across keystroke, text, "
            "audio, and image evidence. The text behavioural prediction is "
            "generated by the trained text_model.joblib pipeline, while text "
            "sentiment is displayed separately as supporting evidence. The image "
            "modality uses the upgraded vision pipeline with generic captioning, "
            "behaviour-aware visual interpretation, visual cue summaries, and "
            "reliability metadata. Weighted late fusion is used because the "
            "modality datasets are not necessarily session-aligned."
        ),
    }

    latest_fusion_result["brief_interpretation"] = build_brief_fusion_interpretation(
        latest_fusion_result
    )

    latest_summary_text = build_summary_text(latest_fusion_result)
    latest_technical_text = build_technical_text(latest_fusion_result)

    output_box.delete("1.0", tk.END)
    output_box.insert(tk.END, latest_summary_text)


# ============================================================
# Technical details
# ============================================================

def show_technical_details() -> None:
    result = require_latest_result()

    if result is None:
        return

    details_window = tk.Toplevel(root)
    details_window.title("Fusion Technical Details")
    details_window.geometry("1120x820")

    title = tk.Label(
        details_window,
        text="Fusion Technical Details",
        font=("Arial", 16, "bold"),
    )
    title.pack(pady=10)

    details_box = scrolledtext.ScrolledText(
        details_window,
        width=136,
        height=44,
        font=("Consolas", 10),
    )
    details_box.pack(padx=12, pady=10, fill=tk.BOTH, expand=True)

    details_box.insert(tk.END, latest_technical_text)
    details_box.configure(state="disabled")

    close_button = tk.Button(
        details_window,
        text="Close",
        command=details_window.destroy,
        width=16,
    )
    close_button.pack(pady=8)


# ============================================================
# Export functions
# ============================================================

def export_latest_result_json() -> None:
    result = require_latest_result()

    if result is None:
        return

    output_path = EXPORT_DIR / f"fusion_result_{timestamp_for_filename()}.json"

    export_payload = {
        **result,
        "presentation_summary": latest_summary_text,
        "technical_report": latest_technical_text,
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(export_payload, f, indent=4, default=str, ensure_ascii=False)

    messagebox.showinfo(
        "Export Complete",
        f"JSON result exported successfully:\n\n{output_path}",
    )


def export_latest_result_csv() -> None:
    result = require_latest_result()

    if result is None:
        return

    output_path = EXPORT_DIR / f"fusion_result_{timestamp_for_filename()}.csv"

    text_sentiment = result.get("text_sentiment", {})
    fusion_weights = result.get("fusion_weights", {})
    visual_summary = result.get("image_visual_cue_summary", {})

    row: dict[str, Any] = {
        "created_at": result.get("created_at"),
        "modality": result.get("modality"),

        "final_prediction": result.get("final_prediction"),
        "final_confidence": result.get("final_confidence"),
        "final_confidence_percent": format_percent(safe_float(result.get("final_confidence"))),
        "fusion_method": result.get("fusion_method"),
        "trained_fusion_model_used": result.get("trained_fusion_model_used"),

        "typed_text_length": result.get("typed_text_length"),
        "keystroke_keydown_count": result.get("keystroke_keydown_count"),

        "keystroke_prediction": result.get("keystroke_prediction"),
        "text_prediction": result.get("text_prediction"),
        "text_confidence": result.get("text_confidence"),
        "audio_prediction": result.get("audio_prediction"),
        "image_prediction": result.get("image_prediction"),

        "sentiment_label": text_sentiment.get("sentiment_label"),
        "sentiment_score": text_sentiment.get("sentiment_score"),
        "sentiment_method": text_sentiment.get("sentiment_method"),
        "sentiment_positive_count": text_sentiment.get("positive_count"),
        "sentiment_negative_count": text_sentiment.get("negative_count"),
        "sentiment_positive_hits": ", ".join(text_sentiment.get("positive_hits", [])),
        "sentiment_negative_hits": ", ".join(text_sentiment.get("negative_hits", [])),

        "selected_audio_path": result.get("selected_audio_path"),
        "selected_image_path": result.get("selected_image_path"),

        "image_caption": result.get("image_caption"),
        "image_generic_caption": result.get("image_generic_caption"),
        "image_behaviour_caption": result.get("image_behaviour_caption"),
        "image_reliability_score": result.get("image_reliability_score"),
        "image_quality_flag": result.get("image_quality_flag"),
        "image_prediction_margin": result.get("image_prediction_margin"),

        "brief_interpretation": result.get("brief_interpretation"),
    }

    if isinstance(visual_summary, dict):
        row.update(
            {
                "image_visual_final_label": visual_summary.get("final_visual_label"),
                "image_visual_final_score": visual_summary.get("final_visual_score"),
                "image_visual_final_margin": visual_summary.get("final_visual_margin"),
                "image_visual_evidence_strength": visual_summary.get("evidence_strength"),
                "image_full_image_cue": visual_summary.get("full_image_cue"),
                "image_scene_cue": visual_summary.get("scene_cue"),
                "image_face_cue": visual_summary.get("face_cue"),
                "image_body_cue": visual_summary.get("body_cue"),
                "image_face_detected": visual_summary.get("face_detected"),
                "image_body_detected": visual_summary.get("body_detected"),
                "image_posture_cue": visual_summary.get("posture_cue"),
                "image_head_position_cue": visual_summary.get("head_position_cue"),
                "image_supporting_cues": compact_list(visual_summary.get("supporting_cues")),
                "image_possible_conflicting_cues": compact_list(
                    visual_summary.get("possible_conflicting_cues")
                ),
                "image_absent_or_uncertain_cues": compact_list(
                    visual_summary.get("absent_or_uncertain_cues")
                ),
            }
        )

    for modality, weight in fusion_weights.items():
        row[f"fusion_weight_{modality}"] = weight

    row.update(flatten_scores(result.get("final_scores", {}), "final"))
    row.update(flatten_scores(result.get("keystroke_scores", {}), "keystroke"))
    row.update(flatten_scores(result.get("text_scores", {}), "text"))
    row.update(flatten_scores(result.get("audio_scores", {}), "audio"))
    row.update(flatten_scores(result.get("image_scores", {}), "image"))

    with open(output_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        writer.writeheader()
        writer.writerow(row)

    messagebox.showinfo(
        "Export Complete",
        f"CSV summary exported successfully:\n\n{output_path}",
    )


def export_latest_result_txt() -> None:
    result = require_latest_result()

    if result is None:
        return

    output_path = EXPORT_DIR / f"fusion_report_{timestamp_for_filename()}.txt"

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("SenseFuzeAI Live Multimodal Fusion Test Report\n")
        f.write("==============================================\n\n")

        f.write("Presentation Summary\n")
        f.write("--------------------\n")
        f.write(latest_summary_text)
        f.write("\n\n")

        f.write("Technical Diagnostic Details\n")
        f.write("----------------------------\n")
        f.write(latest_technical_text)
        f.write("\n")

    messagebox.showinfo(
        "Export Complete",
        f"Fusion report exported successfully:\n\n{output_path}",
    )


def show_export_guidance() -> None:
    messagebox.showinfo(
        "Export Guidance",
        "After running fusion prediction, you can export the latest result as:\n\n"
        "1. JSON: full diagnostic result, summary, and technical report\n"
        "2. CSV: compact one-row summary for spreadsheet analysis\n"
        "3. TXT: readable report containing both presentation summary and technical details\n\n"
        f"Exports are saved in:\n{EXPORT_DIR}",
    )


# ============================================================
# GUI actions
# ============================================================

def reset_session() -> None:
    global selected_audio_path
    global selected_image_path
    global latest_fusion_result
    global latest_summary_text
    global latest_technical_text

    events.clear()
    active_keys.clear()

    selected_audio_path = None
    selected_image_path = None

    latest_fusion_result = None
    latest_summary_text = ""
    latest_technical_text = ""

    text_box.delete("1.0", tk.END)
    output_box.delete("1.0", tk.END)

    audio_label.config(text="No audio file selected.")
    image_label.config(text="No image file selected.")

    output_box.insert(
        tk.END,
        "Session reset.\n"
        "Type naturally, optionally select audio/image files, then run fusion.\n",
    )


def show_guidance() -> None:
    messagebox.showinfo(
        "Fusion Test Guidance",
        "This tool tests keystroke + text + audio + image fusion outside the web UI.\n\n"
        "Recommended demo:\n"
        "1. Type a clear focused/distracted/fatigued/overloaded sentence.\n"
        "2. Type naturally for at least 20 key presses.\n"
        "3. Optionally choose matching audio and image files.\n"
        "4. Click Run Fusion Prediction.\n\n"
        "The main screen shows a concise demo-friendly result.\n"
        "Use 'Show Technical Details' to inspect the full modality evidence.",
    )


# ============================================================
# GUI construction
# ============================================================

root = tk.Tk()
root.title("Live Multimodal Fusion Test")
root.geometry("1160x940")

title_label = tk.Label(
    root,
    text="Live Multimodal Fusion Test",
    font=("Arial", 18, "bold"),
)
title_label.pack(pady=10)

instruction_label = tk.Label(
    root,
    text=(
        "Type naturally below. Optionally select an audio file and image file. "
        "The main output is simplified for prototype demonstration, while "
        "full multimodal diagnostics remain available separately."
    ),
    font=("Arial", 11),
    wraplength=1060,
)
instruction_label.pack(pady=5)

text_box = scrolledtext.ScrolledText(
    root,
    width=132,
    height=8,
    font=("Consolas", 10),
)
text_box.pack(padx=15, pady=10)

text_box.bind("<KeyPress>", on_key_press)
text_box.bind("<KeyRelease>", on_key_release)

file_frame = tk.Frame(root)
file_frame.pack(pady=5)

audio_button = tk.Button(
    file_frame,
    text="Choose Audio File",
    command=choose_audio_file,
    width=24,
)
audio_button.grid(row=0, column=0, padx=8, pady=4)

audio_label = tk.Label(
    file_frame,
    text="No audio file selected.",
    width=104,
    anchor="w",
    fg="gray",
)
audio_label.grid(row=0, column=1, padx=8, pady=4)

image_button = tk.Button(
    file_frame,
    text="Choose Image File",
    command=choose_image_file,
    width=24,
)
image_button.grid(row=1, column=0, padx=8, pady=4)

image_label = tk.Label(
    file_frame,
    text="No image file selected.",
    width=104,
    anchor="w",
    fg="gray",
)
image_label.grid(row=1, column=1, padx=8, pady=4)

button_frame = tk.Frame(root)
button_frame.pack(pady=8)

run_button = tk.Button(
    button_frame,
    text="Run Fusion Prediction",
    command=run_fusion_prediction,
    width=24,
)
run_button.grid(row=0, column=0, padx=6, pady=4)

technical_button = tk.Button(
    button_frame,
    text="Show Technical Details",
    command=show_technical_details,
    width=24,
)
technical_button.grid(row=0, column=1, padx=6, pady=4)

reset_button = tk.Button(
    button_frame,
    text="Reset",
    command=reset_session,
    width=14,
)
reset_button.grid(row=0, column=2, padx=6, pady=4)

guidance_button = tk.Button(
    button_frame,
    text="Guidance",
    command=show_guidance,
    width=14,
)
guidance_button.grid(row=0, column=3, padx=6, pady=4)

export_frame = tk.Frame(root)
export_frame.pack(pady=5)

export_json_button = tk.Button(
    export_frame,
    text="Export JSON",
    command=export_latest_result_json,
    width=18,
)
export_json_button.grid(row=0, column=0, padx=6, pady=4)

export_csv_button = tk.Button(
    export_frame,
    text="Export CSV",
    command=export_latest_result_csv,
    width=18,
)
export_csv_button.grid(row=0, column=1, padx=6, pady=4)

export_txt_button = tk.Button(
    export_frame,
    text="Export Report",
    command=export_latest_result_txt,
    width=18,
)
export_txt_button.grid(row=0, column=2, padx=6, pady=4)

export_help_button = tk.Button(
    export_frame,
    text="Export Help",
    command=show_export_guidance,
    width=18,
)
export_help_button.grid(row=0, column=3, padx=6, pady=4)

output_box = scrolledtext.ScrolledText(
    root,
    width=132,
    height=29,
    font=("Consolas", 9),
)
output_box.pack(padx=15, pady=10)

output_box.insert(
    tk.END,
    "Waiting for multimodal input...\n\n"
    "Steps:\n"
    "1. Type naturally in the text box.\n"
    "2. Select optional audio/image files.\n"
    "3. Click 'Run Fusion Prediction'.\n"
    "4. The main output will show a concise result suitable for video demonstration.\n"
    "5. Click 'Show Technical Details' to inspect full modality evidence.\n"
    "6. Export the latest result as JSON, CSV, or TXT report if needed.\n\n"
    f"Minimum recommended key presses: {MIN_KEYDOWNS}\n\n"
    "Text modality note:\n"
    "The text behavioural prediction uses model_artifacts/text_model.joblib. "
    "Text sentiment is displayed separately as supporting evidence.\n\n"
    "Image modality note:\n"
    "The image behavioural prediction uses the upgraded vision pipeline with "
    "generic captioning, behaviour-aware visual interpretation, visual cue "
    "summary, reliability score, quality flag, and prediction margin.\n\n"
    f"Export directory:\n{EXPORT_DIR}\n",
)

root.mainloop()
