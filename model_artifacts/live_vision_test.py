# === model_artifacts/live_vision_test.py ===
# SenseFuzeAI - Live Vision Behaviour Test GUI
#
# Presentation-first version:
#   - main output shows concise prototype-friendly result
#   - technical details are available through "Show Technical Details"
#   - exports include both summary and full diagnostic information
#
# Tests the synchronised image / vision modality.
#
# Displays:
#   - final visual behaviour prediction
#   - behaviour confidence
#   - behaviour-aware caption summary
#   - image preview
#   - technical visual cue evidence on demand
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
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext

from PIL import Image, ImageTk


# ============================================================
# Project path fix
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from app.models.image_model import analyze_image_file


# ============================================================
# Configuration
# ============================================================

UPLOAD_DIR = PROJECT_ROOT / "data" / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

EXPORT_DIR = PROJECT_ROOT / "model_artifacts" / "exported_results"
EXPORT_DIR.mkdir(parents=True, exist_ok=True)

SUPPORTED_IMAGES = {
    ".jpg",
    ".jpeg",
    ".png",
    ".webp",
}

BEHAVIOUR_CLASSES = [
    "focused",
    "distracted",
    "fatigued",
    "overloaded",
]


# ============================================================
# Runtime state
# ============================================================

selected_image_path: Path | None = None
preview_image_ref = None

latest_vision_result: Dict[str, Any] | None = None
latest_summary_text: str = ""
latest_technical_text: str = ""


# ============================================================
# Generic helpers
# ============================================================

def timestamp_for_filename() -> str:
    """
    Return a filesystem-safe timestamp.
    """
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def safe_float(value: Any, default: float = 0.0) -> float:
    """
    Safely convert a value to float.
    """
    try:
        return float(value)
    except Exception:
        return default


def format_percent(value: float) -> str:
    """
    Format probability value as a percentage.
    """
    safe_value = max(0.0, min(1.0, float(value)))
    return f"{safe_value * 100:.2f}%"


def safe_json_dump(data: Dict[str, Any]) -> str:
    """
    Dump JSON safely for nested diagnostic payloads.
    """
    return json.dumps(data, indent=2, default=str, ensure_ascii=False)


def require_latest_result() -> Dict[str, Any] | None:
    """
    Ensure a latest vision result exists before export/details.
    """
    if latest_vision_result is None:
        messagebox.showwarning(
            "No Result Available",
            "No vision analysis result is available yet.\n\n"
            "Please run a vision prediction first.",
        )
        return None

    return latest_vision_result


# ============================================================
# Score and result extraction helpers
# ============================================================

def normalise_scores(scores: Dict[str, float] | None) -> Dict[str, float]:
    """
    Ensure behaviour scores are non-negative and sum to 1.
    """
    if not scores:
        return {
            label: 1.0 / len(BEHAVIOUR_CLASSES)
            for label in BEHAVIOUR_CLASSES
        }

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


def get_best_label(scores: Dict[str, float]) -> str:
    """
    Return class with highest score.
    """
    scores = normalise_scores(scores)
    return max(scores, key=scores.get)


def extract_scores(image_result: Dict[str, Any]) -> Dict[str, float]:
    """
    Extract behaviour scores from multiple possible image result formats.
    """
    fused_prediction = image_result.get("fused_prediction", {})

    scores = (
        image_result.get("behaviour_scores")
        or image_result.get("scores")
        or image_result.get("class_probabilities")
        or fused_prediction.get("visual_fused_scores")
        or {}
    )

    return normalise_scores(scores)


def get_generic_caption(image_result: Dict[str, Any]) -> str:
    """
    Extract generic BLIP caption.
    """
    return (
        image_result.get("generic_caption")
        or image_result.get("caption")
        or image_result.get("scene_description")
        or ""
    )


def get_behaviour_caption(image_result: Dict[str, Any]) -> str:
    """
    Extract behaviour-aware caption / interpretation.
    """
    return (
        image_result.get("behaviour_caption")
        or image_result.get("behaviour_aware_caption")
        or image_result.get("behaviour_description")
        or ""
    )


def get_visual_cue_summary(image_result: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract visual cue summary as dictionary.
    """
    summary = image_result.get("visual_cue_summary", {})

    if isinstance(summary, dict):
        return summary

    if isinstance(summary, str):
        try:
            return json.loads(summary)
        except Exception:
            return {}

    return {}


def format_scores_percent(scores: Dict[str, float]) -> str:
    """
    Format behaviour scores as percentages for presentation output.
    """
    scores = normalise_scores(scores)
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


def format_scores_decimal(scores: Dict[str, float]) -> str:
    """
    Format behaviour scores as decimal probabilities.
    """
    scores = normalise_scores(scores)

    lines = []

    for label, score in sorted(
        scores.items(),
        key=lambda item: item[1],
        reverse=True,
    ):
        lines.append(f"  {label}: {score:.4f}")

    return "\n".join(lines)


def format_optional_prediction_block(
    title: str,
    prediction: Dict[str, Any] | None,
) -> str:
    """
    Format CLIP sub-model prediction block.
    """
    if prediction is None:
        return f"{title}: not_detected\n"

    lines = [
        f"{title}: {prediction.get('best_label', 'unknown')}",
        f"  score:  {safe_float(prediction.get('best_score')):.4f}",
        f"  margin: {safe_float(prediction.get('margin')):.4f}",
    ]

    return "\n".join(lines) + "\n"


# ============================================================
# File and preview helpers
# ============================================================

def copy_image_to_uploads(path: Path) -> Path:
    """
    Copy selected image into project uploads folder.
    """
    safe_name = path.name.replace(" ", "_")
    destination = UPLOAD_DIR / safe_name

    if destination.resolve() != path.resolve():
        shutil.copy2(path, destination)

    return destination


def load_preview(path: Path) -> None:
    """
    Load selected image preview.
    """
    global preview_image_ref

    try:
        image = Image.open(path).convert("RGB")
        image.thumbnail((430, 310))

        preview_image_ref = ImageTk.PhotoImage(image)

        preview_label.config(image=preview_image_ref, text="")
        preview_label.image = preview_image_ref

    except Exception as error:
        preview_label.config(
            image="",
            text=f"Could not preview image.\n{error}",
        )
        preview_label.image = None


# ============================================================
# Presentation interpretation
# ============================================================

def build_brief_interpretation(result: Dict[str, Any]) -> str:
    """
    Build concise, demo-friendly interpretation.
    """
    prediction = str(result.get("prediction", "unknown")).lower()
    cue_summary = result.get("visual_cue_summary", {})
    behaviour_caption = str(result.get("behaviour_caption", "")).strip()

    supporting_cues = cue_summary.get("supporting_cues", [])
    conflicting_cues = cue_summary.get("possible_conflicting_cues", [])

    if prediction == "focused":
        base = (
            "The system classifies the image as focused because the visual evidence "
            "suggests task engagement, workstation context, and stable posture."
        )
    elif prediction == "distracted":
        base = (
            "The system classifies the image as distracted because the visual evidence "
            "suggests attention diversion, phone/device interruption, or competing activity."
        )
    elif prediction == "fatigued":
        base = (
            "The system classifies the image as fatigued because the visual evidence "
            "suggests tiredness, reduced alertness, low head position, or fatigue-related posture."
        )
    elif prediction == "overloaded":
        base = (
            "The system classifies the image as overloaded because the visual evidence "
            "suggests workload pressure, stress, clutter, or high task demand."
        )
    else:
        base = (
            "The system could not identify a strong visual behavioural cue. "
            "The image should be interpreted as supporting evidence only."
        )

    if supporting_cues:
        top_cues = "; ".join(str(cue) for cue in supporting_cues[:3])
        base += f"\n\nKey supporting cues: {top_cues}."

    if conflicting_cues:
        top_conflicts = "; ".join(str(cue) for cue in conflicting_cues[:2])
        base += f"\n\nPossible conflicting cues: {top_conflicts}."

    if not supporting_cues and behaviour_caption:
        base += f"\n\nBehaviour-aware interpretation: {behaviour_caption}"

    return base


def build_summary_text(result: Dict[str, Any]) -> str:
    """
    Build presentation-friendly main output.
    """
    prediction = str(result.get("prediction", "unknown"))
    confidence = safe_float(result.get("confidence"))
    margin = safe_float(result.get("prediction_margin"))
    reliability_score = safe_float(result.get("reliability_score"))
    quality_flag = result.get("quality_flag", "unknown")
    scores = normalise_scores(result.get("behaviour_scores", {}))

    generic_caption = result.get("generic_caption") or "No generic caption returned."
    behaviour_caption = result.get("behaviour_caption") or "No behaviour-aware caption returned."
    interpretation = result.get("brief_interpretation", "")

    display_prediction = prediction.upper()

    lines = [
        "Live Vision Behaviour Test",
        "=" * 50,
        "",
        f"Final Prediction: {display_prediction}",
        f"Confidence: {format_percent(confidence)}",
        f"Reliability: {format_percent(reliability_score)}",
        f"Quality Flag: {quality_flag}",
        "",
        "Final Behaviour Probabilities",
        "-" * 50,
        format_scores_percent(scores),
        "",
        "Brief Interpretation",
        "-" * 50,
        interpretation,
        "",
        "Caption Evidence",
        "-" * 50,
        f"Generic caption: {generic_caption}",
        "",
        "Behaviour-aware caption:",
        behaviour_caption,
        "",
        "Visual Reliability",
        "-" * 50,
        f"Prediction margin: {margin:.4f}",
        f"Reliability score: {reliability_score:.4f}",
        f"Quality flag: {quality_flag}",
        "",
        "Technical Details",
        "-" * 50,
        "Full CLIP sub-model predictions, visual cue summary,",
        "MediaPipe detection metadata, and diagnostic JSON are available",
        "through the 'Show Technical Details' button and export functions.",
        "",
        "Report Export",
        "-" * 50,
        "Full result available as JSON, CSV, or TXT report.",
    ]

    return "\n".join(lines)


def build_technical_text(result: Dict[str, Any]) -> str:
    """
    Build detailed diagnostic output.
    """
    image_result = result.get("raw_image_result", {})
    scores = normalise_scores(result.get("behaviour_scores", {}))
    cue_summary = result.get("visual_cue_summary", {})

    behaviour_prediction = image_result.get("behaviour_prediction")
    scene_prediction = image_result.get("scene_prediction")
    face_prediction = image_result.get("face_prediction")
    body_prediction = image_result.get("body_prediction")

    lines = [
        "Live Vision Behaviour Test - Technical Details",
        "=" * 72,
        "",
        f"Created at:              {result.get('created_at')}",
        f"Modality:                {result.get('modality')}",
        f"Selected image path:     {result.get('selected_image_path')}",
        f"Copied image path:       {result.get('copied_image_path')}",
        "",
        "Prediction",
        "-" * 72,
        f"Predicted behaviour:     {result.get('prediction')}",
        f"Confidence:              {safe_float(result.get('confidence')):.4f}",
        f"Prediction margin:       {safe_float(result.get('prediction_margin')):.4f}",
        f"Reliability score:       {safe_float(result.get('reliability_score')):.4f}",
        f"Quality flag:            {result.get('quality_flag')}",
        "",
        "Behaviour scores",
        "-" * 72,
        format_scores_decimal(scores),
        "",
        "Generic visual caption",
        "-" * 72,
        str(result.get("generic_caption", "")),
        "",
        "Behaviour-aware visual interpretation",
        "-" * 72,
        str(result.get("behaviour_caption", "")),
        "",
        "Visual cue summary",
        "-" * 72,
    ]

    if cue_summary:
        for key, value in cue_summary.items():
            lines.append(f"  {key}: {value}")
    else:
        lines.append("  No visual cue summary returned.")

    lines.extend(
        [
            "",
            "Visual sub-model predictions",
            "-" * 72,
            format_optional_prediction_block(
                "Full-image behaviour cue",
                behaviour_prediction,
            ).rstrip(),
            format_optional_prediction_block(
                "Scene cue",
                scene_prediction,
            ).rstrip(),
            format_optional_prediction_block(
                "Face cue",
                face_prediction,
            ).rstrip(),
            format_optional_prediction_block(
                "Body cue",
                body_prediction,
            ).rstrip(),
            "",
            "Detection metadata",
            "-" * 72,
            f"  face_detected:       {image_result.get('face_detected')}",
            f"  face_confidence:     {safe_float(image_result.get('face_confidence')):.4f}",
            f"  body_detected:       {image_result.get('body_detected')}",
            f"  pose_visibility:     {safe_float(image_result.get('pose_visibility')):.4f}",
            f"  shoulder_visibility: {safe_float(image_result.get('shoulder_visibility')):.4f}",
            f"  posture_cue:         {image_result.get('posture_cue')}",
            f"  head_position_cue:   {image_result.get('head_position_cue')}",
            "",
            "Full diagnostic JSON",
            "-" * 72,
            safe_json_dump(image_result),
            "",
            "Methodological note",
            "-" * 72,
            str(result.get("methodological_note", "")),
        ]
    )

    return "\n".join(lines)


# ============================================================
# Image analysis
# ============================================================

def run_vision_prediction() -> None:
    """
    Run vision analysis and display concise demo-friendly result.
    """
    global selected_image_path
    global latest_vision_result
    global latest_summary_text
    global latest_technical_text

    output_box.delete("1.0", tk.END)

    if selected_image_path is None:
        output_box.insert(tk.END, "No image file selected.\n")
        return

    if not selected_image_path.exists():
        output_box.insert(
            tk.END,
            f"Image file not found:\n{selected_image_path}\n",
        )
        return

    if selected_image_path.suffix.lower() not in SUPPORTED_IMAGES:
        output_box.insert(
            tk.END,
            f"Unsupported image format: {selected_image_path.suffix}\n"
            f"Supported formats: {', '.join(sorted(SUPPORTED_IMAGES))}\n",
        )
        return

    try:
        output_box.insert(tk.END, "Analysing image file...\n\n")
        output_box.insert(
            tk.END,
            "The first run may take longer because CLIP / BLIP / MediaPipe models are loading.\n",
        )
        root.update_idletasks()

        copied_image_path = copy_image_to_uploads(selected_image_path)
        image_result = analyze_image_file(str(copied_image_path))

        generic_caption = get_generic_caption(image_result)
        behaviour_caption = get_behaviour_caption(image_result)
        behaviour_scores = extract_scores(image_result)
        visual_cue_summary = get_visual_cue_summary(image_result)

        prediction = (
            image_result.get("predicted_label")
            or image_result.get("predicted_behaviour")
            or get_best_label(behaviour_scores)
        )

        confidence = safe_float(
            image_result.get(
                "prediction_score",
                behaviour_scores.get(prediction, max(behaviour_scores.values())),
            )
        )

        prediction_margin = safe_float(image_result.get("prediction_margin"))
        reliability_score = safe_float(image_result.get("reliability_score"))
        quality_flag = image_result.get("quality_flag", "unknown")

        latest_vision_result = {
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "modality": "vision",
            "selected_image_path": str(selected_image_path),
            "copied_image_path": str(copied_image_path),
            "prediction": prediction,
            "confidence": confidence,
            "prediction_margin": prediction_margin,
            "reliability_score": reliability_score,
            "quality_flag": quality_flag,
            "behaviour_scores": behaviour_scores,
            "generic_caption": generic_caption,
            "behaviour_caption": behaviour_caption,
            "visual_cue_summary": visual_cue_summary,
            "raw_image_result": image_result,
            "supported_classes": BEHAVIOUR_CLASSES,
            "methodological_note": (
                "This vision test uses CLIP zero-shot visual prompting, BLIP "
                "captioning, MediaPipe face/body detection, and a behaviour-aware "
                "caption layer. It estimates behavioural state from visible cues "
                "such as posture, attention, desk clutter, lighting, device use, "
                "and workspace context. It should be interpreted as one evidence "
                "source within the wider multimodal system."
            ),
        }

        latest_vision_result["brief_interpretation"] = build_brief_interpretation(
            latest_vision_result
        )

        latest_summary_text = build_summary_text(latest_vision_result)
        latest_technical_text = build_technical_text(latest_vision_result)

        output_box.delete("1.0", tk.END)
        output_box.insert(tk.END, latest_summary_text)

    except Exception as error:
        latest_vision_result = None
        latest_summary_text = ""
        latest_technical_text = ""

        output_box.delete("1.0", tk.END)
        output_box.insert(tk.END, "Image analysis failed.\n\n")
        output_box.insert(tk.END, str(error))
        output_box.insert(
            tk.END,
            "\n\nChecklist:\n"
            "1. Confirm the image file is valid.\n"
            "2. Confirm required packages are installed: torch, transformers, pillow, numpy, mediapipe.\n"
            "3. Confirm local models exist in the models/ directory.\n"
            "4. First run may take longer because CLIP / BLIP / MediaPipe models need to load.\n",
        )


# ============================================================
# Technical details
# ============================================================

def show_technical_details() -> None:
    """
    Open technical details in a separate window.
    """
    result = require_latest_result()

    if result is None:
        return

    details_window = tk.Toplevel(root)
    details_window.title("Vision Technical Details")
    details_window.geometry("1080x800")

    title = tk.Label(
        details_window,
        text="Vision Technical Details",
        font=("Arial", 16, "bold"),
    )
    title.pack(pady=10)

    details_box = scrolledtext.ScrolledText(
        details_window,
        width=130,
        height=42,
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
    """
    Export latest vision result as JSON.
    """
    result = require_latest_result()

    if result is None:
        return

    output_path = EXPORT_DIR / f"vision_result_{timestamp_for_filename()}.json"

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
    """
    Export latest vision result as one-row CSV summary.
    """
    result = require_latest_result()

    if result is None:
        return

    output_path = EXPORT_DIR / f"vision_result_{timestamp_for_filename()}.csv"

    scores = normalise_scores(result.get("behaviour_scores", {}))
    cue_summary = result.get("visual_cue_summary", {})

    supporting_cues = cue_summary.get("supporting_cues", [])
    conflicting_cues = cue_summary.get("possible_conflicting_cues", [])
    uncertain_cues = cue_summary.get("absent_or_uncertain_cues", [])

    row: Dict[str, Any] = {
        "created_at": result.get("created_at"),
        "modality": result.get("modality"),
        "selected_image_path": result.get("selected_image_path"),
        "copied_image_path": result.get("copied_image_path"),
        "prediction": result.get("prediction"),
        "confidence": result.get("confidence"),
        "confidence_percent": format_percent(safe_float(result.get("confidence"))),
        "prediction_margin": result.get("prediction_margin"),
        "reliability_score": result.get("reliability_score"),
        "quality_flag": result.get("quality_flag"),
        "generic_caption": result.get("generic_caption"),
        "behaviour_caption": result.get("behaviour_caption"),
        "brief_interpretation": result.get("brief_interpretation"),
        "supporting_cues": " | ".join(str(item) for item in supporting_cues),
        "possible_conflicting_cues": " | ".join(str(item) for item in conflicting_cues),
        "absent_or_uncertain_cues": " | ".join(str(item) for item in uncertain_cues),
    }

    for label in BEHAVIOUR_CLASSES:
        row[f"prob_{label}"] = scores.get(label, 0.0)

    raw = result.get("raw_image_result", {})

    row.update(
        {
            "face_detected": raw.get("face_detected"),
            "face_confidence": raw.get("face_confidence"),
            "body_detected": raw.get("body_detected"),
            "pose_visibility": raw.get("pose_visibility"),
            "shoulder_visibility": raw.get("shoulder_visibility"),
            "posture_cue": raw.get("posture_cue"),
            "head_position_cue": raw.get("head_position_cue"),
            "method": raw.get("method"),
        }
    )

    with open(output_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        writer.writeheader()
        writer.writerow(row)

    messagebox.showinfo(
        "Export Complete",
        f"CSV summary exported successfully:\n\n{output_path}",
    )


def export_latest_result_txt() -> None:
    """
    Export latest vision result as human-readable TXT report.
    """
    result = require_latest_result()

    if result is None:
        return

    output_path = EXPORT_DIR / f"vision_report_{timestamp_for_filename()}.txt"

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("SenseFuzeAI Live Vision Behaviour Test Report\n")
        f.write("=============================================\n\n")

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
        f"Text report exported successfully:\n\n{output_path}",
    )


# ============================================================
# GUI actions
# ============================================================

def choose_image_file() -> None:
    """
    Select image file.
    """
    global selected_image_path
    global latest_vision_result
    global latest_summary_text
    global latest_technical_text

    file_path = filedialog.askopenfilename(
        title="Select an image file",
        filetypes=[
            ("Image files", "*.jpg *.jpeg *.png *.webp"),
            ("All files", "*.*"),
        ],
    )

    if not file_path:
        return

    selected_image_path = Path(file_path)
    latest_vision_result = None
    latest_summary_text = ""
    latest_technical_text = ""

    selected_file_label.config(text=str(selected_image_path))
    load_preview(selected_image_path)

    output_box.delete("1.0", tk.END)
    output_box.insert(
        tk.END,
        "Image file selected.\n"
        "Click 'Predict Vision Behaviour' to analyse it.\n\n"
        f"{selected_image_path}\n",
    )


def reset_session() -> None:
    """
    Reset GUI state.
    """
    global selected_image_path
    global preview_image_ref
    global latest_vision_result
    global latest_summary_text
    global latest_technical_text

    selected_image_path = None
    preview_image_ref = None
    latest_vision_result = None
    latest_summary_text = ""
    latest_technical_text = ""

    selected_file_label.config(text="No image file selected.")
    preview_label.config(image="", text="Image preview will appear here.")
    preview_label.image = None

    output_box.delete("1.0", tk.END)
    output_box.insert(
        tk.END,
        "Session reset. Select an image file to begin.\n",
    )


def show_guidance() -> None:
    """
    Show vision test guidance.
    """
    messagebox.showinfo(
        "Vision Test Guidance",
        "Use this tool to test the image/vision modality independently.\n\n"
        "Recommended examples:\n\n"
        "Focused:\n"
        "- tidy workspace\n"
        "- attentive person\n"
        "- upright posture\n"
        "- person facing laptop or screen\n\n"
        "Distracted:\n"
        "- phone use\n"
        "- eyes or face looking away\n"
        "- multitasking or interruptions\n\n"
        "Fatigued:\n"
        "- tired posture\n"
        "- low head position\n"
        "- dim lighting\n"
        "- head resting on hand\n\n"
        "Overloaded:\n"
        "- cluttered desk\n"
        "- many papers or screens\n"
        "- tense posture\n"
        "- chaotic workspace\n\n"
        "The main screen shows a concise demo-friendly result. "
        "Use 'Show Technical Details' to inspect CLIP, BLIP, MediaPipe, "
        "visual cue summary, and full diagnostic output.",
    )


def show_export_guidance() -> None:
    """
    Explain export functionality.
    """
    messagebox.showinfo(
        "Export Guidance",
        "After running a vision prediction, you can export the latest result as:\n\n"
        "1. JSON: full diagnostic result, summary, and technical report\n"
        "2. CSV: compact one-row summary for spreadsheet analysis\n"
        "3. TXT: readable report containing both presentation summary and technical details\n\n"
        f"Exports are saved in:\n{EXPORT_DIR}",
    )


# ============================================================
# GUI construction
# ============================================================

root = tk.Tk()
root.title("Live Vision Behaviour Test")
root.geometry("1080x900")

title_label = tk.Label(
    root,
    text="Live Vision Behaviour Test",
    font=("Arial", 18, "bold"),
)
title_label.pack(pady=10)

instruction_label = tk.Label(
    root,
    text=(
        "Select a workspace image from your dataset or uploads folder. "
        "The main output is simplified for prototype demonstration, while "
        "full visual diagnostics remain available separately."
    ),
    font=("Arial", 11),
    wraplength=980,
)
instruction_label.pack(pady=5)

selected_file_label = tk.Label(
    root,
    text="No image file selected.",
    font=("Arial", 10),
    fg="gray",
    wraplength=980,
)
selected_file_label.pack(pady=8)

preview_frame = tk.Frame(
    root,
    width=470,
    height=330,
    bd=1,
    relief=tk.SOLID,
)
preview_frame.pack(pady=8)
preview_frame.pack_propagate(False)

preview_label = tk.Label(
    preview_frame,
    text="Image preview will appear here.",
    font=("Arial", 10),
    fg="gray",
)
preview_label.pack(expand=True)

button_frame = tk.Frame(root)
button_frame.pack(pady=8)

choose_button = tk.Button(
    button_frame,
    text="Choose Image File",
    command=choose_image_file,
    width=22,
)
choose_button.grid(row=0, column=0, padx=6, pady=4)

predict_button = tk.Button(
    button_frame,
    text="Predict Vision Behaviour",
    command=run_vision_prediction,
    width=26,
)
predict_button.grid(row=0, column=1, padx=6, pady=4)

technical_button = tk.Button(
    button_frame,
    text="Show Technical Details",
    command=show_technical_details,
    width=24,
)
technical_button.grid(row=0, column=2, padx=6, pady=4)

reset_button = tk.Button(
    button_frame,
    text="Reset",
    command=reset_session,
    width=14,
)
reset_button.grid(row=0, column=3, padx=6, pady=4)

help_button = tk.Button(
    button_frame,
    text="Guidance",
    command=show_guidance,
    width=14,
)
help_button.grid(row=0, column=4, padx=6, pady=4)

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
    width=124,
    height=24,
    font=("Consolas", 10),
)
output_box.pack(padx=15, pady=15)

output_box.insert(
    tk.END,
    "Waiting for image input...\n\n"
    "Steps:\n"
    "1. Click 'Choose Image File'.\n"
    "2. Select a .jpg, .jpeg, .png, or .webp file.\n"
    "3. Click 'Predict Vision Behaviour'.\n"
    "4. The main output will show a concise result suitable for video demonstration.\n"
    "5. Click 'Show Technical Details' to inspect visual cue summary, CLIP sub-models, "
    "MediaPipe metadata, and diagnostic JSON.\n"
    "6. Export the latest result as JSON, CSV, or TXT report if needed.\n\n"
    f"Export directory:\n{EXPORT_DIR}\n\n"
    "Note: the first run may take longer because CLIP / BLIP / MediaPipe models need to load.\n",
)

root.mainloop()
