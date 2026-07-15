# === model_artifacts/live_audio_test.py ===
# SenseFuzeAI - Live Audio Behaviour Test GUI
#
# Presentation-first version:
#   - main output shows concise prototype-friendly result
#   - technical details are available through "Show Technical Details"
#   - exports include both summary and full diagnostic information
#
# Supports:
#   - audio file selection
#   - live audio behaviour prediction
#   - concise demo output
#   - technical diagnostic output
#   - YAMNet / audio label display
#   - extracted audio feature display
#   - export latest result as JSON, CSV, and TXT report

from __future__ import annotations

import csv
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import tkinter as tk
from tkinter import filedialog, scrolledtext, messagebox


# ============================================================
# Project path setup
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from app.models.audio_models import analyze_audio_file
from app.fusion import audio_to_behaviour_scores


# ============================================================
# Configuration
# ============================================================

UPLOAD_DIR = PROJECT_ROOT / "data" / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

EXPORT_DIR = PROJECT_ROOT / "model_artifacts" / "exported_results"
EXPORT_DIR.mkdir(parents=True, exist_ok=True)

SUPPORTED_AUDIO = {
    ".wav",
    ".mp3",
    ".m4a",
    ".flac",
    ".ogg",
    ".aac",
}

BEHAVIOUR_CLASSES = [
    "focused",
    "distracted",
    "fatigued",
    "overloaded",
]

selected_audio_path: Path | None = None

latest_audio_result: dict[str, Any] | None = None
latest_summary_text: str = ""
latest_technical_text: str = ""


# ============================================================
# Helper functions
# ============================================================

def timestamp_for_filename() -> str:
    """
    Return a filesystem-safe timestamp.
    """
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def normalise_scores(scores: dict[str, float] | None) -> dict[str, float]:
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


def get_best_label(scores: dict[str, float]) -> str:
    """
    Return the behaviour label with the highest score.
    """
    scores = normalise_scores(scores)
    return max(scores, key=scores.get)


def copy_audio_to_uploads(path: Path) -> Path:
    """
    Copy selected audio file into the project uploads folder.
    """
    safe_name = path.name.replace(" ", "_")
    destination = UPLOAD_DIR / safe_name

    if destination.resolve() != path.resolve():
        shutil.copy2(path, destination)

    return destination


def extract_audio_outputs(
    audio_result: dict[str, Any],
) -> tuple[list[Any], dict[str, Any], dict[str, float] | None, str | None]:
    """
    Handle both older and upgraded audio result formats.

    Older possible format:
        labels
        features

    Newer possible format:
        yamnet_labels
        audio_features
        behaviour_scores
        predicted_label
    """
    labels = (
        audio_result.get("yamnet_labels")
        or audio_result.get("labels")
        or []
    )

    features = (
        audio_result.get("audio_features")
        or audio_result.get("features")
        or {}
    )

    model_scores = (
        audio_result.get("behaviour_scores")
        or audio_result.get("scores")
        or audio_result.get("class_probabilities")
        or None
    )

    predicted_label = (
        audio_result.get("predicted_label")
        or audio_result.get("predicted_behaviour")
        or None
    )

    return labels, features, model_scores, predicted_label


def safe_float(value: Any) -> float | None:
    """
    Convert a value to float if possible.
    """
    try:
        return float(value)
    except Exception:
        return None


def format_percent(value: float) -> str:
    """
    Convert probability value into percentage string.
    """
    safe = max(0.0, min(1.0, float(value)))
    return f"{safe * 100:.2f}%"


def format_scores_decimal(scores: dict[str, float]) -> str:
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


def format_scores_percent(scores: dict[str, float]) -> str:
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


def format_audio_labels(labels: list[Any], limit: int | None = None) -> str:
    """
    Format YAMNet / audio labels.
    """
    if not labels:
        return "  No audio labels returned."

    items = labels[:limit] if limit is not None else labels
    lines = []

    for item in items:
        try:
            label, score = item
            lines.append(f"  {label}: {float(score):.4f}")
        except Exception:
            lines.append(f"  {item}")

    return "\n".join(lines)


def format_audio_labels_short(labels: list[Any], limit: int = 3) -> str:
    """
    Format top audio labels in a short presentation-friendly form.
    """
    if not labels:
        return "No audio labels returned."

    parts = []

    for item in labels[:limit]:
        try:
            label, score = item
            parts.append(f"{label} ({float(score):.3f})")
        except Exception:
            parts.append(str(item))

    return ", ".join(parts)


def require_latest_result() -> dict[str, Any] | None:
    """
    Ensure that an audio result exists before exporting.
    """
    if latest_audio_result is None:
        messagebox.showwarning(
            "No Result Available",
            "No audio analysis result is available yet.\n\n"
            "Please run an audio prediction first.",
        )
        return None

    return latest_audio_result


# ============================================================
# Presentation interpretation
# ============================================================

def build_brief_interpretation(
    prediction: str,
    labels: list[Any],
    score_source: str,
) -> str:
    """
    Build concise, demo-friendly interpretation text.
    """
    label_text = " ".join(str(item[0]).lower() if isinstance(item, (list, tuple)) and item else str(item).lower() for item in labels)

    if prediction == "focused":
        if any(word in label_text for word in ["silence", "quiet", "library", "inside", "room"]):
            return (
                "The system classifies the audio context as focused because the "
                "detected sound profile suggests a quiet or stable environment."
            )

        return (
            "The system classifies the audio context as focused because the "
            "available acoustic evidence is most consistent with a calm work setting."
        )

    if prediction == "distracted":
        if any(word in label_text for word in ["speech", "conversation", "laughter", "crowd", "phone", "music"]):
            return (
                "The system classifies the audio context as distracted because "
                "the detected sounds suggest speech, conversation, or competing activity."
            )

        return (
            "The system classifies the audio context as distracted because the "
            "sound evidence may indicate interruptions or competing acoustic cues."
        )

    if prediction == "fatigued":
        if any(word in label_text for word in ["breathing", "snoring", "yawn", "sleep", "sigh"]):
            return (
                "The system classifies the audio context as fatigued because "
                "the detected sound profile may indicate tiredness or low-energy cues."
            )

        return (
            "The system classifies the audio context as fatigued because the "
            "acoustic evidence is most consistent with reduced alertness."
        )

    if prediction == "overloaded":
        if any(word in label_text for word in ["alarm", "shout", "scream", "traffic", "engine", "construction", "bang", "noise"]):
            return (
                "The system classifies the audio context as overloaded because "
                "the detected sounds suggest noisy, stressful, or high-pressure surroundings."
            )

        return (
            "The system classifies the audio context as overloaded because the "
            "acoustic evidence suggests a potentially stressful environment."
        )

    return (
        "The system could not identify a strong behavioural audio cue. "
        "The result should be interpreted as supporting contextual evidence."
    )


def build_summary_text(result: dict[str, Any]) -> str:
    """
    Build presentation-friendly output.
    """
    prediction = str(result.get("prediction", "unknown"))
    confidence = float(result.get("confidence", 0.0))
    scores = normalise_scores(result.get("behaviour_scores", {}))
    labels = result.get("yamnet_labels", [])
    interpretation = result.get("brief_interpretation", "")

    display_prediction = prediction.upper()

    lines = [
        "Live Audio Behaviour Test",
        "=" * 50,
        "",
        f"Final Prediction: {display_prediction}",
        f"Confidence: {format_percent(confidence)}",
        f"Score Source: {result.get('score_source')}",
        "",
        "Final Behaviour Probabilities",
        "-" * 50,
        format_scores_percent(scores),
        "",
        "Top Acoustic Evidence",
        "-" * 50,
        format_audio_labels_short(labels, limit=3),
        "",
        "Brief Interpretation",
        "-" * 50,
        interpretation,
        "",
        "Technical Details",
        "-" * 50,
        "Detailed audio labels, extracted audio features, raw model output,",
        "and diagnostic metadata are available through the",
        "'Show Technical Details' button and export functions.",
        "",
        "Report Export",
        "-" * 50,
        "Full result available as JSON, CSV, or TXT report.",
    ]

    return "\n".join(lines)


def build_technical_text(result: dict[str, Any]) -> str:
    """
    Build full technical diagnostic output.
    """
    scores = normalise_scores(result.get("behaviour_scores", {}))
    labels = result.get("yamnet_labels", [])
    features = result.get("audio_features", {})

    lines = [
        "Live Audio Behaviour Test - Technical Details",
        "=" * 64,
        "",
        f"Created at:              {result.get('created_at')}",
        f"Selected audio file:     {result.get('selected_audio_path')}",
        f"Copied audio path:       {result.get('copied_audio_path')}",
        f"Predicted behaviour:     {result.get('prediction')}",
        f"Confidence:              {float(result.get('confidence', 0.0)):.4f}",
        f"Score source:            {result.get('score_source')}",
        "",
        "Behaviour scores:",
        format_scores_decimal(scores),
        "",
        "Top YAMNet audio labels:",
        format_audio_labels(labels),
        "",
        "Extracted audio features:",
    ]

    if features:
        for name, value in features.items():
            numeric_value = safe_float(value)

            if numeric_value is not None:
                lines.append(f"  {name}: {numeric_value:.4f}")
            else:
                lines.append(f"  {name}: {value}")
    else:
        lines.append("  No extracted audio features returned.")

    lines.extend(
        [
            "",
            "Raw audio result JSON:",
            json.dumps(result.get("raw_audio_result", {}), indent=2, default=str),
            "",
            "Methodological note:",
            str(result.get("methodological_note", "")),
        ]
    )

    return "\n".join(lines)


# ============================================================
# Audio analysis
# ============================================================

def run_audio_prediction() -> None:
    """
    Run audio analysis and display the presentation-friendly result in the GUI.
    """
    global selected_audio_path
    global latest_audio_result
    global latest_summary_text
    global latest_technical_text

    output_box.delete("1.0", tk.END)

    if selected_audio_path is None:
        output_box.insert(tk.END, "No audio file selected.\n")
        return

    if not selected_audio_path.exists():
        output_box.insert(
            tk.END,
            f"Audio file not found:\n{selected_audio_path}\n",
        )
        return

    if selected_audio_path.suffix.lower() not in SUPPORTED_AUDIO:
        output_box.insert(
            tk.END,
            f"Unsupported audio format: {selected_audio_path.suffix}\n"
            f"Supported formats: {', '.join(sorted(SUPPORTED_AUDIO))}\n",
        )
        return

    try:
        output_box.insert(tk.END, "Analysing audio file...\n\n")
        root.update_idletasks()

        copied_audio_path = copy_audio_to_uploads(selected_audio_path)
        raw_audio_result = analyze_audio_file(str(copied_audio_path))

        labels, features, model_scores, predicted_label = extract_audio_outputs(
            raw_audio_result
        )

        if model_scores:
            behaviour_scores = normalise_scores(model_scores)
            score_source = "audio_model_behaviour_scores"
        else:
            behaviour_scores = normalise_scores(
                audio_to_behaviour_scores(labels)
            )
            score_source = "yamnet_label_rule_mapping"

        prediction = predicted_label or get_best_label(behaviour_scores)
        confidence = float(
            behaviour_scores.get(prediction, max(behaviour_scores.values()))
        )

        brief_interpretation = build_brief_interpretation(
            prediction=prediction,
            labels=labels,
            score_source=score_source,
        )

        latest_audio_result = {
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "modality": "audio",
            "selected_audio_path": str(selected_audio_path),
            "copied_audio_path": str(copied_audio_path),
            "prediction": prediction,
            "confidence": confidence,
            "behaviour_scores": behaviour_scores,
            "score_source": score_source,
            "yamnet_labels": labels,
            "audio_features": features,
            "raw_audio_result": raw_audio_result,
            "supported_classes": BEHAVIOUR_CLASSES,
            "brief_interpretation": brief_interpretation,
            "methodological_note": (
                "This is an audio-context prediction. It estimates behavioural "
                "state from acoustic context such as silence, chatter, alarms, "
                "traffic, breathing, yawning, or stressful noise. Audio alone "
                "should be treated as supporting evidence, not definitive "
                "behavioural diagnosis."
            ),
        }

        latest_summary_text = build_summary_text(latest_audio_result)
        latest_technical_text = build_technical_text(latest_audio_result)

        output_box.delete("1.0", tk.END)
        output_box.insert(tk.END, latest_summary_text)

    except Exception as error:
        latest_audio_result = None
        latest_summary_text = ""
        latest_technical_text = ""

        output_box.delete("1.0", tk.END)
        output_box.insert(tk.END, "Audio analysis failed.\n\n")
        output_box.insert(tk.END, str(error))


# ============================================================
# Technical details window
# ============================================================

def show_technical_details() -> None:
    """
    Open a separate window showing the detailed diagnostic output.
    """
    result = require_latest_result()

    if result is None:
        return

    details_window = tk.Toplevel(root)
    details_window.title("Audio Technical Details")
    details_window.geometry("980x760")

    title = tk.Label(
        details_window,
        text="Audio Technical Details",
        font=("Arial", 16, "bold"),
    )
    title.pack(pady=10)

    details_box = scrolledtext.ScrolledText(
        details_window,
        width=115,
        height=38,
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
    Export the latest audio result as JSON.
    """
    result = require_latest_result()

    if result is None:
        return

    output_path = EXPORT_DIR / f"audio_result_{timestamp_for_filename()}.json"

    export_payload = {
        **result,
        "presentation_summary": latest_summary_text,
        "technical_report": latest_technical_text,
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(export_payload, f, indent=4, default=str)

    messagebox.showinfo(
        "Export Complete",
        f"JSON result exported successfully:\n\n{output_path}",
    )


def export_latest_result_csv() -> None:
    """
    Export the latest audio result as a one-row CSV summary.
    """
    result = require_latest_result()

    if result is None:
        return

    output_path = EXPORT_DIR / f"audio_result_{timestamp_for_filename()}.csv"

    row: dict[str, Any] = {
        "created_at": result.get("created_at"),
        "modality": result.get("modality"),
        "selected_audio_path": result.get("selected_audio_path"),
        "copied_audio_path": result.get("copied_audio_path"),
        "prediction": result.get("prediction"),
        "confidence": result.get("confidence"),
        "confidence_percent": format_percent(float(result.get("confidence", 0.0))),
        "score_source": result.get("score_source"),
        "brief_interpretation": result.get("brief_interpretation"),
    }

    scores = normalise_scores(result.get("behaviour_scores", {}))

    for label in BEHAVIOUR_CLASSES:
        row[f"score_{label}"] = scores.get(label, 0.0)

    labels = result.get("yamnet_labels", [])

    if labels:
        top_label_strings = []

        for item in labels[:5]:
            try:
                label, score = item
                top_label_strings.append(f"{label}:{float(score):.4f}")
            except Exception:
                top_label_strings.append(str(item))

        row["top_audio_labels"] = " | ".join(top_label_strings)
    else:
        row["top_audio_labels"] = ""

    features = result.get("audio_features", {})

    for feature_name, feature_value in features.items():
        row[f"feature_{feature_name}"] = feature_value

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
    Export the latest audio result as a human-readable report.
    """
    result = require_latest_result()

    if result is None:
        return

    output_path = EXPORT_DIR / f"audio_report_{timestamp_for_filename()}.txt"

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("SenseFuzeAI Live Audio Behaviour Test Report\n")
        f.write("===========================================\n\n")

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

def choose_audio_file() -> None:
    """
    Select an audio file from the filesystem.
    """
    global selected_audio_path
    global latest_audio_result
    global latest_summary_text
    global latest_technical_text

    file_path = filedialog.askopenfilename(
        title="Select an audio file",
        filetypes=[
            ("Audio files", "*.wav *.mp3 *.m4a *.flac *.ogg *.aac"),
            ("All files", "*.*"),
        ],
    )

    if not file_path:
        return

    selected_audio_path = Path(file_path)

    latest_audio_result = None
    latest_summary_text = ""
    latest_technical_text = ""

    selected_file_label.config(text=str(selected_audio_path))

    output_box.delete("1.0", tk.END)
    output_box.insert(
        tk.END,
        "Audio file selected.\n"
        "Click 'Predict Audio Behaviour' to analyse it.\n\n"
        f"{selected_audio_path}\n",
    )


def reset_session() -> None:
    """
    Reset the GUI state.
    """
    global selected_audio_path
    global latest_audio_result
    global latest_summary_text
    global latest_technical_text

    selected_audio_path = None

    latest_audio_result = None
    latest_summary_text = ""
    latest_technical_text = ""

    selected_file_label.config(text="No audio file selected.")

    output_box.delete("1.0", tk.END)
    output_box.insert(
        tk.END,
        "Session reset. Select an audio file to begin.\n",
    )


def show_guidance() -> None:
    """
    Show testing guidance.
    """
    messagebox.showinfo(
        "Audio Test Guidance",
        "Recommended tests:\n\n"
        "Focused:\n"
        "- silence\n"
        "- quiet room\n"
        "- library ambience\n\n"
        "Distracted:\n"
        "- speech\n"
        "- conversation\n"
        "- café chatter\n"
        "- phone ringing\n\n"
        "Fatigued:\n"
        "- yawning\n"
        "- breathing\n"
        "- snoring\n"
        "- low quiet ambience\n\n"
        "Overloaded:\n"
        "- alarm\n"
        "- shouting\n"
        "- loud traffic\n"
        "- construction\n"
        "- chaotic noise\n\n"
        "The main screen shows a concise demo-friendly result. "
        "Use 'Show Technical Details' to inspect labels, features, and raw output.",
    )


def show_export_guidance() -> None:
    """
    Explain export functionality.
    """
    messagebox.showinfo(
        "Export Guidance",
        "After running an audio prediction, you can export the latest result as:\n\n"
        "1. JSON: full diagnostic result, summary, and technical report\n"
        "2. CSV: compact one-row summary for spreadsheet analysis\n"
        "3. TXT: readable report containing both the presentation summary and technical details\n\n"
        f"Exports are saved in:\n{EXPORT_DIR}",
    )


# ============================================================
# GUI construction
# ============================================================

root = tk.Tk()
root.title("Live Audio Behaviour Test")
root.geometry("1000x820")

title_label = tk.Label(
    root,
    text="Live Audio Behaviour Test",
    font=("Arial", 18, "bold"),
)
title_label.pack(pady=10)

instruction_label = tk.Label(
    root,
    text=(
        "Select an audio file from your dataset or uploads folder. "
        "The main result is simplified for prototype demonstration, while full "
        "technical diagnostics remain available separately."
    ),
    font=("Arial", 11),
    wraplength=920,
)
instruction_label.pack(pady=5)

selected_file_label = tk.Label(
    root,
    text="No audio file selected.",
    font=("Arial", 10),
    fg="gray",
    wraplength=920,
)
selected_file_label.pack(pady=8)

button_frame = tk.Frame(root)
button_frame.pack(pady=8)

choose_button = tk.Button(
    button_frame,
    text="Choose Audio File",
    command=choose_audio_file,
    width=22,
)
choose_button.grid(row=0, column=0, padx=6, pady=4)

predict_button = tk.Button(
    button_frame,
    text="Predict Audio Behaviour",
    command=run_audio_prediction,
    width=24,
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
    width=112,
    height=28,
    font=("Consolas", 10),
)
output_box.pack(padx=15, pady=15)

output_box.insert(
    tk.END,
    "Waiting for audio input...\n\n"
    "Steps:\n"
    "1. Click 'Choose Audio File'.\n"
    "2. Select a .wav, .mp3, .m4a, .flac, .ogg, or .aac file.\n"
    "3. Click 'Predict Audio Behaviour'.\n"
    "4. The main output will show a concise result suitable for video demonstration.\n"
    "5. Click 'Show Technical Details' to inspect audio labels, features, and raw output.\n"
    "6. Export the latest result as JSON, CSV, or TXT if needed.\n\n"
    f"Export directory:\n{EXPORT_DIR}\n",
)

root.mainloop()
