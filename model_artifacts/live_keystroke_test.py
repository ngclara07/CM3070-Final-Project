# === model_artifacts/live_keystroke_test.py ===
# SenseFuzeAI - Live Keystroke Behaviour Test GUI
#
# Presentation-first version:
#   - main output shows concise prototype-friendly result
#   - technical details are available through "Show Technical Details"
#   - exports include both summary and full diagnostic information
#
# Supports:
#   - live keystroke timing capture
#   - hybrid model + rule-based keystroke behaviour prediction
#   - smoothed prediction output
#   - concise demo output
#   - technical diagnostic output
#   - export latest result as JSON, CSV, and TXT report

from __future__ import annotations

import csv
import json
import sys
import time
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

import tkinter as tk
from tkinter import scrolledtext, messagebox


# ============================================================
# Project path setup
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from app.models.keystroke_model import predict_keystroke_behaviour


# ============================================================
# Configuration
# ============================================================

BEHAVIOUR_CLASSES = ["focused", "distracted", "fatigued", "overloaded"]

EXPORT_DIR = PROJECT_ROOT / "model_artifacts" / "exported_results"
EXPORT_DIR.mkdir(parents=True, exist_ok=True)

MIN_KEYDOWNS = 25
WINDOW_KEYDOWNS = 60
PREDICT_EVERY_KEYDOWNS = 10
SMOOTHING_WINDOW = 5
CONFIDENCE_THRESHOLD = 0.40

MODEL_WEIGHT = 0.30
RULE_WEIGHT = 0.70

IGNORED_KEYS = {
    "shift_l", "shift_r", "shift",
    "control_l", "control_r", "ctrl",
    "alt_l", "alt_r", "alt",
    "caps_lock", "tab",
    "left", "right", "up", "down",
    "home", "end", "prior", "next",
    "escape", "insert",
    "command", "option",
}

events: List[Dict[str, Any]] = []
active_keys: set[str] = set()
prediction_history = deque(maxlen=SMOOTHING_WINDOW)

latest_keystroke_result: dict[str, Any] | None = None
latest_summary_text: str = ""
latest_technical_text: str = ""


# ============================================================
# General helpers
# ============================================================

def timestamp_for_filename() -> str:
    """
    Return a filesystem-safe timestamp.
    """
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def normalise_key(event) -> str | None:
    """
    Convert Tkinter key event into a clean symbolic key name.
    """
    keysym = str(event.keysym).lower()

    if keysym in IGNORED_KEYS:
        return None

    if keysym == "backspace":
        return "backspace"

    if keysym == "delete":
        return "delete"

    if keysym == "space":
        return "space"

    if keysym in {"return", "enter"}:
        return "enter"

    if event.char and len(event.char) == 1 and event.char.isprintable():
        return event.char.lower()

    return None


def count_keydowns(source_events: List[Dict[str, Any]]) -> int:
    """
    Count keydown events.
    """
    return sum(1 for event in source_events if event.get("type") == "down")


def get_recent_window(
    source_events: List[Dict[str, Any]],
    max_keydowns: int,
) -> List[Dict[str, Any]]:
    """
    Return the most recent event window containing up to max_keydowns keydown events.
    """
    keydown_seen = 0
    selected: List[Dict[str, Any]] = []

    for event in reversed(source_events):
        selected.append(event)

        if event.get("type") == "down":
            keydown_seen += 1

        if keydown_seen >= max_keydowns:
            break

    return list(reversed(selected))


def normalize_scores(scores: Dict[str, float] | None) -> Dict[str, float]:
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


def safe_float(value: Any) -> float | None:
    """
    Convert value to float if possible.
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


def format_scores_decimal(scores: Dict[str, float]) -> str:
    """
    Format score dictionary as decimal probabilities.
    """
    scores = normalize_scores(scores)

    lines = []

    for label, score in sorted(
        scores.items(),
        key=lambda item: item[1],
        reverse=True,
    ):
        lines.append(f"  {label}: {score:.4f}")

    return "\n".join(lines)


def format_scores_percent(scores: Dict[str, float]) -> str:
    """
    Format score dictionary as percentage probabilities for presentation output.
    """
    scores = normalize_scores(scores)

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


def require_latest_result() -> dict[str, Any] | None:
    """
    Ensure that a latest keystroke result exists before exporting.
    """
    if latest_keystroke_result is None:
        messagebox.showwarning(
            "No Result Available",
            "No keystroke prediction result is available yet.\n\n"
            "Please type enough keystrokes and run a prediction first.",
        )
        return None

    return latest_keystroke_result


# ============================================================
# Rule-based scoring
# ============================================================

def compute_rule_based_scores(features: Dict[str, float]) -> Dict[str, float]:
    """
    Estimate keystroke behaviour state using interpretable heuristic rules.
    """
    typing_speed = features.get("typing_speed", 0.0)
    hold_mean = features.get("hold_mean", 0.0)
    delay_mean = features.get("delay_mean", 0.0)
    delay_std = features.get("delay_std", 0.0)
    delay_cv = features.get("delay_cv", 0.0)

    pause_1000 = features.get("pause_ratio_1000", 0.0)
    pause_2000 = features.get("pause_ratio_2000", 0.0)
    mental_block = features.get("mental_block_ratio_5000", 0.0)

    correction_ratio = features.get("correction_ratio", 0.0)
    repeated_key_ratio = features.get("repeated_key_ratio", 0.0)
    rapid_burst_ratio = features.get("rapid_burst_ratio", 0.0)

    fits_starts = features.get("fits_starts_index", 0.0)
    rhythm_consistency = features.get("rhythm_consistency", 0.0)
    error_rate_proxy = features.get("error_rate_proxy", 0.0)

    scores = {
        "focused": 0.25,
        "distracted": 0.25,
        "fatigued": 0.25,
        "overloaded": 0.25,
    }

    focused_pattern = (
        typing_speed >= 0.0018
        and rhythm_consistency >= 0.65
        and delay_std <= 350
        and error_rate_proxy <= 0.10
        and pause_1000 <= 0.08
    )

    distracted_pattern = (
        pause_1000 >= 0.10
        or pause_2000 >= 0.05
        or delay_cv >= 0.90
    ) and error_rate_proxy <= 0.18

    fatigued_pattern = (
        typing_speed < 0.0022
        and delay_mean >= 450
        and hold_mean >= 110
    ) or mental_block >= 0.04

    overloaded_pattern = (
        error_rate_proxy >= 0.20
        or correction_ratio >= 0.16
        or repeated_key_ratio >= 0.20
        or fits_starts >= 850
        or rapid_burst_ratio >= 0.50
    )

    if focused_pattern:
        scores["focused"] += 0.60
        scores["distracted"] -= 0.10
        scores["fatigued"] -= 0.05
        scores["overloaded"] -= 0.20

    if distracted_pattern:
        scores["distracted"] += 0.50
        scores["focused"] -= 0.10
        scores["overloaded"] -= 0.05

    if fatigued_pattern:
        scores["fatigued"] += 0.50
        scores["focused"] -= 0.05
        scores["overloaded"] -= 0.10

    if overloaded_pattern:
        scores["overloaded"] += 0.60
        scores["focused"] -= 0.20
        scores["fatigued"] -= 0.05
    else:
        scores["overloaded"] -= 0.20

    return normalize_scores(scores)


def hybrid_scores(
    model_scores: Dict[str, float],
    rule_scores: Dict[str, float],
) -> Dict[str, float]:
    """
    Combine trained model probabilities and rule-based scores.
    """
    model_scores = normalize_scores(model_scores)
    rule_scores = normalize_scores(rule_scores)

    fused = {}

    for label in BEHAVIOUR_CLASSES:
        fused[label] = (
            MODEL_WEIGHT * model_scores.get(label, 0.0)
            + RULE_WEIGHT * rule_scores.get(label, 0.0)
        )

    return normalize_scores(fused)


def smooth_scores(new_scores: Dict[str, float]) -> Dict[str, float]:
    """
    Smooth predictions across recent prediction outputs.
    """
    prediction_history.append(normalize_scores(new_scores))

    averaged = {label: 0.0 for label in BEHAVIOUR_CLASSES}

    for score_map in prediction_history:
        for label in BEHAVIOUR_CLASSES:
            averaged[label] += score_map.get(label, 0.0)

    for label in BEHAVIOUR_CLASSES:
        averaged[label] /= len(prediction_history)

    return normalize_scores(averaged)


def decide_prediction(scores: Dict[str, float]) -> Tuple[str, float]:
    """
    Choose final label, or return uncertain if confidence is weak.
    """
    scores = normalize_scores(scores)
    ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)

    best_label, best_score = ranked[0]
    _, second_score = ranked[1]

    margin = best_score - second_score

    if best_score < CONFIDENCE_THRESHOLD or margin < 0.08:
        return "uncertain", best_score

    return best_label, best_score


# ============================================================
# Presentation interpretation
# ============================================================

def build_brief_interpretation(
    final_prediction: str,
    features: Dict[str, float],
) -> str:
    """
    Build concise, demo-friendly interpretation text.
    """
    typing_speed = float(features.get("typing_speed", 0.0))
    delay_mean = float(features.get("delay_mean", 0.0))
    delay_std = float(features.get("delay_std", 0.0))
    rhythm_consistency = float(features.get("rhythm_consistency", 0.0))
    error_rate_proxy = float(features.get("error_rate_proxy", 0.0))
    correction_ratio = float(features.get("correction_ratio", 0.0))
    pause_1000 = float(features.get("pause_ratio_1000", 0.0))
    fits_starts = float(features.get("fits_starts_index", 0.0))

    if final_prediction == "focused":
        if rhythm_consistency >= 0.65:
            return (
                "The system classifies the user as focused because the typing "
                "pattern shows relatively stable rhythm and consistent key timing."
            )
        return (
            "The system classifies the user as focused because the recent typing "
            "pattern appears more stable than the alternative behavioural states."
        )

    if final_prediction == "distracted":
        if pause_1000 >= 0.10 or delay_std >= 350:
            return (
                "The system classifies the user as distracted because the typing "
                "pattern shows irregular pauses and inconsistent timing."
            )
        return (
            "The system classifies the user as distracted because the recent "
            "keystroke rhythm suggests interruptions or reduced consistency."
        )

    if final_prediction == "fatigued":
        if typing_speed < 0.0022 or delay_mean >= 450:
            return (
                "The system classifies the user as fatigued because the typing "
                "pattern shows slower rhythm and longer inter-key delays."
            )
        return (
            "The system classifies the user as fatigued because the recent typing "
            "pattern suggests reduced speed or lower interaction energy."
        )

    if final_prediction == "overloaded":
        if error_rate_proxy >= 0.20 or correction_ratio >= 0.16 or fits_starts >= 850:
            return (
                "The system classifies the user as overloaded because the typing "
                "pattern shows erratic rhythm, correction behaviour, or burst-like changes."
            )
        return (
            "The system classifies the user as overloaded because the recent "
            "keystroke pattern appears less stable and more erratic."
        )

    return (
        "The system is uncertain because the behavioural probabilities are close "
        "or the confidence threshold has not been strongly exceeded."
    )


def build_summary_text(result: Dict[str, Any]) -> str:
    """
    Build presentation-friendly output.
    """
    final_prediction = str(result.get("final_prediction", "unknown"))
    confidence = float(result.get("final_confidence", 0.0))
    mode = "Auto" if result.get("auto_prediction") else "Manual"

    smoothed_scores = normalize_scores(
        result.get("smoothed_hybrid_probabilities", {})
    )

    interpretation = result.get("brief_interpretation", "")

    display_prediction = final_prediction.upper()

    lines = [
        "Live Keystroke Behaviour Test",
        "=" * 50,
        "",
        f"Final Prediction: {display_prediction}",
        f"Confidence: {format_percent(confidence)}",
        f"Prediction Mode: {mode}",
        f"Valid Key Presses: {result.get('total_valid_key_presses')}",
        "",
        "Final Behaviour Probabilities",
        "-" * 50,
        format_scores_percent(smoothed_scores),
        "",
        "Brief Interpretation",
        "-" * 50,
        interpretation,
        "",
        "Technical Details",
        "-" * 50,
        "Detailed model probabilities, rule-based scores, smoothing output,",
        "and extracted keystroke features are available through the",
        "'Show Technical Details' button and export functions.",
        "",
        "Report Export",
        "-" * 50,
        "Full result available as JSON, CSV, or TXT report.",
    ]

    return "\n".join(lines)


def build_technical_text(result: Dict[str, Any]) -> str:
    """
    Build full technical diagnostic output.
    """
    lines = [
        "Live Keystroke Behaviour Test - Technical Details",
        "=" * 64,
        "",
        f"Created at:                 {result.get('created_at')}",
        f"Prediction mode:            {'auto' if result.get('auto_prediction') else 'manual'}",
        f"Total valid key presses:    {result.get('total_valid_key_presses')}",
        f"Recent window key presses:  {result.get('recent_window_key_presses')}",
        f"Hybrid weighting:           {MODEL_WEIGHT:.2f} model / {RULE_WEIGHT:.2f} rules",
        f"Smoothing window:           last {result.get('smoothing_window_used')} predictions",
        f"Confidence threshold:       {CONFIDENCE_THRESHOLD:.2f}",
        "",
        f"Raw model prediction:       {result.get('raw_model_prediction')}",
        f"Final live prediction:      {result.get('final_prediction')}",
        f"Final confidence:           {float(result.get('final_confidence', 0.0)):.4f}",
        "",
        "Final smoothed hybrid probabilities:",
        format_scores_decimal(result.get("smoothed_hybrid_probabilities", {})),
        "",
        "Hybrid probabilities before smoothing:",
        format_scores_decimal(result.get("hybrid_probabilities", {})),
        "",
        "Rule-based probabilities:",
        format_scores_decimal(result.get("rule_based_probabilities", {})),
        "",
        "Raw model probabilities:",
        format_scores_decimal(result.get("raw_model_probabilities", {})),
        "",
        "Extracted recent-window features:",
    ]

    features = result.get("features", {})

    if features:
        for name, value in features.items():
            numeric_value = safe_float(value)

            if numeric_value is not None:
                lines.append(f"  {name}: {numeric_value:.4f}")
            else:
                lines.append(f"  {name}: {value}")
    else:
        lines.append("  No features available.")

    lines.extend(
        [
            "",
            "Behaviour interpretation:",
            "- focused: fast, consistent rhythm, low errors",
            "- distracted: irregular pauses and inconsistent speed",
            "- fatigued: slower typing, higher inter-key interval, possible mental blocks",
            "- overloaded: erratic fits-and-starts rhythm and high error proxy",
            "",
            "Methodological note:",
            str(result.get("methodological_note", "")),
        ]
    )

    return "\n".join(lines)


# ============================================================
# Keystroke event handlers
# ============================================================

def on_key_press(event) -> None:
    """
    Capture keydown event and optionally auto-run prediction.
    """
    key = normalise_key(event)

    if key is None:
        return

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

    keydown_count = count_keydowns(events)

    if (
        keydown_count >= MIN_KEYDOWNS
        and keydown_count % PREDICT_EVERY_KEYDOWNS == 0
    ):
        run_prediction(auto=True)


def on_key_release(event) -> None:
    """
    Capture keyup event.
    """
    key = normalise_key(event)

    if key is None:
        return

    active_keys.discard(key)

    events.append(
        {
            "type": "up",
            "key": key,
            "ts": time.perf_counter(),
        }
    )


# ============================================================
# Prediction
# ============================================================

def run_prediction(auto: bool = False) -> None:
    """
    Run keystroke behaviour prediction.
    """
    global latest_keystroke_result
    global latest_summary_text
    global latest_technical_text

    keydown_count = count_keydowns(events)

    output_box.delete("1.0", tk.END)

    if keydown_count < MIN_KEYDOWNS:
        output_box.insert(
            tk.END,
            f"Need at least {MIN_KEYDOWNS} valid key presses.\n"
            f"Current valid key presses: {keydown_count}\n",
        )
        return

    recent_events = get_recent_window(events, WINDOW_KEYDOWNS)
    recent_keydowns = count_keydowns(recent_events)

    try:
        raw_prediction, raw_probabilities, features = predict_keystroke_behaviour(
            recent_events
        )

    except Exception as error:
        latest_keystroke_result = None
        latest_summary_text = ""
        latest_technical_text = ""

        output_box.insert(tk.END, "Prediction failed.\n\n")
        output_box.insert(tk.END, str(error))
        output_box.insert(
            tk.END,
            "\n\nMake sure you have run:\n"
            "python data_cleaning.py\n"
            "python train_keystroke_model.py\n",
        )
        return

    raw_probabilities = normalize_scores(raw_probabilities)
    rule_scores = compute_rule_based_scores(features)
    fused_scores = hybrid_scores(raw_probabilities, rule_scores)
    smoothed_scores = smooth_scores(fused_scores)

    final_prediction, confidence = decide_prediction(smoothed_scores)

    brief_interpretation = build_brief_interpretation(
        final_prediction=final_prediction,
        features=features,
    )

    latest_keystroke_result = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "modality": "keystroke",
        "auto_prediction": bool(auto),
        "total_valid_key_presses": int(keydown_count),
        "recent_window_key_presses": int(recent_keydowns),
        "minimum_keydowns_required": int(MIN_KEYDOWNS),
        "window_keydowns": int(WINDOW_KEYDOWNS),
        "prediction_every_keydowns": int(PREDICT_EVERY_KEYDOWNS),
        "smoothing_window_config": int(SMOOTHING_WINDOW),
        "smoothing_window_used": int(len(prediction_history)),
        "confidence_threshold": float(CONFIDENCE_THRESHOLD),
        "model_weight": float(MODEL_WEIGHT),
        "rule_weight": float(RULE_WEIGHT),
        "raw_model_prediction": str(raw_prediction),
        "final_prediction": str(final_prediction),
        "final_confidence": float(confidence),
        "raw_model_probabilities": raw_probabilities,
        "rule_based_probabilities": rule_scores,
        "hybrid_probabilities": fused_scores,
        "smoothed_hybrid_probabilities": smoothed_scores,
        "features": features,
        "recent_events": recent_events,
        "supported_classes": BEHAVIOUR_CLASSES,
        "brief_interpretation": brief_interpretation,
        "methodological_note": (
            "This live keystroke module estimates behavioural typing state from "
            "keystroke timing features. The final live prediction combines the "
            "trained keystroke model with interpretable rule-based scoring and "
            "temporal smoothing. It should be treated as behavioural interaction "
            "evidence rather than direct psychological ground truth."
        ),
    }

    latest_summary_text = build_summary_text(latest_keystroke_result)
    latest_technical_text = build_technical_text(latest_keystroke_result)

    output_box.insert(tk.END, latest_summary_text)


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
    details_window.title("Keystroke Technical Details")
    details_window.geometry("980x760")

    title = tk.Label(
        details_window,
        text="Keystroke Technical Details",
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
    Export latest keystroke result as JSON.
    """
    result = require_latest_result()

    if result is None:
        return

    output_path = EXPORT_DIR / f"keystroke_result_{timestamp_for_filename()}.json"

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
    Export latest keystroke result as a one-row CSV summary.
    """
    result = require_latest_result()

    if result is None:
        return

    output_path = EXPORT_DIR / f"keystroke_result_{timestamp_for_filename()}.csv"

    row: dict[str, Any] = {
        "created_at": result.get("created_at"),
        "modality": result.get("modality"),
        "prediction_mode": "auto" if result.get("auto_prediction") else "manual",
        "total_valid_key_presses": result.get("total_valid_key_presses"),
        "recent_window_key_presses": result.get("recent_window_key_presses"),
        "raw_model_prediction": result.get("raw_model_prediction"),
        "final_prediction": result.get("final_prediction"),
        "final_confidence": result.get("final_confidence"),
        "final_confidence_percent": format_percent(
            float(result.get("final_confidence", 0.0))
        ),
        "model_weight": result.get("model_weight"),
        "rule_weight": result.get("rule_weight"),
        "confidence_threshold": result.get("confidence_threshold"),
        "smoothing_window_used": result.get("smoothing_window_used"),
        "brief_interpretation": result.get("brief_interpretation"),
    }

    score_groups = {
        "raw_model": result.get("raw_model_probabilities", {}),
        "rule_based": result.get("rule_based_probabilities", {}),
        "hybrid": result.get("hybrid_probabilities", {}),
        "smoothed": result.get("smoothed_hybrid_probabilities", {}),
    }

    for group_name, score_map in score_groups.items():
        score_map = normalize_scores(score_map)

        for label in BEHAVIOUR_CLASSES:
            row[f"{group_name}_{label}"] = score_map.get(label, 0.0)

    features = result.get("features", {})

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
    Export latest keystroke result as a human-readable TXT report.
    """
    result = require_latest_result()

    if result is None:
        return

    output_path = EXPORT_DIR / f"keystroke_report_{timestamp_for_filename()}.txt"

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("SenseFuzeAI Live Keystroke Behaviour Test Report\n")
        f.write("================================================\n\n")

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

def reset_session() -> None:
    """
    Reset typing session and prediction state.
    """
    global latest_keystroke_result
    global latest_summary_text
    global latest_technical_text

    events.clear()
    active_keys.clear()
    prediction_history.clear()

    latest_keystroke_result = None
    latest_summary_text = ""
    latest_technical_text = ""

    text_box.delete("1.0", tk.END)
    output_box.delete("1.0", tk.END)

    output_box.insert(
        tk.END,
        "Session reset.\n"
        "Start typing again. Prediction becomes more stable after 40-60 valid key presses.\n",
    )


def show_guidance() -> None:
    """
    Show keystroke test guidance.
    """
    messagebox.showinfo(
        "Keystroke Test Guidance",
        "Recommended testing process:\n\n"
        "1. Type naturally in the text box.\n"
        f"2. Use at least {MIN_KEYDOWNS} valid key presses.\n"
        "3. The system will auto-predict periodically.\n"
        "4. You may also click 'Predict Now'.\n"
        "5. The main screen shows a concise demo-friendly result.\n"
        "6. Click 'Show Technical Details' for the full diagnostic output.\n"
        "7. Export the latest result after prediction.\n\n"
        "Interpretation:\n"
        "- Focused: faster, stable typing rhythm with low correction behaviour.\n"
        "- Distracted: irregular pauses and inconsistent rhythm.\n"
        "- Fatigued: slower typing and longer inter-key intervals.\n"
        "- Overloaded: erratic burst patterns and high correction/error proxy.",
    )


def show_export_guidance() -> None:
    """
    Explain export functionality.
    """
    messagebox.showinfo(
        "Export Guidance",
        "After running a keystroke prediction, you can export the latest result as:\n\n"
        "1. JSON: full diagnostic result, recent events, summary, and technical report\n"
        "2. CSV: compact one-row summary for spreadsheet analysis\n"
        "3. TXT: readable report containing both the presentation summary and technical details\n\n"
        f"Exports are saved in:\n{EXPORT_DIR}",
    )


# ============================================================
# GUI construction
# ============================================================

root = tk.Tk()
root.title("Live Keystroke Behaviour Test")
root.geometry("1000x820")

title_label = tk.Label(
    root,
    text="Live Keystroke Behaviour Test",
    font=("Arial", 18, "bold"),
)
title_label.pack(pady=10)

instruction = tk.Label(
    root,
    text=(
        "Type naturally in the box below. "
        "The main result is simplified for prototype demonstration, while full "
        "technical diagnostics remain available separately."
    ),
    font=("Arial", 12),
    wraplength=920,
)
instruction.pack(pady=5)

text_box = scrolledtext.ScrolledText(
    root,
    width=112,
    height=12,
    font=("Arial", 12),
)
text_box.pack(padx=10, pady=10)

text_box.bind("<KeyPress>", on_key_press)
text_box.bind("<KeyRelease>", on_key_release)

button_frame = tk.Frame(root)
button_frame.pack(pady=5)

predict_button = tk.Button(
    button_frame,
    text="Predict Now",
    command=lambda: run_prediction(auto=False),
    width=18,
)
predict_button.grid(row=0, column=0, padx=6, pady=4)

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
    width=112,
    height=25,
    font=("Consolas", 10),
)
output_box.pack(padx=10, pady=10)

output_box.insert(
    tk.END,
    "Waiting for typing input...\n\n"
    f"Minimum valid key presses: {MIN_KEYDOWNS}\n"
    f"Prediction window: last {WINDOW_KEYDOWNS} valid key presses\n"
    f"Smoothing: last {SMOOTHING_WINDOW} predictions\n"
    f"Hybrid scoring: {MODEL_WEIGHT:.2f} model / {RULE_WEIGHT:.2f} rules\n\n"
    "Prototype demo mode:\n"
    "- The main output will show a concise result suitable for video demonstration.\n"
    "- Click 'Show Technical Details' to inspect raw probabilities and features.\n"
    "- Export JSON, CSV, or TXT after prediction if needed.\n\n"
    f"Export directory:\n{EXPORT_DIR}\n",
)

root.mainloop()
