# === collect_multimodal_session.py ===
# SenseFuzeAI - Guided Session-Aligned Multimodal Data Collector
# Saves new biometric feature rows to retroactive_keystroke_features.csv

from __future__ import annotations

import csv
import json
import shutil
import statistics
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent

SESSION_ROOT = PROJECT_ROOT / "data" / "session_aligned"
TEXT_DIR = SESSION_ROOT / "texts"
KEYSTROKE_DIR = SESSION_ROOT / "keystrokes"
AUDIO_DIR = SESSION_ROOT / "audio"
IMAGE_DIR = SESSION_ROOT / "images"

FEATURES_CSV = SESSION_ROOT / "retroactive_keystroke_features.csv"

BEHAVIOUR_CLASSES = [
    "focused",
    "distracted",
    "fatigued",
    "overloaded",
]

SUPPORTED_AUDIO = {".wav", ".mp3", ".m4a", ".flac", ".ogg", ".aac"}
SUPPORTED_IMAGES = {".jpg", ".jpeg", ".png", ".webp"}

MIN_RECOMMENDED_KEYDOWNS = 20

for directory in [SESSION_ROOT, TEXT_DIR, KEYSTROKE_DIR, AUDIO_DIR, IMAGE_DIR]:
    directory.mkdir(parents=True, exist_ok=True)


GUIDED_SCENARIOS = {
    "focused": {
        "text_prompt": (
            "Type a focused work sentence such as:\n"
            "\"I am concentrating on my assignment and making steady progress with my work.\""
        ),
        "typing_guidance": (
            "Typing pattern target:\n"
            "- Type at a steady, consistent rhythm.\n"
            "- Keep pauses short, usually below 1 second.\n"
            "- Avoid unnecessary backspaces or corrections.\n"
            "- Use smooth word-to-word transitions.\n\n"
            "Expected biometric cues:\n"
            "- stable delay_mean\n"
            "- low delay_std\n"
            "- low pause_ratio_1000\n"
            "- low correction_ratio\n"
            "- higher rhythm_consistency"
        ),
        "audio_guidance": "Upload quiet-room, silence, library ambience, or soft keyboard audio.",
        "image_guidance": (
            "Upload an image showing upright posture, organised workspace, "
            "screen attention, and minimal distractions."
        ),
    },
    "distracted": {
        "text_prompt": (
            "Type a distracted work sentence such as:\n"
            "\"I keep checking my phone and switching between tabs instead of finishing my work.\""
        ),
        "typing_guidance": (
            "Typing pattern target:\n"
            "- Type in an uneven rhythm.\n"
            "- Add 2–4 short interruptions while typing.\n"
            "- Pause for about 1–3 seconds between some words or phrases.\n"
            "- Make a small number of corrections using Backspace.\n\n"
            "Expected biometric cues:\n"
            "- increased delay_std\n"
            "- moderate pause_ratio_1000\n"
            "- moderate correction_ratio\n"
            "- lower rhythm_consistency\n"
            "- possible fits_starts_index increase"
        ),
        "audio_guidance": "Upload phone notification, conversation, TV, crowd, chatter, or interruption audio.",
        "image_guidance": (
            "Upload an image showing phone use, looking away, multitasking, "
            "or visible distraction."
        ),
    },
    "fatigued": {
        "text_prompt": (
            "Type a fatigued work sentence such as:\n"
            "\"I feel tired and slow, and it is difficult to stay alert while working.\""
        ),
        "typing_guidance": (
            "Typing pattern target:\n"
            "- Type slower than usual.\n"
            "- Add longer pauses, around 2–5 seconds, especially between clauses.\n"
            "- Avoid frantic corrections; the pattern should feel low-energy.\n\n"
            "Expected biometric cues:\n"
            "- lower typing_speed\n"
            "- higher delay_mean\n"
            "- higher pause_ratio_2000\n"
            "- possible mental_block_ratio_5000 increase\n"
            "- reduced rhythm_consistency"
        ),
        "audio_guidance": (
            "Upload quiet ambience, yawning, sighing, low-energy breathing, "
            "or dim-room background audio."
        ),
        "image_guidance": (
            "Upload an image showing tired posture, head resting, dim lighting, "
            "or reduced alertness."
        ),
    },
    "overloaded": {
        "text_prompt": (
            "Type an overloaded work sentence such as:\n"
            "\"I feel overwhelmed because there are too many tasks, deadlines, and messages to handle.\""
        ),
        "typing_guidance": (
            "Typing pattern target:\n"
            "- Type with an irregular, pressured rhythm.\n"
            "- Alternate between fast bursts and sudden pauses.\n"
            "- Include several corrections using Backspace.\n\n"
            "Expected biometric cues:\n"
            "- higher burstiness_proxy\n"
            "- higher fits_starts_index\n"
            "- increased correction_ratio\n"
            "- increased delay_std\n"
            "- lower rhythm_consistency"
        ),
        "audio_guidance": "Upload noisy, chaotic, busy office, alarm-like, traffic, or pressure-related audio.",
        "image_guidance": (
            "Upload an image showing cluttered desk, many papers, stressed posture, "
            "or visible workload pressure."
        ),
    },
}


# ============================================================
# RUNTIME STATE
# ============================================================

keystroke_events: List[Dict[str, Any]] = []
active_keys: set[str] = set()

selected_audio_path: Optional[Path] = None
selected_image_path: Optional[Path] = None


# ============================================================
# UTILITIES
# ============================================================

def generate_session_id(label: str) -> str:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    short_id = uuid.uuid4().hex[:8]
    return f"{label}_{timestamp}_{short_id}"


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


def count_keydowns() -> int:
    return sum(1 for event in keystroke_events if event.get("type") == "down")


def safe_mean(values: List[float]) -> float:
    return statistics.mean(values) if values else 0.0


def safe_std(values: List[float]) -> float:
    return statistics.stdev(values) if len(values) >= 2 else 0.0


def validate_keystroke_events() -> Dict[str, Any]:
    keydown_count = count_keydowns()
    event_count = len(keystroke_events)
    expected_event_count = keydown_count * 2

    validation_passed = event_count == expected_event_count

    validation_message = (
        "OK - Session validation passed"
        if validation_passed
        else "WARNING - Event count mismatch. Some key release events may not have been captured."
    )

    return {
        "keydown_count": keydown_count,
        "event_count": event_count,
        "expected_event_count": expected_event_count,
        "validation_passed": validation_passed,
        "validation_message": validation_message,
    }


def extract_keystroke_features(
    events: List[Dict[str, Any]],
    typed_text: str,
) -> Dict[str, Any]:
    downs = [event for event in events if event.get("type") == "down"]
    down_times = [
        event["timestamp_perf"]
        for event in downs
        if "timestamp_perf" in event
    ]

    delays = [
        down_times[index] - down_times[index - 1]
        for index in range(1, len(down_times))
    ]

    hold_times: List[float] = []
    unmatched_downs: Dict[str, List[float]] = {}

    for event in events:
        key = event.get("key")
        event_type = event.get("type")
        timestamp = event.get("timestamp_perf")

        if key is None or timestamp is None:
            continue

        if event_type == "down":
            unmatched_downs.setdefault(key, []).append(timestamp)

        elif event_type == "up":
            if key in unmatched_downs and unmatched_downs[key]:
                down_time = unmatched_downs[key].pop(0)
                hold_times.append(timestamp - down_time)

    total_duration = (
        down_times[-1] - down_times[0]
        if len(down_times) >= 2
        else 0.0
    )

    keydown_count = len(downs)
    word_count = len(typed_text.split())

    correction_count = sum(
        1 for event in downs
        if event.get("key") in {"backspace", "delete"}
    )

    pauses_1000 = [delay for delay in delays if delay >= 1.0]
    pauses_2000 = [delay for delay in delays if delay >= 2.0]
    pauses_5000 = [delay for delay in delays if delay >= 5.0]

    delay_mean = safe_mean(delays)
    delay_std = safe_std(delays)

    rhythm_consistency = 1.0 / (1.0 + delay_std) if delay_std > 0 else 1.0

    typing_speed_kps = (
        keydown_count / total_duration
        if total_duration > 0
        else 0.0
    )

    typing_speed_wpm = (
        (word_count / total_duration) * 60
        if total_duration > 0
        else 0.0
    )

    burstiness_proxy = (
        delay_std / delay_mean
        if delay_mean > 0
        else 0.0
    )

    fits_starts_index = (
        len(pauses_1000) / len(delays)
        if delays
        else 0.0
    )

    return {
        "total_duration_sec": round(total_duration, 4),
        "keydown_count": keydown_count,
        "word_count": word_count,
        "typing_speed_kps": round(typing_speed_kps, 4),
        "typing_speed_wpm": round(typing_speed_wpm, 4),
        "delay_mean": round(delay_mean, 4),
        "delay_std": round(delay_std, 4),
        "delay_min": round(min(delays), 4) if delays else 0.0,
        "delay_max": round(max(delays), 4) if delays else 0.0,
        "hold_mean": round(safe_mean(hold_times), 4),
        "hold_std": round(safe_std(hold_times), 4),
        "pause_count_1000": len(pauses_1000),
        "pause_count_2000": len(pauses_2000),
        "pause_count_5000": len(pauses_5000),
        "pause_ratio_1000": round(len(pauses_1000) / len(delays), 4) if delays else 0.0,
        "pause_ratio_2000": round(len(pauses_2000) / len(delays), 4) if delays else 0.0,
        "mental_block_ratio_5000": round(len(pauses_5000) / len(delays), 4) if delays else 0.0,
        "correction_count": correction_count,
        "correction_ratio": round(correction_count / keydown_count, 4) if keydown_count else 0.0,
        "rhythm_consistency": round(rhythm_consistency, 4),
        "burstiness_proxy": round(burstiness_proxy, 4),
        "fits_starts_index": round(fits_starts_index, 4),
    }


def copy_optional_file(
    source_path: Optional[Path],
    destination_dir: Path,
    session_id: str,
    supported_extensions: set[str],
) -> str:
    if source_path is None:
        return ""

    if not source_path.exists():
        raise FileNotFoundError(f"Selected file does not exist: {source_path}")

    suffix = source_path.suffix.lower()

    if suffix not in supported_extensions:
        raise ValueError(
            f"Unsupported file extension: {suffix}. "
            f"Supported: {', '.join(sorted(supported_extensions))}"
        )

    destination = destination_dir / f"{session_id}{suffix}"
    shutil.copy2(source_path, destination)

    return str(destination.relative_to(SESSION_ROOT))


def append_feature_row(row: Dict[str, Any]) -> None:
    file_exists = FEATURES_CSV.exists()

    fieldnames = [
        "session_id",
        "label",
        "created_at",
        "text_path",
        "keystroke_path",
        "audio_path",
        "image_path",
        "typed_text",
        "typed_text_length",
        "keydown_count",
        "event_count",
        "expected_event_count",
        "keystroke_validation_passed",
        "keystroke_validation_message",
        "total_duration_sec",
        "typing_speed_kps",
        "typing_speed_wpm",
        "delay_mean",
        "delay_std",
        "delay_min",
        "delay_max",
        "hold_mean",
        "hold_std",
        "pause_count_1000",
        "pause_count_2000",
        "pause_count_5000",
        "pause_ratio_1000",
        "pause_ratio_2000",
        "mental_block_ratio_5000",
        "correction_count",
        "correction_ratio",
        "rhythm_consistency",
        "burstiness_proxy",
        "fits_starts_index",
        "audio_provided",
        "image_provided",
        "collection_mode",
        "notes",
    ]

    with FEATURES_CSV.open("a", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)

        if not file_exists:
            writer.writeheader()

        writer.writerow(row)


def update_status(message: str) -> None:
    status_box.delete("1.0", tk.END)
    status_box.insert(tk.END, message)


# ============================================================
# GUIDED INSTRUCTIONS
# ============================================================

def update_guided_instructions() -> None:
    label = label_var.get().strip().lower()
    scenario = GUIDED_SCENARIOS.get(label, {})

    guidance_box.configure(state="normal")
    guidance_box.delete("1.0", tk.END)

    guidance_box.insert(tk.END, f"Selected behaviour: {label.upper()}\n")
    guidance_box.insert(tk.END, "=" * 78 + "\n\n")

    guidance_box.insert(tk.END, "1. Text input guidance\n")
    guidance_box.insert(tk.END, "-" * 78 + "\n")
    guidance_box.insert(tk.END, scenario.get("text_prompt", "") + "\n\n")

    guidance_box.insert(tk.END, "2. Keystroke capture guidance\n")
    guidance_box.insert(tk.END, "-" * 78 + "\n")
    guidance_box.insert(tk.END, scenario.get("typing_guidance", "") + "\n")
    guidance_box.insert(tk.END, f"Recommended minimum key presses: {MIN_RECOMMENDED_KEYDOWNS}\n\n")

    guidance_box.insert(tk.END, "3. Audio upload guidance\n")
    guidance_box.insert(tk.END, "-" * 78 + "\n")
    guidance_box.insert(tk.END, scenario.get("audio_guidance", "") + "\n\n")

    guidance_box.insert(tk.END, "4. Image upload guidance\n")
    guidance_box.insert(tk.END, "-" * 78 + "\n")
    guidance_box.insert(tk.END, scenario.get("image_guidance", "") + "\n\n")

    guidance_box.insert(tk.END, "Checklist before saving\n")
    guidance_box.insert(tk.END, "-" * 78 + "\n")
    guidance_box.insert(
        tk.END,
        "- The typed text should match the selected behavioural state.\n"
        "- The typing pattern should be natural for that behavioural state.\n"
        "- The audio file should represent the same session context.\n"
        "- The image should visually support the selected state.\n"
        "- All evidence should correspond to the same label/session.\n",
    )

    guidance_box.configure(state="disabled")


# ============================================================
# KEYSTROKE CAPTURE
# ============================================================

def on_key_press(event) -> None:
    key = normalise_key(event)

    if key in active_keys:
        return

    active_keys.add(key)

    keystroke_events.append(
        {
            "type": "down",
            "key": key,
            "timestamp_perf": time.perf_counter(),
            "timestamp_epoch": time.time(),
        }
    )


def on_key_release(event) -> None:
    key = normalise_key(event)
    active_keys.discard(key)

    keystroke_events.append(
        {
            "type": "up",
            "key": key,
            "timestamp_perf": time.perf_counter(),
            "timestamp_epoch": time.time(),
        }
    )


# ============================================================
# FILE SELECTION
# ============================================================

def choose_audio_file() -> None:
    global selected_audio_path

    file_path = filedialog.askopenfilename(
        title="Select session audio file",
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
        title="Select session image file",
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
# SESSION RESET
# ============================================================

def reset_session_state() -> None:
    global selected_audio_path, selected_image_path

    keystroke_events.clear()
    active_keys.clear()

    selected_audio_path = None
    selected_image_path = None

    text_box.delete("1.0", tk.END)
    notes_box.delete("1.0", tk.END)

    audio_label.config(text="No audio file selected.")
    image_label.config(text="No image file selected.")

    update_status(
        "Session reset.\n\n"
        "Select a behavioural label, follow the guided instructions, type naturally, "
        "attach session-matching audio/image files, then save the session.\n"
    )


# ============================================================
# SAVE SESSION
# ============================================================

def save_session() -> None:
    label = label_var.get().strip().lower()
    typed_text = text_box.get("1.0", tk.END).strip()
    notes = notes_box.get("1.0", tk.END).strip()

    if label not in BEHAVIOUR_CLASSES:
        messagebox.showerror("Missing Label", "Please select a behavioural label.")
        return

    if not typed_text:
        messagebox.showerror("Missing Text", "Please type text before saving the session.")
        return

    validation = validate_keystroke_events()
    features = extract_keystroke_features(keystroke_events, typed_text)

    keydown_count = validation["keydown_count"]

    if keydown_count < MIN_RECOMMENDED_KEYDOWNS:
        proceed = messagebox.askyesno(
            "Low Keystroke Count",
            f"Only {keydown_count} key presses were captured.\n\n"
            f"For stronger keystroke modelling, at least "
            f"{MIN_RECOMMENDED_KEYDOWNS} key presses are recommended.\n\n"
            "Save anyway?",
        )
        if not proceed:
            return

    if selected_audio_path is None:
        proceed = messagebox.askyesno(
            "Audio Missing",
            "No audio file was selected.\n\n"
            "For stronger session alignment, each session should ideally include "
            "audio from the same behavioural context.\n\n"
            "Save anyway?",
        )
        if not proceed:
            return

    if selected_image_path is None:
        proceed = messagebox.askyesno(
            "Image Missing",
            "No image file was selected.\n\n"
            "For stronger session alignment, each session should ideally include "
            "an image from the same behavioural context.\n\n"
            "Save anyway?",
        )
        if not proceed:
            return

    try:
        session_id = generate_session_id(label)
        created_at = datetime.now().isoformat(timespec="seconds")

        text_path = TEXT_DIR / f"{session_id}.txt"
        keystroke_path = KEYSTROKE_DIR / f"{session_id}.json"

        text_path.write_text(typed_text, encoding="utf-8")

        keystroke_payload = {
            "session_id": session_id,
            "label": label,
            "created_at": created_at,
            "events": keystroke_events,
            "features": features,
            "keydown_count": validation["keydown_count"],
            "event_count": validation["event_count"],
            "expected_event_count": validation["expected_event_count"],
            "validation_passed": validation["validation_passed"],
            "validation_message": validation["validation_message"],
        }

        keystroke_path.write_text(
            json.dumps(keystroke_payload, indent=4),
            encoding="utf-8",
        )

        audio_relative_path = copy_optional_file(
            source_path=selected_audio_path,
            destination_dir=AUDIO_DIR,
            session_id=session_id,
            supported_extensions=SUPPORTED_AUDIO,
        )

        image_relative_path = copy_optional_file(
            source_path=selected_image_path,
            destination_dir=IMAGE_DIR,
            session_id=session_id,
            supported_extensions=SUPPORTED_IMAGES,
        )

        row = {
            "session_id": session_id,
            "label": label,
            "created_at": created_at,
            "text_path": str(text_path.relative_to(SESSION_ROOT)),
            "keystroke_path": str(keystroke_path.relative_to(SESSION_ROOT)),
            "audio_path": audio_relative_path,
            "image_path": image_relative_path,
            "typed_text": typed_text,
            "typed_text_length": len(typed_text),
            "keydown_count": validation["keydown_count"],
            "event_count": validation["event_count"],
            "expected_event_count": validation["expected_event_count"],
            "keystroke_validation_passed": validation["validation_passed"],
            "keystroke_validation_message": validation["validation_message"],
            "total_duration_sec": features["total_duration_sec"],
            "typing_speed_kps": features["typing_speed_kps"],
            "typing_speed_wpm": features["typing_speed_wpm"],
            "delay_mean": features["delay_mean"],
            "delay_std": features["delay_std"],
            "delay_min": features["delay_min"],
            "delay_max": features["delay_max"],
            "hold_mean": features["hold_mean"],
            "hold_std": features["hold_std"],
            "pause_count_1000": features["pause_count_1000"],
            "pause_count_2000": features["pause_count_2000"],
            "pause_count_5000": features["pause_count_5000"],
            "pause_ratio_1000": features["pause_ratio_1000"],
            "pause_ratio_2000": features["pause_ratio_2000"],
            "mental_block_ratio_5000": features["mental_block_ratio_5000"],
            "correction_count": features["correction_count"],
            "correction_ratio": features["correction_ratio"],
            "rhythm_consistency": features["rhythm_consistency"],
            "burstiness_proxy": features["burstiness_proxy"],
            "fits_starts_index": features["fits_starts_index"],
            "audio_provided": bool(audio_relative_path),
            "image_provided": bool(image_relative_path),
            "collection_mode": "guided_session_aligned_with_keystroke_biometrics",
            "notes": notes,
        }

        append_feature_row(row)

        feature_summary = (
            f"Keystroke biometric features:\n"
            f"- Duration: {features['total_duration_sec']} sec\n"
            f"- Typing speed: {features['typing_speed_wpm']} WPM\n"
            f"- Delay mean: {features['delay_mean']} sec\n"
            f"- Delay std: {features['delay_std']} sec\n"
            f"- Pause ratio >1s: {features['pause_ratio_1000']}\n"
            f"- Pause ratio >2s: {features['pause_ratio_2000']}\n"
            f"- Correction ratio: {features['correction_ratio']}\n"
            f"- Rhythm consistency: {features['rhythm_consistency']}\n"
            f"- Burstiness proxy: {features['burstiness_proxy']}\n"
            f"- Fits-starts index: {features['fits_starts_index']}\n"
        )

        update_status(
            "Session saved successfully.\n\n"
            f"Session ID: {session_id}\n"
            f"Label: {label}\n\n"
            f"Text file: {row['text_path']}\n"
            f"Keystroke file: {row['keystroke_path']}\n"
            f"Audio file: {audio_relative_path or 'not provided'}\n"
            f"Image file: {image_relative_path or 'not provided'}\n\n"
            f"Keydown Count: {validation['keydown_count']}\n"
            f"Event Count: {validation['event_count']}\n"
            f"Expected Event Count: {validation['expected_event_count']}\n"
            f"{validation['validation_message']}\n\n"
            f"{feature_summary}\n"
            f"Features CSV: {FEATURES_CSV}\n"
        )

        messagebox.showinfo(
            "Session Saved",
            "Session saved successfully.\n\n"
            f"Session ID:\n{session_id}\n\n"
            f"Keydown Count: {validation['keydown_count']}\n"
            f"Event Count: {validation['event_count']}\n"
            f"Expected Event Count: {validation['expected_event_count']}\n\n"
            f"{validation['validation_message']}\n\n"
            f"Saved feature row to:\n{FEATURES_CSV}\n\n"
            f"Typing Speed: {features['typing_speed_wpm']} WPM\n"
            f"Delay Mean: {features['delay_mean']} sec\n"
            f"Delay Std: {features['delay_std']} sec\n"
            f"Pause Ratio >1s: {features['pause_ratio_1000']}\n"
            f"Correction Ratio: {features['correction_ratio']}",
        )

    except Exception as error:
        update_status(f"Failed to save session.\n\n{error}\n")
        messagebox.showerror("Save Failed", str(error))


# ============================================================
# GUIDANCE POPUP
# ============================================================

def show_guidance() -> None:
    messagebox.showinfo(
        "Session Collection Guidance",
        "This tool collects one session-aligned multimodal sample at a time.\n\n"
        "Each saved session contains:\n"
        "- typed text\n"
        "- raw keystroke events\n"
        "- extracted keystroke biometric features\n"
        "- audio file\n"
        "- image file\n"
        "- behavioural label\n\n"
        "Feature rows are saved to:\n"
        f"{FEATURES_CSV}\n\n"
        "Recommended minimum:\n"
        "- 10 sessions per class = 40 total\n"
        "- Better: 20 sessions per class = 80 total\n"
        "- Target for your project: 77 sessions per class = 308 total\n\n"
        "The output folder is:\n"
        f"{SESSION_ROOT}",
    )


# ============================================================
# GUI
# ============================================================

root = tk.Tk()
root.title("SenseFuzeAI Guided Session-Aligned Data Collector")
root.geometry("1120x930")

title_label = tk.Label(
    root,
    text="SenseFuzeAI Guided Session-Aligned Data Collector",
    font=("Arial", 18, "bold"),
)
title_label.pack(pady=10)

instruction_label = tk.Label(
    root,
    text=(
        "Collect session-aligned multimodal samples with guided instructions. "
        "This version saves raw keystrokes, extracted biometric features, and feature rows "
        "to retroactive_keystroke_features.csv."
    ),
    font=("Arial", 11),
    wraplength=1040,
)
instruction_label.pack(pady=5)

label_frame = tk.Frame(root)
label_frame.pack(pady=8)

tk.Label(
    label_frame,
    text="Behaviour label:",
    font=("Arial", 11, "bold"),
).grid(row=0, column=0, padx=8)

label_var = tk.StringVar(value="focused")

for index, behaviour in enumerate(BEHAVIOUR_CLASSES, start=1):
    tk.Radiobutton(
        label_frame,
        text=behaviour,
        variable=label_var,
        value=behaviour,
        font=("Arial", 10),
        command=update_guided_instructions,
    ).grid(row=0, column=index, padx=8)

guidance_title = tk.Label(
    root,
    text="Guided collection instructions",
    font=("Arial", 11, "bold"),
)
guidance_title.pack(anchor="w", padx=25, pady=(8, 2))

guidance_box = scrolledtext.ScrolledText(
    root,
    width=130,
    height=12,
    font=("Consolas", 9),
    bg="#f7f7f7",
)
guidance_box.pack(padx=25, pady=5)

text_label = tk.Label(
    root,
    text="Typed text input",
    font=("Arial", 11, "bold"),
)
text_label.pack(anchor="w", padx=25, pady=(10, 2))

text_box = scrolledtext.ScrolledText(
    root,
    width=130,
    height=7,
    font=("Consolas", 10),
)
text_box.pack(padx=25, pady=5)

text_box.bind("<KeyPress>", on_key_press)
text_box.bind("<KeyRelease>", on_key_release)

file_frame = tk.Frame(root)
file_frame.pack(pady=8)

audio_button = tk.Button(
    file_frame,
    text="Choose Audio File",
    command=choose_audio_file,
    width=22,
)
audio_button.grid(row=0, column=0, padx=8, pady=4)

audio_label = tk.Label(
    file_frame,
    text="No audio file selected.",
    width=112,
    anchor="w",
    fg="gray",
)
audio_label.grid(row=0, column=1, padx=8, pady=4)

image_button = tk.Button(
    file_frame,
    text="Choose Image File",
    command=choose_image_file,
    width=22,
)
image_button.grid(row=1, column=0, padx=8, pady=4)

image_label = tk.Label(
    file_frame,
    text="No image file selected.",
    width=112,
    anchor="w",
    fg="gray",
)
image_label.grid(row=1, column=1, padx=8, pady=4)

notes_label = tk.Label(
    root,
    text="Optional notes",
    font=("Arial", 11, "bold"),
)
notes_label.pack(anchor="w", padx=25, pady=(8, 2))

notes_box = scrolledtext.ScrolledText(
    root,
    width=130,
    height=3,
    font=("Consolas", 10),
)
notes_box.pack(padx=25, pady=5)

button_frame = tk.Frame(root)
button_frame.pack(pady=10)

save_button = tk.Button(
    button_frame,
    text="Save Session",
    command=save_session,
    width=22,
)
save_button.grid(row=0, column=0, padx=8)

reset_button = tk.Button(
    button_frame,
    text="Reset",
    command=reset_session_state,
    width=16,
)
reset_button.grid(row=0, column=1, padx=8)

guidance_button = tk.Button(
    button_frame,
    text="Guidance",
    command=show_guidance,
    width=16,
)
guidance_button.grid(row=0, column=2, padx=8)

status_label = tk.Label(
    root,
    text="Collection status",
    font=("Arial", 11, "bold"),
)
status_label.pack(anchor="w", padx=25, pady=(8, 2))

status_box = scrolledtext.ScrolledText(
    root,
    width=130,
    height=7,
    font=("Consolas", 10),
)
status_box.pack(padx=25, pady=8)

status_box.insert(
    tk.END,
    "Ready to collect guided session-aligned data.\n\n"
    f"Output folder:\n{SESSION_ROOT}\n\n"
    f"Features CSV:\n{FEATURES_CSV}\n\n"
    "This version records:\n"
    "1. Raw keystroke down/up events.\n"
    "2. Hold-time features.\n"
    "3. Inter-key delay features.\n"
    "4. Pause ratios.\n"
    "5. Correction ratios.\n"
    "6. Rhythm, burstiness, and fits-starts indicators.\n",
)

update_guided_instructions()

root.mainloop()
