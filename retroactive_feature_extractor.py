# === retroactive_feature_extractor.py ===
# SenseFuzeAI - Retroactive Keystroke Feature Extractor
# Adds biometric features to previously collected keystroke JSON files.

from __future__ import annotations

import csv
import json
import statistics
from pathlib import Path
from typing import Any, Dict, List


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent

SESSION_ROOT = PROJECT_ROOT / "data" / "session_aligned"
KEYSTROKE_DIR = SESSION_ROOT / "keystrokes"
TEXT_DIR = SESSION_ROOT / "texts"

OUTPUT_CSV = SESSION_ROOT / "retroactive_keystroke_features.csv"

TARGET_LABEL = "fatigued"


# ============================================================
# FEATURE EXTRACTION HELPERS
# ============================================================

def safe_mean(values: List[float]) -> float:
    return statistics.mean(values) if values else 0.0


def safe_std(values: List[float]) -> float:
    return statistics.stdev(values) if len(values) >= 2 else 0.0


def extract_keystroke_features(
    events: List[Dict[str, Any]],
    typed_text: str,
) -> Dict[str, Any]:
    downs = [e for e in events if e.get("type") == "down"]
    down_times = [e["timestamp_perf"] for e in downs if "timestamp_perf" in e]

    delays = [
        down_times[i] - down_times[i - 1]
        for i in range(1, len(down_times))
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
        1 for e in downs
        if e.get("key") in {"backspace", "delete"}
    )

    pauses_1000 = [d for d in delays if d >= 1.0]
    pauses_2000 = [d for d in delays if d >= 2.0]
    pauses_5000 = [d for d in delays if d >= 5.0]

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


# ============================================================
# FILE HELPERS
# ============================================================

def load_typed_text(payload: Dict[str, Any], json_path: Path) -> str:
    if "typed_text" in payload:
        return str(payload["typed_text"])

    text_path = TEXT_DIR / f"{json_path.stem}.txt"

    if text_path.exists():
        return text_path.read_text(encoding="utf-8").strip()

    return ""


def build_csv_row(
    json_path: Path,
    payload: Dict[str, Any],
    features: Dict[str, Any],
    typed_text: str,
) -> Dict[str, Any]:
    return {
        "session_id": payload.get("session_id", json_path.stem),
        "label": payload.get("label", ""),
        "keystroke_file": str(json_path.relative_to(SESSION_ROOT)),
        "typed_text": typed_text,
        "typed_text_length": len(typed_text),
        **features,
        "validation_passed": payload.get("validation_passed", ""),
        "event_count": payload.get("event_count", len(payload.get("events", []))),
        "expected_event_count": payload.get("expected_event_count", ""),
    }


# ============================================================
# MAIN PROCESS
# ============================================================

def main() -> None:
    if not KEYSTROKE_DIR.exists():
        raise FileNotFoundError(f"Keystroke directory not found: {KEYSTROKE_DIR}")

    json_files = sorted(KEYSTROKE_DIR.glob(f"{TARGET_LABEL}_*.json"))

    if not json_files:
        print(f"No JSON files found for label: {TARGET_LABEL}")
        print(f"Checked folder: {KEYSTROKE_DIR}")
        return

    rows: List[Dict[str, Any]] = []

    updated_count = 0
    skipped_count = 0
    error_count = 0

    for json_path in json_files:
        try:
            payload = json.loads(json_path.read_text(encoding="utf-8"))

            label = str(payload.get("label", "")).lower()

            if label != TARGET_LABEL:
                skipped_count += 1
                continue

            events = payload.get("events", [])

            if not isinstance(events, list) or not events:
                print(f"Skipped empty/invalid events: {json_path.name}")
                skipped_count += 1
                continue

            typed_text = load_typed_text(payload, json_path)

            features = extract_keystroke_features(events, typed_text)

            payload["features"] = features
            payload["feature_extraction_version"] = "retroactive_v1"

            json_path.write_text(
                json.dumps(payload, indent=4),
                encoding="utf-8",
            )

            row = build_csv_row(json_path, payload, features, typed_text)
            rows.append(row)

            updated_count += 1

        except Exception as error:
            error_count += 1
            print(f"ERROR processing {json_path.name}: {error}")

    if rows:
        fieldnames = list(rows[0].keys())

        with OUTPUT_CSV.open("w", encoding="utf-8-sig", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    print("=" * 70)
    print("Retroactive feature extraction complete.")
    print("=" * 70)
    print(f"Target label:        {TARGET_LABEL}")
    print(f"JSON files found:    {len(json_files)}")
    print(f"Updated files:       {updated_count}")
    print(f"Skipped files:       {skipped_count}")
    print(f"Errors:              {error_count}")
    print(f"Output CSV:          {OUTPUT_CSV}")
    print("=" * 70)


if __name__ == "__main__":
    main()
