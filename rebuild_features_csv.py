from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, List


PROJECT_ROOT = Path(__file__).resolve().parent
SESSION_ROOT = PROJECT_ROOT / "data" / "session_aligned"

KEYSTROKE_DIR = SESSION_ROOT / "keystrokes"
TEXT_DIR = SESSION_ROOT / "texts"
AUDIO_DIR = SESSION_ROOT / "audio"
IMAGE_DIR = SESSION_ROOT / "images"

OUTPUT_CSV = SESSION_ROOT / "retroactive_keystroke_features.csv"


FIELDNAMES = [
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


def find_optional_file(folder: Path, session_id: str) -> str:
    matches = list(folder.glob(f"{session_id}.*"))
    if not matches:
        return ""
    return str(matches[0].relative_to(SESSION_ROOT))


def load_text(session_id: str, payload: Dict[str, Any]) -> str:
    if "typed_text" in payload:
        return str(payload["typed_text"])

    text_file = TEXT_DIR / f"{session_id}.txt"
    if text_file.exists():
        return text_file.read_text(encoding="utf-8").strip()

    return ""


def build_row(json_path: Path) -> Dict[str, Any]:
    payload = json.loads(json_path.read_text(encoding="utf-8"))

    session_id = payload.get("session_id", json_path.stem)
    label = payload.get("label", "")
    created_at = payload.get("created_at", "")

    typed_text = load_text(session_id, payload)
    features = payload.get("features", {})

    audio_path = find_optional_file(AUDIO_DIR, session_id)
    image_path = find_optional_file(IMAGE_DIR, session_id)

    return {
        "session_id": session_id,
        "label": label,
        "created_at": created_at,
        "text_path": str((TEXT_DIR / f"{session_id}.txt").relative_to(SESSION_ROOT)),
        "keystroke_path": str(json_path.relative_to(SESSION_ROOT)),
        "audio_path": audio_path,
        "image_path": image_path,
        "typed_text": typed_text,
        "typed_text_length": len(typed_text),

        "keydown_count": payload.get("keydown_count", features.get("keydown_count", "")),
        "event_count": payload.get("event_count", ""),
        "expected_event_count": payload.get("expected_event_count", ""),
        "keystroke_validation_passed": payload.get("validation_passed", ""),
        "keystroke_validation_message": payload.get("validation_message", ""),

        "total_duration_sec": features.get("total_duration_sec", ""),
        "typing_speed_kps": features.get("typing_speed_kps", ""),
        "typing_speed_wpm": features.get("typing_speed_wpm", ""),
        "delay_mean": features.get("delay_mean", ""),
        "delay_std": features.get("delay_std", ""),
        "delay_min": features.get("delay_min", ""),
        "delay_max": features.get("delay_max", ""),
        "hold_mean": features.get("hold_mean", ""),
        "hold_std": features.get("hold_std", ""),
        "pause_count_1000": features.get("pause_count_1000", ""),
        "pause_count_2000": features.get("pause_count_2000", ""),
        "pause_count_5000": features.get("pause_count_5000", ""),
        "pause_ratio_1000": features.get("pause_ratio_1000", ""),
        "pause_ratio_2000": features.get("pause_ratio_2000", ""),
        "mental_block_ratio_5000": features.get("mental_block_ratio_5000", ""),
        "correction_count": features.get("correction_count", ""),
        "correction_ratio": features.get("correction_ratio", ""),
        "rhythm_consistency": features.get("rhythm_consistency", ""),
        "burstiness_proxy": features.get("burstiness_proxy", ""),
        "fits_starts_index": features.get("fits_starts_index", ""),

        "audio_provided": bool(audio_path),
        "image_provided": bool(image_path),
        "collection_mode": payload.get(
            "collection_mode",
            "guided_session_aligned_with_keystroke_biometrics",
        ),
        "notes": payload.get("notes", ""),
    }


def main() -> None:
    json_files = sorted(KEYSTROKE_DIR.glob("*.json"))

    rows: List[Dict[str, Any]] = []

    for json_path in json_files:
        try:
            rows.append(build_row(json_path))
        except Exception as error:
            print(f"Skipped {json_path.name}: {error}")

    with OUTPUT_CSV.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)

    print("Clean CSV rebuilt successfully.")
    print(f"Rows written: {len(rows)}")
    print(f"Output: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
