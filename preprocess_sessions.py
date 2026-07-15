# preprocess_sessions.py

import json
import re
from pathlib import Path

import pandas as pd
from sklearn.preprocessing import StandardScaler


DATASET_ROOT = Path("data/session_aligned")

TEXT_DIR = DATASET_ROOT / "texts"
KEYSTROKE_DIR = DATASET_ROOT / "keystrokes"
AUDIO_DIR = DATASET_ROOT / "audio"
IMAGE_DIR = DATASET_ROOT / "images"

OUTPUT_DIR = Path("data/processed")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

LABELS = {"focused", "fatigued", "distracted", "overloaded"}


def clean_text(text: str) -> str:
    text = text.strip()
    text = re.sub(r"\s+", " ", text)
    return text


def load_keystroke_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    features = data.get("features", {})

    return {
        "session_id": data.get("session_id"),
        "label": data.get("label"),
        "created_at": data.get("created_at"),
        "keydown_count_json": data.get("keydown_count"),
        "event_count": data.get("event_count"),
        "expected_event_count": data.get("expected_event_count"),
        "validation_passed": data.get("validation_passed"),
        "validation_message": data.get("validation_message"),
        **features,
    }


def validate_row(row: dict) -> list[str]:
    problems = []

    if row["label"] not in LABELS:
        problems.append("invalid_label")

    if row["event_count"] != row["expected_event_count"]:
        problems.append("event_count_mismatch")

    if row["validation_passed"] is not True:
        problems.append("validation_failed")

    return problems


def build_master_table() -> pd.DataFrame:
    records = []

    json_files = sorted(KEYSTROKE_DIR.glob("*.json"))

    if not json_files:
        raise FileNotFoundError(f"No keystroke JSON files found in: {KEYSTROKE_DIR}")

    for json_path in json_files:
        row = load_keystroke_json(json_path)
        session_id = row["session_id"]

        text_path = TEXT_DIR / f"{session_id}.txt"
        audio_path = AUDIO_DIR / f"{session_id}.mp3"
        image_path = IMAGE_DIR / f"{session_id}.jpg"

        row["text_path"] = str(text_path)
        row["keystroke_path"] = str(json_path)
        row["audio_path"] = str(audio_path)
        row["image_path"] = str(image_path)

        row["text_exists"] = text_path.exists()
        row["audio_exists"] = audio_path.exists()
        row["image_exists"] = image_path.exists()

        row["text"] = clean_text(text_path.read_text(encoding="utf-8")) if text_path.exists() else ""

        row["problems"] = validate_row(row)

        if not row["text_exists"]:
            row["problems"].append("missing_text")
        if not row["audio_exists"]:
            row["problems"].append("missing_audio")
        if not row["image_exists"]:
            row["problems"].append("missing_image")

        records.append(row)

    df = pd.DataFrame(records)

    if df["session_id"].duplicated().any():
        duplicated = df[df["session_id"].duplicated()]["session_id"].tolist()
        raise ValueError(f"Duplicate session IDs found: {duplicated}")

    df["is_clean"] = df["problems"].apply(lambda x: len(x) == 0)

    return df


def preprocess_numeric_features(df: pd.DataFrame) -> tuple[pd.DataFrame, StandardScaler]:
    exclude = {
        "session_id", "label", "created_at",
        "text_path", "keystroke_path", "audio_path", "image_path",
        "text", "validation_message", "problems",
        "validation_passed", "text_exists", "audio_exists", "image_exists",
        "is_clean",
    }

    numeric_cols = [
        col for col in df.columns
        if col not in exclude and pd.api.types.is_numeric_dtype(df[col])
    ]

    clean_df = df[df["is_clean"]].copy()

    scaler = StandardScaler()
    clean_df[numeric_cols] = scaler.fit_transform(clean_df[numeric_cols])

    return clean_df, scaler


def main():
    master_df = build_master_table()

    master_path = OUTPUT_DIR / "master_sessions_raw.csv"
    master_df.to_csv(master_path, index=False)

    clean_df, scaler = preprocess_numeric_features(master_df)

    clean_path = OUTPUT_DIR / "master_sessions_clean_scaled.csv"
    clean_df.to_csv(clean_path, index=False)

    label_summary = clean_df["label"].value_counts().sort_index()
    label_summary.to_csv(OUTPUT_DIR / "label_distribution.csv")

    invalid = master_df[~master_df["is_clean"]]
    if len(invalid) > 0:
        invalid_path = OUTPUT_DIR / "invalid_sessions.csv"
        invalid.to_csv(invalid_path, index=False)
    else:
        invalid_path = None

    print("Preprocessing complete.")
    print(f"Raw master table: {master_path}")
    print(f"Clean scaled table: {clean_path}")
    print(f"Label distribution: {OUTPUT_DIR / 'label_distribution.csv'}")

    if invalid_path:
        print(f"Invalid sessions: {invalid_path}")
    else:
        print("No invalid sessions found.")


if __name__ == "__main__":
    main()
