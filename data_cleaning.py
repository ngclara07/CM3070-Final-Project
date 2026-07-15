# === data_cleaning.py ===
# EmoSurv keystroke preprocessing pipeline
#
# Source dataset:
# EmoSurv: A typing biometric (Keystroke dynamics) dataset with emotion labels
# Reference: https://ieee-dataport.org/open-access/emosurv-typing-biometric-keystroke-dynamics-dataset-emotion-labels-created-using
#
# Important methodological note:
# EmoSurv provides emotion labels, not direct behavioural-state labels.
# Therefore, this script derives behavioural states using heuristic rules
# over keystroke dynamics features. The resulting labels are pseudo-labels
# for prototype training and baseline experimentation.
# 
# Behaviour labels are heuristic pseudo-labels derived from keystroke dynamics.
# They are not direct ground-truth cognitive labels.

from __future__ import annotations

import json
import os
from typing import Dict, List

import numpy as np
import pandas as pd


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data", "EmoSurv_IEEE")
OUTPUT_DIR = os.path.join(BASE_DIR, "emosurv_processed")

FIXED_CSV_PATH = os.path.join(DATA_DIR, "Fixed Text Typing Dataset.csv")
FREE_CSV_PATH = os.path.join(DATA_DIR, "Free Text Typing Dataset.csv")

os.makedirs(OUTPUT_DIR, exist_ok=True)


VALID_EMOTIONS = {"H", "S", "A", "C", "N"}

WINDOW_SIZE = 30
WINDOW_STEP = 15
MIN_KEYS_PER_WINDOW = 15

BEHAVIOUR_STATES = ["focused", "distracted", "fatigued", "overloaded"]


EXPECTED_CORE_COLUMNS = [
    "userid", "emotionindex", "index", "keycode",
    "keydown", "keyup",
    "d1u1", "d1u2", "d1d2",
    "u1d2", "u1u2",
    "d1u3", "d1d3",
    "answer",
]

OPTIONAL_COLUMNS = ["_id"]


def normalize_column_name(col: str) -> str:
    col = str(col).strip().replace("\ufeff", "")
    col = col.replace(".", "").replace(" ", "").replace("-", "").lower()

    mapping = {
        "userid": "userid",
        "emotionindex": "emotionindex",
        "index": "index",
        "keycode": "keycode",
        "keydown": "keydown",
        "keyup": "keyup",
        "d1u1": "d1u1",
        "d1u2": "d1u2",
        "d1d2": "d1d2",
        "u1d2": "u1d2",
        "u1u2": "u1u2",
        "d1u3": "d1u3",
        "d1d3": "d1d3",
        "answer": "answer",
        "_id": "_id",
        "id": "_id",
    }

    return mapping.get(col, col)


def load_emosurv_csv(path: str, source_name: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")

    df = pd.read_csv(
        path,
        sep=";",
        engine="python",
        dtype=str,
        keep_default_na=False,
    )

    df.columns = [normalize_column_name(c) for c in df.columns]

    allowed = set(EXPECTED_CORE_COLUMNS + OPTIONAL_COLUMNS)
    df = df[[c for c in df.columns if c in allowed]].copy()

    for col in EXPECTED_CORE_COLUMNS:
        if col not in df.columns:
            df[col] = np.nan

    ordered_cols = [c for c in OPTIONAL_COLUMNS if c in df.columns] + EXPECTED_CORE_COLUMNS
    df = df[ordered_cols]

    for col in df.columns:
        df[col] = df[col].astype(str).str.strip()

    df = df.replace({"": np.nan, "NA": np.nan, "nan": np.nan, "None": np.nan, "null": np.nan})

    numeric_cols = [
        "userid", "index", "keydown", "keyup",
        "d1u1", "d1u2", "d1d2",
        "u1d2", "u1u2",
        "d1u3", "d1d3",
    ]

    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["emotionindex"] = df["emotionindex"].astype(str).str.upper().str.strip()
    df["answer"] = df["answer"].astype(str).str.upper().str.strip()
    df["keycode"] = df["keycode"].astype(str).str.strip()

    df = df.dropna(subset=["userid", "emotionindex", "index", "keycode"])
    df = df[df["emotionindex"].isin(VALID_EMOTIONS)].copy()

    df["source_dataset"] = source_name

    return df.sort_values(["userid", "emotionindex", "index"]).reset_index(drop=True)


def mark_backspace_delete(key: str) -> Dict[str, int]:
    if pd.isna(key):
        return {"is_backspace": 0, "is_delete": 0, "is_correction": 0}

    k = str(key).lower()

    is_backspace = int("backspace" in k or k in {"8", "\\b", "key.backspace"})
    is_delete = int("delete" in k or k in {"46", "key.delete"})

    return {
        "is_backspace": is_backspace,
        "is_delete": is_delete,
        "is_correction": int(is_backspace or is_delete),
    }


def enrich_key_flags(df: pd.DataFrame) -> pd.DataFrame:
    flags = df["keycode"].apply(mark_backspace_delete).apply(pd.Series)
    return pd.concat([df, flags], axis=1)


def add_basic_event_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["hold_time_raw"] = df["keyup"] - df["keydown"]
    df["hold_time"] = df["d1u1"].where(df["d1u1"].notna(), df["hold_time_raw"])

    df["is_pause_500"] = (df["d1d2"] > 500).astype(int)
    df["is_pause_1000"] = (df["d1d2"] > 1000).astype(int)
    df["is_pause_2000"] = (df["d1d2"] > 2000).astype(int)
    df["is_mental_block_5000"] = (df["d1d2"] > 5000).astype(int)

    df["valid_hold"] = ((df["hold_time"].notna()) & (df["hold_time"] >= 0)).astype(int)
    df["valid_delay"] = ((df["d1d2"].notna()) & (df["d1d2"] >= 0)).astype(int)

    return df


def clean_array(x) -> np.ndarray:
    arr = np.asarray(x, dtype=float)
    return arr[~np.isnan(arr)]


def safe_mean(x) -> float:
    arr = clean_array(x)
    return float(np.mean(arr)) if len(arr) else np.nan


def safe_std(x) -> float:
    arr = clean_array(x)
    return float(np.std(arr)) if len(arr) else np.nan


def safe_quantile(x, q: float) -> float:
    arr = clean_array(x)
    return float(np.quantile(arr, q)) if len(arr) else np.nan


def safe_cv(x) -> float:
    arr = clean_array(x)
    if len(arr) == 0:
        return np.nan
    mean = np.mean(arr)
    if mean <= 0:
        return np.nan
    return float(np.std(arr) / mean)


def compute_repeated_key_ratio(keys: List[str], delays: np.ndarray) -> float:
    if len(keys) < 2:
        return 0.0

    repeated = 0
    comparisons = 0

    for i in range(1, len(keys)):
        current_key = str(keys[i]).lower()
        previous_key = str(keys[i - 1]).lower()

        comparisons += 1

        delay = delays[i - 1] if i - 1 < len(delays) else np.nan

        if current_key == previous_key and (pd.isna(delay) or delay < 300):
            repeated += 1

    return float(repeated / max(comparisons, 1))


def compute_fits_starts_index(delays: np.ndarray) -> float:
    arr = clean_array(delays)

    if len(arr) < 3:
        return 0.0

    acceleration = np.diff(arr)
    return float(np.std(acceleration))


def build_windowed_samples(df: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict] = []

    group_cols = ["source_dataset", "userid", "emotionindex"]

    for group_key, g in df.groupby(group_cols, dropna=False):
        source_dataset, userid, emotionindex = group_key
        g = g.sort_values("index").reset_index(drop=True)

        if len(g) < MIN_KEYS_PER_WINDOW:
            continue

        window_id = 0

        for start in range(0, len(g), WINDOW_STEP):
            end = start + WINDOW_SIZE
            w = g.iloc[start:end].copy()

            if len(w) < MIN_KEYS_PER_WINDOW:
                continue

            window_id += 1

            keydown_start = w["keydown"].min()
            keyup_end = w["keyup"].max()

            duration = keyup_end - keydown_start if pd.notna(keydown_start) and pd.notna(keyup_end) else np.nan

            delays = w["d1d2"].dropna().values
            holds = w["hold_time"].dropna().values

            key_count = len(w)

            typing_speed = np.nan
            if pd.notna(duration) and duration > 0:
                typing_speed = key_count / duration

            delay_q25 = safe_quantile(delays, 0.25)
            delay_q75 = safe_quantile(delays, 0.75)

            burstiness_proxy = (
                delay_q75 - delay_q25
                if pd.notna(delay_q25) and pd.notna(delay_q75)
                else np.nan
            )

            keys = [str(k).lower() for k in w["keycode"].tolist()]

            repeated_key_ratio = compute_repeated_key_ratio(keys, delays)
            rapid_burst_ratio = float(np.mean(delays < 120)) if len(delays) else 0.0
            mental_block_ratio = float(w["is_mental_block_5000"].mean())
            fits_starts_index = compute_fits_starts_index(delays)

            delay_mean = safe_mean(delays)
            delay_std = safe_std(delays)
            hold_mean = safe_mean(holds)
            hold_std = safe_std(holds)

            rhythm_consistency = 1.0 / (1.0 + safe_cv(delays)) if pd.notna(safe_cv(delays)) else 0.0

            correction_ratio = float(w["is_correction"].mean())
            error_rate_proxy = min(1.0, correction_ratio + repeated_key_ratio)

            row = {
                "source_dataset": source_dataset,
                "userid": int(userid),
                "emotionindex": emotionindex,
                "sample_id": f"{source_dataset}_u{int(userid)}_{emotionindex}_w{window_id}",

                "window_start_index": int(w["index"].min()),
                "window_end_index": int(w["index"].max()),

                "keystroke_count": int(key_count),
                "chunk_duration": float(duration) if pd.notna(duration) else np.nan,

                "typing_speed": float(typing_speed) if pd.notna(typing_speed) else np.nan,

                "hold_mean": hold_mean,
                "hold_std": hold_std,

                "delay_mean": delay_mean,
                "delay_std": delay_std,
                "delay_cv": safe_cv(delays),

                "pause_ratio_500": float(w["is_pause_500"].mean()),
                "pause_ratio_1000": float(w["is_pause_1000"].mean()),
                "pause_ratio_2000": float(w["is_pause_2000"].mean()),
                "mental_block_ratio_5000": mental_block_ratio,

                "correction_ratio": correction_ratio,
                "repeated_key_ratio": repeated_key_ratio,
                "rapid_burst_ratio": rapid_burst_ratio,
                "burstiness_proxy": burstiness_proxy,
                "fits_starts_index": fits_starts_index,
                "rhythm_consistency": rhythm_consistency,
                "error_rate_proxy": error_rate_proxy,

                "valid_hold_ratio": float(w["valid_hold"].mean()),
                "valid_delay_ratio": float(w["valid_delay"].mean()),
            }

            rows.append(row)

    return pd.DataFrame(rows)


def normalize_features(features: pd.DataFrame) -> pd.DataFrame:
    df = features.copy()

    numeric_cols = df.select_dtypes(include=[np.number]).columns

    for col in numeric_cols:
        mu = df[col].mean(skipna=True)
        std = df[col].std(skipna=True)

        if pd.isna(std) or std == 0:
            df[col + "_z"] = 0.0
        else:
            df[col + "_z"] = (df[col] - mu) / std

    return df


def derive_behaviour_state(row: pd.Series) -> str:
    def z(name: str) -> float:
        value = row.get(name + "_z", 0.0)
        return 0.0 if pd.isna(value) else float(value)

    focused_score = (
        1.3 * z("typing_speed")
        + 1.2 * z("rhythm_consistency")
        - 1.0 * z("error_rate_proxy")
        - 0.8 * z("pause_ratio_1000")
        - 0.7 * z("delay_std")
        - 0.5 * z("mental_block_ratio_5000")
    )

    distracted_score = (
        1.2 * z("pause_ratio_1000")
        + 0.9 * z("pause_ratio_2000")
        + 0.8 * z("delay_std")
        + 0.6 * z("delay_cv")
        - 0.3 * z("error_rate_proxy")
    )

    fatigued_score = (
        -1.1 * z("typing_speed")
        + 1.0 * z("delay_mean")
        + 0.8 * z("hold_mean")
        + 0.8 * z("mental_block_ratio_5000")
        + 0.5 * z("correction_ratio")
        - 0.4 * z("rapid_burst_ratio")
    )

    overloaded_score = (
        1.2 * z("error_rate_proxy")
        + 1.1 * z("fits_starts_index")
        + 0.9 * z("burstiness_proxy")
        + 0.8 * z("delay_std")
        + 0.8 * z("correction_ratio")
        + 0.6 * z("rapid_burst_ratio")
        - 0.3 * z("rhythm_consistency")
    )

    scores = {
        "focused": focused_score,
        "distracted": distracted_score,
        "fatigued": fatigued_score,
        "overloaded": overloaded_score,
    }

    return max(scores, key=scores.get)


def add_behaviour_labels(features: pd.DataFrame) -> pd.DataFrame:
    if features.empty:
        df = features.copy()
        df["behaviour_state"] = pd.Series(dtype="object")
        return df

    df = normalize_features(features)
    df["behaviour_state"] = df.apply(derive_behaviour_state, axis=1)
    return df


def process_one_dataset(path: str, source_name: str, cleaned_name: str, samples_name: str):
    print(f"\nProcessing dataset: {source_name}")
    print(f"Reading file: {path}")

    events = load_emosurv_csv(path, source_name)
    events = enrich_key_flags(events)
    events = add_basic_event_features(events)

    clean_path = os.path.join(OUTPUT_DIR, cleaned_name)
    events.to_csv(clean_path, index=False)

    samples = build_windowed_samples(events)

    samples_path = os.path.join(OUTPUT_DIR, samples_name)
    samples.to_csv(samples_path, index=False)

    print(f"Event rows: {len(events)}")
    print(f"Window samples: {len(samples)}")

    return events, samples


def build_summary(combined: pd.DataFrame) -> Dict:
    return {
        "methodological_note": (
            "Behaviour labels are pseudo-labels derived from typing dynamics. "
            "Focus is represented by high speed, rhythm consistency, and low error proxy. "
            "Distraction is represented by long pauses and inconsistent speed. "
            "Fatigue is represented by lower speed, higher IKI, backspaces, and mental blocks. "
            "Overload/stress is represented by fits-and-starts rhythm and high error proxy."
        ),
        "combined_training_samples": int(len(combined)),
        "behaviour_distribution": combined["behaviour_state"].value_counts().to_dict()
        if not combined.empty else {},
        "feature_columns": list(combined.columns),
        "window_config": {
            "window_size": WINDOW_SIZE,
            "window_step": WINDOW_STEP,
            "min_keys_per_window": MIN_KEYS_PER_WINDOW,
        },
    }


def main() -> None:
    print("==========================================")
    print("Keystroke dynamics preprocessing pipeline")
    print("==========================================")

    fixed_events, fixed_samples = process_one_dataset(
        FIXED_CSV_PATH,
        "fixed_text",
        "clean_fixed_events.csv",
        "fixed_window_samples.csv",
    )

    free_events, free_samples = process_one_dataset(
        FREE_CSV_PATH,
        "free_text",
        "clean_free_events.csv",
        "free_window_samples.csv",
    )

    combined_unlabelled = pd.concat([fixed_samples, free_samples], ignore_index=True)

    combined_unlabelled_path = os.path.join(OUTPUT_DIR, "combined_window_samples_unlabelled.csv")
    combined_unlabelled.to_csv(combined_unlabelled_path, index=False)

    combined = add_behaviour_labels(combined_unlabelled)
    combined = combined[combined["behaviour_state"].isin(BEHAVIOUR_STATES)].copy()

    combined_path = os.path.join(OUTPUT_DIR, "combined_behaviour_samples.csv")
    combined.to_csv(combined_path, index=False)

    summary_path = os.path.join(OUTPUT_DIR, "summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(build_summary(combined), f, indent=2)

    print("\nProcessing complete.")
    print(f"Training file: {combined_path}")
    print(f"Summary file:  {summary_path}")

    print("\nBehaviour distribution:")
    print(combined["behaviour_state"].value_counts())


if __name__ == "__main__":
    main()
