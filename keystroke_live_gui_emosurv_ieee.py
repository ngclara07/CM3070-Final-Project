# ============================================================================
# keystroke_live_gui_emosurv_ieee.py
#
# SenseFuzeAI
# EmoSurv IEEE Keystroke Dynamics Comparison + Live GUI
#
# FINAL COMPARISON VERSION
#
# Source datasets used:
#   data/EmoSurv_IEEE/Fixed Text Typing Dataset.csv
#   data/EmoSurv_IEEE/Free Text Typing Dataset.csv
#
# Explicitly excluded:
#   Frequency Dataset.csv
#   Participants Information.csv
#
# ---------------------------------------------------------------------------
# METHODOLOGICAL WARNING
# ---------------------------------------------------------------------------
#
# EmoSurv contains emotion labels:
#
#   A = Angry
#   C = Calm
#   H = Happy
#   N = Neutral
#   S = Sad
#
# It does NOT provide SenseFuzeAI behavioural-state labels.
#
# Therefore the following SenseFuzeAI outputs are WEAKLY SUPERVISED PROXIES:
#
#   focused
#   distracted
#   fatigued
#   overloaded
#
# These must NOT be described as original EmoSurv behavioural ground truth.
#
# Proxy construction:
#
#   Calm   -> focused
#   Angry  -> overloaded
#   Sad    -> fatigued
#
#   Neutral / Happy:
#       high typing irregularity -> distracted
#       otherwise                -> focused
#
# This script is intended for an external-dataset comparison against the
# project's own session-aligned behavioural dataset.
#
# ---------------------------------------------------------------------------
# FINAL IMPROVEMENTS
# ---------------------------------------------------------------------------
#
# 1. Participant-independent GroupShuffleSplit retained.
# 2. Baseline and balanced Random Forest candidates compared.
# 3. Selection prioritises macro F1, then macro recall, then accuracy.
# 4. Per-class precision / recall / F1 are written to reports.
# 5. Training feature distributions are saved.
# 6. Live GUI compares live feature values against training distributions.
# 7. Live out-of-distribution diagnostics are displayed.
# ============================================================================

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import tkinter as tk
from tkinter import messagebox

from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# ============================================================================
# PROJECT PATHS
# ============================================================================

ROOT_DIR = Path(__file__).resolve().parent

EMOSURV_DIR = ROOT_DIR / "data" / "EmoSurv_IEEE"

FIXED_TEXT_PATH = EMOSURV_DIR / "Fixed Text Typing Dataset.csv"
FREE_TEXT_PATH = EMOSURV_DIR / "Free Text Typing Dataset.csv"

OUTPUT_DIR = (
    ROOT_DIR
    / "data"
    / "processed"
    / "emosurv_ieee_comparison"
)

MODEL_DIR = (
    ROOT_DIR
    / "models"
    / "keystroke_emosurv_demo"
)

MODEL_PATH = (
    MODEL_DIR
    / "keystroke_emosurv_pipeline.joblib"
)

FEATURE_COLUMNS_PATH = (
    MODEL_DIR
    / "feature_columns.json"
)

METADATA_PATH = (
    MODEL_DIR
    / "metadata.json"
)

FEATURE_DISTRIBUTION_PATH = (
    MODEL_DIR
    / "training_feature_distribution.json"
)

WINDOW_DATASET_PATH = (
    OUTPUT_DIR
    / "emosurv_behaviour_proxy_windows.csv"
)

MODEL_COMPARISON_PATH = (
    OUTPUT_DIR
    / "model_comparison.csv"
)

CLASSIFICATION_REPORT_PATH = (
    OUTPUT_DIR
    / "classification_report.csv"
)

CONFUSION_MATRIX_PATH = (
    OUTPUT_DIR
    / "confusion_matrix.csv"
)

EVALUATION_REPORT_PATH = (
    OUTPUT_DIR
    / "emosurv_keystroke_evaluation.txt"
)

LIVE_LOG_PATH = (
    OUTPUT_DIR
    / "emosurv_live_predictions.csv"
)


# ============================================================================
# CONFIGURATION
# ============================================================================

EMOTION_CODES = {
    "A": "Angry",
    "C": "Calm",
    "H": "Happy",
    "N": "Neutral",
    "S": "Sad",
}

BEHAVIOURAL_CLASSES = [
    "focused",
    "distracted",
    "fatigued",
    "overloaded",
]

WINDOW_SIZE = 40
WINDOW_STEP = 40
MIN_WINDOW_SIZE = 20

MIN_LIVE_KEYPRESSES = 20

TEST_SIZE = 0.20
RANDOM_STATE = 42

# ---------------------------------------------------------------------------
# Plausibility limits for milliseconds
# ---------------------------------------------------------------------------

DWELL_MIN_MS = 5.0
DWELL_MAX_MS = 2000.0

DD_MIN_MS = 5.0
DD_MAX_MS = 10000.0

UD_MIN_MS = -2000.0
UD_MAX_MS = 10000.0

# ---------------------------------------------------------------------------
# Live feature-distribution diagnostics
#
# Values with absolute robust z-score above this limit are treated as
# substantially outside the training distribution.
# ---------------------------------------------------------------------------

LIVE_OOD_Z_THRESHOLD = 3.5

# Fraction of features allowed to be strongly OOD before the entire live
# sample receives a distribution warning.
LIVE_OOD_FEATURE_RATIO_WARNING = 0.25


# ============================================================================
# FEATURE SCHEMA
# ============================================================================

FEATURE_COLUMNS = [
    "dwell_mean_ms",
    "dwell_std_ms",
    "dwell_median_ms",
    "dwell_min_ms",
    "dwell_max_ms",

    "dd_mean_ms",
    "dd_std_ms",
    "dd_median_ms",
    "dd_min_ms",
    "dd_max_ms",

    "ud_mean_ms",
    "ud_std_ms",
    "ud_median_ms",
    "ud_min_ms",
    "ud_max_ms",

    "typing_speed_kps",

    "pause_ratio_500",
    "pause_ratio_1000",
    "pause_ratio_2000",

    "correction_ratio",
    "space_ratio",

    "rhythm_cv",
    "overlap_ratio",
]


# ============================================================================
# GENERAL UTILITIES
# ============================================================================

def safe_float(
    value: Any,
    default: float = 0.0,
) -> float:
    """
    Convert a value to finite float.
    """
    try:
        result = float(value)

        if math.isfinite(result):
            return result

    except (TypeError, ValueError):
        pass

    return default


def safe_mean(values: list[float]) -> float:
    if not values:
        return 0.0

    return float(statistics.mean(values))


def safe_std(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0

    return float(statistics.stdev(values))


def safe_median(values: list[float]) -> float:
    if not values:
        return 0.0

    return float(statistics.median(values))


def safe_min(values: list[float]) -> float:
    if not values:
        return 0.0

    return float(min(values))


def safe_max(values: list[float]) -> float:
    if not values:
        return 0.0

    return float(max(values))


def confidence_level(gap: float) -> str:
    """
    Convert top-two probability gap into an interpretable confidence category.
    """
    if gap >= 0.35:
        return "High"

    if gap >= 0.15:
        return "Medium"

    return "Low"


# ============================================================================
# EMOSURV NUMBER PARSING
# ============================================================================

def parse_emosurv_number(value: Any) -> float:
    """
    Parse EmoSurv timing values.

    Examples:
        "90"
        "-273"
        "1,58E+12"

    Decimal commas are converted to decimal points.
    """

    if value is None:
        return float("nan")

    try:
        if pd.isna(value):
            return float("nan")
    except Exception:
        pass

    text = str(value).strip()

    if not text:
        return float("nan")

    if text.lower() in {
        "nan",
        "none",
        "null",
        "na",
    }:
        return float("nan")

    text = text.replace(",", ".")

    try:
        value_float = float(text)

        if math.isfinite(value_float):
            return value_float

    except ValueError:
        pass

    return float("nan")


def clean_numeric_series(
    series: pd.Series,
    minimum: float,
    maximum: float,
) -> pd.Series:
    values = series.map(parse_emosurv_number)

    return values.where(
        (values >= minimum)
        & (values <= maximum)
    )


# ============================================================================
# KEY NORMALISATION
# ============================================================================

def normalise_key(value: Any) -> str:
    if value is None:
        return ""

    try:
        if pd.isna(value):
            return ""
    except Exception:
        pass

    text = str(value)
    lowered = text.lower()

    if text in {"\\b", "\b"}:
        return "backspace"

    if lowered in {
        "backspace",
        "delete",
        "del",
        "\\u0008",
        "\\x08",
    }:
        return "backspace"

    if text == " ":
        return "space"

    if lowered in {
        "space",
        "spacebar",
    }:
        return "space"

    return lowered


# ============================================================================
# ROBUST CSV LOADING
# ============================================================================

def read_emosurv_csv(path: Path) -> pd.DataFrame:
    """
    Robust loader for EmoSurv CSV files.

    The datasets are normally semicolon-separated while timing values
    may contain decimal commas.
    """

    attempts = [
        (";", "utf-8"),
        (";", "utf-8-sig"),
        (";", "latin-1"),
        ("\t", "utf-8"),
    ]

    errors: list[str] = []

    for separator, encoding in attempts:
        try:
            df = pd.read_csv(
                path,
                sep=separator,
                engine="python",
                encoding=encoding,
                dtype=str,
                keep_default_na=True,
            )

            if len(df.columns) < 8:
                raise ValueError(
                    f"Only {len(df.columns)} columns detected."
                )

            print(
                f"CSV parsing successful: "
                f"delimiter={separator!r}, "
                f"encoding={encoding!r}"
            )

            return df

        except Exception as exc:
            errors.append(
                f"delimiter={separator!r}, "
                f"encoding={encoding!r}: {exc}"
            )

    # -----------------------------------------------------------------------
    # Final fallback
    # -----------------------------------------------------------------------

    try:
        df = pd.read_csv(
            path,
            sep=None,
            engine="python",
            encoding="utf-8-sig",
            dtype=str,
            keep_default_na=True,
        )

        if len(df.columns) < 8:
            raise ValueError(
                f"Only {len(df.columns)} columns detected."
            )

        print(
            "CSV parsing successful using automatic delimiter detection."
        )

        return df

    except Exception as exc:
        errors.append(
            f"automatic delimiter detection: {exc}"
        )

        raise RuntimeError(
            f"Could not parse EmoSurv file:\n"
            f"{path}\n\n"
            "Attempted configurations:\n"
            + "\n".join(
                f"  - {entry}"
                for entry in errors
            )
        ) from exc


# ============================================================================
# LOAD ONE EMOSURV SOURCE
# ============================================================================

def load_emosurv_file(
    path: Path,
    source_type: str,
) -> pd.DataFrame:
    """
    Load one EmoSurv dataset.

    emotionLabel is OPTIONAL.

    The canonical readable label is derived from emotionIndex when necessary.
    """

    if not path.exists():
        raise FileNotFoundError(
            f"Required EmoSurv dataset not found:\n{path}"
        )

    print()
    print(f"Loading {path.name}...")

    df = read_emosurv_csv(path)

    df.columns = [
        str(column).strip()
        for column in df.columns
    ]

    print(f"Rows detected   : {len(df):,}")
    print(f"Columns detected: {len(df.columns)}")
    print(f"Column names    : {list(df.columns)}")

    # -----------------------------------------------------------------------
    # Standardise participant field
    # -----------------------------------------------------------------------

    if "userId" in df.columns:
        df = df.rename(
            columns={
                "userId": "user_id",
            }
        )

    elif "userid" in df.columns:
        df = df.rename(
            columns={
                "userid": "user_id",
            }
        )

    else:
        raise ValueError(
            f"{path.name} contains neither 'userId' nor 'userid'.\n\n"
            f"Detected columns:\n"
            f"{list(df.columns)}"
        )

    # -----------------------------------------------------------------------
    # Validate only RAW source columns
    # -----------------------------------------------------------------------

    required_columns = {
        "user_id",
        "emotionIndex",
        "index",
        "keyCode",
        "D1U1",
        "D1D2",
        "U1D2",
    }

    missing_columns = (
        required_columns
        - set(df.columns)
    )

    if missing_columns:
        raise ValueError(
            f"{path.name} is missing required columns:\n"
            f"{sorted(missing_columns)}\n\n"
            f"Detected columns:\n"
            f"{list(df.columns)}"
        )

    # -----------------------------------------------------------------------
    # Participant
    # -----------------------------------------------------------------------

    df["user_id"] = (
        df["user_id"]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    # -----------------------------------------------------------------------
    # Emotion code
    # -----------------------------------------------------------------------

    df["emotion_code"] = (
        df["emotionIndex"]
        .fillna("")
        .astype(str)
        .str.strip()
        .str.upper()
    )

    derived_emotion_labels = (
        df["emotion_code"]
        .map(EMOTION_CODES)
    )

    # -----------------------------------------------------------------------
    # Existing enriched emotionLabel is optional
    # -----------------------------------------------------------------------

    if "emotionLabel" in df.columns:
        existing = (
            df["emotionLabel"]
            .fillna("")
            .astype(str)
            .str.strip()
        )

        df["emotion_label"] = existing.where(
            existing != "",
            derived_emotion_labels,
        )

        print(
            "emotionLabel detected and preserved where available."
        )

    else:
        df["emotion_label"] = derived_emotion_labels

        print(
            "NOTE: emotionLabel is not present in the raw dataset."
        )

        print(
            "      emotion_label was derived from emotionIndex."
        )

    # -----------------------------------------------------------------------
    # Remove unknown emotions
    # -----------------------------------------------------------------------

    valid_emotion_mask = (
        df["emotion_code"]
        .isin(EMOTION_CODES.keys())
    )

    invalid_emotion_count = int(
        (~valid_emotion_mask).sum()
    )

    if invalid_emotion_count > 0:
        print(
            f"WARNING: Removing {invalid_emotion_count:,} "
            "rows with invalid emotion codes."
        )

    df = df[
        valid_emotion_mask
    ].copy()

    # -----------------------------------------------------------------------
    # Ordering
    # -----------------------------------------------------------------------

    df["index_numeric"] = pd.to_numeric(
        df["index"],
        errors="coerce",
    )

    # -----------------------------------------------------------------------
    # Timing features
    # -----------------------------------------------------------------------

    df["dwell_ms"] = clean_numeric_series(
        df["D1U1"],
        DWELL_MIN_MS,
        DWELL_MAX_MS,
    )

    df["dd_ms"] = clean_numeric_series(
        df["D1D2"],
        DD_MIN_MS,
        DD_MAX_MS,
    )

    df["ud_ms"] = clean_numeric_series(
        df["U1D2"],
        UD_MIN_MS,
        UD_MAX_MS,
    )

    # -----------------------------------------------------------------------
    # Keys
    # -----------------------------------------------------------------------

    df["key_normalised"] = (
        df["keyCode"]
        .map(normalise_key)
    )

    # -----------------------------------------------------------------------
    # Source
    # -----------------------------------------------------------------------

    df["source_type"] = source_type
    df["source_file"] = path.name

    # -----------------------------------------------------------------------
    # Diagnostics
    # -----------------------------------------------------------------------

    print(f"Rows retained       : {len(df):,}")

    print(
        f"Valid dwell values  : "
        f"{int(df['dwell_ms'].notna().sum()):,}"
    )

    print(
        f"Valid D1D2 values   : "
        f"{int(df['dd_ms'].notna().sum()):,}"
    )

    print(
        f"Valid U1D2 values   : "
        f"{int(df['ud_ms'].notna().sum()):,}"
    )

    print(
        f"Emotion codes       : "
        f"{sorted(df['emotion_code'].unique())}"
    )

    print(
        f"Emotion labels      : "
        f"{sorted(df['emotion_label'].dropna().unique())}"
    )

    return df


# ============================================================================
# LOAD BOTH EMOSURV DATASETS
# ============================================================================

def load_emosurv_datasets() -> pd.DataFrame:
    print("=" * 88)
    print("Loading EmoSurv IEEE fixed-text and free-text datasets")
    print("=" * 88)

    fixed_df = load_emosurv_file(
        FIXED_TEXT_PATH,
        source_type="fixed_text",
    )

    free_df = load_emosurv_file(
        FREE_TEXT_PATH,
        source_type="free_text",
    )

    combined = pd.concat(
        [
            fixed_df,
            free_df,
        ],
        ignore_index=True,
        sort=False,
    )

    if combined.empty:
        raise ValueError(
            "Combined EmoSurv dataframe is empty."
        )

    print()
    print("=" * 88)
    print("COMBINED EMOSURV DATASET")
    print("=" * 88)

    print(
        f"Total raw keystroke rows : "
        f"{len(combined):,}"
    )

    print(
        f"Participants              : "
        f"{combined['user_id'].nunique()}"
    )

    print()
    print("Emotion distribution:")

    print(
        combined[
            "emotion_label"
        ]
        .value_counts()
    )

    print()
    print("Source distribution:")

    print(
        combined[
            "source_type"
        ]
        .value_counts()
    )

    return combined


# ============================================================================
# FEATURE EXTRACTION
# ============================================================================

def extract_window_features(
    window: pd.DataFrame,
) -> dict[str, float]:
    dwell = (
        window["dwell_ms"]
        .dropna()
        .astype(float)
        .tolist()
    )

    dd = (
        window["dd_ms"]
        .dropna()
        .astype(float)
        .tolist()
    )

    ud = (
        window["ud_ms"]
        .dropna()
        .astype(float)
        .tolist()
    )

    keys = (
        window["key_normalised"]
        .fillna("")
        .astype(str)
        .tolist()
    )

    positive_dd = [
        value
        for value in dd
        if value > 0
    ]

    mean_dd = safe_mean(
        positive_dd
    )

    dd_std = safe_std(
        positive_dd
    )

    typing_speed_kps = (
        1000.0 / mean_dd
        if mean_dd > 0
        else 0.0
    )

    def ratio_above(
        threshold: float,
    ) -> float:
        if not positive_dd:
            return 0.0

        return float(
            sum(
                value >= threshold
                for value in positive_dd
            )
            / len(positive_dd)
        )

    key_count = len(keys)

    correction_count = sum(
        key in {
            "backspace",
            "delete",
        }
        for key in keys
    )

    correction_ratio = (
        correction_count / key_count
        if key_count
        else 0.0
    )

    space_count = sum(
        key == "space"
        for key in keys
    )

    space_ratio = (
        space_count / key_count
        if key_count
        else 0.0
    )

    rhythm_cv = (
        dd_std / mean_dd
        if mean_dd > 0
        else 0.0
    )

    overlap_ratio = (
        sum(
            value < 0
            for value in ud
        )
        / len(ud)
        if ud
        else 0.0
    )

    features = {
        "dwell_mean_ms": safe_mean(dwell),
        "dwell_std_ms": safe_std(dwell),
        "dwell_median_ms": safe_median(dwell),
        "dwell_min_ms": safe_min(dwell),
        "dwell_max_ms": safe_max(dwell),

        "dd_mean_ms": mean_dd,
        "dd_std_ms": dd_std,
        "dd_median_ms": safe_median(positive_dd),
        "dd_min_ms": safe_min(positive_dd),
        "dd_max_ms": safe_max(positive_dd),

        "ud_mean_ms": safe_mean(ud),
        "ud_std_ms": safe_std(ud),
        "ud_median_ms": safe_median(ud),
        "ud_min_ms": safe_min(ud),
        "ud_max_ms": safe_max(ud),

        "typing_speed_kps": typing_speed_kps,

        "pause_ratio_500": ratio_above(500.0),
        "pause_ratio_1000": ratio_above(1000.0),
        "pause_ratio_2000": ratio_above(2000.0),

        "correction_ratio": correction_ratio,
        "space_ratio": space_ratio,

        "rhythm_cv": rhythm_cv,
        "overlap_ratio": overlap_ratio,
    }

    return {
        key: safe_float(value)
        for key, value in features.items()
    }


# ============================================================================
# BUILD TEMPORAL WINDOWS
# ============================================================================

def build_window_dataset(
    raw_df: pd.DataFrame,
) -> pd.DataFrame:
    print()
    print("=" * 88)
    print("BUILDING EMOSURV KEYSTROKE WINDOWS")
    print("=" * 88)

    rows: list[
        dict[str, Any]
    ] = []

    window_id = 0

    group_columns = [
        "source_type",
        "user_id",
        "emotion_code",
        "emotion_label",
    ]

    grouped = raw_df.groupby(
        group_columns,
        dropna=False,
        sort=False,
    )

    for (
        source_type,
        user_id,
        emotion_code,
        emotion_label,
    ), group in grouped:
        group = (
            group
            .sort_values(
                by="index_numeric",
                na_position="last",
            )
            .reset_index(
                drop=True
            )
        )

        for start in range(
            0,
            len(group),
            WINDOW_STEP,
        ):
            window = group.iloc[
                start:
                start + WINDOW_SIZE
            ].copy()

            if len(window) < MIN_WINDOW_SIZE:
                continue

            valid_dwell_count = int(
                window[
                    "dwell_ms"
                ]
                .notna()
                .sum()
            )

            valid_dd_count = int(
                window[
                    "dd_ms"
                ]
                .notna()
                .sum()
            )

            valid_ud_count = int(
                window[
                    "ud_ms"
                ]
                .notna()
                .sum()
            )

            if valid_dwell_count < 5:
                continue

            if valid_dd_count < 5:
                continue

            features = extract_window_features(
                window
            )

            rows.append(
                {
                    "window_id": window_id,
                    "source_type": str(source_type),
                    "user_id": str(user_id),
                    "emotion_code": str(emotion_code),
                    "emotion_label": str(emotion_label),

                    "window_keystrokes": int(
                        len(window)
                    ),

                    "valid_dwell_count": (
                        valid_dwell_count
                    ),

                    "valid_dd_count": (
                        valid_dd_count
                    ),

                    "valid_ud_count": (
                        valid_ud_count
                    ),

                    **features,
                }
            )

            window_id += 1

    windows = pd.DataFrame(
        rows
    )

    if windows.empty:
        raise ValueError(
            "No valid EmoSurv typing windows were generated."
        )

    print(
        f"Generated windows : {len(windows):,}"
    )

    print(
        f"Participants      : "
        f"{windows['user_id'].nunique()}"
    )

    print()
    print("Windows by emotion:")

    print(
        windows[
            "emotion_label"
        ]
        .value_counts()
    )

    return windows


# ============================================================================
# ROBUST Z-SCORE
# ============================================================================

def robust_zscore(
    series: pd.Series,
) -> pd.Series:
    values = pd.to_numeric(
        series,
        errors="coerce",
    )

    median = values.median()

    mad = (
        values
        .sub(median)
        .abs()
        .median()
    )

    if (
        pd.isna(mad)
        or mad <= 1e-12
    ):
        return pd.Series(
            np.zeros(
                len(values),
                dtype=float,
            ),
            index=values.index,
        )

    robust_scale = (
        1.4826
        * mad
    )

    return (
        values - median
    ) / robust_scale


# ============================================================================
# BEHAVIOURAL PROXY CONSTRUCTION
# ============================================================================

def assign_behaviour_proxy_labels(
    windows: pd.DataFrame,
) -> tuple[
    pd.DataFrame,
    float,
]:
    """
    Weakly supervised behavioural proxy construction.

    DO NOT replace this with direct typing-speed threshold rules.
    """

    df = windows.copy()

    df["z_rhythm_cv"] = robust_zscore(
        df["rhythm_cv"]
    )

    df["z_pause_ratio_1000"] = robust_zscore(
        df["pause_ratio_1000"]
    )

    df["z_correction_ratio"] = robust_zscore(
        df["correction_ratio"]
    )

    df["z_dd_std_ms"] = robust_zscore(
        df["dd_std_ms"]
    )

    df["distraction_index"] = (
        0.35 * df["z_rhythm_cv"]
        + 0.30 * df["z_pause_ratio_1000"]
        + 0.20 * df["z_correction_ratio"]
        + 0.15 * df["z_dd_std_ms"]
    )

    neutral_happy_mask = (
        df[
            "emotion_code"
        ]
        .isin(
            [
                "N",
                "H",
            ]
        )
    )

    calibration_values = (
        df.loc[
            neutral_happy_mask,
            "distraction_index",
        ]
        .dropna()
    )

    if calibration_values.empty:
        raise ValueError(
            "Unable to calibrate distraction proxy threshold."
        )

    distraction_threshold = float(
        calibration_values.quantile(
            0.60
        )
    )

    df["behaviour_proxy"] = ""

    # -----------------------------------------------------------------------
    # Direct semantic proxies
    # -----------------------------------------------------------------------

    df.loc[
        df["emotion_code"] == "C",
        "behaviour_proxy",
    ] = "focused"

    df.loc[
        df["emotion_code"] == "A",
        "behaviour_proxy",
    ] = "overloaded"

    df.loc[
        df["emotion_code"] == "S",
        "behaviour_proxy",
    ] = "fatigued"

    # -----------------------------------------------------------------------
    # Neutral / Happy require timing evidence
    # -----------------------------------------------------------------------

    distracted_mask = (
        neutral_happy_mask
        & (
            df[
                "distraction_index"
            ]
            >= distraction_threshold
        )
    )

    df.loc[
        distracted_mask,
        "behaviour_proxy",
    ] = "distracted"

    remaining_focused_mask = (
        neutral_happy_mask
        & ~distracted_mask
    )

    df.loc[
        remaining_focused_mask,
        "behaviour_proxy",
    ] = "focused"

    df = df[
        df[
            "behaviour_proxy"
        ]
        .isin(
            BEHAVIOURAL_CLASSES
        )
    ].copy()

    print()
    print("=" * 88)
    print("BEHAVIOURAL PROXY DISTRIBUTION")
    print("=" * 88)

    print(
        df[
            "behaviour_proxy"
        ]
        .value_counts()
        .reindex(
            BEHAVIOURAL_CLASSES,
            fill_value=0,
        )
    )

    print()
    print(
        f"Distraction-index threshold: "
        f"{distraction_threshold:.6f}"
    )

    return (
        df,
        distraction_threshold,
    )


# ============================================================================
# TRAINING FEATURE DISTRIBUTION
# ============================================================================

def calculate_feature_distribution(
    dataset: pd.DataFrame,
) -> dict[str, dict[str, float]]:
    """
    Calculate robust training-distribution statistics used by the live GUI.
    """

    distributions: dict[
        str,
        dict[str, float],
    ] = {}

    for feature in FEATURE_COLUMNS:
        series = pd.to_numeric(
            dataset[
                feature
            ],
            errors="coerce",
        )

        series = series[
            np.isfinite(
                series
            )
        ]

        if series.empty:
            distributions[
                feature
            ] = {
                "count": 0,
                "mean": 0.0,
                "std": 0.0,
                "median": 0.0,
                "mad": 0.0,
                "q01": 0.0,
                "q05": 0.0,
                "q25": 0.0,
                "q75": 0.0,
                "q95": 0.0,
                "q99": 0.0,
                "min": 0.0,
                "max": 0.0,
            }

            continue

        median = float(
            series.median()
        )

        mad = float(
            (
                series
                .sub(median)
                .abs()
                .median()
            )
        )

        distributions[
            feature
        ] = {
            "count": int(
                len(series)
            ),

            "mean": float(
                series.mean()
            ),

            "std": float(
                series.std(
                    ddof=1
                )
            )
            if len(series) > 1
            else 0.0,

            "median": median,
            "mad": mad,

            "q01": float(
                series.quantile(
                    0.01
                )
            ),

            "q05": float(
                series.quantile(
                    0.05
                )
            ),

            "q25": float(
                series.quantile(
                    0.25
                )
            ),

            "q75": float(
                series.quantile(
                    0.75
                )
            ),

            "q95": float(
                series.quantile(
                    0.95
                )
            ),

            "q99": float(
                series.quantile(
                    0.99
                )
            ),

            "min": float(
                series.min()
            ),

            "max": float(
                series.max()
            ),
        }

    return distributions


# ============================================================================
# MODEL BUILDERS
# ============================================================================

def build_baseline_model() -> Pipeline:
    """
    Baseline Random Forest.

    No class weighting.
    """

    return Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="median",
                ),
            ),

            (
                "scaler",
                StandardScaler(),
            ),

            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=500,
                    max_depth=None,
                    min_samples_split=4,
                    min_samples_leaf=2,
                    class_weight=None,
                    random_state=RANDOM_STATE,
                    n_jobs=-1,
                ),
            ),
        ]
    )


def build_balanced_model() -> Pipeline:
    """
    Balanced Random Forest candidate.

    balanced_subsample recalculates weights for each bootstrap sample.
    """

    return Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="median",
                ),
            ),

            (
                "scaler",
                StandardScaler(),
            ),

            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=500,
                    max_depth=None,
                    min_samples_split=4,
                    min_samples_leaf=2,
                    class_weight="balanced_subsample",
                    random_state=RANDOM_STATE,
                    n_jobs=-1,
                ),
            ),
        ]
    )


# ============================================================================
# GROUP-AWARE SPLIT
# ============================================================================

def find_group_aware_split(
    dataset: pd.DataFrame,
) -> tuple[
    np.ndarray,
    np.ndarray,
]:
    """
    Find a participant-independent split containing all four classes
    in both train and test where possible.
    """

    X = dataset[
        FEATURE_COLUMNS
    ]

    y = dataset[
        "behaviour_proxy"
    ]

    groups = dataset[
        "user_id"
    ]

    expected_classes = set(
        BEHAVIOURAL_CLASSES
    )

    best_split: (
        tuple[
            np.ndarray,
            np.ndarray,
        ]
        | None
    ) = None

    best_test_class_count = -1

    for attempt in range(
        500
    ):
        splitter = GroupShuffleSplit(
            n_splits=1,
            test_size=TEST_SIZE,
            random_state=(
                RANDOM_STATE
                + attempt
            ),
        )

        train_index, test_index = next(
            splitter.split(
                X,
                y,
                groups=groups,
            )
        )

        train_classes = set(
            y.iloc[
                train_index
            ]
        )

        test_classes = set(
            y.iloc[
                test_index
            ]
        )

        if not expected_classes.issubset(
            train_classes
        ):
            continue

        if len(
            test_classes
        ) > best_test_class_count:
            best_split = (
                train_index,
                test_index,
            )

            best_test_class_count = len(
                test_classes
            )

        if expected_classes.issubset(
            test_classes
        ):
            return (
                train_index,
                test_index,
            )

    if best_split is None:
        raise RuntimeError(
            "Could not construct participant-independent "
            "train/test partitions."
        )

    print(
        "WARNING: The best available test split did not "
        "contain all four classes."
    )

    return best_split


# ============================================================================
# METRIC COMPUTATION
# ============================================================================

def compute_classification_metrics(
    y_true: pd.Series,
    y_pred: np.ndarray,
) -> dict[str, Any]:
    accuracy = float(
        accuracy_score(
            y_true,
            y_pred,
        )
    )

    (
        macro_precision,
        macro_recall,
        macro_f1,
        _,
    ) = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=BEHAVIOURAL_CLASSES,
        average="macro",
        zero_division=0,
    )

    (
        weighted_precision,
        weighted_recall,
        weighted_f1,
        _,
    ) = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=BEHAVIOURAL_CLASSES,
        average="weighted",
        zero_division=0,
    )

    report_text = classification_report(
        y_true,
        y_pred,
        labels=BEHAVIOURAL_CLASSES,
        zero_division=0,
    )

    report_dict = classification_report(
        y_true,
        y_pred,
        labels=BEHAVIOURAL_CLASSES,
        zero_division=0,
        output_dict=True,
    )

    matrix = confusion_matrix(
        y_true,
        y_pred,
        labels=BEHAVIOURAL_CLASSES,
    )

    per_class = {}

    for label in BEHAVIOURAL_CLASSES:
        class_metrics = report_dict.get(
            label,
            {},
        )

        per_class[
            label
        ] = {
            "precision": float(
                class_metrics.get(
                    "precision",
                    0.0,
                )
            ),

            "recall": float(
                class_metrics.get(
                    "recall",
                    0.0,
                )
            ),

            "f1": float(
                class_metrics.get(
                    "f1-score",
                    0.0,
                )
            ),

            "support": int(
                class_metrics.get(
                    "support",
                    0,
                )
            ),
        }

    return {
        "accuracy": accuracy,

        "macro_precision": float(
            macro_precision
        ),

        "macro_recall": float(
            macro_recall
        ),

        "macro_f1": float(
            macro_f1
        ),

        "weighted_precision": float(
            weighted_precision
        ),

        "weighted_recall": float(
            weighted_recall
        ),

        "weighted_f1": float(
            weighted_f1
        ),

        "classification_report_text": (
            report_text
        ),

        "classification_report": (
            report_dict
        ),

        "per_class": (
            per_class
        ),

        "confusion_matrix": (
            matrix.tolist()
        ),
    }


# ============================================================================
# EVALUATE BASELINE + BALANCED MODELS
# ============================================================================

def evaluate_candidate_models(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> tuple[
    str,
    Pipeline,
    dict[str, dict[str, Any]],
]:
    """
    Compare:

        baseline_random_forest
        balanced_random_forest

    Selection:
        1. highest macro F1
        2. highest macro recall
        3. highest accuracy
    """

    candidates = {
        "baseline_random_forest": (
            build_baseline_model()
        ),

        "balanced_random_forest": (
            build_balanced_model()
        ),
    }

    results: dict[
        str,
        dict[str, Any],
    ] = {}

    fitted_models: dict[
        str,
        Pipeline,
    ] = {}

    for name, model in candidates.items():
        print()
        print(
            f"Training candidate: {name}"
        )

        start = time.perf_counter()

        model.fit(
            X_train,
            y_train,
        )

        runtime = (
            time.perf_counter()
            - start
        )

        predictions = model.predict(
            X_test
        )

        metrics = compute_classification_metrics(
            y_test,
            predictions,
        )

        metrics[
            "runtime_seconds"
        ] = float(
            runtime
        )

        results[
            name
        ] = metrics

        fitted_models[
            name
        ] = model

        print(
            f"  Accuracy     : "
            f"{metrics['accuracy']:.4f}"
        )

        print(
            f"  Macro F1     : "
            f"{metrics['macro_f1']:.4f}"
        )

        print(
            f"  Macro Recall : "
            f"{metrics['macro_recall']:.4f}"
        )

        print(
            f"  Weighted F1  : "
            f"{metrics['weighted_f1']:.4f}"
        )

    ranking = sorted(
        results.keys(),
        key=lambda name: (
            results[
                name
            ][
                "macro_f1"
            ],
            results[
                name
            ][
                "macro_recall"
            ],
            results[
                name
            ][
                "accuracy"
            ],
        ),
        reverse=True,
    )

    selected_name = (
        ranking[0]
    )

    selected_model = (
        fitted_models[
            selected_name
        ]
    )

    return (
        selected_name,
        selected_model,
        results,
    )


# ============================================================================
# SAVE MODEL COMPARISON
# ============================================================================

def save_model_comparison(
    results: dict[
        str,
        dict[str, Any],
    ],
) -> None:
    rows = []

    for name, metrics in results.items():
        rows.append(
            {
                "model": name,

                "accuracy": (
                    metrics[
                        "accuracy"
                    ]
                ),

                "macro_precision": (
                    metrics[
                        "macro_precision"
                    ]
                ),

                "macro_recall": (
                    metrics[
                        "macro_recall"
                    ]
                ),

                "macro_f1": (
                    metrics[
                        "macro_f1"
                    ]
                ),

                "weighted_precision": (
                    metrics[
                        "weighted_precision"
                    ]
                ),

                "weighted_recall": (
                    metrics[
                        "weighted_recall"
                    ]
                ),

                "weighted_f1": (
                    metrics[
                        "weighted_f1"
                    ]
                ),

                "runtime_seconds": (
                    metrics[
                        "runtime_seconds"
                    ]
                ),
            }
        )

    comparison_df = pd.DataFrame(
        rows
    )

    comparison_df = (
        comparison_df
        .sort_values(
            by=[
                "macro_f1",
                "macro_recall",
                "accuracy",
            ],
            ascending=False,
        )
    )

    comparison_df.to_csv(
        MODEL_COMPARISON_PATH,
        index=False,
    )


# ============================================================================
# TRAINING PIPELINE
# ============================================================================

def train_emosurv_model() -> tuple[
    Pipeline,
    dict[str, Any],
]:
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # -----------------------------------------------------------------------
    # Load dataset
    # -----------------------------------------------------------------------

    raw_df = load_emosurv_datasets()

    # -----------------------------------------------------------------------
    # Build windows
    # -----------------------------------------------------------------------

    windows = build_window_dataset(
        raw_df
    )

    # -----------------------------------------------------------------------
    # Proxy labels
    # -----------------------------------------------------------------------

    (
        dataset,
        distraction_threshold,
    ) = assign_behaviour_proxy_labels(
        windows
    )

    # -----------------------------------------------------------------------
    # Numeric sanitation
    # -----------------------------------------------------------------------

    for feature in FEATURE_COLUMNS:
        dataset[
            feature
        ] = pd.to_numeric(
            dataset[
                feature
            ],
            errors="coerce",
        )

        values = dataset[
            feature
        ].to_numpy(
            dtype=float
        )

        bad = ~np.isfinite(
            values
        )

        if bad.any():
            dataset.loc[
                bad,
                feature,
            ] = np.nan

    # -----------------------------------------------------------------------
    # Validate classes
    # -----------------------------------------------------------------------

    class_distribution = (
        dataset[
            "behaviour_proxy"
        ]
        .value_counts()
        .reindex(
            BEHAVIOURAL_CLASSES,
            fill_value=0,
        )
    )

    missing_classes = [
        label
        for label in BEHAVIOURAL_CLASSES
        if class_distribution[
            label
        ] == 0
    ]

    if missing_classes:
        raise ValueError(
            "Proxy dataset is missing behavioural classes: "
            f"{missing_classes}"
        )

    # -----------------------------------------------------------------------
    # Save complete proxy dataset
    # -----------------------------------------------------------------------

    dataset.to_csv(
        WINDOW_DATASET_PATH,
        index=False,
    )

    # -----------------------------------------------------------------------
    # Participant-independent split
    # -----------------------------------------------------------------------

    (
        train_index,
        test_index,
    ) = find_group_aware_split(
        dataset
    )

    train_df = (
        dataset.iloc[
            train_index
        ]
        .copy()
    )

    test_df = (
        dataset.iloc[
            test_index
        ]
        .copy()
    )

    train_users = set(
        train_df[
            "user_id"
        ]
    )

    test_users = set(
        test_df[
            "user_id"
        ]
    )

    participant_overlap = (
        train_users
        & test_users
    )

    if participant_overlap:
        raise RuntimeError(
            "Participant leakage detected."
        )

    X_train = train_df[
        FEATURE_COLUMNS
    ]

    y_train = train_df[
        "behaviour_proxy"
    ]

    X_test = test_df[
        FEATURE_COLUMNS
    ]

    y_test = test_df[
        "behaviour_proxy"
    ]

    print()
    print("=" * 88)
    print("PARTICIPANT-INDEPENDENT TRAIN / TEST SPLIT")
    print("=" * 88)

    print(
        f"Training windows      : "
        f"{len(train_df):,}"
    )

    print(
        f"Testing windows       : "
        f"{len(test_df):,}"
    )

    print(
        f"Training participants : "
        f"{train_df['user_id'].nunique()}"
    )

    print(
        f"Testing participants  : "
        f"{test_df['user_id'].nunique()}"
    )

    print(
        f"Participant overlap   : "
        f"{len(participant_overlap)}"
    )

    print()
    print(
        "Training class distribution:"
    )

    print(
        y_train
        .value_counts()
        .reindex(
            BEHAVIOURAL_CLASSES,
            fill_value=0,
        )
    )

    print()
    print(
        "Testing class distribution:"
    )

    print(
        y_test
        .value_counts()
        .reindex(
            BEHAVIOURAL_CLASSES,
            fill_value=0,
        )
    )

    # -----------------------------------------------------------------------
    # Compare baseline and balanced classifiers
    # -----------------------------------------------------------------------

    print()
    print("=" * 88)
    print("MODEL COMPARISON")
    print("=" * 88)

    (
        selected_name,
        selected_evaluation_model,
        candidate_results,
    ) = evaluate_candidate_models(
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
    )

    save_model_comparison(
        candidate_results
    )

    selected_metrics = (
        candidate_results[
            selected_name
        ]
    )

    print()
    print("=" * 88)
    print("SELECTED EMOSURV COMPARISON MODEL")
    print("=" * 88)

    print(
        f"Selected classifier : "
        f"{selected_name}"
    )

    print(
        f"Accuracy            : "
        f"{selected_metrics['accuracy']:.4f}"
    )

    print(
        f"Macro Precision     : "
        f"{selected_metrics['macro_precision']:.4f}"
    )

    print(
        f"Macro Recall        : "
        f"{selected_metrics['macro_recall']:.4f}"
    )

    print(
        f"Macro F1            : "
        f"{selected_metrics['macro_f1']:.4f}"
    )

    print(
        f"Weighted F1         : "
        f"{selected_metrics['weighted_f1']:.4f}"
    )

    print()
    print(
        "Per-class held-out metrics:"
    )

    for label in BEHAVIOURAL_CLASSES:
        class_metrics = (
            selected_metrics[
                "per_class"
            ][
                label
            ]
        )

        print(
            f"  {label:<12} "
            f"precision={class_metrics['precision']:.4f} "
            f"recall={class_metrics['recall']:.4f} "
            f"f1={class_metrics['f1']:.4f} "
            f"support={class_metrics['support']}"
        )

    # -----------------------------------------------------------------------
    # Save selected classification report
    # -----------------------------------------------------------------------

    report_df = pd.DataFrame(
        selected_metrics[
            "classification_report"
        ]
    ).transpose()

    report_df.to_csv(
        CLASSIFICATION_REPORT_PATH
    )

    confusion_df = pd.DataFrame(
        selected_metrics[
            "confusion_matrix"
        ],
        index=BEHAVIOURAL_CLASSES,
        columns=BEHAVIOURAL_CLASSES,
    )

    confusion_df.to_csv(
        CONFUSION_MATRIX_PATH
    )

    # -----------------------------------------------------------------------
    # Training feature distribution
    #
    # Use the complete proxy dataset because the final deployed model below
    # will also be trained on all available proxy windows.
    # -----------------------------------------------------------------------

    feature_distribution = (
        calculate_feature_distribution(
            dataset
        )
    )

    FEATURE_DISTRIBUTION_PATH.write_text(
        json.dumps(
            feature_distribution,
            indent=4,
        ),
        encoding="utf-8",
    )

    # -----------------------------------------------------------------------
    # Final model training
    #
    # Rebuild model configuration selected by held-out comparison,
    # then train it on the complete proxy dataset.
    # -----------------------------------------------------------------------

    if (
        selected_name
        == "balanced_random_forest"
    ):
        final_model = (
            build_balanced_model()
        )

    else:
        final_model = (
            build_baseline_model()
        )

    print()
    print("=" * 88)
    print("TRAINING FINAL EMOSURV COMPARISON MODEL")
    print("=" * 88)

    final_start = (
        time.perf_counter()
    )

    final_model.fit(
        dataset[
            FEATURE_COLUMNS
        ],
        dataset[
            "behaviour_proxy"
        ],
    )

    final_runtime = (
        time.perf_counter()
        - final_start
    )

    joblib.dump(
        final_model,
        MODEL_PATH,
    )

    FEATURE_COLUMNS_PATH.write_text(
        json.dumps(
            FEATURE_COLUMNS,
            indent=4,
        ),
        encoding="utf-8",
    )

    # -----------------------------------------------------------------------
    # Metadata
    # -----------------------------------------------------------------------

    metadata = {
        "created_at": (
            datetime.now().isoformat(
                timespec="seconds"
            )
        ),

        "project": (
            "SenseFuzeAI"
        ),

        "model_type": (
            "EmoSurv IEEE keystroke behavioural-proxy comparison model"
        ),

        "source_datasets": [
            str(
                FIXED_TEXT_PATH
            ),
            str(
                FREE_TEXT_PATH
            ),
        ],

        "datasets_intentionally_excluded": [
            "Frequency Dataset.csv",
            "Participants Information.csv",
        ],

        "original_emotion_codes": (
            EMOTION_CODES
        ),

        "behavioural_classes": (
            BEHAVIOURAL_CLASSES
        ),

        "methodological_status": (
            "weakly_supervised_behavioural_proxy_labels"
        ),

        "methodological_warning": (
            "EmoSurv does not provide SenseFuzeAI behavioural "
            "ground-truth labels. Behavioural states are derived "
            "as experimental proxy labels from original emotion "
            "labels plus typing-dynamics irregularity."
        ),

        "proxy_mapping": {
            "Calm": "focused",
            "Angry": "overloaded",
            "Sad": "fatigued",

            "Neutral_Happy": (
                "focused or distracted according to "
                "typing irregularity"
            ),
        },

        "distraction_index_weights": {
            "rhythm_cv": 0.35,
            "pause_ratio_1000": 0.30,
            "correction_ratio": 0.20,
            "dd_std_ms": 0.15,
        },

        "distraction_threshold_quantile": (
            0.60
        ),

        "distraction_threshold": (
            distraction_threshold
        ),

        "window_size": (
            WINDOW_SIZE
        ),

        "window_step": (
            WINDOW_STEP
        ),

        "minimum_window_size": (
            MIN_WINDOW_SIZE
        ),

        "num_proxy_windows": int(
            len(dataset)
        ),

        "num_participants": int(
            dataset[
                "user_id"
            ]
            .nunique()
        ),

        "feature_columns": (
            FEATURE_COLUMNS
        ),

        "num_features": int(
            len(FEATURE_COLUMNS)
        ),

        "class_distribution": (
            class_distribution
            .to_dict()
        ),

        "evaluation_method": (
            "participant-independent GroupShuffleSplit"
        ),

        "model_comparison": (
            "baseline Random Forest versus "
            "balanced_subsample Random Forest"
        ),

        "selection_rule": (
            "highest held-out macro F1, "
            "then macro recall, then accuracy"
        ),

        "selected_classifier": (
            selected_name
        ),

        "candidate_metrics": {
            candidate_name: {
                "accuracy": (
                    metrics[
                        "accuracy"
                    ]
                ),

                "macro_precision": (
                    metrics[
                        "macro_precision"
                    ]
                ),

                "macro_recall": (
                    metrics[
                        "macro_recall"
                    ]
                ),

                "macro_f1": (
                    metrics[
                        "macro_f1"
                    ]
                ),

                "weighted_precision": (
                    metrics[
                        "weighted_precision"
                    ]
                ),

                "weighted_recall": (
                    metrics[
                        "weighted_recall"
                    ]
                ),

                "weighted_f1": (
                    metrics[
                        "weighted_f1"
                    ]
                ),

                "per_class": (
                    metrics[
                        "per_class"
                    ]
                ),
            }

            for (
                candidate_name,
                metrics,
            ) in candidate_results.items()
        },

        "test_size": (
            TEST_SIZE
        ),

        "random_state": (
            RANDOM_STATE
        ),

        "train_windows": int(
            len(train_df)
        ),

        "test_windows": int(
            len(test_df)
        ),

        "train_participants": int(
            train_df[
                "user_id"
            ]
            .nunique()
        ),

        "test_participants": int(
            test_df[
                "user_id"
            ]
            .nunique()
        ),

        "participant_overlap": int(
            len(
                participant_overlap
            )
        ),

        "held_out_accuracy": (
            selected_metrics[
                "accuracy"
            ]
        ),

        "held_out_macro_precision": (
            selected_metrics[
                "macro_precision"
            ]
        ),

        "held_out_macro_recall": (
            selected_metrics[
                "macro_recall"
            ]
        ),

        "held_out_macro_f1": (
            selected_metrics[
                "macro_f1"
            ]
        ),

        "held_out_weighted_precision": (
            selected_metrics[
                "weighted_precision"
            ]
        ),

        "held_out_weighted_recall": (
            selected_metrics[
                "weighted_recall"
            ]
        ),

        "held_out_weighted_f1": (
            selected_metrics[
                "weighted_f1"
            ]
        ),

        "held_out_per_class": (
            selected_metrics[
                "per_class"
            ]
        ),

        "feature_distribution_path": (
            str(
                FEATURE_DISTRIBUTION_PATH
            )
        ),

        "live_distribution_check": {
            "robust_z_threshold": (
                LIVE_OOD_Z_THRESHOLD
            ),

            "warning_feature_ratio": (
                LIVE_OOD_FEATURE_RATIO_WARNING
            ),
        },

        "final_training_runtime_seconds": (
            float(
                final_runtime
            )
        ),

        "model_path": (
            str(
                MODEL_PATH
            )
        ),

        "feature_columns_path": (
            str(
                FEATURE_COLUMNS_PATH
            )
        ),

        "dataset_path": (
            str(
                WINDOW_DATASET_PATH
            )
        ),

        "model_comparison_path": (
            str(
                MODEL_COMPARISON_PATH
            )
        ),

        "evaluation_report_path": (
            str(
                EVALUATION_REPORT_PATH
            )
        ),

        "classification_report_path": (
            str(
                CLASSIFICATION_REPORT_PATH
            )
        ),

        "confusion_matrix_path": (
            str(
                CONFUSION_MATRIX_PATH
            )
        ),
    }

    METADATA_PATH.write_text(
        json.dumps(
            metadata,
            indent=4,
        ),
        encoding="utf-8",
    )

    # -----------------------------------------------------------------------
    # Human-readable report
    # -----------------------------------------------------------------------

    report_lines = [
        "SenseFuzeAI EmoSurv IEEE Keystroke Comparison Report",
        "=" * 58,
        "",

        (
            "Created: "
            + datetime.now().isoformat(
                timespec="seconds"
            )
        ),

        "",
        "METHODOLOGICAL STATUS",
        "--------------------",

        (
            "The EmoSurv datasets contain emotion labels rather "
            "than SenseFuzeAI behavioural-state labels."
        ),

        (
            "Focused, distracted, fatigued and overloaded are "
            "therefore weakly supervised proxy labels."
        ),

        "",
        "Proxy mapping:",

        "  Calm   -> focused",
        "  Angry  -> overloaded",
        "  Sad    -> fatigued",

        (
            "  Neutral / Happy -> focused or distracted according "
            "to typing-dynamics irregularity"
        ),

        "",
        (
            f"Distraction threshold: "
            f"{distraction_threshold:.6f}"
        ),

        "",
        "DATASET",

        (
            f"Proxy windows       : "
            f"{len(dataset)}"
        ),

        (
            f"Participants        : "
            f"{dataset['user_id'].nunique()}"
        ),

        (
            f"Training windows    : "
            f"{len(train_df)}"
        ),

        (
            f"Testing windows     : "
            f"{len(test_df)}"
        ),

        (
            f"Training users      : "
            f"{train_df['user_id'].nunique()}"
        ),

        (
            f"Testing users       : "
            f"{test_df['user_id'].nunique()}"
        ),

        (
            f"Participant overlap : "
            f"{len(participant_overlap)}"
        ),

        "",
        "CLASS DISTRIBUTION",
        "------------------",
        str(
            class_distribution
        ),

        "",
        "MODEL COMPARISON",
        "----------------",
    ]

    for (
        candidate_name,
        metrics,
    ) in candidate_results.items():
        report_lines.extend(
            [
                "",
                candidate_name,

                (
                    f"  Accuracy     : "
                    f"{metrics['accuracy']:.4f}"
                ),

                (
                    f"  Macro P      : "
                    f"{metrics['macro_precision']:.4f}"
                ),

                (
                    f"  Macro Recall : "
                    f"{metrics['macro_recall']:.4f}"
                ),

                (
                    f"  Macro F1     : "
                    f"{metrics['macro_f1']:.4f}"
                ),

                (
                    f"  Weighted F1  : "
                    f"{metrics['weighted_f1']:.4f}"
                ),
            ]
        )

    report_lines.extend(
        [
            "",
            "SELECTED MODEL",
            "--------------",

            (
                f"Classifier      : "
                f"{selected_name}"
            ),

            (
                f"Accuracy        : "
                f"{selected_metrics['accuracy']:.4f}"
            ),

            (
                f"Macro Precision : "
                f"{selected_metrics['macro_precision']:.4f}"
            ),

            (
                f"Macro Recall    : "
                f"{selected_metrics['macro_recall']:.4f}"
            ),

            (
                f"Macro F1        : "
                f"{selected_metrics['macro_f1']:.4f}"
            ),

            (
                f"Weighted F1     : "
                f"{selected_metrics['weighted_f1']:.4f}"
            ),

            "",
            "PER-CLASS METRICS",
            "-----------------",
        ]
    )

    for label in BEHAVIOURAL_CLASSES:
        class_metrics = (
            selected_metrics[
                "per_class"
            ][
                label
            ]
        )

        report_lines.append(
            f"{label:<12} "
            f"P={class_metrics['precision']:.4f} "
            f"R={class_metrics['recall']:.4f} "
            f"F1={class_metrics['f1']:.4f} "
            f"N={class_metrics['support']}"
        )

    report_lines.extend(
        [
            "",
            "CLASSIFICATION REPORT",
            "---------------------",

            selected_metrics[
                "classification_report_text"
            ],

            "",
            "CONFUSION MATRIX",
            "----------------",

            str(
                selected_metrics[
                    "confusion_matrix"
                ]
            ),

            "",
            "LIVE-DISTRIBUTION DIAGNOSTICS",
            "-----------------------------",

            (
                "The live GUI compares each observed feature against "
                "the EmoSurv proxy training distribution using robust "
                "median/MAD z-scores."
            ),

            (
                "Large training-vs-live distribution differences are "
                "reported as diagnostic warnings and are not used to "
                "force a different behavioural prediction."
            ),
        ]
    )

    EVALUATION_REPORT_PATH.write_text(
        "\n".join(
            report_lines
        ),
        encoding="utf-8",
    )

    print()
    print("=" * 88)
    print("EMOSURV COMPARISON TRAINING COMPLETE")
    print("=" * 88)

    print(
        f"\nSelected model:\n  "
        f"{selected_name}"
    )

    print(
        f"\nSaved model:\n  "
        f"{MODEL_PATH}"
    )

    print(
        f"\nModel comparison:\n  "
        f"{MODEL_COMPARISON_PATH}"
    )

    print(
        f"\nClassification report:\n  "
        f"{CLASSIFICATION_REPORT_PATH}"
    )

    print(
        f"\nConfusion matrix:\n  "
        f"{CONFUSION_MATRIX_PATH}"
    )

    print(
        f"\nFeature distribution:\n  "
        f"{FEATURE_DISTRIBUTION_PATH}"
    )

    print(
        f"\nMetadata:\n  "
        f"{METADATA_PATH}"
    )

    return (
        final_model,
        metadata,
    )


# ============================================================================
# LOAD SAVED MODEL
# ============================================================================

def load_existing_model() -> tuple[
    Pipeline,
    dict[str, Any],
]:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Missing model:\n{MODEL_PATH}\n\n"
            "Run:\n"
            "python keystroke_live_gui_emosurv_ieee.py --retrain"
        )

    if not METADATA_PATH.exists():
        raise FileNotFoundError(
            f"Missing metadata:\n{METADATA_PATH}"
        )

    model = joblib.load(
        MODEL_PATH
    )

    metadata = json.loads(
        METADATA_PATH.read_text(
            encoding="utf-8"
        )
    )

    return (
        model,
        metadata,
    )


# ============================================================================
# LOAD TRAINING FEATURE DISTRIBUTION
# ============================================================================

def load_feature_distribution() -> dict[
    str,
    dict[str, float],
]:
    if not FEATURE_DISTRIBUTION_PATH.exists():
        return {}

    return json.loads(
        FEATURE_DISTRIBUTION_PATH.read_text(
            encoding="utf-8"
        )
    )


# ============================================================================
# LIVE FEATURE EXTRACTION
# ============================================================================

def extract_live_features(
    events: list[
        dict[str, Any]
    ],
) -> dict[str, float]:
    keydowns = [
        event
        for event in events
        if event.get(
            "type"
        ) == "down"
    ]

    if not keydowns:
        return {
            feature: 0.0
            for feature in FEATURE_COLUMNS
        }

    # -----------------------------------------------------------------------
    # Key-down timestamps
    # -----------------------------------------------------------------------

    down_times = [
        safe_float(
            event.get(
                "timestamp_perf",
                0.0,
            )
        )
        for event in keydowns
    ]

    # -----------------------------------------------------------------------
    # D1D2-style intervals
    # -----------------------------------------------------------------------

    dd_ms: list[
        float
    ] = []

    for index in range(
        len(
            down_times
        )
        - 1
    ):
        interval = (
            down_times[
                index + 1
            ]
            - down_times[
                index
            ]
        ) * 1000.0

        if (
            DD_MIN_MS
            <= interval
            <= DD_MAX_MS
        ):
            dd_ms.append(
                interval
            )

    # -----------------------------------------------------------------------
    # Dwell times
    # -----------------------------------------------------------------------

    active: dict[
        str,
        list[float],
    ] = {}

    completed: list[
        tuple[
            str,
            float,
            float,
        ]
    ] = []

    dwell_ms: list[
        float
    ] = []

    for event in events:
        key = str(
            event.get(
                "key",
                "",
            )
        ).lower()

        timestamp = safe_float(
            event.get(
                "timestamp_perf",
                0.0,
            )
        )

        event_type = event.get(
            "type"
        )

        if event_type == "down":
            active.setdefault(
                key,
                [],
            ).append(
                timestamp
            )

        elif event_type == "up":
            if (
                key in active
                and active[
                    key
                ]
            ):
                down_timestamp = (
                    active[
                        key
                    ]
                    .pop(0)
                )

                dwell = (
                    timestamp
                    - down_timestamp
                ) * 1000.0

                if (
                    DWELL_MIN_MS
                    <= dwell
                    <= DWELL_MAX_MS
                ):
                    dwell_ms.append(
                        dwell
                    )

                completed.append(
                    (
                        key,
                        down_timestamp,
                        timestamp,
                    )
                )

    # -----------------------------------------------------------------------
    # U1D2-style intervals
    # -----------------------------------------------------------------------

    completed.sort(
        key=lambda item: item[1]
    )

    ud_ms: list[
        float
    ] = []

    for index in range(
        len(
            completed
        )
        - 1
    ):
        current_up = (
            completed[
                index
            ][2]
        )

        next_down = (
            completed[
                index + 1
            ][1]
        )

        interval = (
            next_down
            - current_up
        ) * 1000.0

        if (
            UD_MIN_MS
            <= interval
            <= UD_MAX_MS
        ):
            ud_ms.append(
                interval
            )

    positive_dd = [
        value
        for value in dd_ms
        if value > 0
    ]

    mean_dd = safe_mean(
        positive_dd
    )

    dd_std = safe_std(
        positive_dd
    )

    typing_speed_kps = (
        1000.0 / mean_dd
        if mean_dd > 0
        else 0.0
    )

    def ratio_above(
        threshold: float,
    ) -> float:
        if not positive_dd:
            return 0.0

        return float(
            sum(
                value >= threshold
                for value
                in positive_dd
            )
            / len(
                positive_dd
            )
        )

    keys = [
        str(
            event.get(
                "key",
                "",
            )
        ).lower()

        for event
        in keydowns
    ]

    correction_ratio = (
        sum(
            key
            in {
                "backspace",
                "delete",
            }

            for key
            in keys
        )
        / len(
            keys
        )

        if keys
        else 0.0
    )

    space_ratio = (
        sum(
            key == "space"
            for key in keys
        )
        / len(
            keys
        )

        if keys
        else 0.0
    )

    rhythm_cv = (
        dd_std / mean_dd
        if mean_dd > 0
        else 0.0
    )

    overlap_ratio = (
        sum(
            value < 0
            for value in ud_ms
        )
        / len(
            ud_ms
        )

        if ud_ms
        else 0.0
    )

    result = {
        "dwell_mean_ms": safe_mean(
            dwell_ms
        ),

        "dwell_std_ms": safe_std(
            dwell_ms
        ),

        "dwell_median_ms": safe_median(
            dwell_ms
        ),

        "dwell_min_ms": safe_min(
            dwell_ms
        ),

        "dwell_max_ms": safe_max(
            dwell_ms
        ),

        "dd_mean_ms": mean_dd,

        "dd_std_ms": dd_std,

        "dd_median_ms": safe_median(
            positive_dd
        ),

        "dd_min_ms": safe_min(
            positive_dd
        ),

        "dd_max_ms": safe_max(
            positive_dd
        ),

        "ud_mean_ms": safe_mean(
            ud_ms
        ),

        "ud_std_ms": safe_std(
            ud_ms
        ),

        "ud_median_ms": safe_median(
            ud_ms
        ),

        "ud_min_ms": safe_min(
            ud_ms
        ),

        "ud_max_ms": safe_max(
            ud_ms
        ),

        "typing_speed_kps": (
            typing_speed_kps
        ),

        "pause_ratio_500": (
            ratio_above(
                500.0
            )
        ),

        "pause_ratio_1000": (
            ratio_above(
                1000.0
            )
        ),

        "pause_ratio_2000": (
            ratio_above(
                2000.0
            )
        ),

        "correction_ratio": (
            correction_ratio
        ),

        "space_ratio": (
            space_ratio
        ),

        "rhythm_cv": (
            rhythm_cv
        ),

        "overlap_ratio": (
            overlap_ratio
        ),
    }

    return {
        feature: safe_float(
            result.get(
                feature,
                0.0,
            )
        )

        for feature
        in FEATURE_COLUMNS
    }


# ============================================================================
# LIVE TRAINING-DISTRIBUTION DIAGNOSTICS
# ============================================================================

def compare_live_to_training_distribution(
    live_features: dict[
        str,
        float,
    ],
    distribution: dict[
        str,
        dict[
            str,
            float,
        ],
    ],
) -> dict[str, Any]:
    """
    Compare live values against EmoSurv training-distribution statistics.

    Robust z-score:
        (x - median) / (1.4826 * MAD)

    This is diagnostic only.

    It does NOT override the classifier.
    """

    feature_results: dict[
        str,
        dict[str, Any],
    ] = {}

    ood_features: list[
        str
    ] = []

    for feature in FEATURE_COLUMNS:
        live_value = safe_float(
            live_features.get(
                feature,
                0.0,
            )
        )

        stats = distribution.get(
            feature,
            {},
        )

        median = safe_float(
            stats.get(
                "median",
                0.0,
            )
        )

        mad = safe_float(
            stats.get(
                "mad",
                0.0,
            )
        )

        q05 = safe_float(
            stats.get(
                "q05",
                0.0,
            )
        )

        q95 = safe_float(
            stats.get(
                "q95",
                0.0,
            )
        )

        if mad > 1e-12:
            robust_z = (
                live_value
                - median
            ) / (
                1.4826
                * mad
            )

        else:
            robust_z = 0.0

        strongly_ood = (
            abs(
                robust_z
            )
            >= LIVE_OOD_Z_THRESHOLD
        )

        outside_central_90 = (
            live_value < q05
            or live_value > q95
        )

        if strongly_ood:
            ood_features.append(
                feature
            )

        feature_results[
            feature
        ] = {
            "live_value": (
                live_value
            ),

            "training_median": (
                median
            ),

            "training_q05": (
                q05
            ),

            "training_q95": (
                q95
            ),

            "robust_z": float(
                robust_z
            ),

            "outside_training_central_90": (
                outside_central_90
            ),

            "strongly_out_of_distribution": (
                strongly_ood
            ),
        }

    ood_ratio = (
        len(
            ood_features
        )
        / len(
            FEATURE_COLUMNS
        )
    )

    warning = (
        ood_ratio
        >= LIVE_OOD_FEATURE_RATIO_WARNING
    )

    return {
        "features": (
            feature_results
        ),

        "strongly_ood_features": (
            ood_features
        ),

        "strongly_ood_count": int(
            len(
                ood_features
            )
        ),

        "feature_count": int(
            len(
                FEATURE_COLUMNS
            )
        ),

        "strongly_ood_ratio": float(
            ood_ratio
        ),

        "distribution_warning": (
            warning
        ),
    }


# ============================================================================
# LIVE PREDICTION
# ============================================================================

def predict_live_behaviour(
    model: Pipeline,
    events: list[
        dict[str, Any]
    ],
    feature_distribution: dict[
        str,
        dict[
            str,
            float,
        ],
    ],
) -> dict[str, Any]:
    features = extract_live_features(
        events
    )

    X = pd.DataFrame(
        [
            {
                feature: features[
                    feature
                ]
                for feature in FEATURE_COLUMNS
            }
        ],
        columns=FEATURE_COLUMNS,
    )

    prediction = str(
        model.predict(
            X
        )[0]
    )

    probabilities = (
        model.predict_proba(
            X
        )[0]
    )

    classes = [
        str(
            class_name
        )
        for class_name
        in model.classes_
    ]

    probability_dict = {
        class_name: float(
            probability
        )

        for (
            class_name,
            probability,
        )
        in zip(
            classes,
            probabilities,
        )
    }

    ranked = sorted(
        probability_dict.items(),
        key=lambda item: item[1],
        reverse=True,
    )

    top_class, top_probability = (
        ranked[0]
    )

    second_class, second_probability = (
        ranked[1]
    )

    confidence_gap = (
        top_probability
        - second_probability
    )

    distribution_check = (
        compare_live_to_training_distribution(
            live_features=features,
            distribution=feature_distribution,
        )
    )

    return {
        "prediction": (
            prediction
        ),

        "current_state": (
            top_class
        ),

        "confidence": float(
            top_probability
        ),

        "confidence_percent": float(
            top_probability
            * 100.0
        ),

        "confidence_gap": float(
            confidence_gap
        ),

        "confidence_level": (
            confidence_level(
                confidence_gap
            )
        ),

        "second_class": (
            second_class
        ),

        "second_probability": float(
            second_probability
        ),

        "probabilities": (
            probability_dict
        ),

        "features": (
            features
        ),

        "distribution_check": (
            distribution_check
        ),
    }


# ============================================================================
# LIVE LOG
# ============================================================================

def initialise_live_log() -> None:
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    if LIVE_LOG_PATH.exists():
        return

    with LIVE_LOG_PATH.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.writer(
            file
        )

        writer.writerow(
            [
                "timestamp",
                "keydown_count",
                "text_length",
                "current_state",
                "confidence",
                "confidence_level",
                "second_class",
                "second_probability",
                "confidence_gap",
                "distribution_warning",
                "ood_feature_count",
                "ood_feature_ratio",
                "probabilities_json",
                "features_json",
            ]
        )


# ============================================================================
# TKINTER GUI
# ============================================================================

class EmoSurvKeystrokeApp:
    def __init__(
        self,
        root: tk.Tk,
        model: Pipeline,
        metadata: dict[str, Any],
    ) -> None:
        self.root = root
        self.model = model
        self.metadata = metadata

        self.feature_distribution = (
            load_feature_distribution()
        )

        self.events: list[
            dict[str, Any]
        ] = []

        self.active_keys: set[
            str
        ] = set()

        self.root.title(
            "SenseFuzeAI - EmoSurv IEEE Keystroke Comparison"
        )

        self.root.geometry(
            "1180x900"
        )

        self.root.minsize(
            960,
            760,
        )

        self.root.configure(
            bg="#07111f"
        )

        initialise_live_log()

        self.build_ui()

    # ======================================================================
    # GUI
    # ======================================================================

    def build_ui(
        self,
    ) -> None:
        tk.Label(
            self.root,

            text=(
                "SenseFuzeAI EmoSurv IEEE "
                "Keystroke Comparison"
            ),

            font=(
                "Arial",
                21,
                "bold",
            ),

            fg="#74f7ff",
            bg="#07111f",

        ).pack(
            pady=(
                12,
                4,
            )
        )

        tk.Label(
            self.root,

            text=(
                "External Fixed-Text + Free-Text "
                "Keystroke Dataset"
            ),

            font=(
                "Arial",
                11,
            ),

            fg="white",
            bg="#07111f",

        ).pack()

        tk.Label(
            self.root,

            text=(
                "Important: displayed behavioural states are "
                "weakly supervised EmoSurv proxy labels, "
                "not original behavioural ground truth."
            ),

            font=(
                "Arial",
                10,
                "bold",
            ),

            fg="#ffd166",
            bg="#07111f",

        ).pack(
            pady=(
                3,
                10,
            )
        )

        # ------------------------------------------------------------------
        # Status
        # ------------------------------------------------------------------

        status_frame = tk.Frame(
            self.root,
            bg="#10203a",
            padx=14,
            pady=10,
        )

        status_frame.pack(
            fill="x",
            padx=20,
            pady=6,
        )

        selected_classifier = (
            self.metadata.get(
                "selected_classifier",
                "unknown",
            )
        )

        macro_f1 = float(
            self.metadata.get(
                "held_out_macro_f1",
                0.0,
            )
        )

        macro_recall = float(
            self.metadata.get(
                "held_out_macro_recall",
                0.0,
            )
        )

        tk.Label(
            status_frame,

            text=(
                f"Model: "
                f"{selected_classifier}"
            ),

            font=(
                "Arial",
                10,
                "bold",
            ),

            fg="#66ffd6",
            bg="#10203a",

        ).grid(
            row=0,
            column=0,
            padx=10,
            sticky="w",
        )

        tk.Label(
            status_frame,

            text=(
                f"Held-out Macro F1: "
                f"{macro_f1:.3f}"
            ),

            font=(
                "Arial",
                10,
                "bold",
            ),

            fg="#74f7ff",
            bg="#10203a",

        ).grid(
            row=0,
            column=1,
            padx=10,
        )

        tk.Label(
            status_frame,

            text=(
                f"Held-out Macro Recall: "
                f"{macro_recall:.3f}"
            ),

            font=(
                "Arial",
                10,
                "bold",
            ),

            fg="#74f7ff",
            bg="#10203a",

        ).grid(
            row=0,
            column=2,
            padx=10,
        )

        self.distribution_status_label = tk.Label(
            status_frame,

            text=(
                "Live Distribution: "
                "waiting for input"
            ),

            font=(
                "Arial",
                10,
                "bold",
            ),

            fg="#cbd6ff",
            bg="#10203a",

        )

        self.distribution_status_label.grid(
            row=0,
            column=3,
            padx=10,
            sticky="e",
        )

        # ------------------------------------------------------------------
        # Typing input
        # ------------------------------------------------------------------

        input_frame = tk.LabelFrame(
            self.root,

            text=(
                "Live Keystroke Input"
            ),

            font=(
                "Arial",
                12,
                "bold",
            ),

            fg="#74f7ff",
            bg="#07111f",

            padx=12,
            pady=10,
        )

        input_frame.pack(
            fill="both",
            padx=20,
            pady=8,
        )

        self.text_widget = tk.Text(
            input_frame,

            height=8,

            font=(
                "Arial",
                12,
            ),

            bg="#0b1220",
            fg="white",

            insertbackground="white",

            wrap="word",
        )

        self.text_widget.pack(
            fill="both",
            expand=True,
        )

        self.text_widget.bind(
            "<KeyPress>",
            self.on_key_down,
        )

        self.text_widget.bind(
            "<KeyRelease>",
            self.on_key_up,
        )

        # ------------------------------------------------------------------
        # Live metrics
        # ------------------------------------------------------------------

        metrics_frame = tk.Frame(
            self.root,
            bg="#07111f",
        )

        metrics_frame.pack(
            pady=6,
        )

        self.key_count_label = tk.Label(
            metrics_frame,

            text="Keypresses: 0",

            width=18,

            font=(
                "Arial",
                10,
                "bold",
            ),

            fg="#cbd6ff",
            bg="#07111f",
        )

        self.key_count_label.grid(
            row=0,
            column=0,
            padx=5,
        )

        self.speed_label = tk.Label(
            metrics_frame,

            text=(
                "Speed: "
                "0.00 keys/sec"
            ),

            width=24,

            font=(
                "Arial",
                10,
                "bold",
            ),

            fg="#cbd6ff",
            bg="#07111f",
        )

        self.speed_label.grid(
            row=0,
            column=1,
            padx=5,
        )

        self.pause_label = tk.Label(
            metrics_frame,

            text=(
                "Pause ratio ≥1s: "
                "0.000"
            ),

            width=24,

            font=(
                "Arial",
                10,
                "bold",
            ),

            fg="#cbd6ff",
            bg="#07111f",
        )

        self.pause_label.grid(
            row=0,
            column=2,
            padx=5,
        )

        self.rhythm_label = tk.Label(
            metrics_frame,

            text=(
                "Rhythm CV: "
                "0.000"
            ),

            width=20,

            font=(
                "Arial",
                10,
                "bold",
            ),

            fg="#cbd6ff",
            bg="#07111f",
        )

        self.rhythm_label.grid(
            row=0,
            column=3,
            padx=5,
        )

        self.correction_label = tk.Label(
            metrics_frame,

            text=(
                "Correction ratio: "
                "0.000"
            ),

            width=24,

            font=(
                "Arial",
                10,
                "bold",
            ),

            fg="#cbd6ff",
            bg="#07111f",
        )

        self.correction_label.grid(
            row=0,
            column=4,
            padx=5,
        )

        # ------------------------------------------------------------------
        # Buttons
        # ------------------------------------------------------------------

        button_frame = tk.Frame(
            self.root,
            bg="#07111f",
        )

        button_frame.pack(
            pady=8,
        )

        tk.Button(
            button_frame,

            text="Run Prediction",

            command=(
                self.run_prediction
            ),

            width=22,

            font=(
                "Arial",
                11,
                "bold",
            ),

            bg="#2E86C1",
            fg="white",

        ).grid(
            row=0,
            column=0,
            padx=8,
        )

        tk.Button(
            button_frame,

            text="Reset Session",

            command=(
                self.reset
            ),

            width=18,

            font=(
                "Arial",
                11,
                "bold",
            ),

            bg="#4a5568",
            fg="white",

        ).grid(
            row=0,
            column=1,
            padx=8,
        )

        # ------------------------------------------------------------------
        # Prediction
        # ------------------------------------------------------------------

        result_frame = tk.Frame(
            self.root,

            bg="#10203a",

            padx=20,
            pady=14,
        )

        result_frame.pack(
            fill="x",
            padx=20,
            pady=8,
        )

        tk.Label(
            result_frame,

            text=(
                "Current Behavioural Proxy State"
            ),

            font=(
                "Arial",
                13,
                "bold",
            ),

            fg="#cbd6ff",
            bg="#10203a",

        ).pack()

        self.state_label = tk.Label(
            result_frame,

            text="—",

            font=(
                "Arial",
                36,
                "bold",
            ),

            fg="#74f7ff",
            bg="#10203a",
        )

        self.state_label.pack(
            pady=4,
        )

        self.confidence_label = tk.Label(
            result_frame,

            text="Confidence: —",

            font=(
                "Arial",
                17,
                "bold",
            ),

            fg="white",
            bg="#10203a",
        )

        self.confidence_label.pack(
            pady=2,
        )

        self.confidence_level_label = tk.Label(
            result_frame,

            text=(
                "Prediction Confidence: —"
            ),

            font=(
                "Arial",
                13,
                "bold",
            ),

            fg="#cbd6ff",
            bg="#10203a",
        )

        self.confidence_level_label.pack(
            pady=2,
        )

        # ------------------------------------------------------------------
        # Technical detail
        # ------------------------------------------------------------------

        technical_frame = tk.LabelFrame(
            self.root,

            text=(
                "Technical + Training-vs-Live Diagnostics"
            ),

            font=(
                "Arial",
                11,
                "bold",
            ),

            fg="#74f7ff",
            bg="#07111f",

            padx=10,
            pady=8,
        )

        technical_frame.pack(
            fill="both",
            expand=True,
            padx=20,
            pady=8,
        )

        self.details_text = tk.Text(
            technical_frame,

            height=18,

            font=(
                "Consolas",
                9,
            ),

            bg="#0b1220",
            fg="#dbeafe",

            wrap="none",
        )

        self.details_text.pack(
            fill="both",
            expand=True,
        )

    # ======================================================================
    # Tk key normalisation
    # ======================================================================

    @staticmethod
    def normalise_tk_key(
        event: tk.Event,
    ) -> str:
        keysym = str(
            event.keysym
        ).lower()

        if keysym == "backspace":
            return "backspace"

        if keysym == "delete":
            return "delete"

        if keysym == "space":
            return "space"

        character = getattr(
            event,
            "char",
            "",
        )

        if character:
            return str(
                character
            ).lower()

        return keysym

    # ======================================================================
    # Event handlers
    # ======================================================================

    def on_key_down(
        self,
        event: tk.Event,
    ) -> None:
        key = self.normalise_tk_key(
            event
        )

        if key in self.active_keys:
            return

        self.active_keys.add(
            key
        )

        self.events.append(
            {
                "type": "down",

                "key": (
                    key
                ),

                "timestamp_perf": (
                    time.perf_counter()
                ),

                "timestamp_epoch": (
                    time.time()
                ),
            }
        )

        self.update_live_metrics()

    def on_key_up(
        self,
        event: tk.Event,
    ) -> None:
        key = self.normalise_tk_key(
            event
        )

        self.active_keys.discard(
            key
        )

        self.events.append(
            {
                "type": "up",

                "key": (
                    key
                ),

                "timestamp_perf": (
                    time.perf_counter()
                ),

                "timestamp_epoch": (
                    time.time()
                ),
            }
        )

        self.update_live_metrics()

    # ======================================================================
    # Live metric display
    # ======================================================================

    def update_live_metrics(
        self,
    ) -> None:
        keydowns = sum(
            event.get(
                "type"
            ) == "down"

            for event
            in self.events
        )

        features = extract_live_features(
            self.events
        )

        self.key_count_label.config(
            text=(
                f"Keypresses: "
                f"{keydowns}"
            )
        )

        self.speed_label.config(
            text=(
                f"Speed: "
                f"{features['typing_speed_kps']:.2f} "
                "keys/sec"
            )
        )

        self.pause_label.config(
            text=(
                f"Pause ratio ≥1s: "
                f"{features['pause_ratio_1000']:.3f}"
            )
        )

        self.rhythm_label.config(
            text=(
                f"Rhythm CV: "
                f"{features['rhythm_cv']:.3f}"
            )
        )

        self.correction_label.config(
            text=(
                f"Correction ratio: "
                f"{features['correction_ratio']:.3f}"
            )
        )

    # ======================================================================
    # Prediction
    # ======================================================================

    def run_prediction(
        self,
    ) -> None:
        try:
            keydown_count = sum(
                event.get(
                    "type"
                ) == "down"

                for event
                in self.events
            )

            if (
                keydown_count
                < MIN_LIVE_KEYPRESSES
            ):
                raise ValueError(
                    f"At least {MIN_LIVE_KEYPRESSES} "
                    "keypresses are required."
                )

            result = predict_live_behaviour(
                model=self.model,
                events=self.events,
                feature_distribution=(
                    self.feature_distribution
                ),
            )

            current_state = (
                result[
                    "current_state"
                ]
            )

            confidence = float(
                result[
                    "confidence"
                ]
            )

            level = (
                result[
                    "confidence_level"
                ]
            )

            distribution_result = (
                result[
                    "distribution_check"
                ]
            )

            distribution_warning = bool(
                distribution_result[
                    "distribution_warning"
                ]
            )

            # ----------------------------------------------------------------
            # Prediction UI
            # ----------------------------------------------------------------

            self.state_label.config(
                text=(
                    current_state.upper()
                )
            )

            self.confidence_label.config(
                text=(
                    f"Confidence: "
                    f"{confidence * 100:.2f}%"
                )
            )

            confidence_colour = {
                "High": "#66ffd6",
                "Medium": "#ffd166",
                "Low": "#ff6b8a",
            }.get(
                level,
                "#cbd6ff",
            )

            self.confidence_level_label.config(
                text=(
                    f"Prediction Confidence: "
                    f"{level}"
                ),
                fg=confidence_colour,
            )

            if distribution_warning:
                self.distribution_status_label.config(
                    text=(
                        "Live Distribution: "
                        "WARNING - differs from EmoSurv training data"
                    ),
                    fg="#ff6b8a",
                )

            else:
                self.distribution_status_label.config(
                    text=(
                        "Live Distribution: "
                        "reasonably compatible"
                    ),
                    fg="#66ffd6",
                )

            # ----------------------------------------------------------------
            # Detailed probabilities
            # ----------------------------------------------------------------

            ranked_probabilities = sorted(
                result[
                    "probabilities"
                ].items(),
                key=lambda item: item[1],
                reverse=True,
            )

            details = [
                "SenseFuzeAI EmoSurv IEEE Live Prediction",
                "=" * 74,
                "",

                (
                    "IMPORTANT: These behavioural labels are "
                    "weakly supervised proxy outputs."
                ),

                (
                    "They are NOT original EmoSurv "
                    "behavioural ground truth."
                ),

                "",
                "PREDICTION",
                "----------",

                (
                    f"Top class              : "
                    f"{result['current_state']}"
                ),

                (
                    f"Confidence             : "
                    f"{result['confidence']:.4f}"
                ),

                (
                    f"Confidence level       : "
                    f"{result['confidence_level']}"
                ),

                (
                    f"Second class           : "
                    f"{result['second_class']}"
                ),

                (
                    f"Second probability     : "
                    f"{result['second_probability']:.4f}"
                ),

                (
                    f"Confidence gap         : "
                    f"{result['confidence_gap']:.4f}"
                ),

                "",
                "PROBABILITY DISTRIBUTION",
                "------------------------",
                "",
            ]

            for (
                label,
                probability,
            ) in ranked_probabilities:
                bar_length = int(
                    probability
                    * 32
                )

                bar = (
                    "█"
                    * bar_length
                )

                details.append(
                    f"{label:12s}: "
                    f"{probability * 100:6.2f}%  "
                    f"{bar}"
                )

            # ----------------------------------------------------------------
            # OOD diagnostics
            # ----------------------------------------------------------------

            details.extend(
                [
                    "",
                    "TRAINING-vs-LIVE DISTRIBUTION CHECK",
                    "-----------------------------------",

                    (
                        f"Strongly OOD features  : "
                        f"{distribution_result['strongly_ood_count']}"
                        f"/"
                        f"{distribution_result['feature_count']}"
                    ),

                    (
                        f"Strongly OOD ratio     : "
                        f"{distribution_result['strongly_ood_ratio']:.3f}"
                    ),

                    (
                        f"Distribution warning   : "
                        f"{distribution_warning}"
                    ),

                    "",
                    (
                        "A distribution warning means the live "
                        "typing pattern differs substantially from "
                        "the EmoSurv training-data feature range."
                    ),

                    (
                        "The warning does NOT change the predicted "
                        "class; it only identifies possible "
                        "training-deployment domain shift."
                    ),

                    "",
                    "FEATURE DIAGNOSTICS",
                    "-------------------",

                    (
                        f"{'Feature':24s}"
                        f"{'Live':>12s}"
                        f"{'Median':>12s}"
                        f"{'Zrobust':>12s}"
                        f"{'OOD':>8s}"
                    ),

                    "-" * 70,
                ]
            )

            for feature in FEATURE_COLUMNS:
                diagnostic = (
                    distribution_result[
                        "features"
                    ][
                        feature
                    ]
                )

                details.append(
                    f"{feature:24s}"
                    f"{diagnostic['live_value']:>12.4f}"
                    f"{diagnostic['training_median']:>12.4f}"
                    f"{diagnostic['robust_z']:>12.3f}"
                    f"{str(diagnostic['strongly_out_of_distribution']):>8s}"
                )

            self.details_text.delete(
                "1.0",
                tk.END,
            )

            self.details_text.insert(
                tk.END,
                "\n".join(
                    details
                ),
            )

            # ----------------------------------------------------------------
            # Logging
            # ----------------------------------------------------------------

            text_value = (
                self.text_widget.get(
                    "1.0",
                    tk.END,
                )
                .strip()
            )

            with LIVE_LOG_PATH.open(
                "a",
                newline="",
                encoding="utf-8",
            ) as file:
                writer = csv.writer(
                    file
                )

                writer.writerow(
                    [
                        datetime.now().isoformat(
                            timespec="seconds"
                        ),

                        keydown_count,

                        len(
                            text_value
                        ),

                        result[
                            "current_state"
                        ],

                        result[
                            "confidence"
                        ],

                        result[
                            "confidence_level"
                        ],

                        result[
                            "second_class"
                        ],

                        result[
                            "second_probability"
                        ],

                        result[
                            "confidence_gap"
                        ],

                        distribution_warning,

                        distribution_result[
                            "strongly_ood_count"
                        ],

                        distribution_result[
                            "strongly_ood_ratio"
                        ],

                        json.dumps(
                            result[
                                "probabilities"
                            ]
                        ),

                        json.dumps(
                            result[
                                "features"
                            ]
                        ),
                    ]
                )

        except Exception as exc:
            messagebox.showerror(
                "Prediction Error",
                str(exc),
            )

    # ======================================================================
    # Reset
    # ======================================================================

    def reset(
        self,
    ) -> None:
        self.events.clear()

        self.active_keys.clear()

        self.text_widget.delete(
            "1.0",
            tk.END,
        )

        self.key_count_label.config(
            text="Keypresses: 0"
        )

        self.speed_label.config(
            text=(
                "Speed: "
                "0.00 keys/sec"
            )
        )

        self.pause_label.config(
            text=(
                "Pause ratio ≥1s: "
                "0.000"
            )
        )

        self.rhythm_label.config(
            text=(
                "Rhythm CV: "
                "0.000"
            )
        )

        self.correction_label.config(
            text=(
                "Correction ratio: "
                "0.000"
            )
        )

        self.state_label.config(
            text="—"
        )

        self.confidence_label.config(
            text="Confidence: —"
        )

        self.confidence_level_label.config(
            text=(
                "Prediction Confidence: —"
            ),
            fg="#cbd6ff",
        )

        self.distribution_status_label.config(
            text=(
                "Live Distribution: "
                "waiting for input"
            ),
            fg="#cbd6ff",
        )

        self.details_text.delete(
            "1.0",
            tk.END,
        )


# ============================================================================
# MAIN
# ============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "SenseFuzeAI EmoSurv IEEE "
            "keystroke comparison pipeline."
        )
    )

    parser.add_argument(
        "--retrain",
        action="store_true",
        help=(
            "Rebuild the EmoSurv proxy dataset, "
            "evaluate baseline and balanced Random Forests, "
            "select the best candidate and retrain the final model."
        ),
    )

    parser.add_argument(
        "--train-only",
        action="store_true",
        help=(
            "Train/retrain and exit without launching "
            "the graphical interface."
        ),
    )

    args = parser.parse_args()

    print("=" * 88)
    print("SenseFuzeAI - EmoSurv IEEE Keystroke Comparison")
    print("=" * 88)

    print()
    print("Running script:")
    print(
        f"  {Path(__file__).resolve()}"
    )

    print()
    print("Datasets used:")

    print(
        f"  {FIXED_TEXT_PATH}"
    )

    print(
        f"  {FREE_TEXT_PATH}"
    )

    print()
    print("Datasets intentionally excluded:")

    print(
        "  Frequency Dataset.csv"
    )

    print(
        "  Participants Information.csv"
    )

    print()
    print("Original EmoSurv emotion labels:")

    for code, label in EMOTION_CODES.items():
        print(
            f"  {code} = {label}"
        )

    print()
    print("IMPORTANT:")

    print(
        "The four SenseFuzeAI behavioural states are "
        "weakly supervised proxy labels."
    )

    print(
        "They must not be interpreted as original "
        "EmoSurv behavioural ground truth."
    )

    should_train = (
        args.retrain
        or not MODEL_PATH.exists()
        or not METADATA_PATH.exists()
        or not FEATURE_COLUMNS_PATH.exists()
        or not FEATURE_DISTRIBUTION_PATH.exists()
    )

    if should_train:
        print()
        print(
            "Training/rebuilding EmoSurv comparison model..."
        )

        model, metadata = train_emosurv_model()

    else:
        print()
        print(
            "Loading existing EmoSurv comparison model..."
        )

        model, metadata = load_existing_model()

        print(
            f"Loaded model:\n  "
            f"{MODEL_PATH}"
        )

    if args.train_only:
        print()
        print(
            "Training/evaluation completed."
        )

        print(
            "GUI launch skipped because "
            "--train-only was supplied."
        )

        return

    print()
    print(
        "Launching EmoSurv comparison GUI..."
    )

    root = tk.Tk()

    EmoSurvKeystrokeApp(
        root=root,
        model=model,
        metadata=metadata,
    )

    root.mainloop()


if __name__ == "__main__":
    main()
