# === train_fusion_model.py ===
#
# SenseFuzeAI
# Experimental Multimodal Late-Fusion Training and Model Selection Pipeline
#
# -------------------------------------------------------------------------
# PURPOSE
# -------------------------------------------------------------------------
#
# Builds an experimental score-level fusion dataset from:
#
#   1. Audio model predictions
#   2. Original image model predictions
#   3. Webcam-calibrated image model predictions
#   4. Keystroke model predictions
#   5. Text model predictions
#
# Each modality contributes four class-probability/score features:
#
#   focused
#   distracted
#   fatigued
#   overloaded
#
# The script then compares several fusion classifiers.
#
# -------------------------------------------------------------------------
# IMPORTANT IMAGE PIPELINE DESIGN
# -------------------------------------------------------------------------
#
# Original/static image data:
#
#   processed_image_dataset*.csv
#       -> model_artifacts/image_model.joblib
#
# Webcam/video calibration data:
#
#   data/webcam_calibration_clip_features.csv
#       -> models/image_demo/image_pipeline_webcam_calibrated.joblib
#
# The original image artifact is NOT overwritten.
#
# -------------------------------------------------------------------------
# IMPORTANT METHODOLOGICAL NOTE
# -------------------------------------------------------------------------
#
# This script still creates an EXPERIMENTAL score-level fusion dataset by
# stacking modality-level observations. These rows do not necessarily
# represent one shared multimodal interaction/session.
#
# For the final session-aligned SenseFuzeAI implementation, the dedicated
# session-aligned fusion pipeline remains methodologically preferable.
#
# However, this script is useful for:
#
#   - experimental late-fusion comparison,
#   - dissertation evidence,
#   - modality-model orchestration,
#   - testing score-level fusion,
#   - incorporating webcam-domain calibration into image evidence.
#
# -------------------------------------------------------------------------
# OUTPUT
# -------------------------------------------------------------------------
#
# model_artifacts/fusion_model.joblib
# model_artifacts/fusion_model_meta.json
# model_artifacts/fusion_training_report.txt
# model_artifacts/fusion_training_dataset.csv
# model_artifacts/visual_reports/fusion/*
#
# -------------------------------------------------------------------------

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)
from sklearn.model_selection import (
    StratifiedGroupKFold,
    train_test_split,
)
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from utils.training_visuals import generate_training_visuals


# =============================================================================
# PROJECT PATHS
# =============================================================================

ROOT_DIR = Path(__file__).resolve().parent

MODEL_DIR = ROOT_DIR / "model_artifacts"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

VISUAL_REPORT_DIR = MODEL_DIR / "visual_reports"


# =============================================================================
# MODALITY DATASETS
# =============================================================================

AUDIO_DATA = (
    ROOT_DIR
    / "data"
    / "processed"
    / "processed_audio_dataset.csv"
)

IMAGE_DATA_CANDIDATES = [
    ROOT_DIR
    / "data"
    / "processed"
    / "processed_image_dataset_v2.csv",

    ROOT_DIR
    / "data"
    / "processed"
    / "processed_image_dataset.csv",
]

KEYSTROKE_DATA = (
    ROOT_DIR
    / "emosurv_processed"
    / "combined_behaviour_samples.csv"
)

TEXT_DATA = (
    ROOT_DIR
    / "data"
    / "processed"
    / "processed_text_dataset.csv"
)


# =============================================================================
# WEBCAM CALIBRATION DATA
# =============================================================================

WEBCAM_CALIBRATION_DATA_CANDIDATES = [
    # Preferred location from the current calibration pipeline.
    ROOT_DIR
    / "data"
    / "webcam_calibration_clip_features.csv",

    # Optional compatibility location.
    ROOT_DIR
    / "data"
    / "processed"
    / "webcam_calibration_clip_features.csv",
]

WEBCAM_CALIBRATED_IMAGE_MODEL = (
    ROOT_DIR
    / "models"
    / "image_demo"
    / "image_pipeline_webcam_calibrated.joblib"
)

WEBCAM_IMAGE_FEATURE_COLUMNS = (
    ROOT_DIR
    / "models"
    / "image_demo"
    / "feature_columns.json"
)

WEBCAM_IMAGE_METADATA = (
    ROOT_DIR
    / "models"
    / "image_demo"
    / "webcam_calibrated_metadata.json"
)

CLIP_MODEL_PATH = (
    ROOT_DIR
    / "models"
    / "clip-vit-large-patch14"
)

EXPECTED_CLIP_DIMENSION = 768


# =============================================================================
# ORIGINAL MODALITY MODEL ARTIFACTS
# =============================================================================

AUDIO_MODEL = (
    MODEL_DIR
    / "audio_model.joblib"
)

AUDIO_META = (
    MODEL_DIR
    / "audio_model_meta.json"
)

ORIGINAL_IMAGE_MODEL = (
    MODEL_DIR
    / "image_model.joblib"
)

ORIGINAL_IMAGE_META = (
    MODEL_DIR
    / "image_model_meta.json"
)

KEYSTROKE_MODEL = (
    MODEL_DIR
    / "keystroke_model.joblib"
)

KEYSTROKE_META = (
    MODEL_DIR
    / "keystroke_model_meta.json"
)

TEXT_MODEL = (
    MODEL_DIR
    / "text_model.joblib"
)

TEXT_META = (
    MODEL_DIR
    / "text_model_meta.json"
)


# =============================================================================
# FUSION OUTPUTS
# =============================================================================

FUSION_MODEL = (
    MODEL_DIR
    / "fusion_model.joblib"
)

FUSION_META = (
    MODEL_DIR
    / "fusion_model_meta.json"
)

FUSION_REPORT = (
    MODEL_DIR
    / "fusion_training_report.txt"
)

FUSION_DATASET_OUT = (
    MODEL_DIR
    / "fusion_training_dataset.csv"
)


# =============================================================================
# CONFIGURATION
# =============================================================================

CLASSES = [
    "focused",
    "distracted",
    "fatigued",
    "overloaded",
]

RANDOM_STATE = 42

# Approximately equivalent to a 20% holdout.
N_SPLITS = 5

# Used only as fallback if StratifiedGroupKFold cannot be constructed.
TEST_SIZE = 0.20


# =============================================================================
# GENERAL UTILITIES
# =============================================================================

def heading(title: str) -> None:
    print()
    print("=" * 84)
    print(title)
    print("=" * 84)


def resolve_existing_path(
    candidates: list[Path],
) -> Path | None:
    for path in candidates:
        if path.exists():
            return path

    return None


def load_json(
    path: Path,
) -> dict[str, Any]:
    if not path.exists():
        return {}

    try:
        with path.open(
            "r",
            encoding="utf-8",
        ) as f:
            value = json.load(f)

        return (
            value
            if isinstance(value, dict)
            else {}
        )

    except Exception:
        return {}


def load_json_list(
    path: Path,
) -> list[str]:
    if not path.exists():
        return []

    try:
        with path.open(
            "r",
            encoding="utf-8",
        ) as f:
            value = json.load(f)

        if not isinstance(
            value,
            list,
        ):
            return []

        return [
            str(item)
            for item in value
        ]

    except Exception:
        return []


def normalise_label(
    value: Any,
) -> str:
    if pd.isna(value):
        return ""

    return (
        str(value)
        .strip()
        .lower()
    )


def remove_error_rows(
    df: pd.DataFrame,
) -> pd.DataFrame:
    if "error" not in df.columns:
        return df.copy()

    values = (
        df["error"]
        .astype(str)
        .str.strip()
        .str.lower()
    )

    valid = (
        df["error"].isna()
        | (values == "")
        | (values == "nan")
        | (values == "none")
    )

    return df[
        valid
    ].copy()


# =============================================================================
# PROBABILITY FEATURES
# =============================================================================

def probability_columns(
    prefix: str,
) -> list[str]:
    return [
        f"{prefix}_{label}_prob"
        for label in CLASSES
    ]


def empty_prob_row(
    prefix: str,
) -> dict[str, float]:
    return {
        column: 0.0
        for column
        in probability_columns(
            prefix
        )
    }


def get_model_classes(
    model: Any,
) -> list[str]:
    classes = list(
        getattr(
            model,
            "classes_",
            [],
        )
    )

    if (
        not classes
        and hasattr(
            model,
            "named_steps",
        )
    ):
        final_estimator = list(
            model.named_steps.values()
        )[-1]

        classes = list(
            getattr(
                final_estimator,
                "classes_",
                [],
            )
        )

    return [
        normalise_label(
            label
        )
        for label in classes
    ]


def softmax(
    values: np.ndarray,
) -> np.ndarray:
    values = np.asarray(
        values,
        dtype=float,
    )

    values = (
        values
        - np.max(
            values
        )
    )

    exp_values = np.exp(
        values
    )

    total = float(
        np.sum(
            exp_values
        )
    )

    if total <= 0:
        return (
            np.ones_like(
                exp_values
            )
            / len(
                exp_values
            )
        )

    return (
        exp_values
        / total
    )


def prediction_scores_as_dict(
    model: Any,
    X: Any,
    prefix: str,
) -> pd.DataFrame:
    """
    Convert predictions from a trained modality model into four
    fusion-compatible score columns.

    Priority:
        1. predict_proba()
        2. decision_function() -> softmax
        3. one-hot predict() fallback
    """

    model_classes = (
        get_model_classes(
            model
        )
    )

    if not model_classes:
        model_classes = (
            CLASSES.copy()
        )

    rows: list[
        dict[str, float]
    ] = []

    # -------------------------------------------------------------------------
    # Probability-capable classifier
    # -------------------------------------------------------------------------

    if hasattr(
        model,
        "predict_proba",
    ):
        probabilities = (
            model.predict_proba(
                X
            )
        )

        for row_probs in probabilities:
            row = empty_prob_row(
                prefix
            )

            for (
                label,
                probability,
            ) in zip(
                model_classes,
                row_probs,
            ):
                label = normalise_label(
                    label
                )

                if label in CLASSES:
                    row[
                        f"{prefix}_{label}_prob"
                    ] = float(
                        probability
                    )

            rows.append(
                row
            )

        return pd.DataFrame(
            rows
        )

    # -------------------------------------------------------------------------
    # Decision-function classifier
    # -------------------------------------------------------------------------

    if hasattr(
        model,
        "decision_function",
    ):
        scores = model.decision_function(
            X
        )

        scores = np.asarray(
            scores
        )

        if scores.ndim == 1:
            scores = np.vstack(
                [
                    -scores,
                    scores,
                ]
            ).T

        for row_scores in scores:
            row = empty_prob_row(
                prefix
            )

            probabilities = softmax(
                row_scores
            )

            for (
                label,
                probability,
            ) in zip(
                model_classes,
                probabilities,
            ):
                label = normalise_label(
                    label
                )

                if label in CLASSES:
                    row[
                        f"{prefix}_{label}_prob"
                    ] = float(
                        probability
                    )

            rows.append(
                row
            )

        return pd.DataFrame(
            rows
        )

    # -------------------------------------------------------------------------
    # Hard prediction fallback
    # -------------------------------------------------------------------------

    predictions = model.predict(
        X
    )

    for prediction in predictions:
        row = empty_prob_row(
            prefix
        )

        prediction = normalise_label(
            prediction
        )

        if prediction in CLASSES:
            row[
                f"{prefix}_{prediction}_prob"
            ] = 1.0

        else:
            for label in CLASSES:
                row[
                    f"{prefix}_{label}_prob"
                ] = (
                    1.0
                    / len(
                        CLASSES
                    )
                )

        rows.append(
            row
        )

    return pd.DataFrame(
        rows
    )


# =============================================================================
# DATAFRAME HELPERS
# =============================================================================

def numeric_feature_matrix(
    df: pd.DataFrame,
    feature_columns: list[str],
) -> pd.DataFrame:
    X = df.reindex(
        columns=feature_columns,
    ).copy()

    for column in X.columns:
        X[column] = pd.to_numeric(
            X[column],
            errors="coerce",
        )

    X = X.replace(
        [
            np.inf,
            -np.inf,
        ],
        np.nan,
    )

    X = X.fillna(
        0.0
    )

    return X


def ensure_probability_columns(
    df: pd.DataFrame,
) -> pd.DataFrame:
    df = df.copy()

    for prefix in [
        "audio",
        "image",
        "keystroke",
        "text",
    ]:
        for column in probability_columns(
            prefix
        ):
            if column not in df.columns:
                df[
                    column
                ] = 0.0

    return df


def add_modality_presence_flags(
    df: pd.DataFrame,
) -> pd.DataFrame:
    df = df.copy()

    df["has_audio"] = (
        df[
            probability_columns(
                "audio"
            )
        ]
        .sum(axis=1)
        > 0
    ).astype(int)

    df["has_image"] = (
        df[
            probability_columns(
                "image"
            )
        ]
        .sum(axis=1)
        > 0
    ).astype(int)

    df["has_keystroke"] = (
        df[
            probability_columns(
                "keystroke"
            )
        ]
        .sum(axis=1)
        > 0
    ).astype(int)

    df["has_text"] = (
        df[
            probability_columns(
                "text"
            )
        ]
        .sum(axis=1)
        > 0
    ).astype(int)

    return df


def unique_row_groups(
    prefix: str,
    row_count: int,
) -> list[str]:
    return [
        f"{prefix}::row::{index}"
        for index
        in range(
            row_count
        )
    ]


# =============================================================================
# AUDIO FUSION ROWS
# =============================================================================

def build_audio_fusion_rows() -> pd.DataFrame:
    heading(
        "BUILDING AUDIO FUSION ROWS"
    )

    if not AUDIO_DATA.exists():
        print(
            f"Audio dataset not found: "
            f"{AUDIO_DATA}"
        )

        return pd.DataFrame()

    if (
        not AUDIO_MODEL.exists()
        or not AUDIO_META.exists()
    ):
        print(
            "Audio model or metadata not found."
        )

        return pd.DataFrame()

    df = pd.read_csv(
        AUDIO_DATA
    )

    meta = load_json(
        AUDIO_META
    )

    model = joblib.load(
        AUDIO_MODEL
    )

    if "label" not in df.columns:
        print(
            "Audio dataset missing label column."
        )

        return pd.DataFrame()

    df["label"] = (
        df["label"]
        .apply(
            normalise_label
        )
    )

    df = remove_error_rows(
        df
    )

    df = df[
        df["label"].isin(
            CLASSES
        )
    ].copy()

    features = meta.get(
        "features",
        [],
    )

    if not features:
        print(
            "Audio metadata missing feature list."
        )

        return pd.DataFrame()

    if df.empty:
        print(
            "No valid audio rows available."
        )

        return pd.DataFrame()

    X = numeric_feature_matrix(
        df,
        features,
    )

    probabilities = (
        prediction_scores_as_dict(
            model=model,
            X=X,
            prefix="audio",
        )
    )

    output = pd.concat(
        [
            df[
                ["label"]
            ]
            .reset_index(
                drop=True
            ),

            probabilities.reset_index(
                drop=True
            ),
        ],
        axis=1,
    )

    output[
        "source_modality"
    ] = "audio"

    output[
        "source_variant"
    ] = "original_audio"

    output[
        "source_group"
    ] = unique_row_groups(
        "audio",
        len(
            output
        ),
    )

    print(
        f"Audio fusion rows: "
        f"{len(output)}"
    )

    return output


# =============================================================================
# ORIGINAL IMAGE FUSION ROWS
# =============================================================================

def build_original_image_fusion_rows() -> pd.DataFrame:
    heading(
        "BUILDING ORIGINAL IMAGE FUSION ROWS"
    )

    image_data = (
        resolve_existing_path(
            IMAGE_DATA_CANDIDATES
        )
    )

    if image_data is None:
        print(
            "Original image dataset not found."
        )

        return pd.DataFrame()

    if (
        not ORIGINAL_IMAGE_MODEL.exists()
        or not ORIGINAL_IMAGE_META.exists()
    ):
        print(
            "Original image model or metadata not found."
        )

        return pd.DataFrame()

    df = pd.read_csv(
        image_data
    )

    meta = load_json(
        ORIGINAL_IMAGE_META
    )

    model = joblib.load(
        ORIGINAL_IMAGE_MODEL
    )

    if "label" not in df.columns:
        print(
            "Original image dataset missing label."
        )

        return pd.DataFrame()

    df["label"] = (
        df["label"]
        .apply(
            normalise_label
        )
    )

    df = remove_error_rows(
        df
    )

    df = df[
        df["label"].isin(
            CLASSES
        )
    ].copy()

    features = meta.get(
        "features",
        [],
    )

    if not features:
        print(
            "Original image metadata missing feature list."
        )

        return pd.DataFrame()

    if df.empty:
        print(
            "No valid original image rows."
        )

        return pd.DataFrame()

    X = numeric_feature_matrix(
        df,
        features,
    )

    probabilities = (
        prediction_scores_as_dict(
            model=model,
            X=X,
            prefix="image",
        )
    )

    output = pd.concat(
        [
            df[
                ["label"]
            ]
            .reset_index(
                drop=True
            ),

            probabilities.reset_index(
                drop=True
            ),
        ],
        axis=1,
    )

    output[
        "source_modality"
    ] = "image"

    output[
        "source_variant"
    ] = "original_image"

    # If the source dataset has a genuine session identifier, retain grouping.
    if "session_id" in df.columns:
        output[
            "source_group"
        ] = (
            "image_original::"
            + df[
                "session_id"
            ]
            .astype(str)
            .reset_index(
                drop=True
            )
        )

    else:
        output[
            "source_group"
        ] = unique_row_groups(
            "image_original",
            len(
                output
            ),
        )

    print(
        f"Original image fusion rows: "
        f"{len(output)}"
    )

    return output


# =============================================================================
# WEBCAM-CALIBRATED IMAGE FUSION ROWS
# =============================================================================

def build_webcam_image_fusion_rows() -> pd.DataFrame:
    heading(
        "BUILDING WEBCAM-CALIBRATED IMAGE FUSION ROWS"
    )

    calibration_data = (
        resolve_existing_path(
            WEBCAM_CALIBRATION_DATA_CANDIDATES
        )
    )

    if calibration_data is None:
        print(
            "Webcam calibration CLIP dataset not found."
        )

        return pd.DataFrame()

    if not WEBCAM_CALIBRATED_IMAGE_MODEL.exists():
        print(
            "Webcam-calibrated image model not found:"
        )

        print(
            WEBCAM_CALIBRATED_IMAGE_MODEL
        )

        return pd.DataFrame()

    if not WEBCAM_IMAGE_FEATURE_COLUMNS.exists():
        print(
            "Webcam image feature schema not found:"
        )

        print(
            WEBCAM_IMAGE_FEATURE_COLUMNS
        )

        return pd.DataFrame()

    feature_columns = load_json_list(
        WEBCAM_IMAGE_FEATURE_COLUMNS
    )

    if not feature_columns:
        print(
            "Webcam image feature schema is empty."
        )

        return pd.DataFrame()

    if len(
        feature_columns
    ) != EXPECTED_CLIP_DIMENSION:
        print(
            "Warning: expected "
            f"{EXPECTED_CLIP_DIMENSION} CLIP features, "
            f"received {len(feature_columns)}."
        )

    df = pd.read_csv(
        calibration_data
    )

    if "label" not in df.columns:
        print(
            "Webcam calibration dataset missing label column."
        )

        return pd.DataFrame()

    df["label"] = (
        df["label"]
        .apply(
            normalise_label
        )
    )

    df = df[
        df["label"].isin(
            CLASSES
        )
    ].copy()

    if df.empty:
        print(
            "No valid webcam calibration rows."
        )

        return pd.DataFrame()

    missing = [
        column
        for column
        in feature_columns
        if column not in df.columns
    ]

    if missing:
        print(
            "Webcam calibration dataset is incompatible "
            "with the calibrated image model."
        )

        print(
            f"Missing feature examples: "
            f"{missing[:20]}"
        )

        return pd.DataFrame()

    X = numeric_feature_matrix(
        df,
        feature_columns,
    )

    model = joblib.load(
        WEBCAM_CALIBRATED_IMAGE_MODEL
    )

    probabilities = (
        prediction_scores_as_dict(
            model=model,
            X=X,
            prefix="image",
        )
    )

    output = pd.concat(
        [
            df[
                ["label"]
            ]
            .reset_index(
                drop=True
            ),

            probabilities.reset_index(
                drop=True
            ),
        ],
        axis=1,
    )

    output[
        "source_modality"
    ] = "image"

    output[
        "source_variant"
    ] = "webcam_calibrated"

    # -------------------------------------------------------------------------
    # Critical leakage protection
    # -------------------------------------------------------------------------
    #
    # Multiple calibration frames may come from the same source video.
    # All frames belonging to one source video receive the same group ID.
    #
    # StratifiedGroupKFold later ensures the group remains entirely in either
    # training or testing.
    # -------------------------------------------------------------------------

    if "source_video" in df.columns:
        output[
            "source_group"
        ] = (
            "webcam::"
            + df[
                "source_video"
            ]
            .astype(str)
            .reset_index(
                drop=True
            )
        )

    else:
        print(
            "Warning: calibration dataset has no source_video column."
        )

        print(
            "Each webcam frame will be treated as an independent group."
        )

        output[
            "source_group"
        ] = unique_row_groups(
            "webcam",
            len(
                output
            ),
        )

    print(
        f"Webcam calibration rows: "
        f"{len(output)}"
    )

    print(
        "Webcam source groups: "
        f"{output['source_group'].nunique()}"
    )

    print()
    print(
        "Webcam label distribution:"
    )

    print(
        output[
            "label"
        ]
        .value_counts()
        .reindex(
            CLASSES,
            fill_value=0,
        )
    )

    return output


# =============================================================================
# KEYSTROKE FUSION ROWS
# =============================================================================

def build_keystroke_fusion_rows() -> pd.DataFrame:
    heading(
        "BUILDING KEYSTROKE FUSION ROWS"
    )

    if not KEYSTROKE_DATA.exists():
        print(
            f"Keystroke dataset not found: "
            f"{KEYSTROKE_DATA}"
        )

        return pd.DataFrame()

    if (
        not KEYSTROKE_MODEL.exists()
        or not KEYSTROKE_META.exists()
    ):
        print(
            "Keystroke model or metadata not found."
        )

        return pd.DataFrame()

    df = pd.read_csv(
        KEYSTROKE_DATA
    )

    meta = load_json(
        KEYSTROKE_META
    )

    model = joblib.load(
        KEYSTROKE_MODEL
    )

    target = meta.get(
        "target",
        "behaviour_state",
    )

    features = meta.get(
        "features",
        [],
    )

    if target not in df.columns:
        print(
            "Keystroke dataset missing target column: "
            f"{target}"
        )

        return pd.DataFrame()

    if not features:
        print(
            "Keystroke metadata missing feature list."
        )

        return pd.DataFrame()

    df[target] = (
        df[target]
        .apply(
            normalise_label
        )
    )

    df = df[
        df[target].isin(
            CLASSES
        )
    ].copy()

    if df.empty:
        print(
            "No valid keystroke rows."
        )

        return pd.DataFrame()

    X = numeric_feature_matrix(
        df,
        features,
    )

    probabilities = (
        prediction_scores_as_dict(
            model=model,
            X=X,
            prefix="keystroke",
        )
    )

    output = pd.concat(
        [
            df[
                [target]
            ]
            .rename(
                columns={
                    target: "label"
                }
            )
            .reset_index(
                drop=True
            ),

            probabilities.reset_index(
                drop=True
            ),
        ],
        axis=1,
    )

    output[
        "source_modality"
    ] = "keystroke"

    output[
        "source_variant"
    ] = "original_keystroke"

    if "session_id" in df.columns:
        output[
            "source_group"
        ] = (
            "keystroke::"
            + df[
                "session_id"
            ]
            .astype(str)
            .reset_index(
                drop=True
            )
        )

    else:
        output[
            "source_group"
        ] = unique_row_groups(
            "keystroke",
            len(
                output
            ),
        )

    print(
        f"Keystroke fusion rows: "
        f"{len(output)}"
    )

    return output


# =============================================================================
# TEXT FUSION ROWS
# =============================================================================

def build_text_fusion_rows() -> pd.DataFrame:
    heading(
        "BUILDING TEXT FUSION ROWS"
    )

    if not TEXT_DATA.exists():
        print(
            f"Text dataset not found: "
            f"{TEXT_DATA}"
        )

        return pd.DataFrame()

    if not TEXT_MODEL.exists():
        print(
            f"Text model not found: "
            f"{TEXT_MODEL}"
        )

        return pd.DataFrame()

    df = pd.read_csv(
        TEXT_DATA
    )

    model = joblib.load(
        TEXT_MODEL
    )

    meta = load_json(
        TEXT_META
    )

    text_column = meta.get(
        "text_column",
        "text",
    )

    target_column = meta.get(
        "target",
        "label",
    )

    if (
        text_column not in df.columns
        or target_column not in df.columns
    ):
        print(
            "Text dataset must contain "
            f"'{text_column}' and '{target_column}'."
        )

        return pd.DataFrame()

    df[
        text_column
    ] = (
        df[
            text_column
        ]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    df[
        target_column
    ] = (
        df[
            target_column
        ]
        .apply(
            normalise_label
        )
    )

    df = remove_error_rows(
        df
    )

    df = df[
        df[
            text_column
        ]
        != ""
    ].copy()

    df = df[
        df[
            target_column
        ]
        .isin(
            CLASSES
        )
    ].copy()

    if df.empty:
        print(
            "No valid text rows."
        )

        return pd.DataFrame()

    probabilities = (
        prediction_scores_as_dict(
            model=model,
            X=df[
                text_column
            ].astype(str),
            prefix="text",
        )
    )

    output = pd.concat(
        [
            df[
                [target_column]
            ]
            .rename(
                columns={
                    target_column:
                    "label"
                }
            )
            .reset_index(
                drop=True
            ),

            probabilities.reset_index(
                drop=True
            ),
        ],
        axis=1,
    )

    output[
        "source_modality"
    ] = "text"

    output[
        "source_variant"
    ] = "original_text"

    if "session_id" in df.columns:
        output[
            "source_group"
        ] = (
            "text::"
            + df[
                "session_id"
            ]
            .astype(str)
            .reset_index(
                drop=True
            )
        )

    else:
        output[
            "source_group"
        ] = unique_row_groups(
            "text",
            len(
                output
            ),
        )

    print(
        f"Text fusion rows: "
        f"{len(output)}"
    )

    return output


# =============================================================================
# FUSION DATASET
# =============================================================================

def build_fusion_dataset() -> pd.DataFrame:
    heading(
        "BUILDING EXPERIMENTAL FUSION DATASET"
    )

    parts = [
        build_audio_fusion_rows(),
        build_original_image_fusion_rows(),
        build_webcam_image_fusion_rows(),
        build_keystroke_fusion_rows(),
        build_text_fusion_rows(),
    ]

    parts = [
        part
        for part in parts
        if not part.empty
    ]

    if not parts:
        raise ValueError(
            "No modality prediction data are available "
            "for fusion training."
        )

    df = pd.concat(
        parts,
        ignore_index=True,
        sort=False,
    )

    df[
        "label"
    ] = (
        df[
            "label"
        ]
        .apply(
            normalise_label
        )
    )

    df = df[
        df[
            "label"
        ]
        .isin(
            CLASSES
        )
    ].copy()

    df = ensure_probability_columns(
        df
    )

    df = add_modality_presence_flags(
        df
    )

    if "source_group" not in df.columns:
        df[
            "source_group"
        ] = unique_row_groups(
            "fusion",
            len(
                df
            ),
        )

    df[
        "source_group"
    ] = (
        df[
            "source_group"
        ]
        .fillna("")
        .astype(str)
    )

    empty_group_mask = (
        df[
            "source_group"
        ].str.strip()
        == ""
    )

    if empty_group_mask.any():
        replacements = unique_row_groups(
            "fusion_fallback",
            int(
                empty_group_mask.sum()
            ),
        )

        df.loc[
            empty_group_mask,
            "source_group",
        ] = replacements

    return df


def validate_fusion_dataset(
    df: pd.DataFrame,
) -> None:
    counts = (
        df[
            "label"
        ]
        .value_counts()
    )

    missing_classes = [
        label
        for label in CLASSES
        if label not in counts.index
    ]

    if missing_classes:
        print(
            "WARNING: missing fusion classes:"
        )

        print(
            missing_classes
        )

    insufficient = (
        counts[
            counts < 2
        ]
    )

    if not insufficient.empty:
        raise ValueError(
            "Some classes have fewer than two fusion rows:\n"
            f"{insufficient}"
        )


# =============================================================================
# LEAKAGE-AWARE SPLIT
# =============================================================================

def group_aware_train_test_split(
    df: pd.DataFrame,
    feature_columns: list[str],
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.Series,
    pd.Series,
    pd.DataFrame,
    pd.DataFrame,
    str,
]:
    """
    Prefer StratifiedGroupKFold.

    This prevents calibration frames from the same source video from appearing
    in both the training and held-out test partitions.

    If group-aware splitting unexpectedly fails, a stratified row split is
    used as a documented fallback.
    """

    X = df[
        feature_columns
    ].copy()

    y = df[
        "label"
    ].copy()

    groups = df[
        "source_group"
    ].copy()

    try:
        splitter = (
            StratifiedGroupKFold(
                n_splits=N_SPLITS,
                shuffle=True,
                random_state=RANDOM_STATE,
            )
        )

        splits = list(
            splitter.split(
                X,
                y,
                groups=groups,
            )
        )

        if not splits:
            raise RuntimeError(
                "No StratifiedGroupKFold split generated."
            )

        train_indices, test_indices = (
            splits[0]
        )

        train_df = (
            df.iloc[
                train_indices
            ]
            .copy()
        )

        test_df = (
            df.iloc[
                test_indices
            ]
            .copy()
        )

        train_groups = set(
            train_df[
                "source_group"
            ]
        )

        test_groups = set(
            test_df[
                "source_group"
            ]
        )

        overlap = (
            train_groups
            & test_groups
        )

        if overlap:
            raise RuntimeError(
                "Group leakage detected between "
                "training and test partitions."
            )

        split_method = (
            "StratifiedGroupKFold "
            "with source_group leakage protection"
        )

        return (
            train_df[
                feature_columns
            ],
            test_df[
                feature_columns
            ],
            train_df[
                "label"
            ],
            test_df[
                "label"
            ],
            train_df,
            test_df,
            split_method,
        )

    except Exception as exc:
        print()
        print(
            "WARNING: group-aware split failed."
        )

        print(
            f"Reason: {exc}"
        )

        print(
            "Using stratified row-level fallback."
        )

        (
            train_indices,
            test_indices,
        ) = train_test_split(
            np.arange(
                len(
                    df
                )
            ),
            test_size=TEST_SIZE,
            stratify=y,
            random_state=RANDOM_STATE,
        )

        train_df = (
            df.iloc[
                train_indices
            ]
            .copy()
        )

        test_df = (
            df.iloc[
                test_indices
            ]
            .copy()
        )

        split_method = (
            "fallback_stratified_train_test_split"
        )

        return (
            train_df[
                feature_columns
            ],
            test_df[
                feature_columns
            ],
            train_df[
                "label"
            ],
            test_df[
                "label"
            ],
            train_df,
            test_df,
            split_method,
        )


# =============================================================================
# CANDIDATE FUSION MODELS
# =============================================================================

def build_candidate_pipelines() -> dict[str, Pipeline]:
    return {
        "svm_rbf": Pipeline(
            steps=[
                (
                    "imputer",
                    SimpleImputer(
                        strategy="median"
                    ),
                ),
                (
                    "scaler",
                    StandardScaler(),
                ),
                (
                    "model",
                    SVC(
                        kernel="rbf",
                        C=1.0,
                        gamma="scale",
                        class_weight="balanced",
                        probability=True,
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),

        "svm_linear": Pipeline(
            steps=[
                (
                    "imputer",
                    SimpleImputer(
                        strategy="median"
                    ),
                ),
                (
                    "scaler",
                    StandardScaler(),
                ),
                (
                    "model",
                    SVC(
                        kernel="linear",
                        C=1.0,
                        class_weight="balanced",
                        probability=True,
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),

        "random_forest": Pipeline(
            steps=[
                (
                    "imputer",
                    SimpleImputer(
                        strategy="median"
                    ),
                ),
                (
                    "model",
                    RandomForestClassifier(
                        n_estimators=300,
                        min_samples_split=4,
                        min_samples_leaf=2,
                        class_weight="balanced",
                        random_state=RANDOM_STATE,
                        n_jobs=-1,
                    ),
                ),
            ]
        ),

        "logistic_regression": Pipeline(
            steps=[
                (
                    "imputer",
                    SimpleImputer(
                        strategy="median"
                    ),
                ),
                (
                    "scaler",
                    StandardScaler(),
                ),
                (
                    "model",
                    LogisticRegression(
                        max_iter=3000,
                        class_weight="balanced",
                        solver="lbfgs",
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),

        "decision_tree": Pipeline(
            steps=[
                (
                    "imputer",
                    SimpleImputer(
                        strategy="median"
                    ),
                ),
                (
                    "model",
                    DecisionTreeClassifier(
                        min_samples_split=4,
                        min_samples_leaf=2,
                        class_weight="balanced",
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),

        "gaussian_naive_bayes": Pipeline(
            steps=[
                (
                    "imputer",
                    SimpleImputer(
                        strategy="median"
                    ),
                ),
                (
                    "scaler",
                    StandardScaler(),
                ),
                (
                    "model",
                    GaussianNB(),
                ),
            ]
        ),

        "knn_3": Pipeline(
            steps=[
                (
                    "imputer",
                    SimpleImputer(
                        strategy="median"
                    ),
                ),
                (
                    "scaler",
                    StandardScaler(),
                ),
                (
                    "model",
                    KNeighborsClassifier(
                        n_neighbors=3,
                        weights="distance",
                    ),
                ),
            ]
        ),

        "knn_5": Pipeline(
            steps=[
                (
                    "imputer",
                    SimpleImputer(
                        strategy="median"
                    ),
                ),
                (
                    "scaler",
                    StandardScaler(),
                ),
                (
                    "model",
                    KNeighborsClassifier(
                        n_neighbors=5,
                        weights="distance",
                    ),
                ),
            ]
        ),
    }


# =============================================================================
# METRICS
# =============================================================================

def compute_metrics(
    y_true: pd.Series,
    y_pred: Any,
) -> dict[str, Any]:
    accuracy = accuracy_score(
        y_true,
        y_pred,
    )

    (
        macro_precision,
        macro_recall,
        macro_f1,
        _,
    ) = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=CLASSES,
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
        labels=CLASSES,
        average="weighted",
        zero_division=0,
    )

    report_text = (
        classification_report(
            y_true,
            y_pred,
            labels=CLASSES,
            zero_division=0,
        )
    )

    report_dict = (
        classification_report(
            y_true,
            y_pred,
            labels=CLASSES,
            zero_division=0,
            output_dict=True,
        )
    )

    matrix = confusion_matrix(
        y_true,
        y_pred,
        labels=CLASSES,
    )

    return {
        "accuracy": float(
            accuracy
        ),

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

        "confusion_matrix": (
            matrix.tolist()
        ),
    }


# =============================================================================
# MODEL EVALUATION
# =============================================================================

def evaluate_candidates(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
) -> tuple[
    str,
    Pipeline,
    dict[str, dict[str, Any]],
]:
    candidates = (
        build_candidate_pipelines()
    )

    results: dict[
        str,
        dict[str, Any],
    ] = {}

    best_name: str | None = None

    best_model: Pipeline | None = None

    best_macro_f1 = -1.0

    best_accuracy = -1.0

    for (
        name,
        pipeline,
    ) in candidates.items():
        print()
        print(
            f"Training fusion candidate: {name}"
        )

        try:
            pipeline.fit(
                X_train,
                y_train,
            )

            predictions = (
                pipeline.predict(
                    X_test
                )
            )

            metrics = compute_metrics(
                y_test,
                predictions,
            )

            metrics[
                "status"
            ] = "evaluated"

            results[
                name
            ] = metrics

            print(
                f"  Accuracy    : "
                f"{metrics['accuracy']:.4f}"
            )

            print(
                f"  Macro F1    : "
                f"{metrics['macro_f1']:.4f}"
            )

            print(
                f"  Weighted F1 : "
                f"{metrics['weighted_f1']:.4f}"
            )

            better = (
                metrics[
                    "macro_f1"
                ]
                > best_macro_f1
                or (
                    metrics[
                        "macro_f1"
                    ]
                    == best_macro_f1
                    and metrics[
                        "accuracy"
                    ]
                    > best_accuracy
                )
            )

            if better:
                best_name = name

                best_model = (
                    pipeline
                )

                best_macro_f1 = (
                    metrics[
                        "macro_f1"
                    ]
                )

                best_accuracy = (
                    metrics[
                        "accuracy"
                    ]
                )

        except Exception as exc:
            print(
                f"  FAILED: {exc}"
            )

            results[
                name
            ] = {
                "status": "error",
                "reason": str(
                    exc
                ),
                "accuracy": 0.0,
                "macro_precision": 0.0,
                "macro_recall": 0.0,
                "macro_f1": 0.0,
                "weighted_precision": 0.0,
                "weighted_recall": 0.0,
                "weighted_f1": 0.0,
                "classification_report_text": (
                    f"Model failed: {exc}"
                ),
                "classification_report": {},
                "confusion_matrix": [],
            }

    if (
        best_name is None
        or best_model is None
    ):
        raise RuntimeError(
            "No fusion candidate model trained successfully."
        )

    return (
        best_name,
        best_model,
        results,
    )


def candidate_summary(
    results: dict[
        str,
        dict[str, Any],
    ],
) -> dict[str, dict[str, Any]]:
    summary: dict[
        str,
        dict[str, Any],
    ] = {}

    for (
        name,
        metrics,
    ) in results.items():
        summary[
            name
        ] = {
            "status": metrics.get(
                "status",
                "evaluated",
            ),

            "reason": metrics.get(
                "reason"
            ),

            "accuracy": metrics.get(
                "accuracy",
                0.0,
            ),

            "macro_precision": metrics.get(
                "macro_precision",
                0.0,
            ),

            "macro_recall": metrics.get(
                "macro_recall",
                0.0,
            ),

            "macro_f1": metrics.get(
                "macro_f1",
                0.0,
            ),

            "weighted_precision": metrics.get(
                "weighted_precision",
                0.0,
            ),

            "weighted_recall": metrics.get(
                "weighted_recall",
                0.0,
            ),

            "weighted_f1": metrics.get(
                "weighted_f1",
                0.0,
            ),

            "confusion_matrix": metrics.get(
                "confusion_matrix",
                [],
            ),
        }

    return summary


# =============================================================================
# SAVE METADATA
# =============================================================================

def save_metadata(
    best_name: str,
    best_metrics: dict[str, Any],
    candidate_results: dict[str, dict[str, Any]],
    df: pd.DataFrame,
    feature_columns: list[str],
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    split_method: str,
    visual_paths: dict[str, str],
) -> None:
    webcam_data = (
        resolve_existing_path(
            WEBCAM_CALIBRATION_DATA_CANDIDATES
        )
    )

    metadata = {
        "model_name": (
            "experimental_multimodal_"
            f"late_fusion_{best_name}"
        ),

        "created_at": (
            datetime.now()
            .astimezone()
            .isoformat()
        ),

        "classes": (
            CLASSES
        ),

        "features": (
            feature_columns
        ),

        "selection_metric": (
            "macro_f1_then_accuracy"
        ),

        "selected_classifier": (
            best_name
        ),

        "accuracy": (
            best_metrics[
                "accuracy"
            ]
        ),

        "macro_precision": (
            best_metrics[
                "macro_precision"
            ]
        ),

        "macro_recall": (
            best_metrics[
                "macro_recall"
            ]
        ),

        "macro_f1": (
            best_metrics[
                "macro_f1"
            ]
        ),

        "weighted_precision": (
            best_metrics[
                "weighted_precision"
            ]
        ),

        "weighted_recall": (
            best_metrics[
                "weighted_recall"
            ]
        ),

        "weighted_f1": (
            best_metrics[
                "weighted_f1"
            ]
        ),

        "method": (
            "experimental_score_level_late_fusion"
        ),

        "model_path": str(
            FUSION_MODEL
        ),

        "fusion_dataset_path": str(
            FUSION_DATASET_OUT
        ),

        "split_method": (
            split_method
        ),

        "group_column": (
            "source_group"
        ),

        "random_state": (
            RANDOM_STATE
        ),

        "rows_used": int(
            len(
                df
            )
        ),

        "train_samples": int(
            len(
                train_df
            )
        ),

        "test_samples": int(
            len(
                test_df
            )
        ),

        "source_modality_distribution": (
            df[
                "source_modality"
            ]
            .value_counts()
            .to_dict()
        ),

        "source_variant_distribution": (
            df[
                "source_variant"
            ]
            .value_counts()
            .to_dict()
        ),

        "class_distribution": (
            df[
                "label"
            ]
            .value_counts()
            .reindex(
                CLASSES,
                fill_value=0,
            )
            .to_dict()
        ),

        "candidate_model_comparison": (
            candidate_summary(
                candidate_results
            )
        ),

        "visual_reports": (
            visual_paths
        ),

        "modality_models": {
            "audio_model": str(
                AUDIO_MODEL
            ),

            "original_image_model": str(
                ORIGINAL_IMAGE_MODEL
            ),

            "webcam_calibrated_image_model": str(
                WEBCAM_CALIBRATED_IMAGE_MODEL
            ),

            "keystroke_model": str(
                KEYSTROKE_MODEL
            ),

            "text_model": str(
                TEXT_MODEL
            ),
        },

        "webcam_calibration": {
            "enabled": (
                webcam_data
                is not None
                and WEBCAM_CALIBRATED_IMAGE_MODEL.exists()
            ),

            "dataset": (
                str(
                    webcam_data
                )
                if webcam_data
                is not None
                else None
            ),

            "model": str(
                WEBCAM_CALIBRATED_IMAGE_MODEL
            ),

            "metadata": str(
                WEBCAM_IMAGE_METADATA
            ),

            "clip_encoder": str(
                CLIP_MODEL_PATH
            ),

            "clip_embedding_dimension": (
                EXPECTED_CLIP_DIMENSION
            ),

            "original_image_artifact_preserved": (
                True
            ),
        },

        "methodological_note": (
            "The image component now incorporates both the original image "
            "classifier and a separately trained webcam-calibrated classifier. "
            "Webcam frames originating from the same source video are grouped "
            "during model evaluation to reduce frame-level leakage. "
            "The overall fusion dataset remains experimental because modality "
            "rows are stacked rather than guaranteed to correspond to the "
            "same multimodal user session."
        ),
    }

    with FUSION_META.open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            metadata,
            f,
            indent=4,
        )


# =============================================================================
# TRAINING REPORT
# =============================================================================

def save_training_report(
    best_name: str,
    best_metrics: dict[str, Any],
    candidate_results: dict[str, dict[str, Any]],
    df: pd.DataFrame,
    feature_columns: list[str],
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    split_method: str,
    visual_paths: dict[str, str],
) -> None:
    sorted_candidates = sorted(
        candidate_results.items(),
        key=lambda item: (
            item[1].get(
                "macro_f1",
                0.0,
            ),
            item[1].get(
                "accuracy",
                0.0,
            ),
        ),
        reverse=True,
    )

    with FUSION_REPORT.open(
        "w",
        encoding="utf-8",
    ) as f:
        f.write(
            "SenseFuzeAI Experimental Fusion "
            "Training and Selection Report\n"
        )

        f.write(
            "=" * 66
            + "\n\n"
        )

        f.write(
            "Created at: "
            f"{datetime.now().astimezone().isoformat()}\n"
        )

        f.write(
            f"Fusion dataset: {FUSION_DATASET_OUT}\n"
        )

        f.write(
            f"Model artifact: {FUSION_MODEL}\n"
        )

        f.write(
            f"Rows used: {len(df)}\n"
        )

        f.write(
            f"Train samples: {len(train_df)}\n"
        )

        f.write(
            f"Test samples: {len(test_df)}\n"
        )

        f.write(
            f"Split method: {split_method}\n"
        )

        f.write(
            "Selection rule: highest macro F1, "
            "then highest accuracy\n\n"
        )

        # ---------------------------------------------------------------------
        # Image calibration information
        # ---------------------------------------------------------------------

        f.write(
            "Image Pipeline Configuration\n"
        )

        f.write(
            "----------------------------\n"
        )

        f.write(
            "Original image model:\n"
            f"{ORIGINAL_IMAGE_MODEL}\n\n"
        )

        f.write(
            "Webcam-calibrated image model:\n"
            f"{WEBCAM_CALIBRATED_IMAGE_MODEL}\n\n"
        )

        f.write(
            "Frozen CLIP encoder:\n"
            f"{CLIP_MODEL_PATH}\n\n"
        )

        f.write(
            "Original image artifact overwritten: No\n\n"
        )

        # ---------------------------------------------------------------------
        # Dataset distributions
        # ---------------------------------------------------------------------

        f.write(
            "Rows by source modality:\n"
        )

        f.write(
            str(
                df[
                    "source_modality"
                ]
                .value_counts()
            )
        )

        f.write(
            "\n\n"
        )

        f.write(
            "Rows by source variant:\n"
        )

        f.write(
            str(
                df[
                    "source_variant"
                ]
                .value_counts()
            )
        )

        f.write(
            "\n\n"
        )

        f.write(
            "Class distribution:\n"
        )

        f.write(
            str(
                df[
                    "label"
                ]
                .value_counts()
                .reindex(
                    CLASSES,
                    fill_value=0,
                )
            )
        )

        f.write(
            "\n\n"
        )

        f.write(
            "Train class distribution:\n"
        )

        f.write(
            str(
                train_df[
                    "label"
                ]
                .value_counts()
                .reindex(
                    CLASSES,
                    fill_value=0,
                )
            )
        )

        f.write(
            "\n\n"
        )

        f.write(
            "Test class distribution:\n"
        )

        f.write(
            str(
                test_df[
                    "label"
                ]
                .value_counts()
                .reindex(
                    CLASSES,
                    fill_value=0,
                )
            )
        )

        f.write(
            "\n\n"
        )

        # ---------------------------------------------------------------------
        # Candidate comparison
        # ---------------------------------------------------------------------

        f.write(
            "Candidate Model Comparison\n"
        )

        f.write(
            "--------------------------\n"
        )

        f.write(
            f"{'Candidate':<28} "
            f"{'Status':<10} "
            f"{'Accuracy':>10} "
            f"{'Macro F1':>10} "
            f"{'Weighted F1':>12}\n"
        )

        for (
            name,
            metrics,
        ) in sorted_candidates:
            f.write(
                f"{name:<28} "
                f"{metrics.get('status', 'evaluated'):<10} "
                f"{metrics.get('accuracy', 0.0):>10.4f} "
                f"{metrics.get('macro_f1', 0.0):>10.4f} "
                f"{metrics.get('weighted_f1', 0.0):>12.4f}\n"
            )

        f.write(
            f"\nSelected model: {best_name}\n\n"
        )

        # ---------------------------------------------------------------------
        # Selected model
        # ---------------------------------------------------------------------

        f.write(
            "Selected Model Metrics\n"
        )

        f.write(
            "----------------------\n"
        )

        f.write(
            f"Accuracy: "
            f"{best_metrics['accuracy']:.4f}\n"
        )

        f.write(
            f"Macro Precision: "
            f"{best_metrics['macro_precision']:.4f}\n"
        )

        f.write(
            f"Macro Recall: "
            f"{best_metrics['macro_recall']:.4f}\n"
        )

        f.write(
            f"Macro F1: "
            f"{best_metrics['macro_f1']:.4f}\n"
        )

        f.write(
            f"Weighted Precision: "
            f"{best_metrics['weighted_precision']:.4f}\n"
        )

        f.write(
            f"Weighted Recall: "
            f"{best_metrics['weighted_recall']:.4f}\n"
        )

        f.write(
            f"Weighted F1: "
            f"{best_metrics['weighted_f1']:.4f}\n\n"
        )

        f.write(
            "Selected Model Classification Report\n"
        )

        f.write(
            "------------------------------------\n"
        )

        f.write(
            best_metrics[
                "classification_report_text"
            ]
        )

        f.write(
            "\nConfusion matrix labels:\n"
        )

        f.write(
            str(
                CLASSES
            )
        )

        f.write(
            "\n\nConfusion matrix:\n"
        )

        f.write(
            str(
                best_metrics[
                    "confusion_matrix"
                ]
            )
        )

        # ---------------------------------------------------------------------
        # Features
        # ---------------------------------------------------------------------

        f.write(
            "\n\nFusion Feature Columns\n"
        )

        f.write(
            "----------------------\n"
        )

        for feature in feature_columns:
            f.write(
                f"- {feature}\n"
            )

        # ---------------------------------------------------------------------
        # Visual reports
        # ---------------------------------------------------------------------

        f.write(
            "\nVisual Report Artefacts\n"
        )

        f.write(
            "-----------------------\n"
        )

        for (
            name,
            path,
        ) in visual_paths.items():
            f.write(
                f"{name}: {path}\n"
            )

        # ---------------------------------------------------------------------
        # Methodological note
        # ---------------------------------------------------------------------

        f.write(
            "\nMethodological Note\n"
        )

        f.write(
            "-------------------\n"
        )

        f.write(
            "The image branch now contains two independently preserved "
            "classifiers: the original static-image model and the "
            "webcam-calibrated image model. Both contribute image-class "
            "probability evidence to the experimental fusion dataset.\n\n"
        )

        f.write(
            "Frames extracted from the same webcam calibration source video "
            "share a common source_group. StratifiedGroupKFold therefore "
            "prevents those frames from appearing in both training and test "
            "partitions, reducing frame-level leakage.\n\n"
        )

        f.write(
            "The overall fusion experiment remains a score-level late-fusion "
            "experiment because its rows originate from separate modality "
            "datasets and are not necessarily observations from the same "
            "real user session. The session-aligned final fusion pipeline "
            "should remain the primary implementation for final behavioural "
            "inference."
        )


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    heading(
        "SenseFuzeAI Experimental Fusion Training"
    )

    print(
        f"Project root:\n{ROOT_DIR}"
    )

    # -------------------------------------------------------------------------
    # Build fusion dataset
    # -------------------------------------------------------------------------

    df = build_fusion_dataset()

    feature_columns = (
        probability_columns(
            "audio"
        )
        + probability_columns(
            "image"
        )
        + probability_columns(
            "keystroke"
        )
        + probability_columns(
            "text"
        )
        + [
            "has_audio",
            "has_image",
            "has_keystroke",
            "has_text",
        ]
    )

    df = (
        df
        .dropna(
            subset=[
                "label"
            ]
        )
        .copy()
    )

    validate_fusion_dataset(
        df
    )

    # -------------------------------------------------------------------------
    # Diagnostics
    # -------------------------------------------------------------------------

    heading(
        "FUSION DATASET SUMMARY"
    )

    print(
        f"Total fusion rows: "
        f"{len(df)}"
    )

    print()
    print(
        "Rows by modality:"
    )

    print(
        df[
            "source_modality"
        ]
        .value_counts()
    )

    print()
    print(
        "Rows by modality variant:"
    )

    print(
        df[
            "source_variant"
        ]
        .value_counts()
    )

    print()
    print(
        "Class distribution:"
    )

    print(
        df[
            "label"
        ]
        .value_counts()
        .reindex(
            CLASSES,
            fill_value=0,
        )
    )

    print()
    print(
        "Unique source groups:"
    )

    print(
        df[
            "source_group"
        ]
        .nunique()
    )

    # Save complete experimental fusion rows.
    df.to_csv(
        FUSION_DATASET_OUT,
        index=False,
    )

    # -------------------------------------------------------------------------
    # Leakage-aware split
    # -------------------------------------------------------------------------

    (
        X_train,
        X_test,
        y_train,
        y_test,
        train_df,
        test_df,
        split_method,
    ) = group_aware_train_test_split(
        df=df,
        feature_columns=feature_columns,
    )

    heading(
        "TRAIN / TEST SPLIT"
    )

    print(
        f"Method: {split_method}"
    )

    print(
        f"Training rows: "
        f"{len(train_df)}"
    )

    print(
        f"Testing rows : "
        f"{len(test_df)}"
    )

    train_groups = set(
        train_df[
            "source_group"
        ]
    )

    test_groups = set(
        test_df[
            "source_group"
        ]
    )

    overlap = (
        train_groups
        & test_groups
    )

    print(
        f"Source-group overlap: "
        f"{len(overlap)}"
    )

    print()
    print(
        "Training class distribution:"
    )

    print(
        y_train
        .value_counts()
        .reindex(
            CLASSES,
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
            CLASSES,
            fill_value=0,
        )
    )

    # -------------------------------------------------------------------------
    # Candidate training
    # -------------------------------------------------------------------------

    heading(
        "FUSION CANDIDATE MODEL EVALUATION"
    )

    (
        best_name,
        best_model,
        candidate_results,
    ) = evaluate_candidates(
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
    )

    best_metrics = (
        candidate_results[
            best_name
        ]
    )

    # -------------------------------------------------------------------------
    # Best model
    # -------------------------------------------------------------------------

    heading(
        "SELECTED FUSION MODEL"
    )

    print(
        f"Model       : {best_name}"
    )

    print(
        f"Accuracy    : "
        f"{best_metrics['accuracy']:.4f}"
    )

    print(
        f"Macro F1    : "
        f"{best_metrics['macro_f1']:.4f}"
    )

    print(
        f"Weighted F1 : "
        f"{best_metrics['weighted_f1']:.4f}"
    )

    # -------------------------------------------------------------------------
    # Visual reports
    # -------------------------------------------------------------------------

    heading(
        "GENERATING VISUAL REPORTS"
    )

    visual_paths = (
        generate_training_visuals(
            modality_name="fusion",
            candidate_results=candidate_results,
            best_metrics=best_metrics,
            y_all=df[
                "label"
            ],
            class_labels=CLASSES,
            output_root=VISUAL_REPORT_DIR,
        )
    )

    # -------------------------------------------------------------------------
    # Save selected model
    # -------------------------------------------------------------------------

    joblib.dump(
        best_model,
        FUSION_MODEL,
    )

    # -------------------------------------------------------------------------
    # Metadata
    # -------------------------------------------------------------------------

    save_metadata(
        best_name=best_name,
        best_metrics=best_metrics,
        candidate_results=candidate_results,
        df=df,
        feature_columns=feature_columns,
        train_df=train_df,
        test_df=test_df,
        split_method=split_method,
        visual_paths=visual_paths,
    )

    # -------------------------------------------------------------------------
    # Human-readable report
    # -------------------------------------------------------------------------

    save_training_report(
        best_name=best_name,
        best_metrics=best_metrics,
        candidate_results=candidate_results,
        df=df,
        feature_columns=feature_columns,
        train_df=train_df,
        test_df=test_df,
        split_method=split_method,
        visual_paths=visual_paths,
    )

    # -------------------------------------------------------------------------
    # Final output
    # -------------------------------------------------------------------------

    heading(
        "FUSION TRAINING COMPLETE"
    )

    print(
        "Selected model classification report:"
    )

    print()

    print(
        best_metrics[
            "classification_report_text"
        ]
    )

    print()
    print(
        "Saved model:"
    )

    print(
        FUSION_MODEL
    )

    print()
    print(
        "Saved metadata:"
    )

    print(
        FUSION_META
    )

    print()
    print(
        "Saved training report:"
    )

    print(
        FUSION_REPORT
    )

    print()
    print(
        "Saved fusion dataset:"
    )

    print(
        FUSION_DATASET_OUT
    )

    print()
    print(
        "Image-model configuration:"
    )

    print(
        f"  Original image model:\n"
        f"    {ORIGINAL_IMAGE_MODEL}"
    )

    print(
        f"  Webcam-calibrated model:\n"
        f"    {WEBCAM_CALIBRATED_IMAGE_MODEL}"
    )

    print(
        f"  Frozen CLIP encoder:\n"
        f"    {CLIP_MODEL_PATH}"
    )

    print()
    print(
        "Original image artifact preserved: YES"
    )

    print()
    print(
        "Visual reports:"
    )

    for (
        name,
        path,
    ) in visual_paths.items():
        print(
            f"  {name}: {path}"
        )


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    main()
