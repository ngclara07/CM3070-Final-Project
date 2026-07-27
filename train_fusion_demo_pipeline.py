# === train_fusion_demo_pipeline.py ===
#
# SenseFuzeAI
# Final Session-Aligned Multimodal Fusion Demo Training Pipeline
#
# =============================================================================
# PURPOSE
# =============================================================================
#
# Trains the final session-aligned multimodal fusion classifier using:
#
#   1. Keystroke dynamics features
#   2. MPNet text embeddings
#   3. Audio acoustic + WavLM features
#   4. CLIP image embeddings
#   5. Webcam-calibrated image classifier probabilities
#
# IMPORTANT:
#
# The webcam calibration dataset itself is NOT merged into the multimodal
# session dataset because the calibration videos are not necessarily the same
# sessions represented by:
#
#       data/processed/master_sessions_raw.csv
#
# Instead:
#
#       session-aligned CLIP image embedding
#                    ↓
#       webcam-calibrated image classifier
#                    ↓
#       four calibrated probability features
#
# These probability features are then appended to the existing image features
# before multimodal fusion.
#
# This preserves session alignment and avoids introducing unrelated webcam
# calibration rows into the final multimodal training dataset.
#
# =============================================================================
# INPUTS
# =============================================================================
#
# data/processed/master_sessions_raw.csv
# data/processed/text_features.csv
# data/processed/audio_features.csv
# data/processed/image_features.csv
#
# models/image_demo/image_pipeline_webcam_calibrated.joblib
# models/image_demo/feature_columns.json
#
# =============================================================================
# OUTPUTS
# =============================================================================
#
# models/fusion_demo/fusion_pipeline.joblib
# models/fusion_demo/feature_columns.json
# models/fusion_demo/model_selection_results.csv
# models/fusion_demo/metadata.json
# models/fusion_demo/fusion_training_dataset.csv
#
# =============================================================================

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import (
    StratifiedKFold,
    cross_validate,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC


# =============================================================================
# PROJECT PATHS
# =============================================================================

ROOT_DIR = Path(__file__).resolve().parent


# =============================================================================
# SESSION-ALIGNED FEATURE FILES
# =============================================================================

KEYSTROKE_PATH = (
    ROOT_DIR
    / "data"
    / "processed"
    / "master_sessions_raw.csv"
)

TEXT_PATH = (
    ROOT_DIR
    / "data"
    / "processed"
    / "text_features.csv"
)

AUDIO_PATH = (
    ROOT_DIR
    / "data"
    / "processed"
    / "audio_features.csv"
)

IMAGE_PATH = (
    ROOT_DIR
    / "data"
    / "processed"
    / "image_features.csv"
)


# =============================================================================
# WEBCAM-CALIBRATED IMAGE MODEL
# =============================================================================

WEBCAM_IMAGE_MODEL_PATH = (
    ROOT_DIR
    / "models"
    / "image_demo"
    / "image_pipeline_webcam_calibrated.joblib"
)

WEBCAM_IMAGE_FEATURE_COLUMNS_PATH = (
    ROOT_DIR
    / "models"
    / "image_demo"
    / "feature_columns.json"
)

WEBCAM_IMAGE_METADATA_PATH = (
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


# =============================================================================
# OUTPUT
# =============================================================================

OUTPUT_DIR = (
    ROOT_DIR
    / "models"
    / "fusion_demo"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

FUSION_MODEL_PATH = (
    OUTPUT_DIR
    / "fusion_pipeline.joblib"
)

FUSION_FEATURE_COLUMNS_PATH = (
    OUTPUT_DIR
    / "feature_columns.json"
)

MODEL_SELECTION_RESULTS_PATH = (
    OUTPUT_DIR
    / "model_selection_results.csv"
)

METADATA_PATH = (
    OUTPUT_DIR
    / "metadata.json"
)

FUSION_DATASET_PATH = (
    OUTPUT_DIR
    / "fusion_training_dataset.csv"
)


# =============================================================================
# CONFIGURATION
# =============================================================================

LABEL_COL = "label"
SESSION_COL = "session_id"

CLASSES = [
    "focused",
    "distracted",
    "fatigued",
    "overloaded",
]

RANDOM_STATE = 42
MAX_CV_SPLITS = 5


# =============================================================================
# KEYSTROKE NON-FEATURE COLUMNS
# =============================================================================

KEYSTROKE_NON_FEATURE_COLUMNS = {
    "session_id",
    "label",
    "created_at",
    "keydown_count_json",
    "event_count",
    "expected_event_count",
    "validation_passed",
    "validation_message",
    "text_path",
    "keystroke_path",
    "audio_path",
    "image_path",
    "text_exists",
    "audio_exists",
    "image_exists",
    "text",
    "problems",
    "is_clean",
}


# =============================================================================
# WEBCAM CALIBRATION OUTPUT FEATURE NAMES
# =============================================================================

WEBCAM_PROBABILITY_COLUMNS = [
    "image_webcam_focused_prob",
    "image_webcam_distracted_prob",
    "image_webcam_fatigued_prob",
    "image_webcam_overloaded_prob",
]

WEBCAM_CONFIDENCE_COLUMNS = [
    "image_webcam_top_probability",
    "image_webcam_confidence_gap",
]


# =============================================================================
# OPTIONAL LABEL-ENCODED CLASSIFIER WRAPPER
# =============================================================================

class LabelEncodedClassifier(
    BaseEstimator,
    ClassifierMixin,
):
    """
    Wrapper allowing classifiers such as XGBoost to train on encoded numeric
    targets while exposing original SenseFuzeAI class labels.
    """

    def __init__(
        self,
        classifier,
    ):
        self.classifier = classifier
        self.label_encoder = LabelEncoder()

    def fit(
        self,
        X,
        y,
    ):
        y_encoded = (
            self.label_encoder
            .fit_transform(y)
        )

        self.classifier.fit(
            X,
            y_encoded,
        )

        self.classes_ = (
            self.label_encoder
            .classes_
        )

        return self

    def predict(
        self,
        X,
    ):
        prediction = (
            self.classifier
            .predict(X)
        )

        prediction = np.asarray(
            prediction,
        ).astype(int)

        return (
            self.label_encoder
            .inverse_transform(
                prediction
            )
        )

    def predict_proba(
        self,
        X,
    ):
        return (
            self.classifier
            .predict_proba(X)
        )


# =============================================================================
# GENERAL UTILITIES
# =============================================================================

def print_heading(
    title: str,
) -> None:
    print()
    print("=" * 88)
    print(title)
    print("=" * 88)


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


def load_json(
    path: Path,
) -> Any:
    if not path.exists():
        return None

    with path.open(
        "r",
        encoding="utf-8",
    ) as f:
        return json.load(f)


def clean_numeric_dataframe(
    df: pd.DataFrame,
    columns: list[str],
) -> pd.DataFrame:
    output = df[
        columns
    ].copy()

    for column in output.columns:
        output[
            column
        ] = pd.to_numeric(
            output[column],
            errors="coerce",
        )

    output = output.replace(
        [
            np.inf,
            -np.inf,
        ],
        np.nan,
    )

    output = output.fillna(
        0.0
    )

    return output


def get_model_classes(
    model: Any,
) -> list[str]:
    """
    Obtain class ordering from a classifier or sklearn Pipeline.
    """

    classes = getattr(
        model,
        "classes_",
        None,
    )

    if classes is not None:
        return [
            normalise_label(
                label
            )
            for label in classes
        ]

    if hasattr(
        model,
        "named_steps",
    ):
        for step in reversed(
            list(
                model.named_steps.values()
            )
        ):
            classes = getattr(
                step,
                "classes_",
                None,
            )

            if classes is not None:
                return [
                    normalise_label(
                        label
                    )
                    for label in classes
                ]

    return []


# =============================================================================
# KEYSTROKE FEATURES
# =============================================================================

def load_keystroke_features() -> pd.DataFrame:
    print_heading(
        "LOADING KEYSTROKE FEATURES"
    )

    if not KEYSTROKE_PATH.exists():
        raise FileNotFoundError(
            f"Missing keystroke file:\n{KEYSTROKE_PATH}"
        )

    df = pd.read_csv(
        KEYSTROKE_PATH
    )

    if SESSION_COL not in df.columns:
        raise ValueError(
            f"Keystroke dataset missing '{SESSION_COL}'."
        )

    if LABEL_COL not in df.columns:
        raise ValueError(
            f"Keystroke dataset missing '{LABEL_COL}'."
        )

    # -------------------------------------------------------------------------
    # Clean-session filtering
    # -------------------------------------------------------------------------

    if "is_clean" in df.columns:
        before_count = len(
            df
        )

        clean_values = (
            df["is_clean"]
            .astype(str)
            .str.strip()
            .str.lower()
        )

        df = df[
            clean_values.isin(
                [
                    "true",
                    "1",
                    "yes",
                ]
            )
        ].copy()

        after_count = len(
            df
        )

        print(
            f"Clean-session filtering: "
            f"{before_count} -> {after_count}"
        )

    # -------------------------------------------------------------------------
    # Labels
    # -------------------------------------------------------------------------

    df[
        LABEL_COL
    ] = (
        df[
            LABEL_COL
        ]
        .apply(
            normalise_label
        )
    )

    df = df[
        df[
            LABEL_COL
        ]
        .isin(
            CLASSES
        )
    ].copy()

    # -------------------------------------------------------------------------
    # Numeric feature selection
    # -------------------------------------------------------------------------

    feature_columns = [
        column
        for column
        in df.columns
        if (
            column
            not in KEYSTROKE_NON_FEATURE_COLUMNS
            and pd.api.types.is_numeric_dtype(
                df[column]
            )
        )
    ]

    if not feature_columns:
        raise ValueError(
            "No numeric keystroke features found."
        )

    output = df[
        [
            SESSION_COL,
            LABEL_COL,
        ]
        + feature_columns
    ].copy()

    # Prefix feature names so collisions cannot occur.
    rename_map = {
        column: (
            column
            if column.startswith(
                "keystroke_"
            )
            else f"keystroke_{column}"
        )
        for column
        in feature_columns
    }

    output = output.rename(
        columns=rename_map
    )

    print(
        f"Keystroke sessions: "
        f"{len(output)}"
    )

    print(
        f"Keystroke features: "
        f"{len(feature_columns)}"
    )

    return output


# =============================================================================
# GENERIC MODALITY FEATURE LOADER
# =============================================================================

def load_modality_features(
    path: Path,
    modality_name: str,
) -> pd.DataFrame:
    print_heading(
        f"LOADING {modality_name.upper()} FEATURES"
    )

    if not path.exists():
        raise FileNotFoundError(
            f"Missing {modality_name} feature file:\n"
            f"{path}"
        )

    df = pd.read_csv(
        path
    )

    if SESSION_COL not in df.columns:
        raise ValueError(
            f"{path} must contain '{SESSION_COL}'."
        )

    if LABEL_COL not in df.columns:
        raise ValueError(
            f"{path} must contain '{LABEL_COL}'."
        )

    df[
        LABEL_COL
    ] = (
        df[
            LABEL_COL
        ]
        .apply(
            normalise_label
        )
    )

    df = df[
        df[
            LABEL_COL
        ]
        .isin(
            CLASSES
        )
    ].copy()

    # -------------------------------------------------------------------------
    # Retain only numeric feature columns
    # -------------------------------------------------------------------------

    feature_columns = [
        column
        for column
        in df.columns
        if (
            column
            not in {
                SESSION_COL,
                LABEL_COL,
            }
            and pd.api.types.is_numeric_dtype(
                df[column]
            )
        )
    ]

    if not feature_columns:
        raise ValueError(
            f"No numeric {modality_name} features found."
        )

    output = df[
        [
            SESSION_COL,
            LABEL_COL,
        ]
        + feature_columns
    ].copy()

    print(
        f"{modality_name.capitalize()} sessions: "
        f"{len(output)}"
    )

    print(
        f"{modality_name.capitalize()} features: "
        f"{len(feature_columns)}"
    )

    return output


# =============================================================================
# WEBCAM-CALIBRATED IMAGE FEATURES
# =============================================================================

def add_webcam_calibrated_image_features(
    image_df: pd.DataFrame,
) -> tuple[
    pd.DataFrame,
    dict[str, Any],
]:
    """
    Run the webcam-calibrated image classifier against each session-aligned
    CLIP embedding.

    The calibration video rows themselves are never added to the session
    dataset.

    Instead the calibrated classifier produces probability features from the
    already session-aligned image embeddings.
    """

    print_heading(
        "ADDING WEBCAM-CALIBRATED IMAGE FEATURES"
    )

    status = {
        "enabled": False,
        "model_path": str(
            WEBCAM_IMAGE_MODEL_PATH
        ),
        "reason": None,
        "probability_columns": [],
    }

    # -------------------------------------------------------------------------
    # Artifact checks
    # -------------------------------------------------------------------------

    if not WEBCAM_IMAGE_MODEL_PATH.exists():
        status[
            "reason"
        ] = (
            "Webcam-calibrated image model not found."
        )

        print(
            "WARNING:"
        )

        print(
            status[
                "reason"
            ]
        )

        print(
            WEBCAM_IMAGE_MODEL_PATH
        )

        return (
            image_df,
            status,
        )

    if not WEBCAM_IMAGE_FEATURE_COLUMNS_PATH.exists():
        status[
            "reason"
        ] = (
            "Webcam image feature schema not found."
        )

        print(
            "WARNING:"
        )

        print(
            status[
                "reason"
            ]
        )

        return (
            image_df,
            status,
        )

    # -------------------------------------------------------------------------
    # Load calibrated feature schema
    # -------------------------------------------------------------------------

    webcam_feature_columns = load_json(
        WEBCAM_IMAGE_FEATURE_COLUMNS_PATH
    )

    if not isinstance(
        webcam_feature_columns,
        list,
    ):
        status[
            "reason"
        ] = (
            "Webcam image feature schema is not a list."
        )

        print(
            "WARNING:"
        )

        print(
            status[
                "reason"
            ]
        )

        return (
            image_df,
            status,
        )

    webcam_feature_columns = [
        str(
            column
        )
        for column
        in webcam_feature_columns
    ]

    if not webcam_feature_columns:
        status[
            "reason"
        ] = (
            "Webcam image feature schema is empty."
        )

        print(
            "WARNING:"
        )

        print(
            status[
                "reason"
            ]
        )

        return (
            image_df,
            status,
        )

    # -------------------------------------------------------------------------
    # Verify session image feature compatibility
    # -------------------------------------------------------------------------

    missing_columns = [
        column
        for column
        in webcam_feature_columns
        if column
        not in image_df.columns
    ]

    if missing_columns:
        status[
            "reason"
        ] = (
            "Session-aligned image feature file does not contain "
            "all CLIP features required by the calibrated model."
        )

        print(
            "WARNING:"
        )

        print(
            status[
                "reason"
            ]
        )

        print()
        print(
            "Missing examples:"
        )

        print(
            missing_columns[
                :20
            ]
        )

        print()
        print(
            "Fusion training will continue using the original "
            "session-aligned image features only."
        )

        return (
            image_df,
            status,
        )

    # -------------------------------------------------------------------------
    # Prepare model input
    # -------------------------------------------------------------------------

    X_image = clean_numeric_dataframe(
        image_df,
        webcam_feature_columns,
    )

    calibrated_model = joblib.load(
        WEBCAM_IMAGE_MODEL_PATH
    )

    model_classes = get_model_classes(
        calibrated_model
    )

    print(
        f"Calibrated model classes: "
        f"{model_classes}"
    )

    # -------------------------------------------------------------------------
    # Probabilities
    # -------------------------------------------------------------------------

    if not hasattr(
        calibrated_model,
        "predict_proba",
    ):
        status[
            "reason"
        ] = (
            "Calibrated image model does not expose predict_proba()."
        )

        print(
            "WARNING:"
        )

        print(
            status[
                "reason"
            ]
        )

        return (
            image_df,
            status,
        )

    probabilities = (
        calibrated_model
        .predict_proba(
            X_image
        )
    )

    probabilities = np.asarray(
        probabilities,
        dtype=float,
    )

    # -------------------------------------------------------------------------
    # Initialise all four SenseFuzeAI classes
    # -------------------------------------------------------------------------

    probability_lookup = {
        label: np.zeros(
            len(
                image_df
            ),
            dtype=float,
        )
        for label in CLASSES
    }

    for class_index, class_name in enumerate(
        model_classes
    ):
        if (
            class_name in CLASSES
            and class_index
            < probabilities.shape[1]
        ):
            probability_lookup[
                class_name
            ] = probabilities[
                :,
                class_index,
            ]

    output = image_df.copy()

    for label in CLASSES:
        output[
            f"image_webcam_{label}_prob"
        ] = (
            probability_lookup[
                label
            ]
        )

    # -------------------------------------------------------------------------
    # Useful calibration-confidence features
    # -------------------------------------------------------------------------

    probability_matrix = np.column_stack(
        [
            output[
                f"image_webcam_{label}_prob"
            ].to_numpy(
                dtype=float
            )
            for label
            in CLASSES
        ]
    )

    sorted_probabilities = np.sort(
        probability_matrix,
        axis=1,
    )

    output[
        "image_webcam_top_probability"
    ] = (
        sorted_probabilities[
            :,
            -1,
        ]
    )

    if (
        sorted_probabilities.shape[1]
        >= 2
    ):
        output[
            "image_webcam_confidence_gap"
        ] = (
            sorted_probabilities[
                :,
                -1,
            ]
            - sorted_probabilities[
                :,
                -2,
            ]
        )

    else:
        output[
            "image_webcam_confidence_gap"
        ] = 0.0

    status.update(
        {
            "enabled": True,
            "reason": None,
            "model_classes": model_classes,
            "input_feature_count": int(
                len(
                    webcam_feature_columns
                )
            ),
            "probability_columns": (
                WEBCAM_PROBABILITY_COLUMNS
                + WEBCAM_CONFIDENCE_COLUMNS
            ),
        }
    )

    print(
        "Webcam-calibrated image augmentation: ENABLED"
    )

    print(
        f"Input CLIP features: "
        f"{len(webcam_feature_columns)}"
    )

    print(
        f"Added fusion features: "
        f"{len(WEBCAM_PROBABILITY_COLUMNS + WEBCAM_CONFIDENCE_COLUMNS)}"
    )

    print()
    print(
        "Mean calibrated probabilities:"
    )

    for label in CLASSES:
        column = (
            f"image_webcam_{label}_prob"
        )

        print(
            f"  {label:<12}: "
            f"{output[column].mean():.4f}"
        )

    return (
        output,
        status,
    )


# =============================================================================
# SESSION VALIDATION
# =============================================================================

def validate_unique_sessions(
    df: pd.DataFrame,
    name: str,
) -> None:
    duplicates = df[
        SESSION_COL
    ].duplicated(
        keep=False
    )

    if duplicates.any():
        duplicated_ids = (
            df.loc[
                duplicates,
                SESSION_COL,
            ]
            .astype(str)
            .unique()
        )

        raise ValueError(
            f"{name} contains duplicate session_id rows. "
            f"Examples: {duplicated_ids[:10].tolist()}"
        )


def print_session_overlap(
    reference_df: pd.DataFrame,
    other_df: pd.DataFrame,
    modality_name: str,
) -> None:
    reference_sessions = set(
        reference_df[
            SESSION_COL
        ].astype(str)
    )

    other_sessions = set(
        other_df[
            SESSION_COL
        ].astype(str)
    )

    overlap = (
        reference_sessions
        & other_sessions
    )

    print(
        f"{modality_name.capitalize()} session overlap: "
        f"{len(overlap)} / {len(reference_sessions)}"
    )


# =============================================================================
# FUSION DATASET CONSTRUCTION
# =============================================================================

def build_fusion_dataset() -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.Series,
    list[str],
    dict[str, Any],
]:
    print_heading(
        "BUILDING SESSION-ALIGNED MULTIMODAL FUSION DATASET"
    )

    # -------------------------------------------------------------------------
    # Load all session-aligned modalities
    # -------------------------------------------------------------------------

    key_df = load_keystroke_features()

    text_df = load_modality_features(
        TEXT_PATH,
        "text",
    )

    audio_df = load_modality_features(
        AUDIO_PATH,
        "audio",
    )

    image_df = load_modality_features(
        IMAGE_PATH,
        "image",
    )

    # -------------------------------------------------------------------------
    # Validate one row per session per modality
    # -------------------------------------------------------------------------

    validate_unique_sessions(
        key_df,
        "Keystroke dataset",
    )

    validate_unique_sessions(
        text_df,
        "Text feature dataset",
    )

    validate_unique_sessions(
        audio_df,
        "Audio feature dataset",
    )

    validate_unique_sessions(
        image_df,
        "Image feature dataset",
    )

    # -------------------------------------------------------------------------
    # Report alignment
    # -------------------------------------------------------------------------

    print_heading(
        "SESSION ALIGNMENT CHECK"
    )

    print(
        f"Reference keystroke sessions: "
        f"{len(key_df)}"
    )

    print_session_overlap(
        key_df,
        text_df,
        "text",
    )

    print_session_overlap(
        key_df,
        audio_df,
        "audio",
    )

    print_session_overlap(
        key_df,
        image_df,
        "image",
    )

    # -------------------------------------------------------------------------
    # Apply webcam calibration model to existing session image embeddings
    # -------------------------------------------------------------------------

    (
        image_df,
        webcam_status,
    ) = add_webcam_calibrated_image_features(
        image_df
    )

    # -------------------------------------------------------------------------
    # Merge by BOTH session_id and label
    # -------------------------------------------------------------------------

    fusion_df = (
        key_df
        .merge(
            text_df,
            on=[
                SESSION_COL,
                LABEL_COL,
            ],
            how="inner",
            validate="one_to_one",
        )
    )

    fusion_df = (
        fusion_df
        .merge(
            audio_df,
            on=[
                SESSION_COL,
                LABEL_COL,
            ],
            how="inner",
            validate="one_to_one",
        )
    )

    fusion_df = (
        fusion_df
        .merge(
            image_df,
            on=[
                SESSION_COL,
                LABEL_COL,
            ],
            how="inner",
            validate="one_to_one",
        )
    )

    if fusion_df.empty:
        raise ValueError(
            "Fusion dataframe is empty after session-aligned "
            "modality merging."
        )

    # -------------------------------------------------------------------------
    # Ensure one fused row per session
    # -------------------------------------------------------------------------

    validate_unique_sessions(
        fusion_df,
        "Final fusion dataset",
    )

    # -------------------------------------------------------------------------
    # Label normalisation
    # -------------------------------------------------------------------------

    fusion_df[
        LABEL_COL
    ] = (
        fusion_df[
            LABEL_COL
        ]
        .apply(
            normalise_label
        )
    )

    fusion_df = fusion_df[
        fusion_df[
            LABEL_COL
        ]
        .isin(
            CLASSES
        )
    ].copy()

    # -------------------------------------------------------------------------
    # Numeric fusion features
    # -------------------------------------------------------------------------

    feature_columns = [
        column
        for column
        in fusion_df.columns
        if (
            column
            not in {
                SESSION_COL,
                LABEL_COL,
            }
            and pd.api.types.is_numeric_dtype(
                fusion_df[column]
            )
        )
    ]

    if not feature_columns:
        raise ValueError(
            "No numeric multimodal fusion features found."
        )

    # -------------------------------------------------------------------------
    # Clean final feature matrix
    # -------------------------------------------------------------------------

    X = clean_numeric_dataframe(
        fusion_df,
        feature_columns,
    )

    y = (
        fusion_df[
            LABEL_COL
        ]
        .copy()
    )

    return (
        fusion_df,
        X,
        y,
        feature_columns,
        webcam_status,
    )


# =============================================================================
# MODEL DEFINITIONS
# =============================================================================

def build_models() -> dict[str, Pipeline]:
    models: dict[
        str,
        Pipeline,
    ] = {
        "random_forest": Pipeline(
            [
                (
                    "classifier",
                    RandomForestClassifier(
                        n_estimators=500,
                        class_weight="balanced",
                        random_state=RANDOM_STATE,
                        n_jobs=-1,
                    ),
                ),
            ]
        ),

        "svm_rbf": Pipeline(
            [
                (
                    "scaler",
                    StandardScaler(),
                ),
                (
                    "classifier",
                    SVC(
                        kernel="rbf",
                        class_weight="balanced",
                        probability=True,
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
    }

    # -------------------------------------------------------------------------
    # XGBoost
    # -------------------------------------------------------------------------

    try:
        from xgboost import XGBClassifier

        models[
            "xgboost"
        ] = Pipeline(
            [
                (
                    "scaler",
                    StandardScaler(),
                ),
                (
                    "classifier",
                    LabelEncodedClassifier(
                        XGBClassifier(
                            n_estimators=500,
                            learning_rate=0.03,
                            max_depth=4,
                            subsample=0.9,
                            colsample_bytree=0.9,
                            objective="multi:softprob",
                            eval_metric="mlogloss",
                            random_state=RANDOM_STATE,
                        )
                    ),
                ),
            ]
        )

    except ImportError:
        print(
            "XGBoost not installed. "
            "Skipping xgboost."
        )

    # -------------------------------------------------------------------------
    # LightGBM
    # -------------------------------------------------------------------------

    try:
        from lightgbm import LGBMClassifier

        models[
            "lightgbm"
        ] = Pipeline(
            [
                (
                    "classifier",
                    LGBMClassifier(
                        n_estimators=500,
                        learning_rate=0.03,
                        class_weight="balanced",
                        random_state=RANDOM_STATE,
                        verbose=-1,
                    ),
                ),
            ]
        )

    except ImportError:
        print(
            "LightGBM not installed. "
            "Skipping lightgbm."
        )

    # -------------------------------------------------------------------------
    # CatBoost
    # -------------------------------------------------------------------------

    try:
        from catboost import CatBoostClassifier

        models[
            "catboost"
        ] = Pipeline(
            [
                (
                    "classifier",
                    CatBoostClassifier(
                        iterations=500,
                        learning_rate=0.03,
                        depth=5,
                        loss_function="MultiClass",
                        auto_class_weights="Balanced",
                        random_seed=RANDOM_STATE,
                        verbose=False,
                    ),
                ),
            ]
        )

    except ImportError:
        print(
            "CatBoost not installed. "
            "Skipping catboost."
        )

    return models


# =============================================================================
# CROSS-VALIDATION CONFIGURATION
# =============================================================================

def determine_cv_splits(
    y: pd.Series,
) -> int:
    counts = (
        y.value_counts()
        .reindex(
            CLASSES,
            fill_value=0,
        )
    )

    non_zero_counts = (
        counts[
            counts > 0
        ]
    )

    if len(
        non_zero_counts
    ) < 2:
        raise ValueError(
            "At least two behavioural classes are required."
        )

    minimum_class_count = int(
        non_zero_counts.min()
    )

    if minimum_class_count < 2:
        raise ValueError(
            "At least two samples per represented class are required "
            "for stratified cross-validation."
        )

    return min(
        MAX_CV_SPLITS,
        minimum_class_count,
    )


# =============================================================================
# MODEL SELECTION
# =============================================================================

def select_best_model(
    models: dict[str, Pipeline],
    X: pd.DataFrame,
    y: pd.Series,
) -> tuple[
    str,
    Pipeline,
    pd.DataFrame,
]:
    cv_splits = determine_cv_splits(
        y
    )

    print(
        f"\nUsing {cv_splits}-fold "
        "stratified cross-validation."
    )

    cv = StratifiedKFold(
        n_splits=cv_splits,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    rows: list[
        dict[str, Any]
    ] = []

    scoring = {
        "accuracy": "accuracy",
        "macro_f1": "f1_macro",
    }

    for (
        name,
        pipeline,
    ) in models.items():
        print()
        print(
            f"Cross-validating: {name}"
        )

        try:
            scores = cross_validate(
                pipeline,
                X,
                y,
                cv=cv,
                scoring=scoring,
                return_train_score=False,
                error_score="raise",
            )

            accuracy_scores = (
                scores[
                    "test_accuracy"
                ]
            )

            f1_scores = (
                scores[
                    "test_macro_f1"
                ]
            )

            row = {
                "model": name,

                "status": "evaluated",

                "cv_accuracy_mean": float(
                    accuracy_scores.mean()
                ),

                "cv_accuracy_std": float(
                    accuracy_scores.std()
                ),

                "cv_macro_f1_mean": float(
                    f1_scores.mean()
                ),

                "cv_macro_f1_std": float(
                    f1_scores.std()
                ),
            }

            rows.append(
                row
            )

            print(
                f"  Accuracy : "
                f"{row['cv_accuracy_mean']:.4f} "
                f"+/- {row['cv_accuracy_std']:.4f}"
            )

            print(
                f"  Macro F1 : "
                f"{row['cv_macro_f1_mean']:.4f} "
                f"+/- {row['cv_macro_f1_std']:.4f}"
            )

        except Exception as exc:
            print(
                f"  FAILED: {exc}"
            )

            rows.append(
                {
                    "model": name,
                    "status": "error",
                    "error": str(
                        exc
                    ),
                    "cv_accuracy_mean": 0.0,
                    "cv_accuracy_std": 0.0,
                    "cv_macro_f1_mean": 0.0,
                    "cv_macro_f1_std": 0.0,
                }
            )

    results_df = pd.DataFrame(
        rows
    )

    valid_results = results_df[
        results_df[
            "status"
        ]
        == "evaluated"
    ].copy()

    if valid_results.empty:
        raise RuntimeError(
            "No candidate fusion model completed "
            "cross-validation successfully."
        )

    # Macro-F1 first, accuracy as tie-break.
    valid_results = (
        valid_results
        .sort_values(
            by=[
                "cv_macro_f1_mean",
                "cv_accuracy_mean",
            ],
            ascending=[
                False,
                False,
            ],
        )
    )

    best_model_name = str(
        valid_results
        .iloc[0][
            "model"
        ]
    )

    best_pipeline = models[
        best_model_name
    ]

    results_df = (
        results_df
        .sort_values(
            by=[
                "cv_macro_f1_mean",
                "cv_accuracy_mean",
            ],
            ascending=[
                False,
                False,
            ],
        )
        .reset_index(
            drop=True
        )
    )

    return (
        best_model_name,
        best_pipeline,
        results_df,
    )


# =============================================================================
# TRAINING-SET DIAGNOSTICS
# =============================================================================

def evaluate_final_training_fit(
    model: Pipeline,
    X: pd.DataFrame,
    y: pd.Series,
) -> dict[str, Any]:
    """
    This is explicitly a training-set diagnostic, NOT held-out performance.
    """

    prediction = model.predict(
        X
    )

    accuracy = accuracy_score(
        y,
        prediction,
    )

    macro_f1 = f1_score(
        y,
        prediction,
        labels=CLASSES,
        average="macro",
        zero_division=0,
    )

    report = classification_report(
        y,
        prediction,
        labels=CLASSES,
        zero_division=0,
    )

    matrix = confusion_matrix(
        y,
        prediction,
        labels=CLASSES,
    )

    return {
        "training_accuracy": float(
            accuracy
        ),

        "training_macro_f1": float(
            macro_f1
        ),

        "classification_report": report,

        "confusion_matrix": (
            matrix.tolist()
        ),
    }


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    print_heading(
        "SenseFuzeAI Session-Aligned Fusion Demo Training"
    )

    print(
        f"Project root:\n  {ROOT_DIR}"
    )

    # -------------------------------------------------------------------------
    # Build session-aligned fusion dataset
    # -------------------------------------------------------------------------

    (
        fusion_df,
        X,
        y,
        feature_columns,
        webcam_status,
    ) = build_fusion_dataset()

    print_heading(
        "FINAL FUSION DATASET"
    )

    print(
        f"Aligned sessions : "
        f"{len(fusion_df)}"
    )

    print(
        f"Fusion features  : "
        f"{len(feature_columns)}"
    )

    print()
    print(
        "Behavioural class distribution:"
    )

    print(
        y.value_counts()
        .reindex(
            CLASSES,
            fill_value=0,
        )
    )

    print()
    print(
        "Webcam calibration augmentation:"
    )

    if webcam_status[
        "enabled"
    ]:
        print(
            "  ENABLED"
        )

        print(
            f"  Added features: "
            f"{webcam_status['probability_columns']}"
        )

    else:
        print(
            "  DISABLED"
        )

        print(
            f"  Reason: "
            f"{webcam_status['reason']}"
        )

    # -------------------------------------------------------------------------
    # Save the exact session-aligned training dataset
    # -------------------------------------------------------------------------

    fusion_df.to_csv(
        FUSION_DATASET_PATH,
        index=False,
    )

    # -------------------------------------------------------------------------
    # Candidate model construction
    # -------------------------------------------------------------------------

    models = build_models()

    if not models:
        raise RuntimeError(
            "No candidate fusion models available."
        )

    # -------------------------------------------------------------------------
    # Cross-validation model selection
    # -------------------------------------------------------------------------

    print_heading(
        "FUSION MODEL SELECTION"
    )

    (
        best_model_name,
        best_pipeline,
        cv_results,
    ) = select_best_model(
        models=models,
        X=X,
        y=y,
    )

    cv_results.to_csv(
        MODEL_SELECTION_RESULTS_PATH,
        index=False,
    )

    print()
    print(
        "Fusion model selection results:"
    )

    print(
        cv_results.to_string(
            index=False
        )
    )

    print()
    print(
        f"Selected fusion model: "
        f"{best_model_name}"
    )

    # -------------------------------------------------------------------------
    # Fit selected model on ALL session-aligned data
    # -------------------------------------------------------------------------

    print_heading(
        "TRAINING FINAL FUSION MODEL"
    )

    best_pipeline.fit(
        X,
        y,
    )

    final_fit = (
        evaluate_final_training_fit(
            best_pipeline,
            X,
            y,
        )
    )

    print(
        f"Training-fit accuracy : "
        f"{final_fit['training_accuracy']:.4f}"
    )

    print(
        f"Training-fit macro F1 : "
        f"{final_fit['training_macro_f1']:.4f}"
    )

    print()
    print(
        "NOTE: these are training-fit diagnostics. "
        "Use the cross-validation scores above as the "
        "generalisation estimate."
    )

    # -------------------------------------------------------------------------
    # Save fusion model
    # -------------------------------------------------------------------------

    joblib.dump(
        best_pipeline,
        FUSION_MODEL_PATH,
    )

    # -------------------------------------------------------------------------
    # Save feature schema
    # -------------------------------------------------------------------------

    with FUSION_FEATURE_COLUMNS_PATH.open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            feature_columns,
            f,
            indent=4,
        )

    # -------------------------------------------------------------------------
    # Metadata
    # -------------------------------------------------------------------------

    selected_result = (
        cv_results[
            cv_results[
                "model"
            ]
            == best_model_name
        ]
        .iloc[0]
    )

    metadata = {
        "created_at": (
            datetime.now()
            .astimezone()
            .isoformat()
        ),

        "pipeline_type": (
            "session_aligned_multimodal_fusion"
        ),

        "num_samples": int(
            len(
                fusion_df
            )
        ),

        "num_features": int(
            len(
                feature_columns
            )
        ),

        "classes": (
            CLASSES
        ),

        "modalities": [
            "keystroke",
            "text",
            "audio",
            "image",
        ],

        "session_column": (
            SESSION_COL
        ),

        "label_column": (
            LABEL_COL
        ),

        "session_alignment": (
            True
        ),

        "feature_columns": (
            feature_columns
        ),

        "candidate_models": (
            list(
                models.keys()
            )
        ),

        "selected_model": (
            best_model_name
        ),

        "selection_metric": (
            "stratified_cross_validation_macro_f1_"
            "with_accuracy_tiebreak"
        ),

        "cv_accuracy_mean": float(
            selected_result[
                "cv_accuracy_mean"
            ]
        ),

        "cv_accuracy_std": float(
            selected_result[
                "cv_accuracy_std"
            ]
        ),

        "cv_macro_f1_mean": float(
            selected_result[
                "cv_macro_f1_mean"
            ]
        ),

        "cv_macro_f1_std": float(
            selected_result[
                "cv_macro_f1_std"
            ]
        ),

        "training_fit_accuracy": (
            final_fit[
                "training_accuracy"
            ]
        ),

        "training_fit_macro_f1": (
            final_fit[
                "training_macro_f1"
            ]
        ),

        "model_artifact": str(
            FUSION_MODEL_PATH
        ),

        "feature_schema": str(
            FUSION_FEATURE_COLUMNS_PATH
        ),

        "training_dataset": str(
            FUSION_DATASET_PATH
        ),

        "source_files": {
            "keystroke": str(
                KEYSTROKE_PATH
            ),

            "text": str(
                TEXT_PATH
            ),

            "audio": str(
                AUDIO_PATH
            ),

            "image": str(
                IMAGE_PATH
            ),
        },

        "webcam_image_calibration": {
            "enabled": bool(
                webcam_status[
                    "enabled"
                ]
            ),

            "calibrated_model": str(
                WEBCAM_IMAGE_MODEL_PATH
            ),

            "calibrated_model_metadata": str(
                WEBCAM_IMAGE_METADATA_PATH
            ),

            "calibrated_feature_schema": str(
                WEBCAM_IMAGE_FEATURE_COLUMNS_PATH
            ),

            "pretrained_visual_encoder": str(
                CLIP_MODEL_PATH
            ),

            "calibration_features_added": (
                webcam_status.get(
                    "probability_columns",
                    [],
                )
            ),

            "reason_if_disabled": (
                webcam_status.get(
                    "reason"
                )
            ),

            "original_image_artifact_preserved": True,

            "calibration_dataset_directly_merged_into_fusion": False,
        },

        "class_distribution": (
            y.value_counts()
            .reindex(
                CLASSES,
                fill_value=0,
            )
            .to_dict()
        ),

        "methodological_note": (
            "The final fusion demo model uses only session-aligned "
            "multimodal rows. Webcam calibration videos are not inserted "
            "as independent fusion observations. Instead, the separately "
            "trained webcam-calibrated image classifier is applied to the "
            "CLIP embedding belonging to each aligned session image. Its "
            "four behavioural-state probabilities and confidence statistics "
            "are appended as image-derived fusion features. This preserves "
            "the original multimodal session alignment while allowing the "
            "live webcam domain calibration to influence final fusion."
        ),
    }

    with METADATA_PATH.open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            metadata,
            f,
            indent=4,
        )

    # -------------------------------------------------------------------------
    # Final summary
    # -------------------------------------------------------------------------

    print_heading(
        "FUSION DEMO PIPELINE COMPLETE"
    )

    print(
        f"Selected model:\n"
        f"  {best_model_name}"
    )

    print()
    print(
        "Cross-validation performance:"
    )

    print(
        f"  Accuracy : "
        f"{float(selected_result['cv_accuracy_mean']):.4f} "
        f"+/- "
        f"{float(selected_result['cv_accuracy_std']):.4f}"
    )

    print(
        f"  Macro F1 : "
        f"{float(selected_result['cv_macro_f1_mean']):.4f} "
        f"+/- "
        f"{float(selected_result['cv_macro_f1_std']):.4f}"
    )

    print()
    print(
        "Webcam calibration:"
    )

    print(
        "  ENABLED"
        if webcam_status[
            "enabled"
        ]
        else "  DISABLED"
    )

    if webcam_status[
        "enabled"
    ]:
        print(
            "  Original image embeddings retained."
        )

        print(
            "  Original image artifact preserved."
        )

        print(
            "  Calibrated webcam probability "
            "features added to fusion."
        )

    print()
    print(
        "Saved artifacts:"
    )

    print(
        f"  Model:\n"
        f"    {FUSION_MODEL_PATH}"
    )

    print(
        f"  Feature schema:\n"
        f"    {FUSION_FEATURE_COLUMNS_PATH}"
    )

    print(
        f"  Metadata:\n"
        f"    {METADATA_PATH}"
    )

    print(
        f"  CV results:\n"
        f"    {MODEL_SELECTION_RESULTS_PATH}"
    )

    print(
        f"  Training dataset:\n"
        f"    {FUSION_DATASET_PATH}"
    )


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    main()
