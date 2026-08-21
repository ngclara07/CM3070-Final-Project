"""
evaluate_missing_modality_robustness.py

SenseFuzeAI
Inference-Time Single-Modality-Loss Robustness Experiment
Final Report, Chapter 5, Sub-section 5.6.2

This experiment is methodologically distinct from leave-one-modality-out
ablation.

For every StratifiedKFold split:

    1. Train the complete 2,373-feature fusion model once.
    2. Evaluate the untouched held-out fold.
    3. Keep that trained model fixed.
    4. Zero-mask one modality in the held-out feature matrix.
    5. Re-evaluate without retraining.

The experiment therefore measures classifier-level predictive sensitivity
to synthetic modality loss.

IMPORTANT:
The deployed SenseFuzeAI runtime requires the complete multimodal feature
contract. Zero masking in this script is an experimental robustness stress
condition; it is not a claim that the production application supports
operation with physically absent modalities.

Outputs:

    data/processed/final_experiments/missing_modality_robustness/
        missing_modality_results.csv
        missing_modality_fold_results.csv
        missing_modality_metadata.json
        evaluate_missing_modality_robustness.py

Run:

    python -m py_compile evaluate_missing_modality_robustness.py
    python evaluate_missing_modality_robustness.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import shutil
import sys

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import joblib
import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline


# =============================================================================
# PROJECT PATHS
# =============================================================================

ROOT_DIR = Path(__file__).resolve().parent

FUSION_DIR = (
    ROOT_DIR
    / "models"
    / "fusion_demo"
)

DATASET_PATH = (
    FUSION_DIR
    / "fusion_training_dataset.csv"
)

FEATURE_SCHEMA_PATH = (
    FUSION_DIR
    / "feature_columns.json"
)

METADATA_PATH = (
    FUSION_DIR
    / "metadata.json"
)

DEPLOYED_MODEL_PATH = (
    FUSION_DIR
    / "fusion_pipeline.joblib"
)

FINAL_INFERENCE_PATH = (
    ROOT_DIR
    / "final_multimodal_inference.py"
)

ABLATION_METADATA_PATH = (
    ROOT_DIR
    / "data"
    / "processed"
    / "final_experiments"
    / "leave_one_modality_out"
    / "ablation_metadata.json"
)

DEFAULT_OUTPUT_DIR = (
    ROOT_DIR
    / "data"
    / "processed"
    / "final_experiments"
    / "missing_modality_robustness"
)


# =============================================================================
# CANONICAL PROTOCOL
# =============================================================================

SESSION_COL = "session_id"
LABEL_COL = "label"

CLASSES = [
    "focused",
    "distracted",
    "fatigued",
    "overloaded",
]

EXPECTED_CLASS_COUNTS = {
    "focused": 77,
    "distracted": 77,
    "fatigued": 77,
    "overloaded": 78,
}

EXPECTED_SAMPLE_COUNT = 309
EXPECTED_TOTAL_FEATURES = 2373

EXPECTED_GROUP_COUNTS = {
    "keystroke": 22,
    "text": 768,
    "audio": 809,
    "vision": 768,
    "derived": 6,
}

EXPECTED_MASK_COUNTS = {
    "All modalities available": 0,
    "Keystroke unavailable": 22,
    "Text unavailable": 768,
    "Audio unavailable": 809,
    "Vision unavailable": 774,
}

EXPECTED_DERIVED_FEATURES = [
    "image_webcam_focused_prob",
    "image_webcam_distracted_prob",
    "image_webcam_fatigued_prob",
    "image_webcam_overloaded_prob",
    "image_webcam_top_probability",
    "image_webcam_confidence_gap",
]

CV_SPLITS = 5
RANDOM_STATE = 42

EXPECTED_PYTHON_VERSION = "3.11.9"

FOLD_SD_DDOF = 0

REPRODUCTION_TOLERANCE = 1e-10

MASK_VALUE = 0.0

DEFAULT_SEVERE_IMBALANCE_THRESHOLD = 0.80


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass(frozen=True)
class MissingModalityCondition:
    name: str
    masked_groups: tuple[str, ...]
    masked_features: tuple[str, ...]
    masked_feature_count: int


@dataclass
class FittedFold:
    fold: int
    train_indices: np.ndarray
    test_indices: np.ndarray
    model: Pipeline
    baseline_accuracy: float
    baseline_macro_f1: float


# =============================================================================
# GENERAL HELPERS
# =============================================================================

def print_heading(
    title: str,
) -> None:
    print()
    print("=" * 100)
    print(title)
    print("=" * 100)


def require_file(
    path: Path,
    description: str,
) -> None:
    if not path.exists():
        raise FileNotFoundError(
            f"{description} not found:\n{path}"
        )

    if not path.is_file():
        raise ValueError(
            f"{description} is not a file:\n{path}"
        )


def normalise_label(
    value: Any,
) -> str:
    if pd.isna(value):
        return ""

    return str(value).strip().lower()


def load_json(
    path: Path,
) -> Any:
    require_file(
        path,
        "JSON artifact",
    )

    with path.open(
        "r",
        encoding="utf-8",
    ) as file_handle:
        return json.load(file_handle)


def sha256_file(
    path: Path,
) -> str:
    digest = hashlib.sha256()

    with path.open(
        "rb",
    ) as file_handle:
        while True:
            chunk = file_handle.read(
                1024 * 1024
            )

            if not chunk:
                break

            digest.update(chunk)

    return digest.hexdigest()


def current_python_version() -> str:
    return (
        f"{sys.version_info.major}."
        f"{sys.version_info.minor}."
        f"{sys.version_info.micro}"
    )


def safe_json_value(
    value: Any,
) -> Any:
    if isinstance(
        value,
        np.ndarray,
    ):
        return [
            safe_json_value(item)
            for item in value.tolist()
        ]

    if isinstance(
        value,
        np.generic,
    ):
        return safe_json_value(
            value.item()
        )

    if value is None:
        return None

    if isinstance(
        value,
        (
            str,
            int,
            bool,
        ),
    ):
        return value

    if isinstance(
        value,
        float,
    ):
        if math.isfinite(value):
            return value

        return str(value)

    if isinstance(
        value,
        Path,
    ):
        return str(value)

    if isinstance(
        value,
        dict,
    ):
        return {
            str(key): safe_json_value(child)
            for key, child in value.items()
        }

    if isinstance(
        value,
        (
            list,
            tuple,
            set,
        ),
    ):
        return [
            safe_json_value(child)
            for child in value
        ]

    return repr(value)


def write_json(
    path: Path,
    payload: dict[str, Any],
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open(
        "w",
        encoding="utf-8",
    ) as file_handle:
        json.dump(
            safe_json_value(payload),
            file_handle,
            indent=2,
            ensure_ascii=False,
        )


# =============================================================================
# AUTHORITATIVE METADATA
# =============================================================================

def validate_metadata(
    metadata: dict[str, Any],
) -> None:
    if not isinstance(
        metadata,
        dict,
    ):
        raise ValueError(
            "Fusion metadata must be a JSON object."
        )

    observed_samples = int(
        metadata.get(
            "num_samples",
            -1,
        )
    )

    if observed_samples != EXPECTED_SAMPLE_COUNT:
        raise ValueError(
            "Fusion metadata sample-count mismatch.\n"
            f"Expected: {EXPECTED_SAMPLE_COUNT}\n"
            f"Observed: {observed_samples}"
        )

    observed_features = int(
        metadata.get(
            "num_features",
            -1,
        )
    )

    if observed_features != EXPECTED_TOTAL_FEATURES:
        raise ValueError(
            "Fusion metadata feature-count mismatch.\n"
            f"Expected: {EXPECTED_TOTAL_FEATURES}\n"
            f"Observed: {observed_features}"
        )

    selected_model = (
        str(
            metadata.get(
                "selected_model",
                "",
            )
        )
        .strip()
        .lower()
    )

    if selected_model != "random_forest":
        raise ValueError(
            "The authoritative final fusion model is not "
            "identified as Random Forest.\n"
            f"Observed: {selected_model!r}"
        )

    metadata_classes = [
        normalise_label(value)
        for value in metadata.get(
            "classes",
            [],
        )
    ]

    if metadata_classes != CLASSES:
        raise ValueError(
            "Fusion metadata class order mismatch.\n"
            f"Expected: {CLASSES}\n"
            f"Observed: {metadata_classes}"
        )

    observed_distribution = {
        normalise_label(key): int(value)
        for key, value in metadata.get(
            "class_distribution",
            {},
        ).items()
    }

    if observed_distribution != EXPECTED_CLASS_COUNTS:
        raise ValueError(
            "Fusion metadata class-distribution mismatch.\n"
            f"Expected: {EXPECTED_CLASS_COUNTS}\n"
            f"Observed: {observed_distribution}"
        )

    required_cv_fields = [
        "cv_accuracy_mean",
        "cv_accuracy_std",
        "cv_macro_f1_mean",
        "cv_macro_f1_std",
    ]

    missing_fields = [
        field
        for field in required_cv_fields
        if field not in metadata
    ]

    if missing_fields:
        raise ValueError(
            "Fusion metadata is missing authoritative "
            "cross-validation metrics:\n"
            f"{missing_fields}"
        )

    webcam_metadata = metadata.get(
        "webcam_image_calibration",
        {},
    )

    if not isinstance(
        webcam_metadata,
        dict,
    ):
        raise ValueError(
            "webcam_image_calibration metadata "
            "must be a JSON object."
        )

    if not bool(
        webcam_metadata.get(
            "enabled",
            False,
        )
    ):
        raise ValueError(
            "The final 2,373-feature metadata does not "
            "report webcam visual augmentation as enabled."
        )

    observed_derived_features = [
        str(value)
        for value in webcam_metadata.get(
            "calibration_features_added",
            [],
        )
    ]

    if (
        observed_derived_features
        != EXPECTED_DERIVED_FEATURES
    ):
        raise ValueError(
            "Visual-derived feature definition mismatch.\n"
            f"Expected: {EXPECTED_DERIVED_FEATURES}\n"
            f"Observed: {observed_derived_features}"
        )


# =============================================================================
# FEATURE SCHEMA
# =============================================================================

def load_feature_schema() -> list[str]:
    payload = load_json(
        FEATURE_SCHEMA_PATH
    )

    if not isinstance(
        payload,
        list,
    ):
        raise ValueError(
            "feature_columns.json must contain a JSON list."
        )

    feature_columns = [
        str(column).strip()
        for column in payload
    ]

    if (
        len(feature_columns)
        != EXPECTED_TOTAL_FEATURES
    ):
        raise ValueError(
            "Fusion feature-schema dimension mismatch.\n"
            f"Expected: {EXPECTED_TOTAL_FEATURES}\n"
            f"Observed: {len(feature_columns)}"
        )

    if any(
        not column
        for column in feature_columns
    ):
        raise ValueError(
            "Fusion feature schema contains an empty name."
        )

    if (
        len(set(feature_columns))
        != len(feature_columns)
    ):
        raise ValueError(
            "Fusion feature schema contains duplicate names."
        )

    return feature_columns


def resolve_feature_groups(
    feature_columns: list[str],
) -> dict[str, list[str]]:
    groups = {
        "keystroke": [
            column
            for column in feature_columns
            if column.startswith(
                "keystroke_"
            )
        ],

        "text": [
            column
            for column in feature_columns
            if column.startswith(
                "text_mpnet_emb_"
            )
        ],

        "audio": [
            column
            for column in feature_columns
            if column.startswith(
                "audio_"
            )
        ],

        "vision": [
            column
            for column in feature_columns
            if column.startswith(
                "image_clip_emb_"
            )
        ],

        "derived": [
            column
            for column in feature_columns
            if column.startswith(
                "image_webcam_"
            )
        ],
    }

    for (
        group_name,
        expected_count,
    ) in EXPECTED_GROUP_COUNTS.items():
        observed_count = len(
            groups[group_name]
        )

        if observed_count != expected_count:
            raise ValueError(
                "Feature-group count mismatch.\n"
                f"Group: {group_name}\n"
                f"Expected: {expected_count}\n"
                f"Observed: {observed_count}"
            )

    if (
        groups["derived"]
        != EXPECTED_DERIVED_FEATURES
    ):
        raise ValueError(
            "Derived visual feature list mismatch.\n"
            f"Expected: {EXPECTED_DERIVED_FEATURES}\n"
            f"Observed: {groups['derived']}"
        )

    all_grouped_features: list[str] = []

    for columns in groups.values():
        all_grouped_features.extend(
            columns
        )

    if (
        len(all_grouped_features)
        != EXPECTED_TOTAL_FEATURES
    ):
        grouped_set = set(
            all_grouped_features
        )

        unclassified = [
            column
            for column in feature_columns
            if column not in grouped_set
        ]

        raise ValueError(
            "Feature schema is not fully explained by "
            "the validated modality groups.\n"
            f"Grouped: {len(all_grouped_features)}\n"
            f"Expected: {EXPECTED_TOTAL_FEATURES}\n"
            f"Unclassified examples: {unclassified[:20]}"
        )

    if (
        len(set(all_grouped_features))
        != EXPECTED_TOTAL_FEATURES
    ):
        raise ValueError(
            "At least one feature belongs to more than "
            "one resolved feature group."
        )

    if (
        set(all_grouped_features)
        != set(feature_columns)
    ):
        raise ValueError(
            "Resolved feature groups do not exactly match "
            "the persisted feature schema."
        )

    return groups


# =============================================================================
# AUTHORITATIVE DATASET
# =============================================================================

def clean_numeric_dataframe(
    dataframe: pd.DataFrame,
    feature_columns: list[str],
) -> pd.DataFrame:
    output = dataframe[
        feature_columns
    ].copy()

    for column in feature_columns:
        output[column] = pd.to_numeric(
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

    values = output.to_numpy(
        dtype=np.float64
    )

    if not np.all(
        np.isfinite(values)
    ):
        raise ValueError(
            "Non-finite values remain after numeric cleaning."
        )

    return output


def load_authoritative_dataset(
    feature_columns: list[str],
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.Series,
]:
    require_file(
        DATASET_PATH,
        "Authoritative fusion training dataset",
    )

    dataframe = pd.read_csv(
        DATASET_PATH
    )

    if len(dataframe) != EXPECTED_SAMPLE_COUNT:
        raise ValueError(
            "Authoritative dataset row-count mismatch.\n"
            f"Expected: {EXPECTED_SAMPLE_COUNT}\n"
            f"Observed: {len(dataframe)}"
        )

    for required_column in [
        SESSION_COL,
        LABEL_COL,
    ]:
        if required_column not in dataframe.columns:
            raise ValueError(
                "Authoritative dataset is missing "
                f"required column {required_column!r}."
            )

    missing_features = [
        column
        for column in feature_columns
        if column not in dataframe.columns
    ]

    if missing_features:
        raise ValueError(
            "Authoritative dataset is missing persisted "
            "fusion features.\n"
            f"Missing count: {len(missing_features)}\n"
            f"Examples: {missing_features[:20]}"
        )

    dataset_feature_columns = [
        column
        for column in dataframe.columns
        if column not in {
            SESSION_COL,
            LABEL_COL,
        }
    ]

    if (
        dataset_feature_columns
        != feature_columns
    ):
        raise ValueError(
            "Feature order in fusion_training_dataset.csv "
            "does not exactly match feature_columns.json."
        )

    if dataframe[
        SESSION_COL
    ].isna().any():
        raise ValueError(
            "Dataset contains missing session IDs."
        )

    session_ids = (
        dataframe[
            SESSION_COL
        ]
        .astype(str)
    )

    if session_ids.duplicated().any():
        duplicate_values = (
            session_ids[
                session_ids.duplicated(
                    keep=False
                )
            ]
            .unique()
            .tolist()
        )

        raise ValueError(
            "Dataset contains duplicate session IDs.\n"
            f"Examples: {duplicate_values[:20]}"
        )

    labels = (
        dataframe[
            LABEL_COL
        ]
        .apply(normalise_label)
    )

    unexpected_labels = (
        set(labels.unique())
        - set(CLASSES)
    )

    if unexpected_labels:
        raise ValueError(
            "Dataset contains unsupported labels.\n"
            f"Unexpected: {sorted(unexpected_labels)}"
        )

    observed_counts = {
        label: int(
            (labels == label).sum()
        )
        for label in CLASSES
    }

    if observed_counts != EXPECTED_CLASS_COUNTS:
        raise ValueError(
            "Authoritative dataset class-distribution mismatch.\n"
            f"Expected: {EXPECTED_CLASS_COUNTS}\n"
            f"Observed: {observed_counts}"
        )

    X = clean_numeric_dataframe(
        dataframe,
        feature_columns,
    )

    y = labels.copy()

    return (
        dataframe,
        X,
        y,
    )


# =============================================================================
# DEPLOYED MODEL
# =============================================================================

def load_deployed_pipeline() -> tuple[
    Pipeline,
    RandomForestClassifier,
    list[dict[str, Any]],
]:
    require_file(
        DEPLOYED_MODEL_PATH,
        "Deployed fusion model",
    )

    model = joblib.load(
        DEPLOYED_MODEL_PATH
    )

    if not isinstance(
        model,
        Pipeline,
    ):
        raise TypeError(
            "Expected deployed fusion artifact to be "
            "an sklearn Pipeline.\n"
            f"Observed: {type(model)}"
        )

    random_forests: list[
        tuple[
            str,
            RandomForestClassifier,
        ]
    ] = []

    preprocessing_steps: list[
        dict[str, Any]
    ] = []

    for (
        step_name,
        step,
    ) in model.steps:
        if isinstance(
            step,
            RandomForestClassifier,
        ):
            random_forests.append(
                (
                    step_name,
                    step,
                )
            )
        else:
            preprocessing_steps.append(
                {
                    "name": step_name,
                    "type": (
                        f"{type(step).__module__}."
                        f"{type(step).__name__}"
                    ),
                    "parameters": (
                        safe_json_value(
                            step.get_params(
                                deep=False
                            )
                        )
                        if hasattr(
                            step,
                            "get_params",
                        )
                        else {}
                    ),
                }
            )

    if len(random_forests) != 1:
        raise ValueError(
            "Deployed pipeline must contain exactly one "
            "RandomForestClassifier.\n"
            f"Observed: {len(random_forests)}"
        )

    _, classifier = random_forests[0]

    parameters = classifier.get_params(
        deep=False
    )

    expected_parameters = {
        "n_estimators": 500,
        "class_weight": "balanced",
        "random_state": RANDOM_STATE,
        "n_jobs": -1,
    }

    mismatches: dict[
        str,
        dict[str, Any],
    ] = {}

    for (
        parameter_name,
        expected_value,
    ) in expected_parameters.items():
        observed_value = parameters.get(
            parameter_name
        )

        if observed_value != expected_value:
            mismatches[
                parameter_name
            ] = {
                "expected": expected_value,
                "observed": observed_value,
            }

    if mismatches:
        raise ValueError(
            "Deployed Random Forest configuration mismatch.\n"
            + json.dumps(
                mismatches,
                indent=2,
            )
        )

    model_feature_count = getattr(
        model,
        "n_features_in_",
        None,
    )

    if (
        model_feature_count is not None
        and int(model_feature_count)
        != EXPECTED_TOTAL_FEATURES
    ):
        raise ValueError(
            "Deployed pipeline feature-count mismatch.\n"
            f"Expected: {EXPECTED_TOTAL_FEATURES}\n"
            f"Observed: {model_feature_count}"
        )

    return (
        model,
        classifier,
        preprocessing_steps,
    )


# =============================================================================
# MISSING-MODALITY CONDITIONS
# =============================================================================

def build_conditions(
    feature_columns: list[str],
    feature_groups: dict[str, list[str]],
) -> list[MissingModalityCondition]:
    definitions: list[
        tuple[
            str,
            tuple[str, ...],
        ]
    ] = [
        (
            "All modalities available",
            (),
        ),
        (
            "Keystroke unavailable",
            (
                "keystroke",
            ),
        ),
        (
            "Text unavailable",
            (
                "text",
            ),
        ),
        (
            "Audio unavailable",
            (
                "audio",
            ),
        ),
        (
            "Vision unavailable",
            (
                "vision",
                "derived",
            ),
        ),
    ]

    conditions: list[
        MissingModalityCondition
    ] = []

    for (
        condition_name,
        masked_group_names,
    ) in definitions:
        masked_set: set[str] = set()

        for group_name in masked_group_names:
            masked_set.update(
                feature_groups[
                    group_name
                ]
            )

        masked_features = [
            column
            for column in feature_columns
            if column in masked_set
        ]

        expected_count = EXPECTED_MASK_COUNTS[
            condition_name
        ]

        if len(masked_features) != expected_count:
            raise ValueError(
                "Mask feature-count mismatch.\n"
                f"Condition: {condition_name}\n"
                f"Expected: {expected_count}\n"
                f"Observed: {len(masked_features)}"
            )

        conditions.append(
            MissingModalityCondition(
                name=condition_name,
                masked_groups=tuple(
                    masked_group_names
                ),
                masked_features=tuple(
                    masked_features
                ),
                masked_feature_count=len(
                    masked_features
                ),
            )
        )

    return conditions


# =============================================================================
# ZERO MASKING
# =============================================================================

def apply_zero_mask(
    X: pd.DataFrame,
    condition: MissingModalityCondition,
) -> pd.DataFrame:
    masked = X.copy(
        deep=True
    )

    if not condition.masked_features:
        if not np.array_equal(
            masked.to_numpy(
                dtype=np.float64
            ),
            X.to_numpy(
                dtype=np.float64
            ),
        ):
            raise RuntimeError(
                "Baseline copy unexpectedly differs "
                "from the original held-out matrix."
            )

        return masked

    missing_columns = [
        column
        for column in condition.masked_features
        if column not in masked.columns
    ]

    if missing_columns:
        raise ValueError(
            "Mask contains features absent from "
            "the held-out matrix.\n"
            f"Examples: {missing_columns[:20]}"
        )

    masked_feature_set = set(
        condition.masked_features
    )

    unmasked_columns = [
        column
        for column in X.columns
        if column not in masked_feature_set
    ]

    original_unmasked = (
        X[
            unmasked_columns
        ]
        .to_numpy(
            dtype=np.float64,
            copy=True,
        )
    )

    masked.loc[
        :,
        list(
            condition.masked_features
        ),
    ] = MASK_VALUE

    masked_block = (
        masked[
            list(
                condition.masked_features
            )
        ]
        .to_numpy(
            dtype=np.float64
        )
    )

    if not np.all(
        masked_block == MASK_VALUE
    ):
        raise RuntimeError(
            "Synthetic zero mask was not applied correctly."
        )

    after_unmasked = (
        masked[
            unmasked_columns
        ]
        .to_numpy(
            dtype=np.float64
        )
    )

    if not np.array_equal(
        original_unmasked,
        after_unmasked,
    ):
        raise RuntimeError(
            "Synthetic masking modified unmasked features."
        )

    if list(
        masked.columns
    ) != list(
        X.columns
    ):
        raise RuntimeError(
            "Synthetic masking changed feature order."
        )

    if masked.shape[1] != EXPECTED_TOTAL_FEATURES:
        raise RuntimeError(
            "Synthetic masking changed classifier "
            "input dimensionality."
        )

    return masked


# =============================================================================
# FROZEN CV FOLDS
# =============================================================================

def build_frozen_folds(
    X: pd.DataFrame,
    y: pd.Series,
    session_ids: pd.Series,
) -> tuple[
    list[
        tuple[
            np.ndarray,
            np.ndarray,
        ]
    ],
    list[dict[str, Any]],
]:
    cv = StratifiedKFold(
        n_splits=CV_SPLITS,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    folds = list(
        cv.split(
            X,
            y,
        )
    )

    if len(folds) != CV_SPLITS:
        raise RuntimeError(
            "Unexpected CV fold count."
        )

    manifest: list[
        dict[str, Any]
    ] = []

    for (
        fold_number,
        (
            train_indices,
            test_indices,
        ),
    ) in enumerate(
        folds,
        start=1,
    ):
        train_session_ids = (
            session_ids.iloc[
                train_indices
            ]
            .astype(str)
            .tolist()
        )

        test_session_ids = (
            session_ids.iloc[
                test_indices
            ]
            .astype(str)
            .tolist()
        )

        overlap = (
            set(train_session_ids)
            & set(test_session_ids)
        )

        if overlap:
            raise RuntimeError(
                "Train/test session overlap detected "
                f"in fold {fold_number}.\n"
                f"Examples: {sorted(overlap)[:20]}"
            )

        train_labels = y.iloc[
            train_indices
        ]

        test_labels = y.iloc[
            test_indices
        ]

        manifest.append(
            {
                "fold": fold_number,

                "train_count": int(
                    len(train_indices)
                ),

                "test_count": int(
                    len(test_indices)
                ),

                "train_indices": [
                    int(value)
                    for value in train_indices
                ],

                "test_indices": [
                    int(value)
                    for value in test_indices
                ],

                "train_session_ids": (
                    train_session_ids
                ),

                "test_session_ids": (
                    test_session_ids
                ),

                "train_class_distribution": {
                    label: int(
                        (
                            train_labels
                            == label
                        ).sum()
                    )
                    for label in CLASSES
                },

                "test_class_distribution": {
                    label: int(
                        (
                            test_labels
                            == label
                        ).sum()
                    )
                    for label in CLASSES
                },
            }
        )

    return (
        folds,
        manifest,
    )


# =============================================================================
# CROSS-EXPERIMENT FOLD PARITY
# =============================================================================

def verify_ablation_fold_parity(
    current_manifest: list[dict[str, Any]],
) -> dict[str, Any]:
    if not ABLATION_METADATA_PATH.exists():
        return {
            "available": False,
            "matched": None,
            "path": str(
                ABLATION_METADATA_PATH
            ),
            "note": (
                "Section 5.6.1 metadata was not found. "
                "The deterministic StratifiedKFold protocol "
                "was reconstructed directly."
            ),
        }

    previous_metadata = load_json(
        ABLATION_METADATA_PATH
    )

    if not isinstance(
        previous_metadata,
        dict,
    ):
        raise ValueError(
            "Existing ablation metadata is invalid."
        )

    previous_cross_validation = (
        previous_metadata.get(
            "cross_validation",
            {},
        )
    )

    if not isinstance(
        previous_cross_validation,
        dict,
    ):
        raise ValueError(
            "Existing ablation metadata does not contain "
            "valid cross-validation metadata."
        )

    previous_manifest = (
        previous_cross_validation.get(
            "fold_manifest",
            [],
        )
    )

    if not isinstance(
        previous_manifest,
        list,
    ):
        raise ValueError(
            "Existing ablation fold manifest is invalid."
        )

    if (
        len(previous_manifest)
        != len(current_manifest)
    ):
        raise RuntimeError(
            "Section 5.6.2 fold count differs from "
            "Section 5.6.1."
        )

    mismatches: list[
        dict[str, Any]
    ] = []

    for (
        current,
        previous,
    ) in zip(
        current_manifest,
        previous_manifest,
    ):
        current_fold = int(
            current["fold"]
        )

        previous_fold = int(
            previous.get(
                "fold",
                -1,
            )
        )

        if current_fold != previous_fold:
            mismatches.append(
                {
                    "fold": current_fold,
                    "field": "fold",
                }
            )

        for field in [
            "train_session_ids",
            "test_session_ids",
        ]:
            current_values = [
                str(value)
                for value in current.get(
                    field,
                    [],
                )
            ]

            previous_values = [
                str(value)
                for value in previous.get(
                    field,
                    [],
                )
            ]

            if current_values != previous_values:
                mismatches.append(
                    {
                        "fold": current_fold,
                        "field": field,
                        "current_count": len(
                            current_values
                        ),
                        "previous_count": len(
                            previous_values
                        ),
                    }
                )

    if mismatches:
        raise RuntimeError(
            "Cross-experiment fold parity check failed.\n"
            f"Mismatches: {mismatches[:10]}"
        )

    return {
        "available": True,
        "matched": True,
        "path": str(
            ABLATION_METADATA_PATH
        ),
        "sha256": sha256_file(
            ABLATION_METADATA_PATH
        ),
        "verified_fields": [
            "fold",
            "train_session_ids",
            "test_session_ids",
        ],
    }


# =============================================================================
# PREDICTION DIAGNOSTICS
# =============================================================================

def prediction_diagnostics(
    y_true: pd.Series,
    predictions: np.ndarray,
    *,
    severe_imbalance_threshold: float,
) -> dict[str, Any]:
    prediction_labels = pd.Series(
        [
            normalise_label(value)
            for value in predictions
        ]
    )

    unexpected = (
        set(
            prediction_labels.unique()
        )
        - set(CLASSES)
    )

    if unexpected:
        raise ValueError(
            "Classifier produced unsupported labels.\n"
            f"Unexpected: {sorted(unexpected)}"
        )

    sample_count = int(
        len(prediction_labels)
    )

    if sample_count <= 0:
        raise ValueError(
            "Cannot evaluate an empty prediction set."
        )

    predicted_counts = {
        label: int(
            (
                prediction_labels
                == label
            ).sum()
        )
        for label in CLASSES
    }

    predicted_proportions = {
        label: float(
            predicted_counts[label]
            / sample_count
        )
        for label in CLASSES
    }

    dominant_class = max(
        CLASSES,
        key=lambda label: (
            predicted_counts[label]
        ),
    )

    dominant_proportion = float(
        predicted_proportions[
            dominant_class
        ]
    )

    unique_predicted_classes = sum(
        1
        for label in CLASSES
        if predicted_counts[label] > 0
    )

    class_collapse = (
        unique_predicted_classes == 1
    )

    severe_prediction_imbalance = (
        dominant_proportion
        >= severe_imbalance_threshold
    )

    matrix = confusion_matrix(
        y_true,
        prediction_labels,
        labels=CLASSES,
    )

    return {
        "predicted_counts": (
            predicted_counts
        ),

        "predicted_proportions": (
            predicted_proportions
        ),

        "dominant_predicted_class": (
            dominant_class
        ),

        "dominant_predicted_proportion": (
            dominant_proportion
        ),

        "unique_predicted_classes": int(
            unique_predicted_classes
        ),

        "class_collapse": bool(
            class_collapse
        ),

        "severe_prediction_imbalance": bool(
            severe_prediction_imbalance
        ),

        "confusion_matrix": (
            matrix.tolist()
        ),
    }


def evaluate_prediction_set(
    y_true: pd.Series,
    predictions: np.ndarray,
    *,
    severe_imbalance_threshold: float,
) -> dict[str, Any]:
    accuracy = accuracy_score(
        y_true,
        predictions,
    )

    macro_f1 = f1_score(
        y_true,
        predictions,
        labels=CLASSES,
        average="macro",
        zero_division=0,
    )

    diagnostics = prediction_diagnostics(
        y_true,
        predictions,
        severe_imbalance_threshold=(
            severe_imbalance_threshold
        ),
    )

    return {
        "accuracy": float(
            accuracy
        ),

        "macro_f1": float(
            macro_f1
        ),

        **diagnostics,
    }


# =============================================================================
# FOLD RESULT ROW
# =============================================================================

def make_fold_result_row(
    *,
    condition_name: str,
    fold_number: int,
    train_samples: int,
    test_samples: int,
    masked_groups: tuple[str, ...],
    masked_feature_count: int,
    metrics: dict[str, Any],
    baseline_accuracy: float,
    baseline_macro_f1: float,
) -> dict[str, Any]:
    observed_accuracy = float(
        metrics["accuracy"]
    )

    observed_macro_f1 = float(
        metrics["macro_f1"]
    )

    delta_accuracy = float(
        baseline_accuracy
        - observed_accuracy
    )

    delta_macro_f1 = float(
        baseline_macro_f1
        - observed_macro_f1
    )

    row: dict[str, Any] = {
        "condition": (
            condition_name
        ),

        "fold": int(
            fold_number
        ),

        "input_dimensions": (
            EXPECTED_TOTAL_FEATURES
        ),

        "masked_groups": (
            "+".join(masked_groups)
            if masked_groups
            else ""
        ),

        "masked_feature_count": int(
            masked_feature_count
        ),

        "train_samples": int(
            train_samples
        ),

        "test_samples": int(
            test_samples
        ),

        "accuracy": (
            observed_accuracy
        ),

        "macro_f1": (
            observed_macro_f1
        ),

        "baseline_accuracy_same_fold": float(
            baseline_accuracy
        ),

        "baseline_macro_f1_same_fold": float(
            baseline_macro_f1
        ),

        "delta_accuracy": (
            delta_accuracy
        ),

        "delta_macro_f1": (
            delta_macro_f1
        ),

        "dominant_predicted_class": (
            metrics[
                "dominant_predicted_class"
            ]
        ),

        "dominant_predicted_proportion": float(
            metrics[
                "dominant_predicted_proportion"
            ]
        ),

        "unique_predicted_classes": int(
            metrics[
                "unique_predicted_classes"
            ]
        ),

        "class_collapse": bool(
            metrics[
                "class_collapse"
            ]
        ),

        "severe_prediction_imbalance": bool(
            metrics[
                "severe_prediction_imbalance"
            ]
        ),
    }

    for label in CLASSES:
        row[
            f"predicted_{label}"
        ] = int(
            metrics[
                "predicted_counts"
            ][
                label
            ]
        )

    return row


# =============================================================================
# BASELINE MODEL FITTING
# =============================================================================

def fit_complete_fold_models_and_baseline(
    *,
    deployed_pipeline: Pipeline,
    X_full: pd.DataFrame,
    y: pd.Series,
    folds: list[
        tuple[
            np.ndarray,
            np.ndarray,
        ]
    ],
    severe_imbalance_threshold: float,
) -> tuple[
    list[FittedFold],
    list[dict[str, Any]],
    list[str],
    list[str],
]:
    fitted_folds: list[
        FittedFold
    ] = []

    baseline_rows: list[
        dict[str, Any]
    ] = []

    pooled_true: list[str] = []
    pooled_predictions: list[str] = []

    for (
        fold_number,
        (
            train_indices,
            test_indices,
        ),
    ) in enumerate(
        folds,
        start=1,
    ):
        model = clone(
            deployed_pipeline
        )

        X_train = X_full.iloc[
            train_indices
        ]

        X_test = X_full.iloc[
            test_indices
        ]

        y_train = y.iloc[
            train_indices
        ]

        y_test = y.iloc[
            test_indices
        ]

        # Exactly one model fit for this fold.
        model.fit(
            X_train,
            y_train,
        )

        predictions = np.asarray(
            model.predict(
                X_test
            ),
            dtype=object,
        )

        metrics = evaluate_prediction_set(
            y_test,
            predictions,
            severe_imbalance_threshold=(
                severe_imbalance_threshold
            ),
        )

        baseline_accuracy = float(
            metrics[
                "accuracy"
            ]
        )

        baseline_macro_f1 = float(
            metrics[
                "macro_f1"
            ]
        )

        baseline_rows.append(
            make_fold_result_row(
                condition_name=(
                    "All modalities available"
                ),

                fold_number=(
                    fold_number
                ),

                train_samples=len(
                    train_indices
                ),

                test_samples=len(
                    test_indices
                ),

                masked_groups=(),

                masked_feature_count=0,

                metrics=metrics,

                baseline_accuracy=(
                    baseline_accuracy
                ),

                baseline_macro_f1=(
                    baseline_macro_f1
                ),
            )
        )

        fitted_folds.append(
            FittedFold(
                fold=fold_number,

                train_indices=(
                    train_indices
                ),

                test_indices=(
                    test_indices
                ),

                model=model,

                baseline_accuracy=(
                    baseline_accuracy
                ),

                baseline_macro_f1=(
                    baseline_macro_f1
                ),
            )
        )

        pooled_true.extend(
            [
                normalise_label(value)
                for value in y_test.tolist()
            ]
        )

        pooled_predictions.extend(
            [
                normalise_label(value)
                for value in predictions.tolist()
            ]
        )

        print(
            f"  Fold {fold_number}: "
            f"accuracy={baseline_accuracy:.4f}, "
            f"macro-F1={baseline_macro_f1:.4f}"
        )

    if len(fitted_folds) != CV_SPLITS:
        raise RuntimeError(
            "Unexpected fitted baseline model count."
        )

    return (
        fitted_folds,
        baseline_rows,
        pooled_true,
        pooled_predictions,
    )


# =============================================================================
# SUMMARY STATISTICS
# =============================================================================

def summarise_fold_metrics(
    fold_rows: list[dict[str, Any]],
) -> dict[str, float]:
    if len(fold_rows) != CV_SPLITS:
        raise RuntimeError(
            "Expected exactly five fold results."
        )

    accuracies = np.asarray(
        [
            float(row["accuracy"])
            for row in fold_rows
        ],
        dtype=np.float64,
    )

    macro_f1_values = np.asarray(
        [
            float(row["macro_f1"])
            for row in fold_rows
        ],
        dtype=np.float64,
    )

    delta_accuracy_values = np.asarray(
        [
            float(
                row["delta_accuracy"]
            )
            for row in fold_rows
        ],
        dtype=np.float64,
    )

    delta_macro_f1_values = np.asarray(
        [
            float(
                row["delta_macro_f1"]
            )
            for row in fold_rows
        ],
        dtype=np.float64,
    )

    return {
        "mean_accuracy": float(
            np.mean(accuracies)
        ),

        "accuracy_sd": float(
            np.std(
                accuracies,
                ddof=FOLD_SD_DDOF,
            )
        ),

        "mean_macro_f1": float(
            np.mean(
                macro_f1_values
            )
        ),

        "macro_f1_sd": float(
            np.std(
                macro_f1_values,
                ddof=FOLD_SD_DDOF,
            )
        ),

        "mean_delta_accuracy": float(
            np.mean(
                delta_accuracy_values
            )
        ),

        "mean_delta_macro_f1": float(
            np.mean(
                delta_macro_f1_values
            )
        ),
    }


# =============================================================================
# BASELINE REPRODUCTION
# =============================================================================

def compare_metric(
    *,
    observed: float,
    expected: float,
    tolerance: float,
) -> Optional[dict[str, float]]:
    difference = abs(
        float(observed)
        - float(expected)
    )

    if difference <= tolerance:
        return None

    return {
        "observed": float(
            observed
        ),

        "expected": float(
            expected
        ),

        "absolute_difference": float(
            difference
        ),

        "tolerance": float(
            tolerance
        ),
    }


def verify_baseline_reproduction(
    baseline_summary: dict[str, float],
    metadata: dict[str, Any],
    *,
    tolerance: float,
) -> dict[str, Any]:
    references = {
        "mean_accuracy": float(
            metadata[
                "cv_accuracy_mean"
            ]
        ),

        "accuracy_sd": float(
            metadata[
                "cv_accuracy_std"
            ]
        ),

        "mean_macro_f1": float(
            metadata[
                "cv_macro_f1_mean"
            ]
        ),

        "macro_f1_sd": float(
            metadata[
                "cv_macro_f1_std"
            ]
        ),
    }

    failures: dict[
        str,
        dict[str, float],
    ] = {}

    for (
        metric_name,
        expected_value,
    ) in references.items():
        failure = compare_metric(
            observed=float(
                baseline_summary[
                    metric_name
                ]
            ),

            expected=(
                expected_value
            ),

            tolerance=(
                tolerance
            ),
        )

        if failure is not None:
            failures[
                metric_name
            ] = failure

    return {
        "passed": (
            not failures
        ),

        "reference_source": str(
            METADATA_PATH
        ),

        "tolerance": float(
            tolerance
        ),

        "reference": references,

        "observed": {
            key: float(
                baseline_summary[
                    key
                ]
            )
            for key in references
        },

        "failures": failures,
    }


# =============================================================================
# MISSING-CONDITION EVALUATION
# =============================================================================

def evaluate_missing_condition(
    *,
    condition: MissingModalityCondition,
    fitted_folds: list[FittedFold],
    X_full: pd.DataFrame,
    y: pd.Series,
    severe_imbalance_threshold: float,
) -> tuple[
    list[dict[str, Any]],
    list[str],
    list[str],
]:
    """
    Evaluate a synthetic missing-modality condition.

    No fitting or refitting occurs inside this function.
    """

    fold_rows: list[
        dict[str, Any]
    ] = []

    pooled_true: list[str] = []
    pooled_predictions: list[str] = []

    for fitted_fold in fitted_folds:
        X_test_original = X_full.iloc[
            fitted_fold.test_indices
        ]

        y_test = y.iloc[
            fitted_fold.test_indices
        ]

        X_test_masked = apply_zero_mask(
            X_test_original,
            condition,
        )

        predictions = np.asarray(
            fitted_fold.model.predict(
                X_test_masked
            ),
            dtype=object,
        )

        metrics = evaluate_prediction_set(
            y_test,
            predictions,
            severe_imbalance_threshold=(
                severe_imbalance_threshold
            ),
        )

        current_accuracy = float(
            metrics[
                "accuracy"
            ]
        )

        current_macro_f1 = float(
            metrics[
                "macro_f1"
            ]
        )

        delta_f1 = float(
            fitted_fold.baseline_macro_f1
            - current_macro_f1
        )

        fold_rows.append(
            make_fold_result_row(
                condition_name=(
                    condition.name
                ),

                fold_number=(
                    fitted_fold.fold
                ),

                train_samples=len(
                    fitted_fold.train_indices
                ),

                test_samples=len(
                    fitted_fold.test_indices
                ),

                masked_groups=(
                    condition.masked_groups
                ),

                masked_feature_count=(
                    condition.masked_feature_count
                ),

                metrics=metrics,

                baseline_accuracy=(
                    fitted_fold.baseline_accuracy
                ),

                baseline_macro_f1=(
                    fitted_fold.baseline_macro_f1
                ),
            )
        )

        pooled_true.extend(
            [
                normalise_label(value)
                for value in y_test.tolist()
            ]
        )

        pooled_predictions.extend(
            [
                normalise_label(value)
                for value in predictions.tolist()
            ]
        )

        # Important:
        # Keep the formatted expression on one physical source line.
        print(
            f"  Fold {fitted_fold.fold}: "
            f"accuracy={current_accuracy:.4f}, "
            f"macro-F1={current_macro_f1:.4f}, "
            f"ΔF1={delta_f1:.4f}"
        )

    return (
        fold_rows,
        pooled_true,
        pooled_predictions,
    )


# =============================================================================
# CONDITION SUMMARY
# =============================================================================

def summarise_condition(
    *,
    condition: MissingModalityCondition,
    fold_rows: list[dict[str, Any]],
    pooled_true: list[str],
    pooled_predictions: list[str],
    severe_imbalance_threshold: float,
) -> tuple[
    dict[str, Any],
    dict[str, Any],
]:
    fold_summary = summarise_fold_metrics(
        fold_rows
    )

    pooled_true_series = pd.Series(
        pooled_true
    )

    pooled_prediction_array = np.asarray(
        pooled_predictions,
        dtype=object,
    )

    pooled_metrics = evaluate_prediction_set(
        pooled_true_series,
        pooled_prediction_array,
        severe_imbalance_threshold=(
            severe_imbalance_threshold
        ),
    )

    result_row: dict[
        str,
        Any,
    ] = {
        "condition": (
            condition.name
        ),

        "input_dimensions": (
            EXPECTED_TOTAL_FEATURES
        ),

        "masked_groups": (
            "+".join(
                condition.masked_groups
            )
            if condition.masked_groups
            else ""
        ),

        "masked_feature_count": (
            condition.masked_feature_count
        ),

        "mean_accuracy": (
            fold_summary[
                "mean_accuracy"
            ]
        ),

        "accuracy_sd": (
            fold_summary[
                "accuracy_sd"
            ]
        ),

        "mean_macro_f1": (
            fold_summary[
                "mean_macro_f1"
            ]
        ),

        "macro_f1_sd": (
            fold_summary[
                "macro_f1_sd"
            ]
        ),

        "mean_delta_accuracy": (
            fold_summary[
                "mean_delta_accuracy"
            ]
        ),

        "delta_macro_f1": (
            fold_summary[
                "mean_delta_macro_f1"
            ]
        ),

        "pooled_accuracy": (
            pooled_metrics[
                "accuracy"
            ]
        ),

        "pooled_macro_f1": (
            pooled_metrics[
                "macro_f1"
            ]
        ),

        "dominant_predicted_class": (
            pooled_metrics[
                "dominant_predicted_class"
            ]
        ),

        "dominant_predicted_proportion": (
            pooled_metrics[
                "dominant_predicted_proportion"
            ]
        ),

        "unique_predicted_classes": (
            pooled_metrics[
                "unique_predicted_classes"
            ]
        ),

        "class_collapse": (
            pooled_metrics[
                "class_collapse"
            ]
        ),

        "severe_prediction_imbalance": (
            pooled_metrics[
                "severe_prediction_imbalance"
            ]
        ),
    }

    for label in CLASSES:
        result_row[
            f"predicted_{label}"
        ] = int(
            pooled_metrics[
                "predicted_counts"
            ][label]
        )

    detailed_diagnostics = {
        "condition": (
            condition.name
        ),

        "pooled_sample_count": len(
            pooled_true
        ),

        "pooled_accuracy": (
            pooled_metrics[
                "accuracy"
            ]
        ),

        "pooled_macro_f1": (
            pooled_metrics[
                "macro_f1"
            ]
        ),

        "predicted_counts": (
            pooled_metrics[
                "predicted_counts"
            ]
        ),

        "predicted_proportions": (
            pooled_metrics[
                "predicted_proportions"
            ]
        ),

        "dominant_predicted_class": (
            pooled_metrics[
                "dominant_predicted_class"
            ]
        ),

        "dominant_predicted_proportion": (
            pooled_metrics[
                "dominant_predicted_proportion"
            ]
        ),

        "unique_predicted_classes": (
            pooled_metrics[
                "unique_predicted_classes"
            ]
        ),

        "class_collapse": (
            pooled_metrics[
                "class_collapse"
            ]
        ),

        "severe_prediction_imbalance": (
            pooled_metrics[
                "severe_prediction_imbalance"
            ]
        ),

        "confusion_matrix_labels": (
            CLASSES
        ),

        "confusion_matrix": (
            pooled_metrics[
                "confusion_matrix"
            ]
        ),
    }

    return (
        result_row,
        detailed_diagnostics,
    )


# =============================================================================
# REPORTING
# =============================================================================

def tied_rows(
    dataframe: pd.DataFrame,
    *,
    column: str,
    target_value: float,
    tolerance: float = 1e-12,
) -> pd.DataFrame:
    return dataframe[
        np.isclose(
            dataframe[
                column
            ].astype(float),
            target_value,
            rtol=0.0,
            atol=tolerance,
        )
    ]


def print_markdown_table(
    results: pd.DataFrame,
) -> None:
    print()

    print(
        "| Inference condition | Mean Accuracy | "
        "Mean Macro-F1 | Macro-F1 SD | Δ Macro-F1 |"
    )

    print(
        "|---|---:|---:|---:|---:|"
    )

    for (
        _,
        row,
    ) in results.iterrows():
        print(
            "| "
            f"{row['condition']} | "
            f"{float(row['mean_accuracy']):.4f} | "
            f"{float(row['mean_macro_f1']):.4f} | "
            f"{float(row['macro_f1_sd']):.4f} | "
            f"{float(row['delta_macro_f1']):.4f} |"
        )


def print_interpretation(
    results: pd.DataFrame,
    *,
    severe_imbalance_threshold: float,
) -> None:
    stressed = (
        results[
            results[
                "condition"
            ]
            != "All modalities available"
        ]
        .copy()
    )

    if stressed.empty:
        raise RuntimeError(
            "No missing-modality conditions "
            "are available for interpretation."
        )

    largest_delta = float(
        stressed[
            "delta_macro_f1"
        ].max()
    )

    smallest_delta = float(
        stressed[
            "delta_macro_f1"
        ].min()
    )

    largest_rows = tied_rows(
        stressed,
        column=(
            "delta_macro_f1"
        ),
        target_value=(
            largest_delta
        ),
    )

    smallest_rows = tied_rows(
        stressed,
        column=(
            "delta_macro_f1"
        ),
        target_value=(
            smallest_delta
        ),
    )

    largest_names = (
        largest_rows[
            "condition"
        ]
        .astype(str)
        .tolist()
    )

    smallest_names = (
        smallest_rows[
            "condition"
        ]
        .astype(str)
        .tolist()
    )

    collapsed_rows = stressed[
        stressed[
            "class_collapse"
        ].astype(bool)
    ]

    severe_rows = stressed[
        stressed[
            "severe_prediction_imbalance"
        ].astype(bool)
    ]

    print()
    print(
        "Factual interpretation"
    )
    print(
        "----------------------"
    )

    print(
        "Greatest predictive degradation: "
        + ", ".join(largest_names)
        + f" (Δ macro-F1 = {largest_delta:.4f})."
    )

    print(
        "Least predictive degradation: "
        + ", ".join(smallest_names)
        + f" (Δ macro-F1 = {smallest_delta:.4f})."
    )

    if collapsed_rows.empty:
        print(
            "No missing-modality condition produced "
            "complete single-class prediction collapse."
        )
    else:
        print(
            "Complete prediction collapse occurred under: "
            + ", ".join(
                collapsed_rows[
                    "condition"
                ]
                .astype(str)
                .tolist()
            )
            + "."
        )

    if severe_rows.empty:
        print(
            "No missing-modality condition exceeded "
            "the predefined severe-imbalance threshold "
            f"of {severe_imbalance_threshold:.2f}."
        )
    else:
        severe_descriptions: list[str] = []

        for (
            _,
            row,
        ) in severe_rows.iterrows():
            severe_descriptions.append(
                f"{row['condition']} "
                f"({row['dominant_predicted_class']}="
                f"{float(row['dominant_predicted_proportion']):.4f})"
            )

        print(
            "Severe prediction-imbalance flag triggered for: "
            + ", ".join(
                severe_descriptions
            )
            + "."
        )

    print()

    print(
        "Software robustness: the production SenseFuzeAI "
        "runtime still expects the complete feature contract "
        "and does not expose these zero-masked configurations "
        "as supported three-modality operating modes."
    )

    print()

    print(
        "Predictive robustness: this experiment measures "
        "the change in held-out classifier predictions after "
        "one modality is synthetically neutralised without "
        "retraining."
    )

    print()

    print(
        "Any robustness conclusion should therefore be based "
        "on the measured degradation and prediction-distribution "
        "diagnostics rather than terminology alone."
    )


# =============================================================================
# SCRIPT SNAPSHOT
# =============================================================================

def save_script_snapshot(
    output_dir: Path,
) -> Path:
    source_path = (
        Path(__file__)
        .resolve()
    )

    destination = (
        output_dir
        / source_path.name
    )

    if source_path != destination.resolve():
        shutil.copy2(
            source_path,
            destination,
        )

    return destination


# =============================================================================
# EXPERIMENT METADATA
# =============================================================================

def build_metadata_payload(
    *,
    source_metadata: dict[str, Any],
    feature_columns: list[str],
    feature_groups: dict[str, list[str]],
    conditions: list[MissingModalityCondition],
    classifier: RandomForestClassifier,
    preprocessing_steps: list[dict[str, Any]],
    fold_manifest: list[dict[str, Any]],
    fold_parity: dict[str, Any],
    reproduction: dict[str, Any],
    results: pd.DataFrame,
    detailed_diagnostics: dict[str, Any],
    severe_imbalance_threshold: float,
    output_dir: Path,
) -> dict[str, Any]:
    masking_rules: dict[
        str,
        Any,
    ] = {}

    for condition in conditions:
        masking_rules[
            condition.name
        ] = {
            "masked_groups": list(
                condition.masked_groups
            ),

            "masked_feature_count": (
                condition.masked_feature_count
            ),

            "masked_features": list(
                condition.masked_features
            ),

            "replacement_value": (
                MASK_VALUE
            ),

            "classifier_input_dimension_after_masking": (
                EXPECTED_TOTAL_FEATURES
            ),

            "retraining_after_masking": (
                False
            ),
        }

    visual_dependency_decisions = {
        feature: {
            "source_modality": (
                "vision"
            ),

            "treatment_when_vision_unavailable": (
                "zero_mask"
            ),

            "reason": (
                "This predictor is generated from the "
                "CLIP/image branch. Retaining it in the "
                "Vision-unavailable condition would preserve "
                "visual information."
            ),
        }
        for feature in EXPECTED_DERIVED_FEATURES
    }

    runtime_source_hash = (
        sha256_file(
            FINAL_INFERENCE_PATH
        )
        if FINAL_INFERENCE_PATH.exists()
        else None
    )

    return {
        "project": (
            "SenseFuzeAI"
        ),

        "experiment": (
            "inference_time_single_modality_loss_robustness"
        ),

        "report_section": (
            "5.6.2"
        ),

        "created_at": (
            datetime.now()
            .astimezone()
            .isoformat()
        ),

        "experiment_type": (
            "classifier_level_synthetic_zero_masking_stress_test"
        ),

        "methodological_scope": {
            "production_missing_modality_mode": (
                False
            ),

            "production_runtime_requires_complete_feature_schema": (
                True
            ),

            "masking_interpretation": (
                "Zero masking is a controlled schema-preserving "
                "classifier stress intervention."
            ),
        },

        "execution_environment": {
            "python_version": (
                current_python_version()
            ),

            "python_full": (
                sys.version
            ),

            "platform": (
                platform.platform()
            ),

            "processor": (
                platform.processor()
            ),

            "logical_cpu_count": (
                os.cpu_count()
            ),

            "execution_target": (
                "CPU"
            ),
        },

        "dataset": {
            "path": str(
                DATASET_PATH
            ),

            "sha256": sha256_file(
                DATASET_PATH
            ),

            "sample_count": (
                EXPECTED_SAMPLE_COUNT
            ),

            "class_distribution": (
                EXPECTED_CLASS_COUNTS
            ),

            "session_column": (
                SESSION_COL
            ),

            "label_column": (
                LABEL_COL
            ),
        },

        "feature_schema": {
            "path": str(
                FEATURE_SCHEMA_PATH
            ),

            "sha256": sha256_file(
                FEATURE_SCHEMA_PATH
            ),

            "total_features": len(
                feature_columns
            ),

            "group_counts": {
                group: len(columns)
                for (
                    group,
                    columns,
                ) in feature_groups.items()
            },

            "groups": (
                feature_groups
            ),
        },

        "runtime_contract_source": {
            "path": str(
                FINAL_INFERENCE_PATH
            ),

            "sha256": (
                runtime_source_hash
            ),
        },

        "model": {
            "artifact": str(
                DEPLOYED_MODEL_PATH
            ),

            "artifact_sha256": (
                sha256_file(
                    DEPLOYED_MODEL_PATH
                )
            ),

            "selected_model": (
                source_metadata.get(
                    "selected_model"
                )
            ),

            "random_forest_parameters": (
                safe_json_value(
                    classifier.get_params(
                        deep=False
                    )
                )
            ),

            "preprocessing_steps": (
                preprocessing_steps
            ),

            "fitting_protocol": (
                "One complete-feature model clone is fitted "
                "per training fold. The same fitted model is "
                "then reused for every held-out masking condition."
            ),

            "number_of_fold_model_fits": (
                CV_SPLITS
            ),

            "number_of_missing_condition_refits": (
                0
            ),
        },

        "cross_validation": {
            "type": (
                "StratifiedKFold"
            ),

            "n_splits": (
                CV_SPLITS
            ),

            "shuffle": (
                True
            ),

            "random_state": (
                RANDOM_STATE
            ),

            "fold_standard_deviation_ddof": (
                FOLD_SD_DDOF
            ),

            "fold_assignments_identical_across_conditions": (
                True
            ),

            "ablation_fold_parity": (
                fold_parity
            ),

            "fold_manifest": (
                fold_manifest
            ),
        },

        "metrics": {
            "primary": (
                "macro_f1"
            ),

            "complementary": (
                "accuracy"
            ),

            "delta_definition": (
                "For each fold, baseline score minus "
                "missing-modality score. The summary delta "
                "is the mean of the five paired fold deltas."
            ),

            "variability": (
                "Population standard deviation across five folds."
            ),
        },

        "masking": {
            "replacement_value": (
                MASK_VALUE
            ),

            "schema_dimension_preserved": (
                True
            ),

            "rules": (
                masking_rules
            ),

            "visual_derived_feature_decisions": (
                visual_dependency_decisions
            ),
        },

        "prediction_imbalance_diagnostics": {
            "complete_class_collapse_definition": (
                "Only one behavioural class is predicted."
            ),

            "severe_prediction_imbalance_threshold": (
                severe_imbalance_threshold
            ),

            "threshold_note": (
                "This is an operational descriptive "
                "screening threshold for this experiment."
            ),
        },

        "baseline_reproduction_check": (
            reproduction
        ),

        "authoritative_training_reference": {
            "metadata_path": str(
                METADATA_PATH
            ),

            "metadata_sha256": (
                sha256_file(
                    METADATA_PATH
                )
            ),

            "cv_accuracy_mean": (
                source_metadata.get(
                    "cv_accuracy_mean"
                )
            ),

            "cv_accuracy_std": (
                source_metadata.get(
                    "cv_accuracy_std"
                )
            ),

            "cv_macro_f1_mean": (
                source_metadata.get(
                    "cv_macro_f1_mean"
                )
            ),

            "cv_macro_f1_std": (
                source_metadata.get(
                    "cv_macro_f1_std"
                )
            ),
        },

        "summary_results": (
            results.to_dict(
                orient="records"
            )
        ),

        "detailed_prediction_diagnostics": (
            detailed_diagnostics
        ),

        "output_directory": str(
            output_dir
        ),

        "interpretation_constraint": (
            "Results measure classifier-level predictive "
            "sensitivity under synthetic zero masking. They "
            "do not demonstrate that the deployed application "
            "supports a physically absent modality."
        ),
    }


# =============================================================================
# CLI
# =============================================================================

def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run SenseFuzeAI inference-time "
            "single-modality-loss robustness testing."
        )
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=(
            "Output directory for experiment evidence."
        ),
    )

    parser.add_argument(
        "--reproduction-tolerance",
        type=float,
        default=REPRODUCTION_TOLERANCE,
        help=(
            "Absolute tolerance for baseline reproduction."
        ),
    )

    parser.add_argument(
        "--severe-imbalance-threshold",
        type=float,
        default=(
            DEFAULT_SEVERE_IMBALANCE_THRESHOLD
        ),
        help=(
            "Dominant-class proportion used to flag "
            "severe prediction imbalance. Default: 0.80."
        ),
    )

    parser.add_argument(
        "--allow-python-version-mismatch",
        action="store_true",
        help=(
            "Allow execution under a Python version "
            "other than 3.11.9."
        ),
    )

    return parser


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    parser = build_argument_parser()

    args = parser.parse_args()

    output_dir = Path(
        args.output_dir
    ).resolve()

    reproduction_tolerance = float(
        args.reproduction_tolerance
    )

    severe_imbalance_threshold = float(
        args.severe_imbalance_threshold
    )

    if reproduction_tolerance < 0.0:
        parser.error(
            "--reproduction-tolerance must be non-negative."
        )

    if not (
        0.0
        <= severe_imbalance_threshold
        <= 1.0
    ):
        parser.error(
            "--severe-imbalance-threshold "
            "must be between 0 and 1."
        )

    if (
        current_python_version()
        != EXPECTED_PYTHON_VERSION
        and
        not args.allow_python_version_mismatch
    ):
        raise RuntimeError(
            "Python version differs from the documented "
            "final experiment environment.\n"
            f"Expected: {EXPECTED_PYTHON_VERSION}\n"
            f"Observed: {current_python_version()}\n"
            "Use --allow-python-version-mismatch only "
            "when intentionally documented."
        )

    for (
        path,
        description,
    ) in [
        (
            DATASET_PATH,
            "Fusion training dataset",
        ),
        (
            FEATURE_SCHEMA_PATH,
            "Fusion feature schema",
        ),
        (
            METADATA_PATH,
            "Fusion metadata",
        ),
        (
            DEPLOYED_MODEL_PATH,
            "Deployed fusion model",
        ),
        (
            FINAL_INFERENCE_PATH,
            "Canonical final inference implementation",
        ),
    ]:
        require_file(
            path,
            description,
        )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    print_heading(
        "SenseFuzeAI Missing-Modality Robustness Experiment"
    )

    print(
        f"Project root : {ROOT_DIR}"
    )
    print(
        f"Dataset      : {DATASET_PATH}"
    )
    print(
        f"Schema       : {FEATURE_SCHEMA_PATH}"
    )
    print(
        f"Model        : {DEPLOYED_MODEL_PATH}"
    )
    print(
        f"Output       : {output_dir}"
    )

    print()
    print(
        "Experiment type:"
    )
    print(
        "  classifier-level synthetic zero-mask stress test"
    )

    print()
    print(
        "IMPORTANT:"
    )
    print(
        "  Production modality gating is unchanged."
    )
    print(
        "  Missing modalities are simulated only in "
        "held-out classifier feature vectors."
    )

    # =========================================================================
    # LOAD AUTHORITATIVE ARTIFACTS
    # =========================================================================

    metadata_raw = load_json(
        METADATA_PATH
    )

    if not isinstance(
        metadata_raw,
        dict,
    ):
        raise ValueError(
            "metadata.json must contain a JSON object."
        )

    source_metadata: dict[
        str,
        Any,
    ] = metadata_raw

    validate_metadata(
        source_metadata
    )

    feature_columns = (
        load_feature_schema()
    )

    feature_groups = (
        resolve_feature_groups(
            feature_columns
        )
    )

    (
        dataframe,
        X_full,
        y,
    ) = load_authoritative_dataset(
        feature_columns
    )

    (
        deployed_pipeline,
        classifier,
        preprocessing_steps,
    ) = load_deployed_pipeline()

    # =========================================================================
    # FEATURE AUDIT
    # =========================================================================

    print_heading(
        "VALIDATED FEATURE GROUPS"
    )

    print(
        f"Keystroke : {len(feature_groups['keystroke'])}"
    )
    print(
        f"Text      : {len(feature_groups['text'])}"
    )
    print(
        f"Audio     : {len(feature_groups['audio'])}"
    )
    print(
        f"Vision    : {len(feature_groups['vision'])}"
    )
    print(
        f"Derived   : {len(feature_groups['derived'])}"
    )
    print(
        "TOTAL     : "
        f"{sum(len(group) for group in feature_groups.values())}"
    )

    print()
    print(
        "Vision-dependent derived predictors:"
    )

    for feature in feature_groups[
        "derived"
    ]:
        print(
            f"  - {feature}"
        )

    # =========================================================================
    # CONDITIONS
    # =========================================================================

    conditions = build_conditions(
        feature_columns,
        feature_groups,
    )

    print_heading(
        "INFERENCE CONDITIONS"
    )

    for condition in conditions:
        print(
            f"{condition.name:28s} "
            f"input_dim={EXPECTED_TOTAL_FEATURES:4d} "
            f"zero_masked="
            f"{condition.masked_feature_count:4d}"
        )

    # =========================================================================
    # FROZEN FOLDS
    # =========================================================================

    (
        folds,
        fold_manifest,
    ) = build_frozen_folds(
        X_full,
        y,
        dataframe[
            SESSION_COL
        ],
    )

    fold_parity = (
        verify_ablation_fold_parity(
            fold_manifest
        )
    )

    print_heading(
        "CROSS-EXPERIMENT FOLD CHECK"
    )

    if fold_parity[
        "available"
    ]:
        print(
            "Section 5.6.1 ablation metadata: FOUND"
        )
        print(
            "Train/test session assignments: IDENTICAL"
        )
    else:
        print(
            "Section 5.6.1 ablation metadata: NOT FOUND"
        )
        print(
            "Folds reconstructed deterministically from "
            "the same StratifiedKFold protocol."
        )

    # =========================================================================
    # BASELINE
    # =========================================================================

    print_heading(
        "ALL-MODALITIES BASELINE REPRODUCTION"
    )

    (
        fitted_folds,
        baseline_fold_rows,
        baseline_pooled_true,
        baseline_pooled_predictions,
    ) = fit_complete_fold_models_and_baseline(
        deployed_pipeline=(
            deployed_pipeline
        ),

        X_full=(
            X_full
        ),

        y=(
            y
        ),

        folds=(
            folds
        ),

        severe_imbalance_threshold=(
            severe_imbalance_threshold
        ),
    )

    baseline_summary_metrics = (
        summarise_fold_metrics(
            baseline_fold_rows
        )
    )

    reproduction = (
        verify_baseline_reproduction(
            baseline_summary_metrics,
            source_metadata,
            tolerance=(
                reproduction_tolerance
            ),
        )
    )

    print()

    print(
        "Reference mean accuracy : "
        f"{float(source_metadata['cv_accuracy_mean']):.12f}"
    )

    print(
        "Observed mean accuracy  : "
        f"{baseline_summary_metrics['mean_accuracy']:.12f}"
    )

    print(
        "Reference mean macro-F1 : "
        f"{float(source_metadata['cv_macro_f1_mean']):.12f}"
    )

    print(
        "Observed mean macro-F1  : "
        f"{baseline_summary_metrics['mean_macro_f1']:.12f}"
    )

    if not reproduction[
        "passed"
    ]:
        failure_path = (
            output_dir
            / "missing_modality_reproduction_failure.json"
        )

        failure_payload = {
            "project": (
                "SenseFuzeAI"
            ),

            "experiment": (
                "missing_modality_robustness"
            ),

            "status": (
                "STOPPED_BEFORE_STRESS_CONDITIONS"
            ),

            "reason": (
                "Untouched all-modalities baseline "
                "did not reproduce authoritative CV metrics."
            ),

            "reproduction_check": (
                reproduction
            ),

            "dataset": {
                "path": str(
                    DATASET_PATH
                ),

                "sha256": sha256_file(
                    DATASET_PATH
                ),
            },

            "feature_schema": {
                "path": str(
                    FEATURE_SCHEMA_PATH
                ),

                "sha256": sha256_file(
                    FEATURE_SCHEMA_PATH
                ),

                "feature_count": len(
                    feature_columns
                ),
            },

            "model": {
                "path": str(
                    DEPLOYED_MODEL_PATH
                ),

                "sha256": sha256_file(
                    DEPLOYED_MODEL_PATH
                ),

                "random_forest_parameters": (
                    safe_json_value(
                        classifier.get_params(
                            deep=False
                        )
                    )
                ),
            },

            "fold_parity": (
                fold_parity
            ),
        }

        write_json(
            failure_path,
            failure_payload,
        )

        print()
        print(
            "BASELINE REPRODUCTION CHECK: FAIL"
        )
        print(
            "Missing-modality conditions were NOT executed."
        )
        print(
            f"Diagnostic evidence:\n{failure_path}"
        )

        raise SystemExit(
            1
        )

    print()
    print(
        "BASELINE REPRODUCTION CHECK: PASS"
    )

    # =========================================================================
    # BASELINE SUMMARY
    # =========================================================================

    baseline_condition = conditions[
        0
    ]

    (
        baseline_summary_row,
        baseline_diagnostics,
    ) = summarise_condition(
        condition=(
            baseline_condition
        ),

        fold_rows=(
            baseline_fold_rows
        ),

        pooled_true=(
            baseline_pooled_true
        ),

        pooled_predictions=(
            baseline_pooled_predictions
        ),

        severe_imbalance_threshold=(
            severe_imbalance_threshold
        ),
    )

    all_fold_rows = list(
        baseline_fold_rows
    )

    summary_rows = [
        baseline_summary_row
    ]

    detailed_diagnostics: dict[
        str,
        Any,
    ] = {
        baseline_condition.name: (
            baseline_diagnostics
        )
    }

    # =========================================================================
    # MISSING-MODALITY CONDITIONS
    # NO MODEL FITTING BELOW THIS POINT
    # =========================================================================

    for condition in conditions[
        1:
    ]:
        print_heading(
            condition.name
        )

        (
            condition_fold_rows,
            pooled_true,
            pooled_predictions,
        ) = evaluate_missing_condition(
            condition=(
                condition
            ),

            fitted_folds=(
                fitted_folds
            ),

            X_full=(
                X_full
            ),

            y=(
                y
            ),

            severe_imbalance_threshold=(
                severe_imbalance_threshold
            ),
        )

        all_fold_rows.extend(
            condition_fold_rows
        )

        (
            summary_row,
            diagnostics,
        ) = summarise_condition(
            condition=(
                condition
            ),

            fold_rows=(
                condition_fold_rows
            ),

            pooled_true=(
                pooled_true
            ),

            pooled_predictions=(
                pooled_predictions
            ),

            severe_imbalance_threshold=(
                severe_imbalance_threshold
            ),
        )

        summary_rows.append(
            summary_row
        )

        detailed_diagnostics[
            condition.name
        ] = diagnostics

    # =========================================================================
    # OUTPUT DATAFRAMES
    # =========================================================================

    results_dataframe = pd.DataFrame(
        summary_rows
    )

    fold_dataframe = pd.DataFrame(
        all_fold_rows
    )

    results_columns = [
        "condition",
        "input_dimensions",
        "masked_groups",
        "masked_feature_count",
        "mean_accuracy",
        "accuracy_sd",
        "mean_macro_f1",
        "macro_f1_sd",
        "mean_delta_accuracy",
        "delta_macro_f1",
        "pooled_accuracy",
        "pooled_macro_f1",
        "predicted_focused",
        "predicted_distracted",
        "predicted_fatigued",
        "predicted_overloaded",
        "dominant_predicted_class",
        "dominant_predicted_proportion",
        "unique_predicted_classes",
        "class_collapse",
        "severe_prediction_imbalance",
    ]

    fold_columns = [
        "condition",
        "fold",
        "input_dimensions",
        "masked_groups",
        "masked_feature_count",
        "train_samples",
        "test_samples",
        "accuracy",
        "macro_f1",
        "baseline_accuracy_same_fold",
        "baseline_macro_f1_same_fold",
        "delta_accuracy",
        "delta_macro_f1",
        "predicted_focused",
        "predicted_distracted",
        "predicted_fatigued",
        "predicted_overloaded",
        "dominant_predicted_class",
        "dominant_predicted_proportion",
        "unique_predicted_classes",
        "class_collapse",
        "severe_prediction_imbalance",
    ]

    results_dataframe = (
        results_dataframe[
            results_columns
        ]
    )

    fold_dataframe = (
        fold_dataframe[
            fold_columns
        ]
    )

    baseline_summary_mask = (
        results_dataframe[
            "condition"
        ]
        == "All modalities available"
    )

    results_dataframe.loc[
        baseline_summary_mask,
        "mean_delta_accuracy",
    ] = 0.0

    results_dataframe.loc[
        baseline_summary_mask,
        "delta_macro_f1",
    ] = 0.0

    baseline_fold_mask = (
        fold_dataframe[
            "condition"
        ]
        == "All modalities available"
    )

    fold_dataframe.loc[
        baseline_fold_mask,
        "delta_accuracy",
    ] = 0.0

    fold_dataframe.loc[
        baseline_fold_mask,
        "delta_macro_f1",
    ] = 0.0

    # =========================================================================
    # SAVE EVIDENCE
    # =========================================================================

    results_path = (
        output_dir
        / "missing_modality_results.csv"
    )

    fold_results_path = (
        output_dir
        / "missing_modality_fold_results.csv"
    )

    metadata_output_path = (
        output_dir
        / "missing_modality_metadata.json"
    )

    results_dataframe.to_csv(
        results_path,
        index=False,
    )

    fold_dataframe.to_csv(
        fold_results_path,
        index=False,
    )

    final_metadata = (
        build_metadata_payload(
            source_metadata=(
                source_metadata
            ),

            feature_columns=(
                feature_columns
            ),

            feature_groups=(
                feature_groups
            ),

            conditions=(
                conditions
            ),

            classifier=(
                classifier
            ),

            preprocessing_steps=(
                preprocessing_steps
            ),

            fold_manifest=(
                fold_manifest
            ),

            fold_parity=(
                fold_parity
            ),

            reproduction=(
                reproduction
            ),

            results=(
                results_dataframe
            ),

            detailed_diagnostics=(
                detailed_diagnostics
            ),

            severe_imbalance_threshold=(
                severe_imbalance_threshold
            ),

            output_dir=(
                output_dir
            ),
        )
    )

    write_json(
        metadata_output_path,
        final_metadata,
    )

    script_snapshot = (
        save_script_snapshot(
            output_dir
        )
    )

    # =========================================================================
    # FINAL REPORT
    # =========================================================================

    print_heading(
        "FINAL MISSING-MODALITY RESULTS"
    )

    print_markdown_table(
        results_dataframe
    )

    print_interpretation(
        results_dataframe,
        severe_imbalance_threshold=(
            severe_imbalance_threshold
        ),
    )

    print_heading(
        "SAVED EVIDENCE"
    )

    print(
        f"Summary results : {results_path}"
    )

    print(
        f"Fold results    : {fold_results_path}"
    )

    print(
        f"Metadata        : {metadata_output_path}"
    )

    print(
        f"Script snapshot : {script_snapshot}"
    )

    print()
    print(
        "Model fitting audit:"
    )

    print(
        f"  Complete models fitted : {len(fitted_folds)}"
    )

    print(
        "  Stress-condition refits: 0"
    )

    print()
    print(
        "FINAL RESULT: PASS"
    )


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    main()
