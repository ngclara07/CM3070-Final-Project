"""
evaluate_leave_one_modality_out.py

SenseFuzeAI
Final Leave-One-Modality-Out Ablation Experiment
Final Report, Chapter 5, Sub-section 5.6.1

=============================================================================
PURPOSE
=============================================================================

Evaluate the predictive contribution of each SenseFuzeAI modality by
retraining the deployed Random Forest classifier after removing one modality
at a time.

Configurations:

    1. Full fusion
    2. Fusion - Keystroke
    3. Fusion - Text
    4. Fusion - Audio
    5. Fusion - Vision

Current final fusion representation:

    keystroke                         22
    MPNet text                       768
    audio                            809
    CLIP vision                      768
    webcam-derived visual features    6
                                    ----
                                    2373

The six image_webcam_* predictors are generated from the visual/CLIP branch.
They are therefore removed together with the 768 CLIP predictors when Vision
is ablated.

=============================================================================
AUTHORITATIVE ARTIFACTS
=============================================================================

Dataset:
    models/fusion_demo/fusion_training_dataset.csv

Feature schema:
    models/fusion_demo/feature_columns.json

Training metadata:
    models/fusion_demo/metadata.json

Deployed model:
    models/fusion_demo/fusion_pipeline.joblib

=============================================================================
VALIDATION PROTOCOL
=============================================================================

Records:
    309

Classes:
    focused       77
    distracted    77
    fatigued      77
    overloaded    78

Cross-validation:
    StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=42,
    )

Primary metric:
    macro-F1

Complementary metric:
    accuracy

Variability:
    population standard deviation across five folds, ddof=0

All five configurations use the same frozen fold assignments.

Every reduced configuration is retrained from scratch on the remaining
features. No trained estimator from Full fusion is reused for an ablation.

=============================================================================
REPRODUCTION GUARD
=============================================================================

Full fusion is evaluated first.

The resulting:

    mean accuracy
    accuracy SD
    mean macro-F1
    macro-F1 SD

must reproduce the corresponding values stored in:

    models/fusion_demo/metadata.json

If reproduction fails, the script stops before executing ablated conditions.

=============================================================================
OUTPUT
=============================================================================

Default directory:

    data/processed/final_experiments/leave_one_modality_out/

Generated files:

    ablation_results.csv
    ablation_fold_results.csv
    ablation_metadata.json
    evaluate_leave_one_modality_out.py

=============================================================================
RUN
=============================================================================

Syntax check:

    python -m py_compile evaluate_leave_one_modality_out.py

Experiment:

    python evaluate_leave_one_modality_out.py

=============================================================================
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
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline


# =============================================================================
# PROJECT PATHS
# =============================================================================

ROOT_DIR = Path(__file__).resolve().parent

FUSION_DIR = ROOT_DIR / "models" / "fusion_demo"

DATASET_PATH = FUSION_DIR / "fusion_training_dataset.csv"

FEATURE_SCHEMA_PATH = FUSION_DIR / "feature_columns.json"

METADATA_PATH = FUSION_DIR / "metadata.json"

DEPLOYED_MODEL_PATH = FUSION_DIR / "fusion_pipeline.joblib"

DEFAULT_OUTPUT_DIR = (
    ROOT_DIR
    / "data"
    / "processed"
    / "final_experiments"
    / "leave_one_modality_out"
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

EXPECTED_CONFIGURATION_DIMENSIONS = {
    "Full fusion": 2373,
    "Fusion - Keystroke": 2351,
    "Fusion - Text": 1605,
    "Fusion - Audio": 1564,
    "Fusion - Vision": 1599,
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

# train_fusion_demo_pipeline.py uses numpy.std() with its default ddof=0.
FOLD_SD_DDOF = 0

REPRODUCTION_TOLERANCE = 1e-10


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass(frozen=True)
class AblationConfiguration:
    """Definition of one ablation configuration."""

    name: str
    removed_groups: tuple[str, ...]
    removed_features: tuple[str, ...]
    retained_features: tuple[str, ...]
    dimension: int


# =============================================================================
# GENERAL UTILITIES
# =============================================================================

def print_heading(title: str) -> None:
    print()
    print("=" * 96)
    print(title)
    print("=" * 96)


def require_file(path: Path, description: str) -> None:
    if not path.exists():
        raise FileNotFoundError(
            f"{description} not found:\n{path}"
        )

    if not path.is_file():
        raise ValueError(
            f"{description} is not a file:\n{path}"
        )


def normalise_label(value: Any) -> str:
    if pd.isna(value):
        return ""

    return str(value).strip().lower()


def load_json(path: Path) -> Any:
    require_file(
        path,
        "JSON artifact",
    )

    with path.open(
        "r",
        encoding="utf-8",
    ) as file_handle:
        return json.load(file_handle)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as file_handle:
        while True:
            chunk = file_handle.read(1024 * 1024)

            if not chunk:
                break

            digest.update(chunk)

    return digest.hexdigest()


def safe_json_value(value: Any) -> Any:
    """
    Convert common Python/sklearn values into JSON-safe values.
    """

    if value is None:
        return None

    if isinstance(
        value,
        (
            str,
            int,
            float,
            bool,
        ),
    ):
        if isinstance(value, float) and not math.isfinite(value):
            return str(value)

        return value

    if isinstance(value, Path):
        return str(value)

    if isinstance(value, dict):
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


def current_python_version() -> str:
    return (
        f"{sys.version_info.major}."
        f"{sys.version_info.minor}."
        f"{sys.version_info.micro}"
    )


# =============================================================================
# AUTHORITATIVE METADATA
# =============================================================================

def validate_metadata(
    metadata: dict[str, Any],
) -> None:
    """
    Validate that metadata describes the expected final 2,373-feature
    Random Forest fusion system.
    """

    if not isinstance(metadata, dict):
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
            "The authoritative fusion metadata does not identify "
            "Random Forest as the selected model.\n"
            f"Observed selected model: {selected_model!r}"
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
            "Fusion metadata class ordering differs from "
            "the canonical experiment protocol.\n"
            f"Expected: {CLASSES}\n"
            f"Observed: {metadata_classes}"
        )

    class_distribution = {
        normalise_label(key): int(value)
        for key, value in metadata.get(
            "class_distribution",
            {},
        ).items()
    }

    if class_distribution != EXPECTED_CLASS_COUNTS:
        raise ValueError(
            "Fusion metadata class distribution mismatch.\n"
            f"Expected: {EXPECTED_CLASS_COUNTS}\n"
            f"Observed: {class_distribution}"
        )

    webcam_metadata = metadata.get(
        "webcam_image_calibration",
        {},
    )

    if not isinstance(webcam_metadata, dict):
        raise ValueError(
            "webcam_image_calibration metadata must be an object."
        )

    if not bool(
        webcam_metadata.get(
            "enabled",
            False,
        )
    ):
        raise ValueError(
            "The authoritative 2,373-feature fusion metadata does not "
            "report webcam visual augmentation as enabled."
        )

    added_features = [
        str(value)
        for value in webcam_metadata.get(
            "calibration_features_added",
            [],
        )
    ]

    if added_features != EXPECTED_DERIVED_FEATURES:
        raise ValueError(
            "Unexpected webcam-derived feature definition.\n"
            f"Expected: {EXPECTED_DERIVED_FEATURES}\n"
            f"Observed: {added_features}"
        )

    required_cv_fields = [
        "cv_accuracy_mean",
        "cv_accuracy_std",
        "cv_macro_f1_mean",
        "cv_macro_f1_std",
    ]

    missing_cv_fields = [
        field
        for field in required_cv_fields
        if field not in metadata
    ]

    if missing_cv_fields:
        raise ValueError(
            "Fusion metadata is missing required CV result fields:\n"
            f"{missing_cv_fields}"
        )


# =============================================================================
# FEATURE SCHEMA
# =============================================================================

def load_feature_schema() -> list[str]:
    value = load_json(
        FEATURE_SCHEMA_PATH
    )

    if not isinstance(value, list):
        raise ValueError(
            "Fusion feature schema must be a JSON list."
        )

    columns = [
        str(column).strip()
        for column in value
    ]

    if len(columns) != EXPECTED_TOTAL_FEATURES:
        raise ValueError(
            "Fusion feature-schema dimension mismatch.\n"
            f"Expected: {EXPECTED_TOTAL_FEATURES}\n"
            f"Observed: {len(columns)}"
        )

    if any(not column for column in columns):
        raise ValueError(
            "Fusion schema contains an empty feature name."
        )

    if len(set(columns)) != len(columns):
        counts = pd.Series(columns).value_counts()

        duplicates = [
            str(name)
            for name, count in counts.items()
            if int(count) > 1
        ]

        raise ValueError(
            "Fusion schema contains duplicate feature names.\n"
            f"Examples: {duplicates[:20]}"
        )

    return columns


def resolve_feature_groups(
    feature_columns: list[str],
) -> dict[str, list[str]]:
    """
    Resolve feature groups from explicit persisted names.

    No positional slicing is used.
    """

    groups = {
        "keystroke": [
            column
            for column in feature_columns
            if column.startswith("keystroke_")
        ],

        "text": [
            column
            for column in feature_columns
            if column.startswith("text_mpnet_emb_")
        ],

        "audio": [
            column
            for column in feature_columns
            if column.startswith("audio_")
        ],

        "vision": [
            column
            for column in feature_columns
            if column.startswith("image_clip_emb_")
        ],

        "derived": [
            column
            for column in feature_columns
            if column.startswith("image_webcam_")
        ],
    }

    for group_name, expected_count in EXPECTED_GROUP_COUNTS.items():
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

    if groups["derived"] != EXPECTED_DERIVED_FEATURES:
        raise ValueError(
            "The six derived visual predictors do not match "
            "the authoritative expected feature list.\n"
            f"Expected: {EXPECTED_DERIVED_FEATURES}\n"
            f"Observed: {groups['derived']}"
        )

    observed_membership: list[str] = []

    for group_columns in groups.values():
        observed_membership.extend(
            group_columns
        )

    if len(observed_membership) != EXPECTED_TOTAL_FEATURES:
        grouped_set = set(
            observed_membership
        )

        unclassified = [
            column
            for column in feature_columns
            if column not in grouped_set
        ]

        raise ValueError(
            "The persisted schema is not fully explained by "
            "the validated feature groups.\n"
            f"Grouped feature count: {len(observed_membership)}\n"
            f"Expected: {EXPECTED_TOTAL_FEATURES}\n"
            f"Unclassified examples: {unclassified[:30]}"
        )

    if len(set(observed_membership)) != EXPECTED_TOTAL_FEATURES:
        raise ValueError(
            "One or more persisted features were assigned "
            "to multiple modality groups."
        )

    if set(observed_membership) != set(feature_columns):
        raise ValueError(
            "Resolved feature-group membership does not exactly "
            "match the persisted fusion schema."
        )

    return groups


# =============================================================================
# AUTHORITATIVE DATASET
# =============================================================================

def clean_numeric_dataframe(
    dataframe: pd.DataFrame,
    feature_columns: list[str],
) -> pd.DataFrame:
    """
    Reproduce deterministic numeric cleaning used by the final
    fusion training pipeline.

    Operations:
        numeric coercion
        +/- infinity -> NaN
        NaN -> 0.0

    No learned preprocessing is performed here.
    """

    output = dataframe[
        feature_columns
    ].copy()

    for column in output.columns:
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

    output = output.fillna(0.0)

    values = output.to_numpy(
        dtype=np.float64
    )

    if not np.all(
        np.isfinite(values)
    ):
        raise ValueError(
            "Non-finite values remain after canonical numeric cleaning."
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

    required_non_features = {
        SESSION_COL,
        LABEL_COL,
    }

    missing_non_features = (
        required_non_features
        - set(dataframe.columns)
    )

    if missing_non_features:
        raise ValueError(
            "Authoritative fusion dataset is missing required "
            "identifier columns.\n"
            f"Missing: {sorted(missing_non_features)}"
        )

    missing_features = [
        column
        for column in feature_columns
        if column not in dataframe.columns
    ]

    if missing_features:
        raise ValueError(
            "Authoritative dataset does not contain the complete "
            "persisted fusion feature schema.\n"
            f"Missing count: {len(missing_features)}\n"
            f"Examples: {missing_features[:30]}"
        )

    dataset_feature_columns = [
        column
        for column in dataframe.columns
        if column not in {
            SESSION_COL,
            LABEL_COL,
        }
    ]

    if dataset_feature_columns != feature_columns:
        raise ValueError(
            "Feature order in fusion_training_dataset.csv does not "
            "exactly match models/fusion_demo/feature_columns.json.\n"
            "The experiment will not silently reorder an incompatible "
            "training artifact."
        )

    if dataframe[SESSION_COL].isna().any():
        raise ValueError(
            "Authoritative dataset contains missing session IDs."
        )

    session_ids = (
        dataframe[SESSION_COL]
        .astype(str)
    )

    if session_ids.duplicated().any():
        duplicate_mask = session_ids.duplicated(
            keep=False
        )

        duplicates = (
            session_ids[
                duplicate_mask
            ]
            .unique()
            .tolist()
        )

        raise ValueError(
            "Authoritative dataset contains duplicate session IDs.\n"
            f"Examples: {duplicates[:20]}"
        )

    labels = (
        dataframe[LABEL_COL]
        .apply(normalise_label)
    )

    if (labels == "").any():
        raise ValueError(
            "Authoritative dataset contains an empty label."
        )

    unexpected_labels = (
        set(labels.unique())
        - set(CLASSES)
    )

    if unexpected_labels:
        raise ValueError(
            "Unexpected behavioural labels in authoritative dataset.\n"
            f"Observed unexpected labels: {sorted(unexpected_labels)}"
        )

    class_counts = {
        label: int(
            (labels == label).sum()
        )
        for label in CLASSES
    }

    if class_counts != EXPECTED_CLASS_COUNTS:
        raise ValueError(
            "Authoritative dataset class-distribution mismatch.\n"
            f"Expected: {EXPECTED_CLASS_COUNTS}\n"
            f"Observed: {class_counts}"
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
# DEPLOYED RANDOM FOREST
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

    if not isinstance(model, Pipeline):
        raise TypeError(
            "Expected deployed fusion artifact to be an sklearn Pipeline.\n"
            f"Observed: {type(model)}"
        )

    random_forests: list[
        tuple[str, RandomForestClassifier]
    ] = []

    preprocessing_steps: list[
        dict[str, Any]
    ] = []

    for name, step in model.steps:
        if isinstance(
            step,
            RandomForestClassifier,
        ):
            random_forests.append(
                (
                    name,
                    step,
                )
            )
        else:
            step_metadata = {
                "name": name,
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

            preprocessing_steps.append(
                step_metadata
            )

    if len(random_forests) != 1:
        raise ValueError(
            "Deployed fusion pipeline must contain exactly one "
            "RandomForestClassifier.\n"
            f"Observed count: {len(random_forests)}"
        )

    _, classifier = random_forests[0]

    params = classifier.get_params(
        deep=False
    )

    mandatory_checks = {
        "n_estimators": 500,
        "class_weight": "balanced",
        "random_state": 42,
        "n_jobs": -1,
    }

    mismatches: dict[
        str,
        dict[str, Any],
    ] = {}

    for parameter, expected in mandatory_checks.items():
        observed = params.get(
            parameter
        )

        if observed != expected:
            mismatches[parameter] = {
                "expected": expected,
                "observed": observed,
            }

    if mismatches:
        raise ValueError(
            "Deployed Random Forest does not match the "
            "authoritative final-fusion configuration.\n"
            + json.dumps(
                mismatches,
                indent=2,
            )
        )

    return (
        model,
        classifier,
        preprocessing_steps,
    )


# =============================================================================
# ABLATION CONFIGURATIONS
# =============================================================================

def build_configurations(
    feature_columns: list[str],
    groups: dict[str, list[str]],
) -> list[AblationConfiguration]:
    group_removals: list[
        tuple[
            str,
            tuple[str, ...],
        ]
    ] = [
        (
            "Full fusion",
            (),
        ),
        (
            "Fusion - Keystroke",
            (
                "keystroke",
            ),
        ),
        (
            "Fusion - Text",
            (
                "text",
            ),
        ),
        (
            "Fusion - Audio",
            (
                "audio",
            ),
        ),
        (
            "Fusion - Vision",
            (
                "vision",
                "derived",
            ),
        ),
    ]

    configurations: list[
        AblationConfiguration
    ] = []

    for (
        configuration_name,
        removed_group_names,
    ) in group_removals:
        removed_set: set[str] = set()

        for group_name in removed_group_names:
            removed_set.update(
                groups[group_name]
            )

        retained = [
            column
            for column in feature_columns
            if column not in removed_set
        ]

        removed = [
            column
            for column in feature_columns
            if column in removed_set
        ]

        expected_dimension = (
            EXPECTED_CONFIGURATION_DIMENSIONS[
                configuration_name
            ]
        )

        if len(retained) != expected_dimension:
            raise ValueError(
                "Ablation dimension mismatch.\n"
                f"Configuration: {configuration_name}\n"
                f"Expected: {expected_dimension}\n"
                f"Observed: {len(retained)}"
            )

        configurations.append(
            AblationConfiguration(
                name=configuration_name,
                removed_groups=tuple(
                    removed_group_names
                ),
                removed_features=tuple(
                    removed
                ),
                retained_features=tuple(
                    retained
                ),
                dimension=len(
                    retained
                ),
            )
        )

    return configurations


# =============================================================================
# CROSS-VALIDATION
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
    """
    Generate the fold assignments once.

    These exact train/test indices are reused by all ablation conditions.
    """

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
            "Unexpected number of generated CV folds."
        )

    manifest: list[
        dict[str, Any]
    ] = []

    for fold_index, (
        train_indices,
        test_indices,
    ) in enumerate(
        folds,
        start=1,
    ):
        train_labels = y.iloc[
            train_indices
        ]

        test_labels = y.iloc[
            test_indices
        ]

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
                "Train/test session overlap detected in fold "
                f"{fold_index}: {sorted(overlap)[:20]}"
            )

        manifest.append(
            {
                "fold": fold_index,

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


def evaluate_configuration(
    configuration: AblationConfiguration,
    deployed_pipeline: Pipeline,
    X_full: pd.DataFrame,
    y: pd.Series,
    folds: list[
        tuple[
            np.ndarray,
            np.ndarray,
        ]
    ],
) -> list[dict[str, Any]]:
    """
    Retrain and evaluate one feature configuration over the frozen folds.
    """

    X = X_full.loc[
        :,
        list(
            configuration.retained_features
        ),
    ]

    if X.shape[1] != configuration.dimension:
        raise RuntimeError(
            "Runtime feature dimension differs from "
            "validated configuration definition."
        )

    fold_rows: list[
        dict[str, Any]
    ] = []

    for fold_number, (
        train_indices,
        test_indices,
    ) in enumerate(
        folds,
        start=1,
    ):
        # Fresh unfitted clone for every fold.
        model = clone(
            deployed_pipeline
        )

        X_train = X.iloc[
            train_indices
        ]

        X_test = X.iloc[
            test_indices
        ]

        y_train = y.iloc[
            train_indices
        ]

        y_test = y.iloc[
            test_indices
        ]

        model.fit(
            X_train,
            y_train,
        )

        predictions = model.predict(
            X_test
        )

        accuracy = accuracy_score(
            y_test,
            predictions,
        )

        macro_f1 = f1_score(
            y_test,
            predictions,
            labels=CLASSES,
            average="macro",
            zero_division=0,
        )

        fold_rows.append(
            {
                "configuration": (
                    configuration.name
                ),

                "fold": int(
                    fold_number
                ),

                "dimensions": int(
                    configuration.dimension
                ),

                "removed_groups": (
                    "+".join(
                        configuration.removed_groups
                    )
                    if configuration.removed_groups
                    else ""
                ),

                "removed_feature_count": int(
                    len(
                        configuration.removed_features
                    )
                ),

                "train_samples": int(
                    len(train_indices)
                ),

                "test_samples": int(
                    len(test_indices)
                ),

                "accuracy": float(
                    accuracy
                ),

                "macro_f1": float(
                    macro_f1
                ),
            }
        )

        print(
            f"  Fold {fold_number}: "
            f"accuracy={accuracy:.4f}, "
            f"macro-F1={macro_f1:.4f}"
        )

    return fold_rows


# =============================================================================
# SUMMARY METRICS
# =============================================================================

def summarise_configuration(
    configuration: AblationConfiguration,
    fold_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    accuracies = np.asarray(
        [
            row["accuracy"]
            for row in fold_rows
        ],
        dtype=np.float64,
    )

    macro_f1_scores = np.asarray(
        [
            row["macro_f1"]
            for row in fold_rows
        ],
        dtype=np.float64,
    )

    if len(accuracies) != CV_SPLITS:
        raise RuntimeError(
            "Unexpected fold count while summarising configuration."
        )

    if len(macro_f1_scores) != CV_SPLITS:
        raise RuntimeError(
            "Unexpected macro-F1 fold count while summarising configuration."
        )

    return {
        "configuration": (
            configuration.name
        ),

        "dimensions": int(
            configuration.dimension
        ),

        "removed_groups": (
            "+".join(
                configuration.removed_groups
            )
            if configuration.removed_groups
            else ""
        ),

        "removed_feature_count": int(
            len(
                configuration.removed_features
            )
        ),

        "mean_accuracy": float(
            np.mean(
                accuracies
            )
        ),

        "accuracy_sd": float(
            np.std(
                accuracies,
                ddof=FOLD_SD_DDOF,
            )
        ),

        "mean_macro_f1": float(
            np.mean(
                macro_f1_scores
            )
        ),

        "macro_f1_sd": float(
            np.std(
                macro_f1_scores,
                ddof=FOLD_SD_DDOF,
            )
        ),
    }


# =============================================================================
# FULL-FUSION REPRODUCTION CHECK
# =============================================================================

def compare_metric(
    *,
    metric_name: str,
    observed: float,
    expected: float,
    tolerance: float,
) -> Optional[dict[str, float]]:
    """
    Compare one reproduced metric against its authoritative reference.

    Returns:
        None when the metric reproduces within tolerance.

        Otherwise a dictionary describing the mismatch.
    """

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


def verify_full_fusion_reproduction(
    full_summary: dict[str, Any],
    metadata: dict[str, Any],
    *,
    tolerance: float,
) -> dict[str, Any]:
    """
    Confirm that newly computed Full-fusion CV scores reproduce
    the saved final-training metadata.
    """

    comparisons = {
        "mean_accuracy": (
            float(
                full_summary[
                    "mean_accuracy"
                ]
            ),
            float(
                metadata[
                    "cv_accuracy_mean"
                ]
            ),
        ),

        "accuracy_sd": (
            float(
                full_summary[
                    "accuracy_sd"
                ]
            ),
            float(
                metadata[
                    "cv_accuracy_std"
                ]
            ),
        ),

        "mean_macro_f1": (
            float(
                full_summary[
                    "mean_macro_f1"
                ]
            ),
            float(
                metadata[
                    "cv_macro_f1_mean"
                ]
            ),
        ),

        "macro_f1_sd": (
            float(
                full_summary[
                    "macro_f1_sd"
                ]
            ),
            float(
                metadata[
                    "cv_macro_f1_std"
                ]
            ),
        ),
    }

    failures: dict[
        str,
        dict[str, float],
    ] = {}

    for metric_name, (
        observed,
        expected,
    ) in comparisons.items():
        failure = compare_metric(
            metric_name=metric_name,
            observed=observed,
            expected=expected,
            tolerance=tolerance,
        )

        if failure is not None:
            failures[
                metric_name
            ] = failure

    return {
        "passed": (
            not failures
        ),

        "tolerance": float(
            tolerance
        ),

        "reference_source": str(
            METADATA_PATH
        ),

        "reference": {
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
        },

        "observed": {
            "mean_accuracy": float(
                full_summary[
                    "mean_accuracy"
                ]
            ),

            "accuracy_sd": float(
                full_summary[
                    "accuracy_sd"
                ]
            ),

            "mean_macro_f1": float(
                full_summary[
                    "mean_macro_f1"
                ]
            ),

            "macro_f1_sd": float(
                full_summary[
                    "macro_f1_sd"
                ]
            ),
        },

        "failures": failures,
    }


# =============================================================================
# OUTPUT HELPERS
# =============================================================================

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
            safe_json_value(
                payload
            ),
            file_handle,
            indent=2,
            ensure_ascii=False,
        )


def save_script_snapshot(
    output_dir: Path,
) -> Path:
    source_path = Path(
        __file__
    ).resolve()

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


def print_markdown_table(
    summary_dataframe: pd.DataFrame,
) -> None:
    print()

    print(
        "| Configuration | Dimensions | Mean Accuracy | Accuracy SD | "
        "Mean Macro-F1 | Macro-F1 SD | Δ Macro-F1 |"
    )

    print(
        "|---|---:|---:|---:|---:|---:|---:|"
    )

    for _, row in summary_dataframe.iterrows():
        print(
            "| "
            f"{row['configuration']} | "
            f"{int(row['dimensions'])} | "
            f"{float(row['mean_accuracy']):.4f} | "
            f"{float(row['accuracy_sd']):.4f} | "
            f"{float(row['mean_macro_f1']):.4f} | "
            f"{float(row['macro_f1_sd']):.4f} | "
            f"{float(row['delta_macro_f1']):.4f} |"
        )


def print_interpretation(
    summary_dataframe: pd.DataFrame,
) -> None:
    ablated = (
        summary_dataframe[
            summary_dataframe[
                "configuration"
            ]
            != "Full fusion"
        ]
        .copy()
    )

    if ablated.empty:
        raise RuntimeError(
            "No ablated configurations are available for interpretation."
        )

    largest_row = (
        ablated
        .sort_values(
            "delta_macro_f1",
            ascending=False,
        )
        .iloc[0]
    )

    smallest_row = (
        ablated
        .sort_values(
            "delta_macro_f1",
            ascending=True,
        )
        .iloc[0]
    )

    keystroke_matches = (
        ablated[
            ablated[
                "configuration"
            ]
            == "Fusion - Keystroke"
        ]
    )

    if keystroke_matches.empty:
        raise RuntimeError(
            "Fusion - Keystroke result is missing."
        )

    keystroke_row = (
        keystroke_matches.iloc[0]
    )

    improved = (
        ablated[
            ablated[
                "delta_macro_f1"
            ]
            < 0.0
        ]
        .copy()
    )

    print()
    print(
        "Factual interpretation"
    )

    print(
        "----------------------"
    )

    print(
        "Largest predictive degradation: "
        f"{largest_row['configuration']} "
        f"(Δ macro-F1 = "
        f"{float(largest_row['delta_macro_f1']):.4f})."
    )

    print(
        "Smallest degradation / greatest relative result: "
        f"{smallest_row['configuration']} "
        f"(Δ macro-F1 = "
        f"{float(smallest_row['delta_macro_f1']):.4f})."
    )

    print(
        "Leave-keystroke-out Δ macro-F1: "
        f"{float(keystroke_row['delta_macro_f1']):.4f}."
    )

    if improved.empty:
        print(
            "No ablation produced a higher mean macro-F1 "
            "than Full fusion."
        )
    else:
        descriptions = [
            (
                f"{row['configuration']} "
                f"(Δ={float(row['delta_macro_f1']):.4f})"
            )
            for _, row in improved.iterrows()
        ]

        print(
            "Ablation configuration(s) with higher mean macro-F1 "
            "than Full fusion: "
            + ", ".join(
                descriptions
            )
            + "."
        )

    print()

    print(
        "Interpretation is restricted to predictive contribution "
        "within this dataset; no causal behavioural importance "
        "is inferred."
    )


# =============================================================================
# METADATA
# =============================================================================

def build_metadata_payload(
    *,
    source_metadata: dict[str, Any],
    feature_columns: list[str],
    feature_groups: dict[str, list[str]],
    configurations: list[AblationConfiguration],
    classifier: RandomForestClassifier,
    preprocessing_steps: list[dict[str, Any]],
    fold_manifest: list[dict[str, Any]],
    reproduction: dict[str, Any],
    summary_dataframe: pd.DataFrame,
    output_dir: Path,
) -> dict[str, Any]:
    derived_dependency_decisions = {
        feature: {
            "depends_on": [
                "vision",
            ],

            "rule": (
                "The feature is produced by the webcam-calibrated "
                "image classifier from the session's CLIP image "
                "embedding; therefore it is removed whenever the "
                "vision modality is ablated."
            ),
        }
        for feature in EXPECTED_DERIVED_FEATURES
    }

    configuration_metadata: dict[
        str,
        dict[str, Any],
    ] = {}

    for configuration in configurations:
        configuration_metadata[
            configuration.name
        ] = {
            "dimension": (
                configuration.dimension
            ),

            "removed_groups": list(
                configuration.removed_groups
            ),

            "removed_feature_count": len(
                configuration.removed_features
            ),

            "removed_features": list(
                configuration.removed_features
            ),
        }

    return {
        "project": (
            "SenseFuzeAI"
        ),

        "experiment": (
            "leave_one_modality_out_ablation"
        ),

        "report_section": (
            "5.6.1"
        ),

        "created_at": (
            datetime.now()
            .astimezone()
            .isoformat()
        ),

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
                "CPU only"
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

            "feature_group_counts": {
                group: len(columns)
                for group, columns
                in feature_groups.items()
            },

            "feature_groups": (
                feature_groups
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

            "retraining_policy": (
                "A fresh sklearn clone of the deployed final "
                "pipeline is fitted independently inside every "
                "training fold for every configuration."
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

            "fold_assignments_identical_across_configurations": (
                True
            ),

            "fold_standard_deviation_ddof": (
                FOLD_SD_DDOF
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

            "variability": (
                "population standard deviation "
                "across the five fold scores"
            ),
        },

        "derived_feature_dependency_decisions": (
            derived_dependency_decisions
        ),

        "configurations": (
            configuration_metadata
        ),

        "full_fusion_reproduction_check": (
            reproduction
        ),

        "source_training_metadata": {
            "metadata_path": str(
                METADATA_PATH
            ),

            "metadata_sha256": (
                sha256_file(
                    METADATA_PATH
                )
            ),

            "reference_cv_accuracy_mean": (
                source_metadata.get(
                    "cv_accuracy_mean"
                )
            ),

            "reference_cv_accuracy_std": (
                source_metadata.get(
                    "cv_accuracy_std"
                )
            ),

            "reference_cv_macro_f1_mean": (
                source_metadata.get(
                    "cv_macro_f1_mean"
                )
            ),

            "reference_cv_macro_f1_std": (
                source_metadata.get(
                    "cv_macro_f1_std"
                )
            ),
        },

        "results": (
            summary_dataframe.to_dict(
                orient="records"
            )
        ),

        "output_directory": str(
            output_dir
        ),

        "interpretation_constraint": (
            "Results describe predictive contribution within "
            "this dataset only and must not be interpreted as "
            "causal behavioural importance."
        ),
    }


# =============================================================================
# COMMAND-LINE ARGUMENTS
# =============================================================================

def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run the final SenseFuzeAI leakage-safe "
            "leave-one-modality-out ablation experiment."
        )
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=(
            "Directory for ablation CSV/JSON evidence."
        ),
    )

    parser.add_argument(
        "--reproduction-tolerance",
        type=float,
        default=REPRODUCTION_TOLERANCE,
        help=(
            "Absolute tolerance used when comparing the "
            "recomputed Full-fusion CV metrics with the "
            "saved training metadata."
        ),
    )

    parser.add_argument(
        "--allow-python-version-mismatch",
        action="store_true",
        help=(
            "Permit execution under a Python version other "
            "than the documented report version 3.11.9."
        ),
    )

    return parser


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    parser = build_argument_parser()
    args = parser.parse_args()

    output_dir = (
        Path(
            args.output_dir
        )
        .resolve()
    )

    reproduction_tolerance = float(
        args.reproduction_tolerance
    )

    if reproduction_tolerance < 0.0:
        parser.error(
            "--reproduction-tolerance must be non-negative."
        )

    if (
        current_python_version()
        != EXPECTED_PYTHON_VERSION
        and
        not args.allow_python_version_mismatch
    ):
        raise RuntimeError(
            "Python environment differs from the documented "
            "final experiment protocol.\n"
            f"Expected: {EXPECTED_PYTHON_VERSION}\n"
            f"Observed: {current_python_version()}\n"
            "Use --allow-python-version-mismatch only if this "
            "difference is intentional and documented."
        )

    # -------------------------------------------------------------------------
    # Required artifact checks
    # -------------------------------------------------------------------------

    for path, description in [
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
        "SenseFuzeAI Leave-One-Modality-Out Ablation"
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

    # -------------------------------------------------------------------------
    # Load authoritative artifacts
    # -------------------------------------------------------------------------

    source_metadata_raw = load_json(
        METADATA_PATH
    )

    if not isinstance(
        source_metadata_raw,
        dict,
    ):
        raise ValueError(
            "models/fusion_demo/metadata.json must contain a JSON object."
        )

    source_metadata: dict[
        str,
        Any,
    ] = source_metadata_raw

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

    # -------------------------------------------------------------------------
    # Informational feature audit
    # -------------------------------------------------------------------------

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
        f"{sum(len(v) for v in feature_groups.values())}"
    )

    print()
    print(
        "Derived visual predictors:"
    )

    for feature in feature_groups[
        "derived"
    ]:
        print(
            f"  - {feature}"
        )

    print()

    print(
        "Dependency rule: all six image_webcam_* predictors "
        "depend on the CLIP/image branch and are therefore "
        "removed with Vision."
    )

    # -------------------------------------------------------------------------
    # Build configurations
    # -------------------------------------------------------------------------

    configurations = (
        build_configurations(
            feature_columns,
            feature_groups,
        )
    )

    print_heading(
        "ABLATION CONFIGURATIONS"
    )

    for configuration in configurations:
        print(
            f"{configuration.name:22s} "
            f"dimension={configuration.dimension:4d} "
            f"removed={len(configuration.removed_features):4d}"
        )

    # -------------------------------------------------------------------------
    # One frozen fold definition shared by all configurations
    # -------------------------------------------------------------------------

    folds, fold_manifest = (
        build_frozen_folds(
            X_full,
            y,
            dataframe[
                SESSION_COL
            ],
        )
    )

    # -------------------------------------------------------------------------
    # Full fusion reproduction first
    # -------------------------------------------------------------------------

    full_configuration = (
        configurations[0]
    )

    print_heading(
        "FULL FUSION REPRODUCTION"
    )

    full_fold_rows = (
        evaluate_configuration(
            full_configuration,
            deployed_pipeline,
            X_full,
            y,
            folds,
        )
    )

    full_summary = (
        summarise_configuration(
            full_configuration,
            full_fold_rows,
        )
    )

    reproduction = (
        verify_full_fusion_reproduction(
            full_summary,
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
        f"{float(full_summary['mean_accuracy']):.12f}"
    )

    print(
        "Reference mean macro-F1 : "
        f"{float(source_metadata['cv_macro_f1_mean']):.12f}"
    )

    print(
        "Observed mean macro-F1  : "
        f"{float(full_summary['mean_macro_f1']):.12f}"
    )

    if not reproduction["passed"]:
        failure_path = (
            output_dir
            / "ablation_reproduction_failure.json"
        )

        failure_payload = {
            "project": (
                "SenseFuzeAI"
            ),

            "experiment": (
                "leave_one_modality_out_ablation"
            ),

            "status": (
                "STOPPED_BEFORE_ABLATION"
            ),

            "reason": (
                "Full-fusion cross-validation did not "
                "reproduce the authoritative saved metrics."
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
        }

        write_json(
            failure_path,
            failure_payload,
        )

        print()
        print(
            "REPRODUCTION CHECK: FAIL"
        )

        print(
            "Ablated configurations were NOT executed."
        )

        print(
            f"Diagnostic report:\n{failure_path}"
        )

        raise SystemExit(1)

    print()
    print(
        "REPRODUCTION CHECK: PASS"
    )

    # -------------------------------------------------------------------------
    # Run reduced configurations only after reproduction passes
    # -------------------------------------------------------------------------

    all_fold_rows = list(
        full_fold_rows
    )

    summary_rows = [
        full_summary
    ]

    for configuration in configurations[1:]:
        print_heading(
            configuration.name
        )

        fold_rows = (
            evaluate_configuration(
                configuration,
                deployed_pipeline,
                X_full,
                y,
                folds,
            )
        )

        all_fold_rows.extend(
            fold_rows
        )

        summary_rows.append(
            summarise_configuration(
                configuration,
                fold_rows,
            )
        )

    # -------------------------------------------------------------------------
    # Delta macro-F1
    # -------------------------------------------------------------------------

    full_mean_macro_f1 = float(
        summary_rows[0][
            "mean_macro_f1"
        ]
    )

    for row in summary_rows:
        row["delta_macro_f1"] = float(
            full_mean_macro_f1
            - float(
                row[
                    "mean_macro_f1"
                ]
            )
        )

    # Avoid -0.0 in report output.
    if abs(
        float(
            summary_rows[0][
                "delta_macro_f1"
            ]
        )
    ) <= 1e-15:
        summary_rows[0][
            "delta_macro_f1"
        ] = 0.0

    # -------------------------------------------------------------------------
    # Build output DataFrames
    # -------------------------------------------------------------------------

    summary_dataframe = pd.DataFrame(
        summary_rows
    )

    fold_dataframe = pd.DataFrame(
        all_fold_rows
    )

    summary_column_order = [
        "configuration",
        "dimensions",
        "removed_groups",
        "removed_feature_count",
        "mean_accuracy",
        "accuracy_sd",
        "mean_macro_f1",
        "macro_f1_sd",
        "delta_macro_f1",
    ]

    fold_column_order = [
        "configuration",
        "fold",
        "dimensions",
        "removed_groups",
        "removed_feature_count",
        "train_samples",
        "test_samples",
        "accuracy",
        "macro_f1",
    ]

    summary_dataframe = (
        summary_dataframe[
            summary_column_order
        ]
    )

    fold_dataframe = (
        fold_dataframe[
            fold_column_order
        ]
    )

    # -------------------------------------------------------------------------
    # Save CSV evidence
    # -------------------------------------------------------------------------

    summary_path = (
        output_dir
        / "ablation_results.csv"
    )

    fold_path = (
        output_dir
        / "ablation_fold_results.csv"
    )

    summary_dataframe.to_csv(
        summary_path,
        index=False,
    )

    fold_dataframe.to_csv(
        fold_path,
        index=False,
    )

    # -------------------------------------------------------------------------
    # Save metadata
    # -------------------------------------------------------------------------

    metadata_payload = (
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
            configurations=(
                configurations
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
            reproduction=(
                reproduction
            ),
            summary_dataframe=(
                summary_dataframe
            ),
            output_dir=(
                output_dir
            ),
        )
    )

    metadata_output_path = (
        output_dir
        / "ablation_metadata.json"
    )

    write_json(
        metadata_output_path,
        metadata_payload,
    )

    script_snapshot = (
        save_script_snapshot(
            output_dir
        )
    )

    # -------------------------------------------------------------------------
    # Final terminal report
    # -------------------------------------------------------------------------

    print_heading(
        "FINAL ABLATION RESULTS"
    )

    print_markdown_table(
        summary_dataframe
    )

    print_interpretation(
        summary_dataframe
    )

    print_heading(
        "SAVED EVIDENCE"
    )

    print(
        f"Summary results : {summary_path}"
    )

    print(
        f"Fold results    : {fold_path}"
    )

    print(
        f"Metadata        : {metadata_output_path}"
    )

    print(
        f"Script snapshot : {script_snapshot}"
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
