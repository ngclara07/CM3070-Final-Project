# === train_keystroke_dataset_comparison.py ===
#
# SenseFuzeAI
# EmoSurv IEEE + SenseFuzeAI Keystroke Dataset Comparison
#
# =============================================================================
# PURPOSE
# =============================================================================
#
# This script evaluates the supervisor-recommended dataset strategy:
#
#   1. Train an EmoSurv-only keystroke baseline.
#   2. Train a SenseFuzeAI-only keystroke model.
#   3. Add SenseFuzeAI training data to the EmoSurv training data.
#   4. Evaluate the augmented model on the EXACT SAME held-out
#      EmoSurv participants used by the EmoSurv-only baseline.
#   5. Perform additional cross-dataset generalisation experiments.
#
#
# =============================================================================
# PRIMARY SUPERVISOR-FACING EXPERIMENT
# =============================================================================
#
# Experiment A:
#
#   Train:
#       EmoSurv training participants
#
#   Test:
#       held-out EmoSurv participants
#
#
# Experiment C:
#
#   Train:
#       SAME EmoSurv training participants
#       +
#       SenseFuzeAI training sessions/groups
#
#   Test:
#       EXACT SAME held-out EmoSurv participants as Experiment A
#
#
# Primary comparison:
#
#       Delta Macro-F1
#       =
#       Macro-F1(C)
#       -
#       Macro-F1(A)
#
#
# =============================================================================
# ADDITIONAL EXPERIMENTS
# =============================================================================
#
# A:
#   EmoSurv train -> EmoSurv holdout
#
# B:
#   SenseFuzeAI train -> SenseFuzeAI holdout
#
# C:
#   EmoSurv train + SenseFuzeAI train -> SAME EmoSurv holdout as A
#
# D:
#   EmoSurv train -> SenseFuzeAI holdout
#
# E:
#   EmoSurv train + SenseFuzeAI train -> SAME SenseFuzeAI holdout as D
#
# F:
#   SenseFuzeAI train -> EmoSurv holdout
#
#
# =============================================================================
# LABEL MODES
# =============================================================================
#
# PRIMARY 3-CLASS MODE
#
#   focused
#   fatigued
#   overloaded
#
# EmoSurv mappings:
#
#   Calm  -> focused
#   Sad   -> fatigued
#   Angry -> overloaded
#
# Happy and Neutral are excluded from this primary experiment.
#
#
# EXPLORATORY 4-CLASS MODE
#
#   focused
#   distracted
#   fatigued
#   overloaded
#
# The four-class EmoSurv labels are weakly supervised proxy labels.
# They are NOT original EmoSurv behavioural-state ground truth.
#
#
# =============================================================================
# IMPORTANT METHODOLOGICAL RULES
# =============================================================================
#
# 1. One identical classifier configuration is used in every experiment.
#
# 2. Metadata columns such as:
#
#       dataset_source
#       participant_id
#       session_id
#       sample_id
#       label_origin
#
#    are NEVER used as model features.
#
# 3. EmoSurv train/test separation is participant-independent.
#
# 4. SenseFuzeAI uses participant-independent splitting where genuine
#    multiple participant IDs exist.
#
# 5. If historical SenseFuzeAI data contains only one participant,
#    session_id is used as the grouping variable so windows from the
#    same session cannot appear in both training and testing.
#
# 6. The split manifest is frozen after its first creation.
#
# 7. --rebuild-splits should only be used when the underlying dataset
#    intentionally changes. It must not be used merely to search for
#    better evaluation results.
#
# =============================================================================


from __future__ import annotations


# =============================================================================
# Imports
# =============================================================================

import argparse
import hashlib
import json
import math
import time

from pathlib import Path
from typing import Any


import joblib
import numpy as np
import pandas as pd


from sklearn.ensemble import (
    RandomForestClassifier,
)

from sklearn.impute import (
    SimpleImputer,
)

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_recall_fscore_support,
)

from sklearn.model_selection import (
    GroupShuffleSplit,
)

from sklearn.pipeline import (
    Pipeline,
)

from sklearn.preprocessing import (
    StandardScaler,
)


# =============================================================================
# Reuse the harmonised EmoSurv feature schema
# =============================================================================

from keystroke_live_gui_emosurv_ieee import (
    FEATURE_COLUMNS,
)


# =============================================================================
# Paths
# =============================================================================

ROOT_DIR = (
    Path(__file__)
    .resolve()
    .parent
)


DATASET_DIR = (
    ROOT_DIR
    / "data"
    / "processed"
    / "keystroke_dataset_comparison"
)


OUTPUT_ROOT = (
    DATASET_DIR
    / "results"
)


# =============================================================================
# Configuration
# =============================================================================

RANDOM_STATE = 42

TEST_SIZE = 0.20

BOOTSTRAP_REPETITIONS = 1000


THREE_CLASS_LABELS = (
    "focused",
    "fatigued",
    "overloaded",
)


FOUR_CLASS_LABELS = (
    "focused",
    "distracted",
    "fatigued",
    "overloaded",
)


# =============================================================================
# Fixed model
# =============================================================================

def build_fixed_model() -> Pipeline:
    """
    Build the single classifier configuration used for all experiments.

    Keeping the classifier fixed ensures that differences between
    experiments primarily reflect differences in the training dataset
    rather than differences in model architecture or hyperparameters.
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


# =============================================================================
# General utilities
# =============================================================================

def atomic_write_json(
    value: Any,
    path: Path,
) -> None:
    """
    Atomically write JSON output.
    """

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary_path = (
        path.with_suffix(
            path.suffix
            + ".tmp"
        )
    )

    temporary_path.write_text(
        json.dumps(
            value,
            indent=4,
            allow_nan=True,
        ),
        encoding="utf-8",
    )

    temporary_path.replace(
        path
    )


def dataset_signature(
    dataframe: pd.DataFrame,
) -> str:
    """
    Create a deterministic signature based on sample IDs.

    This allows the script to detect whether a frozen train/test
    split is being applied to a different dataset.
    """

    if (
        dataframe.empty
        or
        "sample_id"
        not in dataframe.columns
    ):

        return ""

    sample_ids = (
        dataframe[
            "sample_id"
        ]
        .astype(str)
        .sort_values()
        .tolist()
    )

    text = "\n".join(
        sample_ids
    )

    return (
        hashlib.sha256(
            text.encode(
                "utf-8"
            )
        )
        .hexdigest()
    )


def safe_json_number(
    value: Any,
) -> float | int | None:
    """
    Convert numeric values to JSON-safe Python values.
    """

    if value is None:

        return None

    if isinstance(
        value,
        (
            np.integer,
            int,
        ),
    ):

        return int(
            value
        )

    try:

        number = float(
            value
        )

    except (
        TypeError,
        ValueError,
    ):

        return None

    if not math.isfinite(
        number
    ):

        return None

    return number


# =============================================================================
# Label mode
# =============================================================================

def labels_for_mode(
    label_mode: str,
) -> tuple[str, ...]:

    if label_mode == "three":

        return THREE_CLASS_LABELS

    if label_mode == "four":

        return FOUR_CLASS_LABELS

    raise ValueError(
        f"Unsupported label mode: "
        f"{label_mode!r}"
    )


def dataset_paths(
    label_mode: str,
) -> tuple[
    Path,
    Path,
]:

    suffix = (
        "3class"
        if label_mode
        == "three"
        else "4class"
    )

    emosurv_path = (
        DATASET_DIR
        / (
            f"emosurv_harmonised_"
            f"{suffix}.csv"
        )
    )

    sensefuzeai_path = (
        DATASET_DIR
        / (
            f"sensefuzeai_harmonised_"
            f"{suffix}.csv"
        )
    )

    return (
        emosurv_path,
        sensefuzeai_path,
    )


# =============================================================================
# Dataset loading and validation
# =============================================================================

def load_dataset(
    path: Path,
    *,
    labels: tuple[str, ...],
    name: str,
) -> pd.DataFrame:

    if not path.exists():

        raise FileNotFoundError(
            f"{name} was not found:\n"
            f"{path}\n\n"
            "Run build_keystroke_dataset_comparison.py first."
        )

    dataframe = pd.read_csv(
        path
    )

    required_columns = {
        "dataset_source",
        "participant_id",
        "session_id",
        "sample_id",
        "source_type",
        "label",
        "label_origin",
        "original_label",
        *FEATURE_COLUMNS,
    }

    missing_columns = (
        required_columns
        -
        set(
            dataframe.columns
        )
    )

    if missing_columns:

        raise ValueError(
            f"{name} is missing required columns:\n"
            f"{sorted(missing_columns)}"
        )

    if dataframe.empty:

        raise ValueError(
            f"{name} is empty."
        )

    if dataframe[
        "sample_id"
    ].duplicated().any():

        duplicated = (
            dataframe.loc[
                dataframe[
                    "sample_id"
                ].duplicated(),
                "sample_id",
            ]
            .astype(str)
            .tolist()
        )

        raise ValueError(
            f"{name} contains duplicate sample IDs.\n"
            f"Examples: {duplicated[:10]}"
        )

    dataframe[
        "label"
    ] = (
        dataframe[
            "label"
        ]
        .astype(str)
        .str.strip()
        .str.lower()
    )

    observed_labels = set(
        dataframe[
            "label"
        ]
        .unique()
    )

    unexpected_labels = (
        observed_labels
        -
        set(
            labels
        )
    )

    if unexpected_labels:

        raise ValueError(
            f"{name} contains unexpected labels:\n"
            f"{sorted(unexpected_labels)}"
        )

    # -------------------------------------------------------------------------
    # Force feature columns to numeric
    # -------------------------------------------------------------------------

    for feature in FEATURE_COLUMNS:

        dataframe[
            feature
        ] = pd.to_numeric(
            dataframe[
                feature
            ],
            errors="coerce",
        )

        values = (
            dataframe[
                feature
            ]
            .to_numpy(
                dtype=float
            )
        )

        infinite_mask = np.isinf(
            values
        )

        if infinite_mask.any():

            dataframe.loc[
                infinite_mask,
                feature,
            ] = np.nan

    return (
        dataframe
        .reset_index(
            drop=True
        )
    )


# =============================================================================
# Split diagnostics
# =============================================================================

def print_dataset_summary(
    name: str,
    dataframe: pd.DataFrame,
) -> None:

    print()
    print(
        name
    )

    print(
        "-" * 72
    )

    print(
        f"Rows         : "
        f"{len(dataframe):,}"
    )

    print(
        f"Participants : "
        f"{dataframe['participant_id'].nunique():,}"
    )

    print(
        f"Sessions     : "
        f"{dataframe['session_id'].nunique():,}"
    )

    print()

    print(
        "Class distribution:"
    )

    print(
        dataframe[
            "label"
        ]
        .value_counts()
        .sort_index()
    )


# =============================================================================
# Group-aware splitting
# =============================================================================

def split_has_all_labels(
    train_dataframe: pd.DataFrame,
    test_dataframe: pd.DataFrame,
    labels: tuple[str, ...],
) -> bool:

    required = set(
        labels
    )

    train_labels = set(
        train_dataframe[
            "label"
        ]
    )

    test_labels = set(
        test_dataframe[
            "label"
        ]
    )

    return (
        required.issubset(
            train_labels
        )
        and
        required.issubset(
            test_labels
        )
    )


def find_group_split(
    dataframe: pd.DataFrame,
    *,
    group_column: str,
    labels: tuple[str, ...],
    test_size: float,
    random_state: int,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
]:
    """
    Search for a deterministic group-aware split containing all
    requested classes in both training and test partitions.

    Multiple deterministic random seeds are attempted because some
    group configurations may place an entire class in one partition.
    """

    if group_column not in dataframe.columns:

        raise ValueError(
            f"Grouping column {group_column!r} "
            "does not exist."
        )

    groups = (
        dataframe[
            group_column
        ]
        .astype(str)
    )

    unique_group_count = (
        groups.nunique()
    )

    if unique_group_count < 2:

        raise ValueError(
            f"Cannot split using {group_column!r}: "
            f"only {unique_group_count} unique group exists."
        )

    X = (
        dataframe[
            FEATURE_COLUMNS
        ]
    )

    y = (
        dataframe[
            "label"
        ]
    )

    best_split: (
        tuple[
            pd.DataFrame,
            pd.DataFrame,
        ]
        | None
    ) = None

    best_test_class_count = -1

    for attempt in range(
        1000
    ):

        splitter = (
            GroupShuffleSplit(
                n_splits=1,
                test_size=test_size,
                random_state=(
                    random_state
                    + attempt
                ),
            )
        )

        (
            train_index,
            test_index,
        ) = next(
            splitter.split(
                X,
                y,
                groups=groups,
            )
        )

        train_dataframe = (
            dataframe.iloc[
                train_index
            ]
            .copy()
            .reset_index(
                drop=True
            )
        )

        test_dataframe = (
            dataframe.iloc[
                test_index
            ]
            .copy()
            .reset_index(
                drop=True
            )
        )

        test_class_count = (
            test_dataframe[
                "label"
            ]
            .nunique()
        )

        if (
            test_class_count
            >
            best_test_class_count
        ):

            best_test_class_count = (
                test_class_count
            )

            best_split = (
                train_dataframe,
                test_dataframe,
            )

        if split_has_all_labels(
            train_dataframe,
            test_dataframe,
            labels,
        ):

            return (
                train_dataframe,
                test_dataframe,
            )

    if best_split is None:

        raise RuntimeError(
            "Could not construct any group-aware split."
        )

    (
        train_dataframe,
        test_dataframe,
    ) = best_split

    raise RuntimeError(
        f"Could not construct a {group_column}-aware "
        "train/test split containing all required classes "
        "in both partitions.\n\n"
        f"Train labels: "
        f"{sorted(train_dataframe['label'].unique())}\n"
        f"Test labels: "
        f"{sorted(test_dataframe['label'].unique())}"
    )


def choose_sensefuze_split_strategy(
    dataframe: pd.DataFrame,
    labels: tuple[str, ...],
) -> str:
    """
    Prefer participant-independent splitting when genuine multiple
    participants exist.

    When historical SenseFuzeAI data belongs to a single participant,
    session_id is used to prevent windows from the same session leaking
    across training and testing.
    """

    participant_count = (
        dataframe[
            "participant_id"
        ]
        .nunique()
    )

    if participant_count >= 2:

        try:

            find_group_split(
                dataframe,
                group_column=(
                    "participant_id"
                ),
                labels=labels,
                test_size=TEST_SIZE,
                random_state=RANDOM_STATE,
            )

            return (
                "participant_id"
            )

        except (
            RuntimeError,
            ValueError,
        ):

            print(
                "WARNING:"
            )

            print(
                "Participant-aware SenseFuzeAI split could "
                "not preserve all requested classes."
            )

            print(
                "Falling back to session-aware splitting."
            )

    return (
        "session_id"
    )


# =============================================================================
# Frozen split manifests
# =============================================================================

def create_split_manifest(
    *,
    emosurv: pd.DataFrame,
    sensefuzeai: pd.DataFrame,
    labels: tuple[str, ...],
    output_dir: Path,
) -> dict[str, Any]:

    # -------------------------------------------------------------------------
    # EmoSurv:
    # participant-independent split
    # -------------------------------------------------------------------------

    (
        emosurv_train,
        emosurv_test,
    ) = find_group_split(
        emosurv,
        group_column=(
            "participant_id"
        ),
        labels=labels,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
    )

    # -------------------------------------------------------------------------
    # SenseFuzeAI:
    # participant-independent if possible,
    # otherwise session-independent
    # -------------------------------------------------------------------------

    sensefuze_group_column = (
        choose_sensefuze_split_strategy(
            sensefuzeai,
            labels,
        )
    )

    (
        sensefuze_train,
        sensefuze_test,
    ) = find_group_split(
        sensefuzeai,
        group_column=(
            sensefuze_group_column
        ),
        labels=labels,
        test_size=TEST_SIZE,
        random_state=(
            RANDOM_STATE
            + 10000
        ),
    )

    manifest = {
        "random_state":
            RANDOM_STATE,

        "test_size":
            TEST_SIZE,

        "labels":
            list(
                labels
            ),

        "dataset_signatures": {
            "emosurv":
                dataset_signature(
                    emosurv
                ),

            "sensefuzeai":
                dataset_signature(
                    sensefuzeai
                ),
        },

        "emosurv": {
            "group_column":
                "participant_id",

            "train_groups":
                sorted(
                    emosurv_train[
                        "participant_id"
                    ]
                    .astype(str)
                    .unique()
                    .tolist()
                ),

            "test_groups":
                sorted(
                    emosurv_test[
                        "participant_id"
                    ]
                    .astype(str)
                    .unique()
                    .tolist()
                ),
        },

        "sensefuzeai": {
            "group_column":
                sensefuze_group_column,

            "train_groups":
                sorted(
                    sensefuze_train[
                        sensefuze_group_column
                    ]
                    .astype(str)
                    .unique()
                    .tolist()
                ),

            "test_groups":
                sorted(
                    sensefuze_test[
                        sensefuze_group_column
                    ]
                    .astype(str)
                    .unique()
                    .tolist()
                ),
        },
    }

    atomic_write_json(
        manifest,
        output_dir
        / "split_manifest.json",
    )

    return manifest


def load_or_create_split_manifest(
    *,
    emosurv: pd.DataFrame,
    sensefuzeai: pd.DataFrame,
    labels: tuple[str, ...],
    output_dir: Path,
    rebuild: bool,
) -> dict[str, Any]:

    manifest_path = (
        output_dir
        / "split_manifest.json"
    )

    if (
        manifest_path.exists()
        and
        not rebuild
    ):

        manifest = json.loads(
            manifest_path.read_text(
                encoding="utf-8"
            )
        )

        expected_emosurv_signature = (
            dataset_signature(
                emosurv
            )
        )

        expected_sensefuze_signature = (
            dataset_signature(
                sensefuzeai
            )
        )

        stored_signatures = (
            manifest.get(
                "dataset_signatures",
                {},
            )
        )

        if (
            stored_signatures.get(
                "emosurv"
            )
            != expected_emosurv_signature
        ):

            raise RuntimeError(
                "The EmoSurv harmonised dataset has changed "
                "since the held-out split was frozen.\n\n"
                "If this change was intentional, rerun with "
                "--rebuild-splits and document that the "
                "experimental split was regenerated."
            )

        if (
            stored_signatures.get(
                "sensefuzeai"
            )
            != expected_sensefuze_signature
        ):

            raise RuntimeError(
                "The SenseFuzeAI harmonised dataset has changed "
                "since the held-out split was frozen.\n\n"
                "If this change was intentional, rerun with "
                "--rebuild-splits and document that the "
                "experimental split was regenerated."
            )

        print()
        print(
            "Using existing frozen split manifest:"
        )

        print(
            manifest_path
        )

        return manifest

    print()
    print(
        "Creating new frozen split manifest..."
    )

    return (
        create_split_manifest(
            emosurv=emosurv,
            sensefuzeai=sensefuzeai,
            labels=labels,
            output_dir=output_dir,
        )
    )


def apply_manifest_split(
    dataframe: pd.DataFrame,
    split_definition: dict[
        str,
        Any,
    ],
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
]:

    group_column = (
        split_definition[
            "group_column"
        ]
    )

    train_groups = {
        str(
            value
        )
        for value
        in split_definition[
            "train_groups"
        ]
    }

    test_groups = {
        str(
            value
        )
        for value
        in split_definition[
            "test_groups"
        ]
    }

    overlap = (
        train_groups
        &
        test_groups
    )

    if overlap:

        raise RuntimeError(
            "Frozen split manifest contains "
            f"group leakage:\n{sorted(overlap)}"
        )

    group_series = (
        dataframe[
            group_column
        ]
        .astype(str)
    )

    train_dataframe = (
        dataframe[
            group_series.isin(
                train_groups
            )
        ]
        .copy()
        .reset_index(
            drop=True
        )
    )

    test_dataframe = (
        dataframe[
            group_series.isin(
                test_groups
            )
        ]
        .copy()
        .reset_index(
            drop=True
        )
    )

    if train_dataframe.empty:

        raise RuntimeError(
            "Frozen split generated an empty "
            "training partition."
        )

    if test_dataframe.empty:

        raise RuntimeError(
            "Frozen split generated an empty "
            "test partition."
        )

    return (
        train_dataframe,
        test_dataframe,
    )


# =============================================================================
# Probability handling
# =============================================================================

def full_probability_matrix(
    model: Pipeline,
    X: pd.DataFrame,
    labels: tuple[str, ...],
) -> np.ndarray:
    """
    Return probabilities in the canonical requested label order.
    """

    model_probabilities = (
        model.predict_proba(
            X
        )
    )

    model_classes = [
        str(
            value
        )
        for value
        in model.classes_
    ]

    output = np.zeros(
        (
            len(
                X
            ),
            len(
                labels
            ),
        ),
        dtype=float,
    )

    for label_index, label in enumerate(
        labels
    ):

        if label not in model_classes:

            continue

        model_class_index = (
            model_classes.index(
                label
            )
        )

        output[
            :,
            label_index,
        ] = (
            model_probabilities[
                :,
                model_class_index
            ]
        )

    # -------------------------------------------------------------------------
    # Numerical safety
    # -------------------------------------------------------------------------

    output = np.where(
        np.isfinite(
            output
        ),
        output,
        0.0,
    )

    output = np.clip(
        output,
        0.0,
        None,
    )

    row_sums = (
        output.sum(
            axis=1,
            keepdims=True,
        )
    )

    valid_rows = (
        row_sums[
            :,
            0
        ]
        > 0.0
    )

    output[
        valid_rows
    ] = (
        output[
            valid_rows
        ]
        /
        row_sums[
            valid_rows
        ]
    )

    invalid_rows = (
        ~valid_rows
    )

    if invalid_rows.any():

        output[
            invalid_rows
        ] = (
            1.0
            /
            len(
                labels
            )
        )

    return output


# =============================================================================
# Evaluation metrics
# =============================================================================

def multiclass_brier_score(
    y_true: pd.Series,
    probabilities: np.ndarray,
    labels: tuple[str, ...],
) -> float:
    """
    Compute multiclass Brier score.
    """

    truth_matrix = np.zeros_like(
        probabilities,
        dtype=float,
    )

    label_to_index = {
        label:
            index
        for index, label
        in enumerate(
            labels
        )
    }

    for row_index, label in enumerate(
        y_true.astype(str)
    ):

        if label not in label_to_index:

            continue

        truth_matrix[
            row_index,
            label_to_index[
                label
            ],
        ] = 1.0

    score = np.mean(
        np.sum(
            (
                probabilities
                -
                truth_matrix
            )
            ** 2,
            axis=1,
        )
    )

    return float(
        score
    )


def compute_metrics(
    y_true: pd.Series,
    y_pred: np.ndarray,
    probabilities: np.ndarray,
    labels: tuple[str, ...],
) -> dict[str, Any]:

    (
        macro_precision,
        macro_recall,
        macro_f1,
        _,
    ) = (
        precision_recall_fscore_support(
            y_true,
            y_pred,
            labels=list(
                labels
            ),
            average="macro",
            zero_division=0,
        )
    )

    (
        weighted_precision,
        weighted_recall,
        weighted_f1,
        _,
    ) = (
        precision_recall_fscore_support(
            y_true,
            y_pred,
            labels=list(
                labels
            ),
            average="weighted",
            zero_division=0,
        )
    )

    report = (
        classification_report(
            y_true,
            y_pred,
            labels=list(
                labels
            ),
            output_dict=True,
            zero_division=0,
        )
    )

    matrix = (
        confusion_matrix(
            y_true,
            y_pred,
            labels=list(
                labels
            ),
        )
    )

    accuracy = float(
        accuracy_score(
            y_true,
            y_pred,
        )
    )

    balanced_accuracy = float(
        balanced_accuracy_score(
            y_true,
            y_pred,
        )
    )

    # -------------------------------------------------------------------------
    # log_loss requires probability columns to follow lexicographic class order.
    # The project's probability matrix uses the explicit canonical label order,
    # so reorder the columns before calling sklearn.metrics.log_loss
    # -------------------------------------------------------------------------

    log_loss_labels = sorted(
        labels
    )

    log_loss_column_indices = [
        labels.index(
            label
        )
        for label
        in log_loss_labels
    ]

    log_loss_probabilities = (
        probabilities[
            :,
            log_loss_column_indices
        ]
    )

    probability_log_loss = float(
        log_loss(
            y_true,
            log_loss_probabilities,
            labels=log_loss_labels,
        )
    )

    brier = (
        multiclass_brier_score(
            y_true,
            probabilities,
            labels,
        )
    )

    return {
        "accuracy":
            accuracy,

        "balanced_accuracy":
            balanced_accuracy,

        "macro_precision":
            float(
                macro_precision
            ),

        "macro_recall":
            float(
                macro_recall
            ),

        "macro_f1":
            float(
                macro_f1
            ),

        "weighted_precision":
            float(
                weighted_precision
            ),

        "weighted_recall":
            float(
                weighted_recall
            ),

        "weighted_f1":
            float(
                weighted_f1
            ),

        "log_loss":
            probability_log_loss,

        "multiclass_brier_score":
            brier,

        "classification_report":
            report,

        "confusion_matrix":
            matrix.tolist(),
    }


# =============================================================================
# Run one experiment
# =============================================================================

def run_experiment(
    *,
    name: str,
    train_dataframe: pd.DataFrame,
    test_dataframe: pd.DataFrame,
    labels: tuple[str, ...],
    output_dir: Path,
) -> dict[str, Any]:

    experiment_dir = (
        output_dir
        / name
    )

    experiment_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # -------------------------------------------------------------------------
    # Strict predictive feature isolation
    # -------------------------------------------------------------------------

    X_train = (
        train_dataframe[
            FEATURE_COLUMNS
        ]
        .copy()
    )

    y_train = (
        train_dataframe[
            "label"
        ]
        .copy()
    )

    X_test = (
        test_dataframe[
            FEATURE_COLUMNS
        ]
        .copy()
    )

    y_test = (
        test_dataframe[
            "label"
        ]
        .copy()
    )

    train_classes = set(
        y_train
    )

    required_classes = set(
        labels
    )

    missing_train_classes = (
        required_classes
        -
        train_classes
    )

    if missing_train_classes:

        raise RuntimeError(
            f"{name}: training partition is missing "
            f"required classes:\n"
            f"{sorted(missing_train_classes)}"
        )

    # -------------------------------------------------------------------------
    # Model fitting
    # -------------------------------------------------------------------------

    model = (
        build_fixed_model()
    )

    fit_start = (
        time.perf_counter()
    )

    model.fit(
        X_train,
        y_train,
    )

    fit_runtime_seconds = (
        time.perf_counter()
        -
        fit_start
    )

    # -------------------------------------------------------------------------
    # Inference
    # -------------------------------------------------------------------------

    inference_start = (
        time.perf_counter()
    )

    predictions = (
        model.predict(
            X_test
        )
    )

    probabilities = (
        full_probability_matrix(
            model,
            X_test,
            labels,
        )
    )

    inference_runtime_seconds = (
        time.perf_counter()
        -
        inference_start
    )

    # -------------------------------------------------------------------------
    # Metrics
    # -------------------------------------------------------------------------

    metrics = (
        compute_metrics(
            y_test,
            predictions,
            probabilities,
            labels,
        )
    )

    metrics.update(
        {
            "experiment":
                name,

            "train_rows":
                int(
                    len(
                        train_dataframe
                    )
                ),

            "test_rows":
                int(
                    len(
                        test_dataframe
                    )
                ),

            "train_participants":
                int(
                    train_dataframe[
                        "participant_id"
                    ]
                    .nunique()
                ),

            "test_participants":
                int(
                    test_dataframe[
                        "participant_id"
                    ]
                    .nunique()
                ),

            "train_sessions":
                int(
                    train_dataframe[
                        "session_id"
                    ]
                    .nunique()
                ),

            "test_sessions":
                int(
                    test_dataframe[
                        "session_id"
                    ]
                    .nunique()
                ),

            "train_sources":
                (
                    train_dataframe[
                        "dataset_source"
                    ]
                    .value_counts()
                    .to_dict()
                ),

            "test_sources":
                (
                    test_dataframe[
                        "dataset_source"
                    ]
                    .value_counts()
                    .to_dict()
                ),

            "fit_runtime_seconds":
                float(
                    fit_runtime_seconds
                ),

            "inference_runtime_seconds":
                float(
                    inference_runtime_seconds
                ),

            "inference_runtime_per_sample_ms":
                float(
                    (
                        inference_runtime_seconds
                        /
                        max(
                            len(
                                test_dataframe
                            ),
                            1,
                        )
                    )
                    * 1000.0
                ),
        }
    )

    # -------------------------------------------------------------------------
    # Save predictions
    # -------------------------------------------------------------------------

    metadata_columns = [
        "dataset_source",
        "participant_id",
        "session_id",
        "sample_id",
        "source_type",
        "label",
        "label_origin",
        "original_label",
    ]

    prediction_dataframe = (
        test_dataframe[
            metadata_columns
        ]
        .copy()
    )

    prediction_dataframe[
        "predicted_label"
    ] = (
        predictions
    )

    prediction_dataframe[
        "prediction_correct"
    ] = (
        prediction_dataframe[
            "label"
        ]
        ==
        prediction_dataframe[
            "predicted_label"
        ]
    )

    for label_index, label in enumerate(
        labels
    ):

        prediction_dataframe[
            f"prob_{label}"
        ] = (
            probabilities[
                :,
                label_index
            ]
        )

    prediction_dataframe.to_csv(
        experiment_dir
        / "predictions.csv",
        index=False,
    )

    # -------------------------------------------------------------------------
    # Save confusion matrix
    # -------------------------------------------------------------------------

    confusion_dataframe = pd.DataFrame(
        metrics[
            "confusion_matrix"
        ],
        index=list(
            labels
        ),
        columns=list(
            labels
        ),
    )

    confusion_dataframe.index.name = (
        "actual_label"
    )

    confusion_dataframe.columns.name = (
        "predicted_label"
    )

    confusion_dataframe.to_csv(
        experiment_dir
        / "confusion_matrix.csv"
    )

    # -------------------------------------------------------------------------
    # Save classification report
    # -------------------------------------------------------------------------

    classification_report_dataframe = (
        pd.DataFrame(
            metrics[
                "classification_report"
            ]
        )
        .transpose()
    )

    classification_report_dataframe.to_csv(
        experiment_dir
        / "classification_report.csv"
    )

    # -------------------------------------------------------------------------
    # Save model
    # -------------------------------------------------------------------------

    joblib.dump(
        model,
        experiment_dir
        / "model.joblib",
    )

    # -------------------------------------------------------------------------
    # Save concise JSON metrics
    # -------------------------------------------------------------------------

    metrics_json = {
        key:
            value
        for key, value
        in metrics.items()
        if key not in {
            "classification_report",
            "confusion_matrix",
        }
    }

    atomic_write_json(
        metrics_json,
        experiment_dir
        / "metrics.json",
    )

    # -------------------------------------------------------------------------
    # Console summary
    # -------------------------------------------------------------------------

    print()
    print(
        name
    )

    print(
        "-" * 88
    )

    print(
        f"Training rows       : "
        f"{len(train_dataframe):,}"
    )

    print(
        f"Test rows           : "
        f"{len(test_dataframe):,}"
    )

    print(
        f"Accuracy            : "
        f"{metrics['accuracy']:.4f}"
    )

    print(
        f"Balanced Accuracy   : "
        f"{metrics['balanced_accuracy']:.4f}"
    )

    print(
        f"Macro Precision     : "
        f"{metrics['macro_precision']:.4f}"
    )

    print(
        f"Macro Recall        : "
        f"{metrics['macro_recall']:.4f}"
    )

    print(
        f"Macro F1            : "
        f"{metrics['macro_f1']:.4f}"
    )

    print(
        f"Weighted F1         : "
        f"{metrics['weighted_f1']:.4f}"
    )

    print(
        f"Log Loss            : "
        f"{metrics['log_loss']:.4f}"
    )

    print(
        f"Multiclass Brier    : "
        f"{metrics['multiclass_brier_score']:.4f}"
    )

    print(
        f"Fit runtime         : "
        f"{fit_runtime_seconds:.4f} sec"
    )

    print(
        f"Inference runtime   : "
        f"{inference_runtime_seconds:.4f} sec"
    )

    return {
        "metrics":
            metrics,

        "predictions":
            np.asarray(
                predictions
            ),

        "probabilities":
            probabilities,

        "test_dataframe":
            test_dataframe.copy(),

        "model":
            model,
    }


# =============================================================================
# Paired group bootstrap
# =============================================================================

def paired_group_bootstrap_delta(
    *,
    y_true: pd.Series,
    prediction_a: np.ndarray,
    prediction_b: np.ndarray,
    groups: pd.Series,
    labels: tuple[str, ...],
    repetitions: int,
    random_state: int,
) -> dict[str, Any]:
    """
    Estimate uncertainty for the difference in macro F1 between two models
    evaluated on exactly the same grouped test observations.

    Groups are resampled rather than individual windows to reduce the risk
    of overstating certainty when multiple windows belong to one participant
    or session.
    """

    y_values = (
        y_true
        .astype(str)
        .to_numpy()
    )

    group_values = (
        groups
        .astype(str)
        .to_numpy()
    )

    unique_groups = (
        pd.Series(
            group_values
        )
        .unique()
        .tolist()
    )

    observed_f1_a = float(
        f1_score(
            y_values,
            prediction_a,
            labels=list(
                labels
            ),
            average="macro",
            zero_division=0,
        )
    )

    observed_f1_b = float(
        f1_score(
            y_values,
            prediction_b,
            labels=list(
                labels
            ),
            average="macro",
            zero_division=0,
        )
    )

    observed_delta = (
        observed_f1_b
        -
        observed_f1_a
    )

    if len(
        unique_groups
    ) < 2:

        return {
            "model_a_macro_f1":
                observed_f1_a,

            "model_b_macro_f1":
                observed_f1_b,

            "observed_delta":
                float(
                    observed_delta
                ),

            "ci95_lower":
                None,

            "ci95_upper":
                None,

            "bootstrap_repetitions":
                0,

            "bootstrap_group_count":
                len(
                    unique_groups
                ),

            "note":
                (
                    "Confidence interval not calculated because "
                    "fewer than two independent groups were available."
                ),
        }

    group_to_indices = {
        group:
            np.flatnonzero(
                group_values
                == group
            )
        for group
        in unique_groups
    }

    rng = (
        np.random.default_rng(
            random_state
        )
    )

    deltas: list[
        float
    ] = []

    for _ in range(
        repetitions
    ):

        sampled_groups = (
            rng.choice(
                unique_groups,
                size=len(
                    unique_groups
                ),
                replace=True,
            )
        )

        sampled_index_parts = [
            group_to_indices[
                str(
                    group
                )
            ]
            for group
            in sampled_groups
        ]

        sampled_indices = (
            np.concatenate(
                sampled_index_parts
            )
        )

        sampled_truth = (
            y_values[
                sampled_indices
            ]
        )

        sampled_prediction_a = (
            prediction_a[
                sampled_indices
            ]
        )

        sampled_prediction_b = (
            prediction_b[
                sampled_indices
            ]
        )

        f1_a = float(
            f1_score(
                sampled_truth,
                sampled_prediction_a,
                labels=list(
                    labels
                ),
                average="macro",
                zero_division=0,
            )
        )

        f1_b = float(
            f1_score(
                sampled_truth,
                sampled_prediction_b,
                labels=list(
                    labels
                ),
                average="macro",
                zero_division=0,
            )
        )

        deltas.append(
            f1_b
            -
            f1_a
        )

    lower = float(
        np.quantile(
            deltas,
            0.025,
        )
    )

    upper = float(
        np.quantile(
            deltas,
            0.975,
        )
    )

    return {
        "model_a_macro_f1":
            observed_f1_a,

        "model_b_macro_f1":
            observed_f1_b,

        "observed_delta":
            float(
                observed_delta
            ),

        "ci95_lower":
            lower,

        "ci95_upper":
            upper,

        "bootstrap_repetitions":
            int(
                repetitions
            ),

        "bootstrap_group_count":
            int(
                len(
                    unique_groups
                )
            ),
    }


# =============================================================================
# Result table helper
# =============================================================================

def metric_row(
    experiment_name: str,
    result: dict[
        str,
        Any,
    ],
) -> dict[str, Any]:

    metrics = (
        result[
            "metrics"
        ]
    )

    return {
        "experiment":
            experiment_name,

        "accuracy":
            metrics[
                "accuracy"
            ],

        "balanced_accuracy":
            metrics[
                "balanced_accuracy"
            ],

        "macro_precision":
            metrics[
                "macro_precision"
            ],

        "macro_recall":
            metrics[
                "macro_recall"
            ],

        "macro_f1":
            metrics[
                "macro_f1"
            ],

        "weighted_f1":
            metrics[
                "weighted_f1"
            ],

        "log_loss":
            metrics[
                "log_loss"
            ],

        "multiclass_brier_score":
            metrics[
                "multiclass_brier_score"
            ],

        "train_rows":
            metrics[
                "train_rows"
            ],

        "test_rows":
            metrics[
                "test_rows"
            ],

        "train_participants":
            metrics[
                "train_participants"
            ],

        "test_participants":
            metrics[
                "test_participants"
            ],

        "train_sessions":
            metrics[
                "train_sessions"
            ],

        "test_sessions":
            metrics[
                "test_sessions"
            ],

        "fit_runtime_seconds":
            metrics[
                "fit_runtime_seconds"
            ],

        "inference_runtime_seconds":
            metrics[
                "inference_runtime_seconds"
            ],

        "inference_runtime_per_sample_ms":
            metrics[
                "inference_runtime_per_sample_ms"
            ],
    }


# =============================================================================
# Markdown report
# =============================================================================

def format_interval_value(
    value: Any,
) -> str:

    if value is None:

        return "N/A"

    try:

        numeric = float(
            value
        )

    except (
        TypeError,
        ValueError,
    ):

        return "N/A"

    if not math.isfinite(
        numeric
    ):

        return "N/A"

    return (
        f"{numeric:.4f}"
    )


def write_markdown_report(
    *,
    output_path: Path,
    label_mode: str,
    results: dict[
        str,
        dict[str, Any],
    ],
    augmentation_effects: dict[
        str,
        Any,
    ],
    manifest: dict[
        str,
        Any,
    ],
) -> None:

    lines = [
        "# SenseFuzeAI Keystroke Dataset Comparison",
        "",
        f"Label mode: `{label_mode}`",
        "",
        (
            "The EmoSurv behavioural classes used in this comparison "
            "are proxy mappings and must not be interpreted as original "
            "EmoSurv behavioural-state ground truth."
        ),
        "",
        "## Experimental Design",
        "",
        (
            "- **A — EmoSurv baseline:** EmoSurv training participants "
            "evaluated on held-out EmoSurv participants."
        ),
        (
            "- **B — SenseFuzeAI only:** SenseFuzeAI training groups "
            "evaluated on a SenseFuzeAI holdout."
        ),
        (
            "- **C — Augmented to EmoSurv:** EmoSurv training data plus "
            "SenseFuzeAI training data evaluated on the exact same "
            "EmoSurv holdout used by A."
        ),
        (
            "- **D — EmoSurv to SenseFuzeAI:** EmoSurv training data "
            "evaluated on the SenseFuzeAI holdout."
        ),
        (
            "- **E — Augmented to SenseFuzeAI:** EmoSurv training data "
            "plus SenseFuzeAI training data evaluated on the exact same "
            "SenseFuzeAI holdout used by D."
        ),
        (
            "- **F — SenseFuzeAI to EmoSurv:** SenseFuzeAI training data "
            "evaluated on the EmoSurv holdout."
        ),
        "",
        "## Results",
        "",
        (
            "| Experiment | Accuracy | Balanced Accuracy | "
            "Macro Precision | Macro Recall | Macro F1 |"
        ),
        (
            "|---|---:|---:|---:|---:|---:|"
        ),
    ]

    experiment_order = [
        "A_emosurv_baseline",
        "B_sensefuzeai_only",
        "C_augmented_to_emosurv_test",
        "D_emosurv_to_sensefuzeai",
        "E_augmented_to_sensefuzeai_test",
        "F_sensefuzeai_to_emosurv",
    ]

    for experiment_name in experiment_order:

        metrics = (
            results[
                experiment_name
            ][
                "metrics"
            ]
        )

        lines.append(
            (
                f"| {experiment_name} "
                f"| {metrics['accuracy']:.4f} "
                f"| {metrics['balanced_accuracy']:.4f} "
                f"| {metrics['macro_precision']:.4f} "
                f"| {metrics['macro_recall']:.4f} "
                f"| {metrics['macro_f1']:.4f} |"
            )
        )

    primary_effect = (
        augmentation_effects[
            "A_vs_C"
        ]
    )

    secondary_effect = (
        augmentation_effects[
            "D_vs_E"
        ]
    )

    lines.extend(
        [
            "",
            "## Primary Supervisor-Facing Comparison",
            "",
            (
                "Experiment A and Experiment C are evaluated on the "
                "exact same held-out EmoSurv participants."
            ),
            "",
            (
                f"- A Macro F1: "
                f"{primary_effect['model_a_macro_f1']:.4f}"
            ),
            (
                f"- C Macro F1: "
                f"{primary_effect['model_b_macro_f1']:.4f}"
            ),
            (
                f"- Macro-F1 change A -> C: "
                f"{primary_effect['observed_delta']:+.4f}"
            ),
            (
                "- 95% paired group-bootstrap interval: "
                f"[{format_interval_value(primary_effect['ci95_lower'])}, "
                f"{format_interval_value(primary_effect['ci95_upper'])}]"
            ),
            "",
            (
                "A positive delta means that the augmented training "
                "dataset achieved a higher macro F1 on the same unseen "
                "EmoSurv participant test set. A negative delta indicates "
                "that augmentation reduced performance under this "
                "experimental configuration."
            ),
            "",
            "## Secondary Cross-Dataset Augmentation Comparison",
            "",
            (
                "Experiment D and Experiment E are evaluated on the "
                "exact same SenseFuzeAI holdout."
            ),
            "",
            (
                f"- D Macro F1: "
                f"{secondary_effect['model_a_macro_f1']:.4f}"
            ),
            (
                f"- E Macro F1: "
                f"{secondary_effect['model_b_macro_f1']:.4f}"
            ),
            (
                f"- Macro-F1 change D -> E: "
                f"{secondary_effect['observed_delta']:+.4f}"
            ),
            (
                "- 95% paired group-bootstrap interval: "
                f"[{format_interval_value(secondary_effect['ci95_lower'])}, "
                f"{format_interval_value(secondary_effect['ci95_upper'])}]"
            ),
            "",
            "## Split Information",
            "",
            (
                "EmoSurv grouping variable: "
                f"`{manifest['emosurv']['group_column']}`"
            ),
            (
                "SenseFuzeAI grouping variable: "
                f"`{manifest['sensefuzeai']['group_column']}`"
            ),
            "",
            (
                "The split manifest is frozen after creation and is reused "
                "unless `--rebuild-splits` is explicitly supplied."
            ),
            "",
            "## Methodological Interpretation",
            "",
            (
                "The three-class experiment is the primary conservative "
                "dataset-comparison experiment. The four-class experiment "
                "is exploratory because EmoSurv does not provide the four "
                "SenseFuzeAI behavioural states as original ground-truth labels."
            ),
        ]
    )

    output_path.write_text(
        "\n".join(
            lines
        ),
        encoding="utf-8",
    )


# =============================================================================
# Run complete experiment for one label mode
# =============================================================================

def run_label_mode(
    *,
    label_mode: str,
    rebuild_splits: bool,
    bootstrap_repetitions: int,
) -> None:

    labels = (
        labels_for_mode(
            label_mode
        )
    )

    (
        emosurv_path,
        sensefuzeai_path,
    ) = (
        dataset_paths(
            label_mode
        )
    )

    # -------------------------------------------------------------------------
    # Load harmonised datasets
    # -------------------------------------------------------------------------

    emosurv = (
        load_dataset(
            emosurv_path,
            labels=labels,
            name=(
                "EmoSurv harmonised dataset"
            ),
        )
    )

    sensefuzeai = (
        load_dataset(
            sensefuzeai_path,
            labels=labels,
            name=(
                "SenseFuzeAI harmonised dataset"
            ),
        )
    )

    print()
    print("=" * 88)
    print(
        f"DATASET COMPARISON: "
        f"{label_mode.upper()}-CLASS MODE"
    )
    print("=" * 88)

    print_dataset_summary(
        "EmoSurv",
        emosurv,
    )

    print_dataset_summary(
        "SenseFuzeAI",
        sensefuzeai,
    )

    # -------------------------------------------------------------------------
    # Output directory
    # -------------------------------------------------------------------------

    output_dir = (
        OUTPUT_ROOT
        / (
            "three_class_primary"
            if label_mode
            == "three"
            else "four_class_exploratory"
        )
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # -------------------------------------------------------------------------
    # Frozen split
    # -------------------------------------------------------------------------

    manifest = (
        load_or_create_split_manifest(
            emosurv=emosurv,
            sensefuzeai=sensefuzeai,
            labels=labels,
            output_dir=output_dir,
            rebuild=rebuild_splits,
        )
    )

    (
        emosurv_train,
        emosurv_test,
    ) = (
        apply_manifest_split(
            emosurv,
            manifest[
                "emosurv"
            ],
        )
    )

    (
        sensefuzeai_train,
        sensefuzeai_test,
    ) = (
        apply_manifest_split(
            sensefuzeai,
            manifest[
                "sensefuzeai"
            ],
        )
    )

    # -------------------------------------------------------------------------
    # Leakage validation
    # -------------------------------------------------------------------------

    emosurv_train_participants = set(
        emosurv_train[
            "participant_id"
        ]
        .astype(str)
    )

    emosurv_test_participants = set(
        emosurv_test[
            "participant_id"
        ]
        .astype(str)
    )

    emosurv_overlap = (
        emosurv_train_participants
        &
        emosurv_test_participants
    )

    if emosurv_overlap:

        raise RuntimeError(
            "EmoSurv participant leakage detected:\n"
            f"{sorted(emosurv_overlap)}"
        )

    sensefuze_group_column = (
        manifest[
            "sensefuzeai"
        ][
            "group_column"
        ]
    )

    sensefuze_train_groups = set(
        sensefuzeai_train[
            sensefuze_group_column
        ]
        .astype(str)
    )

    sensefuze_test_groups = set(
        sensefuzeai_test[
            sensefuze_group_column
        ]
        .astype(str)
    )

    sensefuze_overlap = (
        sensefuze_train_groups
        &
        sensefuze_test_groups
    )

    if sensefuze_overlap:

        raise RuntimeError(
            "SenseFuzeAI train/test group leakage detected:\n"
            f"{sorted(sensefuze_overlap)}"
        )

    # -------------------------------------------------------------------------
    # Combined training data
    #
    # IMPORTANT:
    #
    # Only training partitions are combined.
    #
    # Neither EmoSurv test observations nor SenseFuzeAI test observations
    # enter combined training.
    # -------------------------------------------------------------------------

    combined_train = (
        pd.concat(
            [
                emosurv_train,
                sensefuzeai_train,
            ],
            ignore_index=True,
            sort=False,
        )
    )

    print()
    print("=" * 88)
    print("FROZEN PARTITIONS")
    print("=" * 88)

    print(
        f"EmoSurv train rows      : "
        f"{len(emosurv_train):,}"
    )

    print(
        f"EmoSurv test rows       : "
        f"{len(emosurv_test):,}"
    )

    print(
        f"EmoSurv train users     : "
        f"{emosurv_train['participant_id'].nunique():,}"
    )

    print(
        f"EmoSurv test users      : "
        f"{emosurv_test['participant_id'].nunique():,}"
    )

    print()

    print(
        f"SenseFuzeAI train rows  : "
        f"{len(sensefuzeai_train):,}"
    )

    print(
        f"SenseFuzeAI test rows   : "
        f"{len(sensefuzeai_test):,}"
    )

    print(
        f"SenseFuze split group   : "
        f"{sensefuze_group_column}"
    )

    print()

    print(
        f"Combined train rows     : "
        f"{len(combined_train):,}"
    )

    # -------------------------------------------------------------------------
    # Run experiments
    # -------------------------------------------------------------------------

    results: dict[
        str,
        dict[str, Any],
    ] = {}

    # -------------------------------------------------------------------------
    # A — EmoSurv baseline
    # -------------------------------------------------------------------------

    results[
        "A_emosurv_baseline"
    ] = (
        run_experiment(
            name=(
                "A_emosurv_baseline"
            ),
            train_dataframe=(
                emosurv_train
            ),
            test_dataframe=(
                emosurv_test
            ),
            labels=labels,
            output_dir=output_dir,
        )
    )

    # -------------------------------------------------------------------------
    # B — SenseFuzeAI only
    # -------------------------------------------------------------------------

    results[
        "B_sensefuzeai_only"
    ] = (
        run_experiment(
            name=(
                "B_sensefuzeai_only"
            ),
            train_dataframe=(
                sensefuzeai_train
            ),
            test_dataframe=(
                sensefuzeai_test
            ),
            labels=labels,
            output_dir=output_dir,
        )
    )

    # -------------------------------------------------------------------------
    # C — Augmented -> SAME EmoSurv test as A
    # -------------------------------------------------------------------------

    results[
        "C_augmented_to_emosurv_test"
    ] = (
        run_experiment(
            name=(
                "C_augmented_to_emosurv_test"
            ),
            train_dataframe=(
                combined_train
            ),
            test_dataframe=(
                emosurv_test
            ),
            labels=labels,
            output_dir=output_dir,
        )
    )

    # -------------------------------------------------------------------------
    # D — EmoSurv -> SenseFuzeAI test
    # -------------------------------------------------------------------------

    results[
        "D_emosurv_to_sensefuzeai"
    ] = (
        run_experiment(
            name=(
                "D_emosurv_to_sensefuzeai"
            ),
            train_dataframe=(
                emosurv_train
            ),
            test_dataframe=(
                sensefuzeai_test
            ),
            labels=labels,
            output_dir=output_dir,
        )
    )

    # -------------------------------------------------------------------------
    # E — Augmented -> SAME SenseFuzeAI test as D
    # -------------------------------------------------------------------------

    results[
        "E_augmented_to_sensefuzeai_test"
    ] = (
        run_experiment(
            name=(
                "E_augmented_to_sensefuzeai_test"
            ),
            train_dataframe=(
                combined_train
            ),
            test_dataframe=(
                sensefuzeai_test
            ),
            labels=labels,
            output_dir=output_dir,
        )
    )

    # -------------------------------------------------------------------------
    # F — SenseFuzeAI -> EmoSurv test
    # -------------------------------------------------------------------------

    results[
        "F_sensefuzeai_to_emosurv"
    ] = (
        run_experiment(
            name=(
                "F_sensefuzeai_to_emosurv"
            ),
            train_dataframe=(
                sensefuzeai_train
            ),
            test_dataframe=(
                emosurv_test
            ),
            labels=labels,
            output_dir=output_dir,
        )
    )

    # -------------------------------------------------------------------------
    # Verify A and C really share identical test observations
    # -------------------------------------------------------------------------

    a_test_ids = (
        results[
            "A_emosurv_baseline"
        ][
            "test_dataframe"
        ][
            "sample_id"
        ]
        .astype(str)
        .tolist()
    )

    c_test_ids = (
        results[
            "C_augmented_to_emosurv_test"
        ][
            "test_dataframe"
        ][
            "sample_id"
        ]
        .astype(str)
        .tolist()
    )

    if a_test_ids != c_test_ids:

        raise RuntimeError(
            "Critical experimental error: "
            "Experiment A and Experiment C do not "
            "contain identical EmoSurv test samples."
        )

    # -------------------------------------------------------------------------
    # Verify D and E share identical SenseFuzeAI test observations
    # -------------------------------------------------------------------------

    d_test_ids = (
        results[
            "D_emosurv_to_sensefuzeai"
        ][
            "test_dataframe"
        ][
            "sample_id"
        ]
        .astype(str)
        .tolist()
    )

    e_test_ids = (
        results[
            "E_augmented_to_sensefuzeai_test"
        ][
            "test_dataframe"
        ][
            "sample_id"
        ]
        .astype(str)
        .tolist()
    )

    if d_test_ids != e_test_ids:

        raise RuntimeError(
            "Critical experimental error: "
            "Experiment D and Experiment E do not "
            "contain identical SenseFuzeAI test samples."
        )

    # -------------------------------------------------------------------------
    # Paired augmentation effect:
    #
    # A vs C
    # -------------------------------------------------------------------------

    a_vs_c = (
        paired_group_bootstrap_delta(
            y_true=(
                results[
                    "A_emosurv_baseline"
                ][
                    "test_dataframe"
                ][
                    "label"
                ]
            ),
            prediction_a=(
                results[
                    "A_emosurv_baseline"
                ][
                    "predictions"
                ]
            ),
            prediction_b=(
                results[
                    "C_augmented_to_emosurv_test"
                ][
                    "predictions"
                ]
            ),
            groups=(
                results[
                    "A_emosurv_baseline"
                ][
                    "test_dataframe"
                ][
                    "participant_id"
                ]
            ),
            labels=labels,
            repetitions=(
                bootstrap_repetitions
            ),
            random_state=(
                RANDOM_STATE
                + 20000
            ),
        )
    )

    # -------------------------------------------------------------------------
    # D vs E
    # -------------------------------------------------------------------------

    d_vs_e = (
        paired_group_bootstrap_delta(
            y_true=(
                results[
                    "D_emosurv_to_sensefuzeai"
                ][
                    "test_dataframe"
                ][
                    "label"
                ]
            ),
            prediction_a=(
                results[
                    "D_emosurv_to_sensefuzeai"
                ][
                    "predictions"
                ]
            ),
            prediction_b=(
                results[
                    "E_augmented_to_sensefuzeai_test"
                ][
                    "predictions"
                ]
            ),
            groups=(
                results[
                    "D_emosurv_to_sensefuzeai"
                ][
                    "test_dataframe"
                ][
                    sensefuze_group_column
                ]
            ),
            labels=labels,
            repetitions=(
                bootstrap_repetitions
            ),
            random_state=(
                RANDOM_STATE
                + 30000
            ),
        )
    )

    augmentation_effects = {
        "A_vs_C": {
            **a_vs_c,

            "interpretation":
                (
                    "Effect of adding SenseFuzeAI training data "
                    "when evaluated on the exact same held-out "
                    "EmoSurv participants."
                ),
        },

        "D_vs_E": {
            **d_vs_e,

            "interpretation":
                (
                    "Effect of adding SenseFuzeAI training data "
                    "when evaluated on the exact same "
                    "SenseFuzeAI holdout."
                ),
        },
    }

    atomic_write_json(
        augmentation_effects,
        output_dir
        / "augmentation_effects.json",
    )

    # -------------------------------------------------------------------------
    # Comparison summary CSV
    # -------------------------------------------------------------------------

    experiment_order = [
        "A_emosurv_baseline",
        "B_sensefuzeai_only",
        "C_augmented_to_emosurv_test",
        "D_emosurv_to_sensefuzeai",
        "E_augmented_to_sensefuzeai_test",
        "F_sensefuzeai_to_emosurv",
    ]

    summary_rows = [
        metric_row(
            experiment_name,
            results[
                experiment_name
            ],
        )
        for experiment_name
        in experiment_order
    ]

    comparison_summary = pd.DataFrame(
        summary_rows
    )

    comparison_summary.to_csv(
        output_dir
        / "comparison_summary.csv",
        index=False,
    )

    # -------------------------------------------------------------------------
    # Split summary CSV
    # -------------------------------------------------------------------------

    split_summary = pd.DataFrame(
        [
            {
                "partition":
                    "emosurv_train",

                "rows":
                    len(
                        emosurv_train
                    ),

                "participants":
                    emosurv_train[
                        "participant_id"
                    ]
                    .nunique(),

                "sessions":
                    emosurv_train[
                        "session_id"
                    ]
                    .nunique(),

                "group_column":
                    "participant_id",
            },

            {
                "partition":
                    "emosurv_test",

                "rows":
                    len(
                        emosurv_test
                    ),

                "participants":
                    emosurv_test[
                        "participant_id"
                    ]
                    .nunique(),

                "sessions":
                    emosurv_test[
                        "session_id"
                    ]
                    .nunique(),

                "group_column":
                    "participant_id",
            },

            {
                "partition":
                    "sensefuzeai_train",

                "rows":
                    len(
                        sensefuzeai_train
                    ),

                "participants":
                    sensefuzeai_train[
                        "participant_id"
                    ]
                    .nunique(),

                "sessions":
                    sensefuzeai_train[
                        "session_id"
                    ]
                    .nunique(),

                "group_column":
                    sensefuze_group_column,
            },

            {
                "partition":
                    "sensefuzeai_test",

                "rows":
                    len(
                        sensefuzeai_test
                    ),

                "participants":
                    sensefuzeai_test[
                        "participant_id"
                    ]
                    .nunique(),

                "sessions":
                    sensefuzeai_test[
                        "session_id"
                    ]
                    .nunique(),

                "group_column":
                    sensefuze_group_column,
            },

            {
                "partition":
                    "combined_train",

                "rows":
                    len(
                        combined_train
                    ),

                "participants":
                    combined_train[
                        "participant_id"
                    ]
                    .nunique(),

                "sessions":
                    combined_train[
                        "session_id"
                    ]
                    .nunique(),

                "group_column":
                    "mixed_training_sources",
            },
        ]
    )

    split_summary.to_csv(
        output_dir
        / "split_summary.csv",
        index=False,
    )

    # -------------------------------------------------------------------------
    # Source/class distribution
    # -------------------------------------------------------------------------

    source_class_rows: list[
        dict[str, Any]
    ] = []

    datasets_to_report = {
        "emosurv_train":
            emosurv_train,

        "emosurv_test":
            emosurv_test,

        "sensefuzeai_train":
            sensefuzeai_train,

        "sensefuzeai_test":
            sensefuzeai_test,

        "combined_train":
            combined_train,
    }

    for dataset_name, dataframe in (
        datasets_to_report.items()
    ):

        grouped = (
            dataframe
            .groupby(
                [
                    "dataset_source",
                    "label",
                ]
            )
            .size()
        )

        for (
            dataset_source,
            label,
        ), count in grouped.items():

            source_class_rows.append(
                {
                    "partition":
                        dataset_name,

                    "dataset_source":
                        dataset_source,

                    "label":
                        label,

                    "rows":
                        int(
                            count
                        ),
                }
            )

    source_class_distribution = (
        pd.DataFrame(
            source_class_rows
        )
    )

    source_class_distribution.to_csv(
        output_dir
        / "source_class_distribution.csv",
        index=False,
    )

    # -------------------------------------------------------------------------
    # Markdown report
    # -------------------------------------------------------------------------

    write_markdown_report(
        output_path=(
            output_dir
            / "comparison_report.md"
        ),
        label_mode=(
            label_mode
        ),
        results=(
            results
        ),
        augmentation_effects=(
            augmentation_effects
        ),
        manifest=(
            manifest
        ),
    )

    # -------------------------------------------------------------------------
    # Final console summary
    # -------------------------------------------------------------------------

    print()
    print("=" * 88)
    print(
        f"COMPARISON COMPLETE: "
        f"{label_mode.upper()}-CLASS MODE"
    )
    print("=" * 88)

    print()
    print(
        "PRIMARY SUPERVISOR-FACING RESULT"
    )

    print(
        "-" * 88
    )

    print(
        "Experiment A "
        "(EmoSurv baseline) Macro F1:"
    )

    print(
        f"  "
        f"{a_vs_c['model_a_macro_f1']:.4f}"
    )

    print()

    print(
        "Experiment C "
        "(EmoSurv + SenseFuzeAI) Macro F1:"
    )

    print(
        f"  "
        f"{a_vs_c['model_b_macro_f1']:.4f}"
    )

    print()

    print(
        "Macro-F1 change A -> C:"
    )

    print(
        f"  "
        f"{a_vs_c['observed_delta']:+.4f}"
    )

    print()

    print(
        "95% paired group-bootstrap interval:"
    )

    print(
        "  ["
        f"{format_interval_value(a_vs_c['ci95_lower'])}, "
        f"{format_interval_value(a_vs_c['ci95_upper'])}"
        "]"
    )

    print()
    print(
        "SECONDARY CROSS-DATASET RESULT"
    )

    print(
        "-" * 88
    )

    print(
        "Macro-F1 change D -> E:"
    )

    print(
        f"  "
        f"{d_vs_e['observed_delta']:+.4f}"
    )

    print()

    print(
        "95% paired group-bootstrap interval:"
    )

    print(
        "  ["
        f"{format_interval_value(d_vs_e['ci95_lower'])}, "
        f"{format_interval_value(d_vs_e['ci95_upper'])}"
        "]"
    )

    print()
    print(
        "Results saved to:"
    )

    print(
        f"  {output_dir}"
    )


# =============================================================================
# Command-line interface
# =============================================================================

def main() -> None:

    parser = argparse.ArgumentParser(
        description=(
            "Run the supervisor-recommended EmoSurv IEEE + "
            "SenseFuzeAI keystroke dataset comparison."
        )
    )

    parser.add_argument(
        "--label-mode",
        choices=[
            "three",
            "four",
            "all",
        ],
        default="all",
        help=(
            "three = conservative primary 3-class experiment; "
            "four = exploratory weakly supervised 4-class experiment; "
            "all = run both."
        ),
    )

    parser.add_argument(
        "--rebuild-splits",
        action="store_true",
        help=(
            "Explicitly regenerate the frozen held-out split manifests. "
            "Do not use this merely to search for better results."
        ),
    )

    parser.add_argument(
        "--bootstrap-repetitions",
        type=int,
        default=(
            BOOTSTRAP_REPETITIONS
        ),
        help=(
            "Number of paired group-bootstrap repetitions used "
            "to estimate uncertainty for augmentation effects."
        ),
    )

    args = parser.parse_args()

    if args.bootstrap_repetitions < 100:

        raise ValueError(
            "--bootstrap-repetitions must be at least 100."
        )

    if args.label_mode == "all":

        modes = [
            "three",
            "four",
        ]

    else:

        modes = [
            args.label_mode
        ]

    for mode in modes:

        run_label_mode(
            label_mode=(
                mode
            ),
            rebuild_splits=(
                args.rebuild_splits
            ),
            bootstrap_repetitions=(
                args.bootstrap_repetitions
            ),
        )


# =============================================================================
# Entry point
# =============================================================================

if __name__ == "__main__":

    main()
