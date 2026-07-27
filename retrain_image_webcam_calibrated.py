# retrain_image_webcam_calibrated.py

from __future__ import annotations

import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


# =============================================================================
# PROJECT CONFIGURATION
# =============================================================================

ROOT_DIR = Path(__file__).resolve().parent

DATA_DIR = ROOT_DIR / "data"
PROCESSED_DIR = DATA_DIR / "processed"

MODEL_DIR = ROOT_DIR / "models" / "image_demo"

REPORT_DIR = (
    PROCESSED_DIR
    / "webcam_calibration_evaluation"
)


# =============================================================================
# PRETRAINED CLIP MODEL
# =============================================================================

# IMPORTANT:
# This is the SAME pretrained CLIP model used by:
#
#   build_webcam_calibration_dataset.py
#   image_live_gui.py
#   final_multimodal_inference.py
#
# The calibration CSV already contains embeddings produced using this model,
# so this script does not need to run CLIP inference again.

CLIP_MODEL_PATH = (
    ROOT_DIR
    / "models"
    / "clip-vit-large-patch14"
)

CLIP_MODEL_NAME = "clip-vit-large-patch14"

EXPECTED_CLIP_DIMENSION = 768


# =============================================================================
# INPUT DATASETS
# =============================================================================

WEBCAM_DATASET_PATH = (
    PROCESSED_DIR
    / "webcam_calibration_clip_features.csv"
)

# Optional original CLIP image dataset.
#
# If this file exists AND contains the same 768 image_clip_emb_* features,
# it can be included as additional training data.
#
# The held-out test set will still contain webcam data only.

BASE_IMAGE_DATASET_PATH = (
    PROCESSED_DIR
    / "image_features.csv"
)


# =============================================================================
# MODEL OUTPUTS
# =============================================================================

ORIGINAL_MODEL_PATH = (
    MODEL_DIR
    / "image_pipeline.joblib"
)

CALIBRATED_MODEL_PATH = (
    MODEL_DIR
    / "image_pipeline_webcam_calibrated.joblib"
)

FEATURE_COLUMNS_PATH = (
    MODEL_DIR
    / "feature_columns.json"
)

CALIBRATED_METADATA_PATH = (
    MODEL_DIR
    / "webcam_calibrated_metadata.json"
)


# =============================================================================
# REPORT OUTPUTS
# =============================================================================

CANDIDATE_RESULTS_PATH = (
    REPORT_DIR
    / "candidate_model_results.csv"
)

CLASSIFICATION_REPORT_PATH = (
    REPORT_DIR
    / "classification_report.csv"
)

CONFUSION_MATRIX_PATH = (
    REPORT_DIR
    / "confusion_matrix.csv"
)

CONFUSION_MATRIX_PLOT_PATH = (
    REPORT_DIR
    / "confusion_matrix.png"
)

MODEL_COMPARISON_PLOT_PATH = (
    REPORT_DIR
    / "candidate_model_comparison.png"
)

TRAINING_REPORT_PATH = (
    REPORT_DIR
    / "webcam_calibrated_training_report.txt"
)

SUMMARY_JSON_PATH = (
    REPORT_DIR
    / "webcam_calibrated_training_summary.json"
)


# =============================================================================
# BEHAVIOURAL CLASSES
# =============================================================================

LABELS = [
    "focused",
    "distracted",
    "fatigued",
    "overloaded",
]


# =============================================================================
# TRAINING CONFIGURATION
# =============================================================================

RANDOM_STATE = 42

# Number of group-aware folds.
#
# Each source video is treated as one group.
# Frames from the same video will therefore not appear in both training
# and testing partitions.
N_SPLITS = 5

# Include original CLIP image dataset if it is schema-compatible.
USE_EXISTING_IMAGE_DATA = True

# Once model selection is complete, fit the selected model using all
# available webcam calibration data plus compatible original image data.
FIT_FINAL_MODEL_ON_ALL_AVAILABLE_DATA = True


# =============================================================================
# GENERAL HELPERS
# =============================================================================

def print_heading(title: str) -> None:
    print()
    print("=" * 88)
    print(title)
    print("=" * 88)


def ensure_directories() -> None:
    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )


def expected_feature_columns() -> list[str]:
    """
    Return the exact CLIP ViT-L/14 embedding schema expected by the
    image classifier.
    """

    return [
        f"image_clip_emb_{index}"
        for index in range(
            EXPECTED_CLIP_DIMENSION
        )
    ]


# =============================================================================
# VALIDATION
# =============================================================================

def validate_clip_model() -> None:
    """
    Verify that the pretrained CLIP ViT-L/14 model exists locally.

    We intentionally do not load CLIP here because the calibration dataset
    already contains CLIP embeddings generated by
    build_webcam_calibration_dataset.py.
    """

    if not CLIP_MODEL_PATH.exists():
        raise FileNotFoundError(
            "Pretrained CLIP model directory not found:\n"
            f"{CLIP_MODEL_PATH}\n\n"
            "The webcam calibration pipeline must use the same pretrained "
            "CLIP ViT-L/14 model as the live image GUI."
        )

    print(
        "Pretrained image encoder:"
    )

    print(
        f"  {CLIP_MODEL_PATH}"
    )

    print(
        f"Expected embedding dimension: "
        f"{EXPECTED_CLIP_DIMENSION}"
    )


def validate_feature_schema(
    dataframe: pd.DataFrame,
    dataset_name: str,
) -> list[str]:
    """
    Ensure a dataset contains exactly the required 768 CLIP features.
    """

    feature_columns = (
        expected_feature_columns()
    )

    missing = [
        column
        for column in feature_columns
        if column not in dataframe.columns
    ]

    if missing:
        raise ValueError(
            f"{dataset_name} is incompatible with the current "
            f"CLIP-based image pipeline.\n\n"
            f"Expected {EXPECTED_CLIP_DIMENSION} columns:\n"
            "image_clip_emb_0 ... image_clip_emb_767\n\n"
            f"Missing examples: {missing[:20]}"
        )

    return feature_columns


def normalise_labels(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    if "label" not in dataframe.columns:
        raise ValueError(
            "Dataset does not contain the required 'label' column."
        )

    result = dataframe.copy()

    result["label"] = (
        result["label"]
        .astype(str)
        .str.strip()
        .str.lower()
    )

    invalid_labels = sorted(
        set(
            result["label"].unique()
        )
        - set(LABELS)
    )

    if invalid_labels:
        raise ValueError(
            "Unexpected behavioural labels found:\n"
            f"{invalid_labels}"
        )

    return result


def clean_feature_matrix(
    dataframe: pd.DataFrame,
    feature_columns: list[str],
) -> pd.DataFrame:
    """
    Convert the CLIP feature matrix to clean float32 values.
    """

    x = dataframe[
        feature_columns
    ].copy()

    x = x.apply(
        pd.to_numeric,
        errors="coerce",
    )

    x = x.replace(
        [np.inf, -np.inf],
        np.nan,
    )

    invalid_count = int(
        x.isna().sum().sum()
    )

    if invalid_count > 0:
        raise ValueError(
            "Feature matrix contains "
            f"{invalid_count} invalid/NaN value(s)."
        )

    return x.astype(
        np.float32
    )


# =============================================================================
# LOAD WEBCAM CALIBRATION DATA
# =============================================================================

def load_webcam_dataset() -> pd.DataFrame:
    if not WEBCAM_DATASET_PATH.exists():
        raise FileNotFoundError(
            "Webcam calibration feature dataset not found:\n"
            f"{WEBCAM_DATASET_PATH}\n\n"
            "Run this first:\n\n"
            "python build_webcam_calibration_dataset.py"
        )

    dataframe = pd.read_csv(
        WEBCAM_DATASET_PATH
    )

    if dataframe.empty:
        raise ValueError(
            "Webcam calibration dataset is empty."
        )

    dataframe = normalise_labels(
        dataframe
    )

    validate_feature_schema(
        dataframe,
        "Webcam calibration dataset",
    )

    if "source_video" not in dataframe.columns:
        raise ValueError(
            "The webcam calibration dataset does not contain "
            "'source_video'.\n\n"
            "source_video is required for group-aware evaluation."
        )

    dataframe["source_video"] = (
        dataframe["source_video"]
        .astype(str)
    )

    return dataframe


# =============================================================================
# OPTIONAL ORIGINAL IMAGE DATA
# =============================================================================

def load_base_image_dataset(
    feature_columns: list[str],
) -> pd.DataFrame | None:
    """
    Load the project's original image CLIP dataset if compatible.

    It is used only as supplementary training data.

    The evaluation set remains unseen webcam videos.
    """

    if not USE_EXISTING_IMAGE_DATA:
        print(
            "Original image dataset inclusion disabled."
        )

        return None

    if not BASE_IMAGE_DATASET_PATH.exists():
        print()
        print(
            "Optional original image dataset not found:"
        )

        print(
            f"  {BASE_IMAGE_DATASET_PATH}"
        )

        print(
            "Training will continue using webcam calibration data only."
        )

        return None

    try:
        dataframe = pd.read_csv(
            BASE_IMAGE_DATASET_PATH
        )

        if dataframe.empty:
            print(
                "Original image dataset is empty. Skipping."
            )

            return None

        dataframe = normalise_labels(
            dataframe
        )

        missing = [
            column
            for column in feature_columns
            if column not in dataframe.columns
        ]

        if missing:
            print()
            print(
                "WARNING:"
            )

            print(
                "The existing image dataset does not use the same "
                "768-dimensional CLIP feature schema."
            )

            print(
                "It will NOT be merged into webcam calibration training."
            )

            print(
                f"Missing examples: {missing[:10]}"
            )

            return None

        return dataframe

    except Exception as exc:
        print()
        print(
            "WARNING:"
        )

        print(
            "Could not load existing image dataset."
        )

        print(
            f"Reason: {exc}"
        )

        print(
            "Continuing with webcam data only."
        )

        return None


# =============================================================================
# DATASET SUMMARY
# =============================================================================

def print_dataset_summary(
    webcam_df: pd.DataFrame,
    base_df: pd.DataFrame | None,
) -> None:
    print_heading(
        "WEBCAM CALIBRATION DATASET"
    )

    print(
        f"Total webcam frames : "
        f"{len(webcam_df)}"
    )

    print(
        f"Source videos       : "
        f"{webcam_df['source_video'].nunique()}"
    )

    print()
    print(
        "Webcam samples by behavioural class:"
    )

    for label in LABELS:
        class_df = webcam_df[
            webcam_df["label"] == label
        ]

        print(
            f"  {label:12s}: "
            f"{len(class_df):4d} frames | "
            f"{class_df['source_video'].nunique():3d} videos"
        )

    print()

    if base_df is not None:
        print(
            f"Compatible original image samples: "
            f"{len(base_df)}"
        )

        print()

        for label in LABELS:
            count = int(
                (
                    base_df["label"]
                    == label
                ).sum()
            )

            print(
                f"  {label:12s}: "
                f"{count:4d}"
            )

    else:
        print(
            "Compatible original image data: not included"
        )


# =============================================================================
# GROUP-AWARE SPLIT
# =============================================================================

def create_group_aware_split(
    webcam_df: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Create a train/test split where source videos are the groups.

    This is critical because multiple extracted frames from the same video
    are highly correlated.
    """

    y = webcam_df[
        "label"
    ].to_numpy()

    groups = webcam_df[
        "source_video"
    ].to_numpy()

    dummy_x = np.zeros(
        shape=(
            len(webcam_df),
            1,
        ),
        dtype=np.float32,
    )

    splitter = StratifiedGroupKFold(
        n_splits=N_SPLITS,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    splits = list(
        splitter.split(
            dummy_x,
            y,
            groups=groups,
        )
    )

    if not splits:
        raise RuntimeError(
            "Could not create group-aware webcam split."
        )

    train_indices, test_indices = (
        splits[0]
    )

    return (
        np.asarray(
            train_indices
        ),
        np.asarray(
            test_indices
        ),
    )


def validate_group_split(
    webcam_df: pd.DataFrame,
    train_indices: np.ndarray,
    test_indices: np.ndarray,
) -> None:
    train_df = webcam_df.iloc[
        train_indices
    ]

    test_df = webcam_df.iloc[
        test_indices
    ]

    train_videos = set(
        train_df["source_video"]
    )

    test_videos = set(
        test_df["source_video"]
    )

    overlap = (
        train_videos
        & test_videos
    )

    if overlap:
        raise RuntimeError(
            "DATA LEAKAGE DETECTED.\n"
            "The same source video appears in both "
            "training and testing partitions."
        )

    print_heading(
        "GROUP-AWARE TRAIN / TEST SPLIT"
    )

    print(
        f"Training webcam frames : "
        f"{len(train_df)}"
    )

    print(
        f"Testing webcam frames  : "
        f"{len(test_df)}"
    )

    print(
        f"Training videos        : "
        f"{len(train_videos)}"
    )

    print(
        f"Testing videos         : "
        f"{len(test_videos)}"
    )

    print(
        f"Video overlap          : "
        f"{len(overlap)}"
    )

    print()
    print(
        "Held-out webcam test distribution:"
    )

    for label in LABELS:
        subset = test_df[
            test_df["label"] == label
        ]

        print(
            f"  {label:12s}: "
            f"{len(subset):4d} frames | "
            f"{subset['source_video'].nunique():3d} videos"
        )


# =============================================================================
# CLASSIFIERS
# =============================================================================

def build_candidate_models() -> dict[str, Any]:
    """
    Models are deliberately lightweight because CLIP has already performed
    the high-dimensional visual representation learning.
    """

    models: dict[str, Any] = {
        # ---------------------------------------------------------------------
        # Logistic Regression
        # ---------------------------------------------------------------------

        "logistic_regression": Pipeline(
            [
                (
                    "scaler",
                    StandardScaler(),
                ),
                (
                    "classifier",
                    LogisticRegression(
                        max_iter=3000,
                        class_weight="balanced",
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),

        # ---------------------------------------------------------------------
        # Linear SVM
        # ---------------------------------------------------------------------

        "svm_linear": Pipeline(
            [
                (
                    "scaler",
                    StandardScaler(),
                ),
                (
                    "classifier",
                    SVC(
                        kernel="linear",
                        probability=True,
                        class_weight="balanced",
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),

        # ---------------------------------------------------------------------
        # RBF SVM
        # ---------------------------------------------------------------------

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
                        C=2.0,
                        gamma="scale",
                        probability=True,
                        class_weight="balanced",
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),

        # ---------------------------------------------------------------------
        # Random Forest
        # ---------------------------------------------------------------------

        "random_forest": RandomForestClassifier(
            n_estimators=300,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
    }

    return models


# =============================================================================
# CANDIDATE EVALUATION
# =============================================================================

def evaluate_candidate(
    name: str,
    model: Any,
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_test: pd.DataFrame,
    y_test: pd.Series,
) -> tuple[dict[str, Any], Any]:
    print()
    print(
        f"Training candidate: {name}"
    )

    fitted_model = clone(
        model
    )

    start = time.perf_counter()

    fitted_model.fit(
        x_train,
        y_train,
    )

    predictions = fitted_model.predict(
        x_test
    )

    runtime = (
        time.perf_counter()
        - start
    )

    accuracy = accuracy_score(
        y_test,
        predictions,
    )

    macro_precision = precision_score(
        y_test,
        predictions,
        labels=LABELS,
        average="macro",
        zero_division=0,
    )

    macro_recall = recall_score(
        y_test,
        predictions,
        labels=LABELS,
        average="macro",
        zero_division=0,
    )

    macro_f1 = f1_score(
        y_test,
        predictions,
        labels=LABELS,
        average="macro",
        zero_division=0,
    )

    weighted_f1 = f1_score(
        y_test,
        predictions,
        labels=LABELS,
        average="weighted",
        zero_division=0,
    )

    result = {
        "model": name,
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
        "weighted_f1": float(
            weighted_f1
        ),
        "runtime_seconds": float(
            runtime
        ),
        "status": "evaluated",
    }

    print(
        f"  Accuracy        : "
        f"{accuracy:.4f}"
    )

    print(
        f"  Macro Precision : "
        f"{macro_precision:.4f}"
    )

    print(
        f"  Macro Recall    : "
        f"{macro_recall:.4f}"
    )

    print(
        f"  Macro F1        : "
        f"{macro_f1:.4f}"
    )

    print(
        f"  Weighted F1     : "
        f"{weighted_f1:.4f}"
    )

    print(
        f"  Runtime         : "
        f"{runtime:.2f} sec"
    )

    return (
        result,
        fitted_model,
    )


# =============================================================================
# VISUAL REPORTS
# =============================================================================

def save_candidate_plot(
    results_df: pd.DataFrame,
) -> None:
    plot_df = (
        results_df
        .sort_values(
            "macro_f1",
            ascending=True,
        )
        .copy()
    )

    y_positions = np.arange(
        len(plot_df)
    )

    width = 0.35

    fig, ax = plt.subplots(
        figsize=(10, 6)
    )

    ax.barh(
        y_positions - width / 2,
        plot_df["macro_f1"],
        height=width,
        label="Macro F1",
        color="#00bcd4",
    )

    ax.barh(
        y_positions + width / 2,
        plot_df["accuracy"],
        height=width,
        label="Accuracy",
        color="#7c4dff",
    )

    ax.set_yticks(
        y_positions
    )

    ax.set_yticklabels(
        plot_df["model"]
    )

    ax.set_xlim(
        0,
        1.05,
    )

    ax.set_xlabel(
        "Score"
    )

    ax.set_title(
        "Webcam-Calibrated Image Model Comparison"
    )

    ax.grid(
        axis="x",
        alpha=0.2,
    )

    ax.legend()

    fig.tight_layout()

    fig.savefig(
        MODEL_COMPARISON_PLOT_PATH,
        dpi=180,
        bbox_inches="tight",
    )

    plt.close(
        fig
    )


def save_confusion_matrix_plot(
    matrix: np.ndarray,
) -> None:
    fig, ax = plt.subplots(
        figsize=(7, 6)
    )

    image = ax.imshow(
        matrix,
        cmap="Blues",
    )

    fig.colorbar(
        image,
        ax=ax,
    )

    ax.set_xticks(
        np.arange(
            len(LABELS)
        )
    )

    ax.set_yticks(
        np.arange(
            len(LABELS)
        )
    )

    ax.set_xticklabels(
        LABELS,
        rotation=30,
        ha="right",
    )

    ax.set_yticklabels(
        LABELS
    )

    ax.set_xlabel(
        "Predicted Class"
    )

    ax.set_ylabel(
        "True Class"
    )

    ax.set_title(
        "Held-Out Webcam Confusion Matrix"
    )

    for row in range(
        matrix.shape[0]
    ):
        for column in range(
            matrix.shape[1]
        ):
            ax.text(
                column,
                row,
                str(
                    matrix[
                        row,
                        column,
                    ]
                ),
                ha="center",
                va="center",
                fontweight="bold",
            )

    fig.tight_layout()

    fig.savefig(
        CONFUSION_MATRIX_PLOT_PATH,
        dpi=180,
        bbox_inches="tight",
    )

    plt.close(
        fig
    )


# =============================================================================
# TEXT REPORT
# =============================================================================

def write_text_report(
    webcam_df: pd.DataFrame,
    base_df: pd.DataFrame | None,
    train_indices: np.ndarray,
    test_indices: np.ndarray,
    results_df: pd.DataFrame,
    selected_name: str,
    selected_metrics: dict[str, Any],
    classification_text: str,
    matrix: np.ndarray,
    final_training_rows: int,
) -> None:
    train_df = webcam_df.iloc[
        train_indices
    ]

    test_df = webcam_df.iloc[
        test_indices
    ]

    lines = [
        "SenseFuzeAI Webcam-Calibrated Image Model Training Report",
        "=" * 70,
        "",
        f"Created at: {datetime.now().isoformat()}",
        "",
        "Pretrained Visual Encoder",
        "-" * 70,
        f"Model: {CLIP_MODEL_NAME}",
        f"Model path: {CLIP_MODEL_PATH}",
        f"Embedding dimension: {EXPECTED_CLIP_DIMENSION}",
        (
            "CLIP fine-tuned: No. "
            "Pretrained CLIP remains frozen."
        ),
        "",
        "Calibration Dataset",
        "-" * 70,
        f"Dataset: {WEBCAM_DATASET_PATH}",
        f"Total webcam frame samples: {len(webcam_df)}",
        f"Unique source videos: {webcam_df['source_video'].nunique()}",
        "",
        "Webcam class distribution:",
        str(
            webcam_df[
                "label"
            ]
            .value_counts()
            .sort_index()
        ),
        "",
        "Evaluation Design",
        "-" * 70,
        "Split method: StratifiedGroupKFold",
        "Grouping variable: source_video",
        f"Number of folds: {N_SPLITS}",
        f"Training webcam frames: {len(train_df)}",
        f"Testing webcam frames: {len(test_df)}",
        f"Training source videos: {train_df['source_video'].nunique()}",
        f"Testing source videos: {test_df['source_video'].nunique()}",
        "",
        "Held-out webcam distribution:",
        str(
            test_df[
                "label"
            ]
            .value_counts()
            .sort_index()
        ),
        "",
        "Original CLIP Image Data",
        "-" * 70,
        (
            f"Included as supplementary training data: "
            f"{base_df is not None}"
        ),
        (
            f"Compatible original image rows: "
            f"{0 if base_df is None else len(base_df)}"
        ),
        "",
        "Candidate Model Comparison",
        "-" * 70,
    ]

    for _, row in results_df.iterrows():
        lines.append(
            f"{row['model']:24s} "
            f"Accuracy={row['accuracy']:.4f} "
            f"MacroF1={row['macro_f1']:.4f} "
            f"MacroRecall={row['macro_recall']:.4f} "
            f"WeightedF1={row['weighted_f1']:.4f}"
        )

    lines.extend(
        [
            "",
            "Selected Model",
            "-" * 70,
            f"Selected classifier: {selected_name}",
            "Selection rule: highest macro F1, then highest accuracy",
            f"Accuracy: {selected_metrics['accuracy']:.4f}",
            f"Macro Precision: {selected_metrics['macro_precision']:.4f}",
            f"Macro Recall: {selected_metrics['macro_recall']:.4f}",
            f"Macro F1: {selected_metrics['macro_f1']:.4f}",
            f"Weighted F1: {selected_metrics['weighted_f1']:.4f}",
            "",
            "Held-Out Webcam Classification Report",
            "-" * 70,
            classification_text,
            "",
            "Confusion Matrix Labels",
            "-" * 70,
            str(LABELS),
            "",
            "Confusion Matrix",
            "-" * 70,
            str(matrix),
            "",
            "Final Deployment Artifact",
            "-" * 70,
            f"Final training rows: {final_training_rows}",
            f"Calibrated model: {CALIBRATED_MODEL_PATH}",
            f"Original model preserved: {ORIGINAL_MODEL_PATH}",
            "",
            "Methodological Note",
            "-" * 70,
            (
                "The pretrained CLIP ViT-L/14 model is used as a frozen "
                "visual feature extractor. The calibration process retrains "
                "only the downstream behavioural-state classifier."
            ),
            "",
            (
                "Evaluation is grouped by source video to prevent frames "
                "originating from the same video appearing in both training "
                "and test sets. This reduces frame-level leakage and produces "
                "a more defensible estimate of live webcam generalisation."
            ),
        ]
    )

    TRAINING_REPORT_PATH.write_text(
        "\n".join(
            lines
        ),
        encoding="utf-8",
    )


# =============================================================================
# MAIN
# =============================================================================

def main() -> int:
    ensure_directories()

    print_heading(
        "SenseFuzeAI Webcam-Calibrated Image Training"
    )

    print(
        f"Project root:\n"
        f"  {ROOT_DIR}"
    )

    # -------------------------------------------------------------------------
    # Verify same pretrained CLIP encoder
    # -------------------------------------------------------------------------

    validate_clip_model()

    # -------------------------------------------------------------------------
    # Load calibration dataset
    # -------------------------------------------------------------------------

    webcam_df = (
        load_webcam_dataset()
    )

    feature_columns = (
        validate_feature_schema(
            webcam_df,
            "Webcam calibration dataset",
        )
    )

    # -------------------------------------------------------------------------
    # Optional original CLIP dataset
    # -------------------------------------------------------------------------

    base_df = (
        load_base_image_dataset(
            feature_columns
        )
    )

    print_dataset_summary(
        webcam_df,
        base_df,
    )

    # -------------------------------------------------------------------------
    # Group-aware webcam holdout
    # -------------------------------------------------------------------------

    train_indices, test_indices = (
        create_group_aware_split(
            webcam_df
        )
    )

    validate_group_split(
        webcam_df,
        train_indices,
        test_indices,
    )

    webcam_train_df = (
        webcam_df
        .iloc[
            train_indices
        ]
        .copy()
    )

    webcam_test_df = (
        webcam_df
        .iloc[
            test_indices
        ]
        .copy()
    )

    # -------------------------------------------------------------------------
    # Combine training data
    # -------------------------------------------------------------------------

    training_parts = [
        webcam_train_df
    ]

    if base_df is not None:
        training_parts.append(
            base_df
        )

    combined_train_df = pd.concat(
        training_parts,
        ignore_index=True,
    )

    print_heading(
        "TRAINING DATA"
    )

    print(
        f"Webcam training frames   : "
        f"{len(webcam_train_df)}"
    )

    print(
        f"Original image samples   : "
        f"{0 if base_df is None else len(base_df)}"
    )

    print(
        f"Combined training rows   : "
        f"{len(combined_train_df)}"
    )

    print(
        f"Held-out webcam test rows: "
        f"{len(webcam_test_df)}"
    )

    # -------------------------------------------------------------------------
    # Matrices
    # -------------------------------------------------------------------------

    x_train = clean_feature_matrix(
        combined_train_df,
        feature_columns,
    )

    y_train = (
        combined_train_df[
            "label"
        ]
        .astype(str)
    )

    x_test = clean_feature_matrix(
        webcam_test_df,
        feature_columns,
    )

    y_test = (
        webcam_test_df[
            "label"
        ]
        .astype(str)
    )

    # -------------------------------------------------------------------------
    # Candidate models
    # -------------------------------------------------------------------------

    print_heading(
        "CANDIDATE MODEL EVALUATION"
    )

    candidate_models = (
        build_candidate_models()
    )

    candidate_results = []
    fitted_candidates = {}

    for name, candidate in candidate_models.items():
        try:
            result, fitted_model = (
                evaluate_candidate(
                    name=name,
                    model=candidate,
                    x_train=x_train,
                    y_train=y_train,
                    x_test=x_test,
                    y_test=y_test,
                )
            )

            candidate_results.append(
                result
            )

            fitted_candidates[
                name
            ] = fitted_model

        except Exception as exc:
            print(
                f"  FAILED: {exc}"
            )

            candidate_results.append(
                {
                    "model": name,
                    "accuracy": np.nan,
                    "macro_precision": np.nan,
                    "macro_recall": np.nan,
                    "macro_f1": np.nan,
                    "weighted_f1": np.nan,
                    "runtime_seconds": np.nan,
                    "status": f"failed: {exc}",
                }
            )

    results_df = pd.DataFrame(
        candidate_results
    )

    evaluated_df = (
        results_df[
            results_df["status"]
            == "evaluated"
        ]
        .copy()
    )

    if evaluated_df.empty:
        raise RuntimeError(
            "All candidate models failed."
        )

    # -------------------------------------------------------------------------
    # Select by Macro-F1 first, then accuracy
    # -------------------------------------------------------------------------

    evaluated_df = (
        evaluated_df
        .sort_values(
            by=[
                "macro_f1",
                "accuracy",
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

    results_df.to_csv(
        CANDIDATE_RESULTS_PATH,
        index=False,
    )

    print_heading(
        "RANKED RESULTS"
    )

    print(
        evaluated_df[
            [
                "model",
                "accuracy",
                "macro_precision",
                "macro_recall",
                "macro_f1",
                "weighted_f1",
                "runtime_seconds",
            ]
        ].to_string(
            index=False
        )
    )

    selected_name = str(
        evaluated_df.iloc[
            0
        ]["model"]
    )

    selected_metrics = (
        evaluated_df.iloc[
            0
        ].to_dict()
    )

    selected_validation_model = (
        fitted_candidates[
            selected_name
        ]
    )

    # -------------------------------------------------------------------------
    # Detailed evaluation
    # -------------------------------------------------------------------------

    predictions = (
        selected_validation_model.predict(
            x_test
        )
    )

    report_text = (
        classification_report(
            y_test,
            predictions,
            labels=LABELS,
            zero_division=0,
        )
    )

    report_dict = (
        classification_report(
            y_test,
            predictions,
            labels=LABELS,
            output_dict=True,
            zero_division=0,
        )
    )

    matrix = confusion_matrix(
        y_test,
        predictions,
        labels=LABELS,
    )

    print_heading(
        "SELECTED MODEL HELD-OUT WEBCAM PERFORMANCE"
    )

    print(
        f"Selected model: "
        f"{selected_name}"
    )

    print()
    print(
        report_text
    )

    print(
        "Confusion matrix:"
    )

    print(
        matrix
    )

    # -------------------------------------------------------------------------
    # Save detailed metrics
    # -------------------------------------------------------------------------

    pd.DataFrame(
        report_dict
    ).transpose().to_csv(
        CLASSIFICATION_REPORT_PATH
    )

    pd.DataFrame(
        matrix,
        index=[
            f"actual_{label}"
            for label in LABELS
        ],
        columns=[
            f"predicted_{label}"
            for label in LABELS
        ],
    ).to_csv(
        CONFUSION_MATRIX_PATH
    )

    save_candidate_plot(
        evaluated_df
    )

    save_confusion_matrix_plot(
        matrix
    )

    # -------------------------------------------------------------------------
    # Final training dataset
    # -------------------------------------------------------------------------

    if FIT_FINAL_MODEL_ON_ALL_AVAILABLE_DATA:
        final_parts = [
            webcam_df
        ]

        if base_df is not None:
            final_parts.append(
                base_df
            )

        final_training_df = pd.concat(
            final_parts,
            ignore_index=True,
        )

    else:
        final_training_df = (
            combined_train_df.copy()
        )

    x_final = clean_feature_matrix(
        final_training_df,
        feature_columns,
    )

    y_final = (
        final_training_df[
            "label"
        ]
        .astype(str)
    )

    # -------------------------------------------------------------------------
    # Retrain best candidate
    # -------------------------------------------------------------------------

    final_model = clone(
        candidate_models[
            selected_name
        ]
    )

    print_heading(
        "TRAINING FINAL CALIBRATED MODEL"
    )

    print(
        f"Classifier: "
        f"{selected_name}"
    )

    print(
        f"Training rows: "
        f"{len(final_training_df)}"
    )

    final_start = (
        time.perf_counter()
    )

    final_model.fit(
        x_final,
        y_final,
    )

    final_runtime = (
        time.perf_counter()
        - final_start
    )

    print(
        f"Final training runtime: "
        f"{final_runtime:.2f} sec"
    )

    # -------------------------------------------------------------------------
    # Save calibrated model
    # -------------------------------------------------------------------------

    joblib.dump(
        final_model,
        CALIBRATED_MODEL_PATH,
    )

    # -------------------------------------------------------------------------
    # Feature schema
    # -------------------------------------------------------------------------

    FEATURE_COLUMNS_PATH.write_text(
        json.dumps(
            feature_columns,
            indent=4,
        ),
        encoding="utf-8",
    )

    # -------------------------------------------------------------------------
    # Metadata
    # -------------------------------------------------------------------------

    metadata = {
        "created_at": (
            datetime.now()
            .astimezone()
            .isoformat()
        ),

        "model_type": (
            "webcam_calibrated_image_classifier"
        ),

        "pretrained_visual_encoder": {
            "name": CLIP_MODEL_NAME,
            "path": str(
                CLIP_MODEL_PATH
            ),
            "frozen": True,
            "fine_tuned": False,
            "embedding_dimension": (
                EXPECTED_CLIP_DIMENSION
            ),
            "feature_prefix": (
                "image_clip_emb_"
            ),
        },

        "webcam_calibration_dataset": str(
            WEBCAM_DATASET_PATH
        ),

        "base_image_dataset": (
            str(
                BASE_IMAGE_DATASET_PATH
            )
            if base_df is not None
            else None
        ),

        "existing_image_data_included": (
            base_df is not None
        ),

        "webcam_frame_samples": int(
            len(webcam_df)
        ),

        "unique_source_videos": int(
            webcam_df[
                "source_video"
            ].nunique()
        ),

        "feature_dimension": (
            EXPECTED_CLIP_DIMENSION
        ),

        "feature_columns": (
            feature_columns
        ),

        "behavioural_labels": (
            LABELS
        ),

        "split_method": (
            "StratifiedGroupKFold"
        ),

        "group_column": (
            "source_video"
        ),

        "number_of_folds": (
            N_SPLITS
        ),

        "selection_metric": (
            "macro-F1, then accuracy"
        ),

        "candidate_models": list(
            candidate_models.keys()
        ),

        "selected_model": (
            selected_name
        ),

        "held_out_webcam_metrics": {
            "accuracy": float(
                selected_metrics[
                    "accuracy"
                ]
            ),
            "macro_precision": float(
                selected_metrics[
                    "macro_precision"
                ]
            ),
            "macro_recall": float(
                selected_metrics[
                    "macro_recall"
                ]
            ),
            "macro_f1": float(
                selected_metrics[
                    "macro_f1"
                ]
            ),
            "weighted_f1": float(
                selected_metrics[
                    "weighted_f1"
                ]
            ),
        },

        "final_training_rows": int(
            len(final_training_df)
        ),

        "final_fit_runtime_seconds": float(
            final_runtime
        ),

        "original_model_preserved": str(
            ORIGINAL_MODEL_PATH
        ),

        "calibrated_model_path": str(
            CALIBRATED_MODEL_PATH
        ),
    }

    CALIBRATED_METADATA_PATH.write_text(
        json.dumps(
            metadata,
            indent=4,
        ),
        encoding="utf-8",
    )

    # -------------------------------------------------------------------------
    # Summary JSON
    # -------------------------------------------------------------------------

    summary = {
        "created_at": (
            datetime.now()
            .astimezone()
            .isoformat()
        ),

        "clip_model": (
            CLIP_MODEL_NAME
        ),

        "clip_model_path": str(
            CLIP_MODEL_PATH
        ),

        "clip_embedding_dimension": (
            EXPECTED_CLIP_DIMENSION
        ),

        "selected_classifier": (
            selected_name
        ),

        "selection_rule": (
            "highest macro F1, then highest accuracy"
        ),

        "webcam_samples": int(
            len(webcam_df)
        ),

        "unique_videos": int(
            webcam_df[
                "source_video"
            ].nunique()
        ),

        "held_out_accuracy": float(
            selected_metrics[
                "accuracy"
            ]
        ),

        "held_out_macro_f1": float(
            selected_metrics[
                "macro_f1"
            ]
        ),

        "held_out_macro_recall": float(
            selected_metrics[
                "macro_recall"
            ]
        ),

        "confusion_matrix_labels": (
            LABELS
        ),

        "confusion_matrix": (
            matrix.tolist()
        ),

        "candidate_results": (
            evaluated_df
            .to_dict(
                orient="records"
            )
        ),

        "calibrated_model": str(
            CALIBRATED_MODEL_PATH
        ),
    }

    SUMMARY_JSON_PATH.write_text(
        json.dumps(
            summary,
            indent=4,
        ),
        encoding="utf-8",
    )

    # -------------------------------------------------------------------------
    # TXT report
    # -------------------------------------------------------------------------

    write_text_report(
        webcam_df=webcam_df,
        base_df=base_df,
        train_indices=train_indices,
        test_indices=test_indices,
        results_df=evaluated_df,
        selected_name=selected_name,
        selected_metrics=selected_metrics,
        classification_text=report_text,
        matrix=matrix,
        final_training_rows=len(
            final_training_df
        ),
    )

    # -------------------------------------------------------------------------
    # Final console output
    # -------------------------------------------------------------------------

    print_heading(
        "WEBCAM-CALIBRATED MODEL COMPLETE"
    )

    print(
        "Pretrained visual model:"
    )

    print(
        f"  {CLIP_MODEL_PATH}"
    )

    print()
    print(
        "CLIP status:"
    )

    print(
        "  Frozen pretrained feature extractor"
    )

    print(
        "  No CLIP fine-tuning performed"
    )

    print()
    print(
        f"Selected classifier:\n"
        f"  {selected_name}"
    )

    print()
    print(
        f"Held-out webcam accuracy:\n"
        f"  {selected_metrics['accuracy']:.4f}"
    )

    print()
    print(
        f"Held-out webcam macro F1:\n"
        f"  {selected_metrics['macro_f1']:.4f}"
    )

    print()
    print(
        f"Held-out webcam macro recall:\n"
        f"  {selected_metrics['macro_recall']:.4f}"
    )

    print()
    print(
        "Calibrated model saved:"
    )

    print(
        f"  {CALIBRATED_MODEL_PATH}"
    )

    print()
    print(
        "Original image model preserved:"
    )

    print(
        f"  {ORIGINAL_MODEL_PATH}"
    )

    print()
    print(
        "Metadata:"
    )

    print(
        f"  {CALIBRATED_METADATA_PATH}"
    )

    print()
    print(
        "Evaluation directory:"
    )

    print(
        f"  {REPORT_DIR}"
    )

    print()
    print(
        "IMPORTANT NEXT STEP:"
    )

    print(
        "Review classification_report.csv and confusion_matrix.png."
    )

    print(
        "Confirm that all four behavioural classes have reasonable recall "
        "before switching image_live_gui.py to the calibrated model."
    )

    return 0


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    try:
        raise SystemExit(
            main()
        )

    except KeyboardInterrupt:
        print()
        print(
            "Training interrupted by user."
        )

        raise SystemExit(
            130
        )

    except Exception as exc:
        print_heading(
            "WEBCAM CALIBRATION TRAINING FAILED"
        )

        print(
            str(exc)
        )

        raise SystemExit(
            1
        )
