# === train_image_model.py ===
# SenseFuzeAI - Image Model Training and Selection Pipeline
#
# Safer improved version.
#
# This pipeline avoids direct label-leakage columns such as:
#   - visual_fused_label
#   - visual_fused_score
#   - visual_fused_margin
#   - majority_visual_label
#   - visual_votes
#   - agreement_with_folder_label
#   - quality_flag
#   - reliability_score
#
# Instead, it uses safer visual evidence features:
#   - image dimensions
#   - face/body detection metadata
#   - MediaPipe posture/head-position cues encoded as categorical features
#   - raw CLIP prompt score columns:
#       behaviour_*_score
#       scene_*_score
#       face_*_score
#       body_*_score
#
# Candidate algorithms:
#   - SVM RBF
#   - SVM linear
#   - Random Forest
#   - Logistic Regression
#   - Decision Tree
#   - Gaussian Naive Bayes
#   - kNN
#
# Best model selection:
#   1. highest macro F1-score
#   2. highest accuracy if macro F1 is tied
#
# Input:
#   data/processed/processed_image_dataset.csv
#   or data/processed/processed_image_dataset_v2.csv
#
# Output:
#   model_artifacts/image_model.joblib
#   model_artifacts/image_model_meta.json
#   model_artifacts/image_training_report.txt
#   model_artifacts/visual_reports/image/*.png
#   model_artifacts/visual_reports/image/*.csv

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier


# ============================================================
# Configuration
# ============================================================

DATA_CANDIDATES = [
    Path("data/processed/processed_image_dataset.csv"),
    Path("data/processed/processed_image_dataset_v2.csv"),
]

MODEL_DIR = Path("model_artifacts")
MODEL_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATH = MODEL_DIR / "image_model.joblib"
META_PATH = MODEL_DIR / "image_model_meta.json"
REPORT_PATH = MODEL_DIR / "image_training_report.txt"

VISUAL_REPORT_DIR = MODEL_DIR / "visual_reports" / "image"
VISUAL_REPORT_DIR.mkdir(parents=True, exist_ok=True)

TARGET = "label"

CLASSES = [
    "focused",
    "distracted",
    "fatigued",
    "overloaded",
]

RANDOM_STATE = 42
TEST_SIZE = 0.20

# Keep only higher-confidence processed image rows by default.
# This avoids weak/ambiguous image samples dominating the evaluation.
RECOMMENDED_QUALITY = {
    "high_reliability",
    "moderate_reliability",
}

# Feature mode:
#   "strict_metadata"      = safest but weak
#   "safe_visual_features" = recommended middle-ground
FEATURE_MODE = "safe_visual_features"


# ============================================================
# Columns deliberately excluded to prevent leakage
# ============================================================

LEAKAGE_OR_NON_FEATURE_COLUMNS = {
    # target / split / identity
    "label",
    "split",
    "filepath",
    "filename",
    "error",

    # text/caption fields
    "scene_description",
    "behaviour_description",
    "generic_caption",
    "behaviour_caption",
    "visual_cue_summary",
    "supporting_visual_cues",
    "possible_conflicting_cues",
    "absent_or_uncertain_cues",

    # crop paths / boxes
    "face_box",
    "body_box",
    "face_crop_path",
    "body_crop_path",

    # direct pseudo-label outputs
    "visual_fused_label",
    "visual_fused_second_label",
    "predicted_behaviour_cue",
    "predicted_scene_cue",
    "predicted_face_cue",
    "predicted_body_cue",
    "majority_visual_label",
    "visual_votes",

    # agreement with the folder label is highly leaky
    "agreement_with_folder_label",
    "agreement_ratio",
    "majority_vote_count",
    "total_votes",

    # derived fused metrics
    "visual_fused_score",
    "visual_fused_second_score",
    "visual_fused_margin",

    # final fused class probability outputs
    # These are already the output of a visual fusion rule and can inflate results.
    "focused_score",
    "distracted_score",
    "fatigued_score",
    "overloaded_score",

    # quality/reliability derived from fused visual margin/detection
    "quality_flag",
    "reliability_score",
}


STRICT_METADATA_FEATURES = [
    "width",
    "height",
    "face_detected",
    "face_confidence",
    "body_detected",
    "pose_visibility",
    "shoulder_visibility",
]


SAFE_NUMERIC_PREFIXES = [
    "behaviour_",
    "scene_",
    "face_",
    "body_",
]


SAFE_CATEGORICAL_COLUMNS = [
    "posture_cue",
    "head_position_cue",
]


# ============================================================
# Utility functions
# ============================================================

def resolve_data_path() -> Path:
    """
    Resolve the first available processed image dataset.
    """
    for path in DATA_CANDIDATES:
        if path.exists():
            return path

    raise FileNotFoundError(
        "Missing image dataset. Expected one of:\n"
        + "\n".join(str(path) for path in DATA_CANDIDATES)
        + "\n\nRun process_image_data.py first."
    )


def normalise_label(value: Any) -> str:
    """
    Convert labels into canonical lowercase strings.
    """
    if pd.isna(value):
        return ""

    return str(value).strip().lower()


def remove_error_rows(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove failed preprocessing rows if an error column exists.
    """
    if "error" not in df.columns:
        return df.copy()

    error_values = df["error"].astype(str).str.strip().str.lower()

    return df[
        df["error"].isna()
        | (error_values == "")
        | (error_values == "nan")
        | (error_values == "none")
    ].copy()


def load_dataset(data_path: Path) -> pd.DataFrame:
    """
    Load and clean the processed image dataset.
    """
    df = pd.read_csv(data_path)

    if TARGET not in df.columns:
        raise ValueError("Image CSV must contain a 'label' column.")

    original_rows = len(df)

    df = df.copy()
    df[TARGET] = df[TARGET].apply(normalise_label)

    df = remove_error_rows(df)
    df = df[df[TARGET].isin(CLASSES)].copy()

    quality_filter_used = False

    if "quality_flag" in df.columns:
        filtered = df[df["quality_flag"].isin(RECOMMENDED_QUALITY)].copy()

        if not filtered.empty:
            df = filtered
            quality_filter_used = True
        else:
            print(
                "Warning: quality filtering produced an empty dataset. "
                "Training will continue using all valid rows."
            )

    if df.empty:
        raise ValueError("No valid image samples available after filtering.")

    df.attrs["original_rows"] = original_rows
    df.attrs["quality_filter_used"] = quality_filter_used

    return df


def validate_class_distribution(df: pd.DataFrame) -> None:
    """
    Validate basic class availability.
    """
    counts = df[TARGET].value_counts()

    missing_classes = [
        label for label in CLASSES
        if label not in counts.index
    ]

    if missing_classes:
        print(f"Warning: missing classes in dataset: {missing_classes}")

    too_small = counts[counts < 2]

    if not too_small.empty:
        raise ValueError(
            "Some classes have fewer than 2 samples, so stratified splitting "
            f"is impossible:\n{too_small}"
        )


# ============================================================
# Feature selection
# ============================================================

def is_safe_prompt_score_column(column: str) -> bool:
    """
    Select raw prompt score columns only.

    Examples retained:
        behaviour_focused_score
        scene_overloaded_score
        face_fatigued_score
        body_distracted_score

    Examples excluded:
        predicted_behaviour_score
        visual_fused_score
    """
    if not column.endswith("_score"):
        return False

    if column in LEAKAGE_OR_NON_FEATURE_COLUMNS:
        return False

    return any(column.startswith(prefix) for prefix in SAFE_NUMERIC_PREFIXES)


def get_base_numeric_features(df: pd.DataFrame) -> list[str]:
    """
    Select numeric features according to the configured feature mode.
    """
    numeric_cols = df.select_dtypes(include=["number", "bool"]).columns.tolist()

    if FEATURE_MODE == "strict_metadata":
        return [
            col for col in STRICT_METADATA_FEATURES
            if col in df.columns
        ]

    if FEATURE_MODE == "safe_visual_features":
        selected: list[str] = []

        for col in STRICT_METADATA_FEATURES:
            if col in df.columns:
                selected.append(col)

        for col in numeric_cols:
            if is_safe_prompt_score_column(col):
                selected.append(col)

        # Remove duplicates while preserving order.
        seen = set()
        unique_selected = []

        for col in selected:
            if col not in seen:
                seen.add(col)
                unique_selected.append(col)

        return unique_selected

    raise ValueError(
        f"Unknown FEATURE_MODE: {FEATURE_MODE}. "
        "Use 'strict_metadata' or 'safe_visual_features'."
    )


def prepare_feature_dataframe(
    df: pd.DataFrame,
    feature_cols: list[str],
) -> tuple[pd.DataFrame, list[str]]:
    """
    Build final model-ready feature matrix.

    Numeric features are coerced to numbers.
    Safe categorical posture/head-position cues are one-hot encoded.
    """
    X_parts: list[pd.DataFrame] = []

    numeric_df = df.reindex(columns=feature_cols, fill_value=0.0).copy()

    for col in numeric_df.columns:
        numeric_df[col] = pd.to_numeric(numeric_df[col], errors="coerce").fillna(0.0)

    X_parts.append(numeric_df)

    available_categorical = [
        col for col in SAFE_CATEGORICAL_COLUMNS
        if col in df.columns and FEATURE_MODE == "safe_visual_features"
    ]

    if available_categorical:
        categorical_df = df[available_categorical].fillna("unknown").astype(str)
        categorical_encoded = pd.get_dummies(
            categorical_df,
            prefix=available_categorical,
            dtype=float,
        )
        X_parts.append(categorical_encoded)

    X = pd.concat(X_parts, axis=1)

    final_features = list(X.columns)

    return X, final_features


def split_dataset(
    df: pd.DataFrame,
    X: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, str]:
    """
    Use existing split column when available; otherwise use stratified split.
    """
    y = df[TARGET]

    if "split" in df.columns:
        split_values = df["split"].astype(str).str.strip().str.lower()

        train_mask = split_values == "train"
        test_mask = split_values == "test"

        if train_mask.any() and test_mask.any():
            print("Using existing train/test split from image dataset.")

            return (
                X.loc[train_mask].copy(),
                X.loc[test_mask].copy(),
                y.loc[train_mask].copy(),
                y.loc[test_mask].copy(),
                "existing_split_column",
            )

        print(
            "Warning: split column exists but train/test rows are incomplete. "
            "Using stratified train_test_split instead."
        )

    print("Using stratified train_test_split.")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        stratify=y,
        random_state=RANDOM_STATE,
    )

    return X_train, X_test, y_train, y_test, "stratified_train_test_split"


# ============================================================
# Model candidates
# ============================================================

def build_candidate_pipelines() -> dict[str, Pipeline]:
    """
    Build candidate model pipelines.
    """
    return {
        "svm_rbf": Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
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
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
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
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "model",
                    RandomForestClassifier(
                        n_estimators=500,
                        max_depth=None,
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
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                (
                    "model",
                    LogisticRegression(
                        max_iter=3000,
                        class_weight="balanced",
                        random_state=RANDOM_STATE,
                        solver="lbfgs",
                    ),
                ),
            ]
        ),
        "decision_tree": Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "model",
                    DecisionTreeClassifier(
                        criterion="gini",
                        max_depth=None,
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
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("model", GaussianNB()),
            ]
        ),
        "knn_3": Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
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
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
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


# ============================================================
# Metrics
# ============================================================

def compute_metrics(
    y_true: pd.Series,
    y_pred: Any,
) -> dict[str, Any]:
    """
    Compute standard evaluation metrics.
    """
    accuracy = accuracy_score(y_true, y_pred)

    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=CLASSES,
        average="macro",
        zero_division=0,
    )

    weighted_precision, weighted_recall, weighted_f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=CLASSES,
        average="weighted",
        zero_division=0,
    )

    report_text = classification_report(
        y_true,
        y_pred,
        labels=CLASSES,
        zero_division=0,
    )

    report_dict = classification_report(
        y_true,
        y_pred,
        labels=CLASSES,
        zero_division=0,
        output_dict=True,
    )

    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=CLASSES,
    )

    return {
        "accuracy": float(accuracy),
        "macro_precision": float(macro_precision),
        "macro_recall": float(macro_recall),
        "macro_f1": float(macro_f1),
        "weighted_precision": float(weighted_precision),
        "weighted_recall": float(weighted_recall),
        "weighted_f1": float(weighted_f1),
        "classification_report_text": report_text,
        "classification_report": report_dict,
        "confusion_matrix": cm.tolist(),
    }


def evaluate_candidates(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
) -> tuple[str, Pipeline, dict[str, dict[str, Any]]]:
    """
    Train all candidate models and select the best by macro F1, then accuracy.
    """
    candidates = build_candidate_pipelines()

    results: dict[str, dict[str, Any]] = {}

    best_name: str | None = None
    best_model: Pipeline | None = None
    best_macro_f1 = -1.0
    best_accuracy = -1.0

    for name, pipeline in candidates.items():
        print(f"\nTraining candidate: {name}")

        try:
            pipeline.fit(X_train, y_train)
            y_pred = pipeline.predict(X_test)

            metrics = compute_metrics(y_test, y_pred)
            results[name] = metrics

            print(
                f"  Accuracy: {metrics['accuracy']:.4f} | "
                f"Macro F1: {metrics['macro_f1']:.4f} | "
                f"Weighted F1: {metrics['weighted_f1']:.4f}"
            )

            is_better = (
                metrics["macro_f1"] > best_macro_f1
                or (
                    metrics["macro_f1"] == best_macro_f1
                    and metrics["accuracy"] > best_accuracy
                )
            )

            if is_better:
                best_name = name
                best_model = pipeline
                best_macro_f1 = metrics["macro_f1"]
                best_accuracy = metrics["accuracy"]

        except Exception as exc:
            results[name] = {
                "status": "error",
                "reason": str(exc),
                "accuracy": 0.0,
                "macro_precision": 0.0,
                "macro_recall": 0.0,
                "macro_f1": 0.0,
                "weighted_precision": 0.0,
                "weighted_recall": 0.0,
                "weighted_f1": 0.0,
                "classification_report_text": f"Model failed: {exc}",
                "classification_report": {},
                "confusion_matrix": [],
            }

            print(f"  Failed: {exc}")

    if best_name is None or best_model is None:
        raise RuntimeError("No candidate image model was successfully trained.")

    return best_name, best_model, results


def candidate_summary(
    candidate_results: dict[str, dict[str, Any]]
) -> dict[str, dict[str, Any]]:
    """
    Compact candidate summary for metadata.
    """
    summary = {}

    for name, metrics in candidate_results.items():
        summary[name] = {
            "status": metrics.get("status", "evaluated"),
            "reason": metrics.get("reason"),
            "accuracy": metrics.get("accuracy", 0.0),
            "macro_precision": metrics.get("macro_precision", 0.0),
            "macro_recall": metrics.get("macro_recall", 0.0),
            "macro_f1": metrics.get("macro_f1", 0.0),
            "weighted_precision": metrics.get("weighted_precision", 0.0),
            "weighted_recall": metrics.get("weighted_recall", 0.0),
            "weighted_f1": metrics.get("weighted_f1", 0.0),
            "confusion_matrix": metrics.get("confusion_matrix", []),
        }

    return summary


# ============================================================
# Visual report generation
# ============================================================

def sorted_candidate_rows(
    candidate_results: dict[str, dict[str, Any]]
) -> list[dict[str, Any]]:
    """
    Convert candidate results to sorted row dictionaries.
    """
    rows = []

    for name, metrics in candidate_results.items():
        rows.append(
            {
                "candidate": name,
                "status": metrics.get("status", "evaluated"),
                "accuracy": metrics.get("accuracy", 0.0),
                "macro_precision": metrics.get("macro_precision", 0.0),
                "macro_recall": metrics.get("macro_recall", 0.0),
                "macro_f1": metrics.get("macro_f1", 0.0),
                "weighted_precision": metrics.get("weighted_precision", 0.0),
                "weighted_recall": metrics.get("weighted_recall", 0.0),
                "weighted_f1": metrics.get("weighted_f1", 0.0),
            }
        )

    rows.sort(
        key=lambda row: (
            row["macro_f1"],
            row["accuracy"],
        ),
        reverse=True,
    )

    return rows


def save_candidate_comparison_visuals(
    candidate_results: dict[str, dict[str, Any]],
) -> dict[str, str]:
    """
    Save candidate comparison CSV, bar chart, and table image.
    """
    rows = sorted_candidate_rows(candidate_results)
    df = pd.DataFrame(rows)

    csv_path = VISUAL_REPORT_DIR / "image_candidate_model_comparison.csv"
    chart_path = VISUAL_REPORT_DIR / "image_candidate_model_comparison.png"
    table_path = VISUAL_REPORT_DIR / "image_candidate_model_table.png"

    df.to_csv(csv_path, index=False)

    plot_df = df.melt(
        id_vars=["candidate"],
        value_vars=["accuracy", "macro_f1", "weighted_f1"],
        var_name="metric",
        value_name="score",
    )

    plt.figure(figsize=(12, 6))
    sns.barplot(
        data=plot_df,
        x="candidate",
        y="score",
        hue="metric",
    )
    plt.ylim(0, 1.05)
    plt.title("Image Candidate Model Comparison")
    plt.xlabel("Candidate model")
    plt.ylabel("Score")
    plt.xticks(rotation=35, ha="right")
    plt.tight_layout()
    plt.savefig(chart_path, dpi=200)
    plt.close()

    table_df = df[
        [
            "candidate",
            "accuracy",
            "macro_f1",
            "weighted_f1",
            "status",
        ]
    ].copy()

    for col in ["accuracy", "macro_f1", "weighted_f1"]:
        table_df[col] = table_df[col].map(lambda value: f"{float(value):.4f}")

    fig, ax = plt.subplots(figsize=(12, max(3, 0.45 * len(table_df) + 1)))
    ax.axis("off")
    table = ax.table(
        cellText=table_df.values,
        colLabels=table_df.columns,
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.4)
    plt.title("Image Candidate Model Metrics", pad=20)
    plt.tight_layout()
    plt.savefig(table_path, dpi=200)
    plt.close()

    return {
        "candidate_comparison_csv": str(csv_path),
        "candidate_comparison_chart": str(chart_path),
        "candidate_model_table": str(table_path),
    }


def save_class_distribution_chart(
    y: pd.Series,
) -> str:
    """
    Save class distribution bar chart.
    """
    output_path = VISUAL_REPORT_DIR / "image_class_distribution.png"

    counts = y.value_counts().reindex(CLASSES, fill_value=0)

    plt.figure(figsize=(8, 5))
    sns.barplot(
        x=counts.index,
        y=counts.values,
    )
    plt.title("Image Dataset Class Distribution")
    plt.xlabel("Class")
    plt.ylabel("Samples")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()

    return str(output_path)


def save_confusion_matrix_heatmap(
    confusion_matrix_values: list[list[int]],
) -> str:
    """
    Save confusion matrix heatmap.
    """
    output_path = VISUAL_REPORT_DIR / "image_confusion_matrix.png"

    cm_df = pd.DataFrame(
        confusion_matrix_values,
        index=CLASSES,
        columns=CLASSES,
    )

    plt.figure(figsize=(7, 6))
    sns.heatmap(
        cm_df,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=True,
    )
    plt.title("Image Model Confusion Matrix")
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()

    return str(output_path)


def save_classification_report_visuals(
    report_dict: dict[str, Any],
) -> dict[str, str]:
    """
    Save classification report CSV and heatmap.
    """
    csv_path = VISUAL_REPORT_DIR / "image_classification_report.csv"
    heatmap_path = VISUAL_REPORT_DIR / "image_classification_report_heatmap.png"

    rows = []

    for label in CLASSES:
        metrics = report_dict.get(label, {})
        rows.append(
            {
                "class": label,
                "precision": float(metrics.get("precision", 0.0)),
                "recall": float(metrics.get("recall", 0.0)),
                "f1_score": float(metrics.get("f1-score", 0.0)),
                "support": int(metrics.get("support", 0)),
            }
        )

    df = pd.DataFrame(rows)
    df.to_csv(csv_path, index=False)

    heatmap_df = df.set_index("class")[["precision", "recall", "f1_score"]]

    plt.figure(figsize=(7, 5))
    sns.heatmap(
        heatmap_df,
        annot=True,
        fmt=".3f",
        cmap="YlGnBu",
        vmin=0,
        vmax=1,
    )
    plt.title("Image Classification Report")
    plt.tight_layout()
    plt.savefig(heatmap_path, dpi=200)
    plt.close()

    return {
        "classification_report_csv": str(csv_path),
        "classification_report_heatmap": str(heatmap_path),
    }


def generate_visual_reports(
    candidate_results: dict[str, dict[str, Any]],
    best_metrics: dict[str, Any],
    y_all: pd.Series,
) -> dict[str, str]:
    """
    Generate all visual evaluation outputs.
    """
    outputs: dict[str, str] = {}

    outputs.update(save_candidate_comparison_visuals(candidate_results))
    outputs["confusion_matrix_heatmap"] = save_confusion_matrix_heatmap(
        best_metrics["confusion_matrix"]
    )
    outputs.update(
        save_classification_report_visuals(
            best_metrics["classification_report"]
        )
    )
    outputs["class_distribution_chart"] = save_class_distribution_chart(y_all)

    return outputs


# ============================================================
# Saving
# ============================================================

def save_metadata(
    best_name: str,
    best_metrics: dict[str, Any],
    candidate_results: dict[str, dict[str, Any]],
    df: pd.DataFrame,
    data_path: Path,
    selected_features: list[str],
    split_method: str,
    y_train: pd.Series,
    y_test: pd.Series,
    visual_report_paths: dict[str, str],
) -> None:
    """
    Save JSON metadata.
    """
    metadata = {
        "model_name": f"image_{best_name}",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "dataset_path": str(data_path),
        "model_path": str(MODEL_PATH),
        "target": TARGET,
        "classes": CLASSES,
        "feature_mode": FEATURE_MODE,
        "features": selected_features,
        "feature_count": len(selected_features),
        "selection_metric": "macro_f1_then_accuracy",
        "selected_classifier": best_name,

        "accuracy": best_metrics["accuracy"],
        "macro_precision": best_metrics["macro_precision"],
        "macro_recall": best_metrics["macro_recall"],
        "macro_f1": best_metrics["macro_f1"],
        "weighted_precision": best_metrics["weighted_precision"],
        "weighted_recall": best_metrics["weighted_recall"],
        "weighted_f1": best_metrics["weighted_f1"],

        "method": f"safe_processed_visual_features_plus_{best_name}",
        "split_method": split_method,
        "test_size": TEST_SIZE,
        "random_state": RANDOM_STATE,

        "original_rows": int(df.attrs.get("original_rows", len(df))),
        "rows_used": int(len(df)),
        "train_samples": int(len(y_train)),
        "test_samples": int(len(y_test)),

        "class_distribution": df[TARGET].value_counts().reindex(CLASSES, fill_value=0).to_dict(),
        "train_class_distribution": y_train.value_counts().reindex(CLASSES, fill_value=0).to_dict(),
        "test_class_distribution": y_test.value_counts().reindex(CLASSES, fill_value=0).to_dict(),

        "quality_filter": sorted(list(RECOMMENDED_QUALITY)),
        "quality_filter_used": bool(df.attrs.get("quality_filter_used", False)),

        "excluded_leakage_columns": sorted(list(LEAKAGE_OR_NON_FEATURE_COLUMNS)),
        "candidate_model_comparison": candidate_summary(candidate_results),
        "visual_reports": visual_report_paths,

        "note": (
            "This image model uses a safer feature selection strategy. It excludes "
            "direct visual-fusion outputs, majority-vote fields, agreement-with-folder "
            "columns, quality flags, reliability scores, captions, and pseudo-label "
            "text fields. It retains detection metadata, raw CLIP prompt score columns, "
            "and one-hot encoded MediaPipe posture/head-position cues. This is intended "
            "to improve performance over metadata-only training while avoiding the most "
            "problematic leakage columns."
        ),
    }

    with open(META_PATH, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=4)


def save_training_report(
    best_name: str,
    best_metrics: dict[str, Any],
    candidate_results: dict[str, dict[str, Any]],
    df: pd.DataFrame,
    data_path: Path,
    selected_features: list[str],
    split_method: str,
    y_train: pd.Series,
    y_test: pd.Series,
    visual_report_paths: dict[str, str],
) -> None:
    """
    Save human-readable text report.
    """
    sorted_candidates = sorted(
        candidate_results.items(),
        key=lambda item: (
            item[1].get("macro_f1", 0.0),
            item[1].get("accuracy", 0.0),
        ),
        reverse=True,
    )

    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write("Image Model Training and Selection Report\n")
        f.write("=========================================\n\n")

        f.write(f"Created at: {datetime.now().isoformat(timespec='seconds')}\n")
        f.write(f"Dataset: {data_path}\n")
        f.write(f"Model artifact: {MODEL_PATH}\n")
        f.write(f"Feature mode: {FEATURE_MODE}\n")
        f.write(f"Original rows: {df.attrs.get('original_rows', len(df))}\n")
        f.write(f"Rows used: {len(df)}\n")
        f.write(f"Split method: {split_method}\n")
        f.write(f"Train samples: {len(y_train)}\n")
        f.write(f"Test samples: {len(y_test)}\n")
        f.write("Selection rule: highest macro F1, then highest accuracy\n\n")

        f.write("Class distribution:\n")
        f.write(str(df[TARGET].value_counts().reindex(CLASSES, fill_value=0)))
        f.write("\n\n")

        if "quality_flag" in df.columns:
            f.write("Quality distribution after filtering:\n")
            f.write(str(df["quality_flag"].value_counts()))
            f.write("\n\n")

        f.write("Candidate Model Comparison\n")
        f.write("--------------------------\n")
        f.write(
            f"{'Candidate':<28} {'Status':<10} "
            f"{'Accuracy':>10} {'Macro F1':>10} {'Weighted F1':>12}\n"
        )

        for name, metrics in sorted_candidates:
            f.write(
                f"{name:<28} {metrics.get('status', 'evaluated'):<10} "
                f"{metrics.get('accuracy', 0.0):>10.4f} "
                f"{metrics.get('macro_f1', 0.0):>10.4f} "
                f"{metrics.get('weighted_f1', 0.0):>12.4f}\n"
            )

        f.write(f"\nSelected model: {best_name}\n\n")

        f.write("Selected Model Metrics\n")
        f.write("----------------------\n")
        f.write(f"Accuracy: {best_metrics['accuracy']:.4f}\n")
        f.write(f"Macro Precision: {best_metrics['macro_precision']:.4f}\n")
        f.write(f"Macro Recall: {best_metrics['macro_recall']:.4f}\n")
        f.write(f"Macro F1: {best_metrics['macro_f1']:.4f}\n")
        f.write(f"Weighted Precision: {best_metrics['weighted_precision']:.4f}\n")
        f.write(f"Weighted Recall: {best_metrics['weighted_recall']:.4f}\n")
        f.write(f"Weighted F1: {best_metrics['weighted_f1']:.4f}\n\n")

        f.write("Selected Model Classification Report\n")
        f.write("------------------------------------\n")
        f.write(best_metrics["classification_report_text"])

        f.write("\nConfusion matrix labels:\n")
        f.write(str(CLASSES))

        f.write("\n\nConfusion matrix:\n")
        f.write(str(best_metrics["confusion_matrix"]))

        f.write("\n\nSelected feature columns\n")
        f.write("------------------------\n")
        for feature in selected_features:
            f.write(f"- {feature}\n")

        f.write("\nVisual report files\n")
        f.write("-------------------\n")
        for name, path in visual_report_paths.items():
            f.write(f"{name}: {path}\n")

        f.write("\nMethodological note\n")
        f.write("-------------------\n")
        f.write(
            "This safer image training script avoids direct visual-fusion labels, "
            "quality/reliability fields, majority-vote fields, captions, and "
            "agreement-with-folder columns. The purpose is to improve performance "
            "beyond strict metadata-only training while avoiding the strongest "
            "sources of feature leakage. If performance is lower than the earlier "
            "100% result, this is expected and more defensible.\n"
        )


# ============================================================
# Main execution
# ============================================================

def main() -> None:
    print("==========================================")
    print("SenseFuzeAI Image Model Training Pipeline")
    print("==========================================")

    data_path = resolve_data_path()
    print(f"Dataset: {data_path}")
    print(f"Feature mode: {FEATURE_MODE}")

    df = load_dataset(data_path)
    validate_class_distribution(df)

    base_feature_cols = get_base_numeric_features(df)

    if not base_feature_cols:
        raise ValueError(
            "No usable image feature columns found. "
            "Check processed_image_dataset.csv and feature selection settings."
        )

    X, selected_features = prepare_feature_dataframe(df, base_feature_cols)

    print(f"\nRows after filtering: {len(df)}")
    print(f"Number of selected feature columns: {len(selected_features)}")

    print("\nSelected feature columns:")
    for feature in selected_features:
        print(f"  - {feature}")

    print("\nClass distribution:")
    print(df[TARGET].value_counts().reindex(CLASSES, fill_value=0))

    if "quality_flag" in df.columns:
        print("\nQuality distribution after filtering:")
        print(df["quality_flag"].value_counts())

    X_train, X_test, y_train, y_test, split_method = split_dataset(df, X)

    print("\nTrain class distribution:")
    print(y_train.value_counts().reindex(CLASSES, fill_value=0))

    print("\nTest class distribution:")
    print(y_test.value_counts().reindex(CLASSES, fill_value=0))

    print("\nTraining and comparing image classifiers...")
    best_name, best_model, candidate_results = evaluate_candidates(
        X_train,
        X_test,
        y_train,
        y_test,
    )

    best_metrics = candidate_results[best_name]

    print("\nBest image model selected:")
    print(f"  Model: {best_name}")
    print(f"  Accuracy: {best_metrics['accuracy']:.4f}")
    print(f"  Macro F1: {best_metrics['macro_f1']:.4f}")
    print(f"  Weighted F1: {best_metrics['weighted_f1']:.4f}")

    print("\nGenerating visual training reports...")
    visual_report_paths = generate_visual_reports(
        candidate_results=candidate_results,
        best_metrics=best_metrics,
        y_all=df[TARGET],
    )

    print("Saving selected image model artifact...")
    joblib.dump(best_model, MODEL_PATH)

    save_metadata(
        best_name=best_name,
        best_metrics=best_metrics,
        candidate_results=candidate_results,
        df=df,
        data_path=data_path,
        selected_features=selected_features,
        split_method=split_method,
        y_train=y_train,
        y_test=y_test,
        visual_report_paths=visual_report_paths,
    )

    save_training_report(
        best_name=best_name,
        best_metrics=best_metrics,
        candidate_results=candidate_results,
        df=df,
        data_path=data_path,
        selected_features=selected_features,
        split_method=split_method,
        y_train=y_train,
        y_test=y_test,
        visual_report_paths=visual_report_paths,
    )

    print("\nSelected model classification report:")
    print(best_metrics["classification_report_text"])

    print("\nSaved:")
    print(MODEL_PATH)
    print(META_PATH)
    print(REPORT_PATH)

    print("\nVisual reports:")
    for name, path in visual_report_paths.items():
        print(f"{name}: {path}")

    print(
        "\nNote: This version is safer than the earlier high-accuracy version "
        "because it excludes direct fused-label, reliability, quality, majority-vote, "
        "agreement, and caption fields. Accuracy should improve over strict metadata-only "
        "training, but may remain lower than the earlier leaked-feature result."
    )


if __name__ == "__main__":
    main() 
