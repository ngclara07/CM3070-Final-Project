# === train_audio_model.py ===
# SenseFuzeAI - Audio Model Training and Selection Pipeline
#
# Trains and compares multiple classifiers for the audio modality.
#
# Candidate algorithms:
#   - SVM
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
#   data/processed/processed_audio_dataset.csv
#
# Output:
#   model_artifacts/audio_model.joblib
#   model_artifacts/audio_model_meta.json
#   model_artifacts/audio_training_report.txt
#   model_artifacts/visual_reports/audio/*.png
#   model_artifacts/visual_reports/audio/*.csv

from __future__ import annotations

import json
from pathlib import Path
from datetime import datetime
from typing import Any

import joblib
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
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from utils.training_visuals import generate_training_visuals


# ============================================================
# Configuration
# ============================================================

DATA_PATH = Path("data/processed/processed_audio_dataset.csv")

MODEL_DIR = Path("model_artifacts")
MODEL_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATH = MODEL_DIR / "audio_model.joblib"
META_PATH = MODEL_DIR / "audio_model_meta.json"
REPORT_PATH = MODEL_DIR / "audio_training_report.txt"

VISUAL_REPORT_DIR = MODEL_DIR / "visual_reports"

TARGET = "label"

CLASSES = [
    "focused",
    "distracted",
    "fatigued",
    "overloaded",
]

RANDOM_STATE = 42
TEST_SIZE = 0.20

DROP_COLUMNS = {
    "filepath",
    "filename",
    "label",
    "file_extension",
    "split",
    "error",
}


# ============================================================
# Utility functions
# ============================================================

def normalise_label(value: Any) -> str:
    """
    Convert labels into the canonical SenseFuzeAI lowercase format.
    """
    if pd.isna(value):
        return ""

    return str(value).strip().lower()


def remove_error_rows(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove rows marked with preprocessing errors, if an error column exists.
    """
    if "error" not in df.columns:
        return df

    error_values = df["error"].astype(str).str.strip().str.lower()

    return df[
        df["error"].isna()
        | (error_values == "")
        | (error_values == "nan")
        | (error_values == "none")
    ].copy()


# ============================================================
# Data loading and validation
# ============================================================

def load_dataset() -> pd.DataFrame:
    """
    Load and clean the processed audio dataset.
    """
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Missing file: {DATA_PATH}\n"
            "Run process_audio_data.py first."
        )

    df = pd.read_csv(DATA_PATH)

    if TARGET not in df.columns:
        raise ValueError("Audio CSV must contain a 'label' column.")

    df[TARGET] = df[TARGET].apply(normalise_label)
    df = df[df[TARGET].isin(CLASSES)].copy()
    df = remove_error_rows(df)

    if df.empty:
        raise ValueError("No valid audio samples available after filtering.")

    return df


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    """
    Select numeric feature columns for model training.
    """
    numeric_cols = df.select_dtypes(include=["number", "bool"]).columns.tolist()

    return [
        col for col in numeric_cols
        if col not in DROP_COLUMNS
    ]


def validate_class_distribution(df: pd.DataFrame) -> None:
    """
    Validate that classes are suitable for stratified splitting.
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


def split_dataset(
    df: pd.DataFrame,
    feature_cols: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, str]:
    """
    Use an existing train/test split if available.

    If the split column is absent or incomplete, create a stratified split.
    """
    if "split" in df.columns:
        split_values = df["split"].astype(str).str.strip().str.lower()

        train_df = df[split_values == "train"].copy()
        test_df = df[split_values == "test"].copy()

        if not train_df.empty and not test_df.empty:
            print("Using existing train/test split from audio dataset.")

            return (
                train_df[feature_cols],
                test_df[feature_cols],
                train_df[TARGET],
                test_df[TARGET],
                "existing_split_column",
            )

        print(
            "Warning: split column exists but train/test rows are incomplete. "
            "Using stratified train_test_split instead."
        )

    print("Using stratified train_test_split.")

    X_train, X_test, y_train, y_test = train_test_split(
        df[feature_cols],
        df[TARGET],
        test_size=TEST_SIZE,
        stratify=df[TARGET],
        random_state=RANDOM_STATE,
    )

    return X_train, X_test, y_train, y_test, "stratified_train_test_split"


# ============================================================
# Candidate model construction
# ============================================================

def build_candidate_pipelines() -> dict[str, Pipeline]:
    """
    Build candidate classifiers for audio model selection.
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
                ("scaler", StandardScaler()),
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
                ("scaler", StandardScaler()),
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
# Metrics and model selection
# ============================================================

def compute_metrics(y_true: pd.Series, y_pred: Any) -> dict[str, Any]:
    """
    Compute multi-class classification metrics.
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
    Train all candidate models and select the best-performing classifier.
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
        raise RuntimeError("No candidate audio model was successfully trained.")

    return best_name, best_model, results


def candidate_summary(
    candidate_results: dict[str, dict[str, Any]]
) -> dict[str, dict[str, Any]]:
    """
    Create a compact metadata-safe summary of candidate results.
    """
    summary: dict[str, dict[str, Any]] = {}

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
# Saving metadata and reports
# ============================================================

def save_metadata(
    best_name: str,
    best_metrics: dict[str, Any],
    candidate_results: dict[str, dict[str, Any]],
    df: pd.DataFrame,
    feature_cols: list[str],
    split_method: str,
    y_train: pd.Series,
    y_test: pd.Series,
    visual_paths: dict[str, str],
) -> None:
    """
    Save machine-readable metadata for the selected audio model.
    """
    metadata = {
        "model_name": f"audio_{best_name}",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "dataset_path": str(DATA_PATH),
        "model_path": str(MODEL_PATH),
        "target": TARGET,
        "classes": CLASSES,
        "features": feature_cols,
        "selection_metric": "macro_f1_then_accuracy",
        "selected_classifier": best_name,
        "accuracy": best_metrics["accuracy"],
        "macro_precision": best_metrics["macro_precision"],
        "macro_recall": best_metrics["macro_recall"],
        "macro_f1": best_metrics["macro_f1"],
        "weighted_precision": best_metrics["weighted_precision"],
        "weighted_recall": best_metrics["weighted_recall"],
        "weighted_f1": best_metrics["weighted_f1"],
        "method": f"librosa_handcrafted_audio_features_plus_{best_name}",
        "split_method": split_method,
        "test_size": TEST_SIZE,
        "random_state": RANDOM_STATE,
        "rows_used": int(len(df)),
        "train_samples": int(len(y_train)),
        "test_samples": int(len(y_test)),
        "class_distribution": (
            df[TARGET]
            .value_counts()
            .reindex(CLASSES, fill_value=0)
            .to_dict()
        ),
        "train_class_distribution": (
            y_train
            .value_counts()
            .reindex(CLASSES, fill_value=0)
            .to_dict()
        ),
        "test_class_distribution": (
            y_test
            .value_counts()
            .reindex(CLASSES, fill_value=0)
            .to_dict()
        ),
        "candidate_model_comparison": candidate_summary(candidate_results),
        "visual_reports": visual_paths,
        "note": (
            "This trained audio model uses processed Librosa handcrafted audio "
            "features. Multiple classifiers were compared, and the saved "
            "audio_model.joblib is the best candidate by macro F1-score, "
            "with accuracy used as the tie-breaker. Visual report artefacts "
            "are also generated for use in the final report and prototype video."
        ),
    }

    with open(META_PATH, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=4)


def save_training_report(
    best_name: str,
    best_metrics: dict[str, Any],
    candidate_results: dict[str, dict[str, Any]],
    df: pd.DataFrame,
    feature_cols: list[str],
    split_method: str,
    y_train: pd.Series,
    y_test: pd.Series,
    visual_paths: dict[str, str],
) -> None:
    """
    Save a human-readable audio training report.
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
        f.write("Audio Model Training and Selection Report\n")
        f.write("=========================================\n\n")
        f.write(f"Created at: {datetime.now().isoformat(timespec='seconds')}\n")
        f.write(f"Dataset: {DATA_PATH}\n")
        f.write(f"Model artifact: {MODEL_PATH}\n")
        f.write(f"Rows used: {len(df)}\n")
        f.write(f"Split method: {split_method}\n")
        f.write(f"Train samples: {len(y_train)}\n")
        f.write(f"Test samples: {len(y_test)}\n")
        f.write("Selection rule: highest macro F1, then highest accuracy\n\n")

        f.write("Class distribution:\n")
        f.write(str(df[TARGET].value_counts().reindex(CLASSES, fill_value=0)))
        f.write("\n\n")

        f.write("Train class distribution:\n")
        f.write(str(y_train.value_counts().reindex(CLASSES, fill_value=0)))
        f.write("\n\n")

        f.write("Test class distribution:\n")
        f.write(str(y_test.value_counts().reindex(CLASSES, fill_value=0)))
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

        f.write("\n\nFeature columns:\n")
        for feature in feature_cols:
            f.write(f"- {feature}\n")

        f.write("\nVisual Report Artefacts\n")
        f.write("-----------------------\n")
        for name, path in visual_paths.items():
            f.write(f"{name}: {path}\n")

        f.write("\nMethodological note:\n")
        f.write(
            "The audio model selection process compares several multi-class "
            "classification algorithms. The selected model is chosen using "
            "macro F1-score as the primary criterion because all four behaviour "
            "classes are important. Accuracy is used only as a tie-breaker."
        )


# ============================================================
# Main execution
# ============================================================

def main() -> None:
    print("==========================================")
    print("SenseFuzeAI Audio Model Training Pipeline")
    print("==========================================")

    df = load_dataset()
    validate_class_distribution(df)

    feature_cols = get_feature_columns(df)

    if not feature_cols:
        raise ValueError("No numeric audio feature columns found.")

    print(f"\nRows after filtering: {len(df)}")
    print(f"Number of feature columns: {len(feature_cols)}")

    print("\nClass distribution:")
    print(df[TARGET].value_counts().reindex(CLASSES, fill_value=0))

    X_train, X_test, y_train, y_test, split_method = split_dataset(
        df,
        feature_cols,
    )

    print("\nTrain class distribution:")
    print(y_train.value_counts().reindex(CLASSES, fill_value=0))

    print("\nTest class distribution:")
    print(y_test.value_counts().reindex(CLASSES, fill_value=0))

    print("\nTraining and comparing audio classifiers...")
    best_name, best_model, candidate_results = evaluate_candidates(
        X_train,
        X_test,
        y_train,
        y_test,
    )

    best_metrics = candidate_results[best_name]

    print("\nBest audio model selected:")
    print(f"  Model: {best_name}")
    print(f"  Accuracy: {best_metrics['accuracy']:.4f}")
    print(f"  Macro F1: {best_metrics['macro_f1']:.4f}")
    print(f"  Weighted F1: {best_metrics['weighted_f1']:.4f}")

    print("\nGenerating visual training reports...")
    visual_paths = generate_training_visuals(
        modality_name="audio",
        candidate_results=candidate_results,
        best_metrics=best_metrics,
        y_all=df[TARGET],
        class_labels=CLASSES,
        output_root=VISUAL_REPORT_DIR,
    )

    print("Saving selected audio model artifact...")
    joblib.dump(best_model, MODEL_PATH)

    save_metadata(
        best_name=best_name,
        best_metrics=best_metrics,
        candidate_results=candidate_results,
        df=df,
        feature_cols=feature_cols,
        split_method=split_method,
        y_train=y_train,
        y_test=y_test,
        visual_paths=visual_paths,
    )

    save_training_report(
        best_name=best_name,
        best_metrics=best_metrics,
        candidate_results=candidate_results,
        df=df,
        feature_cols=feature_cols,
        split_method=split_method,
        y_train=y_train,
        y_test=y_test,
        visual_paths=visual_paths,
    )

    print("\nSelected model classification report:")
    print(best_metrics["classification_report_text"])

    print("\nSaved:")
    print(MODEL_PATH)
    print(META_PATH)
    print(REPORT_PATH)

    print("\nVisual reports:")
    for name, path in visual_paths.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
