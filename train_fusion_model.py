# === train_fusion_model.py ===
# SenseFuzeAI - Experimental Fusion Model Training and Selection Pipeline
#
# Builds an experimental late-fusion dataset using prediction outputs from:
#   audio_model.joblib
#   image_model.joblib
#   keystroke_model.joblib
#   text_model.joblib
#
# Then trains and compares multiple classifiers:
#   - SVM
#   - Random Forest
#   - Logistic Regression
#   - Decision Tree
#   - Gaussian Naive Bayes
#   - kNN
#
# Important methodological note:
#   This fusion model is experimental because the modality datasets are not
#   necessarily session-aligned. The main application can still use weighted
#   late fusion as the recommended runtime method.
#
# Input:
#   model_artifacts/audio_model.joblib
#   model_artifacts/image_model.joblib
#   model_artifacts/keystroke_model.joblib
#   model_artifacts/text_model.joblib
#
# Output:
#   model_artifacts/fusion_model.joblib
#   model_artifacts/fusion_model_meta.json
#   model_artifacts/fusion_training_report.txt
#   model_artifacts/fusion_training_dataset.csv
#   model_artifacts/visual_reports/fusion/*.png
#   model_artifacts/visual_reports/fusion/*.csv

from __future__ import annotations

import json
from pathlib import Path
from datetime import datetime
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
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from utils.training_visuals import generate_training_visuals


# ============================================================
# Paths
# ============================================================

MODEL_DIR = Path("model_artifacts")
MODEL_DIR.mkdir(parents=True, exist_ok=True)

AUDIO_DATA = Path("data/processed/processed_audio_dataset.csv")

IMAGE_DATA_CANDIDATES = [
    Path("data/processed/processed_image_dataset_v2.csv"),
    Path("data/processed/processed_image_dataset.csv"),
]

KEYSTROKE_DATA = Path("emosurv_processed/combined_behaviour_samples.csv")
TEXT_DATA = Path("data/processed/processed_text_dataset.csv")

AUDIO_MODEL = MODEL_DIR / "audio_model.joblib"
AUDIO_META = MODEL_DIR / "audio_model_meta.json"

IMAGE_MODEL = MODEL_DIR / "image_model.joblib"
IMAGE_META = MODEL_DIR / "image_model_meta.json"

KEYSTROKE_MODEL = MODEL_DIR / "keystroke_model.joblib"
KEYSTROKE_META = MODEL_DIR / "keystroke_model_meta.json"

TEXT_MODEL = MODEL_DIR / "text_model.joblib"
TEXT_META = MODEL_DIR / "text_model_meta.json"

FUSION_MODEL = MODEL_DIR / "fusion_model.joblib"
FUSION_META = MODEL_DIR / "fusion_model_meta.json"
FUSION_REPORT = MODEL_DIR / "fusion_training_report.txt"
FUSION_DATASET_OUT = MODEL_DIR / "fusion_training_dataset.csv"

VISUAL_REPORT_DIR = MODEL_DIR / "visual_reports"


# ============================================================
# Configuration
# ============================================================

CLASSES = [
    "focused",
    "distracted",
    "fatigued",
    "overloaded",
]

RANDOM_STATE = 42
TEST_SIZE = 0.20


# ============================================================
# Utility functions
# ============================================================

def resolve_existing_path(candidates: list[Path]) -> Path | None:
    """
    Return the first existing path from a list of candidate paths.
    """
    for path in candidates:
        if path.exists():
            return path

    return None


def load_json(path: Path) -> dict[str, Any]:
    """
    Safely load a JSON file.
    """
    if not path.exists():
        return {}

    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def normalise_label(value: Any) -> str:
    """
    Convert labels into canonical lowercase SenseFuzeAI format.
    """
    if pd.isna(value):
        return ""

    return str(value).strip().lower()


def probability_columns(prefix: str) -> list[str]:
    """
    Return the standard probability-column names for one modality.
    """
    return [
        f"{prefix}_{label}_prob"
        for label in CLASSES
    ]


def empty_prob_row(prefix: str) -> dict[str, float]:
    """
    Return an empty probability row for one modality.
    """
    return {
        col: 0.0
        for col in probability_columns(prefix)
    }


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


def get_model_classes(model: Any) -> list[str]:
    """
    Extract class ordering from a scikit-learn estimator or pipeline.
    """
    classes = list(getattr(model, "classes_", []))

    if not classes and hasattr(model, "named_steps"):
        final_estimator = list(model.named_steps.values())[-1]
        classes = list(getattr(final_estimator, "classes_", []))

    return [
        str(label)
        for label in classes
    ]


def softmax(values: np.ndarray) -> np.ndarray:
    """
    Convert arbitrary decision scores into probability-like values.
    """
    values = np.asarray(values, dtype=float)
    values = values - np.max(values)
    exp_values = np.exp(values)
    total = np.sum(exp_values)

    if total <= 0:
        return np.ones_like(exp_values) / len(exp_values)

    return exp_values / total


def prediction_scores_as_dict(
    model: Any,
    X: Any,
    prefix: str,
) -> pd.DataFrame:
    """
    Convert a modality model's output into class score columns.

    Priority:
      1. predict_proba(), if available
      2. decision_function(), converted through softmax
      3. one-hot encoding from predict()

    This makes the fusion pipeline compatible with classifiers such as
    LinearSVC, which do not provide predict_proba().
    """
    model_classes = get_model_classes(model)

    if not model_classes:
        model_classes = CLASSES

    rows: list[dict[str, float]] = []

    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(X)

        for row_probs in probabilities:
            row = empty_prob_row(prefix)

            for label, prob in zip(model_classes, row_probs):
                label = str(label)

                if label in CLASSES:
                    row[f"{prefix}_{label}_prob"] = float(prob)

            rows.append(row)

        return pd.DataFrame(rows)

    if hasattr(model, "decision_function"):
        decision_scores = model.decision_function(X)
        decision_scores = np.asarray(decision_scores)

        if decision_scores.ndim == 1:
            decision_scores = np.vstack([-decision_scores, decision_scores]).T

        for row_scores in decision_scores:
            row = empty_prob_row(prefix)
            probabilities = softmax(row_scores)

            for label, prob in zip(model_classes, probabilities):
                label = str(label)

                if label in CLASSES:
                    row[f"{prefix}_{label}_prob"] = float(prob)

            rows.append(row)

        return pd.DataFrame(rows)

    predictions = model.predict(X)

    for prediction in predictions:
        row = empty_prob_row(prefix)
        prediction = str(prediction)

        if prediction in CLASSES:
            row[f"{prefix}_{prediction}_prob"] = 1.0
        else:
            for label in CLASSES:
                row[f"{prefix}_{label}_prob"] = 1.0 / len(CLASSES)

        rows.append(row)

    return pd.DataFrame(rows)


def ensure_probability_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure all modality score columns exist.
    """
    df = df.copy()

    for prefix in ["audio", "image", "keystroke", "text"]:
        for col in probability_columns(prefix):
            if col not in df.columns:
                df[col] = 0.0

    return df


def add_modality_presence_flags(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add binary flags indicating which modality contributed prediction scores.
    """
    df = df.copy()

    df["has_audio"] = (
        df[probability_columns("audio")].sum(axis=1) > 0
    ).astype(int)

    df["has_image"] = (
        df[probability_columns("image")].sum(axis=1) > 0
    ).astype(int)

    df["has_keystroke"] = (
        df[probability_columns("keystroke")].sum(axis=1) > 0
    ).astype(int)

    df["has_text"] = (
        df[probability_columns("text")].sum(axis=1) > 0
    ).astype(int)

    return df


# ============================================================
# Modality fusion-row builders
# ============================================================

def build_audio_fusion_rows() -> pd.DataFrame:
    """
    Build fusion rows from the trained audio model.
    """
    if not AUDIO_DATA.exists():
        print(f"Audio dataset not found: {AUDIO_DATA}")
        return pd.DataFrame()

    if not AUDIO_MODEL.exists() or not AUDIO_META.exists():
        print("Audio model or metadata not found.")
        return pd.DataFrame()

    df = pd.read_csv(AUDIO_DATA)
    meta = load_json(AUDIO_META)
    model = joblib.load(AUDIO_MODEL)

    if "label" not in df.columns:
        print("Audio dataset missing label column.")
        return pd.DataFrame()

    df["label"] = df["label"].apply(normalise_label)
    df = remove_error_rows(df)
    df = df[df["label"].isin(CLASSES)].copy()

    if df.empty:
        print("No valid audio rows available for fusion.")
        return pd.DataFrame()

    features = meta.get("features", [])

    if not features:
        print("Audio metadata missing feature list.")
        return pd.DataFrame()

    X = df.reindex(columns=features, fill_value=0.0)

    for col in X.columns:
        X[col] = pd.to_numeric(X[col], errors="coerce").fillna(0.0)

    prob_df = prediction_scores_as_dict(model, X, "audio")

    output = pd.concat(
        [
            df[["label"]].reset_index(drop=True),
            prob_df.reset_index(drop=True),
        ],
        axis=1,
    )

    output["source_modality"] = "audio"

    return output


def build_image_fusion_rows() -> pd.DataFrame:
    """
    Build fusion rows from the trained image model.
    """
    image_data = resolve_existing_path(IMAGE_DATA_CANDIDATES)

    if image_data is None:
        print("Image dataset not found.")
        return pd.DataFrame()

    if not IMAGE_MODEL.exists() or not IMAGE_META.exists():
        print("Image model or metadata not found.")
        return pd.DataFrame()

    df = pd.read_csv(image_data)
    meta = load_json(IMAGE_META)
    model = joblib.load(IMAGE_MODEL)

    if "label" not in df.columns:
        print("Image dataset missing label column.")
        return pd.DataFrame()

    df["label"] = df["label"].apply(normalise_label)
    df = remove_error_rows(df)
    df = df[df["label"].isin(CLASSES)].copy()

    if df.empty:
        print("No valid image rows available for fusion.")
        return pd.DataFrame()

    features = meta.get("features", [])

    if not features:
        print("Image metadata missing feature list.")
        return pd.DataFrame()

    X = df.reindex(columns=features, fill_value=0.0)

    for col in X.columns:
        X[col] = pd.to_numeric(X[col], errors="coerce").fillna(0.0)

    prob_df = prediction_scores_as_dict(model, X, "image")

    output = pd.concat(
        [
            df[["label"]].reset_index(drop=True),
            prob_df.reset_index(drop=True),
        ],
        axis=1,
    )

    output["source_modality"] = "image"

    return output


def build_keystroke_fusion_rows() -> pd.DataFrame:
    """
    Build fusion rows from the trained keystroke model.
    """
    if not KEYSTROKE_DATA.exists():
        print(f"Keystroke dataset not found: {KEYSTROKE_DATA}")
        return pd.DataFrame()

    if not KEYSTROKE_MODEL.exists() or not KEYSTROKE_META.exists():
        print("Keystroke model or metadata not found.")
        return pd.DataFrame()

    df = pd.read_csv(KEYSTROKE_DATA)
    meta = load_json(KEYSTROKE_META)
    model = joblib.load(KEYSTROKE_MODEL)

    target = meta.get("target", "behaviour_state")
    features = meta.get("features", [])

    if target not in df.columns:
        print(f"Keystroke dataset missing target column: {target}")
        return pd.DataFrame()

    if not features:
        print("Keystroke metadata missing feature list.")
        return pd.DataFrame()

    df[target] = df[target].apply(normalise_label)
    df = df[df[target].isin(CLASSES)].copy()

    if df.empty:
        print("No valid keystroke rows available for fusion.")
        return pd.DataFrame()

    X = df.reindex(columns=features, fill_value=0.0)

    for col in X.columns:
        X[col] = pd.to_numeric(X[col], errors="coerce").fillna(0.0)

    prob_df = prediction_scores_as_dict(model, X, "keystroke")

    output = pd.concat(
        [
            df[[target]]
            .rename(columns={target: "label"})
            .reset_index(drop=True),
            prob_df.reset_index(drop=True),
        ],
        axis=1,
    )

    output["source_modality"] = "keystroke"

    return output


def build_text_fusion_rows() -> pd.DataFrame:
    """
    Build fusion rows from the trained text model.

    Text pipeline:
        processed_text_dataset.csv
        -> text_model.joblib
        -> class scores/probabilities
    """
    if not TEXT_DATA.exists():
        print(f"Text dataset not found: {TEXT_DATA}")
        return pd.DataFrame()

    if not TEXT_MODEL.exists():
        print(f"Text model not found: {TEXT_MODEL}")
        return pd.DataFrame()

    df = pd.read_csv(TEXT_DATA)
    model = joblib.load(TEXT_MODEL)
    meta = load_json(TEXT_META)

    text_col = meta.get("text_column", "text")
    target_col = meta.get("target", "label")

    if text_col not in df.columns or target_col not in df.columns:
        print(
            f"Text dataset must contain '{text_col}' and '{target_col}' columns."
        )
        return pd.DataFrame()

    df[text_col] = df[text_col].fillna("").astype(str).str.strip()
    df[target_col] = df[target_col].apply(normalise_label)

    df = remove_error_rows(df)
    df = df[df[text_col] != ""].copy()
    df = df[df[target_col].isin(CLASSES)].copy()

    if df.empty:
        print("No valid text rows available for fusion.")
        return pd.DataFrame()

    prob_df = prediction_scores_as_dict(
        model=model,
        X=df[text_col].astype(str),
        prefix="text",
    )

    output = pd.concat(
        [
            df[[target_col]]
            .rename(columns={target_col: "label"})
            .reset_index(drop=True),
            prob_df.reset_index(drop=True),
        ],
        axis=1,
    )

    output["source_modality"] = "text"

    return output


# ============================================================
# Fusion dataset construction
# ============================================================

def build_fusion_dataset() -> pd.DataFrame:
    """
    Build the experimental fusion dataset.
    """
    parts = [
        build_audio_fusion_rows(),
        build_image_fusion_rows(),
        build_keystroke_fusion_rows(),
        build_text_fusion_rows(),
    ]

    parts = [
        part for part in parts
        if not part.empty
    ]

    if not parts:
        raise ValueError("No modality prediction data available for fusion training.")

    df = pd.concat(parts, ignore_index=True)

    df["label"] = df["label"].apply(normalise_label)
    df = df[df["label"].isin(CLASSES)].copy()

    df = ensure_probability_columns(df)
    df = add_modality_presence_flags(df)

    return df


def validate_fusion_dataset(df: pd.DataFrame) -> None:
    """
    Validate class coverage in the fusion dataset.
    """
    counts = df["label"].value_counts()

    missing_classes = [
        class_name for class_name in CLASSES
        if class_name not in counts.index
    ]

    if missing_classes:
        print(f"Warning: fusion dataset missing classes: {missing_classes}")

    too_small = counts[counts < 2]

    if not too_small.empty:
        raise ValueError(
            "Some classes have fewer than 2 samples. "
            "Stratified train/test split is not possible:\n"
            f"{too_small}"
        )


# ============================================================
# Candidate model construction
# ============================================================

def build_candidate_pipelines() -> dict[str, Pipeline]:
    """
    Build candidate classifiers for experimental fusion model selection.
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
    Compute multi-class fusion classification metrics.
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
    Train all candidate fusion models and select the best candidate.
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
        raise RuntimeError("No candidate fusion model was successfully trained.")

    return best_name, best_model, results


def candidate_summary(
    candidate_results: dict[str, dict[str, Any]]
) -> dict[str, dict[str, Any]]:
    """
    Build a compact candidate comparison summary for metadata.
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
    y_train: pd.Series,
    y_test: pd.Series,
    visual_paths: dict[str, str],
) -> None:
    """
    Save machine-readable metadata for the selected experimental fusion model.
    """
    metadata = {
        "model_name": f"experimental_multimodal_late_fusion_{best_name}",
        "created_at": datetime.now().isoformat(timespec="seconds"),
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
        "method": (
            "experimental_late_fusion_using_modality_prediction_outputs_"
            f"plus_{best_name}"
        ),
        "model_path": str(FUSION_MODEL),
        "fusion_dataset_path": str(FUSION_DATASET_OUT),
        "test_size": TEST_SIZE,
        "random_state": RANDOM_STATE,
        "rows_used": int(len(df)),
        "train_samples": int(len(y_train)),
        "test_samples": int(len(y_test)),
        "source_modality_distribution": (
            df["source_modality"]
            .value_counts()
            .to_dict()
        ),
        "class_distribution": (
            df["label"]
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
        "modality_models": {
            "audio_model": str(AUDIO_MODEL),
            "image_model": str(IMAGE_MODEL),
            "keystroke_model": str(KEYSTROKE_MODEL),
            "text_model": str(TEXT_MODEL),
        },
        "text_modality_note": (
            "The text modality uses the trained text_model.joblib artifact "
            "created by train_text_model.py. That artifact contains a TF-IDF "
            "vectorizer and the best selected classifier from the candidate "
            "model comparison. If the selected text classifier does not expose "
            "predict_proba(), this fusion script uses decision_function() or "
            "one-hot prediction fallback to generate fusion-compatible scores."
        ),
        "important_note": (
            "This fusion model is experimental. It is trained using stacked "
            "modality-level prediction rows rather than true session-aligned "
            "multimodal samples. The main application may still use weighted "
            "late fusion as the recommended runtime method. A stronger trained "
            "fusion model would require shared session_id values linking "
            "keystroke, text, audio, and image data from the same interaction."
        ),
    }

    with open(FUSION_META, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=4)


def save_training_report(
    best_name: str,
    best_metrics: dict[str, Any],
    candidate_results: dict[str, dict[str, Any]],
    df: pd.DataFrame,
    feature_cols: list[str],
    y_train: pd.Series,
    y_test: pd.Series,
    visual_paths: dict[str, str],
) -> None:
    """
    Save a human-readable experimental fusion training report.
    """
    sorted_candidates = sorted(
        candidate_results.items(),
        key=lambda item: (
            item[1].get("macro_f1", 0.0),
            item[1].get("accuracy", 0.0),
        ),
        reverse=True,
    )

    with open(FUSION_REPORT, "w", encoding="utf-8") as f:
        f.write("Experimental Fusion Model Training and Selection Report\n")
        f.write("======================================================\n\n")
        f.write(f"Created at: {datetime.now().isoformat(timespec='seconds')}\n")
        f.write(f"Fusion dataset: {FUSION_DATASET_OUT}\n")
        f.write(f"Model artifact: {FUSION_MODEL}\n")
        f.write(f"Rows used: {len(df)}\n")
        f.write(f"Train samples: {len(y_train)}\n")
        f.write(f"Test samples: {len(y_test)}\n")
        f.write("Selection rule: highest macro F1, then highest accuracy\n\n")

        f.write("Rows by source modality:\n")
        f.write(str(df["source_modality"].value_counts()))
        f.write("\n\n")

        f.write("Class distribution:\n")
        f.write(str(df["label"].value_counts().reindex(CLASSES, fill_value=0)))
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

        f.write("\nImportant methodological note:\n")
        f.write(
            "This fusion model is experimental because the rows are stacked "
            "from modality-level datasets and are not guaranteed to represent "
            "the same real user session. Weighted late fusion remains the "
            "recommended runtime fusion method unless session-aligned data is "
            "collected. The visual report artefacts are intended for project "
            "reporting, model-comparison evidence, and prototype demonstration."
        )


# ============================================================
# Main execution
# ============================================================

def main() -> None:
    print("==========================================")
    print("SenseFuzeAI Experimental Fusion Training")
    print("==========================================")

    print("\nBuilding fusion dataset...")
    df = build_fusion_dataset()

    feature_cols = (
        probability_columns("audio")
        + probability_columns("image")
        + probability_columns("keystroke")
        + probability_columns("text")
        + [
            "has_audio",
            "has_image",
            "has_keystroke",
            "has_text",
        ]
    )

    df = df.dropna(subset=["label"]).copy()
    validate_fusion_dataset(df)

    X = df[feature_cols]
    y = df["label"]

    print("\nFusion dataset size:", len(df))

    print("\nRows by source modality:")
    print(df["source_modality"].value_counts())

    print("\nClass distribution:")
    print(y.value_counts().reindex(CLASSES, fill_value=0))

    df.to_csv(FUSION_DATASET_OUT, index=False)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        stratify=y,
        random_state=RANDOM_STATE,
    )

    print("\nTrain class distribution:")
    print(y_train.value_counts().reindex(CLASSES, fill_value=0))

    print("\nTest class distribution:")
    print(y_test.value_counts().reindex(CLASSES, fill_value=0))

    print("\nTraining and comparing fusion classifiers...")
    best_name, best_model, candidate_results = evaluate_candidates(
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
    )

    best_metrics = candidate_results[best_name]

    print("\nBest fusion model selected:")
    print(f"  Model: {best_name}")
    print(f"  Accuracy: {best_metrics['accuracy']:.4f}")
    print(f"  Macro F1: {best_metrics['macro_f1']:.4f}")
    print(f"  Weighted F1: {best_metrics['weighted_f1']:.4f}")

    print("\nGenerating visual training reports...")
    visual_paths = generate_training_visuals(
        modality_name="fusion",
        candidate_results=candidate_results,
        best_metrics=best_metrics,
        y_all=df["label"],
        class_labels=CLASSES,
        output_root=VISUAL_REPORT_DIR,
    )

    print("Saving selected experimental fusion model artifact...")
    joblib.dump(best_model, FUSION_MODEL)

    save_metadata(
        best_name=best_name,
        best_metrics=best_metrics,
        candidate_results=candidate_results,
        df=df,
        feature_cols=feature_cols,
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
        y_train=y_train,
        y_test=y_test,
        visual_paths=visual_paths,
    )

    print("\nSelected model classification report:")
    print(best_metrics["classification_report_text"])

    print("\nSaved:")
    print(FUSION_MODEL)
    print(FUSION_META)
    print(FUSION_REPORT)
    print(FUSION_DATASET_OUT)

    print("\nVisual reports:")
    for name, path in visual_paths.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
