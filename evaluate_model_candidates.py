# === evaluate_model_candidates.py ===
# SenseFuzeAI - Full Model Candidate Evaluation Pipeline
#
# Purpose:
#   Generate defensible model-evaluation evidence for the final report.
#
# Evaluates:
#   1. Keystroke model
#   2. Text model
#   3. Audio model / YAMNet-assisted audio pipeline
#   4. Image / vision model pipeline
#   5. Experimental fusion model
#
# Outputs:
#   model_artifacts/model_evaluation/
#       overall_candidate_summary.csv
#       overall_candidate_summary.json
#       evaluation_report.txt
#       <modality>/
#           predictions_*.csv
#           classification_report_*.csv
#           confusion_matrix_*.png
#
# Important:
#   This script is for model evaluation evidence.
#   It does not replace the final runtime web-app fusion logic.
#
# Recommended use:
#   python evaluate_model_candidates.py
#   python evaluate_model_candidates.py --runtime-audio --audio-limit 40
#   python evaluate_model_candidates.py --runtime-image --image-limit 40
#   python evaluate_model_candidates.py --runtime-audio --runtime-image

from __future__ import annotations

import argparse
import json
import os
import sys
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import joblib
import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

try:
    import matplotlib.pyplot as plt
    import seaborn as sns

    PLOTTING_AVAILABLE = True
except Exception:
    PLOTTING_AVAILABLE = False


# ============================================================
# Project paths
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

MODEL_DIR = PROJECT_ROOT / "model_artifacts"

OUTPUT_DIR = MODEL_DIR / "model_evaluation"

BEHAVIOUR_CLASSES = [
    "focused",
    "distracted",
    "fatigued",
    "overloaded",
]


# ============================================================
# Dataset paths
# ============================================================

TEXT_DATASET = PROJECT_ROOT / "data" / "processed" / "processed_text_dataset.csv"
AUDIO_DATASET = PROJECT_ROOT / "data" / "processed" / "processed_audio_dataset.csv"
IMAGE_DATASET = PROJECT_ROOT / "data" / "processed" / "processed_image_dataset.csv"
KEYSTROKE_DATASET = PROJECT_ROOT / "emosurv_processed" / "combined_behaviour_samples.csv"
FUSION_DATASET = MODEL_DIR / "fusion_training_dataset.csv"


# ============================================================
# Model artifact paths
# ============================================================

TEXT_MODEL_PATH = MODEL_DIR / "text_model.joblib"
TEXT_META_PATH = MODEL_DIR / "text_model_meta.json"

AUDIO_MODEL_PATH = MODEL_DIR / "audio_model.joblib"
AUDIO_META_PATH = MODEL_DIR / "audio_model_meta.json"

IMAGE_MODEL_PATH = MODEL_DIR / "image_model.joblib"
IMAGE_META_PATH = MODEL_DIR / "image_model_meta.json"

KEYSTROKE_MODEL_PATH = MODEL_DIR / "keystroke_model.joblib"
KEYSTROKE_META_PATH = MODEL_DIR / "keystroke_model_meta.json"

FUSION_MODEL_PATH = MODEL_DIR / "fusion_model.joblib"
FUSION_META_PATH = MODEL_DIR / "fusion_model_meta.json"


# ============================================================
# Utility functions
# ============================================================

def print_header() -> None:
    print("==============================================")
    print("SenseFuzeAI Full Model Candidate Evaluation")
    print("==============================================")
    print(f"Project root: {PROJECT_ROOT}")
    print(f"Output dir:   {OUTPUT_DIR}")
    print("==============================================\n")


def normalise_label(value: Any) -> str:
    return str(value or "").strip().lower()


def ensure_output_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def load_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}

    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def safe_read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        warnings.warn(f"Dataset not found: {path}")
        return pd.DataFrame()

    try:
        return pd.read_csv(path)
    except Exception as error:
        warnings.warn(f"Could not read dataset {path}: {error}")
        return pd.DataFrame()


def find_label_column(df: pd.DataFrame) -> Optional[str]:
    """
    Identify the ground-truth behaviour label column.

    Supports both generic names such as 'label' and project-specific names
    such as 'behaviour_state' used in the keystroke dataset.
    """
    candidates = [
        "label",
        "behaviour",
        "behavior",
        "behaviour_label",
        "behavior_label",
        "behaviour_state",
        "behavior_state",
        "state",
        "class",
        "class_label",
        "target",
        "target_label",
        "category",
        "condition",
        "mental_state",
        "cognitive_state",
        "workload_state",
    ]

    for column in candidates:
        if column in df.columns:
            return column

    # Fallback: search for a column whose values look like behaviour labels.
    behaviour_set = set(BEHAVIOUR_CLASSES)

    for column in df.columns:
        try:
            values = (
                df[column]
                .dropna()
                .astype(str)
                .str.strip()
                .str.lower()
                .unique()
                .tolist()
            )

            value_set = set(values)

            if value_set and value_set.issubset(behaviour_set):
                return column

            if len(value_set.intersection(behaviour_set)) >= 2:
                return column

        except Exception:
            continue

    return None


def find_text_column(df: pd.DataFrame) -> Optional[str]:
    candidates = [
        "text",
        "raw_text",
        "typed_text",
        "input_text",
        "sentence",
        "content",
        "message",
        "transcript",
    ]

    for column in candidates:
        if column in df.columns:
            return column

    object_columns = [
        column
        for column in df.columns
        if df[column].dtype == object and column != find_label_column(df)
    ]

    if object_columns:
        return object_columns[0]

    return None


def find_filepath_column(df: pd.DataFrame) -> Optional[str]:
    candidates = [
        "filepath",
        "file_path",
        "path",
        "audio_path",
        "image_path",
        "filename",
    ]

    for column in candidates:
        if column in df.columns:
            return column

    return None


def resolve_dataset_path(value: Any) -> Path:
    """
    Resolve a file path from a dataset row.

    Handles:
      - absolute paths
      - project-relative paths
      - Windows-style paths stored in CSV
    """
    raw = str(value or "").strip()

    if not raw:
        return Path("")

    path = Path(raw)

    if path.exists():
        return path

    project_relative = PROJECT_ROOT / raw

    if project_relative.exists():
        return project_relative

    return path


def clean_evaluation_df(
    df: pd.DataFrame,
    label_column: str,
) -> pd.DataFrame:
    df = df.copy()
    df[label_column] = df[label_column].map(normalise_label)

    df = df[df[label_column].isin(BEHAVIOUR_CLASSES)]
    df = df.dropna(subset=[label_column])

    return df


def sample_df(
    df: pd.DataFrame,
    limit: Optional[int],
    random_state: int = 42,
) -> pd.DataFrame:
    if limit is None or limit <= 0 or len(df) <= limit:
        return df

    return df.sample(n=limit, random_state=random_state).reset_index(drop=True)


def infer_feature_columns(
    df: pd.DataFrame,
    meta: Dict[str, Any],
    excluded_columns: Iterable[str],
) -> List[str]:
    """
    Infer numeric model feature columns.

    Priority:
      1. metadata fields
      2. numeric columns excluding labels, filenames, split columns, etc.
    """
    for key in [
        "features",
        "feature_columns",
        "selected_feature_columns",
        "model_features",
    ]:
        value = meta.get(key)

        if isinstance(value, list) and value:
            return [str(column) for column in value]

    excluded = set(excluded_columns)

    return [
        column
        for column in df.columns
        if column not in excluded
        and pd.api.types.is_numeric_dtype(df[column])
    ]


def predict_with_model_artifact(
    model_path: Path,
    meta_path: Path,
    df: pd.DataFrame,
    label_column: str,
    extra_excluded: Optional[Iterable[str]] = None,
) -> Tuple[List[str], Optional[str]]:
    """
    Predict labels from a saved sklearn/joblib model.
    """
    if not model_path.exists():
        return [], f"Missing model artifact: {model_path}"

    model = joblib.load(model_path)
    meta = load_json(meta_path)

    excluded = {
        label_column,
        "split",
        "filepath",
        "file_path",
        "path",
        "filename",
        "source_modality",
        "error",
    }

    if extra_excluded:
        excluded.update(extra_excluded)

    feature_columns = infer_feature_columns(
        df=df,
        meta=meta,
        excluded_columns=excluded,
    )

    if not feature_columns:
        return [], "No usable feature columns found."

    X = df.reindex(columns=feature_columns, fill_value=0.0)

    try:
        predictions = model.predict(X)
        return [normalise_label(item) for item in predictions], None

    except Exception as error:
        return [], str(error)


def compute_metrics(
    y_true: List[str],
    y_pred: List[str],
) -> Dict[str, Any]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(
            f1_score(
                y_true,
                y_pred,
                labels=BEHAVIOUR_CLASSES,
                average="macro",
                zero_division=0,
            )
        ),
        "weighted_f1": float(
            f1_score(
                y_true,
                y_pred,
                labels=BEHAVIOUR_CLASSES,
                average="weighted",
                zero_division=0,
            )
        ),
        "support": int(len(y_true)),
    }


def save_classification_report_csv(
    y_true: List[str],
    y_pred: List[str],
    output_path: Path,
) -> None:
    report = classification_report(
        y_true,
        y_pred,
        labels=BEHAVIOUR_CLASSES,
        output_dict=True,
        zero_division=0,
    )

    pd.DataFrame(report).transpose().to_csv(output_path)


def save_confusion_matrix_plot(
    y_true: List[str],
    y_pred: List[str],
    output_path: Path,
    title: str,
) -> None:
    if not PLOTTING_AVAILABLE:
        return

    matrix = confusion_matrix(
        y_true,
        y_pred,
        labels=BEHAVIOUR_CLASSES,
    )

    plt.figure(figsize=(8, 6))
    sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=BEHAVIOUR_CLASSES,
        yticklabels=BEHAVIOUR_CLASSES,
    )
    plt.title(title)
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def evaluate_candidate_predictions(
    *,
    modality: str,
    candidate_name: str,
    y_true: List[str],
    y_pred: List[str],
    output_dir: Path,
    notes: str = "",
) -> Dict[str, Any]:
    ensure_output_dir(output_dir)

    valid_pairs = [
        (true, pred)
        for true, pred in zip(y_true, y_pred)
        if true in BEHAVIOUR_CLASSES and pred in BEHAVIOUR_CLASSES
    ]

    if not valid_pairs:
        return {
            "modality": modality,
            "candidate": candidate_name,
            "status": "failed",
            "error": "No valid prediction pairs.",
            "notes": notes,
        }

    clean_true = [item[0] for item in valid_pairs]
    clean_pred = [item[1] for item in valid_pairs]

    metrics = compute_metrics(clean_true, clean_pred)

    predictions_path = output_dir / f"predictions_{candidate_name}.csv"
    report_path = output_dir / f"classification_report_{candidate_name}.csv"
    matrix_path = output_dir / f"confusion_matrix_{candidate_name}.png"

    pd.DataFrame(
        {
            "true_label": clean_true,
            "predicted_label": clean_pred,
            "correct": [
                true == pred
                for true, pred in zip(clean_true, clean_pred)
            ],
        }
    ).to_csv(predictions_path, index=False)

    save_classification_report_csv(
        clean_true,
        clean_pred,
        report_path,
    )

    save_confusion_matrix_plot(
        clean_true,
        clean_pred,
        matrix_path,
        title=f"{modality} - {candidate_name}",
    )

    return {
        "modality": modality,
        "candidate": candidate_name,
        "status": "ok",
        **metrics,
        "predictions_csv": str(predictions_path),
        "classification_report_csv": str(report_path),
        "confusion_matrix_png": str(matrix_path) if PLOTTING_AVAILABLE else "",
        "notes": notes,
    }


def print_candidate_result(result: Dict[str, Any]) -> None:
    status = result.get("status")

    if status != "ok":
        print(f"  {result.get('candidate')}: FAILED - {result.get('error')}")
        return

    print(
        f"  {result.get('candidate')}: "
        f"Accuracy={result.get('accuracy'):.4f} | "
        f"Macro F1={result.get('macro_f1'):.4f} | "
        f"Weighted F1={result.get('weighted_f1'):.4f} | "
        f"Support={result.get('support')}"
    )


# ============================================================
# Text evaluation
# ============================================================

POSITIVE_FOCUS_WORDS = {
    "focus",
    "focused",
    "concentrate",
    "concentrating",
    "study",
    "studying",
    "productive",
    "calm",
    "steady",
    "clear",
    "work",
    "working",
}

DISTRACTION_WORDS = {
    "distracted",
    "phone",
    "scrolling",
    "checking",
    "tab",
    "tabs",
    "interrupted",
    "conversation",
    "noise",
}

FATIGUE_WORDS = {
    "tired",
    "sleepy",
    "fatigued",
    "drained",
    "exhausted",
    "slow",
    "yawning",
    "mental",
}

OVERLOAD_WORDS = {
    "overwhelmed",
    "overloaded",
    "stress",
    "stressed",
    "deadline",
    "deadlines",
    "pressure",
    "too much",
    "urgent",
    "panic",
}


def rule_based_text_baseline(text: str) -> str:
    lowered = str(text or "").lower()

    scores = {
        "focused": 0,
        "distracted": 0,
        "fatigued": 0,
        "overloaded": 0,
    }

    for word in POSITIVE_FOCUS_WORDS:
        if word in lowered:
            scores["focused"] += 1

    for word in DISTRACTION_WORDS:
        if word in lowered:
            scores["distracted"] += 1

    for word in FATIGUE_WORDS:
        if word in lowered:
            scores["fatigued"] += 1

    for word in OVERLOAD_WORDS:
        if word in lowered:
            scores["overloaded"] += 1

    if max(scores.values()) <= 0:
        return "focused"

    return max(scores, key=scores.get)


def evaluate_text(args: argparse.Namespace) -> List[Dict[str, Any]]:
    print("\nEvaluating text modality...")

    output_dir = OUTPUT_DIR / "text"
    df = safe_read_csv(TEXT_DATASET)

    if df.empty:
        return [{
            "modality": "text",
            "candidate": "text_dataset",
            "status": "failed",
            "error": f"Dataset unavailable: {TEXT_DATASET}",
        }]

    label_column = find_label_column(df)
    text_column = find_text_column(df)

    if not label_column or not text_column:
        return [{
            "modality": "text",
            "candidate": "text_dataset",
            "status": "failed",
            "error": "Could not identify label/text columns.",
        }]

    df = clean_evaluation_df(df, label_column)
    df = sample_df(df, args.text_limit)

    y_true = df[label_column].tolist()
    results: List[Dict[str, Any]] = []

    # Candidate 1: trained text model.
    if TEXT_MODEL_PATH.exists():
        try:
            model = joblib.load(TEXT_MODEL_PATH)
            y_pred = [
                normalise_label(item)
                for item in model.predict(df[text_column].astype(str).tolist())
            ]

            result = evaluate_candidate_predictions(
                modality="text",
                candidate_name="trained_text_model",
                y_true=y_true,
                y_pred=y_pred,
                output_dir=output_dir,
                notes=(
                    "Selected text classifier loaded from text_model.joblib. "
                    "This evaluates the trained text behaviour model on the "
                    "processed text dataset."
                ),
            )
            results.append(result)
            print_candidate_result(result)

        except Exception as error:
            results.append({
                "modality": "text",
                "candidate": "trained_text_model",
                "status": "failed",
                "error": str(error),
            })

    # Candidate 2: lightweight lexical baseline.
    y_pred_rule = [
        rule_based_text_baseline(text)
        for text in df[text_column].astype(str).tolist()
    ]

    result = evaluate_candidate_predictions(
        modality="text",
        candidate_name="rule_based_keyword_baseline",
        y_true=y_true,
        y_pred=y_pred_rule,
        output_dir=output_dir,
        notes=(
            "Transparent lexical baseline. Used as a simple comparison point "
            "to justify why the trained text model is preferable."
        ),
    )
    results.append(result)
    print_candidate_result(result)

    return results


# ============================================================
# Keystroke evaluation
# ============================================================

def rule_based_keystroke_baseline(row: pd.Series) -> str:
    typing_speed = float(row.get("typing_speed", 0.0) or 0.0)
    delay_mean = float(row.get("delay_mean", 0.0) or 0.0)
    delay_std = float(row.get("delay_std", 0.0) or 0.0)
    correction_ratio = float(row.get("correction_ratio", 0.0) or 0.0)
    error_rate_proxy = float(row.get("error_rate_proxy", 0.0) or 0.0)
    rhythm_consistency = float(row.get("rhythm_consistency", 0.0) or 0.0)
    pause_1000 = float(row.get("pause_ratio_1000", 0.0) or 0.0)

    if correction_ratio >= 0.16 or error_rate_proxy >= 0.20:
        return "overloaded"

    if delay_mean >= 450 and typing_speed < 0.0022:
        return "fatigued"

    if pause_1000 >= 0.10 or delay_std >= 450:
        return "distracted"

    if rhythm_consistency >= 0.60:
        return "focused"

    return "focused"


def evaluate_keystroke(args: argparse.Namespace) -> List[Dict[str, Any]]:
    print("\nEvaluating keystroke modality...")

    output_dir = OUTPUT_DIR / "keystroke"
    df = safe_read_csv(KEYSTROKE_DATASET)

    if df.empty:
        return [{
            "modality": "keystroke",
            "candidate": "keystroke_dataset",
            "status": "failed",
            "error": f"Dataset unavailable: {KEYSTROKE_DATASET}",
        }]

    label_column = find_label_column(df)

    if not label_column:
        return [{
            "modality": "keystroke",
            "candidate": "keystroke_dataset",
            "status": "failed",
            "error": "Could not identify label column.",
        }]

    df = clean_evaluation_df(df, label_column)
    df = sample_df(df, args.keystroke_limit)

    y_true = df[label_column].tolist()
    results: List[Dict[str, Any]] = []

    # Candidate 1: trained keystroke model.
    y_pred, error = predict_with_model_artifact(
        model_path=KEYSTROKE_MODEL_PATH,
        meta_path=KEYSTROKE_META_PATH,
        df=df,
        label_column=label_column,
    )

    if error:
        results.append({
            "modality": "keystroke",
            "candidate": "trained_keystroke_model",
            "status": "failed",
            "error": error,
        })
    else:
        result = evaluate_candidate_predictions(
            modality="keystroke",
            candidate_name="trained_keystroke_model",
            y_true=y_true,
            y_pred=y_pred,
            output_dir=output_dir,
            notes=(
                "Selected keystroke classifier evaluated on processed "
                "keystroke feature data."
            ),
        )
        results.append(result)
        print_candidate_result(result)

    # Candidate 2: rule-based baseline.
    y_pred_rule = [
        rule_based_keystroke_baseline(row)
        for _, row in df.iterrows()
    ]

    result = evaluate_candidate_predictions(
        modality="keystroke",
        candidate_name="rule_based_keystroke_baseline",
        y_true=y_true,
        y_pred=y_pred_rule,
        output_dir=output_dir,
        notes=(
            "Transparent rule baseline using typing speed, pauses, rhythm, "
            "and correction/error indicators."
        ),
    )
    results.append(result)
    print_candidate_result(result)

    return results


# ============================================================
# Audio evaluation
# ============================================================

def evaluate_audio(args: argparse.Namespace) -> List[Dict[str, Any]]:
    print("\nEvaluating audio modality...")

    output_dir = OUTPUT_DIR / "audio"
    df = safe_read_csv(AUDIO_DATASET)

    if df.empty:
        return [{
            "modality": "audio",
            "candidate": "audio_dataset",
            "status": "failed",
            "error": f"Dataset unavailable: {AUDIO_DATASET}",
        }]

    label_column = find_label_column(df)

    if not label_column:
        return [{
            "modality": "audio",
            "candidate": "audio_dataset",
            "status": "failed",
            "error": "Could not identify label column.",
        }]

    df = clean_evaluation_df(df, label_column)
    df = sample_df(df, args.audio_limit)

    y_true = df[label_column].tolist()
    results: List[Dict[str, Any]] = []

    # Candidate 1: trained audio model using precomputed feature columns.
    y_pred, error = predict_with_model_artifact(
        model_path=AUDIO_MODEL_PATH,
        meta_path=AUDIO_META_PATH,
        df=df,
        label_column=label_column,
    )

    if error:
        results.append({
            "modality": "audio",
            "candidate": "trained_audio_feature_model",
            "status": "failed",
            "error": error,
        })
    else:
        result = evaluate_candidate_predictions(
            modality="audio",
            candidate_name="trained_audio_feature_model",
            y_true=y_true,
            y_pred=y_pred,
            output_dir=output_dir,
            notes=(
                "Trained librosa-feature audio classifier evaluated using "
                "processed audio feature columns."
            ),
        )
        results.append(result)
        print_candidate_result(result)

    # Candidate 2: full runtime audio pipeline.
    if args.runtime_audio:
        try:
            from app.models.audio_models import analyze_audio_file

            path_column = find_filepath_column(df)

            if not path_column:
                results.append({
                    "modality": "audio",
                    "candidate": "runtime_audio_pipeline",
                    "status": "failed",
                    "error": "No filepath column found for runtime audio evaluation.",
                })
            else:
                runtime_true: List[str] = []
                runtime_pred: List[str] = []

                for _, row in df.iterrows():
                    audio_path = resolve_dataset_path(row[path_column])

                    if not audio_path.exists():
                        continue

                    try:
                        result_payload = analyze_audio_file(str(audio_path))
                        prediction = normalise_label(
                            result_payload.get("predicted_label")
                            or result_payload.get("predicted_behaviour")
                        )

                        runtime_true.append(row[label_column])
                        runtime_pred.append(prediction)

                    except Exception as error:
                        warnings.warn(
                            f"Runtime audio evaluation failed for {audio_path}: {error}"
                        )

                result = evaluate_candidate_predictions(
                    modality="audio",
                    candidate_name="runtime_audio_yamnet_pipeline",
                    y_true=runtime_true,
                    y_pred=runtime_pred,
                    output_dir=output_dir,
                    notes=(
                        "Full runtime audio pipeline using trained audio model, "
                        "local/TFHub YAMNet, and heuristic acoustic-context mapping."
                    ),
                )
                results.append(result)
                print_candidate_result(result)

        except Exception as error:
            results.append({
                "modality": "audio",
                "candidate": "runtime_audio_pipeline",
                "status": "failed",
                "error": str(error),
            })

    return results


# ============================================================
# Image / vision evaluation
# ============================================================

def evaluate_image(args: argparse.Namespace) -> List[Dict[str, Any]]:
    print("\nEvaluating image / vision modality...")

    output_dir = OUTPUT_DIR / "image"
    df = safe_read_csv(IMAGE_DATASET)

    if df.empty:
        return [{
            "modality": "image",
            "candidate": "image_dataset",
            "status": "failed",
            "error": f"Dataset unavailable: {IMAGE_DATASET}",
        }]

    label_column = find_label_column(df)

    if not label_column:
        return [{
            "modality": "image",
            "candidate": "image_dataset",
            "status": "failed",
            "error": "Could not identify label column.",
        }]

    df = clean_evaluation_df(df, label_column)
    df = sample_df(df, args.image_limit)

    y_true = df[label_column].tolist()
    results: List[Dict[str, Any]] = []

    # Candidate 1: trained image model.
    y_pred, error = predict_with_model_artifact(
        model_path=IMAGE_MODEL_PATH,
        meta_path=IMAGE_META_PATH,
        df=df,
        label_column=label_column,
    )

    if error:
        results.append({
            "modality": "image",
            "candidate": "trained_image_model",
            "status": "failed",
            "error": error,
        })
    else:
        result = evaluate_candidate_predictions(
            modality="image",
            candidate_name="trained_image_model",
            y_true=y_true,
            y_pred=y_pred,
            output_dir=output_dir,
            notes=(
                "Trained image classifier evaluated using safe visual features "
                "from processed_image_dataset.csv."
            ),
        )
        results.append(result)
        print_candidate_result(result)

    # Candidate 2: precomputed CLIP/BLIP/MediaPipe visual fused label.
    if "visual_fused_label" in df.columns:
        y_pred_visual = [
            normalise_label(item)
            for item in df["visual_fused_label"].tolist()
        ]

        result = evaluate_candidate_predictions(
            modality="image",
            candidate_name="precomputed_clip_blip_mediapipe_fusion",
            y_true=y_true,
            y_pred=y_pred_visual,
            output_dir=output_dir,
            notes=(
                "Precomputed vision pipeline output from process_image_data.py. "
                "Represents CLIP prompt scoring, BLIP captioning, and MediaPipe "
                "face/body/posture analysis."
            ),
        )
        results.append(result)
        print_candidate_result(result)

    # Candidate 3: full-image CLIP behaviour cue only.
    if "predicted_behaviour_cue" in df.columns:
        y_pred_behaviour = [
            normalise_label(item)
            for item in df["predicted_behaviour_cue"].tolist()
        ]

        result = evaluate_candidate_predictions(
            modality="image",
            candidate_name="clip_full_image_behaviour_cue",
            y_true=y_true,
            y_pred=y_pred_behaviour,
            output_dir=output_dir,
            notes=(
                "Single visual cue candidate using the full-image CLIP "
                "behaviour prompt classification."
            ),
        )
        results.append(result)
        print_candidate_result(result)

    # Candidate 4: runtime image pipeline.
    if args.runtime_image:
        try:
            from app.models.image_model import analyze_image_file

            path_column = find_filepath_column(df)

            if not path_column:
                results.append({
                    "modality": "image",
                    "candidate": "runtime_image_pipeline",
                    "status": "failed",
                    "error": "No filepath column found for runtime image evaluation.",
                })
            else:
                runtime_true: List[str] = []
                runtime_pred: List[str] = []

                for _, row in df.iterrows():
                    image_path = resolve_dataset_path(row[path_column])

                    if not image_path.exists():
                        continue

                    try:
                        result_payload = analyze_image_file(str(image_path))
                        prediction = normalise_label(
                            result_payload.get("predicted_label")
                            or result_payload.get("predicted_behaviour")
                        )

                        runtime_true.append(row[label_column])
                        runtime_pred.append(prediction)

                    except Exception as error:
                        warnings.warn(
                            f"Runtime image evaluation failed for {image_path}: {error}"
                        )

                result = evaluate_candidate_predictions(
                    modality="image",
                    candidate_name="runtime_clip_blip_mediapipe_pipeline",
                    y_true=runtime_true,
                    y_pred=runtime_pred,
                    output_dir=output_dir,
                    notes=(
                        "Full runtime image pipeline using CLIP, BLIP captioning, "
                        "MediaPipe face/body detection, and behaviour-aware captions."
                    ),
                )
                results.append(result)
                print_candidate_result(result)

        except Exception as error:
            results.append({
                "modality": "image",
                "candidate": "runtime_image_pipeline",
                "status": "failed",
                "error": str(error),
            })

    return results


# ============================================================
# Fusion evaluation
# ============================================================

def evaluate_fusion(args: argparse.Namespace) -> List[Dict[str, Any]]:
    print("\nEvaluating experimental fusion model...")

    output_dir = OUTPUT_DIR / "fusion"
    df = safe_read_csv(FUSION_DATASET)

    if df.empty:
        return [{
            "modality": "fusion",
            "candidate": "fusion_dataset",
            "status": "failed",
            "error": (
                f"Fusion dataset unavailable: {FUSION_DATASET}. "
                "Run train_fusion_model.py first."
            ),
        }]

    label_column = find_label_column(df)

    if not label_column:
        return [{
            "modality": "fusion",
            "candidate": "fusion_dataset",
            "status": "failed",
            "error": "Could not identify label column.",
        }]

    df = clean_evaluation_df(df, label_column)
    df = sample_df(df, args.fusion_limit)

    y_true = df[label_column].tolist()
    results: List[Dict[str, Any]] = []

    y_pred, error = predict_with_model_artifact(
        model_path=FUSION_MODEL_PATH,
        meta_path=FUSION_META_PATH,
        df=df,
        label_column=label_column,
    )

    if error:
        results.append({
            "modality": "fusion",
            "candidate": "experimental_trained_fusion_model",
            "status": "failed",
            "error": error,
        })
    else:
        result = evaluate_candidate_predictions(
            modality="fusion",
            candidate_name="experimental_trained_fusion_model",
            y_true=y_true,
            y_pred=y_pred,
            output_dir=output_dir,
            notes=(
                "Experimental trained fusion model evaluated on the constructed "
                "fusion dataset. This is useful for model-selection evidence, "
                "but final runtime deployment should use calibrated dynamic "
                "weighted late fusion unless true session-aligned multimodal "
                "data is available."
            ),
        )
        results.append(result)
        print_candidate_result(result)

    return results


# ============================================================
# Summary output
# ============================================================

def save_overall_summary(results: List[Dict[str, Any]]) -> None:
    ensure_output_dir(OUTPUT_DIR)

    summary_csv = OUTPUT_DIR / "overall_candidate_summary.csv"
    summary_json = OUTPUT_DIR / "overall_candidate_summary.json"
    report_txt = OUTPUT_DIR / "evaluation_report.txt"

    df = pd.DataFrame(results)
    df.to_csv(summary_csv, index=False)

    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4, default=str)

    with open(report_txt, "w", encoding="utf-8") as f:
        f.write("SenseFuzeAI Full Model Candidate Evaluation Report\n")
        f.write("=================================================\n\n")
        f.write(f"Generated at: {datetime.now().isoformat(timespec='seconds')}\n")
        f.write(f"Project root: {PROJECT_ROOT}\n")
        f.write(f"Output directory: {OUTPUT_DIR}\n\n")

        f.write("Important methodological note\n")
        f.write("-----------------------------\n")
        f.write(
            "This evaluation compares available modality-level models, "
            "baselines, and precomputed pipeline outputs. Runtime web-app "
            "fusion should remain calibrated dynamic weighted late fusion "
            "unless true session-aligned multimodal data is collected for "
            "training a defensible supervised fusion model.\n\n"
        )

        f.write("Candidate results\n")
        f.write("-----------------\n")

        for result in results:
            f.write(f"\nModality:  {result.get('modality')}\n")
            f.write(f"Candidate: {result.get('candidate')}\n")
            f.write(f"Status:    {result.get('status')}\n")

            if result.get("status") == "ok":
                f.write(f"Accuracy:  {result.get('accuracy'):.4f}\n")
                f.write(f"Macro F1:  {result.get('macro_f1'):.4f}\n")
                f.write(f"Weighted F1: {result.get('weighted_f1'):.4f}\n")
                f.write(f"Support:   {result.get('support')}\n")
            else:
                f.write(f"Error:     {result.get('error')}\n")

            if result.get("notes"):
                f.write(f"Notes:     {result.get('notes')}\n")

    print("\nSaved overall evaluation outputs:")
    print(f"  {summary_csv}")
    print(f"  {summary_json}")
    print(f"  {report_txt}")


# ============================================================
# CLI
# ============================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate SenseFuzeAI modality model candidates."
    )

    parser.add_argument(
        "--skip-keystroke",
        action="store_true",
        help="Skip keystroke evaluation.",
    )

    parser.add_argument(
        "--skip-text",
        action="store_true",
        help="Skip text evaluation.",
    )

    parser.add_argument(
        "--skip-audio",
        action="store_true",
        help="Skip audio evaluation.",
    )

    parser.add_argument(
        "--skip-image",
        action="store_true",
        help="Skip image evaluation.",
    )

    parser.add_argument(
        "--skip-fusion",
        action="store_true",
        help="Skip fusion evaluation.",
    )

    parser.add_argument(
        "--runtime-audio",
        action="store_true",
        help="Run full runtime audio pipeline using analyze_audio_file(). Slower.",
    )

    parser.add_argument(
        "--runtime-image",
        action="store_true",
        help="Run full runtime image pipeline using analyze_image_file(). Slower.",
    )

    parser.add_argument(
        "--keystroke-limit",
        type=int,
        default=0,
        help="Limit keystroke evaluation rows. 0 means all.",
    )

    parser.add_argument(
        "--text-limit",
        type=int,
        default=0,
        help="Limit text evaluation rows. 0 means all.",
    )

    parser.add_argument(
        "--audio-limit",
        type=int,
        default=80,
        help="Limit audio evaluation rows. Recommended small if runtime-audio is enabled.",
    )

    parser.add_argument(
        "--image-limit",
        type=int,
        default=80,
        help="Limit image evaluation rows. Recommended small if runtime-image is enabled.",
    )

    parser.add_argument(
        "--fusion-limit",
        type=int,
        default=0,
        help="Limit fusion evaluation rows. 0 means all.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    print_header()
    ensure_output_dir(OUTPUT_DIR)

    all_results: List[Dict[str, Any]] = []

    if not args.skip_keystroke:
        all_results.extend(evaluate_keystroke(args))

    if not args.skip_text:
        all_results.extend(evaluate_text(args))

    if not args.skip_audio:
        all_results.extend(evaluate_audio(args))

    if not args.skip_image:
        all_results.extend(evaluate_image(args))

    if not args.skip_fusion:
        all_results.extend(evaluate_fusion(args))

    save_overall_summary(all_results)

    print("\nEvaluation complete.")
    print(f"Reports saved to:\n{OUTPUT_DIR}")


if __name__ == "__main__":
    main()
