# tests/test_02_integration.py

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import ModuleType

import joblib
import pandas as pd


# ============================================================
# Project paths
# ============================================================

ROOT_DIR = Path(__file__).resolve().parents[1]

INFERENCE_PATH = ROOT_DIR / "final_multimodal_inference.py"

IMAGE_MODEL_DIR = ROOT_DIR / "models" / "image_demo"

ORIGINAL_IMAGE_MODEL_PATH = (
    IMAGE_MODEL_DIR / "image_pipeline.joblib"
)

CALIBRATED_IMAGE_MODEL_PATH = (
    IMAGE_MODEL_DIR / "image_pipeline_webcam_calibrated.joblib"
)

CALIBRATED_METADATA_PATH = (
    IMAGE_MODEL_DIR / "webcam_calibrated_metadata.json"
)

IMAGE_FEATURE_SCHEMA_PATH = (
    IMAGE_MODEL_DIR / "feature_columns.json"
)

WEBCAM_EVALUATION_DIR = (
    ROOT_DIR
    / "data"
    / "processed"
    / "webcam_calibration_evaluation"
)


# ============================================================
# Calibration artifact candidates
# ============================================================

# Support both possible output layouts:
#
#   data/webcam_calibration_clip_features.csv
#
# or
#
#   data/processed/webcam_calibration_clip_features.csv
#
# This makes the tests consistent with either calibration-script layout.

WEBCAM_FEATURE_DATA_CANDIDATES = [
    ROOT_DIR
    / "data"
    / "webcam_calibration_clip_features.csv",

    ROOT_DIR
    / "data"
    / "processed"
    / "webcam_calibration_clip_features.csv",
]

WEBCAM_FEATURE_SUMMARY_CANDIDATES = [
    ROOT_DIR
    / "data"
    / "webcam_calibration_clip_features_summary.json",

    ROOT_DIR
    / "data"
    / "processed"
    / "webcam_calibration_clip_features_summary.json",
]


EXPECTED_CLASSES = {
    "focused",
    "distracted",
    "fatigued",
    "overloaded",
}


# ============================================================
# Utility helpers
# ============================================================

def resolve_existing_path(
    candidates: list[Path],
    artifact_name: str,
) -> Path:
    """
    Return the first existing path from a list of valid artifact locations.

    Raises an informative assertion error if the artifact cannot be found.
    """

    for path in candidates:
        if path.exists():
            return path

    candidate_text = "\n".join(
        f"  - {path}"
        for path in candidates
    )

    raise AssertionError(
        f"Could not find {artifact_name}.\n"
        f"Checked the following locations:\n"
        f"{candidate_text}"
    )


def load_inference_module() -> ModuleType:
    """
    Load final_multimodal_inference.py directly from its file path.
    """

    assert INFERENCE_PATH.exists(), (
        f"Missing file: {INFERENCE_PATH}"
    )

    spec = importlib.util.spec_from_file_location(
        "sensefuze_final_multimodal_inference",
        INFERENCE_PATH,
    )

    assert spec is not None
    assert spec.loader is not None

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    return module


def extract_model_classes(model) -> set[str]:
    """
    Extract behavioural classes from a scikit-learn model or pipeline.
    """

    classes = list(
        getattr(model, "classes_", [])
    )

    if not classes and hasattr(model, "named_steps"):
        final_estimator = list(
            model.named_steps.values()
        )[-1]

        classes = list(
            getattr(final_estimator, "classes_", [])
        )

    return {
        str(label).strip().lower()
        for label in classes
    }


# ============================================================
# Model artifact integration tests
# ============================================================

def test_required_model_artifacts_exist():
    print(
        "\n[INTEGRATION] "
        "Checking required model artifacts..."
    )

    required_paths = [
        ROOT_DIR
        / "models"
        / "keystroke_demo"
        / "keystroke_pipeline.joblib",

        ROOT_DIR
        / "models"
        / "text_demo"
        / "text_pipeline.joblib",

        ROOT_DIR
        / "models"
        / "audio_demo"
        / "audio_pipeline.joblib",

        ORIGINAL_IMAGE_MODEL_PATH,

        CALIBRATED_IMAGE_MODEL_PATH,

        ROOT_DIR
        / "models"
        / "fusion_demo"
        / "fusion_pipeline.joblib",

        ROOT_DIR
        / "models"
        / "fusion_demo"
        / "feature_columns.json",

        ROOT_DIR
        / "models"
        / "all-mpnet-base-v2",

        ROOT_DIR
        / "models"
        / "wavlm-base-plus",

        ROOT_DIR
        / "models"
        / "clip-vit-large-patch14",
    ]

    missing = [
        str(path)
        for path in required_paths
        if not path.exists()
    ]

    assert not missing, (
        f"Missing model artifacts: {missing}"
    )

    print(
        f"       PASS: {len(required_paths)} required "
        "model artifacts/directories found."
    )


def test_original_and_calibrated_image_models_are_both_preserved():
    print(
        "\n[INTEGRATION] Checking original and "
        "webcam-calibrated image models..."
    )

    assert ORIGINAL_IMAGE_MODEL_PATH.exists(), (
        f"Missing original image model: "
        f"{ORIGINAL_IMAGE_MODEL_PATH}"
    )

    assert CALIBRATED_IMAGE_MODEL_PATH.exists(), (
        f"Missing calibrated image model: "
        f"{CALIBRATED_IMAGE_MODEL_PATH}"
    )

    assert (
        ORIGINAL_IMAGE_MODEL_PATH
        != CALIBRATED_IMAGE_MODEL_PATH
    )

    print(
        "       PASS: Original image model remains preserved."
    )

    print(
        "       PASS: Separate webcam-calibrated "
        "image model exists."
    )


def test_webcam_calibrated_image_model_loadable():
    print(
        "\n[INTEGRATION] Loading webcam-calibrated "
        "image classifier..."
    )

    assert CALIBRATED_IMAGE_MODEL_PATH.exists(), (
        f"Missing calibrated model: "
        f"{CALIBRATED_IMAGE_MODEL_PATH}"
    )

    model = joblib.load(
        CALIBRATED_IMAGE_MODEL_PATH
    )

    assert model is not None

    assert hasattr(model, "predict"), (
        "Calibrated model does not implement predict()."
    )

    assert hasattr(model, "predict_proba"), (
        "Calibrated model does not implement predict_proba()."
    )

    classes = extract_model_classes(model)

    if classes:
        assert EXPECTED_CLASSES.issubset(classes), (
            "Calibrated model classes are incomplete.\n"
            f"Expected: {sorted(EXPECTED_CLASSES)}\n"
            f"Found: {sorted(classes)}"
        )

    print(
        "       PASS: Calibrated classifier loads successfully."
    )

    print(
        "       Model classes:",
        sorted(classes)
        if classes
        else "pipeline-managed",
    )


def test_webcam_calibration_metadata_exists_and_valid():
    print(
        "\n[INTEGRATION] Checking "
        "webcam-calibration metadata..."
    )

    assert CALIBRATED_METADATA_PATH.exists(), (
        f"Missing metadata: "
        f"{CALIBRATED_METADATA_PATH}"
    )

    with CALIBRATED_METADATA_PATH.open(
        "r",
        encoding="utf-8",
    ) as f:
        metadata = json.load(f)

    assert isinstance(metadata, dict), (
        "Calibration metadata must be a JSON object."
    )

    assert len(metadata) > 0, (
        "Calibration metadata is empty."
    )

    print(
        "       PASS: Webcam-calibrated metadata exists "
        f"with {len(metadata)} metadata fields."
    )


# ============================================================
# Webcam calibration dataset tests
# ============================================================

def test_webcam_calibration_dataset_exists():
    print(
        "\n[INTEGRATION] Checking "
        "webcam-calibration feature dataset..."
    )

    feature_path = resolve_existing_path(
        WEBCAM_FEATURE_DATA_CANDIDATES,
        "webcam calibration CLIP feature dataset",
    )

    print(
        f"       Resolved dataset: {feature_path}"
    )

    df = pd.read_csv(feature_path)

    assert not df.empty, (
        "Webcam calibration feature dataset is empty."
    )

    assert "label" in df.columns, (
        "Calibration dataset does not contain "
        "the required 'label' column."
    )

    labels = set(
        df["label"]
        .dropna()
        .astype(str)
        .str.strip()
        .str.lower()
        .unique()
    )

    assert EXPECTED_CLASSES.issubset(labels), (
        "Calibration dataset does not contain "
        "all four behavioural classes.\n"
        f"Expected: {sorted(EXPECTED_CLASSES)}\n"
        f"Found: {sorted(labels)}"
    )

    clip_columns = [
        col
        for col in df.columns
        if col.startswith("image_clip_emb_")
    ]

    assert len(clip_columns) > 0, (
        "Calibration dataset contains no "
        "image_clip_emb_* features."
    )

    # CLIP ViT-L/14 should normally produce 768 dimensions.
    assert len(clip_columns) == 768, (
        "Unexpected CLIP embedding dimension.\n"
        f"Expected: 768\n"
        f"Found: {len(clip_columns)}"
    )

    print(
        f"       PASS: Calibration dataset rows = {len(df)}"
    )

    print(
        f"       PASS: CLIP embedding features = "
        f"{len(clip_columns)}"
    )

    print(
        f"       PASS: Behavioural classes = "
        f"{sorted(labels)}"
    )


def test_webcam_calibration_summary_exists():
    print(
        "\n[INTEGRATION] Checking "
        "webcam-calibration summary..."
    )

    summary_path = resolve_existing_path(
        WEBCAM_FEATURE_SUMMARY_CANDIDATES,
        "webcam calibration summary JSON",
    )

    print(
        f"       Resolved summary: {summary_path}"
    )

    with summary_path.open(
        "r",
        encoding="utf-8",
    ) as f:
        summary = json.load(f)

    assert isinstance(summary, dict), (
        "Calibration summary must be a JSON object."
    )

    assert len(summary) > 0, (
        "Calibration summary JSON is empty."
    )

    print(
        "       PASS: Calibration summary JSON is valid."
    )


def test_webcam_evaluation_results_exist():
    print(
        "\n[INTEGRATION] Checking webcam-calibration "
        "evaluation directory..."
    )

    assert WEBCAM_EVALUATION_DIR.exists(), (
        f"Missing evaluation directory: "
        f"{WEBCAM_EVALUATION_DIR}"
    )

    assert WEBCAM_EVALUATION_DIR.is_dir(), (
        f"Expected directory but found something else: "
        f"{WEBCAM_EVALUATION_DIR}"
    )

    files = [
        path
        for path in WEBCAM_EVALUATION_DIR.rglob("*")
        if path.is_file()
    ]

    assert files, (
        "No evaluation artifacts found in "
        f"{WEBCAM_EVALUATION_DIR}"
    )

    print(
        f"       PASS: {len(files)} webcam "
        "evaluation artifact(s) found."
    )


# ============================================================
# Fusion schema integration tests
# ============================================================

def test_fusion_feature_schema_contains_all_modalities():
    print(
        "\n[INTEGRATION] Checking fusion feature schema..."
    )

    schema_path = (
        ROOT_DIR
        / "models"
        / "fusion_demo"
        / "feature_columns.json"
    )

    assert schema_path.exists(), (
        f"Missing schema: {schema_path}"
    )

    with schema_path.open(
        "r",
        encoding="utf-8",
    ) as f:
        columns = json.load(f)

    assert isinstance(columns, list)
    assert len(columns) > 0

    has_text = any(
        col.startswith("text_mpnet_emb_")
        for col in columns
    )

    has_audio = any(
        col.startswith("audio_wavlm_emb_")
        or col.startswith("audio_")
        for col in columns
    )

    has_image = any(
        col.startswith("image_clip_emb_")
        or col.startswith("image_")
        for col in columns
    )

    has_keystroke = any(
        "keydown" in col
        or "typing" in col
        or "delay_" in col
        or "hold_" in col
        for col in columns
    )

    assert has_text, (
        "Fusion schema contains no text features."
    )

    assert has_audio, (
        "Fusion schema contains no audio features."
    )

    assert has_image, (
        "Fusion schema contains no image features."
    )

    assert has_keystroke, (
        "Fusion schema contains no keystroke features."
    )

    print(
        f"       PASS: Fusion feature count = "
        f"{len(columns)}"
    )

    print(
        "       PASS: Keystroke modality present."
    )

    print(
        "       PASS: Text modality present."
    )

    print(
        "       PASS: Audio modality present."
    )

    print(
        "       PASS: Image modality present."
    )


def test_image_feature_schema_uses_clip_embeddings():
    print(
        "\n[INTEGRATION] Checking image feature schema..."
    )

    assert IMAGE_FEATURE_SCHEMA_PATH.exists(), (
        f"Missing image feature schema: "
        f"{IMAGE_FEATURE_SCHEMA_PATH}"
    )

    with IMAGE_FEATURE_SCHEMA_PATH.open(
        "r",
        encoding="utf-8",
    ) as f:
        columns = json.load(f)

    assert isinstance(columns, list)
    assert len(columns) > 0

    clip_columns = [
        col
        for col in columns
        if col.startswith("image_clip_emb_")
    ]

    assert len(clip_columns) > 0

    assert len(clip_columns) == 768, (
        "Image feature schema does not match "
        "CLIP ViT-L/14 output dimensionality.\n"
        f"Expected: 768\n"
        f"Found: {len(clip_columns)}"
    )

    print(
        f"       PASS: Image model expects "
        f"{len(clip_columns)} CLIP embedding dimensions."
    )


# ============================================================
# Final inference integration tests
# ============================================================

def test_final_inference_class_importable():
    print(
        "\n[INTEGRATION] Importing final multimodal "
        "inference module..."
    )

    module = load_inference_module()

    assert hasattr(
        module,
        "FinalMultimodalInference",
    )

    assert (
        module.FinalMultimodalInference
        is not None
    )

    print(
        "       PASS: FinalMultimodalInference "
        "class is importable."
    )


def test_final_inference_references_clip_model():
    print(
        "\n[INTEGRATION] Checking final inference "
        "CLIP integration..."
    )

    content = INFERENCE_PATH.read_text(
        encoding="utf-8"
    )

    assert "clip-vit-large-patch14" in content
    assert "CLIPModel" in content
    assert "CLIPProcessor" in content
    assert "image_clip_emb_" in content

    print(
        "       PASS: Final inference pipeline uses "
        "the expected pretrained CLIP visual encoder."
    )
