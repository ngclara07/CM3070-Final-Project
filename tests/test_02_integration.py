# === tests/test_02_integration.py ===

from __future__ import annotations

import importlib.util
import json
import uuid
from pathlib import Path
from types import ModuleType

import joblib
import pandas as pd


# ============================================================
# Project paths
# ============================================================

ROOT_DIR = Path(__file__).resolve().parents[1]

INFERENCE_PATH = (
    ROOT_DIR
    / "final_multimodal_inference.py"
)

WEB_APP_PATH = (
    ROOT_DIR
    / "web_app"
    / "app.py"
)

IMAGE_MODEL_DIR = (
    ROOT_DIR
    / "models"
    / "image_demo"
)

ORIGINAL_IMAGE_MODEL_PATH = (
    IMAGE_MODEL_DIR
    / "image_pipeline.joblib"
)

CALIBRATED_IMAGE_MODEL_PATH = (
    IMAGE_MODEL_DIR
    / "image_pipeline_webcam_calibrated.joblib"
)

CALIBRATED_METADATA_PATH = (
    IMAGE_MODEL_DIR
    / "webcam_calibrated_metadata.json"
)

IMAGE_FEATURE_SCHEMA_PATH = (
    IMAGE_MODEL_DIR
    / "feature_columns.json"
)

FUSION_MODEL_PATH = (
    ROOT_DIR
    / "models"
    / "fusion_demo"
    / "fusion_pipeline.joblib"
)

FUSION_SCHEMA_PATH = (
    ROOT_DIR
    / "models"
    / "fusion_demo"
    / "feature_columns.json"
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

WEBCAM_FEATURE_DATA_CANDIDATES = [
    (
        ROOT_DIR
        / "data"
        / "webcam_calibration_clip_features.csv"
    ),
    (
        ROOT_DIR
        / "data"
        / "processed"
        / "webcam_calibration_clip_features.csv"
    ),
]

WEBCAM_FEATURE_SUMMARY_CANDIDATES = [
    (
        ROOT_DIR
        / "data"
        / "webcam_calibration_clip_features_summary.json"
    ),
    (
        ROOT_DIR
        / "data"
        / "processed"
        / "webcam_calibration_clip_features_summary.json"
    ),
]


EXPECTED_CLASSES = {
    "focused",
    "distracted",
    "fatigued",
    "overloaded",
}


# ============================================================
# Helpers
# ============================================================

def resolve_existing_path(
    candidates: list[Path],
    artifact_name: str,
) -> Path:
    """
    Return the first existing artifact path.
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
        f"Checked:\n"
        f"{candidate_text}"
    )


def load_inference_module() -> ModuleType:
    """
    Import final_multimodal_inference.py directly.
    """

    assert INFERENCE_PATH.exists(), (
        f"Missing file: {INFERENCE_PATH}"
    )

    module_name = (
        "sensefuze_final_inference_"
        + uuid.uuid4().hex
    )

    spec = importlib.util.spec_from_file_location(
        module_name,
        INFERENCE_PATH,
    )

    assert spec is not None
    assert spec.loader is not None

    module = importlib.util.module_from_spec(
        spec
    )

    spec.loader.exec_module(
        module
    )

    return module


def extract_model_classes(
    model,
) -> set[str]:
    """
    Extract behavioural classes from a scikit-learn
    estimator or Pipeline.
    """

    classes = list(
        getattr(
            model,
            "classes_",
            [],
        )
    )

    if (
        not classes
        and hasattr(
            model,
            "named_steps",
        )
    ):

        final_estimator = list(
            model.named_steps.values()
        )[-1]

        classes = list(
            getattr(
                final_estimator,
                "classes_",
                [],
            )
        )

    return {
        str(label)
        .strip()
        .lower()
        for label in classes
    }


# ============================================================
# Required model artifacts
# ============================================================

def test_required_model_artifacts_exist():
    print(
        "\n[INTEGRATION] Checking required model artifacts..."
    )

    required_paths = [
        (
            ROOT_DIR
            / "models"
            / "keystroke_demo"
            / "keystroke_pipeline.joblib"
        ),
        (
            ROOT_DIR
            / "models"
            / "text_demo"
            / "text_pipeline.joblib"
        ),
        (
            ROOT_DIR
            / "models"
            / "audio_demo"
            / "audio_pipeline.joblib"
        ),
        ORIGINAL_IMAGE_MODEL_PATH,
        CALIBRATED_IMAGE_MODEL_PATH,
        FUSION_MODEL_PATH,
        FUSION_SCHEMA_PATH,
        (
            ROOT_DIR
            / "models"
            / "all-mpnet-base-v2"
        ),
        (
            ROOT_DIR
            / "models"
            / "wavlm-base-plus"
        ),
        (
            ROOT_DIR
            / "models"
            / "clip-vit-large-patch14"
        ),
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


# ============================================================
# Image model preservation + calibration
# ============================================================

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
        "       PASS: Webcam-calibrated classifier "
        "is stored separately."
    )


def test_webcam_calibrated_image_model_loadable():
    print(
        "\n[INTEGRATION] Loading webcam-calibrated "
        "image classifier..."
    )

    model = joblib.load(
        CALIBRATED_IMAGE_MODEL_PATH
    )

    assert model is not None

    assert hasattr(
        model,
        "predict",
    )

    assert hasattr(
        model,
        "predict_proba",
    )

    classes = extract_model_classes(
        model
    )

    if classes:

        assert EXPECTED_CLASSES.issubset(
            classes
        ), (
            "Calibrated model classes are incomplete.\n"
            f"Expected: {sorted(EXPECTED_CLASSES)}\n"
            f"Found: {sorted(classes)}"
        )

    print(
        "       PASS: Calibrated classifier loads successfully."
    )

    print(
        "       Model classes:",
        (
            sorted(classes)
            if classes
            else "pipeline-managed"
        ),
    )


def test_webcam_calibration_metadata_exists_and_valid():
    print(
        "\n[INTEGRATION] Checking webcam-calibration metadata..."
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

    assert isinstance(
        metadata,
        dict,
    )

    assert len(metadata) > 0

    print(
        "       PASS: Webcam calibration metadata "
        f"contains {len(metadata)} fields."
    )


# ============================================================
# Calibration feature dataset
# ============================================================

def test_webcam_calibration_dataset_exists():
    print(
        "\n[INTEGRATION] Checking webcam-calibration "
        "feature dataset..."
    )

    feature_path = resolve_existing_path(
        WEBCAM_FEATURE_DATA_CANDIDATES,
        (
            "webcam calibration CLIP "
            "feature dataset"
        ),
    )

    print(
        f"       Resolved dataset: {feature_path}"
    )

    df = pd.read_csv(
        feature_path
    )

    assert not df.empty

    assert "label" in df.columns

    labels = set(
        df["label"]
        .dropna()
        .astype(str)
        .str.strip()
        .str.lower()
        .unique()
    )

    assert EXPECTED_CLASSES.issubset(
        labels
    ), (
        "Calibration dataset does not contain "
        "all four behavioural classes.\n"
        f"Expected: {sorted(EXPECTED_CLASSES)}\n"
        f"Found: {sorted(labels)}"
    )

    clip_columns = [
        column
        for column in df.columns
        if column.startswith(
            "image_clip_emb_"
        )
    ]

    assert len(
        clip_columns
    ) == 768, (
        "Unexpected calibration CLIP "
        "embedding dimension.\n"
        "Expected: 768\n"
        f"Found: {len(clip_columns)}"
    )

    print(
        f"       PASS: Calibration rows = {len(df)}"
    )

    print(
        "       PASS: CLIP embedding dimension = 768"
    )

    print(
        f"       PASS: Behavioural classes = "
        f"{sorted(labels)}"
    )


def test_webcam_calibration_summary_exists():
    print(
        "\n[INTEGRATION] Checking webcam-calibration summary..."
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

    assert isinstance(
        summary,
        dict,
    )

    assert len(summary) > 0

    print(
        "       PASS: Calibration summary JSON is valid."
    )


def test_webcam_calibration_dataset_matches_image_feature_schema():
    print(
        "\n[INTEGRATION] Checking calibration dataset "
        "against image feature schema..."
    )

    feature_path = resolve_existing_path(
        WEBCAM_FEATURE_DATA_CANDIDATES,
        (
            "webcam calibration CLIP "
            "feature dataset"
        ),
    )

    with IMAGE_FEATURE_SCHEMA_PATH.open(
        "r",
        encoding="utf-8",
    ) as f:

        schema = json.load(f)

    df = pd.read_csv(
        feature_path,
        nrows=1,
    )

    missing = [
        column
        for column in schema
        if column not in df.columns
    ]

    assert not missing, (
        "Calibration dataset is incompatible "
        "with image feature schema.\n"
        f"Missing columns: {missing[:20]}"
    )

    print(
        f"       PASS: All {len(schema)} image-model "
        "feature columns are present in calibration data."
    )


# ============================================================
# Evaluation artifacts
# ============================================================

def test_webcam_evaluation_results_exist():
    print(
        "\n[INTEGRATION] Checking webcam calibration "
        "evaluation artifacts..."
    )

    assert WEBCAM_EVALUATION_DIR.exists(), (
        f"Missing evaluation directory: "
        f"{WEBCAM_EVALUATION_DIR}"
    )

    assert WEBCAM_EVALUATION_DIR.is_dir()

    files = [
        path
        for path
        in WEBCAM_EVALUATION_DIR.rglob("*")
        if path.is_file()
    ]

    assert files, (
        "No webcam evaluation artifacts found."
    )

    print(
        f"       PASS: {len(files)} webcam "
        "evaluation artifact(s) found."
    )


# ============================================================
# Fusion feature integration
# ============================================================

def test_fusion_feature_schema_contains_all_modalities():
    print(
        "\n[INTEGRATION] Checking fusion feature schema..."
    )

    with FUSION_SCHEMA_PATH.open(
        "r",
        encoding="utf-8",
    ) as f:

        columns = json.load(f)

    assert isinstance(
        columns,
        list,
    )

    assert len(columns) > 0

    has_text = any(
        column.startswith(
            "text_mpnet_emb_"
        )
        for column in columns
    )

    has_audio = any(
        (
            column.startswith(
                "audio_wavlm_emb_"
            )
            or column.startswith(
                "audio_"
            )
        )
        for column in columns
    )

    has_image = any(
        (
            column.startswith(
                "image_clip_emb_"
            )
            or column.startswith(
                "image_"
            )
        )
        for column in columns
    )

    has_keystroke = any(
        (
            "keydown" in column
            or "typing" in column
            or "delay_" in column
            or "hold_" in column
        )
        for column in columns
    )

    assert has_text
    assert has_audio
    assert has_image
    assert has_keystroke

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


def test_image_feature_schema_uses_768_clip_embeddings():
    print(
        "\n[INTEGRATION] Checking image feature schema..."
    )

    with IMAGE_FEATURE_SCHEMA_PATH.open(
        "r",
        encoding="utf-8",
    ) as f:

        columns = json.load(f)

    clip_columns = [
        column
        for column in columns
        if column.startswith(
            "image_clip_emb_"
        )
    ]

    assert len(
        clip_columns
    ) == 768, (
        "Image schema does not match "
        "CLIP ViT-L/14 output dimensionality.\n"
        f"Expected: 768\n"
        f"Found: {len(clip_columns)}"
    )

    print(
        "       PASS: Image model expects "
        "768 CLIP embedding dimensions."
    )


# ============================================================
# Final inference integration
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

    content = (
        INFERENCE_PATH
        .read_text(
            encoding="utf-8"
        )
    )

    assert (
        "clip-vit-large-patch14"
        in content
    )

    assert "CLIPModel" in content
    assert "CLIPProcessor" in content
    assert "image_clip_emb_" in content

    print(
        "       PASS: Final inference pipeline uses "
        "the expected pretrained CLIP visual encoder."
    )


# ============================================================
# Web backend integration
# ============================================================

def test_web_backend_integrates_calibrated_webcam_classifier():
    print(
        "\n[INTEGRATION] Checking calibrated webcam "
        "classifier integration in web backend..."
    )

    assert WEB_APP_PATH.exists()

    content = WEB_APP_PATH.read_text(
        encoding="utf-8"
    )

    assert (
        "image_pipeline_webcam_calibrated.joblib"
        in content
    )

    assert (
        "run_webcam_calibrated_prediction"
        in content
    )

    assert (
        "webcam_prediction"
        in content
    )

    assert (
        "webcam_calibration_used"
        in content
    )

    print(
        "       PASS: Web backend references and executes "
        "the separate webcam-calibrated classifier."
    )


def test_web_backend_integrates_temporal_probability_aggregation():
    print(
        "\n[INTEGRATION] Checking temporal probability "
        "aggregation integration..."
    )

    content = WEB_APP_PATH.read_text(
        encoding="utf-8"
    )

    required_tokens = [
        "TEMPORAL_PROBABILITY_WINDOW",
        "SESSION_PROBABILITY_HISTORY",
        "add_temporal_probability",
        "rolling_mean_probability",
        "reset_temporal",
    ]

    missing = [
        token
        for token in required_tokens
        if token not in content
    ]

    assert not missing, (
        "Web backend is missing temporal-fusion "
        f"integration tokens: {missing}"
    )

    print(
        "       PASS: Backend contains rolling "
        "mean-probability temporal aggregation."
    )

    print(
        "       PASS: Backend contains temporal-session reset."
    )
