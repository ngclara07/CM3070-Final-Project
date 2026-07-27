# tests/test_sensefuzeai.py

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import ModuleType

import joblib
from fastapi.testclient import TestClient


# ============================================================
# Project paths
# ============================================================

ROOT_DIR = Path(__file__).resolve().parents[1]
APP_PATH = ROOT_DIR / "web_app" / "app.py"

EXPECTED_CLASSES = {
    "focused",
    "distracted",
    "fatigued",
    "overloaded",
}


# ============================================================
# Webcam calibration artifacts
# ============================================================

CALIBRATED_MODEL_PATH = (
    ROOT_DIR
    / "models"
    / "image_demo"
    / "image_pipeline_webcam_calibrated.joblib"
)

CALIBRATED_METADATA_PATH = (
    ROOT_DIR
    / "models"
    / "image_demo"
    / "webcam_calibrated_metadata.json"
)

WEBCAM_EVALUATION_DIR = (
    ROOT_DIR
    / "data"
    / "processed"
    / "webcam_calibration_evaluation"
)

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


# ============================================================
# Helper functions
# ============================================================

def resolve_existing_path(
    candidates: list[Path],
    artifact_name: str,
) -> Path:
    """
    Return the first existing path from a list of acceptable locations.

    Raises an informative AssertionError if none exist.
    """

    for path in candidates:
        if path.exists():
            return path

    checked = "\n".join(
        f"  - {path}"
        for path in candidates
    )

    raise AssertionError(
        f"Could not find {artifact_name}.\n"
        f"Checked the following locations:\n"
        f"{checked}"
    )


def load_web_app_module() -> ModuleType:
    """
    Load web_app/app.py directly from its file path.
    """

    assert APP_PATH.exists(), (
        f"Missing application: {APP_PATH}"
    )

    spec = importlib.util.spec_from_file_location(
        "sensefuze_web_app_smoke",
        APP_PATH,
    )

    assert spec is not None
    assert spec.loader is not None

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    return module


def extract_model_classes(model) -> set[str]:
    """
    Extract class labels from a scikit-learn estimator or pipeline.
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
# Smoke tests
# ============================================================

def test_required_project_files_exist():
    print(
        "\n[SMOKE] Checking major project files..."
    )

    required_files = [
        ROOT_DIR / "final_multimodal_inference.py",

        ROOT_DIR / "keystroke_live_gui.py",
        ROOT_DIR / "text_live_gui.py",
        ROOT_DIR / "audio_live_gui.py",
        ROOT_DIR / "image_live_gui.py",
        ROOT_DIR / "live_fusion_gui.py",

        ROOT_DIR / "build_webcam_calibration_dataset.py",
        ROOT_DIR / "retrain_image_webcam_calibrated.py",

        ROOT_DIR / "web_app" / "app.py",
        ROOT_DIR / "web_app" / "templates" / "index.html",
        ROOT_DIR / "web_app" / "static" / "script.js",
        ROOT_DIR / "web_app" / "static" / "style.css",
    ]

    missing = [
        str(path)
        for path in required_files
        if not path.exists()
    ]

    assert not missing, (
        f"Missing required files: {missing}"
    )

    print(
        f"       PASS: All {len(required_files)} "
        "major project files exist."
    )


def test_webcam_calibrated_artifacts_exist():
    print(
        "\n[SMOKE] Checking webcam-calibrated artifacts..."
    )

    assert CALIBRATED_MODEL_PATH.exists(), (
        f"Missing calibrated image model: "
        f"{CALIBRATED_MODEL_PATH}"
    )

    assert CALIBRATED_METADATA_PATH.exists(), (
        f"Missing calibrated metadata: "
        f"{CALIBRATED_METADATA_PATH}"
    )

    feature_data_path = resolve_existing_path(
        WEBCAM_FEATURE_DATA_CANDIDATES,
        "webcam calibration CLIP feature dataset",
    )

    summary_path = resolve_existing_path(
        WEBCAM_FEATURE_SUMMARY_CANDIDATES,
        "webcam calibration summary JSON",
    )

    assert feature_data_path.is_file(), (
        f"Calibration dataset is not a file: "
        f"{feature_data_path}"
    )

    assert summary_path.is_file(), (
        f"Calibration summary is not a file: "
        f"{summary_path}"
    )

    print(
        "       PASS: Webcam-calibrated image model exists."
    )

    print(
        "       PASS: Webcam-calibrated metadata exists."
    )

    print(
        f"       PASS: Calibration feature dataset resolved at:\n"
        f"             {feature_data_path}"
    )

    print(
        f"       PASS: Calibration summary resolved at:\n"
        f"             {summary_path}"
    )


def test_webcam_calibrated_model_loadable():
    print(
        "\n[SMOKE] Loading webcam-calibrated model..."
    )

    assert CALIBRATED_MODEL_PATH.exists(), (
        f"Missing calibrated model: "
        f"{CALIBRATED_MODEL_PATH}"
    )

    model = joblib.load(
        CALIBRATED_MODEL_PATH
    )

    assert model is not None

    assert hasattr(model, "predict"), (
        "Calibrated model does not expose predict()."
    )

    assert hasattr(model, "predict_proba"), (
        "Calibrated model does not expose predict_proba()."
    )

    classes = extract_model_classes(model)

    if classes:
        assert EXPECTED_CLASSES.issubset(classes), (
            "Calibrated model does not contain all four "
            "behavioural classes.\n"
            f"Expected: {sorted(EXPECTED_CLASSES)}\n"
            f"Found: {sorted(classes)}"
        )

    print(
        "       PASS: Webcam-calibrated classifier "
        "loads successfully."
    )

    print(
        f"       Model classes: "
        f"{sorted(classes) if classes else 'pipeline-managed'}"
    )


def test_webcam_calibration_metadata_valid():
    print(
        "\n[SMOKE] Checking webcam-calibration metadata..."
    )

    assert CALIBRATED_METADATA_PATH.exists()

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
        f"       PASS: Calibration metadata contains "
        f"{len(metadata)} fields."
    )


def test_webcam_evaluation_artifacts_exist():
    print(
        "\n[SMOKE] Checking webcam-calibration "
        "evaluation artifacts..."
    )

    assert WEBCAM_EVALUATION_DIR.exists(), (
        f"Missing evaluation directory: "
        f"{WEBCAM_EVALUATION_DIR}"
    )

    assert WEBCAM_EVALUATION_DIR.is_dir()

    files = [
        path
        for path in WEBCAM_EVALUATION_DIR.rglob("*")
        if path.is_file()
    ]

    assert files, (
        "No webcam-calibration evaluation artifacts found."
    )

    print(
        f"       PASS: {len(files)} evaluation artifact(s) found."
    )


def test_fusion_feature_schema_valid():
    print(
        "\n[SMOKE] Checking final fusion feature schema..."
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

    assert has_text
    assert has_audio
    assert has_image
    assert has_keystroke

    print(
        f"       PASS: Fusion schema contains "
        f"{len(columns)} features across all modalities."
    )


def test_confidence_level_logic():
    print(
        "\n[SMOKE] Testing confidence-level helper..."
    )

    module = load_web_app_module()

    assert module.get_confidence_level(0.40) == "High"
    assert module.get_confidence_level(0.20) == "Medium"
    assert module.get_confidence_level(0.05) == "Low"

    print(
        "       PASS: Confidence-level helper works correctly."
    )


def test_keystroke_count_extraction():
    print(
        "\n[SMOKE] Testing keystroke event counting..."
    )

    module = load_web_app_module()

    events = [
        {"type": "down", "key": "a"},
        {"type": "up", "key": "a"},
        {"type": "down", "key": "b"},
    ]

    result = module.extract_keystroke_count(
        json.dumps(events)
    )

    assert result == 2

    print(
        "       PASS: Keystroke counting returned 2."
    )


def test_fallback_probability_contract():
    print(
        "\n[SMOKE] Testing four-state fallback contract..."
    )

    module = load_web_app_module()

    probabilities = module.fallback_prediction(
        "Testing SenseFuzeAI behavioural state prediction."
    )

    assert set(probabilities) == EXPECTED_CLASSES

    total_probability = sum(
        probabilities.values()
    )

    assert abs(total_probability - 1.0) < 1e-6

    for label, probability in probabilities.items():
        assert 0.0 <= probability <= 1.0, (
            f"Invalid probability for {label}: "
            f"{probability}"
        )

    print(
        "       PASS: Fallback prediction contains "
        "exactly four behavioural states."
    )

    print(
        f"       Probability sum: "
        f"{total_probability:.6f}"
    )


def test_web_health_endpoint():
    print(
        "\n[SMOKE] Testing web health endpoint..."
    )

    module = load_web_app_module()
    client = TestClient(module.app)

    response = client.get("/health")

    assert response.status_code == 200

    data = response.json()

    assert data["status"] == "ok"
    assert "service" in data
    assert "timestamp" in data

    print(
        "       PASS: /health returned HTTP 200."
    )


def test_web_model_status_endpoint():
    print(
        "\n[SMOKE] Testing model-status endpoint..."
    )

    module = load_web_app_module()
    client = TestClient(module.app)

    response = client.get("/model-status")

    assert response.status_code == 200

    data = response.json()

    required_fields = {
        "text_model",
        "audio_model",
        "image_model",
        "keystroke_model",
        "fusion_model",
        "inference_backend",
        "error",
    }

    missing_fields = (
        required_fields
        - set(data.keys())
    )

    assert not missing_fields, (
        f"Missing model-status fields: "
        f"{sorted(missing_fields)}"
    )

    print(
        "       PASS: /model-status exposes expected fields."
    )


def test_predict_live_rejects_invalid_input():
    print(
        "\n[SMOKE] Testing invalid live-prediction request..."
    )

    module = load_web_app_module()
    client = TestClient(module.app)

    response = client.post(
        "/predict_live",
        data={
            "text": "too short",
            "keystroke_events": "[]",
        },
    )

    assert response.status_code == 400

    detail = response.json().get(
        "detail",
        "",
    )

    assert "20 text characters" in detail

    print(
        "       PASS: Invalid live input correctly "
        "returns HTTP 400."
    )


def test_webcam_frontend_capture_support():
    print(
        "\n[SMOKE] Checking browser webcam capture support..."
    )

    html_path = (
        ROOT_DIR
        / "web_app"
        / "templates"
        / "index.html"
    )

    script_path = (
        ROOT_DIR
        / "web_app"
        / "static"
        / "script.js"
    )

    assert html_path.exists()
    assert script_path.exists()

    html = html_path.read_text(
        encoding="utf-8"
    ).lower()

    script = script_path.read_text(
        encoding="utf-8"
    )

    assert 'id="webcam"' in html
    assert 'id="framecanvas"' in html

    assert "getUserMedia" in script
    assert "captureWebcamFrame" in script
    assert "image_frame" in script

    print(
        "       PASS: Webcam HTML element exists."
    )

    print(
        "       PASS: Browser webcam capture is implemented."
    )

    print(
        "       PASS: Captured webcam frames are submitted "
        "to the backend."
    )
