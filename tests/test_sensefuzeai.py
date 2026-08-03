# === tests/test_sensefuzeai.py ===

from __future__ import annotations

import importlib.util
import json
import uuid
from pathlib import Path
from types import ModuleType

import joblib
from fastapi.testclient import TestClient


# ============================================================
# Project paths
# ============================================================

ROOT_DIR = Path(__file__).resolve().parents[1]

APP_PATH = (
    ROOT_DIR
    / "web_app"
    / "app.py"
)

HTML_PATH = (
    ROOT_DIR
    / "web_app"
    / "templates"
    / "index.html"
)

SCRIPT_PATH = (
    ROOT_DIR
    / "web_app"
    / "static"
    / "script.js"
)

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


# ============================================================
# Helpers
# ============================================================

def resolve_existing_path(
    candidates: list[Path],
    artifact_name: str,
) -> Path:

    for path in candidates:

        if path.exists():
            return path

    checked = "\n".join(
        f"  - {path}"
        for path in candidates
    )

    raise AssertionError(
        f"Could not find {artifact_name}.\n"
        f"Checked:\n{checked}"
    )


def load_web_app_module() -> ModuleType:

    assert APP_PATH.exists(), (
        f"Missing application: {APP_PATH}"
    )

    module_name = (
        "sensefuze_web_app_smoke_"
        + uuid.uuid4().hex
    )

    spec = importlib.util.spec_from_file_location(
        module_name,
        APP_PATH,
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
# Major files
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
        APP_PATH,
        HTML_PATH,
        SCRIPT_PATH,
        (
            ROOT_DIR
            / "web_app"
            / "static"
            / "style.css"
        ),
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


# ============================================================
# Webcam artifacts
# ============================================================

def test_webcam_calibrated_artifacts_exist():
    print(
        "\n[SMOKE] Checking webcam-calibrated artifacts..."
    )

    assert CALIBRATED_MODEL_PATH.exists()
    assert CALIBRATED_METADATA_PATH.exists()

    feature_path = resolve_existing_path(
        WEBCAM_FEATURE_DATA_CANDIDATES,
        (
            "webcam calibration "
            "CLIP feature dataset"
        ),
    )

    summary_path = resolve_existing_path(
        WEBCAM_FEATURE_SUMMARY_CANDIDATES,
        "webcam calibration summary JSON",
    )

    assert feature_path.is_file()
    assert summary_path.is_file()

    print(
        "       PASS: Webcam-calibrated artifacts are present."
    )


def test_webcam_calibrated_model_loadable():
    print(
        "\n[SMOKE] Loading webcam-calibrated model..."
    )

    model = joblib.load(
        CALIBRATED_MODEL_PATH
    )

    assert model is not None
    assert hasattr(model, "predict")
    assert hasattr(model, "predict_proba")

    classes = extract_model_classes(
        model
    )

    if classes:

        assert EXPECTED_CLASSES.issubset(
            classes
        )

    print(
        "       PASS: Webcam-calibrated classifier "
        "loads successfully."
    )


def test_webcam_calibration_metadata_valid():
    print(
        "\n[SMOKE] Checking webcam calibration metadata..."
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
        f"       PASS: Calibration metadata contains "
        f"{len(metadata)} fields."
    )


def test_webcam_evaluation_artifacts_exist():
    print(
        "\n[SMOKE] Checking webcam evaluation artifacts..."
    )

    assert WEBCAM_EVALUATION_DIR.exists()

    files = [
        path
        for path
        in WEBCAM_EVALUATION_DIR.rglob("*")
        if path.is_file()
    ]

    assert files

    print(
        f"       PASS: {len(files)} webcam evaluation "
        "artifact(s) found."
    )


# ============================================================
# Fusion schema
# ============================================================

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

    with schema_path.open(
        "r",
        encoding="utf-8",
    ) as f:

        columns = json.load(f)

    assert isinstance(
        columns,
        list,
    )

    assert len(columns) > 0

    assert any(
        column.startswith(
            "text_mpnet_emb_"
        )
        for column in columns
    )

    assert any(
        column.startswith(
            "audio_"
        )
        for column in columns
    )

    assert any(
        column.startswith(
            "image_clip_emb_"
        )
        for column in columns
    )

    assert any(
        (
            "keydown" in column
            or "typing" in column
            or "delay_" in column
            or "hold_" in column
        )
        for column in columns
    )

    print(
        f"       PASS: Fusion schema contains "
        f"{len(columns)} multimodal features."
    )


# ============================================================
# Basic helpers
# ============================================================

def test_confidence_level_logic():
    print(
        "\n[SMOKE] Testing confidence-level helper..."
    )

    module = load_web_app_module()

    assert (
        module.get_confidence_level(
            0.40
        )
        == "High"
    )

    assert (
        module.get_confidence_level(
            0.20
        )
        == "Medium"
    )

    assert (
        module.get_confidence_level(
            0.05
        )
        == "Low"
    )

    print(
        "       PASS: Confidence-level helper works."
    )


def test_keystroke_count_extraction():
    print(
        "\n[SMOKE] Testing keystroke counting..."
    )

    module = load_web_app_module()

    events = [
        {
            "type": "down",
            "key": "a",
        },
        {
            "type": "up",
            "key": "a",
        },
        {
            "type": "down",
            "key": "b",
        },
    ]

    result = (
        module.extract_keystroke_count(
            json.dumps(events)
        )
    )

    assert result == 2

    print(
        "       PASS: Keystroke counting returned 2."
    )


def test_fallback_probability_contract():
    print(
        "\n[SMOKE] Testing fallback prediction contract..."
    )

    module = load_web_app_module()

    probabilities = (
        module.fallback_prediction(
            (
                "Testing SenseFuzeAI "
                "behavioural state prediction."
            )
        )
    )

    assert (
        set(probabilities)
        == EXPECTED_CLASSES
    )

    assert abs(
        sum(probabilities.values())
        - 1.0
    ) < 1e-6

    print(
        "       PASS: Fallback prediction returns "
        "four valid classes."
    )


# ============================================================
# Temporal aggregation smoke tests
# ============================================================

def test_temporal_probability_aggregation_contract():
    print(
        "\n[SMOKE] Testing temporal probability contract..."
    )

    module = load_web_app_module()

    session_id = (
        "smoke-temporal-"
        + uuid.uuid4().hex
    )

    module.clear_temporal_session(
        session_id
    )

    probabilities, count = (
        module.add_temporal_probability(
            session_id,
            {
                "focused": 0.60,
                "distracted": 0.20,
                "fatigued": 0.10,
                "overloaded": 0.10,
            },
        )
    )

    assert count == 1

    assert (
        set(probabilities)
        == EXPECTED_CLASSES
    )

    assert abs(
        sum(probabilities.values())
        - 1.0
    ) < 1e-9

    module.clear_temporal_session(
        session_id
    )

    print(
        "       PASS: Temporal probability history "
        "returns a valid four-class distribution."
    )


# ============================================================
# HTTP endpoints
# ============================================================

def test_web_health_endpoint():
    print(
        "\n[SMOKE] Testing web health endpoint..."
    )

    module = load_web_app_module()
    client = TestClient(module.app)

    response = client.get(
        "/health"
    )

    assert response.status_code == 200

    data = response.json()

    assert data["status"] == "ok"

    assert (
        "temporal_probability_window"
        in data
    )

    print(
        "       PASS: /health returned HTTP 200."
    )


def test_web_model_status_endpoint():
    print(
        "\n[SMOKE] Testing model-status endpoint..."
    )

    module = load_web_app_module()
    client = TestClient(module.app)

    response = client.get(
        "/model-status"
    )

    assert response.status_code == 200

    data = response.json()

    required_fields = {
        "text_model",
        "audio_model",
        "image_model",
        "webcam_calibrated_image_model",
        "keystroke_model",
        "fusion_model",
        "temporal_probability_window",
        "inference_backend",
        "error",
    }

    missing = (
        required_fields
        - set(data)
    )

    assert not missing

    print(
        "       PASS: /model-status exposes "
        "current model and temporal fields."
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
            "session_id": (
                "smoke-invalid-input"
            ),
            "text": "too short",
            "keystroke_events": "[]",
        },
    )

    assert response.status_code == 400

    assert (
        "20 text characters"
        in response.json().get(
            "detail",
            "",
        )
    )

    print(
        "       PASS: Invalid live input returns HTTP 400."
    )


def test_reset_temporal_endpoint():
    print(
        "\n[SMOKE] Testing temporal reset endpoint..."
    )

    module = load_web_app_module()
    client = TestClient(module.app)

    session_id = (
        "smoke-reset-"
        + uuid.uuid4().hex
    )

    module.add_temporal_probability(
        session_id,
        {
            "focused": 0.60,
            "distracted": 0.20,
            "fatigued": 0.10,
            "overloaded": 0.10,
        },
    )

    response = client.post(
        "/reset_temporal",
        data={
            "session_id": session_id,
        },
    )

    assert response.status_code == 200

    data = response.json()

    assert data["status"] == "ok"
    assert data["temporal_samples"] == 0

    print(
        "       PASS: /reset_temporal clears "
        "session probability history."
    )


# ============================================================
# Frontend smoke tests
# ============================================================

def test_webcam_frontend_capture_support():
    print(
        "\n[SMOKE] Checking browser webcam capture support..."
    )

    html = HTML_PATH.read_text(
        encoding="utf-8"
    ).lower()

    script = SCRIPT_PATH.read_text(
        encoding="utf-8"
    )

    assert 'id="webcam"' in html
    assert 'id="framecanvas"' in html

    assert "getUserMedia" in script
    assert "captureWebcamFrame" in script
    assert "image_frame" in script

    print(
        "       PASS: Browser webcam capture is implemented."
    )


def test_temporal_frontend_support():
    print(
        "\n[SMOKE] Checking temporal frontend support..."
    )

    html = HTML_PATH.read_text(
        encoding="utf-8"
    ).lower()

    script = SCRIPT_PATH.read_text(
        encoding="utf-8"
    )

    assert (
        'id="temporalsamples"'
        in html
    )

    assert (
        'id="rawprediction"'
        in html
    )

    assert (
        'id="resettemporalbtn"'
        in html
    )

    assert (
        "temporal_samples"
        in script
    )

    assert (
        "raw_prediction"
        in script
    )

    assert (
        "/reset_temporal"
        in script
    )

    print(
        "       PASS: Temporal prediction diagnostics "
        "are available in the web interface."
    )
