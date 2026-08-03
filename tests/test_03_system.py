# === tests/test_03_system.py ===

from __future__ import annotations

import importlib.util
import json
import uuid
from pathlib import Path
from types import ModuleType

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


# ============================================================
# Helpers
# ============================================================

def load_web_app_module() -> ModuleType:
    """
    Load web_app/app.py without requiring package installation.
    """

    assert APP_PATH.exists(), (
        f"Missing web application: {APP_PATH}"
    )

    module_name = (
        "sensefuze_web_app_system_"
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


def build_key_events(
    count: int,
) -> str:
    """
    Build synthetic browser-style key-down/key-up events.
    """

    events = []

    for index in range(count):

        key = chr(
            ord("a")
            + (
                index % 26
            )
        )

        events.append(
            {
                "type": "down",
                "key": key,
                "timestamp_perf": (
                    index * 0.10
                ),
                "timestamp_epoch": (
                    index * 0.10
                ),
            }
        )

        events.append(
            {
                "type": "up",
                "key": key,
                "timestamp_perf": (
                    index * 0.10
                    + 0.05
                ),
                "timestamp_epoch": (
                    index * 0.10
                    + 0.05
                ),
            }
        )

    return json.dumps(
        events
    )


# ============================================================
# Health endpoint
# ============================================================

def test_web_health_endpoint():
    print(
        "\n[SYSTEM] Testing /health endpoint..."
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
        "SenseFuzeAI"
        in data["service"]
    )

    assert "timestamp" in data

    assert (
        "temporal_probability_window"
        in data
    )

    assert (
        data["temporal_probability_window"]
        == module.TEMPORAL_PROBABILITY_WINDOW
    )

    print(
        "       PASS: /health returned HTTP 200."
    )

    print(
        f"       Temporal window: "
        f"{data['temporal_probability_window']}"
    )


# ============================================================
# Model-status endpoint
# ============================================================

def test_web_model_status_endpoint():
    print(
        "\n[SYSTEM] Testing /model-status endpoint..."
    )

    module = load_web_app_module()
    client = TestClient(module.app)

    response = client.get(
        "/model-status"
    )

    assert response.status_code == 200

    data = response.json()

    required_keys = {
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

    missing_keys = (
        required_keys
        - set(data.keys())
    )

    assert not missing_keys, (
        f"/model-status missing keys: "
        f"{sorted(missing_keys)}"
    )

    assert (
        data["temporal_probability_window"]
        == module.TEMPORAL_PROBABILITY_WINDOW
    )

    print(
        "       PASS: /model-status returned expected "
        "model and temporal-fusion fields."
    )


# ============================================================
# Webcam calibration integration
# ============================================================

def test_web_app_contains_webcam_calibration_support():
    print(
        "\n[SYSTEM] Checking web application "
        "webcam-calibration support..."
    )

    content = APP_PATH.read_text(
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

    calibrated_model = (
        ROOT_DIR
        / "models"
        / "image_demo"
        / "image_pipeline_webcam_calibrated.joblib"
    )

    assert calibrated_model.exists(), (
        f"Missing calibrated webcam classifier: "
        f"{calibrated_model}"
    )

    print(
        "       PASS: Web backend integrates "
        "the webcam-calibrated classifier."
    )


# ============================================================
# Input validation
# ============================================================

def test_predict_live_rejects_short_text():
    print(
        "\n[SYSTEM] Testing rejection of "
        "insufficient text input..."
    )

    module = load_web_app_module()
    client = TestClient(module.app)

    response = client.post(
        "/predict_live",
        data={
            "session_id": (
                "system-short-text"
            ),
            "text": "too short",
            "keystroke_events": "[]",
        },
    )

    assert response.status_code == 400

    detail = (
        response.json()["detail"]
    )

    assert (
        "20 text characters"
        in detail
    )

    print(
        "       PASS: Short text correctly "
        "rejected with HTTP 400."
    )


def test_predict_live_rejects_insufficient_keystrokes():
    print(
        "\n[SYSTEM] Testing rejection of "
        "insufficient keystrokes..."
    )

    module = load_web_app_module()
    client = TestClient(module.app)

    response = client.post(
        "/predict_live",
        data={
            "session_id": (
                "system-short-keys"
            ),
            "text": (
                "This is sufficiently long text "
                "for behavioural prediction "
                "system testing."
            ),
            "keystroke_events": "[]",
        },
    )

    assert response.status_code == 400

    detail = (
        response.json()["detail"]
    )

    assert (
        "20 keypresses"
        in detail
    )

    print(
        "       PASS: Insufficient keypresses "
        "correctly rejected with HTTP 400."
    )


def test_keystroke_threshold_accepts_twenty_keydowns():
    print(
        "\n[SYSTEM] Testing twenty-keypress threshold helper..."
    )

    module = load_web_app_module()

    event_json = (
        build_key_events(20)
    )

    result = (
        module.extract_keystroke_count(
            event_json
        )
    )

    assert result == 20

    print(
        f"       PASS: Exactly {result} key-down "
        "events were recognised."
    )


# ============================================================
# Temporal API
# ============================================================

def test_reset_temporal_endpoint():
    print(
        "\n[SYSTEM] Testing /reset_temporal endpoint..."
    )

    module = load_web_app_module()
    client = TestClient(module.app)

    session_id = (
        "system-reset-"
        + uuid.uuid4().hex
    )

    # Manually add history before endpoint reset.
    module.add_temporal_probability(
        session_id,
        {
            "focused": 0.70,
            "distracted": 0.10,
            "fatigued": 0.10,
            "overloaded": 0.10,
        },
    )

    assert (
        session_id
        in module.SESSION_PROBABILITY_HISTORY
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

    assert (
        data["session_id"]
        == session_id
    )

    assert (
        data["temporal_samples"]
        == 0
    )

    assert (
        data["temporal_window"]
        == module.TEMPORAL_PROBABILITY_WINDOW
    )

    assert (
        session_id
        not in module.SESSION_PROBABILITY_HISTORY
    )

    print(
        "       PASS: Temporal state is removed "
        "through the HTTP endpoint."
    )


def test_reset_temporal_rejects_empty_session_id():
    print(
        "\n[SYSTEM] Testing empty temporal session ID..."
    )

    module = load_web_app_module()
    client = TestClient(module.app)

    response = client.post(
        "/reset_temporal",
        data={
            "session_id": "   ",
        },
    )

    assert response.status_code == 400

    assert (
        "session_id"
        in response.json()["detail"]
    )

    print(
        "       PASS: Empty temporal session identifier "
        "is rejected."
    )


# ============================================================
# Frontend system integration
# ============================================================

def test_web_frontend_contains_webcam_capture_components():
    print(
        "\n[SYSTEM] Checking browser webcam "
        "capture components..."
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

    html = html_path.read_text(
        encoding="utf-8"
    ).lower()

    script = script_path.read_text(
        encoding="utf-8"
    )

    assert 'id="webcam"' in html

    assert (
        'id="framecanvas"'
        in html
    )

    assert "getUserMedia" in script

    assert (
        "captureWebcamFrame"
        in script
    )

    assert "image_frame" in script

    print(
        "       PASS: Webcam video/canvas elements exist."
    )

    print(
        "       PASS: Browser webcam capture is implemented."
    )


def test_web_frontend_contains_temporal_prediction_components():
    print(
        "\n[SYSTEM] Checking frontend temporal "
        "prediction components..."
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

    html = html_path.read_text(
        encoding="utf-8"
    ).lower()

    script = script_path.read_text(
        encoding="utf-8"
    )

    required_html_ids = [
        'id="temporalsamples"',
        'id="temporalwindow"',
        'id="temporalwindowstatus"',
        'id="rawprediction"',
        'id="rawconfidence"',
        'id="resettemporalbtn"',
    ]

    for element_id in required_html_ids:

        assert element_id in html, (
            "Missing temporal frontend element: "
            f"{element_id}"
        )

    required_script_tokens = [
        "session_id",
        "resetTemporalWindow",
        "/reset_temporal",
        "temporal_samples",
        "raw_prediction",
        "raw_probabilities",
    ]

    for token in required_script_tokens:

        assert token in script, (
            "Missing temporal frontend logic: "
            f"{token}"
        )

    print(
        "       PASS: Frontend exposes raw and "
        "temporally aggregated outputs."
    )

    print(
        "       PASS: Frontend can reset temporal history."
    )
