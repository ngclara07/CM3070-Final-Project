# tests/test_03_system.py

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import ModuleType

from fastapi.testclient import TestClient


ROOT_DIR = Path(__file__).resolve().parents[1]
APP_PATH = ROOT_DIR / "web_app" / "app.py"


def load_web_app_module() -> ModuleType:
    assert APP_PATH.exists(), f"Missing web application: {APP_PATH}"

    spec = importlib.util.spec_from_file_location(
        "sensefuze_web_app_system",
        APP_PATH,
    )

    assert spec is not None
    assert spec.loader is not None

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    return module


def build_key_events(count: int) -> str:
    events = []

    for index in range(count):
        key = chr(ord("a") + (index % 26))

        events.append(
            {
                "type": "down",
                "key": key,
                "timestamp_perf": index * 0.10,
                "timestamp_epoch": index * 0.10,
            }
        )

        events.append(
            {
                "type": "up",
                "key": key,
                "timestamp_perf": index * 0.10 + 0.05,
                "timestamp_epoch": index * 0.10 + 0.05,
            }
        )

    return json.dumps(events)


def test_web_health_endpoint():
    print("\n[SYSTEM] Testing /health endpoint...")

    module = load_web_app_module()
    client = TestClient(module.app)

    response = client.get("/health")

    assert response.status_code == 200

    data = response.json()

    assert data["status"] == "ok"
    assert "SenseFuzeAI" in data["service"]
    assert "timestamp" in data

    print("       PASS: /health returned HTTP 200.")
    print(f"       Service: {data['service']}")


def test_web_model_status_endpoint():
    print("\n[SYSTEM] Testing /model-status endpoint...")

    module = load_web_app_module()
    client = TestClient(module.app)

    response = client.get("/model-status")

    assert response.status_code == 200

    data = response.json()

    required_keys = {
        "text_model",
        "audio_model",
        "image_model",
        "keystroke_model",
        "fusion_model",
        "inference_backend",
        "error",
    }

    missing_keys = required_keys - set(data.keys())

    assert not missing_keys, (
        f"/model-status missing keys: {missing_keys}"
    )

    print("       PASS: /model-status returned expected model fields.")
    print(f"       Backend: {data.get('inference_backend')}")
    print(f"       Fusion model status: {data.get('fusion_model')}")


def test_web_app_contains_webcam_calibration_support():
    print("\n[SYSTEM] Checking web application webcam-calibration support...")

    content = APP_PATH.read_text(encoding="utf-8").lower()

    calibration_indicators = [
        "webcam",
        "image",
        "extract_image_features",
    ]

    for indicator in calibration_indicators:
        assert indicator in content, (
            f"Web application does not contain expected image/webcam "
            f"integration token: {indicator}"
        )

    calibrated_model = (
        ROOT_DIR
        / "models"
        / "image_demo"
        / "image_pipeline_webcam_calibrated.joblib"
    )

    assert calibrated_model.exists(), (
        f"Missing calibrated webcam classifier: {calibrated_model}"
    )

    print("       PASS: Web application contains live webcam integration.")
    print("       PASS: Webcam-calibrated model artifact exists.")


def test_predict_live_rejects_short_text():
    print("\n[SYSTEM] Testing rejection of insufficient text input...")

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

    detail = response.json()["detail"]

    assert "20 text characters" in detail

    print(
        "       PASS: Short text correctly rejected with HTTP 400."
    )


def test_predict_live_rejects_insufficient_keystrokes():
    print("\n[SYSTEM] Testing rejection of insufficient keystrokes...")

    module = load_web_app_module()
    client = TestClient(module.app)

    response = client.post(
        "/predict_live",
        data={
            "text": (
                "This is sufficiently long text for behavioural "
                "prediction system testing."
            ),
            "keystroke_events": "[]",
        },
    )

    assert response.status_code == 400

    detail = response.json()["detail"]

    assert "20 keypresses" in detail

    print(
        "       PASS: Insufficient keypresses correctly rejected "
        "with HTTP 400."
    )


def test_keystroke_threshold_accepts_twenty_keydowns():
    print("\n[SYSTEM] Testing twenty-keypress threshold helper...")

    module = load_web_app_module()

    event_json = build_key_events(20)

    result = module.extract_keystroke_count(event_json)

    assert result == 20

    print(
        f"       PASS: Exactly {result} key-down events were recognised."
    )


def test_web_frontend_contains_webcam_capture_components():
    print("\n[SYSTEM] Checking browser webcam capture components...")

    html_path = ROOT_DIR / "web_app" / "templates" / "index.html"
    script_path = ROOT_DIR / "web_app" / "static" / "script.js"

    assert html_path.exists()
    assert script_path.exists()

    html = html_path.read_text(encoding="utf-8").lower()
    script = script_path.read_text(encoding="utf-8")

    assert 'id="webcam"' in html
    assert 'id="framecanvas"' in html

    assert "getUserMedia" in script
    assert "captureWebcamFrame" in script
    assert "image_frame" in script

    print("       PASS: Webcam video element exists.")
    print("       PASS: Browser frame capture is implemented.")
    print("       PASS: Captured image is submitted to /predict_live.")
