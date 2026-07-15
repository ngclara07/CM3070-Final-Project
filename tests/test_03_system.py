# tests/test_03_system.py

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

from fastapi.testclient import TestClient


ROOT_DIR = Path(__file__).resolve().parents[1]
APP_PATH = ROOT_DIR / "web_app" / "app.py"


def load_web_app_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location("sensefuze_web_app", APP_PATH)
    assert spec is not None
    assert spec.loader is not None

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_web_health_endpoint():
    module = load_web_app_module()
    client = TestClient(module.app)

    response = client.get("/health")

    assert response.status_code == 200

    data = response.json()
    assert data["status"] == "ok"
    assert "SenseFuzeAI" in data["service"]


def test_web_model_status_endpoint():
    module = load_web_app_module()
    client = TestClient(module.app)

    response = client.get("/model-status")

    assert response.status_code == 200

    data = response.json()
    assert "fusion_model" in data
    assert "inference_backend" in data
    assert "error" in data


def test_predict_live_rejects_short_text():
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
    assert "20 text characters" in response.json()["detail"]


def test_predict_live_rejects_insufficient_keystrokes():
    module = load_web_app_module()
    client = TestClient(module.app)

    response = client.post(
        "/predict_live",
        data={
            "text": "This is long enough text for behavioural prediction testing.",
            "keystroke_events": "[]",
        },
    )

    assert response.status_code == 400
    assert "20 keypresses" in response.json()["detail"]
