# tests/test_sensefuzeai.py (optional)

from __future__ import annotations

import importlib.util
import json
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


def test_required_project_files_exist():
    required_files = [
        ROOT_DIR / "final_multimodal_inference.py",
        ROOT_DIR / "keystroke_live_gui.py",
        ROOT_DIR / "text_live_gui.py",
        ROOT_DIR / "audio_live_gui.py",
        ROOT_DIR / "image_live_gui.py",
        ROOT_DIR / "live_fusion_gui.py",
        ROOT_DIR / "web_app" / "app.py",
        ROOT_DIR / "web_app" / "templates" / "index.html",
        ROOT_DIR / "web_app" / "static" / "script.js",
        ROOT_DIR / "web_app" / "static" / "style.css",
    ]

    missing = [str(path) for path in required_files if not path.exists()]
    assert not missing, f"Missing required files: {missing}"


def test_fusion_feature_schema_valid():
    schema_path = ROOT_DIR / "models" / "fusion_demo" / "feature_columns.json"

    assert schema_path.exists(), f"Missing schema: {schema_path}"

    with schema_path.open("r", encoding="utf-8") as f:
        columns = json.load(f)

    assert isinstance(columns, list)
    assert len(columns) > 0
    assert any(col.startswith("text_mpnet_emb_") for col in columns)
    assert any(col.startswith("audio_wavlm_emb_") for col in columns)
    assert any(col.startswith("image_clip_emb_") for col in columns)
    assert any("keydown" in col or "typing" in col for col in columns)


def test_confidence_level_logic():
    module = load_web_app_module()

    assert module.get_confidence_level(0.40) == "High"
    assert module.get_confidence_level(0.20) == "Medium"
    assert module.get_confidence_level(0.05) == "Low"


def test_keystroke_count_extraction():
    module = load_web_app_module()

    events = [
        {"type": "down", "key": "a"},
        {"type": "up", "key": "a"},
        {"type": "down", "key": "b"},
    ]

    assert module.extract_keystroke_count(json.dumps(events)) == 2


def test_web_health_endpoint():
    module = load_web_app_module()
    client = TestClient(module.app)

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_web_model_status_endpoint():
    module = load_web_app_module()
    client = TestClient(module.app)

    response = client.get("/model-status")

    assert response.status_code == 200

    data = response.json()
    assert "fusion_model" in data
    assert "inference_backend" in data
    assert "error" in data


def test_predict_live_rejects_invalid_input():
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
