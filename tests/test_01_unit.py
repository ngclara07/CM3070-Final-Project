# tests/test_01_unit.py

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import ModuleType


ROOT_DIR = Path(__file__).resolve().parents[1]
APP_PATH = ROOT_DIR / "web_app" / "app.py"


def load_web_app_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location("sensefuze_web_app", APP_PATH)
    assert spec is not None
    assert spec.loader is not None

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_confidence_level_logic():
    app = load_web_app_module()

    assert app.get_confidence_level(0.40) == "High"
    assert app.get_confidence_level(0.20) == "Medium"
    assert app.get_confidence_level(0.05) == "Low"


def test_keystroke_count_extraction():
    app = load_web_app_module()

    events = [
        {"type": "down", "key": "a"},
        {"type": "up", "key": "a"},
        {"type": "down", "key": "b"},
    ]

    assert app.extract_keystroke_count(json.dumps(events)) == 2


def test_invalid_keystroke_json_returns_zero():
    app = load_web_app_module()

    assert app.extract_keystroke_count("invalid-json") == 0


def test_fallback_prediction_outputs_valid_distribution():
    app = load_web_app_module()

    probabilities = app.fallback_prediction("I feel tired and distracted.")

    assert set(probabilities.keys()) == {
        "focused",
        "distracted",
        "fatigued",
        "overloaded",
    }

    assert abs(sum(probabilities.values()) - 1.0) < 1e-6
