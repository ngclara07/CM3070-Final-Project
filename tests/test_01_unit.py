# tests/test_01_unit.py

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import ModuleType


ROOT_DIR = Path(__file__).resolve().parents[1]
APP_PATH = ROOT_DIR / "web_app" / "app.py"

EXPECTED_CLASSES = {
    "focused",
    "distracted",
    "fatigued",
    "overloaded",
}


def load_web_app_module() -> ModuleType:
    """
    Load web_app/app.py directly from its file path.

    This avoids dependency on the current Python import path and allows
    pytest to execute the tests from the project root reliably.
    """
    assert APP_PATH.exists(), f"Missing web application: {APP_PATH}"

    spec = importlib.util.spec_from_file_location(
        "sensefuze_web_app_unit",
        APP_PATH,
    )

    assert spec is not None
    assert spec.loader is not None

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    return module


def test_confidence_level_logic():
    print("\n[UNIT] Testing confidence-level thresholds...")

    app = load_web_app_module()

    assert app.get_confidence_level(0.40) == "High"
    assert app.get_confidence_level(0.35) == "High"

    assert app.get_confidence_level(0.20) == "Medium"
    assert app.get_confidence_level(0.15) == "Medium"

    assert app.get_confidence_level(0.05) == "Low"
    assert app.get_confidence_level(0.00) == "Low"

    print("       PASS: High / Medium / Low thresholds behave correctly.")


def test_keystroke_count_extraction():
    print("\n[UNIT] Testing keystroke key-down counting...")

    app = load_web_app_module()

    events = [
        {"type": "down", "key": "a"},
        {"type": "up", "key": "a"},
        {"type": "down", "key": "b"},
        {"type": "up", "key": "b"},
        {"type": "down", "key": "space"},
    ]

    result = app.extract_keystroke_count(json.dumps(events))

    assert result == 3

    print(f"       PASS: Expected 3 key-down events, received {result}.")


def test_invalid_keystroke_json_returns_zero():
    print("\n[UNIT] Testing malformed keystroke JSON handling...")

    app = load_web_app_module()

    result = app.extract_keystroke_count("invalid-json")

    assert result == 0

    print("       PASS: Invalid JSON safely produces zero keypresses.")


def test_empty_keystroke_list_returns_zero():
    print("\n[UNIT] Testing empty keystroke input...")

    app = load_web_app_module()

    result = app.extract_keystroke_count("[]")

    assert result == 0

    print("       PASS: Empty event list produces zero keypresses.")


def test_fallback_prediction_outputs_valid_distribution():
    print("\n[UNIT] Testing fallback behavioural probability distribution...")

    app = load_web_app_module()

    probabilities = app.fallback_prediction(
        "I feel tired and distracted while trying to complete this task."
    )

    assert set(probabilities.keys()) == EXPECTED_CLASSES

    total_probability = sum(probabilities.values())

    assert abs(total_probability - 1.0) < 1e-6

    for label, probability in probabilities.items():
        assert 0.0 <= probability <= 1.0, (
            f"Invalid probability for {label}: {probability}"
        )

    print(
        "       PASS: Four behavioural classes returned and "
        f"probabilities sum to {total_probability:.6f}."
    )


def test_fallback_prediction_can_emphasise_fatigued():
    print("\n[UNIT] Testing fatigue cue handling in fallback prediction...")

    app = load_web_app_module()

    probabilities = app.fallback_prediction(
        "I am extremely tired, sleepy and exhausted."
    )

    assert probabilities["fatigued"] > 0.20

    print(
        "       PASS: Fatigue-related text increases the "
        f"fatigued score to {probabilities['fatigued']:.4f}."
    )


def test_prediction_normalisation_returns_one_primary_state():
    print("\n[UNIT] Testing final single-state prediction normalization...")

    app = load_web_app_module()

    raw_result = {
        "prediction": "focused",
        "probabilities": {
            "focused": 0.60,
            "distracted": 0.20,
            "fatigued": 0.12,
            "overloaded": 0.08,
        },
        "device": "cpu",
        "feature_dimension": 100,
        "used_modalities": {
            "text": True,
            "keystroke": True,
            "audio": True,
            "image": True,
        },
    }

    result = app.normalise_prediction_result(raw_result)

    assert result["current_state"] == "focused"
    assert result["prediction"] == "focused"
    assert abs(result["confidence"] - 0.60) < 1e-9
    assert abs(result["confidence_gap"] - 0.40) < 1e-9
    assert result["confidence_level"] == "High"

    assert result["current_state"] in EXPECTED_CLASSES

    print(
        "       PASS: Final output exposes one primary behavioural state "
        "with confidence information."
    )
