# === tests/test_01_unit.py ===

from __future__ import annotations

import importlib.util
import json
import uuid
from pathlib import Path
from types import ModuleType


# ============================================================
# Project configuration
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
# Module loader
# ============================================================

def load_web_app_module() -> ModuleType:
    """
    Load web_app/app.py directly from its file path.

    The module is loaded without starting the FastAPI lifespan, so these
    tests remain lightweight and do not initialise large pretrained models.
    """

    assert APP_PATH.exists(), (
        f"Missing web application: {APP_PATH}"
    )

    module_name = (
        "sensefuze_web_app_unit_"
        + uuid.uuid4().hex
    )

    spec = importlib.util.spec_from_file_location(
        module_name,
        APP_PATH,
    )

    assert spec is not None
    assert spec.loader is not None

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    return module


# ============================================================
# Confidence-level tests
# ============================================================

def test_confidence_level_logic():
    print(
        "\n[UNIT] Testing confidence-level thresholds..."
    )

    app = load_web_app_module()

    assert app.get_confidence_level(0.40) == "High"
    assert app.get_confidence_level(0.35) == "High"

    assert app.get_confidence_level(0.20) == "Medium"
    assert app.get_confidence_level(0.15) == "Medium"

    assert app.get_confidence_level(0.1499) == "Low"
    assert app.get_confidence_level(0.05) == "Low"
    assert app.get_confidence_level(0.00) == "Low"

    print(
        "       PASS: High / Medium / Low thresholds "
        "behave correctly."
    )


# ============================================================
# Keystroke tests
# ============================================================

def test_keystroke_count_extraction():
    print(
        "\n[UNIT] Testing keystroke key-down counting..."
    )

    app = load_web_app_module()

    events = [
        {"type": "down", "key": "a"},
        {"type": "up", "key": "a"},
        {"type": "down", "key": "b"},
        {"type": "up", "key": "b"},
        {"type": "down", "key": "space"},
    ]

    result = app.extract_keystroke_count(
        json.dumps(events)
    )

    assert result == 3

    print(
        f"       PASS: Expected 3 key-down events, "
        f"received {result}."
    )


def test_invalid_keystroke_json_returns_zero():
    print(
        "\n[UNIT] Testing malformed keystroke JSON handling..."
    )

    app = load_web_app_module()

    result = app.extract_keystroke_count(
        "invalid-json"
    )

    assert result == 0

    print(
        "       PASS: Invalid JSON safely produces "
        "zero keypresses."
    )


def test_empty_keystroke_list_returns_zero():
    print(
        "\n[UNIT] Testing empty keystroke input..."
    )

    app = load_web_app_module()

    result = app.extract_keystroke_count(
        "[]"
    )

    assert result == 0

    print(
        "       PASS: Empty event list produces "
        "zero keypresses."
    )


# ============================================================
# Fallback prediction tests
# ============================================================

def test_fallback_prediction_outputs_valid_distribution():
    print(
        "\n[UNIT] Testing fallback behavioural "
        "probability distribution..."
    )

    app = load_web_app_module()

    probabilities = app.fallback_prediction(
        (
            "I feel tired and distracted while trying "
            "to complete this task."
        )
    )

    assert set(probabilities.keys()) == EXPECTED_CLASSES

    total_probability = sum(
        probabilities.values()
    )

    assert abs(
        total_probability - 1.0
    ) < 1e-6

    for label, probability in probabilities.items():

        assert 0.0 <= probability <= 1.0, (
            f"Invalid probability for {label}: "
            f"{probability}"
        )

    print(
        "       PASS: Four behavioural classes returned "
        f"and probabilities sum to "
        f"{total_probability:.6f}."
    )


def test_fallback_prediction_can_emphasise_fatigued():
    print(
        "\n[UNIT] Testing fatigue cue handling "
        "in fallback prediction..."
    )

    app = load_web_app_module()

    probabilities = app.fallback_prediction(
        "I am extremely tired, sleepy and exhausted."
    )

    assert probabilities["fatigued"] > 0.20

    print(
        "       PASS: Fatigue-related text increases "
        f"fatigued probability to "
        f"{probabilities['fatigued']:.4f}."
    )


# ============================================================
# Probability normalisation tests
# ============================================================

def test_probability_normalisation_sums_to_one():
    print(
        "\n[UNIT] Testing probability normalisation..."
    )

    app = load_web_app_module()

    probabilities = (
        app.normalise_probability_distribution(
            {
                "focused": 4.0,
                "distracted": 2.0,
                "fatigued": 1.0,
                "overloaded": 1.0,
            }
        )
    )

    assert set(probabilities) == EXPECTED_CLASSES

    assert abs(
        sum(probabilities.values()) - 1.0
    ) < 1e-9

    assert abs(
        probabilities["focused"] - 0.50
    ) < 1e-9

    print(
        "       PASS: Arbitrary non-negative scores are "
        "normalised to a valid four-class distribution."
    )


def test_probability_normalisation_handles_zero_distribution():
    print(
        "\n[UNIT] Testing zero-distribution fallback..."
    )

    app = load_web_app_module()

    probabilities = (
        app.normalise_probability_distribution(
            {
                "focused": 0.0,
                "distracted": 0.0,
                "fatigued": 0.0,
                "overloaded": 0.0,
            }
        )
    )

    for label in EXPECTED_CLASSES:
        assert abs(
            probabilities[label] - 0.25
        ) < 1e-9

    print(
        "       PASS: Zero-valued distribution becomes "
        "a uniform four-class distribution."
    )


# ============================================================
# Temporal probability aggregation tests
# ============================================================

def test_temporal_probability_aggregation_uses_mean():
    print(
        "\n[UNIT] Testing temporal mean-probability "
        "aggregation..."
    )

    app = load_web_app_module()

    session_id = (
        "unit-temporal-mean-"
        + uuid.uuid4().hex
    )

    app.clear_temporal_session(
        session_id
    )

    first = {
        "focused": 0.80,
        "distracted": 0.10,
        "fatigued": 0.05,
        "overloaded": 0.05,
    }

    second = {
        "focused": 0.20,
        "distracted": 0.60,
        "fatigued": 0.10,
        "overloaded": 0.10,
    }

    aggregated_1, count_1 = (
        app.add_temporal_probability(
            session_id,
            first,
        )
    )

    aggregated_2, count_2 = (
        app.add_temporal_probability(
            session_id,
            second,
        )
    )

    assert count_1 == 1
    assert count_2 == 2

    assert abs(
        aggregated_1["focused"] - 0.80
    ) < 1e-9

    # Mean:
    # focused    = (0.80 + 0.20) / 2 = 0.50
    # distracted = (0.10 + 0.60) / 2 = 0.35

    assert abs(
        aggregated_2["focused"] - 0.50
    ) < 1e-9

    assert abs(
        aggregated_2["distracted"] - 0.35
    ) < 1e-9

    assert abs(
        sum(aggregated_2.values()) - 1.0
    ) < 1e-9

    app.clear_temporal_session(
        session_id
    )

    print(
        "       PASS: Temporal aggregation computes "
        "the arithmetic mean of recent probability vectors."
    )


def test_temporal_probability_window_keeps_latest_five():
    print(
        "\n[UNIT] Testing rolling temporal window limit..."
    )

    app = load_web_app_module()

    session_id = (
        "unit-window-"
        + uuid.uuid4().hex
    )

    app.clear_temporal_session(
        session_id
    )

    # Observation 1 strongly favours focused.
    app.add_temporal_probability(
        session_id,
        {
            "focused": 1.0,
            "distracted": 0.0,
            "fatigued": 0.0,
            "overloaded": 0.0,
        },
    )

    # Observations 2-6 strongly favour distracted.
    for _ in range(5):

        aggregated, count = (
            app.add_temporal_probability(
                session_id,
                {
                    "focused": 0.0,
                    "distracted": 1.0,
                    "fatigued": 0.0,
                    "overloaded": 0.0,
                },
            )
        )

    assert (
        count
        == app.TEMPORAL_PROBABILITY_WINDOW
    )

    # The initial focused observation has been removed.
    assert abs(
        aggregated["focused"] - 0.0
    ) < 1e-9

    assert abs(
        aggregated["distracted"] - 1.0
    ) < 1e-9

    app.clear_temporal_session(
        session_id
    )

    print(
        "       PASS: Temporal history retains only "
        "the latest five prediction vectors."
    )


def test_temporal_sessions_are_isolated():
    print(
        "\n[UNIT] Testing session-isolated temporal histories..."
    )

    app = load_web_app_module()

    session_a = (
        "unit-session-a-"
        + uuid.uuid4().hex
    )

    session_b = (
        "unit-session-b-"
        + uuid.uuid4().hex
    )

    app.clear_temporal_session(
        session_a
    )

    app.clear_temporal_session(
        session_b
    )

    result_a, count_a = (
        app.add_temporal_probability(
            session_a,
            {
                "focused": 1.0,
                "distracted": 0.0,
                "fatigued": 0.0,
                "overloaded": 0.0,
            },
        )
    )

    result_b, count_b = (
        app.add_temporal_probability(
            session_b,
            {
                "focused": 0.0,
                "distracted": 0.0,
                "fatigued": 1.0,
                "overloaded": 0.0,
            },
        )
    )

    assert count_a == 1
    assert count_b == 1

    assert result_a["focused"] == 1.0
    assert result_b["fatigued"] == 1.0

    app.clear_temporal_session(
        session_a
    )

    app.clear_temporal_session(
        session_b
    )

    print(
        "       PASS: Browser/session probability histories "
        "remain independent."
    )


def test_temporal_session_reset_clears_history():
    print(
        "\n[UNIT] Testing temporal history reset..."
    )

    app = load_web_app_module()

    session_id = (
        "unit-reset-"
        + uuid.uuid4().hex
    )

    app.add_temporal_probability(
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
        in app.SESSION_PROBABILITY_HISTORY
    )

    app.clear_temporal_session(
        session_id
    )

    assert (
        session_id
        not in app.SESSION_PROBABILITY_HISTORY
    )

    assert (
        session_id
        not in app.SESSION_LAST_SEEN
    )

    print(
        "       PASS: Temporal session state is completely cleared."
    )


# ============================================================
# Final output contract
# ============================================================

def test_prediction_normalisation_returns_temporal_primary_state():
    print(
        "\n[UNIT] Testing final temporal prediction contract..."
    )

    app = load_web_app_module()

    session_id = (
        "unit-normalise-"
        + uuid.uuid4().hex
    )

    app.clear_temporal_session(
        session_id
    )

    raw_result_1 = {
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
        "webcam_prediction": None,
    }

    first_result = (
        app.normalise_prediction_result(
            raw=raw_result_1,
            session_id=session_id,
        )
    )

    assert (
        first_result["current_state"]
        == "focused"
    )

    assert (
        first_result["prediction"]
        == "focused"
    )

    assert abs(
        first_result["confidence"] - 0.60
    ) < 1e-9

    assert abs(
        first_result["confidence_gap"] - 0.40
    ) < 1e-9

    assert (
        first_result["confidence_level"]
        == "High"
    )

    assert (
        first_result["raw_prediction"]
        == "focused"
    )

    assert (
        first_result["temporal_samples"]
        == 1
    )

    assert (
        first_result["temporal_window"]
        == app.TEMPORAL_PROBABILITY_WINDOW
    )

    assert (
        first_result["temporal_aggregation"]
        == "rolling_mean_probability"
    )

    assert (
        first_result["current_state"]
        in EXPECTED_CLASSES
    )

    app.clear_temporal_session(
        session_id
    )

    print(
        "       PASS: Final output exposes raw and temporally "
        "aggregated prediction information."
    )


def test_temporal_final_state_can_differ_from_latest_raw_state():
    print(
        "\n[UNIT] Testing temporal result against "
        "latest raw observation..."
    )

    app = load_web_app_module()

    session_id = (
        "unit-raw-vs-aggregate-"
        + uuid.uuid4().hex
    )

    app.clear_temporal_session(
        session_id
    )

    focused_observation = {
        "prediction": "focused",
        "probabilities": {
            "focused": 0.80,
            "distracted": 0.10,
            "fatigued": 0.05,
            "overloaded": 0.05,
        },
        "device": "cpu",
        "feature_dimension": 100,
        "used_modalities": {
            "text": True,
            "keystroke": True,
            "audio": True,
            "image": True,
        },
        "webcam_prediction": None,
    }

    distracted_observation = {
        "prediction": "distracted",
        "probabilities": {
            "focused": 0.30,
            "distracted": 0.50,
            "fatigued": 0.10,
            "overloaded": 0.10,
        },
        "device": "cpu",
        "feature_dimension": 100,
        "used_modalities": {
            "text": True,
            "keystroke": True,
            "audio": True,
            "image": True,
        },
        "webcam_prediction": None,
    }

    # Three focused observations establish temporal history.
    for _ in range(3):

        app.normalise_prediction_result(
            raw=focused_observation,
            session_id=session_id,
        )

    # Latest raw observation switches to distracted.
    result = app.normalise_prediction_result(
        raw=distracted_observation,
        session_id=session_id,
    )

    assert (
        result["raw_prediction"]
        == "distracted"
    )

    # Temporal mean should remain focused.
    assert (
        result["current_state"]
        == "focused"
    )

    assert (
        result["temporal_samples"]
        == 4
    )

    app.clear_temporal_session(
        session_id
    )

    print(
        "       PASS: Final temporal result is not simply "
        "the latest raw prediction."
    )
