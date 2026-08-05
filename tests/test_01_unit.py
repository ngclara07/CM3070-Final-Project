# === tests/test_01_unit.py ===

from __future__ import annotations

import importlib.util
import json
import math
import sys
import uuid

from pathlib import Path
from types import ModuleType

import pytest


# ============================================================
# Project paths
# ============================================================

ROOT_DIR = (
    Path(__file__)
    .resolve()
    .parents[1]
)

APP_PATH = (
    ROOT_DIR
    / "web_app"
    / "app.py"
)

if str(ROOT_DIR) not in sys.path:

    sys.path.insert(
        0,
        str(ROOT_DIR),
    )


# ============================================================
# Canonical temporal implementation
# ============================================================

from temporal_fusion import (
    LABELS,
    TEMPORAL_PROBABILITY_WINDOW,
    StaleGenerationError,
    TemporalFusionEngine,
    aggregate_probability_history,
    confidence_level,
    normalise_probability_dict,
    summarise_probability_dict,
    validate_probability_distribution,
)


EXPECTED_LABELS = (
    "focused",
    "distracted",
    "fatigued",
    "overloaded",
)


# ============================================================
# Lightweight web module loader
# ============================================================

def load_web_app_module() -> ModuleType:
    """
    Import web_app/app.py without running the FastAPI lifespan.

    The module is inserted into sys.modules before execution because
    SessionState is a dataclass and Python's dataclass machinery expects
    its defining module to be registered.
    """

    assert APP_PATH.exists(), (
        f"Missing web backend: {APP_PATH}"
    )

    module_name = (
        "sensefuze_unit_web_"
        + uuid.uuid4().hex
    )

    spec = (
        importlib.util.spec_from_file_location(
            module_name,
            APP_PATH,
        )
    )

    assert spec is not None
    assert spec.loader is not None

    module = (
        importlib.util.module_from_spec(
            spec
        )
    )

    sys.modules[
        module_name
    ] = module

    try:

        spec.loader.exec_module(
            module
        )

    except Exception:

        sys.modules.pop(
            module_name,
            None,
        )

        raise

    return module


@pytest.fixture(
    scope="module"
)
def web_app() -> ModuleType:

    return (
        load_web_app_module()
    )


# ============================================================
# Canonical constants
# ============================================================

def test_canonical_behavioural_labels():

    assert tuple(
        LABELS
    ) == EXPECTED_LABELS


def test_temporal_window_is_five():

    assert (
        TEMPORAL_PROBABILITY_WINDOW
        == 5
    )


# ============================================================
# Probability normalisation
# ============================================================

def test_probability_normalisation():

    probabilities = (
        normalise_probability_dict(
            {
                "focused":
                    4.0,

                "distracted":
                    2.0,

                "fatigued":
                    1.0,

                "overloaded":
                    1.0,
            },
            labels=LABELS,
        )
    )

    assert (
        tuple(
            probabilities.keys()
        )
        == EXPECTED_LABELS
    )

    assert math.isclose(
        sum(
            probabilities.values()
        ),
        1.0,
        abs_tol=1e-12,
    )

    assert math.isclose(
        probabilities[
            "focused"
        ],
        0.5,
        abs_tol=1e-12,
    )


def test_probability_normalisation_clamps_invalid_values():

    probabilities = (
        normalise_probability_dict(
            {
                "focused":
                    float("nan"),

                "distracted":
                    float("inf"),

                "fatigued":
                    -10.0,

                "overloaded":
                    2.0,
            },
            labels=LABELS,
        )
    )

    assert probabilities == {
        "focused":
            0.0,

        "distracted":
            0.0,

        "fatigued":
            0.0,

        "overloaded":
            1.0,
    }


def test_zero_distribution_becomes_uniform():

    probabilities = (
        normalise_probability_dict(
            {
                label:
                    0.0
                for label
                in LABELS
            },
            labels=LABELS,
        )
    )

    for label in LABELS:

        assert math.isclose(
            probabilities[
                label
            ],
            0.25,
            abs_tol=1e-12,
        )


def test_probability_validator_accepts_valid_distribution():

    validation = (
        validate_probability_distribution(
            {
                "focused":
                    0.25,

                "distracted":
                    0.25,

                "fatigued":
                    0.25,

                "overloaded":
                    0.25,
            },
            labels=LABELS,
        )
    )

    assert validation[
        "valid"
    ]

    assert math.isclose(
        validation[
            "probability_sum"
        ],
        1.0,
        abs_tol=1e-12,
    )


def test_probability_validator_rejects_invalid_sum():

    validation = (
        validate_probability_distribution(
            {
                "focused":
                    0.8,

                "distracted":
                    0.8,

                "fatigued":
                    0.0,

                "overloaded":
                    0.0,
            },
            labels=LABELS,
        )
    )

    assert not validation[
        "valid"
    ]


# ============================================================
# Confidence
# ============================================================

@pytest.mark.parametrize(
    (
        "gap",
        "expected",
    ),
    [
        (
            0.00,
            "Low",
        ),
        (
            0.149999,
            "Low",
        ),
        (
            0.15,
            "Medium",
        ),
        (
            0.20,
            "Medium",
        ),
        (
            0.349999,
            "Medium",
        ),
        (
            0.35,
            "High",
        ),
        (
            0.80,
            "High",
        ),
    ],
)
def test_confidence_thresholds(
    gap: float,
    expected: str,
):

    assert (
        confidence_level(
            gap
        )
        == expected
    )


def test_probability_summary():

    summary = (
        summarise_probability_dict(
            {
                "focused":
                    0.62,

                "distracted":
                    0.14,

                "fatigued":
                    0.12,

                "overloaded":
                    0.12,
            },
            labels=LABELS,
        )
    )

    assert (
        summary[
            "current_state"
        ]
        == "focused"
    )

    assert math.isclose(
        summary[
            "confidence"
        ],
        0.62,
        abs_tol=1e-12,
    )

    assert (
        summary[
            "second_class"
        ]
        == "distracted"
    )

    assert math.isclose(
        summary[
            "confidence_gap"
        ],
        0.48,
        abs_tol=1e-12,
    )

    assert (
        summary[
            "confidence_level"
        ]
        == "High"
    )


# ============================================================
# Temporal arithmetic aggregation
# ============================================================

def test_temporal_aggregation_uses_arithmetic_mean():

    history = [
        {
            "focused":
                0.80,
            "distracted":
                0.10,
            "fatigued":
                0.05,
            "overloaded":
                0.05,
        },
        {
            "focused":
                0.20,
            "distracted":
                0.60,
            "fatigued":
                0.10,
            "overloaded":
                0.10,
        },
    ]

    result = (
        aggregate_probability_history(
            history,
            labels=LABELS,
        )
    )

    probabilities = (
        result[
            "probabilities"
        ]
    )

    assert math.isclose(
        probabilities[
            "focused"
        ],
        0.50,
        abs_tol=1e-12,
    )

    assert math.isclose(
        probabilities[
            "distracted"
        ],
        0.35,
        abs_tol=1e-12,
    )

    assert math.isclose(
        probabilities[
            "fatigued"
        ],
        0.075,
        abs_tol=1e-12,
    )

    assert math.isclose(
        probabilities[
            "overloaded"
        ],
        0.075,
        abs_tol=1e-12,
    )


# ============================================================
# TemporalFusionEngine
# ============================================================

def test_temporal_engine_first_observation_equals_raw():

    engine = (
        TemporalFusionEngine()
    )

    raw = {
        "focused":
            0.70,

        "distracted":
            0.10,

        "fatigued":
            0.10,

        "overloaded":
            0.10,
    }

    result = (
        engine.append(
            raw
        )
    )

    assert (
        result[
            "temporal_samples"
        ]
        == 1
    )

    assert not result[
        "temporal_window_full"
    ]

    assert (
        result[
            "probabilities"
        ]
        == raw
    )


def test_temporal_engine_retains_latest_five_only():

    engine = (
        TemporalFusionEngine()
    )

    engine.append(
        {
            "focused":
                1.0,
            "distracted":
                0.0,
            "fatigued":
                0.0,
            "overloaded":
                0.0,
        }
    )

    result = None

    for _ in range(
        5
    ):

        result = (
            engine.append(
                {
                    "focused":
                        0.0,
                    "distracted":
                        1.0,
                    "fatigued":
                        0.0,
                    "overloaded":
                        0.0,
                }
            )
        )

    assert result is not None

    assert (
        result[
            "temporal_samples"
        ]
        == 5
    )

    assert result[
        "temporal_window_full"
    ]

    assert math.isclose(
        result[
            "probabilities"
        ][
            "focused"
        ],
        0.0,
        abs_tol=1e-12,
    )

    assert math.isclose(
        result[
            "probabilities"
        ][
            "distracted"
        ],
        1.0,
        abs_tol=1e-12,
    )


def test_temporal_reset_increments_generation():

    engine = (
        TemporalFusionEngine()
    )

    assert (
        engine.capture_generation()
        == 0
    )

    engine.append(
        {
            "focused":
                0.7,
            "distracted":
                0.1,
            "fatigued":
                0.1,
            "overloaded":
                0.1,
        }
    )

    generation = (
        engine.reset()
    )

    assert generation == 1

    assert (
        engine.capture_generation()
        == 1
    )

    assert (
        engine.sample_count
        == 0
    )


def test_stale_generation_is_rejected():

    engine = (
        TemporalFusionEngine()
    )

    stale_generation = (
        engine.capture_generation()
    )

    engine.reset()

    with pytest.raises(
        StaleGenerationError
    ):

        engine.append(
            {
                "focused":
                    0.7,
                "distracted":
                    0.1,
                "fatigued":
                    0.1,
                "overloaded":
                    0.1,
            },
            expected_generation=(
                stale_generation
            ),
        )

    assert (
        engine.sample_count
        == 0
    )


def test_temporal_engines_are_isolated():

    first = (
        TemporalFusionEngine()
    )

    second = (
        TemporalFusionEngine()
    )

    first_result = (
        first.append(
            {
                "focused":
                    1.0,
                "distracted":
                    0.0,
                "fatigued":
                    0.0,
                "overloaded":
                    0.0,
            }
        )
    )

    second_result = (
        second.append(
            {
                "focused":
                    0.0,
                "distracted":
                    0.0,
                "fatigued":
                    1.0,
                "overloaded":
                    0.0,
            }
        )
    )

    assert (
        first_result[
            "current_state"
        ]
        == "focused"
    )

    assert (
        second_result[
            "current_state"
        ]
        == "fatigued"
    )

    assert (
        first.sample_count
        == 1
    )

    assert (
        second.sample_count
        == 1
    )


# ============================================================
# Web keystroke helpers
# ============================================================

def build_events(
    count: int,
) -> list[
    dict[str, object]
]:

    events: list[
        dict[str, object]
    ] = []

    for index in range(
        count
    ):

        timestamp = (
            index
            * 0.10
        )

        events.append(
            {
                "type":
                    "down",

                "key":
                    chr(
                        ord("a")
                        + (
                            index
                            % 26
                        )
                    ),

                "timestamp_perf":
                    timestamp,

                "timestamp_epoch":
                    timestamp,
            }
        )

        events.append(
            {
                "type":
                    "up",

                "key":
                    chr(
                        ord("a")
                        + (
                            index
                            % 26
                        )
                    ),

                "timestamp_perf":
                    timestamp
                    + 0.04,

                "timestamp_epoch":
                    timestamp
                    + 0.04,
            }
        )

    return events


def test_web_parse_keystrokes(
    web_app: ModuleType,
):

    events = (
        build_events(
            20
        )
    )

    parsed = (
        web_app.parse_keystrokes(
            json.dumps(
                events
            )
        )
    )

    assert (
        parsed
        == events
    )

    assert (
        web_app.count_keydowns(
            parsed
        )
        == 20
    )


def test_web_parse_invalid_keystrokes_returns_empty(
    web_app: ModuleType,
):

    assert (
        web_app.parse_keystrokes(
            "invalid-json"
        )
        == []
    )

    assert (
        web_app.parse_keystrokes(
            "{}"
        )
        == []
    )


def test_web_keystroke_feature_construction(
    web_app: ModuleType,
):

    events = (
        build_events(
            20
        )
    )

    features = (
        web_app.build_live_keystroke_features(
            "this is a natural typing test",
            events,
        )
    )

    assert (
        features[
            "keydown_count"
        ]
        == 20
    )

    assert (
        features[
            "word_count"
        ]
        == 6
    )

    assert (
        features[
            "total_duration_sec"
        ]
        > 0
    )

    assert (
        features[
            "typing_speed_kps"
        ]
        > 0
    )

    assert (
        features[
            "hold_mean"
        ]
        > 0
    )


def test_web_keystroke_feature_threshold(
    web_app: ModuleType,
):

    with pytest.raises(
        ValueError
    ):

        web_app.build_live_keystroke_features(
            "insufficient",
            build_events(
                19
            ),
        )


# ============================================================
# Session and upload helpers
# ============================================================

def test_validate_session_id(
    web_app: ModuleType,
):

    assert (
        web_app.validate_session_id(
            "valid-session"
        )
        == "valid-session"
    )

    with pytest.raises(
        Exception
    ) as exc_info:

        web_app.validate_session_id(
            "   "
        )

    assert (
        getattr(
            exc_info.value,
            "status_code",
            None,
        )
        == 400
    )


def test_safe_suffix(
    web_app: ModuleType,
):

    assert (
        web_app.safe_suffix(
            "sample.wav",
            ".bin",
        )
        == ".wav"
    )

    assert (
        web_app.safe_suffix(
            "sample",
            ".bin",
        )
        == ".bin"
    )

    assert (
        web_app.safe_suffix(
            (
                "sample."
                + "x" * 30
            ),
            ".bin",
        )
        == ".bin"
    )
