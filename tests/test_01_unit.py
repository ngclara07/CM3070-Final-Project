# === tests/test_01_unit.py ===

from __future__ import annotations

import importlib.util
import json
import math
import sys
import uuid
import wave

from pathlib import Path
from types import ModuleType

import numpy as np
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
    Import web_app/app.py without executing the FastAPI lifespan.

    This allows unit tests to exercise backend helpers without loading
    the heavyweight pretrained MPNet/WavLM/CLIP models.
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
# Synthetic helpers
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

        key = chr(
            ord("a")
            + (
                index
                % 26
            )
        )

        events.append(
            {
                "type":
                    "down",

                "key":
                    key,

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
                    key,

                "timestamp_perf":
                    timestamp
                    + 0.04,

                "timestamp_epoch":
                    timestamp
                    + 0.04,
            }
        )

    return events


def make_sine_pcm16(
    *,
    sample_rate: int,
    duration_seconds: float,
    frequency_hz: float = 440.0,
    amplitude: float = 0.40,
) -> bytes:

    sample_count = int(
        sample_rate
        * duration_seconds
    )

    t = (
        np.arange(
            sample_count,
            dtype=np.float32,
        )
        / float(
            sample_rate
        )
    )

    waveform = (
        amplitude
        * np.sin(
            2.0
            * np.pi
            * frequency_hz
            * t
        )
    )

    pcm = (
        np.clip(
            waveform,
            -1.0,
            1.0,
        )
        * 32767.0
    ).astype(
        "<i2"
    )

    return pcm.tobytes()


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

    assert (
        first.sample_count
        == 1
    )

    assert (
        second.sample_count
        == 0
    )


# ============================================================
# Web keystroke helpers
# ============================================================

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

    assert parsed == events

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

    features = (
        web_app.build_live_keystroke_features(
            "this is a natural typing test",
            build_events(
                20
            ),
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
# Session / path helpers
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


# ============================================================
# Continuous PCM16 audio diagnostics
# ============================================================

def test_pcm16_empty_audio_is_reported(
    web_app: ModuleType,
):

    result = (
        web_app.analyse_pcm16_bytes(
            b""
        )
    )

    assert (
        result[
            "condition"
        ]
        == "empty"
    )

    assert (
        result[
            "duration_sec"
        ]
        == 0.0
    )


def test_pcm16_silence_is_near_silence(
    web_app: ModuleType,
):

    pcm = (
        np.zeros(
            int(
                web_app.TARGET_SR
                * 2.0
            ),
            dtype="<i2",
        )
        .tobytes()
    )

    result = (
        web_app.analyse_pcm16_bytes(
            pcm
        )
    )

    assert (
        result[
            "condition"
        ]
        == "near-silence"
    )

    assert result[
        "duration_sec"
    ] == pytest.approx(
        2.0,
        abs=0.001,
    )

    assert (
        result[
            "dbfs"
        ]
        <= web_app.NEAR_SILENCE_DBFS
    )


def test_pcm16_active_tone_is_detected(
    web_app: ModuleType,
):

    pcm = (
        make_sine_pcm16(
            sample_rate=(
                web_app.TARGET_SR
            ),
            duration_seconds=2.0,
            amplitude=0.40,
        )
    )

    result = (
        web_app.analyse_pcm16_bytes(
            pcm
        )
    )

    assert (
        result[
            "condition"
        ]
        == "active-audio"
    )

    assert result[
        "duration_sec"
    ] == pytest.approx(
        2.0,
        abs=0.001,
    )

    assert (
        result[
            "dbfs"
        ]
        > web_app.QUIET_AUDIO_DBFS
    )


def test_pcm16_odd_trailing_byte_is_ignored(
    web_app: ModuleType,
):

    pcm = (
        np.zeros(
            web_app.TARGET_SR,
            dtype="<i2",
        )
        .tobytes()
        + b"\x00"
    )

    result = (
        web_app.analyse_pcm16_bytes(
            pcm
        )
    )

    assert result[
        "duration_sec"
    ] == pytest.approx(
        1.0,
        abs=0.001,
    )


def test_write_pcm16_wav_creates_valid_wave_file(
    web_app: ModuleType,
    tmp_path: Path,
):

    pcm = (
        np.zeros(
            web_app.TARGET_SR,
            dtype="<i2",
        )
        .tobytes()
    )

    output_path = (
        tmp_path
        / "stream_snapshot.wav"
    )

    result_path = (
        web_app.write_pcm16_wav(
            path=output_path,
            pcm_bytes=pcm,
        )
    )

    assert (
        result_path
        == output_path
    )

    assert output_path.exists()

    with wave.open(
        str(
            output_path
        ),
        "rb",
    ) as wav_file:

        assert (
            wav_file.getnchannels()
            == 1
        )

        assert (
            wav_file.getsampwidth()
            == 2
        )

        assert (
            wav_file.getframerate()
            == web_app.TARGET_SR
        )

        assert (
            wav_file.getnframes()
            == web_app.TARGET_SR
        )


def test_write_pcm16_wav_rejects_empty_buffer(
    web_app: ModuleType,
    tmp_path: Path,
):

    with pytest.raises(
        ValueError
    ):

        web_app.write_pcm16_wav(
            path=(
                tmp_path
                / "empty.wav"
            ),
            pcm_bytes=b"",
        )


# ============================================================
# Continuous audio session state
# ============================================================

def test_session_state_contains_streaming_audio_state(
    web_app: ModuleType,
):

    state = (
        web_app.SessionState()
    )

    assert (
        state.audio_stream_active
        is False
    )

    assert (
        state.audio_stream_token
        is None
    )

    assert isinstance(
        state.audio_pcm_buffer,
        bytearray,
    )

    assert (
        len(
            state.audio_pcm_buffer
        )
        == 0
    )

    assert (
        state.audio_stream_packets
        == 0
    )


def test_clear_audio_stream_state(
    web_app: ModuleType,
):

    state = (
        web_app.SessionState()
    )

    state.audio_stream_active = True
    state.audio_stream_token = "token"
    state.audio_pcm_buffer.extend(
        b"\x00\x00" * 100
    )
    state.audio_stream_packets = 7
    state.audio_stream_last_packet_at = 123.0

    web_app.clear_audio_stream_state(
        state
    )

    assert not state.audio_stream_active
    assert state.audio_stream_token is None
    assert len(state.audio_pcm_buffer) == 0
    assert state.audio_stream_packets == 0
    assert (
        state.audio_stream_last_packet_at
        is None
    )


def test_audio_stream_capacity_is_ten_second_pcm16_window(
    web_app: ModuleType,
):

    expected_bytes = int(
        web_app.AUDIO_STREAM_WINDOW_SECONDS
        * web_app.TARGET_SR
        * 2
    )

    assert (
        web_app.AUDIO_STREAM_WINDOW_SECONDS
        == pytest.approx(
            10.0
        )
    )

    assert (
        expected_bytes
        == 320_000
    )


def test_audio_stream_minimum_readiness_window(
    web_app: ModuleType,
):

    assert (
        web_app.AUDIO_STREAM_MIN_SECONDS
        > 0
    )

    assert (
        web_app.AUDIO_STREAM_MIN_SECONDS
        <= web_app.AUDIO_STREAM_WINDOW_SECONDS
    )


# ============================================================
# Result transformation with streamed audio
# ============================================================

def test_build_prediction_result_includes_stream_metadata(
    web_app: ModuleType,
):

    engine = (
        TemporalFusionEngine()
    )

    generation = (
        engine.capture_generation()
    )

    raw_result = {
        "prediction":
            "focused",

        "probabilities": {
            "focused":
                0.70,
            "distracted":
                0.10,
            "fatigued":
                0.10,
            "overloaded":
                0.10,
        },

        "feature_dimension":
            123,

        "device":
            "cpu",

        "used_modalities": {
            "keystroke":
                True,
            "text":
                True,
            "audio":
                True,
            "image":
                True,
        },

        "image_calibration": {
            "enabled":
                False,
        },
    }

    result = (
        web_app.build_prediction_result(
            raw_result=raw_result,
            temporal_engine=engine,
            expected_generation=(
                generation
            ),
            audio_diagnostics={
                "condition":
                    "active-audio",
                "duration_sec":
                    3.0,
                "rms":
                    0.1,
                "dbfs":
                    -20.0,
            },
            audio_source_kind=(
                "microphone_stream"
            ),
            audio_buffered_seconds=(
                3.0
            ),
            visual_mode=(
                "image"
            ),
            visual_name=(
                "frame.png"
            ),
        )
    )

    assert (
        result[
            "audio_source_kind"
        ]
        == "microphone_stream"
    )

    assert (
        result[
            "audio_stream_buffered_seconds"
        ]
        == pytest.approx(
            3.0
        )
    )

    assert (
        result[
            "temporal_samples"
        ]
        == 1
    )

    assert (
        result[
            "current_state"
        ]
        == "focused"
    )

    assert (
        result[
            "runtime_validation"
        ][
            "pass"
        ]
    )
