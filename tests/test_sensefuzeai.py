# === tests/test_sensefuzeai.py ===

from __future__ import annotations

import importlib.util
import json
import shutil
import sys
import uuid

from pathlib import Path
from types import ModuleType

from fastapi.testclient import (
    TestClient,
)


# ============================================================
# Paths
# ============================================================

ROOT_DIR = (
    Path(__file__)
    .resolve()
    .parents[1]
)

if str(ROOT_DIR) not in sys.path:

    sys.path.insert(
        0,
        str(ROOT_DIR),
    )


TEMPORAL_PATH = (
    ROOT_DIR
    / "temporal_fusion.py"
)

FINAL_INFERENCE_PATH = (
    ROOT_DIR
    / "final_multimodal_inference.py"
)

LIVE_GUI_PATH = (
    ROOT_DIR
    / "live_fusion_gui.py"
)

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

STYLE_PATH = (
    ROOT_DIR
    / "web_app"
    / "static"
    / "style.css"
)

FUSION_SCHEMA_PATH = (
    ROOT_DIR
    / "models"
    / "fusion_demo"
    / "feature_columns.json"
)


# ============================================================
# Canonical temporal imports
# ============================================================

from temporal_fusion import (
    LABELS,
    TEMPORAL_PROBABILITY_WINDOW,
    StaleGenerationError,
    TemporalFusionEngine,
    normalise_probability_dict,
)


# ============================================================
# Module loader
# ============================================================

def load_web_app_module() -> ModuleType:

    module_name = (
        "sensefuze_smoke_web_"
        + uuid.uuid4().hex
    )

    spec = (
        importlib.util
        .spec_from_file_location(
            module_name,
            APP_PATH,
        )
    )

    assert spec is not None
    assert spec.loader is not None

    module = (
        importlib.util
        .module_from_spec(
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


def cleanup_session(
    module: ModuleType,
    session_id: str,
) -> None:

    shutil.rmtree(
        module.session_directory(
            session_id
        ),
        ignore_errors=True,
    )

    with module.SESSION_LOCK:

        module.SESSION_STATES.pop(
            session_id,
            None,
        )


# ============================================================
# Major files
# ============================================================

def test_required_project_files_exist():

    required = [
        TEMPORAL_PATH,
        FINAL_INFERENCE_PATH,
        LIVE_GUI_PATH,
        APP_PATH,
        HTML_PATH,
        SCRIPT_PATH,
        STYLE_PATH,
        (
            ROOT_DIR
            / "evaluate_multimodal_results.py"
        ),
        (
            ROOT_DIR
            / "train_multimodal_comparison.py"
        ),
    ]

    assert all(
        path.exists()
        for path in required
    )


# ============================================================
# Canonical temporal smoke
# ============================================================

def test_temporal_fusion_smoke():

    assert (
        tuple(
            LABELS
        )
        == (
            "focused",
            "distracted",
            "fatigued",
            "overloaded",
        )
    )

    assert (
        TEMPORAL_PROBABILITY_WINDOW
        == 5
    )

    engine = (
        TemporalFusionEngine()
    )

    first = (
        engine.append(
            {
                "focused":
                    0.60,
                "distracted":
                    0.20,
                "fatigued":
                    0.10,
                "overloaded":
                    0.10,
            }
        )
    )

    assert (
        first[
            "current_state"
        ]
        == "focused"
    )

    assert (
        first[
            "temporal_samples"
        ]
        == 1
    )

    assert abs(
        sum(
            first[
                "probabilities"
            ].values()
        )
        - 1.0
    ) < 1e-12


def test_temporal_reset_smoke():

    engine = (
        TemporalFusionEngine()
    )

    old_generation = (
        engine.capture_generation()
    )

    engine.reset()

    assert (
        engine.capture_generation()
        == old_generation + 1
    )

    with pytest_raises_stale_generation():

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
            },
            expected_generation=(
                old_generation
            ),
        )


class pytest_raises_stale_generation:
    """
    Minimal local context manager so this smoke module does not
    require importing pytest solely for a single assertion.
    """

    def __enter__(
        self,
    ):

        return self

    def __exit__(
        self,
        exc_type,
        exc_value,
        traceback,
    ) -> bool:

        if exc_type is None:

            raise AssertionError(
                "Stale temporal generation was accepted."
            )

        if not issubclass(
            exc_type,
            StaleGenerationError,
        ):

            return False

        return True


def test_probability_normalisation_smoke():

    probabilities = (
        normalise_probability_dict(
            {
                "focused":
                    2,
                "distracted":
                    1,
                "fatigued":
                    1,
                "overloaded":
                    0,
            },
            labels=LABELS,
        )
    )

    assert abs(
        sum(
            probabilities.values()
        )
        - 1.0
    ) < 1e-12

    assert (
        probabilities[
            "focused"
        ]
        == 0.5
    )


# ============================================================
# Fusion schema
# ============================================================

def test_fusion_schema_is_multimodal():

    columns = json.loads(
        FUSION_SCHEMA_PATH
        .read_text(
            encoding="utf-8"
        )
    )

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
            or
            "typing" in column
            or
            "delay_" in column
            or
            "hold_" in column
        )
        for column in columns
    )


# ============================================================
# Architecture regression
# ============================================================

def test_final_inference_remains_stateless():

    source = (
        FINAL_INFERENCE_PATH
        .read_text(
            encoding="utf-8"
        )
    )

    assert (
        "from temporal_fusion import"
        in source
    )

    assert (
        "TemporalFusionEngine("
        not in source
    )

    assert (
        '"temporal_fusion_applied":'
        in source
    )


def test_desktop_and_web_use_shared_temporal_engine():

    desktop = (
        LIVE_GUI_PATH
        .read_text(
            encoding="utf-8"
        )
    )

    backend = (
        APP_PATH
        .read_text(
            encoding="utf-8"
        )
    )

    assert (
        "TemporalFusionEngine"
        in desktop
    )

    assert (
        "TemporalFusionEngine"
        in backend
    )

    assert (
        "SESSION_PROBABILITY_HISTORY"
        not in backend
    )

    assert (
        "add_temporal_probability"
        not in backend
    )


# ============================================================
# Web status smoke
# ============================================================

def test_web_health_and_streaming_status_routes():

    module = (
        load_web_app_module()
    )

    client = (
        TestClient(
            module.app
        )
    )

    original_predictor = (
        module.predictor
    )

    try:

        module.predictor = object()

        health = client.get(
            "/health"
        )

        assert health.status_code == 200

        health_data = (
            health.json()
        )

        assert (
            health_data[
                "status"
            ]
            == "ok"
        )

        assert (
            health_data[
                "audio_stream_window_seconds"
            ]
            ==
            module
            .AUDIO_STREAM_WINDOW_SECONDS
        )

        status = client.get(
            "/model-status"
        )

        assert status.status_code == 200

        data = status.json()

        assert (
            data[
                "temporal_probability_window"
            ]
            == 5
        )

        assert (
            data[
                "temporal_fusion_backend"
            ]
            ==
            (
                "temporal_fusion."
                "TemporalFusionEngine"
            )
        )

        assert (
            data[
                "audio_stream_transport"
            ]
            == "websocket_pcm16_mono"
        )

        assert (
            data[
                "audio_source_policy"
            ]
            ==
            (
                "fixed_file_or_continuous_"
                "microphone_stream"
            )
        )

        assert (
            data[
                "stream_packets_reset_temporal"
            ]
            is False
        )

    finally:

        module.predictor = (
            original_predictor
        )


# ============================================================
# Continuous audio lifecycle smoke
# ============================================================

def test_continuous_audio_start_stop_smoke():

    module = (
        load_web_app_module()
    )

    client = (
        TestClient(
            module.app
        )
    )

    session_id = (
        "smoke-audio-"
        + uuid.uuid4().hex
    )

    try:

        start = client.post(
            "/audio_stream/start",
            data={
                "session_id":
                    session_id,
            },
        )

        assert (
            start.status_code
            == 200
        )

        start_data = (
            start.json()
        )

        assert start_data[
            "stream_token"
        ]

        assert (
            start_data[
                "audio_source_kind"
            ]
            == "microphone_stream"
        )

        assert (
            start_data[
                "audio_ready"
            ]
            is False
        )

        first_generation = (
            start_data[
                "generation"
            ]
        )

        stop = client.post(
            "/audio_stream/stop",
            data={
                "session_id":
                    session_id,
            },
        )

        assert (
            stop.status_code
            == 200
        )

        stop_data = (
            stop.json()
        )

        assert (
            stop_data[
                "audio_ready"
            ]
            is False
        )

        assert (
            stop_data[
                "generation"
            ]
            == first_generation + 1
        )

    finally:

        cleanup_session(
            module,
            session_id,
        )


# ============================================================
# Temporal reset route smoke
# ============================================================

def test_web_temporal_reset_preserves_microphone_state():

    module = (
        load_web_app_module()
    )

    client = (
        TestClient(
            module.app
        )
    )

    session_id = (
        "smoke-reset-"
        + uuid.uuid4().hex
    )

    try:

        with module.SESSION_LOCK:

            state = (
                module.get_session(
                    session_id
                )
            )

            state.audio_source_kind = (
                "microphone_stream"
            )

            state.audio_name = (
                "Live microphone"
            )

            state.audio_stream_active = True

            state.audio_stream_token = (
                "smoke-token"
            )

            state.audio_pcm_buffer.extend(
                b"\x00\x00" * 1000
            )

            state.temporal_fusion.append(
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

            old_generation = (
                state
                .temporal_fusion
                .generation
            )

        response = client.post(
            "/reset_temporal",
            data={
                "session_id":
                    session_id,
            },
        )

        assert response.status_code == 200

        data = response.json()

        assert (
            data[
                "generation"
            ]
            == old_generation + 1
        )

        assert (
            data[
                "temporal_samples"
            ]
            == 0
        )

        with module.SESSION_LOCK:

            state = (
                module.get_session(
                    session_id
                )
            )

            assert state.audio_stream_active

            assert (
                state.audio_stream_token
                == "smoke-token"
            )

            assert (
                len(
                    state.audio_pcm_buffer
                )
                > 0
            )

    finally:

        cleanup_session(
            module,
            session_id,
        )


# ============================================================
# Frontend smoke
# ============================================================

def test_frontend_supports_complete_source_lifecycle():

    script = (
        SCRIPT_PATH
        .read_text(
            encoding="utf-8"
        )
    )

    required = [
        "/set_audio_source",
        "/audio_stream/start",
        "/audio_stream/stop",
        "/ws/audio/",
        "/set_visual_image",
        "/set_visual_video",
        "/set_visual_webcam",
        "/stop_visual",
        "/predict_live",
        "/reset_temporal",
        "/full_reset",
        "serverGeneration",
        "clientEpoch",
        "captureWebcamFrame",
        "webcam_frame",
        "startMicrophoneStream",
        "stopMicrophoneStream",
        "float32ToPCM16Buffer",
    ]

    for token in required:

        assert token in script


def test_frontend_does_not_contain_old_one_shot_microphone():

    script = (
        SCRIPT_PATH
        .read_text(
            encoding="utf-8"
        )
    )

    forbidden = [
        "recordMicrophoneOnce",
        "AUDIO_CAPTURE_SECONDS",
        "forceExactDuration",
        "MediaRecorder",
    ]

    for token in forbidden:

        assert token not in script


def test_frontend_does_not_contain_old_temporal_implementation():

    script = (
        SCRIPT_PATH
        .read_text(
            encoding="utf-8"
        )
    )

    forbidden = [
        "temporalProbabilityHistory",
        "aggregateProbabilityHistory",
        "confidenceLevelFromGap",
    ]

    for token in forbidden:

        assert token not in script


def test_frontend_exposes_continuous_audio_ui():

    html = (
        HTML_PATH
        .read_text(
            encoding="utf-8"
        )
        .lower()
    )

    required = [
        'id="startmicbtn"',
        'id="stopmicbtn"',
        'id="audiostreamstate"',
        'id="audiobufferedseconds"',
        'id="audiolivelevel"',
        'id="audiopacketcount"',
        'id="audiodiagnostic"',
    ]

    for token in required:

        assert token in html


def test_frontend_exposes_raw_and_temporal_output():

    html = (
        HTML_PATH
        .read_text(
            encoding="utf-8"
        )
        .lower()
    )

    required = [
        'id="prediction"',
        'id="confidencepercent"',
        'id="probabilities"',
        'id="rawprediction"',
        'id="rawconfidence"',
        'id="rawprobabilities"',
        'id="temporalsamples"',
        'id="temporalwindow"',
        'id="resettemporalbtn"',
    ]

    for token in required:

        assert token in html
