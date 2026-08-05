# === tests/test_03_system.py ===

from __future__ import annotations

import importlib.util
import json
import shutil
import sys
import uuid

from pathlib import Path
from types import ModuleType

import cv2
import numpy as np
import pytest

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

if str(ROOT_DIR) not in sys.path:

    sys.path.insert(
        0,
        str(ROOT_DIR),
    )


# ============================================================
# Loader
# ============================================================

def load_web_app_module() -> ModuleType:

    module_name = (
        "sensefuze_system_web_"
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
def app_module() -> ModuleType:

    return (
        load_web_app_module()
    )


@pytest.fixture
def client(
    app_module: ModuleType,
):

    # Do NOT use TestClient as a context manager here.
    #
    # That avoids executing the FastAPI lifespan and therefore avoids
    # loading MPNet/WavLM/CLIP during these deterministic system tests.
    return (
        TestClient(
            app_module.app
        )
    )


# ============================================================
# Synthetic inputs
# ============================================================

def build_key_events(
    count: int,
) -> str:

    events = []

    for index in range(
        count
    ):

        key = chr(
            ord("a")
            + (
                index
                % 26
            )
        )

        timestamp = (
            index
            * 0.10
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
                    + 0.05,

                "timestamp_epoch":
                    timestamp
                    + 0.05,
            }
        )

    return json.dumps(
        events
    )


def build_png_bytes() -> bytes:

    image = np.zeros(
        (
            8,
            8,
            3,
        ),
        dtype=np.uint8,
    )

    image[
        :,
        :,
        1,
    ] = 180

    success, encoded = (
        cv2.imencode(
            ".png",
            image,
        )
    )

    assert success

    return (
        encoded.tobytes()
    )


def cleanup_session_directory(
    app_module: ModuleType,
    session_id: str,
) -> None:

    directory = (
        app_module
        .session_directory(
            session_id
        )
    )

    shutil.rmtree(
        directory,
        ignore_errors=True,
    )

    with app_module.SESSION_LOCK:

        app_module.SESSION_STATES.pop(
            session_id,
            None,
        )


# ============================================================
# Deterministic predictor
# ============================================================

class SequencePredictor:

    def __init__(
        self,
    ) -> None:

        self.index = 0

        self.observations = [
            {
                "prediction":
                    "focused",

                "probabilities": {
                    "focused":
                        0.80,
                    "distracted":
                        0.10,
                    "fatigued":
                        0.05,
                    "overloaded":
                        0.05,
                },
            },
            {
                "prediction":
                    "distracted",

                "probabilities": {
                    "focused":
                        0.20,
                    "distracted":
                        0.60,
                    "fatigued":
                        0.10,
                    "overloaded":
                        0.10,
                },
            },
        ]

    def predict(
        self,
        *,
        keystroke_json,
        text,
        audio_path,
        image_path,
    ):

        observation = (
            self.observations[
                min(
                    self.index,
                    len(
                        self.observations
                    )
                    - 1,
                )
            ]
        )

        self.index += 1

        return {
            **observation,

            "current_state":
                observation[
                    "prediction"
                ],

            "feature_dimension":
                1234,

            "device":
                "cpu-test",

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


# ============================================================
# Health / model status
# ============================================================

def test_health_endpoint(
    app_module: ModuleType,
    client: TestClient,
):

    original_predictor = (
        app_module.predictor
    )

    try:

        app_module.predictor = (
            object()
        )

        response = client.get(
            "/health"
        )

        assert (
            response.status_code
            == 200
        )

        data = response.json()

        assert (
            data[
                "status"
            ]
            == "ok"
        )

        assert (
            data[
                "temporal_probability_window"
            ]
            ==
            app_module
            .TEMPORAL_PROBABILITY_WINDOW
        )

        assert (
            data[
                "live_interval_ms"
            ]
            == 2500
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

    finally:

        app_module.predictor = (
            original_predictor
        )


def test_model_status_endpoint(
    app_module: ModuleType,
    client: TestClient,
):

    response = client.get(
        "/model-status"
    )

    assert (
        response.status_code
        == 200
    )

    data = response.json()

    required = {
        "text_model",
        "audio_model",
        "image_model",
        "keystroke_model",
        "fusion_model",
        "webcam_calibrated_image_model",
        "inference_backend",
        "temporal_fusion_backend",
        "temporal_probability_window",
        "live_interval_ms",
        "audio_capture_seconds",
        "target_audio_sample_rate",
        "min_text_chars",
        "min_keypresses",
        "audio_source_policy",
        "visual_source_modes",
        "error",
    }

    assert (
        required
        <= set(
            data
        )
    )

    assert (
        data[
            "temporal_probability_window"
        ]
        == 5
    )

    assert (
        set(
            data[
                "visual_source_modes"
            ]
        )
        == {
            "image",
            "video",
            "webcam",
        }
    )

    assert (
        data[
            "audio_source_policy"
        ]
        ==
        "fixed_until_replaced_or_reset"
    )

    if "labels" in data:

        assert (
            tuple(
                data[
                    "labels"
                ]
            )
            == (
                "focused",
                "distracted",
                "fatigued",
                "overloaded",
            )
        )


# ============================================================
# Input gating
# ============================================================

def test_predict_live_rejects_short_text(
    app_module: ModuleType,
    client: TestClient,
):

    original_predictor = (
        app_module.predictor
    )

    try:

        app_module.predictor = (
            object()
        )

        response = client.post(
            "/predict_live",
            data={
                "session_id":
                    "system-short-text",

                "generation":
                    "0",

                "text":
                    "too short",

                "keystroke_events":
                    "[]",

                "visual_mode":
                    "none",
            },
        )

        assert (
            response.status_code
            == 409
        )

        assert (
            "Text not ready"
            in str(
                response.json()[
                    "detail"
                ]
            )
        )

    finally:

        app_module.predictor = (
            original_predictor
        )


def test_predict_live_rejects_insufficient_keystrokes(
    app_module: ModuleType,
    client: TestClient,
):

    original_predictor = (
        app_module.predictor
    )

    try:

        app_module.predictor = (
            object()
        )

        response = client.post(
            "/predict_live",
            data={
                "session_id":
                    "system-short-keys",

                "generation":
                    "0",

                "text":
                    (
                        "This text is comfortably longer "
                        "than twenty characters."
                    ),

                "keystroke_events":
                    "[]",

                "visual_mode":
                    "none",
            },
        )

        assert (
            response.status_code
            == 409
        )

        assert (
            "Keystrokes not ready"
            in str(
                response.json()[
                    "detail"
                ]
            )
        )

    finally:

        app_module.predictor = (
            original_predictor
        )


def test_exactly_twenty_keydowns_are_recognised(
    app_module: ModuleType,
):

    events = (
        app_module
        .parse_keystrokes(
            build_key_events(
                20
            )
        )
    )

    assert (
        app_module.count_keydowns(
            events
        )
        == 20
    )


# ============================================================
# Temporal reset endpoint
# ============================================================

def test_reset_temporal_endpoint(
    app_module: ModuleType,
    client: TestClient,
):

    session_id = (
        "system-temporal-reset-"
        + uuid.uuid4().hex
    )

    try:

        with (
            app_module
            .SESSION_LOCK
        ):

            state = (
                app_module
                .get_session(
                    session_id
                )
            )

            state.audio_name = (
                "keep-me.wav"
            )

            old_generation = (
                state
                .temporal_fusion
                .capture_generation()
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

        response = client.post(
            "/reset_temporal",
            data={
                "session_id":
                    session_id,
            },
        )

        assert (
            response.status_code
            == 200
        )

        data = response.json()

        assert (
            data[
                "generation"
            ]
            == old_generation
            + 1
        )

        assert (
            data[
                "temporal_samples"
            ]
            == 0
        )

        with (
            app_module
            .SESSION_LOCK
        ):

            state = (
                app_module
                .get_session(
                    session_id
                )
            )

            # Temporal reset preserves sources.
            assert (
                state.audio_name
                == "keep-me.wav"
            )

            assert (
                state
                .temporal_fusion
                .sample_count
                == 0
            )

    finally:

        cleanup_session_directory(
            app_module,
            session_id,
        )


def test_reset_temporal_rejects_empty_session_id(
    client: TestClient,
):

    response = client.post(
        "/reset_temporal",
        data={
            "session_id":
                "   ",
        },
    )

    assert (
        response.status_code
        == 400
    )


# ============================================================
# Full reset
# ============================================================

def test_full_reset_clears_persistent_sources(
    app_module: ModuleType,
    client: TestClient,
):

    session_id = (
        "system-full-reset-"
        + uuid.uuid4().hex
    )

    try:

        with (
            app_module
            .SESSION_LOCK
        ):

            state = (
                app_module
                .get_session(
                    session_id
                )
            )

            state.audio_name = (
                "audio.wav"
            )

            state.audio_source_kind = (
                "file"
            )

            state.visual_mode = (
                "image"
            )

            state.visual_name = (
                "frame.png"
            )

        response = client.post(
            "/full_reset",
            data={
                "session_id":
                    session_id,
            },
        )

        assert (
            response.status_code
            == 200
        )

        data = response.json()

        assert not data[
            "audio_ready"
        ]

        assert not data[
            "visual_ready"
        ]

        assert (
            data[
                "visual_mode"
            ]
            == "none"
        )

        with (
            app_module
            .SESSION_LOCK
        ):

            state = (
                app_module
                .get_session(
                    session_id
                )
            )

            assert (
                state.audio_path
                is None
            )

            assert (
                state.audio_name
                is None
            )

            assert (
                state.visual_mode
                == "none"
            )

            assert (
                state
                .temporal_fusion
                .sample_count
                == 0
            )

    finally:

        cleanup_session_directory(
            app_module,
            session_id,
        )


# ============================================================
# Stale-generation HTTP protection
# ============================================================

def test_predict_live_rejects_stale_generation(
    app_module: ModuleType,
    client: TestClient,
):

    session_id = (
        "system-stale-"
        + uuid.uuid4().hex
    )

    original_predictor = (
        app_module.predictor
    )

    try:

        app_module.predictor = (
            object()
        )

        visual_response = client.post(
            "/set_visual_webcam",
            data={
                "session_id":
                    session_id,
            },
        )

        assert (
            visual_response.status_code
            == 200
        )

        current_generation = (
            visual_response.json()[
                "generation"
            ]
        )

        response = client.post(
            "/predict_live",
            data={
                "session_id":
                    session_id,

                "generation":
                    str(
                        current_generation
                        - 1
                    ),

                "text":
                    (
                        "This text contains enough characters "
                        "for a valid prediction."
                    ),

                "keystroke_events":
                    build_key_events(
                        20
                    ),

                "visual_mode":
                    "webcam",
            },
        )

        assert (
            response.status_code
            == 409
        )

        detail = response.json()[
            "detail"
        ]

        assert (
            detail[
                "type"
            ]
            == "stale_generation"
        )

        assert (
            detail[
                "generation"
            ]
            == current_generation
        )

    finally:

        app_module.predictor = (
            original_predictor
        )

        cleanup_session_directory(
            app_module,
            session_id,
        )


# ============================================================
# End-to-end web fusion with deterministic model
# ============================================================

def test_complete_mock_live_fusion_pipeline(
    app_module: ModuleType,
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
):

    session_id = (
        "system-complete-"
        + uuid.uuid4().hex
    )

    original_predictor = (
        app_module.predictor
    )

    predictor = (
        SequencePredictor()
    )

    try:

        app_module.predictor = (
            predictor
        )

        monkeypatch.setattr(
            app_module,
            "analyse_audio_file",
            lambda _path: {
                "condition":
                    "near-silence",

                "duration_sec":
                    10.0,

                "rms":
                    0.0,

                "dbfs":
                    -95.0,

                "note":
                    (
                        "Valid quiet-environment "
                        "audio input."
                    ),
            },
        )

        # ----------------------------------------------------
        # Fixed audio source
        # ----------------------------------------------------

        audio_response = client.post(
            "/set_audio_source",
            data={
                "session_id":
                    session_id,

                "source_kind":
                    "file",
            },
            files={
                "audio_file": (
                    "quiet.wav",
                    b"test-audio-content",
                    "audio/wav",
                ),
            },
        )

        assert (
            audio_response.status_code
            == 200
        )

        audio_generation = (
            audio_response.json()[
                "generation"
            ]
        )

        assert (
            audio_generation
            == 1
        )

        # ----------------------------------------------------
        # Static image source
        # ----------------------------------------------------

        image_response = client.post(
            "/set_visual_image",
            data={
                "session_id":
                    session_id,
            },
            files={
                "image_file": (
                    "frame.png",
                    build_png_bytes(),
                    "image/png",
                ),
            },
        )

        assert (
            image_response.status_code
            == 200
        )

        generation = (
            image_response.json()[
                "generation"
            ]
        )

        assert (
            generation
            == 2
        )

        request_data = {
            "session_id":
                session_id,

            "generation":
                str(
                    generation
                ),

            "text":
                (
                    "I am typing naturally on my "
                    "computer keyboard."
                ),

            "keystroke_events":
                build_key_events(
                    20
                ),

            "visual_mode":
                "image",
        }

        # ----------------------------------------------------
        # First observation
        # ----------------------------------------------------

        first_response = client.post(
            "/predict_live",
            data=request_data,
        )

        assert (
            first_response.status_code
            == 200
        )

        first = (
            first_response.json()
        )

        assert (
            first[
                "raw_top_class"
            ]
            == "focused"
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

        assert (
            first[
                "generation"
            ]
            == generation
        )

        # ----------------------------------------------------
        # Second observation
        #
        # Latest raw result is distracted, but equal averaging
        # of the two observations remains focused.
        # ----------------------------------------------------

        second_response = client.post(
            "/predict_live",
            data=request_data,
        )

        assert (
            second_response.status_code
            == 200
        )

        second = (
            second_response.json()
        )

        assert (
            second[
                "raw_top_class"
            ]
            == "distracted"
        )

        assert (
            second[
                "current_state"
            ]
            == "focused"
        )

        assert (
            second[
                "temporal_samples"
            ]
            == 2
        )

        assert abs(
            second[
                "probabilities"
            ][
                "focused"
            ]
            - 0.50
        ) < 1e-9

        assert abs(
            second[
                "probabilities"
            ][
                "distracted"
            ]
            - 0.35
        ) < 1e-9

        assert (
            second[
                "audio_diagnostics"
            ][
                "condition"
            ]
            == "near-silence"
        )

        assert (
            second[
                "visual_source_type"
            ]
            == "image"
        )

        assert (
            set(
                second[
                    "probabilities"
                ]
            )
            == {
                "focused",
                "distracted",
                "fatigued",
                "overloaded",
            }
        )

    finally:

        app_module.predictor = (
            original_predictor
        )

        cleanup_session_directory(
            app_module,
            session_id,
        )


# ============================================================
# Source-change reset behaviour
# ============================================================

def test_visual_source_change_resets_temporal_history(
    app_module: ModuleType,
    client: TestClient,
):

    session_id = (
        "system-source-change-"
        + uuid.uuid4().hex
    )

    try:

        with (
            app_module
            .SESSION_LOCK
        ):

            state = (
                app_module
                .get_session(
                    session_id
                )
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

            previous_generation = (
                state
                .temporal_fusion
                .generation
            )

        response = client.post(
            "/set_visual_webcam",
            data={
                "session_id":
                    session_id,
            },
        )

        assert (
            response.status_code
            == 200
        )

        data = response.json()

        assert (
            data[
                "generation"
            ]
            == previous_generation
            + 1
        )

        with (
            app_module
            .SESSION_LOCK
        ):

            state = (
                app_module
                .get_session(
                    session_id
                )
            )

            assert (
                state
                .temporal_fusion
                .sample_count
                == 0
            )

    finally:

        cleanup_session_directory(
            app_module,
            session_id,
        )


def test_stop_webcam_does_not_increment_generation(
    app_module: ModuleType,
    client: TestClient,
):

    session_id = (
        "system-stop-"
        + uuid.uuid4().hex
    )

    try:

        start = client.post(
            "/set_visual_webcam",
            data={
                "session_id":
                    session_id,
            },
        )

        generation = (
            start.json()[
                "generation"
            ]
        )

        stop = client.post(
            "/stop_visual",
            data={
                "session_id":
                    session_id,
            },
        )

        assert (
            stop.status_code
            == 200
        )

        assert (
            stop.json()[
                "generation"
            ]
            == generation
        )

        assert (
            stop.json()[
                "visual_mode"
            ]
            == "none"
        )

    finally:

        cleanup_session_directory(
            app_module,
            session_id,
        )


# ============================================================
# Frontend integration
# ============================================================

def test_frontend_contains_required_visual_capture_components():

    html = (
        HTML_PATH.read_text(
            encoding="utf-8"
        )
        .lower()
    )

    script = (
        SCRIPT_PATH.read_text(
            encoding="utf-8"
        )
    )

    assert (
        'id="webcam"'
        in html
    )

    assert (
        'id="framecanvas"'
        in html
    )

    assert (
        "getUserMedia"
        in script
    )

    assert (
        "captureWebcamFrame"
        in script
    )

    assert (
        "webcam_frame"
        in script
    )


def test_frontend_contains_generation_safety():

    script = (
        SCRIPT_PATH.read_text(
            encoding="utf-8"
        )
    )

    required = [
        "serverGeneration",
        "clientEpoch",
        "beginStateChange",
        "stale_generation",
        "stale_result",
        "stale_session",
        "visual_mode_mismatch",
    ]

    for token in required:

        assert token in script


def test_frontend_uses_fixed_audio_source():

    script = (
        SCRIPT_PATH.read_text(
            encoding="utf-8"
        )
    )

    assert (
        "/set_audio_source"
        in script
    )

    assert (
        "setAudioFile"
        in script
    )

    assert (
        "recordMicrophoneOnce"
        in script
    )

    assert (
        "MediaRecorder"
        not in script
    )


def test_frontend_displays_raw_and_temporal_results():

    html = (
        HTML_PATH.read_text(
            encoding="utf-8"
        )
        .lower()
    )

    script = (
        SCRIPT_PATH.read_text(
            encoding="utf-8"
        )
    )

    required_html = [
        'id="prediction"',
        'id="confidencepercent"',
        'id="probabilities"',
        'id="rawprediction"',
        'id="rawconfidence"',
        'id="temporalsamples"',
        'id="temporalwindow"',
        'id="resettemporalbtn"',
    ]

    for token in required_html:

        assert token in html

    required_script = [
        "raw_top_class",
        "raw_probabilities",
        "temporal_samples",
        "temporal_window",
        "confidence_gap",
        "confidence_level",
    ]

    for token in required_script:

        assert token in script
