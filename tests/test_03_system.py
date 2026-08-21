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
) -> TestClient:

    # Lifespan intentionally not executed.
    # Deterministic system tests replace the real predictor.
    return TestClient(
        app_module.app
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
            16,
            16,
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

    return encoded.tobytes()


def make_pcm16(
    app_module: ModuleType,
    seconds: float,
    amplitude: float = 0.0,
) -> bytes:

    count = int(
        app_module.TARGET_SR
        * seconds
    )

    if amplitude == 0.0:

        return (
            np.zeros(
                count,
                dtype="<i2",
            )
            .tobytes()
        )

    t = (
        np.arange(
            count,
            dtype=np.float32,
        )
        / app_module.TARGET_SR
    )

    waveform = (
        amplitude
        * np.sin(
            2.0
            * np.pi
            * 440.0
            * t
        )
    )

    return (
        (
            np.clip(
                waveform,
                -1.0,
                1.0,
            )
            * 32767.0
        )
        .astype("<i2")
        .tobytes()
    )


def cleanup_session_directory(
    app_module: ModuleType,
    session_id: str,
) -> None:

    directory = (
        app_module.session_directory(
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

        assert Path(
            keystroke_json
        ).exists()

        assert Path(
            audio_path
        ).exists()

        assert Path(
            image_path
        ).exists()

        assert text

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

        app_module.predictor = object()

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
                "target_audio_sample_rate"
            ]
            ==
            app_module
            .TARGET_SR
        )

        assert (
            data[
                "audio_stream_window_seconds"
            ]
            ==
            app_module
            .AUDIO_STREAM_WINDOW_SECONDS
        )

        assert (
            data[
                "audio_stream_min_seconds"
            ]
            ==
            app_module
            .AUDIO_STREAM_MIN_SECONDS
        )

    finally:

        app_module.predictor = (
            original_predictor
        )


def test_model_status_endpoint_contains_streaming_audio_contract(
    client: TestClient,
):

    response = client.get(
        "/model-status"
    )

    assert response.status_code == 200

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
        "target_audio_sample_rate",
        "audio_stream_window_seconds",
        "audio_stream_min_seconds",
        "audio_stream_transport",
        "audio_source_policy",
        "stream_packets_reset_temporal",
        "min_text_chars",
        "min_keypresses",
        "visual_source_modes",
        "error",
    }

    assert required <= set(data)

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

    assert (
        data[
            "temporal_probability_window"
        ]
        == 5
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

        app_module.predictor = object()

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

        app_module.predictor = object()

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
        app_module.parse_keystrokes(
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
# Temporal reset
# ============================================================

def test_temporal_reset_preserves_active_microphone_stream(
    app_module: ModuleType,
    client: TestClient,
):

    session_id = (
        "system-temporal-stream-"
        + uuid.uuid4().hex
    )

    try:

        with app_module.SESSION_LOCK:

            state = (
                app_module.get_session(
                    session_id
                )
            )

            state.audio_name = (
                "Live microphone"
            )

            state.audio_source_kind = (
                "microphone_stream"
            )

            state.audio_stream_active = True

            state.audio_stream_token = (
                "stream-token"
            )

            state.audio_pcm_buffer.extend(
                make_pcm16(
                    app_module,
                    3.0,
                )
            )

            state.audio_stream_packets = 4

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

        with app_module.SESSION_LOCK:

            state = (
                app_module.get_session(
                    session_id
                )
            )

            assert (
                state.audio_stream_active
                is True
            )

            assert (
                state.audio_stream_token
                == "stream-token"
            )

            assert (
                len(
                    state.audio_pcm_buffer
                )
                > 0
            )

            assert (
                state.audio_source_kind
                == "microphone_stream"
            )

    finally:

        cleanup_session_directory(
            app_module,
            session_id,
        )


# ============================================================
# Full reset
# ============================================================

def test_full_reset_clears_continuous_audio_and_visual_state(
    app_module: ModuleType,
    client: TestClient,
):

    session_id = (
        "system-full-reset-"
        + uuid.uuid4().hex
    )

    try:

        with app_module.SESSION_LOCK:

            state = (
                app_module.get_session(
                    session_id
                )
            )

            state.audio_name = (
                "Live microphone"
            )

            state.audio_source_kind = (
                "microphone_stream"
            )

            state.audio_stream_active = True

            state.audio_stream_token = (
                "token"
            )

            state.audio_pcm_buffer.extend(
                b"\x00\x00" * 1000
            )

            state.audio_stream_packets = 5

            state.visual_mode = "image"
            state.visual_name = "frame.png"

        response = client.post(
            "/full_reset",
            data={
                "session_id":
                    session_id,
            },
        )

        assert response.status_code == 200

        data = response.json()

        assert not data[
            "audio_ready"
        ]

        assert not data[
            "audio_stream_active"
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

        with app_module.SESSION_LOCK:

            state = (
                app_module.get_session(
                    session_id
                )
            )

            assert state.audio_path is None
            assert state.audio_name is None
            assert state.audio_source_kind is None
            assert not state.audio_stream_active
            assert state.audio_stream_token is None
            assert len(state.audio_pcm_buffer) == 0
            assert state.audio_stream_packets == 0
            assert state.visual_mode == "none"
            assert state.temporal_fusion.sample_count == 0

    finally:

        cleanup_session_directory(
            app_module,
            session_id,
        )


# ============================================================
# Stale-generation protection
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

        app_module.predictor = object()

        visual_response = client.post(
            "/set_visual_webcam",
            data={
                "session_id":
                    session_id,
            },
        )

        assert visual_response.status_code == 200

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

        detail = (
            response.json()[
                "detail"
            ]
        )

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
# Fixed-audio backward compatibility
# ============================================================

def test_complete_mock_fixed_audio_pipeline(
    app_module: ModuleType,
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
):

    session_id = (
        "system-fixed-audio-"
        + uuid.uuid4().hex
    )

    original_predictor = (
        app_module.predictor
    )

    predictor = SequencePredictor()

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
                    "Test audio.",
            },
        )

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

        assert audio_response.status_code == 200

        assert (
            audio_response.json()[
                "audio_source_kind"
            ]
            == "file"
        )

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

        assert image_response.status_code == 200

        generation = (
            image_response.json()[
                "generation"
            ]
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

        first = client.post(
            "/predict_live",
            data=request_data,
        )

        assert first.status_code == 200

        result = first.json()

        assert (
            result[
                "current_state"
            ]
            == "focused"
        )

        assert (
            result[
                "audio_source_kind"
            ]
            == "file"
        )

        assert (
            result[
                "temporal_samples"
            ]
            == 1
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
# Continuous microphone end-to-end pipeline
# ============================================================

def test_complete_mock_continuous_audio_pipeline(
    app_module: ModuleType,
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
):

    session_id = (
        "system-streaming-audio-"
        + uuid.uuid4().hex
    )

    original_predictor = (
        app_module.predictor
    )

    predictor = SequencePredictor()

    try:

        app_module.predictor = predictor

        monkeypatch.setattr(
            app_module,
            "AUDIO_STREAM_ACK_SECONDS",
            0.0,
        )

        # ----------------------------------------------------
        # Start continuous microphone source.
        # ----------------------------------------------------

        audio_response = client.post(
            "/audio_stream/start",
            data={
                "session_id":
                    session_id,
            },
        )

        assert audio_response.status_code == 200

        audio_data = audio_response.json()

        assert (
            audio_data[
                "audio_source_kind"
            ]
            == "microphone_stream"
        )

        assert (
            audio_data[
                "generation"
            ]
            == 1
        )

        stream_token = (
            audio_data[
                "stream_token"
            ]
        )

        # ----------------------------------------------------
        # Set image source.
        # This is another source change, therefore generation=2.
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

        assert image_response.status_code == 200

        generation = (
            image_response.json()[
                "generation"
            ]
        )

        assert generation == 2

        # ----------------------------------------------------
        # Keep WebSocket open throughout inference.
        # ----------------------------------------------------

        with client.websocket_connect(
            (
                f"/ws/audio/{session_id}"
                f"?token={stream_token}"
            )
        ) as websocket:

            first_pcm = make_pcm16(
                app_module,
                (
                    app_module
                    .AUDIO_STREAM_MIN_SECONDS
                    + 0.75
                ),
            )

            websocket.send_bytes(
                first_pcm
            )

            audio_status = (
                websocket.receive_json()
            )

            assert (
                audio_status[
                    "type"
                ]
                == "audio_status"
            )

            assert (
                audio_status[
                    "audio_ready"
                ]
                is True
            )

            # PCM arrival itself MUST NOT reset generation.
            with app_module.SESSION_LOCK:

                state = (
                    app_module.get_session(
                        session_id
                    )
                )

                assert (
                    state
                    .temporal_fusion
                    .generation
                    == generation
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
                        "I am currently writing and reviewing "
                        "a substantial section of my report."
                    ),

                "keystroke_events":
                    build_key_events(
                        20
                    ),

                "visual_mode":
                    "image",
            }

            # ------------------------------------------------
            # Prediction 1
            # ------------------------------------------------

            first_response = client.post(
                "/predict_live",
                data=request_data,
            )

            assert first_response.status_code == 200

            first = first_response.json()

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
                    "audio_source_kind"
                ]
                == "microphone_stream"
            )

            assert (
                first[
                    "audio_stream_buffered_seconds"
                ]
                >=
                app_module
                .AUDIO_STREAM_MIN_SECONDS
            )

            # ------------------------------------------------
            # New live microphone samples arrive.
            # ------------------------------------------------

            websocket.send_bytes(
                make_pcm16(
                    app_module,
                    1.0,
                    amplitude=0.20,
                )
            )

            next_audio_status = (
                websocket.receive_json()
            )

            assert (
                next_audio_status[
                    "packets_received"
                ]
                >= 2
            )

            with app_module.SESSION_LOCK:

                state = (
                    app_module.get_session(
                        session_id
                    )
                )

                # Still the same generation.
                assert (
                    state
                    .temporal_fusion
                    .generation
                    == generation
                )

            # ------------------------------------------------
            # Prediction 2
            # ------------------------------------------------

            second_response = client.post(
                "/predict_live",
                data=request_data,
            )

            assert second_response.status_code == 200

            second = second_response.json()

            assert (
                second[
                    "raw_top_class"
                ]
                == "distracted"
            )

            # Mean of:
            # focused     = (0.80 + 0.20) / 2 = 0.50
            # distracted  = (0.10 + 0.60) / 2 = 0.35
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

            assert (
                second[
                    "probabilities"
                ][
                    "focused"
                ]
                == pytest.approx(
                    0.50
                )
            )

            assert (
                second[
                    "probabilities"
                ][
                    "distracted"
                ]
                == pytest.approx(
                    0.35
                )
            )

            assert (
                second[
                    "audio_source_kind"
                ]
                == "microphone_stream"
            )

            # ------------------------------------------------
            # Explicit stop is a source change and therefore
            # invalidates the previous temporal generation.
            # ------------------------------------------------

            stop = client.post(
                "/audio_stream/stop",
                data={
                    "session_id":
                        session_id,
                },
            )

            assert stop.status_code == 200

            assert (
                stop.json()[
                    "generation"
                ]
                == generation + 1
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
# Rolling buffer capacity
# ============================================================

def test_continuous_audio_buffer_keeps_latest_window_only(
    app_module: ModuleType,
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
):

    session_id = (
        "system-buffer-window-"
        + uuid.uuid4().hex
    )

    try:

        monkeypatch.setattr(
            app_module,
            "AUDIO_STREAM_ACK_SECONDS",
            0.0,
        )

        start = client.post(
            "/audio_stream/start",
            data={
                "session_id":
                    session_id,
            },
        )

        token = (
            start.json()[
                "stream_token"
            ]
        )

        with client.websocket_connect(
            (
                f"/ws/audio/{session_id}"
                f"?token={token}"
            )
        ) as websocket:

            # Send 15 seconds. The server should retain only
            # AUDIO_STREAM_WINDOW_SECONDS (10 seconds).
            websocket.send_bytes(
                make_pcm16(
                    app_module,
                    15.0,
                )
            )

            status = (
                websocket.receive_json()
            )

            assert (
                status[
                    "buffered_seconds"
                ]
                ==
                pytest.approx(
                    app_module
                    .AUDIO_STREAM_WINDOW_SECONDS,
                    abs=0.01,
                )
            )

            max_bytes = int(
                app_module
                .AUDIO_STREAM_WINDOW_SECONDS
                * app_module.TARGET_SR
                * 2
            )

            with app_module.SESSION_LOCK:

                state = (
                    app_module.get_session(
                        session_id
                    )
                )

                assert (
                    len(
                        state.audio_pcm_buffer
                    )
                    <= max_bytes
                )

            client.post(
                "/audio_stream/stop",
                data={
                    "session_id":
                        session_id,
                },
            )

    finally:

        cleanup_session_directory(
            app_module,
            session_id,
        )


# ============================================================
# Visual-source behaviour
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

        with app_module.SESSION_LOCK:

            state = (
                app_module.get_session(
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

        assert response.status_code == 200

        assert (
            response.json()[
                "generation"
            ]
            == previous_generation + 1
        )

        with app_module.SESSION_LOCK:

            state = (
                app_module.get_session(
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


def test_stop_webcam_preserves_current_generation(
    app_module: ModuleType,
    client: TestClient,
):

    session_id = (
        "system-stop-visual-"
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

        assert stop.status_code == 200

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
# Frontend system contract
# ============================================================

def test_frontend_contains_continuous_audio_controls():

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
        'id="audiofileinput"',
        'id="audiostreamstate"',
        'id="audiobufferedseconds"',
        'id="audiolivelevel"',
        'id="audiopacketcount"',
    ]

    for token in required:

        assert token in html


def test_frontend_contains_continuous_audio_transport():

    script = (
        SCRIPT_PATH
        .read_text(
            encoding="utf-8"
        )
    )

    required = [
        "startMicrophoneStream",
        "stopMicrophoneStream",
        "/audio_stream/start",
        "/audio_stream/stop",
        "/ws/audio/",
        "new WebSocket",
        "float32ToPCM16Buffer",
        "resampleLinear",
        "audio_pcm_buffer",
    ]

    # audio_pcm_buffer exists only server-side, therefore exclude
    # it from JS token verification.
    for token in required[:-1]:

        assert token in script

    assert (
        "recordMicrophoneOnce"
        not in script
    )


def test_frontend_contains_generation_safety():

    script = (
        SCRIPT_PATH
        .read_text(
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
