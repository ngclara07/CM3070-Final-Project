# === tests/test_02_integration.py ===

from __future__ import annotations

import ast
import importlib.util
import json
import shutil
import sys
import uuid

from pathlib import Path
from types import ModuleType

import joblib
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

if str(ROOT_DIR) not in sys.path:

    sys.path.insert(
        0,
        str(ROOT_DIR),
    )


TEMPORAL_PATH = (
    ROOT_DIR
    / "temporal_fusion.py"
)

INFERENCE_PATH = (
    ROOT_DIR
    / "final_multimodal_inference.py"
)

WEB_APP_PATH = (
    ROOT_DIR
    / "web_app"
    / "app.py"
)

EVALUATION_PATH = (
    ROOT_DIR
    / "evaluate_multimodal_results.py"
)

COMPARISON_PATH = (
    ROOT_DIR
    / "train_multimodal_comparison.py"
)


FUSION_MODEL_PATH = (
    ROOT_DIR
    / "models"
    / "fusion_demo"
    / "fusion_pipeline.joblib"
)

FUSION_SCHEMA_PATH = (
    ROOT_DIR
    / "models"
    / "fusion_demo"
    / "feature_columns.json"
)


IMAGE_MODEL_DIR = (
    ROOT_DIR
    / "models"
    / "image_demo"
)

ORIGINAL_IMAGE_MODEL_PATH = (
    IMAGE_MODEL_DIR
    / "image_pipeline.joblib"
)

CALIBRATED_IMAGE_MODEL_PATH = (
    IMAGE_MODEL_DIR
    / "image_pipeline_webcam_calibrated.joblib"
)

IMAGE_FEATURE_SCHEMA_PATH = (
    IMAGE_MODEL_DIR
    / "feature_columns.json"
)

CALIBRATED_METADATA_PATH = (
    IMAGE_MODEL_DIR
    / "webcam_calibrated_metadata.json"
)


EXPECTED_LABELS = {
    "focused",
    "distracted",
    "fatigued",
    "overloaded",
}


# ============================================================
# Module helper
# ============================================================

def load_module_from_path(
    path: Path,
    prefix: str,
) -> ModuleType:

    assert path.exists(), (
        f"Missing module: {path}"
    )

    module_name = (
        prefix
        + "_"
        + uuid.uuid4().hex
    )

    spec = (
        importlib.util
        .spec_from_file_location(
            module_name,
            path,
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
        load_module_from_path(
            WEB_APP_PATH,
            "sensefuze_integration_web",
        )
    )


@pytest.fixture
def client(
    app_module: ModuleType,
) -> TestClient:

    # Deliberately avoid TestClient(...) as a context manager.
    # This keeps the heavyweight model-loading lifespan disabled
    # for deterministic HTTP/WebSocket integration tests.
    return TestClient(
        app_module.app
    )


# ============================================================
# Generic helpers
# ============================================================

def extract_model_classes(
    model,
) -> set[str]:

    classes = getattr(
        model,
        "classes_",
        None,
    )

    if (
        classes is None
        and
        hasattr(
            model,
            "named_steps",
        )
    ):

        for step in reversed(
            list(
                model.named_steps.values()
            )
        ):

            classes = getattr(
                step,
                "classes_",
                None,
            )

            if classes is not None:
                break

    if classes is None:
        return set()

    return {
        str(value)
        .strip()
        .lower()
        for value
        in classes
    }


def imported_names(
    path: Path,
    module_name: str,
) -> set[str]:

    tree = ast.parse(
        path.read_text(
            encoding="utf-8"
        )
    )

    names: set[str] = set()

    for node in ast.walk(
        tree
    ):

        if (
            isinstance(
                node,
                ast.ImportFrom,
            )
            and
            node.module
            == module_name
        ):

            for alias in node.names:
                names.add(
                    alias.name
                )

    return names


def calls_name(
    path: Path,
    target: str,
) -> bool:

    tree = ast.parse(
        path.read_text(
            encoding="utf-8"
        )
    )

    for node in ast.walk(
        tree
    ):

        if not isinstance(
            node,
            ast.Call,
        ):
            continue

        function = node.func

        if (
            isinstance(
                function,
                ast.Name,
            )
            and
            function.id
            == target
        ):
            return True

        if (
            isinstance(
                function,
                ast.Attribute,
            )
            and
            function.attr
            == target
        ):
            return True

    return False


def cleanup_session(
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


def make_pcm16(
    app_module: ModuleType,
    seconds: float,
    amplitude: float = 0.0,
) -> bytes:

    sample_count = int(
        app_module.TARGET_SR
        * seconds
    )

    if amplitude == 0.0:

        return (
            np.zeros(
                sample_count,
                dtype="<i2",
            )
            .tobytes()
        )

    time_axis = (
        np.arange(
            sample_count,
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
            * time_axis
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


# ============================================================
# Artifact integration
# ============================================================

def test_required_project_model_artifacts_exist():

    required = [
        (
            ROOT_DIR
            / "models"
            / "keystroke_demo"
            / "keystroke_pipeline.joblib"
        ),
        (
            ROOT_DIR
            / "models"
            / "text_demo"
            / "text_pipeline.joblib"
        ),
        (
            ROOT_DIR
            / "models"
            / "audio_demo"
            / "audio_pipeline.joblib"
        ),
        ORIGINAL_IMAGE_MODEL_PATH,
        CALIBRATED_IMAGE_MODEL_PATH,
        FUSION_MODEL_PATH,
        FUSION_SCHEMA_PATH,
        (
            ROOT_DIR
            / "models"
            / "all-mpnet-base-v2"
        ),
        (
            ROOT_DIR
            / "models"
            / "wavlm-base-plus"
        ),
        (
            ROOT_DIR
            / "models"
            / "clip-vit-large-patch14"
        ),
    ]

    missing = [
        str(path)
        for path in required
        if not path.exists()
    ]

    assert not missing, (
        "Missing required model artifacts:\n"
        + "\n".join(
            missing
        )
    )


def test_original_and_calibrated_image_models_are_preserved():

    assert ORIGINAL_IMAGE_MODEL_PATH.exists()
    assert CALIBRATED_IMAGE_MODEL_PATH.exists()

    assert (
        ORIGINAL_IMAGE_MODEL_PATH
        != CALIBRATED_IMAGE_MODEL_PATH
    )


def test_webcam_calibration_metadata_valid():

    assert CALIBRATED_METADATA_PATH.exists()

    metadata = json.loads(
        CALIBRATED_METADATA_PATH
        .read_text(
            encoding="utf-8"
        )
    )

    assert isinstance(
        metadata,
        dict,
    )

    assert metadata


# ============================================================
# Feature schemas
# ============================================================

def test_fusion_schema_contains_all_modalities():

    columns = json.loads(
        FUSION_SCHEMA_PATH
        .read_text(
            encoding="utf-8"
        )
    )

    assert isinstance(
        columns,
        list,
    )

    assert columns

    assert (
        len(columns)
        == len(
            set(columns)
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


def test_image_schema_matches_clip_projection_dimension():

    columns = json.loads(
        IMAGE_FEATURE_SCHEMA_PATH
        .read_text(
            encoding="utf-8"
        )
    )

    clip_columns = [
        column
        for column in columns
        if column.startswith(
            "image_clip_emb_"
        )
    ]

    assert (
        len(
            clip_columns
        )
        == 768
    )


def test_webcam_augmented_fusion_schema_is_complete_when_present():

    columns = json.loads(
        FUSION_SCHEMA_PATH
        .read_text(
            encoding="utf-8"
        )
    )

    webcam_columns = {
        column
        for column in columns
        if column.startswith(
            "image_webcam_"
        )
    }

    if not webcam_columns:
        return

    expected = {
        "image_webcam_focused_prob",
        "image_webcam_distracted_prob",
        "image_webcam_fatigued_prob",
        "image_webcam_overloaded_prob",
        "image_webcam_top_probability",
        "image_webcam_confidence_gap",
    }

    assert webcam_columns == expected


# ============================================================
# Trained-model contracts
# ============================================================

def test_fusion_model_loadable_and_classes_valid():

    model = joblib.load(
        FUSION_MODEL_PATH
    )

    assert hasattr(
        model,
        "predict"
    )

    assert hasattr(
        model,
        "predict_proba"
    )

    classes = (
        extract_model_classes(
            model
        )
    )

    if classes:

        assert (
            classes
            == EXPECTED_LABELS
        )


def test_fusion_model_feature_count_matches_schema_when_available():

    model = joblib.load(
        FUSION_MODEL_PATH
    )

    columns = json.loads(
        FUSION_SCHEMA_PATH
        .read_text(
            encoding="utf-8"
        )
    )

    feature_count = getattr(
        model,
        "n_features_in_",
        None,
    )

    if feature_count is not None:

        assert (
            int(feature_count)
            == len(columns)
        )


def test_webcam_calibrated_model_loadable():

    model = joblib.load(
        CALIBRATED_IMAGE_MODEL_PATH
    )

    assert hasattr(
        model,
        "predict"
    )

    assert hasattr(
        model,
        "predict_proba"
    )

    classes = (
        extract_model_classes(
            model
        )
    )

    if classes:

        assert (
            classes
            == EXPECTED_LABELS
        )


# ============================================================
# Final inference architecture
# ============================================================

def test_final_inference_class_importable():

    module = (
        load_module_from_path(
            INFERENCE_PATH,
            "sensefuze_inference",
        )
    )

    assert hasattr(
        module,
        "FinalMultimodalInference",
    )


def test_final_inference_imports_shared_probability_contract():

    imported = (
        imported_names(
            INFERENCE_PATH,
            "temporal_fusion",
        )
    )

    required = {
        "LABELS",
        "normalise_probability_dict",
        "summarise_probability_dict",
        "validate_probability_distribution",
    }

    assert required <= imported


def test_final_inference_does_not_own_temporal_engine():

    assert not calls_name(
        INFERENCE_PATH,
        "TemporalFusionEngine",
    )


def test_final_inference_public_api_preserved():

    module = (
        load_module_from_path(
            INFERENCE_PATH,
            "sensefuze_inference_api",
        )
    )

    inference_class = (
        module.FinalMultimodalInference
    )

    required_methods = {
        "build_fusion_dataframe",
        "extract_audio_features",
        "extract_clip_embedding",
        "extract_image_features",
        "extract_keystroke_features",
        "extract_text_features",
        "extract_webcam_calibration_features",
        "predict",
    }

    for method in required_methods:

        assert hasattr(
            inference_class,
            method,
        )


# ============================================================
# Web / temporal integration
# ============================================================

def test_web_session_state_has_independent_temporal_engines(
    app_module: ModuleType,
):

    first = (
        app_module.SessionState()
    )

    second = (
        app_module.SessionState()
    )

    assert (
        first.temporal_fusion
        is not
        second.temporal_fusion
    )

    first.temporal_fusion.append(
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
        first.temporal_fusion
        .sample_count
        == 1
    )

    assert (
        second.temporal_fusion
        .sample_count
        == 0
    )


def test_web_session_state_exposes_continuous_audio_fields(
    app_module: ModuleType,
):

    state = (
        app_module.SessionState()
    )

    assert hasattr(
        state,
        "audio_stream_active"
    )

    assert hasattr(
        state,
        "audio_stream_token"
    )

    assert hasattr(
        state,
        "audio_pcm_buffer"
    )

    assert hasattr(
        state,
        "audio_stream_packets"
    )

    assert hasattr(
        state,
        "audio_stream_last_packet_at"
    )


def test_web_build_prediction_result_supports_stream_metadata(
    app_module: ModuleType,
):

    engine = (
        app_module.TemporalFusionEngine()
    )

    generation = (
        engine.capture_generation()
    )

    raw_result = {
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

        "feature_dimension":
            100,

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
        app_module
        .build_prediction_result(
            raw_result=raw_result,
            temporal_engine=engine,
            expected_generation=(
                generation
            ),
            audio_diagnostics={
                "condition":
                    "quiet",
            },
            audio_source_kind=(
                "microphone_stream"
            ),
            audio_buffered_seconds=(
                4.0
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
            4.0
        )
    )

    assert (
        result[
            "temporal_samples"
        ]
        == 1
    )


# ============================================================
# Model-status streaming contract
# ============================================================

def test_model_status_exposes_continuous_audio_contract(
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
        "target_audio_sample_rate",
        "audio_stream_window_seconds",
        "audio_stream_min_seconds",
        "audio_stream_transport",
        "audio_source_policy",
        "stream_packets_reset_temporal",
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


# ============================================================
# Audio stream HTTP lifecycle
# ============================================================

def test_audio_stream_start_creates_token_and_resets_once(
    app_module: ModuleType,
    client: TestClient,
):

    session_id = (
        "integration-audio-start-"
        + uuid.uuid4().hex
    )

    try:

        response = client.post(
            "/audio_stream/start",
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
                "audio_source_kind"
            ]
            == "microphone_stream"
        )

        assert (
            data[
                "audio_ready"
            ]
            is False
        )

        assert data[
            "stream_token"
        ]

        assert (
            data[
                "generation"
            ]
            == 1
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
                ==
                data[
                    "stream_token"
                ]
            )

            assert (
                state.audio_source_kind
                == "microphone_stream"
            )

            assert (
                len(
                    state.audio_pcm_buffer
                )
                == 0
            )

    finally:

        cleanup_session(
            app_module,
            session_id,
        )


def test_websocket_audio_fills_buffer_without_resetting_generation(
    app_module: ModuleType,
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
):

    session_id = (
        "integration-audio-ws-"
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

        assert start.status_code == 200

        start_data = start.json()

        token = (
            start_data[
                "stream_token"
            ]
        )

        generation = (
            start_data[
                "generation"
            ]
        )

        pcm = make_pcm16(
            app_module,
            (
                app_module
                .AUDIO_STREAM_MIN_SECONDS
                + 0.5
            ),
        )

        with client.websocket_connect(
            (
                f"/ws/audio/{session_id}"
                f"?token={token}"
            )
        ) as websocket:

            websocket.send_bytes(
                pcm
            )

            message = (
                websocket.receive_json()
            )

            assert (
                message[
                    "type"
                ]
                == "audio_status"
            )

            assert (
                message[
                    "audio_ready"
                ]
                is True
            )

            assert (
                message[
                    "buffered_seconds"
                ]
                >=
                app_module
                .AUDIO_STREAM_MIN_SECONDS
            )

            assert (
                message[
                    "packets_received"
                ]
                >= 1
            )

            with app_module.SESSION_LOCK:

                state = (
                    app_module.get_session(
                        session_id
                    )
                )

                # Critical invariant:
                # incoming PCM packets must NOT reset temporal fusion.
                assert (
                    state
                    .temporal_fusion
                    .generation
                    == generation
                )

                assert (
                    len(
                        state.audio_pcm_buffer
                    )
                    > 0
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

            assert (
                stop.json()[
                    "generation"
                ]
                == generation + 1
            )

    finally:

        cleanup_session(
            app_module,
            session_id,
        )


def test_websocket_rejects_invalid_stream_token(
    app_module: ModuleType,
    client: TestClient,
):

    session_id = (
        "integration-bad-token-"
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

        assert start.status_code == 200

        with client.websocket_connect(
            (
                f"/ws/audio/{session_id}"
                "?token=definitely-invalid"
            )
        ) as websocket:

            response = (
                websocket.receive_json()
            )

            assert (
                response[
                    "type"
                ]
                == "error"
            )

            assert (
                "token"
                in response[
                    "message"
                ].lower()
            )

    finally:

        cleanup_session(
            app_module,
            session_id,
        )


def test_audio_stream_stop_clears_stream_state(
    app_module: ModuleType,
    client: TestClient,
):

    session_id = (
        "integration-audio-stop-"
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

        generation = (
            start.json()[
                "generation"
            ]
        )

        response = client.post(
            "/audio_stream/stop",
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

        assert (
            data[
                "audio_source_kind"
            ]
            is None
        )

        assert (
            data[
                "generation"
            ]
            == generation + 1
        )

        with app_module.SESSION_LOCK:

            state = (
                app_module.get_session(
                    session_id
                )
            )

            assert not state.audio_stream_active
            assert state.audio_stream_token is None
            assert len(state.audio_pcm_buffer) == 0
            assert state.audio_source_kind is None

    finally:

        cleanup_session(
            app_module,
            session_id,
        )


# ============================================================
# Fixed-file compatibility
# ============================================================

def test_fixed_audio_file_fallback_remains_supported(
    app_module: ModuleType,
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
):

    session_id = (
        "integration-file-audio-"
        + uuid.uuid4().hex
    )

    try:

        monkeypatch.setattr(
            app_module,
            "analyse_audio_file",
            lambda _path: {
                "condition":
                    "quiet",
                "duration_sec":
                    4.0,
                "rms":
                    0.01,
                "dbfs":
                    -40.0,
                "note":
                    "Test audio.",
            },
        )

        response = client.post(
            "/set_audio_source",
            data={
                "session_id":
                    session_id,

                "source_kind":
                    "file",
            },
            files={
                "audio_file": (
                    "test.wav",
                    b"synthetic-test-audio",
                    "audio/wav",
                ),
            },
        )

        assert (
            response.status_code
            == 200
        )

        data = response.json()

        assert (
            data[
                "audio_source_kind"
            ]
            == "file"
        )

        assert (
            data[
                "audio_ready"
            ]
            is True
        )

        with app_module.SESSION_LOCK:

            state = (
                app_module.get_session(
                    session_id
                )
            )

            assert (
                state.audio_source_kind
                == "file"
            )

            assert (
                state.audio_path
                is not None
            )

            assert (
                state.audio_stream_active
                is False
            )

    finally:

        cleanup_session(
            app_module,
            session_id,
        )


def test_switching_from_stream_to_file_invalidates_stream(
    app_module: ModuleType,
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
):

    session_id = (
        "integration-switch-audio-"
        + uuid.uuid4().hex
    )

    try:

        monkeypatch.setattr(
            app_module,
            "analyse_audio_file",
            lambda _path: {
                "condition":
                    "quiet",
                "duration_sec":
                    3.0,
                "rms":
                    0.01,
                "dbfs":
                    -40.0,
                "note":
                    "Test.",
            },
        )

        start = client.post(
            "/audio_stream/start",
            data={
                "session_id":
                    session_id,
            },
        )

        first_generation = (
            start.json()[
                "generation"
            ]
        )

        file_response = client.post(
            "/set_audio_source",
            data={
                "session_id":
                    session_id,

                "source_kind":
                    "file",
            },
            files={
                "audio_file": (
                    "replacement.wav",
                    b"replacement-audio",
                    "audio/wav",
                ),
            },
        )

        assert (
            file_response.status_code
            == 200
        )

        assert (
            file_response.json()[
                "generation"
            ]
            == first_generation + 1
        )

        with app_module.SESSION_LOCK:

            state = (
                app_module.get_session(
                    session_id
                )
            )

            assert (
                state.audio_source_kind
                == "file"
            )

            assert not state.audio_stream_active
            assert state.audio_stream_token is None
            assert len(state.audio_pcm_buffer) == 0

    finally:

        cleanup_session(
            app_module,
            session_id,
        )


# ============================================================
# Canonical predictor delegation
# ============================================================

def test_run_canonical_prediction_delegates_exact_api(
    app_module: ModuleType,
):

    class DummyPredictor:

        def __init__(
            self,
        ):

            self.arguments = None

        def predict(
            self,
            **kwargs,
        ):

            self.arguments = kwargs

            return {
                "probabilities": {
                    "focused":
                        0.4,
                    "distracted":
                        0.2,
                    "fatigued":
                        0.2,
                    "overloaded":
                        0.2,
                }
            }

    dummy = DummyPredictor()

    original = (
        app_module.predictor
    )

    app_module.predictor = dummy

    try:

        keystroke = (
            ROOT_DIR
            / "dummy_keys.json"
        )

        audio = (
            ROOT_DIR
            / "dummy_audio.wav"
        )

        image = (
            ROOT_DIR
            / "dummy_image.jpg"
        )

        app_module.run_canonical_prediction(
            keystroke_json=(
                keystroke
            ),
            text=(
                "example text"
            ),
            audio_path=(
                audio
            ),
            image_path=(
                image
            ),
        )

        assert dummy.arguments == {
            "keystroke_json":
                keystroke,

            "text":
                "example text",

            "audio_path":
                audio,

            "image_path":
                image,
        }

    finally:

        app_module.predictor = (
            original
        )


# ============================================================
# Evaluation / training integration
# ============================================================

def test_evaluator_uses_canonical_temporal_engine():

    imported = (
        imported_names(
            EVALUATION_PATH,
            "temporal_fusion",
        )
    )

    assert (
        "TemporalFusionEngine"
        in imported
    )

    assert calls_name(
        EVALUATION_PATH,
        "TemporalFusionEngine",
    )


def test_multimodal_comparison_uses_canonical_temporal_engine():

    imported = (
        imported_names(
            COMPARISON_PATH,
            "temporal_fusion",
        )
    )

    assert (
        "TemporalFusionEngine"
        in imported
    )

    assert calls_name(
        COMPARISON_PATH,
        "TemporalFusionEngine",
    )


def test_multimodal_comparison_uses_group_aware_splitting():

    source = (
        COMPARISON_PATH
        .read_text(
            encoding="utf-8"
        )
    )

    assert (
        "StratifiedGroupKFold"
        in source
    )

    assert (
        "train_test_split"
        not in source
    )
