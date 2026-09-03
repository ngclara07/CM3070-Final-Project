# === web_app/app.py ===
# command to run: python -m uvicorn web_app.app:app --reload

from __future__ import annotations

import asyncio
import base64
import csv
import hashlib
import json
import math
import os
import statistics
import sys
import threading
import time
import uuid
import wave

from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import cv2
import librosa
import numpy as np
import uvicorn

from fastapi import (
    FastAPI,
    File,
    Form,
    HTTPException,
    Request,
    UploadFile,
    WebSocket,
    WebSocketDisconnect,
)

from fastapi.responses import (
    HTMLResponse,
    JSONResponse,
)

from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates


# ============================================================
# PATHS
# ============================================================

ROOT_DIR = (
    Path(__file__)
    .resolve()
    .parents[1]
)

WEB_DIR = (
    Path(__file__)
    .resolve()
    .parent
)

UPLOAD_DIR = (
    WEB_DIR
    / "uploads"
)

OUTPUT_DIR = (
    WEB_DIR
    / "output"
)

LOG_FILE = (
    OUTPUT_DIR
    / "live_predictions.csv"
)

UPLOAD_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

if str(ROOT_DIR) not in sys.path:
    sys.path.insert(
        0,
        str(ROOT_DIR),
    )

os.chdir(
    ROOT_DIR
)


# ============================================================
# CANONICAL INFERENCE IMPLEMENTATIONS
# ============================================================

from final_multimodal_inference import (  # noqa: E402
    FinalMultimodalInference,
)

from temporal_fusion import (  # noqa: E402
    LABELS,
    TEMPORAL_PROBABILITY_WINDOW,
    PROBABILITY_SUM_TOLERANCE,
    StaleGenerationError,
    TemporalFusionEngine,
    summarise_probability_dict,
    validate_probability_distribution,
)


# ============================================================
# CONFIGURATION
# ============================================================

MIN_TEXT_CHARS = 20
MIN_KEYPRESSES = 20

LIVE_INTERVAL_MS = 2500

TARGET_SR = 16000

# Continuous microphone server-side rolling window.
AUDIO_STREAM_WINDOW_SECONDS = 10.0

# Inference will not use microphone audio until at least this
# much continuous PCM has arrived.
AUDIO_STREAM_MIN_SECONDS = 2.0

# WebSocket diagnostic acknowledgement frequency.
AUDIO_STREAM_ACK_SECONDS = 0.40

# Maximum accepted binary WebSocket frame.
MAX_AUDIO_PACKET_BYTES = 1024 * 1024

NEAR_SILENCE_DBFS = -50.0
QUIET_AUDIO_DBFS = -35.0


# ============================================================
# GENERIC HELPERS
# ============================================================

def safe_mean(
    values: list[float],
) -> float:

    return (
        statistics.mean(values)
        if values
        else 0.0
    )


def safe_std(
    values: list[float],
) -> float:

    return (
        statistics.stdev(values)
        if len(values) >= 2
        else 0.0
    )


def safe_delete(
    path: Optional[Path],
) -> None:

    if path is None:
        return

    try:
        path.unlink(
            missing_ok=True
        )
    except Exception:
        pass


# ============================================================
# KEYSTROKE FEATURE CONSTRUCTION
# ============================================================

def build_live_keystroke_features(
    typed_text: str,
    events: list[
        dict[str, Any]
    ],
) -> dict[str, float]:

    downs = [
        event
        for event in events
        if event.get("type")
        == "down"
    ]

    down_times = [
        float(
            event["timestamp_perf"]
        )
        for event in downs
        if event.get(
            "timestamp_perf"
        )
        is not None
    ]

    if len(down_times) < 2:
        raise ValueError(
            "Not enough keystroke timing data."
        )

    keydown_count = len(downs)

    if keydown_count < MIN_KEYPRESSES:
        raise ValueError(
            f"At least {MIN_KEYPRESSES} "
            "key-down events are required."
        )

    delays = [
        down_times[index]
        - down_times[index - 1]
        for index
        in range(
            1,
            len(down_times),
        )
    ]

    hold_times: list[float] = []

    active_downs: dict[
        str,
        list[float],
    ] = {}

    for event in events:

        key = event.get("key")
        event_type = event.get("type")
        timestamp = event.get(
            "timestamp_perf"
        )

        if (
            key is None
            or timestamp is None
        ):
            continue

        timestamp = float(
            timestamp
        )

        if event_type == "down":

            active_downs.setdefault(
                str(key),
                [],
            ).append(
                timestamp
            )

        elif event_type == "up":

            queue = active_downs.get(
                str(key)
            )

            if queue:

                down_time = queue.pop(0)

                duration = (
                    timestamp
                    - down_time
                )

                if duration >= 0.0:
                    hold_times.append(
                        duration
                    )

    total_duration = (
        down_times[-1]
        - down_times[0]
    )

    word_count = len(
        typed_text.split()
    )

    correction_count = sum(
        1
        for event in downs
        if event.get("key")
        in {
            "backspace",
            "delete",
        }
    )

    pauses_1000 = [
        value
        for value in delays
        if value >= 1.0
    ]

    pauses_2000 = [
        value
        for value in delays
        if value >= 2.0
    ]

    pauses_5000 = [
        value
        for value in delays
        if value >= 5.0
    ]

    delay_mean = safe_mean(
        delays
    )

    delay_std = safe_std(
        delays
    )

    return {
        "total_duration_sec":
            round(
                total_duration,
                4,
            ),

        "keydown_count":
            keydown_count,

        "word_count":
            word_count,

        "typing_speed_kps":
            (
                round(
                    keydown_count
                    / total_duration,
                    4,
                )
                if total_duration > 0.0
                else 0.0
            ),

        "typing_speed_wpm":
            (
                round(
                    (
                        word_count
                        / total_duration
                    )
                    * 60.0,
                    4,
                )
                if total_duration > 0.0
                else 0.0
            ),

        "delay_mean":
            round(
                delay_mean,
                4,
            ),

        "delay_std":
            round(
                delay_std,
                4,
            ),

        "delay_min":
            (
                round(
                    min(delays),
                    4,
                )
                if delays
                else 0.0
            ),

        "delay_max":
            (
                round(
                    max(delays),
                    4,
                )
                if delays
                else 0.0
            ),

        "hold_mean":
            round(
                safe_mean(
                    hold_times
                ),
                4,
            ),

        "hold_std":
            round(
                safe_std(
                    hold_times
                ),
                4,
            ),

        "pause_count_1000":
            len(
                pauses_1000
            ),

        "pause_count_2000":
            len(
                pauses_2000
            ),

        "pause_count_5000":
            len(
                pauses_5000
            ),

        "pause_ratio_1000":
            (
                round(
                    len(pauses_1000)
                    / len(delays),
                    4,
                )
                if delays
                else 0.0
            ),

        "pause_ratio_2000":
            (
                round(
                    len(pauses_2000)
                    / len(delays),
                    4,
                )
                if delays
                else 0.0
            ),

        "mental_block_ratio_5000":
            (
                round(
                    len(pauses_5000)
                    / len(delays),
                    4,
                )
                if delays
                else 0.0
            ),

        "correction_count":
            correction_count,

        "correction_ratio":
            (
                round(
                    correction_count
                    / keydown_count,
                    4,
                )
                if keydown_count
                else 0.0
            ),

        "rhythm_consistency":
            (
                round(
                    1.0
                    / (
                        1.0
                        + delay_std
                    ),
                    4,
                )
                if delays
                else 1.0
            ),

        "burstiness_proxy":
            (
                round(
                    delay_std
                    / delay_mean,
                    4,
                )
                if delay_mean > 0.0
                else 0.0
            ),

        "fits_starts_index":
            (
                round(
                    len(pauses_1000)
                    / len(delays),
                    4,
                )
                if delays
                else 0.0
            ),
    }


# ============================================================
# AUDIO DIAGNOSTICS
# ============================================================

def classify_audio_energy(
    *,
    duration: float,
    rms: float,
    dbfs: float,
) -> dict[str, Any]:

    if dbfs <= NEAR_SILENCE_DBFS:

        condition = "near-silence"

        note = (
            "Valid quiet-environment audio input; "
            "it does not force the focused label."
        )

    elif dbfs <= QUIET_AUDIO_DBFS:

        condition = "quiet"

        note = (
            "Low-energy audio input."
        )

    else:

        condition = "active-audio"

        note = (
            "Audible signal detected."
        )

    return {
        "condition":
            condition,

        "duration_sec":
            float(duration),

        "rms":
            float(rms),

        "dbfs":
            float(dbfs),

        "note":
            note,
    }


def analyse_audio_file(
    path: Path,
) -> dict[str, Any]:

    try:

        waveform, sample_rate = (
            librosa.load(
                path,
                sr=TARGET_SR,
                mono=True,
                duration=20.0,
            )
        )

        waveform = np.asarray(
            waveform,
            dtype=np.float32,
        )

        if waveform.size == 0:

            return {
                "condition":
                    "empty",

                "duration_sec":
                    0.0,

                "rms":
                    0.0,

                "dbfs":
                    -120.0,

                "note":
                    "Audio contains no samples.",
            }

        duration = (
            waveform.size
            / sample_rate
        )

        rms = float(
            np.sqrt(
                np.mean(
                    np.square(
                        waveform
                    )
                )
            )
        )

        dbfs = (
            20.0
            * math.log10(
                max(
                    rms,
                    1e-12,
                )
            )
        )

        return classify_audio_energy(
            duration=duration,
            rms=rms,
            dbfs=dbfs,
        )

    except Exception as exc:

        return {
            "condition":
                "unknown",

            "duration_sec":
                None,

            "rms":
                None,

            "dbfs":
                None,

            "note":
                (
                    "Audio diagnostic failed: "
                    f"{exc}"
                ),
        }


def analyse_pcm16_bytes(
    pcm_bytes: bytes,
) -> dict[str, Any]:

    if not pcm_bytes:

        return {
            "condition":
                "empty",

            "duration_sec":
                0.0,

            "rms":
                0.0,

            "dbfs":
                -120.0,

            "note":
                "No streamed microphone samples available.",
        }

    usable_length = (
        len(pcm_bytes)
        - (
            len(pcm_bytes)
            % 2
        )
    )

    if usable_length <= 0:

        return {
            "condition":
                "empty",

            "duration_sec":
                0.0,

            "rms":
                0.0,

            "dbfs":
                -120.0,

            "note":
                "No complete PCM16 samples available.",
        }

    samples = (
        np.frombuffer(
            pcm_bytes[:usable_length],
            dtype="<i2",
        )
        .astype(
            np.float32
        )
        / 32768.0
    )

    duration = (
        samples.size
        / TARGET_SR
    )

    rms = float(
        np.sqrt(
            np.mean(
                np.square(
                    samples
                )
            )
        )
    )

    dbfs = (
        20.0
        * math.log10(
            max(
                rms,
                1e-12,
            )
        )
    )

    return classify_audio_energy(
        duration=duration,
        rms=rms,
        dbfs=dbfs,
    )


def write_pcm16_wav(
    *,
    path: Path,
    pcm_bytes: bytes,
) -> Path:

    usable_length = (
        len(pcm_bytes)
        - (
            len(pcm_bytes)
            % 2
        )
    )

    if usable_length <= 0:

        raise ValueError(
            "Cannot create WAV from empty PCM data."
        )

    with wave.open(
        str(path),
        "wb",
    ) as wav_file:

        wav_file.setnchannels(
            1
        )

        wav_file.setsampwidth(
            2
        )

        wav_file.setframerate(
            TARGET_SR
        )

        wav_file.writeframes(
            pcm_bytes[
                :usable_length
            ]
        )

    return path


# ============================================================
# SESSION STATE
# ============================================================

@dataclass
class SessionState:

    temporal_fusion: TemporalFusionEngine = (
        field(
            default_factory=(
                TemporalFusionEngine
            )
        )
    )

    last_seen: float = field(
        default_factory=time.time
    )

    # --------------------------------------------------------
    # Fixed audio-file mode
    # --------------------------------------------------------

    audio_path: Optional[
        Path
    ] = None

    audio_name: Optional[
        str
    ] = None

    audio_source_kind: Optional[
        str
    ] = None

    audio_diagnostics: dict[
        str,
        Any
    ] = field(
        default_factory=dict
    )

    # --------------------------------------------------------
    # Continuous microphone mode
    # --------------------------------------------------------

    audio_stream_active: bool = (
        False
    )

    audio_stream_token: Optional[
        str
    ] = None

    audio_pcm_buffer: bytearray = field(
        default_factory=bytearray
    )

    audio_stream_packets: int = 0

    audio_stream_last_packet_at: Optional[
        float
    ] = None

    # --------------------------------------------------------
    # Visual source
    # --------------------------------------------------------

    visual_mode: str = (
        "none"
    )

    visual_path: Optional[
        Path
    ] = None

    visual_name: Optional[
        str
    ] = None

    visual_started_at: Optional[
        float
    ] = None


SESSION_STATES: dict[
    str,
    SessionState
] = {}

SESSION_LOCK = (
    threading.RLock()
)

PREDICTOR_LOCK = (
    threading.Lock()
)

LOG_LOCK = (
    threading.Lock()
)


# ============================================================
# PREDICTOR
# ============================================================

predictor: Optional[
    FinalMultimodalInference
] = None


MODEL_STATUS: dict[
    str,
    Any
] = {

    "text_model":
        False,

    "audio_model":
        False,

    "image_model":
        False,

    "keystroke_model":
        False,

    "fusion_model":
        False,

    "webcam_calibrated_image_model":
        False,

    "inference_backend":
        (
            "final_multimodal_inference."
            "FinalMultimodalInference"
        ),

    "temporal_fusion_backend":
        (
            "temporal_fusion."
            "TemporalFusionEngine"
        ),

    "labels":
        list(LABELS),

    "fallback_enabled":
        False,

    "temporal_probability_window":
        TEMPORAL_PROBABILITY_WINDOW,

    "live_interval_ms":
        LIVE_INTERVAL_MS,

    "target_audio_sample_rate":
        TARGET_SR,

    "audio_stream_window_seconds":
        AUDIO_STREAM_WINDOW_SECONDS,

    "audio_stream_min_seconds":
        AUDIO_STREAM_MIN_SECONDS,

    "audio_stream_transport":
        "websocket_pcm16_mono",

    "audio_source_policy":
        (
            "fixed_file_or_continuous_"
            "microphone_stream"
        ),

    "min_text_chars":
        MIN_TEXT_CHARS,

    "min_keypresses":
        MIN_KEYPRESSES,

    "visual_source_modes": [
        "image",
        "video",
        "webcam",
    ],

    # Source changes reset temporal history.
    # Individual microphone PCM packets do NOT.
    "input_change_resets_temporal":
        True,

    "stream_packets_reset_temporal":
        False,

    "error":
        None,
}


def initialise_models() -> None:

    global predictor

    try:

        predictor = (
            FinalMultimodalInference()
        )

        MODEL_STATUS.update(
            {
                "text_model":
                    True,

                "audio_model":
                    True,

                "image_model":
                    True,

                "keystroke_model":
                    True,

                "fusion_model":
                    True,

                "webcam_calibrated_image_model":
                    (
                        predictor.webcam_image_model
                        is not None
                    ),

                "error":
                    None,
            }
        )

    except Exception as exc:

        predictor = None

        MODEL_STATUS.update(
            {
                "fusion_model":
                    False,

                "error":
                    str(exc),
            }
        )


# ============================================================
# SESSION HELPERS
# ============================================================

def validate_session_id(
    session_id: str,
) -> str:

    value = str(
        session_id
    ).strip()

    if not value:

        raise HTTPException(
            status_code=400,
            detail=(
                "session_id is required."
            ),
        )

    if len(value) > 200:

        raise HTTPException(
            status_code=400,
            detail=(
                "session_id is too long."
            ),
        )

    return value


def get_session(
    session_id: str,
) -> SessionState:

    state = SESSION_STATES.get(
        session_id
    )

    if state is None:

        state = SessionState()

        SESSION_STATES[
            session_id
        ] = state

    state.last_seen = (
        time.time()
    )

    return state


def session_directory(
    session_id: str,
) -> Path:

    token = (
        hashlib.sha256(
            session_id.encode(
                "utf-8"
            )
        )
        .hexdigest()[:24]
    )

    directory = (
        UPLOAD_DIR
        / token
    )

    directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    return directory


def reset_temporal_for_source_change(
    state: SessionState,
) -> int:

    generation = (
        state.temporal_fusion
        .reset()
    )

    state.last_seen = (
        time.time()
    )

    return generation


def clear_audio_stream_state(
    state: SessionState,
) -> None:

    state.audio_stream_active = (
        False
    )

    state.audio_stream_token = (
        None
    )

    state.audio_pcm_buffer.clear()

    state.audio_stream_packets = 0

    state.audio_stream_last_packet_at = (
        None
    )


def safe_suffix(
    filename: Optional[str],
    default: str,
) -> str:

    suffix = Path(
        filename or ""
    ).suffix.lower()

    if not suffix:
        return default

    if len(suffix) > 10:
        return default

    return suffix


async def save_upload(
    *,
    session_id: str,
    upload: UploadFile,
    prefix: str,
    default_suffix: str,
) -> Path:

    content = await upload.read()

    if not content:

        raise HTTPException(
            status_code=400,
            detail=(
                f"{prefix} upload is empty."
            ),
        )

    suffix = safe_suffix(
        upload.filename,
        default_suffix,
    )

    path = (
        session_directory(
            session_id
        )
        / (
            f"{prefix}_"
            f"{uuid.uuid4().hex}"
            f"{suffix}"
        )
    )

    path.write_bytes(
        content
    )

    return path


# ============================================================
# VISUAL CONVERSION
# ============================================================

def canonicalise_webcam_frame(
    image_frame: str,
    output_path: Path,
) -> Path:

    if not image_frame:

        raise ValueError(
            "Webcam frame is missing."
        )

    if "," not in image_frame:

        raise ValueError(
            "Invalid webcam frame data."
        )

    _header, encoded = (
        image_frame.split(
            ",",
            1,
        )
    )

    raw = base64.b64decode(
        encoded
    )

    array = np.frombuffer(
        raw,
        dtype=np.uint8,
    )

    frame = cv2.imdecode(
        array,
        cv2.IMREAD_COLOR,
    )

    if frame is None:

        raise ValueError(
            "Could not decode webcam frame."
        )

    success = cv2.imwrite(
        str(output_path),
        frame,
    )

    if not success:

        raise RuntimeError(
            "Could not save webcam snapshot."
        )

    return output_path


def extract_video_snapshot(
    *,
    video_path: Path,
    started_at: float,
    output_path: Path,
) -> Path:

    capture = cv2.VideoCapture(
        str(video_path)
    )

    if not capture.isOpened():

        capture.release()

        raise RuntimeError(
            "Could not open selected video."
        )

    try:

        fps = float(
            capture.get(
                cv2.CAP_PROP_FPS
            )
        )

        frame_count = float(
            capture.get(
                cv2.CAP_PROP_FRAME_COUNT
            )
        )

        if (
            not np.isfinite(fps)
            or fps <= 0.0
        ):
            fps = 30.0

        duration = (
            frame_count / fps
            if frame_count > 0.0
            else 0.0
        )

        elapsed = max(
            0.0,
            time.monotonic()
            - started_at,
        )

        if duration > 0.0:

            position_seconds = (
                elapsed % duration
            )

            capture.set(
                cv2.CAP_PROP_POS_MSEC,
                position_seconds
                * 1000.0,
            )

        success, frame = (
            capture.read()
        )

        if not success:

            capture.set(
                cv2.CAP_PROP_POS_FRAMES,
                0,
            )

            success, frame = (
                capture.read()
            )

        if not success:

            raise RuntimeError(
                "Could not read video frame."
            )

        success = cv2.imwrite(
            str(output_path),
            frame,
        )

        if not success:

            raise RuntimeError(
                "Could not save video snapshot."
            )

        return output_path

    finally:

        capture.release()


# ============================================================
# KEYSTROKE FILE HELPERS
# ============================================================

def parse_keystrokes(
    raw_events: str,
) -> list[dict[str, Any]]:

    try:

        parsed = json.loads(
            raw_events
        )

    except Exception:

        return []

    if not isinstance(
        parsed,
        list,
    ):
        return []

    return [
        event
        for event in parsed
        if isinstance(
            event,
            dict,
        )
    ]


def count_keydowns(
    events: list[
        dict[str, Any]
    ],
) -> int:

    return sum(
        1
        for event in events
        if event.get("type")
        == "down"
    )


def create_keystroke_json(
    *,
    session_id: str,
    text: str,
    events: list[
        dict[str, Any]
    ],
) -> Path:

    features = (
        build_live_keystroke_features(
            text,
            events,
        )
    )

    path = (
        session_directory(
            session_id
        )
        / (
            "keystrokes_"
            f"{uuid.uuid4().hex}"
            ".json"
        )
    )

    payload = {
        "features":
            features,

        "events":
            events,

        "typed_text":
            text,
    }

    path.write_text(
        json.dumps(
            payload,
            indent=2,
        ),
        encoding="utf-8",
    )

    return path


# ============================================================
# RAW MODEL INVOCATION
# ============================================================

def run_canonical_prediction(
    *,
    keystroke_json: Path,
    text: str,
    audio_path: Path,
    image_path: Path,
) -> dict[str, Any]:

    if predictor is None:

        raise RuntimeError(
            "Canonical fusion model is unavailable."
        )

    with PREDICTOR_LOCK:

        return predictor.predict(
            keystroke_json=(
                keystroke_json
            ),

            text=text,

            audio_path=(
                audio_path
            ),

            image_path=(
                image_path
            ),
        )


# ============================================================
# RESULT TRANSFORMATION
# ============================================================

def build_prediction_result(
    *,
    raw_result: dict[str, Any],
    temporal_engine:
        TemporalFusionEngine,
    expected_generation: int,
    audio_diagnostics:
        dict[str, Any],
    audio_source_kind:
        Optional[str],
    audio_buffered_seconds:
        Optional[float],
    visual_mode: str,
    visual_name:
        Optional[str],
) -> dict[str, Any]:

    raw_summary = (
        summarise_probability_dict(
            raw_result.get(
                "probabilities",
                {},
            ),
            labels=LABELS,
        )
    )

    raw_probabilities = (
        raw_summary[
            "probabilities"
        ]
    )

    temporal = (
        temporal_engine.append(
            raw_probabilities,
            expected_generation=(
                expected_generation
            ),
        )
    )

    raw_validation = (
        validate_probability_distribution(
            raw_probabilities,
            labels=LABELS,
            tolerance=(
                PROBABILITY_SUM_TOLERANCE
            ),
        )
    )

    temporal_validation = (
        validate_probability_distribution(
            temporal[
                "probabilities"
            ],
            labels=LABELS,
            tolerance=(
                PROBABILITY_SUM_TOLERANCE
            ),
        )
    )

    runtime_validation_pass = (
        raw_validation["valid"]
        and
        temporal_validation["valid"]
        and
        temporal["current_state"]
        in LABELS
    )

    image_calibration = (
        raw_result.get(
            "image_calibration"
        )
        or {}
    )

    webcam_prediction = None

    if image_calibration.get(
        "enabled"
    ):

        probability = (
            image_calibration.get(
                "top_probability"
            )
        )

        webcam_prediction = {
            "current_state":
                image_calibration.get(
                    "current_state"
                ),

            "confidence":
                probability,

            "confidence_percent":
                (
                    float(probability)
                    * 100.0
                    if probability
                    is not None
                    else None
                ),

            "confidence_gap":
                image_calibration.get(
                    "confidence_gap"
                ),

            "probabilities":
                image_calibration.get(
                    "probabilities"
                ),
        }

    return {
        "prediction":
            temporal[
                "current_state"
            ],

        "current_state":
            temporal[
                "current_state"
            ],

        "confidence":
            temporal[
                "confidence"
            ],

        "confidence_percent":
            temporal[
                "confidence_percent"
            ],

        "confidence_level":
            temporal[
                "confidence_level"
            ],

        "confidence_gap":
            temporal[
                "confidence_gap"
            ],

        "second_class":
            temporal[
                "second_class"
            ],

        "second_probability":
            temporal[
                "second_probability"
            ],

        "probabilities":
            temporal[
                "probabilities"
            ],

        "raw_prediction":
            raw_result.get(
                "prediction",
                raw_summary[
                    "current_state"
                ],
            ),

        "raw_top_class":
            raw_summary[
                "current_state"
            ],

        "raw_confidence":
            raw_summary[
                "confidence"
            ],

        "raw_confidence_percent":
            raw_summary[
                "confidence_percent"
            ],

        "raw_probabilities":
            raw_probabilities,

        "temporal_samples":
            temporal[
                "temporal_samples"
            ],

        "temporal_window":
            temporal[
                "temporal_window"
            ],

        "temporal_window_full":
            temporal[
                "temporal_window_full"
            ],

        "generation":
            temporal[
                "generation"
            ],

        "feature_dimension":
            raw_result.get(
                "feature_dimension"
            ),

        "device":
            raw_result.get(
                "device"
            ),

        "used_modalities":
            raw_result.get(
                "used_modalities",
                {},
            ),

        "visual_source_type":
            visual_mode,

        "visual_source_name":
            visual_name,

        "audio_source_kind":
            audio_source_kind,

        "audio_stream_buffered_seconds":
            audio_buffered_seconds,

        "webcam_calibration_used":
            webcam_prediction
            is not None,

        "webcam_prediction":
            webcam_prediction,

        "audio_diagnostics":
            audio_diagnostics,

        "runtime_validation": {
            "pass":
                runtime_validation_pass,

            "raw_probability_sum":
                raw_validation[
                    "probability_sum"
                ],

            "temporal_probability_sum":
                temporal_validation[
                    "probability_sum"
                ],

            "probability_ranges_valid":
                (
                    raw_validation[
                        "ranges_valid"
                    ]
                    and
                    temporal_validation[
                        "ranges_valid"
                    ]
                ),

            "temporal_window_full":
                temporal[
                    "temporal_window_full"
                ],
        },

        "behavioural_accuracy": {
            "status":
                "not_established",

            "reason":
                (
                    "Runtime inference proves pipeline "
                    "operation, not classifier accuracy."
                ),
        },
    }


# ============================================================
# LOGGING
# ============================================================

LOG_COLUMNS = [
    "timestamp",
    "session_id",
    "generation",
    "text_length",
    "keystroke_count",
    "audio_source",
    "audio_source_kind",
    "audio_condition",
    "audio_rms",
    "audio_dbfs",
    "audio_buffered_seconds",
    "visual_source_type",
    "visual_source_name",
    "raw_fusion_state",
    "raw_fusion_confidence",
    "raw_probabilities",
    "final_state",
    "final_confidence",
    "temporal_probabilities",
    "confidence_level",
    "confidence_gap",
    "temporal_samples",
    "temporal_window",
    "runtime_validation_pass",
    "feature_dimension",
    "used_modalities",
]


def initialise_log_file() -> None:

    if LOG_FILE.exists():

        try:

            with LOG_FILE.open(
                "r",
                encoding="utf-8",
            ) as file_handle:

                first_line = (
                    file_handle
                    .readline()
                    .strip()
                )

            expected = ",".join(
                LOG_COLUMNS
            )

            if first_line == expected:
                return

            backup = (
                LOG_FILE.with_name(
                    "live_predictions_backup_"
                    + datetime.now()
                    .strftime(
                        "%Y%m%d_%H%M%S"
                    )
                    + ".csv"
                )
            )

            LOG_FILE.rename(
                backup
            )

        except Exception:

            pass

    with LOG_FILE.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file_handle:

        csv.writer(
            file_handle
        ).writerow(
            LOG_COLUMNS
        )


def log_prediction(
    *,
    session_id: str,
    generation: int,
    text: str,
    keystroke_count: int,
    audio_name:
        Optional[str],
    result:
        dict[str, Any],
) -> None:

    audio = (
        result.get(
            "audio_diagnostics"
        )
        or {}
    )

    validation = (
        result.get(
            "runtime_validation"
        )
        or {}
    )

    row = [
        datetime.now(
            timezone.utc
        ).isoformat(),

        session_id,

        generation,

        len(text),

        keystroke_count,

        audio_name,

        result.get(
            "audio_source_kind"
        ),

        audio.get(
            "condition"
        ),

        audio.get(
            "rms"
        ),

        audio.get(
            "dbfs"
        ),

        result.get(
            "audio_stream_buffered_seconds"
        ),

        result.get(
            "visual_source_type"
        ),

        result.get(
            "visual_source_name"
        ),

        result.get(
            "raw_top_class"
        ),

        result.get(
            "raw_confidence"
        ),

        json.dumps(
            result.get(
                "raw_probabilities"
            )
        ),

        result.get(
            "current_state"
        ),

        result.get(
            "confidence"
        ),

        json.dumps(
            result.get(
                "probabilities"
            )
        ),

        result.get(
            "confidence_level"
        ),

        result.get(
            "confidence_gap"
        ),

        result.get(
            "temporal_samples"
        ),

        result.get(
            "temporal_window"
        ),

        validation.get(
            "pass"
        ),

        result.get(
            "feature_dimension"
        ),

        json.dumps(
            result.get(
                "used_modalities",
                {},
            )
        ),
    ]

    with LOG_LOCK:

        with LOG_FILE.open(
            "a",
            newline="",
            encoding="utf-8",
        ) as file_handle:

            csv.writer(
                file_handle
            ).writerow(
                row
            )


# ============================================================
# LIFESPAN
# ============================================================

@asynccontextmanager
async def lifespan(
    _app: FastAPI,
):

    initialise_log_file()

    initialise_models()

    yield


# ============================================================
# FASTAPI
# ============================================================

app = FastAPI(
    title=(
        "SenseFuzeAI Live Fusion Web App"
    ),
    lifespan=lifespan,
)


app.mount(
    "/static",
    StaticFiles(
        directory=(
            WEB_DIR
            / "static"
        )
    ),
    name="static",
)


templates = (
    Jinja2Templates(
        directory=(
            WEB_DIR
            / "templates"
        )
    )
)


# ============================================================
# BASIC ROUTES
# ============================================================

@app.get(
    "/",
    response_class=HTMLResponse,
)
def index(
    request: Request,
) -> HTMLResponse:

    return templates.TemplateResponse(
        request=request,
        name="index.html",
    )


@app.get(
    "/health"
)
def health() -> dict[str, Any]:

    return {
        "status":
            (
                "ok"
                if predictor
                is not None
                else "error"
            ),

        "timestamp":
            datetime.now(
                timezone.utc
            ).isoformat(),

        "temporal_probability_window":
            TEMPORAL_PROBABILITY_WINDOW,

        "live_interval_ms":
            LIVE_INTERVAL_MS,

        "audio_stream_window_seconds":
            AUDIO_STREAM_WINDOW_SECONDS,

        "audio_stream_min_seconds":
            AUDIO_STREAM_MIN_SECONDS,

        "target_audio_sample_rate":
            TARGET_SR,

        "temporal_fusion_backend":
            (
                "temporal_fusion."
                "TemporalFusionEngine"
            ),
    }


@app.get(
    "/model-status"
)
def model_status() -> dict[str, Any]:

    return dict(
        MODEL_STATUS
    )


# ============================================================
# FIXED AUDIO FILE
# ============================================================

@app.post(
    "/set_audio_source"
)
async def set_audio_source(

    session_id: str = Form(...),

    source_kind: str = Form(
        "file"
    ),

    audio_file: UploadFile = File(...),

) -> JSONResponse:

    session_id = validate_session_id(
        session_id
    )

    path = await save_upload(
        session_id=session_id,
        upload=audio_file,
        prefix="audio",
        default_suffix=".wav",
    )

    diagnostics = await asyncio.to_thread(
        analyse_audio_file,
        path,
    )

    old_path: Optional[Path] = None

    with SESSION_LOCK:

        state = get_session(
            session_id
        )

        old_path = (
            state.audio_path
        )

        generation = (
            reset_temporal_for_source_change(
                state
            )
        )

        clear_audio_stream_state(
            state
        )

        state.audio_path = path

        state.audio_name = (
            audio_file.filename
            or path.name
        )

        state.audio_source_kind = (
            "file"
        )

        state.audio_diagnostics = (
            diagnostics
        )

    if (
        old_path is not None
        and old_path != path
    ):
        safe_delete(
            old_path
        )

    return JSONResponse(
        {
            "status":
                "ok",

            "generation":
                generation,

            "audio_ready":
                True,

            "audio_name":
                (
                    audio_file.filename
                    or path.name
                ),

            "audio_source_kind":
                "file",

            "audio_diagnostics":
                diagnostics,

            "temporal_samples":
                0,

            "temporal_window":
                TEMPORAL_PROBABILITY_WINDOW,
        }
    )


# ============================================================
# START CONTINUOUS AUDIO STREAM
# ============================================================

@app.post(
    "/audio_stream/start"
)
async def start_audio_stream(

    session_id: str = Form(...),

) -> JSONResponse:

    session_id = validate_session_id(
        session_id
    )

    old_path: Optional[Path] = None

    with SESSION_LOCK:

        state = get_session(
            session_id
        )

        old_path = state.audio_path

        generation = (
            reset_temporal_for_source_change(
                state
            )
        )

        state.audio_path = None

        state.audio_name = (
            "Live microphone"
        )

        state.audio_source_kind = (
            "microphone_stream"
        )

        state.audio_diagnostics = {
            "condition":
                "buffering",

            "duration_sec":
                0.0,

            "rms":
                0.0,

            "dbfs":
                -120.0,

            "note":
                (
                    "Waiting for continuous "
                    "microphone samples."
                ),
        }

        clear_audio_stream_state(
            state
        )

        state.audio_stream_active = (
            True
        )

        state.audio_stream_token = (
            uuid.uuid4().hex
        )

        stream_token = (
            state.audio_stream_token
        )

    safe_delete(
        old_path
    )

    return JSONResponse(
        {
            "status":
                "ok",

            "generation":
                generation,

            "stream_token":
                stream_token,

            "audio_ready":
                False,

            "audio_source_kind":
                "microphone_stream",

            "target_sample_rate":
                TARGET_SR,

            "audio_stream_window_seconds":
                AUDIO_STREAM_WINDOW_SECONDS,

            "audio_stream_min_seconds":
                AUDIO_STREAM_MIN_SECONDS,

            "temporal_samples":
                0,

            "temporal_window":
                TEMPORAL_PROBABILITY_WINDOW,
        }
    )


# ============================================================
# STOP CONTINUOUS AUDIO STREAM
# ============================================================

@app.post(
    "/audio_stream/stop"
)
async def stop_audio_stream(

    session_id: str = Form(...),

) -> JSONResponse:

    session_id = validate_session_id(
        session_id
    )

    old_path: Optional[Path] = None

    with SESSION_LOCK:

        state = get_session(
            session_id
        )

        stream_was_active = (
            state.audio_stream_active
            or
            state.audio_source_kind
            == "microphone_stream"
        )

        if stream_was_active:

            generation = (
                reset_temporal_for_source_change(
                    state
                )
            )

        else:

            generation = (
                state.temporal_fusion
                .generation
            )

        old_path = state.audio_path

        state.audio_path = None
        state.audio_name = None
        state.audio_source_kind = None

        state.audio_diagnostics = {}

        clear_audio_stream_state(
            state
        )

    safe_delete(
        old_path
    )

    return JSONResponse(
        {
            "status":
                "ok",

            "generation":
                generation,

            "audio_ready":
                False,

            "audio_source_kind":
                None,

            "temporal_window":
                TEMPORAL_PROBABILITY_WINDOW,
        }
    )


# ============================================================
# CONTINUOUS PCM16 WEBSOCKET
# ============================================================

@app.websocket(
    "/ws/audio/{session_id}"
)
async def audio_stream_socket(
    websocket: WebSocket,
    session_id: str,
    token: str,
) -> None:

    await websocket.accept()

    try:

        session_id = (
            validate_session_id(
                session_id
            )
        )

    except HTTPException:

        await websocket.close(
            code=1008
        )

        return

    with SESSION_LOCK:

        state = get_session(
            session_id
        )

        if (
            not state.audio_stream_active
            or
            state.audio_stream_token
            != token
        ):

            valid = False

        else:

            valid = True

    if not valid:

        await websocket.send_json(
            {
                "type":
                    "error",

                "message":
                    "Invalid or expired audio stream token.",
            }
        )

        await websocket.close(
            code=1008
        )

        return

    max_buffer_bytes = int(
        AUDIO_STREAM_WINDOW_SECONDS
        * TARGET_SR
        * 2
    )

    minimum_buffer_bytes = int(
        AUDIO_STREAM_MIN_SECONDS
        * TARGET_SR
        * 2
    )

    last_ack = 0.0

    try:

        while True:

            message = (
                await websocket.receive()
            )

            if (
                message.get("type")
                == "websocket.disconnect"
            ):
                break

            chunk = message.get(
                "bytes"
            )

            if chunk is None:
                continue

            if len(chunk) == 0:
                continue

            if (
                len(chunk)
                > MAX_AUDIO_PACKET_BYTES
            ):

                await websocket.send_json(
                    {
                        "type":
                            "error",

                        "message":
                            (
                                "Audio packet exceeded "
                                "server size limit."
                            ),
                    }
                )

                continue

            # PCM16 must contain complete two-byte samples.
            if len(chunk) % 2 != 0:
                chunk = chunk[:-1]

            if not chunk:
                continue

            now = time.monotonic()

            with SESSION_LOCK:

                state = get_session(
                    session_id
                )

                if (
                    not state.audio_stream_active
                    or
                    state.audio_stream_token
                    != token
                ):

                    stream_valid = False

                    snapshot = b""

                    packets = (
                        state.audio_stream_packets
                    )

                else:

                    stream_valid = True

                    state.audio_pcm_buffer.extend(
                        chunk
                    )

                    overflow = (
                        len(
                            state.audio_pcm_buffer
                        )
                        - max_buffer_bytes
                    )

                    if overflow > 0:

                        # Preserve PCM16 sample alignment.
                        if overflow % 2 != 0:
                            overflow += 1

                        del (
                            state.audio_pcm_buffer[
                                :overflow
                            ]
                        )

                    state.audio_stream_packets += 1

                    state.audio_stream_last_packet_at = (
                        time.time()
                    )

                    state.last_seen = (
                        time.time()
                    )

                    snapshot = bytes(
                        state.audio_pcm_buffer
                    )

                    packets = (
                        state.audio_stream_packets
                    )

            if not stream_valid:

                await websocket.send_json(
                    {
                        "type":
                            "error",

                        "message":
                            "Audio stream is no longer active.",
                    }
                )

                await websocket.close(
                    code=1008
                )

                return

            should_ack = (
                now
                - last_ack
                >= AUDIO_STREAM_ACK_SECONDS
            )

            audio_ready = (
                len(snapshot)
                >= minimum_buffer_bytes
            )

            if should_ack:

                diagnostics = (
                    analyse_pcm16_bytes(
                        snapshot
                    )
                )

                buffered_seconds = (
                    len(snapshot)
                    / (
                        TARGET_SR
                        * 2
                    )
                )

                with SESSION_LOCK:

                    state = get_session(
                        session_id
                    )

                    if (
                        state.audio_stream_active
                        and
                        state.audio_stream_token
                        == token
                    ):

                        state.audio_diagnostics = (
                            diagnostics
                        )

                await websocket.send_json(
                    {
                        "type":
                            "audio_status",

                        "audio_ready":
                            audio_ready,

                        "buffered_seconds":
                            buffered_seconds,

                        "packets_received":
                            packets,

                        "audio_diagnostics":
                            diagnostics,
                    }
                )

                last_ack = now

    except WebSocketDisconnect:

        pass

    finally:

        # Unexpected transport loss means the microphone source
        # is no longer valid. Reset temporal history once so an
        # in-flight prediction cannot append after audio loss.
        with SESSION_LOCK:

            state = SESSION_STATES.get(
                session_id
            )

            if (
                state is not None
                and
                state.audio_stream_active
                and
                state.audio_stream_token
                == token
            ):

                state.temporal_fusion.reset()

                state.audio_name = None

                state.audio_source_kind = None

                state.audio_diagnostics = {}

                clear_audio_stream_state(
                    state
                )

                state.last_seen = (
                    time.time()
                )


# ============================================================
# IMAGE SOURCE
# ============================================================

@app.post(
    "/set_visual_image"
)
async def set_visual_image(

    session_id: str = Form(...),

    image_file: UploadFile = File(...),

) -> JSONResponse:

    session_id = validate_session_id(
        session_id
    )

    path = await save_upload(
        session_id=session_id,
        upload=image_file,
        prefix="image",
        default_suffix=".jpg",
    )

    image = cv2.imread(
        str(path),
        cv2.IMREAD_COLOR,
    )

    if image is None:

        safe_delete(path)

        raise HTTPException(
            status_code=400,
            detail=(
                "Uploaded image could not be decoded."
            ),
        )

    with SESSION_LOCK:

        state = get_session(
            session_id
        )

        generation = (
            reset_temporal_for_source_change(
                state
            )
        )

        state.visual_mode = "image"

        state.visual_path = path

        state.visual_name = (
            image_file.filename
            or path.name
        )

        state.visual_started_at = None

    return JSONResponse(
        {
            "status":
                "ok",

            "generation":
                generation,

            "visual_ready":
                True,

            "visual_mode":
                "image",

            "visual_name":
                (
                    image_file.filename
                    or path.name
                ),

            "temporal_samples":
                0,

            "temporal_window":
                TEMPORAL_PROBABILITY_WINDOW,
        }
    )


# ============================================================
# VIDEO SOURCE
# ============================================================

@app.post(
    "/set_visual_video"
)
async def set_visual_video(

    session_id: str = Form(...),

    video_file: UploadFile = File(...),

) -> JSONResponse:

    session_id = validate_session_id(
        session_id
    )

    path = await save_upload(
        session_id=session_id,
        upload=video_file,
        prefix="video",
        default_suffix=".mp4",
    )

    capture = cv2.VideoCapture(
        str(path)
    )

    opened = capture.isOpened()

    capture.release()

    if not opened:

        safe_delete(
            path
        )

        raise HTTPException(
            status_code=400,
            detail=(
                "Uploaded video could not be "
                "opened by OpenCV."
            ),
        )

    with SESSION_LOCK:

        state = get_session(
            session_id
        )

        generation = (
            reset_temporal_for_source_change(
                state
            )
        )

        state.visual_mode = "video"

        state.visual_path = path

        state.visual_name = (
            video_file.filename
            or path.name
        )

        state.visual_started_at = (
            time.monotonic()
        )

    return JSONResponse(
        {
            "status":
                "ok",

            "generation":
                generation,

            "visual_ready":
                True,

            "visual_mode":
                "video",

            "visual_name":
                (
                    video_file.filename
                    or path.name
                ),

            "temporal_samples":
                0,

            "temporal_window":
                TEMPORAL_PROBABILITY_WINDOW,
        }
    )


# ============================================================
# WEBCAM SOURCE
# ============================================================

@app.post(
    "/set_visual_webcam"
)
async def set_visual_webcam(

    session_id: str = Form(...),

) -> JSONResponse:

    session_id = validate_session_id(
        session_id
    )

    with SESSION_LOCK:

        state = get_session(
            session_id
        )

        generation = (
            reset_temporal_for_source_change(
                state
            )
        )

        state.visual_mode = "webcam"

        state.visual_path = None

        state.visual_name = "Webcam"

        state.visual_started_at = (
            time.monotonic()
        )

    return JSONResponse(
        {
            "status":
                "ok",

            "generation":
                generation,

            "visual_ready":
                True,

            "visual_mode":
                "webcam",

            "visual_name":
                "Webcam",

            "temporal_samples":
                0,

            "temporal_window":
                TEMPORAL_PROBABILITY_WINDOW,
        }
    )


# ============================================================
# STOP VISUAL
# ============================================================

@app.post(
    "/stop_visual"
)
async def stop_visual(

    session_id: str = Form(...),

) -> JSONResponse:

    session_id = validate_session_id(
        session_id
    )

    with SESSION_LOCK:

        state = get_session(
            session_id
        )

        if state.visual_mode in {
            "video",
            "webcam",
        }:

            state.visual_mode = "none"

            state.visual_path = None

            state.visual_name = None

            state.visual_started_at = None

        generation = (
            state.temporal_fusion
            .generation
        )

        visual_mode = (
            state.visual_mode
        )

    return JSONResponse(
        {
            "status":
                "ok",

            "generation":
                generation,

            "visual_mode":
                visual_mode,

            "visual_ready":
                (
                    visual_mode
                    == "image"
                ),
        }
    )


# ============================================================
# LIVE PREDICTION
# ============================================================

@app.post(
    "/predict_live"
)
async def predict_live(

    session_id: str = Form(...),

    generation: int = Form(...),

    text: str = Form(...),

    keystroke_events: str = Form(...),

    visual_mode: str = Form(...),

    webcam_frame:
        Optional[str] = Form(
            None
        ),

) -> JSONResponse:

    if predictor is None:

        raise HTTPException(
            status_code=503,
            detail=(
                "Canonical fusion backend "
                "is unavailable: "
                f"{MODEL_STATUS.get('error')}"
            ),
        )

    session_id = validate_session_id(
        session_id
    )

    text = text.strip()

    events = parse_keystrokes(
        keystroke_events
    )

    keydown_count = count_keydowns(
        events
    )

    if len(text) < MIN_TEXT_CHARS:

        raise HTTPException(
            status_code=409,
            detail=(
                f"Text not ready: "
                f"{len(text)}/"
                f"{MIN_TEXT_CHARS}."
            ),
        )

    if keydown_count < MIN_KEYPRESSES:

        raise HTTPException(
            status_code=409,
            detail=(
                f"Keystrokes not ready: "
                f"{keydown_count}/"
                f"{MIN_KEYPRESSES}."
            ),
        )

    # --------------------------------------------------------
    # Snapshot session state.
    #
    # Audio is copied into memory under the lock so that a
    # concurrent source change cannot delete/replace the data
    # while the model is reading it.
    # --------------------------------------------------------

    captured_audio_bytes: Optional[
        bytes
    ] = None

    captured_audio_pcm: Optional[
        bytes
    ] = None

    captured_audio_suffix = ".wav"

    with SESSION_LOCK:

        state = get_session(
            session_id
        )

        current_generation = (
            state.temporal_fusion
            .capture_generation()
        )

        if (
            int(generation)
            != current_generation
        ):

            raise HTTPException(
                status_code=409,
                detail={
                    "type":
                        "stale_generation",

                    "generation":
                        current_generation,
                },
            )

        if (
            visual_mode
            != state.visual_mode
        ):

            raise HTTPException(
                status_code=409,
                detail={
                    "type":
                        "visual_mode_mismatch",

                    "generation":
                        current_generation,

                    "visual_mode":
                        state.visual_mode,
                },
            )

        captured_generation = (
            current_generation
        )

        temporal_engine = (
            state.temporal_fusion
        )

        audio_name = state.audio_name

        audio_source_kind = (
            state.audio_source_kind
        )

        captured_visual_mode = (
            state.visual_mode
        )

        visual_path = state.visual_path

        visual_name = state.visual_name

        visual_started_at = (
            state.visual_started_at
        )

        # ----------------------------------------------------
        # Streaming microphone snapshot
        # ----------------------------------------------------

        if (
            audio_source_kind
            == "microphone_stream"
        ):

            if not state.audio_stream_active:

                raise HTTPException(
                    status_code=409,
                    detail=(
                        "Continuous microphone "
                        "stream is not active."
                    ),
                )

            captured_audio_pcm = bytes(
                state.audio_pcm_buffer
            )

            buffered_seconds = (
                len(
                    captured_audio_pcm
                )
                / (
                    TARGET_SR
                    * 2
                )
            )

            if (
                buffered_seconds
                < AUDIO_STREAM_MIN_SECONDS
            ):

                raise HTTPException(
                    status_code=409,
                    detail=(
                        "Continuous microphone "
                        "buffer is still warming up: "
                        f"{buffered_seconds:.2f}/"
                        f"{AUDIO_STREAM_MIN_SECONDS:.2f}s."
                    ),
                )

            audio_diagnostics = dict(
                state.audio_diagnostics
            )

        # ----------------------------------------------------
        # Fixed-file audio snapshot
        # ----------------------------------------------------

        elif (
            state.audio_path
            is not None
            and
            state.audio_path.exists()
        ):

            captured_audio_bytes = (
                state.audio_path
                .read_bytes()
            )

            captured_audio_suffix = (
                state.audio_path.suffix
                or ".wav"
            )

            buffered_seconds = None

            audio_diagnostics = dict(
                state.audio_diagnostics
            )

        else:

            raise HTTPException(
                status_code=409,
                detail=(
                    "Audio modality is required."
                ),
            )

    # --------------------------------------------------------
    # Visual gating
    # --------------------------------------------------------

    if (
        captured_visual_mode
        not in {
            "image",
            "video",
            "webcam",
        }
    ):

        raise HTTPException(
            status_code=409,
            detail=(
                "Visual modality is required."
            ),
        )

    keystroke_path: Optional[
        Path
    ] = None

    temporary_audio_path: Optional[
        Path
    ] = None

    temporary_image_path: Optional[
        Path
    ] = None

    try:

        # ----------------------------------------------------
        # Keystroke input
        # ----------------------------------------------------

        keystroke_path = (
            create_keystroke_json(
                session_id=session_id,
                text=text,
                events=events,
            )
        )

        # ----------------------------------------------------
        # Create immutable audio snapshot used by this
        # particular inference request.
        # ----------------------------------------------------

        if captured_audio_pcm is not None:

            temporary_audio_path = (
                session_directory(
                    session_id
                )
                / (
                    "live_audio_snapshot_"
                    f"{uuid.uuid4().hex}"
                    ".wav"
                )
            )

            await asyncio.to_thread(
                write_pcm16_wav,
                path=temporary_audio_path,
                pcm_bytes=captured_audio_pcm,
            )

            audio_path = (
                temporary_audio_path
            )

            audio_diagnostics = (
                analyse_pcm16_bytes(
                    captured_audio_pcm
                )
            )

        else:

            temporary_audio_path = (
                session_directory(
                    session_id
                )
                / (
                    "audio_snapshot_"
                    f"{uuid.uuid4().hex}"
                    f"{captured_audio_suffix}"
                )
            )

            temporary_audio_path.write_bytes(
                captured_audio_bytes
                or b""
            )

            audio_path = (
                temporary_audio_path
            )

        # ----------------------------------------------------
        # Static image
        # ----------------------------------------------------

        if (
            captured_visual_mode
            == "image"
        ):

            if (
                visual_path is None
                or
                not visual_path.exists()
            ):

                raise HTTPException(
                    status_code=409,
                    detail=(
                        "Selected image is unavailable."
                    ),
                )

            image_path = visual_path

        # ----------------------------------------------------
        # Video
        # ----------------------------------------------------

        elif (
            captured_visual_mode
            == "video"
        ):

            if (
                visual_path is None
                or
                not visual_path.exists()
                or
                visual_started_at
                is None
            ):

                raise HTTPException(
                    status_code=409,
                    detail=(
                        "Selected video is unavailable."
                    ),
                )

            temporary_image_path = (
                session_directory(
                    session_id
                )
                / (
                    "video_frame_"
                    f"{uuid.uuid4().hex}"
                    ".jpg"
                )
            )

            image_path = await asyncio.to_thread(
                extract_video_snapshot,
                video_path=visual_path,
                started_at=visual_started_at,
                output_path=(
                    temporary_image_path
                ),
            )

        # ----------------------------------------------------
        # Webcam
        # ----------------------------------------------------

        else:

            if not webcam_frame:

                raise HTTPException(
                    status_code=409,
                    detail=(
                        "Current webcam frame "
                        "is required."
                    ),
                )

            temporary_image_path = (
                session_directory(
                    session_id
                )
                / (
                    "webcam_frame_"
                    f"{uuid.uuid4().hex}"
                    ".jpg"
                )
            )

            image_path = await asyncio.to_thread(
                canonicalise_webcam_frame,
                webcam_frame,
                temporary_image_path,
            )

        # ----------------------------------------------------
        # Raw multimodal inference
        # ----------------------------------------------------

        raw_result = await asyncio.to_thread(
            run_canonical_prediction,
            keystroke_json=(
                keystroke_path
            ),
            text=text,
            audio_path=(
                audio_path
            ),
            image_path=(
                image_path
            ),
        )

        # ----------------------------------------------------
        # Temporal aggregation
        # ----------------------------------------------------

        try:

            result = (
                build_prediction_result(
                    raw_result=(
                        raw_result
                    ),

                    temporal_engine=(
                        temporal_engine
                    ),

                    expected_generation=(
                        captured_generation
                    ),

                    audio_diagnostics=(
                        audio_diagnostics
                    ),

                    audio_source_kind=(
                        audio_source_kind
                    ),

                    audio_buffered_seconds=(
                        buffered_seconds
                    ),

                    visual_mode=(
                        captured_visual_mode
                    ),

                    visual_name=(
                        visual_name
                    ),
                )
            )

        except StaleGenerationError as exc:

            with SESSION_LOCK:

                current_state = (
                    get_session(
                        session_id
                    )
                )

                current_generation = (
                    current_state
                    .temporal_fusion
                    .generation
                )

            raise HTTPException(
                status_code=409,
                detail={
                    "type":
                        "stale_result",

                    "generation":
                        current_generation,

                    "message":
                        str(exc),
                },
            ) from exc

        # ----------------------------------------------------
        # Confirm session/engine identity did not change
        # during inference.
        # ----------------------------------------------------

        with SESSION_LOCK:

            current_state = get_session(
                session_id
            )

            if (
                current_state.temporal_fusion
                is not temporal_engine
            ):

                raise HTTPException(
                    status_code=409,
                    detail={
                        "type":
                            "stale_session",

                        "generation":
                            (
                                current_state
                                .temporal_fusion
                                .generation
                            ),
                    },
                )

            current_state.last_seen = (
                time.time()
            )

        result[
            "session_id"
        ] = session_id

        result[
            "audio_source_name"
        ] = audio_name

        log_prediction(
            session_id=session_id,
            generation=(
                result[
                    "generation"
                ]
            ),
            text=text,
            keystroke_count=(
                keydown_count
            ),
            audio_name=(
                audio_name
            ),
            result=result,
        )

        return JSONResponse(
            result
        )

    except HTTPException:

        raise

    except Exception as exc:

        raise HTTPException(
            status_code=500,
            detail=(
                f"{type(exc).__name__}: "
                f"{exc}"
            ),
        ) from exc

    finally:

        safe_delete(
            keystroke_path
        )

        safe_delete(
            temporary_audio_path
        )

        safe_delete(
            temporary_image_path
        )


# ============================================================
# TEMPORAL RESET
#
# IMPORTANT:
# The microphone WebSocket stays alive.
# Only probability history / generation is reset.
# ============================================================

@app.post(
    "/reset_temporal"
)
async def reset_temporal(

    session_id: str = Form(...),

) -> JSONResponse:

    session_id = validate_session_id(
        session_id
    )

    with SESSION_LOCK:

        state = get_session(
            session_id
        )

        generation = (
            state.temporal_fusion
            .reset()
        )

        state.last_seen = (
            time.time()
        )

        temporal_status = (
            state.temporal_fusion
            .status()
        )

    return JSONResponse(
        {
            "status":
                "ok",

            "session_id":
                session_id,

            "generation":
                generation,

            "temporal_samples":
                temporal_status[
                    "temporal_samples"
                ],

            "temporal_window":
                temporal_status[
                    "temporal_window"
                ],

            "temporal_window_full":
                temporal_status[
                    "temporal_window_full"
                ],

            "message":
                (
                    "Temporal probability "
                    "history reset."
                ),
        }
    )


# ============================================================
# FULL RESET
# ============================================================

@app.post(
    "/full_reset"
)
async def full_reset(

    session_id: str = Form(...),

) -> JSONResponse:

    session_id = validate_session_id(
        session_id
    )

    old_audio_path: Optional[
        Path
    ] = None

    with SESSION_LOCK:

        state = get_session(
            session_id
        )

        generation = (
            state.temporal_fusion
            .reset()
        )

        old_audio_path = (
            state.audio_path
        )

        state.audio_path = None
        state.audio_name = None
        state.audio_source_kind = None
        state.audio_diagnostics = {}

        clear_audio_stream_state(
            state
        )

        state.visual_mode = "none"

        state.visual_path = None

        state.visual_name = None

        state.visual_started_at = None

        state.last_seen = (
            time.time()
        )

        temporal_status = (
            state.temporal_fusion
            .status()
        )

    safe_delete(
        old_audio_path
    )

    return JSONResponse(
        {
            "status":
                "ok",

            "session_id":
                session_id,

            "generation":
                generation,

            "temporal_samples":
                temporal_status[
                    "temporal_samples"
                ],

            "temporal_window":
                temporal_status[
                    "temporal_window"
                ],

            "temporal_window_full":
                temporal_status[
                    "temporal_window_full"
                ],

            "audio_ready":
                False,

            "audio_stream_active":
                False,

            "visual_ready":
                False,

            "visual_mode":
                "none",

            "message":
                "Full session reset.",
        }
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    uvicorn.run(
        "web_app.app:app",
        host="127.0.0.1",
        port=8000,
        reload=False,
    )
