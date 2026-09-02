# === web_app/app.py ===
#
# Local development:
#   python -m uvicorn web_app.app:app --reload
#
# Render production:
#   uvicorn web_app.app:app --host 0.0.0.0 --port $PORT
#
# IMPORTANT:
# Heavy ML models are intentionally loaded lazily.
# They are NOT loaded during FastAPI/Uvicorn startup.
# This allows Render to bind its HTTP port promptly.

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
import traceback
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
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates


# ============================================================
# PATHS
# ============================================================

ROOT_DIR = Path(__file__).resolve().parents[1]
WEB_DIR = Path(__file__).resolve().parent

UPLOAD_DIR = WEB_DIR / "uploads"
OUTPUT_DIR = WEB_DIR / "output"
LOG_FILE = OUTPUT_DIR / "live_predictions.csv"

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

# Compatibility with project modules using repository-relative paths.
os.chdir(ROOT_DIR)


# ============================================================
# TEMPORAL FUSION
#
# The heavy multimodal inference class is intentionally NOT
# imported at module scope.
# ============================================================

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

MIN_TEXT_CHARS = int(
    os.environ.get(
        "SENSEFUZE_MIN_TEXT_CHARS",
        "20",
    )
)

MIN_KEYPRESSES = int(
    os.environ.get(
        "SENSEFUZE_MIN_KEYPRESSES",
        "20",
    )
)

# 2.5 s was too aggressive for CPU MPNet + WavLM + CLIP.
LIVE_INTERVAL_MS = int(
    os.environ.get(
        "SENSEFUZE_LIVE_INTERVAL_MS",
        "15000",
    )
)

TARGET_SR = 16000

AUDIO_STREAM_WINDOW_SECONDS = float(
    os.environ.get(
        "SENSEFUZE_AUDIO_WINDOW_SECONDS",
        "10",
    )
)

AUDIO_STREAM_MIN_SECONDS = float(
    os.environ.get(
        "SENSEFUZE_AUDIO_MIN_SECONDS",
        "2",
    )
)

AUDIO_STREAM_ACK_SECONDS = 0.40

MAX_AUDIO_PACKET_BYTES = (
    1024 * 1024
)

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
        path.unlink(missing_ok=True)
    except Exception:
        pass


# ============================================================
# KEYSTROKE FEATURES
# ============================================================

def build_live_keystroke_features(
    typed_text: str,
    events: list[dict[str, Any]],
) -> dict[str, float]:

    downs = [
        event
        for event in events
        if event.get("type") == "down"
    ]

    down_times = [
        float(event["timestamp_perf"])
        for event in downs
        if event.get("timestamp_perf")
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
        in range(1, len(down_times))
    ]

    active_downs: dict[
        str,
        list[float],
    ] = {}

    hold_times: list[float] = []

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

        timestamp = float(timestamp)

        if event_type == "down":
            active_downs.setdefault(
                str(key),
                [],
            ).append(timestamp)

        elif event_type == "up":
            queue = active_downs.get(
                str(key)
            )

            if queue:
                down_time = queue.pop(0)
                duration = (
                    timestamp - down_time
                )

                if duration >= 0:
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

    delay_mean = safe_mean(delays)
    delay_std = safe_std(delays)

    return {
        "total_duration_sec":
            round(total_duration, 4),

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
                if total_duration > 0
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
                if total_duration > 0
                else 0.0
            ),

        "delay_mean":
            round(delay_mean, 4),

        "delay_std":
            round(delay_std, 4),

        "delay_min":
            (
                round(min(delays), 4)
                if delays
                else 0.0
            ),

        "delay_max":
            (
                round(max(delays), 4)
                if delays
                else 0.0
            ),

        "hold_mean":
            round(
                safe_mean(hold_times),
                4,
            ),

        "hold_std":
            round(
                safe_std(hold_times),
                4,
            ),

        "pause_count_1000":
            len(pauses_1000),

        "pause_count_2000":
            len(pauses_2000),

        "pause_count_5000":
            len(pauses_5000),

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
                    / (1.0 + delay_std),
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
                if delay_mean > 0
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
            "quietness alone does not determine "
            "the behavioural class."
        )

    elif dbfs <= QUIET_AUDIO_DBFS:
        condition = "quiet"
        note = "Low-energy audio input."

    else:
        condition = "active-audio"
        note = "Audible signal detected."

    return {
        "condition": condition,
        "duration_sec": float(duration),
        "rms": float(rms),
        "dbfs": float(dbfs),
        "note": note,
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
                "condition": "empty",
                "duration_sec": 0.0,
                "rms": 0.0,
                "dbfs": -120.0,
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
                    np.square(waveform)
                )
            )
        )

        dbfs = (
            20.0
            * math.log10(
                max(rms, 1e-12)
            )
        )

        return classify_audio_energy(
            duration=duration,
            rms=rms,
            dbfs=dbfs,
        )

    except Exception as exc:
        return {
            "condition": "unknown",
            "duration_sec": None,
            "rms": None,
            "dbfs": None,
            "note":
                (
                    "Audio diagnostic failed: "
                    f"{type(exc).__name__}: "
                    f"{exc}"
                ),
        }


def analyse_pcm16_bytes(
    pcm_bytes: bytes,
) -> dict[str, Any]:

    if not pcm_bytes:
        return {
            "condition": "empty",
            "duration_sec": 0.0,
            "rms": 0.0,
            "dbfs": -120.0,
            "note":
                (
                    "No streamed microphone "
                    "samples available."
                ),
        }

    usable_length = (
        len(pcm_bytes)
        - (len(pcm_bytes) % 2)
    )

    if usable_length <= 0:
        return {
            "condition": "empty",
            "duration_sec": 0.0,
            "rms": 0.0,
            "dbfs": -120.0,
            "note":
                (
                    "No complete PCM16 "
                    "samples available."
                ),
        }

    samples = (
        np.frombuffer(
            pcm_bytes[:usable_length],
            dtype="<i2",
        )
        .astype(np.float32)
        / 32768.0
    )

    duration = (
        samples.size / TARGET_SR
    )

    rms = float(
        np.sqrt(
            np.mean(
                np.square(samples)
            )
        )
    )

    dbfs = (
        20.0
        * math.log10(
            max(rms, 1e-12)
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
        - (len(pcm_bytes) % 2)
    )

    if usable_length <= 0:
        raise ValueError(
            "Cannot create WAV from empty PCM data."
        )

    with wave.open(
        str(path),
        "wb",
    ) as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(
            TARGET_SR
        )
        wav_file.writeframes(
            pcm_bytes[:usable_length]
        )

    return path


# ============================================================
# SESSION STATE
# ============================================================

@dataclass
class SessionState:

    temporal_fusion: TemporalFusionEngine = (
        field(
            default_factory=TemporalFusionEngine
        )
    )

    last_seen: float = field(
        default_factory=time.time
    )

    # Fixed audio file.
    audio_path: Optional[Path] = None
    audio_name: Optional[str] = None
    audio_source_kind: Optional[str] = None
    audio_diagnostics: dict[str, Any] = field(
        default_factory=dict
    )

    # Microphone configuration and transport are deliberately
    # represented separately.
    #
    # active     = source is logically configured
    # connected  = WebSocket is currently attached
    audio_stream_active: bool = False
    audio_stream_connected: bool = False

    audio_stream_token: Optional[str] = None

    audio_pcm_buffer: bytearray = field(
        default_factory=bytearray
    )

    audio_stream_packets: int = 0

    audio_stream_last_packet_at: Optional[
        float
    ] = None

    # Visual state.
    visual_mode: str = "none"
    visual_path: Optional[Path] = None
    visual_name: Optional[str] = None
    visual_started_at: Optional[float] = None


SESSION_STATES: dict[
    str,
    SessionState,
] = {}

SESSION_LOCK = threading.RLock()

MODEL_INIT_LOCK = threading.Lock()
PREDICTOR_LOCK = threading.Lock()
LOG_LOCK = threading.Lock()


# ============================================================
# MODEL STATE
# ============================================================

predictor: Optional[Any] = None

MODEL_STATUS: dict[str, Any] = {
    "state": "not_loaded",
    "initialised": False,
    "lazy_loading": True,

    "text_model": False,
    "audio_model": False,
    "image_model": False,
    "keystroke_model": False,
    "fusion_model": False,

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

    "min_text_chars":
        MIN_TEXT_CHARS,

    "min_keypresses":
        MIN_KEYPRESSES,

    "visual_source_modes": [
        "image",
        "video",
        "webcam",
    ],

    "error": None,

    "_traceback": None,
}


def initialise_models() -> None:
    global predictor

    if predictor is not None:
        return

    with MODEL_INIT_LOCK:

        if predictor is not None:
            return

        MODEL_STATUS.update(
            {
                "state": "loading",
                "initialised": False,
                "error": None,
                "_traceback": None,
            }
        )

        try:
            print(
                "[models] Importing inference backend...",
                flush=True,
            )

            from final_multimodal_inference import (
                FinalMultimodalInference,
            )

            print(
                "[models] Constructing fusion predictor...",
                flush=True,
            )

            instance = (
                FinalMultimodalInference()
            )

            predictor = instance

            runtime = {}

            try:
                runtime = (
                    instance.runtime_status()
                )
            except Exception:
                runtime = {}

            MODEL_STATUS.update(
                {
                    "state": "ready",
                    "initialised": True,
                    "fusion_model": True,
                    "keystroke_model": True,

                    "text_model":
                        bool(
                            runtime.get(
                                "text_model_loaded",
                                False,
                            )
                        ),

                    "audio_model":
                        bool(
                            runtime.get(
                                "audio_model_loaded",
                                False,
                            )
                        ),

                    "image_model":
                        bool(
                            runtime.get(
                                "image_model_loaded",
                                False,
                            )
                        ),

                    "webcam_calibrated_image_model":
                        bool(
                            getattr(
                                instance,
                                "webcam_image_model",
                                None,
                            )
                            is not None
                        ),

                    "error": None,
                    "_traceback": None,
                }
            )

            print(
                "[models] Fusion predictor ready.",
                flush=True,
            )

        except Exception as exc:
            predictor = None

            tb = traceback.format_exc()

            MODEL_STATUS.update(
                {
                    "state": "failed",
                    "initialised": False,
                    "fusion_model": False,
                    "keystroke_model": False,
                    "text_model": False,
                    "audio_model": False,
                    "image_model": False,
                    "webcam_calibrated_image_model":
                        False,

                    "error":
                        (
                            f"{type(exc).__name__}: "
                            f"{exc}"
                        ),

                    "_traceback": tb,
                }
            )

            print(
                "\n"
                "========================================\n"
                "MODEL INITIALISATION FAILED\n"
                "========================================\n"
                f"{tb}"
                "========================================",
                flush=True,
            )

            raise


def public_model_status() -> dict[str, Any]:

    result = {
        key: value
        for key, value
        in MODEL_STATUS.items()
        if not key.startswith("_")
    }

    result["predictor_loaded"] = (
        predictor is not None
    )

    if predictor is not None:
        try:
            runtime = (
                predictor.runtime_status()
            )

            result["runtime"] = (
                runtime
            )

            result["text_model"] = bool(
                runtime.get(
                    "text_model_loaded",
                    result["text_model"],
                )
            )

            result["audio_model"] = bool(
                runtime.get(
                    "audio_model_loaded",
                    result["audio_model"],
                )
            )

            result["image_model"] = bool(
                runtime.get(
                    "image_model_loaded",
                    result["image_model"],
                )
            )

        except Exception:
            pass

    return result


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
            detail="session_id is required.",
        )

    if len(value) > 200:
        raise HTTPException(
            status_code=400,
            detail="session_id is too long.",
        )

    return value


def get_session(
    session_id: str,
) -> SessionState:

    state = (
        SESSION_STATES.get(
            session_id
        )
    )

    if state is None:
        state = SessionState()
        SESSION_STATES[
            session_id
        ] = state

    state.last_seen = time.time()

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
        UPLOAD_DIR / token
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
        state.temporal_fusion.reset()
    )

    state.last_seen = time.time()

    return generation


def clear_audio_transport(
    state: SessionState,
    *,
    clear_buffer: bool,
) -> None:
    """
    Clear WebSocket transport state without necessarily removing
    the logical microphone source.
    """

    state.audio_stream_connected = False
    state.audio_stream_token = None

    if clear_buffer:
        state.audio_pcm_buffer.clear()
        state.audio_stream_packets = 0
        state.audio_stream_last_packet_at = None


def clear_audio_source(
    state: SessionState,
) -> None:

    state.audio_path = None
    state.audio_name = None
    state.audio_source_kind = None
    state.audio_diagnostics = {}

    state.audio_stream_active = False

    clear_audio_transport(
        state,
        clear_buffer=True,
    )


def safe_suffix(
    filename: Optional[str],
    default: str,
) -> str:

    suffix = Path(
        filename or ""
    ).suffix.lower()

    if (
        not suffix
        or len(suffix) > 10
    ):
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

    path.write_bytes(content)

    return path


# ============================================================
# VISUAL HELPERS
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

    _, encoded = (
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

    if not cv2.imwrite(
        str(output_path),
        frame,
    ):
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
            or fps <= 0
        ):
            fps = 30.0

        duration = (
            frame_count / fps
            if frame_count > 0
            else 0.0
        )

        elapsed = max(
            0.0,
            (
                time.monotonic()
                - started_at
            ),
        )

        if duration > 0:
            capture.set(
                cv2.CAP_PROP_POS_MSEC,
                (
                    elapsed % duration
                )
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

        if not cv2.imwrite(
            str(output_path),
            frame,
        ):
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
        item
        for item in parsed
        if isinstance(item, dict)
    ]


def count_keydowns(
    events: list[dict[str, Any]],
) -> int:

    return sum(
        1
        for event in events
        if event.get("type") == "down"
    )


def create_keystroke_json(
    *,
    session_id: str,
    text: str,
    events: list[dict[str, Any]],
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

    path.write_text(
        json.dumps(
            {
                "features": features,
                "events": events,
                "typed_text": text,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    return path


# ============================================================
# RAW PREDICTION
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
            keystroke_json=keystroke_json,
            text=text,
            audio_path=audio_path,
            image_path=image_path,
        )


# ============================================================
# RESULT TRANSFORMATION
# ============================================================

def build_prediction_result(
    *,
    raw_result: dict[str, Any],
    temporal_engine: TemporalFusionEngine,
    expected_generation: int,
    audio_diagnostics: dict[str, Any],
    audio_source_kind: Optional[str],
    audio_buffered_seconds: Optional[float],
    visual_mode: str,
    visual_name: Optional[str],
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
        raw_summary["probabilities"]
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
            temporal["probabilities"],
            labels=LABELS,
            tolerance=(
                PROBABILITY_SUM_TOLERANCE
            ),
        )
    )

    runtime_validation_pass = (
        raw_validation["valid"]
        and temporal_validation["valid"]
        and temporal["current_state"]
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
            temporal["confidence"],

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
                    "Runtime inference demonstrates "
                    "pipeline operation; it does not "
                    "establish classifier accuracy."
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
            ) as handle:
                first_line = (
                    handle.readline()
                    .strip()
                )

            if first_line == ",".join(
                LOG_COLUMNS
            ):
                return

        except Exception:
            pass

    with LOG_FILE.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        csv.writer(
            handle
        ).writerow(
            LOG_COLUMNS
        )


def log_prediction(
    *,
    session_id: str,
    generation: int,
    text: str,
    keystroke_count: int,
    audio_name: Optional[str],
    result: dict[str, Any],
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

        audio.get("condition"),
        audio.get("rms"),
        audio.get("dbfs"),

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

        validation.get("pass"),

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
        ) as handle:
            csv.writer(
                handle
            ).writerow(row)


# ============================================================
# FASTAPI LIFESPAN
# ============================================================

@asynccontextmanager
async def lifespan(
    _app: FastAPI,
):

    print(
        "[startup] Initialising web application...",
        flush=True,
    )

    try:
        initialise_log_file()

        print(
            "[startup] Prediction log ready.",
            flush=True,
        )

    except Exception as exc:
        print(
            "[startup] Log setup warning: "
            f"{type(exc).__name__}: "
            f"{exc}",
            flush=True,
        )

    # Do not initialise heavy ML models here.
    print(
        "[startup] Web application ready. "
        "Fusion backend is lazy.",
        flush=True,
    )

    yield

    print(
        "[shutdown] Web application shutting down.",
        flush=True,
    )


# ============================================================
# APPLICATION
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
        directory=str(
            WEB_DIR / "static"
        )
    ),
    name="static",
)

templates = Jinja2Templates(
    directory=str(
        WEB_DIR / "templates"
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


@app.get("/health")
def health() -> dict[str, Any]:

    return {
        "status": "ok",
        "service": "web",
        "model_state":
            MODEL_STATUS["state"],

        "predictor_loaded":
            predictor is not None,

        "timestamp":
            datetime.now(
                timezone.utc
            ).isoformat(),
    }


@app.get("/model-status")
def model_status() -> dict[str, Any]:

    return public_model_status()


@app.post("/initialize-models")
async def initialize_models_endpoint() -> JSONResponse:

    if predictor is not None:
        return JSONResponse(
            {
                "status": "ready",
                "model_status":
                    public_model_status(),
            }
        )

    try:
        await asyncio.to_thread(
            initialise_models
        )

        return JSONResponse(
            {
                "status": "ready",
                "model_status":
                    public_model_status(),
            }
        )

    except Exception as exc:
        return JSONResponse(
            {
                "status": "failed",

                "exception_type":
                    type(exc).__name__,

                "exception":
                    str(exc),

                "model_status":
                    public_model_status(),
            },
            status_code=500,
        )


# ============================================================
# SESSION STATUS
#
# Used by browser recovery after connection/process disruption.
# ============================================================

@app.get("/session-status/{session_id}")
def session_status(
    session_id: str,
) -> dict[str, Any]:

    session_id = validate_session_id(
        session_id
    )

    with SESSION_LOCK:
        state = get_session(
            session_id
        )

        buffered_seconds = (
            len(
                state.audio_pcm_buffer
            )
            / (TARGET_SR * 2)
        )

        return {
            "status": "ok",

            "generation":
                state.temporal_fusion
                .generation,

            "audio_source_kind":
                state.audio_source_kind,

            "audio_stream_active":
                state.audio_stream_active,

            "audio_stream_connected":
                state.audio_stream_connected,

            "audio_buffered_seconds":
                buffered_seconds,

            "audio_ready":
                (
                    state.audio_source_kind
                    == "file"
                    or
                    (
                        state.audio_source_kind
                        == "microphone_stream"
                        and
                        buffered_seconds
                        >=
                        AUDIO_STREAM_MIN_SECONDS
                    )
                ),

            "visual_mode":
                state.visual_mode,

            "visual_name":
                state.visual_name,
        }


# ============================================================
# FIXED AUDIO FILE
# ============================================================

@app.post("/set_audio_source")
async def set_audio_source(
    session_id: str = Form(...),
    source_kind: str = Form("file"),
    audio_file: UploadFile = File(...),
) -> JSONResponse:

    del source_kind

    session_id = validate_session_id(
        session_id
    )

    path = await save_upload(
        session_id=session_id,
        upload=audio_file,
        prefix="audio",
        default_suffix=".wav",
    )

    diagnostics = (
        await asyncio.to_thread(
            analyse_audio_file,
            path,
        )
    )

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

        # Explicit switch to fixed-file audio.
        state.audio_stream_active = False

        clear_audio_transport(
            state,
            clear_buffer=True,
        )

        state.audio_path = path

        state.audio_name = (
            audio_file.filename
            or path.name
        )

        state.audio_source_kind = "file"

        state.audio_diagnostics = (
            diagnostics
        )

    if old_path != path:
        safe_delete(old_path)

    return JSONResponse(
        {
            "status": "ok",
            "generation": generation,
            "audio_ready": True,

            "audio_name":
                audio_file.filename
                or path.name,

            "audio_source_kind":
                "file",

            "audio_diagnostics":
                diagnostics,

            "temporal_samples": 0,

            "temporal_window":
                TEMPORAL_PROBABILITY_WINDOW,
        }
    )


# ============================================================
# MICROPHONE CONFIGURATION
# ============================================================

def configure_microphone_stream(
    state: SessionState,
    *,
    reset_generation: bool,
) -> tuple[int, str]:

    if reset_generation:
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

    old_audio_path = (
        state.audio_path
    )

    state.audio_path = None

    state.audio_name = (
        "Live microphone"
    )

    state.audio_source_kind = (
        "microphone_stream"
    )

    state.audio_stream_active = True
    state.audio_stream_connected = False

    # A new token invalidates an older socket without
    # destroying the logical microphone source.
    state.audio_stream_token = (
        uuid.uuid4().hex
    )

    state.audio_pcm_buffer.clear()
    state.audio_stream_packets = 0
    state.audio_stream_last_packet_at = None

    state.audio_diagnostics = {
        "condition": "buffering",
        "duration_sec": 0.0,
        "rms": 0.0,
        "dbfs": -120.0,
        "note":
            (
                "Waiting for continuous "
                "microphone samples."
            ),
    }

    safe_delete(old_audio_path)

    return (
        generation,
        state.audio_stream_token,
    )


@app.post("/audio_stream/start")
async def start_audio_stream(
    session_id: str = Form(...),
) -> JSONResponse:

    session_id = validate_session_id(
        session_id
    )

    with SESSION_LOCK:
        state = get_session(
            session_id
        )

        generation, token = (
            configure_microphone_stream(
                state,
                reset_generation=True,
            )
        )

    return JSONResponse(
        {
            "status": "ok",
            "generation": generation,
            "stream_token": token,
            "audio_ready": False,

            "audio_source_kind":
                "microphone_stream",

            "target_sample_rate":
                TARGET_SR,

            "audio_stream_window_seconds":
                AUDIO_STREAM_WINDOW_SECONDS,

            "audio_stream_min_seconds":
                AUDIO_STREAM_MIN_SECONDS,

            "temporal_samples": 0,

            "temporal_window":
                TEMPORAL_PROBABILITY_WINDOW,
        }
    )


# ============================================================
# MICROPHONE RECONNECT
#
# This route is intentionally non-destructive.
#
# If the server still has the session, generation is preserved.
# If the server process restarted and lost the session, the
# microphone source is reconstructed and the returned generation
# becomes authoritative for the browser.
# ============================================================

@app.post("/audio_stream/reconnect")
async def reconnect_audio_stream(
    session_id: str = Form(...),
) -> JSONResponse:

    session_id = validate_session_id(
        session_id
    )

    with SESSION_LOCK:
        state = get_session(
            session_id
        )

        already_microphone = (
            state.audio_source_kind
            == "microphone_stream"
            and
            state.audio_stream_active
        )

        generation, token = (
            configure_microphone_stream(
                state,
                reset_generation=(
                    not already_microphone
                ),
            )
        )

    return JSONResponse(
        {
            "status": "ok",

            "generation":
                generation,

            "stream_token":
                token,

            "audio_ready":
                False,

            "audio_source_kind":
                "microphone_stream",

            "recovered":
                True,

            "target_sample_rate":
                TARGET_SR,

            "audio_stream_window_seconds":
                AUDIO_STREAM_WINDOW_SECONDS,

            "audio_stream_min_seconds":
                AUDIO_STREAM_MIN_SECONDS,
        }
    )


# ============================================================
# EXPLICIT MICROPHONE STOP
# ============================================================

@app.post("/audio_stream/stop")
async def stop_audio_stream(
    session_id: str = Form(...),
) -> JSONResponse:

    session_id = validate_session_id(
        session_id
    )

    with SESSION_LOCK:
        state = get_session(
            session_id
        )

        had_audio = (
            state.audio_source_kind
            is not None
        )

        generation = (
            reset_temporal_for_source_change(
                state
            )
            if had_audio
            else
            state.temporal_fusion
            .generation
        )

        old_path = state.audio_path

        clear_audio_source(
            state
        )

    safe_delete(old_path)

    return JSONResponse(
        {
            "status": "ok",

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
# MICROPHONE WEBSOCKET
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

        valid = (
            state.audio_stream_active
            and
            state.audio_source_kind
            == "microphone_stream"
            and
            state.audio_stream_token
            == token
        )

        if valid:
            state.audio_stream_connected = (
                True
            )

    if not valid:
        await websocket.send_json(
            {
                "type": "error",

                "message":
                    (
                        "Invalid or expired "
                        "audio stream token."
                    ),
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

            if not chunk:
                continue

            if (
                len(chunk)
                > MAX_AUDIO_PACKET_BYTES
            ):
                await websocket.send_json(
                    {
                        "type": "error",

                        "message":
                            (
                                "Audio packet exceeded "
                                "server size limit."
                            ),
                    }
                )

                continue

            if len(chunk) % 2:
                chunk = chunk[:-1]

            if not chunk:
                continue

            now = time.monotonic()

            with SESSION_LOCK:
                state = get_session(
                    session_id
                )

                stream_valid = (
                    state.audio_stream_active
                    and
                    state.audio_source_kind
                    == "microphone_stream"
                    and
                    state.audio_stream_token
                    == token
                )

                if not stream_valid:
                    snapshot = b""

                    packets = (
                        state.audio_stream_packets
                    )

                else:
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
                        if overflow % 2:
                            overflow += 1

                        del state.audio_pcm_buffer[
                            :overflow
                        ]

                    state.audio_stream_packets += 1

                    state.audio_stream_last_packet_at = (
                        time.time()
                    )

                    state.audio_stream_connected = (
                        True
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
                try:
                    await websocket.send_json(
                        {
                            "type": "error",

                            "message":
                                (
                                    "Audio stream token "
                                    "is no longer current."
                                ),
                        }
                    )
                except Exception:
                    pass

                await websocket.close(
                    code=1008
                )

                return

            if (
                now - last_ack
                >=
                AUDIO_STREAM_ACK_SECONDS
            ):
                diagnostics = (
                    analyse_pcm16_bytes(
                        snapshot
                    )
                )

                buffered_seconds = (
                    len(snapshot)
                    / (TARGET_SR * 2)
                )

                audio_ready = (
                    len(snapshot)
                    >= minimum_buffer_bytes
                )

                with SESSION_LOCK:
                    state = get_session(
                        session_id
                    )

                    if (
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

    except Exception as exc:
        print(
            "[audio-ws] "
            f"{type(exc).__name__}: "
            f"{exc}",
            flush=True,
        )

    finally:
        # CRITICAL FIX:
        #
        # An unexpected transport disconnect does NOT destroy
        # the logical microphone source, temporal generation,
        # visual source, or session configuration.
        #
        # It only marks this specific socket as disconnected.
        with SESSION_LOCK:
            state = SESSION_STATES.get(
                session_id
            )

            if (
                state is not None
                and
                state.audio_stream_token
                == token
            ):
                state.audio_stream_connected = (
                    False
                )

                state.last_seen = (
                    time.time()
                )


# ============================================================
# VISUAL SOURCES
# ============================================================

@app.post("/set_visual_image")
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

    if (
        cv2.imread(
            str(path),
            cv2.IMREAD_COLOR,
        )
        is None
    ):
        safe_delete(path)

        raise HTTPException(
            status_code=400,
            detail=(
                "Uploaded image could "
                "not be decoded."
            ),
        )

    with SESSION_LOCK:
        state = get_session(
            session_id
        )

        old_path = (
            state.visual_path
            if state.visual_mode
            in {"image", "video"}
            else None
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

    if (
        old_path is not None
        and old_path != path
    ):
        safe_delete(old_path)

    return JSONResponse(
        {
            "status": "ok",
            "generation": generation,
            "visual_ready": True,
            "visual_mode": "image",

            "visual_name":
                image_file.filename
                or path.name,

            "temporal_samples": 0,

            "temporal_window":
                TEMPORAL_PROBABILITY_WINDOW,
        }
    )


@app.post("/set_visual_video")
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
        safe_delete(path)

        raise HTTPException(
            status_code=400,
            detail=(
                "Uploaded video could "
                "not be opened."
            ),
        )

    with SESSION_LOCK:
        state = get_session(
            session_id
        )

        old_path = (
            state.visual_path
            if state.visual_mode
            in {"image", "video"}
            else None
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

    if (
        old_path is not None
        and old_path != path
    ):
        safe_delete(old_path)

    return JSONResponse(
        {
            "status": "ok",
            "generation": generation,
            "visual_ready": True,
            "visual_mode": "video",

            "visual_name":
                video_file.filename
                or path.name,

            "temporal_samples": 0,

            "temporal_window":
                TEMPORAL_PROBABILITY_WINDOW,
        }
    )


@app.post("/set_visual_webcam")
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

        same_mode = (
            state.visual_mode
            == "webcam"
        )

        # Recovery/re-registration of the same webcam source
        # does not unnecessarily reset temporal history.
        generation = (
            state.temporal_fusion
            .generation
            if same_mode
            else
            reset_temporal_for_source_change(
                state
            )
        )

        state.visual_mode = "webcam"
        state.visual_path = None
        state.visual_name = "Webcam"

        if (
            state.visual_started_at
            is None
        ):
            state.visual_started_at = (
                time.monotonic()
            )

    return JSONResponse(
        {
            "status": "ok",
            "generation": generation,
            "visual_ready": True,
            "visual_mode": "webcam",
            "visual_name": "Webcam",

            "temporal_samples":
                (
                    state.temporal_fusion
                    .status()[
                        "temporal_samples"
                    ]
                ),

            "temporal_window":
                TEMPORAL_PROBABILITY_WINDOW,
        }
    )


@app.post("/stop_visual")
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

        changed = (
            state.visual_mode
            != "none"
        )

        old_path = (
            state.visual_path
            if state.visual_mode
            in {"image", "video"}
            else None
        )

        generation = (
            reset_temporal_for_source_change(
                state
            )
            if changed
            else
            state.temporal_fusion
            .generation
        )

        state.visual_mode = "none"
        state.visual_path = None
        state.visual_name = None
        state.visual_started_at = None

    safe_delete(old_path)

    return JSONResponse(
        {
            "status": "ok",
            "generation": generation,
            "visual_mode": "none",
            "visual_ready": False,
        }
    )


# ============================================================
# LIVE PREDICTION
# ============================================================

@app.post("/predict_live")
async def predict_live(
    session_id: str = Form(...),
    generation: int = Form(...),
    text: str = Form(...),
    keystroke_events: str = Form(...),
    visual_mode: str = Form(...),
    webcam_frame: Optional[str] = Form(None),
) -> JSONResponse:

    if predictor is None:
        try:
            await asyncio.to_thread(
                initialise_models
            )

        except Exception as exc:
            raise HTTPException(
                status_code=503,
                detail=(
                    "Canonical fusion backend "
                    "could not initialise: "
                    f"{type(exc).__name__}: "
                    f"{exc}"
                ),
            ) from exc

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

        # Generation is authoritative on the server.
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

                    "visual_mode":
                        state.visual_mode,

                    "audio_source_kind":
                        state.audio_source_kind,
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

                    "requested_visual_mode":
                        visual_mode,
                },
            )

        captured_generation = (
            current_generation
        )

        temporal_engine = (
            state.temporal_fusion
        )

        audio_name = (
            state.audio_name
        )

        audio_source_kind = (
            state.audio_source_kind
        )

        captured_visual_mode = (
            state.visual_mode
        )

        visual_path = (
            state.visual_path
        )

        visual_name = (
            state.visual_name
        )

        visual_started_at = (
            state.visual_started_at
        )

        if (
            audio_source_kind
            == "microphone_stream"
        ):
            if (
                not
                state.audio_stream_active
            ):
                raise HTTPException(
                    status_code=409,
                    detail={
                        "type":
                            "audio_state_mismatch",

                        "generation":
                            current_generation,

                        "audio_source_kind":
                            state.audio_source_kind,

                        "message":
                            (
                                "Microphone source is "
                                "not configured."
                            ),
                    },
                )

            captured_audio_pcm = bytes(
                state.audio_pcm_buffer
            )

            buffered_seconds = (
                len(captured_audio_pcm)
                / (TARGET_SR * 2)
            )

            if (
                buffered_seconds
                < AUDIO_STREAM_MIN_SECONDS
            ):
                raise HTTPException(
                    status_code=409,
                    detail=(
                        "Microphone buffer is "
                        "still warming up: "
                        f"{buffered_seconds:.2f}/"
                        f"{AUDIO_STREAM_MIN_SECONDS:.2f}s."
                    ),
                )

            audio_diagnostics = dict(
                state.audio_diagnostics
            )

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
                detail={
                    "type":
                        "audio_state_mismatch",

                    "generation":
                        current_generation,

                    "audio_source_kind":
                        state.audio_source_kind,

                    "message":
                        (
                            "Audio modality is required."
                        ),
                },
            )

    if captured_visual_mode not in {
        "image",
        "video",
        "webcam",
    }:
        raise HTTPException(
            status_code=409,
            detail={
                "type":
                    "visual_mode_mismatch",

                "generation":
                    captured_generation,

                "visual_mode":
                    captured_visual_mode,
            },
        )

    keystroke_path = None
    temporary_audio_path = None
    temporary_image_path = None

    try:
        keystroke_path = (
            create_keystroke_json(
                session_id=session_id,
                text=text,
                events=events,
            )
        )

        if captured_audio_pcm is not None:
            temporary_audio_path = (
                session_directory(
                    session_id
                )
                / (
                    "live_audio_"
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
                    detail={
                        "type":
                            "visual_mode_mismatch",

                        "generation":
                            captured_generation,

                        "visual_mode":
                            "none",

                        "message":
                            (
                                "Registered image "
                                "is unavailable."
                            ),
                    },
                )

            image_path = (
                visual_path
            )

        elif (
            captured_visual_mode
            == "video"
        ):
            if (
                visual_path is None
                or
                not visual_path.exists()
                or
                visual_started_at is None
            ):
                raise HTTPException(
                    status_code=409,
                    detail={
                        "type":
                            "visual_mode_mismatch",

                        "generation":
                            captured_generation,

                        "visual_mode":
                            "none",

                        "message":
                            (
                                "Registered video "
                                "is unavailable."
                            ),
                    },
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

            image_path = (
                await asyncio.to_thread(
                    extract_video_snapshot,
                    video_path=visual_path,
                    started_at=visual_started_at,
                    output_path=(
                        temporary_image_path
                    ),
                )
            )

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

            image_path = (
                await asyncio.to_thread(
                    canonicalise_webcam_frame,
                    webcam_frame,
                    temporary_image_path,
                )
            )

        print(
            "[prediction] Beginning inference "
            f"session={session_id} "
            f"generation={captured_generation}",
            flush=True,
        )

        inference_started = (
            time.monotonic()
        )

        raw_result = (
            await asyncio.to_thread(
                run_canonical_prediction,
                keystroke_json=(
                    keystroke_path
                ),
                text=text,
                audio_path=audio_path,
                image_path=image_path,
            )
        )

        inference_seconds = (
            time.monotonic()
            - inference_started
        )

        print(
            "[prediction] Raw inference finished "
            f"in {inference_seconds:.2f}s",
            flush=True,
        )

        try:
            result = (
                build_prediction_result(
                    raw_result=raw_result,
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
                current_generation = (
                    get_session(
                        session_id
                    )
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

        # Confirm the session's temporal object has not been replaced.
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

        result["session_id"] = (
            session_id
        )

        result["audio_source_name"] = (
            audio_name
        )

        result[
            "inference_seconds"
        ] = round(
            inference_seconds,
            3,
        )

        try:
            log_prediction(
                session_id=session_id,
                generation=result[
                    "generation"
                ],
                text=text,
                keystroke_count=(
                    keydown_count
                ),
                audio_name=audio_name,
                result=result,
            )

        except Exception as exc:
            print(
                "[logging] Warning: "
                f"{type(exc).__name__}: "
                f"{exc}",
                flush=True,
            )

        return JSONResponse(
            result
        )

    except HTTPException:
        raise

    except Exception as exc:
        print(
            "[prediction] Failure:\n"
            f"{traceback.format_exc()}",
            flush=True,
        )

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
# ============================================================

@app.post("/reset_temporal")
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
            "status": "ok",
            "session_id": session_id,
            "generation": generation,

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
        }
    )


# ============================================================
# FULL RESET
# ============================================================

@app.post("/full_reset")
async def full_reset(
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

        old_audio_path = (
            state.audio_path
        )

        old_visual_path = (
            state.visual_path
            if state.visual_mode
            in {"image", "video"}
            else None
        )

        clear_audio_source(
            state
        )

        state.visual_mode = "none"
        state.visual_path = None
        state.visual_name = None
        state.visual_started_at = None

        temporal_status = (
            state.temporal_fusion
            .status()
        )

    safe_delete(
        old_audio_path
    )

    safe_delete(
        old_visual_path
    )

    return JSONResponse(
        {
            "status": "ok",
            "session_id": session_id,
            "generation": generation,

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

            "audio_stream_connected":
                False,

            "visual_ready":
                False,

            "visual_mode":
                "none",
        }
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    port = int(
        os.environ.get(
            "PORT",
            "8000",
        )
    )

    uvicorn.run(
        "web_app.app:app",
        host="0.0.0.0",
        port=port,
        reload=False,
    )
