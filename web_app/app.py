# === web_app/app.py ===
# run and launch the web application: uvicorn web_app.app:app --reload | python app.py

from __future__ import annotations

import base64
import csv
import json
import os
import sys
import threading
import time
import uuid

from collections import deque
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import joblib
import numpy as np
import pandas as pd
import uvicorn

from fastapi import (
    FastAPI,
    File,
    Form,
    HTTPException,
    Request,
    UploadFile,
)

from fastapi.responses import (
    HTMLResponse,
    JSONResponse,
)

from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates


# ============================================================
# Project paths
# ============================================================

ROOT_DIR = Path(__file__).resolve().parents[1]
WEB_DIR = Path(__file__).resolve().parent

UPLOAD_DIR = WEB_DIR / "uploads"
OUTPUT_DIR = WEB_DIR / "output"

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

os.chdir(ROOT_DIR)

if str(ROOT_DIR) not in sys.path:
    sys.path.insert(
        0,
        str(ROOT_DIR),
    )


# ============================================================
# Behavioural classes
# ============================================================

LABELS = [
    "focused",
    "distracted",
    "fatigued",
    "overloaded",
]


# ============================================================
# Input requirements
# ============================================================

MIN_TEXT_CHARS = 20
MIN_KEYPRESSES = 20


# ============================================================
# Temporal fusion configuration
# ============================================================

TEMPORAL_PROBABILITY_WINDOW = 5

# Remove abandoned browser sessions after this period.
SESSION_HISTORY_TTL_SECONDS = 60 * 60

# Each browser session receives its own rolling probability history.
SESSION_PROBABILITY_HISTORY: dict[
    str,
    deque[dict[str, float]],
] = {}

SESSION_LAST_SEEN: dict[
    str,
    float,
] = {}

SESSION_HISTORY_LOCK = (
    threading.Lock()
)


# ============================================================
# Webcam-calibrated image classifier
# ============================================================

IMAGE_MODEL_DIR = (
    ROOT_DIR
    / "models"
    / "image_demo"
)

WEBCAM_IMAGE_MODEL_PATH = (
    IMAGE_MODEL_DIR
    / "image_pipeline_webcam_calibrated.joblib"
)

IMAGE_FEATURE_COLUMNS_PATH = (
    IMAGE_MODEL_DIR
    / "feature_columns.json"
)

webcam_image_pipeline = None
webcam_image_feature_columns: list[str] = []


# ============================================================
# Model status
# ============================================================

MODEL_STATUS = {
    "text_model": False,
    "audio_model": False,
    "image_model": False,
    "webcam_calibrated_image_model": False,
    "keystroke_model": False,
    "fusion_model": False,
    "temporal_probability_window": (
        TEMPORAL_PROBABILITY_WINDOW
    ),
    "inference_backend": "fallback",
    "error": None,
}

predictor = None


# ============================================================
# Logging
# ============================================================

def initialise_log_file() -> None:

    if LOG_FILE.exists():
        return

    with LOG_FILE.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:

        writer = csv.writer(f)

        writer.writerow(
            [
                "timestamp",
                "session_id",
                "text_length",
                "keystroke_count",
                "audio_available",
                "image_available",

                "raw_fusion_state",
                "raw_fusion_confidence",

                "final_state",
                "final_confidence",
                "confidence_level",
                "confidence_gap",

                "temporal_samples",
                "temporal_window",

                "webcam_state",
                "webcam_confidence",

                "feature_dimension",
                "used_modalities",
            ]
        )


# ============================================================
# Model loading
# ============================================================

def initialise_models() -> None:

    global predictor
    global webcam_image_pipeline
    global webcam_image_feature_columns

    errors: list[str] = []

    # --------------------------------------------------------
    # Main multimodal inference pipeline
    # --------------------------------------------------------

    try:

        from final_multimodal_inference import (
            FinalMultimodalInference,
        )

        predictor = (
            FinalMultimodalInference()
        )

        MODEL_STATUS.update(
            {
                "text_model": True,
                "audio_model": True,
                "image_model": True,
                "keystroke_model": True,
                "fusion_model": True,
                "inference_backend": (
                    "final_multimodal_inference."
                    "FinalMultimodalInference"
                ),
            }
        )

    except Exception as exc:

        predictor = None

        MODEL_STATUS.update(
            {
                "text_model": False,
                "audio_model": False,
                "image_model": False,
                "keystroke_model": False,
                "fusion_model": False,
                "inference_backend": "fallback",
            }
        )

        errors.append(
            f"Fusion backend: {exc}"
        )

    # --------------------------------------------------------
    # Separate webcam-calibrated image classifier
    # --------------------------------------------------------

    try:

        if not WEBCAM_IMAGE_MODEL_PATH.exists():

            raise FileNotFoundError(
                "Missing webcam-calibrated "
                f"model: {WEBCAM_IMAGE_MODEL_PATH}"
            )

        if not IMAGE_FEATURE_COLUMNS_PATH.exists():

            raise FileNotFoundError(
                "Missing image feature schema: "
                f"{IMAGE_FEATURE_COLUMNS_PATH}"
            )

        webcam_image_pipeline = (
            joblib.load(
                WEBCAM_IMAGE_MODEL_PATH
            )
        )

        with IMAGE_FEATURE_COLUMNS_PATH.open(
            "r",
            encoding="utf-8",
        ) as f:

            webcam_image_feature_columns = (
                json.load(f)
            )

        MODEL_STATUS[
            "webcam_calibrated_image_model"
        ] = True

    except Exception as exc:

        webcam_image_pipeline = None
        webcam_image_feature_columns = []

        MODEL_STATUS[
            "webcam_calibrated_image_model"
        ] = False

        errors.append(
            f"Webcam classifier: {exc}"
        )

    MODEL_STATUS["error"] = (
        " | ".join(errors)
        if errors
        else None
    )


# ============================================================
# Lifespan
# ============================================================

@asynccontextmanager
async def lifespan(app: FastAPI):

    initialise_log_file()
    initialise_models()

    yield


# ============================================================
# FastAPI application
# ============================================================

app = FastAPI(
    title="SenseFuzeAI Live Fusion Web App",
    lifespan=lifespan,
)

app.mount(
    "/static",
    StaticFiles(
        directory=WEB_DIR / "static"
    ),
    name="static",
)

templates = Jinja2Templates(
    directory=WEB_DIR / "templates"
)


# ============================================================
# Routes
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
def health() -> Dict[str, Any]:

    return {
        "status": "ok",
        "service": (
            "SenseFuzeAI Live Fusion"
        ),
        "timestamp": (
            datetime.utcnow().isoformat()
        ),
        "temporal_probability_window": (
            TEMPORAL_PROBABILITY_WINDOW
        ),
    }


@app.get("/model-status")
def model_status() -> Dict[str, Any]:

    return MODEL_STATUS


# ============================================================
# Uploaded input helpers
# ============================================================

def save_base64_image(
    image_frame: Optional[str],
) -> Optional[Path]:

    if not image_frame:
        return None

    try:

        if "," not in image_frame:
            return None

        _, encoded = (
            image_frame.split(
                ",",
                1,
            )
        )

        image_path = (
            UPLOAD_DIR
            / f"frame_{uuid.uuid4().hex}.jpg"
        )

        image_path.write_bytes(
            base64.b64decode(encoded)
        )

        return image_path

    except Exception:
        return None


async def save_audio_chunk(
    audio_chunk: Optional[UploadFile],
) -> Optional[Path]:

    if audio_chunk is None:
        return None

    content = await audio_chunk.read()

    if not content:
        return None

    audio_path = (
        UPLOAD_DIR
        / f"audio_{uuid.uuid4().hex}.webm"
    )

    audio_path.write_bytes(content)

    return audio_path


# ============================================================
# Keystroke parsing
# ============================================================

def parse_keystrokes(
    keystroke_events: str,
) -> list[dict[str, Any]]:

    try:

        events = json.loads(
            keystroke_events
        )

        if isinstance(events, list):
            return events

        return []

    except Exception:
        return []


def extract_keystroke_count(
    keystroke_events: str,
) -> int:

    return sum(
        1
        for event
        in parse_keystrokes(
            keystroke_events
        )
        if event.get("type")
        == "down"
    )


# ============================================================
# Live keystroke features
# ============================================================

def live_keystroke_features(
    events: list[dict[str, Any]],
) -> dict[str, float]:

    downs = [
        e
        for e in events
        if e.get("type")
        == "down"
    ]

    down_times = [
        float(
            e.get(
                "timestamp_perf",
                0.0,
            )
        )
        for e in downs
    ]

    intervals = (
        np.diff(down_times)
        if len(down_times) >= 2
        else np.array([])
    )

    hold_times = []
    active = {}

    for event in events:

        key = event.get("key")

        timestamp = float(
            event.get(
                "timestamp_perf",
                0.0,
            )
        )

        if event.get("type") == "down":

            active[key] = timestamp

        elif (
            event.get("type") == "up"
            and key in active
        ):

            hold_times.append(
                max(
                    0.0,
                    timestamp
                    - active.pop(key),
                )
            )

    hold_times = np.array(
        hold_times,
        dtype=float,
    )

    duration = (
        max(down_times)
        - min(down_times)
        if len(down_times) >= 2
        else 0.0
    )

    key_count = len(downs)

    pauses_1000 = sum(
        1
        for value in intervals
        if value >= 1.0
    )

    pauses_2000 = sum(
        1
        for value in intervals
        if value >= 2.0
    )

    pauses_5000 = sum(
        1
        for value in intervals
        if value >= 5.0
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

    interval_mean = (
        float(np.mean(intervals))
        if len(intervals)
        else 0.0
    )

    interval_std = (
        float(np.std(intervals))
        if len(intervals)
        else 0.0
    )

    return {
        "total_duration_sec": round(
            float(duration),
            4,
        ),

        "keydown_count": int(
            key_count
        ),

        "word_count": 0,

        "typing_speed_kps": (
            round(
                float(
                    key_count
                    / duration
                ),
                4,
            )
            if duration > 0
            else 0.0
        ),

        "typing_speed_wpm": 0.0,

        "delay_mean": (
            round(
                interval_mean,
                4,
            )
        ),

        "delay_std": (
            round(
                float(
                    np.std(
                        intervals,
                        ddof=1,
                    )
                ),
                4,
            )
            if len(intervals) >= 2
            else 0.0
        ),

        "delay_min": (
            round(
                float(
                    np.min(
                        intervals
                    )
                ),
                4,
            )
            if len(intervals)
            else 0.0
        ),

        "delay_max": (
            round(
                float(
                    np.max(
                        intervals
                    )
                ),
                4,
            )
            if len(intervals)
            else 0.0
        ),

        "hold_mean": (
            round(
                float(
                    np.mean(
                        hold_times
                    )
                ),
                4,
            )
            if len(hold_times)
            else 0.0
        ),

        "hold_std": (
            round(
                float(
                    np.std(
                        hold_times,
                        ddof=1,
                    )
                ),
                4,
            )
            if len(hold_times) >= 2
            else 0.0
        ),

        "pause_count_1000": int(
            pauses_1000
        ),

        "pause_count_2000": int(
            pauses_2000
        ),

        "pause_count_5000": int(
            pauses_5000
        ),

        "pause_ratio_1000": (
            round(
                float(
                    pauses_1000
                    / len(intervals)
                ),
                4,
            )
            if len(intervals)
            else 0.0
        ),

        "pause_ratio_2000": (
            round(
                float(
                    pauses_2000
                    / len(intervals)
                ),
                4,
            )
            if len(intervals)
            else 0.0
        ),

        "mental_block_ratio_5000": (
            round(
                float(
                    pauses_5000
                    / len(intervals)
                ),
                4,
            )
            if len(intervals)
            else 0.0
        ),

        "correction_count": int(
            correction_count
        ),

        "correction_ratio": (
            round(
                float(
                    correction_count
                    / key_count
                ),
                4,
            )
            if key_count
            else 0.0
        ),

        "rhythm_consistency": (
            round(
                float(
                    1.0
                    / (
                        1.0
                        + interval_std
                    )
                ),
                4,
            )
            if len(intervals)
            else 1.0
        ),

        "burstiness_proxy": (
            round(
                float(
                    interval_std
                    / interval_mean
                ),
                4,
            )
            if (
                len(intervals)
                and interval_mean > 0
            )
            else 0.0
        ),

        "fits_starts_index": (
            round(
                float(
                    pauses_1000
                    / len(intervals)
                ),
                4,
            )
            if len(intervals)
            else 0.0
        ),
    }


# ============================================================
# Fallback classifier
# ============================================================

def fallback_prediction(
    text: str,
) -> Dict[str, float]:

    lower_text = text.lower()

    scores = {
        "focused": 0.40,
        "distracted": 0.20,
        "fatigued": 0.20,
        "overloaded": 0.20,
    }

    if any(
        word in lower_text
        for word in [
            "tired",
            "sleepy",
            "exhausted",
            "fatigue",
        ]
    ):
        scores["fatigued"] += 0.25

    if any(
        word in lower_text
        for word in [
            "confused",
            "too much",
            "stress",
            "overload",
        ]
    ):
        scores["overloaded"] += 0.25

    if any(
        word in lower_text
        for word in [
            "distracted",
            "bored",
            "phone",
            "noise",
        ]
    ):
        scores["distracted"] += 0.25

    total = sum(
        scores.values()
    )

    return {
        label: value / total
        for label, value
        in scores.items()
    }


# ============================================================
# Confidence
# ============================================================

def get_confidence_level(
    confidence_gap: float,
) -> str:

    if confidence_gap >= 0.35:
        return "High"

    if confidence_gap >= 0.15:
        return "Medium"

    return "Low"


# ============================================================
# Probability helpers
# ============================================================

def normalise_probability_distribution(
    probabilities: dict[str, float],
) -> dict[str, float]:

    output = {
        label: max(
            0.0,
            float(
                probabilities.get(
                    label,
                    0.0,
                )
            ),
        )
        for label in LABELS
    }

    total = sum(
        output.values()
    )

    if total <= 0:

        return {
            label: 1.0 / len(LABELS)
            for label in LABELS
        }

    return {
        label: value / total
        for label, value
        in output.items()
    }


# ============================================================
# Webcam-calibrated classifier
# ============================================================

def run_webcam_calibrated_prediction(
    image_features: Optional[
        dict[str, float]
    ],
) -> Optional[dict[str, Any]]:

    if (
        image_features is None
        or webcam_image_pipeline is None
        or not webcam_image_feature_columns
    ):
        return None

    missing = [
        column
        for column
        in webcam_image_feature_columns
        if column
        not in image_features
    ]

    if missing:

        raise ValueError(
            "Webcam calibrated image "
            "feature mismatch. "
            f"Missing: {missing[:20]}"
        )

    row = pd.DataFrame(
        [
            [
                float(
                    image_features[column]
                )
                for column
                in webcam_image_feature_columns
            ]
        ],
        columns=(
            webcam_image_feature_columns
        ),
    )

    prediction = (
        webcam_image_pipeline
        .predict(row)[0]
    )

    probabilities = (
        webcam_image_pipeline
        .predict_proba(row)[0]
    )

    classes = (
        webcam_image_pipeline.classes_
    )

    probability_dict = {
        str(label): float(prob)
        for label, prob
        in zip(
            classes,
            probabilities,
        )
    }

    probability_dict = (
        normalise_probability_distribution(
            probability_dict
        )
    )

    ranked = sorted(
        probability_dict.items(),
        key=lambda item: item[1],
        reverse=True,
    )

    top_class, top_probability = (
        ranked[0]
    )

    second_class, second_probability = (
        ranked[1]
    )

    return {
        "prediction": str(
            prediction
        ),
        "current_state": (
            top_class
        ),
        "confidence": float(
            top_probability
        ),
        "confidence_percent": round(
            top_probability * 100,
            2,
        ),
        "second_class": second_class,
        "second_probability": float(
            second_probability
        ),
        "confidence_gap": float(
            top_probability
            - second_probability
        ),
        "confidence_level": (
            get_confidence_level(
                top_probability
                - second_probability
            )
        ),
        "probabilities": (
            probability_dict
        ),
        "classifier": (
            "webcam-calibrated"
        ),
    }


# ============================================================
# Multimodal prediction backend
# ============================================================

def run_prediction_backend(
    text: str,
    keystroke_events: str,
    image_path: Optional[Path],
    audio_path: Optional[Path],
) -> Dict[str, Any]:

    # --------------------------------------------------------
    # Fallback mode
    # --------------------------------------------------------

    if predictor is None:

        probabilities = (
            fallback_prediction(text)
        )

        return {
            "prediction": max(
                probabilities,
                key=probabilities.get,
            ),
            "probabilities": probabilities,
            "device": "cpu",
            "feature_dimension": "fallback",
            "used_modalities": {
                "text": bool(text),
                "keystroke": bool(
                    keystroke_events
                ),
                "audio": (
                    audio_path
                    is not None
                ),
                "image": (
                    image_path
                    is not None
                ),
            },
            "webcam_prediction": None,
        }

    # --------------------------------------------------------
    # Build fusion feature vector
    # --------------------------------------------------------

    features: dict[str, Any] = {}

    events = parse_keystrokes(
        keystroke_events
    )

    keystroke_features = (
        live_keystroke_features(
            events
        )
    )

    text_word_count = len(
        text.split()
    )

    if (
        keystroke_features.get(
            "total_duration_sec",
            0,
        )
        > 0
    ):

        keystroke_features[
            "word_count"
        ] = text_word_count

        keystroke_features[
            "typing_speed_wpm"
        ] = round(
            (
                text_word_count
                / keystroke_features[
                    "total_duration_sec"
                ]
            )
            * 60,
            4,
        )

    features.update(
        keystroke_features
    )

    features.update(
        predictor.extract_text_features(
            text
        )
    )

    # --------------------------------------------------------
    # Audio
    # --------------------------------------------------------

    if audio_path is not None:

        features.update(
            predictor.extract_audio_features(
                audio_path
            )
        )

    # --------------------------------------------------------
    # Image
    # --------------------------------------------------------

    image_features = None

    if image_path is not None:

        image_features = (
            predictor.extract_image_features(
                image_path
            )
        )

        features.update(
            image_features
        )

    # --------------------------------------------------------
    # Webcam-calibrated diagnostic
    # --------------------------------------------------------

    webcam_prediction = (
        run_webcam_calibrated_prediction(
            image_features
        )
    )

    # --------------------------------------------------------
    # Fusion model vector
    # --------------------------------------------------------

    row = {
        column: float(
            features.get(
                column,
                0.0,
            )
        )
        for column
        in predictor.feature_columns
    }

    x = pd.DataFrame(
        [row],
        columns=predictor.feature_columns,
    )

    prediction = (
        predictor.fusion_model
        .predict(x)[0]
    )

    if hasattr(
        predictor.fusion_model,
        "predict_proba",
    ):

        probabilities = (
            predictor.fusion_model
            .predict_proba(x)[0]
        )

        classes = (
            predictor.fusion_model
            .classes_
        )

        probability_dict = {
            str(label): float(probability)
            for label, probability
            in zip(
                classes,
                probabilities,
            )
        }

    else:

        probability_dict = {
            label: 0.10
            for label in LABELS
        }

        probability_dict[
            str(prediction)
        ] = 0.70

    probability_dict = (
        normalise_probability_distribution(
            probability_dict
        )
    )

    return {
        "prediction": str(
            prediction
        ),
        "probabilities": (
            probability_dict
        ),
        "device": str(
            predictor.device
        ),
        "feature_dimension": int(
            x.shape[1]
        ),
        "used_modalities": {
            "text": True,
            "keystroke": True,
            "audio": (
                audio_path
                is not None
            ),
            "image": (
                image_path
                is not None
            ),
        },
        "webcam_prediction": (
            webcam_prediction
        ),
    }


# ============================================================
# Temporal-session helpers
# ============================================================

def cleanup_expired_sessions() -> None:

    now = time.time()

    expired = [
        session_id
        for session_id, last_seen
        in SESSION_LAST_SEEN.items()
        if (
            now - last_seen
            > SESSION_HISTORY_TTL_SECONDS
        )
    ]

    for session_id in expired:

        SESSION_PROBABILITY_HISTORY.pop(
            session_id,
            None,
        )

        SESSION_LAST_SEEN.pop(
            session_id,
            None,
        )


def add_temporal_probability(
    session_id: str,
    probabilities: dict[str, float],
) -> tuple[
    dict[str, float],
    int,
]:

    probabilities = (
        normalise_probability_distribution(
            probabilities
        )
    )

    with SESSION_HISTORY_LOCK:

        cleanup_expired_sessions()

        if (
            session_id
            not in SESSION_PROBABILITY_HISTORY
        ):

            SESSION_PROBABILITY_HISTORY[
                session_id
            ] = deque(
                maxlen=(
                    TEMPORAL_PROBABILITY_WINDOW
                )
            )

        history = (
            SESSION_PROBABILITY_HISTORY[
                session_id
            ]
        )

        history.append(
            probabilities
        )

        SESSION_LAST_SEEN[
            session_id
        ] = time.time()

        aggregated = {}

        for label in LABELS:

            aggregated[label] = float(
                np.mean(
                    [
                        observation.get(
                            label,
                            0.0,
                        )
                        for observation
                        in history
                    ]
                )
            )

        aggregated = (
            normalise_probability_distribution(
                aggregated
            )
        )

        return (
            aggregated,
            len(history),
        )


def clear_temporal_session(
    session_id: str,
) -> None:

    with SESSION_HISTORY_LOCK:

        SESSION_PROBABILITY_HISTORY.pop(
            session_id,
            None,
        )

        SESSION_LAST_SEEN.pop(
            session_id,
            None,
        )


# ============================================================
# Result normalisation + temporal aggregation
# ============================================================

def normalise_prediction_result(
    raw: Dict[str, Any],
    session_id: str,
) -> Dict[str, Any]:

    # --------------------------------------------------------
    # Raw current fusion result
    # --------------------------------------------------------

    raw_probabilities = (
        normalise_probability_distribution(
            {
                str(key): float(value)
                for key, value
                in raw[
                    "probabilities"
                ].items()
            }
        )
    )

    raw_ranked = sorted(
        raw_probabilities.items(),
        key=lambda item: item[1],
        reverse=True,
    )

    raw_state, raw_confidence = (
        raw_ranked[0]
    )

    # --------------------------------------------------------
    # Temporal aggregation
    # --------------------------------------------------------

    (
        aggregated_probabilities,
        temporal_samples,
    ) = add_temporal_probability(
        session_id=session_id,
        probabilities=raw_probabilities,
    )

    ranked = sorted(
        aggregated_probabilities.items(),
        key=lambda item: item[1],
        reverse=True,
    )

    current_state, confidence = (
        ranked[0]
    )

    (
        second_state,
        second_probability,
    ) = ranked[1]

    confidence_gap = (
        confidence
        - second_probability
    )

    # --------------------------------------------------------
    # Final API contract
    # --------------------------------------------------------

    return {
        "session_id": session_id,

        "prediction": current_state,
        "current_state": current_state,

        "confidence": float(
            confidence
        ),

        "confidence_percent": round(
            confidence * 100,
            2,
        ),

        "confidence_gap": float(
            confidence_gap
        ),

        "confidence_level": (
            get_confidence_level(
                confidence_gap
            )
        ),

        # Final temporally aggregated distribution.
        "probabilities": (
            aggregated_probabilities
        ),

        # Current unsmoothed fusion result.
        "raw_prediction": (
            raw_state
        ),

        "raw_confidence": float(
            raw_confidence
        ),

        "raw_confidence_percent": round(
            raw_confidence * 100,
            2,
        ),

        "raw_probabilities": (
            raw_probabilities
        ),

        "temporal_samples": (
            temporal_samples
        ),

        "temporal_window": (
            TEMPORAL_PROBABILITY_WINDOW
        ),

        "temporal_aggregation": (
            "rolling_mean_probability"
        ),

        "technical_details": {
            "top_class": (
                current_state
            ),
            "second_class": (
                second_state
            ),
            "second_probability": (
                float(
                    second_probability
                )
            ),
            "confidence_gap": (
                float(
                    confidence_gap
                )
            ),
            "raw_top_class": (
                raw_state
            ),
            "raw_top_probability": (
                float(
                    raw_confidence
                )
            ),
            "temporal_samples": (
                temporal_samples
            ),
            "temporal_window": (
                TEMPORAL_PROBABILITY_WINDOW
            ),
            "temporal_aggregation": (
                "rolling_mean_probability"
            ),
            "device": raw.get(
                "device",
                "unknown",
            ),
            "feature_dimension": (
                raw.get(
                    "feature_dimension",
                    "unknown",
                )
            ),
            "used_modalities": (
                raw.get(
                    "used_modalities",
                    {},
                )
            ),
        },

        "device": raw.get(
            "device",
            "unknown",
        ),

        "feature_dimension": (
            raw.get(
                "feature_dimension",
                "unknown",
            )
        ),

        "used_modalities": (
            raw.get(
                "used_modalities",
                {},
            )
        ),

        "webcam_calibration_used": (
            raw.get(
                "webcam_prediction"
            )
            is not None
        ),

        "webcam_prediction": (
            raw.get(
                "webcam_prediction"
            )
        ),
    }


# ============================================================
# Prediction logging
# ============================================================

def log_prediction(
    session_id: str,
    text: str,
    keystroke_count: int,
    audio_available: bool,
    image_available: bool,
    result: Dict[str, Any],
) -> None:

    webcam_result = (
        result.get(
            "webcam_prediction"
        )
        or {}
    )

    with LOG_FILE.open(
        "a",
        newline="",
        encoding="utf-8",
    ) as f:

        writer = csv.writer(f)

        writer.writerow(
            [
                datetime.utcnow().isoformat(),

                session_id,

                len(text),

                keystroke_count,

                audio_available,

                image_available,

                result.get(
                    "raw_prediction"
                ),

                result.get(
                    "raw_confidence"
                ),

                result[
                    "current_state"
                ],

                result[
                    "confidence"
                ],

                result[
                    "confidence_level"
                ],

                result[
                    "confidence_gap"
                ],

                result[
                    "temporal_samples"
                ],

                result[
                    "temporal_window"
                ],

                webcam_result.get(
                    "current_state"
                ),

                webcam_result.get(
                    "confidence"
                ),

                result[
                    "feature_dimension"
                ],

                json.dumps(
                    result.get(
                        "used_modalities",
                        {},
                    )
                ),
            ]
        )


# ============================================================
# Prediction endpoint
# ============================================================

@app.post("/predict_live")
async def predict_live(

    session_id: str = Form(""),

    text: str = Form(""),

    keystroke_events: str = Form(
        "[]"
    ),

    image_frame: Optional[str] = Form(
        None
    ),

    audio_chunk: Optional[UploadFile] = File(
        None
    ),

) -> JSONResponse:

    text = text.strip()
    session_id = session_id.strip()

    # Older clients/tests that do not send session_id
    # still work, but receive a one-request history.
    if not session_id:

        session_id = (
            "legacy-"
            + uuid.uuid4().hex
        )

    keystroke_count = (
        extract_keystroke_count(
            keystroke_events
        )
    )

    if len(text) < MIN_TEXT_CHARS:

        raise HTTPException(
            status_code=400,
            detail=(
                "At least 20 text "
                "characters are required."
            ),
        )

    if keystroke_count < MIN_KEYPRESSES:

        raise HTTPException(
            status_code=400,
            detail=(
                "At least 20 keypresses "
                "are required."
            ),
        )

    image_path = save_base64_image(
        image_frame
    )

    audio_path = await save_audio_chunk(
        audio_chunk
    )

    try:

        raw_result = (
            run_prediction_backend(
                text=text,
                keystroke_events=(
                    keystroke_events
                ),
                image_path=image_path,
                audio_path=audio_path,
            )
        )

        result = (
            normalise_prediction_result(
                raw=raw_result,
                session_id=session_id,
            )
        )

        log_prediction(
            session_id=session_id,
            text=text,
            keystroke_count=(
                keystroke_count
            ),
            audio_available=(
                audio_path
                is not None
            ),
            image_available=(
                image_path
                is not None
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
            detail=str(exc),
        )

    finally:

        # Temporary live media files are no longer
        # required after feature extraction.
        for temp_path in [
            image_path,
            audio_path,
        ]:

            if (
                temp_path is not None
                and temp_path.exists()
            ):

                try:
                    temp_path.unlink()
                except OSError:
                    pass


# ============================================================
# Temporal reset endpoint
# ============================================================

@app.post("/reset_temporal")
async def reset_temporal(
    session_id: str = Form(...),
) -> JSONResponse:

    session_id = (
        session_id.strip()
    )

    if not session_id:

        raise HTTPException(
            status_code=400,
            detail=(
                "session_id is required."
            ),
        )

    clear_temporal_session(
        session_id
    )

    return JSONResponse(
        {
            "status": "ok",
            "session_id": session_id,
            "message": (
                "Temporal probability "
                "history cleared."
            ),
            "temporal_samples": 0,
            "temporal_window": (
                TEMPORAL_PROBABILITY_WINDOW
            ),
        }
    )


# ============================================================
# Entry point
# ============================================================

if __name__ == "__main__":

    uvicorn.run(
        "app:app",
        host="127.0.0.1",
        port=8000,
        reload=True,
    )
