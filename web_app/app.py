# === web_app/app.py ===

from __future__ import annotations

import base64
import csv
import json
import os
import sys
import uuid

from contextlib import asynccontextmanager
from datetime import datetime, timezone
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

from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates


# =============================================================================
# PROJECT PATHS
# =============================================================================

ROOT_DIR = Path(__file__).resolve().parents[1]
WEB_DIR = Path(__file__).resolve().parent

UPLOAD_DIR = WEB_DIR / "uploads"
OUTPUT_DIR = WEB_DIR / "output"
LOG_FILE = OUTPUT_DIR / "live_predictions.csv"

IMAGE_MODEL_DIR = ROOT_DIR / "models" / "image_demo"

WEBCAM_CALIBRATED_IMAGE_MODEL_PATH = (
    IMAGE_MODEL_DIR
    / "image_pipeline_webcam_calibrated.joblib"
)

IMAGE_FEATURE_COLUMNS_PATH = (
    IMAGE_MODEL_DIR
    / "feature_columns.json"
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


# =============================================================================
# CONSTANTS
# =============================================================================

LABELS = [
    "focused",
    "distracted",
    "fatigued",
    "overloaded",
]

MIN_TEXT_CHARS = 20
MIN_KEYPRESSES = 20


# =============================================================================
# GLOBAL MODEL STATE
# =============================================================================

predictor = None

webcam_image_model = None
webcam_image_feature_columns: list[str] = []


MODEL_STATUS = {
    "text_model": False,
    "audio_model": False,
    "image_model": False,
    "webcam_calibrated_image_model": False,
    "keystroke_model": False,
    "fusion_model": False,
    "inference_backend": "fallback",
    "webcam_image_backend": "unavailable",
    "error": None,
    "webcam_error": None,
}


# =============================================================================
# LOG FILE
# =============================================================================

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
                "text_length",
                "keystroke_count",
                "audio_available",
                "image_available",
                "webcam_calibrated_model_used",
                "webcam_predicted_state",
                "webcam_confidence",
                "predicted_label",
                "confidence",
                "confidence_level",
                "confidence_gap",
                "feature_dimension",
                "used_modalities",
            ]
        )


# =============================================================================
# MODEL INITIALISATION
# =============================================================================

def initialise_models() -> None:

    global predictor
    global webcam_image_model
    global webcam_image_feature_columns

    errors: list[str] = []

    # -------------------------------------------------------------------------
    # Final multimodal inference backend
    # -------------------------------------------------------------------------

    try:
        from final_multimodal_inference import FinalMultimodalInference

        predictor = FinalMultimodalInference()

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

        errors.append(
            f"Fusion inference backend: {exc}"
        )

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

    # -------------------------------------------------------------------------
    # Webcam calibrated image classifier
    # -------------------------------------------------------------------------

    try:

        if not WEBCAM_CALIBRATED_IMAGE_MODEL_PATH.exists():

            raise FileNotFoundError(
                "Webcam-calibrated image model not found: "
                f"{WEBCAM_CALIBRATED_IMAGE_MODEL_PATH}"
            )

        if not IMAGE_FEATURE_COLUMNS_PATH.exists():

            raise FileNotFoundError(
                "Image feature schema not found: "
                f"{IMAGE_FEATURE_COLUMNS_PATH}"
            )

        webcam_image_model = joblib.load(
            WEBCAM_CALIBRATED_IMAGE_MODEL_PATH
        )

        with IMAGE_FEATURE_COLUMNS_PATH.open(
            "r",
            encoding="utf-8",
        ) as f:

            webcam_image_feature_columns = json.load(f)

        if not isinstance(
            webcam_image_feature_columns,
            list,
        ):
            raise ValueError(
                "Image feature_columns.json must contain a list."
            )

        MODEL_STATUS.update(
            {
                "webcam_calibrated_image_model": True,
                "webcam_image_backend": str(
                    WEBCAM_CALIBRATED_IMAGE_MODEL_PATH
                ),
                "webcam_error": None,
            }
        )

    except Exception as exc:

        webcam_image_model = None
        webcam_image_feature_columns = []

        MODEL_STATUS.update(
            {
                "webcam_calibrated_image_model": False,
                "webcam_image_backend": "unavailable",
                "webcam_error": str(exc),
            }
        )

        errors.append(
            f"Webcam image classifier: {exc}"
        )

    MODEL_STATUS["error"] = (
        " | ".join(errors)
        if errors
        else None
    )


# =============================================================================
# FASTAPI LIFESPAN
# =============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):

    initialise_log_file()
    initialise_models()

    yield


# =============================================================================
# FASTAPI APPLICATION
# =============================================================================

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


# =============================================================================
# BASIC ROUTES
# =============================================================================

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
        "service": "SenseFuzeAI Live Fusion",
        "timestamp": datetime.now(
            timezone.utc
        ).isoformat(),
    }


@app.get("/model-status")
def model_status() -> Dict[str, Any]:

    return MODEL_STATUS


# =============================================================================
# FILE UTILITIES
# =============================================================================

def save_base64_image(
    image_frame: Optional[str],
) -> Optional[Path]:

    if not image_frame:
        return None

    try:

        if "," in image_frame:
            _, encoded = image_frame.split(
                ",",
                1,
            )
        else:
            encoded = image_frame

        image_path = (
            UPLOAD_DIR
            / f"frame_{uuid.uuid4().hex}.jpg"
        )

        image_path.write_bytes(
            base64.b64decode(
                encoded
            )
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

    suffix = Path(
        audio_chunk.filename or ""
    ).suffix.lower()

    if not suffix:
        suffix = ".webm"

    audio_path = (
        UPLOAD_DIR
        / f"audio_{uuid.uuid4().hex}{suffix}"
    )

    audio_path.write_bytes(
        content
    )

    return audio_path


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


# =============================================================================
# KEYSTROKE UTILITIES
# =============================================================================

def parse_keystrokes(
    keystroke_events: str,
) -> list[dict[str, Any]]:

    try:

        events = json.loads(
            keystroke_events
        )

        return (
            events
            if isinstance(events, list)
            else []
        )

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
        if event.get("type") == "down"
    )


def live_keystroke_features(
    events: list[dict[str, Any]],
) -> dict[str, float]:

    downs = [
        event
        for event in events
        if event.get("type") == "down"
    ]

    down_times = [
        float(
            event.get(
                "timestamp_perf",
                0.0,
            )
        )
        for event in downs
    ]

    intervals = (
        np.diff(down_times)
        if len(down_times) >= 2
        else np.array([])
    )

    hold_times: list[float] = []
    active: dict[str, float] = {}

    for event in events:

        key = str(
            event.get(
                "key",
                "",
            )
        )

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

    hold_array = np.asarray(
        hold_times,
        dtype=float,
    )

    duration = (
        max(down_times)
        - min(down_times)
        if len(down_times) >= 2
        else 0.0
    )

    key_count = len(
        downs
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

    pause_count_1000 = sum(
        1
        for delay in intervals
        if delay >= 1.0
    )

    pause_count_2000 = sum(
        1
        for delay in intervals
        if delay >= 2.0
    )

    pause_count_5000 = sum(
        1
        for delay in intervals
        if delay >= 5.0
    )

    delay_mean = (
        float(
            np.mean(intervals)
        )
        if len(intervals)
        else 0.0
    )

    delay_std = (
        float(
            np.std(
                intervals,
                ddof=1,
            )
        )
        if len(intervals) >= 2
        else 0.0
    )

    return {
        "total_duration_sec":
            round(
                float(duration),
                4,
            ),

        "keydown_count":
            int(key_count),

        "word_count":
            0,

        "typing_speed_kps":
            round(
                float(
                    key_count / duration
                ),
                4,
            )
            if duration > 0
            else 0.0,

        "typing_speed_wpm":
            0.0,

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
            round(
                float(
                    np.min(intervals)
                ),
                4,
            )
            if len(intervals)
            else 0.0,

        "delay_max":
            round(
                float(
                    np.max(intervals)
                ),
                4,
            )
            if len(intervals)
            else 0.0,

        "hold_mean":
            round(
                float(
                    np.mean(hold_array)
                ),
                4,
            )
            if len(hold_array)
            else 0.0,

        "hold_std":
            round(
                float(
                    np.std(
                        hold_array,
                        ddof=1,
                    )
                ),
                4,
            )
            if len(hold_array) >= 2
            else 0.0,

        "pause_count_1000":
            int(pause_count_1000),

        "pause_count_2000":
            int(pause_count_2000),

        "pause_count_5000":
            int(pause_count_5000),

        "pause_ratio_1000":
            round(
                float(
                    pause_count_1000
                    / len(intervals)
                ),
                4,
            )
            if len(intervals)
            else 0.0,

        "pause_ratio_2000":
            round(
                float(
                    pause_count_2000
                    / len(intervals)
                ),
                4,
            )
            if len(intervals)
            else 0.0,

        "mental_block_ratio_5000":
            round(
                float(
                    pause_count_5000
                    / len(intervals)
                ),
                4,
            )
            if len(intervals)
            else 0.0,

        "correction_count":
            int(correction_count),

        "correction_ratio":
            round(
                float(
                    correction_count
                    / key_count
                ),
                4,
            )
            if key_count
            else 0.0,

        "rhythm_consistency":
            round(
                float(
                    1.0
                    / (
                        1.0
                        + delay_std
                    )
                ),
                4,
            ),

        "burstiness_proxy":
            round(
                float(
                    delay_std
                    / delay_mean
                ),
                4,
            )
            if delay_mean > 0
            else 0.0,

        "fits_starts_index":
            round(
                float(
                    pause_count_1000
                    / len(intervals)
                ),
                4,
            )
            if len(intervals)
            else 0.0,
    }


# =============================================================================
# FALLBACK
# =============================================================================

def fallback_prediction(
    text: str,
) -> Dict[str, float]:

    lower_text = (
        text.lower()
    )

    scores = {
        "focused": 0.40,
        "distracted": 0.20,
        "fatigued": 0.20,
        "overloaded": 0.20,
    }

    if any(
        word in lower_text
        for word
        in [
            "tired",
            "sleepy",
            "exhausted",
            "fatigue",
        ]
    ):
        scores["fatigued"] += 0.25

    if any(
        word in lower_text
        for word
        in [
            "confused",
            "too much",
            "stress",
            "overload",
        ]
    ):
        scores["overloaded"] += 0.25

    if any(
        word in lower_text
        for word
        in [
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
        label:
            value / total
        for label, value
        in scores.items()
    }


# =============================================================================
# MODEL UTILITIES
# =============================================================================

def get_model_classes(
    model: Any,
) -> list[str]:

    classes = getattr(
        model,
        "classes_",
        None,
    )

    if classes is not None:

        return [
            str(value)
            .strip()
            .lower()
            for value in classes
        ]

    if hasattr(
        model,
        "named_steps",
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

                return [
                    str(value)
                    .strip()
                    .lower()
                    for value in classes
                ]

    return []


def softmax(
    values: np.ndarray,
) -> np.ndarray:

    values = np.asarray(
        values,
        dtype=float,
    )

    values = (
        values
        - np.max(values)
    )

    exp_values = np.exp(
        values
    )

    total = float(
        exp_values.sum()
    )

    if total <= 0:

        return np.full(
            len(values),
            1.0 / len(values),
        )

    return (
        exp_values
        / total
    )


def model_probability_dict(
    model: Any,
    x: pd.DataFrame,
) -> dict[str, float]:

    classes = get_model_classes(
        model
    )

    output = {
        label: 0.0
        for label in LABELS
    }

    if hasattr(
        model,
        "predict_proba",
    ):

        probabilities = (
            model.predict_proba(x)[0]
        )

        for label, probability in zip(
            classes,
            probabilities,
        ):

            if label in LABELS:
                output[label] = float(
                    probability
                )

    elif hasattr(
        model,
        "decision_function",
    ):

        scores = np.asarray(
            model.decision_function(x)
        )

        if scores.ndim > 1:
            scores = scores[0]

        probabilities = softmax(
            scores
        )

        for label, probability in zip(
            classes,
            probabilities,
        ):

            if label in LABELS:
                output[label] = float(
                    probability
                )

    else:

        prediction = (
            str(
                model.predict(x)[0]
            )
            .strip()
            .lower()
        )

        if prediction in output:
            output[prediction] = 1.0

    total = sum(
        output.values()
    )

    if total <= 0:

        return {
            label:
                1.0 / len(LABELS)
            for label
            in LABELS
        }

    return {
        label:
            probability / total
        for label, probability
        in output.items()
    }


# =============================================================================
# WEBCAM CALIBRATED IMAGE CLASSIFIER
# =============================================================================

def run_webcam_calibrated_classifier(
    image_features: dict[str, float],
) -> Optional[dict[str, Any]]:

    if webcam_image_model is None:
        return None

    if not webcam_image_feature_columns:
        return None

    missing = [
        column
        for column
        in webcam_image_feature_columns
        if column not in image_features
    ]

    if missing:

        raise ValueError(
            "Webcam-calibrated image feature mismatch. "
            f"Missing columns: {missing[:20]}"
        )

    x = pd.DataFrame(
        [
            [
                float(
                    image_features[column]
                )
                for column
                in webcam_image_feature_columns
            ]
        ],
        columns=webcam_image_feature_columns,
    )

    probabilities = model_probability_dict(
        webcam_image_model,
        x,
    )

    ranked = sorted(
        probabilities.items(),
        key=lambda item: item[1],
        reverse=True,
    )

    top_class, top_probability = (
        ranked[0]
    )

    second_class, second_probability = (
        ranked[1]
    )

    gap = float(
        top_probability
        - second_probability
    )

    return {
        "prediction":
            top_class,

        "current_state":
            top_class,

        "confidence":
            float(
                top_probability
            ),

        "confidence_percent":
            round(
                top_probability * 100,
                2,
            ),

        "second_class":
            second_class,

        "second_probability":
            float(
                second_probability
            ),

        "confidence_gap":
            gap,

        "confidence_level":
            get_confidence_level(
                gap
            ),

        "probabilities":
            probabilities,

        "classifier":
            "webcam_calibrated_image_model",
    }


# =============================================================================
# CONFIDENCE
# =============================================================================

def get_confidence_level(
    confidence_gap: float,
) -> str:

    if confidence_gap >= 0.35:
        return "High"

    if confidence_gap >= 0.15:
        return "Medium"

    return "Low"


# =============================================================================
# FUSION BACKEND
# =============================================================================

def run_prediction_backend(
    text: str,
    keystroke_events: str,
    image_path: Optional[Path],
    audio_path: Optional[Path],
) -> Dict[str, Any]:

    # -------------------------------------------------------------------------
    # Fallback mode
    # -------------------------------------------------------------------------

    if predictor is None:

        return {
            "probabilities":
                fallback_prediction(
                    text
                ),

            "device":
                "cpu",

            "feature_dimension":
                "fallback",

            "used_modalities": {
                "text":
                    bool(text),

                "keystroke":
                    bool(
                        keystroke_events
                    ),

                "audio":
                    audio_path
                    is not None,

                "image":
                    image_path
                    is not None,
            },

            "webcam_calibration_used":
                False,

            "image_modality":
                None,
        }

    # -------------------------------------------------------------------------
    # Feature extraction
    # -------------------------------------------------------------------------

    features: dict[
        str,
        Any,
    ] = {}

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

    duration = float(
        keystroke_features.get(
            "total_duration_sec",
            0.0,
        )
    )

    keystroke_features[
        "word_count"
    ] = text_word_count

    keystroke_features[
        "typing_speed_wpm"
    ] = (
        round(
            (
                text_word_count
                / duration
            )
            * 60,
            4,
        )
        if duration > 0
        else 0.0
    )

    features.update(
        keystroke_features
    )

    features.update(
        predictor.extract_text_features(
            text
        )
    )

    # -------------------------------------------------------------------------
    # Presence features
    # -------------------------------------------------------------------------

    features["has_text"] = 1.0
    features["has_keystroke"] = 1.0

    features["has_audio"] = (
        1.0
        if audio_path is not None
        else 0.0
    )

    features["has_image"] = (
        1.0
        if image_path is not None
        else 0.0
    )

    # -------------------------------------------------------------------------
    # Audio
    # -------------------------------------------------------------------------

    if audio_path is not None:

        features.update(
            predictor.extract_audio_features(
                audio_path
            )
        )

    # -------------------------------------------------------------------------
    # Webcam / image
    # -------------------------------------------------------------------------

    image_modality_result = None
    webcam_calibration_used = False

    if image_path is not None:

        image_features = (
            predictor.extract_image_features(
                image_path
            )
        )

        features.update(
            image_features
        )

        image_modality_result = (
            run_webcam_calibrated_classifier(
                image_features
            )
        )

        if image_modality_result is not None:

            webcam_calibration_used = True

            # -------------------------------------------------------------
            # Make calibrated image probabilities available to fusion.
            #
            # Different training revisions may use different column names.
            # Supplying all aliases is harmless because the fusion vector
            # below selects ONLY predictor.feature_columns.
            # -------------------------------------------------------------

            for label in LABELS:

                probability = float(
                    image_modality_result[
                        "probabilities"
                    ][label]
                )

                features[
                    f"image_{label}_prob"
                ] = probability

                features[
                    f"webcam_{label}_prob"
                ] = probability

                features[
                    f"image_calibrated_{label}_prob"
                ] = probability

                features[
                    f"webcam_calibrated_{label}_prob"
                ] = probability

    # -------------------------------------------------------------------------
    # Build fusion vector using the EXACT saved schema
    # -------------------------------------------------------------------------

    row = {
        column:
            float(
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

    # -------------------------------------------------------------------------
    # Fusion prediction
    # -------------------------------------------------------------------------

    prediction = (
        predictor
        .fusion_model
        .predict(x)[0]
    )

    if hasattr(
        predictor.fusion_model,
        "predict_proba",
    ):

        probabilities = (
            predictor
            .fusion_model
            .predict_proba(x)[0]
        )

        classes = (
            predictor
            .fusion_model
            .classes_
        )

        probability_dict = {
            str(class_name):
                float(probability)
            for class_name, probability
            in zip(
                classes,
                probabilities,
            )
        }

    else:

        probability_dict = {
            label:
                0.10
            for label
            in LABELS
        }

        predicted_label = str(
            prediction
        )

        probability_dict[
            predicted_label
        ] = 0.70

        total = sum(
            probability_dict.values()
        )

        probability_dict = {
            label:
                value / total
            for label, value
            in probability_dict.items()
        }

    return {
        "prediction":
            str(prediction),

        "probabilities":
            probability_dict,

        "device":
            str(
                predictor.device
            ),

        "feature_dimension":
            int(
                x.shape[1]
            ),

        "used_modalities": {
            "text":
                True,

            "keystroke":
                True,

            "audio":
                audio_path
                is not None,

            "image":
                image_path
                is not None,
        },

        "webcam_calibration_used":
            webcam_calibration_used,

        "image_modality":
            image_modality_result,
    }


# =============================================================================
# RESULT NORMALISATION
# =============================================================================

def normalise_prediction_result(
    raw: Dict[str, Any],
) -> Dict[str, Any]:

    probabilities = {
        str(label):
            float(probability)
        for label, probability
        in raw[
            "probabilities"
        ].items()
    }

    ranked = sorted(
        probabilities.items(),
        key=lambda item: item[1],
        reverse=True,
    )

    current_state, confidence = (
        ranked[0]
    )

    if len(ranked) > 1:

        second_state, second_probability = (
            ranked[1]
        )

    else:

        second_state = "none"
        second_probability = 0.0

    confidence_gap = float(
        confidence
        - second_probability
    )

    confidence_level = (
        get_confidence_level(
            confidence_gap
        )
    )

    return {
        "prediction":
            current_state,

        "current_state":
            current_state,

        "confidence":
            float(
                confidence
            ),

        "confidence_percent":
            round(
                confidence * 100,
                2,
            ),

        "confidence_gap":
            confidence_gap,

        "confidence_level":
            confidence_level,

        "probabilities":
            probabilities,

        "webcam_calibration_used":
            raw.get(
                "webcam_calibration_used",
                False,
            ),

        "image_modality":
            raw.get(
                "image_modality"
            ),

        "technical_details": {
            "top_class":
                current_state,

            "second_class":
                second_state,

            "second_probability":
                float(
                    second_probability
                ),

            "confidence_gap":
                confidence_gap,

            "device":
                raw.get(
                    "device",
                    "unknown",
                ),

            "feature_dimension":
                raw.get(
                    "feature_dimension",
                    "unknown",
                ),

            "used_modalities":
                raw.get(
                    "used_modalities",
                    {},
                ),

            "webcam_calibration_used":
                raw.get(
                    "webcam_calibration_used",
                    False,
                ),
        },

        "device":
            raw.get(
                "device",
                "unknown",
            ),

        "feature_dimension":
            raw.get(
                "feature_dimension",
                "unknown",
            ),

        "used_modalities":
            raw.get(
                "used_modalities",
                {},
            ),
    }


# =============================================================================
# LOGGING
# =============================================================================

def log_prediction(
    text: str,
    keystroke_count: int,
    audio_available: bool,
    image_available: bool,
    result: Dict[str, Any],
) -> None:

    image_result = (
        result.get(
            "image_modality"
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
                datetime.now(
                    timezone.utc
                ).isoformat(),

                len(text),

                keystroke_count,

                audio_available,

                image_available,

                bool(
                    result.get(
                        "webcam_calibration_used",
                        False,
                    )
                ),

                image_result.get(
                    "current_state",
                    "",
                ),

                image_result.get(
                    "confidence",
                    "",
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


# =============================================================================
# LIVE PREDICTION ENDPOINT
# =============================================================================

@app.post("/predict_live")
async def predict_live(
    text: str = Form(""),
    keystroke_events: str = Form("[]"),
    image_frame: Optional[str] = Form(None),
    audio_chunk: Optional[UploadFile] = File(None),
) -> JSONResponse:

    text = text.strip()

    keystroke_count = (
        extract_keystroke_count(
            keystroke_events
        )
    )

    if len(text) < MIN_TEXT_CHARS:

        raise HTTPException(
            status_code=400,
            detail=(
                f"At least {MIN_TEXT_CHARS} "
                "text characters are required."
            ),
        )

    if keystroke_count < MIN_KEYPRESSES:

        raise HTTPException(
            status_code=400,
            detail=(
                f"At least {MIN_KEYPRESSES} "
                "keypresses are required."
            ),
        )

    image_path = (
        save_base64_image(
            image_frame
        )
    )

    audio_path = await save_audio_chunk(
        audio_chunk
    )

    try:

        raw_result = (
            run_prediction_backend(
                text=text,
                keystroke_events=keystroke_events,
                image_path=image_path,
                audio_path=audio_path,
            )
        )

        result = (
            normalise_prediction_result(
                raw_result
            )
        )

        log_prediction(
            text=text,
            keystroke_count=keystroke_count,
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
            detail=(
                "Prediction backend failed: "
                f"{exc}"
            ),
        ) from exc

    finally:

        # Prevent temporary browser captures from accumulating forever.
        safe_delete(
            image_path
        )

        safe_delete(
            audio_path
        )


# =============================================================================
# DIRECT EXECUTION
# =============================================================================

if __name__ == "__main__":

    uvicorn.run(
        "app:app",
        host="127.0.0.1",
        port=8000,
        reload=True,
    )
