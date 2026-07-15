# web_app/app.py

from __future__ import annotations

import base64
import csv
import json
import os
import sys
import uuid
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates


ROOT_DIR = Path(__file__).resolve().parents[1]
WEB_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = WEB_DIR / "uploads"
OUTPUT_DIR = WEB_DIR / "output"
LOG_FILE = OUTPUT_DIR / "live_predictions.csv"

UPLOAD_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

os.chdir(ROOT_DIR)

if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


MODEL_STATUS = {
    "text_model": False,
    "audio_model": False,
    "image_model": False,
    "keystroke_model": False,
    "fusion_model": False,
    "inference_backend": "fallback",
    "error": None,
}

predictor = None


def initialise_log_file() -> None:
    if LOG_FILE.exists():
        return

    with LOG_FILE.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "timestamp",
                "text_length",
                "keystroke_count",
                "audio_available",
                "image_available",
                "predicted_label",
                "confidence",
                "confidence_level",
                "confidence_gap",
                "feature_dimension",
                "used_modalities",
            ]
        )


def initialise_models() -> None:
    global predictor

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
                "inference_backend": "final_multimodal_inference.FinalMultimodalInference",
                "error": None,
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
                "error": str(exc),
            }
        )


@asynccontextmanager
async def lifespan(app: FastAPI):
    initialise_log_file()
    initialise_models()
    yield


app = FastAPI(
    title="SenseFuzeAI Live Fusion Web App",
    lifespan=lifespan,
)

app.mount("/static", StaticFiles(directory=WEB_DIR / "static"), name="static")
templates = Jinja2Templates(directory=WEB_DIR / "templates")


@app.get("/", response_class=HTMLResponse)
def index(request: Request) -> HTMLResponse:
    return templates.TemplateResponse(request=request, name="index.html")


@app.get("/health")
def health() -> Dict[str, Any]:
    return {
        "status": "ok",
        "service": "SenseFuzeAI Live Fusion",
        "timestamp": datetime.utcnow().isoformat(),
    }


@app.get("/model-status")
def model_status() -> Dict[str, Any]:
    return MODEL_STATUS


def save_base64_image(image_frame: Optional[str]) -> Optional[Path]:
    if not image_frame:
        return None

    try:
        _, encoded = image_frame.split(",", 1)
        image_path = UPLOAD_DIR / f"frame_{uuid.uuid4().hex}.jpg"
        image_path.write_bytes(base64.b64decode(encoded))
        return image_path
    except Exception:
        return None


async def save_audio_chunk(audio_chunk: Optional[UploadFile]) -> Optional[Path]:
    if audio_chunk is None:
        return None

    content = await audio_chunk.read()

    if not content:
        return None

    audio_path = UPLOAD_DIR / f"audio_{uuid.uuid4().hex}.webm"
    audio_path.write_bytes(content)
    return audio_path


def parse_keystrokes(keystroke_events: str) -> list[dict[str, Any]]:
    try:
        events = json.loads(keystroke_events)
        return events if isinstance(events, list) else []
    except Exception:
        return []


def extract_keystroke_count(keystroke_events: str) -> int:
    return sum(
        1 for event in parse_keystrokes(keystroke_events)
        if event.get("type") == "down"
    )


def live_keystroke_features(events: list[dict[str, Any]]) -> dict[str, float]:
    downs = [e for e in events if e.get("type") == "down"]
    down_times = [float(e.get("timestamp_perf", 0.0)) for e in downs]
    intervals = np.diff(down_times) if len(down_times) >= 2 else np.array([])

    hold_times = []
    active = {}

    for event in events:
        key = event.get("key")
        timestamp = float(event.get("timestamp_perf", 0.0))

        if event.get("type") == "down":
            active[key] = timestamp
        elif event.get("type") == "up" and key in active:
            hold_times.append(max(0.0, timestamp - active.pop(key)))

    hold_times = np.array(hold_times, dtype=float)

    duration = max(down_times) - min(down_times) if len(down_times) >= 2 else 0.0
    key_count = len(downs)

    return {
        "total_duration_sec": round(float(duration), 4),
        "keydown_count": int(key_count),
        "word_count": 0,
        "typing_speed_kps": round(float(key_count / duration), 4) if duration > 0 else 0.0,
        "typing_speed_wpm": 0.0,
        "delay_mean": round(float(np.mean(intervals)), 4) if len(intervals) else 0.0,
        "delay_std": round(float(np.std(intervals, ddof=1)), 4) if len(intervals) >= 2 else 0.0,
        "delay_min": round(float(np.min(intervals)), 4) if len(intervals) else 0.0,
        "delay_max": round(float(np.max(intervals)), 4) if len(intervals) else 0.0,
        "hold_mean": round(float(np.mean(hold_times)), 4) if len(hold_times) else 0.0,
        "hold_std": round(float(np.std(hold_times, ddof=1)), 4) if len(hold_times) >= 2 else 0.0,
        "pause_count_1000": int(sum(1 for x in intervals if x >= 1.0)),
        "pause_count_2000": int(sum(1 for x in intervals if x >= 2.0)),
        "pause_count_5000": int(sum(1 for x in intervals if x >= 5.0)),
        "pause_ratio_1000": round(float(sum(1 for x in intervals if x >= 1.0) / len(intervals)), 4) if len(intervals) else 0.0,
        "pause_ratio_2000": round(float(sum(1 for x in intervals if x >= 2.0) / len(intervals)), 4) if len(intervals) else 0.0,
        "mental_block_ratio_5000": round(float(sum(1 for x in intervals if x >= 5.0) / len(intervals)), 4) if len(intervals) else 0.0,
        "correction_count": int(sum(1 for e in downs if e.get("key") in {"backspace", "delete"})),
        "correction_ratio": round(float(sum(1 for e in downs if e.get("key") in {"backspace", "delete"}) / key_count), 4) if key_count else 0.0,
        "rhythm_consistency": round(float(1.0 / (1.0 + np.std(intervals))), 4) if len(intervals) else 1.0,
        "burstiness_proxy": round(float(np.std(intervals) / np.mean(intervals)), 4) if len(intervals) and np.mean(intervals) > 0 else 0.0,
        "fits_starts_index": round(float(sum(1 for x in intervals if x >= 1.0) / len(intervals)), 4) if len(intervals) else 0.0,
    }


def fallback_prediction(text: str) -> Dict[str, float]:
    lower_text = text.lower()

    scores = {
        "focused": 0.40,
        "distracted": 0.20,
        "fatigued": 0.20,
        "overloaded": 0.20,
    }

    if any(w in lower_text for w in ["tired", "sleepy", "exhausted", "fatigue"]):
        scores["fatigued"] += 0.25

    if any(w in lower_text for w in ["confused", "too much", "stress", "overload"]):
        scores["overloaded"] += 0.25

    if any(w in lower_text for w in ["distracted", "bored", "phone", "noise"]):
        scores["distracted"] += 0.25

    total = sum(scores.values())
    return {label: value / total for label, value in scores.items()}


def get_confidence_level(confidence_gap: float) -> str:
    if confidence_gap >= 0.35:
        return "High"
    if confidence_gap >= 0.15:
        return "Medium"
    return "Low"


def run_prediction_backend(
    text: str,
    keystroke_events: str,
    image_path: Optional[Path],
    audio_path: Optional[Path],
) -> Dict[str, Any]:
    if predictor is None:
        return {
            "probabilities": fallback_prediction(text),
            "device": "cpu",
            "feature_dimension": "fallback",
            "used_modalities": {
                "text": bool(text),
                "keystroke": bool(keystroke_events),
                "audio": audio_path is not None,
                "image": image_path is not None,
            },
        }

    features: dict[str, Any] = {}

    events = parse_keystrokes(keystroke_events)
    keystroke_features = live_keystroke_features(events)

    text_word_count = len(text.split())
    if keystroke_features.get("total_duration_sec", 0) > 0:
        keystroke_features["word_count"] = text_word_count
        keystroke_features["typing_speed_wpm"] = round(
            (text_word_count / keystroke_features["total_duration_sec"]) * 60,
            4,
        )

    features.update(keystroke_features)
    features.update(predictor.extract_text_features(text))

    if audio_path is not None:
        features.update(predictor.extract_audio_features(audio_path))

    if image_path is not None:
        features.update(predictor.extract_image_features(image_path))

    row = {
        col: float(features.get(col, 0.0))
        for col in predictor.feature_columns
    }

    x = pd.DataFrame([row], columns=predictor.feature_columns)

    prediction = predictor.fusion_model.predict(x)[0]

    if hasattr(predictor.fusion_model, "predict_proba"):
        probabilities = predictor.fusion_model.predict_proba(x)[0]
        classes = predictor.fusion_model.classes_
        probability_dict = {
            str(cls): float(prob)
            for cls, prob in zip(classes, probabilities)
        }
    else:
        probability_dict = {
            "focused": 0.25,
            "distracted": 0.25,
            "fatigued": 0.25,
            "overloaded": 0.25,
        }
        probability_dict[str(prediction)] = 0.70

    return {
        "prediction": str(prediction),
        "probabilities": probability_dict,
        "device": str(predictor.device),
        "feature_dimension": int(x.shape[1]),
        "used_modalities": {
            "text": True,
            "keystroke": True,
            "audio": audio_path is not None,
            "image": image_path is not None,
        },
    }


def normalise_prediction_result(raw: Dict[str, Any]) -> Dict[str, Any]:
    probabilities = {str(k): float(v) for k, v in raw["probabilities"].items()}
    ranked = sorted(probabilities.items(), key=lambda x: x[1], reverse=True)

    current_state, confidence = ranked[0]
    second_state, second_probability = ranked[1] if len(ranked) > 1 else ("none", 0.0)
    confidence_gap = confidence - second_probability

    return {
        "prediction": current_state,
        "current_state": current_state,
        "confidence": confidence,
        "confidence_percent": round(confidence * 100, 2),
        "confidence_gap": confidence_gap,
        "confidence_level": get_confidence_level(confidence_gap),
        "probabilities": probabilities,
        "technical_details": {
            "top_class": current_state,
            "second_class": second_state,
            "second_probability": second_probability,
            "confidence_gap": confidence_gap,
            "device": raw.get("device", "unknown"),
            "feature_dimension": raw.get("feature_dimension", "unknown"),
            "used_modalities": raw.get("used_modalities", {}),
        },
        "device": raw.get("device", "unknown"),
        "feature_dimension": raw.get("feature_dimension", "unknown"),
        "used_modalities": raw.get("used_modalities", {}),
    }


def log_prediction(
    text: str,
    keystroke_count: int,
    audio_available: bool,
    image_available: bool,
    result: Dict[str, Any],
) -> None:
    with LOG_FILE.open("a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                datetime.utcnow().isoformat(),
                len(text),
                keystroke_count,
                audio_available,
                image_available,
                result["current_state"],
                result["confidence"],
                result["confidence_level"],
                result["confidence_gap"],
                result["feature_dimension"],
                json.dumps(result.get("used_modalities", {})),
            ]
        )


@app.post("/predict_live")
async def predict_live(
    text: str = Form(""),
    keystroke_events: str = Form("[]"),
    image_frame: Optional[str] = Form(None),
    audio_chunk: Optional[UploadFile] = File(None),
) -> JSONResponse:
    text = text.strip()
    keystroke_count = extract_keystroke_count(keystroke_events)

    if len(text) < 20:
        raise HTTPException(
            status_code=400,
            detail="At least 20 text characters are required.",
        )

    if keystroke_count < 20:
        raise HTTPException(
            status_code=400,
            detail="At least 20 keypresses are required.",
        )

    image_path = save_base64_image(image_frame)
    audio_path = await save_audio_chunk(audio_chunk)

    raw_result = run_prediction_backend(
        text=text,
        keystroke_events=keystroke_events,
        image_path=image_path,
        audio_path=audio_path,
    )

    result = normalise_prediction_result(raw_result)

    log_prediction(
        text=text,
        keystroke_count=keystroke_count,
        audio_available=audio_path is not None,
        image_available=image_path is not None,
        result=result,
    )

    return JSONResponse(result)


if __name__ == "__main__":
    uvicorn.run(
        "app:app",
        host="127.0.0.1",
        port=8000,
        reload=True,
    )
