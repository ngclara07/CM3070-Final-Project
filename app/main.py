# === app/main.py ===
# SenseFuzeAI - FastAPI Application Entry Point
#
# Runtime modalities:
#   - keystroke
#   - text
#   - audio
#   - image
#
# Text modality uses:
#   model_artifacts/text_model.joblib
#
# Fusion method:
#   dynamic weighted late fusion
#
# Important:
#   Missing optional modalities do not contribute fusion weight.
#   Available modality weights are re-normalised dynamically.

from __future__ import annotations

import json
import uuid
from pathlib import Path
from typing import Dict, Optional

from fastapi import FastAPI, File, Form, Request, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from app.fusion import (
    audio_to_behaviour_scores,
    explain_fusion_inputs,
    fuse_predictions_with_metadata,
    get_fusion_weights,
    image_to_behaviour_scores,
    normalize_scores,
    prediction_from_scores,
    uniform_scores,
)

from app.models.audio_models import analyze_audio_file
from app.models.image_model import analyze_image_file
from app.models.keystroke_model import predict_keystroke_behaviour
from app.models.text_model import analyze_text


# ============================================================
# Application paths
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

STATIC_DIR = BASE_DIR / "app" / "static"
TEMPLATES_DIR = BASE_DIR / "app" / "templates"

TEMP_UPLOAD_DIR = BASE_DIR / "data" / "temp_uploads"
TEMP_UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# FastAPI setup
# ============================================================

app = FastAPI(title="SenseFuzeAI Multimodal Behaviour Analyzer")

app.mount(
    "/static",
    StaticFiles(directory=str(STATIC_DIR)),
    name="static",
)

templates = Jinja2Templates(directory=str(TEMPLATES_DIR))


# ============================================================
# Score utilities
# ============================================================

def clean_scores(scores: Dict[str, float] | None) -> Dict[str, float]:
    """
    Normalise behaviour scores using the shared fusion utility.
    """
    return normalize_scores(scores)


def extract_audio_scores(audio_result: Dict) -> Dict[str, float]:
    """
    Extract behaviour scores from the audio analysis output.

    Priority:
        1. trained model behaviour scores / probabilities
        2. fallback YAMNet semantic labels
        3. uniform scores
    """
    if not audio_result:
        return uniform_scores()

    for key in [
        "behaviour_scores",
        "prediction_scores",
        "scores",
        "class_probabilities",
    ]:
        maybe_scores = audio_result.get(key)

        if isinstance(maybe_scores, dict):
            return clean_scores(maybe_scores)

    labels = (
        audio_result.get("yamnet_labels")
        or audio_result.get("labels")
        or []
    )

    if isinstance(labels, list):
        return audio_to_behaviour_scores(labels)

    return uniform_scores()


def extract_image_scores(image_result: Dict) -> Dict[str, float]:
    """
    Extract behaviour scores from the image analysis output.

    Priority:
        1. trained model behaviour scores / probabilities
        2. visual fused scores
        3. fallback caption-based heuristic
        4. uniform scores
    """
    if not image_result:
        return uniform_scores()

    for key in [
        "behaviour_scores",
        "prediction_scores",
        "scores",
        "class_probabilities",
    ]:
        maybe_scores = image_result.get(key)

        if isinstance(maybe_scores, dict):
            return clean_scores(maybe_scores)

    fused = image_result.get("fused_prediction", {})

    if isinstance(fused, dict):
        fused_scores = fused.get("visual_fused_scores")

        if isinstance(fused_scores, dict):
            return clean_scores(fused_scores)

    caption = (
        image_result.get("behaviour_caption")
        or image_result.get("behaviour_aware_caption")
        or image_result.get("caption")
        or image_result.get("scene_description")
        or ""
    )

    return image_to_behaviour_scores(caption)


def extract_text_scores(text_result: Dict) -> Dict[str, float]:
    """
    Extract behaviour scores from the trained text model output.

    Expected updated text model fields:
        behaviour_scores
        predicted_behaviour
        behaviour_confidence
        sentiment_label
        sentiment_score
    """
    if not text_result:
        return uniform_scores()

    maybe_scores = text_result.get("behaviour_scores")

    if isinstance(maybe_scores, dict):
        return clean_scores(maybe_scores)

    return uniform_scores()


def get_image_caption_fields(image_result: Dict) -> tuple[str, str]:
    """
    Extract generic and behaviour-aware image captions from the vision result.
    """
    generic_caption = (
        image_result.get("generic_caption")
        or image_result.get("caption")
        or image_result.get("scene_description")
        or ""
    )

    behaviour_caption = (
        image_result.get("behaviour_caption")
        or image_result.get("behaviour_aware_caption")
        or image_result.get("behaviour_description")
        or ""
    )

    return generic_caption, behaviour_caption


# ============================================================
# File utilities
# ============================================================

async def save_temp_upload_file(
    upload_file: UploadFile,
    upload_dir: Path,
    prefix: str,
) -> Path:
    """
    Save an uploaded file temporarily using a unique safe filename.

    The returned file path is deleted after analysis.
    """
    original_filename = Path(upload_file.filename or "uploaded_file").name
    suffix = Path(original_filename).suffix

    safe_filename = f"{prefix}_{uuid.uuid4().hex}{suffix}"
    output_path = upload_dir / safe_filename

    content = await upload_file.read()

    with output_path.open("wb") as file:
        file.write(content)

    return output_path


def delete_temp_file(path: Optional[Path]) -> None:
    """
    Delete a temporary upload file safely.
    """
    if path is None:
        return

    try:
        if path.exists() and path.is_file():
            path.unlink()
    except Exception:
        pass


def has_uploaded_file(upload_file: Optional[UploadFile]) -> bool:
    """
    Check whether an optional file upload was actually provided.
    """
    return upload_file is not None and bool(upload_file.filename)


# ============================================================
# Routes
# ============================================================

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    """
    Render the main web interface.
    """
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={"request": request},
    )


@app.get("/health")
async def health():
    """
    Lightweight health/status endpoint.
    """
    return {
        "status": "ok",
        "system": "SenseFuzeAI",
        "modalities": ["keystroke", "text", "audio", "image"],
        "required_runtime_modalities": ["keystroke", "text"],
        "optional_runtime_modalities": ["audio", "image"],
        "text_model": "model_artifacts/text_model.joblib",
        "text_sentiment": "supporting_rule_based_lexical_sentiment",
        "fusion": "dynamic_weighted_late_fusion",
        "base_fusion_weights": get_fusion_weights(),
        "missing_modality_policy": (
            "Missing optional modalities receive zero effective fusion weight. "
            "Available modality weights are re-normalised dynamically."
        ),
        "trained_fusion_model": False,
    }


@app.post("/analyze")
async def analyze(
    typed_text: str = Form(...),
    keystroke_events_json: str = Form(...),
    audio_file: Optional[UploadFile] = File(default=None),
    image_file: Optional[UploadFile] = File(default=None),
):
    """
    Run the full multimodal analysis pipeline.

    Runtime behaviour:
        - text and keystroke are treated as baseline modalities
        - audio and image are optional
        - missing optional modalities are excluded from effective fusion weights
    """
    audio_path: Optional[Path] = None
    image_path: Optional[Path] = None

    try:
        typed_text = str(typed_text or "").strip()

        # ====================================================
        # 1. Keystroke modality
        # ====================================================

        try:
            events = json.loads(keystroke_events_json) if keystroke_events_json else []

            if not isinstance(events, list):
                raise ValueError("keystroke_events_json must decode to a list.")

        except Exception as error:
            events = []
            keystroke_json_error = str(error)
        else:
            keystroke_json_error = ""

        try:
            keystroke_prediction, keystroke_scores, keystroke_features = (
                predict_keystroke_behaviour(events)
            )
            keystroke_scores = clean_scores(keystroke_scores)

            if keystroke_json_error:
                keystroke_features["keystroke_json_warning"] = keystroke_json_error

        except Exception as error:
            keystroke_prediction = "unavailable"
            keystroke_scores = uniform_scores()
            keystroke_features = {
                "error": str(error),
                "captured_events": len(events),
            }

            if keystroke_json_error:
                keystroke_features["keystroke_json_error"] = keystroke_json_error

        # ====================================================
        # 2. Text modality
        # ====================================================

        try:
            text_result = analyze_text(typed_text)
            text_scores = extract_text_scores(text_result)

            text_prediction = text_result.get(
                "predicted_behaviour",
                prediction_from_scores(text_scores),
            )

            text_confidence = float(
                text_result.get(
                    "behaviour_confidence",
                    max(text_scores.values()),
                )
            )

            text_sentiment = {
                "sentiment_label": text_result.get("sentiment_label", "unknown"),
                "sentiment_score": text_result.get("sentiment_score", 0.0),
                "sentiment_method": text_result.get("sentiment_method", "unknown"),
                "positive_count": text_result.get("sentiment_positive_count", 0),
                "negative_count": text_result.get("sentiment_negative_count", 0),
                "positive_hits": text_result.get("sentiment_positive_hits", []),
                "negative_hits": text_result.get("sentiment_negative_hits", []),
            }

        except Exception as error:
            text_result = {
                "status": "error",
                "label": "unavailable",
                "score": 0.0,
                "sentiment_label": "unavailable",
                "sentiment_score": 0.0,
                "predicted_behaviour": "unavailable",
                "behaviour_scores": uniform_scores(),
                "error": str(error),
            }

            text_scores = uniform_scores()
            text_prediction = "unavailable"
            text_confidence = 0.0

            text_sentiment = {
                "sentiment_label": "unavailable",
                "sentiment_score": 0.0,
                "sentiment_method": "unavailable",
                "positive_count": 0,
                "negative_count": 0,
                "positive_hits": [],
                "negative_hits": [],
            }

        # ====================================================
        # 3. Audio modality
        # ====================================================

        audio_result = {
            "status": "not_provided",
            "labels": [],
            "yamnet_labels": [],
        }

        audio_scores = uniform_scores()
        audio_prediction = "not_provided"

        audio_uploaded = has_uploaded_file(audio_file)
        audio_analyzed = False

        if audio_uploaded:
            try:
                audio_path = await save_temp_upload_file(
                    upload_file=audio_file,
                    upload_dir=TEMP_UPLOAD_DIR,
                    prefix="audio",
                )

                audio_result = analyze_audio_file(str(audio_path))
                audio_result["status"] = "analyzed"
                audio_result["uploaded_file"] = Path(audio_file.filename or "").name
                audio_result["temporary_file_used"] = audio_path.name

                audio_scores = extract_audio_scores(audio_result)

                audio_prediction = audio_result.get(
                    "predicted_label",
                    audio_result.get(
                        "predicted_behaviour",
                        prediction_from_scores(audio_scores),
                    ),
                )

                audio_analyzed = True

            except Exception as error:
                audio_result = {
                    "status": "error",
                    "error": str(error),
                    "uploaded_file": Path(audio_file.filename or "").name,
                    "labels": [],
                    "yamnet_labels": [],
                }

                audio_scores = uniform_scores()
                audio_prediction = "unavailable"
                audio_analyzed = False

        # ====================================================
        # 4. Image modality
        # ====================================================

        image_result = {
            "status": "not_provided",
        }

        image_scores = uniform_scores()
        image_prediction = "not_provided"

        image_caption = ""
        image_generic_caption = ""
        image_behaviour_caption = ""

        image_uploaded = has_uploaded_file(image_file)
        image_analyzed = False

        if image_uploaded:
            try:
                image_path = await save_temp_upload_file(
                    upload_file=image_file,
                    upload_dir=TEMP_UPLOAD_DIR,
                    prefix="image",
                )

                image_result = analyze_image_file(str(image_path))
                image_result["status"] = "analyzed"
                image_result["uploaded_file"] = Path(image_file.filename or "").name
                image_result["temporary_file_used"] = image_path.name

                image_generic_caption, image_behaviour_caption = get_image_caption_fields(
                    image_result
                )

                image_caption = image_generic_caption

                image_scores = extract_image_scores(image_result)

                image_prediction = image_result.get(
                    "predicted_label",
                    image_result.get(
                        "predicted_behaviour",
                        prediction_from_scores(image_scores),
                    ),
                )

                image_analyzed = True

            except Exception as error:
                image_result = {
                    "status": "error",
                    "error": str(error),
                    "uploaded_file": Path(image_file.filename or "").name,
                }

                image_scores = uniform_scores()
                image_prediction = "unavailable"

                image_caption = ""
                image_generic_caption = ""
                image_behaviour_caption = ""

                image_analyzed = False

        # ====================================================
        # 5. Runtime fusion with dynamic missing-modality policy
        # ====================================================

        keystroke_available = keystroke_prediction not in {
            "unavailable",
            "insufficient_data",
            "",
            None,
        }

        text_available = text_prediction not in {
            "unavailable",
            "",
            None,
        }

        audio_available = (
            audio_analyzed
            and audio_result.get("status") == "analyzed"
            and audio_prediction not in {
                "unavailable",
                "not_provided",
                "",
                None,
            }
        )

        image_available = (
            image_analyzed
            and image_result.get("status") == "analyzed"
            and image_prediction not in {
                "unavailable",
                "not_provided",
                "",
                None,
            }
        )

        fusion_output = fuse_predictions_with_metadata(
            keystroke_scores=keystroke_scores,
            text_scores=text_scores,
            audio_scores=audio_scores,
            image_scores=image_scores,
            keystroke_available=keystroke_available,
            text_available=text_available,
            audio_available=audio_available,
            image_available=image_available,
            use_trained_fusion=False,
        )

        final_scores = fusion_output["final_scores"]
        final_prediction = fusion_output["final_prediction"]
        final_confidence = fusion_output["final_confidence"]

        modality_scores = fusion_output.get(
            "modality_scores",
            explain_fusion_inputs(
                keystroke_scores=keystroke_scores,
                text_scores=text_scores,
                audio_scores=audio_scores,
                image_scores=image_scores,
            ),
        )

        # ====================================================
        # 6. Response payload
        # ====================================================

        response_payload = {
            "final_prediction": final_prediction,
            "final_scores": final_scores,
            "final_confidence": final_confidence,

            "keystroke_prediction": keystroke_prediction,
            "keystroke_scores": keystroke_scores,
            "keystroke_features": keystroke_features,

            "text_prediction": text_prediction,
            "text_confidence": text_confidence,
            "text_result": text_result,
            "text_scores": text_scores,
            "text_sentiment": text_sentiment,

            "audio_prediction": audio_prediction,
            "audio_result": audio_result,
            "audio_scores": audio_scores,

            "image_prediction": image_prediction,
            "image_result": image_result,
            "image_caption": image_caption,
            "image_generic_caption": image_generic_caption,
            "image_behaviour_caption": image_behaviour_caption,
            "image_scores": image_scores,

            "modality_scores": modality_scores,

            "fusion_method": fusion_output["fusion_method"],
            "fusion_weights": fusion_output["effective_fusion_weights"],
            "base_fusion_weights": fusion_output["base_fusion_weights"],
            "effective_fusion_weights": fusion_output["effective_fusion_weights"],
            "uploaded_modalities": fusion_output["uploaded_modalities"],
            "missing_modalities": fusion_output["missing_modalities"],
            "fusion_note": fusion_output["fusion_note"],

            "modality_availability": {
                "keystroke": keystroke_available,
                "text": text_available,
                "audio": audio_available,
                "image": image_available,
            },

            "uploaded_files": {
                "audio": Path(audio_file.filename).name
                if audio_uploaded and audio_file and audio_file.filename
                else None,
                "image": Path(image_file.filename).name
                if image_uploaded and image_file and image_file.filename
                else None,
            },

            "trained_fusion_model_used": False,
        }

        return JSONResponse(response_payload)

    except Exception as error:
        return JSONResponse(
            {
                "error": str(error),
                "message": "Analysis failed inside the multimodal pipeline.",
            },
            status_code=500,
        )

    finally:
        delete_temp_file(audio_path)
        delete_temp_file(image_path)
