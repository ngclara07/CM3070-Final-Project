"""
final_multimodal_inference.py

SenseFuzeAI
Canonical stateless multimodal behavioural-state inference.

Deployment changes:
- fusion artifact is loaded immediately
- MPNet/WavLM/CLIP are loaded only when required
- local pretrained directories are preferred
- Hugging Face IDs are fallback sources
- optional low-memory encoder unloading is supported
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import statistics
import threading
from pathlib import Path
from typing import Any

import joblib
import librosa
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from PIL import Image

from temporal_fusion import (
    LABELS,
    PROBABILITY_SUM_TOLERANCE,
    normalise_probability_dict,
    summarise_probability_dict,
    validate_probability_distribution,
)


# ============================================================
# PATHS
# ============================================================

ROOT_DIR = Path(__file__).resolve().parent

FUSION_MODEL_PATH = (
    ROOT_DIR
    / "models"
    / "fusion_demo"
    / "fusion_pipeline.joblib"
)

FUSION_FEATURE_COLUMNS_PATH = (
    ROOT_DIR
    / "models"
    / "fusion_demo"
    / "feature_columns.json"
)

LOCAL_TEXT_MODEL_PATH = (
    ROOT_DIR
    / "models"
    / "all-mpnet-base-v2"
)

LOCAL_WAVLM_MODEL_PATH = (
    ROOT_DIR
    / "models"
    / "wavlm-base-plus"
)

LOCAL_CLIP_MODEL_PATH = (
    ROOT_DIR
    / "models"
    / "clip-vit-large-patch14"
)

WEBCAM_IMAGE_MODEL_PATH = (
    ROOT_DIR
    / "models"
    / "image_demo"
    / "image_pipeline_webcam_calibrated.joblib"
)

WEBCAM_IMAGE_FEATURE_COLUMNS_PATH = (
    ROOT_DIR
    / "models"
    / "image_demo"
    / "feature_columns.json"
)

WEBCAM_IMAGE_METADATA_PATH = (
    ROOT_DIR
    / "models"
    / "image_demo"
    / "webcam_calibrated_metadata.json"
)


# ============================================================
# REMOTE FALLBACK IDS
# ============================================================

TEXT_MODEL_ID = os.environ.get(
    "SENSEFUZE_TEXT_MODEL",
    "sentence-transformers/all-mpnet-base-v2",
)

WAVLM_MODEL_ID = os.environ.get(
    "SENSEFUZE_WAVLM_MODEL",
    "microsoft/wavlm-base-plus",
)

CLIP_MODEL_ID = os.environ.get(
    "SENSEFUZE_CLIP_MODEL",
    "openai/clip-vit-large-patch14",
)


# ============================================================
# CONFIGURATION
# ============================================================

TARGET_SR = 16000
MAX_AUDIO_SECONDS = 20

torch.set_num_threads(
    max(
        1,
        int(
            os.environ.get(
                "SENSEFUZE_TORCH_THREADS",
                "1",
            )
        ),
    )
)

LOW_MEMORY_MODE = (
    os.environ.get(
        "SENSEFUZE_LOW_MEMORY",
        (
            "1"
            if os.environ.get("RENDER")
            else "0"
        ),
    )
    .strip()
    .lower()
    in {"1", "true", "yes", "on"}
)

CLASSES = list(LABELS)

WEBCAM_PROBABILITY_COLUMNS = [
    "image_webcam_focused_prob",
    "image_webcam_distracted_prob",
    "image_webcam_fatigued_prob",
    "image_webcam_overloaded_prob",
]

WEBCAM_CONFIDENCE_COLUMNS = [
    "image_webcam_top_probability",
    "image_webcam_confidence_gap",
]

EXPECTED_WEBCAM_FUSION_COLUMNS = {
    *WEBCAM_PROBABILITY_COLUMNS,
    *WEBCAM_CONFIDENCE_COLUMNS,
}


# ============================================================
# GENERAL UTILITIES
# ============================================================

def get_device() -> torch.device:
    # Render standard CPU services do not provide a CUDA GPU.
    return torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )


def clean_float(value: Any) -> float:
    try:
        result = float(value)

        if np.isfinite(result):
            return result
    except (
        TypeError,
        ValueError,
        OverflowError,
    ):
        pass

    return 0.0


def load_json_list(
    path: Path,
) -> list[str]:

    if not path.exists():
        raise FileNotFoundError(
            f"JSON list file not found:\n{path}"
        )

    with path.open(
        "r",
        encoding="utf-8",
    ) as handle:
        value = json.load(handle)

    if not isinstance(value, list):
        raise ValueError(
            f"Expected JSON list:\n{path}"
        )

    result = [
        str(item).strip()
        for item in value
    ]

    if not result or any(
        not item for item in result
    ):
        raise ValueError(
            f"Invalid JSON feature list:\n{path}"
        )

    return result


def load_optional_json_object(
    path: Path,
) -> dict[str, Any]:

    if not path.exists():
        return {}

    try:
        with path.open(
            "r",
            encoding="utf-8",
        ) as handle:
            value = json.load(handle)

        return (
            value
            if isinstance(value, dict)
            else {}
        )

    except Exception:
        return {}


def normalise_label(value: Any) -> str:
    return (
        ""
        if value is None
        else str(value)
            .strip()
            .lower()
    )


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
            normalise_label(value)
            for value in classes
        ]

    if hasattr(
        model,
        "named_steps",
    ):
        for step in reversed(
            list(
                model.named_steps
                .values()
            )
        ):
            classes = getattr(
                step,
                "classes_",
                None,
            )

            if classes is not None:
                return [
                    normalise_label(value)
                    for value in classes
                ]

    return []


def softmax(
    values: np.ndarray,
) -> np.ndarray:

    values = np.asarray(
        values,
        dtype=np.float64,
    ).reshape(-1)

    if values.size == 0:
        raise ValueError(
            "Cannot softmax an empty array."
        )

    values = np.nan_to_num(values)

    values -= np.max(values)

    exp_values = np.exp(values)

    total = float(
        exp_values.sum()
    )

    if total <= 0 or not np.isfinite(total):
        return (
            np.ones_like(exp_values)
            / len(exp_values)
        )

    return exp_values / total


def resolve_model_source(
    local_path: Path,
    remote_id: str,
) -> str:

    if local_path.exists():
        return str(local_path)

    return remote_id


def release_memory() -> None:
    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ============================================================
# INFERENCE
# ============================================================

class FinalMultimodalInference:

    def __init__(self) -> None:

        # ----------------------------------------------
        # Only project-specific artifacts are mandatory
        # at predictor construction time.
        # ----------------------------------------------

        if not FUSION_MODEL_PATH.exists():
            raise FileNotFoundError(
                "Fusion model artifact is missing:\n"
                f"{FUSION_MODEL_PATH}\n\n"
                "This trained project-specific artifact "
                "must exist in the deployed repository "
                "or attached storage."
            )

        if not FUSION_FEATURE_COLUMNS_PATH.exists():
            raise FileNotFoundError(
                "Fusion feature schema is missing:\n"
                f"{FUSION_FEATURE_COLUMNS_PATH}"
            )

        self.device = get_device()

        self.fusion_model = joblib.load(
            FUSION_MODEL_PATH
        )

        self.feature_columns = (
            load_json_list(
                FUSION_FEATURE_COLUMNS_PATH
            )
        )

        self.expected_text_embedding_dim = (
            self._schema_embedding_dimension(
                self.feature_columns,
                "text_mpnet_emb_",
            )
        )

        self.expected_wavlm_embedding_dim = (
            self._schema_embedding_dimension(
                self.feature_columns,
                "audio_wavlm_emb_",
            )
        )

        self.expected_clip_embedding_dim = (
            self._schema_embedding_dimension(
                self.feature_columns,
                "image_clip_emb_",
            )
        )

        # ----------------------------------------------
        # Heavy encoders deliberately begin unloaded.
        # ----------------------------------------------

        self.text_model = None

        self.wavlm_extractor = None
        self.wavlm_model = None

        self.clip_processor = None
        self.clip_model = None

        self._encoder_lock = (
            threading.RLock()
        )

        self.webcam_image_model = None

        self.webcam_image_feature_columns: list[
            str
        ] = []

        self.webcam_image_metadata: dict[
            str,
            Any
        ] = {}

        self._load_webcam_calibration_model()

        self._validate_model_classes(
            self.fusion_model,
            "Fusion classifier",
        )

        self._validate_model_feature_schema(
            self.fusion_model,
            self.feature_columns,
            "Fusion classifier",
        )

        print(
            "FinalMultimodalInference core initialised.",
            flush=True,
        )

        print(
            f"Device: {self.device}",
            flush=True,
        )

        print(
            f"Fusion features: {len(self.feature_columns)}",
            flush=True,
        )

        print(
            f"Low-memory mode: {LOW_MEMORY_MODE}",
            flush=True,
        )


    # ========================================================
    # STATUS
    # ========================================================

    def runtime_status(
        self,
    ) -> dict[str, Any]:

        return {
            "fusion_model_loaded":
                self.fusion_model is not None,

            "text_model_loaded":
                self.text_model is not None,

            "audio_model_loaded":
                self.wavlm_model is not None,

            "image_model_loaded":
                self.clip_model is not None,

            "webcam_calibration_loaded":
                self.webcam_image_model
                is not None,

            "device":
                str(self.device),

            "low_memory_mode":
                LOW_MEMORY_MODE,

            "text_model_source":
                resolve_model_source(
                    LOCAL_TEXT_MODEL_PATH,
                    TEXT_MODEL_ID,
                ),

            "wavlm_model_source":
                resolve_model_source(
                    LOCAL_WAVLM_MODEL_PATH,
                    WAVLM_MODEL_ID,
                ),

            "clip_model_source":
                resolve_model_source(
                    LOCAL_CLIP_MODEL_PATH,
                    CLIP_MODEL_ID,
                ),
        }


    # ========================================================
    # SCHEMA VALIDATION
    # ========================================================

    @staticmethod
    def _schema_embedding_dimension(
        columns: list[str],
        prefix: str,
    ) -> int | None:

        matching = [
            column
            for column in columns
            if column.startswith(prefix)
        ]

        if not matching:
            return None

        indices = []

        for column in matching:
            suffix = column[
                len(prefix):
            ]

            if not suffix.isdigit():
                raise ValueError(
                    "Invalid embedding feature "
                    f"column: {column}"
                )

            indices.append(
                int(suffix)
            )

        indices.sort()

        expected = list(
            range(indices[-1] + 1)
        )

        if indices != expected:
            raise ValueError(
                f"Non-contiguous schema: {prefix}"
            )

        return indices[-1] + 1


    @staticmethod
    def _validate_model_feature_schema(
        model: Any,
        columns: list[str],
        model_name: str,
    ) -> None:

        n_features = getattr(
            model,
            "n_features_in_",
            None,
        )

        if (
            n_features is not None
            and int(n_features)
            != len(columns)
        ):
            raise ValueError(
                f"{model_name} feature-count mismatch. "
                f"Model={n_features}, "
                f"schema={len(columns)}"
            )

        feature_names = getattr(
            model,
            "feature_names_in_",
            None,
        )

        if feature_names is not None:
            model_columns = [
                str(value)
                for value in feature_names
            ]

            if model_columns != columns:
                raise ValueError(
                    f"{model_name} feature-name/"
                    "order mismatch."
                )


    @staticmethod
    def _validate_model_classes(
        model: Any,
        model_name: str,
    ) -> list[str]:

        classes = get_model_classes(
            model
        )

        if not classes:
            raise ValueError(
                f"{model_name} exposes no "
                "class ordering."
            )

        if set(classes) != set(LABELS):
            raise ValueError(
                f"{model_name} class mismatch. "
                f"Expected={list(LABELS)}, "
                f"observed={classes}"
            )

        return classes


    # ========================================================
    # LAZY ENCODERS
    # ========================================================

    def _ensure_text_model(self):
        with self._encoder_lock:
            if self.text_model is None:
                from sentence_transformers import (
                    SentenceTransformer,
                )

                source = resolve_model_source(
                    LOCAL_TEXT_MODEL_PATH,
                    TEXT_MODEL_ID,
                )

                print(
                    "[inference] Loading MPNet: "
                    f"{source}",
                    flush=True,
                )

                self.text_model = (
                    SentenceTransformer(
                        source,
                        device=str(self.device),
                    )
                )

        return self.text_model


    def _ensure_audio_model(self):
        with self._encoder_lock:
            if (
                self.wavlm_extractor is None
                or self.wavlm_model is None
            ):
                from transformers import (
                    Wav2Vec2FeatureExtractor,
                    WavLMModel,
                )

                source = resolve_model_source(
                    LOCAL_WAVLM_MODEL_PATH,
                    WAVLM_MODEL_ID,
                )

                print(
                    "[inference] Loading WavLM: "
                    f"{source}",
                    flush=True,
                )

                self.wavlm_extractor = (
                    Wav2Vec2FeatureExtractor
                    .from_pretrained(source)
                )

                self.wavlm_model = (
                    WavLMModel
                    .from_pretrained(source)
                    .to(self.device)
                )

                self.wavlm_model.eval()

        return (
            self.wavlm_extractor,
            self.wavlm_model,
        )


    def _ensure_clip_model(self):
        with self._encoder_lock:
            if (
                self.clip_processor is None
                or self.clip_model is None
            ):
                from transformers import (
                    CLIPModel,
                    CLIPProcessor,
                )

                source = resolve_model_source(
                    LOCAL_CLIP_MODEL_PATH,
                    CLIP_MODEL_ID,
                )

                print(
                    "[inference] Loading CLIP: "
                    f"{source}",
                    flush=True,
                )

                self.clip_processor = (
                    CLIPProcessor
                    .from_pretrained(source)
                )

                self.clip_model = (
                    CLIPModel
                    .from_pretrained(source)
                    .to(self.device)
                )

                self.clip_model.eval()

        return (
            self.clip_processor,
            self.clip_model,
        )


    def _release_text_model(self):
        if LOW_MEMORY_MODE:
            self.text_model = None
            release_memory()


    def _release_audio_model(self):
        if LOW_MEMORY_MODE:
            self.wavlm_model = None
            self.wavlm_extractor = None
            release_memory()


    def _release_clip_model(self):
        if LOW_MEMORY_MODE:
            self.clip_model = None
            self.clip_processor = None
            release_memory()


    # ========================================================
    # OPTIONAL WEBCAM CLASSIFIER
    # ========================================================

    def _load_webcam_calibration_model(
        self,
    ) -> None:

        required = {
            column
            for column
            in self.feature_columns
            if column.startswith(
                "image_webcam_"
            )
        }

        if not required:
            return

        if (
            required
            != EXPECTED_WEBCAM_FUSION_COLUMNS
        ):
            raise ValueError(
                "Unsupported/incomplete "
                "image_webcam_* fusion schema."
            )

        if not WEBCAM_IMAGE_MODEL_PATH.exists():
            raise FileNotFoundError(
                "Fusion schema requires webcam "
                "calibration model, but it is missing:\n"
                f"{WEBCAM_IMAGE_MODEL_PATH}"
            )

        if not (
            WEBCAM_IMAGE_FEATURE_COLUMNS_PATH
            .exists()
        ):
            raise FileNotFoundError(
                "Webcam feature schema missing:\n"
                f"{WEBCAM_IMAGE_FEATURE_COLUMNS_PATH}"
            )

        self.webcam_image_feature_columns = (
            load_json_list(
                WEBCAM_IMAGE_FEATURE_COLUMNS_PATH
            )
        )

        self.webcam_image_model = (
            joblib.load(
                WEBCAM_IMAGE_MODEL_PATH
            )
        )

        self.webcam_image_metadata = (
            load_optional_json_object(
                WEBCAM_IMAGE_METADATA_PATH
            )
        )

        self._validate_model_classes(
            self.webcam_image_model,
            (
                "Webcam-calibrated "
                "image classifier"
            ),
        )

        self._validate_model_feature_schema(
            self.webcam_image_model,
            self.webcam_image_feature_columns,
            (
                "Webcam-calibrated "
                "image classifier"
            ),
        )


    # ========================================================
    # PROBABILITIES
    # ========================================================

    def _predict_probability_dict(
        self,
        model: Any,
        X: pd.DataFrame,
        *,
        model_name: str,
    ) -> dict[str, float]:

        classes = get_model_classes(
            model
        )

        if hasattr(
            model,
            "predict_proba",
        ):
            values = np.asarray(
                model.predict_proba(X)[0],
                dtype=np.float64,
            ).reshape(-1)

        elif hasattr(
            model,
            "decision_function",
        ):
            scores = np.asarray(
                model.decision_function(X),
                dtype=np.float64,
            )

            if scores.ndim > 1:
                scores = scores[0]

            values = softmax(scores)

        else:
            predicted = normalise_label(
                model.predict(X)[0]
            )

            values = np.asarray(
                [
                    1.0
                    if label == predicted
                    else 0.0
                    for label in classes
                ],
                dtype=np.float64,
            )

        probabilities = (
            normalise_probability_dict(
                {
                    class_name:
                        clean_float(probability)
                    for class_name, probability
                    in zip(classes, values)
                },
                labels=LABELS,
            )
        )

        validation = (
            validate_probability_distribution(
                probabilities,
                labels=LABELS,
                tolerance=(
                    PROBABILITY_SUM_TOLERANCE
                ),
            )
        )

        if not validation["valid"]:
            raise ValueError(
                f"{model_name} generated "
                "invalid probabilities."
            )

        return probabilities


    # ========================================================
    # KEYSTROKES
    # ========================================================

    def extract_keystroke_features(
        self,
        keystroke_json_path: Path,
    ) -> dict[str, Any]:

        path = Path(
            keystroke_json_path
        )

        if not path.exists():
            raise FileNotFoundError(
                f"Keystroke JSON missing:\n{path}"
            )

        with path.open(
            "r",
            encoding="utf-8",
        ) as handle:
            data = json.load(handle)

        raw_features = data.get(
            "features"
        )

        if not isinstance(
            raw_features,
            dict,
        ):
            raise ValueError(
                "Keystroke JSON requires "
                "a 'features' object."
            )

        output = {}

        for key, value in (
            raw_features.items()
        ):
            name = str(key)
            numeric = clean_float(value)

            output[name] = numeric

            if not name.startswith(
                "keystroke_"
            ):
                output[
                    f"keystroke_{name}"
                ] = numeric

        return output


    # ========================================================
    # TEXT
    # ========================================================

    def extract_text_features(
        self,
        text: str,
    ) -> dict[str, float]:

        text = str(text).strip()

        if not text:
            raise ValueError(
                "Text input is empty."
            )

        model = (
            self._ensure_text_model()
        )

        try:
            embedding = model.encode(
                [text],
                normalize_embeddings=True,
                show_progress_bar=False,
                convert_to_numpy=True,
            )[0]

            embedding = np.asarray(
                embedding,
                dtype=np.float32,
            ).reshape(-1)

        finally:
            self._release_text_model()

        if (
            self.expected_text_embedding_dim
            is not None
            and embedding.size
            != self.expected_text_embedding_dim
        ):
            raise ValueError(
                "MPNet embedding dimension "
                "does not match fusion schema. "
                f"Expected="
                f"{self.expected_text_embedding_dim}, "
                f"observed={embedding.size}"
            )

        return {
            f"text_mpnet_emb_{index}":
                clean_float(value)
            for index, value
            in enumerate(embedding)
        }


    # ========================================================
    # AUDIO
    # ========================================================

    def extract_audio_features(
        self,
        audio_path: Path,
    ) -> dict[str, float]:

        path = Path(audio_path)

        if not path.exists():
            raise FileNotFoundError(
                f"Audio file missing:\n{path}"
            )

        waveform, sample_rate = (
            librosa.load(
                path,
                sr=TARGET_SR,
                mono=True,
            )
        )

        waveform = np.asarray(
            waveform,
            dtype=np.float32,
        )

        if waveform.size == 0:
            raise ValueError(
                "Audio file contains no samples."
            )

        waveform = waveform[
            :
            TARGET_SR *
            MAX_AUDIO_SECONDS
        ]

        duration = (
            librosa.get_duration(
                y=waveform,
                sr=sample_rate,
            )
        )

        rms = librosa.feature.rms(
            y=waveform
        )[0]

        zcr = (
            librosa.feature
            .zero_crossing_rate(
                waveform
            )[0]
        )

        mfcc = librosa.feature.mfcc(
            y=waveform,
            sr=sample_rate,
            n_mfcc=13,
        )

        centroid = (
            librosa.feature
            .spectral_centroid(
                y=waveform,
                sr=sample_rate,
            )[0]
        )

        bandwidth = (
            librosa.feature
            .spectral_bandwidth(
                y=waveform,
                sr=sample_rate,
            )[0]
        )

        rolloff = (
            librosa.feature
            .spectral_rolloff(
                y=waveform,
                sr=sample_rate,
            )[0]
        )

        pitches, magnitudes = (
            librosa.piptrack(
                y=waveform,
                sr=sample_rate,
            )
        )

        positive = magnitudes[
            magnitudes > 0
        ]

        threshold = (
            float(np.median(positive))
            if positive.size
            else 0.0
        )

        pitch_values = pitches[
            magnitudes > threshold
        ]

        pitch_values = pitch_values[
            pitch_values > 0
        ]

        features = {
            "audio_duration":
                clean_float(duration),

            "audio_rms_mean":
                clean_float(
                    np.mean(rms)
                ),

            "audio_rms_std":
                clean_float(
                    np.std(rms)
                ),

            "audio_zcr_mean":
                clean_float(
                    np.mean(zcr)
                ),

            "audio_zcr_std":
                clean_float(
                    np.std(zcr)
                ),

            "audio_spectral_centroid_mean":
                clean_float(
                    np.mean(centroid)
                ),

            "audio_spectral_centroid_std":
                clean_float(
                    np.std(centroid)
                ),

            "audio_spectral_bandwidth_mean":
                clean_float(
                    np.mean(bandwidth)
                ),

            "audio_spectral_bandwidth_std":
                clean_float(
                    np.std(bandwidth)
                ),

            "audio_spectral_rolloff_mean":
                clean_float(
                    np.mean(rolloff)
                ),

            "audio_spectral_rolloff_std":
                clean_float(
                    np.std(rolloff)
                ),

            "audio_pitch_mean":
                clean_float(
                    np.mean(pitch_values)
                )
                if pitch_values.size
                else 0.0,

            "audio_pitch_std":
                clean_float(
                    np.std(pitch_values)
                )
                if pitch_values.size
                else 0.0,

            "audio_pitch_min":
                clean_float(
                    np.min(pitch_values)
                )
                if pitch_values.size
                else 0.0,

            "audio_pitch_max":
                clean_float(
                    np.max(pitch_values)
                )
                if pitch_values.size
                else 0.0,
        }

        for index in range(13):
            features[
                f"audio_mfcc_{index}_mean"
            ] = clean_float(
                np.mean(mfcc[index])
            )

            features[
                f"audio_mfcc_{index}_std"
            ] = clean_float(
                np.std(mfcc[index])
            )

        extractor, model = (
            self._ensure_audio_model()
        )

        try:
            inputs = extractor(
                waveform,
                sampling_rate=TARGET_SR,
                return_tensors="pt",
                padding=True,
            )

            inputs = {
                key:
                    value.to(self.device)
                for key, value
                in inputs.items()
            }

            with torch.inference_mode():
                outputs = model(**inputs)

            embedding = (
                outputs.last_hidden_state
                .mean(dim=1)
                .squeeze(0)
            )

            embedding = F.normalize(
                embedding,
                p=2,
                dim=0,
            )

            embedding = (
                embedding
                .detach()
                .cpu()
                .numpy()
                .astype(np.float32)
                .reshape(-1)
            )

        finally:
            self._release_audio_model()

        if (
            self.expected_wavlm_embedding_dim
            is not None
            and embedding.size
            != self.expected_wavlm_embedding_dim
        ):
            raise ValueError(
                "WavLM embedding dimension mismatch. "
                f"Expected="
                f"{self.expected_wavlm_embedding_dim}, "
                f"observed={embedding.size}"
            )

        for index, value in (
            enumerate(embedding)
        ):
            features[
                f"audio_wavlm_emb_{index}"
            ] = clean_float(value)

        return features


    # ========================================================
    # IMAGE
    # ========================================================

    def extract_clip_embedding(
        self,
        image: Image.Image,
    ) -> np.ndarray:

        processor, model = (
            self._ensure_clip_model()
        )

        try:
            inputs = processor(
                images=image.convert("RGB"),
                return_tensors="pt",
            )

            pixel_values = (
                inputs["pixel_values"]
                .to(self.device)
            )

            with torch.inference_mode():
                output = (
                    model.get_image_features(
                        pixel_values=(
                            pixel_values
                        )
                    )
                )

            if isinstance(
                output,
                torch.Tensor,
            ):
                image_features = output

            elif hasattr(
                output,
                "image_embeds",
            ):
                image_features = (
                    output.image_embeds
                )

            else:
                raise TypeError(
                    "Unsupported CLIP output "
                    f"type: {type(output)}"
                )

            image_features = F.normalize(
                image_features,
                p=2,
                dim=-1,
            )

            embedding = (
                image_features
                .squeeze(0)
                .detach()
                .cpu()
                .numpy()
                .astype(np.float32)
                .reshape(-1)
            )

        finally:
            self._release_clip_model()

        if (
            self.expected_clip_embedding_dim
            is not None
            and embedding.size
            != self.expected_clip_embedding_dim
        ):
            raise ValueError(
                "CLIP embedding dimension mismatch. "
                f"Expected="
                f"{self.expected_clip_embedding_dim}, "
                f"observed={embedding.size}"
            )

        return embedding


    def extract_webcam_calibration_features(
        self,
        clip_features: dict[str, float],
    ) -> dict[str, float]:

        if self.webcam_image_model is None:
            return {}

        missing = [
            column
            for column
            in self.webcam_image_feature_columns
            if column not in clip_features
        ]

        if missing:
            raise ValueError(
                "Webcam calibration CLIP "
                f"features missing: {missing[:10]}"
            )

        X = pd.DataFrame(
            [
                {
                    column:
                        clean_float(
                            clip_features[column]
                        )
                    for column
                    in self.webcam_image_feature_columns
                }
            ],
            columns=(
                self.webcam_image_feature_columns
            ),
        )

        probabilities = (
            self._predict_probability_dict(
                self.webcam_image_model,
                X,
                model_name=(
                    "Webcam-calibrated "
                    "image classifier"
                ),
            )
        )

        summary = (
            summarise_probability_dict(
                probabilities,
                labels=LABELS,
            )
        )

        return {
            "image_webcam_focused_prob":
                probabilities["focused"],

            "image_webcam_distracted_prob":
                probabilities[
                    "distracted"
                ],

            "image_webcam_fatigued_prob":
                probabilities["fatigued"],

            "image_webcam_overloaded_prob":
                probabilities["overloaded"],

            "image_webcam_top_probability":
                float(
                    summary["confidence"]
                ),

            "image_webcam_confidence_gap":
                float(
                    summary["confidence_gap"]
                ),
        }


    def extract_image_features(
        self,
        image_path: Path,
    ) -> dict[str, float]:

        path = Path(image_path)

        if not path.exists():
            raise FileNotFoundError(
                f"Image file missing:\n{path}"
            )

        with Image.open(path) as handle:
            embedding = (
                self.extract_clip_embedding(
                    handle.convert("RGB")
                )
            )

        clip_features = {
            f"image_clip_emb_{index}":
                clean_float(value)
            for index, value
            in enumerate(embedding)
        }

        return {
            **clip_features,
            **self.extract_webcam_calibration_features(
                clip_features
            ),
        }


    # ========================================================
    # FUSION
    # ========================================================

    def build_fusion_dataframe(
        self,
        features: dict[str, Any],
    ) -> pd.DataFrame:

        missing = [
            column
            for column
            in self.feature_columns
            if column not in features
        ]

        if missing:
            raise ValueError(
                "Fusion feature mismatch. "
                f"Missing {len(missing)} "
                f"features; examples: "
                f"{missing[:30]}"
            )

        row = {
            column:
                clean_float(
                    features[column]
                )
            for column
            in self.feature_columns
        }

        X = pd.DataFrame(
            [row],
            columns=self.feature_columns,
        )

        if not np.all(
            np.isfinite(
                X.to_numpy(
                    dtype=np.float64
                )
            )
        ):
            raise ValueError(
                "Fusion input contains "
                "non-finite values."
            )

        return X


    def predict(
        self,
        keystroke_json: Path,
        text: str,
        audio_path: Path,
        image_path: Path,
    ) -> dict[str, Any]:

        features = {}

        # Sequential extraction is deliberate for
        # low-memory deployment.
        features.update(
            self.extract_keystroke_features(
                keystroke_json
            )
        )

        features.update(
            self.extract_text_features(
                text
            )
        )

        features.update(
            self.extract_audio_features(
                audio_path
            )
        )

        image_features = (
            self.extract_image_features(
                image_path
            )
        )

        features.update(
            image_features
        )

        X = self.build_fusion_dataframe(
            features
        )

        native_prediction = (
            normalise_label(
                self.fusion_model
                .predict(X)[0]
            )
        )

        probabilities = (
            self._predict_probability_dict(
                self.fusion_model,
                X,
                model_name=(
                    "Fusion classifier"
                ),
            )
        )

        summary = (
            summarise_probability_dict(
                probabilities,
                labels=LABELS,
            )
        )

        webcam_probabilities = None
        webcam_summary = None

        if (
            self.webcam_image_model
            is not None
        ):
            webcam_probabilities = (
                normalise_probability_dict(
                    {
                        label:
                            image_features.get(
                                (
                                    "image_webcam_"
                                    f"{label}_prob"
                                ),
                                0.0,
                            )
                        for label in LABELS
                    },
                    labels=LABELS,
                )
            )

            webcam_summary = (
                summarise_probability_dict(
                    webcam_probabilities,
                    labels=LABELS,
                )
            )

        return {
            "prediction":
                summary["current_state"],

            "current_state":
                summary["current_state"],

            "confidence":
                summary["confidence"],

            "confidence_percent":
                summary[
                    "confidence_percent"
                ],

            "probabilities":
                probabilities,

            "feature_dimension":
                int(X.shape[1]),

            "device":
                str(self.device),

            "used_modalities": {
                "keystroke": True,
                "text": True,
                "audio": True,
                "image": True,

                "webcam_image_calibration":
                    self.webcam_image_model
                    is not None,
            },

            "image_calibration": {
                "enabled":
                    webcam_summary
                    is not None,

                "current_state":
                    (
                        webcam_summary[
                            "current_state"
                        ]
                        if webcam_summary
                        else None
                    ),

                "probabilities":
                    webcam_probabilities,

                "top_probability":
                    (
                        webcam_summary[
                            "confidence"
                        ]
                        if webcam_summary
                        else None
                    ),

                "confidence_gap":
                    (
                        webcam_summary[
                            "confidence_gap"
                        ]
                        if webcam_summary
                        else None
                    ),

                "metadata":
                    self.webcam_image_metadata,
            },

            "technical_details": {
                "native_model_prediction":
                    native_prediction,

                "feature_dimension":
                    int(X.shape[1]),

                "device":
                    str(self.device),

                "low_memory_mode":
                    LOW_MEMORY_MODE,

                "temporal_fusion_applied":
                    False,
            },
        }


# ============================================================
# CLI
# ============================================================

def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--keystroke_json",
        required=True,
        type=Path,
    )

    parser.add_argument(
        "--text",
        required=True,
    )

    parser.add_argument(
        "--audio",
        required=True,
        type=Path,
    )

    parser.add_argument(
        "--image",
        required=True,
        type=Path,
    )

    args = parser.parse_args()

    inference = (
        FinalMultimodalInference()
    )

    result = inference.predict(
        keystroke_json=(
            args.keystroke_json
        ),
        text=args.text,
        audio_path=args.audio,
        image_path=args.image,
    )

    print(
        json.dumps(
            result,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
