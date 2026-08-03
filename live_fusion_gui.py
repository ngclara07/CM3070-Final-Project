# === live_fusion_gui.py ===
#
# SenseFuzeAI
# Live Session-Aligned Multimodal Behavioural Fusion GUI
#
# =============================================================================
# MODALITIES
# =============================================================================
#
# 1. Keystroke dynamics
# 2. MPNet text embeddings
# 3. Librosa + WavLM audio features
# 4. CLIP image embeddings
# 5. Webcam-calibrated image probabilities
#
# The final fusion classifier is loaded from:
#
#     models/fusion_demo/fusion_pipeline.joblib
#
# The webcam-calibrated image classifier is loaded separately from:
#
#     models/image_demo/image_pipeline_webcam_calibrated.joblib
#
# The original image artifact is NOT replaced.
#
# =============================================================================

from __future__ import annotations

import json
import statistics
import threading
import time
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox
from typing import Any

import cv2
import joblib
import librosa
import numpy as np
import pandas as pd
import sounddevice as sd
import torch
import torch.nn.functional as F

from PIL import Image, ImageTk
from sentence_transformers import SentenceTransformer
from transformers import (
    CLIPModel,
    CLIPProcessor,
    Wav2Vec2FeatureExtractor,
    WavLMModel,
)


# =============================================================================
# TORCH
# =============================================================================

torch.set_num_threads(2)


# =============================================================================
# PROJECT ROOT
# =============================================================================

ROOT_DIR = Path(__file__).resolve().parent


# =============================================================================
# FINAL FUSION MODEL
# =============================================================================

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


# =============================================================================
# PRETRAINED MODELS
# =============================================================================

TEXT_MODEL_PATH = (
    ROOT_DIR
    / "models"
    / "all-mpnet-base-v2"
)

WAVLM_MODEL_PATH = (
    ROOT_DIR
    / "models"
    / "wavlm-base-plus"
)

CLIP_MODEL_PATH = (
    ROOT_DIR
    / "models"
    / "clip-vit-large-patch14"
)


# =============================================================================
# WEBCAM-CALIBRATED IMAGE CLASSIFIER
# =============================================================================

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


# =============================================================================
# CONFIGURATION
# =============================================================================

TARGET_SR = 16000
MAX_AUDIO_SECONDS = 20
MIC_RECORD_SECONDS = 10

MIN_KEYDOWNS = 20
MIN_TEXT_CHARS = 20

WEBCAM_FEATURE_INTERVAL_SEC = 2.0

DISPLAY_SIZE = (
    320,
    220,
)

CLASSES = [
    "focused",
    "distracted",
    "fatigued",
    "overloaded",
]


# =============================================================================
# GENERAL UTILITIES
# =============================================================================

def get_device() -> torch.device:
    return torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )


def confidence_level(
    gap: float,
) -> str:
    if gap >= 0.35:
        return "High"

    if gap >= 0.15:
        return "Medium"

    return "Low"


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


def clean_float(
    value: Any,
) -> float:
    try:
        value = float(value)

        if np.isfinite(value):
            return value

    except Exception:
        pass

    return 0.0


def normalise_label(
    value: Any,
) -> str:
    return (
        str(value)
        .strip()
        .lower()
    )


def normalise_key(
    event,
) -> str:

    if event.keysym == "BackSpace":
        return "backspace"

    if event.keysym == "Delete":
        return "delete"

    if event.keysym == "space":
        return "space"

    if len(event.char) == 1:
        return event.char.lower()

    return event.keysym.lower()


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
            normalise_label(label)
            for label
            in classes
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
                    normalise_label(label)
                    for label
                    in classes
                ]

    return []


def softmax(
    values: np.ndarray,
) -> np.ndarray:

    values = np.asarray(
        values,
        dtype=float,
    )

    values -= np.max(
        values
    )

    exp_values = np.exp(
        values
    )

    total = np.sum(
        exp_values
    )

    if total <= 0:
        return (
            np.ones_like(exp_values)
            / len(exp_values)
        )

    return exp_values / total


# =============================================================================
# MAIN GUI CLASS
# =============================================================================

class FusionDemoApp:

    def __init__(
        self,
        root: tk.Tk,
    ) -> None:

        self.root = root

        self.root.title(
            "SenseFuzeAI Live Multimodal Fusion GUI"
        )

        self.root.geometry(
            "1180x860"
        )

        self.root.minsize(
            980,
            760,
        )

        self.root.configure(
            bg="#07111f"
        )

        # ---------------------------------------------------------------------
        # Validate required artifacts
        # ---------------------------------------------------------------------

        self.validate_paths()

        # ---------------------------------------------------------------------
        # Device
        # ---------------------------------------------------------------------

        self.device = get_device()

        # ---------------------------------------------------------------------
        # Fusion model
        # ---------------------------------------------------------------------

        self.fusion_model = joblib.load(
            FUSION_MODEL_PATH
        )

        with FUSION_FEATURE_COLUMNS_PATH.open(
            "r",
            encoding="utf-8",
        ) as f:
            self.fusion_feature_columns = json.load(f)

        # ---------------------------------------------------------------------
        # Text model
        # ---------------------------------------------------------------------

        self.text_model = SentenceTransformer(
            str(TEXT_MODEL_PATH)
        )

        # ---------------------------------------------------------------------
        # WavLM
        # ---------------------------------------------------------------------

        self.wavlm_extractor = (
            Wav2Vec2FeatureExtractor
            .from_pretrained(
                str(WAVLM_MODEL_PATH)
            )
        )

        self.wavlm_model = (
            WavLMModel
            .from_pretrained(
                str(WAVLM_MODEL_PATH)
            )
            .to(
                self.device
            )
        )

        self.wavlm_model.eval()

        # ---------------------------------------------------------------------
        # CLIP
        # ---------------------------------------------------------------------

        self.clip_processor = (
            CLIPProcessor
            .from_pretrained(
                str(CLIP_MODEL_PATH)
            )
        )

        self.clip_model = (
            CLIPModel
            .from_pretrained(
                str(CLIP_MODEL_PATH)
            )
            .to(
                self.device
            )
        )

        self.clip_model.eval()

        # ---------------------------------------------------------------------
        # Webcam calibrated classifier
        # ---------------------------------------------------------------------

        self.webcam_image_model = None
        self.webcam_image_feature_columns: list[str] = []

        self.load_webcam_calibration_model()

        # ---------------------------------------------------------------------
        # Keystrokes
        # ---------------------------------------------------------------------

        self.keystroke_events: list[
            dict[str, Any]
        ] = []

        self.active_keys: set[str] = set()

        # ---------------------------------------------------------------------
        # Audio state
        # ---------------------------------------------------------------------

        self.audio_features_cache: dict[
            str,
            float,
        ] | None = None

        self.audio_source_name: str | None = None

        # ---------------------------------------------------------------------
        # Image state
        # ---------------------------------------------------------------------

        self.image_features_cache: dict[
            str,
            float,
        ] | None = None

        self.image_source_name: str | None = None

        self.current_frame: Image.Image | None = None

        self.preview_image = None

        # ---------------------------------------------------------------------
        # Webcam
        # ---------------------------------------------------------------------

        self.capture = None

        self.running_webcam = False

        self.last_webcam_feature_time = 0.0

        self.image_processing = False

        # ---------------------------------------------------------------------
        # Fusion processing
        # ---------------------------------------------------------------------

        self.fusion_processing = False

        # ---------------------------------------------------------------------
        # UI
        # ---------------------------------------------------------------------

        self.build_ui()

        self.update_readiness()


    # =========================================================================
    # PATH VALIDATION
    # =========================================================================

    def validate_paths(
        self,
    ) -> None:

        required_paths = [
            FUSION_MODEL_PATH,
            FUSION_FEATURE_COLUMNS_PATH,
            TEXT_MODEL_PATH,
            WAVLM_MODEL_PATH,
            CLIP_MODEL_PATH,
        ]

        missing = [
            path
            for path
            in required_paths
            if not path.exists()
        ]

        if missing:
            raise FileNotFoundError(
                "Missing required model files:\n\n"
                + "\n".join(
                    str(path)
                    for path
                    in missing
                )
            )


    # =========================================================================
    # CALIBRATED IMAGE MODEL
    # =========================================================================

    def load_webcam_calibration_model(
        self,
    ) -> None:

        webcam_required = any(
            column.startswith(
                "image_webcam_"
            )
            for column
            in self.fusion_feature_columns
        )

        if not webcam_required:
            return

        if not WEBCAM_IMAGE_MODEL_PATH.exists():
            raise FileNotFoundError(
                "Fusion model expects calibrated webcam image features, "
                "but the calibrated model is missing:\n"
                f"{WEBCAM_IMAGE_MODEL_PATH}"
            )

        if not WEBCAM_IMAGE_FEATURE_COLUMNS_PATH.exists():
            raise FileNotFoundError(
                "Calibrated image feature schema missing:\n"
                f"{WEBCAM_IMAGE_FEATURE_COLUMNS_PATH}"
            )

        self.webcam_image_model = joblib.load(
            WEBCAM_IMAGE_MODEL_PATH
        )

        with WEBCAM_IMAGE_FEATURE_COLUMNS_PATH.open(
            "r",
            encoding="utf-8",
        ) as f:
            self.webcam_image_feature_columns = json.load(f)


    # =========================================================================
    # UI
    # =========================================================================

    def build_ui(
        self,
    ) -> None:

        # ---------------------------------------------------------------------
        # Header
        # ---------------------------------------------------------------------

        tk.Label(
            self.root,
            text="SenseFuzeAI Live Multimodal Fusion System",
            font=("Arial", 22, "bold"),
            fg="#74f7ff",
            bg="#07111f",
        ).pack(
            pady=(12, 3)
        )

        tk.Label(
            self.root,
            text=(
                "Keystroke · Text · Audio · CLIP Vision · "
                "Webcam-Calibrated Image Fusion"
            ),
            font=("Arial", 11),
            fg="white",
            bg="#07111f",
        ).pack(
            pady=(0, 8)
        )

        # ---------------------------------------------------------------------
        # Readiness panel
        # ---------------------------------------------------------------------

        readiness_frame = tk.Frame(
            self.root,
            bg="#10203a",
            padx=12,
            pady=10,
        )

        readiness_frame.pack(
            fill="x",
            padx=18,
            pady=6,
        )

        self.fusion_ready_label = tk.Label(
            readiness_frame,
            text="Fusion Model: Loaded",
            font=("Arial", 10, "bold"),
            fg="#66ffd6",
            bg="#10203a",
        )

        self.fusion_ready_label.grid(
            row=0,
            column=0,
            padx=10,
        )

        self.text_ready_label = tk.Label(
            readiness_frame,
            text="Text: Missing",
            font=("Arial", 10, "bold"),
            fg="#ffb3b3",
            bg="#10203a",
        )

        self.text_ready_label.grid(
            row=0,
            column=1,
            padx=10,
        )

        self.key_ready_label = tk.Label(
            readiness_frame,
            text="Keystroke: Missing",
            font=("Arial", 10, "bold"),
            fg="#ffb3b3",
            bg="#10203a",
        )

        self.key_ready_label.grid(
            row=0,
            column=2,
            padx=10,
        )

        self.audio_ready_label = tk.Label(
            readiness_frame,
            text="Audio: Missing",
            font=("Arial", 10, "bold"),
            fg="#ffb3b3",
            bg="#10203a",
        )

        self.audio_ready_label.grid(
            row=0,
            column=3,
            padx=10,
        )

        self.image_ready_label = tk.Label(
            readiness_frame,
            text="Image: Missing",
            font=("Arial", 10, "bold"),
            fg="#ffb3b3",
            bg="#10203a",
        )

        self.image_ready_label.grid(
            row=0,
            column=4,
            padx=10,
        )

        self.device_label = tk.Label(
            readiness_frame,
            text=f"Device: {self.device}",
            font=("Arial", 10, "bold"),
            fg="#74f7ff",
            bg="#10203a",
        )

        self.device_label.grid(
            row=0,
            column=5,
            padx=10,
        )

        # ---------------------------------------------------------------------
        # Main input area
        # ---------------------------------------------------------------------

        main_frame = tk.Frame(
            self.root,
            bg="#07111f",
        )

        main_frame.pack(
            fill="both",
            expand=True,
            padx=18,
            pady=5,
        )

        left_frame = tk.Frame(
            main_frame,
            bg="#07111f",
        )

        left_frame.pack(
            side="left",
            fill="both",
            expand=True,
            padx=(0, 8),
        )

        right_frame = tk.Frame(
            main_frame,
            bg="#07111f",
        )

        right_frame.pack(
            side="left",
            fill="both",
            expand=True,
            padx=(8, 0),
        )

        # ---------------------------------------------------------------------
        # Text + keystroke
        # ---------------------------------------------------------------------

        text_frame = tk.LabelFrame(
            left_frame,
            text="Text + Keystroke Input",
            font=("Arial", 11, "bold"),
            fg="#74f7ff",
            bg="#07111f",
            padx=8,
            pady=8,
        )

        text_frame.pack(
            fill="both",
            expand=True,
            pady=5,
        )

        self.text_box = tk.Text(
            text_frame,
            height=10,
            width=60,
            font=("Arial", 11),
            bg="#0b1220",
            fg="white",
            insertbackground="white",
        )

        self.text_box.pack(
            fill="both",
            expand=True,
        )

        self.text_box.bind(
            "<KeyPress>",
            self.on_key_press,
        )

        self.text_box.bind(
            "<KeyRelease>",
            self.on_key_release,
        )

        self.text_box.bind(
            "<<Modified>>",
            self.on_text_modified,
        )

        # ---------------------------------------------------------------------
        # Audio controls
        # ---------------------------------------------------------------------

        audio_frame = tk.LabelFrame(
            left_frame,
            text="Audio Input",
            font=("Arial", 11, "bold"),
            fg="#74f7ff",
            bg="#07111f",
            padx=8,
            pady=8,
        )

        audio_frame.pack(
            fill="x",
            pady=5,
        )

        tk.Button(
            audio_frame,
            text="Choose Audio File",
            command=self.choose_audio_file_threaded,
            width=20,
        ).grid(
            row=0,
            column=0,
            padx=5,
        )

        tk.Button(
            audio_frame,
            text="Record Microphone",
            command=self.record_microphone_threaded,
            width=20,
            bg="#00a884",
            fg="white",
        ).grid(
            row=0,
            column=1,
            padx=5,
        )

        self.audio_label = tk.Label(
            audio_frame,
            text="Audio not loaded.",
            fg="#ffb3b3",
            bg="#07111f",
        )

        self.audio_label.grid(
            row=1,
            column=0,
            columnspan=2,
            pady=6,
        )

        # ---------------------------------------------------------------------
        # Image controls
        # ---------------------------------------------------------------------

        image_frame = tk.LabelFrame(
            right_frame,
            text="Image / Webcam Input",
            font=("Arial", 11, "bold"),
            fg="#74f7ff",
            bg="#07111f",
            padx=8,
            pady=8,
        )

        image_frame.pack(
            fill="x",
            pady=5,
        )

        tk.Button(
            image_frame,
            text="Choose Image",
            command=self.choose_image_file_threaded,
            width=15,
        ).grid(
            row=0,
            column=0,
            padx=4,
        )

        tk.Button(
            image_frame,
            text="Start Webcam",
            command=self.start_webcam,
            width=15,
            bg="#00a884",
            fg="white",
        ).grid(
            row=0,
            column=1,
            padx=4,
        )

        tk.Button(
            image_frame,
            text="Stop Webcam",
            command=self.stop_webcam,
            width=15,
            bg="#c0392b",
            fg="white",
        ).grid(
            row=0,
            column=2,
            padx=4,
        )

        self.image_label = tk.Label(
            image_frame,
            text="Image not loaded.",
            fg="#ffb3b3",
            bg="#07111f",
        )

        self.image_label.grid(
            row=1,
            column=0,
            columnspan=3,
            pady=6,
        )

        self.preview_label = tk.Label(
            right_frame,
            bg="#07111f",
        )

        self.preview_label.pack(
            pady=8,
        )

        # ---------------------------------------------------------------------
        # Main result panel
        # ---------------------------------------------------------------------

        result_frame = tk.Frame(
            right_frame,
            bg="#10203a",
            padx=14,
            pady=12,
        )

        result_frame.pack(
            fill="x",
            pady=5,
        )

        tk.Label(
            result_frame,
            text="Current Behavioural State",
            font=("Arial", 12, "bold"),
            fg="#cbd6ff",
            bg="#10203a",
        ).pack()

        self.state_label = tk.Label(
            result_frame,
            text="—",
            font=("Arial", 34, "bold"),
            fg="#74f7ff",
            bg="#10203a",
        )

        self.state_label.pack(
            pady=3,
        )

        self.confidence_label = tk.Label(
            result_frame,
            text="Confidence: —",
            font=("Arial", 16, "bold"),
            fg="white",
            bg="#10203a",
        )

        self.confidence_label.pack()

        self.confidence_level_label = tk.Label(
            result_frame,
            text="Prediction Confidence: —",
            font=("Arial", 13, "bold"),
            fg="#cbd6ff",
            bg="#10203a",
        )

        self.confidence_level_label.pack(
            pady=2,
        )

        # ---------------------------------------------------------------------
        # Controls
        # ---------------------------------------------------------------------

        control_frame = tk.Frame(
            self.root,
            bg="#07111f",
        )

        control_frame.pack(
            pady=6,
        )

        tk.Button(
            control_frame,
            text="Run Fusion Prediction",
            command=self.predict_fusion_threaded,
            font=("Arial", 12, "bold"),
            bg="#2E86C1",
            fg="white",
            width=24,
        ).grid(
            row=0,
            column=0,
            padx=8,
        )

        tk.Button(
            control_frame,
            text="Reset Session",
            command=self.reset,
            font=("Arial", 11, "bold"),
            bg="#4a5568",
            fg="white",
            width=18,
        ).grid(
            row=0,
            column=1,
            padx=8,
        )

        self.status_label = tk.Label(
            self.root,
            text="System ready.",
            font=("Arial", 10),
            fg="#cbd6ff",
            bg="#07111f",
        )

        self.status_label.pack(
            pady=3,
        )

        # ---------------------------------------------------------------------
        # Technical information
        # ---------------------------------------------------------------------

        technical_frame = tk.LabelFrame(
            self.root,
            text="Technical Details",
            font=("Arial", 10, "bold"),
            fg="#74f7ff",
            bg="#07111f",
            padx=8,
            pady=6,
        )

        technical_frame.pack(
            fill="both",
            expand=False,
            padx=18,
            pady=(4, 10),
        )

        self.prob_text = tk.Text(
            technical_frame,
            height=5,
            font=("Consolas", 9),
            bg="#0b1220",
            fg="#dbeafe",
        )

        self.prob_text.pack(
            side="left",
            fill="both",
            expand=True,
            padx=(0, 4),
        )

        self.info_text = tk.Text(
            technical_frame,
            height=5,
            font=("Consolas", 9),
            bg="#0b1220",
            fg="#dbeafe",
        )

        self.info_text.pack(
            side="left",
            fill="both",
            expand=True,
            padx=(4, 0),
        )


    # =========================================================================
    # READINESS
    # =========================================================================

    def update_readiness(
        self,
    ) -> None:

        text = self.text_box.get(
            "1.0",
            tk.END,
        ).strip()

        keydowns = sum(
            1
            for event
            in self.keystroke_events
            if event.get("type") == "down"
        )

        text_ready = (
            len(text)
            >= MIN_TEXT_CHARS
        )

        key_ready = (
            keydowns
            >= MIN_KEYDOWNS
        )

        audio_ready = (
            self.audio_features_cache
            is not None
        )

        image_ready = (
            self.image_features_cache
            is not None
        )

        self.text_ready_label.config(
            text=(
                "Text: Ready"
                if text_ready
                else f"Text: {len(text)}/{MIN_TEXT_CHARS}"
            ),
            fg=(
                "#66ffd6"
                if text_ready
                else "#ffb3b3"
            ),
        )

        self.key_ready_label.config(
            text=(
                "Keystroke: Ready"
                if key_ready
                else f"Keystroke: {keydowns}/{MIN_KEYDOWNS}"
            ),
            fg=(
                "#66ffd6"
                if key_ready
                else "#ffb3b3"
            ),
        )

        self.audio_ready_label.config(
            text=(
                "Audio: Ready"
                if audio_ready
                else "Audio: Missing"
            ),
            fg=(
                "#66ffd6"
                if audio_ready
                else "#ffb3b3"
            ),
        )

        self.image_ready_label.config(
            text=(
                "Image: Ready"
                if image_ready
                else "Image: Missing"
            ),
            fg=(
                "#66ffd6"
                if image_ready
                else "#ffb3b3"
            ),
        )


    # =========================================================================
    # TEXT / KEYSTROKE
    # =========================================================================

    def on_text_modified(
        self,
        _event,
    ) -> None:

        if self.text_box.edit_modified():
            self.text_box.edit_modified(
                False
            )

            self.update_readiness()


    def on_key_press(
        self,
        event,
    ) -> None:

        key = normalise_key(
            event
        )

        if key in self.active_keys:
            return

        self.active_keys.add(
            key
        )

        self.keystroke_events.append(
            {
                "type": "down",
                "key": key,
                "timestamp_perf": time.perf_counter(),
                "timestamp_epoch": time.time(),
            }
        )

        self.update_readiness()


    def on_key_release(
        self,
        event,
    ) -> None:

        key = normalise_key(
            event
        )

        self.active_keys.discard(
            key
        )

        self.keystroke_events.append(
            {
                "type": "up",
                "key": key,
                "timestamp_perf": time.perf_counter(),
                "timestamp_epoch": time.time(),
            }
        )

        self.update_readiness()


    # =========================================================================
    # KEYSTROKE FEATURES
    # =========================================================================

    def extract_keystroke_features(
        self,
    ) -> dict[str, float]:

        typed_text = self.text_box.get(
            "1.0",
            tk.END,
        ).strip()

        downs = [
            event
            for event
            in self.keystroke_events
            if event.get("type") == "down"
        ]

        down_times = [
            float(
                event[
                    "timestamp_perf"
                ]
            )
            for event
            in downs
            if "timestamp_perf"
            in event
        ]

        if len(
            down_times
        ) < 2:
            raise ValueError(
                "Not enough keystroke events."
            )

        keydown_count = len(
            downs
        )

        if keydown_count < MIN_KEYDOWNS:
            raise ValueError(
                f"Need at least {MIN_KEYDOWNS} key presses."
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

        hold_times: list[
            float
        ] = []

        unmatched_downs: dict[
            str,
            list[float],
        ] = {}

        for event in self.keystroke_events:

            key = event.get(
                "key"
            )

            event_type = event.get(
                "type"
            )

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

                unmatched_downs.setdefault(
                    key,
                    [],
                ).append(
                    timestamp
                )

            elif event_type == "up":

                if (
                    key
                    in unmatched_downs
                    and unmatched_downs[
                        key
                    ]
                ):
                    down_time = (
                        unmatched_downs[
                            key
                        ]
                        .pop(0)
                    )

                    hold_times.append(
                        timestamp
                        - down_time
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
            for event
            in downs
            if event.get(
                "key"
            )
            in {
                "backspace",
                "delete",
            }
        )

        pauses_1000 = [
            value
            for value
            in delays
            if value >= 1.0
        ]

        pauses_2000 = [
            value
            for value
            in delays
            if value >= 2.0
        ]

        pauses_5000 = [
            value
            for value
            in delays
            if value >= 5.0
        ]

        delay_mean = safe_mean(
            delays
        )

        delay_std = safe_std(
            delays
        )

        rhythm_consistency = (
            1.0
            / (
                1.0
                + delay_std
            )
            if delay_std > 0
            else 1.0
        )

        raw_features = {
            "total_duration_sec":
                total_duration,

            "keydown_count":
                keydown_count,

            "word_count":
                word_count,

            "typing_speed_kps":
                (
                    keydown_count
                    / total_duration
                    if total_duration > 0
                    else 0.0
                ),

            "typing_speed_wpm":
                (
                    (
                        word_count
                        / total_duration
                    )
                    * 60
                    if total_duration > 0
                    else 0.0
                ),

            "delay_mean":
                delay_mean,

            "delay_std":
                delay_std,

            "delay_min":
                min(delays)
                if delays
                else 0.0,

            "delay_max":
                max(delays)
                if delays
                else 0.0,

            "hold_mean":
                safe_mean(
                    hold_times
                ),

            "hold_std":
                safe_std(
                    hold_times
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
                    len(
                        pauses_1000
                    )
                    / len(
                        delays
                    )
                    if delays
                    else 0.0
                ),

            "pause_ratio_2000":
                (
                    len(
                        pauses_2000
                    )
                    / len(
                        delays
                    )
                    if delays
                    else 0.0
                ),

            "mental_block_ratio_5000":
                (
                    len(
                        pauses_5000
                    )
                    / len(
                        delays
                    )
                    if delays
                    else 0.0
                ),

            "correction_count":
                correction_count,

            "correction_ratio":
                (
                    correction_count
                    / keydown_count
                    if keydown_count
                    else 0.0
                ),

            "rhythm_consistency":
                rhythm_consistency,

            "burstiness_proxy":
                (
                    delay_std
                    / delay_mean
                    if delay_mean > 0
                    else 0.0
                ),

            "fits_starts_index":
                (
                    len(
                        pauses_1000
                    )
                    / len(
                        delays
                    )
                    if delays
                    else 0.0
                ),
        }

        output: dict[
            str,
            float,
        ] = {}

        for key, value in raw_features.items():

            clean_value = clean_float(
                value
            )

            # Original key
            output[
                key
            ] = clean_value

            # Fusion-training schema key
            output[
                f"keystroke_{key}"
            ] = clean_value

        return output


    # =========================================================================
    # TEXT FEATURES
    # =========================================================================

    def extract_text_features(
        self,
    ) -> dict[str, float]:

        text = self.text_box.get(
            "1.0",
            tk.END,
        ).strip()

        if len(
            text
        ) < MIN_TEXT_CHARS:
            raise ValueError(
                f"Need at least {MIN_TEXT_CHARS} text characters."
            )

        embedding = self.text_model.encode(
            [text],
            normalize_embeddings=True,
            show_progress_bar=False,
            convert_to_numpy=True,
        )[0]

        return {
            f"text_mpnet_emb_{index}":
            clean_float(
                value
            )

            for index, value
            in enumerate(
                embedding
            )
        }


    # =========================================================================
    # AUDIO
    # =========================================================================

    def load_audio_waveform(
        self,
        audio_path: Path,
    ) -> tuple[
        np.ndarray,
        int,
    ]:

        waveform, sr = librosa.load(
            audio_path,
            sr=TARGET_SR,
            mono=True,
        )

        waveform = waveform.astype(
            np.float32
        )

        if len(
            waveform
        ) == 0:
            raise ValueError(
                "Audio waveform is empty."
            )

        waveform = waveform[
            : TARGET_SR
            * MAX_AUDIO_SECONDS
        ]

        return (
            waveform,
            sr,
        )


    def extract_audio_features_from_waveform(
        self,
        waveform: np.ndarray,
        sr: int,
    ) -> dict[str, float]:

        waveform = np.asarray(
            waveform,
            dtype=np.float32,
        )

        waveform = waveform[
            : TARGET_SR
            * MAX_AUDIO_SECONDS
        ]

        duration = librosa.get_duration(
            y=waveform,
            sr=sr,
        )

        rms = librosa.feature.rms(
            y=waveform
        )[0]

        zcr = librosa.feature.zero_crossing_rate(
            waveform
        )[0]

        mfcc = librosa.feature.mfcc(
            y=waveform,
            sr=sr,
            n_mfcc=13,
        )

        spectral_centroid = (
            librosa.feature
            .spectral_centroid(
                y=waveform,
                sr=sr,
            )[0]
        )

        spectral_bandwidth = (
            librosa.feature
            .spectral_bandwidth(
                y=waveform,
                sr=sr,
            )[0]
        )

        spectral_rolloff = (
            librosa.feature
            .spectral_rolloff(
                y=waveform,
                sr=sr,
            )[0]
        )

        pitches, magnitudes = (
            librosa.piptrack(
                y=waveform,
                sr=sr,
            )
        )

        if np.any(
            magnitudes > 0
        ):
            threshold = np.median(
                magnitudes[
                    magnitudes > 0
                ]
            )

        else:
            threshold = 0.0

        pitch_values = pitches[
            magnitudes > threshold
        ]

        pitch_values = pitch_values[
            pitch_values > 0
        ]

        features = {
            "audio_duration":
                duration,

            "audio_rms_mean":
                np.mean(rms),

            "audio_rms_std":
                np.std(rms),

            "audio_zcr_mean":
                np.mean(zcr),

            "audio_zcr_std":
                np.std(zcr),

            "audio_spectral_centroid_mean":
                np.mean(
                    spectral_centroid
                ),

            "audio_spectral_centroid_std":
                np.std(
                    spectral_centroid
                ),

            "audio_spectral_bandwidth_mean":
                np.mean(
                    spectral_bandwidth
                ),

            "audio_spectral_bandwidth_std":
                np.std(
                    spectral_bandwidth
                ),

            "audio_spectral_rolloff_mean":
                np.mean(
                    spectral_rolloff
                ),

            "audio_spectral_rolloff_std":
                np.std(
                    spectral_rolloff
                ),

            "audio_pitch_mean":
                (
                    np.mean(
                        pitch_values
                    )
                    if len(
                        pitch_values
                    )
                    else 0.0
                ),

            "audio_pitch_std":
                (
                    np.std(
                        pitch_values
                    )
                    if len(
                        pitch_values
                    )
                    else 0.0
                ),

            "audio_pitch_min":
                (
                    np.min(
                        pitch_values
                    )
                    if len(
                        pitch_values
                    )
                    else 0.0
                ),

            "audio_pitch_max":
                (
                    np.max(
                        pitch_values
                    )
                    if len(
                        pitch_values
                    )
                    else 0.0
                ),
        }

        for index in range(
            13
        ):

            features[
                f"audio_mfcc_{index}_mean"
            ] = np.mean(
                mfcc[
                    index
                ]
            )

            features[
                f"audio_mfcc_{index}_std"
            ] = np.std(
                mfcc[
                    index
                ]
            )

        inputs = self.wavlm_extractor(
            waveform,
            sampling_rate=TARGET_SR,
            return_tensors="pt",
            padding=True,
        )

        inputs = {
            key: value.to(
                self.device
            )
            for key, value
            in inputs.items()
        }

        with torch.inference_mode():

            outputs = self.wavlm_model(
                **inputs
            )

        embedding = (
            outputs
            .last_hidden_state
            .mean(
                dim=1
            )
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
        )

        for index, value in enumerate(
            embedding
        ):

            features[
                f"audio_wavlm_emb_{index}"
            ] = value

        return {
            key: clean_float(
                value
            )
            for key, value
            in features.items()
        }


    def choose_audio_file_threaded(
        self,
    ) -> None:

        file_path = filedialog.askopenfilename(
            title="Select audio file",
            filetypes=[
                (
                    "Audio files",
                    "*.wav *.mp3 *.m4a *.flac *.ogg *.aac",
                ),
                (
                    "All files",
                    "*.*",
                ),
            ],
        )

        if not file_path:
            return

        threading.Thread(
            target=self.load_audio_file_worker,
            args=(
                Path(
                    file_path
                ),
            ),
            daemon=True,
        ).start()


    def load_audio_file_worker(
        self,
        path: Path,
    ) -> None:

        try:

            self.set_status(
                "Extracting audio features..."
            )

            waveform, sr = (
                self.load_audio_waveform(
                    path
                )
            )

            features = (
                self.extract_audio_features_from_waveform(
                    waveform,
                    sr,
                )
            )

            self.audio_features_cache = (
                features
            )

            self.audio_source_name = (
                path.name
            )

            self.root.after(
                0,
                lambda: self.audio_label.config(
                    text=(
                        f"Audio loaded: "
                        f"{path.name}"
                    ),
                    fg="#66ffd6",
                ),
            )

            self.set_status(
                "Audio features ready."
            )

            self.root.after(
                0,
                self.update_readiness,
            )

        except Exception as exc:

            self.show_error(
                "Audio Error",
                exc,
            )


    def record_microphone_threaded(
        self,
    ) -> None:

        threading.Thread(
            target=self.record_microphone_worker,
            daemon=True,
        ).start()


    def record_microphone_worker(
        self,
    ) -> None:

        try:

            self.set_status(
                f"Recording microphone for "
                f"{MIC_RECORD_SECONDS} seconds..."
            )

            recording = sd.rec(
                int(
                    MIC_RECORD_SECONDS
                    * TARGET_SR
                ),
                samplerate=TARGET_SR,
                channels=1,
                dtype="float32",
            )

            sd.wait()

            waveform = (
                recording
                .flatten()
                .astype(
                    np.float32
                )
            )

            if len(
                waveform
            ) == 0:
                raise ValueError(
                    "No microphone audio captured."
                )

            self.set_status(
                "Extracting microphone audio features..."
            )

            features = (
                self.extract_audio_features_from_waveform(
                    waveform,
                    TARGET_SR,
                )
            )

            self.audio_features_cache = (
                features
            )

            self.audio_source_name = (
                "microphone"
            )

            self.root.after(
                0,
                lambda: self.audio_label.config(
                    text=(
                        "Audio loaded: "
                        "microphone recording"
                    ),
                    fg="#66ffd6",
                ),
            )

            self.set_status(
                "Microphone audio ready."
            )

            self.root.after(
                0,
                self.update_readiness,
            )

        except Exception as exc:

            self.show_error(
                "Microphone Error",
                exc,
            )


    # =========================================================================
    # IMAGE / CLIP
    # =========================================================================

    def extract_clip_embedding(
        self,
        image: Image.Image,
    ) -> np.ndarray:

        image = image.convert(
            "RGB"
        )

        inputs = self.clip_processor(
            images=image,
            return_tensors="pt",
        )

        pixel_values = (
            inputs[
                "pixel_values"
            ]
            .to(
                self.device
            )
        )

        with torch.inference_mode():

            try:

                output = (
                    self.clip_model
                    .get_image_features(
                        pixel_values=pixel_values
                    )
                )

                if isinstance(
                    output,
                    torch.Tensor,
                ):
                    image_features = (
                        output
                    )

                elif hasattr(
                    output,
                    "image_embeds",
                ):
                    image_features = (
                        output.image_embeds
                    )

                elif hasattr(
                    output,
                    "pooler_output",
                ):
                    image_features = (
                        output.pooler_output
                    )

                elif hasattr(
                    output,
                    "last_hidden_state",
                ):
                    image_features = (
                        output
                        .last_hidden_state
                        .mean(
                            dim=1
                        )
                    )

                else:
                    raise TypeError(
                        "Unsupported CLIP output type: "
                        f"{type(output)}"
                    )

            except Exception:

                output = (
                    self.clip_model
                    .vision_model(
                        pixel_values=pixel_values
                    )
                )

                if hasattr(
                    output,
                    "pooler_output",
                ):
                    image_features = (
                        output.pooler_output
                    )

                elif hasattr(
                    output,
                    "last_hidden_state",
                ):
                    image_features = (
                        output
                        .last_hidden_state
                        .mean(
                            dim=1
                        )
                    )

                else:
                    raise TypeError(
                        "Unsupported CLIP vision output."
                    )

        image_features = F.normalize(
            image_features,
            p=2,
            dim=-1,
        )

        return (
            image_features
            .squeeze(0)
            .detach()
            .cpu()
            .numpy()
        )


    # =========================================================================
    # CALIBRATED IMAGE PROBABILITIES
    # =========================================================================

    def extract_webcam_calibrated_features(
        self,
        clip_features: dict[
            str,
            float,
        ],
    ) -> dict[str, float]:

        if self.webcam_image_model is None:
            return {}

        missing = [
            column
            for column
            in self.webcam_image_feature_columns
            if column
            not in clip_features
        ]

        if missing:
            raise ValueError(
                "Calibrated image feature mismatch.\n"
                f"Missing columns: {missing[:20]}"
            )

        row = {
            column: clean_float(
                clip_features[
                    column
                ]
            )
            for column
            in self.webcam_image_feature_columns
        }

        x = pd.DataFrame(
            [row],
            columns=self.webcam_image_feature_columns,
        )

        classes = get_model_classes(
            self.webcam_image_model
        )

        probabilities_lookup = {
            label: 0.0
            for label
            in CLASSES
        }

        if hasattr(
            self.webcam_image_model,
            "predict_proba",
        ):

            probabilities = (
                self.webcam_image_model
                .predict_proba(
                    x
                )[0]
            )

            for label, probability in zip(
                classes,
                probabilities,
            ):

                if label in CLASSES:

                    probabilities_lookup[
                        label
                    ] = clean_float(
                        probability
                    )

        elif hasattr(
            self.webcam_image_model,
            "decision_function",
        ):

            scores = np.asarray(
                self.webcam_image_model
                .decision_function(
                    x
                )
            )

            if scores.ndim > 1:
                scores = scores[
                    0
                ]

            probabilities = softmax(
                scores
            )

            for label, probability in zip(
                classes,
                probabilities,
            ):

                if label in CLASSES:

                    probabilities_lookup[
                        label
                    ] = clean_float(
                        probability
                    )

        else:

            prediction = normalise_label(
                self.webcam_image_model
                .predict(
                    x
                )[0]
            )

            if prediction in CLASSES:

                probabilities_lookup[
                    prediction
                ] = 1.0

        # ---------------------------------------------------------------------
        # Defensive normalization
        # ---------------------------------------------------------------------

        values = np.asarray(
            [
                probabilities_lookup[
                    label
                ]
                for label
                in CLASSES
            ],
            dtype=float,
        )

        total = values.sum()

        if total > 0:
            values /= total

        else:
            values[:] = (
                1.0
                / len(
                    CLASSES
                )
            )

        probabilities_lookup = {
            label: float(
                probability
            )
            for label, probability
            in zip(
                CLASSES,
                values,
            )
        }

        ranked = sorted(
            probabilities_lookup.items(),
            key=lambda item: item[1],
            reverse=True,
        )

        top_probability = (
            ranked[0][1]
        )

        second_probability = (
            ranked[1][1]
        )

        return {
            "image_webcam_focused_prob":
                probabilities_lookup[
                    "focused"
                ],

            "image_webcam_distracted_prob":
                probabilities_lookup[
                    "distracted"
                ],

            "image_webcam_fatigued_prob":
                probabilities_lookup[
                    "fatigued"
                ],

            "image_webcam_overloaded_prob":
                probabilities_lookup[
                    "overloaded"
                ],

            "image_webcam_top_probability":
                top_probability,

            "image_webcam_confidence_gap":
                (
                    top_probability
                    - second_probability
                ),
        }


    def extract_image_features_from_pil(
        self,
        image: Image.Image,
    ) -> dict[str, float]:

        embedding = (
            self.extract_clip_embedding(
                image
            )
        )

        clip_features = {
            f"image_clip_emb_{index}":
            clean_float(
                value
            )

            for index, value
            in enumerate(
                embedding
            )
        }

        calibrated_features = (
            self.extract_webcam_calibrated_features(
                clip_features
            )
        )

        return {
            **clip_features,
            **calibrated_features,
        }


    # =========================================================================
    # IMAGE FILE
    # =========================================================================

    def choose_image_file_threaded(
        self,
    ) -> None:

        self.stop_webcam()

        file_path = filedialog.askopenfilename(
            title="Select image file",
            filetypes=[
                (
                    "Image files",
                    "*.jpg *.jpeg *.png *.webp",
                ),
                (
                    "All files",
                    "*.*",
                ),
            ],
        )

        if not file_path:
            return

        threading.Thread(
            target=self.load_image_worker,
            args=(
                Path(
                    file_path
                ),
            ),
            daemon=True,
        ).start()


    def load_image_worker(
        self,
        path: Path,
    ) -> None:

        try:

            self.set_status(
                "Extracting image features..."
            )

            image = Image.open(
                path
            ).convert(
                "RGB"
            )

            features = (
                self.extract_image_features_from_pil(
                    image
                )
            )

            self.current_frame = (
                image
            )

            self.image_features_cache = (
                features
            )

            self.image_source_name = (
                path.name
            )

            self.root.after(
                0,
                lambda: self.show_preview(
                    image
                ),
            )

            self.root.after(
                0,
                lambda: self.image_label.config(
                    text=(
                        f"Image loaded: "
                        f"{path.name}"
                    ),
                    fg="#66ffd6",
                ),
            )

            self.set_status(
                "Image features ready."
            )

            self.root.after(
                0,
                self.update_readiness,
            )

        except Exception as exc:

            self.show_error(
                "Image Error",
                exc,
            )


    # =========================================================================
    # WEBCAM
    # =========================================================================

    def start_webcam(
        self,
    ) -> None:

        self.stop_webcam()

        self.capture = cv2.VideoCapture(
            0
        )

        if not self.capture.isOpened():

            self.capture = None

            messagebox.showerror(
                "Webcam Error",
                "Could not access webcam.",
            )

            return

        self.running_webcam = True

        self.image_source_name = (
            "webcam"
        )

        self.last_webcam_feature_time = (
            0.0
        )

        self.image_label.config(
            text="Webcam active.",
            fg="#66ffd6",
        )

        self.status_label.config(
            text=(
                "Webcam running. "
                "Calibrated visual features updating."
            )
        )

        self.webcam_loop()


    def webcam_loop(
        self,
    ) -> None:

        if (
            not self.running_webcam
            or self.capture is None
        ):
            return

        ret, frame = (
            self.capture.read()
        )

        if not ret:

            self.stop_webcam()

            self.status_label.config(
                text="Webcam frame capture failed."
            )

            return

        frame_rgb = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB,
        )

        image = Image.fromarray(
            frame_rgb
        )

        self.current_frame = (
            image
        )

        self.show_preview(
            image
        )

        now = time.time()

        if (
            now
            - self.last_webcam_feature_time
            >= WEBCAM_FEATURE_INTERVAL_SEC
            and not self.image_processing
        ):

            self.last_webcam_feature_time = (
                now
            )

            frame_copy = image.copy()

            threading.Thread(
                target=self.update_webcam_features_worker,
                args=(
                    frame_copy,
                ),
                daemon=True,
            ).start()

        self.root.after(
            30,
            self.webcam_loop,
        )


    def update_webcam_features_worker(
        self,
        image: Image.Image,
    ) -> None:

        if self.image_processing:
            return

        try:

            self.image_processing = (
                True
            )

            features = (
                self.extract_image_features_from_pil(
                    image
                )
            )

            self.image_features_cache = (
                features
            )

            self.image_source_name = (
                "webcam"
            )

            self.root.after(
                0,
                lambda: self.image_label.config(
                    text=(
                        "Webcam calibrated "
                        "image features ready."
                    ),
                    fg="#66ffd6",
                ),
            )

            self.root.after(
                0,
                self.update_readiness,
            )

        except Exception as exc:

            self.set_status(
                f"Webcam feature extraction failed: "
                f"{exc}"
            )

        finally:

            self.image_processing = (
                False
            )


    def stop_webcam(
        self,
    ) -> None:

        self.running_webcam = (
            False
        )

        if self.capture is not None:

            self.capture.release()

            self.capture = (
                None
            )


    def show_preview(
        self,
        image: Image.Image,
    ) -> None:

        display = image.copy()

        display.thumbnail(
            DISPLAY_SIZE
        )

        self.preview_image = (
            ImageTk.PhotoImage(
                display
            )
        )

        self.preview_label.config(
            image=self.preview_image
        )


    # =========================================================================
    # BUILD FINAL FUSION VECTOR
    # =========================================================================

    def build_fusion_vector(
        self,
    ) -> tuple[
        pd.DataFrame,
        dict[str, Any],
    ]:

        features: dict[
            str,
            Any,
        ] = {}

        # ---------------------------------------------------------------------
        # Required live modalities
        # ---------------------------------------------------------------------

        keystroke_features = (
            self.extract_keystroke_features()
        )

        text_features = (
            self.extract_text_features()
        )

        if self.audio_features_cache is None:

            raise ValueError(
                "Audio features are missing. "
                "Choose an audio file or record microphone."
            )

        if self.image_features_cache is None:

            raise ValueError(
                "Image features are missing. "
                "Choose an image or start webcam."
            )

        features.update(
            keystroke_features
        )

        features.update(
            text_features
        )

        features.update(
            self.audio_features_cache
        )

        features.update(
            self.image_features_cache
        )

        # ---------------------------------------------------------------------
        # Validate exact trained schema
        # ---------------------------------------------------------------------

        missing = [
            column
            for column
            in self.fusion_feature_columns
            if column
            not in features
        ]

        if missing:

            raise ValueError(
                "Fusion feature mismatch.\n\n"
                f"Expected features: "
                f"{len(self.fusion_feature_columns)}\n"
                f"Available runtime features: "
                f"{len(features)}\n"
                f"Missing count: "
                f"{len(missing)}\n\n"
                f"Missing examples:\n"
                f"{missing[:30]}"
            )

        row = {
            column: clean_float(
                features[
                    column
                ]
            )
            for column
            in self.fusion_feature_columns
        }

        x = pd.DataFrame(
            [row],
            columns=self.fusion_feature_columns,
        )

        return (
            x,
            features,
        )


    # =========================================================================
    # FUSION PREDICTION
    # =========================================================================

    def predict_fusion_threaded(
        self,
    ) -> None:

        if self.fusion_processing:
            return

        threading.Thread(
            target=self.predict_fusion_worker,
            daemon=True,
        ).start()


    def predict_fusion_worker(
        self,
    ) -> None:

        try:

            self.fusion_processing = (
                True
            )

            self.set_status(
                "Building multimodal fusion vector..."
            )

            start_time = (
                time.perf_counter()
            )

            x, features = (
                self.build_fusion_vector()
            )

            prediction = (
                self.fusion_model
                .predict(
                    x
                )[0]
            )

            # -----------------------------------------------------------------
            # Final fusion probabilities
            # -----------------------------------------------------------------

            probabilities_lookup = {
                label: 0.0
                for label
                in CLASSES
            }

            classes = get_model_classes(
                self.fusion_model
            )

            if hasattr(
                self.fusion_model,
                "predict_proba",
            ):

                probabilities = (
                    self.fusion_model
                    .predict_proba(
                        x
                    )[0]
                )

                for label, probability in zip(
                    classes,
                    probabilities,
                ):

                    if label in CLASSES:

                        probabilities_lookup[
                            label
                        ] = clean_float(
                            probability
                        )

            elif hasattr(
                self.fusion_model,
                "decision_function",
            ):

                scores = np.asarray(
                    self.fusion_model
                    .decision_function(
                        x
                    )
                )

                if scores.ndim > 1:
                    scores = scores[
                        0
                    ]

                probabilities = softmax(
                    scores
                )

                for label, probability in zip(
                    classes,
                    probabilities,
                ):

                    if label in CLASSES:

                        probabilities_lookup[
                            label
                        ] = clean_float(
                            probability
                        )

            else:

                predicted_label = normalise_label(
                    prediction
                )

                if predicted_label in CLASSES:

                    probabilities_lookup[
                        predicted_label
                    ] = 1.0

            # -----------------------------------------------------------------
            # Normalize
            # -----------------------------------------------------------------

            values = np.asarray(
                [
                    probabilities_lookup[
                        label
                    ]
                    for label
                    in CLASSES
                ],
                dtype=float,
            )

            total = values.sum()

            if total > 0:
                values /= total

            else:
                values[:] = (
                    1.0
                    / len(
                        CLASSES
                    )
                )

            probabilities_lookup = {
                label: float(
                    probability
                )
                for label, probability
                in zip(
                    CLASSES,
                    values,
                )
            }

            ranked = sorted(
                probabilities_lookup.items(),
                key=lambda item: item[1],
                reverse=True,
            )

            current_state = (
                ranked[0][0]
            )

            confidence = (
                ranked[0][1]
            )

            second_class = (
                ranked[1][0]
            )

            second_probability = (
                ranked[1][1]
            )

            gap = (
                confidence
                - second_probability
            )

            runtime = (
                time.perf_counter()
                - start_time
            )

            # -----------------------------------------------------------------
            # Image calibration diagnostic
            # -----------------------------------------------------------------

            webcam_probs = {
                label:
                features.get(
                    f"image_webcam_{label}_prob",
                    None,
                )
                for label
                in CLASSES
            }

            webcam_ranked = [
                (
                    label,
                    value,
                )
                for label, value
                in webcam_probs.items()
                if value is not None
            ]

            webcam_ranked.sort(
                key=lambda item: item[1],
                reverse=True,
            )

            webcam_state = (
                webcam_ranked[0][0]
                if webcam_ranked
                else "N/A"
            )

            result = {
                "prediction":
                    str(
                        prediction
                    ),

                "current_state":
                    current_state,

                "confidence":
                    confidence,

                "confidence_percent":
                    confidence * 100,

                "confidence_level":
                    confidence_level(
                        gap
                    ),

                "second_class":
                    second_class,

                "second_probability":
                    second_probability,

                "confidence_gap":
                    gap,

                "probabilities":
                    probabilities_lookup,

                "feature_dimension":
                    int(
                        x.shape[1]
                    ),

                "runtime_seconds":
                    runtime,

                "device":
                    str(
                        self.device
                    ),

                "audio_source":
                    self.audio_source_name,

                "image_source":
                    self.image_source_name,

                "webcam_calibration_enabled":
                    (
                        self.webcam_image_model
                        is not None
                    ),

                "webcam_state":
                    webcam_state,

                "webcam_probabilities":
                    webcam_probs,

                "webcam_top_probability":
                    features.get(
                        "image_webcam_top_probability"
                    ),

                "webcam_confidence_gap":
                    features.get(
                        "image_webcam_confidence_gap"
                    ),
            }

            self.root.after(
                0,
                lambda:
                self.update_prediction_ui(
                    result
                ),
            )

            self.set_status(
                "Fusion prediction complete."
            )

        except Exception as exc:

            self.show_error(
                "Fusion Prediction Error",
                exc,
            )

        finally:

            self.fusion_processing = (
                False
            )


    # =========================================================================
    # RESULT UI
    # =========================================================================

    def update_prediction_ui(
        self,
        result: dict[
            str,
            Any,
        ],
    ) -> None:

        self.state_label.config(
            text=(
                result[
                    "current_state"
                ]
                .upper()
            )
        )

        self.confidence_label.config(
            text=(
                f"Confidence: "
                f"{result['confidence_percent']:.2f}%"
            )
        )

        level = result[
            "confidence_level"
        ]

        colour = {
            "High":
                "#66ffd6",

            "Medium":
                "#ffd166",

            "Low":
                "#ff6b8a",
        }.get(
            level,
            "#cbd6ff",
        )

        self.confidence_level_label.config(
            text=(
                f"Prediction Confidence: "
                f"{level}"
            ),
            fg=colour,
        )

        # ---------------------------------------------------------------------
        # Final fusion probabilities
        # ---------------------------------------------------------------------

        prob_lines = [
            "FINAL FUSION PROBABILITIES",
            "",
        ]

        for label, probability in sorted(
            result[
                "probabilities"
            ].items(),
            key=lambda item: item[1],
            reverse=True,
        ):

            bar = (
                "█"
                * int(
                    probability
                    * 28
                )
            )

            prob_lines.append(
                f"{label:12s}: "
                f"{probability * 100:6.2f}% "
                f"{bar}"
            )

        self.prob_text.delete(
            "1.0",
            tk.END,
        )

        self.prob_text.insert(
            tk.END,
            "\n".join(
                prob_lines
            ),
        )

        # ---------------------------------------------------------------------
        # Diagnostics
        # ---------------------------------------------------------------------

        info_lines = [
            "FUSION DIAGNOSTICS",
            "",
            (
                f"Current state          : "
                f"{result['current_state']}"
            ),
            (
                f"Confidence             : "
                f"{result['confidence_percent']:.2f}%"
            ),
            (
                f"Second class           : "
                f"{result['second_class']}"
            ),
            (
                f"Confidence gap         : "
                f"{result['confidence_gap']:.4f}"
            ),
            (
                f"Feature dimension      : "
                f"{result['feature_dimension']}"
            ),
            (
                f"Audio source           : "
                f"{result['audio_source']}"
            ),
            (
                f"Image source           : "
                f"{result['image_source']}"
            ),
            (
                f"Image calibration      : "
                f"{result['webcam_calibration_enabled']}"
            ),
            (
                f"Calibrated image state : "
                f"{result['webcam_state']}"
            ),
            (
                f"Image calibration gap  : "
                f"{result['webcam_confidence_gap']}"
            ),
            (
                f"Runtime                : "
                f"{result['runtime_seconds']:.4f} sec"
            ),
            (
                f"Device                 : "
                f"{result['device']}"
            ),
        ]

        self.info_text.delete(
            "1.0",
            tk.END,
        )

        self.info_text.insert(
            tk.END,
            "\n".join(
                info_lines
            ),
        )


    # =========================================================================
    # THREAD-SAFE UI HELPERS
    # =========================================================================

    def set_status(
        self,
        text: str,
    ) -> None:

        self.root.after(
            0,
            lambda:
            self.status_label.config(
                text=text
            ),
        )


    def show_error(
        self,
        title: str,
        error: Exception,
    ) -> None:

        self.root.after(
            0,
            lambda:
            messagebox.showerror(
                title,
                str(
                    error
                ),
            ),
        )

        self.set_status(
            f"{title}: {error}"
        )


    # =========================================================================
    # RESET
    # =========================================================================

    def reset(
        self,
    ) -> None:

        self.stop_webcam()

        sd.stop()

        self.keystroke_events = (
            []
        )

        self.active_keys = (
            set()
        )

        self.audio_features_cache = (
            None
        )

        self.audio_source_name = (
            None
        )

        self.image_features_cache = (
            None
        )

        self.image_source_name = (
            None
        )

        self.current_frame = (
            None
        )

        self.preview_image = (
            None
        )

        self.text_box.delete(
            "1.0",
            tk.END,
        )

        self.preview_label.config(
            image=""
        )

        self.audio_label.config(
            text="Audio not loaded.",
            fg="#ffb3b3",
        )

        self.image_label.config(
            text="Image not loaded.",
            fg="#ffb3b3",
        )

        self.state_label.config(
            text="—",
            fg="#74f7ff",
        )

        self.confidence_label.config(
            text="Confidence: —"
        )

        self.confidence_level_label.config(
            text="Prediction Confidence: —",
            fg="#cbd6ff",
        )

        self.prob_text.delete(
            "1.0",
            tk.END,
        )

        self.info_text.delete(
            "1.0",
            tk.END,
        )

        self.status_label.config(
            text="Session reset."
        )

        self.update_readiness()


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:

    root = tk.Tk()

    app = FusionDemoApp(
        root
    )

    def on_close() -> None:

        app.stop_webcam()

        sd.stop()

        root.destroy()

    root.protocol(
        "WM_DELETE_WINDOW",
        on_close,
    )

    root.mainloop()


if __name__ == "__main__":
    main()
