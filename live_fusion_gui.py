# live_fusion_gui.py

from __future__ import annotations

import json
import statistics
import threading
import time
from collections import deque
from pathlib import Path
from tkinter import filedialog, messagebox
import tkinter as tk

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


# ============================================================
# Torch configuration
# ============================================================

torch.set_num_threads(2)


# ============================================================
# Paths
# ============================================================

FUSION_MODEL_DIR = Path("models/fusion_demo")

FUSION_MODEL_PATH = (
    FUSION_MODEL_DIR / "fusion_pipeline.joblib"
)

FUSION_FEATURE_COLUMNS_PATH = (
    FUSION_MODEL_DIR / "feature_columns.json"
)

TEXT_MODEL_PATH = Path(
    "models/all-mpnet-base-v2"
)

WAVLM_MODEL_PATH = Path(
    "models/wavlm-base-plus"
)

CLIP_MODEL_PATH = Path(
    "models/clip-vit-large-patch14"
)


# Image-only artifacts are retained separately.
# They are NOT substituted for the fusion model.
IMAGE_MODEL_DIR = Path("models/image_demo")

ORIGINAL_IMAGE_MODEL_PATH = (
    IMAGE_MODEL_DIR / "image_pipeline.joblib"
)

WEBCAM_CALIBRATED_IMAGE_MODEL_PATH = (
    IMAGE_MODEL_DIR
    / "image_pipeline_webcam_calibrated.joblib"
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
# Audio configuration
# ============================================================

TARGET_SR = 16000
MAX_AUDIO_SECONDS = 20
MIC_RECORD_SECONDS = 10


# ============================================================
# Input requirements
# ============================================================

MIN_KEYDOWNS = 20
MIN_TEXT_CHARS = 20


# ============================================================
# Live prediction configuration
# ============================================================

LIVE_FUSION_ENABLED = True

# How often the system attempts a multimodal prediction.
LIVE_FUSION_INTERVAL_MS = 2500

# Rolling temporal fusion-output history.
TEMPORAL_PROBABILITY_WINDOW = 5

# Webcam / uploaded-video CLIP extraction interval.
VISUAL_FEATURE_INTERVAL_SEC = 1.5

DISPLAY_SIZE = (360, 240)


# ============================================================
# Utility functions
# ============================================================

def get_device() -> torch.device:
    return torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )


def normalise_key(event) -> str:
    if event.keysym == "BackSpace":
        return "backspace"

    if event.keysym == "Delete":
        return "delete"

    if event.keysym == "space":
        return "space"

    if len(event.char) == 1:
        return event.char.lower()

    return event.keysym.lower()


def safe_mean(values) -> float:
    return (
        statistics.mean(values)
        if values
        else 0.0
    )


def safe_std(values) -> float:
    return (
        statistics.stdev(values)
        if len(values) >= 2
        else 0.0
    )


def confidence_level(gap: float) -> str:
    if gap >= 0.35:
        return "High"

    if gap >= 0.15:
        return "Medium"

    return "Low"


def normalise_probability_dict(
    probabilities: dict[str, float],
) -> dict[str, float]:
    """
    Ensure the probability dictionary contains all four classes
    and sums approximately to one.
    """

    output = {
        label: float(
            probabilities.get(label, 0.0)
        )
        for label in LABELS
    }

    total = sum(output.values())

    if total <= 0:
        return {
            label: 1.0 / len(LABELS)
            for label in LABELS
        }

    return {
        label: value / total
        for label, value in output.items()
    }


# ============================================================
# CLIP helpers
# ============================================================

def extract_clip_embedding(
    image: Image.Image,
    model: CLIPModel,
    processor: CLIPProcessor,
    device: torch.device,
) -> np.ndarray:

    image = image.convert("RGB")

    inputs = processor(
        images=image,
        return_tensors="pt",
    )

    pixel_values = inputs[
        "pixel_values"
    ].to(device)

    with torch.inference_mode():

        try:
            output = model.get_image_features(
                pixel_values=pixel_values
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
                    .mean(dim=1)
                )

            else:
                raise TypeError(
                    "Unsupported CLIP output "
                    f"type: {type(output)}"
                )

        except Exception:

            output = model.vision_model(
                pixel_values=pixel_values
            )

            if (
                hasattr(
                    output,
                    "pooler_output",
                )
                and output.pooler_output
                is not None
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
                    .mean(dim=1)
                )

            else:
                raise TypeError(
                    "Unsupported CLIP vision "
                    f"output type: {type(output)}"
                )

    image_features = F.normalize(
        image_features,
        p=2,
        dim=-1,
    )

    return (
        image_features
        .squeeze(0)
        .cpu()
        .numpy()
    )


# ============================================================
# Main application
# ============================================================

class FusionDemoApp:

    def __init__(
        self,
        root: tk.Tk,
    ):
        self.root = root

        self.root.title(
            "SenseFuzeAI Live Multimodal Fusion GUI"
        )

        self.root.geometry(
            "1280x860"
        )

        self.root.minsize(
            1000,
            740,
        )

        self.root.configure(
            bg="#07111f"
        )

        # ----------------------------------------------------
        # Validate core artifacts
        # ----------------------------------------------------

        required_paths = [
            FUSION_MODEL_PATH,
            FUSION_FEATURE_COLUMNS_PATH,
            TEXT_MODEL_PATH,
            WAVLM_MODEL_PATH,
            CLIP_MODEL_PATH,
        ]

        missing = [
            str(path)
            for path in required_paths
            if not path.exists()
        ]

        if missing:
            raise FileNotFoundError(
                "Missing required artifacts:\n"
                + "\n".join(missing)
            )

        # ----------------------------------------------------
        # Device
        # ----------------------------------------------------

        self.device = get_device()

        # ----------------------------------------------------
        # Fusion model
        # ----------------------------------------------------

        self.pipeline = joblib.load(
            FUSION_MODEL_PATH
        )

        with FUSION_FEATURE_COLUMNS_PATH.open(
            "r",
            encoding="utf-8",
        ) as f:
            self.fusion_feature_columns = (
                json.load(f)
            )

        # ----------------------------------------------------
        # Text model
        # ----------------------------------------------------

        self.text_model = (
            SentenceTransformer(
                str(TEXT_MODEL_PATH)
            )
        )

        # ----------------------------------------------------
        # Audio model
        # ----------------------------------------------------

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
            .to(self.device)
        )

        self.wavlm_model.eval()

        # ----------------------------------------------------
        # CLIP
        # ----------------------------------------------------

        self.clip_processor = (
            CLIPProcessor.from_pretrained(
                str(CLIP_MODEL_PATH)
            )
        )

        self.clip_model = (
            CLIPModel.from_pretrained(
                str(CLIP_MODEL_PATH)
            )
            .to(self.device)
        )

        self.clip_model.eval()

        # ----------------------------------------------------
        # Optional image-only classifiers
        # ----------------------------------------------------

        self.original_image_pipeline = None
        self.webcam_image_pipeline = None

        if ORIGINAL_IMAGE_MODEL_PATH.exists():
            self.original_image_pipeline = (
                joblib.load(
                    ORIGINAL_IMAGE_MODEL_PATH
                )
            )

        if (
            WEBCAM_CALIBRATED_IMAGE_MODEL_PATH
            .exists()
        ):
            self.webcam_image_pipeline = (
                joblib.load(
                    WEBCAM_CALIBRATED_IMAGE_MODEL_PATH
                )
            )

        # ----------------------------------------------------
        # Keystroke state
        # ----------------------------------------------------

        self.keystroke_events = []
        self.active_keys = set()

        # ----------------------------------------------------
        # Audio state
        # ----------------------------------------------------

        self.audio_features_cache = None
        self.audio_source_name = None

        # ----------------------------------------------------
        # Visual state
        # ----------------------------------------------------

        self.image_features_cache = None

        self.image_source_name = None

        # none | image | video | webcam
        self.image_source_type = "none"

        self.current_frame = None
        self.preview_image = None

        self.capture = None
        self.running_visual_stream = False

        self.last_visual_feature_time = 0.0

        self.visual_processing_busy = False

        # ----------------------------------------------------
        # Prediction concurrency
        # ----------------------------------------------------

        self.fusion_prediction_busy = False

        # ----------------------------------------------------
        # Temporal fusion probability history
        # ----------------------------------------------------

        self.probability_history = deque(
            maxlen=TEMPORAL_PROBABILITY_WINDOW
        )

        # ----------------------------------------------------
        # Build GUI
        # ----------------------------------------------------

        self.build_ui()

        # ----------------------------------------------------
        # Start live fusion scheduler
        # ----------------------------------------------------

        if LIVE_FUSION_ENABLED:
            self.root.after(
                LIVE_FUSION_INTERVAL_MS,
                self.live_fusion_tick,
            )

    # ========================================================
    # UI
    # ========================================================

    def build_ui(self) -> None:

        # ----------------------------------------------------
        # Header
        # ----------------------------------------------------

        tk.Label(
            self.root,
            text=(
                "SenseFuzeAI Live "
                "Multimodal Fusion System"
            ),
            font=("Arial", 22, "bold"),
            fg="#74f7ff",
            bg="#07111f",
        ).pack(
            pady=(12, 4)
        )

        tk.Label(
            self.root,
            text=(
                "Keystroke · Text · Audio · "
                "CLIP Vision · Temporal Probability Fusion"
            ),
            font=("Arial", 11),
            fg="white",
            bg="#07111f",
        ).pack(
            pady=(0, 8)
        )

        # ----------------------------------------------------
        # Readiness bar
        # ----------------------------------------------------

        status_frame = tk.Frame(
            self.root,
            bg="#10203a",
            padx=14,
            pady=10,
        )

        status_frame.pack(
            fill="x",
            padx=18,
            pady=6,
        )

        self.fusion_status_label = tk.Label(
            status_frame,
            text="Fusion Model: Loaded",
            font=("Arial", 10, "bold"),
            fg="#66ffd6",
            bg="#10203a",
        )

        self.fusion_status_label.pack(
            side="left",
            padx=10,
        )

        self.text_ready_label = tk.Label(
            status_frame,
            text=f"Text: 0/{MIN_TEXT_CHARS}",
            font=("Arial", 10, "bold"),
            fg="#ffb3b3",
            bg="#10203a",
        )

        self.text_ready_label.pack(
            side="left",
            padx=10,
        )

        self.key_ready_label = tk.Label(
            status_frame,
            text=f"Keystroke: 0/{MIN_KEYDOWNS}",
            font=("Arial", 10, "bold"),
            fg="#ffb3b3",
            bg="#10203a",
        )

        self.key_ready_label.pack(
            side="left",
            padx=10,
        )

        self.audio_ready_label = tk.Label(
            status_frame,
            text="Audio: Missing",
            font=("Arial", 10, "bold"),
            fg="#ffb3b3",
            bg="#10203a",
        )

        self.audio_ready_label.pack(
            side="left",
            padx=10,
        )

        self.image_ready_label = tk.Label(
            status_frame,
            text="Image: Missing",
            font=("Arial", 10, "bold"),
            fg="#ffb3b3",
            bg="#10203a",
        )

        self.image_ready_label.pack(
            side="left",
            padx=10,
        )

        self.device_label = tk.Label(
            status_frame,
            text=f"Device: {self.device}",
            font=("Arial", 10, "bold"),
            fg="#74f7ff",
            bg="#10203a",
        )

        self.device_label.pack(
            side="left",
            padx=10,
        )

        # ----------------------------------------------------
        # Main two-column area
        # ----------------------------------------------------

        main_frame = tk.Frame(
            self.root,
            bg="#07111f",
        )

        main_frame.pack(
            fill="both",
            expand=True,
            padx=18,
            pady=8,
        )

        main_frame.grid_columnconfigure(
            0,
            weight=3,
        )

        main_frame.grid_columnconfigure(
            1,
            weight=2,
        )

        main_frame.grid_rowconfigure(
            0,
            weight=1,
        )

        left_frame = tk.Frame(
            main_frame,
            bg="#07111f",
        )

        left_frame.grid(
            row=0,
            column=0,
            sticky="nsew",
            padx=(0, 8),
        )

        right_frame = tk.Frame(
            main_frame,
            bg="#07111f",
        )

        right_frame.grid(
            row=0,
            column=1,
            sticky="nsew",
            padx=(8, 0),
        )

        # ----------------------------------------------------
        # Text / keystroke
        # ----------------------------------------------------

        text_frame = tk.LabelFrame(
            left_frame,
            text="Text + Keystroke Input",
            font=("Arial", 11, "bold"),
            fg="#74f7ff",
            bg="#07111f",
            padx=10,
            pady=8,
        )

        text_frame.pack(
            fill="both",
            expand=True,
            pady=(0, 8),
        )

        self.text_box = tk.Text(
            text_frame,
            height=11,
            font=("Arial", 12),
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

        # ----------------------------------------------------
        # Audio
        # ----------------------------------------------------

        audio_frame = tk.LabelFrame(
            left_frame,
            text="Audio Input",
            font=("Arial", 11, "bold"),
            fg="#74f7ff",
            bg="#07111f",
            padx=10,
            pady=8,
        )

        audio_frame.pack(
            fill="x",
            pady=(8, 0),
        )

        tk.Button(
            audio_frame,
            text="Choose Audio File",
            command=self.choose_audio_file,
            width=20,
            font=("Arial", 10, "bold"),
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
            font=("Arial", 10, "bold"),
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
            wraplength=550,
        )

        self.audio_label.grid(
            row=1,
            column=0,
            columnspan=2,
            pady=8,
        )

        # ----------------------------------------------------
        # Visual controls
        # ----------------------------------------------------

        visual_frame = tk.LabelFrame(
            right_frame,
            text="Image / Video / Webcam Input",
            font=("Arial", 11, "bold"),
            fg="#74f7ff",
            bg="#07111f",
            padx=10,
            pady=8,
        )

        visual_frame.pack(
            fill="x",
            pady=(0, 8),
        )

        tk.Button(
            visual_frame,
            text="Choose Image",
            command=self.choose_image_file,
            width=14,
            font=("Arial", 9, "bold"),
        ).grid(
            row=0,
            column=0,
            padx=4,
        )

        tk.Button(
            visual_frame,
            text="Choose Video",
            command=self.choose_video_file,
            width=14,
            bg="#6c5ce7",
            fg="white",
            font=("Arial", 9, "bold"),
        ).grid(
            row=0,
            column=1,
            padx=4,
        )

        tk.Button(
            visual_frame,
            text="Start Webcam",
            command=self.start_webcam,
            width=14,
            bg="#00a884",
            fg="white",
            font=("Arial", 9, "bold"),
        ).grid(
            row=0,
            column=2,
            padx=4,
        )

        tk.Button(
            visual_frame,
            text="Stop",
            command=self.stop_visual_stream,
            width=10,
            bg="#c0392b",
            fg="white",
            font=("Arial", 9, "bold"),
        ).grid(
            row=0,
            column=3,
            padx=4,
        )

        self.image_label = tk.Label(
            visual_frame,
            text="Visual input not loaded.",
            fg="#ffb3b3",
            bg="#07111f",
            wraplength=450,
        )

        self.image_label.grid(
            row=1,
            column=0,
            columnspan=4,
            pady=8,
        )

        # ----------------------------------------------------
        # Preview
        # ----------------------------------------------------

        self.preview_label = tk.Label(
            right_frame,
            bg="#07111f",
        )

        self.preview_label.pack(
            pady=6
        )

        # ----------------------------------------------------
        # Result
        # ----------------------------------------------------

        result_frame = tk.Frame(
            right_frame,
            bg="#10203a",
            padx=14,
            pady=14,
        )

        result_frame.pack(
            fill="x",
            pady=8,
        )

        tk.Label(
            result_frame,
            text="Current Behavioural State",
            font=("Arial", 12, "bold"),
            fg="#cbd6ff",
            bg="#10203a",
        ).pack()

        self.result_label = tk.Label(
            result_frame,
            text="—",
            font=("Arial", 34, "bold"),
            fg="#74f7ff",
            bg="#10203a",
        )

        self.result_label.pack(
            pady=(6, 2)
        )

        self.confidence_label = tk.Label(
            result_frame,
            text="Confidence: —",
            font=("Arial", 17, "bold"),
            fg="white",
            bg="#10203a",
        )

        self.confidence_label.pack(
            pady=2
        )

        self.confidence_level_label = tk.Label(
            result_frame,
            text="Prediction Confidence: —",
            font=("Arial", 13, "bold"),
            fg="#cbd6ff",
            bg="#10203a",
        )

        self.confidence_level_label.pack(
            pady=2
        )

        self.temporal_label = tk.Label(
            result_frame,
            text=(
                "Temporal window: "
                f"0/{TEMPORAL_PROBABILITY_WINDOW}"
            ),
            font=("Arial", 10),
            fg="#cbd6ff",
            bg="#10203a",
        )

        self.temporal_label.pack(
            pady=(4, 0)
        )

        # ----------------------------------------------------
        # Controls
        # ----------------------------------------------------

        control_frame = tk.Frame(
            self.root,
            bg="#07111f",
        )

        control_frame.pack(
            pady=5
        )

        tk.Button(
            control_frame,
            text="Run Fusion Prediction",
            command=self.predict_fusion_threaded,
            width=26,
            font=("Arial", 12, "bold"),
            bg="#2E86C1",
            fg="white",
        ).grid(
            row=0,
            column=0,
            padx=8,
        )

        tk.Button(
            control_frame,
            text="Reset Temporal Window",
            command=self.reset_temporal_history,
            width=22,
            font=("Arial", 10, "bold"),
            bg="#6c5ce7",
            fg="white",
        ).grid(
            row=0,
            column=1,
            padx=8,
        )

        tk.Button(
            control_frame,
            text="Reset Session",
            command=self.reset,
            width=18,
            font=("Arial", 10, "bold"),
            bg="#4a5568",
            fg="white",
        ).grid(
            row=0,
            column=2,
            padx=8,
        )

        # ----------------------------------------------------
        # Status
        # ----------------------------------------------------

        self.status_label = tk.Label(
            self.root,
            text="System ready.",
            fg="#cbd6ff",
            bg="#07111f",
            font=("Arial", 10),
        )

        self.status_label.pack(
            pady=4
        )

        # ----------------------------------------------------
        # Technical details
        # ----------------------------------------------------

        technical_frame = tk.LabelFrame(
            self.root,
            text="Technical Details",
            font=("Arial", 11, "bold"),
            fg="#74f7ff",
            bg="#07111f",
            padx=10,
            pady=8,
        )

        technical_frame.pack(
            fill="both",
            expand=True,
            padx=18,
            pady=(4, 10),
        )

        detail_inner = tk.Frame(
            technical_frame,
            bg="#07111f",
        )

        detail_inner.pack(
            fill="both",
            expand=True,
        )

        self.prob_text = tk.Text(
            detail_inner,
            height=7,
            width=72,
            font=("Consolas", 9),
            bg="#0b1220",
            fg="#dbeafe",
        )

        self.prob_text.pack(
            side="left",
            fill="both",
            expand=True,
            padx=(0, 5),
        )

        self.info_text = tk.Text(
            detail_inner,
            height=7,
            width=72,
            font=("Consolas", 9),
            bg="#0b1220",
            fg="#dbeafe",
        )

        self.info_text.pack(
            side="left",
            fill="both",
            expand=True,
            padx=(5, 0),
        )

    # ========================================================
    # Readiness
    # ========================================================

    def update_readiness(self) -> None:

        text = self.text_box.get(
            "1.0",
            tk.END,
        ).strip()

        text_count = len(text)

        keydown_count = self.count_keydowns()

        self.text_ready_label.config(
            text=(
                f"Text: {text_count}/"
                f"{MIN_TEXT_CHARS}"
            ),
            fg=(
                "#66ffd6"
                if text_count
                >= MIN_TEXT_CHARS
                else "#ffb3b3"
            ),
        )

        self.key_ready_label.config(
            text=(
                f"Keystroke: {keydown_count}/"
                f"{MIN_KEYDOWNS}"
            ),
            fg=(
                "#66ffd6"
                if keydown_count
                >= MIN_KEYDOWNS
                else "#ffb3b3"
            ),
        )

        self.audio_ready_label.config(
            text=(
                "Audio: Ready"
                if self.audio_features_cache
                is not None
                else "Audio: Missing"
            ),
            fg=(
                "#66ffd6"
                if self.audio_features_cache
                is not None
                else "#ffb3b3"
            ),
        )

        self.image_ready_label.config(
            text=(
                "Image: Ready"
                if self.image_features_cache
                is not None
                else "Image: Missing"
            ),
            fg=(
                "#66ffd6"
                if self.image_features_cache
                is not None
                else "#ffb3b3"
            ),
        )

    def fusion_inputs_ready(self) -> bool:

        text = self.text_box.get(
            "1.0",
            tk.END,
        ).strip()

        return (
            len(text) >= MIN_TEXT_CHARS
            and self.count_keydowns()
            >= MIN_KEYDOWNS
            and self.audio_features_cache
            is not None
            and self.image_features_cache
            is not None
        )

    # ========================================================
    # Text events
    # ========================================================

    def on_text_modified(
        self,
        _event,
    ) -> None:

        self.text_box.edit_modified(
            False
        )

        self.update_readiness()

    # ========================================================
    # Keystrokes
    # ========================================================

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
                "timestamp_perf": (
                    time.perf_counter()
                ),
                "timestamp_epoch": (
                    time.time()
                ),
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
                "timestamp_perf": (
                    time.perf_counter()
                ),
                "timestamp_epoch": (
                    time.time()
                ),
            }
        )

        self.update_readiness()

    def count_keydowns(self) -> int:
        return sum(
            1
            for event
            in self.keystroke_events
            if event.get("type")
            == "down"
        )

    # ========================================================
    # Keystroke feature extraction
    # ========================================================

    def extract_keystroke_features(
        self,
        typed_text: str,
        events: list[dict],
    ) -> dict[str, float]:

        downs = [
            event
            for event in events
            if event.get("type")
            == "down"
        ]

        down_times = [
            event["timestamp_perf"]
            for event in downs
            if "timestamp_perf"
            in event
        ]

        if len(down_times) < 2:
            raise ValueError(
                "Not enough keystroke events."
            )

        keydown_count = len(
            downs
        )

        if keydown_count < MIN_KEYDOWNS:
            raise ValueError(
                f"Need at least "
                f"{MIN_KEYDOWNS} key presses."
            )

        delays = [
            down_times[i]
            - down_times[i - 1]
            for i in range(
                1,
                len(down_times),
            )
        ]

        hold_times = []

        unmatched_downs = {}

        for event in events:

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

            if event_type == "down":

                unmatched_downs.setdefault(
                    key,
                    [],
                ).append(
                    timestamp
                )

            elif event_type == "up":

                if (
                    key in unmatched_downs
                    and unmatched_downs[key]
                ):
                    down_time = (
                        unmatched_downs[
                            key
                        ].pop(0)
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
            for event in downs
            if event.get("key")
            in {
                "backspace",
                "delete",
            }
        )

        pauses_1000 = [
            delay
            for delay in delays
            if delay >= 1.0
        ]

        pauses_2000 = [
            delay
            for delay in delays
            if delay >= 2.0
        ]

        pauses_5000 = [
            delay
            for delay in delays
            if delay >= 5.0
        ]

        delay_mean = safe_mean(
            delays
        )

        delay_std = safe_std(
            delays
        )

        rhythm_consistency = (
            1.0
            / (1.0 + delay_std)
            if delay_std > 0
            else 1.0
        )

        return {
            "total_duration_sec": round(
                total_duration,
                4,
            ),

            "keydown_count": (
                keydown_count
            ),

            "word_count": (
                word_count
            ),

            "typing_speed_kps": round(
                keydown_count
                / total_duration,
                4,
            )
            if total_duration > 0
            else 0.0,

            "typing_speed_wpm": round(
                (
                    word_count
                    / total_duration
                )
                * 60,
                4,
            )
            if total_duration > 0
            else 0.0,

            "delay_mean": round(
                delay_mean,
                4,
            ),

            "delay_std": round(
                delay_std,
                4,
            ),

            "delay_min": round(
                min(delays),
                4,
            )
            if delays
            else 0.0,

            "delay_max": round(
                max(delays),
                4,
            )
            if delays
            else 0.0,

            "hold_mean": round(
                safe_mean(
                    hold_times
                ),
                4,
            ),

            "hold_std": round(
                safe_std(
                    hold_times
                ),
                4,
            ),

            "pause_count_1000": (
                len(pauses_1000)
            ),

            "pause_count_2000": (
                len(pauses_2000)
            ),

            "pause_count_5000": (
                len(pauses_5000)
            ),

            "pause_ratio_1000": round(
                len(pauses_1000)
                / len(delays),
                4,
            )
            if delays
            else 0.0,

            "pause_ratio_2000": round(
                len(pauses_2000)
                / len(delays),
                4,
            )
            if delays
            else 0.0,

            "mental_block_ratio_5000": (
                round(
                    len(pauses_5000)
                    / len(delays),
                    4,
                )
                if delays
                else 0.0
            ),

            "correction_count": (
                correction_count
            ),

            "correction_ratio": round(
                correction_count
                / keydown_count,
                4,
            )
            if keydown_count
            else 0.0,

            "rhythm_consistency": round(
                rhythm_consistency,
                4,
            ),

            "burstiness_proxy": round(
                delay_std
                / delay_mean,
                4,
            )
            if delay_mean > 0
            else 0.0,

            "fits_starts_index": round(
                len(pauses_1000)
                / len(delays),
                4,
            )
            if delays
            else 0.0,
        }

    # ========================================================
    # Text features
    # ========================================================

    def extract_text_features(
        self,
        text: str,
    ) -> dict[str, float]:

        if (
            len(text)
            < MIN_TEXT_CHARS
        ):
            raise ValueError(
                f"Need at least "
                f"{MIN_TEXT_CHARS} text characters."
            )

        embedding = self.text_model.encode(
            [text],
            normalize_embeddings=True,
            show_progress_bar=False,
            convert_to_numpy=True,
        )[0]

        return {
            f"text_mpnet_emb_{i}": (
                float(value)
            )
            for i, value
            in enumerate(embedding)
        }

    # ========================================================
    # Audio
    # ========================================================

    def load_audio_waveform(
        self,
        audio_path: Path,
    ):

        waveform, sr = librosa.load(
            audio_path,
            sr=TARGET_SR,
            mono=True,
        )

        waveform = waveform.astype(
            np.float32
        )

        if len(waveform) == 0:
            raise ValueError(
                "Audio waveform is empty."
            )

        max_samples = (
            TARGET_SR
            * MAX_AUDIO_SECONDS
        )

        waveform = waveform[
            :max_samples
        ]

        return waveform, sr

    def extract_audio_features_from_waveform(
        self,
        waveform,
        sr,
    ) -> dict[str, float]:

        if len(waveform) == 0:
            raise ValueError(
                "Audio waveform is empty."
            )

        duration = (
            librosa.get_duration(
                y=waveform,
                sr=sr,
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

        pitch_values = (
            pitch_values[
                pitch_values > 0
            ]
        )

        features = {
            "audio_duration": float(
                duration
            ),

            "audio_rms_mean": float(
                np.mean(rms)
            ),

            "audio_rms_std": float(
                np.std(rms)
            ),

            "audio_zcr_mean": float(
                np.mean(zcr)
            ),

            "audio_zcr_std": float(
                np.std(zcr)
            ),

            "audio_spectral_centroid_mean":
                float(
                    np.mean(
                        spectral_centroid
                    )
                ),

            "audio_spectral_centroid_std":
                float(
                    np.std(
                        spectral_centroid
                    )
                ),

            "audio_spectral_bandwidth_mean":
                float(
                    np.mean(
                        spectral_bandwidth
                    )
                ),

            "audio_spectral_bandwidth_std":
                float(
                    np.std(
                        spectral_bandwidth
                    )
                ),

            "audio_spectral_rolloff_mean":
                float(
                    np.mean(
                        spectral_rolloff
                    )
                ),

            "audio_spectral_rolloff_std":
                float(
                    np.std(
                        spectral_rolloff
                    )
                ),

            "audio_pitch_mean": (
                float(
                    np.mean(
                        pitch_values
                    )
                )
                if len(pitch_values)
                else 0.0
            ),

            "audio_pitch_std": (
                float(
                    np.std(
                        pitch_values
                    )
                )
                if len(pitch_values)
                else 0.0
            ),

            "audio_pitch_min": (
                float(
                    np.min(
                        pitch_values
                    )
                )
                if len(pitch_values)
                else 0.0
            ),

            "audio_pitch_max": (
                float(
                    np.max(
                        pitch_values
                    )
                )
                if len(pitch_values)
                else 0.0
            ),
        }

        for i in range(13):

            features[
                f"audio_mfcc_{i}_mean"
            ] = float(
                np.mean(
                    mfcc[i]
                )
            )

            features[
                f"audio_mfcc_{i}_std"
            ] = float(
                np.std(
                    mfcc[i]
                )
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
            .cpu()
            .numpy()
        )

        for i, value in enumerate(
            embedding
        ):
            features[
                f"audio_wavlm_emb_{i}"
            ] = float(value)

        cleaned = {}

        for key, value in features.items():

            value = float(value)

            cleaned[key] = (
                value
                if np.isfinite(value)
                else 0.0
            )

        return cleaned

    def choose_audio_file(
        self,
    ) -> None:

        file_path = (
            filedialog.askopenfilename(
                title="Select audio file",
                filetypes=[
                    (
                        "Audio files",
                        "*.wav *.mp3 *.m4a "
                        "*.flac *.ogg *.aac",
                    ),
                    (
                        "All files",
                        "*.*",
                    ),
                ],
            )
        )

        if not file_path:
            return

        self.status_label.config(
            text=(
                "Extracting audio features..."
            )
        )

        self.root.update_idletasks()

        try:

            waveform, sr = (
                self.load_audio_waveform(
                    Path(file_path)
                )
            )

            self.audio_features_cache = (
                self.extract_audio_features_from_waveform(
                    waveform,
                    sr,
                )
            )

            self.audio_source_name = (
                Path(file_path).name
            )

            self.audio_label.config(
                text=(
                    "Audio loaded: "
                    f"{self.audio_source_name}"
                ),
                fg="#66ffd6",
            )

            self.status_label.config(
                text="Audio features ready."
            )

            self.reset_temporal_history(
                silent=True
            )

            self.update_readiness()

        except Exception as exc:

            messagebox.showerror(
                "Audio Error",
                str(exc),
            )

            self.status_label.config(
                text=(
                    "Audio feature extraction "
                    "failed."
                )
            )

    def record_microphone_threaded(
        self,
    ) -> None:

        threading.Thread(
            target=self.record_microphone,
            daemon=True,
        ).start()

    def record_microphone(
        self,
    ) -> None:

        try:

            self.root.after(
                0,
                lambda:
                self.status_label.config(
                    text=(
                        "Recording microphone for "
                        f"{MIC_RECORD_SECONDS} "
                        "seconds..."
                    )
                ),
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
                lambda:
                self.audio_label.config(
                    text=(
                        "Audio loaded: "
                        "microphone recording"
                    ),
                    fg="#66ffd6",
                ),
            )

            self.root.after(
                0,
                lambda:
                self.status_label.config(
                    text=(
                        "Microphone audio "
                        "features ready."
                    )
                ),
            )

            self.root.after(
                0,
                lambda:
                self.reset_temporal_history(
                    silent=True
                ),
            )

            self.root.after(
                0,
                self.update_readiness,
            )

        except Exception as exc:

            error_message = str(exc)

            self.root.after(
                0,
                lambda msg=error_message:
                messagebox.showerror(
                    "Microphone Error",
                    msg,
                ),
            )

    # ========================================================
    # Image features
    # ========================================================

    def extract_image_features_from_pil(
        self,
        image: Image.Image,
    ) -> dict[str, float]:

        embedding = extract_clip_embedding(
            image=image,
            model=self.clip_model,
            processor=self.clip_processor,
            device=self.device,
        )

        return {
            f"image_clip_emb_{i}": (
                float(value)
            )
            for i, value
            in enumerate(embedding)
        }

    def show_preview(
        self,
        image: Image.Image,
    ) -> None:

        display = image.copy()

        display.thumbnail(
            DISPLAY_SIZE,
            Image.Resampling.LANCZOS,
        )

        self.preview_image = (
            ImageTk.PhotoImage(
                display
            )
        )

        self.preview_label.config(
            image=self.preview_image
        )

    # ========================================================
    # Static image
    # ========================================================

    def choose_image_file(
        self,
    ) -> None:

        self.stop_visual_stream(
            update_status=False
        )

        file_path = (
            filedialog.askopenfilename(
                title="Select image file",
                filetypes=[
                    (
                        "Image files",
                        "*.jpg *.jpeg "
                        "*.png *.webp",
                    ),
                    (
                        "All files",
                        "*.*",
                    ),
                ],
            )
        )

        if not file_path:
            return

        try:

            image = (
                Image.open(
                    file_path
                )
                .convert("RGB")
            )

            self.current_frame = (
                image.copy()
            )

            self.image_features_cache = (
                self.extract_image_features_from_pil(
                    image
                )
            )

            self.image_source_type = (
                "image"
            )

            self.image_source_name = (
                Path(file_path).name
            )

            self.show_preview(
                image
            )

            self.image_label.config(
                text=(
                    "Image loaded: "
                    f"{self.image_source_name}"
                ),
                fg="#66ffd6",
            )

            self.status_label.config(
                text="Image features ready."
            )

            self.reset_temporal_history(
                silent=True
            )

            self.update_readiness()

        except Exception as exc:

            messagebox.showerror(
                "Image Error",
                str(exc),
            )

    # ========================================================
    # Uploaded video
    # ========================================================

    def choose_video_file(
        self,
    ) -> None:

        self.stop_visual_stream(
            update_status=False
        )

        file_path = (
            filedialog.askopenfilename(
                title="Select video file",
                filetypes=[
                    (
                        "Video files",
                        "*.mp4 *.avi *.mov "
                        "*.mkv *.webm",
                    ),
                    (
                        "All files",
                        "*.*",
                    ),
                ],
            )
        )

        if not file_path:
            return

        capture = cv2.VideoCapture(
            str(file_path)
        )

        if not capture.isOpened():

            capture.release()

            messagebox.showerror(
                "Video Error",
                "Could not open selected video.",
            )

            return

        self.capture = capture

        self.running_visual_stream = True

        self.image_source_type = (
            "video"
        )

        self.image_source_name = (
            Path(file_path).name
        )

        self.last_visual_feature_time = (
            0.0
        )

        self.image_label.config(
            text=(
                "Video running: "
                f"{self.image_source_name}"
            ),
            fg="#66ffd6",
        )

        self.status_label.config(
            text=(
                "Uploaded video running."
            )
        )

        self.reset_temporal_history(
            silent=True
        )

        self.visual_stream_loop()

    # ========================================================
    # Webcam
    # ========================================================

    def start_webcam(
        self,
    ) -> None:

        self.stop_visual_stream(
            update_status=False
        )

        capture = cv2.VideoCapture(
            0
        )

        if not capture.isOpened():

            capture.release()

            messagebox.showerror(
                "Webcam Error",
                "Could not access webcam.",
            )

            return

        capture.set(
            cv2.CAP_PROP_FRAME_WIDTH,
            640,
        )

        capture.set(
            cv2.CAP_PROP_FRAME_HEIGHT,
            480,
        )

        self.capture = capture

        self.running_visual_stream = True

        self.image_source_type = (
            "webcam"
        )

        self.image_source_name = (
            "webcam"
        )

        self.last_visual_feature_time = (
            0.0
        )

        self.image_label.config(
            text="Live webcam running.",
            fg="#66ffd6",
        )

        self.status_label.config(
            text="Webcam running."
        )

        self.reset_temporal_history(
            silent=True
        )

        self.visual_stream_loop()

    # ========================================================
    # Video / webcam loop
    # ========================================================

    def visual_stream_loop(
        self,
    ) -> None:

        if (
            not self.running_visual_stream
            or self.capture is None
        ):
            return

        ret, frame = (
            self.capture.read()
        )

        if not ret:

            if (
                self.image_source_type
                == "video"
            ):

                self.running_visual_stream = (
                    False
                )

                self.capture.release()
                self.capture = None

                self.status_label.config(
                    text=(
                        "Uploaded video finished."
                    )
                )

                return

            self.stop_visual_stream()

            return

        frame_rgb = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB,
        )

        image = Image.fromarray(
            frame_rgb
        )

        self.current_frame = (
            image.copy()
        )

        self.show_preview(
            image
        )

        now = time.time()

        if (
            now
            - self.last_visual_feature_time
            >= VISUAL_FEATURE_INTERVAL_SEC
        ):

            self.last_visual_feature_time = (
                now
            )

            frame_snapshot = (
                image.copy()
            )

            threading.Thread(
                target=self.update_visual_features,
                args=(frame_snapshot,),
                daemon=True,
            ).start()

        self.root.after(
            30,
            self.visual_stream_loop,
        )

    def update_visual_features(
        self,
        frame_snapshot: Image.Image,
    ) -> None:

        if self.visual_processing_busy:
            return

        try:

            self.visual_processing_busy = (
                True
            )

            features = (
                self.extract_image_features_from_pil(
                    frame_snapshot
                )
            )

            self.image_features_cache = (
                features
            )

            self.root.after(
                0,
                self.update_readiness,
            )

        except Exception:
            pass

        finally:
            self.visual_processing_busy = (
                False
            )

    def stop_visual_stream(
        self,
        update_status: bool = True,
    ) -> None:

        self.running_visual_stream = (
            False
        )

        if self.capture is not None:

            self.capture.release()
            self.capture = None

        if (
            update_status
            and self.image_source_type
            in {
                "video",
                "webcam",
            }
        ):
            self.status_label.config(
                text=(
                    "Video/webcam stopped."
                )
            )

    # ========================================================
    # Fusion vector
    # ========================================================

    def build_fusion_vector(
        self,
        typed_text: str,
        keystroke_events: list[dict],
        audio_features: dict,
        image_features: dict,
    ):

        features = {}

        features.update(
            self.extract_keystroke_features(
                typed_text,
                keystroke_events,
            )
        )

        features.update(
            self.extract_text_features(
                typed_text
            )
        )

        if audio_features is None:
            raise ValueError(
                "Audio features are missing."
            )

        if image_features is None:
            raise ValueError(
                "Image features are missing."
            )

        features.update(
            audio_features
        )

        features.update(
            image_features
        )

        missing = [
            column
            for column
            in self.fusion_feature_columns
            if column not in features
        ]

        if missing:
            raise ValueError(
                "Fusion feature mismatch.\n\n"
                f"Missing columns: "
                f"{missing[:30]}"
            )

        x = pd.DataFrame(
            [
                [
                    features[column]
                    for column
                    in self.fusion_feature_columns
                ]
            ],
            columns=(
                self.fusion_feature_columns
            ),
        )

        return x, features

    # ========================================================
    # Temporal probability aggregation
    # ========================================================

    def add_probability_observation(
        self,
        probabilities: dict[str, float],
    ) -> None:

        probabilities = (
            normalise_probability_dict(
                probabilities
            )
        )

        self.probability_history.append(
            probabilities
        )

    def aggregate_probability_history(
        self,
    ) -> dict[str, float]:

        if not self.probability_history:

            return {
                label: 1.0
                / len(LABELS)
                for label in LABELS
            }

        aggregate = {}

        for label in LABELS:

            values = [
                observation.get(
                    label,
                    0.0,
                )
                for observation
                in self.probability_history
            ]

            aggregate[label] = float(
                np.mean(values)
            )

        return normalise_probability_dict(
            aggregate
        )

    def reset_temporal_history(
        self,
        silent: bool = False,
    ) -> None:

        self.probability_history.clear()

        self.temporal_label.config(
            text=(
                "Temporal window: "
                f"0/{TEMPORAL_PROBABILITY_WINDOW}"
            )
        )

        if not silent:
            self.status_label.config(
                text=(
                    "Temporal probability "
                    "history cleared."
                )
            )

    # ========================================================
    # Fusion prediction scheduling
    # ========================================================

    def live_fusion_tick(
        self,
    ) -> None:

        try:

            self.update_readiness()

            if (
                LIVE_FUSION_ENABLED
                and self.fusion_inputs_ready()
                and not self.fusion_prediction_busy
            ):
                self.predict_fusion_threaded(
                    mode="Live"
                )

        finally:

            self.root.after(
                LIVE_FUSION_INTERVAL_MS,
                self.live_fusion_tick,
            )

    # ========================================================
    # Prediction
    # ========================================================

    def predict_fusion_threaded(
        self,
        mode: str = "Manual",
    ) -> None:

        if self.fusion_prediction_busy:
            return

        # ----------------------------------------------------
        # Snapshot all GUI/runtime state on main thread
        # ----------------------------------------------------

        typed_text = self.text_box.get(
            "1.0",
            tk.END,
        ).strip()

        keystroke_snapshot = [
            dict(event)
            for event
            in self.keystroke_events
        ]

        audio_snapshot = (
            dict(
                self.audio_features_cache
            )
            if self.audio_features_cache
            is not None
            else None
        )

        image_snapshot = (
            dict(
                self.image_features_cache
            )
            if self.image_features_cache
            is not None
            else None
        )

        audio_source = (
            self.audio_source_name
        )

        image_source = (
            self.image_source_name
        )

        image_source_type = (
            self.image_source_type
        )

        thread = threading.Thread(
            target=self.predict_fusion,
            args=(
                mode,
                typed_text,
                keystroke_snapshot,
                audio_snapshot,
                image_snapshot,
                audio_source,
                image_source,
                image_source_type,
            ),
            daemon=True,
        )

        thread.start()

    def predict_fusion(
        self,
        mode: str,
        typed_text: str,
        keystroke_events: list[dict],
        audio_features: dict | None,
        image_features: dict | None,
        audio_source: str | None,
        image_source: str | None,
        image_source_type: str,
    ) -> None:

        try:

            self.fusion_prediction_busy = (
                True
            )

            self.root.after(
                0,
                lambda:
                self.status_label.config(
                    text=(
                        "Building multimodal "
                        "fusion vector..."
                    )
                ),
            )

            start_time = (
                time.perf_counter()
            )

            x, features = (
                self.build_fusion_vector(
                    typed_text=typed_text,
                    keystroke_events=(
                        keystroke_events
                    ),
                    audio_features=(
                        audio_features
                    ),
                    image_features=(
                        image_features
                    ),
                )
            )

            # ------------------------------------------------
            # Raw fusion model prediction
            # ------------------------------------------------

            raw_prediction = (
                self.pipeline.predict(
                    x
                )[0]
            )

            if not hasattr(
                self.pipeline,
                "predict_proba",
            ):
                raise TypeError(
                    "Fusion model does not expose "
                    "predict_proba()."
                )

            raw_probabilities = (
                self.pipeline.predict_proba(
                    x
                )[0]
            )

            classes = [
                str(label)
                for label
                in self.pipeline.classes_
            ]

            raw_probability_dict = {
                label: float(probability)
                for label, probability
                in zip(
                    classes,
                    raw_probabilities,
                )
            }

            raw_probability_dict = (
                normalise_probability_dict(
                    raw_probability_dict
                )
            )

            raw_ranked = sorted(
                raw_probability_dict.items(),
                key=lambda item: item[1],
                reverse=True,
            )

            raw_top_class = (
                raw_ranked[0][0]
            )

            raw_top_probability = (
                raw_ranked[0][1]
            )

            # ------------------------------------------------
            # Temporal smoothing
            # ------------------------------------------------

            self.add_probability_observation(
                raw_probability_dict
            )

            aggregated_probabilities = (
                self.aggregate_probability_history()
            )

            aggregated_ranked = sorted(
                aggregated_probabilities.items(),
                key=lambda item: item[1],
                reverse=True,
            )

            final_class = (
                aggregated_ranked[0][0]
            )

            final_probability = float(
                aggregated_ranked[0][1]
            )

            second_class = (
                aggregated_ranked[1][0]
            )

            second_probability = float(
                aggregated_ranked[1][1]
            )

            confidence_gap = (
                final_probability
                - second_probability
            )

            level = confidence_level(
                confidence_gap
            )

            runtime = (
                time.perf_counter()
                - start_time
            )

            result = {
                "mode": mode,

                "raw_prediction": str(
                    raw_prediction
                ),

                "raw_top_class": (
                    raw_top_class
                ),

                "raw_top_probability": (
                    raw_top_probability
                ),

                "raw_probabilities": (
                    raw_probability_dict
                ),

                "current_state": (
                    final_class
                ),

                "confidence": (
                    final_probability
                ),

                "confidence_percent": (
                    final_probability
                    * 100.0
                ),

                "confidence_level": (
                    level
                ),

                "second_class": (
                    second_class
                ),

                "second_probability": (
                    second_probability
                ),

                "confidence_gap": (
                    confidence_gap
                ),

                "probabilities": (
                    aggregated_probabilities
                ),

                "temporal_samples": (
                    len(
                        self.probability_history
                    )
                ),

                "temporal_window": (
                    TEMPORAL_PROBABILITY_WINDOW
                ),

                "feature_dimension": int(
                    x.shape[1]
                ),

                "keydown_count": (
                    features.get(
                        "keydown_count"
                    )
                ),

                "word_count": (
                    features.get(
                        "word_count"
                    )
                ),

                "audio_source": (
                    audio_source
                ),

                "image_source": (
                    image_source
                ),

                "image_source_type": (
                    image_source_type
                ),

                "runtime_seconds": (
                    runtime
                ),

                "device": str(
                    self.device
                ),
            }

            self.root.after(
                0,
                lambda:
                self.update_prediction_ui(
                    result
                ),
            )

            self.root.after(
                0,
                lambda:
                self.status_label.config(
                    text=(
                        f"{mode} fusion "
                        "prediction complete."
                    )
                ),
            )

        except Exception as exc:

            error_message = str(exc)

            self.root.after(
                0,
                lambda msg=error_message:
                messagebox.showerror(
                    "Fusion Prediction Error",
                    msg,
                ),
            )

            self.root.after(
                0,
                lambda:
                self.status_label.config(
                    text=(
                        "Fusion prediction "
                        "failed."
                    )
                ),
            )

        finally:

            self.fusion_prediction_busy = (
                False
            )

    # ========================================================
    # Prediction display
    # ========================================================

    def update_prediction_ui(
        self,
        result: dict,
    ) -> None:

        # ----------------------------------------------------
        # Final temporally aggregated result
        # ----------------------------------------------------

        self.result_label.config(
            text=(
                result[
                    "current_state"
                ].upper()
            )
        )

        self.confidence_label.config(
            text=(
                "Confidence: "
                f"{result['confidence_percent']:.2f}%"
            )
        )

        level = result[
            "confidence_level"
        ]

        colour = {
            "High": "#66ffd6",
            "Medium": "#ffd166",
            "Low": "#ff6b8a",
        }.get(
            level,
            "#cbd6ff",
        )

        self.confidence_level_label.config(
            text=(
                "Prediction Confidence: "
                f"{level}"
            ),
            fg=colour,
        )

        self.temporal_label.config(
            text=(
                "Temporal window: "
                f"{result['temporal_samples']}/"
                f"{result['temporal_window']}"
            )
        )

        # ----------------------------------------------------
        # Aggregated probability distribution
        # ----------------------------------------------------

        prob_lines = [
            "TEMPORALLY AGGREGATED "
            "FUSION PROBABILITIES:",
            "",
        ]

        ranked = sorted(
            result[
                "probabilities"
            ].items(),
            key=lambda item: item[1],
            reverse=True,
        )

        for label, probability in ranked:

            bar_length = int(
                probability * 30
            )

            bar = (
                "█"
                * bar_length
            )

            prob_lines.append(
                f"{label:12s}: "
                f"{probability * 100:6.2f}%  "
                f"{bar}"
            )

        prob_lines.extend(
            [
                "",
                "CURRENT RAW FUSION OUTPUT:",
                "",
            ]
        )

        raw_ranked = sorted(
            result[
                "raw_probabilities"
            ].items(),
            key=lambda item: item[1],
            reverse=True,
        )

        for label, probability in raw_ranked:

            prob_lines.append(
                f"{label:12s}: "
                f"{probability * 100:6.2f}%"
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

        # ----------------------------------------------------
        # Diagnostics
        # ----------------------------------------------------

        info_lines = [
            "Multimodal fusion diagnostics:",
            "",

            (
                f"Mode                    : "
                f"{result['mode']}"
            ),

            (
                f"Final displayed state   : "
                f"{result['current_state']}"
            ),

            (
                f"Aggregated confidence   : "
                f"{result['confidence_percent']:.2f}%"
            ),

            (
                f"Confidence level        : "
                f"{result['confidence_level']}"
            ),

            (
                f"Second class            : "
                f"{result['second_class']}"
            ),

            (
                f"Confidence gap          : "
                f"{result['confidence_gap']:.4f}"
            ),

            "",
            (
                f"Raw current top class   : "
                f"{result['raw_top_class']}"
            ),

            (
                f"Raw current probability : "
                f"{result['raw_top_probability'] * 100:.2f}%"
            ),

            "",
            (
                f"Temporal samples        : "
                f"{result['temporal_samples']}"
            ),

            (
                f"Temporal max window     : "
                f"{result['temporal_window']}"
            ),

            "",
            (
                f"Keystroke keydowns      : "
                f"{result['keydown_count']}"
            ),

            (
                f"Text word count         : "
                f"{result['word_count']}"
            ),

            (
                f"Audio source            : "
                f"{result['audio_source']}"
            ),

            (
                f"Visual source type      : "
                f"{result['image_source_type']}"
            ),

            (
                f"Visual source           : "
                f"{result['image_source']}"
            ),

            (
                f"Fusion feature dimension: "
                f"{result['feature_dimension']}"
            ),

            (
                f"Runtime                 : "
                f"{result['runtime_seconds']:.4f} sec"
            ),

            (
                f"Device                  : "
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

    # ========================================================
    # Reset
    # ========================================================

    def reset(
        self,
    ) -> None:

        self.stop_visual_stream(
            update_status=False
        )

        # Keystroke
        self.keystroke_events = []
        self.active_keys = set()

        # Audio
        self.audio_features_cache = None
        self.audio_source_name = None

        # Visual
        self.image_features_cache = None
        self.image_source_name = None
        self.image_source_type = "none"

        self.current_frame = None
        self.preview_image = None

        # Temporal
        self.probability_history.clear()

        # Prediction
        self.fusion_prediction_busy = False

        # UI
        self.text_box.delete(
            "1.0",
            tk.END,
        )

        self.audio_label.config(
            text="Audio not loaded.",
            fg="#ffb3b3",
        )

        self.image_label.config(
            text="Visual input not loaded.",
            fg="#ffb3b3",
        )

        self.preview_label.config(
            image=""
        )

        self.result_label.config(
            text="—"
        )

        self.confidence_label.config(
            text="Confidence: —"
        )

        self.confidence_level_label.config(
            text="Prediction Confidence: —",
            fg="#cbd6ff",
        )

        self.temporal_label.config(
            text=(
                "Temporal window: "
                f"0/{TEMPORAL_PROBABILITY_WINDOW}"
            )
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
            text="System reset."
        )

        self.update_readiness()


# ============================================================
# Entry point
# ============================================================

def main() -> None:

    root = tk.Tk()

    app = FusionDemoApp(
        root
    )

    def on_close() -> None:

        app.stop_visual_stream(
            update_status=False
        )

        root.destroy()

    root.protocol(
        "WM_DELETE_WINDOW",
        on_close,
    )

    root.mainloop()


if __name__ == "__main__":
    main()
