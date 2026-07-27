# === image_live_gui.py ===
#
# SenseFuzeAI
# Image / Video / Webcam Behavioural-State GUI
#
# ============================================================================
# RUNTIME MODEL POLICY
# ============================================================================
#
# Uploaded still image:
#     models/image_demo/image_pipeline.joblib
#
# Uploaded video:
#     models/image_demo/image_pipeline.joblib
#
# Live webcam:
#     models/image_demo/image_pipeline_webcam_calibrated.joblib
#
# Both classifiers consume the same frozen CLIP ViT-L/14 embeddings.
#
# IMPORTANT:
# The displayed behavioural state is ALWAYS one of:
#
#     focused
#     distracted
#     fatigued
#     overloaded
#
# "Uncertain" is NOT treated as a fifth behavioural class.
# Low confidence is reported independently through:
#
#     - confidence percentage
#     - confidence level
#     - confidence gap
#
# Live webcam predictions are stabilized by averaging several consecutive
# probability distributions rather than voting on hard labels.
# ============================================================================

from __future__ import annotations

import csv
import json
import threading
import time
import tkinter as tk

from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from tkinter import filedialog, messagebox
from typing import Any

import cv2
import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from PIL import Image, ImageTk
from transformers import CLIPModel, CLIPProcessor


# ============================================================================
# PROJECT PATHS
# ============================================================================

ROOT_DIR = Path(__file__).resolve().parent

MODEL_DIR = ROOT_DIR / "models" / "image_demo"

ORIGINAL_MODEL_PATH = (
    MODEL_DIR
    / "image_pipeline.joblib"
)

WEBCAM_CALIBRATED_MODEL_PATH = (
    MODEL_DIR
    / "image_pipeline_webcam_calibrated.joblib"
)

FEATURE_COLUMNS_PATH = (
    MODEL_DIR
    / "feature_columns.json"
)

CLIP_MODEL_PATH = (
    ROOT_DIR
    / "models"
    / "clip-vit-large-patch14"
)

OUTPUT_DIR = (
    ROOT_DIR
    / "data"
    / "processed"
)

LOG_PATH = (
    OUTPUT_DIR
    / "image_live_gui_predictions.csv"
)


# ============================================================================
# CONFIGURATION
# ============================================================================

LABELS = [
    "focused",
    "distracted",
    "fatigued",
    "overloaded",
]

# One CLIP prediction approximately every second.
LIVE_PREDICTION_INTERVAL_SEC = 1.0

# Average the most recent N calibrated webcam probability distributions.
#
# 5 is a reasonable compromise:
# - much more stable than a single frame;
# - still changes within several seconds.
LIVE_PROBABILITY_WINDOW = 5

# Preview dimensions only.
# This does NOT resize the CLIP input itself; CLIPProcessor handles that.
DISPLAY_SIZE = (
    360,
    220,
)

WINDOW_SIZE = "1120x780"
WINDOW_MIN_WIDTH = 900
WINDOW_MIN_HEIGHT = 700


# ============================================================================
# GENERAL UTILITIES
# ============================================================================

def get_device() -> torch.device:
    return torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )


def normalise_label(value: Any) -> str:
    return (
        str(value)
        .strip()
        .lower()
    )


def clean_float(value: Any) -> float:
    try:
        number = float(value)

        if np.isfinite(number):
            return number

    except Exception:
        pass

    return 0.0


def confidence_level(
    top_probability: float,
    confidence_gap: float,
) -> str:
    """
    Human-readable confidence descriptor.

    This is a diagnostic category only. It does NOT replace the model class.

    A prediction therefore remains, for example:

        focused + Low confidence

    rather than becoming a synthetic "uncertain" class.
    """

    if (
        top_probability >= 0.70
        and confidence_gap >= 0.25
    ):
        return "High"

    if (
        top_probability >= 0.45
        and confidence_gap >= 0.10
    ):
        return "Medium"

    return "Low"


def get_model_classes(model: Any) -> list[str]:
    """
    Retrieve class ordering from a scikit-learn estimator or pipeline.
    """

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


def softmax(values: np.ndarray) -> np.ndarray:
    values = np.asarray(
        values,
        dtype=float,
    )

    values = (
        values
        - np.max(values)
    )

    exp_values = np.exp(values)

    total = float(
        np.sum(exp_values)
    )

    if total <= 0:
        return np.full(
            len(exp_values),
            1.0 / len(exp_values),
            dtype=float,
        )

    return exp_values / total


# ============================================================================
# CLIP MODEL
# ============================================================================

def build_clip_model(
    device: torch.device,
) -> tuple[CLIPModel, CLIPProcessor]:

    processor = (
        CLIPProcessor
        .from_pretrained(
            str(CLIP_MODEL_PATH)
        )
    )

    model = (
        CLIPModel
        .from_pretrained(
            str(CLIP_MODEL_PATH)
        )
        .to(device)
    )

    model.eval()

    return (
        model,
        processor,
    )


def extract_image_embedding_from_pil(
    image: Image.Image,
    model: CLIPModel,
    processor: CLIPProcessor,
    device: torch.device,
) -> np.ndarray:
    """
    Extract one normalized 768-dimensional CLIP image embedding.

    Includes compatibility handling for multiple transformers versions.
    """

    image = image.convert(
        "RGB"
    )

    inputs = processor(
        images=image,
        return_tensors="pt",
    )

    pixel_values = (
        inputs["pixel_values"]
        .to(device)
    )

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
                    "Unsupported CLIP get_image_features output: "
                    f"{type(output)}"
                )

        except Exception:
            # Compatibility fallback for transformers versions where
            # get_image_features returns a different output container.
            vision_output = model.vision_model(
                pixel_values=pixel_values
            )

            if hasattr(
                vision_output,
                "pooler_output",
            ):
                image_features = (
                    vision_output.pooler_output
                )

            elif hasattr(
                vision_output,
                "last_hidden_state",
            ):
                image_features = (
                    vision_output
                    .last_hidden_state
                    .mean(dim=1)
                )

            else:
                raise TypeError(
                    "Unsupported CLIP vision output: "
                    f"{type(vision_output)}"
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
    )

    return np.asarray(
        embedding,
        dtype=np.float32,
    )


# ============================================================================
# MAIN GUI
# ============================================================================

class ImageDemoApp:

    def __init__(
        self,
        root: tk.Tk,
    ) -> None:

        self.root = root

        self.root.title(
            "SenseFuzeAI Image Live GUI"
        )

        self.root.geometry(
            WINDOW_SIZE
        )

        self.root.minsize(
            WINDOW_MIN_WIDTH,
            WINDOW_MIN_HEIGHT,
        )

        self.root.configure(
            bg="#07111f"
        )

        # --------------------------------------------------------------------
        # Required artifacts
        # --------------------------------------------------------------------

        self.validate_artifacts()

        OUTPUT_DIR.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.initialise_log_file()

        # --------------------------------------------------------------------
        # Classifiers
        # --------------------------------------------------------------------

        self.original_pipeline = joblib.load(
            ORIGINAL_MODEL_PATH
        )

        self.webcam_pipeline = joblib.load(
            WEBCAM_CALIBRATED_MODEL_PATH
        )

        # --------------------------------------------------------------------
        # Feature schema
        # --------------------------------------------------------------------

        with FEATURE_COLUMNS_PATH.open(
            "r",
            encoding="utf-8",
        ) as f:
            self.feature_columns = json.load(f)

        if not isinstance(
            self.feature_columns,
            list,
        ):
            raise ValueError(
                "feature_columns.json must contain a JSON list."
            )

        # --------------------------------------------------------------------
        # CLIP
        # --------------------------------------------------------------------

        self.device = get_device()

        (
            self.clip_model,
            self.clip_processor,
        ) = build_clip_model(
            self.device
        )

        # --------------------------------------------------------------------
        # Runtime state
        # --------------------------------------------------------------------

        self.capture: cv2.VideoCapture | None = None

        self.running_video = False

        self.current_frame: Image.Image | None = None

        self.preview_image = None

        self.current_source_type = "none"

        self.current_source_name = "none"

        self.last_prediction_time = 0.0

        self.prediction_busy = False

        # Rolling webcam probabilities.
        #
        # Each item is:
        #
        # {
        #   "focused": ...,
        #   "distracted": ...,
        #   "fatigued": ...,
        #   "overloaded": ...
        # }
        #
        self.webcam_probability_history: deque[
            dict[str, float]
        ] = deque(
            maxlen=LIVE_PROBABILITY_WINDOW
        )

        # --------------------------------------------------------------------
        # UI
        # --------------------------------------------------------------------

        self.build_ui()


    # ========================================================================
    # ARTIFACT VALIDATION
    # ========================================================================

    def validate_artifacts(
        self,
    ) -> None:

        required = [
            ORIGINAL_MODEL_PATH,
            WEBCAM_CALIBRATED_MODEL_PATH,
            FEATURE_COLUMNS_PATH,
            CLIP_MODEL_PATH,
        ]

        missing = [
            path
            for path
            in required
            if not path.exists()
        ]

        if missing:
            raise FileNotFoundError(
                "Required image inference artifacts are missing:\n\n"
                + "\n".join(
                    str(path)
                    for path
                    in missing
                )
            )


    # ========================================================================
    # LOGGING
    # ========================================================================

    def initialise_log_file(
        self,
    ) -> None:

        if LOG_PATH.exists():
            return

        with LOG_PATH.open(
            "w",
            newline="",
            encoding="utf-8",
        ) as f:

            writer = csv.writer(f)

            writer.writerow(
                [
                    "timestamp",
                    "mode",
                    "source_type",
                    "source",
                    "classifier",
                    "current_state",
                    "raw_top_class",
                    "raw_top_probability",
                    "confidence",
                    "confidence_level",
                    "second_class",
                    "second_probability",
                    "confidence_gap",
                    "feature_dimension",
                    "probability_window_size",
                    "runtime_seconds",
                    "device",
                    "probabilities_json",
                ]
            )


    # ========================================================================
    # UI
    # ========================================================================

    def build_ui(
        self,
    ) -> None:

        # --------------------------------------------------------------------
        # Heading
        # --------------------------------------------------------------------

        tk.Label(
            self.root,
            text=(
                "SenseFuzeAI Image / Video "
                "Behavioural State Classifier"
            ),
            font=("Arial", 20, "bold"),
            fg="#74f7ff",
            bg="#07111f",
        ).pack(
            pady=(10, 3)
        )

        tk.Label(
            self.root,
            text=(
                "CLIP visual embeddings with a separately "
                "webcam-calibrated live classifier"
            ),
            font=("Arial", 11),
            fg="white",
            bg="#07111f",
        ).pack(
            pady=(0, 5)
        )

        # --------------------------------------------------------------------
        # Model status
        # --------------------------------------------------------------------

        status_frame = tk.Frame(
            self.root,
            bg="#10203a",
            padx=14,
            pady=9,
        )

        status_frame.pack(
            fill="x",
            padx=18,
            pady=5,
        )

        self.model_status_label = tk.Label(
            status_frame,
            text="Models: Loaded",
            font=("Arial", 10, "bold"),
            fg="#66ffd6",
            bg="#10203a",
        )

        self.model_status_label.grid(
            row=0,
            column=0,
            padx=8,
            sticky="w",
        )

        self.image_ready_label = tk.Label(
            status_frame,
            text="Visual Input: Missing",
            font=("Arial", 10, "bold"),
            fg="#ffb3b3",
            bg="#10203a",
        )

        self.image_ready_label.grid(
            row=0,
            column=1,
            padx=20,
            sticky="w",
        )

        self.device_label = tk.Label(
            status_frame,
            text=f"Device: {self.device}",
            font=("Arial", 10, "bold"),
            fg="#74f7ff",
            bg="#10203a",
        )

        self.device_label.grid(
            row=0,
            column=2,
            padx=8,
            sticky="w",
        )

        # --------------------------------------------------------------------
        # Controls
        # --------------------------------------------------------------------

        button_frame = tk.Frame(
            self.root,
            bg="#07111f",
        )

        button_frame.pack(
            pady=5
        )

        tk.Button(
            button_frame,
            text="Choose Image",
            command=self.choose_image,
            width=15,
            font=("Arial", 10, "bold"),
        ).grid(
            row=0,
            column=0,
            padx=4,
        )

        tk.Button(
            button_frame,
            text="Choose Video",
            command=self.choose_video,
            width=15,
            font=("Arial", 10, "bold"),
        ).grid(
            row=0,
            column=1,
            padx=4,
        )

        tk.Button(
            button_frame,
            text="Start Webcam",
            command=self.start_webcam,
            width=15,
            bg="#00a884",
            fg="white",
            font=("Arial", 10, "bold"),
        ).grid(
            row=0,
            column=2,
            padx=4,
        )

        tk.Button(
            button_frame,
            text="Stop",
            command=self.stop_video,
            width=10,
            bg="#c0392b",
            fg="white",
            font=("Arial", 10, "bold"),
        ).grid(
            row=0,
            column=3,
            padx=4,
        )

        tk.Button(
            button_frame,
            text="Manual Prediction",
            command=self.predict_current_frame_threaded,
            width=20,
            bg="#2E86C1",
            fg="white",
            font=("Arial", 10, "bold"),
        ).grid(
            row=0,
            column=4,
            padx=4,
        )

        tk.Button(
            button_frame,
            text="Reset",
            command=self.reset,
            width=12,
            bg="#4a5568",
            fg="white",
            font=("Arial", 10, "bold"),
        ).grid(
            row=0,
            column=5,
            padx=4,
        )

        # --------------------------------------------------------------------
        # Status messages
        # --------------------------------------------------------------------

        self.status_label = tk.Label(
            self.root,
            text="System ready.",
            fg="#cbd6ff",
            bg="#07111f",
            font=("Arial", 10),
        )

        self.status_label.pack(
            pady=(3, 1)
        )

        self.classifier_label = tk.Label(
            self.root,
            text="Active classifier: —",
            fg="#ffd166",
            bg="#07111f",
            font=("Arial", 11, "bold"),
        )

        self.classifier_label.pack(
            pady=(1, 4)
        )

        # --------------------------------------------------------------------
        # Primary result
        # --------------------------------------------------------------------

        result_frame = tk.Frame(
            self.root,
            bg="#10203a",
            padx=18,
            pady=11,
        )

        result_frame.pack(
            fill="x",
            padx=18,
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
            pady=(3, 1)
        )

        self.confidence_label = tk.Label(
            result_frame,
            text="Confidence: —",
            font=("Arial", 18, "bold"),
            fg="white",
            bg="#10203a",
        )

        self.confidence_label.pack(
            pady=1
        )

        self.confidence_level_label = tk.Label(
            result_frame,
            text="Prediction Confidence: —",
            font=("Arial", 13, "bold"),
            fg="#cbd6ff",
            bg="#10203a",
        )

        self.confidence_level_label.pack(
            pady=1
        )

        # --------------------------------------------------------------------
        # Lower area
        # --------------------------------------------------------------------

        lower_frame = tk.Frame(
            self.root,
            bg="#07111f",
        )

        lower_frame.pack(
            fill="both",
            expand=True,
            padx=18,
            pady=5,
        )

        # Preview -------------------------------------------------------------

        preview_frame = tk.LabelFrame(
            lower_frame,
            text="Visual Input",
            font=("Arial", 10, "bold"),
            fg="#74f7ff",
            bg="#07111f",
            padx=8,
            pady=8,
        )

        preview_frame.pack(
            side="left",
            fill="both",
            expand=False,
            padx=(0, 8),
        )

        self.preview_label = tk.Label(
            preview_frame,
            bg="#07111f",
        )

        self.preview_label.pack(
            pady=4
        )

        # Technical -----------------------------------------------------------

        technical_frame = tk.LabelFrame(
            lower_frame,
            text="Technical Details",
            font=("Arial", 11, "bold"),
            fg="#74f7ff",
            bg="#07111f",
            padx=8,
            pady=7,
        )

        technical_frame.pack(
            side="left",
            fill="both",
            expand=True,
            padx=(8, 0),
        )

        self.prob_text = tk.Text(
            technical_frame,
            height=7,
            width=75,
            font=("Consolas", 9),
            bg="#0b1220",
            fg="#dbeafe",
        )

        self.prob_text.pack(
            fill="both",
            expand=True,
            pady=(2, 4),
        )

        self.info_text = tk.Text(
            technical_frame,
            height=7,
            width=75,
            font=("Consolas", 9),
            bg="#0b1220",
            fg="#dbeafe",
        )

        self.info_text.pack(
            fill="both",
            expand=True,
            pady=(4, 2),
        )


    # ========================================================================
    # SOURCE / CLASSIFIER POLICY
    # ========================================================================

    def get_active_pipeline(
        self,
        source_type: str,
    ) -> tuple[Any, str]:

        if source_type == "webcam":
            return (
                self.webcam_pipeline,
                "Webcam-Calibrated Model",
            )

        return (
            self.original_pipeline,
            "Original Image Model",
        )


    # ========================================================================
    # IMAGE FEATURES
    # ========================================================================

    def frame_to_features(
        self,
        image: Image.Image,
    ) -> pd.DataFrame:

        embedding = extract_image_embedding_from_pil(
            image=image,
            model=self.clip_model,
            processor=self.clip_processor,
            device=self.device,
        )

        features = {
            f"image_clip_emb_{index}":
            clean_float(value)

            for index, value
            in enumerate(
                embedding
            )
        }

        missing = [
            column
            for column
            in self.feature_columns
            if column not in features
        ]

        if missing:
            raise ValueError(
                "Image feature schema mismatch.\n\n"
                f"Expected columns: {len(self.feature_columns)}\n"
                f"Extracted embedding dimensions: {len(embedding)}\n"
                f"Missing examples: {missing[:20]}"
            )

        return pd.DataFrame(
            [
                [
                    features[column]
                    for column
                    in self.feature_columns
                ]
            ],
            columns=self.feature_columns,
        )


    # ========================================================================
    # MODEL PROBABILITY EXTRACTION
    # ========================================================================

    def predict_probabilities(
        self,
        pipeline: Any,
        x: pd.DataFrame,
    ) -> dict[str, float]:

        classes = get_model_classes(
            pipeline
        )

        probability_lookup = {
            label: 0.0
            for label
            in LABELS
        }

        # Preferred path ------------------------------------------------------

        if hasattr(
            pipeline,
            "predict_proba",
        ):

            probabilities = (
                pipeline
                .predict_proba(x)[0]
            )

            for label, probability in zip(
                classes,
                probabilities,
            ):

                if label in LABELS:
                    probability_lookup[
                        label
                    ] = clean_float(
                        probability
                    )

        # Decision-function fallback -----------------------------------------

        elif hasattr(
            pipeline,
            "decision_function",
        ):

            scores = np.asarray(
                pipeline.decision_function(x)
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
                    probability_lookup[
                        label
                    ] = clean_float(
                        probability
                    )

        # Hard-prediction fallback -------------------------------------------

        else:

            prediction = normalise_label(
                pipeline.predict(x)[0]
            )

            if prediction not in LABELS:
                raise ValueError(
                    "Classifier returned unsupported label: "
                    f"{prediction}"
                )

            probability_lookup[
                prediction
            ] = 1.0

        # Defensive normalization --------------------------------------------

        values = np.asarray(
            [
                probability_lookup[label]
                for label
                in LABELS
            ],
            dtype=float,
        )

        total = float(
            values.sum()
        )

        if total > 0:
            values = values / total

        else:
            values = np.full(
                len(LABELS),
                1.0 / len(LABELS),
                dtype=float,
            )

        return {
            label: float(probability)
            for label, probability
            in zip(
                LABELS,
                values,
            )
        }


    # ========================================================================
    # WEBCAM TEMPORAL STABILIZATION
    # ========================================================================

    def stabilize_webcam_probabilities(
        self,
        raw_probabilities: dict[str, float],
    ) -> dict[str, float]:

        self.webcam_probability_history.append(
            raw_probabilities.copy()
        )

        stabilized: dict[
            str,
            float,
        ] = {}

        for label in LABELS:

            values = [
                frame_probs[label]
                for frame_probs
                in self.webcam_probability_history
            ]

            stabilized[label] = float(
                np.mean(values)
            )

        # Defensive normalization.
        total = sum(
            stabilized.values()
        )

        if total > 0:
            stabilized = {
                label: probability / total
                for label, probability
                in stabilized.items()
            }

        return stabilized


    # ========================================================================
    # CHOOSE IMAGE
    # ========================================================================

    def choose_image(
        self,
    ) -> None:

        self.stop_video(
            update_status=False
        )

        self.webcam_probability_history.clear()

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

        try:
            image = Image.open(
                file_path
            ).convert(
                "RGB"
            )

        except Exception as exc:
            messagebox.showerror(
                "Image Error",
                str(exc),
            )

            return

        self.current_frame = image

        self.current_source_type = (
            "image"
        )

        self.current_source_name = (
            Path(file_path).name
        )

        self.show_pil_image(
            image
        )

        self.image_ready_label.config(
            text="Visual Input: Ready",
            fg="#66ffd6",
        )

        self.status_label.config(
            text=(
                f"Loaded image: "
                f"{Path(file_path).name}"
            )
        )

        self.classifier_label.config(
            text=(
                "Active classifier: "
                "Original Image Model"
            )
        )

        self.clear_prediction()


    # ========================================================================
    # CHOOSE VIDEO
    # ========================================================================

    def choose_video(
        self,
    ) -> None:

        self.stop_video(
            update_status=False
        )

        self.webcam_probability_history.clear()

        file_path = filedialog.askopenfilename(
            title="Select video file",
            filetypes=[
                (
                    "Video files",
                    "*.mp4 *.avi *.mov *.mkv *.webm",
                ),
                (
                    "All files",
                    "*.*",
                ),
            ],
        )

        if not file_path:
            return

        capture = cv2.VideoCapture(
            file_path
        )

        if not capture.isOpened():
            capture.release()

            messagebox.showerror(
                "Video Error",
                "Could not open selected video.",
            )

            return

        self.capture = capture

        self.running_video = True

        self.current_source_type = (
            "video"
        )

        self.current_source_name = (
            Path(file_path).name
        )

        self.image_ready_label.config(
            text="Visual Input: Ready",
            fg="#66ffd6",
        )

        self.status_label.config(
            text=(
                f"Playing video: "
                f"{self.current_source_name}"
            )
        )

        self.classifier_label.config(
            text=(
                "Active classifier: "
                "Original Image Model"
            )
        )

        self.video_loop()


    # ========================================================================
    # START WEBCAM
    # ========================================================================

    def start_webcam(
        self,
    ) -> None:

        self.stop_video(
            update_status=False
        )

        self.webcam_probability_history.clear()

        # CAP_DSHOW often reduces Windows webcam startup latency.
        if hasattr(
            cv2,
            "CAP_DSHOW",
        ):
            capture = cv2.VideoCapture(
                0,
                cv2.CAP_DSHOW,
            )

        else:
            capture = cv2.VideoCapture(
                0
            )

        if not capture.isOpened():

            capture.release()

            # Generic fallback.
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

        self.capture = capture

        self.running_video = True

        self.current_source_type = (
            "webcam"
        )

        self.current_source_name = (
            "webcam"
        )

        self.last_prediction_time = (
            0.0
        )

        self.image_ready_label.config(
            text="Visual Input: Ready",
            fg="#66ffd6",
        )

        self.status_label.config(
            text=(
                "Webcam live prediction enabled. "
                "Building temporal probability window..."
            )
        )

        self.classifier_label.config(
            text=(
                "Active classifier: "
                "Webcam-Calibrated Model"
            )
        )

        self.clear_prediction()

        self.video_loop()


    # ========================================================================
    # VIDEO / WEBCAM LOOP
    # ========================================================================

    def video_loop(
        self,
    ) -> None:

        if (
            not self.running_video
            or self.capture is None
        ):
            return

        ret, frame = (
            self.capture.read()
        )

        if not ret:

            if self.current_source_type == "video":

                self.status_label.config(
                    text="Video ended."
                )

            else:

                self.status_label.config(
                    text="Webcam frame capture failed."
                )

            self.stop_video(
                update_status=False
            )

            return

        frame_rgb = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB,
        )

        pil_image = Image.fromarray(
            frame_rgb
        )

        self.current_frame = (
            pil_image
        )

        self.show_pil_image(
            pil_image
        )

        now = time.time()

        if (
            now
            - self.last_prediction_time
            >= LIVE_PREDICTION_INTERVAL_SEC
        ):

            self.last_prediction_time = (
                now
            )

            self.predict_current_frame_threaded(
                mode="Live",
                source_name=self.current_source_name,
            )

        self.root.after(
            30,
            self.video_loop,
        )


    # ========================================================================
    # PREVIEW
    # ========================================================================

    def show_pil_image(
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


    # ========================================================================
    # PREDICTION THREAD
    # ========================================================================

    def predict_current_frame_threaded(
        self,
        mode: str = "Manual",
        source_name: str | None = None,
    ) -> None:

        if self.prediction_busy:
            return

        if self.current_frame is None:

            messagebox.showerror(
                "Prediction Error",
                "No image/frame available for prediction.",
            )

            return

        # Critical:
        # snapshot the current frame so the worker is not racing with
        # subsequent webcam frames.
        frame_copy = (
            self.current_frame.copy()
        )

        source_type = (
            self.current_source_type
        )

        source_name = (
            source_name
            or self.current_source_name
        )

        self.prediction_busy = (
            True
        )

        threading.Thread(
            target=self.predict_current_frame,
            args=(
                frame_copy,
                mode,
                source_type,
                source_name,
            ),
            daemon=True,
        ).start()


    # ========================================================================
    # PREDICTION
    # ========================================================================

    def predict_current_frame(
        self,
        image: Image.Image,
        mode: str,
        source_type: str,
        source_name: str,
    ) -> None:

        try:

            self.root.after(
                0,
                lambda:
                self.status_label.config(
                    text=(
                        "Extracting CLIP "
                        "visual features..."
                    )
                ),
            )

            start = (
                time.perf_counter()
            )

            # ----------------------------------------------------------------
            # Exact same CLIP representation as training.
            # ----------------------------------------------------------------

            x = self.frame_to_features(
                image
            )

            # ----------------------------------------------------------------
            # Select classifier by source domain.
            # ----------------------------------------------------------------

            (
                pipeline,
                classifier_name,
            ) = self.get_active_pipeline(
                source_type
            )

            # ----------------------------------------------------------------
            # Probability distribution
            # ----------------------------------------------------------------

            raw_probabilities = (
                self.predict_probabilities(
                    pipeline,
                    x,
                )
            )

            raw_ranked = sorted(
                raw_probabilities.items(),
                key=lambda item: item[1],
                reverse=True,
            )

            raw_top_class = (
                raw_ranked[0][0]
            )

            raw_top_probability = float(
                raw_ranked[0][1]
            )

            # ----------------------------------------------------------------
            # Temporal smoothing ONLY for live webcam.
            #
            # Instead of:
            #
            #     class1, class2, class1 ...
            #
            # and then majority voting, average the actual model probabilities.
            # ----------------------------------------------------------------

            if (
                source_type == "webcam"
                and mode == "Live"
            ):

                display_probabilities = (
                    self.stabilize_webcam_probabilities(
                        raw_probabilities
                    )
                )

            else:

                display_probabilities = (
                    raw_probabilities.copy()
                )

            ranked = sorted(
                display_probabilities.items(),
                key=lambda item: item[1],
                reverse=True,
            )

            # ----------------------------------------------------------------
            # ALWAYS select the top member of the FOUR trained classes.
            #
            # No fifth "uncertain" class is introduced.
            # ----------------------------------------------------------------

            current_state = (
                ranked[0][0]
            )

            confidence = float(
                ranked[0][1]
            )

            second_class = (
                ranked[1][0]
            )

            second_probability = float(
                ranked[1][1]
            )

            gap = float(
                confidence
                - second_probability
            )

            level = confidence_level(
                confidence,
                gap,
            )

            runtime = (
                time.perf_counter()
                - start
            )

            result = {
                "mode":
                    mode,

                "source_type":
                    source_type,

                "source":
                    source_name,

                "classifier":
                    classifier_name,

                "current_state":
                    current_state,

                "raw_top_class":
                    raw_top_class,

                "raw_top_probability":
                    raw_top_probability,

                "confidence":
                    confidence,

                "confidence_percent":
                    confidence * 100.0,

                "confidence_level":
                    level,

                "second_class":
                    second_class,

                "second_probability":
                    second_probability,

                "confidence_gap":
                    gap,

                "probabilities":
                    display_probabilities,

                "raw_probabilities":
                    raw_probabilities,

                "feature_dimension":
                    int(x.shape[1]),

                "probability_window_size":
                    (
                        len(
                            self.webcam_probability_history
                        )
                        if (
                            source_type == "webcam"
                            and mode == "Live"
                        )
                        else 1
                    ),

                "runtime_seconds":
                    runtime,

                "device":
                    str(self.device),
            }

            self.log_prediction(
                result
            )

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
                        f"{mode} prediction complete."
                    )
                ),
            )

        except Exception as exc:

            error_message = str(
                exc
            )

            self.root.after(
                0,
                lambda:
                messagebox.showerror(
                    "Prediction Error",
                    error_message,
                ),
            )

            self.root.after(
                0,
                lambda:
                self.status_label.config(
                    text=(
                        "Prediction failed: "
                        f"{error_message}"
                    )
                ),
            )

        finally:

            self.prediction_busy = (
                False
            )


    # ========================================================================
    # UI RESULT
    # ========================================================================

    def update_prediction_ui(
        self,
        result: dict[str, Any],
    ) -> None:

        # --------------------------------------------------------------------
        # One of exactly four behavioural labels
        # --------------------------------------------------------------------

        self.state_label.config(
            text=(
                result[
                    "current_state"
                ]
                .upper()
            ),
            fg="#74f7ff",
        )

        self.confidence_label.config(
            text=(
                f"Confidence: "
                f"{result['confidence_percent']:.2f}%"
            )
        )

        level = (
            result[
                "confidence_level"
            ]
        )

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
                f"Prediction Confidence: "
                f"{level}"
            ),
            fg=colour,
        )

        self.classifier_label.config(
            text=(
                "Active classifier: "
                f"{result['classifier']}"
            )
        )

        # --------------------------------------------------------------------
        # Probability distribution
        # --------------------------------------------------------------------

        if (
            result["source_type"] == "webcam"
            and result["mode"] == "Live"
        ):

            heading = (
                "Stabilized webcam probability distribution:"
            )

        else:

            heading = (
                "Probability distribution:"
            )

        prob_lines = [
            heading,
            "",
        ]

        sorted_probs = sorted(
            result[
                "probabilities"
            ].items(),
            key=lambda item: item[1],
            reverse=True,
        )

        for label, probability in sorted_probs:

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

        # Show the current individual frame as diagnostic evidence.
        if (
            result["source_type"] == "webcam"
            and result["mode"] == "Live"
        ):

            prob_lines.extend(
                [
                    "",
                    (
                        "Current raw frame:"
                    ),
                ]
            )

            raw_sorted = sorted(
                result[
                    "raw_probabilities"
                ].items(),
                key=lambda item: item[1],
                reverse=True,
            )

            for label, probability in raw_sorted:

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

        # --------------------------------------------------------------------
        # Diagnostics
        # --------------------------------------------------------------------

        info_lines = [
            "Image / video diagnostics:",
            "",
            (
                f"Mode                    : "
                f"{result['mode']}"
            ),
            (
                f"Source type             : "
                f"{result['source_type']}"
            ),
            (
                f"Source                  : "
                f"{result['source']}"
            ),
            (
                f"Classifier              : "
                f"{result['classifier']}"
            ),
            (
                f"Displayed state         : "
                f"{result['current_state']}"
            ),
            (
                f"Raw top class           : "
                f"{result['raw_top_class']}"
            ),
            (
                f"Raw top probability     : "
                f"{result['raw_top_probability'] * 100:.2f}%"
            ),
            (
                f"Displayed confidence    : "
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
                f"Second probability      : "
                f"{result['second_probability'] * 100:.2f}%"
            ),
            (
                f"Confidence gap          : "
                f"{result['confidence_gap']:.4f}"
            ),
            (
                f"Feature dimension       : "
                f"{result['feature_dimension']}"
            ),
            (
                f"Probability window      : "
                f"{result['probability_window_size']}"
            ),
            (
                f"Runtime                 : "
                f"{result['runtime_seconds']:.4f} sec"
            ),
            (
                f"Device                  : "
                f"{result['device']}"
            ),
            (
                f"Logged to               : "
                f"{LOG_PATH}"
            ),
            (
                f"Timestamp               : "
                f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
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


    # ========================================================================
    # LOG RESULT
    # ========================================================================

    def log_prediction(
        self,
        result: dict[str, Any],
    ) -> None:

        with LOG_PATH.open(
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
                    result["mode"],
                    result["source_type"],
                    result["source"],
                    result["classifier"],
                    result["current_state"],
                    result["raw_top_class"],
                    result["raw_top_probability"],
                    result["confidence"],
                    result["confidence_level"],
                    result["second_class"],
                    result["second_probability"],
                    result["confidence_gap"],
                    result["feature_dimension"],
                    result["probability_window_size"],
                    result["runtime_seconds"],
                    result["device"],
                    json.dumps(
                        result["probabilities"]
                    ),
                ]
            )


    # ========================================================================
    # STOP
    # ========================================================================

    def stop_video(
        self,
        update_status: bool = True,
    ) -> None:

        self.running_video = (
            False
        )

        if self.capture is not None:

            self.capture.release()

            self.capture = (
                None
            )

        if update_status:

            self.status_label.config(
                text=(
                    "Video/webcam stopped."
                )
            )


    # ========================================================================
    # CLEAR RESULT
    # ========================================================================

    def clear_prediction(
        self,
    ) -> None:

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


    # ========================================================================
    # RESET
    # ========================================================================

    def reset(
        self,
    ) -> None:

        self.stop_video(
            update_status=False
        )

        self.current_frame = (
            None
        )

        self.preview_image = (
            None
        )

        self.current_source_type = (
            "none"
        )

        self.current_source_name = (
            "none"
        )

        self.prediction_busy = (
            False
        )

        self.last_prediction_time = (
            0.0
        )

        self.webcam_probability_history.clear()

        self.preview_label.config(
            image=""
        )

        self.image_ready_label.config(
            text="Visual Input: Missing",
            fg="#ffb3b3",
        )

        self.classifier_label.config(
            text="Active classifier: —"
        )

        self.status_label.config(
            text="System reset."
        )

        self.clear_prediction()


# ============================================================================
# MAIN
# ============================================================================

def main() -> None:

    root = tk.Tk()

    app = ImageDemoApp(
        root
    )

    def on_close() -> None:

        app.stop_video(
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
