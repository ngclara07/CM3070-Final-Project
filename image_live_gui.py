# image_live_gui.py

from __future__ import annotations

import csv
import json
import threading
import time
from collections import Counter
from datetime import datetime
from pathlib import Path
from tkinter import filedialog, messagebox
import tkinter as tk

import cv2
import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from PIL import Image, ImageTk
from transformers import CLIPModel, CLIPProcessor


# =============================================================================
# PROJECT PATHS
# =============================================================================

ROOT_DIR = Path(__file__).resolve().parent

MODEL_DIR = ROOT_DIR / "models" / "image_demo"

# Original image classifier.
# This artifact remains untouched.
ORIGINAL_MODEL_PATH = (
    MODEL_DIR
    / "image_pipeline.joblib"
)

# Newly calibrated webcam/video classifier.
CALIBRATED_MODEL_PATH = (
    MODEL_DIR
    / "image_pipeline_webcam_calibrated.joblib"
)

FEATURE_COLUMNS_PATH = (
    MODEL_DIR
    / "feature_columns.json"
)

# Same pretrained CLIP visual encoder used during training/calibration.
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


# =============================================================================
# CONFIGURATION
# =============================================================================

LABELS = [
    "focused",
    "distracted",
    "fatigued",
    "overloaded",
]

EXPECTED_FEATURE_DIMENSION = 768

LIVE_PREDICTION_INTERVAL_SEC = 1.0

# A short smoothing window stabilises live webcam/video output without
# locking the prediction into one class for too long.
PREDICTION_SMOOTHING_WINDOW = 3

# If the top two classes are too close, display UNCERTAIN.
UNCERTAINTY_GAP_THRESHOLD = 0.10

# Optional minimum top-class probability.
MIN_TOP_PROBABILITY = 0.40

DISPLAY_SIZE = (
    360,
    220,
)

torch.set_num_threads(2)


# =============================================================================
# DEVICE
# =============================================================================

def get_device() -> torch.device:
    return torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )


# =============================================================================
# CONFIDENCE
# =============================================================================

def confidence_level(
    gap: float,
) -> str:
    if gap >= 0.35:
        return "High"

    if gap >= 0.15:
        return "Medium"

    return "Low"


# =============================================================================
# CLIP
# =============================================================================

def build_clip_model(
    device: torch.device,
):
    processor = (
        CLIPProcessor.from_pretrained(
            str(CLIP_MODEL_PATH)
        )
    )

    model = (
        CLIPModel.from_pretrained(
            str(CLIP_MODEL_PATH)
        )
    )

    model.to(
        device
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
    Extract the same 768-dimensional normalized CLIP embedding used by the
    image training and webcam calibration pipelines.
    """

    image = image.convert(
        "RGB"
    )

    inputs = processor(
        images=image,
        return_tensors="pt",
    )

    pixel_values = inputs[
        "pixel_values"
    ].to(
        device
    )

    with torch.inference_mode():
        try:
            output = (
                model.get_image_features(
                    pixel_values=pixel_values
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
                    output.last_hidden_state
                    .mean(dim=1)
                )

            else:
                raise TypeError(
                    "Unsupported CLIP output type: "
                    f"{type(output)}"
                )

        except Exception:
            # Compatibility fallback for transformer versions where
            # get_image_features() returns a different wrapper object.
            output = model.vision_model(
                pixel_values=pixel_values
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
                    output.last_hidden_state
                    .mean(dim=1)
                )

            else:
                raise TypeError(
                    "Unsupported CLIP vision output type: "
                    f"{type(output)}"
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
    )

    if embedding.ndim != 1:
        raise ValueError(
            "CLIP embedding should be one-dimensional. "
            f"Received shape {embedding.shape}."
        )

    if embedding.shape[0] != EXPECTED_FEATURE_DIMENSION:
        raise ValueError(
            "Unexpected CLIP embedding dimension. "
            f"Expected {EXPECTED_FEATURE_DIMENSION}, "
            f"received {embedding.shape[0]}."
        )

    if not np.all(
        np.isfinite(
            embedding
        )
    ):
        raise ValueError(
            "CLIP embedding contains NaN or infinity."
        )

    return embedding


# =============================================================================
# APPLICATION
# =============================================================================

class ImageDemoApp:
    def __init__(
        self,
        root: tk.Tk,
    ):
        self.root = root

        self.root.title(
            "SenseFuzeAI Image Live GUI"
        )

        self.root.geometry(
            "1080x760"
        )

        self.root.minsize(
            900,
            720,
        )

        self.root.configure(
            bg="#07111f"
        )

        # ---------------------------------------------------------------------
        # Validate model artifacts
        # ---------------------------------------------------------------------

        if not ORIGINAL_MODEL_PATH.exists():
            raise FileNotFoundError(
                "Original image model not found:\n"
                f"{ORIGINAL_MODEL_PATH}"
            )

        if not CALIBRATED_MODEL_PATH.exists():
            raise FileNotFoundError(
                "Webcam-calibrated image model not found:\n"
                f"{CALIBRATED_MODEL_PATH}\n\n"
                "Run retrain_image_webcam_calibrated.py first."
            )

        if not FEATURE_COLUMNS_PATH.exists():
            raise FileNotFoundError(
                "Image feature schema not found:\n"
                f"{FEATURE_COLUMNS_PATH}"
            )

        if not CLIP_MODEL_PATH.exists():
            raise FileNotFoundError(
                "Pretrained CLIP model not found:\n"
                f"{CLIP_MODEL_PATH}"
            )

        # ---------------------------------------------------------------------
        # Output/logging
        # ---------------------------------------------------------------------

        OUTPUT_DIR.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.initialise_log_file()

        # ---------------------------------------------------------------------
        # Load BOTH classifiers
        # ---------------------------------------------------------------------

        self.original_pipeline = joblib.load(
            ORIGINAL_MODEL_PATH
        )

        self.calibrated_pipeline = joblib.load(
            CALIBRATED_MODEL_PATH
        )

        # ---------------------------------------------------------------------
        # Load feature schema
        # ---------------------------------------------------------------------

        with FEATURE_COLUMNS_PATH.open(
            "r",
            encoding="utf-8",
        ) as f:
            self.feature_columns = json.load(
                f
            )

        if len(
            self.feature_columns
        ) != EXPECTED_FEATURE_DIMENSION:
            raise ValueError(
                "Unexpected image feature schema size.\n"
                f"Expected {EXPECTED_FEATURE_DIMENSION}, "
                f"received {len(self.feature_columns)}."
            )

        # ---------------------------------------------------------------------
        # Load frozen pretrained CLIP
        # ---------------------------------------------------------------------

        self.device = get_device()

        (
            self.clip_model,
            self.clip_processor,
        ) = build_clip_model(
            self.device
        )

        # ---------------------------------------------------------------------
        # Runtime state
        # ---------------------------------------------------------------------

        self.capture = None

        self.running_video = False

        self.current_frame: (
            Image.Image
            | None
        ) = None

        self.preview_image = None

        self.last_prediction_time = 0.0

        self.prediction_busy = False

        self.prediction_history: list[str] = []

        # Source can be:
        #
        #   none
        #   image
        #   video
        #   webcam
        #
        self.current_source_type = "none"

        self.current_source_name = "none"

        self.build_ui()


    # =========================================================================
    # LOGGING
    # =========================================================================

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
            writer = csv.writer(
                f
            )

            writer.writerow(
                [
                    "timestamp",
                    "mode",
                    "source_type",
                    "source",
                    "model_used",
                    "current_state",
                    "raw_top_class",
                    "confidence",
                    "confidence_level",
                    "second_class",
                    "second_probability",
                    "confidence_gap",
                    "uncertain",
                    "feature_dimension",
                    "runtime_seconds",
                    "device",
                    "probabilities_json",
                ]
            )


    # =========================================================================
    # USER INTERFACE
    # =========================================================================

    def build_ui(
        self,
    ) -> None:
        # ---------------------------------------------------------------------
        # Header
        # ---------------------------------------------------------------------

        tk.Label(
            self.root,
            text=(
                "SenseFuzeAI Image / Video "
                "Behavioural State Classifier"
            ),
            font=(
                "Arial",
                20,
                "bold",
            ),
            fg="#74f7ff",
            bg="#07111f",
        ).pack(
            pady=(
                10,
                4,
            )
        )

        tk.Label(
            self.root,
            text=(
                "Original image classifier + "
                "webcam-calibrated live classifier "
                "using frozen CLIP ViT-L/14 embeddings"
            ),
            font=(
                "Arial",
                10,
            ),
            fg="white",
            bg="#07111f",
        ).pack(
            pady=(
                0,
                6,
            )
        )

        # ---------------------------------------------------------------------
        # System status
        # ---------------------------------------------------------------------

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

        self.original_model_status_label = tk.Label(
            status_frame,
            text="Original Model: Loaded",
            font=(
                "Arial",
                10,
                "bold",
            ),
            fg="#66ffd6",
            bg="#10203a",
        )

        self.original_model_status_label.grid(
            row=0,
            column=0,
            sticky="w",
            padx=8,
        )

        self.calibrated_model_status_label = tk.Label(
            status_frame,
            text="Webcam Model: Loaded",
            font=(
                "Arial",
                10,
                "bold",
            ),
            fg="#66ffd6",
            bg="#10203a",
        )

        self.calibrated_model_status_label.grid(
            row=0,
            column=1,
            sticky="w",
            padx=18,
        )

        self.image_ready_label = tk.Label(
            status_frame,
            text="Image Readiness: Missing",
            font=(
                "Arial",
                10,
                "bold",
            ),
            fg="#ffb3b3",
            bg="#10203a",
        )

        self.image_ready_label.grid(
            row=0,
            column=2,
            sticky="w",
            padx=18,
        )

        self.device_label = tk.Label(
            status_frame,
            text=f"Device: {self.device}",
            font=(
                "Arial",
                10,
                "bold",
            ),
            fg="#74f7ff",
            bg="#10203a",
        )

        self.device_label.grid(
            row=0,
            column=3,
            sticky="w",
            padx=8,
        )

        # ---------------------------------------------------------------------
        # Controls
        # ---------------------------------------------------------------------

        button_frame = tk.Frame(
            self.root,
            bg="#07111f",
        )

        button_frame.pack(
            pady=6
        )

        tk.Button(
            button_frame,
            text="Choose Image",
            command=self.choose_image,
            width=15,
            font=(
                "Arial",
                10,
                "bold",
            ),
        ).grid(
            row=0,
            column=0,
            padx=4,
            pady=4,
        )

        tk.Button(
            button_frame,
            text="Choose Video",
            command=self.choose_video,
            width=15,
            font=(
                "Arial",
                10,
                "bold",
            ),
        ).grid(
            row=0,
            column=1,
            padx=4,
            pady=4,
        )

        tk.Button(
            button_frame,
            text="Start Webcam",
            command=self.start_webcam,
            width=15,
            bg="#00a884",
            fg="white",
            font=(
                "Arial",
                10,
                "bold",
            ),
        ).grid(
            row=0,
            column=2,
            padx=4,
            pady=4,
        )

        tk.Button(
            button_frame,
            text="Stop",
            command=self.stop_video,
            width=10,
            bg="#c0392b",
            fg="white",
            font=(
                "Arial",
                10,
                "bold",
            ),
        ).grid(
            row=0,
            column=3,
            padx=4,
            pady=4,
        )

        tk.Button(
            button_frame,
            text="Manual Prediction",
            command=self.predict_current_frame_threaded,
            width=19,
            bg="#2E86C1",
            fg="white",
            font=(
                "Arial",
                10,
                "bold",
            ),
        ).grid(
            row=0,
            column=4,
            padx=4,
            pady=4,
        )

        tk.Button(
            button_frame,
            text="Reset",
            command=self.reset,
            width=11,
            bg="#4a5568",
            fg="white",
            font=(
                "Arial",
                10,
                "bold",
            ),
        ).grid(
            row=0,
            column=5,
            padx=4,
            pady=4,
        )

        # ---------------------------------------------------------------------
        # Status text
        # ---------------------------------------------------------------------

        self.status_label = tk.Label(
            self.root,
            text="System ready.",
            fg="#cbd6ff",
            bg="#07111f",
            font=(
                "Arial",
                10,
            ),
        )

        self.status_label.pack(
            pady=4
        )

        self.active_model_label = tk.Label(
            self.root,
            text="Active classifier: —",
            fg="#ffd166",
            bg="#07111f",
            font=(
                "Arial",
                10,
                "bold",
            ),
        )

        self.active_model_label.pack(
            pady=2
        )

        # ---------------------------------------------------------------------
        # Main result
        # ---------------------------------------------------------------------

        result_frame = tk.Frame(
            self.root,
            bg="#10203a",
            padx=18,
            pady=12,
        )

        result_frame.pack(
            fill="x",
            padx=18,
            pady=6,
        )

        tk.Label(
            result_frame,
            text="Current Behavioural State",
            font=(
                "Arial",
                12,
                "bold",
            ),
            fg="#cbd6ff",
            bg="#10203a",
        ).pack()

        self.state_label = tk.Label(
            result_frame,
            text="—",
            font=(
                "Arial",
                32,
                "bold",
            ),
            fg="#74f7ff",
            bg="#10203a",
        )

        self.state_label.pack(
            pady=(
                4,
                1,
            )
        )

        self.confidence_label = tk.Label(
            result_frame,
            text="Confidence: —",
            font=(
                "Arial",
                17,
                "bold",
            ),
            fg="white",
            bg="#10203a",
        )

        self.confidence_label.pack(
            pady=1
        )

        self.confidence_level_label = tk.Label(
            result_frame,
            text="Prediction Confidence: —",
            font=(
                "Arial",
                13,
                "bold",
            ),
            fg="#cbd6ff",
            bg="#10203a",
        )

        self.confidence_level_label.pack(
            pady=1
        )

        # ---------------------------------------------------------------------
        # Lower frame
        # ---------------------------------------------------------------------

        lower_frame = tk.Frame(
            self.root,
            bg="#07111f",
        )

        lower_frame.pack(
            fill="both",
            expand=True,
            padx=18,
            pady=6,
        )

        # Preview
        preview_frame = tk.Frame(
            lower_frame,
            bg="#07111f",
        )

        preview_frame.pack(
            side="left",
            fill="both",
            expand=False,
            padx=(
                0,
                12,
            ),
        )

        self.preview_label = tk.Label(
            preview_frame,
            bg="#07111f",
        )

        self.preview_label.pack(
            pady=6
        )

        # Technical details
        technical_frame = tk.LabelFrame(
            lower_frame,
            text="Technical Details",
            font=(
                "Arial",
                11,
                "bold",
            ),
            fg="#74f7ff",
            bg="#07111f",
            padx=10,
            pady=8,
        )

        technical_frame.pack(
            side="left",
            fill="both",
            expand=True,
        )

        self.prob_text = tk.Text(
            technical_frame,
            height=7,
            width=80,
            font=(
                "Consolas",
                9,
            ),
            bg="#0b1220",
            fg="#dbeafe",
        )

        self.prob_text.pack(
            fill="both",
            expand=True,
            pady=4,
        )

        self.info_text = tk.Text(
            technical_frame,
            height=7,
            width=80,
            font=(
                "Consolas",
                9,
            ),
            bg="#0b1220",
            fg="#dbeafe",
        )

        self.info_text.pack(
            fill="both",
            expand=True,
            pady=4,
        )


    # =========================================================================
    # SOURCE / CLASSIFIER SELECTION
    # =========================================================================

    def get_active_pipeline(
        self,
    ):
        """
        Select the classifier according to the current input domain.

        Still image:
            original image model

        Video/webcam:
            webcam-calibrated model
        """

        if self.current_source_type == "image":
            return (
                self.original_pipeline,
                "original_image_model",
                ORIGINAL_MODEL_PATH,
            )

        if self.current_source_type in {
            "video",
            "webcam",
        }:
            return (
                self.calibrated_pipeline,
                "webcam_calibrated_model",
                CALIBRATED_MODEL_PATH,
            )

        raise ValueError(
            "No valid image/video/webcam source is active."
        )


    # =========================================================================
    # IMAGE INPUT
    # =========================================================================

    def choose_image(
        self,
    ) -> None:
        self.stop_video(
            update_status=False
        )

        self.prediction_history = []

        file_path = (
            filedialog.askopenfilename(
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
        )

        if not file_path:
            return

        try:
            image = (
                Image.open(
                    file_path
                )
                .convert(
                    "RGB"
                )
            )

            self.current_frame = image

            self.current_source_type = (
                "image"
            )

            self.current_source_name = (
                Path(
                    file_path
                ).name
            )

            self.show_pil_image(
                image
            )

            self.image_ready_label.config(
                text="Image Readiness: Ready",
                fg="#66ffd6",
            )

            self.active_model_label.config(
                text=(
                    "Active classifier: "
                    "Original Image Model"
                )
            )

            self.status_label.config(
                text=(
                    f"Loaded image: "
                    f"{self.current_source_name}"
                )
            )

            self.clear_prediction()

        except Exception as exc:
            messagebox.showerror(
                "Image Error",
                str(exc),
            )


    # =========================================================================
    # VIDEO INPUT
    # =========================================================================

    def choose_video(
        self,
    ) -> None:
        self.stop_video(
            update_status=False
        )

        self.prediction_history = []

        file_path = (
            filedialog.askopenfilename(
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
        )

        if not file_path:
            return

        self.capture = (
            cv2.VideoCapture(
                file_path
            )
        )

        if not self.capture.isOpened():
            self.capture = None

            messagebox.showerror(
                "Video Error",
                "Could not open selected video.",
            )

            return

        self.running_video = True

        self.current_source_type = (
            "video"
        )

        self.current_source_name = (
            Path(
                file_path
            ).name
        )

        self.prediction_history = []

        self.last_prediction_time = 0.0

        self.image_ready_label.config(
            text="Image Readiness: Ready",
            fg="#66ffd6",
        )

        self.active_model_label.config(
            text=(
                "Active classifier: "
                "Webcam-Calibrated Model"
            )
        )

        self.status_label.config(
            text=(
                f"Playing video: "
                f"{self.current_source_name}"
            )
        )

        self.video_loop()


    # =========================================================================
    # WEBCAM INPUT
    # =========================================================================

    def start_webcam(
        self,
    ) -> None:
        self.stop_video(
            update_status=False
        )

        self.prediction_history = []

        # CAP_DSHOW is generally more stable on Windows.
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

            messagebox.showerror(
                "Webcam Error",
                "Could not access webcam.",
            )

            return

        # Set a conventional capture resolution.
        capture.set(
            cv2.CAP_PROP_FRAME_WIDTH,
            640,
        )

        capture.set(
            cv2.CAP_PROP_FRAME_HEIGHT,
            480,
        )

        self.capture = capture

        self.running_video = True

        self.current_source_type = (
            "webcam"
        )

        self.current_source_name = (
            "webcam"
        )

        self.last_prediction_time = 0.0

        self.image_ready_label.config(
            text="Image Readiness: Ready",
            fg="#66ffd6",
        )

        self.active_model_label.config(
            text=(
                "Active classifier: "
                "Webcam-Calibrated Model"
            )
        )

        self.status_label.config(
            text=(
                "Webcam live prediction enabled."
            )
        )

        self.video_loop()


    # =========================================================================
    # VIDEO LOOP
    # =========================================================================

    def video_loop(
        self,
    ) -> None:
        if (
            not self.running_video
            or self.capture is None
        ):
            return

        success, frame = (
            self.capture.read()
        )

        if (
            not success
            or frame is None
        ):
            self.status_label.config(
                text="Video/frame capture ended."
            )

            self.stop_video(
                update_status=False
            )

            return

        frame_rgb = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB,
        )

        pil_image = (
            Image.fromarray(
                frame_rgb
            )
        )

        self.current_frame = (
            pil_image
        )

        self.show_pil_image(
            pil_image
        )

        now = time.monotonic()

        if (
            now
            - self.last_prediction_time
            >= LIVE_PREDICTION_INTERVAL_SEC
        ):
            self.last_prediction_time = now

            self.predict_current_frame_threaded(
                mode="Live"
            )

        self.root.after(
            30,
            self.video_loop,
        )


    # =========================================================================
    # PREVIEW
    # =========================================================================

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


    # =========================================================================
    # FEATURE EXTRACTION
    # =========================================================================

    def frame_to_features(
        self,
        image: Image.Image,
    ) -> pd.DataFrame:
        embedding = (
            extract_image_embedding_from_pil(
                image=image,
                model=self.clip_model,
                processor=self.clip_processor,
                device=self.device,
            )
        )

        features = {
            f"image_clip_emb_{index}": float(
                value
            )
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
                "Missing image feature columns:\n"
                f"{missing[:20]}"
            )

        x = pd.DataFrame(
            [
                [
                    features[
                        column
                    ]
                    for column
                    in self.feature_columns
                ]
            ],
            columns=self.feature_columns,
        )

        return x


    # =========================================================================
    # PREDICTION THREAD
    # =========================================================================

    def predict_current_frame_threaded(
        self,
        mode: str = "Manual",
    ) -> None:
        if self.prediction_busy:
            return

        if self.current_frame is None:
            if mode == "Manual":
                messagebox.showerror(
                    "Prediction Error",
                    "No image/frame available for prediction.",
                )

            return

        # Copy the frame so the webcam thread cannot change it while CLIP
        # inference is running.
        frame_copy = (
            self.current_frame.copy()
        )

        source_type = (
            self.current_source_type
        )

        source_name = (
            self.current_source_name
        )

        thread = threading.Thread(
            target=self.predict_current_frame,
            args=(
                frame_copy,
                mode,
                source_type,
                source_name,
            ),
            daemon=True,
        )

        thread.start()


    # =========================================================================
    # PREDICTION
    # =========================================================================

    def predict_current_frame(
        self,
        frame: Image.Image,
        mode: str,
        source_type: str,
        source_name: str,
    ) -> None:
        try:
            self.prediction_busy = True

            self.root.after(
                0,
                lambda: self.status_label.config(
                    text=(
                        "Extracting CLIP visual features..."
                    )
                ),
            )

            start = (
                time.perf_counter()
            )

            # -------------------------------------------------------------
            # Feature extraction
            # -------------------------------------------------------------

            x = self.frame_to_features(
                frame
            )

            # -------------------------------------------------------------
            # Source-specific classifier
            # -------------------------------------------------------------

            if source_type == "image":
                pipeline = (
                    self.original_pipeline
                )

                model_used = (
                    "original_image_model"
                )

                model_path = (
                    ORIGINAL_MODEL_PATH
                )

            elif source_type in {
                "video",
                "webcam",
            }:
                pipeline = (
                    self.calibrated_pipeline
                )

                model_used = (
                    "webcam_calibrated_model"
                )

                model_path = (
                    CALIBRATED_MODEL_PATH
                )

            else:
                raise ValueError(
                    "Unknown source type: "
                    f"{source_type}"
                )

            # -------------------------------------------------------------
            # Classification
            # -------------------------------------------------------------

            prediction = (
                pipeline.predict(
                    x
                )[0]
            )

            if not hasattr(
                pipeline,
                "predict_proba",
            ):
                raise AttributeError(
                    "Selected image classifier does not expose "
                    "predict_proba()."
                )

            probabilities = (
                pipeline.predict_proba(
                    x
                )[0]
            )

            classes = (
                pipeline.classes_
            )

            probability_dict = {
                str(class_label): float(
                    probability
                )
                for class_label, probability
                in zip(
                    classes,
                    probabilities,
                )
            }

            sorted_probs = sorted(
                probability_dict.items(),
                key=lambda item: item[1],
                reverse=True,
            )

            top_class, top_prob = (
                sorted_probs[0]
            )

            if len(
                sorted_probs
            ) >= 2:
                (
                    second_class,
                    second_prob,
                ) = sorted_probs[1]

            else:
                second_class = "none"
                second_prob = 0.0

            gap = float(
                top_prob
                - second_prob
            )

            # -------------------------------------------------------------
            # Uncertainty handling
            # -------------------------------------------------------------

            uncertain = (
                gap
                < UNCERTAINTY_GAP_THRESHOLD
                or top_prob
                < MIN_TOP_PROBABILITY
            )

            if uncertain:
                current_state = (
                    "uncertain"
                )

            elif (
                mode == "Live"
                and source_type
                in {
                    "video",
                    "webcam",
                }
            ):
                self.prediction_history.append(
                    str(
                        top_class
                    )
                )

                if (
                    len(
                        self.prediction_history
                    )
                    > PREDICTION_SMOOTHING_WINDOW
                ):
                    self.prediction_history.pop(
                        0
                    )

                current_state = (
                    Counter(
                        self.prediction_history
                    )
                    .most_common(
                        1
                    )[0][0]
                )

            else:
                current_state = (
                    str(
                        top_class
                    )
                )

            runtime = (
                time.perf_counter()
                - start
            )

            result = {
                "mode": mode,

                "source_type": (
                    source_type
                ),

                "source": (
                    source_name
                ),

                "model_used": (
                    model_used
                ),

                "model_path": (
                    str(
                        model_path
                    )
                ),

                "prediction": (
                    str(
                        prediction
                    )
                ),

                "current_state": (
                    current_state
                ),

                "raw_top_class": (
                    str(
                        top_class
                    )
                ),

                "confidence": float(
                    top_prob
                ),

                "confidence_percent": float(
                    top_prob
                    * 100
                ),

                "confidence_level": (
                    confidence_level(
                        gap
                    )
                ),

                "second_class": (
                    str(
                        second_class
                    )
                ),

                "second_probability": float(
                    second_prob
                ),

                "confidence_gap": (
                    gap
                ),

                "uncertain": bool(
                    uncertain
                ),

                "probabilities": (
                    probability_dict
                ),

                "feature_dimension": int(
                    x.shape[1]
                ),

                "runtime_seconds": float(
                    runtime
                ),

                "device": str(
                    self.device
                ),

                "smoothing_window": (
                    PREDICTION_SMOOTHING_WINDOW
                    if (
                        mode == "Live"
                        and source_type
                        in {
                            "video",
                            "webcam",
                        }
                    )
                    else "N/A"
                ),

                "recent_live_history": (
                    list(
                        self.prediction_history
                    )
                    if (
                        mode == "Live"
                        and source_type
                        in {
                            "video",
                            "webcam",
                        }
                    )
                    else "N/A"
                ),
            }

            self.log_prediction(
                result
            )

            self.root.after(
                0,
                lambda result=result:
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
                lambda error_message=error_message:
                messagebox.showerror(
                    "Prediction Error",
                    error_message,
                ),
            )

            self.root.after(
                0,
                lambda:
                self.status_label.config(
                    text="Prediction failed."
                ),
            )

        finally:
            self.prediction_busy = False


    # =========================================================================
    # RESULT UI
    # =========================================================================

    def update_prediction_ui(
        self,
        result: dict,
    ) -> None:
        # ---------------------------------------------------------------------
        # Behavioural state
        # ---------------------------------------------------------------------

        self.state_label.config(
            text=(
                result[
                    "current_state"
                ]
                .upper()
            )
        )

        if (
            result[
                "current_state"
            ]
            == "uncertain"
        ):
            self.state_label.config(
                fg="#ffd166"
            )

        else:
            self.state_label.config(
                fg="#74f7ff"
            )

        # ---------------------------------------------------------------------
        # Confidence
        # ---------------------------------------------------------------------

        self.confidence_label.config(
            text=(
                "Confidence: "
                f"{result['confidence_percent']:.2f}%"
            )
        )

        level = (
            result[
                "confidence_level"
            ]
        )

        level_colour = {
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
            fg=level_colour,
        )

        # ---------------------------------------------------------------------
        # Model indicator
        # ---------------------------------------------------------------------

        display_model = (
            "Original Image Model"
            if result[
                "model_used"
            ]
            == "original_image_model"
            else "Webcam-Calibrated Model"
        )

        self.active_model_label.config(
            text=(
                "Active classifier: "
                f"{display_model}"
            )
        )

        # ---------------------------------------------------------------------
        # Probability distribution
        # ---------------------------------------------------------------------

        prob_lines = [
            "Probability distribution:",
            "",
        ]

        sorted_probabilities = sorted(
            result[
                "probabilities"
            ].items(),
            key=lambda item:
            item[1],
            reverse=True,
        )

        for (
            label,
            probability,
        ) in sorted_probabilities:
            bar_length = int(
                probability
                * 30
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
            "Image / video diagnostics:",
            "",
            (
                f"Mode                   : "
                f"{result['mode']}"
            ),
            (
                f"Source type            : "
                f"{result['source_type']}"
            ),
            (
                f"Source                 : "
                f"{result['source']}"
            ),
            (
                f"Classifier             : "
                f"{display_model}"
            ),
            (
                f"Displayed state        : "
                f"{result['current_state']}"
            ),
            (
                f"Raw top class          : "
                f"{result['raw_top_class']}"
            ),
            (
                f"Confidence             : "
                f"{result['confidence_percent']:.2f}%"
            ),
            (
                f"Confidence level       : "
                f"{result['confidence_level']}"
            ),
            (
                f"Second-highest class   : "
                f"{result['second_class']}"
            ),
            (
                f"Second probability     : "
                f"{result['second_probability'] * 100:.2f}%"
            ),
            (
                f"Confidence gap         : "
                f"{result['confidence_gap']:.4f}"
            ),
            (
                f"Uncertain              : "
                f"{result['uncertain']}"
            ),
            (
                f"Gap threshold          : "
                f"{UNCERTAINTY_GAP_THRESHOLD:.2f}"
            ),
            (
                f"Minimum top probability: "
                f"{MIN_TOP_PROBABILITY:.2f}"
            ),
            (
                f"Feature dimension      : "
                f"{result['feature_dimension']}"
            ),
            (
                f"Smoothing window       : "
                f"{result['smoothing_window']}"
            ),
            (
                f"Recent live history    : "
                f"{result['recent_live_history']}"
            ),
            (
                f"Runtime                : "
                f"{result['runtime_seconds']:.4f} sec"
            ),
            (
                f"Device                 : "
                f"{result['device']}"
            ),
            (
                f"CLIP encoder           : "
                "clip-vit-large-patch14"
            ),
            (
                f"Logged to              : "
                f"{LOG_PATH}"
            ),
            (
                f"Timestamp              : "
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


    # =========================================================================
    # LOG PREDICTION
    # =========================================================================

    def log_prediction(
        self,
        result: dict,
    ) -> None:
        with LOG_PATH.open(
            "a",
            newline="",
            encoding="utf-8",
        ) as f:
            writer = csv.writer(
                f
            )

            writer.writerow(
                [
                    datetime.now()
                    .astimezone()
                    .isoformat(),

                    result[
                        "mode"
                    ],

                    result[
                        "source_type"
                    ],

                    result[
                        "source"
                    ],

                    result[
                        "model_used"
                    ],

                    result[
                        "current_state"
                    ],

                    result[
                        "raw_top_class"
                    ],

                    result[
                        "confidence"
                    ],

                    result[
                        "confidence_level"
                    ],

                    result[
                        "second_class"
                    ],

                    result[
                        "second_probability"
                    ],

                    result[
                        "confidence_gap"
                    ],

                    result[
                        "uncertain"
                    ],

                    result[
                        "feature_dimension"
                    ],

                    result[
                        "runtime_seconds"
                    ],

                    result[
                        "device"
                    ],

                    json.dumps(
                        result[
                            "probabilities"
                        ]
                    ),
                ]
            )


    # =========================================================================
    # STOP / RESET
    # =========================================================================

    def stop_video(
        self,
        update_status: bool = True,
    ) -> None:
        self.running_video = False

        if self.capture is not None:
            try:
                self.capture.release()
            except Exception:
                pass

            self.capture = None

        self.prediction_history = []

        if update_status:
            self.status_label.config(
                text="Video/webcam stopped."
            )


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


    def reset(
        self,
    ) -> None:
        self.stop_video(
            update_status=False
        )

        self.current_frame = None

        self.preview_image = None

        self.prediction_busy = False

        self.last_prediction_time = 0.0

        self.prediction_history = []

        self.current_source_type = (
            "none"
        )

        self.current_source_name = (
            "none"
        )

        self.preview_label.config(
            image=""
        )

        self.image_ready_label.config(
            text="Image Readiness: Missing",
            fg="#ffb3b3",
        )

        self.active_model_label.config(
            text="Active classifier: —"
        )

        self.status_label.config(
            text="System reset."
        )

        self.clear_prediction()


# =============================================================================
# MAIN
# =============================================================================

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
