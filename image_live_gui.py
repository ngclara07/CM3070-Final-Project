# image_live_gui.py

from __future__ import annotations

import csv
import json
import os
import threading
import time
from collections import deque
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
# Paths
# =============================================================================

MODEL_DIR = Path("models/image_demo")
MODEL_PATH = MODEL_DIR / "image_pipeline.joblib"
FEATURE_COLUMNS_PATH = MODEL_DIR / "feature_columns.json"

CLIP_MODEL_PATH = "models/clip-vit-large-patch14"

OUTPUT_DIR = Path("data/processed")
LOG_PATH = OUTPUT_DIR / "image_live_gui_predictions.csv"

WEBCAM_FRAME_DIR = OUTPUT_DIR / "webcam_debug_frames"


# =============================================================================
# Configuration
# =============================================================================

LABELS = [
    "focused",
    "distracted",
    "fatigued",
    "overloaded",
]

DISPLAY_SIZE = (360, 220)

# Do not immediately classify the first unstable webcam frames.
WEBCAM_WARMUP_SECONDS = 2.0

# Run expensive CLIP inference every N seconds.
LIVE_PREDICTION_INTERVAL_SEC = 1.0

# Temporal averaging is more appropriate than majority voting over labels.
TEMPORAL_PROBABILITY_WINDOW = 6

# Evidence requirements for displaying a definitive live state.
MIN_TOP_PROBABILITY = 0.35
MIN_CONFIDENCE_GAP = 0.08

# Existing descriptive confidence categories.
HIGH_GAP_THRESHOLD = 0.35
MEDIUM_GAP_THRESHOLD = 0.15


# =============================================================================
# Utility functions
# =============================================================================

def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def confidence_level(gap: float) -> str:
    if gap >= HIGH_GAP_THRESHOLD:
        return "High"

    if gap >= MEDIUM_GAP_THRESHOLD:
        return "Medium"

    return "Low"


def build_clip_model(device: torch.device):
    processor = CLIPProcessor.from_pretrained(CLIP_MODEL_PATH)

    model = CLIPModel.from_pretrained(CLIP_MODEL_PATH)
    model.to(device)
    model.eval()

    return model, processor


def extract_image_embedding_from_pil(
    image: Image.Image,
    model: CLIPModel,
    processor: CLIPProcessor,
    device: torch.device,
) -> np.ndarray:
    """
    Extract a normalized CLIP image embedding.

    This function is used for BOTH uploaded images and webcam frames so that
    preprocessing remains identical between both inference paths.
    """

    image = image.convert("RGB")

    inputs = processor(
        images=image,
        return_tensors="pt",
    )

    pixel_values = inputs["pixel_values"].to(device)

    with torch.inference_mode():
        try:
            output = model.get_image_features(
                pixel_values=pixel_values
            )

            if isinstance(output, torch.Tensor):
                image_features = output

            elif hasattr(output, "image_embeds"):
                image_features = output.image_embeds

            elif hasattr(output, "pooler_output"):
                image_features = output.pooler_output

            elif hasattr(output, "last_hidden_state"):
                image_features = output.last_hidden_state.mean(dim=1)

            else:
                raise TypeError(
                    f"Unsupported CLIP output type: {type(output)}"
                )

        except Exception:
            # Compatibility fallback for different transformers versions.
            output = model.vision_model(
                pixel_values=pixel_values
            )

            if hasattr(output, "pooler_output"):
                image_features = output.pooler_output

            elif hasattr(output, "last_hidden_state"):
                image_features = output.last_hidden_state.mean(dim=1)

            else:
                raise TypeError(
                    "Unsupported CLIP vision-model output type: "
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

    return embedding


# =============================================================================
# Application
# =============================================================================

class ImageDemoApp:
    def __init__(self, root: tk.Tk):
        self.root = root

        self.root.title("SenseFuzeAI Image Live GUI")
        self.root.geometry("1080x760")
        self.root.minsize(900, 720)
        self.root.configure(bg="#07111f")

        self.validate_required_files()

        OUTPUT_DIR.mkdir(
            parents=True,
            exist_ok=True,
        )

        WEBCAM_FRAME_DIR.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.initialise_log_file()

        # ---------------------------------------------------------------------
        # Load classifier
        # ---------------------------------------------------------------------

        self.pipeline = joblib.load(MODEL_PATH)

        with FEATURE_COLUMNS_PATH.open(
            "r",
            encoding="utf-8",
        ) as file:
            self.feature_columns = json.load(file)

        if len(self.feature_columns) != 768:
            raise ValueError(
                "Expected 768 CLIP image features, but "
                f"feature_columns.json contains {len(self.feature_columns)}."
            )

        # ---------------------------------------------------------------------
        # Load CLIP
        # ---------------------------------------------------------------------

        self.device = get_device()

        self.clip_model, self.clip_processor = build_clip_model(
            self.device
        )

        # ---------------------------------------------------------------------
        # Runtime state
        # ---------------------------------------------------------------------

        self.capture = None
        self.running_video = False

        self.current_frame: Image.Image | None = None
        self.preview_image = None

        self.frame_lock = threading.Lock()

        self.last_prediction_time = 0.0
        self.webcam_started_at = 0.0

        self.prediction_busy = False

        # Store probability dictionaries rather than labels.
        self.live_probability_history: deque[dict[str, float]] = deque(
            maxlen=TEMPORAL_PROBABILITY_WINDOW
        )

        self.last_source_name = None

        self.build_ui()

    # =========================================================================
    # Validation / logging
    # =========================================================================

    def validate_required_files(self) -> None:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(
                f"Image model not found: {MODEL_PATH}"
            )

        if not FEATURE_COLUMNS_PATH.exists():
            raise FileNotFoundError(
                f"Feature columns not found: {FEATURE_COLUMNS_PATH}"
            )

        if not Path(CLIP_MODEL_PATH).exists():
            raise FileNotFoundError(
                f"CLIP model not found: {CLIP_MODEL_PATH}"
            )

    def initialise_log_file(self) -> None:
        """
        Re-create the CSV if an older incompatible header is detected.
        """

        expected_header = [
            "timestamp",
            "mode",
            "source",
            "displayed_state",
            "raw_top_class",
            "raw_top_probability",
            "stabilized_top_class",
            "stabilized_top_probability",
            "confidence_level",
            "second_class",
            "confidence_gap",
            "feature_dimension",
            "runtime_seconds",
            "device",
            "raw_probabilities_json",
            "stabilized_probabilities_json",
        ]

        recreate = False

        if LOG_PATH.exists():
            try:
                with LOG_PATH.open(
                    "r",
                    newline="",
                    encoding="utf-8",
                ) as file:
                    reader = csv.reader(file)
                    existing_header = next(reader, [])

                recreate = existing_header != expected_header

            except Exception:
                recreate = True

        if recreate:
            backup_path = LOG_PATH.with_name(
                f"{LOG_PATH.stem}_old_"
                f"{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                f"{LOG_PATH.suffix}"
            )

            LOG_PATH.rename(backup_path)

        if not LOG_PATH.exists():
            with LOG_PATH.open(
                "w",
                newline="",
                encoding="utf-8",
            ) as file:
                writer = csv.writer(file)
                writer.writerow(expected_header)

    # =========================================================================
    # UI
    # =========================================================================

    def build_ui(self) -> None:
        tk.Label(
            self.root,
            text="SenseFuzeAI Image / Video Behavioural State Classifier",
            font=("Arial", 20, "bold"),
            fg="#74f7ff",
            bg="#07111f",
        ).pack(
            pady=(10, 4)
        )

        tk.Label(
            self.root,
            text=(
                "CLIP-based image inference with "
                "temporally stabilized live webcam prediction"
            ),
            font=("Arial", 11),
            fg="white",
            bg="#07111f",
        ).pack(
            pady=(0, 6)
        )

        # ---------------------------------------------------------------------
        # Status
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

        self.model_status_label = tk.Label(
            status_frame,
            text="Model Status: Loaded",
            font=("Arial", 11, "bold"),
            fg="#66ffd6",
            bg="#10203a",
        )

        self.model_status_label.grid(
            row=0,
            column=0,
            sticky="w",
            padx=8,
        )

        self.image_ready_label = tk.Label(
            status_frame,
            text="Image Readiness: Missing",
            font=("Arial", 11, "bold"),
            fg="#ffb3b3",
            bg="#10203a",
        )

        self.image_ready_label.grid(
            row=0,
            column=1,
            sticky="w",
            padx=24,
        )

        self.device_label = tk.Label(
            status_frame,
            text=f"Device: {self.device}",
            font=("Arial", 11, "bold"),
            fg="#74f7ff",
            bg="#10203a",
        )

        self.device_label.grid(
            row=0,
            column=2,
            sticky="w",
            padx=8,
        )

        # ---------------------------------------------------------------------
        # Buttons
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
            font=("Arial", 10, "bold"),
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
            font=("Arial", 10, "bold"),
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
            font=("Arial", 10, "bold"),
        ).grid(
            row=0,
            column=2,
            padx=4,
            pady=4,
        )

        tk.Button(
            button_frame,
            text="Save Webcam Frame",
            command=self.save_current_frame,
            width=18,
            bg="#7c4dff",
            fg="white",
            font=("Arial", 10, "bold"),
        ).grid(
            row=0,
            column=3,
            padx=4,
            pady=4,
        )

        tk.Button(
            button_frame,
            text="Stop",
            command=self.stop_video,
            width=9,
            bg="#c0392b",
            fg="white",
            font=("Arial", 10, "bold"),
        ).grid(
            row=0,
            column=4,
            padx=4,
            pady=4,
        )

        tk.Button(
            button_frame,
            text="Manual Prediction",
            command=self.predict_current_frame_threaded,
            width=18,
            bg="#2E86C1",
            fg="white",
            font=("Arial", 10, "bold"),
        ).grid(
            row=0,
            column=5,
            padx=4,
            pady=4,
        )

        tk.Button(
            button_frame,
            text="Reset",
            command=self.reset,
            width=10,
            bg="#4a5568",
            fg="white",
            font=("Arial", 10, "bold"),
        ).grid(
            row=0,
            column=6,
            padx=4,
            pady=4,
        )

        # ---------------------------------------------------------------------
        # Runtime status
        # ---------------------------------------------------------------------

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

        # ---------------------------------------------------------------------
        # Primary result
        # ---------------------------------------------------------------------

        result_frame = tk.Frame(
            self.root,
            bg="#10203a",
            padx=18,
            pady=14,
        )

        result_frame.pack(
            fill="x",
            padx=18,
            pady=8,
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
            pady=(4, 2)
        )

        self.confidence_label = tk.Label(
            result_frame,
            text="Confidence: —",
            font=("Arial", 18, "bold"),
            fg="white",
            bg="#10203a",
        )

        self.confidence_label.pack(
            pady=2
        )

        self.confidence_level_label = tk.Label(
            result_frame,
            text="Prediction Confidence: —",
            font=("Arial", 14, "bold"),
            fg="#cbd6ff",
            bg="#10203a",
        )

        self.confidence_level_label.pack(
            pady=2
        )

        # ---------------------------------------------------------------------
        # Lower area
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

        preview_frame = tk.Frame(
            lower_frame,
            bg="#07111f",
        )

        preview_frame.pack(
            side="left",
            fill="both",
            expand=False,
            padx=(0, 12),
        )

        self.preview_label = tk.Label(
            preview_frame,
            bg="#07111f",
        )

        self.preview_label.pack(
            pady=6
        )

        technical_frame = tk.LabelFrame(
            lower_frame,
            text="Technical Details",
            font=("Arial", 11, "bold"),
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
            font=("Consolas", 9),
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
            font=("Consolas", 9),
            bg="#0b1220",
            fg="#dbeafe",
        )

        self.info_text.pack(
            fill="both",
            expand=True,
            pady=4,
        )

    # =========================================================================
    # Source management
    # =========================================================================

    def reset_temporal_history(self) -> None:
        self.live_probability_history.clear()

    def choose_image(self) -> None:
        self.stop_video(
            update_status=False
        )

        self.reset_temporal_history()

        file_path = filedialog.askopenfilename(
            title="Select image file",
            filetypes=[
                (
                    "Image files",
                    "*.jpg *.jpeg *.png *.webp"
                ),
                (
                    "All files",
                    "*.*"
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

        with self.frame_lock:
            self.current_frame = image.copy()

        self.last_source_name = Path(file_path).name

        self.show_pil_image(
            image
        )

        self.image_ready_label.config(
            text="Image Readiness: Ready",
            fg="#66ffd6",
        )

        self.status_label.config(
            text=f"Loaded image: {Path(file_path).name}"
        )

        self.clear_prediction()

    def choose_video(self) -> None:
        self.stop_video(
            update_status=False
        )

        self.reset_temporal_history()

        file_path = filedialog.askopenfilename(
            title="Select video file",
            filetypes=[
                (
                    "Video files",
                    "*.mp4 *.avi *.mov *.mkv *.webm"
                ),
                (
                    "All files",
                    "*.*"
                ),
            ],
        )

        if not file_path:
            return

        self.capture = cv2.VideoCapture(
            file_path
        )

        if not self.capture.isOpened():
            self.capture = None

            messagebox.showerror(
                "Video Error",
                "Could not open selected video.",
            )

            return

        self.running_video = True
        self.last_prediction_time = 0.0
        self.webcam_started_at = time.monotonic()

        source_name = Path(file_path).name
        self.last_source_name = source_name

        self.image_ready_label.config(
            text="Image Readiness: Ready",
            fg="#66ffd6",
        )

        self.status_label.config(
            text=f"Playing video: {source_name}"
        )

        self.video_loop(
            source_name=source_name,
            webcam_mode=False,
        )

    def open_webcam(self):
        """
        Use DirectShow on Windows when possible because it often gives more
        predictable webcam startup behaviour. Fall back to the default backend.
        """

        if os.name == "nt":
            capture = cv2.VideoCapture(
                0,
                cv2.CAP_DSHOW,
            )

            if capture.isOpened():
                return capture

            capture.release()

        return cv2.VideoCapture(0)

    def start_webcam(self) -> None:
        self.stop_video(
            update_status=False
        )

        self.reset_temporal_history()

        self.capture = self.open_webcam()

        if self.capture is None or not self.capture.isOpened():
            self.capture = None

            messagebox.showerror(
                "Webcam Error",
                "Could not access webcam.",
            )

            return

        # Request a conventional webcam resolution.
        # The CLIP processor will still perform its own model-specific resize.
        self.capture.set(
            cv2.CAP_PROP_FRAME_WIDTH,
            640,
        )

        self.capture.set(
            cv2.CAP_PROP_FRAME_HEIGHT,
            480,
        )

        self.capture.set(
            cv2.CAP_PROP_FPS,
            30,
        )

        self.running_video = True

        self.last_prediction_time = 0.0
        self.webcam_started_at = time.monotonic()

        self.last_source_name = "webcam"

        self.image_ready_label.config(
            text="Image Readiness: Warming Up",
            fg="#ffd166",
        )

        self.status_label.config(
            text=(
                "Webcam started. Waiting briefly for "
                "exposure/autofocus stabilization..."
            )
        )

        self.video_loop(
            source_name="webcam",
            webcam_mode=True,
        )

    # =========================================================================
    # Video / webcam loop
    # =========================================================================

    def video_loop(
        self,
        source_name: str = "video",
        webcam_mode: bool = False,
    ) -> None:
        if not self.running_video:
            return

        if self.capture is None:
            return

        ret, frame = self.capture.read()

        if not ret:
            self.status_label.config(
                text="Video/webcam frame capture failed."
            )

            self.stop_video()
            return

        frame_rgb = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB,
        )

        pil_image = Image.fromarray(
            frame_rgb
        )

        # Use a copy so that the inference thread does not race with the
        # continually changing webcam frame.
        with self.frame_lock:
            self.current_frame = pil_image.copy()

        self.show_pil_image(
            pil_image
        )

        now = time.monotonic()

        if webcam_mode:
            elapsed = now - self.webcam_started_at

            if elapsed < WEBCAM_WARMUP_SECONDS:
                remaining = max(
                    0.0,
                    WEBCAM_WARMUP_SECONDS - elapsed,
                )

                self.status_label.config(
                    text=(
                        "Webcam stabilizing... "
                        f"{remaining:.1f} seconds"
                    )
                )

                self.root.after(
                    30,
                    lambda: self.video_loop(
                        source_name,
                        webcam_mode,
                    ),
                )

                return

            self.image_ready_label.config(
                text="Image Readiness: Ready",
                fg="#66ffd6",
            )

        if (
            now - self.last_prediction_time
            >= LIVE_PREDICTION_INTERVAL_SEC
        ):
            self.last_prediction_time = now

            self.predict_current_frame_threaded(
                mode="Live",
                source_name=source_name,
            )

        self.root.after(
            30,
            lambda: self.video_loop(
                source_name,
                webcam_mode,
            ),
        )

    # =========================================================================
    # Preview / debug saving
    # =========================================================================

    def show_pil_image(
        self,
        image: Image.Image,
    ) -> None:
        display = image.copy()

        display.thumbnail(
            DISPLAY_SIZE
        )

        self.preview_image = ImageTk.PhotoImage(
            display
        )

        self.preview_label.config(
            image=self.preview_image
        )

    def get_current_frame_copy(
        self,
    ) -> Image.Image | None:
        with self.frame_lock:
            if self.current_frame is None:
                return None

            return self.current_frame.copy()

    def save_current_frame(self) -> None:
        frame = self.get_current_frame_copy()

        if frame is None:
            messagebox.showwarning(
                "No Frame",
                "No image or webcam frame is currently available.",
            )
            return

        timestamp = datetime.now().strftime(
            "%Y%m%d_%H%M%S_%f"
        )

        output_path = (
            WEBCAM_FRAME_DIR
            / f"webcam_frame_{timestamp}.jpg"
        )

        frame.save(
            output_path,
            format="JPEG",
            quality=95,
        )

        self.status_label.config(
            text=f"Frame saved: {output_path}"
        )

        messagebox.showinfo(
            "Frame Saved",
            (
                "The exact current frame was saved.\n\n"
                f"{output_path}\n\n"
                "You can now load this same file with "
                "'Choose Image' and compare the prediction."
            ),
        )

    # =========================================================================
    # Feature extraction
    # =========================================================================

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

        if embedding.ndim != 1:
            raise ValueError(
                f"Expected 1-D CLIP embedding; received {embedding.shape}."
            )

        features = {
            f"image_clip_emb_{index}": float(value)
            for index, value in enumerate(embedding)
        }

        missing = [
            column
            for column in self.feature_columns
            if column not in features
        ]

        if missing:
            raise ValueError(
                "Image feature schema mismatch. "
                f"Missing columns: {missing[:20]}"
            )

        extra = [
            column
            for column in features
            if column not in self.feature_columns
        ]

        if extra:
            raise ValueError(
                "Image feature schema mismatch. "
                f"Unexpected columns: {extra[:20]}"
            )

        return pd.DataFrame(
            [
                [
                    features[column]
                    for column in self.feature_columns
                ]
            ],
            columns=self.feature_columns,
        )

    # =========================================================================
    # Prediction helpers
    # =========================================================================

    def classifier_probabilities(
        self,
        x: pd.DataFrame,
    ) -> dict[str, float]:
        if not hasattr(
            self.pipeline,
            "predict_proba",
        ):
            raise AttributeError(
                "The loaded image pipeline does not provide predict_proba()."
            )

        probabilities = self.pipeline.predict_proba(
            x
        )[0]

        classes = self.pipeline.classes_

        result = {
            str(label): float(probability)
            for label, probability
            in zip(classes, probabilities)
        }

        # Ensure expected classes exist.
        for label in LABELS:
            result.setdefault(
                label,
                0.0,
            )

        total = sum(
            result.values()
        )

        if total <= 0:
            raise ValueError(
                "Classifier returned an invalid probability distribution."
            )

        return {
            label: value / total
            for label, value in result.items()
        }

    def average_live_probabilities(
        self,
    ) -> dict[str, float]:
        if not self.live_probability_history:
            return {
                label: 0.0
                for label in LABELS
            }

        averaged = {}

        for label in LABELS:
            values = [
                probabilities.get(
                    label,
                    0.0,
                )
                for probabilities
                in self.live_probability_history
            ]

            averaged[label] = float(
                np.mean(values)
            )

        total = sum(
            averaged.values()
        )

        if total > 0:
            averaged = {
                label: probability / total
                for label, probability
                in averaged.items()
            }

        return averaged

    @staticmethod
    def rank_probabilities(
        probabilities: dict[str, float],
    ):
        return sorted(
            probabilities.items(),
            key=lambda item: item[1],
            reverse=True,
        )

    def decide_displayed_state(
        self,
        probabilities: dict[str, float],
    ):
        ranked = self.rank_probabilities(
            probabilities
        )

        top_class, top_probability = ranked[0]

        if len(ranked) >= 2:
            second_class, second_probability = ranked[1]
        else:
            second_class, second_probability = "none", 0.0

        gap = float(
            top_probability - second_probability
        )

        sufficiently_confident = (
            top_probability >= MIN_TOP_PROBABILITY
            and gap >= MIN_CONFIDENCE_GAP
        )

        if sufficiently_confident:
            displayed_state = top_class
        else:
            displayed_state = "uncertain"

        return (
            displayed_state,
            top_class,
            float(top_probability),
            second_class,
            float(second_probability),
            gap,
        )

    # =========================================================================
    # Prediction
    # =========================================================================

    def predict_current_frame_threaded(
        self,
        mode: str = "Manual",
        source_name: str = "current frame",
    ) -> None:
        if self.prediction_busy:
            return

        frame = self.get_current_frame_copy()

        if frame is None:
            messagebox.showwarning(
                "Prediction",
                "No image/frame is available for prediction.",
            )

            return

        threading.Thread(
            target=self.predict_frame,
            args=(
                frame,
                mode,
                source_name,
            ),
            daemon=True,
        ).start()

    def predict_frame(
        self,
        frame: Image.Image,
        mode: str,
        source_name: str,
    ) -> None:
        try:
            self.prediction_busy = True

            self.root.after(
                0,
                lambda: self.status_label.config(
                    text="Extracting CLIP visual features..."
                ),
            )

            start = time.perf_counter()

            x = self.frame_to_features(
                frame
            )

            raw_probabilities = self.classifier_probabilities(
                x
            )

            raw_ranked = self.rank_probabilities(
                raw_probabilities
            )

            raw_top_class, raw_top_probability = raw_ranked[0]

            # -----------------------------------------------------------------
            # Live webcam/video: temporal probability averaging
            # -----------------------------------------------------------------

            if mode == "Live":
                self.live_probability_history.append(
                    dict(raw_probabilities)
                )

                stabilized_probabilities = (
                    self.average_live_probabilities()
                )

            else:
                # Manual/upload predictions remain single-image predictions.
                stabilized_probabilities = dict(
                    raw_probabilities
                )

            (
                displayed_state,
                stabilized_top_class,
                stabilized_top_probability,
                second_class,
                second_probability,
                confidence_gap,
            ) = self.decide_displayed_state(
                stabilized_probabilities
            )

            runtime = (
                time.perf_counter()
                - start
            )

            result = {
                "mode": mode,
                "source": source_name,

                "displayed_state": displayed_state,

                "raw_top_class": raw_top_class,
                "raw_top_probability": float(
                    raw_top_probability
                ),

                "stabilized_top_class": stabilized_top_class,
                "stabilized_top_probability": float(
                    stabilized_top_probability
                ),

                "confidence_percent": float(
                    stabilized_top_probability * 100
                ),

                "confidence_level": confidence_level(
                    confidence_gap
                ),

                "second_class": second_class,
                "second_probability": second_probability,
                "confidence_gap": confidence_gap,

                "raw_probabilities": raw_probabilities,
                "stabilized_probabilities": stabilized_probabilities,

                "history_length": len(
                    self.live_probability_history
                ),

                "temporal_window": (
                    TEMPORAL_PROBABILITY_WINDOW
                    if mode == "Live"
                    else "N/A"
                ),

                "min_top_probability": MIN_TOP_PROBABILITY,
                "min_confidence_gap": MIN_CONFIDENCE_GAP,

                "feature_dimension": int(
                    x.shape[1]
                ),

                "runtime_seconds": runtime,
                "device": str(
                    self.device
                ),
            }

            self.root.after(
                0,
                lambda: self.update_prediction_ui(
                    result
                ),
            )

            self.log_prediction(
                result
            )

            self.root.after(
                0,
                lambda: self.status_label.config(
                    text=(
                        f"{mode} prediction complete. "
                        f"Raw: {raw_top_class} | "
                        f"Stabilized: {stabilized_top_class}"
                    )
                ),
            )

        except Exception as exc:
            error_message = str(exc)

            self.root.after(
                0,
                lambda message=error_message: messagebox.showerror(
                    "Prediction Error",
                    message,
                ),
            )

            self.root.after(
                0,
                lambda: self.status_label.config(
                    text="Prediction failed."
                ),
            )

        finally:
            self.prediction_busy = False

    # =========================================================================
    # UI result rendering
    # =========================================================================

    def update_prediction_ui(
        self,
        result: dict,
    ) -> None:
        displayed_state = result[
            "displayed_state"
        ]

        self.state_label.config(
            text=displayed_state.upper()
        )

        if displayed_state == "uncertain":
            self.state_label.config(
                fg="#ffd166"
            )
        else:
            self.state_label.config(
                fg="#74f7ff"
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
            text=f"Prediction Confidence: {level}",
            fg=colour,
        )

        # ---------------------------------------------------------------------
        # Probability panel
        # ---------------------------------------------------------------------

        prob_lines = [
            "Stabilized probability distribution:",
            "",
        ]

        stabilized_ranked = self.rank_probabilities(
            result["stabilized_probabilities"]
        )

        for label, probability in stabilized_ranked:
            bar_length = int(
                probability * 30
            )

            bar = "█" * bar_length

            prob_lines.append(
                f"{label:12s}: "
                f"{probability * 100:6.2f}%  "
                f"{bar}"
            )

        prob_lines.extend(
            [
                "",
                "Current raw frame:",
            ]
        )

        raw_ranked = self.rank_probabilities(
            result["raw_probabilities"]
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
            "\n".join(prob_lines),
        )

        # ---------------------------------------------------------------------
        # Diagnostics
        # ---------------------------------------------------------------------

        info_lines = [
            "Image/video diagnostics:",
            "",
            f"Mode                    : {result['mode']}",
            f"Source                  : {result['source']}",
            f"Displayed state         : {result['displayed_state']}",
            f"Raw top class           : {result['raw_top_class']}",
            f"Raw top probability     : {result['raw_top_probability'] * 100:.2f}%",
            f"Stabilized top class    : {result['stabilized_top_class']}",
            f"Stabilized probability  : {result['stabilized_top_probability'] * 100:.2f}%",
            f"Second class            : {result['second_class']}",
            f"Confidence gap          : {result['confidence_gap']:.4f}",
            f"Confidence level        : {result['confidence_level']}",
            f"Required top probability: {result['min_top_probability']:.2f}",
            f"Required gap            : {result['min_confidence_gap']:.2f}",
            f"Temporal history        : {result['history_length']}",
            f"Temporal window         : {result['temporal_window']}",
            f"Feature dimension       : {result['feature_dimension']}",
            f"Runtime                 : {result['runtime_seconds']:.4f} sec",
            f"Device                  : {result['device']}",
            f"Logged to               : {LOG_PATH}",
            f"Timestamp               : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        ]

        self.info_text.delete(
            "1.0",
            tk.END,
        )

        self.info_text.insert(
            tk.END,
            "\n".join(info_lines),
        )

    # =========================================================================
    # Logging
    # =========================================================================

    def log_prediction(
        self,
        result: dict,
    ) -> None:
        with LOG_PATH.open(
            "a",
            newline="",
            encoding="utf-8",
        ) as file:
            writer = csv.writer(
                file
            )

            writer.writerow(
                [
                    datetime.now().astimezone().isoformat(),
                    result["mode"],
                    result["source"],
                    result["displayed_state"],
                    result["raw_top_class"],
                    result["raw_top_probability"],
                    result["stabilized_top_class"],
                    result["stabilized_top_probability"],
                    result["confidence_level"],
                    result["second_class"],
                    result["confidence_gap"],
                    result["feature_dimension"],
                    result["runtime_seconds"],
                    result["device"],
                    json.dumps(
                        result["raw_probabilities"]
                    ),
                    json.dumps(
                        result["stabilized_probabilities"]
                    ),
                ]
            )

    # =========================================================================
    # Stop/reset
    # =========================================================================

    def stop_video(
        self,
        update_status: bool = True,
    ) -> None:
        self.running_video = False

        if self.capture is not None:
            self.capture.release()
            self.capture = None

        if update_status:
            self.status_label.config(
                text="Video/webcam stopped."
            )

    def clear_prediction(self) -> None:
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

    def reset(self) -> None:
        self.stop_video(
            update_status=False
        )

        with self.frame_lock:
            self.current_frame = None

        self.preview_image = None

        self.prediction_busy = False

        self.last_prediction_time = 0.0
        self.webcam_started_at = 0.0

        self.reset_temporal_history()

        self.last_source_name = None

        self.preview_label.config(
            image=""
        )

        self.image_ready_label.config(
            text="Image Readiness: Missing",
            fg="#ffb3b3",
        )

        self.status_label.config(
            text="System reset."
        )

        self.clear_prediction()


# =============================================================================
# Entry point
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
