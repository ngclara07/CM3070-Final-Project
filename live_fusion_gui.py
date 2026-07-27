# image_live_gui.py

from __future__ import annotations

import csv
import json
import threading
import time
from collections import Counter, deque
from datetime import datetime
from pathlib import Path
from tkinter import filedialog, messagebox
import tkinter as tk

import cv2
import joblib
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image, ImageTk
from transformers import CLIPModel, CLIPProcessor


# ============================================================
# Paths
# ============================================================

MODEL_DIR = Path("models/image_demo")

# Original static-image classifier
ORIGINAL_MODEL_PATH = (
    MODEL_DIR / "image_pipeline.joblib"
)

# Webcam/video calibrated classifier
WEBCAM_CALIBRATED_MODEL_PATH = (
    MODEL_DIR / "image_pipeline_webcam_calibrated.joblib"
)

FEATURE_COLUMNS_PATH = (
    MODEL_DIR / "feature_columns.json"
)

CLIP_MODEL_PATH = Path(
    "models/clip-vit-large-patch14"
)

OUTPUT_DIR = Path("data/processed")
LOG_PATH = OUTPUT_DIR / "image_live_gui_predictions.csv"


# ============================================================
# Configuration
# ============================================================

LIVE_PREDICTION_INTERVAL_SEC = 1.0

# Smooth predictions for video/webcam.
PREDICTION_SMOOTHING_WINDOW = 5

DISPLAY_SIZE = (360, 220)

LABELS = [
    "focused",
    "distracted",
    "fatigued",
    "overloaded",
]


# ============================================================
# Utility functions
# ============================================================

def get_device() -> torch.device:
    return torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )


def confidence_level(gap: float) -> str:
    if gap >= 0.35:
        return "High"

    if gap >= 0.15:
        return "Medium"

    return "Low"


def build_clip_model(
    device: torch.device,
) -> tuple[CLIPModel, CLIPProcessor]:

    processor = CLIPProcessor.from_pretrained(
        str(CLIP_MODEL_PATH)
    )

    model = CLIPModel.from_pretrained(
        str(CLIP_MODEL_PATH)
    )

    model.to(device)
    model.eval()

    return model, processor


def extract_image_embedding_from_pil(
    image: Image.Image,
    model: CLIPModel,
    processor: CLIPProcessor,
    device: torch.device,
):
    """
    Extract one normalized CLIP ViT-L/14 image embedding.

    The fallback path supports Transformers versions where
    get_image_features() may return a structured model output
    rather than a raw torch.Tensor.
    """

    image = image.convert("RGB")

    inputs = processor(
        images=image,
        return_tensors="pt",
    )

    pixel_values = inputs["pixel_values"].to(
        device
    )

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
                image_features = (
                    output.last_hidden_state.mean(dim=1)
                )

            else:
                raise TypeError(
                    "Unsupported CLIP image-feature "
                    f"output type: {type(output)}"
                )

        except Exception:
            vision_output = model.vision_model(
                pixel_values=pixel_values
            )

            if (
                hasattr(vision_output, "pooler_output")
                and vision_output.pooler_output
                is not None
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
                    "Unsupported CLIP vision-model "
                    f"output type: {type(vision_output)}"
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

    return embedding


# ============================================================
# Main GUI
# ============================================================

class ImageDemoApp:
    def __init__(
        self,
        root: tk.Tk,
    ):
        self.root = root

        self.root.title(
            "SenseFuzeAI Image / Video Live GUI"
        )

        self.root.geometry("1180x780")
        self.root.minsize(960, 720)

        self.root.configure(
            bg="#07111f"
        )

        # ----------------------------------------------------
        # Validate required artifacts
        # ----------------------------------------------------

        if not ORIGINAL_MODEL_PATH.exists():
            raise FileNotFoundError(
                "Original image classifier not found:\n"
                f"{ORIGINAL_MODEL_PATH}"
            )

        if not WEBCAM_CALIBRATED_MODEL_PATH.exists():
            raise FileNotFoundError(
                "Webcam-calibrated classifier not found:\n"
                f"{WEBCAM_CALIBRATED_MODEL_PATH}"
            )

        if not FEATURE_COLUMNS_PATH.exists():
            raise FileNotFoundError(
                "Image feature schema not found:\n"
                f"{FEATURE_COLUMNS_PATH}"
            )

        if not CLIP_MODEL_PATH.exists():
            raise FileNotFoundError(
                "CLIP model not found:\n"
                f"{CLIP_MODEL_PATH}"
            )

        # ----------------------------------------------------
        # Output
        # ----------------------------------------------------

        OUTPUT_DIR.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.initialise_log_file()

        # ----------------------------------------------------
        # Load classifiers
        # ----------------------------------------------------

        self.original_pipeline = joblib.load(
            ORIGINAL_MODEL_PATH
        )

        self.webcam_pipeline = joblib.load(
            WEBCAM_CALIBRATED_MODEL_PATH
        )

        with FEATURE_COLUMNS_PATH.open(
            "r",
            encoding="utf-8",
        ) as f:
            self.feature_columns = json.load(f)

        # ----------------------------------------------------
        # CLIP
        # ----------------------------------------------------

        self.device = get_device()

        (
            self.clip_model,
            self.clip_processor,
        ) = build_clip_model(
            self.device
        )

        # ----------------------------------------------------
        # Runtime state
        # ----------------------------------------------------

        self.capture: cv2.VideoCapture | None = None

        self.running_video = False

        self.current_frame: Image.Image | None = None

        self.preview_image = None

        self.current_source_name = "none"

        # image | video | webcam | none
        self.current_source_type = "none"

        self.last_prediction_time = 0.0

        self.prediction_busy = False

        self.prediction_history: deque[str] = deque(
            maxlen=PREDICTION_SMOOTHING_WINDOW
        )

        self.build_ui()

    # ========================================================
    # Logging
    # ========================================================

    def initialise_log_file(self) -> None:

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
                    "confidence",
                    "confidence_level",
                    "second_class",
                    "confidence_gap",
                    "feature_dimension",
                    "runtime_seconds",
                    "device",
                    "probabilities_json",
                ]
            )

    # ========================================================
    # GUI construction
    # ========================================================

    def build_ui(self) -> None:

        # ----------------------------------------------------
        # Header
        # ----------------------------------------------------

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
            pady=(10, 4)
        )

        tk.Label(
            self.root,
            text=(
                "Static Image · Uploaded Video · "
                "Live Webcam · CLIP Visual Embeddings"
            ),
            font=("Arial", 11),
            fg="white",
            bg="#07111f",
        ).pack(
            pady=(0, 6)
        )

        # ----------------------------------------------------
        # Status panel
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

        self.model_status_label = tk.Label(
            status_frame,
            text="Models: Loaded",
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
            text="Visual Input: Missing",
            font=("Arial", 11, "bold"),
            fg="#ffb3b3",
            bg="#10203a",
        )

        self.image_ready_label.grid(
            row=0,
            column=1,
            sticky="w",
            padx=20,
        )

        self.classifier_label = tk.Label(
            status_frame,
            text="Classifier: —",
            font=("Arial", 11, "bold"),
            fg="#ffd166",
            bg="#10203a",
        )

        self.classifier_label.grid(
            row=0,
            column=2,
            sticky="w",
            padx=20,
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
            column=3,
            sticky="w",
            padx=8,
        )

        # ----------------------------------------------------
        # Buttons
        # ----------------------------------------------------

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
            padx=5,
            pady=4,
        )

        # NEW
        tk.Button(
            button_frame,
            text="Choose Video",
            command=self.choose_video,
            width=15,
            bg="#6c5ce7",
            fg="white",
            font=("Arial", 10, "bold"),
        ).grid(
            row=0,
            column=1,
            padx=5,
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
            padx=5,
            pady=4,
        )

        tk.Button(
            button_frame,
            text="Stop Video",
            command=self.stop_video_source,
            width=13,
            bg="#c0392b",
            fg="white",
            font=("Arial", 10, "bold"),
        ).grid(
            row=0,
            column=3,
            padx=5,
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
            column=4,
            padx=5,
            pady=4,
        )

        tk.Button(
            button_frame,
            text="Reset",
            command=self.reset,
            width=11,
            bg="#4a5568",
            fg="white",
            font=("Arial", 10, "bold"),
        ).grid(
            row=0,
            column=5,
            padx=5,
            pady=4,
        )

        # ----------------------------------------------------
        # Runtime status
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
        # Prediction result
        # ----------------------------------------------------

        result_frame = tk.Frame(
            self.root,
            bg="#10203a",
            padx=18,
            pady=12,
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

        # ----------------------------------------------------
        # Lower panel
        # ----------------------------------------------------

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
        preview_frame = tk.LabelFrame(
            lower_frame,
            text="Visual Preview",
            font=("Arial", 11, "bold"),
            fg="#74f7ff",
            bg="#07111f",
            padx=8,
            pady=8,
        )

        preview_frame.pack(
            side="left",
            fill="y",
            padx=(0, 12),
        )

        self.preview_label = tk.Label(
            preview_frame,
            bg="#07111f",
        )

        self.preview_label.pack(
            pady=6
        )

        self.source_label = tk.Label(
            preview_frame,
            text="Source: none",
            font=("Arial", 9),
            fg="#cbd6ff",
            bg="#07111f",
            wraplength=340,
        )

        self.source_label.pack(
            pady=4
        )

        # Technical details
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
            width=75,
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
            width=75,
            font=("Consolas", 9),
            bg="#0b1220",
            fg="#dbeafe",
        )

        self.info_text.pack(
            fill="both",
            expand=True,
            pady=4,
        )

    # ========================================================
    # Classifier routing
    # ========================================================

    def get_active_pipeline(self):
        """
        Static image:
            original classifier

        Uploaded video:
            webcam-calibrated classifier

        Webcam:
            webcam-calibrated classifier
        """

        if self.current_source_type == "image":
            return (
                self.original_pipeline,
                "Original Static-Image Model",
            )

        if self.current_source_type in {
            "video",
            "webcam",
        }:
            return (
                self.webcam_pipeline,
                "Webcam-Calibrated Model",
            )

        # Defensive fallback
        return (
            self.original_pipeline,
            "Original Static-Image Model",
        )

    # ========================================================
    # Static image
    # ========================================================

    def choose_image(self) -> None:

        self.stop_video_source(
            update_status=False
        )

        self.prediction_history.clear()

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
            ).convert("RGB")

            self.current_frame = image.copy()

            self.current_source_type = "image"
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

            self.classifier_label.config(
                text=(
                    "Classifier: "
                    "Original Static-Image Model"
                ),
                fg="#74f7ff",
            )

            self.source_label.config(
                text=(
                    f"Source: Image · "
                    f"{self.current_source_name}"
                )
            )

            self.status_label.config(
                text=(
                    "Static image loaded. "
                    "Ready for manual prediction."
                )
            )

            self.clear_prediction()

        except Exception as exc:
            messagebox.showerror(
                "Image Error",
                str(exc),
            )

    # ========================================================
    # Uploaded video
    # ========================================================

    def choose_video(self) -> None:
        """
        Load and play a user-selected video.

        Each sampled frame is evaluated automatically using
        the webcam-calibrated image classifier.
        """

        self.stop_video_source(
            update_status=False
        )

        self.prediction_history.clear()

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

        self.current_source_type = "video"
        self.current_source_name = (
            Path(file_path).name
        )

        self.running_video = True
        self.last_prediction_time = 0.0

        self.image_ready_label.config(
            text="Visual Input: Ready",
            fg="#66ffd6",
        )

        self.classifier_label.config(
            text=(
                "Classifier: "
                "Webcam-Calibrated Model"
            ),
            fg="#ffd166",
        )

        self.source_label.config(
            text=(
                f"Source: Video · "
                f"{self.current_source_name}"
            )
        )

        self.status_label.config(
            text=(
                "Uploaded video running with "
                "live behavioural prediction."
            )
        )

        self.clear_prediction()

        self.video_loop()

    # ========================================================
    # Webcam
    # ========================================================

    def start_webcam(self) -> None:

        self.stop_video_source(
            update_status=False
        )

        self.prediction_history.clear()

        capture = cv2.VideoCapture(0)

        if not capture.isOpened():

            capture.release()

            messagebox.showerror(
                "Webcam Error",
                "Could not access webcam.",
            )

            return

        # Optional webcam settings.
        capture.set(
            cv2.CAP_PROP_FRAME_WIDTH,
            640,
        )

        capture.set(
            cv2.CAP_PROP_FRAME_HEIGHT,
            480,
        )

        self.capture = capture

        self.current_source_type = "webcam"
        self.current_source_name = "webcam"

        self.running_video = True
        self.last_prediction_time = 0.0

        self.image_ready_label.config(
            text="Visual Input: Ready",
            fg="#66ffd6",
        )

        self.classifier_label.config(
            text=(
                "Classifier: "
                "Webcam-Calibrated Model"
            ),
            fg="#ffd166",
        )

        self.source_label.config(
            text="Source: Live Webcam"
        )

        self.status_label.config(
            text=(
                "Webcam live behavioural "
                "prediction enabled."
            )
        )

        self.clear_prediction()

        self.video_loop()

    # ========================================================
    # Shared webcam/video loop
    # ========================================================

    def video_loop(self) -> None:
        """
        Shared playback loop for uploaded videos and webcam.
        """

        if (
            not self.running_video
            or self.capture is None
        ):
            return

        ret, frame = self.capture.read()

        if not ret:

            if self.current_source_type == "video":

                self.running_video = False

                self.capture.release()
                self.capture = None

                self.status_label.config(
                    text="Uploaded video finished."
                )

                return

            self.status_label.config(
                text="Unable to read webcam frame."
            )

            self.stop_video_source()

            return

        frame_rgb = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB,
        )

        pil_image = Image.fromarray(
            frame_rgb
        )

        # Keep an independent current frame.
        self.current_frame = pil_image.copy()

        self.show_pil_image(
            pil_image
        )

        current_time = time.time()

        if (
            current_time
            - self.last_prediction_time
            >= LIVE_PREDICTION_INTERVAL_SEC
        ):

            self.last_prediction_time = (
                current_time
            )

            self.predict_current_frame_threaded(
                mode="Live",
                source_name=self.current_source_name,
            )

        # Approx. 30 FPS preview.
        self.root.after(
            30,
            self.video_loop,
        )

    # ========================================================
    # Preview
    # ========================================================

    def show_pil_image(
        self,
        image: Image.Image,
    ) -> None:

        display = image.copy()

        display.thumbnail(
            DISPLAY_SIZE,
            Image.Resampling.LANCZOS,
        )

        self.preview_image = ImageTk.PhotoImage(
            display
        )

        self.preview_label.config(
            image=self.preview_image
        )

    # ========================================================
    # CLIP feature extraction
    # ========================================================

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
            f"image_clip_emb_{i}": float(value)
            for i, value
            in enumerate(embedding)
        }

        missing = [
            column
            for column in self.feature_columns
            if column not in features
        ]

        if missing:
            raise ValueError(
                "Missing image feature columns.\n"
                f"First missing columns: "
                f"{missing[:20]}"
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

    # ========================================================
    # Prediction
    # ========================================================

    def predict_current_frame_threaded(
        self,
        mode: str = "Manual",
        source_name: str | None = None,
    ) -> None:

        if self.prediction_busy:
            return

        if self.current_frame is None:

            messagebox.showwarning(
                "Prediction",
                "No image or video frame is available.",
            )

            return

        # Important:
        # Copy the frame so the video loop cannot replace it
        # while the worker is performing inference.
        frame_snapshot = (
            self.current_frame.copy()
        )

        if source_name is None:
            source_name = (
                self.current_source_name
            )

        thread = threading.Thread(
            target=self.predict_current_frame,
            args=(
                frame_snapshot,
                mode,
                source_name,
            ),
            daemon=True,
        )

        thread.start()

    def predict_current_frame(
        self,
        frame_snapshot: Image.Image,
        mode: str,
        source_name: str,
    ) -> None:

        try:
            self.prediction_busy = True

            self.root.after(
                0,
                lambda: self.status_label.config(
                    text=(
                        "Extracting CLIP visual "
                        "features..."
                    )
                ),
            )

            start_time = time.perf_counter()

            x = self.frame_to_features(
                frame_snapshot
            )

            (
                active_pipeline,
                classifier_name,
            ) = self.get_active_pipeline()

            prediction = (
                active_pipeline.predict(x)[0]
            )

            probabilities = (
                active_pipeline.predict_proba(x)[0]
            )

            classes = active_pipeline.classes_

            sorted_probs = sorted(
                zip(
                    classes,
                    probabilities,
                ),
                key=lambda item: item[1],
                reverse=True,
            )

            if len(sorted_probs) < 2:
                raise ValueError(
                    "Classifier returned fewer than "
                    "two probability classes."
                )

            top_class = str(
                sorted_probs[0][0]
            )

            top_prob = float(
                sorted_probs[0][1]
            )

            second_class = str(
                sorted_probs[1][0]
            )

            second_prob = float(
                sorted_probs[1][1]
            )

            gap = (
                top_prob
                - second_prob
            )

            # ------------------------------------------------
            # Static image:
            # directly show classifier prediction.
            #
            # Video/webcam:
            # majority smoothing across recent frame labels.
            # ------------------------------------------------

            if (
                mode == "Live"
                and self.current_source_type
                in {"video", "webcam"}
            ):

                self.prediction_history.append(
                    top_class
                )

                current_state = Counter(
                    self.prediction_history
                ).most_common(1)[0][0]

            else:

                current_state = top_class

            runtime = (
                time.perf_counter()
                - start_time
            )

            probability_dict = {
                str(label): float(prob)
                for label, prob
                in sorted_probs
            }

            result = {
                "mode": mode,
                "source_type": (
                    self.current_source_type
                ),
                "source": source_name,
                "classifier": classifier_name,
                "prediction": str(prediction),
                "current_state": current_state,
                "raw_top_class": top_class,
                "confidence": top_prob,
                "confidence_percent": (
                    top_prob * 100.0
                ),
                "confidence_level": (
                    confidence_level(gap)
                ),
                "second_class": second_class,
                "second_probability": (
                    second_prob
                ),
                "confidence_gap": gap,
                "probabilities": (
                    probability_dict
                ),
                "feature_dimension": int(
                    x.shape[1]
                ),
                "runtime_seconds": runtime,
                "device": str(self.device),
                "smoothing_window": (
                    PREDICTION_SMOOTHING_WINDOW
                    if mode == "Live"
                    else "N/A"
                ),
                "recent_live_history": (
                    list(
                        self.prediction_history
                    )
                    if mode == "Live"
                    else "N/A"
                ),
            }

            self.log_prediction(
                result
            )

            self.root.after(
                0,
                lambda: self.update_prediction_ui(
                    result
                ),
            )

            self.root.after(
                0,
                lambda: self.status_label.config(
                    text=(
                        f"{mode} prediction complete."
                    )
                ),
            )

        except Exception as exc:

            error_message = str(exc)

            self.root.after(
                0,
                lambda msg=error_message:
                messagebox.showerror(
                    "Prediction Error",
                    msg,
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

    # ========================================================
    # Prediction display
    # ========================================================

    def update_prediction_ui(
        self,
        result: dict,
    ) -> None:

        self.state_label.config(
            text=(
                result["current_state"]
                .upper()
            ),
            fg="#74f7ff",
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

        self.classifier_label.config(
            text=(
                "Classifier: "
                f"{result['classifier']}"
            )
        )

        # ----------------------------------------------------
        # Probability distribution
        # ----------------------------------------------------

        prob_lines = [
            "Probability distribution:",
            "",
        ]

        for (
            label,
            probability,
        ) in result[
            "probabilities"
        ].items():

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

        self.prob_text.delete(
            "1.0",
            tk.END,
        )

        self.prob_text.insert(
            tk.END,
            "\n".join(prob_lines),
        )

        # ----------------------------------------------------
        # Diagnostics
        # ----------------------------------------------------

        info_lines = [
            "Image / video diagnostics:",
            "",
            (
                f"Mode                  : "
                f"{result['mode']}"
            ),
            (
                f"Source type           : "
                f"{result['source_type']}"
            ),
            (
                f"Source                : "
                f"{result['source']}"
            ),
            (
                f"Classifier            : "
                f"{result['classifier']}"
            ),
            (
                f"Displayed state       : "
                f"{result['current_state']}"
            ),
            (
                f"Raw top class         : "
                f"{result['raw_top_class']}"
            ),
            (
                f"Confidence            : "
                f"{result['confidence_percent']:.2f}%"
            ),
            (
                f"Confidence level      : "
                f"{result['confidence_level']}"
            ),
            (
                f"Second-highest class  : "
                f"{result['second_class']}"
            ),
            (
                f"Confidence gap        : "
                f"{result['confidence_gap']:.4f}"
            ),
            (
                f"Feature dimension     : "
                f"{result['feature_dimension']}"
            ),
            (
                f"Smoothing window      : "
                f"{result['smoothing_window']}"
            ),
            (
                f"Recent live history   : "
                f"{result['recent_live_history']}"
            ),
            (
                f"Runtime               : "
                f"{result['runtime_seconds']:.4f} sec"
            ),
            (
                f"Device                : "
                f"{result['device']}"
            ),
            (
                f"Logged to             : "
                f"{LOG_PATH}"
            ),
            (
                f"Timestamp             : "
                f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
            ),
        ]

        self.info_text.delete(
            "1.0",
            tk.END,
        )

        self.info_text.insert(
            tk.END,
            "\n".join(info_lines),
        )

    # ========================================================
    # Logging
    # ========================================================

    def log_prediction(
        self,
        result: dict,
    ) -> None:

        with LOG_PATH.open(
            "a",
            newline="",
            encoding="utf-8",
        ) as f:

            writer = csv.writer(f)

            writer.writerow(
                [
                    datetime.now().isoformat(),
                    result["mode"],
                    result["source_type"],
                    result["source"],
                    result["classifier"],
                    result["current_state"],
                    result["raw_top_class"],
                    result["confidence"],
                    result["confidence_level"],
                    result["second_class"],
                    result["confidence_gap"],
                    result["feature_dimension"],
                    result["runtime_seconds"],
                    result["device"],
                    json.dumps(
                        result["probabilities"]
                    ),
                ]
            )

    # ========================================================
    # Stop source
    # ========================================================

    def stop_video_source(
        self,
        update_status: bool = True,
    ) -> None:

        self.running_video = False

        if self.capture is not None:

            self.capture.release()
            self.capture = None

        if (
            update_status
            and self.current_source_type
            in {"video", "webcam"}
        ):
            self.status_label.config(
                text=(
                    "Video/webcam source stopped."
                )
            )

    # Backward-compatible name if anything else calls it.
    def stop_video(self) -> None:
        self.stop_video_source()

    # ========================================================
    # Clear result
    # ========================================================

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

    # ========================================================
    # Reset
    # ========================================================

    def reset(self) -> None:

        self.stop_video_source(
            update_status=False
        )

        self.current_frame = None
        self.preview_image = None

        self.current_source_type = "none"
        self.current_source_name = "none"

        self.prediction_busy = False
        self.last_prediction_time = 0.0

        self.prediction_history.clear()

        self.preview_label.config(
            image=""
        )

        self.source_label.config(
            text="Source: none"
        )

        self.image_ready_label.config(
            text="Visual Input: Missing",
            fg="#ffb3b3",
        )

        self.classifier_label.config(
            text="Classifier: —",
            fg="#ffd166",
        )

        self.status_label.config(
            text="System reset."
        )

        self.clear_prediction()


# ============================================================
# Application entry point
# ============================================================

def main() -> None:

    root = tk.Tk()

    app = ImageDemoApp(
        root
    )

    def on_close() -> None:

        app.stop_video_source(
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
