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
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image, ImageTk
from transformers import CLIPModel, CLIPProcessor


MODEL_DIR = Path("models/image_demo")
MODEL_PATH = MODEL_DIR / "image_pipeline.joblib"
FEATURE_COLUMNS_PATH = MODEL_DIR / "feature_columns.json"

CLIP_MODEL_PATH = "models/clip-vit-large-patch14"

OUTPUT_DIR = Path("data/processed")
LOG_PATH = OUTPUT_DIR / "image_live_gui_predictions.csv"

LIVE_PREDICTION_INTERVAL_SEC = 1.0
PREDICTION_SMOOTHING_WINDOW = 1
UNCERTAINTY_GAP_THRESHOLD = 0.10
DISPLAY_SIZE = (360, 220)

LABELS = ["focused", "distracted", "fatigued", "overloaded"]


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def confidence_level(gap: float) -> str:
    if gap >= 0.35:
        return "High"
    if gap >= 0.15:
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
):
    image = image.convert("RGB")

    inputs = processor(images=image, return_tensors="pt")
    pixel_values = inputs["pixel_values"].to(device)

    with torch.no_grad():
        try:
            output = model.get_image_features(pixel_values=pixel_values)

            if isinstance(output, torch.Tensor):
                image_features = output
            elif hasattr(output, "image_embeds"):
                image_features = output.image_embeds
            elif hasattr(output, "pooler_output"):
                image_features = output.pooler_output
            elif hasattr(output, "last_hidden_state"):
                image_features = output.last_hidden_state.mean(dim=1)
            else:
                raise TypeError(f"Unsupported CLIP output type: {type(output)}")

        except Exception:
            output = model.vision_model(pixel_values=pixel_values)

            if hasattr(output, "pooler_output"):
                image_features = output.pooler_output
            elif hasattr(output, "last_hidden_state"):
                image_features = output.last_hidden_state.mean(dim=1)
            else:
                raise TypeError(f"Unsupported CLIP vision output type: {type(output)}")

    image_features = F.normalize(image_features, p=2, dim=-1)
    return image_features.squeeze(0).cpu().numpy()


class ImageDemoApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("SenseFuzeAI Image Live GUI")
        self.root.geometry("1080x760")
        self.root.minsize(900, 720)
        self.root.configure(bg="#07111f")

        if not MODEL_PATH.exists():
            raise FileNotFoundError(f"Image model not found: {MODEL_PATH}")

        if not FEATURE_COLUMNS_PATH.exists():
            raise FileNotFoundError(f"Feature columns not found: {FEATURE_COLUMNS_PATH}")

        if not Path(CLIP_MODEL_PATH).exists():
            raise FileNotFoundError(f"CLIP model not found: {CLIP_MODEL_PATH}")

        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        self.initialise_log_file()

        self.pipeline = joblib.load(MODEL_PATH)

        with open(FEATURE_COLUMNS_PATH, "r", encoding="utf-8") as f:
            self.feature_columns = json.load(f)

        self.device = get_device()
        self.clip_model, self.clip_processor = build_clip_model(self.device)

        self.capture = None
        self.running_video = False
        self.current_frame: Image.Image | None = None
        self.preview_image = None
        self.last_prediction_time = 0.0
        self.prediction_busy = False
        self.prediction_history: list[str] = []

        self.build_ui()

    def initialise_log_file(self) -> None:
        if LOG_PATH.exists():
            return

        with LOG_PATH.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "timestamp",
                    "mode",
                    "source",
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

    def build_ui(self) -> None:
        tk.Label(
            self.root,
            text="SenseFuzeAI Image / Video Behavioural State Classifier",
            font=("Arial", 20, "bold"),
            fg="#74f7ff",
            bg="#07111f",
        ).pack(pady=(10, 4))

        tk.Label(
            self.root,
            text="Image-only behavioural-state prediction using CLIP visual embeddings",
            font=("Arial", 11),
            fg="white",
            bg="#07111f",
        ).pack(pady=(0, 6))

        status_frame = tk.Frame(self.root, bg="#10203a", padx=14, pady=10)
        status_frame.pack(fill="x", padx=18, pady=6)

        self.model_status_label = tk.Label(
            status_frame,
            text="Model Status: Loaded",
            font=("Arial", 11, "bold"),
            fg="#66ffd6",
            bg="#10203a",
        )
        self.model_status_label.grid(row=0, column=0, sticky="w", padx=8)

        self.image_ready_label = tk.Label(
            status_frame,
            text="Image Readiness: Missing",
            font=("Arial", 11, "bold"),
            fg="#ffb3b3",
            bg="#10203a",
        )
        self.image_ready_label.grid(row=0, column=1, sticky="w", padx=24)

        self.device_label = tk.Label(
            status_frame,
            text=f"Device: {self.device}",
            font=("Arial", 11, "bold"),
            fg="#74f7ff",
            bg="#10203a",
        )
        self.device_label.grid(row=0, column=2, sticky="w", padx=8)

        button_frame = tk.Frame(self.root, bg="#07111f")
        button_frame.pack(pady=6)

        tk.Button(button_frame, text="Choose Image", command=self.choose_image, width=16, font=("Arial", 10, "bold")).grid(row=0, column=0, padx=5, pady=4)
        tk.Button(button_frame, text="Choose Video", command=self.choose_video, width=16, font=("Arial", 10, "bold")).grid(row=0, column=1, padx=5, pady=4)
        tk.Button(button_frame, text="Start Webcam", command=self.start_webcam, width=16, bg="#00a884", fg="white", font=("Arial", 10, "bold")).grid(row=0, column=2, padx=5, pady=4)
        tk.Button(button_frame, text="Stop", command=self.stop_video, width=10, bg="#c0392b", fg="white", font=("Arial", 10, "bold")).grid(row=0, column=3, padx=5, pady=4)
        tk.Button(button_frame, text="Manual Prediction", command=self.predict_current_frame_threaded, width=20, bg="#2E86C1", fg="white", font=("Arial", 10, "bold")).grid(row=0, column=4, padx=5, pady=4)
        tk.Button(button_frame, text="Reset", command=self.reset, width=12, bg="#4a5568", fg="white", font=("Arial", 10, "bold")).grid(row=0, column=5, padx=5, pady=4)

        self.status_label = tk.Label(
            self.root,
            text="System ready.",
            fg="#cbd6ff",
            bg="#07111f",
            font=("Arial", 10),
        )
        self.status_label.pack(pady=4)

        result_frame = tk.Frame(self.root, bg="#10203a", padx=18, pady=14)
        result_frame.pack(fill="x", padx=18, pady=8)

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
        self.state_label.pack(pady=(4, 2))

        self.confidence_label = tk.Label(
            result_frame,
            text="Confidence: —",
            font=("Arial", 18, "bold"),
            fg="white",
            bg="#10203a",
        )
        self.confidence_label.pack(pady=2)

        self.confidence_level_label = tk.Label(
            result_frame,
            text="Prediction Confidence: —",
            font=("Arial", 14, "bold"),
            fg="#cbd6ff",
            bg="#10203a",
        )
        self.confidence_level_label.pack(pady=2)

        lower_frame = tk.Frame(self.root, bg="#07111f")
        lower_frame.pack(fill="both", expand=True, padx=18, pady=6)

        preview_frame = tk.Frame(lower_frame, bg="#07111f")
        preview_frame.pack(side="left", fill="both", expand=False, padx=(0, 12))

        self.preview_label = tk.Label(preview_frame, bg="#07111f")
        self.preview_label.pack(pady=6)

        technical_frame = tk.LabelFrame(
            lower_frame,
            text="Technical Details",
            font=("Arial", 11, "bold"),
            fg="#74f7ff",
            bg="#07111f",
            padx=10,
            pady=8,
        )
        technical_frame.pack(side="left", fill="both", expand=True)

        self.prob_text = tk.Text(
            technical_frame,
            height=7,
            width=80,
            font=("Consolas", 9),
            bg="#0b1220",
            fg="#dbeafe",
        )
        self.prob_text.pack(fill="both", expand=True, pady=4)

        self.info_text = tk.Text(
            technical_frame,
            height=7,
            width=80,
            font=("Consolas", 9),
            bg="#0b1220",
            fg="#dbeafe",
        )
        self.info_text.pack(fill="both", expand=True, pady=4)

    def choose_image(self) -> None:
        self.stop_video()
        self.prediction_history = []

        file_path = filedialog.askopenfilename(
            title="Select image file",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.webp"), ("All files", "*.*")],
        )

        if not file_path:
            return

        image = Image.open(file_path).convert("RGB")
        self.current_frame = image
        self.show_pil_image(image)

        self.image_ready_label.config(text="Image Readiness: Ready", fg="#66ffd6")
        self.status_label.config(text=f"Loaded image: {Path(file_path).name}")
        self.clear_prediction()

    def choose_video(self) -> None:
        self.stop_video()
        self.prediction_history = []

        file_path = filedialog.askopenfilename(
            title="Select video file",
            filetypes=[("Video files", "*.mp4 *.avi *.mov *.mkv *.webm"), ("All files", "*.*")],
        )

        if not file_path:
            return

        self.capture = cv2.VideoCapture(file_path)

        if not self.capture.isOpened():
            messagebox.showerror("Video Error", "Could not open selected video.")
            return

        self.running_video = True
        self.image_ready_label.config(text="Image Readiness: Ready", fg="#66ffd6")
        self.status_label.config(text=f"Playing video: {Path(file_path).name}")
        self.video_loop(source_name=Path(file_path).name)

    def start_webcam(self) -> None:
        self.stop_video()
        self.prediction_history = []

        self.capture = cv2.VideoCapture(0)

        if not self.capture.isOpened():
            messagebox.showerror("Webcam Error", "Could not access webcam.")
            return

        self.running_video = True
        self.image_ready_label.config(text="Image Readiness: Ready", fg="#66ffd6")
        self.status_label.config(text="Webcam live prediction enabled.")
        self.video_loop(source_name="webcam")

    def video_loop(self, source_name: str = "video") -> None:
        if not self.running_video or self.capture is None:
            return

        ret, frame = self.capture.read()

        if not ret:
            self.status_label.config(text="Video ended.")
            self.stop_video()
            return

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(frame_rgb)
        self.current_frame = pil_image

        self.show_pil_image(pil_image)

        now = time.time()
        if now - self.last_prediction_time >= LIVE_PREDICTION_INTERVAL_SEC:
            self.last_prediction_time = now
            self.predict_current_frame_threaded(mode="Live", source_name=source_name)

        self.root.after(30, lambda: self.video_loop(source_name=source_name))

    def show_pil_image(self, image: Image.Image) -> None:
        display = image.copy()
        display.thumbnail(DISPLAY_SIZE)

        self.preview_image = ImageTk.PhotoImage(display)
        self.preview_label.config(image=self.preview_image)

    def frame_to_features(self, image: Image.Image) -> pd.DataFrame:
        embedding = extract_image_embedding_from_pil(
            image=image,
            model=self.clip_model,
            processor=self.clip_processor,
            device=self.device,
        )

        features = {
            f"image_clip_emb_{i}": float(value)
            for i, value in enumerate(embedding)
        }

        missing = [col for col in self.feature_columns if col not in features]

        if missing:
            raise ValueError(f"Missing image feature columns: {missing[:20]}")

        return pd.DataFrame(
            [[features[col] for col in self.feature_columns]],
            columns=self.feature_columns,
        )

    def predict_current_frame_threaded(
        self,
        mode: str = "Manual",
        source_name: str = "current frame",
    ) -> None:
        if self.prediction_busy:
            return

        threading.Thread(
            target=self.predict_current_frame,
            args=(mode, source_name),
            daemon=True,
        ).start()

    def predict_current_frame(
        self,
        mode: str = "Manual",
        source_name: str = "current frame",
    ) -> None:
        try:
            if self.current_frame is None:
                raise ValueError("No image/frame available for prediction.")

            self.prediction_busy = True
            self.root.after(0, lambda: self.status_label.config(text="Extracting CLIP visual features..."))

            start = time.perf_counter()
            x = self.frame_to_features(self.current_frame)

            prediction = self.pipeline.predict(x)[0]
            probabilities = self.pipeline.predict_proba(x)[0]
            classes = self.pipeline.classes_

            sorted_probs = sorted(
                zip(classes, probabilities),
                key=lambda item: item[1],
                reverse=True,
            )

            top_class, top_prob = sorted_probs[0]
            second_class, second_prob = sorted_probs[1]
            gap = float(top_prob - second_prob)

            if gap < UNCERTAINTY_GAP_THRESHOLD:
                current_state = "uncertain"
            elif mode == "Live":
                self.prediction_history.append(str(top_class))

                if len(self.prediction_history) > PREDICTION_SMOOTHING_WINDOW:
                    self.prediction_history.pop(0)

                current_state = Counter(self.prediction_history).most_common(1)[0][0]
            else:
                current_state = str(top_class)

            runtime = time.perf_counter() - start

            result = {
                "mode": mode,
                "source": source_name,
                "prediction": str(prediction),
                "current_state": current_state,
                "raw_top_class": str(top_class),
                "confidence": float(top_prob),
                "confidence_percent": float(top_prob * 100),
                "confidence_level": confidence_level(gap),
                "second_class": str(second_class),
                "second_probability": float(second_prob),
                "confidence_gap": gap,
                "probabilities": {str(cls): float(prob) for cls, prob in sorted_probs},
                "feature_dimension": int(x.shape[1]),
                "runtime_seconds": runtime,
                "device": str(self.device),
                "smoothing_window": PREDICTION_SMOOTHING_WINDOW if mode == "Live" else "N/A",
                "uncertainty_threshold": UNCERTAINTY_GAP_THRESHOLD,
                "recent_live_history": list(self.prediction_history) if mode == "Live" else "N/A",
            }

            self.root.after(0, lambda: self.update_prediction_ui(result))
            self.log_prediction(result)

            self.root.after(0, lambda: self.status_label.config(text=f"{mode} prediction complete."))

        except Exception as exc:
            self.root.after(0, lambda: messagebox.showerror("Prediction Error", str(exc)))
            self.root.after(0, lambda: self.status_label.config(text="Prediction failed."))

        finally:
            self.prediction_busy = False

    def update_prediction_ui(self, result: dict) -> None:
        self.state_label.config(text=result["current_state"].upper())

        if result["current_state"] == "uncertain":
            self.state_label.config(fg="#ffd166")
        else:
            self.state_label.config(fg="#74f7ff")

        self.confidence_label.config(text=f"Confidence: {result['confidence_percent']:.2f}%")

        level = result["confidence_level"]
        colour = {
            "High": "#66ffd6",
            "Medium": "#ffd166",
            "Low": "#ff6b8a",
        }.get(level, "#cbd6ff")

        self.confidence_level_label.config(
            text=f"Prediction Confidence: {level}",
            fg=colour,
        )

        prob_lines = ["Probability distribution:", ""]

        for label, probability in result["probabilities"].items():
            bar_length = int(probability * 32)
            bar = "█" * bar_length
            prob_lines.append(f"{label:12s}: {probability * 100:6.2f}%  {bar}")

        self.prob_text.delete("1.0", tk.END)
        self.prob_text.insert(tk.END, "\n".join(prob_lines))

        info_lines = [
            "Image/video diagnostics:",
            "",
            f"Mode                 : {result['mode']}",
            f"Source               : {result['source']}",
            f"Displayed state      : {result['current_state']}",
            f"Raw top class        : {result['raw_top_class']}",
            f"Confidence           : {result['confidence_percent']:.2f}%",
            f"Confidence level     : {result['confidence_level']}",
            f"Second-highest class : {result['second_class']}",
            f"Confidence gap       : {result['confidence_gap']:.4f}",
            f"Uncertainty threshold: {result['uncertainty_threshold']:.2f}",
            f"Feature dimension    : {result['feature_dimension']}",
            f"Smoothing window     : {result['smoothing_window']}",
            f"Recent live history  : {result['recent_live_history']}",
            f"Runtime              : {result['runtime_seconds']:.4f} sec",
            f"Device               : {result['device']}",
            f"Logged to            : {LOG_PATH}",
            f"Timestamp            : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        ]

        self.info_text.delete("1.0", tk.END)
        self.info_text.insert(tk.END, "\n".join(info_lines))

    def log_prediction(self, result: dict) -> None:
        with LOG_PATH.open("a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    datetime.utcnow().isoformat(),
                    result["mode"],
                    result["source"],
                    result["current_state"],
                    result["raw_top_class"],
                    result["confidence"],
                    result["confidence_level"],
                    result["second_class"],
                    result["confidence_gap"],
                    result["feature_dimension"],
                    result["runtime_seconds"],
                    result["device"],
                    json.dumps(result["probabilities"]),
                ]
            )

    def stop_video(self) -> None:
        self.running_video = False

        if self.capture is not None:
            self.capture.release()
            self.capture = None

        self.status_label.config(text="Video/webcam stopped.")

    def clear_prediction(self) -> None:
        self.state_label.config(text="—", fg="#74f7ff")
        self.confidence_label.config(text="Confidence: —")
        self.confidence_level_label.config(
            text="Prediction Confidence: —",
            fg="#cbd6ff",
        )
        self.prob_text.delete("1.0", tk.END)
        self.info_text.delete("1.0", tk.END)

    def reset(self) -> None:
        self.stop_video()

        self.current_frame = None
        self.preview_image = None
        self.prediction_busy = False
        self.last_prediction_time = 0.0
        self.prediction_history = []

        self.preview_label.config(image="")
        self.image_ready_label.config(text="Image Readiness: Missing", fg="#ffb3b3")
        self.status_label.config(text="System reset.")

        self.clear_prediction()


def main() -> None:
    root = tk.Tk()
    app = ImageDemoApp(root)

    def on_close() -> None:
        app.stop_video()
        root.destroy()

    root.protocol("WM_DELETE_WINDOW", on_close)
    root.mainloop()


if __name__ == "__main__":
    main()
