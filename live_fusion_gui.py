# live_fusion_gui.py

from __future__ import annotations

import csv
import json
import joblib
import statistics
import threading
import time
from datetime import datetime
from pathlib import Path
from tkinter import filedialog, messagebox
import tkinter as tk
from typing import Any

import cv2
import librosa
import numpy as np
import pandas as pd
import sounddevice as sd
import torch
import torch.nn.functional as F
from PIL import Image, ImageTk
from sentence_transformers import SentenceTransformer
from transformers import Wav2Vec2FeatureExtractor, WavLMModel, CLIPModel, CLIPProcessor


torch.set_num_threads(2)

FUSION_MODEL_DIR = Path("models/fusion_demo")
FUSION_MODEL_PATH = FUSION_MODEL_DIR / "fusion_pipeline.joblib"
FUSION_FEATURE_COLUMNS_PATH = FUSION_MODEL_DIR / "feature_columns.json"

TEXT_MODEL_PATH = "models/all-mpnet-base-v2"
WAVLM_MODEL_PATH = "models/wavlm-base-plus"
CLIP_MODEL_PATH = "models/clip-vit-large-patch14"

OUTPUT_DIR = Path("data/processed")
LOG_PATH = OUTPUT_DIR / "fusion_live_gui_predictions.csv"

TARGET_SR = 16000
MAX_AUDIO_SECONDS = 20
MIC_RECORD_SECONDS = 10

MIN_KEYDOWNS = 20
MIN_TEXT_CHARS = 20

WEBCAM_PREDICTION_INTERVAL_SEC = 2.0
DISPLAY_SIZE = (320, 240)


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def confidence_level(gap: float) -> str:
    if gap >= 0.35:
        return "High"
    if gap >= 0.15:
        return "Medium"
    return "Low"


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


def safe_mean(values):
    return statistics.mean(values) if values else 0.0


def safe_std(values):
    return statistics.stdev(values) if len(values) >= 2 else 0.0


class FusionDemoApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("SenseFuzeAI Live Multimodal Fusion GUI")
        self.root.geometry("1220x940")
        self.root.configure(bg="#07111f")

        if not FUSION_MODEL_PATH.exists():
            raise FileNotFoundError(f"Fusion model not found: {FUSION_MODEL_PATH}")

        if not FUSION_FEATURE_COLUMNS_PATH.exists():
            raise FileNotFoundError(f"Fusion feature schema not found: {FUSION_FEATURE_COLUMNS_PATH}")

        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        self.initialise_log_file()

        self.pipeline = joblib.load(FUSION_MODEL_PATH)

        with open(FUSION_FEATURE_COLUMNS_PATH, "r", encoding="utf-8") as f:
            self.fusion_feature_columns = json.load(f)

        self.device = get_device()

        self.text_model = SentenceTransformer(TEXT_MODEL_PATH)

        self.wavlm_extractor = Wav2Vec2FeatureExtractor.from_pretrained(WAVLM_MODEL_PATH)
        self.wavlm_model = WavLMModel.from_pretrained(WAVLM_MODEL_PATH).to(self.device)
        self.wavlm_model.eval()

        self.clip_processor = CLIPProcessor.from_pretrained(CLIP_MODEL_PATH)
        self.clip_model = CLIPModel.from_pretrained(CLIP_MODEL_PATH).to(self.device)
        self.clip_model.eval()

        self.keystroke_events: list[dict[str, Any]] = []
        self.active_keys = set()

        self.audio_features_cache = None
        self.audio_source_name = None

        self.image_features_cache = None
        self.image_source_name = None
        self.current_frame = None
        self.preview_image = None

        self.capture = None
        self.running_webcam = False
        self.last_webcam_prediction_time = 0.0
        self.processing = False

        self.build_ui()

    def initialise_log_file(self) -> None:
        if LOG_PATH.exists():
            return

        with LOG_PATH.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "timestamp",
                    "current_state",
                    "confidence",
                    "confidence_level",
                    "second_class",
                    "confidence_gap",
                    "feature_dimension",
                    "keydown_count",
                    "word_count",
                    "audio_source",
                    "image_source",
                    "runtime_seconds",
                    "device",
                ]
            )

    def build_ui(self) -> None:
        tk.Label(
            self.root,
            text="SenseFuzeAI Multimodal Behavioural AI Fusion System",
            font=("Arial", 22, "bold"),
            fg="#74f7ff",
            bg="#07111f",
        ).pack(pady=12)

        tk.Label(
            self.root,
            text="Fuses keystroke dynamics, text semantics, audio representations, and visual embeddings",
            font=("Arial", 12),
            fg="white",
            bg="#07111f",
        ).pack(pady=4)

        readiness_frame = tk.Frame(self.root, bg="#10203a", padx=16, pady=12)
        readiness_frame.pack(fill="x", padx=24, pady=12)

        self.model_ready_label = self.create_status_label(readiness_frame, "Fusion Model: Loaded", "#66ffd6", 0)
        self.text_ready_label = self.create_status_label(readiness_frame, "Text: Missing", "#ffb3b3", 1)
        self.key_ready_label = self.create_status_label(readiness_frame, "Keystroke: Missing", "#ffb3b3", 2)
        self.audio_ready_label = self.create_status_label(readiness_frame, "Audio: Missing", "#ffb3b3", 3)
        self.image_ready_label = self.create_status_label(readiness_frame, "Image: Missing", "#ffb3b3", 4)

        main_frame = tk.Frame(self.root, bg="#07111f")
        main_frame.pack(pady=8)

        left_frame = tk.Frame(main_frame, bg="#07111f")
        left_frame.grid(row=0, column=0, padx=12, sticky="n")

        right_frame = tk.Frame(main_frame, bg="#07111f")
        right_frame.grid(row=0, column=1, padx=12, sticky="n")

        tk.Label(
            left_frame,
            text="Text + Keystroke Input",
            font=("Arial", 12, "bold"),
            fg="#cbd6ff",
            bg="#07111f",
        ).pack(anchor="w")

        self.text_box = tk.Text(
            left_frame,
            height=10,
            width=65,
            font=("Arial", 12),
            wrap="word",
            bg="#f7f9fc",
            fg="#111827",
        )
        self.text_box.pack(pady=6)
        self.text_box.bind("<KeyPress>", self.on_key_press)
        self.text_box.bind("<KeyRelease>", self.on_key_release)
        self.text_box.bind("<KeyRelease>", self.on_key_release, add="+")

        metric_frame = tk.Frame(left_frame, bg="#07111f")
        metric_frame.pack(fill="x", pady=4)

        self.char_count_var = self.create_metric_card(metric_frame, "Characters", "0", 0)
        self.key_count_var = self.create_metric_card(metric_frame, "Keypresses", "0", 1)
        self.word_count_var = self.create_metric_card(metric_frame, "Words", "0", 2)

        audio_frame = tk.LabelFrame(
            left_frame,
            text="Audio Input",
            font=("Arial", 11, "bold"),
            fg="#74f7ff",
            bg="#07111f",
            padx=8,
            pady=8,
        )
        audio_frame.pack(fill="x", pady=8)

        tk.Button(audio_frame, text="Choose Audio File", command=self.choose_audio_file, width=22).grid(row=0, column=0, padx=5)
        tk.Button(audio_frame, text="Record Microphone", command=self.record_microphone_threaded, width=22).grid(row=0, column=1, padx=5)

        self.audio_label = tk.Label(
            audio_frame,
            text="Audio: not loaded",
            fg="#cbd6ff",
            bg="#07111f",
            wraplength=560,
        )
        self.audio_label.grid(row=1, column=0, columnspan=2, pady=6)

        image_frame = tk.LabelFrame(
            right_frame,
            text="Image / Webcam Input",
            font=("Arial", 11, "bold"),
            fg="#74f7ff",
            bg="#07111f",
            padx=8,
            pady=8,
        )
        image_frame.pack(fill="x", pady=0)

        tk.Button(image_frame, text="Choose Image", command=self.choose_image_file, width=18).grid(row=0, column=0, padx=5)
        tk.Button(image_frame, text="Start Webcam", command=self.start_webcam, width=18).grid(row=0, column=1, padx=5)
        tk.Button(image_frame, text="Stop Webcam", command=self.stop_webcam, width=18).grid(row=0, column=2, padx=5)

        self.image_label = tk.Label(
            image_frame,
            text="Image: not loaded",
            fg="#cbd6ff",
            bg="#07111f",
            wraplength=460,
        )
        self.image_label.grid(row=1, column=0, columnspan=3, pady=6)

        self.preview_label = tk.Label(right_frame, bg="#07111f")
        self.preview_label.pack(pady=8)

        control_frame = tk.Frame(self.root, bg="#07111f")
        control_frame.pack(pady=8)

        tk.Button(
            control_frame,
            text="Fusion Behavioural Prediction",
            command=self.predict_fusion_threaded,
            font=("Arial", 13, "bold"),
            bg="#1F618D",
            fg="white",
            width=30,
        ).grid(row=0, column=0, padx=8)

        tk.Button(
            control_frame,
            text="Reset Session",
            command=self.reset,
            font=("Arial", 12, "bold"),
            bg="#4a5568",
            fg="white",
            width=18,
        ).grid(row=0, column=1, padx=8)

        self.status_label = tk.Label(
            self.root,
            text=f"System ready. Device: {self.device}",
            fg="#cbd6ff",
            bg="#07111f",
            font=("Arial", 10),
        )
        self.status_label.pack(pady=4)

        result_frame = tk.Frame(self.root, bg="#10203a", padx=24, pady=22)
        result_frame.pack(fill="x", padx=24, pady=12)

        tk.Label(
            result_frame,
            text="Current Behavioural State",
            font=("Arial", 14, "bold"),
            fg="#cbd6ff",
            bg="#10203a",
        ).pack()

        self.state_label = tk.Label(
            result_frame,
            text="—",
            font=("Arial", 48, "bold"),
            fg="#74f7ff",
            bg="#10203a",
        )
        self.state_label.pack(pady=6)

        self.confidence_label = tk.Label(
            result_frame,
            text="Confidence: —",
            font=("Arial", 20, "bold"),
            fg="white",
            bg="#10203a",
        )
        self.confidence_label.pack(pady=4)

        self.confidence_level_label = tk.Label(
            result_frame,
            text="Prediction Confidence: —",
            font=("Arial", 16, "bold"),
            fg="#cbd6ff",
            bg="#10203a",
        )
        self.confidence_level_label.pack(pady=4)

        technical_frame = tk.LabelFrame(
            self.root,
            text="Technical Details",
            font=("Arial", 12, "bold"),
            fg="#74f7ff",
            bg="#07111f",
            padx=12,
            pady=10,
        )
        technical_frame.pack(fill="both", expand=True, padx=24, pady=8)

        self.prob_text = tk.Text(
            technical_frame,
            height=7,
            width=125,
            font=("Consolas", 10),
            bg="#0b1220",
            fg="#dbeafe",
        )
        self.prob_text.pack(pady=5)

        self.info_text = tk.Text(
            technical_frame,
            height=9,
            width=125,
            font=("Consolas", 10),
            bg="#0b1220",
            fg="#dbeafe",
        )
        self.info_text.pack(pady=5)

    def create_status_label(self, parent, text: str, colour: str, column: int):
        label = tk.Label(
            parent,
            text=text,
            font=("Arial", 11, "bold"),
            fg=colour,
            bg="#10203a",
        )
        label.grid(row=0, column=column, sticky="w", padx=12)
        return label

    def create_metric_card(self, parent, title: str, value: str, column: int) -> tk.StringVar:
        frame = tk.Frame(parent, bg="#10203a", padx=12, pady=8)
        frame.grid(row=0, column=column, padx=5, sticky="nsew")
        parent.grid_columnconfigure(column, weight=1)

        tk.Label(frame, text=title, font=("Arial", 9, "bold"), fg="#cbd6ff", bg="#10203a").pack(anchor="w")

        value_var = tk.StringVar(value=value)

        tk.Label(frame, textvariable=value_var, font=("Arial", 15, "bold"), fg="#74f7ff", bg="#10203a").pack(anchor="w")

        return value_var

    def update_readiness(self) -> None:
        text = self.text_box.get("1.0", tk.END).strip()
        keydowns = sum(1 for e in self.keystroke_events if e.get("type") == "down")

        self.char_count_var.set(str(len(text)))
        self.key_count_var.set(str(keydowns))
        self.word_count_var.set(str(len(text.split())))

        self.text_ready_label.config(
            text=f"Text: {'Ready' if len(text) >= MIN_TEXT_CHARS else 'Missing'}",
            fg="#66ffd6" if len(text) >= MIN_TEXT_CHARS else "#ffb3b3",
        )

        self.key_ready_label.config(
            text=f"Keystroke: {'Ready' if keydowns >= MIN_KEYDOWNS else 'Missing'}",
            fg="#66ffd6" if keydowns >= MIN_KEYDOWNS else "#ffb3b3",
        )

        self.audio_ready_label.config(
            text=f"Audio: {'Ready' if self.audio_features_cache is not None else 'Missing'}",
            fg="#66ffd6" if self.audio_features_cache is not None else "#ffb3b3",
        )

        self.image_ready_label.config(
            text=f"Image: {'Ready' if self.image_features_cache is not None else 'Missing'}",
            fg="#66ffd6" if self.image_features_cache is not None else "#ffb3b3",
        )

    def on_key_press(self, event) -> None:
        key = normalise_key(event)

        if key in self.active_keys:
            return

        self.active_keys.add(key)

        self.keystroke_events.append(
            {
                "type": "down",
                "key": key,
                "timestamp_perf": time.perf_counter(),
                "timestamp_epoch": time.time(),
            }
        )

        self.update_readiness()

    def on_key_release(self, event) -> None:
        key = normalise_key(event)
        self.active_keys.discard(key)

        self.keystroke_events.append(
            {
                "type": "up",
                "key": key,
                "timestamp_perf": time.perf_counter(),
                "timestamp_epoch": time.time(),
            }
        )

        self.update_readiness()

    def extract_keystroke_features(self) -> dict[str, Any]:
        typed_text = self.text_box.get("1.0", tk.END).strip()

        downs = [e for e in self.keystroke_events if e.get("type") == "down"]
        down_times = [e["timestamp_perf"] for e in downs if "timestamp_perf" in e]

        if len(down_times) < 2:
            raise ValueError("Not enough keystroke events.")

        keydown_count = len(downs)

        if keydown_count < MIN_KEYDOWNS:
            raise ValueError(f"Need at least {MIN_KEYDOWNS} key presses.")

        delays = [down_times[i] - down_times[i - 1] for i in range(1, len(down_times))]

        hold_times = []
        unmatched_downs = {}

        for event in self.keystroke_events:
            key = event.get("key")
            event_type = event.get("type")
            timestamp = event.get("timestamp_perf")

            if key is None or timestamp is None:
                continue

            if event_type == "down":
                unmatched_downs.setdefault(key, []).append(timestamp)
            elif event_type == "up":
                if key in unmatched_downs and unmatched_downs[key]:
                    down_time = unmatched_downs[key].pop(0)
                    hold_times.append(timestamp - down_time)

        total_duration = down_times[-1] - down_times[0]
        word_count = len(typed_text.split())

        correction_count = sum(
            1 for event in downs
            if event.get("key") in {"backspace", "delete"}
        )

        pauses_1000 = [d for d in delays if d >= 1.0]
        pauses_2000 = [d for d in delays if d >= 2.0]
        pauses_5000 = [d for d in delays if d >= 5.0]

        delay_mean = safe_mean(delays)
        delay_std = safe_std(delays)
        rhythm_consistency = 1.0 / (1.0 + delay_std) if delay_std > 0 else 1.0

        return {
            "total_duration_sec": round(total_duration, 4),
            "keydown_count": keydown_count,
            "word_count": word_count,
            "typing_speed_kps": round(keydown_count / total_duration, 4) if total_duration > 0 else 0.0,
            "typing_speed_wpm": round((word_count / total_duration) * 60, 4) if total_duration > 0 else 0.0,
            "delay_mean": round(delay_mean, 4),
            "delay_std": round(delay_std, 4),
            "delay_min": round(min(delays), 4) if delays else 0.0,
            "delay_max": round(max(delays), 4) if delays else 0.0,
            "hold_mean": round(safe_mean(hold_times), 4),
            "hold_std": round(safe_std(hold_times), 4),
            "pause_count_1000": len(pauses_1000),
            "pause_count_2000": len(pauses_2000),
            "pause_count_5000": len(pauses_5000),
            "pause_ratio_1000": round(len(pauses_1000) / len(delays), 4) if delays else 0.0,
            "pause_ratio_2000": round(len(pauses_2000) / len(delays), 4) if delays else 0.0,
            "mental_block_ratio_5000": round(len(pauses_5000) / len(delays), 4) if delays else 0.0,
            "correction_count": correction_count,
            "correction_ratio": round(correction_count / keydown_count, 4) if keydown_count else 0.0,
            "rhythm_consistency": round(rhythm_consistency, 4),
            "burstiness_proxy": round(delay_std / delay_mean, 4) if delay_mean > 0 else 0.0,
            "fits_starts_index": round(len(pauses_1000) / len(delays), 4) if delays else 0.0,
        }

    def extract_text_features(self) -> dict[str, float]:
        text = self.text_box.get("1.0", tk.END).strip()

        if len(text) < MIN_TEXT_CHARS:
            raise ValueError(f"Need at least {MIN_TEXT_CHARS} text characters.")

        embedding = self.text_model.encode(
            [text],
            normalize_embeddings=True,
            show_progress_bar=False,
            convert_to_numpy=True,
        )[0]

        return {f"text_mpnet_emb_{i}": float(value) for i, value in enumerate(embedding)}

    def load_audio_waveform(self, audio_path: Path):
        waveform, sr = librosa.load(audio_path, sr=TARGET_SR, mono=True)
        waveform = waveform.astype(np.float32)
        waveform = waveform[: TARGET_SR * MAX_AUDIO_SECONDS]

        if len(waveform) == 0:
            raise ValueError("Empty audio file.")

        return waveform, sr

    def extract_audio_features_from_waveform(self, waveform, sr):
        duration = librosa.get_duration(y=waveform, sr=sr)

        rms = librosa.feature.rms(y=waveform)[0]
        zcr = librosa.feature.zero_crossing_rate(waveform)[0]
        mfcc = librosa.feature.mfcc(y=waveform, sr=sr, n_mfcc=13)

        spectral_centroid = librosa.feature.spectral_centroid(y=waveform, sr=sr)[0]
        spectral_bandwidth = librosa.feature.spectral_bandwidth(y=waveform, sr=sr)[0]
        spectral_rolloff = librosa.feature.spectral_rolloff(y=waveform, sr=sr)[0]

        pitches, magnitudes = librosa.piptrack(y=waveform, sr=sr)
        threshold = np.median(magnitudes[magnitudes > 0]) if np.any(magnitudes > 0) else 0
        pitch_values = pitches[magnitudes > threshold]
        pitch_values = pitch_values[pitch_values > 0]

        features = {
            "audio_duration": float(duration),
            "audio_rms_mean": float(np.mean(rms)),
            "audio_rms_std": float(np.std(rms)),
            "audio_zcr_mean": float(np.mean(zcr)),
            "audio_zcr_std": float(np.std(zcr)),
            "audio_spectral_centroid_mean": float(np.mean(spectral_centroid)),
            "audio_spectral_centroid_std": float(np.std(spectral_centroid)),
            "audio_spectral_bandwidth_mean": float(np.mean(spectral_bandwidth)),
            "audio_spectral_bandwidth_std": float(np.std(spectral_bandwidth)),
            "audio_spectral_rolloff_mean": float(np.mean(spectral_rolloff)),
            "audio_spectral_rolloff_std": float(np.std(spectral_rolloff)),
            "audio_pitch_mean": float(np.mean(pitch_values)) if len(pitch_values) else 0.0,
            "audio_pitch_std": float(np.std(pitch_values)) if len(pitch_values) else 0.0,
            "audio_pitch_min": float(np.min(pitch_values)) if len(pitch_values) else 0.0,
            "audio_pitch_max": float(np.max(pitch_values)) if len(pitch_values) else 0.0,
        }

        for i in range(13):
            features[f"audio_mfcc_{i}_mean"] = float(np.mean(mfcc[i]))
            features[f"audio_mfcc_{i}_std"] = float(np.std(mfcc[i]))

        inputs = self.wavlm_extractor(
            waveform,
            sampling_rate=TARGET_SR,
            return_tensors="pt",
            padding=True,
        )
        inputs = {key: value.to(self.device) for key, value in inputs.items()}

        with torch.inference_mode():
            outputs = self.wavlm_model(**inputs)

        embedding = outputs.last_hidden_state.mean(dim=1).squeeze(0)
        norm = embedding.norm(p=2)

        if norm.item() > 0:
            embedding = embedding / norm

        embedding = embedding.cpu().numpy()

        for i, value in enumerate(embedding):
            features[f"audio_wavlm_emb_{i}"] = float(value)

        return {
            key: float(value) if np.isfinite(value) else 0.0
            for key, value in features.items()
        }

    def choose_audio_file(self) -> None:
        file_path = filedialog.askopenfilename(
            title="Select audio file",
            filetypes=[
                ("Audio files", "*.wav *.mp3 *.m4a *.flac *.ogg *.aac"),
                ("All files", "*.*"),
            ],
        )

        if not file_path:
            return

        try:
            self.status_label.config(text="Extracting audio features...")
            self.root.update_idletasks()

            waveform, sr = self.load_audio_waveform(Path(file_path))
            self.audio_features_cache = self.extract_audio_features_from_waveform(waveform, sr)
            self.audio_source_name = Path(file_path).name

            self.audio_label.config(text=f"Audio loaded: {self.audio_source_name}", fg="#66ffd6")
            self.status_label.config(text="Audio features ready.")
            self.update_readiness()

        except Exception as exc:
            messagebox.showerror("Audio Error", str(exc))
            self.status_label.config(text="Audio feature extraction failed.")

    def record_microphone_threaded(self) -> None:
        threading.Thread(target=self.record_microphone, daemon=True).start()

    def record_microphone(self) -> None:
        try:
            self.root.after(0, lambda: self.status_label.config(text=f"Recording microphone for {MIC_RECORD_SECONDS} seconds..."))

            recording = sd.rec(
                int(MIC_RECORD_SECONDS * TARGET_SR),
                samplerate=TARGET_SR,
                channels=1,
                dtype="float32",
            )
            sd.wait()

            waveform = recording.flatten().astype(np.float32)

            self.root.after(0, lambda: self.status_label.config(text="Extracting microphone audio features..."))

            self.audio_features_cache = self.extract_audio_features_from_waveform(waveform, TARGET_SR)
            self.audio_source_name = "microphone"

            self.root.after(0, lambda: self.audio_label.config(text="Audio loaded: microphone recording", fg="#66ffd6"))
            self.root.after(0, lambda: self.status_label.config(text="Microphone audio features ready."))
            self.root.after(0, self.update_readiness)

        except Exception as exc:
            self.root.after(0, lambda: messagebox.showerror("Microphone Error", str(exc)))
            self.root.after(0, lambda: self.status_label.config(text="Microphone failed."))


    def extract_image_features_from_pil(self, image: Image.Image):
        image = image.convert("RGB")

        inputs = self.clip_processor(images=image, return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(self.device)

        with torch.no_grad():
            try:
                output = self.clip_model.get_image_features(pixel_values=pixel_values)

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
                output = self.clip_model.vision_model(pixel_values=pixel_values)

                if hasattr(output, "pooler_output"):
                    image_features = output.pooler_output
                elif hasattr(output, "last_hidden_state"):
                    image_features = output.last_hidden_state.mean(dim=1)
                else:
                    raise TypeError(f"Unsupported CLIP vision output type: {type(output)}")

        image_features = F.normalize(image_features, p=2, dim=-1)
        embedding = image_features.squeeze(0).cpu().numpy()

        return {
            f"image_clip_emb_{i}": float(value)
            for i, value in enumerate(embedding)
        }


    def choose_image_file(self) -> None:
        self.stop_webcam()

        file_path = filedialog.askopenfilename(
            title="Select image file",
            filetypes=[
                ("Image files", "*.jpg *.jpeg *.png *.webp"),
                ("All files", "*.*"),
            ],
        )

        if not file_path:
            return

        try:
            image = Image.open(file_path).convert("RGB")
            self.current_frame = image
            self.image_features_cache = self.extract_image_features_from_pil(image)
            self.image_source_name = Path(file_path).name

            self.show_preview(image)
            self.image_label.config(text=f"Image loaded: {self.image_source_name}", fg="#66ffd6")
            self.status_label.config(text="Image features ready.")
            self.update_readiness()

        except Exception as exc:
            messagebox.showerror("Image Error", str(exc))

    def start_webcam(self) -> None:
        self.stop_webcam()

        self.capture = cv2.VideoCapture(0)

        if not self.capture.isOpened():
            messagebox.showerror("Webcam Error", "Could not access webcam.")
            return

        self.running_webcam = True
        self.image_source_name = "webcam"
        self.status_label.config(text="Webcam running.")
        self.webcam_loop()

    def webcam_loop(self) -> None:
        if not self.running_webcam or self.capture is None:
            return

        ret, frame = self.capture.read()

        if not ret:
            self.stop_webcam()
            return

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(frame_rgb)

        self.current_frame = image
        self.show_preview(image)

        now = time.time()
        if now - self.last_webcam_prediction_time >= WEBCAM_PREDICTION_INTERVAL_SEC:
            self.last_webcam_prediction_time = now
            threading.Thread(target=self.update_webcam_image_features, daemon=True).start()

        self.root.after(30, self.webcam_loop)

    def update_webcam_image_features(self) -> None:
        if self.processing or self.current_frame is None:
            return

        try:
            self.processing = True
            self.image_features_cache = self.extract_image_features_from_pil(self.current_frame)
            self.root.after(0, lambda: self.image_label.config(text="Image loaded: latest webcam frame", fg="#66ffd6"))
            self.root.after(0, self.update_readiness)
        finally:
            self.processing = False

    def stop_webcam(self) -> None:
        self.running_webcam = False

        if self.capture is not None:
            self.capture.release()
            self.capture = None

    def show_preview(self, image: Image.Image) -> None:
        display = image.copy()
        display.thumbnail(DISPLAY_SIZE)

        self.preview_image = ImageTk.PhotoImage(display)
        self.preview_label.config(image=self.preview_image)

    def build_fusion_vector(self):
        features = {}

        features.update(self.extract_keystroke_features())
        features.update(self.extract_text_features())

        if self.audio_features_cache is None:
            raise ValueError("Audio features are missing. Choose an audio file or record microphone.")

        if self.image_features_cache is None:
            raise ValueError("Image features are missing. Choose image or start webcam.")

        features.update(self.audio_features_cache)
        features.update(self.image_features_cache)

        missing = [col for col in self.fusion_feature_columns if col not in features]

        if missing:
            raise ValueError("Fusion feature mismatch.\n\n" f"Missing columns: {missing[:30]}")

        x = pd.DataFrame(
            [[features[col] for col in self.fusion_feature_columns]],
            columns=self.fusion_feature_columns,
        )

        return x, features

    def predict_fusion_threaded(self) -> None:
        threading.Thread(target=self.predict_fusion, daemon=True).start()

    def predict_fusion(self) -> None:
        try:
            self.root.after(0, lambda: self.status_label.config(text="Building fusion vector..."))
            start = time.perf_counter()

            x, features = self.build_fusion_vector()

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
            runtime = time.perf_counter() - start

            result = {
                "prediction": str(prediction),
                "current_state": str(top_class),
                "confidence": float(top_prob),
                "confidence_percent": float(top_prob * 100),
                "confidence_level": confidence_level(gap),
                "second_class": str(second_class),
                "second_probability": float(second_prob),
                "confidence_gap": gap,
                "probabilities": {str(cls): float(prob) for cls, prob in sorted_probs},
                "feature_dimension": int(x.shape[1]),
                "keydown_count": features.get("keydown_count"),
                "word_count": features.get("word_count"),
                "audio_source": self.audio_source_name,
                "image_source": self.image_source_name,
                "runtime_seconds": runtime,
                "device": str(self.device),
            }

            self.root.after(0, lambda: self.update_prediction_ui(result))
            self.log_prediction(result)

            self.root.after(0, lambda: self.status_label.config(text="Fusion behavioural prediction complete."))

        except Exception as exc:
            self.root.after(0, lambda: self.status_label.config(text="Fusion prediction failed."))
            self.root.after(0, lambda: messagebox.showerror("Fusion Prediction Error", str(exc)))

    def update_prediction_ui(self, result: dict[str, Any]) -> None:
        self.state_label.config(text=result["current_state"].upper())

        self.confidence_label.config(
            text=f"Confidence: {result['confidence_percent']:.2f}%"
        )

        colour = {
            "High": "#66ffd6",
            "Medium": "#ffd166",
            "Low": "#ff6b8a",
        }.get(result["confidence_level"], "#cbd6ff")

        self.confidence_level_label.config(
            text=f"Prediction Confidence: {result['confidence_level']}",
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
            "Fusion diagnostics:",
            "",
            f"Current state        : {result['current_state']}",
            f"Confidence           : {result['confidence_percent']:.2f}%",
            f"Confidence level     : {result['confidence_level']}",
            f"Second-highest class : {result['second_class']}",
            f"Confidence gap       : {result['confidence_gap']:.4f}",
            f"Feature dimension    : {result['feature_dimension']}",
            f"Keystroke keydowns   : {result['keydown_count']}",
            f"Text word count      : {result['word_count']}",
            f"Audio source         : {result['audio_source']}",
            f"Image source         : {result['image_source']}",
            f"Runtime              : {result['runtime_seconds']:.4f} sec",
            f"Device               : {result['device']}",
            f"Logged to            : {LOG_PATH}",
            f"Timestamp            : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        ]

        self.info_text.delete("1.0", tk.END)
        self.info_text.insert(tk.END, "\n".join(info_lines))

    def log_prediction(self, result: dict[str, Any]) -> None:
        with LOG_PATH.open("a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    datetime.utcnow().isoformat(),
                    result["current_state"],
                    result["confidence"],
                    result["confidence_level"],
                    result["second_class"],
                    result["confidence_gap"],
                    result["feature_dimension"],
                    result["keydown_count"],
                    result["word_count"],
                    result["audio_source"],
                    result["image_source"],
                    result["runtime_seconds"],
                    result["device"],
                ]
            )

    def clear_prediction(self) -> None:
        self.state_label.config(text="—")
        self.confidence_label.config(text="Confidence: —")
        self.confidence_level_label.config(
            text="Prediction Confidence: —",
            fg="#cbd6ff",
        )
        self.prob_text.delete("1.0", tk.END)
        self.info_text.delete("1.0", tk.END)

    def reset(self) -> None:
        self.stop_webcam()

        self.keystroke_events = []
        self.active_keys = set()

        self.audio_features_cache = None
        self.audio_source_name = None

        self.image_features_cache = None
        self.image_source_name = None
        self.current_frame = None
        self.preview_image = None

        self.text_box.delete("1.0", tk.END)
        self.audio_label.config(text="Audio: not loaded", fg="#cbd6ff")
        self.image_label.config(text="Image: not loaded", fg="#cbd6ff")
        self.preview_label.config(image="")

        self.status_label.config(text=f"System reset. Device: {self.device}")

        self.clear_prediction()
        self.update_readiness()


def main() -> None:
    root = tk.Tk()
    app = FusionDemoApp(root)

    def on_close() -> None:
        app.stop_webcam()
        root.destroy()

    root.protocol("WM_DELETE_WINDOW", on_close)
    root.mainloop()


if __name__ == "__main__":
    main()
