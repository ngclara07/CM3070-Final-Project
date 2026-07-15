# audio_live_gui.py

from __future__ import annotations

import csv
import json
import threading
import time
import warnings
from datetime import datetime
from pathlib import Path
from tkinter import filedialog, messagebox
import tkinter as tk

import joblib
import librosa
import numpy as np
import pandas as pd
import sounddevice as sd
import soundfile as sf
import torch
from transformers import Wav2Vec2FeatureExtractor, WavLMModel


torch.set_num_threads(2)

MODEL_DIR = Path("models/audio_demo")
MODEL_PATH = MODEL_DIR / "audio_pipeline.joblib"
FEATURE_COLUMNS_PATH = MODEL_DIR / "feature_columns.json"

WAVLM_MODEL_PATH = "models/wavlm-base-plus"

OUTPUT_DIR = Path("data/processed")
LOG_PATH = OUTPUT_DIR / "audio_live_gui_predictions.csv"

TARGET_SR = 16000
MAX_AUDIO_SECONDS = 20

MIC_RECORD_SECONDS = 30
LIVE_MIC_ENABLED = False
LIVE_MIC_INTERVAL_MS = 12000

LABELS = ["focused", "distracted", "fatigued", "overloaded"]


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def confidence_level(gap: float) -> str:
    if gap >= 0.35:
        return "High"
    if gap >= 0.15:
        return "Medium"
    return "Low"


def load_audio(audio_path: Path):
    waveform, sr = librosa.load(audio_path, sr=TARGET_SR, mono=True)
    waveform = waveform.astype(np.float32)

    if len(waveform) == 0:
        raise ValueError(f"Empty audio file: {audio_path}")

    max_samples = TARGET_SR * MAX_AUDIO_SECONDS
    waveform = waveform[:max_samples]

    return waveform, sr


def build_wavlm(device: torch.device):
    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(WAVLM_MODEL_PATH)
    model = WavLMModel.from_pretrained(WAVLM_MODEL_PATH)
    model.to(device)
    model.eval()
    return model, feature_extractor


def extract_wavlm_features(waveform, model, feature_extractor, device):
    inputs = feature_extractor(
        waveform,
        sampling_rate=TARGET_SR,
        return_tensors="pt",
        padding=True,
    )
    inputs = {key: value.to(device) for key, value in inputs.items()}

    with torch.inference_mode():
        outputs = model(**inputs)

    embedding = outputs.last_hidden_state.mean(dim=1).squeeze(0)
    norm = embedding.norm(p=2)

    if norm.item() > 0:
        embedding = embedding / norm

    embedding = embedding.cpu().numpy()

    return {
        f"audio_wavlm_emb_{i}": float(value)
        for i, value in enumerate(embedding)
    }


def extract_librosa_features(waveform, sr):
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

    return features


def clean_feature_values(features):
    cleaned = {}

    for key, value in features.items():
        try:
            value = float(value)
            cleaned[key] = value if np.isfinite(value) else 0.0
        except Exception:
            cleaned[key] = 0.0

    return cleaned


class AudioDemoApp:
    def __init__(self, root: tk.Tk):
        warnings.filterwarnings("ignore")

        self.root = root
        self.root.title("SenseFuzeAI Audio Live GUI")
        self.root.geometry("1080x860")
        self.root.configure(bg="#07111f")

        self.pipeline = None
        self.feature_columns = []
        self.device = get_device()
        self.wavlm_model = None
        self.wavlm_extractor = None

        self.selected_audio_path: Path | None = None
        self.live_mic_enabled = LIVE_MIC_ENABLED
        self.is_processing = False

        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

        self.initialise_log_file()
        self.load_models()
        self.build_ui()

    def initialise_log_file(self) -> None:
        if LOG_PATH.exists():
            return

        with LOG_PATH.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "timestamp",
                    "source",
                    "prediction",
                    "confidence",
                    "confidence_level",
                    "second_class",
                    "confidence_gap",
                    "audio_duration",
                    "feature_dimension",
                    "runtime_seconds",
                    "device",
                ]
            )

    def load_models(self) -> None:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(f"Audio model not found: {MODEL_PATH}")

        if not FEATURE_COLUMNS_PATH.exists():
            raise FileNotFoundError(f"Audio feature schema not found: {FEATURE_COLUMNS_PATH}")

        if not Path(WAVLM_MODEL_PATH).exists():
            raise FileNotFoundError(f"WavLM model not found: {WAVLM_MODEL_PATH}")

        self.pipeline = joblib.load(MODEL_PATH)

        with open(FEATURE_COLUMNS_PATH, "r", encoding="utf-8") as f:
            self.feature_columns = json.load(f)

        self.wavlm_model, self.wavlm_extractor = build_wavlm(self.device)

    def build_ui(self) -> None:
        tk.Label(
            self.root,
            text="SenseFuzeAI Audio Behavioural State Classifier",
            font=("Arial", 22, "bold"),
            fg="#74f7ff",
            bg="#07111f",
        ).pack(pady=14)

        tk.Label(
            self.root,
            text="Audio-only behavioural-state prediction using WavLM and Librosa features",
            font=("Arial", 12),
            fg="white",
            bg="#07111f",
        ).pack(pady=4)

        status_frame = tk.Frame(self.root, bg="#10203a", padx=16, pady=12)
        status_frame.pack(fill="x", padx=24, pady=14)

        self.model_status_label = tk.Label(
            status_frame,
            text="Model Status: Loaded",
            font=("Arial", 12, "bold"),
            fg="#66ffd6",
            bg="#10203a",
        )
        self.model_status_label.grid(row=0, column=0, sticky="w", padx=8)

        self.audio_ready_label = tk.Label(
            status_frame,
            text="Audio Readiness: Missing",
            font=("Arial", 12, "bold"),
            fg="#ffb3b3",
            bg="#10203a",
        )
        self.audio_ready_label.grid(row=0, column=1, sticky="w", padx=32)

        self.device_label = tk.Label(
            status_frame,
            text=f"Device: {self.device}",
            font=("Arial", 12, "bold"),
            fg="#74f7ff",
            bg="#10203a",
        )
        self.device_label.grid(row=0, column=2, sticky="w", padx=8)

        controls = tk.Frame(self.root, bg="#07111f")
        controls.pack(pady=10)

        tk.Button(
            controls,
            text="Choose Audio File",
            command=self.choose_audio_file,
            width=20,
            font=("Arial", 11, "bold"),
        ).grid(row=0, column=0, padx=6, pady=5)

        tk.Button(
            controls,
            text="Play Audio",
            command=self.play_audio_file,
            width=18,
            font=("Arial", 11, "bold"),
        ).grid(row=0, column=1, padx=6, pady=5)

        tk.Button(
            controls,
            text="Predict Uploaded Audio",
            command=self.predict_file_threaded,
            width=24,
            bg="#2E86C1",
            fg="white",
            font=("Arial", 11, "bold"),
        ).grid(row=0, column=2, padx=6, pady=5)

        tk.Button(
            controls,
            text="Record Mic Once",
            command=self.predict_microphone_threaded,
            width=20,
            font=("Arial", 11, "bold"),
        ).grid(row=0, column=3, padx=6, pady=5)

        controls_2 = tk.Frame(self.root, bg="#07111f")
        controls_2.pack(pady=4)

        tk.Button(
            controls_2,
            text="Start Live Mic Prediction",
            command=self.start_live_mic,
            width=25,
            bg="#00a884",
            fg="white",
            font=("Arial", 11, "bold"),
        ).grid(row=0, column=0, padx=6, pady=5)

        tk.Button(
            controls_2,
            text="Stop Live Mic Prediction",
            command=self.stop_live_mic,
            width=25,
            bg="#c0392b",
            fg="white",
            font=("Arial", 11, "bold"),
        ).grid(row=0, column=1, padx=6, pady=5)

        tk.Button(
            controls_2,
            text="Reset Session",
            command=self.reset,
            width=18,
            bg="#4a5568",
            fg="white",
            font=("Arial", 11, "bold"),
        ).grid(row=0, column=2, padx=6, pady=5)

        self.file_label = tk.Label(
            self.root,
            text="No audio file selected.",
            fg="#cbd6ff",
            bg="#07111f",
            wraplength=980,
            font=("Arial", 10),
        )
        self.file_label.pack(pady=8)

        self.status_label = tk.Label(
            self.root,
            text="System ready.",
            fg="#cbd6ff",
            bg="#07111f",
            font=("Arial", 11),
        )
        self.status_label.pack(pady=4)

        result_frame = tk.Frame(self.root, bg="#10203a", padx=24, pady=22)
        result_frame.pack(fill="x", padx=24, pady=14)

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
            font=("Arial", 44, "bold"),
            fg="#74f7ff",
            bg="#10203a",
        )
        self.state_label.pack(pady=8)

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
        technical_frame.pack(fill="both", expand=True, padx=24, pady=10)

        self.prob_text = tk.Text(
            technical_frame,
            height=8,
            width=110,
            font=("Consolas", 10),
            bg="#0b1220",
            fg="#dbeafe",
        )
        self.prob_text.pack(pady=6)

        self.info_text = tk.Text(
            technical_frame,
            height=9,
            width=110,
            font=("Consolas", 10),
            bg="#0b1220",
            fg="#dbeafe",
        )
        self.info_text.pack(pady=6)

    def choose_audio_file(self) -> None:
        file_path = filedialog.askopenfilename(
            title="Select audio file",
            filetypes=[
                ("Audio files", "*.wav *.mp3 *.m4a *.flac *.ogg *.aac"),
                ("All files", "*.*"),
            ],
        )

        if file_path:
            self.selected_audio_path = Path(file_path)
            self.file_label.config(text=str(self.selected_audio_path))
            self.audio_ready_label.config(text="Audio Readiness: Ready", fg="#66ffd6")
            self.status_label.config(text="Audio file selected.")
            self.clear_prediction()

    def play_audio_file(self) -> None:
        try:
            if self.selected_audio_path is None:
                raise ValueError("Please select an audio file first.")

            waveform, sr = sf.read(str(self.selected_audio_path), dtype="float32")

            if waveform.ndim > 1:
                waveform = waveform.mean(axis=1)

            self.status_label.config(text="Playing uploaded audio...")
            sd.stop()
            sd.play(waveform, sr)

        except Exception as exc:
            messagebox.showerror("Playback Error", str(exc))

    def record_microphone(self):
        self.root.after(
            0,
            lambda: self.status_label.config(
                text=f"Recording microphone for {MIC_RECORD_SECONDS} seconds..."
            ),
        )

        recording = sd.rec(
            int(MIC_RECORD_SECONDS * TARGET_SR),
            samplerate=TARGET_SR,
            channels=1,
            dtype="float32",
        )
        sd.wait()

        waveform = recording.flatten().astype(np.float32)

        if len(waveform) == 0:
            raise ValueError("No microphone audio captured.")

        return waveform, TARGET_SR

    def extract_features_from_waveform(self, waveform, sr):
        max_samples = TARGET_SR * MAX_AUDIO_SECONDS
        waveform = waveform[:max_samples]

        features = clean_feature_values(
            {
                **extract_librosa_features(waveform, sr),
                **extract_wavlm_features(
                    waveform=waveform,
                    model=self.wavlm_model,
                    feature_extractor=self.wavlm_extractor,
                    device=self.device,
                ),
            }
        )

        missing = [col for col in self.feature_columns if col not in features]

        if missing:
            raise ValueError(f"Missing audio feature columns: {missing[:20]}")

        x = pd.DataFrame(
            [[features[col] for col in self.feature_columns]],
            columns=self.feature_columns,
        )

        return x, features

    def predict_from_waveform(self, waveform, sr, source_name: str) -> dict:
        start = time.perf_counter()

        x, features = self.extract_features_from_waveform(waveform, sr)

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
        level = confidence_level(gap)
        runtime = time.perf_counter() - start

        return {
            "source": source_name,
            "prediction": str(prediction),
            "current_state": str(top_class),
            "confidence": float(top_prob),
            "confidence_percent": float(top_prob * 100),
            "confidence_level": level,
            "second_class": str(second_class),
            "second_probability": float(second_prob),
            "confidence_gap": gap,
            "probabilities": {str(cls): float(prob) for cls, prob in sorted_probs},
            "feature_dimension": int(x.shape[1]),
            "audio_duration": float(features.get("audio_duration", 0.0)),
            "runtime_seconds": runtime,
            "device": str(self.device),
        }

    def update_prediction_ui(self, result: dict) -> None:
        self.state_label.config(text=result["current_state"].upper())
        self.confidence_label.config(
            text=f"Confidence: {result['confidence_percent']:.2f}%"
        )

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
            "Audio diagnostics:",
            "",
            f"Source               : {result['source']}",
            f"Current state        : {result['current_state']}",
            f"Confidence           : {result['confidence_percent']:.2f}%",
            f"Confidence level     : {result['confidence_level']}",
            f"Second-highest class : {result['second_class']}",
            f"Confidence gap       : {result['confidence_gap']:.4f}",
            f"Feature dimension    : {result['feature_dimension']}",
            f"Audio duration       : {result['audio_duration']:.4f} sec",
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
                    result["source"],
                    result["current_state"],
                    result["confidence"],
                    result["confidence_level"],
                    result["second_class"],
                    result["confidence_gap"],
                    result["audio_duration"],
                    result["feature_dimension"],
                    result["runtime_seconds"],
                    result["device"],
                ]
            )

    def predict_file_threaded(self) -> None:
        threading.Thread(target=self.predict_file, daemon=True).start()

    def predict_file(self) -> None:
        try:
            if self.selected_audio_path is None:
                raise ValueError("Please select an audio file first.")

            self.is_processing = True
            self.root.after(0, lambda: self.status_label.config(text="Extracting uploaded audio features..."))

            waveform, sr = load_audio(self.selected_audio_path)
            result = self.predict_from_waveform(waveform, sr, self.selected_audio_path.name)

            self.root.after(0, lambda: self.update_prediction_ui(result))
            self.log_prediction(result)

            self.root.after(0, lambda: self.status_label.config(text="Uploaded audio prediction complete."))

        except Exception as exc:
            self.root.after(0, lambda: self.status_label.config(text="Prediction failed."))
            self.root.after(0, lambda: messagebox.showerror("Prediction Error", str(exc)))

        finally:
            self.is_processing = False

    def predict_microphone_threaded(self) -> None:
        threading.Thread(target=self.predict_microphone, daemon=True).start()

    def predict_microphone(self) -> None:
        try:
            self.is_processing = True

            waveform, sr = self.record_microphone()

            self.root.after(0, lambda: self.audio_ready_label.config(text="Audio Readiness: Ready", fg="#66ffd6"))
            self.root.after(0, lambda: self.status_label.config(text="Extracting microphone audio features..."))

            result = self.predict_from_waveform(waveform, sr, "microphone recording")

            self.root.after(0, lambda: self.update_prediction_ui(result))
            self.log_prediction(result)

            self.root.after(0, lambda: self.status_label.config(text="Microphone prediction complete."))

        except Exception as exc:
            self.root.after(0, lambda: self.status_label.config(text="Microphone prediction failed."))
            self.root.after(0, lambda: messagebox.showerror("Microphone Error", str(exc)))

        finally:
            self.is_processing = False

    def start_live_mic(self) -> None:
        self.live_mic_enabled = True
        self.status_label.config(text="Live microphone prediction enabled.")
        self.root.after(500, self.live_mic_loop)

    def stop_live_mic(self) -> None:
        self.live_mic_enabled = False
        self.status_label.config(text="Live microphone prediction stopped.")

    def live_mic_loop(self) -> None:
        if not self.live_mic_enabled:
            return

        if not self.is_processing:
            self.predict_microphone_threaded()

        self.root.after(LIVE_MIC_INTERVAL_MS, self.live_mic_loop)

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
        self.live_mic_enabled = False
        self.selected_audio_path = None
        sd.stop()

        self.file_label.config(text="No audio file selected.")
        self.audio_ready_label.config(text="Audio Readiness: Missing", fg="#ffb3b3")
        self.status_label.config(text="System reset.")
        self.clear_prediction()


def main() -> None:
    root = tk.Tk()
    AudioDemoApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
