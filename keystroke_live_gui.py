# keystroke_live_gui.py

from __future__ import annotations

import csv
import json
import joblib
import statistics
import time
import tkinter as tk
from datetime import datetime
from pathlib import Path
from tkinter import messagebox
from typing import Any

import pandas as pd


MODEL_DIR = Path("models/keystroke_demo")
MODEL_PATH = MODEL_DIR / "keystroke_pipeline.joblib"
FEATURE_COLUMNS_PATH = MODEL_DIR / "feature_columns.json"

OUTPUT_DIR = Path("data/processed")
LOG_PATH = OUTPUT_DIR / "keystroke_live_gui_predictions.csv"

MIN_RECOMMENDED_KEYDOWNS = 20
LIVE_PREDICTION_ENABLED = True
LIVE_PREDICTION_INTERVAL_MS = 1500

LABELS = ["focused", "distracted", "fatigued", "overloaded"]


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


def confidence_level(gap: float) -> str:
    if gap >= 0.35:
        return "High"
    if gap >= 0.15:
        return "Medium"
    return "Low"


class KeystrokeDemoApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("SenseFuzeAI Keystroke Live GUI")
        self.root.geometry("1080x860")
        self.root.configure(bg="#07111f")

        if not MODEL_PATH.exists():
            raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

        if not FEATURE_COLUMNS_PATH.exists():
            raise FileNotFoundError(f"Feature columns not found: {FEATURE_COLUMNS_PATH}")

        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        self.initialise_log_file()

        self.pipeline = joblib.load(MODEL_PATH)

        with open(FEATURE_COLUMNS_PATH, "r", encoding="utf-8") as f:
            self.feature_columns = json.load(f)

        self.events: list[dict[str, Any]] = []
        self.active_keys = set()

        self.build_ui()

        if LIVE_PREDICTION_ENABLED:
            self.root.after(LIVE_PREDICTION_INTERVAL_MS, self.live_predict)

    def initialise_log_file(self) -> None:
        if LOG_PATH.exists():
            return

        with LOG_PATH.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "timestamp",
                    "mode",
                    "current_state",
                    "confidence",
                    "confidence_level",
                    "second_class",
                    "confidence_gap",
                    "keydown_count",
                    "word_count",
                    "typing_speed_wpm",
                    "typing_speed_kps",
                    "feature_dimension",
                ]
            )

    def build_ui(self) -> None:
        tk.Label(
            self.root,
            text="SenseFuzeAI Keystroke Behavioural State Classifier",
            font=("Arial", 22, "bold"),
            fg="#74f7ff",
            bg="#07111f",
        ).pack(pady=14)

        tk.Label(
            self.root,
            text="Keystroke-only behavioural-state prediction using timing and rhythm features",
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

        self.key_ready_label = tk.Label(
            status_frame,
            text=f"Keystroke Readiness: Missing 0/{MIN_RECOMMENDED_KEYDOWNS}",
            font=("Arial", 12, "bold"),
            fg="#ffb3b3",
            bg="#10203a",
        )
        self.key_ready_label.grid(row=0, column=1, sticky="w", padx=32)

        self.live_status_label = tk.Label(
            status_frame,
            text="Live Prediction: Enabled",
            font=("Arial", 12, "bold"),
            fg="#74f7ff",
            bg="#10203a",
        )
        self.live_status_label.grid(row=0, column=2, sticky="w", padx=8)

        self.text_box = tk.Text(
            self.root,
            height=10,
            width=105,
            font=("Arial", 13),
            wrap="word",
            bg="#f7f9fc",
            fg="#111827",
            insertbackground="#111827",
        )
        self.text_box.pack(padx=24, pady=10)

        self.text_box.bind("<KeyPress>", self.on_key_press)
        self.text_box.bind("<KeyRelease>", self.on_key_release)

        metric_frame = tk.Frame(self.root, bg="#07111f")
        metric_frame.pack(fill="x", padx=24, pady=6)

        self.key_count_var = self.create_metric_card(metric_frame, "Keypresses", "0", 0)
        self.word_count_var = self.create_metric_card(metric_frame, "Words", "0", 1)
        self.duration_var = self.create_metric_card(metric_frame, "Duration", "0.00s", 2)
        self.speed_var = self.create_metric_card(metric_frame, "Typing Speed", "0.00 WPM", 3)

        button_frame = tk.Frame(self.root, bg="#07111f")
        button_frame.pack(pady=10)

        tk.Button(
            button_frame,
            text="Final Manual Prediction",
            command=self.predict,
            font=("Arial", 12, "bold"),
            bg="#2E86C1",
            fg="white",
            width=28,
        ).grid(row=0, column=0, padx=10)

        tk.Button(
            button_frame,
            text="Reset Session",
            command=self.reset,
            font=("Arial", 12, "bold"),
            bg="#4a5568",
            fg="white",
            width=18,
        ).grid(row=0, column=1, padx=10)

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
            height=7,
            width=110,
            font=("Consolas", 10),
            bg="#0b1220",
            fg="#dbeafe",
        )
        self.prob_text.pack(pady=6)

        self.feature_text = tk.Text(
            technical_frame,
            height=10,
            width=110,
            font=("Consolas", 10),
            bg="#0b1220",
            fg="#dbeafe",
        )
        self.feature_text.pack(pady=6)

    def create_metric_card(
        self,
        parent: tk.Frame,
        title: str,
        value: str,
        column: int,
    ) -> tk.StringVar:
        frame = tk.Frame(parent, bg="#10203a", padx=16, pady=10)
        frame.grid(row=0, column=column, padx=8, sticky="nsew")
        parent.grid_columnconfigure(column, weight=1)

        tk.Label(
            frame,
            text=title,
            font=("Arial", 10, "bold"),
            fg="#cbd6ff",
            bg="#10203a",
        ).pack(anchor="w")

        value_var = tk.StringVar(value=value)

        tk.Label(
            frame,
            textvariable=value_var,
            font=("Arial", 18, "bold"),
            fg="#74f7ff",
            bg="#10203a",
        ).pack(anchor="w", pady=4)

        return value_var

    def on_key_press(self, event) -> None:
        key = normalise_key(event)

        if key in self.active_keys:
            return

        self.active_keys.add(key)

        self.events.append(
            {
                "type": "down",
                "key": key,
                "timestamp_perf": time.perf_counter(),
                "timestamp_epoch": time.time(),
            }
        )

        self.update_readiness_metrics()

    def on_key_release(self, event) -> None:
        key = normalise_key(event)
        self.active_keys.discard(key)

        self.events.append(
            {
                "type": "up",
                "key": key,
                "timestamp_perf": time.perf_counter(),
                "timestamp_epoch": time.time(),
            }
        )

        self.update_readiness_metrics()

    def count_keydowns(self) -> int:
        return sum(1 for event in self.events if event.get("type") == "down")

    def get_typed_text(self) -> str:
        return self.text_box.get("1.0", tk.END).strip()

    def update_readiness_metrics(self) -> None:
        keydown_count = self.count_keydowns()
        typed_text = self.get_typed_text()
        word_count = len(typed_text.split())

        downs = [event for event in self.events if event.get("type") == "down"]
        down_times = [event["timestamp_perf"] for event in downs if "timestamp_perf" in event]

        duration = down_times[-1] - down_times[0] if len(down_times) >= 2 else 0.0
        wpm = (word_count / duration) * 60 if duration > 0 else 0.0

        self.key_count_var.set(str(keydown_count))
        self.word_count_var.set(str(word_count))
        self.duration_var.set(f"{duration:.2f}s")
        self.speed_var.set(f"{wpm:.2f} WPM")

        if keydown_count >= MIN_RECOMMENDED_KEYDOWNS:
            self.key_ready_label.config(
                text=f"Keystroke Readiness: Ready {keydown_count}/{MIN_RECOMMENDED_KEYDOWNS}",
                fg="#66ffd6",
            )
        else:
            self.key_ready_label.config(
                text=f"Keystroke Readiness: Missing {keydown_count}/{MIN_RECOMMENDED_KEYDOWNS}",
                fg="#ffb3b3",
            )

    def extract_features(self):
        typed_text = self.get_typed_text()

        downs = [event for event in self.events if event.get("type") == "down"]
        down_times = [event["timestamp_perf"] for event in downs if "timestamp_perf" in event]

        if len(down_times) < 2:
            raise ValueError("Not enough keystroke events. Type a longer sample.")

        delays = [
            down_times[index] - down_times[index - 1]
            for index in range(1, len(down_times))
        ]

        hold_times = []
        unmatched_downs = {}

        for event in self.events:
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

        total_duration = down_times[-1] - down_times[0] if len(down_times) >= 2 else 0.0
        keydown_count = len(downs)
        word_count = len(typed_text.split())

        if keydown_count < MIN_RECOMMENDED_KEYDOWNS:
            raise ValueError(
                f"Only {keydown_count} key presses captured. "
                f"Please type at least {MIN_RECOMMENDED_KEYDOWNS} key presses."
            )

        correction_count = sum(
            1 for event in downs
            if event.get("key") in {"backspace", "delete"}
        )

        pauses_1000 = [delay for delay in delays if delay >= 1.0]
        pauses_2000 = [delay for delay in delays if delay >= 2.0]
        pauses_5000 = [delay for delay in delays if delay >= 5.0]

        delay_mean = safe_mean(delays)
        delay_std = safe_std(delays)

        rhythm_consistency = 1.0 / (1.0 + delay_std) if delay_std > 0 else 1.0
        typing_speed_kps = keydown_count / total_duration if total_duration > 0 else 0.0
        typing_speed_wpm = (word_count / total_duration) * 60 if total_duration > 0 else 0.0
        burstiness_proxy = delay_std / delay_mean if delay_mean > 0 else 0.0
        fits_starts_index = len(pauses_1000) / len(delays) if delays else 0.0

        raw_features = {
            "total_duration_sec": round(total_duration, 4),
            "keydown_count": keydown_count,
            "word_count": word_count,
            "typing_speed_kps": round(typing_speed_kps, 4),
            "typing_speed_wpm": round(typing_speed_wpm, 4),
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
            "burstiness_proxy": round(burstiness_proxy, 4),
            "fits_starts_index": round(fits_starts_index, 4),
        }

        missing = [col for col in self.feature_columns if col not in raw_features]

        if missing:
            raise ValueError(f"Missing live features: {missing}")

        x = pd.DataFrame(
            [[raw_features[col] for col in self.feature_columns]],
            columns=self.feature_columns,
        )

        return x, raw_features

    def predict_from_features(self, mode: str) -> dict[str, Any]:
        x, raw_features = self.extract_features()

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

        return {
            "mode": mode,
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
            "raw_features": raw_features,
        }

    def update_prediction_ui(self, result: dict[str, Any]) -> None:
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

        features = result["raw_features"]

        feature_lines = [
            "Keystroke diagnostics:",
            "",
            f"Mode                 : {result['mode']}",
            f"Current state        : {result['current_state']}",
            f"Confidence           : {result['confidence_percent']:.2f}%",
            f"Confidence level     : {result['confidence_level']}",
            f"Second-highest class : {result['second_class']}",
            f"Confidence gap       : {result['confidence_gap']:.4f}",
            f"Feature dimension    : {result['feature_dimension']}",
            f"Logged to            : {LOG_PATH}",
            f"Timestamp            : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "Extracted keystroke features:",
            "",
        ]

        for feature in self.feature_columns:
            feature_lines.append(f"{feature:25s}: {features[feature]}")

        self.feature_text.delete("1.0", tk.END)
        self.feature_text.insert(tk.END, "\n".join(feature_lines))

    def log_prediction(self, result: dict[str, Any]) -> None:
        features = result["raw_features"]

        with LOG_PATH.open("a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    datetime.utcnow().isoformat(),
                    result["mode"],
                    result["current_state"],
                    result["confidence"],
                    result["confidence_level"],
                    result["second_class"],
                    result["confidence_gap"],
                    features.get("keydown_count", 0),
                    features.get("word_count", 0),
                    features.get("typing_speed_wpm", 0.0),
                    features.get("typing_speed_kps", 0.0),
                    result["feature_dimension"],
                ]
            )

    def live_predict(self) -> None:
        try:
            keydown_count = self.count_keydowns()

            if keydown_count >= MIN_RECOMMENDED_KEYDOWNS:
                result = self.predict_from_features(mode="Live")
                self.update_prediction_ui(result)
            else:
                self.state_label.config(text="—")
                self.confidence_label.config(
                    text=f"Waiting for keypresses: {keydown_count}/{MIN_RECOMMENDED_KEYDOWNS}"
                )

        except Exception:
            pass

        self.root.after(LIVE_PREDICTION_INTERVAL_MS, self.live_predict)

    def predict(self) -> None:
        try:
            result = self.predict_from_features(mode="Final")
            self.update_prediction_ui(result)
            self.log_prediction(result)
        except Exception as exc:
            messagebox.showerror("Prediction Error", str(exc))

    def reset(self) -> None:
        self.events = []
        self.active_keys = set()

        self.text_box.delete("1.0", tk.END)

        self.key_count_var.set("0")
        self.word_count_var.set("0")
        self.duration_var.set("0.00s")
        self.speed_var.set("0.00 WPM")

        self.key_ready_label.config(
            text=f"Keystroke Readiness: Missing 0/{MIN_RECOMMENDED_KEYDOWNS}",
            fg="#ffb3b3",
        )

        self.state_label.config(text="—")
        self.confidence_label.config(text="Confidence: —")
        self.confidence_level_label.config(
            text="Prediction Confidence: —",
            fg="#cbd6ff",
        )

        self.prob_text.delete("1.0", tk.END)
        self.feature_text.delete("1.0", tk.END)


def main() -> None:
    root = tk.Tk()
    KeystrokeDemoApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
