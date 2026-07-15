# text_live_gui.py

from __future__ import annotations

import csv
import time
from datetime import datetime
from pathlib import Path
from tkinter import messagebox
import tkinter as tk
from typing import Any

import joblib
import pandas as pd
from sentence_transformers import SentenceTransformer


MODEL_DIR = Path("models/text_demo")
MODEL_PATH = MODEL_DIR / "text_pipeline.joblib"

TEXT_EMBEDDING_MODEL = Path("models/all-mpnet-base-v2")

OUTPUT_DIR = Path("data/processed")
LOG_PATH = OUTPUT_DIR / "text_live_gui_predictions.csv"

MIN_CHARS_FOR_LIVE_PREDICTION = 20
LIVE_PREDICTION_ENABLED = True
LIVE_PREDICTION_INTERVAL_MS = 1500

LABELS = ["focused", "distracted", "fatigued", "overloaded"]


def confidence_level(gap: float) -> str:
    if gap >= 0.35:
        return "High"
    if gap >= 0.15:
        return "Medium"
    return "Low"


class TextDemoApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("SenseFuzeAI Text Live GUI")
        self.root.geometry("1080x860")
        self.root.configure(bg="#07111f")

        if not MODEL_PATH.exists():
            raise FileNotFoundError(f"Text model not found: {MODEL_PATH}")

        if not TEXT_EMBEDDING_MODEL.exists():
            raise FileNotFoundError(f"MPNet model not found: {TEXT_EMBEDDING_MODEL}")

        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        self.initialise_log_file()

        self.pipeline = joblib.load(MODEL_PATH)
        self.embedding_model = SentenceTransformer(str(TEXT_EMBEDDING_MODEL))

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
                    "character_count",
                    "word_count",
                    "embedding_dimension",
                    "runtime_seconds",
                ]
            )

    def build_ui(self) -> None:
        tk.Label(
            self.root,
            text="SenseFuzeAI Text Behavioural State Classifier",
            font=("Arial", 22, "bold"),
            fg="#74f7ff",
            bg="#07111f",
        ).pack(pady=14)

        tk.Label(
            self.root,
            text="Text-only behavioural-state prediction using MPNet semantic embeddings",
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

        self.text_ready_label = tk.Label(
            status_frame,
            text=f"Text Readiness: Missing 0/{MIN_CHARS_FOR_LIVE_PREDICTION}",
            font=("Arial", 12, "bold"),
            fg="#ffb3b3",
            bg="#10203a",
        )
        self.text_ready_label.grid(row=0, column=1, sticky="w", padx=32)

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
        self.text_box.bind("<KeyRelease>", lambda event: self.update_readiness_metrics())

        metric_frame = tk.Frame(self.root, bg="#07111f")
        metric_frame.pack(fill="x", padx=24, pady=6)

        self.char_count_var = self.create_metric_card(metric_frame, "Characters", "0", 0)
        self.word_count_var = self.create_metric_card(metric_frame, "Words", "0", 1)
        self.embedding_var = self.create_metric_card(metric_frame, "Embedding", "768", 2)
        self.model_var = self.create_metric_card(metric_frame, "Model", "MPNet", 3)

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

        self.info_text = tk.Text(
            technical_frame,
            height=10,
            width=110,
            font=("Consolas", 10),
            bg="#0b1220",
            fg="#dbeafe",
        )
        self.info_text.pack(pady=6)

        self.update_readiness_metrics()

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

    def get_text(self) -> str:
        return self.text_box.get("1.0", tk.END).strip()

    def update_readiness_metrics(self) -> None:
        text = self.get_text()
        char_count = len(text)
        word_count = len(text.split())

        self.char_count_var.set(str(char_count))
        self.word_count_var.set(str(word_count))

        if char_count >= MIN_CHARS_FOR_LIVE_PREDICTION:
            self.text_ready_label.config(
                text=f"Text Readiness: Ready {char_count}/{MIN_CHARS_FOR_LIVE_PREDICTION}",
                fg="#66ffd6",
            )
        else:
            self.text_ready_label.config(
                text=f"Text Readiness: Missing {char_count}/{MIN_CHARS_FOR_LIVE_PREDICTION}",
                fg="#ffb3b3",
            )

    def extract_embedding(self, text: str) -> pd.DataFrame:
        embedding = self.embedding_model.encode(
            [text],
            normalize_embeddings=True,
            show_progress_bar=False,
            convert_to_numpy=True,
        )

        columns = [f"text_mpnet_emb_{i}" for i in range(embedding.shape[1])]
        return pd.DataFrame(embedding, columns=columns)

    def predict_from_text(self, mode: str) -> dict[str, Any]:
        text = self.get_text()

        if len(text) < MIN_CHARS_FOR_LIVE_PREDICTION:
            raise ValueError(
                f"Please enter at least {MIN_CHARS_FOR_LIVE_PREDICTION} characters."
            )

        start = time.perf_counter()

        x = self.extract_embedding(text)

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
            "character_count": len(text),
            "word_count": len(text.split()),
            "embedding_dimension": int(x.shape[1]),
            "runtime_seconds": runtime,
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

        info_lines = [
            "Text diagnostics:",
            "",
            f"Mode                 : {result['mode']}",
            f"Current state        : {result['current_state']}",
            f"Confidence           : {result['confidence_percent']:.2f}%",
            f"Confidence level     : {result['confidence_level']}",
            f"Second-highest class : {result['second_class']}",
            f"Confidence gap       : {result['confidence_gap']:.4f}",
            f"Character count      : {result['character_count']}",
            f"Word count           : {result['word_count']}",
            f"Embedding dimension  : {result['embedding_dimension']}",
            f"Runtime              : {result['runtime_seconds']:.4f} sec",
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
                    result["mode"],
                    result["current_state"],
                    result["confidence"],
                    result["confidence_level"],
                    result["second_class"],
                    result["confidence_gap"],
                    result["character_count"],
                    result["word_count"],
                    result["embedding_dimension"],
                    result["runtime_seconds"],
                ]
            )

    def live_predict(self) -> None:
        try:
            self.update_readiness_metrics()

            text = self.get_text()

            if len(text) >= MIN_CHARS_FOR_LIVE_PREDICTION:
                result = self.predict_from_text(mode="Live")
                self.update_prediction_ui(result)
            else:
                self.state_label.config(text="—")
                self.confidence_label.config(
                    text=(
                        f"Waiting for text: "
                        f"{len(text)}/{MIN_CHARS_FOR_LIVE_PREDICTION}"
                    )
                )

        except Exception:
            pass

        self.root.after(LIVE_PREDICTION_INTERVAL_MS, self.live_predict)

    def predict(self) -> None:
        try:
            result = self.predict_from_text(mode="Final")
            self.update_prediction_ui(result)
            self.log_prediction(result)
        except Exception as exc:
            messagebox.showerror("Prediction Error", str(exc))

    def reset(self) -> None:
        self.text_box.delete("1.0", tk.END)

        self.state_label.config(text="—")
        self.confidence_label.config(text="Confidence: —")
        self.confidence_level_label.config(
            text="Prediction Confidence: —",
            fg="#cbd6ff",
        )

        self.prob_text.delete("1.0", tk.END)
        self.info_text.delete("1.0", tk.END)

        self.update_readiness_metrics()


def main() -> None:
    root = tk.Tk()
    TextDemoApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
