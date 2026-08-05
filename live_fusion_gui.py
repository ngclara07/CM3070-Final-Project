# live_fusion_gui.py

from __future__ import annotations

import json
import math
import statistics
import tempfile
import threading
import time
import uuid
import wave

from pathlib import Path
from typing import Any, Optional

import cv2
import librosa
import numpy as np
import sounddevice as sd
import tkinter as tk

from PIL import Image, ImageTk
from tkinter import filedialog, messagebox, ttk

from final_multimodal_inference import FinalMultimodalInference

from temporal_fusion import (
    LABELS,
    TEMPORAL_PROBABILITY_WINDOW,
    PROBABILITY_SUM_TOLERANCE,
    StaleGenerationError,
    TemporalFusionEngine,
    summarise_probability_dict,
    validate_probability_distribution,
)


# ============================================================
# Configuration
# ============================================================

MIN_TEXT_CHARS = 20
MIN_KEYDOWNS = 20

LIVE_FUSION_ENABLED = True
LIVE_FUSION_INTERVAL_MS = 2500

MIC_RECORD_SECONDS = 10
TARGET_SR = 16000

DISPLAY_SIZE = (
    320,
    180,
)

NEAR_SILENCE_DBFS = -50.0
QUIET_AUDIO_DBFS = -35.0


# ============================================================
# Colours
# ============================================================

BG = "#07111f"
PANEL = "#10203a"
INNER = "#0b1220"

CYAN = "#74f7ff"
GREEN = "#66ffd6"
WHITE = "#ffffff"
MUTED = "#cbd5e1"
DIM = "#94a3b8"

RED = "#ff8080"
MISSING = "#ffb3b3"

YELLOW = "#fbbf24"
BLUE = "#2563eb"
PURPLE = "#6d5dfc"
DANGER = "#b91c1c"
AUDIO_GREEN = "#00a884"


# ============================================================
# Generic numerical helpers
#
# These are keystroke helpers only.
# Temporal probability mathematics are NOT implemented here.
# ============================================================

def safe_mean(
    values: list[float],
) -> float:

    return (
        statistics.mean(values)
        if values
        else 0.0
    )


def safe_std(
    values: list[float],
) -> float:

    return (
        statistics.stdev(values)
        if len(values) >= 2
        else 0.0
    )


# ============================================================
# Keystroke feature construction
# ============================================================

def build_live_keystroke_features(
    typed_text: str,
    events: list[dict[str, Any]],
) -> dict[str, float]:

    downs = [
        event
        for event in events
        if event.get("type") == "down"
    ]

    down_times = [
        float(
            event["timestamp_perf"]
        )
        for event in downs
        if event.get(
            "timestamp_perf"
        )
        is not None
    ]

    if len(down_times) < 2:

        raise ValueError(
            "Not enough keystroke timing data."
        )

    keydown_count = len(
        downs
    )

    if keydown_count < MIN_KEYDOWNS:

        raise ValueError(
            f"At least {MIN_KEYDOWNS} "
            "key-down events are required."
        )

    delays = [
        down_times[index]
        - down_times[index - 1]
        for index
        in range(
            1,
            len(down_times),
        )
    ]

    hold_times: list[float] = []

    active_downs: dict[
        str,
        list[float],
    ] = {}

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
            or
            timestamp is None
        ):

            continue

        timestamp = float(
            timestamp
        )

        if event_type == "down":

            active_downs.setdefault(
                str(key),
                [],
            ).append(
                timestamp
            )

        elif event_type == "up":

            queue = (
                active_downs.get(
                    str(key)
                )
            )

            if queue:

                down_time = (
                    queue.pop(0)
                )

                duration = (
                    timestamp
                    - down_time
                )

                if duration >= 0.0:

                    hold_times.append(
                        duration
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
        for event
        in downs
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

    return {
        "total_duration_sec":
            round(
                total_duration,
                4,
            ),

        "keydown_count":
            keydown_count,

        "word_count":
            word_count,

        "typing_speed_kps":
            (
                round(
                    keydown_count
                    / total_duration,
                    4,
                )
                if total_duration > 0.0
                else 0.0
            ),

        "typing_speed_wpm":
            (
                round(
                    (
                        word_count
                        / total_duration
                    )
                    * 60.0,
                    4,
                )
                if total_duration > 0.0
                else 0.0
            ),

        "delay_mean":
            round(
                delay_mean,
                4,
            ),

        "delay_std":
            round(
                delay_std,
                4,
            ),

        "delay_min":
            (
                round(
                    min(delays),
                    4,
                )
                if delays
                else 0.0
            ),

        "delay_max":
            (
                round(
                    max(delays),
                    4,
                )
                if delays
                else 0.0
            ),

        "hold_mean":
            round(
                safe_mean(
                    hold_times
                ),
                4,
            ),

        "hold_std":
            round(
                safe_std(
                    hold_times
                ),
                4,
            ),

        "pause_count_1000":
            len(
                pauses_1000
            ),

        "pause_count_2000":
            len(
                pauses_2000
            ),

        "pause_count_5000":
            len(
                pauses_5000
            ),

        "pause_ratio_1000":
            (
                round(
                    len(
                        pauses_1000
                    )
                    / len(delays),
                    4,
                )
                if delays
                else 0.0
            ),

        "pause_ratio_2000":
            (
                round(
                    len(
                        pauses_2000
                    )
                    / len(delays),
                    4,
                )
                if delays
                else 0.0
            ),

        "mental_block_ratio_5000":
            (
                round(
                    len(
                        pauses_5000
                    )
                    / len(delays),
                    4,
                )
                if delays
                else 0.0
            ),

        "correction_count":
            correction_count,

        "correction_ratio":
            (
                round(
                    correction_count
                    / keydown_count,
                    4,
                )
                if keydown_count
                else 0.0
            ),

        "rhythm_consistency":
            (
                round(
                    1.0
                    / (
                        1.0
                        + delay_std
                    ),
                    4,
                )
                if delays
                else 1.0
            ),

        "burstiness_proxy":
            (
                round(
                    delay_std
                    / delay_mean,
                    4,
                )
                if delay_mean > 0.0
                else 0.0
            ),

        "fits_starts_index":
            (
                round(
                    len(
                        pauses_1000
                    )
                    / len(delays),
                    4,
                )
                if delays
                else 0.0
            ),
    }


# ============================================================
# Audio diagnostics
#
# Diagnostic only.
# Does not override model probabilities.
# ============================================================

def analyse_audio_file(
    path: Path,
) -> dict[str, Any]:

    try:

        waveform, sample_rate = (
            librosa.load(
                path,
                sr=TARGET_SR,
                mono=True,
                duration=20.0,
            )
        )

        waveform = np.asarray(
            waveform,
            dtype=np.float32,
        )

        if waveform.size == 0:

            return {
                "condition":
                    "empty",

                "analysed_duration_sec":
                    0.0,

                "rms":
                    0.0,

                "dbfs":
                    -120.0,

                "note":
                    "Audio contains no samples.",
            }

        analysed_duration = (
            len(waveform)
            / sample_rate
        )

        rms = float(
            np.sqrt(
                np.mean(
                    np.square(
                        waveform
                    )
                )
            )
        )

        dbfs = (
            20.0
            * math.log10(
                max(
                    rms,
                    1e-12,
                )
            )
        )

        if dbfs <= NEAR_SILENCE_DBFS:

            condition = (
                "near-silence"
            )

            note = (
                "Valid quiet-environment audio input; "
                "it does not force the focused label."
            )

        elif dbfs <= QUIET_AUDIO_DBFS:

            condition = "quiet"

            note = (
                "Low-energy audio input."
            )

        else:

            condition = (
                "active-audio"
            )

            note = (
                "Audible signal detected."
            )

        return {
            "condition":
                condition,

            "analysed_duration_sec":
                float(
                    analysed_duration
                ),

            "rms":
                rms,

            "dbfs":
                float(
                    dbfs
                ),

            "note":
                note,
        }

    except Exception as exc:

        return {
            "condition":
                "unknown",

            "analysed_duration_sec":
                None,

            "rms":
                None,

            "dbfs":
                None,

            "note":
                (
                    "Audio diagnostic failed: "
                    f"{exc}"
                ),
        }


# ============================================================
# Scrollable result frame
# ============================================================

class ScrollableFrame(tk.Frame):

    def __init__(
        self,
        parent: tk.Widget,
        *,
        bg: str,
    ) -> None:

        super().__init__(
            parent,
            bg=bg,
        )

        self.canvas = tk.Canvas(
            self,
            bg=bg,
            highlightthickness=0,
            borderwidth=0,
        )

        self.scrollbar = ttk.Scrollbar(
            self,
            orient="vertical",
            command=self.canvas.yview,
        )

        self.content = tk.Frame(
            self.canvas,
            bg=bg,
        )

        self.window_id = (
            self.canvas.create_window(
                (0, 0),
                window=self.content,
                anchor="nw",
            )
        )

        self.canvas.configure(
            yscrollcommand=(
                self.scrollbar.set
            )
        )

        self.canvas.grid(
            row=0,
            column=0,
            sticky="nsew",
        )

        self.scrollbar.grid(
            row=0,
            column=1,
            sticky="ns",
        )

        self.grid_rowconfigure(
            0,
            weight=1,
        )

        self.grid_columnconfigure(
            0,
            weight=1,
        )

        self.content.bind(
            "<Configure>",
            self._update_scroll_region,
        )

        self.canvas.bind(
            "<Configure>",
            self._resize_content,
        )

        self.canvas.bind(
            "<Enter>",
            self._bind_mousewheel,
        )

        self.canvas.bind(
            "<Leave>",
            self._unbind_mousewheel,
        )

    def _update_scroll_region(
        self,
        _event: Any,
    ) -> None:

        self.canvas.configure(
            scrollregion=(
                self.canvas.bbox(
                    "all"
                )
            )
        )

    def _resize_content(
        self,
        event: Any,
    ) -> None:

        self.canvas.itemconfigure(
            self.window_id,
            width=event.width,
        )

    def _bind_mousewheel(
        self,
        _event: Any,
    ) -> None:

        self.canvas.bind_all(
            "<MouseWheel>",
            self._on_mousewheel,
        )

    def _unbind_mousewheel(
        self,
        _event: Any,
    ) -> None:

        self.canvas.unbind_all(
            "<MouseWheel>"
        )

    def _on_mousewheel(
        self,
        event: Any,
    ) -> None:

        self.canvas.yview_scroll(
            int(
                -1
                * (
                    event.delta
                    / 120
                )
            ),
            "units",
        )


# ============================================================
# Main GUI
# ============================================================

class FusionDemoApp:

    def __init__(
        self,
        root: tk.Tk,
    ) -> None:

        self.root = root

        self.root.title(
            "SenseFuzeAI Live Multimodal Fusion GUI"
        )

        self.root.configure(
            bg=BG
        )

        screen_width = (
            self.root.winfo_screenwidth()
        )

        screen_height = (
            self.root.winfo_screenheight()
        )

        initial_width = min(
            1500,
            max(
                1050,
                int(
                    screen_width
                    * 0.90
                ),
            ),
        )

        initial_height = min(
            900,
            max(
                680,
                int(
                    screen_height
                    * 0.82
                ),
            ),
        )

        self.root.geometry(
            f"{initial_width}x"
            f"{initial_height}"
        )

        self.root.minsize(
            1000,
            650,
        )

        self.closed = False

        # ----------------------------------------------------
        # Canonical raw multimodal inference
        # ----------------------------------------------------

        self.predictor = (
            FinalMultimodalInference()
        )

        # ----------------------------------------------------
        # Canonical temporal fusion
        #
        # This is now the ONLY temporal-history implementation
        # used by this GUI.
        # ----------------------------------------------------

        self.temporal_fusion = (
            TemporalFusionEngine(
                window_size=(
                    TEMPORAL_PROBABILITY_WINDOW
                ),
                labels=LABELS,
            )
        )

        # ----------------------------------------------------
        # Keystrokes
        # ----------------------------------------------------

        self.keystroke_events: list[
            dict[str, Any]
        ] = []

        self.active_keys: set[
            str
        ] = set()

        # ----------------------------------------------------
        # Audio
        # ----------------------------------------------------

        self.audio_path: Optional[
            Path
        ] = None

        self.audio_source_name: Optional[
            str
        ] = None

        self.audio_diagnostics: Optional[
            dict[str, Any]
        ] = None

        # ----------------------------------------------------
        # Visual
        # ----------------------------------------------------

        self.image_path: Optional[
            Path
        ] = None

        self.image_source_name: Optional[
            str
        ] = None

        self.image_source_type = (
            "none"
        )

        self.capture: Optional[
            cv2.VideoCapture
        ] = None

        self.running_visual_stream = (
            False
        )

        self.current_frame: Optional[
            np.ndarray
        ] = None

        self.preview_image: Optional[
            ImageTk.PhotoImage
        ] = None

        # ----------------------------------------------------
        # Prediction lifecycle
        # ----------------------------------------------------

        self.fusion_prediction_busy = (
            False
        )

        # ----------------------------------------------------
        # Optional labelled-test diagnostics
        # ----------------------------------------------------

        self.expected_state_var = (
            tk.StringVar(
                value="unlabelled"
            )
        )

        self.live_labelled_trials = 0
        self.live_labelled_matches = 0

        self.last_evaluated_generation = (
            -1
        )

        # ----------------------------------------------------
        # Temporary files
        # ----------------------------------------------------

        self.temp_directory = (
            tempfile.TemporaryDirectory(
                prefix="sensefuze_gui_"
            )
        )

        self.temp_dir = Path(
            self.temp_directory.name
        )

        # ----------------------------------------------------
        # UI
        # ----------------------------------------------------

        self._build_ui()

        self.root.protocol(
            "WM_DELETE_WINDOW",
            self.on_close,
        )

        if LIVE_FUSION_ENABLED:

            self.root.after(
                LIVE_FUSION_INTERVAL_MS,
                self.live_fusion_tick,
            )

    # ========================================================
    # UI
    # ========================================================

    def _build_ui(
        self,
    ) -> None:

        self.root.grid_rowconfigure(
            0,
            weight=0,
        )

        self.root.grid_rowconfigure(
            1,
            weight=0,
        )

        self.root.grid_rowconfigure(
            2,
            weight=1,
        )

        self.root.grid_rowconfigure(
            3,
            weight=0,
        )

        self.root.grid_columnconfigure(
            0,
            weight=1,
        )

        self._build_header()
        self._build_readiness()
        self._build_body()
        self._build_footer()

        self.update_readiness()

    def _build_header(
        self,
    ) -> None:

        frame = tk.Frame(
            self.root,
            bg=BG,
        )

        frame.grid(
            row=0,
            column=0,
            sticky="ew",
            padx=18,
            pady=(
                8,
                2,
            ),
        )

        tk.Label(
            frame,
            text=(
                "SenseFuzeAI Live "
                "Multimodal Fusion System"
            ),
            font=(
                "Arial",
                22,
                "bold",
            ),
            fg=CYAN,
            bg=BG,
        ).pack()

        tk.Label(
            frame,
            text=(
                "Text · Keystroke · Audio · "
                "Vision · Temporal Probability Fusion"
            ),
            font=(
                "Arial",
                11,
            ),
            fg=WHITE,
            bg=BG,
        ).pack(
            pady=(
                2,
                3,
            )
        )

    def _build_readiness(
        self,
    ) -> None:

        frame = tk.Frame(
            self.root,
            bg=PANEL,
            padx=12,
            pady=7,
        )

        frame.grid(
            row=1,
            column=0,
            sticky="ew",
            padx=18,
            pady=4,
        )

        self.model_ready_label = tk.Label(
            frame,
            text="Fusion Model: Loaded",
            font=(
                "Arial",
                10,
                "bold",
            ),
            fg=GREEN,
            bg=PANEL,
        )

        self.model_ready_label.grid(
            row=0,
            column=0,
            padx=10,
            sticky="w",
        )

        self.text_ready_label = tk.Label(
            frame,
            text=(
                f"Text: 0/"
                f"{MIN_TEXT_CHARS}"
            ),
            fg=MISSING,
            bg=PANEL,
        )

        self.text_ready_label.grid(
            row=0,
            column=1,
            padx=10,
        )

        self.key_ready_label = tk.Label(
            frame,
            text=(
                f"Keystroke: 0/"
                f"{MIN_KEYDOWNS}"
            ),
            fg=MISSING,
            bg=PANEL,
        )

        self.key_ready_label.grid(
            row=0,
            column=2,
            padx=10,
        )

        self.audio_ready_label = tk.Label(
            frame,
            text="Audio: Missing",
            fg=MISSING,
            bg=PANEL,
        )

        self.audio_ready_label.grid(
            row=0,
            column=3,
            padx=10,
        )

        self.image_ready_label = tk.Label(
            frame,
            text="Image: Missing",
            fg=MISSING,
            bg=PANEL,
        )

        self.image_ready_label.grid(
            row=0,
            column=4,
            padx=10,
        )

        frame.grid_columnconfigure(
            5,
            weight=1,
        )

    def _build_body(
        self,
    ) -> None:

        body = tk.Frame(
            self.root,
            bg=BG,
        )

        body.grid(
            row=2,
            column=0,
            sticky="nsew",
            padx=18,
            pady=5,
        )

        body.grid_rowconfigure(
            0,
            weight=1,
        )

        body.grid_columnconfigure(
            0,
            weight=3,
            uniform="body",
        )

        body.grid_columnconfigure(
            1,
            weight=2,
            uniform="body",
        )

        left = tk.Frame(
            body,
            bg=BG,
        )

        left.grid(
            row=0,
            column=0,
            sticky="nsew",
            padx=(
                0,
                7,
            ),
        )

        left.grid_rowconfigure(
            0,
            weight=1,
        )

        left.grid_rowconfigure(
            1,
            weight=0,
        )

        left.grid_columnconfigure(
            0,
            weight=1,
        )

        self._build_text_panel(
            left
        )

        self._build_audio_panel(
            left
        )

        right = tk.Frame(
            body,
            bg=BG,
        )

        right.grid(
            row=0,
            column=1,
            sticky="nsew",
            padx=(
                7,
                0,
            ),
        )

        right.grid_columnconfigure(
            0,
            weight=1,
        )

        right.grid_rowconfigure(
            0,
            weight=0,
        )

        right.grid_rowconfigure(
            1,
            weight=1,
        )

        self._build_visual_panel(
            right
        )

        self._build_result_panel(
            right
        )

    def _build_text_panel(
        self,
        parent: tk.Widget,
    ) -> None:

        frame = tk.LabelFrame(
            parent,
            text="Text + Keystroke Input",
            font=(
                "Arial",
                11,
                "bold",
            ),
            fg=CYAN,
            bg=BG,
            padx=7,
            pady=5,
        )

        frame.grid(
            row=0,
            column=0,
            sticky="nsew",
            pady=(
                0,
                5,
            ),
        )

        frame.grid_rowconfigure(
            0,
            weight=1,
        )

        frame.grid_columnconfigure(
            0,
            weight=1,
        )

        self.text_box = tk.Text(
            frame,
            font=(
                "Arial",
                12,
            ),
            bg=INNER,
            fg=WHITE,
            insertbackground=WHITE,
            wrap="word",
        )

        self.text_box.grid(
            row=0,
            column=0,
            sticky="nsew",
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

    def _build_audio_panel(
        self,
        parent: tk.Widget,
    ) -> None:

        frame = tk.LabelFrame(
            parent,
            text="Audio Input",
            font=(
                "Arial",
                11,
                "bold",
            ),
            fg=CYAN,
            bg=BG,
            padx=8,
            pady=5,
        )

        frame.grid(
            row=1,
            column=0,
            sticky="ew",
            pady=(
                5,
                0,
            ),
        )

        frame.grid_columnconfigure(
            2,
            weight=1,
        )

        tk.Button(
            frame,
            text="Choose Audio File",
            command=self.choose_audio_file,
            width=20,
        ).grid(
            row=0,
            column=0,
            padx=4,
            pady=2,
            sticky="w",
        )

        tk.Button(
            frame,
            text=(
                f"Record Microphone "
                f"({MIC_RECORD_SECONDS}s)"
            ),
            command=self.record_microphone,
            width=23,
            bg=AUDIO_GREEN,
            fg=WHITE,
        ).grid(
            row=0,
            column=1,
            padx=4,
            pady=2,
            sticky="w",
        )

        self.audio_label = tk.Label(
            frame,
            text="Audio not loaded.",
            fg=MISSING,
            bg=BG,
            anchor="w",
            justify="left",
        )

        self.audio_label.grid(
            row=1,
            column=0,
            columnspan=3,
            sticky="ew",
            padx=5,
            pady=(
                4,
                1,
            ),
        )

        self.audio_diagnostic_label = tk.Label(
            frame,
            text="Audio condition: —",
            fg=MUTED,
            bg=BG,
            anchor="w",
            justify="left",
            wraplength=760,
            font=(
                "Arial",
                9,
            ),
        )

        self.audio_diagnostic_label.grid(
            row=2,
            column=0,
            columnspan=3,
            sticky="ew",
            padx=5,
            pady=(
                1,
                2,
            ),
        )

    def _build_visual_panel(
        self,
        parent: tk.Widget,
    ) -> None:

        frame = tk.LabelFrame(
            parent,
            text="Visual Input",
            font=(
                "Arial",
                11,
                "bold",
            ),
            fg=CYAN,
            bg=BG,
            padx=7,
            pady=4,
        )

        frame.grid(
            row=0,
            column=0,
            sticky="ew",
            pady=(
                0,
                5,
            ),
        )

        controls = tk.Frame(
            frame,
            bg=BG,
        )

        controls.pack(
            fill="x",
        )

        tk.Button(
            controls,
            text="Choose Image",
            command=self.choose_image,
        ).pack(
            side="left",
            padx=3,
        )

        tk.Button(
            controls,
            text="Choose Video",
            command=self.choose_video,
        ).pack(
            side="left",
            padx=3,
        )

        tk.Button(
            controls,
            text="Start Webcam",
            command=self.start_webcam,
        ).pack(
            side="left",
            padx=3,
        )

        tk.Button(
            controls,
            text="Stop Visual",
            command=self.stop_visual_stream,
        ).pack(
            side="left",
            padx=3,
        )

        self.visual_source_label = tk.Label(
            frame,
            text="Visual input not loaded.",
            fg=MISSING,
            bg=BG,
            anchor="center",
            justify="center",
            wraplength=520,
            font=(
                "Arial",
                9,
            ),
        )

        self.visual_source_label.pack(
            fill="x",
            pady=(
                3,
                1,
            ),
        )

        self.preview_label = tk.Label(
            frame,
            bg=BG,
        )

        self.preview_label.pack(
            pady=(
                1,
                2,
            ),
        )

    def _build_result_panel(
        self,
        parent: tk.Widget,
    ) -> None:

        outer = tk.LabelFrame(
            parent,
            text="Temporal Fusion Result",
            font=(
                "Arial",
                11,
                "bold",
            ),
            fg=CYAN,
            bg=BG,
            padx=5,
            pady=4,
        )

        outer.grid(
            row=1,
            column=0,
            sticky="nsew",
        )

        outer.grid_rowconfigure(
            0,
            weight=1,
        )

        outer.grid_columnconfigure(
            0,
            weight=1,
        )

        scrolling = ScrollableFrame(
            outer,
            bg=BG,
        )

        scrolling.grid(
            row=0,
            column=0,
            sticky="nsew",
        )

        content = (
            scrolling.content
        )

        content.grid_columnconfigure(
            0,
            weight=1,
        )

        self.result_label = tk.Label(
            content,
            text="Waiting for inputs",
            font=(
                "Arial",
                20,
                "bold",
            ),
            fg=GREEN,
            bg=BG,
        )

        self.result_label.grid(
            row=0,
            column=0,
            sticky="ew",
            pady=(
                2,
                1,
            ),
        )

        self.confidence_label = tk.Label(
            content,
            text="Confidence: —",
            font=(
                "Arial",
                10,
            ),
            fg=WHITE,
            bg=BG,
        )

        self.confidence_label.grid(
            row=1,
            column=0,
            sticky="ew",
        )

        self.raw_label = tk.Label(
            content,
            text="Raw fusion: —",
            font=(
                "Arial",
                9,
            ),
            fg=MUTED,
            bg=BG,
        )

        self.raw_label.grid(
            row=2,
            column=0,
            sticky="ew",
        )

        self.temporal_label = tk.Label(
            content,
            text=(
                "Temporal samples: "
                f"0/"
                f"{TEMPORAL_PROBABILITY_WINDOW}"
            ),
            font=(
                "Arial",
                9,
            ),
            fg=MUTED,
            bg=BG,
        )

        self.temporal_label.grid(
            row=3,
            column=0,
            sticky="ew",
            pady=(
                0,
                3,
            ),
        )

        probability_frame = tk.LabelFrame(
            content,
            text="Full Probability Distribution",
            font=(
                "Arial",
                9,
                "bold",
            ),
            fg=CYAN,
            bg=BG,
            padx=5,
            pady=3,
        )

        probability_frame.grid(
            row=4,
            column=0,
            sticky="ew",
            padx=3,
            pady=3,
        )

        probability_frame.grid_columnconfigure(
            0,
            weight=2,
        )

        probability_frame.grid_columnconfigure(
            1,
            weight=1,
        )

        probability_frame.grid_columnconfigure(
            2,
            weight=1,
        )

        tk.Label(
            probability_frame,
            text="Class",
            font=(
                "Arial",
                9,
                "bold",
            ),
            fg=DIM,
            bg=BG,
        ).grid(
            row=0,
            column=0,
            sticky="w",
            padx=7,
            pady=1,
        )

        tk.Label(
            probability_frame,
            text="Raw",
            font=(
                "Arial",
                9,
                "bold",
            ),
            fg=YELLOW,
            bg=BG,
        ).grid(
            row=0,
            column=1,
            padx=7,
            pady=1,
        )

        tk.Label(
            probability_frame,
            text="Temporal",
            font=(
                "Arial",
                9,
                "bold",
            ),
            fg=GREEN,
            bg=BG,
        ).grid(
            row=0,
            column=2,
            padx=7,
            pady=1,
        )

        self.raw_probability_labels: dict[
            str,
            tk.Label
        ] = {}

        self.temporal_probability_labels: dict[
            str,
            tk.Label
        ] = {}

        for row_index, label in enumerate(
            LABELS,
            start=1,
        ):

            tk.Label(
                probability_frame,
                text=label.capitalize(),
                font=(
                    "Arial",
                    9,
                ),
                fg=WHITE,
                bg=BG,
                anchor="w",
            ).grid(
                row=row_index,
                column=0,
                sticky="w",
                padx=7,
                pady=1,
            )

            raw_value = tk.Label(
                probability_frame,
                text="—",
                font=(
                    "Consolas",
                    9,
                    "bold",
                ),
                fg=YELLOW,
                bg=BG,
            )

            raw_value.grid(
                row=row_index,
                column=1,
                padx=7,
                pady=1,
            )

            temporal_value = tk.Label(
                probability_frame,
                text="—",
                font=(
                    "Consolas",
                    9,
                    "bold",
                ),
                fg=GREEN,
                bg=BG,
            )

            temporal_value.grid(
                row=row_index,
                column=2,
                padx=7,
                pady=1,
            )

            self.raw_probability_labels[
                label
            ] = raw_value

            self.temporal_probability_labels[
                label
            ] = temporal_value

        self.probability_sum_label = tk.Label(
            probability_frame,
            text=(
                "Raw sum: —    "
                "Temporal sum: —"
            ),
            font=(
                "Arial",
                8,
            ),
            fg=MUTED,
            bg=BG,
        )

        self.probability_sum_label.grid(
            row=5,
            column=0,
            columnspan=3,
            sticky="ew",
            pady=(
                3,
                0,
            ),
        )

        self.validation_label = tk.Label(
            content,
            text=(
                "Runtime validation: waiting.\n"
                "Behavioural accuracy requires "
                "labelled repeated trials."
            ),
            justify="left",
            anchor="w",
            fg=MUTED,
            bg=BG,
            wraplength=520,
            font=(
                "Arial",
                8,
            ),
        )

        self.validation_label.grid(
            row=5,
            column=0,
            sticky="ew",
            padx=6,
            pady=(
                2,
                2,
            ),
        )

        expected_frame = tk.Frame(
            content,
            bg=BG,
        )

        expected_frame.grid(
            row=6,
            column=0,
            sticky="ew",
            padx=5,
            pady=(
                2,
                5,
            ),
        )

        tk.Label(
            expected_frame,
            text="Expected state:",
            fg=MUTED,
            bg=BG,
            font=(
                "Arial",
                8,
            ),
        ).pack(
            side="left",
            padx=(
                0,
                4,
            ),
        )

        expected_combo = ttk.Combobox(
            expected_frame,
            textvariable=(
                self.expected_state_var
            ),
            values=[
                "unlabelled",
                *LABELS,
            ],
            width=12,
            state="readonly",
        )

        expected_combo.pack(
            side="left",
            padx=3,
        )

        self.live_test_label = tk.Label(
            expected_frame,
            text="Live labelled checks: 0",
            fg=MUTED,
            bg=BG,
            font=(
                "Arial",
                8,
            ),
        )

        self.live_test_label.pack(
            side="left",
            padx=8,
        )

    def _build_footer(
        self,
    ) -> None:

        footer = tk.Frame(
            self.root,
            bg=BG,
            padx=16,
            pady=4,
        )

        footer.grid(
            row=3,
            column=0,
            sticky="ew",
        )

        footer.grid_columnconfigure(
            0,
            weight=1,
        )

        controls = tk.Frame(
            footer,
            bg=BG,
        )

        controls.grid(
            row=0,
            column=0,
            pady=(
                0,
                3,
            ),
        )

        tk.Button(
            controls,
            text="Run Prediction Now",
            command=(
                self.predict_fusion_threaded
            ),
            width=20,
            font=(
                "Arial",
                10,
                "bold",
            ),
            bg=BLUE,
            fg=WHITE,
        ).grid(
            row=0,
            column=0,
            padx=5,
        )

        tk.Button(
            controls,
            text="Reset Temporal Window",
            command=(
                self.reset_temporal_history
            ),
            width=22,
            font=(
                "Arial",
                10,
                "bold",
            ),
            bg=PURPLE,
            fg=WHITE,
        ).grid(
            row=0,
            column=1,
            padx=5,
        )

        tk.Button(
            controls,
            text="Full Reset",
            command=self.reset,
            width=16,
            font=(
                "Arial",
                10,
                "bold",
            ),
            bg=DANGER,
            fg=WHITE,
        ).grid(
            row=0,
            column=2,
            padx=5,
        )

        self.status_label = tk.Label(
            footer,
            text=(
                "Waiting for all four modalities."
            ),
            fg=MUTED,
            bg=BG,
            anchor="w",
            justify="left",
            font=(
                "Arial",
                9,
            ),
        )

        self.status_label.grid(
            row=1,
            column=0,
            sticky="ew",
            padx=4,
        )

    # ========================================================
    # Keyboard
    # ========================================================

    @staticmethod
    def normalise_key(
        event: Any,
    ) -> str:

        if event.keysym == "BackSpace":
            return "backspace"

        if event.keysym == "Delete":
            return "delete"

        if event.keysym == "space":
            return "space"

        if len(event.char) == 1:
            return event.char.lower()

        return event.keysym.lower()

    def on_key_press(
        self,
        event: Any,
    ) -> None:

        key = self.normalise_key(
            event
        )

        if key in self.active_keys:
            return

        self.active_keys.add(
            key
        )

        self.keystroke_events.append(
            {
                "type":
                    "down",

                "key":
                    key,

                "timestamp_perf":
                    time.perf_counter(),

                "timestamp_epoch":
                    time.time(),
            }
        )

        self.update_readiness()

    def on_key_release(
        self,
        event: Any,
    ) -> None:

        key = self.normalise_key(
            event
        )

        self.active_keys.discard(
            key
        )

        self.keystroke_events.append(
            {
                "type":
                    "up",

                "key":
                    key,

                "timestamp_perf":
                    time.perf_counter(),

                "timestamp_epoch":
                    time.time(),
            }
        )

        self.update_readiness()

    def on_text_modified(
        self,
        _event: Any,
    ) -> None:

        try:

            self.text_box.edit_modified(
                False
            )

        except Exception:

            pass

        self.update_readiness()

    # ========================================================
    # Input readiness
    # ========================================================

    def current_text(
        self,
    ) -> str:

        return (
            self.text_box
            .get(
                "1.0",
                "end-1c",
            )
            .strip()
        )

    def count_keydowns(
        self,
    ) -> int:

        return sum(
            1
            for event
            in self.keystroke_events
            if event.get("type")
            == "down"
        )

    def visual_ready(
        self,
    ) -> bool:

        if (
            self.image_source_type
            == "image"
        ):

            return bool(
                self.image_path
                and
                self.image_path.exists()
            )

        return (
            self.current_frame
            is not None
        )

    def fusion_inputs_ready(
        self,
    ) -> bool:

        return (
            len(
                self.current_text()
            )
            >= MIN_TEXT_CHARS

            and

            self.count_keydowns()
            >= MIN_KEYDOWNS

            and

            self.audio_path
            is not None

            and

            self.audio_path.exists()

            and

            self.visual_ready()
        )

    def update_readiness(
        self,
    ) -> None:

        text_count = len(
            self.current_text()
        )

        key_count = (
            self.count_keydowns()
        )

        text_ok = (
            text_count
            >= MIN_TEXT_CHARS
        )

        key_ok = (
            key_count
            >= MIN_KEYDOWNS
        )

        audio_ok = bool(
            self.audio_path
            and
            self.audio_path.exists()
        )

        image_ok = (
            self.visual_ready()
        )

        self.text_ready_label.config(
            text=(
                f"Text: "
                f"{text_count}/"
                f"{MIN_TEXT_CHARS}"
            ),
            fg=(
                GREEN
                if text_ok
                else MISSING
            ),
        )

        self.key_ready_label.config(
            text=(
                f"Keystroke: "
                f"{key_count}/"
                f"{MIN_KEYDOWNS}"
            ),
            fg=(
                GREEN
                if key_ok
                else MISSING
            ),
        )

        self.audio_ready_label.config(
            text=(
                "Audio: Ready"
                if audio_ok
                else "Audio: Missing"
            ),
            fg=(
                GREEN
                if audio_ok
                else MISSING
            ),
        )

        self.image_ready_label.config(
            text=(
                "Image: Ready"
                if image_ok
                else "Image: Missing"
            ),
            fg=(
                GREEN
                if image_ok
                else MISSING
            ),
        )

    # ========================================================
    # Audio
    # ========================================================

    def choose_audio_file(
        self,
    ) -> None:

        filename = (
            filedialog.askopenfilename(
                title="Choose Audio File",
                filetypes=[
                    (
                        "Audio files",
                        (
                            "*.wav *.mp3 *.flac "
                            "*.ogg *.m4a *.webm"
                        ),
                    ),
                    (
                        "All Files",
                        "*.*",
                    ),
                ],
            )
        )

        if not filename:
            return

        self.audio_path = Path(
            filename
        )

        self.audio_source_name = (
            self.audio_path.name
        )

        self.audio_diagnostics = (
            analyse_audio_file(
                self.audio_path
            )
        )

        self.audio_label.config(
            text=(
                "Audio: "
                f"{self.audio_source_name}"
            ),
            fg=GREEN,
        )

        self.update_audio_diagnostic_ui()

        # New source = new temporal generation.
        self.reset_temporal_history(
            silent=True
        )

        self.update_readiness()

    def update_audio_diagnostic_ui(
        self,
    ) -> None:

        diagnostic = (
            self.audio_diagnostics
            or {}
        )

        condition = diagnostic.get(
            "condition",
            "unknown",
        )

        duration = diagnostic.get(
            "analysed_duration_sec"
        )

        dbfs = diagnostic.get(
            "dbfs"
        )

        note = diagnostic.get(
            "note",
            "",
        )

        duration_text = (
            f"{duration:.2f}s"
            if isinstance(
                duration,
                (
                    int,
                    float,
                ),
            )
            else "—"
        )

        dbfs_text = (
            f"{dbfs:.1f} dBFS"
            if isinstance(
                dbfs,
                (
                    int,
                    float,
                ),
            )
            else "—"
        )

        self.audio_diagnostic_label.config(
            text=(
                f"Audio condition: "
                f"{condition} | "
                f"Analysed: "
                f"{duration_text} | "
                f"Level: "
                f"{dbfs_text}\n"
                f"{note}"
            )
        )

    def record_microphone(
        self,
    ) -> None:

        self.audio_label.config(
            text=(
                f"Recording microphone "
                f"for {MIC_RECORD_SECONDS}s..."
            ),
            fg=YELLOW,
        )

        threading.Thread(
            target=(
                self.record_microphone_worker
            ),
            daemon=True,
        ).start()

    def record_microphone_worker(
        self,
    ) -> None:

        try:

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
                recording.reshape(-1)
            )

            waveform = np.clip(
                waveform,
                -1.0,
                1.0,
            )

            pcm16 = (
                waveform
                * 32767.0
            ).astype(
                np.int16
            )

            path = (
                self.temp_dir
                / "microphone.wav"
            )

            with wave.open(
                str(path),
                "wb",
            ) as wav_file:

                wav_file.setnchannels(
                    1
                )

                wav_file.setsampwidth(
                    2
                )

                wav_file.setframerate(
                    TARGET_SR
                )

                wav_file.writeframes(
                    pcm16.tobytes()
                )

            if not self.closed:

                self.root.after(
                    0,
                    lambda:
                        self.finish_microphone_recording(
                            path
                        ),
                )

        except Exception as exc:

            if not self.closed:

                self.root.after(
                    0,
                    lambda error=exc:
                        messagebox.showerror(
                            "Microphone Error",
                            str(error),
                        ),
                )

    def finish_microphone_recording(
        self,
        path: Path,
    ) -> None:

        self.audio_path = path

        self.audio_source_name = (
            "microphone"
        )

        self.audio_diagnostics = (
            analyse_audio_file(
                path
            )
        )

        self.audio_label.config(
            text=(
                "Microphone recording ready "
                f"({MIC_RECORD_SECONDS}s)."
            ),
            fg=GREEN,
        )

        self.update_audio_diagnostic_ui()

        self.reset_temporal_history(
            silent=True
        )

        self.update_readiness()

    # ========================================================
    # Visual
    # ========================================================

    def choose_image(
        self,
    ) -> None:

        filename = (
            filedialog.askopenfilename(
                title="Choose Image",
                filetypes=[
                    (
                        "Images",
                        (
                            "*.jpg *.jpeg *.png "
                            "*.bmp *.webp"
                        ),
                    ),
                    (
                        "All Files",
                        "*.*",
                    ),
                ],
            )
        )

        if not filename:
            return

        self.stop_visual_stream(
            update_status=False
        )

        self.image_source_type = (
            "image"
        )

        self.image_path = Path(
            filename
        )

        self.image_source_name = (
            self.image_path.name
        )

        self.current_frame = None

        self.display_image_file(
            self.image_path
        )

        self.visual_source_label.config(
            text=(
                "Image: "
                f"{self.image_source_name}"
            ),
            fg=GREEN,
        )

        self.reset_temporal_history(
            silent=True
        )

        self.update_readiness()

    def choose_video(
        self,
    ) -> None:

        filename = (
            filedialog.askopenfilename(
                title="Choose Video",
                filetypes=[
                    (
                        "Videos",
                        (
                            "*.mp4 *.avi *.mov "
                            "*.mkv *.webm"
                        ),
                    ),
                    (
                        "All Files",
                        "*.*",
                    ),
                ],
            )
        )

        if not filename:
            return

        self.stop_visual_stream(
            update_status=False
        )

        capture = cv2.VideoCapture(
            filename
        )

        if not capture.isOpened():

            capture.release()

            messagebox.showerror(
                "Video Error",
                "Could not open video.",
            )

            return

        self.capture = capture

        self.image_source_type = (
            "video"
        )

        self.image_source_name = (
            Path(filename).name
        )

        self.image_path = None
        self.current_frame = None

        self.running_visual_stream = (
            True
        )

        self.visual_source_label.config(
            text=(
                "Video: "
                f"{self.image_source_name}"
            ),
            fg=GREEN,
        )

        self.reset_temporal_history(
            silent=True
        )

        self.visual_tick()

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
                "Could not open webcam.",
            )

            return

        self.capture = capture

        self.image_source_type = (
            "webcam"
        )

        self.image_source_name = (
            "webcam"
        )

        self.image_path = None
        self.current_frame = None

        self.running_visual_stream = (
            True
        )

        self.visual_source_label.config(
            text="Webcam active.",
            fg=GREEN,
        )

        self.reset_temporal_history(
            silent=True
        )

        self.visual_tick()

    def stop_visual_stream(
        self,
        update_status: bool = True,
    ) -> None:

        self.running_visual_stream = (
            False
        )

        if self.capture is not None:

            try:

                self.capture.release()

            except Exception:

                pass

        self.capture = None

        if (
            self.image_source_type
            in {
                "video",
                "webcam",
            }
        ):

            self.current_frame = None

            self.image_source_type = (
                "none"
            )

            self.image_source_name = None

            self.preview_image = None

            try:

                self.preview_label.config(
                    image=""
                )

            except Exception:

                pass

        if update_status:

            self.visual_source_label.config(
                text="Visual stream stopped.",
                fg=MISSING,
            )

        self.update_readiness()

    def visual_tick(
        self,
    ) -> None:

        if (
            not self.running_visual_stream
            or
            self.capture is None
        ):

            return

        success, frame = (
            self.capture.read()
        )

        if not success:

            if (
                self.image_source_type
                == "video"
            ):

                self.capture.set(
                    cv2.CAP_PROP_POS_FRAMES,
                    0,
                )

                self.root.after(
                    30,
                    self.visual_tick,
                )

                return

            self.stop_visual_stream()

            return

        self.current_frame = (
            frame.copy()
        )

        self.display_cv_frame(
            frame
        )

        self.update_readiness()

        self.root.after(
            30,
            self.visual_tick,
        )

    def display_image_file(
        self,
        path: Path,
    ) -> None:

        image = (
            Image.open(
                path
            )
            .convert(
                "RGB"
            )
        )

        image.thumbnail(
            DISPLAY_SIZE,
            Image.Resampling.LANCZOS,
        )

        self.preview_image = (
            ImageTk.PhotoImage(
                image
            )
        )

        self.preview_label.config(
            image=self.preview_image
        )

    def display_cv_frame(
        self,
        frame: np.ndarray,
    ) -> None:

        rgb = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB,
        )

        image = Image.fromarray(
            rgb
        )

        image.thumbnail(
            DISPLAY_SIZE,
            Image.Resampling.LANCZOS,
        )

        self.preview_image = (
            ImageTk.PhotoImage(
                image
            )
        )

        self.preview_label.config(
            image=self.preview_image
        )

    # ========================================================
    # Prediction snapshots
    # ========================================================

    def create_image_snapshot(
        self,
    ) -> tuple[
        Path,
        bool,
    ]:

        if (
            self.image_source_type
            == "image"
            and
            self.image_path is not None
            and
            self.image_path.exists()
        ):

            return (
                self.image_path,
                False,
            )

        if self.current_frame is None:

            raise ValueError(
                "Current visual frame is unavailable."
            )

        path = (
            self.temp_dir
            / (
                "frame_"
                f"{uuid.uuid4().hex}"
                ".jpg"
            )
        )

        success = cv2.imwrite(
            str(path),
            self.current_frame.copy(),
        )

        if not success:

            raise RuntimeError(
                "Could not create visual snapshot."
            )

        return (
            path,
            True,
        )

    def create_keystroke_json(
        self,
        text: str,
        events: list[
            dict[str, Any]
        ],
    ) -> Path:

        features = (
            build_live_keystroke_features(
                text,
                events,
            )
        )

        path = (
            self.temp_dir
            / (
                "keystrokes_"
                f"{uuid.uuid4().hex}"
                ".json"
            )
        )

        payload = {
            "features":
                features,

            "events":
                events,

            "typed_text":
                text,
        }

        path.write_text(
            json.dumps(
                payload,
                indent=2,
            ),
            encoding="utf-8",
        )

        return path

    # ========================================================
    # Scheduler
    # ========================================================

    def live_fusion_tick(
        self,
    ) -> None:

        if self.closed:
            return

        try:

            self.update_readiness()

            if (
                not self.fusion_prediction_busy
                and
                self.fusion_inputs_ready()
            ):

                self.predict_fusion_threaded()

        finally:

            if (
                LIVE_FUSION_ENABLED
                and
                not self.closed
            ):

                self.root.after(
                    LIVE_FUSION_INTERVAL_MS,
                    self.live_fusion_tick,
                )

    # ========================================================
    # Prediction
    # ========================================================

    def predict_fusion_threaded(
        self,
    ) -> None:

        if self.fusion_prediction_busy:
            return

        if not self.fusion_inputs_ready():

            self.status_label.config(
                text=(
                    "Prediction blocked: "
                    "text, keystrokes, audio and "
                    "visual input are all required."
                )
            )

            return

        text = self.current_text()

        events = [
            dict(event)
            for event
            in self.keystroke_events
        ]

        audio_path = Path(
            self.audio_path
        )

        try:

            (
                image_path,
                image_is_temporary,
            ) = (
                self.create_image_snapshot()
            )

            keystroke_path = (
                self.create_keystroke_json(
                    text,
                    events,
                )
            )

        except Exception as exc:

            self.show_prediction_error(
                exc
            )

            return

        prediction_generation = (
            self.temporal_fusion
            .capture_generation()
        )

        self.fusion_prediction_busy = (
            True
        )

        self.status_label.config(
            text=(
                "Running canonical multimodal "
                "fusion inference..."
            )
        )

        threading.Thread(
            target=self.prediction_worker,
            kwargs={
                "generation":
                    prediction_generation,

                "text":
                    text,

                "audio_path":
                    audio_path,

                "image_path":
                    image_path,

                "image_is_temporary":
                    image_is_temporary,

                "keystroke_path":
                    keystroke_path,
            },
            daemon=True,
        ).start()

    def prediction_worker(
        self,
        *,
        generation: int,
        text: str,
        audio_path: Path,
        image_path: Path,
        image_is_temporary: bool,
        keystroke_path: Path,
    ) -> None:

        started = (
            time.perf_counter()
        )

        try:

            raw_result = (
                self.predictor.predict(
                    keystroke_json=(
                        keystroke_path
                    ),
                    text=text,
                    audio_path=(
                        audio_path
                    ),
                    image_path=(
                        image_path
                    ),
                )
            )

            runtime = (
                time.perf_counter()
                - started
            )

            if not self.closed:

                self.root.after(
                    0,
                    lambda:
                        self.apply_prediction(
                            generation=(
                                generation
                            ),
                            raw_result=(
                                raw_result
                            ),
                            runtime=(
                                runtime
                            ),
                        ),
                )

        except Exception as exc:

            if not self.closed:

                self.root.after(
                    0,
                    lambda error=exc:
                        self.apply_prediction_error(
                            generation,
                            error,
                        ),
                )

        finally:

            try:

                keystroke_path.unlink(
                    missing_ok=True
                )

            except Exception:

                pass

            if image_is_temporary:

                try:

                    image_path.unlink(
                        missing_ok=True
                    )

                except Exception:

                    pass

            if not self.closed:

                self.root.after(
                    0,
                    self.finish_prediction,
                )

    def apply_prediction(
        self,
        *,
        generation: int,
        raw_result: dict[str, Any],
        runtime: float,
    ) -> None:

        # ----------------------------------------------------
        # Raw probability summary uses shared implementation.
        # ----------------------------------------------------

        raw_summary = (
            summarise_probability_dict(
                raw_result.get(
                    "probabilities",
                    {},
                ),
                labels=LABELS,
            )
        )

        raw_probabilities = (
            raw_summary[
                "probabilities"
            ]
        )

        raw_state = (
            raw_summary[
                "current_state"
            ]
        )

        raw_confidence = (
            raw_summary[
                "confidence"
            ]
        )

        # ----------------------------------------------------
        # Temporal append + generation check are atomic inside
        # TemporalFusionEngine.
        # ----------------------------------------------------

        try:

            temporal = (
                self.temporal_fusion.append(
                    raw_probabilities,
                    expected_generation=(
                        generation
                    ),
                )
            )

        except StaleGenerationError:

            # Prediction started before reset/source change.
            # Do not show it and do not add it to history.
            return

        # ----------------------------------------------------
        # Main result
        # ----------------------------------------------------

        self.result_label.config(
            text=(
                temporal[
                    "current_state"
                ].upper()
            ),
            fg=GREEN,
        )

        self.confidence_label.config(
            text=(
                f"Confidence: "
                f"{temporal['confidence_percent']:.2f}%"
                f" | "
                f"{temporal['confidence_level']}"
                f" | Gap: "
                f"{temporal['confidence_gap']:.4f}"
            )
        )

        self.raw_label.config(
            text=(
                f"Raw fusion: "
                f"{raw_state} "
                f"("
                f"{raw_confidence * 100.0:.2f}%"
                f")"
            )
        )

        self.temporal_label.config(
            text=(
                f"Temporal samples: "
                f"{temporal['temporal_samples']}/"
                f"{temporal['temporal_window']}"
            )
        )

        # ----------------------------------------------------
        # Full probability table
        # ----------------------------------------------------

        for label in LABELS:

            self.raw_probability_labels[
                label
            ].config(
                text=(
                    f"{raw_probabilities[label] * 100.0:.2f}%"
                )
            )

            self.temporal_probability_labels[
                label
            ].config(
                text=(
                    f"{temporal['probabilities'][label] * 100.0:.2f}%"
                )
            )

        raw_validation = (
            validate_probability_distribution(
                raw_probabilities,
                labels=LABELS,
                tolerance=(
                    PROBABILITY_SUM_TOLERANCE
                ),
            )
        )

        temporal_validation = (
            validate_probability_distribution(
                temporal[
                    "probabilities"
                ],
                labels=LABELS,
                tolerance=(
                    PROBABILITY_SUM_TOLERANCE
                ),
            )
        )

        raw_sum = (
            raw_validation[
                "probability_sum"
            ]
        )

        temporal_sum = (
            temporal_validation[
                "probability_sum"
            ]
        )

        self.probability_sum_label.config(
            text=(
                f"Raw sum: "
                f"{raw_sum:.6f}"
                f"    "
                f"Temporal sum: "
                f"{temporal_sum:.6f}"
            )
        )

        runtime_pass = (
            raw_validation[
                "valid"
            ]
            and
            temporal_validation[
                "valid"
            ]
            and
            temporal[
                "current_state"
            ]
            in LABELS
        )

        full_window = bool(
            temporal[
                "temporal_window_full"
            ]
        )

        self.validation_label.config(
            text=(
                "Runtime validation: "
                f"{'PASS' if runtime_pass else 'CHECK'}"
                f" | Probabilities valid"
                f" | Temporal window: "
                f"{'FULL' if full_window else 'WARMING UP'}"
                "\nBehavioural accuracy requires "
                "labelled repeated trials."
            ),
            fg=(
                GREEN
                if runtime_pass
                else YELLOW
            ),
        )

        # ----------------------------------------------------
        # Optional labelled-condition diagnostic
        # ----------------------------------------------------

        expected = (
            self.expected_state_var.get()
        )

        temporal_generation = int(
            temporal[
                "generation"
            ]
        )

        if (
            full_window
            and
            expected in LABELS
            and
            self.last_evaluated_generation
            != temporal_generation
        ):

            self.live_labelled_trials += 1

            if (
                temporal[
                    "current_state"
                ]
                == expected
            ):

                self.live_labelled_matches += 1

            self.last_evaluated_generation = (
                temporal_generation
            )

        self.update_live_test_label()

        # ----------------------------------------------------
        # Technical status
        # ----------------------------------------------------

        calibration = (
            raw_result.get(
                "image_calibration"
            )
            or {}
        )

        calibration_state = (
            calibration.get(
                "current_state"
            )
        )

        audio_condition = (
            (
                self.audio_diagnostics
                or {}
            ).get(
                "condition",
                "unknown",
            )
        )

        self.status_label.config(
            text=(
                "Prediction successful"
                f" | Raw={raw_state}"
                f" | Temporal="
                f"{temporal['current_state']}"
                f" | Samples="
                f"{temporal['temporal_samples']}/"
                f"{temporal['temporal_window']}"
                f" | Generation="
                f"{temporal['generation']}"
                f" | Runtime="
                f"{runtime:.2f}s"
                f" | Audio="
                f"{audio_condition}"
                f" | Image calibration="
                f"{calibration_state or 'not used'}"
            )
        )

    def update_live_test_label(
        self,
    ) -> None:

        if (
            self.live_labelled_trials
            <= 0
        ):

            self.live_test_label.config(
                text=(
                    "Live labelled checks: 0"
                )
            )

            return

        rate = (
            self.live_labelled_matches
            / self.live_labelled_trials
            * 100.0
        )

        self.live_test_label.config(
            text=(
                "Live labelled checks: "
                f"{self.live_labelled_matches}/"
                f"{self.live_labelled_trials}"
                f" ({rate:.1f}%)"
            )
        )

    # ========================================================
    # Prediction errors
    # ========================================================

    def apply_prediction_error(
        self,
        generation: int,
        error: Exception,
    ) -> None:

        if not (
            self.temporal_fusion
            .is_generation_current(
                generation
            )
        ):

            return

        self.show_prediction_error(
            error
        )

    def show_prediction_error(
        self,
        error: Exception,
    ) -> None:

        self.result_label.config(
            text="Prediction error",
            fg=RED,
        )

        self.raw_label.config(
            text=(
                f"{type(error).__name__}: "
                f"{error}"
            )
        )

        self.status_label.config(
            text=(
                "Prediction failed: "
                f"{type(error).__name__}: "
                f"{error}"
            )
        )

    def finish_prediction(
        self,
    ) -> None:

        self.fusion_prediction_busy = (
            False
        )

    # ========================================================
    # Reset
    # ========================================================

    def clear_prediction_display(
        self,
    ) -> None:

        self.result_label.config(
            text="Waiting for inputs",
            fg=GREEN,
        )

        self.confidence_label.config(
            text="Confidence: —"
        )

        self.raw_label.config(
            text="Raw fusion: —"
        )

        self.temporal_label.config(
            text=(
                "Temporal samples: "
                f"0/"
                f"{TEMPORAL_PROBABILITY_WINDOW}"
            )
        )

        for label in LABELS:

            self.raw_probability_labels[
                label
            ].config(
                text="—"
            )

            self.temporal_probability_labels[
                label
            ].config(
                text="—"
            )

        self.probability_sum_label.config(
            text=(
                "Raw sum: —    "
                "Temporal sum: —"
            )
        )

        self.validation_label.config(
            text=(
                "Runtime validation: waiting.\n"
                "Behavioural accuracy requires "
                "labelled repeated trials."
            ),
            fg=MUTED,
        )

    def reset_temporal_history(
        self,
        silent: bool = False,
    ) -> None:

        new_generation = (
            self.temporal_fusion.reset()
        )

        self.last_evaluated_generation = (
            -1
        )

        self.clear_prediction_display()

        if not silent:

            self.status_label.config(
                text=(
                    "Temporal probability "
                    "history reset"
                    f" | Generation="
                    f"{new_generation}."
                )
            )

    def reset(
        self,
    ) -> None:

        # Canonical temporal reset.
        self.reset_temporal_history(
            silent=True
        )

        self.stop_visual_stream(
            update_status=False
        )

        # Visual
        self.image_path = None
        self.image_source_name = None

        self.image_source_type = (
            "none"
        )

        self.current_frame = None
        self.preview_image = None

        self.preview_label.config(
            image=""
        )

        self.visual_source_label.config(
            text=(
                "Visual input not loaded."
            ),
            fg=MISSING,
        )

        # Text + keystrokes
        self.text_box.delete(
            "1.0",
            tk.END,
        )

        self.keystroke_events.clear()
        self.active_keys.clear()

        # Audio
        self.audio_path = None
        self.audio_source_name = None
        self.audio_diagnostics = None

        self.audio_label.config(
            text="Audio not loaded.",
            fg=MISSING,
        )

        self.audio_diagnostic_label.config(
            text="Audio condition: —"
        )

        # Labelled test state
        self.expected_state_var.set(
            "unlabelled"
        )

        self.live_labelled_trials = 0
        self.live_labelled_matches = 0

        self.last_evaluated_generation = (
            -1
        )

        self.update_live_test_label()

        self.update_readiness()

        self.status_label.config(
            text="Full session reset."
        )

    # ========================================================
    # Shutdown
    # ========================================================

    def on_close(
        self,
    ) -> None:

        self.closed = True

        # Invalidate any pending prediction.
        self.temporal_fusion.reset()

        try:

            self.stop_visual_stream(
                update_status=False
            )

        except Exception:

            pass

        try:

            self.temp_directory.cleanup()

        except Exception:

            pass

        self.root.destroy()


# ============================================================
# Entry point
# ============================================================

def main() -> None:

    root = tk.Tk()

    FusionDemoApp(
        root
    )

    root.mainloop()


if __name__ == "__main__":
    main()
