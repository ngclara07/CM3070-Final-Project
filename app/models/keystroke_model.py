# === app/models/keystroke_model.py ===

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd


BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODEL_DIR = os.path.join(BASE_DIR, "model_artifacts")

MODEL_PATH = os.path.join(MODEL_DIR, "keystroke_model.joblib")
META_PATH = os.path.join(MODEL_DIR, "keystroke_model_meta.json")


FEATURES = [
    "typing_speed",
    "hold_mean",
    "hold_std",
    "delay_mean",
    "delay_std",
    "delay_cv",
    "pause_ratio_500",
    "pause_ratio_1000",
    "pause_ratio_2000",
    "mental_block_ratio_5000",
    "correction_ratio",
    "repeated_key_ratio",
    "rapid_burst_ratio",
    "burstiness_proxy",
    "fits_starts_index",
    "rhythm_consistency",
    "error_rate_proxy",
]


def load_keystroke_artifacts():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Keystroke model not found: {MODEL_PATH}. "
            "Run train_keystroke_model.py first."
        )

    if not os.path.exists(META_PATH):
        raise FileNotFoundError(
            f"Keystroke metadata not found: {META_PATH}. "
            "Run train_keystroke_model.py first."
        )

    model = joblib.load(MODEL_PATH)

    with open(META_PATH, "r", encoding="utf-8") as f:
        meta = json.load(f)

    return model, meta


def _empty_features() -> Dict[str, float]:
    return {feature: 0.0 for feature in FEATURES}


def _to_milliseconds(values: List[float]) -> np.ndarray:
    arr = np.asarray(values, dtype=float)

    if len(arr) < 2:
        return arr

    diffs = np.diff(np.sort(arr))
    diffs = diffs[diffs >= 0]

    if len(diffs) == 0:
        return arr

    median_diff = float(np.median(diffs))

    if median_diff < 10:
        return arr * 1000.0

    return arr


def _mean(arr) -> float:
    arr = np.asarray(arr, dtype=float)
    return float(np.mean(arr)) if len(arr) else 0.0


def _std(arr) -> float:
    arr = np.asarray(arr, dtype=float)
    return float(np.std(arr)) if len(arr) else 0.0


def _cv(arr) -> float:
    arr = np.asarray(arr, dtype=float)

    if len(arr) == 0:
        return 0.0

    mean = float(np.mean(arr))

    if mean <= 0:
        return 0.0

    return float(np.std(arr) / mean)


def _iqr(arr) -> float:
    arr = np.asarray(arr, dtype=float)

    if len(arr) < 2:
        return 0.0

    return float(np.quantile(arr, 0.75) - np.quantile(arr, 0.25))


def _fits_starts(arr) -> float:
    arr = np.asarray(arr, dtype=float)

    if len(arr) < 3:
        return 0.0

    acceleration = np.diff(arr)
    return float(np.std(acceleration))


def _is_correction_key(key: Any) -> bool:
    k = str(key).lower()

    return k in {
        "backspace",
        "delete",
        "key.backspace",
        "key.delete",
        "8",
        "46",
    }


def _repeated_key_ratio(downs: List[Dict[str, Any]], delays: np.ndarray) -> float:
    if len(downs) < 2:
        return 0.0

    repeated = 0
    comparisons = 0

    for i in range(1, len(downs)):
        current_key = str(downs[i].get("key", "")).lower()
        previous_key = str(downs[i - 1].get("key", "")).lower()

        delay = delays[i - 1] if i - 1 < len(delays) else 9999

        comparisons += 1

        if current_key == previous_key and delay < 300:
            repeated += 1

    return float(repeated / max(comparisons, 1))


def extract_keystroke_features(events: List[Dict[str, Any]]) -> Dict[str, float]:
    if not events:
        return _empty_features()

    valid_events = [
        e for e in events
        if "type" in e and "ts" in e
    ]

    if not valid_events:
        return _empty_features()

    raw_ts = [float(e["ts"]) for e in valid_events]
    ms_ts = _to_milliseconds(raw_ts)

    converted = []

    for e, ts_ms in zip(valid_events, ms_ts):
        row = dict(e)
        row["ts_ms"] = float(ts_ms)
        converted.append(row)

    downs = [e for e in converted if e.get("type") == "down"]
    down_times = [float(e["ts_ms"]) for e in downs]

    delays = np.diff(down_times) if len(down_times) > 1 else np.array([])

    hold_times: List[float] = []
    active: Dict[str, List[float]] = {}

    for e in converted:
        key = str(e.get("key", ""))
        ts = float(e.get("ts_ms", 0.0))

        if e.get("type") == "down":
            active.setdefault(key, []).append(ts)

        elif e.get("type") == "up":
            if key in active and active[key]:
                down_ts = active[key].pop(0)
                hold = ts - down_ts

                if hold >= 0:
                    hold_times.append(hold)

    start = min(float(e["ts_ms"]) for e in converted)
    end = max(float(e["ts_ms"]) for e in converted)

    duration_ms = max(end - start, 1e-6)

    total_keys = max(len(downs), 1)

    correction_count = sum(
        1 for e in downs if _is_correction_key(e.get("key"))
    )

    correction_ratio = correction_count / total_keys
    repeated_key_ratio = _repeated_key_ratio(downs, delays)

    error_rate_proxy = min(1.0, correction_ratio + repeated_key_ratio)

    rapid_burst_ratio = float(np.mean(delays < 120)) if len(delays) else 0.0
    mental_block_ratio = float(np.mean(delays > 5000)) if len(delays) else 0.0

    delay_cv = _cv(delays)
    rhythm_consistency = 1.0 / (1.0 + delay_cv) if delay_cv > 0 else 1.0

    features = {
        "typing_speed": float(len(downs) / duration_ms),

        "hold_mean": _mean(hold_times),
        "hold_std": _std(hold_times),

        "delay_mean": _mean(delays),
        "delay_std": _std(delays),
        "delay_cv": delay_cv,

        "pause_ratio_500": float(np.mean(delays > 500)) if len(delays) else 0.0,
        "pause_ratio_1000": float(np.mean(delays > 1000)) if len(delays) else 0.0,
        "pause_ratio_2000": float(np.mean(delays > 2000)) if len(delays) else 0.0,
        "mental_block_ratio_5000": mental_block_ratio,

        "correction_ratio": float(correction_ratio),
        "repeated_key_ratio": float(repeated_key_ratio),
        "rapid_burst_ratio": rapid_burst_ratio,
        "burstiness_proxy": _iqr(delays),
        "fits_starts_index": _fits_starts(delays),
        "rhythm_consistency": float(rhythm_consistency),
        "error_rate_proxy": float(error_rate_proxy),
    }

    return features


def predict_keystroke_behaviour(
    events: List[Dict[str, Any]]
) -> Tuple[str, Dict[str, float], Dict[str, float]]:

    model, meta = load_keystroke_artifacts()

    model_features = meta.get("features", FEATURES)
    model_classes = meta.get("classes", ["focused", "distracted", "fatigued", "overloaded"])

    features = extract_keystroke_features(events)

    X = pd.DataFrame([features])
    X = X.reindex(columns=model_features, fill_value=0.0)

    pred = model.predict(X)[0]

    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X)[0]

        if hasattr(model, "named_steps") and "model" in model.named_steps:
            classes = model.named_steps["model"].classes_
        else:
            classes = getattr(model, "classes_", model_classes)

        score_map = {
            str(label): float(prob)
            for label, prob in zip(classes, probs)
        }

    else:
        score_map = {str(c): 0.0 for c in model_classes}
        score_map[str(pred)] = 1.0

    return str(pred), score_map, features
