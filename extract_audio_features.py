# extract_audio_features.py
# Hybrid audio feature extraction:
# 1. WavLM pretrained speech embeddings
# 2. Librosa acoustic/prosodic behavioural features
# 3. Optional YAMNet environmental audio embeddings

from pathlib import Path
import warnings
import time

import numpy as np
import pandas as pd
import torch
import librosa
from transformers import Wav2Vec2FeatureExtractor, WavLMModel


torch.set_num_threads(2)

DATA_PATH = Path("data/processed/master_sessions_clean_scaled.csv")
OUTPUT_PATH = Path("data/processed/audio_features.csv")

SESSION_COL = "session_id"
LABEL_COL = "label"
AUDIO_PATH_COL = "audio_path"

WAVLM_MODEL_PATH = "models/wavlm-base-plus"
TARGET_SR = 16000
MAX_AUDIO_SECONDS = 20

USE_YAMNET = False
YAMNET_MODEL_PATH = "models/yamnet"

MAX_ROWS = None  # set to 5 for quick testing


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_audio(audio_path: Path) -> tuple[np.ndarray, int]:
    waveform, sr = librosa.load(audio_path, sr=TARGET_SR, mono=True)
    waveform = waveform.astype(np.float32)

    if len(waveform) == 0:
        raise ValueError(f"Empty audio file: {audio_path}")

    max_samples = TARGET_SR * MAX_AUDIO_SECONDS
    if len(waveform) > max_samples:
        waveform = waveform[:max_samples]

    return waveform, sr


def build_wavlm(device: torch.device):
    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(WAVLM_MODEL_PATH)
    model = WavLMModel.from_pretrained(WAVLM_MODEL_PATH)
    model.to(device)
    model.eval()
    return model, feature_extractor


def extract_wavlm_features(
    waveform: np.ndarray,
    model: WavLMModel,
    feature_extractor: Wav2Vec2FeatureExtractor,
    device: torch.device,
) -> dict:
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


def extract_librosa_features(waveform: np.ndarray, sr: int) -> dict:
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
    }

    if len(pitch_values) > 0:
        features.update({
            "audio_pitch_mean": float(np.mean(pitch_values)),
            "audio_pitch_std": float(np.std(pitch_values)),
            "audio_pitch_min": float(np.min(pitch_values)),
            "audio_pitch_max": float(np.max(pitch_values)),
        })
    else:
        features.update({
            "audio_pitch_mean": 0.0,
            "audio_pitch_std": 0.0,
            "audio_pitch_min": 0.0,
            "audio_pitch_max": 0.0,
        })

    for i in range(13):
        features[f"audio_mfcc_{i}_mean"] = float(np.mean(mfcc[i]))
        features[f"audio_mfcc_{i}_std"] = float(np.std(mfcc[i]))

    return features


def build_yamnet():
    if not USE_YAMNET:
        return None

    try:
        import tensorflow_hub as hub
        return hub.load(str(YAMNET_MODEL_PATH))
    except Exception as error:
        raise RuntimeError(
            "YAMNet could not be loaded. Set USE_YAMNET = False, "
            "or install tensorflow/tensorflow_hub and check the local model path."
        ) from error


def extract_yamnet_features(waveform: np.ndarray, yamnet_model) -> dict:
    if yamnet_model is None:
        return {}

    import tensorflow as tf

    waveform_tf = tf.convert_to_tensor(waveform, dtype=tf.float32)
    scores, embeddings, _ = yamnet_model(waveform_tf)

    scores_np = scores.numpy()
    embeddings_np = embeddings.numpy()

    features = {
        "audio_yamnet_score_mean": float(np.mean(scores_np)),
        "audio_yamnet_score_std": float(np.std(scores_np)),
        "audio_yamnet_score_max": float(np.max(scores_np)),
    }

    for i, value in enumerate(embeddings_np.mean(axis=0)):
        features[f"audio_yamnet_emb_{i}_mean"] = float(value)

    for i, value in enumerate(embeddings_np.std(axis=0)):
        features[f"audio_yamnet_emb_{i}_std"] = float(value)

    return features


def clean_feature_values(features: dict) -> dict:
    cleaned = {}

    for key, value in features.items():
        try:
            value = float(value)
            cleaned[key] = value if np.isfinite(value) else 0.0
        except Exception:
            cleaned[key] = 0.0

    return cleaned


def main():
    warnings.filterwarnings("ignore")

    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)

    if MAX_ROWS is not None:
        df = df.head(MAX_ROWS).copy()

    required_cols = {SESSION_COL, LABEL_COL, AUDIO_PATH_COL}
    missing_cols = required_cols - set(df.columns)

    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    device = get_device()

    print(f"Using device: {device}")
    print(f"Using WavLM model: {WAVLM_MODEL_PATH}")
    print(f"Using YAMNet: {USE_YAMNET}")
    print(f"Target sample rate: {TARGET_SR}")
    print(f"Maximum audio duration: {MAX_AUDIO_SECONDS}s")
    print(f"Samples to process: {len(df)}")

    wavlm_model, wavlm_extractor = build_wavlm(device)
    yamnet_model = build_yamnet()

    rows = []
    start_all = time.time()

    for _, row in df.iterrows():
        audio_path = Path(row[AUDIO_PATH_COL])
        current = len(rows) + 1

        print(f"Processing {current}/{len(df)}: {audio_path}", flush=True)
        start_one = time.time()

        if not audio_path.exists():
            raise FileNotFoundError(f"Missing audio file: {audio_path}")

        waveform, sr = load_audio(audio_path)

        all_features = clean_feature_values({
            **extract_librosa_features(waveform, sr),
            **extract_wavlm_features(
                waveform=waveform,
                model=wavlm_model,
                feature_extractor=wavlm_extractor,
                device=device,
            ),
            **extract_yamnet_features(waveform, yamnet_model),
        })

        rows.append({
            SESSION_COL: row[SESSION_COL],
            LABEL_COL: row[LABEL_COL],
            **all_features,
        })

        print(
            f"Finished {current}/{len(df)} in {time.time() - start_one:.2f}s",
            flush=True,
        )

    output_df = pd.DataFrame(rows)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    output_df.to_csv(OUTPUT_PATH, index=False)

    print("\nHybrid audio feature extraction complete.")
    print(f"Samples: {len(output_df)}")
    print(f"Audio features: {output_df.shape[1] - 2}")
    print(f"Saved to: {OUTPUT_PATH}")
    print(f"Total time: {(time.time() - start_all) / 60:.2f} minutes")


if __name__ == "__main__":
    main()
