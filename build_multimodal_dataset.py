# build_multimodal_dataset.py

from pathlib import Path
import pandas as pd


KEYSTROKE_PATH = Path("data/processed/master_sessions_clean_scaled.csv")
TEXT_FEATURES_PATH = Path("data/processed/text_features.csv")
AUDIO_FEATURES_PATH = Path("data/processed/audio_features.csv")
IMAGE_FEATURES_PATH = Path("data/processed/image_features.csv")

OUTPUT_PATH = Path("data/processed/multimodal_features.csv")

SESSION_COL = "session_id"
LABEL_COL = "label"

NON_KEYSTROKE_FEATURE_COLUMNS = {
    "created_at",
    "text_path",
    "keystroke_path",
    "audio_path",
    "image_path",
    "text",
    "validation_message",
    "problems",
    "validation_passed",
    "text_exists",
    "audio_exists",
    "image_exists",
    "is_clean",
}


def load_csv(path: Path, name: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"{name} file not found: {path}")

    df = pd.read_csv(path)

    if SESSION_COL not in df.columns:
        raise ValueError(f"{name} is missing column: {SESSION_COL}")

    if LABEL_COL not in df.columns:
        raise ValueError(f"{name} is missing column: {LABEL_COL}")

    if df[SESSION_COL].duplicated().any():
        duplicates = df[df[SESSION_COL].duplicated()][SESSION_COL].tolist()
        raise ValueError(f"{name} contains duplicate session IDs: {duplicates[:10]}")

    return df


def prepare_keystroke_features(df: pd.DataFrame) -> pd.DataFrame:
    keep_cols = [
        col for col in df.columns
        if col not in NON_KEYSTROKE_FEATURE_COLUMNS
    ]

    df = df[keep_cols].copy()

    technical_count_cols = {
        "keydown_count_json",
        "event_count",
        "expected_event_count",
    }

    df = df.drop(
        columns=[col for col in technical_count_cols if col in df.columns],
        errors="ignore",
    )

    return df


def drop_duplicate_label(df: pd.DataFrame, prefix_name: str) -> pd.DataFrame:
    if LABEL_COL in df.columns:
        df = df.drop(columns=[LABEL_COL])

    if df.columns.duplicated().any():
        duplicated = df.columns[df.columns.duplicated()].tolist()
        raise ValueError(f"{prefix_name} has duplicated columns: {duplicated}")

    return df


def main():
    keystroke_df = load_csv(KEYSTROKE_PATH, "Keystroke dataset")
    text_df = load_csv(TEXT_FEATURES_PATH, "Text features")
    audio_df = load_csv(AUDIO_FEATURES_PATH, "Audio features")
    image_df = load_csv(IMAGE_FEATURES_PATH, "Image features")

    keystroke_df = prepare_keystroke_features(keystroke_df)

    text_df = drop_duplicate_label(text_df, "Text features")
    audio_df = drop_duplicate_label(audio_df, "Audio features")
    image_df = drop_duplicate_label(image_df, "Image features")

    merged_df = keystroke_df.merge(text_df, on=SESSION_COL, how="inner")
    merged_df = merged_df.merge(audio_df, on=SESSION_COL, how="inner")
    merged_df = merged_df.merge(image_df, on=SESSION_COL, how="inner")

    expected_sessions = len(keystroke_df)

    if len(merged_df) != expected_sessions:
        raise ValueError(
            f"Merge lost samples. Expected {expected_sessions}, got {len(merged_df)}."
        )

    if merged_df[SESSION_COL].duplicated().any():
        raise ValueError("Merged dataset contains duplicate session IDs.")

    if merged_df[LABEL_COL].isna().any():
        raise ValueError("Merged dataset contains missing labels.")

    numeric_feature_cols = [
        col for col in merged_df.columns
        if col not in {SESSION_COL, LABEL_COL}
        and pd.api.types.is_numeric_dtype(merged_df[col])
    ]

    non_numeric = [
        col for col in merged_df.columns
        if col not in {SESSION_COL, LABEL_COL}
        and not pd.api.types.is_numeric_dtype(merged_df[col])
    ]

    if non_numeric:
        raise ValueError(f"Unexpected non-numeric feature columns: {non_numeric}")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    merged_df.to_csv(OUTPUT_PATH, index=False)

    print("Multimodal dataset created successfully.")
    print(f"Samples: {len(merged_df)}")
    print(f"Total columns: {merged_df.shape[1]}")
    print(f"Numeric feature columns: {len(numeric_feature_cols)}")
    print("\nLabel distribution:")
    print(merged_df[LABEL_COL].value_counts().sort_index())
    print(f"\nSaved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
