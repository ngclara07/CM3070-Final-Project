# extract_text_features.py
# MPNet-based text feature extraction

from pathlib import Path

import pandas as pd
from sentence_transformers import SentenceTransformer


DATA_PATH = Path("data/processed/master_sessions_clean_scaled.csv")
OUTPUT_PATH = Path("data/processed/text_features.csv")

SESSION_COL = "session_id"
LABEL_COL = "label"
TEXT_COL = "text"

MODEL_PATH = "models/all-mpnet-base-v2"


def load_clean_sessions(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Input dataset not found: {path}")

    df = pd.read_csv(path)

    required_cols = {
        SESSION_COL,
        LABEL_COL,
        TEXT_COL,
    }

    missing = required_cols - set(df.columns)

    if missing:
        raise ValueError(
            f"Missing required columns: {missing}"
        )

    df = df[
        [
            SESSION_COL,
            LABEL_COL,
            TEXT_COL,
        ]
    ].copy()

    df[TEXT_COL] = (
        df[TEXT_COL]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    empty_rows = df[df[TEXT_COL] == ""]

    if len(empty_rows) > 0:
        raise ValueError(
            f"{len(empty_rows)} rows contain empty text."
        )

    if df[SESSION_COL].duplicated().any():
        raise ValueError(
            "Duplicate session_id values found."
        )

    return df


def load_model() -> SentenceTransformer:
    model_path = Path(MODEL_PATH)

    if not model_path.exists():
        raise FileNotFoundError(
            f"Model not found: {model_path}\n"
            "Run download_mpnet_model.py first."
        )

    print(f"Loading MPNet model: {model_path}")

    model = SentenceTransformer(str(model_path))

    return model


def extract_embeddings(
    texts: list[str],
    model: SentenceTransformer,
):
    embeddings = model.encode(
        texts,
        batch_size=32,
        show_progress_bar=True,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )

    return embeddings


def build_feature_dataframe(
    df: pd.DataFrame,
    embeddings,
) -> pd.DataFrame:

    feature_cols = [
        f"text_mpnet_emb_{i}"
        for i in range(embeddings.shape[1])
    ]

    embedding_df = pd.DataFrame(
        embeddings,
        columns=feature_cols,
    )

    output_df = pd.concat(
        [
            df[
                [
                    SESSION_COL,
                    LABEL_COL,
                ]
            ].reset_index(drop=True),

            embedding_df.reset_index(drop=True),
        ],
        axis=1,
    )

    return output_df


def main():
    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    df = load_clean_sessions(DATA_PATH)

    print(f"Loaded {len(df)} text samples")

    model = load_model()

    texts = df[TEXT_COL].tolist()

    print(
        f"Extracting MPNet embeddings "
        f"for {len(texts)} samples..."
    )

    embeddings = extract_embeddings(
        texts=texts,
        model=model,
    )

    output_df = build_feature_dataframe(
        df=df,
        embeddings=embeddings,
    )

    output_df.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    print("\nMPNet text feature extraction complete.")
    print(f"Samples: {len(output_df)}")
    print(
        f"Embedding dimensions: "
        f"{output_df.shape[1] - 2}"
    )
    print(f"Saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
