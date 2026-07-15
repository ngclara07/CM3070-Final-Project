# train_fusion_demo_pipeline.py

from pathlib import Path
import json
import joblib

import pandas as pd

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import StratifiedKFold, cross_val_score

from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC


KEYSTROKE_PATH = Path("data/processed/master_sessions_raw.csv")
TEXT_PATH = Path("data/processed/text_features.csv")
AUDIO_PATH = Path("data/processed/audio_features.csv")
IMAGE_PATH = Path("data/processed/image_features.csv")

OUTPUT_DIR = Path("models/fusion_demo")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

LABEL_COL = "label"
SESSION_COL = "session_id"

KEYSTROKE_NON_FEATURE_COLUMNS = {
    "session_id", "label", "created_at",
    "keydown_count_json", "event_count", "expected_event_count",
    "validation_passed", "validation_message",
    "text_path", "keystroke_path", "audio_path", "image_path",
    "text_exists", "audio_exists", "image_exists",
    "text", "problems", "is_clean",
}


class LabelEncodedClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, classifier):
        self.classifier = classifier
        self.label_encoder = LabelEncoder()

    def fit(self, X, y):
        y_encoded = self.label_encoder.fit_transform(y)
        self.classifier.fit(X, y_encoded)
        self.classes_ = self.label_encoder.classes_
        return self

    def predict(self, X):
        y_pred = self.classifier.predict(X)
        return self.label_encoder.inverse_transform(y_pred.astype(int))

    def predict_proba(self, X):
        return self.classifier.predict_proba(X)


def load_keystroke_features():
    if not KEYSTROKE_PATH.exists():
        raise FileNotFoundError(f"Missing keystroke file: {KEYSTROKE_PATH}")

    df = pd.read_csv(KEYSTROKE_PATH)

    if "is_clean" in df.columns:
        before_count = len(df)
        df = df[df["is_clean"] == True].copy()
        after_count = len(df)
        print(f"Clean-session filtering applied: {before_count} → {after_count}")

    feature_cols = [
        col for col in df.columns
        if col not in KEYSTROKE_NON_FEATURE_COLUMNS
        and pd.api.types.is_numeric_dtype(df[col])
    ]

    if not feature_cols:
        raise ValueError("No numeric keystroke features found.")

    return df[[SESSION_COL, LABEL_COL] + feature_cols].copy()


def load_modality_features(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"Missing feature file: {path}")

    df = pd.read_csv(path)

    if SESSION_COL not in df.columns or LABEL_COL not in df.columns:
        raise ValueError(f"{path} must contain session_id and label columns.")

    return df.copy()


def build_fusion_dataset():
    key_df = load_keystroke_features()
    text_df = load_modality_features(TEXT_PATH)
    audio_df = load_modality_features(AUDIO_PATH)
    image_df = load_modality_features(IMAGE_PATH)

    fusion_df = key_df.merge(text_df, on=[SESSION_COL, LABEL_COL], how="inner")
    fusion_df = fusion_df.merge(audio_df, on=[SESSION_COL, LABEL_COL], how="inner")
    fusion_df = fusion_df.merge(image_df, on=[SESSION_COL, LABEL_COL], how="inner")

    if fusion_df.empty:
        raise ValueError("Fusion dataframe is empty after merging modalities.")

    feature_cols = [
        col for col in fusion_df.columns
        if col not in {SESSION_COL, LABEL_COL}
        and pd.api.types.is_numeric_dtype(fusion_df[col])
    ]

    if not feature_cols:
        raise ValueError("No numeric fusion features found.")

    X = fusion_df[feature_cols].copy()
    y = fusion_df[LABEL_COL].copy()

    return fusion_df, X, y, feature_cols


def build_models():
    models = {
        "random_forest": Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", RandomForestClassifier(
                n_estimators=500,
                class_weight="balanced",
                random_state=42,
            )),
        ]),

        "svm_rbf": Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", SVC(
                kernel="rbf",
                class_weight="balanced",
                probability=True,
                random_state=42,
            )),
        ]),
    }

    try:
        from xgboost import XGBClassifier

        models["xgboost"] = Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", LabelEncodedClassifier(
                XGBClassifier(
                    n_estimators=500,
                    learning_rate=0.03,
                    max_depth=4,
                    subsample=0.9,
                    colsample_bytree=0.9,
                    objective="multi:softprob",
                    eval_metric="mlogloss",
                    random_state=42,
                )
            )),
        ])

    except ImportError:
        print("XGBoost not installed. Skipping xgboost.")

    try:
        from lightgbm import LGBMClassifier

        models["lightgbm"] = Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", LGBMClassifier(
                n_estimators=500,
                learning_rate=0.03,
                class_weight="balanced",
                random_state=42,
                verbose=-1,
            )),
        ])

    except ImportError:
        print("LightGBM not installed. Skipping lightgbm.")

    try:
        from catboost import CatBoostClassifier

        models["catboost"] = Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", CatBoostClassifier(
                iterations=500,
                learning_rate=0.03,
                depth=5,
                loss_function="MultiClass",
                auto_class_weights="Balanced",
                random_seed=42,
                verbose=False,
            )),
        ])

    except ImportError:
        print("CatBoost not installed. Skipping catboost.")

    return models


def select_best_model(models, X, y):
    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=42,
    )

    rows = []

    for name, pipeline in models.items():
        print(f"Cross-validating: {name}")

        accuracy_scores = cross_val_score(
            pipeline,
            X,
            y,
            cv=cv,
            scoring="accuracy",
        )

        f1_scores = cross_val_score(
            pipeline,
            X,
            y,
            cv=cv,
            scoring="f1_macro",
        )

        rows.append({
            "model": name,
            "cv_accuracy_mean": float(accuracy_scores.mean()),
            "cv_accuracy_std": float(accuracy_scores.std()),
            "cv_macro_f1_mean": float(f1_scores.mean()),
            "cv_macro_f1_std": float(f1_scores.std()),
        })

    results_df = pd.DataFrame(rows).sort_values(
        by="cv_macro_f1_mean",
        ascending=False,
    )

    best_model_name = results_df.iloc[0]["model"]
    best_pipeline = models[best_model_name]

    return best_model_name, best_pipeline, results_df


def main():
    fusion_df, X, y, feature_cols = build_fusion_dataset()

    print("\nFusion dataset loaded successfully.")
    print(f"Samples: {len(fusion_df)}")
    print(f"Features: {len(feature_cols)}")

    print("\nLabel distribution:")
    print(y.value_counts().sort_index())

    models = build_models()

    if not models:
        raise RuntimeError("No candidate models available.")

    best_model_name, best_pipeline, cv_results = select_best_model(
        models=models,
        X=X,
        y=y,
    )

    cv_results.to_csv(
        OUTPUT_DIR / "model_selection_results.csv",
        index=False,
    )

    print("\nFusion model selection results:")
    print(cv_results)

    print(f"\nBest fusion model: {best_model_name}")

    best_pipeline.fit(X, y)

    joblib.dump(best_pipeline, OUTPUT_DIR / "fusion_pipeline.joblib")

    with open(OUTPUT_DIR / "feature_columns.json", "w", encoding="utf-8") as f:
        json.dump(feature_cols, f, indent=4)

    metadata = {
        "num_samples": int(len(fusion_df)),
        "num_features": int(len(feature_cols)),
        "modalities": ["keystroke", "text", "audio", "image"],
        "candidate_models": list(models.keys()),
        "selected_model": best_model_name,
        "selection_metric": "5-fold CV macro-F1",
        "model_artifact": str(OUTPUT_DIR / "fusion_pipeline.joblib"),
    }

    with open(OUTPUT_DIR / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=4)

    print("\nFusion demo pipeline saved successfully.")
    print(f"Selected model: {best_model_name}")
    print(f"Model path: {OUTPUT_DIR / 'fusion_pipeline.joblib'}")
    print(f"Feature schema: {OUTPUT_DIR / 'feature_columns.json'}")
    print(f"Metadata: {OUTPUT_DIR / 'metadata.json'}")


if __name__ == "__main__":
    main()
