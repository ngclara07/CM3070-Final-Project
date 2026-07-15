# train_keystroke_demo_pipeline.py

from pathlib import Path
import json
import joblib

import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_val_score

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB


DATA_PATH = Path("data/processed/master_sessions_raw.csv")
OUTPUT_DIR = Path("models/keystroke_demo")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

LABEL_COL = "label"

NON_FEATURE_COLUMNS = {
    "session_id", "label", "created_at",
    "text_path", "keystroke_path", "audio_path", "image_path",
    "text", "validation_message", "problems",
    "validation_passed", "text_exists", "audio_exists", "image_exists",
    "is_clean", "keydown_count_json", "event_count", "expected_event_count",
}


def load_dataset():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)

    if "is_clean" in df.columns:
        before_count = len(df)
        df = df[df["is_clean"] == True].copy()
        after_count = len(df)
        print(f"Clean-session filtering applied: {before_count} → {after_count}")

    if LABEL_COL not in df.columns:
        raise ValueError(f"Missing label column: {LABEL_COL}")

    feature_cols = [
        col for col in df.columns
        if col not in NON_FEATURE_COLUMNS
        and pd.api.types.is_numeric_dtype(df[col])
    ]

    if not feature_cols:
        raise ValueError("No numeric keystroke feature columns found.")

    X = df[feature_cols].copy()
    y = df[LABEL_COL].copy()

    return df, X, y, feature_cols


def build_models():
    return {
        "logistic_regression": Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(
                max_iter=3000,
                class_weight="balanced",
                random_state=42,
            )),
        ]),

        "decision_tree": Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", DecisionTreeClassifier(
                class_weight="balanced",
                random_state=42,
            )),
        ]),

        "random_forest": Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", RandomForestClassifier(
                n_estimators=300,
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

        "knn": Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", KNeighborsClassifier(
                n_neighbors=5,
                weights="distance",
            )),
        ]),

        "gaussian_naive_bayes": Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", GaussianNB()),
        ]),
    }


def select_best_model(models, X, y):
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    rows = []

    for name, pipeline in models.items():
        print(f"Cross-validating: {name}")

        acc_scores = cross_val_score(
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
            "cv_accuracy_mean": float(acc_scores.mean()),
            "cv_accuracy_std": float(acc_scores.std()),
            "cv_macro_f1_mean": float(f1_scores.mean()),
            "cv_macro_f1_std": float(f1_scores.std()),
        })

    results_df = pd.DataFrame(rows).sort_values(
        by="cv_macro_f1_mean",
        ascending=False,
    )

    best_name = results_df.iloc[0]["model"]
    best_pipeline = models[best_name]

    return best_name, best_pipeline, results_df


def main():
    df, X, y, feature_cols = load_dataset()

    print("\nDataset loaded successfully.")
    print(f"Samples used: {len(df)}")
    print(f"Features: {len(feature_cols)}")

    print("\nLabel distribution:")
    print(y.value_counts().sort_index())

    models = build_models()

    best_model_name, best_pipeline, cv_results = select_best_model(models, X, y)

    cv_results.to_csv(
        OUTPUT_DIR / "model_selection_results.csv",
        index=False,
    )

    print("\nModel selection results:")
    print(cv_results)

    print(f"\nBest keystroke model: {best_model_name}")

    best_pipeline.fit(X, y)

    joblib.dump(best_pipeline, OUTPUT_DIR / "keystroke_pipeline.joblib")

    with open(OUTPUT_DIR / "feature_columns.json", "w", encoding="utf-8") as f:
        json.dump(feature_cols, f, indent=4)

    metadata = {
        "data_path": str(DATA_PATH),
        "label_column": LABEL_COL,
        "num_samples": int(len(df)),
        "num_features": int(len(feature_cols)),
        "feature_columns": feature_cols,
        "candidate_models": list(models.keys()),
        "selected_model": best_model_name,
        "selection_metric": "5-fold CV macro-F1",
        "clean_sessions_only": True,
    }

    with open(OUTPUT_DIR / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=4)

    print("\nKeystroke demo pipeline saved successfully.")
    print(f"Selected model: {best_model_name}")
    print(f"Model path: {OUTPUT_DIR / 'keystroke_pipeline.joblib'}")
    print(f"Feature schema: {OUTPUT_DIR / 'feature_columns.json'}")
    print(f"Metadata: {OUTPUT_DIR / 'metadata.json'}")


if __name__ == "__main__":
    main()
