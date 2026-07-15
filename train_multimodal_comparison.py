# train_multimodal_comparison.py

from pathlib import Path
import json
import joblib
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate, cross_val_score
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC


warnings.filterwarnings("ignore", category=UserWarning)

DATA_PATH = Path("data/processed/multimodal_features.csv")
OUTPUT_DIR = Path("data/processed/multimodal_comparison_results")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SESSION_COL = "session_id"
LABEL_COL = "label"

CV_SPLITS = 5
RANDOM_STATE = 42
N_ESTIMATORS = 100


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


def load_dataset() -> pd.DataFrame:
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)

    required = {SESSION_COL, LABEL_COL}
    missing = required - set(df.columns)

    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    if df[SESSION_COL].duplicated().any():
        raise ValueError("Duplicate session_id values found.")

    return df


def get_feature_groups(df: pd.DataFrame) -> dict[str, list[str]]:
    all_feature_cols = [
        col for col in df.columns
        if col not in {SESSION_COL, LABEL_COL}
        and pd.api.types.is_numeric_dtype(df[col])
    ]

    text_cols = [col for col in all_feature_cols if col.startswith("text_")]
    audio_cols = [col for col in all_feature_cols if col.startswith("audio_")]
    image_cols = [col for col in all_feature_cols if col.startswith("image_")]

    pretrained_cols = set(text_cols + audio_cols + image_cols)

    keystroke_cols = [
        col for col in all_feature_cols
        if col not in pretrained_cols
    ]

    groups = {
        "keystroke_only": keystroke_cols,
        "text_only": text_cols,
        "audio_only": audio_cols,
        "image_only": image_cols,

        "keystroke_text": keystroke_cols + text_cols,
        "keystroke_audio": keystroke_cols + audio_cols,
        "keystroke_image": keystroke_cols + image_cols,
        "text_audio": text_cols + audio_cols,
        "text_image": text_cols + image_cols,
        "audio_image": audio_cols + image_cols,

        "keystroke_text_audio": keystroke_cols + text_cols + audio_cols,
        "keystroke_text_image": keystroke_cols + text_cols + image_cols,
        "keystroke_audio_image": keystroke_cols + audio_cols + image_cols,
        "text_audio_image": text_cols + audio_cols + image_cols,

        "multimodal_all": keystroke_cols + text_cols + audio_cols + image_cols,
    }

    for name, cols in groups.items():
        if not cols:
            raise ValueError(f"No features found for group: {name}")

    return groups


def build_models() -> dict:
    models = {
        "logistic_regression": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(
                max_iter=3000,
                class_weight="balanced",
                random_state=RANDOM_STATE,
            )),
        ]),

        "random_forest": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", RandomForestClassifier(
                n_estimators=N_ESTIMATORS,
                class_weight="balanced",
                random_state=RANDOM_STATE,
                n_jobs=-1,
            )),
        ]),

        "svm_rbf": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", SVC(
                kernel="rbf",
                class_weight="balanced",
                probability=True,
                random_state=RANDOM_STATE,
            )),
        ]),
    }

    try:
        from xgboost import XGBClassifier

        models["xgboost"] = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LabelEncodedClassifier(
                XGBClassifier(
                    n_estimators=N_ESTIMATORS,
                    learning_rate=0.05,
                    max_depth=3,
                    subsample=0.9,
                    colsample_bytree=0.9,
                    objective="multi:softprob",
                    eval_metric="mlogloss",
                    random_state=RANDOM_STATE,
                    n_jobs=-1,
                )
            )),
        ])
    except ImportError:
        print("XGBoost not installed. Skipping xgboost.")

    try:
        from lightgbm import LGBMClassifier

        models["lightgbm"] = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LGBMClassifier(
                n_estimators=N_ESTIMATORS,
                learning_rate=0.05,
                class_weight="balanced",
                random_state=RANDOM_STATE,
                n_jobs=-1,
                verbose=-1,
            )),
        ])
    except ImportError:
        print("LightGBM not installed. Skipping lightgbm.")

    try:
        from catboost import CatBoostClassifier

        models["catboost"] = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", CatBoostClassifier(
                iterations=N_ESTIMATORS,
                learning_rate=0.05,
                depth=4,
                loss_function="MultiClass",
                auto_class_weights="Balanced",
                random_seed=RANDOM_STATE,
                verbose=False,
                thread_count=-1,
            )),
        ])
    except ImportError:
        print("CatBoost not installed. Skipping catboost.")

    return models


def run_label_permutation_test(
    df: pd.DataFrame,
    feature_cols: list[str],
    model,
    group_name: str,
    n_repeats: int = 5,
) -> dict:
    X = df[feature_cols].reset_index(drop=True)
    y = df[LABEL_COL].reset_index(drop=True)

    cv = StratifiedKFold(
        n_splits=CV_SPLITS,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    scores = []

    for seed in range(n_repeats):
        shuffled_y = y.sample(
            frac=1,
            random_state=seed,
        ).reset_index(drop=True)

        score = cross_val_score(
            model,
            X,
            shuffled_y,
            cv=cv,
            scoring="f1_macro",
            n_jobs=1,
        ).mean()

        scores.append(score)

    return {
        "feature_group": group_name,
        "permutation_macro_f1_mean": float(np.mean(scores)),
        "permutation_macro_f1_std": float(np.std(scores)),
    }


def run_robustness_checks(
    df: pd.DataFrame,
    feature_groups: dict[str, list[str]],
) -> None:
    print("\nRunning leakage / robustness checks...")

    models = build_models()

    groups_to_check = [
        "keystroke_only",
        "text_only",
        "audio_only",
        "image_only",
        "multimodal_all",
    ]

    checks = []

    for group_name in groups_to_check:
        if group_name not in feature_groups:
            continue

        print(f"Permutation test: {group_name}")

        result = run_label_permutation_test(
            df=df,
            feature_cols=feature_groups[group_name],
            model=models["logistic_regression"],
            group_name=group_name,
            n_repeats=5,
        )

        checks.append(result)

    checks_df = pd.DataFrame(checks)

    output_path = OUTPUT_DIR / "leakage_permutation_check.csv"
    checks_df.to_csv(output_path, index=False)

    notes = [
        "Leakage / robustness validation notes",
        "====================================",
        "",
        "Interpretation:",
        "- Four-class chance-level macro-F1 is approximately 0.25.",
        "- If permutation macro-F1 is near 0.25, this is good.",
        "- If permutation macro-F1 remains high, there may be data leakage.",
        "- If text_only performance is near-perfect, guided text prompts may encode labels strongly.",
        "- This should be discussed as a dataset limitation if applicable.",
        "",
    ]

    notes_path = OUTPUT_DIR / "robustness_notes.txt"
    notes_path.write_text("\n".join(notes), encoding="utf-8")

    print("\nPermutation leakage check:")
    print(checks_df)
    print(f"\nSaved to: {output_path}")


def cross_validate_group(
    X: pd.DataFrame,
    y: pd.Series,
    models: dict,
    group_name: str,
) -> list[dict]:

    cv = StratifiedKFold(
        n_splits=CV_SPLITS,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    scoring = {
        "accuracy": "accuracy",
        "macro_f1": "f1_macro",
    }

    rows = []

    for model_name, model in models.items():
        print(f"Cross-validating {group_name} / {model_name}...")

        scores = cross_validate(
            model,
            X,
            y,
            cv=cv,
            scoring=scoring,
            n_jobs=1,
            return_train_score=False,
        )

        rows.append({
            "feature_group": group_name,
            "model": model_name,
            "num_features": X.shape[1],
            "cv_accuracy_mean": float(scores["test_accuracy"].mean()),
            "cv_accuracy_std": float(scores["test_accuracy"].std()),
            "cv_macro_f1_mean": float(scores["test_macro_f1"].mean()),
            "cv_macro_f1_std": float(scores["test_macro_f1"].std()),
            "fit_time_mean_sec": float(scores["fit_time"].mean()),
            "score_time_mean_sec": float(scores["score_time"].mean()),
        })

    return rows


def evaluate_best_model(
    df: pd.DataFrame,
    feature_cols: list[str],
    group_name: str,
    model_name: str,
    model,
) -> dict:

    X = df[feature_cols]
    y = df[LABEL_COL]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        stratify=y,
        random_state=RANDOM_STATE,
    )

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    macro_f1 = f1_score(y_test, y_pred, average="macro")

    report_text = classification_report(y_test, y_pred)

    report_path = OUTPUT_DIR / f"{group_name}_{model_name}_classification_report.txt"
    report_path.write_text(report_text, encoding="utf-8")

    labels = sorted(y.unique())
    cm = confusion_matrix(y_test, y_pred, labels=labels)

    plt.figure(figsize=(7, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
    )
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    plt.title(f"{group_name} / {model_name}")
    plt.tight_layout()
    plt.savefig(
        OUTPUT_DIR / f"{group_name}_{model_name}_confusion_matrix.png",
        dpi=300,
    )
    plt.close()

    joblib.dump(
        model,
        OUTPUT_DIR / f"{group_name}_{model_name}.joblib",
    )

    return {
        "feature_group": group_name,
        "best_model": model_name,
        "num_features": len(feature_cols),
        "test_accuracy": float(acc),
        "test_macro_f1": float(macro_f1),
    }


def main():
    df = load_dataset()
    feature_groups = get_feature_groups(df)

    print("\nMultimodal dataset loaded successfully.")
    print(f"Samples: {len(df)}")

    y = df[LABEL_COL]

    print("\nLabel distribution:")
    print(y.value_counts().sort_index())

    print("\nFeature groups:")
    for group_name, cols in feature_groups.items():
        print(f"  {group_name}: {len(cols)} features")

    run_robustness_checks(df, feature_groups)

    models = build_models()

    all_cv_rows = []

    for group_name, cols in feature_groups.items():
        X = df[cols]

        rows = cross_validate_group(
            X=X,
            y=y,
            models=models,
            group_name=group_name,
        )

        all_cv_rows.extend(rows)

    cv_results = pd.DataFrame(all_cv_rows).sort_values(
        by=["feature_group", "cv_macro_f1_mean"],
        ascending=[True, False],
    )

    cv_path = OUTPUT_DIR / "cross_validation_comparison.csv"
    cv_results.to_csv(cv_path, index=False)

    best_per_group = (
        cv_results
        .sort_values("cv_macro_f1_mean", ascending=False)
        .groupby("feature_group", as_index=False)
        .first()
    )

    best_path = OUTPUT_DIR / "best_model_per_feature_group.csv"
    best_per_group.to_csv(best_path, index=False)

    test_rows = []

    for _, row in best_per_group.iterrows():
        group_name = row["feature_group"]
        model_name = row["model"]

        model = build_models()[model_name]

        result = evaluate_best_model(
            df=df,
            feature_cols=feature_groups[group_name],
            group_name=group_name,
            model_name=model_name,
            model=model,
        )

        test_rows.append(result)

    test_results = pd.DataFrame(test_rows).sort_values(
        by="test_macro_f1",
        ascending=False,
    )

    test_path = OUTPUT_DIR / "test_set_comparison.csv"
    test_results.to_csv(test_path, index=False)

    with open(OUTPUT_DIR / "feature_groups.json", "w", encoding="utf-8") as f:
        json.dump(feature_groups, f, indent=4)

    metadata = {
        "dataset": str(DATA_PATH),
        "num_samples": int(len(df)),
        "cv_splits": CV_SPLITS,
        "n_estimators_or_iterations": N_ESTIMATORS,
        "models": list(models.keys()),
        "feature_groups": {
            name: len(cols)
            for name, cols in feature_groups.items()
        },
    }

    with open(OUTPUT_DIR / "comparison_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=4)

    print("\nCross-validation comparison:")
    print(cv_results)

    print("\nBest model per feature group:")
    print(best_per_group)

    print("\nTest-set comparison:")
    print(test_results)

    print(f"\nOutputs saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
