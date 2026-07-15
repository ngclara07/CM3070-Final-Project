# train_keystroke_baseline.py

from pathlib import Path
import json
import joblib

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB


DATA_PATH = Path("data/processed/master_sessions_clean_scaled.csv")
OUTPUT_DIR = Path("data/processed/keystroke_baseline_results")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

LABEL_COL = "label"

NON_FEATURE_COLUMNS = {
    "session_id", "label", "created_at",
    "text_path", "keystroke_path", "audio_path", "image_path",
    "text", "validation_message", "problems",
    "validation_passed", "text_exists", "audio_exists", "image_exists",
    "is_clean", "keydown_count_json", "event_count", "expected_event_count",
}


def load_dataset(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    df = pd.read_csv(path)

    if LABEL_COL not in df.columns:
        raise ValueError(f"Missing label column: {LABEL_COL}")

    feature_cols = [
        col for col in df.columns
        if col not in NON_FEATURE_COLUMNS
        and pd.api.types.is_numeric_dtype(df[col])
    ]

    if not feature_cols:
        raise ValueError("No numeric keystroke feature columns found.")

    return df[feature_cols].copy(), df[LABEL_COL].copy(), feature_cols


def verify_feature_set(feature_cols):
    leakage_cols = {
        "session_id", "label", "text", "created_at",
        "validation_message", "text_path", "keystroke_path",
        "audio_path", "image_path", "keydown_count_json",
        "event_count", "expected_event_count",
    }

    leaked = sorted(set(feature_cols).intersection(leakage_cols))

    print("\nFeature columns used:")
    for col in feature_cols:
        print(f"  - {col}")

    if leaked:
        raise ValueError(f"Data leakage detected: {leaked}")

    print("\nFeature-set verification passed.")


def build_models():
    return {
        "logistic_regression": LogisticRegression(
            max_iter=3000,
            class_weight="balanced",
            random_state=42,
        ),
        "decision_tree": DecisionTreeClassifier(
            class_weight="balanced",
            random_state=42,
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=300,
            class_weight="balanced",
            random_state=42,
        ),
        "svm_rbf": SVC(
            kernel="rbf",
            class_weight="balanced",
            probability=True,
            random_state=42,
        ),
        "knn": KNeighborsClassifier(
            n_neighbors=5,
            weights="distance",
        ),
        "gaussian_naive_bayes": GaussianNB(),
    }


def save_random_forest_feature_importance(X, y):
    rf = RandomForestClassifier(
        n_estimators=300,
        class_weight="balanced",
        random_state=42,
    )

    rf.fit(X, y)

    importance_df = pd.DataFrame({
        "feature": X.columns,
        "importance": rf.feature_importances_,
    }).sort_values(by="importance", ascending=False)

    importance_df.to_csv(
        OUTPUT_DIR / "random_forest_feature_importance.csv",
        index=False,
    )

    print("\nTop 15 Random Forest feature importances:")
    print(importance_df.head(15))


def cross_validate_models(models, X, y):
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    rows = []

    for name, model in models.items():
        print(f"Cross-validating: {name}")

        acc_scores = cross_val_score(model, X, y, cv=cv, scoring="accuracy")
        f1_scores = cross_val_score(model, X, y, cv=cv, scoring="f1_macro")

        rows.append({
            "model": name,
            "cv_accuracy_mean": acc_scores.mean(),
            "cv_accuracy_std": acc_scores.std(),
            "cv_macro_f1_mean": f1_scores.mean(),
            "cv_macro_f1_std": f1_scores.std(),
        })

    return pd.DataFrame(rows).sort_values(
        by="cv_macro_f1_mean",
        ascending=False,
    )


def evaluate_model(name, model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    macro_f1 = f1_score(y_test, y_pred, average="macro")

    report_text = classification_report(y_test, y_pred)
    report_dict = classification_report(y_test, y_pred, output_dict=True)

    with open(OUTPUT_DIR / f"{name}_classification_report.txt", "w", encoding="utf-8") as f:
        f.write(report_text)

    labels = sorted(y_test.unique())
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
    plt.title(f"Confusion Matrix: {name}")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f"{name}_confusion_matrix.png", dpi=300)
    plt.close()

    joblib.dump(model, OUTPUT_DIR / f"{name}_model.joblib")

    return {
        "model": name,
        "test_accuracy": accuracy,
        "test_macro_f1": macro_f1,
        "classification_report": report_dict,
    }


def main():
    X, y, feature_cols = load_dataset(DATA_PATH)

    verify_feature_set(feature_cols)
    save_random_forest_feature_importance(X, y)

    print("\nDataset loaded successfully.")
    print(f"Samples: {len(X)}")
    print(f"Features: {len(feature_cols)}")

    print("\nLabel distribution:")
    print(y.value_counts().sort_index())

    models = build_models()

    cv_results = cross_validate_models(models, X, y)
    cv_results.to_csv(OUTPUT_DIR / "cross_validation_results.csv", index=False)

    print("\nCross-validation results:")
    print(cv_results)

    best_model_name = cv_results.iloc[0]["model"]
    print(f"\nBest model by CV macro-F1: {best_model_name}")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        stratify=y,
        random_state=42,
    )

    final_results = []

    for name, model in models.items():
        print(f"\nEvaluating test set: {name}")
        result = evaluate_model(name, model, X_train, X_test, y_train, y_test)
        final_results.append({
            "model": result["model"],
            "test_accuracy": result["test_accuracy"],
            "test_macro_f1": result["test_macro_f1"],
        })

    final_results_df = pd.DataFrame(final_results).sort_values(
        by="test_macro_f1",
        ascending=False,
    )

    final_results_df.to_csv(OUTPUT_DIR / "test_set_results.csv", index=False)

    with open(OUTPUT_DIR / "feature_columns.json", "w", encoding="utf-8") as f:
        json.dump(feature_cols, f, indent=4)

    print("\nTest-set results:")
    print(final_results_df)

    print(f"\nAll keystroke baseline outputs saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
