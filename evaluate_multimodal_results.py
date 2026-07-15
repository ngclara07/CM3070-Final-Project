# evaluate_multimodal_results.py

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


FEATURE_GROUP_COL = "feature_group"

DEFAULT_RESULTS_DIR = Path("data/processed/multimodal_comparison_results")
DEFAULT_OUTPUT_DIR = Path("data/processed/multimodal_evaluation_summary")

CV_FILE = "cross_validation_comparison.csv"
TEST_FILE = "test_set_comparison.csv"
BEST_FILE = "best_model_per_feature_group.csv"
PERMUTATION_FILE = "leakage_permutation_check.csv"
METADATA_FILE = "comparison_metadata.json"
FEATURE_GROUPS_FILE = "feature_groups.json"

PLOT_DPI = 300


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate robust reports for multimodal comparison experiments.",
    )

    parser.add_argument(
        "--results-dir",
        type=Path,
        default=DEFAULT_RESULTS_DIR,
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
    )

    parser.add_argument(
        "--baseline-group",
        type=str,
        default="keystroke_only",
        help="Feature group used for improvement/delta comparisons.",
    )

    parser.add_argument(
        "--primary-metric",
        type=str,
        default="test_macro_f1",
        choices=["test_macro_f1", "test_accuracy", "cv_macro_f1_mean", "cv_accuracy_mean"],
    )

    parser.add_argument(
        "--plot-format",
        type=str,
        default="png",
        choices=["png", "pdf", "svg"],
    )

    parser.add_argument(
        "--style",
        type=str,
        default="whitegrid",
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
    )

    return parser.parse_args()


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="[%(levelname)s] %(message)s",
    )


def read_csv_required(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Required file not found: {path}")

    df = pd.read_csv(path)

    if df.empty:
        raise ValueError(f"File is empty: {path}")

    return df


def read_csv_optional(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        logging.warning("Optional file not found: %s", path)
        return None

    df = pd.read_csv(path)

    if df.empty:
        logging.warning("Optional file is empty: %s", path)
        return None

    return df


def read_json_optional(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        logging.warning("Optional JSON file not found: %s", path)
        return None

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def validate_columns(df: pd.DataFrame, required: set[str], name: str) -> None:
    missing = required - set(df.columns)

    if missing:
        raise ValueError(f"{name} missing required columns: {sorted(missing)}")


def validate_inputs(
    cv_df: pd.DataFrame,
    test_df: pd.DataFrame,
    best_df: pd.DataFrame | None,
    permutation_df: pd.DataFrame | None,
) -> None:
    validate_columns(
        cv_df,
        {
            FEATURE_GROUP_COL,
            "model",
            "num_features",
            "cv_accuracy_mean",
            "cv_accuracy_std",
            "cv_macro_f1_mean",
            "cv_macro_f1_std",
            "fit_time_mean_sec",
            "score_time_mean_sec",
        },
        "cross_validation_comparison.csv",
    )

    validate_columns(
        test_df,
        {
            FEATURE_GROUP_COL,
            "best_model",
            "num_features",
            "test_accuracy",
            "test_macro_f1",
        },
        "test_set_comparison.csv",
    )

    if best_df is not None:
        validate_columns(
            best_df,
            {
                FEATURE_GROUP_COL,
                "model",
                "num_features",
                "cv_accuracy_mean",
                "cv_macro_f1_mean",
            },
            "best_model_per_feature_group.csv",
        )

    if permutation_df is not None:
        validate_columns(
            permutation_df,
            {
                FEATURE_GROUP_COL,
                "permutation_macro_f1_mean",
                "permutation_macro_f1_std",
            },
            "leakage_permutation_check.csv",
        )


def coerce_numeric(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    df = df.copy()

    for col in cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def clean_inputs(
    cv_df: pd.DataFrame,
    test_df: pd.DataFrame,
    best_df: pd.DataFrame | None,
    permutation_df: pd.DataFrame | None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame | None, pd.DataFrame | None]:
    cv_df = coerce_numeric(
        cv_df,
        [
            "num_features",
            "cv_accuracy_mean",
            "cv_accuracy_std",
            "cv_macro_f1_mean",
            "cv_macro_f1_std",
            "fit_time_mean_sec",
            "score_time_mean_sec",
        ],
    )

    test_df = coerce_numeric(
        test_df,
        [
            "num_features",
            "test_accuracy",
            "test_macro_f1",
        ],
    )

    if best_df is not None:
        best_df = coerce_numeric(
            best_df,
            [
                "num_features",
                "cv_accuracy_mean",
                "cv_accuracy_std",
                "cv_macro_f1_mean",
                "cv_macro_f1_std",
            ],
        )

    if permutation_df is not None:
        permutation_df = coerce_numeric(
            permutation_df,
            [
                "permutation_macro_f1_mean",
                "permutation_macro_f1_std",
            ],
        )

    cv_df = cv_df.dropna(subset=["cv_macro_f1_mean", "cv_accuracy_mean"])
    test_df = test_df.dropna(subset=["test_macro_f1", "test_accuracy"])

    for df in [cv_df, test_df, best_df, permutation_df]:
        if df is not None and FEATURE_GROUP_COL in df.columns:
            df[FEATURE_GROUP_COL] = df[FEATURE_GROUP_COL].astype(str)

    return cv_df, test_df, best_df, permutation_df


def get_best_cv_per_group(cv_df: pd.DataFrame) -> pd.DataFrame:
    return (
        cv_df.sort_values("cv_macro_f1_mean", ascending=False)
        .groupby(FEATURE_GROUP_COL, as_index=False)
        .first()
        .sort_values("cv_macro_f1_mean", ascending=False)
        .reset_index(drop=True)
    )


def build_combined_table(
    cv_df: pd.DataFrame,
    test_df: pd.DataFrame,
    permutation_df: pd.DataFrame | None,
    baseline_group: str | None,
) -> pd.DataFrame:
    best_cv = get_best_cv_per_group(cv_df)

    combined = pd.merge(
        best_cv,
        test_df,
        on=FEATURE_GROUP_COL,
        how="outer",
        suffixes=("_cv", "_test"),
    )

    if permutation_df is not None:
        combined = pd.merge(
            combined,
            permutation_df,
            on=FEATURE_GROUP_COL,
            how="left",
        )

        combined["cv_minus_permutation_macro_f1"] = (
            combined["cv_macro_f1_mean"] - combined["permutation_macro_f1_mean"]
        )

    combined["generalization_gap_macro_f1"] = (
        combined["cv_macro_f1_mean"] - combined["test_macro_f1"]
    )

    combined["generalization_gap_accuracy"] = (
        combined["cv_accuracy_mean"] - combined["test_accuracy"]
    )

    combined = combined.sort_values("test_macro_f1", ascending=False).reset_index(drop=True)
    combined["overall_test_rank"] = combined.index + 1

    if baseline_group:
        baseline_rows = combined[combined[FEATURE_GROUP_COL] == baseline_group]

        if baseline_rows.empty:
            logging.warning("Baseline group not found: %s", baseline_group)
        else:
            baseline = baseline_rows.iloc[0]

            combined["delta_test_macro_f1_vs_baseline"] = (
                combined["test_macro_f1"] - baseline["test_macro_f1"]
            )
            combined["delta_test_accuracy_vs_baseline"] = (
                combined["test_accuracy"] - baseline["test_accuracy"]
            )
            combined["delta_cv_macro_f1_vs_baseline"] = (
                combined["cv_macro_f1_mean"] - baseline["cv_macro_f1_mean"]
            )

    return combined


def save_tables(
    cv_df: pd.DataFrame,
    test_df: pd.DataFrame,
    combined_df: pd.DataFrame,
    output_dir: Path,
) -> None:
    cv_ranked = cv_df.sort_values("cv_macro_f1_mean", ascending=False).reset_index(drop=True)
    test_ranked = test_df.sort_values("test_macro_f1", ascending=False).reset_index(drop=True)

    cv_ranked["cv_rank"] = cv_ranked.index + 1
    test_ranked["test_rank"] = test_ranked.index + 1

    cv_ranked.to_csv(output_dir / "ranked_cross_validation_results.csv", index=False)
    test_ranked.to_csv(output_dir / "ranked_test_results.csv", index=False)
    combined_df.to_csv(output_dir / "combined_ranked_results.csv", index=False)

    combined_df.round(4).to_html(
        output_dir / "combined_ranked_results.html",
        index=False,
    )


def save_plot(path: Path, plot_format: str) -> None:
    plt.tight_layout()
    plt.savefig(path.with_suffix(f".{plot_format}"), dpi=PLOT_DPI)
    plt.close()


def plot_cv_macro_f1(cv_df: pd.DataFrame, output_dir: Path, plot_format: str) -> None:
    best_cv = get_best_cv_per_group(cv_df)

    plt.figure(figsize=(11, 6))
    sns.barplot(
        data=best_cv,
        x="cv_macro_f1_mean",
        y=FEATURE_GROUP_COL,
        color="steelblue",
    )

    for i, row in enumerate(best_cv.itertuples()):
        plt.errorbar(
            x=row.cv_macro_f1_mean,
            y=i,
            xerr=row.cv_macro_f1_std,
            fmt="none",
            ecolor="black",
            capsize=4,
        )

    plt.xlabel("Mean Cross-Validation Macro-F1")
    plt.ylabel("Feature Group")
    plt.title("Best Cross-Validation Macro-F1 by Feature Group")
    plt.xlim(0, 1.05)

    save_plot(output_dir / "cv_macro_f1_comparison", plot_format)


def plot_test_macro_f1(test_df: pd.DataFrame, output_dir: Path, plot_format: str) -> None:
    df = test_df.sort_values("test_macro_f1", ascending=False)

    plt.figure(figsize=(11, 6))
    sns.barplot(
        data=df,
        x="test_macro_f1",
        y=FEATURE_GROUP_COL,
        color="darkseagreen",
    )

    plt.xlabel("Held-Out Test Macro-F1")
    plt.ylabel("Feature Group")
    plt.title("Held-Out Test Macro-F1 by Feature Group")
    plt.xlim(0, 1.05)

    save_plot(output_dir / "test_macro_f1_comparison", plot_format)


def plot_accuracy_vs_f1(test_df: pd.DataFrame, output_dir: Path, plot_format: str) -> None:
    plt.figure(figsize=(9, 7))

    sns.scatterplot(
        data=test_df,
        x="test_accuracy",
        y="test_macro_f1",
        hue=FEATURE_GROUP_COL,
        s=120,
    )

    for _, row in test_df.iterrows():
        plt.text(
            row["test_accuracy"] + 0.004,
            row["test_macro_f1"] + 0.004,
            str(row[FEATURE_GROUP_COL]),
            fontsize=8,
        )

    plt.xlabel("Test Accuracy")
    plt.ylabel("Test Macro-F1")
    plt.title("Held-Out Accuracy vs Macro-F1")
    plt.xlim(0, 1.05)
    plt.ylim(0, 1.05)
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")

    save_plot(output_dir / "accuracy_vs_macro_f1", plot_format)


def plot_model_heatmap(cv_df: pd.DataFrame, output_dir: Path, plot_format: str) -> None:
    pivot = cv_df.pivot_table(
        index=FEATURE_GROUP_COL,
        columns="model",
        values="cv_macro_f1_mean",
        aggfunc="mean",
    )

    pivot = pivot.loc[pivot.mean(axis=1).sort_values(ascending=False).index]

    plt.figure(figsize=(12, max(6, 0.45 * len(pivot))))
    sns.heatmap(
        pivot,
        annot=True,
        fmt=".3f",
        cmap="viridis",
        linewidths=0.5,
    )

    plt.xlabel("Model")
    plt.ylabel("Feature Group")
    plt.title("Cross-Validation Macro-F1 Heatmap")

    save_plot(output_dir / "cv_model_feature_group_heatmap", plot_format)


def plot_generalization_gap(
    combined_df: pd.DataFrame,
    output_dir: Path,
    plot_format: str,
) -> None:
    df = combined_df.sort_values("generalization_gap_macro_f1", ascending=False)

    plt.figure(figsize=(11, 6))
    sns.barplot(
        data=df,
        x="generalization_gap_macro_f1",
        y=FEATURE_GROUP_COL,
        color="indianred",
    )

    plt.axvline(0, color="black", linestyle="--", linewidth=1)
    plt.xlabel("CV Macro-F1 - Test Macro-F1")
    plt.ylabel("Feature Group")
    plt.title("Generalization Gap by Feature Group")

    save_plot(output_dir / "generalization_gap_macro_f1", plot_format)


def plot_permutation_check(
    permutation_df: pd.DataFrame,
    output_dir: Path,
    plot_format: str,
) -> None:
    df = permutation_df.sort_values("permutation_macro_f1_mean", ascending=False)

    plt.figure(figsize=(10, 5))
    sns.barplot(
        data=df,
        x="permutation_macro_f1_mean",
        y=FEATURE_GROUP_COL,
        color="mediumpurple",
    )

    for i, row in enumerate(df.itertuples()):
        plt.errorbar(
            x=row.permutation_macro_f1_mean,
            y=i,
            xerr=row.permutation_macro_f1_std,
            fmt="none",
            ecolor="black",
            capsize=4,
        )

    plt.axvline(0.25, color="red", linestyle="--", label="Approx. 4-class chance level")
    plt.xlabel("Permutation Macro-F1")
    plt.ylabel("Feature Group")
    plt.title("Label-Permutation Leakage Check")
    plt.xlim(0, 1.05)
    plt.legend()

    save_plot(output_dir / "permutation_leakage_check", plot_format)


def plot_num_features_vs_performance(
    combined_df: pd.DataFrame,
    output_dir: Path,
    plot_format: str,
) -> None:
    plt.figure(figsize=(9, 6))

    sns.scatterplot(
        data=combined_df,
        x="num_features_test",
        y="test_macro_f1",
        hue=FEATURE_GROUP_COL,
        s=120,
    )

    plt.xlabel("Number of Features")
    plt.ylabel("Test Macro-F1")
    plt.title("Feature Count vs Test Macro-F1")
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")

    save_plot(output_dir / "num_features_vs_test_macro_f1", plot_format)


def compute_summary_dict(
    cv_df: pd.DataFrame,
    test_df: pd.DataFrame,
    combined_df: pd.DataFrame,
    metadata: dict[str, Any] | None,
    baseline_group: str | None,
) -> dict[str, Any]:
    best_cv = cv_df.sort_values("cv_macro_f1_mean", ascending=False).iloc[0]
    best_test = test_df.sort_values("test_macro_f1", ascending=False).iloc[0]

    return {
        "best_cross_validation": {
            "feature_group": best_cv[FEATURE_GROUP_COL],
            "model": best_cv["model"],
            "num_features": int(best_cv["num_features"]),
            "cv_accuracy_mean": float(best_cv["cv_accuracy_mean"]),
            "cv_accuracy_std": float(best_cv["cv_accuracy_std"]),
            "cv_macro_f1_mean": float(best_cv["cv_macro_f1_mean"]),
            "cv_macro_f1_std": float(best_cv["cv_macro_f1_std"]),
        },
        "best_test": {
            "feature_group": best_test[FEATURE_GROUP_COL],
            "model": best_test["best_model"],
            "num_features": int(best_test["num_features"]),
            "test_accuracy": float(best_test["test_accuracy"]),
            "test_macro_f1": float(best_test["test_macro_f1"]),
        },
        "baseline_group": baseline_group,
        "num_cv_rows": int(len(cv_df)),
        "num_test_rows": int(len(test_df)),
        "num_feature_groups": int(combined_df[FEATURE_GROUP_COL].nunique()),
        "metadata": metadata,
    }


def generate_markdown_summary(
    cv_df: pd.DataFrame,
    test_df: pd.DataFrame,
    combined_df: pd.DataFrame,
    permutation_df: pd.DataFrame | None,
    metadata: dict[str, Any] | None,
    output_dir: Path,
    baseline_group: str | None,
    top_k: int,
) -> None:
    best_cv = cv_df.sort_values("cv_macro_f1_mean", ascending=False).iloc[0]
    best_test = test_df.sort_values("test_macro_f1", ascending=False).iloc[0]

    top_combined = combined_df.head(top_k).round(4).to_markdown(index=False)

    metadata_text = ""
    if metadata:
        metadata_text = f"""
## Experiment Metadata

- Dataset: `{metadata.get("dataset", "unknown")}`
- Number of samples: `{metadata.get("num_samples", "unknown")}`
- CV splits: `{metadata.get("cv_splits", "unknown")}`
- Models: `{", ".join(metadata.get("models", []))}`
"""

    permutation_text = ""
    if permutation_df is not None:
        permutation_text = """

## Leakage / Robustness Check

The file `leakage_permutation_check.csv` was found and included.

Permutation macro-F1 should usually be close to chance level.  
For a balanced four-class task, chance-level macro-F1 is approximately `0.25`.

High permutation performance may indicate leakage, duplicated information, or label-coded artifacts.
"""

    baseline_text = ""
    if baseline_group:
        baseline_text = f"""

## Baseline Comparison

Baseline feature group: `{baseline_group}`

The combined table includes delta columns where the baseline group is available.
"""

    text = f"""# Multimodal Evaluation Summary

## Best Cross-Validation Result

- Feature group: `{best_cv[FEATURE_GROUP_COL]}`
- Model: `{best_cv["model"]}`
- Number of features: `{int(best_cv["num_features"])}`
- CV accuracy: `{best_cv["cv_accuracy_mean"]:.4f} ± {best_cv["cv_accuracy_std"]:.4f}`
- CV macro-F1: `{best_cv["cv_macro_f1_mean"]:.4f} ± {best_cv["cv_macro_f1_std"]:.4f}`

## Best Held-Out Test Result

- Feature group: `{best_test[FEATURE_GROUP_COL]}`
- Model: `{best_test["best_model"]}`
- Number of features: `{int(best_test["num_features"])}`
- Test accuracy: `{best_test["test_accuracy"]:.4f}`
- Test macro-F1: `{best_test["test_macro_f1"]:.4f}`

{metadata_text}
{baseline_text}
{permutation_text}

## Top Combined Results

{top_combined}

## Generated Tables

- `ranked_cross_validation_results.csv`
- `ranked_test_results.csv`
- `combined_ranked_results.csv`
- `combined_ranked_results.html`
- `evaluation_summary.json`
- `evaluation_summary.md`

## Generated Plots

- `cv_macro_f1_comparison`
- `test_macro_f1_comparison`
- `accuracy_vs_macro_f1`
- `cv_model_feature_group_heatmap`
- `generalization_gap_macro_f1`
- `num_features_vs_test_macro_f1`
- `permutation_leakage_check`, if permutation results exist

## Interpretation Guide

Use cross-validation macro-F1 for model-selection comparisons across feature groups.

Use held-out test macro-F1 as the final estimate of generalization.

A positive delta for `multimodal_all` over single-modality feature groups supports the empirical value of multimodal fusion.

A large positive generalization gap means cross-validation performance is higher than held-out test performance, which may indicate instability, overfitting, or sensitivity to the train-test split.
"""

    (output_dir / "evaluation_summary.md").write_text(text, encoding="utf-8")


def save_json_summary(
    summary: dict[str, Any],
    output_dir: Path,
) -> None:
    with open(output_dir / "evaluation_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=4)


def check_training_artifacts(results_dir: Path, output_dir: Path) -> None:
    rows = []

    for path in sorted(results_dir.glob("*")):
        rows.append(
            {
                "artifact": path.name,
                "type": "directory" if path.is_dir() else "file",
                "size_bytes": path.stat().st_size if path.is_file() else np.nan,
            }
        )

    pd.DataFrame(rows).to_csv(output_dir / "available_training_artifacts.csv", index=False)


def run_evaluation(args: argparse.Namespace) -> None:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style=args.style)

    cv_df = read_csv_required(args.results_dir / CV_FILE)
    test_df = read_csv_required(args.results_dir / TEST_FILE)
    best_df = read_csv_optional(args.results_dir / BEST_FILE)
    permutation_df = read_csv_optional(args.results_dir / PERMUTATION_FILE)
    metadata = read_json_optional(args.results_dir / METADATA_FILE)

    validate_inputs(cv_df, test_df, best_df, permutation_df)

    cv_df, test_df, best_df, permutation_df = clean_inputs(
        cv_df,
        test_df,
        best_df,
        permutation_df,
    )

    combined_df = build_combined_table(
        cv_df=cv_df,
        test_df=test_df,
        permutation_df=permutation_df,
        baseline_group=args.baseline_group,
    )

    save_tables(
        cv_df=cv_df,
        test_df=test_df,
        combined_df=combined_df,
        output_dir=args.output_dir,
    )

    plot_cv_macro_f1(cv_df, args.output_dir, args.plot_format)
    plot_test_macro_f1(test_df, args.output_dir, args.plot_format)
    plot_accuracy_vs_f1(test_df, args.output_dir, args.plot_format)
    plot_model_heatmap(cv_df, args.output_dir, args.plot_format)
    plot_generalization_gap(combined_df, args.output_dir, args.plot_format)
    plot_num_features_vs_performance(combined_df, args.output_dir, args.plot_format)

    if permutation_df is not None:
        plot_permutation_check(permutation_df, args.output_dir, args.plot_format)

    summary = compute_summary_dict(
        cv_df=cv_df,
        test_df=test_df,
        combined_df=combined_df,
        metadata=metadata,
        baseline_group=args.baseline_group,
    )

    save_json_summary(summary, args.output_dir)

    generate_markdown_summary(
        cv_df=cv_df,
        test_df=test_df,
        combined_df=combined_df,
        permutation_df=permutation_df,
        metadata=metadata,
        output_dir=args.output_dir,
        baseline_group=args.baseline_group,
        top_k=args.top_k,
    )

    check_training_artifacts(args.results_dir, args.output_dir)

    logging.info("Evaluation report complete.")
    logging.info("Outputs saved to: %s", args.output_dir)


def main() -> None:
    setup_logging()
    args = parse_args()
    run_evaluation(args)


if __name__ == "__main__":
    main()
