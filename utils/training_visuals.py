# utils/training_visuals.py
# SenseFuzeAI - Training Visualisation Utilities
#
# Generates visual artefacts for model training reports:
#   - candidate model comparison CSV
#   - candidate model comparison bar chart
#   - classification report CSV
#   - classification report heatmap
#   - confusion matrix heatmap
#   - class distribution bar chart

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd


def ensure_output_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def candidate_results_to_dataframe(
    candidate_results: dict[str, dict[str, Any]]
) -> pd.DataFrame:
    rows = []

    for model_name, metrics in candidate_results.items():
        rows.append(
            {
                "model": model_name,
                "status": metrics.get("status", "evaluated"),
                "accuracy": float(metrics.get("accuracy", 0.0)),
                "macro_precision": float(metrics.get("macro_precision", 0.0)),
                "macro_recall": float(metrics.get("macro_recall", 0.0)),
                "macro_f1": float(metrics.get("macro_f1", 0.0)),
                "weighted_precision": float(metrics.get("weighted_precision", 0.0)),
                "weighted_recall": float(metrics.get("weighted_recall", 0.0)),
                "weighted_f1": float(metrics.get("weighted_f1", 0.0)),
                "reason": metrics.get("reason", ""),
            }
        )

    df = pd.DataFrame(rows)

    if not df.empty:
        df = df.sort_values(
            by=["macro_f1", "accuracy"],
            ascending=False,
        ).reset_index(drop=True)

    return df


def save_candidate_comparison_csv(
    candidate_results: dict[str, dict[str, Any]],
    output_dir: Path,
    modality_name: str,
) -> Path:
    ensure_output_dir(output_dir)

    df = candidate_results_to_dataframe(candidate_results)
    output_path = output_dir / f"{modality_name}_candidate_model_comparison.csv"
    df.to_csv(output_path, index=False)

    return output_path


def save_candidate_comparison_chart(
    candidate_results: dict[str, dict[str, Any]],
    output_dir: Path,
    modality_name: str,
) -> Path:
    ensure_output_dir(output_dir)

    df = candidate_results_to_dataframe(candidate_results)

    output_path = output_dir / f"{modality_name}_candidate_model_comparison.png"

    if df.empty:
        return output_path

    x_labels = df["model"].tolist()
    x_positions = range(len(x_labels))

    plt.figure(figsize=(13, 6))

    plt.bar(
        [x - 0.25 for x in x_positions],
        df["accuracy"],
        width=0.25,
        label="Accuracy",
    )

    plt.bar(
        x_positions,
        df["macro_f1"],
        width=0.25,
        label="Macro F1",
    )

    plt.bar(
        [x + 0.25 for x in x_positions],
        df["weighted_f1"],
        width=0.25,
        label="Weighted F1",
    )

    plt.title(f"{modality_name.title()} Candidate Model Comparison")
    plt.xlabel("Candidate classifier")
    plt.ylabel("Score")
    plt.ylim(0, 1.05)
    plt.xticks(list(x_positions), x_labels, rotation=35, ha="right")
    plt.legend()
    plt.grid(axis="y", alpha=0.25)
    plt.tight_layout()

    plt.savefig(output_path, dpi=200)
    plt.close()

    return output_path


def classification_report_to_dataframe(
    classification_report_dict: dict[str, Any],
    class_labels: list[str],
) -> pd.DataFrame:
    rows = []

    for label in class_labels:
        class_metrics = classification_report_dict.get(label, {})

        rows.append(
            {
                "class": label,
                "precision": float(class_metrics.get("precision", 0.0)),
                "recall": float(class_metrics.get("recall", 0.0)),
                "f1_score": float(class_metrics.get("f1-score", 0.0)),
                "support": int(class_metrics.get("support", 0)),
            }
        )

    return pd.DataFrame(rows)


def save_classification_report_csv(
    classification_report_dict: dict[str, Any],
    class_labels: list[str],
    output_dir: Path,
    modality_name: str,
) -> Path:
    ensure_output_dir(output_dir)

    df = classification_report_to_dataframe(
        classification_report_dict,
        class_labels,
    )

    output_path = output_dir / f"{modality_name}_classification_report.csv"
    df.to_csv(output_path, index=False)

    return output_path


def save_classification_report_heatmap(
    classification_report_dict: dict[str, Any],
    class_labels: list[str],
    output_dir: Path,
    modality_name: str,
) -> Path:
    ensure_output_dir(output_dir)

    df = classification_report_to_dataframe(
        classification_report_dict,
        class_labels,
    )

    metric_df = df.set_index("class")[["precision", "recall", "f1_score"]]

    output_path = output_dir / f"{modality_name}_classification_report_heatmap.png"

    plt.figure(figsize=(8, 4.8))
    plt.imshow(metric_df.values, aspect="auto", vmin=0, vmax=1)

    plt.title(f"{modality_name.title()} Classification Report")
    plt.xticks(range(len(metric_df.columns)), metric_df.columns)
    plt.yticks(range(len(metric_df.index)), metric_df.index)

    for row_idx in range(metric_df.shape[0]):
        for col_idx in range(metric_df.shape[1]):
            value = metric_df.iloc[row_idx, col_idx]
            plt.text(
                col_idx,
                row_idx,
                f"{value:.2f}",
                ha="center",
                va="center",
                color="white" if value < 0.5 else "black",
                fontsize=10,
                fontweight="bold",
            )

    plt.colorbar(label="Score")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()

    return output_path


def save_confusion_matrix_heatmap(
    confusion_matrix_values: list[list[int]],
    class_labels: list[str],
    output_dir: Path,
    modality_name: str,
) -> Path:
    ensure_output_dir(output_dir)

    output_path = output_dir / f"{modality_name}_confusion_matrix.png"

    matrix_df = pd.DataFrame(
        confusion_matrix_values,
        index=class_labels,
        columns=class_labels,
    )

    plt.figure(figsize=(7, 6))
    plt.imshow(matrix_df.values, aspect="auto")

    plt.title(f"{modality_name.title()} Confusion Matrix")
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    plt.xticks(range(len(class_labels)), class_labels, rotation=35, ha="right")
    plt.yticks(range(len(class_labels)), class_labels)

    max_value = matrix_df.values.max() if matrix_df.values.size else 0

    for row_idx in range(matrix_df.shape[0]):
        for col_idx in range(matrix_df.shape[1]):
            value = int(matrix_df.iloc[row_idx, col_idx])
            plt.text(
                col_idx,
                row_idx,
                str(value),
                ha="center",
                va="center",
                color="white" if max_value > 0 and value > max_value / 2 else "black",
                fontsize=11,
                fontweight="bold",
            )

    plt.colorbar(label="Number of samples")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()

    return output_path


def save_class_distribution_chart(
    labels: pd.Series,
    class_labels: list[str],
    output_dir: Path,
    modality_name: str,
) -> Path:
    ensure_output_dir(output_dir)

    counts = labels.value_counts().reindex(class_labels, fill_value=0)

    output_path = output_dir / f"{modality_name}_class_distribution.png"

    plt.figure(figsize=(8, 4.8))
    plt.bar(counts.index, counts.values)

    plt.title(f"{modality_name.title()} Class Distribution")
    plt.xlabel("Class")
    plt.ylabel("Number of samples")
    plt.grid(axis="y", alpha=0.25)

    for idx, value in enumerate(counts.values):
        plt.text(
            idx,
            value,
            str(int(value)),
            ha="center",
            va="bottom",
            fontweight="bold",
        )

    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()

    return output_path


def save_metric_table_image(
    candidate_results: dict[str, dict[str, Any]],
    output_dir: Path,
    modality_name: str,
) -> Path:
    ensure_output_dir(output_dir)

    df = candidate_results_to_dataframe(candidate_results)

    display_cols = [
        "model",
        "accuracy",
        "macro_f1",
        "weighted_f1",
    ]

    table_df = df[display_cols].copy()

    for col in ["accuracy", "macro_f1", "weighted_f1"]:
        table_df[col] = table_df[col].map(lambda value: f"{value:.4f}")

    output_path = output_dir / f"{modality_name}_candidate_model_table.png"

    fig_height = max(2.6, 0.45 * len(table_df) + 1.5)
    fig, ax = plt.subplots(figsize=(10, fig_height))
    ax.axis("off")

    table = ax.table(
        cellText=table_df.values,
        colLabels=["Model", "Accuracy", "Macro F1", "Weighted F1"],
        loc="center",
        cellLoc="center",
    )

    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.4)

    for key, cell in table.get_celld().items():
        row_idx, _ = key

        if row_idx == 0:
            cell.set_text_props(weight="bold")
            cell.set_facecolor("#d9eaf7")
        elif row_idx == 1:
            cell.set_facecolor("#d9f7df")

    plt.title(
        f"{modality_name.title()} Candidate Model Ranking",
        fontweight="bold",
        pad=16,
    )

    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()

    return output_path


def generate_training_visuals(
    modality_name: str,
    candidate_results: dict[str, dict[str, Any]],
    best_metrics: dict[str, Any],
    y_all: pd.Series,
    class_labels: list[str],
    output_root: Path,
) -> dict[str, str]:
    """
    Generate all standard visual training artefacts for one modality.

    Returns a dictionary of output file paths.
    """
    output_dir = output_root / modality_name
    ensure_output_dir(output_dir)

    generated = {}

    generated["candidate_comparison_csv"] = str(
        save_candidate_comparison_csv(
            candidate_results,
            output_dir,
            modality_name,
        )
    )

    generated["candidate_comparison_chart"] = str(
        save_candidate_comparison_chart(
            candidate_results,
            output_dir,
            modality_name,
        )
    )

    generated["candidate_model_table"] = str(
        save_metric_table_image(
            candidate_results,
            output_dir,
            modality_name,
        )
    )

    generated["classification_report_csv"] = str(
        save_classification_report_csv(
            best_metrics.get("classification_report", {}),
            class_labels,
            output_dir,
            modality_name,
        )
    )

    generated["classification_report_heatmap"] = str(
        save_classification_report_heatmap(
            best_metrics.get("classification_report", {}),
            class_labels,
            output_dir,
            modality_name,
        )
    )

    generated["confusion_matrix_heatmap"] = str(
        save_confusion_matrix_heatmap(
            best_metrics.get("confusion_matrix", []),
            class_labels,
            output_dir,
            modality_name,
        )
    )

    generated["class_distribution_chart"] = str(
        save_class_distribution_chart(
            y_all,
            class_labels,
            output_dir,
            modality_name,
        )
    )

    return generated
