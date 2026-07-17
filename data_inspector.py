# filename: data_checker.py
#
# Purpose:
# Inspect multimodal feature data and model-comparison outputs.
#
# Supported checks:
# - dataset row count;
# - class distribution;
# - duplicate and missing session IDs;
# - missing and non-finite feature values;
# - cross-validation results for every feature group;
# - selected model per feature group;
# - supplementary test-set results;
# - combined unimodal and multimodal comparison.
#
# Example commands:
#
# python data_checker.py
# python data_checker.py --mode all
# python data_checker.py --mode features
# python data_checker.py --mode results
# python data_checker.py --mode group --feature-group text_only
# python data_checker.py --mode group --feature-group multimodal_all

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


# ==============================================================
# Default paths
# ==============================================================

DEFAULT_FEATURES_PATH = Path(
    "data/processed/multimodal_features.csv"
)

DEFAULT_RESULTS_DIR = Path(
    "data/processed/multimodal_comparison_results"
)

CROSS_VALIDATION_FILENAME = (
    "cross_validation_comparison.csv"
)

BEST_MODEL_FILENAME = (
    "best_model_per_feature_group.csv"
)

TEST_SET_FILENAME = (
    "test_set_comparison.csv"
)


# ==============================================================
# Supported feature groups
# ==============================================================

FEATURE_GROUPS: tuple[str, ...] = (
    "keystroke_only",
    "text_only",
    "audio_only",
    "image_only",
    "multimodal_all",
)


# ==============================================================
# Display configuration
# ==============================================================

CROSS_VALIDATION_COLUMNS: tuple[str, ...] = (
    "model",
    "num_features",
    "cv_accuracy_mean",
    "cv_accuracy_std",
    "cv_macro_f1_mean",
    "cv_macro_f1_std",
    "fit_time_mean_sec",
    "score_time_mean_sec",
)


# ==============================================================
# General helpers
# ==============================================================

def print_heading(title: str) -> None:
    """
    Print a clearly separated report heading.
    """
    separator = "=" * 78

    print(f"\n{separator}")
    print(title)
    print(separator)


def print_subheading(title: str) -> None:
    """
    Print a smaller report heading.
    """
    separator = "-" * 78

    print(f"\n{title}")
    print(separator)


def validate_file(path: Path, description: str) -> None:
    """
    Confirm that a required file exists.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"{description} was not found:\n{path}"
        )

    if not path.is_file():
        raise FileNotFoundError(
            f"{description} is not a file:\n{path}"
        )


def validate_directory(
    path: Path,
    description: str,
) -> None:
    """
    Confirm that a required directory exists.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"{description} was not found:\n{path}"
        )

    if not path.is_dir():
        raise NotADirectoryError(
            f"{description} is not a directory:\n{path}"
        )


def read_csv_file(
    path: Path,
    description: str,
) -> pd.DataFrame:
    """
    Read a CSV file with controlled error reporting.
    """
    validate_file(path, description)

    try:
        dataframe = pd.read_csv(path)
    except pd.errors.EmptyDataError as error:
        raise ValueError(
            f"{description} is empty:\n{path}"
        ) from error
    except pd.errors.ParserError as error:
        raise ValueError(
            f"{description} could not be parsed:\n{path}"
        ) from error

    return dataframe


def available_columns(
    dataframe: pd.DataFrame,
    requested_columns: Iterable[str],
) -> list[str]:
    """
    Return only requested columns that exist.
    """
    return [
        column
        for column in requested_columns
        if column in dataframe.columns
    ]


def print_dataframe(
    dataframe: pd.DataFrame,
    empty_message: str,
) -> None:
    """
    Print a dataframe or an explanatory empty message.
    """
    if dataframe.empty:
        print(empty_message)
        return

    print(
        dataframe.to_string(
            index=False,
        )
    )


# ==============================================================
# Feature-dataset inspection
# ==============================================================

def inspect_feature_dataset(
    features_path: Path,
) -> None:
    """
    Inspect multimodal_features.csv.

    Checks:
    - total rows and columns;
    - class distribution;
    - duplicate session IDs;
    - missing session IDs;
    - missing values;
    - non-numeric values;
    - infinite values;
    - constant feature columns.
    """
    print_heading("MULTIMODAL FEATURE-DATA INSPECTION")

    dataframe = read_csv_file(
        features_path,
        "Multimodal feature dataset",
    )

    print(f"\nFile: {features_path}")
    print(f"Total rows: {len(dataframe)}")
    print(f"Total columns: {len(dataframe.columns)}")

    inspect_class_distribution(dataframe)
    inspect_session_identifiers(dataframe)
    inspect_feature_values(dataframe)


def inspect_class_distribution(
    dataframe: pd.DataFrame,
) -> None:
    """
    Display class counts and percentages.
    """
    print_subheading("CLASS DISTRIBUTION")

    if "label" not in dataframe.columns:
        print(
            "The dataset does not contain a 'label' column."
        )
        return

    class_counts = dataframe[
        "label"
    ].value_counts(
        dropna=False,
    )

    if class_counts.empty:
        print("No class labels were found.")
        return

    total_rows = len(dataframe)

    distribution = class_counts.rename(
        "count"
    ).reset_index()

    distribution.columns = [
        "label",
        "count",
    ]

    distribution["percentage"] = (
        distribution["count"]
        / total_rows
        * 100
    ).round(2)

    print_dataframe(
        distribution,
        "No class-distribution rows were found.",
    )


def inspect_session_identifiers(
    dataframe: pd.DataFrame,
) -> None:
    """
    Inspect duplicate, missing, and blank session identifiers.
    """
    print_subheading("SESSION-ID VALIDATION")

    if "session_id" not in dataframe.columns:
        print(
            "The dataset does not contain a "
            "'session_id' column."
        )
        return

    session_series = dataframe["session_id"]

    missing_rows = dataframe[
        session_series.isna()
    ]

    blank_rows = dataframe[
        session_series.notna()
        & session_series.astype(str).str.strip().eq("")
    ]

    duplicate_mask = (
        session_series.notna()
        & session_series.duplicated(
            keep=False,
        )
    )

    duplicate_columns = available_columns(
        dataframe,
        (
            "session_id",
            "label",
        ),
    )

    duplicate_rows = dataframe.loc[
        duplicate_mask,
        duplicate_columns,
    ].sort_values(
        by="session_id",
    )

    print(
        f"Missing session IDs: "
        f"{len(missing_rows)}"
    )

    print(
        f"Blank session IDs: "
        f"{len(blank_rows)}"
    )

    print(
        f"Rows with duplicated session IDs: "
        f"{len(duplicate_rows)}"
    )

    print("\nDuplicate session-ID rows:")

    print_dataframe(
        duplicate_rows,
        "None",
    )

    if not missing_rows.empty:
        print("\nRows with missing session IDs:")

        print_dataframe(
            missing_rows,
            "None",
        )

    if not blank_rows.empty:
        print("\nRows with blank session IDs:")

        print_dataframe(
            blank_rows,
            "None",
        )


def inspect_feature_values(
    dataframe: pd.DataFrame,
) -> None:
    """
    Inspect feature columns for missing, non-numeric,
    infinite, and constant values.
    """
    print_subheading("FEATURE-VALUE VALIDATION")

    identifier_columns = [
        column
        for column in (
            "session_id",
            "label",
        )
        if column in dataframe.columns
    ]

    feature_dataframe = dataframe.drop(
        columns=identifier_columns,
        errors="ignore",
    )

    if feature_dataframe.empty:
        print(
            "No feature columns remain after removing "
            "identifier columns."
        )
        return

    original_missing_count = int(
        feature_dataframe.isna().sum().sum()
    )

    numeric_features = feature_dataframe.apply(
        pd.to_numeric,
        errors="coerce",
    )

    converted_missing_count = int(
        numeric_features.isna().sum().sum()
    )

    non_numeric_count = (
        converted_missing_count
        - original_missing_count
    )

    numeric_array = numeric_features.to_numpy(
        dtype=float,
    )

    infinite_mask = np.isinf(
        numeric_array,
    )

    infinite_count = int(
        infinite_mask.sum()
    )

    finite_count = int(
        np.isfinite(
            numeric_array,
        ).sum()
    )

    total_values = int(
        numeric_array.size
    )

    constant_columns = [
        column
        for column in numeric_features.columns
        if numeric_features[column].nunique(
            dropna=True,
        )
        <= 1
    ]

    print(
        f"Feature columns: "
        f"{len(feature_dataframe.columns)}"
    )

    print(
        f"Total feature values: "
        f"{total_values}"
    )

    print(
        f"Original missing feature values: "
        f"{original_missing_count}"
    )

    print(
        f"Non-numeric feature values converted to NaN: "
        f"{non_numeric_count}"
    )

    print(
        f"Infinite feature values: "
        f"{infinite_count}"
    )

    print(
        f"Finite feature values: "
        f"{finite_count}"
    )

    print(
        f"Constant or effectively empty feature columns: "
        f"{len(constant_columns)}"
    )

    if constant_columns:
        for column in constant_columns:
            print(f"  - {column}")

    missing_by_column = (
        numeric_features
        .isna()
        .sum()
    )

    missing_by_column = missing_by_column[
        missing_by_column > 0
    ].sort_values(
        ascending=False,
    )

    if not missing_by_column.empty:
        print("\nMissing values by feature column:")

        missing_report = (
            missing_by_column
            .rename("missing_count")
            .reset_index()
        )

        missing_report.columns = [
            "feature",
            "missing_count",
        ]

        print_dataframe(
            missing_report,
            "None",
        )


# ==============================================================
# Result-file loading
# ==============================================================

def load_result_tables(
    results_dir: Path,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    """
    Load cross-validation, selected-model, and test-set tables.
    """
    validate_directory(
        results_dir,
        "Model-comparison results directory",
    )

    cross_validation_path = (
        results_dir
        / CROSS_VALIDATION_FILENAME
    )

    best_model_path = (
        results_dir
        / BEST_MODEL_FILENAME
    )

    test_set_path = (
        results_dir
        / TEST_SET_FILENAME
    )

    cross_validation_df = read_csv_file(
        cross_validation_path,
        "Cross-validation comparison file",
    )

    best_model_df = read_csv_file(
        best_model_path,
        "Best-model-per-feature-group file",
    )

    test_set_df = read_csv_file(
        test_set_path,
        "Test-set comparison file",
    )

    required_dataframes = {
        CROSS_VALIDATION_FILENAME:
            cross_validation_df,
        BEST_MODEL_FILENAME:
            best_model_df,
        TEST_SET_FILENAME:
            test_set_df,
    }

    for filename, dataframe in (
        required_dataframes.items()
    ):
        if "feature_group" not in dataframe.columns:
            raise ValueError(
                f"'{filename}' does not contain the "
                "required 'feature_group' column."
            )

    return (
        cross_validation_df,
        best_model_df,
        test_set_df,
    )


# ==============================================================
# Individual feature-group reporting
# ==============================================================

def inspect_feature_group(
    feature_group: str,
    cross_validation_df: pd.DataFrame,
    best_model_df: pd.DataFrame,
    test_set_df: pd.DataFrame,
) -> None:
    """
    Inspect all result tables for one feature group.
    """
    if feature_group not in FEATURE_GROUPS:
        raise ValueError(
            f"Unsupported feature group: "
            f"{feature_group}"
        )

    display_name = feature_group.replace(
        "_",
        " ",
    ).upper()

    print_heading(
        f"{display_name} RESULTS"
    )

    inspect_cross_validation_group(
        feature_group=feature_group,
        cross_validation_df=cross_validation_df,
    )

    inspect_best_model_group(
        feature_group=feature_group,
        best_model_df=best_model_df,
    )

    inspect_test_result_group(
        feature_group=feature_group,
        test_set_df=test_set_df,
    )


def inspect_cross_validation_group(
    feature_group: str,
    cross_validation_df: pd.DataFrame,
) -> None:
    """
    Display ranked cross-validation results for one group.
    """
    print_subheading(
        "CROSS-VALIDATION RESULTS"
    )

    columns = available_columns(
        cross_validation_df,
        CROSS_VALIDATION_COLUMNS,
    )

    subset = cross_validation_df.loc[
        cross_validation_df[
            "feature_group"
        ]
        == feature_group,
        columns,
    ].copy()

    if (
        not subset.empty
        and "cv_macro_f1_mean"
        in subset.columns
    ):
        subset.sort_values(
            "cv_macro_f1_mean",
            ascending=False,
            inplace=True,
        )

    print_dataframe(
        subset,
        (
            "No cross-validation rows were found "
            f"for '{feature_group}'."
        ),
    )


def inspect_best_model_group(
    feature_group: str,
    best_model_df: pd.DataFrame,
) -> None:
    """
    Display the selected model for one feature group.
    """
    print_subheading(
        "SELECTED MODEL"
    )

    subset = best_model_df.loc[
        best_model_df[
            "feature_group"
        ]
        == feature_group
    ].copy()

    print_dataframe(
        subset,
        (
            "No selected-model row was found "
            f"for '{feature_group}'."
        ),
    )


def inspect_test_result_group(
    feature_group: str,
    test_set_df: pd.DataFrame,
) -> None:
    """
    Display supplementary test-set results for one group.
    """
    print_subheading(
        "SUPPLEMENTARY TEST-SET RESULT"
    )

    subset = test_set_df.loc[
        test_set_df[
            "feature_group"
        ]
        == feature_group
    ].copy()

    print_dataframe(
        subset,
        (
            "No test-set row was found "
            f"for '{feature_group}'."
        ),
    )


# ==============================================================
# All-group reporting
# ==============================================================

def inspect_all_feature_groups(
    cross_validation_df: pd.DataFrame,
    best_model_df: pd.DataFrame,
    test_set_df: pd.DataFrame,
) -> None:
    """
    Inspect every supported unimodal and fusion feature group.
    """
    for feature_group in FEATURE_GROUPS:
        inspect_feature_group(
            feature_group=feature_group,
            cross_validation_df=(
                cross_validation_df
            ),
            best_model_df=best_model_df,
            test_set_df=test_set_df,
        )


def inspect_selected_test_comparison(
    test_set_df: pd.DataFrame,
) -> None:
    """
    Display selected unimodal and fusion test-set results together.
    """
    print_heading(
        "SELECTED UNIMODAL AND FUSION TEST RESULTS"
    )

    subset = test_set_df.loc[
        test_set_df[
            "feature_group"
        ].isin(
            FEATURE_GROUPS,
        )
    ].copy()

    if subset.empty:
        print(
            "No selected unimodal or fusion "
            "test-set rows were found."
        )
        return

    feature_group_order = {
        feature_group: index
        for index, feature_group
        in enumerate(FEATURE_GROUPS)
    }

    subset["_group_order"] = (
        subset["feature_group"]
        .map(feature_group_order)
    )

    subset.sort_values(
        by="_group_order",
        inplace=True,
    )

    subset.drop(
        columns=["_group_order"],
        inplace=True,
    )

    print_dataframe(
        subset,
        "No comparison rows were found.",
    )


def inspect_cross_validation_summary(
    cross_validation_df: pd.DataFrame,
) -> None:
    """
    Display the highest-ranked cross-validation row
    for every feature group.
    """
    print_heading(
        "BEST CROSS-VALIDATION RESULT PER FEATURE GROUP"
    )

    if "cv_macro_f1_mean" not in (
        cross_validation_df.columns
    ):
        print(
            "The cross-validation table does not "
            "contain 'cv_macro_f1_mean'."
        )
        return

    relevant_rows = cross_validation_df.loc[
        cross_validation_df[
            "feature_group"
        ].isin(
            FEATURE_GROUPS,
        )
    ].copy()

    if relevant_rows.empty:
        print(
            "No supported feature groups were found."
        )
        return

    best_indices = (
        relevant_rows
        .groupby(
            "feature_group",
        )["cv_macro_f1_mean"]
        .idxmax()
    )

    best_rows = relevant_rows.loc[
        best_indices
    ].copy()

    order_mapping = {
        feature_group: index
        for index, feature_group
        in enumerate(FEATURE_GROUPS)
    }

    best_rows["_group_order"] = (
        best_rows["feature_group"]
        .map(order_mapping)
    )

    best_rows.sort_values(
        "_group_order",
        inplace=True,
    )

    columns = available_columns(
        best_rows,
        (
            "feature_group",
            *CROSS_VALIDATION_COLUMNS,
        ),
    )

    best_rows = best_rows[
        columns
    ]

    print_dataframe(
        best_rows,
        "No best-model rows were found.",
    )


# ==============================================================
# Result-directory inspection
# ==============================================================

def inspect_result_directory(
    results_dir: Path,
    feature_group: str | None = None,
    inspect_every_group: bool = False,
) -> None:
    """
    Inspect model-comparison result files.

    Args:
        results_dir:
            Directory containing the result CSV files.
        feature_group:
            Optional single feature group.
        inspect_every_group:
            Whether to print full details for all groups.
    """
    (
        cross_validation_df,
        best_model_df,
        test_set_df,
    ) = load_result_tables(results_dir)

    print_heading(
        "MODEL-COMPARISON RESULT INSPECTION"
    )

    print(f"\nDirectory: {results_dir}")
    print(
        "Cross-validation rows: "
        f"{len(cross_validation_df)}"
    )
    print(
        "Selected-model rows: "
        f"{len(best_model_df)}"
    )
    print(
        "Test-set rows: "
        f"{len(test_set_df)}"
    )

    if feature_group is not None:
        inspect_feature_group(
            feature_group=feature_group,
            cross_validation_df=(
                cross_validation_df
            ),
            best_model_df=best_model_df,
            test_set_df=test_set_df,
        )
    elif inspect_every_group:
        inspect_all_feature_groups(
            cross_validation_df=(
                cross_validation_df
            ),
            best_model_df=best_model_df,
            test_set_df=test_set_df,
        )
    else:
        inspect_cross_validation_summary(
            cross_validation_df,
        )

    inspect_selected_test_comparison(
        test_set_df,
    )


# ==============================================================
# Command-line interface
# ==============================================================

def parse_arguments() -> argparse.Namespace:
    """
    Parse command-line options.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Inspect multimodal feature data and "
            "model-comparison result files."
        ),
        formatter_class=(
            argparse.ArgumentDefaultsHelpFormatter
        ),
    )

    parser.add_argument(
        "--mode",
        choices=(
            "all",
            "features",
            "results",
            "group",
        ),
        default="all",
        help=(
            "Select which inspection workflow to run."
        ),
    )

    parser.add_argument(
        "--features-path",
        type=Path,
        default=DEFAULT_FEATURES_PATH,
        help=(
            "Path to multimodal_features.csv."
        ),
    )

    parser.add_argument(
        "--results-dir",
        type=Path,
        default=DEFAULT_RESULTS_DIR,
        help=(
            "Directory containing comparison-result CSV files."
        ),
    )

    parser.add_argument(
        "--feature-group",
        choices=FEATURE_GROUPS,
        default="multimodal_all",
        help=(
            "Feature group used when --mode group "
            "is selected."
        ),
    )

    parser.add_argument(
        "--all-groups",
        action="store_true",
        help=(
            "Print full result details for every "
            "feature group when inspecting results."
        ),
    )

    return parser.parse_args()


# ==============================================================
# Main entry point
# ==============================================================

def main() -> None:
    """
    Run the selected inspection workflow.
    """
    arguments = parse_arguments()

    if arguments.mode in (
        "all",
        "features",
    ):
        inspect_feature_dataset(
            arguments.features_path,
        )

    if arguments.mode == "group":
        inspect_result_directory(
            results_dir=arguments.results_dir,
            feature_group=arguments.feature_group,
        )

    elif arguments.mode in (
        "all",
        "results",
    ):
        inspect_result_directory(
            results_dir=arguments.results_dir,
            inspect_every_group=(
                arguments.all_groups
            ),
        )


if __name__ == "__main__":
    try:
        main()
    except (
        FileNotFoundError,
        NotADirectoryError,
        ValueError,
        KeyError,
        pd.errors.ParserError,
    ) as error:
        print(
            "\nData-checking process failed:\n"
            f"{error}",
            file=sys.stderr,
        )

        raise SystemExit(1) from error


# command examples 
# 1. python data_checker.py
# 2. python data_checker.py --mode features (inspect only the multimodal feature dataset)
# 3. python data_checker.py --mode results (inspect the results summary only)
# 4. python data_checker.py --mode results --all-groups (print full results for every modality and fusion group)
# 5. python data_checker.py --mode group --feature-group keystroke_only (inspect keystroke-only results)
# 6. python data_checker.py --mode group --feature-group text_only (inspect text-only results)
# 7. python data_checker.py --mode group --feature-group audio_only (inspect audio-only results)
# 8. python data_checker.py --mode group --feature-group image_only (inspect image-only results)
# 9. python data_checker.py --mode group --feature-group multimodal_all (inspect multimodal fusion results)
# 10. python data_checker.py --mode group --feature-group path\to\features.csv (use a different feature-data file)
# 11. python data_checker.py --mode group --feature-group path\to\comparison.results (use a different results directory)