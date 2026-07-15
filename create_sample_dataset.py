# filename: create_sample_dataset.py
#
# purpose:
# create a small, balanced, reproducible multimodal sample dataset
# containing five complete sessions from each behavioural class.
#
# the script:
# - uses retroactive_keystroke_features.csv as the authoritative
#   source of session IDs and behavioural labels;
# - selects five sessions from each of four behavioural classes;
# - requires matching audio, image, keystroke and text files;
# - uses metadata.csv where matching metadata rows are available;
# - reconstructs missing metadata rows from the feature CSV;
# - recreates sample_data/ to prevent stale files;
# - creates filtered CSV files and verification reports;
# - reports output counts and total size.

from __future__ import annotations

import shutil
from collections import OrderedDict
from pathlib import Path
from typing import Iterable

import pandas as pd


# ==============================================================
# Configuration
# ==============================================================

PROJECT_ROOT = Path(__file__).resolve().parent

SOURCE_ROOT = PROJECT_ROOT / "data" / "session_aligned"
OUTPUT_ROOT = PROJECT_ROOT / "sample_data"

METADATA_CSV = SOURCE_ROOT / "metadata.csv"
KEYSTROKE_FEATURES_CSV = (
    SOURCE_ROOT / "retroactive_keystroke_features.csv"
)

MODALITY_DIRECTORIES: dict[str, Path] = {
    "audio": SOURCE_ROOT / "audio",
    "images": SOURCE_ROOT / "images",
    "keystrokes": SOURCE_ROOT / "keystrokes",
    "texts": SOURCE_ROOT / "texts",
}

SAMPLES_PER_CLASS = 5

# These are the four classes actually present in
# retroactive_keystroke_features.csv.
TARGET_CLASSES: tuple[str, ...] = (
    "fatigued",
    "focused",
    "distracted",
    "overloaded",
)

SESSION_ID_COLUMN_CANDIDATES: tuple[str, ...] = (
    "session_id",
    "session",
    "session_name",
    "session_identifier",
    "recording_id",
    "sample_id",
    "id",
)

BEHAVIOURAL_CLASS_COLUMN_CANDIDATES: tuple[str, ...] = (
    "behavioural_class",
    "behavioral_class",
    "behaviour_class",
    "behavior_class",
    "behaviour",
    "behavior",
    "class",
    "label",
    "target",
    "state",
    "condition",
    "emotion",
    "category",
)


# ==============================================================
# Column-detection helpers
# ==============================================================

def detect_column(
    dataframe: pd.DataFrame,
    candidates: tuple[str, ...],
    csv_name: str,
    column_description: str,
) -> str:
    """
    Detect a required column using case-insensitive candidate names.
    """
    normalised_columns = {
        str(column).strip().lower(): str(column)
        for column in dataframe.columns
    }

    for candidate in candidates:
        if candidate in normalised_columns:
            return normalised_columns[candidate]

    raise ValueError(
        f"Unable to identify the {column_description} column "
        f"in '{csv_name}'.\n"
        f"Available columns: {list(dataframe.columns)}\n\n"
        "Update the relevant candidate list in the configuration."
    )


def detect_session_id_column(
    dataframe: pd.DataFrame,
    csv_name: str,
) -> str:
    """Identify the session-ID column."""
    return detect_column(
        dataframe=dataframe,
        candidates=SESSION_ID_COLUMN_CANDIDATES,
        csv_name=csv_name,
        column_description="session-ID",
    )


def detect_behavioural_class_column(
    dataframe: pd.DataFrame,
    csv_name: str,
) -> str:
    """Identify the behavioural-class column."""
    return detect_column(
        dataframe=dataframe,
        candidates=BEHAVIOURAL_CLASS_COLUMN_CANDIDATES,
        csv_name=csv_name,
        column_description="behavioural-class",
    )


# ==============================================================
# Normalisation helpers
# ==============================================================

def normalise_session_id(value: object) -> str:
    """Convert a session identifier into a consistent string."""
    if pd.isna(value):
        return ""

    return str(value).strip()


def normalise_class_label(value: object) -> str:
    """Convert a class label into lower-case text."""
    if pd.isna(value):
        return ""

    return str(value).strip().lower()


# ==============================================================
# File discovery and matching
# ==============================================================

def filename_matches_session(
    file_path: Path,
    session_id: str,
) -> bool:
    """
    Determine whether a file belongs to a session.

    A match is accepted when the session ID:
    - equals the complete filename stem;
    - appears at the beginning of the filename; or
    - appears elsewhere in the filename.
    """
    filename_stem = file_path.stem.lower()
    session_token = session_id.lower()

    return (
        filename_stem == session_token
        or filename_stem.startswith(f"{session_token}_")
        or filename_stem.startswith(f"{session_token}-")
        or session_token in filename_stem
    )


def find_matching_files(
    source_directory: Path,
    session_id: str,
) -> list[Path]:
    """Find all files belonging to a session."""
    if not source_directory.exists():
        raise FileNotFoundError(
            f"Required modality directory does not exist: "
            f"{source_directory}"
        )

    return sorted(
        path
        for path in source_directory.rglob("*")
        if path.is_file()
        and filename_matches_session(path, session_id)
    )


def get_session_modality_files(
    session_id: str,
) -> dict[str, list[Path]]:
    """Return matching files for every required modality."""
    return {
        modality_name: find_matching_files(
            source_directory,
            session_id,
        )
        for modality_name, source_directory
        in MODALITY_DIRECTORIES.items()
    }


def session_has_all_modalities(
    session_id: str,
) -> bool:
    """
    Return True only when the session contains at least one file
    for every required modality.
    """
    modality_files = get_session_modality_files(session_id)

    return all(modality_files.values())


def copy_files_preserving_relative_paths(
    files: Iterable[Path],
    source_directory: Path,
    output_directory: Path,
) -> list[str]:
    """
    Copy files while preserving any nested source structure.

    Returns:
        Paths relative to sample_data/, using forward slashes.
    """
    copied_paths: list[str] = []

    for source_file in files:
        relative_path = source_file.relative_to(source_directory)
        destination_file = output_directory / relative_path

        destination_file.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        shutil.copy2(
            source_file,
            destination_file,
        )

        copied_paths.append(
            destination_file.relative_to(OUTPUT_ROOT).as_posix()
        )

    return copied_paths


# ==============================================================
# Validation
# ==============================================================

def validate_configuration() -> None:
    """Validate sample-size and class configuration."""
    if SAMPLES_PER_CLASS <= 0:
        raise ValueError(
            "SAMPLES_PER_CLASS must be greater than zero."
        )

    if not TARGET_CLASSES:
        raise ValueError(
            "TARGET_CLASSES must contain at least one class."
        )

    normalised_targets = tuple(
        normalise_class_label(class_label)
        for class_label in TARGET_CLASSES
    )

    if any(not class_label for class_label in normalised_targets):
        raise ValueError(
            "TARGET_CLASSES cannot contain blank labels."
        )

    if len(set(normalised_targets)) != len(normalised_targets):
        raise ValueError(
            "TARGET_CLASSES contains duplicate labels."
        )


def validate_input_paths() -> None:
    """Confirm that all required inputs exist."""
    required_paths = [
        METADATA_CSV,
        KEYSTROKE_FEATURES_CSV,
        *MODALITY_DIRECTORIES.values(),
    ]

    missing_paths = [
        path
        for path in required_paths
        if not path.exists()
    ]

    if missing_paths:
        formatted_paths = "\n".join(
            f"  - {path}"
            for path in missing_paths
        )

        raise FileNotFoundError(
            "The following required paths were not found:\n"
            f"{formatted_paths}"
        )


def validate_available_classes(
    keystroke_df: pd.DataFrame,
) -> None:
    """
    Confirm that the feature CSV contains all requested classes.
    """
    class_counts = (
        keystroke_df["_normalised_class_label"]
        .value_counts()
        .sort_index()
    )

    print("\nAvailable source classes:")

    for class_label, count in class_counts.items():
        print(f"  {class_label}: {count}")

    available_classes = set(class_counts.index)

    missing_classes = [
        normalise_class_label(class_label)
        for class_label in TARGET_CLASSES
        if normalise_class_label(class_label)
        not in available_classes
    ]

    if missing_classes:
        raise ValueError(
            "The feature dataset does not contain all required "
            "behavioural classes.\n"
            f"Required classes: {list(TARGET_CLASSES)}\n"
            f"Available classes: {sorted(available_classes)}\n"
            f"Missing classes: {missing_classes}"
        )


# ==============================================================
# Balanced-session selection
# ==============================================================

def select_balanced_sessions(
    keystroke_df: pd.DataFrame,
) -> OrderedDict[str, list[str]]:
    """
    Select the first five complete aligned sessions from each class.

    Selection order follows retroactive_keystroke_features.csv.
    """
    target_classes = tuple(
        normalise_class_label(class_label)
        for class_label in TARGET_CLASSES
    )

    selected_by_class: OrderedDict[str, list[str]] = OrderedDict(
        (class_label, [])
        for class_label in target_classes
    )

    seen_session_ids: set[str] = set()

    for _, row in keystroke_df.iterrows():
        session_id = row["_normalised_session_id"]
        class_label = row["_normalised_class_label"]

        if not session_id or not class_label:
            continue

        if class_label not in selected_by_class:
            continue

        if session_id in seen_session_ids:
            continue

        if (
            len(selected_by_class[class_label])
            >= SAMPLES_PER_CLASS
        ):
            continue

        modality_files = get_session_modality_files(
            session_id
        )

        missing_modalities = [
            modality_name
            for modality_name, files
            in modality_files.items()
            if not files
        ]

        if missing_modalities:
            print(
                f"Skipping incomplete session '{session_id}' "
                f"from class '{class_label}'. "
                f"Missing: {missing_modalities}"
            )
            continue

        selected_by_class[class_label].append(session_id)
        seen_session_ids.add(session_id)

    incomplete_classes = {
        class_label: session_ids
        for class_label, session_ids
        in selected_by_class.items()
        if len(session_ids) < SAMPLES_PER_CLASS
    }

    if incomplete_classes:
        details = "\n".join(
            f"  - {class_label}: "
            f"{len(session_ids)}/{SAMPLES_PER_CLASS}"
            for class_label, session_ids
            in incomplete_classes.items()
        )

        raise ValueError(
            "Insufficient complete aligned sessions for one or "
            "more required classes:\n"
            f"{details}\n\n"
            "Every selected session must have matching audio, "
            "image, keystroke and text files."
        )

    return selected_by_class


# ==============================================================
# Metadata construction
# ==============================================================

def build_sample_metadata(
    metadata_df: pd.DataFrame,
    keystroke_df: pd.DataFrame,
    selected_session_ids: list[str],
    metadata_id_column: str,
    keystroke_id_column: str,
) -> pd.DataFrame:
    """
    Build metadata for all selected sessions.

    Existing rows from metadata.csv are used when available.
    Missing rows are reconstructed from matching records in
    retroactive_keystroke_features.csv.
    """
    selected_id_set = set(selected_session_ids)

    metadata_lookup = {
        normalise_session_id(row[metadata_id_column]): row
        for _, row in metadata_df.iterrows()
        if normalise_session_id(row[metadata_id_column])
    }

    keystroke_lookup = {
        normalise_session_id(row[keystroke_id_column]): row
        for _, row in keystroke_df.iterrows()
        if normalise_session_id(row[keystroke_id_column])
    }

    metadata_columns = list(metadata_df.columns)
    reconstructed_rows: list[dict[str, object]] = []

    for session_id in selected_session_ids:
        if session_id not in selected_id_set:
            continue

        if session_id in metadata_lookup:
            source_row = metadata_lookup[session_id]
        else:
            source_row = keystroke_lookup[session_id]

        reconstructed_row = {
            column: source_row[column]
            if column in source_row.index
            else pd.NA
            for column in metadata_columns
        }

        reconstructed_rows.append(reconstructed_row)

    return pd.DataFrame(
        reconstructed_rows,
        columns=metadata_columns,
    )


# ==============================================================
# Output-directory helpers
# ==============================================================

def recreate_output_directory() -> None:
    """
    Delete and recreate sample_data/ to prevent stale files.
    """
    if OUTPUT_ROOT.exists():
        shutil.rmtree(OUTPUT_ROOT)

    OUTPUT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    for modality_name in MODALITY_DIRECTORIES:
        (OUTPUT_ROOT / modality_name).mkdir(
            parents=True,
            exist_ok=True,
        )


def calculate_output_statistics() -> tuple[int, int]:
    """Return total output-file count and total size in bytes."""
    output_files = [
        path
        for path in OUTPUT_ROOT.rglob("*")
        if path.is_file()
    ]

    total_bytes = sum(
        path.stat().st_size
        for path in output_files
    )

    return len(output_files), total_bytes


def validate_output_counts(
    selected_session_count: int,
) -> None:
    """
    Confirm that each modality contains at least one file per
    selected session.
    """
    problems: list[str] = []

    for modality_name in MODALITY_DIRECTORIES:
        file_count = sum(
            1
            for path
            in (OUTPUT_ROOT / modality_name).rglob("*")
            if path.is_file()
        )

        if file_count < selected_session_count:
            problems.append(
                f"{modality_name}: "
                f"{file_count} file(s); expected at least "
                f"{selected_session_count}"
            )

    if problems:
        formatted = "\n".join(
            f"  - {problem}"
            for problem in problems
        )

        raise RuntimeError(
            "Generated output failed modality validation:\n"
            f"{formatted}"
        )


# ==============================================================
# Main generation process
# ==============================================================

def create_sample_dataset() -> None:
    """Create the balanced multimodal sample dataset."""
    validate_configuration()
    validate_input_paths()

    print("Reading source CSV files...")

    metadata_df = pd.read_csv(METADATA_CSV)
    keystroke_df = pd.read_csv(
        KEYSTROKE_FEATURES_CSV
    )

    metadata_id_column = detect_session_id_column(
        metadata_df,
        METADATA_CSV.name,
    )

    keystroke_id_column = detect_session_id_column(
        keystroke_df,
        KEYSTROKE_FEATURES_CSV.name,
    )

    # The behavioural label is deliberately read from the
    # feature CSV because metadata.csv contains only fatigued rows.
    behavioural_class_column = (
        detect_behavioural_class_column(
            keystroke_df,
            KEYSTROKE_FEATURES_CSV.name,
        )
    )

    print(
        f"Metadata session-ID column: "
        f"{metadata_id_column}"
    )
    print(
        f"Feature session-ID column: "
        f"{keystroke_id_column}"
    )
    print(
        f"Behavioural-class source: "
        f"{KEYSTROKE_FEATURES_CSV.name}"
    )
    print(
        f"Behavioural-class column: "
        f"{behavioural_class_column}"
    )

    metadata_df["_normalised_session_id"] = (
        metadata_df[metadata_id_column]
        .map(normalise_session_id)
    )

    keystroke_df["_normalised_session_id"] = (
        keystroke_df[keystroke_id_column]
        .map(normalise_session_id)
    )

    keystroke_df["_normalised_class_label"] = (
        keystroke_df[behavioural_class_column]
        .map(normalise_class_label)
    )

    validate_available_classes(keystroke_df)

    selected_by_class = select_balanced_sessions(
        keystroke_df
    )

    selected_session_ids = [
        session_id
        for session_ids in selected_by_class.values()
        for session_id in session_ids
    ]

    selected_class_by_session = {
        session_id: class_label
        for class_label, session_ids
        in selected_by_class.items()
        for session_id in session_ids
    }

    print("\nBalanced selection:")

    for class_label, session_ids in selected_by_class.items():
        print(
            f"  {class_label}: "
            f"{len(session_ids)} session(s)"
        )

    print(
        f"\nTotal selected sessions: "
        f"{len(selected_session_ids)}"
    )

    recreate_output_directory()

    selected_id_set = set(selected_session_ids)

    sample_keystroke_df = keystroke_df[
        keystroke_df["_normalised_session_id"].isin(
            selected_id_set
        )
    ].copy()

    selection_order = {
        session_id: index
        for index, session_id
        in enumerate(selected_session_ids)
    }

    sample_keystroke_df["_selection_order"] = (
        sample_keystroke_df["_normalised_session_id"]
        .map(selection_order)
    )

    sample_keystroke_df.sort_values(
        "_selection_order",
        inplace=True,
    )

    sample_metadata_df = build_sample_metadata(
        metadata_df=metadata_df,
        keystroke_df=keystroke_df,
        selected_session_ids=selected_session_ids,
        metadata_id_column=metadata_id_column,
        keystroke_id_column=keystroke_id_column,
    )

    # Ensure labels in the output metadata reflect the authoritative
    # labels from the feature CSV.
    if "label" in sample_metadata_df.columns:
        sample_metadata_df["label"] = (
            sample_metadata_df[metadata_id_column]
            .map(
                lambda session_id: selected_class_by_session.get(
                    normalise_session_id(session_id),
                    "",
                )
            )
        )

    sample_metadata_df.drop(
        columns=["_normalised_session_id"],
        errors="ignore",
        inplace=True,
    )

    sample_keystroke_df.drop(
        columns=[
            "_normalised_session_id",
            "_normalised_class_label",
            "_selection_order",
        ],
        errors="ignore",
        inplace=True,
    )

    sample_metadata_df.to_csv(
        OUTPUT_ROOT / "metadata.csv",
        index=False,
    )

    sample_keystroke_df.to_csv(
        OUTPUT_ROOT
        / "retroactive_keystroke_features.csv",
        index=False,
    )

    print("\nCopying aligned modality files...")

    manifest_rows: list[dict[str, str | int]] = []

    for session_id in selected_session_ids:
        class_label = selected_class_by_session[
            session_id
        ]

        modality_files = get_session_modality_files(
            session_id
        )

        manifest_row: dict[str, str | int] = {
            "session_id": session_id,
            "behavioural_class": class_label,
        }

        for modality_name, source_directory in (
            MODALITY_DIRECTORIES.items()
        ):
            copied_files = (
                copy_files_preserving_relative_paths(
                    files=modality_files[modality_name],
                    source_directory=source_directory,
                    output_directory=(
                        OUTPUT_ROOT / modality_name
                    ),
                )
            )

            manifest_row[
                f"{modality_name}_file_count"
            ] = len(copied_files)

            manifest_row[
                f"{modality_name}_files"
            ] = ";".join(copied_files)

        manifest_rows.append(manifest_row)

    pd.DataFrame(manifest_rows).to_csv(
        OUTPUT_ROOT / "sample_manifest.csv",
        index=False,
    )

    selected_sessions_df = pd.DataFrame(
        [
            {
                "session_id": session_id,
                "behavioural_class":
                    selected_class_by_session[session_id],
            }
            for session_id in selected_session_ids
        ]
    )

    selected_sessions_df.to_csv(
        OUTPUT_ROOT / "selected_sessions.csv",
        index=False,
    )

    class_distribution_df = (
        selected_sessions_df
        .groupby(
            "behavioural_class",
            sort=False,
        )
        .size()
        .reset_index(name="sample_count")
    )

    class_distribution_df.to_csv(
        OUTPUT_ROOT / "class_distribution.csv",
        index=False,
    )

    validate_output_counts(
        selected_session_count=len(
            selected_session_ids
        )
    )

    file_count, total_bytes = (
        calculate_output_statistics()
    )

    total_megabytes = total_bytes / (1024 ** 2)
    total_gigabytes = total_bytes / (1024 ** 3)

    print("\nSample dataset created successfully.")
    print(f"Output directory: {OUTPUT_ROOT}")
    print(
        f"Metadata rows: "
        f"{len(sample_metadata_df)}"
    )
    print(
        f"Keystroke-feature rows: "
        f"{len(sample_keystroke_df)}"
    )

    print("\nBehavioural-class distribution:")

    for _, row in class_distribution_df.iterrows():
        print(
            f"  {row['behavioural_class']}: "
            f"{row['sample_count']}"
        )

    print("\nCopied modality-file counts:")

    for modality_name in MODALITY_DIRECTORIES:
        modality_file_count = sum(
            1
            for path
            in (OUTPUT_ROOT / modality_name).rglob("*")
            if path.is_file()
        )

        print(
            f"  {modality_name}: "
            f"{modality_file_count}"
        )

    print("\nOutput summary:")
    print(f"  Total files: {file_count}")
    print(f"  Total size: {total_megabytes:.2f} MB")
    print(f"  Total size: {total_gigabytes:.3f} GB")

    print(
        "\nReview sample_manifest.csv, selected_sessions.csv "
        "and class_distribution.csv before committing."
    )


def main() -> None:
    """Program entry point."""
    try:
        create_sample_dataset()
    except Exception as error:
        print(
            "\nSample creation failed:\n"
            f"{error}"
        )

        raise SystemExit(1) from error


if __name__ == "__main__":
    main()
