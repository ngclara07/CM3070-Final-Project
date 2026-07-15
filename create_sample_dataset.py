# filename: create_sample_dataset.py
#
# This script:
# - reads metadata.csv and retroactive_keystroke_features.csv;
# - identifies session identifiers present in both CSV files;
# - detects the behavioural-class column in metadata.csv;
# - selects the first five common sessions from each class;
# - creates sample_data/;
# - creates audio/, images/, keystrokes/, and texts/;
# - copies matching files from each modality;
# - saves filtered versions of both CSV files;
# - creates a manifest describing the selected sessions and files.

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

SAMPLES_PER_CLASS = 5

# Set this to 4 when the intended sample must contain exactly
# four behavioural classes and 20 sessions in total.
EXPECTED_CLASS_COUNT = 4

MODALITY_DIRECTORIES = {
    "audio": SOURCE_ROOT / "audio",
    "images": SOURCE_ROOT / "images",
    "keystrokes": SOURCE_ROOT / "keystrokes",
    "texts": SOURCE_ROOT / "texts",
}

SESSION_ID_COLUMN_CANDIDATES = (
    "session_id",
    "session",
    "session_name",
    "session_identifier",
    "recording_id",
    "sample_id",
    "id",
)

BEHAVIOURAL_CLASS_COLUMN_CANDIDATES = (
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
        f"in {csv_name}.\n"
        f"Available columns: {list(dataframe.columns)}\n\n"
        f"Add the correct column name to the appropriate "
        f"candidate list near the top of this script."
    )


def detect_session_id_column(
    dataframe: pd.DataFrame,
    csv_name: str,
) -> str:
    """
    Identify the session-ID column.
    """
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
    """
    Identify the behavioural-class column.
    """
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
    """
    Convert a session identifier into a consistent string.
    """
    if pd.isna(value):
        return ""

    return str(value).strip()


def normalise_class_label(value: object) -> str:
    """
    Convert a behavioural-class value into a consistent string.
    """
    if pd.isna(value):
        return ""

    return str(value).strip()


# ==============================================================
# File-matching helpers
# ==============================================================

def filename_matches_session(
    file_path: Path,
    session_id: str,
) -> bool:
    """
    Determine whether a filename belongs to a session.

    Matching supports filenames where the session identifier:
    - is the complete filename stem;
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
    """
    Find all files in a modality directory matching a session ID.
    """
    if not source_directory.exists():
        raise FileNotFoundError(
            f"Required source directory does not exist: "
            f"{source_directory}"
        )

    return sorted(
        path
        for path in source_directory.rglob("*")
        if path.is_file()
        and filename_matches_session(path, session_id)
    )


def session_has_all_modalities(session_id: str) -> bool:
    """
    Return True only when the session has at least one file
    in every required modality directory.
    """
    return all(
        find_matching_files(source_directory, session_id)
        for source_directory in MODALITY_DIRECTORIES.values()
    )


def copy_files_preserving_relative_paths(
    files: Iterable[Path],
    source_directory: Path,
    output_directory: Path,
) -> list[str]:
    """
    Copy files while preserving any source subdirectory structure.
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

def validate_input_paths() -> None:
    """
    Confirm that all required source files and directories exist.
    """
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


# ==============================================================
# Balanced-session selection
# ==============================================================

def select_balanced_sessions(
    metadata_df: pd.DataFrame,
    common_session_ids: set[str],
    class_column: str,
) -> OrderedDict[str, list[str]]:
    """
    Select the first SAMPLES_PER_CLASS complete sessions from each
    behavioural class.

    Ordering follows metadata.csv.
    """
    selected_by_class: OrderedDict[str, list[str]] = OrderedDict()
    seen_session_ids: set[str] = set()

    for _, row in metadata_df.iterrows():
        session_id = row["_normalised_session_id"]
        class_label = row["_normalised_class_label"]

        if not session_id or not class_label:
            continue

        if session_id not in common_session_ids:
            continue

        if session_id in seen_session_ids:
            continue

        selected_for_class = selected_by_class.setdefault(
            class_label,
            [],
        )

        if len(selected_for_class) >= SAMPLES_PER_CLASS:
            continue

        if not session_has_all_modalities(session_id):
            print(
                "Skipping incomplete session "
                f"'{session_id}' from class '{class_label}'."
            )
            continue

        selected_for_class.append(session_id)
        seen_session_ids.add(session_id)

    incomplete_classes = {
        class_label: session_ids
        for class_label, session_ids in selected_by_class.items()
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
            "Unable to select the required number of complete "
            "sessions for every behavioural class:\n"
            f"{details}\n\n"
            "Confirm that each class has sufficient sessions "
            "present in both CSV files and all four modality "
            "directories."
        )

    if EXPECTED_CLASS_COUNT is not None:
        actual_class_count = len(selected_by_class)

        if actual_class_count != EXPECTED_CLASS_COUNT:
            raise ValueError(
                f"Expected {EXPECTED_CLASS_COUNT} behavioural "
                f"classes, but detected {actual_class_count}: "
                f"{list(selected_by_class.keys())}\n\n"
                "Review the behavioural-class column or update "
                "EXPECTED_CLASS_COUNT in the configuration."
            )

    return selected_by_class


# ==============================================================
# Main process
# ==============================================================

def create_sample_dataset() -> None:
    """
    Create a balanced multimodal demonstration dataset.
    """
    validate_input_paths()

    print("Reading source CSV files...")

    metadata_df = pd.read_csv(METADATA_CSV)
    keystroke_df = pd.read_csv(
        KEYSTROKE_FEATURES_CSV,
    )

    metadata_id_column = detect_session_id_column(
        metadata_df,
        METADATA_CSV.name,
    )

    keystroke_id_column = detect_session_id_column(
        keystroke_df,
        KEYSTROKE_FEATURES_CSV.name,
    )

    behavioural_class_column = (
        detect_behavioural_class_column(
            metadata_df,
            METADATA_CSV.name,
        )
    )

    print(
        f"Metadata session-ID column: "
        f"{metadata_id_column}"
    )
    print(
        f"Keystroke session-ID column: "
        f"{keystroke_id_column}"
    )
    print(
        f"Behavioural-class column: "
        f"{behavioural_class_column}"
    )

    metadata_df["_normalised_session_id"] = (
        metadata_df[metadata_id_column]
        .map(normalise_session_id)
    )

    metadata_df["_normalised_class_label"] = (
        metadata_df[behavioural_class_column]
        .map(normalise_class_label)
    )

    keystroke_df["_normalised_session_id"] = (
        keystroke_df[keystroke_id_column]
        .map(normalise_session_id)
    )

    metadata_session_ids = set(
        metadata_df["_normalised_session_id"]
    )

    keystroke_session_ids = set(
        keystroke_df["_normalised_session_id"]
    )

    common_session_ids = (
        metadata_session_ids
        & keystroke_session_ids
    )

    common_session_ids.discard("")

    if not common_session_ids:
        raise ValueError(
            "No common session identifiers were found between "
            "metadata.csv and "
            "retroactive_keystroke_features.csv."
        )

    print(
        f"Common sessions found: "
        f"{len(common_session_ids)}"
    )

    selected_by_class = select_balanced_sessions(
        metadata_df=metadata_df,
        common_session_ids=common_session_ids,
        class_column=behavioural_class_column,
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
            f"{len(session_ids)} sessions"
        )

    print(
        f"\nTotal selected sessions: "
        f"{len(selected_session_ids)}"
    )

    # Recreate the output directory to prevent stale files from
    # previous script executions.
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

    selected_id_set = set(selected_session_ids)

    sample_metadata_df = metadata_df[
        metadata_df["_normalised_session_id"].isin(
            selected_id_set
        )
    ].copy()

    sample_keystroke_df = keystroke_df[
        keystroke_df["_normalised_session_id"].isin(
            selected_id_set
        )
    ].copy()

    # Preserve the balanced selection order in the output CSVs.
    selection_order = {
        session_id: index
        for index, session_id
        in enumerate(selected_session_ids)
    }

    sample_metadata_df["_selection_order"] = (
        sample_metadata_df["_normalised_session_id"]
        .map(selection_order)
    )

    sample_keystroke_df["_selection_order"] = (
        sample_keystroke_df["_normalised_session_id"]
        .map(selection_order)
    )

    sample_metadata_df.sort_values(
        "_selection_order",
        inplace=True,
    )

    sample_keystroke_df.sort_values(
        "_selection_order",
        inplace=True,
    )

    sample_metadata_df.drop(
        columns=[
            "_normalised_session_id",
            "_normalised_class_label",
            "_selection_order",
        ],
        inplace=True,
    )

    sample_keystroke_df.drop(
        columns=[
            "_normalised_session_id",
            "_selection_order",
        ],
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

    manifest_rows: list[dict[str, str | int]] = []

    print("\nCopying aligned modality files...")

    for session_id in selected_session_ids:
        class_label = selected_class_by_session[
            session_id
        ]

        manifest_row: dict[str, str | int] = {
            "session_id": session_id,
            "behavioural_class": class_label,
        }

        for modality_name, source_directory in (
            MODALITY_DIRECTORIES.items()
        ):
            matching_files = find_matching_files(
                source_directory,
                session_id,
            )

            copied_files = (
                copy_files_preserving_relative_paths(
                    matching_files,
                    source_directory,
                    OUTPUT_ROOT / modality_name,
                )
            )

            manifest_row[
                f"{modality_name}_file_count"
            ] = len(copied_files)

            manifest_row[
                f"{modality_name}_files"
            ] = ";".join(copied_files)

        manifest_rows.append(manifest_row)

    manifest_df = pd.DataFrame(manifest_rows)

    manifest_df.to_csv(
        OUTPUT_ROOT / "sample_manifest.csv",
        index=False,
    )

    selected_sessions_rows = [
        {
            "session_id": session_id,
            "behavioural_class":
                selected_class_by_session[session_id],
        }
        for session_id in selected_session_ids
    ]

    selected_sessions_df = pd.DataFrame(
        selected_sessions_rows
    )

    selected_sessions_df.to_csv(
        OUTPUT_ROOT / "selected_sessions.csv",
        index=False,
    )

    class_summary_df = (
        selected_sessions_df
        .groupby(
            "behavioural_class",
            sort=False,
        )
        .size()
        .reset_index(name="sample_count")
    )

    class_summary_df.to_csv(
        OUTPUT_ROOT / "class_distribution.csv",
        index=False,
    )

    print("\nSample dataset created successfully.")
    print(f"Output directory: {OUTPUT_ROOT}")

    print(
        f"Metadata rows: "
        f"{len(sample_metadata_df)}"
    )

    print(
        "Keystroke-feature rows: "
        f"{len(sample_keystroke_df)}"
    )

    print("\nBehavioural-class distribution:")

    for _, row in class_summary_df.iterrows():
        print(
            f"  {row['behavioural_class']}: "
            f"{row['sample_count']}"
        )

    print("\nCopied file counts:")

    for modality_name in MODALITY_DIRECTORIES:
        file_count = sum(
            1
            for path
            in (OUTPUT_ROOT / modality_name).rglob("*")
            if path.is_file()
        )

        print(
            f"  {modality_name}: "
            f"{file_count}"
        )

    print(
        "\nReview sample_manifest.csv and "
        "class_distribution.csv before committing."
    )


if __name__ == "__main__":
    try:
        create_sample_dataset()
    except Exception as error:
        print(
            f"\nSample creation failed: "
            f"{error}"
        )
        raise SystemExit(1) from error
