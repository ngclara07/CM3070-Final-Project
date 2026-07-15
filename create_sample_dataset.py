# filename: create_sample_dataset.py
# this python script will:
## - read both CSV files from "data/session_aligned/";
## - identify session identifiers appearing in both CSV files;
## - select the first 20 common sessions;
## - create "sample/data";
## - create "audio/", "image/", "keystrokes/", and "text/";
## - copy matching files from each modality;
## - save filtered versions of both CSV files;
## - create a manifest describing what was copied 


from __future__ import annotations

import shutil
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
KEYSTROKE_FEATURES_CSV = SOURCE_ROOT / "retroactive_keystroke_features.csv"

SAMPLE_SIZE = 20

MODALITY_DIRECTORIES = {
    "audio": SOURCE_ROOT / "audio",
    "images": SOURCE_ROOT / "images",
    "keystrokes": SOURCE_ROOT / "keystrokes",
    "texts": SOURCE_ROOT / "texts",
}

# The script checks these names in order.
SESSION_ID_COLUMN_CANDIDATES = (
    "session_id",
    "session",
    "session_name",
    "session_identifier",
    "recording_id",
    "sample_id",
    "id",
)


# ==============================================================
# Helper functions
# ==============================================================

def detect_session_id_column(
    dataframe: pd.DataFrame,
    csv_name: str,
) -> str:
    """
    Identify the session-ID column using common candidate names.
    """
    normalised_columns = {
        str(column).strip().lower(): str(column)
        for column in dataframe.columns
    }

    for candidate in SESSION_ID_COLUMN_CANDIDATES:
        if candidate in normalised_columns:
            return normalised_columns[candidate]

    raise ValueError(
        f"Unable to identify the session-ID column in {csv_name}.\n"
        f"Available columns: {list(dataframe.columns)}\n\n"
        "Add the correct column name to "
        "SESSION_ID_COLUMN_CANDIDATES near the top of the script."
    )


def normalise_session_id(value: object) -> str:
    """
    Convert a session identifier into a consistent string.
    """
    return str(value).strip()


def filename_matches_session(
    file_path: Path,
    session_id: str,
) -> bool:
    """
    Determine whether a filename belongs to a session.

    This supports filenames where the session identifier:
    - is the complete filename stem;
    - appears at the beginning of the filename; or
    - appears elsewhere in a descriptive long filename.
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

    return [
        path
        for path in source_directory.rglob("*")
        if path.is_file()
        and filename_matches_session(path, session_id)
    ]


def copy_files_preserving_relative_paths(
    files: Iterable[Path],
    source_directory: Path,
    output_directory: Path,
) -> list[str]:
    """
    Copy files while preserving any subdirectory structure.
    """
    copied_paths: list[str] = []

    for source_file in files:
        relative_path = source_file.relative_to(source_directory)
        destination_file = output_directory / relative_path

        destination_file.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_file, destination_file)

        copied_paths.append(
            destination_file.relative_to(OUTPUT_ROOT).as_posix()
        )

    return copied_paths


def validate_input_paths() -> None:
    """
    Confirm that all required inputs exist.
    """
    required_paths = [
        METADATA_CSV,
        KEYSTROKE_FEATURES_CSV,
        *MODALITY_DIRECTORIES.values(),
    ]

    missing_paths = [
        path for path in required_paths
        if not path.exists()
    ]

    if missing_paths:
        formatted = "\n".join(
            f"  - {path}" for path in missing_paths
        )
        raise FileNotFoundError(
            "The following required paths were not found:\n"
            f"{formatted}"
        )


# ==============================================================
# Main process
# ==============================================================

def create_sample_dataset() -> None:
    validate_input_paths()

    print("Reading source CSV files...")

    metadata_df = pd.read_csv(METADATA_CSV)
    keystroke_df = pd.read_csv(KEYSTROKE_FEATURES_CSV)

    metadata_id_column = detect_session_id_column(
        metadata_df,
        METADATA_CSV.name,
    )
    keystroke_id_column = detect_session_id_column(
        keystroke_df,
        KEYSTROKE_FEATURES_CSV.name,
    )

    metadata_df["_normalised_session_id"] = (
        metadata_df[metadata_id_column]
        .map(normalise_session_id)
    )

    keystroke_df["_normalised_session_id"] = (
        keystroke_df[keystroke_id_column]
        .map(normalise_session_id)
    )

    metadata_session_ids = list(
        dict.fromkeys(
            metadata_df["_normalised_session_id"].tolist()
        )
    )

    keystroke_session_ids = set(
        keystroke_df["_normalised_session_id"].tolist()
    )

    # Preserve the ordering from metadata.csv.
    common_session_ids = [
        session_id
        for session_id in metadata_session_ids
        if session_id in keystroke_session_ids
    ]

    if not common_session_ids:
        raise ValueError(
            "No common session identifiers were found between "
            "metadata.csv and retroactive_keystroke_features.csv."
        )

    selected_session_ids = common_session_ids[:SAMPLE_SIZE]

    if len(selected_session_ids) < SAMPLE_SIZE:
        print(
            f"Warning: only {len(selected_session_ids)} common "
            f"sessions were available; requested {SAMPLE_SIZE}."
        )

    print(
        f"Selected {len(selected_session_ids)} common sessions."
    )

    # Recreate the sample directory to prevent stale files.
    if OUTPUT_ROOT.exists():
        shutil.rmtree(OUTPUT_ROOT)

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    for modality_name in MODALITY_DIRECTORIES:
        (OUTPUT_ROOT / modality_name).mkdir(
            parents=True,
            exist_ok=True,
        )

    selected_id_set = set(selected_session_ids)

    sample_metadata_df = metadata_df[
        metadata_df["_normalised_session_id"].isin(selected_id_set)
    ].copy()

    sample_keystroke_df = keystroke_df[
        keystroke_df["_normalised_session_id"].isin(selected_id_set)
    ].copy()

    sample_metadata_df.drop(
        columns=["_normalised_session_id"],
        inplace=True,
    )
    sample_keystroke_df.drop(
        columns=["_normalised_session_id"],
        inplace=True,
    )

    sample_metadata_df.to_csv(
        OUTPUT_ROOT / "metadata.csv",
        index=False,
    )

    sample_keystroke_df.to_csv(
        OUTPUT_ROOT / "retroactive_keystroke_features.csv",
        index=False,
    )

    manifest_rows: list[dict[str, str | int]] = []

    print("Copying aligned modality files...")

    for session_id in selected_session_ids:
        manifest_row: dict[str, str | int] = {
            "session_id": session_id,
        }

        for modality_name, source_directory in (
            MODALITY_DIRECTORIES.items()
        ):
            matching_files = find_matching_files(
                source_directory,
                session_id,
            )

            copied_files = copy_files_preserving_relative_paths(
                matching_files,
                source_directory,
                OUTPUT_ROOT / modality_name,
            )

            manifest_row[f"{modality_name}_file_count"] = (
                len(copied_files)
            )
            manifest_row[f"{modality_name}_files"] = ";".join(
                copied_files
            )

            if not copied_files:
                print(
                    f"Warning: no {modality_name} file was found "
                    f"for session '{session_id}'."
                )

        manifest_rows.append(manifest_row)

    manifest_df = pd.DataFrame(manifest_rows)
    manifest_df.to_csv(
        OUTPUT_ROOT / "sample_manifest.csv",
        index=False,
    )

    selected_sessions_df = pd.DataFrame(
        {"session_id": selected_session_ids}
    )
    selected_sessions_df.to_csv(
        OUTPUT_ROOT / "selected_sessions.csv",
        index=False,
    )

    print("\nSample dataset created successfully.")
    print(f"Output directory: {OUTPUT_ROOT}")
    print(
        f"Metadata rows: {len(sample_metadata_df)}"
    )
    print(
        "Keystroke-feature rows: "
        f"{len(sample_keystroke_df)}"
    )

    print("\nCopied file counts:")
    for modality_name in MODALITY_DIRECTORIES:
        file_count = sum(
            1
            for path in (OUTPUT_ROOT / modality_name).rglob("*")
            if path.is_file()
        )
        print(f"  {modality_name}: {file_count}")

    print(
        "\nReview sample_manifest.csv for missing or duplicate "
        "modality files before committing."
    )


if __name__ == "__main__":
    try:
        create_sample_dataset()
    except Exception as error:
        print(f"\nSample creation failed: {error}")
        raise SystemExit(1) from error
