"""
test_05_keystroke_dataset_comparison.py

SenseFuzeAI
Keystroke Dataset Comparison / Harmonisation Verification

=============================================================================
PURPOSE
=============================================================================

Tests the supervisor-requested EmoSurv IEEE versus SenseFuzeAI keystroke
comparison pipeline.

The tests focus on:

1. Common 23-feature harmonisation contract.
2. Dataset provenance and source separation.
3. Three-class primary and four-class exploratory label contracts.
4. Combined datasets being exact unions of source datasets.
5. Unique sample identifiers.
6. Numeric feature validity.
7. Participant/session metadata integrity.
8. Frozen train/test split manifests.
9. Participant/session leakage prevention.
10. Baseline-versus-augmented experiments using identical held-out samples.
11. Comparison summaries containing all six canonical experiments.
12. Source-code contracts for the builder and trainer.

IMPORTANT
=============================================================================

These tests DO NOT retrain the models.

They inspect source contracts and already-generated derived experiment
artifacts. If derived artifacts have not yet been generated, the relevant
artifact tests are skipped with an explanatory message.

Original source datasets are never modified by this test module.
"""

from __future__ import annotations

import ast
import json
import math

from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import pytest


# =============================================================================
# PROJECT PATHS
# =============================================================================

ROOT_DIR = (
    Path(__file__)
    .resolve()
    .parents[1]
)

BUILDER_FILE = (
    ROOT_DIR
    / "build_keystroke_dataset_comparison.py"
)

TRAINER_FILE = (
    ROOT_DIR
    / "train_keystroke_dataset_comparison.py"
)

EMOSURV_IMPLEMENTATION_FILE = (
    ROOT_DIR
    / "keystroke_live_gui_emosurv_ieee.py"
)

OUTPUT_DIR = (
    ROOT_DIR
    / "data"
    / "processed"
    / "keystroke_dataset_comparison"
)

EMOSURV_3_PATH = (
    OUTPUT_DIR
    / "emosurv_harmonised_3class.csv"
)

SENSEFUZE_3_PATH = (
    OUTPUT_DIR
    / "sensefuzeai_harmonised_3class.csv"
)

COMBINED_3_PATH = (
    OUTPUT_DIR
    / "combined_harmonised_3class.csv"
)

EMOSURV_4_PATH = (
    OUTPUT_DIR
    / "emosurv_harmonised_4class.csv"
)

SENSEFUZE_4_PATH = (
    OUTPUT_DIR
    / "sensefuzeai_harmonised_4class.csv"
)

COMBINED_4_PATH = (
    OUTPUT_DIR
    / "combined_harmonised_4class.csv"
)

FEATURE_SCHEMA_PATH = (
    OUTPUT_DIR
    / "harmonised_feature_columns.json"
)

BUILD_METADATA_PATH = (
    OUTPUT_DIR
    / "build_metadata.json"
)

THREE_RESULTS_DIR = (
    OUTPUT_DIR
    / "results"
    / "three_class_primary"
)

FOUR_RESULTS_DIR = (
    OUTPUT_DIR
    / "results"
    / "four_class_exploratory"
)


# =============================================================================
# CANONICAL EXPECTATIONS
# =============================================================================

EXPECTED_FEATURE_COLUMNS = [
    "dwell_mean_ms",
    "dwell_std_ms",
    "dwell_median_ms",
    "dwell_min_ms",
    "dwell_max_ms",

    "dd_mean_ms",
    "dd_std_ms",
    "dd_median_ms",
    "dd_min_ms",
    "dd_max_ms",

    "ud_mean_ms",
    "ud_std_ms",
    "ud_median_ms",
    "ud_min_ms",
    "ud_max_ms",

    "typing_speed_kps",

    "pause_ratio_500",
    "pause_ratio_1000",
    "pause_ratio_2000",

    "correction_ratio",
    "space_ratio",

    "rhythm_cv",
    "overlap_ratio",
]

EXPECTED_METADATA_COLUMNS = [
    "dataset_source",
    "participant_id",
    "session_id",
    "sample_id",
    "source_type",
    "label",
    "label_origin",
    "original_label",
    "window_index",
    "window_keystrokes",
]

THREE_CLASS_LABELS = {
    "focused",
    "fatigued",
    "overloaded",
}

FOUR_CLASS_LABELS = {
    "focused",
    "distracted",
    "fatigued",
    "overloaded",
}

EXPECTED_EXPERIMENTS = {
    "A_emosurv_baseline",
    "B_sensefuzeai_only",
    "C_augmented_to_emosurv_test",
    "D_emosurv_to_sensefuzeai",
    "E_augmented_to_sensefuzeai_test",
    "F_sensefuzeai_to_emosurv",
}

EXPECTED_DATASET_SOURCES = {
    "emosurv",
    "sensefuzeai",
}

EXPECTED_WINDOW_MIN = 20
EXPECTED_WINDOW_MAX = 40
EXPECTED_RANDOM_STATE = 42
EXPECTED_TEST_SIZE = 0.20


# =============================================================================
# HELPERS
# =============================================================================

def require_file(
    path: Path,
) -> Path:
    assert path.exists(), (
        f"Required file does not exist:\n{path}"
    )

    assert path.is_file(), (
        f"Expected a file but found something else:\n{path}"
    )

    return path


def require_derived_files(
    paths: Iterable[Path],
) -> None:
    missing = [
        path
        for path in paths
        if not path.exists()
    ]

    if missing:
        pytest.skip(
            "Derived keystroke-comparison artifacts are not available. "
            "Run build_keystroke_dataset_comparison.py and "
            "train_keystroke_dataset_comparison.py first.\n"
            + "\n".join(
                str(path)
                for path in missing
            )
        )


def read_source(
    path: Path,
) -> str:
    require_file(
        path
    )

    return path.read_text(
        encoding="utf-8"
    )


def parse_source(
    path: Path,
) -> ast.Module:
    source = read_source(
        path
    )

    return ast.parse(
        source,
        filename=str(path),
    )


def imported_names_from(
    tree: ast.Module,
    module_name: str,
) -> set[str]:
    output: set[str] = set()

    for node in ast.walk(
        tree
    ):
        if not isinstance(
            node,
            ast.ImportFrom,
        ):
            continue

        if node.module != module_name:
            continue

        for alias in node.names:
            output.add(
                alias.name
            )

    return output


def dotted_name(
    node: ast.AST,
) -> str | None:
    if isinstance(
        node,
        ast.Name,
    ):
        return node.id

    if isinstance(
        node,
        ast.Attribute,
    ):
        prefix = dotted_name(
            node.value
        )

        if prefix:
            return (
                f"{prefix}.{node.attr}"
            )

        return node.attr

    return None


def contains_named_call(
    tree: ast.Module,
    function_name: str,
) -> bool:
    for node in ast.walk(
        tree
    ):
        if not isinstance(
            node,
            ast.Call,
        ):
            continue

        name = dotted_name(
            node.func
        )

        if name is None:
            continue

        if (
            name == function_name
            or
            name.endswith(
                "." + function_name
            )
        ):
            return True

    return False


def read_csv(
    path: Path,
) -> pd.DataFrame:
    require_file(
        path
    )

    return pd.read_csv(
        path
    )


def read_json(
    path: Path,
) -> Any:
    require_file(
        path
    )

    return json.loads(
        path.read_text(
            encoding="utf-8"
        )
    )


def assert_nonblank_series(
    series: pd.Series,
    *,
    name: str,
) -> None:
    values = (
        series
        .fillna("")
        .astype(str)
        .str.strip()
    )

    invalid = (
        values
        .str.lower()
        .isin(
            {
                "",
                "nan",
                "none",
                "null",
            }
        )
    )

    assert not invalid.any(), (
        f"{name} contains blank or invalid identifiers."
    )


def assert_numeric_features_valid(
    dataframe: pd.DataFrame,
) -> None:
    for feature in EXPECTED_FEATURE_COLUMNS:
        converted = pd.to_numeric(
            dataframe[feature],
            errors="coerce",
        )

        original_non_null = (
            dataframe[feature]
            .notna()
        )

        conversion_failed = (
            original_non_null
            & converted.isna()
        )

        assert not conversion_failed.any(), (
            f"Feature {feature!r} contains "
            "non-numeric non-null values."
        )

        finite_values = (
            converted
            .dropna()
            .to_numpy(
                dtype=float
            )
        )

        assert np.isfinite(
            finite_values
        ).all(), (
            f"Feature {feature!r} contains "
            "infinite values."
        )


def assert_dataset_contract(
    dataframe: pd.DataFrame,
    *,
    allowed_labels: set[str],
    expected_sources: set[str],
) -> None:
    assert not dataframe.empty, (
        "Dataset must not be empty."
    )

    required_columns = set(
        EXPECTED_METADATA_COLUMNS
        + EXPECTED_FEATURE_COLUMNS
    )

    missing = (
        required_columns
        - set(
            dataframe.columns
        )
    )

    assert not missing, (
        "Dataset is missing required columns:\n"
        f"{sorted(missing)}"
    )

    assert_nonblank_series(
        dataframe["sample_id"],
        name="sample_id",
    )

    assert_nonblank_series(
        dataframe["session_id"],
        name="session_id",
    )

    assert_nonblank_series(
        dataframe["participant_id"],
        name="participant_id",
    )

    assert not (
        dataframe[
            "sample_id"
        ]
        .duplicated()
        .any()
    ), (
        "Duplicate sample_id values detected."
    )

    observed_labels = set(
        dataframe[
            "label"
        ]
        .astype(str)
    )

    assert observed_labels.issubset(
        allowed_labels
    ), (
        "Unexpected labels detected:\n"
        f"{sorted(observed_labels - allowed_labels)}"
    )

    observed_sources = set(
        dataframe[
            "dataset_source"
        ]
        .astype(str)
    )

    assert observed_sources == expected_sources, (
        "Unexpected dataset_source values.\n"
        f"Expected: {sorted(expected_sources)}\n"
        f"Actual:   {sorted(observed_sources)}"
    )

    window_sizes = pd.to_numeric(
        dataframe[
            "window_keystrokes"
        ],
        errors="coerce",
    )

    assert window_sizes.notna().all(), (
        "window_keystrokes contains invalid values."
    )

    assert (
        window_sizes
        >= EXPECTED_WINDOW_MIN
    ).all(), (
        "A harmonised observation contains fewer than "
        f"{EXPECTED_WINDOW_MIN} keystrokes."
    )

    assert (
        window_sizes
        <= EXPECTED_WINDOW_MAX
    ).all(), (
        "A harmonised observation contains more than "
        f"{EXPECTED_WINDOW_MAX} keystrokes."
    )

    assert_numeric_features_valid(
        dataframe
    )


def flatten_list_fields(
    value: Any,
    prefix: str = "",
) -> list[
    tuple[
        str,
        list[Any],
    ]
]:
    output: list[
        tuple[
            str,
            list[Any],
        ]
    ] = []

    if isinstance(
        value,
        dict,
    ):
        for key, child in value.items():
            child_prefix = (
                f"{prefix}.{key}"
                if prefix
                else str(key)
            )

            output.extend(
                flatten_list_fields(
                    child,
                    child_prefix,
                )
            )

    elif isinstance(
        value,
        list,
    ):
        output.append(
            (
                prefix,
                value,
            )
        )

    return output


def collect_manifest_values(
    manifest: Any,
    *,
    required_tokens: tuple[str, ...],
    any_tokens: tuple[str, ...] = (),
) -> set[str]:
    output: set[str] = set()

    for path, values in flatten_list_fields(
        manifest
    ):
        lowered_path = (
            path.lower()
        )

        if not all(
            token.lower()
            in lowered_path
            for token
            in required_tokens
        ):
            continue

        if (
            any_tokens
            and not any(
                token.lower()
                in lowered_path
                for token
                in any_tokens
            )
        ):
            continue

        for value in values:
            if isinstance(
                value,
                (
                    str,
                    int,
                    float,
                ),
            ):
                text = str(
                    value
                ).strip()

                if (
                    text
                    and text.lower()
                    not in {
                        "nan",
                        "none",
                        "null",
                    }
                ):
                    output.add(
                        text
                    )

    return output


def recursively_find_scalar(
    value: Any,
    key_name: str,
) -> list[Any]:
    output: list[Any] = []

    if isinstance(
        value,
        dict,
    ):
        for key, child in value.items():
            if (
                str(key)
                .lower()
                == key_name.lower()
            ):
                output.append(
                    child
                )

            output.extend(
                recursively_find_scalar(
                    child,
                    key_name,
                )
            )

    elif isinstance(
        value,
        list,
    ):
        for child in value:
            output.extend(
                recursively_find_scalar(
                    child,
                    key_name,
                )
            )

    return output


def find_prediction_csv(
    results_dir: Path,
    experiment_name: str,
) -> Path | None:
    experiment_dir = (
        results_dir
        / experiment_name
    )

    if not experiment_dir.exists():
        return None

    candidates = sorted(
        experiment_dir.rglob(
            "*.csv"
        )
    )

    preferred = [
        path
        for path in candidates
        if "prediction"
        in path.name.lower()
    ]

    for path in (
        preferred
        + candidates
    ):
        try:
            dataframe = pd.read_csv(
                path,
                nrows=5,
            )
        except Exception:
            continue

        if "sample_id" in dataframe.columns:
            return path

    return None


# =============================================================================
# SOURCE-CONTRACT TESTS
# =============================================================================

def test_builder_uses_canonical_emosurv_feature_extraction() -> None:
    tree = parse_source(
        BUILDER_FILE
    )

    imported = imported_names_from(
        tree,
        "keystroke_live_gui_emosurv_ieee",
    )

    required_imports = {
        "FEATURE_COLUMNS",
        "WINDOW_SIZE",
        "WINDOW_STEP",
        "MIN_WINDOW_SIZE",
        "extract_live_features",
        "build_window_dataset",
        "load_emosurv_datasets",
        "assign_behaviour_proxy_labels",
    }

    missing = (
        required_imports
        - imported
    )

    assert not missing, (
        "Dataset builder is no longer importing the "
        "canonical EmoSurv feature/window implementation:\n"
        f"{sorted(missing)}"
    )

    assert contains_named_call(
        tree,
        "extract_live_features",
    ), (
        "SenseFuzeAI harmonisation no longer appears "
        "to call extract_live_features()."
    )

    source = read_source(
        BUILDER_FILE
    )

    required_terms = {
        "combined_harmonised_3class.csv",
        "combined_harmonised_4class.csv",
        "dataset_source",
        "participant_id",
        "session_id",
        "sample_id",
        "label_origin",
    }

    missing_terms = {
        term
        for term in required_terms
        if term not in source
    }

    assert not missing_terms, (
        "Builder source is missing required "
        "harmonisation/provenance concepts:\n"
        f"{sorted(missing_terms)}"
    )


def test_trainer_contains_group_aware_frozen_split_contract() -> None:
    source = read_source(
        TRAINER_FILE
    )

    required_terms = {
        "A_emosurv_baseline",
        "B_sensefuzeai_only",
        "C_augmented_to_emosurv_test",
        "D_emosurv_to_sensefuzeai",
        "E_augmented_to_sensefuzeai_test",
        "F_sensefuzeai_to_emosurv",
        "participant_id",
        "session_id",
        "split_manifest",
        "macro_f1",
        "balanced_accuracy",
        "--rebuild-splits",
    }

    missing_terms = {
        term
        for term in required_terms
        if term not in source
    }

    assert not missing_terms, (
        "Training comparison source is missing "
        "required experiment/split concepts:\n"
        f"{sorted(missing_terms)}"
    )

    assert (
        "GroupShuffleSplit"
        in source
        or
        "StratifiedGroupKFold"
        in source
        or
        "group-aware"
        in source.lower()
        or
        "group_aware"
        in source.lower()
    ), (
        "Training comparison script does not appear "
        "to implement group-aware splitting."
    )

    assert (
        "bootstrap"
        in source.lower()
    ), (
        "Training comparison script no longer appears "
        "to implement the paired bootstrap analysis."
    )

    assert (
        "train_test_split"
        not in source
    ), (
        "Row-level train_test_split appears in the "
        "dataset-comparison trainer. Group-aware "
        "splitting should be retained."
    )


# =============================================================================
# HARMONISATION-ARTIFACT TESTS
# =============================================================================

def test_harmonised_feature_schema_is_exact_23_feature_contract() -> None:
    require_derived_files(
        [
            FEATURE_SCHEMA_PATH,
        ]
    )

    schema = read_json(
        FEATURE_SCHEMA_PATH
    )

    assert isinstance(
        schema,
        list,
    )

    assert schema == EXPECTED_FEATURE_COLUMNS, (
        "harmonised_feature_columns.json does not "
        "match the canonical 23-feature schema."
    )

    assert len(
        schema
    ) == 23

    metadata_overlap = (
        set(schema)
        & set(
            EXPECTED_METADATA_COLUMNS
        )
    )

    assert not metadata_overlap, (
        "Metadata columns have leaked into the "
        "classifier feature schema:\n"
        f"{sorted(metadata_overlap)}"
    )


@pytest.mark.parametrize(
    (
        "path",
        "labels",
        "sources",
    ),
    [
        (
            EMOSURV_3_PATH,
            THREE_CLASS_LABELS,
            {"emosurv"},
        ),
        (
            SENSEFUZE_3_PATH,
            THREE_CLASS_LABELS,
            {"sensefuzeai"},
        ),
        (
            COMBINED_3_PATH,
            THREE_CLASS_LABELS,
            EXPECTED_DATASET_SOURCES,
        ),
        (
            EMOSURV_4_PATH,
            FOUR_CLASS_LABELS,
            {"emosurv"},
        ),
        (
            SENSEFUZE_4_PATH,
            FOUR_CLASS_LABELS,
            {"sensefuzeai"},
        ),
        (
            COMBINED_4_PATH,
            FOUR_CLASS_LABELS,
            EXPECTED_DATASET_SOURCES,
        ),
    ],
)
def test_harmonised_dataset_contract(
    path: Path,
    labels: set[str],
    sources: set[str],
) -> None:
    require_derived_files(
        [
            path,
        ]
    )

    dataframe = read_csv(
        path
    )

    assert_dataset_contract(
        dataframe,
        allowed_labels=labels,
        expected_sources=sources,
    )


def test_three_class_primary_excludes_distracted() -> None:
    require_derived_files(
        [
            EMOSURV_3_PATH,
            SENSEFUZE_3_PATH,
            COMBINED_3_PATH,
        ]
    )

    for path in (
        EMOSURV_3_PATH,
        SENSEFUZE_3_PATH,
        COMBINED_3_PATH,
    ):
        dataframe = read_csv(
            path
        )

        assert (
            "distracted"
            not in set(
                dataframe[
                    "label"
                ]
                .astype(str)
            )
        ), (
            f"Primary three-class dataset unexpectedly "
            f"contains distracted:\n{path}"
        )


def test_combined_three_class_is_exact_source_union() -> None:
    require_derived_files(
        [
            EMOSURV_3_PATH,
            SENSEFUZE_3_PATH,
            COMBINED_3_PATH,
        ]
    )

    emosurv = read_csv(
        EMOSURV_3_PATH
    )

    sensefuze = read_csv(
        SENSEFUZE_3_PATH
    )

    combined = read_csv(
        COMBINED_3_PATH
    )

    emosurv_ids = set(
        emosurv[
            "sample_id"
        ]
        .astype(str)
    )

    sensefuze_ids = set(
        sensefuze[
            "sample_id"
        ]
        .astype(str)
    )

    combined_ids = set(
        combined[
            "sample_id"
        ]
        .astype(str)
    )

    assert not (
        emosurv_ids
        & sensefuze_ids
    ), (
        "Source sample IDs overlap."
    )

    assert combined_ids == (
        emosurv_ids
        | sensefuze_ids
    ), (
        "Combined three-class dataset is not the "
        "exact union of the two source datasets."
    )

    assert len(
        combined
    ) == (
        len(
            emosurv
        )
        + len(
            sensefuze
        )
    )


def test_combined_four_class_is_exact_source_union() -> None:
    require_derived_files(
        [
            EMOSURV_4_PATH,
            SENSEFUZE_4_PATH,
            COMBINED_4_PATH,
        ]
    )

    emosurv = read_csv(
        EMOSURV_4_PATH
    )

    sensefuze = read_csv(
        SENSEFUZE_4_PATH
    )

    combined = read_csv(
        COMBINED_4_PATH
    )

    emosurv_ids = set(
        emosurv[
            "sample_id"
        ]
        .astype(str)
    )

    sensefuze_ids = set(
        sensefuze[
            "sample_id"
        ]
        .astype(str)
    )

    combined_ids = set(
        combined[
            "sample_id"
        ]
        .astype(str)
    )

    assert not (
        emosurv_ids
        & sensefuze_ids
    )

    assert combined_ids == (
        emosurv_ids
        | sensefuze_ids
    )

    assert len(
        combined
    ) == (
        len(
            emosurv
        )
        + len(
            sensefuze
        )
    )


def test_label_provenance_is_preserved() -> None:
    require_derived_files(
        [
            EMOSURV_3_PATH,
            EMOSURV_4_PATH,
            SENSEFUZE_4_PATH,
        ]
    )

    emosurv_three = read_csv(
        EMOSURV_3_PATH
    )

    emosurv_four = read_csv(
        EMOSURV_4_PATH
    )

    sensefuze = read_csv(
        SENSEFUZE_4_PATH
    )

    assert (
        emosurv_three[
            "label_origin"
        ]
        .astype(str)
        .str.contains(
            "emosurv",
            case=False,
        )
        .all()
    )

    assert (
        emosurv_four[
            "label_origin"
        ]
        .astype(str)
        .str.contains(
            "weak",
            case=False,
        )
        .all()
    ), (
        "Four-class EmoSurv rows are no longer clearly "
        "marked as weakly supervised proxy labels."
    )

    assert (
        sensefuze[
            "label_origin"
        ]
        .astype(str)
        .str.contains(
            "sensefuzeai",
            case=False,
        )
        .all()
    )

    assert (
        sensefuze[
            "label"
        ]
        .astype(str)
        .values
        ==
        sensefuze[
            "original_label"
        ]
        .astype(str)
        .values
    ).all(), (
        "SenseFuzeAI collected behavioural labels should "
        "remain their own original labels."
    )


def test_participant_and_session_source_prefixes() -> None:
    require_derived_files(
        [
            EMOSURV_4_PATH,
            SENSEFUZE_4_PATH,
        ]
    )

    emosurv = read_csv(
        EMOSURV_4_PATH
    )

    sensefuze = read_csv(
        SENSEFUZE_4_PATH
    )

    assert (
        emosurv[
            "participant_id"
        ]
        .astype(str)
        .str.startswith(
            "emosurv_"
        )
        .all()
    )

    assert (
        sensefuze[
            "participant_id"
        ]
        .astype(str)
        .str.startswith(
            "sensefuzeai_"
        )
        .all()
    )

    assert (
        emosurv[
            "participant_id"
        ]
        .nunique()
        > 1
    ), (
        "EmoSurv should provide multiple participant IDs."
    )

    assert (
        sensefuze[
            "session_id"
        ]
        .nunique()
        >= sensefuze[
            "participant_id"
        ]
        .nunique()
    )


def test_build_metadata_matches_feature_contract() -> None:
    require_derived_files(
        [
            BUILD_METADATA_PATH,
        ]
    )

    metadata = read_json(
        BUILD_METADATA_PATH
    )

    assert metadata[
        "feature_count"
    ] == 23

    assert metadata[
        "feature_columns"
    ] == EXPECTED_FEATURE_COLUMNS

    assert set(
        metadata[
            "primary_three_class_labels"
        ]
    ) == THREE_CLASS_LABELS

    exploratory_status = str(
        metadata.get(
            "exploratory_four_class_status",
            "",
        )
    ).lower()

    assert (
        "weak"
        in exploratory_status
    ), (
        "Four-class methodological status is not "
        "clearly identified as weak supervision."
    )


# =============================================================================
# SPLIT / LEAKAGE TESTS
# =============================================================================

@pytest.mark.parametrize(
    "results_dir",
    [
        THREE_RESULTS_DIR,
        FOUR_RESULTS_DIR,
    ],
)
def test_frozen_split_manifest_exists_and_has_expected_configuration(
    results_dir: Path,
) -> None:
    manifest_path = (
        results_dir
        / "split_manifest.json"
    )

    require_derived_files(
        [
            manifest_path,
        ]
    )

    manifest = read_json(
        manifest_path
    )

    random_states = recursively_find_scalar(
        manifest,
        "random_state",
    )

    if random_states:
        assert EXPECTED_RANDOM_STATE in {
            int(value)
            for value in random_states
            if isinstance(
                value,
                (
                    int,
                    float,
                    str,
                ),
            )
            and str(value)
            .strip()
            .lstrip("-")
            .isdigit()
        }

    test_sizes = recursively_find_scalar(
        manifest,
        "test_size",
    )

    if test_sizes:
        assert any(
            math.isclose(
                float(value),
                EXPECTED_TEST_SIZE,
                rel_tol=0.0,
                abs_tol=1e-12,
            )
            for value in test_sizes
        )


@pytest.mark.parametrize(
    "results_dir",
    [
        THREE_RESULTS_DIR,
        FOUR_RESULTS_DIR,
    ],
)
def test_emosurv_participant_split_has_no_leakage(
    results_dir: Path,
) -> None:
    manifest_path = (
        results_dir
        / "split_manifest.json"
    )

    require_derived_files(
        [
            manifest_path,
        ]
    )

    manifest = read_json(
        manifest_path
    )

    train_groups = collect_manifest_values(
        manifest,
        required_tokens=(
            "emosurv",
            "train",
        ),
        any_tokens=(
            "participant",
            "group",
        ),
    )

    test_groups = collect_manifest_values(
        manifest,
        required_tokens=(
            "emosurv",
            "test",
        ),
        any_tokens=(
            "participant",
            "group",
        ),
    )

    assert train_groups, (
        "Could not locate EmoSurv training participant/group "
        "identifiers in split_manifest.json."
    )

    assert test_groups, (
        "Could not locate EmoSurv testing participant/group "
        "identifiers in split_manifest.json."
    )

    overlap = (
        train_groups
        & test_groups
    )

    assert not overlap, (
        "EmoSurv participant leakage detected:\n"
        f"{sorted(overlap)[:20]}"
    )


@pytest.mark.parametrize(
    "results_dir",
    [
        THREE_RESULTS_DIR,
        FOUR_RESULTS_DIR,
    ],
)
def test_sensefuzeai_session_split_has_no_leakage(
    results_dir: Path,
) -> None:
    manifest_path = (
        results_dir
        / "split_manifest.json"
    )

    require_derived_files(
        [
            manifest_path,
        ]
    )

    manifest = read_json(
        manifest_path
    )

    train_groups = collect_manifest_values(
        manifest,
        required_tokens=(
            "sensefuze",
            "train",
        ),
        any_tokens=(
            "session",
            "group",
        ),
    )

    test_groups = collect_manifest_values(
        manifest,
        required_tokens=(
            "sensefuze",
            "test",
        ),
        any_tokens=(
            "session",
            "group",
        ),
    )

    assert train_groups, (
        "Could not locate SenseFuzeAI training "
        "session/group identifiers in split manifest."
    )

    assert test_groups, (
        "Could not locate SenseFuzeAI testing "
        "session/group identifiers in split manifest."
    )

    overlap = (
        train_groups
        & test_groups
    )

    assert not overlap, (
        "SenseFuzeAI session leakage detected:\n"
        f"{sorted(overlap)[:20]}"
    )


@pytest.mark.parametrize(
    "results_dir",
    [
        THREE_RESULTS_DIR,
        FOUR_RESULTS_DIR,
    ],
)
def test_baseline_and_augmented_emosurv_use_same_holdout_samples(
    results_dir: Path,
) -> None:
    baseline_path = find_prediction_csv(
        results_dir,
        "A_emosurv_baseline",
    )

    augmented_path = find_prediction_csv(
        results_dir,
        "C_augmented_to_emosurv_test",
    )

    if (
        baseline_path is None
        or augmented_path is None
    ):
        pytest.skip(
            "Prediction CSV files with sample_id were not found "
            "for both A and C experiments."
        )

    baseline = read_csv(
        baseline_path
    )

    augmented = read_csv(
        augmented_path
    )

    baseline_ids = set(
        baseline[
            "sample_id"
        ]
        .astype(str)
    )

    augmented_ids = set(
        augmented[
            "sample_id"
        ]
        .astype(str)
    )

    assert baseline_ids, (
        "Experiment A prediction set is empty."
    )

    assert baseline_ids == augmented_ids, (
        "Experiments A and C were not evaluated on "
        "the exact same held-out EmoSurv samples."
    )


@pytest.mark.parametrize(
    "results_dir",
    [
        THREE_RESULTS_DIR,
        FOUR_RESULTS_DIR,
    ],
)
def test_transfer_and_augmented_sensefuzeai_use_same_holdout_samples(
    results_dir: Path,
) -> None:
    baseline_path = find_prediction_csv(
        results_dir,
        "D_emosurv_to_sensefuzeai",
    )

    augmented_path = find_prediction_csv(
        results_dir,
        "E_augmented_to_sensefuzeai_test",
    )

    if (
        baseline_path is None
        or augmented_path is None
    ):
        pytest.skip(
            "Prediction CSV files with sample_id were not found "
            "for both D and E experiments."
        )

    baseline = read_csv(
        baseline_path
    )

    augmented = read_csv(
        augmented_path
    )

    baseline_ids = set(
        baseline[
            "sample_id"
        ]
        .astype(str)
    )

    augmented_ids = set(
        augmented[
            "sample_id"
        ]
        .astype(str)
    )

    assert baseline_ids, (
        "Experiment D prediction set is empty."
    )

    assert baseline_ids == augmented_ids, (
        "Experiments D and E were not evaluated on "
        "the exact same held-out SenseFuzeAI samples."
    )


# =============================================================================
# EXPERIMENT-OUTPUT TESTS
# =============================================================================

@pytest.mark.parametrize(
    "results_dir",
    [
        THREE_RESULTS_DIR,
        FOUR_RESULTS_DIR,
    ],
)
def test_comparison_summary_contains_all_six_experiments(
    results_dir: Path,
) -> None:
    summary_path = (
        results_dir
        / "comparison_summary.csv"
    )

    require_derived_files(
        [
            summary_path,
        ]
    )

    dataframe = read_csv(
        summary_path
    )

    assert "experiment" in dataframe.columns

    observed = set(
        dataframe[
            "experiment"
        ]
        .astype(str)
    )

    assert observed == EXPECTED_EXPERIMENTS, (
        "Comparison summary does not contain exactly "
        "the canonical A-F experiments.\n"
        f"Missing: {sorted(EXPECTED_EXPERIMENTS - observed)}\n"
        f"Extra:   {sorted(observed - EXPECTED_EXPERIMENTS)}"
    )

    assert len(
        dataframe
    ) == 6

    for metric in (
        "accuracy",
        "balanced_accuracy",
        "macro_f1",
    ):
        assert metric in dataframe.columns

        values = pd.to_numeric(
            dataframe[
                metric
            ],
            errors="coerce",
        )

        assert values.notna().all(), (
            f"{metric} contains missing or invalid values."
        )

        assert (
            (
                values
                >= 0.0
            )
            &
            (
                values
                <= 1.0
            )
        ).all(), (
            f"{metric} must remain within [0, 1]."
        )


def test_consolidated_summary_if_present() -> None:
    consolidated_path = (
        OUTPUT_DIR
        / "results"
        / "all_dataset_comparison_summary.csv"
    )

    if not consolidated_path.exists():
        pytest.skip(
            "Optional consolidated comparison summary "
            "has not been generated."
        )

    dataframe = read_csv(
        consolidated_path
    )

    required_columns = {
        "analysis",
        "experiment",
        "accuracy",
        "balanced_accuracy",
        "macro_f1",
    }

    missing = (
        required_columns
        - set(
            dataframe.columns
        )
    )

    assert not missing

    assert len(
        dataframe
    ) == 12, (
        "Expected 6 primary + 6 exploratory "
        "experiment rows."
    )

    analyses = set(
        dataframe[
            "analysis"
        ]
        .astype(str)
    )

    assert analyses == {
        "three_class_primary",
        "four_class_exploratory",
    }


# =============================================================================
# END
# =============================================================================
