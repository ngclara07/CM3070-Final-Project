"""
inspect_final_experiments_preflight.py

SenseFuzeAI
Final Evaluation Experiments - Read-Only Pre-Flight Inspector

=============================================================================
PURPOSE
=============================================================================

This script inspects the existing SenseFuzeAI repository BEFORE implementing
or running the following final experiments:

    1. Leave-one-modality-out ablation
    2. Inference-time missing-modality robustness
    3. Repeated CPU-based latency benchmarking

The script is intentionally READ-ONLY with respect to the existing project
source code, datasets, and model artifacts.

It attempts to establish, from the ACTUAL repository:

    - the 309-record Chapter-5 evaluation dataset;
    - the target-label column and class distribution;
    - the persisted fusion feature schema;
    - the exact feature count and ordering;
    - explicit feature-group definitions where they exist;
    - the six additional/derived predictors, where identifiable;
    - derived-feature modality dependencies, where explicitly documented;
    - the deployed Random Forest classifier configuration;
    - preprocessing contained in the deployed model pipeline;
    - the existing five-fold stratified CV implementation;
    - the production missing-input / zero-fill implementation;
    - the FastAPI live prediction route;
    - available sample/raw inputs suitable for latency benchmarking;
    - methodological inconsistencies which must be resolved before
      implementing the three new experiments.

IMPORTANT
=============================================================================

This script DOES NOT:

    - train a model;
    - modify a model;
    - modify source code;
    - alter datasets;
    - execute the three final experiments;
    - invent missing feature-group mappings;
    - invent hyperparameters;
    - assume that feature positions correspond to modalities unless supported
      by an existing explicit schema or clearly named feature columns.

It writes only inspection reports under:

    data/processed/final_experiment_preflight/

=============================================================================
EXPECTED REPORT PROTOCOL
=============================================================================

The protocol supplied for the final experiments is:

    records:                 309

    class distribution:
        focused:             77
        distracted:          77
        fatigued:            77
        overloaded:          78

    cross-validation:
        StratifiedKFold
        n_splits=5
        shuffle=True
        random_state=42

    primary metric:
        macro-F1

    complementary metric:
        accuracy

    variability:
        standard deviation across five folds

    execution target:
        CPU only

    report Python version:
        3.11.9

    proposed feature accounting:
        keystroke:           22
        text:                768
        audio:               809
        vision:              768
        derived:             6
        --------------------------------
        total:               2373

The inspector treats 2373 as the PROPOSED experimental specification, but
does not assume that the repository currently implements that exact count.

=============================================================================
RUN
=============================================================================

From the project root:

    python inspect_final_experiments_preflight.py

Optional explicit root:

    python inspect_final_experiments_preflight.py --root .

Optional verbose output:

    python inspect_final_experiments_preflight.py --verbose

=============================================================================
OUTPUT
=============================================================================

JSON:
    data/processed/final_experiment_preflight/preflight_report.json

Markdown:
    data/processed/final_experiment_preflight/preflight_report.md

Exit code:

    0   inspection completed; no blocking methodological inconsistency detected
    1   inspection completed; one or more blocking inconsistencies/unresolved
        requirements remain

=============================================================================
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import platform
import re
import sys
import traceback

from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Optional

import pandas as pd


# =============================================================================
# OPTIONAL DEPENDENCIES
# =============================================================================

try:
    import joblib
except Exception:
    joblib = None


try:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.pipeline import Pipeline
except Exception:
    RandomForestClassifier = None
    Pipeline = None


# =============================================================================
# EXPECTED FINAL-EXPERIMENT PROTOCOL
# =============================================================================

EXPECTED_RECORD_COUNT = 309

EXPECTED_CLASS_COUNTS = {
    "focused": 77,
    "distracted": 77,
    "fatigued": 77,
    "overloaded": 78,
}

EXPECTED_CLASSES = tuple(
    EXPECTED_CLASS_COUNTS.keys()
)

EXPECTED_CV_FOLDS = 5
EXPECTED_CV_SHUFFLE = True
EXPECTED_RANDOM_STATE = 42

EXPECTED_REPORT_PYTHON_VERSION = "3.11.9"

EXPECTED_FEATURE_GROUP_COUNTS = {
    "keystroke": 22,
    "text": 768,
    "audio": 809,
    "vision": 768,
    "derived": 6,
}

EXPECTED_FEATURE_COUNT = sum(
    EXPECTED_FEATURE_GROUP_COUNTS.values()
)

assert EXPECTED_FEATURE_COUNT == 2373


# =============================================================================
# SEARCH CONFIGURATION
# =============================================================================

EXCLUDED_DIRECTORIES = {
    ".git",
    ".venv",
    "venv",
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".idea",
    ".vscode",
    "node_modules",
    "dist",
    "build",
}

LABEL_COLUMN_CANDIDATES = (
    "label",
    "behaviour_label",
    "behavior_label",
    "behaviour_state",
    "behavior_state",
    "state",
    "target",
    "class",
    "y",
)

FEATURE_SCHEMA_KEYS = (
    "feature_columns",
    "features",
    "feature_names",
    "fusion_feature_columns",
    "fusion_features",
)

FEATURE_GROUP_KEY_CANDIDATES = {
    "keystroke": (
        "keystroke_feature_columns",
        "keystroke_features",
        "keystroke",
    ),
    "text": (
        "text_feature_columns",
        "text_features",
        "text",
    ),
    "audio": (
        "audio_feature_columns",
        "audio_features",
        "audio",
    ),
    "vision": (
        "image_feature_columns",
        "vision_feature_columns",
        "visual_feature_columns",
        "image_features",
        "vision_features",
        "visual_features",
        "image",
        "vision",
        "visual",
    ),
    "derived": (
        "derived_feature_columns",
        "derived_features",
        "additional_feature_columns",
        "additional_features",
        "fusion_derived_features",
    ),
}

DERIVED_DEPENDENCY_KEYS = (
    "derived_feature_dependencies",
    "feature_dependencies",
    "derived_dependencies",
    "fusion_feature_dependencies",
)

LIKELY_DATASET_NAME_TERMS = (
    "multimodal",
    "fusion",
    "feature",
)

LIKELY_SCHEMA_NAME_TERMS = (
    "feature",
    "schema",
    "column",
    "metadata",
)

LIKELY_FUSION_MODEL_TERMS = (
    "fusion",
    "multimodal",
)

SOURCE_SEARCH_TERMS = {
    "cv": (
        "StratifiedKFold",
        "n_splits",
        "shuffle",
        "random_state",
        "macro_f1",
        "f1_score",
        "accuracy_score",
    ),
    "missing_modality": (
        "fillna",
        "fill_value",
        "np.zeros",
        "numpy.zeros",
        "zero",
        "missing",
        "reindex",
    ),
    "fastapi": (
        "/predict_live",
        "predict_live",
        "FinalMultimodalInference",
        "predictor.predict",
    ),
}


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class DatasetCandidate:
    path: str
    rows: int
    columns: int
    label_column: Optional[str]
    class_counts: dict[str, int]
    exact_protocol_match: bool
    error: Optional[str] = None


@dataclass
class SchemaCandidate:
    path: str
    source: str
    key: Optional[str]
    feature_count: int
    exact_2373_match: bool
    first_features: list[str]
    last_features: list[str]


@dataclass
class ModelArtifactInspection:
    path: str
    loaded: bool
    artifact_type: Optional[str]
    classifier_type: Optional[str]
    classifier_parameters: dict[str, Any]
    preprocessing_steps: list[dict[str, Any]]
    error: Optional[str]


@dataclass
class CVEvidence:
    path: str
    line_number: int
    line: str


@dataclass
class RouteEvidence:
    path: str
    method: str
    route: str
    function_name: str
    line_number: int


@dataclass
class SourceEvidence:
    path: str
    line_number: int
    line: str


@dataclass
class Issue:
    severity: str
    code: str
    message: str


# =============================================================================
# GENERAL HELPERS
# =============================================================================

def normalise_label(
    value: Any,
) -> str:
    return str(
        value
    ).strip().lower()


def safe_json_value(
    value: Any,
) -> Any:
    """
    Convert arbitrary model parameters into JSON-safe values.
    """

    if value is None:
        return None

    if isinstance(
        value,
        (
            str,
            int,
            float,
            bool,
        ),
    ):
        return value

    if isinstance(
        value,
        Path,
    ):
        return str(
            value
        )

    if isinstance(
        value,
        dict,
    ):
        return {
            str(key):
                safe_json_value(
                    child
                )
            for key, child
            in value.items()
        }

    if isinstance(
        value,
        (
            list,
            tuple,
            set,
        ),
    ):
        return [
            safe_json_value(
                child
            )
            for child
            in value
        ]

    return repr(
        value
    )


def relative_path(
    path: Path,
    root: Path,
) -> str:
    try:
        return str(
            path.resolve()
            .relative_to(
                root.resolve()
            )
        )
    except Exception:
        return str(
            path
        )


def should_skip_path(
    path: Path,
) -> bool:
    return any(
        part in EXCLUDED_DIRECTORIES
        for part
        in path.parts
    )


def iter_files(
    root: Path,
    suffixes: Iterable[str],
) -> Iterable[Path]:
    wanted = {
        suffix.lower()
        for suffix
        in suffixes
    }

    for path in root.rglob(
        "*"
    ):
        if should_skip_path(
            path
        ):
            continue

        if not path.is_file():
            continue

        if path.suffix.lower() in wanted:
            yield path


def read_text_safe(
    path: Path,
) -> Optional[str]:
    try:
        return path.read_text(
            encoding="utf-8"
        )
    except UnicodeDecodeError:
        try:
            return path.read_text(
                encoding="utf-8",
                errors="replace",
            )
        except Exception:
            return None
    except Exception:
        return None


def python_version_short() -> str:
    return (
        f"{sys.version_info.major}."
        f"{sys.version_info.minor}."
        f"{sys.version_info.micro}"
    )


# =============================================================================
# DATASET INSPECTION
# =============================================================================

def detect_label_column(
    dataframe: pd.DataFrame,
) -> Optional[str]:

    direct_lookup = {
        str(column).lower():
            str(column)
        for column
        in dataframe.columns
    }

    for candidate in LABEL_COLUMN_CANDIDATES:
        if candidate in direct_lookup:
            return direct_lookup[
                candidate
            ]

    expected = set(
        EXPECTED_CLASSES
    )

    possible: list[
        str
    ] = []

    for column in dataframe.columns:
        series = dataframe[
            column
        ]

        if (
            series.dtype
            == object
            or
            str(
                series.dtype
            ).startswith(
                "string"
            )
        ):
            unique = {
                normalise_label(
                    value
                )
                for value
                in series.dropna().unique()
            }

            if unique and unique.issubset(
                expected
            ):
                possible.append(
                    str(
                        column
                    )
                )

    if len(
        possible
    ) == 1:
        return possible[
            0
        ]

    return None


def inspect_csv_dataset(
    path: Path,
    root: Path,
) -> DatasetCandidate:

    try:
        dataframe = pd.read_csv(
            path
        )

        label_column = detect_label_column(
            dataframe
        )

        counts: dict[
            str,
            int
        ] = {}

        if label_column is not None:
            normalised = (
                dataframe[
                    label_column
                ]
                .map(
                    normalise_label
                )
            )

            counts = {
                str(key):
                    int(value)
                for key, value
                in normalised.value_counts()
                .to_dict()
                .items()
            }

        exact = (
            len(
                dataframe
            )
            == EXPECTED_RECORD_COUNT
            and
            counts
            == EXPECTED_CLASS_COUNTS
        )

        return DatasetCandidate(
            path=relative_path(
                path,
                root,
            ),
            rows=int(
                len(
                    dataframe
                )
            ),
            columns=int(
                len(
                    dataframe.columns
                )
            ),
            label_column=label_column,
            class_counts=counts,
            exact_protocol_match=exact,
        )

    except Exception as exc:
        return DatasetCandidate(
            path=relative_path(
                path,
                root,
            ),
            rows=-1,
            columns=-1,
            label_column=None,
            class_counts={},
            exact_protocol_match=False,
            error=(
                f"{type(exc).__name__}: "
                f"{exc}"
            ),
        )


def find_dataset_candidates(
    root: Path,
    *,
    verbose: bool,
) -> list[
    DatasetCandidate
]:

    candidates: list[
        DatasetCandidate
    ] = []

    csv_paths = list(
        iter_files(
            root,
            {
                ".csv",
            },
        )
    )

    # Inspect likely files first.
    csv_paths.sort(
        key=lambda path: (
            0
            if any(
                term in path.name.lower()
                for term
                in LIKELY_DATASET_NAME_TERMS
            )
            else 1,
            len(
                path.parts
            ),
            str(
                path
            ),
        )
    )

    for index, path in enumerate(
        csv_paths,
        start=1,
    ):
        if verbose:
            print(
                "[dataset] "
                f"{index}/{len(csv_paths)} "
                f"{relative_path(path, root)}"
            )

        # Avoid accidentally loading huge raw external datasets.
        try:
            size_mb = (
                path.stat().st_size
                / (
                    1024
                    * 1024
                )
            )
        except Exception:
            size_mb = 0.0

        if size_mb > 250.0:
            continue

        candidate = inspect_csv_dataset(
            path,
            root,
        )

        # Keep exact matches and plausible 309-row datasets.
        if (
            candidate.exact_protocol_match
            or
            candidate.rows
            == EXPECTED_RECORD_COUNT
        ):
            candidates.append(
                candidate
            )

    candidates.sort(
        key=lambda item: (
            not item.exact_protocol_match,
            item.path,
        )
    )

    return candidates


# =============================================================================
# FEATURE-SCHEMA INSPECTION
# =============================================================================

def extract_string_lists_from_json(
    value: Any,
    *,
    prefix: str = "",
) -> list[
    tuple[
        str,
        list[str],
    ]
]:

    output: list[
        tuple[
            str,
            list[str],
        ]
    ] = []

    if isinstance(
        value,
        dict,
    ):
        for key, child in value.items():
            next_prefix = (
                f"{prefix}.{key}"
                if prefix
                else str(
                    key
                )
            )

            output.extend(
                extract_string_lists_from_json(
                    child,
                    prefix=next_prefix,
                )
            )

    elif isinstance(
        value,
        list,
    ):
        if (
            value
            and
            all(
                isinstance(
                    item,
                    str,
                )
                for item
                in value
            )
        ):
            output.append(
                (
                    prefix,
                    list(
                        value
                    ),
                )
            )

    return output


def inspect_schema_json(
    path: Path,
    root: Path,
) -> list[
    SchemaCandidate
]:

    try:
        payload = json.loads(
            path.read_text(
                encoding="utf-8"
            )
        )
    except Exception:
        return []

    string_lists = (
        extract_string_lists_from_json(
            payload
        )
    )

    output: list[
        SchemaCandidate
    ] = []

    for key, features in string_lists:
        lowered = key.lower()

        likely_feature_list = (
            len(
                features
            )
            >= 20
            and
            (
                any(
                    term in lowered
                    for term
                    in FEATURE_SCHEMA_KEYS
                )
                or
                "feature"
                in lowered
                or
                len(
                    features
                )
                in {
                    2367,
                    2373,
                }
            )
        )

        if not likely_feature_list:
            continue

        output.append(
            SchemaCandidate(
                path=relative_path(
                    path,
                    root,
                ),
                source="json",
                key=key,
                feature_count=len(
                    features
                ),
                exact_2373_match=(
                    len(
                        features
                    )
                    == EXPECTED_FEATURE_COUNT
                ),
                first_features=features[
                    :10
                ],
                last_features=features[
                    -10:
                ],
            )
        )

    return output


def literal_string_list(
    node: ast.AST,
) -> Optional[
    list[str]
]:
    try:
        value = ast.literal_eval(
            node
        )
    except Exception:
        return None

    if not isinstance(
        value,
        (
            list,
            tuple,
        ),
    ):
        return None

    if not all(
        isinstance(
            item,
            str,
        )
        for item
        in value
    ):
        return None

    return list(
        value
    )


def extract_ast_string_lists(
    path: Path,
    root: Path,
) -> list[
    SchemaCandidate
]:

    source = read_text_safe(
        path
    )

    if source is None:
        return []

    try:
        tree = ast.parse(
            source,
            filename=str(
                path
            ),
        )
    except Exception:
        return []

    output: list[
        SchemaCandidate
    ] = []

    for node in tree.body:
        variable_name: Optional[
            str
        ] = None

        value_node: Optional[
            ast.AST
        ] = None

        if isinstance(
            node,
            ast.Assign,
        ):
            if len(
                node.targets
            ) != 1:
                continue

            target = node.targets[
                0
            ]

            if isinstance(
                target,
                ast.Name,
            ):
                variable_name = (
                    target.id
                )

            value_node = (
                node.value
            )

        elif isinstance(
            node,
            ast.AnnAssign,
        ):
            if isinstance(
                node.target,
                ast.Name,
            ):
                variable_name = (
                    node.target.id
                )

            value_node = (
                node.value
            )

        if (
            variable_name is None
            or
            value_node is None
        ):
            continue

        features = literal_string_list(
            value_node
        )

        if features is None:
            continue

        lowered = variable_name.lower()

        if (
            len(
                features
            )
            < 20
        ):
            continue

        if (
            "feature"
            not in lowered
            and
            len(
                features
            )
            not in {
                2367,
                2373,
            }
        ):
            continue

        output.append(
            SchemaCandidate(
                path=relative_path(
                    path,
                    root,
                ),
                source="python_literal",
                key=variable_name,
                feature_count=len(
                    features
                ),
                exact_2373_match=(
                    len(
                        features
                    )
                    == EXPECTED_FEATURE_COUNT
                ),
                first_features=features[
                    :10
                ],
                last_features=features[
                    -10:
                ],
            )
        )

    return output


def find_schema_candidates(
    root: Path,
) -> list[
    SchemaCandidate
]:

    output: list[
        SchemaCandidate
    ] = []

    for path in iter_files(
        root,
        {
            ".json",
        },
    ):
        lowered_name = (
            path.name.lower()
        )

        if not any(
            term in lowered_name
            for term
            in LIKELY_SCHEMA_NAME_TERMS
        ):
            continue

        output.extend(
            inspect_schema_json(
                path,
                root,
            )
        )

    for path in iter_files(
        root,
        {
            ".py",
        },
    ):
        output.extend(
            extract_ast_string_lists(
                path,
                root,
            )
        )

    # Deduplicate.
    unique: dict[
        tuple[
            str,
            str,
            Optional[str],
            int,
        ],
        SchemaCandidate,
    ] = {}

    for candidate in output:
        key = (
            candidate.path,
            candidate.source,
            candidate.key,
            candidate.feature_count,
        )

        unique[
            key
        ] = candidate

    result = list(
        unique.values()
    )

    result.sort(
        key=lambda item: (
            not item.exact_2373_match,
            0
            if item.feature_count
            in {
                2367,
                2373,
            }
            else 1,
            item.path,
            str(
                item.key
            ),
        )
    )

    return result


def load_schema_feature_list(
    root: Path,
    candidate: SchemaCandidate,
) -> Optional[
    list[str]
]:

    path = (
        root
        / candidate.path
    )

    if candidate.source == "json":
        try:
            payload = json.loads(
                path.read_text(
                    encoding="utf-8"
                )
            )
        except Exception:
            return None

        current: Any = payload

        if candidate.key:
            for part in candidate.key.split(
                "."
            ):
                if not isinstance(
                    current,
                    dict,
                ):
                    return None

                if part not in current:
                    return None

                current = current[
                    part
                ]

        if (
            isinstance(
                current,
                list,
            )
            and
            all(
                isinstance(
                    item,
                    str,
                )
                for item
                in current
            )
        ):
            return list(
                current
            )

        return None

    if candidate.source == "python_literal":
        source = read_text_safe(
            path
        )

        if source is None:
            return None

        try:
            tree = ast.parse(
                source
            )
        except Exception:
            return None

        for node in tree.body:
            variable_name: Optional[
                str
            ] = None

            value_node: Optional[
                ast.AST
            ] = None

            if isinstance(
                node,
                ast.Assign,
            ):
                if (
                    len(
                        node.targets
                    )
                    == 1
                    and
                    isinstance(
                        node.targets[
                            0
                        ],
                        ast.Name,
                    )
                ):
                    variable_name = (
                        node.targets[
                            0
                        ].id
                    )

                value_node = (
                    node.value
                )

            elif isinstance(
                node,
                ast.AnnAssign,
            ):
                if isinstance(
                    node.target,
                    ast.Name,
                ):
                    variable_name = (
                        node.target.id
                    )

                value_node = (
                    node.value
                )

            if (
                variable_name
                == candidate.key
                and
                value_node
                is not None
            ):
                return literal_string_list(
                    value_node
                )

    return None


# =============================================================================
# EXPLICIT FEATURE-GROUP INSPECTION
# =============================================================================

def find_feature_groups_in_json(
    root: Path,
) -> list[
    dict[str, Any]
]:

    findings: list[
        dict[str, Any]
    ] = []

    for path in iter_files(
        root,
        {
            ".json",
        },
    ):
        try:
            payload = json.loads(
                path.read_text(
                    encoding="utf-8"
                )
            )
        except Exception:
            continue

        if not isinstance(
            payload,
            dict,
        ):
            continue

        for group, key_candidates in (
            FEATURE_GROUP_KEY_CANDIDATES.items()
        ):
            for key in key_candidates:
                value = payload.get(
                    key
                )

                if (
                    isinstance(
                        value,
                        list,
                    )
                    and
                    all(
                        isinstance(
                            item,
                            str,
                        )
                        for item
                        in value
                    )
                ):
                    findings.append(
                        {
                            "path":
                                relative_path(
                                    path,
                                    root,
                                ),
                            "group":
                                group,
                            "key":
                                key,
                            "feature_count":
                                len(
                                    value
                                ),
                            "features":
                                value,
                            "evidence_type":
                                "json_explicit",
                        }
                    )

    return findings


def find_feature_groups_in_python(
    root: Path,
) -> list[
    dict[str, Any]
]:

    findings: list[
        dict[str, Any]
    ] = []

    expected_variable_names: dict[
        str,
        str,
    ] = {}

    for group, candidates in (
        FEATURE_GROUP_KEY_CANDIDATES.items()
    ):
        for candidate in candidates:
            expected_variable_names[
                candidate.upper()
            ] = group

    for path in iter_files(
        root,
        {
            ".py",
        },
    ):
        source = read_text_safe(
            path
        )

        if source is None:
            continue

        try:
            tree = ast.parse(
                source,
                filename=str(
                    path
                ),
            )
        except Exception:
            continue

        for node in tree.body:
            variable_name: Optional[
                str
            ] = None

            value_node: Optional[
                ast.AST
            ] = None

            if isinstance(
                node,
                ast.Assign,
            ):
                if (
                    len(
                        node.targets
                    )
                    == 1
                    and
                    isinstance(
                        node.targets[
                            0
                        ],
                        ast.Name,
                    )
                ):
                    variable_name = (
                        node.targets[
                            0
                        ].id
                    )

                value_node = (
                    node.value
                )

            elif isinstance(
                node,
                ast.AnnAssign,
            ):
                if isinstance(
                    node.target,
                    ast.Name,
                ):
                    variable_name = (
                        node.target.id
                    )

                value_node = (
                    node.value
                )

            if (
                variable_name is None
                or
                value_node is None
            ):
                continue

            uppercase = (
                variable_name.upper()
            )

            group = (
                expected_variable_names.get(
                    uppercase
                )
            )

            if group is None:
                # Additional conservative matching.
                if (
                    "FEATURE"
                    in uppercase
                    and
                    "KEYSTROKE"
                    in uppercase
                ):
                    group = "keystroke"

                elif (
                    "FEATURE"
                    in uppercase
                    and
                    "TEXT"
                    in uppercase
                ):
                    group = "text"

                elif (
                    "FEATURE"
                    in uppercase
                    and
                    "AUDIO"
                    in uppercase
                ):
                    group = "audio"

                elif (
                    "FEATURE"
                    in uppercase
                    and
                    (
                        "IMAGE"
                        in uppercase
                        or
                        "VISION"
                        in uppercase
                        or
                        "VISUAL"
                        in uppercase
                    )
                ):
                    group = "vision"

                elif (
                    "FEATURE"
                    in uppercase
                    and
                    (
                        "DERIVED"
                        in uppercase
                        or
                        "ADDITIONAL"
                        in uppercase
                    )
                ):
                    group = "derived"

            if group is None:
                continue

            values = literal_string_list(
                value_node
            )

            if values is None:
                continue

            findings.append(
                {
                    "path":
                        relative_path(
                            path,
                            root,
                        ),
                    "group":
                        group,
                    "key":
                        variable_name,
                    "feature_count":
                        len(
                            values
                        ),
                    "features":
                        values,
                    "evidence_type":
                        "python_literal_explicit",
                }
            )

    return findings


def consolidate_feature_groups(
    root: Path,
) -> dict[
    str,
    Any
]:

    findings = (
        find_feature_groups_in_json(
            root
        )
        +
        find_feature_groups_in_python(
            root
        )
    )

    by_group: dict[
        str,
        list[dict[str, Any]],
    ] = {
        group: []
        for group
        in EXPECTED_FEATURE_GROUP_COUNTS
    }

    for finding in findings:
        by_group[
            finding[
                "group"
            ]
        ].append(
            finding
        )

    summary: dict[
        str,
        Any
    ] = {}

    for group, expected_count in (
        EXPECTED_FEATURE_GROUP_COUNTS.items()
    ):
        candidates = by_group[
            group
        ]

        exact = [
            candidate
            for candidate
            in candidates
            if candidate[
                "feature_count"
            ]
            == expected_count
        ]

        chosen = (
            exact[
                0
            ]
            if exact
            else
            (
                candidates[
                    0
                ]
                if candidates
                else None
            )
        )

        summary[
            group
        ] = {
            "expected_count":
                expected_count,
            "resolved":
                chosen is not None,
            "exact_expected_count":
                (
                    chosen is not None
                    and
                    chosen[
                        "feature_count"
                    ]
                    == expected_count
                ),
            "chosen":
                chosen,
            "all_candidates":
                candidates,
        }

    return summary


# =============================================================================
# DERIVED-FEATURE DEPENDENCIES
# =============================================================================

def find_dependency_dicts(
    root: Path,
) -> list[
    dict[str, Any]
]:

    findings: list[
        dict[str, Any]
    ] = []

    for path in iter_files(
        root,
        {
            ".json",
        },
    ):
        try:
            payload = json.loads(
                path.read_text(
                    encoding="utf-8"
                )
            )
        except Exception:
            continue

        if not isinstance(
            payload,
            dict,
        ):
            continue

        for key in DERIVED_DEPENDENCY_KEYS:
            value = payload.get(
                key
            )

            if isinstance(
                value,
                dict,
            ):
                findings.append(
                    {
                        "path":
                            relative_path(
                                path,
                                root,
                            ),
                        "key":
                            key,
                        "mapping":
                            value,
                        "evidence_type":
                            "json_explicit",
                    }
                )

    for path in iter_files(
        root,
        {
            ".py",
        },
    ):
        source = read_text_safe(
            path
        )

        if source is None:
            continue

        try:
            tree = ast.parse(
                source,
                filename=str(
                    path
                ),
            )
        except Exception:
            continue

        for node in tree.body:
            variable_name: Optional[
                str
            ] = None

            value_node: Optional[
                ast.AST
            ] = None

            if isinstance(
                node,
                ast.Assign,
            ):
                if (
                    len(
                        node.targets
                    )
                    == 1
                    and
                    isinstance(
                        node.targets[
                            0
                        ],
                        ast.Name,
                    )
                ):
                    variable_name = (
                        node.targets[
                            0
                        ].id
                    )

                value_node = (
                    node.value
                )

            elif isinstance(
                node,
                ast.AnnAssign,
            ):
                if isinstance(
                    node.target,
                    ast.Name,
                ):
                    variable_name = (
                        node.target.id
                    )

                value_node = (
                    node.value
                )

            if (
                variable_name is None
                or
                value_node is None
            ):
                continue

            if variable_name.lower() not in (
                DERIVED_DEPENDENCY_KEYS
            ):
                continue

            try:
                value = ast.literal_eval(
                    value_node
                )
            except Exception:
                continue

            if isinstance(
                value,
                dict,
            ):
                findings.append(
                    {
                        "path":
                            relative_path(
                                path,
                                root,
                            ),
                        "key":
                            variable_name,
                        "mapping":
                            value,
                        "evidence_type":
                            "python_literal_explicit",
                    }
                )

    return findings


def locate_feature_occurrences(
    root: Path,
    feature_names: list[str],
) -> dict[
    str,
    list[dict[str, Any]],
]:

    output: dict[
        str,
        list[dict[str, Any]],
    ] = {
        feature: []
        for feature
        in feature_names
    }

    if not feature_names:
        return output

    for path in iter_files(
        root,
        {
            ".py",
            ".json",
            ".md",
        },
    ):
        text = read_text_safe(
            path
        )

        if text is None:
            continue

        lines = text.splitlines()

        for feature in feature_names:
            for line_number, line in enumerate(
                lines,
                start=1,
            ):
                if feature in line:
                    output[
                        feature
                    ].append(
                        {
                            "path":
                                relative_path(
                                    path,
                                    root,
                                ),
                            "line_number":
                                line_number,
                            "line":
                                line.strip()[
                                    :400
                                ],
                        }
                    )

    return output


# =============================================================================
# MODEL ARTIFACT INSPECTION
# =============================================================================

def find_random_forest_in_object(
    artifact: Any,
) -> tuple[
    Optional[Any],
    list[dict[str, Any]],
]:

    preprocessing: list[
        dict[str, Any]
    ] = []

    if (
        RandomForestClassifier is not None
        and
        isinstance(
            artifact,
            RandomForestClassifier,
        )
    ):
        return (
            artifact,
            preprocessing,
        )

    if (
        Pipeline is not None
        and
        isinstance(
            artifact,
            Pipeline,
        )
    ):
        classifier: Optional[
            Any
        ] = None

        for name, step in artifact.steps:
            step_info = {
                "name":
                    name,
                "type":
                    (
                        f"{type(step).__module__}."
                        f"{type(step).__name__}"
                    ),
            }

            if hasattr(
                step,
                "get_params",
            ):
                try:
                    step_info[
                        "parameters"
                    ] = safe_json_value(
                        step.get_params(
                            deep=False
                        )
                    )
                except Exception:
                    pass

            if (
                RandomForestClassifier is not None
                and
                isinstance(
                    step,
                    RandomForestClassifier,
                )
            ):
                classifier = (
                    step
                )
            else:
                preprocessing.append(
                    step_info
                )

        return (
            classifier,
            preprocessing,
        )

    # Generic sklearn-compatible containers.
    if hasattr(
        artifact,
        "named_steps",
    ):
        try:
            named_steps = dict(
                artifact.named_steps
            )
        except Exception:
            named_steps = {}

        classifier = None

        for name, step in named_steps.items():
            if (
                RandomForestClassifier is not None
                and
                isinstance(
                    step,
                    RandomForestClassifier,
                )
            ):
                classifier = (
                    step
                )
            else:
                preprocessing.append(
                    {
                        "name":
                            str(
                                name
                            ),
                        "type":
                            (
                                f"{type(step).__module__}."
                                f"{type(step).__name__}"
                            ),
                        "parameters":
                            safe_json_value(
                                step.get_params(
                                    deep=False
                                )
                            )
                            if hasattr(
                                step,
                                "get_params",
                            )
                            else {},
                    }
                )

        return (
            classifier,
            preprocessing,
        )

    return (
        None,
        preprocessing,
    )


def inspect_model_artifact(
    path: Path,
    root: Path,
) -> ModelArtifactInspection:

    if joblib is None:
        return ModelArtifactInspection(
            path=relative_path(
                path,
                root,
            ),
            loaded=False,
            artifact_type=None,
            classifier_type=None,
            classifier_parameters={},
            preprocessing_steps=[],
            error=(
                "joblib is unavailable in the current "
                "Python environment."
            ),
        )

    try:
        artifact = joblib.load(
            path
        )

        classifier, preprocessing = (
            find_random_forest_in_object(
                artifact
            )
        )

        classifier_type: Optional[
            str
        ] = None

        classifier_parameters: dict[
            str,
            Any
        ] = {}

        if classifier is not None:
            classifier_type = (
                f"{type(classifier).__module__}."
                f"{type(classifier).__name__}"
            )

            if hasattr(
                classifier,
                "get_params",
            ):
                classifier_parameters = (
                    safe_json_value(
                        classifier.get_params(
                            deep=False
                        )
                    )
                )

        return ModelArtifactInspection(
            path=relative_path(
                path,
                root,
            ),
            loaded=True,
            artifact_type=(
                f"{type(artifact).__module__}."
                f"{type(artifact).__name__}"
            ),
            classifier_type=classifier_type,
            classifier_parameters=(
                classifier_parameters
            ),
            preprocessing_steps=(
                preprocessing
            ),
            error=None,
        )

    except Exception as exc:
        return ModelArtifactInspection(
            path=relative_path(
                path,
                root,
            ),
            loaded=False,
            artifact_type=None,
            classifier_type=None,
            classifier_parameters={},
            preprocessing_steps=[],
            error=(
                f"{type(exc).__name__}: "
                f"{exc}"
            ),
        )


def find_fusion_model_artifacts(
    root: Path,
) -> list[
    ModelArtifactInspection
]:

    paths = list(
        iter_files(
            root,
            {
                ".joblib",
                ".pkl",
                ".pickle",
            },
        )
    )

    paths.sort(
        key=lambda path: (
            0
            if any(
                term in str(
                    path
                ).lower()
                for term
                in LIKELY_FUSION_MODEL_TERMS
            )
            else 1,
            str(
                path
            ),
        )
    )

    results: list[
        ModelArtifactInspection
    ] = []

    # Load only likely fusion/multimodal artifacts.
    for path in paths:
        lowered = str(
            path
        ).lower()

        if not any(
            term in lowered
            for term
            in LIKELY_FUSION_MODEL_TERMS
        ):
            continue

        results.append(
            inspect_model_artifact(
                path,
                root,
            )
        )

    return results


# =============================================================================
# SOURCE INSPECTION
# =============================================================================

def source_evidence(
    root: Path,
    terms: Iterable[str],
) -> list[
    SourceEvidence
]:

    lowered_terms = [
        term.lower()
        for term
        in terms
    ]

    output: list[
        SourceEvidence
    ] = []

    for path in iter_files(
        root,
        {
            ".py",
        },
    ):
        text = read_text_safe(
            path
        )

        if text is None:
            continue

        for line_number, line in enumerate(
            text.splitlines(),
            start=1,
        ):
            lowered_line = (
                line.lower()
            )

            if any(
                term
                in lowered_line
                for term
                in lowered_terms
            ):
                output.append(
                    SourceEvidence(
                        path=relative_path(
                            path,
                            root,
                        ),
                        line_number=(
                            line_number
                        ),
                        line=(
                            line.strip()[
                                :500
                            ]
                        ),
                    )
                )

    return output


def inspect_stratified_kfold_calls(
    root: Path,
) -> list[
    dict[str, Any]
]:

    findings: list[
        dict[str, Any]
    ] = []

    for path in iter_files(
        root,
        {
            ".py",
        },
    ):
        source = read_text_safe(
            path
        )

        if source is None:
            continue

        try:
            tree = ast.parse(
                source,
                filename=str(
                    path
                ),
            )
        except Exception:
            continue

        for node in ast.walk(
            tree
        ):
            if not isinstance(
                node,
                ast.Call,
            ):
                continue

            function_name: Optional[
                str
            ] = None

            if isinstance(
                node.func,
                ast.Name,
            ):
                function_name = (
                    node.func.id
                )

            elif isinstance(
                node.func,
                ast.Attribute,
            ):
                function_name = (
                    node.func.attr
                )

            if function_name != "StratifiedKFold":
                continue

            kwargs: dict[
                str,
                Any
            ] = {}

            for keyword in node.keywords:
                if keyword.arg is None:
                    continue

                try:
                    kwargs[
                        keyword.arg
                    ] = ast.literal_eval(
                        keyword.value
                    )
                except Exception:
                    kwargs[
                        keyword.arg
                    ] = (
                        ast.unparse(
                            keyword.value
                        )
                        if hasattr(
                            ast,
                            "unparse",
                        )
                        else "<expression>"
                    )

            findings.append(
                {
                    "path":
                        relative_path(
                            path,
                            root,
                        ),
                    "line_number":
                        getattr(
                            node,
                            "lineno",
                            -1,
                        ),
                    "kwargs":
                        kwargs,
                    "protocol_match":
                        (
                            kwargs.get(
                                "n_splits"
                            )
                            == EXPECTED_CV_FOLDS
                            and
                            kwargs.get(
                                "shuffle"
                            )
                            is EXPECTED_CV_SHUFFLE
                            and
                            kwargs.get(
                                "random_state"
                            )
                            == EXPECTED_RANDOM_STATE
                        ),
                }
            )

    return findings


def inspect_fastapi_routes(
    root: Path,
) -> list[
    RouteEvidence
]:

    output: list[
        RouteEvidence
    ] = []

    for path in iter_files(
        root,
        {
            ".py",
        },
    ):
        source = read_text_safe(
            path
        )

        if source is None:
            continue

        if (
            "FastAPI"
            not in source
            and
            "@app."
            not in source
        ):
            continue

        try:
            tree = ast.parse(
                source,
                filename=str(
                    path
                ),
            )
        except Exception:
            continue

        for node in tree.body:
            if not isinstance(
                node,
                (
                    ast.FunctionDef,
                    ast.AsyncFunctionDef,
                ),
            ):
                continue

            for decorator in node.decorator_list:
                if not isinstance(
                    decorator,
                    ast.Call,
                ):
                    continue

                if not isinstance(
                    decorator.func,
                    ast.Attribute,
                ):
                    continue

                method = decorator.func.attr.upper()

                if method not in {
                    "GET",
                    "POST",
                    "PUT",
                    "PATCH",
                    "DELETE",
                }:
                    continue

                if not decorator.args:
                    continue

                first = decorator.args[
                    0
                ]

                if not (
                    isinstance(
                        first,
                        ast.Constant,
                    )
                    and
                    isinstance(
                        first.value,
                        str,
                    )
                ):
                    continue

                route = first.value

                if (
                    "predict"
                    not in route.lower()
                    and
                    "model"
                    not in route.lower()
                    and
                    "audio"
                    not in route.lower()
                    and
                    "visual"
                    not in route.lower()
                ):
                    continue

                output.append(
                    RouteEvidence(
                        path=relative_path(
                            path,
                            root,
                        ),
                        method=method,
                        route=route,
                        function_name=(
                            node.name
                        ),
                        line_number=(
                            getattr(
                                node,
                                "lineno",
                                -1,
                            )
                        ),
                    )
                )

    return output


# =============================================================================
# LATENCY INPUT INSPECTION
# =============================================================================

def count_files_with_suffixes(
    directory: Path,
    suffixes: set[str],
) -> int:

    if not directory.exists():
        return 0

    count = 0

    for path in directory.rglob(
        "*"
    ):
        if (
            path.is_file()
            and
            path.suffix.lower()
            in suffixes
        ):
            count += 1

    return count


def inspect_latency_inputs(
    root: Path,
) -> dict[
    str,
    Any
]:

    candidate_roots = [
        root
        / "sample_data",

        root
        / "data"
        / "session_aligned",

        root
        / "web_app"
        / "output",
    ]

    results: list[
        dict[str, Any]
    ] = []

    for directory in candidate_roots:
        if not directory.exists():
            continue

        record = {
            "path":
                relative_path(
                    directory,
                    root,
                ),

            "audio_files":
                count_files_with_suffixes(
                    directory,
                    {
                        ".wav",
                        ".mp3",
                        ".flac",
                        ".m4a",
                        ".ogg",
                    },
                ),

            "image_files":
                count_files_with_suffixes(
                    directory,
                    {
                        ".jpg",
                        ".jpeg",
                        ".png",
                        ".bmp",
                        ".webp",
                    },
                ),

            "video_files":
                count_files_with_suffixes(
                    directory,
                    {
                        ".mp4",
                        ".avi",
                        ".mov",
                        ".mkv",
                        ".webm",
                    },
                ),

            "json_files":
                count_files_with_suffixes(
                    directory,
                    {
                        ".json",
                    },
                ),

            "text_files":
                count_files_with_suffixes(
                    directory,
                    {
                        ".txt",
                    },
                ),
        }

        results.append(
            record
        )

    return {
        "candidate_input_directories":
            results,
        "suitable_input_directory_found":
            any(
                (
                    item[
                        "audio_files"
                    ]
                    > 0
                    and
                    item[
                        "image_files"
                    ]
                    > 0
                )
                for item
                in results
            ),
    }


# =============================================================================
# README / FEATURE-COUNT CONSISTENCY
# =============================================================================

def find_feature_count_mentions(
    root: Path,
) -> list[
    dict[str, Any]
]:

    patterns = [
        re.compile(
            r"\b2367\b"
        ),
        re.compile(
            r"\b2,367\b"
        ),
        re.compile(
            r"\b2373\b"
        ),
        re.compile(
            r"\b2,373\b"
        ),
    ]

    results: list[
        dict[str, Any]
    ] = []

    for path in iter_files(
        root,
        {
            ".md",
            ".py",
            ".json",
            ".txt",
        },
    ):
        text = read_text_safe(
            path
        )

        if text is None:
            continue

        for line_number, line in enumerate(
            text.splitlines(),
            start=1,
        ):
            if any(
                pattern.search(
                    line
                )
                for pattern
                in patterns
            ):
                results.append(
                    {
                        "path":
                            relative_path(
                                path,
                                root,
                            ),
                        "line_number":
                            line_number,
                        "line":
                            line.strip()[
                                :500
                            ],
                    }
                )

    return results


# =============================================================================
# ISSUE ANALYSIS
# =============================================================================

def build_issues(
    *,
    datasets: list[DatasetCandidate],
    schemas: list[SchemaCandidate],
    feature_groups: dict[str, Any],
    dependency_findings: list[dict[str, Any]],
    models: list[ModelArtifactInspection],
    cv_calls: list[dict[str, Any]],
    routes: list[RouteEvidence],
    missing_evidence: list[SourceEvidence],
    latency_inputs: dict[str, Any],
    feature_count_mentions: list[dict[str, Any]],
) -> list[
    Issue
]:

    issues: list[
        Issue
    ] = []

    exact_datasets = [
        item
        for item
        in datasets
        if item.exact_protocol_match
    ]

    if not exact_datasets:
        issues.append(
            Issue(
                severity="BLOCKER",
                code="DATASET_NOT_RESOLVED",
                message=(
                    "No CSV was found that exactly matches "
                    "309 rows with class counts "
                    "77/77/77/78 for "
                    "focused/distracted/fatigued/overloaded."
                ),
            )
        )

    elif len(
        exact_datasets
    ) > 1:
        issues.append(
            Issue(
                severity="WARNING",
                code="MULTIPLE_DATASET_MATCHES",
                message=(
                    "More than one dataset exactly matches "
                    "the proposed Chapter-5 protocol. "
                    "The authoritative evaluation dataset "
                    "must be selected explicitly."
                ),
            )
        )

    exact_2373 = [
        item
        for item
        in schemas
        if item.feature_count
        == EXPECTED_FEATURE_COUNT
    ]

    found_2367 = [
        item
        for item
        in schemas
        if item.feature_count
        == 2367
    ]

    if not exact_2373:
        if found_2367:
            issues.append(
                Issue(
                    severity="BLOCKER",
                    code="FEATURE_COUNT_2367_VS_2373",
                    message=(
                        "Repository schema evidence contains "
                        "2,367 features, while the proposed "
                        "final-experiment protocol specifies "
                        "2,373 = 22 + 768 + 809 + 768 + 6. "
                        "Resolve this before ablation or "
                        "missing-modality experiments."
                    ),
                )
            )
        else:
            issues.append(
                Issue(
                    severity="BLOCKER",
                    code="FEATURE_SCHEMA_NOT_RESOLVED",
                    message=(
                        "No persisted 2,373-feature schema "
                        "was located."
                    ),
                )
            )

    for group, details in (
        feature_groups.items()
    ):
        if not details[
            "resolved"
        ]:
            issues.append(
                Issue(
                    severity="BLOCKER",
                    code=(
                        "FEATURE_GROUP_NOT_RESOLVED_"
                        + group.upper()
                    ),
                    message=(
                        f"No explicit repository definition "
                        f"was found for the {group} feature "
                        f"group. The inspector will not "
                        f"invent feature boundaries."
                    ),
                )
            )

        elif not details[
            "exact_expected_count"
        ]:
            observed = (
                details[
                    "chosen"
                ][
                    "feature_count"
                ]
            )

            expected = (
                details[
                    "expected_count"
                ]
            )

            issues.append(
                Issue(
                    severity="BLOCKER",
                    code=(
                        "FEATURE_GROUP_COUNT_MISMATCH_"
                        + group.upper()
                    ),
                    message=(
                        f"{group} feature group contains "
                        f"{observed} features; proposed "
                        f"protocol expects {expected}."
                    ),
                )
            )

    derived_details = feature_groups.get(
        "derived",
        {},
    )

    if (
        not derived_details.get(
            "resolved",
            False,
        )
        or
        not derived_details.get(
            "exact_expected_count",
            False,
        )
    ):
        issues.append(
            Issue(
                severity="BLOCKER",
                code="SIX_DERIVED_FEATURES_NOT_RESOLVED",
                message=(
                    "The six derived predictors have not "
                    "been resolved explicitly."
                ),
            )
        )

    if not dependency_findings:
        issues.append(
            Issue(
                severity="BLOCKER",
                code="DERIVED_DEPENDENCIES_NOT_RESOLVED",
                message=(
                    "No explicit mapping was found describing "
                    "which modalities each derived predictor "
                    "depends on. This is required for a valid "
                    "leave-one-modality-out experiment."
                ),
            )
        )

    rf_models = [
        model
        for model
        in models
        if model.classifier_type
        and
        "RandomForestClassifier"
        in model.classifier_type
    ]

    if not rf_models:
        issues.append(
            Issue(
                severity="BLOCKER",
                code="DEPLOYED_RF_NOT_RESOLVED",
                message=(
                    "No loadable fusion model artifact "
                    "containing RandomForestClassifier was "
                    "resolved."
                ),
            )
        )

    elif len(
        rf_models
    ) > 1:
        parameter_signatures = {
            json.dumps(
                model.classifier_parameters,
                sort_keys=True,
                default=str,
            )
            for model
            in rf_models
        }

        if len(
            parameter_signatures
        ) > 1:
            issues.append(
                Issue(
                    severity="WARNING",
                    code="MULTIPLE_RF_CONFIGURATIONS",
                    message=(
                        "Multiple fusion Random Forest "
                        "artifacts with different parameter "
                        "configurations were found. Confirm "
                        "which one corresponds to the "
                        "Chapter-5 complete-fusion result."
                    ),
                )
            )

    matching_cv = [
        item
        for item
        in cv_calls
        if item[
            "protocol_match"
        ]
    ]

    if not matching_cv:
        issues.append(
            Issue(
                severity="BLOCKER",
                code="CV_PROTOCOL_NOT_RESOLVED",
                message=(
                    "No StratifiedKFold call was found with "
                    "n_splits=5, shuffle=True, "
                    "random_state=42."
                ),
            )
        )

    predict_routes = [
        route
        for route
        in routes
        if route.route
        == "/predict_live"
    ]

    if not predict_routes:
        issues.append(
            Issue(
                severity="BLOCKER",
                code="LIVE_PREDICT_ROUTE_NOT_RESOLVED",
                message=(
                    "FastAPI POST /predict_live was not "
                    "located."
                ),
            )
        )

    if not missing_evidence:
        issues.append(
            Issue(
                severity="BLOCKER",
                code="ZERO_FILL_NOT_RESOLVED",
                message=(
                    "No source evidence relating to "
                    "missing-value or zero-fill behaviour "
                    "was located. Do not implement the "
                    "missing-modality experiment until the "
                    "production behaviour is confirmed."
                ),
            )
        )

    if not latency_inputs.get(
        "suitable_input_directory_found",
        False,
    ):
        issues.append(
            Issue(
                severity="WARNING",
                code="LATENCY_INPUTS_NOT_CONFIRMED",
                message=(
                    "No obvious repository directory was "
                    "found containing both audio and image "
                    "inputs suitable for end-to-end latency "
                    "benchmarking."
                ),
            )
        )

    has_2367_mention = any(
        (
            "2367"
            in item[
                "line"
            ]
            or
            "2,367"
            in item[
                "line"
            ]
        )
        for item
        in feature_count_mentions
    )

    has_2373_mention = any(
        (
            "2373"
            in item[
                "line"
            ]
            or
            "2,373"
            in item[
                "line"
            ]
        )
        for item
        in feature_count_mentions
    )

    if (
        has_2367_mention
        and
        has_2373_mention
    ):
        issues.append(
            Issue(
                severity="BLOCKER",
                code="DOCUMENTED_FEATURE_COUNT_CONFLICT",
                message=(
                    "Repository documentation/source contains "
                    "both 2,367 and 2,373 feature-count "
                    "references. Determine the authoritative "
                    "fusion schema before proceeding."
                ),
            )
        )

    if (
        python_version_short()
        != EXPECTED_REPORT_PYTHON_VERSION
    ):
        issues.append(
            Issue(
                severity="WARNING",
                code="PYTHON_VERSION_DIFFERENCE",
                message=(
                    "Current interpreter is "
                    f"{python_version_short()}, while the "
                    "specified report environment is "
                    f"{EXPECTED_REPORT_PYTHON_VERSION}."
                ),
            )
        )

    return issues


# =============================================================================
# REPORT GENERATION
# =============================================================================

def choose_primary_dataset(
    datasets: list[
        DatasetCandidate
    ],
) -> Optional[
    DatasetCandidate
]:

    exact = [
        item
        for item
        in datasets
        if item.exact_protocol_match
    ]

    if len(
        exact
    ) == 1:
        return exact[
            0
        ]

    return None


def choose_primary_schema(
    schemas: list[
        SchemaCandidate
    ],
) -> Optional[
    SchemaCandidate
]:

    exact = [
        item
        for item
        in schemas
        if item.feature_count
        == EXPECTED_FEATURE_COUNT
    ]

    if len(
        exact
    ) == 1:
        return exact[
            0
        ]

    if exact:
        return exact[
            0
        ]

    candidates_2367 = [
        item
        for item
        in schemas
        if item.feature_count
        == 2367
    ]

    if candidates_2367:
        return candidates_2367[
            0
        ]

    return None


def choose_primary_rf(
    models: list[
        ModelArtifactInspection
    ],
) -> Optional[
    ModelArtifactInspection
]:

    candidates = [
        model
        for model
        in models
        if (
            model.classifier_type
            and
            "RandomForestClassifier"
            in model.classifier_type
        )
    ]

    if not candidates:
        return None

    # Prefer canonical fusion_demo path where available.
    candidates.sort(
        key=lambda model: (
            0
            if "fusion_demo"
            in model.path.lower()
            else 1,
            0
            if "fusion_pipeline"
            in model.path.lower()
            else 1,
            model.path,
        )
    )

    return candidates[
        0
    ]


def build_report(
    *,
    root: Path,
    datasets: list[DatasetCandidate],
    schemas: list[SchemaCandidate],
    feature_groups: dict[str, Any],
    dependency_findings: list[dict[str, Any]],
    derived_occurrences: dict[str, list[dict[str, Any]]],
    models: list[ModelArtifactInspection],
    cv_calls: list[dict[str, Any]],
    cv_evidence: list[SourceEvidence],
    missing_evidence: list[SourceEvidence],
    routes: list[RouteEvidence],
    fastapi_evidence: list[SourceEvidence],
    latency_inputs: dict[str, Any],
    feature_count_mentions: list[dict[str, Any]],
    issues: list[Issue],
) -> dict[str, Any]:

    primary_dataset = choose_primary_dataset(
        datasets
    )

    primary_schema = choose_primary_schema(
        schemas
    )

    primary_rf = choose_primary_rf(
        models
    )

    blocker_count = sum(
        1
        for issue
        in issues
        if issue.severity
        == "BLOCKER"
    )

    warning_count = sum(
        1
        for issue
        in issues
        if issue.severity
        == "WARNING"
    )

    return {
        "project":
            "SenseFuzeAI",

        "inspection":
            "final evaluation experiments pre-flight",

        "read_only":
            True,

        "repository_root":
            str(
                root.resolve()
            ),

        "environment": {
            "python_version":
                python_version_short(),

            "python_full":
                sys.version,

            "platform":
                platform.platform(),

            "processor":
                platform.processor(),

            "cpu_count":
                os.cpu_count(),
        },

        "required_protocol": {
            "records":
                EXPECTED_RECORD_COUNT,

            "class_counts":
                EXPECTED_CLASS_COUNTS,

            "cv": {
                "type":
                    "StratifiedKFold",

                "n_splits":
                    EXPECTED_CV_FOLDS,

                "shuffle":
                    EXPECTED_CV_SHUFFLE,

                "random_state":
                    EXPECTED_RANDOM_STATE,
            },

            "metrics": {
                "primary":
                    "macro_f1",

                "complementary":
                    "accuracy",

                "variability":
                    "standard deviation across five folds",
            },

            "execution":
                "CPU only",

            "report_python_version":
                EXPECTED_REPORT_PYTHON_VERSION,

            "proposed_feature_groups":
                EXPECTED_FEATURE_GROUP_COUNTS,

            "proposed_total_features":
                EXPECTED_FEATURE_COUNT,
        },

        "resolved": {
            "dataset":
                (
                    asdict(
                        primary_dataset
                    )
                    if primary_dataset
                    else None
                ),

            "feature_schema":
                (
                    asdict(
                        primary_schema
                    )
                    if primary_schema
                    else None
                ),

            "feature_groups":
                feature_groups,

            "derived_dependency_mapping":
                (
                    dependency_findings[
                        0
                    ]
                    if dependency_findings
                    else None
                ),

            "deployed_random_forest":
                (
                    asdict(
                        primary_rf
                    )
                    if primary_rf
                    else None
                ),

            "matching_cv_implementations":
                [
                    item
                    for item
                    in cv_calls
                    if item[
                        "protocol_match"
                    ]
                ],

            "predict_live_routes":
                [
                    asdict(
                        route
                    )
                    for route
                    in routes
                    if route.route
                    == "/predict_live"
                ],

            "latency_inputs":
                latency_inputs,
        },

        "all_candidates": {
            "datasets":
                [
                    asdict(
                        item
                    )
                    for item
                    in datasets
                ],

            "schemas":
                [
                    asdict(
                        item
                    )
                    for item
                    in schemas
                ],

            "model_artifacts":
                [
                    asdict(
                        item
                    )
                    for item
                    in models
                ],

            "cv_calls":
                cv_calls,

            "fastapi_routes":
                [
                    asdict(
                        item
                    )
                    for item
                    in routes
                ],

            "derived_dependency_findings":
                dependency_findings,

            "derived_feature_occurrences":
                derived_occurrences,

            "feature_count_mentions":
                feature_count_mentions,
        },

        "source_evidence": {
            "cross_validation":
                [
                    asdict(
                        item
                    )
                    for item
                    in cv_evidence[
                        :100
                    ]
                ],

            "missing_modality_zero_fill":
                [
                    asdict(
                        item
                    )
                    for item
                    in missing_evidence[
                        :150
                    ]
                ],

            "fastapi_prediction":
                [
                    asdict(
                        item
                    )
                    for item
                    in fastapi_evidence[
                        :100
                    ]
                ],
        },

        "issues":
            [
                asdict(
                    issue
                )
                for issue
                in issues
            ],

        "summary": {
            "blockers":
                blocker_count,

            "warnings":
                warning_count,

            "ready_for_experiment_implementation":
                blocker_count
                == 0,
        },

        "proposed_scripts_after_approval": [
            {
                "filename":
                    "evaluate_leave_one_modality_out.py",

                "purpose":
                    (
                        "Five-fold Random Forest retraining "
                        "with one modality removed at a time, "
                        "using the authoritative feature schema "
                        "and derived-feature dependency map."
                    ),
            },
            {
                "filename":
                    "evaluate_missing_modality_robustness.py",

                "purpose":
                    (
                        "Inference-time modality suppression "
                        "using the frozen complete-fusion model "
                        "and exactly the production missing-input "
                        "representation."
                    ),
            },
            {
                "filename":
                    "benchmark_cpu_inference_latency.py",

                "purpose":
                    (
                        "Repeated CPU-only latency benchmarking "
                        "of the canonical inference path with "
                        "warm-up, repeated measurements, "
                        "distribution summaries and CSV/JSON "
                        "exports."
                    ),
            },
        ],
    }


def markdown_table_row(
    cells: Iterable[Any],
) -> str:
    return (
        "| "
        + " | ".join(
            str(
                cell
            ).replace(
                "\n",
                " "
            )
            for cell
            in cells
        )
        + " |"
    )


def report_to_markdown(
    report: dict[str, Any],
) -> str:

    lines: list[
        str
    ] = []

    lines.append(
        "# SenseFuzeAI Final Experiment Pre-Flight Report"
    )

    lines.append(
        ""
    )

    lines.append(
        (
            "**Status:** "
            + (
                "READY FOR EXPERIMENT IMPLEMENTATION"
                if report[
                    "summary"
                ][
                    "ready_for_experiment_implementation"
                ]
                else
                "NOT READY — RESOLVE BLOCKERS FIRST"
            )
        )
    )

    lines.append(
        ""
    )

    lines.append(
        "## Required Protocol"
    )

    lines.append(
        ""
    )

    protocol = report[
        "required_protocol"
    ]

    lines.append(
        markdown_table_row(
            [
                "Item",
                "Required value",
            ]
        )
    )

    lines.append(
        markdown_table_row(
            [
                "---",
                "---",
            ]
        )
    )

    lines.append(
        markdown_table_row(
            [
                "Records",
                protocol[
                    "records"
                ],
            ]
        )
    )

    lines.append(
        markdown_table_row(
            [
                "Class counts",
                json.dumps(
                    protocol[
                        "class_counts"
                    ],
                    sort_keys=True,
                ),
            ]
        )
    )

    lines.append(
        markdown_table_row(
            [
                "CV",
                (
                    "StratifiedKFold("
                    "n_splits=5, "
                    "shuffle=True, "
                    "random_state=42)"
                ),
            ]
        )
    )

    lines.append(
        markdown_table_row(
            [
                "Primary metric",
                "macro-F1",
            ]
        )
    )

    lines.append(
        markdown_table_row(
            [
                "Complementary metric",
                "accuracy",
            ]
        )
    )

    lines.append(
        markdown_table_row(
            [
                "Variability",
                "fold standard deviation",
            ]
        )
    )

    lines.append(
        markdown_table_row(
            [
                "Proposed total features",
                protocol[
                    "proposed_total_features"
                ],
            ]
        )
    )

    lines.append(
        ""
    )

    lines.append(
        "## Resolved Dataset"
    )

    lines.append(
        ""
    )

    dataset = report[
        "resolved"
    ][
        "dataset"
    ]

    if dataset:
        lines.append(
            f"- Path: `{dataset['path']}`"
        )

        lines.append(
            f"- Rows: {dataset['rows']}"
        )

        lines.append(
            f"- Label column: `{dataset['label_column']}`"
        )

        lines.append(
            (
                "- Class counts: `"
                + json.dumps(
                    dataset[
                        "class_counts"
                    ],
                    sort_keys=True,
                )
                + "`"
            )
        )

    else:
        lines.append(
            "Not uniquely resolved."
        )

    lines.append(
        ""
    )

    lines.append(
        "## Resolved Feature Schema"
    )

    lines.append(
        ""
    )

    schema = report[
        "resolved"
    ][
        "feature_schema"
    ]

    if schema:
        lines.append(
            f"- Path: `{schema['path']}`"
        )

        lines.append(
            f"- Source: `{schema['source']}`"
        )

        lines.append(
            f"- Key: `{schema['key']}`"
        )

        lines.append(
            f"- Feature count: **{schema['feature_count']}**"
        )

    else:
        lines.append(
            "Not resolved."
        )

    lines.append(
        ""
    )

    lines.append(
        "## Feature Groups"
    )

    lines.append(
        ""
    )

    lines.append(
        markdown_table_row(
            [
                "Group",
                "Expected",
                "Resolved",
                "Observed",
                "Evidence",
            ]
        )
    )

    lines.append(
        markdown_table_row(
            [
                "---",
                "---:",
                "---",
                "---:",
                "---",
            ]
        )
    )

    for group, details in report[
        "resolved"
    ][
        "feature_groups"
    ].items():

        chosen = details.get(
            "chosen"
        )

        lines.append(
            markdown_table_row(
                [
                    group,
                    details[
                        "expected_count"
                    ],
                    details[
                        "resolved"
                    ],
                    (
                        chosen[
                            "feature_count"
                        ]
                        if chosen
                        else ""
                    ),
                    (
                        (
                            f"{chosen['path']} :: "
                            f"{chosen['key']}"
                        )
                        if chosen
                        else ""
                    ),
                ]
            )
        )

    lines.append(
        ""
    )

    lines.append(
        "## Deployed Random Forest"
    )

    lines.append(
        ""
    )

    rf = report[
        "resolved"
    ][
        "deployed_random_forest"
    ]

    if rf:
        lines.append(
            f"- Artifact: `{rf['path']}`"
        )

        lines.append(
            f"- Classifier: `{rf['classifier_type']}`"
        )

        lines.append(
            ""
        )

        lines.append(
            "```json"
        )

        lines.append(
            json.dumps(
                rf[
                    "classifier_parameters"
                ],
                indent=2,
                sort_keys=True,
            )
        )

        lines.append(
            "```"
        )

        lines.append(
            ""
        )

        lines.append(
            "### Preprocessing"
        )

        lines.append(
            ""
        )

        lines.append(
            "```json"
        )

        lines.append(
            json.dumps(
                rf[
                    "preprocessing_steps"
                ],
                indent=2,
                sort_keys=True,
            )
        )

        lines.append(
            "```"
        )

    else:
        lines.append(
            "Not resolved."
        )

    lines.append(
        ""
    )

    lines.append(
        "## Methodological Issues"
    )

    lines.append(
        ""
    )

    issues = report[
        "issues"
    ]

    if not issues:
        lines.append(
            "No blocking issues detected."
        )

    else:
        for issue in issues:
            lines.append(
                (
                    f"- **{issue['severity']} — "
                    f"{issue['code']}**: "
                    f"{issue['message']}"
                )
            )

    lines.append(
        ""
    )

    lines.append(
        "## Proposed Scripts After Approval"
    )

    lines.append(
        ""
    )

    for script in report[
        "proposed_scripts_after_approval"
    ]:
        lines.append(
            (
                f"- `{script['filename']}` — "
                f"{script['purpose']}"
            )
        )

    lines.append(
        ""
    )

    return "\n".join(
        lines
    )


# =============================================================================
# CONSOLE SUMMARY
# =============================================================================

def print_summary(
    report: dict[str, Any],
) -> None:

    print()
    print(
        "=" * 88
    )
    print(
        "SenseFuzeAI Final Evaluation Experiments - Pre-Flight Inspection"
    )
    print(
        "=" * 88
    )

    resolved = report[
        "resolved"
    ]

    print()

    print(
        "1. Dataset:"
    )

    dataset = resolved[
        "dataset"
    ]

    if dataset:
        print(
            f"   {dataset['path']}"
        )
        print(
            f"   rows={dataset['rows']}"
        )
        print(
            f"   label={dataset['label_column']}"
        )
        print(
            f"   classes={dataset['class_counts']}"
        )
    else:
        print(
            "   NOT UNIQUELY RESOLVED"
        )

    print()

    print(
        "2. Feature schema:"
    )

    schema = resolved[
        "feature_schema"
    ]

    if schema:
        print(
            f"   {schema['path']}"
        )
        print(
            f"   key={schema['key']}"
        )
        print(
            f"   feature_count={schema['feature_count']}"
        )
    else:
        print(
            "   NOT RESOLVED"
        )

    print()

    print(
        "3. Feature groups:"
    )

    for group, details in resolved[
        "feature_groups"
    ].items():

        chosen = details.get(
            "chosen"
        )

        if chosen:
            print(
                (
                    f"   {group:10s} "
                    f"expected={details['expected_count']:4d} "
                    f"observed={chosen['feature_count']:4d} "
                    f"source={chosen['path']}::{chosen['key']}"
                )
            )
        else:
            print(
                (
                    f"   {group:10s} "
                    f"expected={details['expected_count']:4d} "
                    "observed=UNRESOLVED"
                )
            )

    print()

    print(
        "4. Derived-feature dependency mapping:"
    )

    dependency = resolved[
        "derived_dependency_mapping"
    ]

    if dependency:
        print(
            f"   {dependency['path']}::{dependency['key']}"
        )
        print(
            (
                "   "
                + json.dumps(
                    dependency[
                        "mapping"
                    ],
                    sort_keys=True,
                )
            )
        )
    else:
        print(
            "   NOT RESOLVED"
        )

    print()

    print(
        "5. Deployed Random Forest:"
    )

    rf = resolved[
        "deployed_random_forest"
    ]

    if rf:
        print(
            f"   artifact={rf['path']}"
        )
        print(
            f"   classifier={rf['classifier_type']}"
        )

        for key, value in sorted(
            rf[
                "classifier_parameters"
            ].items()
        ):
            print(
                f"   {key}={value}"
            )
    else:
        print(
            "   NOT RESOLVED"
        )

    print()

    print(
        "6. Matching five-fold CV implementations:"
    )

    cv_matches = resolved[
        "matching_cv_implementations"
    ]

    if cv_matches:
        for item in cv_matches:
            print(
                (
                    f"   {item['path']}:"
                    f"{item['line_number']} "
                    f"{item['kwargs']}"
                )
            )
    else:
        print(
            "   NONE RESOLVED"
        )

    print()

    print(
        "7. FastAPI /predict_live:"
    )

    routes = resolved[
        "predict_live_routes"
    ]

    if routes:
        for route in routes:
            print(
                (
                    f"   {route['method']} "
                    f"{route['route']} "
                    f"-> {route['path']}:"
                    f"{route['line_number']} "
                    f"{route['function_name']}()"
                )
            )
    else:
        print(
            "   NOT RESOLVED"
        )

    print()

    print(
        "8. Latency benchmark inputs:"
    )

    for item in resolved[
        "latency_inputs"
    ][
        "candidate_input_directories"
    ]:
        print(
            (
                f"   {item['path']}: "
                f"audio={item['audio_files']}, "
                f"image={item['image_files']}, "
                f"video={item['video_files']}, "
                f"json={item['json_files']}, "
                f"text={item['text_files']}"
            )
        )

    print()

    print(
        "9. Methodological issues:"
    )

    if not report[
        "issues"
    ]:
        print(
            "   None."
        )
    else:
        for issue in report[
            "issues"
        ]:
            print(
                (
                    f"   [{issue['severity']}] "
                    f"{issue['code']}: "
                    f"{issue['message']}"
                )
            )

    print()

    print(
        "=" * 88
    )

    if report[
        "summary"
    ][
        "ready_for_experiment_implementation"
    ]:
        print(
            "PRE-FLIGHT RESULT: READY FOR EXPERIMENT IMPLEMENTATION"
        )
    else:
        print(
            "PRE-FLIGHT RESULT: NOT READY - RESOLVE BLOCKERS FIRST"
        )

    print(
        "=" * 88
    )

    print()


# =============================================================================
# ARGUMENT PARSER
# =============================================================================

def build_argument_parser() -> argparse.ArgumentParser:

    parser = argparse.ArgumentParser(
        description=(
            "Read-only SenseFuzeAI pre-flight inspection "
            "for final ablation, missing-modality and "
            "latency experiments."
        )
    )

    parser.add_argument(
        "--root",
        type=Path,
        default=Path(
            "."
        ),
        help=(
            "SenseFuzeAI project root. "
            "Default: current directory."
        ),
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Report directory. Default: "
            "<root>/data/processed/"
            "final_experiment_preflight/"
        ),
    )

    parser.add_argument(
        "--verbose",
        action="store_true",
        help=(
            "Print progress while scanning datasets."
        ),
    )

    return parser


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:

    parser = build_argument_parser()

    args = parser.parse_args()

    root = args.root.resolve()

    if not root.exists():
        parser.error(
            f"Project root does not exist: {root}"
        )

    if not root.is_dir():
        parser.error(
            f"Project root is not a directory: {root}"
        )

    output_dir = (
        args.output_dir.resolve()
        if args.output_dir
        else
        (
            root
            / "data"
            / "processed"
            / "final_experiment_preflight"
        )
    )

    print()
    print(
        "Inspecting SenseFuzeAI repository..."
    )
    print(
        f"Root: {root}"
    )
    print(
        "No training or experiment execution will occur."
    )
    print()

    try:

        datasets = find_dataset_candidates(
            root,
            verbose=(
                args.verbose
            ),
        )

        schemas = find_schema_candidates(
            root
        )

        feature_groups = (
            consolidate_feature_groups(
                root
            )
        )

        dependency_findings = (
            find_dependency_dicts(
                root
            )
        )

        derived_features: list[
            str
        ] = []

        derived_group = feature_groups.get(
            "derived",
            {}
        )

        chosen_derived = derived_group.get(
            "chosen"
        )

        if chosen_derived:
            derived_features = list(
                chosen_derived.get(
                    "features",
                    [],
                )
            )

        derived_occurrences = (
            locate_feature_occurrences(
                root,
                derived_features,
            )
        )

        models = find_fusion_model_artifacts(
            root
        )

        cv_calls = (
            inspect_stratified_kfold_calls(
                root
            )
        )

        cv_evidence = source_evidence(
            root,
            SOURCE_SEARCH_TERMS[
                "cv"
            ],
        )

        missing_evidence = source_evidence(
            root,
            SOURCE_SEARCH_TERMS[
                "missing_modality"
            ],
        )

        routes = inspect_fastapi_routes(
            root
        )

        fastapi_evidence = source_evidence(
            root,
            SOURCE_SEARCH_TERMS[
                "fastapi"
            ],
        )

        latency_inputs = inspect_latency_inputs(
            root
        )

        feature_count_mentions = (
            find_feature_count_mentions(
                root
            )
        )

        issues = build_issues(
            datasets=datasets,
            schemas=schemas,
            feature_groups=feature_groups,
            dependency_findings=(
                dependency_findings
            ),
            models=models,
            cv_calls=cv_calls,
            routes=routes,
            missing_evidence=(
                missing_evidence
            ),
            latency_inputs=(
                latency_inputs
            ),
            feature_count_mentions=(
                feature_count_mentions
            ),
        )

        report = build_report(
            root=root,
            datasets=datasets,
            schemas=schemas,
            feature_groups=(
                feature_groups
            ),
            dependency_findings=(
                dependency_findings
            ),
            derived_occurrences=(
                derived_occurrences
            ),
            models=models,
            cv_calls=cv_calls,
            cv_evidence=cv_evidence,
            missing_evidence=(
                missing_evidence
            ),
            routes=routes,
            fastapi_evidence=(
                fastapi_evidence
            ),
            latency_inputs=(
                latency_inputs
            ),
            feature_count_mentions=(
                feature_count_mentions
            ),
            issues=issues,
        )

        output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        json_path = (
            output_dir
            / "preflight_report.json"
        )

        markdown_path = (
            output_dir
            / "preflight_report.md"
        )

        with json_path.open(
            "w",
            encoding="utf-8",
        ) as file_handle:
            json.dump(
                report,
                file_handle,
                indent=2,
                ensure_ascii=False,
            )

        markdown_path.write_text(
            report_to_markdown(
                report
            ),
            encoding="utf-8",
        )

        print_summary(
            report
        )

        print(
            f"JSON report:     {json_path}"
        )

        print(
            f"Markdown report: {markdown_path}"
        )

        print()

        blocker_count = report[
            "summary"
        ][
            "blockers"
        ]

        raise SystemExit(
            1
            if blocker_count
            > 0
            else 0
        )

    except SystemExit:
        raise

    except Exception as exc:

        print(
            "PRE-FLIGHT INSPECTION FAILED"
        )

        print(
            f"{type(exc).__name__}: {exc}"
        )

        if args.verbose:
            print()
            traceback.print_exc()

        raise SystemExit(
            1
        )


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    main()
