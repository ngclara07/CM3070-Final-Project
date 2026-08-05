"""
evaluate_multimodal_results.py

SenseFuzeAI
Canonical Multimodal + Temporal Fusion Evaluation Pipeline

=============================================================================
PURPOSE
=============================================================================

This script evaluates the final SenseFuzeAI multimodal behavioural-state
classifier at TWO explicitly separated inference stages:

    1. RAW MULTIMODAL FUSION
       ---------------------------------------------------------
       FinalMultimodalInference.predict(...)
                    |
                    v
       one four-class probability vector

    2. TEMPORAL MULTIMODAL FUSION
       ---------------------------------------------------------
       raw probability vectors
                    |
                    v
       TemporalFusionEngine
                    |
                    v
       rolling arithmetic probability mean
                    |
                    v
       temporally stabilised behavioural prediction


Canonical classes:

    focused
    distracted
    fatigued
    overloaded


Canonical temporal implementation:

    temporal_fusion.py


=============================================================================
IMPORTANT ARCHITECTURAL RULE
=============================================================================

This evaluation script DOES NOT independently implement:

    - probability normalisation
    - temporal averaging
    - confidence-gap thresholds
    - rolling-window state
    - temporal reset semantics

Those behaviours are imported directly from temporal_fusion.py.

This prevents evaluation-time logic from silently diverging from:

    live_fusion_gui.py
    web_app/app.py


=============================================================================
SUPPORTED EVALUATION
=============================================================================

When ground-truth labels are available, this script calculates:

    RAW FUSION:
        - accuracy
        - balanced accuracy
        - macro precision
        - macro recall
        - macro F1
        - weighted F1
        - multiclass log loss
        - multiclass Brier score
        - expected calibration error
        - confusion matrix
        - per-class precision / recall / F1

    TEMPORAL FUSION:
        - same metrics as above

    FULL-WINDOW TEMPORAL FUSION:
        - evaluates only observations where temporal_samples == window size

    STABILITY:
        - raw label-switch rate
        - temporal label-switch rate
        - difference in switching behaviour

    PARITY:
        - compares recomputed TemporalFusionEngine probabilities with any
          temporal probabilities already stored in the source file

        This is useful for verifying that web/desktop logged output remains
        identical to the canonical temporal implementation.


=============================================================================
SEQUENCE BOUNDARIES
=============================================================================

Temporal fusion MUST NOT combine unrelated experimental conditions.

When available, this script automatically groups observations by:

    session_id
    trial_id / condition_id
    generation

The generation field is particularly important because:

    reset()
        ->
    generation increment
        ->
    new temporal history

Therefore observations from different generations are never averaged together.


=============================================================================
GROUND-TRUTH REQUIREMENT
=============================================================================

Classifier accuracy CANNOT be calculated without ground-truth labels.

If no ground-truth column is available, this script still performs:

    - temporal reconstruction
    - raw/temporal probability validation
    - temporal parity checks
    - switch-rate analysis
    - descriptive confidence statistics

but reports behavioural accuracy as:

    NOT ESTABLISHED


=============================================================================
"""

from __future__ import annotations

import argparse
import ast
import json
import math
import sys

from pathlib import Path
from typing import Any, Iterable, Mapping, Optional

import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    log_loss,
    precision_recall_fscore_support,
)

from temporal_fusion import (
    LABELS,
    TEMPORAL_PROBABILITY_WINDOW,
    PROBABILITY_SUM_TOLERANCE,
    TemporalFusionEngine,
    normalise_probability_dict,
    summarise_probability_dict,
    validate_probability_distribution,
)


# =============================================================================
# CONSTANTS
# =============================================================================

CANONICAL_LABELS = tuple(
    LABELS
)


# -----------------------------------------------------------------------------
# Automatic column discovery
# -----------------------------------------------------------------------------

GROUND_TRUTH_CANDIDATES = [
    "true_label",
    "ground_truth",
    "expected_state",
    "true_state",
    "target",
    "behaviour_state",
    "behavior_state",
    "label",
]

SESSION_COLUMN_CANDIDATES = [
    "session_id",
    "session",
    "participant_session",
]

TRIAL_COLUMN_CANDIDATES = [
    "trial_id",
    "condition_id",
    "trial",
    "condition",
    "run_id",
]

GENERATION_COLUMN_CANDIDATES = [
    "generation",
    "reset_generation",
]

ORDER_COLUMN_CANDIDATES = [
    "sequence_index",
    "observation_index",
    "sample_index",
    "timestamp_epoch",
    "timestamp_perf",
    "timestamp",
    "datetime",
    "time",
]

RAW_PROBABILITY_COLUMN_CANDIDATES = [
    "raw_probabilities",
    "raw_probability_dict",
    "raw_probs",
]

LOGGED_TEMPORAL_PROBABILITY_COLUMN_CANDIDATES = [
    "temporal_probabilities",
    "temporal_probability_dict",
    "temporal_probs",
]

RAW_STATE_COLUMN_CANDIDATES = [
    "raw_fusion_state",
    "raw_top_class",
    "raw_prediction",
]

TEMPORAL_STATE_COLUMN_CANDIDATES = [
    "final_state",
    "temporal_state",
    "current_state",
]

RUNTIME_COLUMN_CANDIDATES = [
    "runtime_sec",
    "runtime_seconds",
    "inference_runtime_sec",
    "prediction_runtime_sec",
]


# =============================================================================
# JSON SERIALISATION
# =============================================================================

def json_default(
    value: Any,
) -> Any:
    """
    JSON serializer for NumPy/Pandas values.
    """

    if isinstance(
        value,
        np.integer,
    ):

        return int(
            value
        )

    if isinstance(
        value,
        np.floating,
    ):

        numeric = float(
            value
        )

        if math.isfinite(
            numeric
        ):

            return numeric

        return None

    if isinstance(
        value,
        np.ndarray,
    ):

        return value.tolist()

    if isinstance(
        value,
        Path,
    ):

        return str(
            value
        )

    if pd.isna(
        value
    ):

        return None

    raise TypeError(
        f"Object of type {type(value).__name__} "
        "is not JSON serialisable."
    )


def write_json(
    path: Path,
    value: Any,
) -> None:
    """
    Save structured JSON output.
    """

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open(
        "w",
        encoding="utf-8",
    ) as file_handle:

        json.dump(
            value,
            file_handle,
            indent=4,
            default=json_default,
        )


# =============================================================================
# INPUT LOADING
# =============================================================================

def load_results_file(
    path: Path,
) -> pd.DataFrame:
    """
    Load supported evaluation result formats.
    """

    path = Path(
        path
    )

    if not path.exists():

        raise FileNotFoundError(
            f"Evaluation input not found:\n{path}"
        )

    suffix = (
        path.suffix
        .lower()
    )

    if suffix == ".csv":

        dataframe = (
            pd.read_csv(
                path
            )
        )

    elif suffix in {
        ".jsonl",
        ".ndjson",
    }:

        dataframe = (
            pd.read_json(
                path,
                lines=True,
            )
        )

    elif suffix == ".json":

        with path.open(
            "r",
            encoding="utf-8",
        ) as file_handle:

            content = json.load(
                file_handle
            )

        if isinstance(
            content,
            list,
        ):

            dataframe = (
                pd.DataFrame(
                    content
                )
            )

        elif (
            isinstance(
                content,
                dict,
            )
            and
            isinstance(
                content.get(
                    "results"
                ),
                list,
            )
        ):

            dataframe = (
                pd.DataFrame(
                    content[
                        "results"
                    ]
                )
            )

        else:

            raise ValueError(
                "JSON evaluation input must be either:\n"
                "  - a list of result objects; or\n"
                "  - an object containing a 'results' list."
            )

    elif suffix == ".parquet":

        dataframe = (
            pd.read_parquet(
                path
            )
        )

    else:

        raise ValueError(
            "Unsupported evaluation input format.\n"
            "Supported formats: "
            ".csv, .json, .jsonl, .ndjson, .parquet"
        )

    if dataframe.empty:

        raise ValueError(
            "Evaluation input contains no rows."
        )

    dataframe = (
        dataframe
        .reset_index(
            drop=True
        )
    )

    dataframe[
        "__source_row"
    ] = np.arange(
        len(
            dataframe
        ),
        dtype=int,
    )

    return dataframe


# =============================================================================
# COLUMN RESOLUTION
# =============================================================================

def find_first_existing_column(
    dataframe: pd.DataFrame,
    candidates: Iterable[str],
) -> Optional[str]:

    for candidate in candidates:

        if candidate in dataframe.columns:

            return candidate

    return None


def resolve_column(
    dataframe: pd.DataFrame,
    explicit: Optional[str],
    candidates: Iterable[str],
    *,
    required: bool = False,
    description: str = "column",
) -> Optional[str]:

    if explicit:

        if explicit not in dataframe.columns:

            raise ValueError(
                f"Requested {description} does not exist:\n"
                f"{explicit}"
            )

        return explicit

    discovered = (
        find_first_existing_column(
            dataframe,
            candidates,
        )
    )

    if (
        discovered is None
        and
        required
    ):

        raise ValueError(
            f"Could not automatically identify {description}."
        )

    return discovered


# =============================================================================
# LABEL HANDLING
# =============================================================================

def normalise_label(
    value: Any,
) -> Optional[str]:
    """
    Convert a ground-truth/predicted label into canonical form.

    Invalid/unlabelled values become None.
    """

    if value is None:

        return None

    try:

        if pd.isna(
            value
        ):

            return None

    except Exception:

        pass

    label = (
        str(value)
        .strip()
        .lower()
    )

    if label in CANONICAL_LABELS:

        return label

    return None


# =============================================================================
# PROBABILITY PARSING
# =============================================================================

def parse_probability_mapping(
    value: Any,
) -> dict[str, float]:
    """
    Parse a probability mapping from:

        - dict
        - JSON string
        - Python dictionary-like string

    Then apply canonical probability normalisation.
    """

    parsed: Any = None

    if isinstance(
        value,
        Mapping,
    ):

        parsed = dict(
            value
        )

    elif isinstance(
        value,
        str,
    ):

        text = (
            value.strip()
        )

        if not text:

            raise ValueError(
                "Empty probability string."
            )

        try:

            parsed = json.loads(
                text
            )

        except Exception:

            try:

                parsed = (
                    ast.literal_eval(
                        text
                    )
                )

            except Exception as exc:

                raise ValueError(
                    "Could not parse probability dictionary."
                ) from exc

    else:

        raise ValueError(
            "Probability value is not a dictionary or dictionary string."
        )

    if not isinstance(
        parsed,
        Mapping,
    ):

        raise ValueError(
            "Parsed probability value is not a mapping."
        )

    return (
        normalise_probability_dict(
            parsed,
            labels=CANONICAL_LABELS,
        )
    )


def find_separate_probability_columns(
    dataframe: pd.DataFrame,
    *,
    prefix_type: str,
) -> Optional[dict[str, str]]:
    """
    Detect one-column-per-class probability layouts.

    Examples accepted for raw probabilities:

        raw_focused_prob
        raw_focused_probability
        focused_raw_prob

    Examples accepted for temporal probabilities:

        temporal_focused_prob
        temporal_focused_probability
        focused_temporal_prob
    """

    mapping: dict[
        str,
        str
    ] = {}

    for label in CANONICAL_LABELS:

        candidates = [
            f"{prefix_type}_{label}_prob",
            f"{prefix_type}_{label}_probability",
            f"{label}_{prefix_type}_prob",
            f"{label}_{prefix_type}_probability",
        ]

        found = (
            find_first_existing_column(
                dataframe,
                candidates,
            )
        )

        if found is None:

            return None

        mapping[
            label
        ] = found

    return mapping


def extract_probability_vectors(
    dataframe: pd.DataFrame,
    *,
    explicit_column: Optional[str],
    mapping_candidates: Iterable[str],
    prefix_type: str,
    required: bool,
) -> tuple[
    Optional[list[dict[str, float]]],
    Optional[str],
]:

    mapping_column = (
        resolve_column(
            dataframe,
            explicit_column,
            mapping_candidates,
            required=False,
            description=(
                f"{prefix_type} probability column"
            ),
        )
    )

    if mapping_column:

        vectors: list[
            dict[str, float]
        ] = []

        for row_index, value in enumerate(
            dataframe[
                mapping_column
            ].tolist()
        ):

            try:

                vector = (
                    parse_probability_mapping(
                        value
                    )
                )

            except Exception as exc:

                raise ValueError(
                    f"Could not parse {prefix_type} probabilities "
                    f"at input row {row_index} "
                    f"from column '{mapping_column}'."
                ) from exc

            vectors.append(
                vector
            )

        return (
            vectors,
            mapping_column,
        )

    separate_columns = (
        find_separate_probability_columns(
            dataframe,
            prefix_type=(
                prefix_type
            ),
        )
    )

    if separate_columns:

        vectors = []

        for _, row in dataframe.iterrows():

            raw_mapping = {
                label:
                    row[
                        separate_columns[
                            label
                        ]
                    ]
                for label in CANONICAL_LABELS
            }

            vectors.append(
                normalise_probability_dict(
                    raw_mapping,
                    labels=(
                        CANONICAL_LABELS
                    ),
                )
            )

        description = (
            ", ".join(
                separate_columns.values()
            )
        )

        return (
            vectors,
            description,
        )

    if required:

        raise ValueError(
            f"Could not locate {prefix_type} probability values.\n\n"
            "Expected either a probability-dictionary column or "
            "one probability column per behavioural class.\n\n"
            f"For raw fusion evaluation, a typical column is:\n"
            "    raw_probabilities"
        )

    return (
        None,
        None,
    )


# =============================================================================
# SEQUENCE CONFIGURATION
# =============================================================================

def determine_group_columns(
    dataframe: pd.DataFrame,
    *,
    session_column: Optional[str],
    trial_column: Optional[str],
    generation_column: Optional[str],
) -> tuple[
    list[str],
    dict[str, Optional[str]],
]:

    resolved_session = (
        resolve_column(
            dataframe,
            session_column,
            SESSION_COLUMN_CANDIDATES,
            description="session column",
        )
    )

    resolved_trial = (
        resolve_column(
            dataframe,
            trial_column,
            TRIAL_COLUMN_CANDIDATES,
            description="trial/condition column",
        )
    )

    resolved_generation = (
        resolve_column(
            dataframe,
            generation_column,
            GENERATION_COLUMN_CANDIDATES,
            description="generation column",
        )
    )

    group_columns: list[
        str
    ] = []

    for column in (
        resolved_session,
        resolved_trial,
        resolved_generation,
    ):

        if (
            column
            and
            column not in group_columns
        ):

            group_columns.append(
                column
            )

    if not group_columns:

        dataframe[
            "__single_sequence"
        ] = "all"

        group_columns = [
            "__single_sequence"
        ]

    return (
        group_columns,
        {
            "session_column":
                resolved_session,

            "trial_column":
                resolved_trial,

            "generation_column":
                resolved_generation,
        },
    )


def determine_order_column(
    dataframe: pd.DataFrame,
    explicit: Optional[str],
) -> Optional[str]:

    return (
        resolve_column(
            dataframe,
            explicit,
            ORDER_COLUMN_CANDIDATES,
            description=(
                "sequence ordering column"
            ),
        )
    )


# =============================================================================
# CANONICAL TEMPORAL RECONSTRUCTION
# =============================================================================

def reconstruct_temporal_predictions(
    dataframe: pd.DataFrame,
    raw_vectors: list[
        dict[str, float]
    ],
    *,
    group_columns: list[str],
    order_column: Optional[str],
    window_size: int,
) -> pd.DataFrame:
    """
    Reconstruct temporal predictions exclusively through
    TemporalFusionEngine.

    A separate engine is created for every temporal sequence.
    """

    working = (
        dataframe.copy()
    )

    working[
        "__raw_probability_object"
    ] = raw_vectors

    sort_columns = list(
        group_columns
    )

    if order_column:

        sort_columns.append(
            order_column
        )

    else:

        sort_columns.append(
            "__source_row"
        )

    working = (
        working.sort_values(
            by=sort_columns,
            kind="mergesort",
        )
        .reset_index(
            drop=True
        )
    )

    computed_rows: list[
        dict[str, Any]
    ] = []

    grouped = (
        working.groupby(
            group_columns,
            dropna=False,
            sort=False,
        )
    )

    for group_key, group in grouped:

        engine = (
            TemporalFusionEngine(
                window_size=(
                    window_size
                ),
                labels=(
                    CANONICAL_LABELS
                ),
            )
        )

        for _, row in group.iterrows():

            raw_probabilities = (
                row[
                    "__raw_probability_object"
                ]
            )

            raw_summary = (
                summarise_probability_dict(
                    raw_probabilities,
                    labels=(
                        CANONICAL_LABELS
                    ),
                )
            )

            temporal_summary = (
                engine.append(
                    raw_probabilities
                )
            )

            raw_validation = (
                validate_probability_distribution(
                    raw_probabilities,
                    labels=(
                        CANONICAL_LABELS
                    ),
                    tolerance=(
                        PROBABILITY_SUM_TOLERANCE
                    ),
                )
            )

            temporal_validation = (
                validate_probability_distribution(
                    temporal_summary[
                        "probabilities"
                    ],
                    labels=(
                        CANONICAL_LABELS
                    ),
                    tolerance=(
                        PROBABILITY_SUM_TOLERANCE
                    ),
                )
            )

            result = (
                row.to_dict()
            )

            result.update(
                {
                    "eval_raw_state":
                        raw_summary[
                            "current_state"
                        ],

                    "eval_raw_confidence":
                        raw_summary[
                            "confidence"
                        ],

                    "eval_raw_confidence_percent":
                        raw_summary[
                            "confidence_percent"
                        ],

                    "eval_raw_second_class":
                        raw_summary[
                            "second_class"
                        ],

                    "eval_raw_second_probability":
                        raw_summary[
                            "second_probability"
                        ],

                    "eval_raw_confidence_gap":
                        raw_summary[
                            "confidence_gap"
                        ],

                    "eval_raw_confidence_level":
                        raw_summary[
                            "confidence_level"
                        ],

                    "eval_raw_probabilities":
                        json.dumps(
                            raw_probabilities
                        ),

                    "eval_raw_probability_sum":
                        raw_validation[
                            "probability_sum"
                        ],

                    "eval_raw_probability_valid":
                        raw_validation[
                            "valid"
                        ],

                    "eval_temporal_state":
                        temporal_summary[
                            "current_state"
                        ],

                    "eval_temporal_confidence":
                        temporal_summary[
                            "confidence"
                        ],

                    "eval_temporal_confidence_percent":
                        temporal_summary[
                            "confidence_percent"
                        ],

                    "eval_temporal_second_class":
                        temporal_summary[
                            "second_class"
                        ],

                    "eval_temporal_second_probability":
                        temporal_summary[
                            "second_probability"
                        ],

                    "eval_temporal_confidence_gap":
                        temporal_summary[
                            "confidence_gap"
                        ],

                    "eval_temporal_confidence_level":
                        temporal_summary[
                            "confidence_level"
                        ],

                    "eval_temporal_probabilities":
                        json.dumps(
                            temporal_summary[
                                "probabilities"
                            ]
                        ),

                    "eval_temporal_samples":
                        temporal_summary[
                            "temporal_samples"
                        ],

                    "eval_temporal_window":
                        temporal_summary[
                            "temporal_window"
                        ],

                    "eval_temporal_window_full":
                        temporal_summary[
                            "temporal_window_full"
                        ],

                    "eval_temporal_probability_sum":
                        temporal_validation[
                            "probability_sum"
                        ],

                    "eval_temporal_probability_valid":
                        temporal_validation[
                            "valid"
                        ],
                }
            )

            for label in CANONICAL_LABELS:

                result[
                    f"eval_raw_{label}_prob"
                ] = float(
                    raw_probabilities[
                        label
                    ]
                )

                result[
                    f"eval_temporal_{label}_prob"
                ] = float(
                    temporal_summary[
                        "probabilities"
                    ][
                        label
                    ]
                )

            computed_rows.append(
                result
            )

    output = (
        pd.DataFrame(
            computed_rows
        )
    )

    output = (
        output.sort_values(
            "__source_row",
            kind="mergesort",
        )
        .reset_index(
            drop=True
        )
    )

    return output


# =============================================================================
# PROBABILITY MATRIX
# =============================================================================

def probability_matrix(
    dataframe: pd.DataFrame,
    *,
    prefix: str,
) -> np.ndarray:

    return np.asarray(
        [
            [
                float(
                    row[
                        f"{prefix}_{label}_prob"
                    ]
                )
                for label
                in CANONICAL_LABELS
            ]
            for _, row
            in dataframe.iterrows()
        ],
        dtype=np.float64,
    )


# =============================================================================
# CALIBRATION METRICS
# =============================================================================

def multiclass_brier_score(
    y_true: list[str],
    probabilities: np.ndarray,
) -> float:
    """
    Multiclass Brier score:

        mean(
            sum_c(
                (p_c - y_c)^2
            )
        )
    """

    label_index = {
        label:
            index
        for index, label
        in enumerate(
            CANONICAL_LABELS
        )
    }

    targets = np.zeros(
        (
            len(y_true),
            len(
                CANONICAL_LABELS
            ),
        ),
        dtype=np.float64,
    )

    for row_index, label in enumerate(
        y_true
    ):

        targets[
            row_index,
            label_index[
                label
            ],
        ] = 1.0

    return float(
        np.mean(
            np.sum(
                np.square(
                    probabilities
                    - targets
                ),
                axis=1,
            )
        )
    )


def expected_calibration_error(
    y_true: list[str],
    y_pred: list[str],
    probabilities: np.ndarray,
    *,
    bins: int = 10,
) -> float:
    """
    Standard confidence-based Expected Calibration Error.
    """

    if len(y_true) == 0:

        return 0.0

    confidence = (
        np.max(
            probabilities,
            axis=1,
        )
    )

    correct = np.asarray(
        [
            true_label
            == predicted_label
            for true_label, predicted_label
            in zip(
                y_true,
                y_pred,
            )
        ],
        dtype=np.float64,
    )

    edges = np.linspace(
        0.0,
        1.0,
        bins + 1,
    )

    ece = 0.0

    for index in range(
        bins
    ):

        lower = edges[
            index
        ]

        upper = edges[
            index + 1
        ]

        if index == bins - 1:

            mask = (
                (confidence >= lower)
                &
                (confidence <= upper)
            )

        else:

            mask = (
                (confidence >= lower)
                &
                (confidence < upper)
            )

        count = int(
            np.sum(
                mask
            )
        )

        if count == 0:

            continue

        bin_accuracy = float(
            np.mean(
                correct[
                    mask
                ]
            )
        )

        bin_confidence = float(
            np.mean(
                confidence[
                    mask
                ]
            )
        )

        weight = (
            count
            / len(
                y_true
            )
        )

        ece += (
            weight
            * abs(
                bin_accuracy
                - bin_confidence
            )
        )

    return float(
        ece
    )


# =============================================================================
# CLASSIFICATION METRICS
# =============================================================================

def calculate_classification_metrics(
    dataframe: pd.DataFrame,
    *,
    truth_column: str,
    prediction_column: str,
    probability_prefix: str,
) -> tuple[
    dict[str, Any],
    pd.DataFrame,
    pd.DataFrame,
]:
    """
    Calculate complete classification evaluation.
    """

    valid = (
        dataframe[
            truth_column
        ].isin(
            CANONICAL_LABELS
        )
        &
        dataframe[
            prediction_column
        ].isin(
            CANONICAL_LABELS
        )
    )

    evaluation = (
        dataframe.loc[
            valid
        ]
        .copy()
    )

    if evaluation.empty:

        raise ValueError(
            "No valid labelled observations are available "
            "for classification evaluation."
        )

    y_true = (
        evaluation[
            truth_column
        ]
        .tolist()
    )

    y_pred = (
        evaluation[
            prediction_column
        ]
        .tolist()
    )

    probabilities = (
        probability_matrix(
            evaluation,
            prefix=(
                probability_prefix
            ),
        )
    )

    accuracy = float(
        accuracy_score(
            y_true,
            y_pred,
        )
    )

    balanced_accuracy = float(
        balanced_accuracy_score(
            y_true,
            y_pred,
        )
    )

    (
        macro_precision,
        macro_recall,
        macro_f1,
        _,
    ) = (
        precision_recall_fscore_support(
            y_true,
            y_pred,
            labels=(
                CANONICAL_LABELS
            ),
            average="macro",
            zero_division=0,
        )
    )

    (
        _weighted_precision,
        _weighted_recall,
        weighted_f1,
        _,
    ) = (
        precision_recall_fscore_support(
            y_true,
            y_pred,
            labels=(
                CANONICAL_LABELS
            ),
            average="weighted",
            zero_division=0,
        )
    )

    multiclass_log_loss = float(
        log_loss(
            y_true,
            probabilities,
            labels=(
                CANONICAL_LABELS
            ),
        )
    )

    brier = (
        multiclass_brier_score(
            y_true,
            probabilities,
        )
    )

    ece = (
        expected_calibration_error(
            y_true,
            y_pred,
            probabilities,
        )
    )

    report_dictionary = (
        classification_report(
            y_true,
            y_pred,
            labels=(
                CANONICAL_LABELS
            ),
            output_dict=True,
            zero_division=0,
        )
    )

    report_dataframe = (
        pd.DataFrame(
            report_dictionary
        )
        .transpose()
    )

    matrix = (
        confusion_matrix(
            y_true,
            y_pred,
            labels=(
                CANONICAL_LABELS
            ),
        )
    )

    confusion_dataframe = (
        pd.DataFrame(
            matrix,
            index=[
                f"true_{label}"
                for label
                in CANONICAL_LABELS
            ],
            columns=[
                f"pred_{label}"
                for label
                in CANONICAL_LABELS
            ],
        )
    )

    summary = {
        "labelled_observations":
            len(
                evaluation
            ),

        "accuracy":
            accuracy,

        "balanced_accuracy":
            balanced_accuracy,

        "macro_precision":
            float(
                macro_precision
            ),

        "macro_recall":
            float(
                macro_recall
            ),

        "macro_f1":
            float(
                macro_f1
            ),

        "weighted_f1":
            float(
                weighted_f1
            ),

        "multiclass_log_loss":
            multiclass_log_loss,

        "multiclass_brier_score":
            brier,

        "expected_calibration_error":
            ece,
    }

    return (
        summary,
        report_dataframe,
        confusion_dataframe,
    )


# =============================================================================
# SWITCH-RATE / STABILITY EVALUATION
# =============================================================================

def calculate_switch_rates(
    dataframe: pd.DataFrame,
    *,
    group_columns: list[str],
    raw_prediction_column: str,
    temporal_prediction_column: str,
) -> tuple[
    dict[str, Any],
    pd.DataFrame,
]:
    """
    Compare label-switch frequency between raw and temporal predictions.
    """

    sequence_rows: list[
        dict[str, Any]
    ] = []

    total_raw_switches = 0
    total_temporal_switches = 0
    total_possible_transitions = 0

    grouped = (
        dataframe.groupby(
            group_columns,
            dropna=False,
            sort=False,
        )
    )

    for group_key, group in grouped:

        raw_predictions = (
            group[
                raw_prediction_column
            ]
            .tolist()
        )

        temporal_predictions = (
            group[
                temporal_prediction_column
            ]
            .tolist()
        )

        possible = max(
            0,
            len(group)
            - 1,
        )

        raw_switches = sum(
            1
            for index
            in range(
                1,
                len(
                    raw_predictions
                ),
            )
            if (
                raw_predictions[
                    index
                ]
                !=
                raw_predictions[
                    index - 1
                ]
            )
        )

        temporal_switches = sum(
            1
            for index
            in range(
                1,
                len(
                    temporal_predictions
                ),
            )
            if (
                temporal_predictions[
                    index
                ]
                !=
                temporal_predictions[
                    index - 1
                ]
            )
        )

        raw_rate = (
            raw_switches
            / possible
            if possible > 0
            else 0.0
        )

        temporal_rate = (
            temporal_switches
            / possible
            if possible > 0
            else 0.0
        )

        if isinstance(
            group_key,
            tuple,
        ):

            group_values = (
                group_key
            )

        else:

            group_values = (
                group_key,
            )

        row = {
            column:
                value
            for column, value
            in zip(
                group_columns,
                group_values,
            )
        }

        row.update(
            {
                "observations":
                    len(
                        group
                    ),

                "possible_transitions":
                    possible,

                "raw_switches":
                    raw_switches,

                "raw_switch_rate":
                    raw_rate,

                "temporal_switches":
                    temporal_switches,

                "temporal_switch_rate":
                    temporal_rate,

                "switch_rate_difference":
                    (
                        temporal_rate
                        - raw_rate
                    ),

                "switch_rate_reduction":
                    (
                        raw_rate
                        - temporal_rate
                    ),
            }
        )

        sequence_rows.append(
            row
        )

        total_raw_switches += (
            raw_switches
        )

        total_temporal_switches += (
            temporal_switches
        )

        total_possible_transitions += (
            possible
        )

    overall_raw_rate = (
        total_raw_switches
        / total_possible_transitions
        if total_possible_transitions > 0
        else 0.0
    )

    overall_temporal_rate = (
        total_temporal_switches
        / total_possible_transitions
        if total_possible_transitions > 0
        else 0.0
    )

    summary = {
        "sequence_count":
            len(
                sequence_rows
            ),

        "possible_transitions":
            total_possible_transitions,

        "raw_switches":
            total_raw_switches,

        "temporal_switches":
            total_temporal_switches,

        "raw_switch_rate":
            overall_raw_rate,

        "temporal_switch_rate":
            overall_temporal_rate,

        "absolute_switch_rate_reduction":
            (
                overall_raw_rate
                - overall_temporal_rate
            ),

        "relative_switch_rate_reduction":
            (
                (
                    overall_raw_rate
                    - overall_temporal_rate
                )
                / overall_raw_rate
                if overall_raw_rate > 0
                else 0.0
            ),
    }

    return (
        summary,
        pd.DataFrame(
            sequence_rows
        ),
    )


# =============================================================================
# LOGGED TEMPORAL PARITY
# =============================================================================

def evaluate_temporal_parity(
    dataframe: pd.DataFrame,
    logged_temporal_vectors: Optional[
        list[
            dict[str, float]
        ]
    ],
    *,
    tolerance: float,
) -> Optional[
    dict[str, Any]
]:
    """
    Compare logged temporal probabilities with probabilities recomputed
    through TemporalFusionEngine.

    This verifies that historical application output follows the same
    canonical temporal mathematics.
    """

    if logged_temporal_vectors is None:

        return None

    if len(
        logged_temporal_vectors
    ) != len(
        dataframe
    ):

        raise ValueError(
            "Logged temporal probability count does not match "
            "the evaluation dataset."
        )

    maximum_errors: list[
        float
    ] = []

    mean_errors: list[
        float
    ] = []

    class_matches: list[
        bool
    ] = []

    within_tolerance: list[
        bool
    ] = []

    for row_index, logged in enumerate(
        logged_temporal_vectors
    ):

        row = dataframe.iloc[
            row_index
        ]

        computed = {
            label:
                float(
                    row[
                        f"eval_temporal_{label}_prob"
                    ]
                )
            for label in CANONICAL_LABELS
        }

        errors = [
            abs(
                logged[
                    label
                ]
                - computed[
                    label
                ]
            )
            for label in CANONICAL_LABELS
        ]

        maximum_error = max(
            errors
        )

        mean_error = float(
            np.mean(
                errors
            )
        )

        logged_state = (
            summarise_probability_dict(
                logged,
                labels=(
                    CANONICAL_LABELS
                ),
            )[
                "current_state"
            ]
        )

        computed_state = (
            row[
                "eval_temporal_state"
            ]
        )

        maximum_errors.append(
            maximum_error
        )

        mean_errors.append(
            mean_error
        )

        class_matches.append(
            logged_state
            == computed_state
        )

        within_tolerance.append(
            maximum_error
            <= tolerance
        )

    return {
        "observations":
            len(
                dataframe
            ),

        "tolerance":
            tolerance,

        "all_within_tolerance":
            all(
                within_tolerance
            ),

        "within_tolerance_count":
            sum(
                within_tolerance
            ),

        "within_tolerance_rate":
            float(
                np.mean(
                    within_tolerance
                )
            ),

        "argmax_match_count":
            sum(
                class_matches
            ),

        "argmax_match_rate":
            float(
                np.mean(
                    class_matches
                )
            ),

        "maximum_absolute_probability_error":
            max(
                maximum_errors
            )
            if maximum_errors
            else 0.0,

        "mean_absolute_probability_error":
            float(
                np.mean(
                    mean_errors
                )
            )
            if mean_errors
            else 0.0,
    }


# =============================================================================
# LOGGED STATE PARITY
# =============================================================================

def evaluate_state_parity(
    dataframe: pd.DataFrame,
    logged_column: Optional[str],
    computed_column: str,
) -> Optional[
    dict[str, Any]
]:

    if not logged_column:

        return None

    valid = (
        dataframe[
            logged_column
        ]
        .apply(
            normalise_label
        )
    )

    comparisons: list[
        bool
    ] = []

    compared = 0

    for logged, computed in zip(
        valid,
        dataframe[
            computed_column
        ],
    ):

        if logged is None:

            continue

        compared += 1

        comparisons.append(
            logged
            == computed
        )

    if compared == 0:

        return {
            "column":
                logged_column,

            "compared":
                0,

            "matches":
                0,

            "match_rate":
                None,
        }

    return {
        "column":
            logged_column,

        "compared":
            compared,

        "matches":
            sum(
                comparisons
            ),

        "match_rate":
            float(
                np.mean(
                    comparisons
                )
            ),
    }


# =============================================================================
# DESCRIPTIVE PREDICTION STATISTICS
# =============================================================================

def descriptive_prediction_statistics(
    dataframe: pd.DataFrame,
) -> dict[str, Any]:

    def state_counts(
        column: str,
    ) -> dict[str, int]:

        counts = (
            dataframe[
                column
            ]
            .value_counts()
            .to_dict()
        )

        return {
            label:
                int(
                    counts.get(
                        label,
                        0,
                    )
                )
            for label
            in CANONICAL_LABELS
        }

    return {
        "raw_state_counts":
            state_counts(
                "eval_raw_state"
            ),

        "temporal_state_counts":
            state_counts(
                "eval_temporal_state"
            ),

        "raw_confidence_mean":
            float(
                dataframe[
                    "eval_raw_confidence"
                ]
                .mean()
            ),

        "raw_confidence_std":
            float(
                dataframe[
                    "eval_raw_confidence"
                ]
                .std(
                    ddof=0
                )
            ),

        "temporal_confidence_mean":
            float(
                dataframe[
                    "eval_temporal_confidence"
                ]
                .mean()
            ),

        "temporal_confidence_std":
            float(
                dataframe[
                    "eval_temporal_confidence"
                ]
                .std(
                    ddof=0
                )
            ),

        "raw_confidence_gap_mean":
            float(
                dataframe[
                    "eval_raw_confidence_gap"
                ]
                .mean()
            ),

        "temporal_confidence_gap_mean":
            float(
                dataframe[
                    "eval_temporal_confidence_gap"
                ]
                .mean()
            ),

        "full_window_observations":
            int(
                dataframe[
                    "eval_temporal_window_full"
                ]
                .sum()
            ),
    }


# =============================================================================
# RUNTIME STATISTICS
# =============================================================================

def calculate_runtime_statistics(
    dataframe: pd.DataFrame,
    runtime_column: Optional[str],
) -> Optional[
    dict[str, Any]
]:

    if not runtime_column:

        return None

    values = (
        pd.to_numeric(
            dataframe[
                runtime_column
            ],
            errors="coerce",
        )
        .dropna()
    )

    values = values[
        values >= 0
    ]

    if values.empty:

        return None

    return {
        "column":
            runtime_column,

        "count":
            int(
                len(
                    values
                )
            ),

        "mean_seconds":
            float(
                values.mean()
            ),

        "median_seconds":
            float(
                values.median()
            ),

        "std_seconds":
            float(
                values.std(
                    ddof=0
                )
            ),

        "min_seconds":
            float(
                values.min()
            ),

        "max_seconds":
            float(
                values.max()
            ),

        "p95_seconds":
            float(
                values.quantile(
                    0.95
                )
            ),
    }


# =============================================================================
# METRIC OUTPUT
# =============================================================================

def save_classification_outputs(
    *,
    output_directory: Path,
    name: str,
    summary: dict[str, Any],
    report: pd.DataFrame,
    confusion: pd.DataFrame,
) -> None:

    write_json(
        output_directory
        / f"{name}_metrics.json",
        summary,
    )

    report.to_csv(
        output_directory
        / f"{name}_classification_report.csv",
        index=True,
    )

    confusion.to_csv(
        output_directory
        / f"{name}_confusion_matrix.csv",
        index=True,
    )


# =============================================================================
# CONSOLE REPORT
# =============================================================================

def print_metric_block(
    title: str,
    metrics: dict[str, Any],
) -> None:

    print()

    print(
        "=" * 80
    )

    print(
        title
    )

    print(
        "=" * 80
    )

    preferred_keys = [
        "labelled_observations",
        "accuracy",
        "balanced_accuracy",
        "macro_precision",
        "macro_recall",
        "macro_f1",
        "weighted_f1",
        "multiclass_log_loss",
        "multiclass_brier_score",
        "expected_calibration_error",
    ]

    for key in preferred_keys:

        if key not in metrics:

            continue

        value = (
            metrics[
                key
            ]
        )

        if isinstance(
            value,
            float,
        ):

            print(
                f"{key:35s}: "
                f"{value:.6f}"
            )

        else:

            print(
                f"{key:35s}: "
                f"{value}"
            )


# =============================================================================
# MAIN EVALUATION
# =============================================================================

def evaluate(
    *,
    input_path: Path,
    output_directory: Path,
    truth_column: Optional[str],
    session_column: Optional[str],
    trial_column: Optional[str],
    generation_column: Optional[str],
    order_column: Optional[str],
    raw_probabilities_column: Optional[str],
    logged_temporal_probabilities_column: Optional[str],
    logged_raw_state_column: Optional[str],
    logged_temporal_state_column: Optional[str],
    runtime_column: Optional[str],
    window_size: int,
    parity_tolerance: float,
    require_ground_truth: bool,
) -> dict[str, Any]:

    if window_size <= 0:

        raise ValueError(
            "Temporal window size must be greater than zero."
        )

    if parity_tolerance < 0:

        raise ValueError(
            "Parity tolerance cannot be negative."
        )

    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    # -------------------------------------------------------------------------
    # Load source data
    # -------------------------------------------------------------------------

    dataframe = (
        load_results_file(
            input_path
        )
    )

    # -------------------------------------------------------------------------
    # Ground truth
    # -------------------------------------------------------------------------

    resolved_truth_column = (
        resolve_column(
            dataframe,
            truth_column,
            GROUND_TRUTH_CANDIDATES,
            required=False,
            description=(
                "ground-truth column"
            ),
        )
    )

    if (
        require_ground_truth
        and
        resolved_truth_column
        is None
    ):

        raise ValueError(
            "Ground truth is required but no ground-truth "
            "column could be identified."
        )

    if resolved_truth_column:

        dataframe[
            "__ground_truth"
        ] = (
            dataframe[
                resolved_truth_column
            ]
            .apply(
                normalise_label
            )
        )

    else:

        dataframe[
            "__ground_truth"
        ] = None

    # -------------------------------------------------------------------------
    # Raw probability input
    # -------------------------------------------------------------------------

    (
        raw_vectors,
        raw_probability_source,
    ) = (
        extract_probability_vectors(
            dataframe,
            explicit_column=(
                raw_probabilities_column
            ),
            mapping_candidates=(
                RAW_PROBABILITY_COLUMN_CANDIDATES
            ),
            prefix_type="raw",
            required=True,
        )
    )

    assert (
        raw_vectors
        is not None
    )

    # -------------------------------------------------------------------------
    # Optional logged temporal probabilities
    # -------------------------------------------------------------------------

    (
        logged_temporal_vectors,
        logged_temporal_probability_source,
    ) = (
        extract_probability_vectors(
            dataframe,
            explicit_column=(
                logged_temporal_probabilities_column
            ),
            mapping_candidates=(
                LOGGED_TEMPORAL_PROBABILITY_COLUMN_CANDIDATES
            ),
            prefix_type=(
                "temporal"
            ),
            required=False,
        )
    )

    # -------------------------------------------------------------------------
    # Sequence grouping
    # -------------------------------------------------------------------------

    (
        group_columns,
        sequence_columns,
    ) = (
        determine_group_columns(
            dataframe,
            session_column=(
                session_column
            ),
            trial_column=(
                trial_column
            ),
            generation_column=(
                generation_column
            ),
        )
    )

    resolved_order_column = (
        determine_order_column(
            dataframe,
            order_column,
        )
    )

    # -------------------------------------------------------------------------
    # Reconstruct canonical temporal fusion
    # -------------------------------------------------------------------------

    evaluated = (
        reconstruct_temporal_predictions(
            dataframe,
            raw_vectors,
            group_columns=(
                group_columns
            ),
            order_column=(
                resolved_order_column
            ),
            window_size=(
                window_size
            ),
        )
    )

    # -------------------------------------------------------------------------
    # Resolve logged-state columns
    # -------------------------------------------------------------------------

    resolved_logged_raw_state = (
        resolve_column(
            evaluated,
            logged_raw_state_column,
            RAW_STATE_COLUMN_CANDIDATES,
            description=(
                "logged raw-state column"
            ),
        )
    )

    resolved_logged_temporal_state = (
        resolve_column(
            evaluated,
            logged_temporal_state_column,
            TEMPORAL_STATE_COLUMN_CANDIDATES,
            description=(
                "logged temporal-state column"
            ),
        )
    )

    resolved_runtime_column = (
        resolve_column(
            evaluated,
            runtime_column,
            RUNTIME_COLUMN_CANDIDATES,
            description=(
                "runtime column"
            ),
        )
    )

    # -------------------------------------------------------------------------
    # Save row-level canonical reconstruction
    # -------------------------------------------------------------------------

    drop_internal = [
        "__raw_probability_object",
    ]

    row_output = (
        evaluated.drop(
            columns=[
                column
                for column in drop_internal
                if column
                in evaluated.columns
            ]
        )
    )

    row_output.to_csv(
        output_directory
        / "row_level_evaluation.csv",
        index=False,
    )

    # -------------------------------------------------------------------------
    # Stability / switching
    # -------------------------------------------------------------------------

    (
        stability_summary,
        sequence_switches,
    ) = (
        calculate_switch_rates(
            evaluated,
            group_columns=(
                group_columns
            ),
            raw_prediction_column=(
                "eval_raw_state"
            ),
            temporal_prediction_column=(
                "eval_temporal_state"
            ),
        )
    )

    sequence_switches.to_csv(
        output_directory
        / "sequence_switch_rates.csv",
        index=False,
    )

    # -------------------------------------------------------------------------
    # Temporal parity
    #
    # logged vectors were created before dataframe sorting. The evaluated
    # output has been restored to original source-row order, so alignment
    # remains correct.
    # -------------------------------------------------------------------------

    temporal_parity = (
        evaluate_temporal_parity(
            evaluated,
            logged_temporal_vectors,
            tolerance=(
                parity_tolerance
            ),
        )
    )

    raw_state_parity = (
        evaluate_state_parity(
            evaluated,
            resolved_logged_raw_state,
            "eval_raw_state",
        )
    )

    temporal_state_parity = (
        evaluate_state_parity(
            evaluated,
            resolved_logged_temporal_state,
            "eval_temporal_state",
        )
    )

    # -------------------------------------------------------------------------
    # Descriptive prediction statistics
    # -------------------------------------------------------------------------

    descriptive = (
        descriptive_prediction_statistics(
            evaluated
        )
    )

    runtime_statistics = (
        calculate_runtime_statistics(
            evaluated,
            resolved_runtime_column,
        )
    )

    # -------------------------------------------------------------------------
    # Labelled classification evaluation
    # -------------------------------------------------------------------------

    raw_metrics = None
    temporal_metrics = None
    full_window_metrics = None

    valid_ground_truth_count = int(
        evaluated[
            "__ground_truth"
        ]
        .isin(
            CANONICAL_LABELS
        )
        .sum()
    )

    if valid_ground_truth_count > 0:

        (
            raw_metrics,
            raw_report,
            raw_confusion,
        ) = (
            calculate_classification_metrics(
                evaluated,
                truth_column=(
                    "__ground_truth"
                ),
                prediction_column=(
                    "eval_raw_state"
                ),
                probability_prefix=(
                    "eval_raw"
                ),
            )
        )

        save_classification_outputs(
            output_directory=(
                output_directory
            ),
            name=(
                "raw_fusion"
            ),
            summary=(
                raw_metrics
            ),
            report=(
                raw_report
            ),
            confusion=(
                raw_confusion
            ),
        )

        (
            temporal_metrics,
            temporal_report,
            temporal_confusion,
        ) = (
            calculate_classification_metrics(
                evaluated,
                truth_column=(
                    "__ground_truth"
                ),
                prediction_column=(
                    "eval_temporal_state"
                ),
                probability_prefix=(
                    "eval_temporal"
                ),
            )
        )

        save_classification_outputs(
            output_directory=(
                output_directory
            ),
            name=(
                "temporal_fusion_all_samples"
            ),
            summary=(
                temporal_metrics
            ),
            report=(
                temporal_report
            ),
            confusion=(
                temporal_confusion
            ),
        )

        full_window = (
            evaluated[
                evaluated[
                    "eval_temporal_window_full"
                ]
                == True  # noqa: E712
            ]
            .copy()
        )

        full_window_labelled = (
            full_window[
                "__ground_truth"
            ]
            .isin(
                CANONICAL_LABELS
            )
            .sum()
        )

        if full_window_labelled > 0:

            (
                full_window_metrics,
                full_window_report,
                full_window_confusion,
            ) = (
                calculate_classification_metrics(
                    full_window,
                    truth_column=(
                        "__ground_truth"
                    ),
                    prediction_column=(
                        "eval_temporal_state"
                    ),
                    probability_prefix=(
                        "eval_temporal"
                    ),
                )
            )

            save_classification_outputs(
                output_directory=(
                    output_directory
                ),
                name=(
                    "temporal_fusion_full_window"
                ),
                summary=(
                    full_window_metrics
                ),
                report=(
                    full_window_report
                ),
                confusion=(
                    full_window_confusion
                ),
            )

    elif require_ground_truth:

        raise ValueError(
            "Ground-truth column was found, but it contains no "
            "valid canonical behavioural labels."
        )

    # -------------------------------------------------------------------------
    # Raw vs temporal comparison
    # -------------------------------------------------------------------------

    raw_vs_temporal: dict[
        str,
        Any
    ] = {
        "behavioural_accuracy_available":
            (
                raw_metrics
                is not None
                and
                temporal_metrics
                is not None
            ),

        "raw_switch_rate":
            stability_summary[
                "raw_switch_rate"
            ],

        "temporal_switch_rate":
            stability_summary[
                "temporal_switch_rate"
            ],

        "absolute_switch_rate_reduction":
            stability_summary[
                "absolute_switch_rate_reduction"
            ],

        "relative_switch_rate_reduction":
            stability_summary[
                "relative_switch_rate_reduction"
            ],
    }

    if (
        raw_metrics
        is not None
        and
        temporal_metrics
        is not None
    ):

        raw_vs_temporal.update(
            {
                "raw_accuracy":
                    raw_metrics[
                        "accuracy"
                    ],

                "temporal_accuracy":
                    temporal_metrics[
                        "accuracy"
                    ],

                "accuracy_difference":
                    (
                        temporal_metrics[
                            "accuracy"
                        ]
                        -
                        raw_metrics[
                            "accuracy"
                        ]
                    ),

                "raw_macro_f1":
                    raw_metrics[
                        "macro_f1"
                    ],

                "temporal_macro_f1":
                    temporal_metrics[
                        "macro_f1"
                    ],

                "macro_f1_difference":
                    (
                        temporal_metrics[
                            "macro_f1"
                        ]
                        -
                        raw_metrics[
                            "macro_f1"
                        ]
                    ),

                "raw_log_loss":
                    raw_metrics[
                        "multiclass_log_loss"
                    ],

                "temporal_log_loss":
                    temporal_metrics[
                        "multiclass_log_loss"
                    ],

                "log_loss_difference":
                    (
                        temporal_metrics[
                            "multiclass_log_loss"
                        ]
                        -
                        raw_metrics[
                            "multiclass_log_loss"
                        ]
                    ),

                "raw_brier_score":
                    raw_metrics[
                        "multiclass_brier_score"
                    ],

                "temporal_brier_score":
                    temporal_metrics[
                        "multiclass_brier_score"
                    ],
            }
        )

    # -------------------------------------------------------------------------
    # Overall report
    # -------------------------------------------------------------------------

    summary: dict[
        str,
        Any
    ] = {
        "evaluation_name":
            (
                "SenseFuzeAI Multimodal "
                "Raw-vs-Temporal Evaluation"
            ),

        "input_file":
            str(
                input_path
            ),

        "output_directory":
            str(
                output_directory
            ),

        "canonical_labels":
            list(
                CANONICAL_LABELS
            ),

        "temporal_backend":
            (
                "temporal_fusion."
                "TemporalFusionEngine"
            ),

        "temporal_window":
            window_size,

        "total_observations":
            int(
                len(
                    evaluated
                )
            ),

        "valid_ground_truth_observations":
            valid_ground_truth_count,

        "behavioural_accuracy_status":
            (
                "evaluated"
                if valid_ground_truth_count > 0
                else "not_established"
            ),

        "behavioural_accuracy_note":
            (
                (
                    "Accuracy metrics were calculated against "
                    "available ground-truth labels."
                )
                if valid_ground_truth_count > 0
                else
                (
                    "No valid ground-truth labels were available. "
                    "Pipeline operation and confidence do not establish "
                    "behavioural classification accuracy."
                )
            ),

        "input_schema": {
            "ground_truth_column":
                resolved_truth_column,

            "raw_probability_source":
                raw_probability_source,

            "logged_temporal_probability_source":
                (
                    logged_temporal_probability_source
                ),

            "group_columns":
                group_columns,

            "order_column":
                resolved_order_column,

            "session_column":
                sequence_columns[
                    "session_column"
                ],

            "trial_column":
                sequence_columns[
                    "trial_column"
                ],

            "generation_column":
                sequence_columns[
                    "generation_column"
                ],

            "logged_raw_state_column":
                resolved_logged_raw_state,

            "logged_temporal_state_column":
                resolved_logged_temporal_state,

            "runtime_column":
                resolved_runtime_column,
        },

        "raw_fusion":
            raw_metrics,

        "temporal_fusion_all_samples":
            temporal_metrics,

        "temporal_fusion_full_window":
            full_window_metrics,

        "raw_vs_temporal":
            raw_vs_temporal,

        "stability":
            stability_summary,

        "descriptive_statistics":
            descriptive,

        "runtime_statistics":
            runtime_statistics,

        "temporal_probability_parity":
            temporal_parity,

        "logged_raw_state_parity":
            raw_state_parity,

        "logged_temporal_state_parity":
            temporal_state_parity,
    }

    write_json(
        output_directory
        / "evaluation_summary.json",
        summary,
    )

    write_json(
        output_directory
        / "stability_metrics.json",
        stability_summary,
    )

    if temporal_parity is not None:

        write_json(
            output_directory
            / "temporal_parity.json",
            temporal_parity,
        )

    # -------------------------------------------------------------------------
    # Console output
    # -------------------------------------------------------------------------

    print()

    print(
        "=" * 80
    )

    print(
        "SenseFuzeAI Multimodal Evaluation"
    )

    print(
        "=" * 80
    )

    print(
        f"Input observations              : "
        f"{len(evaluated)}"
    )

    print(
        f"Canonical temporal window       : "
        f"{window_size}"
    )

    print(
        f"Sequence grouping               : "
        f"{group_columns}"
    )

    print(
        f"Sequence order                  : "
        f"{resolved_order_column or '__source_row'}"
    )

    print(
        f"Ground-truth column             : "
        f"{resolved_truth_column or 'NOT AVAILABLE'}"
    )

    print(
        f"Valid labelled observations     : "
        f"{valid_ground_truth_count}"
    )

    if raw_metrics is not None:

        print_metric_block(
            "RAW MULTIMODAL FUSION",
            raw_metrics,
        )

    if temporal_metrics is not None:

        print_metric_block(
            "TEMPORAL FUSION — ALL OBSERVATIONS",
            temporal_metrics,
        )

    if full_window_metrics is not None:

        print_metric_block(
            (
                "TEMPORAL FUSION — "
                "FULL WINDOW ONLY"
            ),
            full_window_metrics,
        )

    print()

    print(
        "=" * 80
    )

    print(
        "TEMPORAL STABILITY"
    )

    print(
        "=" * 80
    )

    print(
        f"Raw switch rate                 : "
        f"{stability_summary['raw_switch_rate']:.6f}"
    )

    print(
        f"Temporal switch rate            : "
        f"{stability_summary['temporal_switch_rate']:.6f}"
    )

    print(
        f"Absolute reduction              : "
        f"{stability_summary['absolute_switch_rate_reduction']:.6f}"
    )

    print(
        f"Relative reduction              : "
        f"{stability_summary['relative_switch_rate_reduction']:.2%}"
    )

    if temporal_parity is not None:

        print()

        print(
            "=" * 80
        )

        print(
            "TEMPORAL IMPLEMENTATION PARITY"
        )

        print(
            "=" * 80
        )

        print(
            f"All probabilities within tol.   : "
            f"{temporal_parity['all_within_tolerance']}"
        )

        print(
            f"Argmax match rate               : "
            f"{temporal_parity['argmax_match_rate']:.6f}"
        )

        print(
            f"Maximum probability error       : "
            f"{temporal_parity['maximum_absolute_probability_error']:.12f}"
        )

    if valid_ground_truth_count == 0:

        print()

        print(
            "NOTE:"
        )

        print(
            "No ground-truth behavioural labels were available."
        )

        print(
            "Therefore this run validates temporal processing and "
            "software behaviour, but DOES NOT establish classifier accuracy."
        )

    print()

    print(
        f"Evaluation outputs written to:\n"
        f"{output_directory}"
    )

    return summary


# =============================================================================
# COMMAND-LINE INTERFACE
# =============================================================================

def build_argument_parser() -> argparse.ArgumentParser:

    parser = argparse.ArgumentParser(
        description=(
            "Evaluate SenseFuzeAI raw multimodal predictions "
            "against canonical TemporalFusionEngine output."
        )
    )

    parser.add_argument(
        "--input",
        required=True,
        type=Path,
        help=(
            "CSV/JSON/JSONL/Parquet file containing "
            "raw multimodal probability results."
        ),
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=(
            Path(
                "evaluation_output"
            )
            / "multimodal_temporal"
        ),
        help=(
            "Directory for evaluation reports."
        ),
    )

    parser.add_argument(
        "--truth-column",
        default=None,
        help=(
            "Ground-truth behavioural-state column. "
            "Automatically discovered when omitted."
        ),
    )

    parser.add_argument(
        "--session-column",
        default=None,
        help=(
            "Session identifier column."
        ),
    )

    parser.add_argument(
        "--trial-column",
        default=None,
        help=(
            "Trial/condition identifier column."
        ),
    )

    parser.add_argument(
        "--generation-column",
        default=None,
        help=(
            "Temporal reset-generation column."
        ),
    )

    parser.add_argument(
        "--order-column",
        default=None,
        help=(
            "Chronological sequence-order column."
        ),
    )

    parser.add_argument(
        "--raw-probabilities-column",
        default=None,
        help=(
            "Column containing raw four-class probability dictionaries."
        ),
    )

    parser.add_argument(
        "--logged-temporal-probabilities-column",
        default=None,
        help=(
            "Optional column containing previously logged temporal "
            "probability dictionaries for parity checking."
        ),
    )

    parser.add_argument(
        "--logged-raw-state-column",
        default=None,
        help=(
            "Optional stored raw-state column for parity verification."
        ),
    )

    parser.add_argument(
        "--logged-temporal-state-column",
        default=None,
        help=(
            "Optional stored temporal-state column for parity verification."
        ),
    )

    parser.add_argument(
        "--runtime-column",
        default=None,
        help=(
            "Optional prediction runtime column."
        ),
    )

    parser.add_argument(
        "--window-size",
        type=int,
        default=(
            TEMPORAL_PROBABILITY_WINDOW
        ),
        help=(
            "Temporal probability window. "
            "Defaults to the canonical temporal_fusion.py value."
        ),
    )

    parser.add_argument(
        "--parity-tolerance",
        type=float,
        default=1e-9,
        help=(
            "Maximum absolute probability difference considered "
            "equivalent during temporal parity checks."
        ),
    )

    parser.add_argument(
        "--require-ground-truth",
        action="store_true",
        help=(
            "Fail if no valid ground-truth labels are available."
        ),
    )

    return parser


def main() -> None:

    parser = (
        build_argument_parser()
    )

    args = (
        parser.parse_args()
    )

    try:

        evaluate(
            input_path=(
                args.input
            ),

            output_directory=(
                args.output_dir
            ),

            truth_column=(
                args.truth_column
            ),

            session_column=(
                args.session_column
            ),

            trial_column=(
                args.trial_column
            ),

            generation_column=(
                args.generation_column
            ),

            order_column=(
                args.order_column
            ),

            raw_probabilities_column=(
                args.raw_probabilities_column
            ),

            logged_temporal_probabilities_column=(
                args.logged_temporal_probabilities_column
            ),

            logged_raw_state_column=(
                args.logged_raw_state_column
            ),

            logged_temporal_state_column=(
                args.logged_temporal_state_column
            ),

            runtime_column=(
                args.runtime_column
            ),

            window_size=(
                args.window_size
            ),

            parity_tolerance=(
                args.parity_tolerance
            ),

            require_ground_truth=(
                args.require_ground_truth
            ),
        )

    except Exception as exc:

        print(
            (
                "\nEvaluation failed:\n"
                f"{type(exc).__name__}: "
                f"{exc}"
            ),
            file=sys.stderr,
        )

        raise SystemExit(
            1
        ) from exc


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":

    main()
