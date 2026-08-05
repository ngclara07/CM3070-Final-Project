"""
temporal_fusion.py

Canonical temporal-probability fusion implementation for SenseFuzeAI.

Purpose
-------
This module is the single source of truth for temporal fusion behaviour
used by:

    - live_fusion_gui.py
    - web_app/app.py
    - offline evaluation scripts
    - automated tests

The canonical processing pipeline is:

    raw multimodal probability vector
                |
                v
    probability sanitisation / normalisation
                |
                v
    rolling temporal history
                |
                v
    arithmetic mean probability per class
                |
                v
    final normalisation
                |
                v
    temporal argmax prediction
                |
                v
    confidence + confidence gap + confidence level

Canonical behavioural classes
-----------------------------
    focused
    distracted
    fatigued
    overloaded

Canonical temporal window
-------------------------
    5 multimodal observations

Canonical confidence metric
---------------------------
    confidence = highest temporal class probability

Canonical confidence gap
------------------------
    gap = highest probability - second-highest probability

Canonical confidence levels
---------------------------
    High   : gap >= 0.35
    Medium : gap >= 0.15 and gap < 0.35
    Low    : gap < 0.15

Probability policy
------------------
    - missing classes -> 0.0
    - NaN -> 0.0
    - +/- infinity -> 0.0
    - negative values -> clamped to 0.0
    - positive values -> retained
    - distribution -> normalised to sum to 1.0
    - zero-total distribution -> uniform distribution

Reset policy
------------
A temporal reset:

    1. increments the generation number;
    2. clears all temporal probability observations;
    3. causes predictions started under an older generation to become stale.

This allows desktop and web applications to safely reject a prediction
that began before a reset/source change but completed afterwards.

Important architectural boundary
--------------------------------
This module DOES NOT extract text/audio/image/keystroke features and
DOES NOT invoke FinalMultimodalInference.

FinalMultimodalInference should remain responsible for producing ONE
raw multimodal probability vector.

This module operates AFTER that raw multimodal inference.
"""

from __future__ import annotations

import math
import threading

from collections import deque
from collections.abc import Iterable, Mapping
from typing import Any, Final


# ============================================================
# Canonical behavioural classes
# ============================================================

LABELS: Final[tuple[str, ...]] = (
    "focused",
    "distracted",
    "fatigued",
    "overloaded",
)


# ============================================================
# Canonical temporal configuration
# ============================================================

TEMPORAL_PROBABILITY_WINDOW: Final[int] = 5


# ============================================================
# Canonical confidence thresholds
# ============================================================

CONFIDENCE_HIGH_GAP: Final[float] = 0.35
CONFIDENCE_MEDIUM_GAP: Final[float] = 0.15


# ============================================================
# Numerical validation tolerance
# ============================================================

PROBABILITY_SUM_TOLERANCE: Final[float] = 1e-6


# ============================================================
# Public exports
# ============================================================

__all__ = [
    "LABELS",
    "TEMPORAL_PROBABILITY_WINDOW",
    "CONFIDENCE_HIGH_GAP",
    "CONFIDENCE_MEDIUM_GAP",
    "PROBABILITY_SUM_TOLERANCE",
    "TemporalFusionError",
    "EmptyTemporalHistoryError",
    "StaleGenerationError",
    "confidence_level",
    "normalise_probability_dict",
    "normalize_probability_dict",
    "rank_probabilities",
    "summarise_probability_dict",
    "aggregate_probability_history",
    "validate_probability_distribution",
    "TemporalFusionEngine",
]


# ============================================================
# Exceptions
# ============================================================

class TemporalFusionError(Exception):
    """
    Base exception for canonical temporal-fusion errors.
    """


class EmptyTemporalHistoryError(TemporalFusionError):
    """
    Raised when temporal aggregation is requested without
    any probability observations.
    """


class StaleGenerationError(TemporalFusionError):
    """
    Raised when a result belongs to a prediction that started
    before the most recent temporal/full reset.
    """


# ============================================================
# Internal helpers
# ============================================================

def _validate_window_size(
    window_size: int,
) -> int:
    """
    Validate and return a positive temporal-window size.
    """

    if isinstance(window_size, bool):

        raise ValueError(
            "window_size must be a positive integer."
        )

    try:

        value = int(
            window_size
        )

    except (
        TypeError,
        ValueError,
    ) as exc:

        raise ValueError(
            "window_size must be a positive integer."
        ) from exc

    if value <= 0:

        raise ValueError(
            "window_size must be greater than zero."
        )

    return value


def _validate_labels(
    labels: Iterable[str],
) -> tuple[str, ...]:
    """
    Validate a behavioural-class sequence.

    Normally callers should use the canonical LABELS constant.
    """

    result = tuple(
        str(label).strip()
        for label in labels
    )

    if not result:

        raise ValueError(
            "At least one behavioural class is required."
        )

    if any(
        not label
        for label in result
    ):

        raise ValueError(
            "Behavioural class names cannot be empty."
        )

    if len(
        set(result)
    ) != len(result):

        raise ValueError(
            "Behavioural class names must be unique."
        )

    if len(result) < 2:

        raise ValueError(
            "Temporal fusion requires at least two classes."
        )

    return result


def _safe_probability_value(
    value: Any,
) -> float:
    """
    Convert an arbitrary probability-like value into a safe
    non-negative finite float.

    Policy:
        invalid       -> 0.0
        NaN           -> 0.0
        infinity      -> 0.0
        negative      -> 0.0
        finite >= 0   -> unchanged
    """

    try:

        probability = float(
            value
        )

    except (
        TypeError,
        ValueError,
        OverflowError,
    ):

        return 0.0

    if not math.isfinite(
        probability
    ):

        return 0.0

    return max(
        0.0,
        probability,
    )


# ============================================================
# Confidence calculation
# ============================================================

def confidence_level(
    gap: float,
) -> str:
    """
    Convert the top-vs-second probability gap into the
    canonical SenseFuzeAI confidence level.

    Parameters
    ----------
    gap:
        Difference between the highest and second-highest
        temporal probabilities.

    Returns
    -------
    str
        "High", "Medium", or "Low".
    """

    try:

        value = float(
            gap
        )

    except (
        TypeError,
        ValueError,
        OverflowError,
    ):

        value = 0.0

    if not math.isfinite(
        value
    ):

        value = 0.0

    value = max(
        0.0,
        value,
    )

    if (
        value
        >= CONFIDENCE_HIGH_GAP
    ):

        return "High"

    if (
        value
        >= CONFIDENCE_MEDIUM_GAP
    ):

        return "Medium"

    return "Low"


# ============================================================
# Probability normalisation
# ============================================================

def normalise_probability_dict(
    probabilities: Mapping[str, Any],
    *,
    labels: Iterable[str] = LABELS,
) -> dict[str, float]:
    """
    Convert an arbitrary probability mapping into a canonical
    probability distribution.

    Exactly the configured behavioural classes are returned,
    in the configured order.

    Extra dictionary keys are deliberately ignored.

    Missing class
        -> 0.0

    NaN / infinity / invalid
        -> 0.0

    Negative value
        -> 0.0

    Zero total
        -> uniform distribution

    Otherwise
        -> divide every class value by the total

    Parameters
    ----------
    probabilities:
        Mapping containing model probability-like values.

    labels:
        Behavioural class ordering. Defaults to the canonical
        SenseFuzeAI LABELS.

    Returns
    -------
    dict[str, float]
        Normalised probability distribution whose values sum
        to approximately 1.0.
    """

    canonical_labels = (
        _validate_labels(
            labels
        )
    )

    if not isinstance(
        probabilities,
        Mapping,
    ):

        raise TypeError(
            "probabilities must be a mapping "
            "from class names to values."
        )

    cleaned: dict[
        str,
        float
    ] = {}

    for label in canonical_labels:

        cleaned[label] = (
            _safe_probability_value(
                probabilities.get(
                    label,
                    0.0,
                )
            )
        )

    total = math.fsum(
        cleaned.values()
    )

    if (
        not math.isfinite(total)
        or
        total <= 0.0
    ):

        uniform = (
            1.0
            / len(
                canonical_labels
            )
        )

        return {
            label: uniform
            for label
            in canonical_labels
        }

    normalised = {
        label:
            cleaned[label]
            / total
        for label
        in canonical_labels
    }

    # A final mathematical normalisation protects against
    # accumulated floating-point rounding.
    final_total = math.fsum(
        normalised.values()
    )

    if (
        not math.isfinite(
            final_total
        )
        or
        final_total <= 0.0
    ):

        uniform = (
            1.0
            / len(
                canonical_labels
            )
        )

        return {
            label: uniform
            for label
            in canonical_labels
        }

    return {
        label:
            normalised[label]
            / final_total
        for label
        in canonical_labels
    }


# American-English alias.
#
# The British spelling remains canonical because existing
# SenseFuzeAI code uses normalise_probability_dict().
normalize_probability_dict = (
    normalise_probability_dict
)


# ============================================================
# Probability ranking
# ============================================================

def rank_probabilities(
    probabilities: Mapping[str, Any],
    *,
    labels: Iterable[str] = LABELS,
) -> list[
    tuple[str, float]
]:
    """
    Normalise and rank a probability distribution.

    Ties are resolved deterministically using the canonical
    label order.

    This reproduces the stable ordering behaviour expected by
    the existing desktop/web implementations.
    """

    canonical_labels = (
        _validate_labels(
            labels
        )
    )

    normalised = (
        normalise_probability_dict(
            probabilities,
            labels=canonical_labels,
        )
    )

    label_order = {
        label: index
        for index, label
        in enumerate(
            canonical_labels
        )
    }

    ranked = sorted(
        normalised.items(),
        key=lambda item: (
            -item[1],
            label_order[
                item[0]
            ],
        ),
    )

    return ranked


# ============================================================
# Single probability-vector summary
# ============================================================

def summarise_probability_dict(
    probabilities: Mapping[str, Any],
    *,
    labels: Iterable[str] = LABELS,
) -> dict[str, Any]:
    """
    Summarise one probability vector without temporal history.

    This is useful for displaying the current RAW fusion result.

    Returns
    -------
    dict
        {
            "current_state": ...,
            "confidence": ...,
            "confidence_percent": ...,
            "probabilities": ...,
            "second_class": ...,
            "second_probability": ...,
            "confidence_gap": ...,
            "confidence_level": ...
        }
    """

    canonical_labels = (
        _validate_labels(
            labels
        )
    )

    normalised = (
        normalise_probability_dict(
            probabilities,
            labels=canonical_labels,
        )
    )

    ranked = (
        rank_probabilities(
            normalised,
            labels=canonical_labels,
        )
    )

    top_class = (
        ranked[0][0]
    )

    top_probability = float(
        ranked[0][1]
    )

    second_class = (
        ranked[1][0]
    )

    second_probability = float(
        ranked[1][1]
    )

    gap = max(
        0.0,
        top_probability
        - second_probability,
    )

    return {
        "current_state":
            top_class,

        "confidence":
            top_probability,

        "confidence_percent":
            top_probability
            * 100.0,

        "probabilities":
            normalised,

        "second_class":
            second_class,

        "second_probability":
            second_probability,

        "confidence_gap":
            gap,

        "confidence_level":
            confidence_level(
                gap
            ),
    }


# ============================================================
# Temporal aggregation
# ============================================================

def aggregate_probability_history(
    history: Iterable[
        Mapping[str, Any]
    ],
    *,
    window_size: int = (
        TEMPORAL_PROBABILITY_WINDOW
    ),
    labels: Iterable[str] = LABELS,
) -> dict[str, Any]:
    """
    Apply the canonical SenseFuzeAI temporal-fusion algorithm.

    Algorithm
    ---------
    1. Convert history to chronological observations.
    2. Retain only the latest ``window_size`` observations.
    3. Normalise every individual probability vector.
    4. Compute the arithmetic mean probability for each class.
    5. Normalise the averaged probability vector.
    6. Select the class with the highest temporal probability.
    7. Confidence = highest temporal probability.
    8. Gap = top probability - second-highest probability.
    9. Map the gap to Low / Medium / High.

    No weighting or exponential smoothing is used.

    Every observation contributes equally:

        mean(P_c) = sum(P_i,c) / N

    Parameters
    ----------
    history:
        Chronologically ordered probability observations.

    window_size:
        Maximum number of latest observations included.

    labels:
        Class ordering.

    Returns
    -------
    dict
        Canonical temporal-fusion result.

    Raises
    ------
    EmptyTemporalHistoryError
        If history contains no observations.
    """

    canonical_labels = (
        _validate_labels(
            labels
        )
    )

    canonical_window = (
        _validate_window_size(
            window_size
        )
    )

    observations = list(
        history
    )

    if not observations:

        raise EmptyTemporalHistoryError(
            "Temporal probability history is empty."
        )

    # Canonical rolling-window semantics:
    # use only the latest N observations.
    observations = (
        observations[
            -canonical_window:
        ]
    )

    normalised_observations = [
        normalise_probability_dict(
            observation,
            labels=canonical_labels,
        )
        for observation
        in observations
    ]

    sample_count = len(
        normalised_observations
    )

    averaged: dict[
        str,
        float
    ] = {}

    for label in canonical_labels:

        averaged[label] = (
            math.fsum(
                observation[
                    label
                ]
                for observation
                in normalised_observations
            )
            / sample_count
        )

    # Preserve the established behaviour of normalising the
    # averaged probability vector before ranking.
    averaged = (
        normalise_probability_dict(
            averaged,
            labels=canonical_labels,
        )
    )

    summary = (
        summarise_probability_dict(
            averaged,
            labels=canonical_labels,
        )
    )

    return {
        "current_state":
            summary[
                "current_state"
            ],

        "confidence":
            summary[
                "confidence"
            ],

        "confidence_percent":
            summary[
                "confidence_percent"
            ],

        "probabilities":
            summary[
                "probabilities"
            ],

        "second_class":
            summary[
                "second_class"
            ],

        "second_probability":
            summary[
                "second_probability"
            ],

        "confidence_gap":
            summary[
                "confidence_gap"
            ],

        "confidence_level":
            summary[
                "confidence_level"
            ],

        "temporal_samples":
            sample_count,

        "temporal_window":
            canonical_window,

        "temporal_window_full":
            (
                sample_count
                == canonical_window
            ),
    }


# ============================================================
# Probability validation
# ============================================================

def validate_probability_distribution(
    probabilities: Mapping[str, Any],
    *,
    labels: Iterable[str] = LABELS,
    tolerance: float = (
        PROBABILITY_SUM_TOLERANCE
    ),
) -> dict[str, Any]:
    """
    Validate an already-normalised probability distribution.

    This is intended for runtime diagnostics and testing.

    Unlike normalise_probability_dict(), this function DOES NOT
    modify the supplied values.

    Returns
    -------
    dict
        {
            "valid": bool,
            "probability_sum": float,
            "sum_valid": bool,
            "ranges_valid": bool,
            "finite_valid": bool,
            "labels_complete": bool,
            "extra_labels": [...],
            "missing_labels": [...]
        }
    """

    canonical_labels = (
        _validate_labels(
            labels
        )
    )

    if not isinstance(
        probabilities,
        Mapping,
    ):

        return {
            "valid":
                False,

            "probability_sum":
                0.0,

            "sum_valid":
                False,

            "ranges_valid":
                False,

            "finite_valid":
                False,

            "labels_complete":
                False,

            "extra_labels":
                [],

            "missing_labels":
                list(
                    canonical_labels
                ),
        }

    missing_labels = [
        label
        for label
        in canonical_labels
        if label not in probabilities
    ]

    extra_labels = [
        str(label)
        for label
        in probabilities.keys()
        if label not in canonical_labels
    ]

    values: list[float] = []

    finite_valid = True
    ranges_valid = True

    for label in canonical_labels:

        if label not in probabilities:

            finite_valid = False
            ranges_valid = False
            continue

        try:

            value = float(
                probabilities[label]
            )

        except (
            TypeError,
            ValueError,
            OverflowError,
        ):

            finite_valid = False
            ranges_valid = False
            continue

        if not math.isfinite(
            value
        ):

            finite_valid = False
            ranges_valid = False
            continue

        values.append(
            value
        )

        if not (
            0.0
            <= value
            <= 1.0
        ):

            ranges_valid = False

    probability_sum = (
        math.fsum(values)
        if values
        else 0.0
    )

    try:

        tolerance_value = abs(
            float(
                tolerance
            )
        )

    except (
        TypeError,
        ValueError,
        OverflowError,
    ):

        tolerance_value = (
            PROBABILITY_SUM_TOLERANCE
        )

    if not math.isfinite(
        tolerance_value
    ):

        tolerance_value = (
            PROBABILITY_SUM_TOLERANCE
        )

    sum_valid = (
        finite_valid
        and
        abs(
            probability_sum
            - 1.0
        )
        <= tolerance_value
    )

    labels_complete = (
        len(
            missing_labels
        )
        == 0
    )

    valid = (
        labels_complete
        and
        finite_valid
        and
        ranges_valid
        and
        sum_valid
    )

    return {
        "valid":
            valid,

        "probability_sum":
            probability_sum,

        "sum_valid":
            sum_valid,

        "ranges_valid":
            ranges_valid,

        "finite_valid":
            finite_valid,

        "labels_complete":
            labels_complete,

        "extra_labels":
            extra_labels,

        "missing_labels":
            missing_labels,
    }


# ============================================================
# Stateful canonical temporal engine
# ============================================================

class TemporalFusionEngine:
    """
    Thread-safe canonical temporal-fusion state manager.

    This class is recommended for both:

        live_fusion_gui.py
        web_app/app.py

    It centralises:

        - rolling history
        - five-sample maximum window
        - probability normalisation
        - temporal aggregation
        - confidence calculation
        - generation-safe reset behaviour

    Generation safety
    -----------------
    Before a potentially long-running multimodal prediction:

        generation = engine.capture_generation()

    After inference completes:

        result = engine.append(
            raw_probabilities,
            expected_generation=generation,
        )

    If reset() occurred while inference was running, append()
    raises StaleGenerationError and the old result cannot enter
    the new temporal history.
    """

    def __init__(
        self,
        *,
        window_size: int = (
            TEMPORAL_PROBABILITY_WINDOW
        ),
        labels: Iterable[str] = LABELS,
        initial_generation: int = 0,
    ) -> None:

        self._labels = (
            _validate_labels(
                labels
            )
        )

        self._window_size = (
            _validate_window_size(
                window_size
            )
        )

        try:

            generation = int(
                initial_generation
            )

        except (
            TypeError,
            ValueError,
        ) as exc:

            raise ValueError(
                "initial_generation must "
                "be a non-negative integer."
            ) from exc

        if generation < 0:

            raise ValueError(
                "initial_generation must "
                "be non-negative."
            )

        self._generation = (
            generation
        )

        self._history: deque[
            dict[str, float]
        ] = deque(
            maxlen=(
                self._window_size
            )
        )

        self._lock = (
            threading.RLock()
        )

    # --------------------------------------------------------
    # Read-only configuration/state properties
    # --------------------------------------------------------

    @property
    def labels(
        self,
    ) -> tuple[str, ...]:
        """
        Canonical class ordering used by this engine.
        """

        return self._labels

    @property
    def window_size(
        self,
    ) -> int:
        """
        Maximum rolling temporal-window length.
        """

        return self._window_size

    @property
    def generation(
        self,
    ) -> int:
        """
        Current reset generation.
        """

        with self._lock:

            return self._generation

    @property
    def sample_count(
        self,
    ) -> int:
        """
        Current number of observations in temporal history.
        """

        with self._lock:

            return len(
                self._history
            )

    @property
    def is_full(
        self,
    ) -> bool:
        """
        True when the rolling temporal window has reached
        its configured maximum length.
        """

        with self._lock:

            return (
                len(
                    self._history
                )
                ==
                self._window_size
            )

    # --------------------------------------------------------
    # Generation helpers
    # --------------------------------------------------------

    def capture_generation(
        self,
    ) -> int:
        """
        Capture the generation associated with a prediction
        immediately before inference starts.
        """

        with self._lock:

            return self._generation

    def is_generation_current(
        self,
        generation: int,
    ) -> bool:
        """
        Return True only if generation still matches the
        engine's current generation.
        """

        try:

            candidate = int(
                generation
            )

        except (
            TypeError,
            ValueError,
        ):

            return False

        with self._lock:

            return (
                candidate
                == self._generation
            )

    def assert_generation(
        self,
        generation: int,
    ) -> None:
        """
        Raise StaleGenerationError when a prediction belongs
        to an older reset generation.
        """

        try:

            candidate = int(
                generation
            )

        except (
            TypeError,
            ValueError,
        ) as exc:

            raise StaleGenerationError(
                "Prediction generation is invalid."
            ) from exc

        with self._lock:

            current = (
                self._generation
            )

        if candidate != current:

            raise StaleGenerationError(
                "Prediction belongs to a stale "
                "temporal generation: "
                f"prediction={candidate}, "
                f"current={current}."
            )

    # --------------------------------------------------------
    # History inspection
    # --------------------------------------------------------

    def history_snapshot(
        self,
    ) -> tuple[
        dict[str, float],
        ...
    ]:
        """
        Return a defensive copy of temporal history.

        The returned observations are chronological:
        oldest -> newest.
        """

        with self._lock:

            return tuple(
                dict(
                    observation
                )
                for observation
                in self._history
            )

    # --------------------------------------------------------
    # Append / aggregate
    # --------------------------------------------------------

    def append(
        self,
        probabilities: Mapping[
            str,
            Any
        ],
        *,
        expected_generation: int | None = None,
    ) -> dict[str, Any]:
        """
        Add ONE raw multimodal probability vector to temporal
        history and return the new canonical temporal result.

        Parameters
        ----------
        probabilities:
            Raw probabilities produced by one call to
            FinalMultimodalInference.predict(...).

        expected_generation:
            Generation captured before inference began.

            If supplied and it no longer matches the engine,
            StaleGenerationError is raised and the observation
            is NOT appended.

        Returns
        -------
        dict
            Canonical temporal aggregation result plus:

                "generation"
        """

        normalised = (
            normalise_probability_dict(
                probabilities,
                labels=self._labels,
            )
        )

        with self._lock:

            if (
                expected_generation
                is not None
            ):

                try:

                    candidate_generation = int(
                        expected_generation
                    )

                except (
                    TypeError,
                    ValueError,
                ) as exc:

                    raise StaleGenerationError(
                        "Prediction generation is invalid."
                    ) from exc

                if (
                    candidate_generation
                    != self._generation
                ):

                    raise StaleGenerationError(
                        "Prediction completed after "
                        "a temporal reset and was rejected: "
                        f"prediction="
                        f"{candidate_generation}, "
                        f"current="
                        f"{self._generation}."
                    )

            self._history.append(
                normalised
            )

            result = (
                aggregate_probability_history(
                    self._history,
                    window_size=(
                        self._window_size
                    ),
                    labels=(
                        self._labels
                    ),
                )
            )

            result[
                "generation"
            ] = self._generation

            return result

    def aggregate(
        self,
    ) -> dict[str, Any]:
        """
        Recalculate the temporal result without appending a
        new observation.

        Raises
        ------
        EmptyTemporalHistoryError
            If no observations have been added.
        """

        with self._lock:

            result = (
                aggregate_probability_history(
                    self._history,
                    window_size=(
                        self._window_size
                    ),
                    labels=(
                        self._labels
                    ),
                )
            )

            result[
                "generation"
            ] = self._generation

            return result

    # --------------------------------------------------------
    # Reset
    # --------------------------------------------------------

    def reset(
        self,
    ) -> int:
        """
        Perform the canonical temporal reset.

        Behaviour:
            1. increment generation;
            2. clear temporal history;
            3. return the new generation.

        This same method should be called for:

            - Reset Temporal Window
            - Full Reset
            - audio-source replacement
            - image-source replacement
            - video-source replacement
            - webcam-source replacement

        Additional modality/session state is cleared by the
        calling application when performing a full reset.
        """

        with self._lock:

            self._generation += 1

            self._history.clear()

            return self._generation

    # --------------------------------------------------------
    # Status representation
    # --------------------------------------------------------

    def status(
        self,
    ) -> dict[str, Any]:
        """
        Return lightweight temporal-engine status suitable for
        GUI/web diagnostics.
        """

        with self._lock:

            sample_count = len(
                self._history
            )

            return {
                "generation":
                    self._generation,

                "temporal_samples":
                    sample_count,

                "temporal_window":
                    self._window_size,

                "temporal_window_full":
                    (
                        sample_count
                        == self._window_size
                    ),

                "labels":
                    list(
                        self._labels
                    ),
            }

    def __len__(
        self,
    ) -> int:
        """
        Return current temporal sample count.
        """

        return self.sample_count

    def __repr__(
        self,
    ) -> str:

        status = self.status()

        return (
            "TemporalFusionEngine("
            f"generation="
            f"{status['generation']}, "
            f"samples="
            f"{status['temporal_samples']}/"
            f"{status['temporal_window']}"
            ")"
        )


# ============================================================
# Built-in deterministic smoke test
# ============================================================

def _self_test() -> None:
    """
    Lightweight dependency-free smoke test.

    This is not a replacement for pytest tests.

    Run with:

        python temporal_fusion.py
    """

    # --------------------------------------------------------
    # 1. Normalisation
    # --------------------------------------------------------

    normalised = (
        normalise_probability_dict(
            {
                "focused":
                    60.0,

                "distracted":
                    20.0,

                "fatigued":
                    10.0,

                "overloaded":
                    10.0,
            }
        )
    )

    assert abs(
        math.fsum(
            normalised.values()
        )
        - 1.0
    ) <= (
        PROBABILITY_SUM_TOLERANCE
    )

    assert abs(
        normalised[
            "focused"
        ]
        - 0.60
    ) <= 1e-12

    # --------------------------------------------------------
    # 2. Negative-value policy
    # --------------------------------------------------------

    negative_test = (
        normalise_probability_dict(
            {
                "focused":
                    -10.0,

                "distracted":
                    1.0,

                "fatigued":
                    0.0,

                "overloaded":
                    0.0,
            }
        )
    )

    assert (
        negative_test[
            "focused"
        ]
        == 0.0
    )

    assert abs(
        negative_test[
            "distracted"
        ]
        - 1.0
    ) <= 1e-12

    # --------------------------------------------------------
    # 3. Zero-total -> uniform
    # --------------------------------------------------------

    uniform = (
        normalise_probability_dict(
            {
                label: 0.0
                for label
                in LABELS
            }
        )
    )

    for label in LABELS:

        assert abs(
            uniform[label]
            - 0.25
        ) <= 1e-12

    # --------------------------------------------------------
    # 4. Confidence thresholds
    # --------------------------------------------------------

    assert (
        confidence_level(
            0.10
        )
        == "Low"
    )

    assert (
        confidence_level(
            0.15
        )
        == "Medium"
    )

    assert (
        confidence_level(
            0.349999
        )
        == "Medium"
    )

    assert (
        confidence_level(
            0.35
        )
        == "High"
    )

    # --------------------------------------------------------
    # 5. Temporal engine
    # --------------------------------------------------------

    engine = (
        TemporalFusionEngine()
    )

    generation_zero = (
        engine.capture_generation()
    )

    assert (
        generation_zero
        == 0
    )

    observations = [
        {
            "focused":
                0.60,

            "distracted":
                0.20,

            "fatigued":
                0.10,

            "overloaded":
                0.10,
        },
        {
            "focused":
                0.62,

            "distracted":
                0.18,

            "fatigued":
                0.10,

            "overloaded":
                0.10,
        },
        {
            "focused":
                0.64,

            "distracted":
                0.16,

            "fatigued":
                0.10,

            "overloaded":
                0.10,
        },
        {
            "focused":
                0.66,

            "distracted":
                0.14,

            "fatigued":
                0.10,

            "overloaded":
                0.10,
        },
        {
            "focused":
                0.68,

            "distracted":
                0.12,

            "fatigued":
                0.10,

            "overloaded":
                0.10,
        },
    ]

    result: dict[
        str,
        Any
    ] = {}

    for observation in observations:

        result = engine.append(
            observation,
            expected_generation=(
                generation_zero
            ),
        )

    assert (
        result[
            "current_state"
        ]
        == "focused"
    )

    assert (
        result[
            "temporal_samples"
        ]
        == 5
    )

    assert (
        result[
            "temporal_window"
        ]
        == 5
    )

    assert (
        result[
            "temporal_window_full"
        ]
        is True
    )

    # Mean focused probability:
    #
    # (0.60 + 0.62 + 0.64 + 0.66 + 0.68) / 5
    # = 0.64
    assert abs(
        result[
            "probabilities"
        ][
            "focused"
        ]
        - 0.64
    ) <= 1e-12

    # --------------------------------------------------------
    # 6. Window rollover
    # --------------------------------------------------------

    result = engine.append(
        {
            "focused":
                0.70,

            "distracted":
                0.10,

            "fatigued":
                0.10,

            "overloaded":
                0.10,
        },
        expected_generation=(
            generation_zero
        ),
    )

    # Still exactly five because oldest item is discarded.
    assert (
        result[
            "temporal_samples"
        ]
        == 5
    )

    assert (
        len(engine)
        == 5
    )

    # --------------------------------------------------------
    # 7. Reset
    # --------------------------------------------------------

    new_generation = (
        engine.reset()
    )

    assert (
        new_generation
        == 1
    )

    assert (
        engine.sample_count
        == 0
    )

    assert (
        engine.is_full
        is False
    )

    # --------------------------------------------------------
    # 8. Stale generation rejection
    # --------------------------------------------------------

    stale_was_rejected = False

    try:

        engine.append(
            {
                "focused":
                    0.70,

                "distracted":
                    0.10,

                "fatigued":
                    0.10,

                "overloaded":
                    0.10,
            },
            expected_generation=(
                generation_zero
            ),
        )

    except StaleGenerationError:

        stale_was_rejected = True

    assert (
        stale_was_rejected
        is True
    )

    assert (
        engine.sample_count
        == 0
    )

    print(
        "temporal_fusion.py self-test: PASS"
    )


# ============================================================
# Direct execution
# ============================================================

if __name__ == "__main__":

    _self_test()
