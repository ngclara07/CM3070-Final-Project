"""
run_all_tests.py

SenseFuzeAI
Canonical Project-Wide Automated Test Runner

=============================================================================
PURPOSE
=============================================================================

This script performs the principal automated software-verification pass for
the current SenseFuzeAI architecture.

The current canonical processing architecture is:

    FinalMultimodalInference
            |
            | ONE raw four-class probability vector
            v
    TemporalFusionEngine
            |
            +--> Desktop GUI
            |
            +--> Web backend
            |
            +--> Offline evaluation


The project also contains a separate external keystroke-dataset comparison
pipeline:

    EmoSurv IEEE keystrokes
            |
            +--> canonical 23-feature representation
            |
    SenseFuzeAI raw keystrokes
            |
            +--> same canonical 23-feature representation
            |
            v
    Harmonised keystroke datasets
            |
            +--> EmoSurv baseline
            +--> SenseFuzeAI-only
            +--> augmented training
            +--> cross-dataset evaluation


This test runner verifies that:

    1. temporal_fusion.py is mathematically correct;

    2. probability normalisation is canonical;

    3. the temporal history uses a maximum window of five observations;

    4. temporal probabilities are the equal arithmetic mean of the latest
       observations;

    5. confidence is the highest temporal probability;

    6. confidence gap is top probability minus second-highest probability;

    7. confidence thresholds remain:
           High   >= 0.35
           Medium >= 0.15
           Low    <  0.15

    8. reset increments the temporal generation and clears history;

    9. stale predictions cannot enter a new generation;

   10. multiple TemporalFusionEngine consumers produce identical results;

   11. final_multimodal_inference.py remains STATELESS and does not maintain
       temporal history;

   12. live_fusion_gui.py uses the shared TemporalFusionEngine rather than
       duplicating temporal mathematics;

   13. web_app/app.py uses one TemporalFusionEngine per browser session;

   14. web_app/static/script.js remains a browser acquisition/presentation
       layer and does not implement temporal probability mathematics;

   15. evaluation/training comparison scripts use the same canonical temporal
       implementation;

   16. all critical Python source files compile successfully;

   17. script.js passes `node --check` when Node.js is available;

   18. the project pytest suite is executed when tests/ exists;

   19. optionally, FinalMultimodalInference can be fully instantiated to
       verify the large pretrained/model artifact dependency chain;

   20. build_keystroke_dataset_comparison.py reuses the canonical EmoSurv
       feature/window representation and preserves dataset provenance;

   21. train_keystroke_dataset_comparison.py retains the A-F experimental
       design, group-aware frozen splits and paired bootstrap comparison;

   22. the pytest suite verifies the harmonised datasets, leakage prevention,
       identical held-out evaluation sets and experiment artifacts.


=============================================================================
DESIGN PRINCIPLE
=============================================================================

Heavy pretrained models are NOT loaded by default.

The normal test command:

    python run_all_tests.py

therefore performs fast mathematical, source-contract, syntax and pytest
verification.

To additionally instantiate all large inference artifacts:

    python run_all_tests.py --with-model-smoke


=============================================================================
OUTPUT
=============================================================================

A machine-readable JSON report is written by default to:

    data/processed/test_results/run_all_tests_summary.json

The command exits with:

    0   all mandatory checks passed
    1   one or more mandatory checks failed

=============================================================================
"""

from __future__ import annotations

import argparse
import ast
import importlib.util
import json
import math
import platform
import py_compile
import random
import shutil
import subprocess
import sys
import time
import traceback

from dataclasses import (
    asdict,
    dataclass,
)

from pathlib import Path
from typing import (
    Any,
    Callable,
    Optional,
)


# =============================================================================
# PROJECT PATHS
# =============================================================================

ROOT_DIR = (
    Path(__file__)
    .resolve()
    .parent
)

TEMPORAL_FUSION_FILE = (
    ROOT_DIR
    / "temporal_fusion.py"
)

FINAL_INFERENCE_FILE = (
    ROOT_DIR
    / "final_multimodal_inference.py"
)

LIVE_GUI_FILE = (
    ROOT_DIR
    / "live_fusion_gui.py"
)

WEB_APP_FILE = (
    ROOT_DIR
    / "web_app"
    / "app.py"
)

WEB_SCRIPT_FILE = (
    ROOT_DIR
    / "web_app"
    / "static"
    / "script.js"
)

EVALUATION_FILE = (
    ROOT_DIR
    / "evaluate_multimodal_results.py"
)

COMPARISON_FILE = (
    ROOT_DIR
    / "train_multimodal_comparison.py"
)

KEYSTROKE_DATASET_BUILDER_FILE = (
    ROOT_DIR
    / "build_keystroke_dataset_comparison.py"
)

KEYSTROKE_DATASET_TRAINER_FILE = (
    ROOT_DIR
    / "train_keystroke_dataset_comparison.py"
)

TESTS_DIR = (
    ROOT_DIR
    / "tests"
)

DEFAULT_REPORT_PATH = (
    ROOT_DIR
    / "data"
    / "processed"
    / "test_results"
    / "run_all_tests_summary.json"
)


# =============================================================================
# CANONICAL EXPECTATIONS
# =============================================================================

EXPECTED_LABELS = (
    "focused",
    "distracted",
    "fatigued",
    "overloaded",
)

EXPECTED_TEMPORAL_WINDOW = 5

EXPECTED_HIGH_GAP = 0.35

EXPECTED_MEDIUM_GAP = 0.15

EXPECTED_LIVE_INTERVAL_MS = 2500

EXPECTED_MIN_TEXT_CHARS = 20

EXPECTED_MIN_KEYDOWNS = 20


# =============================================================================
# TEST RESULT STRUCTURES
# =============================================================================

@dataclass
class CheckResult:
    """
    One project verification result.
    """

    name: str
    category: str
    status: str
    duration_seconds: float
    detail: str


class SkipCheck(Exception):
    """
    Used when an optional test cannot or should not run.
    """


class WarningCheck(Exception):
    """
    Used for a non-fatal verification warning.
    """


# =============================================================================
# TEST RUNNER
# =============================================================================

class TestRunner:
    """
    Lightweight project test orchestrator.
    """

    def __init__(
        self,
        *,
        verbose: bool = False,
    ) -> None:

        self.verbose = verbose

        self.results: list[
            CheckResult
        ] = []

    def run(
        self,
        name: str,
        category: str,
        test_function: Callable[
            [],
            Optional[str],
        ],
    ) -> None:

        started = (
            time.perf_counter()
        )

        try:

            detail = (
                test_function()
                or "Passed."
            )

            status = "PASS"

        except SkipCheck as exc:

            status = "SKIP"
            detail = str(exc)

        except WarningCheck as exc:

            status = "WARN"
            detail = str(exc)

        except Exception as exc:

            status = "FAIL"

            detail = (
                f"{type(exc).__name__}: "
                f"{exc}"
            )

            if self.verbose:

                detail += (
                    "\n\n"
                    + traceback.format_exc()
                )

        duration = (
            time.perf_counter()
            - started
        )

        result = CheckResult(
            name=name,
            category=category,
            status=status,
            duration_seconds=duration,
            detail=detail,
        )

        self.results.append(
            result
        )

        print(
            f"[{status:4s}] "
            f"{category:18s} "
            f"{name}"
            f" ({duration:.3f}s)"
        )

        if (
            status
            in {
                "FAIL",
                "WARN",
            }
            or
            self.verbose
        ):

            print(
                f"       {detail}"
            )

    def counts(
        self,
    ) -> dict[str, int]:

        counts = {
            "PASS": 0,
            "FAIL": 0,
            "WARN": 0,
            "SKIP": 0,
        }

        for result in self.results:

            counts[
                result.status
            ] += 1

        return counts


# =============================================================================
# GENERAL ASSERTION HELPERS
# =============================================================================

def assert_true(
    condition: bool,
    message: str,
) -> None:

    if not condition:

        raise AssertionError(
            message
        )


def assert_equal(
    actual: Any,
    expected: Any,
    message: str,
) -> None:

    if actual != expected:

        raise AssertionError(
            f"{message}\n"
            f"Expected: {expected!r}\n"
            f"Actual:   {actual!r}"
        )


def assert_close(
    actual: float,
    expected: float,
    *,
    tolerance: float = 1e-12,
    message: str,
) -> None:

    actual_value = float(
        actual
    )

    expected_value = float(
        expected
    )

    if not math.isclose(
        actual_value,
        expected_value,
        rel_tol=0.0,
        abs_tol=tolerance,
    ):

        raise AssertionError(
            f"{message}\n"
            f"Expected: {expected_value:.15f}\n"
            f"Actual:   {actual_value:.15f}\n"
            f"Tolerance:{tolerance}"
        )


def assert_probability_dict_close(
    actual: dict[str, float],
    expected: dict[str, float],
    *,
    tolerance: float = 1e-12,
) -> None:

    assert_equal(
        tuple(
            actual.keys()
        ),
        tuple(
            expected.keys()
        ),
        (
            "Probability dictionaries "
            "do not have identical class ordering."
        ),
    )

    for label in expected:

        assert_close(
            actual[
                label
            ],
            expected[
                label
            ],
            tolerance=tolerance,
            message=(
                "Probability mismatch "
                f"for class '{label}'."
            ),
        )


# =============================================================================
# FILE / AST HELPERS
# =============================================================================

def require_file(
    path: Path,
) -> None:

    if not path.exists():

        raise FileNotFoundError(
            f"Required project file not found:\n{path}"
        )

    if not path.is_file():

        raise ValueError(
            f"Expected a file:\n{path}"
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


def parse_python_ast(
    path: Path,
) -> ast.Module:

    source = read_source(
        path
    )

    return ast.parse(
        source,
        filename=str(
            path
        ),
    )


def dotted_name(
    node: ast.AST,
) -> Optional[str]:
    """
    Convert an AST Name/Attribute chain to a dotted string.

    Example:
        self.predictor.predict
    """

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


def imported_names_from(
    tree: ast.Module,
    module_name: str,
) -> set[str]:

    names: set[
        str
    ] = set()

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

            names.add(
                alias.name
            )

    return names


def top_level_function_names(
    tree: ast.Module,
) -> set[str]:

    return {
        node.name
        for node in tree.body
        if isinstance(
            node,
            (
                ast.FunctionDef,
                ast.AsyncFunctionDef,
            ),
        )
    }


def find_class(
    tree: ast.Module,
    class_name: str,
) -> ast.ClassDef:

    for node in tree.body:

        if (
            isinstance(
                node,
                ast.ClassDef,
            )
            and
            node.name
            == class_name
        ):

            return node

    raise AssertionError(
        f"Required class not found: "
        f"{class_name}"
    )


def class_method(
    class_node: ast.ClassDef,
    method_name: str,
) -> ast.FunctionDef | ast.AsyncFunctionDef:

    for node in class_node.body:

        if (
            isinstance(
                node,
                (
                    ast.FunctionDef,
                    ast.AsyncFunctionDef,
                ),
            )
            and
            node.name
            == method_name
        ):

            return node

    raise AssertionError(
        f"Required method not found: "
        f"{class_node.name}.{method_name}"
    )


def top_level_literal(
    tree: ast.Module,
    variable_name: str,
) -> Any:

    for node in tree.body:

        value_node: Optional[
            ast.AST
        ] = None

        if isinstance(
            node,
            ast.Assign,
        ):

            for target in node.targets:

                if (
                    isinstance(
                        target,
                        ast.Name,
                    )
                    and
                    target.id
                    == variable_name
                ):

                    value_node = node.value
                    break

        elif (
            isinstance(
                node,
                ast.AnnAssign,
            )
            and
            isinstance(
                node.target,
                ast.Name,
            )
            and
            node.target.id
            == variable_name
        ):

            value_node = node.value

        if value_node is not None:

            try:

                return ast.literal_eval(
                    value_node
                )

            except Exception as exc:

                raise AssertionError(
                    f"{variable_name} is not "
                    "a simple literal value."
                ) from exc

    raise AssertionError(
        f"Top-level constant not found: "
        f"{variable_name}"
    )


def collect_fastapi_routes(
    tree: ast.Module,
) -> set[
    tuple[str, str]
]:

    routes: set[
        tuple[str, str]
    ] = set()

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

            decorator_name = dotted_name(
                decorator.func
            )

            if decorator_name not in {
                "app.get",
                "app.post",
                "app.put",
                "app.delete",
                "app.patch",
            }:

                continue

            if not decorator.args:

                continue

            route_node = decorator.args[
                0
            ]

            if not (
                isinstance(
                    route_node,
                    ast.Constant,
                )
                and
                isinstance(
                    route_node.value,
                    str,
                )
            ):

                continue

            method = (
                decorator_name
                .split(
                    ".",
                    1,
                )[1]
                .upper()
            )

            routes.add(
                (
                    method,
                    route_node.value,
                )
            )

    return routes


def find_call_keywords(
    tree: ast.Module,
    dotted_callable_name: str,
) -> list[
    set[str]
]:

    matches: list[
        set[str]
    ] = []

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

        if name != dotted_callable_name:

            continue

        matches.append(
            {
                keyword.arg
                for keyword
                in node.keywords
                if keyword.arg
                is not None
            }
        )

    return matches


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

        if (
            name == function_name
            or
            (
                name is not None
                and
                name.endswith(
                    "."
                    + function_name
                )
            )
        ):

            return True

    return False


# =============================================================================
# TEMPORAL-FUSION MATHEMATICAL TESTS
# =============================================================================

def import_temporal_module() -> Any:

    require_file(
        TEMPORAL_FUSION_FILE
    )

    if str(
        ROOT_DIR
    ) not in sys.path:

        sys.path.insert(
            0,
            str(
                ROOT_DIR
            ),
        )

    import temporal_fusion

    return temporal_fusion


def test_temporal_constants() -> str:

    module = import_temporal_module()

    assert_equal(
        tuple(
            module.LABELS
        ),
        EXPECTED_LABELS,
        "Canonical behavioural classes changed.",
    )

    assert_equal(
        module.TEMPORAL_PROBABILITY_WINDOW,
        EXPECTED_TEMPORAL_WINDOW,
        "Canonical temporal window changed.",
    )

    assert_close(
        module.CONFIDENCE_HIGH_GAP,
        EXPECTED_HIGH_GAP,
        message=(
            "High-confidence threshold changed."
        ),
    )

    assert_close(
        module.CONFIDENCE_MEDIUM_GAP,
        EXPECTED_MEDIUM_GAP,
        message=(
            "Medium-confidence threshold changed."
        ),
    )

    return (
        "Canonical labels/window/confidence "
        "thresholds verified."
    )


def test_probability_normalisation() -> str:

    module = import_temporal_module()

    normalised = (
        module
        .normalise_probability_dict(
            {
                "focused": 60.0,
                "distracted": 20.0,
                "fatigued": 10.0,
                "overloaded": 10.0,
            }
        )
    )

    expected = {
        "focused": 0.60,
        "distracted": 0.20,
        "fatigued": 0.10,
        "overloaded": 0.10,
    }

    assert_probability_dict_close(
        normalised,
        expected,
    )

    assert_close(
        math.fsum(
            normalised.values()
        ),
        1.0,
        tolerance=1e-12,
        message=(
            "Normalised probabilities "
            "must sum to one."
        ),
    )

    return (
        "Positive probability normalisation verified."
    )


def test_invalid_probability_policy() -> str:

    module = import_temporal_module()

    result = (
        module
        .normalise_probability_dict(
            {
                "focused": float("nan"),
                "distracted": float("inf"),
                "fatigued": -5.0,
                "overloaded": 2.0,
            }
        )
    )

    expected = {
        "focused": 0.0,
        "distracted": 0.0,
        "fatigued": 0.0,
        "overloaded": 1.0,
    }

    assert_probability_dict_close(
        result,
        expected,
    )

    zero_total = (
        module
        .normalise_probability_dict(
            {
                label: 0.0
                for label
                in EXPECTED_LABELS
            }
        )
    )

    uniform = {
        label: 0.25
        for label
        in EXPECTED_LABELS
    }

    assert_probability_dict_close(
        zero_total,
        uniform,
    )

    return (
        "NaN/infinity/negative/zero-total "
        "probability policy verified."
    )


def test_confidence_thresholds() -> str:

    module = import_temporal_module()

    assertions = [
        (
            0.0,
            "Low",
        ),
        (
            0.149999,
            "Low",
        ),
        (
            0.15,
            "Medium",
        ),
        (
            0.349999,
            "Medium",
        ),
        (
            0.35,
            "High",
        ),
        (
            0.80,
            "High",
        ),
    ]

    for gap, expected in assertions:

        actual = module.confidence_level(
            gap
        )

        assert_equal(
            actual,
            expected,
            (
                "Incorrect confidence "
                f"level for gap {gap}."
            ),
        )

    return (
        "Low/Medium/High confidence "
        "boundaries verified."
    )


def test_temporal_arithmetic_mean() -> str:

    module = import_temporal_module()

    history = [
        {
            "focused": 0.60,
            "distracted": 0.20,
            "fatigued": 0.10,
            "overloaded": 0.10,
        },
        {
            "focused": 0.62,
            "distracted": 0.18,
            "fatigued": 0.10,
            "overloaded": 0.10,
        },
        {
            "focused": 0.64,
            "distracted": 0.16,
            "fatigued": 0.10,
            "overloaded": 0.10,
        },
        {
            "focused": 0.66,
            "distracted": 0.14,
            "fatigued": 0.10,
            "overloaded": 0.10,
        },
        {
            "focused": 0.68,
            "distracted": 0.12,
            "fatigued": 0.10,
            "overloaded": 0.10,
        },
    ]

    result = (
        module
        .aggregate_probability_history(
            history
        )
    )

    expected = {
        "focused": 0.64,
        "distracted": 0.16,
        "fatigued": 0.10,
        "overloaded": 0.10,
    }

    assert_probability_dict_close(
        result[
            "probabilities"
        ],
        expected,
    )

    assert_equal(
        result[
            "current_state"
        ],
        "focused",
        "Temporal argmax is incorrect.",
    )

    assert_close(
        result[
            "confidence"
        ],
        0.64,
        message=(
            "Temporal confidence must equal "
            "the top averaged probability."
        ),
    )

    assert_close(
        result[
            "confidence_gap"
        ],
        0.48,
        message=(
            "Temporal confidence gap "
            "is incorrect."
        ),
    )

    assert_equal(
        result[
            "confidence_level"
        ],
        "High",
        "Expected a High confidence level.",
    )

    assert_equal(
        result[
            "temporal_samples"
        ],
        5,
        "Temporal sample count is incorrect.",
    )

    assert_true(
        bool(
            result[
                "temporal_window_full"
            ]
        ),
        "Five observations should fill the temporal window.",
    )

    return (
        "Equal arithmetic five-observation "
        "probability averaging verified."
    )


def test_temporal_window_rollover() -> str:

    module = import_temporal_module()

    engine = module.TemporalFusionEngine()

    one_hot = {
        "focused":
            {
                "focused": 1.0,
                "distracted": 0.0,
                "fatigued": 0.0,
                "overloaded": 0.0,
            },

        "distracted":
            {
                "focused": 0.0,
                "distracted": 1.0,
                "fatigued": 0.0,
                "overloaded": 0.0,
            },

        "fatigued":
            {
                "focused": 0.0,
                "distracted": 0.0,
                "fatigued": 1.0,
                "overloaded": 0.0,
            },

        "overloaded":
            {
                "focused": 0.0,
                "distracted": 0.0,
                "fatigued": 0.0,
                "overloaded": 1.0,
            },
    }

    sequence = [
        "focused",
        "focused",
        "distracted",
        "fatigued",
        "overloaded",
        "overloaded",
    ]

    result = None

    for label in sequence:

        result = engine.append(
            one_hot[
                label
            ]
        )

    assert_true(
        result is not None,
        "No temporal result produced.",
    )

    assert_equal(
        engine.sample_count,
        5,
        "Temporal history exceeded maxlen=5.",
    )

    expected = {
        "focused": 0.20,
        "distracted": 0.20,
        "fatigued": 0.20,
        "overloaded": 0.40,
    }

    assert_probability_dict_close(
        result[
            "probabilities"
        ],
        expected,
    )

    assert_equal(
        result[
            "current_state"
        ],
        "overloaded",
        (
            "Window rollover did not discard "
            "the oldest observation correctly."
        ),
    )

    return (
        "Rolling maxlen=5 behaviour verified."
    )


def test_reset_and_stale_generation() -> str:

    module = import_temporal_module()

    engine = module.TemporalFusionEngine()

    generation_zero = (
        engine.capture_generation()
    )

    assert_equal(
        generation_zero,
        0,
        "Initial generation should be zero.",
    )

    engine.append(
        {
            "focused": 0.7,
            "distracted": 0.1,
            "fatigued": 0.1,
            "overloaded": 0.1,
        },
        expected_generation=(
            generation_zero
        ),
    )

    assert_equal(
        engine.sample_count,
        1,
        "Observation was not appended.",
    )

    generation_one = (
        engine.reset()
    )

    assert_equal(
        generation_one,
        1,
        "Reset did not increment generation.",
    )

    assert_equal(
        engine.sample_count,
        0,
        "Reset did not clear history.",
    )

    stale_rejected = False

    try:

        engine.append(
            {
                "focused": 0.7,
                "distracted": 0.1,
                "fatigued": 0.1,
                "overloaded": 0.1,
            },
            expected_generation=(
                generation_zero
            ),
        )

    except module.StaleGenerationError:

        stale_rejected = True

    assert_true(
        stale_rejected,
        (
            "An observation from an old generation "
            "was not rejected."
        ),
    )

    assert_equal(
        engine.sample_count,
        0,
        (
            "Rejected stale result must not "
            "enter temporal history."
        ),
    )

    engine.append(
        {
            "focused": 0.2,
            "distracted": 0.6,
            "fatigued": 0.1,
            "overloaded": 0.1,
        },
        expected_generation=(
            generation_one
        ),
    )

    assert_equal(
        engine.sample_count,
        1,
        (
            "Current-generation observation "
            "was not accepted."
        ),
    )

    return (
        "Generation increment, reset and "
        "stale-result rejection verified."
    )


def test_temporal_consumer_parity() -> str:
    """
    Simulate desktop/backend consumers of the same shared engine.

    Identical probability streams and reset points must produce
    numerically identical outputs.
    """

    module = import_temporal_module()

    desktop_engine = (
        module.TemporalFusionEngine()
    )

    web_engine = (
        module.TemporalFusionEngine()
    )

    generator = random.Random(
        42
    )

    reset_points = {
        8,
        19,
        31,
    }

    for index in range(
        40
    ):

        if index in reset_points:

            desktop_generation = (
                desktop_engine.reset()
            )

            web_generation = (
                web_engine.reset()
            )

            assert_equal(
                desktop_generation,
                web_generation,
                (
                    "Desktop/web simulated "
                    "generations diverged."
                ),
            )

        raw = {
            label:
                generator.random()
            for label
            in EXPECTED_LABELS
        }

        desktop_result = (
            desktop_engine.append(
                raw,
                expected_generation=(
                    desktop_engine
                    .capture_generation()
                ),
            )
        )

        web_result = (
            web_engine.append(
                raw,
                expected_generation=(
                    web_engine
                    .capture_generation()
                ),
            )
        )

        assert_equal(
            desktop_result[
                "current_state"
            ],
            web_result[
                "current_state"
            ],
            (
                "Simulated desktop/web "
                "temporal labels diverged."
            ),
        )

        assert_equal(
            desktop_result[
                "confidence_level"
            ],
            web_result[
                "confidence_level"
            ],
            (
                "Simulated desktop/web confidence "
                "levels diverged."
            ),
        )

        assert_equal(
            desktop_result[
                "temporal_samples"
            ],
            web_result[
                "temporal_samples"
            ],
            (
                "Simulated desktop/web history "
                "lengths diverged."
            ),
        )

        assert_probability_dict_close(
            desktop_result[
                "probabilities"
            ],
            web_result[
                "probabilities"
            ],
            tolerance=1e-15,
        )

        assert_close(
            desktop_result[
                "confidence_gap"
            ],
            web_result[
                "confidence_gap"
            ],
            tolerance=1e-15,
            message=(
                "Simulated desktop/web "
                "confidence gaps diverged."
            ),
        )

    return (
        "40 deterministic observations with "
        "multiple resets produced exact "
        "desktop/web engine parity."
    )


def test_probability_validator() -> str:

    module = import_temporal_module()

    valid = (
        module
        .validate_probability_distribution(
            {
                "focused": 0.25,
                "distracted": 0.25,
                "fatigued": 0.25,
                "overloaded": 0.25,
            }
        )
    )

    assert_true(
        bool(
            valid[
                "valid"
            ]
        ),
        "Valid distribution was rejected.",
    )

    invalid = (
        module
        .validate_probability_distribution(
            {
                "focused": 0.7,
                "distracted": 0.7,
                "fatigued": 0.0,
                "overloaded": 0.0,
            }
        )
    )

    assert_true(
        not bool(
            invalid[
                "valid"
            ]
        ),
        "Non-unit probability distribution was accepted.",
    )

    return (
        "Probability-distribution diagnostics verified."
    )


# =============================================================================
# PYTHON SOURCE CONTRACT TESTS
# =============================================================================

FORBIDDEN_DUPLICATE_TEMPORAL_FUNCTIONS = {
    "normalise_probability_dict",
    "normalize_probability_dict",
    "aggregate_probability_history",
    "confidence_level",
}


def assert_no_local_temporal_math(
    tree: ast.Module,
    *,
    filename: str,
) -> None:

    functions = top_level_function_names(
        tree
    )

    duplicates = (
        functions
        &
        FORBIDDEN_DUPLICATE_TEMPORAL_FUNCTIONS
    )

    if duplicates:

        raise AssertionError(
            f"{filename} duplicates canonical "
            "temporal mathematics:\n"
            f"{sorted(duplicates)}"
        )


def test_final_inference_source_contract() -> str:

    tree = parse_python_ast(
        FINAL_INFERENCE_FILE
    )

    imported = imported_names_from(
        tree,
        "temporal_fusion",
    )

    required_imports = {
        "LABELS",
        "normalise_probability_dict",
        "summarise_probability_dict",
        "validate_probability_distribution",
    }

    missing_imports = (
        required_imports
        - imported
    )

    assert_true(
        not missing_imports,
        (
            "final_multimodal_inference.py "
            "is missing canonical temporal/probability imports:\n"
            f"{sorted(missing_imports)}"
        ),
    )

    assert_no_local_temporal_math(
        tree,
        filename=(
            "final_multimodal_inference.py"
        ),
    )

    assert_true(
        not contains_named_call(
            tree,
            "TemporalFusionEngine",
        ),
        (
            "FinalMultimodalInference must remain "
            "stateless and must not instantiate "
            "TemporalFusionEngine."
        ),
    )

    inference_class = find_class(
        tree,
        "FinalMultimodalInference",
    )

    predict_method = class_method(
        inference_class,
        "predict",
    )

    argument_names = [
        argument.arg
        for argument
        in predict_method.args.args
    ]

    assert_equal(
        argument_names,
        [
            "self",
            "keystroke_json",
            "text",
            "audio_path",
            "image_path",
        ],
        (
            "FinalMultimodalInference.predict(...) "
            "public API changed."
        ),
    )

    required_methods = {
        "build_fusion_dataframe",
        "extract_audio_features",
        "extract_clip_embedding",
        "extract_image_features",
        "extract_keystroke_features",
        "extract_text_features",
        "extract_webcam_calibration_features",
        "predict",
    }

    observed_methods = {
        node.name
        for node
        in inference_class.body
        if isinstance(
            node,
            (
                ast.FunctionDef,
                ast.AsyncFunctionDef,
            ),
        )
    }

    missing_methods = (
        required_methods
        - observed_methods
    )

    assert_true(
        not missing_methods,
        (
            "Canonical inference methods are missing:\n"
            f"{sorted(missing_methods)}"
        ),
    )

    return (
        "Stateless raw-inference API and shared "
        "probability contract verified."
    )


def test_live_gui_source_contract() -> str:

    tree = parse_python_ast(
        LIVE_GUI_FILE
    )

    imported = imported_names_from(
        tree,
        "temporal_fusion",
    )

    required = {
        "LABELS",
        "TEMPORAL_PROBABILITY_WINDOW",
        "TemporalFusionEngine",
        "StaleGenerationError",
    }

    missing = (
        required
        - imported
    )

    assert_true(
        not missing,
        (
            "live_fusion_gui.py is missing "
            "shared temporal imports:\n"
            f"{sorted(missing)}"
        ),
    )

    assert_no_local_temporal_math(
        tree,
        filename="live_fusion_gui.py",
    )

    assert_true(
        "probability_history"
        not in read_source(
            LIVE_GUI_FILE
        ),
        (
            "Desktop GUI still appears to maintain "
            "its own probability_history."
        ),
    )

    assert_equal(
        top_level_literal(
            tree,
            "LIVE_FUSION_INTERVAL_MS",
        ),
        EXPECTED_LIVE_INTERVAL_MS,
        "Desktop prediction cadence changed.",
    )

    assert_equal(
        top_level_literal(
            tree,
            "MIN_TEXT_CHARS",
        ),
        EXPECTED_MIN_TEXT_CHARS,
        "Desktop minimum text threshold changed.",
    )

    assert_equal(
        top_level_literal(
            tree,
            "MIN_KEYDOWNS",
        ),
        EXPECTED_MIN_KEYDOWNS,
        "Desktop minimum key-down threshold changed.",
    )

    app_class = find_class(
        tree,
        "FusionDemoApp",
    )

    for method in (
        "predict_fusion_threaded",
        "apply_prediction",
        "reset_temporal_history",
        "reset",
    ):

        class_method(
            app_class,
            method,
        )

    predictor_calls = find_call_keywords(
        tree,
        "self.predictor.predict",
    )

    assert_true(
        len(
            predictor_calls
        )
        >= 1,
        (
            "Desktop GUI no longer calls "
            "self.predictor.predict(...)."
        ),
    )

    expected_keywords = {
        "keystroke_json",
        "text",
        "audio_path",
        "image_path",
    }

    assert_true(
        any(
            expected_keywords
            <= keywords
            for keywords
            in predictor_calls
        ),
        (
            "Desktop predictor call does not supply "
            "the canonical four inference arguments."
        ),
    )

    return (
        "Desktop shared temporal engine, cadence, "
        "gating constants and raw inference call verified."
    )


def test_web_backend_source_contract() -> str:

    tree = parse_python_ast(
        WEB_APP_FILE
    )

    source = read_source(
        WEB_APP_FILE
    )

    imported = imported_names_from(
        tree,
        "temporal_fusion",
    )

    required_imports = {
        "LABELS",
        "TEMPORAL_PROBABILITY_WINDOW",
        "TemporalFusionEngine",
        "StaleGenerationError",
    }

    missing = (
        required_imports
        - imported
    )

    assert_true(
        not missing,
        (
            "web_app/app.py is missing shared "
            "temporal imports:\n"
            f"{sorted(missing)}"
        ),
    )

    assert_no_local_temporal_math(
        tree,
        filename="web_app/app.py",
    )

    assert_true(
        "probability_history"
        not in source,
        (
            "Web backend still appears to maintain "
            "a duplicated probability_history."
        ),
    )

    session_class = find_class(
        tree,
        "SessionState",
    )

    session_source = (
        ast.get_source_segment(
            source,
            session_class,
        )
        or ""
    )

    assert_true(
        "TemporalFusionEngine"
        in session_source,
        (
            "SessionState does not appear to own "
            "a per-session TemporalFusionEngine."
        ),
    )

    routes = collect_fastapi_routes(
        tree
    )

    required_routes = {
        (
            "POST",
            "/set_audio_source",
        ),
        (
            "POST",
            "/set_visual_image",
        ),
        (
            "POST",
            "/set_visual_video",
        ),
        (
            "POST",
            "/set_visual_webcam",
        ),
        (
            "POST",
            "/stop_visual",
        ),
        (
            "POST",
            "/predict_live",
        ),
        (
            "POST",
            "/reset_temporal",
        ),
        (
            "POST",
            "/full_reset",
        ),
        (
            "GET",
            "/model-status",
        ),
    }

    missing_routes = (
        required_routes
        - routes
    )

    assert_true(
        not missing_routes,
        (
            "Web backend is missing required routes:\n"
            f"{sorted(missing_routes)}"
        ),
    )

    predictor_calls = find_call_keywords(
        tree,
        "predictor.predict",
    )

    expected_keywords = {
        "keystroke_json",
        "text",
        "audio_path",
        "image_path",
    }

    assert_true(
        any(
            expected_keywords
            <= keywords
            for keywords
            in predictor_calls
        ),
        (
            "Web canonical predictor invocation does not "
            "supply all four modalities."
        ),
    )

    return (
        "Per-session temporal engine, required web "
        "routes and canonical predictor call verified."
    )


def test_evaluation_source_contract() -> str:

    tree = parse_python_ast(
        EVALUATION_FILE
    )

    imported = imported_names_from(
        tree,
        "temporal_fusion",
    )

    assert_true(
        "TemporalFusionEngine"
        in imported,
        (
            "evaluate_multimodal_results.py "
            "does not import TemporalFusionEngine."
        ),
    )

    assert_no_local_temporal_math(
        tree,
        filename=(
            "evaluate_multimodal_results.py"
        ),
    )

    assert_true(
        contains_named_call(
            tree,
            "TemporalFusionEngine",
        ),
        (
            "Evaluation script imports but does not "
            "use the canonical temporal engine."
        ),
    )

    source = read_source(
        EVALUATION_FILE
    )

    required_terms = {
        "raw",
        "temporal",
        "generation",
        "confusion_matrix",
        "macro_f1",
    }

    missing_terms = {
        term
        for term
        in required_terms
        if term
        not in source
    }

    assert_true(
        not missing_terms,
        (
            "Evaluation script is missing expected "
            "raw/temporal evaluation concepts:\n"
            f"{sorted(missing_terms)}"
        ),
    )

    return (
        "Raw-vs-temporal evaluation uses "
        "the canonical engine."
    )


def test_comparison_source_contract() -> str:

    tree = parse_python_ast(
        COMPARISON_FILE
    )

    source = read_source(
        COMPARISON_FILE
    )

    imported = imported_names_from(
        tree,
        "temporal_fusion",
    )

    assert_true(
        "TemporalFusionEngine"
        in imported,
        (
            "train_multimodal_comparison.py "
            "does not import TemporalFusionEngine."
        ),
    )

    assert_no_local_temporal_math(
        tree,
        filename=(
            "train_multimodal_comparison.py"
        ),
    )

    assert_true(
        contains_named_call(
            tree,
            "TemporalFusionEngine",
        ),
        (
            "Multimodal comparison script does not "
            "apply canonical temporal fusion."
        ),
    )

    assert_true(
        "StratifiedGroupKFold"
        in source,
        (
            "Comparison experiment does not appear "
            "to implement group-aware splitting."
        ),
    )

    assert_true(
        "train_test_split"
        not in source,
        (
            "Legacy row-level train_test_split is "
            "still present in the comparison script."
        ),
    )

    assert_true(
        ".duplicated().any()"
        not in source,
        (
            "Legacy duplicate-session rejection "
            "still appears to be present."
        ),
    )

    return (
        "Group-aware raw-vs-temporal comparison "
        "contract verified."
    )


# =============================================================================
# KEYSTROKE DATASET COMPARISON SOURCE CONTRACTS
# =============================================================================

def test_keystroke_dataset_builder_source_contract() -> str:

    tree = parse_python_ast(
        KEYSTROKE_DATASET_BUILDER_FILE
    )

    source = read_source(
        KEYSTROKE_DATASET_BUILDER_FILE
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

    missing_imports = (
        required_imports
        - imported
    )

    assert_true(
        not missing_imports,
        (
            "build_keystroke_dataset_comparison.py "
            "is missing canonical EmoSurv "
            "feature/window imports:\n"
            f"{sorted(missing_imports)}"
        ),
    )

    assert_true(
        contains_named_call(
            tree,
            "extract_live_features",
        ),
        (
            "SenseFuzeAI harmonisation does not "
            "appear to call the canonical "
            "extract_live_features() implementation."
        ),
    )

    required_terms = {
        "dataset_source",
        "participant_id",
        "session_id",
        "sample_id",
        "label_origin",
        "combined_harmonised_3class.csv",
        "combined_harmonised_4class.csv",
    }

    missing_terms = {
        term
        for term
        in required_terms
        if term
        not in source
    }

    assert_true(
        not missing_terms,
        (
            "Keystroke dataset builder is missing "
            "required harmonisation/provenance "
            "concepts:\n"
            f"{sorted(missing_terms)}"
        ),
    )

    return (
        "Common EmoSurv/SenseFuzeAI 23-feature "
        "harmonisation and provenance contract verified."
    )


def test_keystroke_dataset_trainer_source_contract() -> str:

    tree = parse_python_ast(
        KEYSTROKE_DATASET_TRAINER_FILE
    )

    source = read_source(
        KEYSTROKE_DATASET_TRAINER_FILE
    )

    assert_true(
        tree is not None,
        (
            "Unable to parse "
            "train_keystroke_dataset_comparison.py."
        ),
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
        for term
        in required_terms
        if term
        not in source
    }

    assert_true(
        not missing_terms,
        (
            "Keystroke comparison trainer is "
            "missing required experiment/split "
            "concepts:\n"
            f"{sorted(missing_terms)}"
        ),
    )

    group_aware_present = (
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
    )

    assert_true(
        group_aware_present,
        (
            "Keystroke dataset comparison no longer "
            "appears to use group-aware splitting."
        ),
    )

    assert_true(
        "bootstrap"
        in source.lower(),
        (
            "Paired bootstrap uncertainty analysis "
            "no longer appears in the keystroke "
            "dataset-comparison trainer."
        ),
    )

    assert_true(
        "train_test_split"
        not in source,
        (
            "Row-level train_test_split appears in "
            "train_keystroke_dataset_comparison.py. "
            "Participant/session-aware grouping "
            "must be retained."
        ),
    )

    return (
        "A-F experiment design, frozen split, "
        "group-aware evaluation and bootstrap "
        "comparison contract verified."
    )


# =============================================================================
# JAVASCRIPT CONTRACT TEST
# =============================================================================

def test_javascript_source_contract() -> str:

    source = read_source(
        WEB_SCRIPT_FILE
    )

    required_fragments = [
        "serverGeneration",
        "clientEpoch",
        "runLivePrediction",
        "/set_audio_source",
        "/set_visual_image",
        "/set_visual_video",
        "/set_visual_webcam",
        "/predict_live",
        "/reset_temporal",
        "/full_reset",
        "setInterval",
        "data.labels",
    ]

    missing = [
        fragment
        for fragment
        in required_fragments
        if fragment
        not in source
    ]

    assert_true(
        not missing,
        (
            "script.js is missing expected "
            "client/backend integration pieces:\n"
            f"{missing}"
        ),
    )

    # The browser must not independently own canonical
    # temporal probability mathematics.
    suspicious_fragments = [
        "aggregateProbabilityHistory",
        "aggregate_probability_history",
        "normaliseProbabilityDict",
        "normalizeProbabilityDict",
        "confidenceLevelFromGap",
        "temporalProbabilityHistory",
    ]

    found = [
        fragment
        for fragment
        in suspicious_fragments
        if fragment
        in source
    ]

    assert_true(
        not found,
        (
            "script.js appears to duplicate canonical "
            "temporal mathematics:\n"
            f"{found}"
        ),
    )

    # Fixed-audio architecture should not continuously create
    # MediaRecorder chunks every fusion cycle.
    assert_true(
        "MediaRecorder"
        not in source,
        (
            "script.js contains MediaRecorder. "
            "Verify that browser audio has not reverted "
            "to continuously replaced recordings."
        ),
    )

    return (
        "Browser client remains acquisition/display "
        "oriented with generation-safe server integration."
    )


# =============================================================================
# PYTHON SYNTAX CHECKS
# =============================================================================

def compile_python_file(
    path: Path,
) -> str:

    require_file(
        path
    )

    py_compile.compile(
        str(
            path
        ),
        doraise=True,
    )

    return (
        f"Compiled successfully: "
        f"{path.relative_to(ROOT_DIR)}"
    )


# =============================================================================
# NODE.JS SYNTAX CHECK
# =============================================================================

def run_node_syntax_check(
    *,
    require_node: bool,
    timeout_seconds: int,
) -> str:

    require_file(
        WEB_SCRIPT_FILE
    )

    node = shutil.which(
        "node"
    )

    if not node:

        message = (
            "Node.js is not available; "
            "script.js syntax check was skipped."
        )

        if require_node:

            raise RuntimeError(
                message
            )

        raise SkipCheck(
            message
        )

    completed = subprocess.run(
        [
            node,
            "--check",
            str(
                WEB_SCRIPT_FILE
            ),
        ],
        cwd=ROOT_DIR,
        capture_output=True,
        text=True,
        timeout=(
            timeout_seconds
        ),
        check=False,
    )

    if completed.returncode != 0:

        raise RuntimeError(
            (
                completed.stderr
                or completed.stdout
                or "node --check failed."
            ).strip()
        )

    return (
        "node --check passed for "
        "web_app/static/script.js."
    )


# =============================================================================
# PYTEST
# =============================================================================

def run_pytest_suite(
    *,
    require_pytest: bool,
    timeout_seconds: int,
) -> str:

    if not TESTS_DIR.exists():

        message = (
            "tests/ directory does not exist."
        )

        if require_pytest:

            raise RuntimeError(
                message
            )

        raise SkipCheck(
            message
        )

    test_files = list(
        TESTS_DIR.rglob(
            "test_*.py"
        )
    )

    if not test_files:

        message = (
            "tests/ exists but contains "
            "no test_*.py files."
        )

        if require_pytest:

            raise RuntimeError(
                message
            )

        raise SkipCheck(
            message
        )

    if (
        importlib.util.find_spec(
            "pytest"
        )
        is None
    ):

        message = (
            "pytest is not installed."
        )

        if require_pytest:

            raise RuntimeError(
                message
            )

        raise SkipCheck(
            message
        )

    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            str(
                TESTS_DIR
            ),
            "-q",
        ],
        cwd=ROOT_DIR,
        capture_output=True,
        text=True,
        timeout=(
            timeout_seconds
        ),
        check=False,
    )

    output = (
        (
            completed.stdout
            or ""
        )
        +
        (
            "\n"
            + completed.stderr
            if completed.stderr
            else ""
        )
    ).strip()

    if completed.returncode != 0:

        raise RuntimeError(
            (
                "pytest failed.\n\n"
                + output
            )
        )

    return (
        "pytest passed.\n"
        + output
    )


# =============================================================================
# OPTIONAL HEAVY MODEL SMOKE TEST
# =============================================================================

def run_model_initialisation_smoke(
    *,
    enabled: bool,
    timeout_seconds: int,
) -> str:

    if not enabled:

        raise SkipCheck(
            (
                "Heavy pretrained-model smoke test "
                "disabled. Use --with-model-smoke "
                "to enable it."
            )
        )

    require_file(
        FINAL_INFERENCE_FILE
    )

    command = (
        "from final_multimodal_inference "
        "import FinalMultimodalInference; "
        "m = FinalMultimodalInference(); "
        "print('MODEL_SMOKE_PASS'); "
        "print('feature_count=', len(m.feature_columns)); "
        "print('device=', m.device)"
    )

    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            command,
        ],
        cwd=ROOT_DIR,
        capture_output=True,
        text=True,
        timeout=(
            timeout_seconds
        ),
        check=False,
    )

    output = (
        (
            completed.stdout
            or ""
        )
        +
        (
            "\n"
            + completed.stderr
            if completed.stderr
            else ""
        )
    ).strip()

    if completed.returncode != 0:

        raise RuntimeError(
            (
                "FinalMultimodalInference "
                "initialisation failed.\n\n"
                + output
            )
        )

    assert_true(
        "MODEL_SMOKE_PASS"
        in output,
        (
            "Model smoke subprocess exited successfully "
            "but did not emit its completion marker."
        ),
    )

    return output


# =============================================================================
# TEMPORAL SELF-TEST SUBPROCESS
# =============================================================================

def run_temporal_self_test_subprocess(
    *,
    timeout_seconds: int,
) -> str:

    require_file(
        TEMPORAL_FUSION_FILE
    )

    completed = subprocess.run(
        [
            sys.executable,
            str(
                TEMPORAL_FUSION_FILE
            ),
        ],
        cwd=ROOT_DIR,
        capture_output=True,
        text=True,
        timeout=(
            timeout_seconds
        ),
        check=False,
    )

    output = (
        (
            completed.stdout
            or ""
        )
        +
        (
            "\n"
            + completed.stderr
            if completed.stderr
            else ""
        )
    ).strip()

    if completed.returncode != 0:

        raise RuntimeError(
            (
                "temporal_fusion.py self-test failed.\n\n"
                + output
            )
        )

    assert_true(
        "self-test: PASS"
        in output,
        (
            "temporal_fusion.py did not report "
            "a PASS completion marker."
        ),
    )

    return output


# =============================================================================
# REPORT
# =============================================================================

def save_report(
    *,
    path: Path,
    runner: TestRunner,
    arguments: argparse.Namespace,
) -> None:

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    counts = runner.counts()

    payload = {
        "project":
            "SenseFuzeAI",

        "test_runner":
            "run_all_tests.py",

        "timestamp_epoch":
            time.time(),

        "environment": {
            "python_version":
                sys.version,

            "python_executable":
                sys.executable,

            "platform":
                platform.platform(),

            "project_root":
                str(
                    ROOT_DIR
                ),
        },

        "canonical_expectations": {
            "labels":
                list(
                    EXPECTED_LABELS
                ),

            "temporal_window":
                EXPECTED_TEMPORAL_WINDOW,

            "confidence_high_gap":
                EXPECTED_HIGH_GAP,

            "confidence_medium_gap":
                EXPECTED_MEDIUM_GAP,

            "live_interval_ms":
                EXPECTED_LIVE_INTERVAL_MS,

            "min_text_chars":
                EXPECTED_MIN_TEXT_CHARS,

            "min_keydowns":
                EXPECTED_MIN_KEYDOWNS,

            "keystroke_dataset_comparison": {
                "builder":
                    str(
                        KEYSTROKE_DATASET_BUILDER_FILE.name
                    ),

                "trainer":
                    str(
                        KEYSTROKE_DATASET_TRAINER_FILE.name
                    ),

                "primary_analysis":
                    "three_class",

                "exploratory_analysis":
                    "four_class",
            },
        },

        "options": {
            "with_model_smoke":
                bool(
                    arguments.with_model_smoke
                ),

            "require_node":
                bool(
                    arguments.require_node
                ),

            "require_pytest":
                bool(
                    arguments.require_pytest
                ),

            "strict":
                bool(
                    arguments.strict
                ),
        },

        "summary":
            counts,

        "results": [
            asdict(
                result
            )
            for result
            in runner.results
        ],
    }

    with path.open(
        "w",
        encoding="utf-8",
    ) as file_handle:

        json.dump(
            payload,
            file_handle,
            indent=4,
        )


# =============================================================================
# COMMAND-LINE ARGUMENTS
# =============================================================================

def build_argument_parser() -> argparse.ArgumentParser:

    parser = argparse.ArgumentParser(
        description=(
            "Run canonical SenseFuzeAI software, "
            "temporal-fusion and dataset-comparison tests."
        )
    )

    parser.add_argument(
        "--with-model-smoke",
        action="store_true",
        help=(
            "Load all pretrained/model artifacts by "
            "instantiating FinalMultimodalInference."
        ),
    )

    parser.add_argument(
        "--require-node",
        action="store_true",
        help=(
            "Treat missing Node.js as a test failure."
        ),
    )

    parser.add_argument(
        "--require-pytest",
        action="store_true",
        help=(
            "Treat a missing pytest/tests suite "
            "as a test failure."
        ),
    )

    parser.add_argument(
        "--strict",
        action="store_true",
        help=(
            "Treat WARN results as failures."
        ),
    )

    parser.add_argument(
        "--verbose",
        action="store_true",
        help=(
            "Print detailed exception tracebacks."
        ),
    )

    parser.add_argument(
        "--subprocess-timeout",
        type=int,
        default=120,
        help=(
            "Timeout in seconds for ordinary "
            "subprocess-based checks."
        ),
    )

    parser.add_argument(
        "--model-timeout",
        type=int,
        default=600,
        help=(
            "Timeout in seconds for the optional "
            "heavy model initialisation smoke test."
        ),
    )

    parser.add_argument(
        "--report",
        type=Path,
        default=(
            DEFAULT_REPORT_PATH
        ),
        help=(
            "JSON summary report path."
        ),
    )

    return parser


# =============================================================================
# MAIN TEST EXECUTION
# =============================================================================

def main() -> None:

    parser = build_argument_parser()

    args = parser.parse_args()

    if (
        args.subprocess_timeout
        <= 0
    ):

        parser.error(
            "--subprocess-timeout must be positive."
        )

    if (
        args.model_timeout
        <= 0
    ):

        parser.error(
            "--model-timeout must be positive."
        )

    runner = TestRunner(
        verbose=(
            args.verbose
        )
    )

    print()

    print(
        "=" * 88
    )

    print(
        "SenseFuzeAI Canonical Automated Test Suite"
    )

    print(
        "=" * 88
    )

    print(
        f"Project root: {ROOT_DIR}"
    )

    print()

    # =========================================================================
    # 1. Python syntax
    # =========================================================================

    python_files = [
        TEMPORAL_FUSION_FILE,
        FINAL_INFERENCE_FILE,
        LIVE_GUI_FILE,
        WEB_APP_FILE,
        EVALUATION_FILE,
        COMPARISON_FILE,
        KEYSTROKE_DATASET_BUILDER_FILE,
        KEYSTROKE_DATASET_TRAINER_FILE,
    ]

    for path in python_files:

        relative_name = str(
            path.relative_to(
                ROOT_DIR
            )
        )

        runner.run(
            (
                "py_compile "
                + relative_name
            ),
            "syntax",
            lambda path=path:
                compile_python_file(
                    path
                ),
        )

    # =========================================================================
    # 2. Canonical temporal mathematics
    # =========================================================================

    runner.run(
        "canonical constants",
        "temporal",
        test_temporal_constants,
    )

    runner.run(
        "probability normalisation",
        "temporal",
        test_probability_normalisation,
    )

    runner.run(
        "invalid probability policy",
        "temporal",
        test_invalid_probability_policy,
    )

    runner.run(
        "confidence thresholds",
        "temporal",
        test_confidence_thresholds,
    )

    runner.run(
        "five-sample arithmetic mean",
        "temporal",
        test_temporal_arithmetic_mean,
    )

    runner.run(
        "rolling-window rollover",
        "temporal",
        test_temporal_window_rollover,
    )

    runner.run(
        "reset + stale generation",
        "temporal",
        test_reset_and_stale_generation,
    )

    runner.run(
        "consumer parity",
        "temporal",
        test_temporal_consumer_parity,
    )

    runner.run(
        "probability validator",
        "temporal",
        test_probability_validator,
    )

    runner.run(
        "built-in temporal self-test",
        "temporal",
        lambda:
            run_temporal_self_test_subprocess(
                timeout_seconds=(
                    args.subprocess_timeout
                )
            ),
    )

    # =========================================================================
    # 3. Architecture/source contracts
    # =========================================================================

    runner.run(
        "stateless final inference",
        "architecture",
        test_final_inference_source_contract,
    )

    runner.run(
        "desktop shared temporal engine",
        "architecture",
        test_live_gui_source_contract,
    )

    runner.run(
        "web shared temporal engine",
        "architecture",
        test_web_backend_source_contract,
    )

    runner.run(
        "evaluation temporal parity",
        "architecture",
        test_evaluation_source_contract,
    )

    runner.run(
        "comparison group-aware temporal evaluation",
        "architecture",
        test_comparison_source_contract,
    )

    # -------------------------------------------------------------------------
    # New EmoSurv / SenseFuzeAI keystroke dataset-comparison contracts
    # -------------------------------------------------------------------------

    runner.run(
        "keystroke dataset harmonisation contract",
        "dataset-comparison",
        test_keystroke_dataset_builder_source_contract,
    )

    runner.run(
        "keystroke dataset comparison split contract",
        "dataset-comparison",
        test_keystroke_dataset_trainer_source_contract,
    )

    runner.run(
        "JavaScript client contract",
        "architecture",
        test_javascript_source_contract,
    )

    # =========================================================================
    # 4. JavaScript syntax
    # =========================================================================

    runner.run(
        "node --check script.js",
        "javascript",
        lambda:
            run_node_syntax_check(
                require_node=(
                    args.require_node
                ),
                timeout_seconds=(
                    args.subprocess_timeout
                ),
            ),
    )

    # =========================================================================
    # 5. Project pytest suite
    #
    # This automatically discovers:
    #
    #   tests/test_01_unit.py
    #   tests/test_02_integration.py
    #   tests/test_03_system.py
    #   tests/test_04_acceptance.py
    #   tests/test_05_keystroke_dataset_comparison.py
    #   tests/test_sensefuzeai.py
    #
    # and any future test_*.py files.
    # =========================================================================

    runner.run(
        "pytest tests/",
        "pytest",
        lambda:
            run_pytest_suite(
                require_pytest=(
                    args.require_pytest
                ),
                timeout_seconds=(
                    args.subprocess_timeout
                ),
            ),
    )

    # =========================================================================
    # 6. Optional heavyweight model-artifact smoke test
    # =========================================================================

    runner.run(
        "FinalMultimodalInference initialisation",
        "model-smoke",
        lambda:
            run_model_initialisation_smoke(
                enabled=(
                    args.with_model_smoke
                ),
                timeout_seconds=(
                    args.model_timeout
                ),
            ),
    )

    # =========================================================================
    # Summary
    # =========================================================================

    counts = runner.counts()

    save_report(
        path=(
            args.report
        ),
        runner=runner,
        arguments=args,
    )

    print()

    print(
        "=" * 88
    )

    print(
        "TEST SUMMARY"
    )

    print(
        "=" * 88
    )

    print(
        f"PASS : {counts['PASS']}"
    )

    print(
        f"FAIL : {counts['FAIL']}"
    )

    print(
        f"WARN : {counts['WARN']}"
    )

    print(
        f"SKIP : {counts['SKIP']}"
    )

    print()

    print(
        "JSON report:"
    )

    print(
        args.report
    )

    mandatory_failure = (
        counts[
            "FAIL"
        ]
        > 0
    )

    strict_warning_failure = (
        args.strict
        and
        counts[
            "WARN"
        ]
        > 0
    )

    if (
        mandatory_failure
        or
        strict_warning_failure
    ):

        print()

        print(
            "FINAL RESULT: FAIL"
        )

        raise SystemExit(
            1
        )

    print()

    print(
        "FINAL RESULT: PASS"
    )

    raise SystemExit(
        0
    )


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":

    main()
