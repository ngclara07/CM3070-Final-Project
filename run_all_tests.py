"""
run_all_tests.py

SenseFuzeAI
Canonical Project-Wide Automated Test Runner

=============================================================================
PURPOSE
=============================================================================

This script performs the principal automated software-verification pass for
the current SenseFuzeAI architecture.

Canonical multimodal processing:

    Keystroke ───────┐
    Text ────────────┤
    Audio ───────────┼──> FinalMultimodalInference
    Image ───────────┘              |
                                    |
                                    | raw four-class
                                    | probability vector
                                    v
                           TemporalFusionEngine
                                    |
                      +-------------+-------------+
                      |             |             |
                      v             v             v
                 Desktop GUI    Web backend   Evaluation


Continuous browser-audio architecture:

    Browser microphone
            |
            v
      Web Audio API
            |
            | Float32 samples
            v
      client-side resampling
            |
            | mono 16-kHz PCM16
            v
         WebSocket
            |
            v
    FastAPI per-session
    rolling PCM16 buffer
            |
            | immutable WAV snapshot
            v
    FinalMultimodalInference


IMPORTANT TEMPORAL INVARIANT
=============================================================================

Starting microphone
    -> temporal RESET once

Incoming PCM packet
    -> NO temporal reset

Incoming PCM packet
    -> NO temporal reset

Prediction
    -> append temporal observation

More PCM packets
    -> NO temporal reset

Prediction
    -> append temporal observation

Stopping microphone
    -> temporal RESET once


Ordinary streamed PCM packets MUST NOT reset temporal fusion. Otherwise the
five-observation temporal window could never fill.


DESIGN PRINCIPLE
=============================================================================

Heavy pretrained models are NOT loaded by default.

Normal verification:

    python run_all_tests.py

Verbose verification:

    python run_all_tests.py --verbose

Strict verification:

    python run_all_tests.py --require-pytest --strict --verbose

Heavy model dependency verification:

    python run_all_tests.py --with-model-smoke


OUTPUT
=============================================================================

Default JSON report:

    data/processed/test_results/run_all_tests_summary.json

Exit codes:

    0   all mandatory checks passed
    1   one or more mandatory checks failed
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

WEB_HTML_FILE = (
    ROOT_DIR
    / "web_app"
    / "templates"
    / "index.html"
)

WEB_SCRIPT_FILE = (
    ROOT_DIR
    / "web_app"
    / "static"
    / "script.js"
)

WEB_STYLE_FILE = (
    ROOT_DIR
    / "web_app"
    / "static"
    / "style.css"
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


# -----------------------------------------------------------------------------
# Continuous microphone expectations
# -----------------------------------------------------------------------------

EXPECTED_AUDIO_SAMPLE_RATE = 16000

EXPECTED_AUDIO_STREAM_WINDOW_SECONDS = 10.0

EXPECTED_AUDIO_STREAM_MIN_SECONDS = 2.0

EXPECTED_AUDIO_STREAM_TRANSPORT = (
    "websocket_pcm16_mono"
)

EXPECTED_AUDIO_SOURCE_POLICY = (
    "fixed_file_or_continuous_microphone_stream"
)


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
    Used for non-fatal verification warnings.
    """


# =============================================================================
# TEST RUNNER
# =============================================================================

class TestRunner:
    """
    Lightweight project verification orchestrator.
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
            detail = str(
                exc
            )

        except WarningCheck as exc:

            status = "WARN"
            detail = str(
                exc
            )

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
            f"{category:21s} "
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
# ASSERTION HELPERS
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
            f"Expected:  {expected_value:.15f}\n"
            f"Actual:    {actual_value:.15f}\n"
            f"Tolerance: {tolerance}"
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
            "Probability dictionaries do not "
            "have identical class ordering."
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
            "Required project file not found:\n"
            f"{path}"
        )

    if not path.is_file():

        raise ValueError(
            "Expected a file:\n"
            f"{path}"
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
    tree: ast.AST,
    module_name: str,
) -> set[str]:

    names: set[str] = set()

    for node in ast.walk(
        tree
    ):

        if not isinstance(
            node,
            ast.ImportFrom,
        ):

            continue

        if (
            node.module
            != module_name
        ):

            continue

        names.update(
            alias.name
            for alias
            in node.names
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
        "Required class not found: "
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
        "Required method not found: "
        f"{class_node.name}.{method_name}"
    )


def top_level_function(
    tree: ast.Module,
    function_name: str,
) -> ast.FunctionDef | ast.AsyncFunctionDef:

    for node in tree.body:

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
            == function_name
        ):

            return node

    raise AssertionError(
        "Required function not found: "
        f"{function_name}"
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

                    value_node = (
                        node.value
                    )

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

            value_node = (
                node.value
            )

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
        "Top-level constant not found: "
        f"{variable_name}"
    )


def static_ast_value(
    node: ast.AST,
) -> Any:
    """
    Evaluate simple compile-time AST values.

    This deliberately supports Python source formatting such as:

        (
            "fixed_file_or_"
            "continuous_microphone_stream"
        )

    because the AST represents adjacent literals as their semantic
    concatenated value.

    It also supports explicit literal string addition.
    """

    if isinstance(
        node,
        ast.Constant,
    ):

        return node.value

    if isinstance(
        node,
        ast.Tuple,
    ):

        return tuple(
            static_ast_value(
                item
            )
            for item
            in node.elts
        )

    if isinstance(
        node,
        ast.List,
    ):

        return [
            static_ast_value(
                item
            )
            for item
            in node.elts
        ]

    if isinstance(
        node,
        ast.Set,
    ):

        return {
            static_ast_value(
                item
            )
            for item
            in node.elts
        }

    if isinstance(
        node,
        ast.Dict,
    ):

        return {
            static_ast_value(
                key
            ):
            static_ast_value(
                value
            )
            for key, value
            in zip(
                node.keys,
                node.values,
            )
            if key is not None
        }

    if (
        isinstance(
            node,
            ast.BinOp,
        )
        and
        isinstance(
            node.op,
            ast.Add,
        )
    ):

        return (
            static_ast_value(
                node.left
            )
            +
            static_ast_value(
                node.right
            )
        )

    if (
        isinstance(
            node,
            ast.UnaryOp,
        )
        and
        isinstance(
            node.op,
            ast.USub,
        )
    ):

        return -static_ast_value(
            node.operand
        )

    raise ValueError(
        "AST expression is not a supported "
        "compile-time literal."
    )


def mapping_literal_value(
    tree: ast.Module,
    *,
    mapping_name: str,
    key: str,
) -> Any:
    """
    Obtain a literal entry from a named dictionary assignment.

    Unlike raw source-string searching, this checks Python semantics.
    Therefore line wrapping and adjacent literal concatenation do not
    produce false failures.
    """

    mapping_node: Optional[
        ast.Dict
    ] = None

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
                    == mapping_name
                ):

                    value_node = (
                        node.value
                    )

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
            == mapping_name
        ):

            value_node = (
                node.value
            )

        if isinstance(
            value_node,
            ast.Dict,
        ):

            mapping_node = (
                value_node
            )

            break

    if mapping_node is None:

        raise AssertionError(
            "Required mapping not found: "
            f"{mapping_name}"
        )

    for key_node, value_node in zip(
        mapping_node.keys,
        mapping_node.values,
    ):

        if key_node is None:

            continue

        try:

            candidate_key = (
                static_ast_value(
                    key_node
                )
            )

        except Exception:

            continue

        if candidate_key != key:

            continue

        try:

            return static_ast_value(
                value_node
            )

        except Exception as exc:

            raise AssertionError(
                f"{mapping_name}[{key!r}] "
                "is not represented by a "
                "static literal."
            ) from exc

    raise AssertionError(
        f"{mapping_name} does not contain "
        f"required key {key!r}."
    )


def class_fields(
    class_node: ast.ClassDef,
) -> set[str]:

    fields: set[str] = set()

    for node in class_node.body:

        if (
            isinstance(
                node,
                ast.AnnAssign,
            )
            and
            isinstance(
                node.target,
                ast.Name,
            )
        ):

            fields.add(
                node.target.id
            )

        elif isinstance(
            node,
            ast.Assign,
        ):

            for target in node.targets:

                if isinstance(
                    target,
                    ast.Name,
                ):

                    fields.add(
                        target.id
                    )

    return fields


def collect_fastapi_routes(
    tree: ast.Module,
) -> set[
    tuple[str, str]
]:

    routes: set[
        tuple[str, str]
    ] = set()

    supported = {
        "app.get":
            "GET",

        "app.post":
            "POST",

        "app.put":
            "PUT",

        "app.delete":
            "DELETE",

        "app.patch":
            "PATCH",

        "app.websocket":
            "WEBSOCKET",
    }

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

            decorator_name = (
                dotted_name(
                    decorator.func
                )
            )

            if (
                decorator_name
                not in supported
            ):

                continue

            if not decorator.args:

                continue

            route_node = (
                decorator.args[
                    0
                ]
            )

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

            routes.add(
                (
                    supported[
                        decorator_name
                    ],
                    route_node.value,
                )
            )

    return routes


def find_call_keywords(
    tree_or_node: ast.AST,
    dotted_callable_name: str,
) -> list[
    set[str]
]:

    matches: list[
        set[str]
    ] = []

    for node in ast.walk(
        tree_or_node
    ):

        if not isinstance(
            node,
            ast.Call,
        ):

            continue

        name = (
            dotted_name(
                node.func
            )
        )

        if (
            name
            != dotted_callable_name
        ):

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
    tree_or_node: ast.AST,
    function_name: str,
) -> bool:

    for node in ast.walk(
        tree_or_node
    ):

        if not isinstance(
            node,
            ast.Call,
        ):

            continue

        name = (
            dotted_name(
                node.func
            )
        )

        if (
            name
            == function_name
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


def ast_contains_string_literal(
    node: ast.AST,
    expected: str,
) -> bool:

    for candidate in ast.walk(
        node
    ):

        if (
            isinstance(
                candidate,
                ast.Constant,
            )
            and
            isinstance(
                candidate.value,
                str,
            )
            and
            candidate.value
            == expected
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

    module = (
        import_temporal_module()
    )

    assert_equal(
        tuple(
            module.LABELS
        ),
        EXPECTED_LABELS,
        (
            "Canonical behavioural "
            "classes changed."
        ),
    )

    assert_equal(
        module.TEMPORAL_PROBABILITY_WINDOW,
        EXPECTED_TEMPORAL_WINDOW,
        (
            "Canonical temporal "
            "window changed."
        ),
    )

    assert_close(
        module.CONFIDENCE_HIGH_GAP,
        EXPECTED_HIGH_GAP,
        message=(
            "High-confidence "
            "threshold changed."
        ),
    )

    assert_close(
        module.CONFIDENCE_MEDIUM_GAP,
        EXPECTED_MEDIUM_GAP,
        message=(
            "Medium-confidence "
            "threshold changed."
        ),
    )

    return (
        "Canonical labels/window/confidence "
        "thresholds verified."
    )


def test_probability_normalisation() -> str:

    module = (
        import_temporal_module()
    )

    normalised = (
        module
        .normalise_probability_dict(
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

    expected = {
        "focused":
            0.60,

        "distracted":
            0.20,

        "fatigued":
            0.10,

        "overloaded":
            0.10,
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
        "Positive probability "
        "normalisation verified."
    )


def test_invalid_probability_policy() -> str:

    module = (
        import_temporal_module()
    )

    result = (
        module
        .normalise_probability_dict(
            {
                "focused":
                    float("nan"),

                "distracted":
                    float("inf"),

                "fatigued":
                    -5.0,

                "overloaded":
                    2.0,
            }
        )
    )

    expected = {
        "focused":
            0.0,

        "distracted":
            0.0,

        "fatigued":
            0.0,

        "overloaded":
            1.0,
    }

    assert_probability_dict_close(
        result,
        expected,
    )

    zero_total = (
        module
        .normalise_probability_dict(
            {
                label:
                    0.0
                for label
                in EXPECTED_LABELS
            }
        )
    )

    uniform = {
        label:
            0.25
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

    module = (
        import_temporal_module()
    )

    cases = [
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

    for gap, expected in cases:

        actual = (
            module.confidence_level(
                gap
            )
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

    module = (
        import_temporal_module()
    )

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
        "focused":
            0.64,

        "distracted":
            0.16,

        "fatigued":
            0.10,

        "overloaded":
            0.10,
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
        (
            "Expected High confidence."
        ),
    )

    assert_equal(
        result[
            "temporal_samples"
        ],
        5,
        (
            "Temporal sample count "
            "is incorrect."
        ),
    )

    assert_true(
        bool(
            result[
                "temporal_window_full"
            ]
        ),
        (
            "Five observations should "
            "fill the temporal window."
        ),
    )

    return (
        "Equal arithmetic five-observation "
        "probability averaging verified."
    )


def test_temporal_window_rollover() -> str:

    module = (
        import_temporal_module()
    )

    engine = (
        module.TemporalFusionEngine()
    )

    one_hot = {
        "focused": {
            "focused": 1.0,
            "distracted": 0.0,
            "fatigued": 0.0,
            "overloaded": 0.0,
        },

        "distracted": {
            "focused": 0.0,
            "distracted": 1.0,
            "fatigued": 0.0,
            "overloaded": 0.0,
        },

        "fatigued": {
            "focused": 0.0,
            "distracted": 0.0,
            "fatigued": 1.0,
            "overloaded": 0.0,
        },

        "overloaded": {
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

        result = (
            engine.append(
                one_hot[
                    label
                ]
            )
        )

    assert_true(
        result is not None,
        (
            "No temporal result produced."
        ),
    )

    assert_equal(
        engine.sample_count,
        5,
        (
            "Temporal history exceeded "
            "maximum window size."
        ),
    )

    expected = {
        "focused":
            0.20,

        "distracted":
            0.20,

        "fatigued":
            0.20,

        "overloaded":
            0.40,
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
            "Window rollover did not "
            "discard the oldest observation."
        ),
    )

    return (
        "Rolling maxlen=5 "
        "behaviour verified."
    )


def test_reset_and_stale_generation() -> str:

    module = (
        import_temporal_module()
    )

    engine = (
        module.TemporalFusionEngine()
    )

    generation_zero = (
        engine.capture_generation()
    )

    assert_equal(
        generation_zero,
        0,
        (
            "Initial generation "
            "should be zero."
        ),
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
        (
            "Observation was not appended."
        ),
    )

    generation_one = (
        engine.reset()
    )

    assert_equal(
        generation_one,
        1,
        (
            "Reset did not increment "
            "generation."
        ),
    )

    assert_equal(
        engine.sample_count,
        0,
        (
            "Reset did not clear history."
        ),
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
            "A stale observation was "
            "not rejected."
        ),
    )

    assert_equal(
        engine.sample_count,
        0,
        (
            "Rejected stale result entered "
            "temporal history."
        ),
    )

    return (
        "Generation reset and stale-result "
        "rejection verified."
    )


def test_temporal_consumer_parity() -> str:

    module = (
        import_temporal_module()
    )

    first_engine = (
        module.TemporalFusionEngine()
    )

    second_engine = (
        module.TemporalFusionEngine()
    )

    generator = (
        random.Random(
            42
        )
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

            first_generation = (
                first_engine.reset()
            )

            second_generation = (
                second_engine.reset()
            )

            assert_equal(
                first_generation,
                second_generation,
                (
                    "Consumer generations diverged."
                ),
            )

        raw = {
            label:
                generator.random()
            for label
            in EXPECTED_LABELS
        }

        first_result = (
            first_engine.append(
                raw,
                expected_generation=(
                    first_engine
                    .capture_generation()
                ),
            )
        )

        second_result = (
            second_engine.append(
                raw,
                expected_generation=(
                    second_engine
                    .capture_generation()
                ),
            )
        )

        assert_equal(
            first_result[
                "current_state"
            ],
            second_result[
                "current_state"
            ],
            (
                "Temporal labels diverged."
            ),
        )

        assert_probability_dict_close(
            first_result[
                "probabilities"
            ],
            second_result[
                "probabilities"
            ],
            tolerance=1e-15,
        )

    return (
        "40 deterministic observations "
        "produced exact consumer parity."
    )


def test_probability_validator() -> str:

    module = (
        import_temporal_module()
    )

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
        (
            "Valid distribution "
            "was rejected."
        ),
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
        (
            "Invalid probability "
            "distribution was accepted."
        ),
    )

    return (
        "Probability-distribution "
        "diagnostics verified."
    )


# =============================================================================
# SOURCE CONTRACT HELPERS
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

    functions = (
        top_level_function_names(
            tree
        )
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


# =============================================================================
# FINAL INFERENCE CONTRACT
# =============================================================================

def test_final_inference_source_contract() -> str:

    tree = (
        parse_python_ast(
            FINAL_INFERENCE_FILE
        )
    )

    imported = (
        imported_names_from(
            tree,
            "temporal_fusion",
        )
    )

    required_imports = {
        "LABELS",
        "normalise_probability_dict",
        "summarise_probability_dict",
        "validate_probability_distribution",
    }

    missing = (
        required_imports
        - imported
    )

    assert_true(
        not missing,
        (
            "final_multimodal_inference.py "
            "is missing canonical imports:\n"
            f"{sorted(missing)}"
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
            "FinalMultimodalInference must "
            "remain temporally stateless."
        ),
    )

    inference_class = (
        find_class(
            tree,
            "FinalMultimodalInference",
        )
    )

    predict_method = (
        class_method(
            inference_class,
            "predict",
        )
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
            "Canonical inference methods "
            "are missing:\n"
            f"{sorted(missing_methods)}"
        ),
    )

    return (
        "Stateless raw-inference API and "
        "probability contract verified."
    )


# =============================================================================
# DESKTOP CONTRACT
# =============================================================================

def test_live_gui_source_contract() -> str:

    tree = (
        parse_python_ast(
            LIVE_GUI_FILE
        )
    )

    imported = (
        imported_names_from(
            tree,
            "temporal_fusion",
        )
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

    assert_equal(
        top_level_literal(
            tree,
            "LIVE_FUSION_INTERVAL_MS",
        ),
        EXPECTED_LIVE_INTERVAL_MS,
        (
            "Desktop prediction cadence changed."
        ),
    )

    assert_equal(
        top_level_literal(
            tree,
            "MIN_TEXT_CHARS",
        ),
        EXPECTED_MIN_TEXT_CHARS,
        (
            "Desktop minimum text "
            "threshold changed."
        ),
    )

    assert_equal(
        top_level_literal(
            tree,
            "MIN_KEYDOWNS",
        ),
        EXPECTED_MIN_KEYDOWNS,
        (
            "Desktop minimum key-down "
            "threshold changed."
        ),
    )

    app_class = (
        find_class(
            tree,
            "FusionDemoApp",
        )
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

    predictor_calls = (
        find_call_keywords(
            tree,
            "self.predictor.predict",
        )
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
            "Desktop predictor call does not "
            "supply all four modalities."
        ),
    )

    return (
        "Desktop shared temporal engine, "
        "cadence and inference contract verified."
    )


# =============================================================================
# WEB BACKEND CONTRACT
# =============================================================================

def test_web_backend_source_contract() -> str:

    tree = (
        parse_python_ast(
            WEB_APP_FILE
        )
    )

    imported = (
        imported_names_from(
            tree,
            "temporal_fusion",
        )
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
            "web_app/app.py is missing "
            "canonical temporal imports:\n"
            f"{sorted(missing)}"
        ),
    )

    assert_no_local_temporal_math(
        tree,
        filename="web_app/app.py",
    )

    session_class = (
        find_class(
            tree,
            "SessionState",
        )
    )

    fields = (
        class_fields(
            session_class
        )
    )

    required_fields = {
        "temporal_fusion",

        "audio_path",
        "audio_name",
        "audio_source_kind",
        "audio_diagnostics",

        "audio_stream_active",
        "audio_stream_token",
        "audio_pcm_buffer",
        "audio_stream_packets",
        "audio_stream_last_packet_at",

        "visual_mode",
        "visual_path",
        "visual_name",
        "visual_started_at",
    }

    missing_fields = (
        required_fields
        - fields
    )

    assert_true(
        not missing_fields,
        (
            "SessionState is missing "
            "required fields:\n"
            f"{sorted(missing_fields)}"
        ),
    )

    routes = (
        collect_fastapi_routes(
            tree
        )
    )

    required_routes = {
        (
            "GET",
            "/health",
        ),
        (
            "GET",
            "/model-status",
        ),
        (
            "POST",
            "/set_audio_source",
        ),
        (
            "POST",
            "/audio_stream/start",
        ),
        (
            "POST",
            "/audio_stream/stop",
        ),
        (
            "WEBSOCKET",
            "/ws/audio/{session_id}",
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
    }

    missing_routes = (
        required_routes
        - routes
    )

    assert_true(
        not missing_routes,
        (
            "Web backend is missing "
            "required routes:\n"
            f"{sorted(missing_routes)}"
        ),
    )

    predictor_calls = (
        find_call_keywords(
            tree,
            "predictor.predict",
        )
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
            "Web predictor invocation does "
            "not supply all four modalities."
        ),
    )

    return (
        "Per-session temporal engine, "
        "continuous audio state, routes and "
        "canonical predictor contract verified."
    )


# =============================================================================
# CONTINUOUS AUDIO CONTRACTS
# =============================================================================

def test_web_streaming_constants() -> str:

    tree = (
        parse_python_ast(
            WEB_APP_FILE
        )
    )

    target_sr = (
        top_level_literal(
            tree,
            "TARGET_SR",
        )
    )

    stream_window = (
        top_level_literal(
            tree,
            "AUDIO_STREAM_WINDOW_SECONDS",
        )
    )

    stream_minimum = (
        top_level_literal(
            tree,
            "AUDIO_STREAM_MIN_SECONDS",
        )
    )

    assert_equal(
        target_sr,
        EXPECTED_AUDIO_SAMPLE_RATE,
        (
            "Continuous microphone target "
            "sampling rate changed."
        ),
    )

    assert_close(
        stream_window,
        EXPECTED_AUDIO_STREAM_WINDOW_SECONDS,
        message=(
            "Continuous microphone rolling "
            "window duration changed."
        ),
    )

    assert_close(
        stream_minimum,
        EXPECTED_AUDIO_STREAM_MIN_SECONDS,
        message=(
            "Continuous microphone minimum "
            "warm-up duration changed."
        ),
    )

    expected_bytes = int(
        EXPECTED_AUDIO_STREAM_WINDOW_SECONDS
        * EXPECTED_AUDIO_SAMPLE_RATE
        * 2
    )

    assert_equal(
        expected_bytes,
        320_000,
        (
            "Expected ten-second PCM16 "
            "capacity is incorrect."
        ),
    )

    return (
        "16-kHz PCM16, 10-second rolling "
        "window and 2-second warm-up verified."
    )


def test_web_streaming_model_status_contract() -> str:
    """
    Semantic MODEL_STATUS test.

    IMPORTANT:
    Do NOT search the raw source for the complete policy string.

    Python may validly represent:

        (
            "fixed_file_or_"
            "continuous_microphone_stream"
        )

    across separate physical lines. The AST correctly represents the
    resulting runtime value as one string.
    """

    tree = (
        parse_python_ast(
            WEB_APP_FILE
        )
    )

    transport = (
        mapping_literal_value(
            tree,
            mapping_name="MODEL_STATUS",
            key="audio_stream_transport",
        )
    )

    source_policy = (
        mapping_literal_value(
            tree,
            mapping_name="MODEL_STATUS",
            key="audio_source_policy",
        )
    )

    packet_reset_policy = (
        mapping_literal_value(
            tree,
            mapping_name="MODEL_STATUS",
            key="stream_packets_reset_temporal",
        )
    )

    assert_equal(
        transport,
        EXPECTED_AUDIO_STREAM_TRANSPORT,
        (
            "MODEL_STATUS audio stream "
            "transport is incorrect."
        ),
    )

    assert_equal(
        source_policy,
        EXPECTED_AUDIO_SOURCE_POLICY,
        (
            "MODEL_STATUS audio source "
            "policy is incorrect."
        ),
    )

    assert_equal(
        packet_reset_policy,
        False,
        (
            "Ordinary microphone packets "
            "must not reset temporal fusion."
        ),
    )

    return (
        "MODEL_STATUS continuous-audio "
        "contract verified semantically via AST."
    )


def test_stream_packets_do_not_reset_temporal_generation() -> str:

    tree = (
        parse_python_ast(
            WEB_APP_FILE
        )
    )

    source = (
        read_source(
            WEB_APP_FILE
        )
    )

    websocket_function = (
        top_level_function(
            tree,
            "audio_stream_socket",
        )
    )

    websocket_source = (
        ast.get_source_segment(
            source,
            websocket_function,
        )
        or ""
    )

    assert_true(
        "audio_pcm_buffer.extend"
        in websocket_source,
        (
            "Audio WebSocket handler does "
            "not append incoming PCM data."
        ),
    )

    assert_true(
        "audio_stream_packets"
        in websocket_source,
        (
            "Audio WebSocket handler does "
            "not track packet receipt."
        ),
    )

    assert_true(
        "reset_temporal_for_source_change"
        not in websocket_source,
        (
            "Incoming microphone packets "
            "appear to reset temporal fusion. "
            "This would prevent the temporal "
            "window from filling."
        ),
    )

    return (
        "Incoming PCM packets preserve "
        "the temporal generation."
    )


def test_microphone_source_changes_reset_temporal() -> str:

    tree = (
        parse_python_ast(
            WEB_APP_FILE
        )
    )

    source = (
        read_source(
            WEB_APP_FILE
        )
    )

    start_function = (
        top_level_function(
            tree,
            "start_audio_stream",
        )
    )

    stop_function = (
        top_level_function(
            tree,
            "stop_audio_stream",
        )
    )

    start_source = (
        ast.get_source_segment(
            source,
            start_function,
        )
        or ""
    )

    stop_source = (
        ast.get_source_segment(
            source,
            stop_function,
        )
        or ""
    )

    assert_true(
        "reset_temporal_for_source_change"
        in start_source,
        (
            "Starting microphone must reset "
            "temporal history once."
        ),
    )

    assert_true(
        "reset_temporal_for_source_change"
        in stop_source,
        (
            "Stopping microphone must reset "
            "temporal history once."
        ),
    )

    return (
        "Microphone start/stop generation "
        "semantics verified."
    )


def test_predict_live_supports_stream_audio_snapshot() -> str:
    """
    Verify continuous microphone inference structurally.

    The old runner incorrectly required the final response key
    'audio_stream_buffered_seconds' to be physically present inside
    predict_live().

    That is unnecessarily restrictive. It is valid for predict_live()
    to calculate/pass audio_buffered_seconds and for
    build_prediction_result() to expose the final response field.
    """

    tree = (
        parse_python_ast(
            WEB_APP_FILE
        )
    )

    source = (
        read_source(
            WEB_APP_FILE
        )
    )

    predict_function = (
        top_level_function(
            tree,
            "predict_live",
        )
    )

    predict_source = (
        ast.get_source_segment(
            source,
            predict_function,
        )
        or ""
    )

    core_stream_requirements = [
        "microphone_stream",
        "audio_pcm_buffer",
        "AUDIO_STREAM_MIN_SECONDS",
        "write_pcm16_wav",
    ]

    missing_core = [
        fragment
        for fragment
        in core_stream_requirements
        if fragment
        not in predict_source
    ]

    assert_true(
        not missing_core,
        (
            "predict_live() is missing core "
            "continuous microphone snapshot "
            "behaviour:\n"
            f"{missing_core}"
        ),
    )

    # -------------------------------------------------------------------------
    # Determine whether buffered-audio metadata is correctly propagated.
    #
    # Accepted architectures:
    #
    # A:
    #   predict_live directly writes:
    #       result["audio_stream_buffered_seconds"] = ...
    #
    # B:
    #   predict_live passes:
    #       audio_buffered_seconds=...
    #
    #   and build_prediction_result produces:
    #       "audio_stream_buffered_seconds": ...
    # -------------------------------------------------------------------------

    builder_function = (
        top_level_function(
            tree,
            "build_prediction_result",
        )
    )

    builder_source = (
        ast.get_source_segment(
            source,
            builder_function,
        )
        or ""
    )

    builder_calls = (
        find_call_keywords(
            predict_function,
            "build_prediction_result",
        )
    )

    predict_directly_exposes_metadata = (
        "audio_stream_buffered_seconds"
        in predict_source
    )

    builder_exposes_final_field = (
        "audio_stream_buffered_seconds"
        in builder_source
        or
        ast_contains_string_literal(
            builder_function,
            "audio_stream_buffered_seconds",
        )
    )

    predict_passes_buffer_metadata = (
        any(
            "audio_buffered_seconds"
            in keywords
            or
            "audio_stream_buffered_seconds"
            in keywords
            for keywords
            in builder_calls
        )
    )

    builder_accepts_buffer_metadata = (
        "audio_buffered_seconds"
        in {
            argument.arg
            for argument
            in (
                builder_function.args.args
                +
                builder_function.args.kwonlyargs
            )
        }
        or
        "audio_stream_buffered_seconds"
        in {
            argument.arg
            for argument
            in (
                builder_function.args.args
                +
                builder_function.args.kwonlyargs
            )
        }
    )

    metadata_pipeline_valid = (
        predict_directly_exposes_metadata
        or
        (
            predict_passes_buffer_metadata
            and
            builder_accepts_buffer_metadata
            and
            builder_exposes_final_field
        )
    )

    assert_true(
        metadata_pipeline_valid,
        (
            "Continuous microphone buffering "
            "metadata is not propagated from "
            "predict_live() to the prediction "
            "response."
        ),
    )

    return (
        "predict_live() snapshots continuous PCM16 "
        "audio and propagates buffered-duration "
        "metadata correctly."
    )


# =============================================================================
# EVALUATION CONTRACT
# =============================================================================

def test_evaluation_source_contract() -> str:

    tree = (
        parse_python_ast(
            EVALUATION_FILE
        )
    )

    imported = (
        imported_names_from(
            tree,
            "temporal_fusion",
        )
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
            "Evaluation script does not "
            "use the canonical temporal engine."
        ),
    )

    source = (
        read_source(
            EVALUATION_FILE
        )
    )

    required_terms = {
        "raw",
        "temporal",
        "generation",
        "confusion_matrix",
        "macro_f1",
    }

    missing = {
        term
        for term
        in required_terms
        if term
        not in source
    }

    assert_true(
        not missing,
        (
            "Evaluation script is missing "
            "expected concepts:\n"
            f"{sorted(missing)}"
        ),
    )

    return (
        "Raw-vs-temporal evaluation "
        "uses the canonical engine."
    )


# =============================================================================
# MULTIMODAL COMPARISON CONTRACT
# =============================================================================

def test_comparison_source_contract() -> str:

    tree = (
        parse_python_ast(
            COMPARISON_FILE
        )
    )

    source = (
        read_source(
            COMPARISON_FILE
        )
    )

    imported = (
        imported_names_from(
            tree,
            "temporal_fusion",
        )
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
            "Multimodal comparison script "
            "does not apply temporal fusion."
        ),
    )

    assert_true(
        "StratifiedGroupKFold"
        in source,
        (
            "Comparison experiment does not "
            "use group-aware splitting."
        ),
    )

    assert_true(
        "train_test_split"
        not in source,
        (
            "Legacy row-level train_test_split "
            "is still present."
        ),
    )

    return (
        "Group-aware raw-vs-temporal "
        "comparison contract verified."
    )


# =============================================================================
# KEYSTROKE DATASET COMPARISON CONTRACTS
# =============================================================================

def test_keystroke_dataset_builder_source_contract() -> str:

    tree = (
        parse_python_ast(
            KEYSTROKE_DATASET_BUILDER_FILE
        )
    )

    source = (
        read_source(
            KEYSTROKE_DATASET_BUILDER_FILE
        )
    )

    imported = (
        imported_names_from(
            tree,
            "keystroke_live_gui_emosurv_ieee",
        )
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
            "is missing canonical EmoSurv imports:\n"
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
            "call extract_live_features()."
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
            "Keystroke dataset builder is "
            "missing harmonisation/provenance "
            "concepts:\n"
            f"{sorted(missing_terms)}"
        ),
    )

    return (
        "EmoSurv/SenseFuzeAI feature "
        "harmonisation and provenance verified."
    )


def test_keystroke_dataset_trainer_source_contract() -> str:

    tree = (
        parse_python_ast(
            KEYSTROKE_DATASET_TRAINER_FILE
        )
    )

    source = (
        read_source(
            KEYSTROKE_DATASET_TRAINER_FILE
        )
    )

    assert_true(
        tree is not None,
        (
            "Unable to parse keystroke "
            "dataset comparison trainer."
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
            "missing experiment/split concepts:\n"
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
            "Keystroke dataset comparison "
            "does not appear group-aware."
        ),
    )

    assert_true(
        "bootstrap"
        in source.lower(),
        (
            "Bootstrap uncertainty analysis "
            "is no longer present."
        ),
    )

    assert_true(
        "train_test_split"
        not in source,
        (
            "Row-level train_test_split appears "
            "in the keystroke comparison trainer."
        ),
    )

    return (
        "A-F experiment design, frozen split, "
        "group-aware evaluation and bootstrap "
        "analysis verified."
    )


# =============================================================================
# JAVASCRIPT CONTRACT
# =============================================================================

def test_javascript_source_contract() -> str:

    source = (
        read_source(
            WEB_SCRIPT_FILE
        )
    )

    required_fragments = [
        # Generation-safe client state.
        "serverGeneration",
        "clientEpoch",
        "runLivePrediction",

        # Fixed-file audio fallback.
        "/set_audio_source",
        "setAudioFile",

        # Continuous microphone architecture.
        "startMicrophoneStream",
        "stopMicrophoneStream",
        "/audio_stream/start",
        "/audio_stream/stop",
        "/ws/audio/",
        "new WebSocket",
        "AudioContext",
        "createMediaStreamSource",
        "createScriptProcessor",
        "resampleLinear",
        "float32ToPCM16Buffer",
        "microphoneStreaming",
        "audioBufferedSec",
        "audioPackets",

        # Visual sources.
        "/set_visual_image",
        "/set_visual_video",
        "/set_visual_webcam",
        "/stop_visual",
        "captureWebcamFrame",
        "webcam_frame",

        # Prediction/reset.
        "/predict_live",
        "/reset_temporal",
        "/full_reset",

        # Live prediction cadence.
        "setInterval",

        # Backend status labels.
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
            "integration pieces:\n"
            f"{missing}"
        ),
    )

    suspicious_temporal_fragments = [
        "aggregateProbabilityHistory",
        "aggregate_probability_history",
        "normaliseProbabilityDict",
        "normalizeProbabilityDict",
        "confidenceLevelFromGap",
        "temporalProbabilityHistory",
    ]

    duplicated_temporal_math = [
        fragment
        for fragment
        in suspicious_temporal_fragments
        if fragment
        in source
    ]

    assert_true(
        not duplicated_temporal_math,
        (
            "script.js appears to duplicate "
            "canonical temporal mathematics:\n"
            f"{duplicated_temporal_math}"
        ),
    )

    old_audio_fragments = [
        "recordMicrophoneOnce",
        "AUDIO_CAPTURE_SECONDS",
        "forceExactDuration",
        "MediaRecorder",
    ]

    obsolete_audio = [
        fragment
        for fragment
        in old_audio_fragments
        if fragment
        in source
    ]

    assert_true(
        not obsolete_audio,
        (
            "script.js still contains obsolete "
            "one-shot microphone architecture:\n"
            f"{obsolete_audio}"
        ),
    )

    return (
        "Browser client uses continuous "
        "PCM16 WebSocket acquisition without "
        "duplicating temporal mathematics."
    )


# =============================================================================
# HTML CONTRACT
# =============================================================================

def test_html_streaming_audio_contract() -> str:

    source = (
        read_source(
            WEB_HTML_FILE
        )
        .lower()
    )

    required_audio_ids = [
        'id="startmicbtn"',
        'id="stopmicbtn"',
        'id="chooseaudiobtn"',
        'id="audiofileinput"',
        'id="audiostreamstate"',
        'id="audiobufferedseconds"',
        'id="audiolivelevel"',
        'id="audiopacketcount"',
        'id="audiostatus"',
        'id="audiodiagnostic"',
    ]

    missing_audio = [
        identifier
        for identifier
        in required_audio_ids
        if identifier
        not in source
    ]

    assert_true(
        not missing_audio,
        (
            "index.html is missing continuous "
            "microphone UI elements:\n"
            f"{missing_audio}"
        ),
    )

    required_result_ids = [
        'id="prediction"',
        'id="confidencepercent"',
        'id="confidencelevel"',
        'id="rawprediction"',
        'id="rawconfidence"',
        'id="probabilities"',
        'id="rawprobabilities"',
        'id="temporalsamples"',
        'id="temporalwindow"',
        'id="resettemporalbtn"',
    ]

    missing_results = [
        identifier
        for identifier
        in required_result_ids
        if identifier
        not in source
    ]

    assert_true(
        not missing_results,
        (
            "index.html is missing required "
            "prediction/result elements:\n"
            f"{missing_results}"
        ),
    )

    return (
        "Continuous audio controls and "
        "raw/temporal result UI verified."
    )


# =============================================================================
# CSS CONTRACT
# =============================================================================

def test_css_streaming_audio_contract() -> str:

    source = (
        read_source(
            WEB_STYLE_FILE
        )
    )

    required_fragments = [
        ".audio-card.streaming",
        ".audio-stream-grid",
        ".audio-control-grid",
        ".audio-bars",
        ".source-btn-live",
        ".source-btn-stop",
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
            "style.css is missing continuous "
            "audio presentation styles:\n"
            f"{missing}"
        ),
    )

    return (
        "Continuous microphone visual "
        "states verified in style.css."
    )


# =============================================================================
# PYTHON SYNTAX
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
        "Compiled successfully: "
        f"{path.relative_to(ROOT_DIR)}"
    )


# =============================================================================
# NODE.JS SYNTAX
# =============================================================================

def run_node_syntax_check(
    *,
    require_node: bool,
    timeout_seconds: int,
) -> str:

    require_file(
        WEB_SCRIPT_FILE
    )

    node = (
        shutil.which(
            "node"
        )
    )

    if not node:

        message = (
            "Node.js is not available; "
            "script.js syntax check skipped."
        )

        if require_node:

            raise RuntimeError(
                message
            )

        raise SkipCheck(
            message
        )

    completed = (
        subprocess.run(
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
    )

    if (
        completed.returncode
        != 0
    ):

        raise RuntimeError(
            (
                completed.stderr
                or
                completed.stdout
                or
                "node --check failed."
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

    test_files = sorted(
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

    completed = (
        subprocess.run(
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

    if (
        completed.returncode
        != 0
    ):

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
# OPTIONAL MODEL SMOKE TEST
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

    completed = (
        subprocess.run(
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

    if (
        completed.returncode
        != 0
    ):

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
            "Model smoke subprocess did "
            "not emit completion marker."
        ),
    )

    return output


# =============================================================================
# TEMPORAL SELF-TEST
# =============================================================================

def run_temporal_self_test_subprocess(
    *,
    timeout_seconds: int,
) -> str:

    require_file(
        TEMPORAL_FUSION_FILE
    )

    completed = (
        subprocess.run(
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

    if (
        completed.returncode
        != 0
    ):

        raise RuntimeError(
            (
                "temporal_fusion.py "
                "self-test failed.\n\n"
                + output
            )
        )

    assert_true(
        "self-test: PASS"
        in output,
        (
            "temporal_fusion.py did "
            "not report PASS."
        ),
    )

    return output


# =============================================================================
# JSON REPORT
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

    counts = (
        runner.counts()
    )

    payload = {
        "project":
            "SenseFuzeAI",

        "test_runner":
            "run_all_tests.py",

        "architecture":
            (
                "continuous_multimodal_"
                "temporal_fusion"
            ),

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

            "continuous_audio": {
                "target_sample_rate":
                    EXPECTED_AUDIO_SAMPLE_RATE,

                "pcm_format":
                    "signed_16_bit_little_endian",

                "channels":
                    1,

                "transport":
                    EXPECTED_AUDIO_STREAM_TRANSPORT,

                "rolling_window_seconds":
                    EXPECTED_AUDIO_STREAM_WINDOW_SECONDS,

                "minimum_ready_seconds":
                    EXPECTED_AUDIO_STREAM_MIN_SECONDS,

                "audio_source_policy":
                    EXPECTED_AUDIO_SOURCE_POLICY,

                "stream_packets_reset_temporal":
                    False,

                "microphone_start_resets_temporal":
                    True,

                "microphone_stop_resets_temporal":
                    True,

                "temporal_only_reset_preserves_stream":
                    True,
            },

            "keystroke_dataset_comparison": {
                "builder":
                    KEYSTROKE_DATASET_BUILDER_FILE.name,

                "trainer":
                    KEYSTROKE_DATASET_TRAINER_FILE.name,

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
            "Run canonical SenseFuzeAI continuous "
            "multimodal, temporal-fusion and "
            "dataset-comparison verification."
        )
    )

    parser.add_argument(
        "--with-model-smoke",
        action="store_true",
        help=(
            "Instantiate FinalMultimodalInference "
            "and load all pretrained/model artifacts."
        ),
    )

    parser.add_argument(
        "--require-node",
        action="store_true",
        help=(
            "Treat missing Node.js as a failure."
        ),
    )

    parser.add_argument(
        "--require-pytest",
        action="store_true",
        help=(
            "Treat missing pytest/tests as a failure."
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
        default=300,
        help=(
            "Timeout in seconds for normal "
            "subprocess checks and pytest."
        ),
    )

    parser.add_argument(
        "--model-timeout",
        type=int,
        default=900,
        help=(
            "Timeout in seconds for optional "
            "model initialisation."
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
# MAIN
# =============================================================================

def main() -> None:

    parser = (
        build_argument_parser()
    )

    args = (
        parser.parse_args()
    )

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
        "=" * 96
    )

    print(
        "SenseFuzeAI Canonical Automated Test Suite"
    )

    print(
        "Continuous Multimodal + Temporal Fusion Verification"
    )

    print(
        "=" * 96
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
    # 2. Test-file syntax
    # =========================================================================

    if TESTS_DIR.exists():

        for path in sorted(
            TESTS_DIR.rglob(
                "test_*.py"
            )
        ):

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
                "test-syntax",
                lambda path=path:
                    compile_python_file(
                        path
                    ),
            )


    # =========================================================================
    # 3. Canonical temporal mathematics
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
    # 4. Architecture
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
        "web multimodal source contract",
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


    # =========================================================================
    # 5. Continuous web microphone
    # =========================================================================

    runner.run(
        "streaming audio constants",
        "continuous-audio",
        test_web_streaming_constants,
    )

    runner.run(
        "streaming model-status contract",
        "continuous-audio",
        test_web_streaming_model_status_contract,
    )

    runner.run(
        "PCM packets preserve temporal generation",
        "continuous-audio",
        test_stream_packets_do_not_reset_temporal_generation,
    )

    runner.run(
        "microphone start/stop reset semantics",
        "continuous-audio",
        test_microphone_source_changes_reset_temporal,
    )

    runner.run(
        "predict_live streamed audio snapshot",
        "continuous-audio",
        test_predict_live_supports_stream_audio_snapshot,
    )

    runner.run(
        "JavaScript continuous microphone contract",
        "continuous-audio",
        test_javascript_source_contract,
    )

    runner.run(
        "HTML continuous microphone controls",
        "continuous-audio",
        test_html_streaming_audio_contract,
    )

    runner.run(
        "CSS continuous microphone states",
        "continuous-audio",
        test_css_streaming_audio_contract,
    )


    # =========================================================================
    # 6. Keystroke dataset comparison
    # =========================================================================

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


    # =========================================================================
    # 7. JavaScript syntax
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
    # 8. Complete pytest suite
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
    # 9. Optional heavyweight model smoke test
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
    # 10. Report / summary
    # =========================================================================

    counts = (
        runner.counts()
    )

    save_report(
        path=(
            args.report
        ),
        runner=runner,
        arguments=args,
    )


    print()

    print(
        "=" * 96
    )

    print(
        "TEST SUMMARY"
    )

    print(
        "=" * 96
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
