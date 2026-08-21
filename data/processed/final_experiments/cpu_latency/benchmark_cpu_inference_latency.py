"""
benchmark_cpu_inference_latency.py

SenseFuzeAI
CPU-Only Inference Latency Benchmark
Section 5.7

Purpose
-------
Measure computational responsiveness of the current SenseFuzeAI production
inference implementation without confusing acquisition duration with
processing latency.

Measured stages
---------------
1. Cold application/model initialisation
2. Keystroke processing
3. Text feature extraction
4. Audio feature extraction
5. Visual feature extraction
6. Schema reconstruction + fusion/classifier
7. Warm end-to-end processing

Important methodological rules
------------------------------
- CPU only.
- No model-download time.
- No microphone recording duration.
- No waiting for user typing.
- No webcam acquisition waiting.
- No deliberate temporal collection window.
- Warm end-to-end is measured independently.
- Component timings are NOT summed to create end-to-end latency.
- Multiple representative sessions are cycled across warm measurements.
- Cold-start runs use fresh Python processes.

Environment information
-----------------------
CPU model, logical/physical CPU counts, RAM, operating system, Python
version and relevant library versions are persisted in the benchmark
metadata.

This implementation deliberately has NO psutil dependency.

Outputs
-------
data/processed/final_experiments/cpu_latency/

    latency_results.csv
    latency_raw_runs.csv
    latency_environment.json
    benchmark_cpu_inference_latency.py

Run
---
Syntax validation:

    python -m py_compile benchmark_cpu_inference_latency.py

Short diagnostic run:

    python benchmark_cpu_inference_latency.py --cold-repetitions 1 --warm-repetitions 3 --warmups 1 --max-sessions 4

Final report-facing run:

    python benchmark_cpu_inference_latency.py
"""

from __future__ import annotations

# =============================================================================
# STANDARD LIBRARY IMPORTS
# =============================================================================

import argparse
import gc
import hashlib
import importlib
import importlib.metadata
import json
import math
import os
import platform
import shutil
import subprocess
import sys
import time

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Optional, Sequence


# =============================================================================
# FORCE CPU / OFFLINE MODEL USE BEFORE IMPORTING ML CODE
# =============================================================================

# Hide CUDA devices before importing production code that may import PyTorch.
os.environ["CUDA_VISIBLE_DEVICES"] = ""

# Avoid tokenizer worker warnings/non-deterministic worker setup.
os.environ.setdefault(
    "TOKENIZERS_PARALLELISM",
    "false",
)

# The benchmark must not silently include model-download time.
os.environ.setdefault(
    "HF_HUB_OFFLINE",
    "1",
)

os.environ.setdefault(
    "TRANSFORMERS_OFFLINE",
    "1",
)


# =============================================================================
# THIRD-PARTY IMPORTS
# =============================================================================

import numpy as np
import pandas as pd


# =============================================================================
# PROJECT PATHS
# =============================================================================

ROOT_DIR = (
    Path(__file__)
    .resolve()
    .parent
)

FINAL_INFERENCE_PATH = (
    ROOT_DIR
    / "final_multimodal_inference.py"
)

WEB_APP_PATH = (
    ROOT_DIR
    / "web_app"
    / "app.py"
)

SAMPLE_DATA_DIR = (
    ROOT_DIR
    / "sample_data"
)

SESSION_ALIGNED_DIR = (
    ROOT_DIR
    / "data"
    / "session_aligned"
)

FUSION_DIR = (
    ROOT_DIR
    / "models"
    / "fusion_demo"
)

FUSION_DATASET_PATH = (
    FUSION_DIR
    / "fusion_training_dataset.csv"
)

FEATURE_SCHEMA_PATH = (
    FUSION_DIR
    / "feature_columns.json"
)

FUSION_MODEL_PATH = (
    FUSION_DIR
    / "fusion_pipeline.joblib"
)

DEFAULT_OUTPUT_DIR = (
    ROOT_DIR
    / "data"
    / "processed"
    / "final_experiments"
    / "cpu_latency"
)


# =============================================================================
# BENCHMARK PROTOCOL
# =============================================================================

EXPECTED_PYTHON_VERSION = "3.11.9"

EXPECTED_FEATURE_COUNT = 2373

DEFAULT_COLD_REPETITIONS = 5

DEFAULT_WARM_REPETITIONS = 30

DEFAULT_WARMUPS = 3

DEFAULT_MAX_SESSIONS = 20

MINIMUM_REPRESENTATIVE_SESSIONS = 4

# Current SenseFuzeAI live prediction cadence.
OPERATIONAL_UPDATE_CADENCE_MS = 2500.0

# NumPy population SD.
SD_DDOF = 0

COLD_CHILD_SENTINEL = "SENSEFUZE_COLD_READY="

STAGE_COLD = (
    "Cold application/model initialisation"
)

STAGE_KEYSTROKE = (
    "Keystroke processing"
)

STAGE_TEXT = (
    "Text feature extraction"
)

STAGE_AUDIO = (
    "Audio feature extraction"
)

STAGE_VISUAL = (
    "Visual feature extraction"
)

STAGE_FUSION = (
    "Fusion/classifier"
)

STAGE_END_TO_END = (
    "Warm end-to-end processing"
)

STAGE_ORDER = [
    STAGE_COLD,
    STAGE_KEYSTROKE,
    STAGE_TEXT,
    STAGE_AUDIO,
    STAGE_VISUAL,
    STAGE_FUSION,
    STAGE_END_TO_END,
]

SUPPORTED_AUDIO_EXTENSIONS = {
    ".wav",
    ".mp3",
    ".m4a",
    ".flac",
    ".ogg",
    ".aac",
}

SUPPORTED_IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".webp",
    ".bmp",
}

CANONICAL_CLASSES = [
    "focused",
    "distracted",
    "fatigued",
    "overloaded",
]


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass(frozen=True)
class BenchmarkSample:
    """
    One already-acquired complete multimodal session.
    """

    session_id: str
    label: str
    text: str
    text_path: Path
    keystroke_path: Path
    audio_path: Path
    image_path: Path


@dataclass(frozen=True)
class FusionBenchmarkSample:
    """
    One persisted 2,373-feature row used only for isolated fusion timing.
    """

    session_id: str
    label: str
    features: dict[str, float]


# =============================================================================
# GENERAL UTILITIES
# =============================================================================

def print_heading(
    title: str,
) -> None:
    print()
    print("=" * 104)
    print(title)
    print("=" * 104)


def require_file(
    path: Path,
    description: str,
) -> None:
    if not path.exists():
        raise FileNotFoundError(
            f"{description} not found:\n{path}"
        )

    if not path.is_file():
        raise ValueError(
            f"{description} is not a file:\n{path}"
        )


def normalise_label(
    value: Any,
) -> str:
    if value is None:
        return ""

    return (
        str(value)
        .strip()
        .lower()
    )


def current_python_version() -> str:
    return (
        f"{sys.version_info.major}."
        f"{sys.version_info.minor}."
        f"{sys.version_info.micro}"
    )


def sha256_file(
    path: Path,
) -> str:
    digest = hashlib.sha256()

    with path.open(
        "rb"
    ) as file_handle:
        while True:
            chunk = file_handle.read(
                1024 * 1024
            )

            if not chunk:
                break

            digest.update(
                chunk
            )

    return digest.hexdigest()


def safe_json_value(
    value: Any,
) -> Any:
    """
    Convert common Python/NumPy values to JSON-safe representations.
    """

    if isinstance(
        value,
        np.generic,
    ):
        return safe_json_value(
            value.item()
        )

    if isinstance(
        value,
        np.ndarray,
    ):
        return [
            safe_json_value(item)
            for item in value.tolist()
        ]

    if value is None:
        return None

    if isinstance(
        value,
        (
            str,
            int,
            bool,
        ),
    ):
        return value

    if isinstance(
        value,
        float,
    ):
        if math.isfinite(value):
            return value

        return str(value)

    if isinstance(
        value,
        Path,
    ):
        return str(value)

    if isinstance(
        value,
        dict,
    ):
        return {
            str(key): safe_json_value(child)
            for key, child in value.items()
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
            safe_json_value(child)
            for child in value
        ]

    return repr(value)


def write_json(
    path: Path,
    payload: dict[str, Any],
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open(
        "w",
        encoding="utf-8",
    ) as file_handle:
        json.dump(
            safe_json_value(payload),
            file_handle,
            indent=2,
            ensure_ascii=False,
        )


def package_version(
    distribution_name: str,
) -> Optional[str]:
    try:
        return importlib.metadata.version(
            distribution_name
        )

    except Exception:
        return None


# =============================================================================
# SAFE SUBPROCESS UTILITY
# =============================================================================

def run_short_command(
    command: Sequence[str],
    *,
    timeout_seconds: int = 10,
) -> Optional[str]:
    """
    Execute a small operating-system information command.

    Returns stripped stdout on success, otherwise None.
    """

    try:
        completed = subprocess.run(
            list(command),
            cwd=ROOT_DIR,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            check=False,
        )

        if completed.returncode != 0:
            return None

        output = (
            completed.stdout
            or ""
        ).strip()

        if not output:
            return None

        return output

    except Exception:
        return None


# =============================================================================
# CPU / MEMORY ENVIRONMENT
#
# IMPORTANT:
# No psutil import is used anywhere in this script.
# =============================================================================

def get_windows_powershell_value(
    expression: str,
) -> Optional[str]:
    """
    Return a single PowerShell value when running on Windows.
    """

    if platform.system().lower() != "windows":
        return None

    return run_short_command(
        [
            "powershell",
            "-NoProfile",
            "-NonInteractive",
            "-Command",
            expression,
        ]
    )


def get_cpu_model() -> str:
    """
    Return the most specific CPU model available without external packages.
    """

    system_name = platform.system().lower()

    # -------------------------------------------------------------------------
    # Windows
    # -------------------------------------------------------------------------

    if system_name == "windows":
        result = get_windows_powershell_value(
            (
                "Get-CimInstance Win32_Processor | "
                "Select-Object -First 1 -ExpandProperty Name"
            )
        )

        if result:
            return result.strip()

        processor_identifier = (
            os.environ.get(
                "PROCESSOR_IDENTIFIER",
                ""
            )
            .strip()
        )

        if processor_identifier:
            return processor_identifier

    # -------------------------------------------------------------------------
    # Linux
    # -------------------------------------------------------------------------

    if system_name == "linux":
        cpuinfo_path = Path(
            "/proc/cpuinfo"
        )

        if cpuinfo_path.exists():
            try:
                for line in cpuinfo_path.read_text(
                    encoding="utf-8",
                    errors="ignore",
                ).splitlines():
                    if line.lower().startswith(
                        "model name"
                    ):
                        _, value = line.split(
                            ":",
                            1,
                        )

                        value = value.strip()

                        if value:
                            return value

            except Exception:
                pass

    # -------------------------------------------------------------------------
    # macOS
    # -------------------------------------------------------------------------

    if system_name == "darwin":
        result = run_short_command(
            [
                "sysctl",
                "-n",
                "machdep.cpu.brand_string",
            ]
        )

        if result:
            return result.strip()

    # -------------------------------------------------------------------------
    # Generic fallback
    # -------------------------------------------------------------------------

    processor = (
        platform.processor()
        or ""
    ).strip()

    if processor:
        return processor

    return "unavailable"


def get_physical_cpu_count() -> Optional[int]:
    """
    Return physical core count without using psutil.

    Windows:
        Win32_Processor.NumberOfCores

    Linux:
        lscpu physical core/socket pairs where available

    macOS:
        sysctl hw.physicalcpu
    """

    system_name = platform.system().lower()

    # -------------------------------------------------------------------------
    # Windows
    # -------------------------------------------------------------------------

    if system_name == "windows":
        result = get_windows_powershell_value(
            (
                "(Get-CimInstance Win32_Processor | "
                "Measure-Object -Property NumberOfCores -Sum).Sum"
            )
        )

        if result:
            try:
                return int(
                    float(
                        result.strip()
                    )
                )

            except ValueError:
                pass

    # -------------------------------------------------------------------------
    # Linux
    # -------------------------------------------------------------------------

    if system_name == "linux":
        result = run_short_command(
            [
                "lscpu",
                "-p=CORE,SOCKET",
            ]
        )

        if result:
            physical_pairs: set[
                tuple[str, str]
            ] = set()

            for line in result.splitlines():
                line = line.strip()

                if (
                    not line
                    or line.startswith("#")
                ):
                    continue

                parts = [
                    value.strip()
                    for value in line.split(",")
                ]

                if len(parts) >= 2:
                    physical_pairs.add(
                        (
                            parts[0],
                            parts[1],
                        )
                    )

            if physical_pairs:
                return len(
                    physical_pairs
                )

    # -------------------------------------------------------------------------
    # macOS
    # -------------------------------------------------------------------------

    if system_name == "darwin":
        result = run_short_command(
            [
                "sysctl",
                "-n",
                "hw.physicalcpu",
            ]
        )

        if result:
            try:
                return int(
                    result.strip()
                )

            except ValueError:
                pass

    return None


def get_total_ram_bytes() -> Optional[int]:
    """
    Return installed physical memory without using psutil.
    """

    system_name = platform.system().lower()

    # -------------------------------------------------------------------------
    # Windows
    # -------------------------------------------------------------------------

    if system_name == "windows":
        result = get_windows_powershell_value(
            (
                "(Get-CimInstance "
                "Win32_ComputerSystem).TotalPhysicalMemory"
            )
        )

        if result:
            try:
                return int(
                    float(
                        result.strip()
                    )
                )

            except ValueError:
                pass

    # -------------------------------------------------------------------------
    # Linux / other POSIX with sysconf
    # -------------------------------------------------------------------------

    if system_name == "linux":
        try:
            page_size = os.sysconf(
                "SC_PAGE_SIZE"
            )

            physical_pages = os.sysconf(
                "SC_PHYS_PAGES"
            )

            return int(
                page_size
                * physical_pages
            )

        except (
            AttributeError,
            ValueError,
            OSError,
        ):
            pass

    # -------------------------------------------------------------------------
    # macOS
    # -------------------------------------------------------------------------

    if system_name == "darwin":
        result = run_short_command(
            [
                "sysctl",
                "-n",
                "hw.memsize",
            ]
        )

        if result:
            try:
                return int(
                    result.strip()
                )

            except ValueError:
                pass

    return None


def bytes_to_gib(
    value: Optional[int],
) -> Optional[float]:
    if value is None:
        return None

    return float(
        value
        / (
            1024 ** 3
        )
    )


def collect_environment() -> dict[str, Any]:
    total_ram_bytes = (
        get_total_ram_bytes()
    )

    return {
        "captured_at": (
            datetime.now()
            .astimezone()
            .isoformat()
        ),

        "operating_system": (
            platform.platform()
        ),

        "system": (
            platform.system()
        ),

        "release": (
            platform.release()
        ),

        "machine": (
            platform.machine()
        ),

        "cpu_model": (
            get_cpu_model()
        ),

        "logical_cpu_count": (
            os.cpu_count()
        ),

        "physical_cpu_count": (
            get_physical_cpu_count()
        ),

        "total_ram_bytes": (
            total_ram_bytes
        ),

        "total_ram_gib": (
            bytes_to_gib(
                total_ram_bytes
            )
        ),

        "python_version": (
            current_python_version()
        ),

        "python_full": (
            sys.version
        ),

        "python_executable": (
            sys.executable
        ),

        "cpu_only_configuration": {
            "CUDA_VISIBLE_DEVICES": (
                os.environ.get(
                    "CUDA_VISIBLE_DEVICES"
                )
            ),

            "HF_HUB_OFFLINE": (
                os.environ.get(
                    "HF_HUB_OFFLINE"
                )
            ),

            "TRANSFORMERS_OFFLINE": (
                os.environ.get(
                    "TRANSFORMERS_OFFLINE"
                )
            ),

            "required_predictor_device": (
                "cpu"
            ),
        },

        "library_versions": {
            "numpy": (
                package_version(
                    "numpy"
                )
            ),

            "pandas": (
                package_version(
                    "pandas"
                )
            ),

            "scikit-learn": (
                package_version(
                    "scikit-learn"
                )
            ),

            "torch": (
                package_version(
                    "torch"
                )
            ),

            "sentence-transformers": (
                package_version(
                    "sentence-transformers"
                )
            ),

            "transformers": (
                package_version(
                    "transformers"
                )
            ),

            "librosa": (
                package_version(
                    "librosa"
                )
            ),

            "joblib": (
                package_version(
                    "joblib"
                )
            ),

            "fastapi": (
                package_version(
                    "fastapi"
                )
            ),

            "jinja2": (
                package_version(
                    "Jinja2"
                )
            ),

            "uvicorn": (
                package_version(
                    "uvicorn"
                )
            ),
        },
    }


# =============================================================================
# PRODUCTION INFERENCE IMPORT
# =============================================================================

def load_production_inference_class() -> Any:
    require_file(
        FINAL_INFERENCE_PATH,
        "Canonical inference implementation",
    )

    # Ensure the project root can be imported even when this script
    # is invoked using an absolute path.
    root_string = str(
        ROOT_DIR
    )

    if root_string not in sys.path:
        sys.path.insert(
            0,
            root_string,
        )

    module = importlib.import_module(
        "final_multimodal_inference"
    )

    inference_class = getattr(
        module,
        "FinalMultimodalInference",
        None,
    )

    if inference_class is None:
        raise RuntimeError(
            "final_multimodal_inference.py does not expose "
            "FinalMultimodalInference."
        )

    return inference_class


def create_cpu_predictor() -> Any:
    inference_class = (
        load_production_inference_class()
    )

    predictor = inference_class()

    observed_device = (
        str(
            getattr(
                predictor,
                "device",
                "",
            )
        )
        .strip()
        .lower()
    )

    if observed_device != "cpu":
        raise RuntimeError(
            "CPU-only benchmark requirement failed.\n"
            f"Expected predictor device: cpu\n"
            f"Observed predictor device: {observed_device!r}"
        )

    feature_columns = getattr(
        predictor,
        "feature_columns",
        None,
    )

    if not isinstance(
        feature_columns,
        list,
    ):
        raise RuntimeError(
            "Canonical predictor does not expose "
            "feature_columns as a list."
        )

    if len(
        feature_columns
    ) != EXPECTED_FEATURE_COUNT:
        raise RuntimeError(
            "Canonical predictor feature-schema mismatch.\n"
            f"Expected: {EXPECTED_FEATURE_COUNT}\n"
            f"Observed: {len(feature_columns)}"
        )

    return predictor


# =============================================================================
# COLD-START CHILD PROCESS
# =============================================================================

def cold_child_main() -> None:
    """
    Internal fresh-process child.

    Successful completion means FinalMultimodalInference has fully loaded
    and is ready to execute inference.
    """

    predictor = (
        create_cpu_predictor()
    )

    payload = {
        "status": "ready",

        "pid": (
            os.getpid()
        ),

        "device": (
            str(
                predictor.device
            )
        ),

        "feature_count": (
            len(
                predictor.feature_columns
            )
        ),

        "python_version": (
            current_python_version()
        ),
    }

    print(
        COLD_CHILD_SENTINEL
        + json.dumps(
            payload,
            ensure_ascii=False,
        ),
        flush=True,
    )


def extract_cold_child_payload(
    stdout: str,
) -> dict[str, Any]:
    for line in reversed(
        stdout.splitlines()
    ):
        if line.startswith(
            COLD_CHILD_SENTINEL
        ):
            raw_payload = line[
                len(
                    COLD_CHILD_SENTINEL
                ):
            ]

            payload = json.loads(
                raw_payload
            )

            if not isinstance(
                payload,
                dict,
            ):
                raise RuntimeError(
                    "Cold child returned an invalid "
                    "readiness payload."
                )

            return payload

    raise RuntimeError(
        "Cold child did not emit the expected readiness marker.\n\n"
        f"Captured stdout:\n{stdout[-4000:]}"
    )


def benchmark_cold_initialisation(
    *,
    repetitions: int,
    timeout_seconds: int,
) -> list[dict[str, Any]]:
    """
    Measure cold initialisation using fresh Python processes.

    The parent wall-clock timer includes Python process startup, production
    module import and FinalMultimodalInference construction.

    The child exits immediately after signalling readiness, so process
    shutdown contributes only a small conservative overhead.
    """

    rows: list[
        dict[str, Any]
    ] = []

    script_path = (
        Path(__file__)
        .resolve()
    )

    for repetition in range(
        1,
        repetitions + 1,
    ):
        environment = (
            os.environ.copy()
        )

        environment[
            "CUDA_VISIBLE_DEVICES"
        ] = ""

        environment[
            "TOKENIZERS_PARALLELISM"
        ] = "false"

        environment[
            "HF_HUB_OFFLINE"
        ] = "1"

        environment[
            "TRANSFORMERS_OFFLINE"
        ] = "1"

        command = [
            sys.executable,
            str(
                script_path
            ),
            "--cold-child",
        ]

        gc.collect()

        start_ns = (
            time.perf_counter_ns()
        )

        completed = subprocess.run(
            command,
            cwd=ROOT_DIR,
            env=environment,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            check=False,
        )

        end_ns = (
            time.perf_counter_ns()
        )

        if completed.returncode != 0:
            raise RuntimeError(
                "Fresh-process cold initialisation failed.\n\n"
                f"Repetition: {repetition}\n"
                f"Return code: {completed.returncode}\n\n"
                f"STDOUT:\n{completed.stdout[-5000:]}\n\n"
                f"STDERR:\n{completed.stderr[-5000:]}"
            )

        child_payload = (
            extract_cold_child_payload(
                completed.stdout
            )
        )

        child_device = (
            str(
                child_payload.get(
                    "device",
                    "",
                )
            )
            .strip()
            .lower()
        )

        if child_device != "cpu":
            raise RuntimeError(
                "Cold child did not initialise "
                "the predictor on CPU."
            )

        child_feature_count = int(
            child_payload.get(
                "feature_count",
                -1,
            )
        )

        if (
            child_feature_count
            != EXPECTED_FEATURE_COUNT
        ):
            raise RuntimeError(
                "Cold child feature count mismatch.\n"
                f"Expected: {EXPECTED_FEATURE_COUNT}\n"
                f"Observed: {child_feature_count}"
            )

        latency_ns = (
            end_ns
            - start_ns
        )

        latency_ms = (
            latency_ns
            / 1_000_000.0
        )

        rows.append(
            {
                "stage": (
                    STAGE_COLD
                ),

                "run_type": (
                    "cold_fresh_process"
                ),

                "repetition": (
                    repetition
                ),

                "input_index": (
                    None
                ),

                "session_id": (
                    "fresh_process"
                ),

                "label": "",

                "latency_ns": (
                    int(
                        latency_ns
                    )
                ),

                "latency_ms": (
                    float(
                        latency_ms
                    )
                ),

                "child_pid": (
                    child_payload.get(
                        "pid"
                    )
                ),
            }
        )

        print(
            f"  Cold repetition "
            f"{repetition}/{repetitions}: "
            f"{latency_ms:.2f} ms"
        )

    return rows


# =============================================================================
# INPUT DISCOVERY
# =============================================================================

def first_existing_directory(
    root: Path,
    names: Sequence[str],
) -> Path:
    """
    Support both historical and current folder naming conventions.

    If no named subdirectory exists, return root itself. This allows
    datasets whose modality files are stored directly beneath the root.
    """

    for name in names:
        candidate = (
            root
            / name
        )

        if candidate.is_dir():
            return candidate

    return root


def build_file_index(
    directory: Path,
    *,
    extensions: Optional[set[str]] = None,
) -> dict[str, Path]:
    if not directory.exists():
        return {}

    output: dict[
        str,
        Path
    ] = {}

    for path in sorted(
        directory.iterdir()
    ):
        if not path.is_file():
            continue

        if (
            extensions is not None
            and
            path.suffix.lower()
            not in extensions
        ):
            continue

        output[
            path.stem
        ] = path

    return output


def load_sample_label(
    keystroke_path: Path,
) -> str:
    try:
        with keystroke_path.open(
            "r",
            encoding="utf-8",
        ) as file_handle:
            payload = json.load(
                file_handle
            )

    except Exception:
        return ""

    if not isinstance(
        payload,
        dict,
    ):
        return ""

    return normalise_label(
        payload.get(
            "label",
            "",
        )
    )


def validate_keystroke_json(
    path: Path,
) -> bool:
    try:
        with path.open(
            "r",
            encoding="utf-8",
        ) as file_handle:
            payload = json.load(
                file_handle
            )

        if not isinstance(
            payload,
            dict,
        ):
            return False

        features = payload.get(
            "features"
        )

        if not isinstance(
            features,
            dict,
        ):
            return False

        return len(
            features
        ) > 0

    except Exception:
        return False


def discover_complete_sessions(
    dataset_root: Path,
) -> list[BenchmarkSample]:
    """
    Discover complete already-acquired multimodal sessions.

    Supported layouts include:

        root/texts/
        root/text/
        root/keystrokes/
        root/json/
        root/audio/
        root/images/
        root/image/

    and direct files beneath root.
    """

    if not dataset_root.exists():
        return []

    text_dir = first_existing_directory(
        dataset_root,
        [
            "texts",
            "text",
        ],
    )

    keystroke_dir = first_existing_directory(
        dataset_root,
        [
            "keystrokes",
            "keystroke",
            "json",
        ],
    )

    audio_dir = first_existing_directory(
        dataset_root,
        [
            "audio",
            "audios",
        ],
    )

    image_dir = first_existing_directory(
        dataset_root,
        [
            "images",
            "image",
        ],
    )

    text_index = build_file_index(
        text_dir,
        extensions={
            ".txt",
        },
    )

    keystroke_index = build_file_index(
        keystroke_dir,
        extensions={
            ".json",
        },
    )

    audio_index = build_file_index(
        audio_dir,
        extensions=(
            SUPPORTED_AUDIO_EXTENSIONS
        ),
    )

    image_index = build_file_index(
        image_dir,
        extensions=(
            SUPPORTED_IMAGE_EXTENSIONS
        ),
    )

    common_session_ids = (
        set(
            text_index
        )
        & set(
            keystroke_index
        )
        & set(
            audio_index
        )
        & set(
            image_index
        )
    )

    samples: list[
        BenchmarkSample
    ] = []

    for session_id in sorted(
        common_session_ids
    ):
        text_path = (
            text_index[
                session_id
            ]
        )

        keystroke_path = (
            keystroke_index[
                session_id
            ]
        )

        audio_path = (
            audio_index[
                session_id
            ]
        )

        image_path = (
            image_index[
                session_id
            ]
        )

        try:
            text = (
                text_path.read_text(
                    encoding="utf-8"
                )
                .strip()
            )

        except Exception:
            continue

        # Production text readiness currently expects meaningful text.
        if len(text) < 20:
            continue

        if not validate_keystroke_json(
            keystroke_path
        ):
            continue

        if (
            audio_path.stat().st_size
            <= 0
        ):
            continue

        if (
            image_path.stat().st_size
            <= 0
        ):
            continue

        label = load_sample_label(
            keystroke_path
        )

        samples.append(
            BenchmarkSample(
                session_id=(
                    session_id
                ),

                label=(
                    label
                ),

                text=(
                    text
                ),

                text_path=(
                    text_path
                ),

                keystroke_path=(
                    keystroke_path
                ),

                audio_path=(
                    audio_path
                ),

                image_path=(
                    image_path
                ),
            )
        )

    return samples


def select_balanced_samples(
    candidates: list[BenchmarkSample],
    *,
    maximum: int,
) -> list[BenchmarkSample]:
    """
    Prefer a roughly class-balanced subset where labels are available.
    """

    if maximum <= 0:
        raise ValueError(
            "maximum must be positive."
        )

    if len(candidates) <= maximum:
        return list(
            candidates
        )

    grouped: dict[
        str,
        list[BenchmarkSample]
    ] = {
        label: []
        for label
        in CANONICAL_CLASSES
    }

    for sample in candidates:
        if sample.label in grouped:
            grouped[
                sample.label
            ].append(
                sample
            )

    selected: list[
        BenchmarkSample
    ] = []

    positions = {
        label: 0
        for label
        in CANONICAL_CLASSES
    }

    while len(selected) < maximum:
        added = False

        for label in CANONICAL_CLASSES:
            index = positions[
                label
            ]

            group = grouped[
                label
            ]

            if index < len(group):
                selected.append(
                    group[index]
                )

                positions[
                    label
                ] += 1

                added = True

                if len(selected) >= maximum:
                    break

        if not added:
            break

    # Fill remaining places from any sessions not already selected.
    if len(selected) < maximum:
        selected_ids = {
            sample.session_id
            for sample
            in selected
        }

        remaining = [
            sample
            for sample
            in candidates
            if sample.session_id
            not in selected_ids
        ]

        selected.extend(
            remaining[
                :
                maximum
                - len(selected)
            ]
        )

    return selected


def resolve_representative_samples(
    *,
    explicit_input_root: Optional[Path],
    maximum: int,
) -> tuple[
    Path,
    list[BenchmarkSample],
]:
    candidate_roots: list[
        Path
    ] = []

    if explicit_input_root is not None:
        candidate_roots.append(
            explicit_input_root.resolve()
        )

    else:
        candidate_roots.extend(
            [
                SAMPLE_DATA_DIR,
                SESSION_ALIGNED_DIR,
            ]
        )

    diagnostics: dict[
        str,
        int
    ] = {}

    for candidate_root in candidate_roots:
        samples = discover_complete_sessions(
            candidate_root
        )

        diagnostics[
            str(candidate_root)
        ] = len(
            samples
        )

        if (
            len(samples)
            >= MINIMUM_REPRESENTATIVE_SESSIONS
        ):
            selected = select_balanced_samples(
                samples,
                maximum=maximum,
            )

            return (
                candidate_root,
                selected,
            )

    raise RuntimeError(
        "Could not locate a sufficient complete multimodal "
        "benchmark input set.\n"
        f"At least {MINIMUM_REPRESENTATIVE_SESSIONS} "
        "complete sessions are required.\n"
        "Observed complete-session counts:\n"
        + json.dumps(
            diagnostics,
            indent=2,
        )
    )


# =============================================================================
# FUSION-ONLY INPUTS
# =============================================================================

def load_feature_schema() -> list[str]:
    require_file(
        FEATURE_SCHEMA_PATH,
        "Fusion feature schema",
    )

    with FEATURE_SCHEMA_PATH.open(
        "r",
        encoding="utf-8",
    ) as file_handle:
        payload = json.load(
            file_handle
        )

    if not isinstance(
        payload,
        list,
    ):
        raise ValueError(
            "feature_columns.json must contain a JSON list."
        )

    feature_columns = [
        str(value)
        for value
        in payload
    ]

    if (
        len(feature_columns)
        != EXPECTED_FEATURE_COUNT
    ):
        raise ValueError(
            "Fusion feature-schema dimension mismatch.\n"
            f"Expected: {EXPECTED_FEATURE_COUNT}\n"
            f"Observed: {len(feature_columns)}"
        )

    if (
        len(set(feature_columns))
        != EXPECTED_FEATURE_COUNT
    ):
        raise ValueError(
            "Fusion feature schema contains duplicate names."
        )

    return feature_columns


def build_fusion_benchmark_samples(
    representative_samples: list[BenchmarkSample],
    feature_columns: list[str],
) -> list[FusionBenchmarkSample]:
    """
    Build isolated fusion-stage inputs from persisted full feature rows.

    These rows are NOT used for text/audio/image extraction timings.
    """

    require_file(
        FUSION_DATASET_PATH,
        "Fusion training dataset",
    )

    dataframe = pd.read_csv(
        FUSION_DATASET_PATH
    )

    required_columns = {
        "session_id",
        "label",
        *feature_columns,
    }

    missing_columns = (
        required_columns
        - set(
            dataframe.columns
        )
    )

    if missing_columns:
        raise ValueError(
            "Fusion training dataset is missing "
            "required columns.\n"
            f"Missing count: {len(missing_columns)}\n"
            f"Examples: {sorted(missing_columns)[:20]}"
        )

    dataframe = dataframe.copy()

    dataframe[
        "session_id"
    ] = (
        dataframe[
            "session_id"
        ]
        .astype(str)
    )

    row_by_session = {
        str(
            row[
                "session_id"
            ]
        ): row
        for (
            _,
            row,
        )
        in dataframe.iterrows()
    }

    selected_rows: list[
        pd.Series
    ] = []

    selected_ids: set[
        str
    ] = set()

    for sample in representative_samples:
        row = row_by_session.get(
            sample.session_id
        )

        if row is None:
            continue

        selected_rows.append(
            row
        )

        selected_ids.add(
            sample.session_id
        )

    # Fill unmatched demonstration sessions using other valid fusion rows.
    # This affects ONLY the isolated fusion benchmark.
    if (
        len(selected_rows)
        < len(representative_samples)
    ):
        for (
            _,
            row,
        ) in dataframe.iterrows():
            session_id = str(
                row[
                    "session_id"
                ]
            )

            if session_id in selected_ids:
                continue

            selected_rows.append(
                row
            )

            selected_ids.add(
                session_id
            )

            if (
                len(selected_rows)
                >= len(representative_samples)
            ):
                break

    if not selected_rows:
        raise RuntimeError(
            "No persisted complete feature rows "
            "are available for fusion timing."
        )

    output: list[
        FusionBenchmarkSample
    ] = []

    for row in selected_rows:
        features: dict[
            str,
            float
        ] = {}

        for column in feature_columns:
            value = pd.to_numeric(
                row[column],
                errors="coerce",
            )

            if pd.isna(value):
                value = 0.0

            numeric_value = float(
                value
            )

            if not math.isfinite(
                numeric_value
            ):
                numeric_value = 0.0

            features[
                column
            ] = numeric_value

        output.append(
            FusionBenchmarkSample(
                session_id=str(
                    row[
                        "session_id"
                    ]
                ),

                label=normalise_label(
                    row[
                        "label"
                    ]
                ),

                features=features,
            )
        )

    return output


# =============================================================================
# PRODUCTION CONTRACT VALIDATION
# =============================================================================

def validate_predictor_contract(
    predictor: Any,
) -> None:
    required_methods = [
        "extract_keystroke_features",
        "extract_text_features",
        "extract_audio_features",
        "extract_image_features",
        "build_fusion_dataframe",
        "_predict_probability_dict",
        "predict",
    ]

    missing_methods = [
        method_name
        for method_name
        in required_methods
        if not callable(
            getattr(
                predictor,
                method_name,
                None,
            )
        )
    ]

    if missing_methods:
        raise RuntimeError(
            "Canonical predictor no longer exposes the "
            "required production benchmark methods.\n"
            f"Missing: {missing_methods}"
        )

    if not hasattr(
        predictor,
        "fusion_model",
    ):
        raise RuntimeError(
            "Canonical predictor does not expose fusion_model."
        )


def validate_sample_once(
    predictor: Any,
    sample: BenchmarkSample,
) -> None:
    """
    Run one complete production prediction before benchmark timing.
    """

    result = predictor.predict(
        keystroke_json=(
            sample.keystroke_path
        ),

        text=(
            sample.text
        ),

        audio_path=(
            sample.audio_path
        ),

        image_path=(
            sample.image_path
        ),
    )

    if not isinstance(
        result,
        dict,
    ):
        raise RuntimeError(
            "Canonical predictor returned "
            "a non-dictionary result."
        )

    probabilities = result.get(
        "probabilities"
    )

    if not isinstance(
        probabilities,
        dict,
    ):
        raise RuntimeError(
            "Canonical prediction does not expose "
            "a probability dictionary."
        )

    if (
        set(probabilities)
        != set(CANONICAL_CLASSES)
    ):
        raise RuntimeError(
            "Canonical prediction returned an unexpected "
            "probability schema.\n"
            f"Observed keys: {sorted(probabilities)}"
        )


# =============================================================================
# PRODUCTION STAGE CALLS
# =============================================================================

def run_keystroke_stage(
    predictor: Any,
    sample: BenchmarkSample,
) -> Any:
    return predictor.extract_keystroke_features(
        sample.keystroke_path
    )


def run_text_stage(
    predictor: Any,
    sample: BenchmarkSample,
) -> Any:
    return predictor.extract_text_features(
        sample.text
    )


def run_audio_stage(
    predictor: Any,
    sample: BenchmarkSample,
) -> Any:
    return predictor.extract_audio_features(
        sample.audio_path
    )


def run_visual_stage(
    predictor: Any,
    sample: BenchmarkSample,
) -> Any:
    return predictor.extract_image_features(
        sample.image_path
    )


def run_fusion_stage(
    predictor: Any,
    sample: FusionBenchmarkSample,
) -> Any:
    """
    Measure production schema reconstruction plus final classifier logic.

    Neural feature extraction is intentionally excluded from this isolated
    stage.
    """

    X = predictor.build_fusion_dataframe(
        sample.features
    )

    # The canonical complete predict path performs both:
    #   1. native model prediction
    #   2. probability extraction
    native_prediction = (
        predictor
        .fusion_model
        .predict(
            X
        )[0]
    )

    probabilities = (
        predictor
        ._predict_probability_dict(
            predictor.fusion_model,
            X,
            model_name=(
                "Fusion classifier"
            ),
        )
    )

    if not isinstance(
        probabilities,
        dict,
    ):
        raise RuntimeError(
            "Fusion probability helper returned "
            "invalid output."
        )

    return (
        native_prediction,
        probabilities,
    )


def run_end_to_end_stage(
    predictor: Any,
    sample: BenchmarkSample,
) -> Any:
    """
    Independent warm production inference measurement.

    This value is never derived by adding individual stage timings.
    """

    return predictor.predict(
        keystroke_json=(
            sample.keystroke_path
        ),

        text=(
            sample.text
        ),

        audio_path=(
            sample.audio_path
        ),

        image_path=(
            sample.image_path
        ),
    )


# =============================================================================
# HIGH-RESOLUTION TIMING
# =============================================================================

def benchmark_warm_stage(
    *,
    stage_name: str,
    items: Sequence[Any],
    function: Callable[[Any], Any],
    warmups: int,
    repetitions: int,
) -> list[dict[str, Any]]:
    """
    Execute untimed warm-ups followed by measured repetitions.
    """

    if not items:
        raise ValueError(
            f"No inputs available for stage: {stage_name}"
        )

    if repetitions <= 0:
        raise ValueError(
            "repetitions must be positive."
        )

    if warmups < 0:
        raise ValueError(
            "warmups must be non-negative."
        )

    # -------------------------------------------------------------------------
    # Warm-up executions
    # -------------------------------------------------------------------------

    for warmup_index in range(
        warmups
    ):
        item = items[
            warmup_index
            % len(items)
        ]

        function(
            item
        )

    gc.collect()

    rows: list[
        dict[str, Any]
    ] = []

    # -------------------------------------------------------------------------
    # Measured executions
    # -------------------------------------------------------------------------

    for repetition in range(
        1,
        repetitions + 1,
    ):
        input_index = (
            (
                repetition
                - 1
            )
            % len(items)
        )

        item = items[
            input_index
        ]

        start_ns = (
            time.perf_counter_ns()
        )

        function(
            item
        )

        end_ns = (
            time.perf_counter_ns()
        )

        latency_ns = (
            end_ns
            - start_ns
        )

        latency_ms = (
            latency_ns
            / 1_000_000.0
        )

        session_id = str(
            getattr(
                item,
                "session_id",
                "",
            )
        )

        label = str(
            getattr(
                item,
                "label",
                "",
            )
        )

        rows.append(
            {
                "stage": (
                    stage_name
                ),

                "run_type": (
                    "warm"
                ),

                "repetition": (
                    repetition
                ),

                "input_index": (
                    input_index
                ),

                "session_id": (
                    session_id
                ),

                "label": (
                    label
                ),

                "latency_ns": (
                    int(
                        latency_ns
                    )
                ),

                "latency_ms": (
                    float(
                        latency_ms
                    )
                ),

                "child_pid": (
                    None
                ),
            }
        )

        print(
            f"  {stage_name} "
            f"{repetition:02d}/{repetitions}: "
            f"{latency_ms:.2f} ms "
            f"[{session_id}]"
        )

    return rows


# =============================================================================
# SUMMARY STATISTICS
# =============================================================================

def percentile_95(
    values: np.ndarray,
) -> float:
    return float(
        np.percentile(
            values,
            95,
        )
    )


def summarise_stage(
    stage_name: str,
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    matching_rows = [
        row
        for row
        in rows
        if row[
            "stage"
        ]
        == stage_name
    ]

    if not matching_rows:
        raise ValueError(
            f"No timing rows available for stage: "
            f"{stage_name}"
        )

    values = np.asarray(
        [
            float(
                row[
                    "latency_ms"
                ]
            )
            for row
            in matching_rows
        ],
        dtype=np.float64,
    )

    return {
        "stage": (
            stage_name
        ),

        "n": (
            int(
                len(values)
            )
        ),

        "mean_ms": (
            float(
                np.mean(values)
            )
        ),

        "sd_ms": (
            float(
                np.std(
                    values,
                    ddof=SD_DDOF,
                )
            )
        ),

        "median_ms": (
            float(
                np.median(values)
            )
        ),

        "p95_ms": (
            percentile_95(
                values
            )
        ),

        "min_ms": (
            float(
                np.min(values)
            )
        ),

        "max_ms": (
            float(
                np.max(values)
            )
        ),
    }


def build_summary_dataframe(
    raw_rows: list[dict[str, Any]],
) -> pd.DataFrame:
    summaries = [
        summarise_stage(
            stage_name,
            raw_rows,
        )
        for stage_name
        in STAGE_ORDER
    ]

    return pd.DataFrame(
        summaries
    )


# =============================================================================
# REPORT TABLE
# =============================================================================

def print_markdown_table(
    results: pd.DataFrame,
) -> None:
    print()

    print(
        "| Processing stage | n | Mean (ms) | "
        "SD (ms) | Median (ms) | p95 (ms) |"
    )

    print(
        "|---|---:|---:|---:|---:|---:|"
    )

    for (
        _,
        row,
    ) in results.iterrows():
        print(
            "| "
            f"{row['stage']} | "
            f"{int(row['n'])} | "
            f"{float(row['mean_ms']):.2f} | "
            f"{float(row['sd_ms']):.2f} | "
            f"{float(row['median_ms']):.2f} | "
            f"{float(row['p95_ms']):.2f} |"
        )


# =============================================================================
# FINAL INTERPRETATION
# =============================================================================

def get_stage_row(
    dataframe: pd.DataFrame,
    stage: str,
) -> pd.Series:
    matching = dataframe[
        dataframe[
            "stage"
        ]
        == stage
    ]

    if len(matching) != 1:
        raise RuntimeError(
            f"Expected exactly one summary row "
            f"for {stage!r}; observed {len(matching)}."
        )

    return matching.iloc[
        0
    ]


def print_final_interpretation(
    results: pd.DataFrame,
) -> dict[str, Any]:
    warm_end_to_end = get_stage_row(
        results,
        STAGE_END_TO_END,
    )

    cold_start = get_stage_row(
        results,
        STAGE_COLD,
    )

    modality_stages = (
        results[
            results[
                "stage"
            ]
            .isin(
                [
                    STAGE_KEYSTROKE,
                    STAGE_TEXT,
                    STAGE_AUDIO,
                    STAGE_VISUAL,
                ]
            )
        ]
        .copy()
    )

    slowest_modality_row = (
        modality_stages
        .sort_values(
            "mean_ms",
            ascending=False,
        )
        .iloc[0]
    )

    warm_mean = float(
        warm_end_to_end[
            "mean_ms"
        ]
    )

    warm_p95 = float(
        warm_end_to_end[
            "p95_ms"
        ]
    )

    cold_mean = float(
        cold_start[
            "mean_ms"
        ]
    )

    cadence_supported = bool(
        warm_p95
        <= OPERATIONAL_UPDATE_CADENCE_MS
    )

    print()
    print(
        "Benchmark interpretation"
    )
    print(
        "------------------------"
    )

    print(
        "Warm end-to-end mean latency: "
        f"{warm_mean:.2f} ms."
    )

    print(
        "Warm end-to-end p95 latency: "
        f"{warm_p95:.2f} ms."
    )

    print(
        "Slowest modality-specific stage by mean latency: "
        f"{slowest_modality_row['stage']} "
        f"({float(slowest_modality_row['mean_ms']):.2f} ms)."
    )

    print(
        "Cold initialisation mean: "
        f"{cold_mean:.2f} ms."
    )

    print()

    print(
        "Warm end-to-end measured independently: YES."
    )

    print(
        "Warm end-to-end calculated by summing "
        "component means: NO."
    )

    print()

    if cadence_supported:
        print(
            "On this measured CPU environment, warm p95 "
            f"({warm_p95:.2f} ms) is below the current "
            f"{OPERATIONAL_UPDATE_CADENCE_MS:.0f} ms "
            "SenseFuzeAI live update cadence. This supports "
            "the report's near-real-time operational description "
            "for this measured hardware, software environment and "
            "input set only. It does not establish universal "
            "real-time performance."
        )

    else:
        print(
            "On this measured CPU environment, warm p95 "
            f"({warm_p95:.2f} ms) exceeds the current "
            f"{OPERATIONAL_UPDATE_CADENCE_MS:.0f} ms "
            "SenseFuzeAI live update cadence. These measurements "
            "therefore do not support claiming that this CPU setup "
            "consistently completes inference within the current "
            "near-real-time update interval."
        )

    return {
        "warm_end_to_end_mean_ms": (
            warm_mean
        ),

        "warm_end_to_end_p95_ms": (
            warm_p95
        ),

        "slowest_modality_stage": (
            str(
                slowest_modality_row[
                    "stage"
                ]
            )
        ),

        "slowest_modality_mean_ms": (
            float(
                slowest_modality_row[
                    "mean_ms"
                ]
            )
        ),

        "cold_start_mean_ms": (
            cold_mean
        ),

        "operational_update_cadence_ms": (
            OPERATIONAL_UPDATE_CADENCE_MS
        ),

        "warm_p95_within_operational_cadence": (
            cadence_supported
        ),

        "near_real_time_interpretation_scope": (
            "Measured CPU hardware, software environment, "
            "input sizes and implementation only."
        ),
    }


# =============================================================================
# PRODUCTION PATH METADATA
# =============================================================================

def build_production_path_metadata() -> dict[str, Any]:
    return {
        "initialisation": {
            "class": (
                "FinalMultimodalInference"
            ),

            "method": (
                "__init__"
            ),

            "source": (
                str(
                    FINAL_INFERENCE_PATH
                )
            ),

            "description": (
                "Loads final fusion classifier/schema, MPNet, "
                "WavLM, CLIP and webcam-calibrated visual model "
                "where required."
            ),
        },

        "keystroke_processing": {
            "method": (
                "FinalMultimodalInference."
                "extract_keystroke_features"
            ),

            "description": (
                "Loads already-extracted session keystroke "
                "features from the production-compatible JSON."
            ),
        },

        "text_processing": {
            "method": (
                "FinalMultimodalInference."
                "extract_text_features"
            ),

            "description": (
                "Executes the production MPNet text "
                "feature-extraction path."
            ),
        },

        "audio_processing": {
            "method": (
                "FinalMultimodalInference."
                "extract_audio_features"
            ),

            "description": (
                "Decodes already-acquired audio and executes "
                "Librosa/WavLM production feature extraction."
            ),
        },

        "visual_processing": {
            "method": (
                "FinalMultimodalInference."
                "extract_image_features"
            ),

            "description": (
                "Decodes the already-acquired image, executes "
                "CLIP extraction and creates webcam-calibrated "
                "visual-derived predictors."
            ),
        },

        "fusion_schema": {
            "method": (
                "FinalMultimodalInference."
                "build_fusion_dataframe"
            ),

            "description": (
                "Reconstructs the exact persisted "
                "2,373-dimensional fusion row."
            ),
        },

        "fusion_classifier": {
            "members": [
                (
                    "FinalMultimodalInference."
                    "fusion_model.predict"
                ),
                (
                    "FinalMultimodalInference."
                    "_predict_probability_dict"
                ),
            ],

            "description": (
                "Executes the learned fusion classifier and "
                "canonical probability-vector construction."
            ),
        },

        "warm_end_to_end": {
            "method": (
                "FinalMultimodalInference.predict"
            ),

            "description": (
                "Canonical stateless server-side four-modality "
                "raw-inference operation."
            ),

            "measured_independently": (
                True
            ),

            "derived_by_stage_sum": (
                False
            ),
        },

        "web_backend": {
            "source": (
                str(
                    WEB_APP_PATH
                )
            ),

            "endpoint": (
                "POST /predict_live"
            ),

            "canonical_inference_call": (
                "predictor.predict(...)"
            ),
        },
    }


# =============================================================================
# INPUT METADATA
# =============================================================================

def sample_metadata(
    sample: BenchmarkSample,
) -> dict[str, Any]:
    return {
        "session_id": (
            sample.session_id
        ),

        "label": (
            sample.label
        ),

        "text_characters": (
            len(
                sample.text
            )
        ),

        "text_path": (
            str(
                sample.text_path
            )
        ),

        "keystroke_path": (
            str(
                sample.keystroke_path
            )
        ),

        "audio_path": (
            str(
                sample.audio_path
            )
        ),

        "audio_bytes": (
            sample.audio_path.stat().st_size
        ),

        "image_path": (
            str(
                sample.image_path
            )
        ),

        "image_bytes": (
            sample.image_path.stat().st_size
        ),
    }


# =============================================================================
# SCRIPT SNAPSHOT
# =============================================================================

def save_script_snapshot(
    output_dir: Path,
) -> Path:
    source_path = (
        Path(__file__)
        .resolve()
    )

    destination = (
        output_dir
        / source_path.name
    )

    if source_path != destination.resolve():
        shutil.copy2(
            source_path,
            destination,
        )

    return destination


# =============================================================================
# COMMAND-LINE ARGUMENTS
# =============================================================================

def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run the SenseFuzeAI reproducible "
            "CPU-only inference latency benchmark."
        )
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=(
            "Directory for benchmark CSV/JSON evidence."
        ),
    )

    parser.add_argument(
        "--input-root",
        type=Path,
        default=None,
        help=(
            "Optional explicit raw multimodal input root."
        ),
    )

    parser.add_argument(
        "--cold-repetitions",
        type=int,
        default=DEFAULT_COLD_REPETITIONS,
        help=(
            "Fresh-process cold-start repetitions. "
            "Default: 5."
        ),
    )

    parser.add_argument(
        "--warm-repetitions",
        type=int,
        default=DEFAULT_WARM_REPETITIONS,
        help=(
            "Measured warm repetitions per stage. "
            "Default: 30."
        ),
    )

    parser.add_argument(
        "--warmups",
        type=int,
        default=DEFAULT_WARMUPS,
        help=(
            "Untimed warm-up runs before each stage. "
            "Default: 3."
        ),
    )

    parser.add_argument(
        "--max-sessions",
        type=int,
        default=DEFAULT_MAX_SESSIONS,
        help=(
            "Maximum number of representative sessions. "
            "Default: 20."
        ),
    )

    parser.add_argument(
        "--cold-timeout",
        type=int,
        default=900,
        help=(
            "Maximum seconds for each cold process. "
            "Default: 900."
        ),
    )

    parser.add_argument(
        "--allow-python-version-mismatch",
        action="store_true",
        help=(
            "Permit a Python version other than "
            "the documented 3.11.9 environment."
        ),
    )

    # Internal child mode used only by the parent benchmark.
    parser.add_argument(
        "--cold-child",
        action="store_true",
        help=argparse.SUPPRESS,
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

    # -------------------------------------------------------------------------
    # Internal cold-start process
    # -------------------------------------------------------------------------

    if args.cold_child:
        cold_child_main()
        return

    # -------------------------------------------------------------------------
    # Argument validation
    # -------------------------------------------------------------------------

    if args.cold_repetitions <= 0:
        parser.error(
            "--cold-repetitions must be positive."
        )

    if args.warm_repetitions <= 0:
        parser.error(
            "--warm-repetitions must be positive."
        )

    if args.warmups < 0:
        parser.error(
            "--warmups must be zero or positive."
        )

    if args.max_sessions <= 0:
        parser.error(
            "--max-sessions must be positive."
        )

    if args.cold_timeout <= 0:
        parser.error(
            "--cold-timeout must be positive."
        )

    if (
        current_python_version()
        != EXPECTED_PYTHON_VERSION
        and
        not args.allow_python_version_mismatch
    ):
        raise RuntimeError(
            "Python version differs from the documented "
            "final benchmark environment.\n"
            f"Expected: {EXPECTED_PYTHON_VERSION}\n"
            f"Observed: {current_python_version()}\n"
            "Use --allow-python-version-mismatch only "
            "when intentionally documented."
        )

    # -------------------------------------------------------------------------
    # Required project artifacts
    # -------------------------------------------------------------------------

    for (
        path,
        description,
    ) in [
        (
            FINAL_INFERENCE_PATH,
            "Canonical inference implementation",
        ),

        (
            FUSION_DATASET_PATH,
            "Final fusion training dataset",
        ),

        (
            FEATURE_SCHEMA_PATH,
            "Final fusion feature schema",
        ),

        (
            FUSION_MODEL_PATH,
            "Final fusion classifier",
        ),
    ]:
        require_file(
            path,
            description,
        )

    output_dir = (
        Path(
            args.output_dir
        )
        .resolve()
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # =========================================================================
    # INTRODUCTION
    # =========================================================================

    print_heading(
        "SenseFuzeAI CPU-Only Inference Latency Benchmark"
    )

    print(
        f"Project root       : {ROOT_DIR}"
    )

    print(
        f"Python             : {current_python_version()}"
    )

    print(
        "CUDA visible       : "
        f"{os.environ.get('CUDA_VISIBLE_DEVICES')!r}"
    )

    print(
        f"Cold repetitions   : {args.cold_repetitions}"
    )

    print(
        f"Warm repetitions   : {args.warm_repetitions}"
    )

    print(
        f"Warm-ups per stage : {args.warmups}"
    )

    print(
        f"Max input sessions : {args.max_sessions}"
    )

    print(
        f"Output             : {output_dir}"
    )

    print()

    print(
        "Acquisition duration is EXCLUDED "
        "from processing latency."
    )

    print(
        "Warm end-to-end processing is "
        "measured independently."
    )

    raw_rows: list[
        dict[str, Any]
    ] = []

    # =========================================================================
    # 1. COLD INITIALISATION
    # =========================================================================

    print_heading(
        "1. COLD APPLICATION / MODEL INITIALISATION"
    )

    print(
        "Each measurement launches a fresh Python process."
    )

    print(
        "Model downloads are disabled."
    )

    print(
        "Operating-system file caches are not explicitly "
        "flushed; this limitation is recorded in metadata."
    )

    cold_rows = benchmark_cold_initialisation(
        repetitions=(
            args.cold_repetitions
        ),

        timeout_seconds=(
            args.cold_timeout
        ),
    )

    raw_rows.extend(
        cold_rows
    )

    # =========================================================================
    # REPRESENTATIVE INPUTS
    # =========================================================================

    print_heading(
        "REPRESENTATIVE MULTIMODAL INPUT SET"
    )

    (
        input_root,
        representative_samples,
    ) = resolve_representative_samples(
        explicit_input_root=(
            args.input_root
        ),

        maximum=(
            args.max_sessions
        ),
    )

    print(
        f"Input root: {input_root}"
    )

    print(
        "Complete representative sessions: "
        f"{len(representative_samples)}"
    )

    label_counts: dict[
        str,
        int
    ] = {}

    for sample in representative_samples:
        label_counts[
            sample.label
        ] = (
            label_counts.get(
                sample.label,
                0,
            )
            + 1
        )

    print(
        "Label distribution: "
        f"{label_counts}"
    )

    for (
        index,
        sample,
    ) in enumerate(
        representative_samples,
        start=1,
    ):
        print(
            f"  {index:02d}. "
            f"{sample.session_id} "
            f"[{sample.label or 'label unavailable'}]"
        )

    # =========================================================================
    # LOAD WARM PRODUCTION PREDICTOR
    # =========================================================================

    print_heading(
        "LOADING WARM CPU PREDICTOR"
    )

    predictor = (
        create_cpu_predictor()
    )

    validate_predictor_contract(
        predictor
    )

    print(
        "Validated predictor device: "
        f"{predictor.device}"
    )

    print(
        "Validated feature count: "
        f"{len(predictor.feature_columns)}"
    )

    print()

    print(
        "Running one untimed complete production "
        "validation prediction..."
    )

    validate_sample_once(
        predictor,
        representative_samples[
            0
        ],
    )

    print(
        "Production validation prediction: PASS"
    )

    # =========================================================================
    # FUSION-ONLY INPUT SET
    # =========================================================================

    feature_columns = (
        load_feature_schema()
    )

    if (
        list(
            predictor.feature_columns
        )
        != feature_columns
    ):
        raise RuntimeError(
            "Runtime predictor feature schema differs "
            "from feature_columns.json."
        )

    fusion_samples = (
        build_fusion_benchmark_samples(
            representative_samples,
            feature_columns,
        )
    )

    print()

    print(
        "Fusion-only representative rows: "
        f"{len(fusion_samples)}"
    )

    # =========================================================================
    # 2. KEYSTROKE
    # =========================================================================

    print_heading(
        "2. KEYSTROKE PROCESSING"
    )

    keystroke_rows = benchmark_warm_stage(
        stage_name=(
            STAGE_KEYSTROKE
        ),

        items=(
            representative_samples
        ),

        function=lambda sample: (
            run_keystroke_stage(
                predictor,
                sample,
            )
        ),

        warmups=(
            args.warmups
        ),

        repetitions=(
            args.warm_repetitions
        ),
    )

    raw_rows.extend(
        keystroke_rows
    )

    # =========================================================================
    # 3. TEXT
    # =========================================================================

    print_heading(
        "3. TEXT FEATURE EXTRACTION"
    )

    text_rows = benchmark_warm_stage(
        stage_name=(
            STAGE_TEXT
        ),

        items=(
            representative_samples
        ),

        function=lambda sample: (
            run_text_stage(
                predictor,
                sample,
            )
        ),

        warmups=(
            args.warmups
        ),

        repetitions=(
            args.warm_repetitions
        ),
    )

    raw_rows.extend(
        text_rows
    )

    # =========================================================================
    # 4. AUDIO
    # =========================================================================

    print_heading(
        "4. AUDIO FEATURE EXTRACTION"
    )

    audio_rows = benchmark_warm_stage(
        stage_name=(
            STAGE_AUDIO
        ),

        items=(
            representative_samples
        ),

        function=lambda sample: (
            run_audio_stage(
                predictor,
                sample,
            )
        ),

        warmups=(
            args.warmups
        ),

        repetitions=(
            args.warm_repetitions
        ),
    )

    raw_rows.extend(
        audio_rows
    )

    # =========================================================================
    # 5. VISUAL
    # =========================================================================

    print_heading(
        "5. VISUAL FEATURE EXTRACTION"
    )

    visual_rows = benchmark_warm_stage(
        stage_name=(
            STAGE_VISUAL
        ),

        items=(
            representative_samples
        ),

        function=lambda sample: (
            run_visual_stage(
                predictor,
                sample,
            )
        ),

        warmups=(
            args.warmups
        ),

        repetitions=(
            args.warm_repetitions
        ),
    )

    raw_rows.extend(
        visual_rows
    )

    # =========================================================================
    # 6. FUSION / CLASSIFIER
    # =========================================================================

    print_heading(
        "6. SCHEMA RECONSTRUCTION + FUSION / CLASSIFIER"
    )

    print(
        "Persisted complete feature rows are used only "
        "to isolate this final classifier stage."
    )

    fusion_rows = benchmark_warm_stage(
        stage_name=(
            STAGE_FUSION
        ),

        items=(
            fusion_samples
        ),

        function=lambda sample: (
            run_fusion_stage(
                predictor,
                sample,
            )
        ),

        warmups=(
            args.warmups
        ),

        repetitions=(
            args.warm_repetitions
        ),
    )

    raw_rows.extend(
        fusion_rows
    )

    # =========================================================================
    # 7. WARM END-TO-END
    # =========================================================================

    print_heading(
        "7. INDEPENDENT WARM END-TO-END PROCESSING"
    )

    print(
        "Directly calling "
        "FinalMultimodalInference.predict(...)."
    )

    print(
        "This measurement is NOT derived from "
        "the preceding component timings."
    )

    end_to_end_rows = benchmark_warm_stage(
        stage_name=(
            STAGE_END_TO_END
        ),

        items=(
            representative_samples
        ),

        function=lambda sample: (
            run_end_to_end_stage(
                predictor,
                sample,
            )
        ),

        warmups=(
            args.warmups
        ),

        repetitions=(
            args.warm_repetitions
        ),
    )

    raw_rows.extend(
        end_to_end_rows
    )

    # =========================================================================
    # SUMMARY DATA
    # =========================================================================

    raw_dataframe = pd.DataFrame(
        raw_rows
    )

    summary_dataframe = (
        build_summary_dataframe(
            raw_rows
        )
    )

    raw_column_order = [
        "stage",
        "run_type",
        "repetition",
        "input_index",
        "session_id",
        "label",
        "latency_ns",
        "latency_ms",
        "child_pid",
    ]

    summary_column_order = [
        "stage",
        "n",
        "mean_ms",
        "sd_ms",
        "median_ms",
        "p95_ms",
        "min_ms",
        "max_ms",
    ]

    raw_dataframe = raw_dataframe[
        raw_column_order
    ]

    summary_dataframe = summary_dataframe[
        summary_column_order
    ]

    # =========================================================================
    # SAVE CSV RESULTS
    # =========================================================================

    results_path = (
        output_dir
        / "latency_results.csv"
    )

    raw_path = (
        output_dir
        / "latency_raw_runs.csv"
    )

    environment_path = (
        output_dir
        / "latency_environment.json"
    )

    summary_dataframe.to_csv(
        results_path,
        index=False,
    )

    raw_dataframe.to_csv(
        raw_path,
        index=False,
    )

    # =========================================================================
    # INTERPRETATION
    # =========================================================================

    interpretation = (
        print_final_interpretation(
            summary_dataframe
        )
    )

    # =========================================================================
    # ENVIRONMENT / METHODOLOGY METADATA
    # =========================================================================

    environment_payload = (
        collect_environment()
    )

    environment_payload.update(
        {
            "benchmark": {
                "name": (
                    "SenseFuzeAI CPU-only "
                    "inference latency benchmark"
                ),

                "report_section": (
                    "5.7"
                ),

                "timer": (
                    "time.perf_counter_ns"
                ),

                "cold_repetitions": (
                    args.cold_repetitions
                ),

                "warm_repetitions_per_stage": (
                    args.warm_repetitions
                ),

                "warmups_per_stage": (
                    args.warmups
                ),

                "sd_ddof": (
                    SD_DDOF
                ),

                "p95_definition": (
                    "numpy.percentile(values, 95)"
                ),

                "cold_start_method": (
                    "Fresh Python subprocess for every repetition."
                ),

                "cold_start_scope": (
                    "Python process startup, imports and construction "
                    "of FinalMultimodalInference through child process "
                    "completion immediately after readiness."
                ),

                "cold_start_shutdown_note": (
                    "The parent measures through immediate child exit; "
                    "therefore cold-start values include a small "
                    "conservative process-shutdown overhead."
                ),

                "cold_start_os_cache_note": (
                    "Operating-system/file caches were not forcibly "
                    "cleared between fresh processes."
                ),

                "model_downloads_allowed": (
                    False
                ),

                "warm_end_to_end_measured_independently": (
                    True
                ),

                "warm_end_to_end_is_arithmetic_stage_sum": (
                    False
                ),

                "acquisition_duration_included": (
                    False
                ),

                "excluded_acquisition_components": [
                    "waiting for participant typing",
                    "microphone recording duration",
                    "webcam/video acquisition waiting",
                    "2.5-second live update cadence",
                    "temporal observation waiting",
                    "model download time",
                    "browser/network round-trip",
                ],

                "file_decode_included": {
                    "keystroke_json_read": (
                        True
                    ),

                    "audio_file_decode": (
                        True
                    ),

                    "image_file_decode": (
                        True
                    ),

                    "text_file_read": (
                        False
                    ),
                },

                "text_file_read_note": (
                    "Text is loaded before timing because the "
                    "canonical predictor receives already-acquired "
                    "text as a Python string."
                ),

                "fusion_stage_precomputed_features": (
                    True
                ),

                "fusion_stage_precomputed_feature_note": (
                    "Persisted complete feature rows are used only "
                    "for isolated schema/fusion classifier timing. "
                    "They are not used for MPNet, WavLM, CLIP or "
                    "warm end-to-end measurements."
                ),

                "environment_collection_psutil_dependency": (
                    False
                ),
            },

            "production_code_paths": (
                build_production_path_metadata()
            ),

            "artifacts": {
                "final_inference_path": (
                    str(
                        FINAL_INFERENCE_PATH
                    )
                ),

                "final_inference_sha256": (
                    sha256_file(
                        FINAL_INFERENCE_PATH
                    )
                ),

                "fusion_model_path": (
                    str(
                        FUSION_MODEL_PATH
                    )
                ),

                "fusion_model_sha256": (
                    sha256_file(
                        FUSION_MODEL_PATH
                    )
                ),

                "feature_schema_path": (
                    str(
                        FEATURE_SCHEMA_PATH
                    )
                ),

                "feature_schema_sha256": (
                    sha256_file(
                        FEATURE_SCHEMA_PATH
                    )
                ),

                "fusion_dataset_path": (
                    str(
                        FUSION_DATASET_PATH
                    )
                ),

                "fusion_dataset_sha256": (
                    sha256_file(
                        FUSION_DATASET_PATH
                    )
                ),
            },

            "representative_inputs": {
                "source_root": (
                    str(
                        input_root
                    )
                ),

                "session_count": (
                    len(
                        representative_samples
                    )
                ),

                "sessions": [
                    sample_metadata(
                        sample
                    )
                    for sample
                    in representative_samples
                ],
            },

            "summary_results": (
                summary_dataframe.to_dict(
                    orient="records"
                )
            ),

            "report_interpretation": (
                interpretation
            ),
        }
    )

    write_json(
        environment_path,
        environment_payload,
    )

    script_snapshot = (
        save_script_snapshot(
            output_dir
        )
    )

    # =========================================================================
    # FINAL OUTPUT
    # =========================================================================

    print_heading(
        "FINAL LATENCY RESULTS"
    )

    print_markdown_table(
        summary_dataframe
    )

    print()

    print(
        "Minimum and maximum latency values are "
        "also retained in latency_results.csv."
    )

    print_heading(
        "SAVED EVIDENCE"
    )

    print(
        f"Summary results : {results_path}"
    )

    print(
        f"Raw runs        : {raw_path}"
    )

    print(
        f"Environment     : {environment_path}"
    )

    print(
        f"Script snapshot : {script_snapshot}"
    )

    print()

    print(
        "FINAL RESULT: PASS"
    )


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    main()
