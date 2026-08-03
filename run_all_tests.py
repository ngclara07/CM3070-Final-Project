# === run_all_tests.py ===

from __future__ import annotations

import json
import re
import subprocess
import sys
import time

from datetime import datetime
from pathlib import Path
from typing import Any


# ============================================================
# Project paths
# ============================================================

ROOT_DIR = Path(__file__).resolve().parent

TEST_DIR = (
    ROOT_DIR
    / "tests"
)

REPORT_DIR = (
    ROOT_DIR
    / "test_reports"
)


# ============================================================
# Test suite configuration
# ============================================================

TEST_SUITES = [
    (
        "Unit Testing",
        TEST_DIR / "test_01_unit.py",
        (
            "Tests isolated helper functions, probability "
            "normalisation, keystroke processing, confidence "
            "logic, and temporal probability aggregation."
        ),
    ),
    (
        "Integration Testing",
        TEST_DIR / "test_02_integration.py",
        (
            "Tests model artifacts, pretrained encoders, "
            "webcam calibration, feature schemas, final "
            "inference integration, and temporal/webcam "
            "backend integration."
        ),
    ),
    (
        "System Testing",
        TEST_DIR / "test_03_system.py",
        (
            "Tests FastAPI endpoints, input validation, "
            "temporal-session reset, browser webcam "
            "integration, and final application-level "
            "behaviour."
        ),
    ),
    (
        "Acceptance Testing",
        TEST_DIR / "test_04_acceptance.py",
        (
            "Tests whether final project requirements, "
            "multimodal artifacts, webcam calibration, "
            "temporal prediction design, and deployment "
            "interfaces are present."
        ),
    ),
    (
        "Smoke / Regression Testing",
        TEST_DIR / "test_sensefuzeai.py",
        (
            "Performs a compact regression check across "
            "major SenseFuzeAI files, artifacts, endpoints, "
            "webcam calibration, fusion schema, and temporal "
            "prediction functionality."
        ),
    ),
]


# ============================================================
# Pytest output parser
# ============================================================

def extract_pytest_counts(
    output: str,
) -> dict[str, int]:
    """
    Extract pytest result counts from terminal output.

    Supported examples:
        12 passed
        1 failed, 11 passed
        2 skipped, 10 passed
        1 error
    """

    counts = {
        "passed": 0,
        "failed": 0,
        "skipped": 0,
        "errors": 0,
        "xfailed": 0,
        "xpassed": 0,
    }

    patterns = {
        "passed": (
            r"(\d+)\s+passed"
        ),
        "failed": (
            r"(\d+)\s+failed"
        ),
        "skipped": (
            r"(\d+)\s+skipped"
        ),
        "errors": (
            r"(\d+)\s+errors?"
        ),
        "xfailed": (
            r"(\d+)\s+xfailed"
        ),
        "xpassed": (
            r"(\d+)\s+xpassed"
        ),
    }

    for key, pattern in patterns.items():

        matches = re.findall(
            pattern,
            output,
            flags=re.IGNORECASE,
        )

        if matches:

            counts[key] = int(
                matches[-1]
            )

    return counts


# ============================================================
# Individual suite runner
# ============================================================

def run_pytest_suite(
    name: str,
    test_file: Path,
    description: str,
) -> dict[str, Any]:

    print(
        "\n"
        + "=" * 92
    )

    print(
        f"Running {name}"
    )

    print(
        "=" * 92
    )

    print(
        description
    )

    print(
        f"Test file: {test_file}"
    )

    print(
        "-" * 92
    )

    if not test_file.exists():

        print(
            f"ERROR: Missing test file: "
            f"{test_file}"
        )

        return {
            "suite": name,
            "description": description,
            "file": str(test_file),
            "status": "missing",
            "return_code": -1,
            "runtime_seconds": 0.0,
            "counts": {
                "passed": 0,
                "failed": 0,
                "skipped": 0,
                "errors": 0,
                "xfailed": 0,
                "xpassed": 0,
            },
            "stdout": "",
            "stderr": (
                f"Missing test file: "
                f"{test_file}"
            ),
        }

    command = [
        sys.executable,
        "-m",
        "pytest",
        str(test_file),

        # Show individual test names.
        "-v",

        # Do not hide print() output.
        "-s",
    ]

    start_time = (
        time.perf_counter()
    )

    completed = subprocess.run(
        command,
        cwd=ROOT_DIR,
        text=True,
        capture_output=True,
        encoding="utf-8",
        errors="replace",
    )

    runtime_seconds = (
        time.perf_counter()
        - start_time
    )

    stdout = (
        completed.stdout
        or ""
    )

    stderr = (
        completed.stderr
        or ""
    )

    print(
        stdout
    )

    if stderr.strip():

        print(
            "\n--- STDERR ---"
        )

        print(
            stderr
        )

    combined_output = (
        stdout
        + "\n"
        + stderr
    )

    counts = (
        extract_pytest_counts(
            combined_output
        )
    )

    status = (
        "passed"
        if completed.returncode == 0
        else "failed"
    )

    print(
        "-" * 92
    )

    print(
        f"{name} result: "
        f"{status.upper()}"
    )

    print(
        f"Runtime: "
        f"{runtime_seconds:.2f} seconds"
    )

    return {
        "suite": name,
        "description": description,
        "file": str(test_file),
        "status": status,
        "return_code": (
            completed.returncode
        ),
        "runtime_seconds": (
            runtime_seconds
        ),
        "counts": counts,
        "stdout": stdout,
        "stderr": stderr,
    }


# ============================================================
# Markdown + JSON report generation
# ============================================================

def generate_markdown_report(
    results: list[dict[str, Any]],
) -> tuple[Path, Path]:

    REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    timestamp = (
        datetime.now()
        .strftime(
            "%Y%m%d_%H%M%S"
        )
    )

    report_path = (
        REPORT_DIR
        / (
            "sensefuzeai_test_report_"
            f"{timestamp}.md"
        )
    )

    json_path = (
        REPORT_DIR
        / (
            "sensefuzeai_test_report_"
            f"{timestamp}.json"
        )
    )

    total_suites = len(
        results
    )

    passed_suites = sum(
        1
        for result in results
        if result["status"]
        == "passed"
    )

    failed_suites = sum(
        1
        for result in results
        if result["status"]
        == "failed"
    )

    missing_suites = sum(
        1
        for result in results
        if result["status"]
        == "missing"
    )

    total_tests_passed = sum(
        result["counts"]["passed"]
        for result in results
    )

    total_tests_failed = sum(
        result["counts"]["failed"]
        for result in results
    )

    total_tests_skipped = sum(
        result["counts"]["skipped"]
        for result in results
    )

    total_errors = sum(
        result["counts"]["errors"]
        for result in results
    )

    total_xfailed = sum(
        result["counts"]["xfailed"]
        for result in results
    )

    total_xpassed = sum(
        result["counts"]["xpassed"]
        for result in results
    )

    total_runtime = sum(
        result["runtime_seconds"]
        for result in results
    )

    overall_status = (
        "PASSED"
        if (
            failed_suites == 0
            and missing_suites == 0
        )
        else "FAILED"
    )

    generated_time = (
        datetime.now()
        .strftime(
            "%Y-%m-%d %H:%M:%S"
        )
    )

    lines = [
        "# SenseFuzeAI Automated Test Report",
        "",
        f"Generated: **{generated_time}**",
        "",
        f"Overall Status: **{overall_status}**",
        "",
        "## Test Objective",
        "",
        (
            "The purpose of this automated test suite is "
            "to verify the correctness, integration, "
            "system behaviour, acceptance requirements, "
            "and regression stability of the SenseFuzeAI "
            "multimodal behavioural-state prediction "
            "system."
        ),
        "",
        (
            "The tests specifically include the "
            "webcam-calibrated image classifier and the "
            "final rolling temporal probability "
            "aggregation used by the web application."
        ),
        "",
        "## Current Prediction Architecture Under Test",
        "",
        (
            "The final web application evaluates "
            "keystroke, text, audio and image features "
            "through the multimodal fusion classifier. "
            "Each live fusion observation produces a "
            "four-class probability vector."
        ),
        "",
        (
            "The final displayed behavioural state is "
            "computed from the arithmetic mean of the "
            "latest five fusion probability vectors. "
            "The latest raw prediction remains available "
            "as diagnostic information."
        ),
        "",
        (
            "The webcam-calibrated image classifier is "
            "maintained as a separate visual-modality "
            "diagnostic and does not replace the final "
            "multimodal fusion decision."
        ),
        "",
        "## Overall Summary",
        "",
        f"- Total test suites: {total_suites}",
        f"- Passed suites: {passed_suites}",
        f"- Failed suites: {failed_suites}",
        f"- Missing suites: {missing_suites}",
        f"- Individual tests passed: {total_tests_passed}",
        f"- Individual tests failed: {total_tests_failed}",
        f"- Individual tests skipped: {total_tests_skipped}",
        f"- Pytest errors: {total_errors}",
        f"- Expected failures: {total_xfailed}",
        f"- Unexpected passes: {total_xpassed}",
        f"- Total runtime: {total_runtime:.2f} seconds",
        "",
        "## Test Suite Results",
        "",
        (
            "| Suite | Status | Passed | Failed | "
            "Skipped | Errors | Runtime (s) |"
        ),
        (
            "|---|---:|---:|---:|---:|---:|---:|"
        ),
    ]

    for result in results:

        counts = result[
            "counts"
        ]

        lines.append(
            f"| {result['suite']} "
            f"| {result['status'].upper()} "
            f"| {counts['passed']} "
            f"| {counts['failed']} "
            f"| {counts['skipped']} "
            f"| {counts['errors']} "
            f"| {result['runtime_seconds']:.2f} |"
        )

    lines.extend(
        [
            "",
            "## Testing Levels",
            "",
        ]
    )

    for result in results:

        lines.extend(
            [
                f"### {result['suite']}",
                "",
                result["description"],
                "",
                (
                    f"Test file: "
                    f"`{result['file']}`"
                ),
                "",
                (
                    f"Status: "
                    f"**{result['status'].upper()}**"
                ),
                "",
            ]
        )

    lines.extend(
        [
            "## Detailed Test Output",
            "",
        ]
    )

    for result in results:

        lines.extend(
            [
                f"### {result['suite']}",
                "",
                "```text",
                (
                    result["stdout"].strip()
                    or "(no standard output)"
                ),
                "```",
                "",
            ]
        )

        if result["stderr"].strip():

            lines.extend(
                [
                    "**Errors / Warnings**",
                    "",
                    "```text",
                    result["stderr"].strip(),
                    "```",
                    "",
                ]
            )

    lines.extend(
        [
            "## Final Outcome",
            "",
        ]
    )

    if overall_status == "PASSED":

        lines.extend(
            [
                (
                    "All required automated test suites "
                    "completed successfully."
                ),
                "",
                (
                    "The results provide automated evidence "
                    "that the tested utility functions, "
                    "model artifacts, webcam calibration "
                    "pipeline, multimodal integration, "
                    "FastAPI endpoints, temporal "
                    "probability aggregation, browser "
                    "interface, and acceptance requirements "
                    "are operational."
                ),
                "",
                (
                    "These software tests do not by "
                    "themselves establish behavioural-state "
                    "predictive validity. Model accuracy "
                    "should therefore also be reported using "
                    "the held-out evaluation metrics, "
                    "classification reports, confusion "
                    "matrices and multimodal evaluation "
                    "results produced separately by the "
                    "training/evaluation pipeline."
                ),
            ]
        )

    else:

        lines.extend(
            [
                (
                    "One or more automated test suites "
                    "did not complete successfully."
                ),
                "",
                (
                    "The failed or missing components "
                    "should be investigated before treating "
                    "the current implementation as fully "
                    "regression-tested."
                ),
            ]
        )

    report_path.write_text(
        "\n".join(lines),
        encoding="utf-8",
    )

    json_path.write_text(
        json.dumps(
            results,
            indent=4,
            default=str,
        ),
        encoding="utf-8",
    )

    return (
        report_path,
        json_path,
    )


# ============================================================
# Main execution
# ============================================================

def main() -> int:

    print(
        "=" * 92
    )

    print(
        "SenseFuzeAI Automated Test Runner"
    )

    print(
        "=" * 92
    )

    print(
        f"Project root : {ROOT_DIR}"
    )

    print(
        f"Tests        : {TEST_DIR}"
    )

    print(
        f"Reports      : {REPORT_DIR}"
    )

    print(
        f"Suites       : {len(TEST_SUITES)}"
    )

    if not TEST_DIR.exists():

        print(
            "\nERROR: Test directory "
            f"not found: {TEST_DIR}"
        )

        return 1

    results: list[
        dict[str, Any]
    ] = []

    for (
        suite_name,
        test_file,
        description,
    ) in TEST_SUITES:

        result = run_pytest_suite(
            name=suite_name,
            test_file=test_file,
            description=description,
        )

        results.append(
            result
        )

    (
        report_path,
        json_path,
    ) = generate_markdown_report(
        results
    )

    failed_or_missing = [
        result
        for result in results
        if result["status"]
        != "passed"
    ]

    print(
        "\n"
        + "=" * 92
    )

    print(
        "FINAL TEST SUMMARY"
    )

    print(
        "=" * 92
    )

    for result in results:

        counts = result[
            "counts"
        ]

        print(
            f"{result['suite']:<28} "
            f"{result['status'].upper():<8} "
            f"| passed={counts['passed']:<3} "
            f"| failed={counts['failed']:<3} "
            f"| skipped={counts['skipped']:<3} "
            f"| errors={counts['errors']:<3}"
        )

    total_passed = sum(
        result["counts"]["passed"]
        for result in results
    )

    total_failed = sum(
        result["counts"]["failed"]
        for result in results
    )

    total_skipped = sum(
        result["counts"]["skipped"]
        for result in results
    )

    total_errors = sum(
        result["counts"]["errors"]
        for result in results
    )

    print(
        "\nIndividual test totals:"
    )

    print(
        f"  Passed  : {total_passed}"
    )

    print(
        f"  Failed  : {total_failed}"
    )

    print(
        f"  Skipped : {total_skipped}"
    )

    print(
        f"  Errors  : {total_errors}"
    )

    print(
        "\nReports generated:"
    )

    print(
        f"  Markdown : {report_path}"
    )

    print(
        f"  JSON     : {json_path}"
    )

    if failed_or_missing:

        print(
            "\nOverall result: FAILED"
        )

        return 1

    print(
        "\nOverall result: PASSED"
    )

    return 0


if __name__ == "__main__":

    raise SystemExit(
        main()
    )
