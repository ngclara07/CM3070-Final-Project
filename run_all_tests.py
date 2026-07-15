# run_all_tests.py

from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parent
TEST_DIR = ROOT_DIR / "tests"
REPORT_DIR = ROOT_DIR / "test_reports"

TEST_SUITES = [
    ("Unit Testing", TEST_DIR / "test_01_unit.py"),
    ("Integration Testing", TEST_DIR / "test_02_integration.py"),
    ("System Testing", TEST_DIR / "test_03_system.py"),
    ("Acceptance Testing", TEST_DIR / "test_04_acceptance.py"),
]


def run_pytest_suite(name: str, test_file: Path) -> dict:
    print(f"\n{'=' * 80}")
    print(f"Running {name}")
    print(f"{'=' * 80}")

    if not test_file.exists():
        return {
            "suite": name,
            "file": str(test_file),
            "status": "missing",
            "return_code": -1,
            "stdout": "",
            "stderr": f"Missing test file: {test_file}",
        }

    command = [
        sys.executable,
        "-m",
        "pytest",
        str(test_file),
        "-v",
    ]

    completed = subprocess.run(
        command,
        cwd=ROOT_DIR,
        text=True,
        capture_output=True,
    )

    print(completed.stdout)

    if completed.stderr:
        print(completed.stderr)

    return {
        "suite": name,
        "file": str(test_file),
        "status": "passed" if completed.returncode == 0 else "failed",
        "return_code": completed.returncode,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
    }


def generate_markdown_report(results: list[dict]) -> Path:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = REPORT_DIR / f"sensefuzeai_test_report_{timestamp}.md"
    json_path = REPORT_DIR / f"sensefuzeai_test_report_{timestamp}.json"

    total = len(results)
    passed = sum(1 for result in results if result["status"] == "passed")
    failed = sum(1 for result in results if result["status"] == "failed")
    missing = sum(1 for result in results if result["status"] == "missing")

    overall_status = "PASSED" if failed == 0 and missing == 0 else "FAILED"

    lines = [
        "# SenseFuzeAI Test Report",
        "",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        f"Overall Status: **{overall_status}**",
        "",
        "## Summary",
        "",
        f"- Total test suites: {total}",
        f"- Passed suites: {passed}",
        f"- Failed suites: {failed}",
        f"- Missing suites: {missing}",
        "",
        "## Test Suite Results",
        "",
        "| Suite | File | Status | Return Code |",
        "|---|---|---:|---:|",
    ]

    for result in results:
        lines.append(
            f"| {result['suite']} | `{result['file']}` | "
            f"{result['status'].upper()} | {result['return_code']} |"
        )

    lines.extend(["", "## Detailed Output", ""])

    for result in results:
        lines.extend(
            [
                f"### {result['suite']}",
                "",
                f"Status: **{result['status'].upper()}**",
                "",
                "```text",
                result["stdout"].strip() or "(no stdout)",
                "```",
                "",
            ]
        )

        if result["stderr"].strip():
            lines.extend(
                [
                    "Errors / Warnings:",
                    "",
                    "```text",
                    result["stderr"].strip(),
                    "```",
                    "",
                ]
            )

    report_path.write_text("\n".join(lines), encoding="utf-8")
    json_path.write_text(json.dumps(results, indent=4), encoding="utf-8")

    return report_path


def main() -> int:
    print("SenseFuzeAI Automated Test Runner")
    print(f"Project root: {ROOT_DIR}")

    if not TEST_DIR.exists():
        print(f"Test directory not found: {TEST_DIR}")
        return 1

    results = []

    for suite_name, test_file in TEST_SUITES:
        result = run_pytest_suite(suite_name, test_file)
        results.append(result)

    report_path = generate_markdown_report(results)

    failed_or_missing = [
        result for result in results
        if result["status"] != "passed"
    ]

    print("\n" + "=" * 80)
    print("Final Test Summary")
    print("=" * 80)

    for result in results:
        print(f"{result['suite']}: {result['status'].upper()}")

    print(f"\nReport saved to: {report_path}")

    if failed_or_missing:
        print("\nOverall result: FAILED")
        return 1

    print("\nOverall result: PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
