# SenseFuzeAI Test Report

Generated: 2026-07-01 08:40:41

Overall Status: **PASSED**

## Summary

- Total test suites: 4
- Passed suites: 4
- Failed suites: 0
- Missing suites: 0

## Test Suite Results

| Suite | File | Status | Return Code |
|---|---|---:|---:|
| Unit Testing | `C:\Users\Clara\Documents\SIM-UOL Year 3\Semester 2\CM3070 Final project (FP)\assessments_april26\behaviour_app\tests\test_01_unit.py` | PASSED | 0 |
| Integration Testing | `C:\Users\Clara\Documents\SIM-UOL Year 3\Semester 2\CM3070 Final project (FP)\assessments_april26\behaviour_app\tests\test_02_integration.py` | PASSED | 0 |
| System Testing | `C:\Users\Clara\Documents\SIM-UOL Year 3\Semester 2\CM3070 Final project (FP)\assessments_april26\behaviour_app\tests\test_03_system.py` | PASSED | 0 |
| Acceptance Testing | `C:\Users\Clara\Documents\SIM-UOL Year 3\Semester 2\CM3070 Final project (FP)\assessments_april26\behaviour_app\tests\test_04_acceptance.py` | PASSED | 0 |

## Detailed Output

### Unit Testing

Status: **PASSED**

```text
============================= test session starts =============================
platform win32 -- Python 3.11.9, pytest-9.1.1, pluggy-1.6.0 -- C:\Users\Clara\Documents\SIM-UOL Year 3\Semester 2\CM3070 Final project (FP)\assessments_april26\behaviour_app\.venv\Scripts\python.exe
cachedir: .pytest_cache
rootdir: C:\Users\Clara\Documents\SIM-UOL Year 3\Semester 2\CM3070 Final project (FP)\assessments_april26\behaviour_app
plugins: anyio-4.13.0
collecting ... collected 4 items

tests/test_01_unit.py::test_confidence_level_logic PASSED                [ 25%]
tests/test_01_unit.py::test_keystroke_count_extraction PASSED            [ 50%]
tests/test_01_unit.py::test_invalid_keystroke_json_returns_zero PASSED   [ 75%]
tests/test_01_unit.py::test_fallback_prediction_outputs_valid_distribution PASSED [100%]

============================== 4 passed in 3.50s ==============================
```

### Integration Testing

Status: **PASSED**

```text
============================= test session starts =============================
platform win32 -- Python 3.11.9, pytest-9.1.1, pluggy-1.6.0 -- C:\Users\Clara\Documents\SIM-UOL Year 3\Semester 2\CM3070 Final project (FP)\assessments_april26\behaviour_app\.venv\Scripts\python.exe
cachedir: .pytest_cache
rootdir: C:\Users\Clara\Documents\SIM-UOL Year 3\Semester 2\CM3070 Final project (FP)\assessments_april26\behaviour_app
plugins: anyio-4.13.0
collecting ... collected 3 items

tests/test_02_integration.py::test_required_model_artifacts_exist PASSED [ 33%]
tests/test_02_integration.py::test_fusion_feature_schema_contains_all_modalities PASSED [ 66%]
tests/test_02_integration.py::test_final_inference_class_importable PASSED [100%]

============================= 3 passed in 20.01s ==============================
```

### System Testing

Status: **PASSED**

```text
============================= test session starts =============================
platform win32 -- Python 3.11.9, pytest-9.1.1, pluggy-1.6.0 -- C:\Users\Clara\Documents\SIM-UOL Year 3\Semester 2\CM3070 Final project (FP)\assessments_april26\behaviour_app\.venv\Scripts\python.exe
cachedir: .pytest_cache
rootdir: C:\Users\Clara\Documents\SIM-UOL Year 3\Semester 2\CM3070 Final project (FP)\assessments_april26\behaviour_app
plugins: anyio-4.13.0
collecting ... collected 4 items

tests/test_03_system.py::test_web_health_endpoint PASSED                 [ 25%]
tests/test_03_system.py::test_web_model_status_endpoint PASSED           [ 50%]
tests/test_03_system.py::test_predict_live_rejects_short_text PASSED     [ 75%]
tests/test_03_system.py::test_predict_live_rejects_insufficient_keystrokes PASSED [100%]

============================== 4 passed in 3.14s ==============================
```

### Acceptance Testing

Status: **PASSED**

```text
============================= test session starts =============================
platform win32 -- Python 3.11.9, pytest-9.1.1, pluggy-1.6.0 -- C:\Users\Clara\Documents\SIM-UOL Year 3\Semester 2\CM3070 Final project (FP)\assessments_april26\behaviour_app\.venv\Scripts\python.exe
cachedir: .pytest_cache
rootdir: C:\Users\Clara\Documents\SIM-UOL Year 3\Semester 2\CM3070 Final project (FP)\assessments_april26\behaviour_app
plugins: anyio-4.13.0
collecting ... collected 5 items

tests/test_04_acceptance.py::test_project_contains_required_live_interfaces PASSED [ 20%]
tests/test_04_acceptance.py::test_project_uses_at_least_three_pretrained_models PASSED [ 40%]
tests/test_04_acceptance.py::test_project_contains_multimodal_fusion_model PASSED [ 60%]
tests/test_04_acceptance.py::test_final_output_design_supported_in_web_script PASSED [ 80%]
tests/test_04_acceptance.py::test_dissertation_ready_evaluation_scripts_exist PASSED [100%]

============================== 5 passed in 0.08s ==============================
```
