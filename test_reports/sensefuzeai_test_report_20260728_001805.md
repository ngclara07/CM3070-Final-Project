# SenseFuzeAI Automated Test Report

Generated: **2026-07-28 00:18:05**

Overall Status: **PASSED**

## Test Objective

The purpose of this automated test suite is to verify the correctness, integration, system behaviour, and acceptance requirements of the SenseFuzeAI multimodal behavioural-state prediction system, including the webcam-calibrated image-classification pipeline.

## Overall Summary

- Total test suites: 4
- Passed suites: 4
- Failed suites: 0
- Missing suites: 0
- Individual tests passed: 35
- Individual tests failed: 0
- Individual tests skipped: 0
- Pytest errors: 0
- Total runtime: 13.13 seconds

## Test Suite Results

| Suite | Status | Passed | Failed | Skipped | Runtime (s) |
|---|---:|---:|---:|---:|---:|
| Unit Testing | PASSED | 7 | 0 | 0 | 1.62 |
| Integration Testing | PASSED | 11 | 0 | 0 | 9.67 |
| System Testing | PASSED | 7 | 0 | 0 | 1.46 |
| Acceptance Testing | PASSED | 10 | 0 | 0 | 0.38 |

## Testing Levels

### Unit Testing

Tests isolated helper functions and local prediction logic.

Test file: `C:\Users\CLara\Documents\SIM-UOL Year 3\Semester 2\CM3070 Final project (FP)\assessments_april26\behaviour_app\tests\test_01_unit.py`

Status: **PASSED**

### Integration Testing

Tests model artifacts, pretrained encoders, webcam calibration, feature schemas, and component integration.

Test file: `C:\Users\CLara\Documents\SIM-UOL Year 3\Semester 2\CM3070 Final project (FP)\assessments_april26\behaviour_app\tests\test_02_integration.py`

Status: **PASSED**

### System Testing

Tests FastAPI endpoints, input validation, browser webcam integration, and application-level behaviour.

Test file: `C:\Users\CLara\Documents\SIM-UOL Year 3\Semester 2\CM3070 Final project (FP)\assessments_april26\behaviour_app\tests\test_03_system.py`

Status: **PASSED**

### Acceptance Testing

Tests whether final project requirements and deployment artifacts are present.

Test file: `C:\Users\CLara\Documents\SIM-UOL Year 3\Semester 2\CM3070 Final project (FP)\assessments_april26\behaviour_app\tests\test_04_acceptance.py`

Status: **PASSED**

## Detailed Test Output

### Unit Testing

```text
============================= test session starts =============================
platform win32 -- Python 3.11.9, pytest-9.1.1, pluggy-1.6.0 -- C:\Users\CLara\Documents\SIM-UOL Year 3\Semester 2\CM3070 Final project (FP)\assessments_april26\behaviour_app\.venv\Scripts\python.exe
cachedir: .pytest_cache
rootdir: C:\Users\CLara\Documents\SIM-UOL Year 3\Semester 2\CM3070 Final project (FP)\assessments_april26\behaviour_app
plugins: anyio-4.13.0
collecting ... collected 7 items

tests/test_01_unit.py::test_confidence_level_logic 
[UNIT] Testing confidence-level thresholds...
       PASS: High / Medium / Low thresholds behave correctly.
PASSED
tests/test_01_unit.py::test_keystroke_count_extraction 
[UNIT] Testing keystroke key-down counting...
       PASS: Expected 3 key-down events, received 3.
PASSED
tests/test_01_unit.py::test_invalid_keystroke_json_returns_zero 
[UNIT] Testing malformed keystroke JSON handling...
       PASS: Invalid JSON safely produces zero keypresses.
PASSED
tests/test_01_unit.py::test_empty_keystroke_list_returns_zero 
[UNIT] Testing empty keystroke input...
       PASS: Empty event list produces zero keypresses.
PASSED
tests/test_01_unit.py::test_fallback_prediction_outputs_valid_distribution 
[UNIT] Testing fallback behavioural probability distribution...
       PASS: Four behavioural classes returned and probabilities sum to 1.000000.
PASSED
tests/test_01_unit.py::test_fallback_prediction_can_emphasise_fatigued 
[UNIT] Testing fatigue cue handling in fallback prediction...
       PASS: Fatigue-related text increases the fatigued score to 0.3600.
PASSED
tests/test_01_unit.py::test_prediction_normalisation_returns_one_primary_state 
[UNIT] Testing final single-state prediction normalization...
       PASS: Final output exposes one primary behavioural state with confidence information.
PASSED

============================== 7 passed in 0.97s ==============================
```

### Integration Testing

```text
============================= test session starts =============================
platform win32 -- Python 3.11.9, pytest-9.1.1, pluggy-1.6.0 -- C:\Users\CLara\Documents\SIM-UOL Year 3\Semester 2\CM3070 Final project (FP)\assessments_april26\behaviour_app\.venv\Scripts\python.exe
cachedir: .pytest_cache
rootdir: C:\Users\CLara\Documents\SIM-UOL Year 3\Semester 2\CM3070 Final project (FP)\assessments_april26\behaviour_app
plugins: anyio-4.13.0
collecting ... collected 11 items

tests/test_02_integration.py::test_required_model_artifacts_exist 
[INTEGRATION] Checking required model artifacts...
       PASS: 10 required model artifacts/directories found.
PASSED
tests/test_02_integration.py::test_original_and_calibrated_image_models_are_both_preserved 
[INTEGRATION] Checking original and webcam-calibrated image models...
       PASS: Original image model remains preserved.
       PASS: Separate webcam-calibrated image model exists.
PASSED
tests/test_02_integration.py::test_webcam_calibrated_image_model_loadable 
[INTEGRATION] Loading webcam-calibrated image classifier...
       PASS: Calibrated classifier loads successfully.
       Model classes: ['distracted', 'fatigued', 'focused', 'overloaded']
PASSED
tests/test_02_integration.py::test_webcam_calibration_metadata_exists_and_valid 
[INTEGRATION] Checking webcam-calibration metadata...
       PASS: Webcam-calibrated metadata exists with 22 metadata fields.
PASSED
tests/test_02_integration.py::test_webcam_calibration_dataset_exists 
[INTEGRATION] Checking webcam-calibration feature dataset...
       Resolved dataset: C:\Users\CLara\Documents\SIM-UOL Year 3\Semester 2\CM3070 Final project (FP)\assessments_april26\behaviour_app\data\processed\webcam_calibration_clip_features.csv
       PASS: Calibration dataset rows = 320
       PASS: CLIP embedding features = 768
       PASS: Behavioural classes = ['distracted', 'fatigued', 'focused', 'overloaded']
PASSED
tests/test_02_integration.py::test_webcam_calibration_summary_exists 
[INTEGRATION] Checking webcam-calibration summary...
       Resolved summary: C:\Users\CLara\Documents\SIM-UOL Year 3\Semester 2\CM3070 Final project (FP)\assessments_april26\behaviour_app\data\processed\webcam_calibration_clip_features_summary.json
       PASS: Calibration summary JSON is valid.
PASSED
tests/test_02_integration.py::test_webcam_evaluation_results_exist 
[INTEGRATION] Checking webcam-calibration evaluation directory...
       PASS: 7 webcam evaluation artifact(s) found.
PASSED
tests/test_02_integration.py::test_fusion_feature_schema_contains_all_modalities 
[INTEGRATION] Checking fusion feature schema...
       PASS: Fusion feature count = 2373
       PASS: Keystroke modality present.
       PASS: Text modality present.
       PASS: Audio modality present.
       PASS: Image modality present.
PASSED
tests/test_02_integration.py::test_image_feature_schema_uses_clip_embeddings 
[INTEGRATION] Checking image feature schema...
       PASS: Image model expects 768 CLIP embedding dimensions.
PASSED
tests/test_02_integration.py::test_final_inference_class_importable 
[INTEGRATION] Importing final multimodal inference module...
       PASS: FinalMultimodalInference class is importable.
PASSED
tests/test_02_integration.py::test_final_inference_references_clip_model 
[INTEGRATION] Checking final inference CLIP integration...
       PASS: Final inference pipeline uses the expected pretrained CLIP visual encoder.
PASSED

============================= 11 passed in 8.04s ==============================
```

### System Testing

```text
============================= test session starts =============================
platform win32 -- Python 3.11.9, pytest-9.1.1, pluggy-1.6.0 -- C:\Users\CLara\Documents\SIM-UOL Year 3\Semester 2\CM3070 Final project (FP)\assessments_april26\behaviour_app\.venv\Scripts\python.exe
cachedir: .pytest_cache
rootdir: C:\Users\CLara\Documents\SIM-UOL Year 3\Semester 2\CM3070 Final project (FP)\assessments_april26\behaviour_app
plugins: anyio-4.13.0
collecting ... collected 7 items

tests/test_03_system.py::test_web_health_endpoint 
[SYSTEM] Testing /health endpoint...
       PASS: /health returned HTTP 200.
       Service: SenseFuzeAI Live Fusion
PASSED
tests/test_03_system.py::test_web_model_status_endpoint 
[SYSTEM] Testing /model-status endpoint...
       PASS: /model-status returned expected model fields.
       Backend: fallback
       Fusion model status: False
PASSED
tests/test_03_system.py::test_web_app_contains_webcam_calibration_support 
[SYSTEM] Checking web application webcam-calibration support...
       PASS: Web application contains live webcam integration.
       PASS: Webcam-calibrated model artifact exists.
PASSED
tests/test_03_system.py::test_predict_live_rejects_short_text 
[SYSTEM] Testing rejection of insufficient text input...
       PASS: Short text correctly rejected with HTTP 400.
PASSED
tests/test_03_system.py::test_predict_live_rejects_insufficient_keystrokes 
[SYSTEM] Testing rejection of insufficient keystrokes...
       PASS: Insufficient keypresses correctly rejected with HTTP 400.
PASSED
tests/test_03_system.py::test_keystroke_threshold_accepts_twenty_keydowns 
[SYSTEM] Testing twenty-keypress threshold helper...
       PASS: Exactly 20 key-down events were recognised.
PASSED
tests/test_03_system.py::test_web_frontend_contains_webcam_capture_components 
[SYSTEM] Checking browser webcam capture components...
       PASS: Webcam video element exists.
       PASS: Browser frame capture is implemented.
       PASS: Captured image is submitted to /predict_live.
PASSED

============================== 7 passed in 1.01s ==============================
```

### Acceptance Testing

```text
============================= test session starts =============================
platform win32 -- Python 3.11.9, pytest-9.1.1, pluggy-1.6.0 -- C:\Users\CLara\Documents\SIM-UOL Year 3\Semester 2\CM3070 Final project (FP)\assessments_april26\behaviour_app\.venv\Scripts\python.exe
cachedir: .pytest_cache
rootdir: C:\Users\CLara\Documents\SIM-UOL Year 3\Semester 2\CM3070 Final project (FP)\assessments_april26\behaviour_app
plugins: anyio-4.13.0
collecting ... collected 10 items

tests/test_04_acceptance.py::test_project_contains_required_live_interfaces 
[ACCEPTANCE] Checking required live interfaces...
       PASS: All 10 required application/interface files exist.
PASSED
tests/test_04_acceptance.py::test_project_uses_three_pretrained_multimodal_models 
[ACCEPTANCE] Checking pretrained AI model directories...
       PASS: MPNet model available.
       PASS: WavLM model available.
       PASS: CLIP ViT-L/14 model available.
PASSED
tests/test_04_acceptance.py::test_project_contains_webcam_calibration_pipeline 
[ACCEPTANCE] Checking webcam-calibration pipeline...
       PASS: Webcam dataset construction script exists.
       PASS: Webcam-calibrated retraining script exists.
       PASS: Webcam-calibrated classifier exists.
       PASS: Webcam calibration metadata exists.
       PASS: Calibration feature dataset resolved at:
             C:\Users\CLara\Documents\SIM-UOL Year 3\Semester 2\CM3070 Final project (FP)\assessments_april26\behaviour_app\data\processed\webcam_calibration_clip_features.csv
       PASS: Calibration summary resolved at:
             C:\Users\CLara\Documents\SIM-UOL Year 3\Semester 2\CM3070 Final project (FP)\assessments_april26\behaviour_app\data\processed\webcam_calibration_clip_features_summary.json
       PASS: 7 webcam evaluation artifact(s) available.
PASSED
tests/test_04_acceptance.py::test_original_image_model_preserved 
[ACCEPTANCE] Checking original image model preservation...
       PASS: Original image classifier remains untouched.
       PASS: Webcam classifier is stored separately.
PASSED
tests/test_04_acceptance.py::test_webcam_calibration_metadata_is_readable 
[ACCEPTANCE] Checking webcam-calibration metadata readability...
       PASS: Webcam calibration metadata contains 22 fields.
PASSED
tests/test_04_acceptance.py::test_project_contains_multimodal_fusion_model 
[ACCEPTANCE] Checking final multimodal fusion artifact...
       PASS: Fusion classifier exists.
       PASS: Fusion schema contains 2373 features.
PASSED
tests/test_04_acceptance.py::test_final_output_design_supported_in_web_script 
[ACCEPTANCE] Checking final user-facing prediction design...
       PASS: Final behavioural state is displayed.
       PASS: Confidence score is displayed.
       PASS: Probability diagnostics are supported.
PASSED
tests/test_04_acceptance.py::test_web_interface_supports_all_four_behavioural_states 
[ACCEPTANCE] Checking four-class behavioural design...
       PASS: Focused, distracted, fatigued and overloaded are represented.
PASSED
tests/test_04_acceptance.py::test_web_interface_supports_live_webcam_capture 
[ACCEPTANCE] Checking live webcam interface support...
       PASS: Webcam video element is present.
       PASS: Browser webcam capture is implemented.
       PASS: Webcam frames are submitted for inference.
PASSED
tests/test_04_acceptance.py::test_dissertation_ready_evaluation_scripts_exist 
[ACCEPTANCE] Checking evaluation/reporting scripts...
       PASS: Multimodal comparison and evaluation scripts exist.
PASSED

============================= 10 passed in 0.03s ==============================
```

## Final Outcome

All required automated test suites completed successfully.

The automated results provide evidence that the tested unit functions, integration components, web application behaviour, webcam-calibration artifacts, and acceptance requirements are operational.

Model predictive accuracy should additionally be supported by the held-out evaluation metrics generated during model training and webcam calibration.