# SenseFuzeAI Automated Test Report

Generated: **2026-08-03 17:07:49**

Overall Status: **PASSED**

## Test Objective

The purpose of this automated test suite is to verify the correctness, integration, system behaviour, acceptance requirements, and regression stability of the SenseFuzeAI multimodal behavioural-state prediction system.

The tests specifically include the webcam-calibrated image classifier and the final rolling temporal probability aggregation used by the web application.

## Current Prediction Architecture Under Test

The final web application evaluates keystroke, text, audio and image features through the multimodal fusion classifier. Each live fusion observation produces a four-class probability vector.

The final displayed behavioural state is computed from the arithmetic mean of the latest five fusion probability vectors. The latest raw prediction remains available as diagnostic information.

The webcam-calibrated image classifier is maintained as a separate visual-modality diagnostic and does not replace the final multimodal fusion decision.

## Overall Summary

- Total test suites: 5
- Passed suites: 5
- Failed suites: 0
- Missing suites: 0
- Individual tests passed: 67
- Individual tests failed: 0
- Individual tests skipped: 0
- Pytest errors: 0
- Expected failures: 0
- Unexpected passes: 0
- Total runtime: 14.04 seconds

## Test Suite Results

| Suite | Status | Passed | Failed | Skipped | Errors | Runtime (s) |
|---|---:|---:|---:|---:|---:|---:|
| Unit Testing | PASSED | 14 | 0 | 0 | 0 | 1.59 |
| Integration Testing | PASSED | 14 | 0 | 0 | 0 | 7.64 |
| System Testing | PASSED | 10 | 0 | 0 | 0 | 1.66 |
| Acceptance Testing | PASSED | 13 | 0 | 0 | 0 | 0.35 |
| Smoke / Regression Testing | PASSED | 16 | 0 | 0 | 0 | 2.81 |

## Testing Levels

### Unit Testing

Tests isolated helper functions, probability normalisation, keystroke processing, confidence logic, and temporal probability aggregation.

Test file: `C:\Users\CLara\Documents\SIM-UOL Year 3\Semester 2\CM3070 Final project (FP)\assessments_april26\behaviour_app\tests\test_01_unit.py`

Status: **PASSED**

### Integration Testing

Tests model artifacts, pretrained encoders, webcam calibration, feature schemas, final inference integration, and temporal/webcam backend integration.

Test file: `C:\Users\CLara\Documents\SIM-UOL Year 3\Semester 2\CM3070 Final project (FP)\assessments_april26\behaviour_app\tests\test_02_integration.py`

Status: **PASSED**

### System Testing

Tests FastAPI endpoints, input validation, temporal-session reset, browser webcam integration, and final application-level behaviour.

Test file: `C:\Users\CLara\Documents\SIM-UOL Year 3\Semester 2\CM3070 Final project (FP)\assessments_april26\behaviour_app\tests\test_03_system.py`

Status: **PASSED**

### Acceptance Testing

Tests whether final project requirements, multimodal artifacts, webcam calibration, temporal prediction design, and deployment interfaces are present.

Test file: `C:\Users\CLara\Documents\SIM-UOL Year 3\Semester 2\CM3070 Final project (FP)\assessments_april26\behaviour_app\tests\test_04_acceptance.py`

Status: **PASSED**

### Smoke / Regression Testing

Performs a compact regression check across major SenseFuzeAI files, artifacts, endpoints, webcam calibration, fusion schema, and temporal prediction functionality.

Test file: `C:\Users\CLara\Documents\SIM-UOL Year 3\Semester 2\CM3070 Final project (FP)\assessments_april26\behaviour_app\tests\test_sensefuzeai.py`

Status: **PASSED**

## Detailed Test Output

### Unit Testing

```text
============================= test session starts =============================
platform win32 -- Python 3.11.9, pytest-9.1.1, pluggy-1.6.0 -- C:\Users\CLara\Documents\SIM-UOL Year 3\Semester 2\CM3070 Final project (FP)\assessments_april26\behaviour_app\.venv\Scripts\python.exe
cachedir: .pytest_cache
rootdir: C:\Users\CLara\Documents\SIM-UOL Year 3\Semester 2\CM3070 Final project (FP)\assessments_april26\behaviour_app
plugins: anyio-4.13.0
collecting ... collected 14 items

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
       PASS: Fatigue-related text increases fatigued probability to 0.3600.
PASSED
tests/test_01_unit.py::test_probability_normalisation_sums_to_one 
[UNIT] Testing probability normalisation...
       PASS: Arbitrary non-negative scores are normalised to a valid four-class distribution.
PASSED
tests/test_01_unit.py::test_probability_normalisation_handles_zero_distribution 
[UNIT] Testing zero-distribution fallback...
       PASS: Zero-valued distribution becomes a uniform four-class distribution.
PASSED
tests/test_01_unit.py::test_temporal_probability_aggregation_uses_mean 
[UNIT] Testing temporal mean-probability aggregation...
       PASS: Temporal aggregation computes the arithmetic mean of recent probability vectors.
PASSED
tests/test_01_unit.py::test_temporal_probability_window_keeps_latest_five 
[UNIT] Testing rolling temporal window limit...
       PASS: Temporal history retains only the latest five prediction vectors.
PASSED
tests/test_01_unit.py::test_temporal_sessions_are_isolated 
[UNIT] Testing session-isolated temporal histories...
       PASS: Browser/session probability histories remain independent.
PASSED
tests/test_01_unit.py::test_temporal_session_reset_clears_history 
[UNIT] Testing temporal history reset...
       PASS: Temporal session state is completely cleared.
PASSED
tests/test_01_unit.py::test_prediction_normalisation_returns_temporal_primary_state 
[UNIT] Testing final temporal prediction contract...
       PASS: Final output exposes raw and temporally aggregated prediction information.
PASSED
tests/test_01_unit.py::test_temporal_final_state_can_differ_from_latest_raw_state 
[UNIT] Testing temporal result against latest raw observation...
       PASS: Final temporal result is not simply the latest raw prediction.
PASSED

============================= 14 passed in 1.04s ==============================
```

### Integration Testing

```text
============================= test session starts =============================
platform win32 -- Python 3.11.9, pytest-9.1.1, pluggy-1.6.0 -- C:\Users\CLara\Documents\SIM-UOL Year 3\Semester 2\CM3070 Final project (FP)\assessments_april26\behaviour_app\.venv\Scripts\python.exe
cachedir: .pytest_cache
rootdir: C:\Users\CLara\Documents\SIM-UOL Year 3\Semester 2\CM3070 Final project (FP)\assessments_april26\behaviour_app
plugins: anyio-4.13.0
collecting ... collected 14 items

tests/test_02_integration.py::test_required_model_artifacts_exist 
[INTEGRATION] Checking required model artifacts...
       PASS: 10 required model artifacts/directories found.
PASSED
tests/test_02_integration.py::test_original_and_calibrated_image_models_are_both_preserved 
[INTEGRATION] Checking original and webcam-calibrated image models...
       PASS: Original image model remains preserved.
       PASS: Webcam-calibrated classifier is stored separately.
PASSED
tests/test_02_integration.py::test_webcam_calibrated_image_model_loadable 
[INTEGRATION] Loading webcam-calibrated image classifier...
       PASS: Calibrated classifier loads successfully.
       Model classes: ['distracted', 'fatigued', 'focused', 'overloaded']
PASSED
tests/test_02_integration.py::test_webcam_calibration_metadata_exists_and_valid 
[INTEGRATION] Checking webcam-calibration metadata...
       PASS: Webcam calibration metadata contains 22 fields.
PASSED
tests/test_02_integration.py::test_webcam_calibration_dataset_exists 
[INTEGRATION] Checking webcam-calibration feature dataset...
       Resolved dataset: C:\Users\CLara\Documents\SIM-UOL Year 3\Semester 2\CM3070 Final project (FP)\assessments_april26\behaviour_app\data\processed\webcam_calibration_clip_features.csv
       PASS: Calibration rows = 320
       PASS: CLIP embedding dimension = 768
       PASS: Behavioural classes = ['distracted', 'fatigued', 'focused', 'overloaded']
PASSED
tests/test_02_integration.py::test_webcam_calibration_summary_exists 
[INTEGRATION] Checking webcam-calibration summary...
       Resolved summary: C:\Users\CLara\Documents\SIM-UOL Year 3\Semester 2\CM3070 Final project (FP)\assessments_april26\behaviour_app\data\processed\webcam_calibration_clip_features_summary.json
       PASS: Calibration summary JSON is valid.
PASSED
tests/test_02_integration.py::test_webcam_calibration_dataset_matches_image_feature_schema 
[INTEGRATION] Checking calibration dataset against image feature schema...
       PASS: All 768 image-model feature columns are present in calibration data.
PASSED
tests/test_02_integration.py::test_webcam_evaluation_results_exist 
[INTEGRATION] Checking webcam calibration evaluation artifacts...
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
tests/test_02_integration.py::test_image_feature_schema_uses_768_clip_embeddings 
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
tests/test_02_integration.py::test_web_backend_integrates_calibrated_webcam_classifier 
[INTEGRATION] Checking calibrated webcam classifier integration in web backend...
       PASS: Web backend references and executes the separate webcam-calibrated classifier.
PASSED
tests/test_02_integration.py::test_web_backend_integrates_temporal_probability_aggregation 
[INTEGRATION] Checking temporal probability aggregation integration...
       PASS: Backend contains rolling mean-probability temporal aggregation.
       PASS: Backend contains temporal-session reset.
PASSED

============================= 14 passed in 6.37s ==============================
```

### System Testing

```text
============================= test session starts =============================
platform win32 -- Python 3.11.9, pytest-9.1.1, pluggy-1.6.0 -- C:\Users\CLara\Documents\SIM-UOL Year 3\Semester 2\CM3070 Final project (FP)\assessments_april26\behaviour_app\.venv\Scripts\python.exe
cachedir: .pytest_cache
rootdir: C:\Users\CLara\Documents\SIM-UOL Year 3\Semester 2\CM3070 Final project (FP)\assessments_april26\behaviour_app
plugins: anyio-4.13.0
collecting ... collected 10 items

tests/test_03_system.py::test_web_health_endpoint 
[SYSTEM] Testing /health endpoint...
       PASS: /health returned HTTP 200.
       Temporal window: 5
PASSED
tests/test_03_system.py::test_web_model_status_endpoint 
[SYSTEM] Testing /model-status endpoint...
       PASS: /model-status returned expected model and temporal-fusion fields.
PASSED
tests/test_03_system.py::test_web_app_contains_webcam_calibration_support 
[SYSTEM] Checking web application webcam-calibration support...
       PASS: Web backend integrates the webcam-calibrated classifier.
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
tests/test_03_system.py::test_reset_temporal_endpoint 
[SYSTEM] Testing /reset_temporal endpoint...
       PASS: Temporal state is removed through the HTTP endpoint.
PASSED
tests/test_03_system.py::test_reset_temporal_rejects_empty_session_id 
[SYSTEM] Testing empty temporal session ID...
       PASS: Empty temporal session identifier is rejected.
PASSED
tests/test_03_system.py::test_web_frontend_contains_webcam_capture_components 
[SYSTEM] Checking browser webcam capture components...
       PASS: Webcam video/canvas elements exist.
       PASS: Browser webcam capture is implemented.
PASSED
tests/test_03_system.py::test_web_frontend_contains_temporal_prediction_components 
[SYSTEM] Checking frontend temporal prediction components...
       PASS: Frontend exposes raw and temporally aggregated outputs.
       PASS: Frontend can reset temporal history.
PASSED

============================= 10 passed in 1.21s ==============================
```

### Acceptance Testing

```text
============================= test session starts =============================
platform win32 -- Python 3.11.9, pytest-9.1.1, pluggy-1.6.0 -- C:\Users\CLara\Documents\SIM-UOL Year 3\Semester 2\CM3070 Final project (FP)\assessments_april26\behaviour_app\.venv\Scripts\python.exe
cachedir: .pytest_cache
rootdir: C:\Users\CLara\Documents\SIM-UOL Year 3\Semester 2\CM3070 Final project (FP)\assessments_april26\behaviour_app
plugins: anyio-4.13.0
collecting ... collected 13 items

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
       PASS: Webcam calibration pipeline and evaluation artifacts are present.
PASSED
tests/test_04_acceptance.py::test_original_image_model_preserved 
[ACCEPTANCE] Checking original image model preservation...
       PASS: Original image model remains untouched.
       PASS: Webcam-calibrated model is separate.
PASSED
tests/test_04_acceptance.py::test_webcam_calibration_metadata_is_readable 
[ACCEPTANCE] Checking webcam calibration metadata...
       PASS: Webcam calibration metadata contains 22 fields.
PASSED
tests/test_04_acceptance.py::test_project_contains_multimodal_fusion_model 
[ACCEPTANCE] Checking final multimodal fusion artifact...
       PASS: Fusion classifier exists.
       PASS: Fusion schema contains 2373 features.
PASSED
tests/test_04_acceptance.py::test_final_output_design_supported_in_web_application 
[ACCEPTANCE] Checking final user-facing prediction design...
       PASS: Final behavioural state is displayed.
       PASS: Confidence score is displayed.
       PASS: Probability distribution is displayed.
PASSED
tests/test_04_acceptance.py::test_final_web_prediction_uses_temporal_probability_aggregation 
[ACCEPTANCE] Checking final temporal prediction design...
       PASS: Final result uses temporal mean-probability aggregation.
       PASS: Latest raw result remains visible for diagnostic comparison.
       PASS: Temporal history can be reset.
PASSED
tests/test_04_acceptance.py::test_temporal_window_is_five_predictions 
[ACCEPTANCE] Checking selected temporal window size...
       PASS: Final web application uses a five-observation rolling window.
PASSED
tests/test_04_acceptance.py::test_web_interface_supports_all_four_behavioural_states 
[ACCEPTANCE] Checking four-class behavioural design...
       PASS: Focused, distracted, fatigued and overloaded are represented.
PASSED
tests/test_04_acceptance.py::test_web_interface_supports_live_webcam_capture 
[ACCEPTANCE] Checking live webcam interface support...
       PASS: Browser webcam capture is supported.
PASSED
tests/test_04_acceptance.py::test_web_interface_exposes_separate_webcam_calibrated_result 
[ACCEPTANCE] Checking separate webcam modality result...
       PASS: Webcam-calibrated modality result is displayed separately from final fusion output.
PASSED
tests/test_04_acceptance.py::test_dissertation_ready_evaluation_scripts_exist 
[ACCEPTANCE] Checking evaluation/reporting scripts...
       PASS: Multimodal comparison and evaluation scripts exist.
PASSED

============================= 13 passed in 0.03s ==============================
```

### Smoke / Regression Testing

```text
============================= test session starts =============================
platform win32 -- Python 3.11.9, pytest-9.1.1, pluggy-1.6.0 -- C:\Users\CLara\Documents\SIM-UOL Year 3\Semester 2\CM3070 Final project (FP)\assessments_april26\behaviour_app\.venv\Scripts\python.exe
cachedir: .pytest_cache
rootdir: C:\Users\CLara\Documents\SIM-UOL Year 3\Semester 2\CM3070 Final project (FP)\assessments_april26\behaviour_app
plugins: anyio-4.13.0
collecting ... collected 16 items

tests/test_sensefuzeai.py::test_required_project_files_exist 
[SMOKE] Checking major project files...
       PASS: All 12 major project files exist.
PASSED
tests/test_sensefuzeai.py::test_webcam_calibrated_artifacts_exist 
[SMOKE] Checking webcam-calibrated artifacts...
       PASS: Webcam-calibrated artifacts are present.
PASSED
tests/test_sensefuzeai.py::test_webcam_calibrated_model_loadable 
[SMOKE] Loading webcam-calibrated model...
       PASS: Webcam-calibrated classifier loads successfully.
PASSED
tests/test_sensefuzeai.py::test_webcam_calibration_metadata_valid 
[SMOKE] Checking webcam calibration metadata...
       PASS: Calibration metadata contains 22 fields.
PASSED
tests/test_sensefuzeai.py::test_webcam_evaluation_artifacts_exist 
[SMOKE] Checking webcam evaluation artifacts...
       PASS: 7 webcam evaluation artifact(s) found.
PASSED
tests/test_sensefuzeai.py::test_fusion_feature_schema_valid 
[SMOKE] Checking final fusion feature schema...
       PASS: Fusion schema contains 2373 multimodal features.
PASSED
tests/test_sensefuzeai.py::test_confidence_level_logic 
[SMOKE] Testing confidence-level helper...
       PASS: Confidence-level helper works.
PASSED
tests/test_sensefuzeai.py::test_keystroke_count_extraction 
[SMOKE] Testing keystroke counting...
       PASS: Keystroke counting returned 2.
PASSED
tests/test_sensefuzeai.py::test_fallback_probability_contract 
[SMOKE] Testing fallback prediction contract...
       PASS: Fallback prediction returns four valid classes.
PASSED
tests/test_sensefuzeai.py::test_temporal_probability_aggregation_contract 
[SMOKE] Testing temporal probability contract...
       PASS: Temporal probability history returns a valid four-class distribution.
PASSED
tests/test_sensefuzeai.py::test_web_health_endpoint 
[SMOKE] Testing web health endpoint...
       PASS: /health returned HTTP 200.
PASSED
tests/test_sensefuzeai.py::test_web_model_status_endpoint 
[SMOKE] Testing model-status endpoint...
       PASS: /model-status exposes current model and temporal fields.
PASSED
tests/test_sensefuzeai.py::test_predict_live_rejects_invalid_input 
[SMOKE] Testing invalid live-prediction request...
       PASS: Invalid live input returns HTTP 400.
PASSED
tests/test_sensefuzeai.py::test_reset_temporal_endpoint 
[SMOKE] Testing temporal reset endpoint...
       PASS: /reset_temporal clears session probability history.
PASSED
tests/test_sensefuzeai.py::test_webcam_frontend_capture_support 
[SMOKE] Checking browser webcam capture support...
       PASS: Browser webcam capture is implemented.
PASSED
tests/test_sensefuzeai.py::test_temporal_frontend_support 
[SMOKE] Checking temporal frontend support...
       PASS: Temporal prediction diagnostics are available in the web interface.
PASSED

============================= 16 passed in 2.27s ==============================
```

## Final Outcome

All required automated test suites completed successfully.

The results provide automated evidence that the tested utility functions, model artifacts, webcam calibration pipeline, multimodal integration, FastAPI endpoints, temporal probability aggregation, browser interface, and acceptance requirements are operational.

These software tests do not by themselves establish behavioural-state predictive validity. Model accuracy should therefore also be reported using the held-out evaluation metrics, classification reports, confusion matrices and multimodal evaluation results produced separately by the training/evaluation pipeline.