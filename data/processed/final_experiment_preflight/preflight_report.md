# SenseFuzeAI Final Experiment Pre-Flight Report

**Status:** NOT READY — RESOLVE BLOCKERS FIRST

## Required Protocol

| Item | Required value |
| --- | --- |
| Records | 309 |
| Class counts | {"distracted": 77, "fatigued": 77, "focused": 77, "overloaded": 78} |
| CV | StratifiedKFold(n_splits=5, shuffle=True, random_state=42) |
| Primary metric | macro-F1 |
| Complementary metric | accuracy |
| Variability | fold standard deviation |
| Proposed total features | 2373 |

## Resolved Dataset

Not uniquely resolved.

## Resolved Feature Schema

- Path: `models\fusion_demo\feature_columns.json`
- Source: `json`
- Key: ``
- Feature count: **2373**

## Feature Groups

| Group | Expected | Resolved | Observed | Evidence |
| --- | ---: | --- | ---: | --- |
| keystroke | 22 | False |  |  |
| text | 768 | False |  |  |
| audio | 809 | False |  |  |
| vision | 768 | False |  |  |
| derived | 6 | False |  |  |

## Deployed Random Forest

- Artifact: `models\fusion_demo\fusion_pipeline.joblib`
- Classifier: `sklearn.ensemble._forest.RandomForestClassifier`

```json
{
  "bootstrap": true,
  "ccp_alpha": 0.0,
  "class_weight": "balanced",
  "criterion": "gini",
  "max_depth": null,
  "max_features": "sqrt",
  "max_leaf_nodes": null,
  "max_samples": null,
  "min_impurity_decrease": 0.0,
  "min_samples_leaf": 1,
  "min_samples_split": 2,
  "min_weight_fraction_leaf": 0.0,
  "monotonic_cst": null,
  "n_estimators": 500,
  "n_jobs": -1,
  "oob_score": false,
  "random_state": 42,
  "verbose": 0,
  "warm_start": false
}
```

### Preprocessing

```json
[]
```

## Methodological Issues

- **WARNING — MULTIPLE_DATASET_MATCHES**: More than one dataset exactly matches the proposed Chapter-5 protocol. The authoritative evaluation dataset must be selected explicitly.
- **BLOCKER — FEATURE_GROUP_NOT_RESOLVED_KEYSTROKE**: No explicit repository definition was found for the keystroke feature group. The inspector will not invent feature boundaries.
- **BLOCKER — FEATURE_GROUP_NOT_RESOLVED_TEXT**: No explicit repository definition was found for the text feature group. The inspector will not invent feature boundaries.
- **BLOCKER — FEATURE_GROUP_NOT_RESOLVED_AUDIO**: No explicit repository definition was found for the audio feature group. The inspector will not invent feature boundaries.
- **BLOCKER — FEATURE_GROUP_NOT_RESOLVED_VISION**: No explicit repository definition was found for the vision feature group. The inspector will not invent feature boundaries.
- **BLOCKER — FEATURE_GROUP_NOT_RESOLVED_DERIVED**: No explicit repository definition was found for the derived feature group. The inspector will not invent feature boundaries.
- **BLOCKER — SIX_DERIVED_FEATURES_NOT_RESOLVED**: The six derived predictors have not been resolved explicitly.
- **BLOCKER — DERIVED_DEPENDENCIES_NOT_RESOLVED**: No explicit mapping was found describing which modalities each derived predictor depends on. This is required for a valid leave-one-modality-out experiment.
- **WARNING — MULTIPLE_RF_CONFIGURATIONS**: Multiple fusion Random Forest artifacts with different parameter configurations were found. Confirm which one corresponds to the Chapter-5 complete-fusion result.
- **BLOCKER — DOCUMENTED_FEATURE_COUNT_CONFLICT**: Repository documentation/source contains both 2,367 and 2,373 feature-count references. Determine the authoritative fusion schema before proceeding.

## Proposed Scripts After Approval

- `evaluate_leave_one_modality_out.py` — Five-fold Random Forest retraining with one modality removed at a time, using the authoritative feature schema and derived-feature dependency map.
- `evaluate_missing_modality_robustness.py` — Inference-time modality suppression using the frozen complete-fusion model and exactly the production missing-input representation.
- `benchmark_cpu_inference_latency.py` — Repeated CPU-only latency benchmarking of the canonical inference path with warm-up, repeated measurements, distribution summaries and CSV/JSON exports.
