# Multimodal Evaluation Summary

## Best Cross-Validation Result

- Feature group: `keystroke_text_image`
- Model: `random_forest`
- Number of features: `1558`
- CV accuracy: `1.0000 ± 0.0000`
- CV macro-F1: `1.0000 ± 0.0000`

## Best Held-Out Test Result

- Feature group: `keystroke_image`
- Model: `catboost`
- Number of features: `790`
- Test accuracy: `1.0000`
- Test macro-F1: `1.0000`


## Experiment Metadata

- Dataset: `data\processed\multimodal_features.csv`
- Number of samples: `309`
- CV splits: `5`
- Models: `logistic_regression, random_forest, svm_rbf, xgboost, lightgbm, catboost`



## Baseline Comparison

Baseline feature group: `keystroke_only`

The combined table includes delta columns where the baseline group is available.



## Leakage / Robustness Check

The file `leakage_permutation_check.csv` was found and included.

Permutation macro-F1 should usually be close to chance level.  
For a balanced four-class task, chance-level macro-F1 is approximately `0.25`.

High permutation performance may indicate leakage, duplicated information, or label-coded artifacts.


## Top Combined Results

| feature_group         | model               |   num_features_cv |   cv_accuracy_mean |   cv_accuracy_std |   cv_macro_f1_mean |   cv_macro_f1_std |   fit_time_mean_sec |   score_time_mean_sec | best_model          |   num_features_test |   test_accuracy |   test_macro_f1 |   permutation_macro_f1_mean |   permutation_macro_f1_std |   cv_minus_permutation_macro_f1 |   generalization_gap_macro_f1 |   generalization_gap_accuracy |   overall_test_rank |   delta_test_macro_f1_vs_baseline |   delta_test_accuracy_vs_baseline |   delta_cv_macro_f1_vs_baseline |
|:----------------------|:--------------------|------------------:|-------------------:|------------------:|-------------------:|------------------:|--------------------:|----------------------:|:--------------------|--------------------:|----------------:|----------------:|----------------------------:|---------------------------:|--------------------------------:|------------------------------:|------------------------------:|--------------------:|----------------------------------:|----------------------------------:|--------------------------------:|
| keystroke_image       | catboost            |               790 |             0.9935 |            0.0129 |             0.9935 |            0.0129 |              6.6111 |                0.0267 | catboost            |                 790 |          1      |          1      |                    nan      |                   nan      |                        nan      |                       -0.0065 |                       -0.0065 |                   1 |                            0.0162 |                            0.0161 |                          0.0033 |
| keystroke_text_audio  | lightgbm            |              1599 |             0.9968 |            0.0065 |             0.9969 |            0.0063 |              4.3329 |                0.0919 | lightgbm            |                1599 |          1      |          1      |                    nan      |                   nan      |                        nan      |                       -0.0031 |                       -0.0032 |                   2 |                            0.0162 |                            0.0161 |                          0.0066 |
| text_image            | logistic_regression |              1536 |             0.9968 |            0.0065 |             0.9968 |            0.0065 |              0.1577 |                0.0391 | logistic_regression |                1536 |          1      |          1      |                    nan      |                   nan      |                        nan      |                       -0.0032 |                       -0.0032 |                   3 |                            0.0162 |                            0.0161 |                          0.0065 |
| text_audio_image      | logistic_regression |              2345 |             1      |            0      |             1      |            0      |              0.2816 |                0.0768 | logistic_regression |                2345 |          1      |          1      |                    nan      |                   nan      |                        nan      |                        0      |                        0      |                   4 |                            0.0162 |                            0.0161 |                          0.0097 |
| multimodal_all        | random_forest       |              2367 |             1      |            0      |             1      |            0      |              0.5112 |                0.2195 | random_forest       |                2367 |          1      |          1      |                      0.2447 |                     0.0228 |                          0.7553 |                        0      |                        0      |                   5 |                            0.0162 |                            0.0161 |                          0.0097 |
| keystroke_text        | lightgbm            |               790 |             0.9968 |            0.0065 |             0.9969 |            0.0063 |              1.5715 |                0.0401 | lightgbm            |                 790 |          1      |          1      |                    nan      |                   nan      |                        nan      |                       -0.0031 |                       -0.0032 |                   6 |                            0.0162 |                            0.0161 |                          0.0066 |
| keystroke_audio_image | xgboost             |              1599 |             0.9935 |            0.0079 |             0.9935 |            0.0079 |              7.4393 |                0.082  | xgboost             |                1599 |          0.9839 |          0.9844 |                    nan      |                   nan      |                        nan      |                        0.0092 |                        0.0097 |                   7 |                            0.0006 |                            0      |                          0.0033 |
| text_audio            | logistic_regression |              1577 |             0.9935 |            0.008  |             0.9935 |            0.0079 |              0.1507 |                0.0561 | logistic_regression |                1577 |          0.9839 |          0.9839 |                    nan      |                   nan      |                        nan      |                        0.0097 |                        0.0096 |                   8 |                            0.0001 |                            0      |                          0.0033 |
| keystroke_text_image  | random_forest       |              1558 |             1      |            0      |             1      |            0      |              0.6674 |                0.1094 | random_forest       |                1558 |          0.9839 |          0.9839 |                    nan      |                   nan      |                        nan      |                        0.0161 |                        0.0161 |                   9 |                            0.0001 |                            0      |                          0.0097 |
| text_only             | logistic_regression |               768 |             0.9871 |            0.0121 |             0.9871 |            0.0121 |              0.0798 |                0.0226 | logistic_regression |                 768 |          0.9839 |          0.9839 |                      0.2366 |                     0.0288 |                          0.7505 |                        0.0032 |                        0.0032 |                  10 |                            0.0001 |                            0      |                         -0.0032 |

## Generated Tables

- `ranked_cross_validation_results.csv`
- `ranked_test_results.csv`
- `combined_ranked_results.csv`
- `combined_ranked_results.html`
- `evaluation_summary.json`
- `evaluation_summary.md`

## Generated Plots

- `cv_macro_f1_comparison`
- `test_macro_f1_comparison`
- `accuracy_vs_macro_f1`
- `cv_model_feature_group_heatmap`
- `generalization_gap_macro_f1`
- `num_features_vs_test_macro_f1`
- `permutation_leakage_check`, if permutation results exist

## Interpretation Guide

Use cross-validation macro-F1 for model-selection comparisons across feature groups.

Use held-out test macro-F1 as the final estimate of generalization.

A positive delta for `multimodal_all` over single-modality feature groups supports the empirical value of multimodal fusion.

A large positive generalization gap means cross-validation performance is higher than held-out test performance, which may indicate instability, overfitting, or sensitivity to the train-test split.
