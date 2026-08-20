# SenseFuzeAI Keystroke Dataset Comparison

Label mode: `three`

The EmoSurv behavioural classes used in this comparison are proxy mappings and must not be interpreted as original EmoSurv behavioural-state ground truth.

## Experimental Design

- **A — EmoSurv baseline:** EmoSurv training participants evaluated on held-out EmoSurv participants.
- **B — SenseFuzeAI only:** SenseFuzeAI training groups evaluated on a SenseFuzeAI holdout.
- **C — Augmented to EmoSurv:** EmoSurv training data plus SenseFuzeAI training data evaluated on the exact same EmoSurv holdout used by A.
- **D — EmoSurv to SenseFuzeAI:** EmoSurv training data evaluated on the SenseFuzeAI holdout.
- **E — Augmented to SenseFuzeAI:** EmoSurv training data plus SenseFuzeAI training data evaluated on the exact same SenseFuzeAI holdout used by D.
- **F — SenseFuzeAI to EmoSurv:** SenseFuzeAI training data evaluated on the EmoSurv holdout.

## Results

| Experiment | Accuracy | Balanced Accuracy | Macro Precision | Macro Recall | Macro F1 |
|---|---:|---:|---:|---:|---:|
| A_emosurv_baseline | 0.6050 | 0.6183 | 0.6303 | 0.6183 | 0.5834 |
| B_sensefuzeai_only | 0.9775 | 0.9825 | 0.9762 | 0.9825 | 0.9786 |
| C_augmented_to_emosurv_test | 0.5798 | 0.5988 | 0.5842 | 0.5988 | 0.5586 |
| D_emosurv_to_sensefuzeai | 0.4157 | 0.3889 | 0.5709 | 0.3889 | 0.4053 |
| E_augmented_to_sensefuzeai_test | 0.9775 | 0.9825 | 0.9762 | 0.9825 | 0.9786 |
| F_sensefuzeai_to_emosurv | 0.5294 | 0.3967 | 0.3432 | 0.3967 | 0.3662 |

## Primary Supervisor-Facing Comparison

Experiment A and Experiment C are evaluated on the exact same held-out EmoSurv participants.

- A Macro F1: 0.5834
- C Macro F1: 0.5586
- Macro-F1 change A -> C: -0.0248
- 95% paired group-bootstrap interval: [-0.0445, -0.0060]

A positive delta means that the augmented training dataset achieved a higher macro F1 on the same unseen EmoSurv participant test set. A negative delta indicates that augmentation reduced performance under this experimental configuration.

## Secondary Cross-Dataset Augmentation Comparison

Experiment D and Experiment E are evaluated on the exact same SenseFuzeAI holdout.

- D Macro F1: 0.4053
- E Macro F1: 0.9786
- Macro-F1 change D -> E: +0.5734
- 95% paired group-bootstrap interval: [0.4510, 0.7119]

## Split Information

EmoSurv grouping variable: `participant_id`
SenseFuzeAI grouping variable: `session_id`

The split manifest is frozen after creation and is reused unless `--rebuild-splits` is explicitly supplied.

## Methodological Interpretation

The three-class experiment is the primary conservative dataset-comparison experiment. The four-class experiment is exploratory because EmoSurv does not provide the four SenseFuzeAI behavioural states as original ground-truth labels.