<!-- README.md -->

# SenseFuzeAI

**Multimodal Behavioural AI System**

SenseFuzeAI is a multimodal artificial intelligence system for real-time behavioural-state inference. It combines keystroke dynamics, semantic text embeddings, audio representations, visual embeddings, learned multimodal fusion, and temporal probability aggregation to classify a user into one of four behavioural states:

- focused
- distracted
- fatigued
- overloaded

The final system presents one temporally stabilised behavioural-state prediction to the user while retaining the underlying raw fusion prediction, class probabilities, confidence diagnostics, temporal-window information, feature diagnostics, modality metadata, and runtime information for technical evaluation.

---

## Abstract

SenseFuzeAI is a research-oriented multimodal behavioural intelligence platform developed to infer human behavioural states from heterogeneous data streams. The system integrates:

- keystroke timing and rhythm features
- MPNet text embeddings
- WavLM and Librosa audio features
- CLIP image embeddings
- multimodal fusion features

The project investigates whether multimodal fusion improves behavioural-state recognition compared with unimodal classification and whether short-window temporal aggregation can improve prediction stability. It includes model training pipelines, standalone live GUIs, a FastAPI web application, live multimodal acquisition, webcam-specific image calibration, a shared temporal-fusion module, raw-versus-temporal evaluation, group-aware model comparison, automated software testing, and dissertation-ready evaluation outputs.

The final runtime architecture separates raw multimodal inference from temporal post-processing. `final_multimodal_inference.py` produces one stateless four-class probability distribution, while the shared `temporal_fusion.py` module performs temporal aggregation for the desktop application, web application, training comparison, and evaluation workflows.

---

## Research Objective

The project aims to:

1. Investigate behavioural-state recognition using multimodal AI signals.
2. Compare unimodal and multimodal prediction performance.
3. Evaluate multiple machine-learning classifiers across modality combinations.
4. Build a working real-time multimodal behavioural AI system.
5. Investigate whether temporal aggregation of repeated multimodal predictions improves prediction stability.
6. Evaluate raw and temporally aggregated predictions using held-out data and appropriate multiclass metrics.
7. Provide interpretable diagnostics for technical evaluation while presenting one clear behavioural state to the user.
8. Validate the integrated software through unit, integration, system, acceptance, smoke, and runtime testing.

---

## Behavioural States

| Behavioural State | Description |
|---|---|
| Focused | Sustained attention and task engagement |
| Distracted | Reduced attentional stability |
| Fatigued | Low-energy or tired behavioural state |
| Overloaded | High cognitive demand or stress-like overload |

---

## Dataset

The project dataset contains:

| Property | Value |
|---|---:|
| Samples | 309 |
| Classes | 4 |
| Keystroke features | 22 |
| Text features | 768 |
| Audio features | 809 |
| Image features | 768 |
| Fusion features | 2367 |

### Observation and Temporal-Sequence Distinction

The 309 samples represent individual multimodal observations. Temporal fusion is implemented as a post-classification aggregation layer and does not convert these observations into additional training features.

A `session_id` identifies one multimodal observation. Where explicitly collected temporal metadata is available, multiple observations may additionally share a participant/trial/generation grouping and an ordered `sequence_index`.

Historical observations collected before explicit temporal-sequence metadata was introduced remain valid for raw multimodal model training and evaluation. Temporal group identifiers are not fabricated retrospectively for historical samples.

---

## Sample Dataset

To keep this repository lightweight and suitable for GitHub, only a small representative sample dataset is included.

The `sample_data/` directory contains:

- 20 aligned multimodal sessions
- 5 sessions per behavioural class
- balanced class distribution
- audio, image, keystroke and text files for every selected session
- filtered copies of `metadata.csv` and `retroactive_keystroke_features.csv`

The sample dataset is intended for:

- testing the application
- validating the repository structure
- demonstrating the end-to-end processing and inference workflow
- performing lightweight validation and smoke testing

The full research dataset (309 multimodal sessions) is not included in this repository.

The sample dataset should contain only data authorised for academic distribution. Audio recordings, facial images, typed text, keystroke records, sensitive identifiers, and confidential content must be reviewed and appropriately anonymised before publication.

The included class distribution is:

| Behavioural State | Sample Count |
|---|---:|
| Focused | 5 |
| Distracted | 5 |
| Fatigued | 5 |
| Overloaded | 5 |

--- 

## System Architecture

SenseFuzeAI consists of four modality pipelines, a learned multimodal fusion pipeline, and a shared temporal probability-aggregation layer.

The architecture separates feature extraction, raw multimodal classification, and temporal stabilisation:

```text
Keystroke ─┐
Text/MPNet ├──> Multimodal feature vector
Audio/WavLM├──> Raw fusion classifier
Image/CLIP ┘           │
                       ▼
              Four-class probabilities
                       │
                       ▼
               temporal_fusion.py
                       │
              rolling probability mean
                       │
                       ▼
          Final behavioural-state output
```

### 1. Keystroke Pipeline

The keystroke pipeline extracts behavioural typing features such as:

- keydown count
- word count
- typing speed
- inter-key delay
- key hold duration
- pause ratios
- correction behaviour
- rhythm consistency
- burstiness proxy

Main live GUI:

```bash
python keystroke_live_gui.py
```

### 2. Text Pipeline

The text pipeline uses MPNet sentence embeddings to represent semantic and contextual information from user-written text.

Model used: 

> models/all-mpnet-base-v2

Main live GUI:

```bash
python text_live_gui.py
```

### 3. Audio Pipeline

The audio pipeline uses:

- Librosa acoustic features
- WavLM audio embeddings

It supports uploaded audio files and microphone recording.

Silence or near-silence is treated as a valid audio environment and is reported diagnostically. Silence does not automatically force the behavioural prediction to `focused`; the final state remains determined by the learned multimodal classifier and temporal probability aggregation.

Model used:

> models/wavlm-base-plus

Main live GUI:

```bash
python audio_live_gui.py
```

### 4. Image Pipeline

The image pipeline uses CLIP visual embeddings for image, video, and webcam-based behavioural-state inference.

Model used:

> models/clip-vit-large-patch14

In addition to the original image-classification pipeline, the project includes a webcam-calibration workflow designed to improve the alignment between live webcam frames and the behavioural-state image classifier.

The webcam-calibration workflow includes:

- construction of a labelled webcam calibration dataset
- extraction and preparation of webcam frames
- CLIP-based image feature processing
- training and comparison of candidate calibration models
- generation of calibration evaluation reports
- creation of a webcam-calibrated image pipeline for live inference

Calibration scripts:

```bash
python build_webcam_calibration_dataset.py
python retrain_image_webcam_calibrated.py
```

The resulting calibrated model is stored as:

> models/image_demo/image_pipeline_webcam_calibrated.joblib

Calibration metadata is stored as:

> models/image_demo/webcam_calibrated_metadata.json

Main live GUI:

```bash
python image_live_gui.py
```

---

### 5. Fusion Pipeline

The fusion pipeline combines all modality features into a single 2367-dimensional feature vector.

Fusion input:

> keystroke + text + audio + image

Main live GUI:

```bash
python live_fusion_gui.py
```
---

### 6. Temporal Fusion Layer

SenseFuzeAI includes a shared temporal probability-aggregation layer implemented in:

> `temporal_fusion.py`

This module is the canonical source of temporal behaviour for the project. Temporal mathematics is not independently reimplemented in the desktop GUI, web frontend, evaluation script, or training-comparison script.

The temporal layer operates on the four canonical behavioural probabilities:

- focused
- distracted
- fatigued
- overloaded

The canonical temporal window contains the latest five valid multimodal observations.

For observation $t$, the temporally aggregated probability for class $c$ is computed as the arithmetic mean of the available class-probability estimates within the current temporal window:

```math
P_t^{\mathrm{temporal}}(c)
=
\frac{1}{N}
\sum_{i=1}^{N}
P_{t-i+1}^{\mathrm{raw}}(c)
```

where $N$ is the number of available recent predictions in the temporal window, with 1 ≤ N ≤ 5.

The first observation therefore uses one probability vector, the second uses two, and so forth until the five-observation window is full. Once full, the oldest probability vector is discarded when a new observation is added.

The temporal module also provides:

- canonical probability normalisation
- deterministic class ranking
- temporal sample count
- temporal-window status
- final-state selection
- confidence percentage
- second-highest class
- confidence gap
- confidence-level classification
- generation-safe reset behaviour
- stale-generation rejection
- probability-distribution validation

The confidence level is based on the difference between the highest and second-highest temporal class probabilities:

| Confidence Gap | Level |
|---|---|
| >= 0.35 | High |
| >= 0.15 and < 0.35 | Medium |
| < 0.15 | Low |

This confidence level describes class separation within the current prediction. It must not be interpreted as an empirical estimate of model accuracy.

Changing an active audio or visual source resets the temporal history so that observations from unrelated source contexts are not mixed. Explicit temporal reset and full-session reset operations also increment the temporal generation, allowing stale in-flight predictions to be rejected safely.

---

### 6. Webcam Calibration Pipeline

A dedicated webcam-calibration pipeline is included to address the domain difference between the original image-training data and frames captured from a live webcam.

The calibration workflow consists of two principal stages:

1. `build_webcam_calibration_dataset.py` prepares the webcam calibration dataset and associated image features.
2. `retrain_image_webcam_calibrated.py` trains and evaluates candidate image classifiers using the calibration data and produces a webcam-calibrated image pipeline.

Generated calibration outputs include:

- webcam calibration frames
- extracted calibration features
- candidate-model evaluation results
- training summaries and reports
- calibrated model metadata
- the trained webcam-calibrated image pipeline

Evaluation outputs are stored under:

> `data/processed/webcam_calibration_evaluation/`

The calibrated image model can subsequently be used by the live image and multimodal inference workflows to provide image predictions that are better aligned with the live webcam capture environment.

The webcam calibration dataset is intended as a project-specific calibration resource and does not replace the original multimodal research dataset.

---

## User-Facing Prediction Design

The final interface follows a progressive-disclosure design. The primary user-facing output is intentionally concise:

> Current Behavioural State: FOCUSED <br>
> Confidence: 92.40% <br>
> Prediction Confidence: High

The displayed final state is based on the temporally aggregated probability distribution when temporal history is available.

`Confidence` represents the highest current temporal class probability. It does not represent empirically measured model accuracy.

`Prediction Confidence` is derived from the probability gap between the highest and second-highest classes:

- High: confidence gap >= 0.35
- Medium: confidence gap >= 0.15
- Low: confidence gap < 0.15

Technical panels additionally expose:

- raw fusion prediction
- raw fusion probabilities
- temporal prediction
- temporal probabilities
- second-highest class
- confidence gap
- temporal samples
- temporal-window size and fullness
- probability-sum validation
- feature dimension
- active modalities
- audio diagnostics
- calibrated visual diagnostics
- CPU/GPU device
- runtime

This separates the primary behavioural result from implementation and evaluation diagnostics.

---

## Web Application

The project includes a real-time FastAPI web application supporting:

- live text input
- browser keystroke capture
- uploaded audio
- one-shot microphone recording
- persistent fixed audio source until replacement or reset
- static image input
- video input
- live webcam input
- strict four-modality prediction gating
- live multimodal fusion inference
- per-session temporal fusion state
- raw-versus-temporal probability diagnostics
- generation-safe stale-result rejection
- model-readiness monitoring
- temporal reset
- full-session reset
- prediction logging
- technical diagnostics

The browser does not implement the temporal averaging mathematics itself. Temporal history and aggregation are owned by the FastAPI backend through the shared `TemporalFusionEngine`.

Audio is acquired once and retained as the current session audio source until it is replaced or the session is fully reset. It is not repeatedly re-recorded for every 2.5-second prediction cycle.

For visual input:

- static images are stored and reused directly
- uploaded videos are sampled server-side using OpenCV
- webcam frames are captured by the browser and transmitted to the backend
- stopping video/webcam acquisition does not automatically rewrite historical predictions

Main files:

> `web_app/app.py` <br>
> `web_app/templates/index.html` <br>
> `web_app/static/style.css` <br>
> `web_app/static/script.js`

Run the web application from the project root:

```powershell
python -m uvicorn web_app.app:app --host 127.0.0.1 --port 8000
```

Then open: 

> http://127.0.0.1:8000

Useful endpoints:

> GET /health <br>
> GET /model-status <br>
> POST /set_audio_source <br>
> POST /set_visual_image <br>
> POST /set_visual_video <br>
> POST /set_visual_webcam <br>
> POST /stop_visual <br>
> POST /predict_live <br>
> POST /reset_temporal <br>
> POST /full_reset

---

## Temporal Reset and Source Lifecycle

Temporal state is associated with the current multimodal source context.

The temporal history is reset when:

- the active audio source is replaced
- a different static image is selected
- a different video source is selected
- webcam mode becomes a new visual source
- the user explicitly requests a temporal reset
- the user performs a full session reset

Each reset increments a temporal `generation` identifier.

Prediction requests capture the current generation before inference. If the temporal generation changes while inference is running, the stale result is rejected rather than being appended to the new temporal history.

This protects the temporal window from combining predictions generated from different source configurations.

The web application additionally maintains an independent temporal engine for each browser session.

---

## Installation

Create and activate a Python virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Install the required dependencies:

```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The `.venv/` directory is intentionally excluded from version control and should be recreated locally.

---

## Training Scripts 

The project includes updated training and comparison scripts:

```bash
python train_keystroke_baseline.py
python train_keystroke_demo_pipeline.py
python train_text_demo_pipeline.py
python train_audio_demo_pipeline.py
python train_image_demo_pipeline.py
python build_webcam_calibration_dataset.py
python retrain_image_webcam_calibrated.py
python train_fusion_demo_pipeline.py
python train_multimodal_comparison.py
```

The image workflow additionally supports webcam-specific calibration. `build_webcam_calibration_dataset.py` prepares the calibration data, while `retrain_image_webcam_calibrated.py` evaluates candidate classifiers and generates the calibrated image pipeline used by the live webcam workflow.

The multimodal comparison script evaluates multiple classifiers, including:

- Logistic Regression
- Random Forest
- SVM RBF
- XGBoost
- LightGBM
- CatBoost where applicable

The `train_multimodal_comparison.py` also separates model selection from final held-out evaluation. It uses group-aware splitting where suitable grouping metadata is available, compares multiple modality combinations, and evaluates both raw classifier predictions and predictions reconstructed through the canonical `TemporalFusionEngine`.

Important comparison outputs are stored under:

> `data/processed/multimodal_comparison_results/`

and may include:

- `cross_validation_comparison.csv`
- `cross_validation_fold_details.csv`
- `best_model_per_feature_group.csv`
- `test_set_comparison.csv`
- `raw_vs_temporal_test_comparison.csv`
- `leakage_permutation_check.csv`
- held-out prediction CSV files for individual modality combinations
- `multimodal_all_raw_vs_temporal.json`

The held-out prediction files allow the evaluation stage to reproduce temporal aggregation independently rather than relying only on summary scores produced during training.

---

## Reproducing the Sample Dataset

A reproducible dataset-generation script is included.

Run:

```bash
python create_sample_dataset.py
```

The script automatically:

- selects a balanced subset of sessions
- extracts 5 samples from each behavioural class
- copies all associated audio, image, text, and keystroke files
- generates filtered CSV files

The generated `sample_data/` directory contains:

- `audio/`
- `images/`
- `keystrokes/`
- `texts/`
- `metadata.csv`
- `retroactive_keystroke_features.csv`
- `sample_manifest.csv`
- `selected_sessions.csv`
- `class_distribution.csv`

This ensures that the balanced demonstration dataset can be regenerated consistently from the complete aligned dataset. The sample dataset is intended for pipeline validation and demonstration rather than reproduction of the full 309-session evaluation results.

---

## Evaluation

### Multimodal Model Comparison

Run the complete multimodal comparison:

```powershell
python train_multimodal_comparison.py
```

Comparison artifacts are written under:

> `data/processed/multimodal_comparison_results/`

The comparison stage produces held-out prediction files for different modality combinations and classifiers.

For the current all-modality configuration, an example held-out prediction file is:

> `data/processed/multimodal_comparison_results/multimodal_all_catboost_heldout_predictions.csv`

### Raw and Temporal Evaluation

`evaluate_multimodal_results.py` requires an explicit prediction input file.

Example:

```powershell
python evaluate_multimodal_results.py `
    --input "data\processed\multimodal_comparison_results\multimodal_all_catboost_heldout_predictions.csv" `
    --output-dir "evaluation_results\multimodal_all_catboost"
```

To inspect the available evaluator options:

```powershell
python evaluate_multimodal_results.py --help
```

The evaluator reconstructs temporal predictions using the same canonical `TemporalFusionEngine` used by the live applications.

Where ground-truth labels are available, the evaluation workflow can report or derive measures including:

- accuracy
- balanced accuracy
- macro F1
- confusion matrices
- multiclass Brier score
- expected calibration error
- raw fusion performance
- temporal fusion performance using all available temporal samples
- temporal fusion performance once the full five-observation window is reached
- raw-versus-temporal state switching/stability
- probability-distribution validation
- runtime statistics where available
- temporal-parity checks where logged temporal outputs are available

Three evaluation scopes are distinguished:

1. `raw_fusion` — the original stateless multimodal classifier output.
2. `temporal_fusion_all_samples` — temporal aggregation from the first observation onward.
3. `temporal_fusion_full_window` — evaluation restricted to observations for which the canonical five-observation window is full.

Live prediction logs under:

> `web_app/output/`

may also be analysed for runtime, probability validity, temporal consistency, and system behaviour. Live logs should not be interpreted as behavioural-accuracy evidence unless they contain independently supplied ground-truth labels.

---

## Automated Software Testing

SenseFuzeAI includes automated tests at multiple levels:

- unit tests
- integration tests
- system tests
- acceptance tests
- SenseFuzeAI smoke/regression tests

Test files:

> `tests/test_01_unit.py` <br>
> `tests/test_02_integration.py` <br>
> `tests/test_03_system.py` <br>
> `tests/test_04_acceptance.py` <br>
> `tests/test_sensefuzeai.py`

Run the complete pytest suite:

```powershell
python -m pytest tests -v
```

Run the acceptance tests only:

```powershell
python -m pytest tests/test_04_acceptance.py -v
```

Run the SenseFuzeAI smoke suite only:

```powershell
python -m pytest tests/test_sensefuzeai.py -v
```

The project-level test runner can also be executed with:

```powershell
python run_all_tests.py
```

A stronger pre-submission run is:

```powershell
python run_all_tests.py --with-model-smoke --require-pytest
```

If Node.js is installed and JavaScript syntax validation should be mandatory:

```powershell
python run_all_tests.py --with-model-smoke --require-pytest --require-node
```

JavaScript syntax can also be checked directly using:

```powershell
node --check web_app/static/script.js
```

### Saving Test Evidence

Timestamped test evidence can be written to `test_reports/`.

For example:

```powershell
$stamp = Get-Date -Format "yyyyMMdd_HHmmss"

python -m pytest tests/test_sensefuzeai.py -v `
    --junitxml="test_reports\sensefuzeai_test_report_$stamp.xml" `
    2>&1 | Tee-Object -FilePath "test_reports\sensefuzeai_test_report_$stamp.txt"
```

This preserves both a human-readable terminal report and a machine-readable JUnit XML report.

The automated acceptance suite verifies architectural requirements including the presence of the three pretrained model domains, stateless raw inference, shared temporal fusion, per-session web temporal state, four-modality acquisition, reset behaviour, raw-versus-temporal diagnostics, evaluation methodology, and the automated test structure.

---

## Live GUI Applications

Standalone demonstration GUIs are provided for testing and dissertation demonstration.

```bash
python keystroke_live_gui.py
python text_live_gui.py
python audio_live_gui.py
python image_live_gui.py
python live_fusion_gui.py
```

The standalone modality GUIs provide direct modality-level demonstrations, while `live_fusion_gui.py` provides the complete four-modality real-time system.

The multimodal live GUI applies the canonical shared `TemporalFusionEngine` after raw multimodal inference. It requires all four modalities to be ready before producing a multimodal observation and evaluates the input periodically while the live session is active.

> 1. System readiness
> 2. Input/capture controls
> 3. One current behavioural-state prediction
> 4. Confidence percentage
> 5. Prediction confidence level
> 6. Technical probability and diagnostic details
> 7. Prediction logging

Logs are saved under:

> data/processed/

For the multimodal live GUI:

- minimum text length: 20 characters
- minimum captured keydowns: 20
- prediction interval: approximately 2.5 seconds
- temporal probability window: 5 observations
- microphone recording duration: 10 seconds
- target audio sampling rate: 16 kHz

Changing an audio or visual source resets the temporal probability history so that predictions from unrelated source contexts are not mixed.

---

## Model Artifacts 

Expected model directories include:

- `models/keystroke_demo/`
- `models/text_demo/`
- `models/audio_demo/`
- `models/image_demo/`
- `models/fusion_demo/`
- `models/all-mpnet-base-v2/`
- `models/wavlm-base-plus/`
- `models/clip-vit-large-patch14/`

Important model files:

> keystroke_pipeline.joblib <br>
> text_pipeline.joblib <br>
> audio_pipeline.joblib <br>
> image_pipeline.joblib <br>
> image_pipeline_webcam_calibrated.joblib <br>
> webcam_calibrated_metadata.json <br>
> fusion_pipeline.joblib <br>
> feature_columns.json

The webcam-calibrated image pipeline is an additional project-trained model intended for live webcam inference. It complements the original image pipeline rather than replacing the underlying CLIP embedding model.

---

## Pre-trained Models

The repository expects several pre-trained embedding models.

The repository retains small model-support files, including configurations, tokenizers, metadata, and demonstration pipelines. Large downloaded pretrained weight files, particularly `*.safetensors`, are intentionally excluded because they exceed GitHub's ordinary repository-size limits.

The required pretrained models must therefore be downloaded or restored locally before running workflows that depend on their full weights.

Download or place the following models inside the `models/` directory before running the system:

- all-mpnet-base-v2
- wavlm-base-plus
- clip-vit-large-patch14

The application loads these models from their expected local directories under `models/`.

Model-download utilities are provided in the project root, including:

- `download_mpnet_model.py`
- `download_wavlm_model.py`
- `download_image_model.py`
- `download_whisper.py`
- `download_yamnet.py`
- `download_audio_model.py`
- `download_text_model.py`

---

## Inference Script

The canonical multimodal inference implementation is:

> `final_multimodal_inference.py`

The `FinalMultimodalInference` class performs one stateless multimodal observation at a time. It:

1. extracts keystroke features;
2. generates MPNet text embeddings;
3. generates WavLM/audio features;
4. generates CLIP image features;
5. applies optional webcam-calibration diagnostics;
6. constructs the fusion feature vector in the saved training schema order;
7. executes the trained fusion classifier;
8. returns a normalised four-class raw probability distribution.

Temporal history is deliberately **not stored inside this class**. Temporal aggregation is applied externally by `temporal_fusion.py`, allowing desktop, web, training, and evaluation workflows to share the same implementation.

Inspect command-line usage with:

```powershell
python final_multimodal_inference.py --help
```

The inference class can also be instantiated directly by the desktop GUI and FastAPI backend.

This separation establishes the following architecture:

```text
FinalMultimodalInference
        │
        │ one observation
        ▼
Raw four-class probabilities
        │
        ▼
TemporalFusionEngine
        │
        ▼
Final temporally stabilised prediction
```

---

## Explainability and Diagnostics

SenseFuzeAI provides diagnostic outputs including:

- raw behavioural-state prediction
- raw four-class probability distribution
- temporally aggregated behavioural-state prediction
- temporal four-class probability distribution
- confidence percentage
- second-highest class
- confidence gap
- confidence level
- temporal sample count
- temporal-window size and fullness
- probability-sum validation
- temporal generation
- feature dimension
- active modalities
- keystroke timing features
- audio feature summaries
- audio environment diagnostics
- CLIP/image feature status
- webcam-calibrated visual prediction
- CPU/GPU device information
- inference runtime
- logged predictions

These diagnostics are intended for evaluation and dissertation analysis, not as the primary user-facing output.

Confidence and confidence-gap diagnostics describe the current probability distribution; they are not substitutes for held-out empirical accuracy, balanced accuracy, macro F1, calibration, or other evaluation measures.

---

## Technologies Used 

### Machine Learning

* Scikit-learn
* XGBoost
* LightGBM
* CatBoost
* Joblib

### Deep Learning and Embeddings

* PyTorch
* HuggingFace Transformers
* SentenceTransformers
* MPNet
* WavLM
* CLIP

### Audio Processing

* Librosa
* SoundDevice
* SoundFile

### Image and Video Processing

* OpenCV
* Pillow
* CLIP

### Web Application

* FastAPI
* Uvicorn
* Jinja2
* HTML5
* CSS
* JavaScript

### GUI Applications

* Tkinter

---

## Current Limitations

1. The dataset is relatively small, with 309 multimodal observations.
2. Behavioural states are complex, subjective, and can overlap in real-world settings.
3. Behavioural labels may reflect self-report, guided-condition, or heuristic assumptions and therefore contain unavoidable measurement uncertainty.
4. Audio performance can be affected by microphone characteristics, room acoustics, background noise, and browser/device processing.
5. Webcam and image predictions remain sensitive to lighting, camera position, background conditions, frame quality, and differences between calibration and deployment environments.
6. Keystroke behaviour varies substantially across users, keyboards, tasks, and typing contexts.
7. The final live fusion workflow intentionally requires all four modalities to be available; it does not currently provide a learned missing-modality fusion strategy.
8. The five-observation temporal mean improves stability but can delay recognition of genuine rapid behavioural transitions.
9. Historical observations collected before explicit temporal trial metadata was introduced cannot automatically be treated as controlled temporal sequences.
10. Strong claims about temporal accuracy require ordered, labelled and appropriately grouped evaluation sequences rather than isolated live demonstrations.
11. Browser webcam and microphone acquisition cannot be expected to be byte-identical to native desktop OpenCV/SoundDevice acquisition because browser and operating-system processing differ.
12. The complete research dataset is intentionally excluded from the GitHub repository due to repository-size and data-governance considerations.
13. Only a balanced demonstration dataset is distributed for pipeline validation, smoke testing, and repository-level reproducibility.
14. Large pretrained embedding-model weights must be downloaded or restored separately before workflows that require them can run.
15. The webcam-calibrated image model is based on a limited project-specific calibration dataset and should not be interpreted as demonstrating generalisation across different users, cameras, environments, or deployment conditions.
16. SenseFuzeAI is an academic research prototype and is not intended for clinical, safety-critical, employment-monitoring, or production decision-making.

---

## Future Improvements

Potential future work includes:

* larger and more diverse dataset collection
* participant-independent and longitudinal evaluation
* additional controlled temporal-sequence data
* user-specific calibration
* * learned temporal sequence models beyond the current five-observation arithmetic probability mean
* adaptive temporal-window selection
* multimodal transformer architectures
* uncertainty-aware decision logic
* learned missing-modality fusion
* improved probability calibration
* real-time continuous inference optimisation
* broader latency and resource benchmarking
* expanded user testing and interface iteration
* privacy-preserving behavioural modelling
* evaluation across additional cameras, microphones, keyboards, users, and physical environments

---

## Research Contribution

SenseFuzeAI contributes an end-to-end multimodal behavioural AI framework that integrates:

* behavioural keystroke biometrics
* semantic MPNet language embeddings
* WavLM/audio representations
* CLIP visual embeddings
* webcam-specific visual calibration
* learned multimodal feature-level fusion
* a shared five-observation temporal probability-aggregation layer
* stateless raw inference separated from temporal state management
* raw-versus-temporal held-out evaluation
* comparative modality and classifier benchmarking
* group-aware experimental splitting
* real-time desktop fusion inference
* FastAPI web deployment
* generation-safe temporal reset handling
* automated unit, integration, system, acceptance, and smoke testing
* evaluation and diagnostic reporting

The system demonstrates how multiple pretrained AI models from different data domains can be orchestrated with engineered behavioural features, learned multimodal fusion, temporal post-processing, software integration, and evaluation to achieve a unified behavioural-state prediction goal.

---

## Repository Structure

```text
.
├── data/
│   ├── processed/
│   │   ├── keystroke_baseline_results/
│   │   ├── multimodal_comparison_results/
│   │   ├── multimodal_evaluation_summary/
│   │   └── webcam_calibration_evaluation/
│   └── session_aligned/
├── evaluation_results/
├── models/
│   ├── all-mpnet-base-v2/
│   ├── wavlm-base-plus/
│   ├── clip-vit-large-patch14/
│   ├── fusion_demo/
│   └── image_demo/
├── sample_data/
├── test_reports/
├── tests/
│   ├── test_01_unit.py
│   ├── test_02_integration.py
│   ├── test_03_system.py
│   ├── test_04_acceptance.py
│   └── test_sensefuzeai.py
├── utils/
├── web_app/
│   ├── output/
│   ├── static/
│   │   ├── script.js
│   │   └── style.css
│   ├── templates/
│   │   └── index.html
│   └── app.py
├── build_multimodal_dataset.py
├── build_webcam_calibration_dataset.py
├── collect_multimodal_session.py
├── create_sample_dataset.py
├── evaluate_multimodal_results.py
├── final_multimodal_inference.py
├── live_fusion_gui.py
├── retrain_image_webcam_calibrated.py
├── run_all_tests.py
├── temporal_fusion.py
├── train_multimodal_comparison.py
├── train_*.py
├── *_live_gui.py
├── requirements.txt
└── README.md
```

The repository includes application source code, model-training and comparison pipelines, the canonical temporal-fusion implementation, evaluation utilities, automated tests, selected model-support files, processed evaluation artifacts, and a balanced demonstration dataset.

The complete research dataset, large pretrained model weight files, local virtual environments, caches, temporary browser/server uploads, archives, and unnecessary generated artefacts are intentionally excluded where appropriate.

---

## Final Runtime Architecture

The final SenseFuzeAI runtime follows a deliberate separation of responsibilities:

```text
Pretrained / learned modality encoders
          │
          ▼
Keystroke + MPNet + WavLM + CLIP features
          │
          ▼
FinalMultimodalInference
(stateless raw inference)
          │
          ▼
Raw four-class probability vector
          │
          ▼
TemporalFusionEngine
(shared canonical implementation)
          │
          ▼
Five-observation temporal probability distribution
          │
          ▼
Focused / Distracted / Fatigued / Overloaded

---

## Author

Student Name: Clara Ng <br>
SenseFuzeAI Research Project <br>
Multimodal Behavioural AI System

---

## License

This project is intended for academic and research purposes.
