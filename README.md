<!-- README.md -->

# SenseFuzeAI

**Multimodal Behavioural AI System**

SenseFuzeAI is a multimodal artificial intelligence system for real-time behavioural-state inference. It combines keystroke dynamics, semantic text embeddings, audio representations, visual embeddings, and multimodal fusion to classify a user into one of four behavioural states:

- focused
- distracted
- fatigued
- overloaded

The final system is designed to present one clear behavioural-state prediction to the user, while keeping confidence scores, probability distributions, feature diagnostics, and modality metadata available for technical evaluation.

---

## Abstract

SenseFuzeAI is a research-oriented multimodal behavioural intelligence platform developed to infer human behavioural states from heterogeneous data streams. The system integrates:

- keystroke timing and rhythm features
- MPNet text embeddings
- WavLM and Librosa audio features
- CLIP image embeddings
- multimodal fusion features

The project investigates whether multimodal fusion improves behavioural-state recognition compared with unimodal classification. It includes model training pipelines, standalone live GUIs, a FastAPI web application, live multimodal capture, webcam-specific image calibration, evaluation scripts, ranked model comparisons, and dissertation-ready summary outputs.

---

## Research Objective

The project aims to:

1. Investigate behavioural-state recognition using multimodal AI signals.
2. Compare unimodal and multimodal prediction performance.
3. Evaluate multiple machine-learning classifiers across modalities.
4. Build a working real-time multimodal behavioural AI system.
5. Provide interpretable diagnostics for technical evaluation while presenting one clear behavioural state to the user.

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

SenseFuzeAI consists of four unimodal pipelines and one fusion pipeline.

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

The final interface philosophy is: 

> Current Behavioural State: FOCUSED <br>
> Confidence: 92.40% <br>
> Prediction Confidence: High

The system predicts one final behavioural state as the main output.

Technical details such as:

- second-highest class
- confidence gap
- probability distribution
- feature dimension
- active modalities
- CPU/GPU device
- runtime
- diagnostic features

are shown only in technical panels or expandable sections.

---

## Web Application

The project includes a real-time FastAPI web application with:

- live text input
- keystroke capture
- webcam capture
- microphone chunks
- live fusion endpoint
- model readiness panel
- confidence display
- reset controls
- prediction logging
- technical diagnostics

Main files:

> web_app/app.py <br>
> web_app/templates/index.html <br>
> web_app/static/style.css <br>
> web_app/static/script.js

Run the web app: 

```bash
cd web_app
python app.py
```

Then open: 

> http://127.0.0.1:8000

Useful endpoints:

> GET /health <br>
> GET /model-status <br>
> POST /predict_live

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

Run the final multimodal comparison:

```bash
python train_multimodal_comparison.py
```

Then generate evaluation outputs:

```bash
python evaluate_multimodal_results.py
```

Outputs are saved to:

> data/processed/multimodal_evaluation_summary/

Generated artifacts include:

- ranked result tables
- cross-validation and test-set plots
- runtime comparison plots
- permutation leakage plots
- markdown summary
- JSON summary

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

Each GUI now follows the same design pattern:

> 1. System readiness
> 2. Input/capture controls
> 3. One current behavioural-state prediction
> 4. Confidence percentage
> 5. Prediction confidence level
> 6. Technical probability and diagnostic details
> 7. Prediction logging

Logs are saved under:

> data/processed/

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

The final multimodal inference script is:

```bash
python final_multimodal_inference.py
```

Example usage:

```bash
python final_multimodal_inference.py ^
  --keystroke_json path/to/keystroke.json ^
  --text "sample behavioural text" ^
  --audio path/to/audio.wav ^
  --image path/to/image.jpg
```

---

## Explainability and Diagnostics

SenseFuzeAI provides diagnostic outputs including:

- class probability distribution
- confidence gap
- confidence level
- feature dimension
- active modalities
- keystroke timing features
- audio feature summaries
- image embedding status
- device information
- runtime
- logged predictions

These diagnostics are intended for evaluation and dissertation analysis, not as the primary user-facing output.

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

1. The dataset is small, with 309 samples.
2. Behavioural states are complex and can overlap in real-world settings.
3. Labels may contain subjective or heuristic assumptions.
4. Audio performance may be affected by background noise.
5. Webcam and image predictions remain sensitive to lighting, camera position, background conditions, frame quality, and differences between calibration and deployment environments.
6. Keystroke behaviour varies across users and typing contexts.
7. Missing modality handling may reduce fusion reliability.
8. The system is designed for academic research and demonstration, not production deployment.
9. The complete research dataset is intentionally excluded from the GitHub repository due to repository size limitations.
10. Only a balanced demonstration dataset is distributed for pipeline validation, smoke testing, and repository-level reproducibility.
11. Some pre-trained embedding models must be downloaded separately before training or inference.
12. The webcam-calibrated image model is based on a limited project-specific calibration dataset and should not be interpreted as demonstrating generalisation across different users, cameras, environments, or deployment conditions.

---

## Future Improvements 

Potential future work includes:

* larger dataset collection
* user-specific calibration
* temporal sequence modelling
* multimodal transformer architectures
* uncertainty-aware decision logic
* improved missing-modality fusion
* real-time continuous inference optimisation
* user testing and interface refinement
* privacy-preserving behavioural modelling

---

## Research Contribution

SenseFuzeAI contributes an end-to-end multimodal behavioural AI framework that integrates:

* behavioural biometrics
* semantic language embeddings
* speech/audio representations
* visual embeddings
* webcam-specific visual calibration for live image inference
* comparative model benchmarking
* real-time fusion inference
* live GUI demonstrations
* FastAPI web deployment
* evaluation and diagnostic reporting

The system demonstrates how multiple pre-trained AI models can be orchestrated across different data domains to achieve a unified behavioural-state prediction goal.

---

## Repository Structure

```text
.
├── data/
│   └── processed/
│       ├── keystroke_baseline_results/
│       ├── multimodal_comparison_results/
│       ├── multimodal_evaluation_summary/
│       └── webcam_calibration_evaluation/
├── models/
│   ├── fusion_demo/
│   └── image_demo/
├── sample_data/
├── test_reports/
├── tests/
├── utils/
├── web_app/
├── build_webcam_calibration_dataset.py
├── create_sample_dataset.py
├── final_multimodal_inference.py
├── retrain_image_webcam_calibrated.py
├── requirements.txt
├── run_all_tests.py
├── train_*.py
├── *_live_gui.py
└── README.md
```

The repository includes source code, configuration files, selected model-support files, selected processed evaluation outputs, and a balanced sample dataset for demonstration and reproducibility. The complete research dataset, large pretrained weight files, local virtual environments, caches, archives, and other unnecessary generated artefacts are intentionally excluded.

---

## Author

Student Name: Clara Ng <br>
SenseFuzeAI Research Project <br>
Multimodal Behavioural AI System

---

## License

This project is intended for academic and research purposes.
