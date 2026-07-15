<!-- readme.md -->

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

The project investigates whether multimodal fusion improves behavioural-state recognition compared with unimodal classification. It includes model training pipelines, standalone live GUIs, a FastAPI web application, live multimodal capture, evaluation scripts, ranked model comparisons, and dissertation-ready summary outputs.

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

Librosa acoustic features
WavLM audio embeddings

It supports uploaded audio files and microphone recording.

Model used:

> models/wavlm-base-plus

Main live GUI:

```bash
python audio_live_gui.py
```

### 4. Image Pipeline

The image pipeline uses CLIP visual embeddings for image, video, or webcam-based behavioural-state inference.

Model used:

> models/clip-vit-large-patch14

Main live GUI:

```bash
python image_live_gui.py
```

### 5. Fusion Pipeline

The fusion pipeline combines all modality features into a single 2367-dimensional feature vector.

Fusion input:

> keystroke + text + audio + image

Main live GUI:

```bash
python live_fusion_gui.py
```

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
> POST predict_live

## Training Scripts 

The project includes updated training and comparison scripts:

```bash
python train_keystroke_baseline.py
python train_keystroke_demo_pipeline.py
python train_text_demo_pipeline.py
python train_audio_demo_pipeline.py
python train_image_demo_pipeline.py
python train_fusion_demo_pipeline.py
python train_multimodal_comparison.py
```

The multimodal comparison script evaluates multiple classifiers, including:

- Logistic Regression
- Random Forest
- SVM RBF
- XGBoost
- LightGBM
- CatBoost where applicable


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

* ranked result tables
* cross-validation and test-set plots
* runtime comparison plots
* permutation leakage plots
* markdown summary
* JSON summary

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

## Model Artifacts 

Expected model directories include:

> models/keystroke_demo/ <br>
> models/text_demo/ <br>
> models/audio_demo/ <br>
> models/image_demo/ <br>
> models/fusion_demo/ <br>
> models/all-mpnet-base-v2/ <br>
> models/wavlm-base-plus/ <br>
> models/clip-vit-large-patch14/

Important model files:

> keystroke_pipeline.joblib <br>
> text_pipeline.joblib <br>
> audio_pipeline.joblib <br>
> image_pipeline.joblib <br>
> fusion_pipeline.joblib <br>
> feature_columns.json

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

## Explainability and Diagnostics

SenseFuzeAI provides diagnostic outputs including:

* class probability distribution
* confidence gap
* confidence level
* feature dimension
* active modalities
* keystroke timing features
* audio feature summaries
* image embedding status
* device information
* runtime
* logged predictions

These diagnostics are intended for evaluation and dissertation analysis, not as the primary user-facing output.

## Technologies Used 

### Machine Learning

* Scikit-learn
* XGBoost
* LightGSM
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

## Current Limitations

1. The dataset is small, with 309 samples.
2. Behavioural states are complex and can overlap in real-world settings.
3. Labels may contain subjective or heuristic assumptions.
4. Audio performance may be affected by background noise.
5. Webcam and image predictions depend on lighting and frame quality.
6. Keystroke behaviour varies across users and typing contexts.
7. Missing modality handling may reduce fusion reliability.
8. The system is designed for academic research and demonstration, not production deployment.

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

## Research Contribution

SenseFuzeAI contributes an end-to-end multimodal behavioural AI framework that integrates:

* behavioural biometrics
* semantic language embeddings
* speech/audio representations
* visual embeddings
* comparative model benchmarking
* real-time fusion inference
* live GUI demonstrations
* FastAPI web deployment
* evaluation and diagnostic reporting

The system demonstrates how multiple pre-trained AI models can be orchestrated across different data domains to achieve a unified behavioural-state prediction goal.

## Author

Student Name: Clara Ng <br>
Student UOL ID: 240662088 <br>
SenseFuzeAI Research Project <br>
Multimodal Behavioural AI System

## License

This project is intended for academic and research purposes.
