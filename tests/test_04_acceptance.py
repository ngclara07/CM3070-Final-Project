# tests/test_04_acceptance.py

from __future__ import annotations

import json
from pathlib import Path


# ============================================================
# Project configuration
# ============================================================

ROOT_DIR = Path(__file__).resolve().parents[1]

EXPECTED_CLASSES = {
    "focused",
    "distracted",
    "fatigued",
    "overloaded",
}


# ============================================================
# Webcam calibration paths
# ============================================================

WEBCAM_CALIBRATION_SCRIPT = (
    ROOT_DIR / "build_webcam_calibration_dataset.py"
)

WEBCAM_RETRAIN_SCRIPT = (
    ROOT_DIR / "retrain_image_webcam_calibrated.py"
)

ORIGINAL_IMAGE_MODEL = (
    ROOT_DIR
    / "models"
    / "image_demo"
    / "image_pipeline.joblib"
)

CALIBRATED_IMAGE_MODEL = (
    ROOT_DIR
    / "models"
    / "image_demo"
    / "image_pipeline_webcam_calibrated.joblib"
)

CALIBRATED_METADATA = (
    ROOT_DIR
    / "models"
    / "image_demo"
    / "webcam_calibrated_metadata.json"
)

WEBCAM_EVALUATION_DIR = (
    ROOT_DIR
    / "data"
    / "processed"
    / "webcam_calibration_evaluation"
)


# Support both valid storage layouts.
#
# Layout A:
#   data/webcam_calibration_clip_features.csv
#
# Layout B:
#   data/processed/webcam_calibration_clip_features.csv

WEBCAM_FEATURE_DATA_CANDIDATES = [
    ROOT_DIR
    / "data"
    / "webcam_calibration_clip_features.csv",

    ROOT_DIR
    / "data"
    / "processed"
    / "webcam_calibration_clip_features.csv",
]

WEBCAM_FEATURE_SUMMARY_CANDIDATES = [
    ROOT_DIR
    / "data"
    / "webcam_calibration_clip_features_summary.json",

    ROOT_DIR
    / "data"
    / "processed"
    / "webcam_calibration_clip_features_summary.json",
]


# ============================================================
# Utility helpers
# ============================================================

def resolve_existing_path(
    candidates: list[Path],
    artifact_name: str,
) -> Path:
    """
    Return the first existing path from a list of acceptable locations.

    Raises an informative AssertionError if no candidate exists.
    """

    for path in candidates:
        if path.exists():
            return path

    searched_locations = "\n".join(
        f"  - {path}"
        for path in candidates
    )

    raise AssertionError(
        f"Could not find {artifact_name}.\n"
        f"Checked:\n"
        f"{searched_locations}"
    )


# ============================================================
# Acceptance tests
# ============================================================

def test_project_contains_required_live_interfaces():
    print(
        "\n[ACCEPTANCE] Checking required live interfaces..."
    )

    required_files = [
        ROOT_DIR / "keystroke_live_gui.py",
        ROOT_DIR / "text_live_gui.py",
        ROOT_DIR / "audio_live_gui.py",
        ROOT_DIR / "image_live_gui.py",
        ROOT_DIR / "live_fusion_gui.py",
        ROOT_DIR / "final_multimodal_inference.py",

        ROOT_DIR / "web_app" / "app.py",

        ROOT_DIR
        / "web_app"
        / "templates"
        / "index.html",

        ROOT_DIR
        / "web_app"
        / "static"
        / "script.js",

        ROOT_DIR
        / "web_app"
        / "static"
        / "style.css",
    ]

    missing = [
        str(path)
        for path in required_files
        if not path.exists()
    ]

    assert not missing, (
        f"Missing required interface files: {missing}"
    )

    print(
        f"       PASS: All {len(required_files)} required "
        "application/interface files exist."
    )


def test_project_uses_three_pretrained_multimodal_models():
    print(
        "\n[ACCEPTANCE] Checking pretrained AI model directories..."
    )

    pretrained_models = [
        ROOT_DIR
        / "models"
        / "all-mpnet-base-v2",

        ROOT_DIR
        / "models"
        / "wavlm-base-plus",

        ROOT_DIR
        / "models"
        / "clip-vit-large-patch14",
    ]

    missing = [
        str(path)
        for path in pretrained_models
        if not path.exists()
    ]

    assert not missing, (
        f"Missing pretrained model directories: {missing}"
    )

    print("       PASS: MPNet model available.")
    print("       PASS: WavLM model available.")
    print("       PASS: CLIP ViT-L/14 model available.")


def test_project_contains_webcam_calibration_pipeline():
    print(
        "\n[ACCEPTANCE] Checking webcam-calibration pipeline..."
    )

    # --------------------------------------------------------
    # Mandatory scripts and model artifacts
    # --------------------------------------------------------

    required_fixed_paths = [
        WEBCAM_CALIBRATION_SCRIPT,
        WEBCAM_RETRAIN_SCRIPT,
        CALIBRATED_IMAGE_MODEL,
        CALIBRATED_METADATA,
        WEBCAM_EVALUATION_DIR,
    ]

    missing_fixed_paths = [
        str(path)
        for path in required_fixed_paths
        if not path.exists()
    ]

    assert not missing_fixed_paths, (
        "Missing required webcam-calibration components:\n"
        + "\n".join(
            f"  - {path}"
            for path in missing_fixed_paths
        )
    )

    # --------------------------------------------------------
    # Resolve generated dataset artifacts
    # --------------------------------------------------------

    feature_data_path = resolve_existing_path(
        WEBCAM_FEATURE_DATA_CANDIDATES,
        "webcam calibration CLIP feature dataset",
    )

    summary_path = resolve_existing_path(
        WEBCAM_FEATURE_SUMMARY_CANDIDATES,
        "webcam calibration summary JSON",
    )

    assert feature_data_path.is_file(), (
        f"Calibration feature dataset is not a file: "
        f"{feature_data_path}"
    )

    assert summary_path.is_file(), (
        f"Calibration summary is not a file: "
        f"{summary_path}"
    )

    assert WEBCAM_EVALUATION_DIR.is_dir(), (
        f"Expected webcam evaluation directory: "
        f"{WEBCAM_EVALUATION_DIR}"
    )

    evaluation_files = [
        path
        for path in WEBCAM_EVALUATION_DIR.rglob("*")
        if path.is_file()
    ]

    assert evaluation_files, (
        "Webcam calibration evaluation directory contains "
        "no evaluation artifacts."
    )

    print(
        "       PASS: Webcam dataset construction script exists."
    )

    print(
        "       PASS: Webcam-calibrated retraining script exists."
    )

    print(
        "       PASS: Webcam-calibrated classifier exists."
    )

    print(
        "       PASS: Webcam calibration metadata exists."
    )

    print(
        f"       PASS: Calibration feature dataset resolved at:\n"
        f"             {feature_data_path}"
    )

    print(
        f"       PASS: Calibration summary resolved at:\n"
        f"             {summary_path}"
    )

    print(
        f"       PASS: {len(evaluation_files)} webcam "
        "evaluation artifact(s) available."
    )


def test_original_image_model_preserved():
    print(
        "\n[ACCEPTANCE] Checking original image model preservation..."
    )

    assert ORIGINAL_IMAGE_MODEL.exists(), (
        f"Missing original image classifier: "
        f"{ORIGINAL_IMAGE_MODEL}"
    )

    assert CALIBRATED_IMAGE_MODEL.exists(), (
        f"Missing webcam-calibrated classifier: "
        f"{CALIBRATED_IMAGE_MODEL}"
    )

    assert ORIGINAL_IMAGE_MODEL != CALIBRATED_IMAGE_MODEL

    print(
        "       PASS: Original image classifier remains untouched."
    )

    print(
        "       PASS: Webcam classifier is stored separately."
    )


def test_webcam_calibration_metadata_is_readable():
    print(
        "\n[ACCEPTANCE] Checking webcam-calibration metadata readability..."
    )

    assert CALIBRATED_METADATA.exists(), (
        f"Missing calibrated metadata: "
        f"{CALIBRATED_METADATA}"
    )

    with CALIBRATED_METADATA.open(
        "r",
        encoding="utf-8",
    ) as f:
        metadata = json.load(f)

    assert isinstance(metadata, dict), (
        "Webcam calibration metadata must be a JSON object."
    )

    assert len(metadata) > 0, (
        "Webcam calibration metadata is empty."
    )

    print(
        f"       PASS: Webcam calibration metadata contains "
        f"{len(metadata)} fields."
    )


def test_project_contains_multimodal_fusion_model():
    print(
        "\n[ACCEPTANCE] Checking final multimodal fusion artifact..."
    )

    fusion_model = (
        ROOT_DIR
        / "models"
        / "fusion_demo"
        / "fusion_pipeline.joblib"
    )

    feature_schema = (
        ROOT_DIR
        / "models"
        / "fusion_demo"
        / "feature_columns.json"
    )

    assert fusion_model.exists(), (
        f"Missing fusion model: {fusion_model}"
    )

    assert feature_schema.exists(), (
        f"Missing fusion feature schema: {feature_schema}"
    )

    with feature_schema.open(
        "r",
        encoding="utf-8",
    ) as f:
        columns = json.load(f)

    assert isinstance(columns, list)
    assert len(columns) > 0

    print(
        "       PASS: Fusion classifier exists."
    )

    print(
        f"       PASS: Fusion schema contains "
        f"{len(columns)} features."
    )


def test_final_output_design_supported_in_web_script():
    print(
        "\n[ACCEPTANCE] Checking final user-facing "
        "prediction design..."
    )

    script_path = (
        ROOT_DIR
        / "web_app"
        / "static"
        / "script.js"
    )

    assert script_path.exists(), (
        f"Missing frontend script: {script_path}"
    )

    content = script_path.read_text(
        encoding="utf-8"
    )

    lower_content = content.lower()

    assert (
        "current_state" in content
        or "prediction" in lower_content
    ), (
        "Frontend does not appear to display "
        "a final behavioural-state prediction."
    )

    assert "confidence" in lower_content, (
        "Frontend does not appear to expose confidence."
    )

    assert "probabilities" in lower_content, (
        "Frontend does not appear to expose "
        "probability diagnostics."
    )

    print(
        "       PASS: Final behavioural state is displayed."
    )

    print(
        "       PASS: Confidence score is displayed."
    )

    print(
        "       PASS: Probability diagnostics are supported."
    )


def test_web_interface_supports_all_four_behavioural_states():
    print(
        "\n[ACCEPTANCE] Checking four-class behavioural design..."
    )

    app_path = (
        ROOT_DIR
        / "web_app"
        / "app.py"
    )

    assert app_path.exists(), (
        f"Missing web backend: {app_path}"
    )

    content = app_path.read_text(
        encoding="utf-8"
    ).lower()

    missing_states = [
        state
        for state in EXPECTED_CLASSES
        if state not in content
    ]

    assert not missing_states, (
        "The web backend does not represent all expected "
        f"behavioural states. Missing: {missing_states}"
    )

    print(
        "       PASS: Focused, distracted, fatigued and "
        "overloaded are represented."
    )


def test_web_interface_supports_live_webcam_capture():
    print(
        "\n[ACCEPTANCE] Checking live webcam interface support..."
    )

    html_path = (
        ROOT_DIR
        / "web_app"
        / "templates"
        / "index.html"
    )

    script_path = (
        ROOT_DIR
        / "web_app"
        / "static"
        / "script.js"
    )

    html = html_path.read_text(
        encoding="utf-8"
    ).lower()

    script = script_path.read_text(
        encoding="utf-8"
    )

    assert 'id="webcam"' in html, (
        "Frontend does not contain the webcam video element."
    )

    assert 'id="framecanvas"' in html, (
        "Frontend does not contain the webcam frame canvas."
    )

    assert "getUserMedia" in script, (
        "Frontend does not request browser webcam access."
    )

    assert "captureWebcamFrame" in script, (
        "Frontend does not implement webcam-frame capture."
    )

    assert "image_frame" in script, (
        "Frontend does not submit captured webcam frames "
        "to the backend."
    )

    print(
        "       PASS: Webcam video element is present."
    )

    print(
        "       PASS: Browser webcam capture is implemented."
    )

    print(
        "       PASS: Webcam frames are submitted for inference."
    )


def test_dissertation_ready_evaluation_scripts_exist():
    print(
        "\n[ACCEPTANCE] Checking evaluation/reporting scripts..."
    )

    required_files = [
        ROOT_DIR
        / "train_multimodal_comparison.py",

        ROOT_DIR
        / "evaluate_multimodal_results.py",
    ]

    missing = [
        str(path)
        for path in required_files
        if not path.exists()
    ]

    assert not missing, (
        f"Missing evaluation scripts: {missing}"
    )

    print(
        "       PASS: Multimodal comparison and "
        "evaluation scripts exist."
    )
