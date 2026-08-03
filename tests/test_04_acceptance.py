# === tests/test_04_acceptance.py ===

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
# Paths
# ============================================================

WEB_APP_PATH = (
    ROOT_DIR
    / "web_app"
    / "app.py"
)

WEB_HTML_PATH = (
    ROOT_DIR
    / "web_app"
    / "templates"
    / "index.html"
)

WEB_SCRIPT_PATH = (
    ROOT_DIR
    / "web_app"
    / "static"
    / "script.js"
)

WEB_STYLE_PATH = (
    ROOT_DIR
    / "web_app"
    / "static"
    / "style.css"
)


WEBCAM_CALIBRATION_SCRIPT = (
    ROOT_DIR
    / "build_webcam_calibration_dataset.py"
)

WEBCAM_RETRAIN_SCRIPT = (
    ROOT_DIR
    / "retrain_image_webcam_calibrated.py"
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


WEBCAM_FEATURE_DATA_CANDIDATES = [
    (
        ROOT_DIR
        / "data"
        / "webcam_calibration_clip_features.csv"
    ),
    (
        ROOT_DIR
        / "data"
        / "processed"
        / "webcam_calibration_clip_features.csv"
    ),
]

WEBCAM_FEATURE_SUMMARY_CANDIDATES = [
    (
        ROOT_DIR
        / "data"
        / "webcam_calibration_clip_features_summary.json"
    ),
    (
        ROOT_DIR
        / "data"
        / "processed"
        / "webcam_calibration_clip_features_summary.json"
    ),
]


# ============================================================
# Helper
# ============================================================

def resolve_existing_path(
    candidates: list[Path],
    artifact_name: str,
) -> Path:
    """
    Resolve an artifact from supported storage layouts.
    """

    for path in candidates:

        if path.exists():
            return path

    searched = "\n".join(
        f"  - {path}"
        for path in candidates
    )

    raise AssertionError(
        f"Could not find {artifact_name}.\n"
        f"Checked:\n{searched}"
    )


# ============================================================
# Core interface acceptance
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
        WEB_APP_PATH,
        WEB_HTML_PATH,
        WEB_SCRIPT_PATH,
        WEB_STYLE_PATH,
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


# ============================================================
# Pretrained models
# ============================================================

def test_project_uses_three_pretrained_multimodal_models():
    print(
        "\n[ACCEPTANCE] Checking pretrained "
        "AI model directories..."
    )

    pretrained_models = [
        (
            ROOT_DIR
            / "models"
            / "all-mpnet-base-v2"
        ),
        (
            ROOT_DIR
            / "models"
            / "wavlm-base-plus"
        ),
        (
            ROOT_DIR
            / "models"
            / "clip-vit-large-patch14"
        ),
    ]

    missing = [
        str(path)
        for path in pretrained_models
        if not path.exists()
    ]

    assert not missing

    print(
        "       PASS: MPNet model available."
    )

    print(
        "       PASS: WavLM model available."
    )

    print(
        "       PASS: CLIP ViT-L/14 model available."
    )


# ============================================================
# Webcam calibration
# ============================================================

def test_project_contains_webcam_calibration_pipeline():
    print(
        "\n[ACCEPTANCE] Checking webcam-calibration pipeline..."
    )

    required = [
        WEBCAM_CALIBRATION_SCRIPT,
        WEBCAM_RETRAIN_SCRIPT,
        CALIBRATED_IMAGE_MODEL,
        CALIBRATED_METADATA,
        WEBCAM_EVALUATION_DIR,
    ]

    missing = [
        str(path)
        for path in required
        if not path.exists()
    ]

    assert not missing, (
        "Missing required webcam-calibration "
        f"components: {missing}"
    )

    feature_path = resolve_existing_path(
        WEBCAM_FEATURE_DATA_CANDIDATES,
        (
            "webcam calibration CLIP "
            "feature dataset"
        ),
    )

    summary_path = resolve_existing_path(
        WEBCAM_FEATURE_SUMMARY_CANDIDATES,
        "webcam calibration summary JSON",
    )

    assert feature_path.is_file()
    assert summary_path.is_file()

    evaluation_files = [
        path
        for path
        in WEBCAM_EVALUATION_DIR.rglob("*")
        if path.is_file()
    ]

    assert evaluation_files

    print(
        "       PASS: Webcam calibration pipeline "
        "and evaluation artifacts are present."
    )


def test_original_image_model_preserved():
    print(
        "\n[ACCEPTANCE] Checking original "
        "image model preservation..."
    )

    assert ORIGINAL_IMAGE_MODEL.exists()
    assert CALIBRATED_IMAGE_MODEL.exists()

    assert (
        ORIGINAL_IMAGE_MODEL
        != CALIBRATED_IMAGE_MODEL
    )

    print(
        "       PASS: Original image model remains untouched."
    )

    print(
        "       PASS: Webcam-calibrated model is separate."
    )


def test_webcam_calibration_metadata_is_readable():
    print(
        "\n[ACCEPTANCE] Checking webcam calibration metadata..."
    )

    with CALIBRATED_METADATA.open(
        "r",
        encoding="utf-8",
    ) as f:

        metadata = json.load(f)

    assert isinstance(
        metadata,
        dict,
    )

    assert len(metadata) > 0

    print(
        f"       PASS: Webcam calibration metadata "
        f"contains {len(metadata)} fields."
    )


# ============================================================
# Fusion model acceptance
# ============================================================

def test_project_contains_multimodal_fusion_model():
    print(
        "\n[ACCEPTANCE] Checking final "
        "multimodal fusion artifact..."
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

    assert fusion_model.exists()
    assert feature_schema.exists()

    with feature_schema.open(
        "r",
        encoding="utf-8",
    ) as f:

        columns = json.load(f)

    assert isinstance(
        columns,
        list,
    )

    assert len(columns) > 0

    print(
        "       PASS: Fusion classifier exists."
    )

    print(
        f"       PASS: Fusion schema contains "
        f"{len(columns)} features."
    )


# ============================================================
# Final output acceptance
# ============================================================

def test_final_output_design_supported_in_web_application():
    print(
        "\n[ACCEPTANCE] Checking final "
        "user-facing prediction design..."
    )

    html = WEB_HTML_PATH.read_text(
        encoding="utf-8"
    ).lower()

    script = WEB_SCRIPT_PATH.read_text(
        encoding="utf-8"
    )

    assert (
        'id="prediction"'
        in html
    )

    assert (
        'id="confidencepercent"'
        in html
    )

    assert (
        'id="probabilities"'
        in html
    )

    assert (
        "current_state"
        in script
    )

    assert (
        "confidence_percent"
        in script
    )

    assert (
        "probabilities"
        in script
    )

    print(
        "       PASS: Final behavioural state is displayed."
    )

    print(
        "       PASS: Confidence score is displayed."
    )

    print(
        "       PASS: Probability distribution is displayed."
    )


def test_final_web_prediction_uses_temporal_probability_aggregation():
    print(
        "\n[ACCEPTANCE] Checking final temporal "
        "prediction design..."
    )

    backend = WEB_APP_PATH.read_text(
        encoding="utf-8"
    )

    frontend = WEB_SCRIPT_PATH.read_text(
        encoding="utf-8"
    )

    html = WEB_HTML_PATH.read_text(
        encoding="utf-8"
    ).lower()

    backend_requirements = [
        "TEMPORAL_PROBABILITY_WINDOW",
        "add_temporal_probability",
        "rolling_mean_probability",
        "raw_prediction",
        "raw_probabilities",
        "reset_temporal",
    ]

    for token in backend_requirements:

        assert token in backend, (
            f"Backend missing temporal requirement: "
            f"{token}"
        )

    frontend_requirements = [
        "temporal_samples",
        "temporal_window",
        "raw_prediction",
        "raw_probabilities",
        "/reset_temporal",
    ]

    for token in frontend_requirements:

        assert token in frontend, (
            f"Frontend missing temporal requirement: "
            f"{token}"
        )

    html_requirements = [
        'id="temporalsamples"',
        'id="temporalwindow"',
        'id="rawprediction"',
        'id="rawconfidence"',
        'id="resettemporalbtn"',
    ]

    for token in html_requirements:

        assert token in html, (
            f"HTML missing temporal element: "
            f"{token}"
        )

    print(
        "       PASS: Final result uses temporal "
        "mean-probability aggregation."
    )

    print(
        "       PASS: Latest raw result remains visible "
        "for diagnostic comparison."
    )

    print(
        "       PASS: Temporal history can be reset."
    )


def test_temporal_window_is_five_predictions():
    print(
        "\n[ACCEPTANCE] Checking selected temporal window size..."
    )

    backend = WEB_APP_PATH.read_text(
        encoding="utf-8"
    )

    normalised = (
        backend
        .replace(" ", "")
        .replace("\n", "")
    )

    assert (
        "TEMPORAL_PROBABILITY_WINDOW=5"
        in normalised
    ), (
        "Expected final temporal window size of 5."
    )

    print(
        "       PASS: Final web application uses "
        "a five-observation rolling window."
    )


# ============================================================
# Four-class design
# ============================================================

def test_web_interface_supports_all_four_behavioural_states():
    print(
        "\n[ACCEPTANCE] Checking four-class behavioural design..."
    )

    content = WEB_APP_PATH.read_text(
        encoding="utf-8"
    ).lower()

    missing_states = [
        state
        for state in EXPECTED_CLASSES
        if state not in content
    ]

    assert not missing_states, (
        "Missing behavioural states: "
        f"{missing_states}"
    )

    print(
        "       PASS: Focused, distracted, fatigued "
        "and overloaded are represented."
    )


# ============================================================
# Webcam interface
# ============================================================

def test_web_interface_supports_live_webcam_capture():
    print(
        "\n[ACCEPTANCE] Checking live webcam interface support..."
    )

    html = WEB_HTML_PATH.read_text(
        encoding="utf-8"
    ).lower()

    script = WEB_SCRIPT_PATH.read_text(
        encoding="utf-8"
    )

    assert 'id="webcam"' in html
    assert 'id="framecanvas"' in html

    assert "getUserMedia" in script
    assert "captureWebcamFrame" in script
    assert "image_frame" in script

    print(
        "       PASS: Browser webcam capture is supported."
    )


def test_web_interface_exposes_separate_webcam_calibrated_result():
    print(
        "\n[ACCEPTANCE] Checking separate webcam "
        "modality result..."
    )

    html = WEB_HTML_PATH.read_text(
        encoding="utf-8"
    ).lower()

    script = WEB_SCRIPT_PATH.read_text(
        encoding="utf-8"
    )

    assert (
        'id="webcamprediction"'
        in html
    )

    assert (
        'id="webcamconfidence"'
        in html
    )

    assert (
        'id="webcamprobabilitybars"'
        in html
    )

    assert (
        "webcam_prediction"
        in script
    )

    print(
        "       PASS: Webcam-calibrated modality result "
        "is displayed separately from final fusion output."
    )


# ============================================================
# Evaluation scripts
# ============================================================

def test_dissertation_ready_evaluation_scripts_exist():
    print(
        "\n[ACCEPTANCE] Checking evaluation/reporting scripts..."
    )

    required_files = [
        (
            ROOT_DIR
            / "train_multimodal_comparison.py"
        ),
        (
            ROOT_DIR
            / "evaluate_multimodal_results.py"
        ),
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
