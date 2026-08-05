# === tests/test_04_acceptance.py ===

from __future__ import annotations

import ast
import json
import sys

from pathlib import Path


# ============================================================
# Paths
# ============================================================

ROOT_DIR = (
    Path(__file__)
    .resolve()
    .parents[1]
)

if str(ROOT_DIR) not in sys.path:

    sys.path.insert(
        0,
        str(ROOT_DIR),
    )


TEMPORAL_PATH = (
    ROOT_DIR
    / "temporal_fusion.py"
)

FINAL_INFERENCE_PATH = (
    ROOT_DIR
    / "final_multimodal_inference.py"
)

LIVE_GUI_PATH = (
    ROOT_DIR
    / "live_fusion_gui.py"
)

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

EVALUATION_PATH = (
    ROOT_DIR
    / "evaluate_multimodal_results.py"
)

COMPARISON_PATH = (
    ROOT_DIR
    / "train_multimodal_comparison.py"
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

FUSION_MODEL = (
    ROOT_DIR
    / "models"
    / "fusion_demo"
    / "fusion_pipeline.joblib"
)

FUSION_SCHEMA = (
    ROOT_DIR
    / "models"
    / "fusion_demo"
    / "feature_columns.json"
)


EXPECTED_LABELS = (
    "focused",
    "distracted",
    "fatigued",
    "overloaded",
)


# ============================================================
# AST helper
# ============================================================

def imports_from_temporal_fusion(
    path: Path,
) -> set[str]:

    tree = ast.parse(
        path.read_text(
            encoding="utf-8"
        )
    )

    names: set[
        str
    ] = set()

    for node in ast.walk(
        tree
    ):

        if (
            isinstance(
                node,
                ast.ImportFrom,
            )
            and
            node.module
            == "temporal_fusion"
        ):

            names.update(
                alias.name
                for alias
                in node.names
            )

    return names


def calls_temporal_engine(
    path: Path,
) -> bool:

    tree = ast.parse(
        path.read_text(
            encoding="utf-8"
        )
    )

    for node in ast.walk(
        tree
    ):

        if not isinstance(
            node,
            ast.Call,
        ):

            continue

        if (
            isinstance(
                node.func,
                ast.Name,
            )
            and
            node.func.id
            == "TemporalFusionEngine"
        ):

            return True

    return False


# ============================================================
# Core software deliverable
# ============================================================

def test_project_contains_required_application_files():

    required = [
        TEMPORAL_PATH,
        FINAL_INFERENCE_PATH,
        LIVE_GUI_PATH,
        WEB_APP_PATH,
        WEB_HTML_PATH,
        WEB_SCRIPT_PATH,
        WEB_STYLE_PATH,
        EVALUATION_PATH,
        COMPARISON_PATH,
        (
            ROOT_DIR
            / "keystroke_live_gui.py"
        ),
        (
            ROOT_DIR
            / "text_live_gui.py"
        ),
        (
            ROOT_DIR
            / "audio_live_gui.py"
        ),
        (
            ROOT_DIR
            / "image_live_gui.py"
        ),
    ]

    missing = [
        str(path)
        for path
        in required
        if not path.exists()
    ]

    assert not missing, (
        "Missing required final-project files:\n"
        + "\n".join(
            missing
        )
    )


# ============================================================
# Project requirement: >=3 pretrained models / data domains
# ============================================================

def test_project_contains_three_distinct_pretrained_encoders():

    pretrained = {
        "text_mpnet":
            (
                ROOT_DIR
                / "models"
                / "all-mpnet-base-v2"
            ),

        "audio_wavlm":
            (
                ROOT_DIR
                / "models"
                / "wavlm-base-plus"
            ),

        "vision_clip":
            (
                ROOT_DIR
                / "models"
                / "clip-vit-large-patch14"
            ),
    }

    missing = {
        name:
            str(path)
        for name, path
        in pretrained.items()
        if not path.exists()
    }

    assert not missing, (
        "Missing pretrained multimodal models:\n"
        f"{missing}"
    )


def test_final_inference_uses_three_pretrained_domains():

    source = (
        FINAL_INFERENCE_PATH
        .read_text(
            encoding="utf-8"
        )
    )

    required = [
        "SentenceTransformer",
        "WavLMModel",
        "CLIPModel",
        "extract_text_features",
        "extract_audio_features",
        "extract_image_features",
        "extract_keystroke_features",
    ]

    for token in required:

        assert token in source


# ============================================================
# Fusion model
# ============================================================

def test_final_fusion_model_and_schema_exist():

    assert (
        FUSION_MODEL.exists()
    )

    assert (
        FUSION_SCHEMA.exists()
    )

    columns = json.loads(
        FUSION_SCHEMA.read_text(
            encoding="utf-8"
        )
    )

    assert isinstance(
        columns,
        list,
    )

    assert columns


def test_fusion_schema_combines_all_four_modalities():

    columns = json.loads(
        FUSION_SCHEMA.read_text(
            encoding="utf-8"
        )
    )

    assert any(
        column.startswith(
            "text_mpnet_emb_"
        )
        for column in columns
    )

    assert any(
        column.startswith(
            "audio_"
        )
        for column in columns
    )

    assert any(
        column.startswith(
            "image_clip_emb_"
        )
        for column in columns
    )

    assert any(
        (
            "keydown"
            in column
            or
            "typing"
            in column
            or
            "delay_"
            in column
            or
            "hold_"
            in column
        )
        for column in columns
    )


# ============================================================
# Single canonical temporal implementation
# ============================================================

def test_temporal_fusion_is_single_source_of_truth():

    from temporal_fusion import (
        LABELS,
        TEMPORAL_PROBABILITY_WINDOW,
    )

    assert (
        tuple(
            LABELS
        )
        == EXPECTED_LABELS
    )

    assert (
        TEMPORAL_PROBABILITY_WINDOW
        == 5
    )


def test_final_inference_is_stateless():

    imports = (
        imports_from_temporal_fusion(
            FINAL_INFERENCE_PATH
        )
    )

    assert (
        "LABELS"
        in imports
    )

    assert (
        "normalise_probability_dict"
        in imports
    )

    assert not calls_temporal_engine(
        FINAL_INFERENCE_PATH
    )


def test_desktop_uses_shared_temporal_engine():

    imports = (
        imports_from_temporal_fusion(
            LIVE_GUI_PATH
        )
    )

    assert (
        "TemporalFusionEngine"
        in imports
    )

    assert calls_temporal_engine(
        LIVE_GUI_PATH
    )

    source = (
        LIVE_GUI_PATH.read_text(
            encoding="utf-8"
        )
    )

    assert (
        "probability_history"
        not in source
    )


def test_web_backend_uses_per_session_shared_temporal_engine():

    imports = (
        imports_from_temporal_fusion(
            WEB_APP_PATH
        )
    )

    assert (
        "TemporalFusionEngine"
        in imports
    )

    source = (
        WEB_APP_PATH.read_text(
            encoding="utf-8"
        )
    )

    assert (
        "temporal_fusion:"
        in source
    )

    assert (
        "SESSION_PROBABILITY_HISTORY"
        not in source
    )

    assert (
        "add_temporal_probability"
        not in source
    )

    assert (
        "rolling_mean_probability"
        not in source
    )


def test_evaluation_uses_same_temporal_engine():

    assert (
        "TemporalFusionEngine"
        in imports_from_temporal_fusion(
            EVALUATION_PATH
        )
    )

    assert calls_temporal_engine(
        EVALUATION_PATH
    )


def test_training_comparison_uses_same_temporal_engine():

    assert (
        "TemporalFusionEngine"
        in imports_from_temporal_fusion(
            COMPARISON_PATH
        )
    )

    assert calls_temporal_engine(
        COMPARISON_PATH
    )


# ============================================================
# Browser responsibility
# ============================================================

def test_javascript_does_not_own_temporal_mathematics():

    script = (
        WEB_SCRIPT_PATH
        .read_text(
            encoding="utf-8"
        )
    )

    forbidden = [
        "aggregateProbabilityHistory",
        "normaliseProbabilityDict",
        "normalizeProbabilityDict",
        "temporalProbabilityHistory",
        "confidenceLevelFromGap",
    ]

    for token in forbidden:

        assert (
            token
            not in script
        )


def test_javascript_has_generation_safe_client_state():

    script = (
        WEB_SCRIPT_PATH
        .read_text(
            encoding="utf-8"
        )
    )

    required = [
        "serverGeneration",
        "clientEpoch",
        "stale_generation",
        "stale_result",
        "stale_session",
        "visual_mode_mismatch",
    ]

    for token in required:

        assert token in script


# ============================================================
# Four-modality live acquisition
# ============================================================

def test_web_supports_fixed_audio_source():

    backend = (
        WEB_APP_PATH
        .read_text(
            encoding="utf-8"
        )
    )

    frontend = (
        WEB_SCRIPT_PATH
        .read_text(
            encoding="utf-8"
        )
    )

    # --------------------------------------------------------
    # 1. A dedicated endpoint must exist for setting/replacing
    #    the persistent audio source.
    # --------------------------------------------------------

    assert (
        "/set_audio_source"
        in backend
    ), (
        "Backend is missing the dedicated "
        "/set_audio_source endpoint."
    )

    assert (
        "/set_audio_source"
        in frontend
    ), (
        "Frontend does not call "
        "/set_audio_source."
    )

    # --------------------------------------------------------
    # 2. SessionState must retain audio state between
    #    prediction requests.
    # --------------------------------------------------------

    tree = ast.parse(
        backend
    )

    session_class = next(
        (
            node
            for node
            in tree.body
            if (
                isinstance(
                    node,
                    ast.ClassDef,
                )
                and
                node.name
                == "SessionState"
            )
        ),
        None,
    )

    assert session_class is not None, (
        "Backend is missing SessionState."
    )

    session_fields = set()

    for node in session_class.body:

        if (
            isinstance(
                node,
                ast.AnnAssign,
            )
            and
            isinstance(
                node.target,
                ast.Name,
            )
        ):

            session_fields.add(
                node.target.id
            )

        elif isinstance(
            node,
            ast.Assign,
        ):

            for target in node.targets:

                if isinstance(
                    target,
                    ast.Name,
                ):

                    session_fields.add(
                        target.id
                    )

    assert (
        "audio_path"
        in session_fields
    ), (
        "SessionState must persist the selected "
        "audio source path."
    )

    # At least one descriptive/source-state field should also
    # be retained alongside the path.
    assert any(
        field
        in session_fields
        for field
        in {
            "audio_name",
            "audio_source_kind",
            "audio_diagnostics",
        }
    ), (
        "SessionState does not expose persistent "
        "audio-source metadata."
    )

    # --------------------------------------------------------
    # 3. /predict_live must reuse the session audio.
    #
    #    It must NOT require an audio file upload for every
    #    live prediction.
    # --------------------------------------------------------

    predict_live_function = next(
        (
            node
            for node
            in tree.body
            if (
                isinstance(
                    node,
                    (
                        ast.FunctionDef,
                        ast.AsyncFunctionDef,
                    ),
                )
                and
                node.name
                == "predict_live"
            )
        ),
        None,
    )

    assert (
        predict_live_function
        is not None
    ), (
        "Backend is missing predict_live()."
    )

    predict_live_arguments = {
        argument.arg
        for argument
        in (
            predict_live_function.args.args
            +
            predict_live_function.args.kwonlyargs
        )
    }

    assert (
        "audio_file"
        not in predict_live_arguments
    ), (
        "/predict_live should reuse the persistent "
        "session audio source rather than requiring "
        "a new audio upload every prediction."
    )

    # --------------------------------------------------------
    # 4. The dedicated source endpoint should accept the audio
    #    upload.
    # --------------------------------------------------------

    set_audio_function = next(
        (
            node
            for node
            in tree.body
            if (
                isinstance(
                    node,
                    (
                        ast.FunctionDef,
                        ast.AsyncFunctionDef,
                    ),
                )
                and
                node.name
                == "set_audio_source"
            )
        ),
        None,
    )

    assert (
        set_audio_function
        is not None
    ), (
        "Backend is missing set_audio_source()."
    )

    set_audio_arguments = {
        argument.arg
        for argument
        in (
            set_audio_function.args.args
            +
            set_audio_function.args.kwonlyargs
        )
    }

    assert (
        "audio_file"
        in set_audio_arguments
    ), (
        "set_audio_source() should receive "
        "the uploaded audio file."
    )

    # --------------------------------------------------------
    # 5. Browser acquisition should explicitly set audio once,
    #    not run a recurring MediaRecorder capture loop.
    # --------------------------------------------------------

    assert (
        "setAudioFile"
        in frontend
    ), (
        "Frontend is missing the persistent "
        "audio-source helper."
    )

    assert (
        "recordMicrophoneOnce"
        in frontend
    ), (
        "Frontend is missing one-shot "
        "microphone recording."
    )

    assert (
        "MediaRecorder"
        not in frontend
    ), (
        "Frontend appears to contain the old recurring "
        "MediaRecorder architecture."
    )


def test_web_supports_image_video_and_webcam():

    backend = (
        WEB_APP_PATH
        .read_text(
            encoding="utf-8"
        )
    )

    frontend = (
        WEB_SCRIPT_PATH
        .read_text(
            encoding="utf-8"
        )
    )

    endpoints = [
        "/set_visual_image",
        "/set_visual_video",
        "/set_visual_webcam",
        "/stop_visual",
    ]

    for endpoint in endpoints:

        assert endpoint in backend
        assert endpoint in frontend

    assert (
        "captureWebcamFrame"
        in frontend
    )

    assert (
        "webcam_frame"
        in frontend
    )


def test_live_four_modality_gating_is_present():

    desktop = (
        LIVE_GUI_PATH
        .read_text(
            encoding="utf-8"
        )
    )

    backend = (
        WEB_APP_PATH
        .read_text(
            encoding="utf-8"
        )
    )

    assert (
        "MIN_TEXT_CHARS = 20"
        in desktop
    )

    assert (
        "MIN_KEYDOWNS = 20"
        in desktop
    )

    assert (
        "MIN_TEXT_CHARS = 20"
        in backend
    )

    assert (
        "MIN_KEYPRESSES = 20"
        in backend
    )


# ============================================================
# Temporal reset behaviour
# ============================================================

def test_temporal_reset_and_full_reset_endpoints_exist():

    backend = (
        WEB_APP_PATH
        .read_text(
            encoding="utf-8"
        )
    )

    assert (
        '"/reset_temporal"'
        in backend
    )

    assert (
        '"/full_reset"'
        in backend
    )


def test_source_changes_reset_temporal_generation():

    backend = (
        WEB_APP_PATH
        .read_text(
            encoding="utf-8"
        )
    )

    assert (
        "reset_temporal_for_source_change"
        in backend
    )

    assert (
        ".temporal_fusion.reset()"
        in backend
        or
        "state.temporal_fusion.reset()"
        in backend
    )


# ============================================================
# User-facing output
# ============================================================

def test_web_interface_exposes_final_prediction_and_confidence():

    html = (
        WEB_HTML_PATH
        .read_text(
            encoding="utf-8"
        )
        .lower()
    )

    required = [
        'id="prediction"',
        'id="confidencepercent"',
        'id="confidencelevel"',
        'id="confidencegap"',
        'id="probabilities"',
    ]

    for element in required:

        assert element in html


def test_web_interface_exposes_raw_vs_temporal_diagnostics():

    html = (
        WEB_HTML_PATH
        .read_text(
            encoding="utf-8"
        )
        .lower()
    )

    required = [
        'id="rawprediction"',
        'id="rawconfidence"',
        'id="rawprobabilities"',
        'id="temporalsamples"',
        'id="temporalwindow"',
        'id="resettemporalbtn"',
    ]

    for element in required:

        assert element in html


def test_web_interface_exposes_calibrated_visual_diagnostic():

    html = (
        WEB_HTML_PATH
        .read_text(
            encoding="utf-8"
        )
        .lower()
    )

    required = [
        'id="webcamprediction"',
        'id="webcamconfidence"',
        'id="webcamprobabilitybars"',
    ]

    for element in required:

        assert element in html


# ============================================================
# Webcam calibration evidence
# ============================================================

def test_original_and_calibrated_visual_models_are_preserved():

    assert (
        ORIGINAL_IMAGE_MODEL.exists()
    )

    assert (
        CALIBRATED_IMAGE_MODEL.exists()
    )

    assert (
        ORIGINAL_IMAGE_MODEL
        != CALIBRATED_IMAGE_MODEL
    )

    assert (
        CALIBRATED_METADATA.exists()
    )


def test_final_inference_owns_webcam_calibration_integration():

    source = (
        FINAL_INFERENCE_PATH
        .read_text(
            encoding="utf-8"
        )
    )

    assert (
        "image_pipeline_webcam_calibrated.joblib"
        in source
    )

    assert (
        "extract_webcam_calibration_features"
        in source
    )

    assert (
        "image_calibration"
        in source
    )


# ============================================================
# Evaluation methodology
# ============================================================

def test_evaluation_script_distinguishes_raw_and_temporal_results():

    source = (
        EVALUATION_PATH
        .read_text(
            encoding="utf-8"
        )
    )

    required = [
        "raw_fusion",
        "temporal_fusion_all_samples",
        "temporal_fusion_full_window",
        "confusion_matrix",
        "macro_f1",
        "balanced_accuracy",
        "multiclass_brier_score",
        "expected_calibration_error",
    ]

    for token in required:

        assert token in source


def test_comparison_script_uses_untouched_group_aware_holdout():

    source = (
        COMPARISON_PATH
        .read_text(
            encoding="utf-8"
        )
    )

    required = [
        "StratifiedGroupKFold",
        "development",
        "heldout",
        "TemporalFusionEngine",
        "raw_vs_temporal_test_comparison.csv",
        "multimodal_all_raw_vs_temporal.json",
    ]

    for token in required:

        assert token in source

    assert (
        "train_test_split"
        not in source
    )


# ============================================================
# Automated testing deliverable
# ============================================================

def test_complete_automated_test_structure_exists():

    required = [
        (
            ROOT_DIR
            / "run_all_tests.py"
        ),
        (
            ROOT_DIR
            / "tests"
            / "test_01_unit.py"
        ),
        (
            ROOT_DIR
            / "tests"
            / "test_02_integration.py"
        ),
        (
            ROOT_DIR
            / "tests"
            / "test_03_system.py"
        ),
        (
            ROOT_DIR
            / "tests"
            / "test_04_acceptance.py"
        ),
        (
            ROOT_DIR
            / "tests"
            / "test_sensefuzeai.py"
        ),
    ]

    missing = [
        str(path)
        for path
        in required
        if not path.exists()
    ]

    assert not missing
