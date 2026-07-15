# tests/test_04_acceptance.py

from __future__ import annotations

from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]


def test_project_contains_required_live_interfaces():
    required_files = [
        ROOT_DIR / "keystroke_live_gui.py",
        ROOT_DIR / "text_live_gui.py",
        ROOT_DIR / "audio_live_gui.py",
        ROOT_DIR / "image_live_gui.py",
        ROOT_DIR / "live_fusion_gui.py",
        ROOT_DIR / "web_app" / "app.py",
        ROOT_DIR / "web_app" / "templates" / "index.html",
        ROOT_DIR / "web_app" / "static" / "script.js",
        ROOT_DIR / "web_app" / "static" / "style.css",
    ]

    missing = [str(path) for path in required_files if not path.exists()]
    assert not missing, f"Missing required interface files: {missing}"


def test_project_uses_at_least_three_pretrained_models():
    pretrained_models = [
        ROOT_DIR / "models" / "all-mpnet-base-v2",
        ROOT_DIR / "models" / "wavlm-base-plus",
        ROOT_DIR / "models" / "clip-vit-large-patch14",
    ]

    missing = [str(path) for path in pretrained_models if not path.exists()]
    assert not missing, f"Missing pre-trained model directories: {missing}"


def test_project_contains_multimodal_fusion_model():
    assert (ROOT_DIR / "models" / "fusion_demo" / "fusion_pipeline.joblib").exists()
    assert (ROOT_DIR / "models" / "fusion_demo" / "feature_columns.json").exists()


def test_final_output_design_supported_in_web_script():
    script_path = ROOT_DIR / "web_app" / "static" / "script.js"
    content = script_path.read_text(encoding="utf-8")

    assert "current_state" in content or "prediction" in content
    assert "confidence" in content.lower()
    assert "probabilities" in content.lower()


def test_dissertation_ready_evaluation_scripts_exist():
    required_files = [
        ROOT_DIR / "train_multimodal_comparison.py",
        ROOT_DIR / "evaluate_multimodal_results.py",
    ]

    missing = [str(path) for path in required_files if not path.exists()]
    assert not missing, f"Missing evaluation scripts: {missing}"
