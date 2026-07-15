# tests/test_02_integration.py

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import ModuleType

ROOT_DIR = Path(__file__).resolve().parents[1]
INFERENCE_PATH = ROOT_DIR / "final_multimodal_inference.py"


def load_inference_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "sensefuze_final_multimodal_inference",
        INFERENCE_PATH,
    )

    assert spec is not None
    assert spec.loader is not None

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    return module


def test_required_model_artifacts_exist():
    required_paths = [
        ROOT_DIR / "models" / "keystroke_demo" / "keystroke_pipeline.joblib",
        ROOT_DIR / "models" / "text_demo" / "text_pipeline.joblib",
        ROOT_DIR / "models" / "audio_demo" / "audio_pipeline.joblib",
        ROOT_DIR / "models" / "image_demo" / "image_pipeline.joblib",
        ROOT_DIR / "models" / "fusion_demo" / "fusion_pipeline.joblib",
        ROOT_DIR / "models" / "fusion_demo" / "feature_columns.json",
        ROOT_DIR / "models" / "all-mpnet-base-v2",
        ROOT_DIR / "models" / "wavlm-base-plus",
        ROOT_DIR / "models" / "clip-vit-large-patch14",
    ]

    missing = [str(path) for path in required_paths if not path.exists()]
    assert not missing, f"Missing model artifacts: {missing}"


def test_fusion_feature_schema_contains_all_modalities():
    schema_path = ROOT_DIR / "models" / "fusion_demo" / "feature_columns.json"

    with schema_path.open("r", encoding="utf-8") as f:
        columns = json.load(f)

    assert isinstance(columns, list)
    assert len(columns) > 0

    assert any(col.startswith("text_mpnet_emb_") for col in columns)
    assert any(col.startswith("audio_wavlm_emb_") for col in columns)
    assert any(col.startswith("image_clip_emb_") for col in columns)
    assert any("keydown" in col or "typing" in col for col in columns)


def test_final_inference_class_importable():
    assert INFERENCE_PATH.exists(), f"Missing file: {INFERENCE_PATH}"

    module = load_inference_module()

    assert hasattr(module, "FinalMultimodalInference")
    assert module.FinalMultimodalInference is not None
