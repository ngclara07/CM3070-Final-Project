# === download_audio_model.py ===
# SenseFuzeAI - Local Audio Model Downloader
#
# Purpose:
#   Downloads / resolves the YAMNet TensorFlow Hub model once and stores it
#   inside the local project directory:
#
#       models/yamnet/
#
# Why:
#   This avoids relying on TensorFlow Hub temporary cache during prototype
#   demos and final web app testing. The local model improves reproducibility,
#   offline reliability, and loading stability.
#
# Usage:
#   python download_audio_model.py
#   python download_audio_model.py --force
#   python download_audio_model.py --verify-only
#
# Expected final directory:
#   models/yamnet/
#       saved_model.pb
#       variables/
#       assets/
#
# Required packages:
#   pip install tensorflow tensorflow-hub

from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path
from typing import Optional


# Reduce TensorFlow console noise before importing TensorFlow.
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")


# ============================================================
# Configuration
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent

MODELS_DIR = PROJECT_ROOT / "models"
LOCAL_YAMNET_DIR = MODELS_DIR / "yamnet"

TEMP_DOWNLOAD_DIR = MODELS_DIR / "_yamnet_download_tmp"

YAMNET_TFHUB_URL = "https://tfhub.dev/google/yamnet/1"


# ============================================================
# Utility functions
# ============================================================

def print_header() -> None:
    print("==============================================")
    print("SenseFuzeAI Local Audio Model Downloader")
    print("==============================================")
    print(f"Project root:       {PROJECT_ROOT}")
    print(f"Target directory:   {LOCAL_YAMNET_DIR}")
    print(f"TensorFlow Hub URL: {YAMNET_TFHUB_URL}")
    print("==============================================\n")


def fail(message: str, exit_code: int = 1) -> None:
    print(f"\nERROR: {message}")
    raise SystemExit(exit_code)


def ensure_dependencies() -> None:
    """
    Verify that TensorFlow and TensorFlow Hub are installed.
    """
    try:
        import tensorflow as tf  # noqa: F401
    except Exception as exc:
        fail(
            "TensorFlow is not installed or could not be imported.\n\n"
            "Install it with:\n"
            "  pip install tensorflow\n\n"
            f"Original error: {exc}"
        )

    try:
        import tensorflow_hub as hub  # noqa: F401
    except Exception as exc:
        fail(
            "TensorFlow Hub is not installed or could not be imported.\n\n"
            "Install it with:\n"
            "  pip install tensorflow-hub\n\n"
            f"Original error: {exc}"
        )


def get_saved_model_file(model_dir: Path) -> Optional[Path]:
    """
    Return saved_model.pb / saved_model.pbtxt if present.
    """
    saved_model_pb = model_dir / "saved_model.pb"
    saved_model_pbtxt = model_dir / "saved_model.pbtxt"

    if saved_model_pb.exists():
        return saved_model_pb

    if saved_model_pbtxt.exists():
        return saved_model_pbtxt

    return None


def is_valid_saved_model_dir(model_dir: Path) -> bool:
    """
    A TensorFlow SavedModel directory must contain saved_model.pb or
    saved_model.pbtxt. Usually it also contains variables/.
    """
    if not model_dir.exists() or not model_dir.is_dir():
        return False

    return get_saved_model_file(model_dir) is not None


def remove_directory(path: Path) -> None:
    """
    Remove a directory safely.
    """
    if path.exists():
        shutil.rmtree(path)


def copy_model_directory(source_dir: Path, target_dir: Path, force: bool) -> None:
    """
    Copy resolved TFHub model directory into the local project models/yamnet
    directory using a temporary directory first, then atomic replacement.
    """
    if target_dir.exists():
        if not force:
            fail(
                f"Target directory already exists:\n{target_dir}\n\n"
                "Use --force to replace it."
            )

        print(f"Removing existing local model directory:\n{target_dir}\n")
        remove_directory(target_dir)

    remove_directory(TEMP_DOWNLOAD_DIR)

    print("Copying model into local project directory...")
    print(f"Source: {source_dir}")
    print(f"Temp:   {TEMP_DOWNLOAD_DIR}")
    print(f"Target: {target_dir}\n")

    shutil.copytree(source_dir, TEMP_DOWNLOAD_DIR)

    if not is_valid_saved_model_dir(TEMP_DOWNLOAD_DIR):
        remove_directory(TEMP_DOWNLOAD_DIR)
        fail(
            "Copied model directory is not a valid TensorFlow SavedModel.\n"
            "Expected saved_model.pb or saved_model.pbtxt after copying."
        )

    TEMP_DOWNLOAD_DIR.rename(target_dir)

    print("Local model copy complete.\n")


def verify_local_model(model_dir: Path) -> bool:
    """
    Verify that local YAMNet directory appears loadable.
    """
    if not is_valid_saved_model_dir(model_dir):
        print("Local model verification failed.")
        print(f"Directory is missing saved_model.pb / saved_model.pbtxt:\n{model_dir}")
        return False

    saved_model_file = get_saved_model_file(model_dir)

    print("Local model structure looks valid.")
    print(f"SavedModel file: {saved_model_file}")

    variables_dir = model_dir / "variables"
    assets_dir = model_dir / "assets"

    print(f"Variables directory exists: {variables_dir.exists()}")
    print(f"Assets directory exists:    {assets_dir.exists()}")

    return True


def load_local_model_test(model_dir: Path) -> None:
    """
    Test-load the local model using tensorflow_hub.load().
    """
    import tensorflow_hub as hub

    print("\nTesting local model load...")
    model = hub.load(str(model_dir))

    available_attrs = [
        attr
        for attr in ["signatures", "class_map_path"]
        if hasattr(model, attr)
    ]

    print("Local model loaded successfully.")
    print(f"Available attributes: {available_attrs}")

    try:
        class_map_path = model.class_map_path().numpy().decode("utf-8")
        print(f"YAMNet class map path: {class_map_path}")
    except Exception:
        print("YAMNet class_map_path() not available or could not be read.")


def resolve_or_download_tfhub_model(tfhub_url: str) -> Path:
    """
    Resolve/download TensorFlow Hub model and return its local cached directory.

    tensorflow_hub.resolve(url) downloads the module if necessary and returns
    the resolved local module directory.
    """
    import tensorflow_hub as hub

    print("Resolving TensorFlow Hub model...")
    print("This may take some time on the first run.\n")

    resolved_path = Path(hub.resolve(tfhub_url)).resolve()

    print("TensorFlow Hub model resolved.")
    print(f"Resolved cache path:\n{resolved_path}\n")

    if not is_valid_saved_model_dir(resolved_path):
        fail(
            "Resolved TensorFlow Hub path is not a valid SavedModel directory.\n\n"
            f"Resolved path:\n{resolved_path}\n\n"
            "This usually means the TFHub cache is corrupted. Delete the cache:\n"
            '  PowerShell: Remove-Item -Recurse -Force "$env:LOCALAPPDATA\\Temp\\tfhub_modules"\n'
            "Then rerun this script."
        )

    return resolved_path


def write_readme(model_dir: Path, source_url: str) -> None:
    """
    Write a small metadata file next to the local model for reproducibility.
    """
    readme_path = model_dir / "SENSEFUZEAI_MODEL_INFO.txt"

    content = (
        "SenseFuzeAI Local Audio Model\n"
        "=============================\n\n"
        "Model: YAMNet\n"
        f"Source: {source_url}\n"
        "Local purpose: acoustic context analysis for behavioural-state inference\n\n"
        "This model is stored locally to improve demo reliability and avoid runtime\n"
        "dependency on TensorFlow Hub temporary cache or internet access.\n\n"
        "Expected use:\n"
        "  app/models/audio_models.py should load this directory first:\n"
        "      models/yamnet/\n\n"
        "If this directory is missing or invalid, the audio model loader may fall\n"
        "back to TensorFlow Hub online loading depending on your implementation.\n"
    )

    readme_path.write_text(content, encoding="utf-8")


# ============================================================
# Main workflow
# ============================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download and store YAMNet locally for SenseFuzeAI."
    )

    parser.add_argument(
        "--force",
        action="store_true",
        help="Replace existing models/yamnet directory if it already exists.",
    )

    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Only verify the existing local models/yamnet directory.",
    )

    parser.add_argument(
        "--skip-load-test",
        action="store_true",
        help="Skip tensorflow_hub.load() test after copying/verifying.",
    )

    parser.add_argument(
        "--source",
        type=str,
        default=YAMNET_TFHUB_URL,
        help="TensorFlow Hub URL or local SavedModel source path.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    print_header()
    ensure_dependencies()

    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    source = str(args.source).strip()

    if args.verify_only:
        print("Running verification only...\n")

        if not verify_local_model(LOCAL_YAMNET_DIR):
            raise SystemExit(1)

        if not args.skip_load_test:
            load_local_model_test(LOCAL_YAMNET_DIR)

        print("\nVerification complete.")
        return

    if LOCAL_YAMNET_DIR.exists() and not args.force:
        print("Local YAMNet model already exists.\n")

        if verify_local_model(LOCAL_YAMNET_DIR):
            if not args.skip_load_test:
                load_local_model_test(LOCAL_YAMNET_DIR)

            print("\nNo download needed.")
            print("Use --force if you want to replace the existing local model.")
            return

        fail(
            "Existing local model directory is invalid.\n"
            "Run with --force to replace it:\n"
            "  python download_audio_model.py --force"
        )

    source_path = Path(source)

    if source_path.exists():
        resolved_model_dir = source_path.resolve()

        print("Using local source directory instead of TensorFlow Hub URL.")
        print(f"Local source: {resolved_model_dir}\n")

        if not is_valid_saved_model_dir(resolved_model_dir):
            fail(
                "Provided local source directory is not a valid SavedModel.\n"
                f"Source: {resolved_model_dir}"
            )

    else:
        resolved_model_dir = resolve_or_download_tfhub_model(source)

    copy_model_directory(
        source_dir=resolved_model_dir,
        target_dir=LOCAL_YAMNET_DIR,
        force=args.force,
    )

    write_readme(
        model_dir=LOCAL_YAMNET_DIR,
        source_url=source,
    )

    if not verify_local_model(LOCAL_YAMNET_DIR):
        raise SystemExit(1)

    if not args.skip_load_test:
        load_local_model_test(LOCAL_YAMNET_DIR)

    print("\n==============================================")
    print("YAMNet local setup complete.")
    print("==============================================")
    print(f"Local model directory:\n{LOCAL_YAMNET_DIR}\n")
    print("Next step:")
    print("  Update app/models/audio_models.py to load models/yamnet first.")
    print("==============================================")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nDownload interrupted by user.")
        sys.exit(130)
