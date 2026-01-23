"""
Model downloader for Pandora-Torch-GGUF

Downloads TinyLlama 1.1B GGUF from HuggingFace.
"""

import os
from pathlib import Path
from typing import Optional


# Model details from TheBloke
HF_REPO = "TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF"
HF_FILENAME = "tinyllama-1.1b-chat-v1.0.Q5_K_M.gguf"

# Default local path
DEFAULT_MODEL_PATH = Path(__file__).parent.parent / "weights" / "tinyllama-1.1b-q5.gguf"


def download_model(
    destination: Optional[Path] = None,
    repo_id: str = HF_REPO,
    filename: str = HF_FILENAME,
    show_progress: bool = True,
) -> Path:
    """
    Download TinyLlama GGUF from HuggingFace.

    Args:
        destination: Local path for model. Uses DEFAULT_MODEL_PATH if None.
        repo_id: HuggingFace repo ID
        filename: GGUF filename in repo
        show_progress: Show download progress

    Returns:
        Path to downloaded model
    """
    destination = Path(destination) if destination else DEFAULT_MODEL_PATH

    # Check if already exists
    if destination.exists():
        print(f"[pandora-gguf] Model already exists: {destination}")
        return destination

    # Create directory
    destination.parent.mkdir(parents=True, exist_ok=True)

    try:
        from huggingface_hub import hf_hub_download

        print(f"[pandora-gguf] Downloading {filename} from {repo_id}...")
        print(f"[pandora-gguf] This may take a while (~783 MB)")

        downloaded_path = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            local_dir=destination.parent,
            local_dir_use_symlinks=False,
        )

        # Rename to our standard name
        downloaded = Path(downloaded_path)
        if downloaded != destination:
            downloaded.rename(destination)

        print(f"[pandora-gguf] Downloaded to: {destination}")
        return destination

    except ImportError:
        print("[pandora-gguf] huggingface_hub not installed. Install with:")
        print("  pip install huggingface_hub")
        raise

    except Exception as e:
        print(f"[pandora-gguf] Download failed: {e}")
        raise


def ensure_model(
    model_path: Optional[Path] = None,
    auto_download: bool = True,
) -> Path:
    """
    Ensure model exists, downloading if necessary.

    Args:
        model_path: Path to check/download
        auto_download: Download if not found

    Returns:
        Path to model

    Raises:
        FileNotFoundError: If model not found and auto_download=False
    """
    path = Path(model_path) if model_path else DEFAULT_MODEL_PATH

    if path.exists():
        return path

    if auto_download:
        return download_model(path)

    raise FileNotFoundError(
        f"Model not found: {path}\n"
        f"Run with auto_download=True or download manually from:\n"
        f"  https://huggingface.co/{HF_REPO}"
    )


def get_model_info() -> dict:
    """Get information about the model"""
    return {
        "name": "TinyLlama-1.1B-Chat-v1.0",
        "quantization": "Q5_K_M",
        "size_mb": 783,
        "context_length": 2048,
        "repo": HF_REPO,
        "filename": HF_FILENAME,
        "local_path": str(DEFAULT_MODEL_PATH),
        "exists": DEFAULT_MODEL_PATH.exists(),
    }
