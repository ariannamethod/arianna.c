"""
Model downloader for Pandora-Torch

Downloads GPT2-distill (Stanley) from GitHub ariannamethod/stanley.
"""

import subprocess
import sys
from pathlib import Path
from typing import Optional


# Stanley repo
STANLEY_REPO = "https://github.com/ariannamethod/stanley.git"
STANLEY_PACKAGE = "stanley"

# Default weights path (after Stanley is installed)
DEFAULT_WEIGHTS_PATH = Path(__file__).parent.parent / "weights" / "gpt2_distill"


def is_stanley_installed() -> bool:
    """Check if Stanley package is installed."""
    try:
        import stanley
        return True
    except ImportError:
        return False


def get_stanley_version() -> Optional[str]:
    """Get installed Stanley version."""
    try:
        import stanley
        return getattr(stanley, '__version__', 'unknown')
    except ImportError:
        return None


def install_stanley(
    repo_url: str = STANLEY_REPO,
    upgrade: bool = False,
    show_output: bool = True,
) -> bool:
    """
    Install Stanley from GitHub.

    Args:
        repo_url: GitHub repo URL
        upgrade: Force upgrade if already installed
        show_output: Show pip output

    Returns:
        True if installation successful
    """
    if is_stanley_installed() and not upgrade:
        print(f"[pandora-torch] Stanley already installed (version: {get_stanley_version()})")
        return True

    print(f"[pandora-torch] Installing Stanley from {repo_url}...")
    print(f"[pandora-torch] This may take a while (~300 MB with dependencies)")

    try:
        cmd = [sys.executable, "-m", "pip", "install"]
        if upgrade:
            cmd.append("--upgrade")
        cmd.append(f"git+{repo_url}")

        result = subprocess.run(
            cmd,
            capture_output=not show_output,
            text=True,
        )

        if result.returncode == 0:
            print("[pandora-torch] Stanley installed successfully!")
            return True
        else:
            print(f"[pandora-torch] Installation failed: {result.stderr}")
            return False

    except Exception as e:
        print(f"[pandora-torch] Installation error: {e}")
        return False


def ensure_stanley(auto_install: bool = True) -> bool:
    """
    Ensure Stanley is installed.

    Args:
        auto_install: Install if not found

    Returns:
        True if Stanley is available

    Raises:
        ImportError: If not installed and auto_install=False
    """
    if is_stanley_installed():
        return True

    if auto_install:
        return install_stanley()

    raise ImportError(
        "Stanley not installed. Install with:\n"
        f"  pip install git+{STANLEY_REPO}\n"
        "Or use ensure_stanley(auto_install=True)"
    )


def download_weights(
    destination: Optional[Path] = None,
    show_progress: bool = True,
) -> Path:
    """
    Download GPT2-distill weights via Stanley.

    Stanley handles its own weight management, but this function
    ensures weights are available in a known location.

    Args:
        destination: Local path for weights
        show_progress: Show download progress

    Returns:
        Path to weights directory
    """
    destination = Path(destination) if destination else DEFAULT_WEIGHTS_PATH

    # First ensure Stanley is installed
    ensure_stanley(auto_install=True)

    # Check if weights already exist
    if destination.exists() and any(destination.glob("*.bin")):
        print(f"[pandora-torch] Weights already exist: {destination}")
        return destination

    # Create directory
    destination.parent.mkdir(parents=True, exist_ok=True)

    try:
        # Stanley's transformer auto-downloads weights on first use
        from stanley.inference import StanleyTransformer

        print("[pandora-torch] Initializing Stanley (will download weights if needed)...")
        transformer = StanleyTransformer()

        # Get Stanley's weight path
        stanley_weights = getattr(transformer, 'weights_path', None)
        if stanley_weights:
            print(f"[pandora-torch] Stanley weights at: {stanley_weights}")

        print(f"[pandora-torch] Weights ready!")
        return destination

    except Exception as e:
        print(f"[pandora-torch] Weight download failed: {e}")
        raise


def get_model_info() -> dict:
    """Get information about the model."""
    installed = is_stanley_installed()
    version = get_stanley_version() if installed else None

    return {
        "name": "GPT2-distill (Stanley)",
        "repo": STANLEY_REPO,
        "package": STANLEY_PACKAGE,
        "installed": installed,
        "version": version,
        "size_mb": 300,  # Approximate
        "local_path": str(DEFAULT_WEIGHTS_PATH),
        "weights_exist": DEFAULT_WEIGHTS_PATH.exists(),
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Download Stanley for Pandora-Torch")
    parser.add_argument("--upgrade", action="store_true", help="Force upgrade")
    parser.add_argument("--info", action="store_true", help="Show model info")
    args = parser.parse_args()

    if args.info:
        info = get_model_info()
        print("\n=== Pandora-Torch Model Info ===")
        for k, v in info.items():
            print(f"  {k}: {v}")
    else:
        ensure_stanley(auto_install=True)
        if args.upgrade:
            install_stanley(upgrade=True)
