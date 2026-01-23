"""
Pandora-Torch-GGUF — GGUF model vocabulary extraction

"Take the words from the tiny llama, leave the voice"

Uses TinyLlama 1.1B (Q5_K_M) via llama-cpp-python.
"""

from .config import PandoraGGUFConfig, PandoraMode
from .download import download_model, DEFAULT_MODEL_PATH

__all__ = [
    "PandoraGGUFConfig",
    "PandoraMode",
    "download_model",
    "DEFAULT_MODEL_PATH",
]

# Optional: PandoraGGUF requires llama-cpp-python
try:
    from .pandora import PandoraGGUF
    __all__.append("PandoraGGUF")
except ImportError:
    PandoraGGUF = None

__version__ = "0.1.0"
