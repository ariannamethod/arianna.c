"""
Pandora-Torch — PyTorch vocabulary extraction with LoRA delta support

"Take the words, leave the voice"

Extends Pandora with:
- GPT2-distill as external brain
- LoRA delta extraction for training
- Full SARTRE integration
- Async support
"""

from .config import PandoraTorchConfig, PandoraMode
from .sartre import SARTREChecker, ResonancePattern, VagusState

__all__ = [
    "PandoraTorchConfig",
    "PandoraMode",
    "SARTREChecker",
    "ResonancePattern",
    "VagusState",
]

# Optional: PandoraTorch requires torch
try:
    from .pandora import PandoraTorch
    __all__.append("PandoraTorch")
except ImportError:
    PandoraTorch = None

# Async support (no torch required for base classes)
from .async_pandora import AsyncPandoraTorch, AsyncExtractionQueue, AsyncPandoraStream
__all__.extend(["AsyncPandoraTorch", "AsyncExtractionQueue", "AsyncPandoraStream"])

__version__ = "0.1.0"
