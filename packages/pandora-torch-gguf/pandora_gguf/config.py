"""
Configuration for Pandora-Torch-GGUF
"""

from dataclasses import dataclass
from typing import Optional
from enum import Enum
from pathlib import Path


class PandoraMode(Enum):
    """Activation mode for Pandora"""
    OFF = 0
    AUTO = 1      # SARTRE-controlled
    FORCED = 2    # Always active


@dataclass
class PandoraGGUFConfig:
    """Configuration for Pandora-Torch-GGUF"""

    # Model paths
    model_path: str = "weights/tinyllama-1.1b-q5.gguf"
    auto_download: bool = True

    # llama.cpp settings
    n_ctx: int = 2048         # Context size
    n_threads: int = 4        # CPU threads
    n_gpu_layers: int = 0     # GPU offload (0 = CPU only)
    n_batch: int = 512        # Batch size for prompt processing
    verbose: bool = False     # Show llama.cpp output

    # N-gram extraction
    min_ngram: int = 1
    max_ngram: int = 3
    max_ngrams: int = 1000
    min_frequency: int = 3

    # Injection
    injection_strength: float = 0.2

    # Activation mode
    mode: PandoraMode = PandoraMode.AUTO

    # SARTRE thresholds
    coherence_threshold: float = 0.3
    sacred_threshold: float = 0.7

    # Generation
    temperature: float = 0.8
    top_k: int = 40
    top_p: float = 0.95
    repeat_penalty: float = 1.1
    max_tokens: int = 50

    def to_dict(self) -> dict:
        """Convert to dictionary"""
        return {
            "model_path": str(self.model_path),
            "auto_download": self.auto_download,
            "n_ctx": self.n_ctx,
            "n_threads": self.n_threads,
            "n_gpu_layers": self.n_gpu_layers,
            "n_batch": self.n_batch,
            "verbose": self.verbose,
            "min_ngram": self.min_ngram,
            "max_ngram": self.max_ngram,
            "max_ngrams": self.max_ngrams,
            "min_frequency": self.min_frequency,
            "injection_strength": self.injection_strength,
            "mode": self.mode.name,
            "coherence_threshold": self.coherence_threshold,
            "sacred_threshold": self.sacred_threshold,
            "temperature": self.temperature,
            "top_k": self.top_k,
            "top_p": self.top_p,
            "repeat_penalty": self.repeat_penalty,
            "max_tokens": self.max_tokens,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "PandoraGGUFConfig":
        """Create from dictionary"""
        if "mode" in d and isinstance(d["mode"], str):
            d["mode"] = PandoraMode[d["mode"]]
        return cls(**d)
