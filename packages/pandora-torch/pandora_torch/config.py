"""
Configuration for Pandora-Torch
"""

from dataclasses import dataclass, field
from typing import Optional
from enum import Enum


class PandoraMode(Enum):
    """Activation mode for Pandora"""
    OFF = 0
    AUTO = 1      # SARTRE-controlled
    FORCED = 2    # Always active


@dataclass
class PandoraTorchConfig:
    """Configuration for Pandora-Torch vocabulary extraction"""

    # Model paths
    weights_path: str = "weights/gpt2_distill.pt"
    vocab_path: str = "weights/vocab.json"

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
    coherence_threshold: float = 0.3  # Activate below this
    sacred_threshold: float = 0.7     # Deactivate above this

    # LoRA configuration
    lora_rank: int = 8
    lora_alpha: float = 16.0
    lora_dropout: float = 0.0

    # Generation
    temperature: float = 0.8
    top_k: int = 50
    max_generate: int = 50

    # Device
    device: str = "cpu"  # or "cuda"

    def to_dict(self) -> dict:
        """Convert to dictionary"""
        return {
            "weights_path": self.weights_path,
            "vocab_path": self.vocab_path,
            "min_ngram": self.min_ngram,
            "max_ngram": self.max_ngram,
            "max_ngrams": self.max_ngrams,
            "min_frequency": self.min_frequency,
            "injection_strength": self.injection_strength,
            "mode": self.mode.name,
            "coherence_threshold": self.coherence_threshold,
            "sacred_threshold": self.sacred_threshold,
            "lora_rank": self.lora_rank,
            "lora_alpha": self.lora_alpha,
            "lora_dropout": self.lora_dropout,
            "temperature": self.temperature,
            "top_k": self.top_k,
            "max_generate": self.max_generate,
            "device": self.device,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "PandoraTorchConfig":
        """Create from dictionary"""
        if "mode" in d and isinstance(d["mode"], str):
            d["mode"] = PandoraMode[d["mode"]]
        return cls(**d)
