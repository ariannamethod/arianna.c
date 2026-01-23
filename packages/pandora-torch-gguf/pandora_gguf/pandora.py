"""
Pandora-Torch-GGUF — GGUF model vocabulary extraction

"Take the words from the tiny llama, leave the voice"

Uses TinyLlama 1.1B (Q5_K_M) via llama-cpp-python.
"""

from typing import List, Dict, Optional, Callable, Any
from dataclasses import dataclass, field
from pathlib import Path
import json
import time

from llama_cpp import Llama

from .config import PandoraGGUFConfig, PandoraMode
from .download import ensure_model, DEFAULT_MODEL_PATH


@dataclass
class ReleasedNGram:
    """N-gram extracted from GGUF model"""
    tokens: List[int]
    weight: float = 0.1
    frequency: int = 1
    arianna_mapped: bool = False
    arianna_tokens: List[int] = field(default_factory=list)

    @property
    def length(self) -> int:
        return len(self.tokens)


@dataclass
class PandoraGGUFStats:
    """Statistics for Pandora state"""
    total_ngrams: int
    mapped_ngrams: int
    avg_weight: float
    avg_frequency: float
    mode: str
    injection_strength: float
    active: bool
    model_loaded: bool
    model_path: str


class SARTREChecker:
    """SARTRE activation checker (mirrored from pandora-torch)"""

    def __init__(self, coherence_threshold: float = 0.3, sacred_threshold: float = 0.7):
        self.coherence_threshold = coherence_threshold
        self.sacred_threshold = sacred_threshold
        self._last_decision = False

    def check(self, coherence: float, sacred: float, pattern: int) -> bool:
        """Check if should activate"""
        if sacred > self.sacred_threshold:
            self._last_decision = False
            return False
        if pattern == 1:  # CRISIS
            self._last_decision = False
            return False
        if coherence < self.coherence_threshold:
            self._last_decision = True
            return True
        if pattern in (3, 4):  # EMERGENCE, TRANSCENDENCE
            self._last_decision = True
            return True
        return self._last_decision


class PandoraGGUF:
    """
    GGUF model vocabulary extraction using TinyLlama 1.1B.

    Features:
    - Efficient GGUF inference via llama-cpp-python
    - Auto-download from HuggingFace
    - N-gram extraction and mapping
    - Full SARTRE integration
    """

    def __init__(
        self,
        config: Optional[PandoraGGUFConfig] = None,
        model_path: Optional[str] = None,
        auto_download: bool = True,
        mode: str = "auto",
    ):
        self.config = config or PandoraGGUFConfig()
        if model_path:
            self.config.model_path = model_path
        self.config.auto_download = auto_download

        # Parse mode
        if isinstance(mode, str):
            self.config.mode = PandoraMode[mode.upper()]

        # State
        self.ngrams: Dict[tuple, ReleasedNGram] = {}
        self.total_released = 0
        self.successfully_mapped = 0

        # SARTRE checker
        self.sartre = SARTREChecker(
            coherence_threshold=self.config.coherence_threshold,
            sacred_threshold=self.config.sacred_threshold,
        )

        # Model (lazy load)
        self._model: Optional[Llama] = None
        self._model_path: Optional[Path] = None

    # ═══════════════════════════════════════════════════════════════════════════
    # MODEL LOADING
    # ═══════════════════════════════════════════════════════════════════════════

    def _load_model(self) -> None:
        """Load GGUF model"""
        if self._model is not None:
            return

        # Ensure model exists
        self._model_path = ensure_model(
            Path(self.config.model_path),
            auto_download=self.config.auto_download,
        )

        print(f"[pandora-gguf] Loading model: {self._model_path}")

        self._model = Llama(
            model_path=str(self._model_path),
            n_ctx=self.config.n_ctx,
            n_threads=self.config.n_threads,
            n_gpu_layers=self.config.n_gpu_layers,
            n_batch=self.config.n_batch,
            verbose=self.config.verbose,
        )

        print(f"[pandora-gguf] Model loaded successfully")

    def _ensure_model(self) -> bool:
        """Ensure model is loaded"""
        if self._model is None:
            try:
                self._load_model()
            except Exception as e:
                print(f"[pandora-gguf] Failed to load model: {e}")
                return False
        return self._model is not None

    # ═══════════════════════════════════════════════════════════════════════════
    # N-GRAM EXTRACTION
    # ═══════════════════════════════════════════════════════════════════════════

    def extract(
        self,
        tokens: List[int],
        min_n: Optional[int] = None,
        max_n: Optional[int] = None,
    ) -> int:
        """Extract n-grams from token sequence"""
        if not self.is_active():
            return 0

        min_n = min_n or self.config.min_ngram
        max_n = max_n or self.config.max_ngram

        added = 0

        for n in range(min_n, max_n + 1):
            for start in range(len(tokens) - n + 1):
                ngram = tuple(tokens[start:start + n])

                if ngram in self.ngrams:
                    self.ngrams[ngram].frequency += 1
                    self.ngrams[ngram].weight = min(1.0, self.ngrams[ngram].weight + 0.01)
                elif len(self.ngrams) < self.config.max_ngrams:
                    self.ngrams[ngram] = ReleasedNGram(
                        tokens=list(ngram),
                        weight=0.1,
                        frequency=1,
                    )
                    added += 1
                    self.total_released += 1

        return added

    def map_to_arianna(
        self,
        brain_decode: Callable[[int], Optional[str]],
        arianna_encode: Callable[[str], int],
    ) -> int:
        """Map extracted n-grams to Arianna vocabulary"""
        mapped = 0

        for ngram in self.ngrams.values():
            if ngram.arianna_mapped:
                continue

            success = True
            arianna_tokens = []

            for tok in ngram.tokens:
                word = brain_decode(tok)
                if word is None:
                    success = False
                    break

                arianna_id = arianna_encode(word)
                if arianna_id < 0:
                    success = False
                    break

                arianna_tokens.append(arianna_id)

            if success:
                ngram.arianna_mapped = True
                ngram.arianna_tokens = arianna_tokens
                mapped += 1

        self.successfully_mapped = mapped
        return mapped

    # ═══════════════════════════════════════════════════════════════════════════
    # LOGIT INJECTION
    # ═══════════════════════════════════════════════════════════════════════════

    def apply_to_logits(
        self,
        logits: Any,
        context_tokens: List[int],
        vocab_size: Optional[int] = None,
    ) -> Any:
        """Apply released vocabulary to logits"""
        if not self.is_active():
            return logits

        if self.config.injection_strength <= 0:
            return logits

        # Handle numpy/torch arrays
        if hasattr(logits, 'clone'):
            boosted = logits.clone()
        elif hasattr(logits, 'copy'):
            boosted = logits.copy()
        else:
            boosted = list(logits)

        vocab_size = vocab_size or len(boosted)

        for ngram in self.ngrams.values():
            if not ngram.arianna_mapped:
                continue
            if ngram.frequency < self.config.min_frequency:
                continue

            prefix_len = ngram.length - 1

            if prefix_len == 0:
                tok = ngram.arianna_tokens[0]
                if 0 <= tok < vocab_size:
                    boosted[tok] += ngram.weight * self.config.injection_strength * 0.5
                continue

            if prefix_len > len(context_tokens):
                continue

            prefix = ngram.arianna_tokens[:-1]
            context_suffix = context_tokens[-prefix_len:]

            if prefix == context_suffix:
                next_tok = ngram.arianna_tokens[-1]
                if 0 <= next_tok < vocab_size:
                    boost = ngram.weight * self.config.injection_strength
                    boost *= (1.0 + 0.1 * ngram.frequency)
                    boosted[next_tok] += boost

        return boosted

    # ═══════════════════════════════════════════════════════════════════════════
    # FULL PIPELINE
    # ═══════════════════════════════════════════════════════════════════════════

    def process(
        self,
        text: str,
        arianna_encode: Callable[[str], int],
        max_tokens: Optional[int] = None,
    ) -> int:
        """Full pipeline: generate from model, extract, map"""
        if not self.is_active():
            return 0

        if not self._ensure_model():
            return 0

        max_tokens = max_tokens or self.config.max_tokens

        try:
            # Generate from TinyLlama
            output = self._model(
                text,
                max_tokens=max_tokens,
                temperature=self.config.temperature,
                top_k=self.config.top_k,
                top_p=self.config.top_p,
                repeat_penalty=self.config.repeat_penalty,
                echo=False,
            )

            # Get generated text and tokenize
            generated = output["choices"][0]["text"]
            tokens = self._model.tokenize(generated.encode())

            # Extract n-grams
            added = self.extract(tokens)

            # Map to Arianna vocab
            def brain_decode(tok_id: int) -> Optional[str]:
                try:
                    return self._model.detokenize([tok_id]).decode('utf-8', errors='ignore')
                except:
                    return None

            self.map_to_arianna(brain_decode, arianna_encode)

            return added

        except Exception as e:
            print(f"[pandora-gguf] Process error: {e}")
            return 0

    # ═══════════════════════════════════════════════════════════════════════════
    # SARTRE INTEGRATION
    # ═══════════════════════════════════════════════════════════════════════════

    def check_sartre(self, coherence: float, sacred: float, pattern: int) -> bool:
        """Check if should activate based on SARTRE metrics"""
        return self.sartre.check(coherence, sacred, pattern)

    def set_thresholds(
        self,
        coherence_threshold: Optional[float] = None,
        sacred_threshold: Optional[float] = None,
    ) -> None:
        """Update SARTRE thresholds"""
        if coherence_threshold is not None:
            self.sartre.coherence_threshold = coherence_threshold
        if sacred_threshold is not None:
            self.sartre.sacred_threshold = sacred_threshold

    # ═══════════════════════════════════════════════════════════════════════════
    # MODE & STATE
    # ═══════════════════════════════════════════════════════════════════════════

    def is_active(self) -> bool:
        """Check if Pandora is currently active"""
        if self.config.mode == PandoraMode.OFF:
            return False
        if self.config.mode == PandoraMode.FORCED:
            return True
        return True

    def set_mode(self, mode: str) -> None:
        """Set activation mode"""
        self.config.mode = PandoraMode[mode.upper()]

    def set_strength(self, strength: float) -> None:
        """Set injection strength"""
        self.config.injection_strength = max(0.0, min(1.0, strength))

    def clear(self) -> None:
        """Clear all extracted vocabulary"""
        self.ngrams.clear()
        self.total_released = 0
        self.successfully_mapped = 0

    def decay(self, rate: float = 0.9) -> None:
        """Decay n-gram weights"""
        to_remove = []

        for key, ngram in self.ngrams.items():
            ngram.weight *= rate
            if ngram.weight < 0.01:
                to_remove.append(key)

        for key in to_remove:
            del self.ngrams[key]

    # ═══════════════════════════════════════════════════════════════════════════
    # STATS & PERSISTENCE
    # ═══════════════════════════════════════════════════════════════════════════

    def get_stats(self) -> PandoraGGUFStats:
        """Get current statistics"""
        total = len(self.ngrams)
        mapped = sum(1 for ng in self.ngrams.values() if ng.arianna_mapped)
        avg_weight = sum(ng.weight for ng in self.ngrams.values()) / max(1, total)
        avg_freq = sum(ng.frequency for ng in self.ngrams.values()) / max(1, total)

        return PandoraGGUFStats(
            total_ngrams=total,
            mapped_ngrams=mapped,
            avg_weight=avg_weight,
            avg_frequency=avg_freq,
            mode=self.config.mode.name,
            injection_strength=self.config.injection_strength,
            active=self.is_active(),
            model_loaded=self._model is not None,
            model_path=str(self._model_path) if self._model_path else "",
        )

    def save(self, path: str) -> None:
        """Save state to file"""
        data = {
            "config": self.config.to_dict(),
            "ngrams": [
                {
                    "tokens": ng.tokens,
                    "weight": ng.weight,
                    "frequency": ng.frequency,
                    "arianna_mapped": ng.arianna_mapped,
                    "arianna_tokens": ng.arianna_tokens,
                }
                for ng in self.ngrams.values()
            ],
            "total_released": self.total_released,
            "successfully_mapped": self.successfully_mapped,
        }

        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    def load(self, path: str) -> None:
        """Load state from file"""
        with open(path, "r") as f:
            data = json.load(f)

        self.config = PandoraGGUFConfig.from_dict(data["config"])
        self.ngrams = {}

        for ng_data in data["ngrams"]:
            key = tuple(ng_data["tokens"])
            self.ngrams[key] = ReleasedNGram(
                tokens=ng_data["tokens"],
                weight=ng_data["weight"],
                frequency=ng_data["frequency"],
                arianna_mapped=ng_data["arianna_mapped"],
                arianna_tokens=ng_data["arianna_tokens"],
            )

        self.total_released = data["total_released"]
        self.successfully_mapped = data["successfully_mapped"]
