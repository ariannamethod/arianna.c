"""
Pandora-Torch — PyTorch vocabulary extraction

"Take the words, leave the voice"

Uses GPT2-distill (Stanley) as external brain with full SARTRE integration.
"""

import torch
import torch.nn.functional as F
from typing import List, Dict, Tuple, Optional, Callable
from dataclasses import dataclass, field
from collections import defaultdict
import json
import struct
import time

from .config import PandoraTorchConfig, PandoraMode
from .sartre import SARTREChecker, VagusState, ResonancePattern


@dataclass
class ReleasedNGram:
    """N-gram extracted from external brain"""
    tokens: List[int]
    weight: float = 0.1
    frequency: int = 1
    arianna_mapped: bool = False
    arianna_tokens: List[int] = field(default_factory=list)

    @property
    def length(self) -> int:
        return len(self.tokens)


@dataclass
class PandoraStats:
    """Statistics for Pandora state"""
    total_ngrams: int
    mapped_ngrams: int
    avg_weight: float
    avg_frequency: float
    mode: str
    injection_strength: float
    active: bool
    last_reason: str = ""


class PandoraTorch:
    """
    PyTorch vocabulary extraction with LoRA delta support.

    Features:
    - GPT2-distill inference
    - N-gram extraction and mapping
    - Logit injection
    - LoRA delta extraction
    - SARTRE-driven activation
    """

    def __init__(
        self,
        config: Optional[PandoraTorchConfig] = None,
        weights_path: Optional[str] = None,
        mode: str = "auto",
    ):
        self.config = config or PandoraTorchConfig()
        if weights_path:
            self.config.weights_path = weights_path

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
        self._model = None
        self._tokenizer = None
        self._device = self.config.device

    # ═══════════════════════════════════════════════════════════════════════════
    # MODEL LOADING
    # ═══════════════════════════════════════════════════════════════════════════

    def _load_model(self) -> None:
        """Lazy load the external brain model"""
        if self._model is not None:
            return

        try:
            # Try to import Stanley's transformer
            from stanley.inference import StanleyTransformer
            self._model = StanleyTransformer()
            self._model.load_base_weights(self.config.weights_path)
            print(f"[pandora-torch] Loaded Stanley transformer")
        except ImportError:
            # Fallback to simple GPT2 from transformers
            try:
                from transformers import GPT2LMHeadModel, GPT2Tokenizer
                self._model = GPT2LMHeadModel.from_pretrained("distilgpt2")
                self._tokenizer = GPT2Tokenizer.from_pretrained("distilgpt2")
                self._model.eval()
                print(f"[pandora-torch] Loaded distilgpt2 from transformers")
            except ImportError:
                print("[pandora-torch] WARNING: No model available, using mock mode")
                self._model = "mock"

    def _ensure_model(self) -> bool:
        """Ensure model is loaded"""
        if self._model is None:
            self._load_model()
        return self._model is not None and self._model != "mock"

    # ═══════════════════════════════════════════════════════════════════════════
    # N-GRAM EXTRACTION
    # ═══════════════════════════════════════════════════════════════════════════

    def extract(
        self,
        tokens: List[int],
        min_n: Optional[int] = None,
        max_n: Optional[int] = None,
    ) -> int:
        """
        Extract n-grams from token sequence.

        Returns number of new n-grams added.
        """
        if not self.is_active():
            return 0

        min_n = min_n or self.config.min_ngram
        max_n = max_n or self.config.max_ngram

        added = 0

        for n in range(min_n, max_n + 1):
            for start in range(len(tokens) - n + 1):
                ngram = tuple(tokens[start:start + n])

                if ngram in self.ngrams:
                    # Boost existing
                    self.ngrams[ngram].frequency += 1
                    self.ngrams[ngram].weight = min(1.0, self.ngrams[ngram].weight + 0.01)
                elif len(self.ngrams) < self.config.max_ngrams:
                    # Add new
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
        """
        Map extracted n-grams to Arianna vocabulary.

        Returns number successfully mapped.
        """
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
        logits: torch.Tensor,
        context_tokens: List[int],
        vocab_size: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Apply released vocabulary to logits.

        Boosts tokens that match extracted n-gram patterns.
        """
        if not self.is_active():
            return logits

        if self.config.injection_strength <= 0:
            return logits

        vocab_size = vocab_size or logits.shape[-1]
        boosted = logits.clone()

        for ngram in self.ngrams.values():
            if not ngram.arianna_mapped:
                continue
            if ngram.frequency < self.config.min_frequency:
                continue

            prefix_len = ngram.length - 1

            if prefix_len == 0:
                # Unigram boost
                tok = ngram.arianna_tokens[0]
                if 0 <= tok < vocab_size:
                    boosted[..., tok] += ngram.weight * self.config.injection_strength * 0.5
                continue

            if prefix_len > len(context_tokens):
                continue

            # Check prefix match
            prefix = ngram.arianna_tokens[:-1]
            context_suffix = context_tokens[-prefix_len:]

            if prefix == context_suffix:
                # Boost continuation token
                next_tok = ngram.arianna_tokens[-1]
                if 0 <= next_tok < vocab_size:
                    boost = ngram.weight * self.config.injection_strength
                    boost *= (1.0 + 0.1 * ngram.frequency)
                    boosted[..., next_tok] += boost

        return boosted

    def suggest_continuation(self, context_tokens: List[int]) -> int:
        """Suggest next token based on n-gram patterns"""
        if not self.is_active():
            return -1

        best_token = -1
        best_score = 0.0

        for ngram in self.ngrams.values():
            if not ngram.arianna_mapped:
                continue
            if ngram.length < 2:
                continue

            prefix_len = ngram.length - 1
            if prefix_len > len(context_tokens):
                continue

            prefix = ngram.arianna_tokens[:-1]
            context_suffix = context_tokens[-prefix_len:]

            if prefix == context_suffix:
                score = ngram.weight * ngram.frequency
                if score > best_score:
                    best_score = score
                    best_token = ngram.arianna_tokens[-1]

        return best_token

    # ═══════════════════════════════════════════════════════════════════════════
    # FULL PIPELINE
    # ═══════════════════════════════════════════════════════════════════════════

    def process(
        self,
        text: str,
        arianna_encode: Callable[[str], int],
        max_tokens: Optional[int] = None,
    ) -> int:
        """
        Full pipeline: generate from brain, extract, map.

        Returns number of new n-grams.
        """
        if not self.is_active():
            return 0

        if not self._ensure_model():
            return 0

        max_tokens = max_tokens or self.config.max_generate

        try:
            # Generate from external brain
            if hasattr(self._model, 'generate'):
                # Stanley transformer
                input_ids = self._tokenizer.encode(text) if self._tokenizer else [ord(c) for c in text]
                output_ids = self._model.generate(
                    input_ids,
                    max_new_tokens=max_tokens,
                    temperature=self.config.temperature,
                )
            else:
                # Transformers model
                inputs = self._tokenizer(text, return_tensors="pt")
                outputs = self._model.generate(
                    inputs.input_ids,
                    max_new_tokens=max_tokens,
                    temperature=self.config.temperature,
                    do_sample=True,
                    top_k=self.config.top_k,
                    pad_token_id=self._tokenizer.eos_token_id,
                )
                output_ids = outputs[0].tolist()

            # Extract n-grams
            added = self.extract(output_ids)

            # Map to Arianna vocab
            def brain_decode(tok_id: int) -> Optional[str]:
                if self._tokenizer:
                    return self._tokenizer.decode([tok_id])
                return chr(tok_id) if 32 <= tok_id < 127 else None

            self.map_to_arianna(brain_decode, arianna_encode)

            return added

        except Exception as e:
            print(f"[pandora-torch] process error: {e}")
            return 0

    # ═══════════════════════════════════════════════════════════════════════════
    # LORA DELTA EXTRACTION
    # ═══════════════════════════════════════════════════════════════════════════

    def extract_lora_deltas(
        self,
        prompt: str,
        response: str,
    ) -> Optional[Dict[str, torch.Tensor]]:
        """
        Extract LoRA deltas from a prompt-response pair.

        This creates training deltas that can be applied to Arianna.
        """
        try:
            from stanley.trainer import compute_lora_delta, LoRAConfig

            lora_config = LoRAConfig(
                rank=self.config.lora_rank,
                alpha=self.config.lora_alpha,
                dropout=self.config.lora_dropout,
            )

            # Combine prompt and response
            text = f"{prompt}\n{response}"

            # Compute deltas
            deltas = compute_lora_delta(
                text=text,
                config=lora_config,
            )

            return deltas

        except ImportError:
            print("[pandora-torch] LoRA extraction requires Stanley trainer module")
            return None

    # ═══════════════════════════════════════════════════════════════════════════
    # SARTRE INTEGRATION
    # ═══════════════════════════════════════════════════════════════════════════

    def check_sartre(
        self,
        coherence: float,
        sacred: float,
        pattern: ResonancePattern,
    ) -> bool:
        """Check if should activate based on SARTRE metrics"""
        return self.sartre.check(coherence, sacred, pattern)

    def update_sartre_state(self, state: VagusState) -> bool:
        """Update with full Vagus state"""
        return self.sartre.check_state(state)

    def set_thresholds(
        self,
        coherence_threshold: Optional[float] = None,
        sacred_threshold: Optional[float] = None,
    ) -> None:
        """Update SARTRE thresholds"""
        self.sartre.update_thresholds(coherence_threshold, sacred_threshold)

    # ═══════════════════════════════════════════════════════════════════════════
    # MODE & STATE
    # ═══════════════════════════════════════════════════════════════════════════

    def is_active(self) -> bool:
        """Check if Pandora is currently active"""
        if self.config.mode == PandoraMode.OFF:
            return False
        if self.config.mode == PandoraMode.FORCED:
            return True
        # AUTO mode
        return True  # Controlled externally via check_sartre

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
        """Decay n-gram weights, remove weak ones"""
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

    def get_stats(self) -> PandoraStats:
        """Get current statistics"""
        total = len(self.ngrams)
        mapped = sum(1 for ng in self.ngrams.values() if ng.arianna_mapped)
        avg_weight = sum(ng.weight for ng in self.ngrams.values()) / max(1, total)
        avg_freq = sum(ng.frequency for ng in self.ngrams.values()) / max(1, total)

        return PandoraStats(
            total_ngrams=total,
            mapped_ngrams=mapped,
            avg_weight=avg_weight,
            avg_frequency=avg_freq,
            mode=self.config.mode.name,
            injection_strength=self.config.injection_strength,
            active=self.is_active(),
            last_reason=self.sartre.get_activation_reason(),
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

        self.config = PandoraTorchConfig.from_dict(data["config"])
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

        # Update SARTRE thresholds
        self.sartre.update_thresholds(
            self.config.coherence_threshold,
            self.config.sacred_threshold,
        )
