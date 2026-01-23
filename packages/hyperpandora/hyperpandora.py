"""
HyperPandora — Meta-orchestrator for external brain packages

"Choose the right words from the right brain"

Manages multiple Pandora backends and selects the optimal one
based on availability, SARTRE metrics, and task requirements.
"""

import struct
from dataclasses import dataclass, field
from typing import Dict, Optional, Callable, Any, List
from enum import IntEnum
import os
import time


class BrainType(IntEnum):
    """Type of external brain"""
    NONE = 0
    C_PANDORA = 1       # pandora (Pure C, GPT2-30M) - fastest
    TORCH_PANDORA = 2   # pandora-torch (PyTorch, GPT2-distill) - balanced
    GGUF_PANDORA = 3    # pandora-torch-gguf (TinyLlama 1.1B) - richest
    CUSTOM = 4


class SelectionStrategy(IntEnum):
    """Strategy for brain selection"""
    AUTO = 0            # SARTRE-driven selection
    PREFER_FAST = 1     # Prefer C (lightweight, ~60MB)
    PREFER_POWER = 2    # Prefer GGUF (rich vocabulary, ~783MB)
    PREFER_BALANCED = 3 # Prefer PyTorch (balanced)
    ROUND_ROBIN = 4     # Rotate between available
    ADAPTIVE = 5        # Learn from success rates


@dataclass
class BrainInfo:
    """Information about a registered brain"""
    name: str
    brain_type: BrainType
    instance: Any
    priority: int = 0
    capabilities: List[str] = field(default_factory=list)

    # Statistics
    total_calls: int = 0
    total_extracted: int = 0
    avg_latency_ms: float = 0.0
    last_used: float = 0.0


@dataclass
class HyperState:
    """State reported to SARTRE"""
    active: bool = False
    brain_type: BrainType = BrainType.NONE
    brain_name: str = ""
    injection_strength: float = 0.0
    total_ngrams: int = 0
    last_extraction: float = 0.0


class HyperPandora:
    """
    Meta-orchestrator for Pandora packages.

    Features:
    - Register multiple brain backends
    - Auto-select based on SARTRE metrics
    - Report state to SARTRE (via shared memory)
    - Graceful fallback on failures
    """

    def __init__(
        self,
        strategy: SelectionStrategy = SelectionStrategy.AUTO,
        sartre_shm_path: str = "/dev/shm/hyperpandora_state",
    ):
        self.strategy = strategy
        self.sartre_shm_path = sartre_shm_path

        # Registered brains
        self.brains: Dict[str, BrainInfo] = {}
        self._active_brain: Optional[str] = None

        # State
        self.state = HyperState()

        # SARTRE thresholds (mirrored from individual pandoras)
        self.coherence_threshold = 0.3
        self.sacred_threshold = 0.7

        # Statistics
        self.total_selections = 0
        self.selections_by_brain: Dict[str, int] = {}

    # ═══════════════════════════════════════════════════════════════════════════
    # BRAIN REGISTRATION
    # ═══════════════════════════════════════════════════════════════════════════

    def register_brain(
        self,
        name: str,
        brain: Any,
        brain_type: Optional[BrainType] = None,
        priority: int = 0,
        capabilities: Optional[List[str]] = None,
    ) -> None:
        """
        Register a brain backend.

        Args:
            name: Unique identifier (e.g., "c", "torch", "llama")
            brain: Brain instance with process(), apply_to_logits(), etc.
            brain_type: Type enum (auto-detected if None)
            priority: Higher = preferred when multiple available
            capabilities: List of capabilities (e.g., ["lora", "gpu"])
        """
        if brain_type is None:
            # Auto-detect
            if hasattr(brain, '__class__'):
                cls_name = brain.__class__.__name__
                if 'PandoraBox' in cls_name or 'pandora_c' in str(type(brain)):
                    brain_type = BrainType.C_PANDORA
                elif 'PandoraGGUF' in cls_name:
                    brain_type = BrainType.GGUF_PANDORA
                elif 'PandoraTorch' in cls_name:
                    brain_type = BrainType.TORCH_PANDORA
                else:
                    brain_type = BrainType.CUSTOM
            else:
                brain_type = BrainType.CUSTOM

        self.brains[name] = BrainInfo(
            name=name,
            brain_type=brain_type,
            instance=brain,
            priority=priority,
            capabilities=capabilities or [],
        )

        self.selections_by_brain[name] = 0
        print(f"[hyperpandora] Registered brain '{name}' ({brain_type.name})")

    def unregister_brain(self, name: str) -> bool:
        """Remove a brain"""
        if name in self.brains:
            del self.brains[name]
            if self._active_brain == name:
                self._active_brain = None
            return True
        return False

    def list_brains(self) -> List[BrainInfo]:
        """List all registered brains"""
        return list(self.brains.values())

    # ═══════════════════════════════════════════════════════════════════════════
    # BRAIN SELECTION
    # ═══════════════════════════════════════════════════════════════════════════

    def _select_brain(
        self,
        coherence: float = 0.5,
        sacred: float = 0.3,
        pattern: int = 0,  # ResonancePattern value
    ) -> Optional[str]:
        """
        Select optimal brain based on strategy and metrics.

        Returns brain name or None if should deactivate.
        """
        # Check if should deactivate
        if sacred > self.sacred_threshold:
            return None  # Protect voice

        if pattern == 1:  # CRISIS
            return None  # Internal processing

        # No brains available
        if not self.brains:
            return None

        # Helper: find brain by type (highest priority)
        def find_by_type(brain_type: BrainType) -> Optional[str]:
            for name, info in sorted(self.brains.items(), key=lambda x: x[1].priority, reverse=True):
                if info.brain_type == brain_type:
                    return name
            return None

        # Strategy-based selection
        if self.strategy == SelectionStrategy.PREFER_FAST:
            # Prefer C pandora (fastest, ~60MB)
            return find_by_type(BrainType.C_PANDORA) or next(iter(self.brains.keys()))

        elif self.strategy == SelectionStrategy.PREFER_POWER:
            # Prefer GGUF (richest vocabulary, TinyLlama 1.1B)
            return (find_by_type(BrainType.GGUF_PANDORA) or
                    find_by_type(BrainType.TORCH_PANDORA) or
                    next(iter(self.brains.keys())))

        elif self.strategy == SelectionStrategy.PREFER_BALANCED:
            # Prefer PyTorch (balanced speed/quality)
            return (find_by_type(BrainType.TORCH_PANDORA) or
                    find_by_type(BrainType.C_PANDORA) or
                    next(iter(self.brains.keys())))

        elif self.strategy == SelectionStrategy.ROUND_ROBIN:
            # Rotate through brains
            names = list(self.brains.keys())
            if not self._active_brain or self._active_brain not in names:
                return names[0]
            idx = names.index(self._active_brain)
            return names[(idx + 1) % len(names)]

        elif self.strategy == SelectionStrategy.ADAPTIVE:
            # Select based on success rate (total_extracted / total_calls)
            best_name = None
            best_rate = -1.0
            for name, info in self.brains.items():
                if info.total_calls > 0:
                    rate = info.total_extracted / info.total_calls
                    if rate > best_rate:
                        best_rate = rate
                        best_name = name
            return best_name or next(iter(self.brains.keys()))

        else:  # AUTO - SARTRE-driven selection
            # Low coherence - need words fast
            if coherence < self.coherence_threshold:
                return find_by_type(BrainType.C_PANDORA) or next(iter(self.brains.keys()))

            # EMERGENCE - creative expansion, use richest vocabulary
            if pattern == 3:
                return (find_by_type(BrainType.GGUF_PANDORA) or
                        find_by_type(BrainType.TORCH_PANDORA) or
                        next(iter(self.brains.keys())))

            # TRANSCENDENCE - bridging, use balanced
            if pattern == 4:
                return (find_by_type(BrainType.TORCH_PANDORA) or
                        next(iter(self.brains.keys())))

            # Normal state - maintain current or highest priority
            if self._active_brain and self._active_brain in self.brains:
                return self._active_brain

            sorted_brains = sorted(self.brains.items(), key=lambda x: x[1].priority, reverse=True)
            return sorted_brains[0][0] if sorted_brains else None

    # ═══════════════════════════════════════════════════════════════════════════
    # PROCESSING
    # ═══════════════════════════════════════════════════════════════════════════

    def process(
        self,
        text: str,
        arianna_encode: Callable[[str], int],
        coherence: float = 0.5,
        sacred: float = 0.3,
        pattern: int = 0,
        max_tokens: int = 50,
    ) -> int:
        """
        Process text through selected brain.

        Returns number of n-grams extracted.
        """
        # Select brain
        brain_name = self._select_brain(coherence, sacred, pattern)

        if not brain_name:
            self._update_state(active=False)
            return 0

        return self.process_with(brain_name, text, arianna_encode, max_tokens)

    def process_with(
        self,
        brain_name: str,
        text: str,
        arianna_encode: Callable[[str], int],
        max_tokens: int = 50,
    ) -> int:
        """
        Process with specific brain.

        Returns number of n-grams extracted.
        """
        if brain_name not in self.brains:
            return 0

        info = self.brains[brain_name]
        start_time = time.time()

        try:
            # Call brain's process method
            if hasattr(info.instance, 'process'):
                extracted = info.instance.process(text, arianna_encode, max_tokens)
            else:
                # Fallback for C pandora (different API)
                extracted = 0

            # Update stats
            elapsed = (time.time() - start_time) * 1000
            info.total_calls += 1
            info.total_extracted += extracted
            info.avg_latency_ms = (info.avg_latency_ms * (info.total_calls - 1) + elapsed) / info.total_calls
            info.last_used = time.time()

            self._active_brain = brain_name
            self.total_selections += 1
            self.selections_by_brain[brain_name] = self.selections_by_brain.get(brain_name, 0) + 1

            # Update state for SARTRE
            self._update_state(
                active=True,
                brain_type=info.brain_type,
                brain_name=brain_name,
            )

            return extracted

        except Exception as e:
            print(f"[hyperpandora] Error in brain '{brain_name}': {e}")
            # Try fallback
            for name, other_info in self.brains.items():
                if name != brain_name:
                    print(f"[hyperpandora] Falling back to '{name}'")
                    return self.process_with(name, text, arianna_encode, max_tokens)
            return 0

    def apply_to_logits(
        self,
        logits: Any,
        context_tokens: List[int],
        vocab_size: Optional[int] = None,
    ) -> Any:
        """
        Apply active brain's vocabulary to logits.
        """
        if not self._active_brain or self._active_brain not in self.brains:
            return logits

        brain = self.brains[self._active_brain].instance

        if hasattr(brain, 'apply_to_logits'):
            return brain.apply_to_logits(logits, context_tokens, vocab_size)

        return logits

    # ═══════════════════════════════════════════════════════════════════════════
    # STATE MANAGEMENT
    # ═══════════════════════════════════════════════════════════════════════════

    def _update_state(
        self,
        active: bool = False,
        brain_type: BrainType = BrainType.NONE,
        brain_name: str = "",
    ) -> None:
        """Update state and write to shared memory for SARTRE"""
        self.state.active = active
        self.state.brain_type = brain_type
        self.state.brain_name = brain_name

        if active and brain_name in self.brains:
            brain = self.brains[brain_name].instance
            if hasattr(brain, 'config'):
                self.state.injection_strength = getattr(brain.config, 'injection_strength', 0.2)
            if hasattr(brain, 'ngrams'):
                self.state.total_ngrams = len(brain.ngrams)

        self.state.last_extraction = time.time()

        # Write to shared memory for SARTRE
        self._write_sartre_state()

    def _write_sartre_state(self) -> None:
        """Write state to shared memory"""
        try:
            # HyperPandoraState struct:
            # int32 active
            # int32 brain_type
            # float32 injection_strength
            # int32 total_ngrams
            # float64 last_extraction
            data = struct.pack(
                "iifid",
                1 if self.state.active else 0,
                self.state.brain_type.value,
                self.state.injection_strength,
                self.state.total_ngrams,
                self.state.last_extraction,
            )

            with open(self.sartre_shm_path, "wb") as f:
                f.write(data)

        except Exception:
            pass  # Silently fail if can't write

    def get_state(self) -> HyperState:
        """Get current state"""
        return self.state

    def get_active_info(self) -> Optional[BrainInfo]:
        """Get info about active brain"""
        if self._active_brain and self._active_brain in self.brains:
            return self.brains[self._active_brain]
        return None

    # ═══════════════════════════════════════════════════════════════════════════
    # CONFIGURATION
    # ═══════════════════════════════════════════════════════════════════════════

    def set_strategy(self, strategy: SelectionStrategy) -> None:
        """Set selection strategy"""
        self.strategy = strategy

    def set_thresholds(
        self,
        coherence_threshold: Optional[float] = None,
        sacred_threshold: Optional[float] = None,
    ) -> None:
        """Update SARTRE thresholds"""
        if coherence_threshold is not None:
            self.coherence_threshold = coherence_threshold
        if sacred_threshold is not None:
            self.sacred_threshold = sacred_threshold

        # Propagate to all brains
        for info in self.brains.values():
            if hasattr(info.instance, 'set_thresholds'):
                info.instance.set_thresholds(coherence_threshold, sacred_threshold)

    def deactivate_all(self) -> None:
        """Deactivate all brains"""
        self._active_brain = None
        self._update_state(active=False)

        for info in self.brains.values():
            if hasattr(info.instance, 'set_mode'):
                info.instance.set_mode('off')

    def force_brain(self, name: str) -> bool:
        """Force a specific brain to be active"""
        if name not in self.brains:
            return False

        self._active_brain = name
        self._update_state(
            active=True,
            brain_type=self.brains[name].brain_type,
            brain_name=name,
        )
        return True

    # ═══════════════════════════════════════════════════════════════════════════
    # STATISTICS
    # ═══════════════════════════════════════════════════════════════════════════

    def get_stats(self) -> Dict[str, Any]:
        """Get overall statistics"""
        return {
            "total_selections": self.total_selections,
            "active_brain": self._active_brain,
            "strategy": self.strategy.name,
            "brains": {
                name: {
                    "type": info.brain_type.name,
                    "priority": info.priority,
                    "total_calls": info.total_calls,
                    "total_extracted": info.total_extracted,
                    "avg_latency_ms": info.avg_latency_ms,
                    "selection_count": self.selections_by_brain.get(name, 0),
                }
                for name, info in self.brains.items()
            },
        }


# ═══════════════════════════════════════════════════════════════════════════════
# CONVENIENCE FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def create_hyperpandora_with_defaults() -> HyperPandora:
    """
    Create HyperPandora with default brains if available.
    """
    hyper = HyperPandora()

    # Try to register pandora-torch
    try:
        from pandora_torch import PandoraTorch
        if PandoraTorch is not None:
            torch_pandora = PandoraTorch(mode="auto")
            hyper.register_brain(
                "torch",
                torch_pandora,
                BrainType.TORCH_PANDORA,
                priority=10,
                capabilities=["lora", "batched"],
            )
    except ImportError:
        pass

    # C pandora would be registered via FFI or subprocess

    return hyper


def read_hyperpandora_state(shm_path: str = "/dev/shm/hyperpandora_state") -> Optional[HyperState]:
    """
    Read HyperPandora state from shared memory.

    For SARTRE integration.
    """
    try:
        with open(shm_path, "rb") as f:
            data = f.read(24)

        if len(data) < 24:
            return None

        active, brain_type, injection, ngrams, last = struct.unpack("iifid", data)

        return HyperState(
            active=bool(active),
            brain_type=BrainType(brain_type),
            injection_strength=injection,
            total_ngrams=ngrams,
            last_extraction=last,
        )
    except Exception:
        return None
