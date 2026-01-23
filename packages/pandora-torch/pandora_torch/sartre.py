"""
SARTRE Integration for Pandora-Torch

Metric-driven activation based on Vagus/Locus state.
"""

from enum import IntEnum
from dataclasses import dataclass
from typing import Optional


class ResonancePattern(IntEnum):
    """Resonance patterns from Locus"""
    NONE = 0
    CRISIS = 1
    DISSOLUTION = 2
    EMERGENCE = 3
    TRANSCENDENCE = 4


@dataclass
class VagusState:
    """State from VagusSharedState"""
    arousal: float = 0.5
    valence: float = 0.5
    coherence: float = 0.5
    warmth: float = 0.5
    void: float = 0.0
    sacred: float = 0.0
    tension: float = 0.0
    flow: float = 0.0
    stillness: float = 0.5
    pattern: ResonancePattern = ResonancePattern.NONE


class SARTREChecker:
    """
    Check if Pandora should be active based on SARTRE metrics.

    Activation rules:
    - Low coherence (< threshold) -> ACTIVATE (need vocabulary boost)
    - EMERGENCE pattern -> ACTIVATE (creative expansion)
    - TRANSCENDENCE pattern -> ACTIVATE (bridging)
    - High sacred (> threshold) -> DEACTIVATE (protect voice)
    - CRISIS pattern -> DEACTIVATE (internal processing)
    """

    def __init__(
        self,
        coherence_threshold: float = 0.3,
        sacred_threshold: float = 0.7,
    ):
        self.coherence_threshold = coherence_threshold
        self.sacred_threshold = sacred_threshold
        self._last_state: Optional[VagusState] = None
        self._last_decision: bool = False

    def check(
        self,
        coherence: float,
        sacred: float,
        pattern: ResonancePattern,
    ) -> bool:
        """
        Check if Pandora should be active.

        Args:
            coherence: Field coherence (0-1)
            sacred: Sacred chamber activation (0-1)
            pattern: Current resonance pattern

        Returns:
            True if Pandora should activate
        """
        # Deactivate on high sacred (protect voice)
        if sacred > self.sacred_threshold:
            self._last_decision = False
            return False

        # Deactivate on CRISIS (internal processing)
        if pattern == ResonancePattern.CRISIS:
            self._last_decision = False
            return False

        # Activate on low coherence (need vocabulary boost)
        if coherence < self.coherence_threshold:
            self._last_decision = True
            return True

        # Activate on EMERGENCE (creative expansion)
        if pattern == ResonancePattern.EMERGENCE:
            self._last_decision = True
            return True

        # Activate on TRANSCENDENCE (bridging)
        if pattern == ResonancePattern.TRANSCENDENCE:
            self._last_decision = True
            return True

        # Default: maintain previous state
        return self._last_decision

    def check_state(self, state: VagusState) -> bool:
        """Check using full VagusState"""
        self._last_state = state
        return self.check(
            coherence=state.coherence,
            sacred=state.sacred,
            pattern=state.pattern,
        )

    def get_activation_reason(self) -> str:
        """Get human-readable reason for last decision"""
        if self._last_state is None:
            return "no state"

        state = self._last_state

        if state.sacred > self.sacred_threshold:
            return f"high sacred ({state.sacred:.2f} > {self.sacred_threshold})"

        if state.pattern == ResonancePattern.CRISIS:
            return "CRISIS pattern (internal processing)"

        if state.coherence < self.coherence_threshold:
            return f"low coherence ({state.coherence:.2f} < {self.coherence_threshold})"

        if state.pattern == ResonancePattern.EMERGENCE:
            return "EMERGENCE pattern (creative expansion)"

        if state.pattern == ResonancePattern.TRANSCENDENCE:
            return "TRANSCENDENCE pattern (bridging)"

        return "maintaining previous state"

    def update_thresholds(
        self,
        coherence_threshold: Optional[float] = None,
        sacred_threshold: Optional[float] = None,
    ) -> None:
        """Update activation thresholds"""
        if coherence_threshold is not None:
            self.coherence_threshold = coherence_threshold
        if sacred_threshold is not None:
            self.sacred_threshold = sacred_threshold


def read_vagus_state_from_shm(shm_path: str = "/dev/shm/vagus_state") -> Optional[VagusState]:
    """
    Read VagusSharedState from shared memory.

    This connects to the actual Vagus nerve output.
    """
    import struct

    try:
        with open(shm_path, "rb") as f:
            data = f.read(64)  # VagusSharedState size

        if len(data) < 64:
            return None

        # Parse struct (matches vagus.h VagusSharedState)
        values = struct.unpack("ffffffff", data[:32])
        pattern = struct.unpack("i", data[32:36])[0]

        return VagusState(
            arousal=values[0],
            valence=values[1],
            coherence=values[2],
            warmth=values[3],
            void=values[4],
            sacred=values[5],
            tension=values[6],
            flow=values[7],
            pattern=ResonancePattern(pattern) if 0 <= pattern <= 4 else ResonancePattern.NONE,
        )
    except (FileNotFoundError, struct.error):
        return None
