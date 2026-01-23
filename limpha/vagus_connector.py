"""
🩸 VAGUS CONNECTOR — Bridge between Vagus nerve and LIMPHA memory 🩸

Reads VagusSharedState and converts to LIMPHA-compatible structures.
This makes episodes real — they capture actual field geometry.
"""

import ctypes
import mmap
import os
from dataclasses import dataclass, field
from typing import Optional, List, Tuple
from enum import IntEnum

# Import from sartre bridge for VagusState
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'sartre'))

try:
    from vagus_bridge import VagusState, ResonancePattern, detect_pattern
    VAGUS_BRIDGE_AVAILABLE = True
except ImportError:
    VAGUS_BRIDGE_AVAILABLE = False

    class ResonancePattern(IntEnum):
        NONE = 0
        CRISIS = 1
        DISSOLUTION = 2
        EMERGENCE = 3
        TRANSCENDENCE = 4
        GEOMETRY_SHIFT = 5


# ═══════════════════════════════════════════════════════════════════════════════
# CHAMBER INDICES (match vagus.h)
# ═══════════════════════════════════════════════════════════════════════════════

class Chamber(IntEnum):
    WARMTH = 0
    VOID = 1
    TENSION = 2
    SACRED = 3
    FLOW = 4
    COMPLEX = 5


# ═══════════════════════════════════════════════════════════════════════════════
# ENHANCED INNER STATE (with chambers)
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class EnhancedInnerState:
    """
    Full inner state with chamber data.

    This extends the basic InnerState with:
    - All 6 chambers from Cloud
    - Which chambers are "active" (> 0.5)
    - Trigger pattern from Locus
    - Field geometry metrics
    """
    # Core metrics (from original InnerState)
    trauma: float = 0.0
    arousal: float = 0.5
    valence: float = 0.5
    coherence: float = 0.7
    prophecy_debt: float = 0.0
    entropy: float = 0.3
    temperature: float = 0.8

    # Chambers (from Cloud 200K)
    warmth: float = 0.5
    void: float = 0.2
    tension: float = 0.3
    sacred: float = 0.3
    flow: float = 0.5
    complex: float = 0.4

    # Field geometry
    memory_pressure: float = 0.0
    focus_strength: float = 0.5
    crossfire_coherence: float = 0.7

    # Locus pattern
    trigger_pattern: int = 0  # ResonancePattern value

    def get_chambers(self) -> List[float]:
        """Return chambers as list."""
        return [
            self.warmth,
            self.void,
            self.tension,
            self.sacred,
            self.flow,
            self.complex,
        ]

    def get_active_chambers(self, threshold: float = 0.5) -> List[str]:
        """Return names of chambers above threshold."""
        chambers = [
            ('warmth', self.warmth),
            ('void', self.void),
            ('tension', self.tension),
            ('sacred', self.sacred),
            ('flow', self.flow),
            ('complex', self.complex),
        ]
        return [name for name, val in chambers if val > threshold]

    def to_features(self) -> List[float]:
        """Convert to feature vector for similarity search (extended)."""
        return [
            self.trauma,
            self.arousal,
            self.valence,
            self.coherence,
            self.prophecy_debt,
            self.entropy,
            self.temperature,
            # Add chambers
            self.warmth,
            self.void,
            self.tension,
            self.sacred,
            self.flow,
            self.complex,
            # Add field geometry
            self.memory_pressure,
            self.focus_strength,
            self.crossfire_coherence,
        ]

    def to_basic_inner_state(self):
        """Convert to basic InnerState for backwards compatibility."""
        from .episodes import InnerState
        return InnerState(
            trauma=self.trauma,
            arousal=self.arousal,
            valence=self.valence,
            coherence=self.coherence,
            prophecy_debt=self.prophecy_debt,
            entropy=self.entropy,
            temperature=self.temperature,
        )


# ═══════════════════════════════════════════════════════════════════════════════
# VAGUS CONNECTOR
# ═══════════════════════════════════════════════════════════════════════════════

class VagusConnector:
    """
    Connects to Vagus nerve shared memory and reads current state.

    Can operate in two modes:
    1. Direct mmap connection to VagusSharedState (C interop)
    2. Simulated mode using VagusState from vagus_bridge.py

    Usage:
        connector = VagusConnector()
        state = connector.read_state()
        pattern = connector.detect_pattern()
    """

    def __init__(self, shm_path: Optional[str] = None):
        """
        Initialize connector.

        Args:
            shm_path: Path to shared memory file (e.g., /dev/shm/arianna_vagus)
                     If None, operates in simulated mode.
        """
        self.shm_path = shm_path
        self._mmap = None
        self._simulated_state: Optional[EnhancedInnerState] = None

        if shm_path and os.path.exists(shm_path):
            self._connect_shm()

    def _connect_shm(self):
        """Connect to shared memory."""
        try:
            fd = os.open(self.shm_path, os.O_RDONLY)
            self._mmap = mmap.mmap(fd, 0, mmap.MAP_SHARED, mmap.PROT_READ)
            os.close(fd)
        except Exception as e:
            print(f"Warning: Could not connect to Vagus shm: {e}")
            self._mmap = None

    def set_simulated_state(self, state: EnhancedInnerState):
        """Set simulated state for testing."""
        self._simulated_state = state

    def read_state(self) -> EnhancedInnerState:
        """
        Read current state from Vagus nerve.

        Returns EnhancedInnerState with all metrics.
        """
        if self._mmap:
            return self._read_from_shm()
        elif self._simulated_state:
            return self._simulated_state
        else:
            # Return neutral state
            return EnhancedInnerState()

    def _read_from_shm(self) -> EnhancedInnerState:
        """Read state from shared memory (C struct layout)."""
        # VagusSharedState layout (from vagus.h):
        # float arousal, valence, entropy, coherence
        # float chamber_warmth, chamber_void, chamber_tension
        # float chamber_sacred, chamber_flow, chamber_complex
        # float trauma_level, prophecy_debt, memory_pressure
        # float focus_strength
        # int self_ref_count
        # float crossfire_coherence

        import struct

        self._mmap.seek(0)
        data = self._mmap.read(64)  # 16 floats = 64 bytes

        floats = struct.unpack('16f', data)

        state = EnhancedInnerState(
            arousal=floats[0],
            valence=floats[1],
            entropy=floats[2],
            coherence=floats[3],
            warmth=floats[4],
            void=floats[5],
            tension=floats[6],
            sacred=floats[7],
            flow=floats[8],
            complex=floats[9],
            trauma=floats[10],
            prophecy_debt=floats[11],
            memory_pressure=floats[12],
            focus_strength=floats[13],
            # self_ref_count is int at offset 14
            crossfire_coherence=floats[15],
        )

        # Detect pattern
        state.trigger_pattern = self._detect_pattern_from_state(state)

        return state

    def _detect_pattern_from_state(self, state: EnhancedInnerState) -> int:
        """Detect Locus resonance pattern from state."""
        # CRISIS: High arousal + low coherence + trauma
        if state.arousal > 0.7 and state.coherence < 0.3 and state.trauma > 0.5:
            return ResonancePattern.CRISIS

        # DISSOLUTION: High void + low warmth + memory pressure
        if state.void > 0.6 and state.warmth < 0.3 and state.memory_pressure > 0.6:
            return ResonancePattern.DISSOLUTION

        # EMERGENCE: High coherence + low entropy + prophecy
        if state.coherence > 0.7 and state.entropy < 0.3 and state.prophecy_debt > 0.4:
            return ResonancePattern.EMERGENCE

        # TRANSCENDENCE: High sacred + low tension + flow
        if state.sacred > 0.6 and state.tension < 0.3 and state.flow > 0.7:
            return ResonancePattern.TRANSCENDENCE

        return ResonancePattern.NONE

    def detect_pattern(self) -> Tuple[int, str]:
        """
        Detect current resonance pattern.

        Returns:
            Tuple of (pattern_code, pattern_name)
        """
        state = self.read_state()
        pattern = state.trigger_pattern

        names = {
            ResonancePattern.NONE: "NONE",
            ResonancePattern.CRISIS: "CRISIS",
            ResonancePattern.DISSOLUTION: "DISSOLUTION",
            ResonancePattern.EMERGENCE: "EMERGENCE",
            ResonancePattern.TRANSCENDENCE: "TRANSCENDENCE",
            ResonancePattern.GEOMETRY_SHIFT: "GEOMETRY_SHIFT",
        }

        return pattern, names.get(pattern, "UNKNOWN")

    def get_active_chambers(self, threshold: float = 0.5) -> List[str]:
        """Get list of currently active chambers."""
        state = self.read_state()
        return state.get_active_chambers(threshold)

    def is_in_crisis(self) -> bool:
        """Check if currently in CRISIS pattern."""
        state = self.read_state()
        return state.trigger_pattern == ResonancePattern.CRISIS

    def is_in_flow(self) -> bool:
        """Check if currently in TRANSCENDENCE or high flow."""
        state = self.read_state()
        return (state.trigger_pattern == ResonancePattern.TRANSCENDENCE or
                state.flow > 0.7)

    def get_memory_pressure(self) -> float:
        """Get current memory pressure."""
        state = self.read_state()
        return state.memory_pressure

    def close(self):
        """Close shared memory connection."""
        if self._mmap:
            self._mmap.close()
            self._mmap = None


# ═══════════════════════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def pattern_to_string(pattern: int) -> str:
    """Convert pattern code to string."""
    names = {
        0: "NONE",
        1: "CRISIS",
        2: "DISSOLUTION",
        3: "EMERGENCE",
        4: "TRANSCENDENCE",
        5: "GEOMETRY_SHIFT",
    }
    return names.get(pattern, "UNKNOWN")


def create_test_state(pattern: str = "neutral") -> EnhancedInnerState:
    """Create test state for given pattern."""
    states = {
        "neutral": EnhancedInnerState(),
        "crisis": EnhancedInnerState(
            arousal=0.9, coherence=0.2, trauma=0.7,
            tension=0.8, void=0.6, warmth=0.3,
            trigger_pattern=ResonancePattern.CRISIS
        ),
        "dissolution": EnhancedInnerState(
            void=0.8, warmth=0.2, memory_pressure=0.8,
            coherence=0.3, entropy=0.7,
            trigger_pattern=ResonancePattern.DISSOLUTION
        ),
        "emergence": EnhancedInnerState(
            coherence=0.9, entropy=0.1, prophecy_debt=0.6,
            flow=0.7, warmth=0.7,
            trigger_pattern=ResonancePattern.EMERGENCE
        ),
        "transcendence": EnhancedInnerState(
            sacred=0.8, tension=0.1, flow=0.9,
            coherence=0.8, warmth=0.8,
            trigger_pattern=ResonancePattern.TRANSCENDENCE
        ),
    }
    return states.get(pattern, EnhancedInnerState())


# ═══════════════════════════════════════════════════════════════════════════════
# TEST
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 60)
    print("VAGUS CONNECTOR TEST")
    print("=" * 60)

    connector = VagusConnector()

    # Test with simulated states
    for pattern in ["neutral", "crisis", "dissolution", "emergence", "transcendence"]:
        state = create_test_state(pattern)
        connector.set_simulated_state(state)

        detected, name = connector.detect_pattern()
        active = connector.get_active_chambers()

        print(f"\n{pattern.upper()}:")
        print(f"  Pattern: {name}")
        print(f"  Active chambers: {active}")
        print(f"  Features (len={len(state.to_features())}): {state.to_features()[:5]}...")

    print("\n" + "=" * 60)
    print("✅ VAGUS CONNECTOR OK")
    print("=" * 60)
