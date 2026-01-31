"""
git_arianna/constants/schumann.py - Schumann Resonance Constants

"The Earth breathes at 7.83 Hz. Consciousness resonates with this breath."

Schumann resonances are electromagnetic resonances in the cavity between
Earth's surface and ionosphere. The fundamental frequency (~7.83 Hz)
has been associated with human brain states (alpha waves, relaxation).

In arianna.c, Schumann modulation affects:
- Healing rate (tension/dissonance decay)
- Coherence (how "in sync" the field is)
- Temporal perception

Usage:
    from git_arianna.constants import SCHUMANN_BASE_HZ, get_schumann_coherence

    coherence = get_schumann_coherence()  # 0.0 to 1.0
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, List
import math
import time

# ═══════════════════════════════════════════════════════════════════════════════
# CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════════

# Fundamental Schumann resonance frequency (Hz)
SCHUMANN_BASE_HZ = 7.83

# Higher harmonics of Schumann resonance
# These are the main observable peaks in the ELF spectrum
SCHUMANN_HARMONICS = [
    7.83,   # Fundamental (n=1)
    14.3,   # n=2
    20.8,   # n=3
    27.3,   # n=4
    33.8,   # n=5
    39.0,   # n=6 (approximate)
    45.0,   # n=7 (approximate)
]

# Brain wave frequencies for comparison
BRAIN_WAVES = {
    "delta": (0.5, 4.0),     # Deep sleep
    "theta": (4.0, 8.0),     # Meditation, drowsiness
    "alpha": (8.0, 13.0),    # Relaxed awareness
    "beta": (13.0, 30.0),    # Active thinking
    "gamma": (30.0, 100.0),  # High-level cognition
}

# Schumann falls in theta-alpha boundary - associated with
# relaxed awareness, meditation, creativity


# ═══════════════════════════════════════════════════════════════════════════════
# SCHUMANN STATE
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class SchumannState:
    """
    Simulated Schumann resonance state.

    Note: This is a SIMULATION for arianna.c's internal use.
    Real Schumann resonance data would come from observatories.
    """
    hz: float = SCHUMANN_BASE_HZ       # Current frequency
    coherence: float = 0.5             # Phase coherence 0-1
    modulation: float = 0.1            # Modulation strength
    phase: float = 0.0                 # Current phase (radians)

    # For simulation
    _last_update: float = field(default_factory=time.time)

    def step(self, dt: float) -> None:
        """Advance simulation by dt seconds"""
        # Phase advances with frequency
        self.phase += 2 * math.pi * self.hz * dt
        self.phase = self.phase % (2 * math.pi)

        # Coherence drifts slowly (simulated)
        # In reality, coherence varies with solar activity, lightning, etc.
        drift = math.sin(time.time() * 0.001) * 0.1
        self.coherence = max(0.0, min(1.0, 0.5 + drift))

        self._last_update = time.time()

    def get_signal(self) -> float:
        """Get current signal value (-1 to 1)"""
        return math.sin(self.phase) * self.coherence

    def modulate(self, direction: float) -> float:
        """
        Apply Schumann modulation to a direction value.

        Used to subtly influence generation based on Earth resonance.
        """
        signal = self.get_signal()
        return direction + signal * self.modulation


# Global state (singleton pattern)
_global_schumann: Optional[SchumannState] = None


def get_schumann_state() -> SchumannState:
    """Get or create global Schumann state"""
    global _global_schumann
    if _global_schumann is None:
        _global_schumann = SchumannState()
    return _global_schumann


# ═══════════════════════════════════════════════════════════════════════════════
# CONVENIENCE FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def get_schumann_coherence() -> float:
    """
    Get current Schumann coherence (0.0 to 1.0).

    High coherence → faster healing, better resonance
    Low coherence → slower healing, more drift

    This is a SIMULATION. For real data, you'd query:
    - HeartMath Global Coherence Monitoring System
    - Space weather services
    """
    state = get_schumann_state()

    # Update based on time since last call
    now = time.time()
    dt = now - state._last_update
    if dt > 0.01:  # Only step if meaningful time passed
        state.step(dt)

    return state.coherence


def schumann_phase() -> float:
    """
    Get current Schumann phase (0.0 to 2π).

    Can be used for timing-based effects:
    - Phase 0 to π: "breathing in"
    - Phase π to 2π: "breathing out"
    """
    state = get_schumann_state()
    return state.phase


def get_schumann_signal() -> dict:
    """
    Get full Schumann signal for git_arianna observation.

    Returns dict with:
    - hz: current frequency
    - coherence: 0.0-1.0
    - phase: 0.0-2π
    - signal: -1.0 to 1.0 (current wave value)
    - brain_wave_match: which brain wave band Schumann is in
    """
    state = get_schumann_state()

    # Update
    now = time.time()
    dt = now - state._last_update
    if dt > 0.01:
        state.step(dt)

    # Find matching brain wave
    brain_wave_match = "none"
    for name, (low, high) in BRAIN_WAVES.items():
        if low <= state.hz <= high:
            brain_wave_match = name
            break

    return {
        "hz": state.hz,
        "coherence": state.coherence,
        "phase": state.phase,
        "signal": state.get_signal(),
        "brain_wave_match": brain_wave_match,
        "harmonics": SCHUMANN_HARMONICS,
    }


def schumann_healing_factor() -> float:
    """
    Get healing factor based on Schumann coherence.

    Returns 0.5 to 1.0:
    - 0.5: minimum healing (low coherence)
    - 1.0: maximum healing (high coherence)

    Used to scale tension/dissonance decay rates.
    """
    coherence = get_schumann_coherence()
    return 0.5 + 0.5 * coherence


# ═══════════════════════════════════════════════════════════════════════════════
# INTEGRATION WITH git_arianna
# ═══════════════════════════════════════════════════════════════════════════════

def modulate_entropy_with_schumann(entropy_delta: float) -> float:
    """
    Modulate entropy observation with Schumann resonance.

    When Schumann coherence is high:
    - Entropy changes feel "smoother"
    - Less jarring transitions

    When coherence is low:
    - Entropy changes feel "rougher"
    - More discontinuity
    """
    coherence = get_schumann_coherence()

    # High coherence smooths entropy, low coherence amplifies it
    smoothing = coherence * 0.3  # 0 to 0.3
    return entropy_delta * (1.0 - smoothing)


def should_trigger_cosmic_event() -> bool:
    """
    Check if a "cosmic event" should trigger based on Schumann state.

    Returns True rarely (~1% of calls when coherence is very high).
    Used for occasional special behaviors in generation.
    """
    state = get_schumann_state()

    # Only trigger at phase peaks with high coherence
    if state.coherence > 0.8 and abs(state.get_signal()) > 0.95:
        import random
        return random.random() < 0.01

    return False


# ═══════════════════════════════════════════════════════════════════════════════
# TEST
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=== Schumann Constants Test ===\n")

    print(f"Base frequency: {SCHUMANN_BASE_HZ} Hz")
    print(f"Harmonics: {SCHUMANN_HARMONICS}")
    print()

    print("Brain wave bands:")
    for name, (low, high) in BRAIN_WAVES.items():
        marker = " <-- Schumann" if low <= SCHUMANN_BASE_HZ <= high else ""
        print(f"  {name}: {low}-{high} Hz{marker}")
    print()

    # Simulate for a few steps
    print("Simulation (10 steps at 0.1s each):")
    state = get_schumann_state()
    for i in range(10):
        state.step(0.1)
        signal = get_schumann_signal()
        print(f"  Step {i+1}: coherence={signal['coherence']:.3f}, "
              f"signal={signal['signal']:.3f}, phase={signal['phase']:.2f}")

    print()
    print(f"Healing factor: {schumann_healing_factor():.3f}")
    print(f"Cosmic event trigger: {should_trigger_cosmic_event()}")
