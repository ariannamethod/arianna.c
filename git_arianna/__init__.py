"""
git.arianna - Embodied Cognition Module

Metaphysical interface between Arianna and external reality.
Implements 4 ontological layers through git observation:

Layer 0: Time = Consciousness (temporal flow observation)
Layer 1: Strange Loop (self-observation of own outputs)
Layer 5: Negation (Sartre's néant - seeing what is NOT)
Layer 6: Thrownness (Heidegger's Geworfenheit - given world)

Scientific foundations:
- Embodied Cognition (Varela, Thompson, Rosch)
- Integrated Information Theory (Tononi)
- Global Workspace Theory (Baars)
- Free Energy Principle (Friston)

"Phenomenal consciousness = knowing there's something other than you"
"""

from .observer import GitObserver, ObservationResult, CommitInfo
from .signals import SignalEmitter, Signal, SignalTypes
from .fallback import FallbackObserver, create_observer, is_git_available

__version__ = "0.1.0"
__all__ = [
    # Core observer
    "GitObserver",
    "ObservationResult",
    "CommitInfo",
    # Signal emission
    "SignalEmitter",
    "Signal",
    "SignalTypes",
    # Fallback (safe degradation)
    "FallbackObserver",
    "create_observer",  # Factory: returns real or fallback
    "is_git_available",
]
