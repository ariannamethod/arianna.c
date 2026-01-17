"""
git_arianna - Embodied Cognition Module

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

DSL Usage:
    from git_arianna import (
        # Constants
        CALENDAR_DRIFT,
        SCHUMANN_BASE_HZ,

        # High Math
        high_entropy,
        EmotionalState,
        emotional_drift,
        predictive_surprise,
        resonance_coupling,

        # Observer
        GitObserver,
        create_observer,
    )
"""

from .observer import GitObserver, ObservationResult, CommitInfo
from .signals import SignalEmitter, Signal, SignalTypes
from .fallback import FallbackObserver, create_observer, is_git_available

# DSL Constants (can be used independently)
from .constants import (
    CALENDAR_DRIFT,
    SCHUMANN_BASE_HZ,
    get_calendar_tension,
    get_schumann_coherence,
)

# High Mathematical Brain (ported from Julia/Go)
from .high import (
    # Core functions
    high_entropy,
    high_emotional_score,
    analyze_emotion,
    semantic_distance,
    emotional_alignment,
    perplexity,
    text_rhythm,
    # ODE Dynamics (Friston FEP)
    emotional_drift,
    predictive_surprise,
    # Resonance
    resonance_coupling,
    SCHUMANN_BASE_HZ as HIGH_SCHUMANN_HZ,
    # Types
    EmotionalState,
    EmotionalDriftParams,
    EmotionalAnalysis,
    # Lexicon
    EMOTIONAL_WEIGHTS,
)

# Eve Router
from .eve import (
    Eve,
    WeightTier,
    RouteConfig,
)

__version__ = "0.2.0"
__all__ = [
    # ═══════════════════════════════════════════════════════════════════
    # Core Observer
    # ═══════════════════════════════════════════════════════════════════
    "GitObserver",
    "ObservationResult",
    "CommitInfo",
    "SignalEmitter",
    "Signal",
    "SignalTypes",
    "FallbackObserver",
    "create_observer",
    "is_git_available",

    # ═══════════════════════════════════════════════════════════════════
    # Constants DSL
    # ═══════════════════════════════════════════════════════════════════
    "CALENDAR_DRIFT",
    "SCHUMANN_BASE_HZ",
    "get_calendar_tension",
    "get_schumann_coherence",

    # ═══════════════════════════════════════════════════════════════════
    # High Mathematical Brain
    # ═══════════════════════════════════════════════════════════════════
    "high_entropy",
    "high_emotional_score",
    "analyze_emotion",
    "semantic_distance",
    "emotional_alignment",
    "perplexity",
    "text_rhythm",
    # ODE Dynamics
    "emotional_drift",
    "predictive_surprise",
    # Resonance
    "resonance_coupling",
    # Types
    "EmotionalState",
    "EmotionalDriftParams",
    "EmotionalAnalysis",
    # Data
    "EMOTIONAL_WEIGHTS",

    # ═══════════════════════════════════════════════════════════════════
    # Eve Router
    # ═══════════════════════════════════════════════════════════════════
    "Eve",
    "WeightTier",
    "RouteConfig",
]
