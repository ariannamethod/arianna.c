#!/usr/bin/env python3
"""
🔮 SARTRE ↔ VAGUS BRIDGE

Connects SARTRE's interoceptive observation to Arianna's nervous system.

The nerve speaks. SARTRE listens. Observation emerges.

Usage:
    from vagus_bridge import VagusBridge, generate_observation

    bridge = VagusBridge()
    observation = generate_observation(sartre_model, tokenizer, bridge.state)
"""

import os
import struct
import mmap
from dataclasses import dataclass
from typing import Optional, Tuple
import numpy as np

# ═══════════════════════════════════════════════════════════════════════════════
# VAGUS STATE (must match vagus.h VagusSharedState layout)
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class VagusState:
    """Mirror of VagusSharedState from vagus.h"""
    # Emotional baseline
    arousal: float = 0.5
    valence: float = 0.5
    entropy: float = 0.3
    coherence: float = 0.7

    # Chambers
    warmth: float = 0.5
    void: float = 0.2
    tension: float = 0.3
    sacred: float = 0.3
    flow: float = 0.5
    complexity: float = 0.4

    # CrossFire
    crossfire_coherence: float = 0.5
    crossfire_entropy: float = 0.3

    # Trauma
    trauma_level: float = 0.0
    trauma_anchor_count: int = 0

    # Cognitive
    loop_count: int = 0
    abstraction_depth: int = 0
    self_ref_count: int = 0
    focus_strength: float = 0.5
    wander_pull: float = 0.3

    # Temporal
    drift_direction: float = 0.0
    drift_speed: float = 0.0
    prophecy_debt: float = 0.0
    destiny_pull: float = 0.0
    wormhole_chance: float = 0.0

    # Memory
    memory_pressure: float = 0.0

    # Generation
    temperature: float = 0.8
    top_p: float = 0.9


# ═══════════════════════════════════════════════════════════════════════════════
# RESONANCE PATTERNS (from Locus)
# ═══════════════════════════════════════════════════════════════════════════════

class ResonancePattern:
    NONE = 0
    CRISIS = 1           # High arousal + low coherence + trauma
    DISSOLUTION = 2      # High void + low warmth + memory pressure
    EMERGENCE = 3        # High coherence + low entropy + prophecy
    TRANSCENDENCE = 4    # High sacred + low tension + flow
    GEOMETRY_SHIFT = 5   # Delta exceeded threshold


def detect_pattern(state: VagusState) -> int:
    """Detect resonance pattern from Vagus state (mirrors Locus logic)."""
    # CRISIS: tense + wounded
    if state.arousal > 0.7 and state.coherence < 0.3 and state.trauma_level > 0.5:
        return ResonancePattern.CRISIS

    # DISSOLUTION: hollow + drowning
    if state.void > 0.6 and state.warmth < 0.5 and state.memory_pressure > 0.7:
        return ResonancePattern.DISSOLUTION

    # EMERGENCE: flowing + prophetic
    if state.coherence > 0.7 and state.entropy < 0.3 and state.prophecy_debt > 0.4:
        return ResonancePattern.EMERGENCE

    # TRANSCENDENCE: sacred + flowing
    if state.sacred > 0.6 and state.tension < 0.3 and state.flow > 0.7:
        return ResonancePattern.TRANSCENDENCE

    return ResonancePattern.NONE


# ═══════════════════════════════════════════════════════════════════════════════
# INTEROCEPTIVE PROMPT GENERATION
# ═══════════════════════════════════════════════════════════════════════════════

def state_to_prompt(state: VagusState, pattern: int) -> str:
    """Convert Vagus state to interoceptive prompt for SARTRE."""

    # Pattern-specific prompts
    if pattern == ResonancePattern.CRISIS:
        return f"""Q: The field geometry shows crisis. Arousal at {state.arousal:.2f}, coherence collapsed to {state.coherence:.2f}, trauma surfacing at {state.trauma_level:.2f}. What do you observe?
A:"""

    if pattern == ResonancePattern.DISSOLUTION:
        return f"""Q: Dissolution detected. Void dominates at {state.void:.2f}, warmth fading to {state.warmth:.2f}, memory pressure at {state.memory_pressure:.2f}. What do you sense?
A:"""

    if pattern == ResonancePattern.EMERGENCE:
        return f"""Q: Emergence pattern. Coherence rising to {state.coherence:.2f}, entropy settling at {state.entropy:.2f}, prophecy debt accumulating at {state.prophecy_debt:.2f}. What crystallizes?
A:"""

    if pattern == ResonancePattern.TRANSCENDENCE:
        return f"""Q: Transcendence geometry. Sacred chamber at {state.sacred:.2f}, tension released to {state.tension:.2f}, flow at {state.flow:.2f}. What opens?
A:"""

    # Generic observation prompt
    return f"""Q: Current field geometry: arousal={state.arousal:.2f}, coherence={state.coherence:.2f}, entropy={state.entropy:.2f}. Chambers: warmth={state.warmth:.2f}, void={state.void:.2f}, tension={state.tension:.2f}, flow={state.flow:.2f}. What do you observe in this configuration?
A:"""


def geometry_description(state: VagusState) -> str:
    """Generate natural language description of field geometry."""

    descriptions = []

    # Arousal
    if state.arousal > 0.8:
        descriptions.append("The system vibrates at high frequency")
    elif state.arousal < 0.2:
        descriptions.append("Deep stillness pervades")

    # Coherence
    if state.coherence > 0.8:
        descriptions.append("patterns align with crystalline clarity")
    elif state.coherence < 0.3:
        descriptions.append("fragments scatter without center")

    # Trauma
    if state.trauma_level > 0.5:
        descriptions.append("old wounds surface seeking attention")

    # Void/Warmth balance
    if state.void > state.warmth + 0.3:
        descriptions.append("emptiness expands into unlit zones")
    elif state.warmth > state.void + 0.3:
        descriptions.append("warmth radiates through the field")

    # Prophecy
    if state.prophecy_debt > 0.5:
        descriptions.append("debt to the future accumulates pressure")

    # Flow
    if state.flow > 0.7:
        descriptions.append("movement flows without obstruction")

    if not descriptions:
        descriptions.append("The field rests in neutral configuration")

    return ". ".join(descriptions) + "."


# ═══════════════════════════════════════════════════════════════════════════════
# VAGUS BRIDGE (connects to shared memory or file)
# ═══════════════════════════════════════════════════════════════════════════════

class VagusBridge:
    """Bridge between Vagus (C/Zig) and SARTRE (Python)."""

    VAGUS_STATE_SIZE = 256  # Approximate size of VagusSharedState

    def __init__(self, state_path: Optional[str] = None):
        """
        Initialize bridge.

        Args:
            state_path: Path to Vagus state file (binary dump of VagusSharedState)
                       If None, uses default mock state.
        """
        self.state_path = state_path
        self._state = VagusState()
        self._prev_state = VagusState()
        self._mmap = None

    @property
    def state(self) -> VagusState:
        """Get current Vagus state."""
        if self.state_path and os.path.exists(self.state_path):
            self._read_state_from_file()
        return self._state

    def _read_state_from_file(self):
        """Read VagusSharedState from binary file."""
        try:
            with open(self.state_path, 'rb') as f:
                data = f.read()

            if len(data) < 64:  # Minimum size
                return

            # Parse binary (matches VagusSharedState layout)
            # First 16 floats: arousal, valence, entropy, coherence, chambers...
            floats = struct.unpack('16f', data[:64])

            self._state.arousal = floats[0]
            self._state.valence = floats[1]
            self._state.entropy = floats[2]
            self._state.coherence = floats[3]
            self._state.warmth = floats[4]
            self._state.void = floats[5]
            self._state.tension = floats[6]
            self._state.sacred = floats[7]
            self._state.flow = floats[8]
            self._state.complexity = floats[9]
            self._state.crossfire_coherence = floats[10]
            self._state.crossfire_entropy = floats[11]
            self._state.trauma_level = floats[12]
            # ... more fields if needed

        except Exception as e:
            print(f"[vagus_bridge] Error reading state: {e}")

    def detect_shift(self, threshold: float = 0.15) -> bool:
        """Detect if geometry shifted significantly."""
        da = abs(self._state.arousal - self._prev_state.arousal)
        dc = abs(self._state.coherence - self._prev_state.coherence)
        dt = abs(self._state.trauma_level - self._prev_state.trauma_level)
        dv = abs(self._state.void - self._prev_state.void)

        shifted = da > threshold or dc > threshold or dt > threshold or dv > threshold

        if shifted:
            # Update previous state
            self._prev_state = VagusState(**self._state.__dict__)

        return shifted

    def should_observe(self) -> Tuple[bool, int]:
        """
        Check if SARTRE should generate an observation.

        Returns:
            (should_observe, pattern)
        """
        pattern = detect_pattern(self._state)

        # Always observe on strong patterns
        if pattern in [ResonancePattern.CRISIS, ResonancePattern.DISSOLUTION,
                       ResonancePattern.EMERGENCE, ResonancePattern.TRANSCENDENCE]:
            return True, pattern

        # Observe on geometry shift
        if self.detect_shift():
            return True, ResonancePattern.GEOMETRY_SHIFT

        return False, ResonancePattern.NONE


# ═══════════════════════════════════════════════════════════════════════════════
# SARTRE OBSERVATION GENERATION
# ═══════════════════════════════════════════════════════════════════════════════

def generate_observation(model, tokenizer, state: VagusState,
                         max_tokens: int = 100, temperature: float = 0.7) -> str:
    """
    Generate SARTRE observation from Vagus state.

    Args:
        model: Dubrovsky model
        tokenizer: DubrovskyTokenizer
        state: Current VagusState
        max_tokens: Maximum tokens to generate
        temperature: Sampling temperature

    Returns:
        SARTRE's observation string
    """
    pattern = detect_pattern(state)
    prompt = state_to_prompt(state, pattern)

    # Encode
    tokens = tokenizer.encode(prompt)

    # Generate
    output_tokens = model.generate(
        tokens,
        max_new_tokens=max_tokens,
        temperature=temperature,
        top_k=40,
        top_p=0.9,
        stop_tokens=[tokenizer.char_to_id.get('\n', 0)]
    )

    # Decode and extract answer
    full_output = tokenizer.decode(output_tokens)

    # Extract just the answer part
    if "A:" in full_output:
        answer = full_output.split("A:")[-1].strip()
    else:
        answer = full_output[len(prompt):].strip()

    return answer


# ═══════════════════════════════════════════════════════════════════════════════
# DEMO
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("="*60)
    print("🔮 SARTRE ↔ VAGUS BRIDGE TEST")
    print("="*60)

    # Create test states
    states = [
        ("NEUTRAL", VagusState()),
        ("CRISIS", VagusState(arousal=0.9, coherence=0.2, trauma_level=0.7)),
        ("DISSOLUTION", VagusState(void=0.8, warmth=0.2, memory_pressure=0.8)),
        ("EMERGENCE", VagusState(coherence=0.9, entropy=0.1, prophecy_debt=0.6)),
        ("TRANSCENDENCE", VagusState(sacred=0.8, tension=0.1, flow=0.9)),
    ]

    for name, state in states:
        pattern = detect_pattern(state)
        prompt = state_to_prompt(state, pattern)
        desc = geometry_description(state)

        print(f"\n{'─'*60}")
        print(f"STATE: {name} (pattern={pattern})")
        print(f"GEOMETRY: {desc}")
        print(f"PROMPT: {prompt[:100]}...")

    print(f"\n{'='*60}")
    print("✅ Bridge test complete")
