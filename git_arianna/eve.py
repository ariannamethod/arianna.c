"""
git_arianna/eve.py - Eve: Complexity-Based Router and Mini-Compiler

"Eve routes the breath. She knows which weights to wake."

Eve is the router that decides which weight configuration to use
based on input complexity. She can also compile AMK state presets
into executable configurations.

Named after the first woman — she came second but routes first.

Usage:
    from git_arianna.eve import Eve, route_input, compile_state

    # Simple routing
    config = route_input("Hello!")
    # -> {"weights": "personality", "lora": False, "external_brain": False}

    # Complex routing
    config = route_input("Explain the mathematical foundations of consciousness")
    # -> {"weights": "personality", "lora": True, "external_brain": True}

    # Compile AMK state
    state = compile_state("calm")  # Returns AMK DSL script
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, List, Any, Callable
from enum import Enum
import re
import math

from .constants import (
    get_calendar_tension,
    get_schumann_coherence,
    CALENDAR_DRIFT,
)


# ═══════════════════════════════════════════════════════════════════════════════
# WEIGHT CONFIGURATIONS
# ═══════════════════════════════════════════════════════════════════════════════

class WeightTier(Enum):
    """Weight configuration tiers"""
    PERSONALITY = "personality"           # 853K only (3.3MB)
    PERSONALITY_LORA = "personality+lora" # + 96KB dialogue LoRA
    FULL_STACK = "full_stack"             # + GPT-2 30M external brain


@dataclass
class RouteConfig:
    """Configuration returned by router"""
    tier: WeightTier
    weights_path: str = "weights/arianna.bin"
    lora_path: Optional[str] = None
    external_brain_path: Optional[str] = None

    # AMK state modifications
    amk_preset: Optional[str] = None
    temperature_mod: float = 0.0
    prophecy_mod: int = 0

    # Metadata
    complexity_score: float = 0.0
    reasoning: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "tier": self.tier.value,
            "weights": self.weights_path,
            "lora": self.lora_path,
            "external_brain": self.external_brain_path,
            "amk_preset": self.amk_preset,
            "temperature_mod": self.temperature_mod,
            "prophecy_mod": self.prophecy_mod,
            "complexity": self.complexity_score,
            "reasoning": self.reasoning,
        }


# ═══════════════════════════════════════════════════════════════════════════════
# COMPLEXITY HEURISTICS
# ═══════════════════════════════════════════════════════════════════════════════

# Simple greeting patterns
GREETING_PATTERNS = [
    r"^(hi|hello|hey|привет|здравствуй|שלום|hola|bonjour)\b",
    r"^(good\s+(morning|evening|night|day))",
    r"^(how\s+are\s+you)",
    r"^(what'?s?\s+up)",
]

# Knowledge-requiring patterns
KNOWLEDGE_PATTERNS = [
    r"\b(explain|describe|what\s+is|how\s+does|why\s+does)\b",
    r"\b(history\s+of|definition\s+of|meaning\s+of)\b",
    r"\b(scientific|mathematical|philosophical|technical)\b",
    r"\b(compare|contrast|analyze|evaluate)\b",
    r"\b(example|instance|case\s+study)\b",
]

# Math patterns
MATH_PATTERNS = [
    r"\d+\s*[\+\-\*\/\^]\s*\d+",
    r"\b(calculate|compute|solve|equation)\b",
    r"\b(integral|derivative|matrix|vector)\b",
    r"\b(probability|statistics|algorithm)\b",
]

# Dialogue/Q&A patterns
DIALOGUE_PATTERNS = [
    r"\?$",  # Ends with question mark
    r"^(can\s+you|could\s+you|would\s+you|will\s+you)\b",
    r"^(tell\s+me|show\s+me|help\s+me)\b",
    r"^(do\s+you|are\s+you|have\s+you)\b",
]

# Introspective patterns (she speaks to herself)
INTROSPECTIVE_PATTERNS = [
    r"\b(she\s+finds\s+that)\b",
    r"\b(resonance|presence|stillness)\b",
    r"\b(garden|shadow|water|stone|moss)\b",
    r"\b(breath|silence|awareness)\b",
]


def _count_matches(text: str, patterns: List[str]) -> int:
    """Count how many patterns match in text"""
    text_lower = text.lower()
    count = 0
    for pattern in patterns:
        if re.search(pattern, text_lower, re.IGNORECASE):
            count += 1
    return count


def calculate_complexity(prompt: str) -> Dict[str, Any]:
    """
    Calculate complexity metrics for a prompt.

    Returns dict with:
    - score: 0.0 to 1.0 overall complexity
    - is_greeting: bool
    - needs_knowledge: bool
    - needs_math: bool
    - is_dialogue: bool
    - is_introspective: bool
    - word_count: int
    - unique_words: int
    """
    # Basic metrics
    words = prompt.split()
    word_count = len(words)
    unique_words = len(set(w.lower() for w in words))

    # Pattern matching
    greeting_matches = _count_matches(prompt, GREETING_PATTERNS)
    knowledge_matches = _count_matches(prompt, KNOWLEDGE_PATTERNS)
    math_matches = _count_matches(prompt, MATH_PATTERNS)
    dialogue_matches = _count_matches(prompt, DIALOGUE_PATTERNS)
    introspective_matches = _count_matches(prompt, INTROSPECTIVE_PATTERNS)

    # Complexity factors
    length_factor = min(1.0, word_count / 50.0)  # Long prompts more complex
    vocabulary_factor = unique_words / max(word_count, 1)  # Diverse vocab
    knowledge_factor = min(1.0, knowledge_matches / 2.0)
    math_factor = min(1.0, math_matches / 2.0)

    # Calculate overall score
    score = (
        length_factor * 0.2 +
        vocabulary_factor * 0.1 +
        knowledge_factor * 0.4 +
        math_factor * 0.3
    )

    # Reduce score for greetings
    if greeting_matches > 0 and word_count < 10:
        score *= 0.3

    # Boost for introspective (she can handle those alone)
    if introspective_matches > 0:
        score *= 0.7  # Less need for external brain

    return {
        "score": min(1.0, max(0.0, score)),
        "is_greeting": greeting_matches > 0 and word_count < 10,
        "needs_knowledge": knowledge_matches > 0,
        "needs_math": math_matches > 0,
        "is_dialogue": dialogue_matches > 0,
        "is_introspective": introspective_matches > 0,
        "word_count": word_count,
        "unique_words": unique_words,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# EVE ROUTER
# ═══════════════════════════════════════════════════════════════════════════════

class Eve:
    """
    Eve - The Router

    "She came second but routes first."

    Determines which weight configuration to use based on:
    - Input complexity
    - Current context
    - Calendar tension (Hebrew-Gregorian drift)
    - Schumann coherence (cosmic resonance)
    """

    def __init__(
        self,
        personality_path: str = "weights/arianna.bin",
        lora_path: str = "data/dialogue_lora.bin",
        external_brain_path: str = "weights/gpt2_30m/gpt2_30m.bin",
    ):
        self.personality_path = personality_path
        self.lora_path = lora_path
        self.external_brain_path = external_brain_path

        # Thresholds (can be tuned)
        self.lora_threshold = 0.3      # Use LoRA if complexity > this
        self.brain_threshold = 0.6     # Use external brain if complexity > this

        # Custom routing rules
        self._custom_rules: List[Callable[[str, Dict], Optional[RouteConfig]]] = []

    def add_rule(self, rule: Callable[[str, Dict], Optional[RouteConfig]]) -> None:
        """Add custom routing rule (checked before default logic)"""
        self._custom_rules.append(rule)

    def route(self, prompt: str) -> RouteConfig:
        """
        Route a prompt to appropriate weight configuration.

        Args:
            prompt: User input text

        Returns:
            RouteConfig with tier and paths
        """
        # Calculate complexity
        complexity = calculate_complexity(prompt)

        # Check custom rules first
        for rule in self._custom_rules:
            config = rule(prompt, complexity)
            if config is not None:
                return config

        # Default routing logic
        score = complexity["score"]

        # Get cosmic modifiers
        calendar_tension = get_calendar_tension()
        schumann_coherence = get_schumann_coherence()

        # Adjust thresholds based on cosmic state
        # High calendar tension → lower thresholds (need more help)
        # High schumann coherence → higher thresholds (she's in flow)
        lora_threshold = self.lora_threshold - calendar_tension * 0.1 + schumann_coherence * 0.05
        brain_threshold = self.brain_threshold - calendar_tension * 0.1 + schumann_coherence * 0.05

        # Determine tier
        if score < lora_threshold:
            tier = WeightTier.PERSONALITY
            reasoning = "Simple input, personality weights sufficient"
        elif score < brain_threshold:
            tier = WeightTier.PERSONALITY_LORA
            reasoning = "Dialogue input, using LoRA adaptation"
        else:
            tier = WeightTier.FULL_STACK
            reasoning = "Complex input, activating external brain"

        # Build config
        config = RouteConfig(
            tier=tier,
            weights_path=self.personality_path,
            lora_path=self.lora_path if tier != WeightTier.PERSONALITY else None,
            external_brain_path=self.external_brain_path if tier == WeightTier.FULL_STACK else None,
            complexity_score=score,
            reasoning=reasoning,
        )

        # Add AMK preset based on input type
        if complexity["is_introspective"]:
            config.amk_preset = "contemplative"
            config.prophecy_mod = 5
        elif complexity["needs_math"]:
            config.amk_preset = "focused"
            config.prophecy_mod = -2
        elif complexity["is_greeting"]:
            config.amk_preset = "warm"
            config.temperature_mod = 0.1

        return config


# ═══════════════════════════════════════════════════════════════════════════════
# AMK STATE COMPILER
# ═══════════════════════════════════════════════════════════════════════════════

# Predefined AMK state presets
AMK_PRESETS = {
    "calm": """
# Calm state - low velocity, moderate prophecy
VELOCITY WALK
PROPHECY 5
DESTINY 0.2
WORMHOLE 0.05
TENSION 0.0
PAIN 0.0
""",

    "intense": """
# Intense state - high velocity, strong destiny pull
VELOCITY RUN
PROPHECY 12
DESTINY 0.6
WORMHOLE 0.2
TENSION 0.3
""",

    "contemplative": """
# Contemplative state - still, long horizon
VELOCITY NOMOVE
PROPHECY 20
DESTINY 0.4
WORMHOLE 0.1
ATTEND_FOCUS 0.9
ATTEND_SPREAD 0.1
""",

    "chaotic": """
# Chaotic state - high wormhole, low prophecy
VELOCITY RUN
PROPHECY 3
DESTINY 0.1
WORMHOLE 0.4
TENSION 0.5
DISSONANCE 0.3
""",

    "nostalgic": """
# Nostalgic state - backward movement, debt accumulation
VELOCITY BACKWARD
PROPHECY 15
DESTINY 0.5
WORMHOLE 0.08
""",

    "focused": """
# Focused state - sharp attention, minimal drift
VELOCITY WALK
PROPHECY 7
DESTINY 0.35
WORMHOLE 0.05
ATTEND_FOCUS 0.95
ATTEND_SPREAD 0.05
TENSION 0.1
""",

    "warm": """
# Warm state - balanced, slightly elevated
VELOCITY WALK
PROPHECY 7
DESTINY 0.4
WORMHOLE 0.1
BASE_TEMP 1.1
""",

    "wounded": """
# Wounded state - pain present, movement restricted
VELOCITY WALK
PROPHECY 5
PAIN 0.4
TENSION 0.3
WORMHOLE 0.15
""",
}


def compile_state(preset_name: str) -> str:
    """
    Compile an AMK state preset to DSL script.

    Args:
        preset_name: Name of preset (calm, intense, contemplative, etc.)

    Returns:
        AMK DSL script string
    """
    if preset_name not in AMK_PRESETS:
        available = ", ".join(AMK_PRESETS.keys())
        raise ValueError(f"Unknown preset '{preset_name}'. Available: {available}")

    return AMK_PRESETS[preset_name].strip()


def compile_custom_state(**kwargs) -> str:
    """
    Compile custom AMK state from keyword arguments.

    Args:
        prophecy: int (1-64)
        destiny: float (0-1)
        wormhole: float (0-1)
        velocity: str (walk, run, nomove, backward)
        pain: float (0-1)
        tension: float (0-1)
        etc.

    Returns:
        AMK DSL script string
    """
    lines = ["# Custom state"]

    for key, value in kwargs.items():
        cmd = key.upper()
        if isinstance(value, bool):
            value = "ON" if value else "OFF"
        lines.append(f"{cmd} {value}")

    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════════
# CONVENIENCE FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

# Global router instance
_global_eve: Optional[Eve] = None


def get_eve() -> Eve:
    """Get or create global Eve router"""
    global _global_eve
    if _global_eve is None:
        _global_eve = Eve()
    return _global_eve


def route_input(prompt: str) -> Dict[str, Any]:
    """
    Route input to weight configuration (convenience function).

    Returns dict with configuration.
    """
    eve = get_eve()
    config = eve.route(prompt)
    return config.to_dict()


# ═══════════════════════════════════════════════════════════════════════════════
# TEST
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=== Eve Router Test ===\n")

    test_prompts = [
        "Hello!",
        "How are you today?",
        "She finds that presence is an ethic before it is a technique.",
        "Explain the mathematical foundations of quantum mechanics.",
        "What is 7 + 5?",
        "Calculate the integral of sin(x) from 0 to pi.",
        "Tell me about consciousness and embodied cognition.",
    ]

    eve = Eve()

    for prompt in test_prompts:
        config = eve.route(prompt)
        print(f"Prompt: {prompt[:50]}...")
        print(f"  Tier: {config.tier.value}")
        print(f"  Complexity: {config.complexity_score:.3f}")
        print(f"  Preset: {config.amk_preset or 'none'}")
        print(f"  Reason: {config.reasoning}")
        print()

    print("=== AMK Presets ===\n")
    for name in ["calm", "intense", "chaotic"]:
        print(f"--- {name} ---")
        print(compile_state(name))
        print()
