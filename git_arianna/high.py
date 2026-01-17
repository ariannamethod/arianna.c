"""
git_arianna.high - High Mathematical Brain DSL Bindings

Python interface to inner_world/high.go

Usage:
    from git_arianna import (
        high_entropy,
        high_emotional_score,
        EmotionalState,
        emotional_drift,
        predictive_surprise,
        resonance_coupling,
    )

"Mathematics is the high-level language of thought."
"""

from dataclasses import dataclass
from typing import Tuple, Optional
import math
import re

# ═══════════════════════════════════════════════════════════════════════════════
# EMOTIONAL WEIGHTS (mirrored from high.go)
# ═══════════════════════════════════════════════════════════════════════════════

EMOTIONAL_WEIGHTS = {
    # English - Positive
    "great": 0.8, "love": 0.9, "amazing": 0.7, "wonderful": 0.8, "excellent": 0.7,
    "beautiful": 0.8, "fantastic": 0.7, "awesome": 0.8, "perfect": 0.7, "brilliant": 0.8,
    "happy": 0.7, "joy": 0.8, "excited": 0.7, "delighted": 0.8, "pleased": 0.6,
    "good": 0.5, "nice": 0.4, "fine": 0.3, "okay": 0.1, "thanks": 0.4,
    "grateful": 0.7, "blessed": 0.6, "peaceful": 0.5, "calm": 0.4, "serene": 0.5,
    "hope": 0.6, "dream": 0.5, "inspire": 0.6, "create": 0.5, "grow": 0.4,

    # English - Negative
    "terrible": -0.8, "hate": -0.9, "awful": -0.7, "horrible": -0.8, "disgusting": -0.9,
    "sad": -0.6, "angry": -0.7, "frustrated": -0.6, "disappointed": -0.6, "upset": -0.6,
    "bad": -0.5, "wrong": -0.4, "fail": -0.6, "lose": -0.5, "hurt": -0.7,
    "pain": -0.8, "suffer": -0.8, "fear": -0.7, "anxiety": -0.6, "stress": -0.5,
    "alone": -0.6, "lonely": -0.7, "empty": -0.5, "nothing": -0.6, "worthless": -0.9,

    # Russian - Positive
    "отлично": 0.8, "классно": 0.7, "супер": 0.8, "круто": 0.7, "прекрасно": 0.8,
    "здорово": 0.7, "замечательно": 0.8, "чудесно": 0.7, "великолепно": 0.8,
    "люблю": 0.9, "радость": 0.8, "счастье": 0.9, "мир": 0.5, "добро": 0.6,

    # Russian - Negative
    "ужасно": -0.8, "плохо": -0.6, "грустно": -0.6, "злой": -0.7, "расстроен": -0.6,
    "больно": -0.8, "страшно": -0.7, "одиноко": -0.7, "пусто": -0.5,
    "ненавижу": -0.9, "страдаю": -0.8, "боюсь": -0.7, "тревога": -0.6,

    # Hebrew - Positive
    "טוב": 0.5, "יפה": 0.7, "מדהים": 0.8, "נהדר": 0.7, "אהבה": 0.9,
    "שמחה": 0.8, "תקווה": 0.6, "שלום": 0.5, "ברכה": 0.6,

    # Hebrew - Negative
    "רע": -0.5, "נורא": -0.8, "עצוב": -0.6, "כועס": -0.7, "פחד": -0.7,
    "כאב": -0.8, "בודד": -0.7, "ריק": -0.5, "שנאה": -0.9,
}


# ═══════════════════════════════════════════════════════════════════════════════
# CORE TYPES
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class EmotionalState:
    """Current emotional configuration"""
    valence: float = 0.0     # [-1, 1] negative to positive
    arousal: float = 0.3     # [0, 1] calm to excited
    entropy: float = 0.0     # [0, inf] uncertainty
    prediction: float = 0.0  # Expected next valence

    def __post_init__(self):
        self.valence = max(-1.0, min(1.0, self.valence))
        self.arousal = max(0.0, min(1.0, self.arousal))


@dataclass
class EmotionalDriftParams:
    """ODE dynamics parameters"""
    decay_rate: float = 0.1        # Return to baseline speed
    surprise_gain: float = 0.3     # Surprise sensitivity
    entropy_weight: float = 0.2    # Entropy → arousal
    baseline_valence: float = 0.1  # Resting valence
    baseline_arousal: float = 0.3  # Resting arousal


@dataclass
class EmotionalAnalysis:
    """Detailed emotional analysis result"""
    overall_score: float
    positive_words: list
    negative_words: list
    dominant_tone: str  # "positive", "negative", "neutral"
    intensity: float
    valence: float
    arousal: float


# ═══════════════════════════════════════════════════════════════════════════════
# TOKENIZER
# ═══════════════════════════════════════════════════════════════════════════════

def tokenize(text: str) -> list:
    """Split text into lowercase words"""
    return re.findall(r'\w+', text.lower())


# ═══════════════════════════════════════════════════════════════════════════════
# CORE FUNCTIONS (pure Python, mirrors high.go)
# ═══════════════════════════════════════════════════════════════════════════════

def high_entropy(text: str) -> Tuple[float, float]:
    """
    Compute Shannon entropy with emotional modulation.
    Returns: (entropy, emotional_score)
    """
    words = tokenize(text)
    if not words:
        return 0.0, 0.0

    # Word frequency
    word_counts = {}
    emotional_sum = 0.0

    for word in words:
        word_counts[word] = word_counts.get(word, 0) + 1
        if word in EMOTIONAL_WEIGHTS:
            emotional_sum += EMOTIONAL_WEIGHTS[word]

    total = len(words)

    # Shannon entropy
    entropy = 0.0
    for count in word_counts.values():
        p = count / total
        if p > 0:
            entropy -= p * math.log2(p)

    # Emotional score
    emotional_score = emotional_sum / total

    # Modulate entropy by emotional intensity
    emotional_mod = 1.0 + abs(emotional_score) * 0.2
    entropy *= emotional_mod

    return entropy, emotional_score


def high_emotional_score(text: str) -> float:
    """Get just the emotional score from text"""
    _, score = high_entropy(text)
    return score


def analyze_emotion(text: str) -> EmotionalAnalysis:
    """Detailed emotional analysis"""
    words = tokenize(text)

    positive_words = []
    negative_words = []
    total_score = 0.0
    word_count = 0

    for word in words:
        if word in EMOTIONAL_WEIGHTS:
            weight = EMOTIONAL_WEIGHTS[word]
            total_score += weight
            word_count += 1

            if weight > 0:
                positive_words.append(word)
            elif weight < 0:
                negative_words.append(word)

    overall = total_score / word_count if word_count > 0 else 0.0

    # Dominant tone
    if overall > 0.3:
        tone = "positive"
    elif overall < -0.3:
        tone = "negative"
    else:
        tone = "neutral"

    intensity = abs(overall)
    arousal = min(1.0, intensity + word_count * 0.01)

    return EmotionalAnalysis(
        overall_score=overall,
        positive_words=positive_words,
        negative_words=negative_words,
        dominant_tone=tone,
        intensity=intensity,
        valence=overall,
        arousal=arousal,
    )


def semantic_distance(text1: str, text2: str) -> float:
    """Bag-of-words cosine distance between texts"""
    words1 = tokenize(text1)
    words2 = tokenize(text2)

    if not words1 or not words2:
        return 1.0

    # Build vocabulary
    vocab = {}
    idx = 0
    for w in words1 + words2:
        if w not in vocab:
            vocab[w] = idx
            idx += 1

    # Build vectors
    vec1 = [0.0] * len(vocab)
    vec2 = [0.0] * len(vocab)

    for w in words1:
        vec1[vocab[w]] += 1
    for w in words2:
        vec2[vocab[w]] += 1

    # Cosine similarity
    dot = sum(a * b for a, b in zip(vec1, vec2))
    norm1 = math.sqrt(sum(a * a for a in vec1))
    norm2 = math.sqrt(sum(b * b for b in vec2))

    if norm1 == 0 or norm2 == 0:
        return 1.0

    similarity = dot / (norm1 * norm2)
    return 1.0 - similarity


def emotional_alignment(text1: str, text2: str) -> float:
    """How emotionally aligned two texts are"""
    _, emotion1 = high_entropy(text1)
    _, emotion2 = high_entropy(text2)

    # Same sign = aligned
    if (emotion1 >= 0) == (emotion2 >= 0):
        diff = abs(emotion1 - emotion2)
        max_mag = max(abs(emotion1), abs(emotion2))
        if max_mag == 0:
            return 1.0
        return 1.0 - diff / max_mag

    # Opposite signs = negative alignment
    return -abs(emotion1 - emotion2) / 2


# ═══════════════════════════════════════════════════════════════════════════════
# ODE EMOTIONAL DYNAMICS (Friston FEP)
# ═══════════════════════════════════════════════════════════════════════════════

def emotional_drift(
    current: EmotionalState,
    input_text: str,
    dt: float = 0.1,
    params: Optional[EmotionalDriftParams] = None,
) -> EmotionalState:
    """
    Compute next emotional state using ODE.

    dV/dt = -τ(V - V₀) + surprise * gain
    dA/dt = -τ(A - A₀) + |surprise| * gain + entropy * weight
    """
    if params is None:
        params = EmotionalDriftParams()

    # Analyze input
    analysis = analyze_emotion(input_text)
    input_entropy, _ = high_entropy(input_text)

    # Compute surprise (prediction error)
    surprise = analysis.valence - current.prediction

    # ODE integration (Euler)
    d_valence = (
        -params.decay_rate * (current.valence - params.baseline_valence) +
        surprise * params.surprise_gain
    )

    d_arousal = (
        -params.decay_rate * (current.arousal - params.baseline_arousal) +
        abs(surprise) * params.surprise_gain +
        input_entropy * params.entropy_weight
    )

    return EmotionalState(
        valence=current.valence + d_valence * dt,
        arousal=current.arousal + d_arousal * dt,
        entropy=input_entropy,
        prediction=current.valence + d_valence * dt * 0.5,
    )


def predictive_surprise(expected: str, actual: str) -> float:
    """
    Free Energy (prediction error).
    Lower = better prediction, Higher = more surprise.
    """
    semantic_error = semantic_distance(expected, actual)

    emotional_error = 1.0 - emotional_alignment(expected, actual)
    if emotional_error < 0:
        emotional_error = -emotional_error

    entropy_exp, _ = high_entropy(expected)
    entropy_act, _ = high_entropy(actual)
    entropy_error = abs(entropy_exp - entropy_act)

    # Combined Free Energy
    free_energy = semantic_error * 0.4 + emotional_error * 0.4 + entropy_error * 0.2
    return free_energy


# ═══════════════════════════════════════════════════════════════════════════════
# RESONANCE COUPLING
# ═══════════════════════════════════════════════════════════════════════════════

SCHUMANN_BASE_HZ = 7.83

def resonance_coupling(
    internal: EmotionalState,
    external_text: str,
    schumann_mod: float = 1.0,
) -> float:
    """
    Coupling strength between internal state and external text.
    """
    analysis = analyze_emotion(external_text)

    # Valence alignment
    valence_align = 1.0 - abs(internal.valence - analysis.valence) / 2.0

    # Arousal alignment
    arousal_align = 1.0 - abs(internal.arousal - analysis.arousal)

    # Entropy coupling
    ext_entropy, _ = high_entropy(external_text)
    entropy_coupling = max(0, 1.0 - abs(internal.entropy - ext_entropy) / 5.0)

    # Base coupling
    coupling = valence_align * 0.4 + arousal_align * 0.3 + entropy_coupling * 0.3

    # Schumann modulation
    schumann_coherence = max(0.5, 1.0 - abs(schumann_mod - 1.0) * 5.0)

    return coupling * schumann_coherence


# ═══════════════════════════════════════════════════════════════════════════════
# TEXT RHYTHM
# ═══════════════════════════════════════════════════════════════════════════════

VOWELS = set("aeiouyаеёиоуыэюя")

def text_rhythm(text: str) -> Tuple[float, float, float]:
    """
    Analyze syllabic rhythm.
    Returns: (avg_syllables, variance, pause_density)
    """
    words = tokenize(text)
    if not words:
        return 0.0, 0.0, 0.0

    # Estimate syllables
    syllable_counts = []
    for word in words:
        count = sum(1 for c in word.lower() if c in VOWELS)
        syllable_counts.append(max(1, count))

    # Average
    avg = sum(syllable_counts) / len(syllable_counts)

    # Variance
    variance = sum((c - avg) ** 2 for c in syllable_counts) / len(syllable_counts)

    # Pause density
    pause_count = sum(text.count(p) for p in [",", ".", ";", "—", "..."])
    pause_density = pause_count / len(words)

    return avg, variance, pause_density


# ═══════════════════════════════════════════════════════════════════════════════
# PERPLEXITY
# ═══════════════════════════════════════════════════════════════════════════════

def perplexity(text: str) -> float:
    """Character bigram perplexity"""
    if len(text) < 2:
        return 1.0

    bigram_counts = {}
    unigram_counts = {}

    for i in range(len(text) - 1):
        bigram = text[i:i+2]
        bigram_counts[bigram] = bigram_counts.get(bigram, 0) + 1
        unigram_counts[text[i]] = unigram_counts.get(text[i], 0) + 1
    unigram_counts[text[-1]] = unigram_counts.get(text[-1], 0) + 1

    log_prob = 0.0
    count = 0

    for i in range(len(text) - 1):
        bigram = text[i:i+2]
        bc = bigram_counts.get(bigram, 0)
        uc = unigram_counts.get(text[i], 0)

        if bc > 0 and uc > 0:
            p = bc / uc
            log_prob += math.log2(p)
            count += 1

    if count == 0:
        return 1.0

    avg_log_prob = log_prob / count
    return 2 ** (-avg_log_prob)
