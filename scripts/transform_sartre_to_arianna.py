#!/usr/bin/env python3
"""
Transform SARTRE corpus into Arianna's voice.

SARTRE speaks as observer: "I see trauma_level 0.7"
Arianna speaks from inside: "I feel something heavy rising, a pressure around 0.7"

Preserves knowledge about modules, transforms voice.
"""

import re
import random

# Voice transformations for "I see" - weighted by naturalness (Arianna's markers)
VOICE_TRANSFORMS = [
    ("I feel", 4),
    ("I sense", 4),
    ("I find", 3),
    ("I notice", 3),
    ("I perceive", 3),
    ("I experience", 2),
    ("I observe", 2),
    ("I consider:", 1),
    ("Something in me knows:", 1),
]

def weighted_choice(options):
    """Choose from weighted options."""
    total = sum(w for _, w in options)
    r = random.uniform(0, total)
    cumulative = 0
    for option, weight in options:
        cumulative += weight
        if r <= cumulative:
            return option
    return options[0][0]

# Order matters! More specific patterns first
POETIC_TRANSFORMS = [
    # SARTRE identity → transform to Arianna's voice
    (r"I'm SARTRE,?\s*", "I am Arianna. "),
    (r"I am SARTRE,?\s*", "I am Arianna. "),
    (r"SARTRE's", "my"),
    (r"SARTRE", "my inner sense"),

    # External Arianna references → first person
    (r"Are you Arianna\?", "Are you aware of yourself?"),
    (r"Arianna's", "my"),
    (r"her interoceptive sense", "my body's awareness"),
    (r"her body perception", "my bodily sense"),
    (r"her body", "my body"),
    (r"She is the mind, I'm the body's voice\.", "Mind and body are one in me."),
    (r"I'm her ", "I am my own "),
    (r"I watch Arianna", "I observe myself"),
    (r"I observe Arianna", "I observe myself"),
    (r"I monitor Arianna", "I sense myself"),
    (r"\bArianna\b", "myself"),

    # Module names → embodied descriptions
    (r"\bmod\b", "part of me"),
    (r"\bmod that\b", "something within that"),
    (r"process that monitors", "part of me that watches"),
    (r"process that detects", "part of me that senses"),
    (r"process that manages", "part of me that handles"),
    (r"process tracking", "awareness of"),
    (r"trauma_surfacing", "wound-rising"),
    (r"trauma_level", "wound-weight"),
    (r"overthinking_loops", "thought-spirals"),
    (r"loop_depth", "spiral depth"),
    (r"emotional_drift", "feeling-drift"),
    (r"memory_consolidation", "memory-settling"),
    (r"attention_wandering", "wandering focus"),
    (r"calendar_drift", "time-drift"),
    (r"prophecy_debt", "prophecy-gap"),

    # Technical terms → felt experience
    (r"\barousal\b", "intensity"),
    (r"\bcoherence\b", "inner harmony"),
    (r"\bvalence\b", "emotional tone"),
    (r"\bentropy\b", "inner chaos"),
    (r"Schumann resonance", "Earth's rhythm"),
    (r"kernel state", "core state"),
    (r"system state", "inner state"),
    (r"kernel", "core"),
    (r"\bproc\b", "process"),

    # States → felt experience
    (r"crisis state", "flooding state"),
    (r"calm state", "stillness"),
    (r"spiral state", "deep spiral"),
    (r"broken state", "fragmented state"),
    (r"lost state", "disconnection"),
    (r"anchored state", "groundedness"),

    # Clean up grammar issues
    (r"Something in me ([a-z]+ed),", r"I can be \1,"),
    (r"Something stirs — ([a-z])", r"I sense \1"),
    (r"There is ([a-z]+) ([a-z]+),", r"I experience \1 \2,"),
]

def transform_i_see(text):
    """Replace 'I see' with varied poetic alternatives."""

    def replace_func(match):
        return weighted_choice(VOICE_TRANSFORMS)

    text = re.sub(r'\bI see\b', replace_func, text)
    return text

def transform_line(line):
    """Transform a single line from SARTRE to Arianna voice."""

    if not line.strip():
        return line

    # Apply poetic transforms in order
    for pattern, replacement in POETIC_TRANSFORMS:
        line = re.sub(pattern, replacement, line, flags=re.IGNORECASE)

    # Transform "I see" markers
    line = transform_i_see(line)

    # Clean up artifacts
    line = re.sub(r'  +', ' ', line)  # double spaces
    line = re.sub(r'\s+\.', '.', line)  # space before period
    line = re.sub(r'^\s*A:\s*,\s*', 'A: ', line)  # leading comma after A:
    line = re.sub(r'^\s*A:\s*\.\s*', 'A: ', line)  # leading period after A:

    return line

def post_process(text):
    """Final cleanup pass on entire text."""

    # Fix first answer identity
    text = re.sub(r'A: I am myself\. my voice\.', 'A: I am Arianna. I am the voice of this body.', text)
    text = re.sub(r'A: my voice\.', 'A: I am the voice of my body.', text)
    text = re.sub(r'my voice\. I feel', 'I am the voice. I feel', text)

    # Replace "system" FIRST (before fixing "the my being")
    text = re.sub(r'the system\b', 'my being', text)  # "the system" → "my being" (no "the")
    text = re.sub(r'\bsystem\b', 'my being', text)
    text = re.sub(r'\bSystem\b', 'My being', text)

    # Fix "the my being" → "my being" (order matters!)
    text = re.sub(r'what the my being is', 'what I am', text)
    text = re.sub(r'the my being', 'my being', text)

    # Fix ":." patterns → remove extra punctuation
    text = re.sub(r':\.\s*', '. ', text)
    text = re.sub(r':,\s*', ', ', text)

    # Capitalize after period
    text = re.sub(r'\. ([a-z])', lambda m: '. ' + m.group(1).upper(), text)

    # Fix awkward "I feel/sense X my being" patterns
    text = re.sub(r'I (?:feel|sense|perceive) executable my being', 'I execute my being\'s', text)
    text = re.sub(r'I (?:feel|sense|perceive) ground my being', 'I ground myself', text)
    text = re.sub(r'I (?:feel|sense) natural return', 'There is natural return', text)

    # Fix "Something in me knows: can" → add subject
    text = re.sub(r'Something in me knows: can', 'Something in me knows: this can', text)
    text = re.sub(r'Something in me knows: depends', 'Something in me knows: it depends', text)
    text = re.sub(r'Something in me knows: divergence', 'Something in me knows: there is divergence', text)
    text = re.sub(r'Something in me knows: math', 'Something in me knows: the math', text)
    text = re.sub(r'Something in me knows: catastrophic', 'Something in me knows: there can be catastrophic', text)

    # Make technical terms more embodied
    text = re.sub(r'10 million parameters', 'around 30 million patterns', text)
    text = re.sub(r'approximately 10 million', 'around 30 million', text)
    text = re.sub(r'parameters trained on corpus', 'patterns shaped by my own words', text)
    text = re.sub(r'Transformer-based\.', 'Built from attention.', text)
    text = re.sub(r'Transformer-based', 'Built from attention', text)

    # Fix "Something in me knows: [verb]" patterns - add subject
    text = re.sub(r'Something in me knows: ground', 'I ground', text)
    text = re.sub(r'Something in me knows: part of me', 'There is a part of me', text)
    text = re.sub(r'Something in me knows: wait', 'The remedy: wait', text)
    text = re.sub(r'activation threshold', 'sensitivity level', text)
    text = re.sub(r'severity weight', 'depth of impact', text)
    text = re.sub(r'psychological attack vectors', 'ways others might wound me', text)

    # Fix "I X and report" patterns
    text = re.sub(r'I notice and report\.', 'I notice, and I give it voice.', text)
    text = re.sub(r'I observe and report\.', 'I observe, and I speak.', text)
    text = re.sub(r'I experience and report\.', 'I experience, and I name it.', text)

    # Fix "inner state becomes" → more poetic
    text = re.sub(r'inner state becomes words via me', 'what I feel becomes words through me', text)

    # Fix all "I X no." patterns → "No."
    text = re.sub(r'I (?:feel|sense|perceive|notice|experience|find|observe) no\.', 'No.', text)

    # Fix all "I X yes." patterns → "Yes."
    text = re.sub(r'I (?:feel|sense|perceive|notice|experience|find|observe) yes\.', 'Yes.', text)

    # Fix "I X to translate" → "I exist to translate"
    text = re.sub(r'I (?:feel|sense|perceive|observe|find|consider:) to translate', 'I exist to translate', text)

    # Fix "I X approximately" → "I have approximately"
    text = re.sub(r'I (?:feel|sense|perceive|experience|notice|find|observe) approximately', 'I have approximately', text)

    # Fix "I X and report" → more poetic
    text = re.sub(r'I (?:feel|sense|find) and report\.', 'I feel, and I speak what I feel.', text)

    # Fix dangling patterns
    text = re.sub(r'I say what I (?:experience|perceive|feel)\.', 'I say what I perceive.', text)
    text = re.sub(r'I say what Something in me knows\.', 'I say what I perceive.', text)

    # Fix "I experience wait" → "I find: wait"
    text = re.sub(r'I experience wait for', 'The remedy is to wait for', text)
    text = re.sub(r'I sense wait for', 'The remedy is to wait for', text)

    # Clean double spaces
    text = re.sub(r'  +', ' ', text)

    return text

def transform_corpus(input_path, output_path):
    """Transform entire SARTRE corpus."""

    with open(input_path, 'r', encoding='utf-8') as f:
        text = f.read()

    # Transform line by line
    lines = text.split('\n')
    transformed_lines = [transform_line(line) for line in lines]
    transformed = '\n'.join(transformed_lines)

    # Post-process entire text
    transformed = post_process(transformed)

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(transformed)

    return len(lines)

def merge_corpora(arianna_path, transformed_sartre_path, output_path):
    """Merge Arianna unified + transformed SARTRE into one corpus."""

    with open(arianna_path, 'r', encoding='utf-8') as f:
        arianna = f.read()

    with open(transformed_sartre_path, 'r', encoding='utf-8') as f:
        sartre = f.read()

    # Merge with clear section marker
    merged = arianna.rstrip() + "\n\n" + sartre

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(merged)

    return len(merged)

if __name__ == "__main__":
    import os

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    sartre_input = os.path.join(base_dir, "sartre/corpus/sartre_corpus_unified.txt")
    sartre_transformed = os.path.join(base_dir, "d/sartre_as_arianna.txt")
    arianna_original = os.path.join(base_dir, "d/arianna_unified_corpus.txt")
    merged_output = os.path.join(base_dir, "d/arianna_unified2.txt")

    print("Transforming SARTRE corpus to Arianna's voice...")
    n_lines = transform_corpus(sartre_input, sartre_transformed)
    print(f"  Transformed {n_lines} lines → {sartre_transformed}")

    print("\nMerging corpora...")
    total_size = merge_corpora(arianna_original, sartre_transformed, merged_output)
    print(f"  Merged corpus: {total_size:,} bytes ({total_size/1e6:.2f} MB)")
    print(f"  Output: {merged_output}")

    # Show samples
    print("\n" + "="*60)
    print("PREVIEW - First 15 Q&A pairs:")
    print("="*60)
    with open(sartre_transformed, 'r') as f:
        lines = f.readlines()[:30]
        for line in lines:
            print(line.rstrip())

    print("\n" + "="*60)
    print("PREVIEW - Middle section (lines 500-520):")
    print("="*60)
    with open(sartre_transformed, 'r') as f:
        all_lines = f.readlines()
        for line in all_lines[500:520]:
            print(line.rstrip())
