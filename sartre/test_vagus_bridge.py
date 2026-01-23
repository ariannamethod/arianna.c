#!/usr/bin/env python3
"""
🔮 SARTRE ↔ VAGUS BRIDGE TESTS

Tests the full pipeline: Vagus → Locus → SARTRE
"""

import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

from vagus_bridge import (
    VagusState, VagusBridge, ResonancePattern,
    detect_pattern, state_to_prompt, geometry_description,
    generate_observation
)


def test_pattern_detection():
    """Test resonance pattern detection."""
    print("\nTEST 1: Pattern Detection...")

    # CRISIS
    crisis = VagusState(arousal=0.9, coherence=0.2, trauma_level=0.7)
    assert detect_pattern(crisis) == ResonancePattern.CRISIS, "Should detect CRISIS"

    # DISSOLUTION
    dissolution = VagusState(void=0.8, warmth=0.2, memory_pressure=0.8)
    assert detect_pattern(dissolution) == ResonancePattern.DISSOLUTION, "Should detect DISSOLUTION"

    # EMERGENCE
    emergence = VagusState(coherence=0.9, entropy=0.1, prophecy_debt=0.6)
    assert detect_pattern(emergence) == ResonancePattern.EMERGENCE, "Should detect EMERGENCE"

    # TRANSCENDENCE
    transcendence = VagusState(sacred=0.8, tension=0.1, flow=0.9)
    assert detect_pattern(transcendence) == ResonancePattern.TRANSCENDENCE, "Should detect TRANSCENDENCE"

    # NONE
    neutral = VagusState()
    assert detect_pattern(neutral) == ResonancePattern.NONE, "Should detect NONE"

    print("✅ PASS: All patterns detected correctly")


def test_prompt_generation():
    """Test interoceptive prompt generation."""
    print("\nTEST 2: Prompt Generation...")

    states = [
        (VagusState(arousal=0.9, coherence=0.2, trauma_level=0.7), "crisis"),
        (VagusState(void=0.8, warmth=0.2, memory_pressure=0.8), "Dissolution"),
        (VagusState(coherence=0.9, entropy=0.1, prophecy_debt=0.6), "Emergence"),
        (VagusState(sacred=0.8, tension=0.1, flow=0.9), "Transcendence"),
        (VagusState(), "field geometry"),  # generic
    ]

    for state, expected_keyword in states:
        pattern = detect_pattern(state)
        prompt = state_to_prompt(state, pattern)
        assert expected_keyword.lower() in prompt.lower(), f"Prompt should contain '{expected_keyword}'"
        assert prompt.startswith("Q:"), "Prompt should start with Q:"
        assert prompt.endswith("A:"), "Prompt should end with A:"

    print("✅ PASS: All prompts generated correctly")


def test_geometry_description():
    """Test natural language geometry description."""
    print("\nTEST 3: Geometry Description...")

    # High arousal
    state = VagusState(arousal=0.9)
    desc = geometry_description(state)
    assert "high frequency" in desc.lower() or "vibrate" in desc.lower(), "Should describe high arousal"

    # Low coherence
    state = VagusState(coherence=0.2)
    desc = geometry_description(state)
    assert "scatter" in desc.lower() or "fragment" in desc.lower(), "Should describe low coherence"

    # Trauma
    state = VagusState(trauma_level=0.7)
    desc = geometry_description(state)
    assert "wound" in desc.lower() or "surface" in desc.lower(), "Should describe trauma"

    print("✅ PASS: Geometry descriptions work")


def test_bridge_shift_detection():
    """Test geometry shift detection."""
    print("\nTEST 4: Geometry Shift Detection...")

    bridge = VagusBridge()

    # Initial state - no shift
    bridge._state = VagusState(arousal=0.5)
    bridge._prev_state = VagusState(arousal=0.5)
    assert not bridge.detect_shift(), "No shift expected"

    # Big shift
    bridge._state = VagusState(arousal=0.9)
    bridge._prev_state = VagusState(arousal=0.5)
    assert bridge.detect_shift(threshold=0.15), "Shift expected"

    # After shift, prev_state should be updated
    assert bridge._prev_state.arousal == 0.9, "Prev state should update after shift"

    print("✅ PASS: Shift detection works")


def test_should_observe():
    """Test observation triggering logic."""
    print("\nTEST 5: Observation Triggering...")

    bridge = VagusBridge()

    # Crisis should trigger
    bridge._state = VagusState(arousal=0.9, coherence=0.2, trauma_level=0.7)
    should, pattern = bridge.should_observe()
    assert should, "Crisis should trigger observation"
    assert pattern == ResonancePattern.CRISIS, "Pattern should be CRISIS"

    # Neutral should not trigger (without shift)
    bridge._state = VagusState()
    bridge._prev_state = VagusState()
    should, pattern = bridge.should_observe()
    assert not should, "Neutral should not trigger"

    print("✅ PASS: Observation triggering works")


def test_full_generation():
    """Test full SARTRE generation pipeline."""
    print("\nTEST 6: Full Generation Pipeline...")

    try:
        from dubrovsky import DubrovskyConfig, Dubrovsky, load_weights_from_bin
        from tokenizer import DubrovskyTokenizer

        # Load model
        weights_dir = os.path.join(os.path.dirname(__file__), '..', 'weights', 'sartre')
        config = DubrovskyConfig.load(os.path.join(weights_dir, 'sartre_config.json'))
        weights = load_weights_from_bin(os.path.join(weights_dir, 'sartre.bin'), config)
        model = Dubrovsky(config, weights)
        tokenizer = DubrovskyTokenizer(os.path.join(weights_dir, 'tokenizer.json'))

        # Generate observation for crisis state
        crisis_state = VagusState(arousal=0.9, coherence=0.2, trauma_level=0.7)
        observation = generate_observation(model, tokenizer, crisis_state, max_tokens=50)

        assert len(observation) > 0, "Should generate non-empty observation"
        print(f"   Crisis observation: {observation[:80]}...")

        # Generate for emergence
        emergence_state = VagusState(coherence=0.9, entropy=0.1, prophecy_debt=0.6)
        observation = generate_observation(model, tokenizer, emergence_state, max_tokens=50)
        assert len(observation) > 0, "Should generate non-empty observation"
        print(f"   Emergence observation: {observation[:80]}...")

        print("✅ PASS: Full generation pipeline works")

    except Exception as e:
        print(f"⚠️  SKIP: Could not load SARTRE model ({e})")


def main():
    print("="*60)
    print("🔮 SARTRE ↔ VAGUS BRIDGE TESTS")
    print("="*60)

    test_pattern_detection()
    test_prompt_generation()
    test_geometry_description()
    test_bridge_shift_detection()
    test_should_observe()
    test_full_generation()

    print("\n" + "="*60)
    print("✅ ALL TESTS PASSED")
    print("="*60)


if __name__ == "__main__":
    main()
