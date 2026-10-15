package main

import "testing"

// TestHighWiredOverthinking proves the High brain is actually wired into the overthinking
// process — not dormant. Two identical turns: the intra-utterance heuristic alone returns ~0
// here, but the real cross-turn HighNgramOverlap sees a full echo and drives the repetition
// score high. A passing assertion means the Julia brain reached the process.
func TestHighWiredOverthinking(t *testing.T) {
	ol := NewOverthinkingLoops()
	echo := "the resonance of the living field flows between the two of us"
	ol.AnalyzeText(echo)      // establishes lastText
	r := ol.AnalyzeText(echo) // identical echo → cross-turn n-gram overlap is high
	if r.RepetitionScore <= 0.5 {
		t.Errorf("cross-turn echo should raise repetition via the High brain, got %v", r.RepetitionScore)
	}
}

// TestHighWiredStructuralSurprise proves that arbitrary vocabulary receives no
// runtime valence label while cross-turn structural surprise still reaches the body.
func TestHighWiredStructuralSurprise(t *testing.T) {
	iw := NewInnerWorld()
	iw.Start(false)
	ed := iw.GetEmotionalDrift()
	if ed == nil {
		t.Fatal("emotional_drift not registered")
	}
	iw.ProcessText("amber silver cedar river")
	afterFirst := ed.position.Valence
	if afterFirst != 0 {
		t.Fatalf("first arbitrary utterance assigned runtime valence: %v", afterFirst)
	}
	iw.ProcessText("quartz orbit linen thunder")
	afterSecond := ed.position.Valence
	if afterSecond >= afterFirst {
		t.Errorf("cross-turn structural surprise did not reach emotional drift: %v -> %v", afterFirst, afterSecond)
	}
}
