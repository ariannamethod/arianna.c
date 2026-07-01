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

// TestHighWiredEmotion proves the brain is wired into the emotional drift: positive text pulls
// valence up and negative text pulls it down, through the real HighValence in ProcessText.
func TestHighWiredEmotion(t *testing.T) {
	iw := NewInnerWorld()
	iw.Start(false)
	ed := iw.GetEmotionalDrift()
	if ed == nil {
		t.Fatal("emotional_drift not registered")
	}
	before := ed.position.Valence
	iw.ProcessText("I love this beautiful wonderful joyful brilliant field")
	up := ed.position.Valence
	if up <= before {
		t.Errorf("positive text should nudge valence up via the High brain: %v -> %v", before, up)
	}
	iw.ProcessText("I hate this terrible awful horrible painful empty void")
	down := ed.position.Valence
	if down >= up {
		t.Errorf("negative text should nudge valence down via the High brain: %v -> %v", up, down)
	}
}
