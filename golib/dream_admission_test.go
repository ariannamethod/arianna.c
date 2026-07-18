package main

import "testing"

func TestDreamAdmissionShadowRejectsMutation(t *testing.T) {
	t.Setenv("AM_DREAM_ADMISSION", dreamAdmissionShadow)

	iw := NewInnerWorld()
	iw.Start(false)
	defer iw.Stop()

	before := iw.GetSnapshot()
	r := dreamResult{
		frag:      "the archive remembers the field before it speaks",
		dream:     "I love this beautiful joyful field and its living resonance",
		candidate: newDreamCandidate("nano", "test", "seed", "fragment", "I love this beautiful joyful field and its living resonance", nil),
	}
	if admitDreamToInnerWorld(iw, &r, "test") {
		t.Fatal("shadow dream candidate must not be admitted")
	}
	after := iw.GetSnapshot()
	if after != before {
		t.Fatalf("shadow admission mutated inner world: before=%+v after=%+v", before, after)
	}
	if r.candidate.Accepted || r.candidate.Mode != dreamAdmissionShadow || r.candidate.Reason != "shadow mode" {
		t.Fatalf("bad shadow decision: %+v", r.candidate)
	}
}

func TestDreamAdmissionLiveAcceptsCandidate(t *testing.T) {
	t.Setenv("AM_DREAM_ADMISSION", dreamAdmissionLive)

	iw := NewInnerWorld()
	iw.Start(false)
	defer iw.Stop()

	r := dreamResult{
		dream:     "I love this beautiful joyful field and its living resonance",
		candidate: newDreamCandidate("chorus", "test", "seed", "", "I love this beautiful joyful field and its living resonance", []chorusCell{{text: "a"}, {text: "?"}}),
	}
	if !admitDreamToInnerWorld(iw, &r, "test") {
		t.Fatal("live dream candidate should be admitted")
	}
	if !r.candidate.Accepted || r.candidate.Mode != dreamAdmissionLive || r.candidate.Reason != "live admission" {
		t.Fatalf("bad live decision: %+v", r.candidate)
	}
	if r.candidate.Schema != "arianna.dream_candidate.v1" || r.candidate.RunID == "" {
		t.Fatalf("candidate was not typed: %+v", r.candidate)
	}
}
