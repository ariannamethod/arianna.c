package main

import (
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

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

func TestDreamAdmissionShadowWritesJSONLReceipt(t *testing.T) {
	t.Setenv("AM_DREAM_ADMISSION", dreamAdmissionShadow)
	logPath := filepath.Join(t.TempDir(), "dream-admission.jsonl")
	t.Setenv("AM_DREAM_ADMISSION_LOG", logPath)

	iw := NewInnerWorld()
	iw.Start(false)
	defer iw.Stop()

	r := dreamResult{
		dream:     "the field dreams inward before it speaks",
		candidate: newDreamCandidate("nano", "test", "seed", "", "the field dreams inward before it speaks", nil),
	}
	if admitDreamToInnerWorld(iw, &r, "test") {
		t.Fatal("shadow dream candidate must not be admitted")
	}
	raw, err := os.ReadFile(logPath)
	if err != nil {
		t.Fatal(err)
	}
	lines := strings.Split(strings.TrimSpace(string(raw)), "\n")
	if len(lines) != 1 {
		t.Fatalf("expected one JSONL receipt, got %d: %q", len(lines), raw)
	}
	var got dreamCandidate
	if err := json.Unmarshal([]byte(lines[0]), &got); err != nil {
		t.Fatal(err)
	}
	if got.Schema != "arianna.dream_candidate.v1" || got.Mode != dreamAdmissionShadow || got.Accepted || got.Reason != "shadow mode" {
		t.Fatalf("bad shadow receipt: %+v", got)
	}
}

func TestDreamAdmissionLiveFailsClosedWhenRequestedLogCannotWrite(t *testing.T) {
	t.Setenv("AM_DREAM_ADMISSION", dreamAdmissionLive)
	t.Setenv("AM_DREAM_ADMISSION_LOG", filepath.Join(t.TempDir(), "missing", "dream-admission.jsonl"))

	iw := NewInnerWorld()
	iw.Start(false)
	defer iw.Stop()

	r := dreamResult{
		dream:     "the field wants to become state",
		candidate: newDreamCandidate("nano", "test", "seed", "", "the field wants to become state", nil),
	}
	if admitDreamToInnerWorld(iw, &r, "test") {
		t.Fatal("live admission with a requested but unwritable ledger must fail closed")
	}
	if !strings.HasPrefix(r.candidate.Reason, "admission log failed:") {
		t.Fatalf("bad log failure reason: %+v", r.candidate)
	}
}
