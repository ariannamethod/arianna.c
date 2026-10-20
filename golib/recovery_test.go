package main

import (
	"context"
	"math"
	"os"
	"path/filepath"
	"sync"
	"testing"
	"time"
)

func TestInnerRecoveryLeavesBaselineStable(t *testing.T) {
	iw := NewInnerWorld()
	for i := 0; i < 600; i++ {
		iw.Step(0.1)
	}
	s := iw.GetSnapshot()
	if s.Entropy != baselineEntropy || s.Coherence != baselineCoherence {
		t.Fatalf("baseline drifted: entropy=%g coherence=%g", s.Entropy, s.Coherence)
	}
}

func TestInnerRecoveryEscapesBoundaryWithoutTrauma(t *testing.T) {
	iw := NewInnerWorld()
	iw.Start(false)
	defer iw.Stop()
	iw.State.Entropy, iw.State.Coherence = 1, 0
	iw.Step(0.1)
	s := iw.GetSnapshot()
	if s.Entropy >= 1 || s.Coherence <= 0 || s.TraumaLevel != 0 {
		t.Fatalf("state remained pinned: %+v", s)
	}
	for i := 1; i < 600; i++ {
		iw.Step(0.1)
	}
	s = iw.GetSnapshot()
	if s.Entropy >= 0.7 || s.Coherence <= 0.3 {
		t.Fatalf("no sustained recovery after one minute: %+v", s)
	}
}

func TestInnerRecoveryDependsOnElapsedTime(t *testing.T) {
	a, b := NewInnerWorld(), NewInnerWorld()
	a.State.Entropy, a.State.Coherence = 1, 0
	b.State.Entropy, b.State.Coherence = 1, 0
	a.Step(60)
	for i := 0; i < 600; i++ {
		b.Step(0.1)
	}
	x, y := a.GetSnapshot(), b.GetSnapshot()
	if math.Abs(float64(x.Entropy-y.Entropy)) > 1e-5 || math.Abs(float64(x.Coherence-y.Coherence)) > 1e-5 {
		t.Fatalf("tick frequency changed recovery: %+v / %+v", x, y)
	}
}

func TestAutonomousVoiceYieldsToPendingHumans(t *testing.T) {
	tc := &trioCtx{}
	var mu sync.Mutex
	tc.pendingHumans.Add(2)
	for remaining := int64(2); remaining > 0; remaining-- {
		if tc.lockAutonomousVoice(&mu) {
			mu.Unlock()
			t.Fatal("autonomous voice acquired while a human was pending")
		}
		tc.pendingHumans.Add(-1)
	}
	if !tc.lockAutonomousVoice(&mu) {
		t.Fatal("autonomous voice did not resume after the last human turn")
	}
	mu.Unlock()
}

func TestAutonomousVoiceRechecksAfterLock(t *testing.T) {
	tc := &trioCtx{}
	var mu sync.Mutex
	mu.Lock()
	done := make(chan bool, 1)
	go func() {
		locked := tc.lockAutonomousVoice(&mu)
		if locked {
			mu.Unlock()
		}
		done <- locked
	}()
	tc.pendingHumans.Add(1)
	mu.Unlock()
	select {
	case locked := <-done:
		if locked {
			t.Fatal("autonomous voice overtook a waiting human")
		}
	case <-time.After(time.Second):
		t.Fatal("autonomous voice failed to release the lock")
	}
}

func TestChorusReceivesNanoTrainingFrame(t *testing.T) {
	dir := t.TempDir()
	bin := filepath.Join(dir, "chorus")
	captured := filepath.Join(dir, "prompt")
	t.Setenv("TEST_CHORUS_PROMPT", captured)
	script := "#!/bin/sh\nprintf '%s' \"$2\" > \"$TEST_CHORUS_PROMPT\"\nprintf '  r1 cell 0 (T=0.60):  rain\\n'\n"
	if err := os.WriteFile(bin, []byte(script), 0700); err != nil {
		t.Fatal(err)
	}
	cells := choir(context.Background(), bin, "body.gguf", "  дождь\nrain  ", 1)
	got, err := os.ReadFile(captured)
	if err != nil || string(got) != "Q: дождь\nrain\nA:" || len(cells) != 1 {
		t.Fatalf("chorus frame=%q cells=%v err=%v", got, cells, err)
	}
}
