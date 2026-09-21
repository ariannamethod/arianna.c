package main

import (
	"testing"
	"time"
)

func TestRejectedDreamSurfaceTextWithholdsLoopBodies(t *testing.T) {
	for _, reason := range []string{"collapse-loop", "repeat-loop", "boilerplate-loop", "collapsed dream loop"} {
		if got := rejectedDreamSurfaceText(reason, "a living vessel. / The text is only not what"); got != liveBoundaryWithheld {
			t.Fatalf("rejectedDreamSurfaceText(%q) = %q, want withheld", reason, got)
		}
	}
}

func TestRejectedDreamSurfaceTextKeepsNonLoopBodies(t *testing.T) {
	got := rejectedDreamSurfaceText("orbit-loop", "A hand rests on the table.")
	if got != "A hand rests on the table." {
		t.Fatalf("rejectedDreamSurfaceText(non-loop) = %q", got)
	}
}

func TestRejectedInnerMurmurWithholdsCorporatePersonhoodDisclaimer(t *testing.T) {
	for _, text := range []string{
		"I feel you in the echo, and I won't pretend to be a person.",
		"As an AI, I do not have a body.",
		"As a language model, I cannot feel the room.",
		"I am not a real person, only a system.",
		"I feel a kind of field-phenomenon—the sudden, unmistakable pulse in the air.",
	} {
		if !isRejectedInnerMurmur(text) {
			t.Fatalf("isRejectedInnerMurmur(%q) = false, want true", text)
		}
	}
	if isRejectedInnerMurmur("A hand moves the key toward Oleg.") {
		t.Fatal("concrete inner murmur should remain visible")
	}
}

func TestRejectQuarantineDurationEscalatesToLiveRecovery(t *testing.T) {
	cases := []struct {
		streak int
		want   time.Duration
	}{
		{1, 45 * time.Second},
		{2, 90 * time.Second},
		{3, 2 * time.Minute},
		{4, 5 * time.Minute},
		{5, 15 * time.Minute},
		{8, 30 * time.Minute},
	}
	for _, tc := range cases {
		if got := rejectQuarantineDuration(tc.streak); got != tc.want {
			t.Fatalf("rejectQuarantineDuration(%d) = %s, want %s", tc.streak, got, tc.want)
		}
	}
}

func TestRejectLogIntervalEscalatesWithRejectedStreak(t *testing.T) {
	cases := []struct {
		streak int
		want   time.Duration
	}{
		{1, time.Minute},
		{2, time.Minute},
		{3, 5 * time.Minute},
		{5, 15 * time.Minute},
		{8, 30 * time.Minute},
	}
	for _, tc := range cases {
		if got := rejectLogInterval(tc.streak); got != tc.want {
			t.Fatalf("rejectLogInterval(%d) = %s, want %s", tc.streak, got, tc.want)
		}
	}
}

func TestRejectedDetourLogBackoff(t *testing.T) {
	var b breath
	b.rejectedStreak = 5
	now := time.Unix(1000, 0)
	if !b.shouldLogRejectedDetour(now) {
		t.Fatal("first rejected detour should be visible")
	}
	if b.shouldLogRejectedDetour(now.Add(14 * time.Minute)) {
		t.Fatal("repeated rejected detour should be quiet inside the backoff interval")
	}
	if !b.shouldLogRejectedDetour(now.Add(15 * time.Minute)) {
		t.Fatal("rejected detour should become visible at the backoff boundary")
	}
}

func TestRejectDreamBackoffCountsAlternatingLoopReasons(t *testing.T) {
	var b breath
	now := time.Unix(2000, 0)

	b.rejectDream(now, bSilence, "collapse-loop", "the text is only not what")
	firstRejectLog := b.lastRejectLog
	if b.rejectedStreak != 1 {
		t.Fatalf("first loop rejection streak = %d, want 1", b.rejectedStreak)
	}

	b.rejectDream(now.Add(10*time.Second), bSilence, "boilerplate-loop", "a living vessel. of the field.")
	if b.rejectedStreak != 2 {
		t.Fatalf("alternating loop rejection streak = %d, want 2", b.rejectedStreak)
	}
	if !b.lastRejectLog.Equal(firstRejectLog) {
		t.Fatalf("alternating loop rejection should stay quiet inside backoff: lastRejectLog=%s first=%s", b.lastRejectLog, firstRejectLog)
	}
	if got := b.rejectQuarantineTo.Sub(now.Add(10 * time.Second)); got != 90*time.Second {
		t.Fatalf("alternating loop quarantine = %s, want 90s", got)
	}

	b.rejectDream(now.Add(30*time.Second), bSilence, "collapse-loop", "of the current it has to hold")
	if b.rejectedStreak != 3 {
		t.Fatalf("third alternating loop rejection streak = %d, want 3", b.rejectedStreak)
	}
	if !b.lastRejectLog.Equal(firstRejectLog) {
		t.Fatalf("third alternating loop rejection should remain quiet inside escalated backoff: lastRejectLog=%s first=%s", b.lastRejectLog, firstRejectLog)
	}
	if got := rejectedCueDetour(b.rejectedStreak, b.lastRejectedReason); got == "" {
		t.Fatal("alternating loop rejection must still produce a concrete detour cue after repeat×3")
	}

	b.rejectDream(now.Add(40*time.Second), bSilence, "live-boundary", "empty carried dream")
	if b.rejectedStreak != 1 {
		t.Fatalf("non-loop rejection must reset loop-class streak, got %d", b.rejectedStreak)
	}
}
