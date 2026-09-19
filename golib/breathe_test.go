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
