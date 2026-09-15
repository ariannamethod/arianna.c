package main

import "testing"

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
