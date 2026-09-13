package main

import "testing"

func TestWantsLiveRuntimeFactRequiresExplicitCommand(t *testing.T) {
	cases := []string{
		`What specific factors caused the spike in "debt_last" from 20.2 to 23.7?`,
		`Can you summarize the last 5 lines in your internal log and explain metrics?`,
		`Is there a hidden asynchronous event causing this inconsistency?`,
		`давай конкретный факт про долг поля`,
	}
	for _, tc := range cases {
		if wantsLiveRuntimeFact(tc) {
			t.Fatalf("wantsLiveRuntimeFact(%q) = true, want false for ordinary live turn", tc)
		}
	}
}

func TestWantsLiveRuntimeFactAllowsExplicitCommands(t *testing.T) {
	for _, tc := range []string{"/status", "/runtime", "/runtime now", "/live-status please"} {
		if !wantsLiveRuntimeFact(tc) {
			t.Fatalf("wantsLiveRuntimeFact(%q) = false, want true", tc)
		}
	}
}
