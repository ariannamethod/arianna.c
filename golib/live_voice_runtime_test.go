package main

import "testing"

func TestLiveVoiceTokenBudgetDefaultsPastLiveTruncation(t *testing.T) {
	t.Setenv("AM_VOICE_N", "")
	t.Setenv("AM_JANUS_N", "")
	t.Setenv("AM_RESONANCE_N", "")
	if got := liveVoiceTokenBudget("janus"); got != defaultLiveVoiceTokenBudget {
		t.Fatalf("liveVoiceTokenBudget default = %d, want %d", got, defaultLiveVoiceTokenBudget)
	}
	if got := liveVoiceTokenBudget("resonance"); got != defaultLiveVoiceTokenBudget {
		t.Fatalf("liveVoiceTokenBudget resonance default = %d, want %d", got, defaultLiveVoiceTokenBudget)
	}
	if defaultLiveVoiceTokenBudget <= 28 {
		t.Fatalf("defaultLiveVoiceTokenBudget = %d, want above old live truncation budget 28", defaultLiveVoiceTokenBudget)
	}
}

func TestLiveVoiceTokenBudgetGlobalAndRoleOverride(t *testing.T) {
	t.Setenv("AM_VOICE_N", "64")
	t.Setenv("AM_JANUS_N", "")
	t.Setenv("AM_RESONANCE_N", "96")
	if got := liveVoiceTokenBudget("janus"); got != 64 {
		t.Fatalf("liveVoiceTokenBudget janus global = %d, want 64", got)
	}
	if got := liveVoiceTokenBudget("resonance"); got != 96 {
		t.Fatalf("liveVoiceTokenBudget resonance role override = %d, want 96", got)
	}
}

func TestLiveVoiceTokenBudgetRejectsUnsafeValues(t *testing.T) {
	for _, raw := range []string{"15", "513", "noise"} {
		if got, ok := parseLiveVoiceTokenBudget(raw); ok {
			t.Fatalf("parseLiveVoiceTokenBudget(%q) = %d, true; want false", raw, got)
		}
	}
}
