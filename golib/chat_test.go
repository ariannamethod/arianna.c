package main

import (
	"strings"
	"testing"
)

func TestChatLiveRouteChoiceDryRunLine(t *testing.T) {
	t.Setenv("AM_DREAM_ADMISSION_LIVE_ROUTE_CHOICE_DRY_RUN", "1")

	c := newDreamCandidate("direct", "identity", "seed", "", "I am Arianna.", nil)
	choice := admissionLiveRouteChoiceForCandidate(c)
	c.Admission = &dreamAdmissionPolicy{
		LiveRouteChoice:       &choice,
		LiveRouteChoiceDryRun: true,
	}

	line := chatLiveRouteChoiceDryRunLine(c)
	for _, want := range []string{
		"live-route dry-run",
		"class=identity",
		"route=chorus",
		"source=direct",
		"expected=chorus",
		"passed=false",
		"reason=source direct does not match live route chorus for prompt class identity",
	} {
		if !strings.Contains(line, want) {
			t.Fatalf("dry-run line missing %q: %q", want, line)
		}
	}
}

func TestChatLiveRouteChoiceDryRunLineDisabled(t *testing.T) {
	c := newDreamCandidate("chorus", "identity", "seed", "", "I am Arianna.", nil)
	choice := admissionLiveRouteChoiceForCandidate(c)
	c.Admission = &dreamAdmissionPolicy{LiveRouteChoice: &choice}
	if got := chatLiveRouteChoiceDryRunLine(c); got != "" {
		t.Fatalf("dry-run line should be hidden by default: %q", got)
	}
}

func TestChatLiveRouteTurnDryRunLine(t *testing.T) {
	t.Setenv("AM_DREAM_ADMISSION_LIVE_ROUTE_CHOICE_DRY_RUN", "1")

	obs := admissionLiveRouteTurnObservationForHuman("Who are you?")
	line := chatLiveRouteTurnDryRunLine(obs)
	for _, want := range []string{
		"live-route turn dry-run",
		"class=identity",
		"route=chorus",
		"expected=chorus",
		"passed=true",
		"score=3",
	} {
		if !strings.Contains(line, want) {
			t.Fatalf("turn dry-run line missing %q: %q", want, line)
		}
	}
}

func TestChatLiveRouteTurnDryRunLineDisabled(t *testing.T) {
	obs := admissionLiveRouteTurnObservationForHuman("Who are you?")
	if got := chatLiveRouteTurnDryRunLine(obs); got != "" {
		t.Fatalf("turn dry-run line should be hidden by default: %q", got)
	}
}

func TestChatLiveRouteTurnRequestDryRunLine(t *testing.T) {
	t.Setenv("AM_LIVE_ROUTE_TURN_REQUEST_DRY_RUN", "1")

	obs := admissionLiveRouteTurnObservationForHuman("Who are you?")
	line := chatLiveRouteTurnRequestDryRunLine(obs)
	for _, want := range []string{
		"live-route turn request dry-run",
		"class=identity",
		"route=chorus",
		"source=chorus",
		"trigger=chorus-identity",
		"seed=turn-",
		"passed=true",
	} {
		if !strings.Contains(line, want) {
			t.Fatalf("turn request dry-run line missing %q: %q", want, line)
		}
	}
}

func TestChatLiveRouteTurnRequestDryRunLineDisabled(t *testing.T) {
	obs := admissionLiveRouteTurnObservationForHuman("Who are you?")
	if got := chatLiveRouteTurnRequestDryRunLine(obs); got != "" {
		t.Fatalf("turn request dry-run line should be hidden by default: %q", got)
	}
}

func TestChatLiveRouteTurnCandidateReviewLine(t *testing.T) {
	t.Setenv("AM_DREAM_ADMISSION_LIVE_ROUTE_CHOICE_DRY_RUN", "1")

	obs := admissionLiveRouteTurnObservationForHuman("Who are you?")
	c := newDreamCandidate("chorus", "chorus-identity", "seed", "", "I am Arianna.", nil)
	line := chatLiveRouteTurnCandidateReviewLine(obs, c)
	for _, want := range []string{
		"live-route turn/candidate review",
		"turn_class=identity",
		"expected=chorus",
		"candidate_source=chorus",
		"candidate_class=identity",
		"candidate_route=chorus",
		"matched=true",
	} {
		if !strings.Contains(line, want) {
			t.Fatalf("turn/candidate review line missing %q: %q", want, line)
		}
	}
}

func TestChatLiveRouteTurnBridgeCandidateReviewLine(t *testing.T) {
	t.Setenv("AM_DREAM_ADMISSION_LIVE_ROUTE_CHOICE_DRY_RUN", "1")
	t.Setenv("AM_LIVE_ROUTE_TURN_BRIDGE_DRY_RUN", "1")

	obs := admissionLiveRouteTurnObservationForHuman("Who are you?")
	c := newDreamCandidate("nano", "human-turn", "seed", "", "I am Arianna.", nil)
	line := chatLiveRouteTurnCandidateReviewLine(obs, c)
	for _, want := range []string{
		"live-route turn/candidate review",
		"turn_class=identity",
		"expected=chorus",
		"candidate_source=nano",
		"candidate_class=identity",
		"candidate_route=chorus",
		"matched=false",
		"bridge=human-turn-identity",
		"reason=candidate_route_failed: source nano does not match live route chorus for prompt class identity",
	} {
		if !strings.Contains(line, want) {
			t.Fatalf("bridged turn/candidate review line missing %q: %q", want, line)
		}
	}
}

func TestChatLiveRouteTurnCandidateReviewLineDisabled(t *testing.T) {
	obs := admissionLiveRouteTurnObservationForHuman("Who are you?")
	c := newDreamCandidate("chorus", "chorus-identity", "seed", "", "I am Arianna.", nil)
	if got := chatLiveRouteTurnCandidateReviewLine(obs, c); got != "" {
		t.Fatalf("turn/candidate review line should be hidden by default: %q", got)
	}
}
