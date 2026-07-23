package main

import (
	"encoding/json"
	"os"
	"path/filepath"
	"reflect"
	"sort"
	"strings"
	"testing"
)

func TestAdmissionLiveRouteGateSmokeCasesCoverPlan(t *testing.T) {
	cases := admissionLiveRouteGateSmokeCases()
	wantMatched := len(admissionLiveRoutePromptClasses())
	if len(cases) != wantMatched+2 {
		t.Fatalf("bad live route smoke case count: got %d want %d", len(cases), wantMatched+2)
	}
	seenClasses := make(map[string]bool)
	seenRoutes := make(map[string]bool)
	var rejected int
	for _, tc := range cases {
		if tc.wantPassed {
			seenClasses[tc.wantPromptClass] = true
			seenRoutes[tc.wantRoute] = true
			if tc.source != tc.wantSource {
				t.Fatalf("matched smoke case source mismatch: %+v", tc)
			}
			if got := qloopSweepPromptClass(tc.trigger, tc.seed); got != tc.wantPromptClass {
				t.Fatalf("matched smoke trigger %q normalized to %q, want %q", tc.trigger, got, tc.wantPromptClass)
			}
			wantTrigger := admissionLiveRouteGateSmokeTrigger(tc.wantRoute, tc.wantPromptClass)
			if tc.trigger != wantTrigger {
				t.Fatalf("matched smoke trigger mismatch: got %q want %q", tc.trigger, wantTrigger)
			}
			continue
		}
		rejected++
	}
	if rejected != 2 {
		t.Fatalf("expected wrong-source and unknown-class reject cases, got %d", rejected)
	}
	for _, promptClass := range admissionLiveRoutePromptClasses() {
		if !seenClasses[promptClass] {
			t.Fatalf("prompt class %s missing from live route smoke cases", promptClass)
		}
	}
	wantRoutes := []string{"chorus", "direct", "qloop_hint_qa", "qloop_target", "user_bridge"}
	var gotRoutes []string
	for route := range seenRoutes {
		gotRoutes = append(gotRoutes, route)
	}
	sort.Strings(gotRoutes)
	if !reflect.DeepEqual(gotRoutes, wantRoutes) {
		t.Fatalf("bad live route smoke routes: got %v want %v", gotRoutes, wantRoutes)
	}
}

func TestAdmissionLiveRouteGateSmokeWritesBroadMatchedAndRejectedReceipts(t *testing.T) {
	t.Setenv("AM_DREAM_ADMISSION", dreamAdmissionShadow)
	t.Setenv("AM_DREAM_ADMISSION_ALLOWED_SOURCES", "")
	t.Setenv("AM_DREAM_ADMISSION_REQUIRE_LIVE_ROUTE_PLAN", "1")
	logPath := filepath.Join(t.TempDir(), "live-route-gate.jsonl")
	t.Setenv("AM_DREAM_ADMISSION_LOG", logPath)

	if err := runAdmissionLiveRouteGateSmoke(); err != nil {
		t.Fatal(err)
	}

	raw, err := os.ReadFile(logPath)
	if err != nil {
		t.Fatal(err)
	}
	lines := strings.Split(strings.TrimSpace(string(raw)), "\n")
	cases := admissionLiveRouteGateSmokeCases()
	if len(lines) != len(cases) {
		t.Fatalf("expected %d route-gate receipts, got %d: %q", len(cases), len(lines), raw)
	}
	var matched, rejected, unknown int
	seenRoutes := make(map[string]bool)
	for i, line := range lines {
		var got dreamCandidate
		if err := json.Unmarshal([]byte(line), &got); err != nil {
			t.Fatalf("receipt %d: %v", i+1, err)
		}
		if got.Admission == nil || got.Admission.LiveRouteChoice == nil {
			t.Fatalf("receipt %d missing live route choice: %+v", i+1, got.Admission)
		}
		choice := got.Admission.LiveRouteChoice
		seenRoutes[choice.ExpectedSource] = true
		if got.Admission.Passed {
			matched++
			continue
		}
		rejected++
		if choice.Reason == "live route plan failed: unknown_prompt_class" {
			unknown++
		}
	}
	if matched != len(admissionLiveRoutePromptClasses()) || rejected != 2 || unknown != 1 {
		t.Fatalf("bad broad smoke receipt counts: matched=%d rejected=%d unknown=%d", matched, rejected, unknown)
	}
	for _, route := range []string{"chorus", "direct", "qloop_hint_qa", "qloop_target", "user_bridge"} {
		if !seenRoutes[route] {
			t.Fatalf("expected route %s missing from receipts; saw %v", route, seenRoutes)
		}
	}
	if !strings.Contains(string(raw), "source direct does not match live route chorus for prompt class identity") {
		t.Fatalf("wrong-source route reason missing from log: %s", raw)
	}
	if !strings.Contains(string(raw), "live route plan failed: unknown_prompt_class") {
		t.Fatalf("unknown-class route reason missing from log: %s", raw)
	}
	for _, trigger := range []string{"user_bridge-cold-reader", "user_bridge-direct-user", "qloop_target-recipient-lock", "qloop_hint_qa-polyphony", "direct-dream", "chorus-identity", "chorus-unknown-pressure"} {
		if !strings.Contains(string(raw), "\"trigger\":\""+trigger+"\"") {
			t.Fatalf("route-prefixed trigger %q missing from log: %s", trigger, raw)
		}
	}
}

func TestAdmissionLiveRouteChatSmokeWritesDryRunReceipt(t *testing.T) {
	t.Setenv("AM_DREAM_ADMISSION", dreamAdmissionShadow)
	t.Setenv("AM_DREAM_ADMISSION_LIVE_ROUTE_CHOICE_DRY_RUN", "1")
	logPath := filepath.Join(t.TempDir(), "live-route-chat.jsonl")
	t.Setenv("AM_DREAM_ADMISSION_LOG", logPath)

	if err := runAdmissionLiveRouteChatSmoke(); err != nil {
		t.Fatal(err)
	}

	raw, err := os.ReadFile(logPath)
	if err != nil {
		t.Fatal(err)
	}
	var got dreamCandidate
	if err := json.Unmarshal([]byte(strings.TrimSpace(string(raw))), &got); err != nil {
		t.Fatal(err)
	}
	if got.Trigger != "chorus-identity" || got.Admission == nil || !got.Admission.Passed ||
		!got.Admission.LiveRouteChoiceDryRun || got.Admission.LiveRouteChoice == nil {
		t.Fatalf("bad chat dry-run receipt: %+v", got)
	}
	choice := got.Admission.LiveRouteChoice
	if !choice.Passed || choice.PromptClass != "identity" || choice.Route != "chorus" ||
		choice.Source != "chorus" || choice.ExpectedSource != "chorus" {
		t.Fatalf("bad chat dry-run route choice: %+v", choice)
	}
}

func TestAdmissionLiveRouteTurnSmokeWritesObservations(t *testing.T) {
	t.Setenv("AM_DREAM_ADMISSION_LIVE_ROUTE_CHOICE_DRY_RUN", "1")
	logPath := filepath.Join(t.TempDir(), "live-route-turn.jsonl")
	t.Setenv("AM_LIVE_ROUTE_TURN_LOG", logPath)

	if err := runAdmissionLiveRouteTurnSmoke(); err != nil {
		t.Fatal(err)
	}

	raw, err := os.ReadFile(logPath)
	if err != nil {
		t.Fatal(err)
	}
	lines := strings.Split(strings.TrimSpace(string(raw)), "\n")
	if len(lines) != 6 {
		t.Fatalf("expected 6 turn observations, got %d: %s", len(lines), raw)
	}
	var identity, unknown admissionLiveRouteTurnObservation
	if err := json.Unmarshal([]byte(lines[0]), &identity); err != nil {
		t.Fatal(err)
	}
	if err := json.Unmarshal([]byte(lines[len(lines)-1]), &unknown); err != nil {
		t.Fatal(err)
	}
	if identity.Schema != admissionLiveRouteTurnObservationSchema || identity.PromptClass != "identity" ||
		identity.Route != "chorus" || identity.ExpectedSource != "chorus" || !identity.Passed {
		t.Fatalf("bad identity turn observation: %+v", identity)
	}
	if unknown.PromptClass != "unknown" || unknown.Passed || unknown.Reason != "live route plan failed: unknown_prompt_class" {
		t.Fatalf("unknown turn should fail closed: %+v", unknown)
	}
}

func TestAdmissionLiveRouteTurnReviewSmokeWritesReviews(t *testing.T) {
	t.Setenv("AM_DREAM_ADMISSION_LIVE_ROUTE_CHOICE_DRY_RUN", "1")
	logPath := filepath.Join(t.TempDir(), "live-route-turn-review.jsonl")
	t.Setenv("AM_LIVE_ROUTE_TURN_REVIEW_LOG", logPath)

	if err := runAdmissionLiveRouteTurnReviewSmoke(); err != nil {
		t.Fatal(err)
	}

	raw, err := os.ReadFile(logPath)
	if err != nil {
		t.Fatal(err)
	}
	lines := strings.Split(strings.TrimSpace(string(raw)), "\n")
	if len(lines) != 4 {
		t.Fatalf("expected 4 turn/candidate reviews, got %d: %s", len(lines), raw)
	}
	var matched, untyped admissionLiveRouteTurnCandidateReview
	if err := json.Unmarshal([]byte(lines[0]), &matched); err != nil {
		t.Fatal(err)
	}
	if err := json.Unmarshal([]byte(lines[2]), &untyped); err != nil {
		t.Fatal(err)
	}
	if matched.Schema != admissionLiveRouteTurnReviewSchema || !matched.Matched ||
		matched.TurnExpectedSource != "chorus" || matched.CandidateSource != "chorus" {
		t.Fatalf("bad matched review: %+v", matched)
	}
	if untyped.Matched || untyped.CandidatePromptClass != "human-turn" ||
		!strings.Contains(untyped.Reason, "unknown_prompt_class") {
		t.Fatalf("bad untyped nano review: %+v", untyped)
	}
}

func TestAdmissionLiveRouteTurnBridgeSmokeWritesBridgeReviews(t *testing.T) {
	t.Setenv("AM_DREAM_ADMISSION_LIVE_ROUTE_CHOICE_DRY_RUN", "1")
	t.Setenv("AM_LIVE_ROUTE_TURN_BRIDGE_DRY_RUN", "1")
	logPath := filepath.Join(t.TempDir(), "live-route-turn-bridge.jsonl")
	t.Setenv("AM_LIVE_ROUTE_TURN_REVIEW_LOG", logPath)

	if err := runAdmissionLiveRouteTurnBridgeSmoke(); err != nil {
		t.Fatal(err)
	}

	raw, err := os.ReadFile(logPath)
	if err != nil {
		t.Fatal(err)
	}
	lines := strings.Split(strings.TrimSpace(string(raw)), "\n")
	if len(lines) != 4 {
		t.Fatalf("expected 4 turn bridge reviews, got %d: %s", len(lines), raw)
	}
	var bridged, matched int
	for i, line := range lines {
		var got admissionLiveRouteTurnCandidateReview
		if err := json.Unmarshal([]byte(line), &got); err != nil {
			t.Fatalf("bridge review %d: %v", i+1, err)
		}
		if got.Matched {
			matched++
		}
		if got.CandidateBridgeApplied {
			bridged++
			if got.CandidateTrigger != "human-turn" ||
				got.CandidateSource != "nano" ||
				!strings.HasPrefix(got.CandidateBridgeTrigger, "human-turn-") ||
				!strings.Contains(got.Reason, "source nano does not match live route") {
				t.Fatalf("bad bridged review %d: %+v", i+1, got)
			}
		}
	}
	if bridged != 2 || matched != 1 {
		t.Fatalf("bad bridge counts: bridged=%d matched=%d log=%s", bridged, matched, raw)
	}
	if !strings.Contains(string(raw), "\"candidate_bridge_trigger\":\"human-turn-identity\"") ||
		!strings.Contains(string(raw), "\"candidate_bridge_trigger\":\"human-turn-direct-user\"") {
		t.Fatalf("bridge triggers missing from log: %s", raw)
	}
}

func TestAdmissionLiveRouteTurnBridgeAdmissionSmokeWritesAdmissionReceipt(t *testing.T) {
	t.Setenv("AM_DREAM_ADMISSION", dreamAdmissionShadow)
	t.Setenv("AM_DREAM_ADMISSION_LIVE_ROUTE_CHOICE_DRY_RUN", "1")
	t.Setenv("AM_LIVE_ROUTE_TURN_BRIDGE_DRY_RUN", "1")
	logPath := filepath.Join(t.TempDir(), "live-route-turn-bridge-admission.jsonl")
	t.Setenv("AM_DREAM_ADMISSION_LOG", logPath)

	if err := runAdmissionLiveRouteTurnBridgeAdmissionSmoke(); err != nil {
		t.Fatal(err)
	}

	raw, err := os.ReadFile(logPath)
	if err != nil {
		t.Fatal(err)
	}
	var got dreamCandidate
	if err := json.Unmarshal([]byte(strings.TrimSpace(string(raw))), &got); err != nil {
		t.Fatal(err)
	}
	if got.Trigger != "human-turn" || got.Source != "nano" ||
		got.Admission == nil || !got.Admission.LiveRouteTurnBridgeApplied ||
		got.Admission.LiveRouteBridgeTrigger != "human-turn-identity" ||
		got.Admission.LiveRouteChoice == nil ||
		got.Admission.LiveRouteChoice.PromptClass != "identity" ||
		got.Admission.LiveRouteChoice.Source != "nano" ||
		got.Admission.LiveRouteChoice.ExpectedSource != "chorus" ||
		got.Admission.LiveRouteChoice.Passed {
		t.Fatalf("bad turn bridge admission receipt: %+v", got)
	}
}
