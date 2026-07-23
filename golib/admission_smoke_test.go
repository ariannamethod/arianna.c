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
}
