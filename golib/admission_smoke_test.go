package main

import (
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestAdmissionLiveRouteGateSmokeWritesMatchedAndRejectedReceipts(t *testing.T) {
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
	if len(lines) != 2 {
		t.Fatalf("expected two route-gate receipts, got %d: %q", len(lines), raw)
	}
	var first, second dreamCandidate
	if err := json.Unmarshal([]byte(lines[0]), &first); err != nil {
		t.Fatal(err)
	}
	if err := json.Unmarshal([]byte(lines[1]), &second); err != nil {
		t.Fatal(err)
	}
	if first.Admission == nil || !first.Admission.Passed || first.Admission.LiveRoutePlan == nil || first.Admission.LiveRoutePlan.Route != "chorus" {
		t.Fatalf("matched route receipt did not pass: %+v", first.Admission)
	}
	if second.Admission == nil || second.Admission.Passed || second.Admission.LiveRoutePlan == nil || second.Admission.LiveRoutePlan.Route != "chorus" {
		t.Fatalf("wrong-source route receipt did not fail closed: %+v", second.Admission)
	}
	if !stringSliceContains(second.Admission.Reasons, "source direct does not match live route chorus for prompt class identity") {
		t.Fatalf("wrong-source route reason missing: %+v", second.Admission.Reasons)
	}
}
