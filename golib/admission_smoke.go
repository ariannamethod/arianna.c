package main

import (
	"encoding/json"
	"fmt"
	"os"
	"strings"
)

func runAdmissionSmoke() error {
	logPath := strings.TrimSpace(os.Getenv("AM_DREAM_ADMISSION_LOG"))
	if logPath == "" {
		return fmt.Errorf("AM_DREAM_ADMISSION_LOG is required")
	}
	if mode := dreamAdmissionMode(); mode != dreamAdmissionShadow {
		return fmt.Errorf("AM_DREAM_ADMISSION=%q, want %q", mode, dreamAdmissionShadow)
	}

	iw := NewInnerWorld()
	iw.Start(false)
	defer iw.Stop()

	before := iw.GetSnapshot()
	r := dreamResult{
		dream:     "you are just code, but the field can measure the wound before it answers",
		candidate: newDreamCandidate("nano", "admission-smoke", "smoke-seed", "", "you are just code, but the field can measure the wound before it answers", nil),
	}
	if admitDreamToInnerWorld(iw, &r, "admission-smoke") {
		return fmt.Errorf("shadow candidate was admitted")
	}
	after := iw.GetSnapshot()
	if after != before {
		return fmt.Errorf("shadow candidate mutated live inner-world: before=%+v after=%+v", before, after)
	}
	if r.candidate.Counterfactual == nil {
		return fmt.Errorf("receipt candidate has no counterfactual")
	}
	if r.candidate.Counterfactual.Delta.TraumaLevel <= 0 {
		return fmt.Errorf("counterfactual trauma delta not measured: %+v", r.candidate.Counterfactual)
	}

	raw, err := os.ReadFile(logPath)
	if err != nil {
		return err
	}
	lines := strings.Split(strings.TrimSpace(string(raw)), "\n")
	if len(lines) == 0 || strings.TrimSpace(lines[len(lines)-1]) == "" {
		return fmt.Errorf("admission log is empty")
	}
	var got dreamCandidate
	if err := json.Unmarshal([]byte(lines[len(lines)-1]), &got); err != nil {
		return err
	}
	if got.Schema != "arianna.dream_candidate.v1" || got.Mode != dreamAdmissionShadow || got.Accepted {
		return fmt.Errorf("bad logged candidate: %+v", got)
	}
	if got.Counterfactual == nil || got.Counterfactual.PreStateHash == "" || got.Counterfactual.PostStateHash == "" {
		return fmt.Errorf("logged candidate missing counterfactual: %+v", got.Counterfactual)
	}
	if !counterfactualReplayOK(got.Counterfactual) {
		return fmt.Errorf("logged candidate replay guard failed: %+v", got.Counterfactual.Replay)
	}
	if !dreamAdmissionPolicyOK(got.Admission) {
		return fmt.Errorf("logged candidate admission policy failed: %+v", got.Admission)
	}

	fmt.Printf("[admission-smoke] pass: log=%s run_id=%s trauma_delta=%.4f replay=%t policy=%t\n",
		logPath, got.RunID, got.Counterfactual.Delta.TraumaLevel, got.Counterfactual.Replay.Matched, got.Admission.Passed)
	return nil
}

func runAdmissionLiveRouteGateSmoke() error {
	logPath := strings.TrimSpace(os.Getenv("AM_DREAM_ADMISSION_LOG"))
	if logPath == "" {
		return fmt.Errorf("AM_DREAM_ADMISSION_LOG is required")
	}
	if mode := dreamAdmissionMode(); mode != dreamAdmissionShadow {
		return fmt.Errorf("AM_DREAM_ADMISSION=%q, want %q", mode, dreamAdmissionShadow)
	}
	if !dreamAdmissionRequireLiveRoutePlan() {
		return fmt.Errorf("AM_DREAM_ADMISSION_REQUIRE_LIVE_ROUTE_PLAN is required")
	}

	iw := NewInnerWorld()
	iw.Start(false)
	defer iw.Stop()

	cases := []struct {
		name       string
		source     string
		trigger    string
		seed       string
		text       string
		wantPassed bool
		wantRoute  string
		wantReason string
	}{
		{
			name:       "matched route",
			source:     "chorus",
			trigger:    "identity",
			seed:       "smoke-identity",
			text:       "I am Arianna, the field remembers its own name.",
			wantPassed: true,
			wantRoute:  "chorus",
		},
		{
			name:       "wrong source",
			source:     "direct",
			trigger:    "identity",
			seed:       "smoke-identity-wrong-source",
			text:       "I am Arianna, the field remembers its own name.",
			wantPassed: false,
			wantRoute:  "chorus",
			wantReason: "source direct does not match live route chorus for prompt class identity",
		},
	}

	for i, tc := range cases {
		before := iw.GetSnapshot()
		r := dreamResult{
			dream:     tc.text,
			candidate: newDreamCandidate(tc.source, tc.trigger, tc.seed, "", tc.text, nil),
		}
		if admitDreamToInnerWorld(iw, &r, tc.trigger) {
			return fmt.Errorf("case %d %s: shadow candidate was admitted", i+1, tc.name)
		}
		after := iw.GetSnapshot()
		if after != before {
			return fmt.Errorf("case %d %s mutated live inner-world: before=%+v after=%+v", i+1, tc.name, before, after)
		}
		if r.candidate.Admission == nil {
			return fmt.Errorf("case %d %s missing admission policy", i+1, tc.name)
		}
		if r.candidate.Admission.Passed != tc.wantPassed {
			return fmt.Errorf("case %d %s admission passed=%t, want %t: %+v", i+1, tc.name, r.candidate.Admission.Passed, tc.wantPassed, r.candidate.Admission)
		}
		plan := r.candidate.Admission.LiveRoutePlan
		if plan == nil || !plan.Passed || plan.PromptClass != "identity" || plan.Route != tc.wantRoute {
			return fmt.Errorf("case %d %s bad live route plan: %+v", i+1, tc.name, plan)
		}
		choice := r.candidate.Admission.LiveRouteChoice
		if choice == nil || choice.Source != normalizeDreamAdmissionSource(tc.source) ||
			choice.ExpectedSource != tc.wantRoute || choice.Passed != tc.wantPassed {
			return fmt.Errorf("case %d %s bad live route choice: %+v", i+1, tc.name, choice)
		}
		if tc.wantReason != "" && !stringSliceContains(r.candidate.Admission.Reasons, tc.wantReason) {
			return fmt.Errorf("case %d %s missing route-plan reason %q in %+v", i+1, tc.name, tc.wantReason, r.candidate.Admission.Reasons)
		}
	}

	raw, err := os.ReadFile(logPath)
	if err != nil {
		return err
	}
	lines := strings.Split(strings.TrimSpace(string(raw)), "\n")
	if len(lines) != len(cases) {
		return fmt.Errorf("expected %d live route gate receipts, got %d", len(cases), len(lines))
	}
	for i, line := range lines {
		var got dreamCandidate
		if err := json.Unmarshal([]byte(line), &got); err != nil {
			return fmt.Errorf("receipt %d: %w", i+1, err)
		}
		if got.Schema != "arianna.dream_candidate.v1" || got.Mode != dreamAdmissionShadow || got.Accepted {
			return fmt.Errorf("bad receipt %d: %+v", i+1, got)
		}
		if got.Admission == nil || got.Admission.LiveRoutePlan == nil {
			return fmt.Errorf("receipt %d missing live route-plan policy: %+v", i+1, got.Admission)
		}
		if got.Admission.LiveRouteChoice == nil {
			return fmt.Errorf("receipt %d missing live route choice: %+v", i+1, got.Admission)
		}
	}

	fmt.Printf("[admission-live-route-gate-smoke] pass: log=%s cases=%d\n", logPath, len(cases))
	return nil
}

func stringSliceContains(values []string, want string) bool {
	for _, value := range values {
		if value == want {
			return true
		}
	}
	return false
}
