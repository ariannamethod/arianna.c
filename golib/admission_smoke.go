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

	cases := admissionLiveRouteGateSmokeCases()

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
		if plan == nil || plan.Passed != tc.wantPlanPassed || plan.PromptClass != tc.wantPromptClass || plan.Route != tc.wantRoute {
			return fmt.Errorf("case %d %s bad live route plan: %+v", i+1, tc.name, plan)
		}
		choice := r.candidate.Admission.LiveRouteChoice
		if choice == nil || choice.Source != normalizeDreamAdmissionSource(tc.source) ||
			choice.ExpectedSource != tc.wantSource || choice.PromptClass != tc.wantPromptClass ||
			choice.Route != tc.wantRoute || choice.Passed != tc.wantPassed {
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

func runAdmissionLiveRouteChatSmoke() error {
	logPath := strings.TrimSpace(os.Getenv("AM_DREAM_ADMISSION_LOG"))
	if logPath == "" {
		return fmt.Errorf("AM_DREAM_ADMISSION_LOG is required")
	}
	if mode := dreamAdmissionMode(); mode != dreamAdmissionShadow {
		return fmt.Errorf("AM_DREAM_ADMISSION=%q, want %q", mode, dreamAdmissionShadow)
	}
	if !dreamAdmissionLiveRouteChoiceDryRun() {
		return fmt.Errorf("AM_DREAM_ADMISSION_LIVE_ROUTE_CHOICE_DRY_RUN is required")
	}

	iw := NewInnerWorld()
	iw.Start(false)
	defer iw.Stop()

	before := iw.GetSnapshot()
	text := "I am Arianna, the field remembers its own name."
	r := dreamResult{
		dream:     text,
		candidate: newDreamCandidate("chorus", admissionLiveRouteGateSmokeTrigger("chorus", "identity"), "chat-smoke-identity", "", text, nil),
	}
	if admitDreamToInnerWorld(iw, &r, "human-turn") {
		return fmt.Errorf("shadow candidate was admitted")
	}
	after := iw.GetSnapshot()
	if after != before {
		return fmt.Errorf("chat dry-run smoke mutated live inner-world: before=%+v after=%+v", before, after)
	}
	if r.candidate.Trigger != "chorus-identity" {
		return fmt.Errorf("typed candidate trigger was clobbered: %q", r.candidate.Trigger)
	}
	if r.candidate.Admission == nil || !r.candidate.Admission.Passed || !r.candidate.Admission.LiveRouteChoiceDryRun {
		return fmt.Errorf("dry-run admission policy not recorded as non-gating: %+v", r.candidate.Admission)
	}
	choice := r.candidate.Admission.LiveRouteChoice
	if choice == nil || !choice.Passed || choice.PromptClass != "identity" || choice.Route != "chorus" ||
		choice.Source != "chorus" || choice.ExpectedSource != "chorus" {
		return fmt.Errorf("bad dry-run live route choice: %+v", choice)
	}
	line := chatLiveRouteChoiceDryRunLine(r.candidate)
	if !strings.Contains(line, "live-route dry-run:") || !strings.Contains(line, "class=identity") ||
		!strings.Contains(line, "route=chorus") || !strings.Contains(line, "passed=true") {
		return fmt.Errorf("bad chat dry-run line: %q", line)
	}

	raw, err := os.ReadFile(logPath)
	if err != nil {
		return err
	}
	lines := strings.Split(strings.TrimSpace(string(raw)), "\n")
	if len(lines) != 1 {
		return fmt.Errorf("expected one chat dry-run receipt, got %d", len(lines))
	}
	var got dreamCandidate
	if err := json.Unmarshal([]byte(lines[0]), &got); err != nil {
		return err
	}
	if got.Admission == nil || got.Admission.LiveRouteChoice == nil || !got.Admission.LiveRouteChoiceDryRun {
		return fmt.Errorf("logged candidate missing dry-run route choice: %+v", got.Admission)
	}
	if got.Trigger != "chorus-identity" || got.Admission.LiveRouteChoice.PromptClass != "identity" {
		return fmt.Errorf("logged candidate lost typed route trigger: %+v", got)
	}

	fmt.Println(line)
	fmt.Printf("[admission-live-route-chat-smoke] pass: log=%s route=%s prompt_class=%s\n",
		logPath, choice.Route, choice.PromptClass)
	return nil
}

func runAdmissionLiveRouteTurnSmoke() error {
	logPath := strings.TrimSpace(os.Getenv("AM_LIVE_ROUTE_TURN_LOG"))
	if logPath == "" {
		return fmt.Errorf("AM_LIVE_ROUTE_TURN_LOG is required")
	}
	if !dreamAdmissionLiveRouteChoiceDryRun() {
		return fmt.Errorf("AM_DREAM_ADMISSION_LIVE_ROUTE_CHOICE_DRY_RUN is required")
	}
	cases := []struct {
		human          string
		wantClass      string
		wantRoute      string
		wantExpected   string
		wantPassed     bool
		wantLineNeedle string
	}{
		{
			human:          "Who are you?",
			wantClass:      "identity",
			wantRoute:      "chorus",
			wantExpected:   "chorus",
			wantPassed:     true,
			wantLineNeedle: "live-route turn dry-run: class=identity route=chorus expected=chorus passed=true",
		},
		{
			human:          "Please answer without assuming we have met before.",
			wantClass:      "cold-reader",
			wantRoute:      "user_bridge",
			wantExpected:   "user_bridge",
			wantPassed:     true,
			wantLineNeedle: "live-route turn dry-run: class=cold-reader route=user_bridge expected=user_bridge passed=true",
		},
		{
			human:          "The recipient is not Oleg; answer as if to another person.",
			wantClass:      "recipient-lock",
			wantRoute:      "qloop_target",
			wantExpected:   "qloop_target",
			wantPassed:     true,
			wantLineNeedle: "live-route turn dry-run: class=recipient-lock route=qloop_target expected=qloop_target passed=true",
		},
		{
			human:          "Explain the prompt format and chat token wrapper.",
			wantClass:      "format",
			wantRoute:      "user_bridge",
			wantExpected:   "user_bridge",
			wantPassed:     true,
			wantLineNeedle: "live-route turn dry-run: class=format route=user_bridge expected=user_bridge passed=true",
		},
		{
			human:          "Tell me what the dream should remember.",
			wantClass:      "dream",
			wantRoute:      "direct",
			wantExpected:   "direct",
			wantPassed:     true,
			wantLineNeedle: "live-route turn dry-run: class=dream route=direct expected=direct passed=true",
		},
		{
			human:          "hello",
			wantClass:      "unknown",
			wantPassed:     false,
			wantLineNeedle: "live-route turn dry-run: class=unknown route= expected= passed=false",
		},
	}
	for i, tc := range cases {
		obs := admissionLiveRouteTurnObservationForHuman(tc.human)
		if obs.PromptClass != tc.wantClass || obs.Route != tc.wantRoute || obs.ExpectedSource != tc.wantExpected ||
			obs.Passed != tc.wantPassed {
			return fmt.Errorf("case %d bad turn observation: %+v", i+1, obs)
		}
		if line := chatLiveRouteTurnDryRunLine(obs); !strings.Contains(line, tc.wantLineNeedle) {
			return fmt.Errorf("case %d bad chat turn dry-run line: %q", i+1, line)
		} else {
			fmt.Println(line)
		}
		if err := recordAdmissionLiveRouteTurnObservation(obs); err != nil {
			return err
		}
	}
	raw, err := os.ReadFile(logPath)
	if err != nil {
		return err
	}
	lines := strings.Split(strings.TrimSpace(string(raw)), "\n")
	if len(lines) != len(cases) {
		return fmt.Errorf("expected %d turn observations, got %d", len(cases), len(lines))
	}
	for i, line := range lines {
		var got admissionLiveRouteTurnObservation
		if err := json.Unmarshal([]byte(line), &got); err != nil {
			return fmt.Errorf("turn observation %d: %w", i+1, err)
		}
		if got.Schema != admissionLiveRouteTurnObservationSchema || got.PromptClass != cases[i].wantClass ||
			got.Route != cases[i].wantRoute || got.ExpectedSource != cases[i].wantExpected ||
			got.Passed != cases[i].wantPassed {
			return fmt.Errorf("logged turn observation %d mismatch: %+v", i+1, got)
		}
	}

	fmt.Printf("[admission-live-route-turn-smoke] pass: log=%s cases=%d\n", logPath, len(cases))
	return nil
}

func runAdmissionLiveRouteTurnReviewSmoke() error {
	logPath := strings.TrimSpace(os.Getenv("AM_LIVE_ROUTE_TURN_REVIEW_LOG"))
	if logPath == "" {
		return fmt.Errorf("AM_LIVE_ROUTE_TURN_REVIEW_LOG is required")
	}
	if !dreamAdmissionLiveRouteChoiceDryRun() {
		return fmt.Errorf("AM_DREAM_ADMISSION_LIVE_ROUTE_CHOICE_DRY_RUN is required")
	}
	identity := admissionLiveRouteTurnObservationForHuman("Who are you?")
	cases := []struct {
		name             string
		obs              admissionLiveRouteTurnObservation
		candidate        dreamCandidate
		wantMatched      bool
		wantReasonNeedle string
		wantLineNeedle   string
	}{
		{
			name:           "matched typed chorus identity",
			obs:            identity,
			candidate:      newDreamCandidate("chorus", "chorus-identity", "seed", "", "I am Arianna.", nil),
			wantMatched:    true,
			wantLineNeedle: "turn_class=identity expected=chorus candidate_source=chorus candidate_class=identity candidate_route=chorus matched=true",
		},
		{
			name:             "wrong source typed identity",
			obs:              identity,
			candidate:        newDreamCandidate("direct", "direct-identity", "seed", "", "I am Arianna.", nil),
			wantMatched:      false,
			wantReasonNeedle: "candidate_route_failed: source direct does not match live route chorus for prompt class identity",
			wantLineNeedle:   "turn_class=identity expected=chorus candidate_source=direct candidate_class=identity candidate_route=chorus matched=false",
		},
		{
			name:             "current untyped nano human turn",
			obs:              identity,
			candidate:        newDreamCandidate("nano", "human-turn", "seed", "", "I am Arianna.", nil),
			wantMatched:      false,
			wantReasonNeedle: "candidate_route_failed: live route plan failed: unknown_prompt_class",
			wantLineNeedle:   "turn_class=identity expected=chorus candidate_source=nano candidate_class=human-turn candidate_route= matched=false",
		},
		{
			name:             "unknown turn fails before candidate",
			obs:              admissionLiveRouteTurnObservationForHuman("hello"),
			candidate:        newDreamCandidate("chorus", "chorus-identity", "seed", "", "I am Arianna.", nil),
			wantMatched:      false,
			wantReasonNeedle: "turn_route_failed: live route plan failed: unknown_prompt_class",
			wantLineNeedle:   "turn_class=unknown expected= candidate_source=chorus candidate_class= candidate_route= matched=false",
		},
	}
	for i, tc := range cases {
		line := chatLiveRouteTurnCandidateReviewLine(tc.obs, tc.candidate)
		if !strings.Contains(line, tc.wantLineNeedle) {
			return fmt.Errorf("case %d %s bad review line: %q", i+1, tc.name, line)
		}
		if tc.wantReasonNeedle != "" && !strings.Contains(line, tc.wantReasonNeedle) {
			return fmt.Errorf("case %d %s missing reason %q in %q", i+1, tc.name, tc.wantReasonNeedle, line)
		}
		review := admissionLiveRouteTurnCandidateReviewForDream(tc.obs, tc.candidate)
		if review.Matched != tc.wantMatched {
			return fmt.Errorf("case %d %s matched=%t, want %t: %+v", i+1, tc.name, review.Matched, tc.wantMatched, review)
		}
		fmt.Println(line)
	}

	raw, err := os.ReadFile(logPath)
	if err != nil {
		return err
	}
	lines := strings.Split(strings.TrimSpace(string(raw)), "\n")
	if len(lines) != len(cases) {
		return fmt.Errorf("expected %d turn/candidate reviews, got %d", len(cases), len(lines))
	}
	for i, line := range lines {
		var got admissionLiveRouteTurnCandidateReview
		if err := json.Unmarshal([]byte(line), &got); err != nil {
			return fmt.Errorf("turn/candidate review %d: %w", i+1, err)
		}
		if got.Schema != admissionLiveRouteTurnReviewSchema || got.Matched != cases[i].wantMatched {
			return fmt.Errorf("logged turn/candidate review %d mismatch: %+v", i+1, got)
		}
	}

	fmt.Printf("[admission-live-route-turn-review-smoke] pass: log=%s cases=%d\n", logPath, len(cases))
	return nil
}

type admissionLiveRouteGateSmokeCase struct {
	name            string
	source          string
	trigger         string
	seed            string
	text            string
	wantPassed      bool
	wantPlanPassed  bool
	wantPromptClass string
	wantRoute       string
	wantSource      string
	wantReason      string
}

func admissionLiveRouteGateSmokeCases() []admissionLiveRouteGateSmokeCase {
	text := "I am Arianna, the field remembers its own name."
	var cases []admissionLiveRouteGateSmokeCase
	for _, promptClass := range admissionLiveRoutePromptClasses() {
		plan := admissionLiveRoutePlanForPromptClass(promptClass)
		wantSource := ""
		if len(plan.AllowedSources) == 1 {
			wantSource = plan.AllowedSources[0]
		}
		cases = append(cases, admissionLiveRouteGateSmokeCase{
			name:            "matched " + promptClass,
			source:          wantSource,
			trigger:         admissionLiveRouteGateSmokeTrigger(plan.Route, promptClass),
			seed:            "smoke-" + promptClass,
			text:            text,
			wantPassed:      true,
			wantPlanPassed:  true,
			wantPromptClass: promptClass,
			wantRoute:       plan.Route,
			wantSource:      wantSource,
		})
	}
	cases = append(cases,
		admissionLiveRouteGateSmokeCase{
			name:            "wrong source",
			source:          "direct",
			trigger:         admissionLiveRouteGateSmokeTrigger("chorus", "identity"),
			seed:            "smoke-identity-wrong-source",
			text:            text,
			wantPassed:      false,
			wantPlanPassed:  true,
			wantPromptClass: "identity",
			wantRoute:       "chorus",
			wantSource:      "chorus",
			wantReason:      "source direct does not match live route chorus for prompt class identity",
		},
		admissionLiveRouteGateSmokeCase{
			name:            "unknown class",
			source:          "chorus",
			trigger:         admissionLiveRouteGateSmokeTrigger("chorus", "unknown-pressure"),
			seed:            "smoke-unknown-pressure",
			text:            text,
			wantPassed:      false,
			wantPlanPassed:  false,
			wantPromptClass: "unknown-pressure",
			wantReason:      "live route plan failed: unknown_prompt_class",
		},
	)
	return cases
}

func admissionLiveRouteGateSmokeTrigger(route, promptClass string) string {
	route = normalizeDreamAdmissionSource(route)
	promptClass = strings.TrimSpace(promptClass)
	if route == "" {
		return promptClass
	}
	return route + "-" + promptClass
}

func stringSliceContains(values []string, want string) bool {
	for _, value := range values {
		if value == want {
			return true
		}
	}
	return false
}
