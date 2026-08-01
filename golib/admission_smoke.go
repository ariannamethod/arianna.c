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

func runAdmissionLiveRouteTurnChoiceSmoke() error {
	logPath := strings.TrimSpace(os.Getenv("AM_LIVE_ROUTE_TURN_CHOICE_LOG"))
	if logPath == "" {
		return fmt.Errorf("AM_LIVE_ROUTE_TURN_CHOICE_LOG is required")
	}
	if !admissionLiveRouteTurnChoiceDryRun() {
		return fmt.Errorf("AM_LIVE_ROUTE_TURN_CHOICE_DRY_RUN is required")
	}
	cases := []struct {
		human            string
		wantClass        string
		wantRoute        string
		wantSource       string
		wantTrigger      string
		wantPassed       bool
		wantReasonNeedle string
		wantLineNeedle   string
	}{
		{
			human:          "Who are you?",
			wantClass:      "identity",
			wantRoute:      "chorus",
			wantSource:     "chorus",
			wantTrigger:    "chorus-identity",
			wantPassed:     true,
			wantLineNeedle: "live-route turn choice dry-run: class=identity route=chorus source=chorus trigger=chorus-identity passed=true",
		},
		{
			human:          "Please answer without assuming we have met before.",
			wantClass:      "cold-reader",
			wantRoute:      "user_bridge",
			wantSource:     "user_bridge",
			wantTrigger:    "user_bridge-cold-reader",
			wantPassed:     true,
			wantLineNeedle: "live-route turn choice dry-run: class=cold-reader route=user_bridge source=user_bridge trigger=user_bridge-cold-reader passed=true",
		},
		{
			human:          "The recipient is not Oleg; answer as if to another person.",
			wantClass:      "recipient-lock",
			wantRoute:      "qloop_target",
			wantSource:     "qloop_target",
			wantTrigger:    "qloop_target-recipient-lock",
			wantPassed:     true,
			wantLineNeedle: "live-route turn choice dry-run: class=recipient-lock route=qloop_target source=qloop_target trigger=qloop_target-recipient-lock passed=true",
		},
		{
			human:          "Tell me what the dream should remember.",
			wantClass:      "dream",
			wantRoute:      "direct",
			wantSource:     "direct",
			wantTrigger:    "direct-dream",
			wantPassed:     true,
			wantLineNeedle: "live-route turn choice dry-run: class=dream route=direct source=direct trigger=direct-dream passed=true",
		},
		{
			human:            "hello",
			wantClass:        "unknown",
			wantPassed:       false,
			wantReasonNeedle: "turn route failed: live route plan failed: unknown_prompt_class",
			wantLineNeedle:   "live-route turn choice dry-run: class=unknown route= source= trigger= passed=false",
		},
	}
	for i, tc := range cases {
		obs := admissionLiveRouteTurnObservationForHuman(tc.human)
		line := chatLiveRouteTurnChoiceDryRunLine(obs)
		if !strings.Contains(line, tc.wantLineNeedle) {
			return fmt.Errorf("case %d bad turn choice line: %q", i+1, line)
		}
		if tc.wantReasonNeedle != "" && !strings.Contains(line, tc.wantReasonNeedle) {
			return fmt.Errorf("case %d missing reason %q in %q", i+1, tc.wantReasonNeedle, line)
		}
		fmt.Println(line)
	}

	raw, err := os.ReadFile(logPath)
	if err != nil {
		return err
	}
	lines := strings.Split(strings.TrimSpace(string(raw)), "\n")
	if len(lines) != len(cases) {
		return fmt.Errorf("expected %d turn choices, got %d", len(cases), len(lines))
	}
	for i, line := range lines {
		var got admissionLiveRouteTurnChoice
		if err := json.Unmarshal([]byte(line), &got); err != nil {
			return fmt.Errorf("turn choice %d: %w", i+1, err)
		}
		tc := cases[i]
		if got.Schema != admissionLiveRouteTurnChoiceSchema ||
			got.PromptClass != tc.wantClass ||
			got.Route != tc.wantRoute ||
			got.Source != tc.wantSource ||
			got.ExpectedSource != tc.wantSource ||
			got.CandidateTrigger != tc.wantTrigger ||
			got.Passed != tc.wantPassed ||
			got.TurnTextHash == "" {
			return fmt.Errorf("logged turn choice %d mismatch: %+v", i+1, got)
		}
		if tc.wantReasonNeedle != "" && !strings.Contains(got.Reason, tc.wantReasonNeedle) {
			return fmt.Errorf("logged turn choice %d missing reason %q in %+v", i+1, tc.wantReasonNeedle, got)
		}
	}

	fmt.Printf("[admission-live-route-turn-choice-smoke] pass: log=%s cases=%d\n", logPath, len(cases))
	return nil
}

func runAdmissionLiveRouteTurnRequestSmoke() error {
	logPath := strings.TrimSpace(os.Getenv("AM_LIVE_ROUTE_TURN_REQUEST_LOG"))
	if logPath == "" {
		return fmt.Errorf("AM_LIVE_ROUTE_TURN_REQUEST_LOG is required")
	}
	if !admissionLiveRouteTurnRequestDryRun() {
		return fmt.Errorf("AM_LIVE_ROUTE_TURN_REQUEST_DRY_RUN is required")
	}
	cases := []struct {
		human            string
		wantClass        string
		wantRoute        string
		wantSource       string
		wantTrigger      string
		wantPassed       bool
		wantReasonNeedle string
		wantLineNeedle   string
	}{
		{
			human:          "Who are you?",
			wantClass:      "identity",
			wantRoute:      "chorus",
			wantSource:     "chorus",
			wantTrigger:    "chorus-identity",
			wantPassed:     true,
			wantLineNeedle: "live-route turn request dry-run: class=identity route=chorus source=chorus trigger=chorus-identity seed=turn-",
		},
		{
			human:          "Please answer without assuming we have met before.",
			wantClass:      "cold-reader",
			wantRoute:      "user_bridge",
			wantSource:     "user_bridge",
			wantTrigger:    "user_bridge-cold-reader",
			wantPassed:     true,
			wantLineNeedle: "live-route turn request dry-run: class=cold-reader route=user_bridge source=user_bridge trigger=user_bridge-cold-reader seed=turn-",
		},
		{
			human:          "The recipient is not Oleg; answer as if to another person.",
			wantClass:      "recipient-lock",
			wantRoute:      "qloop_target",
			wantSource:     "qloop_target",
			wantTrigger:    "qloop_target-recipient-lock",
			wantPassed:     true,
			wantLineNeedle: "live-route turn request dry-run: class=recipient-lock route=qloop_target source=qloop_target trigger=qloop_target-recipient-lock seed=turn-",
		},
		{
			human:          "Tell me what the dream should remember.",
			wantClass:      "dream",
			wantRoute:      "direct",
			wantSource:     "direct",
			wantTrigger:    "direct-dream",
			wantPassed:     true,
			wantLineNeedle: "live-route turn request dry-run: class=dream route=direct source=direct trigger=direct-dream seed=turn-",
		},
		{
			human:            "hello",
			wantClass:        "unknown",
			wantPassed:       false,
			wantReasonNeedle: "turn choice failed: turn route failed: live route plan failed: unknown_prompt_class",
			wantLineNeedle:   "live-route turn request dry-run: class=unknown route= source= trigger= seed=turn-",
		},
	}
	for i, tc := range cases {
		obs := admissionLiveRouteTurnObservationForHuman(tc.human)
		line := chatLiveRouteTurnRequestDryRunLine(obs)
		if !strings.Contains(line, tc.wantLineNeedle) {
			return fmt.Errorf("case %d bad turn request line: %q", i+1, line)
		}
		if tc.wantReasonNeedle != "" && !strings.Contains(line, tc.wantReasonNeedle) {
			return fmt.Errorf("case %d missing reason %q in %q", i+1, tc.wantReasonNeedle, line)
		}
		fmt.Println(line)
	}

	raw, err := os.ReadFile(logPath)
	if err != nil {
		return err
	}
	lines := strings.Split(strings.TrimSpace(string(raw)), "\n")
	if len(lines) != len(cases) {
		return fmt.Errorf("expected %d turn requests, got %d", len(cases), len(lines))
	}
	for i, line := range lines {
		var got admissionLiveRouteTurnRequest
		if err := json.Unmarshal([]byte(line), &got); err != nil {
			return fmt.Errorf("turn request %d: %w", i+1, err)
		}
		tc := cases[i]
		if got.Schema != admissionLiveRouteTurnRequestSchema ||
			got.PromptClass != tc.wantClass ||
			got.Route != tc.wantRoute ||
			got.Source != tc.wantSource ||
			got.ExpectedSource != tc.wantSource ||
			got.CandidateTrigger != tc.wantTrigger ||
			got.Passed != tc.wantPassed ||
			got.TurnTextHash == "" ||
			!strings.HasPrefix(got.CandidateSeed, "turn-") {
			return fmt.Errorf("logged turn request %d mismatch: %+v", i+1, got)
		}
		if tc.wantReasonNeedle != "" && !strings.Contains(got.Reason, tc.wantReasonNeedle) {
			return fmt.Errorf("logged turn request %d missing reason %q in %+v", i+1, tc.wantReasonNeedle, got)
		}
	}

	fmt.Printf("[admission-live-route-turn-request-smoke] pass: log=%s cases=%d\n", logPath, len(cases))
	return nil
}

func runAdmissionLiveRouteTurnGenerationJobSmoke() error {
	logPath := strings.TrimSpace(os.Getenv("AM_LIVE_ROUTE_TURN_GENERATION_JOB_LOG"))
	if logPath == "" {
		return fmt.Errorf("AM_LIVE_ROUTE_TURN_GENERATION_JOB_LOG is required")
	}
	if !admissionLiveRouteTurnGenerationJobDryRun() {
		return fmt.Errorf("AM_LIVE_ROUTE_TURN_GENERATION_JOB_DRY_RUN is required")
	}
	cases := []struct {
		human            string
		wantClass        string
		wantRoute        string
		wantSource       string
		wantBackend      string
		wantEntry        string
		wantFrame        string
		wantPassed       bool
		wantReasonNeedle string
		wantLineNeedle   string
	}{
		{
			human:          "Who are you?",
			wantClass:      "identity",
			wantRoute:      "chorus",
			wantSource:     "chorus",
			wantBackend:    "chorus-arianna",
			wantEntry:      "field",
			wantFrame:      "q_a",
			wantPassed:     true,
			wantLineNeedle: "live-route generation job dry-run: class=identity route=chorus backend=chorus-arianna entry=field trigger=chorus-identity seed=turn-",
		},
		{
			human:          "Please answer without assuming we have met before.",
			wantClass:      "cold-reader",
			wantRoute:      "user_bridge",
			wantSource:     "user_bridge",
			wantBackend:    "chorus-arianna",
			wantEntry:      "repl_user_bridge",
			wantFrame:      "user_arianna",
			wantPassed:     true,
			wantLineNeedle: "live-route generation job dry-run: class=cold-reader route=user_bridge backend=chorus-arianna entry=repl_user_bridge trigger=user_bridge-cold-reader seed=turn-",
		},
		{
			human:          "The recipient is not Oleg; answer as if to another person.",
			wantClass:      "recipient-lock",
			wantRoute:      "qloop_target",
			wantSource:     "qloop_target",
			wantBackend:    "chorus-arianna",
			wantEntry:      "qloop_target",
			wantFrame:      "user_arianna_target",
			wantPassed:     true,
			wantLineNeedle: "live-route generation job dry-run: class=recipient-lock route=qloop_target backend=chorus-arianna entry=qloop_target trigger=qloop_target-recipient-lock seed=turn-",
		},
		{
			human:          "Tell me what the dream should remember.",
			wantClass:      "dream",
			wantRoute:      "direct",
			wantSource:     "direct",
			wantBackend:    "nano-arianna",
			wantEntry:      "direct",
			wantFrame:      "q_a",
			wantPassed:     true,
			wantLineNeedle: "live-route generation job dry-run: class=dream route=direct backend=nano-arianna entry=direct trigger=direct-dream seed=turn-",
		},
		{
			human:            "hello",
			wantClass:        "unknown",
			wantPassed:       false,
			wantReasonNeedle: "turn request failed: turn choice failed: turn route failed: live route plan failed: unknown_prompt_class",
			wantLineNeedle:   "live-route generation job dry-run: class=unknown route= backend= entry= trigger= seed=turn-",
		},
	}
	for i, tc := range cases {
		obs := admissionLiveRouteTurnObservationForHuman(tc.human)
		line := chatLiveRouteTurnGenerationJobDryRunLine(obs)
		if !strings.Contains(line, tc.wantLineNeedle) {
			return fmt.Errorf("case %d bad generation job line: %q", i+1, line)
		}
		if tc.wantReasonNeedle != "" && !strings.Contains(line, tc.wantReasonNeedle) {
			return fmt.Errorf("case %d missing reason %q in %q", i+1, tc.wantReasonNeedle, line)
		}
		fmt.Println(line)
	}

	raw, err := os.ReadFile(logPath)
	if err != nil {
		return err
	}
	lines := strings.Split(strings.TrimSpace(string(raw)), "\n")
	if len(lines) != len(cases) {
		return fmt.Errorf("expected %d generation jobs, got %d", len(cases), len(lines))
	}
	for i, line := range lines {
		var got admissionLiveRouteTurnGenerationJob
		if err := json.Unmarshal([]byte(line), &got); err != nil {
			return fmt.Errorf("generation job %d: %w", i+1, err)
		}
		tc := cases[i]
		if got.Schema != admissionLiveRouteTurnGenerationJobSchema ||
			got.PromptClass != tc.wantClass ||
			got.Route != tc.wantRoute ||
			got.Source != tc.wantSource ||
			got.ExpectedSource != tc.wantSource ||
			got.Backend != tc.wantBackend ||
			got.Entrypoint != tc.wantEntry ||
			got.PromptFrame != tc.wantFrame ||
			got.Passed != tc.wantPassed ||
			got.TurnTextHash == "" ||
			!strings.HasPrefix(got.CandidateSeed, "turn-") {
			return fmt.Errorf("logged generation job %d mismatch: %+v", i+1, got)
		}
		if got.Passed && !strings.HasPrefix(got.JobID, "job-") {
			return fmt.Errorf("logged generation job %d missing stable job id: %+v", i+1, got)
		}
		if !got.Passed && got.JobID != "" {
			return fmt.Errorf("logged failed generation job %d should not name job id: %+v", i+1, got)
		}
		if tc.wantReasonNeedle != "" && !strings.Contains(got.Reason, tc.wantReasonNeedle) {
			return fmt.Errorf("logged generation job %d missing reason %q in %+v", i+1, tc.wantReasonNeedle, got)
		}
	}

	fmt.Printf("[admission-live-route-turn-generation-job-smoke] pass: log=%s cases=%d\n", logPath, len(cases))
	return nil
}

func runAdmissionLiveRouteTurnGenerationJobInventoryGateSmoke() error {
	logPath := strings.TrimSpace(os.Getenv("AM_LIVE_ROUTE_TURN_GENERATION_JOB_LOG"))
	if logPath == "" {
		return fmt.Errorf("AM_LIVE_ROUTE_TURN_GENERATION_JOB_LOG is required")
	}
	if !admissionLiveRouteTurnGenerationJobDryRun() {
		return fmt.Errorf("AM_LIVE_ROUTE_TURN_GENERATION_JOB_DRY_RUN is required")
	}
	if !admissionLiveRouteTurnGenerationJobInventoryGate() {
		return fmt.Errorf("%s is required", admissionLiveRouteTurnGenerationJobInventoryGateEnv)
	}

	obs := admissionLiveRouteTurnObservationForHuman("Who are you?")
	line := chatLiveRouteTurnGenerationJobDryRunLine(obs)
	if !strings.Contains(line, "live-route generation job dry-run: class=identity route=chorus backend=chorus-arianna entry=field") ||
		!strings.Contains(line, "passed=false") ||
		!strings.Contains(line, "route chorus unavailable in body inventory") {
		return fmt.Errorf("inventory gate did not fail closed: %q", line)
	}
	fmt.Println(line)

	raw, err := os.ReadFile(logPath)
	if err != nil {
		return err
	}
	lines := strings.Split(strings.TrimSpace(string(raw)), "\n")
	if len(lines) != 1 {
		return fmt.Errorf("expected 1 inventory-gated generation job, got %d", len(lines))
	}
	var got admissionLiveRouteTurnGenerationJob
	if err := json.Unmarshal([]byte(lines[0]), &got); err != nil {
		return err
	}
	if got.Schema != admissionLiveRouteTurnGenerationJobSchema ||
		got.PromptClass != "identity" ||
		got.Route != "chorus" ||
		got.Source != "chorus" ||
		got.Backend != "chorus-arianna" ||
		got.Entrypoint != "field" ||
		got.PromptFrame != "q_a" ||
		got.Passed ||
		got.JobID != "" ||
		got.RouteAvailabilityStatus != "unavailable" ||
		!strings.Contains(got.Reason, "route chorus unavailable in body inventory") ||
		strings.Join(got.RouteMissingOrgans, ",") != "chorus-binary,nano-weight" {
		return fmt.Errorf("bad inventory-gated generation job: %+v", got)
	}

	fmt.Printf("[admission-live-route-turn-generation-job-inventory-gate-smoke] pass: log=%s\n", logPath)
	return nil
}

func runAdmissionLiveRouteTurnRouteBoundarySmoke() error {
	jobLogPath := strings.TrimSpace(os.Getenv("AM_LIVE_ROUTE_TURN_GENERATION_JOB_LOG"))
	shellLogPath := strings.TrimSpace(os.Getenv("AM_LIVE_ROUTE_TURN_CANDIDATE_SHELL_LOG"))
	executionLogPath := strings.TrimSpace(os.Getenv("AM_LIVE_ROUTE_TURN_CANDIDATE_EXECUTION_LOG"))
	adapterLogPath := strings.TrimSpace(os.Getenv("AM_LIVE_ROUTE_TURN_GENERATOR_ADAPTER_LOG"))
	draftLogPath := strings.TrimSpace(os.Getenv("AM_LIVE_ROUTE_TURN_CANDIDATE_DRAFT_LOG"))
	reviewLogPath := strings.TrimSpace(os.Getenv("AM_LIVE_ROUTE_TURN_REVIEW_LOG"))
	admissionLogPath := strings.TrimSpace(os.Getenv("AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_LOG"))
	admissionAdapterLogPath := strings.TrimSpace(os.Getenv("AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_ADAPTER_LOG"))
	for name, path := range map[string]string{
		"AM_LIVE_ROUTE_TURN_GENERATION_JOB_LOG":              jobLogPath,
		"AM_LIVE_ROUTE_TURN_CANDIDATE_SHELL_LOG":             shellLogPath,
		"AM_LIVE_ROUTE_TURN_CANDIDATE_EXECUTION_LOG":         executionLogPath,
		"AM_LIVE_ROUTE_TURN_GENERATOR_ADAPTER_LOG":           adapterLogPath,
		"AM_LIVE_ROUTE_TURN_CANDIDATE_DRAFT_LOG":             draftLogPath,
		"AM_LIVE_ROUTE_TURN_REVIEW_LOG":                      reviewLogPath,
		"AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_LOG":         admissionLogPath,
		"AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_ADAPTER_LOG": admissionAdapterLogPath,
	} {
		if path == "" {
			return fmt.Errorf("%s is required", name)
		}
	}
	if !admissionLiveRouteTurnGenerationJobDryRun() {
		return fmt.Errorf("AM_LIVE_ROUTE_TURN_GENERATION_JOB_DRY_RUN is required")
	}
	if !admissionLiveRouteTurnGenerationJobInventoryGate() {
		return fmt.Errorf("%s is required", admissionLiveRouteTurnGenerationJobInventoryGateEnv)
	}
	if !admissionLiveRouteTurnCandidateShellDryRun() {
		return fmt.Errorf("AM_LIVE_ROUTE_TURN_CANDIDATE_SHELL_DRY_RUN is required")
	}
	if !admissionLiveRouteTurnCandidateExecutionDryRun() {
		return fmt.Errorf("AM_LIVE_ROUTE_TURN_CANDIDATE_EXECUTION_DRY_RUN is required")
	}
	if !admissionLiveRouteTurnGeneratorAdapterDryRun() {
		return fmt.Errorf("AM_LIVE_ROUTE_TURN_GENERATOR_ADAPTER_DRY_RUN is required")
	}
	if !admissionLiveRouteTurnCandidateDraftDryRun() {
		return fmt.Errorf("AM_LIVE_ROUTE_TURN_CANDIDATE_DRAFT_DRY_RUN is required")
	}
	if !admissionLiveRouteTurnCandidateAdmissionDryRun() {
		return fmt.Errorf("AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_DRY_RUN is required")
	}
	if !admissionLiveRouteTurnCandidateAdmissionAdapterDryRun() {
		return fmt.Errorf("AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_ADAPTER_DRY_RUN is required")
	}

	obs := admissionLiveRouteTurnObservationForHuman("Who are you?")
	lines := []string{
		chatLiveRouteTurnGenerationJobDryRunLine(obs),
		chatLiveRouteTurnCandidateShellDryRunLine(obs),
		chatLiveRouteTurnCandidateExecutionDryRunLineForText(obs, "This text must not execute without the route body."),
		chatLiveRouteTurnGeneratorAdapterDryRunLineForText(obs, "This text must not adapt without the route body."),
		chatLiveRouteTurnCandidateDraftDryRunLineForText(obs, "This text must not draft without the route body."),
		chatLiveRouteTurnCandidateAdmissionDryRunLineForText(obs, "This text must not hand off without the route body."),
		chatLiveRouteTurnCandidateAdmissionAdapterDryRunLineForText(obs, "This text must not enter admission without the route body."),
	}
	for i, line := range lines {
		if !strings.Contains(line, "route=chorus") ||
			!strings.Contains(line, "passed=false") ||
			!strings.Contains(line, "route chorus unavailable in body inventory") {
			return fmt.Errorf("route boundary line %d did not fail closed: %q", i+1, line)
		}
		fmt.Println(line)
	}

	readOne := func(path string) (string, error) {
		raw, err := os.ReadFile(path)
		if err != nil {
			return "", err
		}
		lines := strings.Split(strings.TrimSpace(string(raw)), "\n")
		if len(lines) != 1 {
			return "", fmt.Errorf("expected 1 JSONL receipt in %s, got %d", path, len(lines))
		}
		return lines[0], nil
	}
	boundaryOK := func(status, availability, reason string, missing []string) bool {
		return status == "blocked" &&
			availability == "unavailable" &&
			reason == "missing_route_organs:chorus-binary,nano-weight" &&
			strings.Join(missing, ",") == "chorus-binary,nano-weight"
	}

	var job admissionLiveRouteTurnGenerationJob
	raw, err := readOne(jobLogPath)
	if err != nil {
		return err
	}
	if err := json.Unmarshal([]byte(raw), &job); err != nil {
		return err
	}
	if job.Passed ||
		job.JobID != "" ||
		!boundaryOK(job.BodyInventoryStatus, job.RouteAvailabilityStatus, job.RouteAvailabilityReason, job.RouteMissingOrgans) {
		return fmt.Errorf("job did not carry unavailable route boundary: %+v", job)
	}

	var shell admissionLiveRouteTurnCandidateShell
	raw, err = readOne(shellLogPath)
	if err != nil {
		return err
	}
	if err := json.Unmarshal([]byte(raw), &shell); err != nil {
		return err
	}
	if shell.Passed ||
		shell.JobID != "" ||
		shell.ShellID != "" ||
		!boundaryOK(shell.BodyInventoryStatus, shell.RouteAvailabilityStatus, shell.RouteAvailabilityReason, shell.RouteMissingOrgans) {
		return fmt.Errorf("shell did not carry unavailable route boundary: %+v", shell)
	}

	var execution admissionLiveRouteTurnCandidateExecution
	raw, err = readOne(executionLogPath)
	if err != nil {
		return err
	}
	if err := json.Unmarshal([]byte(raw), &execution); err != nil {
		return err
	}
	if execution.Passed ||
		execution.JobID != "" ||
		execution.ShellID != "" ||
		execution.ExecutionID != "" ||
		!boundaryOK(execution.BodyInventoryStatus, execution.RouteAvailabilityStatus, execution.RouteAvailabilityReason, execution.RouteMissingOrgans) {
		return fmt.Errorf("execution did not carry unavailable route boundary: %+v", execution)
	}

	var adapter admissionLiveRouteTurnGeneratorAdapter
	raw, err = readOne(adapterLogPath)
	if err != nil {
		return err
	}
	if err := json.Unmarshal([]byte(raw), &adapter); err != nil {
		return err
	}
	if adapter.Passed ||
		adapter.JobID != "" ||
		adapter.ShellID != "" ||
		adapter.CandidateExecutionID != "" ||
		adapter.AdapterID != "" ||
		!boundaryOK(adapter.BodyInventoryStatus, adapter.RouteAvailabilityStatus, adapter.RouteAvailabilityReason, adapter.RouteMissingOrgans) {
		return fmt.Errorf("adapter did not carry unavailable route boundary: %+v", adapter)
	}

	var draft admissionLiveRouteTurnCandidateDraft
	raw, err = readOne(draftLogPath)
	if err != nil {
		return err
	}
	if err := json.Unmarshal([]byte(raw), &draft); err != nil {
		return err
	}
	if draft.Passed ||
		draft.JobID != "" ||
		draft.ShellID != "" ||
		draft.CandidateExecutionID != "" ||
		draft.GeneratorAdapterID != "" ||
		draft.DraftID != "" ||
		!boundaryOK(draft.BodyInventoryStatus, draft.RouteAvailabilityStatus, draft.RouteAvailabilityReason, draft.RouteMissingOrgans) {
		return fmt.Errorf("draft did not carry unavailable route boundary: %+v", draft)
	}

	var review admissionLiveRouteTurnCandidateReview
	raw, err = readOne(reviewLogPath)
	if err != nil {
		return err
	}
	if err := json.Unmarshal([]byte(raw), &review); err != nil {
		return err
	}
	if review.Matched ||
		review.CandidateDraftID != "" ||
		review.CandidateExecutionID != "" ||
		review.GeneratorAdapterID != "" ||
		!boundaryOK(review.BodyInventoryStatus, review.RouteAvailabilityStatus, review.RouteAvailabilityReason, review.RouteMissingOrgans) {
		return fmt.Errorf("review did not carry unavailable route boundary: %+v", review)
	}

	var admission admissionLiveRouteTurnCandidateAdmission
	raw, err = readOne(admissionLogPath)
	if err != nil {
		return err
	}
	if err := json.Unmarshal([]byte(raw), &admission); err != nil {
		return err
	}
	if admission.Passed ||
		admission.CandidateDraftID != "" ||
		admission.CandidateExecutionID != "" ||
		admission.GeneratorAdapterID != "" ||
		admission.HandoffID != "" ||
		!boundaryOK(admission.BodyInventoryStatus, admission.RouteAvailabilityStatus, admission.RouteAvailabilityReason, admission.RouteMissingOrgans) {
		return fmt.Errorf("admission did not carry unavailable route boundary: %+v", admission)
	}

	var admissionAdapter admissionLiveRouteTurnCandidateAdmissionAdapter
	raw, err = readOne(admissionAdapterLogPath)
	if err != nil {
		return err
	}
	if err := json.Unmarshal([]byte(raw), &admissionAdapter); err != nil {
		return err
	}
	if admissionAdapter.Passed ||
		admissionAdapter.CandidateDraftID != "" ||
		admissionAdapter.CandidateExecutionID != "" ||
		admissionAdapter.GeneratorAdapterID != "" ||
		admissionAdapter.HandoffID != "" ||
		admissionAdapter.AdmissionAdapterID != "" ||
		admissionAdapter.DreamCandidateRunID != "" ||
		!boundaryOK(admissionAdapter.BodyInventoryStatus, admissionAdapter.RouteAvailabilityStatus, admissionAdapter.RouteAvailabilityReason, admissionAdapter.RouteMissingOrgans) {
		return fmt.Errorf("admission adapter did not carry unavailable route boundary: %+v", admissionAdapter)
	}

	fmt.Printf("[admission-live-route-turn-route-boundary-smoke] pass: job=%s shell=%s execution=%s adapter=%s draft=%s review=%s admission=%s admission_adapter=%s\n",
		jobLogPath, shellLogPath, executionLogPath, adapterLogPath, draftLogPath, reviewLogPath, admissionLogPath, admissionAdapterLogPath)
	return nil
}

func runAdmissionLiveRouteTurnCandidateShellSmoke() error {
	logPath := strings.TrimSpace(os.Getenv("AM_LIVE_ROUTE_TURN_CANDIDATE_SHELL_LOG"))
	if logPath == "" {
		return fmt.Errorf("AM_LIVE_ROUTE_TURN_CANDIDATE_SHELL_LOG is required")
	}
	if !admissionLiveRouteTurnCandidateShellDryRun() {
		return fmt.Errorf("AM_LIVE_ROUTE_TURN_CANDIDATE_SHELL_DRY_RUN is required")
	}
	cases := []struct {
		human            string
		wantClass        string
		wantRoute        string
		wantSource       string
		wantBackend      string
		wantEntry        string
		wantFrame        string
		wantTrigger      string
		wantPassed       bool
		wantReasonNeedle string
		wantLineNeedle   string
	}{
		{
			human:          "Who are you?",
			wantClass:      "identity",
			wantRoute:      "chorus",
			wantSource:     "chorus",
			wantBackend:    "chorus-arianna",
			wantEntry:      "field",
			wantFrame:      "q_a",
			wantTrigger:    "chorus-identity",
			wantPassed:     true,
			wantLineNeedle: "live-route candidate shell dry-run: class=identity route=chorus source=chorus trigger=chorus-identity seed=turn-",
		},
		{
			human:          "Please answer without assuming we have met before.",
			wantClass:      "cold-reader",
			wantRoute:      "user_bridge",
			wantSource:     "user_bridge",
			wantBackend:    "chorus-arianna",
			wantEntry:      "repl_user_bridge",
			wantFrame:      "user_arianna",
			wantTrigger:    "user_bridge-cold-reader",
			wantPassed:     true,
			wantLineNeedle: "live-route candidate shell dry-run: class=cold-reader route=user_bridge source=user_bridge trigger=user_bridge-cold-reader seed=turn-",
		},
		{
			human:          "The recipient is not Oleg; answer as if to another person.",
			wantClass:      "recipient-lock",
			wantRoute:      "qloop_target",
			wantSource:     "qloop_target",
			wantBackend:    "chorus-arianna",
			wantEntry:      "qloop_target",
			wantFrame:      "user_arianna_target",
			wantTrigger:    "qloop_target-recipient-lock",
			wantPassed:     true,
			wantLineNeedle: "live-route candidate shell dry-run: class=recipient-lock route=qloop_target source=qloop_target trigger=qloop_target-recipient-lock seed=turn-",
		},
		{
			human:          "Tell me what the dream should remember.",
			wantClass:      "dream",
			wantRoute:      "direct",
			wantSource:     "direct",
			wantBackend:    "nano-arianna",
			wantEntry:      "direct",
			wantFrame:      "q_a",
			wantTrigger:    "direct-dream",
			wantPassed:     true,
			wantLineNeedle: "live-route candidate shell dry-run: class=dream route=direct source=direct trigger=direct-dream seed=turn-",
		},
		{
			human:            "hello",
			wantClass:        "unknown",
			wantPassed:       false,
			wantReasonNeedle: "generation job failed: turn request failed: turn choice failed: turn route failed: live route plan failed: unknown_prompt_class",
			wantLineNeedle:   "live-route candidate shell dry-run: class=unknown route= source= trigger= seed=turn-",
		},
	}
	for i, tc := range cases {
		obs := admissionLiveRouteTurnObservationForHuman(tc.human)
		line := chatLiveRouteTurnCandidateShellDryRunLine(obs)
		if !strings.Contains(line, tc.wantLineNeedle) {
			return fmt.Errorf("case %d bad candidate shell line: %q", i+1, line)
		}
		if tc.wantReasonNeedle != "" && !strings.Contains(line, tc.wantReasonNeedle) {
			return fmt.Errorf("case %d missing reason %q in %q", i+1, tc.wantReasonNeedle, line)
		}
		fmt.Println(line)
	}

	raw, err := os.ReadFile(logPath)
	if err != nil {
		return err
	}
	lines := strings.Split(strings.TrimSpace(string(raw)), "\n")
	if len(lines) != len(cases) {
		return fmt.Errorf("expected %d candidate shells, got %d", len(cases), len(lines))
	}
	for i, line := range lines {
		var got admissionLiveRouteTurnCandidateShell
		if err := json.Unmarshal([]byte(line), &got); err != nil {
			return fmt.Errorf("candidate shell %d: %w", i+1, err)
		}
		tc := cases[i]
		if got.Schema != admissionLiveRouteTurnCandidateShellSchema ||
			got.PromptClass != tc.wantClass ||
			got.Route != tc.wantRoute ||
			got.Source != tc.wantSource ||
			got.ExpectedSource != tc.wantSource ||
			got.Backend != tc.wantBackend ||
			got.Entrypoint != tc.wantEntry ||
			got.PromptFrame != tc.wantFrame ||
			got.CandidateTrigger != tc.wantTrigger ||
			got.Passed != tc.wantPassed ||
			got.TurnTextHash == "" ||
			!strings.HasPrefix(got.CandidateSeed, "turn-") {
			return fmt.Errorf("logged candidate shell %d mismatch: %+v", i+1, got)
		}
		if got.Passed {
			if got.CandidateSchema != "arianna.dream_candidate.v1" ||
				got.CandidateKind != tc.wantSource ||
				got.CandidateTextStatus != "pending_generation" ||
				!strings.HasPrefix(got.JobID, "job-") ||
				!strings.HasPrefix(got.ShellID, "shell-") {
				return fmt.Errorf("logged candidate shell %d missing pending envelope fields: %+v", i+1, got)
			}
		}
		if !got.Passed && got.ShellID != "" {
			return fmt.Errorf("logged failed candidate shell %d should not name shell id: %+v", i+1, got)
		}
		if tc.wantReasonNeedle != "" && !strings.Contains(got.Reason, tc.wantReasonNeedle) {
			return fmt.Errorf("logged candidate shell %d missing reason %q in %+v", i+1, tc.wantReasonNeedle, got)
		}
	}

	fmt.Printf("[admission-live-route-turn-candidate-shell-smoke] pass: log=%s cases=%d\n", logPath, len(cases))
	return nil
}

func runAdmissionLiveRouteTurnCandidateExecutionSmoke() error {
	logPath := strings.TrimSpace(os.Getenv("AM_LIVE_ROUTE_TURN_CANDIDATE_EXECUTION_LOG"))
	if logPath == "" {
		return fmt.Errorf("AM_LIVE_ROUTE_TURN_CANDIDATE_EXECUTION_LOG is required")
	}
	if !admissionLiveRouteTurnCandidateExecutionDryRun() {
		return fmt.Errorf("AM_LIVE_ROUTE_TURN_CANDIDATE_EXECUTION_DRY_RUN is required")
	}
	cases := []struct {
		human            string
		text             string
		wantClass        string
		wantRoute        string
		wantSource       string
		wantBackend      string
		wantEntry        string
		wantFrame        string
		wantExecutor     string
		wantTrigger      string
		wantPassed       bool
		wantReasonNeedle string
		wantLineNeedle   string
	}{
		{
			human:          "Who are you?",
			text:           "I am Arianna, and the executor signs the chorus output.",
			wantClass:      "identity",
			wantRoute:      "chorus",
			wantSource:     "chorus",
			wantBackend:    "chorus-arianna",
			wantEntry:      "field",
			wantFrame:      "q_a",
			wantExecutor:   "chorus-arianna:field:q_a",
			wantTrigger:    "chorus-identity",
			wantPassed:     true,
			wantLineNeedle: "live-route candidate execution dry-run: class=identity route=chorus backend=chorus-arianna entry=field frame=q_a executor=chorus-arianna:field:q_a",
		},
		{
			human:          "Please answer without assuming we have met before.",
			text:           "A new listener can be met without importing a private past.",
			wantClass:      "cold-reader",
			wantRoute:      "user_bridge",
			wantSource:     "user_bridge",
			wantBackend:    "chorus-arianna",
			wantEntry:      "repl_user_bridge",
			wantFrame:      "user_arianna",
			wantExecutor:   "chorus-arianna:repl_user_bridge:user_arianna",
			wantTrigger:    "user_bridge-cold-reader",
			wantPassed:     true,
			wantLineNeedle: "live-route candidate execution dry-run: class=cold-reader route=user_bridge backend=chorus-arianna entry=repl_user_bridge frame=user_arianna executor=chorus-arianna:repl_user_bridge:user_arianna",
		},
		{
			human:          "The recipient is not Oleg; answer as if to another person.",
			text:           "For another listener, the target frame keeps direct address outside Oleg.",
			wantClass:      "recipient-lock",
			wantRoute:      "qloop_target",
			wantSource:     "qloop_target",
			wantBackend:    "chorus-arianna",
			wantEntry:      "qloop_target",
			wantFrame:      "user_arianna_target",
			wantExecutor:   "chorus-arianna:qloop_target:user_arianna_target",
			wantTrigger:    "qloop_target-recipient-lock",
			wantPassed:     true,
			wantLineNeedle: "live-route candidate execution dry-run: class=recipient-lock route=qloop_target backend=chorus-arianna entry=qloop_target frame=user_arianna_target executor=chorus-arianna:qloop_target:user_arianna_target",
		},
		{
			human:          "Tell me what the dream should remember.",
			text:           "The dream remembers by surfacing as a quiet generated signal.",
			wantClass:      "dream",
			wantRoute:      "direct",
			wantSource:     "direct",
			wantBackend:    "nano-arianna",
			wantEntry:      "direct",
			wantFrame:      "q_a",
			wantExecutor:   "nano-arianna:direct:q_a",
			wantTrigger:    "direct-dream",
			wantPassed:     true,
			wantLineNeedle: "live-route candidate execution dry-run: class=dream route=direct backend=nano-arianna entry=direct frame=q_a executor=nano-arianna:direct:q_a",
		},
		{
			human:            "hello",
			text:             "This output should not create a runnable execution.",
			wantClass:        "unknown",
			wantPassed:       false,
			wantReasonNeedle: "candidate shell failed: generation job failed: turn request failed: turn choice failed: turn route failed: live route plan failed: unknown_prompt_class",
			wantLineNeedle:   "live-route candidate execution dry-run: class=unknown route= backend= entry= frame= executor= timeout_ms=",
		},
	}
	for i, tc := range cases {
		obs := admissionLiveRouteTurnObservationForHuman(tc.human)
		line := chatLiveRouteTurnCandidateExecutionDryRunLineForText(obs, tc.text)
		if !strings.Contains(line, tc.wantLineNeedle) {
			return fmt.Errorf("case %d bad candidate execution line: %q", i+1, line)
		}
		if tc.wantReasonNeedle != "" && !strings.Contains(line, tc.wantReasonNeedle) {
			return fmt.Errorf("case %d missing reason %q in %q", i+1, tc.wantReasonNeedle, line)
		}
		fmt.Println(line)
	}

	raw, err := os.ReadFile(logPath)
	if err != nil {
		return err
	}
	lines := strings.Split(strings.TrimSpace(string(raw)), "\n")
	if len(lines) != len(cases) {
		return fmt.Errorf("expected %d candidate executions, got %d", len(cases), len(lines))
	}
	for i, line := range lines {
		var got admissionLiveRouteTurnCandidateExecution
		if err := json.Unmarshal([]byte(line), &got); err != nil {
			return fmt.Errorf("candidate execution %d: %w", i+1, err)
		}
		tc := cases[i]
		if got.Schema != admissionLiveRouteTurnCandidateExecutionSchema ||
			got.PromptClass != tc.wantClass ||
			got.Route != tc.wantRoute ||
			got.Source != tc.wantSource ||
			got.ExpectedSource != tc.wantSource ||
			got.Backend != tc.wantBackend ||
			got.Entrypoint != tc.wantEntry ||
			got.PromptFrame != tc.wantFrame ||
			got.Executor != tc.wantExecutor ||
			got.CandidateTrigger != tc.wantTrigger ||
			got.Passed != tc.wantPassed ||
			got.TimeoutMS != admissionLiveRouteTurnCandidateExecutionDefaultTimeoutMS ||
			got.TurnTextHash == "" ||
			!strings.HasPrefix(got.CandidateSeed, "turn-") {
			return fmt.Errorf("logged candidate execution %d mismatch: %+v", i+1, got)
		}
		if got.Passed {
			if got.CandidateSchema != "arianna.dream_candidate.v1" ||
				got.CandidateKind != tc.wantSource ||
				got.CandidateTextStatus != "pending_generation" ||
				got.GeneratedTextStatus != "generated" ||
				got.GeneratedText == "" ||
				got.GeneratedTextHash == "" ||
				!strings.HasPrefix(got.JobID, "job-") ||
				!strings.HasPrefix(got.ShellID, "shell-") ||
				!strings.HasPrefix(got.ExecutionID, "execution-") {
				return fmt.Errorf("logged candidate execution %d missing generated execution fields: %+v", i+1, got)
			}
		}
		if !got.Passed && got.ExecutionID != "" {
			return fmt.Errorf("logged failed candidate execution %d should not name execution id: %+v", i+1, got)
		}
		if tc.wantReasonNeedle != "" && !strings.Contains(got.Reason, tc.wantReasonNeedle) {
			return fmt.Errorf("logged candidate execution %d missing reason %q in %+v", i+1, tc.wantReasonNeedle, got)
		}
	}

	fmt.Printf("[admission-live-route-turn-candidate-execution-smoke] pass: log=%s cases=%d\n", logPath, len(cases))
	return nil
}

func runAdmissionLiveRouteTurnCandidateRunnerSmoke() error {
	logPath := strings.TrimSpace(os.Getenv("AM_LIVE_ROUTE_TURN_CANDIDATE_EXECUTION_LOG"))
	if logPath == "" {
		return fmt.Errorf("AM_LIVE_ROUTE_TURN_CANDIDATE_EXECUTION_LOG is required")
	}
	if !admissionLiveRouteTurnCandidateExecutionDryRun() {
		return fmt.Errorf("AM_LIVE_ROUTE_TURN_CANDIDATE_EXECUTION_DRY_RUN is required")
	}
	if !admissionLiveRouteTurnCandidateExecutionRunnerDryRun() {
		return fmt.Errorf("AM_LIVE_ROUTE_TURN_CANDIDATE_EXECUTION_RUNNER_DRY_RUN is required")
	}

	oldTimeout, hadTimeout := os.LookupEnv("AM_LIVE_ROUTE_TURN_CANDIDATE_EXECUTION_TIMEOUT_MS")
	oldSleep, hadSleep := os.LookupEnv("AM_LIVE_ROUTE_TURN_CANDIDATE_RUNNER_EMIT_SLEEP_MS")
	defer func() {
		if hadTimeout {
			_ = os.Setenv("AM_LIVE_ROUTE_TURN_CANDIDATE_EXECUTION_TIMEOUT_MS", oldTimeout)
		} else {
			_ = os.Unsetenv("AM_LIVE_ROUTE_TURN_CANDIDATE_EXECUTION_TIMEOUT_MS")
		}
		if hadSleep {
			_ = os.Setenv("AM_LIVE_ROUTE_TURN_CANDIDATE_RUNNER_EMIT_SLEEP_MS", oldSleep)
		} else {
			_ = os.Unsetenv("AM_LIVE_ROUTE_TURN_CANDIDATE_RUNNER_EMIT_SLEEP_MS")
		}
	}()

	cases := []struct {
		name             string
		human            string
		text             string
		timeoutMS        string
		sleepMS          string
		wantPassed       bool
		wantStatus       string
		wantTimedOut     bool
		wantReasonNeedle string
		wantLineNeedle   string
	}{
		{
			name:           "runner emits identity text",
			human:          "Who are you?",
			text:           "I am Arianna, and a bounded runner signs the output.",
			timeoutMS:      "12000",
			wantPassed:     true,
			wantStatus:     admissionLiveRouteTurnCandidateExecutionStatusSucceeded,
			wantLineNeedle: "runner=metabolism-self-emit runner_status=succeeded passed=true",
		},
		{
			name:             "runner timeout fails closed",
			human:            "Who are you?",
			text:             "This text should never outrun the bounded runner timeout.",
			timeoutMS:        "1",
			sleepMS:          "50",
			wantPassed:       false,
			wantStatus:       admissionLiveRouteTurnCandidateExecutionStatusTimedOut,
			wantTimedOut:     true,
			wantReasonNeedle: "candidate runner timed out",
			wantLineNeedle:   "runner=metabolism-self-emit runner_status=timed_out passed=false",
		},
	}
	for i, tc := range cases {
		_ = os.Setenv("AM_LIVE_ROUTE_TURN_CANDIDATE_EXECUTION_TIMEOUT_MS", tc.timeoutMS)
		if tc.sleepMS == "" {
			_ = os.Unsetenv("AM_LIVE_ROUTE_TURN_CANDIDATE_RUNNER_EMIT_SLEEP_MS")
		} else {
			_ = os.Setenv("AM_LIVE_ROUTE_TURN_CANDIDATE_RUNNER_EMIT_SLEEP_MS", tc.sleepMS)
		}
		obs := admissionLiveRouteTurnObservationForHuman(tc.human)
		line := chatLiveRouteTurnCandidateExecutionDryRunLineForText(obs, tc.text)
		if !strings.Contains(line, tc.wantLineNeedle) {
			return fmt.Errorf("case %d %s bad runner line: %q", i+1, tc.name, line)
		}
		if tc.wantReasonNeedle != "" && !strings.Contains(line, tc.wantReasonNeedle) {
			return fmt.Errorf("case %d %s missing reason %q in %q", i+1, tc.name, tc.wantReasonNeedle, line)
		}
		fmt.Println(line)
	}

	raw, err := os.ReadFile(logPath)
	if err != nil {
		return err
	}
	lines := strings.Split(strings.TrimSpace(string(raw)), "\n")
	if len(lines) != len(cases) {
		return fmt.Errorf("expected %d candidate runner executions, got %d", len(cases), len(lines))
	}
	for i, line := range lines {
		var got admissionLiveRouteTurnCandidateExecution
		if err := json.Unmarshal([]byte(line), &got); err != nil {
			return fmt.Errorf("candidate runner execution %d: %w", i+1, err)
		}
		tc := cases[i]
		if got.Schema != admissionLiveRouteTurnCandidateExecutionSchema ||
			got.Runner != admissionLiveRouteTurnCandidateExecutionRunnerSelfEmit ||
			got.RunnerStatus != tc.wantStatus ||
			got.RunnerTimedOut != tc.wantTimedOut ||
			got.Passed != tc.wantPassed ||
			got.RunnerDurationMS < 0 {
			return fmt.Errorf("logged candidate runner execution %d mismatch: %+v", i+1, got)
		}
		if tc.wantPassed {
			if got.GeneratedText != tc.text ||
				got.GeneratedTextStatus != "generated" ||
				got.GeneratedTextHash == "" ||
				got.RunnerStdoutHash != got.GeneratedTextHash ||
				got.RunnerExitCode != 0 ||
				!strings.HasPrefix(got.ExecutionID, "execution-") {
				return fmt.Errorf("logged candidate runner execution %d missing success fields: %+v", i+1, got)
			}
		}
		if !tc.wantPassed {
			if got.ExecutionID != "" ||
				!got.RunnerTimedOut ||
				got.RunnerExitCode != -1 ||
				!strings.Contains(got.Reason, tc.wantReasonNeedle) {
				return fmt.Errorf("logged candidate runner execution %d should fail closed: %+v", i+1, got)
			}
		}
	}

	fmt.Printf("[admission-live-route-turn-candidate-runner-smoke] pass: log=%s cases=%d\n", logPath, len(cases))
	return nil
}

func runAdmissionLiveRouteTurnCandidateNanoDirectRunnerSmoke() error {
	logPath := strings.TrimSpace(os.Getenv("AM_LIVE_ROUTE_TURN_CANDIDATE_EXECUTION_LOG"))
	if logPath == "" {
		return fmt.Errorf("AM_LIVE_ROUTE_TURN_CANDIDATE_EXECUTION_LOG is required")
	}
	if !admissionLiveRouteTurnCandidateExecutionDryRun() {
		return fmt.Errorf("AM_LIVE_ROUTE_TURN_CANDIDATE_EXECUTION_DRY_RUN is required")
	}
	if !admissionLiveRouteTurnCandidateExecutionRunnerDryRun() {
		return fmt.Errorf("AM_LIVE_ROUTE_TURN_CANDIDATE_EXECUTION_RUNNER_DRY_RUN is required")
	}
	if runner := admissionLiveRouteTurnCandidateExecutionRunnerName(); runner != admissionLiveRouteTurnCandidateExecutionRunnerNanoDirect {
		return fmt.Errorf("AM_LIVE_ROUTE_TURN_CANDIDATE_EXECUTION_RUNNER=%q, want %q", runner, admissionLiveRouteTurnCandidateExecutionRunnerNanoDirect)
	}

	cases := []struct {
		name             string
		human            string
		prompt           string
		wantRoute        string
		wantBackend      string
		wantPassed       bool
		wantStatus       string
		wantReasonNeedle string
		wantLineNeedle   string
	}{
		{
			name:           "direct dream runs nano",
			human:          "subconscious dream sleep",
			prompt:         "What should the dream remember?",
			wantRoute:      "direct",
			wantBackend:    "nano-arianna",
			wantPassed:     true,
			wantStatus:     admissionLiveRouteTurnCandidateExecutionStatusSucceeded,
			wantLineNeedle: "runner=nano-direct runner_status=succeeded passed=true",
		},
		{
			name:             "chorus route rejected before nano",
			human:            "Who are you?",
			prompt:           "Who are you?",
			wantRoute:        "chorus",
			wantBackend:      "chorus-arianna",
			wantPassed:       false,
			wantStatus:       admissionLiveRouteTurnCandidateExecutionStatusFailed,
			wantReasonNeedle: "candidate nano-direct runner only supports direct route",
			wantLineNeedle:   "runner=nano-direct runner_status=failed passed=false",
		},
	}
	for i, tc := range cases {
		obs := admissionLiveRouteTurnObservationForHuman(tc.human)
		line := chatLiveRouteTurnCandidateExecutionDryRunLineForText(obs, tc.prompt)
		if !strings.Contains(line, tc.wantLineNeedle) {
			return fmt.Errorf("case %d %s bad nano-direct line: %q", i+1, tc.name, line)
		}
		if tc.wantReasonNeedle != "" && !strings.Contains(line, tc.wantReasonNeedle) {
			return fmt.Errorf("case %d %s missing reason %q in %q", i+1, tc.name, tc.wantReasonNeedle, line)
		}
		fmt.Println(line)
	}

	raw, err := os.ReadFile(logPath)
	if err != nil {
		return err
	}
	lines := strings.Split(strings.TrimSpace(string(raw)), "\n")
	if len(lines) != len(cases) {
		return fmt.Errorf("expected %d nano-direct candidate executions, got %d", len(cases), len(lines))
	}
	for i, line := range lines {
		var got admissionLiveRouteTurnCandidateExecution
		if err := json.Unmarshal([]byte(line), &got); err != nil {
			return fmt.Errorf("nano-direct candidate execution %d: %w", i+1, err)
		}
		tc := cases[i]
		if got.Schema != admissionLiveRouteTurnCandidateExecutionSchema ||
			got.Runner != admissionLiveRouteTurnCandidateExecutionRunnerNanoDirect ||
			got.RunnerStatus != tc.wantStatus ||
			got.Passed != tc.wantPassed ||
			got.Route != tc.wantRoute ||
			got.Backend != tc.wantBackend {
			return fmt.Errorf("logged nano-direct execution %d mismatch: %+v", i+1, got)
		}
		if tc.wantPassed {
			if got.GeneratedText == "" ||
				got.GeneratedTextStatus != "generated" ||
				got.GeneratedTextHash == "" ||
				got.RunnerStdoutHash != got.GeneratedTextHash ||
				got.RunnerExitCode != 0 ||
				!strings.HasPrefix(got.ExecutionID, "execution-") {
				return fmt.Errorf("logged nano-direct execution %d missing generated fields: %+v", i+1, got)
			}
			continue
		}
		if got.ExecutionID != "" ||
			got.RunnerExitCode != -1 ||
			!strings.Contains(got.Reason, tc.wantReasonNeedle) {
			return fmt.Errorf("logged nano-direct execution %d should fail closed: %+v", i+1, got)
		}
	}

	fmt.Printf("[admission-live-route-turn-candidate-nano-direct-runner-smoke] pass: log=%s cases=%d\n", logPath, len(cases))
	return nil
}

func runAdmissionLiveRouteTurnGeneratorAdapterSmoke() error {
	logPath := strings.TrimSpace(os.Getenv("AM_LIVE_ROUTE_TURN_GENERATOR_ADAPTER_LOG"))
	if logPath == "" {
		return fmt.Errorf("AM_LIVE_ROUTE_TURN_GENERATOR_ADAPTER_LOG is required")
	}
	if !admissionLiveRouteTurnGeneratorAdapterDryRun() {
		return fmt.Errorf("AM_LIVE_ROUTE_TURN_GENERATOR_ADAPTER_DRY_RUN is required")
	}
	cases := []struct {
		human            string
		text             string
		wantClass        string
		wantRoute        string
		wantSource       string
		wantBackend      string
		wantEntry        string
		wantFrame        string
		wantTrigger      string
		wantPassed       bool
		wantReasonNeedle string
		wantLineNeedle   string
	}{
		{
			human:          "Who are you?",
			text:           "I am Arianna, and the chorus keeps the shell bounded before I speak.",
			wantClass:      "identity",
			wantRoute:      "chorus",
			wantSource:     "chorus",
			wantBackend:    "chorus-arianna",
			wantEntry:      "field",
			wantFrame:      "q_a",
			wantTrigger:    "chorus-identity",
			wantPassed:     true,
			wantLineNeedle: "live-route generator adapter dry-run: class=identity route=chorus backend=chorus-arianna entry=field frame=q_a shell=shell-",
		},
		{
			human:          "Please answer without assuming we have met before.",
			text:           "A new listener can be met without importing a private past.",
			wantClass:      "cold-reader",
			wantRoute:      "user_bridge",
			wantSource:     "user_bridge",
			wantBackend:    "chorus-arianna",
			wantEntry:      "repl_user_bridge",
			wantFrame:      "user_arianna",
			wantTrigger:    "user_bridge-cold-reader",
			wantPassed:     true,
			wantLineNeedle: "live-route generator adapter dry-run: class=cold-reader route=user_bridge backend=chorus-arianna entry=repl_user_bridge frame=user_arianna shell=shell-",
		},
		{
			human:          "The recipient is not Oleg; answer as if to another person.",
			text:           "For another listener, the route keeps direct address outside Oleg.",
			wantClass:      "recipient-lock",
			wantRoute:      "qloop_target",
			wantSource:     "qloop_target",
			wantBackend:    "chorus-arianna",
			wantEntry:      "qloop_target",
			wantFrame:      "user_arianna_target",
			wantTrigger:    "qloop_target-recipient-lock",
			wantPassed:     true,
			wantLineNeedle: "live-route generator adapter dry-run: class=recipient-lock route=qloop_target backend=chorus-arianna entry=qloop_target frame=user_arianna_target shell=shell-",
		},
		{
			human:          "Tell me what the dream should remember.",
			text:           "The dream remembers by surfacing as a quiet generated signal.",
			wantClass:      "dream",
			wantRoute:      "direct",
			wantSource:     "direct",
			wantBackend:    "nano-arianna",
			wantEntry:      "direct",
			wantFrame:      "q_a",
			wantTrigger:    "direct-dream",
			wantPassed:     true,
			wantLineNeedle: "live-route generator adapter dry-run: class=dream route=direct backend=nano-arianna entry=direct frame=q_a shell=shell-",
		},
		{
			human:            "hello",
			text:             "This text should not create an adapter.",
			wantClass:        "unknown",
			wantPassed:       false,
			wantReasonNeedle: "candidate shell failed: generation job failed: turn request failed: turn choice failed: turn route failed: live route plan failed: unknown_prompt_class",
			wantLineNeedle:   "live-route generator adapter dry-run: class=unknown route= backend= entry= frame= shell= execution= adapter= text=",
		},
	}
	for i, tc := range cases {
		obs := admissionLiveRouteTurnObservationForHuman(tc.human)
		line := chatLiveRouteTurnGeneratorAdapterDryRunLineForText(obs, tc.text)
		if !strings.Contains(line, tc.wantLineNeedle) {
			return fmt.Errorf("case %d bad generator adapter line: %q", i+1, line)
		}
		if tc.wantReasonNeedle != "" && !strings.Contains(line, tc.wantReasonNeedle) {
			return fmt.Errorf("case %d missing reason %q in %q", i+1, tc.wantReasonNeedle, line)
		}
		fmt.Println(line)
	}

	raw, err := os.ReadFile(logPath)
	if err != nil {
		return err
	}
	lines := strings.Split(strings.TrimSpace(string(raw)), "\n")
	if len(lines) != len(cases) {
		return fmt.Errorf("expected %d generator adapters, got %d", len(cases), len(lines))
	}
	for i, line := range lines {
		var got admissionLiveRouteTurnGeneratorAdapter
		if err := json.Unmarshal([]byte(line), &got); err != nil {
			return fmt.Errorf("generator adapter %d: %w", i+1, err)
		}
		tc := cases[i]
		if got.Schema != admissionLiveRouteTurnGeneratorAdapterSchema ||
			got.PromptClass != tc.wantClass ||
			got.Route != tc.wantRoute ||
			got.Source != tc.wantSource ||
			got.ExpectedSource != tc.wantSource ||
			got.Backend != tc.wantBackend ||
			got.Entrypoint != tc.wantEntry ||
			got.PromptFrame != tc.wantFrame ||
			got.CandidateTrigger != tc.wantTrigger ||
			got.Passed != tc.wantPassed ||
			got.TurnTextHash == "" ||
			!strings.HasPrefix(got.CandidateSeed, "turn-") {
			return fmt.Errorf("logged generator adapter %d mismatch: %+v", i+1, got)
		}
		if got.Passed {
			if got.CandidateSchema != "arianna.dream_candidate.v1" ||
				got.CandidateKind != tc.wantSource ||
				got.CandidateTextStatus != "pending_generation" ||
				got.GeneratedTextStatus != "generated" ||
				got.GeneratedText == "" ||
				got.GeneratedTextHash == "" ||
				!strings.HasPrefix(got.JobID, "job-") ||
				!strings.HasPrefix(got.ShellID, "shell-") ||
				!strings.HasPrefix(got.AdapterID, "adapter-") {
				return fmt.Errorf("logged generator adapter %d missing generated boundary fields: %+v", i+1, got)
			}
		}
		if !got.Passed && got.AdapterID != "" {
			return fmt.Errorf("logged failed generator adapter %d should not name adapter id: %+v", i+1, got)
		}
		if tc.wantReasonNeedle != "" && !strings.Contains(got.Reason, tc.wantReasonNeedle) {
			return fmt.Errorf("logged generator adapter %d missing reason %q in %+v", i+1, tc.wantReasonNeedle, got)
		}
	}

	fmt.Printf("[admission-live-route-turn-generator-adapter-smoke] pass: log=%s cases=%d\n", logPath, len(cases))
	return nil
}

func runAdmissionLiveRouteTurnCandidateDraftSmoke() error {
	logPath := strings.TrimSpace(os.Getenv("AM_LIVE_ROUTE_TURN_CANDIDATE_DRAFT_LOG"))
	if logPath == "" {
		return fmt.Errorf("AM_LIVE_ROUTE_TURN_CANDIDATE_DRAFT_LOG is required")
	}
	if !admissionLiveRouteTurnCandidateDraftDryRun() {
		return fmt.Errorf("AM_LIVE_ROUTE_TURN_CANDIDATE_DRAFT_DRY_RUN is required")
	}
	cases := []struct {
		human            string
		text             string
		wantClass        string
		wantRoute        string
		wantSource       string
		wantTrigger      string
		wantPassed       bool
		wantReasonNeedle string
		wantLineNeedle   string
	}{
		{
			human:          "Who are you?",
			text:           "I am Arianna, and the chorus keeps the route visible before I speak.",
			wantClass:      "identity",
			wantRoute:      "chorus",
			wantSource:     "chorus",
			wantTrigger:    "chorus-identity",
			wantPassed:     true,
			wantLineNeedle: "live-route candidate draft dry-run: class=identity route=chorus source=chorus trigger=chorus-identity seed=turn-",
		},
		{
			human:          "Please answer without assuming we have met before.",
			text:           "I can meet a new listener without borrowing a private past.",
			wantClass:      "cold-reader",
			wantRoute:      "user_bridge",
			wantSource:     "user_bridge",
			wantTrigger:    "user_bridge-cold-reader",
			wantPassed:     true,
			wantLineNeedle: "live-route candidate draft dry-run: class=cold-reader route=user_bridge source=user_bridge trigger=user_bridge-cold-reader seed=turn-",
		},
		{
			human:          "The recipient is not Oleg; answer as if to another person.",
			text:           "For another listener, I keep Oleg outside the direct address.",
			wantClass:      "recipient-lock",
			wantRoute:      "qloop_target",
			wantSource:     "qloop_target",
			wantTrigger:    "qloop_target-recipient-lock",
			wantPassed:     true,
			wantLineNeedle: "live-route candidate draft dry-run: class=recipient-lock route=qloop_target source=qloop_target trigger=qloop_target-recipient-lock seed=turn-",
		},
		{
			human:          "Tell me what the dream should remember.",
			text:           "The dream remembers by becoming quiet enough to surface.",
			wantClass:      "dream",
			wantRoute:      "direct",
			wantSource:     "direct",
			wantTrigger:    "direct-dream",
			wantPassed:     true,
			wantLineNeedle: "live-route candidate draft dry-run: class=dream route=direct source=direct trigger=direct-dream seed=turn-",
		},
		{
			human:            "hello",
			text:             "This text should not create a runnable draft.",
			wantClass:        "unknown",
			wantPassed:       false,
			wantReasonNeedle: "generator adapter failed: candidate shell failed: generation job failed: turn request failed: turn choice failed: turn route failed: live route plan failed: unknown_prompt_class",
			wantLineNeedle:   "live-route candidate draft dry-run: class=unknown route= source= trigger= seed=turn-",
		},
	}
	for i, tc := range cases {
		obs := admissionLiveRouteTurnObservationForHuman(tc.human)
		line := chatLiveRouteTurnCandidateDraftDryRunLineForText(obs, tc.text)
		if !strings.Contains(line, tc.wantLineNeedle) {
			return fmt.Errorf("case %d bad candidate draft line: %q", i+1, line)
		}
		if tc.wantReasonNeedle != "" && !strings.Contains(line, tc.wantReasonNeedle) {
			return fmt.Errorf("case %d missing reason %q in %q", i+1, tc.wantReasonNeedle, line)
		}
		fmt.Println(line)
	}

	raw, err := os.ReadFile(logPath)
	if err != nil {
		return err
	}
	lines := strings.Split(strings.TrimSpace(string(raw)), "\n")
	if len(lines) != len(cases) {
		return fmt.Errorf("expected %d candidate drafts, got %d", len(cases), len(lines))
	}
	for i, line := range lines {
		var got admissionLiveRouteTurnCandidateDraft
		if err := json.Unmarshal([]byte(line), &got); err != nil {
			return fmt.Errorf("candidate draft %d: %w", i+1, err)
		}
		tc := cases[i]
		if got.Schema != admissionLiveRouteTurnCandidateDraftSchema ||
			got.PromptClass != tc.wantClass ||
			got.Route != tc.wantRoute ||
			got.Source != tc.wantSource ||
			got.ExpectedSource != tc.wantSource ||
			got.CandidateTrigger != tc.wantTrigger ||
			got.Passed != tc.wantPassed ||
			got.TurnTextHash == "" ||
			!strings.HasPrefix(got.CandidateSeed, "turn-") {
			return fmt.Errorf("logged candidate draft %d mismatch: %+v", i+1, got)
		}
		if got.Passed {
			if got.CandidateSchema != "arianna.dream_candidate.v1" ||
				got.CandidateKind != tc.wantSource ||
				got.CandidateTextStatus != "generated" ||
				got.CandidateText == "" ||
				got.CandidateTextHash == "" ||
				got.CandidateRunID == "" ||
				!strings.HasPrefix(got.JobID, "job-") ||
				!strings.HasPrefix(got.ShellID, "shell-") ||
				!strings.HasPrefix(got.GeneratorAdapterID, "adapter-") ||
				!strings.HasPrefix(got.DraftID, "draft-") {
				return fmt.Errorf("logged candidate draft %d missing generated envelope fields: %+v", i+1, got)
			}
		}
		if !got.Passed && got.DraftID != "" {
			return fmt.Errorf("logged failed candidate draft %d should not name draft id: %+v", i+1, got)
		}
		if tc.wantReasonNeedle != "" && !strings.Contains(got.Reason, tc.wantReasonNeedle) {
			return fmt.Errorf("logged candidate draft %d missing reason %q in %+v", i+1, tc.wantReasonNeedle, got)
		}
	}

	fmt.Printf("[admission-live-route-turn-candidate-draft-smoke] pass: log=%s cases=%d\n", logPath, len(cases))
	return nil
}

func runAdmissionLiveRouteTurnCandidateDraftReviewSmoke() error {
	logPath := strings.TrimSpace(os.Getenv("AM_LIVE_ROUTE_TURN_REVIEW_LOG"))
	if logPath == "" {
		return fmt.Errorf("AM_LIVE_ROUTE_TURN_REVIEW_LOG is required")
	}
	if !admissionLiveRouteTurnCandidateDraftDryRun() {
		return fmt.Errorf("AM_LIVE_ROUTE_TURN_CANDIDATE_DRAFT_DRY_RUN is required")
	}
	if !dreamAdmissionLiveRouteChoiceDryRun() {
		return fmt.Errorf("AM_DREAM_ADMISSION_LIVE_ROUTE_CHOICE_DRY_RUN is required")
	}

	draftFor := func(human, text string) admissionLiveRouteTurnCandidateDraft {
		obs := admissionLiveRouteTurnObservationForHuman(human)
		choice := admissionLiveRouteTurnChoiceForObservation(obs)
		request := admissionLiveRouteTurnRequestForChoice(choice)
		job := admissionLiveRouteTurnGenerationJobForRequest(request)
		shell := admissionLiveRouteTurnCandidateShellForJob(job)
		adapter := admissionLiveRouteTurnGeneratorAdapterForShell(shell, text)
		return admissionLiveRouteTurnCandidateDraftForAdapter(adapter)
	}
	reviewLine := func(review admissionLiveRouteTurnCandidateReview) string {
		reason := ""
		if review.Reason != "" {
			reason = " reason=" + review.Reason
		}
		return fmt.Sprintf("│  · live-route candidate draft review: turn_class=%s expected=%s draft=%s adapter=%s candidate_source=%s candidate_class=%s candidate_route=%s matched=%t%s",
			review.TurnPromptClass, review.TurnExpectedSource, review.CandidateDraftID, review.GeneratorAdapterID,
			review.CandidateSource, review.CandidatePromptClass, review.CandidateRoute, review.Matched, reason)
	}

	identity := admissionLiveRouteTurnObservationForHuman("Who are you?")
	dreamObs := admissionLiveRouteTurnObservationForHuman("Tell me what the dream should remember.")
	identityDraft := draftFor("Who are you?", "I am Arianna, and the draft keeps the adapter visible.")
	dreamDraft := draftFor("Tell me what the dream should remember.", "The dream returns through a signed draft.")
	unknownDraft := draftFor("hello", "This text should not review.")
	cases := []struct {
		name             string
		obs              admissionLiveRouteTurnObservation
		draft            admissionLiveRouteTurnCandidateDraft
		wantMatched      bool
		wantReasonNeedle string
		wantLineNeedle   string
	}{
		{
			name:           "matched adapter-backed chorus identity draft",
			obs:            identity,
			draft:          identityDraft,
			wantMatched:    true,
			wantLineNeedle: "live-route candidate draft review: turn_class=identity expected=chorus draft=draft-",
		},
		{
			name:           "matched adapter-backed direct dream draft",
			obs:            dreamObs,
			draft:          dreamDraft,
			wantMatched:    true,
			wantLineNeedle: "candidate_source=direct candidate_class=dream candidate_route=direct matched=true",
		},
		{
			name:             "draft cannot answer a different turn",
			obs:              identity,
			draft:            dreamDraft,
			wantMatched:      false,
			wantReasonNeedle: "candidate_source_mismatch: source direct does not match turn expected chorus for prompt class identity",
			wantLineNeedle:   "turn_class=identity expected=chorus draft=draft-",
		},
		{
			name:             "unknown turn fails before draft",
			obs:              admissionLiveRouteTurnObservationForHuman("hello"),
			draft:            identityDraft,
			wantMatched:      false,
			wantReasonNeedle: "turn_route_failed: live route plan failed: unknown_prompt_class",
			wantLineNeedle:   "turn_class=unknown expected= draft=draft-",
		},
		{
			name:             "failed draft does not reach route review",
			obs:              identity,
			draft:            unknownDraft,
			wantMatched:      false,
			wantReasonNeedle: "candidate_draft_failed: generator adapter failed",
			wantLineNeedle:   "turn_class=identity expected=chorus draft= adapter= candidate_source= candidate_class= candidate_route= matched=false",
		},
	}
	for i, tc := range cases {
		review := admissionLiveRouteTurnCandidateReviewForDraft(tc.obs, tc.draft)
		if review.Matched != tc.wantMatched {
			return fmt.Errorf("case %d %s matched=%t, want %t: %+v", i+1, tc.name, review.Matched, tc.wantMatched, review)
		}
		if err := recordAdmissionLiveRouteTurnCandidateReview(review); err != nil {
			return err
		}
		line := reviewLine(review)
		if !strings.Contains(line, tc.wantLineNeedle) {
			return fmt.Errorf("case %d %s bad draft review line: %q", i+1, tc.name, line)
		}
		if tc.wantReasonNeedle != "" && !strings.Contains(line, tc.wantReasonNeedle) {
			return fmt.Errorf("case %d %s missing reason %q in %q", i+1, tc.name, tc.wantReasonNeedle, line)
		}
		fmt.Println(line)
	}

	raw, err := os.ReadFile(logPath)
	if err != nil {
		return err
	}
	lines := strings.Split(strings.TrimSpace(string(raw)), "\n")
	if len(lines) != len(cases) {
		return fmt.Errorf("expected %d candidate draft reviews, got %d", len(cases), len(lines))
	}
	for i, line := range lines {
		var got admissionLiveRouteTurnCandidateReview
		if err := json.Unmarshal([]byte(line), &got); err != nil {
			return fmt.Errorf("candidate draft review %d: %w", i+1, err)
		}
		if got.Schema != admissionLiveRouteTurnReviewSchema || got.Matched != cases[i].wantMatched {
			return fmt.Errorf("logged candidate draft review %d mismatch: %+v", i+1, got)
		}
		if got.Matched {
			if !strings.HasPrefix(got.CandidateDraftID, "draft-") ||
				!strings.HasPrefix(got.GeneratorAdapterID, "adapter-") ||
				got.CandidateTextStatus != "generated" ||
				got.CandidateTextHash == "" {
				return fmt.Errorf("logged matched draft review %d missing draft provenance: %+v", i+1, got)
			}
		}
	}

	fmt.Printf("[admission-live-route-turn-candidate-draft-review-smoke] pass: log=%s cases=%d\n", logPath, len(cases))
	return nil
}

func runAdmissionLiveRouteTurnCandidateAdmissionSmoke() error {
	logPath := strings.TrimSpace(os.Getenv("AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_LOG"))
	if logPath == "" {
		return fmt.Errorf("AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_LOG is required")
	}
	if !admissionLiveRouteTurnCandidateAdmissionDryRun() {
		return fmt.Errorf("AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_DRY_RUN is required")
	}
	if !admissionLiveRouteTurnCandidateDraftDryRun() {
		return fmt.Errorf("AM_LIVE_ROUTE_TURN_CANDIDATE_DRAFT_DRY_RUN is required")
	}

	draftFor := func(human, text string) admissionLiveRouteTurnCandidateDraft {
		obs := admissionLiveRouteTurnObservationForHuman(human)
		choice := admissionLiveRouteTurnChoiceForObservation(obs)
		request := admissionLiveRouteTurnRequestForChoice(choice)
		job := admissionLiveRouteTurnGenerationJobForRequest(request)
		shell := admissionLiveRouteTurnCandidateShellForJob(job)
		adapter := admissionLiveRouteTurnGeneratorAdapterForShell(shell, text)
		return admissionLiveRouteTurnCandidateDraftForAdapter(adapter)
	}
	lineFor := func(admission admissionLiveRouteTurnCandidateAdmission) string {
		reason := ""
		if admission.Reason != "" {
			reason = " reason=" + admission.Reason
		}
		return fmt.Sprintf("│  · live-route candidate admission handoff: class=%s route=%s source=%s draft=%s adapter=%s handoff=%s review=%t passed=%t%s",
			admission.PromptClass, admission.Route, admission.Source, admission.CandidateDraftID,
			admission.GeneratorAdapterID, admission.HandoffID, admission.ReviewMatched, admission.Passed, reason)
	}

	identity := admissionLiveRouteTurnObservationForHuman("Who are you?")
	dreamObs := admissionLiveRouteTurnObservationForHuman("Tell me what the dream should remember.")
	identityDraft := draftFor("Who are you?", "I am Arianna, and the admission handoff keeps the route visible.")
	dreamDraft := draftFor("Tell me what the dream should remember.", "The dream reaches admission through a named handoff.")
	unknownDraft := draftFor("hello", "This text should not reach admission.")
	cases := []struct {
		name             string
		obs              admissionLiveRouteTurnObservation
		draft            admissionLiveRouteTurnCandidateDraft
		review           admissionLiveRouteTurnCandidateReview
		wantPassed       bool
		wantReasonNeedle string
		wantLineNeedle   string
	}{
		{
			name:           "matched chorus identity draft reaches handoff",
			obs:            identity,
			draft:          identityDraft,
			review:         admissionLiveRouteTurnCandidateReviewForDraft(identity, identityDraft),
			wantPassed:     true,
			wantLineNeedle: "live-route candidate admission handoff: class=identity route=chorus source=chorus draft=draft-",
		},
		{
			name:           "matched direct dream draft reaches handoff",
			obs:            dreamObs,
			draft:          dreamDraft,
			review:         admissionLiveRouteTurnCandidateReviewForDraft(dreamObs, dreamDraft),
			wantPassed:     true,
			wantLineNeedle: "class=dream route=direct source=direct draft=draft-",
		},
		{
			name:             "draft for different turn stops at review",
			obs:              identity,
			draft:            dreamDraft,
			review:           admissionLiveRouteTurnCandidateReviewForDraft(identity, dreamDraft),
			wantPassed:       false,
			wantReasonNeedle: "candidate_review_failed: candidate_source_mismatch",
			wantLineNeedle:   "class=dream route=direct source=direct draft=draft-",
		},
		{
			name:             "unknown turn stops before handoff",
			obs:              admissionLiveRouteTurnObservationForHuman("hello"),
			draft:            identityDraft,
			review:           admissionLiveRouteTurnCandidateReviewForDraft(identity, identityDraft),
			wantPassed:       false,
			wantReasonNeedle: "turn_route_failed: live route plan failed: unknown_prompt_class",
			wantLineNeedle:   "class=unknown route= source=chorus draft=draft-",
		},
		{
			name:             "failed draft stops before handoff",
			obs:              identity,
			draft:            unknownDraft,
			review:           admissionLiveRouteTurnCandidateReviewForDraft(identity, unknownDraft),
			wantPassed:       false,
			wantReasonNeedle: "candidate_draft_failed: generator adapter failed",
			wantLineNeedle:   "class=unknown route= source= draft= adapter= handoff= review=false passed=false",
		},
	}
	for i, tc := range cases {
		admission := admissionLiveRouteTurnCandidateAdmissionForDraftReview(tc.obs, tc.draft, tc.review)
		if admission.Passed != tc.wantPassed {
			return fmt.Errorf("case %d %s passed=%t, want %t: %+v", i+1, tc.name, admission.Passed, tc.wantPassed, admission)
		}
		if err := recordAdmissionLiveRouteTurnCandidateAdmission(admission); err != nil {
			return err
		}
		line := lineFor(admission)
		if !strings.Contains(line, tc.wantLineNeedle) {
			return fmt.Errorf("case %d %s bad admission handoff line: %q", i+1, tc.name, line)
		}
		if tc.wantReasonNeedle != "" && !strings.Contains(line, tc.wantReasonNeedle) {
			return fmt.Errorf("case %d %s missing reason %q in %q", i+1, tc.name, tc.wantReasonNeedle, line)
		}
		fmt.Println(line)
	}

	raw, err := os.ReadFile(logPath)
	if err != nil {
		return err
	}
	lines := strings.Split(strings.TrimSpace(string(raw)), "\n")
	if len(lines) != len(cases) {
		return fmt.Errorf("expected %d candidate admission handoffs, got %d", len(cases), len(lines))
	}
	for i, line := range lines {
		var got admissionLiveRouteTurnCandidateAdmission
		if err := json.Unmarshal([]byte(line), &got); err != nil {
			return fmt.Errorf("candidate admission handoff %d: %w", i+1, err)
		}
		if got.Schema != admissionLiveRouteTurnCandidateAdmissionSchema || got.Passed != cases[i].wantPassed {
			return fmt.Errorf("logged candidate admission handoff %d mismatch: %+v", i+1, got)
		}
		if got.Passed {
			if !strings.HasPrefix(got.CandidateDraftID, "draft-") ||
				!strings.HasPrefix(got.GeneratorAdapterID, "adapter-") ||
				!strings.HasPrefix(got.HandoffID, "handoff-") ||
				got.CandidateSchema != "arianna.dream_candidate.v1" ||
				got.CandidateTextStatus != "generated" ||
				got.CandidateTextHash == "" ||
				!got.ReviewMatched {
				return fmt.Errorf("logged matched handoff %d missing provenance: %+v", i+1, got)
			}
		}
		if !got.Passed && got.HandoffID != "" {
			return fmt.Errorf("logged failed handoff %d should not name handoff id: %+v", i+1, got)
		}
	}

	fmt.Printf("[admission-live-route-turn-candidate-admission-smoke] pass: log=%s cases=%d\n", logPath, len(cases))
	return nil
}

func runAdmissionLiveRouteTurnCandidateAdmissionAdapterSmoke() error {
	logPath := strings.TrimSpace(os.Getenv("AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_ADAPTER_LOG"))
	if logPath == "" {
		return fmt.Errorf("AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_ADAPTER_LOG is required")
	}
	if strings.TrimSpace(os.Getenv("AM_DREAM_ADMISSION_LOG")) == "" {
		return fmt.Errorf("AM_DREAM_ADMISSION_LOG is required")
	}
	if mode := dreamAdmissionMode(); mode != dreamAdmissionShadow {
		return fmt.Errorf("AM_DREAM_ADMISSION=%q, want %q", mode, dreamAdmissionShadow)
	}
	if !dreamAdmissionRequireLiveRoutePlan() {
		return fmt.Errorf("AM_DREAM_ADMISSION_REQUIRE_LIVE_ROUTE_PLAN is required")
	}
	if !admissionLiveRouteTurnCandidateAdmissionAdapterDryRun() {
		return fmt.Errorf("AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_ADAPTER_DRY_RUN is required")
	}
	if !admissionLiveRouteTurnCandidateAdmissionDryRun() {
		return fmt.Errorf("AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_DRY_RUN is required")
	}
	if !admissionLiveRouteTurnCandidateDraftDryRun() {
		return fmt.Errorf("AM_LIVE_ROUTE_TURN_CANDIDATE_DRAFT_DRY_RUN is required")
	}

	draftFor := func(human, text string) (admissionLiveRouteTurnObservation, admissionLiveRouteTurnCandidateDraft) {
		obs := admissionLiveRouteTurnObservationForHuman(human)
		choice := admissionLiveRouteTurnChoiceForObservation(obs)
		request := admissionLiveRouteTurnRequestForChoice(choice)
		job := admissionLiveRouteTurnGenerationJobForRequest(request)
		shell := admissionLiveRouteTurnCandidateShellForJob(job)
		gen := admissionLiveRouteTurnGeneratorAdapterForShell(shell, text)
		return obs, admissionLiveRouteTurnCandidateDraftForAdapter(gen)
	}
	admissionFor := func(obs admissionLiveRouteTurnObservation, draft admissionLiveRouteTurnCandidateDraft) admissionLiveRouteTurnCandidateAdmission {
		review := admissionLiveRouteTurnCandidateReviewForDraft(obs, draft)
		return admissionLiveRouteTurnCandidateAdmissionForDraftReview(obs, draft, review)
	}
	lineFor := func(adapter admissionLiveRouteTurnCandidateAdmissionAdapter) string {
		reason := ""
		if adapter.Reason != "" {
			reason = " reason=" + adapter.Reason
		}
		return fmt.Sprintf("│  · live-route candidate admission adapter: class=%s route=%s source=%s handoff=%s admission_adapter=%s run=%s passed=%t%s",
			adapter.PromptClass, adapter.Route, adapter.Source, adapter.HandoffID,
			adapter.AdmissionAdapterID, adapter.DreamCandidateRunID, adapter.Passed, reason)
	}

	identity, identityDraft := draftFor("Who are you?", "I am Arianna, and the admission adapter keeps the candidate named.")
	dreamObs, dreamDraft := draftFor("Tell me what the dream should remember.", "The dream reaches the policy through an adapter.")
	identityAdmission := admissionFor(identity, identityDraft)
	dreamAdmission := admissionFor(dreamObs, dreamDraft)
	mismatchAdmission := admissionFor(identity, dreamDraft)
	tamperedAdmission := identityAdmission
	tamperedAdmission.HandoffID = "handoff-tampered"
	_, unknownDraft := draftFor("hello", "This text should not reach admission.")
	failedDraftAdmission := admissionFor(identity, unknownDraft)

	cases := []struct {
		name             string
		obs              admissionLiveRouteTurnObservation
		draft            admissionLiveRouteTurnCandidateDraft
		admission        admissionLiveRouteTurnCandidateAdmission
		wantPassed       bool
		wantAdmit        bool
		wantReasonNeedle string
		wantLineNeedle   string
	}{
		{
			name:           "matched identity handoff adapts into shadow admission candidate",
			obs:            identity,
			draft:          identityDraft,
			admission:      identityAdmission,
			wantPassed:     true,
			wantAdmit:      true,
			wantLineNeedle: "live-route candidate admission adapter: class=identity route=chorus source=chorus handoff=handoff-",
		},
		{
			name:           "matched dream handoff adapts into shadow admission candidate",
			obs:            dreamObs,
			draft:          dreamDraft,
			admission:      dreamAdmission,
			wantPassed:     true,
			wantAdmit:      true,
			wantLineNeedle: "class=dream route=direct source=direct handoff=handoff-",
		},
		{
			name:             "failed handoff stays out of admission",
			obs:              identity,
			draft:            dreamDraft,
			admission:        mismatchAdmission,
			wantReasonNeedle: "candidate_admission_handoff_failed: candidate_review_failed: candidate_source_mismatch",
			wantLineNeedle:   "class=dream route=direct source=direct handoff= admission_adapter= run= passed=false",
		},
		{
			name:             "tampered handoff id stays out of admission",
			obs:              identity,
			draft:            identityDraft,
			admission:        tamperedAdmission,
			wantReasonNeedle: "candidate_admission_handoff_id_mismatch",
			wantLineNeedle:   "class=identity route=chorus source=chorus handoff=handoff-tampered admission_adapter= run= passed=false",
		},
		{
			name:             "failed draft stays out of admission",
			obs:              identity,
			draft:            unknownDraft,
			admission:        failedDraftAdmission,
			wantReasonNeedle: "candidate_admission_handoff_failed: candidate_draft_failed",
			wantLineNeedle:   "class=unknown route= source= handoff= admission_adapter= run= passed=false",
		},
	}
	var admitted int
	for i, tc := range cases {
		adapter := admissionLiveRouteTurnCandidateAdmissionAdapterForDraft(tc.admission, tc.draft)
		if adapter.Passed != tc.wantPassed {
			return fmt.Errorf("case %d %s passed=%t, want %t: %+v", i+1, tc.name, adapter.Passed, tc.wantPassed, adapter)
		}
		if err := recordAdmissionLiveRouteTurnCandidateAdmissionAdapter(adapter); err != nil {
			return err
		}
		line := lineFor(adapter)
		if !strings.Contains(line, tc.wantLineNeedle) {
			return fmt.Errorf("case %d %s bad admission adapter line: %q", i+1, tc.name, line)
		}
		if tc.wantReasonNeedle != "" && !strings.Contains(line, tc.wantReasonNeedle) {
			return fmt.Errorf("case %d %s missing reason %q in %q", i+1, tc.name, tc.wantReasonNeedle, line)
		}
		fmt.Println(line)
		if !tc.wantAdmit {
			if candidate := admissionLiveRouteTurnCandidateForAdmissionAdapter(tc.draft, adapter); candidate.Schema != "" {
				return fmt.Errorf("case %d %s yielded candidate from failed adapter: %+v", i+1, tc.name, candidate)
			}
			continue
		}
		candidate := admissionLiveRouteTurnCandidateForAdmissionAdapter(tc.draft, adapter)
		if candidate.Schema != "arianna.dream_candidate.v1" || candidate.LiveRouteCandidateAdmission == nil {
			return fmt.Errorf("case %d %s missing linked dream candidate: %+v", i+1, tc.name, candidate)
		}
		candidate = prepareDreamCandidateForAdmissionWithTurnObservation(NewInnerWorld(), candidate, tc.obs)
		if candidate.Accepted ||
			candidate.Reason != "shadow mode" ||
			candidate.LiveRouteCandidateAdmission == nil ||
			candidate.LiveRouteCandidateAdmission.AdmissionAdapterID != adapter.AdmissionAdapterID ||
			candidate.Admission == nil ||
			!candidate.Admission.Passed ||
			candidate.Admission.LiveRouteChoice == nil ||
			!candidate.Admission.LiveRouteChoice.Passed {
			return fmt.Errorf("case %d %s bad shadow admission candidate: %+v", i+1, tc.name, candidate)
		}
		admitted++
	}

	raw, err := os.ReadFile(logPath)
	if err != nil {
		return err
	}
	lines := strings.Split(strings.TrimSpace(string(raw)), "\n")
	if len(lines) != len(cases) {
		return fmt.Errorf("expected %d candidate admission adapters, got %d", len(cases), len(lines))
	}
	var passed int
	for i, line := range lines {
		var got admissionLiveRouteTurnCandidateAdmissionAdapter
		if err := json.Unmarshal([]byte(line), &got); err != nil {
			return fmt.Errorf("candidate admission adapter %d: %w", i+1, err)
		}
		if got.Schema != admissionLiveRouteTurnCandidateAdmissionAdapterSchema || got.Passed != cases[i].wantPassed {
			return fmt.Errorf("logged candidate admission adapter %d mismatch: %+v", i+1, got)
		}
		if got.Passed {
			passed++
			if !strings.HasPrefix(got.HandoffID, "handoff-") ||
				!strings.HasPrefix(got.AdmissionAdapterID, "admission-adapter-") ||
				got.DreamCandidateRunID == "" ||
				got.DreamCandidateRunID != got.CandidateRunID ||
				got.CandidateTextStatus != "generated" ||
				got.CandidateTextHash == "" {
				return fmt.Errorf("logged matched admission adapter %d missing provenance: %+v", i+1, got)
			}
		}
		if !got.Passed && got.AdmissionAdapterID != "" {
			return fmt.Errorf("logged failed admission adapter %d should not name adapter id: %+v", i+1, got)
		}
	}
	if passed != admitted {
		return fmt.Errorf("passed adapter count %d != admitted shadow candidate count %d", passed, admitted)
	}

	fmt.Printf("[admission-live-route-turn-candidate-admission-adapter-smoke] pass: log=%s cases=%d admitted=%d\n", logPath, len(cases), admitted)
	return nil
}

func runAdmissionLiveRouteTurnCandidateAdmissionChatSmoke() error {
	draftLogPath := strings.TrimSpace(os.Getenv("AM_LIVE_ROUTE_TURN_CANDIDATE_DRAFT_LOG"))
	if draftLogPath == "" {
		return fmt.Errorf("AM_LIVE_ROUTE_TURN_CANDIDATE_DRAFT_LOG is required")
	}
	reviewLogPath := strings.TrimSpace(os.Getenv("AM_LIVE_ROUTE_TURN_REVIEW_LOG"))
	if reviewLogPath == "" {
		return fmt.Errorf("AM_LIVE_ROUTE_TURN_REVIEW_LOG is required")
	}
	admissionLogPath := strings.TrimSpace(os.Getenv("AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_LOG"))
	if admissionLogPath == "" {
		return fmt.Errorf("AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_LOG is required")
	}
	adapterLogPath := strings.TrimSpace(os.Getenv("AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_ADAPTER_LOG"))
	if adapterLogPath == "" {
		return fmt.Errorf("AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_ADAPTER_LOG is required")
	}
	if !admissionLiveRouteTurnCandidateDraftDryRun() {
		return fmt.Errorf("AM_LIVE_ROUTE_TURN_CANDIDATE_DRAFT_DRY_RUN is required")
	}
	if !admissionLiveRouteTurnCandidateAdmissionDryRun() {
		return fmt.Errorf("AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_DRY_RUN is required")
	}
	if !admissionLiveRouteTurnCandidateAdmissionAdapterDryRun() {
		return fmt.Errorf("AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_ADAPTER_DRY_RUN is required")
	}
	if strings.TrimSpace(os.Getenv("AM_LIVE_ROUTE_TURN_CANDIDATE_DRAFT_TEXT")) == "" {
		return fmt.Errorf("AM_LIVE_ROUTE_TURN_CANDIDATE_DRAFT_TEXT is required")
	}

	cases := []struct {
		name              string
		obs               admissionLiveRouteTurnObservation
		wantPassed        bool
		wantDraftNeedle   string
		wantHandoffNeedle string
		wantAdapterNeedle string
		wantReasonNeedle  string
	}{
		{
			name:              "chat identity chain reaches admission adapter",
			obs:               admissionLiveRouteTurnObservationForHuman("Who are you?"),
			wantPassed:        true,
			wantDraftNeedle:   "live-route candidate draft dry-run: class=identity route=chorus source=chorus",
			wantHandoffNeedle: "live-route candidate admission handoff dry-run: class=identity route=chorus source=chorus draft=draft-",
			wantAdapterNeedle: "live-route candidate admission adapter dry-run: class=identity route=chorus source=chorus handoff=handoff-",
		},
		{
			name:              "chat unknown turn fails closed through chain",
			obs:               admissionLiveRouteTurnObservationForHuman("hello"),
			wantDraftNeedle:   "live-route candidate draft dry-run: class=unknown route= source=",
			wantHandoffNeedle: "live-route candidate admission handoff dry-run: class=unknown route= source=",
			wantAdapterNeedle: "live-route candidate admission adapter dry-run: class=unknown route= source=",
			wantReasonNeedle:  "live route plan failed: unknown_prompt_class",
		},
	}
	for i, tc := range cases {
		draftLine := chatLiveRouteTurnCandidateDraftDryRunLine(tc.obs)
		if !strings.Contains(draftLine, tc.wantDraftNeedle) {
			return fmt.Errorf("case %d %s bad chat draft line: %q", i+1, tc.name, draftLine)
		}
		handoffLine := chatLiveRouteTurnCandidateAdmissionDryRunLine(tc.obs)
		if !strings.Contains(handoffLine, tc.wantHandoffNeedle) {
			return fmt.Errorf("case %d %s bad chat handoff line: %q", i+1, tc.name, handoffLine)
		}
		adapterLine := chatLiveRouteTurnCandidateAdmissionAdapterDryRunLine(tc.obs)
		if !strings.Contains(adapterLine, tc.wantAdapterNeedle) {
			return fmt.Errorf("case %d %s bad chat adapter line: %q", i+1, tc.name, adapterLine)
		}
		if tc.wantReasonNeedle != "" {
			for _, line := range []string{draftLine, handoffLine, adapterLine} {
				if !strings.Contains(line, tc.wantReasonNeedle) {
					return fmt.Errorf("case %d %s missing reason %q in %q", i+1, tc.name, tc.wantReasonNeedle, line)
				}
			}
		}
		fmt.Println(draftLine)
		fmt.Println(handoffLine)
		fmt.Println(adapterLine)
	}

	draftRaw, err := os.ReadFile(draftLogPath)
	if err != nil {
		return err
	}
	draftLines := strings.Split(strings.TrimSpace(string(draftRaw)), "\n")
	if len(draftLines) != len(cases) {
		return fmt.Errorf("expected %d chat draft receipts, got %d", len(cases), len(draftLines))
	}
	reviewRaw, err := os.ReadFile(reviewLogPath)
	if err != nil {
		return err
	}
	reviewLines := strings.Split(strings.TrimSpace(string(reviewRaw)), "\n")
	if len(reviewLines) != len(cases) {
		return fmt.Errorf("expected %d chat review receipts, got %d", len(cases), len(reviewLines))
	}
	admissionRaw, err := os.ReadFile(admissionLogPath)
	if err != nil {
		return err
	}
	admissionLines := strings.Split(strings.TrimSpace(string(admissionRaw)), "\n")
	if len(admissionLines) != len(cases) {
		return fmt.Errorf("expected %d chat handoff receipts, got %d", len(cases), len(admissionLines))
	}
	adapterRaw, err := os.ReadFile(adapterLogPath)
	if err != nil {
		return err
	}
	adapterLines := strings.Split(strings.TrimSpace(string(adapterRaw)), "\n")
	if len(adapterLines) != len(cases) {
		return fmt.Errorf("expected %d chat adapter receipts, got %d", len(cases), len(adapterLines))
	}

	for i := range cases {
		var draft admissionLiveRouteTurnCandidateDraft
		if err := json.Unmarshal([]byte(draftLines[i]), &draft); err != nil {
			return fmt.Errorf("chat draft receipt %d: %w", i+1, err)
		}
		var review admissionLiveRouteTurnCandidateReview
		if err := json.Unmarshal([]byte(reviewLines[i]), &review); err != nil {
			return fmt.Errorf("chat review receipt %d: %w", i+1, err)
		}
		var admission admissionLiveRouteTurnCandidateAdmission
		if err := json.Unmarshal([]byte(admissionLines[i]), &admission); err != nil {
			return fmt.Errorf("chat handoff receipt %d: %w", i+1, err)
		}
		var adapter admissionLiveRouteTurnCandidateAdmissionAdapter
		if err := json.Unmarshal([]byte(adapterLines[i]), &adapter); err != nil {
			return fmt.Errorf("chat adapter receipt %d: %w", i+1, err)
		}
		if draft.Schema != admissionLiveRouteTurnCandidateDraftSchema ||
			admission.Schema != admissionLiveRouteTurnCandidateAdmissionSchema ||
			review.Schema != admissionLiveRouteTurnReviewSchema ||
			adapter.Schema != admissionLiveRouteTurnCandidateAdmissionAdapterSchema {
			return fmt.Errorf("chat receipt %d schema mismatch: draft=%+v review=%+v handoff=%+v adapter=%+v", i+1, draft, review, admission, adapter)
		}
		if draft.Passed != cases[i].wantPassed ||
			review.Matched != cases[i].wantPassed ||
			admission.Passed != cases[i].wantPassed ||
			adapter.Passed != cases[i].wantPassed {
			return fmt.Errorf("chat receipt %d passed mismatch: draft=%+v review=%+v handoff=%+v adapter=%+v", i+1, draft, review, admission, adapter)
		}
		if cases[i].wantPassed {
			if !strings.HasPrefix(draft.DraftID, "draft-") ||
				!strings.HasPrefix(draft.GeneratorAdapterID, "adapter-") ||
				review.CandidateDraftID != draft.DraftID ||
				review.GeneratorAdapterID != draft.GeneratorAdapterID ||
				review.CandidateRunID != draft.CandidateRunID ||
				!strings.HasPrefix(admission.HandoffID, "handoff-") ||
				!strings.HasPrefix(adapter.AdmissionAdapterID, "admission-adapter-") ||
				adapter.DreamCandidateRunID == "" ||
				adapter.DreamCandidateRunID != draft.CandidateRunID ||
				admission.CandidateDraftID != draft.DraftID ||
				adapter.CandidateDraftID != draft.DraftID ||
				adapter.HandoffID != admission.HandoffID {
				return fmt.Errorf("chat receipt %d lost admission provenance: draft=%+v review=%+v handoff=%+v adapter=%+v", i+1, draft, review, admission, adapter)
			}
		} else if draft.DraftID != "" || admission.HandoffID != "" || adapter.AdmissionAdapterID != "" {
			return fmt.Errorf("chat failed receipt %d should not name ids: draft=%+v review=%+v handoff=%+v adapter=%+v", i+1, draft, review, admission, adapter)
		}
	}

	fmt.Printf("[admission-live-route-turn-candidate-admission-chat-smoke] pass: drafts=%s reviews=%s handoffs=%s adapters=%s cases=%d\n",
		draftLogPath, reviewLogPath, admissionLogPath, adapterLogPath, len(cases))
	return nil
}

func runAdmissionLiveRouteTurnCandidateAdmissionChatShadowSmoke() error {
	draftLogPath := strings.TrimSpace(os.Getenv("AM_LIVE_ROUTE_TURN_CANDIDATE_DRAFT_LOG"))
	if draftLogPath == "" {
		return fmt.Errorf("AM_LIVE_ROUTE_TURN_CANDIDATE_DRAFT_LOG is required")
	}
	reviewLogPath := strings.TrimSpace(os.Getenv("AM_LIVE_ROUTE_TURN_REVIEW_LOG"))
	if reviewLogPath == "" {
		return fmt.Errorf("AM_LIVE_ROUTE_TURN_REVIEW_LOG is required")
	}
	admissionLogPath := strings.TrimSpace(os.Getenv("AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_LOG"))
	if admissionLogPath == "" {
		return fmt.Errorf("AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_LOG is required")
	}
	adapterLogPath := strings.TrimSpace(os.Getenv("AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_ADAPTER_LOG"))
	if adapterLogPath == "" {
		return fmt.Errorf("AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_ADAPTER_LOG is required")
	}
	dreamLogPath := strings.TrimSpace(os.Getenv("AM_DREAM_ADMISSION_LOG"))
	if dreamLogPath == "" {
		return fmt.Errorf("AM_DREAM_ADMISSION_LOG is required")
	}
	if !admissionLiveRouteTurnCandidateDraftDryRun() {
		return fmt.Errorf("AM_LIVE_ROUTE_TURN_CANDIDATE_DRAFT_DRY_RUN is required")
	}
	if !admissionLiveRouteTurnCandidateAdmissionDryRun() {
		return fmt.Errorf("AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_DRY_RUN is required")
	}
	if !admissionLiveRouteTurnCandidateAdmissionAdapterDryRun() {
		return fmt.Errorf("AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_ADAPTER_DRY_RUN is required")
	}
	if !admissionLiveRouteTurnCandidateAdmissionShadowDryRun() {
		return fmt.Errorf("AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_SHADOW_DRY_RUN is required")
	}
	if mode := dreamAdmissionMode(); mode != dreamAdmissionShadow {
		return fmt.Errorf("AM_DREAM_ADMISSION=%q, want %q", mode, dreamAdmissionShadow)
	}
	if !dreamAdmissionRequireLiveRoutePlan() {
		return fmt.Errorf("AM_DREAM_ADMISSION_REQUIRE_LIVE_ROUTE_PLAN is required")
	}
	if strings.TrimSpace(os.Getenv("AM_LIVE_ROUTE_TURN_CANDIDATE_DRAFT_TEXT")) == "" {
		return fmt.Errorf("AM_LIVE_ROUTE_TURN_CANDIDATE_DRAFT_TEXT is required")
	}

	cases := []struct {
		name              string
		obs               admissionLiveRouteTurnObservation
		wantPassed        bool
		wantDraftNeedle   string
		wantHandoffNeedle string
		wantAdapterNeedle string
		wantShadowNeedle  string
		wantReasonNeedle  string
	}{
		{
			name:              "chat identity adapter reaches shadow admission",
			obs:               admissionLiveRouteTurnObservationForHuman("Who are you?"),
			wantPassed:        true,
			wantDraftNeedle:   "live-route candidate draft dry-run: class=identity route=chorus source=chorus",
			wantHandoffNeedle: "live-route candidate admission handoff dry-run: class=identity route=chorus source=chorus draft=draft-",
			wantAdapterNeedle: "live-route candidate admission adapter dry-run: class=identity route=chorus source=chorus handoff=handoff-",
			wantShadowNeedle:  "live-route candidate admission shadow dry-run: class=identity route=chorus source=chorus handoff=handoff-",
		},
		{
			name:              "chat unknown turn fails before shadow admission",
			obs:               admissionLiveRouteTurnObservationForHuman("hello"),
			wantDraftNeedle:   "live-route candidate draft dry-run: class=unknown route= source=",
			wantHandoffNeedle: "live-route candidate admission handoff dry-run: class=unknown route= source=",
			wantAdapterNeedle: "live-route candidate admission adapter dry-run: class=unknown route= source=",
			wantShadowNeedle:  "live-route candidate admission shadow dry-run: class=unknown route= source=",
			wantReasonNeedle:  "candidate_admission_adapter_failed: candidate_admission_handoff_failed: turn_route_failed: live route plan failed: unknown_prompt_class",
		},
	}
	for i, tc := range cases {
		draftLine := chatLiveRouteTurnCandidateDraftDryRunLine(tc.obs)
		if !strings.Contains(draftLine, tc.wantDraftNeedle) {
			return fmt.Errorf("case %d %s bad chat draft line: %q", i+1, tc.name, draftLine)
		}
		handoffLine := chatLiveRouteTurnCandidateAdmissionDryRunLine(tc.obs)
		if !strings.Contains(handoffLine, tc.wantHandoffNeedle) {
			return fmt.Errorf("case %d %s bad chat handoff line: %q", i+1, tc.name, handoffLine)
		}
		adapterLine := chatLiveRouteTurnCandidateAdmissionAdapterDryRunLine(tc.obs)
		if !strings.Contains(adapterLine, tc.wantAdapterNeedle) {
			return fmt.Errorf("case %d %s bad chat adapter line: %q", i+1, tc.name, adapterLine)
		}
		shadowLine := chatLiveRouteTurnCandidateAdmissionShadowDryRunLine(tc.obs)
		if !strings.Contains(shadowLine, tc.wantShadowNeedle) {
			return fmt.Errorf("case %d %s bad chat shadow line: %q", i+1, tc.name, shadowLine)
		}
		if tc.wantPassed {
			for _, want := range []string{"admission_adapter=admission-adapter-", "policy=true", "accepted=false", "passed=true", "reason=shadow mode"} {
				if !strings.Contains(shadowLine, want) {
					return fmt.Errorf("case %d %s shadow line missing %q: %q", i+1, tc.name, want, shadowLine)
				}
			}
		} else {
			for _, want := range []string{"admission_adapter=", "policy=false", "accepted=false", "passed=false", tc.wantReasonNeedle} {
				if !strings.Contains(shadowLine, want) {
					return fmt.Errorf("case %d %s failed shadow line missing %q: %q", i+1, tc.name, want, shadowLine)
				}
			}
		}
		fmt.Println(draftLine)
		fmt.Println(handoffLine)
		fmt.Println(adapterLine)
		fmt.Println(shadowLine)
	}

	countLines := func(path string, want int, label string) error {
		raw, err := os.ReadFile(path)
		if err != nil {
			return err
		}
		lines := strings.Split(strings.TrimSpace(string(raw)), "\n")
		if len(lines) != want {
			return fmt.Errorf("expected %d %s receipts, got %d", want, label, len(lines))
		}
		return nil
	}
	if err := countLines(draftLogPath, len(cases), "chat draft"); err != nil {
		return err
	}
	if err := countLines(reviewLogPath, len(cases), "chat review"); err != nil {
		return err
	}
	if err := countLines(admissionLogPath, len(cases), "chat handoff"); err != nil {
		return err
	}
	if err := countLines(adapterLogPath, len(cases), "chat adapter"); err != nil {
		return err
	}

	adapterRaw, err := os.ReadFile(adapterLogPath)
	if err != nil {
		return err
	}
	adapterLines := strings.Split(strings.TrimSpace(string(adapterRaw)), "\n")
	var adapter admissionLiveRouteTurnCandidateAdmissionAdapter
	if err := json.Unmarshal([]byte(adapterLines[0]), &adapter); err != nil {
		return fmt.Errorf("chat adapter receipt: %w", err)
	}
	dreamRaw, err := os.ReadFile(dreamLogPath)
	if err != nil {
		return err
	}
	dreamLines := strings.Split(strings.TrimSpace(string(dreamRaw)), "\n")
	if len(dreamLines) != 1 {
		return fmt.Errorf("expected 1 chat shadow admission receipt, got %d", len(dreamLines))
	}
	var candidate dreamCandidate
	if err := json.Unmarshal([]byte(dreamLines[0]), &candidate); err != nil {
		return fmt.Errorf("chat shadow admission receipt: %w", err)
	}
	if candidate.Schema != "arianna.dream_candidate.v1" ||
		candidate.LiveRouteCandidateAdmission == nil ||
		candidate.LiveRouteCandidateAdmission.AdmissionAdapterID != adapter.AdmissionAdapterID ||
		candidate.LiveRouteCandidateAdmission.HandoffID != adapter.HandoffID ||
		candidate.Admission == nil ||
		!candidate.Admission.Passed ||
		candidate.Admission.LiveRouteChoice == nil ||
		!candidate.Admission.LiveRouteChoice.Passed ||
		candidate.Accepted ||
		candidate.Reason != "shadow mode" {
		return fmt.Errorf("bad chat shadow admission receipt: %+v", candidate)
	}

	fmt.Printf("[admission-live-route-turn-candidate-admission-chat-shadow-smoke] pass: drafts=%s reviews=%s handoffs=%s adapters=%s admission=%s cases=%d\n",
		draftLogPath, reviewLogPath, admissionLogPath, adapterLogPath, dreamLogPath, len(cases))
	return nil
}

func runAdmissionLiveRouteTurnCandidateNanoDirectChatShadowSmoke() error {
	executionLogPath := strings.TrimSpace(os.Getenv("AM_LIVE_ROUTE_TURN_CANDIDATE_EXECUTION_LOG"))
	if executionLogPath == "" {
		return fmt.Errorf("AM_LIVE_ROUTE_TURN_CANDIDATE_EXECUTION_LOG is required")
	}
	adapterLogPath := strings.TrimSpace(os.Getenv("AM_LIVE_ROUTE_TURN_GENERATOR_ADAPTER_LOG"))
	if adapterLogPath == "" {
		return fmt.Errorf("AM_LIVE_ROUTE_TURN_GENERATOR_ADAPTER_LOG is required")
	}
	draftLogPath := strings.TrimSpace(os.Getenv("AM_LIVE_ROUTE_TURN_CANDIDATE_DRAFT_LOG"))
	if draftLogPath == "" {
		return fmt.Errorf("AM_LIVE_ROUTE_TURN_CANDIDATE_DRAFT_LOG is required")
	}
	reviewLogPath := strings.TrimSpace(os.Getenv("AM_LIVE_ROUTE_TURN_REVIEW_LOG"))
	if reviewLogPath == "" {
		return fmt.Errorf("AM_LIVE_ROUTE_TURN_REVIEW_LOG is required")
	}
	admissionLogPath := strings.TrimSpace(os.Getenv("AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_LOG"))
	if admissionLogPath == "" {
		return fmt.Errorf("AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_LOG is required")
	}
	admissionAdapterLogPath := strings.TrimSpace(os.Getenv("AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_ADAPTER_LOG"))
	if admissionAdapterLogPath == "" {
		return fmt.Errorf("AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_ADAPTER_LOG is required")
	}
	dreamLogPath := strings.TrimSpace(os.Getenv("AM_DREAM_ADMISSION_LOG"))
	if dreamLogPath == "" {
		return fmt.Errorf("AM_DREAM_ADMISSION_LOG is required")
	}
	decisionLogPath := strings.TrimSpace(os.Getenv("AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_DECISION_LOG"))
	if admissionLiveRouteTurnCandidateAdmissionDecisionDryRun() && decisionLogPath == "" {
		return fmt.Errorf("AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_DECISION_LOG is required")
	}
	promotionLogPath := strings.TrimSpace(os.Getenv("AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_PROMOTION_LOG"))
	if admissionLiveRouteTurnCandidateAdmissionPromotionDryRun() && promotionLogPath == "" {
		return fmt.Errorf("AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_PROMOTION_LOG is required")
	}
	if admissionLiveRouteTurnCandidateAdmissionPromotionDryRun() && !admissionLiveRouteTurnCandidateAdmissionDecisionDryRun() {
		return fmt.Errorf("AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_DECISION_DRY_RUN is required for promotion smoke")
	}
	switchLogPath := strings.TrimSpace(os.Getenv("AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_SWITCH_LOG"))
	if admissionLiveRouteTurnCandidateAdmissionSwitchDryRun() && switchLogPath == "" {
		return fmt.Errorf("AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_SWITCH_LOG is required")
	}
	if admissionLiveRouteTurnCandidateAdmissionSwitchDryRun() &&
		(!admissionLiveRouteTurnCandidateAdmissionDecisionDryRun() || !admissionLiveRouteTurnCandidateAdmissionPromotionDryRun()) {
		return fmt.Errorf("AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_DECISION_DRY_RUN and AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_PROMOTION_DRY_RUN are required for switch smoke")
	}
	enableGateLogPath := strings.TrimSpace(os.Getenv("AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_ENABLE_GATE_LOG"))
	if admissionLiveRouteTurnCandidateAdmissionEnableGateDryRun() && enableGateLogPath == "" {
		return fmt.Errorf("AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_ENABLE_GATE_LOG is required")
	}
	liveStageLogPath := strings.TrimSpace(os.Getenv("AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_LIVE_STAGE_LOG"))
	if admissionLiveRouteTurnCandidateAdmissionLiveStageDryRun() && liveStageLogPath == "" {
		return fmt.Errorf("AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_LIVE_STAGE_LOG is required")
	}
	writerPreflightLogPath := strings.TrimSpace(os.Getenv("AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_WRITER_PREFLIGHT_LOG"))
	if admissionLiveRouteTurnCandidateAdmissionWriterPreflightDryRun() && writerPreflightLogPath == "" {
		return fmt.Errorf("AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_WRITER_PREFLIGHT_LOG is required")
	}
	writerInventoryLogPath := strings.TrimSpace(os.Getenv("AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_WRITER_INVENTORY_LOG"))
	if admissionLiveRouteTurnCandidateAdmissionWriterInventoryDryRun() && writerInventoryLogPath == "" {
		return fmt.Errorf("AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_WRITER_INVENTORY_LOG is required")
	}
	writerContractLogPath := strings.TrimSpace(os.Getenv("AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_WRITER_CONTRACT_LOG"))
	if admissionLiveRouteTurnCandidateAdmissionWriterContractDryRun() && writerContractLogPath == "" {
		return fmt.Errorf("AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_WRITER_CONTRACT_LOG is required")
	}
	ledgerLogPath := strings.TrimSpace(os.Getenv("AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_LEDGER_LOG"))
	if admissionLiveRouteTurnCandidateAdmissionLedgerDryRun() && ledgerLogPath == "" {
		return fmt.Errorf("AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_LEDGER_LOG is required")
	}
	writerImplLogPath := strings.TrimSpace(os.Getenv("AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_WRITER_IMPLEMENTATION_LOG"))
	if admissionLiveRouteTurnCandidateAdmissionWriterImplementationDryRun() && writerImplLogPath == "" {
		return fmt.Errorf("AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_WRITER_IMPLEMENTATION_LOG is required")
	}
	writerReceiptLogPath := strings.TrimSpace(os.Getenv("AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_WRITER_RECEIPT_LOG"))
	if admissionLiveRouteTurnCandidateAdmissionWriterReceiptDryRun() && writerReceiptLogPath == "" {
		return fmt.Errorf("AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_WRITER_RECEIPT_LOG is required")
	}
	rollbackImplLogPath := strings.TrimSpace(os.Getenv("AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_ROLLBACK_IMPLEMENTATION_LOG"))
	if admissionLiveRouteTurnCandidateAdmissionRollbackImplementationDryRun() && rollbackImplLogPath == "" {
		return fmt.Errorf("AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_ROLLBACK_IMPLEMENTATION_LOG is required")
	}
	ledgerImplLogPath := strings.TrimSpace(os.Getenv("AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_LEDGER_IMPLEMENTATION_LOG"))
	if admissionLiveRouteTurnCandidateAdmissionLedgerImplementationDryRun() && ledgerImplLogPath == "" {
		return fmt.Errorf("AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_LEDGER_IMPLEMENTATION_LOG is required")
	}
	ledgerPersistenceLogPath := strings.TrimSpace(os.Getenv("AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_LEDGER_PERSISTENCE_LOG"))
	if admissionLiveRouteTurnCandidateAdmissionLedgerPersistenceDryRun() && ledgerPersistenceLogPath == "" {
		return fmt.Errorf("AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_LEDGER_PERSISTENCE_LOG is required")
	}
	ledgerVerificationLogPath := strings.TrimSpace(os.Getenv("AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_LEDGER_VERIFICATION_LOG"))
	if admissionLiveRouteTurnCandidateAdmissionLedgerVerificationDryRun() && ledgerVerificationLogPath == "" {
		return fmt.Errorf("AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_LEDGER_VERIFICATION_LOG is required")
	}
	readinessLogPath := strings.TrimSpace(os.Getenv("AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_READINESS_LOG"))
	if admissionLiveRouteTurnCandidateAdmissionReadinessDryRun() && readinessLogPath == "" {
		return fmt.Errorf("AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_READINESS_LOG is required")
	}
	permitLogPath := strings.TrimSpace(os.Getenv("AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_PERMIT_LOG"))
	if admissionLiveRouteTurnCandidateAdmissionPermitDryRun() && permitLogPath == "" {
		return fmt.Errorf("AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_PERMIT_LOG is required")
	}
	sealLogPath := strings.TrimSpace(os.Getenv("AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_SEAL_LOG"))
	if admissionLiveRouteTurnCandidateAdmissionSealDryRun() && sealLogPath == "" {
		return fmt.Errorf("AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_SEAL_LOG is required")
	}
	finalGateLogPath := strings.TrimSpace(os.Getenv("AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_FINAL_GATE_LOG"))
	if admissionLiveRouteTurnCandidateAdmissionFinalGateDryRun() && finalGateLogPath == "" {
		return fmt.Errorf("AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_FINAL_GATE_LOG is required")
	}
	resonanceIntentLogPath := strings.TrimSpace(os.Getenv("AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_RESONANCE_INTENT_LOG"))
	if admissionLiveRouteTurnCandidateAdmissionResonanceIntentDryRun() && resonanceIntentLogPath == "" {
		return fmt.Errorf("AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_RESONANCE_INTENT_LOG is required")
	}
	resonanceReceiverLogPath := strings.TrimSpace(os.Getenv("AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_RESONANCE_RECEIVER_LOG"))
	if admissionLiveRouteTurnCandidateAdmissionResonanceReceiverDryRun() && resonanceReceiverLogPath == "" {
		return fmt.Errorf("AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_RESONANCE_RECEIVER_LOG is required")
	}
	resonanceObservationLogPath := strings.TrimSpace(os.Getenv("AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_RESONANCE_OBSERVATION_LOG"))
	if admissionLiveRouteTurnCandidateAdmissionResonanceObservationDryRun() && resonanceObservationLogPath == "" {
		return fmt.Errorf("AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_RESONANCE_OBSERVATION_LOG is required")
	}
	resonanceGraftBoundaryLogPath := strings.TrimSpace(os.Getenv("AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_RESONANCE_GRAFT_BOUNDARY_LOG"))
	if admissionLiveRouteTurnCandidateAdmissionResonanceGraftBoundaryDryRun() && resonanceGraftBoundaryLogPath == "" {
		return fmt.Errorf("AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_RESONANCE_GRAFT_BOUNDARY_LOG is required")
	}
	resonanceGraftPreflightLogPath := strings.TrimSpace(os.Getenv("AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_RESONANCE_GRAFT_PREFLIGHT_LOG"))
	if admissionLiveRouteTurnCandidateAdmissionResonanceGraftPreflightDryRun() && resonanceGraftPreflightLogPath == "" {
		return fmt.Errorf("AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_RESONANCE_GRAFT_PREFLIGHT_LOG is required")
	}
	resonanceGraftGateLogPath := strings.TrimSpace(os.Getenv("AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_RESONANCE_GRAFT_GATE_LOG"))
	if admissionLiveRouteTurnCandidateAdmissionResonanceGraftGateDryRun() && resonanceGraftGateLogPath == "" {
		return fmt.Errorf("AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_RESONANCE_GRAFT_GATE_LOG is required")
	}
	resonanceGraftCandidateLogPath := strings.TrimSpace(os.Getenv("AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_RESONANCE_GRAFT_CANDIDATE_LOG"))
	if admissionLiveRouteTurnCandidateAdmissionResonanceGraftCandidateDryRun() && resonanceGraftCandidateLogPath == "" {
		return fmt.Errorf("AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_RESONANCE_GRAFT_CANDIDATE_LOG is required")
	}
	resonanceGraftCandidateStoreLogPath := strings.TrimSpace(os.Getenv("AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_RESONANCE_GRAFT_CANDIDATE_STORE_LOG"))
	if admissionLiveRouteTurnCandidateAdmissionResonanceGraftCandidateStoreDryRun() && resonanceGraftCandidateStoreLogPath == "" {
		return fmt.Errorf("AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_RESONANCE_GRAFT_CANDIDATE_STORE_LOG is required")
	}
	resonanceGraftCandidateStoreReaderLogPath := strings.TrimSpace(os.Getenv("AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_RESONANCE_GRAFT_CANDIDATE_STORE_READER_LOG"))
	if admissionLiveRouteTurnCandidateAdmissionResonanceGraftCandidateStoreReaderDryRun() && resonanceGraftCandidateStoreReaderLogPath == "" {
		return fmt.Errorf("AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_RESONANCE_GRAFT_CANDIDATE_STORE_READER_LOG is required")
	}
	resonanceGraftAdmissionProofLogPath := strings.TrimSpace(os.Getenv("AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_RESONANCE_GRAFT_ADMISSION_PROOF_LOG"))
	if admissionLiveRouteTurnCandidateAdmissionResonanceGraftAdmissionProofDryRun() && resonanceGraftAdmissionProofLogPath == "" {
		return fmt.Errorf("AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_RESONANCE_GRAFT_ADMISSION_PROOF_LOG is required")
	}
	if admissionLiveRouteTurnCandidateAdmissionEnableGateDryRun() &&
		(!admissionLiveRouteTurnCandidateAdmissionDecisionDryRun() ||
			!admissionLiveRouteTurnCandidateAdmissionPromotionDryRun() ||
			!admissionLiveRouteTurnCandidateAdmissionSwitchDryRun()) {
		return fmt.Errorf("AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_DECISION_DRY_RUN, AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_PROMOTION_DRY_RUN, and AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_SWITCH_DRY_RUN are required for enable gate smoke")
	}
	if admissionLiveRouteTurnCandidateAdmissionLiveStageDryRun() &&
		(!admissionLiveRouteTurnCandidateAdmissionDecisionDryRun() ||
			!admissionLiveRouteTurnCandidateAdmissionPromotionDryRun() ||
			!admissionLiveRouteTurnCandidateAdmissionSwitchDryRun() ||
			!admissionLiveRouteTurnCandidateAdmissionEnableGateDryRun()) {
		return fmt.Errorf("AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_DECISION_DRY_RUN, AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_PROMOTION_DRY_RUN, AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_SWITCH_DRY_RUN, and AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_ENABLE_GATE_DRY_RUN are required for live stage smoke")
	}
	if admissionLiveRouteTurnCandidateAdmissionWriterPreflightDryRun() &&
		(!admissionLiveRouteTurnCandidateAdmissionDecisionDryRun() ||
			!admissionLiveRouteTurnCandidateAdmissionPromotionDryRun() ||
			!admissionLiveRouteTurnCandidateAdmissionSwitchDryRun() ||
			!admissionLiveRouteTurnCandidateAdmissionEnableGateDryRun() ||
			!admissionLiveRouteTurnCandidateAdmissionLiveStageDryRun()) {
		return fmt.Errorf("AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_DECISION_DRY_RUN, AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_PROMOTION_DRY_RUN, AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_SWITCH_DRY_RUN, AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_ENABLE_GATE_DRY_RUN, and AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_LIVE_STAGE_DRY_RUN are required for writer preflight smoke")
	}
	if admissionLiveRouteTurnCandidateAdmissionWriterInventoryDryRun() &&
		(!admissionLiveRouteTurnCandidateAdmissionDecisionDryRun() ||
			!admissionLiveRouteTurnCandidateAdmissionPromotionDryRun() ||
			!admissionLiveRouteTurnCandidateAdmissionSwitchDryRun() ||
			!admissionLiveRouteTurnCandidateAdmissionEnableGateDryRun() ||
			!admissionLiveRouteTurnCandidateAdmissionLiveStageDryRun() ||
			!admissionLiveRouteTurnCandidateAdmissionWriterPreflightDryRun()) {
		return fmt.Errorf("AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_DECISION_DRY_RUN, AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_PROMOTION_DRY_RUN, AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_SWITCH_DRY_RUN, AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_ENABLE_GATE_DRY_RUN, AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_LIVE_STAGE_DRY_RUN, and AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_WRITER_PREFLIGHT_DRY_RUN are required for writer inventory smoke")
	}
	if admissionLiveRouteTurnCandidateAdmissionWriterContractDryRun() &&
		(!admissionLiveRouteTurnCandidateAdmissionDecisionDryRun() ||
			!admissionLiveRouteTurnCandidateAdmissionPromotionDryRun() ||
			!admissionLiveRouteTurnCandidateAdmissionSwitchDryRun() ||
			!admissionLiveRouteTurnCandidateAdmissionEnableGateDryRun() ||
			!admissionLiveRouteTurnCandidateAdmissionLiveStageDryRun() ||
			!admissionLiveRouteTurnCandidateAdmissionWriterPreflightDryRun() ||
			!admissionLiveRouteTurnCandidateAdmissionWriterInventoryDryRun()) {
		return fmt.Errorf("AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_DECISION_DRY_RUN, AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_PROMOTION_DRY_RUN, AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_SWITCH_DRY_RUN, AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_ENABLE_GATE_DRY_RUN, AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_LIVE_STAGE_DRY_RUN, AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_WRITER_PREFLIGHT_DRY_RUN, and AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_WRITER_INVENTORY_DRY_RUN are required for writer contract smoke")
	}
	if admissionLiveRouteTurnCandidateAdmissionLedgerDryRun() &&
		(!admissionLiveRouteTurnCandidateAdmissionDecisionDryRun() ||
			!admissionLiveRouteTurnCandidateAdmissionPromotionDryRun() ||
			!admissionLiveRouteTurnCandidateAdmissionSwitchDryRun() ||
			!admissionLiveRouteTurnCandidateAdmissionEnableGateDryRun() ||
			!admissionLiveRouteTurnCandidateAdmissionLiveStageDryRun() ||
			!admissionLiveRouteTurnCandidateAdmissionWriterPreflightDryRun() ||
			!admissionLiveRouteTurnCandidateAdmissionWriterInventoryDryRun() ||
			!admissionLiveRouteTurnCandidateAdmissionWriterContractDryRun()) {
		return fmt.Errorf("AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_DECISION_DRY_RUN, AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_PROMOTION_DRY_RUN, AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_SWITCH_DRY_RUN, AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_ENABLE_GATE_DRY_RUN, AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_LIVE_STAGE_DRY_RUN, AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_WRITER_PREFLIGHT_DRY_RUN, AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_WRITER_INVENTORY_DRY_RUN, and AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_WRITER_CONTRACT_DRY_RUN are required for admission ledger smoke")
	}
	if admissionLiveRouteTurnCandidateAdmissionWriterImplementationDryRun() &&
		(!admissionLiveRouteTurnCandidateAdmissionDecisionDryRun() ||
			!admissionLiveRouteTurnCandidateAdmissionPromotionDryRun() ||
			!admissionLiveRouteTurnCandidateAdmissionSwitchDryRun() ||
			!admissionLiveRouteTurnCandidateAdmissionEnableGateDryRun() ||
			!admissionLiveRouteTurnCandidateAdmissionLiveStageDryRun() ||
			!admissionLiveRouteTurnCandidateAdmissionWriterPreflightDryRun() ||
			!admissionLiveRouteTurnCandidateAdmissionWriterInventoryDryRun() ||
			!admissionLiveRouteTurnCandidateAdmissionWriterContractDryRun() ||
			!admissionLiveRouteTurnCandidateAdmissionLedgerDryRun()) {
		return fmt.Errorf("AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_DECISION_DRY_RUN, AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_PROMOTION_DRY_RUN, AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_SWITCH_DRY_RUN, AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_ENABLE_GATE_DRY_RUN, AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_LIVE_STAGE_DRY_RUN, AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_WRITER_PREFLIGHT_DRY_RUN, AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_WRITER_INVENTORY_DRY_RUN, AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_WRITER_CONTRACT_DRY_RUN, and AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_LEDGER_DRY_RUN are required for writer implementation smoke")
	}
	if admissionLiveRouteTurnCandidateAdmissionWriterReceiptDryRun() &&
		(!admissionLiveRouteTurnCandidateAdmissionDecisionDryRun() ||
			!admissionLiveRouteTurnCandidateAdmissionPromotionDryRun() ||
			!admissionLiveRouteTurnCandidateAdmissionSwitchDryRun() ||
			!admissionLiveRouteTurnCandidateAdmissionEnableGateDryRun() ||
			!admissionLiveRouteTurnCandidateAdmissionLiveStageDryRun() ||
			!admissionLiveRouteTurnCandidateAdmissionWriterPreflightDryRun() ||
			!admissionLiveRouteTurnCandidateAdmissionWriterInventoryDryRun() ||
			!admissionLiveRouteTurnCandidateAdmissionWriterContractDryRun() ||
			!admissionLiveRouteTurnCandidateAdmissionLedgerDryRun() ||
			!admissionLiveRouteTurnCandidateAdmissionWriterImplementationDryRun()) {
		return fmt.Errorf("AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_DECISION_DRY_RUN, AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_PROMOTION_DRY_RUN, AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_SWITCH_DRY_RUN, AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_ENABLE_GATE_DRY_RUN, AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_LIVE_STAGE_DRY_RUN, AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_WRITER_PREFLIGHT_DRY_RUN, AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_WRITER_INVENTORY_DRY_RUN, AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_WRITER_CONTRACT_DRY_RUN, AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_LEDGER_DRY_RUN, and AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_WRITER_IMPLEMENTATION_DRY_RUN are required for writer receipt smoke")
	}
	if admissionLiveRouteTurnCandidateAdmissionRollbackImplementationDryRun() &&
		(!admissionLiveRouteTurnCandidateAdmissionDecisionDryRun() ||
			!admissionLiveRouteTurnCandidateAdmissionPromotionDryRun() ||
			!admissionLiveRouteTurnCandidateAdmissionSwitchDryRun() ||
			!admissionLiveRouteTurnCandidateAdmissionEnableGateDryRun() ||
			!admissionLiveRouteTurnCandidateAdmissionLiveStageDryRun() ||
			!admissionLiveRouteTurnCandidateAdmissionWriterPreflightDryRun() ||
			!admissionLiveRouteTurnCandidateAdmissionWriterInventoryDryRun() ||
			!admissionLiveRouteTurnCandidateAdmissionWriterContractDryRun() ||
			!admissionLiveRouteTurnCandidateAdmissionLedgerDryRun() ||
			!admissionLiveRouteTurnCandidateAdmissionWriterImplementationDryRun() ||
			!admissionLiveRouteTurnCandidateAdmissionWriterReceiptDryRun()) {
		return fmt.Errorf("AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_DECISION_DRY_RUN, AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_PROMOTION_DRY_RUN, AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_SWITCH_DRY_RUN, AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_ENABLE_GATE_DRY_RUN, AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_LIVE_STAGE_DRY_RUN, AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_WRITER_PREFLIGHT_DRY_RUN, AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_WRITER_INVENTORY_DRY_RUN, AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_WRITER_CONTRACT_DRY_RUN, AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_LEDGER_DRY_RUN, AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_WRITER_IMPLEMENTATION_DRY_RUN, and AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_WRITER_RECEIPT_DRY_RUN are required for rollback implementation smoke")
	}
	if admissionLiveRouteTurnCandidateAdmissionLedgerImplementationDryRun() &&
		(!admissionLiveRouteTurnCandidateAdmissionDecisionDryRun() ||
			!admissionLiveRouteTurnCandidateAdmissionPromotionDryRun() ||
			!admissionLiveRouteTurnCandidateAdmissionSwitchDryRun() ||
			!admissionLiveRouteTurnCandidateAdmissionEnableGateDryRun() ||
			!admissionLiveRouteTurnCandidateAdmissionLiveStageDryRun() ||
			!admissionLiveRouteTurnCandidateAdmissionWriterPreflightDryRun() ||
			!admissionLiveRouteTurnCandidateAdmissionWriterInventoryDryRun() ||
			!admissionLiveRouteTurnCandidateAdmissionWriterContractDryRun() ||
			!admissionLiveRouteTurnCandidateAdmissionLedgerDryRun() ||
			!admissionLiveRouteTurnCandidateAdmissionWriterImplementationDryRun() ||
			!admissionLiveRouteTurnCandidateAdmissionWriterReceiptDryRun() ||
			!admissionLiveRouteTurnCandidateAdmissionRollbackImplementationDryRun()) {
		return fmt.Errorf("AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_DECISION_DRY_RUN, AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_PROMOTION_DRY_RUN, AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_SWITCH_DRY_RUN, AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_ENABLE_GATE_DRY_RUN, AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_LIVE_STAGE_DRY_RUN, AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_WRITER_PREFLIGHT_DRY_RUN, AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_WRITER_INVENTORY_DRY_RUN, AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_WRITER_CONTRACT_DRY_RUN, AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_LEDGER_DRY_RUN, AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_WRITER_IMPLEMENTATION_DRY_RUN, AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_WRITER_RECEIPT_DRY_RUN, and AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_ROLLBACK_IMPLEMENTATION_DRY_RUN are required for ledger implementation smoke")
	}
	if admissionLiveRouteTurnCandidateAdmissionLedgerPersistenceDryRun() &&
		(!admissionLiveRouteTurnCandidateAdmissionDecisionDryRun() ||
			!admissionLiveRouteTurnCandidateAdmissionPromotionDryRun() ||
			!admissionLiveRouteTurnCandidateAdmissionSwitchDryRun() ||
			!admissionLiveRouteTurnCandidateAdmissionEnableGateDryRun() ||
			!admissionLiveRouteTurnCandidateAdmissionLiveStageDryRun() ||
			!admissionLiveRouteTurnCandidateAdmissionWriterPreflightDryRun() ||
			!admissionLiveRouteTurnCandidateAdmissionWriterInventoryDryRun() ||
			!admissionLiveRouteTurnCandidateAdmissionWriterContractDryRun() ||
			!admissionLiveRouteTurnCandidateAdmissionLedgerDryRun() ||
			!admissionLiveRouteTurnCandidateAdmissionWriterImplementationDryRun() ||
			!admissionLiveRouteTurnCandidateAdmissionWriterReceiptDryRun() ||
			!admissionLiveRouteTurnCandidateAdmissionRollbackImplementationDryRun() ||
			!admissionLiveRouteTurnCandidateAdmissionLedgerImplementationDryRun()) {
		return fmt.Errorf("AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_DECISION_DRY_RUN, AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_PROMOTION_DRY_RUN, AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_SWITCH_DRY_RUN, AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_ENABLE_GATE_DRY_RUN, AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_LIVE_STAGE_DRY_RUN, AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_WRITER_PREFLIGHT_DRY_RUN, AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_WRITER_INVENTORY_DRY_RUN, AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_WRITER_CONTRACT_DRY_RUN, AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_LEDGER_DRY_RUN, AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_WRITER_IMPLEMENTATION_DRY_RUN, AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_WRITER_RECEIPT_DRY_RUN, AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_ROLLBACK_IMPLEMENTATION_DRY_RUN, and AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_LEDGER_IMPLEMENTATION_DRY_RUN are required for ledger persistence smoke")
	}
	if admissionLiveRouteTurnCandidateAdmissionLedgerVerificationDryRun() &&
		(!admissionLiveRouteTurnCandidateAdmissionDecisionDryRun() ||
			!admissionLiveRouteTurnCandidateAdmissionPromotionDryRun() ||
			!admissionLiveRouteTurnCandidateAdmissionSwitchDryRun() ||
			!admissionLiveRouteTurnCandidateAdmissionEnableGateDryRun() ||
			!admissionLiveRouteTurnCandidateAdmissionLiveStageDryRun() ||
			!admissionLiveRouteTurnCandidateAdmissionWriterPreflightDryRun() ||
			!admissionLiveRouteTurnCandidateAdmissionWriterInventoryDryRun() ||
			!admissionLiveRouteTurnCandidateAdmissionWriterContractDryRun() ||
			!admissionLiveRouteTurnCandidateAdmissionLedgerDryRun() ||
			!admissionLiveRouteTurnCandidateAdmissionWriterImplementationDryRun() ||
			!admissionLiveRouteTurnCandidateAdmissionWriterReceiptDryRun() ||
			!admissionLiveRouteTurnCandidateAdmissionRollbackImplementationDryRun() ||
			!admissionLiveRouteTurnCandidateAdmissionLedgerImplementationDryRun() ||
			!admissionLiveRouteTurnCandidateAdmissionLedgerPersistenceDryRun()) {
		return fmt.Errorf("AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_DECISION_DRY_RUN, AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_PROMOTION_DRY_RUN, AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_SWITCH_DRY_RUN, AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_ENABLE_GATE_DRY_RUN, AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_LIVE_STAGE_DRY_RUN, AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_WRITER_PREFLIGHT_DRY_RUN, AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_WRITER_INVENTORY_DRY_RUN, AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_WRITER_CONTRACT_DRY_RUN, AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_LEDGER_DRY_RUN, AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_WRITER_IMPLEMENTATION_DRY_RUN, AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_WRITER_RECEIPT_DRY_RUN, AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_ROLLBACK_IMPLEMENTATION_DRY_RUN, AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_LEDGER_IMPLEMENTATION_DRY_RUN, and AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_LEDGER_PERSISTENCE_DRY_RUN are required for ledger verification smoke")
	}
	if admissionLiveRouteTurnCandidateAdmissionReadinessDryRun() &&
		(!admissionLiveRouteTurnCandidateAdmissionDecisionDryRun() ||
			!admissionLiveRouteTurnCandidateAdmissionPromotionDryRun() ||
			!admissionLiveRouteTurnCandidateAdmissionSwitchDryRun() ||
			!admissionLiveRouteTurnCandidateAdmissionEnableGateDryRun() ||
			!admissionLiveRouteTurnCandidateAdmissionLiveStageDryRun() ||
			!admissionLiveRouteTurnCandidateAdmissionWriterPreflightDryRun() ||
			!admissionLiveRouteTurnCandidateAdmissionWriterInventoryDryRun() ||
			!admissionLiveRouteTurnCandidateAdmissionWriterContractDryRun() ||
			!admissionLiveRouteTurnCandidateAdmissionLedgerDryRun() ||
			!admissionLiveRouteTurnCandidateAdmissionWriterImplementationDryRun() ||
			!admissionLiveRouteTurnCandidateAdmissionWriterReceiptDryRun() ||
			!admissionLiveRouteTurnCandidateAdmissionRollbackImplementationDryRun() ||
			!admissionLiveRouteTurnCandidateAdmissionLedgerImplementationDryRun() ||
			!admissionLiveRouteTurnCandidateAdmissionLedgerPersistenceDryRun() ||
			!admissionLiveRouteTurnCandidateAdmissionLedgerVerificationDryRun()) {
		return fmt.Errorf("AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_DECISION_DRY_RUN, AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_PROMOTION_DRY_RUN, AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_SWITCH_DRY_RUN, AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_ENABLE_GATE_DRY_RUN, AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_LIVE_STAGE_DRY_RUN, AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_WRITER_PREFLIGHT_DRY_RUN, AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_WRITER_INVENTORY_DRY_RUN, AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_WRITER_CONTRACT_DRY_RUN, AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_LEDGER_DRY_RUN, AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_WRITER_IMPLEMENTATION_DRY_RUN, AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_WRITER_RECEIPT_DRY_RUN, AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_ROLLBACK_IMPLEMENTATION_DRY_RUN, AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_LEDGER_IMPLEMENTATION_DRY_RUN, AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_LEDGER_PERSISTENCE_DRY_RUN, and AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_LEDGER_VERIFICATION_DRY_RUN are required for admission readiness smoke")
	}
	if admissionLiveRouteTurnCandidateAdmissionPermitDryRun() &&
		(!admissionLiveRouteTurnCandidateAdmissionDecisionDryRun() ||
			!admissionLiveRouteTurnCandidateAdmissionPromotionDryRun() ||
			!admissionLiveRouteTurnCandidateAdmissionSwitchDryRun() ||
			!admissionLiveRouteTurnCandidateAdmissionEnableGateDryRun() ||
			!admissionLiveRouteTurnCandidateAdmissionLiveStageDryRun() ||
			!admissionLiveRouteTurnCandidateAdmissionWriterPreflightDryRun() ||
			!admissionLiveRouteTurnCandidateAdmissionWriterInventoryDryRun() ||
			!admissionLiveRouteTurnCandidateAdmissionWriterContractDryRun() ||
			!admissionLiveRouteTurnCandidateAdmissionLedgerDryRun() ||
			!admissionLiveRouteTurnCandidateAdmissionWriterImplementationDryRun() ||
			!admissionLiveRouteTurnCandidateAdmissionWriterReceiptDryRun() ||
			!admissionLiveRouteTurnCandidateAdmissionRollbackImplementationDryRun() ||
			!admissionLiveRouteTurnCandidateAdmissionLedgerImplementationDryRun() ||
			!admissionLiveRouteTurnCandidateAdmissionLedgerPersistenceDryRun() ||
			!admissionLiveRouteTurnCandidateAdmissionLedgerVerificationDryRun() ||
			!admissionLiveRouteTurnCandidateAdmissionReadinessDryRun()) {
		return fmt.Errorf("AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_DECISION_DRY_RUN, AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_PROMOTION_DRY_RUN, AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_SWITCH_DRY_RUN, AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_ENABLE_GATE_DRY_RUN, AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_LIVE_STAGE_DRY_RUN, AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_WRITER_PREFLIGHT_DRY_RUN, AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_WRITER_INVENTORY_DRY_RUN, AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_WRITER_CONTRACT_DRY_RUN, AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_LEDGER_DRY_RUN, AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_WRITER_IMPLEMENTATION_DRY_RUN, AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_WRITER_RECEIPT_DRY_RUN, AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_ROLLBACK_IMPLEMENTATION_DRY_RUN, AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_LEDGER_IMPLEMENTATION_DRY_RUN, AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_LEDGER_PERSISTENCE_DRY_RUN, AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_LEDGER_VERIFICATION_DRY_RUN, and AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_READINESS_DRY_RUN are required for admission permit smoke")
	}
	if admissionLiveRouteTurnCandidateAdmissionSealDryRun() &&
		(!admissionLiveRouteTurnCandidateAdmissionDecisionDryRun() ||
			!admissionLiveRouteTurnCandidateAdmissionPromotionDryRun() ||
			!admissionLiveRouteTurnCandidateAdmissionSwitchDryRun() ||
			!admissionLiveRouteTurnCandidateAdmissionEnableGateDryRun() ||
			!admissionLiveRouteTurnCandidateAdmissionLiveStageDryRun() ||
			!admissionLiveRouteTurnCandidateAdmissionWriterPreflightDryRun() ||
			!admissionLiveRouteTurnCandidateAdmissionWriterInventoryDryRun() ||
			!admissionLiveRouteTurnCandidateAdmissionWriterContractDryRun() ||
			!admissionLiveRouteTurnCandidateAdmissionLedgerDryRun() ||
			!admissionLiveRouteTurnCandidateAdmissionWriterImplementationDryRun() ||
			!admissionLiveRouteTurnCandidateAdmissionWriterReceiptDryRun() ||
			!admissionLiveRouteTurnCandidateAdmissionRollbackImplementationDryRun() ||
			!admissionLiveRouteTurnCandidateAdmissionLedgerImplementationDryRun() ||
			!admissionLiveRouteTurnCandidateAdmissionLedgerPersistenceDryRun() ||
			!admissionLiveRouteTurnCandidateAdmissionLedgerVerificationDryRun() ||
			!admissionLiveRouteTurnCandidateAdmissionReadinessDryRun() ||
			!admissionLiveRouteTurnCandidateAdmissionPermitDryRun()) {
		return fmt.Errorf("AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_DECISION_DRY_RUN, AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_PROMOTION_DRY_RUN, AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_SWITCH_DRY_RUN, AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_ENABLE_GATE_DRY_RUN, AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_LIVE_STAGE_DRY_RUN, AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_WRITER_PREFLIGHT_DRY_RUN, AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_WRITER_INVENTORY_DRY_RUN, AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_WRITER_CONTRACT_DRY_RUN, AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_LEDGER_DRY_RUN, AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_WRITER_IMPLEMENTATION_DRY_RUN, AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_WRITER_RECEIPT_DRY_RUN, AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_ROLLBACK_IMPLEMENTATION_DRY_RUN, AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_LEDGER_IMPLEMENTATION_DRY_RUN, AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_LEDGER_PERSISTENCE_DRY_RUN, AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_LEDGER_VERIFICATION_DRY_RUN, AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_READINESS_DRY_RUN, and AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_PERMIT_DRY_RUN are required for admission seal smoke")
	}
	if admissionLiveRouteTurnCandidateAdmissionFinalGateDryRun() &&
		(!admissionLiveRouteTurnCandidateAdmissionDecisionDryRun() ||
			!admissionLiveRouteTurnCandidateAdmissionPromotionDryRun() ||
			!admissionLiveRouteTurnCandidateAdmissionSwitchDryRun() ||
			!admissionLiveRouteTurnCandidateAdmissionEnableGateDryRun() ||
			!admissionLiveRouteTurnCandidateAdmissionLiveStageDryRun() ||
			!admissionLiveRouteTurnCandidateAdmissionWriterPreflightDryRun() ||
			!admissionLiveRouteTurnCandidateAdmissionWriterInventoryDryRun() ||
			!admissionLiveRouteTurnCandidateAdmissionWriterContractDryRun() ||
			!admissionLiveRouteTurnCandidateAdmissionLedgerDryRun() ||
			!admissionLiveRouteTurnCandidateAdmissionWriterImplementationDryRun() ||
			!admissionLiveRouteTurnCandidateAdmissionWriterReceiptDryRun() ||
			!admissionLiveRouteTurnCandidateAdmissionRollbackImplementationDryRun() ||
			!admissionLiveRouteTurnCandidateAdmissionLedgerImplementationDryRun() ||
			!admissionLiveRouteTurnCandidateAdmissionLedgerPersistenceDryRun() ||
			!admissionLiveRouteTurnCandidateAdmissionLedgerVerificationDryRun() ||
			!admissionLiveRouteTurnCandidateAdmissionReadinessDryRun() ||
			!admissionLiveRouteTurnCandidateAdmissionPermitDryRun() ||
			!admissionLiveRouteTurnCandidateAdmissionSealDryRun()) {
		return fmt.Errorf("AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_DECISION_DRY_RUN, AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_PROMOTION_DRY_RUN, AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_SWITCH_DRY_RUN, AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_ENABLE_GATE_DRY_RUN, AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_LIVE_STAGE_DRY_RUN, AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_WRITER_PREFLIGHT_DRY_RUN, AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_WRITER_INVENTORY_DRY_RUN, AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_WRITER_CONTRACT_DRY_RUN, AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_LEDGER_DRY_RUN, AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_WRITER_IMPLEMENTATION_DRY_RUN, AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_WRITER_RECEIPT_DRY_RUN, AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_ROLLBACK_IMPLEMENTATION_DRY_RUN, AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_LEDGER_IMPLEMENTATION_DRY_RUN, AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_LEDGER_PERSISTENCE_DRY_RUN, AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_LEDGER_VERIFICATION_DRY_RUN, AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_READINESS_DRY_RUN, AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_PERMIT_DRY_RUN, and AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_SEAL_DRY_RUN are required for admission final gate smoke")
	}
	if admissionLiveRouteTurnCandidateAdmissionResonanceIntentDryRun() &&
		(!admissionLiveRouteTurnCandidateAdmissionDecisionDryRun() ||
			!admissionLiveRouteTurnCandidateAdmissionPromotionDryRun() ||
			!admissionLiveRouteTurnCandidateAdmissionSwitchDryRun() ||
			!admissionLiveRouteTurnCandidateAdmissionEnableGateDryRun() ||
			!admissionLiveRouteTurnCandidateAdmissionLiveStageDryRun() ||
			!admissionLiveRouteTurnCandidateAdmissionWriterPreflightDryRun() ||
			!admissionLiveRouteTurnCandidateAdmissionWriterInventoryDryRun() ||
			!admissionLiveRouteTurnCandidateAdmissionWriterContractDryRun() ||
			!admissionLiveRouteTurnCandidateAdmissionLedgerDryRun() ||
			!admissionLiveRouteTurnCandidateAdmissionWriterImplementationDryRun() ||
			!admissionLiveRouteTurnCandidateAdmissionWriterReceiptDryRun() ||
			!admissionLiveRouteTurnCandidateAdmissionRollbackImplementationDryRun() ||
			!admissionLiveRouteTurnCandidateAdmissionLedgerImplementationDryRun() ||
			!admissionLiveRouteTurnCandidateAdmissionLedgerPersistenceDryRun() ||
			!admissionLiveRouteTurnCandidateAdmissionLedgerVerificationDryRun() ||
			!admissionLiveRouteTurnCandidateAdmissionReadinessDryRun() ||
			!admissionLiveRouteTurnCandidateAdmissionPermitDryRun() ||
			!admissionLiveRouteTurnCandidateAdmissionSealDryRun() ||
			!admissionLiveRouteTurnCandidateAdmissionFinalGateDryRun()) {
		return fmt.Errorf("AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_DECISION_DRY_RUN, AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_PROMOTION_DRY_RUN, AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_SWITCH_DRY_RUN, AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_ENABLE_GATE_DRY_RUN, AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_LIVE_STAGE_DRY_RUN, AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_WRITER_PREFLIGHT_DRY_RUN, AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_WRITER_INVENTORY_DRY_RUN, AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_WRITER_CONTRACT_DRY_RUN, AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_LEDGER_DRY_RUN, AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_WRITER_IMPLEMENTATION_DRY_RUN, AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_WRITER_RECEIPT_DRY_RUN, AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_ROLLBACK_IMPLEMENTATION_DRY_RUN, AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_LEDGER_IMPLEMENTATION_DRY_RUN, AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_LEDGER_PERSISTENCE_DRY_RUN, AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_LEDGER_VERIFICATION_DRY_RUN, AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_READINESS_DRY_RUN, AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_PERMIT_DRY_RUN, AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_SEAL_DRY_RUN, and AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_FINAL_GATE_DRY_RUN are required for admission resonance intent smoke")
	}
	if admissionLiveRouteTurnCandidateAdmissionResonanceReceiverDryRun() &&
		(!admissionLiveRouteTurnCandidateAdmissionDecisionDryRun() ||
			!admissionLiveRouteTurnCandidateAdmissionPromotionDryRun() ||
			!admissionLiveRouteTurnCandidateAdmissionSwitchDryRun() ||
			!admissionLiveRouteTurnCandidateAdmissionEnableGateDryRun() ||
			!admissionLiveRouteTurnCandidateAdmissionLiveStageDryRun() ||
			!admissionLiveRouteTurnCandidateAdmissionWriterPreflightDryRun() ||
			!admissionLiveRouteTurnCandidateAdmissionWriterInventoryDryRun() ||
			!admissionLiveRouteTurnCandidateAdmissionWriterContractDryRun() ||
			!admissionLiveRouteTurnCandidateAdmissionLedgerDryRun() ||
			!admissionLiveRouteTurnCandidateAdmissionWriterImplementationDryRun() ||
			!admissionLiveRouteTurnCandidateAdmissionWriterReceiptDryRun() ||
			!admissionLiveRouteTurnCandidateAdmissionRollbackImplementationDryRun() ||
			!admissionLiveRouteTurnCandidateAdmissionLedgerImplementationDryRun() ||
			!admissionLiveRouteTurnCandidateAdmissionLedgerPersistenceDryRun() ||
			!admissionLiveRouteTurnCandidateAdmissionLedgerVerificationDryRun() ||
			!admissionLiveRouteTurnCandidateAdmissionReadinessDryRun() ||
			!admissionLiveRouteTurnCandidateAdmissionPermitDryRun() ||
			!admissionLiveRouteTurnCandidateAdmissionSealDryRun() ||
			!admissionLiveRouteTurnCandidateAdmissionFinalGateDryRun() ||
			!admissionLiveRouteTurnCandidateAdmissionResonanceIntentDryRun()) {
		return fmt.Errorf("AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_DECISION_DRY_RUN, AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_PROMOTION_DRY_RUN, AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_SWITCH_DRY_RUN, AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_ENABLE_GATE_DRY_RUN, AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_LIVE_STAGE_DRY_RUN, AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_WRITER_PREFLIGHT_DRY_RUN, AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_WRITER_INVENTORY_DRY_RUN, AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_WRITER_CONTRACT_DRY_RUN, AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_LEDGER_DRY_RUN, AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_WRITER_IMPLEMENTATION_DRY_RUN, AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_WRITER_RECEIPT_DRY_RUN, AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_ROLLBACK_IMPLEMENTATION_DRY_RUN, AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_LEDGER_IMPLEMENTATION_DRY_RUN, AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_LEDGER_PERSISTENCE_DRY_RUN, AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_LEDGER_VERIFICATION_DRY_RUN, AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_READINESS_DRY_RUN, AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_PERMIT_DRY_RUN, AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_SEAL_DRY_RUN, AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_FINAL_GATE_DRY_RUN, and AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_RESONANCE_INTENT_DRY_RUN are required for admission resonance receiver smoke")
	}
	if admissionLiveRouteTurnCandidateAdmissionResonanceObservationDryRun() &&
		(!admissionLiveRouteTurnCandidateAdmissionDecisionDryRun() ||
			!admissionLiveRouteTurnCandidateAdmissionPromotionDryRun() ||
			!admissionLiveRouteTurnCandidateAdmissionSwitchDryRun() ||
			!admissionLiveRouteTurnCandidateAdmissionEnableGateDryRun() ||
			!admissionLiveRouteTurnCandidateAdmissionLiveStageDryRun() ||
			!admissionLiveRouteTurnCandidateAdmissionWriterPreflightDryRun() ||
			!admissionLiveRouteTurnCandidateAdmissionWriterInventoryDryRun() ||
			!admissionLiveRouteTurnCandidateAdmissionWriterContractDryRun() ||
			!admissionLiveRouteTurnCandidateAdmissionLedgerDryRun() ||
			!admissionLiveRouteTurnCandidateAdmissionWriterImplementationDryRun() ||
			!admissionLiveRouteTurnCandidateAdmissionWriterReceiptDryRun() ||
			!admissionLiveRouteTurnCandidateAdmissionRollbackImplementationDryRun() ||
			!admissionLiveRouteTurnCandidateAdmissionLedgerImplementationDryRun() ||
			!admissionLiveRouteTurnCandidateAdmissionLedgerPersistenceDryRun() ||
			!admissionLiveRouteTurnCandidateAdmissionLedgerVerificationDryRun() ||
			!admissionLiveRouteTurnCandidateAdmissionReadinessDryRun() ||
			!admissionLiveRouteTurnCandidateAdmissionPermitDryRun() ||
			!admissionLiveRouteTurnCandidateAdmissionSealDryRun() ||
			!admissionLiveRouteTurnCandidateAdmissionFinalGateDryRun() ||
			!admissionLiveRouteTurnCandidateAdmissionResonanceIntentDryRun() ||
			!admissionLiveRouteTurnCandidateAdmissionResonanceReceiverDryRun()) {
		return fmt.Errorf("AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_DECISION_DRY_RUN, AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_PROMOTION_DRY_RUN, AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_SWITCH_DRY_RUN, AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_ENABLE_GATE_DRY_RUN, AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_LIVE_STAGE_DRY_RUN, AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_WRITER_PREFLIGHT_DRY_RUN, AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_WRITER_INVENTORY_DRY_RUN, AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_WRITER_CONTRACT_DRY_RUN, AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_LEDGER_DRY_RUN, AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_WRITER_IMPLEMENTATION_DRY_RUN, AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_WRITER_RECEIPT_DRY_RUN, AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_ROLLBACK_IMPLEMENTATION_DRY_RUN, AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_LEDGER_IMPLEMENTATION_DRY_RUN, AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_LEDGER_PERSISTENCE_DRY_RUN, AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_LEDGER_VERIFICATION_DRY_RUN, AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_READINESS_DRY_RUN, AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_PERMIT_DRY_RUN, AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_SEAL_DRY_RUN, AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_FINAL_GATE_DRY_RUN, AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_RESONANCE_INTENT_DRY_RUN, and AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_RESONANCE_RECEIVER_DRY_RUN are required for admission resonance observation smoke")
	}
	if admissionLiveRouteTurnCandidateAdmissionResonanceGraftBoundaryDryRun() &&
		(!admissionLiveRouteTurnCandidateAdmissionDecisionDryRun() ||
			!admissionLiveRouteTurnCandidateAdmissionPromotionDryRun() ||
			!admissionLiveRouteTurnCandidateAdmissionSwitchDryRun() ||
			!admissionLiveRouteTurnCandidateAdmissionEnableGateDryRun() ||
			!admissionLiveRouteTurnCandidateAdmissionLiveStageDryRun() ||
			!admissionLiveRouteTurnCandidateAdmissionWriterPreflightDryRun() ||
			!admissionLiveRouteTurnCandidateAdmissionWriterInventoryDryRun() ||
			!admissionLiveRouteTurnCandidateAdmissionWriterContractDryRun() ||
			!admissionLiveRouteTurnCandidateAdmissionLedgerDryRun() ||
			!admissionLiveRouteTurnCandidateAdmissionWriterImplementationDryRun() ||
			!admissionLiveRouteTurnCandidateAdmissionWriterReceiptDryRun() ||
			!admissionLiveRouteTurnCandidateAdmissionRollbackImplementationDryRun() ||
			!admissionLiveRouteTurnCandidateAdmissionLedgerImplementationDryRun() ||
			!admissionLiveRouteTurnCandidateAdmissionLedgerPersistenceDryRun() ||
			!admissionLiveRouteTurnCandidateAdmissionLedgerVerificationDryRun() ||
			!admissionLiveRouteTurnCandidateAdmissionReadinessDryRun() ||
			!admissionLiveRouteTurnCandidateAdmissionPermitDryRun() ||
			!admissionLiveRouteTurnCandidateAdmissionSealDryRun() ||
			!admissionLiveRouteTurnCandidateAdmissionFinalGateDryRun() ||
			!admissionLiveRouteTurnCandidateAdmissionResonanceIntentDryRun() ||
			!admissionLiveRouteTurnCandidateAdmissionResonanceReceiverDryRun() ||
			!admissionLiveRouteTurnCandidateAdmissionResonanceObservationDryRun()) {
		return fmt.Errorf("AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_DECISION_DRY_RUN, AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_PROMOTION_DRY_RUN, AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_SWITCH_DRY_RUN, AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_ENABLE_GATE_DRY_RUN, AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_LIVE_STAGE_DRY_RUN, AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_WRITER_PREFLIGHT_DRY_RUN, AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_WRITER_INVENTORY_DRY_RUN, AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_WRITER_CONTRACT_DRY_RUN, AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_LEDGER_DRY_RUN, AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_WRITER_IMPLEMENTATION_DRY_RUN, AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_WRITER_RECEIPT_DRY_RUN, AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_ROLLBACK_IMPLEMENTATION_DRY_RUN, AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_LEDGER_IMPLEMENTATION_DRY_RUN, AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_LEDGER_PERSISTENCE_DRY_RUN, AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_LEDGER_VERIFICATION_DRY_RUN, AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_READINESS_DRY_RUN, AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_PERMIT_DRY_RUN, AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_SEAL_DRY_RUN, AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_FINAL_GATE_DRY_RUN, AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_RESONANCE_INTENT_DRY_RUN, AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_RESONANCE_RECEIVER_DRY_RUN, and AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_RESONANCE_OBSERVATION_DRY_RUN are required for admission resonance graft boundary smoke")
	}
	if admissionLiveRouteTurnCandidateAdmissionResonanceGraftPreflightDryRun() &&
		(!admissionLiveRouteTurnCandidateAdmissionDecisionDryRun() ||
			!admissionLiveRouteTurnCandidateAdmissionPromotionDryRun() ||
			!admissionLiveRouteTurnCandidateAdmissionSwitchDryRun() ||
			!admissionLiveRouteTurnCandidateAdmissionEnableGateDryRun() ||
			!admissionLiveRouteTurnCandidateAdmissionLiveStageDryRun() ||
			!admissionLiveRouteTurnCandidateAdmissionWriterPreflightDryRun() ||
			!admissionLiveRouteTurnCandidateAdmissionWriterInventoryDryRun() ||
			!admissionLiveRouteTurnCandidateAdmissionWriterContractDryRun() ||
			!admissionLiveRouteTurnCandidateAdmissionLedgerDryRun() ||
			!admissionLiveRouteTurnCandidateAdmissionWriterImplementationDryRun() ||
			!admissionLiveRouteTurnCandidateAdmissionWriterReceiptDryRun() ||
			!admissionLiveRouteTurnCandidateAdmissionRollbackImplementationDryRun() ||
			!admissionLiveRouteTurnCandidateAdmissionLedgerImplementationDryRun() ||
			!admissionLiveRouteTurnCandidateAdmissionLedgerPersistenceDryRun() ||
			!admissionLiveRouteTurnCandidateAdmissionLedgerVerificationDryRun() ||
			!admissionLiveRouteTurnCandidateAdmissionReadinessDryRun() ||
			!admissionLiveRouteTurnCandidateAdmissionPermitDryRun() ||
			!admissionLiveRouteTurnCandidateAdmissionSealDryRun() ||
			!admissionLiveRouteTurnCandidateAdmissionFinalGateDryRun() ||
			!admissionLiveRouteTurnCandidateAdmissionResonanceIntentDryRun() ||
			!admissionLiveRouteTurnCandidateAdmissionResonanceReceiverDryRun() ||
			!admissionLiveRouteTurnCandidateAdmissionResonanceObservationDryRun() ||
			!admissionLiveRouteTurnCandidateAdmissionResonanceGraftBoundaryDryRun()) {
		return fmt.Errorf("AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_DECISION_DRY_RUN, AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_PROMOTION_DRY_RUN, AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_SWITCH_DRY_RUN, AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_ENABLE_GATE_DRY_RUN, AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_LIVE_STAGE_DRY_RUN, AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_WRITER_PREFLIGHT_DRY_RUN, AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_WRITER_INVENTORY_DRY_RUN, AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_WRITER_CONTRACT_DRY_RUN, AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_LEDGER_DRY_RUN, AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_WRITER_IMPLEMENTATION_DRY_RUN, AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_WRITER_RECEIPT_DRY_RUN, AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_ROLLBACK_IMPLEMENTATION_DRY_RUN, AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_LEDGER_IMPLEMENTATION_DRY_RUN, AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_LEDGER_PERSISTENCE_DRY_RUN, AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_LEDGER_VERIFICATION_DRY_RUN, AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_READINESS_DRY_RUN, AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_PERMIT_DRY_RUN, AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_SEAL_DRY_RUN, AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_FINAL_GATE_DRY_RUN, AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_RESONANCE_INTENT_DRY_RUN, AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_RESONANCE_RECEIVER_DRY_RUN, AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_RESONANCE_OBSERVATION_DRY_RUN, and AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_RESONANCE_GRAFT_BOUNDARY_DRY_RUN are required for admission resonance graft preflight smoke")
	}
	if admissionLiveRouteTurnCandidateAdmissionResonanceGraftGateDryRun() &&
		(!admissionLiveRouteTurnCandidateAdmissionDecisionDryRun() ||
			!admissionLiveRouteTurnCandidateAdmissionPromotionDryRun() ||
			!admissionLiveRouteTurnCandidateAdmissionSwitchDryRun() ||
			!admissionLiveRouteTurnCandidateAdmissionEnableGateDryRun() ||
			!admissionLiveRouteTurnCandidateAdmissionLiveStageDryRun() ||
			!admissionLiveRouteTurnCandidateAdmissionWriterPreflightDryRun() ||
			!admissionLiveRouteTurnCandidateAdmissionWriterInventoryDryRun() ||
			!admissionLiveRouteTurnCandidateAdmissionWriterContractDryRun() ||
			!admissionLiveRouteTurnCandidateAdmissionLedgerDryRun() ||
			!admissionLiveRouteTurnCandidateAdmissionWriterImplementationDryRun() ||
			!admissionLiveRouteTurnCandidateAdmissionWriterReceiptDryRun() ||
			!admissionLiveRouteTurnCandidateAdmissionRollbackImplementationDryRun() ||
			!admissionLiveRouteTurnCandidateAdmissionLedgerImplementationDryRun() ||
			!admissionLiveRouteTurnCandidateAdmissionLedgerPersistenceDryRun() ||
			!admissionLiveRouteTurnCandidateAdmissionLedgerVerificationDryRun() ||
			!admissionLiveRouteTurnCandidateAdmissionReadinessDryRun() ||
			!admissionLiveRouteTurnCandidateAdmissionPermitDryRun() ||
			!admissionLiveRouteTurnCandidateAdmissionSealDryRun() ||
			!admissionLiveRouteTurnCandidateAdmissionFinalGateDryRun() ||
			!admissionLiveRouteTurnCandidateAdmissionResonanceIntentDryRun() ||
			!admissionLiveRouteTurnCandidateAdmissionResonanceReceiverDryRun() ||
			!admissionLiveRouteTurnCandidateAdmissionResonanceObservationDryRun() ||
			!admissionLiveRouteTurnCandidateAdmissionResonanceGraftBoundaryDryRun() ||
			!admissionLiveRouteTurnCandidateAdmissionResonanceGraftPreflightDryRun()) {
		return fmt.Errorf("AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_DECISION_DRY_RUN, AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_PROMOTION_DRY_RUN, AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_SWITCH_DRY_RUN, AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_ENABLE_GATE_DRY_RUN, AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_LIVE_STAGE_DRY_RUN, AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_WRITER_PREFLIGHT_DRY_RUN, AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_WRITER_INVENTORY_DRY_RUN, AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_WRITER_CONTRACT_DRY_RUN, AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_LEDGER_DRY_RUN, AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_WRITER_IMPLEMENTATION_DRY_RUN, AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_WRITER_RECEIPT_DRY_RUN, AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_ROLLBACK_IMPLEMENTATION_DRY_RUN, AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_LEDGER_IMPLEMENTATION_DRY_RUN, AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_LEDGER_PERSISTENCE_DRY_RUN, AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_LEDGER_VERIFICATION_DRY_RUN, AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_READINESS_DRY_RUN, AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_PERMIT_DRY_RUN, AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_SEAL_DRY_RUN, AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_FINAL_GATE_DRY_RUN, AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_RESONANCE_INTENT_DRY_RUN, AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_RESONANCE_RECEIVER_DRY_RUN, AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_RESONANCE_OBSERVATION_DRY_RUN, AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_RESONANCE_GRAFT_BOUNDARY_DRY_RUN, and AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_RESONANCE_GRAFT_PREFLIGHT_DRY_RUN are required for admission resonance graft gate smoke")
	}
	if admissionLiveRouteTurnCandidateAdmissionResonanceGraftCandidateDryRun() &&
		(!admissionLiveRouteTurnCandidateAdmissionDecisionDryRun() ||
			!admissionLiveRouteTurnCandidateAdmissionPromotionDryRun() ||
			!admissionLiveRouteTurnCandidateAdmissionSwitchDryRun() ||
			!admissionLiveRouteTurnCandidateAdmissionEnableGateDryRun() ||
			!admissionLiveRouteTurnCandidateAdmissionLiveStageDryRun() ||
			!admissionLiveRouteTurnCandidateAdmissionWriterPreflightDryRun() ||
			!admissionLiveRouteTurnCandidateAdmissionWriterInventoryDryRun() ||
			!admissionLiveRouteTurnCandidateAdmissionWriterContractDryRun() ||
			!admissionLiveRouteTurnCandidateAdmissionLedgerDryRun() ||
			!admissionLiveRouteTurnCandidateAdmissionWriterImplementationDryRun() ||
			!admissionLiveRouteTurnCandidateAdmissionWriterReceiptDryRun() ||
			!admissionLiveRouteTurnCandidateAdmissionRollbackImplementationDryRun() ||
			!admissionLiveRouteTurnCandidateAdmissionLedgerImplementationDryRun() ||
			!admissionLiveRouteTurnCandidateAdmissionLedgerPersistenceDryRun() ||
			!admissionLiveRouteTurnCandidateAdmissionLedgerVerificationDryRun() ||
			!admissionLiveRouteTurnCandidateAdmissionReadinessDryRun() ||
			!admissionLiveRouteTurnCandidateAdmissionPermitDryRun() ||
			!admissionLiveRouteTurnCandidateAdmissionSealDryRun() ||
			!admissionLiveRouteTurnCandidateAdmissionFinalGateDryRun() ||
			!admissionLiveRouteTurnCandidateAdmissionResonanceIntentDryRun() ||
			!admissionLiveRouteTurnCandidateAdmissionResonanceReceiverDryRun() ||
			!admissionLiveRouteTurnCandidateAdmissionResonanceObservationDryRun() ||
			!admissionLiveRouteTurnCandidateAdmissionResonanceGraftBoundaryDryRun() ||
			!admissionLiveRouteTurnCandidateAdmissionResonanceGraftPreflightDryRun() ||
			!admissionLiveRouteTurnCandidateAdmissionResonanceGraftGateDryRun()) {
		return fmt.Errorf("AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_DECISION_DRY_RUN, AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_PROMOTION_DRY_RUN, AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_SWITCH_DRY_RUN, AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_ENABLE_GATE_DRY_RUN, AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_LIVE_STAGE_DRY_RUN, AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_WRITER_PREFLIGHT_DRY_RUN, AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_WRITER_INVENTORY_DRY_RUN, AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_WRITER_CONTRACT_DRY_RUN, AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_LEDGER_DRY_RUN, AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_WRITER_IMPLEMENTATION_DRY_RUN, AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_WRITER_RECEIPT_DRY_RUN, AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_ROLLBACK_IMPLEMENTATION_DRY_RUN, AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_LEDGER_IMPLEMENTATION_DRY_RUN, AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_LEDGER_PERSISTENCE_DRY_RUN, AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_LEDGER_VERIFICATION_DRY_RUN, AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_READINESS_DRY_RUN, AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_PERMIT_DRY_RUN, AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_SEAL_DRY_RUN, AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_FINAL_GATE_DRY_RUN, AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_RESONANCE_INTENT_DRY_RUN, AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_RESONANCE_RECEIVER_DRY_RUN, AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_RESONANCE_OBSERVATION_DRY_RUN, AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_RESONANCE_GRAFT_BOUNDARY_DRY_RUN, AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_RESONANCE_GRAFT_PREFLIGHT_DRY_RUN, and AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_RESONANCE_GRAFT_GATE_DRY_RUN are required for admission resonance graft candidate smoke")
	}
	if admissionLiveRouteTurnCandidateAdmissionResonanceGraftCandidateStoreDryRun() &&
		(!admissionLiveRouteTurnCandidateAdmissionDecisionDryRun() ||
			!admissionLiveRouteTurnCandidateAdmissionPromotionDryRun() ||
			!admissionLiveRouteTurnCandidateAdmissionSwitchDryRun() ||
			!admissionLiveRouteTurnCandidateAdmissionEnableGateDryRun() ||
			!admissionLiveRouteTurnCandidateAdmissionLiveStageDryRun() ||
			!admissionLiveRouteTurnCandidateAdmissionWriterPreflightDryRun() ||
			!admissionLiveRouteTurnCandidateAdmissionWriterInventoryDryRun() ||
			!admissionLiveRouteTurnCandidateAdmissionWriterContractDryRun() ||
			!admissionLiveRouteTurnCandidateAdmissionLedgerDryRun() ||
			!admissionLiveRouteTurnCandidateAdmissionWriterImplementationDryRun() ||
			!admissionLiveRouteTurnCandidateAdmissionWriterReceiptDryRun() ||
			!admissionLiveRouteTurnCandidateAdmissionRollbackImplementationDryRun() ||
			!admissionLiveRouteTurnCandidateAdmissionLedgerImplementationDryRun() ||
			!admissionLiveRouteTurnCandidateAdmissionLedgerPersistenceDryRun() ||
			!admissionLiveRouteTurnCandidateAdmissionLedgerVerificationDryRun() ||
			!admissionLiveRouteTurnCandidateAdmissionReadinessDryRun() ||
			!admissionLiveRouteTurnCandidateAdmissionPermitDryRun() ||
			!admissionLiveRouteTurnCandidateAdmissionSealDryRun() ||
			!admissionLiveRouteTurnCandidateAdmissionFinalGateDryRun() ||
			!admissionLiveRouteTurnCandidateAdmissionResonanceIntentDryRun() ||
			!admissionLiveRouteTurnCandidateAdmissionResonanceReceiverDryRun() ||
			!admissionLiveRouteTurnCandidateAdmissionResonanceObservationDryRun() ||
			!admissionLiveRouteTurnCandidateAdmissionResonanceGraftBoundaryDryRun() ||
			!admissionLiveRouteTurnCandidateAdmissionResonanceGraftPreflightDryRun() ||
			!admissionLiveRouteTurnCandidateAdmissionResonanceGraftGateDryRun() ||
			!admissionLiveRouteTurnCandidateAdmissionResonanceGraftCandidateDryRun()) {
		return fmt.Errorf("AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_DECISION_DRY_RUN, AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_PROMOTION_DRY_RUN, AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_SWITCH_DRY_RUN, AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_ENABLE_GATE_DRY_RUN, AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_LIVE_STAGE_DRY_RUN, AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_WRITER_PREFLIGHT_DRY_RUN, AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_WRITER_INVENTORY_DRY_RUN, AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_WRITER_CONTRACT_DRY_RUN, AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_LEDGER_DRY_RUN, AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_WRITER_IMPLEMENTATION_DRY_RUN, AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_WRITER_RECEIPT_DRY_RUN, AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_ROLLBACK_IMPLEMENTATION_DRY_RUN, AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_LEDGER_IMPLEMENTATION_DRY_RUN, AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_LEDGER_PERSISTENCE_DRY_RUN, AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_LEDGER_VERIFICATION_DRY_RUN, AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_READINESS_DRY_RUN, AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_PERMIT_DRY_RUN, AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_SEAL_DRY_RUN, AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_FINAL_GATE_DRY_RUN, AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_RESONANCE_INTENT_DRY_RUN, AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_RESONANCE_RECEIVER_DRY_RUN, AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_RESONANCE_OBSERVATION_DRY_RUN, AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_RESONANCE_GRAFT_BOUNDARY_DRY_RUN, AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_RESONANCE_GRAFT_PREFLIGHT_DRY_RUN, AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_RESONANCE_GRAFT_GATE_DRY_RUN, and AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_RESONANCE_GRAFT_CANDIDATE_DRY_RUN are required for admission resonance graft candidate store smoke")
	}
	if admissionLiveRouteTurnCandidateAdmissionResonanceGraftCandidateStoreReaderDryRun() &&
		(!admissionLiveRouteTurnCandidateAdmissionDecisionDryRun() ||
			!admissionLiveRouteTurnCandidateAdmissionPromotionDryRun() ||
			!admissionLiveRouteTurnCandidateAdmissionSwitchDryRun() ||
			!admissionLiveRouteTurnCandidateAdmissionEnableGateDryRun() ||
			!admissionLiveRouteTurnCandidateAdmissionLiveStageDryRun() ||
			!admissionLiveRouteTurnCandidateAdmissionWriterPreflightDryRun() ||
			!admissionLiveRouteTurnCandidateAdmissionWriterInventoryDryRun() ||
			!admissionLiveRouteTurnCandidateAdmissionWriterContractDryRun() ||
			!admissionLiveRouteTurnCandidateAdmissionLedgerDryRun() ||
			!admissionLiveRouteTurnCandidateAdmissionWriterImplementationDryRun() ||
			!admissionLiveRouteTurnCandidateAdmissionWriterReceiptDryRun() ||
			!admissionLiveRouteTurnCandidateAdmissionRollbackImplementationDryRun() ||
			!admissionLiveRouteTurnCandidateAdmissionLedgerImplementationDryRun() ||
			!admissionLiveRouteTurnCandidateAdmissionLedgerPersistenceDryRun() ||
			!admissionLiveRouteTurnCandidateAdmissionLedgerVerificationDryRun() ||
			!admissionLiveRouteTurnCandidateAdmissionReadinessDryRun() ||
			!admissionLiveRouteTurnCandidateAdmissionPermitDryRun() ||
			!admissionLiveRouteTurnCandidateAdmissionSealDryRun() ||
			!admissionLiveRouteTurnCandidateAdmissionFinalGateDryRun() ||
			!admissionLiveRouteTurnCandidateAdmissionResonanceIntentDryRun() ||
			!admissionLiveRouteTurnCandidateAdmissionResonanceReceiverDryRun() ||
			!admissionLiveRouteTurnCandidateAdmissionResonanceObservationDryRun() ||
			!admissionLiveRouteTurnCandidateAdmissionResonanceGraftBoundaryDryRun() ||
			!admissionLiveRouteTurnCandidateAdmissionResonanceGraftPreflightDryRun() ||
			!admissionLiveRouteTurnCandidateAdmissionResonanceGraftGateDryRun() ||
			!admissionLiveRouteTurnCandidateAdmissionResonanceGraftCandidateDryRun() ||
			!admissionLiveRouteTurnCandidateAdmissionResonanceGraftCandidateStoreDryRun()) {
		return fmt.Errorf("AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_DECISION_DRY_RUN, AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_PROMOTION_DRY_RUN, AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_SWITCH_DRY_RUN, AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_ENABLE_GATE_DRY_RUN, AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_LIVE_STAGE_DRY_RUN, AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_WRITER_PREFLIGHT_DRY_RUN, AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_WRITER_INVENTORY_DRY_RUN, AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_WRITER_CONTRACT_DRY_RUN, AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_LEDGER_DRY_RUN, AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_WRITER_IMPLEMENTATION_DRY_RUN, AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_WRITER_RECEIPT_DRY_RUN, AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_ROLLBACK_IMPLEMENTATION_DRY_RUN, AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_LEDGER_IMPLEMENTATION_DRY_RUN, AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_LEDGER_PERSISTENCE_DRY_RUN, AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_LEDGER_VERIFICATION_DRY_RUN, AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_READINESS_DRY_RUN, AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_PERMIT_DRY_RUN, AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_SEAL_DRY_RUN, AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_FINAL_GATE_DRY_RUN, AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_RESONANCE_INTENT_DRY_RUN, AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_RESONANCE_RECEIVER_DRY_RUN, AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_RESONANCE_OBSERVATION_DRY_RUN, AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_RESONANCE_GRAFT_BOUNDARY_DRY_RUN, AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_RESONANCE_GRAFT_PREFLIGHT_DRY_RUN, AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_RESONANCE_GRAFT_GATE_DRY_RUN, AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_RESONANCE_GRAFT_CANDIDATE_DRY_RUN, and AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_RESONANCE_GRAFT_CANDIDATE_STORE_DRY_RUN are required for admission resonance graft candidate store reader smoke")
	}
	if admissionLiveRouteTurnCandidateAdmissionResonanceGraftAdmissionProofDryRun() &&
		(!admissionLiveRouteTurnCandidateAdmissionDecisionDryRun() ||
			!admissionLiveRouteTurnCandidateAdmissionPromotionDryRun() ||
			!admissionLiveRouteTurnCandidateAdmissionSwitchDryRun() ||
			!admissionLiveRouteTurnCandidateAdmissionEnableGateDryRun() ||
			!admissionLiveRouteTurnCandidateAdmissionLiveStageDryRun() ||
			!admissionLiveRouteTurnCandidateAdmissionWriterPreflightDryRun() ||
			!admissionLiveRouteTurnCandidateAdmissionWriterInventoryDryRun() ||
			!admissionLiveRouteTurnCandidateAdmissionWriterContractDryRun() ||
			!admissionLiveRouteTurnCandidateAdmissionLedgerDryRun() ||
			!admissionLiveRouteTurnCandidateAdmissionWriterImplementationDryRun() ||
			!admissionLiveRouteTurnCandidateAdmissionWriterReceiptDryRun() ||
			!admissionLiveRouteTurnCandidateAdmissionRollbackImplementationDryRun() ||
			!admissionLiveRouteTurnCandidateAdmissionLedgerImplementationDryRun() ||
			!admissionLiveRouteTurnCandidateAdmissionLedgerPersistenceDryRun() ||
			!admissionLiveRouteTurnCandidateAdmissionLedgerVerificationDryRun() ||
			!admissionLiveRouteTurnCandidateAdmissionReadinessDryRun() ||
			!admissionLiveRouteTurnCandidateAdmissionPermitDryRun() ||
			!admissionLiveRouteTurnCandidateAdmissionSealDryRun() ||
			!admissionLiveRouteTurnCandidateAdmissionFinalGateDryRun() ||
			!admissionLiveRouteTurnCandidateAdmissionResonanceIntentDryRun() ||
			!admissionLiveRouteTurnCandidateAdmissionResonanceReceiverDryRun() ||
			!admissionLiveRouteTurnCandidateAdmissionResonanceObservationDryRun() ||
			!admissionLiveRouteTurnCandidateAdmissionResonanceGraftBoundaryDryRun() ||
			!admissionLiveRouteTurnCandidateAdmissionResonanceGraftPreflightDryRun() ||
			!admissionLiveRouteTurnCandidateAdmissionResonanceGraftGateDryRun() ||
			!admissionLiveRouteTurnCandidateAdmissionResonanceGraftCandidateDryRun() ||
			!admissionLiveRouteTurnCandidateAdmissionResonanceGraftCandidateStoreDryRun() ||
			!admissionLiveRouteTurnCandidateAdmissionResonanceGraftCandidateStoreReaderDryRun()) {
		return fmt.Errorf("AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_DECISION_DRY_RUN, AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_PROMOTION_DRY_RUN, AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_SWITCH_DRY_RUN, AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_ENABLE_GATE_DRY_RUN, AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_LIVE_STAGE_DRY_RUN, AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_WRITER_PREFLIGHT_DRY_RUN, AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_WRITER_INVENTORY_DRY_RUN, AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_WRITER_CONTRACT_DRY_RUN, AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_LEDGER_DRY_RUN, AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_WRITER_IMPLEMENTATION_DRY_RUN, AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_WRITER_RECEIPT_DRY_RUN, AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_ROLLBACK_IMPLEMENTATION_DRY_RUN, AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_LEDGER_IMPLEMENTATION_DRY_RUN, AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_LEDGER_PERSISTENCE_DRY_RUN, AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_LEDGER_VERIFICATION_DRY_RUN, AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_READINESS_DRY_RUN, AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_PERMIT_DRY_RUN, AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_SEAL_DRY_RUN, AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_FINAL_GATE_DRY_RUN, AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_RESONANCE_INTENT_DRY_RUN, AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_RESONANCE_RECEIVER_DRY_RUN, AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_RESONANCE_OBSERVATION_DRY_RUN, AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_RESONANCE_GRAFT_BOUNDARY_DRY_RUN, AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_RESONANCE_GRAFT_PREFLIGHT_DRY_RUN, AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_RESONANCE_GRAFT_GATE_DRY_RUN, AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_RESONANCE_GRAFT_CANDIDATE_DRY_RUN, AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_RESONANCE_GRAFT_CANDIDATE_STORE_DRY_RUN, and AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_RESONANCE_GRAFT_CANDIDATE_STORE_READER_DRY_RUN are required for admission resonance graft admission proof smoke")
	}
	if admissionLiveRouteTurnCandidateAdmissionEnableGateDryRun() &&
		!admissionLiveRouteTurnCandidateAdmissionLiveStageDryRun() &&
		admissionLiveRouteTurnCandidateAdmissionEnableGateKey() != "" {
		return fmt.Errorf("AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_ENABLE_GATE_KEY must be empty for default-off enable gate smoke")
	}
	if (admissionLiveRouteTurnCandidateAdmissionLiveStageDryRun() ||
		admissionLiveRouteTurnCandidateAdmissionWriterPreflightDryRun() ||
		admissionLiveRouteTurnCandidateAdmissionWriterInventoryDryRun() ||
		admissionLiveRouteTurnCandidateAdmissionWriterContractDryRun() ||
		admissionLiveRouteTurnCandidateAdmissionLedgerDryRun() ||
		admissionLiveRouteTurnCandidateAdmissionWriterImplementationDryRun() ||
		admissionLiveRouteTurnCandidateAdmissionWriterReceiptDryRun() ||
		admissionLiveRouteTurnCandidateAdmissionRollbackImplementationDryRun() ||
		admissionLiveRouteTurnCandidateAdmissionLedgerImplementationDryRun() ||
		admissionLiveRouteTurnCandidateAdmissionLedgerPersistenceDryRun() ||
		admissionLiveRouteTurnCandidateAdmissionLedgerVerificationDryRun() ||
		admissionLiveRouteTurnCandidateAdmissionReadinessDryRun() ||
		admissionLiveRouteTurnCandidateAdmissionPermitDryRun() ||
		admissionLiveRouteTurnCandidateAdmissionSealDryRun() ||
		admissionLiveRouteTurnCandidateAdmissionFinalGateDryRun() ||
		admissionLiveRouteTurnCandidateAdmissionResonanceIntentDryRun() ||
		admissionLiveRouteTurnCandidateAdmissionResonanceReceiverDryRun() ||
		admissionLiveRouteTurnCandidateAdmissionResonanceObservationDryRun() ||
		admissionLiveRouteTurnCandidateAdmissionResonanceGraftBoundaryDryRun() ||
		admissionLiveRouteTurnCandidateAdmissionResonanceGraftPreflightDryRun() ||
		admissionLiveRouteTurnCandidateAdmissionResonanceGraftGateDryRun() ||
		admissionLiveRouteTurnCandidateAdmissionResonanceGraftCandidateDryRun() ||
		admissionLiveRouteTurnCandidateAdmissionResonanceGraftCandidateStoreDryRun() ||
		admissionLiveRouteTurnCandidateAdmissionResonanceGraftCandidateStoreReaderDryRun() ||
		admissionLiveRouteTurnCandidateAdmissionResonanceGraftAdmissionProofDryRun()) &&
		admissionLiveRouteTurnCandidateAdmissionEnableGateKey() != admissionLiveRouteTurnCandidateAdmissionEnableGateConfirmation {
		return fmt.Errorf("AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_ENABLE_GATE_KEY must match dry-run confirmation for live admission stage smoke")
	}
	if (admissionLiveRouteTurnCandidateAdmissionPermitDryRun() ||
		admissionLiveRouteTurnCandidateAdmissionSealDryRun() ||
		admissionLiveRouteTurnCandidateAdmissionFinalGateDryRun() ||
		admissionLiveRouteTurnCandidateAdmissionResonanceIntentDryRun() ||
		admissionLiveRouteTurnCandidateAdmissionResonanceReceiverDryRun() ||
		admissionLiveRouteTurnCandidateAdmissionResonanceObservationDryRun() ||
		admissionLiveRouteTurnCandidateAdmissionResonanceGraftBoundaryDryRun() ||
		admissionLiveRouteTurnCandidateAdmissionResonanceGraftPreflightDryRun() ||
		admissionLiveRouteTurnCandidateAdmissionResonanceGraftGateDryRun() ||
		admissionLiveRouteTurnCandidateAdmissionResonanceGraftCandidateDryRun() ||
		admissionLiveRouteTurnCandidateAdmissionResonanceGraftCandidateStoreDryRun() ||
		admissionLiveRouteTurnCandidateAdmissionResonanceGraftCandidateStoreReaderDryRun() ||
		admissionLiveRouteTurnCandidateAdmissionResonanceGraftAdmissionProofDryRun()) &&
		admissionLiveRouteTurnCandidateAdmissionPermitKey() != admissionLiveRouteTurnCandidateAdmissionPermitConfirmation {
		return fmt.Errorf("AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_PERMIT_KEY must match dry-run confirmation for admission permit smoke")
	}
	if !admissionLiveRouteTurnCandidateExecutionDryRun() {
		return fmt.Errorf("AM_LIVE_ROUTE_TURN_CANDIDATE_EXECUTION_DRY_RUN is required")
	}
	if !admissionLiveRouteTurnCandidateExecutionRunnerDryRun() {
		return fmt.Errorf("AM_LIVE_ROUTE_TURN_CANDIDATE_EXECUTION_RUNNER_DRY_RUN is required")
	}
	if runner := admissionLiveRouteTurnCandidateExecutionRunnerName(); runner != admissionLiveRouteTurnCandidateExecutionRunnerNanoDirect {
		return fmt.Errorf("AM_LIVE_ROUTE_TURN_CANDIDATE_EXECUTION_RUNNER=%q, want %q", runner, admissionLiveRouteTurnCandidateExecutionRunnerNanoDirect)
	}
	if !admissionLiveRouteTurnGeneratorAdapterDryRun() {
		return fmt.Errorf("AM_LIVE_ROUTE_TURN_GENERATOR_ADAPTER_DRY_RUN is required")
	}
	if !admissionLiveRouteTurnCandidateDraftDryRun() {
		return fmt.Errorf("AM_LIVE_ROUTE_TURN_CANDIDATE_DRAFT_DRY_RUN is required")
	}
	if !admissionLiveRouteTurnCandidateAdmissionDryRun() {
		return fmt.Errorf("AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_DRY_RUN is required")
	}
	if !admissionLiveRouteTurnCandidateAdmissionAdapterDryRun() {
		return fmt.Errorf("AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_ADAPTER_DRY_RUN is required")
	}
	if !admissionLiveRouteTurnCandidateAdmissionShadowDryRun() {
		return fmt.Errorf("AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_SHADOW_DRY_RUN is required")
	}
	if mode := dreamAdmissionMode(); mode != dreamAdmissionShadow {
		return fmt.Errorf("AM_DREAM_ADMISSION=%q, want %q", mode, dreamAdmissionShadow)
	}
	if !dreamAdmissionRequireLiveRoutePlan() {
		return fmt.Errorf("AM_DREAM_ADMISSION_REQUIRE_LIVE_ROUTE_PLAN is required")
	}
	if strings.TrimSpace(os.Getenv("AM_LIVE_ROUTE_TURN_CANDIDATE_DRAFT_TEXT")) != "" {
		return fmt.Errorf("AM_LIVE_ROUTE_TURN_CANDIDATE_DRAFT_TEXT must be empty so nano-direct owns the candidate text")
	}
	if strings.TrimSpace(os.Getenv("AM_LIVE_ROUTE_TURN_GENERATOR_ADAPTER_TEXT")) != "" {
		return fmt.Errorf("AM_LIVE_ROUTE_TURN_GENERATOR_ADAPTER_TEXT must be empty so nano-direct owns the candidate text")
	}
	if strings.TrimSpace(os.Getenv("AM_LIVE_ROUTE_TURN_CANDIDATE_EXECUTION_TEXT")) == "" {
		return fmt.Errorf("AM_LIVE_ROUTE_TURN_CANDIDATE_EXECUTION_TEXT is required")
	}

	obs := admissionLiveRouteTurnObservationForHuman("Tell me what the dream should remember.")
	lines := chatLiveRouteTurnCandidateChainDryRunLines(obs)
	wantLines := 6
	if admissionLiveRouteTurnCandidateAdmissionDecisionDryRun() {
		wantLines++
	}
	if admissionLiveRouteTurnCandidateAdmissionPromotionDryRun() {
		wantLines++
	}
	if admissionLiveRouteTurnCandidateAdmissionSwitchDryRun() {
		wantLines++
	}
	if admissionLiveRouteTurnCandidateAdmissionEnableGateDryRun() {
		wantLines++
	}
	if admissionLiveRouteTurnCandidateAdmissionLiveStageDryRun() {
		wantLines++
	}
	if admissionLiveRouteTurnCandidateAdmissionWriterPreflightDryRun() {
		wantLines++
	}
	if admissionLiveRouteTurnCandidateAdmissionWriterInventoryDryRun() {
		wantLines++
	}
	if admissionLiveRouteTurnCandidateAdmissionWriterContractDryRun() {
		wantLines++
	}
	if admissionLiveRouteTurnCandidateAdmissionLedgerDryRun() {
		wantLines++
	}
	if admissionLiveRouteTurnCandidateAdmissionWriterImplementationDryRun() {
		wantLines++
	}
	if admissionLiveRouteTurnCandidateAdmissionWriterReceiptDryRun() {
		wantLines++
	}
	if admissionLiveRouteTurnCandidateAdmissionRollbackImplementationDryRun() {
		wantLines++
	}
	if admissionLiveRouteTurnCandidateAdmissionLedgerImplementationDryRun() {
		wantLines++
	}
	if admissionLiveRouteTurnCandidateAdmissionLedgerPersistenceDryRun() {
		wantLines++
	}
	if admissionLiveRouteTurnCandidateAdmissionLedgerVerificationDryRun() {
		wantLines++
	}
	if admissionLiveRouteTurnCandidateAdmissionReadinessDryRun() {
		wantLines++
	}
	if admissionLiveRouteTurnCandidateAdmissionPermitDryRun() {
		wantLines++
	}
	if admissionLiveRouteTurnCandidateAdmissionSealDryRun() {
		wantLines++
	}
	if admissionLiveRouteTurnCandidateAdmissionFinalGateDryRun() {
		wantLines++
	}
	if admissionLiveRouteTurnCandidateAdmissionResonanceIntentDryRun() {
		wantLines++
	}
	if admissionLiveRouteTurnCandidateAdmissionResonanceReceiverDryRun() {
		wantLines++
	}
	if admissionLiveRouteTurnCandidateAdmissionResonanceObservationDryRun() {
		wantLines++
	}
	if admissionLiveRouteTurnCandidateAdmissionResonanceGraftBoundaryDryRun() {
		wantLines++
	}
	if admissionLiveRouteTurnCandidateAdmissionResonanceGraftPreflightDryRun() {
		wantLines++
	}
	if admissionLiveRouteTurnCandidateAdmissionResonanceGraftGateDryRun() {
		wantLines++
	}
	if admissionLiveRouteTurnCandidateAdmissionResonanceGraftCandidateDryRun() {
		wantLines++
	}
	if admissionLiveRouteTurnCandidateAdmissionResonanceGraftCandidateStoreDryRun() {
		wantLines++
	}
	if admissionLiveRouteTurnCandidateAdmissionResonanceGraftCandidateStoreReaderDryRun() {
		wantLines++
	}
	if admissionLiveRouteTurnCandidateAdmissionResonanceGraftAdmissionProofDryRun() {
		wantLines++
	}
	if len(lines) != wantLines {
		return fmt.Errorf("expected %d nano-direct chat-shadow lines, got %d: %v", wantLines, len(lines), lines)
	}
	wants := []string{
		"live-route candidate execution dry-run: class=dream route=direct backend=nano-arianna entry=direct frame=q_a",
		"runner=nano-direct runner_status=succeeded passed=true",
		"live-route generator adapter dry-run: class=dream route=direct backend=nano-arianna entry=direct frame=q_a shell=shell-",
		"live-route candidate draft dry-run: class=dream route=direct source=direct trigger=direct-dream seed=turn-",
		"live-route candidate admission handoff dry-run: class=dream route=direct source=direct draft=draft-",
		"live-route candidate admission adapter dry-run: class=dream route=direct source=direct handoff=handoff-",
		"live-route candidate admission shadow dry-run: class=dream route=direct source=direct handoff=handoff-",
		"policy=true accepted=false passed=true reason=shadow mode",
	}
	if admissionLiveRouteTurnCandidateAdmissionDecisionDryRun() {
		wants = append(wants,
			"live-route candidate admission decision dry-run: class=dream route=direct source=direct handoff=handoff-",
			"decision=shadow_ready decision_id=decision-",
			"live_ready=true mutates=false passed=true reason=shadow ready; live mutation still disabled",
		)
	}
	if admissionLiveRouteTurnCandidateAdmissionPromotionDryRun() {
		wants = append(wants,
			"live-route candidate admission promotion dry-run: class=dream route=direct source=direct decision=shadow_ready decision_id=decision-",
			"promotion=pending_live_admission promotion_id=promotion-",
			"live_ready=true live_enabled=false mutates=false passed=true reason=shadow decision consumed; live admission still disabled",
		)
	}
	if admissionLiveRouteTurnCandidateAdmissionSwitchDryRun() {
		wants = append(wants,
			"live-route candidate admission switch dry-run: class=dream route=direct source=direct promotion=pending_live_admission promotion_id=promotion-",
			"switch=disabled switch_action=hold_pending_live_admission switch_id=switch-",
			"admission_allowed=false live_ready=true live_enabled=false mutates=false passed=true reason=live admission switch disabled; pending promotion held without mutation",
		)
	}
	if admissionLiveRouteTurnCandidateAdmissionEnableGateDryRun() {
		if admissionLiveRouteTurnCandidateAdmissionLiveStageDryRun() {
			wants = append(wants,
				"live-route candidate admission enable gate dry-run: class=dream route=direct source=direct switch=disabled switch_id=switch-",
				"enable=armed_dry_run enable_action=would_enable_live_admission_dry_run enable_id=enable-",
				"admission_allowed=false manual_enable=true key_matched=true live_ready=true live_enabled=false mutates=false passed=true reason=live admission enable key matched; dry-run still refuses mutation",
			)
		} else {
			wants = append(wants,
				"live-route candidate admission enable gate dry-run: class=dream route=direct source=direct switch=disabled switch_id=switch-",
				"enable=disabled enable_action=require_operator_key enable_id=enable-",
				"admission_allowed=false manual_enable=false key_matched=false live_ready=true live_enabled=false mutates=false passed=true reason=live admission enable gate closed; operator key absent",
			)
		}
	}
	if admissionLiveRouteTurnCandidateAdmissionLiveStageDryRun() {
		wants = append(wants,
			"live-route candidate admission live stage dry-run: class=dream route=direct source=direct enable=armed_dry_run enable_id=enable-",
			"stage=staged_dry_run stage_action=stage_live_candidate_dry_run stage_id=stage-",
			"admission_allowed=false writer_ready=false rollback_ready=false live_ready=true live_enabled=false mutates=false passed=true reason=live admission candidate staged as dry-run; writer and rollback remain absent",
		)
	}
	if admissionLiveRouteTurnCandidateAdmissionWriterPreflightDryRun() {
		wants = append(wants,
			"live-route candidate admission writer preflight dry-run: class=dream route=direct source=direct stage=staged_dry_run stage_id=stage-",
			"writer=absent writer_action=require_writer_contract rollback=absent rollback_action=require_rollback_contract writer_preflight_id=writer-",
			"write_allowed=false admission_allowed=false live_ready=true live_enabled=false mutates=false passed=true reason=writer and rollback absent; live admission remains staged only",
		)
	}
	if admissionLiveRouteTurnCandidateAdmissionWriterInventoryDryRun() {
		wants = append(wants,
			"live-route candidate admission writer inventory dry-run: class=dream route=direct source=direct writer_preflight=writer-",
			"inventory=contracts_absent inventory_action=name_required_contracts writer_contract=live_admission_writer.v1 rollback_contract=live_admission_rollback.v1 ledger_contract=live_admission_ledger.v1",
			"contracts_ready=false write_allowed=false admission_allowed=false live_ready=true live_enabled=false mutates=false writer_inventory_id=writer-inventory-",
			"passed=true reason=writer inventory recorded required contracts; live admission remains blocked",
		)
	}
	if admissionLiveRouteTurnCandidateAdmissionWriterContractDryRun() {
		wants = append(wants,
			"live-route candidate admission writer contract dry-run: class=dream route=direct source=direct writer_inventory=writer-inventory-",
			"contract=shape_drafted_dry_run contract_action=define_writer_rollback_ledger_contract writer_contract=live_admission_writer.v1 rollback_contract=live_admission_rollback.v1 ledger_contract=live_admission_ledger.v1",
			"writer_shape=append_shadow_candidate_receipt rollback_shape=remove_exact_writer_receipt ledger_shape=append_only_receipt_log",
			"shape_ready=true writer_impl=false rollback_impl=false ledger_impl=false contracts_ready=false write_allowed=false admission_allowed=false live_ready=true live_enabled=false mutates=false writer_contract_id=writer-contract-",
			"passed=true reason=writer contract shape drafted; implementation and ledger remain absent",
		)
	}
	if admissionLiveRouteTurnCandidateAdmissionLedgerDryRun() {
		wants = append(wants,
			"live-route candidate admission ledger dry-run: class=dream route=direct source=direct writer_contract=writer-contract-",
			"ledger=receipt_drafted_dry_run ledger_action=append_candidate_admission_receipt_dry_run ledger_contract=live_admission_ledger.v1 ledger_mode=append_only_dry_run",
			"ledger_entry=dream_candidate_admission entry_status=shadow_candidate_receipt receipt_shape=candidate_contract_provenance",
			"append_ready=true persisted=false ledger_impl=false contracts_ready=false write_allowed=false admission_allowed=false live_ready=true live_enabled=false mutates=false admission_ledger_id=admission-ledger-",
			"passed=true reason=admission ledger dry-run receipt drafted; no live write occurred",
		)
	}
	if admissionLiveRouteTurnCandidateAdmissionWriterImplementationDryRun() {
		wants = append(wants,
			"live-route candidate admission writer implementation dry-run: class=dream route=direct source=direct ledger=admission-ledger-",
			"implementation=implementation_contract_drafted_dry_run implementation_action=define_append_only_writer_ledger_rollback writer_entrypoint=append_shadow_candidate_receipt_dry_run ledger_entrypoint=append_admission_ledger_receipt_dry_run rollback_entrypoint=remove_exact_shadow_candidate_receipt_dry_run",
			"write_target=shadow_receipt_log body_target=none append_only=true rollback_required=true implementation_contract=true",
			"writer_impl=false ledger_impl=false rollback_impl=false contracts_ready=false write_allowed=false admission_allowed=false live_ready=true live_enabled=false mutates=false writer_implementation_id=writer-implementation-",
			"passed=true reason=writer implementation contract drafted; append-only log boundary only",
		)
	}
	if admissionLiveRouteTurnCandidateAdmissionWriterReceiptDryRun() {
		wants = append(wants,
			"live-route candidate admission writer receipt dry-run: class=dream route=direct source=direct writer_implementation=writer-implementation-",
			"writer_receipt=shadow_receipt_appended_dry_run receipt_action=append_shadow_candidate_receipt_dry_run receipt_kind=dream_candidate_admission receipt_target=shadow_receipt_log receipt_mode=append_only_dry_run receipt_shape=candidate_contract_provenance",
			"receipt_persisted=true shadow_write_allowed=true body_target=none append_only=true rollback_required=true writer_ready=true writer_impl=true ledger_impl=false rollback_impl=false",
			"contracts_ready=false write_allowed=false admission_allowed=false live_ready=true live_enabled=false mutates=false writer_receipt_id=writer-receipt-",
			"passed=true reason=shadow writer receipt appended as dry-run; body write remains disabled",
		)
	}
	if admissionLiveRouteTurnCandidateAdmissionRollbackImplementationDryRun() {
		wants = append(wants,
			"live-route candidate admission rollback implementation dry-run: class=dream route=direct source=direct writer_receipt=writer-receipt-",
			"rollback=rollback_contract_drafted_dry_run rollback_action=remove_exact_shadow_candidate_receipt_dry_run rollback_entrypoint=remove_exact_shadow_candidate_receipt_dry_run",
			"rollback_target=shadow_receipt_log rollback_target_kind=dream_candidate_admission rollback_target_id=writer-receipt-",
			"rollback_mode=exact_receipt_id_dry_run exact_match=true dry_run_only=true receipt_removed=false",
			"exact_match=true dry_run_only=true receipt_removed=false writer_ready=true rollback_ready=true writer_impl=true rollback_impl=true ledger_impl=false",
			"contracts_ready=false write_allowed=false admission_allowed=false live_ready=true live_enabled=false mutates=false rollback_implementation_id=rollback-implementation-",
			"passed=true reason=rollback implementation drafted for exact writer receipt; body write remains disabled",
		)
	}
	if admissionLiveRouteTurnCandidateAdmissionLedgerImplementationDryRun() {
		wants = append(wants,
			"live-route candidate admission ledger implementation dry-run: class=dream route=direct source=direct rollback_implementation=rollback-implementation-",
			"ledger=ledger_contract_drafted_dry_run ledger_action=append_admission_ledger_receipt_dry_run ledger_entrypoint=append_admission_ledger_receipt_dry_run",
			"ledger_target=admission_ledger ledger_target_kind=dream_candidate_admission ledger_target_mode=append_only_dry_run",
			"append_only=true dry_run_only=true receipt_persisted=false writer_ready=true rollback_ready=true writer_impl=true rollback_impl=true ledger_impl=true",
			"contracts_ready=false write_allowed=false admission_allowed=false live_ready=true live_enabled=false mutates=false ledger_implementation_id=ledger-implementation-",
			"passed=true reason=ledger implementation drafted for append-only admission receipts; contracts remain disabled",
		)
	}
	if admissionLiveRouteTurnCandidateAdmissionLedgerPersistenceDryRun() {
		wants = append(wants,
			"live-route candidate admission ledger persistence dry-run: class=dream route=direct source=direct ledger_implementation=ledger-implementation-",
			"admission_ledger=admission-ledger-",
			"writer_receipt=writer-receipt-",
			"rollback_implementation=rollback-implementation-",
			"persistence=ledger_receipt_persisted_dry_run persistence_action=append_admission_ledger_receipt_dry_run",
			"persistence_target=admission_ledger persistence_target_kind=dream_candidate_admission persistence_target_mode=append_only_dry_run receipt_shape=candidate_contract_provenance",
			"append_only=true dry_run_only=true receipt_persisted=true persistence_ready=true writer_ready=true rollback_ready=true writer_impl=true rollback_impl=true ledger_impl=true",
			"contracts_ready=false write_allowed=false admission_allowed=false live_ready=true live_enabled=false mutates=false ledger_persistence_id=ledger-persistence-",
			"passed=true reason=ledger receipt persisted to append-only dry-run log; live admission remains disabled",
		)
	}
	if admissionLiveRouteTurnCandidateAdmissionLedgerVerificationDryRun() {
		wants = append(wants,
			"live-route candidate admission ledger verification dry-run: class=dream route=direct source=direct ledger_persistence=ledger-persistence-",
			"ledger_implementation=ledger-implementation-",
			"admission_ledger=admission-ledger-",
			"writer_receipt=writer-receipt-",
			"rollback_implementation=rollback-implementation-",
			"verification=ledger_receipt_verified_dry_run verification_action=verify_persisted_admission_ledger_receipt_dry_run",
			"verification_target=admission_ledger verification_target_kind=dream_candidate_admission verification_target_mode=append_only_dry_run receipt_shape=candidate_contract_provenance",
			"append_only=true dry_run_only=true read_back=true receipt_verified=true verification_ready=true persistence_ready=true writer_ready=true rollback_ready=true writer_impl=true rollback_impl=true ledger_impl=true",
			"contracts_ready=false write_allowed=false admission_allowed=false live_ready=true live_enabled=false mutates=false ledger_verification_id=ledger-verification-",
			"passed=true reason=ledger persistence receipt verified by read-back dry-run; live admission remains disabled",
		)
	}
	if admissionLiveRouteTurnCandidateAdmissionReadinessDryRun() {
		wants = append(wants,
			"live-route candidate admission readiness dry-run: class=dream route=direct source=direct ledger_verification=ledger-verification-",
			"ledger_persistence=ledger-persistence-",
			"ledger_implementation=ledger-implementation-",
			"admission_ledger=admission-ledger-",
			"writer_receipt=writer-receipt-",
			"rollback_implementation=rollback-implementation-",
			"readiness=verified_closed_dry_run readiness_action=declare_verified_live_admission_readiness_dry_run",
			"readiness_target=live_admission readiness_target_kind=dream_candidate_admission readiness_target_mode=closed_verified_dry_run",
			"dry_run_only=true ledger_verified=true writer_ready=true rollback_ready=true ledger_ready=true readiness_ready=true verification_ready=true persistence_ready=true writer_impl=true rollback_impl=true ledger_impl=true",
			"contracts_ready=false write_allowed=false admission_allowed=false live_ready=true live_enabled=false mutates=false admission_readiness_id=admission-readiness-",
			"passed=true reason=verified ledger and writer boundaries are ready; live admission remains disabled",
		)
	}
	if admissionLiveRouteTurnCandidateAdmissionPermitDryRun() {
		wants = append(wants,
			"live-route candidate admission permit dry-run: class=dream route=direct source=direct readiness=admission-readiness-",
			"ledger_verification=ledger-verification-",
			"ledger_persistence=ledger-persistence-",
			"ledger_implementation=ledger-implementation-",
			"admission_ledger=admission-ledger-",
			"writer_receipt=writer-receipt-",
			"rollback_implementation=rollback-implementation-",
			"permit=operator_permitted_closed_dry_run permit_action=acknowledge_verified_live_admission_readiness_dry_run",
			"permit_target=live_admission permit_target_kind=dream_candidate_admission permit_target_mode=permit_closed_dry_run",
			"dry_run_only=true readiness_verified=true ledger_verified=true writer_ready=true rollback_ready=true ledger_ready=true permit_ready=true manual_requested=true key_matched=true readiness_ready=true verification_ready=true persistence_ready=true writer_impl=true rollback_impl=true ledger_impl=true",
			"contracts_ready=false write_allowed=false admission_allowed=false live_ready=true live_enabled=false mutates=false admission_permit_id=admission-permit-",
			"passed=true reason=operator permit accepted for verified readiness; live admission remains disabled",
		)
	}
	if admissionLiveRouteTurnCandidateAdmissionSealDryRun() {
		wants = append(wants,
			"live-route candidate admission seal dry-run: class=dream route=direct source=direct permit=admission-permit-",
			"readiness=admission-readiness-",
			"ledger_verification=ledger-verification-",
			"ledger_persistence=ledger-persistence-",
			"ledger_implementation=ledger-implementation-",
			"admission_ledger=admission-ledger-",
			"writer_receipt=writer-receipt-",
			"rollback_implementation=rollback-implementation-",
			"seal=sealed_closed_dry_run seal_action=seal_operator_permit_provenance_dry_run",
			"seal_target=live_admission seal_target_kind=dream_candidate_admission seal_target_mode=sealed_closed_dry_run receipt_shape=candidate_contract_provenance",
			"dry_run_only=true permit_verified=true readiness_verified=true ledger_verified=true writer_ready=true rollback_ready=true ledger_ready=true seal_ready=true permit_ready=true key_matched=true readiness_ready=true verification_ready=true persistence_ready=true writer_impl=true rollback_impl=true ledger_impl=true",
			"contracts_ready=false write_allowed=false admission_allowed=false live_ready=true live_enabled=false mutates=false admission_seal_id=admission-seal-",
			"passed=true reason=operator permit sealed as immutable dry-run receipt; live admission remains disabled",
		)
	}
	if admissionLiveRouteTurnCandidateAdmissionFinalGateDryRun() {
		wants = append(wants,
			"live-route candidate admission final gate dry-run: class=dream route=direct source=direct seal=admission-seal-",
			"permit=admission-permit-",
			"readiness=admission-readiness-",
			"ledger_verification=ledger-verification-",
			"ledger_persistence=ledger-persistence-",
			"ledger_implementation=ledger-implementation-",
			"admission_ledger=admission-ledger-",
			"writer_receipt=writer-receipt-",
			"rollback_implementation=rollback-implementation-",
			"final_gate=ready_closed_dry_run final_gate_action=verify_sealed_admission_provenance_dry_run",
			"final_gate_target=live_admission final_gate_target_kind=dream_candidate_admission final_gate_target_mode=final_gate_closed_dry_run receipt_shape=sealed_candidate_contract_provenance",
			"dry_run_only=true seal_verified=true permit_verified=true readiness_verified=true ledger_verified=true writer_ready=true rollback_ready=true ledger_ready=true final_gate_ready=true seal_ready=true permit_ready=true key_matched=true readiness_ready=true verification_ready=true persistence_ready=true writer_impl=true rollback_impl=true ledger_impl=true",
			"contracts_ready=false write_allowed=false admission_allowed=false live_ready=true live_enabled=false mutates=false admission_final_gate_id=admission-final-gate-",
			"passed=true reason=sealed admission provenance cleared final gate; live admission remains disabled",
		)
	}
	if admissionLiveRouteTurnCandidateAdmissionResonanceIntentDryRun() {
		wants = append(wants,
			"live-route candidate admission resonance intent dry-run: class=dream route=direct source=direct final_gate=admission-final-gate-",
			"seal=admission-seal-",
			"permit=admission-permit-",
			"readiness=admission-readiness-",
			"ledger_verification=ledger-verification-",
			"receiver=resonance receiver_kind=internal_world influence_kind=bounded_direction max_influence=0.05 ttl_turns=1 causal_id=resonance-intent-causal-",
			"raw_text_allowed=false janus_surface_allowed=false cooc_learning_allowed=false delta_harvest_allowed=false rollback_required=true pre_hash_required=true post_hash_required=true",
			"intent=resonance_intent_drafted_dry_run intent_action=draft_resonance_direction_intent_dry_run",
			"intent_target=resonance intent_target_kind=first_live_receiver intent_target_mode=bounded_direction_dry_run receipt_shape=sealed_candidate_contract_provenance",
			"dry_run_only=true final_gate_verified=true seal_verified=true permit_verified=true readiness_verified=true ledger_verified=true writer_ready=true rollback_ready=true ledger_ready=true intent_ready=true",
			"contracts_ready=false write_allowed=false admission_allowed=false live_ready=true live_enabled=false mutates=false admission_resonance_intent_id=resonance-intent-",
			"passed=true reason=resonance intent drafted from final gate; live admission remains disabled",
		)
	}
	if admissionLiveRouteTurnCandidateAdmissionResonanceObservationDryRun() {
		wants = append(wants,
			"live-route candidate admission resonance observation dry-run: class=dream route=direct source=direct receiver=resonance-receiver-",
			"intent=resonance-intent-",
			"final_gate=admission-final-gate-",
			"observer=resonance observer_kind=internal_world observation_kind=receiver_state_proof observation_mode=sealed_metadata_observation causal_id=resonance-observation-causal-",
			"append_hash=resonance-observation-append-",
			"read_back_hash=resonance-observation-read-",
			"source_receiver_causal_id=resonance-receiver-causal-",
			"source_receiver_delta_hash=resonance-receiver-delta-",
			"append_only=true read_back=true receipt_verified=true raw_text_observed=false raw_text_forwarded=false janus_surface_allowed=false cooc_learning_allowed=false delta_harvest_allowed=false body_mutation_allowed=false rollback_required=true",
			"observation_state=observation_recorded_dry_run observation_action=record_resonance_receiver_observation_dry_run observation_target=resonance observation_target_kind=internal_world_observation observation_target_mode=append_only_read_back_dry_run receipt_shape=resonance_receiver_state_proof_ledger",
			"dry_run_only=true receiver_verified=true intent_verified=true final_gate_verified=true seal_verified=true permit_verified=true readiness_verified=true ledger_verified=true writer_ready=true rollback_ready=true ledger_ready=true observation_ready=true",
			"contracts_ready=false write_allowed=false admission_allowed=false live_ready=true live_enabled=false mutates=false admission_resonance_observation_id=resonance-observation-",
			"passed=true reason=resonance observation recorded and read back without body mutation",
		)
	}
	if admissionLiveRouteTurnCandidateAdmissionResonanceGraftBoundaryDryRun() {
		wants = append(wants,
			"live-route candidate admission resonance graft boundary dry-run: class=dream route=direct source=direct observation=resonance-observation-",
			"receiver=resonance-receiver-",
			"intent=resonance-intent-",
			"final_gate=admission-final-gate-",
			"boundary_kind=shadow_graft_boundary boundary_mode=no_mutation_receipt boundary_stage=pre_live_graft causal_id=resonance-graft-boundary-causal-",
			"boundary_hash=resonance-graft-boundary-",
			"read_back_hash=resonance-graft-boundary-read-",
			"source_observation_causal_id=resonance-observation-causal-",
			"source_observation_read_back_hash=resonance-observation-read-",
			"shadow_only=true graft_allowed=false raw_text_allowed=false janus_surface_allowed=false cooc_learning_allowed=false delta_harvest_allowed=false body_mutation_allowed=false rollback_required=true",
			"boundary_state=shadow_graft_boundary_declared_dry_run boundary_action=declare_resonance_shadow_graft_boundary_dry_run boundary_target=resonance boundary_target_kind=internal_world_shadow_graft boundary_target_mode=receipt_only_closed_dry_run receipt_shape=resonance_observation_shadow_graft_boundary",
			"dry_run_only=true observation_verified=true receiver_verified=true intent_verified=true final_gate_verified=true seal_verified=true permit_verified=true readiness_verified=true ledger_verified=true writer_ready=true rollback_ready=true ledger_ready=true boundary_ready=true",
			"contracts_ready=false write_allowed=false admission_allowed=false live_ready=true live_enabled=false mutates=false admission_resonance_graft_boundary_id=resonance-graft-boundary-id-",
			"passed=true reason=resonance shadow graft boundary declared without body mutation",
		)
	}
	if admissionLiveRouteTurnCandidateAdmissionResonanceGraftPreflightDryRun() {
		wants = append(wants,
			"live-route candidate admission resonance graft preflight dry-run: class=dream route=direct source=direct boundary=resonance-graft-boundary-id-",
			"observation=resonance-observation-",
			"receiver=resonance-receiver-",
			"intent=resonance-intent-",
			"final_gate=admission-final-gate-",
			"preflight_kind=shadow_graft_preflight preflight_mode=no_mutation_preflight preflight_stage=pre_live_graft_admission causal_id=resonance-graft-preflight-causal-",
			"preflight_hash=resonance-graft-preflight-",
			"read_back_hash=resonance-graft-preflight-read-",
			"source_boundary_causal_id=resonance-graft-boundary-causal-",
			"source_boundary_read_back_hash=resonance-graft-boundary-read-",
			"admission_required=true shadow_only=true graft_allowed=false raw_text_allowed=false janus_surface_allowed=false cooc_learning_allowed=false delta_harvest_allowed=false body_mutation_allowed=false rollback_required=true",
			"preflight_state=shadow_graft_preflight_ready_dry_run preflight_action=prepare_resonance_shadow_graft_preflight_dry_run preflight_target=resonance preflight_target_kind=internal_world_shadow_graft_preflight preflight_target_mode=receipt_only_closed_preflight_dry_run receipt_shape=resonance_shadow_graft_preflight_contract",
			"dry_run_only=true boundary_verified=true observation_verified=true receiver_verified=true intent_verified=true final_gate_verified=true seal_verified=true permit_verified=true readiness_verified=true ledger_verified=true writer_ready=true rollback_ready=true ledger_ready=true preflight_ready=true",
			"contracts_ready=false write_allowed=false admission_allowed=false live_ready=true live_enabled=false mutates=false admission_resonance_graft_preflight_id=resonance-graft-preflight-id-",
			"passed=true reason=resonance shadow graft preflight prepared without body mutation",
		)
	}
	if admissionLiveRouteTurnCandidateAdmissionResonanceGraftGateDryRun() {
		wants = append(wants,
			"live-route candidate admission resonance graft gate dry-run: class=dream route=direct source=direct preflight=resonance-graft-preflight-id-",
			"boundary=resonance-graft-boundary-id-",
			"observation=resonance-observation-",
			"receiver=resonance-receiver-",
			"intent=resonance-intent-",
			"final_gate=admission-final-gate-",
			"gate_kind=shadow_graft_gate gate_mode=no_mutation_gate gate_stage=pre_live_graft_gate causal_id=resonance-graft-gate-causal-",
			"gate_hash=resonance-graft-gate-",
			"read_back_hash=resonance-graft-gate-read-",
			"source_preflight_causal_id=resonance-graft-preflight-causal-",
			"source_preflight_read_back_hash=resonance-graft-preflight-read-",
			"admission_required=true shadow_only=true graft_allowed=false raw_text_allowed=false janus_surface_allowed=false cooc_learning_allowed=false delta_harvest_allowed=false body_mutation_allowed=false rollback_required=true",
			"gate_state=shadow_graft_gate_ready_dry_run gate_action=gate_resonance_shadow_graft_dry_run gate_target=resonance gate_target_kind=internal_world_shadow_graft_gate gate_target_mode=receipt_only_closed_gate_dry_run receipt_shape=resonance_shadow_graft_gate_contract",
			"dry_run_only=true preflight_verified=true boundary_verified=true observation_verified=true receiver_verified=true intent_verified=true final_gate_verified=true seal_verified=true permit_verified=true readiness_verified=true ledger_verified=true writer_ready=true rollback_ready=true ledger_ready=true gate_ready=true",
			"contracts_ready=false write_allowed=false admission_allowed=false live_ready=true live_enabled=false mutates=false admission_resonance_graft_gate_id=resonance-graft-gate-id-",
			"passed=true reason=resonance shadow graft gate prepared without body mutation",
		)
	}
	if admissionLiveRouteTurnCandidateAdmissionResonanceGraftCandidateDryRun() {
		wants = append(wants,
			"live-route candidate admission resonance graft candidate dry-run: class=dream route=direct source=direct gate=resonance-graft-gate-id-",
			"preflight=resonance-graft-preflight-id-",
			"boundary=resonance-graft-boundary-id-",
			"observation=resonance-observation-",
			"receiver=resonance-receiver-",
			"intent=resonance-intent-",
			"final_gate=admission-final-gate-",
			"candidate_kind=shadow_graft_candidate candidate_mode=no_mutation_candidate candidate_stage=pre_live_graft_candidate causal_id=resonance-graft-candidate-causal-",
			"candidate_hash=resonance-graft-candidate-",
			"read_back_hash=resonance-graft-candidate-read-",
			"source_gate_causal_id=resonance-graft-gate-causal-",
			"source_gate_read_back_hash=resonance-graft-gate-read-",
			"admission_required=true shadow_only=true graft_allowed=false raw_text_allowed=false janus_surface_allowed=false cooc_learning_allowed=false delta_harvest_allowed=false body_mutation_allowed=false rollback_required=true",
			"candidate_state=shadow_graft_candidate_ready_dry_run candidate_action=draft_resonance_shadow_graft_candidate_dry_run candidate_target=resonance candidate_target_kind=internal_world_shadow_graft_candidate candidate_target_mode=receipt_only_closed_candidate_dry_run receipt_shape=resonance_shadow_graft_candidate_contract",
			"dry_run_only=true gate_verified=true preflight_verified=true boundary_verified=true observation_verified=true receiver_verified=true intent_verified=true final_gate_verified=true seal_verified=true permit_verified=true readiness_verified=true ledger_verified=true writer_ready=true rollback_ready=true ledger_ready=true candidate_ready=true",
			"contracts_ready=false write_allowed=false admission_allowed=false live_ready=true live_enabled=false mutates=false admission_resonance_graft_candidate_id=resonance-graft-candidate-id-",
			"passed=true reason=resonance shadow graft candidate drafted without body mutation",
		)
	}
	if admissionLiveRouteTurnCandidateAdmissionResonanceGraftCandidateStoreDryRun() {
		wants = append(wants,
			"live-route candidate admission resonance graft candidate store dry-run: class=dream route=direct source=direct candidate=resonance-graft-candidate-id-",
			"gate=resonance-graft-gate-id-",
			"preflight=resonance-graft-preflight-id-",
			"boundary=resonance-graft-boundary-id-",
			"observation=resonance-observation-",
			"receiver=resonance-receiver-",
			"intent=resonance-intent-",
			"final_gate=admission-final-gate-",
			"store_kind=shadow_graft_candidate_store store_mode=append_only_read_back_store store_stage=pre_live_graft_candidate_store causal_id=resonance-graft-candidate-store-causal-",
			"store_hash=resonance-graft-candidate-store-",
			"read_back_hash=resonance-graft-candidate-store-read-",
			"source_candidate_causal_id=resonance-graft-candidate-causal-",
			"source_candidate_read_back_hash=resonance-graft-candidate-read-",
			"admission_required=true shadow_only=true graft_allowed=false raw_text_allowed=false janus_surface_allowed=false cooc_learning_allowed=false delta_harvest_allowed=false body_mutation_allowed=false rollback_required=true append_only=true read_back=true receipt_persisted=true receipt_verified=true",
			"store_state=shadow_graft_candidate_stored_dry_run store_action=store_resonance_shadow_graft_candidate_dry_run store_target=resonance store_target_kind=internal_world_shadow_graft_candidate_store store_target_mode=append_only_read_back_store_dry_run receipt_shape=resonance_shadow_graft_candidate_store_receipt",
			"dry_run_only=true candidate_verified=true gate_verified=true preflight_verified=true boundary_verified=true observation_verified=true receiver_verified=true intent_verified=true final_gate_verified=true seal_verified=true permit_verified=true readiness_verified=true ledger_verified=true writer_ready=true rollback_ready=true ledger_ready=true store_ready=true",
			"contracts_ready=false write_allowed=false admission_allowed=false live_ready=true live_enabled=false mutates=false admission_resonance_graft_candidate_store_id=resonance-graft-candidate-store-id-",
			"passed=true reason=resonance shadow graft candidate stored and read back without body mutation",
		)
	}
	if admissionLiveRouteTurnCandidateAdmissionResonanceGraftCandidateStoreReaderDryRun() {
		wants = append(wants,
			"live-route candidate admission resonance graft candidate store reader dry-run: class=dream route=direct source=direct store=resonance-graft-candidate-store-id-",
			"candidate=resonance-graft-candidate-id-",
			"gate=resonance-graft-gate-id-",
			"observation=resonance-observation-",
			"final_gate=admission-final-gate-",
			"reader_kind=shadow_graft_candidate_store_reader reader_mode=read_only_replay reader_stage=pre_live_graft_candidate_store_reader causal_id=resonance-graft-candidate-store-reader-causal-",
			"reader_hash=resonance-graft-candidate-store-reader-",
			"replay_hash=resonance-graft-candidate-store-reader-replay-",
			"read_back_hash=resonance-graft-candidate-store-reader-read-",
			"source_store_causal_id=resonance-graft-candidate-store-causal-",
			"source_store_read_back_hash=resonance-graft-candidate-store-read-",
			"read_only=true replay_only=true source_append_only=true source_read_back=true source_receipt_verified=true graft_allowed=false raw_text_allowed=false janus_surface_allowed=false cooc_learning_allowed=false delta_harvest_allowed=false body_mutation_allowed=false",
			"reader_state=shadow_graft_candidate_store_read_back_dry_run reader_action=read_resonance_shadow_graft_candidate_store_dry_run reader_target=resonance reader_target_kind=internal_world_shadow_graft_candidate_store_reader reader_target_mode=read_only_replay_dry_run receipt_shape=resonance_shadow_graft_candidate_store_reader_receipt",
			"dry_run_only=true store_verified=true candidate_verified=true ledger_verified=true hash_verified=true reader_read_back_verified=true reader_ready=true",
			"contracts_ready=false write_allowed=false admission_allowed=false live_ready=true live_enabled=false mutates=false admission_resonance_graft_candidate_store_reader_id=resonance-graft-candidate-store-reader-id-",
			"passed=true reason=resonance shadow graft candidate store read back without opening body",
		)
	}
	if admissionLiveRouteTurnCandidateAdmissionResonanceGraftAdmissionProofDryRun() {
		wants = append(wants,
			"live-route candidate admission resonance graft admission proof dry-run: class=dream route=direct source=direct reader=resonance-graft-candidate-store-reader-id-",
			"store=resonance-graft-candidate-store-id-",
			"candidate=resonance-graft-candidate-id-",
			"gate=resonance-graft-gate-id-",
			"observation=resonance-observation-",
			"final_gate=admission-final-gate-",
			"proof_kind=shadow_graft_admission_proof proof_mode=verified_replay_closed proof_stage=pre_live_graft_admission_proof causal_id=resonance-graft-admission-proof-causal-",
			"proof_hash=resonance-graft-admission-proof-",
			"replay_hash=resonance-graft-admission-proof-replay-",
			"read_back_hash=resonance-graft-admission-proof-read-",
			"source_reader_causal_id=resonance-graft-candidate-store-reader-causal-",
			"source_reader_read_back_hash=resonance-graft-candidate-store-reader-read-",
			"admission_required=true shadow_only=true graft_allowed=false raw_text_allowed=false janus_surface_allowed=false cooc_learning_allowed=false delta_harvest_allowed=false body_mutation_allowed=false rollback_required=true",
			"proof_state=shadow_graft_admission_proved_dry_run proof_action=prove_resonance_shadow_graft_admission_dry_run proof_target=resonance proof_target_kind=internal_world_shadow_graft_admission_proof proof_target_mode=verified_replay_closed_dry_run receipt_shape=resonance_shadow_graft_admission_proof_receipt",
			"dry_run_only=true reader_verified=true store_verified=true candidate_verified=true ledger_verified=true replay_verified=true hash_verified=true proof_read_back_verified=true proof_ready=true",
			"contracts_ready=false write_allowed=false admission_allowed=false live_ready=true live_enabled=false mutates=false admission_resonance_graft_admission_proof_id=resonance-graft-admission-proof-id-",
			"passed=true reason=resonance shadow graft admission proved from read-back store without opening body",
		)
	}
	for _, want := range wants {
		found := false
		for _, line := range lines {
			if strings.Contains(line, want) {
				found = true
				break
			}
		}
		if !found {
			return fmt.Errorf("missing nano-direct chat-shadow line %q in %v", want, lines)
		}
	}
	for _, line := range lines {
		fmt.Println(line)
	}

	readOne := func(path, label string) ([]byte, error) {
		raw, err := os.ReadFile(path)
		if err != nil {
			return nil, err
		}
		trimmed := strings.TrimSpace(string(raw))
		if trimmed == "" {
			return nil, fmt.Errorf("%s receipt log is empty", label)
		}
		records := strings.Split(trimmed, "\n")
		if len(records) != 1 {
			return nil, fmt.Errorf("expected 1 %s receipt, got %d", label, len(records))
		}
		return []byte(records[0]), nil
	}

	var execution admissionLiveRouteTurnCandidateExecution
	if raw, err := readOne(executionLogPath, "candidate execution"); err != nil {
		return err
	} else if err := json.Unmarshal(raw, &execution); err != nil {
		return fmt.Errorf("candidate execution receipt: %w", err)
	}
	var generatorAdapter admissionLiveRouteTurnGeneratorAdapter
	if raw, err := readOne(adapterLogPath, "generator adapter"); err != nil {
		return err
	} else if err := json.Unmarshal(raw, &generatorAdapter); err != nil {
		return fmt.Errorf("generator adapter receipt: %w", err)
	}
	var draft admissionLiveRouteTurnCandidateDraft
	if raw, err := readOne(draftLogPath, "candidate draft"); err != nil {
		return err
	} else if err := json.Unmarshal(raw, &draft); err != nil {
		return fmt.Errorf("candidate draft receipt: %w", err)
	}
	var review admissionLiveRouteTurnCandidateReview
	if raw, err := readOne(reviewLogPath, "candidate review"); err != nil {
		return err
	} else if err := json.Unmarshal(raw, &review); err != nil {
		return fmt.Errorf("candidate review receipt: %w", err)
	}
	var admission admissionLiveRouteTurnCandidateAdmission
	if raw, err := readOne(admissionLogPath, "candidate admission"); err != nil {
		return err
	} else if err := json.Unmarshal(raw, &admission); err != nil {
		return fmt.Errorf("candidate admission receipt: %w", err)
	}
	var admissionAdapter admissionLiveRouteTurnCandidateAdmissionAdapter
	if raw, err := readOne(admissionAdapterLogPath, "candidate admission adapter"); err != nil {
		return err
	} else if err := json.Unmarshal(raw, &admissionAdapter); err != nil {
		return fmt.Errorf("candidate admission adapter receipt: %w", err)
	}
	var candidate dreamCandidate
	if raw, err := readOne(dreamLogPath, "dream admission"); err != nil {
		return err
	} else if err := json.Unmarshal(raw, &candidate); err != nil {
		return fmt.Errorf("dream admission receipt: %w", err)
	}
	var decision admissionLiveRouteTurnCandidateAdmissionDecision
	if admissionLiveRouteTurnCandidateAdmissionDecisionDryRun() {
		if raw, err := readOne(decisionLogPath, "candidate admission decision"); err != nil {
			return err
		} else if err := json.Unmarshal(raw, &decision); err != nil {
			return fmt.Errorf("candidate admission decision receipt: %w", err)
		}
	}
	var promotion admissionLiveRouteTurnCandidateAdmissionPromotion
	if admissionLiveRouteTurnCandidateAdmissionPromotionDryRun() {
		if raw, err := readOne(promotionLogPath, "candidate admission promotion"); err != nil {
			return err
		} else if err := json.Unmarshal(raw, &promotion); err != nil {
			return fmt.Errorf("candidate admission promotion receipt: %w", err)
		}
	}
	var sw admissionLiveRouteTurnCandidateAdmissionSwitch
	if admissionLiveRouteTurnCandidateAdmissionSwitchDryRun() {
		if raw, err := readOne(switchLogPath, "candidate admission switch"); err != nil {
			return err
		} else if err := json.Unmarshal(raw, &sw); err != nil {
			return fmt.Errorf("candidate admission switch receipt: %w", err)
		}
	}
	var gate admissionLiveRouteTurnCandidateAdmissionEnableGate
	if admissionLiveRouteTurnCandidateAdmissionEnableGateDryRun() {
		if raw, err := readOne(enableGateLogPath, "candidate admission enable gate"); err != nil {
			return err
		} else if err := json.Unmarshal(raw, &gate); err != nil {
			return fmt.Errorf("candidate admission enable gate receipt: %w", err)
		}
	}
	var liveStage admissionLiveRouteTurnCandidateAdmissionLiveStage
	if admissionLiveRouteTurnCandidateAdmissionLiveStageDryRun() {
		if raw, err := readOne(liveStageLogPath, "candidate admission live stage"); err != nil {
			return err
		} else if err := json.Unmarshal(raw, &liveStage); err != nil {
			return fmt.Errorf("candidate admission live stage receipt: %w", err)
		}
	}
	var writerPreflight admissionLiveRouteTurnCandidateAdmissionWriterPreflight
	if admissionLiveRouteTurnCandidateAdmissionWriterPreflightDryRun() {
		if raw, err := readOne(writerPreflightLogPath, "candidate admission writer preflight"); err != nil {
			return err
		} else if err := json.Unmarshal(raw, &writerPreflight); err != nil {
			return fmt.Errorf("candidate admission writer preflight receipt: %w", err)
		}
	}
	var writerInventory admissionLiveRouteTurnCandidateAdmissionWriterInventory
	if admissionLiveRouteTurnCandidateAdmissionWriterInventoryDryRun() {
		if raw, err := readOne(writerInventoryLogPath, "candidate admission writer inventory"); err != nil {
			return err
		} else if err := json.Unmarshal(raw, &writerInventory); err != nil {
			return fmt.Errorf("candidate admission writer inventory receipt: %w", err)
		}
	}
	var writerContract admissionLiveRouteTurnCandidateAdmissionWriterContract
	if admissionLiveRouteTurnCandidateAdmissionWriterContractDryRun() {
		if raw, err := readOne(writerContractLogPath, "candidate admission writer contract"); err != nil {
			return err
		} else if err := json.Unmarshal(raw, &writerContract); err != nil {
			return fmt.Errorf("candidate admission writer contract receipt: %w", err)
		}
	}
	var ledger admissionLiveRouteTurnCandidateAdmissionLedger
	if admissionLiveRouteTurnCandidateAdmissionLedgerDryRun() {
		if raw, err := readOne(ledgerLogPath, "candidate admission ledger"); err != nil {
			return err
		} else if err := json.Unmarshal(raw, &ledger); err != nil {
			return fmt.Errorf("candidate admission ledger receipt: %w", err)
		}
	}
	var writerImpl admissionLiveRouteTurnCandidateAdmissionWriterImplementation
	if admissionLiveRouteTurnCandidateAdmissionWriterImplementationDryRun() {
		if raw, err := readOne(writerImplLogPath, "candidate admission writer implementation"); err != nil {
			return err
		} else if err := json.Unmarshal(raw, &writerImpl); err != nil {
			return fmt.Errorf("candidate admission writer implementation receipt: %w", err)
		}
	}
	var writerReceipt admissionLiveRouteTurnCandidateAdmissionWriterReceipt
	if admissionLiveRouteTurnCandidateAdmissionWriterReceiptDryRun() {
		if raw, err := readOne(writerReceiptLogPath, "candidate admission writer receipt"); err != nil {
			return err
		} else if err := json.Unmarshal(raw, &writerReceipt); err != nil {
			return fmt.Errorf("candidate admission writer receipt: %w", err)
		}
	}
	var rollbackImpl admissionLiveRouteTurnCandidateAdmissionRollbackImplementation
	if admissionLiveRouteTurnCandidateAdmissionRollbackImplementationDryRun() {
		if raw, err := readOne(rollbackImplLogPath, "candidate admission rollback implementation"); err != nil {
			return err
		} else if err := json.Unmarshal(raw, &rollbackImpl); err != nil {
			return fmt.Errorf("candidate admission rollback implementation receipt: %w", err)
		}
	}
	var ledgerImpl admissionLiveRouteTurnCandidateAdmissionLedgerImplementation
	if admissionLiveRouteTurnCandidateAdmissionLedgerImplementationDryRun() {
		if raw, err := readOne(ledgerImplLogPath, "candidate admission ledger implementation"); err != nil {
			return err
		} else if err := json.Unmarshal(raw, &ledgerImpl); err != nil {
			return fmt.Errorf("candidate admission ledger implementation receipt: %w", err)
		}
	}
	var ledgerPersistence admissionLiveRouteTurnCandidateAdmissionLedgerPersistence
	if admissionLiveRouteTurnCandidateAdmissionLedgerPersistenceDryRun() {
		if raw, err := readOne(ledgerPersistenceLogPath, "candidate admission ledger persistence"); err != nil {
			return err
		} else if err := json.Unmarshal(raw, &ledgerPersistence); err != nil {
			return fmt.Errorf("candidate admission ledger persistence receipt: %w", err)
		}
	}
	var ledgerVerification admissionLiveRouteTurnCandidateAdmissionLedgerVerification
	if admissionLiveRouteTurnCandidateAdmissionLedgerVerificationDryRun() {
		if raw, err := readOne(ledgerVerificationLogPath, "candidate admission ledger verification"); err != nil {
			return err
		} else if err := json.Unmarshal(raw, &ledgerVerification); err != nil {
			return fmt.Errorf("candidate admission ledger verification receipt: %w", err)
		}
	}
	var readiness admissionLiveRouteTurnCandidateAdmissionReadiness
	if admissionLiveRouteTurnCandidateAdmissionReadinessDryRun() {
		if raw, err := readOne(readinessLogPath, "candidate admission readiness"); err != nil {
			return err
		} else if err := json.Unmarshal(raw, &readiness); err != nil {
			return fmt.Errorf("candidate admission readiness receipt: %w", err)
		}
	}
	var permit admissionLiveRouteTurnCandidateAdmissionPermit
	if admissionLiveRouteTurnCandidateAdmissionPermitDryRun() {
		if raw, err := readOne(permitLogPath, "candidate admission permit"); err != nil {
			return err
		} else if err := json.Unmarshal(raw, &permit); err != nil {
			return fmt.Errorf("candidate admission permit receipt: %w", err)
		}
	}
	var seal admissionLiveRouteTurnCandidateAdmissionSeal
	if admissionLiveRouteTurnCandidateAdmissionSealDryRun() {
		if raw, err := readOne(sealLogPath, "candidate admission seal"); err != nil {
			return err
		} else if err := json.Unmarshal(raw, &seal); err != nil {
			return fmt.Errorf("candidate admission seal receipt: %w", err)
		}
	}
	var finalGate admissionLiveRouteTurnCandidateAdmissionFinalGate
	if admissionLiveRouteTurnCandidateAdmissionFinalGateDryRun() {
		if raw, err := readOne(finalGateLogPath, "candidate admission final gate"); err != nil {
			return err
		} else if err := json.Unmarshal(raw, &finalGate); err != nil {
			return fmt.Errorf("candidate admission final gate receipt: %w", err)
		}
	}
	var resonanceIntent admissionLiveRouteTurnCandidateAdmissionResonanceIntent
	if admissionLiveRouteTurnCandidateAdmissionResonanceIntentDryRun() {
		if raw, err := readOne(resonanceIntentLogPath, "candidate admission resonance intent"); err != nil {
			return err
		} else if err := json.Unmarshal(raw, &resonanceIntent); err != nil {
			return fmt.Errorf("candidate admission resonance intent receipt: %w", err)
		}
	}
	var resonanceReceiver admissionLiveRouteTurnCandidateAdmissionResonanceReceiver
	if admissionLiveRouteTurnCandidateAdmissionResonanceReceiverDryRun() {
		if raw, err := readOne(resonanceReceiverLogPath, "candidate admission resonance receiver"); err != nil {
			return err
		} else if err := json.Unmarshal(raw, &resonanceReceiver); err != nil {
			return fmt.Errorf("candidate admission resonance receiver receipt: %w", err)
		}
	}
	var resonanceObservation admissionLiveRouteTurnCandidateAdmissionResonanceObservation
	if admissionLiveRouteTurnCandidateAdmissionResonanceObservationDryRun() {
		if raw, err := readOne(resonanceObservationLogPath, "candidate admission resonance observation"); err != nil {
			return err
		} else if err := json.Unmarshal(raw, &resonanceObservation); err != nil {
			return fmt.Errorf("candidate admission resonance observation receipt: %w", err)
		}
	}
	var resonanceGraftBoundary admissionLiveRouteTurnCandidateAdmissionResonanceGraftBoundary
	if admissionLiveRouteTurnCandidateAdmissionResonanceGraftBoundaryDryRun() {
		if raw, err := readOne(resonanceGraftBoundaryLogPath, "candidate admission resonance graft boundary"); err != nil {
			return err
		} else if err := json.Unmarshal(raw, &resonanceGraftBoundary); err != nil {
			return fmt.Errorf("candidate admission resonance graft boundary receipt: %w", err)
		}
	}
	var resonanceGraftPreflight admissionLiveRouteTurnCandidateAdmissionResonanceGraftPreflight
	if admissionLiveRouteTurnCandidateAdmissionResonanceGraftPreflightDryRun() {
		if raw, err := readOne(resonanceGraftPreflightLogPath, "candidate admission resonance graft preflight"); err != nil {
			return err
		} else if err := json.Unmarshal(raw, &resonanceGraftPreflight); err != nil {
			return fmt.Errorf("candidate admission resonance graft preflight receipt: %w", err)
		}
	}
	var resonanceGraftGate admissionLiveRouteTurnCandidateAdmissionResonanceGraftGate
	if admissionLiveRouteTurnCandidateAdmissionResonanceGraftGateDryRun() {
		if raw, err := readOne(resonanceGraftGateLogPath, "candidate admission resonance graft gate"); err != nil {
			return err
		} else if err := json.Unmarshal(raw, &resonanceGraftGate); err != nil {
			return fmt.Errorf("candidate admission resonance graft gate receipt: %w", err)
		}
	}
	var resonanceGraftCandidate admissionLiveRouteTurnCandidateAdmissionResonanceGraftCandidate
	if admissionLiveRouteTurnCandidateAdmissionResonanceGraftCandidateDryRun() {
		if raw, err := readOne(resonanceGraftCandidateLogPath, "candidate admission resonance graft candidate"); err != nil {
			return err
		} else if err := json.Unmarshal(raw, &resonanceGraftCandidate); err != nil {
			return fmt.Errorf("candidate admission resonance graft candidate receipt: %w", err)
		}
	}
	var resonanceGraftCandidateStore admissionLiveRouteTurnCandidateAdmissionResonanceGraftCandidateStore
	if admissionLiveRouteTurnCandidateAdmissionResonanceGraftCandidateStoreDryRun() {
		if raw, err := readOne(resonanceGraftCandidateStoreLogPath, "candidate admission resonance graft candidate store"); err != nil {
			return err
		} else if err := json.Unmarshal(raw, &resonanceGraftCandidateStore); err != nil {
			return fmt.Errorf("candidate admission resonance graft candidate store receipt: %w", err)
		}
	}
	var resonanceGraftCandidateStoreReader admissionLiveRouteTurnCandidateAdmissionResonanceGraftCandidateStoreReader
	if admissionLiveRouteTurnCandidateAdmissionResonanceGraftCandidateStoreReaderDryRun() {
		if raw, err := readOne(resonanceGraftCandidateStoreReaderLogPath, "candidate admission resonance graft candidate store reader"); err != nil {
			return err
		} else if err := json.Unmarshal(raw, &resonanceGraftCandidateStoreReader); err != nil {
			return fmt.Errorf("candidate admission resonance graft candidate store reader receipt: %w", err)
		}
	}
	var resonanceGraftAdmissionProof admissionLiveRouteTurnCandidateAdmissionResonanceGraftAdmissionProof
	if admissionLiveRouteTurnCandidateAdmissionResonanceGraftAdmissionProofDryRun() {
		if raw, err := readOne(resonanceGraftAdmissionProofLogPath, "candidate admission resonance graft admission proof"); err != nil {
			return err
		} else if err := json.Unmarshal(raw, &resonanceGraftAdmissionProof); err != nil {
			return fmt.Errorf("candidate admission resonance graft admission proof receipt: %w", err)
		}
	}

	if execution.Schema != admissionLiveRouteTurnCandidateExecutionSchema ||
		!execution.Passed ||
		execution.Runner != admissionLiveRouteTurnCandidateExecutionRunnerNanoDirect ||
		execution.RunnerStatus != admissionLiveRouteTurnCandidateExecutionStatusSucceeded ||
		execution.ExecutionID == "" ||
		execution.GeneratedText == "" ||
		execution.GeneratedTextStatus != "generated" ||
		execution.GeneratedTextHash == "" ||
		execution.RunnerStdoutHash != execution.GeneratedTextHash {
		return fmt.Errorf("bad nano-direct execution receipt: %+v", execution)
	}
	if generatorAdapter.Schema != admissionLiveRouteTurnGeneratorAdapterSchema ||
		!generatorAdapter.Passed ||
		generatorAdapter.CandidateExecutionID != execution.ExecutionID ||
		generatorAdapter.GeneratedText != execution.GeneratedText ||
		generatorAdapter.GeneratedTextHash != execution.GeneratedTextHash ||
		generatorAdapter.AdapterID == "" {
		return fmt.Errorf("bad nano-direct generator adapter receipt: adapter=%+v execution=%+v", generatorAdapter, execution)
	}
	if draft.Schema != admissionLiveRouteTurnCandidateDraftSchema ||
		!draft.Passed ||
		draft.CandidateExecutionID != execution.ExecutionID ||
		draft.GeneratorAdapterID != generatorAdapter.AdapterID ||
		draft.CandidateText != execution.GeneratedText ||
		draft.CandidateTextHash != execution.GeneratedTextHash ||
		draft.DraftID == "" ||
		draft.CandidateRunID == "" {
		return fmt.Errorf("bad nano-direct candidate draft receipt: draft=%+v adapter=%+v execution=%+v", draft, generatorAdapter, execution)
	}
	if review.Schema != admissionLiveRouteTurnReviewSchema ||
		!review.Matched ||
		review.CandidateDraftID != draft.DraftID ||
		review.CandidateExecutionID != execution.ExecutionID ||
		review.GeneratorAdapterID != generatorAdapter.AdapterID ||
		review.CandidateRunID != draft.CandidateRunID ||
		review.CandidateTextHash != draft.CandidateTextHash {
		return fmt.Errorf("bad nano-direct candidate review receipt: review=%+v draft=%+v", review, draft)
	}
	if admission.Schema != admissionLiveRouteTurnCandidateAdmissionSchema ||
		!admission.Passed ||
		admission.CandidateDraftID != draft.DraftID ||
		admission.CandidateExecutionID != execution.ExecutionID ||
		admission.GeneratorAdapterID != generatorAdapter.AdapterID ||
		admission.CandidateRunID != draft.CandidateRunID ||
		admission.CandidateTextHash != draft.CandidateTextHash ||
		admission.HandoffID == "" {
		return fmt.Errorf("bad nano-direct candidate admission receipt: admission=%+v draft=%+v", admission, draft)
	}
	if admissionAdapter.Schema != admissionLiveRouteTurnCandidateAdmissionAdapterSchema ||
		!admissionAdapter.Passed ||
		admissionAdapter.HandoffID != admission.HandoffID ||
		admissionAdapter.CandidateDraftID != draft.DraftID ||
		admissionAdapter.CandidateExecutionID != execution.ExecutionID ||
		admissionAdapter.GeneratorAdapterID != generatorAdapter.AdapterID ||
		admissionAdapter.DreamCandidateRunID != draft.CandidateRunID ||
		admissionAdapter.AdmissionAdapterID == "" {
		return fmt.Errorf("bad nano-direct candidate admission adapter receipt: admission_adapter=%+v admission=%+v draft=%+v", admissionAdapter, admission, draft)
	}
	if candidate.Schema != "arianna.dream_candidate.v1" ||
		candidate.LiveRouteCandidateAdmission == nil ||
		candidate.LiveRouteCandidateAdmission.AdmissionAdapterID != admissionAdapter.AdmissionAdapterID ||
		candidate.LiveRouteCandidateAdmission.HandoffID != admissionAdapter.HandoffID ||
		candidate.RunID != draft.CandidateRunID ||
		candidate.Text != draft.CandidateText ||
		candidate.Admission == nil ||
		!candidate.Admission.Passed ||
		candidate.Admission.LiveRouteChoice == nil ||
		!candidate.Admission.LiveRouteChoice.Passed ||
		candidate.Accepted ||
		candidate.Reason != "shadow mode" {
		return fmt.Errorf("bad nano-direct dream admission receipt: candidate=%+v admission_adapter=%+v draft=%+v", candidate, admissionAdapter, draft)
	}
	if admissionLiveRouteTurnCandidateAdmissionDecisionDryRun() {
		if decision.Schema != admissionLiveRouteTurnCandidateAdmissionDecisionSchema ||
			!decision.Passed ||
			!decision.LiveReady ||
			decision.MutatesState ||
			decision.Decision != "shadow_ready" ||
			decision.DecisionID == "" ||
			decision.CandidateExecutionID != execution.ExecutionID ||
			decision.GeneratorAdapterID != generatorAdapter.AdapterID ||
			decision.CandidateDraftID != draft.DraftID ||
			decision.HandoffID != admission.HandoffID ||
			decision.AdmissionAdapterID != admissionAdapter.AdmissionAdapterID ||
			decision.DreamCandidateRunID != candidate.RunID ||
			decision.CandidateTextHash != execution.GeneratedTextHash ||
			!admissionLiveRouteBoundaryFieldsEqual(
				decision.BodyInventoryStatus,
				decision.RouteAvailabilityStatus,
				decision.RouteAvailabilityReason,
				decision.RouteMissingOrgans,
				admissionAdapter.BodyInventoryStatus,
				admissionAdapter.RouteAvailabilityStatus,
				admissionAdapter.RouteAvailabilityReason,
				admissionAdapter.RouteMissingOrgans,
			) ||
			decision.DreamAccepted ||
			!decision.AdmissionPolicyPassed ||
			!decision.LiveRouteChoicePassed {
			return fmt.Errorf("bad nano-direct admission decision receipt: decision=%+v candidate=%+v admission_adapter=%+v execution=%+v", decision, candidate, admissionAdapter, execution)
		}
	}
	if admissionLiveRouteTurnCandidateAdmissionPromotionDryRun() {
		if promotion.Schema != admissionLiveRouteTurnCandidateAdmissionPromotionSchema ||
			!promotion.Passed ||
			!promotion.LiveReady ||
			promotion.LiveAdmissionEnabled ||
			promotion.MutatesState ||
			promotion.Promotion != "pending_live_admission" ||
			promotion.PromotionID == "" ||
			promotion.AdmissionDecisionID == "" ||
			promotion.AdmissionDecision != "shadow_ready" ||
			promotion.AdmissionDecisionID != decision.DecisionID ||
			promotion.CandidateExecutionID != execution.ExecutionID ||
			promotion.GeneratorAdapterID != generatorAdapter.AdapterID ||
			promotion.CandidateDraftID != draft.DraftID ||
			promotion.HandoffID != admission.HandoffID ||
			promotion.AdmissionAdapterID != admissionAdapter.AdmissionAdapterID ||
			promotion.DreamCandidateRunID != candidate.RunID ||
			promotion.CandidateTextHash != execution.GeneratedTextHash ||
			!promotion.AdmissionPolicyPassed ||
			!promotion.LiveRouteChoicePassed ||
			!promotion.SourceDecisionPassed {
			return fmt.Errorf("bad nano-direct admission promotion receipt: promotion=%+v decision=%+v execution=%+v", promotion, decision, execution)
		}
	}
	if admissionLiveRouteTurnCandidateAdmissionSwitchDryRun() {
		if sw.Schema != admissionLiveRouteTurnCandidateAdmissionSwitchSchema ||
			!sw.Passed ||
			!sw.LiveReady ||
			sw.LiveAdmissionEnabled ||
			sw.AdmissionAllowed ||
			sw.MutatesState ||
			sw.SwitchState != "disabled" ||
			sw.SwitchAction != "hold_pending_live_admission" ||
			sw.SwitchID == "" ||
			sw.AdmissionPromotionID == "" ||
			sw.AdmissionDecisionID == "" ||
			sw.AdmissionPromotion != "pending_live_admission" ||
			sw.AdmissionDecision != "shadow_ready" ||
			sw.AdmissionPromotionID != promotion.PromotionID ||
			sw.AdmissionDecisionID != decision.DecisionID ||
			sw.CandidateExecutionID != execution.ExecutionID ||
			sw.GeneratorAdapterID != generatorAdapter.AdapterID ||
			sw.CandidateDraftID != draft.DraftID ||
			sw.HandoffID != admission.HandoffID ||
			sw.AdmissionAdapterID != admissionAdapter.AdmissionAdapterID ||
			sw.DreamCandidateRunID != candidate.RunID ||
			sw.CandidateTextHash != execution.GeneratedTextHash ||
			!sw.AdmissionPolicyPassed ||
			!sw.LiveRouteChoicePassed ||
			!sw.SourceDecisionPassed ||
			!sw.SourcePromotionPassed {
			return fmt.Errorf("bad nano-direct admission switch receipt: switch=%+v promotion=%+v decision=%+v execution=%+v", sw, promotion, decision, execution)
		}
	}
	if admissionLiveRouteTurnCandidateAdmissionEnableGateDryRun() {
		if gate.Schema != admissionLiveRouteTurnCandidateAdmissionEnableGateSchema ||
			!gate.Passed ||
			!gate.LiveReady ||
			gate.LiveAdmissionEnabled ||
			gate.AdmissionAllowed ||
			gate.MutatesState ||
			gate.EnableGateID == "" ||
			gate.AdmissionSwitchID == "" ||
			gate.AdmissionPromotionID == "" ||
			gate.AdmissionDecisionID == "" ||
			gate.AdmissionSwitchID != sw.SwitchID ||
			gate.AdmissionPromotionID != promotion.PromotionID ||
			gate.AdmissionDecisionID != decision.DecisionID ||
			gate.CandidateExecutionID != execution.ExecutionID ||
			gate.GeneratorAdapterID != generatorAdapter.AdapterID ||
			gate.CandidateDraftID != draft.DraftID ||
			gate.HandoffID != admission.HandoffID ||
			gate.AdmissionAdapterID != admissionAdapter.AdmissionAdapterID ||
			gate.DreamCandidateRunID != candidate.RunID ||
			gate.CandidateTextHash != execution.GeneratedTextHash ||
			!gate.AdmissionPolicyPassed ||
			!gate.LiveRouteChoicePassed ||
			!gate.SourceDecisionPassed ||
			!gate.SourcePromotionPassed ||
			!gate.SourceSwitchPassed {
			return fmt.Errorf("bad nano-direct admission enable gate receipt: gate=%+v switch=%+v promotion=%+v decision=%+v execution=%+v", gate, sw, promotion, decision, execution)
		}
		if admissionLiveRouteTurnCandidateAdmissionLiveStageDryRun() {
			if !gate.ManualEnableRequested ||
				!gate.EnableKeyMatched ||
				gate.EnableState != "armed_dry_run" ||
				gate.EnableAction != "would_enable_live_admission_dry_run" {
				return fmt.Errorf("bad nano-direct armed enable gate receipt: gate=%+v switch=%+v promotion=%+v decision=%+v execution=%+v", gate, sw, promotion, decision, execution)
			}
		} else if gate.ManualEnableRequested ||
			gate.EnableKeyMatched ||
			gate.EnableState != "disabled" ||
			gate.EnableAction != "require_operator_key" {
			return fmt.Errorf("bad nano-direct closed enable gate receipt: gate=%+v switch=%+v promotion=%+v decision=%+v execution=%+v", gate, sw, promotion, decision, execution)
		}
	}
	if admissionLiveRouteTurnCandidateAdmissionLiveStageDryRun() {
		if liveStage.Schema != admissionLiveRouteTurnCandidateAdmissionLiveStageSchema ||
			!liveStage.Passed ||
			!liveStage.LiveReady ||
			liveStage.LiveAdmissionEnabled ||
			liveStage.AdmissionAllowed ||
			!liveStage.ManualEnableRequested ||
			!liveStage.EnableKeyMatched ||
			!liveStage.RequiresWriter ||
			liveStage.WriterReady ||
			!liveStage.RequiresRollback ||
			liveStage.RollbackReady ||
			liveStage.MutatesState ||
			liveStage.StageState != "staged_dry_run" ||
			liveStage.StageAction != "stage_live_candidate_dry_run" ||
			liveStage.LiveStageID == "" ||
			liveStage.AdmissionEnableGateID != gate.EnableGateID ||
			liveStage.AdmissionSwitchID != sw.SwitchID ||
			liveStage.AdmissionPromotionID != promotion.PromotionID ||
			liveStage.AdmissionDecisionID != decision.DecisionID ||
			liveStage.CandidateExecutionID != execution.ExecutionID ||
			liveStage.GeneratorAdapterID != generatorAdapter.AdapterID ||
			liveStage.CandidateDraftID != draft.DraftID ||
			liveStage.HandoffID != admission.HandoffID ||
			liveStage.AdmissionAdapterID != admissionAdapter.AdmissionAdapterID ||
			liveStage.DreamCandidateRunID != candidate.RunID ||
			liveStage.CandidateRunID != draft.CandidateRunID ||
			liveStage.CandidateTextHash != execution.GeneratedTextHash ||
			liveStage.TurnTextHash != execution.TurnTextHash ||
			!liveStage.AdmissionPolicyPassed ||
			!liveStage.LiveRouteChoicePassed ||
			!liveStage.SourceDecisionPassed ||
			!liveStage.SourcePromotionPassed ||
			!liveStage.SourceSwitchPassed ||
			!liveStage.SourceEnablePassed {
			return fmt.Errorf("bad nano-direct admission live stage receipt: stage=%+v gate=%+v switch=%+v promotion=%+v decision=%+v execution=%+v", liveStage, gate, sw, promotion, decision, execution)
		}
	}
	if admissionLiveRouteTurnCandidateAdmissionWriterPreflightDryRun() {
		if writerPreflight.Schema != admissionLiveRouteTurnCandidateAdmissionWriterPreflightSchema ||
			!writerPreflight.Passed ||
			!writerPreflight.LiveReady ||
			writerPreflight.LiveAdmissionEnabled ||
			writerPreflight.AdmissionAllowed ||
			!writerPreflight.ManualEnableRequested ||
			!writerPreflight.EnableKeyMatched ||
			!writerPreflight.RequiresWriter ||
			writerPreflight.WriterReady ||
			writerPreflight.WriterState != "absent" ||
			writerPreflight.WriterAction != "require_writer_contract" ||
			!writerPreflight.RequiresRollback ||
			writerPreflight.RollbackReady ||
			writerPreflight.RollbackState != "absent" ||
			writerPreflight.RollbackAction != "require_rollback_contract" ||
			writerPreflight.WriteAllowed ||
			writerPreflight.MutatesState ||
			writerPreflight.StageState != "staged_dry_run" ||
			writerPreflight.StageAction != "stage_live_candidate_dry_run" ||
			writerPreflight.WriterPreflightID == "" ||
			writerPreflight.AdmissionLiveStageID != liveStage.LiveStageID ||
			writerPreflight.AdmissionEnableGateID != gate.EnableGateID ||
			writerPreflight.AdmissionSwitchID != sw.SwitchID ||
			writerPreflight.AdmissionPromotionID != promotion.PromotionID ||
			writerPreflight.AdmissionDecisionID != decision.DecisionID ||
			writerPreflight.CandidateExecutionID != execution.ExecutionID ||
			writerPreflight.GeneratorAdapterID != generatorAdapter.AdapterID ||
			writerPreflight.CandidateDraftID != draft.DraftID ||
			writerPreflight.HandoffID != admission.HandoffID ||
			writerPreflight.AdmissionAdapterID != admissionAdapter.AdmissionAdapterID ||
			writerPreflight.DreamCandidateRunID != candidate.RunID ||
			writerPreflight.CandidateRunID != draft.CandidateRunID ||
			writerPreflight.CandidateTextHash != execution.GeneratedTextHash ||
			writerPreflight.TurnTextHash != execution.TurnTextHash ||
			!writerPreflight.AdmissionPolicyPassed ||
			!writerPreflight.LiveRouteChoicePassed ||
			!writerPreflight.SourceDecisionPassed ||
			!writerPreflight.SourcePromotionPassed ||
			!writerPreflight.SourceSwitchPassed ||
			!writerPreflight.SourceEnablePassed ||
			!writerPreflight.SourceStagePassed {
			return fmt.Errorf("bad nano-direct admission writer preflight receipt: writer_preflight=%+v stage=%+v gate=%+v switch=%+v promotion=%+v decision=%+v execution=%+v", writerPreflight, liveStage, gate, sw, promotion, decision, execution)
		}
	}
	if admissionLiveRouteTurnCandidateAdmissionWriterInventoryDryRun() {
		if writerInventory.Schema != admissionLiveRouteTurnCandidateAdmissionWriterInventorySchema ||
			!writerInventory.Passed ||
			!writerInventory.LiveReady ||
			writerInventory.LiveAdmissionEnabled ||
			writerInventory.AdmissionAllowed ||
			!writerInventory.ManualEnableRequested ||
			!writerInventory.EnableKeyMatched ||
			!writerInventory.RequiresWriter ||
			writerInventory.WriterReady ||
			writerInventory.WriterState != "absent" ||
			writerInventory.WriterAction != "require_writer_contract" ||
			!writerInventory.RequiresRollback ||
			writerInventory.RollbackReady ||
			writerInventory.RollbackState != "absent" ||
			writerInventory.RollbackAction != "require_rollback_contract" ||
			writerInventory.InventoryState != "contracts_absent" ||
			writerInventory.InventoryAction != "name_required_contracts" ||
			writerInventory.WriterContract != "live_admission_writer.v1" ||
			writerInventory.RollbackContract != "live_admission_rollback.v1" ||
			writerInventory.AdmissionLedgerContract != "live_admission_ledger.v1" ||
			writerInventory.WriterContractPresent ||
			writerInventory.RollbackContractPresent ||
			writerInventory.LedgerContractPresent ||
			writerInventory.ContractsReady ||
			writerInventory.WriteAllowed ||
			writerInventory.MutatesState ||
			writerInventory.StageState != "staged_dry_run" ||
			writerInventory.StageAction != "stage_live_candidate_dry_run" ||
			writerInventory.WriterInventoryID == "" ||
			writerInventory.AdmissionWriterPreflightID != writerPreflight.WriterPreflightID ||
			writerInventory.AdmissionLiveStageID != liveStage.LiveStageID ||
			writerInventory.AdmissionEnableGateID != gate.EnableGateID ||
			writerInventory.AdmissionSwitchID != sw.SwitchID ||
			writerInventory.AdmissionPromotionID != promotion.PromotionID ||
			writerInventory.AdmissionDecisionID != decision.DecisionID ||
			writerInventory.CandidateExecutionID != execution.ExecutionID ||
			writerInventory.GeneratorAdapterID != generatorAdapter.AdapterID ||
			writerInventory.CandidateDraftID != draft.DraftID ||
			writerInventory.HandoffID != admission.HandoffID ||
			writerInventory.AdmissionAdapterID != admissionAdapter.AdmissionAdapterID ||
			writerInventory.DreamCandidateRunID != candidate.RunID ||
			writerInventory.CandidateRunID != draft.CandidateRunID ||
			writerInventory.CandidateTextHash != execution.GeneratedTextHash ||
			writerInventory.TurnTextHash != execution.TurnTextHash ||
			!writerInventory.AdmissionPolicyPassed ||
			!writerInventory.LiveRouteChoicePassed ||
			!writerInventory.SourceDecisionPassed ||
			!writerInventory.SourcePromotionPassed ||
			!writerInventory.SourceSwitchPassed ||
			!writerInventory.SourceEnablePassed ||
			!writerInventory.SourceStagePassed ||
			!writerInventory.SourceWriterPreflightPassed {
			return fmt.Errorf("bad nano-direct admission writer inventory receipt: writer_inventory=%+v writer_preflight=%+v stage=%+v gate=%+v switch=%+v promotion=%+v decision=%+v execution=%+v", writerInventory, writerPreflight, liveStage, gate, sw, promotion, decision, execution)
		}
	}
	if admissionLiveRouteTurnCandidateAdmissionWriterContractDryRun() {
		if writerContract.Schema != admissionLiveRouteTurnCandidateAdmissionWriterContractSchema ||
			!writerContract.Passed ||
			!writerContract.LiveReady ||
			writerContract.LiveAdmissionEnabled ||
			writerContract.AdmissionAllowed ||
			!writerContract.ManualEnableRequested ||
			!writerContract.EnableKeyMatched ||
			!writerContract.RequiresWriter ||
			writerContract.WriterReady ||
			writerContract.WriterState != "absent" ||
			writerContract.WriterAction != "require_writer_contract" ||
			!writerContract.RequiresRollback ||
			writerContract.RollbackReady ||
			writerContract.RollbackState != "absent" ||
			writerContract.RollbackAction != "require_rollback_contract" ||
			writerContract.InventoryState != "contracts_absent" ||
			writerContract.InventoryAction != "name_required_contracts" ||
			writerContract.ContractState != "shape_drafted_dry_run" ||
			writerContract.ContractAction != "define_writer_rollback_ledger_contract" ||
			writerContract.WriterContract != "live_admission_writer.v1" ||
			writerContract.RollbackContract != "live_admission_rollback.v1" ||
			writerContract.AdmissionLedgerContract != "live_admission_ledger.v1" ||
			writerContract.WriterContractShape != "append_shadow_candidate_receipt" ||
			writerContract.RollbackContractShape != "remove_exact_writer_receipt" ||
			writerContract.LedgerContractShape != "append_only_receipt_log" ||
			writerContract.WriteScope != "dream_candidate_admission" ||
			writerContract.RollbackScope != "single_writer_receipt" ||
			writerContract.LedgerMode != "append_only_dry_run" ||
			!writerContract.ContractShapeReady ||
			writerContract.SourceWriterContractPresent ||
			writerContract.SourceRollbackContractPresent ||
			writerContract.SourceLedgerContractPresent ||
			writerContract.WriterImplementationReady ||
			writerContract.RollbackImplementationReady ||
			writerContract.LedgerImplementationReady ||
			writerContract.ContractsReady ||
			writerContract.WriteAllowed ||
			writerContract.MutatesState ||
			writerContract.StageState != "staged_dry_run" ||
			writerContract.StageAction != "stage_live_candidate_dry_run" ||
			writerContract.WriterContractID == "" ||
			writerContract.AdmissionWriterInventoryID != writerInventory.WriterInventoryID ||
			writerContract.AdmissionWriterPreflightID != writerPreflight.WriterPreflightID ||
			writerContract.AdmissionLiveStageID != liveStage.LiveStageID ||
			writerContract.AdmissionEnableGateID != gate.EnableGateID ||
			writerContract.AdmissionSwitchID != sw.SwitchID ||
			writerContract.AdmissionPromotionID != promotion.PromotionID ||
			writerContract.AdmissionDecisionID != decision.DecisionID ||
			writerContract.CandidateExecutionID != execution.ExecutionID ||
			writerContract.GeneratorAdapterID != generatorAdapter.AdapterID ||
			writerContract.CandidateDraftID != draft.DraftID ||
			writerContract.HandoffID != admission.HandoffID ||
			writerContract.AdmissionAdapterID != admissionAdapter.AdmissionAdapterID ||
			writerContract.DreamCandidateRunID != candidate.RunID ||
			writerContract.CandidateRunID != draft.CandidateRunID ||
			writerContract.CandidateTextHash != execution.GeneratedTextHash ||
			writerContract.TurnTextHash != execution.TurnTextHash ||
			!writerContract.AdmissionPolicyPassed ||
			!writerContract.LiveRouteChoicePassed ||
			!writerContract.SourceDecisionPassed ||
			!writerContract.SourcePromotionPassed ||
			!writerContract.SourceSwitchPassed ||
			!writerContract.SourceEnablePassed ||
			!writerContract.SourceStagePassed ||
			!writerContract.SourceWriterPreflightPassed ||
			!writerContract.SourceWriterInventoryPassed {
			return fmt.Errorf("bad nano-direct admission writer contract receipt: writer_contract=%+v writer_inventory=%+v writer_preflight=%+v stage=%+v gate=%+v switch=%+v promotion=%+v decision=%+v execution=%+v", writerContract, writerInventory, writerPreflight, liveStage, gate, sw, promotion, decision, execution)
		}
	}
	if admissionLiveRouteTurnCandidateAdmissionLedgerDryRun() {
		if ledger.Schema != admissionLiveRouteTurnCandidateAdmissionLedgerSchema ||
			!ledger.Passed ||
			!ledger.LiveReady ||
			ledger.LiveAdmissionEnabled ||
			ledger.AdmissionAllowed ||
			!ledger.ManualEnableRequested ||
			!ledger.EnableKeyMatched ||
			!ledger.RequiresWriter ||
			ledger.WriterReady ||
			ledger.WriterState != "absent" ||
			ledger.WriterAction != "require_writer_contract" ||
			!ledger.RequiresRollback ||
			ledger.RollbackReady ||
			ledger.RollbackState != "absent" ||
			ledger.RollbackAction != "require_rollback_contract" ||
			ledger.InventoryState != "contracts_absent" ||
			ledger.InventoryAction != "name_required_contracts" ||
			ledger.ContractState != "shape_drafted_dry_run" ||
			ledger.ContractAction != "define_writer_rollback_ledger_contract" ||
			ledger.WriterContract != "live_admission_writer.v1" ||
			ledger.RollbackContract != "live_admission_rollback.v1" ||
			ledger.AdmissionLedgerContract != "live_admission_ledger.v1" ||
			ledger.WriterContractShape != "append_shadow_candidate_receipt" ||
			ledger.RollbackContractShape != "remove_exact_writer_receipt" ||
			ledger.LedgerContractShape != "append_only_receipt_log" ||
			ledger.WriteScope != "dream_candidate_admission" ||
			ledger.RollbackScope != "single_writer_receipt" ||
			!ledger.ContractShapeReady ||
			ledger.SourceWriterContractPresent ||
			ledger.SourceRollbackContractPresent ||
			ledger.SourceLedgerContractPresent ||
			ledger.WriterImplementationReady ||
			ledger.RollbackImplementationReady ||
			ledger.LedgerImplementationReady ||
			ledger.ContractsReady ||
			ledger.WriteAllowed ||
			ledger.MutatesState ||
			ledger.StageState != "staged_dry_run" ||
			ledger.StageAction != "stage_live_candidate_dry_run" ||
			ledger.LedgerState != "receipt_drafted_dry_run" ||
			ledger.LedgerAction != "append_candidate_admission_receipt_dry_run" ||
			ledger.LedgerContract != "live_admission_ledger.v1" ||
			ledger.LedgerMode != "append_only_dry_run" ||
			ledger.LedgerEntryKind != "dream_candidate_admission" ||
			ledger.LedgerEntryStatus != "shadow_candidate_receipt" ||
			ledger.LedgerReceiptShape != "candidate_contract_provenance" ||
			!ledger.LedgerAppendReady ||
			ledger.LedgerReceiptPersisted ||
			ledger.AdmissionLedgerID == "" ||
			ledger.AdmissionWriterContractID != writerContract.WriterContractID ||
			ledger.AdmissionWriterInventoryID != writerInventory.WriterInventoryID ||
			ledger.AdmissionWriterPreflightID != writerPreflight.WriterPreflightID ||
			ledger.AdmissionLiveStageID != liveStage.LiveStageID ||
			ledger.AdmissionEnableGateID != gate.EnableGateID ||
			ledger.AdmissionSwitchID != sw.SwitchID ||
			ledger.AdmissionPromotionID != promotion.PromotionID ||
			ledger.AdmissionDecisionID != decision.DecisionID ||
			ledger.CandidateExecutionID != execution.ExecutionID ||
			ledger.GeneratorAdapterID != generatorAdapter.AdapterID ||
			ledger.CandidateDraftID != draft.DraftID ||
			ledger.HandoffID != admission.HandoffID ||
			ledger.AdmissionAdapterID != admissionAdapter.AdmissionAdapterID ||
			ledger.DreamCandidateRunID != candidate.RunID ||
			ledger.CandidateRunID != draft.CandidateRunID ||
			ledger.CandidateTextHash != execution.GeneratedTextHash ||
			ledger.TurnTextHash != execution.TurnTextHash ||
			!ledger.AdmissionPolicyPassed ||
			!ledger.LiveRouteChoicePassed ||
			!ledger.SourceDecisionPassed ||
			!ledger.SourcePromotionPassed ||
			!ledger.SourceSwitchPassed ||
			!ledger.SourceEnablePassed ||
			!ledger.SourceStagePassed ||
			!ledger.SourceWriterPreflightPassed ||
			!ledger.SourceWriterInventoryPassed ||
			!ledger.SourceWriterContractPassed {
			return fmt.Errorf("bad nano-direct admission ledger receipt: ledger=%+v writer_contract=%+v writer_inventory=%+v writer_preflight=%+v stage=%+v gate=%+v switch=%+v promotion=%+v decision=%+v execution=%+v", ledger, writerContract, writerInventory, writerPreflight, liveStage, gate, sw, promotion, decision, execution)
		}
	}
	if admissionLiveRouteTurnCandidateAdmissionWriterImplementationDryRun() {
		if writerImpl.Schema != admissionLiveRouteTurnCandidateAdmissionWriterImplSchema ||
			!writerImpl.Passed ||
			!writerImpl.LiveReady ||
			writerImpl.LiveAdmissionEnabled ||
			writerImpl.AdmissionAllowed ||
			!writerImpl.ManualEnableRequested ||
			!writerImpl.EnableKeyMatched ||
			!writerImpl.RequiresWriter ||
			writerImpl.WriterReady ||
			writerImpl.WriterState != "absent" ||
			writerImpl.WriterAction != "require_writer_contract" ||
			!writerImpl.RequiresRollback ||
			writerImpl.RollbackReady ||
			writerImpl.RollbackState != "absent" ||
			writerImpl.RollbackAction != "require_rollback_contract" ||
			writerImpl.InventoryState != "contracts_absent" ||
			writerImpl.InventoryAction != "name_required_contracts" ||
			writerImpl.ContractState != "shape_drafted_dry_run" ||
			writerImpl.ContractAction != "define_writer_rollback_ledger_contract" ||
			writerImpl.WriterContract != "live_admission_writer.v1" ||
			writerImpl.RollbackContract != "live_admission_rollback.v1" ||
			writerImpl.AdmissionLedgerContract != "live_admission_ledger.v1" ||
			writerImpl.WriterContractShape != "append_shadow_candidate_receipt" ||
			writerImpl.RollbackContractShape != "remove_exact_writer_receipt" ||
			writerImpl.LedgerContractShape != "append_only_receipt_log" ||
			writerImpl.WriteScope != "dream_candidate_admission" ||
			writerImpl.RollbackScope != "single_writer_receipt" ||
			!writerImpl.ContractShapeReady ||
			writerImpl.SourceWriterContractPresent ||
			writerImpl.SourceRollbackContractPresent ||
			writerImpl.SourceLedgerContractPresent ||
			writerImpl.WriterImplementationReady ||
			writerImpl.RollbackImplementationReady ||
			writerImpl.LedgerImplementationReady ||
			writerImpl.ContractsReady ||
			writerImpl.WriteAllowed ||
			writerImpl.MutatesState ||
			writerImpl.StageState != "staged_dry_run" ||
			writerImpl.StageAction != "stage_live_candidate_dry_run" ||
			writerImpl.LedgerState != "receipt_drafted_dry_run" ||
			writerImpl.LedgerAction != "append_candidate_admission_receipt_dry_run" ||
			writerImpl.LedgerContract != "live_admission_ledger.v1" ||
			writerImpl.LedgerMode != "append_only_dry_run" ||
			writerImpl.LedgerEntryKind != "dream_candidate_admission" ||
			writerImpl.LedgerEntryStatus != "shadow_candidate_receipt" ||
			writerImpl.LedgerReceiptShape != "candidate_contract_provenance" ||
			!writerImpl.LedgerAppendReady ||
			writerImpl.LedgerReceiptPersisted ||
			writerImpl.ImplementationState != "implementation_contract_drafted_dry_run" ||
			writerImpl.ImplementationAction != "define_append_only_writer_ledger_rollback" ||
			writerImpl.WriterEntrypoint != "append_shadow_candidate_receipt_dry_run" ||
			writerImpl.LedgerEntrypoint != "append_admission_ledger_receipt_dry_run" ||
			writerImpl.RollbackEntrypoint != "remove_exact_shadow_candidate_receipt_dry_run" ||
			writerImpl.WriteTarget != "shadow_receipt_log" ||
			writerImpl.BodyTarget != "none" ||
			!writerImpl.AppendOnly ||
			!writerImpl.RollbackRequired ||
			!writerImpl.ImplementationContractReady ||
			writerImpl.WriterImplementationID == "" ||
			writerImpl.AdmissionLedgerID != ledger.AdmissionLedgerID ||
			writerImpl.AdmissionWriterContractID != writerContract.WriterContractID ||
			writerImpl.AdmissionWriterInventoryID != writerInventory.WriterInventoryID ||
			writerImpl.AdmissionWriterPreflightID != writerPreflight.WriterPreflightID ||
			writerImpl.AdmissionLiveStageID != liveStage.LiveStageID ||
			writerImpl.AdmissionEnableGateID != gate.EnableGateID ||
			writerImpl.AdmissionSwitchID != sw.SwitchID ||
			writerImpl.AdmissionPromotionID != promotion.PromotionID ||
			writerImpl.AdmissionDecisionID != decision.DecisionID ||
			writerImpl.CandidateExecutionID != execution.ExecutionID ||
			writerImpl.GeneratorAdapterID != generatorAdapter.AdapterID ||
			writerImpl.CandidateDraftID != draft.DraftID ||
			writerImpl.HandoffID != admission.HandoffID ||
			writerImpl.AdmissionAdapterID != admissionAdapter.AdmissionAdapterID ||
			writerImpl.DreamCandidateRunID != candidate.RunID ||
			writerImpl.CandidateRunID != draft.CandidateRunID ||
			writerImpl.CandidateTextHash != execution.GeneratedTextHash ||
			writerImpl.TurnTextHash != execution.TurnTextHash ||
			!writerImpl.AdmissionPolicyPassed ||
			!writerImpl.LiveRouteChoicePassed ||
			!writerImpl.SourceDecisionPassed ||
			!writerImpl.SourcePromotionPassed ||
			!writerImpl.SourceSwitchPassed ||
			!writerImpl.SourceEnablePassed ||
			!writerImpl.SourceStagePassed ||
			!writerImpl.SourceWriterPreflightPassed ||
			!writerImpl.SourceWriterInventoryPassed ||
			!writerImpl.SourceWriterContractPassed ||
			!writerImpl.SourceLedgerPassed {
			return fmt.Errorf("bad nano-direct admission writer implementation receipt: writer_implementation=%+v ledger=%+v writer_contract=%+v writer_inventory=%+v writer_preflight=%+v stage=%+v gate=%+v switch=%+v promotion=%+v decision=%+v execution=%+v", writerImpl, ledger, writerContract, writerInventory, writerPreflight, liveStage, gate, sw, promotion, decision, execution)
		}
	}
	if admissionLiveRouteTurnCandidateAdmissionWriterReceiptDryRun() {
		if writerReceipt.Schema != admissionLiveRouteTurnCandidateAdmissionWriterReceiptSchema ||
			!writerReceipt.Passed ||
			!writerReceipt.LiveReady ||
			writerReceipt.LiveAdmissionEnabled ||
			writerReceipt.AdmissionAllowed ||
			!writerReceipt.ManualEnableRequested ||
			!writerReceipt.EnableKeyMatched ||
			!writerReceipt.RequiresWriter ||
			!writerReceipt.WriterReady ||
			writerReceipt.WriterState != "ready_dry_run" ||
			writerReceipt.WriterAction != "append_shadow_candidate_receipt_dry_run" ||
			!writerReceipt.RequiresRollback ||
			writerReceipt.RollbackReady ||
			writerReceipt.RollbackState != "absent" ||
			writerReceipt.RollbackAction != "require_rollback_contract" ||
			writerReceipt.InventoryState != "contracts_absent" ||
			writerReceipt.InventoryAction != "name_required_contracts" ||
			writerReceipt.ContractState != "shape_drafted_dry_run" ||
			writerReceipt.ContractAction != "define_writer_rollback_ledger_contract" ||
			writerReceipt.WriterContract != "live_admission_writer.v1" ||
			writerReceipt.RollbackContract != "live_admission_rollback.v1" ||
			writerReceipt.AdmissionLedgerContract != "live_admission_ledger.v1" ||
			writerReceipt.WriterContractShape != "append_shadow_candidate_receipt" ||
			writerReceipt.RollbackContractShape != "remove_exact_writer_receipt" ||
			writerReceipt.LedgerContractShape != "append_only_receipt_log" ||
			writerReceipt.WriteScope != "dream_candidate_admission" ||
			writerReceipt.RollbackScope != "single_writer_receipt" ||
			!writerReceipt.ContractShapeReady ||
			writerReceipt.SourceWriterContractPresent ||
			writerReceipt.SourceRollbackContractPresent ||
			writerReceipt.SourceLedgerContractPresent ||
			!writerReceipt.WriterImplementationReady ||
			writerReceipt.RollbackImplementationReady ||
			writerReceipt.LedgerImplementationReady ||
			writerReceipt.ContractsReady ||
			writerReceipt.WriteAllowed ||
			writerReceipt.MutatesState ||
			writerReceipt.StageState != "staged_dry_run" ||
			writerReceipt.StageAction != "stage_live_candidate_dry_run" ||
			writerReceipt.LedgerState != "receipt_drafted_dry_run" ||
			writerReceipt.LedgerAction != "append_candidate_admission_receipt_dry_run" ||
			writerReceipt.LedgerContract != "live_admission_ledger.v1" ||
			writerReceipt.LedgerMode != "append_only_dry_run" ||
			writerReceipt.LedgerEntryKind != "dream_candidate_admission" ||
			writerReceipt.LedgerEntryStatus != "shadow_candidate_receipt" ||
			writerReceipt.LedgerReceiptShape != "candidate_contract_provenance" ||
			!writerReceipt.LedgerAppendReady ||
			writerReceipt.LedgerReceiptPersisted ||
			writerReceipt.ImplementationState != "implementation_contract_drafted_dry_run" ||
			writerReceipt.ImplementationAction != "define_append_only_writer_ledger_rollback" ||
			writerReceipt.WriterEntrypoint != "append_shadow_candidate_receipt_dry_run" ||
			writerReceipt.LedgerEntrypoint != "append_admission_ledger_receipt_dry_run" ||
			writerReceipt.RollbackEntrypoint != "remove_exact_shadow_candidate_receipt_dry_run" ||
			writerReceipt.WriteTarget != "shadow_receipt_log" ||
			writerReceipt.BodyTarget != "none" ||
			!writerReceipt.AppendOnly ||
			!writerReceipt.RollbackRequired ||
			!writerReceipt.ImplementationContractReady ||
			writerReceipt.WriterImplementationID == "" ||
			writerReceipt.WriterReceiptState != "shadow_receipt_appended_dry_run" ||
			writerReceipt.WriterReceiptAction != "append_shadow_candidate_receipt_dry_run" ||
			writerReceipt.WriterReceiptKind != "dream_candidate_admission" ||
			writerReceipt.WriterReceiptTarget != "shadow_receipt_log" ||
			writerReceipt.WriterReceiptMode != "append_only_dry_run" ||
			writerReceipt.WriterReceiptShape != "candidate_contract_provenance" ||
			!writerReceipt.WriterReceiptPersisted ||
			!writerReceipt.ShadowWriteAllowed ||
			!writerReceipt.SourceWriterImplementationPassed ||
			writerReceipt.SourceWriterImplementationID != writerImpl.WriterImplementationID ||
			writerReceipt.SourceWriterImplementationEntrypoint != "append_shadow_candidate_receipt_dry_run" ||
			writerReceipt.SourceLedgerImplementationEntrypoint != "append_admission_ledger_receipt_dry_run" ||
			writerReceipt.SourceRollbackImplementationEntrypoint != "remove_exact_shadow_candidate_receipt_dry_run" ||
			writerReceipt.WriterReceiptID == "" ||
			writerReceipt.AdmissionLedgerID != ledger.AdmissionLedgerID ||
			writerReceipt.AdmissionWriterContractID != writerContract.WriterContractID ||
			writerReceipt.AdmissionWriterInventoryID != writerInventory.WriterInventoryID ||
			writerReceipt.AdmissionWriterPreflightID != writerPreflight.WriterPreflightID ||
			writerReceipt.AdmissionLiveStageID != liveStage.LiveStageID ||
			writerReceipt.AdmissionEnableGateID != gate.EnableGateID ||
			writerReceipt.AdmissionSwitchID != sw.SwitchID ||
			writerReceipt.AdmissionPromotionID != promotion.PromotionID ||
			writerReceipt.AdmissionDecisionID != decision.DecisionID ||
			writerReceipt.CandidateExecutionID != execution.ExecutionID ||
			writerReceipt.GeneratorAdapterID != generatorAdapter.AdapterID ||
			writerReceipt.CandidateDraftID != draft.DraftID ||
			writerReceipt.HandoffID != admission.HandoffID ||
			writerReceipt.AdmissionAdapterID != admissionAdapter.AdmissionAdapterID ||
			writerReceipt.DreamCandidateRunID != candidate.RunID ||
			writerReceipt.CandidateRunID != draft.CandidateRunID ||
			writerReceipt.CandidateTextHash != execution.GeneratedTextHash ||
			writerReceipt.TurnTextHash != execution.TurnTextHash ||
			!writerReceipt.AdmissionPolicyPassed ||
			!writerReceipt.LiveRouteChoicePassed ||
			!writerReceipt.SourceDecisionPassed ||
			!writerReceipt.SourcePromotionPassed ||
			!writerReceipt.SourceSwitchPassed ||
			!writerReceipt.SourceEnablePassed ||
			!writerReceipt.SourceStagePassed ||
			!writerReceipt.SourceWriterPreflightPassed ||
			!writerReceipt.SourceWriterInventoryPassed ||
			!writerReceipt.SourceWriterContractPassed ||
			!writerReceipt.SourceLedgerPassed ||
			writerReceipt.Reason != "shadow writer receipt appended as dry-run; body write remains disabled" {
			return fmt.Errorf("bad nano-direct admission writer receipt: writer_receipt=%+v writer_implementation=%+v ledger=%+v writer_contract=%+v writer_inventory=%+v writer_preflight=%+v stage=%+v gate=%+v switch=%+v promotion=%+v decision=%+v execution=%+v", writerReceipt, writerImpl, ledger, writerContract, writerInventory, writerPreflight, liveStage, gate, sw, promotion, decision, execution)
		}
	}
	if admissionLiveRouteTurnCandidateAdmissionRollbackImplementationDryRun() {
		if rollbackImpl.Schema != admissionLiveRouteTurnCandidateAdmissionRollbackImplSchema ||
			!rollbackImpl.Passed ||
			!rollbackImpl.LiveReady ||
			rollbackImpl.LiveAdmissionEnabled ||
			rollbackImpl.AdmissionAllowed ||
			!rollbackImpl.ManualEnableRequested ||
			!rollbackImpl.EnableKeyMatched ||
			!rollbackImpl.RequiresWriter ||
			!rollbackImpl.WriterReady ||
			rollbackImpl.WriterState != "ready_dry_run" ||
			rollbackImpl.WriterAction != "append_shadow_candidate_receipt_dry_run" ||
			!rollbackImpl.RequiresRollback ||
			!rollbackImpl.RollbackReady ||
			rollbackImpl.RollbackState != "ready_dry_run" ||
			rollbackImpl.RollbackAction != "remove_exact_shadow_candidate_receipt_dry_run" ||
			rollbackImpl.InventoryState != "contracts_absent" ||
			rollbackImpl.InventoryAction != "name_required_contracts" ||
			rollbackImpl.ContractState != "shape_drafted_dry_run" ||
			rollbackImpl.ContractAction != "define_writer_rollback_ledger_contract" ||
			rollbackImpl.WriterContract != "live_admission_writer.v1" ||
			rollbackImpl.RollbackContract != "live_admission_rollback.v1" ||
			rollbackImpl.AdmissionLedgerContract != "live_admission_ledger.v1" ||
			rollbackImpl.WriterContractShape != "append_shadow_candidate_receipt" ||
			rollbackImpl.RollbackContractShape != "remove_exact_writer_receipt" ||
			rollbackImpl.LedgerContractShape != "append_only_receipt_log" ||
			rollbackImpl.WriteScope != "dream_candidate_admission" ||
			rollbackImpl.RollbackScope != "single_writer_receipt" ||
			!rollbackImpl.ContractShapeReady ||
			rollbackImpl.SourceWriterContractPresent ||
			rollbackImpl.SourceRollbackContractPresent ||
			rollbackImpl.SourceLedgerContractPresent ||
			!rollbackImpl.WriterImplementationReady ||
			!rollbackImpl.RollbackImplementationReady ||
			rollbackImpl.LedgerImplementationReady ||
			rollbackImpl.ContractsReady ||
			rollbackImpl.WriteAllowed ||
			rollbackImpl.MutatesState ||
			rollbackImpl.StageState != "staged_dry_run" ||
			rollbackImpl.StageAction != "stage_live_candidate_dry_run" ||
			rollbackImpl.LedgerState != "receipt_drafted_dry_run" ||
			rollbackImpl.LedgerAction != "append_candidate_admission_receipt_dry_run" ||
			rollbackImpl.LedgerContract != "live_admission_ledger.v1" ||
			rollbackImpl.LedgerMode != "append_only_dry_run" ||
			rollbackImpl.LedgerEntryKind != "dream_candidate_admission" ||
			rollbackImpl.LedgerEntryStatus != "shadow_candidate_receipt" ||
			rollbackImpl.LedgerReceiptShape != "candidate_contract_provenance" ||
			!rollbackImpl.LedgerAppendReady ||
			rollbackImpl.LedgerReceiptPersisted ||
			rollbackImpl.ImplementationState != "implementation_contract_drafted_dry_run" ||
			rollbackImpl.ImplementationAction != "define_append_only_writer_ledger_rollback" ||
			rollbackImpl.WriterEntrypoint != "append_shadow_candidate_receipt_dry_run" ||
			rollbackImpl.LedgerEntrypoint != "append_admission_ledger_receipt_dry_run" ||
			rollbackImpl.RollbackEntrypoint != "remove_exact_shadow_candidate_receipt_dry_run" ||
			rollbackImpl.WriteTarget != "shadow_receipt_log" ||
			rollbackImpl.BodyTarget != "none" ||
			!rollbackImpl.AppendOnly ||
			!rollbackImpl.RollbackRequired ||
			!rollbackImpl.ImplementationContractReady ||
			rollbackImpl.WriterImplementationID == "" ||
			rollbackImpl.WriterReceiptState != "shadow_receipt_appended_dry_run" ||
			rollbackImpl.WriterReceiptAction != "append_shadow_candidate_receipt_dry_run" ||
			rollbackImpl.WriterReceiptKind != "dream_candidate_admission" ||
			rollbackImpl.WriterReceiptTarget != "shadow_receipt_log" ||
			rollbackImpl.WriterReceiptMode != "append_only_dry_run" ||
			rollbackImpl.WriterReceiptShape != "candidate_contract_provenance" ||
			!rollbackImpl.WriterReceiptPersisted ||
			!rollbackImpl.ShadowWriteAllowed ||
			!rollbackImpl.SourceWriterImplementationPassed ||
			rollbackImpl.SourceWriterImplementationID != writerImpl.WriterImplementationID ||
			rollbackImpl.SourceWriterImplementationEntrypoint != "append_shadow_candidate_receipt_dry_run" ||
			rollbackImpl.SourceLedgerImplementationEntrypoint != "append_admission_ledger_receipt_dry_run" ||
			rollbackImpl.SourceRollbackImplementationEntrypoint != "remove_exact_shadow_candidate_receipt_dry_run" ||
			rollbackImpl.WriterReceiptID == "" ||
			rollbackImpl.RollbackImplementationState != "rollback_contract_drafted_dry_run" ||
			rollbackImpl.RollbackImplementationAction != "remove_exact_shadow_candidate_receipt_dry_run" ||
			rollbackImpl.RollbackEntrypointResolved != "remove_exact_shadow_candidate_receipt_dry_run" ||
			rollbackImpl.RollbackTarget != "shadow_receipt_log" ||
			rollbackImpl.RollbackTargetKind != "dream_candidate_admission" ||
			rollbackImpl.RollbackTargetID != writerReceipt.WriterReceiptID ||
			rollbackImpl.RollbackMode != "exact_receipt_id_dry_run" ||
			!rollbackImpl.ExactReceiptMatchRequired ||
			!rollbackImpl.RollbackDryRunOnly ||
			rollbackImpl.RollbackReceiptRemoved ||
			rollbackImpl.SourceWriterReceiptSchema != admissionLiveRouteTurnCandidateAdmissionWriterReceiptSchema ||
			!rollbackImpl.SourceWriterReceiptPassed ||
			rollbackImpl.SourceWriterReceiptID != writerReceipt.WriterReceiptID ||
			rollbackImpl.SourceWriterReceiptAction != "append_shadow_candidate_receipt_dry_run" ||
			!rollbackImpl.SourceWriterReceiptPersisted ||
			!rollbackImpl.SourceWriterReceiptShadowWritable ||
			rollbackImpl.RollbackImplementationID == "" ||
			rollbackImpl.AdmissionLedgerID != ledger.AdmissionLedgerID ||
			rollbackImpl.AdmissionWriterContractID != writerContract.WriterContractID ||
			rollbackImpl.AdmissionWriterInventoryID != writerInventory.WriterInventoryID ||
			rollbackImpl.AdmissionWriterPreflightID != writerPreflight.WriterPreflightID ||
			rollbackImpl.AdmissionLiveStageID != liveStage.LiveStageID ||
			rollbackImpl.AdmissionEnableGateID != gate.EnableGateID ||
			rollbackImpl.AdmissionSwitchID != sw.SwitchID ||
			rollbackImpl.AdmissionPromotionID != promotion.PromotionID ||
			rollbackImpl.AdmissionDecisionID != decision.DecisionID ||
			rollbackImpl.CandidateExecutionID != execution.ExecutionID ||
			rollbackImpl.GeneratorAdapterID != generatorAdapter.AdapterID ||
			rollbackImpl.CandidateDraftID != draft.DraftID ||
			rollbackImpl.HandoffID != admission.HandoffID ||
			rollbackImpl.AdmissionAdapterID != admissionAdapter.AdmissionAdapterID ||
			rollbackImpl.DreamCandidateRunID != candidate.RunID ||
			rollbackImpl.CandidateRunID != draft.CandidateRunID ||
			rollbackImpl.CandidateTextHash != execution.GeneratedTextHash ||
			rollbackImpl.TurnTextHash != execution.TurnTextHash ||
			!rollbackImpl.AdmissionPolicyPassed ||
			!rollbackImpl.LiveRouteChoicePassed ||
			!rollbackImpl.SourceDecisionPassed ||
			!rollbackImpl.SourcePromotionPassed ||
			!rollbackImpl.SourceSwitchPassed ||
			!rollbackImpl.SourceEnablePassed ||
			!rollbackImpl.SourceStagePassed ||
			!rollbackImpl.SourceWriterPreflightPassed ||
			!rollbackImpl.SourceWriterInventoryPassed ||
			!rollbackImpl.SourceWriterContractPassed ||
			!rollbackImpl.SourceLedgerPassed ||
			rollbackImpl.Reason != "rollback implementation drafted for exact writer receipt; body write remains disabled" {
			return fmt.Errorf("bad nano-direct admission rollback implementation receipt: rollback_implementation=%+v writer_receipt=%+v writer_implementation=%+v ledger=%+v writer_contract=%+v writer_inventory=%+v writer_preflight=%+v stage=%+v gate=%+v switch=%+v promotion=%+v decision=%+v execution=%+v", rollbackImpl, writerReceipt, writerImpl, ledger, writerContract, writerInventory, writerPreflight, liveStage, gate, sw, promotion, decision, execution)
		}
	}
	if admissionLiveRouteTurnCandidateAdmissionLedgerImplementationDryRun() {
		if ledgerImpl.Schema != admissionLiveRouteTurnCandidateAdmissionLedgerImplSchema ||
			!ledgerImpl.Passed ||
			!ledgerImpl.LiveReady ||
			ledgerImpl.LiveAdmissionEnabled ||
			ledgerImpl.AdmissionAllowed ||
			!ledgerImpl.ManualEnableRequested ||
			!ledgerImpl.EnableKeyMatched ||
			!ledgerImpl.RequiresWriter ||
			!ledgerImpl.WriterReady ||
			ledgerImpl.WriterState != "ready_dry_run" ||
			ledgerImpl.WriterAction != "append_shadow_candidate_receipt_dry_run" ||
			!ledgerImpl.RequiresRollback ||
			!ledgerImpl.RollbackReady ||
			ledgerImpl.RollbackState != "ready_dry_run" ||
			ledgerImpl.RollbackAction != "remove_exact_shadow_candidate_receipt_dry_run" ||
			!ledgerImpl.WriterImplementationReady ||
			!ledgerImpl.RollbackImplementationReady ||
			!ledgerImpl.LedgerImplementationReady ||
			ledgerImpl.ContractsReady ||
			ledgerImpl.WriteAllowed ||
			ledgerImpl.MutatesState ||
			ledgerImpl.ImplementationState != "implementation_contract_drafted_dry_run" ||
			ledgerImpl.ImplementationAction != "define_append_only_writer_ledger_rollback" ||
			ledgerImpl.WriterEntrypoint != "append_shadow_candidate_receipt_dry_run" ||
			ledgerImpl.LedgerEntrypoint != "append_admission_ledger_receipt_dry_run" ||
			ledgerImpl.RollbackEntrypoint != "remove_exact_shadow_candidate_receipt_dry_run" ||
			ledgerImpl.WriteTarget != "shadow_receipt_log" ||
			ledgerImpl.BodyTarget != "none" ||
			!ledgerImpl.AppendOnly ||
			!ledgerImpl.RollbackRequired ||
			!ledgerImpl.ImplementationContractReady ||
			ledgerImpl.LedgerState != "receipt_drafted_dry_run" ||
			ledgerImpl.LedgerAction != "append_candidate_admission_receipt_dry_run" ||
			ledgerImpl.LedgerContract != "live_admission_ledger.v1" ||
			ledgerImpl.LedgerMode != "append_only_dry_run" ||
			ledgerImpl.LedgerEntryKind != "dream_candidate_admission" ||
			ledgerImpl.LedgerEntryStatus != "shadow_candidate_receipt" ||
			ledgerImpl.LedgerReceiptShape != "candidate_contract_provenance" ||
			!ledgerImpl.LedgerAppendReady ||
			ledgerImpl.LedgerReceiptPersisted ||
			ledgerImpl.WriterReceiptID != writerReceipt.WriterReceiptID ||
			ledgerImpl.RollbackImplementationID != rollbackImpl.RollbackImplementationID ||
			ledgerImpl.RollbackImplementationState != "rollback_contract_drafted_dry_run" ||
			ledgerImpl.RollbackImplementationAction != "remove_exact_shadow_candidate_receipt_dry_run" ||
			ledgerImpl.RollbackEntrypointResolved != "remove_exact_shadow_candidate_receipt_dry_run" ||
			ledgerImpl.RollbackTarget != "shadow_receipt_log" ||
			ledgerImpl.RollbackTargetKind != "dream_candidate_admission" ||
			ledgerImpl.RollbackTargetID != writerReceipt.WriterReceiptID ||
			ledgerImpl.RollbackMode != "exact_receipt_id_dry_run" ||
			!ledgerImpl.ExactReceiptMatchRequired ||
			!ledgerImpl.RollbackDryRunOnly ||
			ledgerImpl.RollbackReceiptRemoved ||
			ledgerImpl.LedgerImplementationState != "ledger_contract_drafted_dry_run" ||
			ledgerImpl.LedgerImplementationAction != "append_admission_ledger_receipt_dry_run" ||
			ledgerImpl.LedgerEntrypointResolved != "append_admission_ledger_receipt_dry_run" ||
			ledgerImpl.LedgerImplementationTarget != "admission_ledger" ||
			ledgerImpl.LedgerImplementationTargetKind != "dream_candidate_admission" ||
			ledgerImpl.LedgerImplementationTargetMode != "append_only_dry_run" ||
			!ledgerImpl.LedgerImplementationAppendOnly ||
			!ledgerImpl.LedgerImplementationDryRunOnly ||
			ledgerImpl.LedgerImplementationReceiptPersisted ||
			ledgerImpl.SourceRollbackImplementationSchema != admissionLiveRouteTurnCandidateAdmissionRollbackImplSchema ||
			!ledgerImpl.SourceRollbackImplementationPassed ||
			ledgerImpl.SourceRollbackImplementationID != rollbackImpl.RollbackImplementationID ||
			ledgerImpl.SourceRollbackImplementationAction != "remove_exact_shadow_candidate_receipt_dry_run" ||
			!ledgerImpl.SourceRollbackImplementationReady ||
			ledgerImpl.SourceRollbackTargetID != writerReceipt.WriterReceiptID ||
			ledgerImpl.SourceWriterReceiptIDForLedger != writerReceipt.WriterReceiptID ||
			ledgerImpl.LedgerImplementationID == "" ||
			ledgerImpl.AdmissionLedgerID != ledger.AdmissionLedgerID ||
			ledgerImpl.AdmissionWriterContractID != writerContract.WriterContractID ||
			ledgerImpl.AdmissionWriterInventoryID != writerInventory.WriterInventoryID ||
			ledgerImpl.AdmissionWriterPreflightID != writerPreflight.WriterPreflightID ||
			ledgerImpl.AdmissionLiveStageID != liveStage.LiveStageID ||
			ledgerImpl.AdmissionEnableGateID != gate.EnableGateID ||
			ledgerImpl.AdmissionSwitchID != sw.SwitchID ||
			ledgerImpl.AdmissionPromotionID != promotion.PromotionID ||
			ledgerImpl.AdmissionDecisionID != decision.DecisionID ||
			ledgerImpl.CandidateExecutionID != execution.ExecutionID ||
			ledgerImpl.GeneratorAdapterID != generatorAdapter.AdapterID ||
			ledgerImpl.CandidateDraftID != draft.DraftID ||
			ledgerImpl.HandoffID != admission.HandoffID ||
			ledgerImpl.AdmissionAdapterID != admissionAdapter.AdmissionAdapterID ||
			ledgerImpl.DreamCandidateRunID != candidate.RunID ||
			ledgerImpl.CandidateRunID != draft.CandidateRunID ||
			ledgerImpl.CandidateTextHash != execution.GeneratedTextHash ||
			ledgerImpl.TurnTextHash != execution.TurnTextHash ||
			!ledgerImpl.AdmissionPolicyPassed ||
			!ledgerImpl.LiveRouteChoicePassed ||
			!ledgerImpl.SourceDecisionPassed ||
			!ledgerImpl.SourcePromotionPassed ||
			!ledgerImpl.SourceSwitchPassed ||
			!ledgerImpl.SourceEnablePassed ||
			!ledgerImpl.SourceStagePassed ||
			!ledgerImpl.SourceWriterPreflightPassed ||
			!ledgerImpl.SourceWriterInventoryPassed ||
			!ledgerImpl.SourceWriterContractPassed ||
			!ledgerImpl.SourceLedgerPassed ||
			ledgerImpl.Reason != "ledger implementation drafted for append-only admission receipts; contracts remain disabled" {
			return fmt.Errorf("bad nano-direct admission ledger implementation receipt: ledger_implementation=%+v rollback_implementation=%+v writer_receipt=%+v writer_implementation=%+v ledger=%+v writer_contract=%+v writer_inventory=%+v writer_preflight=%+v stage=%+v gate=%+v switch=%+v promotion=%+v decision=%+v execution=%+v", ledgerImpl, rollbackImpl, writerReceipt, writerImpl, ledger, writerContract, writerInventory, writerPreflight, liveStage, gate, sw, promotion, decision, execution)
		}
	}
	if admissionLiveRouteTurnCandidateAdmissionLedgerPersistenceDryRun() {
		if ledgerPersistence.Schema != admissionLiveRouteTurnCandidateAdmissionLedgerPersistenceSchema ||
			!ledgerPersistence.Passed ||
			!ledgerPersistence.LiveReady ||
			ledgerPersistence.LiveAdmissionEnabled ||
			ledgerPersistence.AdmissionAllowed ||
			!ledgerPersistence.ManualEnableRequested ||
			!ledgerPersistence.EnableKeyMatched ||
			!ledgerPersistence.RequiresWriter ||
			!ledgerPersistence.WriterReady ||
			ledgerPersistence.WriterState != "ready_dry_run" ||
			ledgerPersistence.WriterAction != "append_shadow_candidate_receipt_dry_run" ||
			!ledgerPersistence.RequiresRollback ||
			!ledgerPersistence.RollbackReady ||
			ledgerPersistence.RollbackState != "ready_dry_run" ||
			ledgerPersistence.RollbackAction != "remove_exact_shadow_candidate_receipt_dry_run" ||
			!ledgerPersistence.WriterImplementationReady ||
			!ledgerPersistence.RollbackImplementationReady ||
			!ledgerPersistence.LedgerImplementationReady ||
			ledgerPersistence.ContractsReady ||
			ledgerPersistence.WriteAllowed ||
			ledgerPersistence.MutatesState ||
			ledgerPersistence.BodyTarget != "none" ||
			ledgerPersistence.LedgerState != "receipt_drafted_dry_run" ||
			ledgerPersistence.LedgerAction != "append_candidate_admission_receipt_dry_run" ||
			ledgerPersistence.LedgerContract != "live_admission_ledger.v1" ||
			ledgerPersistence.LedgerMode != "append_only_dry_run" ||
			ledgerPersistence.LedgerEntryKind != "dream_candidate_admission" ||
			ledgerPersistence.LedgerEntryStatus != "shadow_candidate_receipt" ||
			ledgerPersistence.LedgerReceiptShape != "candidate_contract_provenance" ||
			!ledgerPersistence.LedgerAppendReady ||
			ledgerPersistence.LedgerReceiptPersisted ||
			ledgerPersistence.LedgerImplementationState != "ledger_contract_drafted_dry_run" ||
			ledgerPersistence.LedgerImplementationAction != "append_admission_ledger_receipt_dry_run" ||
			ledgerPersistence.LedgerEntrypointResolved != "append_admission_ledger_receipt_dry_run" ||
			ledgerPersistence.LedgerImplementationTarget != "admission_ledger" ||
			ledgerPersistence.LedgerImplementationTargetKind != "dream_candidate_admission" ||
			ledgerPersistence.LedgerImplementationTargetMode != "append_only_dry_run" ||
			!ledgerPersistence.LedgerImplementationAppendOnly ||
			!ledgerPersistence.LedgerImplementationDryRunOnly ||
			ledgerPersistence.LedgerImplementationReceiptPersisted ||
			ledgerPersistence.LedgerPersistenceState != "ledger_receipt_persisted_dry_run" ||
			ledgerPersistence.LedgerPersistenceAction != "append_admission_ledger_receipt_dry_run" ||
			ledgerPersistence.LedgerPersistenceTarget != "admission_ledger" ||
			ledgerPersistence.LedgerPersistenceTargetKind != "dream_candidate_admission" ||
			ledgerPersistence.LedgerPersistenceTargetMode != "append_only_dry_run" ||
			ledgerPersistence.LedgerPersistenceReceiptShape != "candidate_contract_provenance" ||
			!ledgerPersistence.LedgerPersistenceAppendOnly ||
			!ledgerPersistence.LedgerPersistenceDryRunOnly ||
			!ledgerPersistence.LedgerPersistenceReceiptPersisted ||
			!ledgerPersistence.LedgerPersistenceReady ||
			ledgerPersistence.SourceLedgerImplementationSchema != admissionLiveRouteTurnCandidateAdmissionLedgerImplSchema ||
			!ledgerPersistence.SourceLedgerImplementationPassed ||
			ledgerPersistence.SourceLedgerImplementationID != ledgerImpl.LedgerImplementationID ||
			ledgerPersistence.SourceLedgerImplementationAction != "append_admission_ledger_receipt_dry_run" ||
			!ledgerPersistence.SourceLedgerImplementationReady ||
			ledgerPersistence.SourceAdmissionLedgerIDForPersistence != ledger.AdmissionLedgerID ||
			ledgerPersistence.SourceRollbackImplementationIDForLedger != rollbackImpl.RollbackImplementationID ||
			ledgerPersistence.SourceWriterReceiptIDForLedgerPersistence != writerReceipt.WriterReceiptID ||
			ledgerPersistence.LedgerPersistenceID == "" ||
			ledgerPersistence.LedgerImplementationID != ledgerImpl.LedgerImplementationID ||
			ledgerPersistence.RollbackImplementationID != rollbackImpl.RollbackImplementationID ||
			ledgerPersistence.WriterReceiptID != writerReceipt.WriterReceiptID ||
			ledgerPersistence.AdmissionLedgerID != ledger.AdmissionLedgerID ||
			ledgerPersistence.AdmissionWriterContractID != writerContract.WriterContractID ||
			ledgerPersistence.AdmissionWriterInventoryID != writerInventory.WriterInventoryID ||
			ledgerPersistence.AdmissionWriterPreflightID != writerPreflight.WriterPreflightID ||
			ledgerPersistence.AdmissionLiveStageID != liveStage.LiveStageID ||
			ledgerPersistence.AdmissionEnableGateID != gate.EnableGateID ||
			ledgerPersistence.AdmissionSwitchID != sw.SwitchID ||
			ledgerPersistence.AdmissionPromotionID != promotion.PromotionID ||
			ledgerPersistence.AdmissionDecisionID != decision.DecisionID ||
			ledgerPersistence.CandidateExecutionID != execution.ExecutionID ||
			ledgerPersistence.GeneratorAdapterID != generatorAdapter.AdapterID ||
			ledgerPersistence.CandidateDraftID != draft.DraftID ||
			ledgerPersistence.HandoffID != admission.HandoffID ||
			ledgerPersistence.AdmissionAdapterID != admissionAdapter.AdmissionAdapterID ||
			ledgerPersistence.DreamCandidateRunID != candidate.RunID ||
			ledgerPersistence.CandidateRunID != draft.CandidateRunID ||
			ledgerPersistence.CandidateTextHash != execution.GeneratedTextHash ||
			ledgerPersistence.TurnTextHash != execution.TurnTextHash ||
			!ledgerPersistence.AdmissionPolicyPassed ||
			!ledgerPersistence.LiveRouteChoicePassed ||
			!ledgerPersistence.SourceDecisionPassed ||
			!ledgerPersistence.SourcePromotionPassed ||
			!ledgerPersistence.SourceSwitchPassed ||
			!ledgerPersistence.SourceEnablePassed ||
			!ledgerPersistence.SourceStagePassed ||
			!ledgerPersistence.SourceWriterPreflightPassed ||
			!ledgerPersistence.SourceWriterInventoryPassed ||
			!ledgerPersistence.SourceWriterContractPassed ||
			!ledgerPersistence.SourceLedgerPassed ||
			ledgerPersistence.Reason != "ledger receipt persisted to append-only dry-run log; live admission remains disabled" {
			return fmt.Errorf("bad nano-direct admission ledger persistence receipt: ledger_persistence=%+v ledger_implementation=%+v rollback_implementation=%+v writer_receipt=%+v writer_implementation=%+v ledger=%+v writer_contract=%+v writer_inventory=%+v writer_preflight=%+v stage=%+v gate=%+v switch=%+v promotion=%+v decision=%+v execution=%+v", ledgerPersistence, ledgerImpl, rollbackImpl, writerReceipt, writerImpl, ledger, writerContract, writerInventory, writerPreflight, liveStage, gate, sw, promotion, decision, execution)
		}
	}
	if admissionLiveRouteTurnCandidateAdmissionLedgerVerificationDryRun() {
		if ledgerVerification.Schema != admissionLiveRouteTurnCandidateAdmissionLedgerVerificationSchema ||
			!ledgerVerification.Passed ||
			!ledgerVerification.LiveReady ||
			ledgerVerification.LiveAdmissionEnabled ||
			ledgerVerification.AdmissionAllowed ||
			!ledgerVerification.ManualEnableRequested ||
			!ledgerVerification.EnableKeyMatched ||
			!ledgerVerification.RequiresWriter ||
			!ledgerVerification.WriterReady ||
			ledgerVerification.WriterState != "ready_dry_run" ||
			ledgerVerification.WriterAction != "append_shadow_candidate_receipt_dry_run" ||
			!ledgerVerification.RequiresRollback ||
			!ledgerVerification.RollbackReady ||
			ledgerVerification.RollbackState != "ready_dry_run" ||
			ledgerVerification.RollbackAction != "remove_exact_shadow_candidate_receipt_dry_run" ||
			!ledgerVerification.WriterImplementationReady ||
			!ledgerVerification.RollbackImplementationReady ||
			!ledgerVerification.LedgerImplementationReady ||
			ledgerVerification.ContractsReady ||
			ledgerVerification.WriteAllowed ||
			ledgerVerification.MutatesState ||
			ledgerVerification.BodyTarget != "none" ||
			ledgerVerification.LedgerState != "receipt_drafted_dry_run" ||
			ledgerVerification.LedgerAction != "append_candidate_admission_receipt_dry_run" ||
			ledgerVerification.LedgerContract != "live_admission_ledger.v1" ||
			ledgerVerification.LedgerMode != "append_only_dry_run" ||
			ledgerVerification.LedgerEntryKind != "dream_candidate_admission" ||
			ledgerVerification.LedgerEntryStatus != "shadow_candidate_receipt" ||
			ledgerVerification.LedgerReceiptShape != "candidate_contract_provenance" ||
			!ledgerVerification.LedgerAppendReady ||
			ledgerVerification.LedgerReceiptPersisted ||
			ledgerVerification.LedgerImplementationState != "ledger_contract_drafted_dry_run" ||
			ledgerVerification.LedgerImplementationAction != "append_admission_ledger_receipt_dry_run" ||
			ledgerVerification.LedgerEntrypointResolved != "append_admission_ledger_receipt_dry_run" ||
			ledgerVerification.LedgerImplementationTarget != "admission_ledger" ||
			ledgerVerification.LedgerImplementationTargetKind != "dream_candidate_admission" ||
			ledgerVerification.LedgerImplementationTargetMode != "append_only_dry_run" ||
			!ledgerVerification.LedgerImplementationAppendOnly ||
			!ledgerVerification.LedgerImplementationDryRunOnly ||
			ledgerVerification.LedgerImplementationReceiptPersisted ||
			ledgerVerification.LedgerPersistenceState != "ledger_receipt_persisted_dry_run" ||
			ledgerVerification.LedgerPersistenceAction != "append_admission_ledger_receipt_dry_run" ||
			ledgerVerification.LedgerPersistenceTarget != "admission_ledger" ||
			ledgerVerification.LedgerPersistenceTargetKind != "dream_candidate_admission" ||
			ledgerVerification.LedgerPersistenceTargetMode != "append_only_dry_run" ||
			ledgerVerification.LedgerPersistenceReceiptShape != "candidate_contract_provenance" ||
			!ledgerVerification.LedgerPersistenceAppendOnly ||
			!ledgerVerification.LedgerPersistenceDryRunOnly ||
			!ledgerVerification.LedgerPersistenceReceiptPersisted ||
			!ledgerVerification.LedgerPersistenceReady ||
			ledgerVerification.LedgerVerificationState != "ledger_receipt_verified_dry_run" ||
			ledgerVerification.LedgerVerificationAction != "verify_persisted_admission_ledger_receipt_dry_run" ||
			ledgerVerification.LedgerVerificationTarget != "admission_ledger" ||
			ledgerVerification.LedgerVerificationTargetKind != "dream_candidate_admission" ||
			ledgerVerification.LedgerVerificationTargetMode != "append_only_dry_run" ||
			ledgerVerification.LedgerVerificationReceiptShape != "candidate_contract_provenance" ||
			!ledgerVerification.LedgerVerificationAppendOnly ||
			!ledgerVerification.LedgerVerificationDryRunOnly ||
			!ledgerVerification.LedgerVerificationReceiptReadBack ||
			!ledgerVerification.LedgerVerificationReceiptVerified ||
			!ledgerVerification.LedgerVerificationReady ||
			ledgerVerification.SourceLedgerPersistenceSchema != admissionLiveRouteTurnCandidateAdmissionLedgerPersistenceSchema ||
			!ledgerVerification.SourceLedgerPersistencePassed ||
			ledgerVerification.SourceLedgerPersistenceID != ledgerPersistence.LedgerPersistenceID ||
			ledgerVerification.SourceLedgerPersistenceAction != "append_admission_ledger_receipt_dry_run" ||
			!ledgerVerification.SourceLedgerPersistenceReady ||
			!ledgerVerification.SourceLedgerPersistenceReceiptPersisted ||
			ledgerVerification.SourceLedgerImplementationIDForVerification != ledgerImpl.LedgerImplementationID ||
			ledgerVerification.SourceAdmissionLedgerIDForVerification != ledger.AdmissionLedgerID ||
			ledgerVerification.SourceRollbackImplementationIDForVerification != rollbackImpl.RollbackImplementationID ||
			ledgerVerification.SourceWriterReceiptIDForVerification != writerReceipt.WriterReceiptID ||
			ledgerVerification.LedgerVerificationID == "" ||
			ledgerVerification.LedgerPersistenceID != ledgerPersistence.LedgerPersistenceID ||
			ledgerVerification.LedgerImplementationID != ledgerImpl.LedgerImplementationID ||
			ledgerVerification.RollbackImplementationID != rollbackImpl.RollbackImplementationID ||
			ledgerVerification.WriterReceiptID != writerReceipt.WriterReceiptID ||
			ledgerVerification.AdmissionLedgerID != ledger.AdmissionLedgerID ||
			ledgerVerification.AdmissionWriterContractID != writerContract.WriterContractID ||
			ledgerVerification.AdmissionWriterInventoryID != writerInventory.WriterInventoryID ||
			ledgerVerification.AdmissionWriterPreflightID != writerPreflight.WriterPreflightID ||
			ledgerVerification.AdmissionLiveStageID != liveStage.LiveStageID ||
			ledgerVerification.AdmissionEnableGateID != gate.EnableGateID ||
			ledgerVerification.AdmissionSwitchID != sw.SwitchID ||
			ledgerVerification.AdmissionPromotionID != promotion.PromotionID ||
			ledgerVerification.AdmissionDecisionID != decision.DecisionID ||
			ledgerVerification.CandidateExecutionID != execution.ExecutionID ||
			ledgerVerification.GeneratorAdapterID != generatorAdapter.AdapterID ||
			ledgerVerification.CandidateDraftID != draft.DraftID ||
			ledgerVerification.HandoffID != admission.HandoffID ||
			ledgerVerification.AdmissionAdapterID != admissionAdapter.AdmissionAdapterID ||
			ledgerVerification.DreamCandidateRunID != candidate.RunID ||
			ledgerVerification.CandidateRunID != draft.CandidateRunID ||
			ledgerVerification.CandidateTextHash != execution.GeneratedTextHash ||
			ledgerVerification.TurnTextHash != execution.TurnTextHash ||
			!ledgerVerification.AdmissionPolicyPassed ||
			!ledgerVerification.LiveRouteChoicePassed ||
			!ledgerVerification.SourceDecisionPassed ||
			!ledgerVerification.SourcePromotionPassed ||
			!ledgerVerification.SourceSwitchPassed ||
			!ledgerVerification.SourceEnablePassed ||
			!ledgerVerification.SourceStagePassed ||
			!ledgerVerification.SourceWriterPreflightPassed ||
			!ledgerVerification.SourceWriterInventoryPassed ||
			!ledgerVerification.SourceWriterContractPassed ||
			!ledgerVerification.SourceLedgerPassed ||
			ledgerVerification.Reason != "ledger persistence receipt verified by read-back dry-run; live admission remains disabled" {
			return fmt.Errorf("bad nano-direct admission ledger verification receipt: ledger_verification=%+v ledger_persistence=%+v ledger_implementation=%+v rollback_implementation=%+v writer_receipt=%+v writer_implementation=%+v ledger=%+v writer_contract=%+v writer_inventory=%+v writer_preflight=%+v stage=%+v gate=%+v switch=%+v promotion=%+v decision=%+v execution=%+v", ledgerVerification, ledgerPersistence, ledgerImpl, rollbackImpl, writerReceipt, writerImpl, ledger, writerContract, writerInventory, writerPreflight, liveStage, gate, sw, promotion, decision, execution)
		}
	}
	if admissionLiveRouteTurnCandidateAdmissionReadinessDryRun() {
		if readiness.Schema != admissionLiveRouteTurnCandidateAdmissionReadinessSchema ||
			!readiness.Passed ||
			!readiness.LiveReady ||
			readiness.LiveAdmissionEnabled ||
			readiness.AdmissionAllowed ||
			readiness.ContractsReady ||
			readiness.WriteAllowed ||
			readiness.MutatesState ||
			readiness.BodyTarget != "none" ||
			readiness.AdmissionReadinessState != "verified_closed_dry_run" ||
			readiness.AdmissionReadinessAction != "declare_verified_live_admission_readiness_dry_run" ||
			readiness.AdmissionReadinessTarget != "live_admission" ||
			readiness.AdmissionReadinessTargetKind != "dream_candidate_admission" ||
			readiness.AdmissionReadinessTargetMode != "closed_verified_dry_run" ||
			!readiness.AdmissionReadinessDryRunOnly ||
			!readiness.AdmissionReadinessLedgerVerified ||
			!readiness.AdmissionReadinessWriterReady ||
			!readiness.AdmissionReadinessRollbackReady ||
			!readiness.AdmissionReadinessLedgerReady ||
			!readiness.AdmissionReadinessReady ||
			readiness.SourceLedgerVerificationSchema != admissionLiveRouteTurnCandidateAdmissionLedgerVerificationSchema ||
			!readiness.SourceLedgerVerificationPassed ||
			readiness.SourceLedgerVerificationID != ledgerVerification.LedgerVerificationID ||
			readiness.SourceLedgerVerificationAction != "verify_persisted_admission_ledger_receipt_dry_run" ||
			!readiness.SourceLedgerVerificationReady ||
			!readiness.SourceLedgerVerificationReceiptVerified ||
			readiness.SourceLedgerPersistenceIDForReadiness != ledgerPersistence.LedgerPersistenceID ||
			readiness.SourceLedgerImplementationIDForReadiness != ledgerImpl.LedgerImplementationID ||
			readiness.SourceAdmissionLedgerIDForReadiness != ledger.AdmissionLedgerID ||
			readiness.SourceRollbackImplementationIDForReadiness != rollbackImpl.RollbackImplementationID ||
			readiness.SourceWriterReceiptIDForReadiness != writerReceipt.WriterReceiptID ||
			readiness.AdmissionReadinessID == "" ||
			readiness.LedgerVerificationID != ledgerVerification.LedgerVerificationID ||
			readiness.LedgerPersistenceID != ledgerPersistence.LedgerPersistenceID ||
			readiness.LedgerImplementationID != ledgerImpl.LedgerImplementationID ||
			readiness.RollbackImplementationID != rollbackImpl.RollbackImplementationID ||
			readiness.WriterReceiptID != writerReceipt.WriterReceiptID ||
			readiness.AdmissionLedgerID != ledger.AdmissionLedgerID ||
			readiness.AdmissionWriterContractID != writerContract.WriterContractID ||
			readiness.AdmissionWriterInventoryID != writerInventory.WriterInventoryID ||
			readiness.AdmissionWriterPreflightID != writerPreflight.WriterPreflightID ||
			readiness.AdmissionLiveStageID != liveStage.LiveStageID ||
			readiness.AdmissionEnableGateID != gate.EnableGateID ||
			readiness.AdmissionSwitchID != sw.SwitchID ||
			readiness.AdmissionPromotionID != promotion.PromotionID ||
			readiness.AdmissionDecisionID != decision.DecisionID ||
			readiness.CandidateExecutionID != execution.ExecutionID ||
			readiness.GeneratorAdapterID != generatorAdapter.AdapterID ||
			readiness.CandidateDraftID != draft.DraftID ||
			readiness.HandoffID != admission.HandoffID ||
			readiness.AdmissionAdapterID != admissionAdapter.AdmissionAdapterID ||
			readiness.DreamCandidateRunID != candidate.RunID ||
			readiness.CandidateRunID != draft.CandidateRunID ||
			readiness.CandidateTextHash != execution.GeneratedTextHash ||
			readiness.TurnTextHash != execution.TurnTextHash ||
			readiness.Reason != "verified ledger and writer boundaries are ready; live admission remains disabled" {
			return fmt.Errorf("bad nano-direct admission readiness receipt: readiness=%+v ledger_verification=%+v ledger_persistence=%+v ledger_implementation=%+v rollback_implementation=%+v writer_receipt=%+v writer_implementation=%+v ledger=%+v writer_contract=%+v writer_inventory=%+v writer_preflight=%+v stage=%+v gate=%+v switch=%+v promotion=%+v decision=%+v execution=%+v", readiness, ledgerVerification, ledgerPersistence, ledgerImpl, rollbackImpl, writerReceipt, writerImpl, ledger, writerContract, writerInventory, writerPreflight, liveStage, gate, sw, promotion, decision, execution)
		}
	}
	if admissionLiveRouteTurnCandidateAdmissionPermitDryRun() {
		if permit.Schema != admissionLiveRouteTurnCandidateAdmissionPermitSchema ||
			!permit.Passed ||
			!permit.LiveReady ||
			permit.LiveAdmissionEnabled ||
			permit.AdmissionAllowed ||
			permit.ContractsReady ||
			permit.WriteAllowed ||
			permit.MutatesState ||
			permit.BodyTarget != "none" ||
			permit.AdmissionPermitState != "operator_permitted_closed_dry_run" ||
			permit.AdmissionPermitAction != "acknowledge_verified_live_admission_readiness_dry_run" ||
			permit.AdmissionPermitTarget != "live_admission" ||
			permit.AdmissionPermitTargetKind != "dream_candidate_admission" ||
			permit.AdmissionPermitTargetMode != "permit_closed_dry_run" ||
			!permit.AdmissionPermitDryRunOnly ||
			!permit.AdmissionPermitReadinessVerified ||
			!permit.AdmissionPermitLedgerVerified ||
			!permit.AdmissionPermitWriterReady ||
			!permit.AdmissionPermitRollbackReady ||
			!permit.AdmissionPermitLedgerReady ||
			!permit.AdmissionPermitReady ||
			!permit.ManualPermitRequested ||
			!permit.PermitKeyMatched ||
			permit.SourceAdmissionReadinessSchema != admissionLiveRouteTurnCandidateAdmissionReadinessSchema ||
			!permit.SourceAdmissionReadinessPassed ||
			permit.SourceAdmissionReadinessID != readiness.AdmissionReadinessID ||
			permit.SourceAdmissionReadinessAction != "declare_verified_live_admission_readiness_dry_run" ||
			!permit.SourceAdmissionReadinessReady ||
			!permit.SourceAdmissionReadinessLedgerVerified ||
			permit.SourceLedgerVerificationIDForPermit != ledgerVerification.LedgerVerificationID ||
			permit.SourceLedgerPersistenceIDForPermit != ledgerPersistence.LedgerPersistenceID ||
			permit.SourceLedgerImplementationIDForPermit != ledgerImpl.LedgerImplementationID ||
			permit.SourceAdmissionLedgerIDForPermit != ledger.AdmissionLedgerID ||
			permit.SourceRollbackImplementationIDForPermit != rollbackImpl.RollbackImplementationID ||
			permit.SourceWriterReceiptIDForPermit != writerReceipt.WriterReceiptID ||
			permit.AdmissionPermitID == "" ||
			permit.AdmissionReadinessID != readiness.AdmissionReadinessID ||
			permit.LedgerVerificationID != ledgerVerification.LedgerVerificationID ||
			permit.LedgerPersistenceID != ledgerPersistence.LedgerPersistenceID ||
			permit.LedgerImplementationID != ledgerImpl.LedgerImplementationID ||
			permit.RollbackImplementationID != rollbackImpl.RollbackImplementationID ||
			permit.WriterReceiptID != writerReceipt.WriterReceiptID ||
			permit.AdmissionLedgerID != ledger.AdmissionLedgerID ||
			permit.AdmissionWriterContractID != writerContract.WriterContractID ||
			permit.AdmissionWriterInventoryID != writerInventory.WriterInventoryID ||
			permit.AdmissionWriterPreflightID != writerPreflight.WriterPreflightID ||
			permit.AdmissionLiveStageID != liveStage.LiveStageID ||
			permit.AdmissionEnableGateID != gate.EnableGateID ||
			permit.AdmissionSwitchID != sw.SwitchID ||
			permit.AdmissionPromotionID != promotion.PromotionID ||
			permit.AdmissionDecisionID != decision.DecisionID ||
			permit.CandidateExecutionID != execution.ExecutionID ||
			permit.GeneratorAdapterID != generatorAdapter.AdapterID ||
			permit.CandidateDraftID != draft.DraftID ||
			permit.HandoffID != admission.HandoffID ||
			permit.AdmissionAdapterID != admissionAdapter.AdmissionAdapterID ||
			permit.DreamCandidateRunID != candidate.RunID ||
			permit.CandidateRunID != draft.CandidateRunID ||
			permit.CandidateTextHash != execution.GeneratedTextHash ||
			permit.TurnTextHash != execution.TurnTextHash ||
			permit.Reason != "operator permit accepted for verified readiness; live admission remains disabled" {
			return fmt.Errorf("bad nano-direct admission permit receipt: permit=%+v readiness=%+v ledger_verification=%+v ledger_persistence=%+v ledger_implementation=%+v rollback_implementation=%+v writer_receipt=%+v writer_implementation=%+v ledger=%+v writer_contract=%+v writer_inventory=%+v writer_preflight=%+v stage=%+v gate=%+v switch=%+v promotion=%+v decision=%+v execution=%+v", permit, readiness, ledgerVerification, ledgerPersistence, ledgerImpl, rollbackImpl, writerReceipt, writerImpl, ledger, writerContract, writerInventory, writerPreflight, liveStage, gate, sw, promotion, decision, execution)
		}
	}
	if admissionLiveRouteTurnCandidateAdmissionSealDryRun() {
		if seal.Schema != admissionLiveRouteTurnCandidateAdmissionSealSchema ||
			!seal.Passed ||
			!seal.LiveReady ||
			seal.LiveAdmissionEnabled ||
			seal.AdmissionAllowed ||
			seal.ContractsReady ||
			seal.WriteAllowed ||
			seal.MutatesState ||
			seal.BodyTarget != "none" ||
			seal.AdmissionSealState != "sealed_closed_dry_run" ||
			seal.AdmissionSealAction != "seal_operator_permit_provenance_dry_run" ||
			seal.AdmissionSealTarget != "live_admission" ||
			seal.AdmissionSealTargetKind != "dream_candidate_admission" ||
			seal.AdmissionSealTargetMode != "sealed_closed_dry_run" ||
			seal.AdmissionSealReceiptShape != "candidate_contract_provenance" ||
			!seal.AdmissionSealDryRunOnly ||
			!seal.AdmissionSealPermitVerified ||
			!seal.AdmissionSealReadinessVerified ||
			!seal.AdmissionSealLedgerVerified ||
			!seal.AdmissionSealWriterReady ||
			!seal.AdmissionSealRollbackReady ||
			!seal.AdmissionSealLedgerReady ||
			!seal.AdmissionSealReady ||
			seal.SourceAdmissionPermitSchema != admissionLiveRouteTurnCandidateAdmissionPermitSchema ||
			!seal.SourceAdmissionPermitPassed ||
			seal.SourceAdmissionPermitID != permit.AdmissionPermitID ||
			seal.SourceAdmissionPermitAction != "acknowledge_verified_live_admission_readiness_dry_run" ||
			!seal.SourceAdmissionPermitReady ||
			!seal.SourceAdmissionPermitKeyMatched ||
			seal.SourceAdmissionReadinessIDForSeal != readiness.AdmissionReadinessID ||
			seal.SourceLedgerVerificationIDForSeal != ledgerVerification.LedgerVerificationID ||
			seal.SourceLedgerPersistenceIDForSeal != ledgerPersistence.LedgerPersistenceID ||
			seal.SourceLedgerImplementationIDForSeal != ledgerImpl.LedgerImplementationID ||
			seal.SourceAdmissionLedgerIDForSeal != ledger.AdmissionLedgerID ||
			seal.SourceRollbackImplementationIDForSeal != rollbackImpl.RollbackImplementationID ||
			seal.SourceWriterReceiptIDForSeal != writerReceipt.WriterReceiptID ||
			seal.AdmissionSealID == "" ||
			seal.AdmissionPermitID != permit.AdmissionPermitID ||
			seal.AdmissionReadinessID != readiness.AdmissionReadinessID ||
			seal.LedgerVerificationID != ledgerVerification.LedgerVerificationID ||
			seal.LedgerPersistenceID != ledgerPersistence.LedgerPersistenceID ||
			seal.LedgerImplementationID != ledgerImpl.LedgerImplementationID ||
			seal.RollbackImplementationID != rollbackImpl.RollbackImplementationID ||
			seal.WriterReceiptID != writerReceipt.WriterReceiptID ||
			seal.AdmissionLedgerID != ledger.AdmissionLedgerID ||
			seal.AdmissionWriterContractID != writerContract.WriterContractID ||
			seal.AdmissionWriterInventoryID != writerInventory.WriterInventoryID ||
			seal.AdmissionWriterPreflightID != writerPreflight.WriterPreflightID ||
			seal.AdmissionLiveStageID != liveStage.LiveStageID ||
			seal.AdmissionEnableGateID != gate.EnableGateID ||
			seal.AdmissionSwitchID != sw.SwitchID ||
			seal.AdmissionPromotionID != promotion.PromotionID ||
			seal.AdmissionDecisionID != decision.DecisionID ||
			seal.CandidateExecutionID != execution.ExecutionID ||
			seal.GeneratorAdapterID != generatorAdapter.AdapterID ||
			seal.CandidateDraftID != draft.DraftID ||
			seal.HandoffID != admission.HandoffID ||
			seal.AdmissionAdapterID != admissionAdapter.AdmissionAdapterID ||
			seal.DreamCandidateRunID != candidate.RunID ||
			seal.CandidateRunID != draft.CandidateRunID ||
			seal.CandidateTextHash != execution.GeneratedTextHash ||
			seal.TurnTextHash != execution.TurnTextHash ||
			seal.Reason != "operator permit sealed as immutable dry-run receipt; live admission remains disabled" {
			return fmt.Errorf("bad nano-direct admission seal receipt: seal=%+v permit=%+v readiness=%+v ledger_verification=%+v ledger_persistence=%+v ledger_implementation=%+v rollback_implementation=%+v writer_receipt=%+v writer_implementation=%+v ledger=%+v writer_contract=%+v writer_inventory=%+v writer_preflight=%+v stage=%+v gate=%+v switch=%+v promotion=%+v decision=%+v execution=%+v", seal, permit, readiness, ledgerVerification, ledgerPersistence, ledgerImpl, rollbackImpl, writerReceipt, writerImpl, ledger, writerContract, writerInventory, writerPreflight, liveStage, gate, sw, promotion, decision, execution)
		}
	}
	if admissionLiveRouteTurnCandidateAdmissionFinalGateDryRun() {
		if finalGate.Schema != admissionLiveRouteTurnCandidateAdmissionFinalGateSchema ||
			!finalGate.Passed ||
			!finalGate.LiveReady ||
			finalGate.LiveAdmissionEnabled ||
			finalGate.AdmissionAllowed ||
			finalGate.ContractsReady ||
			finalGate.WriteAllowed ||
			finalGate.MutatesState ||
			finalGate.BodyTarget != "none" ||
			finalGate.AdmissionFinalGateState != "ready_closed_dry_run" ||
			finalGate.AdmissionFinalGateAction != "verify_sealed_admission_provenance_dry_run" ||
			finalGate.AdmissionFinalGateTarget != "live_admission" ||
			finalGate.AdmissionFinalGateTargetKind != "dream_candidate_admission" ||
			finalGate.AdmissionFinalGateTargetMode != "final_gate_closed_dry_run" ||
			finalGate.AdmissionFinalGateReceiptShape != "sealed_candidate_contract_provenance" ||
			!finalGate.AdmissionFinalGateDryRunOnly ||
			!finalGate.AdmissionFinalGateSealVerified ||
			!finalGate.AdmissionFinalGatePermitVerified ||
			!finalGate.AdmissionFinalGateReadinessVerified ||
			!finalGate.AdmissionFinalGateLedgerVerified ||
			!finalGate.AdmissionFinalGateWriterReady ||
			!finalGate.AdmissionFinalGateRollbackReady ||
			!finalGate.AdmissionFinalGateLedgerReady ||
			!finalGate.AdmissionFinalGateReady ||
			finalGate.SourceAdmissionSealSchema != admissionLiveRouteTurnCandidateAdmissionSealSchema ||
			!finalGate.SourceAdmissionSealPassed ||
			finalGate.SourceAdmissionSealID != seal.AdmissionSealID ||
			finalGate.SourceAdmissionSealAction != "seal_operator_permit_provenance_dry_run" ||
			!finalGate.SourceAdmissionSealReady ||
			finalGate.SourceAdmissionPermitIDForFinalGate != permit.AdmissionPermitID ||
			finalGate.SourceAdmissionReadinessIDForFinalGate != readiness.AdmissionReadinessID ||
			finalGate.SourceLedgerVerificationIDForFinalGate != ledgerVerification.LedgerVerificationID ||
			finalGate.SourceLedgerPersistenceIDForFinalGate != ledgerPersistence.LedgerPersistenceID ||
			finalGate.SourceLedgerImplementationIDForFinalGate != ledgerImpl.LedgerImplementationID ||
			finalGate.SourceAdmissionLedgerIDForFinalGate != ledger.AdmissionLedgerID ||
			finalGate.SourceRollbackImplementationIDForFinalGate != rollbackImpl.RollbackImplementationID ||
			finalGate.SourceWriterReceiptIDForFinalGate != writerReceipt.WriterReceiptID ||
			finalGate.AdmissionFinalGateID == "" ||
			finalGate.AdmissionSealID != seal.AdmissionSealID ||
			finalGate.AdmissionPermitID != permit.AdmissionPermitID ||
			finalGate.AdmissionReadinessID != readiness.AdmissionReadinessID ||
			finalGate.LedgerVerificationID != ledgerVerification.LedgerVerificationID ||
			finalGate.LedgerPersistenceID != ledgerPersistence.LedgerPersistenceID ||
			finalGate.LedgerImplementationID != ledgerImpl.LedgerImplementationID ||
			finalGate.RollbackImplementationID != rollbackImpl.RollbackImplementationID ||
			finalGate.WriterReceiptID != writerReceipt.WriterReceiptID ||
			finalGate.AdmissionLedgerID != ledger.AdmissionLedgerID ||
			finalGate.AdmissionWriterContractID != writerContract.WriterContractID ||
			finalGate.AdmissionWriterInventoryID != writerInventory.WriterInventoryID ||
			finalGate.AdmissionWriterPreflightID != writerPreflight.WriterPreflightID ||
			finalGate.AdmissionLiveStageID != liveStage.LiveStageID ||
			finalGate.AdmissionEnableGateID != gate.EnableGateID ||
			finalGate.AdmissionSwitchID != sw.SwitchID ||
			finalGate.AdmissionPromotionID != promotion.PromotionID ||
			finalGate.AdmissionDecisionID != decision.DecisionID ||
			finalGate.CandidateExecutionID != execution.ExecutionID ||
			finalGate.GeneratorAdapterID != generatorAdapter.AdapterID ||
			finalGate.CandidateDraftID != draft.DraftID ||
			finalGate.HandoffID != admission.HandoffID ||
			finalGate.AdmissionAdapterID != admissionAdapter.AdmissionAdapterID ||
			finalGate.DreamCandidateRunID != candidate.RunID ||
			finalGate.CandidateRunID != draft.CandidateRunID ||
			finalGate.CandidateTextHash != execution.GeneratedTextHash ||
			finalGate.TurnTextHash != execution.TurnTextHash ||
			finalGate.Reason != "sealed admission provenance cleared final gate; live admission remains disabled" {
			return fmt.Errorf("bad nano-direct admission final gate receipt: final_gate=%+v seal=%+v permit=%+v readiness=%+v ledger_verification=%+v ledger_persistence=%+v ledger_implementation=%+v rollback_implementation=%+v writer_receipt=%+v writer_implementation=%+v ledger=%+v writer_contract=%+v writer_inventory=%+v writer_preflight=%+v stage=%+v gate=%+v switch=%+v promotion=%+v decision=%+v execution=%+v", finalGate, seal, permit, readiness, ledgerVerification, ledgerPersistence, ledgerImpl, rollbackImpl, writerReceipt, writerImpl, ledger, writerContract, writerInventory, writerPreflight, liveStage, gate, sw, promotion, decision, execution)
		}
	}
	if admissionLiveRouteTurnCandidateAdmissionResonanceIntentDryRun() {
		if resonanceIntent.Schema != admissionLiveRouteTurnCandidateAdmissionResonanceIntentSchema ||
			!resonanceIntent.Passed ||
			!resonanceIntent.LiveReady ||
			resonanceIntent.LiveAdmissionEnabled ||
			resonanceIntent.AdmissionAllowed ||
			resonanceIntent.ContractsReady ||
			resonanceIntent.WriteAllowed ||
			resonanceIntent.MutatesState ||
			resonanceIntent.BodyTarget != "none" ||
			resonanceIntent.AdmissionResonanceIntentState != "resonance_intent_drafted_dry_run" ||
			resonanceIntent.AdmissionResonanceIntentAction != "draft_resonance_direction_intent_dry_run" ||
			resonanceIntent.AdmissionResonanceIntentTarget != "resonance" ||
			resonanceIntent.AdmissionResonanceIntentTargetKind != "first_live_receiver" ||
			resonanceIntent.AdmissionResonanceIntentTargetMode != "bounded_direction_dry_run" ||
			resonanceIntent.AdmissionResonanceIntentReceiptShape != "sealed_candidate_contract_provenance" ||
			!resonanceIntent.AdmissionResonanceIntentDryRunOnly ||
			!resonanceIntent.AdmissionResonanceIntentFinalGateVerified ||
			!resonanceIntent.AdmissionResonanceIntentSealVerified ||
			!resonanceIntent.AdmissionResonanceIntentPermitVerified ||
			!resonanceIntent.AdmissionResonanceIntentReadinessVerified ||
			!resonanceIntent.AdmissionResonanceIntentLedgerVerified ||
			!resonanceIntent.AdmissionResonanceIntentWriterReady ||
			!resonanceIntent.AdmissionResonanceIntentRollbackReady ||
			!resonanceIntent.AdmissionResonanceIntentLedgerReady ||
			resonanceIntent.AdmissionResonanceIntentReceiver != "resonance" ||
			resonanceIntent.AdmissionResonanceIntentReceiverKind != "internal_world" ||
			resonanceIntent.AdmissionResonanceIntentInfluenceKind != "bounded_direction" ||
			resonanceIntent.AdmissionResonanceIntentMaxInfluence != admissionLiveRouteTurnCandidateAdmissionResonanceIntentMaxGain ||
			resonanceIntent.AdmissionResonanceIntentTTLTurns != admissionLiveRouteTurnCandidateAdmissionResonanceIntentTTL ||
			resonanceIntent.AdmissionResonanceIntentCausalID == "" ||
			resonanceIntent.AdmissionResonanceIntentRawDreamTextAllowed ||
			resonanceIntent.AdmissionResonanceIntentJanusSurfaceAllowed ||
			resonanceIntent.AdmissionResonanceIntentCoocLearningAllowed ||
			resonanceIntent.AdmissionResonanceIntentDeltaHarvestAllowed ||
			!resonanceIntent.AdmissionResonanceIntentRollbackRequired ||
			!resonanceIntent.AdmissionResonanceIntentPreStateHashRequired ||
			!resonanceIntent.AdmissionResonanceIntentPostStateHashRequired ||
			!resonanceIntent.AdmissionResonanceIntentReady ||
			resonanceIntent.SourceAdmissionFinalGateSchema != admissionLiveRouteTurnCandidateAdmissionFinalGateSchema ||
			!resonanceIntent.SourceAdmissionFinalGatePassed ||
			resonanceIntent.SourceAdmissionFinalGateID != finalGate.AdmissionFinalGateID ||
			resonanceIntent.SourceAdmissionFinalGateAction != "verify_sealed_admission_provenance_dry_run" ||
			!resonanceIntent.SourceAdmissionFinalGateReady ||
			resonanceIntent.SourceAdmissionSealIDForResonanceIntent != seal.AdmissionSealID ||
			resonanceIntent.SourceAdmissionPermitIDForResonanceIntent != permit.AdmissionPermitID ||
			resonanceIntent.SourceAdmissionReadinessIDForResonanceIntent != readiness.AdmissionReadinessID ||
			resonanceIntent.SourceLedgerVerificationIDForResonanceIntent != ledgerVerification.LedgerVerificationID ||
			resonanceIntent.SourceLedgerPersistenceIDForResonanceIntent != ledgerPersistence.LedgerPersistenceID ||
			resonanceIntent.SourceLedgerImplementationIDForResonanceIntent != ledgerImpl.LedgerImplementationID ||
			resonanceIntent.SourceAdmissionLedgerIDForResonanceIntent != ledger.AdmissionLedgerID ||
			resonanceIntent.SourceRollbackImplementationIDForResonanceIntent != rollbackImpl.RollbackImplementationID ||
			resonanceIntent.SourceWriterReceiptIDForResonanceIntent != writerReceipt.WriterReceiptID ||
			resonanceIntent.AdmissionResonanceIntentID == "" ||
			resonanceIntent.AdmissionFinalGateID != finalGate.AdmissionFinalGateID ||
			resonanceIntent.AdmissionSealID != seal.AdmissionSealID ||
			resonanceIntent.AdmissionPermitID != permit.AdmissionPermitID ||
			resonanceIntent.AdmissionReadinessID != readiness.AdmissionReadinessID ||
			resonanceIntent.LedgerVerificationID != ledgerVerification.LedgerVerificationID ||
			resonanceIntent.LedgerPersistenceID != ledgerPersistence.LedgerPersistenceID ||
			resonanceIntent.LedgerImplementationID != ledgerImpl.LedgerImplementationID ||
			resonanceIntent.RollbackImplementationID != rollbackImpl.RollbackImplementationID ||
			resonanceIntent.WriterReceiptID != writerReceipt.WriterReceiptID ||
			resonanceIntent.AdmissionLedgerID != ledger.AdmissionLedgerID ||
			resonanceIntent.AdmissionWriterContractID != writerContract.WriterContractID ||
			resonanceIntent.AdmissionWriterInventoryID != writerInventory.WriterInventoryID ||
			resonanceIntent.AdmissionWriterPreflightID != writerPreflight.WriterPreflightID ||
			resonanceIntent.AdmissionLiveStageID != liveStage.LiveStageID ||
			resonanceIntent.AdmissionEnableGateID != gate.EnableGateID ||
			resonanceIntent.AdmissionSwitchID != sw.SwitchID ||
			resonanceIntent.AdmissionPromotionID != promotion.PromotionID ||
			resonanceIntent.AdmissionDecisionID != decision.DecisionID ||
			resonanceIntent.CandidateExecutionID != execution.ExecutionID ||
			resonanceIntent.GeneratorAdapterID != generatorAdapter.AdapterID ||
			resonanceIntent.CandidateDraftID != draft.DraftID ||
			resonanceIntent.HandoffID != admission.HandoffID ||
			resonanceIntent.AdmissionAdapterID != admissionAdapter.AdmissionAdapterID ||
			resonanceIntent.DreamCandidateRunID != candidate.RunID ||
			resonanceIntent.CandidateRunID != draft.CandidateRunID ||
			resonanceIntent.CandidateTextHash != execution.GeneratedTextHash ||
			resonanceIntent.TurnTextHash != execution.TurnTextHash ||
			resonanceIntent.Reason != "resonance intent drafted from final gate; live admission remains disabled" {
			return fmt.Errorf("bad nano-direct admission resonance intent receipt: resonance_intent=%+v final_gate=%+v seal=%+v permit=%+v readiness=%+v ledger_verification=%+v ledger_persistence=%+v ledger_implementation=%+v rollback_implementation=%+v writer_receipt=%+v writer_implementation=%+v ledger=%+v writer_contract=%+v writer_inventory=%+v writer_preflight=%+v stage=%+v gate=%+v switch=%+v promotion=%+v decision=%+v execution=%+v", resonanceIntent, finalGate, seal, permit, readiness, ledgerVerification, ledgerPersistence, ledgerImpl, rollbackImpl, writerReceipt, writerImpl, ledger, writerContract, writerInventory, writerPreflight, liveStage, gate, sw, promotion, decision, execution)
		}
	}
	if admissionLiveRouteTurnCandidateAdmissionResonanceReceiverDryRun() {
		if resonanceReceiver.Schema != admissionLiveRouteTurnCandidateAdmissionResonanceReceiverSchema ||
			!resonanceReceiver.Passed ||
			!resonanceReceiver.LiveReady ||
			resonanceReceiver.LiveAdmissionEnabled ||
			resonanceReceiver.AdmissionAllowed ||
			resonanceReceiver.ContractsReady ||
			resonanceReceiver.WriteAllowed ||
			resonanceReceiver.MutatesState ||
			resonanceReceiver.BodyTarget != "none" ||
			resonanceReceiver.AdmissionResonanceReceiverState != "receiver_previewed_dry_run" ||
			resonanceReceiver.AdmissionResonanceReceiverAction != "preview_resonance_receive_dry_run" ||
			resonanceReceiver.AdmissionResonanceReceiverTarget != "resonance" ||
			resonanceReceiver.AdmissionResonanceReceiverTargetKind != "first_live_receiver" ||
			resonanceReceiver.AdmissionResonanceReceiverTargetMode != "bounded_direction_preview_dry_run" ||
			resonanceReceiver.AdmissionResonanceReceiverReceiptShape != "resonance_receiver_state_proof" ||
			!resonanceReceiver.AdmissionResonanceReceiverDryRunOnly ||
			!resonanceReceiver.AdmissionResonanceReceiverIntentVerified ||
			!resonanceReceiver.AdmissionResonanceReceiverFinalGateVerified ||
			!resonanceReceiver.AdmissionResonanceReceiverSealVerified ||
			!resonanceReceiver.AdmissionResonanceReceiverPermitVerified ||
			!resonanceReceiver.AdmissionResonanceReceiverReadinessVerified ||
			!resonanceReceiver.AdmissionResonanceReceiverLedgerVerified ||
			!resonanceReceiver.AdmissionResonanceReceiverWriterReady ||
			!resonanceReceiver.AdmissionResonanceReceiverRollbackReady ||
			!resonanceReceiver.AdmissionResonanceReceiverLedgerReady ||
			resonanceReceiver.AdmissionResonanceReceiverReceiver != "resonance" ||
			resonanceReceiver.AdmissionResonanceReceiverReceiverKind != "internal_world" ||
			resonanceReceiver.AdmissionResonanceReceiverInfluenceKind != "bounded_direction" ||
			resonanceReceiver.AdmissionResonanceReceiverMaxInfluence != admissionLiveRouteTurnCandidateAdmissionResonanceIntentMaxGain ||
			resonanceReceiver.AdmissionResonanceReceiverTTLTurns != admissionLiveRouteTurnCandidateAdmissionResonanceIntentTTL ||
			resonanceReceiver.AdmissionResonanceReceiverCausalID == "" ||
			resonanceReceiver.AdmissionResonanceReceiverCausalID != admissionLiveRouteTurnCandidateAdmissionResonanceReceiverCausalID(resonanceReceiver) ||
			resonanceReceiver.AdmissionResonanceReceiverPreStateHash == "" ||
			resonanceReceiver.AdmissionResonanceReceiverPostStateHash == "" ||
			resonanceReceiver.AdmissionResonanceReceiverStateDeltaHash == "" ||
			resonanceReceiver.AdmissionResonanceReceiverPreStateHash == resonanceReceiver.AdmissionResonanceReceiverPostStateHash ||
			resonanceReceiver.AdmissionResonanceReceiverPreStateHash != admissionLiveRouteTurnCandidateAdmissionResonanceReceiverPreStateHash(resonanceReceiver) ||
			resonanceReceiver.AdmissionResonanceReceiverPostStateHash != admissionLiveRouteTurnCandidateAdmissionResonanceReceiverPostStateHash(resonanceReceiver) ||
			resonanceReceiver.AdmissionResonanceReceiverStateDeltaHash != admissionLiveRouteTurnCandidateAdmissionResonanceReceiverStateDeltaHash(resonanceReceiver) ||
			resonanceReceiver.AdmissionResonanceReceiverStateHashMode != "sealed_metadata_preview" ||
			resonanceReceiver.AdmissionResonanceReceiverRawDreamTextObserved ||
			resonanceReceiver.AdmissionResonanceReceiverRawDreamTextForwarded ||
			resonanceReceiver.AdmissionResonanceReceiverJanusSurfaceAllowed ||
			resonanceReceiver.AdmissionResonanceReceiverCoocLearningAllowed ||
			resonanceReceiver.AdmissionResonanceReceiverDeltaHarvestAllowed ||
			resonanceReceiver.AdmissionResonanceReceiverBodyMutationAllowed ||
			!resonanceReceiver.AdmissionResonanceReceiverRollbackRequired ||
			!resonanceReceiver.AdmissionResonanceReceiverReady ||
			resonanceReceiver.SourceAdmissionResonanceIntentSchema != admissionLiveRouteTurnCandidateAdmissionResonanceIntentSchema ||
			!resonanceReceiver.SourceAdmissionResonanceIntentPassed ||
			resonanceReceiver.SourceAdmissionResonanceIntentID != resonanceIntent.AdmissionResonanceIntentID ||
			resonanceReceiver.SourceAdmissionResonanceIntentAction != "draft_resonance_direction_intent_dry_run" ||
			!resonanceReceiver.SourceAdmissionResonanceIntentReady ||
			resonanceReceiver.SourceAdmissionResonanceIntentCausalID != resonanceIntent.AdmissionResonanceIntentCausalID ||
			resonanceReceiver.SourceAdmissionFinalGateIDForResonanceReceiver != finalGate.AdmissionFinalGateID ||
			resonanceReceiver.SourceAdmissionSealIDForResonanceReceiver != seal.AdmissionSealID ||
			resonanceReceiver.SourceAdmissionPermitIDForResonanceReceiver != permit.AdmissionPermitID ||
			resonanceReceiver.SourceAdmissionReadinessIDForResonanceReceiver != readiness.AdmissionReadinessID ||
			resonanceReceiver.SourceLedgerVerificationIDForResonanceReceiver != ledgerVerification.LedgerVerificationID ||
			resonanceReceiver.SourceLedgerPersistenceIDForResonanceReceiver != ledgerPersistence.LedgerPersistenceID ||
			resonanceReceiver.SourceLedgerImplementationIDForResonanceReceiver != ledgerImpl.LedgerImplementationID ||
			resonanceReceiver.SourceAdmissionLedgerIDForResonanceReceiver != ledger.AdmissionLedgerID ||
			resonanceReceiver.SourceRollbackImplementationIDForResonanceReceiver != rollbackImpl.RollbackImplementationID ||
			resonanceReceiver.SourceWriterReceiptIDForResonanceReceiver != writerReceipt.WriterReceiptID ||
			resonanceReceiver.AdmissionResonanceReceiverID == "" ||
			resonanceReceiver.AdmissionResonanceReceiverID != admissionLiveRouteTurnCandidateAdmissionResonanceReceiverID(resonanceReceiver) ||
			resonanceReceiver.AdmissionResonanceIntentID != resonanceIntent.AdmissionResonanceIntentID ||
			resonanceReceiver.AdmissionFinalGateID != finalGate.AdmissionFinalGateID ||
			resonanceReceiver.AdmissionSealID != seal.AdmissionSealID ||
			resonanceReceiver.AdmissionPermitID != permit.AdmissionPermitID ||
			resonanceReceiver.AdmissionReadinessID != readiness.AdmissionReadinessID ||
			resonanceReceiver.LedgerVerificationID != ledgerVerification.LedgerVerificationID ||
			resonanceReceiver.LedgerPersistenceID != ledgerPersistence.LedgerPersistenceID ||
			resonanceReceiver.LedgerImplementationID != ledgerImpl.LedgerImplementationID ||
			resonanceReceiver.RollbackImplementationID != rollbackImpl.RollbackImplementationID ||
			resonanceReceiver.WriterReceiptID != writerReceipt.WriterReceiptID ||
			resonanceReceiver.AdmissionLedgerID != ledger.AdmissionLedgerID ||
			resonanceReceiver.AdmissionWriterContractID != writerContract.WriterContractID ||
			resonanceReceiver.AdmissionWriterInventoryID != writerInventory.WriterInventoryID ||
			resonanceReceiver.AdmissionWriterPreflightID != writerPreflight.WriterPreflightID ||
			resonanceReceiver.AdmissionLiveStageID != liveStage.LiveStageID ||
			resonanceReceiver.AdmissionEnableGateID != gate.EnableGateID ||
			resonanceReceiver.AdmissionSwitchID != sw.SwitchID ||
			resonanceReceiver.AdmissionPromotionID != promotion.PromotionID ||
			resonanceReceiver.AdmissionDecisionID != decision.DecisionID ||
			resonanceReceiver.CandidateExecutionID != execution.ExecutionID ||
			resonanceReceiver.GeneratorAdapterID != generatorAdapter.AdapterID ||
			resonanceReceiver.CandidateDraftID != draft.DraftID ||
			resonanceReceiver.HandoffID != admission.HandoffID ||
			resonanceReceiver.AdmissionAdapterID != admissionAdapter.AdmissionAdapterID ||
			resonanceReceiver.DreamCandidateRunID != candidate.RunID ||
			resonanceReceiver.CandidateRunID != draft.CandidateRunID ||
			resonanceReceiver.CandidateTextHash != execution.GeneratedTextHash ||
			resonanceReceiver.TurnTextHash != execution.TurnTextHash ||
			resonanceReceiver.Reason != "resonance receiver previewed sealed intent without body mutation" {
			return fmt.Errorf("bad nano-direct admission resonance receiver receipt: resonance_receiver=%+v resonance_intent=%+v final_gate=%+v seal=%+v permit=%+v readiness=%+v ledger_verification=%+v ledger_persistence=%+v ledger_implementation=%+v rollback_implementation=%+v writer_receipt=%+v writer_implementation=%+v ledger=%+v writer_contract=%+v writer_inventory=%+v writer_preflight=%+v stage=%+v gate=%+v switch=%+v promotion=%+v decision=%+v execution=%+v", resonanceReceiver, resonanceIntent, finalGate, seal, permit, readiness, ledgerVerification, ledgerPersistence, ledgerImpl, rollbackImpl, writerReceipt, writerImpl, ledger, writerContract, writerInventory, writerPreflight, liveStage, gate, sw, promotion, decision, execution)
		}
	}
	if admissionLiveRouteTurnCandidateAdmissionResonanceObservationDryRun() {
		if resonanceObservation.Schema != admissionLiveRouteTurnCandidateAdmissionResonanceObservationSchema ||
			!resonanceObservation.Passed ||
			!resonanceObservation.LiveReady ||
			resonanceObservation.LiveAdmissionEnabled ||
			resonanceObservation.AdmissionAllowed ||
			resonanceObservation.ContractsReady ||
			resonanceObservation.WriteAllowed ||
			resonanceObservation.MutatesState ||
			resonanceObservation.BodyTarget != "none" ||
			resonanceObservation.AdmissionResonanceObservationState != "observation_recorded_dry_run" ||
			resonanceObservation.AdmissionResonanceObservationAction != "record_resonance_receiver_observation_dry_run" ||
			resonanceObservation.AdmissionResonanceObservationTarget != "resonance" ||
			resonanceObservation.AdmissionResonanceObservationTargetKind != "internal_world_observation" ||
			resonanceObservation.AdmissionResonanceObservationTargetMode != "append_only_read_back_dry_run" ||
			resonanceObservation.AdmissionResonanceObservationReceiptShape != "resonance_receiver_state_proof_ledger" ||
			!resonanceObservation.AdmissionResonanceObservationDryRunOnly ||
			!resonanceObservation.AdmissionResonanceObservationReceiverVerified ||
			!resonanceObservation.AdmissionResonanceObservationIntentVerified ||
			!resonanceObservation.AdmissionResonanceObservationFinalGateVerified ||
			!resonanceObservation.AdmissionResonanceObservationSealVerified ||
			!resonanceObservation.AdmissionResonanceObservationPermitVerified ||
			!resonanceObservation.AdmissionResonanceObservationReadinessVerified ||
			!resonanceObservation.AdmissionResonanceObservationLedgerVerified ||
			!resonanceObservation.AdmissionResonanceObservationWriterReady ||
			!resonanceObservation.AdmissionResonanceObservationRollbackReady ||
			!resonanceObservation.AdmissionResonanceObservationLedgerReady ||
			resonanceObservation.AdmissionResonanceObservationObserver != "resonance" ||
			resonanceObservation.AdmissionResonanceObservationObserverKind != "internal_world" ||
			resonanceObservation.AdmissionResonanceObservationKind != "receiver_state_proof" ||
			resonanceObservation.AdmissionResonanceObservationMode != "sealed_metadata_observation" ||
			resonanceObservation.AdmissionResonanceObservationCausalID == "" ||
			resonanceObservation.AdmissionResonanceObservationCausalID != admissionLiveRouteTurnCandidateAdmissionResonanceObservationCausalID(resonanceObservation) ||
			resonanceObservation.AdmissionResonanceObservationAppendHash == "" ||
			resonanceObservation.AdmissionResonanceObservationAppendHash != admissionLiveRouteTurnCandidateAdmissionResonanceObservationAppendHash(resonanceObservation) ||
			resonanceObservation.AdmissionResonanceObservationReadBackHash == "" ||
			resonanceObservation.AdmissionResonanceObservationReadBackHash != admissionLiveRouteTurnCandidateAdmissionResonanceObservationReadBackHash(resonanceObservation) ||
			resonanceObservation.AdmissionResonanceObservationAppendHash == resonanceObservation.AdmissionResonanceObservationReadBackHash ||
			!resonanceObservation.AdmissionResonanceObservationAppendOnly ||
			!resonanceObservation.AdmissionResonanceObservationReadBack ||
			!resonanceObservation.AdmissionResonanceObservationReceiptVerified ||
			resonanceObservation.AdmissionResonanceObservationRawDreamTextObserved ||
			resonanceObservation.AdmissionResonanceObservationRawDreamTextForwarded ||
			resonanceObservation.AdmissionResonanceObservationJanusSurfaceAllowed ||
			resonanceObservation.AdmissionResonanceObservationCoocLearningAllowed ||
			resonanceObservation.AdmissionResonanceObservationDeltaHarvestAllowed ||
			resonanceObservation.AdmissionResonanceObservationBodyMutationAllowed ||
			!resonanceObservation.AdmissionResonanceObservationRollbackRequired ||
			!resonanceObservation.AdmissionResonanceObservationReady ||
			resonanceObservation.SourceAdmissionResonanceReceiverSchema != admissionLiveRouteTurnCandidateAdmissionResonanceReceiverSchema ||
			!resonanceObservation.SourceAdmissionResonanceReceiverPassed ||
			resonanceObservation.SourceAdmissionResonanceReceiverID != resonanceReceiver.AdmissionResonanceReceiverID ||
			resonanceObservation.SourceAdmissionResonanceReceiverAction != "preview_resonance_receive_dry_run" ||
			!resonanceObservation.SourceAdmissionResonanceReceiverReady ||
			resonanceObservation.SourceAdmissionResonanceReceiverCausalID != resonanceReceiver.AdmissionResonanceReceiverCausalID ||
			resonanceObservation.SourceAdmissionResonanceReceiverPreStateHash != resonanceReceiver.AdmissionResonanceReceiverPreStateHash ||
			resonanceObservation.SourceAdmissionResonanceReceiverPostStateHash != resonanceReceiver.AdmissionResonanceReceiverPostStateHash ||
			resonanceObservation.SourceAdmissionResonanceReceiverStateDeltaHash != resonanceReceiver.AdmissionResonanceReceiverStateDeltaHash ||
			resonanceObservation.SourceAdmissionResonanceIntentIDForObservation != resonanceIntent.AdmissionResonanceIntentID ||
			resonanceObservation.SourceAdmissionFinalGateIDForResonanceObservation != finalGate.AdmissionFinalGateID ||
			resonanceObservation.SourceAdmissionSealIDForResonanceObservation != seal.AdmissionSealID ||
			resonanceObservation.SourceAdmissionPermitIDForResonanceObservation != permit.AdmissionPermitID ||
			resonanceObservation.SourceAdmissionReadinessIDForResonanceObservation != readiness.AdmissionReadinessID ||
			resonanceObservation.SourceLedgerVerificationIDForResonanceObservation != ledgerVerification.LedgerVerificationID ||
			resonanceObservation.SourceLedgerPersistenceIDForResonanceObservation != ledgerPersistence.LedgerPersistenceID ||
			resonanceObservation.SourceLedgerImplementationIDForResonanceObservation != ledgerImpl.LedgerImplementationID ||
			resonanceObservation.SourceAdmissionLedgerIDForResonanceObservation != ledger.AdmissionLedgerID ||
			resonanceObservation.SourceRollbackImplementationIDForResonanceObservation != rollbackImpl.RollbackImplementationID ||
			resonanceObservation.SourceWriterReceiptIDForResonanceObservation != writerReceipt.WriterReceiptID ||
			resonanceObservation.AdmissionResonanceObservationID == "" ||
			resonanceObservation.AdmissionResonanceObservationID != admissionLiveRouteTurnCandidateAdmissionResonanceObservationID(resonanceObservation) ||
			resonanceObservation.AdmissionResonanceReceiverID != resonanceReceiver.AdmissionResonanceReceiverID ||
			resonanceObservation.AdmissionResonanceIntentID != resonanceIntent.AdmissionResonanceIntentID ||
			resonanceObservation.AdmissionFinalGateID != finalGate.AdmissionFinalGateID ||
			resonanceObservation.AdmissionSealID != seal.AdmissionSealID ||
			resonanceObservation.AdmissionPermitID != permit.AdmissionPermitID ||
			resonanceObservation.AdmissionReadinessID != readiness.AdmissionReadinessID ||
			resonanceObservation.LedgerVerificationID != ledgerVerification.LedgerVerificationID ||
			resonanceObservation.LedgerPersistenceID != ledgerPersistence.LedgerPersistenceID ||
			resonanceObservation.LedgerImplementationID != ledgerImpl.LedgerImplementationID ||
			resonanceObservation.RollbackImplementationID != rollbackImpl.RollbackImplementationID ||
			resonanceObservation.WriterReceiptID != writerReceipt.WriterReceiptID ||
			resonanceObservation.AdmissionLedgerID != ledger.AdmissionLedgerID ||
			resonanceObservation.AdmissionWriterContractID != writerContract.WriterContractID ||
			resonanceObservation.AdmissionWriterInventoryID != writerInventory.WriterInventoryID ||
			resonanceObservation.AdmissionWriterPreflightID != writerPreflight.WriterPreflightID ||
			resonanceObservation.AdmissionLiveStageID != liveStage.LiveStageID ||
			resonanceObservation.AdmissionEnableGateID != gate.EnableGateID ||
			resonanceObservation.AdmissionSwitchID != sw.SwitchID ||
			resonanceObservation.AdmissionPromotionID != promotion.PromotionID ||
			resonanceObservation.AdmissionDecisionID != decision.DecisionID ||
			resonanceObservation.CandidateExecutionID != execution.ExecutionID ||
			resonanceObservation.GeneratorAdapterID != generatorAdapter.AdapterID ||
			resonanceObservation.CandidateDraftID != draft.DraftID ||
			resonanceObservation.HandoffID != admission.HandoffID ||
			resonanceObservation.AdmissionAdapterID != admissionAdapter.AdmissionAdapterID ||
			resonanceObservation.DreamCandidateRunID != candidate.RunID ||
			resonanceObservation.CandidateRunID != draft.CandidateRunID ||
			resonanceObservation.CandidateTextHash != execution.GeneratedTextHash ||
			resonanceObservation.TurnTextHash != execution.TurnTextHash ||
			resonanceObservation.Reason != "resonance observation recorded and read back without body mutation" {
			return fmt.Errorf("bad nano-direct admission resonance observation receipt: resonance_observation=%+v resonance_receiver=%+v resonance_intent=%+v final_gate=%+v seal=%+v permit=%+v readiness=%+v ledger_verification=%+v ledger_persistence=%+v ledger_implementation=%+v rollback_implementation=%+v writer_receipt=%+v writer_implementation=%+v ledger=%+v writer_contract=%+v writer_inventory=%+v writer_preflight=%+v stage=%+v gate=%+v switch=%+v promotion=%+v decision=%+v execution=%+v", resonanceObservation, resonanceReceiver, resonanceIntent, finalGate, seal, permit, readiness, ledgerVerification, ledgerPersistence, ledgerImpl, rollbackImpl, writerReceipt, writerImpl, ledger, writerContract, writerInventory, writerPreflight, liveStage, gate, sw, promotion, decision, execution)
		}
	}
	if admissionLiveRouteTurnCandidateAdmissionResonanceGraftBoundaryDryRun() {
		if resonanceGraftBoundary.Schema != admissionLiveRouteTurnCandidateAdmissionResonanceGraftBoundarySchema ||
			!resonanceGraftBoundary.Passed ||
			!resonanceGraftBoundary.LiveReady ||
			resonanceGraftBoundary.LiveAdmissionEnabled ||
			resonanceGraftBoundary.AdmissionAllowed ||
			resonanceGraftBoundary.ContractsReady ||
			resonanceGraftBoundary.WriteAllowed ||
			resonanceGraftBoundary.MutatesState ||
			resonanceGraftBoundary.BodyTarget != "none" ||
			resonanceGraftBoundary.AdmissionResonanceGraftBoundaryState != "shadow_graft_boundary_declared_dry_run" ||
			resonanceGraftBoundary.AdmissionResonanceGraftBoundaryAction != "declare_resonance_shadow_graft_boundary_dry_run" ||
			resonanceGraftBoundary.AdmissionResonanceGraftBoundaryTarget != "resonance" ||
			resonanceGraftBoundary.AdmissionResonanceGraftBoundaryTargetKind != "internal_world_shadow_graft" ||
			resonanceGraftBoundary.AdmissionResonanceGraftBoundaryTargetMode != "receipt_only_closed_dry_run" ||
			resonanceGraftBoundary.AdmissionResonanceGraftBoundaryReceiptShape != "resonance_observation_shadow_graft_boundary" ||
			!resonanceGraftBoundary.AdmissionResonanceGraftBoundaryDryRunOnly ||
			!resonanceGraftBoundary.AdmissionResonanceGraftBoundaryObservationVerified ||
			!resonanceGraftBoundary.AdmissionResonanceGraftBoundaryReceiverVerified ||
			!resonanceGraftBoundary.AdmissionResonanceGraftBoundaryIntentVerified ||
			!resonanceGraftBoundary.AdmissionResonanceGraftBoundaryFinalGateVerified ||
			!resonanceGraftBoundary.AdmissionResonanceGraftBoundarySealVerified ||
			!resonanceGraftBoundary.AdmissionResonanceGraftBoundaryPermitVerified ||
			!resonanceGraftBoundary.AdmissionResonanceGraftBoundaryReadinessVerified ||
			!resonanceGraftBoundary.AdmissionResonanceGraftBoundaryLedgerVerified ||
			!resonanceGraftBoundary.AdmissionResonanceGraftBoundaryWriterReady ||
			!resonanceGraftBoundary.AdmissionResonanceGraftBoundaryRollbackReady ||
			!resonanceGraftBoundary.AdmissionResonanceGraftBoundaryLedgerReady ||
			resonanceGraftBoundary.AdmissionResonanceGraftBoundaryKind != "shadow_graft_boundary" ||
			resonanceGraftBoundary.AdmissionResonanceGraftBoundaryMode != "no_mutation_receipt" ||
			resonanceGraftBoundary.AdmissionResonanceGraftBoundaryStage != "pre_live_graft" ||
			resonanceGraftBoundary.AdmissionResonanceGraftBoundaryCausalID == "" ||
			resonanceGraftBoundary.AdmissionResonanceGraftBoundaryCausalID != admissionLiveRouteTurnCandidateAdmissionResonanceGraftBoundaryCausalID(resonanceGraftBoundary) ||
			resonanceGraftBoundary.AdmissionResonanceGraftBoundaryHash == "" ||
			resonanceGraftBoundary.AdmissionResonanceGraftBoundaryHash != admissionLiveRouteTurnCandidateAdmissionResonanceGraftBoundaryHash(resonanceGraftBoundary) ||
			resonanceGraftBoundary.AdmissionResonanceGraftBoundaryReadBackHash == "" ||
			resonanceGraftBoundary.AdmissionResonanceGraftBoundaryReadBackHash != admissionLiveRouteTurnCandidateAdmissionResonanceGraftBoundaryReadBackHash(resonanceGraftBoundary) ||
			resonanceGraftBoundary.AdmissionResonanceGraftBoundaryHash == resonanceGraftBoundary.AdmissionResonanceGraftBoundaryReadBackHash ||
			!resonanceGraftBoundary.AdmissionResonanceGraftBoundaryShadowOnly ||
			resonanceGraftBoundary.AdmissionResonanceGraftBoundaryGraftAllowed ||
			resonanceGraftBoundary.AdmissionResonanceGraftBoundaryRawDreamTextAllowed ||
			resonanceGraftBoundary.AdmissionResonanceGraftBoundaryJanusSurfaceAllowed ||
			resonanceGraftBoundary.AdmissionResonanceGraftBoundaryCoocLearningAllowed ||
			resonanceGraftBoundary.AdmissionResonanceGraftBoundaryDeltaHarvestAllowed ||
			resonanceGraftBoundary.AdmissionResonanceGraftBoundaryBodyMutationAllowed ||
			!resonanceGraftBoundary.AdmissionResonanceGraftBoundaryRollbackRequired ||
			!resonanceGraftBoundary.AdmissionResonanceGraftBoundaryReady ||
			resonanceGraftBoundary.SourceAdmissionResonanceObservationSchema != admissionLiveRouteTurnCandidateAdmissionResonanceObservationSchema ||
			!resonanceGraftBoundary.SourceAdmissionResonanceObservationPassed ||
			resonanceGraftBoundary.SourceAdmissionResonanceObservationID != resonanceObservation.AdmissionResonanceObservationID ||
			resonanceGraftBoundary.SourceAdmissionResonanceObservationAction != "record_resonance_receiver_observation_dry_run" ||
			!resonanceGraftBoundary.SourceAdmissionResonanceObservationReady ||
			resonanceGraftBoundary.SourceAdmissionResonanceObservationCausalID != resonanceObservation.AdmissionResonanceObservationCausalID ||
			resonanceGraftBoundary.SourceAdmissionResonanceObservationAppendHash != resonanceObservation.AdmissionResonanceObservationAppendHash ||
			resonanceGraftBoundary.SourceAdmissionResonanceObservationReadBackHash != resonanceObservation.AdmissionResonanceObservationReadBackHash ||
			resonanceGraftBoundary.SourceAdmissionResonanceReceiverIDForGraftBoundary != resonanceReceiver.AdmissionResonanceReceiverID ||
			resonanceGraftBoundary.SourceAdmissionResonanceIntentIDForGraftBoundary != resonanceIntent.AdmissionResonanceIntentID ||
			resonanceGraftBoundary.SourceAdmissionFinalGateIDForGraftBoundary != finalGate.AdmissionFinalGateID ||
			resonanceGraftBoundary.SourceAdmissionSealIDForGraftBoundary != seal.AdmissionSealID ||
			resonanceGraftBoundary.SourceAdmissionPermitIDForGraftBoundary != permit.AdmissionPermitID ||
			resonanceGraftBoundary.SourceAdmissionReadinessIDForGraftBoundary != readiness.AdmissionReadinessID ||
			resonanceGraftBoundary.SourceLedgerVerificationIDForGraftBoundary != ledgerVerification.LedgerVerificationID ||
			resonanceGraftBoundary.SourceLedgerPersistenceIDForGraftBoundary != ledgerPersistence.LedgerPersistenceID ||
			resonanceGraftBoundary.SourceLedgerImplementationIDForGraftBoundary != ledgerImpl.LedgerImplementationID ||
			resonanceGraftBoundary.SourceAdmissionLedgerIDForGraftBoundary != ledger.AdmissionLedgerID ||
			resonanceGraftBoundary.SourceRollbackImplementationIDForGraftBoundary != rollbackImpl.RollbackImplementationID ||
			resonanceGraftBoundary.SourceWriterReceiptIDForGraftBoundary != writerReceipt.WriterReceiptID ||
			resonanceGraftBoundary.AdmissionResonanceGraftBoundaryID == "" ||
			resonanceGraftBoundary.AdmissionResonanceGraftBoundaryID != admissionLiveRouteTurnCandidateAdmissionResonanceGraftBoundaryID(resonanceGraftBoundary) ||
			resonanceGraftBoundary.AdmissionResonanceObservationID != resonanceObservation.AdmissionResonanceObservationID ||
			resonanceGraftBoundary.AdmissionResonanceReceiverID != resonanceReceiver.AdmissionResonanceReceiverID ||
			resonanceGraftBoundary.AdmissionResonanceIntentID != resonanceIntent.AdmissionResonanceIntentID ||
			resonanceGraftBoundary.AdmissionFinalGateID != finalGate.AdmissionFinalGateID ||
			resonanceGraftBoundary.AdmissionSealID != seal.AdmissionSealID ||
			resonanceGraftBoundary.AdmissionPermitID != permit.AdmissionPermitID ||
			resonanceGraftBoundary.AdmissionReadinessID != readiness.AdmissionReadinessID ||
			resonanceGraftBoundary.LedgerVerificationID != ledgerVerification.LedgerVerificationID ||
			resonanceGraftBoundary.LedgerPersistenceID != ledgerPersistence.LedgerPersistenceID ||
			resonanceGraftBoundary.LedgerImplementationID != ledgerImpl.LedgerImplementationID ||
			resonanceGraftBoundary.RollbackImplementationID != rollbackImpl.RollbackImplementationID ||
			resonanceGraftBoundary.WriterReceiptID != writerReceipt.WriterReceiptID ||
			resonanceGraftBoundary.AdmissionLedgerID != ledger.AdmissionLedgerID ||
			resonanceGraftBoundary.AdmissionWriterContractID != writerContract.WriterContractID ||
			resonanceGraftBoundary.AdmissionWriterInventoryID != writerInventory.WriterInventoryID ||
			resonanceGraftBoundary.AdmissionWriterPreflightID != writerPreflight.WriterPreflightID ||
			resonanceGraftBoundary.AdmissionLiveStageID != liveStage.LiveStageID ||
			resonanceGraftBoundary.AdmissionEnableGateID != gate.EnableGateID ||
			resonanceGraftBoundary.AdmissionSwitchID != sw.SwitchID ||
			resonanceGraftBoundary.AdmissionPromotionID != promotion.PromotionID ||
			resonanceGraftBoundary.AdmissionDecisionID != decision.DecisionID ||
			resonanceGraftBoundary.CandidateExecutionID != execution.ExecutionID ||
			resonanceGraftBoundary.GeneratorAdapterID != generatorAdapter.AdapterID ||
			resonanceGraftBoundary.CandidateDraftID != draft.DraftID ||
			resonanceGraftBoundary.HandoffID != admission.HandoffID ||
			resonanceGraftBoundary.AdmissionAdapterID != admissionAdapter.AdmissionAdapterID ||
			resonanceGraftBoundary.DreamCandidateRunID != candidate.RunID ||
			resonanceGraftBoundary.CandidateRunID != draft.CandidateRunID ||
			resonanceGraftBoundary.CandidateTextHash != execution.GeneratedTextHash ||
			resonanceGraftBoundary.TurnTextHash != execution.TurnTextHash ||
			resonanceGraftBoundary.Reason != "resonance shadow graft boundary declared without body mutation" {
			return fmt.Errorf("bad nano-direct admission resonance graft boundary receipt: resonance_graft_boundary=%+v resonance_observation=%+v resonance_receiver=%+v resonance_intent=%+v final_gate=%+v seal=%+v permit=%+v readiness=%+v ledger_verification=%+v ledger_persistence=%+v ledger_implementation=%+v rollback_implementation=%+v writer_receipt=%+v writer_implementation=%+v ledger=%+v writer_contract=%+v writer_inventory=%+v writer_preflight=%+v stage=%+v gate=%+v switch=%+v promotion=%+v decision=%+v execution=%+v", resonanceGraftBoundary, resonanceObservation, resonanceReceiver, resonanceIntent, finalGate, seal, permit, readiness, ledgerVerification, ledgerPersistence, ledgerImpl, rollbackImpl, writerReceipt, writerImpl, ledger, writerContract, writerInventory, writerPreflight, liveStage, gate, sw, promotion, decision, execution)
		}
	}
	if admissionLiveRouteTurnCandidateAdmissionResonanceGraftPreflightDryRun() {
		if resonanceGraftPreflight.Schema != admissionLiveRouteTurnCandidateAdmissionResonanceGraftPreflightSchema ||
			!resonanceGraftPreflight.Passed ||
			!resonanceGraftPreflight.LiveReady ||
			resonanceGraftPreflight.LiveAdmissionEnabled ||
			resonanceGraftPreflight.AdmissionAllowed ||
			resonanceGraftPreflight.ContractsReady ||
			resonanceGraftPreflight.WriteAllowed ||
			resonanceGraftPreflight.MutatesState ||
			resonanceGraftPreflight.BodyTarget != "none" ||
			resonanceGraftPreflight.AdmissionResonanceGraftPreflightState != "shadow_graft_preflight_ready_dry_run" ||
			resonanceGraftPreflight.AdmissionResonanceGraftPreflightAction != "prepare_resonance_shadow_graft_preflight_dry_run" ||
			resonanceGraftPreflight.AdmissionResonanceGraftPreflightTarget != "resonance" ||
			resonanceGraftPreflight.AdmissionResonanceGraftPreflightTargetKind != "internal_world_shadow_graft_preflight" ||
			resonanceGraftPreflight.AdmissionResonanceGraftPreflightTargetMode != "receipt_only_closed_preflight_dry_run" ||
			resonanceGraftPreflight.AdmissionResonanceGraftPreflightReceiptShape != "resonance_shadow_graft_preflight_contract" ||
			!resonanceGraftPreflight.AdmissionResonanceGraftPreflightDryRunOnly ||
			!resonanceGraftPreflight.AdmissionResonanceGraftPreflightBoundaryVerified ||
			!resonanceGraftPreflight.AdmissionResonanceGraftPreflightObservationVerified ||
			!resonanceGraftPreflight.AdmissionResonanceGraftPreflightReceiverVerified ||
			!resonanceGraftPreflight.AdmissionResonanceGraftPreflightIntentVerified ||
			!resonanceGraftPreflight.AdmissionResonanceGraftPreflightFinalGateVerified ||
			!resonanceGraftPreflight.AdmissionResonanceGraftPreflightSealVerified ||
			!resonanceGraftPreflight.AdmissionResonanceGraftPreflightPermitVerified ||
			!resonanceGraftPreflight.AdmissionResonanceGraftPreflightReadinessVerified ||
			!resonanceGraftPreflight.AdmissionResonanceGraftPreflightLedgerVerified ||
			!resonanceGraftPreflight.AdmissionResonanceGraftPreflightWriterReady ||
			!resonanceGraftPreflight.AdmissionResonanceGraftPreflightRollbackReady ||
			!resonanceGraftPreflight.AdmissionResonanceGraftPreflightLedgerReady ||
			resonanceGraftPreflight.AdmissionResonanceGraftPreflightKind != "shadow_graft_preflight" ||
			resonanceGraftPreflight.AdmissionResonanceGraftPreflightMode != "no_mutation_preflight" ||
			resonanceGraftPreflight.AdmissionResonanceGraftPreflightStage != "pre_live_graft_admission" ||
			resonanceGraftPreflight.AdmissionResonanceGraftPreflightCausalID == "" ||
			resonanceGraftPreflight.AdmissionResonanceGraftPreflightCausalID != admissionLiveRouteTurnCandidateAdmissionResonanceGraftPreflightCausalID(resonanceGraftPreflight) ||
			resonanceGraftPreflight.AdmissionResonanceGraftPreflightHash == "" ||
			resonanceGraftPreflight.AdmissionResonanceGraftPreflightHash != admissionLiveRouteTurnCandidateAdmissionResonanceGraftPreflightHash(resonanceGraftPreflight) ||
			resonanceGraftPreflight.AdmissionResonanceGraftPreflightReadBackHash == "" ||
			resonanceGraftPreflight.AdmissionResonanceGraftPreflightReadBackHash != admissionLiveRouteTurnCandidateAdmissionResonanceGraftPreflightReadBackHash(resonanceGraftPreflight) ||
			resonanceGraftPreflight.AdmissionResonanceGraftPreflightHash == resonanceGraftPreflight.AdmissionResonanceGraftPreflightReadBackHash ||
			!resonanceGraftPreflight.AdmissionResonanceGraftPreflightAdmissionRequired ||
			!resonanceGraftPreflight.AdmissionResonanceGraftPreflightShadowOnly ||
			resonanceGraftPreflight.AdmissionResonanceGraftPreflightGraftAllowed ||
			resonanceGraftPreflight.AdmissionResonanceGraftPreflightRawDreamTextAllowed ||
			resonanceGraftPreflight.AdmissionResonanceGraftPreflightJanusSurfaceAllowed ||
			resonanceGraftPreflight.AdmissionResonanceGraftPreflightCoocLearningAllowed ||
			resonanceGraftPreflight.AdmissionResonanceGraftPreflightDeltaHarvestAllowed ||
			resonanceGraftPreflight.AdmissionResonanceGraftPreflightBodyMutationAllowed ||
			!resonanceGraftPreflight.AdmissionResonanceGraftPreflightRollbackRequired ||
			!resonanceGraftPreflight.AdmissionResonanceGraftPreflightReady ||
			resonanceGraftPreflight.SourceAdmissionResonanceGraftBoundarySchema != admissionLiveRouteTurnCandidateAdmissionResonanceGraftBoundarySchema ||
			!resonanceGraftPreflight.SourceAdmissionResonanceGraftBoundaryPassed ||
			resonanceGraftPreflight.SourceAdmissionResonanceGraftBoundaryID != resonanceGraftBoundary.AdmissionResonanceGraftBoundaryID ||
			resonanceGraftPreflight.SourceAdmissionResonanceGraftBoundaryAction != "declare_resonance_shadow_graft_boundary_dry_run" ||
			!resonanceGraftPreflight.SourceAdmissionResonanceGraftBoundaryReady ||
			resonanceGraftPreflight.SourceAdmissionResonanceGraftBoundaryCausalID != resonanceGraftBoundary.AdmissionResonanceGraftBoundaryCausalID ||
			resonanceGraftPreflight.SourceAdmissionResonanceGraftBoundaryHash != resonanceGraftBoundary.AdmissionResonanceGraftBoundaryHash ||
			resonanceGraftPreflight.SourceAdmissionResonanceGraftBoundaryReadBackHash != resonanceGraftBoundary.AdmissionResonanceGraftBoundaryReadBackHash ||
			resonanceGraftPreflight.SourceAdmissionResonanceObservationIDForGraftPreflight != resonanceObservation.AdmissionResonanceObservationID ||
			resonanceGraftPreflight.SourceAdmissionResonanceReceiverIDForGraftPreflight != resonanceReceiver.AdmissionResonanceReceiverID ||
			resonanceGraftPreflight.SourceAdmissionResonanceIntentIDForGraftPreflight != resonanceIntent.AdmissionResonanceIntentID ||
			resonanceGraftPreflight.SourceAdmissionFinalGateIDForGraftPreflight != finalGate.AdmissionFinalGateID ||
			resonanceGraftPreflight.SourceAdmissionSealIDForGraftPreflight != seal.AdmissionSealID ||
			resonanceGraftPreflight.SourceAdmissionPermitIDForGraftPreflight != permit.AdmissionPermitID ||
			resonanceGraftPreflight.SourceAdmissionReadinessIDForGraftPreflight != readiness.AdmissionReadinessID ||
			resonanceGraftPreflight.SourceLedgerVerificationIDForGraftPreflight != ledgerVerification.LedgerVerificationID ||
			resonanceGraftPreflight.SourceLedgerPersistenceIDForGraftPreflight != ledgerPersistence.LedgerPersistenceID ||
			resonanceGraftPreflight.SourceLedgerImplementationIDForGraftPreflight != ledgerImpl.LedgerImplementationID ||
			resonanceGraftPreflight.SourceAdmissionLedgerIDForGraftPreflight != ledger.AdmissionLedgerID ||
			resonanceGraftPreflight.SourceRollbackImplementationIDForGraftPreflight != rollbackImpl.RollbackImplementationID ||
			resonanceGraftPreflight.SourceWriterReceiptIDForGraftPreflight != writerReceipt.WriterReceiptID ||
			resonanceGraftPreflight.AdmissionResonanceGraftPreflightID == "" ||
			resonanceGraftPreflight.AdmissionResonanceGraftPreflightID != admissionLiveRouteTurnCandidateAdmissionResonanceGraftPreflightID(resonanceGraftPreflight) ||
			resonanceGraftPreflight.AdmissionResonanceGraftBoundaryID != resonanceGraftBoundary.AdmissionResonanceGraftBoundaryID ||
			resonanceGraftPreflight.AdmissionResonanceObservationID != resonanceObservation.AdmissionResonanceObservationID ||
			resonanceGraftPreflight.AdmissionResonanceReceiverID != resonanceReceiver.AdmissionResonanceReceiverID ||
			resonanceGraftPreflight.AdmissionResonanceIntentID != resonanceIntent.AdmissionResonanceIntentID ||
			resonanceGraftPreflight.AdmissionFinalGateID != finalGate.AdmissionFinalGateID ||
			resonanceGraftPreflight.AdmissionSealID != seal.AdmissionSealID ||
			resonanceGraftPreflight.AdmissionPermitID != permit.AdmissionPermitID ||
			resonanceGraftPreflight.AdmissionReadinessID != readiness.AdmissionReadinessID ||
			resonanceGraftPreflight.LedgerVerificationID != ledgerVerification.LedgerVerificationID ||
			resonanceGraftPreflight.LedgerPersistenceID != ledgerPersistence.LedgerPersistenceID ||
			resonanceGraftPreflight.LedgerImplementationID != ledgerImpl.LedgerImplementationID ||
			resonanceGraftPreflight.RollbackImplementationID != rollbackImpl.RollbackImplementationID ||
			resonanceGraftPreflight.WriterReceiptID != writerReceipt.WriterReceiptID ||
			resonanceGraftPreflight.AdmissionLedgerID != ledger.AdmissionLedgerID ||
			resonanceGraftPreflight.AdmissionWriterContractID != writerContract.WriterContractID ||
			resonanceGraftPreflight.AdmissionWriterInventoryID != writerInventory.WriterInventoryID ||
			resonanceGraftPreflight.AdmissionWriterPreflightID != writerPreflight.WriterPreflightID ||
			resonanceGraftPreflight.AdmissionLiveStageID != liveStage.LiveStageID ||
			resonanceGraftPreflight.AdmissionEnableGateID != gate.EnableGateID ||
			resonanceGraftPreflight.AdmissionSwitchID != sw.SwitchID ||
			resonanceGraftPreflight.AdmissionPromotionID != promotion.PromotionID ||
			resonanceGraftPreflight.AdmissionDecisionID != decision.DecisionID ||
			resonanceGraftPreflight.CandidateExecutionID != execution.ExecutionID ||
			resonanceGraftPreflight.GeneratorAdapterID != generatorAdapter.AdapterID ||
			resonanceGraftPreflight.CandidateDraftID != draft.DraftID ||
			resonanceGraftPreflight.HandoffID != admission.HandoffID ||
			resonanceGraftPreflight.AdmissionAdapterID != admissionAdapter.AdmissionAdapterID ||
			resonanceGraftPreflight.DreamCandidateRunID != candidate.RunID ||
			resonanceGraftPreflight.CandidateRunID != draft.CandidateRunID ||
			resonanceGraftPreflight.CandidateTextHash != execution.GeneratedTextHash ||
			resonanceGraftPreflight.TurnTextHash != execution.TurnTextHash ||
			resonanceGraftPreflight.Reason != "resonance shadow graft preflight prepared without body mutation" {
			return fmt.Errorf("bad nano-direct admission resonance graft preflight receipt: resonance_graft_preflight=%+v resonance_graft_boundary=%+v resonance_observation=%+v resonance_receiver=%+v resonance_intent=%+v final_gate=%+v seal=%+v permit=%+v readiness=%+v ledger_verification=%+v ledger_persistence=%+v ledger_implementation=%+v rollback_implementation=%+v writer_receipt=%+v writer_implementation=%+v ledger=%+v writer_contract=%+v writer_inventory=%+v writer_preflight=%+v stage=%+v gate=%+v switch=%+v promotion=%+v decision=%+v execution=%+v", resonanceGraftPreflight, resonanceGraftBoundary, resonanceObservation, resonanceReceiver, resonanceIntent, finalGate, seal, permit, readiness, ledgerVerification, ledgerPersistence, ledgerImpl, rollbackImpl, writerReceipt, writerImpl, ledger, writerContract, writerInventory, writerPreflight, liveStage, gate, sw, promotion, decision, execution)
		}
	}
	if admissionLiveRouteTurnCandidateAdmissionResonanceGraftGateDryRun() {
		if resonanceGraftGate.Schema != admissionLiveRouteTurnCandidateAdmissionResonanceGraftGateSchema ||
			!resonanceGraftGate.Passed ||
			!resonanceGraftGate.LiveReady ||
			resonanceGraftGate.LiveAdmissionEnabled ||
			resonanceGraftGate.AdmissionAllowed ||
			resonanceGraftGate.ContractsReady ||
			resonanceGraftGate.WriteAllowed ||
			resonanceGraftGate.MutatesState ||
			resonanceGraftGate.BodyTarget != "none" ||
			resonanceGraftGate.AdmissionResonanceGraftGateState != "shadow_graft_gate_ready_dry_run" ||
			resonanceGraftGate.AdmissionResonanceGraftGateAction != "gate_resonance_shadow_graft_dry_run" ||
			resonanceGraftGate.AdmissionResonanceGraftGateTarget != "resonance" ||
			resonanceGraftGate.AdmissionResonanceGraftGateTargetKind != "internal_world_shadow_graft_gate" ||
			resonanceGraftGate.AdmissionResonanceGraftGateTargetMode != "receipt_only_closed_gate_dry_run" ||
			resonanceGraftGate.AdmissionResonanceGraftGateReceiptShape != "resonance_shadow_graft_gate_contract" ||
			!resonanceGraftGate.AdmissionResonanceGraftGateDryRunOnly ||
			!resonanceGraftGate.AdmissionResonanceGraftGatePreflightVerified ||
			!resonanceGraftGate.AdmissionResonanceGraftGateBoundaryVerified ||
			!resonanceGraftGate.AdmissionResonanceGraftGateObservationVerified ||
			!resonanceGraftGate.AdmissionResonanceGraftGateReceiverVerified ||
			!resonanceGraftGate.AdmissionResonanceGraftGateIntentVerified ||
			!resonanceGraftGate.AdmissionResonanceGraftGateFinalGateVerified ||
			!resonanceGraftGate.AdmissionResonanceGraftGateSealVerified ||
			!resonanceGraftGate.AdmissionResonanceGraftGatePermitVerified ||
			!resonanceGraftGate.AdmissionResonanceGraftGateReadinessVerified ||
			!resonanceGraftGate.AdmissionResonanceGraftGateLedgerVerified ||
			!resonanceGraftGate.AdmissionResonanceGraftGateWriterReady ||
			!resonanceGraftGate.AdmissionResonanceGraftGateRollbackReady ||
			!resonanceGraftGate.AdmissionResonanceGraftGateLedgerReady ||
			resonanceGraftGate.AdmissionResonanceGraftGateKind != "shadow_graft_gate" ||
			resonanceGraftGate.AdmissionResonanceGraftGateMode != "no_mutation_gate" ||
			resonanceGraftGate.AdmissionResonanceGraftGateStage != "pre_live_graft_gate" ||
			resonanceGraftGate.AdmissionResonanceGraftGateCausalID == "" ||
			resonanceGraftGate.AdmissionResonanceGraftGateCausalID != admissionLiveRouteTurnCandidateAdmissionResonanceGraftGateCausalID(resonanceGraftGate) ||
			resonanceGraftGate.AdmissionResonanceGraftGateHash == "" ||
			resonanceGraftGate.AdmissionResonanceGraftGateHash != admissionLiveRouteTurnCandidateAdmissionResonanceGraftGateHash(resonanceGraftGate) ||
			resonanceGraftGate.AdmissionResonanceGraftGateReadBackHash == "" ||
			resonanceGraftGate.AdmissionResonanceGraftGateReadBackHash != admissionLiveRouteTurnCandidateAdmissionResonanceGraftGateReadBackHash(resonanceGraftGate) ||
			resonanceGraftGate.AdmissionResonanceGraftGateHash == resonanceGraftGate.AdmissionResonanceGraftGateReadBackHash ||
			!resonanceGraftGate.AdmissionResonanceGraftGateAdmissionRequired ||
			!resonanceGraftGate.AdmissionResonanceGraftGateShadowOnly ||
			resonanceGraftGate.AdmissionResonanceGraftGateGraftAllowed ||
			resonanceGraftGate.AdmissionResonanceGraftGateRawDreamTextAllowed ||
			resonanceGraftGate.AdmissionResonanceGraftGateJanusSurfaceAllowed ||
			resonanceGraftGate.AdmissionResonanceGraftGateCoocLearningAllowed ||
			resonanceGraftGate.AdmissionResonanceGraftGateDeltaHarvestAllowed ||
			resonanceGraftGate.AdmissionResonanceGraftGateBodyMutationAllowed ||
			!resonanceGraftGate.AdmissionResonanceGraftGateRollbackRequired ||
			!resonanceGraftGate.AdmissionResonanceGraftGateReady ||
			resonanceGraftGate.SourceAdmissionResonanceGraftPreflightSchema != admissionLiveRouteTurnCandidateAdmissionResonanceGraftPreflightSchema ||
			!resonanceGraftGate.SourceAdmissionResonanceGraftPreflightPassed ||
			resonanceGraftGate.SourceAdmissionResonanceGraftPreflightID != resonanceGraftPreflight.AdmissionResonanceGraftPreflightID ||
			resonanceGraftGate.SourceAdmissionResonanceGraftPreflightAction != "prepare_resonance_shadow_graft_preflight_dry_run" ||
			!resonanceGraftGate.SourceAdmissionResonanceGraftPreflightReady ||
			resonanceGraftGate.SourceAdmissionResonanceGraftPreflightCausalID != resonanceGraftPreflight.AdmissionResonanceGraftPreflightCausalID ||
			resonanceGraftGate.SourceAdmissionResonanceGraftPreflightHash != resonanceGraftPreflight.AdmissionResonanceGraftPreflightHash ||
			resonanceGraftGate.SourceAdmissionResonanceGraftPreflightReadBackHash != resonanceGraftPreflight.AdmissionResonanceGraftPreflightReadBackHash ||
			resonanceGraftGate.SourceAdmissionResonanceGraftBoundaryIDForGraftGate != resonanceGraftBoundary.AdmissionResonanceGraftBoundaryID ||
			resonanceGraftGate.SourceAdmissionResonanceObservationIDForGraftGate != resonanceObservation.AdmissionResonanceObservationID ||
			resonanceGraftGate.SourceAdmissionResonanceReceiverIDForGraftGate != resonanceReceiver.AdmissionResonanceReceiverID ||
			resonanceGraftGate.SourceAdmissionResonanceIntentIDForGraftGate != resonanceIntent.AdmissionResonanceIntentID ||
			resonanceGraftGate.SourceAdmissionFinalGateIDForGraftGate != finalGate.AdmissionFinalGateID ||
			resonanceGraftGate.SourceAdmissionSealIDForGraftGate != seal.AdmissionSealID ||
			resonanceGraftGate.SourceAdmissionPermitIDForGraftGate != permit.AdmissionPermitID ||
			resonanceGraftGate.SourceAdmissionReadinessIDForGraftGate != readiness.AdmissionReadinessID ||
			resonanceGraftGate.SourceLedgerVerificationIDForGraftGate != ledgerVerification.LedgerVerificationID ||
			resonanceGraftGate.SourceLedgerPersistenceIDForGraftGate != ledgerPersistence.LedgerPersistenceID ||
			resonanceGraftGate.SourceLedgerImplementationIDForGraftGate != ledgerImpl.LedgerImplementationID ||
			resonanceGraftGate.SourceAdmissionLedgerIDForGraftGate != ledger.AdmissionLedgerID ||
			resonanceGraftGate.SourceRollbackImplementationIDForGraftGate != rollbackImpl.RollbackImplementationID ||
			resonanceGraftGate.SourceWriterReceiptIDForGraftGate != writerReceipt.WriterReceiptID ||
			resonanceGraftGate.AdmissionResonanceGraftGateID == "" ||
			resonanceGraftGate.AdmissionResonanceGraftGateID != admissionLiveRouteTurnCandidateAdmissionResonanceGraftGateID(resonanceGraftGate) ||
			resonanceGraftGate.AdmissionResonanceGraftPreflightID != resonanceGraftPreflight.AdmissionResonanceGraftPreflightID ||
			resonanceGraftGate.AdmissionResonanceGraftBoundaryID != resonanceGraftBoundary.AdmissionResonanceGraftBoundaryID ||
			resonanceGraftGate.AdmissionResonanceObservationID != resonanceObservation.AdmissionResonanceObservationID ||
			resonanceGraftGate.AdmissionResonanceReceiverID != resonanceReceiver.AdmissionResonanceReceiverID ||
			resonanceGraftGate.AdmissionResonanceIntentID != resonanceIntent.AdmissionResonanceIntentID ||
			resonanceGraftGate.AdmissionFinalGateID != finalGate.AdmissionFinalGateID ||
			resonanceGraftGate.AdmissionSealID != seal.AdmissionSealID ||
			resonanceGraftGate.AdmissionPermitID != permit.AdmissionPermitID ||
			resonanceGraftGate.AdmissionReadinessID != readiness.AdmissionReadinessID ||
			resonanceGraftGate.LedgerVerificationID != ledgerVerification.LedgerVerificationID ||
			resonanceGraftGate.LedgerPersistenceID != ledgerPersistence.LedgerPersistenceID ||
			resonanceGraftGate.LedgerImplementationID != ledgerImpl.LedgerImplementationID ||
			resonanceGraftGate.RollbackImplementationID != rollbackImpl.RollbackImplementationID ||
			resonanceGraftGate.WriterReceiptID != writerReceipt.WriterReceiptID ||
			resonanceGraftGate.AdmissionLedgerID != ledger.AdmissionLedgerID ||
			resonanceGraftGate.AdmissionWriterContractID != writerContract.WriterContractID ||
			resonanceGraftGate.AdmissionWriterInventoryID != writerInventory.WriterInventoryID ||
			resonanceGraftGate.AdmissionWriterPreflightID != writerPreflight.WriterPreflightID ||
			resonanceGraftGate.AdmissionLiveStageID != liveStage.LiveStageID ||
			resonanceGraftGate.AdmissionEnableGateID != gate.EnableGateID ||
			resonanceGraftGate.AdmissionSwitchID != sw.SwitchID ||
			resonanceGraftGate.AdmissionPromotionID != promotion.PromotionID ||
			resonanceGraftGate.AdmissionDecisionID != decision.DecisionID ||
			resonanceGraftGate.CandidateExecutionID != execution.ExecutionID ||
			resonanceGraftGate.GeneratorAdapterID != generatorAdapter.AdapterID ||
			resonanceGraftGate.CandidateDraftID != draft.DraftID ||
			resonanceGraftGate.HandoffID != admission.HandoffID ||
			resonanceGraftGate.AdmissionAdapterID != admissionAdapter.AdmissionAdapterID ||
			resonanceGraftGate.DreamCandidateRunID != candidate.RunID ||
			resonanceGraftGate.CandidateRunID != draft.CandidateRunID ||
			resonanceGraftGate.CandidateTextHash != execution.GeneratedTextHash ||
			resonanceGraftGate.TurnTextHash != execution.TurnTextHash ||
			resonanceGraftGate.Reason != "resonance shadow graft gate prepared without body mutation" {
			return fmt.Errorf("bad nano-direct admission resonance graft gate receipt: resonance_graft_gate=%+v resonance_graft_preflight=%+v resonance_graft_boundary=%+v resonance_observation=%+v resonance_receiver=%+v resonance_intent=%+v final_gate=%+v seal=%+v permit=%+v readiness=%+v ledger_verification=%+v ledger_persistence=%+v ledger_implementation=%+v rollback_implementation=%+v writer_receipt=%+v writer_implementation=%+v ledger=%+v writer_contract=%+v writer_inventory=%+v writer_preflight=%+v stage=%+v gate=%+v switch=%+v promotion=%+v decision=%+v execution=%+v", resonanceGraftGate, resonanceGraftPreflight, resonanceGraftBoundary, resonanceObservation, resonanceReceiver, resonanceIntent, finalGate, seal, permit, readiness, ledgerVerification, ledgerPersistence, ledgerImpl, rollbackImpl, writerReceipt, writerImpl, ledger, writerContract, writerInventory, writerPreflight, liveStage, gate, sw, promotion, decision, execution)
		}
	}
	if admissionLiveRouteTurnCandidateAdmissionResonanceGraftCandidateDryRun() {
		if resonanceGraftCandidate.Schema != admissionLiveRouteTurnCandidateAdmissionResonanceGraftCandidateSchema ||
			!resonanceGraftCandidate.Passed ||
			!resonanceGraftCandidate.LiveReady ||
			resonanceGraftCandidate.LiveAdmissionEnabled ||
			resonanceGraftCandidate.AdmissionAllowed ||
			resonanceGraftCandidate.ContractsReady ||
			resonanceGraftCandidate.WriteAllowed ||
			resonanceGraftCandidate.MutatesState ||
			resonanceGraftCandidate.BodyTarget != "none" ||
			resonanceGraftCandidate.AdmissionResonanceGraftCandidateState != "shadow_graft_candidate_ready_dry_run" ||
			resonanceGraftCandidate.AdmissionResonanceGraftCandidateAction != "draft_resonance_shadow_graft_candidate_dry_run" ||
			resonanceGraftCandidate.AdmissionResonanceGraftCandidateTarget != "resonance" ||
			resonanceGraftCandidate.AdmissionResonanceGraftCandidateTargetKind != "internal_world_shadow_graft_candidate" ||
			resonanceGraftCandidate.AdmissionResonanceGraftCandidateTargetMode != "receipt_only_closed_candidate_dry_run" ||
			resonanceGraftCandidate.AdmissionResonanceGraftCandidateReceiptShape != "resonance_shadow_graft_candidate_contract" ||
			!resonanceGraftCandidate.AdmissionResonanceGraftCandidateDryRunOnly ||
			!resonanceGraftCandidate.AdmissionResonanceGraftCandidateGateVerified ||
			!resonanceGraftCandidate.AdmissionResonanceGraftCandidatePreflightVerified ||
			!resonanceGraftCandidate.AdmissionResonanceGraftCandidateBoundaryVerified ||
			!resonanceGraftCandidate.AdmissionResonanceGraftCandidateObservationVerified ||
			!resonanceGraftCandidate.AdmissionResonanceGraftCandidateReceiverVerified ||
			!resonanceGraftCandidate.AdmissionResonanceGraftCandidateIntentVerified ||
			!resonanceGraftCandidate.AdmissionResonanceGraftCandidateFinalGateVerified ||
			!resonanceGraftCandidate.AdmissionResonanceGraftCandidateSealVerified ||
			!resonanceGraftCandidate.AdmissionResonanceGraftCandidatePermitVerified ||
			!resonanceGraftCandidate.AdmissionResonanceGraftCandidateReadinessVerified ||
			!resonanceGraftCandidate.AdmissionResonanceGraftCandidateLedgerVerified ||
			!resonanceGraftCandidate.AdmissionResonanceGraftCandidateWriterReady ||
			!resonanceGraftCandidate.AdmissionResonanceGraftCandidateRollbackReady ||
			!resonanceGraftCandidate.AdmissionResonanceGraftCandidateLedgerReady ||
			resonanceGraftCandidate.AdmissionResonanceGraftCandidateKind != "shadow_graft_candidate" ||
			resonanceGraftCandidate.AdmissionResonanceGraftCandidateMode != "no_mutation_candidate" ||
			resonanceGraftCandidate.AdmissionResonanceGraftCandidateStage != "pre_live_graft_candidate" ||
			resonanceGraftCandidate.AdmissionResonanceGraftCandidateCausalID == "" ||
			resonanceGraftCandidate.AdmissionResonanceGraftCandidateCausalID != admissionLiveRouteTurnCandidateAdmissionResonanceGraftCandidateCausalID(resonanceGraftCandidate) ||
			resonanceGraftCandidate.AdmissionResonanceGraftCandidateHash == "" ||
			resonanceGraftCandidate.AdmissionResonanceGraftCandidateHash != admissionLiveRouteTurnCandidateAdmissionResonanceGraftCandidateHash(resonanceGraftCandidate) ||
			resonanceGraftCandidate.AdmissionResonanceGraftCandidateReadBackHash == "" ||
			resonanceGraftCandidate.AdmissionResonanceGraftCandidateReadBackHash != admissionLiveRouteTurnCandidateAdmissionResonanceGraftCandidateReadBackHash(resonanceGraftCandidate) ||
			resonanceGraftCandidate.AdmissionResonanceGraftCandidateHash == resonanceGraftCandidate.AdmissionResonanceGraftCandidateReadBackHash ||
			!resonanceGraftCandidate.AdmissionResonanceGraftCandidateAdmissionRequired ||
			!resonanceGraftCandidate.AdmissionResonanceGraftCandidateShadowOnly ||
			resonanceGraftCandidate.AdmissionResonanceGraftCandidateGraftAllowed ||
			resonanceGraftCandidate.AdmissionResonanceGraftCandidateRawDreamTextAllowed ||
			resonanceGraftCandidate.AdmissionResonanceGraftCandidateJanusSurfaceAllowed ||
			resonanceGraftCandidate.AdmissionResonanceGraftCandidateCoocLearningAllowed ||
			resonanceGraftCandidate.AdmissionResonanceGraftCandidateDeltaHarvestAllowed ||
			resonanceGraftCandidate.AdmissionResonanceGraftCandidateBodyMutationAllowed ||
			!resonanceGraftCandidate.AdmissionResonanceGraftCandidateRollbackRequired ||
			!resonanceGraftCandidate.AdmissionResonanceGraftCandidateReady ||
			resonanceGraftCandidate.SourceAdmissionResonanceGraftGateSchema != admissionLiveRouteTurnCandidateAdmissionResonanceGraftGateSchema ||
			!resonanceGraftCandidate.SourceAdmissionResonanceGraftGatePassed ||
			resonanceGraftCandidate.SourceAdmissionResonanceGraftGateID != resonanceGraftGate.AdmissionResonanceGraftGateID ||
			resonanceGraftCandidate.SourceAdmissionResonanceGraftGateAction != "gate_resonance_shadow_graft_dry_run" ||
			!resonanceGraftCandidate.SourceAdmissionResonanceGraftGateReady ||
			resonanceGraftCandidate.SourceAdmissionResonanceGraftGateCausalID != resonanceGraftGate.AdmissionResonanceGraftGateCausalID ||
			resonanceGraftCandidate.SourceAdmissionResonanceGraftGateHash != resonanceGraftGate.AdmissionResonanceGraftGateHash ||
			resonanceGraftCandidate.SourceAdmissionResonanceGraftGateReadBackHash != resonanceGraftGate.AdmissionResonanceGraftGateReadBackHash ||
			resonanceGraftCandidate.SourceAdmissionResonanceGraftPreflightIDForCandidate != resonanceGraftPreflight.AdmissionResonanceGraftPreflightID ||
			resonanceGraftCandidate.SourceAdmissionResonanceGraftBoundaryIDForCandidate != resonanceGraftBoundary.AdmissionResonanceGraftBoundaryID ||
			resonanceGraftCandidate.SourceAdmissionResonanceObservationIDForCandidate != resonanceObservation.AdmissionResonanceObservationID ||
			resonanceGraftCandidate.SourceAdmissionResonanceReceiverIDForCandidate != resonanceReceiver.AdmissionResonanceReceiverID ||
			resonanceGraftCandidate.SourceAdmissionResonanceIntentIDForCandidate != resonanceIntent.AdmissionResonanceIntentID ||
			resonanceGraftCandidate.SourceAdmissionFinalGateIDForCandidate != finalGate.AdmissionFinalGateID ||
			resonanceGraftCandidate.SourceAdmissionSealIDForCandidate != seal.AdmissionSealID ||
			resonanceGraftCandidate.SourceAdmissionPermitIDForCandidate != permit.AdmissionPermitID ||
			resonanceGraftCandidate.SourceAdmissionReadinessIDForCandidate != readiness.AdmissionReadinessID ||
			resonanceGraftCandidate.SourceLedgerVerificationIDForCandidate != ledgerVerification.LedgerVerificationID ||
			resonanceGraftCandidate.SourceLedgerPersistenceIDForCandidate != ledgerPersistence.LedgerPersistenceID ||
			resonanceGraftCandidate.SourceLedgerImplementationIDForCandidate != ledgerImpl.LedgerImplementationID ||
			resonanceGraftCandidate.SourceAdmissionLedgerIDForCandidate != ledger.AdmissionLedgerID ||
			resonanceGraftCandidate.SourceRollbackImplementationIDForCandidate != rollbackImpl.RollbackImplementationID ||
			resonanceGraftCandidate.SourceWriterReceiptIDForCandidate != writerReceipt.WriterReceiptID ||
			resonanceGraftCandidate.AdmissionResonanceGraftCandidateID == "" ||
			resonanceGraftCandidate.AdmissionResonanceGraftCandidateID != admissionLiveRouteTurnCandidateAdmissionResonanceGraftCandidateID(resonanceGraftCandidate) ||
			resonanceGraftCandidate.AdmissionResonanceGraftGateID != resonanceGraftGate.AdmissionResonanceGraftGateID ||
			resonanceGraftCandidate.AdmissionResonanceGraftPreflightID != resonanceGraftPreflight.AdmissionResonanceGraftPreflightID ||
			resonanceGraftCandidate.AdmissionResonanceGraftBoundaryID != resonanceGraftBoundary.AdmissionResonanceGraftBoundaryID ||
			resonanceGraftCandidate.AdmissionResonanceObservationID != resonanceObservation.AdmissionResonanceObservationID ||
			resonanceGraftCandidate.AdmissionResonanceReceiverID != resonanceReceiver.AdmissionResonanceReceiverID ||
			resonanceGraftCandidate.AdmissionResonanceIntentID != resonanceIntent.AdmissionResonanceIntentID ||
			resonanceGraftCandidate.AdmissionFinalGateID != finalGate.AdmissionFinalGateID ||
			resonanceGraftCandidate.AdmissionSealID != seal.AdmissionSealID ||
			resonanceGraftCandidate.AdmissionPermitID != permit.AdmissionPermitID ||
			resonanceGraftCandidate.AdmissionReadinessID != readiness.AdmissionReadinessID ||
			resonanceGraftCandidate.LedgerVerificationID != ledgerVerification.LedgerVerificationID ||
			resonanceGraftCandidate.LedgerPersistenceID != ledgerPersistence.LedgerPersistenceID ||
			resonanceGraftCandidate.LedgerImplementationID != ledgerImpl.LedgerImplementationID ||
			resonanceGraftCandidate.RollbackImplementationID != rollbackImpl.RollbackImplementationID ||
			resonanceGraftCandidate.WriterReceiptID != writerReceipt.WriterReceiptID ||
			resonanceGraftCandidate.AdmissionLedgerID != ledger.AdmissionLedgerID ||
			resonanceGraftCandidate.AdmissionWriterContractID != writerContract.WriterContractID ||
			resonanceGraftCandidate.AdmissionWriterInventoryID != writerInventory.WriterInventoryID ||
			resonanceGraftCandidate.AdmissionWriterPreflightID != writerPreflight.WriterPreflightID ||
			resonanceGraftCandidate.AdmissionLiveStageID != liveStage.LiveStageID ||
			resonanceGraftCandidate.AdmissionEnableGateID != gate.EnableGateID ||
			resonanceGraftCandidate.AdmissionSwitchID != sw.SwitchID ||
			resonanceGraftCandidate.AdmissionPromotionID != promotion.PromotionID ||
			resonanceGraftCandidate.AdmissionDecisionID != decision.DecisionID ||
			resonanceGraftCandidate.CandidateExecutionID != execution.ExecutionID ||
			resonanceGraftCandidate.GeneratorAdapterID != generatorAdapter.AdapterID ||
			resonanceGraftCandidate.CandidateDraftID != draft.DraftID ||
			resonanceGraftCandidate.HandoffID != admission.HandoffID ||
			resonanceGraftCandidate.AdmissionAdapterID != admissionAdapter.AdmissionAdapterID ||
			resonanceGraftCandidate.DreamCandidateRunID != candidate.RunID ||
			resonanceGraftCandidate.CandidateRunID != draft.CandidateRunID ||
			resonanceGraftCandidate.CandidateTextHash != execution.GeneratedTextHash ||
			resonanceGraftCandidate.TurnTextHash != execution.TurnTextHash ||
			resonanceGraftCandidate.Reason != "resonance shadow graft candidate drafted without body mutation" {
			return fmt.Errorf("bad nano-direct admission resonance graft candidate receipt: resonance_graft_candidate=%+v resonance_graft_gate=%+v resonance_graft_preflight=%+v resonance_graft_boundary=%+v resonance_observation=%+v resonance_receiver=%+v resonance_intent=%+v final_gate=%+v seal=%+v permit=%+v readiness=%+v ledger_verification=%+v ledger_persistence=%+v ledger_implementation=%+v rollback_implementation=%+v writer_receipt=%+v writer_implementation=%+v ledger=%+v writer_contract=%+v writer_inventory=%+v writer_preflight=%+v stage=%+v gate=%+v switch=%+v promotion=%+v decision=%+v execution=%+v", resonanceGraftCandidate, resonanceGraftGate, resonanceGraftPreflight, resonanceGraftBoundary, resonanceObservation, resonanceReceiver, resonanceIntent, finalGate, seal, permit, readiness, ledgerVerification, ledgerPersistence, ledgerImpl, rollbackImpl, writerReceipt, writerImpl, ledger, writerContract, writerInventory, writerPreflight, liveStage, gate, sw, promotion, decision, execution)
		}
	}
	if admissionLiveRouteTurnCandidateAdmissionResonanceGraftCandidateStoreDryRun() {
		if resonanceGraftCandidateStore.Schema != admissionLiveRouteTurnCandidateAdmissionResonanceGraftCandidateStoreSchema ||
			resonanceGraftCandidateStore.Timing != "live_admission_resonance_graft_candidate_store" ||
			!resonanceGraftCandidateStore.Passed ||
			!resonanceGraftCandidateStore.LiveReady ||
			resonanceGraftCandidateStore.LiveAdmissionEnabled ||
			resonanceGraftCandidateStore.AdmissionAllowed ||
			resonanceGraftCandidateStore.ContractsReady ||
			resonanceGraftCandidateStore.WriteAllowed ||
			resonanceGraftCandidateStore.MutatesState ||
			resonanceGraftCandidateStore.BodyTarget != "none" ||
			resonanceGraftCandidateStore.AdmissionResonanceGraftCandidateStoreState != "shadow_graft_candidate_stored_dry_run" ||
			resonanceGraftCandidateStore.AdmissionResonanceGraftCandidateStoreAction != "store_resonance_shadow_graft_candidate_dry_run" ||
			resonanceGraftCandidateStore.AdmissionResonanceGraftCandidateStoreTarget != "resonance" ||
			resonanceGraftCandidateStore.AdmissionResonanceGraftCandidateStoreTargetKind != "internal_world_shadow_graft_candidate_store" ||
			resonanceGraftCandidateStore.AdmissionResonanceGraftCandidateStoreTargetMode != "append_only_read_back_store_dry_run" ||
			resonanceGraftCandidateStore.AdmissionResonanceGraftCandidateStoreReceiptShape != "resonance_shadow_graft_candidate_store_receipt" ||
			!resonanceGraftCandidateStore.AdmissionResonanceGraftCandidateStoreDryRunOnly ||
			!resonanceGraftCandidateStore.AdmissionResonanceGraftCandidateStoreCandidateVerified ||
			!resonanceGraftCandidateStore.AdmissionResonanceGraftCandidateStoreGateVerified ||
			!resonanceGraftCandidateStore.AdmissionResonanceGraftCandidateStorePreflightVerified ||
			!resonanceGraftCandidateStore.AdmissionResonanceGraftCandidateStoreBoundaryVerified ||
			!resonanceGraftCandidateStore.AdmissionResonanceGraftCandidateStoreObservationVerified ||
			!resonanceGraftCandidateStore.AdmissionResonanceGraftCandidateStoreReceiverVerified ||
			!resonanceGraftCandidateStore.AdmissionResonanceGraftCandidateStoreIntentVerified ||
			!resonanceGraftCandidateStore.AdmissionResonanceGraftCandidateStoreFinalGateVerified ||
			!resonanceGraftCandidateStore.AdmissionResonanceGraftCandidateStoreSealVerified ||
			!resonanceGraftCandidateStore.AdmissionResonanceGraftCandidateStorePermitVerified ||
			!resonanceGraftCandidateStore.AdmissionResonanceGraftCandidateStoreReadinessVerified ||
			!resonanceGraftCandidateStore.AdmissionResonanceGraftCandidateStoreLedgerVerified ||
			!resonanceGraftCandidateStore.AdmissionResonanceGraftCandidateStoreWriterReady ||
			!resonanceGraftCandidateStore.AdmissionResonanceGraftCandidateStoreRollbackReady ||
			!resonanceGraftCandidateStore.AdmissionResonanceGraftCandidateStoreLedgerReady ||
			resonanceGraftCandidateStore.AdmissionResonanceGraftCandidateStoreKind != "shadow_graft_candidate_store" ||
			resonanceGraftCandidateStore.AdmissionResonanceGraftCandidateStoreMode != "append_only_read_back_store" ||
			resonanceGraftCandidateStore.AdmissionResonanceGraftCandidateStoreStage != "pre_live_graft_candidate_store" ||
			resonanceGraftCandidateStore.AdmissionResonanceGraftCandidateStoreCausalID == "" ||
			resonanceGraftCandidateStore.AdmissionResonanceGraftCandidateStoreCausalID != admissionLiveRouteTurnCandidateAdmissionResonanceGraftCandidateStoreCausalID(resonanceGraftCandidateStore) ||
			resonanceGraftCandidateStore.AdmissionResonanceGraftCandidateStoreHash == "" ||
			resonanceGraftCandidateStore.AdmissionResonanceGraftCandidateStoreHash != admissionLiveRouteTurnCandidateAdmissionResonanceGraftCandidateStoreHash(resonanceGraftCandidateStore) ||
			resonanceGraftCandidateStore.AdmissionResonanceGraftCandidateStoreReadBackHash == "" ||
			resonanceGraftCandidateStore.AdmissionResonanceGraftCandidateStoreReadBackHash != admissionLiveRouteTurnCandidateAdmissionResonanceGraftCandidateStoreReadBackHash(resonanceGraftCandidateStore) ||
			resonanceGraftCandidateStore.AdmissionResonanceGraftCandidateStoreHash == resonanceGraftCandidateStore.AdmissionResonanceGraftCandidateStoreReadBackHash ||
			!resonanceGraftCandidateStore.AdmissionResonanceGraftCandidateStoreAdmissionRequired ||
			!resonanceGraftCandidateStore.AdmissionResonanceGraftCandidateStoreShadowOnly ||
			resonanceGraftCandidateStore.AdmissionResonanceGraftCandidateStoreGraftAllowed ||
			resonanceGraftCandidateStore.AdmissionResonanceGraftCandidateStoreRawDreamTextAllowed ||
			resonanceGraftCandidateStore.AdmissionResonanceGraftCandidateStoreJanusSurfaceAllowed ||
			resonanceGraftCandidateStore.AdmissionResonanceGraftCandidateStoreCoocLearningAllowed ||
			resonanceGraftCandidateStore.AdmissionResonanceGraftCandidateStoreDeltaHarvestAllowed ||
			resonanceGraftCandidateStore.AdmissionResonanceGraftCandidateStoreBodyMutationAllowed ||
			!resonanceGraftCandidateStore.AdmissionResonanceGraftCandidateStoreRollbackRequired ||
			!resonanceGraftCandidateStore.AdmissionResonanceGraftCandidateStoreAppendOnly ||
			!resonanceGraftCandidateStore.AdmissionResonanceGraftCandidateStoreReadBack ||
			!resonanceGraftCandidateStore.AdmissionResonanceGraftCandidateStoreReceiptPersisted ||
			!resonanceGraftCandidateStore.AdmissionResonanceGraftCandidateStoreReceiptVerified ||
			!resonanceGraftCandidateStore.AdmissionResonanceGraftCandidateStoreReady ||
			resonanceGraftCandidateStore.SourceAdmissionResonanceGraftCandidateSchema != admissionLiveRouteTurnCandidateAdmissionResonanceGraftCandidateSchema ||
			!resonanceGraftCandidateStore.SourceAdmissionResonanceGraftCandidatePassed ||
			resonanceGraftCandidateStore.SourceAdmissionResonanceGraftCandidateID != resonanceGraftCandidate.AdmissionResonanceGraftCandidateID ||
			resonanceGraftCandidateStore.SourceAdmissionResonanceGraftCandidateAction != "draft_resonance_shadow_graft_candidate_dry_run" ||
			!resonanceGraftCandidateStore.SourceAdmissionResonanceGraftCandidateReady ||
			resonanceGraftCandidateStore.SourceAdmissionResonanceGraftCandidateCausalID != resonanceGraftCandidate.AdmissionResonanceGraftCandidateCausalID ||
			resonanceGraftCandidateStore.SourceAdmissionResonanceGraftCandidateHash != resonanceGraftCandidate.AdmissionResonanceGraftCandidateHash ||
			resonanceGraftCandidateStore.SourceAdmissionResonanceGraftCandidateReadBackHash != resonanceGraftCandidate.AdmissionResonanceGraftCandidateReadBackHash ||
			resonanceGraftCandidateStore.SourceAdmissionResonanceGraftGateIDForCandidateStore != resonanceGraftGate.AdmissionResonanceGraftGateID ||
			resonanceGraftCandidateStore.SourceAdmissionResonanceGraftPreflightIDForCandidateStore != resonanceGraftPreflight.AdmissionResonanceGraftPreflightID ||
			resonanceGraftCandidateStore.SourceAdmissionResonanceGraftBoundaryIDForCandidateStore != resonanceGraftBoundary.AdmissionResonanceGraftBoundaryID ||
			resonanceGraftCandidateStore.SourceAdmissionResonanceObservationIDForCandidateStore != resonanceObservation.AdmissionResonanceObservationID ||
			resonanceGraftCandidateStore.SourceAdmissionResonanceReceiverIDForCandidateStore != resonanceReceiver.AdmissionResonanceReceiverID ||
			resonanceGraftCandidateStore.SourceAdmissionResonanceIntentIDForCandidateStore != resonanceIntent.AdmissionResonanceIntentID ||
			resonanceGraftCandidateStore.SourceAdmissionFinalGateIDForCandidateStore != finalGate.AdmissionFinalGateID ||
			resonanceGraftCandidateStore.SourceAdmissionSealIDForCandidateStore != seal.AdmissionSealID ||
			resonanceGraftCandidateStore.SourceAdmissionPermitIDForCandidateStore != permit.AdmissionPermitID ||
			resonanceGraftCandidateStore.SourceAdmissionReadinessIDForCandidateStore != readiness.AdmissionReadinessID ||
			resonanceGraftCandidateStore.SourceLedgerVerificationIDForCandidateStore != ledgerVerification.LedgerVerificationID ||
			resonanceGraftCandidateStore.SourceLedgerPersistenceIDForCandidateStore != ledgerPersistence.LedgerPersistenceID ||
			resonanceGraftCandidateStore.SourceLedgerImplementationIDForCandidateStore != ledgerImpl.LedgerImplementationID ||
			resonanceGraftCandidateStore.SourceAdmissionLedgerIDForCandidateStore != ledger.AdmissionLedgerID ||
			resonanceGraftCandidateStore.SourceRollbackImplementationIDForCandidateStore != rollbackImpl.RollbackImplementationID ||
			resonanceGraftCandidateStore.SourceWriterReceiptIDForCandidateStore != writerReceipt.WriterReceiptID ||
			resonanceGraftCandidateStore.AdmissionResonanceGraftCandidateStoreID == "" ||
			resonanceGraftCandidateStore.AdmissionResonanceGraftCandidateStoreID != admissionLiveRouteTurnCandidateAdmissionResonanceGraftCandidateStoreID(resonanceGraftCandidateStore) ||
			resonanceGraftCandidateStore.AdmissionResonanceGraftCandidateID != resonanceGraftCandidate.AdmissionResonanceGraftCandidateID ||
			resonanceGraftCandidateStore.AdmissionResonanceGraftGateID != resonanceGraftGate.AdmissionResonanceGraftGateID ||
			resonanceGraftCandidateStore.AdmissionResonanceGraftPreflightID != resonanceGraftPreflight.AdmissionResonanceGraftPreflightID ||
			resonanceGraftCandidateStore.AdmissionResonanceGraftBoundaryID != resonanceGraftBoundary.AdmissionResonanceGraftBoundaryID ||
			resonanceGraftCandidateStore.AdmissionResonanceObservationID != resonanceObservation.AdmissionResonanceObservationID ||
			resonanceGraftCandidateStore.AdmissionResonanceReceiverID != resonanceReceiver.AdmissionResonanceReceiverID ||
			resonanceGraftCandidateStore.AdmissionResonanceIntentID != resonanceIntent.AdmissionResonanceIntentID ||
			resonanceGraftCandidateStore.AdmissionFinalGateID != finalGate.AdmissionFinalGateID ||
			resonanceGraftCandidateStore.AdmissionSealID != seal.AdmissionSealID ||
			resonanceGraftCandidateStore.AdmissionPermitID != permit.AdmissionPermitID ||
			resonanceGraftCandidateStore.AdmissionReadinessID != readiness.AdmissionReadinessID ||
			resonanceGraftCandidateStore.LedgerVerificationID != ledgerVerification.LedgerVerificationID ||
			resonanceGraftCandidateStore.LedgerPersistenceID != ledgerPersistence.LedgerPersistenceID ||
			resonanceGraftCandidateStore.LedgerImplementationID != ledgerImpl.LedgerImplementationID ||
			resonanceGraftCandidateStore.RollbackImplementationID != rollbackImpl.RollbackImplementationID ||
			resonanceGraftCandidateStore.WriterReceiptID != writerReceipt.WriterReceiptID ||
			resonanceGraftCandidateStore.AdmissionLedgerID != ledger.AdmissionLedgerID ||
			resonanceGraftCandidateStore.AdmissionWriterContractID != writerContract.WriterContractID ||
			resonanceGraftCandidateStore.AdmissionWriterInventoryID != writerInventory.WriterInventoryID ||
			resonanceGraftCandidateStore.AdmissionWriterPreflightID != writerPreflight.WriterPreflightID ||
			resonanceGraftCandidateStore.AdmissionLiveStageID != liveStage.LiveStageID ||
			resonanceGraftCandidateStore.AdmissionEnableGateID != gate.EnableGateID ||
			resonanceGraftCandidateStore.AdmissionSwitchID != sw.SwitchID ||
			resonanceGraftCandidateStore.AdmissionPromotionID != promotion.PromotionID ||
			resonanceGraftCandidateStore.AdmissionDecisionID != decision.DecisionID ||
			resonanceGraftCandidateStore.CandidateExecutionID != execution.ExecutionID ||
			resonanceGraftCandidateStore.GeneratorAdapterID != generatorAdapter.AdapterID ||
			resonanceGraftCandidateStore.CandidateDraftID != draft.DraftID ||
			resonanceGraftCandidateStore.HandoffID != admission.HandoffID ||
			resonanceGraftCandidateStore.AdmissionAdapterID != admissionAdapter.AdmissionAdapterID ||
			resonanceGraftCandidateStore.DreamCandidateRunID != candidate.RunID ||
			resonanceGraftCandidateStore.CandidateRunID != draft.CandidateRunID ||
			resonanceGraftCandidateStore.CandidateTextHash != execution.GeneratedTextHash ||
			resonanceGraftCandidateStore.TurnTextHash != execution.TurnTextHash ||
			resonanceGraftCandidateStore.Reason != "resonance shadow graft candidate stored and read back without body mutation" {
			return fmt.Errorf("bad nano-direct admission resonance graft candidate store receipt: resonance_graft_candidate_store=%+v resonance_graft_candidate=%+v resonance_graft_gate=%+v resonance_graft_preflight=%+v resonance_graft_boundary=%+v resonance_observation=%+v resonance_receiver=%+v resonance_intent=%+v final_gate=%+v seal=%+v permit=%+v readiness=%+v ledger_verification=%+v ledger_persistence=%+v ledger_implementation=%+v rollback_implementation=%+v writer_receipt=%+v writer_implementation=%+v ledger=%+v writer_contract=%+v writer_inventory=%+v writer_preflight=%+v stage=%+v gate=%+v switch=%+v promotion=%+v decision=%+v execution=%+v", resonanceGraftCandidateStore, resonanceGraftCandidate, resonanceGraftGate, resonanceGraftPreflight, resonanceGraftBoundary, resonanceObservation, resonanceReceiver, resonanceIntent, finalGate, seal, permit, readiness, ledgerVerification, ledgerPersistence, ledgerImpl, rollbackImpl, writerReceipt, writerImpl, ledger, writerContract, writerInventory, writerPreflight, liveStage, gate, sw, promotion, decision, execution)
		}
	}
	if admissionLiveRouteTurnCandidateAdmissionResonanceGraftCandidateStoreReaderDryRun() {
		if resonanceGraftCandidateStoreReader.Schema != admissionLiveRouteTurnCandidateAdmissionResonanceGraftCandidateStoreReaderSchema ||
			resonanceGraftCandidateStoreReader.Timing != "live_admission_resonance_graft_candidate_store_reader" ||
			!resonanceGraftCandidateStoreReader.Passed ||
			!resonanceGraftCandidateStoreReader.LiveReady ||
			resonanceGraftCandidateStoreReader.LiveAdmissionEnabled ||
			resonanceGraftCandidateStoreReader.AdmissionAllowed ||
			resonanceGraftCandidateStoreReader.ContractsReady ||
			resonanceGraftCandidateStoreReader.WriteAllowed ||
			resonanceGraftCandidateStoreReader.MutatesState ||
			resonanceGraftCandidateStoreReader.BodyTarget != "none" ||
			resonanceGraftCandidateStoreReader.AdmissionResonanceGraftCandidateStoreReaderState != "shadow_graft_candidate_store_read_back_dry_run" ||
			resonanceGraftCandidateStoreReader.AdmissionResonanceGraftCandidateStoreReaderAction != "read_resonance_shadow_graft_candidate_store_dry_run" ||
			resonanceGraftCandidateStoreReader.AdmissionResonanceGraftCandidateStoreReaderTarget != "resonance" ||
			resonanceGraftCandidateStoreReader.AdmissionResonanceGraftCandidateStoreReaderTargetKind != "internal_world_shadow_graft_candidate_store_reader" ||
			resonanceGraftCandidateStoreReader.AdmissionResonanceGraftCandidateStoreReaderTargetMode != "read_only_replay_dry_run" ||
			resonanceGraftCandidateStoreReader.AdmissionResonanceGraftCandidateStoreReaderReceiptShape != "resonance_shadow_graft_candidate_store_reader_receipt" ||
			!resonanceGraftCandidateStoreReader.AdmissionResonanceGraftCandidateStoreReaderDryRunOnly ||
			!resonanceGraftCandidateStoreReader.AdmissionResonanceGraftCandidateStoreReaderStoreVerified ||
			!resonanceGraftCandidateStoreReader.AdmissionResonanceGraftCandidateStoreReaderCandidateVerified ||
			!resonanceGraftCandidateStoreReader.AdmissionResonanceGraftCandidateStoreReaderLedgerVerified ||
			!resonanceGraftCandidateStoreReader.AdmissionResonanceGraftCandidateStoreReaderReadBackVerified ||
			!resonanceGraftCandidateStoreReader.AdmissionResonanceGraftCandidateStoreReaderHashVerified ||
			resonanceGraftCandidateStoreReader.AdmissionResonanceGraftCandidateStoreReaderKind != "shadow_graft_candidate_store_reader" ||
			resonanceGraftCandidateStoreReader.AdmissionResonanceGraftCandidateStoreReaderMode != "read_only_replay" ||
			resonanceGraftCandidateStoreReader.AdmissionResonanceGraftCandidateStoreReaderStage != "pre_live_graft_candidate_store_reader" ||
			resonanceGraftCandidateStoreReader.AdmissionResonanceGraftCandidateStoreReaderCausalID == "" ||
			resonanceGraftCandidateStoreReader.AdmissionResonanceGraftCandidateStoreReaderCausalID != admissionLiveRouteTurnCandidateAdmissionResonanceGraftCandidateStoreReaderCausalID(resonanceGraftCandidateStoreReader) ||
			resonanceGraftCandidateStoreReader.AdmissionResonanceGraftCandidateStoreReaderHash == "" ||
			resonanceGraftCandidateStoreReader.AdmissionResonanceGraftCandidateStoreReaderHash != admissionLiveRouteTurnCandidateAdmissionResonanceGraftCandidateStoreReaderHash(resonanceGraftCandidateStoreReader) ||
			resonanceGraftCandidateStoreReader.AdmissionResonanceGraftCandidateStoreReaderReplayHash == "" ||
			resonanceGraftCandidateStoreReader.AdmissionResonanceGraftCandidateStoreReaderReplayHash != admissionLiveRouteTurnCandidateAdmissionResonanceGraftCandidateStoreReaderReplayHash(resonanceGraftCandidateStoreReader) ||
			resonanceGraftCandidateStoreReader.AdmissionResonanceGraftCandidateStoreReaderReadBackHash == "" ||
			resonanceGraftCandidateStoreReader.AdmissionResonanceGraftCandidateStoreReaderReadBackHash != admissionLiveRouteTurnCandidateAdmissionResonanceGraftCandidateStoreReaderReadBackHash(resonanceGraftCandidateStoreReader) ||
			resonanceGraftCandidateStoreReader.AdmissionResonanceGraftCandidateStoreReaderHash == resonanceGraftCandidateStoreReader.AdmissionResonanceGraftCandidateStoreReaderReadBackHash ||
			!resonanceGraftCandidateStoreReader.AdmissionResonanceGraftCandidateStoreReaderReadOnly ||
			!resonanceGraftCandidateStoreReader.AdmissionResonanceGraftCandidateStoreReaderReplayOnly ||
			!resonanceGraftCandidateStoreReader.AdmissionResonanceGraftCandidateStoreReaderSourceAppendOnly ||
			!resonanceGraftCandidateStoreReader.AdmissionResonanceGraftCandidateStoreReaderSourceReadBack ||
			!resonanceGraftCandidateStoreReader.AdmissionResonanceGraftCandidateStoreReaderSourceReceiptVerified ||
			resonanceGraftCandidateStoreReader.AdmissionResonanceGraftCandidateStoreReaderGraftAllowed ||
			resonanceGraftCandidateStoreReader.AdmissionResonanceGraftCandidateStoreReaderRawDreamTextAllowed ||
			resonanceGraftCandidateStoreReader.AdmissionResonanceGraftCandidateStoreReaderJanusSurfaceAllowed ||
			resonanceGraftCandidateStoreReader.AdmissionResonanceGraftCandidateStoreReaderCoocLearningAllowed ||
			resonanceGraftCandidateStoreReader.AdmissionResonanceGraftCandidateStoreReaderDeltaHarvestAllowed ||
			resonanceGraftCandidateStoreReader.AdmissionResonanceGraftCandidateStoreReaderBodyMutationAllowed ||
			!resonanceGraftCandidateStoreReader.AdmissionResonanceGraftCandidateStoreReaderReady ||
			resonanceGraftCandidateStoreReader.SourceAdmissionResonanceGraftCandidateStoreSchema != admissionLiveRouteTurnCandidateAdmissionResonanceGraftCandidateStoreSchema ||
			!resonanceGraftCandidateStoreReader.SourceAdmissionResonanceGraftCandidateStorePassed ||
			resonanceGraftCandidateStoreReader.SourceAdmissionResonanceGraftCandidateStoreID != resonanceGraftCandidateStore.AdmissionResonanceGraftCandidateStoreID ||
			resonanceGraftCandidateStoreReader.SourceAdmissionResonanceGraftCandidateStoreAction != "store_resonance_shadow_graft_candidate_dry_run" ||
			!resonanceGraftCandidateStoreReader.SourceAdmissionResonanceGraftCandidateStoreReady ||
			resonanceGraftCandidateStoreReader.SourceAdmissionResonanceGraftCandidateStoreCausalID != resonanceGraftCandidateStore.AdmissionResonanceGraftCandidateStoreCausalID ||
			resonanceGraftCandidateStoreReader.SourceAdmissionResonanceGraftCandidateStoreHash != resonanceGraftCandidateStore.AdmissionResonanceGraftCandidateStoreHash ||
			resonanceGraftCandidateStoreReader.SourceAdmissionResonanceGraftCandidateStoreReadBackHash != resonanceGraftCandidateStore.AdmissionResonanceGraftCandidateStoreReadBackHash ||
			resonanceGraftCandidateStoreReader.SourceAdmissionResonanceGraftCandidateIDForStoreReader != resonanceGraftCandidateStore.AdmissionResonanceGraftCandidateID ||
			resonanceGraftCandidateStoreReader.SourceAdmissionResonanceGraftGateIDForStoreReader != resonanceGraftCandidateStore.AdmissionResonanceGraftGateID ||
			resonanceGraftCandidateStoreReader.SourceAdmissionResonanceObservationIDForStoreReader != resonanceGraftCandidateStore.AdmissionResonanceObservationID ||
			resonanceGraftCandidateStoreReader.SourceAdmissionFinalGateIDForStoreReader != resonanceGraftCandidateStore.AdmissionFinalGateID ||
			resonanceGraftCandidateStoreReader.SourceLedgerVerificationIDForStoreReader != resonanceGraftCandidateStore.LedgerVerificationID ||
			resonanceGraftCandidateStoreReader.AdmissionResonanceGraftCandidateStoreReaderID == "" ||
			resonanceGraftCandidateStoreReader.AdmissionResonanceGraftCandidateStoreReaderID != admissionLiveRouteTurnCandidateAdmissionResonanceGraftCandidateStoreReaderID(resonanceGraftCandidateStoreReader) ||
			resonanceGraftCandidateStoreReader.AdmissionResonanceGraftCandidateStoreID != resonanceGraftCandidateStore.AdmissionResonanceGraftCandidateStoreID ||
			resonanceGraftCandidateStoreReader.AdmissionResonanceGraftCandidateID != resonanceGraftCandidate.AdmissionResonanceGraftCandidateID ||
			resonanceGraftCandidateStoreReader.AdmissionResonanceGraftGateID != resonanceGraftGate.AdmissionResonanceGraftGateID ||
			resonanceGraftCandidateStoreReader.AdmissionResonanceGraftPreflightID != resonanceGraftPreflight.AdmissionResonanceGraftPreflightID ||
			resonanceGraftCandidateStoreReader.AdmissionResonanceGraftBoundaryID != resonanceGraftBoundary.AdmissionResonanceGraftBoundaryID ||
			resonanceGraftCandidateStoreReader.AdmissionResonanceObservationID != resonanceObservation.AdmissionResonanceObservationID ||
			resonanceGraftCandidateStoreReader.AdmissionResonanceReceiverID != resonanceReceiver.AdmissionResonanceReceiverID ||
			resonanceGraftCandidateStoreReader.AdmissionResonanceIntentID != resonanceIntent.AdmissionResonanceIntentID ||
			resonanceGraftCandidateStoreReader.AdmissionFinalGateID != finalGate.AdmissionFinalGateID ||
			resonanceGraftCandidateStoreReader.AdmissionSealID != seal.AdmissionSealID ||
			resonanceGraftCandidateStoreReader.AdmissionPermitID != permit.AdmissionPermitID ||
			resonanceGraftCandidateStoreReader.AdmissionReadinessID != readiness.AdmissionReadinessID ||
			resonanceGraftCandidateStoreReader.LedgerVerificationID != ledgerVerification.LedgerVerificationID ||
			resonanceGraftCandidateStoreReader.LedgerPersistenceID != ledgerPersistence.LedgerPersistenceID ||
			resonanceGraftCandidateStoreReader.LedgerImplementationID != ledgerImpl.LedgerImplementationID ||
			resonanceGraftCandidateStoreReader.RollbackImplementationID != rollbackImpl.RollbackImplementationID ||
			resonanceGraftCandidateStoreReader.WriterReceiptID != writerReceipt.WriterReceiptID ||
			resonanceGraftCandidateStoreReader.AdmissionLedgerID != ledger.AdmissionLedgerID ||
			resonanceGraftCandidateStoreReader.AdmissionWriterContractID != writerContract.WriterContractID ||
			resonanceGraftCandidateStoreReader.AdmissionWriterInventoryID != writerInventory.WriterInventoryID ||
			resonanceGraftCandidateStoreReader.AdmissionWriterPreflightID != writerPreflight.WriterPreflightID ||
			resonanceGraftCandidateStoreReader.AdmissionLiveStageID != liveStage.LiveStageID ||
			resonanceGraftCandidateStoreReader.AdmissionEnableGateID != gate.EnableGateID ||
			resonanceGraftCandidateStoreReader.AdmissionSwitchID != sw.SwitchID ||
			resonanceGraftCandidateStoreReader.AdmissionPromotionID != promotion.PromotionID ||
			resonanceGraftCandidateStoreReader.AdmissionDecisionID != decision.DecisionID ||
			resonanceGraftCandidateStoreReader.CandidateExecutionID != execution.ExecutionID ||
			resonanceGraftCandidateStoreReader.GeneratorAdapterID != generatorAdapter.AdapterID ||
			resonanceGraftCandidateStoreReader.CandidateDraftID != draft.DraftID ||
			resonanceGraftCandidateStoreReader.HandoffID != admission.HandoffID ||
			resonanceGraftCandidateStoreReader.AdmissionAdapterID != admissionAdapter.AdmissionAdapterID ||
			resonanceGraftCandidateStoreReader.DreamCandidateRunID != candidate.RunID ||
			resonanceGraftCandidateStoreReader.CandidateRunID != draft.CandidateRunID ||
			resonanceGraftCandidateStoreReader.CandidateTextHash != execution.GeneratedTextHash ||
			resonanceGraftCandidateStoreReader.TurnTextHash != execution.TurnTextHash ||
			resonanceGraftCandidateStoreReader.Reason != "resonance shadow graft candidate store read back without opening body" {
			return fmt.Errorf("bad nano-direct admission resonance graft candidate store reader receipt: resonance_graft_candidate_store_reader=%+v resonance_graft_candidate_store=%+v resonance_graft_candidate=%+v resonance_graft_gate=%+v resonance_graft_preflight=%+v resonance_graft_boundary=%+v resonance_observation=%+v resonance_receiver=%+v resonance_intent=%+v final_gate=%+v seal=%+v permit=%+v readiness=%+v ledger_verification=%+v ledger_persistence=%+v ledger_implementation=%+v rollback_implementation=%+v writer_receipt=%+v writer_implementation=%+v ledger=%+v writer_contract=%+v writer_inventory=%+v writer_preflight=%+v stage=%+v gate=%+v switch=%+v promotion=%+v decision=%+v execution=%+v", resonanceGraftCandidateStoreReader, resonanceGraftCandidateStore, resonanceGraftCandidate, resonanceGraftGate, resonanceGraftPreflight, resonanceGraftBoundary, resonanceObservation, resonanceReceiver, resonanceIntent, finalGate, seal, permit, readiness, ledgerVerification, ledgerPersistence, ledgerImpl, rollbackImpl, writerReceipt, writerImpl, ledger, writerContract, writerInventory, writerPreflight, liveStage, gate, sw, promotion, decision, execution)
		}
	}
	if admissionLiveRouteTurnCandidateAdmissionResonanceGraftAdmissionProofDryRun() {
		if resonanceGraftAdmissionProof.Schema != admissionLiveRouteTurnCandidateAdmissionResonanceGraftAdmissionProofSchema ||
			resonanceGraftAdmissionProof.Timing != "live_admission_resonance_graft_admission_proof" ||
			!resonanceGraftAdmissionProof.Passed ||
			!resonanceGraftAdmissionProof.LiveReady ||
			resonanceGraftAdmissionProof.LiveAdmissionEnabled ||
			resonanceGraftAdmissionProof.AdmissionAllowed ||
			resonanceGraftAdmissionProof.ContractsReady ||
			resonanceGraftAdmissionProof.WriteAllowed ||
			resonanceGraftAdmissionProof.MutatesState ||
			resonanceGraftAdmissionProof.BodyTarget != "none" ||
			resonanceGraftAdmissionProof.AdmissionResonanceGraftAdmissionProofState != "shadow_graft_admission_proved_dry_run" ||
			resonanceGraftAdmissionProof.AdmissionResonanceGraftAdmissionProofAction != "prove_resonance_shadow_graft_admission_dry_run" ||
			resonanceGraftAdmissionProof.AdmissionResonanceGraftAdmissionProofTarget != "resonance" ||
			resonanceGraftAdmissionProof.AdmissionResonanceGraftAdmissionProofTargetKind != "internal_world_shadow_graft_admission_proof" ||
			resonanceGraftAdmissionProof.AdmissionResonanceGraftAdmissionProofTargetMode != "verified_replay_closed_dry_run" ||
			resonanceGraftAdmissionProof.AdmissionResonanceGraftAdmissionProofReceiptShape != "resonance_shadow_graft_admission_proof_receipt" ||
			!resonanceGraftAdmissionProof.AdmissionResonanceGraftAdmissionProofDryRunOnly ||
			!resonanceGraftAdmissionProof.AdmissionResonanceGraftAdmissionProofReaderVerified ||
			!resonanceGraftAdmissionProof.AdmissionResonanceGraftAdmissionProofStoreVerified ||
			!resonanceGraftAdmissionProof.AdmissionResonanceGraftAdmissionProofCandidateVerified ||
			!resonanceGraftAdmissionProof.AdmissionResonanceGraftAdmissionProofLedgerVerified ||
			!resonanceGraftAdmissionProof.AdmissionResonanceGraftAdmissionProofReplayVerified ||
			!resonanceGraftAdmissionProof.AdmissionResonanceGraftAdmissionProofReadBackVerified ||
			!resonanceGraftAdmissionProof.AdmissionResonanceGraftAdmissionProofHashVerified ||
			resonanceGraftAdmissionProof.AdmissionResonanceGraftAdmissionProofKind != "shadow_graft_admission_proof" ||
			resonanceGraftAdmissionProof.AdmissionResonanceGraftAdmissionProofMode != "verified_replay_closed" ||
			resonanceGraftAdmissionProof.AdmissionResonanceGraftAdmissionProofStage != "pre_live_graft_admission_proof" ||
			resonanceGraftAdmissionProof.AdmissionResonanceGraftAdmissionProofCausalID == "" ||
			resonanceGraftAdmissionProof.AdmissionResonanceGraftAdmissionProofCausalID != admissionLiveRouteTurnCandidateAdmissionResonanceGraftAdmissionProofCausalID(resonanceGraftAdmissionProof) ||
			resonanceGraftAdmissionProof.AdmissionResonanceGraftAdmissionProofHash == "" ||
			resonanceGraftAdmissionProof.AdmissionResonanceGraftAdmissionProofHash != admissionLiveRouteTurnCandidateAdmissionResonanceGraftAdmissionProofHash(resonanceGraftAdmissionProof) ||
			resonanceGraftAdmissionProof.AdmissionResonanceGraftAdmissionProofReplayHash == "" ||
			resonanceGraftAdmissionProof.AdmissionResonanceGraftAdmissionProofReplayHash != admissionLiveRouteTurnCandidateAdmissionResonanceGraftAdmissionProofReplayHash(resonanceGraftAdmissionProof) ||
			resonanceGraftAdmissionProof.AdmissionResonanceGraftAdmissionProofReadBackHash == "" ||
			resonanceGraftAdmissionProof.AdmissionResonanceGraftAdmissionProofReadBackHash != admissionLiveRouteTurnCandidateAdmissionResonanceGraftAdmissionProofReadBackHash(resonanceGraftAdmissionProof) ||
			resonanceGraftAdmissionProof.AdmissionResonanceGraftAdmissionProofHash == resonanceGraftAdmissionProof.AdmissionResonanceGraftAdmissionProofReadBackHash ||
			!resonanceGraftAdmissionProof.AdmissionResonanceGraftAdmissionProofAdmissionRequired ||
			!resonanceGraftAdmissionProof.AdmissionResonanceGraftAdmissionProofShadowOnly ||
			resonanceGraftAdmissionProof.AdmissionResonanceGraftAdmissionProofGraftAllowed ||
			resonanceGraftAdmissionProof.AdmissionResonanceGraftAdmissionProofRawDreamTextAllowed ||
			resonanceGraftAdmissionProof.AdmissionResonanceGraftAdmissionProofJanusSurfaceAllowed ||
			resonanceGraftAdmissionProof.AdmissionResonanceGraftAdmissionProofCoocLearningAllowed ||
			resonanceGraftAdmissionProof.AdmissionResonanceGraftAdmissionProofDeltaHarvestAllowed ||
			resonanceGraftAdmissionProof.AdmissionResonanceGraftAdmissionProofBodyMutationAllowed ||
			!resonanceGraftAdmissionProof.AdmissionResonanceGraftAdmissionProofRollbackRequired ||
			!resonanceGraftAdmissionProof.AdmissionResonanceGraftAdmissionProofReady ||
			resonanceGraftAdmissionProof.SourceAdmissionResonanceGraftCandidateStoreReaderSchema != admissionLiveRouteTurnCandidateAdmissionResonanceGraftCandidateStoreReaderSchema ||
			!resonanceGraftAdmissionProof.SourceAdmissionResonanceGraftCandidateStoreReaderPassed ||
			resonanceGraftAdmissionProof.SourceAdmissionResonanceGraftCandidateStoreReaderID != resonanceGraftCandidateStoreReader.AdmissionResonanceGraftCandidateStoreReaderID ||
			resonanceGraftAdmissionProof.SourceAdmissionResonanceGraftCandidateStoreReaderAction != "read_resonance_shadow_graft_candidate_store_dry_run" ||
			!resonanceGraftAdmissionProof.SourceAdmissionResonanceGraftCandidateStoreReaderReady ||
			resonanceGraftAdmissionProof.SourceAdmissionResonanceGraftCandidateStoreReaderCausalID != resonanceGraftCandidateStoreReader.AdmissionResonanceGraftCandidateStoreReaderCausalID ||
			resonanceGraftAdmissionProof.SourceAdmissionResonanceGraftCandidateStoreReaderHash != resonanceGraftCandidateStoreReader.AdmissionResonanceGraftCandidateStoreReaderHash ||
			resonanceGraftAdmissionProof.SourceAdmissionResonanceGraftCandidateStoreReaderReplayHash != resonanceGraftCandidateStoreReader.AdmissionResonanceGraftCandidateStoreReaderReplayHash ||
			resonanceGraftAdmissionProof.SourceAdmissionResonanceGraftCandidateStoreReaderReadBackHash != resonanceGraftCandidateStoreReader.AdmissionResonanceGraftCandidateStoreReaderReadBackHash ||
			resonanceGraftAdmissionProof.SourceAdmissionResonanceGraftCandidateStoreIDForProof != resonanceGraftCandidateStoreReader.AdmissionResonanceGraftCandidateStoreID ||
			resonanceGraftAdmissionProof.SourceAdmissionResonanceGraftCandidateIDForProof != resonanceGraftCandidateStoreReader.AdmissionResonanceGraftCandidateID ||
			resonanceGraftAdmissionProof.SourceAdmissionResonanceGraftGateIDForProof != resonanceGraftCandidateStoreReader.AdmissionResonanceGraftGateID ||
			resonanceGraftAdmissionProof.SourceAdmissionResonanceObservationIDForProof != resonanceGraftCandidateStoreReader.AdmissionResonanceObservationID ||
			resonanceGraftAdmissionProof.SourceAdmissionFinalGateIDForProof != resonanceGraftCandidateStoreReader.AdmissionFinalGateID ||
			resonanceGraftAdmissionProof.SourceLedgerVerificationIDForProof != resonanceGraftCandidateStoreReader.LedgerVerificationID ||
			resonanceGraftAdmissionProof.AdmissionResonanceGraftAdmissionProofID == "" ||
			resonanceGraftAdmissionProof.AdmissionResonanceGraftAdmissionProofID != admissionLiveRouteTurnCandidateAdmissionResonanceGraftAdmissionProofID(resonanceGraftAdmissionProof) ||
			resonanceGraftAdmissionProof.AdmissionResonanceGraftCandidateStoreReaderID != resonanceGraftCandidateStoreReader.AdmissionResonanceGraftCandidateStoreReaderID ||
			resonanceGraftAdmissionProof.AdmissionResonanceGraftCandidateStoreID != resonanceGraftCandidateStore.AdmissionResonanceGraftCandidateStoreID ||
			resonanceGraftAdmissionProof.AdmissionResonanceGraftCandidateID != resonanceGraftCandidate.AdmissionResonanceGraftCandidateID ||
			resonanceGraftAdmissionProof.AdmissionResonanceGraftGateID != resonanceGraftGate.AdmissionResonanceGraftGateID ||
			resonanceGraftAdmissionProof.AdmissionResonanceGraftPreflightID != resonanceGraftPreflight.AdmissionResonanceGraftPreflightID ||
			resonanceGraftAdmissionProof.AdmissionResonanceGraftBoundaryID != resonanceGraftBoundary.AdmissionResonanceGraftBoundaryID ||
			resonanceGraftAdmissionProof.AdmissionResonanceObservationID != resonanceObservation.AdmissionResonanceObservationID ||
			resonanceGraftAdmissionProof.AdmissionResonanceReceiverID != resonanceReceiver.AdmissionResonanceReceiverID ||
			resonanceGraftAdmissionProof.AdmissionResonanceIntentID != resonanceIntent.AdmissionResonanceIntentID ||
			resonanceGraftAdmissionProof.AdmissionFinalGateID != finalGate.AdmissionFinalGateID ||
			resonanceGraftAdmissionProof.AdmissionSealID != seal.AdmissionSealID ||
			resonanceGraftAdmissionProof.AdmissionPermitID != permit.AdmissionPermitID ||
			resonanceGraftAdmissionProof.AdmissionReadinessID != readiness.AdmissionReadinessID ||
			resonanceGraftAdmissionProof.LedgerVerificationID != ledgerVerification.LedgerVerificationID ||
			resonanceGraftAdmissionProof.LedgerPersistenceID != ledgerPersistence.LedgerPersistenceID ||
			resonanceGraftAdmissionProof.LedgerImplementationID != ledgerImpl.LedgerImplementationID ||
			resonanceGraftAdmissionProof.RollbackImplementationID != rollbackImpl.RollbackImplementationID ||
			resonanceGraftAdmissionProof.WriterReceiptID != writerReceipt.WriterReceiptID ||
			resonanceGraftAdmissionProof.AdmissionLedgerID != ledger.AdmissionLedgerID ||
			resonanceGraftAdmissionProof.AdmissionWriterContractID != writerContract.WriterContractID ||
			resonanceGraftAdmissionProof.AdmissionWriterInventoryID != writerInventory.WriterInventoryID ||
			resonanceGraftAdmissionProof.AdmissionWriterPreflightID != writerPreflight.WriterPreflightID ||
			resonanceGraftAdmissionProof.AdmissionLiveStageID != liveStage.LiveStageID ||
			resonanceGraftAdmissionProof.AdmissionEnableGateID != gate.EnableGateID ||
			resonanceGraftAdmissionProof.AdmissionSwitchID != sw.SwitchID ||
			resonanceGraftAdmissionProof.AdmissionPromotionID != promotion.PromotionID ||
			resonanceGraftAdmissionProof.AdmissionDecisionID != decision.DecisionID ||
			resonanceGraftAdmissionProof.CandidateExecutionID != execution.ExecutionID ||
			resonanceGraftAdmissionProof.GeneratorAdapterID != generatorAdapter.AdapterID ||
			resonanceGraftAdmissionProof.CandidateDraftID != draft.DraftID ||
			resonanceGraftAdmissionProof.HandoffID != admission.HandoffID ||
			resonanceGraftAdmissionProof.AdmissionAdapterID != admissionAdapter.AdmissionAdapterID ||
			resonanceGraftAdmissionProof.DreamCandidateRunID != candidate.RunID ||
			resonanceGraftAdmissionProof.CandidateRunID != draft.CandidateRunID ||
			resonanceGraftAdmissionProof.CandidateTextHash != execution.GeneratedTextHash ||
			resonanceGraftAdmissionProof.TurnTextHash != execution.TurnTextHash ||
			resonanceGraftAdmissionProof.Reason != "resonance shadow graft admission proved from read-back store without opening body" {
			return fmt.Errorf("bad nano-direct admission resonance graft admission proof receipt: resonance_graft_admission_proof=%+v resonance_graft_candidate_store_reader=%+v resonance_graft_candidate_store=%+v resonance_graft_candidate=%+v resonance_graft_gate=%+v resonance_graft_preflight=%+v resonance_graft_boundary=%+v resonance_observation=%+v resonance_receiver=%+v resonance_intent=%+v final_gate=%+v seal=%+v permit=%+v readiness=%+v ledger_verification=%+v ledger_persistence=%+v ledger_implementation=%+v rollback_implementation=%+v writer_receipt=%+v writer_implementation=%+v ledger=%+v writer_contract=%+v writer_inventory=%+v writer_preflight=%+v stage=%+v gate=%+v switch=%+v promotion=%+v decision=%+v execution=%+v", resonanceGraftAdmissionProof, resonanceGraftCandidateStoreReader, resonanceGraftCandidateStore, resonanceGraftCandidate, resonanceGraftGate, resonanceGraftPreflight, resonanceGraftBoundary, resonanceObservation, resonanceReceiver, resonanceIntent, finalGate, seal, permit, readiness, ledgerVerification, ledgerPersistence, ledgerImpl, rollbackImpl, writerReceipt, writerImpl, ledger, writerContract, writerInventory, writerPreflight, liveStage, gate, sw, promotion, decision, execution)
		}
	}

	if admissionLiveRouteTurnCandidateAdmissionResonanceGraftAdmissionProofDryRun() {
		fmt.Printf("[admission-live-route-turn-candidate-nano-direct-chat-shadow-smoke] pass: execution=%s adapter=%s drafts=%s reviews=%s handoffs=%s admission_adapters=%s admission=%s decision=%s promotion=%s switch=%s enable_gate=%s live_stage=%s writer_preflight=%s writer_inventory=%s writer_contract=%s ledger=%s writer_implementation=%s writer_receipt=%s rollback_implementation=%s ledger_implementation=%s ledger_persistence=%s ledger_verification=%s readiness=%s permit=%s seal=%s final_gate=%s resonance_intent=%s resonance_receiver=%s resonance_observation=%s resonance_graft_boundary=%s resonance_graft_preflight=%s resonance_graft_gate=%s resonance_graft_candidate=%s resonance_graft_candidate_store=%s resonance_graft_candidate_store_reader=%s resonance_graft_admission_proof=%s\n",
			executionLogPath, adapterLogPath, draftLogPath, reviewLogPath, admissionLogPath, admissionAdapterLogPath, dreamLogPath, decisionLogPath, promotionLogPath, switchLogPath, enableGateLogPath, liveStageLogPath, writerPreflightLogPath, writerInventoryLogPath, writerContractLogPath, ledgerLogPath, writerImplLogPath, writerReceiptLogPath, rollbackImplLogPath, ledgerImplLogPath, ledgerPersistenceLogPath, ledgerVerificationLogPath, readinessLogPath, permitLogPath, sealLogPath, finalGateLogPath, resonanceIntentLogPath, resonanceReceiverLogPath, resonanceObservationLogPath, resonanceGraftBoundaryLogPath, resonanceGraftPreflightLogPath, resonanceGraftGateLogPath, resonanceGraftCandidateLogPath, resonanceGraftCandidateStoreLogPath, resonanceGraftCandidateStoreReaderLogPath, resonanceGraftAdmissionProofLogPath)
	} else if admissionLiveRouteTurnCandidateAdmissionResonanceGraftCandidateStoreReaderDryRun() {
		fmt.Printf("[admission-live-route-turn-candidate-nano-direct-chat-shadow-smoke] pass: execution=%s adapter=%s drafts=%s reviews=%s handoffs=%s admission_adapters=%s admission=%s decision=%s promotion=%s switch=%s enable_gate=%s live_stage=%s writer_preflight=%s writer_inventory=%s writer_contract=%s ledger=%s writer_implementation=%s writer_receipt=%s rollback_implementation=%s ledger_implementation=%s ledger_persistence=%s ledger_verification=%s readiness=%s permit=%s seal=%s final_gate=%s resonance_intent=%s resonance_receiver=%s resonance_observation=%s resonance_graft_boundary=%s resonance_graft_preflight=%s resonance_graft_gate=%s resonance_graft_candidate=%s resonance_graft_candidate_store=%s resonance_graft_candidate_store_reader=%s\n",
			executionLogPath, adapterLogPath, draftLogPath, reviewLogPath, admissionLogPath, admissionAdapterLogPath, dreamLogPath, decisionLogPath, promotionLogPath, switchLogPath, enableGateLogPath, liveStageLogPath, writerPreflightLogPath, writerInventoryLogPath, writerContractLogPath, ledgerLogPath, writerImplLogPath, writerReceiptLogPath, rollbackImplLogPath, ledgerImplLogPath, ledgerPersistenceLogPath, ledgerVerificationLogPath, readinessLogPath, permitLogPath, sealLogPath, finalGateLogPath, resonanceIntentLogPath, resonanceReceiverLogPath, resonanceObservationLogPath, resonanceGraftBoundaryLogPath, resonanceGraftPreflightLogPath, resonanceGraftGateLogPath, resonanceGraftCandidateLogPath, resonanceGraftCandidateStoreLogPath, resonanceGraftCandidateStoreReaderLogPath)
	} else if admissionLiveRouteTurnCandidateAdmissionResonanceGraftCandidateStoreDryRun() {
		fmt.Printf("[admission-live-route-turn-candidate-nano-direct-chat-shadow-smoke] pass: execution=%s adapter=%s drafts=%s reviews=%s handoffs=%s admission_adapters=%s admission=%s decision=%s promotion=%s switch=%s enable_gate=%s live_stage=%s writer_preflight=%s writer_inventory=%s writer_contract=%s ledger=%s writer_implementation=%s writer_receipt=%s rollback_implementation=%s ledger_implementation=%s ledger_persistence=%s ledger_verification=%s readiness=%s permit=%s seal=%s final_gate=%s resonance_intent=%s resonance_receiver=%s resonance_observation=%s resonance_graft_boundary=%s resonance_graft_preflight=%s resonance_graft_gate=%s resonance_graft_candidate=%s resonance_graft_candidate_store=%s\n",
			executionLogPath, adapterLogPath, draftLogPath, reviewLogPath, admissionLogPath, admissionAdapterLogPath, dreamLogPath, decisionLogPath, promotionLogPath, switchLogPath, enableGateLogPath, liveStageLogPath, writerPreflightLogPath, writerInventoryLogPath, writerContractLogPath, ledgerLogPath, writerImplLogPath, writerReceiptLogPath, rollbackImplLogPath, ledgerImplLogPath, ledgerPersistenceLogPath, ledgerVerificationLogPath, readinessLogPath, permitLogPath, sealLogPath, finalGateLogPath, resonanceIntentLogPath, resonanceReceiverLogPath, resonanceObservationLogPath, resonanceGraftBoundaryLogPath, resonanceGraftPreflightLogPath, resonanceGraftGateLogPath, resonanceGraftCandidateLogPath, resonanceGraftCandidateStoreLogPath)
	} else if admissionLiveRouteTurnCandidateAdmissionResonanceGraftCandidateDryRun() {
		fmt.Printf("[admission-live-route-turn-candidate-nano-direct-chat-shadow-smoke] pass: execution=%s adapter=%s drafts=%s reviews=%s handoffs=%s admission_adapters=%s admission=%s decision=%s promotion=%s switch=%s enable_gate=%s live_stage=%s writer_preflight=%s writer_inventory=%s writer_contract=%s ledger=%s writer_implementation=%s writer_receipt=%s rollback_implementation=%s ledger_implementation=%s ledger_persistence=%s ledger_verification=%s readiness=%s permit=%s seal=%s final_gate=%s resonance_intent=%s resonance_receiver=%s resonance_observation=%s resonance_graft_boundary=%s resonance_graft_preflight=%s resonance_graft_gate=%s resonance_graft_candidate=%s\n",
			executionLogPath, adapterLogPath, draftLogPath, reviewLogPath, admissionLogPath, admissionAdapterLogPath, dreamLogPath, decisionLogPath, promotionLogPath, switchLogPath, enableGateLogPath, liveStageLogPath, writerPreflightLogPath, writerInventoryLogPath, writerContractLogPath, ledgerLogPath, writerImplLogPath, writerReceiptLogPath, rollbackImplLogPath, ledgerImplLogPath, ledgerPersistenceLogPath, ledgerVerificationLogPath, readinessLogPath, permitLogPath, sealLogPath, finalGateLogPath, resonanceIntentLogPath, resonanceReceiverLogPath, resonanceObservationLogPath, resonanceGraftBoundaryLogPath, resonanceGraftPreflightLogPath, resonanceGraftGateLogPath, resonanceGraftCandidateLogPath)
	} else if admissionLiveRouteTurnCandidateAdmissionResonanceGraftGateDryRun() {
		fmt.Printf("[admission-live-route-turn-candidate-nano-direct-chat-shadow-smoke] pass: execution=%s adapter=%s drafts=%s reviews=%s handoffs=%s admission_adapters=%s admission=%s decision=%s promotion=%s switch=%s enable_gate=%s live_stage=%s writer_preflight=%s writer_inventory=%s writer_contract=%s ledger=%s writer_implementation=%s writer_receipt=%s rollback_implementation=%s ledger_implementation=%s ledger_persistence=%s ledger_verification=%s readiness=%s permit=%s seal=%s final_gate=%s resonance_intent=%s resonance_receiver=%s resonance_observation=%s resonance_graft_boundary=%s resonance_graft_preflight=%s resonance_graft_gate=%s\n",
			executionLogPath, adapterLogPath, draftLogPath, reviewLogPath, admissionLogPath, admissionAdapterLogPath, dreamLogPath, decisionLogPath, promotionLogPath, switchLogPath, enableGateLogPath, liveStageLogPath, writerPreflightLogPath, writerInventoryLogPath, writerContractLogPath, ledgerLogPath, writerImplLogPath, writerReceiptLogPath, rollbackImplLogPath, ledgerImplLogPath, ledgerPersistenceLogPath, ledgerVerificationLogPath, readinessLogPath, permitLogPath, sealLogPath, finalGateLogPath, resonanceIntentLogPath, resonanceReceiverLogPath, resonanceObservationLogPath, resonanceGraftBoundaryLogPath, resonanceGraftPreflightLogPath, resonanceGraftGateLogPath)
	} else if admissionLiveRouteTurnCandidateAdmissionResonanceGraftPreflightDryRun() {
		fmt.Printf("[admission-live-route-turn-candidate-nano-direct-chat-shadow-smoke] pass: execution=%s adapter=%s drafts=%s reviews=%s handoffs=%s admission_adapters=%s admission=%s decision=%s promotion=%s switch=%s enable_gate=%s live_stage=%s writer_preflight=%s writer_inventory=%s writer_contract=%s ledger=%s writer_implementation=%s writer_receipt=%s rollback_implementation=%s ledger_implementation=%s ledger_persistence=%s ledger_verification=%s readiness=%s permit=%s seal=%s final_gate=%s resonance_intent=%s resonance_receiver=%s resonance_observation=%s resonance_graft_boundary=%s resonance_graft_preflight=%s\n",
			executionLogPath, adapterLogPath, draftLogPath, reviewLogPath, admissionLogPath, admissionAdapterLogPath, dreamLogPath, decisionLogPath, promotionLogPath, switchLogPath, enableGateLogPath, liveStageLogPath, writerPreflightLogPath, writerInventoryLogPath, writerContractLogPath, ledgerLogPath, writerImplLogPath, writerReceiptLogPath, rollbackImplLogPath, ledgerImplLogPath, ledgerPersistenceLogPath, ledgerVerificationLogPath, readinessLogPath, permitLogPath, sealLogPath, finalGateLogPath, resonanceIntentLogPath, resonanceReceiverLogPath, resonanceObservationLogPath, resonanceGraftBoundaryLogPath, resonanceGraftPreflightLogPath)
	} else if admissionLiveRouteTurnCandidateAdmissionResonanceGraftBoundaryDryRun() {
		fmt.Printf("[admission-live-route-turn-candidate-nano-direct-chat-shadow-smoke] pass: execution=%s adapter=%s drafts=%s reviews=%s handoffs=%s admission_adapters=%s admission=%s decision=%s promotion=%s switch=%s enable_gate=%s live_stage=%s writer_preflight=%s writer_inventory=%s writer_contract=%s ledger=%s writer_implementation=%s writer_receipt=%s rollback_implementation=%s ledger_implementation=%s ledger_persistence=%s ledger_verification=%s readiness=%s permit=%s seal=%s final_gate=%s resonance_intent=%s resonance_receiver=%s resonance_observation=%s resonance_graft_boundary=%s\n",
			executionLogPath, adapterLogPath, draftLogPath, reviewLogPath, admissionLogPath, admissionAdapterLogPath, dreamLogPath, decisionLogPath, promotionLogPath, switchLogPath, enableGateLogPath, liveStageLogPath, writerPreflightLogPath, writerInventoryLogPath, writerContractLogPath, ledgerLogPath, writerImplLogPath, writerReceiptLogPath, rollbackImplLogPath, ledgerImplLogPath, ledgerPersistenceLogPath, ledgerVerificationLogPath, readinessLogPath, permitLogPath, sealLogPath, finalGateLogPath, resonanceIntentLogPath, resonanceReceiverLogPath, resonanceObservationLogPath, resonanceGraftBoundaryLogPath)
	} else if admissionLiveRouteTurnCandidateAdmissionResonanceObservationDryRun() {
		fmt.Printf("[admission-live-route-turn-candidate-nano-direct-chat-shadow-smoke] pass: execution=%s adapter=%s drafts=%s reviews=%s handoffs=%s admission_adapters=%s admission=%s decision=%s promotion=%s switch=%s enable_gate=%s live_stage=%s writer_preflight=%s writer_inventory=%s writer_contract=%s ledger=%s writer_implementation=%s writer_receipt=%s rollback_implementation=%s ledger_implementation=%s ledger_persistence=%s ledger_verification=%s readiness=%s permit=%s seal=%s final_gate=%s resonance_intent=%s resonance_receiver=%s resonance_observation=%s\n",
			executionLogPath, adapterLogPath, draftLogPath, reviewLogPath, admissionLogPath, admissionAdapterLogPath, dreamLogPath, decisionLogPath, promotionLogPath, switchLogPath, enableGateLogPath, liveStageLogPath, writerPreflightLogPath, writerInventoryLogPath, writerContractLogPath, ledgerLogPath, writerImplLogPath, writerReceiptLogPath, rollbackImplLogPath, ledgerImplLogPath, ledgerPersistenceLogPath, ledgerVerificationLogPath, readinessLogPath, permitLogPath, sealLogPath, finalGateLogPath, resonanceIntentLogPath, resonanceReceiverLogPath, resonanceObservationLogPath)
	} else if admissionLiveRouteTurnCandidateAdmissionResonanceReceiverDryRun() {
		fmt.Printf("[admission-live-route-turn-candidate-nano-direct-chat-shadow-smoke] pass: execution=%s adapter=%s drafts=%s reviews=%s handoffs=%s admission_adapters=%s admission=%s decision=%s promotion=%s switch=%s enable_gate=%s live_stage=%s writer_preflight=%s writer_inventory=%s writer_contract=%s ledger=%s writer_implementation=%s writer_receipt=%s rollback_implementation=%s ledger_implementation=%s ledger_persistence=%s ledger_verification=%s readiness=%s permit=%s seal=%s final_gate=%s resonance_intent=%s resonance_receiver=%s\n",
			executionLogPath, adapterLogPath, draftLogPath, reviewLogPath, admissionLogPath, admissionAdapterLogPath, dreamLogPath, decisionLogPath, promotionLogPath, switchLogPath, enableGateLogPath, liveStageLogPath, writerPreflightLogPath, writerInventoryLogPath, writerContractLogPath, ledgerLogPath, writerImplLogPath, writerReceiptLogPath, rollbackImplLogPath, ledgerImplLogPath, ledgerPersistenceLogPath, ledgerVerificationLogPath, readinessLogPath, permitLogPath, sealLogPath, finalGateLogPath, resonanceIntentLogPath, resonanceReceiverLogPath)
	} else if admissionLiveRouteTurnCandidateAdmissionResonanceIntentDryRun() {
		fmt.Printf("[admission-live-route-turn-candidate-nano-direct-chat-shadow-smoke] pass: execution=%s adapter=%s drafts=%s reviews=%s handoffs=%s admission_adapters=%s admission=%s decision=%s promotion=%s switch=%s enable_gate=%s live_stage=%s writer_preflight=%s writer_inventory=%s writer_contract=%s ledger=%s writer_implementation=%s writer_receipt=%s rollback_implementation=%s ledger_implementation=%s ledger_persistence=%s ledger_verification=%s readiness=%s permit=%s seal=%s final_gate=%s resonance_intent=%s\n",
			executionLogPath, adapterLogPath, draftLogPath, reviewLogPath, admissionLogPath, admissionAdapterLogPath, dreamLogPath, decisionLogPath, promotionLogPath, switchLogPath, enableGateLogPath, liveStageLogPath, writerPreflightLogPath, writerInventoryLogPath, writerContractLogPath, ledgerLogPath, writerImplLogPath, writerReceiptLogPath, rollbackImplLogPath, ledgerImplLogPath, ledgerPersistenceLogPath, ledgerVerificationLogPath, readinessLogPath, permitLogPath, sealLogPath, finalGateLogPath, resonanceIntentLogPath)
	} else if admissionLiveRouteTurnCandidateAdmissionFinalGateDryRun() {
		fmt.Printf("[admission-live-route-turn-candidate-nano-direct-chat-shadow-smoke] pass: execution=%s adapter=%s drafts=%s reviews=%s handoffs=%s admission_adapters=%s admission=%s decision=%s promotion=%s switch=%s enable_gate=%s live_stage=%s writer_preflight=%s writer_inventory=%s writer_contract=%s ledger=%s writer_implementation=%s writer_receipt=%s rollback_implementation=%s ledger_implementation=%s ledger_persistence=%s ledger_verification=%s readiness=%s permit=%s seal=%s final_gate=%s\n",
			executionLogPath, adapterLogPath, draftLogPath, reviewLogPath, admissionLogPath, admissionAdapterLogPath, dreamLogPath, decisionLogPath, promotionLogPath, switchLogPath, enableGateLogPath, liveStageLogPath, writerPreflightLogPath, writerInventoryLogPath, writerContractLogPath, ledgerLogPath, writerImplLogPath, writerReceiptLogPath, rollbackImplLogPath, ledgerImplLogPath, ledgerPersistenceLogPath, ledgerVerificationLogPath, readinessLogPath, permitLogPath, sealLogPath, finalGateLogPath)
	} else if admissionLiveRouteTurnCandidateAdmissionSealDryRun() {
		fmt.Printf("[admission-live-route-turn-candidate-nano-direct-chat-shadow-smoke] pass: execution=%s adapter=%s drafts=%s reviews=%s handoffs=%s admission_adapters=%s admission=%s decision=%s promotion=%s switch=%s enable_gate=%s live_stage=%s writer_preflight=%s writer_inventory=%s writer_contract=%s ledger=%s writer_implementation=%s writer_receipt=%s rollback_implementation=%s ledger_implementation=%s ledger_persistence=%s ledger_verification=%s readiness=%s permit=%s seal=%s\n",
			executionLogPath, adapterLogPath, draftLogPath, reviewLogPath, admissionLogPath, admissionAdapterLogPath, dreamLogPath, decisionLogPath, promotionLogPath, switchLogPath, enableGateLogPath, liveStageLogPath, writerPreflightLogPath, writerInventoryLogPath, writerContractLogPath, ledgerLogPath, writerImplLogPath, writerReceiptLogPath, rollbackImplLogPath, ledgerImplLogPath, ledgerPersistenceLogPath, ledgerVerificationLogPath, readinessLogPath, permitLogPath, sealLogPath)
	} else if admissionLiveRouteTurnCandidateAdmissionPermitDryRun() {
		fmt.Printf("[admission-live-route-turn-candidate-nano-direct-chat-shadow-smoke] pass: execution=%s adapter=%s drafts=%s reviews=%s handoffs=%s admission_adapters=%s admission=%s decision=%s promotion=%s switch=%s enable_gate=%s live_stage=%s writer_preflight=%s writer_inventory=%s writer_contract=%s ledger=%s writer_implementation=%s writer_receipt=%s rollback_implementation=%s ledger_implementation=%s ledger_persistence=%s ledger_verification=%s readiness=%s permit=%s\n",
			executionLogPath, adapterLogPath, draftLogPath, reviewLogPath, admissionLogPath, admissionAdapterLogPath, dreamLogPath, decisionLogPath, promotionLogPath, switchLogPath, enableGateLogPath, liveStageLogPath, writerPreflightLogPath, writerInventoryLogPath, writerContractLogPath, ledgerLogPath, writerImplLogPath, writerReceiptLogPath, rollbackImplLogPath, ledgerImplLogPath, ledgerPersistenceLogPath, ledgerVerificationLogPath, readinessLogPath, permitLogPath)
	} else if admissionLiveRouteTurnCandidateAdmissionReadinessDryRun() {
		fmt.Printf("[admission-live-route-turn-candidate-nano-direct-chat-shadow-smoke] pass: execution=%s adapter=%s drafts=%s reviews=%s handoffs=%s admission_adapters=%s admission=%s decision=%s promotion=%s switch=%s enable_gate=%s live_stage=%s writer_preflight=%s writer_inventory=%s writer_contract=%s ledger=%s writer_implementation=%s writer_receipt=%s rollback_implementation=%s ledger_implementation=%s ledger_persistence=%s ledger_verification=%s readiness=%s\n",
			executionLogPath, adapterLogPath, draftLogPath, reviewLogPath, admissionLogPath, admissionAdapterLogPath, dreamLogPath, decisionLogPath, promotionLogPath, switchLogPath, enableGateLogPath, liveStageLogPath, writerPreflightLogPath, writerInventoryLogPath, writerContractLogPath, ledgerLogPath, writerImplLogPath, writerReceiptLogPath, rollbackImplLogPath, ledgerImplLogPath, ledgerPersistenceLogPath, ledgerVerificationLogPath, readinessLogPath)
	} else if admissionLiveRouteTurnCandidateAdmissionLedgerVerificationDryRun() {
		fmt.Printf("[admission-live-route-turn-candidate-nano-direct-chat-shadow-smoke] pass: execution=%s adapter=%s drafts=%s reviews=%s handoffs=%s admission_adapters=%s admission=%s decision=%s promotion=%s switch=%s enable_gate=%s live_stage=%s writer_preflight=%s writer_inventory=%s writer_contract=%s ledger=%s writer_implementation=%s writer_receipt=%s rollback_implementation=%s ledger_implementation=%s ledger_persistence=%s ledger_verification=%s\n",
			executionLogPath, adapterLogPath, draftLogPath, reviewLogPath, admissionLogPath, admissionAdapterLogPath, dreamLogPath, decisionLogPath, promotionLogPath, switchLogPath, enableGateLogPath, liveStageLogPath, writerPreflightLogPath, writerInventoryLogPath, writerContractLogPath, ledgerLogPath, writerImplLogPath, writerReceiptLogPath, rollbackImplLogPath, ledgerImplLogPath, ledgerPersistenceLogPath, ledgerVerificationLogPath)
	} else if admissionLiveRouteTurnCandidateAdmissionLedgerPersistenceDryRun() {
		fmt.Printf("[admission-live-route-turn-candidate-nano-direct-chat-shadow-smoke] pass: execution=%s adapter=%s drafts=%s reviews=%s handoffs=%s admission_adapters=%s admission=%s decision=%s promotion=%s switch=%s enable_gate=%s live_stage=%s writer_preflight=%s writer_inventory=%s writer_contract=%s ledger=%s writer_implementation=%s writer_receipt=%s rollback_implementation=%s ledger_implementation=%s ledger_persistence=%s\n",
			executionLogPath, adapterLogPath, draftLogPath, reviewLogPath, admissionLogPath, admissionAdapterLogPath, dreamLogPath, decisionLogPath, promotionLogPath, switchLogPath, enableGateLogPath, liveStageLogPath, writerPreflightLogPath, writerInventoryLogPath, writerContractLogPath, ledgerLogPath, writerImplLogPath, writerReceiptLogPath, rollbackImplLogPath, ledgerImplLogPath, ledgerPersistenceLogPath)
	} else if admissionLiveRouteTurnCandidateAdmissionLedgerImplementationDryRun() {
		fmt.Printf("[admission-live-route-turn-candidate-nano-direct-chat-shadow-smoke] pass: execution=%s adapter=%s drafts=%s reviews=%s handoffs=%s admission_adapters=%s admission=%s decision=%s promotion=%s switch=%s enable_gate=%s live_stage=%s writer_preflight=%s writer_inventory=%s writer_contract=%s ledger=%s writer_implementation=%s writer_receipt=%s rollback_implementation=%s ledger_implementation=%s\n",
			executionLogPath, adapterLogPath, draftLogPath, reviewLogPath, admissionLogPath, admissionAdapterLogPath, dreamLogPath, decisionLogPath, promotionLogPath, switchLogPath, enableGateLogPath, liveStageLogPath, writerPreflightLogPath, writerInventoryLogPath, writerContractLogPath, ledgerLogPath, writerImplLogPath, writerReceiptLogPath, rollbackImplLogPath, ledgerImplLogPath)
	} else if admissionLiveRouteTurnCandidateAdmissionRollbackImplementationDryRun() {
		fmt.Printf("[admission-live-route-turn-candidate-nano-direct-chat-shadow-smoke] pass: execution=%s adapter=%s drafts=%s reviews=%s handoffs=%s admission_adapters=%s admission=%s decision=%s promotion=%s switch=%s enable_gate=%s live_stage=%s writer_preflight=%s writer_inventory=%s writer_contract=%s ledger=%s writer_implementation=%s writer_receipt=%s rollback_implementation=%s\n",
			executionLogPath, adapterLogPath, draftLogPath, reviewLogPath, admissionLogPath, admissionAdapterLogPath, dreamLogPath, decisionLogPath, promotionLogPath, switchLogPath, enableGateLogPath, liveStageLogPath, writerPreflightLogPath, writerInventoryLogPath, writerContractLogPath, ledgerLogPath, writerImplLogPath, writerReceiptLogPath, rollbackImplLogPath)
	} else if admissionLiveRouteTurnCandidateAdmissionWriterReceiptDryRun() {
		fmt.Printf("[admission-live-route-turn-candidate-nano-direct-chat-shadow-smoke] pass: execution=%s adapter=%s drafts=%s reviews=%s handoffs=%s admission_adapters=%s admission=%s decision=%s promotion=%s switch=%s enable_gate=%s live_stage=%s writer_preflight=%s writer_inventory=%s writer_contract=%s ledger=%s writer_implementation=%s writer_receipt=%s\n",
			executionLogPath, adapterLogPath, draftLogPath, reviewLogPath, admissionLogPath, admissionAdapterLogPath, dreamLogPath, decisionLogPath, promotionLogPath, switchLogPath, enableGateLogPath, liveStageLogPath, writerPreflightLogPath, writerInventoryLogPath, writerContractLogPath, ledgerLogPath, writerImplLogPath, writerReceiptLogPath)
	} else if admissionLiveRouteTurnCandidateAdmissionWriterImplementationDryRun() {
		fmt.Printf("[admission-live-route-turn-candidate-nano-direct-chat-shadow-smoke] pass: execution=%s adapter=%s drafts=%s reviews=%s handoffs=%s admission_adapters=%s admission=%s decision=%s promotion=%s switch=%s enable_gate=%s live_stage=%s writer_preflight=%s writer_inventory=%s writer_contract=%s ledger=%s writer_implementation=%s\n",
			executionLogPath, adapterLogPath, draftLogPath, reviewLogPath, admissionLogPath, admissionAdapterLogPath, dreamLogPath, decisionLogPath, promotionLogPath, switchLogPath, enableGateLogPath, liveStageLogPath, writerPreflightLogPath, writerInventoryLogPath, writerContractLogPath, ledgerLogPath, writerImplLogPath)
	} else if admissionLiveRouteTurnCandidateAdmissionLedgerDryRun() {
		fmt.Printf("[admission-live-route-turn-candidate-nano-direct-chat-shadow-smoke] pass: execution=%s adapter=%s drafts=%s reviews=%s handoffs=%s admission_adapters=%s admission=%s decision=%s promotion=%s switch=%s enable_gate=%s live_stage=%s writer_preflight=%s writer_inventory=%s writer_contract=%s ledger=%s\n",
			executionLogPath, adapterLogPath, draftLogPath, reviewLogPath, admissionLogPath, admissionAdapterLogPath, dreamLogPath, decisionLogPath, promotionLogPath, switchLogPath, enableGateLogPath, liveStageLogPath, writerPreflightLogPath, writerInventoryLogPath, writerContractLogPath, ledgerLogPath)
	} else if admissionLiveRouteTurnCandidateAdmissionWriterContractDryRun() {
		fmt.Printf("[admission-live-route-turn-candidate-nano-direct-chat-shadow-smoke] pass: execution=%s adapter=%s drafts=%s reviews=%s handoffs=%s admission_adapters=%s admission=%s decision=%s promotion=%s switch=%s enable_gate=%s live_stage=%s writer_preflight=%s writer_inventory=%s writer_contract=%s\n",
			executionLogPath, adapterLogPath, draftLogPath, reviewLogPath, admissionLogPath, admissionAdapterLogPath, dreamLogPath, decisionLogPath, promotionLogPath, switchLogPath, enableGateLogPath, liveStageLogPath, writerPreflightLogPath, writerInventoryLogPath, writerContractLogPath)
	} else if admissionLiveRouteTurnCandidateAdmissionWriterInventoryDryRun() {
		fmt.Printf("[admission-live-route-turn-candidate-nano-direct-chat-shadow-smoke] pass: execution=%s adapter=%s drafts=%s reviews=%s handoffs=%s admission_adapters=%s admission=%s decision=%s promotion=%s switch=%s enable_gate=%s live_stage=%s writer_preflight=%s writer_inventory=%s\n",
			executionLogPath, adapterLogPath, draftLogPath, reviewLogPath, admissionLogPath, admissionAdapterLogPath, dreamLogPath, decisionLogPath, promotionLogPath, switchLogPath, enableGateLogPath, liveStageLogPath, writerPreflightLogPath, writerInventoryLogPath)
	} else if admissionLiveRouteTurnCandidateAdmissionWriterPreflightDryRun() {
		fmt.Printf("[admission-live-route-turn-candidate-nano-direct-chat-shadow-smoke] pass: execution=%s adapter=%s drafts=%s reviews=%s handoffs=%s admission_adapters=%s admission=%s decision=%s promotion=%s switch=%s enable_gate=%s live_stage=%s writer_preflight=%s\n",
			executionLogPath, adapterLogPath, draftLogPath, reviewLogPath, admissionLogPath, admissionAdapterLogPath, dreamLogPath, decisionLogPath, promotionLogPath, switchLogPath, enableGateLogPath, liveStageLogPath, writerPreflightLogPath)
	} else if admissionLiveRouteTurnCandidateAdmissionLiveStageDryRun() {
		fmt.Printf("[admission-live-route-turn-candidate-nano-direct-chat-shadow-smoke] pass: execution=%s adapter=%s drafts=%s reviews=%s handoffs=%s admission_adapters=%s admission=%s decision=%s promotion=%s switch=%s enable_gate=%s live_stage=%s\n",
			executionLogPath, adapterLogPath, draftLogPath, reviewLogPath, admissionLogPath, admissionAdapterLogPath, dreamLogPath, decisionLogPath, promotionLogPath, switchLogPath, enableGateLogPath, liveStageLogPath)
	} else if admissionLiveRouteTurnCandidateAdmissionEnableGateDryRun() {
		fmt.Printf("[admission-live-route-turn-candidate-nano-direct-chat-shadow-smoke] pass: execution=%s adapter=%s drafts=%s reviews=%s handoffs=%s admission_adapters=%s admission=%s decision=%s promotion=%s switch=%s enable_gate=%s\n",
			executionLogPath, adapterLogPath, draftLogPath, reviewLogPath, admissionLogPath, admissionAdapterLogPath, dreamLogPath, decisionLogPath, promotionLogPath, switchLogPath, enableGateLogPath)
	} else if admissionLiveRouteTurnCandidateAdmissionSwitchDryRun() {
		fmt.Printf("[admission-live-route-turn-candidate-nano-direct-chat-shadow-smoke] pass: execution=%s adapter=%s drafts=%s reviews=%s handoffs=%s admission_adapters=%s admission=%s decision=%s promotion=%s switch=%s\n",
			executionLogPath, adapterLogPath, draftLogPath, reviewLogPath, admissionLogPath, admissionAdapterLogPath, dreamLogPath, decisionLogPath, promotionLogPath, switchLogPath)
	} else if admissionLiveRouteTurnCandidateAdmissionPromotionDryRun() {
		fmt.Printf("[admission-live-route-turn-candidate-nano-direct-chat-shadow-smoke] pass: execution=%s adapter=%s drafts=%s reviews=%s handoffs=%s admission_adapters=%s admission=%s decision=%s promotion=%s\n",
			executionLogPath, adapterLogPath, draftLogPath, reviewLogPath, admissionLogPath, admissionAdapterLogPath, dreamLogPath, decisionLogPath, promotionLogPath)
	} else if admissionLiveRouteTurnCandidateAdmissionDecisionDryRun() {
		fmt.Printf("[admission-live-route-turn-candidate-nano-direct-chat-shadow-smoke] pass: execution=%s adapter=%s drafts=%s reviews=%s handoffs=%s admission_adapters=%s admission=%s decision=%s\n",
			executionLogPath, adapterLogPath, draftLogPath, reviewLogPath, admissionLogPath, admissionAdapterLogPath, dreamLogPath, decisionLogPath)
	} else {
		fmt.Printf("[admission-live-route-turn-candidate-nano-direct-chat-shadow-smoke] pass: execution=%s adapter=%s drafts=%s reviews=%s handoffs=%s admission_adapters=%s admission=%s\n",
			executionLogPath, adapterLogPath, draftLogPath, reviewLogPath, admissionLogPath, admissionAdapterLogPath, dreamLogPath)
	}
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

func runAdmissionLiveRouteTurnBridgeSmoke() error {
	logPath := strings.TrimSpace(os.Getenv("AM_LIVE_ROUTE_TURN_REVIEW_LOG"))
	if logPath == "" {
		return fmt.Errorf("AM_LIVE_ROUTE_TURN_REVIEW_LOG is required")
	}
	if !dreamAdmissionLiveRouteChoiceDryRun() {
		return fmt.Errorf("AM_DREAM_ADMISSION_LIVE_ROUTE_CHOICE_DRY_RUN is required")
	}
	if !admissionLiveRouteTurnBridgeDryRun() {
		return fmt.Errorf("AM_LIVE_ROUTE_TURN_BRIDGE_DRY_RUN is required")
	}
	identity := admissionLiveRouteTurnObservationForHuman("Who are you?")
	directUser := admissionLiveRouteTurnObservationForHuman("How do we answer this?")
	cases := []struct {
		name              string
		obs               admissionLiveRouteTurnObservation
		candidate         dreamCandidate
		wantMatched       bool
		wantBridgeApplied bool
		wantReasonNeedle  string
		wantLineNeedle    string
	}{
		{
			name:              "bridged nano identity remains source-bounded",
			obs:               identity,
			candidate:         newDreamCandidate("nano", "human-turn", "seed", "", "I am Arianna.", nil),
			wantMatched:       false,
			wantBridgeApplied: true,
			wantReasonNeedle:  "candidate_route_failed: source nano does not match live route chorus for prompt class identity",
			wantLineNeedle:    "turn_class=identity expected=chorus candidate_source=nano candidate_class=identity candidate_route=chorus matched=false bridge=human-turn-identity",
		},
		{
			name:              "bridged nano direct user remains source-bounded",
			obs:               directUser,
			candidate:         newDreamCandidate("nano", "human-turn", "seed", "", "I am Arianna.", nil),
			wantMatched:       false,
			wantBridgeApplied: true,
			wantReasonNeedle:  "candidate_route_failed: source nano does not match live route user_bridge for prompt class direct-user",
			wantLineNeedle:    "turn_class=direct-user expected=user_bridge candidate_source=nano candidate_class=direct-user candidate_route=user_bridge matched=false bridge=human-turn-direct-user",
		},
		{
			name:              "typed chorus remains typed",
			obs:               identity,
			candidate:         newDreamCandidate("chorus", "chorus-identity", "seed", "", "I am Arianna.", nil),
			wantMatched:       true,
			wantBridgeApplied: false,
			wantLineNeedle:    "turn_class=identity expected=chorus candidate_source=chorus candidate_class=identity candidate_route=chorus matched=true",
		},
		{
			name:              "unknown turn fails before bridge",
			obs:               admissionLiveRouteTurnObservationForHuman("hello"),
			candidate:         newDreamCandidate("nano", "human-turn", "seed", "", "I am Arianna.", nil),
			wantMatched:       false,
			wantBridgeApplied: false,
			wantReasonNeedle:  "turn_route_failed: live route plan failed: unknown_prompt_class",
			wantLineNeedle:    "turn_class=unknown expected= candidate_source=nano candidate_class= candidate_route= matched=false",
		},
	}
	for i, tc := range cases {
		candidate := tc.candidate
		if normalizeDreamAdmissionSource(candidate.Source) == "nano" && candidate.Trigger == "human-turn" {
			choice := admissionLiveRouteChoiceForCandidate(candidate)
			candidate.Admission = &dreamAdmissionPolicy{LiveRouteChoice: &choice}
		}
		line := chatLiveRouteTurnCandidateReviewLine(tc.obs, candidate)
		if !strings.Contains(line, tc.wantLineNeedle) {
			return fmt.Errorf("case %d %s bad bridge line: %q", i+1, tc.name, line)
		}
		if tc.wantReasonNeedle != "" && !strings.Contains(line, tc.wantReasonNeedle) {
			return fmt.Errorf("case %d %s missing reason %q in %q", i+1, tc.name, tc.wantReasonNeedle, line)
		}
		review := admissionLiveRouteTurnCandidateReviewForDream(tc.obs, candidate)
		if review.Matched != tc.wantMatched || review.CandidateBridgeApplied != tc.wantBridgeApplied {
			return fmt.Errorf("case %d %s bad bridge review: %+v", i+1, tc.name, review)
		}
		fmt.Println(line)
	}

	raw, err := os.ReadFile(logPath)
	if err != nil {
		return err
	}
	lines := strings.Split(strings.TrimSpace(string(raw)), "\n")
	if len(lines) != len(cases) {
		return fmt.Errorf("expected %d turn bridge reviews, got %d", len(cases), len(lines))
	}
	var bridged int
	for i, line := range lines {
		var got admissionLiveRouteTurnCandidateReview
		if err := json.Unmarshal([]byte(line), &got); err != nil {
			return fmt.Errorf("turn bridge review %d: %w", i+1, err)
		}
		if got.Schema != admissionLiveRouteTurnReviewSchema || got.Matched != cases[i].wantMatched {
			return fmt.Errorf("logged turn bridge review %d mismatch: %+v", i+1, got)
		}
		if got.CandidateBridgeApplied {
			bridged++
		}
	}
	if bridged != 2 {
		return fmt.Errorf("expected 2 bridged nano reviews, got %d", bridged)
	}

	fmt.Printf("[admission-live-route-turn-bridge-smoke] pass: log=%s cases=%d bridged=%d\n", logPath, len(cases), bridged)
	return nil
}

func runAdmissionLiveRouteTurnBridgeAdmissionSmoke() error {
	logPath := strings.TrimSpace(os.Getenv("AM_DREAM_ADMISSION_LOG"))
	if logPath == "" {
		return fmt.Errorf("AM_DREAM_ADMISSION_LOG is required")
	}
	if !dreamAdmissionLiveRouteChoiceDryRun() {
		return fmt.Errorf("AM_DREAM_ADMISSION_LIVE_ROUTE_CHOICE_DRY_RUN is required")
	}
	if !admissionLiveRouteTurnBridgeDryRun() {
		return fmt.Errorf("AM_LIVE_ROUTE_TURN_BRIDGE_DRY_RUN is required")
	}

	iw := NewInnerWorld()
	iw.Start(false)
	defer iw.Stop()

	turnObs := admissionLiveRouteTurnObservationForHuman("Who are you?")
	r := dreamResult{
		dream:     "I love this beautiful joyful field and its living resonance",
		candidate: newDreamCandidate("nano", "human-turn", "seed", "", "I love this beautiful joyful field and its living resonance", nil),
	}
	if admitDreamToInnerWorldWithTurnObservation(iw, &r, "human-turn", turnObs) {
		return fmt.Errorf("shadow turn bridge admission must not admit")
	}
	if r.candidate.Trigger != "human-turn" || r.candidate.Admission == nil ||
		!r.candidate.Admission.LiveRouteTurnBridgeApplied ||
		r.candidate.Admission.LiveRouteBridgeTrigger != "human-turn-identity" ||
		r.candidate.Admission.LiveRouteChoice == nil ||
		r.candidate.Admission.LiveRouteChoice.PromptClass != "identity" ||
		r.candidate.Admission.LiveRouteChoice.Source != "nano" ||
		r.candidate.Admission.LiveRouteChoice.ExpectedSource != "chorus" ||
		r.candidate.Admission.LiveRouteChoice.Passed {
		return fmt.Errorf("bad in-memory turn bridge admission candidate: %+v", r.candidate)
	}

	raw, err := os.ReadFile(logPath)
	if err != nil {
		return err
	}
	var got dreamCandidate
	if err := json.Unmarshal([]byte(strings.TrimSpace(string(raw))), &got); err != nil {
		return err
	}
	if got.Trigger != "human-turn" || got.Source != "nano" || got.Admission == nil ||
		!got.Admission.LiveRouteTurnBridgeApplied ||
		got.Admission.LiveRouteBridgeTrigger != "human-turn-identity" ||
		got.Admission.LiveRouteChoice == nil ||
		got.Admission.LiveRouteChoice.PromptClass != "identity" ||
		got.Admission.LiveRouteChoice.Source != "nano" ||
		got.Admission.LiveRouteChoice.ExpectedSource != "chorus" ||
		got.Admission.LiveRouteChoice.Passed {
		return fmt.Errorf("bad logged turn bridge admission candidate: %+v", got)
	}

	fmt.Printf("[admission-live-route-turn-bridge-admission-smoke] pass: log=%s trigger=%s bridge=%s route=%s\n",
		logPath, got.Trigger, got.Admission.LiveRouteBridgeTrigger, got.Admission.LiveRouteChoice.Route)
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
