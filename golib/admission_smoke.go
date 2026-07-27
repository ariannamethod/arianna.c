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
	if admissionLiveRouteTurnCandidateAdmissionEnableGateDryRun() &&
		(!admissionLiveRouteTurnCandidateAdmissionDecisionDryRun() ||
			!admissionLiveRouteTurnCandidateAdmissionPromotionDryRun() ||
			!admissionLiveRouteTurnCandidateAdmissionSwitchDryRun()) {
		return fmt.Errorf("AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_DECISION_DRY_RUN, AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_PROMOTION_DRY_RUN, and AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_SWITCH_DRY_RUN are required for enable gate smoke")
	}
	if admissionLiveRouteTurnCandidateAdmissionEnableGateDryRun() &&
		admissionLiveRouteTurnCandidateAdmissionEnableGateKey() != "" {
		return fmt.Errorf("AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_ENABLE_GATE_KEY must be empty for default-off enable gate smoke")
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
		wants = append(wants,
			"live-route candidate admission enable gate dry-run: class=dream route=direct source=direct switch=disabled switch_id=switch-",
			"enable=disabled enable_action=require_operator_key enable_id=enable-",
			"admission_allowed=false manual_enable=false key_matched=false live_ready=true live_enabled=false mutates=false passed=true reason=live admission enable gate closed; operator key absent",
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
			gate.ManualEnableRequested ||
			gate.EnableKeyMatched ||
			gate.MutatesState ||
			gate.EnableState != "disabled" ||
			gate.EnableAction != "require_operator_key" ||
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
	}

	if admissionLiveRouteTurnCandidateAdmissionEnableGateDryRun() {
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
