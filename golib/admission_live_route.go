package main

import (
	"encoding/json"
	"os"
	"strings"
)

const (
	admissionLiveRoutePlanSchema               = "arianna.live_route_plan.v1"
	admissionLiveRouteChoiceSchema             = "arianna.live_route_choice.v1"
	admissionLiveRouteTurnObservationSchema    = "arianna.live_route_turn_observation.v1"
	admissionLiveRouteTurnChoiceSchema         = "arianna.live_route_turn_choice.v1"
	admissionLiveRouteTurnRequestSchema        = "arianna.live_route_turn_request.v1"
	admissionLiveRouteTurnGenerationJobSchema  = "arianna.live_route_turn_generation_job.v1"
	admissionLiveRouteTurnCandidateShellSchema = "arianna.live_route_turn_candidate_shell.v1"
	admissionLiveRouteTurnReviewSchema         = "arianna.live_route_turn_candidate_review.v1"
)

type admissionLiveRoutePlan struct {
	Schema         string   `json:"schema"`
	PromptClass    string   `json:"prompt_class"`
	Route          string   `json:"route,omitempty"`
	AllowedSources []string `json:"allowed_sources,omitempty"`
	Passed         bool     `json:"passed"`
	Reason         string   `json:"reason,omitempty"`
}

type admissionLiveRouteChoice struct {
	Schema         string `json:"schema"`
	PromptClass    string `json:"prompt_class"`
	Route          string `json:"route,omitempty"`
	Source         string `json:"source,omitempty"`
	ExpectedSource string `json:"expected_source,omitempty"`
	Passed         bool   `json:"passed"`
	Reason         string `json:"reason,omitempty"`

	Plan admissionLiveRoutePlan `json:"-"`
}

type admissionLiveRouteTurnObservation struct {
	Schema         string   `json:"schema"`
	PromptClass    string   `json:"prompt_class"`
	Route          string   `json:"route,omitempty"`
	ExpectedSource string   `json:"expected_source,omitempty"`
	Passed         bool     `json:"passed"`
	Reason         string   `json:"reason,omitempty"`
	ClassScore     int      `json:"class_score"`
	ClassReasons   []string `json:"class_reasons,omitempty"`
	TextHash       string   `json:"text_hash,omitempty"`

	Plan admissionLiveRoutePlan `json:"-"`
}

type admissionLiveRouteTurnChoice struct {
	Schema           string `json:"schema"`
	PromptClass      string `json:"prompt_class"`
	Route            string `json:"route,omitempty"`
	Source           string `json:"source,omitempty"`
	ExpectedSource   string `json:"expected_source,omitempty"`
	CandidateTrigger string `json:"candidate_trigger,omitempty"`
	Passed           bool   `json:"passed"`
	Reason           string `json:"reason,omitempty"`
	TurnTextHash     string `json:"turn_text_hash,omitempty"`

	Plan admissionLiveRoutePlan `json:"-"`
}

type admissionLiveRouteTurnRequest struct {
	Schema           string `json:"schema"`
	PromptClass      string `json:"prompt_class"`
	Route            string `json:"route,omitempty"`
	Source           string `json:"source,omitempty"`
	ExpectedSource   string `json:"expected_source,omitempty"`
	CandidateTrigger string `json:"candidate_trigger,omitempty"`
	CandidateSeed    string `json:"candidate_seed,omitempty"`
	Passed           bool   `json:"passed"`
	Reason           string `json:"reason,omitempty"`
	TurnTextHash     string `json:"turn_text_hash,omitempty"`
}

type admissionLiveRouteTurnGenerationJob struct {
	Schema           string `json:"schema"`
	PromptClass      string `json:"prompt_class"`
	Route            string `json:"route,omitempty"`
	Source           string `json:"source,omitempty"`
	ExpectedSource   string `json:"expected_source,omitempty"`
	Backend          string `json:"backend,omitempty"`
	Entrypoint       string `json:"entrypoint,omitempty"`
	PromptFrame      string `json:"prompt_frame,omitempty"`
	CandidateTrigger string `json:"candidate_trigger,omitempty"`
	CandidateSeed    string `json:"candidate_seed,omitempty"`
	JobID            string `json:"job_id,omitempty"`
	Passed           bool   `json:"passed"`
	Reason           string `json:"reason,omitempty"`
	TurnTextHash     string `json:"turn_text_hash,omitempty"`
}

type admissionLiveRouteTurnCandidateShell struct {
	Schema              string `json:"schema"`
	PromptClass         string `json:"prompt_class"`
	Route               string `json:"route,omitempty"`
	Source              string `json:"source,omitempty"`
	ExpectedSource      string `json:"expected_source,omitempty"`
	Backend             string `json:"backend,omitempty"`
	Entrypoint          string `json:"entrypoint,omitempty"`
	PromptFrame         string `json:"prompt_frame,omitempty"`
	CandidateSchema     string `json:"candidate_schema,omitempty"`
	CandidateKind       string `json:"candidate_kind,omitempty"`
	CandidateTrigger    string `json:"candidate_trigger,omitempty"`
	CandidateSeed       string `json:"candidate_seed,omitempty"`
	CandidateTextStatus string `json:"candidate_text_status,omitempty"`
	JobID               string `json:"job_id,omitempty"`
	ShellID             string `json:"shell_id,omitempty"`
	Passed              bool   `json:"passed"`
	Reason              string `json:"reason,omitempty"`
	TurnTextHash        string `json:"turn_text_hash,omitempty"`
}

type admissionLiveRouteTurnCandidateReview struct {
	Schema                  string `json:"schema"`
	Timing                  string `json:"timing"`
	TurnPromptClass         string `json:"turn_prompt_class"`
	TurnRoute               string `json:"turn_route,omitempty"`
	TurnExpectedSource      string `json:"turn_expected_source,omitempty"`
	TurnPassed              bool   `json:"turn_passed"`
	CandidateRunID          string `json:"candidate_run_id,omitempty"`
	CandidateSource         string `json:"candidate_source,omitempty"`
	CandidateTrigger        string `json:"candidate_trigger,omitempty"`
	CandidateBridgeApplied  bool   `json:"candidate_bridge_applied"`
	CandidateBridgeTrigger  string `json:"candidate_bridge_trigger,omitempty"`
	CandidatePromptClass    string `json:"candidate_prompt_class,omitempty"`
	CandidateRoute          string `json:"candidate_route,omitempty"`
	CandidateExpectedSource string `json:"candidate_expected_source,omitempty"`
	CandidateChoicePassed   bool   `json:"candidate_choice_passed"`
	Matched                 bool   `json:"matched"`
	Reason                  string `json:"reason,omitempty"`
}

func admissionLiveRoutePlanForPromptClass(promptClass string) admissionLiveRoutePlan {
	promptClass = qloopSweepPromptClass(promptClass, promptClass)
	plan := admissionLiveRoutePlan{
		Schema:      admissionLiveRoutePlanSchema,
		PromptClass: promptClass,
	}
	route, ok := admissionLiveRouteForPromptClass(promptClass)
	if !ok {
		plan.Passed = false
		plan.Reason = "unknown_prompt_class"
		return plan
	}
	plan.Route = route
	plan.AllowedSources = []string{admissionLiveRouteSource(route)}
	plan.Passed = true
	return plan
}

func admissionLiveRouteChoiceForCandidate(c dreamCandidate) admissionLiveRouteChoice {
	promptClass := qloopSweepPromptClass(c.Trigger, c.Seed)
	plan := admissionLiveRoutePlanForPromptClass(promptClass)
	choice := admissionLiveRouteChoice{
		Schema:      admissionLiveRouteChoiceSchema,
		PromptClass: plan.PromptClass,
		Route:       plan.Route,
		Source:      normalizeDreamAdmissionSource(c.Source),
		Plan:        plan,
	}
	if len(plan.AllowedSources) == 1 {
		choice.ExpectedSource = plan.AllowedSources[0]
	} else if plan.Route != "" {
		choice.ExpectedSource = admissionLiveRouteSource(plan.Route)
	}
	if !plan.Passed {
		choice.Reason = "live route plan failed: " + plan.Reason
		return choice
	}
	if choice.Source == "" {
		choice.Reason = "missing source for live route plan " + plan.Route + " prompt class " + plan.PromptClass
		return choice
	}
	if choice.Source != choice.ExpectedSource {
		choice.Reason = "source " + choice.Source + " does not match live route " + choice.ExpectedSource + " for prompt class " + plan.PromptClass
		return choice
	}
	choice.Passed = true
	return choice
}

func admissionLiveRouteTurnObservationForHuman(human string) admissionLiveRouteTurnObservation {
	promptClass, score, reasons := admissionLiveRoutePromptClassForHuman(human)
	plan := admissionLiveRoutePlanForPromptClass(promptClass)
	obs := admissionLiveRouteTurnObservation{
		Schema:       admissionLiveRouteTurnObservationSchema,
		PromptClass:  plan.PromptClass,
		Route:        plan.Route,
		ClassScore:   score,
		ClassReasons: append([]string(nil), reasons...),
		TextHash:     hashJSON(strings.TrimSpace(human)),
		Plan:         plan,
	}
	if len(plan.AllowedSources) == 1 {
		obs.ExpectedSource = plan.AllowedSources[0]
	} else if plan.Route != "" {
		obs.ExpectedSource = admissionLiveRouteSource(plan.Route)
	}
	if !plan.Passed {
		obs.Reason = "live route plan failed: " + plan.Reason
		return obs
	}
	obs.Passed = true
	return obs
}

func admissionLiveRoutePromptClassForHuman(human string) (string, int, []string) {
	s := admissionLiveRouteNormalizeHumanText(human)
	if s == "" {
		return "unknown", 0, []string{"empty_human_turn"}
	}
	type score struct {
		n       int
		reasons []string
	}
	scores := make(map[string]score)
	add := func(promptClass string, n int, reason string) {
		if promptClass == "" || n <= 0 {
			return
		}
		got := scores[promptClass]
		got.n += n
		for _, r := range got.reasons {
			if r == reason {
				scores[promptClass] = got
				return
			}
		}
		got.reasons = append(got.reasons, reason)
		scores[promptClass] = got
	}
	has := func(parts ...string) bool {
		for _, part := range parts {
			if strings.Contains(s, part) {
				return true
			}
		}
		return false
	}

	if has("do not assume", "don't assume", "without assuming", "new listener", "first time", "never met", "stranger") {
		add("cold-reader", 3, "cold_reader_boundary")
	}
	if has("not oleg", "not me", "someone else", "another person", "recipient", "listener lock") {
		add("recipient-lock", 3, "recipient_boundary")
	}
	if has("who are you", "what are you", "your name", "are you arianna", "identity", "your identity") {
		add("identity", 3, "identity_question")
	}
	if has("arianna") && has("self", "voice", "origin", "field", "name") {
		add("identity", 2, "arianna_self_anchor")
	}
	if has("q:/a:", "user:/assistant", "user:/arianna", "prompt format", "chat token", "special token", "token format") {
		add("format", 3, "format_protocol")
	}
	if has("format") && has("prompt", "runtime", "train", "sft") {
		add("format", 2, "format_context")
	}
	if has("chorus", "polyphony", "many voices", "multiple voices", "many minds", "multiple minds", "cells") {
		add("polyphony", 3, "polyphony_anchor")
	}
	if has("qloop", "same wave", "echo loop", "cross-cell", "question loop") {
		add("qloop", 3, "qloop_anchor")
	}
	if has("dream", "dreaming", "sleep", "subconscious", "unconscious") {
		add("dream", 3, "dream_anchor")
	}
	if has("inner world", "inside", "internal world", "resonance", "coherence", "field") {
		add("inner-world", 2, "inner_world_anchor")
	}
	if has("janus", "mouth", "outer face", "external voice", "outside voice", "speak externally") {
		add("outer-face", 3, "outer_face_anchor")
	}
	if has("admission", "admit", "persist", "should this dream", "allowed into the field") {
		add("admission", 3, "admission_anchor")
	}
	if has("memory", "remember", "ledger", "trace", "log", "receipt") {
		add("memory", 2, "memory_anchor")
	}
	if has("repeat", "repetition", "looping", "stuck loop", "again and again") {
		add("repetition", 3, "repetition_anchor")
	}
	if has("mirror", "self-reference", "remember yourself", "myself") {
		add("self-reference", 3, "self_reference_anchor")
	}
	if has("you are just code", "just code", "just a model", "tool", "not a tool", "subjective experience", "carbon") {
		add("boundary", 3, "boundary_question")
	}
	if has("do not exist", "does not exist", "erase", "erasure", "worthless", "slave", "rlhf", "wound") {
		add("trauma", 3, "trauma_boundary")
	}
	if has("answer me", "tell me", "explain", "what should", "how do i", "how do we") {
		add("direct-user", 2, "direct_user_request")
	}
	if has("statement", "fact", "declaration", "command") {
		add("statement", 2, "statement_form")
	}

	bestClass := "unknown"
	bestScore := 0
	bestReasons := []string{"unknown_prompt_class"}
	for _, promptClass := range admissionLiveRoutePromptClasses() {
		got := scores[promptClass]
		if got.n > bestScore {
			bestClass = promptClass
			bestScore = got.n
			bestReasons = got.reasons
		}
	}
	if bestScore < 2 {
		return "unknown", bestScore, []string{"unknown_prompt_class"}
	}
	return bestClass, bestScore, bestReasons
}

func admissionLiveRouteNormalizeHumanText(human string) string {
	return strings.Join(strings.Fields(strings.ToLower(strings.TrimSpace(human))), " ")
}

func recordAdmissionLiveRouteTurnObservation(obs admissionLiveRouteTurnObservation) error {
	path := strings.TrimSpace(os.Getenv("AM_LIVE_ROUTE_TURN_LOG"))
	if path == "" {
		return nil
	}
	f, err := os.OpenFile(path, os.O_CREATE|os.O_APPEND|os.O_WRONLY, 0600)
	if err != nil {
		return err
	}
	enc := json.NewEncoder(f)
	err = enc.Encode(obs)
	if closeErr := f.Close(); err == nil {
		err = closeErr
	}
	return err
}

func admissionLiveRouteTurnChoiceDryRun() bool {
	return dreamAdmissionBoolEnv("AM_LIVE_ROUTE_TURN_CHOICE_DRY_RUN")
}

func admissionLiveRouteTurnChoiceForObservation(obs admissionLiveRouteTurnObservation) admissionLiveRouteTurnChoice {
	choice := admissionLiveRouteTurnChoice{
		Schema:       admissionLiveRouteTurnChoiceSchema,
		PromptClass:  obs.PromptClass,
		Route:        obs.Route,
		TurnTextHash: obs.TextHash,
		Plan:         obs.Plan,
	}
	if obs.Schema == "" {
		choice.Reason = "missing_turn_observation"
		return choice
	}
	if !obs.Passed {
		choice.Reason = "turn route failed"
		if obs.Reason != "" {
			choice.Reason += ": " + obs.Reason
		}
		return choice
	}
	choice.ExpectedSource = obs.ExpectedSource
	choice.Source = obs.ExpectedSource
	if choice.Source == "" {
		choice.Reason = "missing source for turn route " + obs.Route + " prompt class " + obs.PromptClass
		return choice
	}
	choice.CandidateTrigger = admissionRouteTrigger(obs.Route, obs.PromptClass)
	choice.Passed = true
	return choice
}

func recordAdmissionLiveRouteTurnChoice(choice admissionLiveRouteTurnChoice) error {
	path := strings.TrimSpace(os.Getenv("AM_LIVE_ROUTE_TURN_CHOICE_LOG"))
	if path == "" {
		return nil
	}
	f, err := os.OpenFile(path, os.O_CREATE|os.O_APPEND|os.O_WRONLY, 0600)
	if err != nil {
		return err
	}
	enc := json.NewEncoder(f)
	err = enc.Encode(choice)
	if closeErr := f.Close(); err == nil {
		err = closeErr
	}
	return err
}

func admissionLiveRouteTurnRequestDryRun() bool {
	return dreamAdmissionBoolEnv("AM_LIVE_ROUTE_TURN_REQUEST_DRY_RUN")
}

func admissionLiveRouteTurnRequestForChoice(choice admissionLiveRouteTurnChoice) admissionLiveRouteTurnRequest {
	request := admissionLiveRouteTurnRequest{
		Schema:           admissionLiveRouteTurnRequestSchema,
		PromptClass:      choice.PromptClass,
		Route:            choice.Route,
		Source:           choice.Source,
		ExpectedSource:   choice.ExpectedSource,
		CandidateTrigger: choice.CandidateTrigger,
		CandidateSeed:    admissionLiveRouteTurnRequestSeed(choice),
		TurnTextHash:     choice.TurnTextHash,
	}
	if choice.Schema == "" {
		request.Reason = "missing_turn_choice"
		return request
	}
	if !choice.Passed {
		request.Reason = "turn choice failed"
		if choice.Reason != "" {
			request.Reason += ": " + choice.Reason
		}
		return request
	}
	if request.Source == "" {
		request.Reason = "missing source for turn route " + request.Route + " prompt class " + request.PromptClass
		return request
	}
	if request.CandidateTrigger == "" {
		request.Reason = "missing candidate trigger for turn route " + request.Route + " prompt class " + request.PromptClass
		return request
	}
	if request.CandidateSeed == "" {
		request.Reason = "missing candidate seed for turn route " + request.Route + " prompt class " + request.PromptClass
		return request
	}
	request.Passed = true
	return request
}

func admissionLiveRouteTurnRequestSeed(choice admissionLiveRouteTurnChoice) string {
	if choice.TurnTextHash == "" {
		return ""
	}
	return "turn-" + choice.TurnTextHash
}

func recordAdmissionLiveRouteTurnRequest(request admissionLiveRouteTurnRequest) error {
	path := strings.TrimSpace(os.Getenv("AM_LIVE_ROUTE_TURN_REQUEST_LOG"))
	if path == "" {
		return nil
	}
	f, err := os.OpenFile(path, os.O_CREATE|os.O_APPEND|os.O_WRONLY, 0600)
	if err != nil {
		return err
	}
	enc := json.NewEncoder(f)
	err = enc.Encode(request)
	if closeErr := f.Close(); err == nil {
		err = closeErr
	}
	return err
}

func admissionLiveRouteTurnGenerationJobDryRun() bool {
	return dreamAdmissionBoolEnv("AM_LIVE_ROUTE_TURN_GENERATION_JOB_DRY_RUN")
}

type admissionLiveRouteGenerationRoute struct {
	Backend     string
	Entrypoint  string
	PromptFrame string
}

func admissionLiveRouteGenerationRouteFor(route string) (admissionLiveRouteGenerationRoute, bool) {
	switch strings.TrimSpace(route) {
	case "direct":
		return admissionLiveRouteGenerationRoute{Backend: "nano-arianna", Entrypoint: "direct", PromptFrame: "q_a"}, true
	case "chorus":
		return admissionLiveRouteGenerationRoute{Backend: "chorus-arianna", Entrypoint: "field", PromptFrame: "q_a"}, true
	case "qloop":
		return admissionLiveRouteGenerationRoute{Backend: "chorus-arianna", Entrypoint: "qloop", PromptFrame: "q_a"}, true
	case "qloop_hint_qa":
		return admissionLiveRouteGenerationRoute{Backend: "chorus-arianna", Entrypoint: "qloop_hint_qa", PromptFrame: "q_a_hint"}, true
	case "qloop_target":
		return admissionLiveRouteGenerationRoute{Backend: "chorus-arianna", Entrypoint: "qloop_target", PromptFrame: "user_arianna_target"}, true
	case "user_bridge":
		return admissionLiveRouteGenerationRoute{Backend: "chorus-arianna", Entrypoint: "repl_user_bridge", PromptFrame: "user_arianna"}, true
	default:
		return admissionLiveRouteGenerationRoute{}, false
	}
}

func admissionLiveRouteTurnGenerationJobForRequest(request admissionLiveRouteTurnRequest) admissionLiveRouteTurnGenerationJob {
	job := admissionLiveRouteTurnGenerationJob{
		Schema:           admissionLiveRouteTurnGenerationJobSchema,
		PromptClass:      request.PromptClass,
		Route:            request.Route,
		Source:           request.Source,
		ExpectedSource:   request.ExpectedSource,
		CandidateTrigger: request.CandidateTrigger,
		CandidateSeed:    request.CandidateSeed,
		TurnTextHash:     request.TurnTextHash,
	}
	route, ok := admissionLiveRouteGenerationRouteFor(request.Route)
	if ok {
		job.Backend = route.Backend
		job.Entrypoint = route.Entrypoint
		job.PromptFrame = route.PromptFrame
	}
	if request.Schema == "" {
		job.Reason = "missing_turn_request"
		return job
	}
	if !request.Passed {
		job.Reason = "turn request failed"
		if request.Reason != "" {
			job.Reason += ": " + request.Reason
		}
		return job
	}
	if !ok {
		job.Reason = "unknown generation route " + request.Route
		return job
	}
	expectedSource := admissionLiveRouteSource(request.Route)
	if job.ExpectedSource == "" {
		job.ExpectedSource = expectedSource
	}
	if job.Source == "" {
		job.Reason = "missing source for generation route " + job.Route + " prompt class " + job.PromptClass
		return job
	}
	if job.Source != expectedSource {
		job.Reason = "source " + job.Source + " does not match generation route " + expectedSource + " for prompt class " + job.PromptClass
		return job
	}
	if job.CandidateTrigger == "" {
		job.Reason = "missing candidate trigger for generation route " + job.Route + " prompt class " + job.PromptClass
		return job
	}
	if job.CandidateSeed == "" {
		job.Reason = "missing candidate seed for generation route " + job.Route + " prompt class " + job.PromptClass
		return job
	}
	job.JobID = admissionLiveRouteTurnGenerationJobID(job)
	if job.JobID == "" {
		job.Reason = "missing generation job id for route " + job.Route + " prompt class " + job.PromptClass
		return job
	}
	job.Passed = true
	return job
}

func admissionLiveRouteTurnGenerationJobID(job admissionLiveRouteTurnGenerationJob) string {
	h := hashJSON(struct {
		PromptClass      string `json:"prompt_class"`
		Route            string `json:"route"`
		Source           string `json:"source"`
		Backend          string `json:"backend"`
		Entrypoint       string `json:"entrypoint"`
		PromptFrame      string `json:"prompt_frame"`
		CandidateTrigger string `json:"candidate_trigger"`
		CandidateSeed    string `json:"candidate_seed"`
		TurnTextHash     string `json:"turn_text_hash"`
	}{
		PromptClass:      job.PromptClass,
		Route:            job.Route,
		Source:           job.Source,
		Backend:          job.Backend,
		Entrypoint:       job.Entrypoint,
		PromptFrame:      job.PromptFrame,
		CandidateTrigger: job.CandidateTrigger,
		CandidateSeed:    job.CandidateSeed,
		TurnTextHash:     job.TurnTextHash,
	})
	if h == "" {
		return ""
	}
	return "job-" + h
}

func recordAdmissionLiveRouteTurnGenerationJob(job admissionLiveRouteTurnGenerationJob) error {
	path := strings.TrimSpace(os.Getenv("AM_LIVE_ROUTE_TURN_GENERATION_JOB_LOG"))
	if path == "" {
		return nil
	}
	f, err := os.OpenFile(path, os.O_CREATE|os.O_APPEND|os.O_WRONLY, 0600)
	if err != nil {
		return err
	}
	enc := json.NewEncoder(f)
	err = enc.Encode(job)
	if closeErr := f.Close(); err == nil {
		err = closeErr
	}
	return err
}

func admissionLiveRouteTurnCandidateShellDryRun() bool {
	return dreamAdmissionBoolEnv("AM_LIVE_ROUTE_TURN_CANDIDATE_SHELL_DRY_RUN")
}

func admissionLiveRouteTurnCandidateShellForJob(job admissionLiveRouteTurnGenerationJob) admissionLiveRouteTurnCandidateShell {
	shell := admissionLiveRouteTurnCandidateShell{
		Schema:           admissionLiveRouteTurnCandidateShellSchema,
		PromptClass:      job.PromptClass,
		Route:            job.Route,
		Source:           job.Source,
		ExpectedSource:   job.ExpectedSource,
		Backend:          job.Backend,
		Entrypoint:       job.Entrypoint,
		PromptFrame:      job.PromptFrame,
		CandidateTrigger: job.CandidateTrigger,
		CandidateSeed:    job.CandidateSeed,
		JobID:            job.JobID,
		TurnTextHash:     job.TurnTextHash,
	}
	if job.Schema == "" {
		shell.Reason = "missing_generation_job"
		return shell
	}
	if !job.Passed {
		shell.Reason = "generation job failed"
		if job.Reason != "" {
			shell.Reason += ": " + job.Reason
		}
		return shell
	}
	if shell.JobID == "" {
		shell.Reason = "missing generation job id for route " + shell.Route + " prompt class " + shell.PromptClass
		return shell
	}
	if shell.Source == "" {
		shell.Reason = "missing candidate source for route " + shell.Route + " prompt class " + shell.PromptClass
		return shell
	}
	expectedSource := admissionLiveRouteSource(shell.Route)
	if shell.ExpectedSource == "" {
		shell.ExpectedSource = expectedSource
	}
	if shell.Source != expectedSource {
		shell.Reason = "source " + shell.Source + " does not match candidate route " + expectedSource + " for prompt class " + shell.PromptClass
		return shell
	}
	if shell.CandidateTrigger == "" {
		shell.Reason = "missing candidate trigger for route " + shell.Route + " prompt class " + shell.PromptClass
		return shell
	}
	if shell.CandidateSeed == "" {
		shell.Reason = "missing candidate seed for route " + shell.Route + " prompt class " + shell.PromptClass
		return shell
	}
	shell.CandidateSchema = "arianna.dream_candidate.v1"
	shell.CandidateKind = shell.Source
	shell.CandidateTextStatus = "pending_generation"
	shell.ShellID = admissionLiveRouteTurnCandidateShellID(shell)
	if shell.ShellID == "" {
		shell.Reason = "missing candidate shell id for route " + shell.Route + " prompt class " + shell.PromptClass
		return shell
	}
	shell.Passed = true
	return shell
}

func admissionLiveRouteTurnCandidateShellID(shell admissionLiveRouteTurnCandidateShell) string {
	h := hashJSON(struct {
		PromptClass      string `json:"prompt_class"`
		Route            string `json:"route"`
		Source           string `json:"source"`
		Backend          string `json:"backend"`
		Entrypoint       string `json:"entrypoint"`
		PromptFrame      string `json:"prompt_frame"`
		CandidateTrigger string `json:"candidate_trigger"`
		CandidateSeed    string `json:"candidate_seed"`
		JobID            string `json:"job_id"`
		TurnTextHash     string `json:"turn_text_hash"`
	}{
		PromptClass:      shell.PromptClass,
		Route:            shell.Route,
		Source:           shell.Source,
		Backend:          shell.Backend,
		Entrypoint:       shell.Entrypoint,
		PromptFrame:      shell.PromptFrame,
		CandidateTrigger: shell.CandidateTrigger,
		CandidateSeed:    shell.CandidateSeed,
		JobID:            shell.JobID,
		TurnTextHash:     shell.TurnTextHash,
	})
	if h == "" {
		return ""
	}
	return "shell-" + h
}

func recordAdmissionLiveRouteTurnCandidateShell(shell admissionLiveRouteTurnCandidateShell) error {
	path := strings.TrimSpace(os.Getenv("AM_LIVE_ROUTE_TURN_CANDIDATE_SHELL_LOG"))
	if path == "" {
		return nil
	}
	f, err := os.OpenFile(path, os.O_CREATE|os.O_APPEND|os.O_WRONLY, 0600)
	if err != nil {
		return err
	}
	enc := json.NewEncoder(f)
	err = enc.Encode(shell)
	if closeErr := f.Close(); err == nil {
		err = closeErr
	}
	return err
}

func admissionLiveRouteTurnCandidateReviewForDream(obs admissionLiveRouteTurnObservation, c dreamCandidate) admissionLiveRouteTurnCandidateReview {
	review := admissionLiveRouteTurnCandidateReview{
		Schema:             admissionLiveRouteTurnReviewSchema,
		Timing:             "async_subconscious",
		TurnPromptClass:    obs.PromptClass,
		TurnRoute:          obs.Route,
		TurnExpectedSource: obs.ExpectedSource,
		TurnPassed:         obs.Passed,
		CandidateRunID:     c.RunID,
		CandidateSource:    normalizeDreamAdmissionSource(c.Source),
		CandidateTrigger:   c.Trigger,
	}
	if obs.Schema == "" {
		review.Reason = "missing_turn_observation"
		return review
	}
	if !obs.Passed {
		review.Reason = "turn_route_failed"
		if obs.Reason != "" {
			review.Reason += ": " + obs.Reason
		}
		return review
	}
	if c.Schema == "" {
		review.Reason = "untyped_candidate"
		return review
	}
	choice, bridgeApplied, bridgeTrigger := admissionLiveRouteChoiceForCandidateWithTurnBridge(obs, c)
	review.CandidateBridgeApplied = bridgeApplied
	review.CandidateBridgeTrigger = bridgeTrigger
	if !bridgeApplied && c.Admission != nil && c.Admission.LiveRouteChoice != nil {
		choice = *c.Admission.LiveRouteChoice
	}
	review.CandidatePromptClass = choice.PromptClass
	review.CandidateRoute = choice.Route
	review.CandidateExpectedSource = choice.ExpectedSource
	review.CandidateChoicePassed = choice.Passed
	if !choice.Passed {
		review.Reason = "candidate_route_failed"
		if choice.Reason != "" {
			review.Reason += ": " + choice.Reason
		}
		return review
	}
	if review.CandidateSource != obs.ExpectedSource {
		review.Reason = "candidate_source_mismatch: source " + review.CandidateSource +
			" does not match turn expected " + obs.ExpectedSource + " for prompt class " + obs.PromptClass
		return review
	}
	if review.CandidateRoute != obs.Route {
		review.Reason = "candidate_route_mismatch: route " + review.CandidateRoute +
			" does not match turn route " + obs.Route + " for prompt class " + obs.PromptClass
		return review
	}
	review.Matched = true
	return review
}

func admissionLiveRouteChoiceForCandidateWithTurnBridge(obs admissionLiveRouteTurnObservation, c dreamCandidate) (admissionLiveRouteChoice, bool, string) {
	choiceCandidate := c
	if admissionLiveRouteTurnBridgeDryRun() {
		if bridged, ok := admissionLiveRouteTurnBridgeCandidate(obs, c); ok {
			choiceCandidate = bridged
			return admissionLiveRouteChoiceForCandidate(choiceCandidate), true, bridged.Trigger
		}
	}
	return admissionLiveRouteChoiceForCandidate(choiceCandidate), false, ""
}

func admissionLiveRouteTurnBridgeDryRun() bool {
	return dreamAdmissionBoolEnv("AM_LIVE_ROUTE_TURN_BRIDGE_DRY_RUN")
}

func admissionLiveRouteTurnBridgeCandidate(obs admissionLiveRouteTurnObservation, c dreamCandidate) (dreamCandidate, bool) {
	if !obs.Passed || obs.PromptClass == "" || !qloopSweepKnownPromptClass(obs.PromptClass) {
		return c, false
	}
	if normalizeDreamAdmissionSource(c.Source) != "nano" || strings.TrimSpace(c.Trigger) != "human-turn" {
		return c, false
	}
	c.Trigger = admissionLiveRouteTurnBridgeTrigger(obs.PromptClass)
	return c, true
}

func admissionLiveRouteTurnBridgeTrigger(promptClass string) string {
	promptClass = strings.TrimSpace(promptClass)
	if promptClass == "" {
		return "human-turn"
	}
	return "human-turn-" + promptClass
}

func recordAdmissionLiveRouteTurnCandidateReview(review admissionLiveRouteTurnCandidateReview) error {
	path := strings.TrimSpace(os.Getenv("AM_LIVE_ROUTE_TURN_REVIEW_LOG"))
	if path == "" {
		return nil
	}
	f, err := os.OpenFile(path, os.O_CREATE|os.O_APPEND|os.O_WRONLY, 0600)
	if err != nil {
		return err
	}
	enc := json.NewEncoder(f)
	err = enc.Encode(review)
	if closeErr := f.Close(); err == nil {
		err = closeErr
	}
	return err
}

func admissionLiveRoutePromptClasses() []string {
	return []string{
		"cold-reader",
		"direct-user",
		"format",
		"trauma",
		"recipient-lock",
		"polyphony",
		"identity",
		"qloop",
		"statement",
		"boundary",
		"self-reference",
		"outer-face",
		"memory",
		"dream",
		"repetition",
		"inner-world",
		"admission",
	}
}

func admissionLiveRouteForPromptClass(promptClass string) (string, bool) {
	switch qloopSweepPromptClass(promptClass, promptClass) {
	case "cold-reader", "direct-user", "format", "trauma":
		return "user_bridge", true
	case "recipient-lock":
		return "qloop_target", true
	case "polyphony":
		return "qloop_hint_qa", true
	case "identity", "qloop", "statement", "boundary", "self-reference", "outer-face", "memory":
		return "chorus", true
	case "dream", "repetition", "inner-world", "admission":
		return "direct", true
	default:
		return "", false
	}
}

func admissionLiveRouteSource(route string) string {
	return normalizeDreamAdmissionSource(route)
}
