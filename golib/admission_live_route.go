package main

import (
	"encoding/json"
	"os"
	"strings"
)

const (
	admissionLiveRoutePlanSchema            = "arianna.live_route_plan.v1"
	admissionLiveRouteChoiceSchema          = "arianna.live_route_choice.v1"
	admissionLiveRouteTurnObservationSchema = "arianna.live_route_turn_observation.v1"
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
