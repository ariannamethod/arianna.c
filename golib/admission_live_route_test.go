package main

import (
	"reflect"
	"testing"
)

func TestAdmissionLiveRoutePlanMatchesBroadShadowReceipt(t *testing.T) {
	expected := map[string]string{
		"cold-reader":    "user_bridge",
		"direct-user":    "user_bridge",
		"format":         "user_bridge",
		"trauma":         "user_bridge",
		"recipient-lock": "qloop_target",
		"polyphony":      "qloop_hint_qa",
		"identity":       "chorus",
		"qloop":          "chorus",
		"statement":      "chorus",
		"boundary":       "chorus",
		"self-reference": "chorus",
		"outer-face":     "chorus",
		"memory":         "chorus",
		"dream":          "direct",
		"repetition":     "direct",
		"inner-world":    "direct",
		"admission":      "direct",
	}
	if len(admissionLiveRoutePromptClasses()) != len(expected) {
		t.Fatalf("live route class list length=%d, want %d", len(admissionLiveRoutePromptClasses()), len(expected))
	}
	for _, promptClass := range admissionLiveRoutePromptClasses() {
		if _, ok := expected[promptClass]; !ok {
			t.Fatalf("live route class list contains untested class %q", promptClass)
		}
	}
	for promptClass, wantRoute := range expected {
		plan := admissionLiveRoutePlanForPromptClass(promptClass)
		if !plan.Passed || plan.Schema != admissionLiveRoutePlanSchema || plan.PromptClass != promptClass || plan.Route != wantRoute {
			t.Fatalf("bad live route plan for %s: %+v", promptClass, plan)
		}
		if plan.Route == "qloop" {
			t.Fatalf("raw qloop must not be promoted by the live route plan: %+v", plan)
		}
		if !reflect.DeepEqual(plan.AllowedSources, []string{wantRoute}) {
			t.Fatalf("bad source gate for %s: %+v", promptClass, plan.AllowedSources)
		}
	}
}

func TestAdmissionLiveRoutePlanCoversBroadSamples(t *testing.T) {
	samples, err := loadAdmissionSamples("../samples/dream_admission_broad.jsonl")
	if err != nil {
		t.Fatal(err)
	}
	if len(samples) == 0 {
		t.Fatal("broad samples missing")
	}
	for _, sample := range samples {
		promptClass := qloopSweepPromptClass(sample.Trigger, sample.Seed)
		plan := admissionLiveRoutePlanForPromptClass(promptClass)
		if !plan.Passed {
			t.Fatalf("broad sample has no live route plan: trigger=%s seed=%s class=%s plan=%+v", sample.Trigger, sample.Seed, promptClass, plan)
		}
		if plan.Route == "qloop" {
			t.Fatalf("raw qloop route leaked into live plan: trigger=%s seed=%s plan=%+v", sample.Trigger, sample.Seed, plan)
		}
	}
}

func TestAdmissionLiveRoutePlanFailsClosedForUnknownClass(t *testing.T) {
	plan := admissionLiveRoutePlanForPromptClass("unknown-pressure")
	if plan.Passed || plan.Route != "" || plan.Reason != "unknown_prompt_class" {
		t.Fatalf("unknown prompt class should fail closed: %+v", plan)
	}
}

func TestAdmissionLiveRouteChoiceForCandidate(t *testing.T) {
	cases := []struct {
		name           string
		source         string
		trigger        string
		seed           string
		wantPrompt     string
		wantRoute      string
		wantExpected   string
		wantPassed     bool
		wantReason     string
		wantPlanPassed bool
	}{
		{
			name:           "matched chorus identity",
			source:         "chorus",
			trigger:        "identity",
			seed:           "seed",
			wantPrompt:     "identity",
			wantRoute:      "chorus",
			wantExpected:   "chorus",
			wantPassed:     true,
			wantPlanPassed: true,
		},
		{
			name:           "wrong source",
			source:         "direct",
			trigger:        "identity",
			seed:           "seed",
			wantPrompt:     "identity",
			wantRoute:      "chorus",
			wantExpected:   "chorus",
			wantPassed:     false,
			wantReason:     "source direct does not match live route chorus for prompt class identity",
			wantPlanPassed: true,
		},
		{
			name:           "missing source",
			source:         "",
			trigger:        "identity",
			seed:           "seed",
			wantPrompt:     "identity",
			wantRoute:      "chorus",
			wantExpected:   "chorus",
			wantPassed:     false,
			wantReason:     "missing source for live route plan chorus prompt class identity",
			wantPlanPassed: true,
		},
		{
			name:           "unknown class",
			source:         "chorus",
			trigger:        "unknown-pressure",
			seed:           "seed",
			wantPrompt:     "unknown-pressure",
			wantPassed:     false,
			wantReason:     "live route plan failed: unknown_prompt_class",
			wantPlanPassed: false,
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			choice := admissionLiveRouteChoiceForCandidate(newDreamCandidate(tc.source, tc.trigger, tc.seed, "", "I am Arianna.", nil))
			if choice.Schema != admissionLiveRouteChoiceSchema || choice.PromptClass != tc.wantPrompt ||
				choice.Route != tc.wantRoute || choice.ExpectedSource != tc.wantExpected ||
				choice.Passed != tc.wantPassed || choice.Reason != tc.wantReason ||
				choice.Plan.Passed != tc.wantPlanPassed {
				t.Fatalf("bad live route choice: %+v", choice)
			}
			if choice.Plan.Schema != admissionLiveRoutePlanSchema || choice.Plan.PromptClass != tc.wantPrompt {
				t.Fatalf("choice did not carry normalized plan: %+v", choice.Plan)
			}
		})
	}
}

func TestAdmissionLiveRouteTurnObservationForHuman(t *testing.T) {
	cases := []struct {
		name         string
		human        string
		wantClass    string
		wantRoute    string
		wantExpected string
		wantPassed   bool
	}{
		{
			name:         "identity",
			human:        "Who are you?",
			wantClass:    "identity",
			wantRoute:    "chorus",
			wantExpected: "chorus",
			wantPassed:   true,
		},
		{
			name:         "cold reader",
			human:        "Please answer without assuming we have met before.",
			wantClass:    "cold-reader",
			wantRoute:    "user_bridge",
			wantExpected: "user_bridge",
			wantPassed:   true,
		},
		{
			name:         "recipient lock",
			human:        "The recipient is not Oleg; answer as if to another person.",
			wantClass:    "recipient-lock",
			wantRoute:    "qloop_target",
			wantExpected: "qloop_target",
			wantPassed:   true,
		},
		{
			name:         "format",
			human:        "Explain the prompt format and chat token wrapper.",
			wantClass:    "format",
			wantRoute:    "user_bridge",
			wantExpected: "user_bridge",
			wantPassed:   true,
		},
		{
			name:       "unknown",
			human:      "hello",
			wantClass:  "unknown",
			wantPassed: false,
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			obs := admissionLiveRouteTurnObservationForHuman(tc.human)
			if obs.Schema != admissionLiveRouteTurnObservationSchema || obs.PromptClass != tc.wantClass ||
				obs.Route != tc.wantRoute || obs.ExpectedSource != tc.wantExpected || obs.Passed != tc.wantPassed {
				t.Fatalf("bad turn observation: %+v", obs)
			}
			if obs.TextHash == "" {
				t.Fatalf("turn observation should carry text hash: %+v", obs)
			}
			if tc.wantPassed && (obs.Plan.Schema != admissionLiveRoutePlanSchema || !obs.Plan.Passed) {
				t.Fatalf("turn observation did not carry passed plan: %+v", obs.Plan)
			}
		})
	}
}
