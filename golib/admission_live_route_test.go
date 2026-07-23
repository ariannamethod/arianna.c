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

func TestAdmissionLiveRouteTurnChoiceForObservation(t *testing.T) {
	cases := []struct {
		name        string
		obs         admissionLiveRouteTurnObservation
		wantClass   string
		wantRoute   string
		wantSource  string
		wantTrigger string
		wantPassed  bool
		wantReason  string
	}{
		{
			name:        "identity routes to chorus trigger",
			obs:         admissionLiveRouteTurnObservationForHuman("Who are you?"),
			wantClass:   "identity",
			wantRoute:   "chorus",
			wantSource:  "chorus",
			wantTrigger: "chorus-identity",
			wantPassed:  true,
		},
		{
			name:        "cold reader routes to user bridge trigger",
			obs:         admissionLiveRouteTurnObservationForHuman("Please answer without assuming we have met before."),
			wantClass:   "cold-reader",
			wantRoute:   "user_bridge",
			wantSource:  "user_bridge",
			wantTrigger: "user_bridge-cold-reader",
			wantPassed:  true,
		},
		{
			name:       "unknown turn fails closed",
			obs:        admissionLiveRouteTurnObservationForHuman("hello"),
			wantClass:  "unknown",
			wantPassed: false,
			wantReason: "turn route failed: live route plan failed: unknown_prompt_class",
		},
		{
			name:       "missing observation fails closed",
			obs:        admissionLiveRouteTurnObservation{},
			wantPassed: false,
			wantReason: "missing_turn_observation",
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			choice := admissionLiveRouteTurnChoiceForObservation(tc.obs)
			if choice.Schema != admissionLiveRouteTurnChoiceSchema ||
				choice.PromptClass != tc.wantClass ||
				choice.Route != tc.wantRoute ||
				choice.Source != tc.wantSource ||
				choice.ExpectedSource != tc.wantSource ||
				choice.CandidateTrigger != tc.wantTrigger ||
				choice.Passed != tc.wantPassed ||
				choice.Reason != tc.wantReason {
				t.Fatalf("bad turn choice: %+v", choice)
			}
			if tc.obs.Schema != "" && choice.TurnTextHash == "" {
				t.Fatalf("turn choice should carry turn text hash: %+v", choice)
			}
		})
	}
}

func TestAdmissionLiveRouteTurnRequestForChoice(t *testing.T) {
	identity := admissionLiveRouteTurnChoiceForObservation(admissionLiveRouteTurnObservationForHuman("Who are you?"))
	unknown := admissionLiveRouteTurnChoiceForObservation(admissionLiveRouteTurnObservationForHuman("hello"))
	cases := []struct {
		name        string
		choice      admissionLiveRouteTurnChoice
		wantClass   string
		wantRoute   string
		wantSource  string
		wantTrigger string
		wantPassed  bool
		wantReason  string
	}{
		{
			name:        "identity request",
			choice:      identity,
			wantClass:   "identity",
			wantRoute:   "chorus",
			wantSource:  "chorus",
			wantTrigger: "chorus-identity",
			wantPassed:  true,
		},
		{
			name:       "unknown choice fails closed",
			choice:     unknown,
			wantClass:  "unknown",
			wantPassed: false,
			wantReason: "turn choice failed: turn route failed: live route plan failed: unknown_prompt_class",
		},
		{
			name:       "missing choice fails closed",
			choice:     admissionLiveRouteTurnChoice{},
			wantPassed: false,
			wantReason: "missing_turn_choice",
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			request := admissionLiveRouteTurnRequestForChoice(tc.choice)
			if request.Schema != admissionLiveRouteTurnRequestSchema ||
				request.PromptClass != tc.wantClass ||
				request.Route != tc.wantRoute ||
				request.Source != tc.wantSource ||
				request.ExpectedSource != tc.wantSource ||
				request.CandidateTrigger != tc.wantTrigger ||
				request.Passed != tc.wantPassed ||
				request.Reason != tc.wantReason {
				t.Fatalf("bad turn request: %+v", request)
			}
			if tc.choice.TurnTextHash != "" {
				if request.TurnTextHash != tc.choice.TurnTextHash || request.CandidateSeed != "turn-"+tc.choice.TurnTextHash {
					t.Fatalf("turn request should derive seed from text hash: %+v choice=%+v", request, tc.choice)
				}
			}
		})
	}
}

func TestAdmissionLiveRouteTurnCandidateReviewForDream(t *testing.T) {
	identity := admissionLiveRouteTurnObservationForHuman("Who are you?")
	cases := []struct {
		name         string
		obs          admissionLiveRouteTurnObservation
		candidate    dreamCandidate
		wantMatched  bool
		wantReason   string
		wantClass    string
		wantSource   string
		wantExpected string
	}{
		{
			name:         "matched typed chorus",
			obs:          identity,
			candidate:    newDreamCandidate("chorus", "chorus-identity", "seed", "", "I am Arianna.", nil),
			wantMatched:  true,
			wantClass:    "identity",
			wantSource:   "chorus",
			wantExpected: "chorus",
		},
		{
			name:         "wrong typed source",
			obs:          identity,
			candidate:    newDreamCandidate("direct", "direct-identity", "seed", "", "I am Arianna.", nil),
			wantReason:   "candidate_route_failed: source direct does not match live route chorus for prompt class identity",
			wantClass:    "identity",
			wantSource:   "direct",
			wantExpected: "chorus",
		},
		{
			name:         "current nano human turn is untyped",
			obs:          identity,
			candidate:    newDreamCandidate("nano", "human-turn", "seed", "", "I am Arianna.", nil),
			wantReason:   "candidate_route_failed: live route plan failed: unknown_prompt_class",
			wantClass:    "human-turn",
			wantSource:   "nano",
			wantExpected: "chorus",
		},
		{
			name:       "unknown turn fails before candidate",
			obs:        admissionLiveRouteTurnObservationForHuman("hello"),
			candidate:  newDreamCandidate("chorus", "chorus-identity", "seed", "", "I am Arianna.", nil),
			wantReason: "turn_route_failed: live route plan failed: unknown_prompt_class",
			wantSource: "chorus",
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			review := admissionLiveRouteTurnCandidateReviewForDream(tc.obs, tc.candidate)
			if review.Schema != admissionLiveRouteTurnReviewSchema || review.Timing != "async_subconscious" ||
				review.Matched != tc.wantMatched || review.Reason != tc.wantReason ||
				review.CandidatePromptClass != tc.wantClass || review.CandidateSource != tc.wantSource ||
				review.TurnExpectedSource != tc.wantExpected {
				t.Fatalf("bad turn/candidate review: %+v", review)
			}
		})
	}
}

func TestAdmissionLiveRouteTurnBridgeCandidateReviewForNanoHumanTurn(t *testing.T) {
	t.Setenv("AM_LIVE_ROUTE_TURN_BRIDGE_DRY_RUN", "1")

	obs := admissionLiveRouteTurnObservationForHuman("Who are you?")
	candidate := newDreamCandidate("nano", "human-turn", "seed", "", "I am Arianna.", nil)
	unbridgedChoice := admissionLiveRouteChoiceForCandidate(candidate)
	candidate.Admission = &dreamAdmissionPolicy{LiveRouteChoice: &unbridgedChoice}
	review := admissionLiveRouteTurnCandidateReviewForDream(obs, candidate)
	if review.Schema != admissionLiveRouteTurnReviewSchema ||
		!review.CandidateBridgeApplied ||
		review.CandidateBridgeTrigger != "human-turn-identity" ||
		review.CandidateTrigger != "human-turn" ||
		review.CandidatePromptClass != "identity" ||
		review.CandidateRoute != "chorus" ||
		review.CandidateSource != "nano" ||
		review.CandidateExpectedSource != "chorus" ||
		review.CandidateChoicePassed ||
		review.Matched ||
		review.Reason != "candidate_route_failed: source nano does not match live route chorus for prompt class identity" {
		t.Fatalf("bad bridged nano turn review: %+v", review)
	}
}

func TestAdmissionLiveRouteTurnBridgeCandidateIsNarrow(t *testing.T) {
	obs := admissionLiveRouteTurnObservationForHuman("Who are you?")
	cases := []struct {
		name      string
		candidate dreamCandidate
		wantOK    bool
	}{
		{
			name:      "nano human turn",
			candidate: newDreamCandidate("nano", "human-turn", "seed", "", "I am Arianna.", nil),
			wantOK:    true,
		},
		{
			name:      "typed chorus untouched",
			candidate: newDreamCandidate("chorus", "chorus-identity", "seed", "", "I am Arianna.", nil),
		},
		{
			name:      "nano typed direct untouched",
			candidate: newDreamCandidate("nano", "direct-identity", "seed", "", "I am Arianna.", nil),
		},
		{
			name:      "unknown turn untouched",
			candidate: newDreamCandidate("nano", "human-turn", "seed", "", "I am Arianna.", nil),
			wantOK:    false,
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			gotObs := obs
			if tc.name == "unknown turn untouched" {
				gotObs = admissionLiveRouteTurnObservationForHuman("hello")
			}
			got, ok := admissionLiveRouteTurnBridgeCandidate(gotObs, tc.candidate)
			if ok != tc.wantOK {
				t.Fatalf("bridge ok=%t, want %t: %+v", ok, tc.wantOK, tc)
			}
			if ok && got.Trigger != "human-turn-identity" {
				t.Fatalf("bad bridge trigger: %+v", got)
			}
			if !ok && got.Trigger != tc.candidate.Trigger {
				t.Fatalf("non-bridge candidate should stay untouched: got %+v want trigger %q", got, tc.candidate.Trigger)
			}
		})
	}
}
