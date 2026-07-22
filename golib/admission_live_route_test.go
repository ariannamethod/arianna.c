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
