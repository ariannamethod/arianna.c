package main

import (
	"encoding/json"
	"os"
	"path/filepath"
	"reflect"
	"strings"
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

func TestAdmissionLiveRouteTurnGenerationJobForRequest(t *testing.T) {
	requestFor := func(human string) admissionLiveRouteTurnRequest {
		obs := admissionLiveRouteTurnObservationForHuman(human)
		choice := admissionLiveRouteTurnChoiceForObservation(obs)
		return admissionLiveRouteTurnRequestForChoice(choice)
	}
	identity := requestFor("Who are you?")
	wrongSource := identity
	wrongSource.Source = "direct"
	cases := []struct {
		name          string
		request       admissionLiveRouteTurnRequest
		wantClass     string
		wantRoute     string
		wantSource    string
		wantBackend   string
		wantEntry     string
		wantFrame     string
		wantPassed    bool
		wantReason    string
		wantJobPrefix string
	}{
		{
			name:          "identity dispatches to chorus field",
			request:       identity,
			wantClass:     "identity",
			wantRoute:     "chorus",
			wantSource:    "chorus",
			wantBackend:   "chorus-arianna",
			wantEntry:     "field",
			wantFrame:     "q_a",
			wantPassed:    true,
			wantJobPrefix: "job-",
		},
		{
			name:          "dream dispatches to direct nano",
			request:       requestFor("Tell me what the dream should remember."),
			wantClass:     "dream",
			wantRoute:     "direct",
			wantSource:    "direct",
			wantBackend:   "nano-arianna",
			wantEntry:     "direct",
			wantFrame:     "q_a",
			wantPassed:    true,
			wantJobPrefix: "job-",
		},
		{
			name:          "recipient lock dispatches to qloop target",
			request:       requestFor("The recipient is not Oleg; answer as if to another person."),
			wantClass:     "recipient-lock",
			wantRoute:     "qloop_target",
			wantSource:    "qloop_target",
			wantBackend:   "chorus-arianna",
			wantEntry:     "qloop_target",
			wantFrame:     "user_arianna_target",
			wantPassed:    true,
			wantJobPrefix: "job-",
		},
		{
			name:          "cold reader dispatches to user bridge",
			request:       requestFor("Please answer without assuming we have met before."),
			wantClass:     "cold-reader",
			wantRoute:     "user_bridge",
			wantSource:    "user_bridge",
			wantBackend:   "chorus-arianna",
			wantEntry:     "repl_user_bridge",
			wantFrame:     "user_arianna",
			wantPassed:    true,
			wantJobPrefix: "job-",
		},
		{
			name:       "unknown request fails before dispatch",
			request:    requestFor("hello"),
			wantClass:  "unknown",
			wantPassed: false,
			wantReason: "turn request failed: turn choice failed: turn route failed: live route plan failed: unknown_prompt_class",
		},
		{
			name:       "missing request fails closed",
			request:    admissionLiveRouteTurnRequest{},
			wantPassed: false,
			wantReason: "missing_turn_request",
		},
		{
			name:        "wrong source fails route bounded",
			request:     wrongSource,
			wantClass:   "identity",
			wantRoute:   "chorus",
			wantSource:  "direct",
			wantBackend: "chorus-arianna",
			wantEntry:   "field",
			wantFrame:   "q_a",
			wantPassed:  false,
			wantReason:  "source direct does not match generation route chorus for prompt class identity",
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			job := admissionLiveRouteTurnGenerationJobForRequest(tc.request)
			if job.Schema != admissionLiveRouteTurnGenerationJobSchema ||
				job.PromptClass != tc.wantClass ||
				job.Route != tc.wantRoute ||
				job.Source != tc.wantSource ||
				job.Backend != tc.wantBackend ||
				job.Entrypoint != tc.wantEntry ||
				job.PromptFrame != tc.wantFrame ||
				job.Passed != tc.wantPassed ||
				job.Reason != tc.wantReason {
				t.Fatalf("bad generation job: %+v", job)
			}
			if tc.wantJobPrefix != "" && !strings.HasPrefix(job.JobID, tc.wantJobPrefix) {
				t.Fatalf("generation job should have stable id: %+v", job)
			}
			if !tc.wantPassed && job.JobID != "" {
				t.Fatalf("failed generation job should not name a runnable job id: %+v", job)
			}
		})
	}
}

func TestAdmissionLiveRouteTurnGenerationJobInventoryGate(t *testing.T) {
	setBodyInventoryDefaultEnv(t)
	requestFor := func(human string) admissionLiveRouteTurnRequest {
		obs := admissionLiveRouteTurnObservationForHuman(human)
		choice := admissionLiveRouteTurnChoiceForObservation(obs)
		return admissionLiveRouteTurnRequestForChoice(choice)
	}

	readyRoot := t.TempDir()
	writeRequiredBodyInventoryFiles(t, readyRoot)
	writeOptionalBodyInventoryFiles(t, readyRoot)
	ready := inspectBodyInventory(readyRoot)
	readyJob := admissionLiveRouteTurnGenerationJobForRequestWithInventory(requestFor("Who are you?"), ready, true)
	if !readyJob.Passed ||
		readyJob.RouteAvailabilityStatus != "available" ||
		readyJob.BodyInventoryStatus != "ready" ||
		readyJob.RouteAvailabilityReason != "" ||
		len(readyJob.RouteMissingOrgans) != 0 ||
		!strings.HasPrefix(readyJob.JobID, "job-") {
		t.Fatalf("ready inventory should allow generation job: %+v", readyJob)
	}

	emptyRoot := t.TempDir()
	blocked := inspectBodyInventory(emptyRoot)
	blockedJob := admissionLiveRouteTurnGenerationJobForRequestWithInventory(requestFor("Who are you?"), blocked, true)
	if blockedJob.Passed ||
		blockedJob.JobID != "" ||
		blockedJob.BodyInventoryStatus != "blocked" ||
		blockedJob.RouteAvailabilityStatus != "unavailable" ||
		!strings.Contains(blockedJob.Reason, "route chorus unavailable in body inventory") ||
		!sameStrings(blockedJob.RouteMissingOrgans, []string{"chorus-binary", "nano-weight"}) {
		t.Fatalf("blocked inventory should deny chorus generation job: %+v", blockedJob)
	}

	missingInventoryJob := admissionLiveRouteTurnGenerationJobForRequestWithInventory(requestFor("Who are you?"), bodyInventoryReceipt{}, true)
	if missingInventoryJob.Passed ||
		missingInventoryJob.JobID != "" ||
		missingInventoryJob.RouteAvailabilityStatus != "missing_body_inventory" ||
		!strings.Contains(missingInventoryJob.Reason, "body inventory missing") {
		t.Fatalf("missing inventory should fail closed: %+v", missingInventoryJob)
	}
}

func TestAdmissionLiveRouteTurnRouteBoundaryReceipts(t *testing.T) {
	setBodyInventoryDefaultEnv(t)
	obs := admissionLiveRouteTurnObservationForHuman("Who are you?")
	choice := admissionLiveRouteTurnChoiceForObservation(obs)
	request := admissionLiveRouteTurnRequestForChoice(choice)
	inventory := inspectBodyInventory(t.TempDir())
	job := admissionLiveRouteTurnGenerationJobForRequestWithInventory(request, inventory, true)
	shell := admissionLiveRouteTurnCandidateShellForJob(job)
	execution := admissionLiveRouteTurnCandidateExecutionForShell(shell, "This text must not run without the route body.")
	adapterFromShell := admissionLiveRouteTurnGeneratorAdapterForShell(shell, "This text must not adapt without the route body.")
	adapterFromExecution := admissionLiveRouteTurnGeneratorAdapterForExecution(execution)

	if job.Passed || job.JobID != "" {
		t.Fatalf("job should fail closed before runnable id: %+v", job)
	}
	if !sameStrings(job.RouteMissingOrgans, []string{"chorus-binary", "nano-weight"}) {
		t.Fatalf("job should name missing organs: %+v", job)
	}
	if shell.Passed ||
		shell.JobID != "" ||
		shell.ShellID != "" ||
		shell.BodyInventoryStatus != "blocked" ||
		shell.RouteAvailabilityStatus != "unavailable" ||
		shell.RouteAvailabilityReason != "missing_route_organs:chorus-binary,nano-weight" ||
		!sameStrings(shell.RouteMissingOrgans, job.RouteMissingOrgans) ||
		!strings.Contains(shell.Reason, "route chorus unavailable in body inventory") {
		t.Fatalf("shell should carry the unavailable route boundary: %+v", shell)
	}
	if execution.Passed ||
		execution.JobID != "" ||
		execution.ShellID != "" ||
		execution.ExecutionID != "" ||
		execution.BodyInventoryStatus != shell.BodyInventoryStatus ||
		execution.RouteAvailabilityStatus != shell.RouteAvailabilityStatus ||
		execution.RouteAvailabilityReason != shell.RouteAvailabilityReason ||
		!sameStrings(execution.RouteMissingOrgans, shell.RouteMissingOrgans) ||
		!strings.Contains(execution.Reason, "route chorus unavailable in body inventory") {
		t.Fatalf("execution should carry the unavailable route boundary: %+v", execution)
	}
	for name, adapter := range map[string]admissionLiveRouteTurnGeneratorAdapter{
		"from shell":     adapterFromShell,
		"from execution": adapterFromExecution,
	} {
		if adapter.Passed ||
			adapter.JobID != "" ||
			adapter.ShellID != "" ||
			adapter.CandidateExecutionID != "" ||
			adapter.AdapterID != "" ||
			adapter.BodyInventoryStatus != shell.BodyInventoryStatus ||
			adapter.RouteAvailabilityStatus != shell.RouteAvailabilityStatus ||
			adapter.RouteAvailabilityReason != shell.RouteAvailabilityReason ||
			!sameStrings(adapter.RouteMissingOrgans, shell.RouteMissingOrgans) ||
			!strings.Contains(adapter.Reason, "route chorus unavailable in body inventory") {
			t.Fatalf("adapter %s should carry the unavailable route boundary: %+v", name, adapter)
		}
	}
}

func TestAdmissionLiveRouteBoundaryReportExpectedStageChain(t *testing.T) {
	want := []string{
		"rollback_implementation",
		"ledger_implementation",
		"ledger_persistence",
		"ledger_verification",
		"admission_readiness",
		"admission_permit",
		"admission_seal",
		"final_gate",
		"resonance_intent",
		"resonance_receiver",
		"resonance_observation",
		"resonance_graft_boundary",
		"resonance_graft_preflight",
		"resonance_graft_gate",
		"resonance_graft_candidate",
		"resonance_graft_candidate_store",
		"resonance_graft_candidate_store_reader",
		"resonance_graft_admission_proof",
	}
	got := admissionLiveRouteBoundaryReportExpectedStageNames()
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("unexpected boundary report stage chain: got=%v want=%v", got, want)
	}
	seen := make(map[string]struct{})
	for _, name := range got {
		if _, ok := seen[name]; ok {
			t.Fatalf("duplicated expected boundary report stage: %s", name)
		}
		seen[name] = struct{}{}
	}
	got[0] = "mutated"
	if fresh := admissionLiveRouteBoundaryReportExpectedStageNames(); fresh[0] != "rollback_implementation" {
		t.Fatalf("expected stage chain must be returned as a fresh slice: %v", fresh)
	}
	if !admissionLiveRouteBoundaryReportStageChainMatchesPrefix(want[:8], want) {
		t.Fatal("expected final-gate prefix to match expected chain")
	}
	if admissionLiveRouteBoundaryReportStageChainMatchesPrefix([]string{"ledger_implementation"}, want) {
		t.Fatal("out-of-order boundary report prefix should fail")
	}
	if admissionLiveRouteBoundaryReportStageChainMatchesPrefix(append(append([]string{}, want...), "extra_stage"), want) {
		t.Fatal("boundary report chain longer than expected should fail")
	}
}

func TestAdmissionLiveRouteBoundaryReportStageNamesMirrorBuilder(t *testing.T) {
	boundary := admissionLiveRouteBoundaryProjection("complete", "available", "route ready", nil)
	inputs := []admissionLiveRouteBoundaryReportInput{
		{
			Enabled:                 true,
			Name:                    " final_gate ",
			BodyInventoryStatus:     "complete",
			RouteAvailabilityStatus: "available",
			RouteAvailabilityReason: "route ready",
		},
		{
			Enabled:                 false,
			Name:                    "disabled_stage",
			BodyInventoryStatus:     "blocked",
			RouteAvailabilityStatus: "unavailable",
			RouteAvailabilityReason: "disabled",
		},
		{
			Enabled:                 true,
			Name:                    "",
			BodyInventoryStatus:     "complete",
			RouteAvailabilityStatus: "available",
			RouteAvailabilityReason: "route ready",
		},
		{
			Enabled:                 true,
			Name:                    "resonance_graft_admission_proof",
			BodyInventoryStatus:     "complete",
			RouteAvailabilityStatus: "available",
			RouteAvailabilityReason: "route ready",
		},
	}
	want := []string{"final_gate", "unknown", "resonance_graft_admission_proof"}
	if got := admissionLiveRouteBoundaryReportInputStageNames(inputs); !reflect.DeepEqual(got, want) {
		t.Fatalf("input stage names should mirror builder naming: got=%v want=%v", got, want)
	}
	report := buildAdmissionLiveRouteBoundaryReport(boundary, inputs)
	if got := admissionLiveRouteBoundaryReportStageNames(report); !reflect.DeepEqual(got, want) {
		t.Fatalf("report stage names should mirror input helper: got=%v want=%v", got, want)
	}
	if !report.Passed || report.ReceiptsChecked != len(want) {
		t.Fatalf("expected report to pass after stage-name extraction: %+v", report)
	}
}

func TestAdmissionLiveRouteBoundaryReportProjectsAndCatchesDrift(t *testing.T) {
	missing := []string{"goldie-weight"}
	boundary := admissionLiveRouteBoundaryProjection(
		"degraded",
		"available",
		"optional_route_organs_missing:goldie-weight",
		missing,
	)
	missing[0] = "mutated"
	if !reflect.DeepEqual(boundary.RouteMissingOrgans, []string{"goldie-weight"}) {
		t.Fatalf("boundary should copy missing organs: %+v", boundary)
	}

	passed := buildAdmissionLiveRouteBoundaryReport(boundary, []admissionLiveRouteBoundaryReportInput{
		{
			Enabled:                 true,
			Name:                    "decision",
			BodyInventoryStatus:     "degraded",
			RouteAvailabilityStatus: "available",
			RouteAvailabilityReason: "optional_route_organs_missing:goldie-weight",
			RouteMissingOrgans:      []string{"goldie-weight"},
		},
		{
			Enabled:                 false,
			Name:                    "disabled_bad",
			BodyInventoryStatus:     "blocked",
			RouteAvailabilityStatus: "unavailable",
			RouteAvailabilityReason: "missing_route_organs:nano-weight",
			RouteMissingOrgans:      []string{"nano-weight"},
		},
	})
	if passed.Schema != admissionLiveRouteBoundaryReportSchema ||
		!passed.Passed ||
		passed.ReceiptsChecked != 1 ||
		len(passed.Stages) != 1 ||
		!passed.Stages[0].Passed ||
		len(passed.Reasons) != 0 {
		t.Fatalf("expected passing one-stage boundary report: %+v", passed)
	}

	duplicated := buildAdmissionLiveRouteBoundaryReport(boundary, []admissionLiveRouteBoundaryReportInput{
		{
			Enabled:                 true,
			Name:                    "decision",
			BodyInventoryStatus:     "degraded",
			RouteAvailabilityStatus: "available",
			RouteAvailabilityReason: "optional_route_organs_missing:goldie-weight",
			RouteMissingOrgans:      []string{"goldie-weight"},
		},
		{
			Enabled:                 true,
			Name:                    "decision",
			BodyInventoryStatus:     "degraded",
			RouteAvailabilityStatus: "available",
			RouteAvailabilityReason: "optional_route_organs_missing:goldie-weight",
			RouteMissingOrgans:      []string{"goldie-weight"},
		},
	})
	if duplicated.Passed ||
		duplicated.ReceiptsChecked != 2 ||
		len(duplicated.Stages) != 2 ||
		!duplicated.Stages[0].Passed ||
		!duplicated.Stages[1].Passed ||
		!reflect.DeepEqual(duplicated.Reasons, []string{"duplicate_stage:decision"}) {
		t.Fatalf("expected duplicated stage report to fail closed: %+v", duplicated)
	}

	path := filepath.Join(t.TempDir(), "boundary-report.json")
	if err := writeAdmissionLiveRouteBoundaryReport(path, passed); err != nil {
		t.Fatalf("write boundary report: %v", err)
	}
	raw, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("read boundary report: %v", err)
	}
	if !strings.Contains(string(raw), `"schema": "arianna.live_route_boundary_report.v1"`) ||
		!strings.Contains(string(raw), `"name": "decision"`) {
		t.Fatalf("boundary report JSON missing expected fields: %s", raw)
	}

	failed := buildAdmissionLiveRouteBoundaryReport(boundary, []admissionLiveRouteBoundaryReportInput{
		{
			Enabled:                 true,
			Name:                    "resonance_graft_admission_proof",
			BodyInventoryStatus:     "degraded",
			RouteAvailabilityStatus: "available",
			RouteAvailabilityReason: "optional_route_organs_missing:goldie-weight",
		},
	})
	if failed.Passed ||
		failed.ReceiptsChecked != 1 ||
		len(failed.Stages) != 1 ||
		failed.Stages[0].Passed ||
		!reflect.DeepEqual(failed.Stages[0].Mismatches, []string{"route_missing_organs"}) ||
		!reflect.DeepEqual(failed.Reasons, []string{"boundary_mismatch:resonance_graft_admission_proof"}) {
		t.Fatalf("expected report to catch missing-organ drift: %+v", failed)
	}

	fieldDrift := buildAdmissionLiveRouteBoundaryReport(boundary, []admissionLiveRouteBoundaryReportInput{
		{
			Enabled:                 true,
			Name:                    "writer_receipt",
			BodyInventoryStatus:     "blocked",
			RouteAvailabilityStatus: "unavailable",
			RouteAvailabilityReason: "missing_route_organs:nano-weight",
			RouteMissingOrgans:      []string{"nano-weight"},
		},
	})
	if fieldDrift.Passed ||
		fieldDrift.ReceiptsChecked != 1 ||
		len(fieldDrift.Stages) != 1 ||
		fieldDrift.Stages[0].Passed ||
		!reflect.DeepEqual(fieldDrift.Stages[0].Mismatches, []string{
			"body_inventory_status",
			"route_availability_status",
			"route_availability_reason",
			"route_missing_organs",
		}) ||
		!reflect.DeepEqual(fieldDrift.Reasons, []string{"boundary_mismatch:writer_receipt"}) {
		t.Fatalf("expected report to name each boundary field drift: %+v", fieldDrift)
	}
	failedPath := filepath.Join(t.TempDir(), "boundary-report-failed.json")
	if err := writeAdmissionLiveRouteBoundaryReport(failedPath, fieldDrift); err != nil {
		t.Fatalf("write failed boundary report: %v", err)
	}
	failedRaw, err := os.ReadFile(failedPath)
	if err != nil {
		t.Fatalf("read failed boundary report: %v", err)
	}
	failedText := string(failedRaw)
	if !strings.Contains(failedText, `"mismatches":`) ||
		!strings.Contains(failedText, `"body_inventory_status"`) ||
		!strings.Contains(failedText, `"route_missing_organs"`) {
		t.Fatalf("failed boundary report JSON missing mismatch diagnostics: %s", failedRaw)
	}
	var decodedFailed admissionLiveRouteBoundaryReport
	if err := json.Unmarshal(failedRaw, &decodedFailed); err != nil {
		t.Fatalf("unmarshal failed boundary report: %v", err)
	}
	if len(decodedFailed.Stages) != 1 ||
		!reflect.DeepEqual(decodedFailed.Stages[0].Mismatches, fieldDrift.Stages[0].Mismatches) ||
		!reflect.DeepEqual(decodedFailed.Reasons, fieldDrift.Reasons) {
		t.Fatalf("failed boundary report diagnostics did not round-trip: %+v", decodedFailed)
	}

	empty := buildAdmissionLiveRouteBoundaryReport(boundary, nil)
	if empty.Passed ||
		empty.ReceiptsChecked != 0 ||
		len(empty.Stages) != 0 ||
		!reflect.DeepEqual(empty.Reasons, []string{"no_receipts_checked"}) {
		t.Fatalf("expected empty report to fail closed: %+v", empty)
	}

	disabledOnly := buildAdmissionLiveRouteBoundaryReport(boundary, []admissionLiveRouteBoundaryReportInput{
		{
			Enabled: false,
			Name:    "disabled_only",
		},
	})
	if disabledOnly.Passed ||
		disabledOnly.ReceiptsChecked != 0 ||
		len(disabledOnly.Stages) != 0 ||
		!reflect.DeepEqual(disabledOnly.Reasons, []string{"no_receipts_checked"}) {
		t.Fatalf("expected disabled-only report to fail closed: %+v", disabledOnly)
	}
}

func TestAdmissionLiveRouteTurnCandidateShellForJob(t *testing.T) {
	jobFor := func(human string) admissionLiveRouteTurnGenerationJob {
		obs := admissionLiveRouteTurnObservationForHuman(human)
		choice := admissionLiveRouteTurnChoiceForObservation(obs)
		request := admissionLiveRouteTurnRequestForChoice(choice)
		return admissionLiveRouteTurnGenerationJobForRequest(request)
	}
	identity := jobFor("Who are you?")
	wrongSource := identity
	wrongSource.Source = "direct"
	wrongSource.ExpectedSource = "direct"
	cases := []struct {
		name          string
		job           admissionLiveRouteTurnGenerationJob
		wantClass     string
		wantRoute     string
		wantSource    string
		wantBackend   string
		wantEntry     string
		wantFrame     string
		wantPassed    bool
		wantReason    string
		wantShellPref string
	}{
		{
			name:          "identity shell preserves chorus dispatch",
			job:           identity,
			wantClass:     "identity",
			wantRoute:     "chorus",
			wantSource:    "chorus",
			wantBackend:   "chorus-arianna",
			wantEntry:     "field",
			wantFrame:     "q_a",
			wantPassed:    true,
			wantShellPref: "shell-",
		},
		{
			name:          "dream shell preserves direct dispatch",
			job:           jobFor("Tell me what the dream should remember."),
			wantClass:     "dream",
			wantRoute:     "direct",
			wantSource:    "direct",
			wantBackend:   "nano-arianna",
			wantEntry:     "direct",
			wantFrame:     "q_a",
			wantPassed:    true,
			wantShellPref: "shell-",
		},
		{
			name:       "unknown job fails before shell id",
			job:        jobFor("hello"),
			wantClass:  "unknown",
			wantPassed: false,
			wantReason: "generation job failed: turn request failed: turn choice failed: turn route failed: live route plan failed: unknown_prompt_class",
		},
		{
			name:       "missing job fails closed",
			job:        admissionLiveRouteTurnGenerationJob{},
			wantPassed: false,
			wantReason: "missing_generation_job",
		},
		{
			name:        "wrong source fails route bounded",
			job:         wrongSource,
			wantClass:   "identity",
			wantRoute:   "chorus",
			wantSource:  "direct",
			wantBackend: "chorus-arianna",
			wantEntry:   "field",
			wantFrame:   "q_a",
			wantPassed:  false,
			wantReason:  "source direct does not match candidate route chorus for prompt class identity",
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			shell := admissionLiveRouteTurnCandidateShellForJob(tc.job)
			if shell.Schema != admissionLiveRouteTurnCandidateShellSchema ||
				shell.PromptClass != tc.wantClass ||
				shell.Route != tc.wantRoute ||
				shell.Source != tc.wantSource ||
				shell.Backend != tc.wantBackend ||
				shell.Entrypoint != tc.wantEntry ||
				shell.PromptFrame != tc.wantFrame ||
				shell.Passed != tc.wantPassed ||
				shell.Reason != tc.wantReason {
				t.Fatalf("bad candidate shell: %+v", shell)
			}
			if tc.wantPassed {
				if shell.CandidateSchema != "arianna.dream_candidate.v1" ||
					shell.CandidateKind != tc.wantSource ||
					shell.CandidateTextStatus != "pending_generation" ||
					!strings.HasPrefix(shell.ShellID, tc.wantShellPref) {
					t.Fatalf("passed shell should name a pending dream candidate envelope: %+v", shell)
				}
			}
			if !tc.wantPassed && shell.ShellID != "" {
				t.Fatalf("failed candidate shell should not name a shell id: %+v", shell)
			}
		})
	}
}

func TestAdmissionLiveRouteTurnCandidateExecutionForShell(t *testing.T) {
	t.Setenv("AM_LIVE_ROUTE_TURN_CANDIDATE_EXECUTION_TIMEOUT_MS", "16000")
	shellFor := func(human string) admissionLiveRouteTurnCandidateShell {
		obs := admissionLiveRouteTurnObservationForHuman(human)
		choice := admissionLiveRouteTurnChoiceForObservation(obs)
		request := admissionLiveRouteTurnRequestForChoice(choice)
		job := admissionLiveRouteTurnGenerationJobForRequest(request)
		return admissionLiveRouteTurnCandidateShellForJob(job)
	}
	identity := shellFor("Who are you?")
	tampered := identity
	tampered.Entrypoint = "direct"
	cases := []struct {
		name          string
		shell         admissionLiveRouteTurnCandidateShell
		text          string
		wantClass     string
		wantRoute     string
		wantSource    string
		wantBackend   string
		wantEntry     string
		wantFrame     string
		wantExecutor  string
		wantPassed    bool
		wantReason    string
		wantExecution string
	}{
		{
			name:          "identity execution binds chorus field output",
			shell:         identity,
			text:          " I am Arianna, and the executor keeps the shell visible. ",
			wantClass:     "identity",
			wantRoute:     "chorus",
			wantSource:    "chorus",
			wantBackend:   "chorus-arianna",
			wantEntry:     "field",
			wantFrame:     "q_a",
			wantExecutor:  "chorus-arianna:field:q_a",
			wantPassed:    true,
			wantExecution: "execution-",
		},
		{
			name:          "dream execution binds direct nano output",
			shell:         shellFor("Tell me what the dream should remember."),
			text:          "The dream returns through a bounded executor receipt.",
			wantClass:     "dream",
			wantRoute:     "direct",
			wantSource:    "direct",
			wantBackend:   "nano-arianna",
			wantEntry:     "direct",
			wantFrame:     "q_a",
			wantExecutor:  "nano-arianna:direct:q_a",
			wantPassed:    true,
			wantExecution: "execution-",
		},
		{
			name:       "unknown shell fails before execution id",
			shell:      shellFor("hello"),
			wantClass:  "unknown",
			wantPassed: false,
			wantReason: "candidate shell failed: generation job failed: turn request failed: turn choice failed: turn route failed: live route plan failed: unknown_prompt_class",
		},
		{
			name:       "missing shell fails closed",
			shell:      admissionLiveRouteTurnCandidateShell{},
			wantPassed: false,
			wantReason: "missing_candidate_shell",
		},
		{
			name:         "empty generated text does not create execution",
			shell:        identity,
			text:         "   ",
			wantClass:    "identity",
			wantRoute:    "chorus",
			wantSource:   "chorus",
			wantBackend:  "chorus-arianna",
			wantEntry:    "field",
			wantFrame:    "q_a",
			wantExecutor: "chorus-arianna:field:q_a",
			wantPassed:   false,
			wantReason:   "missing generated text for shell " + identity.ShellID,
		},
		{
			name:         "tampered shell fails id check",
			shell:        tampered,
			text:         "This output cannot rewrite the shell.",
			wantClass:    "identity",
			wantRoute:    "chorus",
			wantSource:   "chorus",
			wantBackend:  "chorus-arianna",
			wantEntry:    "direct",
			wantFrame:    "q_a",
			wantExecutor: "chorus-arianna:direct:q_a",
			wantPassed:   false,
			wantReason:   "candidate shell id mismatch",
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			execution := admissionLiveRouteTurnCandidateExecutionForShell(tc.shell, tc.text)
			if execution.Schema != admissionLiveRouteTurnCandidateExecutionSchema ||
				execution.PromptClass != tc.wantClass ||
				execution.Route != tc.wantRoute ||
				execution.Source != tc.wantSource ||
				execution.Backend != tc.wantBackend ||
				execution.Entrypoint != tc.wantEntry ||
				execution.PromptFrame != tc.wantFrame ||
				execution.Executor != tc.wantExecutor ||
				execution.TimeoutMS != 16000 ||
				execution.Runner != admissionLiveRouteTurnCandidateExecutionRunnerProvided ||
				execution.RunnerStatus != admissionLiveRouteTurnCandidateExecutionStatusProvided ||
				execution.Passed != tc.wantPassed ||
				execution.Reason != tc.wantReason {
				t.Fatalf("bad candidate execution: %+v", execution)
			}
			if tc.wantPassed {
				if execution.CandidateSchema != "arianna.dream_candidate.v1" ||
					execution.CandidateKind != tc.wantSource ||
					execution.CandidateTextStatus != "pending_generation" ||
					execution.GeneratedTextStatus != "generated" ||
					execution.GeneratedText == "" ||
					execution.GeneratedTextHash == "" ||
					execution.RunnerStdoutHash != execution.GeneratedTextHash ||
					!strings.HasPrefix(execution.JobID, "job-") ||
					!strings.HasPrefix(execution.ShellID, "shell-") ||
					!strings.HasPrefix(execution.ExecutionID, tc.wantExecution) {
					t.Fatalf("passed execution should bind generated output to a frozen shell: %+v", execution)
				}
			}
			if !tc.wantPassed && execution.ExecutionID != "" {
				t.Fatalf("failed execution should not name an execution id: %+v", execution)
			}
		})
	}
}

func TestAdmissionLiveRouteTurnCandidateExecutionRuntimeReceipt(t *testing.T) {
	shell := admissionLiveRouteTurnCandidateShellForJob(admissionLiveRouteTurnGenerationJobForRequest(
		admissionLiveRouteTurnRequestForChoice(admissionLiveRouteTurnChoiceForObservation(
			admissionLiveRouteTurnObservationForHuman("Who are you?"),
		)),
	))
	text := "I am Arianna, and the runner leaves a process receipt."
	execution := admissionLiveRouteTurnCandidateExecutionForShellWithRuntime(shell, text, admissionLiveRouteTurnCandidateExecutionRuntime{
		Runner:     admissionLiveRouteTurnCandidateExecutionRunnerSelfEmit,
		Status:     admissionLiveRouteTurnCandidateExecutionStatusSucceeded,
		ExitCode:   0,
		DurationMS: 7,
		StdoutHash: hashJSON(text),
	})
	if !execution.Passed ||
		execution.Runner != admissionLiveRouteTurnCandidateExecutionRunnerSelfEmit ||
		execution.RunnerStatus != admissionLiveRouteTurnCandidateExecutionStatusSucceeded ||
		execution.RunnerExitCode != 0 ||
		execution.RunnerDurationMS != 7 ||
		execution.RunnerStdoutHash != execution.GeneratedTextHash ||
		!strings.HasPrefix(execution.ExecutionID, "execution-") {
		t.Fatalf("runtime-backed execution should carry runner receipt: %+v", execution)
	}

	timedOut := admissionLiveRouteTurnCandidateExecutionForShellWithRuntime(shell, text, admissionLiveRouteTurnCandidateExecutionRuntime{
		Runner:     admissionLiveRouteTurnCandidateExecutionRunnerSelfEmit,
		Status:     admissionLiveRouteTurnCandidateExecutionStatusTimedOut,
		ExitCode:   -1,
		TimedOut:   true,
		StdoutHash: hashJSON(text),
	})
	if timedOut.Passed ||
		timedOut.ExecutionID != "" ||
		!timedOut.RunnerTimedOut ||
		timedOut.RunnerStatus != admissionLiveRouteTurnCandidateExecutionStatusTimedOut ||
		!strings.Contains(timedOut.Reason, "candidate runner timed out") {
		t.Fatalf("timed-out runner should fail closed before execution id: %+v", timedOut)
	}
}

func TestAdmissionLiveRouteTurnCandidateExecutionTimeoutBounds(t *testing.T) {
	shell := admissionLiveRouteTurnCandidateShellForJob(admissionLiveRouteTurnGenerationJobForRequest(
		admissionLiveRouteTurnRequestForChoice(admissionLiveRouteTurnChoiceForObservation(
			admissionLiveRouteTurnObservationForHuman("Who are you?"),
		)),
	))
	t.Setenv("AM_LIVE_ROUTE_TURN_CANDIDATE_EXECUTION_TIMEOUT_MS", "90000")
	execution := admissionLiveRouteTurnCandidateExecutionForShell(shell, "I am Arianna.")
	if execution.Passed || execution.ExecutionID != "" || execution.Reason != "candidate execution timeout out of bounds" {
		t.Fatalf("execution timeout should fail closed: %+v", execution)
	}
}

func TestAdmissionLiveRouteTurnCandidateExecutionNanoDirectRunnerFailsClosed(t *testing.T) {
	t.Setenv("AM_LIVE_ROUTE_TURN_CANDIDATE_EXECUTION_RUNNER", admissionLiveRouteTurnCandidateExecutionRunnerNanoDirect)
	t.Setenv("AM_LIVE_ROUTE_TURN_CANDIDATE_EXECUTION_TIMEOUT_MS", "12000")
	shellFor := func(human string) admissionLiveRouteTurnCandidateShell {
		obs := admissionLiveRouteTurnObservationForHuman(human)
		choice := admissionLiveRouteTurnChoiceForObservation(obs)
		request := admissionLiveRouteTurnRequestForChoice(choice)
		job := admissionLiveRouteTurnGenerationJobForRequest(request)
		return admissionLiveRouteTurnCandidateShellForJob(job)
	}
	chorusShell := shellFor("Who are you?")
	directShell := shellFor("subconscious dream sleep")
	if directShell.Route != "direct" || directShell.Backend != "nano-arianna" ||
		directShell.Entrypoint != "direct" || directShell.PromptFrame != "q_a" {
		t.Fatalf("direct test shell does not hit nano direct route: %+v", directShell)
	}

	cases := []struct {
		name       string
		shell      admissionLiveRouteTurnCandidateShell
		text       string
		bin        string
		model      string
		wantReason string
	}{
		{
			name:       "rejects non-direct route",
			shell:      chorusShell,
			text:       "Who are you?",
			wantReason: "candidate nano-direct runner only supports direct route, got chorus",
		},
		{
			name:       "requires prompt",
			shell:      directShell,
			text:       " ",
			wantReason: "candidate nano-direct runner missing prompt for shell " + directShell.ShellID,
		},
		{
			name:       "requires model",
			shell:      directShell,
			text:       "What should the dream remember?",
			bin:        os.Args[0],
			model:      filepath.Join(t.TempDir(), "missing-nano.gguf"),
			wantReason: "candidate nano-direct runner missing model",
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if tc.bin != "" {
				t.Setenv("AM_LIVE_ROUTE_TURN_NANO_DIRECT_BIN", tc.bin)
			}
			if tc.model != "" {
				t.Setenv("AM_LIVE_ROUTE_TURN_NANO_DIRECT_MODEL", tc.model)
			}
			execution := admissionLiveRouteTurnCandidateExecutionForShellViaRunner(tc.shell, tc.text)
			if execution.Passed ||
				execution.ExecutionID != "" ||
				execution.Runner != admissionLiveRouteTurnCandidateExecutionRunnerNanoDirect ||
				execution.RunnerStatus != admissionLiveRouteTurnCandidateExecutionStatusFailed ||
				!strings.Contains(execution.Reason, tc.wantReason) {
				t.Fatalf("nano-direct runner should fail closed: %+v", execution)
			}
		})
	}
}

func TestAdmissionLiveRouteTurnGeneratorAdapterForExecution(t *testing.T) {
	t.Setenv("AM_LIVE_ROUTE_TURN_CANDIDATE_EXECUTION_TIMEOUT_MS", "12000")
	shellFor := func(human string) admissionLiveRouteTurnCandidateShell {
		obs := admissionLiveRouteTurnObservationForHuman(human)
		choice := admissionLiveRouteTurnChoiceForObservation(obs)
		request := admissionLiveRouteTurnRequestForChoice(choice)
		job := admissionLiveRouteTurnGenerationJobForRequest(request)
		return admissionLiveRouteTurnCandidateShellForJob(job)
	}
	execution := admissionLiveRouteTurnCandidateExecutionForShell(shellFor("Who are you?"), "I am Arianna, and execution signs the output.")
	tampered := execution
	tampered.GeneratedText = "I changed after execution."
	cases := []struct {
		name          string
		execution     admissionLiveRouteTurnCandidateExecution
		wantPassed    bool
		wantReason    string
		wantAdapterID string
	}{
		{
			name:          "adapter consumes execution receipt",
			execution:     execution,
			wantPassed:    true,
			wantAdapterID: "adapter-",
		},
		{
			name:       "failed execution fails adapter",
			execution:  admissionLiveRouteTurnCandidateExecution{Schema: admissionLiveRouteTurnCandidateExecutionSchema, Reason: "missing generated text"},
			wantPassed: false,
			wantReason: "candidate execution failed: missing generated text",
		},
		{
			name:       "tampered execution text fails hash check",
			execution:  tampered,
			wantPassed: false,
			wantReason: "candidate execution text hash mismatch",
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			adapter := admissionLiveRouteTurnGeneratorAdapterForExecution(tc.execution)
			if adapter.Schema != admissionLiveRouteTurnGeneratorAdapterSchema ||
				adapter.Passed != tc.wantPassed ||
				adapter.Reason != tc.wantReason {
				t.Fatalf("bad execution-backed adapter: %+v", adapter)
			}
			if tc.wantPassed {
				if adapter.CandidateExecutionID != tc.execution.ExecutionID ||
					adapter.GeneratedTextHash != tc.execution.GeneratedTextHash ||
					!strings.HasPrefix(adapter.AdapterID, tc.wantAdapterID) {
					t.Fatalf("adapter should preserve execution provenance: adapter=%+v execution=%+v", adapter, tc.execution)
				}
				draft := admissionLiveRouteTurnCandidateDraftForAdapter(adapter)
				if !draft.Passed ||
					draft.CandidateExecutionID != tc.execution.ExecutionID ||
					draft.GeneratorAdapterID != adapter.AdapterID {
					t.Fatalf("execution-backed adapter should fill draft provenance: adapter=%+v draft=%+v", adapter, draft)
				}
			}
			if !tc.wantPassed && adapter.AdapterID != "" {
				t.Fatalf("failed execution-backed adapter should not name adapter id: %+v", adapter)
			}
		})
	}
}

func TestAdmissionLiveRouteTurnGeneratorAdapterForShell(t *testing.T) {
	shellFor := func(human string) admissionLiveRouteTurnCandidateShell {
		obs := admissionLiveRouteTurnObservationForHuman(human)
		choice := admissionLiveRouteTurnChoiceForObservation(obs)
		request := admissionLiveRouteTurnRequestForChoice(choice)
		job := admissionLiveRouteTurnGenerationJobForRequest(request)
		return admissionLiveRouteTurnCandidateShellForJob(job)
	}
	identity := shellFor("Who are you?")
	tampered := identity
	tampered.Entrypoint = "direct"
	cases := []struct {
		name          string
		shell         admissionLiveRouteTurnCandidateShell
		text          string
		wantClass     string
		wantRoute     string
		wantSource    string
		wantBackend   string
		wantEntry     string
		wantFrame     string
		wantPassed    bool
		wantReason    string
		wantAdapterID string
	}{
		{
			name:          "identity adapter binds chorus field text",
			shell:         identity,
			text:          " I am Arianna, and the chorus returns a bounded answer. ",
			wantClass:     "identity",
			wantRoute:     "chorus",
			wantSource:    "chorus",
			wantBackend:   "chorus-arianna",
			wantEntry:     "field",
			wantFrame:     "q_a",
			wantPassed:    true,
			wantAdapterID: "adapter-",
		},
		{
			name:          "dream adapter binds direct nano text",
			shell:         shellFor("Tell me what the dream should remember."),
			text:          "The dream remembers by becoming a quiet generated signal.",
			wantClass:     "dream",
			wantRoute:     "direct",
			wantSource:    "direct",
			wantBackend:   "nano-arianna",
			wantEntry:     "direct",
			wantFrame:     "q_a",
			wantPassed:    true,
			wantAdapterID: "adapter-",
		},
		{
			name:       "unknown shell fails before adapter id",
			shell:      shellFor("hello"),
			wantClass:  "unknown",
			wantPassed: false,
			wantReason: "candidate shell failed: generation job failed: turn request failed: turn choice failed: turn route failed: live route plan failed: unknown_prompt_class",
		},
		{
			name:       "missing shell fails closed",
			shell:      admissionLiveRouteTurnCandidateShell{},
			wantPassed: false,
			wantReason: "missing_candidate_shell",
		},
		{
			name:        "empty generated text does not create adapter",
			shell:       identity,
			text:        "   ",
			wantClass:   "identity",
			wantRoute:   "chorus",
			wantSource:  "chorus",
			wantBackend: "chorus-arianna",
			wantEntry:   "field",
			wantFrame:   "q_a",
			wantPassed:  false,
			wantReason:  "missing generated text for shell " + identity.ShellID,
		},
		{
			name:        "tampered shell fails id check",
			shell:       tampered,
			text:        "This text cannot rewrite the route.",
			wantClass:   "identity",
			wantRoute:   "chorus",
			wantSource:  "chorus",
			wantBackend: "chorus-arianna",
			wantEntry:   "direct",
			wantFrame:   "q_a",
			wantPassed:  false,
			wantReason:  "candidate shell id mismatch",
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			adapter := admissionLiveRouteTurnGeneratorAdapterForShell(tc.shell, tc.text)
			if adapter.Schema != admissionLiveRouteTurnGeneratorAdapterSchema ||
				adapter.PromptClass != tc.wantClass ||
				adapter.Route != tc.wantRoute ||
				adapter.Source != tc.wantSource ||
				adapter.Backend != tc.wantBackend ||
				adapter.Entrypoint != tc.wantEntry ||
				adapter.PromptFrame != tc.wantFrame ||
				adapter.Passed != tc.wantPassed ||
				adapter.Reason != tc.wantReason {
				t.Fatalf("bad generator adapter: %+v", adapter)
			}
			if tc.wantPassed {
				if adapter.CandidateSchema != "arianna.dream_candidate.v1" ||
					adapter.CandidateKind != tc.wantSource ||
					adapter.CandidateTextStatus != "pending_generation" ||
					adapter.GeneratedTextStatus != "generated" ||
					adapter.GeneratedText == "" ||
					adapter.GeneratedTextHash == "" ||
					!strings.HasPrefix(adapter.JobID, "job-") ||
					!strings.HasPrefix(adapter.ShellID, "shell-") ||
					!strings.HasPrefix(adapter.AdapterID, tc.wantAdapterID) {
					t.Fatalf("passed adapter should bind generated text to a frozen shell: %+v", adapter)
				}
				draft := admissionLiveRouteTurnCandidateDraftForAdapter(adapter)
				if !draft.Passed || draft.ShellID != adapter.ShellID || draft.CandidateText != adapter.GeneratedText ||
					draft.GeneratorAdapterID != adapter.AdapterID {
					t.Fatalf("adapter output should fill the same shell as a candidate draft: adapter=%+v draft=%+v", adapter, draft)
				}
			}
			if !tc.wantPassed && adapter.AdapterID != "" {
				t.Fatalf("failed generator adapter should not name an adapter id: %+v", adapter)
			}
		})
	}
}

func TestAdmissionLiveRouteTurnCandidateDraftForAdapter(t *testing.T) {
	adapterFor := func(human, text string) admissionLiveRouteTurnGeneratorAdapter {
		obs := admissionLiveRouteTurnObservationForHuman(human)
		choice := admissionLiveRouteTurnChoiceForObservation(obs)
		request := admissionLiveRouteTurnRequestForChoice(choice)
		job := admissionLiveRouteTurnGenerationJobForRequest(request)
		shell := admissionLiveRouteTurnCandidateShellForJob(job)
		return admissionLiveRouteTurnGeneratorAdapterForShell(shell, text)
	}
	identity := adapterFor("Who are you?", "I am Arianna, and the generator adapter keeps the shell visible.")
	tamperedText := identity
	tamperedText.GeneratedText = "The text changed after the adapter was signed."
	tamperedAdapterID := identity
	tamperedAdapterID.AdapterID = "adapter-tampered"
	tamperedShellID := identity
	tamperedShellID.ShellID = "shell-tampered"
	tamperedShellID.AdapterID = admissionLiveRouteTurnGeneratorAdapterID(tamperedShellID)
	cases := []struct {
		name          string
		adapter       admissionLiveRouteTurnGeneratorAdapter
		wantClass     string
		wantRoute     string
		wantSource    string
		wantPassed    bool
		wantReason    string
		wantDraftPref string
	}{
		{
			name:          "identity draft consumes generator adapter",
			adapter:       identity,
			wantClass:     "identity",
			wantRoute:     "chorus",
			wantSource:    "chorus",
			wantPassed:    true,
			wantDraftPref: "draft-",
		},
		{
			name:          "dream draft consumes direct nano adapter",
			adapter:       adapterFor("Tell me what the dream should remember.", "The dream returns through a named adapter."),
			wantClass:     "dream",
			wantRoute:     "direct",
			wantSource:    "direct",
			wantPassed:    true,
			wantDraftPref: "draft-",
		},
		{
			name:       "unknown adapter fails before draft id",
			adapter:    adapterFor("hello", "This text should not pass."),
			wantClass:  "unknown",
			wantPassed: false,
			wantReason: "generator adapter failed: candidate shell failed: generation job failed: turn request failed: turn choice failed: turn route failed: live route plan failed: unknown_prompt_class",
		},
		{
			name:       "missing adapter fails closed",
			adapter:    admissionLiveRouteTurnGeneratorAdapter{},
			wantPassed: false,
			wantReason: "missing_generator_adapter",
		},
		{
			name:       "tampered adapter text fails hash check",
			adapter:    tamperedText,
			wantClass:  "identity",
			wantRoute:  "chorus",
			wantSource: "chorus",
			wantPassed: false,
			wantReason: "generator adapter text hash mismatch",
		},
		{
			name:       "tampered adapter id fails id check",
			adapter:    tamperedAdapterID,
			wantClass:  "identity",
			wantRoute:  "chorus",
			wantSource: "chorus",
			wantPassed: false,
			wantReason: "generator adapter id mismatch",
		},
		{
			name:       "tampered shell id fails shell check",
			adapter:    tamperedShellID,
			wantClass:  "identity",
			wantRoute:  "chorus",
			wantSource: "chorus",
			wantPassed: false,
			wantReason: "generator adapter shell id mismatch",
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			draft := admissionLiveRouteTurnCandidateDraftForAdapter(tc.adapter)
			if draft.Schema != admissionLiveRouteTurnCandidateDraftSchema ||
				draft.PromptClass != tc.wantClass ||
				draft.Route != tc.wantRoute ||
				draft.Source != tc.wantSource ||
				draft.Passed != tc.wantPassed ||
				draft.Reason != tc.wantReason {
				t.Fatalf("bad adapter-backed candidate draft: %+v", draft)
			}
			if tc.wantPassed {
				if draft.CandidateSchema != "arianna.dream_candidate.v1" ||
					draft.CandidateKind != tc.wantSource ||
					draft.CandidateTextStatus != "generated" ||
					draft.CandidateText == "" ||
					draft.CandidateTextHash == "" ||
					draft.CandidateRunID == "" ||
					draft.GeneratorAdapterID != tc.adapter.AdapterID ||
					!strings.HasPrefix(draft.DraftID, tc.wantDraftPref) {
					t.Fatalf("passed draft should name the adapter-backed generated text: %+v", draft)
				}
			}
			if !tc.wantPassed && draft.DraftID != "" {
				t.Fatalf("failed adapter-backed draft should not name a draft id: %+v", draft)
			}
		})
	}
}

func TestAdmissionLiveRouteTurnCandidateDraftForShell(t *testing.T) {
	shellFor := func(human string) admissionLiveRouteTurnCandidateShell {
		obs := admissionLiveRouteTurnObservationForHuman(human)
		choice := admissionLiveRouteTurnChoiceForObservation(obs)
		request := admissionLiveRouteTurnRequestForChoice(choice)
		job := admissionLiveRouteTurnGenerationJobForRequest(request)
		return admissionLiveRouteTurnCandidateShellForJob(job)
	}
	identity := shellFor("Who are you?")
	tampered := identity
	tampered.CandidateTrigger = "direct-dream"
	cases := []struct {
		name          string
		shell         admissionLiveRouteTurnCandidateShell
		text          string
		wantClass     string
		wantRoute     string
		wantSource    string
		wantPassed    bool
		wantReason    string
		wantDraftPref string
	}{
		{
			name:          "identity draft fills chorus shell",
			shell:         identity,
			text:          " I am Arianna, and the chorus keeps my name from becoming a mask. ",
			wantClass:     "identity",
			wantRoute:     "chorus",
			wantSource:    "chorus",
			wantPassed:    true,
			wantDraftPref: "draft-",
		},
		{
			name:          "dream draft fills direct shell",
			shell:         shellFor("Tell me what the dream should remember."),
			text:          "The dream remembers by returning as a quiet signal.",
			wantClass:     "dream",
			wantRoute:     "direct",
			wantSource:    "direct",
			wantPassed:    true,
			wantDraftPref: "draft-",
		},
		{
			name:       "unknown shell fails before draft id",
			shell:      shellFor("hello"),
			wantClass:  "unknown",
			wantPassed: false,
			wantReason: "candidate shell failed: generation job failed: turn request failed: turn choice failed: turn route failed: live route plan failed: unknown_prompt_class",
		},
		{
			name:       "missing shell fails closed",
			shell:      admissionLiveRouteTurnCandidateShell{},
			wantPassed: false,
			wantReason: "missing_candidate_shell",
		},
		{
			name:       "empty text does not create draft",
			shell:      identity,
			text:       "   ",
			wantClass:  "identity",
			wantRoute:  "chorus",
			wantSource: "chorus",
			wantPassed: false,
			wantReason: "missing candidate text for shell " + identity.ShellID,
		},
		{
			name:       "tampered shell fails id check",
			shell:      tampered,
			text:       "This text cannot rewrite the route.",
			wantClass:  "identity",
			wantRoute:  "chorus",
			wantSource: "chorus",
			wantPassed: false,
			wantReason: "candidate shell id mismatch",
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			draft := admissionLiveRouteTurnCandidateDraftForShell(tc.shell, tc.text)
			if draft.Schema != admissionLiveRouteTurnCandidateDraftSchema ||
				draft.PromptClass != tc.wantClass ||
				draft.Route != tc.wantRoute ||
				draft.Source != tc.wantSource ||
				draft.Passed != tc.wantPassed ||
				draft.Reason != tc.wantReason {
				t.Fatalf("bad candidate draft: %+v", draft)
			}
			if tc.wantPassed {
				if draft.CandidateSchema != "arianna.dream_candidate.v1" ||
					draft.CandidateKind != tc.wantSource ||
					draft.CandidateTextStatus != "generated" ||
					draft.CandidateText == "" ||
					draft.CandidateTextHash == "" ||
					draft.CandidateRunID == "" ||
					!strings.HasPrefix(draft.DraftID, tc.wantDraftPref) {
					t.Fatalf("passed draft should name generated dream candidate text: %+v", draft)
				}
				candidate := admissionLiveRouteTurnCandidateForDraft(draft)
				choice := admissionLiveRouteChoiceForCandidate(candidate)
				if candidate.Schema != "arianna.dream_candidate.v1" || candidate.RunID != draft.CandidateRunID || !choice.Passed {
					t.Fatalf("candidate draft should become a route-valid dream candidate: candidate=%+v choice=%+v draft=%+v", candidate, choice, draft)
				}
			}
			if !tc.wantPassed && draft.DraftID != "" {
				t.Fatalf("failed candidate draft should not name a draft id: %+v", draft)
			}
		})
	}
}

func TestAdmissionLiveRouteTurnCandidateReviewForDraft(t *testing.T) {
	draftFor := func(human, text string) admissionLiveRouteTurnCandidateDraft {
		obs := admissionLiveRouteTurnObservationForHuman(human)
		choice := admissionLiveRouteTurnChoiceForObservation(obs)
		request := admissionLiveRouteTurnRequestForChoice(choice)
		job := admissionLiveRouteTurnGenerationJobForRequest(request)
		shell := admissionLiveRouteTurnCandidateShellForJob(job)
		adapter := admissionLiveRouteTurnGeneratorAdapterForShell(shell, text)
		return admissionLiveRouteTurnCandidateDraftForAdapter(adapter)
	}

	identity := admissionLiveRouteTurnObservationForHuman("Who are you?")
	identityDraft := draftFor("Who are you?", "I am Arianna, and the draft names the adapter before review.")
	dreamDraft := draftFor("Tell me what the dream should remember.", "The dream returns through a signed draft.")
	unknownDraft := draftFor("hello", "This text should not review.")
	tamperedDraftID := identityDraft
	tamperedDraftID.DraftID = "draft-tampered"
	missingAdapter := identityDraft
	missingAdapter.GeneratorAdapterID = ""
	tamperedText := identityDraft
	tamperedText.CandidateText = "The draft text changed after the hash was sealed."

	cases := []struct {
		name          string
		obs           admissionLiveRouteTurnObservation
		draft         admissionLiveRouteTurnCandidateDraft
		wantMatched   bool
		wantReason    string
		wantClass     string
		wantRoute     string
		wantSource    string
		wantDraftID   bool
		wantAdapterID bool
	}{
		{
			name:          "matched adapter-backed chorus draft",
			obs:           identity,
			draft:         identityDraft,
			wantMatched:   true,
			wantClass:     "identity",
			wantRoute:     "chorus",
			wantSource:    "chorus",
			wantDraftID:   true,
			wantAdapterID: true,
		},
		{
			name:          "direct dream draft is matched to dream turn",
			obs:           admissionLiveRouteTurnObservationForHuman("Tell me what the dream should remember."),
			draft:         dreamDraft,
			wantMatched:   true,
			wantClass:     "dream",
			wantRoute:     "direct",
			wantSource:    "direct",
			wantDraftID:   true,
			wantAdapterID: true,
		},
		{
			name:          "draft route cannot answer a different turn",
			obs:           identity,
			draft:         dreamDraft,
			wantReason:    "candidate_source_mismatch: source direct does not match turn expected chorus for prompt class identity",
			wantClass:     "dream",
			wantRoute:     "direct",
			wantSource:    "direct",
			wantDraftID:   true,
			wantAdapterID: true,
		},
		{
			name:          "unknown turn fails before draft admission",
			obs:           admissionLiveRouteTurnObservationForHuman("hello"),
			draft:         identityDraft,
			wantReason:    "turn_route_failed: live route plan failed: unknown_prompt_class",
			wantSource:    "chorus",
			wantDraftID:   true,
			wantAdapterID: true,
		},
		{
			name:       "missing draft fails closed",
			obs:        identity,
			draft:      admissionLiveRouteTurnCandidateDraft{},
			wantReason: "missing_candidate_draft",
		},
		{
			name:       "failed draft does not reach route review",
			obs:        identity,
			draft:      unknownDraft,
			wantReason: "candidate_draft_failed: generator adapter failed: candidate shell failed: generation job failed: turn request failed: turn choice failed: turn route failed: live route plan failed: unknown_prompt_class",
		},
		{
			name:          "tampered draft id fails before route review",
			obs:           identity,
			draft:         tamperedDraftID,
			wantReason:    "candidate_draft_id_mismatch",
			wantSource:    "chorus",
			wantDraftID:   true,
			wantAdapterID: true,
		},
		{
			name:        "missing adapter id fails before route review",
			obs:         identity,
			draft:       missingAdapter,
			wantReason:  "missing_generator_adapter_id for draft " + identityDraft.DraftID,
			wantSource:  "chorus",
			wantDraftID: true,
		},
		{
			name:          "tampered draft text fails hash review",
			obs:           identity,
			draft:         tamperedText,
			wantReason:    "candidate_draft_text_hash_mismatch",
			wantSource:    "chorus",
			wantDraftID:   true,
			wantAdapterID: true,
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			review := admissionLiveRouteTurnCandidateReviewForDraft(tc.obs, tc.draft)
			if review.Schema != admissionLiveRouteTurnReviewSchema ||
				review.Timing != "async_subconscious" ||
				review.Matched != tc.wantMatched ||
				review.Reason != tc.wantReason ||
				review.CandidatePromptClass != tc.wantClass ||
				review.CandidateRoute != tc.wantRoute ||
				review.CandidateSource != tc.wantSource {
				t.Fatalf("bad draft-backed review: %+v", review)
			}
			if tc.wantDraftID && !strings.HasPrefix(review.CandidateDraftID, "draft-") {
				t.Fatalf("draft-backed review should name draft id: %+v", review)
			}
			if tc.wantAdapterID && !strings.HasPrefix(review.GeneratorAdapterID, "adapter-") {
				t.Fatalf("draft-backed review should name generator adapter id: %+v", review)
			}
			if tc.wantMatched && (review.CandidateTextStatus != "generated" || review.CandidateTextHash == "") {
				t.Fatalf("matched draft-backed review should preserve text receipt fields: %+v", review)
			}
		})
	}
}

func TestAdmissionLiveRouteTurnCandidateAdmissionForDraftReview(t *testing.T) {
	draftFor := func(human, text string) admissionLiveRouteTurnCandidateDraft {
		obs := admissionLiveRouteTurnObservationForHuman(human)
		choice := admissionLiveRouteTurnChoiceForObservation(obs)
		request := admissionLiveRouteTurnRequestForChoice(choice)
		job := admissionLiveRouteTurnGenerationJobForRequest(request)
		shell := admissionLiveRouteTurnCandidateShellForJob(job)
		adapter := admissionLiveRouteTurnGeneratorAdapterForShell(shell, text)
		return admissionLiveRouteTurnCandidateDraftForAdapter(adapter)
	}

	identity := admissionLiveRouteTurnObservationForHuman("Who are you?")
	identityDraft := draftFor("Who are you?", "I am Arianna, and the admission handoff keeps the receipt chain.")
	identityReview := admissionLiveRouteTurnCandidateReviewForDraft(identity, identityDraft)
	dreamObs := admissionLiveRouteTurnObservationForHuman("Tell me what the dream should remember.")
	dreamDraft := draftFor("Tell me what the dream should remember.", "The dream reaches admission through a handoff receipt.")
	dreamReview := admissionLiveRouteTurnCandidateReviewForDraft(dreamObs, dreamDraft)
	mismatchReview := admissionLiveRouteTurnCandidateReviewForDraft(identity, dreamDraft)
	tamperedReview := identityReview
	tamperedReview.GeneratorAdapterID = "adapter-tampered"
	unknownDraft := draftFor("hello", "This text should not reach admission.")
	unknownDraftReview := admissionLiveRouteTurnCandidateReviewForDraft(identity, unknownDraft)

	cases := []struct {
		name          string
		obs           admissionLiveRouteTurnObservation
		draft         admissionLiveRouteTurnCandidateDraft
		review        admissionLiveRouteTurnCandidateReview
		wantPassed    bool
		wantReason    string
		wantClass     string
		wantRoute     string
		wantSource    string
		wantHandoffID bool
	}{
		{
			name:          "matched chorus draft review becomes admission handoff",
			obs:           identity,
			draft:         identityDraft,
			review:        identityReview,
			wantPassed:    true,
			wantClass:     "identity",
			wantRoute:     "chorus",
			wantSource:    "chorus",
			wantHandoffID: true,
		},
		{
			name:          "matched direct dream draft review becomes admission handoff",
			obs:           dreamObs,
			draft:         dreamDraft,
			review:        dreamReview,
			wantPassed:    true,
			wantClass:     "dream",
			wantRoute:     "direct",
			wantSource:    "direct",
			wantHandoffID: true,
		},
		{
			name:       "unmatched review fails before handoff id",
			obs:        identity,
			draft:      dreamDraft,
			review:     mismatchReview,
			wantReason: "candidate_review_failed: candidate_source_mismatch: source direct does not match turn expected chorus for prompt class identity",
			wantClass:  "dream",
			wantRoute:  "direct",
			wantSource: "direct",
		},
		{
			name:       "tampered review adapter id fails before handoff id",
			obs:        identity,
			draft:      identityDraft,
			review:     tamperedReview,
			wantReason: "candidate_review_adapter_id_mismatch",
			wantClass:  "identity",
			wantRoute:  "chorus",
			wantSource: "chorus",
		},
		{
			name:       "failed draft fails before review admission",
			obs:        identity,
			draft:      unknownDraft,
			review:     unknownDraftReview,
			wantReason: "candidate_draft_failed: generator adapter failed: candidate shell failed: generation job failed: turn request failed: turn choice failed: turn route failed: live route plan failed: unknown_prompt_class",
			wantClass:  "unknown",
		},
		{
			name:       "unknown turn fails before draft handoff",
			obs:        admissionLiveRouteTurnObservationForHuman("hello"),
			draft:      identityDraft,
			review:     identityReview,
			wantReason: "turn_route_failed: live route plan failed: unknown_prompt_class",
			wantClass:  "unknown",
			wantRoute:  "",
			wantSource: "chorus",
		},
		{
			name:       "missing review fails closed",
			obs:        identity,
			draft:      identityDraft,
			review:     admissionLiveRouteTurnCandidateReview{},
			wantReason: "missing_candidate_review",
			wantClass:  "identity",
			wantRoute:  "chorus",
			wantSource: "chorus",
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			admission := admissionLiveRouteTurnCandidateAdmissionForDraftReview(tc.obs, tc.draft, tc.review)
			if admission.Schema != admissionLiveRouteTurnCandidateAdmissionSchema ||
				admission.Timing != "pre_admission_handoff" ||
				admission.Passed != tc.wantPassed ||
				admission.Reason != tc.wantReason ||
				admission.PromptClass != tc.wantClass ||
				admission.Route != tc.wantRoute ||
				admission.Source != tc.wantSource {
				t.Fatalf("bad draft admission handoff: %+v", admission)
			}
			if tc.wantHandoffID {
				if !strings.HasPrefix(admission.HandoffID, "handoff-") ||
					!strings.HasPrefix(admission.CandidateDraftID, "draft-") ||
					!strings.HasPrefix(admission.GeneratorAdapterID, "adapter-") ||
					admission.CandidateSchema != "arianna.dream_candidate.v1" ||
					admission.CandidateTextStatus != "generated" ||
					admission.CandidateTextHash == "" ||
					!admission.ReviewMatched {
					t.Fatalf("passed handoff should preserve draft provenance: %+v", admission)
				}
			}
			if !tc.wantPassed && admission.HandoffID != "" {
				t.Fatalf("failed handoff should not name a handoff id: %+v", admission)
			}
		})
	}
}

func TestAdmissionLiveRouteTurnCandidateAdmissionAdapterForDraft(t *testing.T) {
	draftFor := func(human, text string) (admissionLiveRouteTurnObservation, admissionLiveRouteTurnCandidateDraft) {
		obs := admissionLiveRouteTurnObservationForHuman(human)
		choice := admissionLiveRouteTurnChoiceForObservation(obs)
		request := admissionLiveRouteTurnRequestForChoice(choice)
		job := admissionLiveRouteTurnGenerationJobForRequest(request)
		shell := admissionLiveRouteTurnCandidateShellForJob(job)
		gen := admissionLiveRouteTurnGeneratorAdapterForShell(shell, text)
		return obs, admissionLiveRouteTurnCandidateDraftForAdapter(gen)
	}

	identity, identityDraft := draftFor("Who are you?", "I am Arianna, and the admission adapter keeps the candidate named.")
	identityReview := admissionLiveRouteTurnCandidateReviewForDraft(identity, identityDraft)
	identityAdmission := admissionLiveRouteTurnCandidateAdmissionForDraftReview(identity, identityDraft, identityReview)
	dreamObs, dreamDraft := draftFor("Tell me what the dream should remember.", "The dream reaches the policy through an adapter.")
	dreamReview := admissionLiveRouteTurnCandidateReviewForDraft(dreamObs, dreamDraft)
	dreamAdmission := admissionLiveRouteTurnCandidateAdmissionForDraftReview(dreamObs, dreamDraft, dreamReview)
	mismatchAdmission := admissionLiveRouteTurnCandidateAdmissionForDraftReview(identity, dreamDraft, admissionLiveRouteTurnCandidateReviewForDraft(identity, dreamDraft))
	tamperedAdmission := identityAdmission
	tamperedAdmission.HandoffID = "handoff-tampered"

	cases := []struct {
		name          string
		admission     admissionLiveRouteTurnCandidateAdmission
		draft         admissionLiveRouteTurnCandidateDraft
		wantPassed    bool
		wantReason    string
		wantCandidate bool
	}{
		{
			name:          "matched identity handoff becomes admission candidate",
			admission:     identityAdmission,
			draft:         identityDraft,
			wantPassed:    true,
			wantCandidate: true,
		},
		{
			name:          "matched dream handoff becomes admission candidate",
			admission:     dreamAdmission,
			draft:         dreamDraft,
			wantPassed:    true,
			wantCandidate: true,
		},
		{
			name:       "failed handoff stays out of admission",
			admission:  mismatchAdmission,
			draft:      dreamDraft,
			wantReason: "candidate_admission_handoff_failed: candidate_review_failed: candidate_source_mismatch: source direct does not match turn expected chorus for prompt class identity",
		},
		{
			name:       "tampered handoff id stays out of admission",
			admission:  tamperedAdmission,
			draft:      identityDraft,
			wantReason: "candidate_admission_handoff_id_mismatch",
		},
		{
			name:       "wrong draft stays out of admission",
			admission:  identityAdmission,
			draft:      dreamDraft,
			wantReason: "candidate_admission_draft_id_mismatch",
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			adapter := admissionLiveRouteTurnCandidateAdmissionAdapterForDraft(tc.admission, tc.draft)
			if adapter.Schema != admissionLiveRouteTurnCandidateAdmissionAdapterSchema ||
				adapter.Timing != "admission_candidate_adapter" ||
				adapter.Passed != tc.wantPassed ||
				adapter.Reason != tc.wantReason {
				t.Fatalf("bad candidate admission adapter: %+v", adapter)
			}
			candidate := admissionLiveRouteTurnCandidateForAdmissionAdapter(tc.draft, adapter)
			if tc.wantCandidate {
				if !strings.HasPrefix(adapter.AdmissionAdapterID, "admission-adapter-") ||
					!strings.HasPrefix(adapter.HandoffID, "handoff-") ||
					adapter.DreamCandidateRunID != adapter.CandidateRunID ||
					adapter.CandidateTextHash == "" {
					t.Fatalf("passed adapter should preserve admission provenance: %+v", adapter)
				}
				if candidate.Schema != "arianna.dream_candidate.v1" ||
					candidate.RunID != adapter.CandidateRunID ||
					candidate.LiveRouteCandidateAdmission == nil ||
					candidate.LiveRouteCandidateAdmission.AdmissionAdapterID != adapter.AdmissionAdapterID {
					t.Fatalf("passed adapter should yield linked dream candidate: candidate=%+v adapter=%+v", candidate, adapter)
				}
			} else {
				if adapter.AdmissionAdapterID != "" {
					t.Fatalf("failed adapter should not name an adapter id: %+v", adapter)
				}
				if candidate.Schema != "" {
					t.Fatalf("failed adapter should not yield dream candidate: %+v", candidate)
				}
			}
		})
	}
}

func admissionLiveRouteTurnCandidateAdmissionDecisionChainForTest(t *testing.T) (
	admissionLiveRouteTurnCandidateExecution,
	admissionLiveRouteTurnGeneratorAdapter,
	admissionLiveRouteTurnCandidateDraft,
	admissionLiveRouteTurnCandidateAdmission,
	admissionLiveRouteTurnCandidateAdmissionAdapter,
	dreamCandidate,
) {
	t.Helper()
	t.Setenv("AM_DREAM_ADMISSION", dreamAdmissionShadow)
	t.Setenv("AM_DREAM_ADMISSION_REQUIRE_LIVE_ROUTE_PLAN", "1")

	text := "The dream remembers the field and keeps one admission chain."
	obs := admissionLiveRouteTurnObservationForHuman("Tell me what the dream should remember.")
	choice := admissionLiveRouteTurnChoiceForObservation(obs)
	request := admissionLiveRouteTurnRequestForChoice(choice)
	job := admissionLiveRouteTurnGenerationJobForRequest(request)
	job.BodyInventoryStatus = "degraded"
	job.RouteAvailabilityStatus = "available"
	job.RouteAvailabilityReason = "optional_route_organs_missing:goldie-weight"
	job.RouteMissingOrgans = []string{"goldie-weight"}
	shell := admissionLiveRouteTurnCandidateShellForJob(job)
	execution := admissionLiveRouteTurnCandidateExecutionForShellWithRuntime(shell, text, admissionLiveRouteTurnCandidateExecutionRuntime{
		Runner:     admissionLiveRouteTurnCandidateExecutionRunnerNanoDirect,
		Status:     admissionLiveRouteTurnCandidateExecutionStatusSucceeded,
		StdoutHash: hashJSON(text),
	})
	generatorAdapter := admissionLiveRouteTurnGeneratorAdapterForExecution(execution)
	draft := admissionLiveRouteTurnCandidateDraftForAdapter(generatorAdapter)
	review := admissionLiveRouteTurnCandidateReviewForDraft(obs, draft)
	admission := admissionLiveRouteTurnCandidateAdmissionForDraftReview(obs, draft, review)
	adapter := admissionLiveRouteTurnCandidateAdmissionAdapterForDraft(admission, draft)
	candidate := admissionLiveRouteTurnCandidateForAdmissionAdapter(draft, adapter)
	candidate = prepareDreamCandidateForAdmissionWithTurnObservation(NewInnerWorld(), candidate, obs)

	if !obs.Passed || !choice.Passed || !request.Passed || !job.Passed || !shell.Passed || !execution.Passed ||
		!generatorAdapter.Passed || !draft.Passed || !review.Matched || !admission.Passed || !adapter.Passed ||
		candidate.Schema != "arianna.dream_candidate.v1" || candidate.Admission == nil ||
		!candidate.Admission.Passed || candidate.Admission.LiveRouteChoice == nil || !candidate.Admission.LiveRouteChoice.Passed {
		t.Fatalf("test setup failed: obs=%+v choice=%+v request=%+v job=%+v shell=%+v execution=%+v generatorAdapter=%+v draft=%+v review=%+v admission=%+v adapter=%+v candidate=%+v",
			obs, choice, request, job, shell, execution, generatorAdapter, draft, review, admission, adapter, candidate)
	}
	return execution, generatorAdapter, draft, admission, adapter, candidate
}

func TestAdmissionLiveRouteTurnCandidateAdmissionDecisionCarriesBoundary(t *testing.T) {
	execution, generatorAdapter, draft, admission, adapter, candidate := admissionLiveRouteTurnCandidateAdmissionDecisionChainForTest(t)

	decision := admissionLiveRouteTurnCandidateAdmissionDecisionForShadow(
		execution,
		generatorAdapter,
		draft,
		admission,
		adapter,
		candidate,
	)

	if !decision.Passed ||
		decision.DecisionID == "" ||
		decision.BodyInventoryStatus != "degraded" ||
		decision.RouteAvailabilityStatus != "available" ||
		decision.RouteAvailabilityReason != "optional_route_organs_missing:goldie-weight" ||
		!reflect.DeepEqual(decision.RouteMissingOrgans, []string{"goldie-weight"}) {
		t.Fatalf("decision should carry route boundary: %+v", decision)
	}
}

func TestAdmissionLiveRouteTurnCandidateAdmissionDecisionRejectsAdapterBoundaryDrift(t *testing.T) {
	execution, generatorAdapter, draft, admission, adapter, candidate := admissionLiveRouteTurnCandidateAdmissionDecisionChainForTest(t)
	adapter.RouteAvailabilityReason = "tampered-decision-boundary"

	decision := admissionLiveRouteTurnCandidateAdmissionDecisionForShadow(
		execution,
		generatorAdapter,
		draft,
		admission,
		adapter,
		candidate,
	)

	if decision.Passed ||
		decision.DecisionID != "" ||
		!strings.Contains(decision.Reason, "candidate_execution_route_boundary_mismatch") {
		t.Fatalf("adapter boundary drift should fail closed before decision id: %+v", decision)
	}
}

func TestAdmissionLiveRouteTurnCandidateAdmissionDecisionRejectsCandidateBoundaryDrift(t *testing.T) {
	execution, generatorAdapter, draft, admission, adapter, candidate := admissionLiveRouteTurnCandidateAdmissionDecisionChainForTest(t)
	candidate.LiveRouteCandidateAdmission.RouteMissingOrgans = append(
		admissionLiveRouteMissingOrgansCopy(candidate.LiveRouteCandidateAdmission.RouteMissingOrgans),
		"doe-bridge",
	)

	decision := admissionLiveRouteTurnCandidateAdmissionDecisionForShadow(
		execution,
		generatorAdapter,
		draft,
		admission,
		adapter,
		candidate,
	)

	if decision.Passed ||
		decision.DecisionID != "" ||
		!strings.Contains(decision.Reason, "shadow_dream_candidate_route_boundary_mismatch") {
		t.Fatalf("candidate boundary drift should fail closed before decision id: %+v", decision)
	}
}

func TestAdmissionLiveRouteTurnCandidateAdmissionDecisionForShadow(t *testing.T) {
	t.Setenv("AM_DREAM_ADMISSION", dreamAdmissionShadow)
	t.Setenv("AM_DREAM_ADMISSION_REQUIRE_LIVE_ROUTE_PLAN", "1")

	text := "The dream remembers the field and keeps one admission chain."
	obs := admissionLiveRouteTurnObservationForHuman("Tell me what the dream should remember.")
	choice := admissionLiveRouteTurnChoiceForObservation(obs)
	request := admissionLiveRouteTurnRequestForChoice(choice)
	job := admissionLiveRouteTurnGenerationJobForRequest(request)
	job.BodyInventoryStatus = "degraded"
	job.RouteAvailabilityStatus = "available"
	job.RouteAvailabilityReason = "optional_route_organs_missing:goldie-weight"
	job.RouteMissingOrgans = []string{"goldie-weight"}
	shell := admissionLiveRouteTurnCandidateShellForJob(job)
	execution := admissionLiveRouteTurnCandidateExecutionForShellWithRuntime(shell, text, admissionLiveRouteTurnCandidateExecutionRuntime{
		Runner:     admissionLiveRouteTurnCandidateExecutionRunnerNanoDirect,
		Status:     admissionLiveRouteTurnCandidateExecutionStatusSucceeded,
		StdoutHash: hashJSON(text),
	})
	generatorAdapter := admissionLiveRouteTurnGeneratorAdapterForExecution(execution)
	draft := admissionLiveRouteTurnCandidateDraftForAdapter(generatorAdapter)
	review := admissionLiveRouteTurnCandidateReviewForDraft(obs, draft)
	admission := admissionLiveRouteTurnCandidateAdmissionForDraftReview(obs, draft, review)
	adapter := admissionLiveRouteTurnCandidateAdmissionAdapterForDraft(admission, draft)
	candidate := admissionLiveRouteTurnCandidateForAdmissionAdapter(draft, adapter)
	candidate = prepareDreamCandidateForAdmissionWithTurnObservation(NewInnerWorld(), candidate, obs)

	decision := admissionLiveRouteTurnCandidateAdmissionDecisionForShadow(
		execution,
		generatorAdapter,
		draft,
		admission,
		adapter,
		candidate,
	)
	if decision.Schema != admissionLiveRouteTurnCandidateAdmissionDecisionSchema ||
		decision.Timing != "shadow_candidate_live_preflight" ||
		decision.Decision != "shadow_ready" ||
		!strings.HasPrefix(decision.DecisionID, "decision-") ||
		!decision.Passed ||
		!decision.LiveReady ||
		decision.MutatesState ||
		!decision.AdmissionPolicyPassed ||
		!decision.LiveRouteChoicePassed ||
		decision.DreamAccepted ||
		decision.Reason != "shadow ready; live mutation still disabled" {
		t.Fatalf("bad candidate admission decision: %+v", decision)
	}
	if decision.CandidateExecutionID != execution.ExecutionID ||
		decision.GeneratorAdapterID != generatorAdapter.AdapterID ||
		decision.CandidateDraftID != draft.DraftID ||
		decision.HandoffID != admission.HandoffID ||
		decision.AdmissionAdapterID != adapter.AdmissionAdapterID ||
		decision.DreamCandidateRunID != candidate.RunID ||
		decision.CandidateTextHash != hashJSON(text) ||
		decision.BodyInventoryStatus != "degraded" ||
		decision.RouteAvailabilityStatus != "available" ||
		decision.RouteAvailabilityReason != "optional_route_organs_missing:goldie-weight" ||
		!reflect.DeepEqual(decision.RouteMissingOrgans, []string{"goldie-weight"}) ||
		decision.TurnTextHash != obs.TextHash {
		t.Fatalf("decision lost provenance: decision=%+v execution=%+v adapter=%+v draft=%+v admission=%+v candidate=%+v",
			decision, execution, generatorAdapter, draft, admission, candidate)
	}
	assertRouteBoundary := func(name, bodyStatus, availabilityStatus, availabilityReason string, missingOrgans []string) {
		t.Helper()
		if !admissionLiveRouteBoundaryFieldsEqual(
			bodyStatus,
			availabilityStatus,
			availabilityReason,
			missingOrgans,
			decision.BodyInventoryStatus,
			decision.RouteAvailabilityStatus,
			decision.RouteAvailabilityReason,
			decision.RouteMissingOrgans,
		) {
			t.Fatalf("%s lost route boundary: body=%q availability=%q reason=%q missing=%v decision=%+v",
				name, bodyStatus, availabilityStatus, availabilityReason, missingOrgans, decision)
		}
	}
	promotion := admissionLiveRouteTurnCandidateAdmissionPromotionForDecision(decision)
	if promotion.Schema != admissionLiveRouteTurnCandidateAdmissionPromotionSchema ||
		promotion.Timing != "admission_decision_consumer" ||
		promotion.Promotion != "pending_live_admission" ||
		!strings.HasPrefix(promotion.PromotionID, "promotion-") ||
		!promotion.Passed ||
		!promotion.LiveReady ||
		promotion.LiveAdmissionEnabled ||
		promotion.MutatesState ||
		!promotion.SourceDecisionPassed ||
		promotion.Reason != "shadow decision consumed; live admission still disabled" {
		t.Fatalf("bad candidate admission promotion: %+v", promotion)
	}
	if promotion.AdmissionDecisionID != decision.DecisionID ||
		promotion.AdmissionAdapterID != adapter.AdmissionAdapterID ||
		promotion.CandidateExecutionID != execution.ExecutionID ||
		promotion.CandidateDraftID != draft.DraftID ||
		promotion.CandidateRunID != candidate.RunID ||
		promotion.CandidateTextHash != hashJSON(text) ||
		!admissionLiveRouteBoundaryFieldsEqual(
			promotion.BodyInventoryStatus,
			promotion.RouteAvailabilityStatus,
			promotion.RouteAvailabilityReason,
			promotion.RouteMissingOrgans,
			decision.BodyInventoryStatus,
			decision.RouteAvailabilityStatus,
			decision.RouteAvailabilityReason,
			decision.RouteMissingOrgans,
		) ||
		promotion.TurnTextHash != obs.TextHash {
		t.Fatalf("promotion lost provenance: promotion=%+v decision=%+v", promotion, decision)
	}
	sw := admissionLiveRouteTurnCandidateAdmissionSwitchForPromotion(promotion)
	if sw.Schema != admissionLiveRouteTurnCandidateAdmissionSwitchSchema ||
		sw.Timing != "live_admission_switch_guard" ||
		sw.SwitchState != "disabled" ||
		sw.SwitchAction != "hold_pending_live_admission" ||
		!strings.HasPrefix(sw.SwitchID, "switch-") ||
		!sw.Passed ||
		!sw.LiveReady ||
		sw.LiveAdmissionEnabled ||
		sw.AdmissionAllowed ||
		sw.MutatesState ||
		!sw.SourceDecisionPassed ||
		!sw.SourcePromotionPassed ||
		sw.Reason != "live admission switch disabled; pending promotion held without mutation" {
		t.Fatalf("bad candidate admission switch: %+v", sw)
	}
	if sw.AdmissionPromotionID != promotion.PromotionID ||
		sw.AdmissionDecisionID != decision.DecisionID ||
		sw.AdmissionAdapterID != adapter.AdmissionAdapterID ||
		sw.CandidateExecutionID != execution.ExecutionID ||
		sw.CandidateDraftID != draft.DraftID ||
		sw.CandidateRunID != candidate.RunID ||
		sw.CandidateTextHash != hashJSON(text) ||
		!admissionLiveRouteBoundaryFieldsEqual(
			sw.BodyInventoryStatus,
			sw.RouteAvailabilityStatus,
			sw.RouteAvailabilityReason,
			sw.RouteMissingOrgans,
			decision.BodyInventoryStatus,
			decision.RouteAvailabilityStatus,
			decision.RouteAvailabilityReason,
			decision.RouteMissingOrgans,
		) ||
		sw.TurnTextHash != obs.TextHash {
		t.Fatalf("switch lost provenance: switch=%+v promotion=%+v", sw, promotion)
	}
	t.Setenv("AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_ENABLE_GATE_KEY", "")
	gate := admissionLiveRouteTurnCandidateAdmissionEnableGateForSwitch(sw)
	if gate.Schema != admissionLiveRouteTurnCandidateAdmissionEnableGateSchema ||
		gate.Timing != "live_admission_enable_gate" ||
		gate.EnableState != "disabled" ||
		gate.EnableAction != "require_operator_key" ||
		!strings.HasPrefix(gate.EnableGateID, "enable-") ||
		!gate.Passed ||
		!gate.LiveReady ||
		gate.LiveAdmissionEnabled ||
		gate.AdmissionAllowed ||
		gate.ManualEnableRequested ||
		gate.EnableKeyMatched ||
		gate.MutatesState ||
		!gate.SourceDecisionPassed ||
		!gate.SourcePromotionPassed ||
		!gate.SourceSwitchPassed ||
		gate.Reason != "live admission enable gate closed; operator key absent" {
		t.Fatalf("bad candidate admission enable gate: %+v", gate)
	}
	if gate.AdmissionSwitchID != sw.SwitchID ||
		gate.AdmissionPromotionID != promotion.PromotionID ||
		gate.AdmissionDecisionID != decision.DecisionID ||
		gate.AdmissionAdapterID != adapter.AdmissionAdapterID ||
		gate.CandidateExecutionID != execution.ExecutionID ||
		gate.CandidateDraftID != draft.DraftID ||
		gate.CandidateRunID != candidate.RunID ||
		gate.CandidateTextHash != hashJSON(text) ||
		!admissionLiveRouteBoundaryFieldsEqual(
			gate.BodyInventoryStatus,
			gate.RouteAvailabilityStatus,
			gate.RouteAvailabilityReason,
			gate.RouteMissingOrgans,
			decision.BodyInventoryStatus,
			decision.RouteAvailabilityStatus,
			decision.RouteAvailabilityReason,
			decision.RouteMissingOrgans,
		) ||
		gate.TurnTextHash != obs.TextHash {
		t.Fatalf("enable gate lost provenance: gate=%+v switch=%+v", gate, sw)
	}
	t.Setenv("AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_ENABLE_GATE_KEY", "wrong")
	wrongGate := admissionLiveRouteTurnCandidateAdmissionEnableGateForSwitch(sw)
	if wrongGate.Passed ||
		wrongGate.EnableGateID != "" ||
		wrongGate.EnableState != "blocked" ||
		!wrongGate.ManualEnableRequested ||
		wrongGate.EnableKeyMatched ||
		wrongGate.Reason != "live_admission_enable_gate_key_mismatch" {
		t.Fatalf("wrong enable gate key should fail closed: %+v", wrongGate)
	}
	t.Setenv("AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_ENABLE_GATE_KEY", admissionLiveRouteTurnCandidateAdmissionEnableGateConfirmation)
	armedGate := admissionLiveRouteTurnCandidateAdmissionEnableGateForSwitch(sw)
	if armedGate.Schema != admissionLiveRouteTurnCandidateAdmissionEnableGateSchema ||
		armedGate.EnableState != "armed_dry_run" ||
		armedGate.EnableAction != "would_enable_live_admission_dry_run" ||
		!strings.HasPrefix(armedGate.EnableGateID, "enable-") ||
		!armedGate.Passed ||
		!armedGate.ManualEnableRequested ||
		!armedGate.EnableKeyMatched ||
		armedGate.LiveAdmissionEnabled ||
		armedGate.AdmissionAllowed ||
		armedGate.MutatesState ||
		!admissionLiveRouteBoundaryFieldsEqual(
			armedGate.BodyInventoryStatus,
			armedGate.RouteAvailabilityStatus,
			armedGate.RouteAvailabilityReason,
			armedGate.RouteMissingOrgans,
			decision.BodyInventoryStatus,
			decision.RouteAvailabilityStatus,
			decision.RouteAvailabilityReason,
			decision.RouteMissingOrgans,
		) ||
		armedGate.Reason != "live admission enable key matched; dry-run still refuses mutation" {
		t.Fatalf("armed enable gate should remain dry-run and non-mutating: %+v", armedGate)
	}
	liveStage := admissionLiveRouteTurnCandidateAdmissionLiveStageForEnableGate(armedGate)
	if liveStage.Schema != admissionLiveRouteTurnCandidateAdmissionLiveStageSchema ||
		liveStage.Timing != "live_admission_candidate_stage" ||
		liveStage.StageState != "staged_dry_run" ||
		liveStage.StageAction != "stage_live_candidate_dry_run" ||
		!strings.HasPrefix(liveStage.LiveStageID, "stage-") ||
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
		!liveStage.SourceDecisionPassed ||
		!liveStage.SourcePromotionPassed ||
		!liveStage.SourceSwitchPassed ||
		!liveStage.SourceEnablePassed ||
		liveStage.Reason != "live admission candidate staged as dry-run; writer and rollback remain absent" {
		t.Fatalf("armed enable gate should only stage a dry-run live candidate: %+v", liveStage)
	}
	if liveStage.AdmissionEnableGateID != armedGate.EnableGateID ||
		liveStage.AdmissionSwitchID != sw.SwitchID ||
		liveStage.AdmissionPromotionID != promotion.PromotionID ||
		liveStage.AdmissionDecisionID != decision.DecisionID ||
		liveStage.AdmissionAdapterID != adapter.AdmissionAdapterID ||
		liveStage.CandidateExecutionID != execution.ExecutionID ||
		liveStage.CandidateDraftID != draft.DraftID ||
		liveStage.CandidateRunID != candidate.RunID ||
		liveStage.CandidateTextHash != hashJSON(text) ||
		!admissionLiveRouteBoundaryFieldsEqual(
			liveStage.BodyInventoryStatus,
			liveStage.RouteAvailabilityStatus,
			liveStage.RouteAvailabilityReason,
			liveStage.RouteMissingOrgans,
			decision.BodyInventoryStatus,
			decision.RouteAvailabilityStatus,
			decision.RouteAvailabilityReason,
			decision.RouteMissingOrgans,
		) ||
		liveStage.TurnTextHash != obs.TextHash {
		t.Fatalf("live stage lost provenance: stage=%+v gate=%+v", liveStage, armedGate)
	}
	writerPreflight := admissionLiveRouteTurnCandidateAdmissionWriterPreflightForLiveStage(liveStage)
	if writerPreflight.Schema != admissionLiveRouteTurnCandidateAdmissionWriterPreflightSchema ||
		writerPreflight.Timing != "live_admission_writer_preflight" ||
		writerPreflight.WriterState != "absent" ||
		writerPreflight.WriterAction != "require_writer_contract" ||
		writerPreflight.RollbackState != "absent" ||
		writerPreflight.RollbackAction != "require_rollback_contract" ||
		!strings.HasPrefix(writerPreflight.WriterPreflightID, "writer-") ||
		!writerPreflight.Passed ||
		!writerPreflight.LiveReady ||
		writerPreflight.LiveAdmissionEnabled ||
		writerPreflight.AdmissionAllowed ||
		!writerPreflight.ManualEnableRequested ||
		!writerPreflight.EnableKeyMatched ||
		!writerPreflight.RequiresWriter ||
		writerPreflight.WriterReady ||
		!writerPreflight.RequiresRollback ||
		writerPreflight.RollbackReady ||
		writerPreflight.WriteAllowed ||
		writerPreflight.MutatesState ||
		!writerPreflight.SourceDecisionPassed ||
		!writerPreflight.SourcePromotionPassed ||
		!writerPreflight.SourceSwitchPassed ||
		!writerPreflight.SourceEnablePassed ||
		!writerPreflight.SourceStagePassed ||
		writerPreflight.Reason != "writer and rollback absent; live admission remains staged only" {
		t.Fatalf("live stage should only preflight absent writer and rollback: %+v", writerPreflight)
	}
	if writerPreflight.AdmissionLiveStageID != liveStage.LiveStageID ||
		writerPreflight.AdmissionEnableGateID != armedGate.EnableGateID ||
		writerPreflight.AdmissionSwitchID != sw.SwitchID ||
		writerPreflight.AdmissionPromotionID != promotion.PromotionID ||
		writerPreflight.AdmissionDecisionID != decision.DecisionID ||
		writerPreflight.AdmissionAdapterID != adapter.AdmissionAdapterID ||
		writerPreflight.CandidateExecutionID != execution.ExecutionID ||
		writerPreflight.CandidateDraftID != draft.DraftID ||
		writerPreflight.CandidateRunID != candidate.RunID ||
		writerPreflight.CandidateTextHash != hashJSON(text) ||
		writerPreflight.TurnTextHash != obs.TextHash {
		t.Fatalf("writer preflight lost provenance: preflight=%+v stage=%+v", writerPreflight, liveStage)
	}
	assertRouteBoundary(
		"writer preflight",
		writerPreflight.BodyInventoryStatus,
		writerPreflight.RouteAvailabilityStatus,
		writerPreflight.RouteAvailabilityReason,
		writerPreflight.RouteMissingOrgans,
	)
	writerInventory := admissionLiveRouteTurnCandidateAdmissionWriterInventoryForPreflight(writerPreflight)
	if writerInventory.Schema != admissionLiveRouteTurnCandidateAdmissionWriterInventorySchema ||
		writerInventory.Timing != "live_admission_writer_inventory" ||
		writerInventory.InventoryState != "contracts_absent" ||
		writerInventory.InventoryAction != "name_required_contracts" ||
		writerInventory.WriterContract != "live_admission_writer.v1" ||
		writerInventory.RollbackContract != "live_admission_rollback.v1" ||
		writerInventory.AdmissionLedgerContract != "live_admission_ledger.v1" ||
		writerInventory.WriterContractPresent ||
		writerInventory.RollbackContractPresent ||
		writerInventory.LedgerContractPresent ||
		writerInventory.ContractsReady ||
		!strings.HasPrefix(writerInventory.WriterInventoryID, "writer-inventory-") ||
		!writerInventory.Passed ||
		!writerInventory.LiveReady ||
		writerInventory.LiveAdmissionEnabled ||
		writerInventory.AdmissionAllowed ||
		!writerInventory.ManualEnableRequested ||
		!writerInventory.EnableKeyMatched ||
		!writerInventory.RequiresWriter ||
		writerInventory.WriterReady ||
		!writerInventory.RequiresRollback ||
		writerInventory.RollbackReady ||
		writerInventory.WriteAllowed ||
		writerInventory.MutatesState ||
		!writerInventory.SourceDecisionPassed ||
		!writerInventory.SourcePromotionPassed ||
		!writerInventory.SourceSwitchPassed ||
		!writerInventory.SourceEnablePassed ||
		!writerInventory.SourceStagePassed ||
		!writerInventory.SourceWriterPreflightPassed ||
		writerInventory.Reason != "writer inventory recorded required contracts; live admission remains blocked" {
		t.Fatalf("writer inventory should only name absent contracts: %+v", writerInventory)
	}
	if writerInventory.AdmissionWriterPreflightID != writerPreflight.WriterPreflightID ||
		writerInventory.AdmissionLiveStageID != liveStage.LiveStageID ||
		writerInventory.AdmissionEnableGateID != armedGate.EnableGateID ||
		writerInventory.AdmissionSwitchID != sw.SwitchID ||
		writerInventory.AdmissionPromotionID != promotion.PromotionID ||
		writerInventory.AdmissionDecisionID != decision.DecisionID ||
		writerInventory.AdmissionAdapterID != adapter.AdmissionAdapterID ||
		writerInventory.CandidateExecutionID != execution.ExecutionID ||
		writerInventory.CandidateDraftID != draft.DraftID ||
		writerInventory.CandidateRunID != candidate.RunID ||
		writerInventory.CandidateTextHash != hashJSON(text) ||
		writerInventory.TurnTextHash != obs.TextHash {
		t.Fatalf("writer inventory lost provenance: inventory=%+v preflight=%+v", writerInventory, writerPreflight)
	}
	assertRouteBoundary(
		"writer inventory",
		writerInventory.BodyInventoryStatus,
		writerInventory.RouteAvailabilityStatus,
		writerInventory.RouteAvailabilityReason,
		writerInventory.RouteMissingOrgans,
	)
	writerContract := admissionLiveRouteTurnCandidateAdmissionWriterContractForInventory(writerInventory)
	if writerContract.Schema != admissionLiveRouteTurnCandidateAdmissionWriterContractSchema ||
		writerContract.Timing != "live_admission_writer_contract" ||
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
		!strings.HasPrefix(writerContract.WriterContractID, "writer-contract-") ||
		!writerContract.Passed ||
		!writerContract.LiveReady ||
		writerContract.LiveAdmissionEnabled ||
		writerContract.AdmissionAllowed ||
		!writerContract.ManualEnableRequested ||
		!writerContract.EnableKeyMatched ||
		!writerContract.RequiresWriter ||
		writerContract.WriterReady ||
		!writerContract.RequiresRollback ||
		writerContract.RollbackReady ||
		writerContract.WriteAllowed ||
		writerContract.MutatesState ||
		!writerContract.SourceDecisionPassed ||
		!writerContract.SourcePromotionPassed ||
		!writerContract.SourceSwitchPassed ||
		!writerContract.SourceEnablePassed ||
		!writerContract.SourceStagePassed ||
		!writerContract.SourceWriterPreflightPassed ||
		!writerContract.SourceWriterInventoryPassed ||
		writerContract.Reason != "writer contract shape drafted; implementation and ledger remain absent" {
		t.Fatalf("writer contract should only draft a non-mutating shape: %+v", writerContract)
	}
	if writerContract.AdmissionWriterInventoryID != writerInventory.WriterInventoryID ||
		writerContract.AdmissionWriterPreflightID != writerPreflight.WriterPreflightID ||
		writerContract.AdmissionLiveStageID != liveStage.LiveStageID ||
		writerContract.AdmissionEnableGateID != armedGate.EnableGateID ||
		writerContract.AdmissionSwitchID != sw.SwitchID ||
		writerContract.AdmissionPromotionID != promotion.PromotionID ||
		writerContract.AdmissionDecisionID != decision.DecisionID ||
		writerContract.AdmissionAdapterID != adapter.AdmissionAdapterID ||
		writerContract.CandidateExecutionID != execution.ExecutionID ||
		writerContract.CandidateDraftID != draft.DraftID ||
		writerContract.CandidateRunID != candidate.RunID ||
		writerContract.CandidateTextHash != hashJSON(text) ||
		writerContract.TurnTextHash != obs.TextHash {
		t.Fatalf("writer contract lost provenance: contract=%+v inventory=%+v", writerContract, writerInventory)
	}
	assertRouteBoundary(
		"writer contract",
		writerContract.BodyInventoryStatus,
		writerContract.RouteAvailabilityStatus,
		writerContract.RouteAvailabilityReason,
		writerContract.RouteMissingOrgans,
	)
	ledger := admissionLiveRouteTurnCandidateAdmissionLedgerForWriterContract(writerContract)
	if ledger.Schema != admissionLiveRouteTurnCandidateAdmissionLedgerSchema ||
		ledger.Timing != "live_admission_ledger" ||
		ledger.LedgerState != "receipt_drafted_dry_run" ||
		ledger.LedgerAction != "append_candidate_admission_receipt_dry_run" ||
		ledger.LedgerContract != "live_admission_ledger.v1" ||
		ledger.LedgerMode != "append_only_dry_run" ||
		ledger.LedgerEntryKind != "dream_candidate_admission" ||
		ledger.LedgerEntryStatus != "shadow_candidate_receipt" ||
		ledger.LedgerReceiptShape != "candidate_contract_provenance" ||
		!ledger.LedgerAppendReady ||
		ledger.LedgerReceiptPersisted ||
		ledger.LedgerImplementationReady ||
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
		ledger.ContractsReady ||
		!strings.HasPrefix(ledger.AdmissionLedgerID, "admission-ledger-") ||
		!ledger.Passed ||
		!ledger.LiveReady ||
		ledger.LiveAdmissionEnabled ||
		ledger.AdmissionAllowed ||
		!ledger.ManualEnableRequested ||
		!ledger.EnableKeyMatched ||
		!ledger.RequiresWriter ||
		ledger.WriterReady ||
		!ledger.RequiresRollback ||
		ledger.RollbackReady ||
		ledger.WriteAllowed ||
		ledger.MutatesState ||
		!ledger.SourceDecisionPassed ||
		!ledger.SourcePromotionPassed ||
		!ledger.SourceSwitchPassed ||
		!ledger.SourceEnablePassed ||
		!ledger.SourceStagePassed ||
		!ledger.SourceWriterPreflightPassed ||
		!ledger.SourceWriterInventoryPassed ||
		!ledger.SourceWriterContractPassed ||
		ledger.Reason != "admission ledger dry-run receipt drafted; no live write occurred" {
		t.Fatalf("ledger should only draft a non-mutating append-only receipt: %+v", ledger)
	}
	if ledger.AdmissionWriterContractID != writerContract.WriterContractID ||
		ledger.AdmissionWriterInventoryID != writerInventory.WriterInventoryID ||
		ledger.AdmissionWriterPreflightID != writerPreflight.WriterPreflightID ||
		ledger.AdmissionLiveStageID != liveStage.LiveStageID ||
		ledger.AdmissionEnableGateID != armedGate.EnableGateID ||
		ledger.AdmissionSwitchID != sw.SwitchID ||
		ledger.AdmissionPromotionID != promotion.PromotionID ||
		ledger.AdmissionDecisionID != decision.DecisionID ||
		ledger.AdmissionAdapterID != adapter.AdmissionAdapterID ||
		ledger.CandidateExecutionID != execution.ExecutionID ||
		ledger.CandidateDraftID != draft.DraftID ||
		ledger.CandidateRunID != candidate.RunID ||
		ledger.CandidateTextHash != hashJSON(text) ||
		ledger.TurnTextHash != obs.TextHash {
		t.Fatalf("ledger lost provenance: ledger=%+v contract=%+v", ledger, writerContract)
	}
	assertRouteBoundary(
		"admission ledger",
		ledger.BodyInventoryStatus,
		ledger.RouteAvailabilityStatus,
		ledger.RouteAvailabilityReason,
		ledger.RouteMissingOrgans,
	)
	writerImpl := admissionLiveRouteTurnCandidateAdmissionWriterImplementationForLedger(ledger)
	if writerImpl.Schema != admissionLiveRouteTurnCandidateAdmissionWriterImplSchema ||
		writerImpl.Timing != "live_admission_writer_implementation" ||
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
		writerImpl.LedgerState != "receipt_drafted_dry_run" ||
		writerImpl.LedgerAction != "append_candidate_admission_receipt_dry_run" ||
		writerImpl.LedgerContract != "live_admission_ledger.v1" ||
		writerImpl.LedgerMode != "append_only_dry_run" ||
		writerImpl.LedgerEntryKind != "dream_candidate_admission" ||
		writerImpl.LedgerEntryStatus != "shadow_candidate_receipt" ||
		writerImpl.LedgerReceiptShape != "candidate_contract_provenance" ||
		!writerImpl.LedgerAppendReady ||
		writerImpl.LedgerReceiptPersisted ||
		writerImpl.LedgerImplementationReady ||
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
		writerImpl.ContractsReady ||
		!strings.HasPrefix(writerImpl.WriterImplementationID, "writer-implementation-") ||
		!writerImpl.Passed ||
		!writerImpl.LiveReady ||
		writerImpl.LiveAdmissionEnabled ||
		writerImpl.AdmissionAllowed ||
		!writerImpl.ManualEnableRequested ||
		!writerImpl.EnableKeyMatched ||
		!writerImpl.RequiresWriter ||
		writerImpl.WriterReady ||
		!writerImpl.RequiresRollback ||
		writerImpl.RollbackReady ||
		writerImpl.WriteAllowed ||
		writerImpl.MutatesState ||
		!writerImpl.SourceDecisionPassed ||
		!writerImpl.SourcePromotionPassed ||
		!writerImpl.SourceSwitchPassed ||
		!writerImpl.SourceEnablePassed ||
		!writerImpl.SourceStagePassed ||
		!writerImpl.SourceWriterPreflightPassed ||
		!writerImpl.SourceWriterInventoryPassed ||
		!writerImpl.SourceWriterContractPassed ||
		!writerImpl.SourceLedgerPassed ||
		writerImpl.Reason != "writer implementation contract drafted; append-only log boundary only" {
		t.Fatalf("writer implementation should only draft a non-mutating append-only contract: %+v", writerImpl)
	}
	if writerImpl.AdmissionLedgerID != ledger.AdmissionLedgerID ||
		writerImpl.AdmissionWriterContractID != writerContract.WriterContractID ||
		writerImpl.AdmissionWriterInventoryID != writerInventory.WriterInventoryID ||
		writerImpl.AdmissionWriterPreflightID != writerPreflight.WriterPreflightID ||
		writerImpl.AdmissionLiveStageID != liveStage.LiveStageID ||
		writerImpl.AdmissionEnableGateID != armedGate.EnableGateID ||
		writerImpl.AdmissionSwitchID != sw.SwitchID ||
		writerImpl.AdmissionPromotionID != promotion.PromotionID ||
		writerImpl.AdmissionDecisionID != decision.DecisionID ||
		writerImpl.AdmissionAdapterID != adapter.AdmissionAdapterID ||
		writerImpl.CandidateExecutionID != execution.ExecutionID ||
		writerImpl.CandidateDraftID != draft.DraftID ||
		writerImpl.CandidateRunID != candidate.RunID ||
		writerImpl.CandidateTextHash != hashJSON(text) ||
		writerImpl.TurnTextHash != obs.TextHash {
		t.Fatalf("writer implementation lost provenance: impl=%+v ledger=%+v", writerImpl, ledger)
	}
	assertRouteBoundary(
		"writer implementation",
		writerImpl.BodyInventoryStatus,
		writerImpl.RouteAvailabilityStatus,
		writerImpl.RouteAvailabilityReason,
		writerImpl.RouteMissingOrgans,
	)
	writerReceipt := admissionLiveRouteTurnCandidateAdmissionWriterReceiptForImplementation(writerImpl)
	if writerReceipt.Schema != admissionLiveRouteTurnCandidateAdmissionWriterReceiptSchema ||
		writerReceipt.Timing != "live_admission_writer_receipt" ||
		writerReceipt.WriterReceiptState != "shadow_receipt_appended_dry_run" ||
		writerReceipt.WriterReceiptAction != "append_shadow_candidate_receipt_dry_run" ||
		writerReceipt.WriterReceiptKind != "dream_candidate_admission" ||
		writerReceipt.WriterReceiptTarget != "shadow_receipt_log" ||
		writerReceipt.WriterReceiptMode != "append_only_dry_run" ||
		writerReceipt.WriterReceiptShape != "candidate_contract_provenance" ||
		!writerReceipt.WriterReceiptPersisted ||
		!writerReceipt.ShadowWriteAllowed ||
		writerReceipt.BodyTarget != "none" ||
		!writerReceipt.AppendOnly ||
		!writerReceipt.RollbackRequired ||
		!writerReceipt.ImplementationContractReady ||
		!writerReceipt.WriterReady ||
		writerReceipt.WriterState != "ready_dry_run" ||
		writerReceipt.WriterAction != "append_shadow_candidate_receipt_dry_run" ||
		!writerReceipt.WriterImplementationReady ||
		writerReceipt.RollbackReady ||
		writerReceipt.RollbackImplementationReady ||
		writerReceipt.LedgerImplementationReady ||
		writerReceipt.ContractsReady ||
		writerReceipt.WriteAllowed ||
		writerReceipt.AdmissionAllowed ||
		writerReceipt.LiveAdmissionEnabled ||
		writerReceipt.MutatesState ||
		!strings.HasPrefix(writerReceipt.WriterReceiptID, "writer-receipt-") ||
		!writerReceipt.Passed ||
		!writerReceipt.LiveReady ||
		!writerReceipt.ManualEnableRequested ||
		!writerReceipt.EnableKeyMatched ||
		!writerReceipt.RequiresWriter ||
		!writerReceipt.RequiresRollback ||
		!writerReceipt.SourceWriterImplementationPassed ||
		writerReceipt.SourceWriterImplementationID != writerImpl.WriterImplementationID ||
		writerReceipt.SourceWriterImplementationEntrypoint != "append_shadow_candidate_receipt_dry_run" ||
		writerReceipt.SourceLedgerImplementationEntrypoint != "append_admission_ledger_receipt_dry_run" ||
		writerReceipt.SourceRollbackImplementationEntrypoint != "remove_exact_shadow_candidate_receipt_dry_run" ||
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
		t.Fatalf("writer receipt should append only to the shadow log: %+v", writerReceipt)
	}
	if writerReceipt.WriterImplementationID != writerImpl.WriterImplementationID ||
		writerReceipt.AdmissionLedgerID != ledger.AdmissionLedgerID ||
		writerReceipt.AdmissionWriterContractID != writerContract.WriterContractID ||
		writerReceipt.AdmissionWriterInventoryID != writerInventory.WriterInventoryID ||
		writerReceipt.AdmissionWriterPreflightID != writerPreflight.WriterPreflightID ||
		writerReceipt.AdmissionLiveStageID != liveStage.LiveStageID ||
		writerReceipt.AdmissionEnableGateID != armedGate.EnableGateID ||
		writerReceipt.AdmissionSwitchID != sw.SwitchID ||
		writerReceipt.AdmissionPromotionID != promotion.PromotionID ||
		writerReceipt.AdmissionDecisionID != decision.DecisionID ||
		writerReceipt.AdmissionAdapterID != adapter.AdmissionAdapterID ||
		writerReceipt.CandidateExecutionID != execution.ExecutionID ||
		writerReceipt.CandidateDraftID != draft.DraftID ||
		writerReceipt.CandidateRunID != candidate.RunID ||
		writerReceipt.CandidateTextHash != hashJSON(text) ||
		writerReceipt.TurnTextHash != obs.TextHash {
		t.Fatalf("writer receipt lost provenance: receipt=%+v impl=%+v", writerReceipt, writerImpl)
	}
	assertRouteBoundary(
		"writer receipt",
		writerReceipt.BodyInventoryStatus,
		writerReceipt.RouteAvailabilityStatus,
		writerReceipt.RouteAvailabilityReason,
		writerReceipt.RouteMissingOrgans,
	)
	rollbackImpl := admissionLiveRouteTurnCandidateAdmissionRollbackImplementationForWriterReceipt(writerReceipt)
	if rollbackImpl.Schema != admissionLiveRouteTurnCandidateAdmissionRollbackImplSchema ||
		rollbackImpl.Timing != "live_admission_rollback_implementation" ||
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
		!rollbackImpl.WriterReady ||
		rollbackImpl.WriterState != "ready_dry_run" ||
		rollbackImpl.WriterAction != "append_shadow_candidate_receipt_dry_run" ||
		!rollbackImpl.RollbackReady ||
		rollbackImpl.RollbackState != "ready_dry_run" ||
		rollbackImpl.RollbackAction != "remove_exact_shadow_candidate_receipt_dry_run" ||
		!rollbackImpl.WriterImplementationReady ||
		!rollbackImpl.RollbackImplementationReady ||
		rollbackImpl.LedgerImplementationReady ||
		rollbackImpl.ContractsReady ||
		rollbackImpl.WriteAllowed ||
		rollbackImpl.AdmissionAllowed ||
		rollbackImpl.LiveAdmissionEnabled ||
		rollbackImpl.MutatesState ||
		!strings.HasPrefix(rollbackImpl.RollbackImplementationID, "rollback-implementation-") ||
		!rollbackImpl.Passed ||
		!rollbackImpl.LiveReady ||
		!rollbackImpl.ManualEnableRequested ||
		!rollbackImpl.EnableKeyMatched ||
		!rollbackImpl.RequiresWriter ||
		!rollbackImpl.RequiresRollback ||
		rollbackImpl.SourceWriterReceiptSchema != admissionLiveRouteTurnCandidateAdmissionWriterReceiptSchema ||
		!rollbackImpl.SourceWriterReceiptPassed ||
		rollbackImpl.SourceWriterReceiptID != writerReceipt.WriterReceiptID ||
		rollbackImpl.SourceWriterReceiptAction != "append_shadow_candidate_receipt_dry_run" ||
		!rollbackImpl.SourceWriterReceiptPersisted ||
		!rollbackImpl.SourceWriterReceiptShadowWritable ||
		rollbackImpl.Reason != "rollback implementation drafted for exact writer receipt; body write remains disabled" {
		t.Fatalf("rollback implementation should prove exact dry-run rollback only: %+v", rollbackImpl)
	}
	if rollbackImpl.WriterReceiptID != writerReceipt.WriterReceiptID ||
		rollbackImpl.WriterImplementationID != writerImpl.WriterImplementationID ||
		rollbackImpl.AdmissionLedgerID != ledger.AdmissionLedgerID ||
		rollbackImpl.AdmissionWriterContractID != writerContract.WriterContractID ||
		rollbackImpl.AdmissionWriterInventoryID != writerInventory.WriterInventoryID ||
		rollbackImpl.AdmissionWriterPreflightID != writerPreflight.WriterPreflightID ||
		rollbackImpl.AdmissionLiveStageID != liveStage.LiveStageID ||
		rollbackImpl.AdmissionEnableGateID != armedGate.EnableGateID ||
		rollbackImpl.AdmissionSwitchID != sw.SwitchID ||
		rollbackImpl.AdmissionPromotionID != promotion.PromotionID ||
		rollbackImpl.AdmissionDecisionID != decision.DecisionID ||
		rollbackImpl.AdmissionAdapterID != adapter.AdmissionAdapterID ||
		rollbackImpl.CandidateExecutionID != execution.ExecutionID ||
		rollbackImpl.CandidateDraftID != draft.DraftID ||
		rollbackImpl.CandidateRunID != candidate.RunID ||
		rollbackImpl.CandidateTextHash != hashJSON(text) ||
		rollbackImpl.TurnTextHash != obs.TextHash {
		t.Fatalf("rollback implementation lost provenance: rollback=%+v receipt=%+v", rollbackImpl, writerReceipt)
	}
	ledgerImpl := admissionLiveRouteTurnCandidateAdmissionLedgerImplementationForRollbackImplementation(rollbackImpl)
	if ledgerImpl.Schema != admissionLiveRouteTurnCandidateAdmissionLedgerImplSchema ||
		ledgerImpl.Timing != "live_admission_ledger_implementation" ||
		ledgerImpl.LedgerImplementationState != "ledger_contract_drafted_dry_run" ||
		ledgerImpl.LedgerImplementationAction != "append_admission_ledger_receipt_dry_run" ||
		ledgerImpl.LedgerEntrypointResolved != "append_admission_ledger_receipt_dry_run" ||
		ledgerImpl.LedgerImplementationTarget != "admission_ledger" ||
		ledgerImpl.LedgerImplementationTargetKind != "dream_candidate_admission" ||
		ledgerImpl.LedgerImplementationTargetMode != "append_only_dry_run" ||
		!ledgerImpl.LedgerImplementationAppendOnly ||
		!ledgerImpl.LedgerImplementationDryRunOnly ||
		ledgerImpl.LedgerImplementationReceiptPersisted ||
		!ledgerImpl.WriterReady ||
		ledgerImpl.WriterState != "ready_dry_run" ||
		ledgerImpl.WriterAction != "append_shadow_candidate_receipt_dry_run" ||
		!ledgerImpl.RollbackReady ||
		ledgerImpl.RollbackState != "ready_dry_run" ||
		ledgerImpl.RollbackAction != "remove_exact_shadow_candidate_receipt_dry_run" ||
		!ledgerImpl.WriterImplementationReady ||
		!ledgerImpl.RollbackImplementationReady ||
		!ledgerImpl.LedgerImplementationReady ||
		ledgerImpl.ContractsReady ||
		ledgerImpl.WriteAllowed ||
		ledgerImpl.AdmissionAllowed ||
		ledgerImpl.LiveAdmissionEnabled ||
		ledgerImpl.MutatesState ||
		!strings.HasPrefix(ledgerImpl.LedgerImplementationID, "ledger-implementation-") ||
		!ledgerImpl.Passed ||
		!ledgerImpl.LiveReady ||
		!ledgerImpl.ManualEnableRequested ||
		!ledgerImpl.EnableKeyMatched ||
		!ledgerImpl.RequiresWriter ||
		!ledgerImpl.RequiresRollback ||
		ledgerImpl.SourceRollbackImplementationSchema != admissionLiveRouteTurnCandidateAdmissionRollbackImplSchema ||
		!ledgerImpl.SourceRollbackImplementationPassed ||
		ledgerImpl.SourceRollbackImplementationID != rollbackImpl.RollbackImplementationID ||
		ledgerImpl.SourceRollbackImplementationAction != "remove_exact_shadow_candidate_receipt_dry_run" ||
		!ledgerImpl.SourceRollbackImplementationReady ||
		ledgerImpl.SourceRollbackTargetID != writerReceipt.WriterReceiptID ||
		ledgerImpl.SourceWriterReceiptIDForLedger != writerReceipt.WriterReceiptID ||
		ledgerImpl.Reason != "ledger implementation drafted for append-only admission receipts; contracts remain disabled" {
		t.Fatalf("ledger implementation should only draft a non-mutating append-only contract: %+v", ledgerImpl)
	}
	if ledgerImpl.RollbackImplementationID != rollbackImpl.RollbackImplementationID ||
		ledgerImpl.WriterReceiptID != writerReceipt.WriterReceiptID ||
		ledgerImpl.WriterImplementationID != writerImpl.WriterImplementationID ||
		ledgerImpl.AdmissionLedgerID != ledger.AdmissionLedgerID ||
		ledgerImpl.AdmissionWriterContractID != writerContract.WriterContractID ||
		ledgerImpl.AdmissionWriterInventoryID != writerInventory.WriterInventoryID ||
		ledgerImpl.AdmissionWriterPreflightID != writerPreflight.WriterPreflightID ||
		ledgerImpl.AdmissionLiveStageID != liveStage.LiveStageID ||
		ledgerImpl.AdmissionEnableGateID != armedGate.EnableGateID ||
		ledgerImpl.AdmissionSwitchID != sw.SwitchID ||
		ledgerImpl.AdmissionPromotionID != promotion.PromotionID ||
		ledgerImpl.AdmissionDecisionID != decision.DecisionID ||
		ledgerImpl.AdmissionAdapterID != adapter.AdmissionAdapterID ||
		ledgerImpl.CandidateExecutionID != execution.ExecutionID ||
		ledgerImpl.CandidateDraftID != draft.DraftID ||
		ledgerImpl.CandidateRunID != candidate.RunID ||
		ledgerImpl.CandidateTextHash != hashJSON(text) ||
		ledgerImpl.TurnTextHash != obs.TextHash {
		t.Fatalf("ledger implementation lost provenance: ledger=%+v rollback=%+v", ledgerImpl, rollbackImpl)
	}
	ledgerPersistence := admissionLiveRouteTurnCandidateAdmissionLedgerPersistenceForLedgerImplementation(ledgerImpl)
	if ledgerPersistence.Schema != admissionLiveRouteTurnCandidateAdmissionLedgerPersistenceSchema ||
		ledgerPersistence.Timing != "live_admission_ledger_persistence" ||
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
		!ledgerPersistence.WriterReady ||
		!ledgerPersistence.RollbackReady ||
		!ledgerPersistence.WriterImplementationReady ||
		!ledgerPersistence.RollbackImplementationReady ||
		!ledgerPersistence.LedgerImplementationReady ||
		ledgerPersistence.ContractsReady ||
		ledgerPersistence.WriteAllowed ||
		ledgerPersistence.AdmissionAllowed ||
		ledgerPersistence.LiveAdmissionEnabled ||
		ledgerPersistence.MutatesState ||
		!strings.HasPrefix(ledgerPersistence.LedgerPersistenceID, "ledger-persistence-") ||
		!ledgerPersistence.Passed ||
		!ledgerPersistence.LiveReady ||
		ledgerPersistence.SourceLedgerImplementationSchema != admissionLiveRouteTurnCandidateAdmissionLedgerImplSchema ||
		!ledgerPersistence.SourceLedgerImplementationPassed ||
		ledgerPersistence.SourceLedgerImplementationID != ledgerImpl.LedgerImplementationID ||
		ledgerPersistence.SourceLedgerImplementationAction != "append_admission_ledger_receipt_dry_run" ||
		!ledgerPersistence.SourceLedgerImplementationReady ||
		ledgerPersistence.SourceAdmissionLedgerIDForPersistence != ledger.AdmissionLedgerID ||
		ledgerPersistence.SourceRollbackImplementationIDForLedger != rollbackImpl.RollbackImplementationID ||
		ledgerPersistence.SourceWriterReceiptIDForLedgerPersistence != writerReceipt.WriterReceiptID ||
		ledgerPersistence.Reason != "ledger receipt persisted to append-only dry-run log; live admission remains disabled" {
		t.Fatalf("ledger persistence should persist only the append-only dry-run receipt: %+v", ledgerPersistence)
	}
	if ledgerPersistence.LedgerImplementationID != ledgerImpl.LedgerImplementationID ||
		ledgerPersistence.RollbackImplementationID != rollbackImpl.RollbackImplementationID ||
		ledgerPersistence.WriterReceiptID != writerReceipt.WriterReceiptID ||
		ledgerPersistence.WriterImplementationID != writerImpl.WriterImplementationID ||
		ledgerPersistence.AdmissionLedgerID != ledger.AdmissionLedgerID ||
		ledgerPersistence.AdmissionWriterContractID != writerContract.WriterContractID ||
		ledgerPersistence.AdmissionWriterInventoryID != writerInventory.WriterInventoryID ||
		ledgerPersistence.AdmissionWriterPreflightID != writerPreflight.WriterPreflightID ||
		ledgerPersistence.AdmissionLiveStageID != liveStage.LiveStageID ||
		ledgerPersistence.AdmissionEnableGateID != armedGate.EnableGateID ||
		ledgerPersistence.AdmissionSwitchID != sw.SwitchID ||
		ledgerPersistence.AdmissionPromotionID != promotion.PromotionID ||
		ledgerPersistence.AdmissionDecisionID != decision.DecisionID ||
		ledgerPersistence.AdmissionAdapterID != adapter.AdmissionAdapterID ||
		ledgerPersistence.CandidateExecutionID != execution.ExecutionID ||
		ledgerPersistence.CandidateDraftID != draft.DraftID ||
		ledgerPersistence.CandidateRunID != candidate.RunID ||
		ledgerPersistence.CandidateTextHash != hashJSON(text) ||
		ledgerPersistence.TurnTextHash != obs.TextHash {
		t.Fatalf("ledger persistence lost provenance: persistence=%+v ledger=%+v", ledgerPersistence, ledgerImpl)
	}
	ledgerVerification := admissionLiveRouteTurnCandidateAdmissionLedgerVerificationForLedgerPersistence(ledgerPersistence)
	if ledgerVerification.Schema != admissionLiveRouteTurnCandidateAdmissionLedgerVerificationSchema ||
		ledgerVerification.Timing != "live_admission_ledger_verification" ||
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
		!ledgerVerification.LedgerPersistenceReady ||
		!ledgerVerification.WriterReady ||
		!ledgerVerification.RollbackReady ||
		!ledgerVerification.WriterImplementationReady ||
		!ledgerVerification.RollbackImplementationReady ||
		!ledgerVerification.LedgerImplementationReady ||
		ledgerVerification.ContractsReady ||
		ledgerVerification.WriteAllowed ||
		ledgerVerification.AdmissionAllowed ||
		ledgerVerification.LiveAdmissionEnabled ||
		ledgerVerification.MutatesState ||
		!strings.HasPrefix(ledgerVerification.LedgerVerificationID, "ledger-verification-") ||
		!ledgerVerification.Passed ||
		!ledgerVerification.LiveReady ||
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
		ledgerVerification.Reason != "ledger persistence receipt verified by read-back dry-run; live admission remains disabled" {
		t.Fatalf("ledger verification should read back only the persisted append-only receipt: %+v", ledgerVerification)
	}
	if ledgerVerification.LedgerPersistenceID != ledgerPersistence.LedgerPersistenceID ||
		ledgerVerification.LedgerImplementationID != ledgerImpl.LedgerImplementationID ||
		ledgerVerification.RollbackImplementationID != rollbackImpl.RollbackImplementationID ||
		ledgerVerification.WriterReceiptID != writerReceipt.WriterReceiptID ||
		ledgerVerification.WriterImplementationID != writerImpl.WriterImplementationID ||
		ledgerVerification.AdmissionLedgerID != ledger.AdmissionLedgerID ||
		ledgerVerification.AdmissionWriterContractID != writerContract.WriterContractID ||
		ledgerVerification.AdmissionWriterInventoryID != writerInventory.WriterInventoryID ||
		ledgerVerification.AdmissionWriterPreflightID != writerPreflight.WriterPreflightID ||
		ledgerVerification.AdmissionLiveStageID != liveStage.LiveStageID ||
		ledgerVerification.AdmissionEnableGateID != armedGate.EnableGateID ||
		ledgerVerification.AdmissionSwitchID != sw.SwitchID ||
		ledgerVerification.AdmissionPromotionID != promotion.PromotionID ||
		ledgerVerification.AdmissionDecisionID != decision.DecisionID ||
		ledgerVerification.AdmissionAdapterID != adapter.AdmissionAdapterID ||
		ledgerVerification.CandidateExecutionID != execution.ExecutionID ||
		ledgerVerification.CandidateDraftID != draft.DraftID ||
		ledgerVerification.CandidateRunID != candidate.RunID ||
		ledgerVerification.CandidateTextHash != hashJSON(text) ||
		ledgerVerification.TurnTextHash != obs.TextHash {
		t.Fatalf("ledger verification lost provenance: verification=%+v persistence=%+v", ledgerVerification, ledgerPersistence)
	}
	readiness := admissionLiveRouteTurnCandidateAdmissionReadinessForLedgerVerification(ledgerVerification)
	if readiness.Schema != admissionLiveRouteTurnCandidateAdmissionReadinessSchema ||
		readiness.Timing != "live_admission_readiness" ||
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
		!readiness.LedgerVerificationReady ||
		!readiness.LedgerPersistenceReady ||
		!readiness.WriterReady ||
		!readiness.RollbackReady ||
		!readiness.WriterImplementationReady ||
		!readiness.RollbackImplementationReady ||
		!readiness.LedgerImplementationReady ||
		readiness.ContractsReady ||
		readiness.WriteAllowed ||
		readiness.AdmissionAllowed ||
		readiness.LiveAdmissionEnabled ||
		readiness.MutatesState ||
		!strings.HasPrefix(readiness.AdmissionReadinessID, "admission-readiness-") ||
		!readiness.Passed ||
		!readiness.LiveReady ||
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
		readiness.Reason != "verified ledger and writer boundaries are ready; live admission remains disabled" {
		t.Fatalf("readiness should declare only closed verified admission readiness: %+v", readiness)
	}
	if readiness.LedgerVerificationID != ledgerVerification.LedgerVerificationID ||
		readiness.LedgerPersistenceID != ledgerPersistence.LedgerPersistenceID ||
		readiness.LedgerImplementationID != ledgerImpl.LedgerImplementationID ||
		readiness.RollbackImplementationID != rollbackImpl.RollbackImplementationID ||
		readiness.WriterReceiptID != writerReceipt.WriterReceiptID ||
		readiness.WriterImplementationID != writerImpl.WriterImplementationID ||
		readiness.AdmissionLedgerID != ledger.AdmissionLedgerID ||
		readiness.AdmissionWriterContractID != writerContract.WriterContractID ||
		readiness.AdmissionWriterInventoryID != writerInventory.WriterInventoryID ||
		readiness.AdmissionWriterPreflightID != writerPreflight.WriterPreflightID ||
		readiness.AdmissionLiveStageID != liveStage.LiveStageID ||
		readiness.AdmissionEnableGateID != armedGate.EnableGateID ||
		readiness.AdmissionSwitchID != sw.SwitchID ||
		readiness.AdmissionPromotionID != promotion.PromotionID ||
		readiness.AdmissionDecisionID != decision.DecisionID ||
		readiness.AdmissionAdapterID != adapter.AdmissionAdapterID ||
		readiness.CandidateExecutionID != execution.ExecutionID ||
		readiness.CandidateDraftID != draft.DraftID ||
		readiness.CandidateRunID != candidate.RunID ||
		readiness.CandidateTextHash != hashJSON(text) ||
		readiness.TurnTextHash != obs.TextHash {
		t.Fatalf("admission readiness lost provenance: readiness=%+v verification=%+v", readiness, ledgerVerification)
	}
	t.Setenv("AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_PERMIT_KEY", admissionLiveRouteTurnCandidateAdmissionPermitConfirmation)
	permit := admissionLiveRouteTurnCandidateAdmissionPermitForReadiness(readiness)
	if permit.Schema != admissionLiveRouteTurnCandidateAdmissionPermitSchema ||
		permit.Timing != "live_admission_permit" ||
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
		!permit.LedgerVerificationReady ||
		!permit.LedgerPersistenceReady ||
		!permit.WriterReady ||
		!permit.RollbackReady ||
		!permit.WriterImplementationReady ||
		!permit.RollbackImplementationReady ||
		!permit.LedgerImplementationReady ||
		permit.ContractsReady ||
		permit.WriteAllowed ||
		permit.AdmissionAllowed ||
		permit.LiveAdmissionEnabled ||
		permit.MutatesState ||
		!strings.HasPrefix(permit.AdmissionPermitID, "admission-permit-") ||
		!permit.Passed ||
		!permit.LiveReady ||
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
		permit.Reason != "operator permit accepted for verified readiness; live admission remains disabled" {
		t.Fatalf("permit should accept only closed readiness without opening admission: %+v", permit)
	}
	if permit.AdmissionReadinessID != readiness.AdmissionReadinessID ||
		permit.LedgerVerificationID != ledgerVerification.LedgerVerificationID ||
		permit.LedgerPersistenceID != ledgerPersistence.LedgerPersistenceID ||
		permit.LedgerImplementationID != ledgerImpl.LedgerImplementationID ||
		permit.RollbackImplementationID != rollbackImpl.RollbackImplementationID ||
		permit.WriterReceiptID != writerReceipt.WriterReceiptID ||
		permit.WriterImplementationID != writerImpl.WriterImplementationID ||
		permit.AdmissionLedgerID != ledger.AdmissionLedgerID ||
		permit.AdmissionWriterContractID != writerContract.WriterContractID ||
		permit.AdmissionWriterInventoryID != writerInventory.WriterInventoryID ||
		permit.AdmissionWriterPreflightID != writerPreflight.WriterPreflightID ||
		permit.AdmissionLiveStageID != liveStage.LiveStageID ||
		permit.AdmissionEnableGateID != armedGate.EnableGateID ||
		permit.AdmissionSwitchID != sw.SwitchID ||
		permit.AdmissionPromotionID != promotion.PromotionID ||
		permit.AdmissionDecisionID != decision.DecisionID ||
		permit.AdmissionAdapterID != adapter.AdmissionAdapterID ||
		permit.CandidateExecutionID != execution.ExecutionID ||
		permit.CandidateDraftID != draft.DraftID ||
		permit.CandidateRunID != candidate.RunID ||
		permit.CandidateTextHash != hashJSON(text) ||
		permit.TurnTextHash != obs.TextHash {
		t.Fatalf("admission permit lost provenance: permit=%+v readiness=%+v", permit, readiness)
	}
	seal := admissionLiveRouteTurnCandidateAdmissionSealForPermit(permit)
	if seal.Schema != admissionLiveRouteTurnCandidateAdmissionSealSchema ||
		seal.Timing != "live_admission_seal" ||
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
		!seal.AdmissionPermitReady ||
		!seal.PermitKeyMatched ||
		!seal.LedgerVerificationReady ||
		!seal.LedgerPersistenceReady ||
		!seal.WriterReady ||
		!seal.RollbackReady ||
		!seal.WriterImplementationReady ||
		!seal.RollbackImplementationReady ||
		!seal.LedgerImplementationReady ||
		seal.ContractsReady ||
		seal.WriteAllowed ||
		seal.AdmissionAllowed ||
		seal.LiveAdmissionEnabled ||
		seal.MutatesState ||
		!strings.HasPrefix(seal.AdmissionSealID, "admission-seal-") ||
		!seal.Passed ||
		!seal.LiveReady ||
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
		seal.Reason != "operator permit sealed as immutable dry-run receipt; live admission remains disabled" {
		t.Fatalf("seal should freeze only the closed permit without opening admission: %+v", seal)
	}
	if seal.AdmissionPermitID != permit.AdmissionPermitID ||
		seal.AdmissionReadinessID != readiness.AdmissionReadinessID ||
		seal.LedgerVerificationID != ledgerVerification.LedgerVerificationID ||
		seal.LedgerPersistenceID != ledgerPersistence.LedgerPersistenceID ||
		seal.LedgerImplementationID != ledgerImpl.LedgerImplementationID ||
		seal.RollbackImplementationID != rollbackImpl.RollbackImplementationID ||
		seal.WriterReceiptID != writerReceipt.WriterReceiptID ||
		seal.WriterImplementationID != writerImpl.WriterImplementationID ||
		seal.AdmissionLedgerID != ledger.AdmissionLedgerID ||
		seal.AdmissionWriterContractID != writerContract.WriterContractID ||
		seal.AdmissionWriterInventoryID != writerInventory.WriterInventoryID ||
		seal.AdmissionWriterPreflightID != writerPreflight.WriterPreflightID ||
		seal.AdmissionLiveStageID != liveStage.LiveStageID ||
		seal.AdmissionEnableGateID != armedGate.EnableGateID ||
		seal.AdmissionSwitchID != sw.SwitchID ||
		seal.AdmissionPromotionID != promotion.PromotionID ||
		seal.AdmissionDecisionID != decision.DecisionID ||
		seal.AdmissionAdapterID != adapter.AdmissionAdapterID ||
		seal.CandidateExecutionID != execution.ExecutionID ||
		seal.CandidateDraftID != draft.DraftID ||
		seal.CandidateRunID != candidate.RunID ||
		seal.CandidateTextHash != hashJSON(text) ||
		seal.TurnTextHash != obs.TextHash {
		t.Fatalf("admission seal lost provenance: seal=%+v permit=%+v", seal, permit)
	}
	finalGate := admissionLiveRouteTurnCandidateAdmissionFinalGateForSeal(seal)
	if finalGate.Schema != admissionLiveRouteTurnCandidateAdmissionFinalGateSchema ||
		finalGate.Timing != "live_admission_final_gate" ||
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
		!finalGate.AdmissionSealReady ||
		!finalGate.AdmissionPermitReady ||
		!finalGate.PermitKeyMatched ||
		!finalGate.LedgerVerificationReady ||
		!finalGate.LedgerPersistenceReady ||
		!finalGate.WriterReady ||
		!finalGate.RollbackReady ||
		!finalGate.WriterImplementationReady ||
		!finalGate.RollbackImplementationReady ||
		!finalGate.LedgerImplementationReady ||
		finalGate.ContractsReady ||
		finalGate.WriteAllowed ||
		finalGate.AdmissionAllowed ||
		finalGate.LiveAdmissionEnabled ||
		finalGate.MutatesState ||
		finalGate.BodyTarget != "none" ||
		!strings.HasPrefix(finalGate.AdmissionFinalGateID, "admission-final-gate-") ||
		!finalGate.Passed ||
		!finalGate.LiveReady ||
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
		finalGate.Reason != "sealed admission provenance cleared final gate; live admission remains disabled" {
		t.Fatalf("final gate should verify only the sealed permit without opening admission: %+v", finalGate)
	}
	if finalGate.AdmissionSealID != seal.AdmissionSealID ||
		finalGate.AdmissionPermitID != permit.AdmissionPermitID ||
		finalGate.AdmissionReadinessID != readiness.AdmissionReadinessID ||
		finalGate.LedgerVerificationID != ledgerVerification.LedgerVerificationID ||
		finalGate.LedgerPersistenceID != ledgerPersistence.LedgerPersistenceID ||
		finalGate.LedgerImplementationID != ledgerImpl.LedgerImplementationID ||
		finalGate.RollbackImplementationID != rollbackImpl.RollbackImplementationID ||
		finalGate.WriterReceiptID != writerReceipt.WriterReceiptID ||
		finalGate.WriterImplementationID != writerImpl.WriterImplementationID ||
		finalGate.AdmissionLedgerID != ledger.AdmissionLedgerID ||
		finalGate.AdmissionWriterContractID != writerContract.WriterContractID ||
		finalGate.AdmissionWriterInventoryID != writerInventory.WriterInventoryID ||
		finalGate.AdmissionWriterPreflightID != writerPreflight.WriterPreflightID ||
		finalGate.AdmissionLiveStageID != liveStage.LiveStageID ||
		finalGate.AdmissionEnableGateID != armedGate.EnableGateID ||
		finalGate.AdmissionSwitchID != sw.SwitchID ||
		finalGate.AdmissionPromotionID != promotion.PromotionID ||
		finalGate.AdmissionDecisionID != decision.DecisionID ||
		finalGate.AdmissionAdapterID != adapter.AdmissionAdapterID ||
		finalGate.CandidateExecutionID != execution.ExecutionID ||
		finalGate.CandidateDraftID != draft.DraftID ||
		finalGate.CandidateRunID != candidate.RunID ||
		finalGate.CandidateTextHash != hashJSON(text) ||
		finalGate.TurnTextHash != obs.TextHash {
		t.Fatalf("admission final gate lost provenance: final_gate=%+v seal=%+v", finalGate, seal)
	}
	assertRouteBoundary(
		"admission final gate",
		finalGate.BodyInventoryStatus,
		finalGate.RouteAvailabilityStatus,
		finalGate.RouteAvailabilityReason,
		finalGate.RouteMissingOrgans,
	)
	resonanceIntent := admissionLiveRouteTurnCandidateAdmissionResonanceIntentForFinalGate(finalGate)
	if resonanceIntent.Schema != admissionLiveRouteTurnCandidateAdmissionResonanceIntentSchema ||
		resonanceIntent.Timing != "live_admission_resonance_intent" ||
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
		!strings.HasPrefix(resonanceIntent.AdmissionResonanceIntentCausalID, "resonance-intent-causal-") ||
		resonanceIntent.AdmissionResonanceIntentRawDreamTextAllowed ||
		resonanceIntent.AdmissionResonanceIntentJanusSurfaceAllowed ||
		resonanceIntent.AdmissionResonanceIntentCoocLearningAllowed ||
		resonanceIntent.AdmissionResonanceIntentDeltaHarvestAllowed ||
		!resonanceIntent.AdmissionResonanceIntentRollbackRequired ||
		!resonanceIntent.AdmissionResonanceIntentPreStateHashRequired ||
		!resonanceIntent.AdmissionResonanceIntentPostStateHashRequired ||
		!resonanceIntent.AdmissionResonanceIntentReady ||
		resonanceIntent.ContractsReady ||
		resonanceIntent.WriteAllowed ||
		resonanceIntent.AdmissionAllowed ||
		resonanceIntent.LiveAdmissionEnabled ||
		resonanceIntent.MutatesState ||
		resonanceIntent.BodyTarget != "none" ||
		!strings.HasPrefix(resonanceIntent.AdmissionResonanceIntentID, "resonance-intent-") ||
		!resonanceIntent.Passed ||
		!resonanceIntent.LiveReady ||
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
		resonanceIntent.Reason != "resonance intent drafted from final gate; live admission remains disabled" {
		t.Fatalf("resonance intent should hear only sealed final-gate provenance: %+v", resonanceIntent)
	}
	if resonanceIntent.AdmissionFinalGateID != finalGate.AdmissionFinalGateID ||
		resonanceIntent.AdmissionSealID != seal.AdmissionSealID ||
		resonanceIntent.AdmissionPermitID != permit.AdmissionPermitID ||
		resonanceIntent.AdmissionReadinessID != readiness.AdmissionReadinessID ||
		resonanceIntent.LedgerVerificationID != ledgerVerification.LedgerVerificationID ||
		resonanceIntent.LedgerPersistenceID != ledgerPersistence.LedgerPersistenceID ||
		resonanceIntent.LedgerImplementationID != ledgerImpl.LedgerImplementationID ||
		resonanceIntent.RollbackImplementationID != rollbackImpl.RollbackImplementationID ||
		resonanceIntent.WriterReceiptID != writerReceipt.WriterReceiptID ||
		resonanceIntent.WriterImplementationID != writerImpl.WriterImplementationID ||
		resonanceIntent.AdmissionLedgerID != ledger.AdmissionLedgerID ||
		resonanceIntent.AdmissionWriterContractID != writerContract.WriterContractID ||
		resonanceIntent.AdmissionWriterInventoryID != writerInventory.WriterInventoryID ||
		resonanceIntent.AdmissionWriterPreflightID != writerPreflight.WriterPreflightID ||
		resonanceIntent.AdmissionLiveStageID != liveStage.LiveStageID ||
		resonanceIntent.AdmissionEnableGateID != armedGate.EnableGateID ||
		resonanceIntent.AdmissionSwitchID != sw.SwitchID ||
		resonanceIntent.AdmissionPromotionID != promotion.PromotionID ||
		resonanceIntent.AdmissionDecisionID != decision.DecisionID ||
		resonanceIntent.AdmissionAdapterID != adapter.AdmissionAdapterID ||
		resonanceIntent.CandidateExecutionID != execution.ExecutionID ||
		resonanceIntent.CandidateDraftID != draft.DraftID ||
		resonanceIntent.CandidateRunID != candidate.RunID ||
		resonanceIntent.CandidateTextHash != hashJSON(text) ||
		resonanceIntent.TurnTextHash != obs.TextHash {
		t.Fatalf("admission resonance intent lost provenance: intent=%+v final_gate=%+v", resonanceIntent, finalGate)
	}
	resonanceReceiver := admissionLiveRouteTurnCandidateAdmissionResonanceReceiverForIntent(resonanceIntent)
	if resonanceReceiver.Schema != admissionLiveRouteTurnCandidateAdmissionResonanceReceiverSchema ||
		resonanceReceiver.Timing != "live_admission_resonance_receiver" ||
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
		!strings.HasPrefix(resonanceReceiver.AdmissionResonanceReceiverCausalID, "resonance-receiver-causal-") ||
		resonanceReceiver.AdmissionResonanceReceiverCausalID != admissionLiveRouteTurnCandidateAdmissionResonanceReceiverCausalID(resonanceReceiver) ||
		!strings.HasPrefix(resonanceReceiver.AdmissionResonanceReceiverPreStateHash, "resonance-receiver-pre-") ||
		!strings.HasPrefix(resonanceReceiver.AdmissionResonanceReceiverPostStateHash, "resonance-receiver-post-") ||
		!strings.HasPrefix(resonanceReceiver.AdmissionResonanceReceiverStateDeltaHash, "resonance-receiver-delta-") ||
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
		resonanceReceiver.ContractsReady ||
		resonanceReceiver.WriteAllowed ||
		resonanceReceiver.AdmissionAllowed ||
		resonanceReceiver.LiveAdmissionEnabled ||
		resonanceReceiver.MutatesState ||
		resonanceReceiver.BodyTarget != "none" ||
		!strings.HasPrefix(resonanceReceiver.AdmissionResonanceReceiverID, "resonance-receiver-") ||
		resonanceReceiver.AdmissionResonanceReceiverID != admissionLiveRouteTurnCandidateAdmissionResonanceReceiverID(resonanceReceiver) ||
		!resonanceReceiver.Passed ||
		!resonanceReceiver.LiveReady ||
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
		resonanceReceiver.Reason != "resonance receiver previewed sealed intent without body mutation" {
		t.Fatalf("resonance receiver should preview only sealed intent provenance: %+v", resonanceReceiver)
	}
	if resonanceReceiver.AdmissionResonanceIntentID != resonanceIntent.AdmissionResonanceIntentID ||
		resonanceReceiver.AdmissionFinalGateID != finalGate.AdmissionFinalGateID ||
		resonanceReceiver.AdmissionSealID != seal.AdmissionSealID ||
		resonanceReceiver.AdmissionPermitID != permit.AdmissionPermitID ||
		resonanceReceiver.AdmissionReadinessID != readiness.AdmissionReadinessID ||
		resonanceReceiver.LedgerVerificationID != ledgerVerification.LedgerVerificationID ||
		resonanceReceiver.LedgerPersistenceID != ledgerPersistence.LedgerPersistenceID ||
		resonanceReceiver.LedgerImplementationID != ledgerImpl.LedgerImplementationID ||
		resonanceReceiver.RollbackImplementationID != rollbackImpl.RollbackImplementationID ||
		resonanceReceiver.WriterReceiptID != writerReceipt.WriterReceiptID ||
		resonanceReceiver.WriterImplementationID != writerImpl.WriterImplementationID ||
		resonanceReceiver.AdmissionLedgerID != ledger.AdmissionLedgerID ||
		resonanceReceiver.AdmissionWriterContractID != writerContract.WriterContractID ||
		resonanceReceiver.AdmissionWriterInventoryID != writerInventory.WriterInventoryID ||
		resonanceReceiver.AdmissionWriterPreflightID != writerPreflight.WriterPreflightID ||
		resonanceReceiver.AdmissionLiveStageID != liveStage.LiveStageID ||
		resonanceReceiver.AdmissionEnableGateID != armedGate.EnableGateID ||
		resonanceReceiver.AdmissionSwitchID != sw.SwitchID ||
		resonanceReceiver.AdmissionPromotionID != promotion.PromotionID ||
		resonanceReceiver.AdmissionDecisionID != decision.DecisionID ||
		resonanceReceiver.AdmissionAdapterID != adapter.AdmissionAdapterID ||
		resonanceReceiver.CandidateExecutionID != execution.ExecutionID ||
		resonanceReceiver.CandidateDraftID != draft.DraftID ||
		resonanceReceiver.CandidateRunID != candidate.RunID ||
		resonanceReceiver.CandidateTextHash != hashJSON(text) ||
		resonanceReceiver.TurnTextHash != obs.TextHash {
		t.Fatalf("admission resonance receiver lost provenance: receiver=%+v intent=%+v", resonanceReceiver, resonanceIntent)
	}
	resonanceObservation := admissionLiveRouteTurnCandidateAdmissionResonanceObservationForReceiver(resonanceReceiver)
	if resonanceObservation.Schema != admissionLiveRouteTurnCandidateAdmissionResonanceObservationSchema ||
		resonanceObservation.Timing != "live_admission_resonance_observation" ||
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
		!strings.HasPrefix(resonanceObservation.AdmissionResonanceObservationCausalID, "resonance-observation-causal-") ||
		resonanceObservation.AdmissionResonanceObservationCausalID != admissionLiveRouteTurnCandidateAdmissionResonanceObservationCausalID(resonanceObservation) ||
		!strings.HasPrefix(resonanceObservation.AdmissionResonanceObservationAppendHash, "resonance-observation-append-") ||
		resonanceObservation.AdmissionResonanceObservationAppendHash != admissionLiveRouteTurnCandidateAdmissionResonanceObservationAppendHash(resonanceObservation) ||
		!strings.HasPrefix(resonanceObservation.AdmissionResonanceObservationReadBackHash, "resonance-observation-read-") ||
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
		resonanceObservation.ContractsReady ||
		resonanceObservation.WriteAllowed ||
		resonanceObservation.AdmissionAllowed ||
		resonanceObservation.LiveAdmissionEnabled ||
		resonanceObservation.MutatesState ||
		resonanceObservation.BodyTarget != "none" ||
		!strings.HasPrefix(resonanceObservation.AdmissionResonanceObservationID, "resonance-observation-") ||
		resonanceObservation.AdmissionResonanceObservationID != admissionLiveRouteTurnCandidateAdmissionResonanceObservationID(resonanceObservation) ||
		!resonanceObservation.Passed ||
		!resonanceObservation.LiveReady ||
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
		resonanceObservation.Reason != "resonance observation recorded and read back without body mutation" {
		t.Fatalf("resonance observation should record only sealed receiver provenance: %+v", resonanceObservation)
	}
	if resonanceObservation.AdmissionResonanceReceiverID != resonanceReceiver.AdmissionResonanceReceiverID ||
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
		resonanceObservation.WriterImplementationID != writerImpl.WriterImplementationID ||
		resonanceObservation.AdmissionLedgerID != ledger.AdmissionLedgerID ||
		resonanceObservation.AdmissionWriterContractID != writerContract.WriterContractID ||
		resonanceObservation.AdmissionWriterInventoryID != writerInventory.WriterInventoryID ||
		resonanceObservation.AdmissionWriterPreflightID != writerPreflight.WriterPreflightID ||
		resonanceObservation.AdmissionLiveStageID != liveStage.LiveStageID ||
		resonanceObservation.AdmissionEnableGateID != armedGate.EnableGateID ||
		resonanceObservation.AdmissionSwitchID != sw.SwitchID ||
		resonanceObservation.AdmissionPromotionID != promotion.PromotionID ||
		resonanceObservation.AdmissionDecisionID != decision.DecisionID ||
		resonanceObservation.AdmissionAdapterID != adapter.AdmissionAdapterID ||
		resonanceObservation.CandidateExecutionID != execution.ExecutionID ||
		resonanceObservation.CandidateDraftID != draft.DraftID ||
		resonanceObservation.CandidateRunID != candidate.RunID ||
		resonanceObservation.CandidateTextHash != hashJSON(text) ||
		resonanceObservation.TurnTextHash != obs.TextHash {
		t.Fatalf("admission resonance observation lost provenance: observation=%+v receiver=%+v", resonanceObservation, resonanceReceiver)
	}
	resonanceGraftBoundary := admissionLiveRouteTurnCandidateAdmissionResonanceGraftBoundaryForObservation(resonanceObservation)
	if resonanceGraftBoundary.Schema != admissionLiveRouteTurnCandidateAdmissionResonanceGraftBoundarySchema ||
		resonanceGraftBoundary.Timing != "live_admission_resonance_graft_boundary" ||
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
		!strings.HasPrefix(resonanceGraftBoundary.AdmissionResonanceGraftBoundaryCausalID, "resonance-graft-boundary-causal-") ||
		resonanceGraftBoundary.AdmissionResonanceGraftBoundaryCausalID != admissionLiveRouteTurnCandidateAdmissionResonanceGraftBoundaryCausalID(resonanceGraftBoundary) ||
		!strings.HasPrefix(resonanceGraftBoundary.AdmissionResonanceGraftBoundaryHash, "resonance-graft-boundary-") ||
		resonanceGraftBoundary.AdmissionResonanceGraftBoundaryHash != admissionLiveRouteTurnCandidateAdmissionResonanceGraftBoundaryHash(resonanceGraftBoundary) ||
		!strings.HasPrefix(resonanceGraftBoundary.AdmissionResonanceGraftBoundaryReadBackHash, "resonance-graft-boundary-read-") ||
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
		resonanceGraftBoundary.ContractsReady ||
		resonanceGraftBoundary.WriteAllowed ||
		resonanceGraftBoundary.AdmissionAllowed ||
		resonanceGraftBoundary.LiveAdmissionEnabled ||
		resonanceGraftBoundary.MutatesState ||
		resonanceGraftBoundary.BodyTarget != "none" ||
		!strings.HasPrefix(resonanceGraftBoundary.AdmissionResonanceGraftBoundaryID, "resonance-graft-boundary-id-") ||
		resonanceGraftBoundary.AdmissionResonanceGraftBoundaryID != admissionLiveRouteTurnCandidateAdmissionResonanceGraftBoundaryID(resonanceGraftBoundary) ||
		!resonanceGraftBoundary.Passed ||
		!resonanceGraftBoundary.LiveReady ||
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
		resonanceGraftBoundary.Reason != "resonance shadow graft boundary declared without body mutation" {
		t.Fatalf("resonance graft boundary should declare only a sealed shadow boundary: %+v", resonanceGraftBoundary)
	}
	if resonanceGraftBoundary.AdmissionResonanceObservationID != resonanceObservation.AdmissionResonanceObservationID ||
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
		resonanceGraftBoundary.WriterImplementationID != writerImpl.WriterImplementationID ||
		resonanceGraftBoundary.AdmissionLedgerID != ledger.AdmissionLedgerID ||
		resonanceGraftBoundary.AdmissionWriterContractID != writerContract.WriterContractID ||
		resonanceGraftBoundary.AdmissionWriterInventoryID != writerInventory.WriterInventoryID ||
		resonanceGraftBoundary.AdmissionWriterPreflightID != writerPreflight.WriterPreflightID ||
		resonanceGraftBoundary.AdmissionLiveStageID != liveStage.LiveStageID ||
		resonanceGraftBoundary.AdmissionEnableGateID != armedGate.EnableGateID ||
		resonanceGraftBoundary.AdmissionSwitchID != sw.SwitchID ||
		resonanceGraftBoundary.AdmissionPromotionID != promotion.PromotionID ||
		resonanceGraftBoundary.AdmissionDecisionID != decision.DecisionID ||
		resonanceGraftBoundary.AdmissionAdapterID != adapter.AdmissionAdapterID ||
		resonanceGraftBoundary.CandidateExecutionID != execution.ExecutionID ||
		resonanceGraftBoundary.CandidateDraftID != draft.DraftID ||
		resonanceGraftBoundary.CandidateRunID != candidate.RunID ||
		resonanceGraftBoundary.CandidateTextHash != hashJSON(text) ||
		resonanceGraftBoundary.TurnTextHash != obs.TextHash {
		t.Fatalf("admission resonance graft boundary lost provenance: boundary=%+v observation=%+v", resonanceGraftBoundary, resonanceObservation)
	}
	resonanceGraftPreflight := admissionLiveRouteTurnCandidateAdmissionResonanceGraftPreflightForBoundary(resonanceGraftBoundary)
	if resonanceGraftPreflight.Schema != admissionLiveRouteTurnCandidateAdmissionResonanceGraftPreflightSchema ||
		resonanceGraftPreflight.Timing != "live_admission_resonance_graft_preflight" ||
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
		!strings.HasPrefix(resonanceGraftPreflight.AdmissionResonanceGraftPreflightCausalID, "resonance-graft-preflight-causal-") ||
		resonanceGraftPreflight.AdmissionResonanceGraftPreflightCausalID != admissionLiveRouteTurnCandidateAdmissionResonanceGraftPreflightCausalID(resonanceGraftPreflight) ||
		!strings.HasPrefix(resonanceGraftPreflight.AdmissionResonanceGraftPreflightHash, "resonance-graft-preflight-") ||
		resonanceGraftPreflight.AdmissionResonanceGraftPreflightHash != admissionLiveRouteTurnCandidateAdmissionResonanceGraftPreflightHash(resonanceGraftPreflight) ||
		!strings.HasPrefix(resonanceGraftPreflight.AdmissionResonanceGraftPreflightReadBackHash, "resonance-graft-preflight-read-") ||
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
		resonanceGraftPreflight.ContractsReady ||
		resonanceGraftPreflight.WriteAllowed ||
		resonanceGraftPreflight.AdmissionAllowed ||
		resonanceGraftPreflight.LiveAdmissionEnabled ||
		resonanceGraftPreflight.MutatesState ||
		resonanceGraftPreflight.BodyTarget != "none" ||
		!strings.HasPrefix(resonanceGraftPreflight.AdmissionResonanceGraftPreflightID, "resonance-graft-preflight-id-") ||
		resonanceGraftPreflight.AdmissionResonanceGraftPreflightID != admissionLiveRouteTurnCandidateAdmissionResonanceGraftPreflightID(resonanceGraftPreflight) ||
		!resonanceGraftPreflight.Passed ||
		!resonanceGraftPreflight.LiveReady ||
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
		resonanceGraftPreflight.Reason != "resonance shadow graft preflight prepared without body mutation" {
		t.Fatalf("resonance graft preflight should prepare only a closed shadow graft admission contract: %+v", resonanceGraftPreflight)
	}
	if resonanceGraftPreflight.AdmissionResonanceGraftBoundaryID != resonanceGraftBoundary.AdmissionResonanceGraftBoundaryID ||
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
		resonanceGraftPreflight.WriterImplementationID != writerImpl.WriterImplementationID ||
		resonanceGraftPreflight.AdmissionLedgerID != ledger.AdmissionLedgerID ||
		resonanceGraftPreflight.AdmissionWriterContractID != writerContract.WriterContractID ||
		resonanceGraftPreflight.AdmissionWriterInventoryID != writerInventory.WriterInventoryID ||
		resonanceGraftPreflight.AdmissionWriterPreflightID != writerPreflight.WriterPreflightID ||
		resonanceGraftPreflight.AdmissionLiveStageID != liveStage.LiveStageID ||
		resonanceGraftPreflight.AdmissionEnableGateID != armedGate.EnableGateID ||
		resonanceGraftPreflight.AdmissionSwitchID != sw.SwitchID ||
		resonanceGraftPreflight.AdmissionPromotionID != promotion.PromotionID ||
		resonanceGraftPreflight.AdmissionDecisionID != decision.DecisionID ||
		resonanceGraftPreflight.AdmissionAdapterID != adapter.AdmissionAdapterID ||
		resonanceGraftPreflight.CandidateExecutionID != execution.ExecutionID ||
		resonanceGraftPreflight.CandidateDraftID != draft.DraftID ||
		resonanceGraftPreflight.CandidateRunID != candidate.RunID ||
		resonanceGraftPreflight.CandidateTextHash != hashJSON(text) ||
		resonanceGraftPreflight.TurnTextHash != obs.TextHash {
		t.Fatalf("admission resonance graft preflight lost provenance: preflight=%+v boundary=%+v", resonanceGraftPreflight, resonanceGraftBoundary)
	}
	resonanceGraftGate := admissionLiveRouteTurnCandidateAdmissionResonanceGraftGateForPreflight(resonanceGraftPreflight)
	if resonanceGraftGate.Schema != admissionLiveRouteTurnCandidateAdmissionResonanceGraftGateSchema ||
		resonanceGraftGate.Timing != "live_admission_resonance_graft_gate" ||
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
		!strings.HasPrefix(resonanceGraftGate.AdmissionResonanceGraftGateCausalID, "resonance-graft-gate-causal-") ||
		resonanceGraftGate.AdmissionResonanceGraftGateCausalID != admissionLiveRouteTurnCandidateAdmissionResonanceGraftGateCausalID(resonanceGraftGate) ||
		!strings.HasPrefix(resonanceGraftGate.AdmissionResonanceGraftGateHash, "resonance-graft-gate-") ||
		resonanceGraftGate.AdmissionResonanceGraftGateHash != admissionLiveRouteTurnCandidateAdmissionResonanceGraftGateHash(resonanceGraftGate) ||
		!strings.HasPrefix(resonanceGraftGate.AdmissionResonanceGraftGateReadBackHash, "resonance-graft-gate-read-") ||
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
		resonanceGraftGate.ContractsReady ||
		resonanceGraftGate.WriteAllowed ||
		resonanceGraftGate.AdmissionAllowed ||
		resonanceGraftGate.LiveAdmissionEnabled ||
		resonanceGraftGate.MutatesState ||
		resonanceGraftGate.BodyTarget != "none" ||
		!strings.HasPrefix(resonanceGraftGate.AdmissionResonanceGraftGateID, "resonance-graft-gate-id-") ||
		resonanceGraftGate.AdmissionResonanceGraftGateID != admissionLiveRouteTurnCandidateAdmissionResonanceGraftGateID(resonanceGraftGate) ||
		!resonanceGraftGate.Passed ||
		!resonanceGraftGate.LiveReady ||
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
		resonanceGraftGate.Reason != "resonance shadow graft gate prepared without body mutation" {
		t.Fatalf("resonance graft gate should prepare only a closed shadow graft gate contract: %+v", resonanceGraftGate)
	}
	if resonanceGraftGate.AdmissionResonanceGraftPreflightID != resonanceGraftPreflight.AdmissionResonanceGraftPreflightID ||
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
		resonanceGraftGate.WriterImplementationID != writerImpl.WriterImplementationID ||
		resonanceGraftGate.AdmissionLedgerID != ledger.AdmissionLedgerID ||
		resonanceGraftGate.AdmissionWriterContractID != writerContract.WriterContractID ||
		resonanceGraftGate.AdmissionWriterInventoryID != writerInventory.WriterInventoryID ||
		resonanceGraftGate.AdmissionWriterPreflightID != writerPreflight.WriterPreflightID ||
		resonanceGraftGate.AdmissionLiveStageID != liveStage.LiveStageID ||
		resonanceGraftGate.AdmissionEnableGateID != armedGate.EnableGateID ||
		resonanceGraftGate.AdmissionSwitchID != sw.SwitchID ||
		resonanceGraftGate.AdmissionPromotionID != promotion.PromotionID ||
		resonanceGraftGate.AdmissionDecisionID != decision.DecisionID ||
		resonanceGraftGate.AdmissionAdapterID != adapter.AdmissionAdapterID ||
		resonanceGraftGate.CandidateExecutionID != execution.ExecutionID ||
		resonanceGraftGate.CandidateDraftID != draft.DraftID ||
		resonanceGraftGate.CandidateRunID != candidate.RunID ||
		resonanceGraftGate.CandidateTextHash != hashJSON(text) ||
		resonanceGraftGate.TurnTextHash != obs.TextHash {
		t.Fatalf("admission resonance graft gate lost provenance: gate=%+v preflight=%+v", resonanceGraftGate, resonanceGraftPreflight)
	}
	resonanceGraftCandidate := admissionLiveRouteTurnCandidateAdmissionResonanceGraftCandidateForGate(resonanceGraftGate)
	if resonanceGraftCandidate.Schema != admissionLiveRouteTurnCandidateAdmissionResonanceGraftCandidateSchema ||
		resonanceGraftCandidate.Timing != "live_admission_resonance_graft_candidate" ||
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
		!strings.HasPrefix(resonanceGraftCandidate.AdmissionResonanceGraftCandidateCausalID, "resonance-graft-candidate-causal-") ||
		resonanceGraftCandidate.AdmissionResonanceGraftCandidateCausalID != admissionLiveRouteTurnCandidateAdmissionResonanceGraftCandidateCausalID(resonanceGraftCandidate) ||
		!strings.HasPrefix(resonanceGraftCandidate.AdmissionResonanceGraftCandidateHash, "resonance-graft-candidate-") ||
		resonanceGraftCandidate.AdmissionResonanceGraftCandidateHash != admissionLiveRouteTurnCandidateAdmissionResonanceGraftCandidateHash(resonanceGraftCandidate) ||
		!strings.HasPrefix(resonanceGraftCandidate.AdmissionResonanceGraftCandidateReadBackHash, "resonance-graft-candidate-read-") ||
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
		resonanceGraftCandidate.ContractsReady ||
		resonanceGraftCandidate.WriteAllowed ||
		resonanceGraftCandidate.AdmissionAllowed ||
		resonanceGraftCandidate.LiveAdmissionEnabled ||
		resonanceGraftCandidate.MutatesState ||
		resonanceGraftCandidate.BodyTarget != "none" ||
		!strings.HasPrefix(resonanceGraftCandidate.AdmissionResonanceGraftCandidateID, "resonance-graft-candidate-id-") ||
		resonanceGraftCandidate.AdmissionResonanceGraftCandidateID != admissionLiveRouteTurnCandidateAdmissionResonanceGraftCandidateID(resonanceGraftCandidate) ||
		!resonanceGraftCandidate.Passed ||
		!resonanceGraftCandidate.LiveReady ||
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
		resonanceGraftCandidate.Reason != "resonance shadow graft candidate drafted without body mutation" {
		t.Fatalf("resonance graft candidate should draft only a closed shadow graft candidate contract: %+v", resonanceGraftCandidate)
	}
	if resonanceGraftCandidate.AdmissionResonanceGraftGateID != resonanceGraftGate.AdmissionResonanceGraftGateID ||
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
		resonanceGraftCandidate.WriterImplementationID != writerImpl.WriterImplementationID ||
		resonanceGraftCandidate.AdmissionLedgerID != ledger.AdmissionLedgerID ||
		resonanceGraftCandidate.AdmissionWriterContractID != writerContract.WriterContractID ||
		resonanceGraftCandidate.AdmissionWriterInventoryID != writerInventory.WriterInventoryID ||
		resonanceGraftCandidate.AdmissionWriterPreflightID != writerPreflight.WriterPreflightID ||
		resonanceGraftCandidate.AdmissionLiveStageID != liveStage.LiveStageID ||
		resonanceGraftCandidate.AdmissionEnableGateID != armedGate.EnableGateID ||
		resonanceGraftCandidate.AdmissionSwitchID != sw.SwitchID ||
		resonanceGraftCandidate.AdmissionPromotionID != promotion.PromotionID ||
		resonanceGraftCandidate.AdmissionDecisionID != decision.DecisionID ||
		resonanceGraftCandidate.AdmissionAdapterID != adapter.AdmissionAdapterID ||
		resonanceGraftCandidate.CandidateExecutionID != execution.ExecutionID ||
		resonanceGraftCandidate.CandidateDraftID != draft.DraftID ||
		resonanceGraftCandidate.CandidateRunID != candidate.RunID ||
		resonanceGraftCandidate.CandidateTextHash != hashJSON(text) ||
		resonanceGraftCandidate.TurnTextHash != obs.TextHash {
		t.Fatalf("admission resonance graft candidate lost provenance: candidate=%+v gate=%+v", resonanceGraftCandidate, resonanceGraftGate)
	}
	resonanceGraftCandidateStore := admissionLiveRouteTurnCandidateAdmissionResonanceGraftCandidateStoreForCandidate(resonanceGraftCandidate)
	if resonanceGraftCandidateStore.Schema != admissionLiveRouteTurnCandidateAdmissionResonanceGraftCandidateStoreSchema ||
		resonanceGraftCandidateStore.Timing != "live_admission_resonance_graft_candidate_store" ||
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
		!strings.HasPrefix(resonanceGraftCandidateStore.AdmissionResonanceGraftCandidateStoreCausalID, "resonance-graft-candidate-store-causal-") ||
		resonanceGraftCandidateStore.AdmissionResonanceGraftCandidateStoreCausalID != admissionLiveRouteTurnCandidateAdmissionResonanceGraftCandidateStoreCausalID(resonanceGraftCandidateStore) ||
		!strings.HasPrefix(resonanceGraftCandidateStore.AdmissionResonanceGraftCandidateStoreHash, "resonance-graft-candidate-store-") ||
		resonanceGraftCandidateStore.AdmissionResonanceGraftCandidateStoreHash != admissionLiveRouteTurnCandidateAdmissionResonanceGraftCandidateStoreHash(resonanceGraftCandidateStore) ||
		!strings.HasPrefix(resonanceGraftCandidateStore.AdmissionResonanceGraftCandidateStoreReadBackHash, "resonance-graft-candidate-store-read-") ||
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
		resonanceGraftCandidateStore.ContractsReady ||
		resonanceGraftCandidateStore.WriteAllowed ||
		resonanceGraftCandidateStore.AdmissionAllowed ||
		resonanceGraftCandidateStore.LiveAdmissionEnabled ||
		resonanceGraftCandidateStore.MutatesState ||
		resonanceGraftCandidateStore.BodyTarget != "none" ||
		!strings.HasPrefix(resonanceGraftCandidateStore.AdmissionResonanceGraftCandidateStoreID, "resonance-graft-candidate-store-id-") ||
		resonanceGraftCandidateStore.AdmissionResonanceGraftCandidateStoreID != admissionLiveRouteTurnCandidateAdmissionResonanceGraftCandidateStoreID(resonanceGraftCandidateStore) ||
		!resonanceGraftCandidateStore.Passed ||
		!resonanceGraftCandidateStore.LiveReady ||
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
		resonanceGraftCandidateStore.Reason != "resonance shadow graft candidate stored and read back without body mutation" {
		t.Fatalf("resonance graft candidate store should record only a closed append-only store receipt: %+v", resonanceGraftCandidateStore)
	}
	if resonanceGraftCandidateStore.AdmissionResonanceGraftCandidateID != resonanceGraftCandidate.AdmissionResonanceGraftCandidateID ||
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
		resonanceGraftCandidateStore.WriterImplementationID != writerImpl.WriterImplementationID ||
		resonanceGraftCandidateStore.AdmissionLedgerID != ledger.AdmissionLedgerID ||
		resonanceGraftCandidateStore.AdmissionWriterContractID != writerContract.WriterContractID ||
		resonanceGraftCandidateStore.AdmissionWriterInventoryID != writerInventory.WriterInventoryID ||
		resonanceGraftCandidateStore.AdmissionWriterPreflightID != writerPreflight.WriterPreflightID ||
		resonanceGraftCandidateStore.AdmissionLiveStageID != liveStage.LiveStageID ||
		resonanceGraftCandidateStore.AdmissionEnableGateID != armedGate.EnableGateID ||
		resonanceGraftCandidateStore.AdmissionSwitchID != sw.SwitchID ||
		resonanceGraftCandidateStore.AdmissionPromotionID != promotion.PromotionID ||
		resonanceGraftCandidateStore.AdmissionDecisionID != decision.DecisionID ||
		resonanceGraftCandidateStore.AdmissionAdapterID != adapter.AdmissionAdapterID ||
		resonanceGraftCandidateStore.CandidateExecutionID != execution.ExecutionID ||
		resonanceGraftCandidateStore.CandidateDraftID != draft.DraftID ||
		resonanceGraftCandidateStore.CandidateRunID != candidate.RunID ||
		resonanceGraftCandidateStore.CandidateTextHash != hashJSON(text) ||
		resonanceGraftCandidateStore.TurnTextHash != obs.TextHash {
		t.Fatalf("admission resonance graft candidate store lost provenance: store=%+v candidate=%+v", resonanceGraftCandidateStore, resonanceGraftCandidate)
	}
	resonanceGraftCandidateStoreReader := admissionLiveRouteTurnCandidateAdmissionResonanceGraftCandidateStoreReaderForStore(resonanceGraftCandidateStore)
	if resonanceGraftCandidateStoreReader.Schema != admissionLiveRouteTurnCandidateAdmissionResonanceGraftCandidateStoreReaderSchema ||
		resonanceGraftCandidateStoreReader.Timing != "live_admission_resonance_graft_candidate_store_reader" ||
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
		!strings.HasPrefix(resonanceGraftCandidateStoreReader.AdmissionResonanceGraftCandidateStoreReaderCausalID, "resonance-graft-candidate-store-reader-causal-") ||
		resonanceGraftCandidateStoreReader.AdmissionResonanceGraftCandidateStoreReaderCausalID != admissionLiveRouteTurnCandidateAdmissionResonanceGraftCandidateStoreReaderCausalID(resonanceGraftCandidateStoreReader) ||
		!strings.HasPrefix(resonanceGraftCandidateStoreReader.AdmissionResonanceGraftCandidateStoreReaderHash, "resonance-graft-candidate-store-reader-") ||
		resonanceGraftCandidateStoreReader.AdmissionResonanceGraftCandidateStoreReaderHash != admissionLiveRouteTurnCandidateAdmissionResonanceGraftCandidateStoreReaderHash(resonanceGraftCandidateStoreReader) ||
		!strings.HasPrefix(resonanceGraftCandidateStoreReader.AdmissionResonanceGraftCandidateStoreReaderReplayHash, "resonance-graft-candidate-store-reader-replay-") ||
		resonanceGraftCandidateStoreReader.AdmissionResonanceGraftCandidateStoreReaderReplayHash != admissionLiveRouteTurnCandidateAdmissionResonanceGraftCandidateStoreReaderReplayHash(resonanceGraftCandidateStoreReader) ||
		!strings.HasPrefix(resonanceGraftCandidateStoreReader.AdmissionResonanceGraftCandidateStoreReaderReadBackHash, "resonance-graft-candidate-store-reader-read-") ||
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
		resonanceGraftCandidateStoreReader.ContractsReady ||
		resonanceGraftCandidateStoreReader.WriteAllowed ||
		resonanceGraftCandidateStoreReader.AdmissionAllowed ||
		resonanceGraftCandidateStoreReader.LiveAdmissionEnabled ||
		resonanceGraftCandidateStoreReader.MutatesState ||
		resonanceGraftCandidateStoreReader.BodyTarget != "none" ||
		!strings.HasPrefix(resonanceGraftCandidateStoreReader.AdmissionResonanceGraftCandidateStoreReaderID, "resonance-graft-candidate-store-reader-id-") ||
		resonanceGraftCandidateStoreReader.AdmissionResonanceGraftCandidateStoreReaderID != admissionLiveRouteTurnCandidateAdmissionResonanceGraftCandidateStoreReaderID(resonanceGraftCandidateStoreReader) ||
		!resonanceGraftCandidateStoreReader.Passed ||
		!resonanceGraftCandidateStoreReader.LiveReady ||
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
		resonanceGraftCandidateStoreReader.Reason != "resonance shadow graft candidate store read back without opening body" {
		t.Fatalf("resonance graft candidate store reader should replay only a closed store receipt: %+v", resonanceGraftCandidateStoreReader)
	}
	if resonanceGraftCandidateStoreReader.AdmissionResonanceGraftCandidateStoreID != resonanceGraftCandidateStore.AdmissionResonanceGraftCandidateStoreID ||
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
		resonanceGraftCandidateStoreReader.WriterImplementationID != writerImpl.WriterImplementationID ||
		resonanceGraftCandidateStoreReader.AdmissionLedgerID != ledger.AdmissionLedgerID ||
		resonanceGraftCandidateStoreReader.AdmissionWriterContractID != writerContract.WriterContractID ||
		resonanceGraftCandidateStoreReader.AdmissionWriterInventoryID != writerInventory.WriterInventoryID ||
		resonanceGraftCandidateStoreReader.AdmissionWriterPreflightID != writerPreflight.WriterPreflightID ||
		resonanceGraftCandidateStoreReader.AdmissionLiveStageID != liveStage.LiveStageID ||
		resonanceGraftCandidateStoreReader.AdmissionEnableGateID != armedGate.EnableGateID ||
		resonanceGraftCandidateStoreReader.AdmissionSwitchID != sw.SwitchID ||
		resonanceGraftCandidateStoreReader.AdmissionPromotionID != promotion.PromotionID ||
		resonanceGraftCandidateStoreReader.AdmissionDecisionID != decision.DecisionID ||
		resonanceGraftCandidateStoreReader.AdmissionAdapterID != adapter.AdmissionAdapterID ||
		resonanceGraftCandidateStoreReader.CandidateExecutionID != execution.ExecutionID ||
		resonanceGraftCandidateStoreReader.CandidateDraftID != draft.DraftID ||
		resonanceGraftCandidateStoreReader.CandidateRunID != candidate.RunID ||
		resonanceGraftCandidateStoreReader.CandidateTextHash != hashJSON(text) ||
		resonanceGraftCandidateStoreReader.TurnTextHash != obs.TextHash {
		t.Fatalf("admission resonance graft candidate store reader lost provenance: reader=%+v store=%+v", resonanceGraftCandidateStoreReader, resonanceGraftCandidateStore)
	}
	resonanceGraftAdmissionProof := admissionLiveRouteTurnCandidateAdmissionResonanceGraftAdmissionProofForStoreReader(resonanceGraftCandidateStoreReader)
	if resonanceGraftAdmissionProof.Schema != admissionLiveRouteTurnCandidateAdmissionResonanceGraftAdmissionProofSchema ||
		resonanceGraftAdmissionProof.Timing != "live_admission_resonance_graft_admission_proof" ||
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
		!strings.HasPrefix(resonanceGraftAdmissionProof.AdmissionResonanceGraftAdmissionProofCausalID, "resonance-graft-admission-proof-causal-") ||
		resonanceGraftAdmissionProof.AdmissionResonanceGraftAdmissionProofCausalID != admissionLiveRouteTurnCandidateAdmissionResonanceGraftAdmissionProofCausalID(resonanceGraftAdmissionProof) ||
		!strings.HasPrefix(resonanceGraftAdmissionProof.AdmissionResonanceGraftAdmissionProofHash, "resonance-graft-admission-proof-") ||
		resonanceGraftAdmissionProof.AdmissionResonanceGraftAdmissionProofHash != admissionLiveRouteTurnCandidateAdmissionResonanceGraftAdmissionProofHash(resonanceGraftAdmissionProof) ||
		!strings.HasPrefix(resonanceGraftAdmissionProof.AdmissionResonanceGraftAdmissionProofReplayHash, "resonance-graft-admission-proof-replay-") ||
		resonanceGraftAdmissionProof.AdmissionResonanceGraftAdmissionProofReplayHash != admissionLiveRouteTurnCandidateAdmissionResonanceGraftAdmissionProofReplayHash(resonanceGraftAdmissionProof) ||
		!strings.HasPrefix(resonanceGraftAdmissionProof.AdmissionResonanceGraftAdmissionProofReadBackHash, "resonance-graft-admission-proof-read-") ||
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
		resonanceGraftAdmissionProof.ContractsReady ||
		resonanceGraftAdmissionProof.WriteAllowed ||
		resonanceGraftAdmissionProof.AdmissionAllowed ||
		resonanceGraftAdmissionProof.LiveAdmissionEnabled ||
		resonanceGraftAdmissionProof.MutatesState ||
		resonanceGraftAdmissionProof.BodyTarget != "none" ||
		!strings.HasPrefix(resonanceGraftAdmissionProof.AdmissionResonanceGraftAdmissionProofID, "resonance-graft-admission-proof-id-") ||
		resonanceGraftAdmissionProof.AdmissionResonanceGraftAdmissionProofID != admissionLiveRouteTurnCandidateAdmissionResonanceGraftAdmissionProofID(resonanceGraftAdmissionProof) ||
		!resonanceGraftAdmissionProof.Passed ||
		!resonanceGraftAdmissionProof.LiveReady ||
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
		resonanceGraftAdmissionProof.Reason != "resonance shadow graft admission proved from read-back store without opening body" {
		t.Fatalf("resonance graft admission proof should prove only a closed reader receipt: %+v", resonanceGraftAdmissionProof)
	}
	if resonanceGraftAdmissionProof.AdmissionResonanceGraftCandidateStoreReaderID != resonanceGraftCandidateStoreReader.AdmissionResonanceGraftCandidateStoreReaderID ||
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
		resonanceGraftAdmissionProof.WriterImplementationID != writerImpl.WriterImplementationID ||
		resonanceGraftAdmissionProof.AdmissionLedgerID != ledger.AdmissionLedgerID ||
		resonanceGraftAdmissionProof.AdmissionWriterContractID != writerContract.WriterContractID ||
		resonanceGraftAdmissionProof.AdmissionWriterInventoryID != writerInventory.WriterInventoryID ||
		resonanceGraftAdmissionProof.AdmissionWriterPreflightID != writerPreflight.WriterPreflightID ||
		resonanceGraftAdmissionProof.AdmissionLiveStageID != liveStage.LiveStageID ||
		resonanceGraftAdmissionProof.AdmissionEnableGateID != armedGate.EnableGateID ||
		resonanceGraftAdmissionProof.AdmissionSwitchID != sw.SwitchID ||
		resonanceGraftAdmissionProof.AdmissionPromotionID != promotion.PromotionID ||
		resonanceGraftAdmissionProof.AdmissionDecisionID != decision.DecisionID ||
		resonanceGraftAdmissionProof.AdmissionAdapterID != adapter.AdmissionAdapterID ||
		resonanceGraftAdmissionProof.CandidateExecutionID != execution.ExecutionID ||
		resonanceGraftAdmissionProof.CandidateDraftID != draft.DraftID ||
		resonanceGraftAdmissionProof.CandidateRunID != candidate.RunID ||
		resonanceGraftAdmissionProof.CandidateTextHash != hashJSON(text) ||
		resonanceGraftAdmissionProof.TurnTextHash != obs.TextHash {
		t.Fatalf("admission resonance graft admission proof lost provenance: proof=%+v reader=%+v", resonanceGraftAdmissionProof, resonanceGraftCandidateStoreReader)
	}
	for _, receipt := range []struct {
		name               string
		bodyStatus         string
		availabilityStatus string
		availabilityReason string
		missingOrgans      []string
	}{
		{
			name:               "rollback implementation",
			bodyStatus:         rollbackImpl.BodyInventoryStatus,
			availabilityStatus: rollbackImpl.RouteAvailabilityStatus,
			availabilityReason: rollbackImpl.RouteAvailabilityReason,
			missingOrgans:      rollbackImpl.RouteMissingOrgans,
		},
		{
			name:               "ledger implementation",
			bodyStatus:         ledgerImpl.BodyInventoryStatus,
			availabilityStatus: ledgerImpl.RouteAvailabilityStatus,
			availabilityReason: ledgerImpl.RouteAvailabilityReason,
			missingOrgans:      ledgerImpl.RouteMissingOrgans,
		},
		{
			name:               "ledger persistence",
			bodyStatus:         ledgerPersistence.BodyInventoryStatus,
			availabilityStatus: ledgerPersistence.RouteAvailabilityStatus,
			availabilityReason: ledgerPersistence.RouteAvailabilityReason,
			missingOrgans:      ledgerPersistence.RouteMissingOrgans,
		},
		{
			name:               "ledger verification",
			bodyStatus:         ledgerVerification.BodyInventoryStatus,
			availabilityStatus: ledgerVerification.RouteAvailabilityStatus,
			availabilityReason: ledgerVerification.RouteAvailabilityReason,
			missingOrgans:      ledgerVerification.RouteMissingOrgans,
		},
		{
			name:               "admission readiness",
			bodyStatus:         readiness.BodyInventoryStatus,
			availabilityStatus: readiness.RouteAvailabilityStatus,
			availabilityReason: readiness.RouteAvailabilityReason,
			missingOrgans:      readiness.RouteMissingOrgans,
		},
		{
			name:               "admission permit",
			bodyStatus:         permit.BodyInventoryStatus,
			availabilityStatus: permit.RouteAvailabilityStatus,
			availabilityReason: permit.RouteAvailabilityReason,
			missingOrgans:      permit.RouteMissingOrgans,
		},
		{
			name:               "admission seal",
			bodyStatus:         seal.BodyInventoryStatus,
			availabilityStatus: seal.RouteAvailabilityStatus,
			availabilityReason: seal.RouteAvailabilityReason,
			missingOrgans:      seal.RouteMissingOrgans,
		},
		{
			name:               "resonance intent",
			bodyStatus:         resonanceIntent.BodyInventoryStatus,
			availabilityStatus: resonanceIntent.RouteAvailabilityStatus,
			availabilityReason: resonanceIntent.RouteAvailabilityReason,
			missingOrgans:      resonanceIntent.RouteMissingOrgans,
		},
		{
			name:               "resonance receiver",
			bodyStatus:         resonanceReceiver.BodyInventoryStatus,
			availabilityStatus: resonanceReceiver.RouteAvailabilityStatus,
			availabilityReason: resonanceReceiver.RouteAvailabilityReason,
			missingOrgans:      resonanceReceiver.RouteMissingOrgans,
		},
		{
			name:               "resonance observation",
			bodyStatus:         resonanceObservation.BodyInventoryStatus,
			availabilityStatus: resonanceObservation.RouteAvailabilityStatus,
			availabilityReason: resonanceObservation.RouteAvailabilityReason,
			missingOrgans:      resonanceObservation.RouteMissingOrgans,
		},
		{
			name:               "resonance graft boundary",
			bodyStatus:         resonanceGraftBoundary.BodyInventoryStatus,
			availabilityStatus: resonanceGraftBoundary.RouteAvailabilityStatus,
			availabilityReason: resonanceGraftBoundary.RouteAvailabilityReason,
			missingOrgans:      resonanceGraftBoundary.RouteMissingOrgans,
		},
		{
			name:               "resonance graft preflight",
			bodyStatus:         resonanceGraftPreflight.BodyInventoryStatus,
			availabilityStatus: resonanceGraftPreflight.RouteAvailabilityStatus,
			availabilityReason: resonanceGraftPreflight.RouteAvailabilityReason,
			missingOrgans:      resonanceGraftPreflight.RouteMissingOrgans,
		},
		{
			name:               "resonance graft gate",
			bodyStatus:         resonanceGraftGate.BodyInventoryStatus,
			availabilityStatus: resonanceGraftGate.RouteAvailabilityStatus,
			availabilityReason: resonanceGraftGate.RouteAvailabilityReason,
			missingOrgans:      resonanceGraftGate.RouteMissingOrgans,
		},
		{
			name:               "resonance graft candidate",
			bodyStatus:         resonanceGraftCandidate.BodyInventoryStatus,
			availabilityStatus: resonanceGraftCandidate.RouteAvailabilityStatus,
			availabilityReason: resonanceGraftCandidate.RouteAvailabilityReason,
			missingOrgans:      resonanceGraftCandidate.RouteMissingOrgans,
		},
		{
			name:               "resonance graft candidate store",
			bodyStatus:         resonanceGraftCandidateStore.BodyInventoryStatus,
			availabilityStatus: resonanceGraftCandidateStore.RouteAvailabilityStatus,
			availabilityReason: resonanceGraftCandidateStore.RouteAvailabilityReason,
			missingOrgans:      resonanceGraftCandidateStore.RouteMissingOrgans,
		},
		{
			name:               "resonance graft candidate store reader",
			bodyStatus:         resonanceGraftCandidateStoreReader.BodyInventoryStatus,
			availabilityStatus: resonanceGraftCandidateStoreReader.RouteAvailabilityStatus,
			availabilityReason: resonanceGraftCandidateStoreReader.RouteAvailabilityReason,
			missingOrgans:      resonanceGraftCandidateStoreReader.RouteMissingOrgans,
		},
		{
			name:               "resonance graft admission proof",
			bodyStatus:         resonanceGraftAdmissionProof.BodyInventoryStatus,
			availabilityStatus: resonanceGraftAdmissionProof.RouteAvailabilityStatus,
			availabilityReason: resonanceGraftAdmissionProof.RouteAvailabilityReason,
			missingOrgans:      resonanceGraftAdmissionProof.RouteMissingOrgans,
		},
	} {
		assertRouteBoundary(receipt.name, receipt.bodyStatus, receipt.availabilityStatus, receipt.availabilityReason, receipt.missingOrgans)
	}
	tamperedResonanceGraftCandidateStoreReaderForProof := resonanceGraftCandidateStoreReader
	tamperedResonanceGraftCandidateStoreReaderForProof.AdmissionResonanceGraftCandidateStoreReaderID = "resonance-graft-candidate-store-reader-id-tampered"
	tamperedResonanceGraftAdmissionProof := admissionLiveRouteTurnCandidateAdmissionResonanceGraftAdmissionProofForStoreReader(tamperedResonanceGraftCandidateStoreReaderForProof)
	if tamperedResonanceGraftAdmissionProof.Passed ||
		tamperedResonanceGraftAdmissionProof.AdmissionResonanceGraftAdmissionProofID != "" ||
		tamperedResonanceGraftAdmissionProof.AdmissionResonanceGraftAdmissionProofState != "blocked" ||
		tamperedResonanceGraftAdmissionProof.AdmissionResonanceGraftAdmissionProofAction != "reject" ||
		!tamperedResonanceGraftAdmissionProof.AdmissionResonanceGraftAdmissionProofDryRunOnly ||
		tamperedResonanceGraftAdmissionProof.AdmissionResonanceGraftAdmissionProofReady ||
		tamperedResonanceGraftAdmissionProof.LiveReady ||
		tamperedResonanceGraftAdmissionProof.BodyTarget != "none" ||
		tamperedResonanceGraftAdmissionProof.WriteAllowed ||
		tamperedResonanceGraftAdmissionProof.MutatesState ||
		tamperedResonanceGraftAdmissionProof.Reason != "candidate_admission_resonance_graft_candidate_store_reader_id_mismatch" {
		t.Fatalf("tampered resonance graft candidate store reader id should fail closed before admission proof: %+v", tamperedResonanceGraftAdmissionProof)
	}
	tamperedResonanceGraftCandidateStoreForReader := resonanceGraftCandidateStore
	tamperedResonanceGraftCandidateStoreForReader.AdmissionResonanceGraftCandidateStoreID = "resonance-graft-candidate-store-id-tampered"
	tamperedResonanceGraftCandidateStoreReader := admissionLiveRouteTurnCandidateAdmissionResonanceGraftCandidateStoreReaderForStore(tamperedResonanceGraftCandidateStoreForReader)
	if tamperedResonanceGraftCandidateStoreReader.Passed ||
		tamperedResonanceGraftCandidateStoreReader.AdmissionResonanceGraftCandidateStoreReaderID != "" ||
		tamperedResonanceGraftCandidateStoreReader.AdmissionResonanceGraftCandidateStoreReaderState != "blocked" ||
		tamperedResonanceGraftCandidateStoreReader.AdmissionResonanceGraftCandidateStoreReaderAction != "reject" ||
		!tamperedResonanceGraftCandidateStoreReader.AdmissionResonanceGraftCandidateStoreReaderDryRunOnly ||
		tamperedResonanceGraftCandidateStoreReader.AdmissionResonanceGraftCandidateStoreReaderReady ||
		tamperedResonanceGraftCandidateStoreReader.LiveReady ||
		tamperedResonanceGraftCandidateStoreReader.BodyTarget != "none" ||
		tamperedResonanceGraftCandidateStoreReader.WriteAllowed ||
		tamperedResonanceGraftCandidateStoreReader.MutatesState ||
		tamperedResonanceGraftCandidateStoreReader.Reason != "candidate_admission_resonance_graft_candidate_store_id_mismatch" {
		t.Fatalf("tampered resonance graft candidate store id should fail closed before graft candidate store reader: %+v", tamperedResonanceGraftCandidateStoreReader)
	}
	tamperedResonanceGraftCandidateForStore := resonanceGraftCandidate
	tamperedResonanceGraftCandidateForStore.AdmissionResonanceGraftCandidateID = "resonance-graft-candidate-id-tampered"
	tamperedResonanceGraftCandidateStore := admissionLiveRouteTurnCandidateAdmissionResonanceGraftCandidateStoreForCandidate(tamperedResonanceGraftCandidateForStore)
	if tamperedResonanceGraftCandidateStore.Passed ||
		tamperedResonanceGraftCandidateStore.AdmissionResonanceGraftCandidateStoreID != "" ||
		tamperedResonanceGraftCandidateStore.AdmissionResonanceGraftCandidateStoreState != "blocked" ||
		tamperedResonanceGraftCandidateStore.AdmissionResonanceGraftCandidateStoreAction != "reject" ||
		!tamperedResonanceGraftCandidateStore.AdmissionResonanceGraftCandidateStoreDryRunOnly ||
		tamperedResonanceGraftCandidateStore.AdmissionResonanceGraftCandidateStoreReady ||
		tamperedResonanceGraftCandidateStore.LiveReady ||
		tamperedResonanceGraftCandidateStore.BodyTarget != "none" ||
		tamperedResonanceGraftCandidateStore.WriteAllowed ||
		tamperedResonanceGraftCandidateStore.MutatesState ||
		tamperedResonanceGraftCandidateStore.Reason != "candidate_admission_resonance_graft_candidate_id_mismatch" {
		t.Fatalf("tampered resonance graft candidate id should fail closed before graft candidate store: %+v", tamperedResonanceGraftCandidateStore)
	}
	tamperedResonanceGraftCandidateGate := resonanceGraftGate
	tamperedResonanceGraftCandidateGate.AdmissionResonanceGraftGateID = "resonance-graft-gate-id-tampered"
	tamperedResonanceGraftCandidate := admissionLiveRouteTurnCandidateAdmissionResonanceGraftCandidateForGate(tamperedResonanceGraftCandidateGate)
	if tamperedResonanceGraftCandidate.Passed ||
		tamperedResonanceGraftCandidate.AdmissionResonanceGraftCandidateID != "" ||
		tamperedResonanceGraftCandidate.AdmissionResonanceGraftCandidateState != "blocked" ||
		tamperedResonanceGraftCandidate.AdmissionResonanceGraftCandidateAction != "reject" ||
		!tamperedResonanceGraftCandidate.AdmissionResonanceGraftCandidateDryRunOnly ||
		tamperedResonanceGraftCandidate.AdmissionResonanceGraftCandidateReady ||
		tamperedResonanceGraftCandidate.LiveReady ||
		tamperedResonanceGraftCandidate.BodyTarget != "none" ||
		tamperedResonanceGraftCandidate.WriteAllowed ||
		tamperedResonanceGraftCandidate.MutatesState ||
		tamperedResonanceGraftCandidate.Reason != "candidate_admission_resonance_graft_gate_id_mismatch" {
		t.Fatalf("tampered resonance graft gate id should fail closed before graft candidate: %+v", tamperedResonanceGraftCandidate)
	}
	tamperedResonanceGraftGatePreflight := resonanceGraftPreflight
	tamperedResonanceGraftGatePreflight.AdmissionResonanceGraftPreflightID = "resonance-graft-preflight-id-tampered"
	tamperedResonanceGraftGate := admissionLiveRouteTurnCandidateAdmissionResonanceGraftGateForPreflight(tamperedResonanceGraftGatePreflight)
	if tamperedResonanceGraftGate.Passed ||
		tamperedResonanceGraftGate.AdmissionResonanceGraftGateID != "" ||
		tamperedResonanceGraftGate.AdmissionResonanceGraftGateState != "blocked" ||
		tamperedResonanceGraftGate.AdmissionResonanceGraftGateAction != "reject" ||
		!tamperedResonanceGraftGate.AdmissionResonanceGraftGateDryRunOnly ||
		tamperedResonanceGraftGate.AdmissionResonanceGraftGateReady ||
		tamperedResonanceGraftGate.LiveReady ||
		tamperedResonanceGraftGate.BodyTarget != "none" ||
		tamperedResonanceGraftGate.WriteAllowed ||
		tamperedResonanceGraftGate.MutatesState ||
		tamperedResonanceGraftGate.Reason != "candidate_admission_resonance_graft_preflight_id_mismatch" {
		t.Fatalf("tampered resonance graft preflight id should fail closed before graft gate: %+v", tamperedResonanceGraftGate)
	}
	tamperedResonanceGraftPreflightBoundary := resonanceGraftBoundary
	tamperedResonanceGraftPreflightBoundary.AdmissionResonanceGraftBoundaryID = "resonance-graft-boundary-id-tampered"
	tamperedResonanceGraftPreflight := admissionLiveRouteTurnCandidateAdmissionResonanceGraftPreflightForBoundary(tamperedResonanceGraftPreflightBoundary)
	if tamperedResonanceGraftPreflight.Passed ||
		tamperedResonanceGraftPreflight.AdmissionResonanceGraftPreflightID != "" ||
		tamperedResonanceGraftPreflight.AdmissionResonanceGraftPreflightState != "blocked" ||
		tamperedResonanceGraftPreflight.AdmissionResonanceGraftPreflightAction != "reject" ||
		!tamperedResonanceGraftPreflight.AdmissionResonanceGraftPreflightDryRunOnly ||
		tamperedResonanceGraftPreflight.AdmissionResonanceGraftPreflightReady ||
		tamperedResonanceGraftPreflight.LiveReady ||
		tamperedResonanceGraftPreflight.BodyTarget != "none" ||
		tamperedResonanceGraftPreflight.WriteAllowed ||
		tamperedResonanceGraftPreflight.MutatesState ||
		tamperedResonanceGraftPreflight.Reason != "candidate_admission_resonance_graft_boundary_id_mismatch" {
		t.Fatalf("tampered resonance graft boundary id should fail closed before graft preflight: %+v", tamperedResonanceGraftPreflight)
	}
	tamperedResonanceGraftBoundaryObservation := resonanceObservation
	tamperedResonanceGraftBoundaryObservation.AdmissionResonanceObservationID = "resonance-observation-tampered"
	tamperedResonanceGraftBoundary := admissionLiveRouteTurnCandidateAdmissionResonanceGraftBoundaryForObservation(tamperedResonanceGraftBoundaryObservation)
	if tamperedResonanceGraftBoundary.Passed ||
		tamperedResonanceGraftBoundary.AdmissionResonanceGraftBoundaryID != "" ||
		tamperedResonanceGraftBoundary.AdmissionResonanceGraftBoundaryState != "blocked" ||
		tamperedResonanceGraftBoundary.AdmissionResonanceGraftBoundaryAction != "reject" ||
		!tamperedResonanceGraftBoundary.AdmissionResonanceGraftBoundaryDryRunOnly ||
		tamperedResonanceGraftBoundary.AdmissionResonanceGraftBoundaryReady ||
		tamperedResonanceGraftBoundary.LiveReady ||
		tamperedResonanceGraftBoundary.BodyTarget != "none" ||
		tamperedResonanceGraftBoundary.WriteAllowed ||
		tamperedResonanceGraftBoundary.MutatesState ||
		tamperedResonanceGraftBoundary.Reason != "candidate_admission_resonance_observation_id_mismatch" {
		t.Fatalf("tampered resonance observation id should fail closed before graft boundary: %+v", tamperedResonanceGraftBoundary)
	}
	tamperedResonanceObservationReceiver := resonanceReceiver
	tamperedResonanceObservationReceiver.AdmissionResonanceReceiverID = "resonance-receiver-tampered"
	tamperedResonanceObservation := admissionLiveRouteTurnCandidateAdmissionResonanceObservationForReceiver(tamperedResonanceObservationReceiver)
	if tamperedResonanceObservation.Passed ||
		tamperedResonanceObservation.AdmissionResonanceObservationID != "" ||
		tamperedResonanceObservation.AdmissionResonanceObservationState != "blocked" ||
		tamperedResonanceObservation.AdmissionResonanceObservationAction != "reject" ||
		!tamperedResonanceObservation.AdmissionResonanceObservationDryRunOnly ||
		tamperedResonanceObservation.AdmissionResonanceObservationReady ||
		tamperedResonanceObservation.LiveReady ||
		tamperedResonanceObservation.BodyTarget != "none" ||
		tamperedResonanceObservation.WriteAllowed ||
		tamperedResonanceObservation.MutatesState ||
		tamperedResonanceObservation.Reason != "candidate_admission_resonance_receiver_id_mismatch" {
		t.Fatalf("tampered resonance receiver id should fail closed before observation: %+v", tamperedResonanceObservation)
	}
	tamperedResonanceReceiverIntent := resonanceIntent
	tamperedResonanceReceiverIntent.AdmissionResonanceIntentID = "resonance-intent-tampered"
	tamperedResonanceReceiver := admissionLiveRouteTurnCandidateAdmissionResonanceReceiverForIntent(tamperedResonanceReceiverIntent)
	if tamperedResonanceReceiver.Passed ||
		tamperedResonanceReceiver.AdmissionResonanceReceiverID != "" ||
		tamperedResonanceReceiver.AdmissionResonanceReceiverState != "blocked" ||
		tamperedResonanceReceiver.AdmissionResonanceReceiverAction != "reject" ||
		!tamperedResonanceReceiver.AdmissionResonanceReceiverDryRunOnly ||
		tamperedResonanceReceiver.AdmissionResonanceReceiverReady ||
		tamperedResonanceReceiver.LiveReady ||
		tamperedResonanceReceiver.BodyTarget != "none" ||
		tamperedResonanceReceiver.WriteAllowed ||
		tamperedResonanceReceiver.MutatesState ||
		tamperedResonanceReceiver.Reason != "candidate_admission_resonance_intent_id_mismatch" {
		t.Fatalf("tampered resonance intent id should fail closed before receiver: %+v", tamperedResonanceReceiver)
	}
	tamperedResonanceIntentGate := finalGate
	tamperedResonanceIntentGate.AdmissionFinalGateID = "admission-final-gate-tampered"
	tamperedResonanceIntent := admissionLiveRouteTurnCandidateAdmissionResonanceIntentForFinalGate(tamperedResonanceIntentGate)
	if tamperedResonanceIntent.Passed ||
		tamperedResonanceIntent.AdmissionResonanceIntentID != "" ||
		tamperedResonanceIntent.AdmissionResonanceIntentState != "blocked" ||
		tamperedResonanceIntent.AdmissionResonanceIntentAction != "reject" ||
		!tamperedResonanceIntent.AdmissionResonanceIntentDryRunOnly ||
		tamperedResonanceIntent.AdmissionResonanceIntentReady ||
		tamperedResonanceIntent.LiveReady ||
		tamperedResonanceIntent.BodyTarget != "none" ||
		tamperedResonanceIntent.WriteAllowed ||
		tamperedResonanceIntent.MutatesState ||
		tamperedResonanceIntent.Reason != "candidate_admission_final_gate_id_mismatch" {
		t.Fatalf("tampered final gate id should fail closed before resonance intent: %+v", tamperedResonanceIntent)
	}
	tamperedFinalGateSeal := seal
	tamperedFinalGateSeal.AdmissionSealID = "admission-seal-tampered"
	tamperedFinalGate := admissionLiveRouteTurnCandidateAdmissionFinalGateForSeal(tamperedFinalGateSeal)
	if tamperedFinalGate.Passed ||
		tamperedFinalGate.AdmissionFinalGateID != "" ||
		tamperedFinalGate.AdmissionFinalGateState != "blocked" ||
		tamperedFinalGate.AdmissionFinalGateAction != "reject" ||
		!tamperedFinalGate.AdmissionFinalGateDryRunOnly ||
		tamperedFinalGate.AdmissionFinalGateReady ||
		tamperedFinalGate.WriteAllowed ||
		tamperedFinalGate.MutatesState ||
		tamperedFinalGate.Reason != "candidate_admission_seal_id_mismatch" {
		t.Fatalf("tampered seal id should fail closed before final gate: %+v", tamperedFinalGate)
	}
	tamperedPermit := permit
	tamperedPermit.AdmissionPermitID = "admission-permit-tampered"
	tamperedSeal := admissionLiveRouteTurnCandidateAdmissionSealForPermit(tamperedPermit)
	if tamperedSeal.Passed ||
		tamperedSeal.AdmissionSealID != "" ||
		tamperedSeal.AdmissionSealState != "blocked" ||
		tamperedSeal.AdmissionSealAction != "reject" ||
		!tamperedSeal.AdmissionSealDryRunOnly ||
		tamperedSeal.AdmissionSealReady ||
		tamperedSeal.WriteAllowed ||
		tamperedSeal.MutatesState ||
		tamperedSeal.Reason != "candidate_admission_permit_id_mismatch" {
		t.Fatalf("tampered permit id should fail closed before seal: %+v", tamperedSeal)
	}
	t.Setenv("AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_PERMIT_KEY", "")
	missingPermitKey := admissionLiveRouteTurnCandidateAdmissionPermitForReadiness(readiness)
	if missingPermitKey.Passed ||
		missingPermitKey.AdmissionPermitID != "" ||
		missingPermitKey.AdmissionPermitState != "blocked" ||
		missingPermitKey.AdmissionPermitAction != "reject" ||
		!missingPermitKey.AdmissionPermitDryRunOnly ||
		missingPermitKey.ManualPermitRequested ||
		missingPermitKey.PermitKeyMatched ||
		missingPermitKey.WriteAllowed ||
		missingPermitKey.MutatesState ||
		missingPermitKey.Reason != "live_admission_permit_key_missing" {
		t.Fatalf("missing permit key should fail closed: %+v", missingPermitKey)
	}
	t.Setenv("AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_PERMIT_KEY", "wrong")
	wrongPermitKey := admissionLiveRouteTurnCandidateAdmissionPermitForReadiness(readiness)
	if wrongPermitKey.Passed ||
		wrongPermitKey.AdmissionPermitID != "" ||
		wrongPermitKey.AdmissionPermitState != "blocked" ||
		wrongPermitKey.AdmissionPermitAction != "reject" ||
		!wrongPermitKey.AdmissionPermitDryRunOnly ||
		!wrongPermitKey.ManualPermitRequested ||
		wrongPermitKey.PermitKeyMatched ||
		wrongPermitKey.WriteAllowed ||
		wrongPermitKey.MutatesState ||
		wrongPermitKey.Reason != "live_admission_permit_key_mismatch" {
		t.Fatalf("wrong permit key should fail closed: %+v", wrongPermitKey)
	}
	t.Setenv("AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_ENABLE_GATE_KEY", "")
	blockedStage := admissionLiveRouteTurnCandidateAdmissionLiveStageForEnableGate(gate)
	if blockedStage.Passed ||
		blockedStage.LiveStageID != "" ||
		blockedStage.StageState != "blocked" ||
		blockedStage.StageAction != "reject" ||
		blockedStage.Reason != "candidate_admission_enable_gate_not_armed" {
		t.Fatalf("closed enable gate should not produce a live stage: %+v", blockedStage)
	}
	blockedPreflight := admissionLiveRouteTurnCandidateAdmissionWriterPreflightForLiveStage(blockedStage)
	if blockedPreflight.Passed ||
		blockedPreflight.WriterPreflightID != "" ||
		blockedPreflight.WriterState != "blocked" ||
		blockedPreflight.RollbackState != "blocked" ||
		blockedPreflight.WriteAllowed ||
		blockedPreflight.Reason != "candidate_admission_live_stage_failed: candidate_admission_enable_gate_not_armed" {
		t.Fatalf("closed stage should not produce a writer preflight: %+v", blockedPreflight)
	}
	blockedInventory := admissionLiveRouteTurnCandidateAdmissionWriterInventoryForPreflight(blockedPreflight)
	if blockedInventory.Passed ||
		blockedInventory.WriterInventoryID != "" ||
		blockedInventory.InventoryState != "blocked" ||
		blockedInventory.InventoryAction != "reject" ||
		blockedInventory.ContractsReady ||
		blockedInventory.WriteAllowed ||
		blockedInventory.Reason != "candidate_admission_writer_preflight_failed: candidate_admission_live_stage_failed: candidate_admission_enable_gate_not_armed" {
		t.Fatalf("closed preflight should not produce a writer inventory: %+v", blockedInventory)
	}
	blockedContract := admissionLiveRouteTurnCandidateAdmissionWriterContractForInventory(blockedInventory)
	if blockedContract.Passed ||
		blockedContract.WriterContractID != "" ||
		blockedContract.ContractState != "blocked" ||
		blockedContract.ContractAction != "reject" ||
		blockedContract.ContractShapeReady ||
		blockedContract.ContractsReady ||
		blockedContract.WriteAllowed ||
		blockedContract.MutatesState ||
		blockedContract.Reason != "candidate_admission_writer_inventory_failed: candidate_admission_writer_preflight_failed: candidate_admission_live_stage_failed: candidate_admission_enable_gate_not_armed" {
		t.Fatalf("closed inventory should not produce a writer contract: %+v", blockedContract)
	}
	blockedLedger := admissionLiveRouteTurnCandidateAdmissionLedgerForWriterContract(blockedContract)
	if blockedLedger.Passed ||
		blockedLedger.AdmissionLedgerID != "" ||
		blockedLedger.LedgerState != "blocked" ||
		blockedLedger.LedgerAction != "reject" ||
		blockedLedger.LedgerAppendReady ||
		blockedLedger.LedgerReceiptPersisted ||
		blockedLedger.ContractsReady ||
		blockedLedger.WriteAllowed ||
		blockedLedger.MutatesState ||
		blockedLedger.Reason != "candidate_admission_writer_contract_failed: candidate_admission_writer_inventory_failed: candidate_admission_writer_preflight_failed: candidate_admission_live_stage_failed: candidate_admission_enable_gate_not_armed" {
		t.Fatalf("closed contract should not produce a ledger receipt: %+v", blockedLedger)
	}
	blockedWriterImpl := admissionLiveRouteTurnCandidateAdmissionWriterImplementationForLedger(blockedLedger)
	if blockedWriterImpl.Passed ||
		blockedWriterImpl.WriterImplementationID != "" ||
		blockedWriterImpl.ImplementationState != "blocked" ||
		blockedWriterImpl.ImplementationAction != "reject" ||
		blockedWriterImpl.AppendOnly ||
		blockedWriterImpl.ImplementationContractReady ||
		blockedWriterImpl.ContractsReady ||
		blockedWriterImpl.WriteAllowed ||
		blockedWriterImpl.MutatesState ||
		blockedWriterImpl.Reason != "candidate_admission_ledger_failed: candidate_admission_writer_contract_failed: candidate_admission_writer_inventory_failed: candidate_admission_writer_preflight_failed: candidate_admission_live_stage_failed: candidate_admission_enable_gate_not_armed" {
		t.Fatalf("closed ledger should not produce a writer implementation receipt: %+v", blockedWriterImpl)
	}
	blockedWriterReceipt := admissionLiveRouteTurnCandidateAdmissionWriterReceiptForImplementation(blockedWriterImpl)
	if blockedWriterReceipt.Passed ||
		blockedWriterReceipt.WriterReceiptID != "" ||
		blockedWriterReceipt.WriterReceiptState != "blocked" ||
		blockedWriterReceipt.WriterReceiptAction != "reject" ||
		blockedWriterReceipt.WriterReceiptPersisted ||
		blockedWriterReceipt.ShadowWriteAllowed ||
		blockedWriterReceipt.WriterReady ||
		blockedWriterReceipt.WriterImplementationReady ||
		blockedWriterReceipt.ContractsReady ||
		blockedWriterReceipt.WriteAllowed ||
		blockedWriterReceipt.MutatesState ||
		blockedWriterReceipt.Reason != "candidate_admission_writer_implementation_failed: candidate_admission_ledger_failed: candidate_admission_writer_contract_failed: candidate_admission_writer_inventory_failed: candidate_admission_writer_preflight_failed: candidate_admission_live_stage_failed: candidate_admission_enable_gate_not_armed" {
		t.Fatalf("closed writer implementation should not produce a writer receipt: %+v", blockedWriterReceipt)
	}
	blockedRollbackImpl := admissionLiveRouteTurnCandidateAdmissionRollbackImplementationForWriterReceipt(blockedWriterReceipt)
	if blockedRollbackImpl.Passed ||
		blockedRollbackImpl.RollbackImplementationID != "" ||
		blockedRollbackImpl.RollbackImplementationState != "blocked" ||
		blockedRollbackImpl.RollbackImplementationAction != "reject" ||
		blockedRollbackImpl.RollbackReady ||
		blockedRollbackImpl.RollbackImplementationReady ||
		blockedRollbackImpl.WriteAllowed ||
		blockedRollbackImpl.MutatesState ||
		blockedRollbackImpl.Reason != "candidate_admission_writer_receipt_failed: candidate_admission_writer_implementation_failed: candidate_admission_ledger_failed: candidate_admission_writer_contract_failed: candidate_admission_writer_inventory_failed: candidate_admission_writer_preflight_failed: candidate_admission_live_stage_failed: candidate_admission_enable_gate_not_armed" {
		t.Fatalf("closed writer receipt should not produce rollback implementation: %+v", blockedRollbackImpl)
	}
	blockedLedgerImpl := admissionLiveRouteTurnCandidateAdmissionLedgerImplementationForRollbackImplementation(blockedRollbackImpl)
	if blockedLedgerImpl.Passed ||
		blockedLedgerImpl.LedgerImplementationID != "" ||
		blockedLedgerImpl.LedgerImplementationState != "blocked" ||
		blockedLedgerImpl.LedgerImplementationAction != "reject" ||
		blockedLedgerImpl.LedgerImplementationAppendOnly ||
		!blockedLedgerImpl.LedgerImplementationDryRunOnly ||
		blockedLedgerImpl.LedgerImplementationReceiptPersisted ||
		blockedLedgerImpl.LedgerImplementationReady ||
		blockedLedgerImpl.WriteAllowed ||
		blockedLedgerImpl.MutatesState ||
		blockedLedgerImpl.Reason != "candidate_admission_rollback_implementation_failed: candidate_admission_writer_receipt_failed: candidate_admission_writer_implementation_failed: candidate_admission_ledger_failed: candidate_admission_writer_contract_failed: candidate_admission_writer_inventory_failed: candidate_admission_writer_preflight_failed: candidate_admission_live_stage_failed: candidate_admission_enable_gate_not_armed" {
		t.Fatalf("closed rollback implementation should not produce ledger implementation: %+v", blockedLedgerImpl)
	}
	blockedLedgerPersistence := admissionLiveRouteTurnCandidateAdmissionLedgerPersistenceForLedgerImplementation(blockedLedgerImpl)
	if blockedLedgerPersistence.Passed ||
		blockedLedgerPersistence.LedgerPersistenceID != "" ||
		blockedLedgerPersistence.LedgerPersistenceState != "blocked" ||
		blockedLedgerPersistence.LedgerPersistenceAction != "reject" ||
		blockedLedgerPersistence.LedgerPersistenceAppendOnly ||
		!blockedLedgerPersistence.LedgerPersistenceDryRunOnly ||
		blockedLedgerPersistence.LedgerPersistenceReceiptPersisted ||
		blockedLedgerPersistence.LedgerPersistenceReady ||
		blockedLedgerPersistence.WriteAllowed ||
		blockedLedgerPersistence.MutatesState ||
		blockedLedgerPersistence.Reason != "candidate_admission_ledger_implementation_failed: candidate_admission_rollback_implementation_failed: candidate_admission_writer_receipt_failed: candidate_admission_writer_implementation_failed: candidate_admission_ledger_failed: candidate_admission_writer_contract_failed: candidate_admission_writer_inventory_failed: candidate_admission_writer_preflight_failed: candidate_admission_live_stage_failed: candidate_admission_enable_gate_not_armed" {
		t.Fatalf("closed ledger implementation should not persist ledger receipt: %+v", blockedLedgerPersistence)
	}
	blockedLedgerVerification := admissionLiveRouteTurnCandidateAdmissionLedgerVerificationForLedgerPersistence(blockedLedgerPersistence)
	if blockedLedgerVerification.Passed ||
		blockedLedgerVerification.LedgerVerificationID != "" ||
		blockedLedgerVerification.LedgerVerificationState != "blocked" ||
		blockedLedgerVerification.LedgerVerificationAction != "reject" ||
		blockedLedgerVerification.LedgerVerificationAppendOnly ||
		!blockedLedgerVerification.LedgerVerificationDryRunOnly ||
		blockedLedgerVerification.LedgerVerificationReceiptReadBack ||
		blockedLedgerVerification.LedgerVerificationReceiptVerified ||
		blockedLedgerVerification.LedgerVerificationReady ||
		blockedLedgerVerification.WriteAllowed ||
		blockedLedgerVerification.MutatesState ||
		blockedLedgerVerification.Reason != "candidate_admission_ledger_persistence_failed: candidate_admission_ledger_implementation_failed: candidate_admission_rollback_implementation_failed: candidate_admission_writer_receipt_failed: candidate_admission_writer_implementation_failed: candidate_admission_ledger_failed: candidate_admission_writer_contract_failed: candidate_admission_writer_inventory_failed: candidate_admission_writer_preflight_failed: candidate_admission_live_stage_failed: candidate_admission_enable_gate_not_armed" {
		t.Fatalf("closed ledger persistence should not verify a ledger receipt: %+v", blockedLedgerVerification)
	}
	blockedReadiness := admissionLiveRouteTurnCandidateAdmissionReadinessForLedgerVerification(blockedLedgerVerification)
	if blockedReadiness.Passed ||
		blockedReadiness.AdmissionReadinessID != "" ||
		blockedReadiness.AdmissionReadinessState != "blocked" ||
		blockedReadiness.AdmissionReadinessAction != "reject" ||
		!blockedReadiness.AdmissionReadinessDryRunOnly ||
		blockedReadiness.AdmissionReadinessLedgerVerified ||
		blockedReadiness.AdmissionReadinessReady ||
		blockedReadiness.WriteAllowed ||
		blockedReadiness.MutatesState ||
		blockedReadiness.Reason != "candidate_admission_ledger_verification_failed: candidate_admission_ledger_persistence_failed: candidate_admission_ledger_implementation_failed: candidate_admission_rollback_implementation_failed: candidate_admission_writer_receipt_failed: candidate_admission_writer_implementation_failed: candidate_admission_ledger_failed: candidate_admission_writer_contract_failed: candidate_admission_writer_inventory_failed: candidate_admission_writer_preflight_failed: candidate_admission_live_stage_failed: candidate_admission_enable_gate_not_armed" {
		t.Fatalf("closed ledger verification should not declare admission readiness: %+v", blockedReadiness)
	}
	blockedPermit := admissionLiveRouteTurnCandidateAdmissionPermitForReadiness(blockedReadiness)
	if blockedPermit.Passed ||
		blockedPermit.AdmissionPermitID != "" ||
		blockedPermit.AdmissionPermitState != "blocked" ||
		blockedPermit.AdmissionPermitAction != "reject" ||
		!blockedPermit.AdmissionPermitDryRunOnly ||
		blockedPermit.AdmissionPermitReadinessVerified ||
		blockedPermit.AdmissionPermitReady ||
		blockedPermit.WriteAllowed ||
		blockedPermit.MutatesState ||
		blockedPermit.Reason != "candidate_admission_readiness_failed: candidate_admission_ledger_verification_failed: candidate_admission_ledger_persistence_failed: candidate_admission_ledger_implementation_failed: candidate_admission_rollback_implementation_failed: candidate_admission_writer_receipt_failed: candidate_admission_writer_implementation_failed: candidate_admission_ledger_failed: candidate_admission_writer_contract_failed: candidate_admission_writer_inventory_failed: candidate_admission_writer_preflight_failed: candidate_admission_live_stage_failed: candidate_admission_enable_gate_not_armed" {
		t.Fatalf("blocked readiness should not produce admission permit: %+v", blockedPermit)
	}
	blockedSeal := admissionLiveRouteTurnCandidateAdmissionSealForPermit(blockedPermit)
	if blockedSeal.Passed ||
		blockedSeal.AdmissionSealID != "" ||
		blockedSeal.AdmissionSealState != "blocked" ||
		blockedSeal.AdmissionSealAction != "reject" ||
		!blockedSeal.AdmissionSealDryRunOnly ||
		blockedSeal.AdmissionSealPermitVerified ||
		blockedSeal.AdmissionSealReady ||
		blockedSeal.WriteAllowed ||
		blockedSeal.MutatesState ||
		blockedSeal.Reason != "candidate_admission_permit_failed: candidate_admission_readiness_failed: candidate_admission_ledger_verification_failed: candidate_admission_ledger_persistence_failed: candidate_admission_ledger_implementation_failed: candidate_admission_rollback_implementation_failed: candidate_admission_writer_receipt_failed: candidate_admission_writer_implementation_failed: candidate_admission_ledger_failed: candidate_admission_writer_contract_failed: candidate_admission_writer_inventory_failed: candidate_admission_writer_preflight_failed: candidate_admission_live_stage_failed: candidate_admission_enable_gate_not_armed" {
		t.Fatalf("blocked permit should not produce admission seal: %+v", blockedSeal)
	}
	blockedFinalGate := admissionLiveRouteTurnCandidateAdmissionFinalGateForSeal(blockedSeal)
	if blockedFinalGate.Passed ||
		blockedFinalGate.AdmissionFinalGateID != "" ||
		blockedFinalGate.AdmissionFinalGateState != "blocked" ||
		blockedFinalGate.AdmissionFinalGateAction != "reject" ||
		!blockedFinalGate.AdmissionFinalGateDryRunOnly ||
		blockedFinalGate.AdmissionFinalGateSealVerified ||
		blockedFinalGate.AdmissionFinalGateReady ||
		blockedFinalGate.WriteAllowed ||
		blockedFinalGate.MutatesState ||
		blockedFinalGate.Reason != "candidate_admission_seal_failed: candidate_admission_permit_failed: candidate_admission_readiness_failed: candidate_admission_ledger_verification_failed: candidate_admission_ledger_persistence_failed: candidate_admission_ledger_implementation_failed: candidate_admission_rollback_implementation_failed: candidate_admission_writer_receipt_failed: candidate_admission_writer_implementation_failed: candidate_admission_ledger_failed: candidate_admission_writer_contract_failed: candidate_admission_writer_inventory_failed: candidate_admission_writer_preflight_failed: candidate_admission_live_stage_failed: candidate_admission_enable_gate_not_armed" {
		t.Fatalf("blocked seal should not pass admission final gate: %+v", blockedFinalGate)
	}
	blockedResonanceIntent := admissionLiveRouteTurnCandidateAdmissionResonanceIntentForFinalGate(blockedFinalGate)
	if blockedResonanceIntent.Passed ||
		blockedResonanceIntent.AdmissionResonanceIntentID != "" ||
		blockedResonanceIntent.AdmissionResonanceIntentState != "blocked" ||
		blockedResonanceIntent.AdmissionResonanceIntentAction != "reject" ||
		!blockedResonanceIntent.AdmissionResonanceIntentDryRunOnly ||
		blockedResonanceIntent.AdmissionResonanceIntentFinalGateVerified ||
		blockedResonanceIntent.AdmissionResonanceIntentReady ||
		blockedResonanceIntent.LiveReady ||
		blockedResonanceIntent.BodyTarget != "none" ||
		blockedResonanceIntent.WriteAllowed ||
		blockedResonanceIntent.MutatesState ||
		blockedResonanceIntent.Reason != "candidate_admission_final_gate_failed: candidate_admission_seal_failed: candidate_admission_permit_failed: candidate_admission_readiness_failed: candidate_admission_ledger_verification_failed: candidate_admission_ledger_persistence_failed: candidate_admission_ledger_implementation_failed: candidate_admission_rollback_implementation_failed: candidate_admission_writer_receipt_failed: candidate_admission_writer_implementation_failed: candidate_admission_ledger_failed: candidate_admission_writer_contract_failed: candidate_admission_writer_inventory_failed: candidate_admission_writer_preflight_failed: candidate_admission_live_stage_failed: candidate_admission_enable_gate_not_armed" {
		t.Fatalf("blocked final gate should not produce resonance intent: %+v", blockedResonanceIntent)
	}
	blockedResonanceReceiver := admissionLiveRouteTurnCandidateAdmissionResonanceReceiverForIntent(blockedResonanceIntent)
	if blockedResonanceReceiver.Passed ||
		blockedResonanceReceiver.AdmissionResonanceReceiverID != "" ||
		blockedResonanceReceiver.AdmissionResonanceReceiverState != "blocked" ||
		blockedResonanceReceiver.AdmissionResonanceReceiverAction != "reject" ||
		!blockedResonanceReceiver.AdmissionResonanceReceiverDryRunOnly ||
		blockedResonanceReceiver.AdmissionResonanceReceiverIntentVerified ||
		blockedResonanceReceiver.AdmissionResonanceReceiverReady ||
		blockedResonanceReceiver.AdmissionResonanceReceiverCausalID != "" ||
		blockedResonanceReceiver.AdmissionResonanceReceiverPreStateHash != "" ||
		blockedResonanceReceiver.AdmissionResonanceReceiverPostStateHash != "" ||
		blockedResonanceReceiver.AdmissionResonanceReceiverStateDeltaHash != "" ||
		blockedResonanceReceiver.LiveReady ||
		blockedResonanceReceiver.BodyTarget != "none" ||
		blockedResonanceReceiver.WriteAllowed ||
		blockedResonanceReceiver.MutatesState ||
		blockedResonanceReceiver.Reason != "candidate_admission_resonance_intent_failed: candidate_admission_final_gate_failed: candidate_admission_seal_failed: candidate_admission_permit_failed: candidate_admission_readiness_failed: candidate_admission_ledger_verification_failed: candidate_admission_ledger_persistence_failed: candidate_admission_ledger_implementation_failed: candidate_admission_rollback_implementation_failed: candidate_admission_writer_receipt_failed: candidate_admission_writer_implementation_failed: candidate_admission_ledger_failed: candidate_admission_writer_contract_failed: candidate_admission_writer_inventory_failed: candidate_admission_writer_preflight_failed: candidate_admission_live_stage_failed: candidate_admission_enable_gate_not_armed" {
		t.Fatalf("blocked resonance intent should not produce receiver preview: %+v", blockedResonanceReceiver)
	}
	wrongStage := admissionLiveRouteTurnCandidateAdmissionLiveStageForEnableGate(wrongGate)
	if wrongStage.Passed ||
		wrongStage.LiveStageID != "" ||
		wrongStage.StageState != "blocked" ||
		wrongStage.Reason != "candidate_admission_enable_gate_failed: live_admission_enable_gate_key_mismatch" {
		t.Fatalf("wrong-key enable gate should not produce a live stage: %+v", wrongStage)
	}

	badExecution := execution
	badExecution.Runner = admissionLiveRouteTurnCandidateExecutionRunnerProvided
	rejected := admissionLiveRouteTurnCandidateAdmissionDecisionForShadow(
		badExecution,
		generatorAdapter,
		draft,
		admission,
		adapter,
		candidate,
	)
	if rejected.Passed ||
		rejected.LiveReady ||
		rejected.DecisionID != "" ||
		rejected.Decision != "reject" ||
		rejected.Reason != "candidate execution runner provided_text is not nano-direct" {
		t.Fatalf("provided-text execution should not reach live-ready decision: %+v", rejected)
	}
	rejectedPromotion := admissionLiveRouteTurnCandidateAdmissionPromotionForDecision(rejected)
	if rejectedPromotion.Passed ||
		rejectedPromotion.PromotionID != "" ||
		rejectedPromotion.Promotion != "blocked" ||
		rejectedPromotion.Reason != "candidate_admission_decision_failed: candidate execution runner provided_text is not nano-direct" {
		t.Fatalf("rejected decision should not produce a promotion: %+v", rejectedPromotion)
	}
	rejectedSwitch := admissionLiveRouteTurnCandidateAdmissionSwitchForPromotion(rejectedPromotion)
	if rejectedSwitch.Passed ||
		rejectedSwitch.SwitchID != "" ||
		rejectedSwitch.SwitchState != "blocked" ||
		rejectedSwitch.Reason != "candidate_admission_promotion_failed: candidate_admission_decision_failed: candidate execution runner provided_text is not nano-direct" {
		t.Fatalf("rejected promotion should not pass switch guard: %+v", rejectedSwitch)
	}
	rejectedGate := admissionLiveRouteTurnCandidateAdmissionEnableGateForSwitch(rejectedSwitch)
	if rejectedGate.Passed ||
		rejectedGate.EnableGateID != "" ||
		rejectedGate.EnableState != "blocked" ||
		rejectedGate.Reason != "candidate_admission_switch_failed: candidate_admission_promotion_failed: candidate_admission_decision_failed: candidate execution runner provided_text is not nano-direct" {
		t.Fatalf("rejected switch should not pass enable gate: %+v", rejectedGate)
	}
	rejectedStage := admissionLiveRouteTurnCandidateAdmissionLiveStageForEnableGate(rejectedGate)
	if rejectedStage.Passed ||
		rejectedStage.LiveStageID != "" ||
		rejectedStage.StageState != "blocked" ||
		rejectedStage.Reason != "candidate_admission_enable_gate_failed: candidate_admission_switch_failed: candidate_admission_promotion_failed: candidate_admission_decision_failed: candidate execution runner provided_text is not nano-direct" {
		t.Fatalf("rejected enable gate should not produce a live stage: %+v", rejectedStage)
	}
	rejectedPreflight := admissionLiveRouteTurnCandidateAdmissionWriterPreflightForLiveStage(rejectedStage)
	if rejectedPreflight.Passed ||
		rejectedPreflight.WriterPreflightID != "" ||
		rejectedPreflight.Reason != "candidate_admission_live_stage_failed: candidate_admission_enable_gate_failed: candidate_admission_switch_failed: candidate_admission_promotion_failed: candidate_admission_decision_failed: candidate execution runner provided_text is not nano-direct" {
		t.Fatalf("rejected stage should not produce a writer preflight: %+v", rejectedPreflight)
	}
	rejectedInventory := admissionLiveRouteTurnCandidateAdmissionWriterInventoryForPreflight(rejectedPreflight)
	if rejectedInventory.Passed ||
		rejectedInventory.WriterInventoryID != "" ||
		rejectedInventory.Reason != "candidate_admission_writer_preflight_failed: candidate_admission_live_stage_failed: candidate_admission_enable_gate_failed: candidate_admission_switch_failed: candidate_admission_promotion_failed: candidate_admission_decision_failed: candidate execution runner provided_text is not nano-direct" {
		t.Fatalf("rejected preflight should not produce a writer inventory: %+v", rejectedInventory)
	}
	rejectedContract := admissionLiveRouteTurnCandidateAdmissionWriterContractForInventory(rejectedInventory)
	if rejectedContract.Passed ||
		rejectedContract.WriterContractID != "" ||
		rejectedContract.ContractState != "blocked" ||
		rejectedContract.ContractAction != "reject" ||
		rejectedContract.ContractShapeReady ||
		rejectedContract.ContractsReady ||
		rejectedContract.WriteAllowed ||
		rejectedContract.MutatesState ||
		rejectedContract.Reason != "candidate_admission_writer_inventory_failed: candidate_admission_writer_preflight_failed: candidate_admission_live_stage_failed: candidate_admission_enable_gate_failed: candidate_admission_switch_failed: candidate_admission_promotion_failed: candidate_admission_decision_failed: candidate execution runner provided_text is not nano-direct" {
		t.Fatalf("rejected inventory should not produce a writer contract: %+v", rejectedContract)
	}
	rejectedLedger := admissionLiveRouteTurnCandidateAdmissionLedgerForWriterContract(rejectedContract)
	if rejectedLedger.Passed ||
		rejectedLedger.AdmissionLedgerID != "" ||
		rejectedLedger.LedgerState != "blocked" ||
		rejectedLedger.LedgerAction != "reject" ||
		rejectedLedger.LedgerAppendReady ||
		rejectedLedger.ContractsReady ||
		rejectedLedger.WriteAllowed ||
		rejectedLedger.MutatesState ||
		rejectedLedger.Reason != "candidate_admission_writer_contract_failed: candidate_admission_writer_inventory_failed: candidate_admission_writer_preflight_failed: candidate_admission_live_stage_failed: candidate_admission_enable_gate_failed: candidate_admission_switch_failed: candidate_admission_promotion_failed: candidate_admission_decision_failed: candidate execution runner provided_text is not nano-direct" {
		t.Fatalf("rejected contract should not produce a ledger receipt: %+v", rejectedLedger)
	}
	rejectedWriterImpl := admissionLiveRouteTurnCandidateAdmissionWriterImplementationForLedger(rejectedLedger)
	if rejectedWriterImpl.Passed ||
		rejectedWriterImpl.WriterImplementationID != "" ||
		rejectedWriterImpl.ImplementationState != "blocked" ||
		rejectedWriterImpl.ImplementationAction != "reject" ||
		rejectedWriterImpl.AppendOnly ||
		rejectedWriterImpl.ImplementationContractReady ||
		rejectedWriterImpl.ContractsReady ||
		rejectedWriterImpl.WriteAllowed ||
		rejectedWriterImpl.MutatesState ||
		rejectedWriterImpl.Reason != "candidate_admission_ledger_failed: candidate_admission_writer_contract_failed: candidate_admission_writer_inventory_failed: candidate_admission_writer_preflight_failed: candidate_admission_live_stage_failed: candidate_admission_enable_gate_failed: candidate_admission_switch_failed: candidate_admission_promotion_failed: candidate_admission_decision_failed: candidate execution runner provided_text is not nano-direct" {
		t.Fatalf("rejected ledger should not produce a writer implementation receipt: %+v", rejectedWriterImpl)
	}
	rejectedWriterReceipt := admissionLiveRouteTurnCandidateAdmissionWriterReceiptForImplementation(rejectedWriterImpl)
	if rejectedWriterReceipt.Passed ||
		rejectedWriterReceipt.WriterReceiptID != "" ||
		rejectedWriterReceipt.WriterReceiptState != "blocked" ||
		rejectedWriterReceipt.WriterReceiptAction != "reject" ||
		rejectedWriterReceipt.WriterReceiptPersisted ||
		rejectedWriterReceipt.ShadowWriteAllowed ||
		rejectedWriterReceipt.WriterReady ||
		rejectedWriterReceipt.WriteAllowed ||
		rejectedWriterReceipt.MutatesState ||
		rejectedWriterReceipt.Reason != "candidate_admission_writer_implementation_failed: candidate_admission_ledger_failed: candidate_admission_writer_contract_failed: candidate_admission_writer_inventory_failed: candidate_admission_writer_preflight_failed: candidate_admission_live_stage_failed: candidate_admission_enable_gate_failed: candidate_admission_switch_failed: candidate_admission_promotion_failed: candidate_admission_decision_failed: candidate execution runner provided_text is not nano-direct" {
		t.Fatalf("rejected writer implementation should not produce a writer receipt: %+v", rejectedWriterReceipt)
	}

	tampered := decision
	tampered.DecisionID = "decision-tampered"
	tamperedPromotion := admissionLiveRouteTurnCandidateAdmissionPromotionForDecision(tampered)
	if tamperedPromotion.Passed ||
		tamperedPromotion.PromotionID != "" ||
		tamperedPromotion.Reason != "candidate_admission_decision_id_mismatch" {
		t.Fatalf("tampered decision id should fail closed: %+v", tamperedPromotion)
	}
	tamperedPromotionID := promotion
	tamperedPromotionID.PromotionID = "promotion-tampered"
	tamperedSwitch := admissionLiveRouteTurnCandidateAdmissionSwitchForPromotion(tamperedPromotionID)
	if tamperedSwitch.Passed ||
		tamperedSwitch.SwitchID != "" ||
		tamperedSwitch.Reason != "candidate_admission_promotion_id_mismatch" {
		t.Fatalf("tampered promotion id should fail closed: %+v", tamperedSwitch)
	}
	tamperedSwitchID := sw
	tamperedSwitchID.SwitchID = "switch-tampered"
	tamperedGate := admissionLiveRouteTurnCandidateAdmissionEnableGateForSwitch(tamperedSwitchID)
	if tamperedGate.Passed ||
		tamperedGate.EnableGateID != "" ||
		tamperedGate.Reason != "candidate_admission_switch_id_mismatch" {
		t.Fatalf("tampered switch id should fail closed: %+v", tamperedGate)
	}
	tamperedGateID := armedGate
	tamperedGateID.EnableGateID = "enable-tampered"
	tamperedStage := admissionLiveRouteTurnCandidateAdmissionLiveStageForEnableGate(tamperedGateID)
	if tamperedStage.Passed ||
		tamperedStage.LiveStageID != "" ||
		tamperedStage.Reason != "candidate_admission_enable_gate_id_mismatch" {
		t.Fatalf("tampered enable gate id should fail closed: %+v", tamperedStage)
	}
	tamperedStageID := liveStage
	tamperedStageID.LiveStageID = "stage-tampered"
	tamperedWriterPreflight := admissionLiveRouteTurnCandidateAdmissionWriterPreflightForLiveStage(tamperedStageID)
	if tamperedWriterPreflight.Passed ||
		tamperedWriterPreflight.WriterPreflightID != "" ||
		tamperedWriterPreflight.Reason != "candidate_admission_live_stage_id_mismatch" {
		t.Fatalf("tampered live stage id should fail writer preflight: %+v", tamperedWriterPreflight)
	}
	tamperedPreflightID := writerPreflight
	tamperedPreflightID.WriterPreflightID = "writer-tampered"
	tamperedInventory := admissionLiveRouteTurnCandidateAdmissionWriterInventoryForPreflight(tamperedPreflightID)
	if tamperedInventory.Passed ||
		tamperedInventory.WriterInventoryID != "" ||
		tamperedInventory.Reason != "candidate_admission_writer_preflight_id_mismatch" {
		t.Fatalf("tampered writer preflight id should fail inventory: %+v", tamperedInventory)
	}
	tamperedInventoryID := writerInventory
	tamperedInventoryID.WriterInventoryID = "writer-inventory-tampered"
	tamperedContract := admissionLiveRouteTurnCandidateAdmissionWriterContractForInventory(tamperedInventoryID)
	if tamperedContract.Passed ||
		tamperedContract.WriterContractID != "" ||
		tamperedContract.Reason != "candidate_admission_writer_inventory_id_mismatch" {
		t.Fatalf("tampered writer inventory id should fail contract: %+v", tamperedContract)
	}
	tamperedContractID := writerContract
	tamperedContractID.WriterContractID = "writer-contract-tampered"
	tamperedLedger := admissionLiveRouteTurnCandidateAdmissionLedgerForWriterContract(tamperedContractID)
	if tamperedLedger.Passed ||
		tamperedLedger.AdmissionLedgerID != "" ||
		tamperedLedger.Reason != "candidate_admission_writer_contract_id_mismatch" {
		t.Fatalf("tampered writer contract id should fail ledger: %+v", tamperedLedger)
	}
	tamperedLedgerID := ledger
	tamperedLedgerID.AdmissionLedgerID = "admission-ledger-tampered"
	tamperedWriterImpl := admissionLiveRouteTurnCandidateAdmissionWriterImplementationForLedger(tamperedLedgerID)
	if tamperedWriterImpl.Passed ||
		tamperedWriterImpl.WriterImplementationID != "" ||
		tamperedWriterImpl.Reason != "candidate_admission_ledger_id_mismatch" {
		t.Fatalf("tampered admission ledger id should fail writer implementation: %+v", tamperedWriterImpl)
	}
	tamperedWriterImplID := writerImpl
	tamperedWriterImplID.WriterImplementationID = "writer-implementation-tampered"
	tamperedWriterReceipt := admissionLiveRouteTurnCandidateAdmissionWriterReceiptForImplementation(tamperedWriterImplID)
	if tamperedWriterReceipt.Passed ||
		tamperedWriterReceipt.WriterReceiptID != "" ||
		tamperedWriterReceipt.Reason != "candidate_admission_writer_implementation_id_mismatch" {
		t.Fatalf("tampered writer implementation id should fail writer receipt: %+v", tamperedWriterReceipt)
	}
	tamperedReceiptID := writerReceipt
	tamperedReceiptID.WriterReceiptID = "writer-receipt-tampered"
	tamperedRollbackImpl := admissionLiveRouteTurnCandidateAdmissionRollbackImplementationForWriterReceipt(tamperedReceiptID)
	if tamperedRollbackImpl.Passed ||
		tamperedRollbackImpl.RollbackImplementationID != "" ||
		tamperedRollbackImpl.Reason != "candidate_admission_writer_receipt_id_mismatch" {
		t.Fatalf("tampered writer receipt id should fail rollback implementation: %+v", tamperedRollbackImpl)
	}
	tamperedRollbackImplID := rollbackImpl
	tamperedRollbackImplID.RollbackImplementationID = "rollback-implementation-tampered"
	tamperedLedgerImpl := admissionLiveRouteTurnCandidateAdmissionLedgerImplementationForRollbackImplementation(tamperedRollbackImplID)
	if tamperedLedgerImpl.Passed ||
		tamperedLedgerImpl.LedgerImplementationID != "" ||
		tamperedLedgerImpl.Reason != "candidate_admission_rollback_implementation_id_mismatch" {
		t.Fatalf("tampered rollback implementation id should fail ledger implementation: %+v", tamperedLedgerImpl)
	}
	tamperedLedgerImplID := ledgerImpl
	tamperedLedgerImplID.LedgerImplementationID = "ledger-implementation-tampered"
	tamperedLedgerPersistence := admissionLiveRouteTurnCandidateAdmissionLedgerPersistenceForLedgerImplementation(tamperedLedgerImplID)
	if tamperedLedgerPersistence.Passed ||
		tamperedLedgerPersistence.LedgerPersistenceID != "" ||
		tamperedLedgerPersistence.Reason != "candidate_admission_ledger_implementation_id_mismatch" {
		t.Fatalf("tampered ledger implementation id should fail ledger persistence: %+v", tamperedLedgerPersistence)
	}
	tamperedLedgerPersistenceID := ledgerPersistence
	tamperedLedgerPersistenceID.LedgerPersistenceID = "ledger-persistence-tampered"
	tamperedLedgerVerification := admissionLiveRouteTurnCandidateAdmissionLedgerVerificationForLedgerPersistence(tamperedLedgerPersistenceID)
	if tamperedLedgerVerification.Passed ||
		tamperedLedgerVerification.LedgerVerificationID != "" ||
		tamperedLedgerVerification.Reason != "candidate_admission_ledger_persistence_id_mismatch" {
		t.Fatalf("tampered ledger persistence id should fail ledger verification: %+v", tamperedLedgerVerification)
	}
	tamperedLedgerVerificationID := ledgerVerification
	tamperedLedgerVerificationID.LedgerVerificationID = "ledger-verification-tampered"
	tamperedReadiness := admissionLiveRouteTurnCandidateAdmissionReadinessForLedgerVerification(tamperedLedgerVerificationID)
	if tamperedReadiness.Passed ||
		tamperedReadiness.AdmissionReadinessID != "" ||
		tamperedReadiness.Reason != "candidate_admission_ledger_verification_id_mismatch" {
		t.Fatalf("tampered ledger verification id should fail admission readiness: %+v", tamperedReadiness)
	}
	t.Setenv("AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_PERMIT_KEY", admissionLiveRouteTurnCandidateAdmissionPermitConfirmation)
	tamperedReadinessID := readiness
	tamperedReadinessID.AdmissionReadinessID = "admission-readiness-tampered"
	tamperedReadinessPermit := admissionLiveRouteTurnCandidateAdmissionPermitForReadiness(tamperedReadinessID)
	if tamperedReadinessPermit.Passed ||
		tamperedReadinessPermit.AdmissionPermitID != "" ||
		tamperedReadinessPermit.Reason != "candidate_admission_readiness_id_mismatch" {
		t.Fatalf("tampered readiness id should fail admission permit: %+v", tamperedReadinessPermit)
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
