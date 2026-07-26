package main

import (
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestChatLiveRouteChoiceDryRunLine(t *testing.T) {
	t.Setenv("AM_DREAM_ADMISSION_LIVE_ROUTE_CHOICE_DRY_RUN", "1")

	c := newDreamCandidate("direct", "identity", "seed", "", "I am Arianna.", nil)
	choice := admissionLiveRouteChoiceForCandidate(c)
	c.Admission = &dreamAdmissionPolicy{
		LiveRouteChoice:       &choice,
		LiveRouteChoiceDryRun: true,
	}

	line := chatLiveRouteChoiceDryRunLine(c)
	for _, want := range []string{
		"live-route dry-run",
		"class=identity",
		"route=chorus",
		"source=direct",
		"expected=chorus",
		"passed=false",
		"reason=source direct does not match live route chorus for prompt class identity",
	} {
		if !strings.Contains(line, want) {
			t.Fatalf("dry-run line missing %q: %q", want, line)
		}
	}
}

func TestChatLiveRouteChoiceDryRunLineDisabled(t *testing.T) {
	c := newDreamCandidate("chorus", "identity", "seed", "", "I am Arianna.", nil)
	choice := admissionLiveRouteChoiceForCandidate(c)
	c.Admission = &dreamAdmissionPolicy{LiveRouteChoice: &choice}
	if got := chatLiveRouteChoiceDryRunLine(c); got != "" {
		t.Fatalf("dry-run line should be hidden by default: %q", got)
	}
}

func TestChatLiveRouteTurnDryRunLine(t *testing.T) {
	t.Setenv("AM_DREAM_ADMISSION_LIVE_ROUTE_CHOICE_DRY_RUN", "1")

	obs := admissionLiveRouteTurnObservationForHuman("Who are you?")
	line := chatLiveRouteTurnDryRunLine(obs)
	for _, want := range []string{
		"live-route turn dry-run",
		"class=identity",
		"route=chorus",
		"expected=chorus",
		"passed=true",
		"score=3",
	} {
		if !strings.Contains(line, want) {
			t.Fatalf("turn dry-run line missing %q: %q", want, line)
		}
	}
}

func TestChatLiveRouteTurnDryRunLineDisabled(t *testing.T) {
	obs := admissionLiveRouteTurnObservationForHuman("Who are you?")
	if got := chatLiveRouteTurnDryRunLine(obs); got != "" {
		t.Fatalf("turn dry-run line should be hidden by default: %q", got)
	}
}

func TestAdmissionLiveRouteTurnObservationDryRunNeededIncludesAdmissionChain(t *testing.T) {
	if admissionLiveRouteTurnObservationDryRunNeeded() {
		t.Fatal("turn observation should be disabled by default")
	}
	t.Setenv("AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_ADAPTER_DRY_RUN", "1")
	if !admissionLiveRouteTurnObservationDryRunNeeded() {
		t.Fatal("candidate admission adapter dry-run must request turn observation")
	}
}

func TestAdmissionLiveRouteTurnObservationDryRunNeededIncludesExecution(t *testing.T) {
	if admissionLiveRouteTurnObservationDryRunNeeded() {
		t.Fatal("turn observation should be disabled by default")
	}
	t.Setenv("AM_LIVE_ROUTE_TURN_CANDIDATE_EXECUTION_DRY_RUN", "1")
	if !admissionLiveRouteTurnObservationDryRunNeeded() {
		t.Fatal("candidate execution dry-run must request turn observation")
	}
}

func TestChatLiveRouteTurnRequestDryRunLine(t *testing.T) {
	t.Setenv("AM_LIVE_ROUTE_TURN_REQUEST_DRY_RUN", "1")

	obs := admissionLiveRouteTurnObservationForHuman("Who are you?")
	line := chatLiveRouteTurnRequestDryRunLine(obs)
	for _, want := range []string{
		"live-route turn request dry-run",
		"class=identity",
		"route=chorus",
		"source=chorus",
		"trigger=chorus-identity",
		"seed=turn-",
		"passed=true",
	} {
		if !strings.Contains(line, want) {
			t.Fatalf("turn request dry-run line missing %q: %q", want, line)
		}
	}
}

func TestChatLiveRouteTurnRequestDryRunLineDisabled(t *testing.T) {
	obs := admissionLiveRouteTurnObservationForHuman("Who are you?")
	if got := chatLiveRouteTurnRequestDryRunLine(obs); got != "" {
		t.Fatalf("turn request dry-run line should be hidden by default: %q", got)
	}
}

func TestChatLiveRouteTurnGenerationJobDryRunLine(t *testing.T) {
	t.Setenv("AM_LIVE_ROUTE_TURN_GENERATION_JOB_DRY_RUN", "1")

	obs := admissionLiveRouteTurnObservationForHuman("Who are you?")
	line := chatLiveRouteTurnGenerationJobDryRunLine(obs)
	for _, want := range []string{
		"live-route generation job dry-run",
		"class=identity",
		"route=chorus",
		"backend=chorus-arianna",
		"entry=field",
		"trigger=chorus-identity",
		"seed=turn-",
		"job=job-",
		"passed=true",
	} {
		if !strings.Contains(line, want) {
			t.Fatalf("generation job dry-run line missing %q: %q", want, line)
		}
	}
}

func TestChatLiveRouteTurnGenerationJobDryRunLineDisabled(t *testing.T) {
	obs := admissionLiveRouteTurnObservationForHuman("Who are you?")
	if got := chatLiveRouteTurnGenerationJobDryRunLine(obs); got != "" {
		t.Fatalf("generation job dry-run line should be hidden by default: %q", got)
	}
}

func TestChatLiveRouteTurnCandidateShellDryRunLine(t *testing.T) {
	t.Setenv("AM_LIVE_ROUTE_TURN_CANDIDATE_SHELL_DRY_RUN", "1")

	obs := admissionLiveRouteTurnObservationForHuman("Who are you?")
	line := chatLiveRouteTurnCandidateShellDryRunLine(obs)
	for _, want := range []string{
		"live-route candidate shell dry-run",
		"class=identity",
		"route=chorus",
		"source=chorus",
		"trigger=chorus-identity",
		"seed=turn-",
		"job=job-",
		"shell=shell-",
		"text=pending_generation",
		"passed=true",
	} {
		if !strings.Contains(line, want) {
			t.Fatalf("candidate shell dry-run line missing %q: %q", want, line)
		}
	}
}

func TestChatLiveRouteTurnCandidateShellDryRunLineDisabled(t *testing.T) {
	obs := admissionLiveRouteTurnObservationForHuman("Who are you?")
	if got := chatLiveRouteTurnCandidateShellDryRunLine(obs); got != "" {
		t.Fatalf("candidate shell dry-run line should be hidden by default: %q", got)
	}
}

func TestChatLiveRouteTurnCandidateExecutionDryRunLine(t *testing.T) {
	t.Setenv("AM_LIVE_ROUTE_TURN_CANDIDATE_EXECUTION_DRY_RUN", "1")
	t.Setenv("AM_LIVE_ROUTE_TURN_CANDIDATE_EXECUTION_TIMEOUT_MS", "14000")
	t.Setenv("AM_LIVE_ROUTE_TURN_CANDIDATE_EXECUTION_TEXT", "I am Arianna, and execution leaves a receipt.")

	obs := admissionLiveRouteTurnObservationForHuman("Who are you?")
	line := chatLiveRouteTurnCandidateExecutionDryRunLine(obs)
	for _, want := range []string{
		"live-route candidate execution dry-run",
		"class=identity",
		"route=chorus",
		"backend=chorus-arianna",
		"entry=field",
		"frame=q_a",
		"executor=chorus-arianna:field:q_a",
		"timeout_ms=14000",
		"shell=shell-",
		"execution=execution-",
		"text=generated",
		"passed=true",
	} {
		if !strings.Contains(line, want) {
			t.Fatalf("candidate execution dry-run line missing %q: %q", want, line)
		}
	}
}

func TestChatLiveRouteTurnCandidateExecutionDryRunLineMissingText(t *testing.T) {
	t.Setenv("AM_LIVE_ROUTE_TURN_CANDIDATE_EXECUTION_DRY_RUN", "1")

	obs := admissionLiveRouteTurnObservationForHuman("Who are you?")
	line := chatLiveRouteTurnCandidateExecutionDryRunLine(obs)
	for _, want := range []string{
		"live-route candidate execution dry-run",
		"class=identity",
		"shell=shell-",
		"execution=",
		"text=pending_generation",
		"passed=false",
		"reason=missing generated text for shell shell-",
	} {
		if !strings.Contains(line, want) {
			t.Fatalf("candidate execution missing-text line missing %q: %q", want, line)
		}
	}
	if strings.Contains(line, "execution=execution-") {
		t.Fatalf("missing-text execution line should not name execution id: %q", line)
	}
}

func TestChatLiveRouteTurnCandidateExecutionDryRunLineDisabled(t *testing.T) {
	obs := admissionLiveRouteTurnObservationForHuman("Who are you?")
	if got := chatLiveRouteTurnCandidateExecutionDryRunLine(obs); got != "" {
		t.Fatalf("candidate execution dry-run line should be hidden by default: %q", got)
	}
}

func TestChatLiveRouteTurnGeneratorAdapterDryRunLine(t *testing.T) {
	t.Setenv("AM_LIVE_ROUTE_TURN_GENERATOR_ADAPTER_DRY_RUN", "1")
	t.Setenv("AM_LIVE_ROUTE_TURN_GENERATOR_ADAPTER_TEXT", "I am Arianna, and the generator cannot rename the shell.")

	obs := admissionLiveRouteTurnObservationForHuman("Who are you?")
	line := chatLiveRouteTurnGeneratorAdapterDryRunLine(obs)
	for _, want := range []string{
		"live-route generator adapter dry-run",
		"class=identity",
		"route=chorus",
		"backend=chorus-arianna",
		"entry=field",
		"frame=q_a",
		"shell=shell-",
		"adapter=adapter-",
		"text=generated",
		"passed=true",
	} {
		if !strings.Contains(line, want) {
			t.Fatalf("generator adapter dry-run line missing %q: %q", want, line)
		}
	}
}

func TestChatLiveRouteTurnGeneratorAdapterDryRunLineUsesExecution(t *testing.T) {
	t.Setenv("AM_LIVE_ROUTE_TURN_GENERATOR_ADAPTER_DRY_RUN", "1")
	t.Setenv("AM_LIVE_ROUTE_TURN_CANDIDATE_EXECUTION_DRY_RUN", "1")
	t.Setenv("AM_LIVE_ROUTE_TURN_CANDIDATE_EXECUTION_TEXT", "I am Arianna, and the adapter consumes execution.")

	obs := admissionLiveRouteTurnObservationForHuman("Who are you?")
	line := chatLiveRouteTurnGeneratorAdapterDryRunLine(obs)
	for _, want := range []string{
		"live-route generator adapter dry-run",
		"class=identity",
		"shell=shell-",
		"execution=execution-",
		"adapter=adapter-",
		"text=generated",
		"passed=true",
	} {
		if !strings.Contains(line, want) {
			t.Fatalf("execution-backed generator adapter line missing %q: %q", want, line)
		}
	}
}

func TestChatLiveRouteTurnGeneratorAdapterDryRunLineMissingText(t *testing.T) {
	t.Setenv("AM_LIVE_ROUTE_TURN_GENERATOR_ADAPTER_DRY_RUN", "1")

	obs := admissionLiveRouteTurnObservationForHuman("Who are you?")
	line := chatLiveRouteTurnGeneratorAdapterDryRunLine(obs)
	for _, want := range []string{
		"live-route generator adapter dry-run",
		"class=identity",
		"shell=shell-",
		"adapter=",
		"text=pending_generation",
		"passed=false",
		"reason=missing generated text for shell shell-",
	} {
		if !strings.Contains(line, want) {
			t.Fatalf("generator adapter missing-text line missing %q: %q", want, line)
		}
	}
	if strings.Contains(line, "adapter=adapter-") {
		t.Fatalf("missing-text adapter line should not name adapter id: %q", line)
	}
}

func TestChatLiveRouteTurnGeneratorAdapterDryRunLineDisabled(t *testing.T) {
	obs := admissionLiveRouteTurnObservationForHuman("Who are you?")
	if got := chatLiveRouteTurnGeneratorAdapterDryRunLine(obs); got != "" {
		t.Fatalf("generator adapter dry-run line should be hidden by default: %q", got)
	}
}

func TestChatLiveRouteTurnCandidateDraftDryRunLine(t *testing.T) {
	t.Setenv("AM_LIVE_ROUTE_TURN_CANDIDATE_DRAFT_DRY_RUN", "1")
	t.Setenv("AM_LIVE_ROUTE_TURN_CANDIDATE_DRAFT_TEXT", "I am Arianna, and the field keeps the route visible.")

	obs := admissionLiveRouteTurnObservationForHuman("Who are you?")
	line := chatLiveRouteTurnCandidateDraftDryRunLine(obs)
	for _, want := range []string{
		"live-route candidate draft dry-run",
		"class=identity",
		"route=chorus",
		"source=chorus",
		"trigger=chorus-identity",
		"seed=turn-",
		"shell=shell-",
		"adapter=adapter-",
		"draft=draft-",
		"run=",
		"text=generated",
		"passed=true",
	} {
		if !strings.Contains(line, want) {
			t.Fatalf("candidate draft dry-run line missing %q: %q", want, line)
		}
	}
}

func TestChatLiveRouteTurnCandidateDraftDryRunLineMissingText(t *testing.T) {
	t.Setenv("AM_LIVE_ROUTE_TURN_CANDIDATE_DRAFT_DRY_RUN", "1")

	obs := admissionLiveRouteTurnObservationForHuman("Who are you?")
	line := chatLiveRouteTurnCandidateDraftDryRunLine(obs)
	for _, want := range []string{
		"live-route candidate draft dry-run",
		"class=identity",
		"shell=shell-",
		"adapter=",
		"text=pending_generation",
		"passed=false",
		"reason=generator adapter failed: missing generated text for shell shell-",
	} {
		if !strings.Contains(line, want) {
			t.Fatalf("candidate draft missing-text line missing %q: %q", want, line)
		}
	}
}

func TestChatLiveRouteTurnCandidateDraftDryRunLineUsesExecution(t *testing.T) {
	t.Setenv("AM_LIVE_ROUTE_TURN_CANDIDATE_DRAFT_DRY_RUN", "1")
	t.Setenv("AM_LIVE_ROUTE_TURN_CANDIDATE_EXECUTION_DRY_RUN", "1")
	t.Setenv("AM_LIVE_ROUTE_TURN_CANDIDATE_DRAFT_TEXT", "I am Arianna, and the draft keeps execution visible.")

	obs := admissionLiveRouteTurnObservationForHuman("Who are you?")
	line := chatLiveRouteTurnCandidateDraftDryRunLine(obs)
	for _, want := range []string{
		"live-route candidate draft dry-run",
		"class=identity",
		"shell=shell-",
		"execution=execution-",
		"adapter=adapter-",
		"draft=draft-",
		"text=generated",
		"passed=true",
	} {
		if !strings.Contains(line, want) {
			t.Fatalf("execution-backed candidate draft line missing %q: %q", want, line)
		}
	}
}

func TestChatLiveRouteTurnCandidateDraftDryRunLineDisabled(t *testing.T) {
	obs := admissionLiveRouteTurnObservationForHuman("Who are you?")
	if got := chatLiveRouteTurnCandidateDraftDryRunLine(obs); got != "" {
		t.Fatalf("candidate draft dry-run line should be hidden by default: %q", got)
	}
}

func TestChatLiveRouteTurnCandidateAdmissionDryRunLine(t *testing.T) {
	t.Setenv("AM_LIVE_ROUTE_TURN_CANDIDATE_DRAFT_DRY_RUN", "1")
	t.Setenv("AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_DRY_RUN", "1")
	t.Setenv("AM_LIVE_ROUTE_TURN_CANDIDATE_DRAFT_TEXT", "I am Arianna, and the handoff keeps the draft named.")

	obs := admissionLiveRouteTurnObservationForHuman("Who are you?")
	line := chatLiveRouteTurnCandidateAdmissionDryRunLine(obs)
	for _, want := range []string{
		"live-route candidate admission handoff dry-run",
		"class=identity",
		"route=chorus",
		"source=chorus",
		"draft=draft-",
		"adapter=adapter-",
		"handoff=handoff-",
		"review=true",
		"passed=true",
	} {
		if !strings.Contains(line, want) {
			t.Fatalf("candidate admission handoff line missing %q: %q", want, line)
		}
	}
}

func TestChatLiveRouteTurnCandidateAdmissionDryRunLineMissingText(t *testing.T) {
	t.Setenv("AM_LIVE_ROUTE_TURN_CANDIDATE_DRAFT_DRY_RUN", "1")
	t.Setenv("AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_DRY_RUN", "1")

	obs := admissionLiveRouteTurnObservationForHuman("Who are you?")
	line := chatLiveRouteTurnCandidateAdmissionDryRunLine(obs)
	for _, want := range []string{
		"live-route candidate admission handoff dry-run",
		"class=identity",
		"draft=",
		"adapter=",
		"handoff=",
		"review=false",
		"passed=false",
		"reason=candidate_draft_failed: generator adapter failed: missing generated text for shell shell-",
	} {
		if !strings.Contains(line, want) {
			t.Fatalf("candidate admission missing-text line missing %q: %q", want, line)
		}
	}
}

func TestChatLiveRouteTurnCandidateAdmissionDryRunLineDisabled(t *testing.T) {
	t.Setenv("AM_LIVE_ROUTE_TURN_CANDIDATE_DRAFT_DRY_RUN", "1")
	obs := admissionLiveRouteTurnObservationForHuman("Who are you?")
	if got := chatLiveRouteTurnCandidateAdmissionDryRunLine(obs); got != "" {
		t.Fatalf("candidate admission handoff line should be hidden by default: %q", got)
	}
}

func TestChatLiveRouteTurnCandidateAdmissionAdapterDryRunLine(t *testing.T) {
	t.Setenv("AM_LIVE_ROUTE_TURN_CANDIDATE_DRAFT_DRY_RUN", "1")
	t.Setenv("AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_DRY_RUN", "1")
	t.Setenv("AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_ADAPTER_DRY_RUN", "1")
	t.Setenv("AM_LIVE_ROUTE_TURN_CANDIDATE_DRAFT_TEXT", "I am Arianna, and the admission adapter keeps provenance intact.")

	obs := admissionLiveRouteTurnObservationForHuman("Who are you?")
	line := chatLiveRouteTurnCandidateAdmissionAdapterDryRunLine(obs)
	for _, want := range []string{
		"live-route candidate admission adapter dry-run",
		"class=identity",
		"route=chorus",
		"source=chorus",
		"handoff=handoff-",
		"admission_adapter=admission-adapter-",
		"run=",
		"passed=true",
	} {
		if !strings.Contains(line, want) {
			t.Fatalf("candidate admission adapter line missing %q: %q", want, line)
		}
	}
}

func TestChatLiveRouteTurnCandidateAdmissionAdapterDryRunLineMissingText(t *testing.T) {
	t.Setenv("AM_LIVE_ROUTE_TURN_CANDIDATE_DRAFT_DRY_RUN", "1")
	t.Setenv("AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_DRY_RUN", "1")
	t.Setenv("AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_ADAPTER_DRY_RUN", "1")

	obs := admissionLiveRouteTurnObservationForHuman("Who are you?")
	line := chatLiveRouteTurnCandidateAdmissionAdapterDryRunLine(obs)
	for _, want := range []string{
		"live-route candidate admission adapter dry-run",
		"class=identity",
		"handoff=",
		"admission_adapter=",
		"run=",
		"passed=false",
		"reason=candidate_admission_handoff_failed: candidate_draft_failed: generator adapter failed: missing generated text for shell shell-",
	} {
		if !strings.Contains(line, want) {
			t.Fatalf("candidate admission adapter missing-text line missing %q: %q", want, line)
		}
	}
}

func TestChatLiveRouteTurnCandidateAdmissionAdapterDryRunLineDisabled(t *testing.T) {
	t.Setenv("AM_LIVE_ROUTE_TURN_CANDIDATE_DRAFT_DRY_RUN", "1")
	t.Setenv("AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_DRY_RUN", "1")
	obs := admissionLiveRouteTurnObservationForHuman("Who are you?")
	if got := chatLiveRouteTurnCandidateAdmissionAdapterDryRunLine(obs); got != "" {
		t.Fatalf("candidate admission adapter line should be hidden by default: %q", got)
	}
}

func TestChatLiveRouteTurnCandidateAdmissionShadowDryRunLine(t *testing.T) {
	t.Setenv("AM_LIVE_ROUTE_TURN_CANDIDATE_DRAFT_DRY_RUN", "1")
	t.Setenv("AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_DRY_RUN", "1")
	t.Setenv("AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_ADAPTER_DRY_RUN", "1")
	t.Setenv("AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_SHADOW_DRY_RUN", "1")
	t.Setenv("AM_DREAM_ADMISSION", dreamAdmissionShadow)
	t.Setenv("AM_DREAM_ADMISSION_REQUIRE_LIVE_ROUTE_PLAN", "1")
	t.Setenv("AM_LIVE_ROUTE_TURN_CANDIDATE_DRAFT_TEXT", "I am Arianna, and the shadow gate sees the same handoff.")
	logPath := filepath.Join(t.TempDir(), "dream-admission-chat-shadow.jsonl")
	t.Setenv("AM_DREAM_ADMISSION_LOG", logPath)

	obs := admissionLiveRouteTurnObservationForHuman("Who are you?")
	line := chatLiveRouteTurnCandidateAdmissionShadowDryRunLine(obs)
	for _, want := range []string{
		"live-route candidate admission shadow dry-run",
		"class=identity",
		"route=chorus",
		"source=chorus",
		"handoff=handoff-",
		"admission_adapter=admission-adapter-",
		"run=",
		"policy=true",
		"accepted=false",
		"passed=true",
		"reason=shadow mode",
	} {
		if !strings.Contains(line, want) {
			t.Fatalf("candidate admission shadow line missing %q: %q", want, line)
		}
	}
	raw, err := os.ReadFile(logPath)
	if err != nil {
		t.Fatal(err)
	}
	var got dreamCandidate
	if err := json.Unmarshal([]byte(strings.TrimSpace(string(raw))), &got); err != nil {
		t.Fatal(err)
	}
	if got.LiveRouteCandidateAdmission == nil ||
		got.LiveRouteCandidateAdmission.AdmissionAdapterID == "" ||
		got.Admission == nil ||
		!got.Admission.Passed ||
		got.Accepted ||
		got.Reason != "shadow mode" {
		t.Fatalf("bad shadow admission receipt: %+v", got)
	}
}

func TestChatLiveRouteTurnCandidateAdmissionShadowDryRunLineRequiresShadowMode(t *testing.T) {
	t.Setenv("AM_LIVE_ROUTE_TURN_CANDIDATE_DRAFT_DRY_RUN", "1")
	t.Setenv("AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_DRY_RUN", "1")
	t.Setenv("AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_ADAPTER_DRY_RUN", "1")
	t.Setenv("AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_SHADOW_DRY_RUN", "1")
	t.Setenv("AM_DREAM_ADMISSION_REQUIRE_LIVE_ROUTE_PLAN", "1")
	t.Setenv("AM_LIVE_ROUTE_TURN_CANDIDATE_DRAFT_TEXT", "I am Arianna, but this gate must stay shadow-only.")

	obs := admissionLiveRouteTurnObservationForHuman("Who are you?")
	line := chatLiveRouteTurnCandidateAdmissionShadowDryRunLine(obs)
	for _, want := range []string{
		"live-route candidate admission shadow dry-run",
		"policy=false",
		"accepted=false",
		"passed=false",
		"reason=AM_DREAM_ADMISSION must be shadow",
	} {
		if !strings.Contains(line, want) {
			t.Fatalf("candidate admission shadow mode guard missing %q: %q", want, line)
		}
	}
}

func TestChatLiveRouteTurnCandidateAdmissionShadowDryRunLineDisabled(t *testing.T) {
	t.Setenv("AM_LIVE_ROUTE_TURN_CANDIDATE_DRAFT_DRY_RUN", "1")
	t.Setenv("AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_DRY_RUN", "1")
	t.Setenv("AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_ADAPTER_DRY_RUN", "1")
	obs := admissionLiveRouteTurnObservationForHuman("Who are you?")
	if got := chatLiveRouteTurnCandidateAdmissionShadowDryRunLine(obs); got != "" {
		t.Fatalf("candidate admission shadow line should be hidden by default: %q", got)
	}
}

func TestChatLiveRouteTurnCandidateReviewLine(t *testing.T) {
	t.Setenv("AM_DREAM_ADMISSION_LIVE_ROUTE_CHOICE_DRY_RUN", "1")

	obs := admissionLiveRouteTurnObservationForHuman("Who are you?")
	c := newDreamCandidate("chorus", "chorus-identity", "seed", "", "I am Arianna.", nil)
	line := chatLiveRouteTurnCandidateReviewLine(obs, c)
	for _, want := range []string{
		"live-route turn/candidate review",
		"turn_class=identity",
		"expected=chorus",
		"candidate_source=chorus",
		"candidate_class=identity",
		"candidate_route=chorus",
		"matched=true",
	} {
		if !strings.Contains(line, want) {
			t.Fatalf("turn/candidate review line missing %q: %q", want, line)
		}
	}
}

func TestChatLiveRouteTurnBridgeCandidateReviewLine(t *testing.T) {
	t.Setenv("AM_DREAM_ADMISSION_LIVE_ROUTE_CHOICE_DRY_RUN", "1")
	t.Setenv("AM_LIVE_ROUTE_TURN_BRIDGE_DRY_RUN", "1")

	obs := admissionLiveRouteTurnObservationForHuman("Who are you?")
	c := newDreamCandidate("nano", "human-turn", "seed", "", "I am Arianna.", nil)
	line := chatLiveRouteTurnCandidateReviewLine(obs, c)
	for _, want := range []string{
		"live-route turn/candidate review",
		"turn_class=identity",
		"expected=chorus",
		"candidate_source=nano",
		"candidate_class=identity",
		"candidate_route=chorus",
		"matched=false",
		"bridge=human-turn-identity",
		"reason=candidate_route_failed: source nano does not match live route chorus for prompt class identity",
	} {
		if !strings.Contains(line, want) {
			t.Fatalf("bridged turn/candidate review line missing %q: %q", want, line)
		}
	}
}

func TestChatLiveRouteTurnCandidateReviewLineDisabled(t *testing.T) {
	obs := admissionLiveRouteTurnObservationForHuman("Who are you?")
	c := newDreamCandidate("chorus", "chorus-identity", "seed", "", "I am Arianna.", nil)
	if got := chatLiveRouteTurnCandidateReviewLine(obs, c); got != "" {
		t.Fatalf("turn/candidate review line should be hidden by default: %q", got)
	}
}
