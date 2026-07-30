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
		"runner=provided_text",
		"runner_status=provided",
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
		"runner=provided_text",
		"runner_status=provided",
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

func TestChatLiveRouteTurnCandidateAdmissionDecisionDryRunLine(t *testing.T) {
	t.Setenv("AM_LIVE_ROUTE_TURN_GENERATOR_ADAPTER_DRY_RUN", "1")
	t.Setenv("AM_LIVE_ROUTE_TURN_GENERATOR_ADAPTER_TEXT", "The dream remembers the field through one chain.")
	t.Setenv("AM_LIVE_ROUTE_TURN_CANDIDATE_DRAFT_DRY_RUN", "1")
	t.Setenv("AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_DRY_RUN", "1")
	t.Setenv("AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_ADAPTER_DRY_RUN", "1")
	t.Setenv("AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_SHADOW_DRY_RUN", "1")
	t.Setenv("AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_DECISION_DRY_RUN", "1")
	t.Setenv("AM_DREAM_ADMISSION", dreamAdmissionShadow)
	t.Setenv("AM_DREAM_ADMISSION_REQUIRE_LIVE_ROUTE_PLAN", "1")
	dir := t.TempDir()
	dreamLog := filepath.Join(dir, "dream-admission-chat-decision.jsonl")
	decisionLog := filepath.Join(dir, "live-route-candidate-admission-decision.jsonl")
	t.Setenv("AM_DREAM_ADMISSION_LOG", dreamLog)
	t.Setenv("AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_DECISION_LOG", decisionLog)

	obs := admissionLiveRouteTurnObservationForHuman("Tell me what the dream should remember.")
	lines := chatLiveRouteTurnCandidateChainDryRunLines(obs)
	if len(lines) != 6 {
		t.Fatalf("expected 6 candidate chain lines, got %d: %v", len(lines), lines)
	}
	decisionLine := lines[len(lines)-1]
	for _, want := range []string{
		"live-route candidate admission decision dry-run",
		"class=dream",
		"route=direct",
		"source=direct",
		"handoff=handoff-",
		"admission_adapter=admission-adapter-",
		"decision=reject",
		"live_ready=false",
		"mutates=false",
		"passed=false",
		"reason=missing_candidate_execution",
	} {
		if !strings.Contains(decisionLine, want) {
			t.Fatalf("candidate admission decision line missing %q: %q", want, decisionLine)
		}
	}
	raw, err := os.ReadFile(decisionLog)
	if err != nil {
		t.Fatal(err)
	}
	var got admissionLiveRouteTurnCandidateAdmissionDecision
	if err := json.Unmarshal([]byte(strings.TrimSpace(string(raw))), &got); err != nil {
		t.Fatal(err)
	}
	if got.Schema != admissionLiveRouteTurnCandidateAdmissionDecisionSchema ||
		got.Passed ||
		got.LiveReady ||
		got.MutatesState ||
		got.DecisionID != "" ||
		got.Reason != "missing_candidate_execution" {
		t.Fatalf("bad candidate admission decision receipt: %+v", got)
	}
}

func TestChatLiveRouteTurnCandidateAdmissionPromotionDryRunLine(t *testing.T) {
	t.Setenv("AM_LIVE_ROUTE_TURN_GENERATOR_ADAPTER_DRY_RUN", "1")
	t.Setenv("AM_LIVE_ROUTE_TURN_GENERATOR_ADAPTER_TEXT", "The dream remembers the field through one chain.")
	t.Setenv("AM_LIVE_ROUTE_TURN_CANDIDATE_DRAFT_DRY_RUN", "1")
	t.Setenv("AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_DRY_RUN", "1")
	t.Setenv("AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_ADAPTER_DRY_RUN", "1")
	t.Setenv("AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_SHADOW_DRY_RUN", "1")
	t.Setenv("AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_DECISION_DRY_RUN", "1")
	t.Setenv("AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_PROMOTION_DRY_RUN", "1")
	t.Setenv("AM_DREAM_ADMISSION", dreamAdmissionShadow)
	t.Setenv("AM_DREAM_ADMISSION_REQUIRE_LIVE_ROUTE_PLAN", "1")
	dir := t.TempDir()
	dreamLog := filepath.Join(dir, "dream-admission-chat-promotion.jsonl")
	decisionLog := filepath.Join(dir, "live-route-candidate-admission-decision.jsonl")
	promotionLog := filepath.Join(dir, "live-route-candidate-admission-promotion.jsonl")
	t.Setenv("AM_DREAM_ADMISSION_LOG", dreamLog)
	t.Setenv("AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_DECISION_LOG", decisionLog)
	t.Setenv("AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_PROMOTION_LOG", promotionLog)

	obs := admissionLiveRouteTurnObservationForHuman("Tell me what the dream should remember.")
	lines := chatLiveRouteTurnCandidateChainDryRunLines(obs)
	if len(lines) != 7 {
		t.Fatalf("expected 7 candidate chain lines, got %d: %v", len(lines), lines)
	}
	promotionLine := lines[len(lines)-1]
	for _, want := range []string{
		"live-route candidate admission promotion dry-run",
		"class=dream",
		"route=direct",
		"source=direct",
		"decision=reject",
		"promotion=blocked",
		"live_ready=false",
		"live_enabled=false",
		"mutates=false",
		"passed=false",
		"reason=candidate_admission_decision_failed: missing_candidate_execution",
	} {
		if !strings.Contains(promotionLine, want) {
			t.Fatalf("candidate admission promotion line missing %q: %q", want, promotionLine)
		}
	}
	raw, err := os.ReadFile(promotionLog)
	if err != nil {
		t.Fatal(err)
	}
	var got admissionLiveRouteTurnCandidateAdmissionPromotion
	if err := json.Unmarshal([]byte(strings.TrimSpace(string(raw))), &got); err != nil {
		t.Fatal(err)
	}
	if got.Schema != admissionLiveRouteTurnCandidateAdmissionPromotionSchema ||
		got.Passed ||
		got.LiveReady ||
		got.LiveAdmissionEnabled ||
		got.MutatesState ||
		got.PromotionID != "" ||
		got.Reason != "candidate_admission_decision_failed: missing_candidate_execution" {
		t.Fatalf("bad candidate admission promotion receipt: %+v", got)
	}
}

func TestChatLiveRouteTurnCandidateAdmissionSwitchDryRunLine(t *testing.T) {
	t.Setenv("AM_LIVE_ROUTE_TURN_GENERATOR_ADAPTER_DRY_RUN", "1")
	t.Setenv("AM_LIVE_ROUTE_TURN_GENERATOR_ADAPTER_TEXT", "The dream remembers the field through one chain.")
	t.Setenv("AM_LIVE_ROUTE_TURN_CANDIDATE_DRAFT_DRY_RUN", "1")
	t.Setenv("AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_DRY_RUN", "1")
	t.Setenv("AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_ADAPTER_DRY_RUN", "1")
	t.Setenv("AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_SHADOW_DRY_RUN", "1")
	t.Setenv("AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_DECISION_DRY_RUN", "1")
	t.Setenv("AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_PROMOTION_DRY_RUN", "1")
	t.Setenv("AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_SWITCH_DRY_RUN", "1")
	t.Setenv("AM_DREAM_ADMISSION", dreamAdmissionShadow)
	t.Setenv("AM_DREAM_ADMISSION_REQUIRE_LIVE_ROUTE_PLAN", "1")
	dir := t.TempDir()
	dreamLog := filepath.Join(dir, "dream-admission-chat-switch.jsonl")
	decisionLog := filepath.Join(dir, "live-route-candidate-admission-decision.jsonl")
	promotionLog := filepath.Join(dir, "live-route-candidate-admission-promotion.jsonl")
	switchLog := filepath.Join(dir, "live-route-candidate-admission-switch.jsonl")
	t.Setenv("AM_DREAM_ADMISSION_LOG", dreamLog)
	t.Setenv("AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_DECISION_LOG", decisionLog)
	t.Setenv("AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_PROMOTION_LOG", promotionLog)
	t.Setenv("AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_SWITCH_LOG", switchLog)

	obs := admissionLiveRouteTurnObservationForHuman("Tell me what the dream should remember.")
	lines := chatLiveRouteTurnCandidateChainDryRunLines(obs)
	if len(lines) != 8 {
		t.Fatalf("expected 8 candidate chain lines, got %d: %v", len(lines), lines)
	}
	switchLine := lines[len(lines)-1]
	for _, want := range []string{
		"live-route candidate admission switch dry-run",
		"class=dream",
		"route=direct",
		"source=direct",
		"promotion=blocked",
		"switch=blocked",
		"switch_action=reject",
		"admission_allowed=false",
		"live_ready=false",
		"live_enabled=false",
		"mutates=false",
		"passed=false",
		"reason=candidate_admission_promotion_failed: candidate_admission_decision_failed: missing_candidate_execution",
	} {
		if !strings.Contains(switchLine, want) {
			t.Fatalf("candidate admission switch line missing %q: %q", want, switchLine)
		}
	}
	raw, err := os.ReadFile(switchLog)
	if err != nil {
		t.Fatal(err)
	}
	var got admissionLiveRouteTurnCandidateAdmissionSwitch
	if err := json.Unmarshal([]byte(strings.TrimSpace(string(raw))), &got); err != nil {
		t.Fatal(err)
	}
	if got.Schema != admissionLiveRouteTurnCandidateAdmissionSwitchSchema ||
		got.Passed ||
		got.LiveReady ||
		got.LiveAdmissionEnabled ||
		got.AdmissionAllowed ||
		got.MutatesState ||
		got.SwitchID != "" ||
		got.Reason != "candidate_admission_promotion_failed: candidate_admission_decision_failed: missing_candidate_execution" {
		t.Fatalf("bad candidate admission switch receipt: %+v", got)
	}
}

func TestChatLiveRouteTurnCandidateAdmissionEnableGateDryRunLine(t *testing.T) {
	t.Setenv("AM_LIVE_ROUTE_TURN_GENERATOR_ADAPTER_DRY_RUN", "1")
	t.Setenv("AM_LIVE_ROUTE_TURN_GENERATOR_ADAPTER_TEXT", "The dream remembers the field through one chain.")
	t.Setenv("AM_LIVE_ROUTE_TURN_CANDIDATE_DRAFT_DRY_RUN", "1")
	t.Setenv("AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_DRY_RUN", "1")
	t.Setenv("AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_ADAPTER_DRY_RUN", "1")
	t.Setenv("AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_SHADOW_DRY_RUN", "1")
	t.Setenv("AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_DECISION_DRY_RUN", "1")
	t.Setenv("AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_PROMOTION_DRY_RUN", "1")
	t.Setenv("AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_SWITCH_DRY_RUN", "1")
	t.Setenv("AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_ENABLE_GATE_DRY_RUN", "1")
	t.Setenv("AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_ENABLE_GATE_KEY", "")
	t.Setenv("AM_DREAM_ADMISSION", dreamAdmissionShadow)
	t.Setenv("AM_DREAM_ADMISSION_REQUIRE_LIVE_ROUTE_PLAN", "1")
	dir := t.TempDir()
	dreamLog := filepath.Join(dir, "dream-admission-chat-enable-gate.jsonl")
	decisionLog := filepath.Join(dir, "live-route-candidate-admission-decision.jsonl")
	promotionLog := filepath.Join(dir, "live-route-candidate-admission-promotion.jsonl")
	switchLog := filepath.Join(dir, "live-route-candidate-admission-switch.jsonl")
	enableGateLog := filepath.Join(dir, "live-route-candidate-admission-enable-gate.jsonl")
	t.Setenv("AM_DREAM_ADMISSION_LOG", dreamLog)
	t.Setenv("AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_DECISION_LOG", decisionLog)
	t.Setenv("AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_PROMOTION_LOG", promotionLog)
	t.Setenv("AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_SWITCH_LOG", switchLog)
	t.Setenv("AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_ENABLE_GATE_LOG", enableGateLog)

	obs := admissionLiveRouteTurnObservationForHuman("Tell me what the dream should remember.")
	lines := chatLiveRouteTurnCandidateChainDryRunLines(obs)
	if len(lines) != 9 {
		t.Fatalf("expected 9 candidate chain lines, got %d: %v", len(lines), lines)
	}
	enableGateLine := lines[len(lines)-1]
	for _, want := range []string{
		"live-route candidate admission enable gate dry-run",
		"class=dream",
		"route=direct",
		"source=direct",
		"switch=blocked",
		"enable=blocked",
		"enable_action=reject",
		"admission_allowed=false",
		"manual_enable=false",
		"key_matched=false",
		"live_ready=false",
		"live_enabled=false",
		"mutates=false",
		"passed=false",
		"reason=candidate_admission_switch_failed: candidate_admission_promotion_failed: candidate_admission_decision_failed: missing_candidate_execution",
	} {
		if !strings.Contains(enableGateLine, want) {
			t.Fatalf("candidate admission enable gate line missing %q: %q", want, enableGateLine)
		}
	}
	raw, err := os.ReadFile(enableGateLog)
	if err != nil {
		t.Fatal(err)
	}
	var got admissionLiveRouteTurnCandidateAdmissionEnableGate
	if err := json.Unmarshal([]byte(strings.TrimSpace(string(raw))), &got); err != nil {
		t.Fatal(err)
	}
	if got.Schema != admissionLiveRouteTurnCandidateAdmissionEnableGateSchema ||
		got.Passed ||
		got.LiveReady ||
		got.LiveAdmissionEnabled ||
		got.AdmissionAllowed ||
		got.ManualEnableRequested ||
		got.EnableKeyMatched ||
		got.MutatesState ||
		got.EnableGateID != "" ||
		got.Reason != "candidate_admission_switch_failed: candidate_admission_promotion_failed: candidate_admission_decision_failed: missing_candidate_execution" {
		t.Fatalf("bad candidate admission enable gate receipt: %+v", got)
	}
}

func TestChatLiveRouteTurnCandidateAdmissionLiveStageDryRunLine(t *testing.T) {
	t.Setenv("AM_LIVE_ROUTE_TURN_GENERATOR_ADAPTER_DRY_RUN", "1")
	t.Setenv("AM_LIVE_ROUTE_TURN_GENERATOR_ADAPTER_TEXT", "The dream remembers the field through one chain.")
	t.Setenv("AM_LIVE_ROUTE_TURN_CANDIDATE_DRAFT_DRY_RUN", "1")
	t.Setenv("AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_DRY_RUN", "1")
	t.Setenv("AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_ADAPTER_DRY_RUN", "1")
	t.Setenv("AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_SHADOW_DRY_RUN", "1")
	t.Setenv("AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_DECISION_DRY_RUN", "1")
	t.Setenv("AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_PROMOTION_DRY_RUN", "1")
	t.Setenv("AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_SWITCH_DRY_RUN", "1")
	t.Setenv("AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_ENABLE_GATE_DRY_RUN", "1")
	t.Setenv("AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_LIVE_STAGE_DRY_RUN", "1")
	t.Setenv("AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_ENABLE_GATE_KEY", admissionLiveRouteTurnCandidateAdmissionEnableGateConfirmation)
	t.Setenv("AM_DREAM_ADMISSION", dreamAdmissionShadow)
	t.Setenv("AM_DREAM_ADMISSION_REQUIRE_LIVE_ROUTE_PLAN", "1")
	dir := t.TempDir()
	dreamLog := filepath.Join(dir, "dream-admission-chat-live-stage.jsonl")
	decisionLog := filepath.Join(dir, "live-route-candidate-admission-decision.jsonl")
	promotionLog := filepath.Join(dir, "live-route-candidate-admission-promotion.jsonl")
	switchLog := filepath.Join(dir, "live-route-candidate-admission-switch.jsonl")
	enableGateLog := filepath.Join(dir, "live-route-candidate-admission-enable-gate.jsonl")
	liveStageLog := filepath.Join(dir, "live-route-candidate-admission-live-stage.jsonl")
	t.Setenv("AM_DREAM_ADMISSION_LOG", dreamLog)
	t.Setenv("AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_DECISION_LOG", decisionLog)
	t.Setenv("AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_PROMOTION_LOG", promotionLog)
	t.Setenv("AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_SWITCH_LOG", switchLog)
	t.Setenv("AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_ENABLE_GATE_LOG", enableGateLog)
	t.Setenv("AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_LIVE_STAGE_LOG", liveStageLog)

	obs := admissionLiveRouteTurnObservationForHuman("Tell me what the dream should remember.")
	lines := chatLiveRouteTurnCandidateChainDryRunLines(obs)
	if len(lines) != 10 {
		t.Fatalf("expected 10 candidate chain lines, got %d: %v", len(lines), lines)
	}
	liveStageLine := lines[len(lines)-1]
	for _, want := range []string{
		"live-route candidate admission live stage dry-run",
		"class=dream",
		"route=direct",
		"source=direct",
		"enable=blocked",
		"stage=blocked",
		"stage_action=reject",
		"admission_allowed=false",
		"writer_ready=false",
		"rollback_ready=false",
		"live_ready=false",
		"live_enabled=false",
		"mutates=false",
		"passed=false",
		"reason=candidate_admission_enable_gate_failed: candidate_admission_switch_failed: candidate_admission_promotion_failed: candidate_admission_decision_failed: missing_candidate_execution",
	} {
		if !strings.Contains(liveStageLine, want) {
			t.Fatalf("candidate admission live stage line missing %q: %q", want, liveStageLine)
		}
	}
	raw, err := os.ReadFile(liveStageLog)
	if err != nil {
		t.Fatal(err)
	}
	var got admissionLiveRouteTurnCandidateAdmissionLiveStage
	if err := json.Unmarshal([]byte(strings.TrimSpace(string(raw))), &got); err != nil {
		t.Fatal(err)
	}
	if got.Schema != admissionLiveRouteTurnCandidateAdmissionLiveStageSchema ||
		got.Passed ||
		got.LiveReady ||
		got.LiveAdmissionEnabled ||
		got.AdmissionAllowed ||
		!got.ManualEnableRequested ||
		!got.EnableKeyMatched ||
		got.MutatesState ||
		got.LiveStageID != "" ||
		got.Reason != "candidate_admission_enable_gate_failed: candidate_admission_switch_failed: candidate_admission_promotion_failed: candidate_admission_decision_failed: missing_candidate_execution" {
		t.Fatalf("bad candidate admission live stage receipt: %+v", got)
	}
}

func TestChatLiveRouteTurnCandidateAdmissionWriterPreflightDryRunLine(t *testing.T) {
	t.Setenv("AM_LIVE_ROUTE_TURN_GENERATOR_ADAPTER_DRY_RUN", "1")
	t.Setenv("AM_LIVE_ROUTE_TURN_GENERATOR_ADAPTER_TEXT", "The dream remembers the field through one chain.")
	t.Setenv("AM_LIVE_ROUTE_TURN_CANDIDATE_DRAFT_DRY_RUN", "1")
	t.Setenv("AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_DRY_RUN", "1")
	t.Setenv("AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_ADAPTER_DRY_RUN", "1")
	t.Setenv("AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_SHADOW_DRY_RUN", "1")
	t.Setenv("AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_DECISION_DRY_RUN", "1")
	t.Setenv("AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_PROMOTION_DRY_RUN", "1")
	t.Setenv("AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_SWITCH_DRY_RUN", "1")
	t.Setenv("AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_ENABLE_GATE_DRY_RUN", "1")
	t.Setenv("AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_LIVE_STAGE_DRY_RUN", "1")
	t.Setenv("AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_WRITER_PREFLIGHT_DRY_RUN", "1")
	t.Setenv("AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_ENABLE_GATE_KEY", admissionLiveRouteTurnCandidateAdmissionEnableGateConfirmation)
	t.Setenv("AM_DREAM_ADMISSION", dreamAdmissionShadow)
	t.Setenv("AM_DREAM_ADMISSION_REQUIRE_LIVE_ROUTE_PLAN", "1")
	dir := t.TempDir()
	dreamLog := filepath.Join(dir, "dream-admission-chat-writer-preflight.jsonl")
	decisionLog := filepath.Join(dir, "live-route-candidate-admission-decision.jsonl")
	promotionLog := filepath.Join(dir, "live-route-candidate-admission-promotion.jsonl")
	switchLog := filepath.Join(dir, "live-route-candidate-admission-switch.jsonl")
	enableGateLog := filepath.Join(dir, "live-route-candidate-admission-enable-gate.jsonl")
	liveStageLog := filepath.Join(dir, "live-route-candidate-admission-live-stage.jsonl")
	writerPreflightLog := filepath.Join(dir, "live-route-candidate-admission-writer-preflight.jsonl")
	t.Setenv("AM_DREAM_ADMISSION_LOG", dreamLog)
	t.Setenv("AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_DECISION_LOG", decisionLog)
	t.Setenv("AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_PROMOTION_LOG", promotionLog)
	t.Setenv("AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_SWITCH_LOG", switchLog)
	t.Setenv("AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_ENABLE_GATE_LOG", enableGateLog)
	t.Setenv("AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_LIVE_STAGE_LOG", liveStageLog)
	t.Setenv("AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_WRITER_PREFLIGHT_LOG", writerPreflightLog)

	obs := admissionLiveRouteTurnObservationForHuman("Tell me what the dream should remember.")
	lines := chatLiveRouteTurnCandidateChainDryRunLines(obs)
	if len(lines) != 11 {
		t.Fatalf("expected 11 candidate chain lines, got %d: %v", len(lines), lines)
	}
	writerPreflightLine := lines[len(lines)-1]
	for _, want := range []string{
		"live-route candidate admission writer preflight dry-run",
		"class=dream",
		"route=direct",
		"source=direct",
		"stage=blocked",
		"writer=blocked",
		"writer_action=reject",
		"rollback=blocked",
		"rollback_action=reject",
		"write_allowed=false",
		"admission_allowed=false",
		"live_ready=false",
		"live_enabled=false",
		"mutates=false",
		"passed=false",
		"reason=candidate_admission_live_stage_failed: candidate_admission_enable_gate_failed: candidate_admission_switch_failed: candidate_admission_promotion_failed: candidate_admission_decision_failed: missing_candidate_execution",
	} {
		if !strings.Contains(writerPreflightLine, want) {
			t.Fatalf("candidate admission writer preflight line missing %q: %q", want, writerPreflightLine)
		}
	}
	raw, err := os.ReadFile(writerPreflightLog)
	if err != nil {
		t.Fatal(err)
	}
	var got admissionLiveRouteTurnCandidateAdmissionWriterPreflight
	if err := json.Unmarshal([]byte(strings.TrimSpace(string(raw))), &got); err != nil {
		t.Fatal(err)
	}
	if got.Schema != admissionLiveRouteTurnCandidateAdmissionWriterPreflightSchema ||
		got.Passed ||
		got.LiveReady ||
		got.LiveAdmissionEnabled ||
		got.AdmissionAllowed ||
		!got.ManualEnableRequested ||
		!got.EnableKeyMatched ||
		got.WriteAllowed ||
		got.MutatesState ||
		got.WriterPreflightID != "" ||
		got.Reason != "candidate_admission_live_stage_failed: candidate_admission_enable_gate_failed: candidate_admission_switch_failed: candidate_admission_promotion_failed: candidate_admission_decision_failed: missing_candidate_execution" {
		t.Fatalf("bad candidate admission writer preflight receipt: %+v", got)
	}
}

func TestChatLiveRouteTurnCandidateAdmissionWriterInventoryDryRunLine(t *testing.T) {
	t.Setenv("AM_LIVE_ROUTE_TURN_GENERATOR_ADAPTER_DRY_RUN", "1")
	t.Setenv("AM_LIVE_ROUTE_TURN_GENERATOR_ADAPTER_TEXT", "The dream remembers the field through one chain.")
	t.Setenv("AM_LIVE_ROUTE_TURN_CANDIDATE_DRAFT_DRY_RUN", "1")
	t.Setenv("AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_DRY_RUN", "1")
	t.Setenv("AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_ADAPTER_DRY_RUN", "1")
	t.Setenv("AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_SHADOW_DRY_RUN", "1")
	t.Setenv("AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_DECISION_DRY_RUN", "1")
	t.Setenv("AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_PROMOTION_DRY_RUN", "1")
	t.Setenv("AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_SWITCH_DRY_RUN", "1")
	t.Setenv("AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_ENABLE_GATE_DRY_RUN", "1")
	t.Setenv("AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_LIVE_STAGE_DRY_RUN", "1")
	t.Setenv("AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_WRITER_PREFLIGHT_DRY_RUN", "1")
	t.Setenv("AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_WRITER_INVENTORY_DRY_RUN", "1")
	t.Setenv("AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_ENABLE_GATE_KEY", admissionLiveRouteTurnCandidateAdmissionEnableGateConfirmation)
	t.Setenv("AM_DREAM_ADMISSION", dreamAdmissionShadow)
	t.Setenv("AM_DREAM_ADMISSION_REQUIRE_LIVE_ROUTE_PLAN", "1")
	dir := t.TempDir()
	dreamLog := filepath.Join(dir, "dream-admission-chat-writer-inventory.jsonl")
	decisionLog := filepath.Join(dir, "live-route-candidate-admission-decision.jsonl")
	promotionLog := filepath.Join(dir, "live-route-candidate-admission-promotion.jsonl")
	switchLog := filepath.Join(dir, "live-route-candidate-admission-switch.jsonl")
	enableGateLog := filepath.Join(dir, "live-route-candidate-admission-enable-gate.jsonl")
	liveStageLog := filepath.Join(dir, "live-route-candidate-admission-live-stage.jsonl")
	writerPreflightLog := filepath.Join(dir, "live-route-candidate-admission-writer-preflight.jsonl")
	writerInventoryLog := filepath.Join(dir, "live-route-candidate-admission-writer-inventory.jsonl")
	t.Setenv("AM_DREAM_ADMISSION_LOG", dreamLog)
	t.Setenv("AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_DECISION_LOG", decisionLog)
	t.Setenv("AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_PROMOTION_LOG", promotionLog)
	t.Setenv("AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_SWITCH_LOG", switchLog)
	t.Setenv("AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_ENABLE_GATE_LOG", enableGateLog)
	t.Setenv("AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_LIVE_STAGE_LOG", liveStageLog)
	t.Setenv("AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_WRITER_PREFLIGHT_LOG", writerPreflightLog)
	t.Setenv("AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_WRITER_INVENTORY_LOG", writerInventoryLog)

	obs := admissionLiveRouteTurnObservationForHuman("Tell me what the dream should remember.")
	lines := chatLiveRouteTurnCandidateChainDryRunLines(obs)
	if len(lines) != 12 {
		t.Fatalf("expected 12 candidate chain lines, got %d: %v", len(lines), lines)
	}
	writerInventoryLine := lines[len(lines)-1]
	for _, want := range []string{
		"live-route candidate admission writer inventory dry-run",
		"class=dream",
		"route=direct",
		"source=direct",
		"writer_preflight=",
		"inventory=blocked",
		"inventory_action=reject",
		"contracts_ready=false",
		"write_allowed=false",
		"admission_allowed=false",
		"live_ready=false",
		"live_enabled=false",
		"mutates=false",
		"writer_inventory_id=",
		"passed=false",
		"reason=candidate_admission_writer_preflight_failed: candidate_admission_live_stage_failed: candidate_admission_enable_gate_failed: candidate_admission_switch_failed: candidate_admission_promotion_failed: candidate_admission_decision_failed: missing_candidate_execution",
	} {
		if !strings.Contains(writerInventoryLine, want) {
			t.Fatalf("candidate admission writer inventory line missing %q: %q", want, writerInventoryLine)
		}
	}
	raw, err := os.ReadFile(writerInventoryLog)
	if err != nil {
		t.Fatal(err)
	}
	var got admissionLiveRouteTurnCandidateAdmissionWriterInventory
	if err := json.Unmarshal([]byte(strings.TrimSpace(string(raw))), &got); err != nil {
		t.Fatal(err)
	}
	if got.Schema != admissionLiveRouteTurnCandidateAdmissionWriterInventorySchema ||
		got.Passed ||
		got.LiveReady ||
		got.LiveAdmissionEnabled ||
		got.AdmissionAllowed ||
		!got.ManualEnableRequested ||
		!got.EnableKeyMatched ||
		got.ContractsReady ||
		got.WriteAllowed ||
		got.MutatesState ||
		got.SourceWriterPreflightPassed ||
		got.WriterInventoryID != "" ||
		got.Reason != "candidate_admission_writer_preflight_failed: candidate_admission_live_stage_failed: candidate_admission_enable_gate_failed: candidate_admission_switch_failed: candidate_admission_promotion_failed: candidate_admission_decision_failed: missing_candidate_execution" {
		t.Fatalf("bad candidate admission writer inventory receipt: %+v", got)
	}
}

func TestChatLiveRouteTurnCandidateAdmissionWriterContractDryRunLine(t *testing.T) {
	t.Setenv("AM_LIVE_ROUTE_TURN_GENERATOR_ADAPTER_DRY_RUN", "1")
	t.Setenv("AM_LIVE_ROUTE_TURN_GENERATOR_ADAPTER_TEXT", "The dream remembers the field through one chain.")
	t.Setenv("AM_LIVE_ROUTE_TURN_CANDIDATE_DRAFT_DRY_RUN", "1")
	t.Setenv("AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_DRY_RUN", "1")
	t.Setenv("AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_ADAPTER_DRY_RUN", "1")
	t.Setenv("AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_SHADOW_DRY_RUN", "1")
	t.Setenv("AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_DECISION_DRY_RUN", "1")
	t.Setenv("AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_PROMOTION_DRY_RUN", "1")
	t.Setenv("AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_SWITCH_DRY_RUN", "1")
	t.Setenv("AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_ENABLE_GATE_DRY_RUN", "1")
	t.Setenv("AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_LIVE_STAGE_DRY_RUN", "1")
	t.Setenv("AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_WRITER_PREFLIGHT_DRY_RUN", "1")
	t.Setenv("AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_WRITER_INVENTORY_DRY_RUN", "1")
	t.Setenv("AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_WRITER_CONTRACT_DRY_RUN", "1")
	t.Setenv("AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_ENABLE_GATE_KEY", admissionLiveRouteTurnCandidateAdmissionEnableGateConfirmation)
	t.Setenv("AM_DREAM_ADMISSION", dreamAdmissionShadow)
	t.Setenv("AM_DREAM_ADMISSION_REQUIRE_LIVE_ROUTE_PLAN", "1")
	dir := t.TempDir()
	dreamLog := filepath.Join(dir, "dream-admission-chat-writer-contract.jsonl")
	decisionLog := filepath.Join(dir, "live-route-candidate-admission-decision.jsonl")
	promotionLog := filepath.Join(dir, "live-route-candidate-admission-promotion.jsonl")
	switchLog := filepath.Join(dir, "live-route-candidate-admission-switch.jsonl")
	enableGateLog := filepath.Join(dir, "live-route-candidate-admission-enable-gate.jsonl")
	liveStageLog := filepath.Join(dir, "live-route-candidate-admission-live-stage.jsonl")
	writerPreflightLog := filepath.Join(dir, "live-route-candidate-admission-writer-preflight.jsonl")
	writerInventoryLog := filepath.Join(dir, "live-route-candidate-admission-writer-inventory.jsonl")
	writerContractLog := filepath.Join(dir, "live-route-candidate-admission-writer-contract.jsonl")
	t.Setenv("AM_DREAM_ADMISSION_LOG", dreamLog)
	t.Setenv("AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_DECISION_LOG", decisionLog)
	t.Setenv("AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_PROMOTION_LOG", promotionLog)
	t.Setenv("AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_SWITCH_LOG", switchLog)
	t.Setenv("AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_ENABLE_GATE_LOG", enableGateLog)
	t.Setenv("AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_LIVE_STAGE_LOG", liveStageLog)
	t.Setenv("AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_WRITER_PREFLIGHT_LOG", writerPreflightLog)
	t.Setenv("AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_WRITER_INVENTORY_LOG", writerInventoryLog)
	t.Setenv("AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_WRITER_CONTRACT_LOG", writerContractLog)

	obs := admissionLiveRouteTurnObservationForHuman("Tell me what the dream should remember.")
	lines := chatLiveRouteTurnCandidateChainDryRunLines(obs)
	if len(lines) != 13 {
		t.Fatalf("expected 13 candidate chain lines, got %d: %v", len(lines), lines)
	}
	writerContractLine := lines[len(lines)-1]
	for _, want := range []string{
		"live-route candidate admission writer contract dry-run",
		"class=dream",
		"route=direct",
		"source=direct",
		"writer_inventory=",
		"contract=blocked",
		"contract_action=reject",
		"shape_ready=false",
		"writer_impl=false rollback_impl=false ledger_impl=false",
		"contracts_ready=false write_allowed=false admission_allowed=false live_ready=false live_enabled=false mutates=false",
		"writer_contract_id=",
		"passed=false",
		"reason=candidate_admission_writer_inventory_failed: candidate_admission_writer_preflight_failed: candidate_admission_live_stage_failed: candidate_admission_enable_gate_failed: candidate_admission_switch_failed: candidate_admission_promotion_failed: candidate_admission_decision_failed: missing_candidate_execution",
	} {
		if !strings.Contains(writerContractLine, want) {
			t.Fatalf("candidate admission writer contract line missing %q: %q", want, writerContractLine)
		}
	}
	raw, err := os.ReadFile(writerContractLog)
	if err != nil {
		t.Fatal(err)
	}
	var got admissionLiveRouteTurnCandidateAdmissionWriterContract
	if err := json.Unmarshal([]byte(strings.TrimSpace(string(raw))), &got); err != nil {
		t.Fatal(err)
	}
	if got.Schema != admissionLiveRouteTurnCandidateAdmissionWriterContractSchema ||
		got.Passed ||
		got.LiveReady ||
		got.LiveAdmissionEnabled ||
		got.AdmissionAllowed ||
		!got.ManualEnableRequested ||
		!got.EnableKeyMatched ||
		got.ContractShapeReady ||
		got.WriterImplementationReady ||
		got.RollbackImplementationReady ||
		got.LedgerImplementationReady ||
		got.ContractsReady ||
		got.WriteAllowed ||
		got.MutatesState ||
		got.SourceWriterInventoryPassed ||
		got.WriterContractID != "" ||
		got.Reason != "candidate_admission_writer_inventory_failed: candidate_admission_writer_preflight_failed: candidate_admission_live_stage_failed: candidate_admission_enable_gate_failed: candidate_admission_switch_failed: candidate_admission_promotion_failed: candidate_admission_decision_failed: missing_candidate_execution" {
		t.Fatalf("bad candidate admission writer contract receipt: %+v", got)
	}
}

func TestChatLiveRouteTurnCandidateAdmissionLedgerDryRunLine(t *testing.T) {
	t.Setenv("AM_LIVE_ROUTE_TURN_GENERATOR_ADAPTER_DRY_RUN", "1")
	t.Setenv("AM_LIVE_ROUTE_TURN_GENERATOR_ADAPTER_TEXT", "The dream remembers the field through one chain.")
	t.Setenv("AM_LIVE_ROUTE_TURN_CANDIDATE_DRAFT_DRY_RUN", "1")
	t.Setenv("AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_DRY_RUN", "1")
	t.Setenv("AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_ADAPTER_DRY_RUN", "1")
	t.Setenv("AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_SHADOW_DRY_RUN", "1")
	t.Setenv("AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_DECISION_DRY_RUN", "1")
	t.Setenv("AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_PROMOTION_DRY_RUN", "1")
	t.Setenv("AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_SWITCH_DRY_RUN", "1")
	t.Setenv("AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_ENABLE_GATE_DRY_RUN", "1")
	t.Setenv("AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_LIVE_STAGE_DRY_RUN", "1")
	t.Setenv("AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_WRITER_PREFLIGHT_DRY_RUN", "1")
	t.Setenv("AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_WRITER_INVENTORY_DRY_RUN", "1")
	t.Setenv("AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_WRITER_CONTRACT_DRY_RUN", "1")
	t.Setenv("AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_LEDGER_DRY_RUN", "1")
	t.Setenv("AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_ENABLE_GATE_KEY", admissionLiveRouteTurnCandidateAdmissionEnableGateConfirmation)
	t.Setenv("AM_DREAM_ADMISSION", dreamAdmissionShadow)
	t.Setenv("AM_DREAM_ADMISSION_REQUIRE_LIVE_ROUTE_PLAN", "1")
	dir := t.TempDir()
	dreamLog := filepath.Join(dir, "dream-admission-chat-ledger.jsonl")
	decisionLog := filepath.Join(dir, "live-route-candidate-admission-decision.jsonl")
	promotionLog := filepath.Join(dir, "live-route-candidate-admission-promotion.jsonl")
	switchLog := filepath.Join(dir, "live-route-candidate-admission-switch.jsonl")
	enableGateLog := filepath.Join(dir, "live-route-candidate-admission-enable-gate.jsonl")
	liveStageLog := filepath.Join(dir, "live-route-candidate-admission-live-stage.jsonl")
	writerPreflightLog := filepath.Join(dir, "live-route-candidate-admission-writer-preflight.jsonl")
	writerInventoryLog := filepath.Join(dir, "live-route-candidate-admission-writer-inventory.jsonl")
	writerContractLog := filepath.Join(dir, "live-route-candidate-admission-writer-contract.jsonl")
	ledgerLog := filepath.Join(dir, "live-route-candidate-admission-ledger.jsonl")
	t.Setenv("AM_DREAM_ADMISSION_LOG", dreamLog)
	t.Setenv("AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_DECISION_LOG", decisionLog)
	t.Setenv("AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_PROMOTION_LOG", promotionLog)
	t.Setenv("AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_SWITCH_LOG", switchLog)
	t.Setenv("AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_ENABLE_GATE_LOG", enableGateLog)
	t.Setenv("AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_LIVE_STAGE_LOG", liveStageLog)
	t.Setenv("AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_WRITER_PREFLIGHT_LOG", writerPreflightLog)
	t.Setenv("AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_WRITER_INVENTORY_LOG", writerInventoryLog)
	t.Setenv("AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_WRITER_CONTRACT_LOG", writerContractLog)
	t.Setenv("AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_LEDGER_LOG", ledgerLog)

	obs := admissionLiveRouteTurnObservationForHuman("Tell me what the dream should remember.")
	lines := chatLiveRouteTurnCandidateChainDryRunLines(obs)
	if len(lines) != 14 {
		t.Fatalf("expected 14 candidate chain lines, got %d: %v", len(lines), lines)
	}
	ledgerLine := lines[len(lines)-1]
	for _, want := range []string{
		"live-route candidate admission ledger dry-run",
		"class=dream",
		"route=direct",
		"source=direct",
		"writer_contract=",
		"ledger=blocked",
		"ledger_action=reject",
		"append_ready=false",
		"persisted=false",
		"ledger_impl=false",
		"contracts_ready=false write_allowed=false admission_allowed=false live_ready=false live_enabled=false mutates=false",
		"admission_ledger_id=",
		"passed=false",
		"reason=candidate_admission_writer_contract_failed: candidate_admission_writer_inventory_failed: candidate_admission_writer_preflight_failed: candidate_admission_live_stage_failed: candidate_admission_enable_gate_failed: candidate_admission_switch_failed: candidate_admission_promotion_failed: candidate_admission_decision_failed: missing_candidate_execution",
	} {
		if !strings.Contains(ledgerLine, want) {
			t.Fatalf("candidate admission ledger line missing %q: %q", want, ledgerLine)
		}
	}
	raw, err := os.ReadFile(ledgerLog)
	if err != nil {
		t.Fatal(err)
	}
	var got admissionLiveRouteTurnCandidateAdmissionLedger
	if err := json.Unmarshal([]byte(strings.TrimSpace(string(raw))), &got); err != nil {
		t.Fatal(err)
	}
	if got.Schema != admissionLiveRouteTurnCandidateAdmissionLedgerSchema ||
		got.Passed ||
		got.LiveReady ||
		got.LiveAdmissionEnabled ||
		got.AdmissionAllowed ||
		!got.ManualEnableRequested ||
		!got.EnableKeyMatched ||
		got.LedgerAppendReady ||
		got.LedgerReceiptPersisted ||
		got.LedgerImplementationReady ||
		got.ContractsReady ||
		got.WriteAllowed ||
		got.MutatesState ||
		got.SourceWriterContractPassed ||
		got.AdmissionLedgerID != "" ||
		got.Reason != "candidate_admission_writer_contract_failed: candidate_admission_writer_inventory_failed: candidate_admission_writer_preflight_failed: candidate_admission_live_stage_failed: candidate_admission_enable_gate_failed: candidate_admission_switch_failed: candidate_admission_promotion_failed: candidate_admission_decision_failed: missing_candidate_execution" {
		t.Fatalf("bad candidate admission ledger receipt: %+v", got)
	}
}

func TestChatLiveRouteTurnCandidateAdmissionWriterImplementationDryRunLine(t *testing.T) {
	t.Setenv("AM_LIVE_ROUTE_TURN_GENERATOR_ADAPTER_DRY_RUN", "1")
	t.Setenv("AM_LIVE_ROUTE_TURN_GENERATOR_ADAPTER_TEXT", "The dream keeps a receipt before it touches the body.")
	t.Setenv("AM_LIVE_ROUTE_TURN_CANDIDATE_DRAFT_DRY_RUN", "1")
	t.Setenv("AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_DRY_RUN", "1")
	t.Setenv("AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_ADAPTER_DRY_RUN", "1")
	t.Setenv("AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_SHADOW_DRY_RUN", "1")
	t.Setenv("AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_DECISION_DRY_RUN", "1")
	t.Setenv("AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_PROMOTION_DRY_RUN", "1")
	t.Setenv("AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_SWITCH_DRY_RUN", "1")
	t.Setenv("AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_ENABLE_GATE_DRY_RUN", "1")
	t.Setenv("AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_LIVE_STAGE_DRY_RUN", "1")
	t.Setenv("AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_WRITER_PREFLIGHT_DRY_RUN", "1")
	t.Setenv("AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_WRITER_INVENTORY_DRY_RUN", "1")
	t.Setenv("AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_WRITER_CONTRACT_DRY_RUN", "1")
	t.Setenv("AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_LEDGER_DRY_RUN", "1")
	t.Setenv("AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_WRITER_IMPLEMENTATION_DRY_RUN", "1")
	t.Setenv("AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_ENABLE_GATE_KEY", admissionLiveRouteTurnCandidateAdmissionEnableGateConfirmation)
	t.Setenv("AM_DREAM_ADMISSION", dreamAdmissionShadow)
	t.Setenv("AM_DREAM_ADMISSION_REQUIRE_LIVE_ROUTE_PLAN", "1")
	dir := t.TempDir()
	dreamLog := filepath.Join(dir, "dream-admission-chat-writer-implementation.jsonl")
	decisionLog := filepath.Join(dir, "live-route-candidate-admission-decision.jsonl")
	promotionLog := filepath.Join(dir, "live-route-candidate-admission-promotion.jsonl")
	switchLog := filepath.Join(dir, "live-route-candidate-admission-switch.jsonl")
	enableGateLog := filepath.Join(dir, "live-route-candidate-admission-enable-gate.jsonl")
	liveStageLog := filepath.Join(dir, "live-route-candidate-admission-live-stage.jsonl")
	writerPreflightLog := filepath.Join(dir, "live-route-candidate-admission-writer-preflight.jsonl")
	writerInventoryLog := filepath.Join(dir, "live-route-candidate-admission-writer-inventory.jsonl")
	writerContractLog := filepath.Join(dir, "live-route-candidate-admission-writer-contract.jsonl")
	ledgerLog := filepath.Join(dir, "live-route-candidate-admission-ledger.jsonl")
	writerImplLog := filepath.Join(dir, "live-route-candidate-admission-writer-implementation.jsonl")
	t.Setenv("AM_DREAM_ADMISSION_LOG", dreamLog)
	t.Setenv("AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_DECISION_LOG", decisionLog)
	t.Setenv("AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_PROMOTION_LOG", promotionLog)
	t.Setenv("AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_SWITCH_LOG", switchLog)
	t.Setenv("AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_ENABLE_GATE_LOG", enableGateLog)
	t.Setenv("AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_LIVE_STAGE_LOG", liveStageLog)
	t.Setenv("AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_WRITER_PREFLIGHT_LOG", writerPreflightLog)
	t.Setenv("AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_WRITER_INVENTORY_LOG", writerInventoryLog)
	t.Setenv("AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_WRITER_CONTRACT_LOG", writerContractLog)
	t.Setenv("AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_LEDGER_LOG", ledgerLog)
	t.Setenv("AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_WRITER_IMPLEMENTATION_LOG", writerImplLog)

	obs := admissionLiveRouteTurnObservationForHuman("Tell me what the dream should remember.")
	lines := chatLiveRouteTurnCandidateChainDryRunLines(obs)
	if len(lines) != 15 {
		t.Fatalf("expected 15 candidate chain lines, got %d: %v", len(lines), lines)
	}
	implLine := lines[len(lines)-1]
	for _, want := range []string{
		"live-route candidate admission writer implementation dry-run",
		"class=dream",
		"route=direct",
		"source=direct",
		"ledger=",
		"implementation=blocked",
		"implementation_action=reject",
		"writer_entrypoint=",
		"ledger_entrypoint=",
		"rollback_entrypoint=",
		"write_target=",
		"body_target=",
		"append_only=false",
		"rollback_required=false",
		"implementation_contract=false",
		"writer_impl=false ledger_impl=false rollback_impl=false",
		"contracts_ready=false write_allowed=false admission_allowed=false live_ready=false live_enabled=false mutates=false",
		"writer_implementation_id=",
		"passed=false",
		"reason=candidate_admission_ledger_failed: candidate_admission_writer_contract_failed: candidate_admission_writer_inventory_failed: candidate_admission_writer_preflight_failed: candidate_admission_live_stage_failed: candidate_admission_enable_gate_failed: candidate_admission_switch_failed: candidate_admission_promotion_failed: candidate_admission_decision_failed: missing_candidate_execution",
	} {
		if !strings.Contains(implLine, want) {
			t.Fatalf("candidate admission writer implementation line missing %q: %q", want, implLine)
		}
	}
	raw, err := os.ReadFile(writerImplLog)
	if err != nil {
		t.Fatal(err)
	}
	var got admissionLiveRouteTurnCandidateAdmissionWriterImplementation
	if err := json.Unmarshal([]byte(strings.TrimSpace(string(raw))), &got); err != nil {
		t.Fatal(err)
	}
	if got.Schema != admissionLiveRouteTurnCandidateAdmissionWriterImplSchema ||
		got.Passed ||
		got.LiveReady ||
		got.LiveAdmissionEnabled ||
		got.AdmissionAllowed ||
		!got.ManualEnableRequested ||
		!got.EnableKeyMatched ||
		got.AppendOnly ||
		got.RollbackRequired ||
		got.ImplementationContractReady ||
		got.WriterImplementationReady ||
		got.LedgerImplementationReady ||
		got.RollbackImplementationReady ||
		got.ContractsReady ||
		got.WriteAllowed ||
		got.MutatesState ||
		got.SourceLedgerPassed ||
		got.WriterImplementationID != "" ||
		got.Reason != "candidate_admission_ledger_failed: candidate_admission_writer_contract_failed: candidate_admission_writer_inventory_failed: candidate_admission_writer_preflight_failed: candidate_admission_live_stage_failed: candidate_admission_enable_gate_failed: candidate_admission_switch_failed: candidate_admission_promotion_failed: candidate_admission_decision_failed: missing_candidate_execution" {
		t.Fatalf("bad candidate admission writer implementation receipt: %+v", got)
	}
}

func TestChatLiveRouteTurnCandidateAdmissionWriterReceiptDryRunLine(t *testing.T) {
	t.Setenv("AM_LIVE_ROUTE_TURN_GENERATOR_ADAPTER_DRY_RUN", "1")
	t.Setenv("AM_LIVE_ROUTE_TURN_GENERATOR_ADAPTER_TEXT", "The dream keeps a receipt before it touches the body.")
	t.Setenv("AM_LIVE_ROUTE_TURN_CANDIDATE_DRAFT_DRY_RUN", "1")
	t.Setenv("AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_DRY_RUN", "1")
	t.Setenv("AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_ADAPTER_DRY_RUN", "1")
	t.Setenv("AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_SHADOW_DRY_RUN", "1")
	t.Setenv("AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_DECISION_DRY_RUN", "1")
	t.Setenv("AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_PROMOTION_DRY_RUN", "1")
	t.Setenv("AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_SWITCH_DRY_RUN", "1")
	t.Setenv("AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_ENABLE_GATE_DRY_RUN", "1")
	t.Setenv("AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_LIVE_STAGE_DRY_RUN", "1")
	t.Setenv("AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_WRITER_PREFLIGHT_DRY_RUN", "1")
	t.Setenv("AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_WRITER_INVENTORY_DRY_RUN", "1")
	t.Setenv("AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_WRITER_CONTRACT_DRY_RUN", "1")
	t.Setenv("AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_LEDGER_DRY_RUN", "1")
	t.Setenv("AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_WRITER_IMPLEMENTATION_DRY_RUN", "1")
	t.Setenv("AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_WRITER_RECEIPT_DRY_RUN", "1")
	t.Setenv("AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_ENABLE_GATE_KEY", admissionLiveRouteTurnCandidateAdmissionEnableGateConfirmation)
	t.Setenv("AM_DREAM_ADMISSION", dreamAdmissionShadow)
	t.Setenv("AM_DREAM_ADMISSION_REQUIRE_LIVE_ROUTE_PLAN", "1")
	dir := t.TempDir()
	dreamLog := filepath.Join(dir, "dream-admission-chat-writer-receipt.jsonl")
	decisionLog := filepath.Join(dir, "live-route-candidate-admission-decision.jsonl")
	promotionLog := filepath.Join(dir, "live-route-candidate-admission-promotion.jsonl")
	switchLog := filepath.Join(dir, "live-route-candidate-admission-switch.jsonl")
	enableGateLog := filepath.Join(dir, "live-route-candidate-admission-enable-gate.jsonl")
	liveStageLog := filepath.Join(dir, "live-route-candidate-admission-live-stage.jsonl")
	writerPreflightLog := filepath.Join(dir, "live-route-candidate-admission-writer-preflight.jsonl")
	writerInventoryLog := filepath.Join(dir, "live-route-candidate-admission-writer-inventory.jsonl")
	writerContractLog := filepath.Join(dir, "live-route-candidate-admission-writer-contract.jsonl")
	ledgerLog := filepath.Join(dir, "live-route-candidate-admission-ledger.jsonl")
	writerImplLog := filepath.Join(dir, "live-route-candidate-admission-writer-implementation.jsonl")
	writerReceiptLog := filepath.Join(dir, "live-route-candidate-admission-writer-receipt.jsonl")
	t.Setenv("AM_DREAM_ADMISSION_LOG", dreamLog)
	t.Setenv("AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_DECISION_LOG", decisionLog)
	t.Setenv("AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_PROMOTION_LOG", promotionLog)
	t.Setenv("AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_SWITCH_LOG", switchLog)
	t.Setenv("AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_ENABLE_GATE_LOG", enableGateLog)
	t.Setenv("AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_LIVE_STAGE_LOG", liveStageLog)
	t.Setenv("AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_WRITER_PREFLIGHT_LOG", writerPreflightLog)
	t.Setenv("AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_WRITER_INVENTORY_LOG", writerInventoryLog)
	t.Setenv("AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_WRITER_CONTRACT_LOG", writerContractLog)
	t.Setenv("AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_LEDGER_LOG", ledgerLog)
	t.Setenv("AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_WRITER_IMPLEMENTATION_LOG", writerImplLog)
	t.Setenv("AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_WRITER_RECEIPT_LOG", writerReceiptLog)

	obs := admissionLiveRouteTurnObservationForHuman("Tell me what the dream should remember.")
	lines := chatLiveRouteTurnCandidateChainDryRunLines(obs)
	if len(lines) != 16 {
		t.Fatalf("expected 16 candidate chain lines, got %d: %v", len(lines), lines)
	}
	receiptLine := lines[len(lines)-1]
	for _, want := range []string{
		"live-route candidate admission writer receipt dry-run",
		"class=dream",
		"route=direct",
		"source=direct",
		"writer_implementation=",
		"writer_receipt=blocked",
		"receipt_action=reject",
		"receipt_persisted=false",
		"shadow_write_allowed=false",
		"writer_ready=false writer_impl=false ledger_impl=false rollback_impl=false",
		"contracts_ready=false write_allowed=false admission_allowed=false live_ready=false live_enabled=false mutates=false",
		"writer_receipt_id=",
		"passed=false",
		"reason=candidate_admission_writer_implementation_failed: candidate_admission_ledger_failed: candidate_admission_writer_contract_failed: candidate_admission_writer_inventory_failed: candidate_admission_writer_preflight_failed: candidate_admission_live_stage_failed: candidate_admission_enable_gate_failed: candidate_admission_switch_failed: candidate_admission_promotion_failed: candidate_admission_decision_failed: missing_candidate_execution",
	} {
		if !strings.Contains(receiptLine, want) {
			t.Fatalf("candidate admission writer receipt line missing %q: %q", want, receiptLine)
		}
	}
	raw, err := os.ReadFile(writerReceiptLog)
	if err != nil {
		t.Fatal(err)
	}
	var got admissionLiveRouteTurnCandidateAdmissionWriterReceipt
	if err := json.Unmarshal([]byte(strings.TrimSpace(string(raw))), &got); err != nil {
		t.Fatal(err)
	}
	if got.Schema != admissionLiveRouteTurnCandidateAdmissionWriterReceiptSchema ||
		got.Passed ||
		got.LiveReady ||
		got.LiveAdmissionEnabled ||
		got.AdmissionAllowed ||
		!got.ManualEnableRequested ||
		!got.EnableKeyMatched ||
		got.WriterReceiptState != "blocked" ||
		got.WriterReceiptAction != "reject" ||
		got.WriterReceiptPersisted ||
		got.ShadowWriteAllowed ||
		got.WriterReady ||
		got.WriterImplementationReady ||
		got.ContractsReady ||
		got.WriteAllowed ||
		got.MutatesState ||
		got.SourceWriterImplementationPassed ||
		got.WriterReceiptID != "" ||
		got.Reason != "candidate_admission_writer_implementation_failed: candidate_admission_ledger_failed: candidate_admission_writer_contract_failed: candidate_admission_writer_inventory_failed: candidate_admission_writer_preflight_failed: candidate_admission_live_stage_failed: candidate_admission_enable_gate_failed: candidate_admission_switch_failed: candidate_admission_promotion_failed: candidate_admission_decision_failed: missing_candidate_execution" {
		t.Fatalf("bad candidate admission writer receipt: %+v", got)
	}
}

func TestChatLiveRouteTurnCandidateAdmissionRollbackImplementationDryRunLine(t *testing.T) {
	t.Setenv("AM_LIVE_ROUTE_TURN_GENERATOR_ADAPTER_DRY_RUN", "1")
	t.Setenv("AM_LIVE_ROUTE_TURN_GENERATOR_ADAPTER_TEXT", "The dream proves rollback before it touches the body.")
	t.Setenv("AM_LIVE_ROUTE_TURN_CANDIDATE_DRAFT_DRY_RUN", "1")
	t.Setenv("AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_DRY_RUN", "1")
	t.Setenv("AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_ADAPTER_DRY_RUN", "1")
	t.Setenv("AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_SHADOW_DRY_RUN", "1")
	t.Setenv("AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_DECISION_DRY_RUN", "1")
	t.Setenv("AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_PROMOTION_DRY_RUN", "1")
	t.Setenv("AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_SWITCH_DRY_RUN", "1")
	t.Setenv("AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_ENABLE_GATE_DRY_RUN", "1")
	t.Setenv("AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_LIVE_STAGE_DRY_RUN", "1")
	t.Setenv("AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_WRITER_PREFLIGHT_DRY_RUN", "1")
	t.Setenv("AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_WRITER_INVENTORY_DRY_RUN", "1")
	t.Setenv("AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_WRITER_CONTRACT_DRY_RUN", "1")
	t.Setenv("AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_LEDGER_DRY_RUN", "1")
	t.Setenv("AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_WRITER_IMPLEMENTATION_DRY_RUN", "1")
	t.Setenv("AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_WRITER_RECEIPT_DRY_RUN", "1")
	t.Setenv("AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_ROLLBACK_IMPLEMENTATION_DRY_RUN", "1")
	t.Setenv("AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_ENABLE_GATE_KEY", admissionLiveRouteTurnCandidateAdmissionEnableGateConfirmation)
	t.Setenv("AM_DREAM_ADMISSION", dreamAdmissionShadow)
	t.Setenv("AM_DREAM_ADMISSION_REQUIRE_LIVE_ROUTE_PLAN", "1")
	dir := t.TempDir()
	dreamLog := filepath.Join(dir, "dream-admission-chat-rollback-implementation.jsonl")
	decisionLog := filepath.Join(dir, "live-route-candidate-admission-decision.jsonl")
	promotionLog := filepath.Join(dir, "live-route-candidate-admission-promotion.jsonl")
	switchLog := filepath.Join(dir, "live-route-candidate-admission-switch.jsonl")
	enableGateLog := filepath.Join(dir, "live-route-candidate-admission-enable-gate.jsonl")
	liveStageLog := filepath.Join(dir, "live-route-candidate-admission-live-stage.jsonl")
	writerPreflightLog := filepath.Join(dir, "live-route-candidate-admission-writer-preflight.jsonl")
	writerInventoryLog := filepath.Join(dir, "live-route-candidate-admission-writer-inventory.jsonl")
	writerContractLog := filepath.Join(dir, "live-route-candidate-admission-writer-contract.jsonl")
	ledgerLog := filepath.Join(dir, "live-route-candidate-admission-ledger.jsonl")
	writerImplLog := filepath.Join(dir, "live-route-candidate-admission-writer-implementation.jsonl")
	writerReceiptLog := filepath.Join(dir, "live-route-candidate-admission-writer-receipt.jsonl")
	rollbackImplLog := filepath.Join(dir, "live-route-candidate-admission-rollback-implementation.jsonl")
	t.Setenv("AM_DREAM_ADMISSION_LOG", dreamLog)
	t.Setenv("AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_DECISION_LOG", decisionLog)
	t.Setenv("AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_PROMOTION_LOG", promotionLog)
	t.Setenv("AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_SWITCH_LOG", switchLog)
	t.Setenv("AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_ENABLE_GATE_LOG", enableGateLog)
	t.Setenv("AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_LIVE_STAGE_LOG", liveStageLog)
	t.Setenv("AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_WRITER_PREFLIGHT_LOG", writerPreflightLog)
	t.Setenv("AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_WRITER_INVENTORY_LOG", writerInventoryLog)
	t.Setenv("AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_WRITER_CONTRACT_LOG", writerContractLog)
	t.Setenv("AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_LEDGER_LOG", ledgerLog)
	t.Setenv("AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_WRITER_IMPLEMENTATION_LOG", writerImplLog)
	t.Setenv("AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_WRITER_RECEIPT_LOG", writerReceiptLog)
	t.Setenv("AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_ROLLBACK_IMPLEMENTATION_LOG", rollbackImplLog)

	obs := admissionLiveRouteTurnObservationForHuman("Tell me what the dream should remember.")
	lines := chatLiveRouteTurnCandidateChainDryRunLines(obs)
	if len(lines) != 17 {
		t.Fatalf("expected 17 candidate chain lines, got %d: %v", len(lines), lines)
	}
	rollbackLine := lines[len(lines)-1]
	for _, want := range []string{
		"live-route candidate admission rollback implementation dry-run",
		"class=dream",
		"route=direct",
		"source=direct",
		"writer_receipt=",
		"rollback=blocked",
		"rollback_action=reject",
		"exact_match=false",
		"dry_run_only=true",
		"receipt_removed=false",
		"writer_ready=false rollback_ready=false writer_impl=false rollback_impl=false ledger_impl=false",
		"contracts_ready=false write_allowed=false admission_allowed=false live_ready=false live_enabled=false mutates=false",
		"rollback_implementation_id=",
		"passed=false",
		"reason=candidate_admission_writer_receipt_failed: candidate_admission_writer_implementation_failed: candidate_admission_ledger_failed: candidate_admission_writer_contract_failed: candidate_admission_writer_inventory_failed: candidate_admission_writer_preflight_failed: candidate_admission_live_stage_failed: candidate_admission_enable_gate_failed: candidate_admission_switch_failed: candidate_admission_promotion_failed: candidate_admission_decision_failed: missing_candidate_execution",
	} {
		if !strings.Contains(rollbackLine, want) {
			t.Fatalf("candidate admission rollback implementation line missing %q: %q", want, rollbackLine)
		}
	}
	raw, err := os.ReadFile(rollbackImplLog)
	if err != nil {
		t.Fatal(err)
	}
	var got admissionLiveRouteTurnCandidateAdmissionRollbackImplementation
	if err := json.Unmarshal([]byte(strings.TrimSpace(string(raw))), &got); err != nil {
		t.Fatal(err)
	}
	if got.Schema != admissionLiveRouteTurnCandidateAdmissionRollbackImplSchema ||
		got.Passed ||
		got.LiveReady ||
		got.LiveAdmissionEnabled ||
		got.AdmissionAllowed ||
		!got.ManualEnableRequested ||
		!got.EnableKeyMatched ||
		got.RollbackImplementationState != "blocked" ||
		got.RollbackImplementationAction != "reject" ||
		got.RollbackReady ||
		got.RollbackImplementationReady ||
		got.WriteAllowed ||
		got.MutatesState ||
		got.SourceWriterReceiptPassed ||
		got.RollbackImplementationID != "" ||
		got.Reason != "candidate_admission_writer_receipt_failed: candidate_admission_writer_implementation_failed: candidate_admission_ledger_failed: candidate_admission_writer_contract_failed: candidate_admission_writer_inventory_failed: candidate_admission_writer_preflight_failed: candidate_admission_live_stage_failed: candidate_admission_enable_gate_failed: candidate_admission_switch_failed: candidate_admission_promotion_failed: candidate_admission_decision_failed: missing_candidate_execution" {
		t.Fatalf("bad candidate admission rollback implementation: %+v", got)
	}
}

func TestChatLiveRouteTurnCandidateAdmissionLedgerImplementationDryRunLine(t *testing.T) {
	t.Setenv("AM_LIVE_ROUTE_TURN_GENERATOR_ADAPTER_DRY_RUN", "1")
	t.Setenv("AM_LIVE_ROUTE_TURN_GENERATOR_ADAPTER_TEXT", "The dream proves the ledger before it touches the body.")
	t.Setenv("AM_LIVE_ROUTE_TURN_CANDIDATE_DRAFT_DRY_RUN", "1")
	t.Setenv("AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_DRY_RUN", "1")
	t.Setenv("AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_ADAPTER_DRY_RUN", "1")
	t.Setenv("AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_SHADOW_DRY_RUN", "1")
	t.Setenv("AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_DECISION_DRY_RUN", "1")
	t.Setenv("AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_PROMOTION_DRY_RUN", "1")
	t.Setenv("AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_SWITCH_DRY_RUN", "1")
	t.Setenv("AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_ENABLE_GATE_DRY_RUN", "1")
	t.Setenv("AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_LIVE_STAGE_DRY_RUN", "1")
	t.Setenv("AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_WRITER_PREFLIGHT_DRY_RUN", "1")
	t.Setenv("AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_WRITER_INVENTORY_DRY_RUN", "1")
	t.Setenv("AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_WRITER_CONTRACT_DRY_RUN", "1")
	t.Setenv("AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_LEDGER_DRY_RUN", "1")
	t.Setenv("AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_WRITER_IMPLEMENTATION_DRY_RUN", "1")
	t.Setenv("AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_WRITER_RECEIPT_DRY_RUN", "1")
	t.Setenv("AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_ROLLBACK_IMPLEMENTATION_DRY_RUN", "1")
	t.Setenv("AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_LEDGER_IMPLEMENTATION_DRY_RUN", "1")
	t.Setenv("AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_ENABLE_GATE_KEY", admissionLiveRouteTurnCandidateAdmissionEnableGateConfirmation)
	t.Setenv("AM_DREAM_ADMISSION", dreamAdmissionShadow)
	t.Setenv("AM_DREAM_ADMISSION_REQUIRE_LIVE_ROUTE_PLAN", "1")
	dir := t.TempDir()
	dreamLog := filepath.Join(dir, "dream-admission-chat-ledger-implementation.jsonl")
	decisionLog := filepath.Join(dir, "live-route-candidate-admission-decision.jsonl")
	promotionLog := filepath.Join(dir, "live-route-candidate-admission-promotion.jsonl")
	switchLog := filepath.Join(dir, "live-route-candidate-admission-switch.jsonl")
	enableGateLog := filepath.Join(dir, "live-route-candidate-admission-enable-gate.jsonl")
	liveStageLog := filepath.Join(dir, "live-route-candidate-admission-live-stage.jsonl")
	writerPreflightLog := filepath.Join(dir, "live-route-candidate-admission-writer-preflight.jsonl")
	writerInventoryLog := filepath.Join(dir, "live-route-candidate-admission-writer-inventory.jsonl")
	writerContractLog := filepath.Join(dir, "live-route-candidate-admission-writer-contract.jsonl")
	ledgerLog := filepath.Join(dir, "live-route-candidate-admission-ledger.jsonl")
	writerImplLog := filepath.Join(dir, "live-route-candidate-admission-writer-implementation.jsonl")
	writerReceiptLog := filepath.Join(dir, "live-route-candidate-admission-writer-receipt.jsonl")
	rollbackImplLog := filepath.Join(dir, "live-route-candidate-admission-rollback-implementation.jsonl")
	ledgerImplLog := filepath.Join(dir, "live-route-candidate-admission-ledger-implementation.jsonl")
	t.Setenv("AM_DREAM_ADMISSION_LOG", dreamLog)
	t.Setenv("AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_DECISION_LOG", decisionLog)
	t.Setenv("AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_PROMOTION_LOG", promotionLog)
	t.Setenv("AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_SWITCH_LOG", switchLog)
	t.Setenv("AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_ENABLE_GATE_LOG", enableGateLog)
	t.Setenv("AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_LIVE_STAGE_LOG", liveStageLog)
	t.Setenv("AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_WRITER_PREFLIGHT_LOG", writerPreflightLog)
	t.Setenv("AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_WRITER_INVENTORY_LOG", writerInventoryLog)
	t.Setenv("AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_WRITER_CONTRACT_LOG", writerContractLog)
	t.Setenv("AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_LEDGER_LOG", ledgerLog)
	t.Setenv("AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_WRITER_IMPLEMENTATION_LOG", writerImplLog)
	t.Setenv("AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_WRITER_RECEIPT_LOG", writerReceiptLog)
	t.Setenv("AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_ROLLBACK_IMPLEMENTATION_LOG", rollbackImplLog)
	t.Setenv("AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_LEDGER_IMPLEMENTATION_LOG", ledgerImplLog)

	obs := admissionLiveRouteTurnObservationForHuman("Tell me what the ledger should remember.")
	lines := chatLiveRouteTurnCandidateChainDryRunLines(obs)
	if len(lines) != 18 {
		t.Fatalf("expected 18 candidate chain lines, got %d: %v", len(lines), lines)
	}
	ledgerLine := lines[len(lines)-1]
	for _, want := range []string{
		"live-route candidate admission ledger implementation dry-run",
		"class=direct-user",
		"route=user_bridge",
		"source=user_bridge",
		"rollback_implementation=",
		"ledger=blocked",
		"ledger_action=reject",
		"append_only=false",
		"dry_run_only=true",
		"receipt_persisted=false",
		"writer_ready=false rollback_ready=false writer_impl=false rollback_impl=false ledger_impl=false",
		"contracts_ready=false write_allowed=false admission_allowed=false live_ready=false live_enabled=false mutates=false",
		"ledger_implementation_id=",
		"passed=false",
		"reason=candidate_admission_rollback_implementation_failed: candidate_admission_writer_receipt_failed: candidate_admission_writer_implementation_failed: candidate_admission_ledger_failed: candidate_admission_writer_contract_failed: candidate_admission_writer_inventory_failed: candidate_admission_writer_preflight_failed: candidate_admission_live_stage_failed: candidate_admission_enable_gate_failed: candidate_admission_switch_failed: candidate_admission_promotion_failed: candidate_admission_decision_failed: missing_candidate_execution",
	} {
		if !strings.Contains(ledgerLine, want) {
			t.Fatalf("candidate admission ledger implementation line missing %q: %q", want, ledgerLine)
		}
	}
	raw, err := os.ReadFile(ledgerImplLog)
	if err != nil {
		t.Fatal(err)
	}
	var got admissionLiveRouteTurnCandidateAdmissionLedgerImplementation
	if err := json.Unmarshal([]byte(strings.TrimSpace(string(raw))), &got); err != nil {
		t.Fatal(err)
	}
	if got.Schema != admissionLiveRouteTurnCandidateAdmissionLedgerImplSchema ||
		got.Passed ||
		got.LiveReady ||
		got.LiveAdmissionEnabled ||
		got.AdmissionAllowed ||
		!got.ManualEnableRequested ||
		!got.EnableKeyMatched ||
		got.LedgerImplementationState != "blocked" ||
		got.LedgerImplementationAction != "reject" ||
		got.LedgerImplementationAppendOnly ||
		!got.LedgerImplementationDryRunOnly ||
		got.LedgerImplementationReceiptPersisted ||
		got.LedgerImplementationReady ||
		got.WriteAllowed ||
		got.MutatesState ||
		got.SourceRollbackImplementationPassed ||
		got.LedgerImplementationID != "" ||
		got.Reason != "candidate_admission_rollback_implementation_failed: candidate_admission_writer_receipt_failed: candidate_admission_writer_implementation_failed: candidate_admission_ledger_failed: candidate_admission_writer_contract_failed: candidate_admission_writer_inventory_failed: candidate_admission_writer_preflight_failed: candidate_admission_live_stage_failed: candidate_admission_enable_gate_failed: candidate_admission_switch_failed: candidate_admission_promotion_failed: candidate_admission_decision_failed: missing_candidate_execution" {
		t.Fatalf("bad candidate admission ledger implementation: %+v", got)
	}
}

func TestChatLiveRouteTurnCandidateAdmissionLedgerPersistenceDryRunLine(t *testing.T) {
	t.Setenv("AM_LIVE_ROUTE_TURN_GENERATOR_ADAPTER_DRY_RUN", "1")
	t.Setenv("AM_LIVE_ROUTE_TURN_GENERATOR_ADAPTER_TEXT", "The ledger persists only after the contract is proven.")
	t.Setenv("AM_LIVE_ROUTE_TURN_CANDIDATE_DRAFT_DRY_RUN", "1")
	t.Setenv("AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_DRY_RUN", "1")
	t.Setenv("AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_ADAPTER_DRY_RUN", "1")
	t.Setenv("AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_SHADOW_DRY_RUN", "1")
	t.Setenv("AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_DECISION_DRY_RUN", "1")
	t.Setenv("AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_PROMOTION_DRY_RUN", "1")
	t.Setenv("AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_SWITCH_DRY_RUN", "1")
	t.Setenv("AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_ENABLE_GATE_DRY_RUN", "1")
	t.Setenv("AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_LIVE_STAGE_DRY_RUN", "1")
	t.Setenv("AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_WRITER_PREFLIGHT_DRY_RUN", "1")
	t.Setenv("AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_WRITER_INVENTORY_DRY_RUN", "1")
	t.Setenv("AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_WRITER_CONTRACT_DRY_RUN", "1")
	t.Setenv("AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_LEDGER_DRY_RUN", "1")
	t.Setenv("AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_WRITER_IMPLEMENTATION_DRY_RUN", "1")
	t.Setenv("AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_WRITER_RECEIPT_DRY_RUN", "1")
	t.Setenv("AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_ROLLBACK_IMPLEMENTATION_DRY_RUN", "1")
	t.Setenv("AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_LEDGER_IMPLEMENTATION_DRY_RUN", "1")
	t.Setenv("AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_LEDGER_PERSISTENCE_DRY_RUN", "1")
	t.Setenv("AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_ENABLE_GATE_KEY", admissionLiveRouteTurnCandidateAdmissionEnableGateConfirmation)
	t.Setenv("AM_DREAM_ADMISSION", dreamAdmissionShadow)
	t.Setenv("AM_DREAM_ADMISSION_REQUIRE_LIVE_ROUTE_PLAN", "1")
	dir := t.TempDir()
	dreamLog := filepath.Join(dir, "dream-admission-chat-ledger-persistence.jsonl")
	decisionLog := filepath.Join(dir, "live-route-candidate-admission-decision.jsonl")
	promotionLog := filepath.Join(dir, "live-route-candidate-admission-promotion.jsonl")
	switchLog := filepath.Join(dir, "live-route-candidate-admission-switch.jsonl")
	enableGateLog := filepath.Join(dir, "live-route-candidate-admission-enable-gate.jsonl")
	liveStageLog := filepath.Join(dir, "live-route-candidate-admission-live-stage.jsonl")
	writerPreflightLog := filepath.Join(dir, "live-route-candidate-admission-writer-preflight.jsonl")
	writerInventoryLog := filepath.Join(dir, "live-route-candidate-admission-writer-inventory.jsonl")
	writerContractLog := filepath.Join(dir, "live-route-candidate-admission-writer-contract.jsonl")
	ledgerLog := filepath.Join(dir, "live-route-candidate-admission-ledger.jsonl")
	writerImplLog := filepath.Join(dir, "live-route-candidate-admission-writer-implementation.jsonl")
	writerReceiptLog := filepath.Join(dir, "live-route-candidate-admission-writer-receipt.jsonl")
	rollbackImplLog := filepath.Join(dir, "live-route-candidate-admission-rollback-implementation.jsonl")
	ledgerImplLog := filepath.Join(dir, "live-route-candidate-admission-ledger-implementation.jsonl")
	ledgerPersistenceLog := filepath.Join(dir, "live-route-candidate-admission-ledger-persistence.jsonl")
	t.Setenv("AM_DREAM_ADMISSION_LOG", dreamLog)
	t.Setenv("AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_DECISION_LOG", decisionLog)
	t.Setenv("AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_PROMOTION_LOG", promotionLog)
	t.Setenv("AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_SWITCH_LOG", switchLog)
	t.Setenv("AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_ENABLE_GATE_LOG", enableGateLog)
	t.Setenv("AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_LIVE_STAGE_LOG", liveStageLog)
	t.Setenv("AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_WRITER_PREFLIGHT_LOG", writerPreflightLog)
	t.Setenv("AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_WRITER_INVENTORY_LOG", writerInventoryLog)
	t.Setenv("AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_WRITER_CONTRACT_LOG", writerContractLog)
	t.Setenv("AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_LEDGER_LOG", ledgerLog)
	t.Setenv("AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_WRITER_IMPLEMENTATION_LOG", writerImplLog)
	t.Setenv("AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_WRITER_RECEIPT_LOG", writerReceiptLog)
	t.Setenv("AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_ROLLBACK_IMPLEMENTATION_LOG", rollbackImplLog)
	t.Setenv("AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_LEDGER_IMPLEMENTATION_LOG", ledgerImplLog)
	t.Setenv("AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_LEDGER_PERSISTENCE_LOG", ledgerPersistenceLog)

	obs := admissionLiveRouteTurnObservationForHuman("Tell me what the ledger should remember.")
	lines := chatLiveRouteTurnCandidateChainDryRunLines(obs)
	if len(lines) != 19 {
		t.Fatalf("expected 19 candidate chain lines, got %d: %v", len(lines), lines)
	}
	persistenceLine := lines[len(lines)-1]
	for _, want := range []string{
		"live-route candidate admission ledger persistence dry-run",
		"class=direct-user",
		"route=user_bridge",
		"source=user_bridge",
		"ledger_implementation=",
		"persistence=blocked",
		"persistence_action=reject",
		"append_only=false",
		"dry_run_only=true",
		"receipt_persisted=false",
		"persistence_ready=false",
		"writer_ready=false rollback_ready=false writer_impl=false rollback_impl=false ledger_impl=false",
		"contracts_ready=false write_allowed=false admission_allowed=false live_ready=false live_enabled=false mutates=false",
		"ledger_persistence_id=",
		"passed=false",
		"reason=candidate_admission_ledger_implementation_failed: candidate_admission_rollback_implementation_failed: candidate_admission_writer_receipt_failed: candidate_admission_writer_implementation_failed: candidate_admission_ledger_failed: candidate_admission_writer_contract_failed: candidate_admission_writer_inventory_failed: candidate_admission_writer_preflight_failed: candidate_admission_live_stage_failed: candidate_admission_enable_gate_failed: candidate_admission_switch_failed: candidate_admission_promotion_failed: candidate_admission_decision_failed: missing_candidate_execution",
	} {
		if !strings.Contains(persistenceLine, want) {
			t.Fatalf("candidate admission ledger persistence line missing %q: %q", want, persistenceLine)
		}
	}
	raw, err := os.ReadFile(ledgerPersistenceLog)
	if err != nil {
		t.Fatal(err)
	}
	var got admissionLiveRouteTurnCandidateAdmissionLedgerPersistence
	if err := json.Unmarshal([]byte(strings.TrimSpace(string(raw))), &got); err != nil {
		t.Fatal(err)
	}
	if got.Schema != admissionLiveRouteTurnCandidateAdmissionLedgerPersistenceSchema ||
		got.Passed ||
		got.LiveReady ||
		got.LiveAdmissionEnabled ||
		got.AdmissionAllowed ||
		!got.ManualEnableRequested ||
		!got.EnableKeyMatched ||
		got.LedgerPersistenceState != "blocked" ||
		got.LedgerPersistenceAction != "reject" ||
		got.LedgerPersistenceAppendOnly ||
		!got.LedgerPersistenceDryRunOnly ||
		got.LedgerPersistenceReceiptPersisted ||
		got.LedgerPersistenceReady ||
		got.WriteAllowed ||
		got.MutatesState ||
		got.SourceLedgerImplementationPassed ||
		got.LedgerPersistenceID != "" ||
		got.Reason != "candidate_admission_ledger_implementation_failed: candidate_admission_rollback_implementation_failed: candidate_admission_writer_receipt_failed: candidate_admission_writer_implementation_failed: candidate_admission_ledger_failed: candidate_admission_writer_contract_failed: candidate_admission_writer_inventory_failed: candidate_admission_writer_preflight_failed: candidate_admission_live_stage_failed: candidate_admission_enable_gate_failed: candidate_admission_switch_failed: candidate_admission_promotion_failed: candidate_admission_decision_failed: missing_candidate_execution" {
		t.Fatalf("bad candidate admission ledger persistence: %+v", got)
	}
}

func TestChatLiveRouteTurnCandidateAdmissionLedgerVerificationDryRunLine(t *testing.T) {
	t.Setenv("AM_LIVE_ROUTE_TURN_GENERATOR_ADAPTER_DRY_RUN", "1")
	t.Setenv("AM_LIVE_ROUTE_TURN_GENERATOR_ADAPTER_TEXT", "The ledger verifies only after the persisted receipt is read back.")
	t.Setenv("AM_LIVE_ROUTE_TURN_CANDIDATE_DRAFT_DRY_RUN", "1")
	t.Setenv("AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_DRY_RUN", "1")
	t.Setenv("AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_ADAPTER_DRY_RUN", "1")
	t.Setenv("AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_SHADOW_DRY_RUN", "1")
	t.Setenv("AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_DECISION_DRY_RUN", "1")
	t.Setenv("AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_PROMOTION_DRY_RUN", "1")
	t.Setenv("AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_SWITCH_DRY_RUN", "1")
	t.Setenv("AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_ENABLE_GATE_DRY_RUN", "1")
	t.Setenv("AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_LIVE_STAGE_DRY_RUN", "1")
	t.Setenv("AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_WRITER_PREFLIGHT_DRY_RUN", "1")
	t.Setenv("AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_WRITER_INVENTORY_DRY_RUN", "1")
	t.Setenv("AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_WRITER_CONTRACT_DRY_RUN", "1")
	t.Setenv("AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_LEDGER_DRY_RUN", "1")
	t.Setenv("AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_WRITER_IMPLEMENTATION_DRY_RUN", "1")
	t.Setenv("AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_WRITER_RECEIPT_DRY_RUN", "1")
	t.Setenv("AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_ROLLBACK_IMPLEMENTATION_DRY_RUN", "1")
	t.Setenv("AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_LEDGER_IMPLEMENTATION_DRY_RUN", "1")
	t.Setenv("AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_LEDGER_PERSISTENCE_DRY_RUN", "1")
	t.Setenv("AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_LEDGER_VERIFICATION_DRY_RUN", "1")
	t.Setenv("AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_READINESS_DRY_RUN", "1")
	t.Setenv("AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_PERMIT_DRY_RUN", "1")
	t.Setenv("AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_SEAL_DRY_RUN", "1")
	t.Setenv("AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_FINAL_GATE_DRY_RUN", "1")
	t.Setenv("AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_RESONANCE_INTENT_DRY_RUN", "1")
	t.Setenv("AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_RESONANCE_RECEIVER_DRY_RUN", "1")
	t.Setenv("AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_RESONANCE_OBSERVATION_DRY_RUN", "1")
	t.Setenv("AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_RESONANCE_GRAFT_BOUNDARY_DRY_RUN", "1")
	t.Setenv("AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_ENABLE_GATE_KEY", admissionLiveRouteTurnCandidateAdmissionEnableGateConfirmation)
	t.Setenv("AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_PERMIT_KEY", admissionLiveRouteTurnCandidateAdmissionPermitConfirmation)
	t.Setenv("AM_DREAM_ADMISSION", dreamAdmissionShadow)
	t.Setenv("AM_DREAM_ADMISSION_REQUIRE_LIVE_ROUTE_PLAN", "1")
	dir := t.TempDir()
	dreamLog := filepath.Join(dir, "dream-admission-chat-ledger-verification.jsonl")
	decisionLog := filepath.Join(dir, "live-route-candidate-admission-decision.jsonl")
	promotionLog := filepath.Join(dir, "live-route-candidate-admission-promotion.jsonl")
	switchLog := filepath.Join(dir, "live-route-candidate-admission-switch.jsonl")
	enableGateLog := filepath.Join(dir, "live-route-candidate-admission-enable-gate.jsonl")
	liveStageLog := filepath.Join(dir, "live-route-candidate-admission-live-stage.jsonl")
	writerPreflightLog := filepath.Join(dir, "live-route-candidate-admission-writer-preflight.jsonl")
	writerInventoryLog := filepath.Join(dir, "live-route-candidate-admission-writer-inventory.jsonl")
	writerContractLog := filepath.Join(dir, "live-route-candidate-admission-writer-contract.jsonl")
	ledgerLog := filepath.Join(dir, "live-route-candidate-admission-ledger.jsonl")
	writerImplLog := filepath.Join(dir, "live-route-candidate-admission-writer-implementation.jsonl")
	writerReceiptLog := filepath.Join(dir, "live-route-candidate-admission-writer-receipt.jsonl")
	rollbackImplLog := filepath.Join(dir, "live-route-candidate-admission-rollback-implementation.jsonl")
	ledgerImplLog := filepath.Join(dir, "live-route-candidate-admission-ledger-implementation.jsonl")
	ledgerPersistenceLog := filepath.Join(dir, "live-route-candidate-admission-ledger-persistence.jsonl")
	ledgerVerificationLog := filepath.Join(dir, "live-route-candidate-admission-ledger-verification.jsonl")
	readinessLog := filepath.Join(dir, "live-route-candidate-admission-readiness.jsonl")
	permitLog := filepath.Join(dir, "live-route-candidate-admission-permit.jsonl")
	sealLog := filepath.Join(dir, "live-route-candidate-admission-seal.jsonl")
	finalGateLog := filepath.Join(dir, "live-route-candidate-admission-final-gate.jsonl")
	resonanceIntentLog := filepath.Join(dir, "live-route-candidate-admission-resonance-intent.jsonl")
	resonanceReceiverLog := filepath.Join(dir, "live-route-candidate-admission-resonance-receiver.jsonl")
	resonanceObservationLog := filepath.Join(dir, "live-route-candidate-admission-resonance-observation.jsonl")
	resonanceGraftBoundaryLog := filepath.Join(dir, "live-route-candidate-admission-resonance-graft-boundary.jsonl")
	t.Setenv("AM_DREAM_ADMISSION_LOG", dreamLog)
	t.Setenv("AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_DECISION_LOG", decisionLog)
	t.Setenv("AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_PROMOTION_LOG", promotionLog)
	t.Setenv("AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_SWITCH_LOG", switchLog)
	t.Setenv("AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_ENABLE_GATE_LOG", enableGateLog)
	t.Setenv("AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_LIVE_STAGE_LOG", liveStageLog)
	t.Setenv("AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_WRITER_PREFLIGHT_LOG", writerPreflightLog)
	t.Setenv("AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_WRITER_INVENTORY_LOG", writerInventoryLog)
	t.Setenv("AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_WRITER_CONTRACT_LOG", writerContractLog)
	t.Setenv("AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_LEDGER_LOG", ledgerLog)
	t.Setenv("AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_WRITER_IMPLEMENTATION_LOG", writerImplLog)
	t.Setenv("AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_WRITER_RECEIPT_LOG", writerReceiptLog)
	t.Setenv("AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_ROLLBACK_IMPLEMENTATION_LOG", rollbackImplLog)
	t.Setenv("AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_LEDGER_IMPLEMENTATION_LOG", ledgerImplLog)
	t.Setenv("AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_LEDGER_PERSISTENCE_LOG", ledgerPersistenceLog)
	t.Setenv("AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_LEDGER_VERIFICATION_LOG", ledgerVerificationLog)
	t.Setenv("AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_READINESS_LOG", readinessLog)
	t.Setenv("AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_PERMIT_LOG", permitLog)
	t.Setenv("AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_SEAL_LOG", sealLog)
	t.Setenv("AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_FINAL_GATE_LOG", finalGateLog)
	t.Setenv("AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_RESONANCE_INTENT_LOG", resonanceIntentLog)
	t.Setenv("AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_RESONANCE_RECEIVER_LOG", resonanceReceiverLog)
	t.Setenv("AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_RESONANCE_OBSERVATION_LOG", resonanceObservationLog)
	t.Setenv("AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_RESONANCE_GRAFT_BOUNDARY_LOG", resonanceGraftBoundaryLog)

	obs := admissionLiveRouteTurnObservationForHuman("Tell me what the ledger should verify.")
	lines := chatLiveRouteTurnCandidateChainDryRunLines(obs)
	if len(lines) != 28 {
		t.Fatalf("expected 28 candidate chain lines, got %d: %v", len(lines), lines)
	}
	verificationLine := lines[len(lines)-9]
	for _, want := range []string{
		"live-route candidate admission ledger verification dry-run",
		"class=direct-user",
		"route=user_bridge",
		"source=user_bridge",
		"ledger_persistence=",
		"verification=blocked",
		"verification_action=reject",
		"append_only=false",
		"dry_run_only=true",
		"read_back=false",
		"receipt_verified=false",
		"verification_ready=false",
		"persistence_ready=false",
		"writer_ready=false rollback_ready=false writer_impl=false rollback_impl=false ledger_impl=false",
		"contracts_ready=false write_allowed=false admission_allowed=false live_ready=false live_enabled=false mutates=false",
		"ledger_verification_id=",
		"passed=false",
		"reason=candidate_admission_ledger_persistence_failed: candidate_admission_ledger_implementation_failed: candidate_admission_rollback_implementation_failed: candidate_admission_writer_receipt_failed: candidate_admission_writer_implementation_failed: candidate_admission_ledger_failed: candidate_admission_writer_contract_failed: candidate_admission_writer_inventory_failed: candidate_admission_writer_preflight_failed: candidate_admission_live_stage_failed: candidate_admission_enable_gate_failed: candidate_admission_switch_failed: candidate_admission_promotion_failed: candidate_admission_decision_failed: missing_candidate_execution",
	} {
		if !strings.Contains(verificationLine, want) {
			t.Fatalf("candidate admission ledger verification line missing %q: %q", want, verificationLine)
		}
	}
	raw, err := os.ReadFile(ledgerVerificationLog)
	if err != nil {
		t.Fatal(err)
	}
	var got admissionLiveRouteTurnCandidateAdmissionLedgerVerification
	if err := json.Unmarshal([]byte(strings.TrimSpace(string(raw))), &got); err != nil {
		t.Fatal(err)
	}
	if got.Schema != admissionLiveRouteTurnCandidateAdmissionLedgerVerificationSchema ||
		got.Passed ||
		got.LiveReady ||
		got.LiveAdmissionEnabled ||
		got.AdmissionAllowed ||
		!got.ManualEnableRequested ||
		!got.EnableKeyMatched ||
		got.LedgerVerificationState != "blocked" ||
		got.LedgerVerificationAction != "reject" ||
		got.LedgerVerificationAppendOnly ||
		!got.LedgerVerificationDryRunOnly ||
		got.LedgerVerificationReceiptReadBack ||
		got.LedgerVerificationReceiptVerified ||
		got.LedgerVerificationReady ||
		got.WriteAllowed ||
		got.MutatesState ||
		got.SourceLedgerPersistencePassed ||
		got.LedgerVerificationID != "" ||
		got.Reason != "candidate_admission_ledger_persistence_failed: candidate_admission_ledger_implementation_failed: candidate_admission_rollback_implementation_failed: candidate_admission_writer_receipt_failed: candidate_admission_writer_implementation_failed: candidate_admission_ledger_failed: candidate_admission_writer_contract_failed: candidate_admission_writer_inventory_failed: candidate_admission_writer_preflight_failed: candidate_admission_live_stage_failed: candidate_admission_enable_gate_failed: candidate_admission_switch_failed: candidate_admission_promotion_failed: candidate_admission_decision_failed: missing_candidate_execution" {
		t.Fatalf("bad candidate admission ledger verification: %+v", got)
	}
	readinessLine := lines[len(lines)-8]
	for _, want := range []string{
		"live-route candidate admission readiness dry-run",
		"class=direct-user",
		"route=user_bridge",
		"source=user_bridge",
		"ledger_verification=",
		"readiness=blocked",
		"readiness_action=reject",
		"dry_run_only=true",
		"ledger_verified=false",
		"writer_ready=false rollback_ready=false ledger_ready=false readiness_ready=false",
		"verification_ready=false persistence_ready=false writer_impl=false rollback_impl=false ledger_impl=false",
		"contracts_ready=false write_allowed=false admission_allowed=false live_ready=false live_enabled=false mutates=false",
		"admission_readiness_id=",
		"passed=false",
		"reason=candidate_admission_ledger_verification_failed: candidate_admission_ledger_persistence_failed: candidate_admission_ledger_implementation_failed: candidate_admission_rollback_implementation_failed: candidate_admission_writer_receipt_failed: candidate_admission_writer_implementation_failed: candidate_admission_ledger_failed: candidate_admission_writer_contract_failed: candidate_admission_writer_inventory_failed: candidate_admission_writer_preflight_failed: candidate_admission_live_stage_failed: candidate_admission_enable_gate_failed: candidate_admission_switch_failed: candidate_admission_promotion_failed: candidate_admission_decision_failed: missing_candidate_execution",
	} {
		if !strings.Contains(readinessLine, want) {
			t.Fatalf("candidate admission readiness line missing %q: %q", want, readinessLine)
		}
	}
	raw, err = os.ReadFile(readinessLog)
	if err != nil {
		t.Fatal(err)
	}
	var readiness admissionLiveRouteTurnCandidateAdmissionReadiness
	if err := json.Unmarshal([]byte(strings.TrimSpace(string(raw))), &readiness); err != nil {
		t.Fatal(err)
	}
	if readiness.Schema != admissionLiveRouteTurnCandidateAdmissionReadinessSchema ||
		readiness.Passed ||
		readiness.LiveReady ||
		readiness.LiveAdmissionEnabled ||
		readiness.AdmissionAllowed ||
		readiness.AdmissionReadinessState != "blocked" ||
		readiness.AdmissionReadinessAction != "reject" ||
		!readiness.AdmissionReadinessDryRunOnly ||
		readiness.AdmissionReadinessLedgerVerified ||
		readiness.AdmissionReadinessWriterReady ||
		readiness.AdmissionReadinessRollbackReady ||
		readiness.AdmissionReadinessLedgerReady ||
		readiness.AdmissionReadinessReady ||
		readiness.WriteAllowed ||
		readiness.MutatesState ||
		readiness.SourceLedgerVerificationPassed ||
		readiness.AdmissionReadinessID != "" ||
		readiness.Reason != "candidate_admission_ledger_verification_failed: candidate_admission_ledger_persistence_failed: candidate_admission_ledger_implementation_failed: candidate_admission_rollback_implementation_failed: candidate_admission_writer_receipt_failed: candidate_admission_writer_implementation_failed: candidate_admission_ledger_failed: candidate_admission_writer_contract_failed: candidate_admission_writer_inventory_failed: candidate_admission_writer_preflight_failed: candidate_admission_live_stage_failed: candidate_admission_enable_gate_failed: candidate_admission_switch_failed: candidate_admission_promotion_failed: candidate_admission_decision_failed: missing_candidate_execution" {
		t.Fatalf("bad candidate admission readiness: %+v", readiness)
	}
	permitLine := lines[len(lines)-7]
	for _, want := range []string{
		"live-route candidate admission permit dry-run",
		"class=direct-user",
		"route=user_bridge",
		"source=user_bridge",
		"readiness=",
		"permit=blocked",
		"permit_action=reject",
		"dry_run_only=true",
		"readiness_verified=false",
		"ledger_verified=false",
		"writer_ready=false rollback_ready=false ledger_ready=false permit_ready=false",
		"manual_requested=true key_matched=true",
		"readiness_ready=false verification_ready=false persistence_ready=false writer_impl=false rollback_impl=false ledger_impl=false",
		"contracts_ready=false write_allowed=false admission_allowed=false live_ready=false live_enabled=false mutates=false",
		"admission_permit_id=",
		"passed=false",
		"reason=candidate_admission_readiness_failed: candidate_admission_ledger_verification_failed: candidate_admission_ledger_persistence_failed: candidate_admission_ledger_implementation_failed: candidate_admission_rollback_implementation_failed: candidate_admission_writer_receipt_failed: candidate_admission_writer_implementation_failed: candidate_admission_ledger_failed: candidate_admission_writer_contract_failed: candidate_admission_writer_inventory_failed: candidate_admission_writer_preflight_failed: candidate_admission_live_stage_failed: candidate_admission_enable_gate_failed: candidate_admission_switch_failed: candidate_admission_promotion_failed: candidate_admission_decision_failed: missing_candidate_execution",
	} {
		if !strings.Contains(permitLine, want) {
			t.Fatalf("candidate admission permit line missing %q: %q", want, permitLine)
		}
	}
	raw, err = os.ReadFile(permitLog)
	if err != nil {
		t.Fatal(err)
	}
	var permit admissionLiveRouteTurnCandidateAdmissionPermit
	if err := json.Unmarshal([]byte(strings.TrimSpace(string(raw))), &permit); err != nil {
		t.Fatal(err)
	}
	if permit.Schema != admissionLiveRouteTurnCandidateAdmissionPermitSchema ||
		permit.Passed ||
		permit.LiveReady ||
		permit.LiveAdmissionEnabled ||
		permit.AdmissionAllowed ||
		permit.AdmissionPermitState != "blocked" ||
		permit.AdmissionPermitAction != "reject" ||
		!permit.AdmissionPermitDryRunOnly ||
		permit.AdmissionPermitReadinessVerified ||
		permit.AdmissionPermitLedgerVerified ||
		permit.AdmissionPermitWriterReady ||
		permit.AdmissionPermitRollbackReady ||
		permit.AdmissionPermitLedgerReady ||
		permit.AdmissionPermitReady ||
		!permit.ManualPermitRequested ||
		!permit.PermitKeyMatched ||
		permit.WriteAllowed ||
		permit.MutatesState ||
		permit.SourceAdmissionReadinessPassed ||
		permit.AdmissionPermitID != "" ||
		permit.Reason != "candidate_admission_readiness_failed: candidate_admission_ledger_verification_failed: candidate_admission_ledger_persistence_failed: candidate_admission_ledger_implementation_failed: candidate_admission_rollback_implementation_failed: candidate_admission_writer_receipt_failed: candidate_admission_writer_implementation_failed: candidate_admission_ledger_failed: candidate_admission_writer_contract_failed: candidate_admission_writer_inventory_failed: candidate_admission_writer_preflight_failed: candidate_admission_live_stage_failed: candidate_admission_enable_gate_failed: candidate_admission_switch_failed: candidate_admission_promotion_failed: candidate_admission_decision_failed: missing_candidate_execution" {
		t.Fatalf("bad candidate admission permit: %+v", permit)
	}
	sealLine := lines[len(lines)-6]
	for _, want := range []string{
		"live-route candidate admission seal dry-run",
		"class=direct-user",
		"route=user_bridge",
		"source=user_bridge",
		"permit=",
		"readiness=",
		"seal=blocked",
		"seal_action=reject",
		"dry_run_only=true",
		"permit_verified=false",
		"readiness_verified=false",
		"ledger_verified=false",
		"writer_ready=false rollback_ready=false ledger_ready=false seal_ready=false",
		"permit_ready=false key_matched=true",
		"readiness_ready=false verification_ready=false persistence_ready=false writer_impl=false rollback_impl=false ledger_impl=false",
		"contracts_ready=false write_allowed=false admission_allowed=false live_ready=false live_enabled=false mutates=false",
		"admission_seal_id=",
		"passed=false",
		"reason=candidate_admission_permit_failed: candidate_admission_readiness_failed: candidate_admission_ledger_verification_failed: candidate_admission_ledger_persistence_failed: candidate_admission_ledger_implementation_failed: candidate_admission_rollback_implementation_failed: candidate_admission_writer_receipt_failed: candidate_admission_writer_implementation_failed: candidate_admission_ledger_failed: candidate_admission_writer_contract_failed: candidate_admission_writer_inventory_failed: candidate_admission_writer_preflight_failed: candidate_admission_live_stage_failed: candidate_admission_enable_gate_failed: candidate_admission_switch_failed: candidate_admission_promotion_failed: candidate_admission_decision_failed: missing_candidate_execution",
	} {
		if !strings.Contains(sealLine, want) {
			t.Fatalf("candidate admission seal line missing %q: %q", want, sealLine)
		}
	}
	raw, err = os.ReadFile(sealLog)
	if err != nil {
		t.Fatal(err)
	}
	var seal admissionLiveRouteTurnCandidateAdmissionSeal
	if err := json.Unmarshal([]byte(strings.TrimSpace(string(raw))), &seal); err != nil {
		t.Fatal(err)
	}
	if seal.Schema != admissionLiveRouteTurnCandidateAdmissionSealSchema ||
		seal.Passed ||
		seal.LiveReady ||
		seal.LiveAdmissionEnabled ||
		seal.AdmissionAllowed ||
		seal.AdmissionSealState != "blocked" ||
		seal.AdmissionSealAction != "reject" ||
		!seal.AdmissionSealDryRunOnly ||
		seal.AdmissionSealPermitVerified ||
		seal.AdmissionSealReadinessVerified ||
		seal.AdmissionSealLedgerVerified ||
		seal.AdmissionSealWriterReady ||
		seal.AdmissionSealRollbackReady ||
		seal.AdmissionSealLedgerReady ||
		seal.AdmissionSealReady ||
		seal.WriteAllowed ||
		seal.MutatesState ||
		seal.SourceAdmissionPermitPassed ||
		seal.AdmissionSealID != "" ||
		seal.Reason != "candidate_admission_permit_failed: candidate_admission_readiness_failed: candidate_admission_ledger_verification_failed: candidate_admission_ledger_persistence_failed: candidate_admission_ledger_implementation_failed: candidate_admission_rollback_implementation_failed: candidate_admission_writer_receipt_failed: candidate_admission_writer_implementation_failed: candidate_admission_ledger_failed: candidate_admission_writer_contract_failed: candidate_admission_writer_inventory_failed: candidate_admission_writer_preflight_failed: candidate_admission_live_stage_failed: candidate_admission_enable_gate_failed: candidate_admission_switch_failed: candidate_admission_promotion_failed: candidate_admission_decision_failed: missing_candidate_execution" {
		t.Fatalf("bad candidate admission seal: %+v", seal)
	}
	finalGateLine := lines[len(lines)-5]
	for _, want := range []string{
		"live-route candidate admission final gate dry-run",
		"class=direct-user",
		"route=user_bridge",
		"source=user_bridge",
		"seal=",
		"permit=",
		"readiness=",
		"ledger_verification=",
		"final_gate=blocked",
		"final_gate_action=reject",
		"dry_run_only=true",
		"seal_verified=false",
		"permit_verified=false",
		"readiness_verified=false",
		"ledger_verified=false",
		"writer_ready=false rollback_ready=false ledger_ready=false final_gate_ready=false",
		"seal_ready=false permit_ready=false key_matched=true",
		"readiness_ready=false verification_ready=false persistence_ready=false writer_impl=false rollback_impl=false ledger_impl=false",
		"contracts_ready=false write_allowed=false admission_allowed=false live_ready=false live_enabled=false mutates=false",
		"admission_final_gate_id=",
		"passed=false",
		"reason=candidate_admission_seal_failed: candidate_admission_permit_failed: candidate_admission_readiness_failed: candidate_admission_ledger_verification_failed: candidate_admission_ledger_persistence_failed: candidate_admission_ledger_implementation_failed: candidate_admission_rollback_implementation_failed: candidate_admission_writer_receipt_failed: candidate_admission_writer_implementation_failed: candidate_admission_ledger_failed: candidate_admission_writer_contract_failed: candidate_admission_writer_inventory_failed: candidate_admission_writer_preflight_failed: candidate_admission_live_stage_failed: candidate_admission_enable_gate_failed: candidate_admission_switch_failed: candidate_admission_promotion_failed: candidate_admission_decision_failed: missing_candidate_execution",
	} {
		if !strings.Contains(finalGateLine, want) {
			t.Fatalf("candidate admission final gate line missing %q: %q", want, finalGateLine)
		}
	}
	raw, err = os.ReadFile(finalGateLog)
	if err != nil {
		t.Fatal(err)
	}
	var finalGate admissionLiveRouteTurnCandidateAdmissionFinalGate
	if err := json.Unmarshal([]byte(strings.TrimSpace(string(raw))), &finalGate); err != nil {
		t.Fatal(err)
	}
	if finalGate.Schema != admissionLiveRouteTurnCandidateAdmissionFinalGateSchema ||
		finalGate.Passed ||
		finalGate.LiveReady ||
		finalGate.LiveAdmissionEnabled ||
		finalGate.AdmissionAllowed ||
		finalGate.AdmissionFinalGateState != "blocked" ||
		finalGate.AdmissionFinalGateAction != "reject" ||
		!finalGate.AdmissionFinalGateDryRunOnly ||
		finalGate.AdmissionFinalGateSealVerified ||
		finalGate.AdmissionFinalGatePermitVerified ||
		finalGate.AdmissionFinalGateReadinessVerified ||
		finalGate.AdmissionFinalGateLedgerVerified ||
		finalGate.AdmissionFinalGateWriterReady ||
		finalGate.AdmissionFinalGateRollbackReady ||
		finalGate.AdmissionFinalGateLedgerReady ||
		finalGate.AdmissionFinalGateReady ||
		finalGate.WriteAllowed ||
		finalGate.MutatesState ||
		finalGate.SourceAdmissionSealPassed ||
		finalGate.AdmissionFinalGateID != "" ||
		finalGate.Reason != "candidate_admission_seal_failed: candidate_admission_permit_failed: candidate_admission_readiness_failed: candidate_admission_ledger_verification_failed: candidate_admission_ledger_persistence_failed: candidate_admission_ledger_implementation_failed: candidate_admission_rollback_implementation_failed: candidate_admission_writer_receipt_failed: candidate_admission_writer_implementation_failed: candidate_admission_ledger_failed: candidate_admission_writer_contract_failed: candidate_admission_writer_inventory_failed: candidate_admission_writer_preflight_failed: candidate_admission_live_stage_failed: candidate_admission_enable_gate_failed: candidate_admission_switch_failed: candidate_admission_promotion_failed: candidate_admission_decision_failed: missing_candidate_execution" {
		t.Fatalf("bad candidate admission final gate: %+v", finalGate)
	}
	resonanceIntentLine := lines[len(lines)-4]
	for _, want := range []string{
		"live-route candidate admission resonance intent dry-run",
		"class=direct-user",
		"route=user_bridge",
		"source=user_bridge",
		"final_gate=",
		"seal=",
		"permit=",
		"readiness=",
		"ledger_verification=",
		"receiver= receiver_kind= influence_kind= max_influence=0.00 ttl_turns=0 causal_id=",
		"raw_text_allowed=false janus_surface_allowed=false cooc_learning_allowed=false delta_harvest_allowed=false rollback_required=false pre_hash_required=false post_hash_required=false",
		"intent=blocked",
		"intent_action=reject",
		"dry_run_only=true",
		"final_gate_verified=false",
		"seal_verified=false",
		"permit_verified=false",
		"readiness_verified=false",
		"ledger_verified=false",
		"writer_ready=false rollback_ready=false ledger_ready=false intent_ready=false",
		"contracts_ready=false write_allowed=false admission_allowed=false live_ready=false live_enabled=false mutates=false",
		"admission_resonance_intent_id=",
		"passed=false",
		"reason=candidate_admission_final_gate_failed: candidate_admission_seal_failed: candidate_admission_permit_failed: candidate_admission_readiness_failed: candidate_admission_ledger_verification_failed: candidate_admission_ledger_persistence_failed: candidate_admission_ledger_implementation_failed: candidate_admission_rollback_implementation_failed: candidate_admission_writer_receipt_failed: candidate_admission_writer_implementation_failed: candidate_admission_ledger_failed: candidate_admission_writer_contract_failed: candidate_admission_writer_inventory_failed: candidate_admission_writer_preflight_failed: candidate_admission_live_stage_failed: candidate_admission_enable_gate_failed: candidate_admission_switch_failed: candidate_admission_promotion_failed: candidate_admission_decision_failed: missing_candidate_execution",
	} {
		if !strings.Contains(resonanceIntentLine, want) {
			t.Fatalf("candidate admission resonance intent line missing %q: %q", want, resonanceIntentLine)
		}
	}
	raw, err = os.ReadFile(resonanceIntentLog)
	if err != nil {
		t.Fatal(err)
	}
	var resonanceIntent admissionLiveRouteTurnCandidateAdmissionResonanceIntent
	if err := json.Unmarshal([]byte(strings.TrimSpace(string(raw))), &resonanceIntent); err != nil {
		t.Fatal(err)
	}
	if resonanceIntent.Schema != admissionLiveRouteTurnCandidateAdmissionResonanceIntentSchema ||
		resonanceIntent.Passed ||
		resonanceIntent.LiveReady ||
		resonanceIntent.LiveAdmissionEnabled ||
		resonanceIntent.AdmissionAllowed ||
		resonanceIntent.AdmissionResonanceIntentState != "blocked" ||
		resonanceIntent.AdmissionResonanceIntentAction != "reject" ||
		!resonanceIntent.AdmissionResonanceIntentDryRunOnly ||
		resonanceIntent.AdmissionResonanceIntentFinalGateVerified ||
		resonanceIntent.AdmissionResonanceIntentSealVerified ||
		resonanceIntent.AdmissionResonanceIntentPermitVerified ||
		resonanceIntent.AdmissionResonanceIntentReadinessVerified ||
		resonanceIntent.AdmissionResonanceIntentLedgerVerified ||
		resonanceIntent.AdmissionResonanceIntentWriterReady ||
		resonanceIntent.AdmissionResonanceIntentRollbackReady ||
		resonanceIntent.AdmissionResonanceIntentLedgerReady ||
		resonanceIntent.AdmissionResonanceIntentReceiver != "" ||
		resonanceIntent.AdmissionResonanceIntentReceiverKind != "" ||
		resonanceIntent.AdmissionResonanceIntentInfluenceKind != "" ||
		resonanceIntent.AdmissionResonanceIntentMaxInfluence != 0 ||
		resonanceIntent.AdmissionResonanceIntentTTLTurns != 0 ||
		resonanceIntent.AdmissionResonanceIntentCausalID != "" ||
		resonanceIntent.AdmissionResonanceIntentRawDreamTextAllowed ||
		resonanceIntent.AdmissionResonanceIntentJanusSurfaceAllowed ||
		resonanceIntent.AdmissionResonanceIntentCoocLearningAllowed ||
		resonanceIntent.AdmissionResonanceIntentDeltaHarvestAllowed ||
		resonanceIntent.AdmissionResonanceIntentRollbackRequired ||
		resonanceIntent.AdmissionResonanceIntentPreStateHashRequired ||
		resonanceIntent.AdmissionResonanceIntentPostStateHashRequired ||
		resonanceIntent.AdmissionResonanceIntentReady ||
		resonanceIntent.WriteAllowed ||
		resonanceIntent.MutatesState ||
		resonanceIntent.SourceAdmissionFinalGatePassed ||
		resonanceIntent.AdmissionResonanceIntentID != "" ||
		resonanceIntent.Reason != "candidate_admission_final_gate_failed: candidate_admission_seal_failed: candidate_admission_permit_failed: candidate_admission_readiness_failed: candidate_admission_ledger_verification_failed: candidate_admission_ledger_persistence_failed: candidate_admission_ledger_implementation_failed: candidate_admission_rollback_implementation_failed: candidate_admission_writer_receipt_failed: candidate_admission_writer_implementation_failed: candidate_admission_ledger_failed: candidate_admission_writer_contract_failed: candidate_admission_writer_inventory_failed: candidate_admission_writer_preflight_failed: candidate_admission_live_stage_failed: candidate_admission_enable_gate_failed: candidate_admission_switch_failed: candidate_admission_promotion_failed: candidate_admission_decision_failed: missing_candidate_execution" {
		t.Fatalf("bad candidate admission resonance intent: %+v", resonanceIntent)
	}
	resonanceReceiverLine := lines[len(lines)-3]
	for _, want := range []string{
		"live-route candidate admission resonance receiver dry-run",
		"class=direct-user",
		"route=user_bridge",
		"source=user_bridge",
		"intent=",
		"final_gate=",
		"seal=",
		"permit=",
		"readiness=",
		"ledger_verification=",
		"receiver= receiver_kind= influence_kind= max_influence=0.00 ttl_turns=0 causal_id= source_causal_id=",
		"pre_state_hash= post_state_hash= delta_hash= state_hash_mode=",
		"raw_text_observed=false raw_text_forwarded=false janus_surface_allowed=false cooc_learning_allowed=false delta_harvest_allowed=false body_mutation_allowed=false rollback_required=false",
		"receiver_state=blocked",
		"receiver_action=reject",
		"dry_run_only=true",
		"intent_verified=false",
		"final_gate_verified=false",
		"seal_verified=false",
		"permit_verified=false",
		"readiness_verified=false",
		"ledger_verified=false",
		"writer_ready=false rollback_ready=false ledger_ready=false receiver_ready=false",
		"contracts_ready=false write_allowed=false admission_allowed=false live_ready=false live_enabled=false mutates=false",
		"admission_resonance_receiver_id=",
		"passed=false",
		"reason=candidate_admission_resonance_intent_failed: candidate_admission_final_gate_failed: candidate_admission_seal_failed: candidate_admission_permit_failed: candidate_admission_readiness_failed: candidate_admission_ledger_verification_failed: candidate_admission_ledger_persistence_failed: candidate_admission_ledger_implementation_failed: candidate_admission_rollback_implementation_failed: candidate_admission_writer_receipt_failed: candidate_admission_writer_implementation_failed: candidate_admission_ledger_failed: candidate_admission_writer_contract_failed: candidate_admission_writer_inventory_failed: candidate_admission_writer_preflight_failed: candidate_admission_live_stage_failed: candidate_admission_enable_gate_failed: candidate_admission_switch_failed: candidate_admission_promotion_failed: candidate_admission_decision_failed: missing_candidate_execution",
	} {
		if !strings.Contains(resonanceReceiverLine, want) {
			t.Fatalf("candidate admission resonance receiver line missing %q: %q", want, resonanceReceiverLine)
		}
	}
	raw, err = os.ReadFile(resonanceReceiverLog)
	if err != nil {
		t.Fatal(err)
	}
	var resonanceReceiver admissionLiveRouteTurnCandidateAdmissionResonanceReceiver
	if err := json.Unmarshal([]byte(strings.TrimSpace(string(raw))), &resonanceReceiver); err != nil {
		t.Fatal(err)
	}
	if resonanceReceiver.Schema != admissionLiveRouteTurnCandidateAdmissionResonanceReceiverSchema ||
		resonanceReceiver.Passed ||
		resonanceReceiver.LiveReady ||
		resonanceReceiver.LiveAdmissionEnabled ||
		resonanceReceiver.AdmissionAllowed ||
		resonanceReceiver.AdmissionResonanceReceiverState != "blocked" ||
		resonanceReceiver.AdmissionResonanceReceiverAction != "reject" ||
		!resonanceReceiver.AdmissionResonanceReceiverDryRunOnly ||
		resonanceReceiver.AdmissionResonanceReceiverIntentVerified ||
		resonanceReceiver.AdmissionResonanceReceiverFinalGateVerified ||
		resonanceReceiver.AdmissionResonanceReceiverSealVerified ||
		resonanceReceiver.AdmissionResonanceReceiverPermitVerified ||
		resonanceReceiver.AdmissionResonanceReceiverReadinessVerified ||
		resonanceReceiver.AdmissionResonanceReceiverLedgerVerified ||
		resonanceReceiver.AdmissionResonanceReceiverWriterReady ||
		resonanceReceiver.AdmissionResonanceReceiverRollbackReady ||
		resonanceReceiver.AdmissionResonanceReceiverLedgerReady ||
		resonanceReceiver.AdmissionResonanceReceiverReceiver != "" ||
		resonanceReceiver.AdmissionResonanceReceiverReceiverKind != "" ||
		resonanceReceiver.AdmissionResonanceReceiverInfluenceKind != "" ||
		resonanceReceiver.AdmissionResonanceReceiverMaxInfluence != 0 ||
		resonanceReceiver.AdmissionResonanceReceiverTTLTurns != 0 ||
		resonanceReceiver.AdmissionResonanceReceiverCausalID != "" ||
		resonanceReceiver.AdmissionResonanceReceiverPreStateHash != "" ||
		resonanceReceiver.AdmissionResonanceReceiverPostStateHash != "" ||
		resonanceReceiver.AdmissionResonanceReceiverStateDeltaHash != "" ||
		resonanceReceiver.AdmissionResonanceReceiverStateHashMode != "" ||
		resonanceReceiver.AdmissionResonanceReceiverRawDreamTextObserved ||
		resonanceReceiver.AdmissionResonanceReceiverRawDreamTextForwarded ||
		resonanceReceiver.AdmissionResonanceReceiverJanusSurfaceAllowed ||
		resonanceReceiver.AdmissionResonanceReceiverCoocLearningAllowed ||
		resonanceReceiver.AdmissionResonanceReceiverDeltaHarvestAllowed ||
		resonanceReceiver.AdmissionResonanceReceiverBodyMutationAllowed ||
		resonanceReceiver.AdmissionResonanceReceiverRollbackRequired ||
		resonanceReceiver.AdmissionResonanceReceiverReady ||
		resonanceReceiver.WriteAllowed ||
		resonanceReceiver.MutatesState ||
		resonanceReceiver.SourceAdmissionResonanceIntentPassed ||
		resonanceReceiver.AdmissionResonanceReceiverID != "" ||
		resonanceReceiver.Reason != "candidate_admission_resonance_intent_failed: candidate_admission_final_gate_failed: candidate_admission_seal_failed: candidate_admission_permit_failed: candidate_admission_readiness_failed: candidate_admission_ledger_verification_failed: candidate_admission_ledger_persistence_failed: candidate_admission_ledger_implementation_failed: candidate_admission_rollback_implementation_failed: candidate_admission_writer_receipt_failed: candidate_admission_writer_implementation_failed: candidate_admission_ledger_failed: candidate_admission_writer_contract_failed: candidate_admission_writer_inventory_failed: candidate_admission_writer_preflight_failed: candidate_admission_live_stage_failed: candidate_admission_enable_gate_failed: candidate_admission_switch_failed: candidate_admission_promotion_failed: candidate_admission_decision_failed: missing_candidate_execution" {
		t.Fatalf("bad candidate admission resonance receiver: %+v", resonanceReceiver)
	}
	resonanceObservationLine := lines[len(lines)-2]
	for _, want := range []string{
		"live-route candidate admission resonance observation dry-run",
		"class=direct-user",
		"route=user_bridge",
		"source=user_bridge",
		"receiver=",
		"intent=",
		"final_gate=",
		"seal=",
		"permit=",
		"readiness=",
		"ledger_verification=",
		"observer= observer_kind= observation_kind= observation_mode= causal_id=",
		"append_hash= read_back_hash= source_receiver_causal_id= source_receiver_delta_hash=",
		"append_only=false read_back=false receipt_verified=false raw_text_observed=false raw_text_forwarded=false janus_surface_allowed=false cooc_learning_allowed=false delta_harvest_allowed=false body_mutation_allowed=false rollback_required=false",
		"observation_state=blocked",
		"observation_action=reject",
		"dry_run_only=true",
		"receiver_verified=false",
		"intent_verified=false",
		"final_gate_verified=false",
		"seal_verified=false",
		"permit_verified=false",
		"readiness_verified=false",
		"ledger_verified=false",
		"writer_ready=false rollback_ready=false ledger_ready=false observation_ready=false",
		"contracts_ready=false write_allowed=false admission_allowed=false live_ready=false live_enabled=false mutates=false",
		"admission_resonance_observation_id=",
		"passed=false",
		"reason=candidate_admission_resonance_receiver_failed: candidate_admission_resonance_intent_failed: candidate_admission_final_gate_failed: candidate_admission_seal_failed: candidate_admission_permit_failed: candidate_admission_readiness_failed: candidate_admission_ledger_verification_failed: candidate_admission_ledger_persistence_failed: candidate_admission_ledger_implementation_failed: candidate_admission_rollback_implementation_failed: candidate_admission_writer_receipt_failed: candidate_admission_writer_implementation_failed: candidate_admission_ledger_failed: candidate_admission_writer_contract_failed: candidate_admission_writer_inventory_failed: candidate_admission_writer_preflight_failed: candidate_admission_live_stage_failed: candidate_admission_enable_gate_failed: candidate_admission_switch_failed: candidate_admission_promotion_failed: candidate_admission_decision_failed: missing_candidate_execution",
	} {
		if !strings.Contains(resonanceObservationLine, want) {
			t.Fatalf("candidate admission resonance observation line missing %q: %q", want, resonanceObservationLine)
		}
	}
	raw, err = os.ReadFile(resonanceObservationLog)
	if err != nil {
		t.Fatal(err)
	}
	var resonanceObservation admissionLiveRouteTurnCandidateAdmissionResonanceObservation
	if err := json.Unmarshal([]byte(strings.TrimSpace(string(raw))), &resonanceObservation); err != nil {
		t.Fatal(err)
	}
	if resonanceObservation.Schema != admissionLiveRouteTurnCandidateAdmissionResonanceObservationSchema ||
		resonanceObservation.Passed ||
		resonanceObservation.LiveReady ||
		resonanceObservation.LiveAdmissionEnabled ||
		resonanceObservation.AdmissionAllowed ||
		resonanceObservation.AdmissionResonanceObservationState != "blocked" ||
		resonanceObservation.AdmissionResonanceObservationAction != "reject" ||
		!resonanceObservation.AdmissionResonanceObservationDryRunOnly ||
		resonanceObservation.AdmissionResonanceObservationReceiverVerified ||
		resonanceObservation.AdmissionResonanceObservationIntentVerified ||
		resonanceObservation.AdmissionResonanceObservationFinalGateVerified ||
		resonanceObservation.AdmissionResonanceObservationSealVerified ||
		resonanceObservation.AdmissionResonanceObservationPermitVerified ||
		resonanceObservation.AdmissionResonanceObservationReadinessVerified ||
		resonanceObservation.AdmissionResonanceObservationLedgerVerified ||
		resonanceObservation.AdmissionResonanceObservationWriterReady ||
		resonanceObservation.AdmissionResonanceObservationRollbackReady ||
		resonanceObservation.AdmissionResonanceObservationLedgerReady ||
		resonanceObservation.AdmissionResonanceObservationObserver != "" ||
		resonanceObservation.AdmissionResonanceObservationObserverKind != "" ||
		resonanceObservation.AdmissionResonanceObservationKind != "" ||
		resonanceObservation.AdmissionResonanceObservationMode != "" ||
		resonanceObservation.AdmissionResonanceObservationCausalID != "" ||
		resonanceObservation.AdmissionResonanceObservationAppendHash != "" ||
		resonanceObservation.AdmissionResonanceObservationReadBackHash != "" ||
		resonanceObservation.AdmissionResonanceObservationAppendOnly ||
		resonanceObservation.AdmissionResonanceObservationReadBack ||
		resonanceObservation.AdmissionResonanceObservationReceiptVerified ||
		resonanceObservation.AdmissionResonanceObservationRawDreamTextObserved ||
		resonanceObservation.AdmissionResonanceObservationRawDreamTextForwarded ||
		resonanceObservation.AdmissionResonanceObservationJanusSurfaceAllowed ||
		resonanceObservation.AdmissionResonanceObservationCoocLearningAllowed ||
		resonanceObservation.AdmissionResonanceObservationDeltaHarvestAllowed ||
		resonanceObservation.AdmissionResonanceObservationBodyMutationAllowed ||
		resonanceObservation.AdmissionResonanceObservationRollbackRequired ||
		resonanceObservation.AdmissionResonanceObservationReady ||
		resonanceObservation.WriteAllowed ||
		resonanceObservation.MutatesState ||
		resonanceObservation.SourceAdmissionResonanceReceiverPassed ||
		resonanceObservation.AdmissionResonanceObservationID != "" ||
		resonanceObservation.Reason != "candidate_admission_resonance_receiver_failed: candidate_admission_resonance_intent_failed: candidate_admission_final_gate_failed: candidate_admission_seal_failed: candidate_admission_permit_failed: candidate_admission_readiness_failed: candidate_admission_ledger_verification_failed: candidate_admission_ledger_persistence_failed: candidate_admission_ledger_implementation_failed: candidate_admission_rollback_implementation_failed: candidate_admission_writer_receipt_failed: candidate_admission_writer_implementation_failed: candidate_admission_ledger_failed: candidate_admission_writer_contract_failed: candidate_admission_writer_inventory_failed: candidate_admission_writer_preflight_failed: candidate_admission_live_stage_failed: candidate_admission_enable_gate_failed: candidate_admission_switch_failed: candidate_admission_promotion_failed: candidate_admission_decision_failed: missing_candidate_execution" {
		t.Fatalf("bad candidate admission resonance observation: %+v", resonanceObservation)
	}
	resonanceGraftBoundaryReason := "candidate_admission_resonance_observation_failed: " + resonanceObservation.Reason
	resonanceGraftBoundaryLine := lines[len(lines)-1]
	for _, want := range []string{
		"live-route candidate admission resonance graft boundary dry-run",
		"class=direct-user",
		"route=user_bridge",
		"source=user_bridge",
		"observation=",
		"receiver=",
		"intent=",
		"final_gate=",
		"seal=",
		"permit=",
		"readiness=",
		"ledger_verification=",
		"boundary_kind= boundary_mode= boundary_stage= causal_id=",
		"boundary_hash= read_back_hash= source_observation_causal_id= source_observation_read_back_hash=",
		"shadow_only=false graft_allowed=false raw_text_allowed=false janus_surface_allowed=false cooc_learning_allowed=false delta_harvest_allowed=false body_mutation_allowed=false rollback_required=false",
		"boundary_state=blocked",
		"boundary_action=reject",
		"boundary_target= boundary_target_kind= boundary_target_mode= receipt_shape=",
		"dry_run_only=true",
		"observation_verified=false",
		"receiver_verified=false",
		"intent_verified=false",
		"final_gate_verified=false",
		"seal_verified=false",
		"permit_verified=false",
		"readiness_verified=false",
		"ledger_verified=false",
		"writer_ready=false rollback_ready=false ledger_ready=false boundary_ready=false",
		"contracts_ready=false write_allowed=false admission_allowed=false live_ready=false live_enabled=false mutates=false",
		"admission_resonance_graft_boundary_id=",
		"passed=false",
		"reason=" + resonanceGraftBoundaryReason,
	} {
		if !strings.Contains(resonanceGraftBoundaryLine, want) {
			t.Fatalf("candidate admission resonance graft boundary line missing %q: %q", want, resonanceGraftBoundaryLine)
		}
	}
	raw, err = os.ReadFile(resonanceGraftBoundaryLog)
	if err != nil {
		t.Fatal(err)
	}
	var resonanceGraftBoundary admissionLiveRouteTurnCandidateAdmissionResonanceGraftBoundary
	if err := json.Unmarshal([]byte(strings.TrimSpace(string(raw))), &resonanceGraftBoundary); err != nil {
		t.Fatal(err)
	}
	if resonanceGraftBoundary.Schema != admissionLiveRouteTurnCandidateAdmissionResonanceGraftBoundarySchema ||
		resonanceGraftBoundary.Passed ||
		resonanceGraftBoundary.LiveReady ||
		resonanceGraftBoundary.LiveAdmissionEnabled ||
		resonanceGraftBoundary.AdmissionAllowed ||
		resonanceGraftBoundary.AdmissionResonanceGraftBoundaryState != "blocked" ||
		resonanceGraftBoundary.AdmissionResonanceGraftBoundaryAction != "reject" ||
		!resonanceGraftBoundary.AdmissionResonanceGraftBoundaryDryRunOnly ||
		resonanceGraftBoundary.AdmissionResonanceGraftBoundaryObservationVerified ||
		resonanceGraftBoundary.AdmissionResonanceGraftBoundaryReceiverVerified ||
		resonanceGraftBoundary.AdmissionResonanceGraftBoundaryIntentVerified ||
		resonanceGraftBoundary.AdmissionResonanceGraftBoundaryFinalGateVerified ||
		resonanceGraftBoundary.AdmissionResonanceGraftBoundarySealVerified ||
		resonanceGraftBoundary.AdmissionResonanceGraftBoundaryPermitVerified ||
		resonanceGraftBoundary.AdmissionResonanceGraftBoundaryReadinessVerified ||
		resonanceGraftBoundary.AdmissionResonanceGraftBoundaryLedgerVerified ||
		resonanceGraftBoundary.AdmissionResonanceGraftBoundaryWriterReady ||
		resonanceGraftBoundary.AdmissionResonanceGraftBoundaryRollbackReady ||
		resonanceGraftBoundary.AdmissionResonanceGraftBoundaryLedgerReady ||
		resonanceGraftBoundary.AdmissionResonanceGraftBoundaryKind != "" ||
		resonanceGraftBoundary.AdmissionResonanceGraftBoundaryMode != "" ||
		resonanceGraftBoundary.AdmissionResonanceGraftBoundaryStage != "" ||
		resonanceGraftBoundary.AdmissionResonanceGraftBoundaryCausalID != "" ||
		resonanceGraftBoundary.AdmissionResonanceGraftBoundaryHash != "" ||
		resonanceGraftBoundary.AdmissionResonanceGraftBoundaryReadBackHash != "" ||
		resonanceGraftBoundary.AdmissionResonanceGraftBoundaryShadowOnly ||
		resonanceGraftBoundary.AdmissionResonanceGraftBoundaryGraftAllowed ||
		resonanceGraftBoundary.AdmissionResonanceGraftBoundaryRawDreamTextAllowed ||
		resonanceGraftBoundary.AdmissionResonanceGraftBoundaryJanusSurfaceAllowed ||
		resonanceGraftBoundary.AdmissionResonanceGraftBoundaryCoocLearningAllowed ||
		resonanceGraftBoundary.AdmissionResonanceGraftBoundaryDeltaHarvestAllowed ||
		resonanceGraftBoundary.AdmissionResonanceGraftBoundaryBodyMutationAllowed ||
		resonanceGraftBoundary.AdmissionResonanceGraftBoundaryRollbackRequired ||
		resonanceGraftBoundary.AdmissionResonanceGraftBoundaryReady ||
		resonanceGraftBoundary.WriteAllowed ||
		resonanceGraftBoundary.MutatesState ||
		resonanceGraftBoundary.BodyTarget != "none" ||
		resonanceGraftBoundary.SourceAdmissionResonanceObservationSchema != admissionLiveRouteTurnCandidateAdmissionResonanceObservationSchema ||
		resonanceGraftBoundary.SourceAdmissionResonanceObservationPassed ||
		resonanceGraftBoundary.SourceAdmissionResonanceObservationID != "" ||
		resonanceGraftBoundary.SourceAdmissionResonanceObservationAction != "reject" ||
		resonanceGraftBoundary.SourceAdmissionResonanceObservationReady ||
		resonanceGraftBoundary.SourceAdmissionResonanceObservationCausalID != "" ||
		resonanceGraftBoundary.SourceAdmissionResonanceObservationAppendHash != "" ||
		resonanceGraftBoundary.SourceAdmissionResonanceObservationReadBackHash != "" ||
		resonanceGraftBoundary.AdmissionResonanceGraftBoundaryID != "" ||
		resonanceGraftBoundary.Reason != resonanceGraftBoundaryReason {
		t.Fatalf("bad candidate admission resonance graft boundary: %+v", resonanceGraftBoundary)
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
