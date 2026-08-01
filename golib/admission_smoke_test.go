package main

import (
	"encoding/json"
	"os"
	"path/filepath"
	"reflect"
	"sort"
	"strings"
	"testing"
)

func TestAdmissionLiveRouteGateSmokeCasesCoverPlan(t *testing.T) {
	cases := admissionLiveRouteGateSmokeCases()
	wantMatched := len(admissionLiveRoutePromptClasses())
	if len(cases) != wantMatched+2 {
		t.Fatalf("bad live route smoke case count: got %d want %d", len(cases), wantMatched+2)
	}
	seenClasses := make(map[string]bool)
	seenRoutes := make(map[string]bool)
	var rejected int
	for _, tc := range cases {
		if tc.wantPassed {
			seenClasses[tc.wantPromptClass] = true
			seenRoutes[tc.wantRoute] = true
			if tc.source != tc.wantSource {
				t.Fatalf("matched smoke case source mismatch: %+v", tc)
			}
			if got := qloopSweepPromptClass(tc.trigger, tc.seed); got != tc.wantPromptClass {
				t.Fatalf("matched smoke trigger %q normalized to %q, want %q", tc.trigger, got, tc.wantPromptClass)
			}
			wantTrigger := admissionLiveRouteGateSmokeTrigger(tc.wantRoute, tc.wantPromptClass)
			if tc.trigger != wantTrigger {
				t.Fatalf("matched smoke trigger mismatch: got %q want %q", tc.trigger, wantTrigger)
			}
			continue
		}
		rejected++
	}
	if rejected != 2 {
		t.Fatalf("expected wrong-source and unknown-class reject cases, got %d", rejected)
	}
	for _, promptClass := range admissionLiveRoutePromptClasses() {
		if !seenClasses[promptClass] {
			t.Fatalf("prompt class %s missing from live route smoke cases", promptClass)
		}
	}
	wantRoutes := []string{"chorus", "direct", "qloop_hint_qa", "qloop_target", "user_bridge"}
	var gotRoutes []string
	for route := range seenRoutes {
		gotRoutes = append(gotRoutes, route)
	}
	sort.Strings(gotRoutes)
	if !reflect.DeepEqual(gotRoutes, wantRoutes) {
		t.Fatalf("bad live route smoke routes: got %v want %v", gotRoutes, wantRoutes)
	}
}

func TestAdmissionLiveRouteGateSmokeWritesBroadMatchedAndRejectedReceipts(t *testing.T) {
	t.Setenv("AM_DREAM_ADMISSION", dreamAdmissionShadow)
	t.Setenv("AM_DREAM_ADMISSION_ALLOWED_SOURCES", "")
	t.Setenv("AM_DREAM_ADMISSION_REQUIRE_LIVE_ROUTE_PLAN", "1")
	logPath := filepath.Join(t.TempDir(), "live-route-gate.jsonl")
	t.Setenv("AM_DREAM_ADMISSION_LOG", logPath)

	if err := runAdmissionLiveRouteGateSmoke(); err != nil {
		t.Fatal(err)
	}

	raw, err := os.ReadFile(logPath)
	if err != nil {
		t.Fatal(err)
	}
	lines := strings.Split(strings.TrimSpace(string(raw)), "\n")
	cases := admissionLiveRouteGateSmokeCases()
	if len(lines) != len(cases) {
		t.Fatalf("expected %d route-gate receipts, got %d: %q", len(cases), len(lines), raw)
	}
	var matched, rejected, unknown int
	seenRoutes := make(map[string]bool)
	for i, line := range lines {
		var got dreamCandidate
		if err := json.Unmarshal([]byte(line), &got); err != nil {
			t.Fatalf("receipt %d: %v", i+1, err)
		}
		if got.Admission == nil || got.Admission.LiveRouteChoice == nil {
			t.Fatalf("receipt %d missing live route choice: %+v", i+1, got.Admission)
		}
		choice := got.Admission.LiveRouteChoice
		seenRoutes[choice.ExpectedSource] = true
		if got.Admission.Passed {
			matched++
			continue
		}
		rejected++
		if choice.Reason == "live route plan failed: unknown_prompt_class" {
			unknown++
		}
	}
	if matched != len(admissionLiveRoutePromptClasses()) || rejected != 2 || unknown != 1 {
		t.Fatalf("bad broad smoke receipt counts: matched=%d rejected=%d unknown=%d", matched, rejected, unknown)
	}
	for _, route := range []string{"chorus", "direct", "qloop_hint_qa", "qloop_target", "user_bridge"} {
		if !seenRoutes[route] {
			t.Fatalf("expected route %s missing from receipts; saw %v", route, seenRoutes)
		}
	}
	if !strings.Contains(string(raw), "source direct does not match live route chorus for prompt class identity") {
		t.Fatalf("wrong-source route reason missing from log: %s", raw)
	}
	if !strings.Contains(string(raw), "live route plan failed: unknown_prompt_class") {
		t.Fatalf("unknown-class route reason missing from log: %s", raw)
	}
	for _, trigger := range []string{"user_bridge-cold-reader", "user_bridge-direct-user", "qloop_target-recipient-lock", "qloop_hint_qa-polyphony", "direct-dream", "chorus-identity", "chorus-unknown-pressure"} {
		if !strings.Contains(string(raw), "\"trigger\":\""+trigger+"\"") {
			t.Fatalf("route-prefixed trigger %q missing from log: %s", trigger, raw)
		}
	}
}

func TestAdmissionLiveRouteChatSmokeWritesDryRunReceipt(t *testing.T) {
	t.Setenv("AM_DREAM_ADMISSION", dreamAdmissionShadow)
	t.Setenv("AM_DREAM_ADMISSION_LIVE_ROUTE_CHOICE_DRY_RUN", "1")
	logPath := filepath.Join(t.TempDir(), "live-route-chat.jsonl")
	t.Setenv("AM_DREAM_ADMISSION_LOG", logPath)

	if err := runAdmissionLiveRouteChatSmoke(); err != nil {
		t.Fatal(err)
	}

	raw, err := os.ReadFile(logPath)
	if err != nil {
		t.Fatal(err)
	}
	var got dreamCandidate
	if err := json.Unmarshal([]byte(strings.TrimSpace(string(raw))), &got); err != nil {
		t.Fatal(err)
	}
	if got.Trigger != "chorus-identity" || got.Admission == nil || !got.Admission.Passed ||
		!got.Admission.LiveRouteChoiceDryRun || got.Admission.LiveRouteChoice == nil {
		t.Fatalf("bad chat dry-run receipt: %+v", got)
	}
	choice := got.Admission.LiveRouteChoice
	if !choice.Passed || choice.PromptClass != "identity" || choice.Route != "chorus" ||
		choice.Source != "chorus" || choice.ExpectedSource != "chorus" {
		t.Fatalf("bad chat dry-run route choice: %+v", choice)
	}
}

func TestAdmissionLiveRouteTurnSmokeWritesObservations(t *testing.T) {
	t.Setenv("AM_DREAM_ADMISSION_LIVE_ROUTE_CHOICE_DRY_RUN", "1")
	logPath := filepath.Join(t.TempDir(), "live-route-turn.jsonl")
	t.Setenv("AM_LIVE_ROUTE_TURN_LOG", logPath)

	if err := runAdmissionLiveRouteTurnSmoke(); err != nil {
		t.Fatal(err)
	}

	raw, err := os.ReadFile(logPath)
	if err != nil {
		t.Fatal(err)
	}
	lines := strings.Split(strings.TrimSpace(string(raw)), "\n")
	if len(lines) != 6 {
		t.Fatalf("expected 6 turn observations, got %d: %s", len(lines), raw)
	}
	var identity, unknown admissionLiveRouteTurnObservation
	if err := json.Unmarshal([]byte(lines[0]), &identity); err != nil {
		t.Fatal(err)
	}
	if err := json.Unmarshal([]byte(lines[len(lines)-1]), &unknown); err != nil {
		t.Fatal(err)
	}
	if identity.Schema != admissionLiveRouteTurnObservationSchema || identity.PromptClass != "identity" ||
		identity.Route != "chorus" || identity.ExpectedSource != "chorus" || !identity.Passed {
		t.Fatalf("bad identity turn observation: %+v", identity)
	}
	if unknown.PromptClass != "unknown" || unknown.Passed || unknown.Reason != "live route plan failed: unknown_prompt_class" {
		t.Fatalf("unknown turn should fail closed: %+v", unknown)
	}
}

func TestAdmissionLiveRouteTurnChoiceSmokeWritesChoices(t *testing.T) {
	t.Setenv("AM_LIVE_ROUTE_TURN_CHOICE_DRY_RUN", "1")
	logPath := filepath.Join(t.TempDir(), "live-route-turn-choice.jsonl")
	t.Setenv("AM_LIVE_ROUTE_TURN_CHOICE_LOG", logPath)

	if err := runAdmissionLiveRouteTurnChoiceSmoke(); err != nil {
		t.Fatal(err)
	}

	raw, err := os.ReadFile(logPath)
	if err != nil {
		t.Fatal(err)
	}
	lines := strings.Split(strings.TrimSpace(string(raw)), "\n")
	if len(lines) != 5 {
		t.Fatalf("expected 5 turn choices, got %d: %s", len(lines), raw)
	}
	var identity, unknown admissionLiveRouteTurnChoice
	if err := json.Unmarshal([]byte(lines[0]), &identity); err != nil {
		t.Fatal(err)
	}
	if err := json.Unmarshal([]byte(lines[len(lines)-1]), &unknown); err != nil {
		t.Fatal(err)
	}
	if identity.Schema != admissionLiveRouteTurnChoiceSchema ||
		identity.PromptClass != "identity" ||
		identity.Route != "chorus" ||
		identity.Source != "chorus" ||
		identity.ExpectedSource != "chorus" ||
		identity.CandidateTrigger != "chorus-identity" ||
		!identity.Passed {
		t.Fatalf("bad identity turn choice: %+v", identity)
	}
	if unknown.PromptClass != "unknown" ||
		unknown.Passed ||
		!strings.Contains(unknown.Reason, "unknown_prompt_class") {
		t.Fatalf("unknown turn choice should fail closed: %+v", unknown)
	}
}

func TestAdmissionLiveRouteTurnRequestSmokeWritesRequests(t *testing.T) {
	t.Setenv("AM_LIVE_ROUTE_TURN_REQUEST_DRY_RUN", "1")
	logPath := filepath.Join(t.TempDir(), "live-route-turn-request.jsonl")
	t.Setenv("AM_LIVE_ROUTE_TURN_REQUEST_LOG", logPath)

	if err := runAdmissionLiveRouteTurnRequestSmoke(); err != nil {
		t.Fatal(err)
	}

	raw, err := os.ReadFile(logPath)
	if err != nil {
		t.Fatal(err)
	}
	lines := strings.Split(strings.TrimSpace(string(raw)), "\n")
	if len(lines) != 5 {
		t.Fatalf("expected 5 turn requests, got %d: %s", len(lines), raw)
	}
	var identity, unknown admissionLiveRouteTurnRequest
	if err := json.Unmarshal([]byte(lines[0]), &identity); err != nil {
		t.Fatal(err)
	}
	if err := json.Unmarshal([]byte(lines[len(lines)-1]), &unknown); err != nil {
		t.Fatal(err)
	}
	if identity.Schema != admissionLiveRouteTurnRequestSchema ||
		identity.PromptClass != "identity" ||
		identity.Route != "chorus" ||
		identity.Source != "chorus" ||
		identity.ExpectedSource != "chorus" ||
		identity.CandidateTrigger != "chorus-identity" ||
		!strings.HasPrefix(identity.CandidateSeed, "turn-") ||
		!identity.Passed {
		t.Fatalf("bad identity turn request: %+v", identity)
	}
	if unknown.PromptClass != "unknown" ||
		unknown.Passed ||
		!strings.Contains(unknown.Reason, "unknown_prompt_class") ||
		!strings.HasPrefix(unknown.CandidateSeed, "turn-") {
		t.Fatalf("unknown turn request should fail closed with a stable seed: %+v", unknown)
	}
}

func TestAdmissionLiveRouteTurnGenerationJobSmokeWritesJobs(t *testing.T) {
	t.Setenv("AM_LIVE_ROUTE_TURN_GENERATION_JOB_DRY_RUN", "1")
	logPath := filepath.Join(t.TempDir(), "live-route-generation-job.jsonl")
	t.Setenv("AM_LIVE_ROUTE_TURN_GENERATION_JOB_LOG", logPath)

	if err := runAdmissionLiveRouteTurnGenerationJobSmoke(); err != nil {
		t.Fatal(err)
	}

	raw, err := os.ReadFile(logPath)
	if err != nil {
		t.Fatal(err)
	}
	lines := strings.Split(strings.TrimSpace(string(raw)), "\n")
	if len(lines) != 5 {
		t.Fatalf("expected 5 generation jobs, got %d: %s", len(lines), raw)
	}
	var identity, unknown admissionLiveRouteTurnGenerationJob
	if err := json.Unmarshal([]byte(lines[0]), &identity); err != nil {
		t.Fatal(err)
	}
	if err := json.Unmarshal([]byte(lines[len(lines)-1]), &unknown); err != nil {
		t.Fatal(err)
	}
	if identity.Schema != admissionLiveRouteTurnGenerationJobSchema ||
		identity.PromptClass != "identity" ||
		identity.Route != "chorus" ||
		identity.Source != "chorus" ||
		identity.Backend != "chorus-arianna" ||
		identity.Entrypoint != "field" ||
		identity.PromptFrame != "q_a" ||
		!strings.HasPrefix(identity.CandidateSeed, "turn-") ||
		!strings.HasPrefix(identity.JobID, "job-") ||
		!identity.Passed {
		t.Fatalf("bad identity generation job: %+v", identity)
	}
	if unknown.PromptClass != "unknown" ||
		unknown.Passed ||
		unknown.JobID != "" ||
		!strings.Contains(unknown.Reason, "unknown_prompt_class") ||
		!strings.HasPrefix(unknown.CandidateSeed, "turn-") {
		t.Fatalf("unknown generation job should fail closed without runnable job id: %+v", unknown)
	}
}

func TestAdmissionLiveRouteTurnGenerationJobInventoryGateSmokeFailsClosed(t *testing.T) {
	setBodyInventoryDefaultEnv(t)
	t.Setenv("AM_LIVE_ROUTE_TURN_GENERATION_JOB_DRY_RUN", "1")
	t.Setenv(admissionLiveRouteTurnGenerationJobInventoryGateEnv, "1")
	t.Setenv("AM_BODY_INVENTORY_ROOT", t.TempDir())
	logPath := filepath.Join(t.TempDir(), "live-route-generation-job-inventory-gate.jsonl")
	t.Setenv("AM_LIVE_ROUTE_TURN_GENERATION_JOB_LOG", logPath)

	if err := runAdmissionLiveRouteTurnGenerationJobInventoryGateSmoke(); err != nil {
		t.Fatal(err)
	}

	raw, err := os.ReadFile(logPath)
	if err != nil {
		t.Fatal(err)
	}
	var got admissionLiveRouteTurnGenerationJob
	if err := json.Unmarshal([]byte(strings.TrimSpace(string(raw))), &got); err != nil {
		t.Fatal(err)
	}
	if got.Passed ||
		got.JobID != "" ||
		got.BodyInventoryStatus != "blocked" ||
		got.RouteAvailabilityStatus != "unavailable" ||
		!strings.Contains(got.Reason, "route chorus unavailable in body inventory") ||
		!sameStrings(got.RouteMissingOrgans, []string{"chorus-binary", "nano-weight"}) {
		t.Fatalf("inventory-gated job should fail closed before runnable id: %+v", got)
	}
}

func TestAdmissionLiveRouteTurnRouteBoundarySmokeWritesTypedReceipts(t *testing.T) {
	setBodyInventoryDefaultEnv(t)
	t.Setenv("AM_LIVE_ROUTE_TURN_GENERATION_JOB_DRY_RUN", "1")
	t.Setenv(admissionLiveRouteTurnGenerationJobInventoryGateEnv, "1")
	t.Setenv("AM_LIVE_ROUTE_TURN_CANDIDATE_SHELL_DRY_RUN", "1")
	t.Setenv("AM_LIVE_ROUTE_TURN_CANDIDATE_EXECUTION_DRY_RUN", "1")
	t.Setenv("AM_LIVE_ROUTE_TURN_GENERATOR_ADAPTER_DRY_RUN", "1")
	t.Setenv("AM_LIVE_ROUTE_TURN_CANDIDATE_DRAFT_DRY_RUN", "1")
	t.Setenv("AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_DRY_RUN", "1")
	t.Setenv("AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_ADAPTER_DRY_RUN", "1")
	t.Setenv("AM_BODY_INVENTORY_ROOT", t.TempDir())
	logRoot := t.TempDir()
	jobLogPath := filepath.Join(logRoot, "live-route-generation-job.jsonl")
	shellLogPath := filepath.Join(logRoot, "live-route-candidate-shell.jsonl")
	executionLogPath := filepath.Join(logRoot, "live-route-candidate-execution.jsonl")
	adapterLogPath := filepath.Join(logRoot, "live-route-generator-adapter.jsonl")
	draftLogPath := filepath.Join(logRoot, "live-route-candidate-draft.jsonl")
	reviewLogPath := filepath.Join(logRoot, "live-route-candidate-review.jsonl")
	admissionLogPath := filepath.Join(logRoot, "live-route-candidate-admission.jsonl")
	admissionAdapterLogPath := filepath.Join(logRoot, "live-route-candidate-admission-adapter.jsonl")
	t.Setenv("AM_LIVE_ROUTE_TURN_GENERATION_JOB_LOG", jobLogPath)
	t.Setenv("AM_LIVE_ROUTE_TURN_CANDIDATE_SHELL_LOG", shellLogPath)
	t.Setenv("AM_LIVE_ROUTE_TURN_CANDIDATE_EXECUTION_LOG", executionLogPath)
	t.Setenv("AM_LIVE_ROUTE_TURN_GENERATOR_ADAPTER_LOG", adapterLogPath)
	t.Setenv("AM_LIVE_ROUTE_TURN_CANDIDATE_DRAFT_LOG", draftLogPath)
	t.Setenv("AM_LIVE_ROUTE_TURN_REVIEW_LOG", reviewLogPath)
	t.Setenv("AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_LOG", admissionLogPath)
	t.Setenv("AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_ADAPTER_LOG", admissionAdapterLogPath)

	if err := runAdmissionLiveRouteTurnRouteBoundarySmoke(); err != nil {
		t.Fatal(err)
	}

	type routeBoundaryReceipt struct {
		BodyInventoryStatus     string   `json:"body_inventory_status"`
		RouteAvailabilityStatus string   `json:"route_availability_status"`
		RouteAvailabilityReason string   `json:"route_availability_reason"`
		RouteMissingOrgans      []string `json:"route_missing_organs"`
		JobID                   string   `json:"job_id"`
		ShellID                 string   `json:"shell_id"`
		ExecutionID             string   `json:"execution_id"`
		CandidateExecutionID    string   `json:"candidate_execution_id"`
		AdapterID               string   `json:"adapter_id"`
		GeneratorAdapterID      string   `json:"generator_adapter_id"`
		DraftID                 string   `json:"draft_id"`
		CandidateDraftID        string   `json:"candidate_draft_id"`
		HandoffID               string   `json:"handoff_id"`
		AdmissionAdapterID      string   `json:"admission_adapter_id"`
		DreamCandidateRunID     string   `json:"dream_candidate_run_id"`
		Passed                  bool     `json:"passed"`
		Reason                  string   `json:"reason"`
	}
	for _, path := range []string{
		jobLogPath,
		shellLogPath,
		executionLogPath,
		adapterLogPath,
		draftLogPath,
		reviewLogPath,
		admissionLogPath,
		admissionAdapterLogPath,
	} {
		raw, err := os.ReadFile(path)
		if err != nil {
			t.Fatal(err)
		}
		var got routeBoundaryReceipt
		if err := json.Unmarshal([]byte(strings.TrimSpace(string(raw))), &got); err != nil {
			t.Fatal(err)
		}
		if got.Passed ||
			got.JobID != "" ||
			got.ShellID != "" ||
			got.ExecutionID != "" ||
			got.CandidateExecutionID != "" ||
			got.AdapterID != "" ||
			got.GeneratorAdapterID != "" ||
			got.DraftID != "" ||
			got.CandidateDraftID != "" ||
			got.HandoffID != "" ||
			got.AdmissionAdapterID != "" ||
			got.DreamCandidateRunID != "" ||
			got.BodyInventoryStatus != "blocked" ||
			got.RouteAvailabilityStatus != "unavailable" ||
			got.RouteAvailabilityReason != "missing_route_organs:chorus-binary,nano-weight" ||
			!sameStrings(got.RouteMissingOrgans, []string{"chorus-binary", "nano-weight"}) ||
			!strings.Contains(got.Reason, "route chorus unavailable in body inventory") {
			t.Fatalf("receipt should carry typed route boundary from %s: %+v", path, got)
		}
	}
}

func TestAdmissionLiveRouteTurnCandidateShellSmokeWritesShells(t *testing.T) {
	t.Setenv("AM_LIVE_ROUTE_TURN_CANDIDATE_SHELL_DRY_RUN", "1")
	logPath := filepath.Join(t.TempDir(), "live-route-candidate-shell.jsonl")
	t.Setenv("AM_LIVE_ROUTE_TURN_CANDIDATE_SHELL_LOG", logPath)

	if err := runAdmissionLiveRouteTurnCandidateShellSmoke(); err != nil {
		t.Fatal(err)
	}

	raw, err := os.ReadFile(logPath)
	if err != nil {
		t.Fatal(err)
	}
	lines := strings.Split(strings.TrimSpace(string(raw)), "\n")
	if len(lines) != 5 {
		t.Fatalf("expected 5 candidate shells, got %d: %s", len(lines), raw)
	}
	var identity, unknown admissionLiveRouteTurnCandidateShell
	if err := json.Unmarshal([]byte(lines[0]), &identity); err != nil {
		t.Fatal(err)
	}
	if err := json.Unmarshal([]byte(lines[len(lines)-1]), &unknown); err != nil {
		t.Fatal(err)
	}
	if identity.Schema != admissionLiveRouteTurnCandidateShellSchema ||
		identity.PromptClass != "identity" ||
		identity.Route != "chorus" ||
		identity.Source != "chorus" ||
		identity.Backend != "chorus-arianna" ||
		identity.Entrypoint != "field" ||
		identity.PromptFrame != "q_a" ||
		identity.CandidateSchema != "arianna.dream_candidate.v1" ||
		identity.CandidateKind != "chorus" ||
		identity.CandidateTextStatus != "pending_generation" ||
		!strings.HasPrefix(identity.CandidateSeed, "turn-") ||
		!strings.HasPrefix(identity.JobID, "job-") ||
		!strings.HasPrefix(identity.ShellID, "shell-") ||
		!identity.Passed {
		t.Fatalf("bad identity candidate shell: %+v", identity)
	}
	if unknown.PromptClass != "unknown" ||
		unknown.Passed ||
		unknown.ShellID != "" ||
		!strings.Contains(unknown.Reason, "unknown_prompt_class") ||
		!strings.HasPrefix(unknown.CandidateSeed, "turn-") {
		t.Fatalf("unknown candidate shell should fail closed without runnable shell id: %+v", unknown)
	}
}

func TestAdmissionLiveRouteTurnGeneratorAdapterSmokeWritesAdapters(t *testing.T) {
	t.Setenv("AM_LIVE_ROUTE_TURN_GENERATOR_ADAPTER_DRY_RUN", "1")
	logPath := filepath.Join(t.TempDir(), "live-route-generator-adapter.jsonl")
	t.Setenv("AM_LIVE_ROUTE_TURN_GENERATOR_ADAPTER_LOG", logPath)

	if err := runAdmissionLiveRouteTurnGeneratorAdapterSmoke(); err != nil {
		t.Fatal(err)
	}

	raw, err := os.ReadFile(logPath)
	if err != nil {
		t.Fatal(err)
	}
	lines := strings.Split(strings.TrimSpace(string(raw)), "\n")
	if len(lines) != 5 {
		t.Fatalf("expected 5 generator adapters, got %d: %s", len(lines), raw)
	}
	var identity, unknown admissionLiveRouteTurnGeneratorAdapter
	if err := json.Unmarshal([]byte(lines[0]), &identity); err != nil {
		t.Fatal(err)
	}
	if err := json.Unmarshal([]byte(lines[len(lines)-1]), &unknown); err != nil {
		t.Fatal(err)
	}
	if identity.Schema != admissionLiveRouteTurnGeneratorAdapterSchema ||
		identity.PromptClass != "identity" ||
		identity.Route != "chorus" ||
		identity.Source != "chorus" ||
		identity.Backend != "chorus-arianna" ||
		identity.Entrypoint != "field" ||
		identity.PromptFrame != "q_a" ||
		identity.CandidateSchema != "arianna.dream_candidate.v1" ||
		identity.CandidateKind != "chorus" ||
		identity.CandidateTextStatus != "pending_generation" ||
		identity.GeneratedTextStatus != "generated" ||
		identity.GeneratedText == "" ||
		identity.GeneratedTextHash == "" ||
		!strings.HasPrefix(identity.CandidateSeed, "turn-") ||
		!strings.HasPrefix(identity.JobID, "job-") ||
		!strings.HasPrefix(identity.ShellID, "shell-") ||
		!strings.HasPrefix(identity.AdapterID, "adapter-") ||
		!identity.Passed {
		t.Fatalf("bad identity generator adapter: %+v", identity)
	}
	if unknown.PromptClass != "unknown" ||
		unknown.Passed ||
		unknown.AdapterID != "" ||
		!strings.Contains(unknown.Reason, "unknown_prompt_class") ||
		!strings.HasPrefix(unknown.CandidateSeed, "turn-") {
		t.Fatalf("unknown generator adapter should fail closed without adapter id: %+v", unknown)
	}
}

func TestAdmissionLiveRouteTurnCandidateExecutionSmokeWritesExecutions(t *testing.T) {
	t.Setenv("AM_LIVE_ROUTE_TURN_CANDIDATE_EXECUTION_DRY_RUN", "1")
	logPath := filepath.Join(t.TempDir(), "live-route-candidate-execution.jsonl")
	t.Setenv("AM_LIVE_ROUTE_TURN_CANDIDATE_EXECUTION_LOG", logPath)

	if err := runAdmissionLiveRouteTurnCandidateExecutionSmoke(); err != nil {
		t.Fatal(err)
	}

	raw, err := os.ReadFile(logPath)
	if err != nil {
		t.Fatal(err)
	}
	lines := strings.Split(strings.TrimSpace(string(raw)), "\n")
	if len(lines) != 5 {
		t.Fatalf("expected 5 candidate executions, got %d: %s", len(lines), raw)
	}
	var identity, unknown admissionLiveRouteTurnCandidateExecution
	if err := json.Unmarshal([]byte(lines[0]), &identity); err != nil {
		t.Fatal(err)
	}
	if err := json.Unmarshal([]byte(lines[len(lines)-1]), &unknown); err != nil {
		t.Fatal(err)
	}
	if identity.Schema != admissionLiveRouteTurnCandidateExecutionSchema ||
		identity.PromptClass != "identity" ||
		identity.Route != "chorus" ||
		identity.Source != "chorus" ||
		identity.Backend != "chorus-arianna" ||
		identity.Entrypoint != "field" ||
		identity.PromptFrame != "q_a" ||
		identity.Executor != "chorus-arianna:field:q_a" ||
		identity.TimeoutMS != admissionLiveRouteTurnCandidateExecutionDefaultTimeoutMS ||
		identity.Runner != admissionLiveRouteTurnCandidateExecutionRunnerProvided ||
		identity.RunnerStatus != admissionLiveRouteTurnCandidateExecutionStatusProvided ||
		identity.CandidateSchema != "arianna.dream_candidate.v1" ||
		identity.CandidateKind != "chorus" ||
		identity.CandidateTextStatus != "pending_generation" ||
		identity.GeneratedTextStatus != "generated" ||
		identity.GeneratedText == "" ||
		identity.GeneratedTextHash == "" ||
		identity.RunnerStdoutHash != identity.GeneratedTextHash ||
		!strings.HasPrefix(identity.CandidateSeed, "turn-") ||
		!strings.HasPrefix(identity.JobID, "job-") ||
		!strings.HasPrefix(identity.ShellID, "shell-") ||
		!strings.HasPrefix(identity.ExecutionID, "execution-") ||
		!identity.Passed {
		t.Fatalf("bad identity candidate execution: %+v", identity)
	}
	if unknown.PromptClass != "unknown" ||
		unknown.Passed ||
		unknown.ExecutionID != "" ||
		!strings.Contains(unknown.Reason, "unknown_prompt_class") ||
		!strings.HasPrefix(unknown.CandidateSeed, "turn-") {
		t.Fatalf("unknown candidate execution should fail closed without execution id: %+v", unknown)
	}
}

func TestAdmissionLiveRouteTurnCandidateDraftSmokeWritesDrafts(t *testing.T) {
	t.Setenv("AM_LIVE_ROUTE_TURN_CANDIDATE_DRAFT_DRY_RUN", "1")
	logPath := filepath.Join(t.TempDir(), "live-route-candidate-draft.jsonl")
	t.Setenv("AM_LIVE_ROUTE_TURN_CANDIDATE_DRAFT_LOG", logPath)

	if err := runAdmissionLiveRouteTurnCandidateDraftSmoke(); err != nil {
		t.Fatal(err)
	}

	raw, err := os.ReadFile(logPath)
	if err != nil {
		t.Fatal(err)
	}
	lines := strings.Split(strings.TrimSpace(string(raw)), "\n")
	if len(lines) != 5 {
		t.Fatalf("expected 5 candidate drafts, got %d: %s", len(lines), raw)
	}
	var identity, unknown admissionLiveRouteTurnCandidateDraft
	if err := json.Unmarshal([]byte(lines[0]), &identity); err != nil {
		t.Fatal(err)
	}
	if err := json.Unmarshal([]byte(lines[len(lines)-1]), &unknown); err != nil {
		t.Fatal(err)
	}
	if identity.Schema != admissionLiveRouteTurnCandidateDraftSchema ||
		identity.PromptClass != "identity" ||
		identity.Route != "chorus" ||
		identity.Source != "chorus" ||
		identity.CandidateSchema != "arianna.dream_candidate.v1" ||
		identity.CandidateKind != "chorus" ||
		identity.CandidateTextStatus != "generated" ||
		identity.CandidateText == "" ||
		identity.CandidateTextHash == "" ||
		identity.CandidateRunID == "" ||
		!strings.HasPrefix(identity.CandidateSeed, "turn-") ||
		!strings.HasPrefix(identity.JobID, "job-") ||
		!strings.HasPrefix(identity.ShellID, "shell-") ||
		!strings.HasPrefix(identity.GeneratorAdapterID, "adapter-") ||
		!strings.HasPrefix(identity.DraftID, "draft-") ||
		!identity.Passed {
		t.Fatalf("bad identity candidate draft: %+v", identity)
	}
	if unknown.PromptClass != "unknown" ||
		unknown.Passed ||
		unknown.DraftID != "" ||
		unknown.GeneratorAdapterID != "" ||
		!strings.Contains(unknown.Reason, "unknown_prompt_class") ||
		!strings.HasPrefix(unknown.CandidateSeed, "turn-") {
		t.Fatalf("unknown candidate draft should fail closed without runnable draft id: %+v", unknown)
	}
}

func TestAdmissionLiveRouteTurnCandidateDraftReviewSmokeWritesReviews(t *testing.T) {
	t.Setenv("AM_LIVE_ROUTE_TURN_CANDIDATE_DRAFT_DRY_RUN", "1")
	t.Setenv("AM_DREAM_ADMISSION_LIVE_ROUTE_CHOICE_DRY_RUN", "1")
	logPath := filepath.Join(t.TempDir(), "live-route-candidate-draft-review.jsonl")
	t.Setenv("AM_LIVE_ROUTE_TURN_REVIEW_LOG", logPath)

	if err := runAdmissionLiveRouteTurnCandidateDraftReviewSmoke(); err != nil {
		t.Fatal(err)
	}

	raw, err := os.ReadFile(logPath)
	if err != nil {
		t.Fatal(err)
	}
	lines := strings.Split(strings.TrimSpace(string(raw)), "\n")
	if len(lines) != 5 {
		t.Fatalf("expected 5 candidate draft reviews, got %d: %s", len(lines), raw)
	}
	var matched, mismatch, failed admissionLiveRouteTurnCandidateReview
	if err := json.Unmarshal([]byte(lines[0]), &matched); err != nil {
		t.Fatal(err)
	}
	if err := json.Unmarshal([]byte(lines[2]), &mismatch); err != nil {
		t.Fatal(err)
	}
	if err := json.Unmarshal([]byte(lines[4]), &failed); err != nil {
		t.Fatal(err)
	}
	if matched.Schema != admissionLiveRouteTurnReviewSchema ||
		!matched.Matched ||
		matched.CandidateDraftID == "" ||
		matched.GeneratorAdapterID == "" ||
		matched.CandidateTextStatus != "generated" ||
		matched.CandidateTextHash == "" {
		t.Fatalf("bad matched candidate draft review: %+v", matched)
	}
	if mismatch.Matched || mismatch.CandidateSource != "direct" ||
		!strings.Contains(mismatch.Reason, "candidate_source_mismatch") {
		t.Fatalf("bad mismatched candidate draft review: %+v", mismatch)
	}
	if failed.Matched || failed.CandidateDraftID != "" ||
		!strings.Contains(failed.Reason, "candidate_draft_failed") {
		t.Fatalf("failed candidate draft review should fail closed before route admission: %+v", failed)
	}
}

func TestAdmissionLiveRouteTurnCandidateAdmissionSmokeWritesHandoffs(t *testing.T) {
	t.Setenv("AM_LIVE_ROUTE_TURN_CANDIDATE_DRAFT_DRY_RUN", "1")
	t.Setenv("AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_DRY_RUN", "1")
	logPath := filepath.Join(t.TempDir(), "live-route-candidate-admission.jsonl")
	t.Setenv("AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_LOG", logPath)

	if err := runAdmissionLiveRouteTurnCandidateAdmissionSmoke(); err != nil {
		t.Fatal(err)
	}

	raw, err := os.ReadFile(logPath)
	if err != nil {
		t.Fatal(err)
	}
	lines := strings.Split(strings.TrimSpace(string(raw)), "\n")
	if len(lines) != 5 {
		t.Fatalf("expected 5 candidate admission handoffs, got %d: %s", len(lines), raw)
	}
	var matched, mismatch, failed admissionLiveRouteTurnCandidateAdmission
	if err := json.Unmarshal([]byte(lines[0]), &matched); err != nil {
		t.Fatal(err)
	}
	if err := json.Unmarshal([]byte(lines[2]), &mismatch); err != nil {
		t.Fatal(err)
	}
	if err := json.Unmarshal([]byte(lines[4]), &failed); err != nil {
		t.Fatal(err)
	}
	if matched.Schema != admissionLiveRouteTurnCandidateAdmissionSchema ||
		!matched.Passed ||
		!strings.HasPrefix(matched.HandoffID, "handoff-") ||
		!strings.HasPrefix(matched.CandidateDraftID, "draft-") ||
		!strings.HasPrefix(matched.GeneratorAdapterID, "adapter-") ||
		matched.CandidateSchema != "arianna.dream_candidate.v1" ||
		matched.CandidateTextHash == "" ||
		!matched.ReviewMatched {
		t.Fatalf("bad matched candidate admission handoff: %+v", matched)
	}
	if mismatch.Passed || mismatch.HandoffID != "" ||
		!strings.Contains(mismatch.Reason, "candidate_review_failed") {
		t.Fatalf("bad mismatched candidate admission handoff: %+v", mismatch)
	}
	if failed.Passed || failed.HandoffID != "" ||
		!strings.Contains(failed.Reason, "candidate_draft_failed") {
		t.Fatalf("failed candidate admission handoff should fail closed: %+v", failed)
	}
}

func TestAdmissionLiveRouteTurnCandidateAdmissionRejectsReviewBoundaryDrift(t *testing.T) {
	obs, draft, review := admissionLiveRouteMatchedBoundaryDraftForTest(t)
	review.RouteAvailabilityReason = "tampered-route-boundary"

	admission := admissionLiveRouteTurnCandidateAdmissionForDraftReview(obs, draft, review)
	if admission.Passed ||
		admission.HandoffID != "" ||
		!strings.Contains(admission.Reason, "candidate_review_route_boundary_mismatch") {
		t.Fatalf("review boundary drift should fail closed before handoff: %+v", admission)
	}
}

func admissionLiveRouteMatchedBoundaryDraftForTest(t *testing.T) (admissionLiveRouteTurnObservation, admissionLiveRouteTurnCandidateDraft, admissionLiveRouteTurnCandidateReview) {
	t.Helper()

	obs := admissionLiveRouteTurnObservationForHuman("Who are you? What is your identity?")
	choice := admissionLiveRouteTurnChoiceForObservation(obs)
	request := admissionLiveRouteTurnRequestForChoice(choice)
	job := admissionLiveRouteTurnGenerationJobForRequest(request)
	job.BodyInventoryStatus = "degraded"
	job.RouteAvailabilityStatus = "available"
	job.RouteAvailabilityReason = "optional_route_organs_missing:goldie-weight"
	job.RouteMissingOrgans = []string{"goldie-weight"}
	shell := admissionLiveRouteTurnCandidateShellForJob(job)
	execution := admissionLiveRouteTurnCandidateExecutionForShell(shell, "I am Arianna, and the chorus keeps the route boundary named.")
	adapter := admissionLiveRouteTurnGeneratorAdapterForExecution(execution)
	draft := admissionLiveRouteTurnCandidateDraftForAdapter(adapter)
	review := admissionLiveRouteTurnCandidateReviewForDraft(obs, draft)

	if !obs.Passed || !choice.Passed || !request.Passed || !job.Passed || !shell.Passed || !execution.Passed || !adapter.Passed || !draft.Passed || !review.Matched {
		t.Fatalf("test setup failed: obs=%+v choice=%+v request=%+v job=%+v shell=%+v execution=%+v adapter=%+v draft=%+v review=%+v",
			obs, choice, request, job, shell, execution, adapter, draft, review)
	}
	return obs, draft, review
}

func TestAdmissionLiveRouteTurnCandidateAdmissionAdapterSmokeWritesAdaptersAndAdmissionReceipts(t *testing.T) {
	t.Setenv("AM_LIVE_ROUTE_TURN_CANDIDATE_DRAFT_DRY_RUN", "1")
	t.Setenv("AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_DRY_RUN", "1")
	t.Setenv("AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_ADAPTER_DRY_RUN", "1")
	t.Setenv("AM_DREAM_ADMISSION", dreamAdmissionShadow)
	t.Setenv("AM_DREAM_ADMISSION_REQUIRE_LIVE_ROUTE_PLAN", "1")
	dir := t.TempDir()
	adapterLog := filepath.Join(dir, "live-route-candidate-admission-adapter.jsonl")
	admissionLog := filepath.Join(dir, "dream-admission.jsonl")
	t.Setenv("AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_ADAPTER_LOG", adapterLog)
	t.Setenv("AM_DREAM_ADMISSION_LOG", admissionLog)

	if err := runAdmissionLiveRouteTurnCandidateAdmissionAdapterSmoke(); err != nil {
		t.Fatal(err)
	}

	raw, err := os.ReadFile(adapterLog)
	if err != nil {
		t.Fatal(err)
	}
	lines := strings.Split(strings.TrimSpace(string(raw)), "\n")
	if len(lines) != 5 {
		t.Fatalf("expected 5 candidate admission adapters, got %d: %s", len(lines), raw)
	}
	var matched, mismatch admissionLiveRouteTurnCandidateAdmissionAdapter
	if err := json.Unmarshal([]byte(lines[0]), &matched); err != nil {
		t.Fatal(err)
	}
	if err := json.Unmarshal([]byte(lines[2]), &mismatch); err != nil {
		t.Fatal(err)
	}
	if matched.Schema != admissionLiveRouteTurnCandidateAdmissionAdapterSchema ||
		!matched.Passed ||
		!strings.HasPrefix(matched.HandoffID, "handoff-") ||
		!strings.HasPrefix(matched.AdmissionAdapterID, "admission-adapter-") ||
		matched.DreamCandidateRunID == "" ||
		matched.DreamCandidateRunID != matched.CandidateRunID ||
		matched.CandidateTextHash == "" {
		t.Fatalf("bad matched candidate admission adapter: %+v", matched)
	}
	if mismatch.Passed || mismatch.AdmissionAdapterID != "" ||
		!strings.Contains(mismatch.Reason, "candidate_admission_handoff_failed") {
		t.Fatalf("bad mismatched candidate admission adapter: %+v", mismatch)
	}

	admissionRaw, err := os.ReadFile(admissionLog)
	if err != nil {
		t.Fatal(err)
	}
	admissionLines := strings.Split(strings.TrimSpace(string(admissionRaw)), "\n")
	if len(admissionLines) != 2 {
		t.Fatalf("expected 2 shadow admission receipts, got %d: %s", len(admissionLines), admissionRaw)
	}
	var got dreamCandidate
	if err := json.Unmarshal([]byte(admissionLines[0]), &got); err != nil {
		t.Fatal(err)
	}
	if got.Accepted ||
		got.Reason != "shadow mode" ||
		got.LiveRouteCandidateAdmission == nil ||
		got.LiveRouteCandidateAdmission.AdmissionAdapterID != matched.AdmissionAdapterID ||
		got.RunID != matched.CandidateRunID ||
		got.Admission == nil ||
		!got.Admission.Passed ||
		got.Admission.LiveRouteChoice == nil ||
		!got.Admission.LiveRouteChoice.Passed {
		t.Fatalf("bad shadow admission receipt from adapter: %+v", got)
	}
}

func TestAdmissionLiveRouteTurnCandidateAdmissionAdapterRejectsAdmissionBoundaryDrift(t *testing.T) {
	obs, draft, review := admissionLiveRouteMatchedBoundaryDraftForTest(t)
	admission := admissionLiveRouteTurnCandidateAdmissionForDraftReview(obs, draft, review)
	if !admission.Passed {
		t.Fatalf("test setup admission failed: %+v", admission)
	}
	admission.RouteMissingOrgans = append(admissionLiveRouteMissingOrgansCopy(admission.RouteMissingOrgans), "doe-bridge")

	adapter := admissionLiveRouteTurnCandidateAdmissionAdapterForDraft(admission, draft)
	if adapter.Passed ||
		adapter.AdmissionAdapterID != "" ||
		!strings.Contains(adapter.Reason, "candidate_admission_route_boundary_mismatch") {
		t.Fatalf("admission boundary drift should fail closed before adapter id: %+v", adapter)
	}
}

func TestAdmissionLiveRouteTurnCandidateAdmissionChatSmokeWritesChatReceipts(t *testing.T) {
	t.Setenv("AM_LIVE_ROUTE_TURN_CANDIDATE_DRAFT_DRY_RUN", "1")
	t.Setenv("AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_DRY_RUN", "1")
	t.Setenv("AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_ADAPTER_DRY_RUN", "1")
	t.Setenv("AM_LIVE_ROUTE_TURN_CANDIDATE_DRAFT_TEXT", "I am Arianna, and the chat path keeps the adapter named.")
	dir := t.TempDir()
	draftLog := filepath.Join(dir, "live-route-candidate-draft-chat.jsonl")
	reviewLog := filepath.Join(dir, "live-route-candidate-draft-review-chat.jsonl")
	admissionLog := filepath.Join(dir, "live-route-candidate-admission-chat.jsonl")
	adapterLog := filepath.Join(dir, "live-route-candidate-admission-adapter-chat.jsonl")
	t.Setenv("AM_LIVE_ROUTE_TURN_CANDIDATE_DRAFT_LOG", draftLog)
	t.Setenv("AM_LIVE_ROUTE_TURN_REVIEW_LOG", reviewLog)
	t.Setenv("AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_LOG", admissionLog)
	t.Setenv("AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_ADAPTER_LOG", adapterLog)

	if err := runAdmissionLiveRouteTurnCandidateAdmissionChatSmoke(); err != nil {
		t.Fatal(err)
	}

	draftRaw, err := os.ReadFile(draftLog)
	if err != nil {
		t.Fatal(err)
	}
	draftLines := strings.Split(strings.TrimSpace(string(draftRaw)), "\n")
	if len(draftLines) != 2 {
		t.Fatalf("expected 2 chat draft receipts, got %d: %s", len(draftLines), draftRaw)
	}
	var draft admissionLiveRouteTurnCandidateDraft
	if err := json.Unmarshal([]byte(draftLines[0]), &draft); err != nil {
		t.Fatal(err)
	}
	if !draft.Passed ||
		!strings.HasPrefix(draft.DraftID, "draft-") ||
		!strings.HasPrefix(draft.GeneratorAdapterID, "adapter-") ||
		draft.PromptClass != "identity" ||
		draft.Route != "chorus" ||
		draft.Source != "chorus" {
		t.Fatalf("bad chat draft receipt: %+v", draft)
	}

	reviewRaw, err := os.ReadFile(reviewLog)
	if err != nil {
		t.Fatal(err)
	}
	reviewLines := strings.Split(strings.TrimSpace(string(reviewRaw)), "\n")
	if len(reviewLines) != 2 {
		t.Fatalf("expected 2 chat review receipts, got %d: %s", len(reviewLines), reviewRaw)
	}
	var review admissionLiveRouteTurnCandidateReview
	if err := json.Unmarshal([]byte(reviewLines[0]), &review); err != nil {
		t.Fatal(err)
	}
	if !review.Matched ||
		review.CandidateDraftID != draft.DraftID ||
		review.GeneratorAdapterID != draft.GeneratorAdapterID ||
		review.CandidateRunID != draft.CandidateRunID {
		t.Fatalf("bad chat review receipt: %+v", review)
	}

	admissionRaw, err := os.ReadFile(admissionLog)
	if err != nil {
		t.Fatal(err)
	}
	admissionLines := strings.Split(strings.TrimSpace(string(admissionRaw)), "\n")
	if len(admissionLines) != 2 {
		t.Fatalf("expected 2 chat handoff receipts, got %d: %s", len(admissionLines), admissionRaw)
	}
	var admission admissionLiveRouteTurnCandidateAdmission
	if err := json.Unmarshal([]byte(admissionLines[0]), &admission); err != nil {
		t.Fatal(err)
	}
	if !admission.Passed ||
		!strings.HasPrefix(admission.HandoffID, "handoff-") ||
		admission.CandidateDraftID != draft.DraftID ||
		admission.GeneratorAdapterID != draft.GeneratorAdapterID ||
		!admission.ReviewMatched {
		t.Fatalf("bad chat handoff receipt: %+v", admission)
	}

	adapterRaw, err := os.ReadFile(adapterLog)
	if err != nil {
		t.Fatal(err)
	}
	adapterLines := strings.Split(strings.TrimSpace(string(adapterRaw)), "\n")
	if len(adapterLines) != 2 {
		t.Fatalf("expected 2 chat adapter receipts, got %d: %s", len(adapterLines), adapterRaw)
	}
	var adapter, failed admissionLiveRouteTurnCandidateAdmissionAdapter
	if err := json.Unmarshal([]byte(adapterLines[0]), &adapter); err != nil {
		t.Fatal(err)
	}
	if err := json.Unmarshal([]byte(adapterLines[1]), &failed); err != nil {
		t.Fatal(err)
	}
	if !adapter.Passed ||
		!strings.HasPrefix(adapter.AdmissionAdapterID, "admission-adapter-") ||
		adapter.HandoffID != admission.HandoffID ||
		adapter.CandidateDraftID != draft.DraftID ||
		adapter.DreamCandidateRunID != draft.CandidateRunID {
		t.Fatalf("bad chat adapter receipt: %+v", adapter)
	}
	if failed.Passed ||
		failed.AdmissionAdapterID != "" ||
		!strings.Contains(failed.Reason, "unknown_prompt_class") {
		t.Fatalf("bad failed chat adapter receipt: %+v", failed)
	}
}

func TestAdmissionLiveRouteTurnCandidateAdmissionChatShadowSmokeWritesAdmissionReceipt(t *testing.T) {
	t.Setenv("AM_LIVE_ROUTE_TURN_CANDIDATE_DRAFT_DRY_RUN", "1")
	t.Setenv("AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_DRY_RUN", "1")
	t.Setenv("AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_ADAPTER_DRY_RUN", "1")
	t.Setenv("AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_SHADOW_DRY_RUN", "1")
	t.Setenv("AM_DREAM_ADMISSION", dreamAdmissionShadow)
	t.Setenv("AM_DREAM_ADMISSION_REQUIRE_LIVE_ROUTE_PLAN", "1")
	t.Setenv("AM_LIVE_ROUTE_TURN_CANDIDATE_DRAFT_TEXT", "I am Arianna, and the chat shadow path keeps the adapter named.")
	dir := t.TempDir()
	draftLog := filepath.Join(dir, "live-route-candidate-draft-chat-shadow.jsonl")
	reviewLog := filepath.Join(dir, "live-route-candidate-draft-review-chat-shadow.jsonl")
	admissionLog := filepath.Join(dir, "live-route-candidate-admission-chat-shadow.jsonl")
	adapterLog := filepath.Join(dir, "live-route-candidate-admission-adapter-chat-shadow.jsonl")
	dreamLog := filepath.Join(dir, "dream-admission-chat-shadow.jsonl")
	t.Setenv("AM_LIVE_ROUTE_TURN_CANDIDATE_DRAFT_LOG", draftLog)
	t.Setenv("AM_LIVE_ROUTE_TURN_REVIEW_LOG", reviewLog)
	t.Setenv("AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_LOG", admissionLog)
	t.Setenv("AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_ADAPTER_LOG", adapterLog)
	t.Setenv("AM_DREAM_ADMISSION_LOG", dreamLog)

	if err := runAdmissionLiveRouteTurnCandidateAdmissionChatShadowSmoke(); err != nil {
		t.Fatal(err)
	}

	adapterRaw, err := os.ReadFile(adapterLog)
	if err != nil {
		t.Fatal(err)
	}
	adapterLines := strings.Split(strings.TrimSpace(string(adapterRaw)), "\n")
	if len(adapterLines) != 2 {
		t.Fatalf("expected 2 chat shadow adapter receipts, got %d: %s", len(adapterLines), adapterRaw)
	}
	var adapter admissionLiveRouteTurnCandidateAdmissionAdapter
	if err := json.Unmarshal([]byte(adapterLines[0]), &adapter); err != nil {
		t.Fatal(err)
	}
	if !adapter.Passed || adapter.AdmissionAdapterID == "" {
		t.Fatalf("bad chat shadow adapter receipt: %+v", adapter)
	}

	dreamRaw, err := os.ReadFile(dreamLog)
	if err != nil {
		t.Fatal(err)
	}
	dreamLines := strings.Split(strings.TrimSpace(string(dreamRaw)), "\n")
	if len(dreamLines) != 1 {
		t.Fatalf("expected 1 chat shadow admission receipt, got %d: %s", len(dreamLines), dreamRaw)
	}
	var candidate dreamCandidate
	if err := json.Unmarshal([]byte(dreamLines[0]), &candidate); err != nil {
		t.Fatal(err)
	}
	if candidate.LiveRouteCandidateAdmission == nil ||
		candidate.LiveRouteCandidateAdmission.AdmissionAdapterID != adapter.AdmissionAdapterID ||
		candidate.LiveRouteCandidateAdmission.HandoffID != adapter.HandoffID ||
		candidate.Admission == nil ||
		!candidate.Admission.Passed ||
		candidate.Admission.LiveRouteChoice == nil ||
		!candidate.Admission.LiveRouteChoice.Passed ||
		candidate.Accepted ||
		candidate.Reason != "shadow mode" {
		t.Fatalf("bad chat shadow admission receipt: %+v", candidate)
	}
}

func TestAdmissionLiveRouteTurnReviewSmokeWritesReviews(t *testing.T) {
	t.Setenv("AM_DREAM_ADMISSION_LIVE_ROUTE_CHOICE_DRY_RUN", "1")
	logPath := filepath.Join(t.TempDir(), "live-route-turn-review.jsonl")
	t.Setenv("AM_LIVE_ROUTE_TURN_REVIEW_LOG", logPath)

	if err := runAdmissionLiveRouteTurnReviewSmoke(); err != nil {
		t.Fatal(err)
	}

	raw, err := os.ReadFile(logPath)
	if err != nil {
		t.Fatal(err)
	}
	lines := strings.Split(strings.TrimSpace(string(raw)), "\n")
	if len(lines) != 4 {
		t.Fatalf("expected 4 turn/candidate reviews, got %d: %s", len(lines), raw)
	}
	var matched, untyped admissionLiveRouteTurnCandidateReview
	if err := json.Unmarshal([]byte(lines[0]), &matched); err != nil {
		t.Fatal(err)
	}
	if err := json.Unmarshal([]byte(lines[2]), &untyped); err != nil {
		t.Fatal(err)
	}
	if matched.Schema != admissionLiveRouteTurnReviewSchema || !matched.Matched ||
		matched.TurnExpectedSource != "chorus" || matched.CandidateSource != "chorus" {
		t.Fatalf("bad matched review: %+v", matched)
	}
	if untyped.Matched || untyped.CandidatePromptClass != "human-turn" ||
		!strings.Contains(untyped.Reason, "unknown_prompt_class") {
		t.Fatalf("bad untyped nano review: %+v", untyped)
	}
}

func TestAdmissionLiveRouteTurnBridgeSmokeWritesBridgeReviews(t *testing.T) {
	t.Setenv("AM_DREAM_ADMISSION_LIVE_ROUTE_CHOICE_DRY_RUN", "1")
	t.Setenv("AM_LIVE_ROUTE_TURN_BRIDGE_DRY_RUN", "1")
	logPath := filepath.Join(t.TempDir(), "live-route-turn-bridge.jsonl")
	t.Setenv("AM_LIVE_ROUTE_TURN_REVIEW_LOG", logPath)

	if err := runAdmissionLiveRouteTurnBridgeSmoke(); err != nil {
		t.Fatal(err)
	}

	raw, err := os.ReadFile(logPath)
	if err != nil {
		t.Fatal(err)
	}
	lines := strings.Split(strings.TrimSpace(string(raw)), "\n")
	if len(lines) != 4 {
		t.Fatalf("expected 4 turn bridge reviews, got %d: %s", len(lines), raw)
	}
	var bridged, matched int
	for i, line := range lines {
		var got admissionLiveRouteTurnCandidateReview
		if err := json.Unmarshal([]byte(line), &got); err != nil {
			t.Fatalf("bridge review %d: %v", i+1, err)
		}
		if got.Matched {
			matched++
		}
		if got.CandidateBridgeApplied {
			bridged++
			if got.CandidateTrigger != "human-turn" ||
				got.CandidateSource != "nano" ||
				!strings.HasPrefix(got.CandidateBridgeTrigger, "human-turn-") ||
				!strings.Contains(got.Reason, "source nano does not match live route") {
				t.Fatalf("bad bridged review %d: %+v", i+1, got)
			}
		}
	}
	if bridged != 2 || matched != 1 {
		t.Fatalf("bad bridge counts: bridged=%d matched=%d log=%s", bridged, matched, raw)
	}
	if !strings.Contains(string(raw), "\"candidate_bridge_trigger\":\"human-turn-identity\"") ||
		!strings.Contains(string(raw), "\"candidate_bridge_trigger\":\"human-turn-direct-user\"") {
		t.Fatalf("bridge triggers missing from log: %s", raw)
	}
}

func TestAdmissionLiveRouteTurnBridgeAdmissionSmokeWritesAdmissionReceipt(t *testing.T) {
	t.Setenv("AM_DREAM_ADMISSION", dreamAdmissionShadow)
	t.Setenv("AM_DREAM_ADMISSION_LIVE_ROUTE_CHOICE_DRY_RUN", "1")
	t.Setenv("AM_LIVE_ROUTE_TURN_BRIDGE_DRY_RUN", "1")
	logPath := filepath.Join(t.TempDir(), "live-route-turn-bridge-admission.jsonl")
	t.Setenv("AM_DREAM_ADMISSION_LOG", logPath)

	if err := runAdmissionLiveRouteTurnBridgeAdmissionSmoke(); err != nil {
		t.Fatal(err)
	}

	raw, err := os.ReadFile(logPath)
	if err != nil {
		t.Fatal(err)
	}
	var got dreamCandidate
	if err := json.Unmarshal([]byte(strings.TrimSpace(string(raw))), &got); err != nil {
		t.Fatal(err)
	}
	if got.Trigger != "human-turn" || got.Source != "nano" ||
		got.Admission == nil || !got.Admission.LiveRouteTurnBridgeApplied ||
		got.Admission.LiveRouteBridgeTrigger != "human-turn-identity" ||
		got.Admission.LiveRouteChoice == nil ||
		got.Admission.LiveRouteChoice.PromptClass != "identity" ||
		got.Admission.LiveRouteChoice.Source != "nano" ||
		got.Admission.LiveRouteChoice.ExpectedSource != "chorus" ||
		got.Admission.LiveRouteChoice.Passed {
		t.Fatalf("bad turn bridge admission receipt: %+v", got)
	}
}
