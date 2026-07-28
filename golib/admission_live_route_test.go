package main

import (
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

func TestAdmissionLiveRouteTurnCandidateAdmissionDecisionForShadow(t *testing.T) {
	t.Setenv("AM_DREAM_ADMISSION", dreamAdmissionShadow)
	t.Setenv("AM_DREAM_ADMISSION_REQUIRE_LIVE_ROUTE_PLAN", "1")

	text := "The dream remembers the field and keeps one admission chain."
	obs := admissionLiveRouteTurnObservationForHuman("Tell me what the dream should remember.")
	choice := admissionLiveRouteTurnChoiceForObservation(obs)
	request := admissionLiveRouteTurnRequestForChoice(choice)
	job := admissionLiveRouteTurnGenerationJobForRequest(request)
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
		decision.TurnTextHash != obs.TextHash {
		t.Fatalf("decision lost provenance: decision=%+v execution=%+v adapter=%+v draft=%+v admission=%+v candidate=%+v",
			decision, execution, generatorAdapter, draft, admission, candidate)
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
