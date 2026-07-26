package main

import (
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
