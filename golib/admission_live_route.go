package main

import (
	"encoding/json"
	"os"
	"strconv"
	"strings"
)

const (
	admissionLiveRoutePlanSchema                                  = "arianna.live_route_plan.v1"
	admissionLiveRouteChoiceSchema                                = "arianna.live_route_choice.v1"
	admissionLiveRouteTurnObservationSchema                       = "arianna.live_route_turn_observation.v1"
	admissionLiveRouteTurnChoiceSchema                            = "arianna.live_route_turn_choice.v1"
	admissionLiveRouteTurnRequestSchema                           = "arianna.live_route_turn_request.v1"
	admissionLiveRouteTurnGenerationJobSchema                     = "arianna.live_route_turn_generation_job.v1"
	admissionLiveRouteTurnCandidateShellSchema                    = "arianna.live_route_turn_candidate_shell.v1"
	admissionLiveRouteTurnCandidateExecutionSchema                = "arianna.live_route_turn_candidate_execution.v1"
	admissionLiveRouteTurnGeneratorAdapterSchema                  = "arianna.live_route_turn_generator_adapter.v1"
	admissionLiveRouteTurnCandidateDraftSchema                    = "arianna.live_route_turn_candidate_draft.v1"
	admissionLiveRouteTurnReviewSchema                            = "arianna.live_route_turn_candidate_review.v1"
	admissionLiveRouteTurnCandidateAdmissionSchema                = "arianna.live_route_turn_candidate_admission.v1"
	admissionLiveRouteTurnCandidateAdmissionAdapterSchema         = "arianna.live_route_turn_candidate_admission_adapter.v1"
	admissionLiveRouteTurnCandidateAdmissionDecisionSchema        = "arianna.live_route_turn_candidate_admission_decision.v1"
	admissionLiveRouteTurnCandidateAdmissionPromotionSchema       = "arianna.live_route_turn_candidate_admission_promotion.v1"
	admissionLiveRouteTurnCandidateAdmissionSwitchSchema          = "arianna.live_route_turn_candidate_admission_switch.v1"
	admissionLiveRouteTurnCandidateAdmissionEnableGateSchema      = "arianna.live_route_turn_candidate_admission_enable_gate.v1"
	admissionLiveRouteTurnCandidateAdmissionLiveStageSchema       = "arianna.live_route_turn_candidate_admission_live_stage.v1"
	admissionLiveRouteTurnCandidateAdmissionWriterPreflightSchema = "arianna.live_route_turn_candidate_admission_writer_preflight.v1"
	admissionLiveRouteTurnCandidateAdmissionWriterInventorySchema = "arianna.live_route_turn_candidate_admission_writer_inventory.v1"
	admissionLiveRouteTurnCandidateAdmissionWriterContractSchema  = "arianna.live_route_turn_candidate_admission_writer_contract.v1"
	admissionLiveRouteTurnCandidateAdmissionLedgerSchema          = "arianna.live_route_turn_candidate_admission_ledger.v1"

	admissionLiveRouteTurnCandidateExecutionDefaultTimeoutMS       = 12000
	admissionLiveRouteTurnCandidateExecutionMaxTimeoutMS           = 60000
	admissionLiveRouteTurnCandidateAdmissionEnableGateConfirmation = "ARIANNA_LIVE_ADMISSION_ENABLE_DRY_RUN_ONLY"

	admissionLiveRouteTurnCandidateExecutionRunnerProvided   = "provided_text"
	admissionLiveRouteTurnCandidateExecutionRunnerSelfEmit   = "metabolism-self-emit"
	admissionLiveRouteTurnCandidateExecutionRunnerNanoDirect = "nano-direct"

	admissionLiveRouteTurnCandidateExecutionStatusProvided  = "provided"
	admissionLiveRouteTurnCandidateExecutionStatusSucceeded = "succeeded"
	admissionLiveRouteTurnCandidateExecutionStatusFailed    = "failed"
	admissionLiveRouteTurnCandidateExecutionStatusTimedOut  = "timed_out"
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

type admissionLiveRouteTurnChoice struct {
	Schema           string `json:"schema"`
	PromptClass      string `json:"prompt_class"`
	Route            string `json:"route,omitempty"`
	Source           string `json:"source,omitempty"`
	ExpectedSource   string `json:"expected_source,omitempty"`
	CandidateTrigger string `json:"candidate_trigger,omitempty"`
	Passed           bool   `json:"passed"`
	Reason           string `json:"reason,omitempty"`
	TurnTextHash     string `json:"turn_text_hash,omitempty"`

	Plan admissionLiveRoutePlan `json:"-"`
}

type admissionLiveRouteTurnRequest struct {
	Schema           string `json:"schema"`
	PromptClass      string `json:"prompt_class"`
	Route            string `json:"route,omitempty"`
	Source           string `json:"source,omitempty"`
	ExpectedSource   string `json:"expected_source,omitempty"`
	CandidateTrigger string `json:"candidate_trigger,omitempty"`
	CandidateSeed    string `json:"candidate_seed,omitempty"`
	Passed           bool   `json:"passed"`
	Reason           string `json:"reason,omitempty"`
	TurnTextHash     string `json:"turn_text_hash,omitempty"`
}

type admissionLiveRouteTurnGenerationJob struct {
	Schema           string `json:"schema"`
	PromptClass      string `json:"prompt_class"`
	Route            string `json:"route,omitempty"`
	Source           string `json:"source,omitempty"`
	ExpectedSource   string `json:"expected_source,omitempty"`
	Backend          string `json:"backend,omitempty"`
	Entrypoint       string `json:"entrypoint,omitempty"`
	PromptFrame      string `json:"prompt_frame,omitempty"`
	CandidateTrigger string `json:"candidate_trigger,omitempty"`
	CandidateSeed    string `json:"candidate_seed,omitempty"`
	JobID            string `json:"job_id,omitempty"`
	Passed           bool   `json:"passed"`
	Reason           string `json:"reason,omitempty"`
	TurnTextHash     string `json:"turn_text_hash,omitempty"`
}

type admissionLiveRouteTurnCandidateShell struct {
	Schema              string `json:"schema"`
	PromptClass         string `json:"prompt_class"`
	Route               string `json:"route,omitempty"`
	Source              string `json:"source,omitempty"`
	ExpectedSource      string `json:"expected_source,omitempty"`
	Backend             string `json:"backend,omitempty"`
	Entrypoint          string `json:"entrypoint,omitempty"`
	PromptFrame         string `json:"prompt_frame,omitempty"`
	CandidateSchema     string `json:"candidate_schema,omitempty"`
	CandidateKind       string `json:"candidate_kind,omitempty"`
	CandidateTrigger    string `json:"candidate_trigger,omitempty"`
	CandidateSeed       string `json:"candidate_seed,omitempty"`
	CandidateTextStatus string `json:"candidate_text_status,omitempty"`
	JobID               string `json:"job_id,omitempty"`
	ShellID             string `json:"shell_id,omitempty"`
	Passed              bool   `json:"passed"`
	Reason              string `json:"reason,omitempty"`
	TurnTextHash        string `json:"turn_text_hash,omitempty"`
}

type admissionLiveRouteTurnCandidateExecution struct {
	Schema              string `json:"schema"`
	PromptClass         string `json:"prompt_class"`
	Route               string `json:"route,omitempty"`
	Source              string `json:"source,omitempty"`
	ExpectedSource      string `json:"expected_source,omitempty"`
	Backend             string `json:"backend,omitempty"`
	Entrypoint          string `json:"entrypoint,omitempty"`
	PromptFrame         string `json:"prompt_frame,omitempty"`
	Executor            string `json:"executor,omitempty"`
	TimeoutMS           int    `json:"timeout_ms,omitempty"`
	Runner              string `json:"runner,omitempty"`
	RunnerStatus        string `json:"runner_status,omitempty"`
	RunnerExitCode      int    `json:"runner_exit_code"`
	RunnerTimedOut      bool   `json:"runner_timed_out"`
	RunnerDurationMS    int64  `json:"runner_duration_ms,omitempty"`
	RunnerStdoutHash    string `json:"runner_stdout_hash,omitempty"`
	RunnerStderrHash    string `json:"runner_stderr_hash,omitempty"`
	CandidateSchema     string `json:"candidate_schema,omitempty"`
	CandidateKind       string `json:"candidate_kind,omitempty"`
	CandidateTrigger    string `json:"candidate_trigger,omitempty"`
	CandidateSeed       string `json:"candidate_seed,omitempty"`
	CandidateTextStatus string `json:"candidate_text_status,omitempty"`
	GeneratedText       string `json:"generated_text,omitempty"`
	GeneratedTextHash   string `json:"generated_text_hash,omitempty"`
	GeneratedTextStatus string `json:"generated_text_status,omitempty"`
	JobID               string `json:"job_id,omitempty"`
	ShellID             string `json:"shell_id,omitempty"`
	ExecutionID         string `json:"execution_id,omitempty"`
	Passed              bool   `json:"passed"`
	Reason              string `json:"reason,omitempty"`
	TurnTextHash        string `json:"turn_text_hash,omitempty"`
}

type admissionLiveRouteTurnCandidateExecutionRuntime struct {
	Runner        string
	Status        string
	ExitCode      int
	TimedOut      bool
	DurationMS    int64
	StdoutHash    string
	StderrHash    string
	FailureReason string
}

type admissionLiveRouteTurnGeneratorAdapter struct {
	Schema               string `json:"schema"`
	PromptClass          string `json:"prompt_class"`
	Route                string `json:"route,omitempty"`
	Source               string `json:"source,omitempty"`
	ExpectedSource       string `json:"expected_source,omitempty"`
	Backend              string `json:"backend,omitempty"`
	Entrypoint           string `json:"entrypoint,omitempty"`
	PromptFrame          string `json:"prompt_frame,omitempty"`
	CandidateSchema      string `json:"candidate_schema,omitempty"`
	CandidateKind        string `json:"candidate_kind,omitempty"`
	CandidateTrigger     string `json:"candidate_trigger,omitempty"`
	CandidateSeed        string `json:"candidate_seed,omitempty"`
	CandidateTextStatus  string `json:"candidate_text_status,omitempty"`
	GeneratedText        string `json:"generated_text,omitempty"`
	GeneratedTextHash    string `json:"generated_text_hash,omitempty"`
	GeneratedTextStatus  string `json:"generated_text_status,omitempty"`
	JobID                string `json:"job_id,omitempty"`
	ShellID              string `json:"shell_id,omitempty"`
	CandidateExecutionID string `json:"candidate_execution_id,omitempty"`
	AdapterID            string `json:"adapter_id,omitempty"`
	Passed               bool   `json:"passed"`
	Reason               string `json:"reason,omitempty"`
	TurnTextHash         string `json:"turn_text_hash,omitempty"`
}

type admissionLiveRouteTurnCandidateDraft struct {
	Schema               string `json:"schema"`
	PromptClass          string `json:"prompt_class"`
	Route                string `json:"route,omitempty"`
	Source               string `json:"source,omitempty"`
	ExpectedSource       string `json:"expected_source,omitempty"`
	Backend              string `json:"backend,omitempty"`
	Entrypoint           string `json:"entrypoint,omitempty"`
	PromptFrame          string `json:"prompt_frame,omitempty"`
	CandidateSchema      string `json:"candidate_schema,omitempty"`
	CandidateKind        string `json:"candidate_kind,omitempty"`
	CandidateTrigger     string `json:"candidate_trigger,omitempty"`
	CandidateSeed        string `json:"candidate_seed,omitempty"`
	CandidateTextStatus  string `json:"candidate_text_status,omitempty"`
	CandidateText        string `json:"candidate_text,omitempty"`
	CandidateTextHash    string `json:"candidate_text_hash,omitempty"`
	CandidateRunID       string `json:"candidate_run_id,omitempty"`
	JobID                string `json:"job_id,omitempty"`
	ShellID              string `json:"shell_id,omitempty"`
	CandidateExecutionID string `json:"candidate_execution_id,omitempty"`
	GeneratorAdapterID   string `json:"generator_adapter_id,omitempty"`
	DraftID              string `json:"draft_id,omitempty"`
	Passed               bool   `json:"passed"`
	Reason               string `json:"reason,omitempty"`
	TurnTextHash         string `json:"turn_text_hash,omitempty"`
}

type admissionLiveRouteTurnCandidateReview struct {
	Schema                  string `json:"schema"`
	Timing                  string `json:"timing"`
	TurnPromptClass         string `json:"turn_prompt_class"`
	TurnRoute               string `json:"turn_route,omitempty"`
	TurnExpectedSource      string `json:"turn_expected_source,omitempty"`
	TurnPassed              bool   `json:"turn_passed"`
	CandidateRunID          string `json:"candidate_run_id,omitempty"`
	CandidateDraftID        string `json:"candidate_draft_id,omitempty"`
	CandidateExecutionID    string `json:"candidate_execution_id,omitempty"`
	GeneratorAdapterID      string `json:"generator_adapter_id,omitempty"`
	CandidateTextStatus     string `json:"candidate_text_status,omitempty"`
	CandidateTextHash       string `json:"candidate_text_hash,omitempty"`
	CandidateSource         string `json:"candidate_source,omitempty"`
	CandidateTrigger        string `json:"candidate_trigger,omitempty"`
	CandidateBridgeApplied  bool   `json:"candidate_bridge_applied"`
	CandidateBridgeTrigger  string `json:"candidate_bridge_trigger,omitempty"`
	CandidatePromptClass    string `json:"candidate_prompt_class,omitempty"`
	CandidateRoute          string `json:"candidate_route,omitempty"`
	CandidateExpectedSource string `json:"candidate_expected_source,omitempty"`
	CandidateChoicePassed   bool   `json:"candidate_choice_passed"`
	Matched                 bool   `json:"matched"`
	Reason                  string `json:"reason,omitempty"`
}

type admissionLiveRouteTurnCandidateAdmission struct {
	Schema               string `json:"schema"`
	Timing               string `json:"timing"`
	PromptClass          string `json:"prompt_class"`
	Route                string `json:"route,omitempty"`
	Source               string `json:"source,omitempty"`
	ExpectedSource       string `json:"expected_source,omitempty"`
	CandidateSchema      string `json:"candidate_schema,omitempty"`
	CandidateKind        string `json:"candidate_kind,omitempty"`
	CandidateTrigger     string `json:"candidate_trigger,omitempty"`
	CandidateSeed        string `json:"candidate_seed,omitempty"`
	CandidateRunID       string `json:"candidate_run_id,omitempty"`
	CandidateDraftID     string `json:"candidate_draft_id,omitempty"`
	CandidateExecutionID string `json:"candidate_execution_id,omitempty"`
	GeneratorAdapterID   string `json:"generator_adapter_id,omitempty"`
	CandidateTextStatus  string `json:"candidate_text_status,omitempty"`
	CandidateTextHash    string `json:"candidate_text_hash,omitempty"`
	ReviewMatched        bool   `json:"review_matched"`
	HandoffID            string `json:"handoff_id,omitempty"`
	Passed               bool   `json:"passed"`
	Reason               string `json:"reason,omitempty"`
	TurnTextHash         string `json:"turn_text_hash,omitempty"`
}

type admissionLiveRouteTurnCandidateAdmissionAdapter struct {
	Schema               string `json:"schema"`
	Timing               string `json:"timing"`
	PromptClass          string `json:"prompt_class"`
	Route                string `json:"route,omitempty"`
	Source               string `json:"source,omitempty"`
	ExpectedSource       string `json:"expected_source,omitempty"`
	CandidateSchema      string `json:"candidate_schema,omitempty"`
	CandidateKind        string `json:"candidate_kind,omitempty"`
	CandidateTrigger     string `json:"candidate_trigger,omitempty"`
	CandidateSeed        string `json:"candidate_seed,omitempty"`
	CandidateRunID       string `json:"candidate_run_id,omitempty"`
	DreamCandidateRunID  string `json:"dream_candidate_run_id,omitempty"`
	CandidateDraftID     string `json:"candidate_draft_id,omitempty"`
	CandidateExecutionID string `json:"candidate_execution_id,omitempty"`
	GeneratorAdapterID   string `json:"generator_adapter_id,omitempty"`
	HandoffID            string `json:"handoff_id,omitempty"`
	AdmissionAdapterID   string `json:"admission_adapter_id,omitempty"`
	CandidateTextStatus  string `json:"candidate_text_status,omitempty"`
	CandidateTextHash    string `json:"candidate_text_hash,omitempty"`
	Passed               bool   `json:"passed"`
	Reason               string `json:"reason,omitempty"`
	TurnTextHash         string `json:"turn_text_hash,omitempty"`
}

type admissionLiveRouteTurnCandidateAdmissionDecision struct {
	Schema                string `json:"schema"`
	Timing                string `json:"timing"`
	Decision              string `json:"decision,omitempty"`
	PromptClass           string `json:"prompt_class"`
	Route                 string `json:"route,omitempty"`
	Source                string `json:"source,omitempty"`
	ExpectedSource        string `json:"expected_source,omitempty"`
	CandidateRunID        string `json:"candidate_run_id,omitempty"`
	CandidateDraftID      string `json:"candidate_draft_id,omitempty"`
	CandidateExecutionID  string `json:"candidate_execution_id,omitempty"`
	GeneratorAdapterID    string `json:"generator_adapter_id,omitempty"`
	HandoffID             string `json:"handoff_id,omitempty"`
	AdmissionAdapterID    string `json:"admission_adapter_id,omitempty"`
	DreamCandidateRunID   string `json:"dream_candidate_run_id,omitempty"`
	DreamCandidateSchema  string `json:"dream_candidate_schema,omitempty"`
	DreamCandidateMode    string `json:"dream_candidate_mode,omitempty"`
	DreamAccepted         bool   `json:"dream_accepted"`
	DreamReason           string `json:"dream_reason,omitempty"`
	CandidateTextStatus   string `json:"candidate_text_status,omitempty"`
	CandidateTextHash     string `json:"candidate_text_hash,omitempty"`
	AdmissionPolicyPassed bool   `json:"admission_policy_passed"`
	LiveRouteChoicePassed bool   `json:"live_route_choice_passed"`
	LiveReady             bool   `json:"live_ready"`
	MutatesState          bool   `json:"mutates_state"`
	DecisionID            string `json:"decision_id,omitempty"`
	Passed                bool   `json:"passed"`
	Reason                string `json:"reason,omitempty"`
	TurnTextHash          string `json:"turn_text_hash,omitempty"`
}

type admissionLiveRouteTurnCandidateAdmissionPromotion struct {
	Schema                string `json:"schema"`
	Timing                string `json:"timing"`
	Promotion             string `json:"promotion,omitempty"`
	PromptClass           string `json:"prompt_class"`
	Route                 string `json:"route,omitempty"`
	Source                string `json:"source,omitempty"`
	ExpectedSource        string `json:"expected_source,omitempty"`
	CandidateRunID        string `json:"candidate_run_id,omitempty"`
	CandidateDraftID      string `json:"candidate_draft_id,omitempty"`
	CandidateExecutionID  string `json:"candidate_execution_id,omitempty"`
	GeneratorAdapterID    string `json:"generator_adapter_id,omitempty"`
	HandoffID             string `json:"handoff_id,omitempty"`
	AdmissionAdapterID    string `json:"admission_adapter_id,omitempty"`
	AdmissionDecisionID   string `json:"admission_decision_id,omitempty"`
	AdmissionDecision     string `json:"admission_decision,omitempty"`
	DreamCandidateRunID   string `json:"dream_candidate_run_id,omitempty"`
	CandidateTextStatus   string `json:"candidate_text_status,omitempty"`
	CandidateTextHash     string `json:"candidate_text_hash,omitempty"`
	AdmissionPolicyPassed bool   `json:"admission_policy_passed"`
	LiveRouteChoicePassed bool   `json:"live_route_choice_passed"`
	SourceDecisionPassed  bool   `json:"source_decision_passed"`
	LiveReady             bool   `json:"live_ready"`
	LiveAdmissionEnabled  bool   `json:"live_admission_enabled"`
	MutatesState          bool   `json:"mutates_state"`
	PromotionID           string `json:"promotion_id,omitempty"`
	Passed                bool   `json:"passed"`
	Reason                string `json:"reason,omitempty"`
	TurnTextHash          string `json:"turn_text_hash,omitempty"`
}

type admissionLiveRouteTurnCandidateAdmissionSwitch struct {
	Schema                string `json:"schema"`
	Timing                string `json:"timing"`
	SwitchState           string `json:"switch_state,omitempty"`
	SwitchAction          string `json:"switch_action,omitempty"`
	PromptClass           string `json:"prompt_class"`
	Route                 string `json:"route,omitempty"`
	Source                string `json:"source,omitempty"`
	ExpectedSource        string `json:"expected_source,omitempty"`
	CandidateRunID        string `json:"candidate_run_id,omitempty"`
	CandidateDraftID      string `json:"candidate_draft_id,omitempty"`
	CandidateExecutionID  string `json:"candidate_execution_id,omitempty"`
	GeneratorAdapterID    string `json:"generator_adapter_id,omitempty"`
	HandoffID             string `json:"handoff_id,omitempty"`
	AdmissionAdapterID    string `json:"admission_adapter_id,omitempty"`
	AdmissionDecisionID   string `json:"admission_decision_id,omitempty"`
	AdmissionPromotionID  string `json:"admission_promotion_id,omitempty"`
	AdmissionDecision     string `json:"admission_decision,omitempty"`
	AdmissionPromotion    string `json:"admission_promotion,omitempty"`
	DreamCandidateRunID   string `json:"dream_candidate_run_id,omitempty"`
	CandidateTextStatus   string `json:"candidate_text_status,omitempty"`
	CandidateTextHash     string `json:"candidate_text_hash,omitempty"`
	AdmissionPolicyPassed bool   `json:"admission_policy_passed"`
	LiveRouteChoicePassed bool   `json:"live_route_choice_passed"`
	SourceDecisionPassed  bool   `json:"source_decision_passed"`
	SourcePromotionPassed bool   `json:"source_promotion_passed"`
	LiveReady             bool   `json:"live_ready"`
	LiveAdmissionEnabled  bool   `json:"live_admission_enabled"`
	AdmissionAllowed      bool   `json:"admission_allowed"`
	MutatesState          bool   `json:"mutates_state"`
	SwitchID              string `json:"switch_id,omitempty"`
	Passed                bool   `json:"passed"`
	Reason                string `json:"reason,omitempty"`
	TurnTextHash          string `json:"turn_text_hash,omitempty"`
}

type admissionLiveRouteTurnCandidateAdmissionEnableGate struct {
	Schema                string `json:"schema"`
	Timing                string `json:"timing"`
	EnableState           string `json:"enable_state,omitempty"`
	EnableAction          string `json:"enable_action,omitempty"`
	PromptClass           string `json:"prompt_class"`
	Route                 string `json:"route,omitempty"`
	Source                string `json:"source,omitempty"`
	ExpectedSource        string `json:"expected_source,omitempty"`
	CandidateRunID        string `json:"candidate_run_id,omitempty"`
	CandidateDraftID      string `json:"candidate_draft_id,omitempty"`
	CandidateExecutionID  string `json:"candidate_execution_id,omitempty"`
	GeneratorAdapterID    string `json:"generator_adapter_id,omitempty"`
	HandoffID             string `json:"handoff_id,omitempty"`
	AdmissionAdapterID    string `json:"admission_adapter_id,omitempty"`
	AdmissionDecisionID   string `json:"admission_decision_id,omitempty"`
	AdmissionPromotionID  string `json:"admission_promotion_id,omitempty"`
	AdmissionSwitchID     string `json:"admission_switch_id,omitempty"`
	AdmissionDecision     string `json:"admission_decision,omitempty"`
	AdmissionPromotion    string `json:"admission_promotion,omitempty"`
	SwitchState           string `json:"switch_state,omitempty"`
	SwitchAction          string `json:"switch_action,omitempty"`
	DreamCandidateRunID   string `json:"dream_candidate_run_id,omitempty"`
	CandidateTextStatus   string `json:"candidate_text_status,omitempty"`
	CandidateTextHash     string `json:"candidate_text_hash,omitempty"`
	AdmissionPolicyPassed bool   `json:"admission_policy_passed"`
	LiveRouteChoicePassed bool   `json:"live_route_choice_passed"`
	SourceDecisionPassed  bool   `json:"source_decision_passed"`
	SourcePromotionPassed bool   `json:"source_promotion_passed"`
	SourceSwitchPassed    bool   `json:"source_switch_passed"`
	LiveReady             bool   `json:"live_ready"`
	LiveAdmissionEnabled  bool   `json:"live_admission_enabled"`
	AdmissionAllowed      bool   `json:"admission_allowed"`
	ManualEnableRequested bool   `json:"manual_enable_requested"`
	EnableKeyMatched      bool   `json:"enable_key_matched"`
	MutatesState          bool   `json:"mutates_state"`
	EnableGateID          string `json:"enable_gate_id,omitempty"`
	Passed                bool   `json:"passed"`
	Reason                string `json:"reason,omitempty"`
	TurnTextHash          string `json:"turn_text_hash,omitempty"`
}

type admissionLiveRouteTurnCandidateAdmissionLiveStage struct {
	Schema                string `json:"schema"`
	Timing                string `json:"timing"`
	StageState            string `json:"stage_state,omitempty"`
	StageAction           string `json:"stage_action,omitempty"`
	PromptClass           string `json:"prompt_class"`
	Route                 string `json:"route,omitempty"`
	Source                string `json:"source,omitempty"`
	ExpectedSource        string `json:"expected_source,omitempty"`
	CandidateRunID        string `json:"candidate_run_id,omitempty"`
	CandidateDraftID      string `json:"candidate_draft_id,omitempty"`
	CandidateExecutionID  string `json:"candidate_execution_id,omitempty"`
	GeneratorAdapterID    string `json:"generator_adapter_id,omitempty"`
	HandoffID             string `json:"handoff_id,omitempty"`
	AdmissionAdapterID    string `json:"admission_adapter_id,omitempty"`
	AdmissionDecisionID   string `json:"admission_decision_id,omitempty"`
	AdmissionPromotionID  string `json:"admission_promotion_id,omitempty"`
	AdmissionSwitchID     string `json:"admission_switch_id,omitempty"`
	AdmissionEnableGateID string `json:"admission_enable_gate_id,omitempty"`
	AdmissionDecision     string `json:"admission_decision,omitempty"`
	AdmissionPromotion    string `json:"admission_promotion,omitempty"`
	SwitchState           string `json:"switch_state,omitempty"`
	SwitchAction          string `json:"switch_action,omitempty"`
	EnableState           string `json:"enable_state,omitempty"`
	EnableAction          string `json:"enable_action,omitempty"`
	DreamCandidateRunID   string `json:"dream_candidate_run_id,omitempty"`
	CandidateTextStatus   string `json:"candidate_text_status,omitempty"`
	CandidateTextHash     string `json:"candidate_text_hash,omitempty"`
	AdmissionPolicyPassed bool   `json:"admission_policy_passed"`
	LiveRouteChoicePassed bool   `json:"live_route_choice_passed"`
	SourceDecisionPassed  bool   `json:"source_decision_passed"`
	SourcePromotionPassed bool   `json:"source_promotion_passed"`
	SourceSwitchPassed    bool   `json:"source_switch_passed"`
	SourceEnablePassed    bool   `json:"source_enable_passed"`
	LiveReady             bool   `json:"live_ready"`
	LiveAdmissionEnabled  bool   `json:"live_admission_enabled"`
	AdmissionAllowed      bool   `json:"admission_allowed"`
	ManualEnableRequested bool   `json:"manual_enable_requested"`
	EnableKeyMatched      bool   `json:"enable_key_matched"`
	RequiresWriter        bool   `json:"requires_writer"`
	WriterReady           bool   `json:"writer_ready"`
	RequiresRollback      bool   `json:"requires_rollback"`
	RollbackReady         bool   `json:"rollback_ready"`
	MutatesState          bool   `json:"mutates_state"`
	LiveStageID           string `json:"live_stage_id,omitempty"`
	Passed                bool   `json:"passed"`
	Reason                string `json:"reason,omitempty"`
	TurnTextHash          string `json:"turn_text_hash,omitempty"`
}

type admissionLiveRouteTurnCandidateAdmissionWriterPreflight struct {
	Schema                string `json:"schema"`
	Timing                string `json:"timing"`
	WriterState           string `json:"writer_state,omitempty"`
	WriterAction          string `json:"writer_action,omitempty"`
	RollbackState         string `json:"rollback_state,omitempty"`
	RollbackAction        string `json:"rollback_action,omitempty"`
	PromptClass           string `json:"prompt_class"`
	Route                 string `json:"route,omitempty"`
	Source                string `json:"source,omitempty"`
	ExpectedSource        string `json:"expected_source,omitempty"`
	CandidateRunID        string `json:"candidate_run_id,omitempty"`
	CandidateDraftID      string `json:"candidate_draft_id,omitempty"`
	CandidateExecutionID  string `json:"candidate_execution_id,omitempty"`
	GeneratorAdapterID    string `json:"generator_adapter_id,omitempty"`
	HandoffID             string `json:"handoff_id,omitempty"`
	AdmissionAdapterID    string `json:"admission_adapter_id,omitempty"`
	AdmissionDecisionID   string `json:"admission_decision_id,omitempty"`
	AdmissionPromotionID  string `json:"admission_promotion_id,omitempty"`
	AdmissionSwitchID     string `json:"admission_switch_id,omitempty"`
	AdmissionEnableGateID string `json:"admission_enable_gate_id,omitempty"`
	AdmissionLiveStageID  string `json:"admission_live_stage_id,omitempty"`
	AdmissionDecision     string `json:"admission_decision,omitempty"`
	AdmissionPromotion    string `json:"admission_promotion,omitempty"`
	SwitchState           string `json:"switch_state,omitempty"`
	SwitchAction          string `json:"switch_action,omitempty"`
	EnableState           string `json:"enable_state,omitempty"`
	EnableAction          string `json:"enable_action,omitempty"`
	StageState            string `json:"stage_state,omitempty"`
	StageAction           string `json:"stage_action,omitempty"`
	DreamCandidateRunID   string `json:"dream_candidate_run_id,omitempty"`
	CandidateTextStatus   string `json:"candidate_text_status,omitempty"`
	CandidateTextHash     string `json:"candidate_text_hash,omitempty"`
	AdmissionPolicyPassed bool   `json:"admission_policy_passed"`
	LiveRouteChoicePassed bool   `json:"live_route_choice_passed"`
	SourceDecisionPassed  bool   `json:"source_decision_passed"`
	SourcePromotionPassed bool   `json:"source_promotion_passed"`
	SourceSwitchPassed    bool   `json:"source_switch_passed"`
	SourceEnablePassed    bool   `json:"source_enable_passed"`
	SourceStagePassed     bool   `json:"source_stage_passed"`
	LiveReady             bool   `json:"live_ready"`
	LiveAdmissionEnabled  bool   `json:"live_admission_enabled"`
	AdmissionAllowed      bool   `json:"admission_allowed"`
	ManualEnableRequested bool   `json:"manual_enable_requested"`
	EnableKeyMatched      bool   `json:"enable_key_matched"`
	RequiresWriter        bool   `json:"requires_writer"`
	WriterReady           bool   `json:"writer_ready"`
	RequiresRollback      bool   `json:"requires_rollback"`
	RollbackReady         bool   `json:"rollback_ready"`
	WriteAllowed          bool   `json:"write_allowed"`
	MutatesState          bool   `json:"mutates_state"`
	WriterPreflightID     string `json:"writer_preflight_id,omitempty"`
	Passed                bool   `json:"passed"`
	Reason                string `json:"reason,omitempty"`
	TurnTextHash          string `json:"turn_text_hash,omitempty"`
}

type admissionLiveRouteTurnCandidateAdmissionWriterInventory struct {
	Schema                      string `json:"schema"`
	Timing                      string `json:"timing"`
	InventoryState              string `json:"inventory_state,omitempty"`
	InventoryAction             string `json:"inventory_action,omitempty"`
	WriterContract              string `json:"writer_contract,omitempty"`
	RollbackContract            string `json:"rollback_contract,omitempty"`
	AdmissionLedgerContract     string `json:"admission_ledger_contract,omitempty"`
	WriterContractPresent       bool   `json:"writer_contract_present"`
	RollbackContractPresent     bool   `json:"rollback_contract_present"`
	LedgerContractPresent       bool   `json:"ledger_contract_present"`
	ContractsReady              bool   `json:"contracts_ready"`
	WriterState                 string `json:"writer_state,omitempty"`
	WriterAction                string `json:"writer_action,omitempty"`
	RollbackState               string `json:"rollback_state,omitempty"`
	RollbackAction              string `json:"rollback_action,omitempty"`
	PromptClass                 string `json:"prompt_class"`
	Route                       string `json:"route,omitempty"`
	Source                      string `json:"source,omitempty"`
	ExpectedSource              string `json:"expected_source,omitempty"`
	CandidateRunID              string `json:"candidate_run_id,omitempty"`
	CandidateDraftID            string `json:"candidate_draft_id,omitempty"`
	CandidateExecutionID        string `json:"candidate_execution_id,omitempty"`
	GeneratorAdapterID          string `json:"generator_adapter_id,omitempty"`
	HandoffID                   string `json:"handoff_id,omitempty"`
	AdmissionAdapterID          string `json:"admission_adapter_id,omitempty"`
	AdmissionDecisionID         string `json:"admission_decision_id,omitempty"`
	AdmissionPromotionID        string `json:"admission_promotion_id,omitempty"`
	AdmissionSwitchID           string `json:"admission_switch_id,omitempty"`
	AdmissionEnableGateID       string `json:"admission_enable_gate_id,omitempty"`
	AdmissionLiveStageID        string `json:"admission_live_stage_id,omitempty"`
	AdmissionWriterPreflightID  string `json:"admission_writer_preflight_id,omitempty"`
	AdmissionDecision           string `json:"admission_decision,omitempty"`
	AdmissionPromotion          string `json:"admission_promotion,omitempty"`
	SwitchState                 string `json:"switch_state,omitempty"`
	SwitchAction                string `json:"switch_action,omitempty"`
	EnableState                 string `json:"enable_state,omitempty"`
	EnableAction                string `json:"enable_action,omitempty"`
	StageState                  string `json:"stage_state,omitempty"`
	StageAction                 string `json:"stage_action,omitempty"`
	DreamCandidateRunID         string `json:"dream_candidate_run_id,omitempty"`
	CandidateTextStatus         string `json:"candidate_text_status,omitempty"`
	CandidateTextHash           string `json:"candidate_text_hash,omitempty"`
	AdmissionPolicyPassed       bool   `json:"admission_policy_passed"`
	LiveRouteChoicePassed       bool   `json:"live_route_choice_passed"`
	SourceDecisionPassed        bool   `json:"source_decision_passed"`
	SourcePromotionPassed       bool   `json:"source_promotion_passed"`
	SourceSwitchPassed          bool   `json:"source_switch_passed"`
	SourceEnablePassed          bool   `json:"source_enable_passed"`
	SourceStagePassed           bool   `json:"source_stage_passed"`
	SourceWriterPreflightPassed bool   `json:"source_writer_preflight_passed"`
	LiveReady                   bool   `json:"live_ready"`
	LiveAdmissionEnabled        bool   `json:"live_admission_enabled"`
	AdmissionAllowed            bool   `json:"admission_allowed"`
	ManualEnableRequested       bool   `json:"manual_enable_requested"`
	EnableKeyMatched            bool   `json:"enable_key_matched"`
	RequiresWriter              bool   `json:"requires_writer"`
	WriterReady                 bool   `json:"writer_ready"`
	RequiresRollback            bool   `json:"requires_rollback"`
	RollbackReady               bool   `json:"rollback_ready"`
	WriteAllowed                bool   `json:"write_allowed"`
	MutatesState                bool   `json:"mutates_state"`
	WriterInventoryID           string `json:"writer_inventory_id,omitempty"`
	Passed                      bool   `json:"passed"`
	Reason                      string `json:"reason,omitempty"`
	TurnTextHash                string `json:"turn_text_hash,omitempty"`
}

type admissionLiveRouteTurnCandidateAdmissionWriterContract struct {
	Schema                        string `json:"schema"`
	Timing                        string `json:"timing"`
	ContractState                 string `json:"contract_state,omitempty"`
	ContractAction                string `json:"contract_action,omitempty"`
	WriterContract                string `json:"writer_contract,omitempty"`
	RollbackContract              string `json:"rollback_contract,omitempty"`
	AdmissionLedgerContract       string `json:"admission_ledger_contract,omitempty"`
	WriterContractShape           string `json:"writer_contract_shape,omitempty"`
	RollbackContractShape         string `json:"rollback_contract_shape,omitempty"`
	LedgerContractShape           string `json:"ledger_contract_shape,omitempty"`
	WriteScope                    string `json:"write_scope,omitempty"`
	RollbackScope                 string `json:"rollback_scope,omitempty"`
	LedgerMode                    string `json:"ledger_mode,omitempty"`
	ContractShapeReady            bool   `json:"contract_shape_ready"`
	SourceWriterContractPresent   bool   `json:"source_writer_contract_present"`
	SourceRollbackContractPresent bool   `json:"source_rollback_contract_present"`
	SourceLedgerContractPresent   bool   `json:"source_ledger_contract_present"`
	WriterImplementationReady     bool   `json:"writer_implementation_ready"`
	RollbackImplementationReady   bool   `json:"rollback_implementation_ready"`
	LedgerImplementationReady     bool   `json:"ledger_implementation_ready"`
	ContractsReady                bool   `json:"contracts_ready"`
	WriterState                   string `json:"writer_state,omitempty"`
	WriterAction                  string `json:"writer_action,omitempty"`
	RollbackState                 string `json:"rollback_state,omitempty"`
	RollbackAction                string `json:"rollback_action,omitempty"`
	PromptClass                   string `json:"prompt_class"`
	Route                         string `json:"route,omitempty"`
	Source                        string `json:"source,omitempty"`
	ExpectedSource                string `json:"expected_source,omitempty"`
	CandidateRunID                string `json:"candidate_run_id,omitempty"`
	CandidateDraftID              string `json:"candidate_draft_id,omitempty"`
	CandidateExecutionID          string `json:"candidate_execution_id,omitempty"`
	GeneratorAdapterID            string `json:"generator_adapter_id,omitempty"`
	HandoffID                     string `json:"handoff_id,omitempty"`
	AdmissionAdapterID            string `json:"admission_adapter_id,omitempty"`
	AdmissionDecisionID           string `json:"admission_decision_id,omitempty"`
	AdmissionPromotionID          string `json:"admission_promotion_id,omitempty"`
	AdmissionSwitchID             string `json:"admission_switch_id,omitempty"`
	AdmissionEnableGateID         string `json:"admission_enable_gate_id,omitempty"`
	AdmissionLiveStageID          string `json:"admission_live_stage_id,omitempty"`
	AdmissionWriterPreflightID    string `json:"admission_writer_preflight_id,omitempty"`
	AdmissionWriterInventoryID    string `json:"admission_writer_inventory_id,omitempty"`
	AdmissionDecision             string `json:"admission_decision,omitempty"`
	AdmissionPromotion            string `json:"admission_promotion,omitempty"`
	SwitchState                   string `json:"switch_state,omitempty"`
	SwitchAction                  string `json:"switch_action,omitempty"`
	EnableState                   string `json:"enable_state,omitempty"`
	EnableAction                  string `json:"enable_action,omitempty"`
	StageState                    string `json:"stage_state,omitempty"`
	StageAction                   string `json:"stage_action,omitempty"`
	InventoryState                string `json:"inventory_state,omitempty"`
	InventoryAction               string `json:"inventory_action,omitempty"`
	DreamCandidateRunID           string `json:"dream_candidate_run_id,omitempty"`
	CandidateTextStatus           string `json:"candidate_text_status,omitempty"`
	CandidateTextHash             string `json:"candidate_text_hash,omitempty"`
	AdmissionPolicyPassed         bool   `json:"admission_policy_passed"`
	LiveRouteChoicePassed         bool   `json:"live_route_choice_passed"`
	SourceDecisionPassed          bool   `json:"source_decision_passed"`
	SourcePromotionPassed         bool   `json:"source_promotion_passed"`
	SourceSwitchPassed            bool   `json:"source_switch_passed"`
	SourceEnablePassed            bool   `json:"source_enable_passed"`
	SourceStagePassed             bool   `json:"source_stage_passed"`
	SourceWriterPreflightPassed   bool   `json:"source_writer_preflight_passed"`
	SourceWriterInventoryPassed   bool   `json:"source_writer_inventory_passed"`
	LiveReady                     bool   `json:"live_ready"`
	LiveAdmissionEnabled          bool   `json:"live_admission_enabled"`
	AdmissionAllowed              bool   `json:"admission_allowed"`
	ManualEnableRequested         bool   `json:"manual_enable_requested"`
	EnableKeyMatched              bool   `json:"enable_key_matched"`
	RequiresWriter                bool   `json:"requires_writer"`
	WriterReady                   bool   `json:"writer_ready"`
	RequiresRollback              bool   `json:"requires_rollback"`
	RollbackReady                 bool   `json:"rollback_ready"`
	WriteAllowed                  bool   `json:"write_allowed"`
	MutatesState                  bool   `json:"mutates_state"`
	WriterContractID              string `json:"writer_contract_id,omitempty"`
	Passed                        bool   `json:"passed"`
	Reason                        string `json:"reason,omitempty"`
	TurnTextHash                  string `json:"turn_text_hash,omitempty"`
}

type admissionLiveRouteTurnCandidateAdmissionLedger struct {
	Schema                        string `json:"schema"`
	Timing                        string `json:"timing"`
	LedgerState                   string `json:"ledger_state,omitempty"`
	LedgerAction                  string `json:"ledger_action,omitempty"`
	LedgerContract                string `json:"ledger_contract,omitempty"`
	LedgerMode                    string `json:"ledger_mode,omitempty"`
	LedgerEntryKind               string `json:"ledger_entry_kind,omitempty"`
	LedgerEntryStatus             string `json:"ledger_entry_status,omitempty"`
	LedgerReceiptShape            string `json:"ledger_receipt_shape,omitempty"`
	LedgerAppendReady             bool   `json:"ledger_append_ready"`
	LedgerReceiptPersisted        bool   `json:"ledger_receipt_persisted"`
	LedgerImplementationReady     bool   `json:"ledger_implementation_ready"`
	ContractState                 string `json:"contract_state,omitempty"`
	ContractAction                string `json:"contract_action,omitempty"`
	WriterContract                string `json:"writer_contract,omitempty"`
	RollbackContract              string `json:"rollback_contract,omitempty"`
	AdmissionLedgerContract       string `json:"admission_ledger_contract,omitempty"`
	WriterContractShape           string `json:"writer_contract_shape,omitempty"`
	RollbackContractShape         string `json:"rollback_contract_shape,omitempty"`
	LedgerContractShape           string `json:"ledger_contract_shape,omitempty"`
	WriteScope                    string `json:"write_scope,omitempty"`
	RollbackScope                 string `json:"rollback_scope,omitempty"`
	ContractShapeReady            bool   `json:"contract_shape_ready"`
	SourceWriterContractPresent   bool   `json:"source_writer_contract_present"`
	SourceRollbackContractPresent bool   `json:"source_rollback_contract_present"`
	SourceLedgerContractPresent   bool   `json:"source_ledger_contract_present"`
	WriterImplementationReady     bool   `json:"writer_implementation_ready"`
	RollbackImplementationReady   bool   `json:"rollback_implementation_ready"`
	ContractsReady                bool   `json:"contracts_ready"`
	WriterState                   string `json:"writer_state,omitempty"`
	WriterAction                  string `json:"writer_action,omitempty"`
	RollbackState                 string `json:"rollback_state,omitempty"`
	RollbackAction                string `json:"rollback_action,omitempty"`
	PromptClass                   string `json:"prompt_class"`
	Route                         string `json:"route,omitempty"`
	Source                        string `json:"source,omitempty"`
	ExpectedSource                string `json:"expected_source,omitempty"`
	CandidateRunID                string `json:"candidate_run_id,omitempty"`
	CandidateDraftID              string `json:"candidate_draft_id,omitempty"`
	CandidateExecutionID          string `json:"candidate_execution_id,omitempty"`
	GeneratorAdapterID            string `json:"generator_adapter_id,omitempty"`
	HandoffID                     string `json:"handoff_id,omitempty"`
	AdmissionAdapterID            string `json:"admission_adapter_id,omitempty"`
	AdmissionDecisionID           string `json:"admission_decision_id,omitempty"`
	AdmissionPromotionID          string `json:"admission_promotion_id,omitempty"`
	AdmissionSwitchID             string `json:"admission_switch_id,omitempty"`
	AdmissionEnableGateID         string `json:"admission_enable_gate_id,omitempty"`
	AdmissionLiveStageID          string `json:"admission_live_stage_id,omitempty"`
	AdmissionWriterPreflightID    string `json:"admission_writer_preflight_id,omitempty"`
	AdmissionWriterInventoryID    string `json:"admission_writer_inventory_id,omitempty"`
	AdmissionWriterContractID     string `json:"admission_writer_contract_id,omitempty"`
	AdmissionDecision             string `json:"admission_decision,omitempty"`
	AdmissionPromotion            string `json:"admission_promotion,omitempty"`
	SwitchState                   string `json:"switch_state,omitempty"`
	SwitchAction                  string `json:"switch_action,omitempty"`
	EnableState                   string `json:"enable_state,omitempty"`
	EnableAction                  string `json:"enable_action,omitempty"`
	StageState                    string `json:"stage_state,omitempty"`
	StageAction                   string `json:"stage_action,omitempty"`
	InventoryState                string `json:"inventory_state,omitempty"`
	InventoryAction               string `json:"inventory_action,omitempty"`
	DreamCandidateRunID           string `json:"dream_candidate_run_id,omitempty"`
	CandidateTextStatus           string `json:"candidate_text_status,omitempty"`
	CandidateTextHash             string `json:"candidate_text_hash,omitempty"`
	AdmissionPolicyPassed         bool   `json:"admission_policy_passed"`
	LiveRouteChoicePassed         bool   `json:"live_route_choice_passed"`
	SourceDecisionPassed          bool   `json:"source_decision_passed"`
	SourcePromotionPassed         bool   `json:"source_promotion_passed"`
	SourceSwitchPassed            bool   `json:"source_switch_passed"`
	SourceEnablePassed            bool   `json:"source_enable_passed"`
	SourceStagePassed             bool   `json:"source_stage_passed"`
	SourceWriterPreflightPassed   bool   `json:"source_writer_preflight_passed"`
	SourceWriterInventoryPassed   bool   `json:"source_writer_inventory_passed"`
	SourceWriterContractPassed    bool   `json:"source_writer_contract_passed"`
	LiveReady                     bool   `json:"live_ready"`
	LiveAdmissionEnabled          bool   `json:"live_admission_enabled"`
	AdmissionAllowed              bool   `json:"admission_allowed"`
	ManualEnableRequested         bool   `json:"manual_enable_requested"`
	EnableKeyMatched              bool   `json:"enable_key_matched"`
	RequiresWriter                bool   `json:"requires_writer"`
	WriterReady                   bool   `json:"writer_ready"`
	RequiresRollback              bool   `json:"requires_rollback"`
	RollbackReady                 bool   `json:"rollback_ready"`
	WriteAllowed                  bool   `json:"write_allowed"`
	MutatesState                  bool   `json:"mutates_state"`
	AdmissionLedgerID             string `json:"admission_ledger_id,omitempty"`
	Passed                        bool   `json:"passed"`
	Reason                        string `json:"reason,omitempty"`
	TurnTextHash                  string `json:"turn_text_hash,omitempty"`
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

func admissionLiveRouteTurnChoiceDryRun() bool {
	return dreamAdmissionBoolEnv("AM_LIVE_ROUTE_TURN_CHOICE_DRY_RUN")
}

func admissionLiveRouteTurnChoiceForObservation(obs admissionLiveRouteTurnObservation) admissionLiveRouteTurnChoice {
	choice := admissionLiveRouteTurnChoice{
		Schema:       admissionLiveRouteTurnChoiceSchema,
		PromptClass:  obs.PromptClass,
		Route:        obs.Route,
		TurnTextHash: obs.TextHash,
		Plan:         obs.Plan,
	}
	if obs.Schema == "" {
		choice.Reason = "missing_turn_observation"
		return choice
	}
	if !obs.Passed {
		choice.Reason = "turn route failed"
		if obs.Reason != "" {
			choice.Reason += ": " + obs.Reason
		}
		return choice
	}
	choice.ExpectedSource = obs.ExpectedSource
	choice.Source = obs.ExpectedSource
	if choice.Source == "" {
		choice.Reason = "missing source for turn route " + obs.Route + " prompt class " + obs.PromptClass
		return choice
	}
	choice.CandidateTrigger = admissionRouteTrigger(obs.Route, obs.PromptClass)
	choice.Passed = true
	return choice
}

func recordAdmissionLiveRouteTurnChoice(choice admissionLiveRouteTurnChoice) error {
	path := strings.TrimSpace(os.Getenv("AM_LIVE_ROUTE_TURN_CHOICE_LOG"))
	if path == "" {
		return nil
	}
	f, err := os.OpenFile(path, os.O_CREATE|os.O_APPEND|os.O_WRONLY, 0600)
	if err != nil {
		return err
	}
	enc := json.NewEncoder(f)
	err = enc.Encode(choice)
	if closeErr := f.Close(); err == nil {
		err = closeErr
	}
	return err
}

func admissionLiveRouteTurnRequestDryRun() bool {
	return dreamAdmissionBoolEnv("AM_LIVE_ROUTE_TURN_REQUEST_DRY_RUN")
}

func admissionLiveRouteTurnRequestForChoice(choice admissionLiveRouteTurnChoice) admissionLiveRouteTurnRequest {
	request := admissionLiveRouteTurnRequest{
		Schema:           admissionLiveRouteTurnRequestSchema,
		PromptClass:      choice.PromptClass,
		Route:            choice.Route,
		Source:           choice.Source,
		ExpectedSource:   choice.ExpectedSource,
		CandidateTrigger: choice.CandidateTrigger,
		CandidateSeed:    admissionLiveRouteTurnRequestSeed(choice),
		TurnTextHash:     choice.TurnTextHash,
	}
	if choice.Schema == "" {
		request.Reason = "missing_turn_choice"
		return request
	}
	if !choice.Passed {
		request.Reason = "turn choice failed"
		if choice.Reason != "" {
			request.Reason += ": " + choice.Reason
		}
		return request
	}
	if request.Source == "" {
		request.Reason = "missing source for turn route " + request.Route + " prompt class " + request.PromptClass
		return request
	}
	if request.CandidateTrigger == "" {
		request.Reason = "missing candidate trigger for turn route " + request.Route + " prompt class " + request.PromptClass
		return request
	}
	if request.CandidateSeed == "" {
		request.Reason = "missing candidate seed for turn route " + request.Route + " prompt class " + request.PromptClass
		return request
	}
	request.Passed = true
	return request
}

func admissionLiveRouteTurnRequestSeed(choice admissionLiveRouteTurnChoice) string {
	if choice.TurnTextHash == "" {
		return ""
	}
	return "turn-" + choice.TurnTextHash
}

func recordAdmissionLiveRouteTurnRequest(request admissionLiveRouteTurnRequest) error {
	path := strings.TrimSpace(os.Getenv("AM_LIVE_ROUTE_TURN_REQUEST_LOG"))
	if path == "" {
		return nil
	}
	f, err := os.OpenFile(path, os.O_CREATE|os.O_APPEND|os.O_WRONLY, 0600)
	if err != nil {
		return err
	}
	enc := json.NewEncoder(f)
	err = enc.Encode(request)
	if closeErr := f.Close(); err == nil {
		err = closeErr
	}
	return err
}

func admissionLiveRouteTurnGenerationJobDryRun() bool {
	return dreamAdmissionBoolEnv("AM_LIVE_ROUTE_TURN_GENERATION_JOB_DRY_RUN")
}

type admissionLiveRouteGenerationRoute struct {
	Backend     string
	Entrypoint  string
	PromptFrame string
}

func admissionLiveRouteGenerationRouteFor(route string) (admissionLiveRouteGenerationRoute, bool) {
	switch strings.TrimSpace(route) {
	case "direct":
		return admissionLiveRouteGenerationRoute{Backend: "nano-arianna", Entrypoint: "direct", PromptFrame: "q_a"}, true
	case "chorus":
		return admissionLiveRouteGenerationRoute{Backend: "chorus-arianna", Entrypoint: "field", PromptFrame: "q_a"}, true
	case "qloop":
		return admissionLiveRouteGenerationRoute{Backend: "chorus-arianna", Entrypoint: "qloop", PromptFrame: "q_a"}, true
	case "qloop_hint_qa":
		return admissionLiveRouteGenerationRoute{Backend: "chorus-arianna", Entrypoint: "qloop_hint_qa", PromptFrame: "q_a_hint"}, true
	case "qloop_target":
		return admissionLiveRouteGenerationRoute{Backend: "chorus-arianna", Entrypoint: "qloop_target", PromptFrame: "user_arianna_target"}, true
	case "user_bridge":
		return admissionLiveRouteGenerationRoute{Backend: "chorus-arianna", Entrypoint: "repl_user_bridge", PromptFrame: "user_arianna"}, true
	default:
		return admissionLiveRouteGenerationRoute{}, false
	}
}

func admissionLiveRouteTurnGenerationJobForRequest(request admissionLiveRouteTurnRequest) admissionLiveRouteTurnGenerationJob {
	job := admissionLiveRouteTurnGenerationJob{
		Schema:           admissionLiveRouteTurnGenerationJobSchema,
		PromptClass:      request.PromptClass,
		Route:            request.Route,
		Source:           request.Source,
		ExpectedSource:   request.ExpectedSource,
		CandidateTrigger: request.CandidateTrigger,
		CandidateSeed:    request.CandidateSeed,
		TurnTextHash:     request.TurnTextHash,
	}
	route, ok := admissionLiveRouteGenerationRouteFor(request.Route)
	if ok {
		job.Backend = route.Backend
		job.Entrypoint = route.Entrypoint
		job.PromptFrame = route.PromptFrame
	}
	if request.Schema == "" {
		job.Reason = "missing_turn_request"
		return job
	}
	if !request.Passed {
		job.Reason = "turn request failed"
		if request.Reason != "" {
			job.Reason += ": " + request.Reason
		}
		return job
	}
	if !ok {
		job.Reason = "unknown generation route " + request.Route
		return job
	}
	expectedSource := admissionLiveRouteSource(request.Route)
	if job.ExpectedSource == "" {
		job.ExpectedSource = expectedSource
	}
	if job.Source == "" {
		job.Reason = "missing source for generation route " + job.Route + " prompt class " + job.PromptClass
		return job
	}
	if job.Source != expectedSource {
		job.Reason = "source " + job.Source + " does not match generation route " + expectedSource + " for prompt class " + job.PromptClass
		return job
	}
	if job.CandidateTrigger == "" {
		job.Reason = "missing candidate trigger for generation route " + job.Route + " prompt class " + job.PromptClass
		return job
	}
	if job.CandidateSeed == "" {
		job.Reason = "missing candidate seed for generation route " + job.Route + " prompt class " + job.PromptClass
		return job
	}
	job.JobID = admissionLiveRouteTurnGenerationJobID(job)
	if job.JobID == "" {
		job.Reason = "missing generation job id for route " + job.Route + " prompt class " + job.PromptClass
		return job
	}
	job.Passed = true
	return job
}

func admissionLiveRouteTurnGenerationJobID(job admissionLiveRouteTurnGenerationJob) string {
	h := hashJSON(struct {
		PromptClass      string `json:"prompt_class"`
		Route            string `json:"route"`
		Source           string `json:"source"`
		Backend          string `json:"backend"`
		Entrypoint       string `json:"entrypoint"`
		PromptFrame      string `json:"prompt_frame"`
		CandidateTrigger string `json:"candidate_trigger"`
		CandidateSeed    string `json:"candidate_seed"`
		TurnTextHash     string `json:"turn_text_hash"`
	}{
		PromptClass:      job.PromptClass,
		Route:            job.Route,
		Source:           job.Source,
		Backend:          job.Backend,
		Entrypoint:       job.Entrypoint,
		PromptFrame:      job.PromptFrame,
		CandidateTrigger: job.CandidateTrigger,
		CandidateSeed:    job.CandidateSeed,
		TurnTextHash:     job.TurnTextHash,
	})
	if h == "" {
		return ""
	}
	return "job-" + h
}

func recordAdmissionLiveRouteTurnGenerationJob(job admissionLiveRouteTurnGenerationJob) error {
	path := strings.TrimSpace(os.Getenv("AM_LIVE_ROUTE_TURN_GENERATION_JOB_LOG"))
	if path == "" {
		return nil
	}
	f, err := os.OpenFile(path, os.O_CREATE|os.O_APPEND|os.O_WRONLY, 0600)
	if err != nil {
		return err
	}
	enc := json.NewEncoder(f)
	err = enc.Encode(job)
	if closeErr := f.Close(); err == nil {
		err = closeErr
	}
	return err
}

func admissionLiveRouteTurnCandidateShellDryRun() bool {
	return dreamAdmissionBoolEnv("AM_LIVE_ROUTE_TURN_CANDIDATE_SHELL_DRY_RUN")
}

func admissionLiveRouteTurnCandidateShellForJob(job admissionLiveRouteTurnGenerationJob) admissionLiveRouteTurnCandidateShell {
	shell := admissionLiveRouteTurnCandidateShell{
		Schema:           admissionLiveRouteTurnCandidateShellSchema,
		PromptClass:      job.PromptClass,
		Route:            job.Route,
		Source:           job.Source,
		ExpectedSource:   job.ExpectedSource,
		Backend:          job.Backend,
		Entrypoint:       job.Entrypoint,
		PromptFrame:      job.PromptFrame,
		CandidateTrigger: job.CandidateTrigger,
		CandidateSeed:    job.CandidateSeed,
		JobID:            job.JobID,
		TurnTextHash:     job.TurnTextHash,
	}
	if job.Schema == "" {
		shell.Reason = "missing_generation_job"
		return shell
	}
	if !job.Passed {
		shell.Reason = "generation job failed"
		if job.Reason != "" {
			shell.Reason += ": " + job.Reason
		}
		return shell
	}
	if shell.JobID == "" {
		shell.Reason = "missing generation job id for route " + shell.Route + " prompt class " + shell.PromptClass
		return shell
	}
	if shell.Source == "" {
		shell.Reason = "missing candidate source for route " + shell.Route + " prompt class " + shell.PromptClass
		return shell
	}
	expectedSource := admissionLiveRouteSource(shell.Route)
	if shell.ExpectedSource == "" {
		shell.ExpectedSource = expectedSource
	}
	if shell.Source != expectedSource {
		shell.Reason = "source " + shell.Source + " does not match candidate route " + expectedSource + " for prompt class " + shell.PromptClass
		return shell
	}
	if shell.CandidateTrigger == "" {
		shell.Reason = "missing candidate trigger for route " + shell.Route + " prompt class " + shell.PromptClass
		return shell
	}
	if shell.CandidateSeed == "" {
		shell.Reason = "missing candidate seed for route " + shell.Route + " prompt class " + shell.PromptClass
		return shell
	}
	shell.CandidateSchema = "arianna.dream_candidate.v1"
	shell.CandidateKind = shell.Source
	shell.CandidateTextStatus = "pending_generation"
	shell.ShellID = admissionLiveRouteTurnCandidateShellID(shell)
	if shell.ShellID == "" {
		shell.Reason = "missing candidate shell id for route " + shell.Route + " prompt class " + shell.PromptClass
		return shell
	}
	shell.Passed = true
	return shell
}

func admissionLiveRouteTurnCandidateShellID(shell admissionLiveRouteTurnCandidateShell) string {
	h := hashJSON(struct {
		PromptClass      string `json:"prompt_class"`
		Route            string `json:"route"`
		Source           string `json:"source"`
		Backend          string `json:"backend"`
		Entrypoint       string `json:"entrypoint"`
		PromptFrame      string `json:"prompt_frame"`
		CandidateTrigger string `json:"candidate_trigger"`
		CandidateSeed    string `json:"candidate_seed"`
		JobID            string `json:"job_id"`
		TurnTextHash     string `json:"turn_text_hash"`
	}{
		PromptClass:      shell.PromptClass,
		Route:            shell.Route,
		Source:           shell.Source,
		Backend:          shell.Backend,
		Entrypoint:       shell.Entrypoint,
		PromptFrame:      shell.PromptFrame,
		CandidateTrigger: shell.CandidateTrigger,
		CandidateSeed:    shell.CandidateSeed,
		JobID:            shell.JobID,
		TurnTextHash:     shell.TurnTextHash,
	})
	if h == "" {
		return ""
	}
	return "shell-" + h
}

func recordAdmissionLiveRouteTurnCandidateShell(shell admissionLiveRouteTurnCandidateShell) error {
	path := strings.TrimSpace(os.Getenv("AM_LIVE_ROUTE_TURN_CANDIDATE_SHELL_LOG"))
	if path == "" {
		return nil
	}
	f, err := os.OpenFile(path, os.O_CREATE|os.O_APPEND|os.O_WRONLY, 0600)
	if err != nil {
		return err
	}
	enc := json.NewEncoder(f)
	err = enc.Encode(shell)
	if closeErr := f.Close(); err == nil {
		err = closeErr
	}
	return err
}

func admissionLiveRouteTurnCandidateExecutionDryRun() bool {
	return dreamAdmissionBoolEnv("AM_LIVE_ROUTE_TURN_CANDIDATE_EXECUTION_DRY_RUN")
}

func admissionLiveRouteTurnCandidateExecutionTimeoutMS() int {
	raw := strings.TrimSpace(os.Getenv("AM_LIVE_ROUTE_TURN_CANDIDATE_EXECUTION_TIMEOUT_MS"))
	if raw == "" {
		return admissionLiveRouteTurnCandidateExecutionDefaultTimeoutMS
	}
	timeout, err := strconv.Atoi(raw)
	if err != nil {
		return -1
	}
	return timeout
}

func admissionLiveRouteTurnCandidateExecutionExecutor(shell admissionLiveRouteTurnCandidateShell) string {
	backend := strings.TrimSpace(shell.Backend)
	entrypoint := strings.TrimSpace(shell.Entrypoint)
	frame := strings.TrimSpace(shell.PromptFrame)
	if backend == "" || entrypoint == "" || frame == "" {
		return ""
	}
	return backend + ":" + entrypoint + ":" + frame
}

func admissionLiveRouteTurnCandidateExecutionForShell(shell admissionLiveRouteTurnCandidateShell, text string) admissionLiveRouteTurnCandidateExecution {
	generated := strings.TrimSpace(text)
	runtime := admissionLiveRouteTurnCandidateExecutionRuntime{
		Runner: admissionLiveRouteTurnCandidateExecutionRunnerProvided,
		Status: admissionLiveRouteTurnCandidateExecutionStatusProvided,
	}
	if generated != "" {
		runtime.StdoutHash = hashJSON(generated)
	}
	return admissionLiveRouteTurnCandidateExecutionForShellWithRuntime(shell, text, runtime)
}

func admissionLiveRouteTurnCandidateExecutionPreflight(shell admissionLiveRouteTurnCandidateShell, runtime admissionLiveRouteTurnCandidateExecutionRuntime) (admissionLiveRouteTurnCandidateExecution, bool) {
	runner := strings.TrimSpace(runtime.Runner)
	status := strings.TrimSpace(runtime.Status)
	execution := admissionLiveRouteTurnCandidateExecution{
		Schema:              admissionLiveRouteTurnCandidateExecutionSchema,
		PromptClass:         shell.PromptClass,
		Route:               shell.Route,
		Source:              shell.Source,
		ExpectedSource:      shell.ExpectedSource,
		Backend:             shell.Backend,
		Entrypoint:          shell.Entrypoint,
		PromptFrame:         shell.PromptFrame,
		Executor:            admissionLiveRouteTurnCandidateExecutionExecutor(shell),
		TimeoutMS:           admissionLiveRouteTurnCandidateExecutionTimeoutMS(),
		Runner:              runner,
		RunnerStatus:        status,
		RunnerExitCode:      runtime.ExitCode,
		RunnerTimedOut:      runtime.TimedOut,
		RunnerDurationMS:    runtime.DurationMS,
		RunnerStdoutHash:    strings.TrimSpace(runtime.StdoutHash),
		RunnerStderrHash:    strings.TrimSpace(runtime.StderrHash),
		CandidateSchema:     shell.CandidateSchema,
		CandidateKind:       shell.CandidateKind,
		CandidateTrigger:    shell.CandidateTrigger,
		CandidateSeed:       shell.CandidateSeed,
		CandidateTextStatus: shell.CandidateTextStatus,
		GeneratedTextStatus: shell.CandidateTextStatus,
		JobID:               shell.JobID,
		ShellID:             shell.ShellID,
		TurnTextHash:        shell.TurnTextHash,
	}
	if shell.Schema == "" {
		execution.Reason = "missing_candidate_shell"
		return execution, false
	}
	if !shell.Passed {
		execution.Reason = "candidate shell failed"
		if shell.Reason != "" {
			execution.Reason += ": " + shell.Reason
		}
		return execution, false
	}
	if execution.ShellID == "" {
		execution.Reason = "missing candidate shell id for route " + execution.Route + " prompt class " + execution.PromptClass
		return execution, false
	}
	if wantShellID := admissionLiveRouteTurnCandidateShellID(shell); wantShellID == "" || execution.ShellID != wantShellID {
		execution.Reason = "candidate shell id mismatch"
		return execution, false
	}
	if execution.Executor == "" {
		execution.Reason = "missing candidate executor for shell " + execution.ShellID
		return execution, false
	}
	if execution.TimeoutMS <= 0 || execution.TimeoutMS > admissionLiveRouteTurnCandidateExecutionMaxTimeoutMS {
		execution.Reason = "candidate execution timeout out of bounds"
		return execution, false
	}
	if execution.CandidateSchema != "arianna.dream_candidate.v1" {
		execution.Reason = "unexpected candidate schema " + execution.CandidateSchema
		return execution, false
	}
	if execution.Source == "" {
		execution.Reason = "missing candidate source for shell " + execution.ShellID
		return execution, false
	}
	expectedSource := admissionLiveRouteSource(execution.Route)
	if execution.ExpectedSource == "" {
		execution.ExpectedSource = expectedSource
	}
	if execution.Source != expectedSource {
		execution.Reason = "source " + execution.Source + " does not match candidate execution route " + expectedSource + " for prompt class " + execution.PromptClass
		return execution, false
	}
	if execution.CandidateKind != execution.Source {
		execution.Reason = "candidate kind " + execution.CandidateKind + " does not match source " + execution.Source
		return execution, false
	}
	if execution.CandidateTextStatus != "pending_generation" {
		execution.Reason = "candidate shell text status is " + execution.CandidateTextStatus
		return execution, false
	}
	route, ok := admissionLiveRouteGenerationRouteFor(execution.Route)
	if !ok {
		execution.Reason = "unknown generation route " + execution.Route
		return execution, false
	}
	if execution.Backend != route.Backend || execution.Entrypoint != route.Entrypoint || execution.PromptFrame != route.PromptFrame {
		execution.Reason = "generation route mismatch for shell " + execution.ShellID
		return execution, false
	}
	if execution.CandidateTrigger == "" {
		execution.Reason = "missing candidate trigger for shell " + execution.ShellID
		return execution, false
	}
	if execution.CandidateSeed == "" {
		execution.Reason = "missing candidate seed for shell " + execution.ShellID
		return execution, false
	}
	if execution.JobID == "" {
		execution.Reason = "missing generation job id for shell " + execution.ShellID
		return execution, false
	}
	return execution, true
}

func admissionLiveRouteTurnCandidateExecutionForShellWithRuntime(shell admissionLiveRouteTurnCandidateShell, text string, runtime admissionLiveRouteTurnCandidateExecutionRuntime) admissionLiveRouteTurnCandidateExecution {
	execution, ok := admissionLiveRouteTurnCandidateExecutionPreflight(shell, runtime)
	if !ok {
		return execution
	}
	if execution.Runner == "" {
		execution.Reason = "missing candidate runner for shell " + execution.ShellID
		return execution
	}
	if execution.RunnerStatus == "" {
		execution.Reason = "missing candidate runner status for shell " + execution.ShellID
		return execution
	}
	if runtime.FailureReason != "" {
		execution.Reason = runtime.FailureReason
		return execution
	}
	if execution.RunnerTimedOut || execution.RunnerStatus == admissionLiveRouteTurnCandidateExecutionStatusTimedOut {
		execution.Reason = "candidate runner timed out for shell " + execution.ShellID
		return execution
	}
	if execution.RunnerStatus != admissionLiveRouteTurnCandidateExecutionStatusProvided &&
		execution.RunnerStatus != admissionLiveRouteTurnCandidateExecutionStatusSucceeded {
		execution.Reason = "candidate runner status " + execution.RunnerStatus + " for shell " + execution.ShellID
		return execution
	}
	generated := strings.TrimSpace(text)
	if generated == "" {
		execution.Reason = "missing generated text for shell " + execution.ShellID
		return execution
	}
	execution.GeneratedText = generated
	execution.GeneratedTextHash = hashJSON(generated)
	if execution.RunnerStdoutHash == "" {
		execution.RunnerStdoutHash = execution.GeneratedTextHash
	}
	if execution.RunnerStdoutHash != execution.GeneratedTextHash {
		execution.Reason = "candidate runner stdout hash mismatch for shell " + execution.ShellID
		return execution
	}
	execution.GeneratedTextStatus = "generated"
	execution.ExecutionID = admissionLiveRouteTurnCandidateExecutionID(execution)
	if execution.ExecutionID == "" {
		execution.Reason = "missing candidate execution id for shell " + execution.ShellID
		return execution
	}
	execution.Passed = true
	return execution
}

func admissionLiveRouteTurnCandidateExecutionID(execution admissionLiveRouteTurnCandidateExecution) string {
	h := hashJSON(struct {
		ShellID           string `json:"shell_id"`
		Backend           string `json:"backend"`
		Entrypoint        string `json:"entrypoint"`
		PromptFrame       string `json:"prompt_frame"`
		Executor          string `json:"executor"`
		TimeoutMS         int    `json:"timeout_ms"`
		Runner            string `json:"runner"`
		RunnerStatus      string `json:"runner_status"`
		RunnerStdoutHash  string `json:"runner_stdout_hash"`
		RunnerStderrHash  string `json:"runner_stderr_hash"`
		GeneratedTextHash string `json:"generated_text_hash"`
	}{
		ShellID:           execution.ShellID,
		Backend:           execution.Backend,
		Entrypoint:        execution.Entrypoint,
		PromptFrame:       execution.PromptFrame,
		Executor:          execution.Executor,
		TimeoutMS:         execution.TimeoutMS,
		Runner:            execution.Runner,
		RunnerStatus:      execution.RunnerStatus,
		RunnerStdoutHash:  execution.RunnerStdoutHash,
		RunnerStderrHash:  execution.RunnerStderrHash,
		GeneratedTextHash: execution.GeneratedTextHash,
	})
	if h == "" {
		return ""
	}
	return "execution-" + h
}

func recordAdmissionLiveRouteTurnCandidateExecution(execution admissionLiveRouteTurnCandidateExecution) error {
	path := strings.TrimSpace(os.Getenv("AM_LIVE_ROUTE_TURN_CANDIDATE_EXECUTION_LOG"))
	if path == "" {
		return nil
	}
	f, err := os.OpenFile(path, os.O_CREATE|os.O_APPEND|os.O_WRONLY, 0600)
	if err != nil {
		return err
	}
	enc := json.NewEncoder(f)
	err = enc.Encode(execution)
	if closeErr := f.Close(); err == nil {
		err = closeErr
	}
	return err
}

func admissionLiveRouteTurnGeneratorAdapterDryRun() bool {
	return dreamAdmissionBoolEnv("AM_LIVE_ROUTE_TURN_GENERATOR_ADAPTER_DRY_RUN")
}

func admissionLiveRouteTurnGeneratorAdapterForShell(shell admissionLiveRouteTurnCandidateShell, text string) admissionLiveRouteTurnGeneratorAdapter {
	adapter := admissionLiveRouteTurnGeneratorAdapter{
		Schema:              admissionLiveRouteTurnGeneratorAdapterSchema,
		PromptClass:         shell.PromptClass,
		Route:               shell.Route,
		Source:              shell.Source,
		ExpectedSource:      shell.ExpectedSource,
		Backend:             shell.Backend,
		Entrypoint:          shell.Entrypoint,
		PromptFrame:         shell.PromptFrame,
		CandidateSchema:     shell.CandidateSchema,
		CandidateKind:       shell.CandidateKind,
		CandidateTrigger:    shell.CandidateTrigger,
		CandidateSeed:       shell.CandidateSeed,
		CandidateTextStatus: shell.CandidateTextStatus,
		GeneratedTextStatus: shell.CandidateTextStatus,
		JobID:               shell.JobID,
		ShellID:             shell.ShellID,
		TurnTextHash:        shell.TurnTextHash,
	}
	if shell.Schema == "" {
		adapter.Reason = "missing_candidate_shell"
		return adapter
	}
	if !shell.Passed {
		adapter.Reason = "candidate shell failed"
		if shell.Reason != "" {
			adapter.Reason += ": " + shell.Reason
		}
		return adapter
	}
	if adapter.ShellID == "" {
		adapter.Reason = "missing candidate shell id for route " + adapter.Route + " prompt class " + adapter.PromptClass
		return adapter
	}
	if wantShellID := admissionLiveRouteTurnCandidateShellID(shell); wantShellID == "" || adapter.ShellID != wantShellID {
		adapter.Reason = "candidate shell id mismatch"
		return adapter
	}
	if adapter.CandidateSchema != "arianna.dream_candidate.v1" {
		adapter.Reason = "unexpected candidate schema " + adapter.CandidateSchema
		return adapter
	}
	if adapter.Source == "" {
		adapter.Reason = "missing candidate source for shell " + adapter.ShellID
		return adapter
	}
	expectedSource := admissionLiveRouteSource(adapter.Route)
	if adapter.ExpectedSource == "" {
		adapter.ExpectedSource = expectedSource
	}
	if adapter.Source != expectedSource {
		adapter.Reason = "source " + adapter.Source + " does not match generator route " + expectedSource + " for prompt class " + adapter.PromptClass
		return adapter
	}
	if adapter.CandidateKind != adapter.Source {
		adapter.Reason = "candidate kind " + adapter.CandidateKind + " does not match source " + adapter.Source
		return adapter
	}
	if adapter.CandidateTextStatus != "pending_generation" {
		adapter.Reason = "candidate shell text status is " + adapter.CandidateTextStatus
		return adapter
	}
	route, ok := admissionLiveRouteGenerationRouteFor(adapter.Route)
	if !ok {
		adapter.Reason = "unknown generation route " + adapter.Route
		return adapter
	}
	if adapter.Backend != route.Backend || adapter.Entrypoint != route.Entrypoint || adapter.PromptFrame != route.PromptFrame {
		adapter.Reason = "generation route mismatch for shell " + adapter.ShellID
		return adapter
	}
	if adapter.CandidateTrigger == "" {
		adapter.Reason = "missing candidate trigger for shell " + adapter.ShellID
		return adapter
	}
	if adapter.CandidateSeed == "" {
		adapter.Reason = "missing candidate seed for shell " + adapter.ShellID
		return adapter
	}
	if adapter.JobID == "" {
		adapter.Reason = "missing generation job id for shell " + adapter.ShellID
		return adapter
	}
	generated := strings.TrimSpace(text)
	if generated == "" {
		adapter.Reason = "missing generated text for shell " + adapter.ShellID
		return adapter
	}
	adapter.GeneratedText = generated
	adapter.GeneratedTextHash = hashJSON(generated)
	adapter.GeneratedTextStatus = "generated"
	adapter.AdapterID = admissionLiveRouteTurnGeneratorAdapterID(adapter)
	if adapter.AdapterID == "" {
		adapter.Reason = "missing generator adapter id for shell " + adapter.ShellID
		return adapter
	}
	adapter.Passed = true
	return adapter
}

func admissionLiveRouteTurnGeneratorAdapterForExecution(execution admissionLiveRouteTurnCandidateExecution) admissionLiveRouteTurnGeneratorAdapter {
	adapter := admissionLiveRouteTurnGeneratorAdapter{
		Schema:               admissionLiveRouteTurnGeneratorAdapterSchema,
		PromptClass:          execution.PromptClass,
		Route:                execution.Route,
		Source:               execution.Source,
		ExpectedSource:       execution.ExpectedSource,
		Backend:              execution.Backend,
		Entrypoint:           execution.Entrypoint,
		PromptFrame:          execution.PromptFrame,
		CandidateSchema:      execution.CandidateSchema,
		CandidateKind:        execution.CandidateKind,
		CandidateTrigger:     execution.CandidateTrigger,
		CandidateSeed:        execution.CandidateSeed,
		CandidateTextStatus:  execution.CandidateTextStatus,
		GeneratedText:        strings.TrimSpace(execution.GeneratedText),
		GeneratedTextHash:    execution.GeneratedTextHash,
		GeneratedTextStatus:  execution.GeneratedTextStatus,
		JobID:                execution.JobID,
		ShellID:              execution.ShellID,
		CandidateExecutionID: execution.ExecutionID,
		TurnTextHash:         execution.TurnTextHash,
	}
	if execution.Schema == "" {
		adapter.Reason = "missing_candidate_execution"
		return adapter
	}
	if !execution.Passed {
		adapter.Reason = "candidate execution failed"
		if execution.Reason != "" {
			adapter.Reason += ": " + execution.Reason
		}
		return adapter
	}
	if execution.ExecutionID == "" {
		adapter.Reason = "missing candidate execution id for shell " + execution.ShellID
		return adapter
	}
	if wantExecutionID := admissionLiveRouteTurnCandidateExecutionID(execution); wantExecutionID == "" || execution.ExecutionID != wantExecutionID {
		adapter.Reason = "candidate execution id mismatch"
		return adapter
	}
	if execution.GeneratedTextStatus != "generated" {
		adapter.Reason = "candidate execution text status is " + execution.GeneratedTextStatus
		return adapter
	}
	generated := strings.TrimSpace(execution.GeneratedText)
	if generated == "" {
		adapter.Reason = "missing generated text for execution " + execution.ExecutionID
		return adapter
	}
	if execution.GeneratedTextHash == "" || execution.GeneratedTextHash != hashJSON(generated) {
		adapter.Reason = "candidate execution text hash mismatch"
		return adapter
	}
	if execution.TimeoutMS <= 0 || execution.TimeoutMS > admissionLiveRouteTurnCandidateExecutionMaxTimeoutMS {
		adapter.Reason = "candidate execution timeout out of bounds"
		return adapter
	}
	if execution.Runner == "" {
		adapter.Reason = "candidate execution runner missing"
		return adapter
	}
	if execution.RunnerStatus != admissionLiveRouteTurnCandidateExecutionStatusProvided &&
		execution.RunnerStatus != admissionLiveRouteTurnCandidateExecutionStatusSucceeded {
		adapter.Reason = "candidate execution runner status " + execution.RunnerStatus
		return adapter
	}
	if execution.RunnerTimedOut {
		adapter.Reason = "candidate execution runner timed out"
		return adapter
	}
	if execution.RunnerStdoutHash == "" || execution.RunnerStdoutHash != execution.GeneratedTextHash {
		adapter.Reason = "candidate execution stdout hash mismatch"
		return adapter
	}
	if execution.Executor != admissionLiveRouteTurnCandidateExecutionExecutor(admissionLiveRouteTurnCandidateShell{
		Backend:     execution.Backend,
		Entrypoint:  execution.Entrypoint,
		PromptFrame: execution.PromptFrame,
	}) {
		adapter.Reason = "candidate execution executor mismatch"
		return adapter
	}
	shell := admissionLiveRouteTurnCandidateShell{
		Schema:              admissionLiveRouteTurnCandidateShellSchema,
		PromptClass:         execution.PromptClass,
		Route:               execution.Route,
		Source:              execution.Source,
		ExpectedSource:      execution.ExpectedSource,
		Backend:             execution.Backend,
		Entrypoint:          execution.Entrypoint,
		PromptFrame:         execution.PromptFrame,
		CandidateSchema:     execution.CandidateSchema,
		CandidateKind:       execution.CandidateKind,
		CandidateTrigger:    execution.CandidateTrigger,
		CandidateSeed:       execution.CandidateSeed,
		CandidateTextStatus: execution.CandidateTextStatus,
		JobID:               execution.JobID,
		ShellID:             execution.ShellID,
		Passed:              true,
		TurnTextHash:        execution.TurnTextHash,
	}
	if wantShellID := admissionLiveRouteTurnCandidateShellID(shell); wantShellID == "" || execution.ShellID != wantShellID {
		adapter.Reason = "candidate execution shell id mismatch"
		return adapter
	}
	route, ok := admissionLiveRouteGenerationRouteFor(execution.Route)
	if !ok {
		adapter.Reason = "unknown generation route " + execution.Route
		return adapter
	}
	if execution.Backend != route.Backend || execution.Entrypoint != route.Entrypoint || execution.PromptFrame != route.PromptFrame {
		adapter.Reason = "generation route mismatch for execution " + execution.ExecutionID
		return adapter
	}
	adapter.AdapterID = admissionLiveRouteTurnGeneratorAdapterID(adapter)
	if adapter.AdapterID == "" {
		adapter.Reason = "missing generator adapter id for execution " + execution.ExecutionID
		return adapter
	}
	adapter.Passed = true
	return adapter
}

func admissionLiveRouteTurnGeneratorAdapterID(adapter admissionLiveRouteTurnGeneratorAdapter) string {
	h := hashJSON(struct {
		ShellID           string `json:"shell_id"`
		Backend           string `json:"backend"`
		Entrypoint        string `json:"entrypoint"`
		PromptFrame       string `json:"prompt_frame"`
		GeneratedTextHash string `json:"generated_text_hash"`
	}{
		ShellID:           adapter.ShellID,
		Backend:           adapter.Backend,
		Entrypoint:        adapter.Entrypoint,
		PromptFrame:       adapter.PromptFrame,
		GeneratedTextHash: adapter.GeneratedTextHash,
	})
	if h == "" {
		return ""
	}
	return "adapter-" + h
}

func recordAdmissionLiveRouteTurnGeneratorAdapter(adapter admissionLiveRouteTurnGeneratorAdapter) error {
	path := strings.TrimSpace(os.Getenv("AM_LIVE_ROUTE_TURN_GENERATOR_ADAPTER_LOG"))
	if path == "" {
		return nil
	}
	f, err := os.OpenFile(path, os.O_CREATE|os.O_APPEND|os.O_WRONLY, 0600)
	if err != nil {
		return err
	}
	enc := json.NewEncoder(f)
	err = enc.Encode(adapter)
	if closeErr := f.Close(); err == nil {
		err = closeErr
	}
	return err
}

func admissionLiveRouteTurnCandidateDraftDryRun() bool {
	return dreamAdmissionBoolEnv("AM_LIVE_ROUTE_TURN_CANDIDATE_DRAFT_DRY_RUN")
}

func admissionLiveRouteTurnCandidateDraftForShell(shell admissionLiveRouteTurnCandidateShell, text string) admissionLiveRouteTurnCandidateDraft {
	draft := admissionLiveRouteTurnCandidateDraft{
		Schema:              admissionLiveRouteTurnCandidateDraftSchema,
		PromptClass:         shell.PromptClass,
		Route:               shell.Route,
		Source:              shell.Source,
		ExpectedSource:      shell.ExpectedSource,
		Backend:             shell.Backend,
		Entrypoint:          shell.Entrypoint,
		PromptFrame:         shell.PromptFrame,
		CandidateSchema:     shell.CandidateSchema,
		CandidateKind:       shell.CandidateKind,
		CandidateTrigger:    shell.CandidateTrigger,
		CandidateSeed:       shell.CandidateSeed,
		CandidateTextStatus: shell.CandidateTextStatus,
		JobID:               shell.JobID,
		ShellID:             shell.ShellID,
		TurnTextHash:        shell.TurnTextHash,
	}
	if shell.Schema == "" {
		draft.Reason = "missing_candidate_shell"
		return draft
	}
	if !shell.Passed {
		draft.Reason = "candidate shell failed"
		if shell.Reason != "" {
			draft.Reason += ": " + shell.Reason
		}
		return draft
	}
	if draft.ShellID == "" {
		draft.Reason = "missing candidate shell id for route " + draft.Route + " prompt class " + draft.PromptClass
		return draft
	}
	if wantShellID := admissionLiveRouteTurnCandidateShellID(shell); wantShellID == "" || draft.ShellID != wantShellID {
		draft.Reason = "candidate shell id mismatch"
		return draft
	}
	if draft.CandidateSchema != "arianna.dream_candidate.v1" {
		draft.Reason = "unexpected candidate schema " + draft.CandidateSchema
		return draft
	}
	if draft.Source == "" {
		draft.Reason = "missing candidate source for shell " + draft.ShellID
		return draft
	}
	expectedSource := admissionLiveRouteSource(draft.Route)
	if draft.ExpectedSource == "" {
		draft.ExpectedSource = expectedSource
	}
	if draft.Source != expectedSource {
		draft.Reason = "source " + draft.Source + " does not match draft route " + expectedSource + " for prompt class " + draft.PromptClass
		return draft
	}
	if draft.CandidateKind != draft.Source {
		draft.Reason = "candidate kind " + draft.CandidateKind + " does not match source " + draft.Source
		return draft
	}
	if draft.CandidateTextStatus != "pending_generation" {
		draft.Reason = "candidate shell text status is " + draft.CandidateTextStatus
		return draft
	}
	if draft.CandidateTrigger == "" {
		draft.Reason = "missing candidate trigger for shell " + draft.ShellID
		return draft
	}
	if draft.CandidateSeed == "" {
		draft.Reason = "missing candidate seed for shell " + draft.ShellID
		return draft
	}

	candidate := newDreamCandidate(draft.Source, draft.CandidateTrigger, draft.CandidateSeed, "", strings.TrimSpace(text), nil)
	if candidate.Text == "" {
		draft.Reason = "missing candidate text for shell " + draft.ShellID
		return draft
	}
	choice := admissionLiveRouteChoiceForCandidate(candidate)
	if !choice.Passed {
		draft.Reason = "candidate route failed: " + choice.Reason
		return draft
	}
	if choice.PromptClass != draft.PromptClass || choice.Route != draft.Route || choice.ExpectedSource != draft.ExpectedSource {
		draft.Reason = "candidate route drift: class " + choice.PromptClass + " route " + choice.Route + " expected " + choice.ExpectedSource
		return draft
	}

	draft.CandidateText = candidate.Text
	draft.CandidateTextHash = hashJSON(candidate.Text)
	draft.CandidateRunID = candidate.RunID
	draft.CandidateTextStatus = "generated"
	draft.DraftID = admissionLiveRouteTurnCandidateDraftID(draft)
	if draft.DraftID == "" {
		draft.Reason = "missing candidate draft id for shell " + draft.ShellID
		return draft
	}
	draft.Passed = true
	return draft
}

func admissionLiveRouteTurnCandidateDraftForAdapter(adapter admissionLiveRouteTurnGeneratorAdapter) admissionLiveRouteTurnCandidateDraft {
	draft := admissionLiveRouteTurnCandidateDraft{
		Schema:               admissionLiveRouteTurnCandidateDraftSchema,
		PromptClass:          adapter.PromptClass,
		Route:                adapter.Route,
		Source:               adapter.Source,
		ExpectedSource:       adapter.ExpectedSource,
		Backend:              adapter.Backend,
		Entrypoint:           adapter.Entrypoint,
		PromptFrame:          adapter.PromptFrame,
		CandidateSchema:      adapter.CandidateSchema,
		CandidateKind:        adapter.CandidateKind,
		CandidateTrigger:     adapter.CandidateTrigger,
		CandidateSeed:        adapter.CandidateSeed,
		CandidateTextStatus:  adapter.GeneratedTextStatus,
		CandidateText:        strings.TrimSpace(adapter.GeneratedText),
		CandidateTextHash:    adapter.GeneratedTextHash,
		JobID:                adapter.JobID,
		ShellID:              adapter.ShellID,
		CandidateExecutionID: adapter.CandidateExecutionID,
		GeneratorAdapterID:   adapter.AdapterID,
		TurnTextHash:         adapter.TurnTextHash,
	}
	if adapter.Schema == "" {
		draft.Reason = "missing_generator_adapter"
		return draft
	}
	if !adapter.Passed {
		draft.Reason = "generator adapter failed"
		if adapter.Reason != "" {
			draft.Reason += ": " + adapter.Reason
		}
		return draft
	}
	if adapter.AdapterID == "" {
		draft.Reason = "missing generator adapter id for shell " + adapter.ShellID
		return draft
	}
	if adapter.GeneratedTextStatus != "generated" {
		draft.Reason = "generator adapter text status is " + adapter.GeneratedTextStatus
		return draft
	}
	generated := strings.TrimSpace(adapter.GeneratedText)
	if generated == "" {
		draft.Reason = "missing generated text for adapter " + adapter.AdapterID
		return draft
	}
	if adapter.GeneratedTextHash == "" || adapter.GeneratedTextHash != hashJSON(generated) {
		draft.Reason = "generator adapter text hash mismatch"
		return draft
	}
	if wantAdapterID := admissionLiveRouteTurnGeneratorAdapterID(adapter); wantAdapterID == "" || adapter.AdapterID != wantAdapterID {
		draft.Reason = "generator adapter id mismatch"
		return draft
	}
	route, ok := admissionLiveRouteGenerationRouteFor(adapter.Route)
	if !ok {
		draft.Reason = "unknown generation route " + adapter.Route
		return draft
	}
	if adapter.Backend != route.Backend || adapter.Entrypoint != route.Entrypoint || adapter.PromptFrame != route.PromptFrame {
		draft.Reason = "generation route mismatch for adapter " + adapter.AdapterID
		return draft
	}
	if adapter.CandidateTextStatus != "pending_generation" {
		draft.Reason = "generator adapter shell text status is " + adapter.CandidateTextStatus
		return draft
	}
	shell := admissionLiveRouteTurnCandidateShell{
		Schema:              admissionLiveRouteTurnCandidateShellSchema,
		PromptClass:         adapter.PromptClass,
		Route:               adapter.Route,
		Source:              adapter.Source,
		ExpectedSource:      adapter.ExpectedSource,
		Backend:             adapter.Backend,
		Entrypoint:          adapter.Entrypoint,
		PromptFrame:         adapter.PromptFrame,
		CandidateSchema:     adapter.CandidateSchema,
		CandidateKind:       adapter.CandidateKind,
		CandidateTrigger:    adapter.CandidateTrigger,
		CandidateSeed:       adapter.CandidateSeed,
		CandidateTextStatus: adapter.CandidateTextStatus,
		JobID:               adapter.JobID,
		ShellID:             adapter.ShellID,
		Passed:              true,
		TurnTextHash:        adapter.TurnTextHash,
	}
	if wantShellID := admissionLiveRouteTurnCandidateShellID(shell); wantShellID == "" || adapter.ShellID != wantShellID {
		draft.Reason = "generator adapter shell id mismatch"
		return draft
	}
	draft = admissionLiveRouteTurnCandidateDraftForShell(shell, generated)
	draft.CandidateExecutionID = adapter.CandidateExecutionID
	draft.GeneratorAdapterID = adapter.AdapterID
	return draft
}

func admissionLiveRouteTurnCandidateDraftID(draft admissionLiveRouteTurnCandidateDraft) string {
	h := hashJSON(struct {
		ShellID           string `json:"shell_id"`
		CandidateRunID    string `json:"candidate_run_id"`
		CandidateTextHash string `json:"candidate_text_hash"`
	}{
		ShellID:           draft.ShellID,
		CandidateRunID:    draft.CandidateRunID,
		CandidateTextHash: draft.CandidateTextHash,
	})
	if h == "" {
		return ""
	}
	return "draft-" + h
}

func admissionLiveRouteTurnCandidateForDraft(draft admissionLiveRouteTurnCandidateDraft) dreamCandidate {
	if !draft.Passed {
		return dreamCandidate{}
	}
	return newDreamCandidate(draft.Source, draft.CandidateTrigger, draft.CandidateSeed, "", draft.CandidateText, nil)
}

func recordAdmissionLiveRouteTurnCandidateDraft(draft admissionLiveRouteTurnCandidateDraft) error {
	path := strings.TrimSpace(os.Getenv("AM_LIVE_ROUTE_TURN_CANDIDATE_DRAFT_LOG"))
	if path == "" {
		return nil
	}
	f, err := os.OpenFile(path, os.O_CREATE|os.O_APPEND|os.O_WRONLY, 0600)
	if err != nil {
		return err
	}
	enc := json.NewEncoder(f)
	err = enc.Encode(draft)
	if closeErr := f.Close(); err == nil {
		err = closeErr
	}
	return err
}

func admissionLiveRouteTurnCandidateReviewForDraft(obs admissionLiveRouteTurnObservation, draft admissionLiveRouteTurnCandidateDraft) admissionLiveRouteTurnCandidateReview {
	review := admissionLiveRouteTurnCandidateReview{
		Schema:               admissionLiveRouteTurnReviewSchema,
		Timing:               "async_subconscious",
		TurnPromptClass:      obs.PromptClass,
		TurnRoute:            obs.Route,
		TurnExpectedSource:   obs.ExpectedSource,
		TurnPassed:           obs.Passed,
		CandidateRunID:       draft.CandidateRunID,
		CandidateDraftID:     draft.DraftID,
		CandidateExecutionID: draft.CandidateExecutionID,
		GeneratorAdapterID:   draft.GeneratorAdapterID,
		CandidateTextStatus:  draft.CandidateTextStatus,
		CandidateTextHash:    draft.CandidateTextHash,
		CandidateSource:      normalizeDreamAdmissionSource(draft.Source),
		CandidateTrigger:     draft.CandidateTrigger,
	}
	if obs.Schema == "" {
		review.Reason = "missing_turn_observation"
		return review
	}
	if !obs.Passed {
		review.Reason = "turn_route_failed"
		if obs.Reason != "" {
			review.Reason += ": " + obs.Reason
		}
		return review
	}
	if draft.Schema == "" {
		review.Reason = "missing_candidate_draft"
		return review
	}
	if !draft.Passed {
		review.Reason = "candidate_draft_failed"
		if draft.Reason != "" {
			review.Reason += ": " + draft.Reason
		}
		return review
	}
	if draft.DraftID == "" {
		review.Reason = "missing_candidate_draft_id"
		return review
	}
	if wantDraftID := admissionLiveRouteTurnCandidateDraftID(draft); wantDraftID == "" || draft.DraftID != wantDraftID {
		review.Reason = "candidate_draft_id_mismatch"
		return review
	}
	if draft.GeneratorAdapterID == "" {
		review.Reason = "missing_generator_adapter_id for draft " + draft.DraftID
		return review
	}
	if draft.CandidateTextStatus != "generated" {
		review.Reason = "candidate_draft_text_status_is " + draft.CandidateTextStatus
		return review
	}
	text := strings.TrimSpace(draft.CandidateText)
	if text == "" {
		review.Reason = "missing_candidate_text for draft " + draft.DraftID
		return review
	}
	if draft.CandidateTextHash == "" || draft.CandidateTextHash != hashJSON(text) {
		review.Reason = "candidate_draft_text_hash_mismatch"
		return review
	}
	candidate := admissionLiveRouteTurnCandidateForDraft(draft)
	if candidate.Schema == "" {
		review.Reason = "candidate_draft_failed"
		return review
	}
	if candidate.RunID != draft.CandidateRunID {
		review.Reason = "candidate_run_id_mismatch"
		return review
	}
	choice := admissionLiveRouteChoiceForCandidate(candidate)
	review.CandidatePromptClass = choice.PromptClass
	review.CandidateRoute = choice.Route
	review.CandidateExpectedSource = choice.ExpectedSource
	review.CandidateChoicePassed = choice.Passed
	if !choice.Passed {
		review.Reason = "candidate_route_failed"
		if choice.Reason != "" {
			review.Reason += ": " + choice.Reason
		}
		return review
	}
	if choice.PromptClass != draft.PromptClass || choice.Route != draft.Route || choice.ExpectedSource != draft.ExpectedSource {
		review.Reason = "candidate_draft_route_mismatch: class " + choice.PromptClass +
			" route " + choice.Route + " expected " + choice.ExpectedSource
		return review
	}
	if review.CandidateSource != obs.ExpectedSource {
		review.Reason = "candidate_source_mismatch: source " + review.CandidateSource +
			" does not match turn expected " + obs.ExpectedSource + " for prompt class " + obs.PromptClass
		return review
	}
	if review.CandidateRoute != obs.Route {
		review.Reason = "candidate_route_mismatch: route " + review.CandidateRoute +
			" does not match turn route " + obs.Route + " for prompt class " + obs.PromptClass
		return review
	}
	review.Matched = true
	return review
}

func admissionLiveRouteTurnCandidateReviewForDream(obs admissionLiveRouteTurnObservation, c dreamCandidate) admissionLiveRouteTurnCandidateReview {
	review := admissionLiveRouteTurnCandidateReview{
		Schema:             admissionLiveRouteTurnReviewSchema,
		Timing:             "async_subconscious",
		TurnPromptClass:    obs.PromptClass,
		TurnRoute:          obs.Route,
		TurnExpectedSource: obs.ExpectedSource,
		TurnPassed:         obs.Passed,
		CandidateRunID:     c.RunID,
		CandidateSource:    normalizeDreamAdmissionSource(c.Source),
		CandidateTrigger:   c.Trigger,
	}
	if obs.Schema == "" {
		review.Reason = "missing_turn_observation"
		return review
	}
	if !obs.Passed {
		review.Reason = "turn_route_failed"
		if obs.Reason != "" {
			review.Reason += ": " + obs.Reason
		}
		return review
	}
	if c.Schema == "" {
		review.Reason = "untyped_candidate"
		return review
	}
	choice, bridgeApplied, bridgeTrigger := admissionLiveRouteChoiceForCandidateWithTurnBridge(obs, c)
	review.CandidateBridgeApplied = bridgeApplied
	review.CandidateBridgeTrigger = bridgeTrigger
	if !bridgeApplied && c.Admission != nil && c.Admission.LiveRouteChoice != nil {
		choice = *c.Admission.LiveRouteChoice
	}
	review.CandidatePromptClass = choice.PromptClass
	review.CandidateRoute = choice.Route
	review.CandidateExpectedSource = choice.ExpectedSource
	review.CandidateChoicePassed = choice.Passed
	if !choice.Passed {
		review.Reason = "candidate_route_failed"
		if choice.Reason != "" {
			review.Reason += ": " + choice.Reason
		}
		return review
	}
	if review.CandidateSource != obs.ExpectedSource {
		review.Reason = "candidate_source_mismatch: source " + review.CandidateSource +
			" does not match turn expected " + obs.ExpectedSource + " for prompt class " + obs.PromptClass
		return review
	}
	if review.CandidateRoute != obs.Route {
		review.Reason = "candidate_route_mismatch: route " + review.CandidateRoute +
			" does not match turn route " + obs.Route + " for prompt class " + obs.PromptClass
		return review
	}
	review.Matched = true
	return review
}

func admissionLiveRouteChoiceForCandidateWithTurnBridge(obs admissionLiveRouteTurnObservation, c dreamCandidate) (admissionLiveRouteChoice, bool, string) {
	choiceCandidate := c
	if admissionLiveRouteTurnBridgeDryRun() {
		if bridged, ok := admissionLiveRouteTurnBridgeCandidate(obs, c); ok {
			choiceCandidate = bridged
			return admissionLiveRouteChoiceForCandidate(choiceCandidate), true, bridged.Trigger
		}
	}
	return admissionLiveRouteChoiceForCandidate(choiceCandidate), false, ""
}

func admissionLiveRouteTurnBridgeDryRun() bool {
	return dreamAdmissionBoolEnv("AM_LIVE_ROUTE_TURN_BRIDGE_DRY_RUN")
}

func admissionLiveRouteTurnBridgeCandidate(obs admissionLiveRouteTurnObservation, c dreamCandidate) (dreamCandidate, bool) {
	if !obs.Passed || obs.PromptClass == "" || !qloopSweepKnownPromptClass(obs.PromptClass) {
		return c, false
	}
	if normalizeDreamAdmissionSource(c.Source) != "nano" || strings.TrimSpace(c.Trigger) != "human-turn" {
		return c, false
	}
	c.Trigger = admissionLiveRouteTurnBridgeTrigger(obs.PromptClass)
	return c, true
}

func admissionLiveRouteTurnBridgeTrigger(promptClass string) string {
	promptClass = strings.TrimSpace(promptClass)
	if promptClass == "" {
		return "human-turn"
	}
	return "human-turn-" + promptClass
}

func recordAdmissionLiveRouteTurnCandidateReview(review admissionLiveRouteTurnCandidateReview) error {
	path := strings.TrimSpace(os.Getenv("AM_LIVE_ROUTE_TURN_REVIEW_LOG"))
	if path == "" {
		return nil
	}
	f, err := os.OpenFile(path, os.O_CREATE|os.O_APPEND|os.O_WRONLY, 0600)
	if err != nil {
		return err
	}
	enc := json.NewEncoder(f)
	err = enc.Encode(review)
	if closeErr := f.Close(); err == nil {
		err = closeErr
	}
	return err
}

func admissionLiveRouteTurnCandidateAdmissionDryRun() bool {
	return dreamAdmissionBoolEnv("AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_DRY_RUN")
}

func admissionLiveRouteTurnCandidateAdmissionForDraftReview(obs admissionLiveRouteTurnObservation, draft admissionLiveRouteTurnCandidateDraft, review admissionLiveRouteTurnCandidateReview) admissionLiveRouteTurnCandidateAdmission {
	admission := admissionLiveRouteTurnCandidateAdmission{
		Schema:               admissionLiveRouteTurnCandidateAdmissionSchema,
		Timing:               "pre_admission_handoff",
		PromptClass:          draft.PromptClass,
		Route:                draft.Route,
		Source:               normalizeDreamAdmissionSource(draft.Source),
		ExpectedSource:       draft.ExpectedSource,
		CandidateSchema:      draft.CandidateSchema,
		CandidateKind:        draft.CandidateKind,
		CandidateTrigger:     draft.CandidateTrigger,
		CandidateSeed:        draft.CandidateSeed,
		CandidateRunID:       draft.CandidateRunID,
		CandidateDraftID:     draft.DraftID,
		CandidateExecutionID: draft.CandidateExecutionID,
		GeneratorAdapterID:   draft.GeneratorAdapterID,
		CandidateTextStatus:  draft.CandidateTextStatus,
		CandidateTextHash:    draft.CandidateTextHash,
		ReviewMatched:        review.Matched,
		TurnTextHash:         draft.TurnTextHash,
	}
	if obs.Schema == "" {
		admission.Reason = "missing_turn_observation"
		return admission
	}
	if !obs.Passed {
		admission.PromptClass = obs.PromptClass
		admission.Route = obs.Route
		admission.ExpectedSource = obs.ExpectedSource
		admission.Reason = "turn_route_failed"
		if obs.Reason != "" {
			admission.Reason += ": " + obs.Reason
		}
		return admission
	}
	if draft.Schema == "" {
		admission.Reason = "missing_candidate_draft"
		return admission
	}
	if !draft.Passed {
		admission.Reason = "candidate_draft_failed"
		if draft.Reason != "" {
			admission.Reason += ": " + draft.Reason
		}
		return admission
	}
	if review.Schema == "" {
		admission.Reason = "missing_candidate_review"
		return admission
	}
	if !review.Matched {
		admission.Reason = "candidate_review_failed"
		if review.Reason != "" {
			admission.Reason += ": " + review.Reason
		}
		return admission
	}
	if review.CandidateDraftID != draft.DraftID {
		admission.Reason = "candidate_review_draft_id_mismatch"
		return admission
	}
	if review.GeneratorAdapterID != draft.GeneratorAdapterID {
		admission.Reason = "candidate_review_adapter_id_mismatch"
		return admission
	}
	if review.CandidateExecutionID != draft.CandidateExecutionID {
		admission.Reason = "candidate_review_execution_id_mismatch"
		return admission
	}
	if review.CandidateRunID != draft.CandidateRunID {
		admission.Reason = "candidate_review_run_id_mismatch"
		return admission
	}
	if review.CandidateTextStatus != draft.CandidateTextStatus || review.CandidateTextHash != draft.CandidateTextHash {
		admission.Reason = "candidate_review_text_mismatch"
		return admission
	}
	if review.TurnPromptClass != obs.PromptClass || review.TurnRoute != obs.Route || review.TurnExpectedSource != obs.ExpectedSource {
		admission.Reason = "candidate_review_turn_mismatch"
		return admission
	}
	if review.CandidateSource != admission.Source ||
		review.CandidateTrigger != draft.CandidateTrigger ||
		review.CandidatePromptClass != draft.PromptClass ||
		review.CandidateRoute != draft.Route ||
		review.CandidateExpectedSource != draft.ExpectedSource {
		admission.Reason = "candidate_review_route_mismatch"
		return admission
	}
	if admission.CandidateDraftID == "" {
		admission.Reason = "missing_candidate_draft_id"
		return admission
	}
	if wantDraftID := admissionLiveRouteTurnCandidateDraftID(draft); wantDraftID == "" || admission.CandidateDraftID != wantDraftID {
		admission.Reason = "candidate_draft_id_mismatch"
		return admission
	}
	if admission.GeneratorAdapterID == "" {
		admission.Reason = "missing_generator_adapter_id for draft " + admission.CandidateDraftID
		return admission
	}
	if admission.CandidateTextStatus != "generated" {
		admission.Reason = "candidate_text_status_is " + admission.CandidateTextStatus
		return admission
	}
	text := strings.TrimSpace(draft.CandidateText)
	if text == "" {
		admission.Reason = "missing_candidate_text for draft " + admission.CandidateDraftID
		return admission
	}
	if admission.CandidateTextHash == "" || admission.CandidateTextHash != hashJSON(text) {
		admission.Reason = "candidate_text_hash_mismatch"
		return admission
	}
	candidate := admissionLiveRouteTurnCandidateForDraft(draft)
	if candidate.Schema != "arianna.dream_candidate.v1" {
		admission.Reason = "candidate_draft_candidate_missing"
		return admission
	}
	if candidate.RunID != admission.CandidateRunID ||
		normalizeDreamAdmissionSource(candidate.Source) != admission.Source ||
		candidate.Trigger != admission.CandidateTrigger ||
		candidate.Seed != admission.CandidateSeed ||
		candidate.Kind != admission.CandidateKind ||
		hashJSON(candidate.Text) != admission.CandidateTextHash {
		admission.Reason = "candidate_draft_candidate_mismatch"
		return admission
	}
	admission.HandoffID = admissionLiveRouteTurnCandidateAdmissionID(admission)
	if admission.HandoffID == "" {
		admission.Reason = "missing_candidate_admission_handoff_id"
		return admission
	}
	admission.Passed = true
	return admission
}

func admissionLiveRouteTurnCandidateAdmissionID(admission admissionLiveRouteTurnCandidateAdmission) string {
	h := hashJSON(struct {
		CandidateDraftID   string `json:"candidate_draft_id"`
		GeneratorAdapterID string `json:"generator_adapter_id"`
		CandidateRunID     string `json:"candidate_run_id"`
		CandidateTextHash  string `json:"candidate_text_hash"`
		TurnTextHash       string `json:"turn_text_hash"`
	}{
		CandidateDraftID:   admission.CandidateDraftID,
		GeneratorAdapterID: admission.GeneratorAdapterID,
		CandidateRunID:     admission.CandidateRunID,
		CandidateTextHash:  admission.CandidateTextHash,
		TurnTextHash:       admission.TurnTextHash,
	})
	if h == "" {
		return ""
	}
	return "handoff-" + h
}

func recordAdmissionLiveRouteTurnCandidateAdmission(admission admissionLiveRouteTurnCandidateAdmission) error {
	path := strings.TrimSpace(os.Getenv("AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_LOG"))
	if path == "" {
		return nil
	}
	f, err := os.OpenFile(path, os.O_CREATE|os.O_APPEND|os.O_WRONLY, 0600)
	if err != nil {
		return err
	}
	enc := json.NewEncoder(f)
	err = enc.Encode(admission)
	if closeErr := f.Close(); err == nil {
		err = closeErr
	}
	return err
}

func admissionLiveRouteTurnCandidateAdmissionAdapterDryRun() bool {
	return dreamAdmissionBoolEnv("AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_ADAPTER_DRY_RUN")
}

func admissionLiveRouteTurnCandidateAdmissionShadowDryRun() bool {
	return dreamAdmissionBoolEnv("AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_SHADOW_DRY_RUN")
}

func admissionLiveRouteTurnCandidateAdmissionDecisionDryRun() bool {
	return dreamAdmissionBoolEnv("AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_DECISION_DRY_RUN")
}

func admissionLiveRouteTurnCandidateAdmissionPromotionDryRun() bool {
	return dreamAdmissionBoolEnv("AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_PROMOTION_DRY_RUN")
}

func admissionLiveRouteTurnCandidateAdmissionSwitchDryRun() bool {
	return dreamAdmissionBoolEnv("AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_SWITCH_DRY_RUN")
}

func admissionLiveRouteTurnCandidateAdmissionEnableGateDryRun() bool {
	return dreamAdmissionBoolEnv("AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_ENABLE_GATE_DRY_RUN")
}

func admissionLiveRouteTurnCandidateAdmissionLiveStageDryRun() bool {
	return dreamAdmissionBoolEnv("AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_LIVE_STAGE_DRY_RUN")
}

func admissionLiveRouteTurnCandidateAdmissionWriterPreflightDryRun() bool {
	return dreamAdmissionBoolEnv("AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_WRITER_PREFLIGHT_DRY_RUN")
}

func admissionLiveRouteTurnCandidateAdmissionWriterInventoryDryRun() bool {
	return dreamAdmissionBoolEnv("AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_WRITER_INVENTORY_DRY_RUN")
}

func admissionLiveRouteTurnCandidateAdmissionWriterContractDryRun() bool {
	return dreamAdmissionBoolEnv("AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_WRITER_CONTRACT_DRY_RUN")
}

func admissionLiveRouteTurnCandidateAdmissionLedgerDryRun() bool {
	return dreamAdmissionBoolEnv("AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_LEDGER_DRY_RUN")
}

func admissionLiveRouteTurnCandidateAdmissionEnableGateKey() string {
	return strings.TrimSpace(os.Getenv("AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_ENABLE_GATE_KEY"))
}

func admissionLiveRouteTurnCandidateAdmissionAdapterForDraft(admission admissionLiveRouteTurnCandidateAdmission, draft admissionLiveRouteTurnCandidateDraft) admissionLiveRouteTurnCandidateAdmissionAdapter {
	adapter := admissionLiveRouteTurnCandidateAdmissionAdapter{
		Schema:               admissionLiveRouteTurnCandidateAdmissionAdapterSchema,
		Timing:               "admission_candidate_adapter",
		PromptClass:          admission.PromptClass,
		Route:                admission.Route,
		Source:               admission.Source,
		ExpectedSource:       admission.ExpectedSource,
		CandidateSchema:      admission.CandidateSchema,
		CandidateKind:        admission.CandidateKind,
		CandidateTrigger:     admission.CandidateTrigger,
		CandidateSeed:        admission.CandidateSeed,
		CandidateRunID:       admission.CandidateRunID,
		CandidateDraftID:     admission.CandidateDraftID,
		CandidateExecutionID: admission.CandidateExecutionID,
		GeneratorAdapterID:   admission.GeneratorAdapterID,
		HandoffID:            admission.HandoffID,
		CandidateTextStatus:  admission.CandidateTextStatus,
		CandidateTextHash:    admission.CandidateTextHash,
		TurnTextHash:         admission.TurnTextHash,
	}
	if admission.Schema == "" {
		adapter.Reason = "missing_candidate_admission_handoff"
		return adapter
	}
	if !admission.Passed {
		adapter.Reason = "candidate_admission_handoff_failed"
		if admission.Reason != "" {
			adapter.Reason += ": " + admission.Reason
		}
		return adapter
	}
	if draft.Schema == "" {
		adapter.Reason = "missing_candidate_draft"
		return adapter
	}
	if !draft.Passed {
		adapter.Reason = "candidate_draft_failed"
		if draft.Reason != "" {
			adapter.Reason += ": " + draft.Reason
		}
		return adapter
	}
	if admission.HandoffID == "" {
		adapter.Reason = "missing_candidate_admission_handoff_id"
		return adapter
	}
	if wantHandoffID := admissionLiveRouteTurnCandidateAdmissionID(admission); wantHandoffID == "" || admission.HandoffID != wantHandoffID {
		adapter.Reason = "candidate_admission_handoff_id_mismatch"
		return adapter
	}
	if admission.CandidateDraftID != draft.DraftID {
		adapter.Reason = "candidate_admission_draft_id_mismatch"
		return adapter
	}
	if admission.GeneratorAdapterID != draft.GeneratorAdapterID {
		adapter.Reason = "candidate_admission_generator_adapter_id_mismatch"
		return adapter
	}
	if admission.CandidateExecutionID != draft.CandidateExecutionID {
		adapter.Reason = "candidate_admission_execution_id_mismatch"
		return adapter
	}
	if admission.CandidateRunID != draft.CandidateRunID {
		adapter.Reason = "candidate_admission_run_id_mismatch"
		return adapter
	}
	if admission.CandidateTextStatus != draft.CandidateTextStatus || admission.CandidateTextHash != draft.CandidateTextHash {
		adapter.Reason = "candidate_admission_text_mismatch"
		return adapter
	}
	if admission.Source != normalizeDreamAdmissionSource(draft.Source) ||
		admission.CandidateTrigger != draft.CandidateTrigger ||
		admission.CandidateSeed != draft.CandidateSeed ||
		admission.CandidateKind != draft.CandidateKind ||
		admission.PromptClass != draft.PromptClass ||
		admission.Route != draft.Route ||
		admission.ExpectedSource != draft.ExpectedSource ||
		admission.TurnTextHash != draft.TurnTextHash {
		adapter.Reason = "candidate_admission_route_mismatch"
		return adapter
	}
	candidate := admissionLiveRouteTurnCandidateForDraft(draft)
	if candidate.Schema != "arianna.dream_candidate.v1" {
		adapter.Reason = "candidate_draft_candidate_missing"
		return adapter
	}
	if candidate.RunID != admission.CandidateRunID ||
		normalizeDreamAdmissionSource(candidate.Source) != admission.Source ||
		candidate.Trigger != admission.CandidateTrigger ||
		candidate.Seed != admission.CandidateSeed ||
		candidate.Kind != admission.CandidateKind ||
		hashJSON(candidate.Text) != admission.CandidateTextHash {
		adapter.Reason = "candidate_draft_candidate_mismatch"
		return adapter
	}
	adapter.DreamCandidateRunID = candidate.RunID
	adapter.AdmissionAdapterID = admissionLiveRouteTurnCandidateAdmissionAdapterID(adapter)
	if adapter.AdmissionAdapterID == "" {
		adapter.Reason = "missing_candidate_admission_adapter_id"
		return adapter
	}
	adapter.Passed = true
	return adapter
}

func admissionLiveRouteTurnCandidateAdmissionAdapterID(adapter admissionLiveRouteTurnCandidateAdmissionAdapter) string {
	h := hashJSON(struct {
		HandoffID          string `json:"handoff_id"`
		CandidateDraftID   string `json:"candidate_draft_id"`
		GeneratorAdapterID string `json:"generator_adapter_id"`
		CandidateRunID     string `json:"candidate_run_id"`
		CandidateTextHash  string `json:"candidate_text_hash"`
		TurnTextHash       string `json:"turn_text_hash"`
	}{
		HandoffID:          adapter.HandoffID,
		CandidateDraftID:   adapter.CandidateDraftID,
		GeneratorAdapterID: adapter.GeneratorAdapterID,
		CandidateRunID:     adapter.CandidateRunID,
		CandidateTextHash:  adapter.CandidateTextHash,
		TurnTextHash:       adapter.TurnTextHash,
	})
	if h == "" {
		return ""
	}
	return "admission-adapter-" + h
}

func admissionLiveRouteTurnCandidateForAdmissionAdapter(draft admissionLiveRouteTurnCandidateDraft, adapter admissionLiveRouteTurnCandidateAdmissionAdapter) dreamCandidate {
	if !adapter.Passed || adapter.AdmissionAdapterID == "" {
		return dreamCandidate{}
	}
	candidate := admissionLiveRouteTurnCandidateForDraft(draft)
	if candidate.Schema != "arianna.dream_candidate.v1" ||
		candidate.RunID != adapter.DreamCandidateRunID ||
		candidate.RunID != adapter.CandidateRunID ||
		hashJSON(candidate.Text) != adapter.CandidateTextHash {
		return dreamCandidate{}
	}
	candidate.LiveRouteCandidateAdmission = &adapter
	return candidate
}

func recordAdmissionLiveRouteTurnCandidateAdmissionAdapter(adapter admissionLiveRouteTurnCandidateAdmissionAdapter) error {
	path := strings.TrimSpace(os.Getenv("AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_ADAPTER_LOG"))
	if path == "" {
		return nil
	}
	f, err := os.OpenFile(path, os.O_CREATE|os.O_APPEND|os.O_WRONLY, 0600)
	if err != nil {
		return err
	}
	enc := json.NewEncoder(f)
	err = enc.Encode(adapter)
	if closeErr := f.Close(); err == nil {
		err = closeErr
	}
	return err
}

func admissionLiveRouteTurnCandidateAdmissionDecisionForShadow(
	execution admissionLiveRouteTurnCandidateExecution,
	generatorAdapter admissionLiveRouteTurnGeneratorAdapter,
	draft admissionLiveRouteTurnCandidateDraft,
	admission admissionLiveRouteTurnCandidateAdmission,
	adapter admissionLiveRouteTurnCandidateAdmissionAdapter,
	candidate dreamCandidate,
) admissionLiveRouteTurnCandidateAdmissionDecision {
	decision := admissionLiveRouteTurnCandidateAdmissionDecision{
		Schema:                admissionLiveRouteTurnCandidateAdmissionDecisionSchema,
		Timing:                "shadow_candidate_live_preflight",
		Decision:              "reject",
		PromptClass:           adapter.PromptClass,
		Route:                 adapter.Route,
		Source:                adapter.Source,
		ExpectedSource:        adapter.ExpectedSource,
		CandidateRunID:        adapter.CandidateRunID,
		CandidateDraftID:      adapter.CandidateDraftID,
		CandidateExecutionID:  adapter.CandidateExecutionID,
		GeneratorAdapterID:    adapter.GeneratorAdapterID,
		HandoffID:             adapter.HandoffID,
		AdmissionAdapterID:    adapter.AdmissionAdapterID,
		DreamCandidateRunID:   adapter.DreamCandidateRunID,
		CandidateTextStatus:   adapter.CandidateTextStatus,
		CandidateTextHash:     adapter.CandidateTextHash,
		DreamCandidateSchema:  candidate.Schema,
		DreamCandidateMode:    candidate.Mode,
		DreamAccepted:         candidate.Accepted,
		DreamReason:           candidate.Reason,
		AdmissionPolicyPassed: candidate.Admission != nil && candidate.Admission.Checked && candidate.Admission.Passed,
		LiveRouteChoicePassed: candidate.Admission != nil && candidate.Admission.LiveRouteChoice != nil && candidate.Admission.LiveRouteChoice.Passed,
		MutatesState:          false,
		TurnTextHash:          adapter.TurnTextHash,
	}
	if adapter.Schema == "" {
		decision.Reason = "missing_candidate_admission_adapter"
		return decision
	}
	if !adapter.Passed {
		decision.Reason = "candidate_admission_adapter_failed"
		if adapter.Reason != "" {
			decision.Reason += ": " + adapter.Reason
		}
		return decision
	}
	if execution.Schema == "" {
		decision.Reason = "missing_candidate_execution"
		return decision
	}
	if !execution.Passed {
		decision.Reason = "candidate_execution_failed"
		if execution.Reason != "" {
			decision.Reason += ": " + execution.Reason
		}
		return decision
	}
	if execution.Runner != admissionLiveRouteTurnCandidateExecutionRunnerNanoDirect {
		decision.Reason = "candidate execution runner " + execution.Runner + " is not nano-direct"
		return decision
	}
	if execution.RunnerStatus != admissionLiveRouteTurnCandidateExecutionStatusSucceeded {
		decision.Reason = "candidate execution runner status " + execution.RunnerStatus
		return decision
	}
	if execution.ExecutionID == "" || execution.ExecutionID != adapter.CandidateExecutionID {
		decision.Reason = "candidate_execution_id_mismatch"
		return decision
	}
	if generatorAdapter.Schema == "" || !generatorAdapter.Passed {
		decision.Reason = "generator_adapter_not_passed"
		if generatorAdapter.Reason != "" {
			decision.Reason += ": " + generatorAdapter.Reason
		}
		return decision
	}
	if generatorAdapter.AdapterID != adapter.GeneratorAdapterID ||
		generatorAdapter.CandidateExecutionID != adapter.CandidateExecutionID ||
		generatorAdapter.GeneratedTextHash != adapter.CandidateTextHash {
		decision.Reason = "generator_adapter_provenance_mismatch"
		return decision
	}
	if draft.Schema == "" || !draft.Passed {
		decision.Reason = "candidate_draft_not_passed"
		if draft.Reason != "" {
			decision.Reason += ": " + draft.Reason
		}
		return decision
	}
	if draft.DraftID != adapter.CandidateDraftID ||
		draft.CandidateExecutionID != adapter.CandidateExecutionID ||
		draft.GeneratorAdapterID != adapter.GeneratorAdapterID ||
		draft.CandidateRunID != adapter.CandidateRunID ||
		draft.CandidateTextHash != adapter.CandidateTextHash {
		decision.Reason = "candidate_draft_provenance_mismatch"
		return decision
	}
	if admission.Schema == "" || !admission.Passed {
		decision.Reason = "candidate_admission_handoff_not_passed"
		if admission.Reason != "" {
			decision.Reason += ": " + admission.Reason
		}
		return decision
	}
	if admission.HandoffID != adapter.HandoffID ||
		admission.CandidateDraftID != adapter.CandidateDraftID ||
		admission.CandidateExecutionID != adapter.CandidateExecutionID ||
		admission.GeneratorAdapterID != adapter.GeneratorAdapterID ||
		admission.CandidateRunID != adapter.CandidateRunID ||
		admission.CandidateTextHash != adapter.CandidateTextHash {
		decision.Reason = "candidate_admission_handoff_provenance_mismatch"
		return decision
	}
	if candidate.Schema == "" {
		decision.Reason = "missing_shadow_dream_candidate"
		return decision
	}
	if candidate.Schema != "arianna.dream_candidate.v1" {
		decision.Reason = "unexpected_shadow_dream_candidate_schema " + candidate.Schema
		return decision
	}
	if candidate.LiveRouteCandidateAdmission == nil {
		decision.Reason = "shadow_dream_candidate_missing_admission_adapter"
		return decision
	}
	if candidate.LiveRouteCandidateAdmission.AdmissionAdapterID != adapter.AdmissionAdapterID ||
		candidate.LiveRouteCandidateAdmission.HandoffID != adapter.HandoffID ||
		candidate.LiveRouteCandidateAdmission.CandidateDraftID != adapter.CandidateDraftID ||
		candidate.LiveRouteCandidateAdmission.CandidateExecutionID != adapter.CandidateExecutionID ||
		candidate.LiveRouteCandidateAdmission.GeneratorAdapterID != adapter.GeneratorAdapterID {
		decision.Reason = "shadow_dream_candidate_adapter_mismatch"
		return decision
	}
	if candidate.RunID != adapter.DreamCandidateRunID || candidate.RunID != adapter.CandidateRunID {
		decision.Reason = "shadow_dream_candidate_run_mismatch"
		return decision
	}
	if hashJSON(candidate.Text) != adapter.CandidateTextHash {
		decision.Reason = "shadow_dream_candidate_text_mismatch"
		return decision
	}
	if candidate.Mode != dreamAdmissionShadow || candidate.Accepted {
		decision.Reason = "shadow_dream_candidate_not_shadow_only"
		return decision
	}
	if candidate.Admission == nil || !candidate.Admission.Checked {
		decision.Reason = "shadow_dream_candidate_missing_admission_policy"
		return decision
	}
	if !candidate.Admission.Passed {
		decision.Reason = "shadow_dream_candidate_admission_policy_failed"
		if len(candidate.Admission.Reasons) > 0 {
			decision.Reason += ": " + strings.Join(candidate.Admission.Reasons, "; ")
		}
		return decision
	}
	if candidate.Admission.LiveRouteChoice == nil || !candidate.Admission.LiveRouteChoice.Passed {
		decision.Reason = "shadow_dream_candidate_live_route_choice_failed"
		if candidate.Admission.LiveRouteChoice != nil && candidate.Admission.LiveRouteChoice.Reason != "" {
			decision.Reason += ": " + candidate.Admission.LiveRouteChoice.Reason
		}
		return decision
	}
	decision.Decision = "shadow_ready"
	decision.LiveReady = true
	decision.DecisionID = admissionLiveRouteTurnCandidateAdmissionDecisionID(decision)
	if decision.DecisionID == "" {
		decision.Reason = "missing_candidate_admission_decision_id"
		return decision
	}
	decision.Passed = true
	decision.Reason = "shadow ready; live mutation still disabled"
	return decision
}

func admissionLiveRouteTurnCandidateAdmissionDecisionID(decision admissionLiveRouteTurnCandidateAdmissionDecision) string {
	h := hashJSON(struct {
		AdmissionAdapterID   string `json:"admission_adapter_id"`
		HandoffID            string `json:"handoff_id"`
		CandidateDraftID     string `json:"candidate_draft_id"`
		CandidateExecutionID string `json:"candidate_execution_id"`
		CandidateRunID       string `json:"candidate_run_id"`
		CandidateTextHash    string `json:"candidate_text_hash"`
		TurnTextHash         string `json:"turn_text_hash"`
	}{
		AdmissionAdapterID:   decision.AdmissionAdapterID,
		HandoffID:            decision.HandoffID,
		CandidateDraftID:     decision.CandidateDraftID,
		CandidateExecutionID: decision.CandidateExecutionID,
		CandidateRunID:       decision.CandidateRunID,
		CandidateTextHash:    decision.CandidateTextHash,
		TurnTextHash:         decision.TurnTextHash,
	})
	if h == "" {
		return ""
	}
	return "decision-" + h
}

func recordAdmissionLiveRouteTurnCandidateAdmissionDecision(decision admissionLiveRouteTurnCandidateAdmissionDecision) error {
	path := strings.TrimSpace(os.Getenv("AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_DECISION_LOG"))
	if path == "" {
		return nil
	}
	f, err := os.OpenFile(path, os.O_CREATE|os.O_APPEND|os.O_WRONLY, 0600)
	if err != nil {
		return err
	}
	enc := json.NewEncoder(f)
	err = enc.Encode(decision)
	if closeErr := f.Close(); err == nil {
		err = closeErr
	}
	return err
}

func admissionLiveRouteTurnCandidateAdmissionPromotionForDecision(decision admissionLiveRouteTurnCandidateAdmissionDecision) admissionLiveRouteTurnCandidateAdmissionPromotion {
	promotion := admissionLiveRouteTurnCandidateAdmissionPromotion{
		Schema:                admissionLiveRouteTurnCandidateAdmissionPromotionSchema,
		Timing:                "admission_decision_consumer",
		Promotion:             "blocked",
		PromptClass:           decision.PromptClass,
		Route:                 decision.Route,
		Source:                decision.Source,
		ExpectedSource:        decision.ExpectedSource,
		CandidateRunID:        decision.CandidateRunID,
		CandidateDraftID:      decision.CandidateDraftID,
		CandidateExecutionID:  decision.CandidateExecutionID,
		GeneratorAdapterID:    decision.GeneratorAdapterID,
		HandoffID:             decision.HandoffID,
		AdmissionAdapterID:    decision.AdmissionAdapterID,
		AdmissionDecisionID:   decision.DecisionID,
		AdmissionDecision:     decision.Decision,
		DreamCandidateRunID:   decision.DreamCandidateRunID,
		CandidateTextStatus:   decision.CandidateTextStatus,
		CandidateTextHash:     decision.CandidateTextHash,
		AdmissionPolicyPassed: decision.AdmissionPolicyPassed,
		LiveRouteChoicePassed: decision.LiveRouteChoicePassed,
		SourceDecisionPassed:  decision.Passed,
		LiveReady:             decision.LiveReady,
		LiveAdmissionEnabled:  false,
		MutatesState:          false,
		TurnTextHash:          decision.TurnTextHash,
	}
	if decision.Schema == "" {
		promotion.Reason = "missing_candidate_admission_decision"
		return promotion
	}
	if decision.Schema != admissionLiveRouteTurnCandidateAdmissionDecisionSchema {
		promotion.Reason = "unexpected_candidate_admission_decision_schema " + decision.Schema
		return promotion
	}
	if !decision.Passed {
		promotion.Reason = "candidate_admission_decision_failed"
		if decision.Reason != "" {
			promotion.Reason += ": " + decision.Reason
		}
		return promotion
	}
	if decision.Decision != "shadow_ready" {
		promotion.Reason = "candidate_admission_decision_not_shadow_ready"
		return promotion
	}
	if !decision.LiveReady {
		promotion.Reason = "candidate_admission_decision_not_live_ready"
		return promotion
	}
	if decision.MutatesState {
		promotion.Reason = "candidate_admission_decision_already_mutates_state"
		return promotion
	}
	if decision.DecisionID == "" {
		promotion.Reason = "missing_candidate_admission_decision_id"
		return promotion
	}
	if wantDecisionID := admissionLiveRouteTurnCandidateAdmissionDecisionID(decision); wantDecisionID == "" || decision.DecisionID != wantDecisionID {
		promotion.Reason = "candidate_admission_decision_id_mismatch"
		return promotion
	}
	if !decision.AdmissionPolicyPassed {
		promotion.Reason = "candidate_admission_decision_policy_not_passed"
		return promotion
	}
	if !decision.LiveRouteChoicePassed {
		promotion.Reason = "candidate_admission_decision_live_route_not_passed"
		return promotion
	}
	if decision.CandidateRunID == "" ||
		decision.CandidateDraftID == "" ||
		decision.CandidateExecutionID == "" ||
		decision.GeneratorAdapterID == "" ||
		decision.HandoffID == "" ||
		decision.AdmissionAdapterID == "" ||
		decision.DreamCandidateRunID == "" ||
		decision.CandidateTextHash == "" ||
		decision.TurnTextHash == "" {
		promotion.Reason = "candidate_admission_decision_missing_provenance"
		return promotion
	}
	promotion.Promotion = "pending_live_admission"
	promotion.PromotionID = admissionLiveRouteTurnCandidateAdmissionPromotionID(promotion)
	if promotion.PromotionID == "" {
		promotion.Reason = "missing_candidate_admission_promotion_id"
		return promotion
	}
	promotion.Passed = true
	promotion.Reason = "shadow decision consumed; live admission still disabled"
	return promotion
}

func admissionLiveRouteTurnCandidateAdmissionPromotionID(promotion admissionLiveRouteTurnCandidateAdmissionPromotion) string {
	h := hashJSON(struct {
		AdmissionDecisionID string `json:"admission_decision_id"`
		AdmissionAdapterID  string `json:"admission_adapter_id"`
		CandidateRunID      string `json:"candidate_run_id"`
		CandidateTextHash   string `json:"candidate_text_hash"`
		TurnTextHash        string `json:"turn_text_hash"`
	}{
		AdmissionDecisionID: promotion.AdmissionDecisionID,
		AdmissionAdapterID:  promotion.AdmissionAdapterID,
		CandidateRunID:      promotion.CandidateRunID,
		CandidateTextHash:   promotion.CandidateTextHash,
		TurnTextHash:        promotion.TurnTextHash,
	})
	if h == "" {
		return ""
	}
	return "promotion-" + h
}

func recordAdmissionLiveRouteTurnCandidateAdmissionPromotion(promotion admissionLiveRouteTurnCandidateAdmissionPromotion) error {
	path := strings.TrimSpace(os.Getenv("AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_PROMOTION_LOG"))
	if path == "" {
		return nil
	}
	f, err := os.OpenFile(path, os.O_CREATE|os.O_APPEND|os.O_WRONLY, 0600)
	if err != nil {
		return err
	}
	enc := json.NewEncoder(f)
	err = enc.Encode(promotion)
	if closeErr := f.Close(); err == nil {
		err = closeErr
	}
	return err
}

func admissionLiveRouteTurnCandidateAdmissionSwitchForPromotion(promotion admissionLiveRouteTurnCandidateAdmissionPromotion) admissionLiveRouteTurnCandidateAdmissionSwitch {
	sw := admissionLiveRouteTurnCandidateAdmissionSwitch{
		Schema:                admissionLiveRouteTurnCandidateAdmissionSwitchSchema,
		Timing:                "live_admission_switch_guard",
		SwitchState:           "blocked",
		SwitchAction:          "reject",
		PromptClass:           promotion.PromptClass,
		Route:                 promotion.Route,
		Source:                promotion.Source,
		ExpectedSource:        promotion.ExpectedSource,
		CandidateRunID:        promotion.CandidateRunID,
		CandidateDraftID:      promotion.CandidateDraftID,
		CandidateExecutionID:  promotion.CandidateExecutionID,
		GeneratorAdapterID:    promotion.GeneratorAdapterID,
		HandoffID:             promotion.HandoffID,
		AdmissionAdapterID:    promotion.AdmissionAdapterID,
		AdmissionDecisionID:   promotion.AdmissionDecisionID,
		AdmissionPromotionID:  promotion.PromotionID,
		AdmissionDecision:     promotion.AdmissionDecision,
		AdmissionPromotion:    promotion.Promotion,
		DreamCandidateRunID:   promotion.DreamCandidateRunID,
		CandidateTextStatus:   promotion.CandidateTextStatus,
		CandidateTextHash:     promotion.CandidateTextHash,
		AdmissionPolicyPassed: promotion.AdmissionPolicyPassed,
		LiveRouteChoicePassed: promotion.LiveRouteChoicePassed,
		SourceDecisionPassed:  promotion.SourceDecisionPassed,
		SourcePromotionPassed: promotion.Passed,
		LiveReady:             promotion.LiveReady,
		LiveAdmissionEnabled:  false,
		AdmissionAllowed:      false,
		MutatesState:          false,
		TurnTextHash:          promotion.TurnTextHash,
	}
	if promotion.Schema == "" {
		sw.Reason = "missing_candidate_admission_promotion"
		return sw
	}
	if promotion.Schema != admissionLiveRouteTurnCandidateAdmissionPromotionSchema {
		sw.Reason = "unexpected_candidate_admission_promotion_schema " + promotion.Schema
		return sw
	}
	if !promotion.Passed {
		sw.Reason = "candidate_admission_promotion_failed"
		if promotion.Reason != "" {
			sw.Reason += ": " + promotion.Reason
		}
		return sw
	}
	if promotion.Promotion != "pending_live_admission" {
		sw.Reason = "candidate_admission_promotion_not_pending_live_admission"
		return sw
	}
	if promotion.PromotionID == "" {
		sw.Reason = "missing_candidate_admission_promotion_id"
		return sw
	}
	if wantPromotionID := admissionLiveRouteTurnCandidateAdmissionPromotionID(promotion); wantPromotionID == "" || promotion.PromotionID != wantPromotionID {
		sw.Reason = "candidate_admission_promotion_id_mismatch"
		return sw
	}
	if !promotion.LiveReady {
		sw.Reason = "candidate_admission_promotion_not_live_ready"
		return sw
	}
	if promotion.LiveAdmissionEnabled {
		sw.Reason = "candidate_admission_promotion_already_live_enabled"
		return sw
	}
	if promotion.MutatesState {
		sw.Reason = "candidate_admission_promotion_already_mutates_state"
		return sw
	}
	if !promotion.SourceDecisionPassed {
		sw.Reason = "candidate_admission_promotion_source_decision_not_passed"
		return sw
	}
	if !promotion.AdmissionPolicyPassed {
		sw.Reason = "candidate_admission_promotion_policy_not_passed"
		return sw
	}
	if !promotion.LiveRouteChoicePassed {
		sw.Reason = "candidate_admission_promotion_live_route_not_passed"
		return sw
	}
	if promotion.AdmissionDecisionID == "" ||
		promotion.AdmissionAdapterID == "" ||
		promotion.CandidateRunID == "" ||
		promotion.CandidateDraftID == "" ||
		promotion.CandidateExecutionID == "" ||
		promotion.GeneratorAdapterID == "" ||
		promotion.HandoffID == "" ||
		promotion.DreamCandidateRunID == "" ||
		promotion.CandidateTextHash == "" ||
		promotion.TurnTextHash == "" {
		sw.Reason = "candidate_admission_promotion_missing_provenance"
		return sw
	}
	sw.SwitchState = "disabled"
	sw.SwitchAction = "hold_pending_live_admission"
	sw.SwitchID = admissionLiveRouteTurnCandidateAdmissionSwitchID(sw)
	if sw.SwitchID == "" {
		sw.Reason = "missing_candidate_admission_switch_id"
		return sw
	}
	sw.Passed = true
	sw.Reason = "live admission switch disabled; pending promotion held without mutation"
	return sw
}

func admissionLiveRouteTurnCandidateAdmissionSwitchID(sw admissionLiveRouteTurnCandidateAdmissionSwitch) string {
	h := hashJSON(struct {
		AdmissionPromotionID string `json:"admission_promotion_id"`
		AdmissionDecisionID  string `json:"admission_decision_id"`
		AdmissionAdapterID   string `json:"admission_adapter_id"`
		CandidateRunID       string `json:"candidate_run_id"`
		CandidateTextHash    string `json:"candidate_text_hash"`
		TurnTextHash         string `json:"turn_text_hash"`
	}{
		AdmissionPromotionID: sw.AdmissionPromotionID,
		AdmissionDecisionID:  sw.AdmissionDecisionID,
		AdmissionAdapterID:   sw.AdmissionAdapterID,
		CandidateRunID:       sw.CandidateRunID,
		CandidateTextHash:    sw.CandidateTextHash,
		TurnTextHash:         sw.TurnTextHash,
	})
	if h == "" {
		return ""
	}
	return "switch-" + h
}

func recordAdmissionLiveRouteTurnCandidateAdmissionSwitch(sw admissionLiveRouteTurnCandidateAdmissionSwitch) error {
	path := strings.TrimSpace(os.Getenv("AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_SWITCH_LOG"))
	if path == "" {
		return nil
	}
	f, err := os.OpenFile(path, os.O_CREATE|os.O_APPEND|os.O_WRONLY, 0600)
	if err != nil {
		return err
	}
	enc := json.NewEncoder(f)
	err = enc.Encode(sw)
	if closeErr := f.Close(); err == nil {
		err = closeErr
	}
	return err
}

func admissionLiveRouteTurnCandidateAdmissionEnableGateForSwitch(sw admissionLiveRouteTurnCandidateAdmissionSwitch) admissionLiveRouteTurnCandidateAdmissionEnableGate {
	key := admissionLiveRouteTurnCandidateAdmissionEnableGateKey()
	gate := admissionLiveRouteTurnCandidateAdmissionEnableGate{
		Schema:                admissionLiveRouteTurnCandidateAdmissionEnableGateSchema,
		Timing:                "live_admission_enable_gate",
		EnableState:           "blocked",
		EnableAction:          "reject",
		PromptClass:           sw.PromptClass,
		Route:                 sw.Route,
		Source:                sw.Source,
		ExpectedSource:        sw.ExpectedSource,
		CandidateRunID:        sw.CandidateRunID,
		CandidateDraftID:      sw.CandidateDraftID,
		CandidateExecutionID:  sw.CandidateExecutionID,
		GeneratorAdapterID:    sw.GeneratorAdapterID,
		HandoffID:             sw.HandoffID,
		AdmissionAdapterID:    sw.AdmissionAdapterID,
		AdmissionDecisionID:   sw.AdmissionDecisionID,
		AdmissionPromotionID:  sw.AdmissionPromotionID,
		AdmissionSwitchID:     sw.SwitchID,
		AdmissionDecision:     sw.AdmissionDecision,
		AdmissionPromotion:    sw.AdmissionPromotion,
		SwitchState:           sw.SwitchState,
		SwitchAction:          sw.SwitchAction,
		DreamCandidateRunID:   sw.DreamCandidateRunID,
		CandidateTextStatus:   sw.CandidateTextStatus,
		CandidateTextHash:     sw.CandidateTextHash,
		AdmissionPolicyPassed: sw.AdmissionPolicyPassed,
		LiveRouteChoicePassed: sw.LiveRouteChoicePassed,
		SourceDecisionPassed:  sw.SourceDecisionPassed,
		SourcePromotionPassed: sw.SourcePromotionPassed,
		SourceSwitchPassed:    sw.Passed,
		LiveReady:             sw.LiveReady,
		LiveAdmissionEnabled:  false,
		AdmissionAllowed:      false,
		ManualEnableRequested: key != "",
		EnableKeyMatched:      key == admissionLiveRouteTurnCandidateAdmissionEnableGateConfirmation,
		MutatesState:          false,
		TurnTextHash:          sw.TurnTextHash,
	}
	if sw.Schema == "" {
		gate.Reason = "missing_candidate_admission_switch"
		return gate
	}
	if sw.Schema != admissionLiveRouteTurnCandidateAdmissionSwitchSchema {
		gate.Reason = "unexpected_candidate_admission_switch_schema " + sw.Schema
		return gate
	}
	if !sw.Passed {
		gate.Reason = "candidate_admission_switch_failed"
		if sw.Reason != "" {
			gate.Reason += ": " + sw.Reason
		}
		return gate
	}
	if sw.SwitchState != "disabled" {
		gate.Reason = "candidate_admission_switch_not_disabled"
		return gate
	}
	if sw.SwitchAction != "hold_pending_live_admission" {
		gate.Reason = "candidate_admission_switch_unexpected_action"
		return gate
	}
	if sw.SwitchID == "" {
		gate.Reason = "missing_candidate_admission_switch_id"
		return gate
	}
	if wantSwitchID := admissionLiveRouteTurnCandidateAdmissionSwitchID(sw); wantSwitchID == "" || sw.SwitchID != wantSwitchID {
		gate.Reason = "candidate_admission_switch_id_mismatch"
		return gate
	}
	if !sw.LiveReady {
		gate.Reason = "candidate_admission_switch_not_live_ready"
		return gate
	}
	if sw.LiveAdmissionEnabled {
		gate.Reason = "candidate_admission_switch_already_live_enabled"
		return gate
	}
	if sw.AdmissionAllowed {
		gate.Reason = "candidate_admission_switch_already_allows_admission"
		return gate
	}
	if sw.MutatesState {
		gate.Reason = "candidate_admission_switch_already_mutates_state"
		return gate
	}
	if !sw.SourcePromotionPassed {
		gate.Reason = "candidate_admission_switch_source_promotion_not_passed"
		return gate
	}
	if !sw.SourceDecisionPassed {
		gate.Reason = "candidate_admission_switch_source_decision_not_passed"
		return gate
	}
	if !sw.AdmissionPolicyPassed {
		gate.Reason = "candidate_admission_switch_policy_not_passed"
		return gate
	}
	if !sw.LiveRouteChoicePassed {
		gate.Reason = "candidate_admission_switch_live_route_not_passed"
		return gate
	}
	if sw.AdmissionPromotionID == "" ||
		sw.AdmissionDecisionID == "" ||
		sw.AdmissionAdapterID == "" ||
		sw.CandidateRunID == "" ||
		sw.CandidateDraftID == "" ||
		sw.CandidateExecutionID == "" ||
		sw.GeneratorAdapterID == "" ||
		sw.HandoffID == "" ||
		sw.DreamCandidateRunID == "" ||
		sw.CandidateTextHash == "" ||
		sw.TurnTextHash == "" {
		gate.Reason = "candidate_admission_switch_missing_provenance"
		return gate
	}
	if gate.ManualEnableRequested && !gate.EnableKeyMatched {
		gate.Reason = "live_admission_enable_gate_key_mismatch"
		return gate
	}
	if gate.ManualEnableRequested {
		gate.EnableState = "armed_dry_run"
		gate.EnableAction = "would_enable_live_admission_dry_run"
		gate.EnableGateID = admissionLiveRouteTurnCandidateAdmissionEnableGateID(gate)
		if gate.EnableGateID == "" {
			gate.Reason = "missing_candidate_admission_enable_gate_id"
			return gate
		}
		gate.Passed = true
		gate.Reason = "live admission enable key matched; dry-run still refuses mutation"
		return gate
	}
	gate.EnableState = "disabled"
	gate.EnableAction = "require_operator_key"
	gate.EnableGateID = admissionLiveRouteTurnCandidateAdmissionEnableGateID(gate)
	if gate.EnableGateID == "" {
		gate.Reason = "missing_candidate_admission_enable_gate_id"
		return gate
	}
	gate.Passed = true
	gate.Reason = "live admission enable gate closed; operator key absent"
	return gate
}

func admissionLiveRouteTurnCandidateAdmissionEnableGateID(gate admissionLiveRouteTurnCandidateAdmissionEnableGate) string {
	h := hashJSON(struct {
		AdmissionSwitchID     string `json:"admission_switch_id"`
		AdmissionPromotionID  string `json:"admission_promotion_id"`
		AdmissionDecisionID   string `json:"admission_decision_id"`
		AdmissionAdapterID    string `json:"admission_adapter_id"`
		CandidateRunID        string `json:"candidate_run_id"`
		CandidateTextHash     string `json:"candidate_text_hash"`
		TurnTextHash          string `json:"turn_text_hash"`
		EnableState           string `json:"enable_state"`
		EnableAction          string `json:"enable_action"`
		ManualEnableRequested bool   `json:"manual_enable_requested"`
		EnableKeyMatched      bool   `json:"enable_key_matched"`
	}{
		AdmissionSwitchID:     gate.AdmissionSwitchID,
		AdmissionPromotionID:  gate.AdmissionPromotionID,
		AdmissionDecisionID:   gate.AdmissionDecisionID,
		AdmissionAdapterID:    gate.AdmissionAdapterID,
		CandidateRunID:        gate.CandidateRunID,
		CandidateTextHash:     gate.CandidateTextHash,
		TurnTextHash:          gate.TurnTextHash,
		EnableState:           gate.EnableState,
		EnableAction:          gate.EnableAction,
		ManualEnableRequested: gate.ManualEnableRequested,
		EnableKeyMatched:      gate.EnableKeyMatched,
	})
	if h == "" {
		return ""
	}
	return "enable-" + h
}

func recordAdmissionLiveRouteTurnCandidateAdmissionEnableGate(gate admissionLiveRouteTurnCandidateAdmissionEnableGate) error {
	path := strings.TrimSpace(os.Getenv("AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_ENABLE_GATE_LOG"))
	if path == "" {
		return nil
	}
	f, err := os.OpenFile(path, os.O_CREATE|os.O_APPEND|os.O_WRONLY, 0600)
	if err != nil {
		return err
	}
	enc := json.NewEncoder(f)
	err = enc.Encode(gate)
	if closeErr := f.Close(); err == nil {
		err = closeErr
	}
	return err
}

func admissionLiveRouteTurnCandidateAdmissionLiveStageForEnableGate(gate admissionLiveRouteTurnCandidateAdmissionEnableGate) admissionLiveRouteTurnCandidateAdmissionLiveStage {
	stage := admissionLiveRouteTurnCandidateAdmissionLiveStage{
		Schema:                admissionLiveRouteTurnCandidateAdmissionLiveStageSchema,
		Timing:                "live_admission_candidate_stage",
		StageState:            "blocked",
		StageAction:           "reject",
		PromptClass:           gate.PromptClass,
		Route:                 gate.Route,
		Source:                gate.Source,
		ExpectedSource:        gate.ExpectedSource,
		CandidateRunID:        gate.CandidateRunID,
		CandidateDraftID:      gate.CandidateDraftID,
		CandidateExecutionID:  gate.CandidateExecutionID,
		GeneratorAdapterID:    gate.GeneratorAdapterID,
		HandoffID:             gate.HandoffID,
		AdmissionAdapterID:    gate.AdmissionAdapterID,
		AdmissionDecisionID:   gate.AdmissionDecisionID,
		AdmissionPromotionID:  gate.AdmissionPromotionID,
		AdmissionSwitchID:     gate.AdmissionSwitchID,
		AdmissionEnableGateID: gate.EnableGateID,
		AdmissionDecision:     gate.AdmissionDecision,
		AdmissionPromotion:    gate.AdmissionPromotion,
		SwitchState:           gate.SwitchState,
		SwitchAction:          gate.SwitchAction,
		EnableState:           gate.EnableState,
		EnableAction:          gate.EnableAction,
		DreamCandidateRunID:   gate.DreamCandidateRunID,
		CandidateTextStatus:   gate.CandidateTextStatus,
		CandidateTextHash:     gate.CandidateTextHash,
		AdmissionPolicyPassed: gate.AdmissionPolicyPassed,
		LiveRouteChoicePassed: gate.LiveRouteChoicePassed,
		SourceDecisionPassed:  gate.SourceDecisionPassed,
		SourcePromotionPassed: gate.SourcePromotionPassed,
		SourceSwitchPassed:    gate.SourceSwitchPassed,
		SourceEnablePassed:    gate.Passed,
		LiveReady:             gate.LiveReady,
		LiveAdmissionEnabled:  false,
		AdmissionAllowed:      false,
		ManualEnableRequested: gate.ManualEnableRequested,
		EnableKeyMatched:      gate.EnableKeyMatched,
		RequiresWriter:        true,
		WriterReady:           false,
		RequiresRollback:      true,
		RollbackReady:         false,
		MutatesState:          false,
		TurnTextHash:          gate.TurnTextHash,
	}
	if gate.Schema == "" {
		stage.Reason = "missing_candidate_admission_enable_gate"
		return stage
	}
	if gate.Schema != admissionLiveRouteTurnCandidateAdmissionEnableGateSchema {
		stage.Reason = "unexpected_candidate_admission_enable_gate_schema " + gate.Schema
		return stage
	}
	if !gate.Passed {
		stage.Reason = "candidate_admission_enable_gate_failed"
		if gate.Reason != "" {
			stage.Reason += ": " + gate.Reason
		}
		return stage
	}
	if gate.EnableState != "armed_dry_run" {
		stage.Reason = "candidate_admission_enable_gate_not_armed"
		return stage
	}
	if gate.EnableAction != "would_enable_live_admission_dry_run" {
		stage.Reason = "candidate_admission_enable_gate_unexpected_action"
		return stage
	}
	if gate.EnableGateID == "" {
		stage.Reason = "missing_candidate_admission_enable_gate_id"
		return stage
	}
	if wantGateID := admissionLiveRouteTurnCandidateAdmissionEnableGateID(gate); wantGateID == "" || gate.EnableGateID != wantGateID {
		stage.Reason = "candidate_admission_enable_gate_id_mismatch"
		return stage
	}
	if !gate.LiveReady {
		stage.Reason = "candidate_admission_enable_gate_not_live_ready"
		return stage
	}
	if gate.LiveAdmissionEnabled {
		stage.Reason = "candidate_admission_enable_gate_already_live_enabled"
		return stage
	}
	if gate.AdmissionAllowed {
		stage.Reason = "candidate_admission_enable_gate_already_allows_admission"
		return stage
	}
	if !gate.ManualEnableRequested {
		stage.Reason = "candidate_admission_enable_gate_missing_manual_enable"
		return stage
	}
	if !gate.EnableKeyMatched {
		stage.Reason = "candidate_admission_enable_gate_key_not_matched"
		return stage
	}
	if gate.MutatesState {
		stage.Reason = "candidate_admission_enable_gate_already_mutates_state"
		return stage
	}
	if !gate.SourceSwitchPassed {
		stage.Reason = "candidate_admission_enable_gate_source_switch_not_passed"
		return stage
	}
	if !gate.SourcePromotionPassed {
		stage.Reason = "candidate_admission_enable_gate_source_promotion_not_passed"
		return stage
	}
	if !gate.SourceDecisionPassed {
		stage.Reason = "candidate_admission_enable_gate_source_decision_not_passed"
		return stage
	}
	if !gate.AdmissionPolicyPassed {
		stage.Reason = "candidate_admission_enable_gate_policy_not_passed"
		return stage
	}
	if !gate.LiveRouteChoicePassed {
		stage.Reason = "candidate_admission_enable_gate_live_route_not_passed"
		return stage
	}
	if gate.AdmissionSwitchID == "" ||
		gate.AdmissionPromotionID == "" ||
		gate.AdmissionDecisionID == "" ||
		gate.AdmissionAdapterID == "" ||
		gate.CandidateRunID == "" ||
		gate.CandidateDraftID == "" ||
		gate.CandidateExecutionID == "" ||
		gate.GeneratorAdapterID == "" ||
		gate.HandoffID == "" ||
		gate.DreamCandidateRunID == "" ||
		gate.CandidateTextHash == "" ||
		gate.TurnTextHash == "" {
		stage.Reason = "candidate_admission_enable_gate_missing_provenance"
		return stage
	}
	stage.StageState = "staged_dry_run"
	stage.StageAction = "stage_live_candidate_dry_run"
	stage.LiveStageID = admissionLiveRouteTurnCandidateAdmissionLiveStageID(stage)
	if stage.LiveStageID == "" {
		stage.Reason = "missing_candidate_admission_live_stage_id"
		return stage
	}
	stage.Passed = true
	stage.Reason = "live admission candidate staged as dry-run; writer and rollback remain absent"
	return stage
}

func admissionLiveRouteTurnCandidateAdmissionLiveStageID(stage admissionLiveRouteTurnCandidateAdmissionLiveStage) string {
	h := hashJSON(struct {
		AdmissionEnableGateID string `json:"admission_enable_gate_id"`
		AdmissionSwitchID     string `json:"admission_switch_id"`
		AdmissionPromotionID  string `json:"admission_promotion_id"`
		AdmissionDecisionID   string `json:"admission_decision_id"`
		AdmissionAdapterID    string `json:"admission_adapter_id"`
		CandidateRunID        string `json:"candidate_run_id"`
		CandidateTextHash     string `json:"candidate_text_hash"`
		TurnTextHash          string `json:"turn_text_hash"`
		StageState            string `json:"stage_state"`
		StageAction           string `json:"stage_action"`
	}{
		AdmissionEnableGateID: stage.AdmissionEnableGateID,
		AdmissionSwitchID:     stage.AdmissionSwitchID,
		AdmissionPromotionID:  stage.AdmissionPromotionID,
		AdmissionDecisionID:   stage.AdmissionDecisionID,
		AdmissionAdapterID:    stage.AdmissionAdapterID,
		CandidateRunID:        stage.CandidateRunID,
		CandidateTextHash:     stage.CandidateTextHash,
		TurnTextHash:          stage.TurnTextHash,
		StageState:            stage.StageState,
		StageAction:           stage.StageAction,
	})
	if h == "" {
		return ""
	}
	return "stage-" + h
}

func recordAdmissionLiveRouteTurnCandidateAdmissionLiveStage(stage admissionLiveRouteTurnCandidateAdmissionLiveStage) error {
	path := strings.TrimSpace(os.Getenv("AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_LIVE_STAGE_LOG"))
	if path == "" {
		return nil
	}
	f, err := os.OpenFile(path, os.O_CREATE|os.O_APPEND|os.O_WRONLY, 0600)
	if err != nil {
		return err
	}
	enc := json.NewEncoder(f)
	err = enc.Encode(stage)
	if closeErr := f.Close(); err == nil {
		err = closeErr
	}
	return err
}

func admissionLiveRouteTurnCandidateAdmissionWriterPreflightForLiveStage(stage admissionLiveRouteTurnCandidateAdmissionLiveStage) admissionLiveRouteTurnCandidateAdmissionWriterPreflight {
	preflight := admissionLiveRouteTurnCandidateAdmissionWriterPreflight{
		Schema:                admissionLiveRouteTurnCandidateAdmissionWriterPreflightSchema,
		Timing:                "live_admission_writer_preflight",
		WriterState:           "blocked",
		WriterAction:          "reject",
		RollbackState:         "blocked",
		RollbackAction:        "reject",
		PromptClass:           stage.PromptClass,
		Route:                 stage.Route,
		Source:                stage.Source,
		ExpectedSource:        stage.ExpectedSource,
		CandidateRunID:        stage.CandidateRunID,
		CandidateDraftID:      stage.CandidateDraftID,
		CandidateExecutionID:  stage.CandidateExecutionID,
		GeneratorAdapterID:    stage.GeneratorAdapterID,
		HandoffID:             stage.HandoffID,
		AdmissionAdapterID:    stage.AdmissionAdapterID,
		AdmissionDecisionID:   stage.AdmissionDecisionID,
		AdmissionPromotionID:  stage.AdmissionPromotionID,
		AdmissionSwitchID:     stage.AdmissionSwitchID,
		AdmissionEnableGateID: stage.AdmissionEnableGateID,
		AdmissionLiveStageID:  stage.LiveStageID,
		AdmissionDecision:     stage.AdmissionDecision,
		AdmissionPromotion:    stage.AdmissionPromotion,
		SwitchState:           stage.SwitchState,
		SwitchAction:          stage.SwitchAction,
		EnableState:           stage.EnableState,
		EnableAction:          stage.EnableAction,
		StageState:            stage.StageState,
		StageAction:           stage.StageAction,
		DreamCandidateRunID:   stage.DreamCandidateRunID,
		CandidateTextStatus:   stage.CandidateTextStatus,
		CandidateTextHash:     stage.CandidateTextHash,
		AdmissionPolicyPassed: stage.AdmissionPolicyPassed,
		LiveRouteChoicePassed: stage.LiveRouteChoicePassed,
		SourceDecisionPassed:  stage.SourceDecisionPassed,
		SourcePromotionPassed: stage.SourcePromotionPassed,
		SourceSwitchPassed:    stage.SourceSwitchPassed,
		SourceEnablePassed:    stage.SourceEnablePassed,
		SourceStagePassed:     stage.Passed,
		LiveReady:             stage.LiveReady,
		LiveAdmissionEnabled:  stage.LiveAdmissionEnabled,
		AdmissionAllowed:      stage.AdmissionAllowed,
		ManualEnableRequested: stage.ManualEnableRequested,
		EnableKeyMatched:      stage.EnableKeyMatched,
		RequiresWriter:        stage.RequiresWriter,
		WriterReady:           stage.WriterReady,
		RequiresRollback:      stage.RequiresRollback,
		RollbackReady:         stage.RollbackReady,
		WriteAllowed:          false,
		MutatesState:          false,
		TurnTextHash:          stage.TurnTextHash,
	}
	if stage.Schema == "" {
		preflight.Reason = "missing_candidate_admission_live_stage"
		return preflight
	}
	if stage.Schema != admissionLiveRouteTurnCandidateAdmissionLiveStageSchema {
		preflight.Reason = "unexpected_candidate_admission_live_stage_schema " + stage.Schema
		return preflight
	}
	if !stage.Passed {
		preflight.Reason = "candidate_admission_live_stage_failed"
		if stage.Reason != "" {
			preflight.Reason += ": " + stage.Reason
		}
		return preflight
	}
	if stage.StageState != "staged_dry_run" {
		preflight.Reason = "candidate_admission_live_stage_not_staged"
		return preflight
	}
	if stage.StageAction != "stage_live_candidate_dry_run" {
		preflight.Reason = "candidate_admission_live_stage_unexpected_action"
		return preflight
	}
	if stage.LiveStageID == "" {
		preflight.Reason = "missing_candidate_admission_live_stage_id"
		return preflight
	}
	if wantStageID := admissionLiveRouteTurnCandidateAdmissionLiveStageID(stage); wantStageID == "" || stage.LiveStageID != wantStageID {
		preflight.Reason = "candidate_admission_live_stage_id_mismatch"
		return preflight
	}
	if !stage.LiveReady {
		preflight.Reason = "candidate_admission_live_stage_not_live_ready"
		return preflight
	}
	if stage.LiveAdmissionEnabled {
		preflight.Reason = "candidate_admission_live_stage_already_live_enabled"
		return preflight
	}
	if stage.AdmissionAllowed {
		preflight.Reason = "candidate_admission_live_stage_already_allows_admission"
		return preflight
	}
	if !stage.ManualEnableRequested {
		preflight.Reason = "candidate_admission_live_stage_missing_manual_enable"
		return preflight
	}
	if !stage.EnableKeyMatched {
		preflight.Reason = "candidate_admission_live_stage_key_not_matched"
		return preflight
	}
	if !stage.RequiresWriter {
		preflight.Reason = "candidate_admission_live_stage_does_not_require_writer"
		return preflight
	}
	if stage.WriterReady {
		preflight.Reason = "candidate_admission_live_stage_writer_already_ready"
		return preflight
	}
	if !stage.RequiresRollback {
		preflight.Reason = "candidate_admission_live_stage_does_not_require_rollback"
		return preflight
	}
	if stage.RollbackReady {
		preflight.Reason = "candidate_admission_live_stage_rollback_already_ready"
		return preflight
	}
	if stage.MutatesState {
		preflight.Reason = "candidate_admission_live_stage_already_mutates_state"
		return preflight
	}
	if !stage.SourceEnablePassed {
		preflight.Reason = "candidate_admission_live_stage_source_enable_not_passed"
		return preflight
	}
	if !stage.SourceSwitchPassed {
		preflight.Reason = "candidate_admission_live_stage_source_switch_not_passed"
		return preflight
	}
	if !stage.SourcePromotionPassed {
		preflight.Reason = "candidate_admission_live_stage_source_promotion_not_passed"
		return preflight
	}
	if !stage.SourceDecisionPassed {
		preflight.Reason = "candidate_admission_live_stage_source_decision_not_passed"
		return preflight
	}
	if !stage.AdmissionPolicyPassed {
		preflight.Reason = "candidate_admission_live_stage_policy_not_passed"
		return preflight
	}
	if !stage.LiveRouteChoicePassed {
		preflight.Reason = "candidate_admission_live_stage_live_route_not_passed"
		return preflight
	}
	if stage.AdmissionEnableGateID == "" ||
		stage.AdmissionSwitchID == "" ||
		stage.AdmissionPromotionID == "" ||
		stage.AdmissionDecisionID == "" ||
		stage.AdmissionAdapterID == "" ||
		stage.CandidateRunID == "" ||
		stage.CandidateDraftID == "" ||
		stage.CandidateExecutionID == "" ||
		stage.GeneratorAdapterID == "" ||
		stage.HandoffID == "" ||
		stage.DreamCandidateRunID == "" ||
		stage.CandidateTextHash == "" ||
		stage.TurnTextHash == "" {
		preflight.Reason = "candidate_admission_live_stage_missing_provenance"
		return preflight
	}
	preflight.WriterState = "absent"
	preflight.WriterAction = "require_writer_contract"
	preflight.RollbackState = "absent"
	preflight.RollbackAction = "require_rollback_contract"
	preflight.WriterPreflightID = admissionLiveRouteTurnCandidateAdmissionWriterPreflightID(preflight)
	if preflight.WriterPreflightID == "" {
		preflight.Reason = "missing_candidate_admission_writer_preflight_id"
		return preflight
	}
	preflight.Passed = true
	preflight.Reason = "writer and rollback absent; live admission remains staged only"
	return preflight
}

func admissionLiveRouteTurnCandidateAdmissionWriterPreflightID(preflight admissionLiveRouteTurnCandidateAdmissionWriterPreflight) string {
	h := hashJSON(struct {
		AdmissionLiveStageID  string `json:"admission_live_stage_id"`
		AdmissionEnableGateID string `json:"admission_enable_gate_id"`
		AdmissionSwitchID     string `json:"admission_switch_id"`
		AdmissionPromotionID  string `json:"admission_promotion_id"`
		AdmissionDecisionID   string `json:"admission_decision_id"`
		AdmissionAdapterID    string `json:"admission_adapter_id"`
		CandidateRunID        string `json:"candidate_run_id"`
		CandidateTextHash     string `json:"candidate_text_hash"`
		TurnTextHash          string `json:"turn_text_hash"`
		WriterState           string `json:"writer_state"`
		WriterAction          string `json:"writer_action"`
		RollbackState         string `json:"rollback_state"`
		RollbackAction        string `json:"rollback_action"`
		WriteAllowed          bool   `json:"write_allowed"`
	}{
		AdmissionLiveStageID:  preflight.AdmissionLiveStageID,
		AdmissionEnableGateID: preflight.AdmissionEnableGateID,
		AdmissionSwitchID:     preflight.AdmissionSwitchID,
		AdmissionPromotionID:  preflight.AdmissionPromotionID,
		AdmissionDecisionID:   preflight.AdmissionDecisionID,
		AdmissionAdapterID:    preflight.AdmissionAdapterID,
		CandidateRunID:        preflight.CandidateRunID,
		CandidateTextHash:     preflight.CandidateTextHash,
		TurnTextHash:          preflight.TurnTextHash,
		WriterState:           preflight.WriterState,
		WriterAction:          preflight.WriterAction,
		RollbackState:         preflight.RollbackState,
		RollbackAction:        preflight.RollbackAction,
		WriteAllowed:          preflight.WriteAllowed,
	})
	if h == "" {
		return ""
	}
	return "writer-" + h
}

func recordAdmissionLiveRouteTurnCandidateAdmissionWriterPreflight(preflight admissionLiveRouteTurnCandidateAdmissionWriterPreflight) error {
	path := strings.TrimSpace(os.Getenv("AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_WRITER_PREFLIGHT_LOG"))
	if path == "" {
		return nil
	}
	f, err := os.OpenFile(path, os.O_CREATE|os.O_APPEND|os.O_WRONLY, 0600)
	if err != nil {
		return err
	}
	enc := json.NewEncoder(f)
	err = enc.Encode(preflight)
	if closeErr := f.Close(); err == nil {
		err = closeErr
	}
	return err
}

func admissionLiveRouteTurnCandidateAdmissionWriterInventoryForPreflight(preflight admissionLiveRouteTurnCandidateAdmissionWriterPreflight) admissionLiveRouteTurnCandidateAdmissionWriterInventory {
	inventory := admissionLiveRouteTurnCandidateAdmissionWriterInventory{
		Schema:                      admissionLiveRouteTurnCandidateAdmissionWriterInventorySchema,
		Timing:                      "live_admission_writer_inventory",
		InventoryState:              "blocked",
		InventoryAction:             "reject",
		WriterState:                 preflight.WriterState,
		WriterAction:                preflight.WriterAction,
		RollbackState:               preflight.RollbackState,
		RollbackAction:              preflight.RollbackAction,
		PromptClass:                 preflight.PromptClass,
		Route:                       preflight.Route,
		Source:                      preflight.Source,
		ExpectedSource:              preflight.ExpectedSource,
		CandidateRunID:              preflight.CandidateRunID,
		CandidateDraftID:            preflight.CandidateDraftID,
		CandidateExecutionID:        preflight.CandidateExecutionID,
		GeneratorAdapterID:          preflight.GeneratorAdapterID,
		HandoffID:                   preflight.HandoffID,
		AdmissionAdapterID:          preflight.AdmissionAdapterID,
		AdmissionDecisionID:         preflight.AdmissionDecisionID,
		AdmissionPromotionID:        preflight.AdmissionPromotionID,
		AdmissionSwitchID:           preflight.AdmissionSwitchID,
		AdmissionEnableGateID:       preflight.AdmissionEnableGateID,
		AdmissionLiveStageID:        preflight.AdmissionLiveStageID,
		AdmissionWriterPreflightID:  preflight.WriterPreflightID,
		AdmissionDecision:           preflight.AdmissionDecision,
		AdmissionPromotion:          preflight.AdmissionPromotion,
		SwitchState:                 preflight.SwitchState,
		SwitchAction:                preflight.SwitchAction,
		EnableState:                 preflight.EnableState,
		EnableAction:                preflight.EnableAction,
		StageState:                  preflight.StageState,
		StageAction:                 preflight.StageAction,
		DreamCandidateRunID:         preflight.DreamCandidateRunID,
		CandidateTextStatus:         preflight.CandidateTextStatus,
		CandidateTextHash:           preflight.CandidateTextHash,
		AdmissionPolicyPassed:       preflight.AdmissionPolicyPassed,
		LiveRouteChoicePassed:       preflight.LiveRouteChoicePassed,
		SourceDecisionPassed:        preflight.SourceDecisionPassed,
		SourcePromotionPassed:       preflight.SourcePromotionPassed,
		SourceSwitchPassed:          preflight.SourceSwitchPassed,
		SourceEnablePassed:          preflight.SourceEnablePassed,
		SourceStagePassed:           preflight.SourceStagePassed,
		SourceWriterPreflightPassed: preflight.Passed,
		LiveReady:                   preflight.LiveReady,
		LiveAdmissionEnabled:        preflight.LiveAdmissionEnabled,
		AdmissionAllowed:            preflight.AdmissionAllowed,
		ManualEnableRequested:       preflight.ManualEnableRequested,
		EnableKeyMatched:            preflight.EnableKeyMatched,
		RequiresWriter:              preflight.RequiresWriter,
		WriterReady:                 preflight.WriterReady,
		RequiresRollback:            preflight.RequiresRollback,
		RollbackReady:               preflight.RollbackReady,
		WriteAllowed:                false,
		MutatesState:                false,
		TurnTextHash:                preflight.TurnTextHash,
	}
	if preflight.Schema == "" {
		inventory.Reason = "missing_candidate_admission_writer_preflight"
		return inventory
	}
	if preflight.Schema != admissionLiveRouteTurnCandidateAdmissionWriterPreflightSchema {
		inventory.Reason = "unexpected_candidate_admission_writer_preflight_schema " + preflight.Schema
		return inventory
	}
	if !preflight.Passed {
		inventory.Reason = "candidate_admission_writer_preflight_failed"
		if preflight.Reason != "" {
			inventory.Reason += ": " + preflight.Reason
		}
		return inventory
	}
	if preflight.WriterState != "absent" {
		inventory.Reason = "candidate_admission_writer_preflight_unexpected_writer_state"
		return inventory
	}
	if preflight.WriterAction != "require_writer_contract" {
		inventory.Reason = "candidate_admission_writer_preflight_unexpected_writer_action"
		return inventory
	}
	if preflight.RollbackState != "absent" {
		inventory.Reason = "candidate_admission_writer_preflight_unexpected_rollback_state"
		return inventory
	}
	if preflight.RollbackAction != "require_rollback_contract" {
		inventory.Reason = "candidate_admission_writer_preflight_unexpected_rollback_action"
		return inventory
	}
	if preflight.WriterPreflightID == "" {
		inventory.Reason = "missing_candidate_admission_writer_preflight_id"
		return inventory
	}
	if wantPreflightID := admissionLiveRouteTurnCandidateAdmissionWriterPreflightID(preflight); wantPreflightID == "" || preflight.WriterPreflightID != wantPreflightID {
		inventory.Reason = "candidate_admission_writer_preflight_id_mismatch"
		return inventory
	}
	if !preflight.LiveReady {
		inventory.Reason = "candidate_admission_writer_preflight_not_live_ready"
		return inventory
	}
	if preflight.LiveAdmissionEnabled {
		inventory.Reason = "candidate_admission_writer_preflight_already_live_enabled"
		return inventory
	}
	if preflight.AdmissionAllowed {
		inventory.Reason = "candidate_admission_writer_preflight_already_allows_admission"
		return inventory
	}
	if !preflight.ManualEnableRequested {
		inventory.Reason = "candidate_admission_writer_preflight_missing_manual_enable"
		return inventory
	}
	if !preflight.EnableKeyMatched {
		inventory.Reason = "candidate_admission_writer_preflight_key_not_matched"
		return inventory
	}
	if !preflight.RequiresWriter {
		inventory.Reason = "candidate_admission_writer_preflight_does_not_require_writer"
		return inventory
	}
	if preflight.WriterReady {
		inventory.Reason = "candidate_admission_writer_preflight_writer_already_ready"
		return inventory
	}
	if !preflight.RequiresRollback {
		inventory.Reason = "candidate_admission_writer_preflight_does_not_require_rollback"
		return inventory
	}
	if preflight.RollbackReady {
		inventory.Reason = "candidate_admission_writer_preflight_rollback_already_ready"
		return inventory
	}
	if preflight.WriteAllowed {
		inventory.Reason = "candidate_admission_writer_preflight_already_allows_write"
		return inventory
	}
	if preflight.MutatesState {
		inventory.Reason = "candidate_admission_writer_preflight_already_mutates_state"
		return inventory
	}
	if preflight.StageState != "staged_dry_run" {
		inventory.Reason = "candidate_admission_writer_preflight_stage_not_staged"
		return inventory
	}
	if preflight.StageAction != "stage_live_candidate_dry_run" {
		inventory.Reason = "candidate_admission_writer_preflight_unexpected_stage_action"
		return inventory
	}
	if !preflight.SourceStagePassed {
		inventory.Reason = "candidate_admission_writer_preflight_source_stage_not_passed"
		return inventory
	}
	if !preflight.SourceEnablePassed {
		inventory.Reason = "candidate_admission_writer_preflight_source_enable_not_passed"
		return inventory
	}
	if !preflight.SourceSwitchPassed {
		inventory.Reason = "candidate_admission_writer_preflight_source_switch_not_passed"
		return inventory
	}
	if !preflight.SourcePromotionPassed {
		inventory.Reason = "candidate_admission_writer_preflight_source_promotion_not_passed"
		return inventory
	}
	if !preflight.SourceDecisionPassed {
		inventory.Reason = "candidate_admission_writer_preflight_source_decision_not_passed"
		return inventory
	}
	if !preflight.AdmissionPolicyPassed {
		inventory.Reason = "candidate_admission_writer_preflight_policy_not_passed"
		return inventory
	}
	if !preflight.LiveRouteChoicePassed {
		inventory.Reason = "candidate_admission_writer_preflight_live_route_not_passed"
		return inventory
	}
	if preflight.AdmissionLiveStageID == "" ||
		preflight.AdmissionEnableGateID == "" ||
		preflight.AdmissionSwitchID == "" ||
		preflight.AdmissionPromotionID == "" ||
		preflight.AdmissionDecisionID == "" ||
		preflight.AdmissionAdapterID == "" ||
		preflight.CandidateRunID == "" ||
		preflight.CandidateDraftID == "" ||
		preflight.CandidateExecutionID == "" ||
		preflight.GeneratorAdapterID == "" ||
		preflight.HandoffID == "" ||
		preflight.DreamCandidateRunID == "" ||
		preflight.CandidateTextHash == "" ||
		preflight.TurnTextHash == "" {
		inventory.Reason = "candidate_admission_writer_preflight_missing_provenance"
		return inventory
	}
	inventory.InventoryState = "contracts_absent"
	inventory.InventoryAction = "name_required_contracts"
	inventory.WriterContract = "live_admission_writer.v1"
	inventory.RollbackContract = "live_admission_rollback.v1"
	inventory.AdmissionLedgerContract = "live_admission_ledger.v1"
	inventory.WriterContractPresent = false
	inventory.RollbackContractPresent = false
	inventory.LedgerContractPresent = false
	inventory.ContractsReady = false
	inventory.WriterInventoryID = admissionLiveRouteTurnCandidateAdmissionWriterInventoryID(inventory)
	if inventory.WriterInventoryID == "" {
		inventory.Reason = "missing_candidate_admission_writer_inventory_id"
		return inventory
	}
	inventory.Passed = true
	inventory.Reason = "writer inventory recorded required contracts; live admission remains blocked"
	return inventory
}

func admissionLiveRouteTurnCandidateAdmissionWriterInventoryID(inventory admissionLiveRouteTurnCandidateAdmissionWriterInventory) string {
	h := hashJSON(struct {
		AdmissionWriterPreflightID string `json:"admission_writer_preflight_id"`
		AdmissionLiveStageID       string `json:"admission_live_stage_id"`
		AdmissionEnableGateID      string `json:"admission_enable_gate_id"`
		AdmissionSwitchID          string `json:"admission_switch_id"`
		AdmissionPromotionID       string `json:"admission_promotion_id"`
		AdmissionDecisionID        string `json:"admission_decision_id"`
		AdmissionAdapterID         string `json:"admission_adapter_id"`
		CandidateRunID             string `json:"candidate_run_id"`
		CandidateTextHash          string `json:"candidate_text_hash"`
		TurnTextHash               string `json:"turn_text_hash"`
		InventoryState             string `json:"inventory_state"`
		InventoryAction            string `json:"inventory_action"`
		WriterContract             string `json:"writer_contract"`
		RollbackContract           string `json:"rollback_contract"`
		AdmissionLedgerContract    string `json:"admission_ledger_contract"`
		ContractsReady             bool   `json:"contracts_ready"`
		WriteAllowed               bool   `json:"write_allowed"`
	}{
		AdmissionWriterPreflightID: inventory.AdmissionWriterPreflightID,
		AdmissionLiveStageID:       inventory.AdmissionLiveStageID,
		AdmissionEnableGateID:      inventory.AdmissionEnableGateID,
		AdmissionSwitchID:          inventory.AdmissionSwitchID,
		AdmissionPromotionID:       inventory.AdmissionPromotionID,
		AdmissionDecisionID:        inventory.AdmissionDecisionID,
		AdmissionAdapterID:         inventory.AdmissionAdapterID,
		CandidateRunID:             inventory.CandidateRunID,
		CandidateTextHash:          inventory.CandidateTextHash,
		TurnTextHash:               inventory.TurnTextHash,
		InventoryState:             inventory.InventoryState,
		InventoryAction:            inventory.InventoryAction,
		WriterContract:             inventory.WriterContract,
		RollbackContract:           inventory.RollbackContract,
		AdmissionLedgerContract:    inventory.AdmissionLedgerContract,
		ContractsReady:             inventory.ContractsReady,
		WriteAllowed:               inventory.WriteAllowed,
	})
	if h == "" {
		return ""
	}
	return "writer-inventory-" + h
}

func recordAdmissionLiveRouteTurnCandidateAdmissionWriterInventory(inventory admissionLiveRouteTurnCandidateAdmissionWriterInventory) error {
	path := strings.TrimSpace(os.Getenv("AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_WRITER_INVENTORY_LOG"))
	if path == "" {
		return nil
	}
	f, err := os.OpenFile(path, os.O_CREATE|os.O_APPEND|os.O_WRONLY, 0600)
	if err != nil {
		return err
	}
	enc := json.NewEncoder(f)
	err = enc.Encode(inventory)
	if closeErr := f.Close(); err == nil {
		err = closeErr
	}
	return err
}

func admissionLiveRouteTurnCandidateAdmissionWriterContractForInventory(inventory admissionLiveRouteTurnCandidateAdmissionWriterInventory) admissionLiveRouteTurnCandidateAdmissionWriterContract {
	contract := admissionLiveRouteTurnCandidateAdmissionWriterContract{
		Schema:                        admissionLiveRouteTurnCandidateAdmissionWriterContractSchema,
		Timing:                        "live_admission_writer_contract",
		ContractState:                 "blocked",
		ContractAction:                "reject",
		WriterContract:                inventory.WriterContract,
		RollbackContract:              inventory.RollbackContract,
		AdmissionLedgerContract:       inventory.AdmissionLedgerContract,
		SourceWriterContractPresent:   inventory.WriterContractPresent,
		SourceRollbackContractPresent: inventory.RollbackContractPresent,
		SourceLedgerContractPresent:   inventory.LedgerContractPresent,
		ContractsReady:                false,
		WriterState:                   inventory.WriterState,
		WriterAction:                  inventory.WriterAction,
		RollbackState:                 inventory.RollbackState,
		RollbackAction:                inventory.RollbackAction,
		PromptClass:                   inventory.PromptClass,
		Route:                         inventory.Route,
		Source:                        inventory.Source,
		ExpectedSource:                inventory.ExpectedSource,
		CandidateRunID:                inventory.CandidateRunID,
		CandidateDraftID:              inventory.CandidateDraftID,
		CandidateExecutionID:          inventory.CandidateExecutionID,
		GeneratorAdapterID:            inventory.GeneratorAdapterID,
		HandoffID:                     inventory.HandoffID,
		AdmissionAdapterID:            inventory.AdmissionAdapterID,
		AdmissionDecisionID:           inventory.AdmissionDecisionID,
		AdmissionPromotionID:          inventory.AdmissionPromotionID,
		AdmissionSwitchID:             inventory.AdmissionSwitchID,
		AdmissionEnableGateID:         inventory.AdmissionEnableGateID,
		AdmissionLiveStageID:          inventory.AdmissionLiveStageID,
		AdmissionWriterPreflightID:    inventory.AdmissionWriterPreflightID,
		AdmissionWriterInventoryID:    inventory.WriterInventoryID,
		AdmissionDecision:             inventory.AdmissionDecision,
		AdmissionPromotion:            inventory.AdmissionPromotion,
		SwitchState:                   inventory.SwitchState,
		SwitchAction:                  inventory.SwitchAction,
		EnableState:                   inventory.EnableState,
		EnableAction:                  inventory.EnableAction,
		StageState:                    inventory.StageState,
		StageAction:                   inventory.StageAction,
		InventoryState:                inventory.InventoryState,
		InventoryAction:               inventory.InventoryAction,
		DreamCandidateRunID:           inventory.DreamCandidateRunID,
		CandidateTextStatus:           inventory.CandidateTextStatus,
		CandidateTextHash:             inventory.CandidateTextHash,
		AdmissionPolicyPassed:         inventory.AdmissionPolicyPassed,
		LiveRouteChoicePassed:         inventory.LiveRouteChoicePassed,
		SourceDecisionPassed:          inventory.SourceDecisionPassed,
		SourcePromotionPassed:         inventory.SourcePromotionPassed,
		SourceSwitchPassed:            inventory.SourceSwitchPassed,
		SourceEnablePassed:            inventory.SourceEnablePassed,
		SourceStagePassed:             inventory.SourceStagePassed,
		SourceWriterPreflightPassed:   inventory.SourceWriterPreflightPassed,
		SourceWriterInventoryPassed:   inventory.Passed,
		LiveReady:                     inventory.LiveReady,
		LiveAdmissionEnabled:          inventory.LiveAdmissionEnabled,
		AdmissionAllowed:              inventory.AdmissionAllowed,
		ManualEnableRequested:         inventory.ManualEnableRequested,
		EnableKeyMatched:              inventory.EnableKeyMatched,
		RequiresWriter:                inventory.RequiresWriter,
		WriterReady:                   inventory.WriterReady,
		RequiresRollback:              inventory.RequiresRollback,
		RollbackReady:                 inventory.RollbackReady,
		WriteAllowed:                  false,
		MutatesState:                  false,
		TurnTextHash:                  inventory.TurnTextHash,
	}
	if inventory.Schema == "" {
		contract.Reason = "missing_candidate_admission_writer_inventory"
		return contract
	}
	if inventory.Schema != admissionLiveRouteTurnCandidateAdmissionWriterInventorySchema {
		contract.Reason = "unexpected_candidate_admission_writer_inventory_schema " + inventory.Schema
		return contract
	}
	if !inventory.Passed {
		contract.Reason = "candidate_admission_writer_inventory_failed"
		if inventory.Reason != "" {
			contract.Reason += ": " + inventory.Reason
		}
		return contract
	}
	if inventory.InventoryState != "contracts_absent" {
		contract.Reason = "candidate_admission_writer_inventory_unexpected_state"
		return contract
	}
	if inventory.InventoryAction != "name_required_contracts" {
		contract.Reason = "candidate_admission_writer_inventory_unexpected_action"
		return contract
	}
	if inventory.WriterInventoryID == "" {
		contract.Reason = "missing_candidate_admission_writer_inventory_id"
		return contract
	}
	if wantInventoryID := admissionLiveRouteTurnCandidateAdmissionWriterInventoryID(inventory); wantInventoryID == "" || inventory.WriterInventoryID != wantInventoryID {
		contract.Reason = "candidate_admission_writer_inventory_id_mismatch"
		return contract
	}
	if inventory.WriterContract != "live_admission_writer.v1" {
		contract.Reason = "candidate_admission_writer_inventory_unexpected_writer_contract"
		return contract
	}
	if inventory.RollbackContract != "live_admission_rollback.v1" {
		contract.Reason = "candidate_admission_writer_inventory_unexpected_rollback_contract"
		return contract
	}
	if inventory.AdmissionLedgerContract != "live_admission_ledger.v1" {
		contract.Reason = "candidate_admission_writer_inventory_unexpected_ledger_contract"
		return contract
	}
	if inventory.WriterContractPresent {
		contract.Reason = "candidate_admission_writer_inventory_writer_contract_already_present"
		return contract
	}
	if inventory.RollbackContractPresent {
		contract.Reason = "candidate_admission_writer_inventory_rollback_contract_already_present"
		return contract
	}
	if inventory.LedgerContractPresent {
		contract.Reason = "candidate_admission_writer_inventory_ledger_contract_already_present"
		return contract
	}
	if inventory.ContractsReady {
		contract.Reason = "candidate_admission_writer_inventory_contracts_already_ready"
		return contract
	}
	if !inventory.LiveReady {
		contract.Reason = "candidate_admission_writer_inventory_not_live_ready"
		return contract
	}
	if inventory.LiveAdmissionEnabled {
		contract.Reason = "candidate_admission_writer_inventory_already_live_enabled"
		return contract
	}
	if inventory.AdmissionAllowed {
		contract.Reason = "candidate_admission_writer_inventory_already_allows_admission"
		return contract
	}
	if !inventory.ManualEnableRequested {
		contract.Reason = "candidate_admission_writer_inventory_missing_manual_enable"
		return contract
	}
	if !inventory.EnableKeyMatched {
		contract.Reason = "candidate_admission_writer_inventory_key_not_matched"
		return contract
	}
	if !inventory.RequiresWriter {
		contract.Reason = "candidate_admission_writer_inventory_does_not_require_writer"
		return contract
	}
	if inventory.WriterReady {
		contract.Reason = "candidate_admission_writer_inventory_writer_already_ready"
		return contract
	}
	if !inventory.RequiresRollback {
		contract.Reason = "candidate_admission_writer_inventory_does_not_require_rollback"
		return contract
	}
	if inventory.RollbackReady {
		contract.Reason = "candidate_admission_writer_inventory_rollback_already_ready"
		return contract
	}
	if inventory.WriteAllowed {
		contract.Reason = "candidate_admission_writer_inventory_already_allows_write"
		return contract
	}
	if inventory.MutatesState {
		contract.Reason = "candidate_admission_writer_inventory_already_mutates_state"
		return contract
	}
	if inventory.StageState != "staged_dry_run" {
		contract.Reason = "candidate_admission_writer_inventory_stage_not_staged"
		return contract
	}
	if inventory.StageAction != "stage_live_candidate_dry_run" {
		contract.Reason = "candidate_admission_writer_inventory_unexpected_stage_action"
		return contract
	}
	if !inventory.SourceWriterPreflightPassed {
		contract.Reason = "candidate_admission_writer_inventory_source_preflight_not_passed"
		return contract
	}
	if !inventory.SourceStagePassed {
		contract.Reason = "candidate_admission_writer_inventory_source_stage_not_passed"
		return contract
	}
	if !inventory.SourceEnablePassed {
		contract.Reason = "candidate_admission_writer_inventory_source_enable_not_passed"
		return contract
	}
	if !inventory.SourceSwitchPassed {
		contract.Reason = "candidate_admission_writer_inventory_source_switch_not_passed"
		return contract
	}
	if !inventory.SourcePromotionPassed {
		contract.Reason = "candidate_admission_writer_inventory_source_promotion_not_passed"
		return contract
	}
	if !inventory.SourceDecisionPassed {
		contract.Reason = "candidate_admission_writer_inventory_source_decision_not_passed"
		return contract
	}
	if !inventory.AdmissionPolicyPassed {
		contract.Reason = "candidate_admission_writer_inventory_policy_not_passed"
		return contract
	}
	if !inventory.LiveRouteChoicePassed {
		contract.Reason = "candidate_admission_writer_inventory_live_route_not_passed"
		return contract
	}
	if inventory.AdmissionWriterPreflightID == "" ||
		inventory.AdmissionLiveStageID == "" ||
		inventory.AdmissionEnableGateID == "" ||
		inventory.AdmissionSwitchID == "" ||
		inventory.AdmissionPromotionID == "" ||
		inventory.AdmissionDecisionID == "" ||
		inventory.AdmissionAdapterID == "" ||
		inventory.CandidateRunID == "" ||
		inventory.CandidateDraftID == "" ||
		inventory.CandidateExecutionID == "" ||
		inventory.GeneratorAdapterID == "" ||
		inventory.HandoffID == "" ||
		inventory.DreamCandidateRunID == "" ||
		inventory.CandidateTextHash == "" ||
		inventory.TurnTextHash == "" {
		contract.Reason = "candidate_admission_writer_inventory_missing_provenance"
		return contract
	}
	contract.ContractState = "shape_drafted_dry_run"
	contract.ContractAction = "define_writer_rollback_ledger_contract"
	contract.WriterContractShape = "append_shadow_candidate_receipt"
	contract.RollbackContractShape = "remove_exact_writer_receipt"
	contract.LedgerContractShape = "append_only_receipt_log"
	contract.WriteScope = "dream_candidate_admission"
	contract.RollbackScope = "single_writer_receipt"
	contract.LedgerMode = "append_only_dry_run"
	contract.ContractShapeReady = true
	contract.WriterImplementationReady = false
	contract.RollbackImplementationReady = false
	contract.LedgerImplementationReady = false
	contract.ContractsReady = false
	contract.WriterContractID = admissionLiveRouteTurnCandidateAdmissionWriterContractID(contract)
	if contract.WriterContractID == "" {
		contract.Reason = "missing_candidate_admission_writer_contract_id"
		return contract
	}
	contract.Passed = true
	contract.Reason = "writer contract shape drafted; implementation and ledger remain absent"
	return contract
}

func admissionLiveRouteTurnCandidateAdmissionWriterContractID(contract admissionLiveRouteTurnCandidateAdmissionWriterContract) string {
	h := hashJSON(struct {
		AdmissionWriterInventoryID string `json:"admission_writer_inventory_id"`
		AdmissionWriterPreflightID string `json:"admission_writer_preflight_id"`
		AdmissionLiveStageID       string `json:"admission_live_stage_id"`
		AdmissionEnableGateID      string `json:"admission_enable_gate_id"`
		AdmissionSwitchID          string `json:"admission_switch_id"`
		AdmissionPromotionID       string `json:"admission_promotion_id"`
		AdmissionDecisionID        string `json:"admission_decision_id"`
		AdmissionAdapterID         string `json:"admission_adapter_id"`
		CandidateRunID             string `json:"candidate_run_id"`
		CandidateTextHash          string `json:"candidate_text_hash"`
		TurnTextHash               string `json:"turn_text_hash"`
		ContractState              string `json:"contract_state"`
		ContractAction             string `json:"contract_action"`
		WriterContract             string `json:"writer_contract"`
		RollbackContract           string `json:"rollback_contract"`
		AdmissionLedgerContract    string `json:"admission_ledger_contract"`
		WriterContractShape        string `json:"writer_contract_shape"`
		RollbackContractShape      string `json:"rollback_contract_shape"`
		LedgerContractShape        string `json:"ledger_contract_shape"`
		WriteScope                 string `json:"write_scope"`
		RollbackScope              string `json:"rollback_scope"`
		LedgerMode                 string `json:"ledger_mode"`
		ContractShapeReady         bool   `json:"contract_shape_ready"`
		ContractsReady             bool   `json:"contracts_ready"`
		WriteAllowed               bool   `json:"write_allowed"`
	}{
		AdmissionWriterInventoryID: contract.AdmissionWriterInventoryID,
		AdmissionWriterPreflightID: contract.AdmissionWriterPreflightID,
		AdmissionLiveStageID:       contract.AdmissionLiveStageID,
		AdmissionEnableGateID:      contract.AdmissionEnableGateID,
		AdmissionSwitchID:          contract.AdmissionSwitchID,
		AdmissionPromotionID:       contract.AdmissionPromotionID,
		AdmissionDecisionID:        contract.AdmissionDecisionID,
		AdmissionAdapterID:         contract.AdmissionAdapterID,
		CandidateRunID:             contract.CandidateRunID,
		CandidateTextHash:          contract.CandidateTextHash,
		TurnTextHash:               contract.TurnTextHash,
		ContractState:              contract.ContractState,
		ContractAction:             contract.ContractAction,
		WriterContract:             contract.WriterContract,
		RollbackContract:           contract.RollbackContract,
		AdmissionLedgerContract:    contract.AdmissionLedgerContract,
		WriterContractShape:        contract.WriterContractShape,
		RollbackContractShape:      contract.RollbackContractShape,
		LedgerContractShape:        contract.LedgerContractShape,
		WriteScope:                 contract.WriteScope,
		RollbackScope:              contract.RollbackScope,
		LedgerMode:                 contract.LedgerMode,
		ContractShapeReady:         contract.ContractShapeReady,
		ContractsReady:             contract.ContractsReady,
		WriteAllowed:               contract.WriteAllowed,
	})
	if h == "" {
		return ""
	}
	return "writer-contract-" + h
}

func recordAdmissionLiveRouteTurnCandidateAdmissionWriterContract(contract admissionLiveRouteTurnCandidateAdmissionWriterContract) error {
	path := strings.TrimSpace(os.Getenv("AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_WRITER_CONTRACT_LOG"))
	if path == "" {
		return nil
	}
	f, err := os.OpenFile(path, os.O_CREATE|os.O_APPEND|os.O_WRONLY, 0600)
	if err != nil {
		return err
	}
	enc := json.NewEncoder(f)
	err = enc.Encode(contract)
	if closeErr := f.Close(); err == nil {
		err = closeErr
	}
	return err
}

func admissionLiveRouteTurnCandidateAdmissionLedgerForWriterContract(contract admissionLiveRouteTurnCandidateAdmissionWriterContract) admissionLiveRouteTurnCandidateAdmissionLedger {
	ledger := admissionLiveRouteTurnCandidateAdmissionLedger{
		Schema:                        admissionLiveRouteTurnCandidateAdmissionLedgerSchema,
		Timing:                        "live_admission_ledger",
		LedgerState:                   "blocked",
		LedgerAction:                  "reject",
		LedgerContract:                contract.AdmissionLedgerContract,
		LedgerMode:                    contract.LedgerMode,
		LedgerImplementationReady:     false,
		ContractState:                 contract.ContractState,
		ContractAction:                contract.ContractAction,
		WriterContract:                contract.WriterContract,
		RollbackContract:              contract.RollbackContract,
		AdmissionLedgerContract:       contract.AdmissionLedgerContract,
		WriterContractShape:           contract.WriterContractShape,
		RollbackContractShape:         contract.RollbackContractShape,
		LedgerContractShape:           contract.LedgerContractShape,
		WriteScope:                    contract.WriteScope,
		RollbackScope:                 contract.RollbackScope,
		ContractShapeReady:            contract.ContractShapeReady,
		SourceWriterContractPresent:   contract.SourceWriterContractPresent,
		SourceRollbackContractPresent: contract.SourceRollbackContractPresent,
		SourceLedgerContractPresent:   contract.SourceLedgerContractPresent,
		WriterImplementationReady:     contract.WriterImplementationReady,
		RollbackImplementationReady:   contract.RollbackImplementationReady,
		ContractsReady:                false,
		WriterState:                   contract.WriterState,
		WriterAction:                  contract.WriterAction,
		RollbackState:                 contract.RollbackState,
		RollbackAction:                contract.RollbackAction,
		PromptClass:                   contract.PromptClass,
		Route:                         contract.Route,
		Source:                        contract.Source,
		ExpectedSource:                contract.ExpectedSource,
		CandidateRunID:                contract.CandidateRunID,
		CandidateDraftID:              contract.CandidateDraftID,
		CandidateExecutionID:          contract.CandidateExecutionID,
		GeneratorAdapterID:            contract.GeneratorAdapterID,
		HandoffID:                     contract.HandoffID,
		AdmissionAdapterID:            contract.AdmissionAdapterID,
		AdmissionDecisionID:           contract.AdmissionDecisionID,
		AdmissionPromotionID:          contract.AdmissionPromotionID,
		AdmissionSwitchID:             contract.AdmissionSwitchID,
		AdmissionEnableGateID:         contract.AdmissionEnableGateID,
		AdmissionLiveStageID:          contract.AdmissionLiveStageID,
		AdmissionWriterPreflightID:    contract.AdmissionWriterPreflightID,
		AdmissionWriterInventoryID:    contract.AdmissionWriterInventoryID,
		AdmissionWriterContractID:     contract.WriterContractID,
		AdmissionDecision:             contract.AdmissionDecision,
		AdmissionPromotion:            contract.AdmissionPromotion,
		SwitchState:                   contract.SwitchState,
		SwitchAction:                  contract.SwitchAction,
		EnableState:                   contract.EnableState,
		EnableAction:                  contract.EnableAction,
		StageState:                    contract.StageState,
		StageAction:                   contract.StageAction,
		InventoryState:                contract.InventoryState,
		InventoryAction:               contract.InventoryAction,
		DreamCandidateRunID:           contract.DreamCandidateRunID,
		CandidateTextStatus:           contract.CandidateTextStatus,
		CandidateTextHash:             contract.CandidateTextHash,
		AdmissionPolicyPassed:         contract.AdmissionPolicyPassed,
		LiveRouteChoicePassed:         contract.LiveRouteChoicePassed,
		SourceDecisionPassed:          contract.SourceDecisionPassed,
		SourcePromotionPassed:         contract.SourcePromotionPassed,
		SourceSwitchPassed:            contract.SourceSwitchPassed,
		SourceEnablePassed:            contract.SourceEnablePassed,
		SourceStagePassed:             contract.SourceStagePassed,
		SourceWriterPreflightPassed:   contract.SourceWriterPreflightPassed,
		SourceWriterInventoryPassed:   contract.SourceWriterInventoryPassed,
		SourceWriterContractPassed:    contract.Passed,
		LiveReady:                     contract.LiveReady,
		LiveAdmissionEnabled:          contract.LiveAdmissionEnabled,
		AdmissionAllowed:              contract.AdmissionAllowed,
		ManualEnableRequested:         contract.ManualEnableRequested,
		EnableKeyMatched:              contract.EnableKeyMatched,
		RequiresWriter:                contract.RequiresWriter,
		WriterReady:                   contract.WriterReady,
		RequiresRollback:              contract.RequiresRollback,
		RollbackReady:                 contract.RollbackReady,
		WriteAllowed:                  false,
		MutatesState:                  false,
		TurnTextHash:                  contract.TurnTextHash,
	}
	if contract.Schema == "" {
		ledger.Reason = "missing_candidate_admission_writer_contract"
		return ledger
	}
	if contract.Schema != admissionLiveRouteTurnCandidateAdmissionWriterContractSchema {
		ledger.Reason = "unexpected_candidate_admission_writer_contract_schema " + contract.Schema
		return ledger
	}
	if !contract.Passed {
		ledger.Reason = "candidate_admission_writer_contract_failed"
		if contract.Reason != "" {
			ledger.Reason += ": " + contract.Reason
		}
		return ledger
	}
	if contract.WriterContractID == "" {
		ledger.Reason = "missing_candidate_admission_writer_contract_id"
		return ledger
	}
	if wantContractID := admissionLiveRouteTurnCandidateAdmissionWriterContractID(contract); wantContractID == "" || contract.WriterContractID != wantContractID {
		ledger.Reason = "candidate_admission_writer_contract_id_mismatch"
		return ledger
	}
	if contract.ContractState != "shape_drafted_dry_run" {
		ledger.Reason = "candidate_admission_writer_contract_unexpected_state"
		return ledger
	}
	if contract.ContractAction != "define_writer_rollback_ledger_contract" {
		ledger.Reason = "candidate_admission_writer_contract_unexpected_action"
		return ledger
	}
	if contract.WriterContract != "live_admission_writer.v1" {
		ledger.Reason = "candidate_admission_writer_contract_unexpected_writer_contract"
		return ledger
	}
	if contract.RollbackContract != "live_admission_rollback.v1" {
		ledger.Reason = "candidate_admission_writer_contract_unexpected_rollback_contract"
		return ledger
	}
	if contract.AdmissionLedgerContract != "live_admission_ledger.v1" {
		ledger.Reason = "candidate_admission_writer_contract_unexpected_ledger_contract"
		return ledger
	}
	if contract.WriterContractShape != "append_shadow_candidate_receipt" {
		ledger.Reason = "candidate_admission_writer_contract_unexpected_writer_shape"
		return ledger
	}
	if contract.RollbackContractShape != "remove_exact_writer_receipt" {
		ledger.Reason = "candidate_admission_writer_contract_unexpected_rollback_shape"
		return ledger
	}
	if contract.LedgerContractShape != "append_only_receipt_log" {
		ledger.Reason = "candidate_admission_writer_contract_unexpected_ledger_shape"
		return ledger
	}
	if contract.WriteScope != "dream_candidate_admission" {
		ledger.Reason = "candidate_admission_writer_contract_unexpected_write_scope"
		return ledger
	}
	if contract.RollbackScope != "single_writer_receipt" {
		ledger.Reason = "candidate_admission_writer_contract_unexpected_rollback_scope"
		return ledger
	}
	if contract.LedgerMode != "append_only_dry_run" {
		ledger.Reason = "candidate_admission_writer_contract_unexpected_ledger_mode"
		return ledger
	}
	if !contract.ContractShapeReady {
		ledger.Reason = "candidate_admission_writer_contract_shape_not_ready"
		return ledger
	}
	if contract.SourceWriterContractPresent {
		ledger.Reason = "candidate_admission_writer_contract_source_writer_present"
		return ledger
	}
	if contract.SourceRollbackContractPresent {
		ledger.Reason = "candidate_admission_writer_contract_source_rollback_present"
		return ledger
	}
	if contract.SourceLedgerContractPresent {
		ledger.Reason = "candidate_admission_writer_contract_source_ledger_present"
		return ledger
	}
	if contract.WriterImplementationReady {
		ledger.Reason = "candidate_admission_writer_contract_writer_impl_already_ready"
		return ledger
	}
	if contract.RollbackImplementationReady {
		ledger.Reason = "candidate_admission_writer_contract_rollback_impl_already_ready"
		return ledger
	}
	if contract.LedgerImplementationReady {
		ledger.Reason = "candidate_admission_writer_contract_ledger_impl_already_ready"
		return ledger
	}
	if contract.ContractsReady {
		ledger.Reason = "candidate_admission_writer_contract_contracts_already_ready"
		return ledger
	}
	if !contract.LiveReady {
		ledger.Reason = "candidate_admission_writer_contract_not_live_ready"
		return ledger
	}
	if contract.LiveAdmissionEnabled {
		ledger.Reason = "candidate_admission_writer_contract_already_live_enabled"
		return ledger
	}
	if contract.AdmissionAllowed {
		ledger.Reason = "candidate_admission_writer_contract_already_allows_admission"
		return ledger
	}
	if !contract.ManualEnableRequested {
		ledger.Reason = "candidate_admission_writer_contract_missing_manual_enable"
		return ledger
	}
	if !contract.EnableKeyMatched {
		ledger.Reason = "candidate_admission_writer_contract_key_not_matched"
		return ledger
	}
	if !contract.RequiresWriter {
		ledger.Reason = "candidate_admission_writer_contract_does_not_require_writer"
		return ledger
	}
	if contract.WriterReady {
		ledger.Reason = "candidate_admission_writer_contract_writer_already_ready"
		return ledger
	}
	if !contract.RequiresRollback {
		ledger.Reason = "candidate_admission_writer_contract_does_not_require_rollback"
		return ledger
	}
	if contract.RollbackReady {
		ledger.Reason = "candidate_admission_writer_contract_rollback_already_ready"
		return ledger
	}
	if contract.WriteAllowed {
		ledger.Reason = "candidate_admission_writer_contract_already_allows_write"
		return ledger
	}
	if contract.MutatesState {
		ledger.Reason = "candidate_admission_writer_contract_already_mutates_state"
		return ledger
	}
	if contract.StageState != "staged_dry_run" {
		ledger.Reason = "candidate_admission_writer_contract_stage_not_staged"
		return ledger
	}
	if contract.StageAction != "stage_live_candidate_dry_run" {
		ledger.Reason = "candidate_admission_writer_contract_unexpected_stage_action"
		return ledger
	}
	if !contract.SourceWriterInventoryPassed {
		ledger.Reason = "candidate_admission_writer_contract_source_inventory_not_passed"
		return ledger
	}
	if !contract.SourceWriterPreflightPassed {
		ledger.Reason = "candidate_admission_writer_contract_source_preflight_not_passed"
		return ledger
	}
	if !contract.SourceStagePassed {
		ledger.Reason = "candidate_admission_writer_contract_source_stage_not_passed"
		return ledger
	}
	if !contract.SourceEnablePassed {
		ledger.Reason = "candidate_admission_writer_contract_source_enable_not_passed"
		return ledger
	}
	if !contract.SourceSwitchPassed {
		ledger.Reason = "candidate_admission_writer_contract_source_switch_not_passed"
		return ledger
	}
	if !contract.SourcePromotionPassed {
		ledger.Reason = "candidate_admission_writer_contract_source_promotion_not_passed"
		return ledger
	}
	if !contract.SourceDecisionPassed {
		ledger.Reason = "candidate_admission_writer_contract_source_decision_not_passed"
		return ledger
	}
	if !contract.AdmissionPolicyPassed {
		ledger.Reason = "candidate_admission_writer_contract_policy_not_passed"
		return ledger
	}
	if !contract.LiveRouteChoicePassed {
		ledger.Reason = "candidate_admission_writer_contract_live_route_not_passed"
		return ledger
	}
	if contract.AdmissionWriterInventoryID == "" ||
		contract.AdmissionWriterPreflightID == "" ||
		contract.AdmissionLiveStageID == "" ||
		contract.AdmissionEnableGateID == "" ||
		contract.AdmissionSwitchID == "" ||
		contract.AdmissionPromotionID == "" ||
		contract.AdmissionDecisionID == "" ||
		contract.AdmissionAdapterID == "" ||
		contract.CandidateRunID == "" ||
		contract.CandidateDraftID == "" ||
		contract.CandidateExecutionID == "" ||
		contract.GeneratorAdapterID == "" ||
		contract.HandoffID == "" ||
		contract.DreamCandidateRunID == "" ||
		contract.CandidateTextHash == "" ||
		contract.TurnTextHash == "" {
		ledger.Reason = "candidate_admission_writer_contract_missing_provenance"
		return ledger
	}
	ledger.LedgerState = "receipt_drafted_dry_run"
	ledger.LedgerAction = "append_candidate_admission_receipt_dry_run"
	ledger.LedgerContract = contract.AdmissionLedgerContract
	ledger.LedgerMode = contract.LedgerMode
	ledger.LedgerEntryKind = "dream_candidate_admission"
	ledger.LedgerEntryStatus = "shadow_candidate_receipt"
	ledger.LedgerReceiptShape = "candidate_contract_provenance"
	ledger.LedgerAppendReady = true
	ledger.LedgerReceiptPersisted = false
	ledger.LedgerImplementationReady = false
	ledger.ContractsReady = false
	ledger.AdmissionLedgerID = admissionLiveRouteTurnCandidateAdmissionLedgerID(ledger)
	if ledger.AdmissionLedgerID == "" {
		ledger.Reason = "missing_candidate_admission_ledger_id"
		return ledger
	}
	ledger.Passed = true
	ledger.Reason = "admission ledger dry-run receipt drafted; no live write occurred"
	return ledger
}

func admissionLiveRouteTurnCandidateAdmissionLedgerID(ledger admissionLiveRouteTurnCandidateAdmissionLedger) string {
	h := hashJSON(struct {
		AdmissionWriterContractID  string `json:"admission_writer_contract_id"`
		AdmissionWriterInventoryID string `json:"admission_writer_inventory_id"`
		AdmissionWriterPreflightID string `json:"admission_writer_preflight_id"`
		AdmissionLiveStageID       string `json:"admission_live_stage_id"`
		AdmissionEnableGateID      string `json:"admission_enable_gate_id"`
		AdmissionSwitchID          string `json:"admission_switch_id"`
		AdmissionPromotionID       string `json:"admission_promotion_id"`
		AdmissionDecisionID        string `json:"admission_decision_id"`
		AdmissionAdapterID         string `json:"admission_adapter_id"`
		CandidateRunID             string `json:"candidate_run_id"`
		CandidateTextHash          string `json:"candidate_text_hash"`
		TurnTextHash               string `json:"turn_text_hash"`
		LedgerState                string `json:"ledger_state"`
		LedgerAction               string `json:"ledger_action"`
		LedgerContract             string `json:"ledger_contract"`
		LedgerMode                 string `json:"ledger_mode"`
		LedgerEntryKind            string `json:"ledger_entry_kind"`
		LedgerEntryStatus          string `json:"ledger_entry_status"`
		LedgerReceiptShape         string `json:"ledger_receipt_shape"`
		LedgerAppendReady          bool   `json:"ledger_append_ready"`
		LedgerReceiptPersisted     bool   `json:"ledger_receipt_persisted"`
		ContractsReady             bool   `json:"contracts_ready"`
		WriteAllowed               bool   `json:"write_allowed"`
		MutatesState               bool   `json:"mutates_state"`
	}{
		AdmissionWriterContractID:  ledger.AdmissionWriterContractID,
		AdmissionWriterInventoryID: ledger.AdmissionWriterInventoryID,
		AdmissionWriterPreflightID: ledger.AdmissionWriterPreflightID,
		AdmissionLiveStageID:       ledger.AdmissionLiveStageID,
		AdmissionEnableGateID:      ledger.AdmissionEnableGateID,
		AdmissionSwitchID:          ledger.AdmissionSwitchID,
		AdmissionPromotionID:       ledger.AdmissionPromotionID,
		AdmissionDecisionID:        ledger.AdmissionDecisionID,
		AdmissionAdapterID:         ledger.AdmissionAdapterID,
		CandidateRunID:             ledger.CandidateRunID,
		CandidateTextHash:          ledger.CandidateTextHash,
		TurnTextHash:               ledger.TurnTextHash,
		LedgerState:                ledger.LedgerState,
		LedgerAction:               ledger.LedgerAction,
		LedgerContract:             ledger.LedgerContract,
		LedgerMode:                 ledger.LedgerMode,
		LedgerEntryKind:            ledger.LedgerEntryKind,
		LedgerEntryStatus:          ledger.LedgerEntryStatus,
		LedgerReceiptShape:         ledger.LedgerReceiptShape,
		LedgerAppendReady:          ledger.LedgerAppendReady,
		LedgerReceiptPersisted:     ledger.LedgerReceiptPersisted,
		ContractsReady:             ledger.ContractsReady,
		WriteAllowed:               ledger.WriteAllowed,
		MutatesState:               ledger.MutatesState,
	})
	if h == "" {
		return ""
	}
	return "admission-ledger-" + h
}

func recordAdmissionLiveRouteTurnCandidateAdmissionLedger(ledger admissionLiveRouteTurnCandidateAdmissionLedger) error {
	path := strings.TrimSpace(os.Getenv("AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_LEDGER_LOG"))
	if path == "" {
		return nil
	}
	f, err := os.OpenFile(path, os.O_CREATE|os.O_APPEND|os.O_WRONLY, 0600)
	if err != nil {
		return err
	}
	enc := json.NewEncoder(f)
	err = enc.Encode(ledger)
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
