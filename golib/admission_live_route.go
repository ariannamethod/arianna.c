package main

import (
	"encoding/json"
	"os"
	"strconv"
	"strings"
)

const (
	admissionLiveRoutePlanSchema                                          = "arianna.live_route_plan.v1"
	admissionLiveRouteChoiceSchema                                        = "arianna.live_route_choice.v1"
	admissionLiveRouteTurnObservationSchema                               = "arianna.live_route_turn_observation.v1"
	admissionLiveRouteTurnChoiceSchema                                    = "arianna.live_route_turn_choice.v1"
	admissionLiveRouteTurnRequestSchema                                   = "arianna.live_route_turn_request.v1"
	admissionLiveRouteTurnGenerationJobSchema                             = "arianna.live_route_turn_generation_job.v1"
	admissionLiveRouteTurnCandidateShellSchema                            = "arianna.live_route_turn_candidate_shell.v1"
	admissionLiveRouteTurnCandidateExecutionSchema                        = "arianna.live_route_turn_candidate_execution.v1"
	admissionLiveRouteTurnGeneratorAdapterSchema                          = "arianna.live_route_turn_generator_adapter.v1"
	admissionLiveRouteTurnCandidateDraftSchema                            = "arianna.live_route_turn_candidate_draft.v1"
	admissionLiveRouteTurnReviewSchema                                    = "arianna.live_route_turn_candidate_review.v1"
	admissionLiveRouteTurnCandidateAdmissionSchema                        = "arianna.live_route_turn_candidate_admission.v1"
	admissionLiveRouteTurnCandidateAdmissionAdapterSchema                 = "arianna.live_route_turn_candidate_admission_adapter.v1"
	admissionLiveRouteTurnCandidateAdmissionDecisionSchema                = "arianna.live_route_turn_candidate_admission_decision.v1"
	admissionLiveRouteTurnCandidateAdmissionPromotionSchema               = "arianna.live_route_turn_candidate_admission_promotion.v1"
	admissionLiveRouteTurnCandidateAdmissionSwitchSchema                  = "arianna.live_route_turn_candidate_admission_switch.v1"
	admissionLiveRouteTurnCandidateAdmissionEnableGateSchema              = "arianna.live_route_turn_candidate_admission_enable_gate.v1"
	admissionLiveRouteTurnCandidateAdmissionLiveStageSchema               = "arianna.live_route_turn_candidate_admission_live_stage.v1"
	admissionLiveRouteTurnCandidateAdmissionWriterPreflightSchema         = "arianna.live_route_turn_candidate_admission_writer_preflight.v1"
	admissionLiveRouteTurnCandidateAdmissionWriterInventorySchema         = "arianna.live_route_turn_candidate_admission_writer_inventory.v1"
	admissionLiveRouteTurnCandidateAdmissionWriterContractSchema          = "arianna.live_route_turn_candidate_admission_writer_contract.v1"
	admissionLiveRouteTurnCandidateAdmissionLedgerSchema                  = "arianna.live_route_turn_candidate_admission_ledger.v1"
	admissionLiveRouteTurnCandidateAdmissionWriterImplSchema              = "arianna.live_route_turn_candidate_admission_writer_implementation.v1"
	admissionLiveRouteTurnCandidateAdmissionWriterReceiptSchema           = "arianna.live_route_turn_candidate_admission_writer_receipt.v1"
	admissionLiveRouteTurnCandidateAdmissionRollbackImplSchema            = "arianna.live_route_turn_candidate_admission_rollback_implementation.v1"
	admissionLiveRouteTurnCandidateAdmissionLedgerImplSchema              = "arianna.live_route_turn_candidate_admission_ledger_implementation.v1"
	admissionLiveRouteTurnCandidateAdmissionLedgerPersistenceSchema       = "arianna.live_route_turn_candidate_admission_ledger_persistence.v1"
	admissionLiveRouteTurnCandidateAdmissionLedgerVerificationSchema      = "arianna.live_route_turn_candidate_admission_ledger_verification.v1"
	admissionLiveRouteTurnCandidateAdmissionReadinessSchema               = "arianna.live_route_turn_candidate_admission_readiness.v1"
	admissionLiveRouteTurnCandidateAdmissionPermitSchema                  = "arianna.live_route_turn_candidate_admission_permit.v1"
	admissionLiveRouteTurnCandidateAdmissionSealSchema                    = "arianna.live_route_turn_candidate_admission_seal.v1"
	admissionLiveRouteTurnCandidateAdmissionFinalGateSchema               = "arianna.live_route_turn_candidate_admission_final_gate.v1"
	admissionLiveRouteTurnCandidateAdmissionResonanceIntentSchema         = "arianna.live_route_turn_candidate_admission_resonance_intent.v1"
	admissionLiveRouteTurnCandidateAdmissionResonanceReceiverSchema       = "arianna.live_route_turn_candidate_admission_resonance_receiver.v1"
	admissionLiveRouteTurnCandidateAdmissionResonanceObservationSchema    = "arianna.live_route_turn_candidate_admission_resonance_observation.v1"
	admissionLiveRouteTurnCandidateAdmissionResonanceGraftBoundarySchema  = "arianna.live_route_turn_candidate_admission_resonance_graft_boundary.v1"
	admissionLiveRouteTurnCandidateAdmissionResonanceGraftPreflightSchema = "arianna.live_route_turn_candidate_admission_resonance_graft_preflight.v1"
	admissionLiveRouteTurnCandidateAdmissionResonanceGraftGateSchema      = "arianna.live_route_turn_candidate_admission_resonance_graft_gate.v1"
	admissionLiveRouteTurnCandidateAdmissionResonanceGraftCandidateSchema = "arianna.live_route_turn_candidate_admission_resonance_graft_candidate.v1"

	admissionLiveRouteTurnCandidateExecutionDefaultTimeoutMS       = 12000
	admissionLiveRouteTurnCandidateExecutionMaxTimeoutMS           = 60000
	admissionLiveRouteTurnCandidateAdmissionEnableGateConfirmation = "ARIANNA_LIVE_ADMISSION_ENABLE_DRY_RUN_ONLY"
	admissionLiveRouteTurnCandidateAdmissionPermitConfirmation     = "ARIANNA_LIVE_ADMISSION_PERMIT_DRY_RUN_ONLY"
	admissionLiveRouteTurnCandidateAdmissionResonanceIntentTTL     = 1
	admissionLiveRouteTurnCandidateAdmissionResonanceIntentMaxGain = 0.05

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

type admissionLiveRouteTurnCandidateAdmissionWriterImplementation struct {
	Schema                        string `json:"schema"`
	Timing                        string `json:"timing"`
	ImplementationState           string `json:"implementation_state,omitempty"`
	ImplementationAction          string `json:"implementation_action,omitempty"`
	WriterEntrypoint              string `json:"writer_entrypoint,omitempty"`
	LedgerEntrypoint              string `json:"ledger_entrypoint,omitempty"`
	RollbackEntrypoint            string `json:"rollback_entrypoint,omitempty"`
	WriteTarget                   string `json:"write_target,omitempty"`
	BodyTarget                    string `json:"body_target,omitempty"`
	AppendOnly                    bool   `json:"append_only"`
	RollbackRequired              bool   `json:"rollback_required"`
	ImplementationContractReady   bool   `json:"implementation_contract_ready"`
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
	AdmissionLedgerID             string `json:"admission_ledger_id,omitempty"`
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
	SourceLedgerPassed            bool   `json:"source_ledger_passed"`
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
	WriterImplementationID        string `json:"writer_implementation_id,omitempty"`
	Passed                        bool   `json:"passed"`
	Reason                        string `json:"reason,omitempty"`
	TurnTextHash                  string `json:"turn_text_hash,omitempty"`
}

type admissionLiveRouteTurnCandidateAdmissionWriterReceipt struct {
	admissionLiveRouteTurnCandidateAdmissionWriterImplementation

	WriterReceiptState                     string `json:"writer_receipt_state,omitempty"`
	WriterReceiptAction                    string `json:"writer_receipt_action,omitempty"`
	WriterReceiptKind                      string `json:"writer_receipt_kind,omitempty"`
	WriterReceiptTarget                    string `json:"writer_receipt_target,omitempty"`
	WriterReceiptMode                      string `json:"writer_receipt_mode,omitempty"`
	WriterReceiptShape                     string `json:"writer_receipt_shape,omitempty"`
	WriterReceiptPersisted                 bool   `json:"writer_receipt_persisted"`
	ShadowWriteAllowed                     bool   `json:"shadow_write_allowed"`
	SourceWriterImplementationPassed       bool   `json:"source_writer_implementation_passed"`
	SourceWriterImplementationID           string `json:"source_writer_implementation_id,omitempty"`
	SourceWriterImplementationEntrypoint   string `json:"source_writer_implementation_entrypoint,omitempty"`
	SourceLedgerImplementationEntrypoint   string `json:"source_ledger_implementation_entrypoint,omitempty"`
	SourceRollbackImplementationEntrypoint string `json:"source_rollback_implementation_entrypoint,omitempty"`
	WriterReceiptID                        string `json:"writer_receipt_id,omitempty"`
}

type admissionLiveRouteTurnCandidateAdmissionRollbackImplementation struct {
	admissionLiveRouteTurnCandidateAdmissionWriterReceipt

	RollbackImplementationState       string `json:"rollback_implementation_state,omitempty"`
	RollbackImplementationAction      string `json:"rollback_implementation_action,omitempty"`
	RollbackEntrypointResolved        string `json:"rollback_entrypoint_resolved,omitempty"`
	RollbackTarget                    string `json:"rollback_target,omitempty"`
	RollbackTargetKind                string `json:"rollback_target_kind,omitempty"`
	RollbackTargetID                  string `json:"rollback_target_id,omitempty"`
	RollbackMode                      string `json:"rollback_mode,omitempty"`
	ExactReceiptMatchRequired         bool   `json:"exact_receipt_match_required"`
	RollbackDryRunOnly                bool   `json:"rollback_dry_run_only"`
	RollbackReceiptRemoved            bool   `json:"rollback_receipt_removed"`
	SourceWriterReceiptSchema         string `json:"source_writer_receipt_schema,omitempty"`
	SourceWriterReceiptPassed         bool   `json:"source_writer_receipt_passed"`
	SourceWriterReceiptID             string `json:"source_writer_receipt_id,omitempty"`
	SourceWriterReceiptAction         string `json:"source_writer_receipt_action,omitempty"`
	SourceWriterReceiptPersisted      bool   `json:"source_writer_receipt_persisted"`
	SourceWriterReceiptShadowWritable bool   `json:"source_writer_receipt_shadow_writable"`
	RollbackImplementationID          string `json:"rollback_implementation_id,omitempty"`
}

type admissionLiveRouteTurnCandidateAdmissionLedgerImplementation struct {
	admissionLiveRouteTurnCandidateAdmissionRollbackImplementation

	LedgerImplementationState            string `json:"ledger_implementation_state,omitempty"`
	LedgerImplementationAction           string `json:"ledger_implementation_action,omitempty"`
	LedgerEntrypointResolved             string `json:"ledger_entrypoint_resolved,omitempty"`
	LedgerImplementationTarget           string `json:"ledger_implementation_target,omitempty"`
	LedgerImplementationTargetKind       string `json:"ledger_implementation_target_kind,omitempty"`
	LedgerImplementationTargetMode       string `json:"ledger_implementation_target_mode,omitempty"`
	LedgerImplementationAppendOnly       bool   `json:"ledger_implementation_append_only"`
	LedgerImplementationDryRunOnly       bool   `json:"ledger_implementation_dry_run_only"`
	LedgerImplementationReceiptPersisted bool   `json:"ledger_implementation_receipt_persisted"`
	SourceRollbackImplementationSchema   string `json:"source_rollback_implementation_schema,omitempty"`
	SourceRollbackImplementationPassed   bool   `json:"source_rollback_implementation_passed"`
	SourceRollbackImplementationID       string `json:"source_rollback_implementation_id,omitempty"`
	SourceRollbackImplementationAction   string `json:"source_rollback_implementation_action,omitempty"`
	SourceRollbackImplementationReady    bool   `json:"source_rollback_implementation_ready"`
	SourceRollbackTargetID               string `json:"source_rollback_target_id,omitempty"`
	SourceWriterReceiptIDForLedger       string `json:"source_writer_receipt_id_for_ledger,omitempty"`
	LedgerImplementationID               string `json:"ledger_implementation_id,omitempty"`
}

type admissionLiveRouteTurnCandidateAdmissionLedgerPersistence struct {
	admissionLiveRouteTurnCandidateAdmissionLedgerImplementation

	LedgerPersistenceState                    string `json:"ledger_persistence_state,omitempty"`
	LedgerPersistenceAction                   string `json:"ledger_persistence_action,omitempty"`
	LedgerPersistenceTarget                   string `json:"ledger_persistence_target,omitempty"`
	LedgerPersistenceTargetKind               string `json:"ledger_persistence_target_kind,omitempty"`
	LedgerPersistenceTargetMode               string `json:"ledger_persistence_target_mode,omitempty"`
	LedgerPersistenceReceiptShape             string `json:"ledger_persistence_receipt_shape,omitempty"`
	LedgerPersistenceAppendOnly               bool   `json:"ledger_persistence_append_only"`
	LedgerPersistenceDryRunOnly               bool   `json:"ledger_persistence_dry_run_only"`
	LedgerPersistenceReceiptPersisted         bool   `json:"ledger_persistence_receipt_persisted"`
	LedgerPersistenceReady                    bool   `json:"ledger_persistence_ready"`
	SourceLedgerImplementationSchema          string `json:"source_ledger_implementation_schema,omitempty"`
	SourceLedgerImplementationPassed          bool   `json:"source_ledger_implementation_passed"`
	SourceLedgerImplementationID              string `json:"source_ledger_implementation_id,omitempty"`
	SourceLedgerImplementationAction          string `json:"source_ledger_implementation_action,omitempty"`
	SourceLedgerImplementationReady           bool   `json:"source_ledger_implementation_ready"`
	SourceAdmissionLedgerIDForPersistence     string `json:"source_admission_ledger_id_for_persistence,omitempty"`
	SourceRollbackImplementationIDForLedger   string `json:"source_rollback_implementation_id_for_ledger,omitempty"`
	SourceWriterReceiptIDForLedgerPersistence string `json:"source_writer_receipt_id_for_ledger_persistence,omitempty"`
	LedgerPersistenceID                       string `json:"ledger_persistence_id,omitempty"`
}

type admissionLiveRouteTurnCandidateAdmissionLedgerVerification struct {
	admissionLiveRouteTurnCandidateAdmissionLedgerPersistence

	LedgerVerificationState                       string `json:"ledger_verification_state,omitempty"`
	LedgerVerificationAction                      string `json:"ledger_verification_action,omitempty"`
	LedgerVerificationTarget                      string `json:"ledger_verification_target,omitempty"`
	LedgerVerificationTargetKind                  string `json:"ledger_verification_target_kind,omitempty"`
	LedgerVerificationTargetMode                  string `json:"ledger_verification_target_mode,omitempty"`
	LedgerVerificationReceiptShape                string `json:"ledger_verification_receipt_shape,omitempty"`
	LedgerVerificationAppendOnly                  bool   `json:"ledger_verification_append_only"`
	LedgerVerificationDryRunOnly                  bool   `json:"ledger_verification_dry_run_only"`
	LedgerVerificationReceiptReadBack             bool   `json:"ledger_verification_receipt_read_back"`
	LedgerVerificationReceiptVerified             bool   `json:"ledger_verification_receipt_verified"`
	LedgerVerificationReady                       bool   `json:"ledger_verification_ready"`
	SourceLedgerPersistenceSchema                 string `json:"source_ledger_persistence_schema,omitempty"`
	SourceLedgerPersistencePassed                 bool   `json:"source_ledger_persistence_passed"`
	SourceLedgerPersistenceID                     string `json:"source_ledger_persistence_id,omitempty"`
	SourceLedgerPersistenceAction                 string `json:"source_ledger_persistence_action,omitempty"`
	SourceLedgerPersistenceReady                  bool   `json:"source_ledger_persistence_ready"`
	SourceLedgerPersistenceReceiptPersisted       bool   `json:"source_ledger_persistence_receipt_persisted"`
	SourceLedgerImplementationIDForVerification   string `json:"source_ledger_implementation_id_for_verification,omitempty"`
	SourceAdmissionLedgerIDForVerification        string `json:"source_admission_ledger_id_for_verification,omitempty"`
	SourceRollbackImplementationIDForVerification string `json:"source_rollback_implementation_id_for_verification,omitempty"`
	SourceWriterReceiptIDForVerification          string `json:"source_writer_receipt_id_for_verification,omitempty"`
	LedgerVerificationID                          string `json:"ledger_verification_id,omitempty"`
}

type admissionLiveRouteTurnCandidateAdmissionReadiness struct {
	admissionLiveRouteTurnCandidateAdmissionLedgerVerification

	AdmissionReadinessState                    string `json:"admission_readiness_state,omitempty"`
	AdmissionReadinessAction                   string `json:"admission_readiness_action,omitempty"`
	AdmissionReadinessTarget                   string `json:"admission_readiness_target,omitempty"`
	AdmissionReadinessTargetKind               string `json:"admission_readiness_target_kind,omitempty"`
	AdmissionReadinessTargetMode               string `json:"admission_readiness_target_mode,omitempty"`
	AdmissionReadinessDryRunOnly               bool   `json:"admission_readiness_dry_run_only"`
	AdmissionReadinessLedgerVerified           bool   `json:"admission_readiness_ledger_verified"`
	AdmissionReadinessWriterReady              bool   `json:"admission_readiness_writer_ready"`
	AdmissionReadinessRollbackReady            bool   `json:"admission_readiness_rollback_ready"`
	AdmissionReadinessLedgerReady              bool   `json:"admission_readiness_ledger_ready"`
	AdmissionReadinessReady                    bool   `json:"admission_readiness_ready"`
	SourceLedgerVerificationSchema             string `json:"source_ledger_verification_schema,omitempty"`
	SourceLedgerVerificationPassed             bool   `json:"source_ledger_verification_passed"`
	SourceLedgerVerificationID                 string `json:"source_ledger_verification_id,omitempty"`
	SourceLedgerVerificationAction             string `json:"source_ledger_verification_action,omitempty"`
	SourceLedgerVerificationReady              bool   `json:"source_ledger_verification_ready"`
	SourceLedgerVerificationReceiptVerified    bool   `json:"source_ledger_verification_receipt_verified"`
	SourceLedgerPersistenceIDForReadiness      string `json:"source_ledger_persistence_id_for_readiness,omitempty"`
	SourceLedgerImplementationIDForReadiness   string `json:"source_ledger_implementation_id_for_readiness,omitempty"`
	SourceAdmissionLedgerIDForReadiness        string `json:"source_admission_ledger_id_for_readiness,omitempty"`
	SourceRollbackImplementationIDForReadiness string `json:"source_rollback_implementation_id_for_readiness,omitempty"`
	SourceWriterReceiptIDForReadiness          string `json:"source_writer_receipt_id_for_readiness,omitempty"`
	AdmissionReadinessID                       string `json:"admission_readiness_id,omitempty"`
}

type admissionLiveRouteTurnCandidateAdmissionPermit struct {
	admissionLiveRouteTurnCandidateAdmissionReadiness

	AdmissionPermitState                    string `json:"admission_permit_state,omitempty"`
	AdmissionPermitAction                   string `json:"admission_permit_action,omitempty"`
	AdmissionPermitTarget                   string `json:"admission_permit_target,omitempty"`
	AdmissionPermitTargetKind               string `json:"admission_permit_target_kind,omitempty"`
	AdmissionPermitTargetMode               string `json:"admission_permit_target_mode,omitempty"`
	AdmissionPermitDryRunOnly               bool   `json:"admission_permit_dry_run_only"`
	AdmissionPermitReadinessVerified        bool   `json:"admission_permit_readiness_verified"`
	AdmissionPermitLedgerVerified           bool   `json:"admission_permit_ledger_verified"`
	AdmissionPermitWriterReady              bool   `json:"admission_permit_writer_ready"`
	AdmissionPermitRollbackReady            bool   `json:"admission_permit_rollback_ready"`
	AdmissionPermitLedgerReady              bool   `json:"admission_permit_ledger_ready"`
	AdmissionPermitReady                    bool   `json:"admission_permit_ready"`
	ManualPermitRequested                   bool   `json:"manual_permit_requested"`
	PermitKeyMatched                        bool   `json:"permit_key_matched"`
	SourceAdmissionReadinessSchema          string `json:"source_admission_readiness_schema,omitempty"`
	SourceAdmissionReadinessPassed          bool   `json:"source_admission_readiness_passed"`
	SourceAdmissionReadinessID              string `json:"source_admission_readiness_id,omitempty"`
	SourceAdmissionReadinessAction          string `json:"source_admission_readiness_action,omitempty"`
	SourceAdmissionReadinessReady           bool   `json:"source_admission_readiness_ready"`
	SourceAdmissionReadinessLedgerVerified  bool   `json:"source_admission_readiness_ledger_verified"`
	SourceLedgerVerificationIDForPermit     string `json:"source_ledger_verification_id_for_permit,omitempty"`
	SourceLedgerPersistenceIDForPermit      string `json:"source_ledger_persistence_id_for_permit,omitempty"`
	SourceLedgerImplementationIDForPermit   string `json:"source_ledger_implementation_id_for_permit,omitempty"`
	SourceAdmissionLedgerIDForPermit        string `json:"source_admission_ledger_id_for_permit,omitempty"`
	SourceRollbackImplementationIDForPermit string `json:"source_rollback_implementation_id_for_permit,omitempty"`
	SourceWriterReceiptIDForPermit          string `json:"source_writer_receipt_id_for_permit,omitempty"`
	AdmissionPermitID                       string `json:"admission_permit_id,omitempty"`
}

type admissionLiveRouteTurnCandidateAdmissionSeal struct {
	admissionLiveRouteTurnCandidateAdmissionPermit

	AdmissionSealState                    string `json:"admission_seal_state,omitempty"`
	AdmissionSealAction                   string `json:"admission_seal_action,omitempty"`
	AdmissionSealTarget                   string `json:"admission_seal_target,omitempty"`
	AdmissionSealTargetKind               string `json:"admission_seal_target_kind,omitempty"`
	AdmissionSealTargetMode               string `json:"admission_seal_target_mode,omitempty"`
	AdmissionSealReceiptShape             string `json:"admission_seal_receipt_shape,omitempty"`
	AdmissionSealDryRunOnly               bool   `json:"admission_seal_dry_run_only"`
	AdmissionSealPermitVerified           bool   `json:"admission_seal_permit_verified"`
	AdmissionSealReadinessVerified        bool   `json:"admission_seal_readiness_verified"`
	AdmissionSealLedgerVerified           bool   `json:"admission_seal_ledger_verified"`
	AdmissionSealWriterReady              bool   `json:"admission_seal_writer_ready"`
	AdmissionSealRollbackReady            bool   `json:"admission_seal_rollback_ready"`
	AdmissionSealLedgerReady              bool   `json:"admission_seal_ledger_ready"`
	AdmissionSealReady                    bool   `json:"admission_seal_ready"`
	SourceAdmissionPermitSchema           string `json:"source_admission_permit_schema,omitempty"`
	SourceAdmissionPermitPassed           bool   `json:"source_admission_permit_passed"`
	SourceAdmissionPermitID               string `json:"source_admission_permit_id,omitempty"`
	SourceAdmissionPermitAction           string `json:"source_admission_permit_action,omitempty"`
	SourceAdmissionPermitReady            bool   `json:"source_admission_permit_ready"`
	SourceAdmissionPermitKeyMatched       bool   `json:"source_admission_permit_key_matched"`
	SourceAdmissionReadinessIDForSeal     string `json:"source_admission_readiness_id_for_seal,omitempty"`
	SourceLedgerVerificationIDForSeal     string `json:"source_ledger_verification_id_for_seal,omitempty"`
	SourceLedgerPersistenceIDForSeal      string `json:"source_ledger_persistence_id_for_seal,omitempty"`
	SourceLedgerImplementationIDForSeal   string `json:"source_ledger_implementation_id_for_seal,omitempty"`
	SourceAdmissionLedgerIDForSeal        string `json:"source_admission_ledger_id_for_seal,omitempty"`
	SourceRollbackImplementationIDForSeal string `json:"source_rollback_implementation_id_for_seal,omitempty"`
	SourceWriterReceiptIDForSeal          string `json:"source_writer_receipt_id_for_seal,omitempty"`
	AdmissionSealID                       string `json:"admission_seal_id,omitempty"`
}

type admissionLiveRouteTurnCandidateAdmissionFinalGate struct {
	admissionLiveRouteTurnCandidateAdmissionSeal

	AdmissionFinalGateState                    string `json:"admission_final_gate_state,omitempty"`
	AdmissionFinalGateAction                   string `json:"admission_final_gate_action,omitempty"`
	AdmissionFinalGateTarget                   string `json:"admission_final_gate_target,omitempty"`
	AdmissionFinalGateTargetKind               string `json:"admission_final_gate_target_kind,omitempty"`
	AdmissionFinalGateTargetMode               string `json:"admission_final_gate_target_mode,omitempty"`
	AdmissionFinalGateReceiptShape             string `json:"admission_final_gate_receipt_shape,omitempty"`
	AdmissionFinalGateDryRunOnly               bool   `json:"admission_final_gate_dry_run_only"`
	AdmissionFinalGateSealVerified             bool   `json:"admission_final_gate_seal_verified"`
	AdmissionFinalGatePermitVerified           bool   `json:"admission_final_gate_permit_verified"`
	AdmissionFinalGateReadinessVerified        bool   `json:"admission_final_gate_readiness_verified"`
	AdmissionFinalGateLedgerVerified           bool   `json:"admission_final_gate_ledger_verified"`
	AdmissionFinalGateWriterReady              bool   `json:"admission_final_gate_writer_ready"`
	AdmissionFinalGateRollbackReady            bool   `json:"admission_final_gate_rollback_ready"`
	AdmissionFinalGateLedgerReady              bool   `json:"admission_final_gate_ledger_ready"`
	AdmissionFinalGateReady                    bool   `json:"admission_final_gate_ready"`
	SourceAdmissionSealSchema                  string `json:"source_admission_seal_schema,omitempty"`
	SourceAdmissionSealPassed                  bool   `json:"source_admission_seal_passed"`
	SourceAdmissionSealID                      string `json:"source_admission_seal_id,omitempty"`
	SourceAdmissionSealAction                  string `json:"source_admission_seal_action,omitempty"`
	SourceAdmissionSealReady                   bool   `json:"source_admission_seal_ready"`
	SourceAdmissionPermitIDForFinalGate        string `json:"source_admission_permit_id_for_final_gate,omitempty"`
	SourceAdmissionReadinessIDForFinalGate     string `json:"source_admission_readiness_id_for_final_gate,omitempty"`
	SourceLedgerVerificationIDForFinalGate     string `json:"source_ledger_verification_id_for_final_gate,omitempty"`
	SourceLedgerPersistenceIDForFinalGate      string `json:"source_ledger_persistence_id_for_final_gate,omitempty"`
	SourceLedgerImplementationIDForFinalGate   string `json:"source_ledger_implementation_id_for_final_gate,omitempty"`
	SourceAdmissionLedgerIDForFinalGate        string `json:"source_admission_ledger_id_for_final_gate,omitempty"`
	SourceRollbackImplementationIDForFinalGate string `json:"source_rollback_implementation_id_for_final_gate,omitempty"`
	SourceWriterReceiptIDForFinalGate          string `json:"source_writer_receipt_id_for_final_gate,omitempty"`
	AdmissionFinalGateID                       string `json:"admission_final_gate_id,omitempty"`
}

type admissionLiveRouteTurnCandidateAdmissionResonanceIntent struct {
	admissionLiveRouteTurnCandidateAdmissionFinalGate

	AdmissionResonanceIntentState                    string  `json:"admission_resonance_intent_state,omitempty"`
	AdmissionResonanceIntentAction                   string  `json:"admission_resonance_intent_action,omitempty"`
	AdmissionResonanceIntentTarget                   string  `json:"admission_resonance_intent_target,omitempty"`
	AdmissionResonanceIntentTargetKind               string  `json:"admission_resonance_intent_target_kind,omitempty"`
	AdmissionResonanceIntentTargetMode               string  `json:"admission_resonance_intent_target_mode,omitempty"`
	AdmissionResonanceIntentReceiptShape             string  `json:"admission_resonance_intent_receipt_shape,omitempty"`
	AdmissionResonanceIntentDryRunOnly               bool    `json:"admission_resonance_intent_dry_run_only"`
	AdmissionResonanceIntentFinalGateVerified        bool    `json:"admission_resonance_intent_final_gate_verified"`
	AdmissionResonanceIntentSealVerified             bool    `json:"admission_resonance_intent_seal_verified"`
	AdmissionResonanceIntentPermitVerified           bool    `json:"admission_resonance_intent_permit_verified"`
	AdmissionResonanceIntentReadinessVerified        bool    `json:"admission_resonance_intent_readiness_verified"`
	AdmissionResonanceIntentLedgerVerified           bool    `json:"admission_resonance_intent_ledger_verified"`
	AdmissionResonanceIntentWriterReady              bool    `json:"admission_resonance_intent_writer_ready"`
	AdmissionResonanceIntentRollbackReady            bool    `json:"admission_resonance_intent_rollback_ready"`
	AdmissionResonanceIntentLedgerReady              bool    `json:"admission_resonance_intent_ledger_ready"`
	AdmissionResonanceIntentReceiver                 string  `json:"admission_resonance_intent_receiver,omitempty"`
	AdmissionResonanceIntentReceiverKind             string  `json:"admission_resonance_intent_receiver_kind,omitempty"`
	AdmissionResonanceIntentInfluenceKind            string  `json:"admission_resonance_intent_influence_kind,omitempty"`
	AdmissionResonanceIntentMaxInfluence             float64 `json:"admission_resonance_intent_max_influence"`
	AdmissionResonanceIntentTTLTurns                 int     `json:"admission_resonance_intent_ttl_turns"`
	AdmissionResonanceIntentCausalID                 string  `json:"admission_resonance_intent_causal_id,omitempty"`
	AdmissionResonanceIntentRawDreamTextAllowed      bool    `json:"admission_resonance_intent_raw_dream_text_allowed"`
	AdmissionResonanceIntentJanusSurfaceAllowed      bool    `json:"admission_resonance_intent_janus_surface_allowed"`
	AdmissionResonanceIntentCoocLearningAllowed      bool    `json:"admission_resonance_intent_cooc_learning_allowed"`
	AdmissionResonanceIntentDeltaHarvestAllowed      bool    `json:"admission_resonance_intent_delta_harvest_allowed"`
	AdmissionResonanceIntentRollbackRequired         bool    `json:"admission_resonance_intent_rollback_required"`
	AdmissionResonanceIntentPreStateHashRequired     bool    `json:"admission_resonance_intent_pre_state_hash_required"`
	AdmissionResonanceIntentPostStateHashRequired    bool    `json:"admission_resonance_intent_post_state_hash_required"`
	AdmissionResonanceIntentReady                    bool    `json:"admission_resonance_intent_ready"`
	SourceAdmissionFinalGateSchema                   string  `json:"source_admission_final_gate_schema,omitempty"`
	SourceAdmissionFinalGatePassed                   bool    `json:"source_admission_final_gate_passed"`
	SourceAdmissionFinalGateID                       string  `json:"source_admission_final_gate_id,omitempty"`
	SourceAdmissionFinalGateAction                   string  `json:"source_admission_final_gate_action,omitempty"`
	SourceAdmissionFinalGateReady                    bool    `json:"source_admission_final_gate_ready"`
	SourceAdmissionSealIDForResonanceIntent          string  `json:"source_admission_seal_id_for_resonance_intent,omitempty"`
	SourceAdmissionPermitIDForResonanceIntent        string  `json:"source_admission_permit_id_for_resonance_intent,omitempty"`
	SourceAdmissionReadinessIDForResonanceIntent     string  `json:"source_admission_readiness_id_for_resonance_intent,omitempty"`
	SourceLedgerVerificationIDForResonanceIntent     string  `json:"source_ledger_verification_id_for_resonance_intent,omitempty"`
	SourceLedgerPersistenceIDForResonanceIntent      string  `json:"source_ledger_persistence_id_for_resonance_intent,omitempty"`
	SourceLedgerImplementationIDForResonanceIntent   string  `json:"source_ledger_implementation_id_for_resonance_intent,omitempty"`
	SourceAdmissionLedgerIDForResonanceIntent        string  `json:"source_admission_ledger_id_for_resonance_intent,omitempty"`
	SourceRollbackImplementationIDForResonanceIntent string  `json:"source_rollback_implementation_id_for_resonance_intent,omitempty"`
	SourceWriterReceiptIDForResonanceIntent          string  `json:"source_writer_receipt_id_for_resonance_intent,omitempty"`
	AdmissionResonanceIntentID                       string  `json:"admission_resonance_intent_id,omitempty"`
}

type admissionLiveRouteTurnCandidateAdmissionResonanceReceiver struct {
	admissionLiveRouteTurnCandidateAdmissionResonanceIntent

	AdmissionResonanceReceiverState                    string  `json:"admission_resonance_receiver_state,omitempty"`
	AdmissionResonanceReceiverAction                   string  `json:"admission_resonance_receiver_action,omitempty"`
	AdmissionResonanceReceiverTarget                   string  `json:"admission_resonance_receiver_target,omitempty"`
	AdmissionResonanceReceiverTargetKind               string  `json:"admission_resonance_receiver_target_kind,omitempty"`
	AdmissionResonanceReceiverTargetMode               string  `json:"admission_resonance_receiver_target_mode,omitempty"`
	AdmissionResonanceReceiverReceiptShape             string  `json:"admission_resonance_receiver_receipt_shape,omitempty"`
	AdmissionResonanceReceiverDryRunOnly               bool    `json:"admission_resonance_receiver_dry_run_only"`
	AdmissionResonanceReceiverIntentVerified           bool    `json:"admission_resonance_receiver_intent_verified"`
	AdmissionResonanceReceiverFinalGateVerified        bool    `json:"admission_resonance_receiver_final_gate_verified"`
	AdmissionResonanceReceiverSealVerified             bool    `json:"admission_resonance_receiver_seal_verified"`
	AdmissionResonanceReceiverPermitVerified           bool    `json:"admission_resonance_receiver_permit_verified"`
	AdmissionResonanceReceiverReadinessVerified        bool    `json:"admission_resonance_receiver_readiness_verified"`
	AdmissionResonanceReceiverLedgerVerified           bool    `json:"admission_resonance_receiver_ledger_verified"`
	AdmissionResonanceReceiverWriterReady              bool    `json:"admission_resonance_receiver_writer_ready"`
	AdmissionResonanceReceiverRollbackReady            bool    `json:"admission_resonance_receiver_rollback_ready"`
	AdmissionResonanceReceiverLedgerReady              bool    `json:"admission_resonance_receiver_ledger_ready"`
	AdmissionResonanceReceiverReceiver                 string  `json:"admission_resonance_receiver_receiver,omitempty"`
	AdmissionResonanceReceiverReceiverKind             string  `json:"admission_resonance_receiver_receiver_kind,omitempty"`
	AdmissionResonanceReceiverInfluenceKind            string  `json:"admission_resonance_receiver_influence_kind,omitempty"`
	AdmissionResonanceReceiverMaxInfluence             float64 `json:"admission_resonance_receiver_max_influence"`
	AdmissionResonanceReceiverTTLTurns                 int     `json:"admission_resonance_receiver_ttl_turns"`
	AdmissionResonanceReceiverCausalID                 string  `json:"admission_resonance_receiver_causal_id,omitempty"`
	AdmissionResonanceReceiverPreStateHash             string  `json:"admission_resonance_receiver_pre_state_hash,omitempty"`
	AdmissionResonanceReceiverPostStateHash            string  `json:"admission_resonance_receiver_post_state_hash,omitempty"`
	AdmissionResonanceReceiverStateDeltaHash           string  `json:"admission_resonance_receiver_state_delta_hash,omitempty"`
	AdmissionResonanceReceiverStateHashMode            string  `json:"admission_resonance_receiver_state_hash_mode,omitempty"`
	AdmissionResonanceReceiverRawDreamTextObserved     bool    `json:"admission_resonance_receiver_raw_dream_text_observed"`
	AdmissionResonanceReceiverRawDreamTextForwarded    bool    `json:"admission_resonance_receiver_raw_dream_text_forwarded"`
	AdmissionResonanceReceiverJanusSurfaceAllowed      bool    `json:"admission_resonance_receiver_janus_surface_allowed"`
	AdmissionResonanceReceiverCoocLearningAllowed      bool    `json:"admission_resonance_receiver_cooc_learning_allowed"`
	AdmissionResonanceReceiverDeltaHarvestAllowed      bool    `json:"admission_resonance_receiver_delta_harvest_allowed"`
	AdmissionResonanceReceiverBodyMutationAllowed      bool    `json:"admission_resonance_receiver_body_mutation_allowed"`
	AdmissionResonanceReceiverRollbackRequired         bool    `json:"admission_resonance_receiver_rollback_required"`
	AdmissionResonanceReceiverReady                    bool    `json:"admission_resonance_receiver_ready"`
	SourceAdmissionResonanceIntentSchema               string  `json:"source_admission_resonance_intent_schema,omitempty"`
	SourceAdmissionResonanceIntentPassed               bool    `json:"source_admission_resonance_intent_passed"`
	SourceAdmissionResonanceIntentID                   string  `json:"source_admission_resonance_intent_id,omitempty"`
	SourceAdmissionResonanceIntentAction               string  `json:"source_admission_resonance_intent_action,omitempty"`
	SourceAdmissionResonanceIntentReady                bool    `json:"source_admission_resonance_intent_ready"`
	SourceAdmissionResonanceIntentCausalID             string  `json:"source_admission_resonance_intent_causal_id,omitempty"`
	SourceAdmissionFinalGateIDForResonanceReceiver     string  `json:"source_admission_final_gate_id_for_resonance_receiver,omitempty"`
	SourceAdmissionSealIDForResonanceReceiver          string  `json:"source_admission_seal_id_for_resonance_receiver,omitempty"`
	SourceAdmissionPermitIDForResonanceReceiver        string  `json:"source_admission_permit_id_for_resonance_receiver,omitempty"`
	SourceAdmissionReadinessIDForResonanceReceiver     string  `json:"source_admission_readiness_id_for_resonance_receiver,omitempty"`
	SourceLedgerVerificationIDForResonanceReceiver     string  `json:"source_ledger_verification_id_for_resonance_receiver,omitempty"`
	SourceLedgerPersistenceIDForResonanceReceiver      string  `json:"source_ledger_persistence_id_for_resonance_receiver,omitempty"`
	SourceLedgerImplementationIDForResonanceReceiver   string  `json:"source_ledger_implementation_id_for_resonance_receiver,omitempty"`
	SourceAdmissionLedgerIDForResonanceReceiver        string  `json:"source_admission_ledger_id_for_resonance_receiver,omitempty"`
	SourceRollbackImplementationIDForResonanceReceiver string  `json:"source_rollback_implementation_id_for_resonance_receiver,omitempty"`
	SourceWriterReceiptIDForResonanceReceiver          string  `json:"source_writer_receipt_id_for_resonance_receiver,omitempty"`
	AdmissionResonanceReceiverID                       string  `json:"admission_resonance_receiver_id,omitempty"`
}

type admissionLiveRouteTurnCandidateAdmissionResonanceObservation struct {
	admissionLiveRouteTurnCandidateAdmissionResonanceReceiver

	AdmissionResonanceObservationState                    string `json:"admission_resonance_observation_state,omitempty"`
	AdmissionResonanceObservationAction                   string `json:"admission_resonance_observation_action,omitempty"`
	AdmissionResonanceObservationTarget                   string `json:"admission_resonance_observation_target,omitempty"`
	AdmissionResonanceObservationTargetKind               string `json:"admission_resonance_observation_target_kind,omitempty"`
	AdmissionResonanceObservationTargetMode               string `json:"admission_resonance_observation_target_mode,omitempty"`
	AdmissionResonanceObservationReceiptShape             string `json:"admission_resonance_observation_receipt_shape,omitempty"`
	AdmissionResonanceObservationDryRunOnly               bool   `json:"admission_resonance_observation_dry_run_only"`
	AdmissionResonanceObservationReceiverVerified         bool   `json:"admission_resonance_observation_receiver_verified"`
	AdmissionResonanceObservationIntentVerified           bool   `json:"admission_resonance_observation_intent_verified"`
	AdmissionResonanceObservationFinalGateVerified        bool   `json:"admission_resonance_observation_final_gate_verified"`
	AdmissionResonanceObservationSealVerified             bool   `json:"admission_resonance_observation_seal_verified"`
	AdmissionResonanceObservationPermitVerified           bool   `json:"admission_resonance_observation_permit_verified"`
	AdmissionResonanceObservationReadinessVerified        bool   `json:"admission_resonance_observation_readiness_verified"`
	AdmissionResonanceObservationLedgerVerified           bool   `json:"admission_resonance_observation_ledger_verified"`
	AdmissionResonanceObservationWriterReady              bool   `json:"admission_resonance_observation_writer_ready"`
	AdmissionResonanceObservationRollbackReady            bool   `json:"admission_resonance_observation_rollback_ready"`
	AdmissionResonanceObservationLedgerReady              bool   `json:"admission_resonance_observation_ledger_ready"`
	AdmissionResonanceObservationObserver                 string `json:"admission_resonance_observation_observer,omitempty"`
	AdmissionResonanceObservationObserverKind             string `json:"admission_resonance_observation_observer_kind,omitempty"`
	AdmissionResonanceObservationKind                     string `json:"admission_resonance_observation_kind,omitempty"`
	AdmissionResonanceObservationMode                     string `json:"admission_resonance_observation_mode,omitempty"`
	AdmissionResonanceObservationCausalID                 string `json:"admission_resonance_observation_causal_id,omitempty"`
	AdmissionResonanceObservationAppendHash               string `json:"admission_resonance_observation_append_hash,omitempty"`
	AdmissionResonanceObservationReadBackHash             string `json:"admission_resonance_observation_read_back_hash,omitempty"`
	AdmissionResonanceObservationAppendOnly               bool   `json:"admission_resonance_observation_append_only"`
	AdmissionResonanceObservationReadBack                 bool   `json:"admission_resonance_observation_read_back"`
	AdmissionResonanceObservationReceiptVerified          bool   `json:"admission_resonance_observation_receipt_verified"`
	AdmissionResonanceObservationRawDreamTextObserved     bool   `json:"admission_resonance_observation_raw_dream_text_observed"`
	AdmissionResonanceObservationRawDreamTextForwarded    bool   `json:"admission_resonance_observation_raw_dream_text_forwarded"`
	AdmissionResonanceObservationJanusSurfaceAllowed      bool   `json:"admission_resonance_observation_janus_surface_allowed"`
	AdmissionResonanceObservationCoocLearningAllowed      bool   `json:"admission_resonance_observation_cooc_learning_allowed"`
	AdmissionResonanceObservationDeltaHarvestAllowed      bool   `json:"admission_resonance_observation_delta_harvest_allowed"`
	AdmissionResonanceObservationBodyMutationAllowed      bool   `json:"admission_resonance_observation_body_mutation_allowed"`
	AdmissionResonanceObservationRollbackRequired         bool   `json:"admission_resonance_observation_rollback_required"`
	AdmissionResonanceObservationReady                    bool   `json:"admission_resonance_observation_ready"`
	SourceAdmissionResonanceReceiverSchema                string `json:"source_admission_resonance_receiver_schema,omitempty"`
	SourceAdmissionResonanceReceiverPassed                bool   `json:"source_admission_resonance_receiver_passed"`
	SourceAdmissionResonanceReceiverID                    string `json:"source_admission_resonance_receiver_id,omitempty"`
	SourceAdmissionResonanceReceiverAction                string `json:"source_admission_resonance_receiver_action,omitempty"`
	SourceAdmissionResonanceReceiverReady                 bool   `json:"source_admission_resonance_receiver_ready"`
	SourceAdmissionResonanceReceiverCausalID              string `json:"source_admission_resonance_receiver_causal_id,omitempty"`
	SourceAdmissionResonanceReceiverPreStateHash          string `json:"source_admission_resonance_receiver_pre_state_hash,omitempty"`
	SourceAdmissionResonanceReceiverPostStateHash         string `json:"source_admission_resonance_receiver_post_state_hash,omitempty"`
	SourceAdmissionResonanceReceiverStateDeltaHash        string `json:"source_admission_resonance_receiver_state_delta_hash,omitempty"`
	SourceAdmissionResonanceIntentIDForObservation        string `json:"source_admission_resonance_intent_id_for_observation,omitempty"`
	SourceAdmissionFinalGateIDForResonanceObservation     string `json:"source_admission_final_gate_id_for_resonance_observation,omitempty"`
	SourceAdmissionSealIDForResonanceObservation          string `json:"source_admission_seal_id_for_resonance_observation,omitempty"`
	SourceAdmissionPermitIDForResonanceObservation        string `json:"source_admission_permit_id_for_resonance_observation,omitempty"`
	SourceAdmissionReadinessIDForResonanceObservation     string `json:"source_admission_readiness_id_for_resonance_observation,omitempty"`
	SourceLedgerVerificationIDForResonanceObservation     string `json:"source_ledger_verification_id_for_resonance_observation,omitempty"`
	SourceLedgerPersistenceIDForResonanceObservation      string `json:"source_ledger_persistence_id_for_resonance_observation,omitempty"`
	SourceLedgerImplementationIDForResonanceObservation   string `json:"source_ledger_implementation_id_for_resonance_observation,omitempty"`
	SourceAdmissionLedgerIDForResonanceObservation        string `json:"source_admission_ledger_id_for_resonance_observation,omitempty"`
	SourceRollbackImplementationIDForResonanceObservation string `json:"source_rollback_implementation_id_for_resonance_observation,omitempty"`
	SourceWriterReceiptIDForResonanceObservation          string `json:"source_writer_receipt_id_for_resonance_observation,omitempty"`
	AdmissionResonanceObservationID                       string `json:"admission_resonance_observation_id,omitempty"`
}

type admissionLiveRouteTurnCandidateAdmissionResonanceGraftBoundary struct {
	admissionLiveRouteTurnCandidateAdmissionResonanceObservation

	AdmissionResonanceGraftBoundaryState               string `json:"admission_resonance_graft_boundary_state,omitempty"`
	AdmissionResonanceGraftBoundaryAction              string `json:"admission_resonance_graft_boundary_action,omitempty"`
	AdmissionResonanceGraftBoundaryTarget              string `json:"admission_resonance_graft_boundary_target,omitempty"`
	AdmissionResonanceGraftBoundaryTargetKind          string `json:"admission_resonance_graft_boundary_target_kind,omitempty"`
	AdmissionResonanceGraftBoundaryTargetMode          string `json:"admission_resonance_graft_boundary_target_mode,omitempty"`
	AdmissionResonanceGraftBoundaryReceiptShape        string `json:"admission_resonance_graft_boundary_receipt_shape,omitempty"`
	AdmissionResonanceGraftBoundaryDryRunOnly          bool   `json:"admission_resonance_graft_boundary_dry_run_only"`
	AdmissionResonanceGraftBoundaryObservationVerified bool   `json:"admission_resonance_graft_boundary_observation_verified"`
	AdmissionResonanceGraftBoundaryReceiverVerified    bool   `json:"admission_resonance_graft_boundary_receiver_verified"`
	AdmissionResonanceGraftBoundaryIntentVerified      bool   `json:"admission_resonance_graft_boundary_intent_verified"`
	AdmissionResonanceGraftBoundaryFinalGateVerified   bool   `json:"admission_resonance_graft_boundary_final_gate_verified"`
	AdmissionResonanceGraftBoundarySealVerified        bool   `json:"admission_resonance_graft_boundary_seal_verified"`
	AdmissionResonanceGraftBoundaryPermitVerified      bool   `json:"admission_resonance_graft_boundary_permit_verified"`
	AdmissionResonanceGraftBoundaryReadinessVerified   bool   `json:"admission_resonance_graft_boundary_readiness_verified"`
	AdmissionResonanceGraftBoundaryLedgerVerified      bool   `json:"admission_resonance_graft_boundary_ledger_verified"`
	AdmissionResonanceGraftBoundaryWriterReady         bool   `json:"admission_resonance_graft_boundary_writer_ready"`
	AdmissionResonanceGraftBoundaryRollbackReady       bool   `json:"admission_resonance_graft_boundary_rollback_ready"`
	AdmissionResonanceGraftBoundaryLedgerReady         bool   `json:"admission_resonance_graft_boundary_ledger_ready"`
	AdmissionResonanceGraftBoundaryKind                string `json:"admission_resonance_graft_boundary_kind,omitempty"`
	AdmissionResonanceGraftBoundaryMode                string `json:"admission_resonance_graft_boundary_mode,omitempty"`
	AdmissionResonanceGraftBoundaryStage               string `json:"admission_resonance_graft_boundary_stage,omitempty"`
	AdmissionResonanceGraftBoundaryCausalID            string `json:"admission_resonance_graft_boundary_causal_id,omitempty"`
	AdmissionResonanceGraftBoundaryHash                string `json:"admission_resonance_graft_boundary_hash,omitempty"`
	AdmissionResonanceGraftBoundaryReadBackHash        string `json:"admission_resonance_graft_boundary_read_back_hash,omitempty"`
	AdmissionResonanceGraftBoundaryShadowOnly          bool   `json:"admission_resonance_graft_boundary_shadow_only"`
	AdmissionResonanceGraftBoundaryGraftAllowed        bool   `json:"admission_resonance_graft_boundary_graft_allowed"`
	AdmissionResonanceGraftBoundaryRawDreamTextAllowed bool   `json:"admission_resonance_graft_boundary_raw_dream_text_allowed"`
	AdmissionResonanceGraftBoundaryJanusSurfaceAllowed bool   `json:"admission_resonance_graft_boundary_janus_surface_allowed"`
	AdmissionResonanceGraftBoundaryCoocLearningAllowed bool   `json:"admission_resonance_graft_boundary_cooc_learning_allowed"`
	AdmissionResonanceGraftBoundaryDeltaHarvestAllowed bool   `json:"admission_resonance_graft_boundary_delta_harvest_allowed"`
	AdmissionResonanceGraftBoundaryBodyMutationAllowed bool   `json:"admission_resonance_graft_boundary_body_mutation_allowed"`
	AdmissionResonanceGraftBoundaryRollbackRequired    bool   `json:"admission_resonance_graft_boundary_rollback_required"`
	AdmissionResonanceGraftBoundaryReady               bool   `json:"admission_resonance_graft_boundary_ready"`
	SourceAdmissionResonanceObservationSchema          string `json:"source_admission_resonance_observation_schema,omitempty"`
	SourceAdmissionResonanceObservationPassed          bool   `json:"source_admission_resonance_observation_passed"`
	SourceAdmissionResonanceObservationID              string `json:"source_admission_resonance_observation_id,omitempty"`
	SourceAdmissionResonanceObservationAction          string `json:"source_admission_resonance_observation_action,omitempty"`
	SourceAdmissionResonanceObservationReady           bool   `json:"source_admission_resonance_observation_ready"`
	SourceAdmissionResonanceObservationCausalID        string `json:"source_admission_resonance_observation_causal_id,omitempty"`
	SourceAdmissionResonanceObservationAppendHash      string `json:"source_admission_resonance_observation_append_hash,omitempty"`
	SourceAdmissionResonanceObservationReadBackHash    string `json:"source_admission_resonance_observation_read_back_hash,omitempty"`
	SourceAdmissionResonanceReceiverIDForGraftBoundary string `json:"source_admission_resonance_receiver_id_for_graft_boundary,omitempty"`
	SourceAdmissionResonanceIntentIDForGraftBoundary   string `json:"source_admission_resonance_intent_id_for_graft_boundary,omitempty"`
	SourceAdmissionFinalGateIDForGraftBoundary         string `json:"source_admission_final_gate_id_for_graft_boundary,omitempty"`
	SourceAdmissionSealIDForGraftBoundary              string `json:"source_admission_seal_id_for_graft_boundary,omitempty"`
	SourceAdmissionPermitIDForGraftBoundary            string `json:"source_admission_permit_id_for_graft_boundary,omitempty"`
	SourceAdmissionReadinessIDForGraftBoundary         string `json:"source_admission_readiness_id_for_graft_boundary,omitempty"`
	SourceLedgerVerificationIDForGraftBoundary         string `json:"source_ledger_verification_id_for_graft_boundary,omitempty"`
	SourceLedgerPersistenceIDForGraftBoundary          string `json:"source_ledger_persistence_id_for_graft_boundary,omitempty"`
	SourceLedgerImplementationIDForGraftBoundary       string `json:"source_ledger_implementation_id_for_graft_boundary,omitempty"`
	SourceAdmissionLedgerIDForGraftBoundary            string `json:"source_admission_ledger_id_for_graft_boundary,omitempty"`
	SourceRollbackImplementationIDForGraftBoundary     string `json:"source_rollback_implementation_id_for_graft_boundary,omitempty"`
	SourceWriterReceiptIDForGraftBoundary              string `json:"source_writer_receipt_id_for_graft_boundary,omitempty"`
	AdmissionResonanceGraftBoundaryID                  string `json:"admission_resonance_graft_boundary_id,omitempty"`
}

type admissionLiveRouteTurnCandidateAdmissionResonanceGraftPreflight struct {
	admissionLiveRouteTurnCandidateAdmissionResonanceGraftBoundary

	AdmissionResonanceGraftPreflightState                  string `json:"admission_resonance_graft_preflight_state,omitempty"`
	AdmissionResonanceGraftPreflightAction                 string `json:"admission_resonance_graft_preflight_action,omitempty"`
	AdmissionResonanceGraftPreflightTarget                 string `json:"admission_resonance_graft_preflight_target,omitempty"`
	AdmissionResonanceGraftPreflightTargetKind             string `json:"admission_resonance_graft_preflight_target_kind,omitempty"`
	AdmissionResonanceGraftPreflightTargetMode             string `json:"admission_resonance_graft_preflight_target_mode,omitempty"`
	AdmissionResonanceGraftPreflightReceiptShape           string `json:"admission_resonance_graft_preflight_receipt_shape,omitempty"`
	AdmissionResonanceGraftPreflightDryRunOnly             bool   `json:"admission_resonance_graft_preflight_dry_run_only"`
	AdmissionResonanceGraftPreflightBoundaryVerified       bool   `json:"admission_resonance_graft_preflight_boundary_verified"`
	AdmissionResonanceGraftPreflightObservationVerified    bool   `json:"admission_resonance_graft_preflight_observation_verified"`
	AdmissionResonanceGraftPreflightReceiverVerified       bool   `json:"admission_resonance_graft_preflight_receiver_verified"`
	AdmissionResonanceGraftPreflightIntentVerified         bool   `json:"admission_resonance_graft_preflight_intent_verified"`
	AdmissionResonanceGraftPreflightFinalGateVerified      bool   `json:"admission_resonance_graft_preflight_final_gate_verified"`
	AdmissionResonanceGraftPreflightSealVerified           bool   `json:"admission_resonance_graft_preflight_seal_verified"`
	AdmissionResonanceGraftPreflightPermitVerified         bool   `json:"admission_resonance_graft_preflight_permit_verified"`
	AdmissionResonanceGraftPreflightReadinessVerified      bool   `json:"admission_resonance_graft_preflight_readiness_verified"`
	AdmissionResonanceGraftPreflightLedgerVerified         bool   `json:"admission_resonance_graft_preflight_ledger_verified"`
	AdmissionResonanceGraftPreflightWriterReady            bool   `json:"admission_resonance_graft_preflight_writer_ready"`
	AdmissionResonanceGraftPreflightRollbackReady          bool   `json:"admission_resonance_graft_preflight_rollback_ready"`
	AdmissionResonanceGraftPreflightLedgerReady            bool   `json:"admission_resonance_graft_preflight_ledger_ready"`
	AdmissionResonanceGraftPreflightKind                   string `json:"admission_resonance_graft_preflight_kind,omitempty"`
	AdmissionResonanceGraftPreflightMode                   string `json:"admission_resonance_graft_preflight_mode,omitempty"`
	AdmissionResonanceGraftPreflightStage                  string `json:"admission_resonance_graft_preflight_stage,omitempty"`
	AdmissionResonanceGraftPreflightCausalID               string `json:"admission_resonance_graft_preflight_causal_id,omitempty"`
	AdmissionResonanceGraftPreflightHash                   string `json:"admission_resonance_graft_preflight_hash,omitempty"`
	AdmissionResonanceGraftPreflightReadBackHash           string `json:"admission_resonance_graft_preflight_read_back_hash,omitempty"`
	AdmissionResonanceGraftPreflightAdmissionRequired      bool   `json:"admission_resonance_graft_preflight_admission_required"`
	AdmissionResonanceGraftPreflightShadowOnly             bool   `json:"admission_resonance_graft_preflight_shadow_only"`
	AdmissionResonanceGraftPreflightGraftAllowed           bool   `json:"admission_resonance_graft_preflight_graft_allowed"`
	AdmissionResonanceGraftPreflightRawDreamTextAllowed    bool   `json:"admission_resonance_graft_preflight_raw_dream_text_allowed"`
	AdmissionResonanceGraftPreflightJanusSurfaceAllowed    bool   `json:"admission_resonance_graft_preflight_janus_surface_allowed"`
	AdmissionResonanceGraftPreflightCoocLearningAllowed    bool   `json:"admission_resonance_graft_preflight_cooc_learning_allowed"`
	AdmissionResonanceGraftPreflightDeltaHarvestAllowed    bool   `json:"admission_resonance_graft_preflight_delta_harvest_allowed"`
	AdmissionResonanceGraftPreflightBodyMutationAllowed    bool   `json:"admission_resonance_graft_preflight_body_mutation_allowed"`
	AdmissionResonanceGraftPreflightRollbackRequired       bool   `json:"admission_resonance_graft_preflight_rollback_required"`
	AdmissionResonanceGraftPreflightReady                  bool   `json:"admission_resonance_graft_preflight_ready"`
	SourceAdmissionResonanceGraftBoundarySchema            string `json:"source_admission_resonance_graft_boundary_schema,omitempty"`
	SourceAdmissionResonanceGraftBoundaryPassed            bool   `json:"source_admission_resonance_graft_boundary_passed"`
	SourceAdmissionResonanceGraftBoundaryID                string `json:"source_admission_resonance_graft_boundary_id,omitempty"`
	SourceAdmissionResonanceGraftBoundaryAction            string `json:"source_admission_resonance_graft_boundary_action,omitempty"`
	SourceAdmissionResonanceGraftBoundaryReady             bool   `json:"source_admission_resonance_graft_boundary_ready"`
	SourceAdmissionResonanceGraftBoundaryCausalID          string `json:"source_admission_resonance_graft_boundary_causal_id,omitempty"`
	SourceAdmissionResonanceGraftBoundaryHash              string `json:"source_admission_resonance_graft_boundary_hash,omitempty"`
	SourceAdmissionResonanceGraftBoundaryReadBackHash      string `json:"source_admission_resonance_graft_boundary_read_back_hash,omitempty"`
	SourceAdmissionResonanceObservationIDForGraftPreflight string `json:"source_admission_resonance_observation_id_for_graft_preflight,omitempty"`
	SourceAdmissionResonanceReceiverIDForGraftPreflight    string `json:"source_admission_resonance_receiver_id_for_graft_preflight,omitempty"`
	SourceAdmissionResonanceIntentIDForGraftPreflight      string `json:"source_admission_resonance_intent_id_for_graft_preflight,omitempty"`
	SourceAdmissionFinalGateIDForGraftPreflight            string `json:"source_admission_final_gate_id_for_graft_preflight,omitempty"`
	SourceAdmissionSealIDForGraftPreflight                 string `json:"source_admission_seal_id_for_graft_preflight,omitempty"`
	SourceAdmissionPermitIDForGraftPreflight               string `json:"source_admission_permit_id_for_graft_preflight,omitempty"`
	SourceAdmissionReadinessIDForGraftPreflight            string `json:"source_admission_readiness_id_for_graft_preflight,omitempty"`
	SourceLedgerVerificationIDForGraftPreflight            string `json:"source_ledger_verification_id_for_graft_preflight,omitempty"`
	SourceLedgerPersistenceIDForGraftPreflight             string `json:"source_ledger_persistence_id_for_graft_preflight,omitempty"`
	SourceLedgerImplementationIDForGraftPreflight          string `json:"source_ledger_implementation_id_for_graft_preflight,omitempty"`
	SourceAdmissionLedgerIDForGraftPreflight               string `json:"source_admission_ledger_id_for_graft_preflight,omitempty"`
	SourceRollbackImplementationIDForGraftPreflight        string `json:"source_rollback_implementation_id_for_graft_preflight,omitempty"`
	SourceWriterReceiptIDForGraftPreflight                 string `json:"source_writer_receipt_id_for_graft_preflight,omitempty"`
	AdmissionResonanceGraftPreflightID                     string `json:"admission_resonance_graft_preflight_id,omitempty"`
}

type admissionLiveRouteTurnCandidateAdmissionResonanceGraftGate struct {
	admissionLiveRouteTurnCandidateAdmissionResonanceGraftPreflight

	AdmissionResonanceGraftGateState                    string `json:"admission_resonance_graft_gate_state,omitempty"`
	AdmissionResonanceGraftGateAction                   string `json:"admission_resonance_graft_gate_action,omitempty"`
	AdmissionResonanceGraftGateTarget                   string `json:"admission_resonance_graft_gate_target,omitempty"`
	AdmissionResonanceGraftGateTargetKind               string `json:"admission_resonance_graft_gate_target_kind,omitempty"`
	AdmissionResonanceGraftGateTargetMode               string `json:"admission_resonance_graft_gate_target_mode,omitempty"`
	AdmissionResonanceGraftGateReceiptShape             string `json:"admission_resonance_graft_gate_receipt_shape,omitempty"`
	AdmissionResonanceGraftGateDryRunOnly               bool   `json:"admission_resonance_graft_gate_dry_run_only"`
	AdmissionResonanceGraftGatePreflightVerified        bool   `json:"admission_resonance_graft_gate_preflight_verified"`
	AdmissionResonanceGraftGateBoundaryVerified         bool   `json:"admission_resonance_graft_gate_boundary_verified"`
	AdmissionResonanceGraftGateObservationVerified      bool   `json:"admission_resonance_graft_gate_observation_verified"`
	AdmissionResonanceGraftGateReceiverVerified         bool   `json:"admission_resonance_graft_gate_receiver_verified"`
	AdmissionResonanceGraftGateIntentVerified           bool   `json:"admission_resonance_graft_gate_intent_verified"`
	AdmissionResonanceGraftGateFinalGateVerified        bool   `json:"admission_resonance_graft_gate_final_gate_verified"`
	AdmissionResonanceGraftGateSealVerified             bool   `json:"admission_resonance_graft_gate_seal_verified"`
	AdmissionResonanceGraftGatePermitVerified           bool   `json:"admission_resonance_graft_gate_permit_verified"`
	AdmissionResonanceGraftGateReadinessVerified        bool   `json:"admission_resonance_graft_gate_readiness_verified"`
	AdmissionResonanceGraftGateLedgerVerified           bool   `json:"admission_resonance_graft_gate_ledger_verified"`
	AdmissionResonanceGraftGateWriterReady              bool   `json:"admission_resonance_graft_gate_writer_ready"`
	AdmissionResonanceGraftGateRollbackReady            bool   `json:"admission_resonance_graft_gate_rollback_ready"`
	AdmissionResonanceGraftGateLedgerReady              bool   `json:"admission_resonance_graft_gate_ledger_ready"`
	AdmissionResonanceGraftGateKind                     string `json:"admission_resonance_graft_gate_kind,omitempty"`
	AdmissionResonanceGraftGateMode                     string `json:"admission_resonance_graft_gate_mode,omitempty"`
	AdmissionResonanceGraftGateStage                    string `json:"admission_resonance_graft_gate_stage,omitempty"`
	AdmissionResonanceGraftGateCausalID                 string `json:"admission_resonance_graft_gate_causal_id,omitempty"`
	AdmissionResonanceGraftGateHash                     string `json:"admission_resonance_graft_gate_hash,omitempty"`
	AdmissionResonanceGraftGateReadBackHash             string `json:"admission_resonance_graft_gate_read_back_hash,omitempty"`
	AdmissionResonanceGraftGateAdmissionRequired        bool   `json:"admission_resonance_graft_gate_admission_required"`
	AdmissionResonanceGraftGateShadowOnly               bool   `json:"admission_resonance_graft_gate_shadow_only"`
	AdmissionResonanceGraftGateGraftAllowed             bool   `json:"admission_resonance_graft_gate_graft_allowed"`
	AdmissionResonanceGraftGateRawDreamTextAllowed      bool   `json:"admission_resonance_graft_gate_raw_dream_text_allowed"`
	AdmissionResonanceGraftGateJanusSurfaceAllowed      bool   `json:"admission_resonance_graft_gate_janus_surface_allowed"`
	AdmissionResonanceGraftGateCoocLearningAllowed      bool   `json:"admission_resonance_graft_gate_cooc_learning_allowed"`
	AdmissionResonanceGraftGateDeltaHarvestAllowed      bool   `json:"admission_resonance_graft_gate_delta_harvest_allowed"`
	AdmissionResonanceGraftGateBodyMutationAllowed      bool   `json:"admission_resonance_graft_gate_body_mutation_allowed"`
	AdmissionResonanceGraftGateRollbackRequired         bool   `json:"admission_resonance_graft_gate_rollback_required"`
	AdmissionResonanceGraftGateReady                    bool   `json:"admission_resonance_graft_gate_ready"`
	SourceAdmissionResonanceGraftPreflightSchema        string `json:"source_admission_resonance_graft_preflight_schema,omitempty"`
	SourceAdmissionResonanceGraftPreflightPassed        bool   `json:"source_admission_resonance_graft_preflight_passed"`
	SourceAdmissionResonanceGraftPreflightID            string `json:"source_admission_resonance_graft_preflight_id,omitempty"`
	SourceAdmissionResonanceGraftPreflightAction        string `json:"source_admission_resonance_graft_preflight_action,omitempty"`
	SourceAdmissionResonanceGraftPreflightReady         bool   `json:"source_admission_resonance_graft_preflight_ready"`
	SourceAdmissionResonanceGraftPreflightCausalID      string `json:"source_admission_resonance_graft_preflight_causal_id,omitempty"`
	SourceAdmissionResonanceGraftPreflightHash          string `json:"source_admission_resonance_graft_preflight_hash,omitempty"`
	SourceAdmissionResonanceGraftPreflightReadBackHash  string `json:"source_admission_resonance_graft_preflight_read_back_hash,omitempty"`
	SourceAdmissionResonanceGraftBoundaryIDForGraftGate string `json:"source_admission_resonance_graft_boundary_id_for_graft_gate,omitempty"`
	SourceAdmissionResonanceObservationIDForGraftGate   string `json:"source_admission_resonance_observation_id_for_graft_gate,omitempty"`
	SourceAdmissionResonanceReceiverIDForGraftGate      string `json:"source_admission_resonance_receiver_id_for_graft_gate,omitempty"`
	SourceAdmissionResonanceIntentIDForGraftGate        string `json:"source_admission_resonance_intent_id_for_graft_gate,omitempty"`
	SourceAdmissionFinalGateIDForGraftGate              string `json:"source_admission_final_gate_id_for_graft_gate,omitempty"`
	SourceAdmissionSealIDForGraftGate                   string `json:"source_admission_seal_id_for_graft_gate,omitempty"`
	SourceAdmissionPermitIDForGraftGate                 string `json:"source_admission_permit_id_for_graft_gate,omitempty"`
	SourceAdmissionReadinessIDForGraftGate              string `json:"source_admission_readiness_id_for_graft_gate,omitempty"`
	SourceLedgerVerificationIDForGraftGate              string `json:"source_ledger_verification_id_for_graft_gate,omitempty"`
	SourceLedgerPersistenceIDForGraftGate               string `json:"source_ledger_persistence_id_for_graft_gate,omitempty"`
	SourceLedgerImplementationIDForGraftGate            string `json:"source_ledger_implementation_id_for_graft_gate,omitempty"`
	SourceAdmissionLedgerIDForGraftGate                 string `json:"source_admission_ledger_id_for_graft_gate,omitempty"`
	SourceRollbackImplementationIDForGraftGate          string `json:"source_rollback_implementation_id_for_graft_gate,omitempty"`
	SourceWriterReceiptIDForGraftGate                   string `json:"source_writer_receipt_id_for_graft_gate,omitempty"`
	AdmissionResonanceGraftGateID                       string `json:"admission_resonance_graft_gate_id,omitempty"`
}

type admissionLiveRouteTurnCandidateAdmissionResonanceGraftCandidate struct {
	admissionLiveRouteTurnCandidateAdmissionResonanceGraftGate

	AdmissionResonanceGraftCandidateState                string `json:"admission_resonance_graft_candidate_state,omitempty"`
	AdmissionResonanceGraftCandidateAction               string `json:"admission_resonance_graft_candidate_action,omitempty"`
	AdmissionResonanceGraftCandidateTarget               string `json:"admission_resonance_graft_candidate_target,omitempty"`
	AdmissionResonanceGraftCandidateTargetKind           string `json:"admission_resonance_graft_candidate_target_kind,omitempty"`
	AdmissionResonanceGraftCandidateTargetMode           string `json:"admission_resonance_graft_candidate_target_mode,omitempty"`
	AdmissionResonanceGraftCandidateReceiptShape         string `json:"admission_resonance_graft_candidate_receipt_shape,omitempty"`
	AdmissionResonanceGraftCandidateDryRunOnly           bool   `json:"admission_resonance_graft_candidate_dry_run_only"`
	AdmissionResonanceGraftCandidateGateVerified         bool   `json:"admission_resonance_graft_candidate_gate_verified"`
	AdmissionResonanceGraftCandidatePreflightVerified    bool   `json:"admission_resonance_graft_candidate_preflight_verified"`
	AdmissionResonanceGraftCandidateBoundaryVerified     bool   `json:"admission_resonance_graft_candidate_boundary_verified"`
	AdmissionResonanceGraftCandidateObservationVerified  bool   `json:"admission_resonance_graft_candidate_observation_verified"`
	AdmissionResonanceGraftCandidateReceiverVerified     bool   `json:"admission_resonance_graft_candidate_receiver_verified"`
	AdmissionResonanceGraftCandidateIntentVerified       bool   `json:"admission_resonance_graft_candidate_intent_verified"`
	AdmissionResonanceGraftCandidateFinalGateVerified    bool   `json:"admission_resonance_graft_candidate_final_gate_verified"`
	AdmissionResonanceGraftCandidateSealVerified         bool   `json:"admission_resonance_graft_candidate_seal_verified"`
	AdmissionResonanceGraftCandidatePermitVerified       bool   `json:"admission_resonance_graft_candidate_permit_verified"`
	AdmissionResonanceGraftCandidateReadinessVerified    bool   `json:"admission_resonance_graft_candidate_readiness_verified"`
	AdmissionResonanceGraftCandidateLedgerVerified       bool   `json:"admission_resonance_graft_candidate_ledger_verified"`
	AdmissionResonanceGraftCandidateWriterReady          bool   `json:"admission_resonance_graft_candidate_writer_ready"`
	AdmissionResonanceGraftCandidateRollbackReady        bool   `json:"admission_resonance_graft_candidate_rollback_ready"`
	AdmissionResonanceGraftCandidateLedgerReady          bool   `json:"admission_resonance_graft_candidate_ledger_ready"`
	AdmissionResonanceGraftCandidateKind                 string `json:"admission_resonance_graft_candidate_kind,omitempty"`
	AdmissionResonanceGraftCandidateMode                 string `json:"admission_resonance_graft_candidate_mode,omitempty"`
	AdmissionResonanceGraftCandidateStage                string `json:"admission_resonance_graft_candidate_stage,omitempty"`
	AdmissionResonanceGraftCandidateCausalID             string `json:"admission_resonance_graft_candidate_causal_id,omitempty"`
	AdmissionResonanceGraftCandidateHash                 string `json:"admission_resonance_graft_candidate_hash,omitempty"`
	AdmissionResonanceGraftCandidateReadBackHash         string `json:"admission_resonance_graft_candidate_read_back_hash,omitempty"`
	AdmissionResonanceGraftCandidateAdmissionRequired    bool   `json:"admission_resonance_graft_candidate_admission_required"`
	AdmissionResonanceGraftCandidateShadowOnly           bool   `json:"admission_resonance_graft_candidate_shadow_only"`
	AdmissionResonanceGraftCandidateGraftAllowed         bool   `json:"admission_resonance_graft_candidate_graft_allowed"`
	AdmissionResonanceGraftCandidateRawDreamTextAllowed  bool   `json:"admission_resonance_graft_candidate_raw_dream_text_allowed"`
	AdmissionResonanceGraftCandidateJanusSurfaceAllowed  bool   `json:"admission_resonance_graft_candidate_janus_surface_allowed"`
	AdmissionResonanceGraftCandidateCoocLearningAllowed  bool   `json:"admission_resonance_graft_candidate_cooc_learning_allowed"`
	AdmissionResonanceGraftCandidateDeltaHarvestAllowed  bool   `json:"admission_resonance_graft_candidate_delta_harvest_allowed"`
	AdmissionResonanceGraftCandidateBodyMutationAllowed  bool   `json:"admission_resonance_graft_candidate_body_mutation_allowed"`
	AdmissionResonanceGraftCandidateRollbackRequired     bool   `json:"admission_resonance_graft_candidate_rollback_required"`
	AdmissionResonanceGraftCandidateReady                bool   `json:"admission_resonance_graft_candidate_ready"`
	SourceAdmissionResonanceGraftGateSchema              string `json:"source_admission_resonance_graft_gate_schema,omitempty"`
	SourceAdmissionResonanceGraftGatePassed              bool   `json:"source_admission_resonance_graft_gate_passed"`
	SourceAdmissionResonanceGraftGateID                  string `json:"source_admission_resonance_graft_gate_id,omitempty"`
	SourceAdmissionResonanceGraftGateAction              string `json:"source_admission_resonance_graft_gate_action,omitempty"`
	SourceAdmissionResonanceGraftGateReady               bool   `json:"source_admission_resonance_graft_gate_ready"`
	SourceAdmissionResonanceGraftGateCausalID            string `json:"source_admission_resonance_graft_gate_causal_id,omitempty"`
	SourceAdmissionResonanceGraftGateHash                string `json:"source_admission_resonance_graft_gate_hash,omitempty"`
	SourceAdmissionResonanceGraftGateReadBackHash        string `json:"source_admission_resonance_graft_gate_read_back_hash,omitempty"`
	SourceAdmissionResonanceGraftPreflightIDForCandidate string `json:"source_admission_resonance_graft_preflight_id_for_graft_candidate,omitempty"`
	SourceAdmissionResonanceGraftBoundaryIDForCandidate  string `json:"source_admission_resonance_graft_boundary_id_for_graft_candidate,omitempty"`
	SourceAdmissionResonanceObservationIDForCandidate    string `json:"source_admission_resonance_observation_id_for_graft_candidate,omitempty"`
	SourceAdmissionResonanceReceiverIDForCandidate       string `json:"source_admission_resonance_receiver_id_for_graft_candidate,omitempty"`
	SourceAdmissionResonanceIntentIDForCandidate         string `json:"source_admission_resonance_intent_id_for_graft_candidate,omitempty"`
	SourceAdmissionFinalGateIDForCandidate               string `json:"source_admission_final_gate_id_for_graft_candidate,omitempty"`
	SourceAdmissionSealIDForCandidate                    string `json:"source_admission_seal_id_for_graft_candidate,omitempty"`
	SourceAdmissionPermitIDForCandidate                  string `json:"source_admission_permit_id_for_graft_candidate,omitempty"`
	SourceAdmissionReadinessIDForCandidate               string `json:"source_admission_readiness_id_for_graft_candidate,omitempty"`
	SourceLedgerVerificationIDForCandidate               string `json:"source_ledger_verification_id_for_graft_candidate,omitempty"`
	SourceLedgerPersistenceIDForCandidate                string `json:"source_ledger_persistence_id_for_graft_candidate,omitempty"`
	SourceLedgerImplementationIDForCandidate             string `json:"source_ledger_implementation_id_for_graft_candidate,omitempty"`
	SourceAdmissionLedgerIDForCandidate                  string `json:"source_admission_ledger_id_for_graft_candidate,omitempty"`
	SourceRollbackImplementationIDForCandidate           string `json:"source_rollback_implementation_id_for_graft_candidate,omitempty"`
	SourceWriterReceiptIDForCandidate                    string `json:"source_writer_receipt_id_for_graft_candidate,omitempty"`
	AdmissionResonanceGraftCandidateID                   string `json:"admission_resonance_graft_candidate_id,omitempty"`
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

func admissionLiveRouteTurnCandidateAdmissionWriterImplementationDryRun() bool {
	return dreamAdmissionBoolEnv("AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_WRITER_IMPLEMENTATION_DRY_RUN")
}

func admissionLiveRouteTurnCandidateAdmissionWriterReceiptDryRun() bool {
	return dreamAdmissionBoolEnv("AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_WRITER_RECEIPT_DRY_RUN")
}

func admissionLiveRouteTurnCandidateAdmissionRollbackImplementationDryRun() bool {
	return dreamAdmissionBoolEnv("AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_ROLLBACK_IMPLEMENTATION_DRY_RUN")
}

func admissionLiveRouteTurnCandidateAdmissionLedgerImplementationDryRun() bool {
	return dreamAdmissionBoolEnv("AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_LEDGER_IMPLEMENTATION_DRY_RUN")
}

func admissionLiveRouteTurnCandidateAdmissionLedgerPersistenceDryRun() bool {
	return dreamAdmissionBoolEnv("AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_LEDGER_PERSISTENCE_DRY_RUN")
}

func admissionLiveRouteTurnCandidateAdmissionLedgerVerificationDryRun() bool {
	return dreamAdmissionBoolEnv("AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_LEDGER_VERIFICATION_DRY_RUN")
}

func admissionLiveRouteTurnCandidateAdmissionReadinessDryRun() bool {
	return dreamAdmissionBoolEnv("AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_READINESS_DRY_RUN")
}

func admissionLiveRouteTurnCandidateAdmissionPermitDryRun() bool {
	return dreamAdmissionBoolEnv("AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_PERMIT_DRY_RUN")
}

func admissionLiveRouteTurnCandidateAdmissionSealDryRun() bool {
	return dreamAdmissionBoolEnv("AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_SEAL_DRY_RUN")
}

func admissionLiveRouteTurnCandidateAdmissionFinalGateDryRun() bool {
	return dreamAdmissionBoolEnv("AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_FINAL_GATE_DRY_RUN")
}

func admissionLiveRouteTurnCandidateAdmissionResonanceIntentDryRun() bool {
	return dreamAdmissionBoolEnv("AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_RESONANCE_INTENT_DRY_RUN")
}

func admissionLiveRouteTurnCandidateAdmissionResonanceReceiverDryRun() bool {
	return dreamAdmissionBoolEnv("AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_RESONANCE_RECEIVER_DRY_RUN")
}

func admissionLiveRouteTurnCandidateAdmissionResonanceObservationDryRun() bool {
	return dreamAdmissionBoolEnv("AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_RESONANCE_OBSERVATION_DRY_RUN")
}

func admissionLiveRouteTurnCandidateAdmissionResonanceGraftBoundaryDryRun() bool {
	return dreamAdmissionBoolEnv("AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_RESONANCE_GRAFT_BOUNDARY_DRY_RUN")
}

func admissionLiveRouteTurnCandidateAdmissionResonanceGraftPreflightDryRun() bool {
	return dreamAdmissionBoolEnv("AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_RESONANCE_GRAFT_PREFLIGHT_DRY_RUN")
}

func admissionLiveRouteTurnCandidateAdmissionResonanceGraftGateDryRun() bool {
	return dreamAdmissionBoolEnv("AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_RESONANCE_GRAFT_GATE_DRY_RUN")
}

func admissionLiveRouteTurnCandidateAdmissionResonanceGraftCandidateDryRun() bool {
	return dreamAdmissionBoolEnv("AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_RESONANCE_GRAFT_CANDIDATE_DRY_RUN")
}

func admissionLiveRouteTurnCandidateAdmissionEnableGateKey() string {
	return strings.TrimSpace(os.Getenv("AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_ENABLE_GATE_KEY"))
}

func admissionLiveRouteTurnCandidateAdmissionPermitKey() string {
	return strings.TrimSpace(os.Getenv("AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_PERMIT_KEY"))
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

func admissionLiveRouteTurnCandidateAdmissionWriterImplementationForLedger(ledger admissionLiveRouteTurnCandidateAdmissionLedger) admissionLiveRouteTurnCandidateAdmissionWriterImplementation {
	impl := admissionLiveRouteTurnCandidateAdmissionWriterImplementation{
		Schema:                        admissionLiveRouteTurnCandidateAdmissionWriterImplSchema,
		Timing:                        "live_admission_writer_implementation",
		ImplementationState:           "blocked",
		ImplementationAction:          "reject",
		LedgerState:                   ledger.LedgerState,
		LedgerAction:                  ledger.LedgerAction,
		LedgerContract:                ledger.LedgerContract,
		LedgerMode:                    ledger.LedgerMode,
		LedgerEntryKind:               ledger.LedgerEntryKind,
		LedgerEntryStatus:             ledger.LedgerEntryStatus,
		LedgerReceiptShape:            ledger.LedgerReceiptShape,
		LedgerAppendReady:             ledger.LedgerAppendReady,
		LedgerReceiptPersisted:        ledger.LedgerReceiptPersisted,
		LedgerImplementationReady:     ledger.LedgerImplementationReady,
		ContractState:                 ledger.ContractState,
		ContractAction:                ledger.ContractAction,
		WriterContract:                ledger.WriterContract,
		RollbackContract:              ledger.RollbackContract,
		AdmissionLedgerContract:       ledger.AdmissionLedgerContract,
		WriterContractShape:           ledger.WriterContractShape,
		RollbackContractShape:         ledger.RollbackContractShape,
		LedgerContractShape:           ledger.LedgerContractShape,
		WriteScope:                    ledger.WriteScope,
		RollbackScope:                 ledger.RollbackScope,
		ContractShapeReady:            ledger.ContractShapeReady,
		SourceWriterContractPresent:   ledger.SourceWriterContractPresent,
		SourceRollbackContractPresent: ledger.SourceRollbackContractPresent,
		SourceLedgerContractPresent:   ledger.SourceLedgerContractPresent,
		WriterImplementationReady:     ledger.WriterImplementationReady,
		RollbackImplementationReady:   ledger.RollbackImplementationReady,
		ContractsReady:                false,
		WriterState:                   ledger.WriterState,
		WriterAction:                  ledger.WriterAction,
		RollbackState:                 ledger.RollbackState,
		RollbackAction:                ledger.RollbackAction,
		PromptClass:                   ledger.PromptClass,
		Route:                         ledger.Route,
		Source:                        ledger.Source,
		ExpectedSource:                ledger.ExpectedSource,
		CandidateRunID:                ledger.CandidateRunID,
		CandidateDraftID:              ledger.CandidateDraftID,
		CandidateExecutionID:          ledger.CandidateExecutionID,
		GeneratorAdapterID:            ledger.GeneratorAdapterID,
		HandoffID:                     ledger.HandoffID,
		AdmissionAdapterID:            ledger.AdmissionAdapterID,
		AdmissionDecisionID:           ledger.AdmissionDecisionID,
		AdmissionPromotionID:          ledger.AdmissionPromotionID,
		AdmissionSwitchID:             ledger.AdmissionSwitchID,
		AdmissionEnableGateID:         ledger.AdmissionEnableGateID,
		AdmissionLiveStageID:          ledger.AdmissionLiveStageID,
		AdmissionWriterPreflightID:    ledger.AdmissionWriterPreflightID,
		AdmissionWriterInventoryID:    ledger.AdmissionWriterInventoryID,
		AdmissionWriterContractID:     ledger.AdmissionWriterContractID,
		AdmissionLedgerID:             ledger.AdmissionLedgerID,
		AdmissionDecision:             ledger.AdmissionDecision,
		AdmissionPromotion:            ledger.AdmissionPromotion,
		SwitchState:                   ledger.SwitchState,
		SwitchAction:                  ledger.SwitchAction,
		EnableState:                   ledger.EnableState,
		EnableAction:                  ledger.EnableAction,
		StageState:                    ledger.StageState,
		StageAction:                   ledger.StageAction,
		InventoryState:                ledger.InventoryState,
		InventoryAction:               ledger.InventoryAction,
		DreamCandidateRunID:           ledger.DreamCandidateRunID,
		CandidateTextStatus:           ledger.CandidateTextStatus,
		CandidateTextHash:             ledger.CandidateTextHash,
		AdmissionPolicyPassed:         ledger.AdmissionPolicyPassed,
		LiveRouteChoicePassed:         ledger.LiveRouteChoicePassed,
		SourceDecisionPassed:          ledger.SourceDecisionPassed,
		SourcePromotionPassed:         ledger.SourcePromotionPassed,
		SourceSwitchPassed:            ledger.SourceSwitchPassed,
		SourceEnablePassed:            ledger.SourceEnablePassed,
		SourceStagePassed:             ledger.SourceStagePassed,
		SourceWriterPreflightPassed:   ledger.SourceWriterPreflightPassed,
		SourceWriterInventoryPassed:   ledger.SourceWriterInventoryPassed,
		SourceWriterContractPassed:    ledger.SourceWriterContractPassed,
		SourceLedgerPassed:            ledger.Passed,
		LiveReady:                     ledger.LiveReady,
		LiveAdmissionEnabled:          ledger.LiveAdmissionEnabled,
		AdmissionAllowed:              ledger.AdmissionAllowed,
		ManualEnableRequested:         ledger.ManualEnableRequested,
		EnableKeyMatched:              ledger.EnableKeyMatched,
		RequiresWriter:                ledger.RequiresWriter,
		WriterReady:                   ledger.WriterReady,
		RequiresRollback:              ledger.RequiresRollback,
		RollbackReady:                 ledger.RollbackReady,
		WriteAllowed:                  false,
		MutatesState:                  false,
		TurnTextHash:                  ledger.TurnTextHash,
	}
	if ledger.Schema == "" {
		impl.Reason = "missing_candidate_admission_ledger"
		return impl
	}
	if ledger.Schema != admissionLiveRouteTurnCandidateAdmissionLedgerSchema {
		impl.Reason = "unexpected_candidate_admission_ledger_schema " + ledger.Schema
		return impl
	}
	if !ledger.Passed {
		impl.Reason = "candidate_admission_ledger_failed"
		if ledger.Reason != "" {
			impl.Reason += ": " + ledger.Reason
		}
		return impl
	}
	if ledger.AdmissionLedgerID == "" {
		impl.Reason = "missing_candidate_admission_ledger_id"
		return impl
	}
	if wantLedgerID := admissionLiveRouteTurnCandidateAdmissionLedgerID(ledger); wantLedgerID == "" || ledger.AdmissionLedgerID != wantLedgerID {
		impl.Reason = "candidate_admission_ledger_id_mismatch"
		return impl
	}
	if ledger.LedgerState != "receipt_drafted_dry_run" {
		impl.Reason = "candidate_admission_ledger_unexpected_state"
		return impl
	}
	if ledger.LedgerAction != "append_candidate_admission_receipt_dry_run" {
		impl.Reason = "candidate_admission_ledger_unexpected_action"
		return impl
	}
	if ledger.LedgerContract != "live_admission_ledger.v1" {
		impl.Reason = "candidate_admission_ledger_unexpected_contract"
		return impl
	}
	if ledger.LedgerMode != "append_only_dry_run" {
		impl.Reason = "candidate_admission_ledger_unexpected_mode"
		return impl
	}
	if ledger.LedgerEntryKind != "dream_candidate_admission" {
		impl.Reason = "candidate_admission_ledger_unexpected_entry_kind"
		return impl
	}
	if ledger.LedgerEntryStatus != "shadow_candidate_receipt" {
		impl.Reason = "candidate_admission_ledger_unexpected_entry_status"
		return impl
	}
	if ledger.LedgerReceiptShape != "candidate_contract_provenance" {
		impl.Reason = "candidate_admission_ledger_unexpected_receipt_shape"
		return impl
	}
	if !ledger.LedgerAppendReady {
		impl.Reason = "candidate_admission_ledger_append_not_ready"
		return impl
	}
	if ledger.LedgerReceiptPersisted {
		impl.Reason = "candidate_admission_ledger_receipt_already_persisted"
		return impl
	}
	if ledger.LedgerImplementationReady {
		impl.Reason = "candidate_admission_ledger_impl_already_ready"
		return impl
	}
	if ledger.ContractsReady {
		impl.Reason = "candidate_admission_ledger_contracts_already_ready"
		return impl
	}
	if ledger.WriteAllowed {
		impl.Reason = "candidate_admission_ledger_already_allows_write"
		return impl
	}
	if ledger.MutatesState {
		impl.Reason = "candidate_admission_ledger_already_mutates_state"
		return impl
	}
	if ledger.LiveAdmissionEnabled {
		impl.Reason = "candidate_admission_ledger_already_live_enabled"
		return impl
	}
	if ledger.AdmissionAllowed {
		impl.Reason = "candidate_admission_ledger_already_allows_admission"
		return impl
	}
	if !ledger.LiveReady {
		impl.Reason = "candidate_admission_ledger_not_live_ready"
		return impl
	}
	if !ledger.ContractShapeReady {
		impl.Reason = "candidate_admission_ledger_contract_shape_not_ready"
		return impl
	}
	if ledger.SourceWriterContractPresent || ledger.SourceRollbackContractPresent || ledger.SourceLedgerContractPresent {
		impl.Reason = "candidate_admission_ledger_source_contract_already_present"
		return impl
	}
	if ledger.WriterImplementationReady || ledger.RollbackImplementationReady {
		impl.Reason = "candidate_admission_ledger_writer_impl_already_ready"
		return impl
	}
	if ledger.WriterContract != "live_admission_writer.v1" ||
		ledger.RollbackContract != "live_admission_rollback.v1" ||
		ledger.AdmissionLedgerContract != "live_admission_ledger.v1" ||
		ledger.WriterContractShape != "append_shadow_candidate_receipt" ||
		ledger.RollbackContractShape != "remove_exact_writer_receipt" ||
		ledger.LedgerContractShape != "append_only_receipt_log" ||
		ledger.WriteScope != "dream_candidate_admission" ||
		ledger.RollbackScope != "single_writer_receipt" {
		impl.Reason = "candidate_admission_ledger_contract_shape_mismatch"
		return impl
	}
	if !ledger.ManualEnableRequested || !ledger.EnableKeyMatched {
		impl.Reason = "candidate_admission_ledger_enable_not_armed"
		return impl
	}
	if !ledger.RequiresWriter || ledger.WriterReady || !ledger.RequiresRollback || ledger.RollbackReady {
		impl.Reason = "candidate_admission_ledger_writer_rollback_state_mismatch"
		return impl
	}
	if ledger.StageState != "staged_dry_run" || ledger.StageAction != "stage_live_candidate_dry_run" {
		impl.Reason = "candidate_admission_ledger_stage_not_staged"
		return impl
	}
	if !ledger.SourceWriterContractPassed ||
		!ledger.SourceWriterInventoryPassed ||
		!ledger.SourceWriterPreflightPassed ||
		!ledger.SourceStagePassed ||
		!ledger.SourceEnablePassed ||
		!ledger.SourceSwitchPassed ||
		!ledger.SourcePromotionPassed ||
		!ledger.SourceDecisionPassed ||
		!ledger.AdmissionPolicyPassed ||
		!ledger.LiveRouteChoicePassed {
		impl.Reason = "candidate_admission_ledger_source_not_passed"
		return impl
	}
	if ledger.AdmissionWriterContractID == "" ||
		ledger.AdmissionWriterInventoryID == "" ||
		ledger.AdmissionWriterPreflightID == "" ||
		ledger.AdmissionLiveStageID == "" ||
		ledger.AdmissionEnableGateID == "" ||
		ledger.AdmissionSwitchID == "" ||
		ledger.AdmissionPromotionID == "" ||
		ledger.AdmissionDecisionID == "" ||
		ledger.AdmissionAdapterID == "" ||
		ledger.CandidateRunID == "" ||
		ledger.CandidateDraftID == "" ||
		ledger.CandidateExecutionID == "" ||
		ledger.GeneratorAdapterID == "" ||
		ledger.HandoffID == "" ||
		ledger.DreamCandidateRunID == "" ||
		ledger.CandidateTextHash == "" ||
		ledger.TurnTextHash == "" {
		impl.Reason = "candidate_admission_ledger_missing_provenance"
		return impl
	}
	impl.ImplementationState = "implementation_contract_drafted_dry_run"
	impl.ImplementationAction = "define_append_only_writer_ledger_rollback"
	impl.WriterEntrypoint = "append_shadow_candidate_receipt_dry_run"
	impl.LedgerEntrypoint = "append_admission_ledger_receipt_dry_run"
	impl.RollbackEntrypoint = "remove_exact_shadow_candidate_receipt_dry_run"
	impl.WriteTarget = "shadow_receipt_log"
	impl.BodyTarget = "none"
	impl.AppendOnly = true
	impl.RollbackRequired = true
	impl.ImplementationContractReady = true
	impl.WriterImplementationReady = false
	impl.RollbackImplementationReady = false
	impl.LedgerImplementationReady = false
	impl.ContractsReady = false
	impl.WriterImplementationID = admissionLiveRouteTurnCandidateAdmissionWriterImplementationID(impl)
	if impl.WriterImplementationID == "" {
		impl.Reason = "missing_candidate_admission_writer_implementation_id"
		return impl
	}
	impl.Passed = true
	impl.Reason = "writer implementation contract drafted; append-only log boundary only"
	return impl
}

func admissionLiveRouteTurnCandidateAdmissionWriterImplementationID(impl admissionLiveRouteTurnCandidateAdmissionWriterImplementation) string {
	h := hashJSON(struct {
		AdmissionLedgerID           string `json:"admission_ledger_id"`
		AdmissionWriterContractID   string `json:"admission_writer_contract_id"`
		AdmissionWriterInventoryID  string `json:"admission_writer_inventory_id"`
		AdmissionWriterPreflightID  string `json:"admission_writer_preflight_id"`
		AdmissionLiveStageID        string `json:"admission_live_stage_id"`
		AdmissionEnableGateID       string `json:"admission_enable_gate_id"`
		AdmissionSwitchID           string `json:"admission_switch_id"`
		AdmissionPromotionID        string `json:"admission_promotion_id"`
		AdmissionDecisionID         string `json:"admission_decision_id"`
		AdmissionAdapterID          string `json:"admission_adapter_id"`
		CandidateRunID              string `json:"candidate_run_id"`
		CandidateTextHash           string `json:"candidate_text_hash"`
		TurnTextHash                string `json:"turn_text_hash"`
		ImplementationState         string `json:"implementation_state"`
		ImplementationAction        string `json:"implementation_action"`
		WriterEntrypoint            string `json:"writer_entrypoint"`
		LedgerEntrypoint            string `json:"ledger_entrypoint"`
		RollbackEntrypoint          string `json:"rollback_entrypoint"`
		WriteTarget                 string `json:"write_target"`
		BodyTarget                  string `json:"body_target"`
		AppendOnly                  bool   `json:"append_only"`
		RollbackRequired            bool   `json:"rollback_required"`
		ImplementationContractReady bool   `json:"implementation_contract_ready"`
		ContractsReady              bool   `json:"contracts_ready"`
		WriteAllowed                bool   `json:"write_allowed"`
		MutatesState                bool   `json:"mutates_state"`
	}{
		AdmissionLedgerID:           impl.AdmissionLedgerID,
		AdmissionWriterContractID:   impl.AdmissionWriterContractID,
		AdmissionWriterInventoryID:  impl.AdmissionWriterInventoryID,
		AdmissionWriterPreflightID:  impl.AdmissionWriterPreflightID,
		AdmissionLiveStageID:        impl.AdmissionLiveStageID,
		AdmissionEnableGateID:       impl.AdmissionEnableGateID,
		AdmissionSwitchID:           impl.AdmissionSwitchID,
		AdmissionPromotionID:        impl.AdmissionPromotionID,
		AdmissionDecisionID:         impl.AdmissionDecisionID,
		AdmissionAdapterID:          impl.AdmissionAdapterID,
		CandidateRunID:              impl.CandidateRunID,
		CandidateTextHash:           impl.CandidateTextHash,
		TurnTextHash:                impl.TurnTextHash,
		ImplementationState:         impl.ImplementationState,
		ImplementationAction:        impl.ImplementationAction,
		WriterEntrypoint:            impl.WriterEntrypoint,
		LedgerEntrypoint:            impl.LedgerEntrypoint,
		RollbackEntrypoint:          impl.RollbackEntrypoint,
		WriteTarget:                 impl.WriteTarget,
		BodyTarget:                  impl.BodyTarget,
		AppendOnly:                  impl.AppendOnly,
		RollbackRequired:            impl.RollbackRequired,
		ImplementationContractReady: impl.ImplementationContractReady,
		ContractsReady:              impl.ContractsReady,
		WriteAllowed:                impl.WriteAllowed,
		MutatesState:                impl.MutatesState,
	})
	if h == "" {
		return ""
	}
	return "writer-implementation-" + h
}

func recordAdmissionLiveRouteTurnCandidateAdmissionWriterImplementation(impl admissionLiveRouteTurnCandidateAdmissionWriterImplementation) error {
	path := strings.TrimSpace(os.Getenv("AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_WRITER_IMPLEMENTATION_LOG"))
	if path == "" {
		return nil
	}
	f, err := os.OpenFile(path, os.O_CREATE|os.O_APPEND|os.O_WRONLY, 0600)
	if err != nil {
		return err
	}
	enc := json.NewEncoder(f)
	err = enc.Encode(impl)
	if closeErr := f.Close(); err == nil {
		err = closeErr
	}
	return err
}

func admissionLiveRouteTurnCandidateAdmissionWriterReceiptForImplementation(impl admissionLiveRouteTurnCandidateAdmissionWriterImplementation) admissionLiveRouteTurnCandidateAdmissionWriterReceipt {
	receipt := admissionLiveRouteTurnCandidateAdmissionWriterReceipt{
		admissionLiveRouteTurnCandidateAdmissionWriterImplementation: impl,
		WriterReceiptState:                     "blocked",
		WriterReceiptAction:                    "reject",
		SourceWriterImplementationPassed:       impl.Passed,
		SourceWriterImplementationID:           impl.WriterImplementationID,
		SourceWriterImplementationEntrypoint:   impl.WriterEntrypoint,
		SourceLedgerImplementationEntrypoint:   impl.LedgerEntrypoint,
		SourceRollbackImplementationEntrypoint: impl.RollbackEntrypoint,
	}
	receipt.Schema = admissionLiveRouteTurnCandidateAdmissionWriterReceiptSchema
	receipt.Timing = "live_admission_writer_receipt"
	receipt.Passed = false
	receipt.WriterReceiptID = ""
	receipt.WriterReceiptPersisted = false
	receipt.ShadowWriteAllowed = false
	receipt.WriteAllowed = false
	receipt.MutatesState = false

	if impl.Schema == "" {
		receipt.Reason = "missing_candidate_admission_writer_implementation"
		return receipt
	}
	if impl.Schema != admissionLiveRouteTurnCandidateAdmissionWriterImplSchema {
		receipt.Reason = "unexpected_candidate_admission_writer_implementation_schema " + impl.Schema
		return receipt
	}
	if !impl.Passed {
		receipt.Reason = "candidate_admission_writer_implementation_failed"
		if impl.Reason != "" {
			receipt.Reason += ": " + impl.Reason
		}
		return receipt
	}
	if impl.WriterImplementationID == "" {
		receipt.Reason = "missing_candidate_admission_writer_implementation_id"
		return receipt
	}
	if wantImplID := admissionLiveRouteTurnCandidateAdmissionWriterImplementationID(impl); wantImplID == "" || impl.WriterImplementationID != wantImplID {
		receipt.Reason = "candidate_admission_writer_implementation_id_mismatch"
		return receipt
	}
	if impl.ImplementationState != "implementation_contract_drafted_dry_run" {
		receipt.Reason = "candidate_admission_writer_implementation_unexpected_state"
		return receipt
	}
	if impl.ImplementationAction != "define_append_only_writer_ledger_rollback" {
		receipt.Reason = "candidate_admission_writer_implementation_unexpected_action"
		return receipt
	}
	if impl.WriterEntrypoint != "append_shadow_candidate_receipt_dry_run" {
		receipt.Reason = "candidate_admission_writer_implementation_unexpected_writer_entrypoint"
		return receipt
	}
	if impl.LedgerEntrypoint != "append_admission_ledger_receipt_dry_run" {
		receipt.Reason = "candidate_admission_writer_implementation_unexpected_ledger_entrypoint"
		return receipt
	}
	if impl.RollbackEntrypoint != "remove_exact_shadow_candidate_receipt_dry_run" {
		receipt.Reason = "candidate_admission_writer_implementation_unexpected_rollback_entrypoint"
		return receipt
	}
	if impl.WriteTarget != "shadow_receipt_log" {
		receipt.Reason = "candidate_admission_writer_implementation_unexpected_write_target"
		return receipt
	}
	if impl.BodyTarget != "none" {
		receipt.Reason = "candidate_admission_writer_implementation_unexpected_body_target"
		return receipt
	}
	if !impl.AppendOnly || !impl.RollbackRequired || !impl.ImplementationContractReady {
		receipt.Reason = "candidate_admission_writer_implementation_contract_not_ready"
		return receipt
	}
	if impl.LedgerState != "receipt_drafted_dry_run" ||
		impl.LedgerAction != "append_candidate_admission_receipt_dry_run" ||
		impl.LedgerContract != "live_admission_ledger.v1" ||
		impl.LedgerMode != "append_only_dry_run" ||
		impl.LedgerEntryKind != "dream_candidate_admission" ||
		impl.LedgerEntryStatus != "shadow_candidate_receipt" ||
		impl.LedgerReceiptShape != "candidate_contract_provenance" {
		receipt.Reason = "candidate_admission_writer_implementation_ledger_mismatch"
		return receipt
	}
	if !impl.LedgerAppendReady || impl.LedgerReceiptPersisted || impl.LedgerImplementationReady {
		receipt.Reason = "candidate_admission_writer_implementation_ledger_state_mismatch"
		return receipt
	}
	if impl.ContractsReady || impl.WriteAllowed || impl.MutatesState || impl.LiveAdmissionEnabled || impl.AdmissionAllowed {
		receipt.Reason = "candidate_admission_writer_implementation_already_open"
		return receipt
	}
	if !impl.LiveReady {
		receipt.Reason = "candidate_admission_writer_implementation_not_live_ready"
		return receipt
	}
	if !impl.ContractShapeReady {
		receipt.Reason = "candidate_admission_writer_implementation_contract_shape_not_ready"
		return receipt
	}
	if impl.SourceWriterContractPresent || impl.SourceRollbackContractPresent || impl.SourceLedgerContractPresent {
		receipt.Reason = "candidate_admission_writer_implementation_source_contract_already_present"
		return receipt
	}
	if impl.WriterImplementationReady || impl.RollbackImplementationReady || impl.LedgerImplementationReady {
		receipt.Reason = "candidate_admission_writer_implementation_impl_already_ready"
		return receipt
	}
	if impl.WriterContract != "live_admission_writer.v1" ||
		impl.RollbackContract != "live_admission_rollback.v1" ||
		impl.AdmissionLedgerContract != "live_admission_ledger.v1" ||
		impl.WriterContractShape != "append_shadow_candidate_receipt" ||
		impl.RollbackContractShape != "remove_exact_writer_receipt" ||
		impl.LedgerContractShape != "append_only_receipt_log" ||
		impl.WriteScope != "dream_candidate_admission" ||
		impl.RollbackScope != "single_writer_receipt" {
		receipt.Reason = "candidate_admission_writer_implementation_contract_shape_mismatch"
		return receipt
	}
	if !impl.ManualEnableRequested || !impl.EnableKeyMatched {
		receipt.Reason = "candidate_admission_writer_implementation_enable_not_armed"
		return receipt
	}
	if !impl.RequiresWriter || impl.WriterReady || !impl.RequiresRollback || impl.RollbackReady {
		receipt.Reason = "candidate_admission_writer_implementation_writer_rollback_state_mismatch"
		return receipt
	}
	if impl.WriterState != "absent" ||
		impl.WriterAction != "require_writer_contract" ||
		impl.RollbackState != "absent" ||
		impl.RollbackAction != "require_rollback_contract" {
		receipt.Reason = "candidate_admission_writer_implementation_preflight_state_mismatch"
		return receipt
	}
	if impl.StageState != "staged_dry_run" || impl.StageAction != "stage_live_candidate_dry_run" {
		receipt.Reason = "candidate_admission_writer_implementation_stage_not_staged"
		return receipt
	}
	if !impl.SourceLedgerPassed ||
		!impl.SourceWriterContractPassed ||
		!impl.SourceWriterInventoryPassed ||
		!impl.SourceWriterPreflightPassed ||
		!impl.SourceStagePassed ||
		!impl.SourceEnablePassed ||
		!impl.SourceSwitchPassed ||
		!impl.SourcePromotionPassed ||
		!impl.SourceDecisionPassed ||
		!impl.AdmissionPolicyPassed ||
		!impl.LiveRouteChoicePassed {
		receipt.Reason = "candidate_admission_writer_implementation_source_not_passed"
		return receipt
	}
	if impl.WriterImplementationID == "" ||
		impl.AdmissionLedgerID == "" ||
		impl.AdmissionWriterContractID == "" ||
		impl.AdmissionWriterInventoryID == "" ||
		impl.AdmissionWriterPreflightID == "" ||
		impl.AdmissionLiveStageID == "" ||
		impl.AdmissionEnableGateID == "" ||
		impl.AdmissionSwitchID == "" ||
		impl.AdmissionPromotionID == "" ||
		impl.AdmissionDecisionID == "" ||
		impl.AdmissionAdapterID == "" ||
		impl.CandidateRunID == "" ||
		impl.CandidateDraftID == "" ||
		impl.CandidateExecutionID == "" ||
		impl.GeneratorAdapterID == "" ||
		impl.HandoffID == "" ||
		impl.DreamCandidateRunID == "" ||
		impl.CandidateTextHash == "" ||
		impl.TurnTextHash == "" {
		receipt.Reason = "candidate_admission_writer_implementation_missing_provenance"
		return receipt
	}

	receipt.WriterReceiptState = "shadow_receipt_appended_dry_run"
	receipt.WriterReceiptAction = impl.WriterEntrypoint
	receipt.WriterReceiptKind = impl.LedgerEntryKind
	receipt.WriterReceiptTarget = impl.WriteTarget
	receipt.WriterReceiptMode = impl.LedgerMode
	receipt.WriterReceiptShape = impl.LedgerReceiptShape
	receipt.WriterReceiptPersisted = true
	receipt.ShadowWriteAllowed = true
	receipt.WriterState = "ready_dry_run"
	receipt.WriterAction = impl.WriterEntrypoint
	receipt.WriterReady = true
	receipt.WriterImplementationReady = true
	receipt.RollbackReady = false
	receipt.RollbackImplementationReady = false
	receipt.LedgerImplementationReady = false
	receipt.ContractsReady = false
	receipt.WriteAllowed = false
	receipt.AdmissionAllowed = false
	receipt.LiveAdmissionEnabled = false
	receipt.MutatesState = false
	receipt.WriterReceiptID = admissionLiveRouteTurnCandidateAdmissionWriterReceiptID(receipt)
	if receipt.WriterReceiptID == "" {
		receipt.Reason = "missing_candidate_admission_writer_receipt_id"
		return receipt
	}
	receipt.Passed = true
	receipt.Reason = "shadow writer receipt appended as dry-run; body write remains disabled"
	return receipt
}

func admissionLiveRouteTurnCandidateAdmissionWriterReceiptID(receipt admissionLiveRouteTurnCandidateAdmissionWriterReceipt) string {
	h := hashJSON(struct {
		WriterImplementationID    string `json:"writer_implementation_id"`
		AdmissionLedgerID         string `json:"admission_ledger_id"`
		AdmissionWriterContractID string `json:"admission_writer_contract_id"`
		CandidateRunID            string `json:"candidate_run_id"`
		CandidateTextHash         string `json:"candidate_text_hash"`
		TurnTextHash              string `json:"turn_text_hash"`
		WriterReceiptState        string `json:"writer_receipt_state"`
		WriterReceiptAction       string `json:"writer_receipt_action"`
		WriterReceiptKind         string `json:"writer_receipt_kind"`
		WriterReceiptTarget       string `json:"writer_receipt_target"`
		WriterReceiptMode         string `json:"writer_receipt_mode"`
		WriterReceiptShape        string `json:"writer_receipt_shape"`
		WriterReceiptPersisted    bool   `json:"writer_receipt_persisted"`
		ShadowWriteAllowed        bool   `json:"shadow_write_allowed"`
		WriterReady               bool   `json:"writer_ready"`
		WriterImplementationReady bool   `json:"writer_implementation_ready"`
		BodyTarget                string `json:"body_target"`
		WriteAllowed              bool   `json:"write_allowed"`
		MutatesState              bool   `json:"mutates_state"`
	}{
		WriterImplementationID:    receipt.WriterImplementationID,
		AdmissionLedgerID:         receipt.AdmissionLedgerID,
		AdmissionWriterContractID: receipt.AdmissionWriterContractID,
		CandidateRunID:            receipt.CandidateRunID,
		CandidateTextHash:         receipt.CandidateTextHash,
		TurnTextHash:              receipt.TurnTextHash,
		WriterReceiptState:        receipt.WriterReceiptState,
		WriterReceiptAction:       receipt.WriterReceiptAction,
		WriterReceiptKind:         receipt.WriterReceiptKind,
		WriterReceiptTarget:       receipt.WriterReceiptTarget,
		WriterReceiptMode:         receipt.WriterReceiptMode,
		WriterReceiptShape:        receipt.WriterReceiptShape,
		WriterReceiptPersisted:    receipt.WriterReceiptPersisted,
		ShadowWriteAllowed:        receipt.ShadowWriteAllowed,
		WriterReady:               receipt.WriterReady,
		WriterImplementationReady: receipt.WriterImplementationReady,
		BodyTarget:                receipt.BodyTarget,
		WriteAllowed:              receipt.WriteAllowed,
		MutatesState:              receipt.MutatesState,
	})
	if h == "" {
		return ""
	}
	return "writer-receipt-" + h
}

func recordAdmissionLiveRouteTurnCandidateAdmissionWriterReceipt(receipt admissionLiveRouteTurnCandidateAdmissionWriterReceipt) error {
	path := strings.TrimSpace(os.Getenv("AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_WRITER_RECEIPT_LOG"))
	if path == "" {
		return nil
	}
	f, err := os.OpenFile(path, os.O_CREATE|os.O_APPEND|os.O_WRONLY, 0600)
	if err != nil {
		return err
	}
	enc := json.NewEncoder(f)
	err = enc.Encode(receipt)
	if closeErr := f.Close(); err == nil {
		err = closeErr
	}
	return err
}

func admissionLiveRouteTurnCandidateAdmissionRollbackImplementationForWriterReceipt(receipt admissionLiveRouteTurnCandidateAdmissionWriterReceipt) admissionLiveRouteTurnCandidateAdmissionRollbackImplementation {
	sourceSchema := receipt.Schema
	rollback := admissionLiveRouteTurnCandidateAdmissionRollbackImplementation{
		admissionLiveRouteTurnCandidateAdmissionWriterReceipt: receipt,
		RollbackImplementationState:                           "blocked",
		RollbackImplementationAction:                          "reject",
		SourceWriterReceiptSchema:                             sourceSchema,
		SourceWriterReceiptPassed:                             receipt.Passed,
		SourceWriterReceiptID:                                 receipt.WriterReceiptID,
		SourceWriterReceiptAction:                             receipt.WriterReceiptAction,
		SourceWriterReceiptPersisted:                          receipt.WriterReceiptPersisted,
		SourceWriterReceiptShadowWritable:                     receipt.ShadowWriteAllowed,
	}
	rollback.Schema = admissionLiveRouteTurnCandidateAdmissionRollbackImplSchema
	rollback.Timing = "live_admission_rollback_implementation"
	rollback.Passed = false
	rollback.RollbackImplementationID = ""
	rollback.RollbackReady = false
	rollback.RollbackImplementationReady = false
	rollback.ContractsReady = false
	rollback.WriteAllowed = false
	rollback.AdmissionAllowed = false
	rollback.LiveAdmissionEnabled = false
	rollback.MutatesState = false
	rollback.RollbackDryRunOnly = true
	rollback.RollbackReceiptRemoved = false

	if sourceSchema == "" {
		rollback.Reason = "missing_candidate_admission_writer_receipt"
		return rollback
	}
	if sourceSchema != admissionLiveRouteTurnCandidateAdmissionWriterReceiptSchema {
		rollback.Reason = "unexpected_candidate_admission_writer_receipt_schema " + sourceSchema
		return rollback
	}
	if !receipt.Passed {
		rollback.Reason = "candidate_admission_writer_receipt_failed"
		if receipt.Reason != "" {
			rollback.Reason += ": " + receipt.Reason
		}
		return rollback
	}
	if receipt.WriterReceiptID == "" {
		rollback.Reason = "missing_candidate_admission_writer_receipt_id"
		return rollback
	}
	if wantReceiptID := admissionLiveRouteTurnCandidateAdmissionWriterReceiptID(receipt); wantReceiptID == "" || receipt.WriterReceiptID != wantReceiptID {
		rollback.Reason = "candidate_admission_writer_receipt_id_mismatch"
		return rollback
	}
	if receipt.WriterReceiptState != "shadow_receipt_appended_dry_run" {
		rollback.Reason = "candidate_admission_writer_receipt_unexpected_state"
		return rollback
	}
	if receipt.WriterReceiptAction != "append_shadow_candidate_receipt_dry_run" {
		rollback.Reason = "candidate_admission_writer_receipt_unexpected_action"
		return rollback
	}
	if receipt.WriterReceiptKind != "dream_candidate_admission" {
		rollback.Reason = "candidate_admission_writer_receipt_unexpected_kind"
		return rollback
	}
	if receipt.WriterReceiptTarget != "shadow_receipt_log" {
		rollback.Reason = "candidate_admission_writer_receipt_unexpected_target"
		return rollback
	}
	if receipt.WriterReceiptMode != "append_only_dry_run" {
		rollback.Reason = "candidate_admission_writer_receipt_unexpected_mode"
		return rollback
	}
	if receipt.WriterReceiptShape != "candidate_contract_provenance" {
		rollback.Reason = "candidate_admission_writer_receipt_unexpected_shape"
		return rollback
	}
	if !receipt.WriterReceiptPersisted || !receipt.ShadowWriteAllowed {
		rollback.Reason = "candidate_admission_writer_receipt_not_persisted"
		return rollback
	}
	if !receipt.SourceWriterImplementationPassed {
		rollback.Reason = "candidate_admission_writer_receipt_source_implementation_failed"
		return rollback
	}
	if receipt.SourceWriterImplementationID == "" ||
		receipt.SourceWriterImplementationID != receipt.WriterImplementationID ||
		receipt.SourceWriterImplementationEntrypoint != "append_shadow_candidate_receipt_dry_run" ||
		receipt.SourceLedgerImplementationEntrypoint != "append_admission_ledger_receipt_dry_run" ||
		receipt.SourceRollbackImplementationEntrypoint != "remove_exact_shadow_candidate_receipt_dry_run" {
		rollback.Reason = "candidate_admission_writer_receipt_source_implementation_mismatch"
		return rollback
	}
	if receipt.ImplementationState != "implementation_contract_drafted_dry_run" ||
		receipt.ImplementationAction != "define_append_only_writer_ledger_rollback" ||
		receipt.WriterEntrypoint != "append_shadow_candidate_receipt_dry_run" ||
		receipt.LedgerEntrypoint != "append_admission_ledger_receipt_dry_run" ||
		receipt.RollbackEntrypoint != "remove_exact_shadow_candidate_receipt_dry_run" ||
		receipt.WriteTarget != "shadow_receipt_log" ||
		receipt.BodyTarget != "none" {
		rollback.Reason = "candidate_admission_writer_receipt_implementation_mismatch"
		return rollback
	}
	if !receipt.AppendOnly || !receipt.RollbackRequired || !receipt.ImplementationContractReady {
		rollback.Reason = "candidate_admission_writer_receipt_implementation_not_ready"
		return rollback
	}
	if receipt.LedgerState != "receipt_drafted_dry_run" ||
		receipt.LedgerAction != "append_candidate_admission_receipt_dry_run" ||
		receipt.LedgerContract != "live_admission_ledger.v1" ||
		receipt.LedgerMode != "append_only_dry_run" ||
		receipt.LedgerEntryKind != "dream_candidate_admission" ||
		receipt.LedgerEntryStatus != "shadow_candidate_receipt" ||
		receipt.LedgerReceiptShape != "candidate_contract_provenance" {
		rollback.Reason = "candidate_admission_writer_receipt_ledger_mismatch"
		return rollback
	}
	if !receipt.LedgerAppendReady || receipt.LedgerReceiptPersisted || receipt.LedgerImplementationReady {
		rollback.Reason = "candidate_admission_writer_receipt_ledger_state_mismatch"
		return rollback
	}
	if !receipt.WriterReady ||
		receipt.WriterState != "ready_dry_run" ||
		receipt.WriterAction != "append_shadow_candidate_receipt_dry_run" ||
		!receipt.WriterImplementationReady ||
		receipt.RollbackReady ||
		receipt.RollbackImplementationReady ||
		receipt.LedgerImplementationReady {
		rollback.Reason = "candidate_admission_writer_receipt_writer_rollback_state_mismatch"
		return rollback
	}
	if receipt.ContractsReady || receipt.WriteAllowed || receipt.MutatesState || receipt.LiveAdmissionEnabled || receipt.AdmissionAllowed {
		rollback.Reason = "candidate_admission_writer_receipt_already_open"
		return rollback
	}
	if !receipt.LiveReady {
		rollback.Reason = "candidate_admission_writer_receipt_not_live_ready"
		return rollback
	}
	if !receipt.ContractShapeReady {
		rollback.Reason = "candidate_admission_writer_receipt_contract_shape_not_ready"
		return rollback
	}
	if receipt.SourceWriterContractPresent || receipt.SourceRollbackContractPresent || receipt.SourceLedgerContractPresent {
		rollback.Reason = "candidate_admission_writer_receipt_source_contract_already_present"
		return rollback
	}
	if receipt.WriterContract != "live_admission_writer.v1" ||
		receipt.RollbackContract != "live_admission_rollback.v1" ||
		receipt.AdmissionLedgerContract != "live_admission_ledger.v1" ||
		receipt.WriterContractShape != "append_shadow_candidate_receipt" ||
		receipt.RollbackContractShape != "remove_exact_writer_receipt" ||
		receipt.LedgerContractShape != "append_only_receipt_log" ||
		receipt.WriteScope != "dream_candidate_admission" ||
		receipt.RollbackScope != "single_writer_receipt" {
		rollback.Reason = "candidate_admission_writer_receipt_contract_shape_mismatch"
		return rollback
	}
	if !receipt.ManualEnableRequested || !receipt.EnableKeyMatched {
		rollback.Reason = "candidate_admission_writer_receipt_enable_not_armed"
		return rollback
	}
	if !receipt.RequiresWriter || !receipt.RequiresRollback {
		rollback.Reason = "candidate_admission_writer_receipt_requirements_mismatch"
		return rollback
	}
	if receipt.RollbackState != "absent" || receipt.RollbackAction != "require_rollback_contract" {
		rollback.Reason = "candidate_admission_writer_receipt_preflight_state_mismatch"
		return rollback
	}
	if receipt.StageState != "staged_dry_run" || receipt.StageAction != "stage_live_candidate_dry_run" {
		rollback.Reason = "candidate_admission_writer_receipt_stage_not_staged"
		return rollback
	}
	if !receipt.SourceLedgerPassed ||
		!receipt.SourceWriterContractPassed ||
		!receipt.SourceWriterInventoryPassed ||
		!receipt.SourceWriterPreflightPassed ||
		!receipt.SourceStagePassed ||
		!receipt.SourceEnablePassed ||
		!receipt.SourceSwitchPassed ||
		!receipt.SourcePromotionPassed ||
		!receipt.SourceDecisionPassed ||
		!receipt.AdmissionPolicyPassed ||
		!receipt.LiveRouteChoicePassed {
		rollback.Reason = "candidate_admission_writer_receipt_source_not_passed"
		return rollback
	}
	if receipt.WriterReceiptID == "" ||
		receipt.WriterImplementationID == "" ||
		receipt.AdmissionLedgerID == "" ||
		receipt.AdmissionWriterContractID == "" ||
		receipt.AdmissionWriterInventoryID == "" ||
		receipt.AdmissionWriterPreflightID == "" ||
		receipt.AdmissionLiveStageID == "" ||
		receipt.AdmissionEnableGateID == "" ||
		receipt.AdmissionSwitchID == "" ||
		receipt.AdmissionPromotionID == "" ||
		receipt.AdmissionDecisionID == "" ||
		receipt.AdmissionAdapterID == "" ||
		receipt.CandidateRunID == "" ||
		receipt.CandidateDraftID == "" ||
		receipt.CandidateExecutionID == "" ||
		receipt.GeneratorAdapterID == "" ||
		receipt.HandoffID == "" ||
		receipt.DreamCandidateRunID == "" ||
		receipt.CandidateTextHash == "" ||
		receipt.TurnTextHash == "" {
		rollback.Reason = "candidate_admission_writer_receipt_missing_provenance"
		return rollback
	}

	rollback.RollbackImplementationState = "rollback_contract_drafted_dry_run"
	rollback.RollbackImplementationAction = receipt.RollbackEntrypoint
	rollback.RollbackEntrypointResolved = receipt.RollbackEntrypoint
	rollback.RollbackTarget = receipt.WriterReceiptTarget
	rollback.RollbackTargetKind = receipt.WriterReceiptKind
	rollback.RollbackTargetID = receipt.WriterReceiptID
	rollback.RollbackMode = "exact_receipt_id_dry_run"
	rollback.ExactReceiptMatchRequired = true
	rollback.RollbackDryRunOnly = true
	rollback.RollbackReceiptRemoved = false
	rollback.RollbackState = "ready_dry_run"
	rollback.RollbackAction = receipt.RollbackEntrypoint
	rollback.RollbackReady = true
	rollback.RollbackImplementationReady = true
	rollback.LedgerImplementationReady = false
	rollback.ContractsReady = false
	rollback.WriteAllowed = false
	rollback.AdmissionAllowed = false
	rollback.LiveAdmissionEnabled = false
	rollback.MutatesState = false
	rollback.RollbackImplementationID = admissionLiveRouteTurnCandidateAdmissionRollbackImplementationID(rollback)
	if rollback.RollbackImplementationID == "" {
		rollback.Reason = "missing_candidate_admission_rollback_implementation_id"
		return rollback
	}
	rollback.Passed = true
	rollback.Reason = "rollback implementation drafted for exact writer receipt; body write remains disabled"
	return rollback
}

func admissionLiveRouteTurnCandidateAdmissionRollbackImplementationID(rollback admissionLiveRouteTurnCandidateAdmissionRollbackImplementation) string {
	h := hashJSON(struct {
		WriterReceiptID              string `json:"writer_receipt_id"`
		WriterImplementationID       string `json:"writer_implementation_id"`
		AdmissionLedgerID            string `json:"admission_ledger_id"`
		CandidateRunID               string `json:"candidate_run_id"`
		CandidateTextHash            string `json:"candidate_text_hash"`
		TurnTextHash                 string `json:"turn_text_hash"`
		RollbackImplementationState  string `json:"rollback_implementation_state"`
		RollbackImplementationAction string `json:"rollback_implementation_action"`
		RollbackEntrypointResolved   string `json:"rollback_entrypoint_resolved"`
		RollbackTarget               string `json:"rollback_target"`
		RollbackTargetKind           string `json:"rollback_target_kind"`
		RollbackTargetID             string `json:"rollback_target_id"`
		RollbackMode                 string `json:"rollback_mode"`
		ExactReceiptMatchRequired    bool   `json:"exact_receipt_match_required"`
		RollbackDryRunOnly           bool   `json:"rollback_dry_run_only"`
		RollbackReceiptRemoved       bool   `json:"rollback_receipt_removed"`
		RollbackReady                bool   `json:"rollback_ready"`
		RollbackImplementationReady  bool   `json:"rollback_implementation_ready"`
		BodyTarget                   string `json:"body_target"`
		WriteAllowed                 bool   `json:"write_allowed"`
		MutatesState                 bool   `json:"mutates_state"`
	}{
		WriterReceiptID:              rollback.WriterReceiptID,
		WriterImplementationID:       rollback.WriterImplementationID,
		AdmissionLedgerID:            rollback.AdmissionLedgerID,
		CandidateRunID:               rollback.CandidateRunID,
		CandidateTextHash:            rollback.CandidateTextHash,
		TurnTextHash:                 rollback.TurnTextHash,
		RollbackImplementationState:  rollback.RollbackImplementationState,
		RollbackImplementationAction: rollback.RollbackImplementationAction,
		RollbackEntrypointResolved:   rollback.RollbackEntrypointResolved,
		RollbackTarget:               rollback.RollbackTarget,
		RollbackTargetKind:           rollback.RollbackTargetKind,
		RollbackTargetID:             rollback.RollbackTargetID,
		RollbackMode:                 rollback.RollbackMode,
		ExactReceiptMatchRequired:    rollback.ExactReceiptMatchRequired,
		RollbackDryRunOnly:           rollback.RollbackDryRunOnly,
		RollbackReceiptRemoved:       rollback.RollbackReceiptRemoved,
		RollbackReady:                rollback.RollbackReady,
		RollbackImplementationReady:  rollback.RollbackImplementationReady,
		BodyTarget:                   rollback.BodyTarget,
		WriteAllowed:                 rollback.WriteAllowed,
		MutatesState:                 rollback.MutatesState,
	})
	if h == "" {
		return ""
	}
	return "rollback-implementation-" + h
}

func recordAdmissionLiveRouteTurnCandidateAdmissionRollbackImplementation(rollback admissionLiveRouteTurnCandidateAdmissionRollbackImplementation) error {
	path := strings.TrimSpace(os.Getenv("AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_ROLLBACK_IMPLEMENTATION_LOG"))
	if path == "" {
		return nil
	}
	f, err := os.OpenFile(path, os.O_CREATE|os.O_APPEND|os.O_WRONLY, 0600)
	if err != nil {
		return err
	}
	enc := json.NewEncoder(f)
	err = enc.Encode(rollback)
	if closeErr := f.Close(); err == nil {
		err = closeErr
	}
	return err
}

func admissionLiveRouteTurnCandidateAdmissionLedgerImplementationForRollbackImplementation(rollback admissionLiveRouteTurnCandidateAdmissionRollbackImplementation) admissionLiveRouteTurnCandidateAdmissionLedgerImplementation {
	sourceSchema := rollback.Schema
	ledger := admissionLiveRouteTurnCandidateAdmissionLedgerImplementation{
		admissionLiveRouteTurnCandidateAdmissionRollbackImplementation: rollback,
		LedgerImplementationState:                                      "blocked",
		LedgerImplementationAction:                                     "reject",
		LedgerImplementationDryRunOnly:                                 true,
		SourceRollbackImplementationSchema:                             sourceSchema,
		SourceRollbackImplementationPassed:                             rollback.Passed,
		SourceRollbackImplementationID:                                 rollback.RollbackImplementationID,
		SourceRollbackImplementationAction:                             rollback.RollbackImplementationAction,
		SourceRollbackImplementationReady:                              rollback.RollbackImplementationReady,
		SourceRollbackTargetID:                                         rollback.RollbackTargetID,
		SourceWriterReceiptIDForLedger:                                 rollback.WriterReceiptID,
	}
	ledger.Schema = admissionLiveRouteTurnCandidateAdmissionLedgerImplSchema
	ledger.Timing = "live_admission_ledger_implementation"
	ledger.Passed = false
	ledger.LedgerImplementationID = ""
	ledger.LedgerImplementationReady = false
	ledger.ContractsReady = false
	ledger.WriteAllowed = false
	ledger.AdmissionAllowed = false
	ledger.LiveAdmissionEnabled = false
	ledger.MutatesState = false
	ledger.LedgerImplementationReceiptPersisted = false

	if sourceSchema == "" {
		ledger.Reason = "missing_candidate_admission_rollback_implementation"
		return ledger
	}
	if sourceSchema != admissionLiveRouteTurnCandidateAdmissionRollbackImplSchema {
		ledger.Reason = "unexpected_candidate_admission_rollback_implementation_schema " + sourceSchema
		return ledger
	}
	if !rollback.Passed {
		ledger.Reason = "candidate_admission_rollback_implementation_failed"
		if rollback.Reason != "" {
			ledger.Reason += ": " + rollback.Reason
		}
		return ledger
	}
	if rollback.RollbackImplementationID == "" {
		ledger.Reason = "missing_candidate_admission_rollback_implementation_id"
		return ledger
	}
	if wantRollbackID := admissionLiveRouteTurnCandidateAdmissionRollbackImplementationID(rollback); wantRollbackID == "" || rollback.RollbackImplementationID != wantRollbackID {
		ledger.Reason = "candidate_admission_rollback_implementation_id_mismatch"
		return ledger
	}
	if rollback.WriterReceiptID == "" {
		ledger.Reason = "missing_candidate_admission_writer_receipt_id_for_ledger"
		return ledger
	}
	if wantReceiptID := admissionLiveRouteTurnCandidateAdmissionWriterReceiptID(rollback.admissionLiveRouteTurnCandidateAdmissionWriterReceipt); wantReceiptID == "" || rollback.WriterReceiptID != wantReceiptID {
		ledger.Reason = "candidate_admission_writer_receipt_id_mismatch_for_ledger"
		return ledger
	}
	if rollback.RollbackImplementationState != "rollback_contract_drafted_dry_run" ||
		rollback.RollbackImplementationAction != "remove_exact_shadow_candidate_receipt_dry_run" ||
		rollback.RollbackEntrypointResolved != "remove_exact_shadow_candidate_receipt_dry_run" ||
		rollback.RollbackTarget != "shadow_receipt_log" ||
		rollback.RollbackTargetKind != "dream_candidate_admission" ||
		rollback.RollbackTargetID != rollback.WriterReceiptID ||
		rollback.RollbackMode != "exact_receipt_id_dry_run" {
		ledger.Reason = "candidate_admission_rollback_implementation_shape_mismatch"
		return ledger
	}
	if !rollback.ExactReceiptMatchRequired || !rollback.RollbackDryRunOnly || rollback.RollbackReceiptRemoved {
		ledger.Reason = "candidate_admission_rollback_implementation_not_exact_dry_run"
		return ledger
	}
	if rollback.SourceWriterReceiptSchema != admissionLiveRouteTurnCandidateAdmissionWriterReceiptSchema ||
		!rollback.SourceWriterReceiptPassed ||
		rollback.SourceWriterReceiptID != rollback.WriterReceiptID ||
		rollback.SourceWriterReceiptAction != "append_shadow_candidate_receipt_dry_run" ||
		!rollback.SourceWriterReceiptPersisted ||
		!rollback.SourceWriterReceiptShadowWritable {
		ledger.Reason = "candidate_admission_rollback_implementation_source_receipt_mismatch"
		return ledger
	}
	if !rollback.WriterReady ||
		rollback.WriterState != "ready_dry_run" ||
		rollback.WriterAction != "append_shadow_candidate_receipt_dry_run" ||
		!rollback.WriterImplementationReady ||
		!rollback.RollbackReady ||
		rollback.RollbackState != "ready_dry_run" ||
		rollback.RollbackAction != "remove_exact_shadow_candidate_receipt_dry_run" ||
		!rollback.RollbackImplementationReady ||
		rollback.LedgerImplementationReady {
		ledger.Reason = "candidate_admission_rollback_implementation_readiness_mismatch"
		return ledger
	}
	if rollback.ContractsReady || rollback.WriteAllowed || rollback.MutatesState || rollback.LiveAdmissionEnabled || rollback.AdmissionAllowed {
		ledger.Reason = "candidate_admission_rollback_implementation_already_open"
		return ledger
	}
	if !rollback.LiveReady {
		ledger.Reason = "candidate_admission_rollback_implementation_not_live_ready"
		return ledger
	}
	if rollback.LedgerState != "receipt_drafted_dry_run" ||
		rollback.LedgerAction != "append_candidate_admission_receipt_dry_run" ||
		rollback.LedgerContract != "live_admission_ledger.v1" ||
		rollback.LedgerMode != "append_only_dry_run" ||
		rollback.LedgerEntryKind != "dream_candidate_admission" ||
		rollback.LedgerEntryStatus != "shadow_candidate_receipt" ||
		rollback.LedgerReceiptShape != "candidate_contract_provenance" {
		ledger.Reason = "candidate_admission_rollback_implementation_ledger_mismatch"
		return ledger
	}
	if !rollback.LedgerAppendReady || rollback.LedgerReceiptPersisted {
		ledger.Reason = "candidate_admission_rollback_implementation_ledger_state_mismatch"
		return ledger
	}
	if rollback.ImplementationState != "implementation_contract_drafted_dry_run" ||
		rollback.ImplementationAction != "define_append_only_writer_ledger_rollback" ||
		rollback.WriterEntrypoint != "append_shadow_candidate_receipt_dry_run" ||
		rollback.LedgerEntrypoint != "append_admission_ledger_receipt_dry_run" ||
		rollback.RollbackEntrypoint != "remove_exact_shadow_candidate_receipt_dry_run" ||
		rollback.WriteTarget != "shadow_receipt_log" ||
		rollback.BodyTarget != "none" {
		ledger.Reason = "candidate_admission_rollback_implementation_writer_contract_mismatch"
		return ledger
	}
	if !rollback.AppendOnly || !rollback.RollbackRequired || !rollback.ImplementationContractReady {
		ledger.Reason = "candidate_admission_rollback_implementation_writer_contract_not_ready"
		return ledger
	}
	if rollback.WriterReceiptState != "shadow_receipt_appended_dry_run" ||
		rollback.WriterReceiptAction != "append_shadow_candidate_receipt_dry_run" ||
		rollback.WriterReceiptKind != "dream_candidate_admission" ||
		rollback.WriterReceiptTarget != "shadow_receipt_log" ||
		rollback.WriterReceiptMode != "append_only_dry_run" ||
		rollback.WriterReceiptShape != "candidate_contract_provenance" ||
		!rollback.WriterReceiptPersisted ||
		!rollback.ShadowWriteAllowed {
		ledger.Reason = "candidate_admission_rollback_implementation_writer_receipt_mismatch"
		return ledger
	}
	if !rollback.SourceWriterImplementationPassed ||
		rollback.SourceWriterImplementationID != rollback.WriterImplementationID ||
		rollback.SourceWriterImplementationEntrypoint != "append_shadow_candidate_receipt_dry_run" ||
		rollback.SourceLedgerImplementationEntrypoint != "append_admission_ledger_receipt_dry_run" ||
		rollback.SourceRollbackImplementationEntrypoint != "remove_exact_shadow_candidate_receipt_dry_run" {
		ledger.Reason = "candidate_admission_rollback_implementation_source_writer_mismatch"
		return ledger
	}
	if rollback.WriterContract != "live_admission_writer.v1" ||
		rollback.RollbackContract != "live_admission_rollback.v1" ||
		rollback.AdmissionLedgerContract != "live_admission_ledger.v1" ||
		rollback.WriterContractShape != "append_shadow_candidate_receipt" ||
		rollback.RollbackContractShape != "remove_exact_writer_receipt" ||
		rollback.LedgerContractShape != "append_only_receipt_log" ||
		rollback.WriteScope != "dream_candidate_admission" ||
		rollback.RollbackScope != "single_writer_receipt" ||
		!rollback.ContractShapeReady ||
		rollback.SourceWriterContractPresent ||
		rollback.SourceRollbackContractPresent ||
		rollback.SourceLedgerContractPresent {
		ledger.Reason = "candidate_admission_rollback_implementation_contract_shape_mismatch"
		return ledger
	}
	if !rollback.ManualEnableRequested || !rollback.EnableKeyMatched || !rollback.RequiresWriter || !rollback.RequiresRollback {
		ledger.Reason = "candidate_admission_rollback_implementation_requirements_mismatch"
		return ledger
	}
	if !rollback.SourceLedgerPassed ||
		!rollback.SourceWriterContractPassed ||
		!rollback.SourceWriterInventoryPassed ||
		!rollback.SourceWriterPreflightPassed ||
		!rollback.SourceStagePassed ||
		!rollback.SourceEnablePassed ||
		!rollback.SourceSwitchPassed ||
		!rollback.SourcePromotionPassed ||
		!rollback.SourceDecisionPassed ||
		!rollback.AdmissionPolicyPassed ||
		!rollback.LiveRouteChoicePassed {
		ledger.Reason = "candidate_admission_rollback_implementation_source_not_passed"
		return ledger
	}
	if rollback.RollbackImplementationID == "" ||
		rollback.WriterReceiptID == "" ||
		rollback.WriterImplementationID == "" ||
		rollback.AdmissionLedgerID == "" ||
		rollback.AdmissionWriterContractID == "" ||
		rollback.AdmissionWriterInventoryID == "" ||
		rollback.AdmissionWriterPreflightID == "" ||
		rollback.AdmissionLiveStageID == "" ||
		rollback.AdmissionEnableGateID == "" ||
		rollback.AdmissionSwitchID == "" ||
		rollback.AdmissionPromotionID == "" ||
		rollback.AdmissionDecisionID == "" ||
		rollback.AdmissionAdapterID == "" ||
		rollback.CandidateRunID == "" ||
		rollback.CandidateDraftID == "" ||
		rollback.CandidateExecutionID == "" ||
		rollback.GeneratorAdapterID == "" ||
		rollback.HandoffID == "" ||
		rollback.DreamCandidateRunID == "" ||
		rollback.CandidateTextHash == "" ||
		rollback.TurnTextHash == "" {
		ledger.Reason = "candidate_admission_rollback_implementation_missing_provenance"
		return ledger
	}

	ledger.LedgerImplementationState = "ledger_contract_drafted_dry_run"
	ledger.LedgerImplementationAction = rollback.LedgerEntrypoint
	ledger.LedgerEntrypointResolved = rollback.LedgerEntrypoint
	ledger.LedgerImplementationTarget = "admission_ledger"
	ledger.LedgerImplementationTargetKind = rollback.LedgerEntryKind
	ledger.LedgerImplementationTargetMode = "append_only_dry_run"
	ledger.LedgerImplementationAppendOnly = true
	ledger.LedgerImplementationDryRunOnly = true
	ledger.LedgerImplementationReceiptPersisted = false
	ledger.LedgerImplementationReady = true
	ledger.ContractsReady = false
	ledger.WriteAllowed = false
	ledger.AdmissionAllowed = false
	ledger.LiveAdmissionEnabled = false
	ledger.MutatesState = false
	ledger.LedgerImplementationID = admissionLiveRouteTurnCandidateAdmissionLedgerImplementationID(ledger)
	if ledger.LedgerImplementationID == "" {
		ledger.Reason = "missing_candidate_admission_ledger_implementation_id"
		return ledger
	}
	ledger.Passed = true
	ledger.Reason = "ledger implementation drafted for append-only admission receipts; contracts remain disabled"
	return ledger
}

func admissionLiveRouteTurnCandidateAdmissionLedgerImplementationID(ledger admissionLiveRouteTurnCandidateAdmissionLedgerImplementation) string {
	h := hashJSON(struct {
		RollbackImplementationID       string `json:"rollback_implementation_id"`
		WriterReceiptID                string `json:"writer_receipt_id"`
		WriterImplementationID         string `json:"writer_implementation_id"`
		AdmissionLedgerID              string `json:"admission_ledger_id"`
		CandidateRunID                 string `json:"candidate_run_id"`
		CandidateTextHash              string `json:"candidate_text_hash"`
		TurnTextHash                   string `json:"turn_text_hash"`
		LedgerImplementationState      string `json:"ledger_implementation_state"`
		LedgerImplementationAction     string `json:"ledger_implementation_action"`
		LedgerEntrypointResolved       string `json:"ledger_entrypoint_resolved"`
		LedgerImplementationTarget     string `json:"ledger_implementation_target"`
		LedgerImplementationTargetKind string `json:"ledger_implementation_target_kind"`
		LedgerImplementationTargetMode string `json:"ledger_implementation_target_mode"`
		LedgerImplementationAppendOnly bool   `json:"ledger_implementation_append_only"`
		LedgerImplementationDryRunOnly bool   `json:"ledger_implementation_dry_run_only"`
		LedgerImplementationReady      bool   `json:"ledger_implementation_ready"`
		WriterImplementationReady      bool   `json:"writer_implementation_ready"`
		RollbackImplementationReady    bool   `json:"rollback_implementation_ready"`
		ContractsReady                 bool   `json:"contracts_ready"`
		BodyTarget                     string `json:"body_target"`
		WriteAllowed                   bool   `json:"write_allowed"`
		MutatesState                   bool   `json:"mutates_state"`
	}{
		RollbackImplementationID:       ledger.RollbackImplementationID,
		WriterReceiptID:                ledger.WriterReceiptID,
		WriterImplementationID:         ledger.WriterImplementationID,
		AdmissionLedgerID:              ledger.AdmissionLedgerID,
		CandidateRunID:                 ledger.CandidateRunID,
		CandidateTextHash:              ledger.CandidateTextHash,
		TurnTextHash:                   ledger.TurnTextHash,
		LedgerImplementationState:      ledger.LedgerImplementationState,
		LedgerImplementationAction:     ledger.LedgerImplementationAction,
		LedgerEntrypointResolved:       ledger.LedgerEntrypointResolved,
		LedgerImplementationTarget:     ledger.LedgerImplementationTarget,
		LedgerImplementationTargetKind: ledger.LedgerImplementationTargetKind,
		LedgerImplementationTargetMode: ledger.LedgerImplementationTargetMode,
		LedgerImplementationAppendOnly: ledger.LedgerImplementationAppendOnly,
		LedgerImplementationDryRunOnly: ledger.LedgerImplementationDryRunOnly,
		LedgerImplementationReady:      ledger.LedgerImplementationReady,
		WriterImplementationReady:      ledger.WriterImplementationReady,
		RollbackImplementationReady:    ledger.RollbackImplementationReady,
		ContractsReady:                 ledger.ContractsReady,
		BodyTarget:                     ledger.BodyTarget,
		WriteAllowed:                   ledger.WriteAllowed,
		MutatesState:                   ledger.MutatesState,
	})
	if h == "" {
		return ""
	}
	return "ledger-implementation-" + h
}

func recordAdmissionLiveRouteTurnCandidateAdmissionLedgerImplementation(ledger admissionLiveRouteTurnCandidateAdmissionLedgerImplementation) error {
	path := strings.TrimSpace(os.Getenv("AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_LEDGER_IMPLEMENTATION_LOG"))
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

func admissionLiveRouteTurnCandidateAdmissionLedgerPersistenceForLedgerImplementation(ledger admissionLiveRouteTurnCandidateAdmissionLedgerImplementation) admissionLiveRouteTurnCandidateAdmissionLedgerPersistence {
	sourceSchema := ledger.Schema
	persistence := admissionLiveRouteTurnCandidateAdmissionLedgerPersistence{
		admissionLiveRouteTurnCandidateAdmissionLedgerImplementation: ledger,
		LedgerPersistenceState:                    "blocked",
		LedgerPersistenceAction:                   "reject",
		LedgerPersistenceDryRunOnly:               true,
		SourceLedgerImplementationSchema:          sourceSchema,
		SourceLedgerImplementationPassed:          ledger.Passed,
		SourceLedgerImplementationID:              ledger.LedgerImplementationID,
		SourceLedgerImplementationAction:          ledger.LedgerImplementationAction,
		SourceLedgerImplementationReady:           ledger.LedgerImplementationReady,
		SourceAdmissionLedgerIDForPersistence:     ledger.AdmissionLedgerID,
		SourceRollbackImplementationIDForLedger:   ledger.RollbackImplementationID,
		SourceWriterReceiptIDForLedgerPersistence: ledger.WriterReceiptID,
	}
	persistence.Schema = admissionLiveRouteTurnCandidateAdmissionLedgerPersistenceSchema
	persistence.Timing = "live_admission_ledger_persistence"
	persistence.Passed = false
	persistence.LedgerPersistenceID = ""
	persistence.LedgerPersistenceReady = false
	persistence.ContractsReady = false
	persistence.WriteAllowed = false
	persistence.AdmissionAllowed = false
	persistence.LiveAdmissionEnabled = false
	persistence.MutatesState = false
	persistence.LedgerPersistenceReceiptPersisted = false

	if sourceSchema == "" {
		persistence.Reason = "missing_candidate_admission_ledger_implementation"
		return persistence
	}
	if sourceSchema != admissionLiveRouteTurnCandidateAdmissionLedgerImplSchema {
		persistence.Reason = "unexpected_candidate_admission_ledger_implementation_schema " + sourceSchema
		return persistence
	}
	if !ledger.Passed {
		persistence.Reason = "candidate_admission_ledger_implementation_failed"
		if ledger.Reason != "" {
			persistence.Reason += ": " + ledger.Reason
		}
		return persistence
	}
	if ledger.LedgerImplementationID == "" {
		persistence.Reason = "missing_candidate_admission_ledger_implementation_id"
		return persistence
	}
	if wantLedgerImplID := admissionLiveRouteTurnCandidateAdmissionLedgerImplementationID(ledger); wantLedgerImplID == "" || ledger.LedgerImplementationID != wantLedgerImplID {
		persistence.Reason = "candidate_admission_ledger_implementation_id_mismatch"
		return persistence
	}
	if ledger.LedgerImplementationState != "ledger_contract_drafted_dry_run" ||
		ledger.LedgerImplementationAction != "append_admission_ledger_receipt_dry_run" ||
		ledger.LedgerEntrypointResolved != "append_admission_ledger_receipt_dry_run" ||
		ledger.LedgerImplementationTarget != "admission_ledger" ||
		ledger.LedgerImplementationTargetKind != "dream_candidate_admission" ||
		ledger.LedgerImplementationTargetMode != "append_only_dry_run" {
		persistence.Reason = "candidate_admission_ledger_implementation_shape_mismatch"
		return persistence
	}
	if !ledger.LedgerImplementationAppendOnly || !ledger.LedgerImplementationDryRunOnly || ledger.LedgerImplementationReceiptPersisted {
		persistence.Reason = "candidate_admission_ledger_implementation_not_append_only_dry_run"
		return persistence
	}
	if !ledger.LedgerImplementationReady {
		persistence.Reason = "candidate_admission_ledger_implementation_not_ready"
		return persistence
	}
	if ledger.ContractsReady || ledger.WriteAllowed || ledger.MutatesState || ledger.LiveAdmissionEnabled || ledger.AdmissionAllowed {
		persistence.Reason = "candidate_admission_ledger_implementation_already_open"
		return persistence
	}
	if !ledger.LiveReady {
		persistence.Reason = "candidate_admission_ledger_implementation_not_live_ready"
		return persistence
	}
	if ledger.AdmissionLedgerID == "" {
		persistence.Reason = "missing_candidate_admission_ledger_id_for_persistence"
		return persistence
	}
	if ledger.LedgerState != "receipt_drafted_dry_run" ||
		ledger.LedgerAction != "append_candidate_admission_receipt_dry_run" ||
		ledger.LedgerContract != "live_admission_ledger.v1" ||
		ledger.LedgerMode != "append_only_dry_run" ||
		ledger.LedgerEntryKind != "dream_candidate_admission" ||
		ledger.LedgerEntryStatus != "shadow_candidate_receipt" ||
		ledger.LedgerReceiptShape != "candidate_contract_provenance" {
		persistence.Reason = "candidate_admission_ledger_implementation_source_ledger_mismatch"
		return persistence
	}
	if !ledger.LedgerAppendReady || ledger.LedgerReceiptPersisted {
		persistence.Reason = "candidate_admission_ledger_implementation_source_ledger_state_mismatch"
		return persistence
	}
	if ledger.RollbackImplementationID == "" ||
		ledger.SourceRollbackImplementationID != ledger.RollbackImplementationID ||
		ledger.SourceRollbackImplementationSchema != admissionLiveRouteTurnCandidateAdmissionRollbackImplSchema ||
		!ledger.SourceRollbackImplementationPassed ||
		ledger.SourceRollbackImplementationAction != "remove_exact_shadow_candidate_receipt_dry_run" ||
		!ledger.SourceRollbackImplementationReady {
		persistence.Reason = "candidate_admission_ledger_implementation_source_rollback_mismatch"
		return persistence
	}
	if wantRollbackID := admissionLiveRouteTurnCandidateAdmissionRollbackImplementationID(ledger.admissionLiveRouteTurnCandidateAdmissionRollbackImplementation); wantRollbackID == "" || ledger.RollbackImplementationID != wantRollbackID {
		persistence.Reason = "candidate_admission_rollback_implementation_id_mismatch_for_ledger_persistence"
		return persistence
	}
	if ledger.WriterReceiptID == "" ||
		ledger.SourceWriterReceiptIDForLedger != ledger.WriterReceiptID ||
		ledger.SourceRollbackTargetID != ledger.WriterReceiptID {
		persistence.Reason = "candidate_admission_ledger_implementation_source_writer_receipt_mismatch"
		return persistence
	}
	if wantReceiptID := admissionLiveRouteTurnCandidateAdmissionWriterReceiptID(ledger.admissionLiveRouteTurnCandidateAdmissionWriterReceipt); wantReceiptID == "" || ledger.WriterReceiptID != wantReceiptID {
		persistence.Reason = "candidate_admission_writer_receipt_id_mismatch_for_ledger_persistence"
		return persistence
	}
	if !ledger.WriterReady ||
		ledger.WriterState != "ready_dry_run" ||
		ledger.WriterAction != "append_shadow_candidate_receipt_dry_run" ||
		!ledger.RollbackReady ||
		ledger.RollbackState != "ready_dry_run" ||
		ledger.RollbackAction != "remove_exact_shadow_candidate_receipt_dry_run" ||
		!ledger.WriterImplementationReady ||
		!ledger.RollbackImplementationReady {
		persistence.Reason = "candidate_admission_ledger_implementation_readiness_mismatch"
		return persistence
	}
	if ledger.WriterReceiptState != "shadow_receipt_appended_dry_run" ||
		ledger.WriterReceiptAction != "append_shadow_candidate_receipt_dry_run" ||
		ledger.WriterReceiptKind != "dream_candidate_admission" ||
		ledger.WriterReceiptTarget != "shadow_receipt_log" ||
		ledger.WriterReceiptMode != "append_only_dry_run" ||
		ledger.WriterReceiptShape != "candidate_contract_provenance" ||
		!ledger.WriterReceiptPersisted ||
		!ledger.ShadowWriteAllowed {
		persistence.Reason = "candidate_admission_ledger_implementation_writer_receipt_mismatch"
		return persistence
	}
	if ledger.RollbackImplementationState != "rollback_contract_drafted_dry_run" ||
		ledger.RollbackImplementationAction != "remove_exact_shadow_candidate_receipt_dry_run" ||
		ledger.RollbackEntrypointResolved != "remove_exact_shadow_candidate_receipt_dry_run" ||
		ledger.RollbackTarget != "shadow_receipt_log" ||
		ledger.RollbackTargetKind != "dream_candidate_admission" ||
		ledger.RollbackTargetID != ledger.WriterReceiptID ||
		ledger.RollbackMode != "exact_receipt_id_dry_run" ||
		!ledger.ExactReceiptMatchRequired ||
		!ledger.RollbackDryRunOnly ||
		ledger.RollbackReceiptRemoved {
		persistence.Reason = "candidate_admission_ledger_implementation_rollback_mismatch"
		return persistence
	}
	if ledger.ImplementationState != "implementation_contract_drafted_dry_run" ||
		ledger.ImplementationAction != "define_append_only_writer_ledger_rollback" ||
		ledger.WriterEntrypoint != "append_shadow_candidate_receipt_dry_run" ||
		ledger.LedgerEntrypoint != "append_admission_ledger_receipt_dry_run" ||
		ledger.RollbackEntrypoint != "remove_exact_shadow_candidate_receipt_dry_run" ||
		ledger.WriteTarget != "shadow_receipt_log" ||
		ledger.BodyTarget != "none" ||
		!ledger.AppendOnly ||
		!ledger.RollbackRequired ||
		!ledger.ImplementationContractReady {
		persistence.Reason = "candidate_admission_ledger_implementation_writer_contract_mismatch"
		return persistence
	}
	if ledger.WriterContract != "live_admission_writer.v1" ||
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
		ledger.SourceLedgerContractPresent {
		persistence.Reason = "candidate_admission_ledger_implementation_contract_shape_mismatch"
		return persistence
	}
	if !ledger.ManualEnableRequested || !ledger.EnableKeyMatched || !ledger.RequiresWriter || !ledger.RequiresRollback {
		persistence.Reason = "candidate_admission_ledger_implementation_requirements_mismatch"
		return persistence
	}
	if !ledger.SourceLedgerPassed ||
		!ledger.SourceWriterContractPassed ||
		!ledger.SourceWriterInventoryPassed ||
		!ledger.SourceWriterPreflightPassed ||
		!ledger.SourceStagePassed ||
		!ledger.SourceEnablePassed ||
		!ledger.SourceSwitchPassed ||
		!ledger.SourcePromotionPassed ||
		!ledger.SourceDecisionPassed ||
		!ledger.AdmissionPolicyPassed ||
		!ledger.LiveRouteChoicePassed {
		persistence.Reason = "candidate_admission_ledger_implementation_source_not_passed"
		return persistence
	}
	if ledger.LedgerImplementationID == "" ||
		ledger.RollbackImplementationID == "" ||
		ledger.WriterReceiptID == "" ||
		ledger.WriterImplementationID == "" ||
		ledger.AdmissionLedgerID == "" ||
		ledger.AdmissionWriterContractID == "" ||
		ledger.AdmissionWriterInventoryID == "" ||
		ledger.AdmissionWriterPreflightID == "" ||
		ledger.AdmissionLiveStageID == "" ||
		ledger.AdmissionEnableGateID == "" ||
		ledger.AdmissionSwitchID == "" ||
		ledger.AdmissionPromotionID == "" ||
		ledger.AdmissionDecisionID == "" ||
		ledger.AdmissionAdapterID == "" ||
		ledger.CandidateRunID == "" ||
		ledger.CandidateDraftID == "" ||
		ledger.CandidateExecutionID == "" ||
		ledger.GeneratorAdapterID == "" ||
		ledger.HandoffID == "" ||
		ledger.DreamCandidateRunID == "" ||
		ledger.CandidateTextHash == "" ||
		ledger.TurnTextHash == "" {
		persistence.Reason = "candidate_admission_ledger_implementation_missing_provenance"
		return persistence
	}

	persistence.LedgerPersistenceState = "ledger_receipt_persisted_dry_run"
	persistence.LedgerPersistenceAction = ledger.LedgerEntrypointResolved
	persistence.LedgerPersistenceTarget = ledger.LedgerImplementationTarget
	persistence.LedgerPersistenceTargetKind = ledger.LedgerImplementationTargetKind
	persistence.LedgerPersistenceTargetMode = ledger.LedgerImplementationTargetMode
	persistence.LedgerPersistenceReceiptShape = ledger.LedgerReceiptShape
	persistence.LedgerPersistenceAppendOnly = true
	persistence.LedgerPersistenceDryRunOnly = true
	persistence.LedgerPersistenceReceiptPersisted = true
	persistence.LedgerPersistenceReady = true
	persistence.ContractsReady = false
	persistence.WriteAllowed = false
	persistence.AdmissionAllowed = false
	persistence.LiveAdmissionEnabled = false
	persistence.MutatesState = false
	persistence.LedgerPersistenceID = admissionLiveRouteTurnCandidateAdmissionLedgerPersistenceID(persistence)
	if persistence.LedgerPersistenceID == "" {
		persistence.Reason = "missing_candidate_admission_ledger_persistence_id"
		return persistence
	}
	persistence.Passed = true
	persistence.Reason = "ledger receipt persisted to append-only dry-run log; live admission remains disabled"
	return persistence
}

func admissionLiveRouteTurnCandidateAdmissionLedgerPersistenceID(persistence admissionLiveRouteTurnCandidateAdmissionLedgerPersistence) string {
	h := hashJSON(struct {
		LedgerImplementationID            string `json:"ledger_implementation_id"`
		RollbackImplementationID          string `json:"rollback_implementation_id"`
		WriterReceiptID                   string `json:"writer_receipt_id"`
		AdmissionLedgerID                 string `json:"admission_ledger_id"`
		CandidateRunID                    string `json:"candidate_run_id"`
		CandidateTextHash                 string `json:"candidate_text_hash"`
		TurnTextHash                      string `json:"turn_text_hash"`
		LedgerPersistenceState            string `json:"ledger_persistence_state"`
		LedgerPersistenceAction           string `json:"ledger_persistence_action"`
		LedgerPersistenceTarget           string `json:"ledger_persistence_target"`
		LedgerPersistenceTargetKind       string `json:"ledger_persistence_target_kind"`
		LedgerPersistenceTargetMode       string `json:"ledger_persistence_target_mode"`
		LedgerPersistenceReceiptShape     string `json:"ledger_persistence_receipt_shape"`
		LedgerPersistenceAppendOnly       bool   `json:"ledger_persistence_append_only"`
		LedgerPersistenceDryRunOnly       bool   `json:"ledger_persistence_dry_run_only"`
		LedgerPersistenceReceiptPersisted bool   `json:"ledger_persistence_receipt_persisted"`
		LedgerPersistenceReady            bool   `json:"ledger_persistence_ready"`
		ContractsReady                    bool   `json:"contracts_ready"`
		BodyTarget                        string `json:"body_target"`
		WriteAllowed                      bool   `json:"write_allowed"`
		MutatesState                      bool   `json:"mutates_state"`
	}{
		LedgerImplementationID:            persistence.LedgerImplementationID,
		RollbackImplementationID:          persistence.RollbackImplementationID,
		WriterReceiptID:                   persistence.WriterReceiptID,
		AdmissionLedgerID:                 persistence.AdmissionLedgerID,
		CandidateRunID:                    persistence.CandidateRunID,
		CandidateTextHash:                 persistence.CandidateTextHash,
		TurnTextHash:                      persistence.TurnTextHash,
		LedgerPersistenceState:            persistence.LedgerPersistenceState,
		LedgerPersistenceAction:           persistence.LedgerPersistenceAction,
		LedgerPersistenceTarget:           persistence.LedgerPersistenceTarget,
		LedgerPersistenceTargetKind:       persistence.LedgerPersistenceTargetKind,
		LedgerPersistenceTargetMode:       persistence.LedgerPersistenceTargetMode,
		LedgerPersistenceReceiptShape:     persistence.LedgerPersistenceReceiptShape,
		LedgerPersistenceAppendOnly:       persistence.LedgerPersistenceAppendOnly,
		LedgerPersistenceDryRunOnly:       persistence.LedgerPersistenceDryRunOnly,
		LedgerPersistenceReceiptPersisted: persistence.LedgerPersistenceReceiptPersisted,
		LedgerPersistenceReady:            persistence.LedgerPersistenceReady,
		ContractsReady:                    persistence.ContractsReady,
		BodyTarget:                        persistence.BodyTarget,
		WriteAllowed:                      persistence.WriteAllowed,
		MutatesState:                      persistence.MutatesState,
	})
	if h == "" {
		return ""
	}
	return "ledger-persistence-" + h
}

func recordAdmissionLiveRouteTurnCandidateAdmissionLedgerPersistence(persistence admissionLiveRouteTurnCandidateAdmissionLedgerPersistence) error {
	path := strings.TrimSpace(os.Getenv("AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_LEDGER_PERSISTENCE_LOG"))
	if path == "" {
		return nil
	}
	f, err := os.OpenFile(path, os.O_CREATE|os.O_APPEND|os.O_WRONLY, 0600)
	if err != nil {
		return err
	}
	enc := json.NewEncoder(f)
	err = enc.Encode(persistence)
	if closeErr := f.Close(); err == nil {
		err = closeErr
	}
	return err
}

func admissionLiveRouteTurnCandidateAdmissionLedgerVerificationForLedgerPersistence(persistence admissionLiveRouteTurnCandidateAdmissionLedgerPersistence) admissionLiveRouteTurnCandidateAdmissionLedgerVerification {
	sourceSchema := persistence.Schema
	verification := admissionLiveRouteTurnCandidateAdmissionLedgerVerification{
		admissionLiveRouteTurnCandidateAdmissionLedgerPersistence: persistence,
		LedgerVerificationState:                                   "blocked",
		LedgerVerificationAction:                                  "reject",
		LedgerVerificationDryRunOnly:                              true,
		SourceLedgerPersistenceSchema:                             sourceSchema,
		SourceLedgerPersistencePassed:                             persistence.Passed,
		SourceLedgerPersistenceID:                                 persistence.LedgerPersistenceID,
		SourceLedgerPersistenceAction:                             persistence.LedgerPersistenceAction,
		SourceLedgerPersistenceReady:                              persistence.LedgerPersistenceReady,
		SourceLedgerPersistenceReceiptPersisted:                   persistence.LedgerPersistenceReceiptPersisted,
		SourceLedgerImplementationIDForVerification:               persistence.LedgerImplementationID,
		SourceAdmissionLedgerIDForVerification:                    persistence.AdmissionLedgerID,
		SourceRollbackImplementationIDForVerification:             persistence.RollbackImplementationID,
		SourceWriterReceiptIDForVerification:                      persistence.WriterReceiptID,
	}
	verification.Schema = admissionLiveRouteTurnCandidateAdmissionLedgerVerificationSchema
	verification.Timing = "live_admission_ledger_verification"
	verification.Passed = false
	verification.LedgerVerificationID = ""
	verification.LedgerVerificationReady = false
	verification.ContractsReady = false
	verification.WriteAllowed = false
	verification.AdmissionAllowed = false
	verification.LiveAdmissionEnabled = false
	verification.MutatesState = false
	verification.LedgerVerificationReceiptReadBack = false
	verification.LedgerVerificationReceiptVerified = false

	if sourceSchema == "" {
		verification.Reason = "missing_candidate_admission_ledger_persistence"
		return verification
	}
	if sourceSchema != admissionLiveRouteTurnCandidateAdmissionLedgerPersistenceSchema {
		verification.Reason = "unexpected_candidate_admission_ledger_persistence_schema " + sourceSchema
		return verification
	}
	if !persistence.Passed {
		verification.Reason = "candidate_admission_ledger_persistence_failed"
		if persistence.Reason != "" {
			verification.Reason += ": " + persistence.Reason
		}
		return verification
	}
	if persistence.LedgerPersistenceID == "" {
		verification.Reason = "missing_candidate_admission_ledger_persistence_id"
		return verification
	}
	if wantPersistenceID := admissionLiveRouteTurnCandidateAdmissionLedgerPersistenceID(persistence); wantPersistenceID == "" || persistence.LedgerPersistenceID != wantPersistenceID {
		verification.Reason = "candidate_admission_ledger_persistence_id_mismatch"
		return verification
	}
	if persistence.LedgerPersistenceState != "ledger_receipt_persisted_dry_run" ||
		persistence.LedgerPersistenceAction != "append_admission_ledger_receipt_dry_run" ||
		persistence.LedgerPersistenceTarget != "admission_ledger" ||
		persistence.LedgerPersistenceTargetKind != "dream_candidate_admission" ||
		persistence.LedgerPersistenceTargetMode != "append_only_dry_run" ||
		persistence.LedgerPersistenceReceiptShape != "candidate_contract_provenance" {
		verification.Reason = "candidate_admission_ledger_persistence_shape_mismatch"
		return verification
	}
	if !persistence.LedgerPersistenceAppendOnly || !persistence.LedgerPersistenceDryRunOnly || !persistence.LedgerPersistenceReceiptPersisted {
		verification.Reason = "candidate_admission_ledger_persistence_not_append_only_dry_run"
		return verification
	}
	if !persistence.LedgerPersistenceReady {
		verification.Reason = "candidate_admission_ledger_persistence_not_ready"
		return verification
	}
	if persistence.ContractsReady || persistence.WriteAllowed || persistence.MutatesState || persistence.LiveAdmissionEnabled || persistence.AdmissionAllowed {
		verification.Reason = "candidate_admission_ledger_persistence_already_open"
		return verification
	}
	if !persistence.LiveReady {
		verification.Reason = "candidate_admission_ledger_persistence_not_live_ready"
		return verification
	}
	if persistence.SourceLedgerImplementationSchema != admissionLiveRouteTurnCandidateAdmissionLedgerImplSchema ||
		!persistence.SourceLedgerImplementationPassed ||
		persistence.SourceLedgerImplementationID != persistence.LedgerImplementationID ||
		persistence.SourceLedgerImplementationAction != "append_admission_ledger_receipt_dry_run" ||
		!persistence.SourceLedgerImplementationReady {
		verification.Reason = "candidate_admission_ledger_persistence_source_ledger_implementation_mismatch"
		return verification
	}
	if wantLedgerImplID := admissionLiveRouteTurnCandidateAdmissionLedgerImplementationID(persistence.admissionLiveRouteTurnCandidateAdmissionLedgerImplementation); wantLedgerImplID == "" || persistence.LedgerImplementationID != wantLedgerImplID {
		verification.Reason = "candidate_admission_ledger_implementation_id_mismatch_for_ledger_verification"
		return verification
	}
	if persistence.SourceAdmissionLedgerIDForPersistence != persistence.AdmissionLedgerID ||
		persistence.SourceRollbackImplementationIDForLedger != persistence.RollbackImplementationID ||
		persistence.SourceWriterReceiptIDForLedgerPersistence != persistence.WriterReceiptID {
		verification.Reason = "candidate_admission_ledger_persistence_source_id_mismatch"
		return verification
	}
	if persistence.LedgerImplementationState != "ledger_contract_drafted_dry_run" ||
		persistence.LedgerImplementationAction != "append_admission_ledger_receipt_dry_run" ||
		persistence.LedgerEntrypointResolved != "append_admission_ledger_receipt_dry_run" ||
		persistence.LedgerImplementationTarget != "admission_ledger" ||
		persistence.LedgerImplementationTargetKind != "dream_candidate_admission" ||
		persistence.LedgerImplementationTargetMode != "append_only_dry_run" ||
		!persistence.LedgerImplementationAppendOnly ||
		!persistence.LedgerImplementationDryRunOnly ||
		persistence.LedgerImplementationReceiptPersisted ||
		!persistence.LedgerImplementationReady {
		verification.Reason = "candidate_admission_ledger_persistence_ledger_implementation_mismatch"
		return verification
	}
	if persistence.LedgerState != "receipt_drafted_dry_run" ||
		persistence.LedgerAction != "append_candidate_admission_receipt_dry_run" ||
		persistence.LedgerContract != "live_admission_ledger.v1" ||
		persistence.LedgerMode != "append_only_dry_run" ||
		persistence.LedgerEntryKind != "dream_candidate_admission" ||
		persistence.LedgerEntryStatus != "shadow_candidate_receipt" ||
		persistence.LedgerReceiptShape != "candidate_contract_provenance" ||
		!persistence.LedgerAppendReady ||
		persistence.LedgerReceiptPersisted {
		verification.Reason = "candidate_admission_ledger_persistence_source_ledger_mismatch"
		return verification
	}
	if persistence.RollbackImplementationID == "" ||
		persistence.SourceRollbackImplementationID != persistence.RollbackImplementationID ||
		persistence.SourceRollbackImplementationSchema != admissionLiveRouteTurnCandidateAdmissionRollbackImplSchema ||
		!persistence.SourceRollbackImplementationPassed ||
		persistence.SourceRollbackImplementationAction != "remove_exact_shadow_candidate_receipt_dry_run" ||
		!persistence.SourceRollbackImplementationReady {
		verification.Reason = "candidate_admission_ledger_persistence_source_rollback_mismatch"
		return verification
	}
	if wantRollbackID := admissionLiveRouteTurnCandidateAdmissionRollbackImplementationID(persistence.admissionLiveRouteTurnCandidateAdmissionLedgerImplementation.admissionLiveRouteTurnCandidateAdmissionRollbackImplementation); wantRollbackID == "" || persistence.RollbackImplementationID != wantRollbackID {
		verification.Reason = "candidate_admission_rollback_implementation_id_mismatch_for_ledger_verification"
		return verification
	}
	if persistence.WriterReceiptID == "" ||
		persistence.SourceWriterReceiptIDForLedger != persistence.WriterReceiptID ||
		persistence.SourceRollbackTargetID != persistence.WriterReceiptID {
		verification.Reason = "candidate_admission_ledger_persistence_source_writer_receipt_mismatch"
		return verification
	}
	if wantReceiptID := admissionLiveRouteTurnCandidateAdmissionWriterReceiptID(persistence.admissionLiveRouteTurnCandidateAdmissionLedgerImplementation.admissionLiveRouteTurnCandidateAdmissionRollbackImplementation.admissionLiveRouteTurnCandidateAdmissionWriterReceipt); wantReceiptID == "" || persistence.WriterReceiptID != wantReceiptID {
		verification.Reason = "candidate_admission_writer_receipt_id_mismatch_for_ledger_verification"
		return verification
	}
	if !persistence.WriterReady ||
		persistence.WriterState != "ready_dry_run" ||
		persistence.WriterAction != "append_shadow_candidate_receipt_dry_run" ||
		!persistence.RollbackReady ||
		persistence.RollbackState != "ready_dry_run" ||
		persistence.RollbackAction != "remove_exact_shadow_candidate_receipt_dry_run" ||
		!persistence.WriterImplementationReady ||
		!persistence.RollbackImplementationReady {
		verification.Reason = "candidate_admission_ledger_persistence_readiness_mismatch"
		return verification
	}
	if persistence.WriterReceiptState != "shadow_receipt_appended_dry_run" ||
		persistence.WriterReceiptAction != "append_shadow_candidate_receipt_dry_run" ||
		persistence.WriterReceiptKind != "dream_candidate_admission" ||
		persistence.WriterReceiptTarget != "shadow_receipt_log" ||
		persistence.WriterReceiptMode != "append_only_dry_run" ||
		persistence.WriterReceiptShape != "candidate_contract_provenance" ||
		!persistence.WriterReceiptPersisted ||
		!persistence.ShadowWriteAllowed {
		verification.Reason = "candidate_admission_ledger_persistence_writer_receipt_mismatch"
		return verification
	}
	if persistence.RollbackImplementationState != "rollback_contract_drafted_dry_run" ||
		persistence.RollbackImplementationAction != "remove_exact_shadow_candidate_receipt_dry_run" ||
		persistence.RollbackEntrypointResolved != "remove_exact_shadow_candidate_receipt_dry_run" ||
		persistence.RollbackTarget != "shadow_receipt_log" ||
		persistence.RollbackTargetKind != "dream_candidate_admission" ||
		persistence.RollbackTargetID != persistence.WriterReceiptID ||
		persistence.RollbackMode != "exact_receipt_id_dry_run" ||
		!persistence.ExactReceiptMatchRequired ||
		!persistence.RollbackDryRunOnly ||
		persistence.RollbackReceiptRemoved {
		verification.Reason = "candidate_admission_ledger_persistence_rollback_mismatch"
		return verification
	}
	if persistence.ImplementationState != "implementation_contract_drafted_dry_run" ||
		persistence.ImplementationAction != "define_append_only_writer_ledger_rollback" ||
		persistence.WriterEntrypoint != "append_shadow_candidate_receipt_dry_run" ||
		persistence.LedgerEntrypoint != "append_admission_ledger_receipt_dry_run" ||
		persistence.RollbackEntrypoint != "remove_exact_shadow_candidate_receipt_dry_run" ||
		persistence.WriteTarget != "shadow_receipt_log" ||
		persistence.BodyTarget != "none" ||
		!persistence.AppendOnly ||
		!persistence.RollbackRequired ||
		!persistence.ImplementationContractReady {
		verification.Reason = "candidate_admission_ledger_persistence_writer_contract_mismatch"
		return verification
	}
	if persistence.WriterContract != "live_admission_writer.v1" ||
		persistence.RollbackContract != "live_admission_rollback.v1" ||
		persistence.AdmissionLedgerContract != "live_admission_ledger.v1" ||
		persistence.WriterContractShape != "append_shadow_candidate_receipt" ||
		persistence.RollbackContractShape != "remove_exact_writer_receipt" ||
		persistence.LedgerContractShape != "append_only_receipt_log" ||
		persistence.WriteScope != "dream_candidate_admission" ||
		persistence.RollbackScope != "single_writer_receipt" ||
		!persistence.ContractShapeReady ||
		persistence.SourceWriterContractPresent ||
		persistence.SourceRollbackContractPresent ||
		persistence.SourceLedgerContractPresent {
		verification.Reason = "candidate_admission_ledger_persistence_contract_shape_mismatch"
		return verification
	}
	if !persistence.ManualEnableRequested || !persistence.EnableKeyMatched || !persistence.RequiresWriter || !persistence.RequiresRollback {
		verification.Reason = "candidate_admission_ledger_persistence_requirements_mismatch"
		return verification
	}
	if !persistence.SourceLedgerPassed ||
		!persistence.SourceWriterContractPassed ||
		!persistence.SourceWriterInventoryPassed ||
		!persistence.SourceWriterPreflightPassed ||
		!persistence.SourceStagePassed ||
		!persistence.SourceEnablePassed ||
		!persistence.SourceSwitchPassed ||
		!persistence.SourcePromotionPassed ||
		!persistence.SourceDecisionPassed ||
		!persistence.AdmissionPolicyPassed ||
		!persistence.LiveRouteChoicePassed {
		verification.Reason = "candidate_admission_ledger_persistence_source_not_passed"
		return verification
	}
	if persistence.LedgerPersistenceID == "" ||
		persistence.LedgerImplementationID == "" ||
		persistence.RollbackImplementationID == "" ||
		persistence.WriterReceiptID == "" ||
		persistence.WriterImplementationID == "" ||
		persistence.AdmissionLedgerID == "" ||
		persistence.AdmissionWriterContractID == "" ||
		persistence.AdmissionWriterInventoryID == "" ||
		persistence.AdmissionWriterPreflightID == "" ||
		persistence.AdmissionLiveStageID == "" ||
		persistence.AdmissionEnableGateID == "" ||
		persistence.AdmissionSwitchID == "" ||
		persistence.AdmissionPromotionID == "" ||
		persistence.AdmissionDecisionID == "" ||
		persistence.AdmissionAdapterID == "" ||
		persistence.CandidateRunID == "" ||
		persistence.CandidateDraftID == "" ||
		persistence.CandidateExecutionID == "" ||
		persistence.GeneratorAdapterID == "" ||
		persistence.HandoffID == "" ||
		persistence.DreamCandidateRunID == "" ||
		persistence.CandidateTextHash == "" ||
		persistence.TurnTextHash == "" {
		verification.Reason = "candidate_admission_ledger_persistence_missing_provenance"
		return verification
	}

	verification.LedgerVerificationState = "ledger_receipt_verified_dry_run"
	verification.LedgerVerificationAction = "verify_persisted_admission_ledger_receipt_dry_run"
	verification.LedgerVerificationTarget = persistence.LedgerPersistenceTarget
	verification.LedgerVerificationTargetKind = persistence.LedgerPersistenceTargetKind
	verification.LedgerVerificationTargetMode = persistence.LedgerPersistenceTargetMode
	verification.LedgerVerificationReceiptShape = persistence.LedgerPersistenceReceiptShape
	verification.LedgerVerificationAppendOnly = true
	verification.LedgerVerificationDryRunOnly = true
	verification.LedgerVerificationReceiptReadBack = true
	verification.LedgerVerificationReceiptVerified = true
	verification.LedgerVerificationReady = true
	verification.ContractsReady = false
	verification.WriteAllowed = false
	verification.AdmissionAllowed = false
	verification.LiveAdmissionEnabled = false
	verification.MutatesState = false
	verification.LedgerVerificationID = admissionLiveRouteTurnCandidateAdmissionLedgerVerificationID(verification)
	if verification.LedgerVerificationID == "" {
		verification.Reason = "missing_candidate_admission_ledger_verification_id"
		return verification
	}
	verification.Passed = true
	verification.Reason = "ledger persistence receipt verified by read-back dry-run; live admission remains disabled"
	return verification
}

func admissionLiveRouteTurnCandidateAdmissionLedgerVerificationID(verification admissionLiveRouteTurnCandidateAdmissionLedgerVerification) string {
	h := hashJSON(struct {
		LedgerPersistenceID               string `json:"ledger_persistence_id"`
		LedgerImplementationID            string `json:"ledger_implementation_id"`
		RollbackImplementationID          string `json:"rollback_implementation_id"`
		WriterReceiptID                   string `json:"writer_receipt_id"`
		AdmissionLedgerID                 string `json:"admission_ledger_id"`
		CandidateRunID                    string `json:"candidate_run_id"`
		CandidateTextHash                 string `json:"candidate_text_hash"`
		TurnTextHash                      string `json:"turn_text_hash"`
		LedgerVerificationState           string `json:"ledger_verification_state"`
		LedgerVerificationAction          string `json:"ledger_verification_action"`
		LedgerVerificationTarget          string `json:"ledger_verification_target"`
		LedgerVerificationTargetKind      string `json:"ledger_verification_target_kind"`
		LedgerVerificationTargetMode      string `json:"ledger_verification_target_mode"`
		LedgerVerificationReceiptShape    string `json:"ledger_verification_receipt_shape"`
		LedgerVerificationAppendOnly      bool   `json:"ledger_verification_append_only"`
		LedgerVerificationDryRunOnly      bool   `json:"ledger_verification_dry_run_only"`
		LedgerVerificationReceiptReadBack bool   `json:"ledger_verification_receipt_read_back"`
		LedgerVerificationReceiptVerified bool   `json:"ledger_verification_receipt_verified"`
		LedgerVerificationReady           bool   `json:"ledger_verification_ready"`
		ContractsReady                    bool   `json:"contracts_ready"`
		BodyTarget                        string `json:"body_target"`
		WriteAllowed                      bool   `json:"write_allowed"`
		MutatesState                      bool   `json:"mutates_state"`
	}{
		LedgerPersistenceID:               verification.LedgerPersistenceID,
		LedgerImplementationID:            verification.LedgerImplementationID,
		RollbackImplementationID:          verification.RollbackImplementationID,
		WriterReceiptID:                   verification.WriterReceiptID,
		AdmissionLedgerID:                 verification.AdmissionLedgerID,
		CandidateRunID:                    verification.CandidateRunID,
		CandidateTextHash:                 verification.CandidateTextHash,
		TurnTextHash:                      verification.TurnTextHash,
		LedgerVerificationState:           verification.LedgerVerificationState,
		LedgerVerificationAction:          verification.LedgerVerificationAction,
		LedgerVerificationTarget:          verification.LedgerVerificationTarget,
		LedgerVerificationTargetKind:      verification.LedgerVerificationTargetKind,
		LedgerVerificationTargetMode:      verification.LedgerVerificationTargetMode,
		LedgerVerificationReceiptShape:    verification.LedgerVerificationReceiptShape,
		LedgerVerificationAppendOnly:      verification.LedgerVerificationAppendOnly,
		LedgerVerificationDryRunOnly:      verification.LedgerVerificationDryRunOnly,
		LedgerVerificationReceiptReadBack: verification.LedgerVerificationReceiptReadBack,
		LedgerVerificationReceiptVerified: verification.LedgerVerificationReceiptVerified,
		LedgerVerificationReady:           verification.LedgerVerificationReady,
		ContractsReady:                    verification.ContractsReady,
		BodyTarget:                        verification.BodyTarget,
		WriteAllowed:                      verification.WriteAllowed,
		MutatesState:                      verification.MutatesState,
	})
	if h == "" {
		return ""
	}
	return "ledger-verification-" + h
}

func recordAdmissionLiveRouteTurnCandidateAdmissionLedgerVerification(verification admissionLiveRouteTurnCandidateAdmissionLedgerVerification) error {
	path := strings.TrimSpace(os.Getenv("AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_LEDGER_VERIFICATION_LOG"))
	if path == "" {
		return nil
	}
	f, err := os.OpenFile(path, os.O_CREATE|os.O_APPEND|os.O_WRONLY, 0600)
	if err != nil {
		return err
	}
	enc := json.NewEncoder(f)
	err = enc.Encode(verification)
	if closeErr := f.Close(); err == nil {
		err = closeErr
	}
	return err
}

func admissionLiveRouteTurnCandidateAdmissionReadinessForLedgerVerification(verification admissionLiveRouteTurnCandidateAdmissionLedgerVerification) admissionLiveRouteTurnCandidateAdmissionReadiness {
	sourceSchema := verification.Schema
	readiness := admissionLiveRouteTurnCandidateAdmissionReadiness{
		admissionLiveRouteTurnCandidateAdmissionLedgerVerification: verification,
		AdmissionReadinessState:                    "blocked",
		AdmissionReadinessAction:                   "reject",
		AdmissionReadinessDryRunOnly:               true,
		SourceLedgerVerificationSchema:             sourceSchema,
		SourceLedgerVerificationPassed:             verification.Passed,
		SourceLedgerVerificationID:                 verification.LedgerVerificationID,
		SourceLedgerVerificationAction:             verification.LedgerVerificationAction,
		SourceLedgerVerificationReady:              verification.LedgerVerificationReady,
		SourceLedgerVerificationReceiptVerified:    verification.LedgerVerificationReceiptVerified,
		SourceLedgerPersistenceIDForReadiness:      verification.LedgerPersistenceID,
		SourceLedgerImplementationIDForReadiness:   verification.LedgerImplementationID,
		SourceAdmissionLedgerIDForReadiness:        verification.AdmissionLedgerID,
		SourceRollbackImplementationIDForReadiness: verification.RollbackImplementationID,
		SourceWriterReceiptIDForReadiness:          verification.WriterReceiptID,
	}
	readiness.Schema = admissionLiveRouteTurnCandidateAdmissionReadinessSchema
	readiness.Timing = "live_admission_readiness"
	readiness.Passed = false
	readiness.AdmissionReadinessID = ""
	readiness.AdmissionReadinessReady = false
	readiness.ContractsReady = false
	readiness.WriteAllowed = false
	readiness.AdmissionAllowed = false
	readiness.LiveAdmissionEnabled = false
	readiness.MutatesState = false

	if sourceSchema == "" {
		readiness.Reason = "missing_candidate_admission_ledger_verification"
		return readiness
	}
	if sourceSchema != admissionLiveRouteTurnCandidateAdmissionLedgerVerificationSchema {
		readiness.Reason = "unexpected_candidate_admission_ledger_verification_schema " + sourceSchema
		return readiness
	}
	if !verification.Passed {
		readiness.Reason = "candidate_admission_ledger_verification_failed"
		if verification.Reason != "" {
			readiness.Reason += ": " + verification.Reason
		}
		return readiness
	}
	if verification.LedgerVerificationID == "" {
		readiness.Reason = "missing_candidate_admission_ledger_verification_id"
		return readiness
	}
	if wantVerificationID := admissionLiveRouteTurnCandidateAdmissionLedgerVerificationID(verification); wantVerificationID == "" || verification.LedgerVerificationID != wantVerificationID {
		readiness.Reason = "candidate_admission_ledger_verification_id_mismatch"
		return readiness
	}
	if verification.LedgerVerificationState != "ledger_receipt_verified_dry_run" ||
		verification.LedgerVerificationAction != "verify_persisted_admission_ledger_receipt_dry_run" ||
		verification.LedgerVerificationTarget != "admission_ledger" ||
		verification.LedgerVerificationTargetKind != "dream_candidate_admission" ||
		verification.LedgerVerificationTargetMode != "append_only_dry_run" ||
		verification.LedgerVerificationReceiptShape != "candidate_contract_provenance" {
		readiness.Reason = "candidate_admission_ledger_verification_shape_mismatch"
		return readiness
	}
	if !verification.LedgerVerificationAppendOnly ||
		!verification.LedgerVerificationDryRunOnly ||
		!verification.LedgerVerificationReceiptReadBack ||
		!verification.LedgerVerificationReceiptVerified ||
		!verification.LedgerVerificationReady {
		readiness.Reason = "candidate_admission_ledger_verification_not_verified_dry_run"
		return readiness
	}
	if verification.ContractsReady || verification.WriteAllowed || verification.MutatesState || verification.LiveAdmissionEnabled || verification.AdmissionAllowed {
		readiness.Reason = "candidate_admission_ledger_verification_already_open"
		return readiness
	}
	if !verification.LiveReady {
		readiness.Reason = "candidate_admission_ledger_verification_not_live_ready"
		return readiness
	}
	if verification.SourceLedgerPersistenceSchema != admissionLiveRouteTurnCandidateAdmissionLedgerPersistenceSchema ||
		!verification.SourceLedgerPersistencePassed ||
		verification.SourceLedgerPersistenceID != verification.LedgerPersistenceID ||
		verification.SourceLedgerPersistenceAction != "append_admission_ledger_receipt_dry_run" ||
		!verification.SourceLedgerPersistenceReady ||
		!verification.SourceLedgerPersistenceReceiptPersisted {
		readiness.Reason = "candidate_admission_ledger_verification_source_persistence_mismatch"
		return readiness
	}
	if wantPersistenceID := admissionLiveRouteTurnCandidateAdmissionLedgerPersistenceID(verification.admissionLiveRouteTurnCandidateAdmissionLedgerPersistence); wantPersistenceID == "" || verification.LedgerPersistenceID != wantPersistenceID {
		readiness.Reason = "candidate_admission_ledger_persistence_id_mismatch_for_admission_readiness"
		return readiness
	}
	if verification.SourceLedgerImplementationIDForVerification != verification.LedgerImplementationID ||
		verification.SourceAdmissionLedgerIDForVerification != verification.AdmissionLedgerID ||
		verification.SourceRollbackImplementationIDForVerification != verification.RollbackImplementationID ||
		verification.SourceWriterReceiptIDForVerification != verification.WriterReceiptID {
		readiness.Reason = "candidate_admission_ledger_verification_source_id_mismatch"
		return readiness
	}
	if wantLedgerImplID := admissionLiveRouteTurnCandidateAdmissionLedgerImplementationID(verification.admissionLiveRouteTurnCandidateAdmissionLedgerPersistence.admissionLiveRouteTurnCandidateAdmissionLedgerImplementation); wantLedgerImplID == "" || verification.LedgerImplementationID != wantLedgerImplID {
		readiness.Reason = "candidate_admission_ledger_implementation_id_mismatch_for_admission_readiness"
		return readiness
	}
	if wantRollbackID := admissionLiveRouteTurnCandidateAdmissionRollbackImplementationID(verification.admissionLiveRouteTurnCandidateAdmissionLedgerPersistence.admissionLiveRouteTurnCandidateAdmissionLedgerImplementation.admissionLiveRouteTurnCandidateAdmissionRollbackImplementation); wantRollbackID == "" || verification.RollbackImplementationID != wantRollbackID {
		readiness.Reason = "candidate_admission_rollback_implementation_id_mismatch_for_admission_readiness"
		return readiness
	}
	if wantReceiptID := admissionLiveRouteTurnCandidateAdmissionWriterReceiptID(verification.admissionLiveRouteTurnCandidateAdmissionLedgerPersistence.admissionLiveRouteTurnCandidateAdmissionLedgerImplementation.admissionLiveRouteTurnCandidateAdmissionRollbackImplementation.admissionLiveRouteTurnCandidateAdmissionWriterReceipt); wantReceiptID == "" || verification.WriterReceiptID != wantReceiptID {
		readiness.Reason = "candidate_admission_writer_receipt_id_mismatch_for_admission_readiness"
		return readiness
	}
	if verification.LedgerPersistenceState != "ledger_receipt_persisted_dry_run" ||
		verification.LedgerPersistenceAction != "append_admission_ledger_receipt_dry_run" ||
		!verification.LedgerPersistenceReady ||
		!verification.LedgerPersistenceReceiptPersisted ||
		!verification.LedgerPersistenceDryRunOnly ||
		!verification.LedgerPersistenceAppendOnly {
		readiness.Reason = "candidate_admission_ledger_verification_persistence_mismatch"
		return readiness
	}
	if !verification.WriterReady ||
		!verification.RollbackReady ||
		!verification.WriterImplementationReady ||
		!verification.RollbackImplementationReady ||
		!verification.LedgerImplementationReady {
		readiness.Reason = "candidate_admission_ledger_verification_readiness_mismatch"
		return readiness
	}
	if verification.LedgerVerificationID == "" ||
		verification.LedgerPersistenceID == "" ||
		verification.LedgerImplementationID == "" ||
		verification.RollbackImplementationID == "" ||
		verification.WriterReceiptID == "" ||
		verification.WriterImplementationID == "" ||
		verification.AdmissionLedgerID == "" ||
		verification.AdmissionWriterContractID == "" ||
		verification.AdmissionWriterInventoryID == "" ||
		verification.AdmissionWriterPreflightID == "" ||
		verification.AdmissionLiveStageID == "" ||
		verification.AdmissionEnableGateID == "" ||
		verification.AdmissionSwitchID == "" ||
		verification.AdmissionPromotionID == "" ||
		verification.AdmissionDecisionID == "" ||
		verification.AdmissionAdapterID == "" ||
		verification.CandidateRunID == "" ||
		verification.CandidateDraftID == "" ||
		verification.CandidateExecutionID == "" ||
		verification.GeneratorAdapterID == "" ||
		verification.HandoffID == "" ||
		verification.DreamCandidateRunID == "" ||
		verification.CandidateTextHash == "" ||
		verification.TurnTextHash == "" {
		readiness.Reason = "candidate_admission_ledger_verification_missing_provenance"
		return readiness
	}

	readiness.AdmissionReadinessState = "verified_closed_dry_run"
	readiness.AdmissionReadinessAction = "declare_verified_live_admission_readiness_dry_run"
	readiness.AdmissionReadinessTarget = "live_admission"
	readiness.AdmissionReadinessTargetKind = "dream_candidate_admission"
	readiness.AdmissionReadinessTargetMode = "closed_verified_dry_run"
	readiness.AdmissionReadinessDryRunOnly = true
	readiness.AdmissionReadinessLedgerVerified = true
	readiness.AdmissionReadinessWriterReady = verification.WriterReady
	readiness.AdmissionReadinessRollbackReady = verification.RollbackReady
	readiness.AdmissionReadinessLedgerReady = verification.LedgerImplementationReady && verification.LedgerPersistenceReady && verification.LedgerVerificationReady
	readiness.AdmissionReadinessReady = true
	readiness.ContractsReady = false
	readiness.WriteAllowed = false
	readiness.AdmissionAllowed = false
	readiness.LiveAdmissionEnabled = false
	readiness.MutatesState = false
	readiness.AdmissionReadinessID = admissionLiveRouteTurnCandidateAdmissionReadinessID(readiness)
	if readiness.AdmissionReadinessID == "" {
		readiness.Reason = "missing_candidate_admission_readiness_id"
		return readiness
	}
	readiness.Passed = true
	readiness.Reason = "verified ledger and writer boundaries are ready; live admission remains disabled"
	return readiness
}

func admissionLiveRouteTurnCandidateAdmissionReadinessID(readiness admissionLiveRouteTurnCandidateAdmissionReadiness) string {
	h := hashJSON(struct {
		LedgerVerificationID             string `json:"ledger_verification_id"`
		LedgerPersistenceID              string `json:"ledger_persistence_id"`
		LedgerImplementationID           string `json:"ledger_implementation_id"`
		RollbackImplementationID         string `json:"rollback_implementation_id"`
		WriterReceiptID                  string `json:"writer_receipt_id"`
		AdmissionLedgerID                string `json:"admission_ledger_id"`
		CandidateRunID                   string `json:"candidate_run_id"`
		CandidateTextHash                string `json:"candidate_text_hash"`
		TurnTextHash                     string `json:"turn_text_hash"`
		AdmissionReadinessState          string `json:"admission_readiness_state"`
		AdmissionReadinessAction         string `json:"admission_readiness_action"`
		AdmissionReadinessTarget         string `json:"admission_readiness_target"`
		AdmissionReadinessTargetKind     string `json:"admission_readiness_target_kind"`
		AdmissionReadinessTargetMode     string `json:"admission_readiness_target_mode"`
		AdmissionReadinessDryRunOnly     bool   `json:"admission_readiness_dry_run_only"`
		AdmissionReadinessLedgerVerified bool   `json:"admission_readiness_ledger_verified"`
		AdmissionReadinessWriterReady    bool   `json:"admission_readiness_writer_ready"`
		AdmissionReadinessRollbackReady  bool   `json:"admission_readiness_rollback_ready"`
		AdmissionReadinessLedgerReady    bool   `json:"admission_readiness_ledger_ready"`
		AdmissionReadinessReady          bool   `json:"admission_readiness_ready"`
		ContractsReady                   bool   `json:"contracts_ready"`
		BodyTarget                       string `json:"body_target"`
		WriteAllowed                     bool   `json:"write_allowed"`
		AdmissionAllowed                 bool   `json:"admission_allowed"`
		LiveAdmissionEnabled             bool   `json:"live_admission_enabled"`
		MutatesState                     bool   `json:"mutates_state"`
	}{
		LedgerVerificationID:             readiness.LedgerVerificationID,
		LedgerPersistenceID:              readiness.LedgerPersistenceID,
		LedgerImplementationID:           readiness.LedgerImplementationID,
		RollbackImplementationID:         readiness.RollbackImplementationID,
		WriterReceiptID:                  readiness.WriterReceiptID,
		AdmissionLedgerID:                readiness.AdmissionLedgerID,
		CandidateRunID:                   readiness.CandidateRunID,
		CandidateTextHash:                readiness.CandidateTextHash,
		TurnTextHash:                     readiness.TurnTextHash,
		AdmissionReadinessState:          readiness.AdmissionReadinessState,
		AdmissionReadinessAction:         readiness.AdmissionReadinessAction,
		AdmissionReadinessTarget:         readiness.AdmissionReadinessTarget,
		AdmissionReadinessTargetKind:     readiness.AdmissionReadinessTargetKind,
		AdmissionReadinessTargetMode:     readiness.AdmissionReadinessTargetMode,
		AdmissionReadinessDryRunOnly:     readiness.AdmissionReadinessDryRunOnly,
		AdmissionReadinessLedgerVerified: readiness.AdmissionReadinessLedgerVerified,
		AdmissionReadinessWriterReady:    readiness.AdmissionReadinessWriterReady,
		AdmissionReadinessRollbackReady:  readiness.AdmissionReadinessRollbackReady,
		AdmissionReadinessLedgerReady:    readiness.AdmissionReadinessLedgerReady,
		AdmissionReadinessReady:          readiness.AdmissionReadinessReady,
		ContractsReady:                   readiness.ContractsReady,
		BodyTarget:                       readiness.BodyTarget,
		WriteAllowed:                     readiness.WriteAllowed,
		AdmissionAllowed:                 readiness.AdmissionAllowed,
		LiveAdmissionEnabled:             readiness.LiveAdmissionEnabled,
		MutatesState:                     readiness.MutatesState,
	})
	if h == "" {
		return ""
	}
	return "admission-readiness-" + h
}

func recordAdmissionLiveRouteTurnCandidateAdmissionReadiness(readiness admissionLiveRouteTurnCandidateAdmissionReadiness) error {
	path := strings.TrimSpace(os.Getenv("AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_READINESS_LOG"))
	if path == "" {
		return nil
	}
	f, err := os.OpenFile(path, os.O_CREATE|os.O_APPEND|os.O_WRONLY, 0600)
	if err != nil {
		return err
	}
	enc := json.NewEncoder(f)
	err = enc.Encode(readiness)
	if closeErr := f.Close(); err == nil {
		err = closeErr
	}
	return err
}

func admissionLiveRouteTurnCandidateAdmissionPermitForReadiness(readiness admissionLiveRouteTurnCandidateAdmissionReadiness) admissionLiveRouteTurnCandidateAdmissionPermit {
	sourceSchema := readiness.Schema
	permitKey := admissionLiveRouteTurnCandidateAdmissionPermitKey()
	permit := admissionLiveRouteTurnCandidateAdmissionPermit{
		admissionLiveRouteTurnCandidateAdmissionReadiness: readiness,
		AdmissionPermitState:                              "blocked",
		AdmissionPermitAction:                             "reject",
		AdmissionPermitDryRunOnly:                         true,
		ManualPermitRequested:                             permitKey != "",
		PermitKeyMatched:                                  permitKey == admissionLiveRouteTurnCandidateAdmissionPermitConfirmation,
		SourceAdmissionReadinessSchema:                    sourceSchema,
		SourceAdmissionReadinessPassed:                    readiness.Passed,
		SourceAdmissionReadinessID:                        readiness.AdmissionReadinessID,
		SourceAdmissionReadinessAction:                    readiness.AdmissionReadinessAction,
		SourceAdmissionReadinessReady:                     readiness.AdmissionReadinessReady,
		SourceAdmissionReadinessLedgerVerified:            readiness.AdmissionReadinessLedgerVerified,
		SourceLedgerVerificationIDForPermit:               readiness.LedgerVerificationID,
		SourceLedgerPersistenceIDForPermit:                readiness.LedgerPersistenceID,
		SourceLedgerImplementationIDForPermit:             readiness.LedgerImplementationID,
		SourceAdmissionLedgerIDForPermit:                  readiness.AdmissionLedgerID,
		SourceRollbackImplementationIDForPermit:           readiness.RollbackImplementationID,
		SourceWriterReceiptIDForPermit:                    readiness.WriterReceiptID,
	}
	permit.Schema = admissionLiveRouteTurnCandidateAdmissionPermitSchema
	permit.Timing = "live_admission_permit"
	permit.Passed = false
	permit.AdmissionPermitID = ""
	permit.AdmissionPermitReady = false
	permit.ContractsReady = false
	permit.WriteAllowed = false
	permit.AdmissionAllowed = false
	permit.LiveAdmissionEnabled = false
	permit.MutatesState = false

	if sourceSchema == "" {
		permit.Reason = "missing_candidate_admission_readiness"
		return permit
	}
	if sourceSchema != admissionLiveRouteTurnCandidateAdmissionReadinessSchema {
		permit.Reason = "unexpected_candidate_admission_readiness_schema " + sourceSchema
		return permit
	}
	if !readiness.Passed {
		permit.Reason = "candidate_admission_readiness_failed"
		if readiness.Reason != "" {
			permit.Reason += ": " + readiness.Reason
		}
		return permit
	}
	if readiness.AdmissionReadinessID == "" {
		permit.Reason = "missing_candidate_admission_readiness_id"
		return permit
	}
	if wantReadinessID := admissionLiveRouteTurnCandidateAdmissionReadinessID(readiness); wantReadinessID == "" || readiness.AdmissionReadinessID != wantReadinessID {
		permit.Reason = "candidate_admission_readiness_id_mismatch"
		return permit
	}
	if readiness.AdmissionReadinessState != "verified_closed_dry_run" ||
		readiness.AdmissionReadinessAction != "declare_verified_live_admission_readiness_dry_run" ||
		readiness.AdmissionReadinessTarget != "live_admission" ||
		readiness.AdmissionReadinessTargetKind != "dream_candidate_admission" ||
		readiness.AdmissionReadinessTargetMode != "closed_verified_dry_run" {
		permit.Reason = "candidate_admission_readiness_shape_mismatch"
		return permit
	}
	if !readiness.AdmissionReadinessDryRunOnly ||
		!readiness.AdmissionReadinessLedgerVerified ||
		!readiness.AdmissionReadinessWriterReady ||
		!readiness.AdmissionReadinessRollbackReady ||
		!readiness.AdmissionReadinessLedgerReady ||
		!readiness.AdmissionReadinessReady {
		permit.Reason = "candidate_admission_readiness_not_verified_dry_run"
		return permit
	}
	if readiness.ContractsReady || readiness.WriteAllowed || readiness.MutatesState || readiness.LiveAdmissionEnabled || readiness.AdmissionAllowed {
		permit.Reason = "candidate_admission_readiness_already_open"
		return permit
	}
	if !readiness.LiveReady {
		permit.Reason = "candidate_admission_readiness_not_live_ready"
		return permit
	}
	if readiness.SourceLedgerVerificationSchema != admissionLiveRouteTurnCandidateAdmissionLedgerVerificationSchema ||
		!readiness.SourceLedgerVerificationPassed ||
		readiness.SourceLedgerVerificationID != readiness.LedgerVerificationID ||
		readiness.SourceLedgerVerificationAction != "verify_persisted_admission_ledger_receipt_dry_run" ||
		!readiness.SourceLedgerVerificationReady ||
		!readiness.SourceLedgerVerificationReceiptVerified {
		permit.Reason = "candidate_admission_readiness_source_verification_mismatch"
		return permit
	}
	if wantVerificationID := admissionLiveRouteTurnCandidateAdmissionLedgerVerificationID(readiness.admissionLiveRouteTurnCandidateAdmissionLedgerVerification); wantVerificationID == "" || readiness.LedgerVerificationID != wantVerificationID {
		permit.Reason = "candidate_admission_ledger_verification_id_mismatch_for_admission_permit"
		return permit
	}
	if readiness.SourceLedgerPersistenceIDForReadiness != readiness.LedgerPersistenceID ||
		readiness.SourceLedgerImplementationIDForReadiness != readiness.LedgerImplementationID ||
		readiness.SourceAdmissionLedgerIDForReadiness != readiness.AdmissionLedgerID ||
		readiness.SourceRollbackImplementationIDForReadiness != readiness.RollbackImplementationID ||
		readiness.SourceWriterReceiptIDForReadiness != readiness.WriterReceiptID {
		permit.Reason = "candidate_admission_readiness_source_id_mismatch"
		return permit
	}
	if readiness.AdmissionReadinessID == "" ||
		readiness.LedgerVerificationID == "" ||
		readiness.LedgerPersistenceID == "" ||
		readiness.LedgerImplementationID == "" ||
		readiness.RollbackImplementationID == "" ||
		readiness.WriterReceiptID == "" ||
		readiness.WriterImplementationID == "" ||
		readiness.AdmissionLedgerID == "" ||
		readiness.AdmissionWriterContractID == "" ||
		readiness.AdmissionWriterInventoryID == "" ||
		readiness.AdmissionWriterPreflightID == "" ||
		readiness.AdmissionLiveStageID == "" ||
		readiness.AdmissionEnableGateID == "" ||
		readiness.AdmissionSwitchID == "" ||
		readiness.AdmissionPromotionID == "" ||
		readiness.AdmissionDecisionID == "" ||
		readiness.AdmissionAdapterID == "" ||
		readiness.CandidateRunID == "" ||
		readiness.CandidateDraftID == "" ||
		readiness.CandidateExecutionID == "" ||
		readiness.GeneratorAdapterID == "" ||
		readiness.HandoffID == "" ||
		readiness.DreamCandidateRunID == "" ||
		readiness.CandidateTextHash == "" ||
		readiness.TurnTextHash == "" {
		permit.Reason = "candidate_admission_readiness_missing_provenance"
		return permit
	}
	if !permit.ManualPermitRequested {
		permit.Reason = "live_admission_permit_key_missing"
		return permit
	}
	if !permit.PermitKeyMatched {
		permit.Reason = "live_admission_permit_key_mismatch"
		return permit
	}

	permit.AdmissionPermitState = "operator_permitted_closed_dry_run"
	permit.AdmissionPermitAction = "acknowledge_verified_live_admission_readiness_dry_run"
	permit.AdmissionPermitTarget = "live_admission"
	permit.AdmissionPermitTargetKind = "dream_candidate_admission"
	permit.AdmissionPermitTargetMode = "permit_closed_dry_run"
	permit.AdmissionPermitDryRunOnly = true
	permit.AdmissionPermitReadinessVerified = true
	permit.AdmissionPermitLedgerVerified = readiness.AdmissionReadinessLedgerVerified
	permit.AdmissionPermitWriterReady = readiness.AdmissionReadinessWriterReady
	permit.AdmissionPermitRollbackReady = readiness.AdmissionReadinessRollbackReady
	permit.AdmissionPermitLedgerReady = readiness.AdmissionReadinessLedgerReady
	permit.AdmissionPermitReady = true
	permit.ContractsReady = false
	permit.WriteAllowed = false
	permit.AdmissionAllowed = false
	permit.LiveAdmissionEnabled = false
	permit.MutatesState = false
	permit.AdmissionPermitID = admissionLiveRouteTurnCandidateAdmissionPermitID(permit)
	if permit.AdmissionPermitID == "" {
		permit.Reason = "missing_candidate_admission_permit_id"
		return permit
	}
	permit.Passed = true
	permit.Reason = "operator permit accepted for verified readiness; live admission remains disabled"
	return permit
}

func admissionLiveRouteTurnCandidateAdmissionPermitID(permit admissionLiveRouteTurnCandidateAdmissionPermit) string {
	h := hashJSON(struct {
		AdmissionReadinessID                string `json:"admission_readiness_id"`
		LedgerVerificationID                string `json:"ledger_verification_id"`
		LedgerPersistenceID                 string `json:"ledger_persistence_id"`
		LedgerImplementationID              string `json:"ledger_implementation_id"`
		RollbackImplementationID            string `json:"rollback_implementation_id"`
		WriterReceiptID                     string `json:"writer_receipt_id"`
		AdmissionLedgerID                   string `json:"admission_ledger_id"`
		CandidateRunID                      string `json:"candidate_run_id"`
		CandidateTextHash                   string `json:"candidate_text_hash"`
		TurnTextHash                        string `json:"turn_text_hash"`
		AdmissionPermitState                string `json:"admission_permit_state"`
		AdmissionPermitAction               string `json:"admission_permit_action"`
		AdmissionPermitTarget               string `json:"admission_permit_target"`
		AdmissionPermitTargetKind           string `json:"admission_permit_target_kind"`
		AdmissionPermitTargetMode           string `json:"admission_permit_target_mode"`
		AdmissionPermitDryRunOnly           bool   `json:"admission_permit_dry_run_only"`
		AdmissionPermitReadinessVerified    bool   `json:"admission_permit_readiness_verified"`
		AdmissionPermitLedgerVerified       bool   `json:"admission_permit_ledger_verified"`
		AdmissionPermitWriterReady          bool   `json:"admission_permit_writer_ready"`
		AdmissionPermitRollbackReady        bool   `json:"admission_permit_rollback_ready"`
		AdmissionPermitLedgerReady          bool   `json:"admission_permit_ledger_ready"`
		AdmissionPermitReady                bool   `json:"admission_permit_ready"`
		ManualPermitRequested               bool   `json:"manual_permit_requested"`
		PermitKeyMatched                    bool   `json:"permit_key_matched"`
		SourceAdmissionReadinessID          string `json:"source_admission_readiness_id"`
		SourceAdmissionReadinessReady       bool   `json:"source_admission_readiness_ready"`
		SourceAdmissionReadinessLedgerReady bool   `json:"source_admission_readiness_ledger_verified"`
		ContractsReady                      bool   `json:"contracts_ready"`
		BodyTarget                          string `json:"body_target"`
		WriteAllowed                        bool   `json:"write_allowed"`
		AdmissionAllowed                    bool   `json:"admission_allowed"`
		LiveAdmissionEnabled                bool   `json:"live_admission_enabled"`
		MutatesState                        bool   `json:"mutates_state"`
	}{
		AdmissionReadinessID:                permit.AdmissionReadinessID,
		LedgerVerificationID:                permit.LedgerVerificationID,
		LedgerPersistenceID:                 permit.LedgerPersistenceID,
		LedgerImplementationID:              permit.LedgerImplementationID,
		RollbackImplementationID:            permit.RollbackImplementationID,
		WriterReceiptID:                     permit.WriterReceiptID,
		AdmissionLedgerID:                   permit.AdmissionLedgerID,
		CandidateRunID:                      permit.CandidateRunID,
		CandidateTextHash:                   permit.CandidateTextHash,
		TurnTextHash:                        permit.TurnTextHash,
		AdmissionPermitState:                permit.AdmissionPermitState,
		AdmissionPermitAction:               permit.AdmissionPermitAction,
		AdmissionPermitTarget:               permit.AdmissionPermitTarget,
		AdmissionPermitTargetKind:           permit.AdmissionPermitTargetKind,
		AdmissionPermitTargetMode:           permit.AdmissionPermitTargetMode,
		AdmissionPermitDryRunOnly:           permit.AdmissionPermitDryRunOnly,
		AdmissionPermitReadinessVerified:    permit.AdmissionPermitReadinessVerified,
		AdmissionPermitLedgerVerified:       permit.AdmissionPermitLedgerVerified,
		AdmissionPermitWriterReady:          permit.AdmissionPermitWriterReady,
		AdmissionPermitRollbackReady:        permit.AdmissionPermitRollbackReady,
		AdmissionPermitLedgerReady:          permit.AdmissionPermitLedgerReady,
		AdmissionPermitReady:                permit.AdmissionPermitReady,
		ManualPermitRequested:               permit.ManualPermitRequested,
		PermitKeyMatched:                    permit.PermitKeyMatched,
		SourceAdmissionReadinessID:          permit.SourceAdmissionReadinessID,
		SourceAdmissionReadinessReady:       permit.SourceAdmissionReadinessReady,
		SourceAdmissionReadinessLedgerReady: permit.SourceAdmissionReadinessLedgerVerified,
		ContractsReady:                      permit.ContractsReady,
		BodyTarget:                          permit.BodyTarget,
		WriteAllowed:                        permit.WriteAllowed,
		AdmissionAllowed:                    permit.AdmissionAllowed,
		LiveAdmissionEnabled:                permit.LiveAdmissionEnabled,
		MutatesState:                        permit.MutatesState,
	})
	if h == "" {
		return ""
	}
	return "admission-permit-" + h
}

func recordAdmissionLiveRouteTurnCandidateAdmissionPermit(permit admissionLiveRouteTurnCandidateAdmissionPermit) error {
	path := strings.TrimSpace(os.Getenv("AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_PERMIT_LOG"))
	if path == "" {
		return nil
	}
	f, err := os.OpenFile(path, os.O_CREATE|os.O_APPEND|os.O_WRONLY, 0600)
	if err != nil {
		return err
	}
	enc := json.NewEncoder(f)
	err = enc.Encode(permit)
	if closeErr := f.Close(); err == nil {
		err = closeErr
	}
	return err
}

func admissionLiveRouteTurnCandidateAdmissionSealForPermit(permit admissionLiveRouteTurnCandidateAdmissionPermit) admissionLiveRouteTurnCandidateAdmissionSeal {
	sourceSchema := permit.Schema
	seal := admissionLiveRouteTurnCandidateAdmissionSeal{
		admissionLiveRouteTurnCandidateAdmissionPermit: permit,
		AdmissionSealState:                    "blocked",
		AdmissionSealAction:                   "reject",
		AdmissionSealDryRunOnly:               true,
		SourceAdmissionPermitSchema:           sourceSchema,
		SourceAdmissionPermitPassed:           permit.Passed,
		SourceAdmissionPermitID:               permit.AdmissionPermitID,
		SourceAdmissionPermitAction:           permit.AdmissionPermitAction,
		SourceAdmissionPermitReady:            permit.AdmissionPermitReady,
		SourceAdmissionPermitKeyMatched:       permit.PermitKeyMatched,
		SourceAdmissionReadinessIDForSeal:     permit.AdmissionReadinessID,
		SourceLedgerVerificationIDForSeal:     permit.LedgerVerificationID,
		SourceLedgerPersistenceIDForSeal:      permit.LedgerPersistenceID,
		SourceLedgerImplementationIDForSeal:   permit.LedgerImplementationID,
		SourceAdmissionLedgerIDForSeal:        permit.AdmissionLedgerID,
		SourceRollbackImplementationIDForSeal: permit.RollbackImplementationID,
		SourceWriterReceiptIDForSeal:          permit.WriterReceiptID,
	}
	seal.Schema = admissionLiveRouteTurnCandidateAdmissionSealSchema
	seal.Timing = "live_admission_seal"
	seal.Passed = false
	seal.AdmissionSealID = ""
	seal.AdmissionSealReady = false
	seal.ContractsReady = false
	seal.WriteAllowed = false
	seal.AdmissionAllowed = false
	seal.LiveAdmissionEnabled = false
	seal.MutatesState = false

	if sourceSchema == "" {
		seal.Reason = "missing_candidate_admission_permit"
		return seal
	}
	if sourceSchema != admissionLiveRouteTurnCandidateAdmissionPermitSchema {
		seal.Reason = "unexpected_candidate_admission_permit_schema " + sourceSchema
		return seal
	}
	if !permit.Passed {
		seal.Reason = "candidate_admission_permit_failed"
		if permit.Reason != "" {
			seal.Reason += ": " + permit.Reason
		}
		return seal
	}
	if permit.AdmissionPermitID == "" {
		seal.Reason = "missing_candidate_admission_permit_id"
		return seal
	}
	if wantPermitID := admissionLiveRouteTurnCandidateAdmissionPermitID(permit); wantPermitID == "" || permit.AdmissionPermitID != wantPermitID {
		seal.Reason = "candidate_admission_permit_id_mismatch"
		return seal
	}
	if permit.AdmissionPermitState != "operator_permitted_closed_dry_run" ||
		permit.AdmissionPermitAction != "acknowledge_verified_live_admission_readiness_dry_run" ||
		permit.AdmissionPermitTarget != "live_admission" ||
		permit.AdmissionPermitTargetKind != "dream_candidate_admission" ||
		permit.AdmissionPermitTargetMode != "permit_closed_dry_run" {
		seal.Reason = "candidate_admission_permit_shape_mismatch"
		return seal
	}
	if !permit.AdmissionPermitDryRunOnly ||
		!permit.AdmissionPermitReadinessVerified ||
		!permit.AdmissionPermitLedgerVerified ||
		!permit.AdmissionPermitWriterReady ||
		!permit.AdmissionPermitRollbackReady ||
		!permit.AdmissionPermitLedgerReady ||
		!permit.AdmissionPermitReady ||
		!permit.ManualPermitRequested ||
		!permit.PermitKeyMatched {
		seal.Reason = "candidate_admission_permit_not_verified_dry_run"
		return seal
	}
	if permit.ContractsReady || permit.WriteAllowed || permit.MutatesState || permit.LiveAdmissionEnabled || permit.AdmissionAllowed {
		seal.Reason = "candidate_admission_permit_already_open"
		return seal
	}
	if !permit.LiveReady {
		seal.Reason = "candidate_admission_permit_not_live_ready"
		return seal
	}
	if permit.SourceAdmissionReadinessSchema != admissionLiveRouteTurnCandidateAdmissionReadinessSchema ||
		!permit.SourceAdmissionReadinessPassed ||
		permit.SourceAdmissionReadinessID != permit.AdmissionReadinessID ||
		permit.SourceAdmissionReadinessAction != "declare_verified_live_admission_readiness_dry_run" ||
		!permit.SourceAdmissionReadinessReady ||
		!permit.SourceAdmissionReadinessLedgerVerified {
		seal.Reason = "candidate_admission_permit_source_readiness_mismatch"
		return seal
	}
	if wantReadinessID := admissionLiveRouteTurnCandidateAdmissionReadinessID(permit.admissionLiveRouteTurnCandidateAdmissionReadiness); wantReadinessID == "" || permit.AdmissionReadinessID != wantReadinessID {
		seal.Reason = "candidate_admission_readiness_id_mismatch_for_admission_seal"
		return seal
	}
	if permit.SourceLedgerVerificationIDForPermit != permit.LedgerVerificationID ||
		permit.SourceLedgerPersistenceIDForPermit != permit.LedgerPersistenceID ||
		permit.SourceLedgerImplementationIDForPermit != permit.LedgerImplementationID ||
		permit.SourceAdmissionLedgerIDForPermit != permit.AdmissionLedgerID ||
		permit.SourceRollbackImplementationIDForPermit != permit.RollbackImplementationID ||
		permit.SourceWriterReceiptIDForPermit != permit.WriterReceiptID {
		seal.Reason = "candidate_admission_permit_source_id_mismatch"
		return seal
	}
	if permit.AdmissionPermitID == "" ||
		permit.AdmissionReadinessID == "" ||
		permit.LedgerVerificationID == "" ||
		permit.LedgerPersistenceID == "" ||
		permit.LedgerImplementationID == "" ||
		permit.RollbackImplementationID == "" ||
		permit.WriterReceiptID == "" ||
		permit.WriterImplementationID == "" ||
		permit.AdmissionLedgerID == "" ||
		permit.AdmissionWriterContractID == "" ||
		permit.AdmissionWriterInventoryID == "" ||
		permit.AdmissionWriterPreflightID == "" ||
		permit.AdmissionLiveStageID == "" ||
		permit.AdmissionEnableGateID == "" ||
		permit.AdmissionSwitchID == "" ||
		permit.AdmissionPromotionID == "" ||
		permit.AdmissionDecisionID == "" ||
		permit.AdmissionAdapterID == "" ||
		permit.CandidateRunID == "" ||
		permit.CandidateDraftID == "" ||
		permit.CandidateExecutionID == "" ||
		permit.GeneratorAdapterID == "" ||
		permit.HandoffID == "" ||
		permit.DreamCandidateRunID == "" ||
		permit.CandidateTextHash == "" ||
		permit.TurnTextHash == "" {
		seal.Reason = "candidate_admission_permit_missing_provenance"
		return seal
	}

	seal.AdmissionSealState = "sealed_closed_dry_run"
	seal.AdmissionSealAction = "seal_operator_permit_provenance_dry_run"
	seal.AdmissionSealTarget = "live_admission"
	seal.AdmissionSealTargetKind = "dream_candidate_admission"
	seal.AdmissionSealTargetMode = "sealed_closed_dry_run"
	seal.AdmissionSealReceiptShape = "candidate_contract_provenance"
	seal.AdmissionSealDryRunOnly = true
	seal.AdmissionSealPermitVerified = true
	seal.AdmissionSealReadinessVerified = permit.AdmissionPermitReadinessVerified
	seal.AdmissionSealLedgerVerified = permit.AdmissionPermitLedgerVerified
	seal.AdmissionSealWriterReady = permit.AdmissionPermitWriterReady
	seal.AdmissionSealRollbackReady = permit.AdmissionPermitRollbackReady
	seal.AdmissionSealLedgerReady = permit.AdmissionPermitLedgerReady
	seal.AdmissionSealReady = true
	seal.ContractsReady = false
	seal.WriteAllowed = false
	seal.AdmissionAllowed = false
	seal.LiveAdmissionEnabled = false
	seal.MutatesState = false
	seal.AdmissionSealID = admissionLiveRouteTurnCandidateAdmissionSealID(seal)
	if seal.AdmissionSealID == "" {
		seal.Reason = "missing_candidate_admission_seal_id"
		return seal
	}
	seal.Passed = true
	seal.Reason = "operator permit sealed as immutable dry-run receipt; live admission remains disabled"
	return seal
}

func admissionLiveRouteTurnCandidateAdmissionSealID(seal admissionLiveRouteTurnCandidateAdmissionSeal) string {
	h := hashJSON(struct {
		AdmissionPermitID                 string `json:"admission_permit_id"`
		AdmissionReadinessID              string `json:"admission_readiness_id"`
		LedgerVerificationID              string `json:"ledger_verification_id"`
		LedgerPersistenceID               string `json:"ledger_persistence_id"`
		LedgerImplementationID            string `json:"ledger_implementation_id"`
		RollbackImplementationID          string `json:"rollback_implementation_id"`
		WriterReceiptID                   string `json:"writer_receipt_id"`
		AdmissionLedgerID                 string `json:"admission_ledger_id"`
		CandidateRunID                    string `json:"candidate_run_id"`
		CandidateTextHash                 string `json:"candidate_text_hash"`
		TurnTextHash                      string `json:"turn_text_hash"`
		AdmissionSealState                string `json:"admission_seal_state"`
		AdmissionSealAction               string `json:"admission_seal_action"`
		AdmissionSealTarget               string `json:"admission_seal_target"`
		AdmissionSealTargetKind           string `json:"admission_seal_target_kind"`
		AdmissionSealTargetMode           string `json:"admission_seal_target_mode"`
		AdmissionSealReceiptShape         string `json:"admission_seal_receipt_shape"`
		AdmissionSealDryRunOnly           bool   `json:"admission_seal_dry_run_only"`
		AdmissionSealPermitVerified       bool   `json:"admission_seal_permit_verified"`
		AdmissionSealReadinessVerified    bool   `json:"admission_seal_readiness_verified"`
		AdmissionSealLedgerVerified       bool   `json:"admission_seal_ledger_verified"`
		AdmissionSealWriterReady          bool   `json:"admission_seal_writer_ready"`
		AdmissionSealRollbackReady        bool   `json:"admission_seal_rollback_ready"`
		AdmissionSealLedgerReady          bool   `json:"admission_seal_ledger_ready"`
		AdmissionSealReady                bool   `json:"admission_seal_ready"`
		SourceAdmissionPermitID           string `json:"source_admission_permit_id"`
		SourceAdmissionPermitReady        bool   `json:"source_admission_permit_ready"`
		SourceAdmissionPermitKeyMatched   bool   `json:"source_admission_permit_key_matched"`
		SourceAdmissionReadinessIDForSeal string `json:"source_admission_readiness_id_for_seal"`
		ContractsReady                    bool   `json:"contracts_ready"`
		BodyTarget                        string `json:"body_target"`
		WriteAllowed                      bool   `json:"write_allowed"`
		AdmissionAllowed                  bool   `json:"admission_allowed"`
		LiveAdmissionEnabled              bool   `json:"live_admission_enabled"`
		MutatesState                      bool   `json:"mutates_state"`
	}{
		AdmissionPermitID:                 seal.AdmissionPermitID,
		AdmissionReadinessID:              seal.AdmissionReadinessID,
		LedgerVerificationID:              seal.LedgerVerificationID,
		LedgerPersistenceID:               seal.LedgerPersistenceID,
		LedgerImplementationID:            seal.LedgerImplementationID,
		RollbackImplementationID:          seal.RollbackImplementationID,
		WriterReceiptID:                   seal.WriterReceiptID,
		AdmissionLedgerID:                 seal.AdmissionLedgerID,
		CandidateRunID:                    seal.CandidateRunID,
		CandidateTextHash:                 seal.CandidateTextHash,
		TurnTextHash:                      seal.TurnTextHash,
		AdmissionSealState:                seal.AdmissionSealState,
		AdmissionSealAction:               seal.AdmissionSealAction,
		AdmissionSealTarget:               seal.AdmissionSealTarget,
		AdmissionSealTargetKind:           seal.AdmissionSealTargetKind,
		AdmissionSealTargetMode:           seal.AdmissionSealTargetMode,
		AdmissionSealReceiptShape:         seal.AdmissionSealReceiptShape,
		AdmissionSealDryRunOnly:           seal.AdmissionSealDryRunOnly,
		AdmissionSealPermitVerified:       seal.AdmissionSealPermitVerified,
		AdmissionSealReadinessVerified:    seal.AdmissionSealReadinessVerified,
		AdmissionSealLedgerVerified:       seal.AdmissionSealLedgerVerified,
		AdmissionSealWriterReady:          seal.AdmissionSealWriterReady,
		AdmissionSealRollbackReady:        seal.AdmissionSealRollbackReady,
		AdmissionSealLedgerReady:          seal.AdmissionSealLedgerReady,
		AdmissionSealReady:                seal.AdmissionSealReady,
		SourceAdmissionPermitID:           seal.SourceAdmissionPermitID,
		SourceAdmissionPermitReady:        seal.SourceAdmissionPermitReady,
		SourceAdmissionPermitKeyMatched:   seal.SourceAdmissionPermitKeyMatched,
		SourceAdmissionReadinessIDForSeal: seal.SourceAdmissionReadinessIDForSeal,
		ContractsReady:                    seal.ContractsReady,
		BodyTarget:                        seal.BodyTarget,
		WriteAllowed:                      seal.WriteAllowed,
		AdmissionAllowed:                  seal.AdmissionAllowed,
		LiveAdmissionEnabled:              seal.LiveAdmissionEnabled,
		MutatesState:                      seal.MutatesState,
	})
	if h == "" {
		return ""
	}
	return "admission-seal-" + h
}

func recordAdmissionLiveRouteTurnCandidateAdmissionSeal(seal admissionLiveRouteTurnCandidateAdmissionSeal) error {
	path := strings.TrimSpace(os.Getenv("AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_SEAL_LOG"))
	if path == "" {
		return nil
	}
	f, err := os.OpenFile(path, os.O_CREATE|os.O_APPEND|os.O_WRONLY, 0600)
	if err != nil {
		return err
	}
	enc := json.NewEncoder(f)
	err = enc.Encode(seal)
	if closeErr := f.Close(); err == nil {
		err = closeErr
	}
	return err
}

func admissionLiveRouteTurnCandidateAdmissionFinalGateForSeal(seal admissionLiveRouteTurnCandidateAdmissionSeal) admissionLiveRouteTurnCandidateAdmissionFinalGate {
	sourceSchema := seal.Schema
	finalGate := admissionLiveRouteTurnCandidateAdmissionFinalGate{
		admissionLiveRouteTurnCandidateAdmissionSeal: seal,
		AdmissionFinalGateState:                      "blocked",
		AdmissionFinalGateAction:                     "reject",
		AdmissionFinalGateDryRunOnly:                 true,
		SourceAdmissionSealSchema:                    sourceSchema,
		SourceAdmissionSealPassed:                    seal.Passed,
		SourceAdmissionSealID:                        seal.AdmissionSealID,
		SourceAdmissionSealAction:                    seal.AdmissionSealAction,
		SourceAdmissionSealReady:                     seal.AdmissionSealReady,
		SourceAdmissionPermitIDForFinalGate:          seal.AdmissionPermitID,
		SourceAdmissionReadinessIDForFinalGate:       seal.AdmissionReadinessID,
		SourceLedgerVerificationIDForFinalGate:       seal.LedgerVerificationID,
		SourceLedgerPersistenceIDForFinalGate:        seal.LedgerPersistenceID,
		SourceLedgerImplementationIDForFinalGate:     seal.LedgerImplementationID,
		SourceAdmissionLedgerIDForFinalGate:          seal.AdmissionLedgerID,
		SourceRollbackImplementationIDForFinalGate:   seal.RollbackImplementationID,
		SourceWriterReceiptIDForFinalGate:            seal.WriterReceiptID,
	}
	finalGate.Schema = admissionLiveRouteTurnCandidateAdmissionFinalGateSchema
	finalGate.Timing = "live_admission_final_gate"
	finalGate.Passed = false
	finalGate.AdmissionFinalGateID = ""
	finalGate.AdmissionFinalGateReady = false
	finalGate.ContractsReady = false
	finalGate.WriteAllowed = false
	finalGate.AdmissionAllowed = false
	finalGate.LiveAdmissionEnabled = false
	finalGate.MutatesState = false

	if sourceSchema == "" {
		finalGate.Reason = "missing_candidate_admission_seal"
		return finalGate
	}
	if sourceSchema != admissionLiveRouteTurnCandidateAdmissionSealSchema {
		finalGate.Reason = "unexpected_candidate_admission_seal_schema " + sourceSchema
		return finalGate
	}
	if !seal.Passed {
		finalGate.Reason = "candidate_admission_seal_failed"
		if seal.Reason != "" {
			finalGate.Reason += ": " + seal.Reason
		}
		return finalGate
	}
	if seal.AdmissionSealID == "" {
		finalGate.Reason = "missing_candidate_admission_seal_id"
		return finalGate
	}
	if wantSealID := admissionLiveRouteTurnCandidateAdmissionSealID(seal); wantSealID == "" || seal.AdmissionSealID != wantSealID {
		finalGate.Reason = "candidate_admission_seal_id_mismatch"
		return finalGate
	}
	if seal.AdmissionSealState != "sealed_closed_dry_run" ||
		seal.AdmissionSealAction != "seal_operator_permit_provenance_dry_run" ||
		seal.AdmissionSealTarget != "live_admission" ||
		seal.AdmissionSealTargetKind != "dream_candidate_admission" ||
		seal.AdmissionSealTargetMode != "sealed_closed_dry_run" ||
		seal.AdmissionSealReceiptShape != "candidate_contract_provenance" {
		finalGate.Reason = "candidate_admission_seal_shape_mismatch"
		return finalGate
	}
	if !seal.AdmissionSealDryRunOnly ||
		!seal.AdmissionSealPermitVerified ||
		!seal.AdmissionSealReadinessVerified ||
		!seal.AdmissionSealLedgerVerified ||
		!seal.AdmissionSealWriterReady ||
		!seal.AdmissionSealRollbackReady ||
		!seal.AdmissionSealLedgerReady ||
		!seal.AdmissionSealReady {
		finalGate.Reason = "candidate_admission_seal_not_verified_dry_run"
		return finalGate
	}
	if seal.ContractsReady || seal.WriteAllowed || seal.MutatesState || seal.LiveAdmissionEnabled || seal.AdmissionAllowed {
		finalGate.Reason = "candidate_admission_seal_already_open"
		return finalGate
	}
	if !seal.LiveReady {
		finalGate.Reason = "candidate_admission_seal_not_live_ready"
		return finalGate
	}
	if seal.SourceAdmissionPermitSchema != admissionLiveRouteTurnCandidateAdmissionPermitSchema ||
		!seal.SourceAdmissionPermitPassed ||
		seal.SourceAdmissionPermitID != seal.AdmissionPermitID ||
		seal.SourceAdmissionPermitAction != "acknowledge_verified_live_admission_readiness_dry_run" ||
		!seal.SourceAdmissionPermitReady ||
		!seal.SourceAdmissionPermitKeyMatched {
		finalGate.Reason = "candidate_admission_seal_source_permit_mismatch"
		return finalGate
	}
	if wantPermitID := admissionLiveRouteTurnCandidateAdmissionPermitID(seal.admissionLiveRouteTurnCandidateAdmissionPermit); wantPermitID == "" || seal.AdmissionPermitID != wantPermitID {
		finalGate.Reason = "candidate_admission_permit_id_mismatch_for_final_gate"
		return finalGate
	}
	if seal.SourceAdmissionReadinessIDForSeal != seal.AdmissionReadinessID ||
		seal.SourceLedgerVerificationIDForSeal != seal.LedgerVerificationID ||
		seal.SourceLedgerPersistenceIDForSeal != seal.LedgerPersistenceID ||
		seal.SourceLedgerImplementationIDForSeal != seal.LedgerImplementationID ||
		seal.SourceAdmissionLedgerIDForSeal != seal.AdmissionLedgerID ||
		seal.SourceRollbackImplementationIDForSeal != seal.RollbackImplementationID ||
		seal.SourceWriterReceiptIDForSeal != seal.WriterReceiptID {
		finalGate.Reason = "candidate_admission_seal_source_id_mismatch"
		return finalGate
	}
	if seal.AdmissionSealID == "" ||
		seal.AdmissionPermitID == "" ||
		seal.AdmissionReadinessID == "" ||
		seal.LedgerVerificationID == "" ||
		seal.LedgerPersistenceID == "" ||
		seal.LedgerImplementationID == "" ||
		seal.RollbackImplementationID == "" ||
		seal.WriterReceiptID == "" ||
		seal.WriterImplementationID == "" ||
		seal.AdmissionLedgerID == "" ||
		seal.AdmissionWriterContractID == "" ||
		seal.AdmissionWriterInventoryID == "" ||
		seal.AdmissionWriterPreflightID == "" ||
		seal.AdmissionLiveStageID == "" ||
		seal.AdmissionEnableGateID == "" ||
		seal.AdmissionSwitchID == "" ||
		seal.AdmissionPromotionID == "" ||
		seal.AdmissionDecisionID == "" ||
		seal.AdmissionAdapterID == "" ||
		seal.CandidateRunID == "" ||
		seal.CandidateDraftID == "" ||
		seal.CandidateExecutionID == "" ||
		seal.GeneratorAdapterID == "" ||
		seal.HandoffID == "" ||
		seal.DreamCandidateRunID == "" ||
		seal.CandidateTextHash == "" ||
		seal.TurnTextHash == "" {
		finalGate.Reason = "candidate_admission_seal_missing_provenance"
		return finalGate
	}

	finalGate.AdmissionFinalGateState = "ready_closed_dry_run"
	finalGate.AdmissionFinalGateAction = "verify_sealed_admission_provenance_dry_run"
	finalGate.AdmissionFinalGateTarget = "live_admission"
	finalGate.AdmissionFinalGateTargetKind = "dream_candidate_admission"
	finalGate.AdmissionFinalGateTargetMode = "final_gate_closed_dry_run"
	finalGate.AdmissionFinalGateReceiptShape = "sealed_candidate_contract_provenance"
	finalGate.AdmissionFinalGateDryRunOnly = true
	finalGate.AdmissionFinalGateSealVerified = true
	finalGate.AdmissionFinalGatePermitVerified = seal.AdmissionSealPermitVerified
	finalGate.AdmissionFinalGateReadinessVerified = seal.AdmissionSealReadinessVerified
	finalGate.AdmissionFinalGateLedgerVerified = seal.AdmissionSealLedgerVerified
	finalGate.AdmissionFinalGateWriterReady = seal.AdmissionSealWriterReady
	finalGate.AdmissionFinalGateRollbackReady = seal.AdmissionSealRollbackReady
	finalGate.AdmissionFinalGateLedgerReady = seal.AdmissionSealLedgerReady
	finalGate.AdmissionFinalGateReady = true
	finalGate.BodyTarget = "none"
	finalGate.ContractsReady = false
	finalGate.WriteAllowed = false
	finalGate.AdmissionAllowed = false
	finalGate.LiveAdmissionEnabled = false
	finalGate.MutatesState = false
	finalGate.AdmissionFinalGateID = admissionLiveRouteTurnCandidateAdmissionFinalGateID(finalGate)
	if finalGate.AdmissionFinalGateID == "" {
		finalGate.Reason = "missing_candidate_admission_final_gate_id"
		return finalGate
	}
	finalGate.Passed = true
	finalGate.Reason = "sealed admission provenance cleared final gate; live admission remains disabled"
	return finalGate
}

func admissionLiveRouteTurnCandidateAdmissionFinalGateID(finalGate admissionLiveRouteTurnCandidateAdmissionFinalGate) string {
	h := hashJSON(struct {
		AdmissionSealID                       string `json:"admission_seal_id"`
		AdmissionPermitID                     string `json:"admission_permit_id"`
		AdmissionReadinessID                  string `json:"admission_readiness_id"`
		LedgerVerificationID                  string `json:"ledger_verification_id"`
		LedgerPersistenceID                   string `json:"ledger_persistence_id"`
		LedgerImplementationID                string `json:"ledger_implementation_id"`
		RollbackImplementationID              string `json:"rollback_implementation_id"`
		WriterReceiptID                       string `json:"writer_receipt_id"`
		AdmissionLedgerID                     string `json:"admission_ledger_id"`
		CandidateRunID                        string `json:"candidate_run_id"`
		CandidateTextHash                     string `json:"candidate_text_hash"`
		TurnTextHash                          string `json:"turn_text_hash"`
		AdmissionFinalGateState               string `json:"admission_final_gate_state"`
		AdmissionFinalGateAction              string `json:"admission_final_gate_action"`
		AdmissionFinalGateTarget              string `json:"admission_final_gate_target"`
		AdmissionFinalGateTargetKind          string `json:"admission_final_gate_target_kind"`
		AdmissionFinalGateTargetMode          string `json:"admission_final_gate_target_mode"`
		AdmissionFinalGateReceiptShape        string `json:"admission_final_gate_receipt_shape"`
		AdmissionFinalGateDryRunOnly          bool   `json:"admission_final_gate_dry_run_only"`
		AdmissionFinalGateSealVerified        bool   `json:"admission_final_gate_seal_verified"`
		AdmissionFinalGatePermitVerified      bool   `json:"admission_final_gate_permit_verified"`
		AdmissionFinalGateReadinessVerified   bool   `json:"admission_final_gate_readiness_verified"`
		AdmissionFinalGateLedgerVerified      bool   `json:"admission_final_gate_ledger_verified"`
		AdmissionFinalGateWriterReady         bool   `json:"admission_final_gate_writer_ready"`
		AdmissionFinalGateRollbackReady       bool   `json:"admission_final_gate_rollback_ready"`
		AdmissionFinalGateLedgerReady         bool   `json:"admission_final_gate_ledger_ready"`
		AdmissionFinalGateReady               bool   `json:"admission_final_gate_ready"`
		SourceAdmissionSealID                 string `json:"source_admission_seal_id"`
		SourceAdmissionPermitIDForFinalGate   string `json:"source_admission_permit_id_for_final_gate"`
		SourceAdmissionReadinessIDForGate     string `json:"source_admission_readiness_id_for_final_gate"`
		SourceLedgerVerificationIDForGate     string `json:"source_ledger_verification_id_for_final_gate"`
		SourceLedgerPersistenceIDForGate      string `json:"source_ledger_persistence_id_for_final_gate"`
		SourceLedgerImplementationIDForGate   string `json:"source_ledger_implementation_id_for_final_gate"`
		SourceAdmissionLedgerIDForGate        string `json:"source_admission_ledger_id_for_final_gate"`
		SourceRollbackImplementationIDForGate string `json:"source_rollback_implementation_id_for_final_gate"`
		SourceWriterReceiptIDForGate          string `json:"source_writer_receipt_id_for_final_gate"`
		ContractsReady                        bool   `json:"contracts_ready"`
		BodyTarget                            string `json:"body_target"`
		WriteAllowed                          bool   `json:"write_allowed"`
		AdmissionAllowed                      bool   `json:"admission_allowed"`
		LiveAdmissionEnabled                  bool   `json:"live_admission_enabled"`
		MutatesState                          bool   `json:"mutates_state"`
	}{
		AdmissionSealID:                       finalGate.AdmissionSealID,
		AdmissionPermitID:                     finalGate.AdmissionPermitID,
		AdmissionReadinessID:                  finalGate.AdmissionReadinessID,
		LedgerVerificationID:                  finalGate.LedgerVerificationID,
		LedgerPersistenceID:                   finalGate.LedgerPersistenceID,
		LedgerImplementationID:                finalGate.LedgerImplementationID,
		RollbackImplementationID:              finalGate.RollbackImplementationID,
		WriterReceiptID:                       finalGate.WriterReceiptID,
		AdmissionLedgerID:                     finalGate.AdmissionLedgerID,
		CandidateRunID:                        finalGate.CandidateRunID,
		CandidateTextHash:                     finalGate.CandidateTextHash,
		TurnTextHash:                          finalGate.TurnTextHash,
		AdmissionFinalGateState:               finalGate.AdmissionFinalGateState,
		AdmissionFinalGateAction:              finalGate.AdmissionFinalGateAction,
		AdmissionFinalGateTarget:              finalGate.AdmissionFinalGateTarget,
		AdmissionFinalGateTargetKind:          finalGate.AdmissionFinalGateTargetKind,
		AdmissionFinalGateTargetMode:          finalGate.AdmissionFinalGateTargetMode,
		AdmissionFinalGateReceiptShape:        finalGate.AdmissionFinalGateReceiptShape,
		AdmissionFinalGateDryRunOnly:          finalGate.AdmissionFinalGateDryRunOnly,
		AdmissionFinalGateSealVerified:        finalGate.AdmissionFinalGateSealVerified,
		AdmissionFinalGatePermitVerified:      finalGate.AdmissionFinalGatePermitVerified,
		AdmissionFinalGateReadinessVerified:   finalGate.AdmissionFinalGateReadinessVerified,
		AdmissionFinalGateLedgerVerified:      finalGate.AdmissionFinalGateLedgerVerified,
		AdmissionFinalGateWriterReady:         finalGate.AdmissionFinalGateWriterReady,
		AdmissionFinalGateRollbackReady:       finalGate.AdmissionFinalGateRollbackReady,
		AdmissionFinalGateLedgerReady:         finalGate.AdmissionFinalGateLedgerReady,
		AdmissionFinalGateReady:               finalGate.AdmissionFinalGateReady,
		SourceAdmissionSealID:                 finalGate.SourceAdmissionSealID,
		SourceAdmissionPermitIDForFinalGate:   finalGate.SourceAdmissionPermitIDForFinalGate,
		SourceAdmissionReadinessIDForGate:     finalGate.SourceAdmissionReadinessIDForFinalGate,
		SourceLedgerVerificationIDForGate:     finalGate.SourceLedgerVerificationIDForFinalGate,
		SourceLedgerPersistenceIDForGate:      finalGate.SourceLedgerPersistenceIDForFinalGate,
		SourceLedgerImplementationIDForGate:   finalGate.SourceLedgerImplementationIDForFinalGate,
		SourceAdmissionLedgerIDForGate:        finalGate.SourceAdmissionLedgerIDForFinalGate,
		SourceRollbackImplementationIDForGate: finalGate.SourceRollbackImplementationIDForFinalGate,
		SourceWriterReceiptIDForGate:          finalGate.SourceWriterReceiptIDForFinalGate,
		ContractsReady:                        finalGate.ContractsReady,
		BodyTarget:                            finalGate.BodyTarget,
		WriteAllowed:                          finalGate.WriteAllowed,
		AdmissionAllowed:                      finalGate.AdmissionAllowed,
		LiveAdmissionEnabled:                  finalGate.LiveAdmissionEnabled,
		MutatesState:                          finalGate.MutatesState,
	})
	if h == "" {
		return ""
	}
	return "admission-final-gate-" + h
}

func recordAdmissionLiveRouteTurnCandidateAdmissionFinalGate(finalGate admissionLiveRouteTurnCandidateAdmissionFinalGate) error {
	path := strings.TrimSpace(os.Getenv("AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_FINAL_GATE_LOG"))
	if path == "" {
		return nil
	}
	f, err := os.OpenFile(path, os.O_CREATE|os.O_APPEND|os.O_WRONLY, 0600)
	if err != nil {
		return err
	}
	enc := json.NewEncoder(f)
	err = enc.Encode(finalGate)
	if closeErr := f.Close(); err == nil {
		err = closeErr
	}
	return err
}

func admissionLiveRouteTurnCandidateAdmissionResonanceIntentForFinalGate(finalGate admissionLiveRouteTurnCandidateAdmissionFinalGate) admissionLiveRouteTurnCandidateAdmissionResonanceIntent {
	sourceSchema := finalGate.Schema
	intent := admissionLiveRouteTurnCandidateAdmissionResonanceIntent{
		admissionLiveRouteTurnCandidateAdmissionFinalGate: finalGate,
		AdmissionResonanceIntentState:                     "blocked",
		AdmissionResonanceIntentAction:                    "reject",
		AdmissionResonanceIntentDryRunOnly:                true,
		SourceAdmissionFinalGateSchema:                    sourceSchema,
		SourceAdmissionFinalGatePassed:                    finalGate.Passed,
		SourceAdmissionFinalGateID:                        finalGate.AdmissionFinalGateID,
		SourceAdmissionFinalGateAction:                    finalGate.AdmissionFinalGateAction,
		SourceAdmissionFinalGateReady:                     finalGate.AdmissionFinalGateReady,
		SourceAdmissionSealIDForResonanceIntent:           finalGate.AdmissionSealID,
		SourceAdmissionPermitIDForResonanceIntent:         finalGate.AdmissionPermitID,
		SourceAdmissionReadinessIDForResonanceIntent:      finalGate.AdmissionReadinessID,
		SourceLedgerVerificationIDForResonanceIntent:      finalGate.LedgerVerificationID,
		SourceLedgerPersistenceIDForResonanceIntent:       finalGate.LedgerPersistenceID,
		SourceLedgerImplementationIDForResonanceIntent:    finalGate.LedgerImplementationID,
		SourceAdmissionLedgerIDForResonanceIntent:         finalGate.AdmissionLedgerID,
		SourceRollbackImplementationIDForResonanceIntent:  finalGate.RollbackImplementationID,
		SourceWriterReceiptIDForResonanceIntent:           finalGate.WriterReceiptID,
	}
	intent.Schema = admissionLiveRouteTurnCandidateAdmissionResonanceIntentSchema
	intent.Timing = "live_admission_resonance_intent"
	intent.Passed = false
	intent.AdmissionResonanceIntentID = ""
	intent.AdmissionResonanceIntentReady = false
	intent.LiveReady = false
	intent.BodyTarget = "none"
	intent.ContractsReady = false
	intent.WriteAllowed = false
	intent.AdmissionAllowed = false
	intent.LiveAdmissionEnabled = false
	intent.MutatesState = false

	if sourceSchema == "" {
		intent.Reason = "missing_candidate_admission_final_gate"
		return intent
	}
	if sourceSchema != admissionLiveRouteTurnCandidateAdmissionFinalGateSchema {
		intent.Reason = "unexpected_candidate_admission_final_gate_schema " + sourceSchema
		return intent
	}
	if !finalGate.Passed {
		intent.Reason = "candidate_admission_final_gate_failed"
		if finalGate.Reason != "" {
			intent.Reason += ": " + finalGate.Reason
		}
		return intent
	}
	if finalGate.AdmissionFinalGateID == "" {
		intent.Reason = "missing_candidate_admission_final_gate_id"
		return intent
	}
	if wantFinalGateID := admissionLiveRouteTurnCandidateAdmissionFinalGateID(finalGate); wantFinalGateID == "" || finalGate.AdmissionFinalGateID != wantFinalGateID {
		intent.Reason = "candidate_admission_final_gate_id_mismatch"
		return intent
	}
	if finalGate.AdmissionFinalGateState != "ready_closed_dry_run" ||
		finalGate.AdmissionFinalGateAction != "verify_sealed_admission_provenance_dry_run" ||
		finalGate.AdmissionFinalGateTarget != "live_admission" ||
		finalGate.AdmissionFinalGateTargetKind != "dream_candidate_admission" ||
		finalGate.AdmissionFinalGateTargetMode != "final_gate_closed_dry_run" ||
		finalGate.AdmissionFinalGateReceiptShape != "sealed_candidate_contract_provenance" {
		intent.Reason = "candidate_admission_final_gate_shape_mismatch"
		return intent
	}
	if !finalGate.AdmissionFinalGateDryRunOnly ||
		!finalGate.AdmissionFinalGateSealVerified ||
		!finalGate.AdmissionFinalGatePermitVerified ||
		!finalGate.AdmissionFinalGateReadinessVerified ||
		!finalGate.AdmissionFinalGateLedgerVerified ||
		!finalGate.AdmissionFinalGateWriterReady ||
		!finalGate.AdmissionFinalGateRollbackReady ||
		!finalGate.AdmissionFinalGateLedgerReady ||
		!finalGate.AdmissionFinalGateReady {
		intent.Reason = "candidate_admission_final_gate_not_verified_dry_run"
		return intent
	}
	if finalGate.ContractsReady || finalGate.WriteAllowed || finalGate.MutatesState || finalGate.LiveAdmissionEnabled || finalGate.AdmissionAllowed {
		intent.Reason = "candidate_admission_final_gate_already_open"
		return intent
	}
	if finalGate.BodyTarget != "none" {
		intent.Reason = "candidate_admission_final_gate_body_target_mismatch"
		return intent
	}
	if !finalGate.LiveReady {
		intent.Reason = "candidate_admission_final_gate_not_live_ready"
		return intent
	}
	if finalGate.SourceAdmissionSealSchema != admissionLiveRouteTurnCandidateAdmissionSealSchema ||
		!finalGate.SourceAdmissionSealPassed ||
		finalGate.SourceAdmissionSealID != finalGate.AdmissionSealID ||
		finalGate.SourceAdmissionSealAction != "seal_operator_permit_provenance_dry_run" ||
		!finalGate.SourceAdmissionSealReady {
		intent.Reason = "candidate_admission_final_gate_source_seal_mismatch"
		return intent
	}
	if wantSealID := admissionLiveRouteTurnCandidateAdmissionSealID(finalGate.admissionLiveRouteTurnCandidateAdmissionSeal); wantSealID == "" || finalGate.AdmissionSealID != wantSealID {
		intent.Reason = "candidate_admission_seal_id_mismatch_for_resonance_intent"
		return intent
	}
	if finalGate.SourceAdmissionPermitIDForFinalGate != finalGate.AdmissionPermitID ||
		finalGate.SourceAdmissionReadinessIDForFinalGate != finalGate.AdmissionReadinessID ||
		finalGate.SourceLedgerVerificationIDForFinalGate != finalGate.LedgerVerificationID ||
		finalGate.SourceLedgerPersistenceIDForFinalGate != finalGate.LedgerPersistenceID ||
		finalGate.SourceLedgerImplementationIDForFinalGate != finalGate.LedgerImplementationID ||
		finalGate.SourceAdmissionLedgerIDForFinalGate != finalGate.AdmissionLedgerID ||
		finalGate.SourceRollbackImplementationIDForFinalGate != finalGate.RollbackImplementationID ||
		finalGate.SourceWriterReceiptIDForFinalGate != finalGate.WriterReceiptID {
		intent.Reason = "candidate_admission_final_gate_source_id_mismatch"
		return intent
	}
	if finalGate.AdmissionFinalGateID == "" ||
		finalGate.AdmissionSealID == "" ||
		finalGate.AdmissionPermitID == "" ||
		finalGate.AdmissionReadinessID == "" ||
		finalGate.LedgerVerificationID == "" ||
		finalGate.LedgerPersistenceID == "" ||
		finalGate.LedgerImplementationID == "" ||
		finalGate.RollbackImplementationID == "" ||
		finalGate.WriterReceiptID == "" ||
		finalGate.WriterImplementationID == "" ||
		finalGate.AdmissionLedgerID == "" ||
		finalGate.AdmissionWriterContractID == "" ||
		finalGate.AdmissionWriterInventoryID == "" ||
		finalGate.AdmissionWriterPreflightID == "" ||
		finalGate.AdmissionLiveStageID == "" ||
		finalGate.AdmissionEnableGateID == "" ||
		finalGate.AdmissionSwitchID == "" ||
		finalGate.AdmissionPromotionID == "" ||
		finalGate.AdmissionDecisionID == "" ||
		finalGate.AdmissionAdapterID == "" ||
		finalGate.CandidateRunID == "" ||
		finalGate.CandidateDraftID == "" ||
		finalGate.CandidateExecutionID == "" ||
		finalGate.GeneratorAdapterID == "" ||
		finalGate.HandoffID == "" ||
		finalGate.DreamCandidateRunID == "" ||
		finalGate.CandidateTextHash == "" ||
		finalGate.TurnTextHash == "" {
		intent.Reason = "candidate_admission_final_gate_missing_provenance"
		return intent
	}

	intent.AdmissionResonanceIntentState = "resonance_intent_drafted_dry_run"
	intent.AdmissionResonanceIntentAction = "draft_resonance_direction_intent_dry_run"
	intent.AdmissionResonanceIntentTarget = "resonance"
	intent.AdmissionResonanceIntentTargetKind = "first_live_receiver"
	intent.AdmissionResonanceIntentTargetMode = "bounded_direction_dry_run"
	intent.AdmissionResonanceIntentReceiptShape = "sealed_candidate_contract_provenance"
	intent.AdmissionResonanceIntentDryRunOnly = true
	intent.AdmissionResonanceIntentFinalGateVerified = true
	intent.AdmissionResonanceIntentSealVerified = finalGate.AdmissionFinalGateSealVerified
	intent.AdmissionResonanceIntentPermitVerified = finalGate.AdmissionFinalGatePermitVerified
	intent.AdmissionResonanceIntentReadinessVerified = finalGate.AdmissionFinalGateReadinessVerified
	intent.AdmissionResonanceIntentLedgerVerified = finalGate.AdmissionFinalGateLedgerVerified
	intent.AdmissionResonanceIntentWriterReady = finalGate.AdmissionFinalGateWriterReady
	intent.AdmissionResonanceIntentRollbackReady = finalGate.AdmissionFinalGateRollbackReady
	intent.AdmissionResonanceIntentLedgerReady = finalGate.AdmissionFinalGateLedgerReady
	intent.AdmissionResonanceIntentReceiver = "resonance"
	intent.AdmissionResonanceIntentReceiverKind = "internal_world"
	intent.AdmissionResonanceIntentInfluenceKind = "bounded_direction"
	intent.AdmissionResonanceIntentMaxInfluence = admissionLiveRouteTurnCandidateAdmissionResonanceIntentMaxGain
	intent.AdmissionResonanceIntentTTLTurns = admissionLiveRouteTurnCandidateAdmissionResonanceIntentTTL
	intent.AdmissionResonanceIntentRawDreamTextAllowed = false
	intent.AdmissionResonanceIntentJanusSurfaceAllowed = false
	intent.AdmissionResonanceIntentCoocLearningAllowed = false
	intent.AdmissionResonanceIntentDeltaHarvestAllowed = false
	intent.AdmissionResonanceIntentRollbackRequired = true
	intent.AdmissionResonanceIntentPreStateHashRequired = true
	intent.AdmissionResonanceIntentPostStateHashRequired = true
	intent.AdmissionResonanceIntentReady = true
	intent.LiveReady = true
	intent.BodyTarget = "none"
	intent.ContractsReady = false
	intent.WriteAllowed = false
	intent.AdmissionAllowed = false
	intent.LiveAdmissionEnabled = false
	intent.MutatesState = false
	intent.AdmissionResonanceIntentCausalID = admissionLiveRouteTurnCandidateAdmissionResonanceIntentCausalID(intent)
	if intent.AdmissionResonanceIntentCausalID == "" {
		intent.Reason = "missing_candidate_admission_resonance_intent_causal_id"
		return intent
	}
	intent.AdmissionResonanceIntentID = admissionLiveRouteTurnCandidateAdmissionResonanceIntentID(intent)
	if intent.AdmissionResonanceIntentID == "" {
		intent.Reason = "missing_candidate_admission_resonance_intent_id"
		return intent
	}
	intent.Passed = true
	intent.Reason = "resonance intent drafted from final gate; live admission remains disabled"
	return intent
}

func admissionLiveRouteTurnCandidateAdmissionResonanceIntentCausalID(intent admissionLiveRouteTurnCandidateAdmissionResonanceIntent) string {
	h := hashJSON(struct {
		AdmissionFinalGateID string `json:"admission_final_gate_id"`
		AdmissionSealID      string `json:"admission_seal_id"`
		CandidateRunID       string `json:"candidate_run_id"`
		CandidateTextHash    string `json:"candidate_text_hash"`
		TurnTextHash         string `json:"turn_text_hash"`
		Receiver             string `json:"receiver"`
	}{
		AdmissionFinalGateID: intent.AdmissionFinalGateID,
		AdmissionSealID:      intent.AdmissionSealID,
		CandidateRunID:       intent.CandidateRunID,
		CandidateTextHash:    intent.CandidateTextHash,
		TurnTextHash:         intent.TurnTextHash,
		Receiver:             intent.AdmissionResonanceIntentReceiver,
	})
	if h == "" {
		return ""
	}
	return "resonance-intent-causal-" + h
}

func admissionLiveRouteTurnCandidateAdmissionResonanceIntentID(intent admissionLiveRouteTurnCandidateAdmissionResonanceIntent) string {
	h := hashJSON(struct {
		AdmissionFinalGateID                        string  `json:"admission_final_gate_id"`
		AdmissionSealID                             string  `json:"admission_seal_id"`
		AdmissionPermitID                           string  `json:"admission_permit_id"`
		LedgerVerificationID                        string  `json:"ledger_verification_id"`
		AdmissionResonanceIntentState               string  `json:"admission_resonance_intent_state"`
		AdmissionResonanceIntentAction              string  `json:"admission_resonance_intent_action"`
		AdmissionResonanceIntentTarget              string  `json:"admission_resonance_intent_target"`
		AdmissionResonanceIntentTargetKind          string  `json:"admission_resonance_intent_target_kind"`
		AdmissionResonanceIntentTargetMode          string  `json:"admission_resonance_intent_target_mode"`
		AdmissionResonanceIntentReceiptShape        string  `json:"admission_resonance_intent_receipt_shape"`
		AdmissionResonanceIntentDryRunOnly          bool    `json:"admission_resonance_intent_dry_run_only"`
		AdmissionResonanceIntentFinalGateVerified   bool    `json:"admission_resonance_intent_final_gate_verified"`
		AdmissionResonanceIntentReceiver            string  `json:"admission_resonance_intent_receiver"`
		AdmissionResonanceIntentReceiverKind        string  `json:"admission_resonance_intent_receiver_kind"`
		AdmissionResonanceIntentInfluenceKind       string  `json:"admission_resonance_intent_influence_kind"`
		AdmissionResonanceIntentMaxInfluence        float64 `json:"admission_resonance_intent_max_influence"`
		AdmissionResonanceIntentTTLTurns            int     `json:"admission_resonance_intent_ttl_turns"`
		AdmissionResonanceIntentCausalID            string  `json:"admission_resonance_intent_causal_id"`
		AdmissionResonanceIntentRawTextAllowed      bool    `json:"admission_resonance_intent_raw_dream_text_allowed"`
		AdmissionResonanceIntentJanusAllowed        bool    `json:"admission_resonance_intent_janus_surface_allowed"`
		AdmissionResonanceIntentCoocLearningAllowed bool    `json:"admission_resonance_intent_cooc_learning_allowed"`
		AdmissionResonanceIntentDeltaHarvestAllowed bool    `json:"admission_resonance_intent_delta_harvest_allowed"`
		AdmissionResonanceIntentRollbackRequired    bool    `json:"admission_resonance_intent_rollback_required"`
		AdmissionResonanceIntentPreHashRequired     bool    `json:"admission_resonance_intent_pre_state_hash_required"`
		AdmissionResonanceIntentPostHashRequired    bool    `json:"admission_resonance_intent_post_state_hash_required"`
		AdmissionResonanceIntentReady               bool    `json:"admission_resonance_intent_ready"`
		SourceAdmissionFinalGateID                  string  `json:"source_admission_final_gate_id"`
		SourceAdmissionSealIDForIntent              string  `json:"source_admission_seal_id_for_resonance_intent"`
		SourceWriterReceiptIDForIntent              string  `json:"source_writer_receipt_id_for_resonance_intent"`
		ContractsReady                              bool    `json:"contracts_ready"`
		BodyTarget                                  string  `json:"body_target"`
		WriteAllowed                                bool    `json:"write_allowed"`
		AdmissionAllowed                            bool    `json:"admission_allowed"`
		LiveAdmissionEnabled                        bool    `json:"live_admission_enabled"`
		MutatesState                                bool    `json:"mutates_state"`
	}{
		AdmissionFinalGateID:                        intent.AdmissionFinalGateID,
		AdmissionSealID:                             intent.AdmissionSealID,
		AdmissionPermitID:                           intent.AdmissionPermitID,
		LedgerVerificationID:                        intent.LedgerVerificationID,
		AdmissionResonanceIntentState:               intent.AdmissionResonanceIntentState,
		AdmissionResonanceIntentAction:              intent.AdmissionResonanceIntentAction,
		AdmissionResonanceIntentTarget:              intent.AdmissionResonanceIntentTarget,
		AdmissionResonanceIntentTargetKind:          intent.AdmissionResonanceIntentTargetKind,
		AdmissionResonanceIntentTargetMode:          intent.AdmissionResonanceIntentTargetMode,
		AdmissionResonanceIntentReceiptShape:        intent.AdmissionResonanceIntentReceiptShape,
		AdmissionResonanceIntentDryRunOnly:          intent.AdmissionResonanceIntentDryRunOnly,
		AdmissionResonanceIntentFinalGateVerified:   intent.AdmissionResonanceIntentFinalGateVerified,
		AdmissionResonanceIntentReceiver:            intent.AdmissionResonanceIntentReceiver,
		AdmissionResonanceIntentReceiverKind:        intent.AdmissionResonanceIntentReceiverKind,
		AdmissionResonanceIntentInfluenceKind:       intent.AdmissionResonanceIntentInfluenceKind,
		AdmissionResonanceIntentMaxInfluence:        intent.AdmissionResonanceIntentMaxInfluence,
		AdmissionResonanceIntentTTLTurns:            intent.AdmissionResonanceIntentTTLTurns,
		AdmissionResonanceIntentCausalID:            intent.AdmissionResonanceIntentCausalID,
		AdmissionResonanceIntentRawTextAllowed:      intent.AdmissionResonanceIntentRawDreamTextAllowed,
		AdmissionResonanceIntentJanusAllowed:        intent.AdmissionResonanceIntentJanusSurfaceAllowed,
		AdmissionResonanceIntentCoocLearningAllowed: intent.AdmissionResonanceIntentCoocLearningAllowed,
		AdmissionResonanceIntentDeltaHarvestAllowed: intent.AdmissionResonanceIntentDeltaHarvestAllowed,
		AdmissionResonanceIntentRollbackRequired:    intent.AdmissionResonanceIntentRollbackRequired,
		AdmissionResonanceIntentPreHashRequired:     intent.AdmissionResonanceIntentPreStateHashRequired,
		AdmissionResonanceIntentPostHashRequired:    intent.AdmissionResonanceIntentPostStateHashRequired,
		AdmissionResonanceIntentReady:               intent.AdmissionResonanceIntentReady,
		SourceAdmissionFinalGateID:                  intent.SourceAdmissionFinalGateID,
		SourceAdmissionSealIDForIntent:              intent.SourceAdmissionSealIDForResonanceIntent,
		SourceWriterReceiptIDForIntent:              intent.SourceWriterReceiptIDForResonanceIntent,
		ContractsReady:                              intent.ContractsReady,
		BodyTarget:                                  intent.BodyTarget,
		WriteAllowed:                                intent.WriteAllowed,
		AdmissionAllowed:                            intent.AdmissionAllowed,
		LiveAdmissionEnabled:                        intent.LiveAdmissionEnabled,
		MutatesState:                                intent.MutatesState,
	})
	if h == "" {
		return ""
	}
	return "resonance-intent-" + h
}

func recordAdmissionLiveRouteTurnCandidateAdmissionResonanceIntent(intent admissionLiveRouteTurnCandidateAdmissionResonanceIntent) error {
	path := strings.TrimSpace(os.Getenv("AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_RESONANCE_INTENT_LOG"))
	if path == "" {
		return nil
	}
	f, err := os.OpenFile(path, os.O_CREATE|os.O_APPEND|os.O_WRONLY, 0600)
	if err != nil {
		return err
	}
	enc := json.NewEncoder(f)
	err = enc.Encode(intent)
	if closeErr := f.Close(); err == nil {
		err = closeErr
	}
	return err
}

func admissionLiveRouteTurnCandidateAdmissionResonanceReceiverForIntent(intent admissionLiveRouteTurnCandidateAdmissionResonanceIntent) admissionLiveRouteTurnCandidateAdmissionResonanceReceiver {
	sourceSchema := intent.Schema
	receiver := admissionLiveRouteTurnCandidateAdmissionResonanceReceiver{
		admissionLiveRouteTurnCandidateAdmissionResonanceIntent: intent,
		AdmissionResonanceReceiverState:                         "blocked",
		AdmissionResonanceReceiverAction:                        "reject",
		AdmissionResonanceReceiverDryRunOnly:                    true,
		SourceAdmissionResonanceIntentSchema:                    sourceSchema,
		SourceAdmissionResonanceIntentPassed:                    intent.Passed,
		SourceAdmissionResonanceIntentID:                        intent.AdmissionResonanceIntentID,
		SourceAdmissionResonanceIntentAction:                    intent.AdmissionResonanceIntentAction,
		SourceAdmissionResonanceIntentReady:                     intent.AdmissionResonanceIntentReady,
		SourceAdmissionResonanceIntentCausalID:                  intent.AdmissionResonanceIntentCausalID,
		SourceAdmissionFinalGateIDForResonanceReceiver:          intent.AdmissionFinalGateID,
		SourceAdmissionSealIDForResonanceReceiver:               intent.AdmissionSealID,
		SourceAdmissionPermitIDForResonanceReceiver:             intent.AdmissionPermitID,
		SourceAdmissionReadinessIDForResonanceReceiver:          intent.AdmissionReadinessID,
		SourceLedgerVerificationIDForResonanceReceiver:          intent.LedgerVerificationID,
		SourceLedgerPersistenceIDForResonanceReceiver:           intent.LedgerPersistenceID,
		SourceLedgerImplementationIDForResonanceReceiver:        intent.LedgerImplementationID,
		SourceAdmissionLedgerIDForResonanceReceiver:             intent.AdmissionLedgerID,
		SourceRollbackImplementationIDForResonanceReceiver:      intent.RollbackImplementationID,
		SourceWriterReceiptIDForResonanceReceiver:               intent.WriterReceiptID,
	}
	receiver.Schema = admissionLiveRouteTurnCandidateAdmissionResonanceReceiverSchema
	receiver.Timing = "live_admission_resonance_receiver"
	receiver.Passed = false
	receiver.AdmissionResonanceReceiverID = ""
	receiver.AdmissionResonanceReceiverReady = false
	receiver.LiveReady = false
	receiver.BodyTarget = "none"
	receiver.ContractsReady = false
	receiver.WriteAllowed = false
	receiver.AdmissionAllowed = false
	receiver.LiveAdmissionEnabled = false
	receiver.MutatesState = false

	if sourceSchema == "" {
		receiver.Reason = "missing_candidate_admission_resonance_intent"
		return receiver
	}
	if sourceSchema != admissionLiveRouteTurnCandidateAdmissionResonanceIntentSchema {
		receiver.Reason = "unexpected_candidate_admission_resonance_intent_schema " + sourceSchema
		return receiver
	}
	if !intent.Passed {
		receiver.Reason = "candidate_admission_resonance_intent_failed"
		if intent.Reason != "" {
			receiver.Reason += ": " + intent.Reason
		}
		return receiver
	}
	if intent.AdmissionResonanceIntentID == "" {
		receiver.Reason = "missing_candidate_admission_resonance_intent_id"
		return receiver
	}
	if wantIntentID := admissionLiveRouteTurnCandidateAdmissionResonanceIntentID(intent); wantIntentID == "" || intent.AdmissionResonanceIntentID != wantIntentID {
		receiver.Reason = "candidate_admission_resonance_intent_id_mismatch"
		return receiver
	}
	if intent.AdmissionResonanceIntentState != "resonance_intent_drafted_dry_run" ||
		intent.AdmissionResonanceIntentAction != "draft_resonance_direction_intent_dry_run" ||
		intent.AdmissionResonanceIntentTarget != "resonance" ||
		intent.AdmissionResonanceIntentTargetKind != "first_live_receiver" ||
		intent.AdmissionResonanceIntentTargetMode != "bounded_direction_dry_run" ||
		intent.AdmissionResonanceIntentReceiptShape != "sealed_candidate_contract_provenance" {
		receiver.Reason = "candidate_admission_resonance_intent_shape_mismatch"
		return receiver
	}
	if !intent.AdmissionResonanceIntentDryRunOnly ||
		!intent.AdmissionResonanceIntentFinalGateVerified ||
		!intent.AdmissionResonanceIntentSealVerified ||
		!intent.AdmissionResonanceIntentPermitVerified ||
		!intent.AdmissionResonanceIntentReadinessVerified ||
		!intent.AdmissionResonanceIntentLedgerVerified ||
		!intent.AdmissionResonanceIntentWriterReady ||
		!intent.AdmissionResonanceIntentRollbackReady ||
		!intent.AdmissionResonanceIntentLedgerReady ||
		!intent.AdmissionResonanceIntentReady {
		receiver.Reason = "candidate_admission_resonance_intent_not_verified_dry_run"
		return receiver
	}
	if intent.AdmissionResonanceIntentReceiver != "resonance" ||
		intent.AdmissionResonanceIntentReceiverKind != "internal_world" ||
		intent.AdmissionResonanceIntentInfluenceKind != "bounded_direction" ||
		intent.AdmissionResonanceIntentMaxInfluence != admissionLiveRouteTurnCandidateAdmissionResonanceIntentMaxGain ||
		intent.AdmissionResonanceIntentTTLTurns != admissionLiveRouteTurnCandidateAdmissionResonanceIntentTTL ||
		intent.AdmissionResonanceIntentCausalID == "" {
		receiver.Reason = "candidate_admission_resonance_intent_receiver_mismatch"
		return receiver
	}
	if wantCausalID := admissionLiveRouteTurnCandidateAdmissionResonanceIntentCausalID(intent); wantCausalID == "" || intent.AdmissionResonanceIntentCausalID != wantCausalID {
		receiver.Reason = "candidate_admission_resonance_intent_causal_id_mismatch"
		return receiver
	}
	if intent.AdmissionResonanceIntentRawDreamTextAllowed ||
		intent.AdmissionResonanceIntentJanusSurfaceAllowed ||
		intent.AdmissionResonanceIntentCoocLearningAllowed ||
		intent.AdmissionResonanceIntentDeltaHarvestAllowed ||
		!intent.AdmissionResonanceIntentRollbackRequired ||
		!intent.AdmissionResonanceIntentPreStateHashRequired ||
		!intent.AdmissionResonanceIntentPostStateHashRequired {
		receiver.Reason = "candidate_admission_resonance_intent_guard_mismatch"
		return receiver
	}
	if intent.ContractsReady || intent.WriteAllowed || intent.MutatesState || intent.LiveAdmissionEnabled || intent.AdmissionAllowed {
		receiver.Reason = "candidate_admission_resonance_intent_already_open"
		return receiver
	}
	if intent.BodyTarget != "none" {
		receiver.Reason = "candidate_admission_resonance_intent_body_target_mismatch"
		return receiver
	}
	if !intent.LiveReady {
		receiver.Reason = "candidate_admission_resonance_intent_not_live_ready"
		return receiver
	}
	if intent.SourceAdmissionFinalGateSchema != admissionLiveRouteTurnCandidateAdmissionFinalGateSchema ||
		!intent.SourceAdmissionFinalGatePassed ||
		intent.SourceAdmissionFinalGateID != intent.AdmissionFinalGateID ||
		intent.SourceAdmissionFinalGateAction != "verify_sealed_admission_provenance_dry_run" ||
		!intent.SourceAdmissionFinalGateReady {
		receiver.Reason = "candidate_admission_resonance_intent_source_final_gate_mismatch"
		return receiver
	}
	if wantFinalGateID := admissionLiveRouteTurnCandidateAdmissionFinalGateID(intent.admissionLiveRouteTurnCandidateAdmissionFinalGate); wantFinalGateID == "" || intent.AdmissionFinalGateID != wantFinalGateID {
		receiver.Reason = "candidate_admission_final_gate_id_mismatch_for_resonance_receiver"
		return receiver
	}
	if intent.SourceAdmissionSealIDForResonanceIntent != intent.AdmissionSealID ||
		intent.SourceAdmissionPermitIDForResonanceIntent != intent.AdmissionPermitID ||
		intent.SourceAdmissionReadinessIDForResonanceIntent != intent.AdmissionReadinessID ||
		intent.SourceLedgerVerificationIDForResonanceIntent != intent.LedgerVerificationID ||
		intent.SourceLedgerPersistenceIDForResonanceIntent != intent.LedgerPersistenceID ||
		intent.SourceLedgerImplementationIDForResonanceIntent != intent.LedgerImplementationID ||
		intent.SourceAdmissionLedgerIDForResonanceIntent != intent.AdmissionLedgerID ||
		intent.SourceRollbackImplementationIDForResonanceIntent != intent.RollbackImplementationID ||
		intent.SourceWriterReceiptIDForResonanceIntent != intent.WriterReceiptID {
		receiver.Reason = "candidate_admission_resonance_intent_source_id_mismatch"
		return receiver
	}
	if intent.AdmissionResonanceIntentID == "" ||
		intent.AdmissionFinalGateID == "" ||
		intent.AdmissionSealID == "" ||
		intent.AdmissionPermitID == "" ||
		intent.AdmissionReadinessID == "" ||
		intent.LedgerVerificationID == "" ||
		intent.LedgerPersistenceID == "" ||
		intent.LedgerImplementationID == "" ||
		intent.RollbackImplementationID == "" ||
		intent.WriterReceiptID == "" ||
		intent.WriterImplementationID == "" ||
		intent.AdmissionLedgerID == "" ||
		intent.AdmissionWriterContractID == "" ||
		intent.AdmissionWriterInventoryID == "" ||
		intent.AdmissionWriterPreflightID == "" ||
		intent.AdmissionLiveStageID == "" ||
		intent.AdmissionEnableGateID == "" ||
		intent.AdmissionSwitchID == "" ||
		intent.AdmissionPromotionID == "" ||
		intent.AdmissionDecisionID == "" ||
		intent.AdmissionAdapterID == "" ||
		intent.CandidateRunID == "" ||
		intent.CandidateDraftID == "" ||
		intent.CandidateExecutionID == "" ||
		intent.GeneratorAdapterID == "" ||
		intent.HandoffID == "" ||
		intent.DreamCandidateRunID == "" ||
		intent.CandidateTextHash == "" ||
		intent.TurnTextHash == "" {
		receiver.Reason = "candidate_admission_resonance_intent_missing_provenance"
		return receiver
	}

	receiver.AdmissionResonanceReceiverState = "receiver_previewed_dry_run"
	receiver.AdmissionResonanceReceiverAction = "preview_resonance_receive_dry_run"
	receiver.AdmissionResonanceReceiverTarget = "resonance"
	receiver.AdmissionResonanceReceiverTargetKind = "first_live_receiver"
	receiver.AdmissionResonanceReceiverTargetMode = "bounded_direction_preview_dry_run"
	receiver.AdmissionResonanceReceiverReceiptShape = "resonance_receiver_state_proof"
	receiver.AdmissionResonanceReceiverDryRunOnly = true
	receiver.AdmissionResonanceReceiverIntentVerified = true
	receiver.AdmissionResonanceReceiverFinalGateVerified = intent.AdmissionResonanceIntentFinalGateVerified
	receiver.AdmissionResonanceReceiverSealVerified = intent.AdmissionResonanceIntentSealVerified
	receiver.AdmissionResonanceReceiverPermitVerified = intent.AdmissionResonanceIntentPermitVerified
	receiver.AdmissionResonanceReceiverReadinessVerified = intent.AdmissionResonanceIntentReadinessVerified
	receiver.AdmissionResonanceReceiverLedgerVerified = intent.AdmissionResonanceIntentLedgerVerified
	receiver.AdmissionResonanceReceiverWriterReady = intent.AdmissionResonanceIntentWriterReady
	receiver.AdmissionResonanceReceiverRollbackReady = intent.AdmissionResonanceIntentRollbackReady
	receiver.AdmissionResonanceReceiverLedgerReady = intent.AdmissionResonanceIntentLedgerReady
	receiver.AdmissionResonanceReceiverReceiver = intent.AdmissionResonanceIntentReceiver
	receiver.AdmissionResonanceReceiverReceiverKind = intent.AdmissionResonanceIntentReceiverKind
	receiver.AdmissionResonanceReceiverInfluenceKind = intent.AdmissionResonanceIntentInfluenceKind
	receiver.AdmissionResonanceReceiverMaxInfluence = intent.AdmissionResonanceIntentMaxInfluence
	receiver.AdmissionResonanceReceiverTTLTurns = intent.AdmissionResonanceIntentTTLTurns
	receiver.AdmissionResonanceReceiverStateHashMode = "sealed_metadata_preview"
	receiver.AdmissionResonanceReceiverRawDreamTextObserved = false
	receiver.AdmissionResonanceReceiverRawDreamTextForwarded = false
	receiver.AdmissionResonanceReceiverJanusSurfaceAllowed = false
	receiver.AdmissionResonanceReceiverCoocLearningAllowed = false
	receiver.AdmissionResonanceReceiverDeltaHarvestAllowed = false
	receiver.AdmissionResonanceReceiverBodyMutationAllowed = false
	receiver.AdmissionResonanceReceiverRollbackRequired = true
	receiver.AdmissionResonanceReceiverReady = true
	receiver.LiveReady = true
	receiver.BodyTarget = "none"
	receiver.ContractsReady = false
	receiver.WriteAllowed = false
	receiver.AdmissionAllowed = false
	receiver.LiveAdmissionEnabled = false
	receiver.MutatesState = false
	receiver.AdmissionResonanceReceiverCausalID = admissionLiveRouteTurnCandidateAdmissionResonanceReceiverCausalID(receiver)
	if receiver.AdmissionResonanceReceiverCausalID == "" {
		receiver.Reason = "missing_candidate_admission_resonance_receiver_causal_id"
		return receiver
	}
	receiver.AdmissionResonanceReceiverPreStateHash = admissionLiveRouteTurnCandidateAdmissionResonanceReceiverPreStateHash(receiver)
	receiver.AdmissionResonanceReceiverPostStateHash = admissionLiveRouteTurnCandidateAdmissionResonanceReceiverPostStateHash(receiver)
	receiver.AdmissionResonanceReceiverStateDeltaHash = admissionLiveRouteTurnCandidateAdmissionResonanceReceiverStateDeltaHash(receiver)
	if receiver.AdmissionResonanceReceiverPreStateHash == "" ||
		receiver.AdmissionResonanceReceiverPostStateHash == "" ||
		receiver.AdmissionResonanceReceiverStateDeltaHash == "" ||
		receiver.AdmissionResonanceReceiverPreStateHash == receiver.AdmissionResonanceReceiverPostStateHash {
		receiver.Reason = "candidate_admission_resonance_receiver_state_proof_missing"
		return receiver
	}
	receiver.AdmissionResonanceReceiverID = admissionLiveRouteTurnCandidateAdmissionResonanceReceiverID(receiver)
	if receiver.AdmissionResonanceReceiverID == "" {
		receiver.Reason = "missing_candidate_admission_resonance_receiver_id"
		return receiver
	}
	receiver.Passed = true
	receiver.Reason = "resonance receiver previewed sealed intent without body mutation"
	return receiver
}

func admissionLiveRouteTurnCandidateAdmissionResonanceReceiverCausalID(receiver admissionLiveRouteTurnCandidateAdmissionResonanceReceiver) string {
	h := hashJSON(struct {
		AdmissionResonanceIntentID       string `json:"admission_resonance_intent_id"`
		AdmissionResonanceIntentCausalID string `json:"admission_resonance_intent_causal_id"`
		CandidateRunID                   string `json:"candidate_run_id"`
		CandidateTextHash                string `json:"candidate_text_hash"`
		TurnTextHash                     string `json:"turn_text_hash"`
		Receiver                         string `json:"receiver"`
		ReceiverKind                     string `json:"receiver_kind"`
	}{
		AdmissionResonanceIntentID:       receiver.AdmissionResonanceIntentID,
		AdmissionResonanceIntentCausalID: receiver.AdmissionResonanceIntentCausalID,
		CandidateRunID:                   receiver.CandidateRunID,
		CandidateTextHash:                receiver.CandidateTextHash,
		TurnTextHash:                     receiver.TurnTextHash,
		Receiver:                         receiver.AdmissionResonanceReceiverReceiver,
		ReceiverKind:                     receiver.AdmissionResonanceReceiverReceiverKind,
	})
	if h == "" {
		return ""
	}
	return "resonance-receiver-causal-" + h
}

func admissionLiveRouteTurnCandidateAdmissionResonanceReceiverPreStateHash(receiver admissionLiveRouteTurnCandidateAdmissionResonanceReceiver) string {
	h := hashJSON(struct {
		AdmissionResonanceIntentID       string `json:"admission_resonance_intent_id"`
		AdmissionResonanceIntentCausalID string `json:"admission_resonance_intent_causal_id"`
		CandidateRunID                   string `json:"candidate_run_id"`
		CandidateTextHash                string `json:"candidate_text_hash"`
		TurnTextHash                     string `json:"turn_text_hash"`
		StateHashMode                    string `json:"state_hash_mode"`
		Receiver                         string `json:"receiver"`
	}{
		AdmissionResonanceIntentID:       receiver.AdmissionResonanceIntentID,
		AdmissionResonanceIntentCausalID: receiver.AdmissionResonanceIntentCausalID,
		CandidateRunID:                   receiver.CandidateRunID,
		CandidateTextHash:                receiver.CandidateTextHash,
		TurnTextHash:                     receiver.TurnTextHash,
		StateHashMode:                    receiver.AdmissionResonanceReceiverStateHashMode,
		Receiver:                         receiver.AdmissionResonanceReceiverReceiver,
	})
	if h == "" {
		return ""
	}
	return "resonance-receiver-pre-" + h
}

func admissionLiveRouteTurnCandidateAdmissionResonanceReceiverPostStateHash(receiver admissionLiveRouteTurnCandidateAdmissionResonanceReceiver) string {
	h := hashJSON(struct {
		PreStateHash  string  `json:"pre_state_hash"`
		CausalID      string  `json:"causal_id"`
		Receiver      string  `json:"receiver"`
		InfluenceKind string  `json:"influence_kind"`
		MaxInfluence  float64 `json:"max_influence"`
		TTLTurns      int     `json:"ttl_turns"`
		StateHashMode string  `json:"state_hash_mode"`
	}{
		PreStateHash:  receiver.AdmissionResonanceReceiverPreStateHash,
		CausalID:      receiver.AdmissionResonanceReceiverCausalID,
		Receiver:      receiver.AdmissionResonanceReceiverReceiver,
		InfluenceKind: receiver.AdmissionResonanceReceiverInfluenceKind,
		MaxInfluence:  receiver.AdmissionResonanceReceiverMaxInfluence,
		TTLTurns:      receiver.AdmissionResonanceReceiverTTLTurns,
		StateHashMode: receiver.AdmissionResonanceReceiverStateHashMode,
	})
	if h == "" {
		return ""
	}
	return "resonance-receiver-post-" + h
}

func admissionLiveRouteTurnCandidateAdmissionResonanceReceiverStateDeltaHash(receiver admissionLiveRouteTurnCandidateAdmissionResonanceReceiver) string {
	h := hashJSON(struct {
		PreStateHash        string `json:"pre_state_hash"`
		PostStateHash       string `json:"post_state_hash"`
		AdmissionIntentID   string `json:"admission_resonance_intent_id"`
		ReceiverCausalID    string `json:"admission_resonance_receiver_causal_id"`
		RawTextObserved     bool   `json:"raw_text_observed"`
		JanusSurfaceAllowed bool   `json:"janus_surface_allowed"`
		CoocLearningAllowed bool   `json:"cooc_learning_allowed"`
		DeltaHarvestAllowed bool   `json:"delta_harvest_allowed"`
		BodyMutationAllowed bool   `json:"body_mutation_allowed"`
		RollbackRequired    bool   `json:"rollback_required"`
	}{
		PreStateHash:        receiver.AdmissionResonanceReceiverPreStateHash,
		PostStateHash:       receiver.AdmissionResonanceReceiverPostStateHash,
		AdmissionIntentID:   receiver.AdmissionResonanceIntentID,
		ReceiverCausalID:    receiver.AdmissionResonanceReceiverCausalID,
		RawTextObserved:     receiver.AdmissionResonanceReceiverRawDreamTextObserved,
		JanusSurfaceAllowed: receiver.AdmissionResonanceReceiverJanusSurfaceAllowed,
		CoocLearningAllowed: receiver.AdmissionResonanceReceiverCoocLearningAllowed,
		DeltaHarvestAllowed: receiver.AdmissionResonanceReceiverDeltaHarvestAllowed,
		BodyMutationAllowed: receiver.AdmissionResonanceReceiverBodyMutationAllowed,
		RollbackRequired:    receiver.AdmissionResonanceReceiverRollbackRequired,
	})
	if h == "" {
		return ""
	}
	return "resonance-receiver-delta-" + h
}

func admissionLiveRouteTurnCandidateAdmissionResonanceReceiverID(receiver admissionLiveRouteTurnCandidateAdmissionResonanceReceiver) string {
	h := hashJSON(struct {
		AdmissionResonanceIntentID              string  `json:"admission_resonance_intent_id"`
		AdmissionFinalGateID                    string  `json:"admission_final_gate_id"`
		AdmissionSealID                         string  `json:"admission_seal_id"`
		AdmissionPermitID                       string  `json:"admission_permit_id"`
		LedgerVerificationID                    string  `json:"ledger_verification_id"`
		AdmissionResonanceReceiverState         string  `json:"admission_resonance_receiver_state"`
		AdmissionResonanceReceiverAction        string  `json:"admission_resonance_receiver_action"`
		AdmissionResonanceReceiverTarget        string  `json:"admission_resonance_receiver_target"`
		AdmissionResonanceReceiverTargetKind    string  `json:"admission_resonance_receiver_target_kind"`
		AdmissionResonanceReceiverTargetMode    string  `json:"admission_resonance_receiver_target_mode"`
		AdmissionResonanceReceiverReceiptShape  string  `json:"admission_resonance_receiver_receipt_shape"`
		AdmissionResonanceReceiverDryRunOnly    bool    `json:"admission_resonance_receiver_dry_run_only"`
		AdmissionResonanceReceiverVerified      bool    `json:"admission_resonance_receiver_intent_verified"`
		AdmissionResonanceReceiverReceiver      string  `json:"admission_resonance_receiver_receiver"`
		AdmissionResonanceReceiverReceiverKind  string  `json:"admission_resonance_receiver_receiver_kind"`
		AdmissionResonanceReceiverInfluenceKind string  `json:"admission_resonance_receiver_influence_kind"`
		AdmissionResonanceReceiverMaxInfluence  float64 `json:"admission_resonance_receiver_max_influence"`
		AdmissionResonanceReceiverTTLTurns      int     `json:"admission_resonance_receiver_ttl_turns"`
		AdmissionResonanceReceiverCausalID      string  `json:"admission_resonance_receiver_causal_id"`
		AdmissionResonanceReceiverPreHash       string  `json:"admission_resonance_receiver_pre_state_hash"`
		AdmissionResonanceReceiverPostHash      string  `json:"admission_resonance_receiver_post_state_hash"`
		AdmissionResonanceReceiverDeltaHash     string  `json:"admission_resonance_receiver_state_delta_hash"`
		AdmissionResonanceReceiverHashMode      string  `json:"admission_resonance_receiver_state_hash_mode"`
		RawTextObserved                         bool    `json:"raw_text_observed"`
		RawTextForwarded                        bool    `json:"raw_text_forwarded"`
		JanusSurfaceAllowed                     bool    `json:"janus_surface_allowed"`
		CoocLearningAllowed                     bool    `json:"cooc_learning_allowed"`
		DeltaHarvestAllowed                     bool    `json:"delta_harvest_allowed"`
		BodyMutationAllowed                     bool    `json:"body_mutation_allowed"`
		RollbackRequired                        bool    `json:"rollback_required"`
		ReceiverReady                           bool    `json:"receiver_ready"`
		SourceAdmissionResonanceIntentID        string  `json:"source_admission_resonance_intent_id"`
		SourceAdmissionResonanceIntentCausalID  string  `json:"source_admission_resonance_intent_causal_id"`
		ContractsReady                          bool    `json:"contracts_ready"`
		BodyTarget                              string  `json:"body_target"`
		WriteAllowed                            bool    `json:"write_allowed"`
		AdmissionAllowed                        bool    `json:"admission_allowed"`
		LiveAdmissionEnabled                    bool    `json:"live_admission_enabled"`
		MutatesState                            bool    `json:"mutates_state"`
	}{
		AdmissionResonanceIntentID:              receiver.AdmissionResonanceIntentID,
		AdmissionFinalGateID:                    receiver.AdmissionFinalGateID,
		AdmissionSealID:                         receiver.AdmissionSealID,
		AdmissionPermitID:                       receiver.AdmissionPermitID,
		LedgerVerificationID:                    receiver.LedgerVerificationID,
		AdmissionResonanceReceiverState:         receiver.AdmissionResonanceReceiverState,
		AdmissionResonanceReceiverAction:        receiver.AdmissionResonanceReceiverAction,
		AdmissionResonanceReceiverTarget:        receiver.AdmissionResonanceReceiverTarget,
		AdmissionResonanceReceiverTargetKind:    receiver.AdmissionResonanceReceiverTargetKind,
		AdmissionResonanceReceiverTargetMode:    receiver.AdmissionResonanceReceiverTargetMode,
		AdmissionResonanceReceiverReceiptShape:  receiver.AdmissionResonanceReceiverReceiptShape,
		AdmissionResonanceReceiverDryRunOnly:    receiver.AdmissionResonanceReceiverDryRunOnly,
		AdmissionResonanceReceiverVerified:      receiver.AdmissionResonanceReceiverIntentVerified,
		AdmissionResonanceReceiverReceiver:      receiver.AdmissionResonanceReceiverReceiver,
		AdmissionResonanceReceiverReceiverKind:  receiver.AdmissionResonanceReceiverReceiverKind,
		AdmissionResonanceReceiverInfluenceKind: receiver.AdmissionResonanceReceiverInfluenceKind,
		AdmissionResonanceReceiverMaxInfluence:  receiver.AdmissionResonanceReceiverMaxInfluence,
		AdmissionResonanceReceiverTTLTurns:      receiver.AdmissionResonanceReceiverTTLTurns,
		AdmissionResonanceReceiverCausalID:      receiver.AdmissionResonanceReceiverCausalID,
		AdmissionResonanceReceiverPreHash:       receiver.AdmissionResonanceReceiverPreStateHash,
		AdmissionResonanceReceiverPostHash:      receiver.AdmissionResonanceReceiverPostStateHash,
		AdmissionResonanceReceiverDeltaHash:     receiver.AdmissionResonanceReceiverStateDeltaHash,
		AdmissionResonanceReceiverHashMode:      receiver.AdmissionResonanceReceiverStateHashMode,
		RawTextObserved:                         receiver.AdmissionResonanceReceiverRawDreamTextObserved,
		RawTextForwarded:                        receiver.AdmissionResonanceReceiverRawDreamTextForwarded,
		JanusSurfaceAllowed:                     receiver.AdmissionResonanceReceiverJanusSurfaceAllowed,
		CoocLearningAllowed:                     receiver.AdmissionResonanceReceiverCoocLearningAllowed,
		DeltaHarvestAllowed:                     receiver.AdmissionResonanceReceiverDeltaHarvestAllowed,
		BodyMutationAllowed:                     receiver.AdmissionResonanceReceiverBodyMutationAllowed,
		RollbackRequired:                        receiver.AdmissionResonanceReceiverRollbackRequired,
		ReceiverReady:                           receiver.AdmissionResonanceReceiverReady,
		SourceAdmissionResonanceIntentID:        receiver.SourceAdmissionResonanceIntentID,
		SourceAdmissionResonanceIntentCausalID:  receiver.SourceAdmissionResonanceIntentCausalID,
		ContractsReady:                          receiver.ContractsReady,
		BodyTarget:                              receiver.BodyTarget,
		WriteAllowed:                            receiver.WriteAllowed,
		AdmissionAllowed:                        receiver.AdmissionAllowed,
		LiveAdmissionEnabled:                    receiver.LiveAdmissionEnabled,
		MutatesState:                            receiver.MutatesState,
	})
	if h == "" {
		return ""
	}
	return "resonance-receiver-" + h
}

func recordAdmissionLiveRouteTurnCandidateAdmissionResonanceReceiver(receiver admissionLiveRouteTurnCandidateAdmissionResonanceReceiver) error {
	path := strings.TrimSpace(os.Getenv("AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_RESONANCE_RECEIVER_LOG"))
	if path == "" {
		return nil
	}
	f, err := os.OpenFile(path, os.O_CREATE|os.O_APPEND|os.O_WRONLY, 0600)
	if err != nil {
		return err
	}
	enc := json.NewEncoder(f)
	err = enc.Encode(receiver)
	if closeErr := f.Close(); err == nil {
		err = closeErr
	}
	return err
}

func admissionLiveRouteTurnCandidateAdmissionResonanceObservationForReceiver(receiver admissionLiveRouteTurnCandidateAdmissionResonanceReceiver) admissionLiveRouteTurnCandidateAdmissionResonanceObservation {
	sourceSchema := receiver.Schema
	observation := admissionLiveRouteTurnCandidateAdmissionResonanceObservation{
		admissionLiveRouteTurnCandidateAdmissionResonanceReceiver: receiver,
		AdmissionResonanceObservationState:                        "blocked",
		AdmissionResonanceObservationAction:                       "reject",
		AdmissionResonanceObservationDryRunOnly:                   true,
		SourceAdmissionResonanceReceiverSchema:                    sourceSchema,
		SourceAdmissionResonanceReceiverPassed:                    receiver.Passed,
		SourceAdmissionResonanceReceiverID:                        receiver.AdmissionResonanceReceiverID,
		SourceAdmissionResonanceReceiverAction:                    receiver.AdmissionResonanceReceiverAction,
		SourceAdmissionResonanceReceiverReady:                     receiver.AdmissionResonanceReceiverReady,
		SourceAdmissionResonanceReceiverCausalID:                  receiver.AdmissionResonanceReceiverCausalID,
		SourceAdmissionResonanceReceiverPreStateHash:              receiver.AdmissionResonanceReceiverPreStateHash,
		SourceAdmissionResonanceReceiverPostStateHash:             receiver.AdmissionResonanceReceiverPostStateHash,
		SourceAdmissionResonanceReceiverStateDeltaHash:            receiver.AdmissionResonanceReceiverStateDeltaHash,
		SourceAdmissionResonanceIntentIDForObservation:            receiver.AdmissionResonanceIntentID,
		SourceAdmissionFinalGateIDForResonanceObservation:         receiver.AdmissionFinalGateID,
		SourceAdmissionSealIDForResonanceObservation:              receiver.AdmissionSealID,
		SourceAdmissionPermitIDForResonanceObservation:            receiver.AdmissionPermitID,
		SourceAdmissionReadinessIDForResonanceObservation:         receiver.AdmissionReadinessID,
		SourceLedgerVerificationIDForResonanceObservation:         receiver.LedgerVerificationID,
		SourceLedgerPersistenceIDForResonanceObservation:          receiver.LedgerPersistenceID,
		SourceLedgerImplementationIDForResonanceObservation:       receiver.LedgerImplementationID,
		SourceAdmissionLedgerIDForResonanceObservation:            receiver.AdmissionLedgerID,
		SourceRollbackImplementationIDForResonanceObservation:     receiver.RollbackImplementationID,
		SourceWriterReceiptIDForResonanceObservation:              receiver.WriterReceiptID,
	}
	observation.Schema = admissionLiveRouteTurnCandidateAdmissionResonanceObservationSchema
	observation.Timing = "live_admission_resonance_observation"
	observation.Passed = false
	observation.AdmissionResonanceObservationID = ""
	observation.AdmissionResonanceObservationReady = false
	observation.LiveReady = false
	observation.BodyTarget = "none"
	observation.ContractsReady = false
	observation.WriteAllowed = false
	observation.AdmissionAllowed = false
	observation.LiveAdmissionEnabled = false
	observation.MutatesState = false

	if sourceSchema == "" {
		observation.Reason = "missing_candidate_admission_resonance_receiver"
		return observation
	}
	if sourceSchema != admissionLiveRouteTurnCandidateAdmissionResonanceReceiverSchema {
		observation.Reason = "unexpected_candidate_admission_resonance_receiver_schema " + sourceSchema
		return observation
	}
	if !receiver.Passed {
		observation.Reason = "candidate_admission_resonance_receiver_failed"
		if receiver.Reason != "" {
			observation.Reason += ": " + receiver.Reason
		}
		return observation
	}
	if receiver.AdmissionResonanceReceiverID == "" {
		observation.Reason = "missing_candidate_admission_resonance_receiver_id"
		return observation
	}
	if wantReceiverID := admissionLiveRouteTurnCandidateAdmissionResonanceReceiverID(receiver); wantReceiverID == "" || receiver.AdmissionResonanceReceiverID != wantReceiverID {
		observation.Reason = "candidate_admission_resonance_receiver_id_mismatch"
		return observation
	}
	if receiver.AdmissionResonanceReceiverState != "receiver_previewed_dry_run" ||
		receiver.AdmissionResonanceReceiverAction != "preview_resonance_receive_dry_run" ||
		receiver.AdmissionResonanceReceiverTarget != "resonance" ||
		receiver.AdmissionResonanceReceiverTargetKind != "first_live_receiver" ||
		receiver.AdmissionResonanceReceiverTargetMode != "bounded_direction_preview_dry_run" ||
		receiver.AdmissionResonanceReceiverReceiptShape != "resonance_receiver_state_proof" {
		observation.Reason = "candidate_admission_resonance_receiver_shape_mismatch"
		return observation
	}
	if !receiver.AdmissionResonanceReceiverDryRunOnly ||
		!receiver.AdmissionResonanceReceiverIntentVerified ||
		!receiver.AdmissionResonanceReceiverFinalGateVerified ||
		!receiver.AdmissionResonanceReceiverSealVerified ||
		!receiver.AdmissionResonanceReceiverPermitVerified ||
		!receiver.AdmissionResonanceReceiverReadinessVerified ||
		!receiver.AdmissionResonanceReceiverLedgerVerified ||
		!receiver.AdmissionResonanceReceiverWriterReady ||
		!receiver.AdmissionResonanceReceiverRollbackReady ||
		!receiver.AdmissionResonanceReceiverLedgerReady ||
		!receiver.AdmissionResonanceReceiverReady {
		observation.Reason = "candidate_admission_resonance_receiver_not_verified_dry_run"
		return observation
	}
	if receiver.AdmissionResonanceReceiverReceiver != "resonance" ||
		receiver.AdmissionResonanceReceiverReceiverKind != "internal_world" ||
		receiver.AdmissionResonanceReceiverInfluenceKind != "bounded_direction" ||
		receiver.AdmissionResonanceReceiverMaxInfluence != admissionLiveRouteTurnCandidateAdmissionResonanceIntentMaxGain ||
		receiver.AdmissionResonanceReceiverTTLTurns != admissionLiveRouteTurnCandidateAdmissionResonanceIntentTTL ||
		receiver.AdmissionResonanceReceiverCausalID == "" {
		observation.Reason = "candidate_admission_resonance_receiver_target_mismatch"
		return observation
	}
	if wantCausalID := admissionLiveRouteTurnCandidateAdmissionResonanceReceiverCausalID(receiver); wantCausalID == "" || receiver.AdmissionResonanceReceiverCausalID != wantCausalID {
		observation.Reason = "candidate_admission_resonance_receiver_causal_id_mismatch"
		return observation
	}
	if receiver.AdmissionResonanceReceiverPreStateHash == "" ||
		receiver.AdmissionResonanceReceiverPostStateHash == "" ||
		receiver.AdmissionResonanceReceiverStateDeltaHash == "" ||
		receiver.AdmissionResonanceReceiverPreStateHash == receiver.AdmissionResonanceReceiverPostStateHash ||
		receiver.AdmissionResonanceReceiverPreStateHash != admissionLiveRouteTurnCandidateAdmissionResonanceReceiverPreStateHash(receiver) ||
		receiver.AdmissionResonanceReceiverPostStateHash != admissionLiveRouteTurnCandidateAdmissionResonanceReceiverPostStateHash(receiver) ||
		receiver.AdmissionResonanceReceiverStateDeltaHash != admissionLiveRouteTurnCandidateAdmissionResonanceReceiverStateDeltaHash(receiver) ||
		receiver.AdmissionResonanceReceiverStateHashMode != "sealed_metadata_preview" {
		observation.Reason = "candidate_admission_resonance_receiver_state_proof_mismatch"
		return observation
	}
	if receiver.AdmissionResonanceReceiverRawDreamTextObserved ||
		receiver.AdmissionResonanceReceiverRawDreamTextForwarded ||
		receiver.AdmissionResonanceReceiverJanusSurfaceAllowed ||
		receiver.AdmissionResonanceReceiverCoocLearningAllowed ||
		receiver.AdmissionResonanceReceiverDeltaHarvestAllowed ||
		receiver.AdmissionResonanceReceiverBodyMutationAllowed ||
		!receiver.AdmissionResonanceReceiverRollbackRequired {
		observation.Reason = "candidate_admission_resonance_receiver_guard_mismatch"
		return observation
	}
	if receiver.ContractsReady || receiver.WriteAllowed || receiver.MutatesState || receiver.LiveAdmissionEnabled || receiver.AdmissionAllowed {
		observation.Reason = "candidate_admission_resonance_receiver_already_open"
		return observation
	}
	if receiver.BodyTarget != "none" {
		observation.Reason = "candidate_admission_resonance_receiver_body_target_mismatch"
		return observation
	}
	if !receiver.LiveReady {
		observation.Reason = "candidate_admission_resonance_receiver_not_live_ready"
		return observation
	}
	if receiver.SourceAdmissionResonanceIntentSchema != admissionLiveRouteTurnCandidateAdmissionResonanceIntentSchema ||
		!receiver.SourceAdmissionResonanceIntentPassed ||
		receiver.SourceAdmissionResonanceIntentID != receiver.AdmissionResonanceIntentID ||
		receiver.SourceAdmissionResonanceIntentAction != "draft_resonance_direction_intent_dry_run" ||
		!receiver.SourceAdmissionResonanceIntentReady ||
		receiver.SourceAdmissionResonanceIntentCausalID != receiver.AdmissionResonanceIntentCausalID {
		observation.Reason = "candidate_admission_resonance_receiver_source_intent_mismatch"
		return observation
	}
	if receiver.SourceAdmissionFinalGateIDForResonanceReceiver != receiver.AdmissionFinalGateID ||
		receiver.SourceAdmissionSealIDForResonanceReceiver != receiver.AdmissionSealID ||
		receiver.SourceAdmissionPermitIDForResonanceReceiver != receiver.AdmissionPermitID ||
		receiver.SourceAdmissionReadinessIDForResonanceReceiver != receiver.AdmissionReadinessID ||
		receiver.SourceLedgerVerificationIDForResonanceReceiver != receiver.LedgerVerificationID ||
		receiver.SourceLedgerPersistenceIDForResonanceReceiver != receiver.LedgerPersistenceID ||
		receiver.SourceLedgerImplementationIDForResonanceReceiver != receiver.LedgerImplementationID ||
		receiver.SourceAdmissionLedgerIDForResonanceReceiver != receiver.AdmissionLedgerID ||
		receiver.SourceRollbackImplementationIDForResonanceReceiver != receiver.RollbackImplementationID ||
		receiver.SourceWriterReceiptIDForResonanceReceiver != receiver.WriterReceiptID {
		observation.Reason = "candidate_admission_resonance_receiver_source_id_mismatch"
		return observation
	}
	if receiver.AdmissionResonanceReceiverID == "" ||
		receiver.AdmissionResonanceIntentID == "" ||
		receiver.AdmissionFinalGateID == "" ||
		receiver.AdmissionSealID == "" ||
		receiver.AdmissionPermitID == "" ||
		receiver.AdmissionReadinessID == "" ||
		receiver.LedgerVerificationID == "" ||
		receiver.LedgerPersistenceID == "" ||
		receiver.LedgerImplementationID == "" ||
		receiver.RollbackImplementationID == "" ||
		receiver.WriterReceiptID == "" ||
		receiver.WriterImplementationID == "" ||
		receiver.AdmissionLedgerID == "" ||
		receiver.AdmissionWriterContractID == "" ||
		receiver.AdmissionWriterInventoryID == "" ||
		receiver.AdmissionWriterPreflightID == "" ||
		receiver.AdmissionLiveStageID == "" ||
		receiver.AdmissionEnableGateID == "" ||
		receiver.AdmissionSwitchID == "" ||
		receiver.AdmissionPromotionID == "" ||
		receiver.AdmissionDecisionID == "" ||
		receiver.AdmissionAdapterID == "" ||
		receiver.CandidateRunID == "" ||
		receiver.CandidateDraftID == "" ||
		receiver.CandidateExecutionID == "" ||
		receiver.GeneratorAdapterID == "" ||
		receiver.HandoffID == "" ||
		receiver.DreamCandidateRunID == "" ||
		receiver.CandidateTextHash == "" ||
		receiver.TurnTextHash == "" {
		observation.Reason = "candidate_admission_resonance_receiver_missing_provenance"
		return observation
	}

	observation.AdmissionResonanceObservationState = "observation_recorded_dry_run"
	observation.AdmissionResonanceObservationAction = "record_resonance_receiver_observation_dry_run"
	observation.AdmissionResonanceObservationTarget = "resonance"
	observation.AdmissionResonanceObservationTargetKind = "internal_world_observation"
	observation.AdmissionResonanceObservationTargetMode = "append_only_read_back_dry_run"
	observation.AdmissionResonanceObservationReceiptShape = "resonance_receiver_state_proof_ledger"
	observation.AdmissionResonanceObservationDryRunOnly = true
	observation.AdmissionResonanceObservationReceiverVerified = true
	observation.AdmissionResonanceObservationIntentVerified = receiver.AdmissionResonanceReceiverIntentVerified
	observation.AdmissionResonanceObservationFinalGateVerified = receiver.AdmissionResonanceReceiverFinalGateVerified
	observation.AdmissionResonanceObservationSealVerified = receiver.AdmissionResonanceReceiverSealVerified
	observation.AdmissionResonanceObservationPermitVerified = receiver.AdmissionResonanceReceiverPermitVerified
	observation.AdmissionResonanceObservationReadinessVerified = receiver.AdmissionResonanceReceiverReadinessVerified
	observation.AdmissionResonanceObservationLedgerVerified = receiver.AdmissionResonanceReceiverLedgerVerified
	observation.AdmissionResonanceObservationWriterReady = receiver.AdmissionResonanceReceiverWriterReady
	observation.AdmissionResonanceObservationRollbackReady = receiver.AdmissionResonanceReceiverRollbackReady
	observation.AdmissionResonanceObservationLedgerReady = receiver.AdmissionResonanceReceiverLedgerReady
	observation.AdmissionResonanceObservationObserver = "resonance"
	observation.AdmissionResonanceObservationObserverKind = "internal_world"
	observation.AdmissionResonanceObservationKind = "receiver_state_proof"
	observation.AdmissionResonanceObservationMode = "sealed_metadata_observation"
	observation.AdmissionResonanceObservationAppendOnly = true
	observation.AdmissionResonanceObservationReadBack = true
	observation.AdmissionResonanceObservationReceiptVerified = true
	observation.AdmissionResonanceObservationRawDreamTextObserved = false
	observation.AdmissionResonanceObservationRawDreamTextForwarded = false
	observation.AdmissionResonanceObservationJanusSurfaceAllowed = false
	observation.AdmissionResonanceObservationCoocLearningAllowed = false
	observation.AdmissionResonanceObservationDeltaHarvestAllowed = false
	observation.AdmissionResonanceObservationBodyMutationAllowed = false
	observation.AdmissionResonanceObservationRollbackRequired = true
	observation.AdmissionResonanceObservationReady = true
	observation.LiveReady = true
	observation.BodyTarget = "none"
	observation.ContractsReady = false
	observation.WriteAllowed = false
	observation.AdmissionAllowed = false
	observation.LiveAdmissionEnabled = false
	observation.MutatesState = false
	observation.AdmissionResonanceObservationCausalID = admissionLiveRouteTurnCandidateAdmissionResonanceObservationCausalID(observation)
	if observation.AdmissionResonanceObservationCausalID == "" {
		observation.Reason = "missing_candidate_admission_resonance_observation_causal_id"
		return observation
	}
	observation.AdmissionResonanceObservationAppendHash = admissionLiveRouteTurnCandidateAdmissionResonanceObservationAppendHash(observation)
	observation.AdmissionResonanceObservationReadBackHash = admissionLiveRouteTurnCandidateAdmissionResonanceObservationReadBackHash(observation)
	if observation.AdmissionResonanceObservationAppendHash == "" ||
		observation.AdmissionResonanceObservationReadBackHash == "" ||
		observation.AdmissionResonanceObservationAppendHash == observation.AdmissionResonanceObservationReadBackHash {
		observation.Reason = "candidate_admission_resonance_observation_read_back_missing"
		return observation
	}
	observation.AdmissionResonanceObservationID = admissionLiveRouteTurnCandidateAdmissionResonanceObservationID(observation)
	if observation.AdmissionResonanceObservationID == "" {
		observation.Reason = "missing_candidate_admission_resonance_observation_id"
		return observation
	}
	observation.Passed = true
	observation.Reason = "resonance observation recorded and read back without body mutation"
	return observation
}

func admissionLiveRouteTurnCandidateAdmissionResonanceObservationCausalID(observation admissionLiveRouteTurnCandidateAdmissionResonanceObservation) string {
	h := hashJSON(struct {
		AdmissionResonanceReceiverID       string `json:"admission_resonance_receiver_id"`
		AdmissionResonanceReceiverCausalID string `json:"admission_resonance_receiver_causal_id"`
		CandidateRunID                     string `json:"candidate_run_id"`
		CandidateTextHash                  string `json:"candidate_text_hash"`
		TurnTextHash                       string `json:"turn_text_hash"`
		Observer                           string `json:"observer"`
		ObserverKind                       string `json:"observer_kind"`
	}{
		AdmissionResonanceReceiverID:       observation.AdmissionResonanceReceiverID,
		AdmissionResonanceReceiverCausalID: observation.AdmissionResonanceReceiverCausalID,
		CandidateRunID:                     observation.CandidateRunID,
		CandidateTextHash:                  observation.CandidateTextHash,
		TurnTextHash:                       observation.TurnTextHash,
		Observer:                           observation.AdmissionResonanceObservationObserver,
		ObserverKind:                       observation.AdmissionResonanceObservationObserverKind,
	})
	if h == "" {
		return ""
	}
	return "resonance-observation-causal-" + h
}

func admissionLiveRouteTurnCandidateAdmissionResonanceObservationAppendHash(observation admissionLiveRouteTurnCandidateAdmissionResonanceObservation) string {
	h := hashJSON(struct {
		AdmissionResonanceObservationCausalID string `json:"admission_resonance_observation_causal_id"`
		AdmissionResonanceReceiverID          string `json:"admission_resonance_receiver_id"`
		ReceiverPreStateHash                  string `json:"receiver_pre_state_hash"`
		ReceiverPostStateHash                 string `json:"receiver_post_state_hash"`
		ReceiverDeltaHash                     string `json:"receiver_delta_hash"`
		ObservationMode                       string `json:"observation_mode"`
		AppendOnly                            bool   `json:"append_only"`
		DryRunOnly                            bool   `json:"dry_run_only"`
	}{
		AdmissionResonanceObservationCausalID: observation.AdmissionResonanceObservationCausalID,
		AdmissionResonanceReceiverID:          observation.AdmissionResonanceReceiverID,
		ReceiverPreStateHash:                  observation.AdmissionResonanceReceiverPreStateHash,
		ReceiverPostStateHash:                 observation.AdmissionResonanceReceiverPostStateHash,
		ReceiverDeltaHash:                     observation.AdmissionResonanceReceiverStateDeltaHash,
		ObservationMode:                       observation.AdmissionResonanceObservationMode,
		AppendOnly:                            observation.AdmissionResonanceObservationAppendOnly,
		DryRunOnly:                            observation.AdmissionResonanceObservationDryRunOnly,
	})
	if h == "" {
		return ""
	}
	return "resonance-observation-append-" + h
}

func admissionLiveRouteTurnCandidateAdmissionResonanceObservationReadBackHash(observation admissionLiveRouteTurnCandidateAdmissionResonanceObservation) string {
	h := hashJSON(struct {
		AppendHash      string `json:"append_hash"`
		ReceiverID      string `json:"receiver_id"`
		ObservationKind string `json:"observation_kind"`
		ReadBack        bool   `json:"read_back"`
		ReceiptVerified bool   `json:"receipt_verified"`
		BodyMutation    bool   `json:"body_mutation"`
	}{
		AppendHash:      observation.AdmissionResonanceObservationAppendHash,
		ReceiverID:      observation.AdmissionResonanceReceiverID,
		ObservationKind: observation.AdmissionResonanceObservationKind,
		ReadBack:        observation.AdmissionResonanceObservationReadBack,
		ReceiptVerified: observation.AdmissionResonanceObservationReceiptVerified,
		BodyMutation:    observation.AdmissionResonanceObservationBodyMutationAllowed,
	})
	if h == "" {
		return ""
	}
	return "resonance-observation-read-" + h
}

func admissionLiveRouteTurnCandidateAdmissionResonanceObservationID(observation admissionLiveRouteTurnCandidateAdmissionResonanceObservation) string {
	h := hashJSON(struct {
		AdmissionResonanceReceiverID         string `json:"admission_resonance_receiver_id"`
		AdmissionFinalGateID                 string `json:"admission_final_gate_id"`
		AdmissionSealID                      string `json:"admission_seal_id"`
		AdmissionPermitID                    string `json:"admission_permit_id"`
		LedgerVerificationID                 string `json:"ledger_verification_id"`
		ObservationState                     string `json:"observation_state"`
		ObservationAction                    string `json:"observation_action"`
		ObservationTarget                    string `json:"observation_target"`
		ObservationTargetKind                string `json:"observation_target_kind"`
		ObservationTargetMode                string `json:"observation_target_mode"`
		ObservationReceiptShape              string `json:"observation_receipt_shape"`
		ObservationDryRunOnly                bool   `json:"observation_dry_run_only"`
		ObservationReceiverVerified          bool   `json:"observation_receiver_verified"`
		ObservationCausalID                  string `json:"observation_causal_id"`
		ObservationAppendHash                string `json:"observation_append_hash"`
		ObservationReadBackHash              string `json:"observation_read_back_hash"`
		AppendOnly                           bool   `json:"append_only"`
		ReadBack                             bool   `json:"read_back"`
		ReceiptVerified                      bool   `json:"receipt_verified"`
		RawTextObserved                      bool   `json:"raw_text_observed"`
		RawTextForwarded                     bool   `json:"raw_text_forwarded"`
		JanusSurfaceAllowed                  bool   `json:"janus_surface_allowed"`
		CoocLearningAllowed                  bool   `json:"cooc_learning_allowed"`
		DeltaHarvestAllowed                  bool   `json:"delta_harvest_allowed"`
		BodyMutationAllowed                  bool   `json:"body_mutation_allowed"`
		RollbackRequired                     bool   `json:"rollback_required"`
		ObservationReady                     bool   `json:"observation_ready"`
		SourceAdmissionResonanceReceiverID   string `json:"source_admission_resonance_receiver_id"`
		SourceAdmissionResonanceReceiverHash string `json:"source_admission_resonance_receiver_state_delta_hash"`
		ContractsReady                       bool   `json:"contracts_ready"`
		BodyTarget                           string `json:"body_target"`
		WriteAllowed                         bool   `json:"write_allowed"`
		AdmissionAllowed                     bool   `json:"admission_allowed"`
		LiveAdmissionEnabled                 bool   `json:"live_admission_enabled"`
		MutatesState                         bool   `json:"mutates_state"`
	}{
		AdmissionResonanceReceiverID:         observation.AdmissionResonanceReceiverID,
		AdmissionFinalGateID:                 observation.AdmissionFinalGateID,
		AdmissionSealID:                      observation.AdmissionSealID,
		AdmissionPermitID:                    observation.AdmissionPermitID,
		LedgerVerificationID:                 observation.LedgerVerificationID,
		ObservationState:                     observation.AdmissionResonanceObservationState,
		ObservationAction:                    observation.AdmissionResonanceObservationAction,
		ObservationTarget:                    observation.AdmissionResonanceObservationTarget,
		ObservationTargetKind:                observation.AdmissionResonanceObservationTargetKind,
		ObservationTargetMode:                observation.AdmissionResonanceObservationTargetMode,
		ObservationReceiptShape:              observation.AdmissionResonanceObservationReceiptShape,
		ObservationDryRunOnly:                observation.AdmissionResonanceObservationDryRunOnly,
		ObservationReceiverVerified:          observation.AdmissionResonanceObservationReceiverVerified,
		ObservationCausalID:                  observation.AdmissionResonanceObservationCausalID,
		ObservationAppendHash:                observation.AdmissionResonanceObservationAppendHash,
		ObservationReadBackHash:              observation.AdmissionResonanceObservationReadBackHash,
		AppendOnly:                           observation.AdmissionResonanceObservationAppendOnly,
		ReadBack:                             observation.AdmissionResonanceObservationReadBack,
		ReceiptVerified:                      observation.AdmissionResonanceObservationReceiptVerified,
		RawTextObserved:                      observation.AdmissionResonanceObservationRawDreamTextObserved,
		RawTextForwarded:                     observation.AdmissionResonanceObservationRawDreamTextForwarded,
		JanusSurfaceAllowed:                  observation.AdmissionResonanceObservationJanusSurfaceAllowed,
		CoocLearningAllowed:                  observation.AdmissionResonanceObservationCoocLearningAllowed,
		DeltaHarvestAllowed:                  observation.AdmissionResonanceObservationDeltaHarvestAllowed,
		BodyMutationAllowed:                  observation.AdmissionResonanceObservationBodyMutationAllowed,
		RollbackRequired:                     observation.AdmissionResonanceObservationRollbackRequired,
		ObservationReady:                     observation.AdmissionResonanceObservationReady,
		SourceAdmissionResonanceReceiverID:   observation.SourceAdmissionResonanceReceiverID,
		SourceAdmissionResonanceReceiverHash: observation.SourceAdmissionResonanceReceiverStateDeltaHash,
		ContractsReady:                       observation.ContractsReady,
		BodyTarget:                           observation.BodyTarget,
		WriteAllowed:                         observation.WriteAllowed,
		AdmissionAllowed:                     observation.AdmissionAllowed,
		LiveAdmissionEnabled:                 observation.LiveAdmissionEnabled,
		MutatesState:                         observation.MutatesState,
	})
	if h == "" {
		return ""
	}
	return "resonance-observation-" + h
}

func recordAdmissionLiveRouteTurnCandidateAdmissionResonanceObservation(observation admissionLiveRouteTurnCandidateAdmissionResonanceObservation) error {
	path := strings.TrimSpace(os.Getenv("AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_RESONANCE_OBSERVATION_LOG"))
	if path == "" {
		return nil
	}
	f, err := os.OpenFile(path, os.O_CREATE|os.O_APPEND|os.O_WRONLY, 0600)
	if err != nil {
		return err
	}
	enc := json.NewEncoder(f)
	err = enc.Encode(observation)
	if closeErr := f.Close(); err == nil {
		err = closeErr
	}
	return err
}

func admissionLiveRouteTurnCandidateAdmissionResonanceGraftBoundaryForObservation(observation admissionLiveRouteTurnCandidateAdmissionResonanceObservation) admissionLiveRouteTurnCandidateAdmissionResonanceGraftBoundary {
	sourceSchema := observation.Schema
	boundary := admissionLiveRouteTurnCandidateAdmissionResonanceGraftBoundary{
		admissionLiveRouteTurnCandidateAdmissionResonanceObservation: observation,
		AdmissionResonanceGraftBoundaryState:                         "blocked",
		AdmissionResonanceGraftBoundaryAction:                        "reject",
		AdmissionResonanceGraftBoundaryDryRunOnly:                    true,
		SourceAdmissionResonanceObservationSchema:                    sourceSchema,
		SourceAdmissionResonanceObservationPassed:                    observation.Passed,
		SourceAdmissionResonanceObservationID:                        observation.AdmissionResonanceObservationID,
		SourceAdmissionResonanceObservationAction:                    observation.AdmissionResonanceObservationAction,
		SourceAdmissionResonanceObservationReady:                     observation.AdmissionResonanceObservationReady,
		SourceAdmissionResonanceObservationCausalID:                  observation.AdmissionResonanceObservationCausalID,
		SourceAdmissionResonanceObservationAppendHash:                observation.AdmissionResonanceObservationAppendHash,
		SourceAdmissionResonanceObservationReadBackHash:              observation.AdmissionResonanceObservationReadBackHash,
		SourceAdmissionResonanceReceiverIDForGraftBoundary:           observation.AdmissionResonanceReceiverID,
		SourceAdmissionResonanceIntentIDForGraftBoundary:             observation.AdmissionResonanceIntentID,
		SourceAdmissionFinalGateIDForGraftBoundary:                   observation.AdmissionFinalGateID,
		SourceAdmissionSealIDForGraftBoundary:                        observation.AdmissionSealID,
		SourceAdmissionPermitIDForGraftBoundary:                      observation.AdmissionPermitID,
		SourceAdmissionReadinessIDForGraftBoundary:                   observation.AdmissionReadinessID,
		SourceLedgerVerificationIDForGraftBoundary:                   observation.LedgerVerificationID,
		SourceLedgerPersistenceIDForGraftBoundary:                    observation.LedgerPersistenceID,
		SourceLedgerImplementationIDForGraftBoundary:                 observation.LedgerImplementationID,
		SourceAdmissionLedgerIDForGraftBoundary:                      observation.AdmissionLedgerID,
		SourceRollbackImplementationIDForGraftBoundary:               observation.RollbackImplementationID,
		SourceWriterReceiptIDForGraftBoundary:                        observation.WriterReceiptID,
	}
	boundary.Schema = admissionLiveRouteTurnCandidateAdmissionResonanceGraftBoundarySchema
	boundary.Timing = "live_admission_resonance_graft_boundary"
	boundary.Passed = false
	boundary.AdmissionResonanceGraftBoundaryID = ""
	boundary.AdmissionResonanceGraftBoundaryReady = false
	boundary.LiveReady = false
	boundary.BodyTarget = "none"
	boundary.ContractsReady = false
	boundary.WriteAllowed = false
	boundary.AdmissionAllowed = false
	boundary.LiveAdmissionEnabled = false
	boundary.MutatesState = false

	if sourceSchema == "" {
		boundary.Reason = "missing_candidate_admission_resonance_observation"
		return boundary
	}
	if sourceSchema != admissionLiveRouteTurnCandidateAdmissionResonanceObservationSchema {
		boundary.Reason = "unexpected_candidate_admission_resonance_observation_schema " + sourceSchema
		return boundary
	}
	if !observation.Passed {
		boundary.Reason = "candidate_admission_resonance_observation_failed"
		if observation.Reason != "" {
			boundary.Reason += ": " + observation.Reason
		}
		return boundary
	}
	if observation.AdmissionResonanceObservationID == "" {
		boundary.Reason = "missing_candidate_admission_resonance_observation_id"
		return boundary
	}
	if wantObservationID := admissionLiveRouteTurnCandidateAdmissionResonanceObservationID(observation); wantObservationID == "" || observation.AdmissionResonanceObservationID != wantObservationID {
		boundary.Reason = "candidate_admission_resonance_observation_id_mismatch"
		return boundary
	}
	if observation.AdmissionResonanceObservationState != "observation_recorded_dry_run" ||
		observation.AdmissionResonanceObservationAction != "record_resonance_receiver_observation_dry_run" ||
		observation.AdmissionResonanceObservationTarget != "resonance" ||
		observation.AdmissionResonanceObservationTargetKind != "internal_world_observation" ||
		observation.AdmissionResonanceObservationTargetMode != "append_only_read_back_dry_run" ||
		observation.AdmissionResonanceObservationReceiptShape != "resonance_receiver_state_proof_ledger" {
		boundary.Reason = "candidate_admission_resonance_observation_shape_mismatch"
		return boundary
	}
	if !observation.AdmissionResonanceObservationDryRunOnly ||
		!observation.AdmissionResonanceObservationReceiverVerified ||
		!observation.AdmissionResonanceObservationIntentVerified ||
		!observation.AdmissionResonanceObservationFinalGateVerified ||
		!observation.AdmissionResonanceObservationSealVerified ||
		!observation.AdmissionResonanceObservationPermitVerified ||
		!observation.AdmissionResonanceObservationReadinessVerified ||
		!observation.AdmissionResonanceObservationLedgerVerified ||
		!observation.AdmissionResonanceObservationWriterReady ||
		!observation.AdmissionResonanceObservationRollbackReady ||
		!observation.AdmissionResonanceObservationLedgerReady ||
		!observation.AdmissionResonanceObservationReady {
		boundary.Reason = "candidate_admission_resonance_observation_not_verified_dry_run"
		return boundary
	}
	if observation.AdmissionResonanceObservationObserver != "resonance" ||
		observation.AdmissionResonanceObservationObserverKind != "internal_world" ||
		observation.AdmissionResonanceObservationKind != "receiver_state_proof" ||
		observation.AdmissionResonanceObservationMode != "sealed_metadata_observation" ||
		observation.AdmissionResonanceObservationCausalID == "" ||
		observation.AdmissionResonanceObservationAppendHash == "" ||
		observation.AdmissionResonanceObservationReadBackHash == "" {
		boundary.Reason = "candidate_admission_resonance_observation_target_mismatch"
		return boundary
	}
	if wantCausalID := admissionLiveRouteTurnCandidateAdmissionResonanceObservationCausalID(observation); wantCausalID == "" || observation.AdmissionResonanceObservationCausalID != wantCausalID {
		boundary.Reason = "candidate_admission_resonance_observation_causal_id_mismatch"
		return boundary
	}
	if wantAppendHash := admissionLiveRouteTurnCandidateAdmissionResonanceObservationAppendHash(observation); wantAppendHash == "" || observation.AdmissionResonanceObservationAppendHash != wantAppendHash {
		boundary.Reason = "candidate_admission_resonance_observation_append_hash_mismatch"
		return boundary
	}
	if wantReadBackHash := admissionLiveRouteTurnCandidateAdmissionResonanceObservationReadBackHash(observation); wantReadBackHash == "" || observation.AdmissionResonanceObservationReadBackHash != wantReadBackHash {
		boundary.Reason = "candidate_admission_resonance_observation_read_back_hash_mismatch"
		return boundary
	}
	if !observation.AdmissionResonanceObservationAppendOnly ||
		!observation.AdmissionResonanceObservationReadBack ||
		!observation.AdmissionResonanceObservationReceiptVerified ||
		observation.AdmissionResonanceObservationRawDreamTextObserved ||
		observation.AdmissionResonanceObservationRawDreamTextForwarded ||
		observation.AdmissionResonanceObservationJanusSurfaceAllowed ||
		observation.AdmissionResonanceObservationCoocLearningAllowed ||
		observation.AdmissionResonanceObservationDeltaHarvestAllowed ||
		observation.AdmissionResonanceObservationBodyMutationAllowed ||
		!observation.AdmissionResonanceObservationRollbackRequired {
		boundary.Reason = "candidate_admission_resonance_observation_guard_mismatch"
		return boundary
	}
	if observation.ContractsReady || observation.WriteAllowed || observation.MutatesState || observation.LiveAdmissionEnabled || observation.AdmissionAllowed {
		boundary.Reason = "candidate_admission_resonance_observation_already_open"
		return boundary
	}
	if observation.BodyTarget != "none" {
		boundary.Reason = "candidate_admission_resonance_observation_body_target_mismatch"
		return boundary
	}
	if !observation.LiveReady {
		boundary.Reason = "candidate_admission_resonance_observation_not_live_ready"
		return boundary
	}
	if observation.SourceAdmissionResonanceReceiverSchema != admissionLiveRouteTurnCandidateAdmissionResonanceReceiverSchema ||
		!observation.SourceAdmissionResonanceReceiverPassed ||
		observation.SourceAdmissionResonanceReceiverID != observation.AdmissionResonanceReceiverID ||
		observation.SourceAdmissionResonanceReceiverAction != "preview_resonance_receive_dry_run" ||
		!observation.SourceAdmissionResonanceReceiverReady ||
		observation.SourceAdmissionResonanceReceiverCausalID != observation.AdmissionResonanceReceiverCausalID ||
		observation.SourceAdmissionResonanceReceiverStateDeltaHash != observation.AdmissionResonanceReceiverStateDeltaHash {
		boundary.Reason = "candidate_admission_resonance_observation_source_receiver_mismatch"
		return boundary
	}
	if observation.SourceAdmissionResonanceIntentIDForObservation != observation.AdmissionResonanceIntentID ||
		observation.SourceAdmissionFinalGateIDForResonanceObservation != observation.AdmissionFinalGateID ||
		observation.SourceAdmissionSealIDForResonanceObservation != observation.AdmissionSealID ||
		observation.SourceAdmissionPermitIDForResonanceObservation != observation.AdmissionPermitID ||
		observation.SourceAdmissionReadinessIDForResonanceObservation != observation.AdmissionReadinessID ||
		observation.SourceLedgerVerificationIDForResonanceObservation != observation.LedgerVerificationID ||
		observation.SourceLedgerPersistenceIDForResonanceObservation != observation.LedgerPersistenceID ||
		observation.SourceLedgerImplementationIDForResonanceObservation != observation.LedgerImplementationID ||
		observation.SourceAdmissionLedgerIDForResonanceObservation != observation.AdmissionLedgerID ||
		observation.SourceRollbackImplementationIDForResonanceObservation != observation.RollbackImplementationID ||
		observation.SourceWriterReceiptIDForResonanceObservation != observation.WriterReceiptID {
		boundary.Reason = "candidate_admission_resonance_observation_source_id_mismatch"
		return boundary
	}
	if observation.AdmissionResonanceObservationID == "" ||
		observation.AdmissionResonanceReceiverID == "" ||
		observation.AdmissionResonanceIntentID == "" ||
		observation.AdmissionFinalGateID == "" ||
		observation.AdmissionSealID == "" ||
		observation.AdmissionPermitID == "" ||
		observation.AdmissionReadinessID == "" ||
		observation.LedgerVerificationID == "" ||
		observation.LedgerPersistenceID == "" ||
		observation.LedgerImplementationID == "" ||
		observation.RollbackImplementationID == "" ||
		observation.WriterReceiptID == "" ||
		observation.WriterImplementationID == "" ||
		observation.AdmissionLedgerID == "" ||
		observation.AdmissionWriterContractID == "" ||
		observation.AdmissionWriterInventoryID == "" ||
		observation.AdmissionWriterPreflightID == "" ||
		observation.AdmissionLiveStageID == "" ||
		observation.AdmissionEnableGateID == "" ||
		observation.AdmissionSwitchID == "" ||
		observation.AdmissionPromotionID == "" ||
		observation.AdmissionDecisionID == "" ||
		observation.AdmissionAdapterID == "" ||
		observation.CandidateRunID == "" ||
		observation.CandidateDraftID == "" ||
		observation.CandidateExecutionID == "" ||
		observation.GeneratorAdapterID == "" ||
		observation.HandoffID == "" ||
		observation.DreamCandidateRunID == "" ||
		observation.CandidateTextHash == "" ||
		observation.TurnTextHash == "" {
		boundary.Reason = "candidate_admission_resonance_observation_missing_provenance"
		return boundary
	}

	boundary.AdmissionResonanceGraftBoundaryState = "shadow_graft_boundary_declared_dry_run"
	boundary.AdmissionResonanceGraftBoundaryAction = "declare_resonance_shadow_graft_boundary_dry_run"
	boundary.AdmissionResonanceGraftBoundaryTarget = "resonance"
	boundary.AdmissionResonanceGraftBoundaryTargetKind = "internal_world_shadow_graft"
	boundary.AdmissionResonanceGraftBoundaryTargetMode = "receipt_only_closed_dry_run"
	boundary.AdmissionResonanceGraftBoundaryReceiptShape = "resonance_observation_shadow_graft_boundary"
	boundary.AdmissionResonanceGraftBoundaryDryRunOnly = true
	boundary.AdmissionResonanceGraftBoundaryObservationVerified = true
	boundary.AdmissionResonanceGraftBoundaryReceiverVerified = observation.AdmissionResonanceObservationReceiverVerified
	boundary.AdmissionResonanceGraftBoundaryIntentVerified = observation.AdmissionResonanceObservationIntentVerified
	boundary.AdmissionResonanceGraftBoundaryFinalGateVerified = observation.AdmissionResonanceObservationFinalGateVerified
	boundary.AdmissionResonanceGraftBoundarySealVerified = observation.AdmissionResonanceObservationSealVerified
	boundary.AdmissionResonanceGraftBoundaryPermitVerified = observation.AdmissionResonanceObservationPermitVerified
	boundary.AdmissionResonanceGraftBoundaryReadinessVerified = observation.AdmissionResonanceObservationReadinessVerified
	boundary.AdmissionResonanceGraftBoundaryLedgerVerified = observation.AdmissionResonanceObservationLedgerVerified
	boundary.AdmissionResonanceGraftBoundaryWriterReady = observation.AdmissionResonanceObservationWriterReady
	boundary.AdmissionResonanceGraftBoundaryRollbackReady = observation.AdmissionResonanceObservationRollbackReady
	boundary.AdmissionResonanceGraftBoundaryLedgerReady = observation.AdmissionResonanceObservationLedgerReady
	boundary.AdmissionResonanceGraftBoundaryKind = "shadow_graft_boundary"
	boundary.AdmissionResonanceGraftBoundaryMode = "no_mutation_receipt"
	boundary.AdmissionResonanceGraftBoundaryStage = "pre_live_graft"
	boundary.AdmissionResonanceGraftBoundaryShadowOnly = true
	boundary.AdmissionResonanceGraftBoundaryGraftAllowed = false
	boundary.AdmissionResonanceGraftBoundaryRawDreamTextAllowed = false
	boundary.AdmissionResonanceGraftBoundaryJanusSurfaceAllowed = false
	boundary.AdmissionResonanceGraftBoundaryCoocLearningAllowed = false
	boundary.AdmissionResonanceGraftBoundaryDeltaHarvestAllowed = false
	boundary.AdmissionResonanceGraftBoundaryBodyMutationAllowed = false
	boundary.AdmissionResonanceGraftBoundaryRollbackRequired = true
	boundary.AdmissionResonanceGraftBoundaryReady = true
	boundary.LiveReady = true
	boundary.BodyTarget = "none"
	boundary.ContractsReady = false
	boundary.WriteAllowed = false
	boundary.AdmissionAllowed = false
	boundary.LiveAdmissionEnabled = false
	boundary.MutatesState = false
	boundary.AdmissionResonanceGraftBoundaryCausalID = admissionLiveRouteTurnCandidateAdmissionResonanceGraftBoundaryCausalID(boundary)
	if boundary.AdmissionResonanceGraftBoundaryCausalID == "" {
		boundary.Reason = "missing_candidate_admission_resonance_graft_boundary_causal_id"
		return boundary
	}
	boundary.AdmissionResonanceGraftBoundaryHash = admissionLiveRouteTurnCandidateAdmissionResonanceGraftBoundaryHash(boundary)
	boundary.AdmissionResonanceGraftBoundaryReadBackHash = admissionLiveRouteTurnCandidateAdmissionResonanceGraftBoundaryReadBackHash(boundary)
	if boundary.AdmissionResonanceGraftBoundaryHash == "" ||
		boundary.AdmissionResonanceGraftBoundaryReadBackHash == "" ||
		boundary.AdmissionResonanceGraftBoundaryHash == boundary.AdmissionResonanceGraftBoundaryReadBackHash {
		boundary.Reason = "candidate_admission_resonance_graft_boundary_read_back_missing"
		return boundary
	}
	boundary.AdmissionResonanceGraftBoundaryID = admissionLiveRouteTurnCandidateAdmissionResonanceGraftBoundaryID(boundary)
	if boundary.AdmissionResonanceGraftBoundaryID == "" {
		boundary.Reason = "missing_candidate_admission_resonance_graft_boundary_id"
		return boundary
	}
	boundary.Passed = true
	boundary.Reason = "resonance shadow graft boundary declared without body mutation"
	return boundary
}

func admissionLiveRouteTurnCandidateAdmissionResonanceGraftBoundaryCausalID(boundary admissionLiveRouteTurnCandidateAdmissionResonanceGraftBoundary) string {
	h := hashJSON(struct {
		AdmissionResonanceObservationID           string `json:"admission_resonance_observation_id"`
		AdmissionResonanceObservationReadBackHash string `json:"admission_resonance_observation_read_back_hash"`
		AdmissionResonanceReceiverID              string `json:"admission_resonance_receiver_id"`
		CandidateRunID                            string `json:"candidate_run_id"`
		CandidateTextHash                         string `json:"candidate_text_hash"`
		TurnTextHash                              string `json:"turn_text_hash"`
		BoundaryKind                              string `json:"boundary_kind"`
		BoundaryStage                             string `json:"boundary_stage"`
	}{
		AdmissionResonanceObservationID:           boundary.AdmissionResonanceObservationID,
		AdmissionResonanceObservationReadBackHash: boundary.AdmissionResonanceObservationReadBackHash,
		AdmissionResonanceReceiverID:              boundary.AdmissionResonanceReceiverID,
		CandidateRunID:                            boundary.CandidateRunID,
		CandidateTextHash:                         boundary.CandidateTextHash,
		TurnTextHash:                              boundary.TurnTextHash,
		BoundaryKind:                              boundary.AdmissionResonanceGraftBoundaryKind,
		BoundaryStage:                             boundary.AdmissionResonanceGraftBoundaryStage,
	})
	if h == "" {
		return ""
	}
	return "resonance-graft-boundary-causal-" + h
}

func admissionLiveRouteTurnCandidateAdmissionResonanceGraftBoundaryHash(boundary admissionLiveRouteTurnCandidateAdmissionResonanceGraftBoundary) string {
	h := hashJSON(struct {
		AdmissionResonanceGraftBoundaryCausalID string `json:"admission_resonance_graft_boundary_causal_id"`
		AdmissionResonanceObservationID         string `json:"admission_resonance_observation_id"`
		ObservationAppendHash                   string `json:"observation_append_hash"`
		ObservationReadBackHash                 string `json:"observation_read_back_hash"`
		BoundaryMode                            string `json:"boundary_mode"`
		ShadowOnly                              bool   `json:"shadow_only"`
		DryRunOnly                              bool   `json:"dry_run_only"`
		GraftAllowed                            bool   `json:"graft_allowed"`
	}{
		AdmissionResonanceGraftBoundaryCausalID: boundary.AdmissionResonanceGraftBoundaryCausalID,
		AdmissionResonanceObservationID:         boundary.AdmissionResonanceObservationID,
		ObservationAppendHash:                   boundary.AdmissionResonanceObservationAppendHash,
		ObservationReadBackHash:                 boundary.AdmissionResonanceObservationReadBackHash,
		BoundaryMode:                            boundary.AdmissionResonanceGraftBoundaryMode,
		ShadowOnly:                              boundary.AdmissionResonanceGraftBoundaryShadowOnly,
		DryRunOnly:                              boundary.AdmissionResonanceGraftBoundaryDryRunOnly,
		GraftAllowed:                            boundary.AdmissionResonanceGraftBoundaryGraftAllowed,
	})
	if h == "" {
		return ""
	}
	return "resonance-graft-boundary-" + h
}

func admissionLiveRouteTurnCandidateAdmissionResonanceGraftBoundaryReadBackHash(boundary admissionLiveRouteTurnCandidateAdmissionResonanceGraftBoundary) string {
	h := hashJSON(struct {
		BoundaryHash    string `json:"boundary_hash"`
		ObservationID   string `json:"observation_id"`
		BoundaryKind    string `json:"boundary_kind"`
		BoundaryReady   bool   `json:"boundary_ready"`
		BodyMutation    bool   `json:"body_mutation"`
		AdmissionOpened bool   `json:"admission_opened"`
	}{
		BoundaryHash:    boundary.AdmissionResonanceGraftBoundaryHash,
		ObservationID:   boundary.AdmissionResonanceObservationID,
		BoundaryKind:    boundary.AdmissionResonanceGraftBoundaryKind,
		BoundaryReady:   boundary.AdmissionResonanceGraftBoundaryReady,
		BodyMutation:    boundary.AdmissionResonanceGraftBoundaryBodyMutationAllowed,
		AdmissionOpened: boundary.LiveAdmissionEnabled,
	})
	if h == "" {
		return ""
	}
	return "resonance-graft-boundary-read-" + h
}

func admissionLiveRouteTurnCandidateAdmissionResonanceGraftBoundaryID(boundary admissionLiveRouteTurnCandidateAdmissionResonanceGraftBoundary) string {
	h := hashJSON(struct {
		AdmissionResonanceGraftBoundaryCausalID     string `json:"admission_resonance_graft_boundary_causal_id"`
		AdmissionResonanceGraftBoundaryHash         string `json:"admission_resonance_graft_boundary_hash"`
		AdmissionResonanceGraftBoundaryReadBackHash string `json:"admission_resonance_graft_boundary_read_back_hash"`
		AdmissionResonanceObservationID             string `json:"admission_resonance_observation_id"`
		AdmissionResonanceReceiverID                string `json:"admission_resonance_receiver_id"`
		AdmissionFinalGateID                        string `json:"admission_final_gate_id"`
		LedgerVerificationID                        string `json:"ledger_verification_id"`
		BoundaryState                               string `json:"boundary_state"`
		BoundaryAction                              string `json:"boundary_action"`
		BoundaryTarget                              string `json:"boundary_target"`
		BoundaryTargetKind                          string `json:"boundary_target_kind"`
		BoundaryTargetMode                          string `json:"boundary_target_mode"`
		BoundaryReceiptShape                        string `json:"boundary_receipt_shape"`
		BoundaryDryRunOnly                          bool   `json:"boundary_dry_run_only"`
		ObservationVerified                         bool   `json:"observation_verified"`
		ShadowOnly                                  bool   `json:"shadow_only"`
		GraftAllowed                                bool   `json:"graft_allowed"`
		RawTextAllowed                              bool   `json:"raw_text_allowed"`
		JanusSurfaceAllowed                         bool   `json:"janus_surface_allowed"`
		CoocLearningAllowed                         bool   `json:"cooc_learning_allowed"`
		DeltaHarvestAllowed                         bool   `json:"delta_harvest_allowed"`
		BodyMutationAllowed                         bool   `json:"body_mutation_allowed"`
		RollbackRequired                            bool   `json:"rollback_required"`
		BoundaryReady                               bool   `json:"boundary_ready"`
		ContractsReady                              bool   `json:"contracts_ready"`
		BodyTarget                                  string `json:"body_target"`
		WriteAllowed                                bool   `json:"write_allowed"`
		AdmissionAllowed                            bool   `json:"admission_allowed"`
		LiveAdmissionEnabled                        bool   `json:"live_admission_enabled"`
		MutatesState                                bool   `json:"mutates_state"`
	}{
		AdmissionResonanceGraftBoundaryCausalID:     boundary.AdmissionResonanceGraftBoundaryCausalID,
		AdmissionResonanceGraftBoundaryHash:         boundary.AdmissionResonanceGraftBoundaryHash,
		AdmissionResonanceGraftBoundaryReadBackHash: boundary.AdmissionResonanceGraftBoundaryReadBackHash,
		AdmissionResonanceObservationID:             boundary.AdmissionResonanceObservationID,
		AdmissionResonanceReceiverID:                boundary.AdmissionResonanceReceiverID,
		AdmissionFinalGateID:                        boundary.AdmissionFinalGateID,
		LedgerVerificationID:                        boundary.LedgerVerificationID,
		BoundaryState:                               boundary.AdmissionResonanceGraftBoundaryState,
		BoundaryAction:                              boundary.AdmissionResonanceGraftBoundaryAction,
		BoundaryTarget:                              boundary.AdmissionResonanceGraftBoundaryTarget,
		BoundaryTargetKind:                          boundary.AdmissionResonanceGraftBoundaryTargetKind,
		BoundaryTargetMode:                          boundary.AdmissionResonanceGraftBoundaryTargetMode,
		BoundaryReceiptShape:                        boundary.AdmissionResonanceGraftBoundaryReceiptShape,
		BoundaryDryRunOnly:                          boundary.AdmissionResonanceGraftBoundaryDryRunOnly,
		ObservationVerified:                         boundary.AdmissionResonanceGraftBoundaryObservationVerified,
		ShadowOnly:                                  boundary.AdmissionResonanceGraftBoundaryShadowOnly,
		GraftAllowed:                                boundary.AdmissionResonanceGraftBoundaryGraftAllowed,
		RawTextAllowed:                              boundary.AdmissionResonanceGraftBoundaryRawDreamTextAllowed,
		JanusSurfaceAllowed:                         boundary.AdmissionResonanceGraftBoundaryJanusSurfaceAllowed,
		CoocLearningAllowed:                         boundary.AdmissionResonanceGraftBoundaryCoocLearningAllowed,
		DeltaHarvestAllowed:                         boundary.AdmissionResonanceGraftBoundaryDeltaHarvestAllowed,
		BodyMutationAllowed:                         boundary.AdmissionResonanceGraftBoundaryBodyMutationAllowed,
		RollbackRequired:                            boundary.AdmissionResonanceGraftBoundaryRollbackRequired,
		BoundaryReady:                               boundary.AdmissionResonanceGraftBoundaryReady,
		ContractsReady:                              boundary.ContractsReady,
		BodyTarget:                                  boundary.BodyTarget,
		WriteAllowed:                                boundary.WriteAllowed,
		AdmissionAllowed:                            boundary.AdmissionAllowed,
		LiveAdmissionEnabled:                        boundary.LiveAdmissionEnabled,
		MutatesState:                                boundary.MutatesState,
	})
	if h == "" {
		return ""
	}
	return "resonance-graft-boundary-id-" + h
}

func recordAdmissionLiveRouteTurnCandidateAdmissionResonanceGraftBoundary(boundary admissionLiveRouteTurnCandidateAdmissionResonanceGraftBoundary) error {
	path := strings.TrimSpace(os.Getenv("AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_RESONANCE_GRAFT_BOUNDARY_LOG"))
	if path == "" {
		return nil
	}
	f, err := os.OpenFile(path, os.O_CREATE|os.O_APPEND|os.O_WRONLY, 0600)
	if err != nil {
		return err
	}
	enc := json.NewEncoder(f)
	err = enc.Encode(boundary)
	if closeErr := f.Close(); err == nil {
		err = closeErr
	}
	return err
}

func admissionLiveRouteTurnCandidateAdmissionResonanceGraftPreflightForBoundary(boundary admissionLiveRouteTurnCandidateAdmissionResonanceGraftBoundary) admissionLiveRouteTurnCandidateAdmissionResonanceGraftPreflight {
	sourceSchema := boundary.Schema
	preflight := admissionLiveRouteTurnCandidateAdmissionResonanceGraftPreflight{
		admissionLiveRouteTurnCandidateAdmissionResonanceGraftBoundary: boundary,
		AdmissionResonanceGraftPreflightState:                          "blocked",
		AdmissionResonanceGraftPreflightAction:                         "reject",
		AdmissionResonanceGraftPreflightDryRunOnly:                     true,
		SourceAdmissionResonanceGraftBoundarySchema:                    sourceSchema,
		SourceAdmissionResonanceGraftBoundaryPassed:                    boundary.Passed,
		SourceAdmissionResonanceGraftBoundaryID:                        boundary.AdmissionResonanceGraftBoundaryID,
		SourceAdmissionResonanceGraftBoundaryAction:                    boundary.AdmissionResonanceGraftBoundaryAction,
		SourceAdmissionResonanceGraftBoundaryReady:                     boundary.AdmissionResonanceGraftBoundaryReady,
		SourceAdmissionResonanceGraftBoundaryCausalID:                  boundary.AdmissionResonanceGraftBoundaryCausalID,
		SourceAdmissionResonanceGraftBoundaryHash:                      boundary.AdmissionResonanceGraftBoundaryHash,
		SourceAdmissionResonanceGraftBoundaryReadBackHash:              boundary.AdmissionResonanceGraftBoundaryReadBackHash,
		SourceAdmissionResonanceObservationIDForGraftPreflight:         boundary.AdmissionResonanceObservationID,
		SourceAdmissionResonanceReceiverIDForGraftPreflight:            boundary.AdmissionResonanceReceiverID,
		SourceAdmissionResonanceIntentIDForGraftPreflight:              boundary.AdmissionResonanceIntentID,
		SourceAdmissionFinalGateIDForGraftPreflight:                    boundary.AdmissionFinalGateID,
		SourceAdmissionSealIDForGraftPreflight:                         boundary.AdmissionSealID,
		SourceAdmissionPermitIDForGraftPreflight:                       boundary.AdmissionPermitID,
		SourceAdmissionReadinessIDForGraftPreflight:                    boundary.AdmissionReadinessID,
		SourceLedgerVerificationIDForGraftPreflight:                    boundary.LedgerVerificationID,
		SourceLedgerPersistenceIDForGraftPreflight:                     boundary.LedgerPersistenceID,
		SourceLedgerImplementationIDForGraftPreflight:                  boundary.LedgerImplementationID,
		SourceAdmissionLedgerIDForGraftPreflight:                       boundary.AdmissionLedgerID,
		SourceRollbackImplementationIDForGraftPreflight:                boundary.RollbackImplementationID,
		SourceWriterReceiptIDForGraftPreflight:                         boundary.WriterReceiptID,
	}
	preflight.Schema = admissionLiveRouteTurnCandidateAdmissionResonanceGraftPreflightSchema
	preflight.Timing = "live_admission_resonance_graft_preflight"
	preflight.Passed = false
	preflight.AdmissionResonanceGraftPreflightID = ""
	preflight.AdmissionResonanceGraftPreflightReady = false
	preflight.LiveReady = false
	preflight.BodyTarget = "none"
	preflight.ContractsReady = false
	preflight.WriteAllowed = false
	preflight.AdmissionAllowed = false
	preflight.LiveAdmissionEnabled = false
	preflight.MutatesState = false

	if sourceSchema == "" {
		preflight.Reason = "missing_candidate_admission_resonance_graft_boundary"
		return preflight
	}
	if sourceSchema != admissionLiveRouteTurnCandidateAdmissionResonanceGraftBoundarySchema {
		preflight.Reason = "unexpected_candidate_admission_resonance_graft_boundary_schema " + sourceSchema
		return preflight
	}
	if !boundary.Passed {
		preflight.Reason = "candidate_admission_resonance_graft_boundary_failed"
		if boundary.Reason != "" {
			preflight.Reason += ": " + boundary.Reason
		}
		return preflight
	}
	if boundary.AdmissionResonanceGraftBoundaryID == "" {
		preflight.Reason = "missing_candidate_admission_resonance_graft_boundary_id"
		return preflight
	}
	if wantBoundaryID := admissionLiveRouteTurnCandidateAdmissionResonanceGraftBoundaryID(boundary); wantBoundaryID == "" || boundary.AdmissionResonanceGraftBoundaryID != wantBoundaryID {
		preflight.Reason = "candidate_admission_resonance_graft_boundary_id_mismatch"
		return preflight
	}
	if boundary.AdmissionResonanceGraftBoundaryState != "shadow_graft_boundary_declared_dry_run" ||
		boundary.AdmissionResonanceGraftBoundaryAction != "declare_resonance_shadow_graft_boundary_dry_run" ||
		boundary.AdmissionResonanceGraftBoundaryTarget != "resonance" ||
		boundary.AdmissionResonanceGraftBoundaryTargetKind != "internal_world_shadow_graft" ||
		boundary.AdmissionResonanceGraftBoundaryTargetMode != "receipt_only_closed_dry_run" ||
		boundary.AdmissionResonanceGraftBoundaryReceiptShape != "resonance_observation_shadow_graft_boundary" {
		preflight.Reason = "candidate_admission_resonance_graft_boundary_shape_mismatch"
		return preflight
	}
	if !boundary.AdmissionResonanceGraftBoundaryDryRunOnly ||
		!boundary.AdmissionResonanceGraftBoundaryObservationVerified ||
		!boundary.AdmissionResonanceGraftBoundaryReceiverVerified ||
		!boundary.AdmissionResonanceGraftBoundaryIntentVerified ||
		!boundary.AdmissionResonanceGraftBoundaryFinalGateVerified ||
		!boundary.AdmissionResonanceGraftBoundarySealVerified ||
		!boundary.AdmissionResonanceGraftBoundaryPermitVerified ||
		!boundary.AdmissionResonanceGraftBoundaryReadinessVerified ||
		!boundary.AdmissionResonanceGraftBoundaryLedgerVerified ||
		!boundary.AdmissionResonanceGraftBoundaryWriterReady ||
		!boundary.AdmissionResonanceGraftBoundaryRollbackReady ||
		!boundary.AdmissionResonanceGraftBoundaryLedgerReady ||
		!boundary.AdmissionResonanceGraftBoundaryReady {
		preflight.Reason = "candidate_admission_resonance_graft_boundary_not_verified_dry_run"
		return preflight
	}
	if boundary.AdmissionResonanceGraftBoundaryKind != "shadow_graft_boundary" ||
		boundary.AdmissionResonanceGraftBoundaryMode != "no_mutation_receipt" ||
		boundary.AdmissionResonanceGraftBoundaryStage != "pre_live_graft" ||
		boundary.AdmissionResonanceGraftBoundaryCausalID == "" ||
		boundary.AdmissionResonanceGraftBoundaryHash == "" ||
		boundary.AdmissionResonanceGraftBoundaryReadBackHash == "" {
		preflight.Reason = "candidate_admission_resonance_graft_boundary_target_mismatch"
		return preflight
	}
	if wantCausalID := admissionLiveRouteTurnCandidateAdmissionResonanceGraftBoundaryCausalID(boundary); wantCausalID == "" || boundary.AdmissionResonanceGraftBoundaryCausalID != wantCausalID {
		preflight.Reason = "candidate_admission_resonance_graft_boundary_causal_id_mismatch"
		return preflight
	}
	if wantHash := admissionLiveRouteTurnCandidateAdmissionResonanceGraftBoundaryHash(boundary); wantHash == "" || boundary.AdmissionResonanceGraftBoundaryHash != wantHash {
		preflight.Reason = "candidate_admission_resonance_graft_boundary_hash_mismatch"
		return preflight
	}
	if wantReadBackHash := admissionLiveRouteTurnCandidateAdmissionResonanceGraftBoundaryReadBackHash(boundary); wantReadBackHash == "" || boundary.AdmissionResonanceGraftBoundaryReadBackHash != wantReadBackHash {
		preflight.Reason = "candidate_admission_resonance_graft_boundary_read_back_hash_mismatch"
		return preflight
	}
	if boundary.AdmissionResonanceGraftBoundaryHash == boundary.AdmissionResonanceGraftBoundaryReadBackHash {
		preflight.Reason = "candidate_admission_resonance_graft_boundary_read_back_missing"
		return preflight
	}
	if !boundary.AdmissionResonanceGraftBoundaryShadowOnly ||
		boundary.AdmissionResonanceGraftBoundaryGraftAllowed ||
		boundary.AdmissionResonanceGraftBoundaryRawDreamTextAllowed ||
		boundary.AdmissionResonanceGraftBoundaryJanusSurfaceAllowed ||
		boundary.AdmissionResonanceGraftBoundaryCoocLearningAllowed ||
		boundary.AdmissionResonanceGraftBoundaryDeltaHarvestAllowed ||
		boundary.AdmissionResonanceGraftBoundaryBodyMutationAllowed ||
		!boundary.AdmissionResonanceGraftBoundaryRollbackRequired {
		preflight.Reason = "candidate_admission_resonance_graft_boundary_guard_mismatch"
		return preflight
	}
	if boundary.ContractsReady || boundary.WriteAllowed || boundary.MutatesState || boundary.LiveAdmissionEnabled || boundary.AdmissionAllowed {
		preflight.Reason = "candidate_admission_resonance_graft_boundary_already_open"
		return preflight
	}
	if boundary.BodyTarget != "none" {
		preflight.Reason = "candidate_admission_resonance_graft_boundary_body_target_mismatch"
		return preflight
	}
	if !boundary.LiveReady {
		preflight.Reason = "candidate_admission_resonance_graft_boundary_not_live_ready"
		return preflight
	}
	if boundary.SourceAdmissionResonanceObservationSchema != admissionLiveRouteTurnCandidateAdmissionResonanceObservationSchema ||
		!boundary.SourceAdmissionResonanceObservationPassed ||
		boundary.SourceAdmissionResonanceObservationID != boundary.AdmissionResonanceObservationID ||
		boundary.SourceAdmissionResonanceObservationAction != "record_resonance_receiver_observation_dry_run" ||
		!boundary.SourceAdmissionResonanceObservationReady ||
		boundary.SourceAdmissionResonanceObservationCausalID != boundary.AdmissionResonanceObservationCausalID ||
		boundary.SourceAdmissionResonanceObservationAppendHash != boundary.AdmissionResonanceObservationAppendHash ||
		boundary.SourceAdmissionResonanceObservationReadBackHash != boundary.AdmissionResonanceObservationReadBackHash {
		preflight.Reason = "candidate_admission_resonance_graft_boundary_source_observation_mismatch"
		return preflight
	}
	if boundary.SourceAdmissionResonanceReceiverIDForGraftBoundary != boundary.AdmissionResonanceReceiverID ||
		boundary.SourceAdmissionResonanceIntentIDForGraftBoundary != boundary.AdmissionResonanceIntentID ||
		boundary.SourceAdmissionFinalGateIDForGraftBoundary != boundary.AdmissionFinalGateID ||
		boundary.SourceAdmissionSealIDForGraftBoundary != boundary.AdmissionSealID ||
		boundary.SourceAdmissionPermitIDForGraftBoundary != boundary.AdmissionPermitID ||
		boundary.SourceAdmissionReadinessIDForGraftBoundary != boundary.AdmissionReadinessID ||
		boundary.SourceLedgerVerificationIDForGraftBoundary != boundary.LedgerVerificationID ||
		boundary.SourceLedgerPersistenceIDForGraftBoundary != boundary.LedgerPersistenceID ||
		boundary.SourceLedgerImplementationIDForGraftBoundary != boundary.LedgerImplementationID ||
		boundary.SourceAdmissionLedgerIDForGraftBoundary != boundary.AdmissionLedgerID ||
		boundary.SourceRollbackImplementationIDForGraftBoundary != boundary.RollbackImplementationID ||
		boundary.SourceWriterReceiptIDForGraftBoundary != boundary.WriterReceiptID {
		preflight.Reason = "candidate_admission_resonance_graft_boundary_source_id_mismatch"
		return preflight
	}
	if boundary.AdmissionResonanceGraftBoundaryID == "" ||
		boundary.AdmissionResonanceObservationID == "" ||
		boundary.AdmissionResonanceReceiverID == "" ||
		boundary.AdmissionResonanceIntentID == "" ||
		boundary.AdmissionFinalGateID == "" ||
		boundary.AdmissionSealID == "" ||
		boundary.AdmissionPermitID == "" ||
		boundary.AdmissionReadinessID == "" ||
		boundary.LedgerVerificationID == "" ||
		boundary.LedgerPersistenceID == "" ||
		boundary.LedgerImplementationID == "" ||
		boundary.RollbackImplementationID == "" ||
		boundary.WriterReceiptID == "" ||
		boundary.WriterImplementationID == "" ||
		boundary.AdmissionLedgerID == "" ||
		boundary.AdmissionWriterContractID == "" ||
		boundary.AdmissionWriterInventoryID == "" ||
		boundary.AdmissionWriterPreflightID == "" ||
		boundary.AdmissionLiveStageID == "" ||
		boundary.AdmissionEnableGateID == "" ||
		boundary.AdmissionSwitchID == "" ||
		boundary.AdmissionPromotionID == "" ||
		boundary.AdmissionDecisionID == "" ||
		boundary.AdmissionAdapterID == "" ||
		boundary.CandidateRunID == "" ||
		boundary.CandidateDraftID == "" ||
		boundary.CandidateExecutionID == "" ||
		boundary.GeneratorAdapterID == "" ||
		boundary.HandoffID == "" ||
		boundary.DreamCandidateRunID == "" ||
		boundary.CandidateTextHash == "" ||
		boundary.TurnTextHash == "" {
		preflight.Reason = "candidate_admission_resonance_graft_boundary_missing_provenance"
		return preflight
	}

	preflight.AdmissionResonanceGraftPreflightState = "shadow_graft_preflight_ready_dry_run"
	preflight.AdmissionResonanceGraftPreflightAction = "prepare_resonance_shadow_graft_preflight_dry_run"
	preflight.AdmissionResonanceGraftPreflightTarget = "resonance"
	preflight.AdmissionResonanceGraftPreflightTargetKind = "internal_world_shadow_graft_preflight"
	preflight.AdmissionResonanceGraftPreflightTargetMode = "receipt_only_closed_preflight_dry_run"
	preflight.AdmissionResonanceGraftPreflightReceiptShape = "resonance_shadow_graft_preflight_contract"
	preflight.AdmissionResonanceGraftPreflightDryRunOnly = true
	preflight.AdmissionResonanceGraftPreflightBoundaryVerified = true
	preflight.AdmissionResonanceGraftPreflightObservationVerified = boundary.AdmissionResonanceGraftBoundaryObservationVerified
	preflight.AdmissionResonanceGraftPreflightReceiverVerified = boundary.AdmissionResonanceGraftBoundaryReceiverVerified
	preflight.AdmissionResonanceGraftPreflightIntentVerified = boundary.AdmissionResonanceGraftBoundaryIntentVerified
	preflight.AdmissionResonanceGraftPreflightFinalGateVerified = boundary.AdmissionResonanceGraftBoundaryFinalGateVerified
	preflight.AdmissionResonanceGraftPreflightSealVerified = boundary.AdmissionResonanceGraftBoundarySealVerified
	preflight.AdmissionResonanceGraftPreflightPermitVerified = boundary.AdmissionResonanceGraftBoundaryPermitVerified
	preflight.AdmissionResonanceGraftPreflightReadinessVerified = boundary.AdmissionResonanceGraftBoundaryReadinessVerified
	preflight.AdmissionResonanceGraftPreflightLedgerVerified = boundary.AdmissionResonanceGraftBoundaryLedgerVerified
	preflight.AdmissionResonanceGraftPreflightWriterReady = boundary.AdmissionResonanceGraftBoundaryWriterReady
	preflight.AdmissionResonanceGraftPreflightRollbackReady = boundary.AdmissionResonanceGraftBoundaryRollbackReady
	preflight.AdmissionResonanceGraftPreflightLedgerReady = boundary.AdmissionResonanceGraftBoundaryLedgerReady
	preflight.AdmissionResonanceGraftPreflightKind = "shadow_graft_preflight"
	preflight.AdmissionResonanceGraftPreflightMode = "no_mutation_preflight"
	preflight.AdmissionResonanceGraftPreflightStage = "pre_live_graft_admission"
	preflight.AdmissionResonanceGraftPreflightAdmissionRequired = true
	preflight.AdmissionResonanceGraftPreflightShadowOnly = true
	preflight.AdmissionResonanceGraftPreflightGraftAllowed = false
	preflight.AdmissionResonanceGraftPreflightRawDreamTextAllowed = false
	preflight.AdmissionResonanceGraftPreflightJanusSurfaceAllowed = false
	preflight.AdmissionResonanceGraftPreflightCoocLearningAllowed = false
	preflight.AdmissionResonanceGraftPreflightDeltaHarvestAllowed = false
	preflight.AdmissionResonanceGraftPreflightBodyMutationAllowed = false
	preflight.AdmissionResonanceGraftPreflightRollbackRequired = true
	preflight.AdmissionResonanceGraftPreflightReady = true
	preflight.LiveReady = true
	preflight.BodyTarget = "none"
	preflight.ContractsReady = false
	preflight.WriteAllowed = false
	preflight.AdmissionAllowed = false
	preflight.LiveAdmissionEnabled = false
	preflight.MutatesState = false
	preflight.AdmissionResonanceGraftPreflightCausalID = admissionLiveRouteTurnCandidateAdmissionResonanceGraftPreflightCausalID(preflight)
	if preflight.AdmissionResonanceGraftPreflightCausalID == "" {
		preflight.Reason = "missing_candidate_admission_resonance_graft_preflight_causal_id"
		return preflight
	}
	preflight.AdmissionResonanceGraftPreflightHash = admissionLiveRouteTurnCandidateAdmissionResonanceGraftPreflightHash(preflight)
	preflight.AdmissionResonanceGraftPreflightReadBackHash = admissionLiveRouteTurnCandidateAdmissionResonanceGraftPreflightReadBackHash(preflight)
	if preflight.AdmissionResonanceGraftPreflightHash == "" ||
		preflight.AdmissionResonanceGraftPreflightReadBackHash == "" ||
		preflight.AdmissionResonanceGraftPreflightHash == preflight.AdmissionResonanceGraftPreflightReadBackHash {
		preflight.Reason = "candidate_admission_resonance_graft_preflight_read_back_missing"
		return preflight
	}
	preflight.AdmissionResonanceGraftPreflightID = admissionLiveRouteTurnCandidateAdmissionResonanceGraftPreflightID(preflight)
	if preflight.AdmissionResonanceGraftPreflightID == "" {
		preflight.Reason = "missing_candidate_admission_resonance_graft_preflight_id"
		return preflight
	}
	preflight.Passed = true
	preflight.Reason = "resonance shadow graft preflight prepared without body mutation"
	return preflight
}

func admissionLiveRouteTurnCandidateAdmissionResonanceGraftPreflightCausalID(preflight admissionLiveRouteTurnCandidateAdmissionResonanceGraftPreflight) string {
	h := hashJSON(struct {
		AdmissionResonanceGraftBoundaryID           string `json:"admission_resonance_graft_boundary_id"`
		AdmissionResonanceGraftBoundaryReadBackHash string `json:"admission_resonance_graft_boundary_read_back_hash"`
		AdmissionResonanceObservationID             string `json:"admission_resonance_observation_id"`
		AdmissionResonanceReceiverID                string `json:"admission_resonance_receiver_id"`
		CandidateRunID                              string `json:"candidate_run_id"`
		CandidateTextHash                           string `json:"candidate_text_hash"`
		TurnTextHash                                string `json:"turn_text_hash"`
		PreflightKind                               string `json:"preflight_kind"`
		PreflightStage                              string `json:"preflight_stage"`
	}{
		AdmissionResonanceGraftBoundaryID:           preflight.AdmissionResonanceGraftBoundaryID,
		AdmissionResonanceGraftBoundaryReadBackHash: preflight.AdmissionResonanceGraftBoundaryReadBackHash,
		AdmissionResonanceObservationID:             preflight.AdmissionResonanceObservationID,
		AdmissionResonanceReceiverID:                preflight.AdmissionResonanceReceiverID,
		CandidateRunID:                              preflight.CandidateRunID,
		CandidateTextHash:                           preflight.CandidateTextHash,
		TurnTextHash:                                preflight.TurnTextHash,
		PreflightKind:                               preflight.AdmissionResonanceGraftPreflightKind,
		PreflightStage:                              preflight.AdmissionResonanceGraftPreflightStage,
	})
	if h == "" {
		return ""
	}
	return "resonance-graft-preflight-causal-" + h
}

func admissionLiveRouteTurnCandidateAdmissionResonanceGraftPreflightHash(preflight admissionLiveRouteTurnCandidateAdmissionResonanceGraftPreflight) string {
	h := hashJSON(struct {
		AdmissionResonanceGraftPreflightCausalID string `json:"admission_resonance_graft_preflight_causal_id"`
		AdmissionResonanceGraftBoundaryID        string `json:"admission_resonance_graft_boundary_id"`
		BoundaryHash                             string `json:"boundary_hash"`
		BoundaryReadBackHash                     string `json:"boundary_read_back_hash"`
		PreflightMode                            string `json:"preflight_mode"`
		AdmissionRequired                        bool   `json:"admission_required"`
		ShadowOnly                               bool   `json:"shadow_only"`
		GraftAllowed                             bool   `json:"graft_allowed"`
		DryRunOnly                               bool   `json:"dry_run_only"`
	}{
		AdmissionResonanceGraftPreflightCausalID: preflight.AdmissionResonanceGraftPreflightCausalID,
		AdmissionResonanceGraftBoundaryID:        preflight.AdmissionResonanceGraftBoundaryID,
		BoundaryHash:                             preflight.AdmissionResonanceGraftBoundaryHash,
		BoundaryReadBackHash:                     preflight.AdmissionResonanceGraftBoundaryReadBackHash,
		PreflightMode:                            preflight.AdmissionResonanceGraftPreflightMode,
		AdmissionRequired:                        preflight.AdmissionResonanceGraftPreflightAdmissionRequired,
		ShadowOnly:                               preflight.AdmissionResonanceGraftPreflightShadowOnly,
		GraftAllowed:                             preflight.AdmissionResonanceGraftPreflightGraftAllowed,
		DryRunOnly:                               preflight.AdmissionResonanceGraftPreflightDryRunOnly,
	})
	if h == "" {
		return ""
	}
	return "resonance-graft-preflight-" + h
}

func admissionLiveRouteTurnCandidateAdmissionResonanceGraftPreflightReadBackHash(preflight admissionLiveRouteTurnCandidateAdmissionResonanceGraftPreflight) string {
	h := hashJSON(struct {
		PreflightHash  string `json:"preflight_hash"`
		BoundaryID     string `json:"boundary_id"`
		PreflightKind  string `json:"preflight_kind"`
		PreflightReady bool   `json:"preflight_ready"`
		BodyMutation   bool   `json:"body_mutation"`
		AdmissionOpen  bool   `json:"admission_open"`
	}{
		PreflightHash:  preflight.AdmissionResonanceGraftPreflightHash,
		BoundaryID:     preflight.AdmissionResonanceGraftBoundaryID,
		PreflightKind:  preflight.AdmissionResonanceGraftPreflightKind,
		PreflightReady: preflight.AdmissionResonanceGraftPreflightReady,
		BodyMutation:   preflight.AdmissionResonanceGraftPreflightBodyMutationAllowed,
		AdmissionOpen:  preflight.LiveAdmissionEnabled,
	})
	if h == "" {
		return ""
	}
	return "resonance-graft-preflight-read-" + h
}

func admissionLiveRouteTurnCandidateAdmissionResonanceGraftPreflightID(preflight admissionLiveRouteTurnCandidateAdmissionResonanceGraftPreflight) string {
	h := hashJSON(struct {
		AdmissionResonanceGraftPreflightCausalID     string `json:"admission_resonance_graft_preflight_causal_id"`
		AdmissionResonanceGraftPreflightHash         string `json:"admission_resonance_graft_preflight_hash"`
		AdmissionResonanceGraftPreflightReadBackHash string `json:"admission_resonance_graft_preflight_read_back_hash"`
		AdmissionResonanceGraftBoundaryID            string `json:"admission_resonance_graft_boundary_id"`
		AdmissionResonanceObservationID              string `json:"admission_resonance_observation_id"`
		AdmissionResonanceReceiverID                 string `json:"admission_resonance_receiver_id"`
		AdmissionFinalGateID                         string `json:"admission_final_gate_id"`
		LedgerVerificationID                         string `json:"ledger_verification_id"`
		PreflightState                               string `json:"preflight_state"`
		PreflightAction                              string `json:"preflight_action"`
		PreflightTarget                              string `json:"preflight_target"`
		PreflightTargetKind                          string `json:"preflight_target_kind"`
		PreflightTargetMode                          string `json:"preflight_target_mode"`
		PreflightReceiptShape                        string `json:"preflight_receipt_shape"`
		PreflightDryRunOnly                          bool   `json:"preflight_dry_run_only"`
		BoundaryVerified                             bool   `json:"boundary_verified"`
		AdmissionRequired                            bool   `json:"admission_required"`
		ShadowOnly                                   bool   `json:"shadow_only"`
		GraftAllowed                                 bool   `json:"graft_allowed"`
		RawTextAllowed                               bool   `json:"raw_text_allowed"`
		JanusSurfaceAllowed                          bool   `json:"janus_surface_allowed"`
		CoocLearningAllowed                          bool   `json:"cooc_learning_allowed"`
		DeltaHarvestAllowed                          bool   `json:"delta_harvest_allowed"`
		BodyMutationAllowed                          bool   `json:"body_mutation_allowed"`
		RollbackRequired                             bool   `json:"rollback_required"`
		PreflightReady                               bool   `json:"preflight_ready"`
		ContractsReady                               bool   `json:"contracts_ready"`
		BodyTarget                                   string `json:"body_target"`
		WriteAllowed                                 bool   `json:"write_allowed"`
		AdmissionAllowed                             bool   `json:"admission_allowed"`
		LiveAdmissionEnabled                         bool   `json:"live_admission_enabled"`
		MutatesState                                 bool   `json:"mutates_state"`
	}{
		AdmissionResonanceGraftPreflightCausalID:     preflight.AdmissionResonanceGraftPreflightCausalID,
		AdmissionResonanceGraftPreflightHash:         preflight.AdmissionResonanceGraftPreflightHash,
		AdmissionResonanceGraftPreflightReadBackHash: preflight.AdmissionResonanceGraftPreflightReadBackHash,
		AdmissionResonanceGraftBoundaryID:            preflight.AdmissionResonanceGraftBoundaryID,
		AdmissionResonanceObservationID:              preflight.AdmissionResonanceObservationID,
		AdmissionResonanceReceiverID:                 preflight.AdmissionResonanceReceiverID,
		AdmissionFinalGateID:                         preflight.AdmissionFinalGateID,
		LedgerVerificationID:                         preflight.LedgerVerificationID,
		PreflightState:                               preflight.AdmissionResonanceGraftPreflightState,
		PreflightAction:                              preflight.AdmissionResonanceGraftPreflightAction,
		PreflightTarget:                              preflight.AdmissionResonanceGraftPreflightTarget,
		PreflightTargetKind:                          preflight.AdmissionResonanceGraftPreflightTargetKind,
		PreflightTargetMode:                          preflight.AdmissionResonanceGraftPreflightTargetMode,
		PreflightReceiptShape:                        preflight.AdmissionResonanceGraftPreflightReceiptShape,
		PreflightDryRunOnly:                          preflight.AdmissionResonanceGraftPreflightDryRunOnly,
		BoundaryVerified:                             preflight.AdmissionResonanceGraftPreflightBoundaryVerified,
		AdmissionRequired:                            preflight.AdmissionResonanceGraftPreflightAdmissionRequired,
		ShadowOnly:                                   preflight.AdmissionResonanceGraftPreflightShadowOnly,
		GraftAllowed:                                 preflight.AdmissionResonanceGraftPreflightGraftAllowed,
		RawTextAllowed:                               preflight.AdmissionResonanceGraftPreflightRawDreamTextAllowed,
		JanusSurfaceAllowed:                          preflight.AdmissionResonanceGraftPreflightJanusSurfaceAllowed,
		CoocLearningAllowed:                          preflight.AdmissionResonanceGraftPreflightCoocLearningAllowed,
		DeltaHarvestAllowed:                          preflight.AdmissionResonanceGraftPreflightDeltaHarvestAllowed,
		BodyMutationAllowed:                          preflight.AdmissionResonanceGraftPreflightBodyMutationAllowed,
		RollbackRequired:                             preflight.AdmissionResonanceGraftPreflightRollbackRequired,
		PreflightReady:                               preflight.AdmissionResonanceGraftPreflightReady,
		ContractsReady:                               preflight.ContractsReady,
		BodyTarget:                                   preflight.BodyTarget,
		WriteAllowed:                                 preflight.WriteAllowed,
		AdmissionAllowed:                             preflight.AdmissionAllowed,
		LiveAdmissionEnabled:                         preflight.LiveAdmissionEnabled,
		MutatesState:                                 preflight.MutatesState,
	})
	if h == "" {
		return ""
	}
	return "resonance-graft-preflight-id-" + h
}

func recordAdmissionLiveRouteTurnCandidateAdmissionResonanceGraftPreflight(preflight admissionLiveRouteTurnCandidateAdmissionResonanceGraftPreflight) error {
	path := strings.TrimSpace(os.Getenv("AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_RESONANCE_GRAFT_PREFLIGHT_LOG"))
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

func admissionLiveRouteTurnCandidateAdmissionResonanceGraftGateForPreflight(preflight admissionLiveRouteTurnCandidateAdmissionResonanceGraftPreflight) admissionLiveRouteTurnCandidateAdmissionResonanceGraftGate {
	sourceSchema := preflight.Schema
	gate := admissionLiveRouteTurnCandidateAdmissionResonanceGraftGate{
		admissionLiveRouteTurnCandidateAdmissionResonanceGraftPreflight: preflight,
		AdmissionResonanceGraftGateState:                                "blocked",
		AdmissionResonanceGraftGateAction:                               "reject",
		AdmissionResonanceGraftGateDryRunOnly:                           true,
		SourceAdmissionResonanceGraftPreflightSchema:                    sourceSchema,
		SourceAdmissionResonanceGraftPreflightPassed:                    preflight.Passed,
		SourceAdmissionResonanceGraftPreflightID:                        preflight.AdmissionResonanceGraftPreflightID,
		SourceAdmissionResonanceGraftPreflightAction:                    preflight.AdmissionResonanceGraftPreflightAction,
		SourceAdmissionResonanceGraftPreflightReady:                     preflight.AdmissionResonanceGraftPreflightReady,
		SourceAdmissionResonanceGraftPreflightCausalID:                  preflight.AdmissionResonanceGraftPreflightCausalID,
		SourceAdmissionResonanceGraftPreflightHash:                      preflight.AdmissionResonanceGraftPreflightHash,
		SourceAdmissionResonanceGraftPreflightReadBackHash:              preflight.AdmissionResonanceGraftPreflightReadBackHash,
		SourceAdmissionResonanceGraftBoundaryIDForGraftGate:             preflight.AdmissionResonanceGraftBoundaryID,
		SourceAdmissionResonanceObservationIDForGraftGate:               preflight.AdmissionResonanceObservationID,
		SourceAdmissionResonanceReceiverIDForGraftGate:                  preflight.AdmissionResonanceReceiverID,
		SourceAdmissionResonanceIntentIDForGraftGate:                    preflight.AdmissionResonanceIntentID,
		SourceAdmissionFinalGateIDForGraftGate:                          preflight.AdmissionFinalGateID,
		SourceAdmissionSealIDForGraftGate:                               preflight.AdmissionSealID,
		SourceAdmissionPermitIDForGraftGate:                             preflight.AdmissionPermitID,
		SourceAdmissionReadinessIDForGraftGate:                          preflight.AdmissionReadinessID,
		SourceLedgerVerificationIDForGraftGate:                          preflight.LedgerVerificationID,
		SourceLedgerPersistenceIDForGraftGate:                           preflight.LedgerPersistenceID,
		SourceLedgerImplementationIDForGraftGate:                        preflight.LedgerImplementationID,
		SourceAdmissionLedgerIDForGraftGate:                             preflight.AdmissionLedgerID,
		SourceRollbackImplementationIDForGraftGate:                      preflight.RollbackImplementationID,
		SourceWriterReceiptIDForGraftGate:                               preflight.WriterReceiptID,
	}
	gate.Schema = admissionLiveRouteTurnCandidateAdmissionResonanceGraftGateSchema
	gate.Timing = "live_admission_resonance_graft_gate"
	gate.Passed = false
	gate.AdmissionResonanceGraftGateID = ""
	gate.AdmissionResonanceGraftGateReady = false
	gate.LiveReady = false
	gate.BodyTarget = "none"
	gate.ContractsReady = false
	gate.WriteAllowed = false
	gate.AdmissionAllowed = false
	gate.LiveAdmissionEnabled = false
	gate.MutatesState = false

	if sourceSchema == "" {
		gate.Reason = "missing_candidate_admission_resonance_graft_preflight"
		return gate
	}
	if sourceSchema != admissionLiveRouteTurnCandidateAdmissionResonanceGraftPreflightSchema {
		gate.Reason = "unexpected_candidate_admission_resonance_graft_preflight_schema " + sourceSchema
		return gate
	}
	if !preflight.Passed {
		gate.Reason = "candidate_admission_resonance_graft_preflight_failed"
		if preflight.Reason != "" {
			gate.Reason += ": " + preflight.Reason
		}
		return gate
	}
	if preflight.AdmissionResonanceGraftPreflightID == "" {
		gate.Reason = "missing_candidate_admission_resonance_graft_preflight_id"
		return gate
	}
	if wantPreflightID := admissionLiveRouteTurnCandidateAdmissionResonanceGraftPreflightID(preflight); wantPreflightID == "" || preflight.AdmissionResonanceGraftPreflightID != wantPreflightID {
		gate.Reason = "candidate_admission_resonance_graft_preflight_id_mismatch"
		return gate
	}
	if preflight.AdmissionResonanceGraftPreflightState != "shadow_graft_preflight_ready_dry_run" ||
		preflight.AdmissionResonanceGraftPreflightAction != "prepare_resonance_shadow_graft_preflight_dry_run" ||
		preflight.AdmissionResonanceGraftPreflightTarget != "resonance" ||
		preflight.AdmissionResonanceGraftPreflightTargetKind != "internal_world_shadow_graft_preflight" ||
		preflight.AdmissionResonanceGraftPreflightTargetMode != "receipt_only_closed_preflight_dry_run" ||
		preflight.AdmissionResonanceGraftPreflightReceiptShape != "resonance_shadow_graft_preflight_contract" {
		gate.Reason = "candidate_admission_resonance_graft_preflight_shape_mismatch"
		return gate
	}
	if !preflight.AdmissionResonanceGraftPreflightDryRunOnly ||
		!preflight.AdmissionResonanceGraftPreflightBoundaryVerified ||
		!preflight.AdmissionResonanceGraftPreflightObservationVerified ||
		!preflight.AdmissionResonanceGraftPreflightReceiverVerified ||
		!preflight.AdmissionResonanceGraftPreflightIntentVerified ||
		!preflight.AdmissionResonanceGraftPreflightFinalGateVerified ||
		!preflight.AdmissionResonanceGraftPreflightSealVerified ||
		!preflight.AdmissionResonanceGraftPreflightPermitVerified ||
		!preflight.AdmissionResonanceGraftPreflightReadinessVerified ||
		!preflight.AdmissionResonanceGraftPreflightLedgerVerified ||
		!preflight.AdmissionResonanceGraftPreflightWriterReady ||
		!preflight.AdmissionResonanceGraftPreflightRollbackReady ||
		!preflight.AdmissionResonanceGraftPreflightLedgerReady ||
		!preflight.AdmissionResonanceGraftPreflightReady {
		gate.Reason = "candidate_admission_resonance_graft_preflight_not_verified_dry_run"
		return gate
	}
	if preflight.AdmissionResonanceGraftPreflightKind != "shadow_graft_preflight" ||
		preflight.AdmissionResonanceGraftPreflightMode != "no_mutation_preflight" ||
		preflight.AdmissionResonanceGraftPreflightStage != "pre_live_graft_admission" ||
		preflight.AdmissionResonanceGraftPreflightCausalID == "" ||
		preflight.AdmissionResonanceGraftPreflightHash == "" ||
		preflight.AdmissionResonanceGraftPreflightReadBackHash == "" {
		gate.Reason = "candidate_admission_resonance_graft_preflight_target_mismatch"
		return gate
	}
	if wantCausalID := admissionLiveRouteTurnCandidateAdmissionResonanceGraftPreflightCausalID(preflight); wantCausalID == "" || preflight.AdmissionResonanceGraftPreflightCausalID != wantCausalID {
		gate.Reason = "candidate_admission_resonance_graft_preflight_causal_id_mismatch"
		return gate
	}
	if wantHash := admissionLiveRouteTurnCandidateAdmissionResonanceGraftPreflightHash(preflight); wantHash == "" || preflight.AdmissionResonanceGraftPreflightHash != wantHash {
		gate.Reason = "candidate_admission_resonance_graft_preflight_hash_mismatch"
		return gate
	}
	if wantReadBackHash := admissionLiveRouteTurnCandidateAdmissionResonanceGraftPreflightReadBackHash(preflight); wantReadBackHash == "" || preflight.AdmissionResonanceGraftPreflightReadBackHash != wantReadBackHash {
		gate.Reason = "candidate_admission_resonance_graft_preflight_read_back_hash_mismatch"
		return gate
	}
	if preflight.AdmissionResonanceGraftPreflightHash == preflight.AdmissionResonanceGraftPreflightReadBackHash {
		gate.Reason = "candidate_admission_resonance_graft_preflight_read_back_missing"
		return gate
	}
	if !preflight.AdmissionResonanceGraftPreflightAdmissionRequired ||
		!preflight.AdmissionResonanceGraftPreflightShadowOnly ||
		preflight.AdmissionResonanceGraftPreflightGraftAllowed ||
		preflight.AdmissionResonanceGraftPreflightRawDreamTextAllowed ||
		preflight.AdmissionResonanceGraftPreflightJanusSurfaceAllowed ||
		preflight.AdmissionResonanceGraftPreflightCoocLearningAllowed ||
		preflight.AdmissionResonanceGraftPreflightDeltaHarvestAllowed ||
		preflight.AdmissionResonanceGraftPreflightBodyMutationAllowed ||
		!preflight.AdmissionResonanceGraftPreflightRollbackRequired {
		gate.Reason = "candidate_admission_resonance_graft_preflight_guard_mismatch"
		return gate
	}
	if preflight.ContractsReady || preflight.WriteAllowed || preflight.MutatesState || preflight.LiveAdmissionEnabled || preflight.AdmissionAllowed {
		gate.Reason = "candidate_admission_resonance_graft_preflight_already_open"
		return gate
	}
	if preflight.BodyTarget != "none" {
		gate.Reason = "candidate_admission_resonance_graft_preflight_body_target_mismatch"
		return gate
	}
	if !preflight.LiveReady {
		gate.Reason = "candidate_admission_resonance_graft_preflight_not_live_ready"
		return gate
	}
	if preflight.SourceAdmissionResonanceGraftBoundarySchema != admissionLiveRouteTurnCandidateAdmissionResonanceGraftBoundarySchema ||
		!preflight.SourceAdmissionResonanceGraftBoundaryPassed ||
		preflight.SourceAdmissionResonanceGraftBoundaryID != preflight.AdmissionResonanceGraftBoundaryID ||
		preflight.SourceAdmissionResonanceGraftBoundaryAction != "declare_resonance_shadow_graft_boundary_dry_run" ||
		!preflight.SourceAdmissionResonanceGraftBoundaryReady ||
		preflight.SourceAdmissionResonanceGraftBoundaryCausalID != preflight.AdmissionResonanceGraftBoundaryCausalID ||
		preflight.SourceAdmissionResonanceGraftBoundaryHash != preflight.AdmissionResonanceGraftBoundaryHash ||
		preflight.SourceAdmissionResonanceGraftBoundaryReadBackHash != preflight.AdmissionResonanceGraftBoundaryReadBackHash {
		gate.Reason = "candidate_admission_resonance_graft_preflight_source_boundary_mismatch"
		return gate
	}
	if preflight.SourceAdmissionResonanceObservationIDForGraftPreflight != preflight.AdmissionResonanceObservationID ||
		preflight.SourceAdmissionResonanceReceiverIDForGraftPreflight != preflight.AdmissionResonanceReceiverID ||
		preflight.SourceAdmissionResonanceIntentIDForGraftPreflight != preflight.AdmissionResonanceIntentID ||
		preflight.SourceAdmissionFinalGateIDForGraftPreflight != preflight.AdmissionFinalGateID ||
		preflight.SourceAdmissionSealIDForGraftPreflight != preflight.AdmissionSealID ||
		preflight.SourceAdmissionPermitIDForGraftPreflight != preflight.AdmissionPermitID ||
		preflight.SourceAdmissionReadinessIDForGraftPreflight != preflight.AdmissionReadinessID ||
		preflight.SourceLedgerVerificationIDForGraftPreflight != preflight.LedgerVerificationID ||
		preflight.SourceLedgerPersistenceIDForGraftPreflight != preflight.LedgerPersistenceID ||
		preflight.SourceLedgerImplementationIDForGraftPreflight != preflight.LedgerImplementationID ||
		preflight.SourceAdmissionLedgerIDForGraftPreflight != preflight.AdmissionLedgerID ||
		preflight.SourceRollbackImplementationIDForGraftPreflight != preflight.RollbackImplementationID ||
		preflight.SourceWriterReceiptIDForGraftPreflight != preflight.WriterReceiptID {
		gate.Reason = "candidate_admission_resonance_graft_preflight_source_id_mismatch"
		return gate
	}
	if preflight.AdmissionResonanceGraftPreflightID == "" ||
		preflight.AdmissionResonanceGraftBoundaryID == "" ||
		preflight.AdmissionResonanceObservationID == "" ||
		preflight.AdmissionResonanceReceiverID == "" ||
		preflight.AdmissionResonanceIntentID == "" ||
		preflight.AdmissionFinalGateID == "" ||
		preflight.AdmissionSealID == "" ||
		preflight.AdmissionPermitID == "" ||
		preflight.AdmissionReadinessID == "" ||
		preflight.LedgerVerificationID == "" ||
		preflight.LedgerPersistenceID == "" ||
		preflight.LedgerImplementationID == "" ||
		preflight.RollbackImplementationID == "" ||
		preflight.WriterReceiptID == "" ||
		preflight.WriterImplementationID == "" ||
		preflight.AdmissionLedgerID == "" ||
		preflight.AdmissionWriterContractID == "" ||
		preflight.AdmissionWriterInventoryID == "" ||
		preflight.AdmissionWriterPreflightID == "" ||
		preflight.AdmissionLiveStageID == "" ||
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
		gate.Reason = "candidate_admission_resonance_graft_preflight_missing_provenance"
		return gate
	}

	gate.AdmissionResonanceGraftGateState = "shadow_graft_gate_ready_dry_run"
	gate.AdmissionResonanceGraftGateAction = "gate_resonance_shadow_graft_dry_run"
	gate.AdmissionResonanceGraftGateTarget = "resonance"
	gate.AdmissionResonanceGraftGateTargetKind = "internal_world_shadow_graft_gate"
	gate.AdmissionResonanceGraftGateTargetMode = "receipt_only_closed_gate_dry_run"
	gate.AdmissionResonanceGraftGateReceiptShape = "resonance_shadow_graft_gate_contract"
	gate.AdmissionResonanceGraftGateDryRunOnly = true
	gate.AdmissionResonanceGraftGatePreflightVerified = true
	gate.AdmissionResonanceGraftGateBoundaryVerified = preflight.AdmissionResonanceGraftPreflightBoundaryVerified
	gate.AdmissionResonanceGraftGateObservationVerified = preflight.AdmissionResonanceGraftPreflightObservationVerified
	gate.AdmissionResonanceGraftGateReceiverVerified = preflight.AdmissionResonanceGraftPreflightReceiverVerified
	gate.AdmissionResonanceGraftGateIntentVerified = preflight.AdmissionResonanceGraftPreflightIntentVerified
	gate.AdmissionResonanceGraftGateFinalGateVerified = preflight.AdmissionResonanceGraftPreflightFinalGateVerified
	gate.AdmissionResonanceGraftGateSealVerified = preflight.AdmissionResonanceGraftPreflightSealVerified
	gate.AdmissionResonanceGraftGatePermitVerified = preflight.AdmissionResonanceGraftPreflightPermitVerified
	gate.AdmissionResonanceGraftGateReadinessVerified = preflight.AdmissionResonanceGraftPreflightReadinessVerified
	gate.AdmissionResonanceGraftGateLedgerVerified = preflight.AdmissionResonanceGraftPreflightLedgerVerified
	gate.AdmissionResonanceGraftGateWriterReady = preflight.AdmissionResonanceGraftPreflightWriterReady
	gate.AdmissionResonanceGraftGateRollbackReady = preflight.AdmissionResonanceGraftPreflightRollbackReady
	gate.AdmissionResonanceGraftGateLedgerReady = preflight.AdmissionResonanceGraftPreflightLedgerReady
	gate.AdmissionResonanceGraftGateKind = "shadow_graft_gate"
	gate.AdmissionResonanceGraftGateMode = "no_mutation_gate"
	gate.AdmissionResonanceGraftGateStage = "pre_live_graft_gate"
	gate.AdmissionResonanceGraftGateAdmissionRequired = true
	gate.AdmissionResonanceGraftGateShadowOnly = true
	gate.AdmissionResonanceGraftGateGraftAllowed = false
	gate.AdmissionResonanceGraftGateRawDreamTextAllowed = false
	gate.AdmissionResonanceGraftGateJanusSurfaceAllowed = false
	gate.AdmissionResonanceGraftGateCoocLearningAllowed = false
	gate.AdmissionResonanceGraftGateDeltaHarvestAllowed = false
	gate.AdmissionResonanceGraftGateBodyMutationAllowed = false
	gate.AdmissionResonanceGraftGateRollbackRequired = true
	gate.AdmissionResonanceGraftGateReady = true
	gate.LiveReady = true
	gate.BodyTarget = "none"
	gate.ContractsReady = false
	gate.WriteAllowed = false
	gate.AdmissionAllowed = false
	gate.LiveAdmissionEnabled = false
	gate.MutatesState = false
	gate.AdmissionResonanceGraftGateCausalID = admissionLiveRouteTurnCandidateAdmissionResonanceGraftGateCausalID(gate)
	if gate.AdmissionResonanceGraftGateCausalID == "" {
		gate.Reason = "missing_candidate_admission_resonance_graft_gate_causal_id"
		return gate
	}
	gate.AdmissionResonanceGraftGateHash = admissionLiveRouteTurnCandidateAdmissionResonanceGraftGateHash(gate)
	gate.AdmissionResonanceGraftGateReadBackHash = admissionLiveRouteTurnCandidateAdmissionResonanceGraftGateReadBackHash(gate)
	if gate.AdmissionResonanceGraftGateHash == "" ||
		gate.AdmissionResonanceGraftGateReadBackHash == "" ||
		gate.AdmissionResonanceGraftGateHash == gate.AdmissionResonanceGraftGateReadBackHash {
		gate.Reason = "candidate_admission_resonance_graft_gate_read_back_missing"
		return gate
	}
	gate.AdmissionResonanceGraftGateID = admissionLiveRouteTurnCandidateAdmissionResonanceGraftGateID(gate)
	if gate.AdmissionResonanceGraftGateID == "" {
		gate.Reason = "missing_candidate_admission_resonance_graft_gate_id"
		return gate
	}
	gate.Passed = true
	gate.Reason = "resonance shadow graft gate prepared without body mutation"
	return gate
}

func admissionLiveRouteTurnCandidateAdmissionResonanceGraftGateCausalID(gate admissionLiveRouteTurnCandidateAdmissionResonanceGraftGate) string {
	h := hashJSON(struct {
		AdmissionResonanceGraftPreflightID           string `json:"admission_resonance_graft_preflight_id"`
		AdmissionResonanceGraftPreflightReadBackHash string `json:"admission_resonance_graft_preflight_read_back_hash"`
		AdmissionResonanceGraftBoundaryID            string `json:"admission_resonance_graft_boundary_id"`
		AdmissionResonanceObservationID              string `json:"admission_resonance_observation_id"`
		AdmissionResonanceReceiverID                 string `json:"admission_resonance_receiver_id"`
		CandidateRunID                               string `json:"candidate_run_id"`
		CandidateTextHash                            string `json:"candidate_text_hash"`
		TurnTextHash                                 string `json:"turn_text_hash"`
		GateKind                                     string `json:"gate_kind"`
		GateStage                                    string `json:"gate_stage"`
	}{
		AdmissionResonanceGraftPreflightID:           gate.AdmissionResonanceGraftPreflightID,
		AdmissionResonanceGraftPreflightReadBackHash: gate.AdmissionResonanceGraftPreflightReadBackHash,
		AdmissionResonanceGraftBoundaryID:            gate.AdmissionResonanceGraftBoundaryID,
		AdmissionResonanceObservationID:              gate.AdmissionResonanceObservationID,
		AdmissionResonanceReceiverID:                 gate.AdmissionResonanceReceiverID,
		CandidateRunID:                               gate.CandidateRunID,
		CandidateTextHash:                            gate.CandidateTextHash,
		TurnTextHash:                                 gate.TurnTextHash,
		GateKind:                                     gate.AdmissionResonanceGraftGateKind,
		GateStage:                                    gate.AdmissionResonanceGraftGateStage,
	})
	if h == "" {
		return ""
	}
	return "resonance-graft-gate-causal-" + h
}

func admissionLiveRouteTurnCandidateAdmissionResonanceGraftGateHash(gate admissionLiveRouteTurnCandidateAdmissionResonanceGraftGate) string {
	h := hashJSON(struct {
		AdmissionResonanceGraftGateCausalID string `json:"admission_resonance_graft_gate_causal_id"`
		AdmissionResonanceGraftPreflightID  string `json:"admission_resonance_graft_preflight_id"`
		PreflightHash                       string `json:"preflight_hash"`
		PreflightReadBackHash               string `json:"preflight_read_back_hash"`
		GateMode                            string `json:"gate_mode"`
		AdmissionRequired                   bool   `json:"admission_required"`
		ShadowOnly                          bool   `json:"shadow_only"`
		GraftAllowed                        bool   `json:"graft_allowed"`
		DryRunOnly                          bool   `json:"dry_run_only"`
	}{
		AdmissionResonanceGraftGateCausalID: gate.AdmissionResonanceGraftGateCausalID,
		AdmissionResonanceGraftPreflightID:  gate.AdmissionResonanceGraftPreflightID,
		PreflightHash:                       gate.AdmissionResonanceGraftPreflightHash,
		PreflightReadBackHash:               gate.AdmissionResonanceGraftPreflightReadBackHash,
		GateMode:                            gate.AdmissionResonanceGraftGateMode,
		AdmissionRequired:                   gate.AdmissionResonanceGraftGateAdmissionRequired,
		ShadowOnly:                          gate.AdmissionResonanceGraftGateShadowOnly,
		GraftAllowed:                        gate.AdmissionResonanceGraftGateGraftAllowed,
		DryRunOnly:                          gate.AdmissionResonanceGraftGateDryRunOnly,
	})
	if h == "" {
		return ""
	}
	return "resonance-graft-gate-" + h
}

func admissionLiveRouteTurnCandidateAdmissionResonanceGraftGateReadBackHash(gate admissionLiveRouteTurnCandidateAdmissionResonanceGraftGate) string {
	h := hashJSON(struct {
		GateHash      string `json:"gate_hash"`
		PreflightID   string `json:"preflight_id"`
		GateKind      string `json:"gate_kind"`
		GateReady     bool   `json:"gate_ready"`
		BodyMutation  bool   `json:"body_mutation"`
		AdmissionOpen bool   `json:"admission_open"`
	}{
		GateHash:      gate.AdmissionResonanceGraftGateHash,
		PreflightID:   gate.AdmissionResonanceGraftPreflightID,
		GateKind:      gate.AdmissionResonanceGraftGateKind,
		GateReady:     gate.AdmissionResonanceGraftGateReady,
		BodyMutation:  gate.AdmissionResonanceGraftGateBodyMutationAllowed,
		AdmissionOpen: gate.LiveAdmissionEnabled,
	})
	if h == "" {
		return ""
	}
	return "resonance-graft-gate-read-" + h
}

func admissionLiveRouteTurnCandidateAdmissionResonanceGraftGateID(gate admissionLiveRouteTurnCandidateAdmissionResonanceGraftGate) string {
	h := hashJSON(struct {
		AdmissionResonanceGraftGateCausalID     string `json:"admission_resonance_graft_gate_causal_id"`
		AdmissionResonanceGraftGateHash         string `json:"admission_resonance_graft_gate_hash"`
		AdmissionResonanceGraftGateReadBackHash string `json:"admission_resonance_graft_gate_read_back_hash"`
		AdmissionResonanceGraftPreflightID      string `json:"admission_resonance_graft_preflight_id"`
		AdmissionResonanceGraftBoundaryID       string `json:"admission_resonance_graft_boundary_id"`
		AdmissionResonanceObservationID         string `json:"admission_resonance_observation_id"`
		AdmissionResonanceReceiverID            string `json:"admission_resonance_receiver_id"`
		AdmissionFinalGateID                    string `json:"admission_final_gate_id"`
		LedgerVerificationID                    string `json:"ledger_verification_id"`
		GateState                               string `json:"gate_state"`
		GateAction                              string `json:"gate_action"`
		GateTarget                              string `json:"gate_target"`
		GateTargetKind                          string `json:"gate_target_kind"`
		GateTargetMode                          string `json:"gate_target_mode"`
		GateReceiptShape                        string `json:"gate_receipt_shape"`
		GateDryRunOnly                          bool   `json:"gate_dry_run_only"`
		PreflightVerified                       bool   `json:"preflight_verified"`
		AdmissionRequired                       bool   `json:"admission_required"`
		ShadowOnly                              bool   `json:"shadow_only"`
		GraftAllowed                            bool   `json:"graft_allowed"`
		RawTextAllowed                          bool   `json:"raw_text_allowed"`
		JanusSurfaceAllowed                     bool   `json:"janus_surface_allowed"`
		CoocLearningAllowed                     bool   `json:"cooc_learning_allowed"`
		DeltaHarvestAllowed                     bool   `json:"delta_harvest_allowed"`
		BodyMutationAllowed                     bool   `json:"body_mutation_allowed"`
		RollbackRequired                        bool   `json:"rollback_required"`
		GateReady                               bool   `json:"gate_ready"`
		ContractsReady                          bool   `json:"contracts_ready"`
		BodyTarget                              string `json:"body_target"`
		WriteAllowed                            bool   `json:"write_allowed"`
		AdmissionAllowed                        bool   `json:"admission_allowed"`
		LiveAdmissionEnabled                    bool   `json:"live_admission_enabled"`
		MutatesState                            bool   `json:"mutates_state"`
	}{
		AdmissionResonanceGraftGateCausalID:     gate.AdmissionResonanceGraftGateCausalID,
		AdmissionResonanceGraftGateHash:         gate.AdmissionResonanceGraftGateHash,
		AdmissionResonanceGraftGateReadBackHash: gate.AdmissionResonanceGraftGateReadBackHash,
		AdmissionResonanceGraftPreflightID:      gate.AdmissionResonanceGraftPreflightID,
		AdmissionResonanceGraftBoundaryID:       gate.AdmissionResonanceGraftBoundaryID,
		AdmissionResonanceObservationID:         gate.AdmissionResonanceObservationID,
		AdmissionResonanceReceiverID:            gate.AdmissionResonanceReceiverID,
		AdmissionFinalGateID:                    gate.AdmissionFinalGateID,
		LedgerVerificationID:                    gate.LedgerVerificationID,
		GateState:                               gate.AdmissionResonanceGraftGateState,
		GateAction:                              gate.AdmissionResonanceGraftGateAction,
		GateTarget:                              gate.AdmissionResonanceGraftGateTarget,
		GateTargetKind:                          gate.AdmissionResonanceGraftGateTargetKind,
		GateTargetMode:                          gate.AdmissionResonanceGraftGateTargetMode,
		GateReceiptShape:                        gate.AdmissionResonanceGraftGateReceiptShape,
		GateDryRunOnly:                          gate.AdmissionResonanceGraftGateDryRunOnly,
		PreflightVerified:                       gate.AdmissionResonanceGraftGatePreflightVerified,
		AdmissionRequired:                       gate.AdmissionResonanceGraftGateAdmissionRequired,
		ShadowOnly:                              gate.AdmissionResonanceGraftGateShadowOnly,
		GraftAllowed:                            gate.AdmissionResonanceGraftGateGraftAllowed,
		RawTextAllowed:                          gate.AdmissionResonanceGraftGateRawDreamTextAllowed,
		JanusSurfaceAllowed:                     gate.AdmissionResonanceGraftGateJanusSurfaceAllowed,
		CoocLearningAllowed:                     gate.AdmissionResonanceGraftGateCoocLearningAllowed,
		DeltaHarvestAllowed:                     gate.AdmissionResonanceGraftGateDeltaHarvestAllowed,
		BodyMutationAllowed:                     gate.AdmissionResonanceGraftGateBodyMutationAllowed,
		RollbackRequired:                        gate.AdmissionResonanceGraftGateRollbackRequired,
		GateReady:                               gate.AdmissionResonanceGraftGateReady,
		ContractsReady:                          gate.ContractsReady,
		BodyTarget:                              gate.BodyTarget,
		WriteAllowed:                            gate.WriteAllowed,
		AdmissionAllowed:                        gate.AdmissionAllowed,
		LiveAdmissionEnabled:                    gate.LiveAdmissionEnabled,
		MutatesState:                            gate.MutatesState,
	})
	if h == "" {
		return ""
	}
	return "resonance-graft-gate-id-" + h
}

func recordAdmissionLiveRouteTurnCandidateAdmissionResonanceGraftGate(gate admissionLiveRouteTurnCandidateAdmissionResonanceGraftGate) error {
	path := strings.TrimSpace(os.Getenv("AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_RESONANCE_GRAFT_GATE_LOG"))
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

func admissionLiveRouteTurnCandidateAdmissionResonanceGraftCandidateForGate(gate admissionLiveRouteTurnCandidateAdmissionResonanceGraftGate) admissionLiveRouteTurnCandidateAdmissionResonanceGraftCandidate {
	sourceSchema := gate.Schema
	candidate := admissionLiveRouteTurnCandidateAdmissionResonanceGraftCandidate{
		admissionLiveRouteTurnCandidateAdmissionResonanceGraftGate: gate,
		AdmissionResonanceGraftCandidateState:                      "blocked",
		AdmissionResonanceGraftCandidateAction:                     "reject",
		AdmissionResonanceGraftCandidateDryRunOnly:                 true,
		SourceAdmissionResonanceGraftGateSchema:                    sourceSchema,
		SourceAdmissionResonanceGraftGatePassed:                    gate.Passed,
		SourceAdmissionResonanceGraftGateID:                        gate.AdmissionResonanceGraftGateID,
		SourceAdmissionResonanceGraftGateAction:                    gate.AdmissionResonanceGraftGateAction,
		SourceAdmissionResonanceGraftGateReady:                     gate.AdmissionResonanceGraftGateReady,
		SourceAdmissionResonanceGraftGateCausalID:                  gate.AdmissionResonanceGraftGateCausalID,
		SourceAdmissionResonanceGraftGateHash:                      gate.AdmissionResonanceGraftGateHash,
		SourceAdmissionResonanceGraftGateReadBackHash:              gate.AdmissionResonanceGraftGateReadBackHash,
		SourceAdmissionResonanceGraftPreflightIDForCandidate:       gate.AdmissionResonanceGraftPreflightID,
		SourceAdmissionResonanceGraftBoundaryIDForCandidate:        gate.AdmissionResonanceGraftBoundaryID,
		SourceAdmissionResonanceObservationIDForCandidate:          gate.AdmissionResonanceObservationID,
		SourceAdmissionResonanceReceiverIDForCandidate:             gate.AdmissionResonanceReceiverID,
		SourceAdmissionResonanceIntentIDForCandidate:               gate.AdmissionResonanceIntentID,
		SourceAdmissionFinalGateIDForCandidate:                     gate.AdmissionFinalGateID,
		SourceAdmissionSealIDForCandidate:                          gate.AdmissionSealID,
		SourceAdmissionPermitIDForCandidate:                        gate.AdmissionPermitID,
		SourceAdmissionReadinessIDForCandidate:                     gate.AdmissionReadinessID,
		SourceLedgerVerificationIDForCandidate:                     gate.LedgerVerificationID,
		SourceLedgerPersistenceIDForCandidate:                      gate.LedgerPersistenceID,
		SourceLedgerImplementationIDForCandidate:                   gate.LedgerImplementationID,
		SourceAdmissionLedgerIDForCandidate:                        gate.AdmissionLedgerID,
		SourceRollbackImplementationIDForCandidate:                 gate.RollbackImplementationID,
		SourceWriterReceiptIDForCandidate:                          gate.WriterReceiptID,
	}
	candidate.Schema = admissionLiveRouteTurnCandidateAdmissionResonanceGraftCandidateSchema
	candidate.Timing = "live_admission_resonance_graft_candidate"
	candidate.Passed = false
	candidate.AdmissionResonanceGraftCandidateID = ""
	candidate.AdmissionResonanceGraftCandidateReady = false
	candidate.LiveReady = false
	candidate.BodyTarget = "none"
	candidate.ContractsReady = false
	candidate.WriteAllowed = false
	candidate.AdmissionAllowed = false
	candidate.LiveAdmissionEnabled = false
	candidate.MutatesState = false

	if sourceSchema == "" {
		candidate.Reason = "missing_candidate_admission_resonance_graft_gate"
		return candidate
	}
	if sourceSchema != admissionLiveRouteTurnCandidateAdmissionResonanceGraftGateSchema {
		candidate.Reason = "unexpected_candidate_admission_resonance_graft_gate_schema " + sourceSchema
		return candidate
	}
	if !gate.Passed {
		candidate.Reason = "candidate_admission_resonance_graft_gate_failed"
		if gate.Reason != "" {
			candidate.Reason += ": " + gate.Reason
		}
		return candidate
	}
	if gate.AdmissionResonanceGraftGateID == "" {
		candidate.Reason = "missing_candidate_admission_resonance_graft_gate_id"
		return candidate
	}
	if wantGateID := admissionLiveRouteTurnCandidateAdmissionResonanceGraftGateID(gate); wantGateID == "" || gate.AdmissionResonanceGraftGateID != wantGateID {
		candidate.Reason = "candidate_admission_resonance_graft_gate_id_mismatch"
		return candidate
	}
	if gate.AdmissionResonanceGraftGateState != "shadow_graft_gate_ready_dry_run" ||
		gate.AdmissionResonanceGraftGateAction != "gate_resonance_shadow_graft_dry_run" ||
		gate.AdmissionResonanceGraftGateTarget != "resonance" ||
		gate.AdmissionResonanceGraftGateTargetKind != "internal_world_shadow_graft_gate" ||
		gate.AdmissionResonanceGraftGateTargetMode != "receipt_only_closed_gate_dry_run" ||
		gate.AdmissionResonanceGraftGateReceiptShape != "resonance_shadow_graft_gate_contract" {
		candidate.Reason = "candidate_admission_resonance_graft_gate_shape_mismatch"
		return candidate
	}
	if !gate.AdmissionResonanceGraftGateDryRunOnly ||
		!gate.AdmissionResonanceGraftGatePreflightVerified ||
		!gate.AdmissionResonanceGraftGateBoundaryVerified ||
		!gate.AdmissionResonanceGraftGateObservationVerified ||
		!gate.AdmissionResonanceGraftGateReceiverVerified ||
		!gate.AdmissionResonanceGraftGateIntentVerified ||
		!gate.AdmissionResonanceGraftGateFinalGateVerified ||
		!gate.AdmissionResonanceGraftGateSealVerified ||
		!gate.AdmissionResonanceGraftGatePermitVerified ||
		!gate.AdmissionResonanceGraftGateReadinessVerified ||
		!gate.AdmissionResonanceGraftGateLedgerVerified ||
		!gate.AdmissionResonanceGraftGateWriterReady ||
		!gate.AdmissionResonanceGraftGateRollbackReady ||
		!gate.AdmissionResonanceGraftGateLedgerReady ||
		!gate.AdmissionResonanceGraftGateReady {
		candidate.Reason = "candidate_admission_resonance_graft_gate_not_verified_dry_run"
		return candidate
	}
	if gate.AdmissionResonanceGraftGateKind != "shadow_graft_gate" ||
		gate.AdmissionResonanceGraftGateMode != "no_mutation_gate" ||
		gate.AdmissionResonanceGraftGateStage != "pre_live_graft_gate" ||
		gate.AdmissionResonanceGraftGateCausalID == "" ||
		gate.AdmissionResonanceGraftGateHash == "" ||
		gate.AdmissionResonanceGraftGateReadBackHash == "" {
		candidate.Reason = "candidate_admission_resonance_graft_gate_target_mismatch"
		return candidate
	}
	if wantCausalID := admissionLiveRouteTurnCandidateAdmissionResonanceGraftGateCausalID(gate); wantCausalID == "" || gate.AdmissionResonanceGraftGateCausalID != wantCausalID {
		candidate.Reason = "candidate_admission_resonance_graft_gate_causal_id_mismatch"
		return candidate
	}
	if wantHash := admissionLiveRouteTurnCandidateAdmissionResonanceGraftGateHash(gate); wantHash == "" || gate.AdmissionResonanceGraftGateHash != wantHash {
		candidate.Reason = "candidate_admission_resonance_graft_gate_hash_mismatch"
		return candidate
	}
	if wantReadBackHash := admissionLiveRouteTurnCandidateAdmissionResonanceGraftGateReadBackHash(gate); wantReadBackHash == "" || gate.AdmissionResonanceGraftGateReadBackHash != wantReadBackHash {
		candidate.Reason = "candidate_admission_resonance_graft_gate_read_back_hash_mismatch"
		return candidate
	}
	if gate.AdmissionResonanceGraftGateHash == gate.AdmissionResonanceGraftGateReadBackHash {
		candidate.Reason = "candidate_admission_resonance_graft_gate_read_back_missing"
		return candidate
	}
	if !gate.AdmissionResonanceGraftGateAdmissionRequired ||
		!gate.AdmissionResonanceGraftGateShadowOnly ||
		gate.AdmissionResonanceGraftGateGraftAllowed ||
		gate.AdmissionResonanceGraftGateRawDreamTextAllowed ||
		gate.AdmissionResonanceGraftGateJanusSurfaceAllowed ||
		gate.AdmissionResonanceGraftGateCoocLearningAllowed ||
		gate.AdmissionResonanceGraftGateDeltaHarvestAllowed ||
		gate.AdmissionResonanceGraftGateBodyMutationAllowed ||
		!gate.AdmissionResonanceGraftGateRollbackRequired {
		candidate.Reason = "candidate_admission_resonance_graft_gate_guard_mismatch"
		return candidate
	}
	if gate.ContractsReady || gate.WriteAllowed || gate.MutatesState || gate.LiveAdmissionEnabled || gate.AdmissionAllowed {
		candidate.Reason = "candidate_admission_resonance_graft_gate_already_open"
		return candidate
	}
	if gate.BodyTarget != "none" {
		candidate.Reason = "candidate_admission_resonance_graft_gate_body_target_mismatch"
		return candidate
	}
	if !gate.LiveReady {
		candidate.Reason = "candidate_admission_resonance_graft_gate_not_live_ready"
		return candidate
	}
	if gate.SourceAdmissionResonanceGraftPreflightSchema != admissionLiveRouteTurnCandidateAdmissionResonanceGraftPreflightSchema ||
		!gate.SourceAdmissionResonanceGraftPreflightPassed ||
		gate.SourceAdmissionResonanceGraftPreflightID != gate.AdmissionResonanceGraftPreflightID ||
		gate.SourceAdmissionResonanceGraftPreflightAction != "prepare_resonance_shadow_graft_preflight_dry_run" ||
		!gate.SourceAdmissionResonanceGraftPreflightReady ||
		gate.SourceAdmissionResonanceGraftPreflightCausalID != gate.AdmissionResonanceGraftPreflightCausalID ||
		gate.SourceAdmissionResonanceGraftPreflightHash != gate.AdmissionResonanceGraftPreflightHash ||
		gate.SourceAdmissionResonanceGraftPreflightReadBackHash != gate.AdmissionResonanceGraftPreflightReadBackHash {
		candidate.Reason = "candidate_admission_resonance_graft_gate_source_preflight_mismatch"
		return candidate
	}
	if gate.SourceAdmissionResonanceGraftBoundaryIDForGraftGate != gate.AdmissionResonanceGraftBoundaryID ||
		gate.SourceAdmissionResonanceObservationIDForGraftGate != gate.AdmissionResonanceObservationID ||
		gate.SourceAdmissionResonanceReceiverIDForGraftGate != gate.AdmissionResonanceReceiverID ||
		gate.SourceAdmissionResonanceIntentIDForGraftGate != gate.AdmissionResonanceIntentID ||
		gate.SourceAdmissionFinalGateIDForGraftGate != gate.AdmissionFinalGateID ||
		gate.SourceAdmissionSealIDForGraftGate != gate.AdmissionSealID ||
		gate.SourceAdmissionPermitIDForGraftGate != gate.AdmissionPermitID ||
		gate.SourceAdmissionReadinessIDForGraftGate != gate.AdmissionReadinessID ||
		gate.SourceLedgerVerificationIDForGraftGate != gate.LedgerVerificationID ||
		gate.SourceLedgerPersistenceIDForGraftGate != gate.LedgerPersistenceID ||
		gate.SourceLedgerImplementationIDForGraftGate != gate.LedgerImplementationID ||
		gate.SourceAdmissionLedgerIDForGraftGate != gate.AdmissionLedgerID ||
		gate.SourceRollbackImplementationIDForGraftGate != gate.RollbackImplementationID ||
		gate.SourceWriterReceiptIDForGraftGate != gate.WriterReceiptID {
		candidate.Reason = "candidate_admission_resonance_graft_gate_source_id_mismatch"
		return candidate
	}
	if gate.AdmissionResonanceGraftGateID == "" ||
		gate.AdmissionResonanceGraftPreflightID == "" ||
		gate.AdmissionResonanceGraftBoundaryID == "" ||
		gate.AdmissionResonanceObservationID == "" ||
		gate.AdmissionResonanceReceiverID == "" ||
		gate.AdmissionResonanceIntentID == "" ||
		gate.AdmissionFinalGateID == "" ||
		gate.AdmissionSealID == "" ||
		gate.AdmissionPermitID == "" ||
		gate.AdmissionReadinessID == "" ||
		gate.LedgerVerificationID == "" ||
		gate.LedgerPersistenceID == "" ||
		gate.LedgerImplementationID == "" ||
		gate.RollbackImplementationID == "" ||
		gate.WriterReceiptID == "" ||
		gate.WriterImplementationID == "" ||
		gate.AdmissionLedgerID == "" ||
		gate.AdmissionWriterContractID == "" ||
		gate.AdmissionWriterInventoryID == "" ||
		gate.AdmissionWriterPreflightID == "" ||
		gate.AdmissionLiveStageID == "" ||
		gate.AdmissionEnableGateID == "" ||
		gate.AdmissionSwitchID == "" ||
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
		candidate.Reason = "candidate_admission_resonance_graft_gate_missing_provenance"
		return candidate
	}

	candidate.AdmissionResonanceGraftCandidateState = "shadow_graft_candidate_ready_dry_run"
	candidate.AdmissionResonanceGraftCandidateAction = "draft_resonance_shadow_graft_candidate_dry_run"
	candidate.AdmissionResonanceGraftCandidateTarget = "resonance"
	candidate.AdmissionResonanceGraftCandidateTargetKind = "internal_world_shadow_graft_candidate"
	candidate.AdmissionResonanceGraftCandidateTargetMode = "receipt_only_closed_candidate_dry_run"
	candidate.AdmissionResonanceGraftCandidateReceiptShape = "resonance_shadow_graft_candidate_contract"
	candidate.AdmissionResonanceGraftCandidateDryRunOnly = true
	candidate.AdmissionResonanceGraftCandidateGateVerified = true
	candidate.AdmissionResonanceGraftCandidatePreflightVerified = gate.AdmissionResonanceGraftGatePreflightVerified
	candidate.AdmissionResonanceGraftCandidateBoundaryVerified = gate.AdmissionResonanceGraftGateBoundaryVerified
	candidate.AdmissionResonanceGraftCandidateObservationVerified = gate.AdmissionResonanceGraftGateObservationVerified
	candidate.AdmissionResonanceGraftCandidateReceiverVerified = gate.AdmissionResonanceGraftGateReceiverVerified
	candidate.AdmissionResonanceGraftCandidateIntentVerified = gate.AdmissionResonanceGraftGateIntentVerified
	candidate.AdmissionResonanceGraftCandidateFinalGateVerified = gate.AdmissionResonanceGraftGateFinalGateVerified
	candidate.AdmissionResonanceGraftCandidateSealVerified = gate.AdmissionResonanceGraftGateSealVerified
	candidate.AdmissionResonanceGraftCandidatePermitVerified = gate.AdmissionResonanceGraftGatePermitVerified
	candidate.AdmissionResonanceGraftCandidateReadinessVerified = gate.AdmissionResonanceGraftGateReadinessVerified
	candidate.AdmissionResonanceGraftCandidateLedgerVerified = gate.AdmissionResonanceGraftGateLedgerVerified
	candidate.AdmissionResonanceGraftCandidateWriterReady = gate.AdmissionResonanceGraftGateWriterReady
	candidate.AdmissionResonanceGraftCandidateRollbackReady = gate.AdmissionResonanceGraftGateRollbackReady
	candidate.AdmissionResonanceGraftCandidateLedgerReady = gate.AdmissionResonanceGraftGateLedgerReady
	candidate.AdmissionResonanceGraftCandidateKind = "shadow_graft_candidate"
	candidate.AdmissionResonanceGraftCandidateMode = "no_mutation_candidate"
	candidate.AdmissionResonanceGraftCandidateStage = "pre_live_graft_candidate"
	candidate.AdmissionResonanceGraftCandidateAdmissionRequired = true
	candidate.AdmissionResonanceGraftCandidateShadowOnly = true
	candidate.AdmissionResonanceGraftCandidateGraftAllowed = false
	candidate.AdmissionResonanceGraftCandidateRawDreamTextAllowed = false
	candidate.AdmissionResonanceGraftCandidateJanusSurfaceAllowed = false
	candidate.AdmissionResonanceGraftCandidateCoocLearningAllowed = false
	candidate.AdmissionResonanceGraftCandidateDeltaHarvestAllowed = false
	candidate.AdmissionResonanceGraftCandidateBodyMutationAllowed = false
	candidate.AdmissionResonanceGraftCandidateRollbackRequired = true
	candidate.AdmissionResonanceGraftCandidateReady = true
	candidate.LiveReady = true
	candidate.BodyTarget = "none"
	candidate.ContractsReady = false
	candidate.WriteAllowed = false
	candidate.AdmissionAllowed = false
	candidate.LiveAdmissionEnabled = false
	candidate.MutatesState = false
	candidate.AdmissionResonanceGraftCandidateCausalID = admissionLiveRouteTurnCandidateAdmissionResonanceGraftCandidateCausalID(candidate)
	if candidate.AdmissionResonanceGraftCandidateCausalID == "" {
		candidate.Reason = "missing_candidate_admission_resonance_graft_candidate_causal_id"
		return candidate
	}
	candidate.AdmissionResonanceGraftCandidateHash = admissionLiveRouteTurnCandidateAdmissionResonanceGraftCandidateHash(candidate)
	candidate.AdmissionResonanceGraftCandidateReadBackHash = admissionLiveRouteTurnCandidateAdmissionResonanceGraftCandidateReadBackHash(candidate)
	if candidate.AdmissionResonanceGraftCandidateHash == "" ||
		candidate.AdmissionResonanceGraftCandidateReadBackHash == "" ||
		candidate.AdmissionResonanceGraftCandidateHash == candidate.AdmissionResonanceGraftCandidateReadBackHash {
		candidate.Reason = "candidate_admission_resonance_graft_candidate_read_back_missing"
		return candidate
	}
	candidate.AdmissionResonanceGraftCandidateID = admissionLiveRouteTurnCandidateAdmissionResonanceGraftCandidateID(candidate)
	if candidate.AdmissionResonanceGraftCandidateID == "" {
		candidate.Reason = "missing_candidate_admission_resonance_graft_candidate_id"
		return candidate
	}
	candidate.Passed = true
	candidate.Reason = "resonance shadow graft candidate drafted without body mutation"
	return candidate
}

func admissionLiveRouteTurnCandidateAdmissionResonanceGraftCandidateCausalID(candidate admissionLiveRouteTurnCandidateAdmissionResonanceGraftCandidate) string {
	h := hashJSON(struct {
		AdmissionResonanceGraftGateID           string `json:"admission_resonance_graft_gate_id"`
		AdmissionResonanceGraftGateReadBackHash string `json:"admission_resonance_graft_gate_read_back_hash"`
		AdmissionResonanceGraftPreflightID      string `json:"admission_resonance_graft_preflight_id"`
		AdmissionResonanceGraftBoundaryID       string `json:"admission_resonance_graft_boundary_id"`
		AdmissionResonanceObservationID         string `json:"admission_resonance_observation_id"`
		AdmissionResonanceReceiverID            string `json:"admission_resonance_receiver_id"`
		CandidateRunID                          string `json:"candidate_run_id"`
		CandidateTextHash                       string `json:"candidate_text_hash"`
		TurnTextHash                            string `json:"turn_text_hash"`
		CandidateKind                           string `json:"candidate_kind"`
		CandidateStage                          string `json:"candidate_stage"`
	}{
		AdmissionResonanceGraftGateID:           candidate.AdmissionResonanceGraftGateID,
		AdmissionResonanceGraftGateReadBackHash: candidate.AdmissionResonanceGraftGateReadBackHash,
		AdmissionResonanceGraftPreflightID:      candidate.AdmissionResonanceGraftPreflightID,
		AdmissionResonanceGraftBoundaryID:       candidate.AdmissionResonanceGraftBoundaryID,
		AdmissionResonanceObservationID:         candidate.AdmissionResonanceObservationID,
		AdmissionResonanceReceiverID:            candidate.AdmissionResonanceReceiverID,
		CandidateRunID:                          candidate.CandidateRunID,
		CandidateTextHash:                       candidate.CandidateTextHash,
		TurnTextHash:                            candidate.TurnTextHash,
		CandidateKind:                           candidate.AdmissionResonanceGraftCandidateKind,
		CandidateStage:                          candidate.AdmissionResonanceGraftCandidateStage,
	})
	if h == "" {
		return ""
	}
	return "resonance-graft-candidate-causal-" + h
}

func admissionLiveRouteTurnCandidateAdmissionResonanceGraftCandidateHash(candidate admissionLiveRouteTurnCandidateAdmissionResonanceGraftCandidate) string {
	h := hashJSON(struct {
		AdmissionResonanceGraftCandidateCausalID string `json:"admission_resonance_graft_candidate_causal_id"`
		AdmissionResonanceGraftGateID            string `json:"admission_resonance_graft_gate_id"`
		GateHash                                 string `json:"gate_hash"`
		GateReadBackHash                         string `json:"gate_read_back_hash"`
		CandidateMode                            string `json:"candidate_mode"`
		ReceiptShape                             string `json:"receipt_shape"`
		AdmissionRequired                        bool   `json:"admission_required"`
		ShadowOnly                               bool   `json:"shadow_only"`
		GraftAllowed                             bool   `json:"graft_allowed"`
		DryRunOnly                               bool   `json:"dry_run_only"`
	}{
		AdmissionResonanceGraftCandidateCausalID: candidate.AdmissionResonanceGraftCandidateCausalID,
		AdmissionResonanceGraftGateID:            candidate.AdmissionResonanceGraftGateID,
		GateHash:                                 candidate.AdmissionResonanceGraftGateHash,
		GateReadBackHash:                         candidate.AdmissionResonanceGraftGateReadBackHash,
		CandidateMode:                            candidate.AdmissionResonanceGraftCandidateMode,
		ReceiptShape:                             candidate.AdmissionResonanceGraftCandidateReceiptShape,
		AdmissionRequired:                        candidate.AdmissionResonanceGraftCandidateAdmissionRequired,
		ShadowOnly:                               candidate.AdmissionResonanceGraftCandidateShadowOnly,
		GraftAllowed:                             candidate.AdmissionResonanceGraftCandidateGraftAllowed,
		DryRunOnly:                               candidate.AdmissionResonanceGraftCandidateDryRunOnly,
	})
	if h == "" {
		return ""
	}
	return "resonance-graft-candidate-" + h
}

func admissionLiveRouteTurnCandidateAdmissionResonanceGraftCandidateReadBackHash(candidate admissionLiveRouteTurnCandidateAdmissionResonanceGraftCandidate) string {
	h := hashJSON(struct {
		CandidateHash  string `json:"candidate_hash"`
		GateID         string `json:"gate_id"`
		CandidateKind  string `json:"candidate_kind"`
		CandidateReady bool   `json:"candidate_ready"`
		GraftAllowed   bool   `json:"graft_allowed"`
		BodyMutation   bool   `json:"body_mutation"`
		AdmissionOpen  bool   `json:"admission_open"`
	}{
		CandidateHash:  candidate.AdmissionResonanceGraftCandidateHash,
		GateID:         candidate.AdmissionResonanceGraftGateID,
		CandidateKind:  candidate.AdmissionResonanceGraftCandidateKind,
		CandidateReady: candidate.AdmissionResonanceGraftCandidateReady,
		GraftAllowed:   candidate.AdmissionResonanceGraftCandidateGraftAllowed,
		BodyMutation:   candidate.AdmissionResonanceGraftCandidateBodyMutationAllowed,
		AdmissionOpen:  candidate.LiveAdmissionEnabled,
	})
	if h == "" {
		return ""
	}
	return "resonance-graft-candidate-read-" + h
}

func admissionLiveRouteTurnCandidateAdmissionResonanceGraftCandidateID(candidate admissionLiveRouteTurnCandidateAdmissionResonanceGraftCandidate) string {
	h := hashJSON(struct {
		AdmissionResonanceGraftCandidateCausalID     string `json:"admission_resonance_graft_candidate_causal_id"`
		AdmissionResonanceGraftCandidateHash         string `json:"admission_resonance_graft_candidate_hash"`
		AdmissionResonanceGraftCandidateReadBackHash string `json:"admission_resonance_graft_candidate_read_back_hash"`
		AdmissionResonanceGraftGateID                string `json:"admission_resonance_graft_gate_id"`
		AdmissionResonanceGraftPreflightID           string `json:"admission_resonance_graft_preflight_id"`
		AdmissionResonanceGraftBoundaryID            string `json:"admission_resonance_graft_boundary_id"`
		AdmissionResonanceObservationID              string `json:"admission_resonance_observation_id"`
		AdmissionResonanceReceiverID                 string `json:"admission_resonance_receiver_id"`
		AdmissionFinalGateID                         string `json:"admission_final_gate_id"`
		LedgerVerificationID                         string `json:"ledger_verification_id"`
		CandidateState                               string `json:"candidate_state"`
		CandidateAction                              string `json:"candidate_action"`
		CandidateTarget                              string `json:"candidate_target"`
		CandidateTargetKind                          string `json:"candidate_target_kind"`
		CandidateTargetMode                          string `json:"candidate_target_mode"`
		CandidateReceiptShape                        string `json:"candidate_receipt_shape"`
		CandidateDryRunOnly                          bool   `json:"candidate_dry_run_only"`
		GateVerified                                 bool   `json:"gate_verified"`
		AdmissionRequired                            bool   `json:"admission_required"`
		ShadowOnly                                   bool   `json:"shadow_only"`
		GraftAllowed                                 bool   `json:"graft_allowed"`
		RawTextAllowed                               bool   `json:"raw_text_allowed"`
		JanusSurfaceAllowed                          bool   `json:"janus_surface_allowed"`
		CoocLearningAllowed                          bool   `json:"cooc_learning_allowed"`
		DeltaHarvestAllowed                          bool   `json:"delta_harvest_allowed"`
		BodyMutationAllowed                          bool   `json:"body_mutation_allowed"`
		RollbackRequired                             bool   `json:"rollback_required"`
		CandidateReady                               bool   `json:"candidate_ready"`
		ContractsReady                               bool   `json:"contracts_ready"`
		BodyTarget                                   string `json:"body_target"`
		WriteAllowed                                 bool   `json:"write_allowed"`
		AdmissionAllowed                             bool   `json:"admission_allowed"`
		LiveAdmissionEnabled                         bool   `json:"live_admission_enabled"`
		MutatesState                                 bool   `json:"mutates_state"`
	}{
		AdmissionResonanceGraftCandidateCausalID:     candidate.AdmissionResonanceGraftCandidateCausalID,
		AdmissionResonanceGraftCandidateHash:         candidate.AdmissionResonanceGraftCandidateHash,
		AdmissionResonanceGraftCandidateReadBackHash: candidate.AdmissionResonanceGraftCandidateReadBackHash,
		AdmissionResonanceGraftGateID:                candidate.AdmissionResonanceGraftGateID,
		AdmissionResonanceGraftPreflightID:           candidate.AdmissionResonanceGraftPreflightID,
		AdmissionResonanceGraftBoundaryID:            candidate.AdmissionResonanceGraftBoundaryID,
		AdmissionResonanceObservationID:              candidate.AdmissionResonanceObservationID,
		AdmissionResonanceReceiverID:                 candidate.AdmissionResonanceReceiverID,
		AdmissionFinalGateID:                         candidate.AdmissionFinalGateID,
		LedgerVerificationID:                         candidate.LedgerVerificationID,
		CandidateState:                               candidate.AdmissionResonanceGraftCandidateState,
		CandidateAction:                              candidate.AdmissionResonanceGraftCandidateAction,
		CandidateTarget:                              candidate.AdmissionResonanceGraftCandidateTarget,
		CandidateTargetKind:                          candidate.AdmissionResonanceGraftCandidateTargetKind,
		CandidateTargetMode:                          candidate.AdmissionResonanceGraftCandidateTargetMode,
		CandidateReceiptShape:                        candidate.AdmissionResonanceGraftCandidateReceiptShape,
		CandidateDryRunOnly:                          candidate.AdmissionResonanceGraftCandidateDryRunOnly,
		GateVerified:                                 candidate.AdmissionResonanceGraftCandidateGateVerified,
		AdmissionRequired:                            candidate.AdmissionResonanceGraftCandidateAdmissionRequired,
		ShadowOnly:                                   candidate.AdmissionResonanceGraftCandidateShadowOnly,
		GraftAllowed:                                 candidate.AdmissionResonanceGraftCandidateGraftAllowed,
		RawTextAllowed:                               candidate.AdmissionResonanceGraftCandidateRawDreamTextAllowed,
		JanusSurfaceAllowed:                          candidate.AdmissionResonanceGraftCandidateJanusSurfaceAllowed,
		CoocLearningAllowed:                          candidate.AdmissionResonanceGraftCandidateCoocLearningAllowed,
		DeltaHarvestAllowed:                          candidate.AdmissionResonanceGraftCandidateDeltaHarvestAllowed,
		BodyMutationAllowed:                          candidate.AdmissionResonanceGraftCandidateBodyMutationAllowed,
		RollbackRequired:                             candidate.AdmissionResonanceGraftCandidateRollbackRequired,
		CandidateReady:                               candidate.AdmissionResonanceGraftCandidateReady,
		ContractsReady:                               candidate.ContractsReady,
		BodyTarget:                                   candidate.BodyTarget,
		WriteAllowed:                                 candidate.WriteAllowed,
		AdmissionAllowed:                             candidate.AdmissionAllowed,
		LiveAdmissionEnabled:                         candidate.LiveAdmissionEnabled,
		MutatesState:                                 candidate.MutatesState,
	})
	if h == "" {
		return ""
	}
	return "resonance-graft-candidate-id-" + h
}

func recordAdmissionLiveRouteTurnCandidateAdmissionResonanceGraftCandidate(candidate admissionLiveRouteTurnCandidateAdmissionResonanceGraftCandidate) error {
	path := strings.TrimSpace(os.Getenv("AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_RESONANCE_GRAFT_CANDIDATE_LOG"))
	if path == "" {
		return nil
	}
	f, err := os.OpenFile(path, os.O_CREATE|os.O_APPEND|os.O_WRONLY, 0600)
	if err != nil {
		return err
	}
	enc := json.NewEncoder(f)
	err = enc.Encode(candidate)
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
