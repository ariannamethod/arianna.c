package main

import (
	"encoding/json"
	"fmt"
	"os"
	"strings"
)

const admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionLiveStageSchema = "arianna.live_route_weighted_admission_resonance_graft_admission_live_stage.v1"

type admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionLiveStageReport struct {
	Schema                                                               string `json:"schema"`
	Status                                                               string `json:"status"`
	Target                                                               string `json:"target"`
	TargetKind                                                           string `json:"target_kind"`
	TargetMode                                                           string `json:"target_mode"`
	Action                                                               string `json:"action"`
	StageState                                                           string `json:"stage_state"`
	StageAction                                                          string `json:"stage_action"`
	EnableState                                                          string `json:"enable_state"`
	EnableAction                                                         string `json:"enable_action"`
	SwitchState                                                          string `json:"switch_state"`
	SwitchAction                                                         string `json:"switch_action"`
	Promotion                                                            string `json:"promotion"`
	WeightedAdmissionResonanceGraftAdmissionLiveStageReady               bool   `json:"weighted_admission_resonance_graft_admission_live_stage_ready"`
	WeightedAdmissionResonanceGraftAdmissionEnableGateConsumed           bool   `json:"weighted_admission_resonance_graft_admission_enable_gate_consumed"`
	WeightedAdmissionResonanceGraftAdmissionEnableGateRequired           bool   `json:"weighted_admission_resonance_graft_admission_enable_gate_required"`
	NextStepBlockedWithoutResonanceGraftAdmissionLiveStage               bool   `json:"next_step_blocked_without_resonance_graft_admission_live_stage"`
	WeightedAdmissionResonanceGraftAdmissionLiveStageID                  string `json:"weighted_admission_resonance_graft_admission_live_stage_id"`
	ReceiptShape                                                         string `json:"receipt_shape"`
	LiveStageKind                                                        string `json:"live_stage_kind"`
	LiveStageMode                                                        string `json:"live_stage_mode"`
	LiveStageStage                                                       string `json:"live_stage_stage"`
	CausalID                                                             string `json:"causal_id"`
	LiveStageHash                                                        string `json:"live_stage_hash"`
	ReadBackHash                                                         string `json:"read_back_hash"`
	EnableGateVerified                                                   bool   `json:"enable_gate_verified"`
	EnableGateHashVerified                                               bool   `json:"enable_gate_hash_verified"`
	EnableGateReadBackVerified                                           bool   `json:"enable_gate_read_back_verified"`
	SwitchVerified                                                       bool   `json:"switch_verified"`
	SwitchHashVerified                                                   bool   `json:"switch_hash_verified"`
	SwitchReadBackVerified                                               bool   `json:"switch_read_back_verified"`
	PromotionVerified                                                    bool   `json:"promotion_verified"`
	PromotionHashVerified                                                bool   `json:"promotion_hash_verified"`
	PromotionReadBackVerified                                            bool   `json:"promotion_read_back_verified"`
	DecisionVerified                                                     bool   `json:"decision_verified"`
	DecisionHashVerified                                                 bool   `json:"decision_hash_verified"`
	DecisionReadBackVerified                                             bool   `json:"decision_read_back_verified"`
	ProofPreconditionVerified                                            bool   `json:"proof_precondition_verified"`
	PreconditionHashVerified                                             bool   `json:"precondition_hash_verified"`
	PreconditionReadBackVerified                                         bool   `json:"precondition_read_back_verified"`
	ProofVerified                                                        bool   `json:"proof_verified"`
	ProofHashVerified                                                    bool   `json:"proof_hash_verified"`
	ProofReadBackVerified                                                bool   `json:"proof_read_back_verified"`
	StoreReaderVerified                                                  bool   `json:"store_reader_verified"`
	StoreVerified                                                        bool   `json:"store_verified"`
	CandidateVerified                                                    bool   `json:"candidate_verified"`
	GateVerified                                                         bool   `json:"gate_verified"`
	PreflightVerified                                                    bool   `json:"preflight_verified"`
	BoundaryVerified                                                     bool   `json:"boundary_verified"`
	ObservationVerified                                                  bool   `json:"observation_verified"`
	ReceiverVerified                                                     bool   `json:"receiver_verified"`
	IntentVerified                                                       bool   `json:"intent_verified"`
	FinalGateVerified                                                    bool   `json:"final_gate_verified"`
	SealVerified                                                         bool   `json:"seal_verified"`
	PermitVerified                                                       bool   `json:"permit_verified"`
	AuthorityVerified                                                    bool   `json:"authority_verified"`
	AdmissionRequired                                                    bool   `json:"admission_required"`
	ShadowOnly                                                           bool   `json:"shadow_only"`
	GraftAllowed                                                         bool   `json:"graft_allowed"`
	DryRunOnly                                                           bool   `json:"dry_run_only"`
	LiveReady                                                            bool   `json:"live_ready"`
	RawDreamTextAllowed                                                  bool   `json:"raw_dream_text_allowed"`
	RawDreamTextObserved                                                 bool   `json:"raw_dream_text_observed"`
	RawDreamTextForwarded                                                bool   `json:"raw_dream_text_forwarded"`
	JanusSurfaceAllowed                                                  bool   `json:"janus_surface_allowed"`
	CoocLearningAllowed                                                  bool   `json:"cooc_learning_allowed"`
	DeltaHarvestAllowed                                                  bool   `json:"delta_harvest_allowed"`
	BodyMutationAllowed                                                  bool   `json:"body_mutation_allowed"`
	RequiresWriter                                                       bool   `json:"requires_writer"`
	WriterReady                                                          bool   `json:"writer_ready"`
	RollbackRequired                                                     bool   `json:"rollback_required"`
	RequiresRollback                                                     bool   `json:"requires_rollback"`
	RollbackReady                                                        bool   `json:"rollback_ready"`
	ReadOnly                                                             bool   `json:"read_only"`
	ReplayOnly                                                           bool   `json:"replay_only"`
	SourceSchema                                                         string `json:"source_schema"`
	SourceStatus                                                         string `json:"source_status"`
	SourceTarget                                                         string `json:"source_target"`
	SourceReport                                                         string `json:"source_report"`
	SourceWeightedAdmissionResonanceGraftAdmissionEnableGateID           string `json:"source_weighted_admission_resonance_graft_admission_enable_gate_id"`
	SourceWeightedAdmissionResonanceGraftAdmissionEnableGateReady        bool   `json:"source_weighted_admission_resonance_graft_admission_enable_gate_ready"`
	SourceWeightedAdmissionResonanceGraftAdmissionEnableGateCausalID     string `json:"source_weighted_admission_resonance_graft_admission_enable_gate_causal_id"`
	SourceWeightedAdmissionResonanceGraftAdmissionEnableGateHash         string `json:"source_weighted_admission_resonance_graft_admission_enable_gate_hash"`
	SourceWeightedAdmissionResonanceGraftAdmissionEnableGateReadBack     string `json:"source_weighted_admission_resonance_graft_admission_enable_gate_read_back_hash"`
	SourceEnableState                                                    string `json:"source_enable_state"`
	SourceEnableAction                                                   string `json:"source_enable_action"`
	SourceEnableGateReceiptShape                                         string `json:"source_enable_gate_receipt_shape"`
	SourceEnableGateKind                                                 string `json:"source_enable_gate_kind"`
	SourceEnableGateMode                                                 string `json:"source_enable_gate_mode"`
	SourceEnableGateStage                                                string `json:"source_enable_gate_stage"`
	SourceEnableGateAdmissionRequired                                    bool   `json:"source_enable_gate_admission_required"`
	SourceEnableGateShadowOnly                                           bool   `json:"source_enable_gate_shadow_only"`
	SourceEnableGateGraftAllowed                                         bool   `json:"source_enable_gate_graft_allowed"`
	SourceEnableGateDryRunOnly                                           bool   `json:"source_enable_gate_dry_run_only"`
	SourceEnableGateLiveReady                                            bool   `json:"source_enable_gate_live_ready"`
	SourceEnableGateRawDreamTextAllowed                                  bool   `json:"source_enable_gate_raw_dream_text_allowed"`
	SourceEnableGateRawDreamTextObserved                                 bool   `json:"source_enable_gate_raw_dream_text_observed"`
	SourceEnableGateRawDreamTextForwarded                                bool   `json:"source_enable_gate_raw_dream_text_forwarded"`
	SourceEnableGateJanusSurfaceAllowed                                  bool   `json:"source_enable_gate_janus_surface_allowed"`
	SourceEnableGateCoocLearningAllowed                                  bool   `json:"source_enable_gate_cooc_learning_allowed"`
	SourceEnableGateDeltaHarvestAllowed                                  bool   `json:"source_enable_gate_delta_harvest_allowed"`
	SourceEnableGateBodyMutationAllowed                                  bool   `json:"source_enable_gate_body_mutation_allowed"`
	SourceEnableGateRollbackRequired                                     bool   `json:"source_enable_gate_rollback_required"`
	SourceEnableGateReadOnly                                             bool   `json:"source_enable_gate_read_only"`
	SourceEnableGateReplayOnly                                           bool   `json:"source_enable_gate_replay_only"`
	SourceEnableGateWriteAllowed                                         bool   `json:"source_enable_gate_write_allowed"`
	SourceEnableGateAdmissionAllowed                                     bool   `json:"source_enable_gate_admission_allowed"`
	SourceEnableGateLiveAdmissionEnabled                                 bool   `json:"source_enable_gate_live_admission_enabled"`
	SourceEnableGateMutatesState                                         bool   `json:"source_enable_gate_mutates_state"`
	SourceEnableGateBodyTarget                                           string `json:"source_enable_gate_body_target"`
	SourceEnableGatePassed                                               bool   `json:"source_enable_gate_passed"`
	SourceEnableGateReason                                               string `json:"source_enable_gate_reason"`
	SourceWeightedAdmissionResonanceGraftAdmissionSwitchID               string `json:"source_weighted_admission_resonance_graft_admission_switch_id"`
	SourceWeightedAdmissionResonanceGraftAdmissionSwitchReady            bool   `json:"source_weighted_admission_resonance_graft_admission_switch_ready"`
	SourceWeightedAdmissionResonanceGraftAdmissionSwitchCausalID         string `json:"source_weighted_admission_resonance_graft_admission_switch_causal_id"`
	SourceWeightedAdmissionResonanceGraftAdmissionSwitchHash             string `json:"source_weighted_admission_resonance_graft_admission_switch_hash"`
	SourceWeightedAdmissionResonanceGraftAdmissionSwitchReadBack         string `json:"source_weighted_admission_resonance_graft_admission_switch_read_back_hash"`
	SourceSwitchState                                                    string `json:"source_switch_state"`
	SourceSwitchAction                                                   string `json:"source_switch_action"`
	SourceSwitchReceiptShape                                             string `json:"source_switch_receipt_shape"`
	SourceSwitchKind                                                     string `json:"source_switch_kind"`
	SourceSwitchMode                                                     string `json:"source_switch_mode"`
	SourceSwitchStage                                                    string `json:"source_switch_stage"`
	SourceSwitchAdmissionRequired                                        bool   `json:"source_switch_admission_required"`
	SourceSwitchShadowOnly                                               bool   `json:"source_switch_shadow_only"`
	SourceSwitchGraftAllowed                                             bool   `json:"source_switch_graft_allowed"`
	SourceSwitchDryRunOnly                                               bool   `json:"source_switch_dry_run_only"`
	SourceSwitchLiveReady                                                bool   `json:"source_switch_live_ready"`
	SourceSwitchRawDreamTextAllowed                                      bool   `json:"source_switch_raw_dream_text_allowed"`
	SourceSwitchRawDreamTextObserved                                     bool   `json:"source_switch_raw_dream_text_observed"`
	SourceSwitchRawDreamTextForwarded                                    bool   `json:"source_switch_raw_dream_text_forwarded"`
	SourceSwitchJanusSurfaceAllowed                                      bool   `json:"source_switch_janus_surface_allowed"`
	SourceSwitchCoocLearningAllowed                                      bool   `json:"source_switch_cooc_learning_allowed"`
	SourceSwitchDeltaHarvestAllowed                                      bool   `json:"source_switch_delta_harvest_allowed"`
	SourceSwitchBodyMutationAllowed                                      bool   `json:"source_switch_body_mutation_allowed"`
	SourceSwitchRollbackRequired                                         bool   `json:"source_switch_rollback_required"`
	SourceSwitchReadOnly                                                 bool   `json:"source_switch_read_only"`
	SourceSwitchReplayOnly                                               bool   `json:"source_switch_replay_only"`
	SourceSwitchWriteAllowed                                             bool   `json:"source_switch_write_allowed"`
	SourceSwitchAdmissionAllowed                                         bool   `json:"source_switch_admission_allowed"`
	SourceSwitchLiveAdmissionEnabled                                     bool   `json:"source_switch_live_admission_enabled"`
	SourceSwitchMutatesState                                             bool   `json:"source_switch_mutates_state"`
	SourceSwitchBodyTarget                                               string `json:"source_switch_body_target"`
	SourceSwitchPassed                                                   bool   `json:"source_switch_passed"`
	SourceSwitchReason                                                   string `json:"source_switch_reason"`
	SourceWeightedAdmissionResonanceGraftAdmissionPromotionID            string `json:"source_weighted_admission_resonance_graft_admission_promotion_id"`
	SourceWeightedAdmissionResonanceGraftAdmissionPromotionReady         bool   `json:"source_weighted_admission_resonance_graft_admission_promotion_ready"`
	SourceWeightedAdmissionResonanceGraftAdmissionPromotionCausalID      string `json:"source_weighted_admission_resonance_graft_admission_promotion_causal_id"`
	SourceWeightedAdmissionResonanceGraftAdmissionPromotionHash          string `json:"source_weighted_admission_resonance_graft_admission_promotion_hash"`
	SourceWeightedAdmissionResonanceGraftAdmissionPromotionReadBack      string `json:"source_weighted_admission_resonance_graft_admission_promotion_read_back_hash"`
	SourcePromotion                                                      string `json:"source_promotion"`
	SourcePromotionAction                                                string `json:"source_promotion_action"`
	SourcePromotionReceiptShape                                          string `json:"source_promotion_receipt_shape"`
	SourcePromotionKind                                                  string `json:"source_promotion_kind"`
	SourcePromotionMode                                                  string `json:"source_promotion_mode"`
	SourcePromotionStage                                                 string `json:"source_promotion_stage"`
	SourcePromotionAdmissionRequired                                     bool   `json:"source_promotion_admission_required"`
	SourcePromotionShadowOnly                                            bool   `json:"source_promotion_shadow_only"`
	SourcePromotionGraftAllowed                                          bool   `json:"source_promotion_graft_allowed"`
	SourcePromotionDryRunOnly                                            bool   `json:"source_promotion_dry_run_only"`
	SourcePromotionLiveReady                                             bool   `json:"source_promotion_live_ready"`
	SourcePromotionRawDreamTextAllowed                                   bool   `json:"source_promotion_raw_dream_text_allowed"`
	SourcePromotionRawDreamTextObserved                                  bool   `json:"source_promotion_raw_dream_text_observed"`
	SourcePromotionRawDreamTextForwarded                                 bool   `json:"source_promotion_raw_dream_text_forwarded"`
	SourcePromotionJanusSurfaceAllowed                                   bool   `json:"source_promotion_janus_surface_allowed"`
	SourcePromotionCoocLearningAllowed                                   bool   `json:"source_promotion_cooc_learning_allowed"`
	SourcePromotionDeltaHarvestAllowed                                   bool   `json:"source_promotion_delta_harvest_allowed"`
	SourcePromotionBodyMutationAllowed                                   bool   `json:"source_promotion_body_mutation_allowed"`
	SourcePromotionRollbackRequired                                      bool   `json:"source_promotion_rollback_required"`
	SourcePromotionReadOnly                                              bool   `json:"source_promotion_read_only"`
	SourcePromotionReplayOnly                                            bool   `json:"source_promotion_replay_only"`
	SourcePromotionWriteAllowed                                          bool   `json:"source_promotion_write_allowed"`
	SourcePromotionAdmissionAllowed                                      bool   `json:"source_promotion_admission_allowed"`
	SourcePromotionLiveAdmissionEnabled                                  bool   `json:"source_promotion_live_admission_enabled"`
	SourcePromotionMutatesState                                          bool   `json:"source_promotion_mutates_state"`
	SourcePromotionBodyTarget                                            string `json:"source_promotion_body_target"`
	SourcePromotionPassed                                                bool   `json:"source_promotion_passed"`
	SourcePromotionReason                                                string `json:"source_promotion_reason"`
	SourceWeightedAdmissionResonanceGraftAdmissionDecisionID             string `json:"source_weighted_admission_resonance_graft_admission_decision_id"`
	SourceWeightedAdmissionResonanceGraftAdmissionDecisionReady          bool   `json:"source_weighted_admission_resonance_graft_admission_decision_ready"`
	SourceWeightedAdmissionResonanceGraftAdmissionProofPreconditionID    string `json:"source_weighted_admission_resonance_graft_admission_proof_precondition_id"`
	SourceWeightedAdmissionResonanceGraftAdmissionProofPreconditionReady bool   `json:"source_weighted_admission_resonance_graft_admission_proof_precondition_ready"`
	SourceWeightedAdmissionResonanceGraftAdmissionProofID                string `json:"source_weighted_admission_resonance_graft_admission_proof_id"`
	SourceWeightedAdmissionResonanceGraftAdmissionProofReady             bool   `json:"source_weighted_admission_resonance_graft_admission_proof_ready"`
	SourceWeightedAdmissionResonanceGraftCandidateStoreReaderID          string `json:"source_weighted_admission_resonance_graft_candidate_store_reader_id"`
	SourceWeightedAdmissionResonanceGraftCandidateStoreReaderReady       bool   `json:"source_weighted_admission_resonance_graft_candidate_store_reader_ready"`
	SourceWeightedAdmissionResonanceGraftCandidateStoreID                string `json:"source_weighted_admission_resonance_graft_candidate_store_id"`
	SourceWeightedAdmissionResonanceGraftCandidateStoreReady             bool   `json:"source_weighted_admission_resonance_graft_candidate_store_ready"`
	SourceWeightedAdmissionResonanceGraftCandidateID                     string `json:"source_weighted_admission_resonance_graft_candidate_id"`
	SourceWeightedAdmissionResonanceGraftCandidateReady                  bool   `json:"source_weighted_admission_resonance_graft_candidate_ready"`
	SourceWeightedAdmissionResonanceGraftGateID                          string `json:"source_weighted_admission_resonance_graft_gate_id"`
	SourceWeightedAdmissionResonanceGraftGateReady                       bool   `json:"source_weighted_admission_resonance_graft_gate_ready"`
	SourceWeightedAdmissionResonanceGraftPreflightID                     string `json:"source_weighted_admission_resonance_graft_preflight_id"`
	SourceWeightedAdmissionResonanceGraftPreflightReady                  bool   `json:"source_weighted_admission_resonance_graft_preflight_ready"`
	SourceWeightedAdmissionResonanceGraftBoundaryID                      string `json:"source_weighted_admission_resonance_graft_boundary_id"`
	SourceWeightedAdmissionResonanceGraftBoundaryReady                   bool   `json:"source_weighted_admission_resonance_graft_boundary_ready"`
	SourceWeightedAdmissionResonanceObservationID                        string `json:"source_weighted_admission_resonance_observation_id"`
	SourceWeightedAdmissionResonanceObservationReady                     bool   `json:"source_weighted_admission_resonance_observation_ready"`
	SourceWeightedAdmissionResonanceReceiverID                           string `json:"source_weighted_admission_resonance_receiver_id"`
	SourceWeightedAdmissionResonanceReceiverReady                        bool   `json:"source_weighted_admission_resonance_receiver_ready"`
	SourceWeightedAdmissionResonanceIntentReady                          bool   `json:"source_weighted_admission_resonance_intent_ready"`
	SourceWeightedAdmissionFinalGateReady                                bool   `json:"source_weighted_admission_final_gate_ready"`
	SourceWeightedAdmissionSealReady                                     bool   `json:"source_weighted_admission_seal_ready"`
	SourceWeightedAdmissionPermitReady                                   bool   `json:"source_weighted_admission_permit_ready"`
	SourceWeightedAdmissionAuthorityConsumed                             bool   `json:"source_weighted_admission_authority_consumed"`
	SourceWeightedAdmissionAuthorityRequired                             bool   `json:"source_weighted_admission_authority_required"`
	BodySmokeWeighted                                                    bool   `json:"body_smoke_weighted"`
	NanoDirectRunner                                                     bool   `json:"nano_direct_runner"`
	NanoDirectFinalGate                                                  bool   `json:"nano_direct_final_gate"`
	ResonanceGraftAdmissionProof                                         bool   `json:"resonance_graft_admission_proof"`
	BoundaryReportFullChain                                              bool   `json:"boundary_report_full_chain"`
	SourceAuthorityGranted                                               bool   `json:"source_authority_granted"`
	AuthorityGranted                                                     bool   `json:"authority_granted"`
	ContractsReady                                                       bool   `json:"contracts_ready"`
	WriteAllowed                                                         bool   `json:"write_allowed"`
	AdmissionAllowed                                                     bool   `json:"admission_allowed"`
	LiveAdmissionEnabled                                                 bool   `json:"live_admission_enabled"`
	MutatesState                                                         bool   `json:"mutates_state"`
	BodyTarget                                                           string `json:"body_target"`
	Passed                                                               bool   `json:"passed"`
	Reason                                                               string `json:"reason"`
}

func runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionLiveStage(args []string) error {
	if len(args) != 2 {
		return fmt.Errorf("usage: --admission-live-route-weighted-admission-resonance-graft-admission-live-stage RESONANCE_GRAFT_ADMISSION_ENABLE_GATE_REPORT RESONANCE_GRAFT_ADMISSION_LIVE_STAGE_REPORT")
	}
	enableGatePath := args[0]
	outputPath := args[1]
	if strings.TrimSpace(outputPath) == "" {
		return fmt.Errorf("weighted admission resonance graft admission live stage output path missing")
	}
	sourceGate, root, err := readAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionEnableGateReportForAssert(enableGatePath)
	if err != nil {
		return err
	}
	if err := admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionEnableGateReportError(sourceGate, root); err != nil {
		return err
	}
	stage := admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionLiveStageReport{
		Schema:         admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionLiveStageSchema,
		Status:         "shadow_graft_admission_live_stage_blocked_dry_run",
		Target:         "live_route_admission_next_step",
		TargetKind:     "weighted_internal_world_shadow_graft_admission_live_stage",
		TargetMode:     "closed_live_stage_guard_dry_run",
		Action:         "block_weighted_resonance_shadow_graft_admission_enable_gate_disabled_dry_run",
		StageState:     "blocked",
		StageAction:    "reject_disabled_enable_gate",
		EnableState:    sourceGate.EnableState,
		EnableAction:   sourceGate.EnableAction,
		SwitchState:    sourceGate.SwitchState,
		SwitchAction:   sourceGate.SwitchAction,
		Promotion:      sourceGate.Promotion,
		ReceiptShape:   "weighted_resonance_shadow_graft_admission_live_stage_receipt",
		LiveStageKind:  "shadow_graft_admission_live_stage",
		LiveStageMode:  "closed_enable_gate_live_stage_guard",
		LiveStageStage: "pre_writer_graft_admission_live_stage",
		WeightedAdmissionResonanceGraftAdmissionLiveStageReady:     true,
		WeightedAdmissionResonanceGraftAdmissionEnableGateConsumed: true,
		WeightedAdmissionResonanceGraftAdmissionEnableGateRequired: true,
		NextStepBlockedWithoutResonanceGraftAdmissionLiveStage:     true,
		EnableGateVerified:           true,
		EnableGateHashVerified:       true,
		EnableGateReadBackVerified:   true,
		SwitchVerified:               sourceGate.SwitchVerified,
		SwitchHashVerified:           sourceGate.SwitchHashVerified,
		SwitchReadBackVerified:       sourceGate.SwitchReadBackVerified,
		PromotionVerified:            sourceGate.PromotionVerified,
		PromotionHashVerified:        sourceGate.PromotionHashVerified,
		PromotionReadBackVerified:    sourceGate.PromotionReadBackVerified,
		DecisionVerified:             sourceGate.DecisionVerified,
		DecisionHashVerified:         sourceGate.DecisionHashVerified,
		DecisionReadBackVerified:     sourceGate.DecisionReadBackVerified,
		ProofPreconditionVerified:    sourceGate.ProofPreconditionVerified,
		PreconditionHashVerified:     sourceGate.PreconditionHashVerified,
		PreconditionReadBackVerified: sourceGate.PreconditionReadBackVerified,
		ProofVerified:                sourceGate.ProofVerified,
		ProofHashVerified:            sourceGate.ProofHashVerified,
		ProofReadBackVerified:        sourceGate.ProofReadBackVerified,
		StoreReaderVerified:          sourceGate.StoreReaderVerified,
		StoreVerified:                sourceGate.StoreVerified,
		CandidateVerified:            sourceGate.CandidateVerified,
		GateVerified:                 sourceGate.GateVerified,
		PreflightVerified:            sourceGate.PreflightVerified,
		BoundaryVerified:             sourceGate.BoundaryVerified,
		ObservationVerified:          sourceGate.ObservationVerified,
		ReceiverVerified:             sourceGate.ReceiverVerified,
		IntentVerified:               sourceGate.IntentVerified,
		FinalGateVerified:            sourceGate.FinalGateVerified,
		SealVerified:                 sourceGate.SealVerified,
		PermitVerified:               sourceGate.PermitVerified,
		AuthorityVerified:            sourceGate.AuthorityVerified,
		AdmissionRequired:            true,
		ShadowOnly:                   true,
		GraftAllowed:                 false,
		DryRunOnly:                   true,
		LiveReady:                    true,
		RawDreamTextAllowed:          false,
		RawDreamTextObserved:         false,
		RawDreamTextForwarded:        false,
		JanusSurfaceAllowed:          false,
		CoocLearningAllowed:          false,
		DeltaHarvestAllowed:          false,
		BodyMutationAllowed:          false,
		RequiresWriter:               true,
		WriterReady:                  false,
		RollbackRequired:             true,
		RequiresRollback:             true,
		RollbackReady:                false,
		ReadOnly:                     true,
		ReplayOnly:                   true,
		SourceSchema:                 sourceGate.Schema,
		SourceStatus:                 sourceGate.Status,
		SourceTarget:                 sourceGate.Target,
		SourceReport:                 enableGatePath,
		SourceWeightedAdmissionResonanceGraftAdmissionEnableGateID:       sourceGate.WeightedAdmissionResonanceGraftAdmissionEnableGateID,
		SourceWeightedAdmissionResonanceGraftAdmissionEnableGateReady:    sourceGate.WeightedAdmissionResonanceGraftAdmissionEnableGateReady,
		SourceWeightedAdmissionResonanceGraftAdmissionEnableGateCausalID: sourceGate.CausalID,
		SourceWeightedAdmissionResonanceGraftAdmissionEnableGateHash:     sourceGate.EnableGateHash,
		SourceWeightedAdmissionResonanceGraftAdmissionEnableGateReadBack: sourceGate.ReadBackHash,
		SourceEnableState:                                                    sourceGate.EnableState,
		SourceEnableAction:                                                   sourceGate.EnableAction,
		SourceEnableGateReceiptShape:                                         sourceGate.ReceiptShape,
		SourceEnableGateKind:                                                 sourceGate.EnableGateKind,
		SourceEnableGateMode:                                                 sourceGate.EnableGateMode,
		SourceEnableGateStage:                                                sourceGate.EnableGateStage,
		SourceEnableGateAdmissionRequired:                                    sourceGate.AdmissionRequired,
		SourceEnableGateShadowOnly:                                           sourceGate.ShadowOnly,
		SourceEnableGateGraftAllowed:                                         sourceGate.GraftAllowed,
		SourceEnableGateDryRunOnly:                                           sourceGate.DryRunOnly,
		SourceEnableGateLiveReady:                                            sourceGate.LiveReady,
		SourceEnableGateRawDreamTextAllowed:                                  sourceGate.RawDreamTextAllowed,
		SourceEnableGateRawDreamTextObserved:                                 sourceGate.RawDreamTextObserved,
		SourceEnableGateRawDreamTextForwarded:                                sourceGate.RawDreamTextForwarded,
		SourceEnableGateJanusSurfaceAllowed:                                  sourceGate.JanusSurfaceAllowed,
		SourceEnableGateCoocLearningAllowed:                                  sourceGate.CoocLearningAllowed,
		SourceEnableGateDeltaHarvestAllowed:                                  sourceGate.DeltaHarvestAllowed,
		SourceEnableGateBodyMutationAllowed:                                  sourceGate.BodyMutationAllowed,
		SourceEnableGateRollbackRequired:                                     sourceGate.RollbackRequired,
		SourceEnableGateReadOnly:                                             sourceGate.ReadOnly,
		SourceEnableGateReplayOnly:                                           sourceGate.ReplayOnly,
		SourceEnableGateWriteAllowed:                                         sourceGate.WriteAllowed,
		SourceEnableGateAdmissionAllowed:                                     sourceGate.AdmissionAllowed,
		SourceEnableGateLiveAdmissionEnabled:                                 sourceGate.LiveAdmissionEnabled,
		SourceEnableGateMutatesState:                                         sourceGate.MutatesState,
		SourceEnableGateBodyTarget:                                           sourceGate.BodyTarget,
		SourceEnableGatePassed:                                               sourceGate.Passed,
		SourceEnableGateReason:                                               sourceGate.Reason,
		SourceWeightedAdmissionResonanceGraftAdmissionSwitchID:               sourceGate.SourceWeightedAdmissionResonanceGraftAdmissionSwitchID,
		SourceWeightedAdmissionResonanceGraftAdmissionSwitchReady:            sourceGate.SourceWeightedAdmissionResonanceGraftAdmissionSwitchReady,
		SourceWeightedAdmissionResonanceGraftAdmissionSwitchCausalID:         sourceGate.SourceWeightedAdmissionResonanceGraftAdmissionSwitchCausalID,
		SourceWeightedAdmissionResonanceGraftAdmissionSwitchHash:             sourceGate.SourceWeightedAdmissionResonanceGraftAdmissionSwitchHash,
		SourceWeightedAdmissionResonanceGraftAdmissionSwitchReadBack:         sourceGate.SourceWeightedAdmissionResonanceGraftAdmissionSwitchReadBack,
		SourceSwitchState:                                                    sourceGate.SourceSwitchState,
		SourceSwitchAction:                                                   sourceGate.SourceSwitchAction,
		SourceSwitchReceiptShape:                                             sourceGate.SourceSwitchReceiptShape,
		SourceSwitchKind:                                                     sourceGate.SourceSwitchKind,
		SourceSwitchMode:                                                     sourceGate.SourceSwitchMode,
		SourceSwitchStage:                                                    sourceGate.SourceSwitchStage,
		SourceSwitchAdmissionRequired:                                        sourceGate.SourceSwitchAdmissionRequired,
		SourceSwitchShadowOnly:                                               sourceGate.SourceSwitchShadowOnly,
		SourceSwitchGraftAllowed:                                             sourceGate.SourceSwitchGraftAllowed,
		SourceSwitchDryRunOnly:                                               sourceGate.SourceSwitchDryRunOnly,
		SourceSwitchLiveReady:                                                sourceGate.SourceSwitchLiveReady,
		SourceSwitchRawDreamTextAllowed:                                      sourceGate.SourceSwitchRawDreamTextAllowed,
		SourceSwitchRawDreamTextObserved:                                     sourceGate.SourceSwitchRawDreamTextObserved,
		SourceSwitchRawDreamTextForwarded:                                    sourceGate.SourceSwitchRawDreamTextForwarded,
		SourceSwitchJanusSurfaceAllowed:                                      sourceGate.SourceSwitchJanusSurfaceAllowed,
		SourceSwitchCoocLearningAllowed:                                      sourceGate.SourceSwitchCoocLearningAllowed,
		SourceSwitchDeltaHarvestAllowed:                                      sourceGate.SourceSwitchDeltaHarvestAllowed,
		SourceSwitchBodyMutationAllowed:                                      sourceGate.SourceSwitchBodyMutationAllowed,
		SourceSwitchRollbackRequired:                                         sourceGate.SourceSwitchRollbackRequired,
		SourceSwitchReadOnly:                                                 sourceGate.SourceSwitchReadOnly,
		SourceSwitchReplayOnly:                                               sourceGate.SourceSwitchReplayOnly,
		SourceSwitchWriteAllowed:                                             sourceGate.SourceSwitchWriteAllowed,
		SourceSwitchAdmissionAllowed:                                         sourceGate.SourceSwitchAdmissionAllowed,
		SourceSwitchLiveAdmissionEnabled:                                     sourceGate.SourceSwitchLiveAdmissionEnabled,
		SourceSwitchMutatesState:                                             sourceGate.SourceSwitchMutatesState,
		SourceSwitchBodyTarget:                                               sourceGate.SourceSwitchBodyTarget,
		SourceSwitchPassed:                                                   sourceGate.SourceSwitchPassed,
		SourceSwitchReason:                                                   sourceGate.SourceSwitchReason,
		SourceWeightedAdmissionResonanceGraftAdmissionPromotionID:            sourceGate.SourceWeightedAdmissionResonanceGraftAdmissionPromotionID,
		SourceWeightedAdmissionResonanceGraftAdmissionPromotionReady:         sourceGate.SourceWeightedAdmissionResonanceGraftAdmissionPromotionReady,
		SourceWeightedAdmissionResonanceGraftAdmissionPromotionCausalID:      sourceGate.SourceWeightedAdmissionResonanceGraftAdmissionPromotionCausalID,
		SourceWeightedAdmissionResonanceGraftAdmissionPromotionHash:          sourceGate.SourceWeightedAdmissionResonanceGraftAdmissionPromotionHash,
		SourceWeightedAdmissionResonanceGraftAdmissionPromotionReadBack:      sourceGate.SourceWeightedAdmissionResonanceGraftAdmissionPromotionReadBack,
		SourcePromotion:                                                      sourceGate.SourcePromotion,
		SourcePromotionAction:                                                sourceGate.SourcePromotionAction,
		SourcePromotionReceiptShape:                                          sourceGate.SourcePromotionReceiptShape,
		SourcePromotionKind:                                                  sourceGate.SourcePromotionKind,
		SourcePromotionMode:                                                  sourceGate.SourcePromotionMode,
		SourcePromotionStage:                                                 sourceGate.SourcePromotionStage,
		SourcePromotionAdmissionRequired:                                     sourceGate.SourcePromotionAdmissionRequired,
		SourcePromotionShadowOnly:                                            sourceGate.SourcePromotionShadowOnly,
		SourcePromotionGraftAllowed:                                          sourceGate.SourcePromotionGraftAllowed,
		SourcePromotionDryRunOnly:                                            sourceGate.SourcePromotionDryRunOnly,
		SourcePromotionLiveReady:                                             sourceGate.SourcePromotionLiveReady,
		SourcePromotionRawDreamTextAllowed:                                   sourceGate.SourcePromotionRawDreamTextAllowed,
		SourcePromotionRawDreamTextObserved:                                  sourceGate.SourcePromotionRawDreamTextObserved,
		SourcePromotionRawDreamTextForwarded:                                 sourceGate.SourcePromotionRawDreamTextForwarded,
		SourcePromotionJanusSurfaceAllowed:                                   sourceGate.SourcePromotionJanusSurfaceAllowed,
		SourcePromotionCoocLearningAllowed:                                   sourceGate.SourcePromotionCoocLearningAllowed,
		SourcePromotionDeltaHarvestAllowed:                                   sourceGate.SourcePromotionDeltaHarvestAllowed,
		SourcePromotionBodyMutationAllowed:                                   sourceGate.SourcePromotionBodyMutationAllowed,
		SourcePromotionRollbackRequired:                                      sourceGate.SourcePromotionRollbackRequired,
		SourcePromotionReadOnly:                                              sourceGate.SourcePromotionReadOnly,
		SourcePromotionReplayOnly:                                            sourceGate.SourcePromotionReplayOnly,
		SourcePromotionWriteAllowed:                                          sourceGate.SourcePromotionWriteAllowed,
		SourcePromotionAdmissionAllowed:                                      sourceGate.SourcePromotionAdmissionAllowed,
		SourcePromotionLiveAdmissionEnabled:                                  sourceGate.SourcePromotionLiveAdmissionEnabled,
		SourcePromotionMutatesState:                                          sourceGate.SourcePromotionMutatesState,
		SourcePromotionBodyTarget:                                            sourceGate.SourcePromotionBodyTarget,
		SourcePromotionPassed:                                                sourceGate.SourcePromotionPassed,
		SourcePromotionReason:                                                sourceGate.SourcePromotionReason,
		SourceWeightedAdmissionResonanceGraftAdmissionDecisionID:             sourceGate.SourceWeightedAdmissionResonanceGraftAdmissionDecisionID,
		SourceWeightedAdmissionResonanceGraftAdmissionDecisionReady:          sourceGate.SourceWeightedAdmissionResonanceGraftAdmissionDecisionReady,
		SourceWeightedAdmissionResonanceGraftAdmissionProofPreconditionID:    sourceGate.SourceWeightedAdmissionResonanceGraftAdmissionProofPreconditionID,
		SourceWeightedAdmissionResonanceGraftAdmissionProofPreconditionReady: sourceGate.SourceWeightedAdmissionResonanceGraftAdmissionProofPreconditionReady,
		SourceWeightedAdmissionResonanceGraftAdmissionProofID:                sourceGate.SourceWeightedAdmissionResonanceGraftAdmissionProofID,
		SourceWeightedAdmissionResonanceGraftAdmissionProofReady:             sourceGate.SourceWeightedAdmissionResonanceGraftAdmissionProofReady,
		SourceWeightedAdmissionResonanceGraftCandidateStoreReaderID:          sourceGate.SourceWeightedAdmissionResonanceGraftCandidateStoreReaderID,
		SourceWeightedAdmissionResonanceGraftCandidateStoreReaderReady:       sourceGate.SourceWeightedAdmissionResonanceGraftCandidateStoreReaderReady,
		SourceWeightedAdmissionResonanceGraftCandidateStoreID:                sourceGate.SourceWeightedAdmissionResonanceGraftCandidateStoreID,
		SourceWeightedAdmissionResonanceGraftCandidateStoreReady:             sourceGate.SourceWeightedAdmissionResonanceGraftCandidateStoreReady,
		SourceWeightedAdmissionResonanceGraftCandidateID:                     sourceGate.SourceWeightedAdmissionResonanceGraftCandidateID,
		SourceWeightedAdmissionResonanceGraftCandidateReady:                  sourceGate.SourceWeightedAdmissionResonanceGraftCandidateReady,
		SourceWeightedAdmissionResonanceGraftGateID:                          sourceGate.SourceWeightedAdmissionResonanceGraftGateID,
		SourceWeightedAdmissionResonanceGraftGateReady:                       sourceGate.SourceWeightedAdmissionResonanceGraftGateReady,
		SourceWeightedAdmissionResonanceGraftPreflightID:                     sourceGate.SourceWeightedAdmissionResonanceGraftPreflightID,
		SourceWeightedAdmissionResonanceGraftPreflightReady:                  sourceGate.SourceWeightedAdmissionResonanceGraftPreflightReady,
		SourceWeightedAdmissionResonanceGraftBoundaryID:                      sourceGate.SourceWeightedAdmissionResonanceGraftBoundaryID,
		SourceWeightedAdmissionResonanceGraftBoundaryReady:                   sourceGate.SourceWeightedAdmissionResonanceGraftBoundaryReady,
		SourceWeightedAdmissionResonanceObservationID:                        sourceGate.SourceWeightedAdmissionResonanceObservationID,
		SourceWeightedAdmissionResonanceObservationReady:                     sourceGate.SourceWeightedAdmissionResonanceObservationReady,
		SourceWeightedAdmissionResonanceReceiverID:                           sourceGate.SourceWeightedAdmissionResonanceReceiverID,
		SourceWeightedAdmissionResonanceReceiverReady:                        sourceGate.SourceWeightedAdmissionResonanceReceiverReady,
		SourceWeightedAdmissionResonanceIntentReady:                          sourceGate.SourceWeightedAdmissionResonanceIntentReady,
		SourceWeightedAdmissionFinalGateReady:                                sourceGate.SourceWeightedAdmissionFinalGateReady,
		SourceWeightedAdmissionSealReady:                                     sourceGate.SourceWeightedAdmissionSealReady,
		SourceWeightedAdmissionPermitReady:                                   sourceGate.SourceWeightedAdmissionPermitReady,
		SourceWeightedAdmissionAuthorityConsumed:                             sourceGate.SourceWeightedAdmissionAuthorityConsumed,
		SourceWeightedAdmissionAuthorityRequired:                             sourceGate.SourceWeightedAdmissionAuthorityRequired,
		BodySmokeWeighted:                                                    sourceGate.BodySmokeWeighted,
		NanoDirectRunner:                                                     sourceGate.NanoDirectRunner,
		NanoDirectFinalGate:                                                  sourceGate.NanoDirectFinalGate,
		ResonanceGraftAdmissionProof:                                         sourceGate.ResonanceGraftAdmissionProof,
		BoundaryReportFullChain:                                              sourceGate.BoundaryReportFullChain,
		SourceAuthorityGranted:                                               sourceGate.SourceAuthorityGranted,
		AuthorityGranted:                                                     false,
		ContractsReady:                                                       false,
		WriteAllowed:                                                         false,
		AdmissionAllowed:                                                     false,
		LiveAdmissionEnabled:                                                 false,
		MutatesState:                                                         false,
		BodyTarget:                                                           "none",
		Passed:                                                               true,
		Reason:                                                               "weighted resonance shadow graft admission live stage blocked by disabled enable gate; writer and rollback remain absent",
	}
	stage.CausalID = admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionLiveStageCausalID(stage)
	stage.LiveStageHash = admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionLiveStageHash(stage)
	stage.ReadBackHash = admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionLiveStageReadBackHash(stage)
	stage.WeightedAdmissionResonanceGraftAdmissionLiveStageID = admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionLiveStageID(stage)
	if stage.CausalID == "" ||
		stage.LiveStageHash == "" ||
		stage.ReadBackHash == "" ||
		stage.WeightedAdmissionResonanceGraftAdmissionLiveStageID == "" ||
		stage.LiveStageHash == stage.ReadBackHash {
		return fmt.Errorf("weighted admission resonance graft admission live stage read-back proof failed")
	}
	raw, err := json.MarshalIndent(stage, "", "  ")
	if err != nil {
		return fmt.Errorf("weighted admission resonance graft admission live stage marshal failed: %w", err)
	}
	raw = append(raw, '\n')
	if err := os.WriteFile(outputPath, raw, 0600); err != nil {
		return fmt.Errorf("weighted admission resonance graft admission live stage write failed: %w", err)
	}
	fmt.Printf("[admission-live-route-weighted-admission-resonance-graft-admission-live-stage] pass: resonance_graft_admission_live_stage_report=%s resonance_graft_admission_enable_gate_report=%s\n", outputPath, enableGatePath)
	return nil
}

func runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionLiveStageAssert(args []string) error {
	if len(args) != 1 {
		return fmt.Errorf("usage: --admission-live-route-weighted-admission-resonance-graft-admission-live-stage-assert REPORT")
	}
	report, root, err := readAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionLiveStageReportForAssert(args[0])
	if err != nil {
		return err
	}
	return admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionLiveStageReportError(report, root)
}

func admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionLiveStageReportError(report admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionLiveStageReport, root map[string]json.RawMessage) error {
	if _, ok := root["schema"]; !ok {
		return fmt.Errorf("weighted admission resonance graft admission live stage schema missing")
	}
	if report.Schema != admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionLiveStageSchema {
		return fmt.Errorf("weighted admission resonance graft admission live stage schema mismatch: got %q want %q", report.Schema, admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionLiveStageSchema)
	}
	if report.Status != "shadow_graft_admission_live_stage_blocked_dry_run" {
		return fmt.Errorf("weighted admission resonance graft admission live stage status mismatch: got %q want %q", report.Status, "shadow_graft_admission_live_stage_blocked_dry_run")
	}
	if report.Target != "live_route_admission_next_step" {
		return fmt.Errorf("weighted admission resonance graft admission live stage target mismatch: got %q want %q", report.Target, "live_route_admission_next_step")
	}
	if report.TargetKind != "weighted_internal_world_shadow_graft_admission_live_stage" {
		return fmt.Errorf("weighted admission resonance graft admission live stage target_kind mismatch: got %q want %q", report.TargetKind, "weighted_internal_world_shadow_graft_admission_live_stage")
	}
	if report.TargetMode != "closed_live_stage_guard_dry_run" {
		return fmt.Errorf("weighted admission resonance graft admission live stage target_mode mismatch: got %q want %q", report.TargetMode, "closed_live_stage_guard_dry_run")
	}
	if report.Action != "block_weighted_resonance_shadow_graft_admission_enable_gate_disabled_dry_run" {
		return fmt.Errorf("weighted admission resonance graft admission live stage action mismatch: got %q want %q", report.Action, "block_weighted_resonance_shadow_graft_admission_enable_gate_disabled_dry_run")
	}
	if report.StageState != "blocked" {
		return fmt.Errorf("weighted admission resonance graft admission live stage stage_state mismatch: got %q want %q", report.StageState, "blocked")
	}
	if report.StageAction != "reject_disabled_enable_gate" {
		return fmt.Errorf("weighted admission resonance graft admission live stage stage_action mismatch: got %q want %q", report.StageAction, "reject_disabled_enable_gate")
	}
	if report.EnableState != "disabled" {
		return fmt.Errorf("weighted admission resonance graft admission live stage enable_state mismatch: got %q want %q", report.EnableState, "disabled")
	}
	if report.EnableAction != "require_operator_key" {
		return fmt.Errorf("weighted admission resonance graft admission live stage enable_action mismatch: got %q want %q", report.EnableAction, "require_operator_key")
	}
	if report.SwitchState != "disabled" {
		return fmt.Errorf("weighted admission resonance graft admission live stage switch_state mismatch: got %q want %q", report.SwitchState, "disabled")
	}
	if report.SwitchAction != "hold_pending_live_admission" {
		return fmt.Errorf("weighted admission resonance graft admission live stage switch_action mismatch: got %q want %q", report.SwitchAction, "hold_pending_live_admission")
	}
	if report.Promotion != "pending_live_admission" {
		return fmt.Errorf("weighted admission resonance graft admission live stage promotion mismatch: got %q want %q", report.Promotion, "pending_live_admission")
	}
	if report.ReceiptShape != "weighted_resonance_shadow_graft_admission_live_stage_receipt" {
		return fmt.Errorf("weighted admission resonance graft admission live stage receipt_shape mismatch: got %q want %q", report.ReceiptShape, "weighted_resonance_shadow_graft_admission_live_stage_receipt")
	}
	if report.LiveStageKind != "shadow_graft_admission_live_stage" ||
		report.LiveStageMode != "closed_enable_gate_live_stage_guard" ||
		report.LiveStageStage != "pre_writer_graft_admission_live_stage" {
		return fmt.Errorf("weighted admission resonance graft admission live stage shape mismatch")
	}
	for _, required := range []struct {
		name  string
		value bool
	}{
		{"weighted_admission_resonance_graft_admission_live_stage_ready", report.WeightedAdmissionResonanceGraftAdmissionLiveStageReady},
		{"weighted_admission_resonance_graft_admission_enable_gate_consumed", report.WeightedAdmissionResonanceGraftAdmissionEnableGateConsumed},
		{"weighted_admission_resonance_graft_admission_enable_gate_required", report.WeightedAdmissionResonanceGraftAdmissionEnableGateRequired},
		{"next_step_blocked_without_resonance_graft_admission_live_stage", report.NextStepBlockedWithoutResonanceGraftAdmissionLiveStage},
		{"enable_gate_verified", report.EnableGateVerified},
		{"enable_gate_hash_verified", report.EnableGateHashVerified},
		{"enable_gate_read_back_verified", report.EnableGateReadBackVerified},
		{"switch_verified", report.SwitchVerified},
		{"switch_hash_verified", report.SwitchHashVerified},
		{"switch_read_back_verified", report.SwitchReadBackVerified},
		{"promotion_verified", report.PromotionVerified},
		{"promotion_hash_verified", report.PromotionHashVerified},
		{"promotion_read_back_verified", report.PromotionReadBackVerified},
		{"decision_verified", report.DecisionVerified},
		{"decision_hash_verified", report.DecisionHashVerified},
		{"decision_read_back_verified", report.DecisionReadBackVerified},
		{"proof_precondition_verified", report.ProofPreconditionVerified},
		{"precondition_hash_verified", report.PreconditionHashVerified},
		{"precondition_read_back_verified", report.PreconditionReadBackVerified},
		{"proof_verified", report.ProofVerified},
		{"proof_hash_verified", report.ProofHashVerified},
		{"proof_read_back_verified", report.ProofReadBackVerified},
		{"store_reader_verified", report.StoreReaderVerified},
		{"store_verified", report.StoreVerified},
		{"candidate_verified", report.CandidateVerified},
		{"gate_verified", report.GateVerified},
		{"preflight_verified", report.PreflightVerified},
		{"boundary_verified", report.BoundaryVerified},
		{"observation_verified", report.ObservationVerified},
		{"receiver_verified", report.ReceiverVerified},
		{"intent_verified", report.IntentVerified},
		{"final_gate_verified", report.FinalGateVerified},
		{"seal_verified", report.SealVerified},
		{"permit_verified", report.PermitVerified},
		{"authority_verified", report.AuthorityVerified},
		{"admission_required", report.AdmissionRequired},
		{"shadow_only", report.ShadowOnly},
		{"dry_run_only", report.DryRunOnly},
		{"live_ready", report.LiveReady},
		{"requires_writer", report.RequiresWriter},
		{"rollback_required", report.RollbackRequired},
		{"requires_rollback", report.RequiresRollback},
		{"read_only", report.ReadOnly},
		{"replay_only", report.ReplayOnly},
		{"source_weighted_admission_resonance_graft_admission_enable_gate_ready", report.SourceWeightedAdmissionResonanceGraftAdmissionEnableGateReady},
		{"source_enable_gate_admission_required", report.SourceEnableGateAdmissionRequired},
		{"source_enable_gate_shadow_only", report.SourceEnableGateShadowOnly},
		{"source_enable_gate_dry_run_only", report.SourceEnableGateDryRunOnly},
		{"source_enable_gate_live_ready", report.SourceEnableGateLiveReady},
		{"source_enable_gate_rollback_required", report.SourceEnableGateRollbackRequired},
		{"source_enable_gate_read_only", report.SourceEnableGateReadOnly},
		{"source_enable_gate_replay_only", report.SourceEnableGateReplayOnly},
		{"source_enable_gate_passed", report.SourceEnableGatePassed},
		{"source_weighted_admission_resonance_graft_admission_switch_ready", report.SourceWeightedAdmissionResonanceGraftAdmissionSwitchReady},
		{"source_switch_admission_required", report.SourceSwitchAdmissionRequired},
		{"source_switch_shadow_only", report.SourceSwitchShadowOnly},
		{"source_switch_dry_run_only", report.SourceSwitchDryRunOnly},
		{"source_switch_live_ready", report.SourceSwitchLiveReady},
		{"source_switch_rollback_required", report.SourceSwitchRollbackRequired},
		{"source_switch_read_only", report.SourceSwitchReadOnly},
		{"source_switch_replay_only", report.SourceSwitchReplayOnly},
		{"source_switch_passed", report.SourceSwitchPassed},
		{"source_weighted_admission_resonance_graft_admission_promotion_ready", report.SourceWeightedAdmissionResonanceGraftAdmissionPromotionReady},
		{"source_promotion_admission_required", report.SourcePromotionAdmissionRequired},
		{"source_promotion_shadow_only", report.SourcePromotionShadowOnly},
		{"source_promotion_dry_run_only", report.SourcePromotionDryRunOnly},
		{"source_promotion_live_ready", report.SourcePromotionLiveReady},
		{"source_promotion_rollback_required", report.SourcePromotionRollbackRequired},
		{"source_promotion_read_only", report.SourcePromotionReadOnly},
		{"source_promotion_replay_only", report.SourcePromotionReplayOnly},
		{"source_promotion_passed", report.SourcePromotionPassed},
		{"source_weighted_admission_resonance_graft_admission_decision_ready", report.SourceWeightedAdmissionResonanceGraftAdmissionDecisionReady},
		{"source_weighted_admission_resonance_graft_admission_proof_precondition_ready", report.SourceWeightedAdmissionResonanceGraftAdmissionProofPreconditionReady},
		{"source_weighted_admission_resonance_graft_admission_proof_ready", report.SourceWeightedAdmissionResonanceGraftAdmissionProofReady},
		{"source_weighted_admission_resonance_graft_candidate_store_reader_ready", report.SourceWeightedAdmissionResonanceGraftCandidateStoreReaderReady},
		{"source_weighted_admission_resonance_graft_candidate_store_ready", report.SourceWeightedAdmissionResonanceGraftCandidateStoreReady},
		{"source_weighted_admission_resonance_graft_candidate_ready", report.SourceWeightedAdmissionResonanceGraftCandidateReady},
		{"source_weighted_admission_resonance_graft_gate_ready", report.SourceWeightedAdmissionResonanceGraftGateReady},
		{"source_weighted_admission_resonance_graft_preflight_ready", report.SourceWeightedAdmissionResonanceGraftPreflightReady},
		{"source_weighted_admission_resonance_graft_boundary_ready", report.SourceWeightedAdmissionResonanceGraftBoundaryReady},
		{"source_weighted_admission_resonance_observation_ready", report.SourceWeightedAdmissionResonanceObservationReady},
		{"source_weighted_admission_resonance_receiver_ready", report.SourceWeightedAdmissionResonanceReceiverReady},
		{"source_weighted_admission_resonance_intent_ready", report.SourceWeightedAdmissionResonanceIntentReady},
		{"source_weighted_admission_final_gate_ready", report.SourceWeightedAdmissionFinalGateReady},
		{"source_weighted_admission_seal_ready", report.SourceWeightedAdmissionSealReady},
		{"source_weighted_admission_permit_ready", report.SourceWeightedAdmissionPermitReady},
		{"source_weighted_admission_authority_consumed", report.SourceWeightedAdmissionAuthorityConsumed},
		{"source_weighted_admission_authority_required", report.SourceWeightedAdmissionAuthorityRequired},
		{"body_smoke_weighted", report.BodySmokeWeighted},
		{"nano_direct_runner", report.NanoDirectRunner},
		{"nano_direct_final_gate", report.NanoDirectFinalGate},
		{"resonance_graft_admission_proof", report.ResonanceGraftAdmissionProof},
		{"boundary_report_full_chain", report.BoundaryReportFullChain},
		{"passed", report.Passed},
	} {
		if !required.value {
			return fmt.Errorf("weighted admission resonance graft admission live stage %s not ready", required.name)
		}
	}
	for _, closed := range []struct {
		name  string
		value bool
	}{
		{"graft_allowed", report.GraftAllowed},
		{"raw_dream_text_allowed", report.RawDreamTextAllowed},
		{"raw_dream_text_observed", report.RawDreamTextObserved},
		{"raw_dream_text_forwarded", report.RawDreamTextForwarded},
		{"janus_surface_allowed", report.JanusSurfaceAllowed},
		{"cooc_learning_allowed", report.CoocLearningAllowed},
		{"delta_harvest_allowed", report.DeltaHarvestAllowed},
		{"body_mutation_allowed", report.BodyMutationAllowed},
		{"writer_ready", report.WriterReady},
		{"rollback_ready", report.RollbackReady},
		{"source_enable_gate_graft_allowed", report.SourceEnableGateGraftAllowed},
		{"source_enable_gate_raw_dream_text_allowed", report.SourceEnableGateRawDreamTextAllowed},
		{"source_enable_gate_raw_dream_text_observed", report.SourceEnableGateRawDreamTextObserved},
		{"source_enable_gate_raw_dream_text_forwarded", report.SourceEnableGateRawDreamTextForwarded},
		{"source_enable_gate_janus_surface_allowed", report.SourceEnableGateJanusSurfaceAllowed},
		{"source_enable_gate_cooc_learning_allowed", report.SourceEnableGateCoocLearningAllowed},
		{"source_enable_gate_delta_harvest_allowed", report.SourceEnableGateDeltaHarvestAllowed},
		{"source_enable_gate_body_mutation_allowed", report.SourceEnableGateBodyMutationAllowed},
		{"source_enable_gate_write_allowed", report.SourceEnableGateWriteAllowed},
		{"source_enable_gate_admission_allowed", report.SourceEnableGateAdmissionAllowed},
		{"source_enable_gate_live_admission_enabled", report.SourceEnableGateLiveAdmissionEnabled},
		{"source_enable_gate_mutates_state", report.SourceEnableGateMutatesState},
		{"source_switch_graft_allowed", report.SourceSwitchGraftAllowed},
		{"source_switch_raw_dream_text_allowed", report.SourceSwitchRawDreamTextAllowed},
		{"source_switch_raw_dream_text_observed", report.SourceSwitchRawDreamTextObserved},
		{"source_switch_raw_dream_text_forwarded", report.SourceSwitchRawDreamTextForwarded},
		{"source_switch_janus_surface_allowed", report.SourceSwitchJanusSurfaceAllowed},
		{"source_switch_cooc_learning_allowed", report.SourceSwitchCoocLearningAllowed},
		{"source_switch_delta_harvest_allowed", report.SourceSwitchDeltaHarvestAllowed},
		{"source_switch_body_mutation_allowed", report.SourceSwitchBodyMutationAllowed},
		{"source_switch_write_allowed", report.SourceSwitchWriteAllowed},
		{"source_switch_admission_allowed", report.SourceSwitchAdmissionAllowed},
		{"source_switch_live_admission_enabled", report.SourceSwitchLiveAdmissionEnabled},
		{"source_switch_mutates_state", report.SourceSwitchMutatesState},
		{"source_promotion_graft_allowed", report.SourcePromotionGraftAllowed},
		{"source_promotion_raw_dream_text_allowed", report.SourcePromotionRawDreamTextAllowed},
		{"source_promotion_raw_dream_text_observed", report.SourcePromotionRawDreamTextObserved},
		{"source_promotion_raw_dream_text_forwarded", report.SourcePromotionRawDreamTextForwarded},
		{"source_promotion_janus_surface_allowed", report.SourcePromotionJanusSurfaceAllowed},
		{"source_promotion_cooc_learning_allowed", report.SourcePromotionCoocLearningAllowed},
		{"source_promotion_delta_harvest_allowed", report.SourcePromotionDeltaHarvestAllowed},
		{"source_promotion_body_mutation_allowed", report.SourcePromotionBodyMutationAllowed},
		{"source_promotion_write_allowed", report.SourcePromotionWriteAllowed},
		{"source_promotion_admission_allowed", report.SourcePromotionAdmissionAllowed},
		{"source_promotion_live_admission_enabled", report.SourcePromotionLiveAdmissionEnabled},
		{"source_promotion_mutates_state", report.SourcePromotionMutatesState},
		{"source_authority_granted", report.SourceAuthorityGranted},
		{"authority_granted", report.AuthorityGranted},
		{"contracts_ready", report.ContractsReady},
		{"write_allowed", report.WriteAllowed},
		{"admission_allowed", report.AdmissionAllowed},
		{"live_admission_enabled", report.LiveAdmissionEnabled},
		{"mutates_state", report.MutatesState},
	} {
		if closed.value {
			return fmt.Errorf("weighted admission resonance graft admission live stage opened %s", closed.name)
		}
	}
	for _, requiredString := range []struct {
		name  string
		value string
	}{
		{"weighted_admission_resonance_graft_admission_live_stage_id", report.WeightedAdmissionResonanceGraftAdmissionLiveStageID},
		{"causal_id", report.CausalID},
		{"live_stage_hash", report.LiveStageHash},
		{"read_back_hash", report.ReadBackHash},
		{"source_report", report.SourceReport},
		{"source_weighted_admission_resonance_graft_admission_enable_gate_id", report.SourceWeightedAdmissionResonanceGraftAdmissionEnableGateID},
		{"source_weighted_admission_resonance_graft_admission_enable_gate_causal_id", report.SourceWeightedAdmissionResonanceGraftAdmissionEnableGateCausalID},
		{"source_weighted_admission_resonance_graft_admission_enable_gate_hash", report.SourceWeightedAdmissionResonanceGraftAdmissionEnableGateHash},
		{"source_weighted_admission_resonance_graft_admission_enable_gate_read_back_hash", report.SourceWeightedAdmissionResonanceGraftAdmissionEnableGateReadBack},
		{"source_weighted_admission_resonance_graft_admission_switch_id", report.SourceWeightedAdmissionResonanceGraftAdmissionSwitchID},
		{"source_weighted_admission_resonance_graft_admission_switch_causal_id", report.SourceWeightedAdmissionResonanceGraftAdmissionSwitchCausalID},
		{"source_weighted_admission_resonance_graft_admission_switch_hash", report.SourceWeightedAdmissionResonanceGraftAdmissionSwitchHash},
		{"source_weighted_admission_resonance_graft_admission_switch_read_back_hash", report.SourceWeightedAdmissionResonanceGraftAdmissionSwitchReadBack},
		{"source_weighted_admission_resonance_graft_admission_promotion_id", report.SourceWeightedAdmissionResonanceGraftAdmissionPromotionID},
		{"source_weighted_admission_resonance_graft_admission_promotion_causal_id", report.SourceWeightedAdmissionResonanceGraftAdmissionPromotionCausalID},
		{"source_weighted_admission_resonance_graft_admission_promotion_hash", report.SourceWeightedAdmissionResonanceGraftAdmissionPromotionHash},
		{"source_weighted_admission_resonance_graft_admission_promotion_read_back_hash", report.SourceWeightedAdmissionResonanceGraftAdmissionPromotionReadBack},
		{"source_weighted_admission_resonance_graft_admission_decision_id", report.SourceWeightedAdmissionResonanceGraftAdmissionDecisionID},
		{"source_weighted_admission_resonance_graft_admission_proof_precondition_id", report.SourceWeightedAdmissionResonanceGraftAdmissionProofPreconditionID},
		{"source_weighted_admission_resonance_graft_admission_proof_id", report.SourceWeightedAdmissionResonanceGraftAdmissionProofID},
		{"source_weighted_admission_resonance_graft_candidate_store_reader_id", report.SourceWeightedAdmissionResonanceGraftCandidateStoreReaderID},
		{"source_weighted_admission_resonance_graft_candidate_store_id", report.SourceWeightedAdmissionResonanceGraftCandidateStoreID},
		{"source_weighted_admission_resonance_graft_candidate_id", report.SourceWeightedAdmissionResonanceGraftCandidateID},
		{"source_weighted_admission_resonance_graft_gate_id", report.SourceWeightedAdmissionResonanceGraftGateID},
		{"source_weighted_admission_resonance_graft_preflight_id", report.SourceWeightedAdmissionResonanceGraftPreflightID},
		{"source_weighted_admission_resonance_graft_boundary_id", report.SourceWeightedAdmissionResonanceGraftBoundaryID},
		{"source_weighted_admission_resonance_observation_id", report.SourceWeightedAdmissionResonanceObservationID},
		{"source_weighted_admission_resonance_receiver_id", report.SourceWeightedAdmissionResonanceReceiverID},
	} {
		if strings.TrimSpace(requiredString.value) == "" {
			return fmt.Errorf("weighted admission resonance graft admission live stage %s missing", requiredString.name)
		}
	}
	if report.SourceSchema != admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionEnableGateSchema {
		return fmt.Errorf("weighted admission resonance graft admission live stage source_schema mismatch: got %q want %q", report.SourceSchema, admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionEnableGateSchema)
	}
	if report.SourceStatus != "shadow_graft_admission_enable_gate_disabled_dry_run" {
		return fmt.Errorf("weighted admission resonance graft admission live stage source_status mismatch: got %q want %q", report.SourceStatus, "shadow_graft_admission_enable_gate_disabled_dry_run")
	}
	if report.SourceTarget != "live_route_admission_next_step" {
		return fmt.Errorf("weighted admission resonance graft admission live stage source_target mismatch: got %q want %q", report.SourceTarget, "live_route_admission_next_step")
	}
	if report.SourceEnableState != "disabled" {
		return fmt.Errorf("weighted admission resonance graft admission live stage source_enable_state mismatch: got %q want %q", report.SourceEnableState, "disabled")
	}
	if report.SourceEnableAction != "require_operator_key" {
		return fmt.Errorf("weighted admission resonance graft admission live stage source_enable_action mismatch: got %q want %q", report.SourceEnableAction, "require_operator_key")
	}
	if report.SourceEnableGateReceiptShape != "weighted_resonance_shadow_graft_admission_enable_gate_receipt" ||
		report.SourceEnableGateKind != "shadow_graft_admission_enable_gate" ||
		report.SourceEnableGateMode != "closed_switch_enable_guard" ||
		report.SourceEnableGateStage != "pre_live_graft_admission_enable_gate" {
		return fmt.Errorf("weighted admission resonance graft admission live stage source enable gate shape mismatch")
	}
	if report.SourceEnableGateBodyTarget != "none" {
		return fmt.Errorf("weighted admission resonance graft admission live stage source_enable_gate_body_target mismatch: got %q want %q", report.SourceEnableGateBodyTarget, "none")
	}
	if report.SourceEnableGateReason != "weighted resonance shadow graft admission enable gate closed; operator key absent and mutation refused" {
		return fmt.Errorf("weighted admission resonance graft admission live stage source_enable_gate_reason mismatch: got %q", report.SourceEnableGateReason)
	}
	if report.EnableState != report.SourceEnableState || report.EnableAction != report.SourceEnableAction {
		return fmt.Errorf("weighted admission resonance graft admission live stage source enable state/action not carried")
	}
	if report.SourceSwitchState != "disabled" {
		return fmt.Errorf("weighted admission resonance graft admission live stage source_switch_state mismatch: got %q want %q", report.SourceSwitchState, "disabled")
	}
	if report.SourceSwitchAction != "hold_pending_live_admission" {
		return fmt.Errorf("weighted admission resonance graft admission live stage source_switch_action mismatch: got %q want %q", report.SourceSwitchAction, "hold_pending_live_admission")
	}
	if report.SwitchState != report.SourceSwitchState || report.SwitchAction != report.SourceSwitchAction {
		return fmt.Errorf("weighted admission resonance graft admission live stage source switch state/action not carried")
	}
	if report.SourceSwitchReceiptShape != "weighted_resonance_shadow_graft_admission_switch_receipt" ||
		report.SourceSwitchKind != "shadow_graft_admission_switch" ||
		report.SourceSwitchMode != "closed_promotion_switch_guard" ||
		report.SourceSwitchStage != "pre_live_graft_admission_switch" {
		return fmt.Errorf("weighted admission resonance graft admission live stage source switch shape mismatch")
	}
	if report.SourceSwitchBodyTarget != "none" {
		return fmt.Errorf("weighted admission resonance graft admission live stage source_switch_body_target mismatch: got %q want %q", report.SourceSwitchBodyTarget, "none")
	}
	if report.SourceSwitchReason != "weighted resonance shadow graft admission promotion held at disabled switch without mutation" {
		return fmt.Errorf("weighted admission resonance graft admission live stage source_switch_reason mismatch: got %q", report.SourceSwitchReason)
	}
	if report.SourcePromotion != "pending_live_admission" {
		return fmt.Errorf("weighted admission resonance graft admission live stage source_promotion mismatch: got %q want %q", report.SourcePromotion, "pending_live_admission")
	}
	if report.SourcePromotionAction != "promote_weighted_resonance_shadow_graft_admission_dry_run" {
		return fmt.Errorf("weighted admission resonance graft admission live stage source_promotion_action mismatch: got %q want %q", report.SourcePromotionAction, "promote_weighted_resonance_shadow_graft_admission_dry_run")
	}
	if report.SourcePromotionReceiptShape != "weighted_resonance_shadow_graft_admission_promotion_receipt" ||
		report.SourcePromotionKind != "shadow_graft_admission_promotion" ||
		report.SourcePromotionMode != "closed_decision_promotion" ||
		report.SourcePromotionStage != "pre_live_graft_admission_promotion" {
		return fmt.Errorf("weighted admission resonance graft admission live stage source promotion shape mismatch")
	}
	if report.SourcePromotionBodyTarget != "none" {
		return fmt.Errorf("weighted admission resonance graft admission live stage source_promotion_body_target mismatch: got %q want %q", report.SourcePromotionBodyTarget, "none")
	}
	if report.BodyTarget != "none" {
		return fmt.Errorf("weighted admission resonance graft admission live stage body_target mismatch: got %q want %q", report.BodyTarget, "none")
	}
	if !strings.HasPrefix(report.WeightedAdmissionResonanceGraftAdmissionLiveStageID, "weighted-resonance-graft-admission-live-stage-id-") {
		return fmt.Errorf("weighted admission resonance graft admission live stage id prefix mismatch")
	}
	if !strings.HasPrefix(report.CausalID, "weighted-resonance-graft-admission-live-stage-causal-") {
		return fmt.Errorf("weighted admission resonance graft admission live stage causal prefix mismatch")
	}
	if !strings.HasPrefix(report.LiveStageHash, "weighted-resonance-graft-admission-live-stage-") {
		return fmt.Errorf("weighted admission resonance graft admission live stage hash prefix mismatch")
	}
	if !strings.HasPrefix(report.ReadBackHash, "weighted-resonance-graft-admission-live-stage-read-") ||
		report.LiveStageHash == report.ReadBackHash {
		return fmt.Errorf("weighted admission resonance graft admission live stage read-back mismatch")
	}
	if !strings.HasPrefix(report.SourceWeightedAdmissionResonanceGraftAdmissionEnableGateID, "weighted-resonance-graft-admission-enable-gate-id-") ||
		!strings.HasPrefix(report.SourceWeightedAdmissionResonanceGraftAdmissionEnableGateCausalID, "weighted-resonance-graft-admission-enable-gate-causal-") ||
		!strings.HasPrefix(report.SourceWeightedAdmissionResonanceGraftAdmissionEnableGateHash, "weighted-resonance-graft-admission-enable-gate-") ||
		!strings.HasPrefix(report.SourceWeightedAdmissionResonanceGraftAdmissionEnableGateReadBack, "weighted-resonance-graft-admission-enable-gate-read-") {
		return fmt.Errorf("weighted admission resonance graft admission live stage source enable gate mismatch")
	}
	if !strings.HasPrefix(report.SourceWeightedAdmissionResonanceGraftAdmissionSwitchID, "weighted-resonance-graft-admission-switch-id-") ||
		!strings.HasPrefix(report.SourceWeightedAdmissionResonanceGraftAdmissionSwitchCausalID, "weighted-resonance-graft-admission-switch-causal-") ||
		!strings.HasPrefix(report.SourceWeightedAdmissionResonanceGraftAdmissionSwitchHash, "weighted-resonance-graft-admission-switch-") ||
		!strings.HasPrefix(report.SourceWeightedAdmissionResonanceGraftAdmissionSwitchReadBack, "weighted-resonance-graft-admission-switch-read-") {
		return fmt.Errorf("weighted admission resonance graft admission live stage source switch mismatch")
	}
	if !strings.HasPrefix(report.SourceWeightedAdmissionResonanceGraftAdmissionPromotionID, "weighted-resonance-graft-admission-promotion-id-") ||
		!strings.HasPrefix(report.SourceWeightedAdmissionResonanceGraftAdmissionPromotionCausalID, "weighted-resonance-graft-admission-promotion-causal-") ||
		!strings.HasPrefix(report.SourceWeightedAdmissionResonanceGraftAdmissionPromotionHash, "weighted-resonance-graft-admission-promotion-") ||
		!strings.HasPrefix(report.SourceWeightedAdmissionResonanceGraftAdmissionPromotionReadBack, "weighted-resonance-graft-admission-promotion-read-") {
		return fmt.Errorf("weighted admission resonance graft admission live stage source promotion mismatch")
	}
	if !strings.HasPrefix(report.SourceWeightedAdmissionResonanceGraftAdmissionDecisionID, "weighted-resonance-graft-admission-decision-id-") {
		return fmt.Errorf("weighted admission resonance graft admission live stage source decision id prefix mismatch")
	}
	if !strings.HasPrefix(report.SourceWeightedAdmissionResonanceGraftAdmissionProofPreconditionID, "weighted-resonance-graft-admission-proof-precondition-id-") {
		return fmt.Errorf("weighted admission resonance graft admission live stage source precondition id prefix mismatch")
	}
	if !strings.HasPrefix(report.SourceWeightedAdmissionResonanceGraftAdmissionProofID, "weighted-resonance-graft-admission-proof-id-") {
		return fmt.Errorf("weighted admission resonance graft admission live stage source proof id prefix mismatch")
	}
	if !strings.HasPrefix(report.SourceWeightedAdmissionResonanceGraftCandidateStoreReaderID, "weighted-resonance-graft-candidate-store-reader-id-") {
		return fmt.Errorf("weighted admission resonance graft admission live stage source reader id prefix mismatch")
	}
	if !strings.HasPrefix(report.SourceWeightedAdmissionResonanceGraftCandidateStoreID, "weighted-resonance-graft-candidate-store-id-") {
		return fmt.Errorf("weighted admission resonance graft admission live stage source store id prefix mismatch")
	}
	if !strings.HasPrefix(report.SourceWeightedAdmissionResonanceGraftCandidateID, "weighted-resonance-graft-candidate-id-") {
		return fmt.Errorf("weighted admission resonance graft admission live stage source candidate id prefix mismatch")
	}
	if !strings.HasPrefix(report.SourceWeightedAdmissionResonanceGraftGateID, "weighted-resonance-graft-gate-id-") {
		return fmt.Errorf("weighted admission resonance graft admission live stage source gate id prefix mismatch")
	}
	if !strings.HasPrefix(report.SourceWeightedAdmissionResonanceGraftPreflightID, "weighted-resonance-graft-preflight-id-") {
		return fmt.Errorf("weighted admission resonance graft admission live stage source preflight id prefix mismatch")
	}
	if !strings.HasPrefix(report.SourceWeightedAdmissionResonanceGraftBoundaryID, "weighted-resonance-graft-boundary-id-") {
		return fmt.Errorf("weighted admission resonance graft admission live stage source boundary id prefix mismatch")
	}
	if !strings.HasPrefix(report.SourceWeightedAdmissionResonanceObservationID, "weighted-resonance-observation-") {
		return fmt.Errorf("weighted admission resonance graft admission live stage source observation id prefix mismatch")
	}
	if !strings.HasPrefix(report.SourceWeightedAdmissionResonanceReceiverID, "weighted-resonance-receiver-") {
		return fmt.Errorf("weighted admission resonance graft admission live stage source receiver id prefix mismatch")
	}
	if report.CausalID == "" || report.CausalID != admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionLiveStageCausalID(report) {
		return fmt.Errorf("weighted admission resonance graft admission live stage causal_id mismatch")
	}
	if report.LiveStageHash == "" || report.LiveStageHash != admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionLiveStageHash(report) {
		return fmt.Errorf("weighted admission resonance graft admission live stage live_stage_hash mismatch")
	}
	if report.ReadBackHash == "" || report.ReadBackHash != admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionLiveStageReadBackHash(report) {
		return fmt.Errorf("weighted admission resonance graft admission live stage read_back_hash mismatch")
	}
	if report.LiveStageHash == report.ReadBackHash {
		return fmt.Errorf("weighted admission resonance graft admission live stage read-back proof collapsed")
	}
	if report.WeightedAdmissionResonanceGraftAdmissionLiveStageID == "" || report.WeightedAdmissionResonanceGraftAdmissionLiveStageID != admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionLiveStageID(report) {
		return fmt.Errorf("weighted admission resonance graft admission live stage id mismatch")
	}
	if report.Reason != "weighted resonance shadow graft admission live stage blocked by disabled enable gate; writer and rollback remain absent" {
		return fmt.Errorf("weighted admission resonance graft admission live stage reason mismatch: got %q", report.Reason)
	}
	return nil
}

func admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionLiveStageCausalID(sw admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionLiveStageReport) string {
	h := hashJSON(struct {
		SourceEnableGateID   string `json:"source_enable_gate_id"`
		SourceEnableGateRead string `json:"source_enable_gate_read_back_hash"`
		SourceSwitchID       string `json:"source_switch_id"`
		Target               string `json:"target"`
		LiveStageKind        string `json:"live_stage_kind"`
		LiveStageStage       string `json:"live_stage_stage"`
	}{
		SourceEnableGateID:   sw.SourceWeightedAdmissionResonanceGraftAdmissionEnableGateID,
		SourceEnableGateRead: sw.SourceWeightedAdmissionResonanceGraftAdmissionEnableGateReadBack,
		SourceSwitchID:       sw.SourceWeightedAdmissionResonanceGraftAdmissionSwitchID,
		Target:               sw.Target,
		LiveStageKind:        sw.LiveStageKind,
		LiveStageStage:       sw.LiveStageStage,
	})
	if h == "" {
		return ""
	}
	return "weighted-resonance-graft-admission-live-stage-causal-" + h
}

func admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionLiveStageHash(sw admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionLiveStageReport) string {
	h := hashJSON(struct {
		CausalID             string `json:"causal_id"`
		SourceEnableGateID   string `json:"source_enable_gate_id"`
		SourceEnableGateHash string `json:"source_enable_gate_hash"`
		SourceEnableGateRead string `json:"source_enable_gate_read_back_hash"`
		StageState           string `json:"stage_state"`
		StageAction          string `json:"stage_action"`
		EnableState          string `json:"enable_state"`
		EnableAction         string `json:"enable_action"`
		SwitchState          string `json:"switch_state"`
		SwitchAction         string `json:"switch_action"`
		Promotion            string `json:"promotion"`
		Action               string `json:"action"`
		ReceiptShape         string `json:"receipt_shape"`
		LiveStageMode        string `json:"live_stage_mode"`
		EnableGateVerified   bool   `json:"enable_gate_verified"`
		RequiresWriter       bool   `json:"requires_writer"`
		WriterReady          bool   `json:"writer_ready"`
		RequiresRollback     bool   `json:"requires_rollback"`
		RollbackReady        bool   `json:"rollback_ready"`
		ReadOnly             bool   `json:"read_only"`
		ReplayOnly           bool   `json:"replay_only"`
		AdmissionRequired    bool   `json:"admission_required"`
		ShadowOnly           bool   `json:"shadow_only"`
		DryRunOnly           bool   `json:"dry_run_only"`
		GraftAllowed         bool   `json:"graft_allowed"`
		BodyMutation         bool   `json:"body_mutation_allowed"`
		LiveAdmission        bool   `json:"live_admission_enabled"`
	}{
		CausalID:             sw.CausalID,
		SourceEnableGateID:   sw.SourceWeightedAdmissionResonanceGraftAdmissionEnableGateID,
		SourceEnableGateHash: sw.SourceWeightedAdmissionResonanceGraftAdmissionEnableGateHash,
		SourceEnableGateRead: sw.SourceWeightedAdmissionResonanceGraftAdmissionEnableGateReadBack,
		StageState:           sw.StageState,
		StageAction:          sw.StageAction,
		EnableState:          sw.EnableState,
		EnableAction:         sw.EnableAction,
		SwitchState:          sw.SwitchState,
		SwitchAction:         sw.SwitchAction,
		Promotion:            sw.Promotion,
		Action:               sw.Action,
		ReceiptShape:         sw.ReceiptShape,
		LiveStageMode:        sw.LiveStageMode,
		EnableGateVerified:   sw.EnableGateVerified,
		RequiresWriter:       sw.RequiresWriter,
		WriterReady:          sw.WriterReady,
		RequiresRollback:     sw.RequiresRollback,
		RollbackReady:        sw.RollbackReady,
		ReadOnly:             sw.ReadOnly,
		ReplayOnly:           sw.ReplayOnly,
		AdmissionRequired:    sw.AdmissionRequired,
		ShadowOnly:           sw.ShadowOnly,
		DryRunOnly:           sw.DryRunOnly,
		GraftAllowed:         sw.GraftAllowed,
		BodyMutation:         sw.BodyMutationAllowed,
		LiveAdmission:        sw.LiveAdmissionEnabled,
	})
	if h == "" {
		return ""
	}
	return "weighted-resonance-graft-admission-live-stage-" + h
}

func admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionLiveStageReadBackHash(sw admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionLiveStageReport) string {
	h := hashJSON(struct {
		LiveStageHash        string `json:"live_stage_hash"`
		SourceEnableGateID   string `json:"source_enable_gate_id"`
		SourceEnableGateRead string `json:"source_enable_gate_read_back_hash"`
		LiveStageKind        string `json:"live_stage_kind"`
		LiveStageReady       bool   `json:"live_stage_ready"`
		EnableGateConsumed   bool   `json:"enable_gate_consumed"`
		RequiresWriter       bool   `json:"requires_writer"`
		WriterReady          bool   `json:"writer_ready"`
		RequiresRollback     bool   `json:"requires_rollback"`
		RollbackReady        bool   `json:"rollback_ready"`
		LiveReady            bool   `json:"live_ready"`
		BodyMutation         bool   `json:"body_mutation"`
		LiveAdmission        bool   `json:"live_admission"`
		WriteAllowed         bool   `json:"write_allowed"`
		AdmissionAllowed     bool   `json:"admission_allowed"`
	}{
		LiveStageHash:        sw.LiveStageHash,
		SourceEnableGateID:   sw.SourceWeightedAdmissionResonanceGraftAdmissionEnableGateID,
		SourceEnableGateRead: sw.SourceWeightedAdmissionResonanceGraftAdmissionEnableGateReadBack,
		LiveStageKind:        sw.LiveStageKind,
		LiveStageReady:       sw.WeightedAdmissionResonanceGraftAdmissionLiveStageReady,
		EnableGateConsumed:   sw.WeightedAdmissionResonanceGraftAdmissionEnableGateConsumed,
		RequiresWriter:       sw.RequiresWriter,
		WriterReady:          sw.WriterReady,
		RequiresRollback:     sw.RequiresRollback,
		RollbackReady:        sw.RollbackReady,
		LiveReady:            sw.LiveReady,
		BodyMutation:         sw.BodyMutationAllowed,
		LiveAdmission:        sw.LiveAdmissionEnabled,
		WriteAllowed:         sw.WriteAllowed,
		AdmissionAllowed:     sw.AdmissionAllowed,
	})
	if h == "" {
		return ""
	}
	return "weighted-resonance-graft-admission-live-stage-read-" + h
}

func admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionLiveStageID(sw admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionLiveStageReport) string {
	h := hashJSON(struct {
		Schema                  string `json:"schema"`
		Status                  string `json:"status"`
		Action                  string `json:"action"`
		StageState              string `json:"stage_state"`
		StageAction             string `json:"stage_action"`
		EnableState             string `json:"enable_state"`
		EnableAction            string `json:"enable_action"`
		SwitchState             string `json:"switch_state"`
		SwitchAction            string `json:"switch_action"`
		Promotion               string `json:"promotion"`
		SourceReport            string `json:"source_report"`
		SourceEnableGateID      string `json:"source_enable_gate_id"`
		SourceEnableGateHash    string `json:"source_enable_gate_hash"`
		SourceEnableGateRead    string `json:"source_enable_gate_read_back_hash"`
		SourceSwitchID          string `json:"source_switch_id"`
		SourceSwitchHash        string `json:"source_switch_hash"`
		SourceSwitchRead        string `json:"source_switch_read_back_hash"`
		SourcePromotionID       string `json:"source_promotion_id"`
		SourceDecisionID        string `json:"source_decision_id"`
		SourceProofID           string `json:"source_proof_id"`
		SourceReaderID          string `json:"source_reader_id"`
		SourceStoreID           string `json:"source_store_id"`
		SourceCandidateID       string `json:"source_candidate_id"`
		SourceGateID            string `json:"source_gate_id"`
		SourcePreflightID       string `json:"source_preflight_id"`
		SourceBoundaryID        string `json:"source_boundary_id"`
		SourceObservationID     string `json:"source_observation_id"`
		SourceReceiverID        string `json:"source_receiver_id"`
		CausalID                string `json:"causal_id"`
		LiveStageHash           string `json:"live_stage_hash"`
		ReadBackHash            string `json:"read_back_hash"`
		Ready                   bool   `json:"ready"`
		ReceiptShape            string `json:"receipt_shape"`
		LiveStageKind           string `json:"live_stage_kind"`
		LiveStageMode           string `json:"live_stage_mode"`
		LiveStageStage          string `json:"live_stage_stage"`
		EnableGateVerified      bool   `json:"enable_gate_verified"`
		SwitchVerified          bool   `json:"switch_verified"`
		PromotionVerified       bool   `json:"promotion_verified"`
		AdmissionRequired       bool   `json:"admission_required"`
		ShadowOnly              bool   `json:"shadow_only"`
		GraftAllowed            bool   `json:"graft_allowed"`
		DryRunOnly              bool   `json:"dry_run_only"`
		RawDreamTextAllowed     bool   `json:"raw_dream_text_allowed"`
		JanusSurfaceAllowed     bool   `json:"janus_surface_allowed"`
		CoocLearningAllowed     bool   `json:"cooc_learning_allowed"`
		DeltaHarvestAllowed     bool   `json:"delta_harvest_allowed"`
		BodyMutationAllowed     bool   `json:"body_mutation_allowed"`
		RequiresWriter          bool   `json:"requires_writer"`
		WriterReady             bool   `json:"writer_ready"`
		RollbackRequired        bool   `json:"rollback_required"`
		RequiresRollback        bool   `json:"requires_rollback"`
		RollbackReady           bool   `json:"rollback_ready"`
		ReadOnly                bool   `json:"read_only"`
		ReplayOnly              bool   `json:"replay_only"`
		LiveReady               bool   `json:"live_ready"`
		ContractsReady          bool   `json:"contracts_ready"`
		BodyTarget              string `json:"body_target"`
		WriteAllowed            bool   `json:"write_allowed"`
		AdmissionAllowed        bool   `json:"admission_allowed"`
		LiveAdmissionEnabled    bool   `json:"live_admission_enabled"`
		MutatesState            bool   `json:"mutates_state"`
		NextStepBlockedWithout  bool   `json:"next_step_blocked_without_resonance_graft_admission_live_stage"`
		SourcePromotionReady    bool   `json:"source_promotion_ready"`
		SourceDecisionReady     bool   `json:"source_decision_ready"`
		SourcePreconditionReady bool   `json:"source_precondition_ready"`
		SourceProofReady        bool   `json:"source_proof_ready"`
		SourceReaderReady       bool   `json:"source_reader_ready"`
		SourceStoreReady        bool   `json:"source_store_ready"`
		SourceCandidateReady    bool   `json:"source_candidate_ready"`
		SourceGateReady         bool   `json:"source_gate_ready"`
		SourcePreflightReady    bool   `json:"source_preflight_ready"`
		SourceBoundaryReady     bool   `json:"source_boundary_ready"`
		SourceObservationReady  bool   `json:"source_observation_ready"`
		SourceReceiverReady     bool   `json:"source_receiver_ready"`
		SourceIntentReady       bool   `json:"source_intent_ready"`
		SourceFinalGateReady    bool   `json:"source_final_gate_ready"`
		SourceSealReady         bool   `json:"source_seal_ready"`
		SourcePermitReady       bool   `json:"source_permit_ready"`
		SourceAuthorityUsed     bool   `json:"source_authority_consumed"`
		SourceAuthorityNeeded   bool   `json:"source_authority_required"`
	}{
		Schema:                  sw.Schema,
		Status:                  sw.Status,
		Action:                  sw.Action,
		StageState:              sw.StageState,
		StageAction:             sw.StageAction,
		EnableState:             sw.EnableState,
		EnableAction:            sw.EnableAction,
		SwitchState:             sw.SwitchState,
		SwitchAction:            sw.SwitchAction,
		Promotion:               sw.Promotion,
		SourceReport:            sw.SourceReport,
		SourceEnableGateID:      sw.SourceWeightedAdmissionResonanceGraftAdmissionEnableGateID,
		SourceEnableGateHash:    sw.SourceWeightedAdmissionResonanceGraftAdmissionEnableGateHash,
		SourceEnableGateRead:    sw.SourceWeightedAdmissionResonanceGraftAdmissionEnableGateReadBack,
		SourceSwitchID:          sw.SourceWeightedAdmissionResonanceGraftAdmissionSwitchID,
		SourceSwitchHash:        sw.SourceWeightedAdmissionResonanceGraftAdmissionSwitchHash,
		SourceSwitchRead:        sw.SourceWeightedAdmissionResonanceGraftAdmissionSwitchReadBack,
		SourcePromotionID:       sw.SourceWeightedAdmissionResonanceGraftAdmissionPromotionID,
		SourceDecisionID:        sw.SourceWeightedAdmissionResonanceGraftAdmissionDecisionID,
		SourceProofID:           sw.SourceWeightedAdmissionResonanceGraftAdmissionProofID,
		SourceReaderID:          sw.SourceWeightedAdmissionResonanceGraftCandidateStoreReaderID,
		SourceStoreID:           sw.SourceWeightedAdmissionResonanceGraftCandidateStoreID,
		SourceCandidateID:       sw.SourceWeightedAdmissionResonanceGraftCandidateID,
		SourceGateID:            sw.SourceWeightedAdmissionResonanceGraftGateID,
		SourcePreflightID:       sw.SourceWeightedAdmissionResonanceGraftPreflightID,
		SourceBoundaryID:        sw.SourceWeightedAdmissionResonanceGraftBoundaryID,
		SourceObservationID:     sw.SourceWeightedAdmissionResonanceObservationID,
		SourceReceiverID:        sw.SourceWeightedAdmissionResonanceReceiverID,
		CausalID:                sw.CausalID,
		LiveStageHash:           sw.LiveStageHash,
		ReadBackHash:            sw.ReadBackHash,
		Ready:                   sw.WeightedAdmissionResonanceGraftAdmissionLiveStageReady,
		ReceiptShape:            sw.ReceiptShape,
		LiveStageKind:           sw.LiveStageKind,
		LiveStageMode:           sw.LiveStageMode,
		LiveStageStage:          sw.LiveStageStage,
		EnableGateVerified:      sw.EnableGateVerified,
		SwitchVerified:          sw.SwitchVerified,
		PromotionVerified:       sw.PromotionVerified,
		AdmissionRequired:       sw.AdmissionRequired,
		ShadowOnly:              sw.ShadowOnly,
		GraftAllowed:            sw.GraftAllowed,
		DryRunOnly:              sw.DryRunOnly,
		RawDreamTextAllowed:     sw.RawDreamTextAllowed,
		JanusSurfaceAllowed:     sw.JanusSurfaceAllowed,
		CoocLearningAllowed:     sw.CoocLearningAllowed,
		DeltaHarvestAllowed:     sw.DeltaHarvestAllowed,
		BodyMutationAllowed:     sw.BodyMutationAllowed,
		RequiresWriter:          sw.RequiresWriter,
		WriterReady:             sw.WriterReady,
		RollbackRequired:        sw.RollbackRequired,
		RequiresRollback:        sw.RequiresRollback,
		RollbackReady:           sw.RollbackReady,
		ReadOnly:                sw.ReadOnly,
		ReplayOnly:              sw.ReplayOnly,
		LiveReady:               sw.LiveReady,
		ContractsReady:          sw.ContractsReady,
		BodyTarget:              sw.BodyTarget,
		WriteAllowed:            sw.WriteAllowed,
		AdmissionAllowed:        sw.AdmissionAllowed,
		LiveAdmissionEnabled:    sw.LiveAdmissionEnabled,
		MutatesState:            sw.MutatesState,
		NextStepBlockedWithout:  sw.NextStepBlockedWithoutResonanceGraftAdmissionLiveStage,
		SourcePromotionReady:    sw.SourceWeightedAdmissionResonanceGraftAdmissionPromotionReady,
		SourceDecisionReady:     sw.SourceWeightedAdmissionResonanceGraftAdmissionDecisionReady,
		SourcePreconditionReady: sw.SourceWeightedAdmissionResonanceGraftAdmissionProofPreconditionReady,
		SourceProofReady:        sw.SourceWeightedAdmissionResonanceGraftAdmissionProofReady,
		SourceReaderReady:       sw.SourceWeightedAdmissionResonanceGraftCandidateStoreReaderReady,
		SourceStoreReady:        sw.SourceWeightedAdmissionResonanceGraftCandidateStoreReady,
		SourceCandidateReady:    sw.SourceWeightedAdmissionResonanceGraftCandidateReady,
		SourceGateReady:         sw.SourceWeightedAdmissionResonanceGraftGateReady,
		SourcePreflightReady:    sw.SourceWeightedAdmissionResonanceGraftPreflightReady,
		SourceBoundaryReady:     sw.SourceWeightedAdmissionResonanceGraftBoundaryReady,
		SourceObservationReady:  sw.SourceWeightedAdmissionResonanceObservationReady,
		SourceReceiverReady:     sw.SourceWeightedAdmissionResonanceReceiverReady,
		SourceIntentReady:       sw.SourceWeightedAdmissionResonanceIntentReady,
		SourceFinalGateReady:    sw.SourceWeightedAdmissionFinalGateReady,
		SourceSealReady:         sw.SourceWeightedAdmissionSealReady,
		SourcePermitReady:       sw.SourceWeightedAdmissionPermitReady,
		SourceAuthorityUsed:     sw.SourceWeightedAdmissionAuthorityConsumed,
		SourceAuthorityNeeded:   sw.SourceWeightedAdmissionAuthorityRequired,
	})
	if h == "" {
		return ""
	}
	return "weighted-resonance-graft-admission-live-stage-id-" + h
}

func readAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionLiveStageReportForAssert(path string) (admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionLiveStageReport, map[string]json.RawMessage, error) {
	var report admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionLiveStageReport
	if strings.TrimSpace(path) == "" {
		return report, nil, fmt.Errorf("weighted admission resonance graft admission live stage path missing")
	}
	raw, err := os.ReadFile(path)
	if err != nil {
		if os.IsNotExist(err) {
			return report, nil, fmt.Errorf("weighted admission resonance graft admission live stage not written")
		}
		return report, nil, err
	}
	if len(raw) == 0 {
		return report, nil, fmt.Errorf("weighted admission resonance graft admission live stage not written")
	}
	var root map[string]json.RawMessage
	if err := json.Unmarshal(raw, &root); err != nil {
		return report, nil, fmt.Errorf("weighted admission resonance graft admission live stage JSON invalid: %w", err)
	}
	if err := json.Unmarshal(raw, &report); err != nil {
		return report, nil, fmt.Errorf("weighted admission resonance graft admission live stage decode failed: %w", err)
	}
	return report, root, nil
}
