package main

import (
	"encoding/json"
	"fmt"
	"os"
	"strings"
)

const admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionEnableGateSchema = "arianna.live_route_weighted_admission_resonance_graft_admission_enable_gate.v1"

type admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionEnableGateReport struct {
	Schema                                                               string `json:"schema"`
	Status                                                               string `json:"status"`
	Target                                                               string `json:"target"`
	TargetKind                                                           string `json:"target_kind"`
	TargetMode                                                           string `json:"target_mode"`
	Action                                                               string `json:"action"`
	EnableState                                                          string `json:"enable_state"`
	EnableAction                                                         string `json:"enable_action"`
	SwitchState                                                          string `json:"switch_state"`
	SwitchAction                                                         string `json:"switch_action"`
	Promotion                                                            string `json:"promotion"`
	WeightedAdmissionResonanceGraftAdmissionEnableGateReady              bool   `json:"weighted_admission_resonance_graft_admission_enable_gate_ready"`
	WeightedAdmissionResonanceGraftAdmissionSwitchConsumed               bool   `json:"weighted_admission_resonance_graft_admission_switch_consumed"`
	WeightedAdmissionResonanceGraftAdmissionSwitchRequired               bool   `json:"weighted_admission_resonance_graft_admission_switch_required"`
	NextStepBlockedWithoutResonanceGraftAdmissionEnableGate              bool   `json:"next_step_blocked_without_resonance_graft_admission_enable_gate"`
	WeightedAdmissionResonanceGraftAdmissionEnableGateID                 string `json:"weighted_admission_resonance_graft_admission_enable_gate_id"`
	ReceiptShape                                                         string `json:"receipt_shape"`
	EnableGateKind                                                       string `json:"enable_gate_kind"`
	EnableGateMode                                                       string `json:"enable_gate_mode"`
	EnableGateStage                                                      string `json:"enable_gate_stage"`
	CausalID                                                             string `json:"causal_id"`
	EnableGateHash                                                       string `json:"enable_gate_hash"`
	ReadBackHash                                                         string `json:"read_back_hash"`
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
	RollbackRequired                                                     bool   `json:"rollback_required"`
	ReadOnly                                                             bool   `json:"read_only"`
	ReplayOnly                                                           bool   `json:"replay_only"`
	SourceSchema                                                         string `json:"source_schema"`
	SourceStatus                                                         string `json:"source_status"`
	SourceTarget                                                         string `json:"source_target"`
	SourceReport                                                         string `json:"source_report"`
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

func runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionEnableGate(args []string) error {
	if len(args) != 2 {
		return fmt.Errorf("usage: --admission-live-route-weighted-admission-resonance-graft-admission-enable-gate RESONANCE_GRAFT_ADMISSION_SWITCH_REPORT RESONANCE_GRAFT_ADMISSION_ENABLE_GATE_REPORT")
	}
	switchPath := args[0]
	outputPath := args[1]
	if strings.TrimSpace(outputPath) == "" {
		return fmt.Errorf("weighted admission resonance graft admission enable gate output path missing")
	}
	sourceSwitch, root, err := readAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionSwitchReportForAssert(switchPath)
	if err != nil {
		return err
	}
	if err := admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionSwitchReportError(sourceSwitch, root); err != nil {
		return err
	}
	gate := admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionEnableGateReport{
		Schema:          admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionEnableGateSchema,
		Status:          "shadow_graft_admission_enable_gate_disabled_dry_run",
		Target:          "live_route_admission_next_step",
		TargetKind:      "weighted_internal_world_shadow_graft_admission_enable_gate",
		TargetMode:      "closed_enable_gate_dry_run",
		Action:          "hold_weighted_resonance_shadow_graft_admission_switch_disabled_dry_run",
		EnableState:     "disabled",
		EnableAction:    "require_operator_key",
		SwitchState:     sourceSwitch.SwitchState,
		SwitchAction:    sourceSwitch.SwitchAction,
		Promotion:       sourceSwitch.Promotion,
		ReceiptShape:    "weighted_resonance_shadow_graft_admission_enable_gate_receipt",
		EnableGateKind:  "shadow_graft_admission_enable_gate",
		EnableGateMode:  "closed_switch_enable_guard",
		EnableGateStage: "pre_live_graft_admission_enable_gate",
		WeightedAdmissionResonanceGraftAdmissionEnableGateReady: true,
		WeightedAdmissionResonanceGraftAdmissionSwitchConsumed:  true,
		WeightedAdmissionResonanceGraftAdmissionSwitchRequired:  true,
		NextStepBlockedWithoutResonanceGraftAdmissionEnableGate: true,
		SwitchVerified:               true,
		SwitchHashVerified:           true,
		SwitchReadBackVerified:       true,
		PromotionVerified:            sourceSwitch.PromotionVerified,
		PromotionHashVerified:        sourceSwitch.PromotionHashVerified,
		PromotionReadBackVerified:    sourceSwitch.PromotionReadBackVerified,
		DecisionVerified:             sourceSwitch.DecisionVerified,
		DecisionHashVerified:         sourceSwitch.DecisionHashVerified,
		DecisionReadBackVerified:     sourceSwitch.DecisionReadBackVerified,
		ProofPreconditionVerified:    sourceSwitch.ProofPreconditionVerified,
		PreconditionHashVerified:     sourceSwitch.PreconditionHashVerified,
		PreconditionReadBackVerified: sourceSwitch.PreconditionReadBackVerified,
		ProofVerified:                sourceSwitch.ProofVerified,
		ProofHashVerified:            sourceSwitch.ProofHashVerified,
		ProofReadBackVerified:        sourceSwitch.ProofReadBackVerified,
		StoreReaderVerified:          sourceSwitch.StoreReaderVerified,
		StoreVerified:                sourceSwitch.StoreVerified,
		CandidateVerified:            sourceSwitch.CandidateVerified,
		GateVerified:                 sourceSwitch.GateVerified,
		PreflightVerified:            sourceSwitch.PreflightVerified,
		BoundaryVerified:             sourceSwitch.BoundaryVerified,
		ObservationVerified:          sourceSwitch.ObservationVerified,
		ReceiverVerified:             sourceSwitch.ReceiverVerified,
		IntentVerified:               sourceSwitch.IntentVerified,
		FinalGateVerified:            sourceSwitch.FinalGateVerified,
		SealVerified:                 sourceSwitch.SealVerified,
		PermitVerified:               sourceSwitch.PermitVerified,
		AuthorityVerified:            sourceSwitch.AuthorityVerified,
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
		RollbackRequired:             true,
		ReadOnly:                     true,
		ReplayOnly:                   true,
		SourceSchema:                 sourceSwitch.Schema,
		SourceStatus:                 sourceSwitch.Status,
		SourceTarget:                 sourceSwitch.Target,
		SourceReport:                 switchPath,
		SourceWeightedAdmissionResonanceGraftAdmissionSwitchID:       sourceSwitch.WeightedAdmissionResonanceGraftAdmissionSwitchID,
		SourceWeightedAdmissionResonanceGraftAdmissionSwitchReady:    sourceSwitch.WeightedAdmissionResonanceGraftAdmissionSwitchReady,
		SourceWeightedAdmissionResonanceGraftAdmissionSwitchCausalID: sourceSwitch.CausalID,
		SourceWeightedAdmissionResonanceGraftAdmissionSwitchHash:     sourceSwitch.SwitchHash,
		SourceWeightedAdmissionResonanceGraftAdmissionSwitchReadBack: sourceSwitch.ReadBackHash,
		SourceSwitchState:                                            sourceSwitch.SwitchState,
		SourceSwitchAction:                                           sourceSwitch.SwitchAction,
		SourceSwitchReceiptShape:                                     sourceSwitch.ReceiptShape,
		SourceSwitchKind:                                             sourceSwitch.SwitchKind,
		SourceSwitchMode:                                             sourceSwitch.SwitchMode,
		SourceSwitchStage:                                            sourceSwitch.SwitchStage,
		SourceSwitchAdmissionRequired:                                sourceSwitch.AdmissionRequired,
		SourceSwitchShadowOnly:                                       sourceSwitch.ShadowOnly,
		SourceSwitchGraftAllowed:                                     sourceSwitch.GraftAllowed,
		SourceSwitchDryRunOnly:                                       sourceSwitch.DryRunOnly,
		SourceSwitchLiveReady:                                        sourceSwitch.LiveReady,
		SourceSwitchRawDreamTextAllowed:                              sourceSwitch.RawDreamTextAllowed,
		SourceSwitchRawDreamTextObserved:                             sourceSwitch.RawDreamTextObserved,
		SourceSwitchRawDreamTextForwarded:                            sourceSwitch.RawDreamTextForwarded,
		SourceSwitchJanusSurfaceAllowed:                              sourceSwitch.JanusSurfaceAllowed,
		SourceSwitchCoocLearningAllowed:                              sourceSwitch.CoocLearningAllowed,
		SourceSwitchDeltaHarvestAllowed:                              sourceSwitch.DeltaHarvestAllowed,
		SourceSwitchBodyMutationAllowed:                              sourceSwitch.BodyMutationAllowed,
		SourceSwitchRollbackRequired:                                 sourceSwitch.RollbackRequired,
		SourceSwitchReadOnly:                                         sourceSwitch.ReadOnly,
		SourceSwitchReplayOnly:                                       sourceSwitch.ReplayOnly,
		SourceSwitchWriteAllowed:                                     sourceSwitch.WriteAllowed,
		SourceSwitchAdmissionAllowed:                                 sourceSwitch.AdmissionAllowed,
		SourceSwitchLiveAdmissionEnabled:                             sourceSwitch.LiveAdmissionEnabled,
		SourceSwitchMutatesState:                                     sourceSwitch.MutatesState,
		SourceSwitchBodyTarget:                                       sourceSwitch.BodyTarget,
		SourceSwitchPassed:                                           sourceSwitch.Passed,
		SourceSwitchReason:                                           sourceSwitch.Reason,
		SourceWeightedAdmissionResonanceGraftAdmissionPromotionID:    sourceSwitch.SourceWeightedAdmissionResonanceGraftAdmissionPromotionID,
		SourceWeightedAdmissionResonanceGraftAdmissionPromotionReady: sourceSwitch.SourceWeightedAdmissionResonanceGraftAdmissionPromotionReady,
		SourceWeightedAdmissionResonanceGraftAdmissionPromotionCausalID: sourceSwitch.SourceWeightedAdmissionResonanceGraftAdmissionPromotionCausalID,
		SourceWeightedAdmissionResonanceGraftAdmissionPromotionHash:     sourceSwitch.SourceWeightedAdmissionResonanceGraftAdmissionPromotionHash,
		SourceWeightedAdmissionResonanceGraftAdmissionPromotionReadBack: sourceSwitch.SourceWeightedAdmissionResonanceGraftAdmissionPromotionReadBack,
		SourcePromotion:                                                      sourceSwitch.SourcePromotion,
		SourcePromotionAction:                                                sourceSwitch.SourcePromotionAction,
		SourcePromotionReceiptShape:                                          sourceSwitch.SourcePromotionReceiptShape,
		SourcePromotionKind:                                                  sourceSwitch.SourcePromotionKind,
		SourcePromotionMode:                                                  sourceSwitch.SourcePromotionMode,
		SourcePromotionStage:                                                 sourceSwitch.SourcePromotionStage,
		SourcePromotionAdmissionRequired:                                     sourceSwitch.SourcePromotionAdmissionRequired,
		SourcePromotionShadowOnly:                                            sourceSwitch.SourcePromotionShadowOnly,
		SourcePromotionGraftAllowed:                                          sourceSwitch.SourcePromotionGraftAllowed,
		SourcePromotionDryRunOnly:                                            sourceSwitch.SourcePromotionDryRunOnly,
		SourcePromotionLiveReady:                                             sourceSwitch.SourcePromotionLiveReady,
		SourcePromotionRawDreamTextAllowed:                                   sourceSwitch.SourcePromotionRawDreamTextAllowed,
		SourcePromotionRawDreamTextObserved:                                  sourceSwitch.SourcePromotionRawDreamTextObserved,
		SourcePromotionRawDreamTextForwarded:                                 sourceSwitch.SourcePromotionRawDreamTextForwarded,
		SourcePromotionJanusSurfaceAllowed:                                   sourceSwitch.SourcePromotionJanusSurfaceAllowed,
		SourcePromotionCoocLearningAllowed:                                   sourceSwitch.SourcePromotionCoocLearningAllowed,
		SourcePromotionDeltaHarvestAllowed:                                   sourceSwitch.SourcePromotionDeltaHarvestAllowed,
		SourcePromotionBodyMutationAllowed:                                   sourceSwitch.SourcePromotionBodyMutationAllowed,
		SourcePromotionRollbackRequired:                                      sourceSwitch.SourcePromotionRollbackRequired,
		SourcePromotionReadOnly:                                              sourceSwitch.SourcePromotionReadOnly,
		SourcePromotionReplayOnly:                                            sourceSwitch.SourcePromotionReplayOnly,
		SourcePromotionWriteAllowed:                                          sourceSwitch.SourcePromotionWriteAllowed,
		SourcePromotionAdmissionAllowed:                                      sourceSwitch.SourcePromotionAdmissionAllowed,
		SourcePromotionLiveAdmissionEnabled:                                  sourceSwitch.SourcePromotionLiveAdmissionEnabled,
		SourcePromotionMutatesState:                                          sourceSwitch.SourcePromotionMutatesState,
		SourcePromotionBodyTarget:                                            sourceSwitch.SourcePromotionBodyTarget,
		SourcePromotionPassed:                                                sourceSwitch.SourcePromotionPassed,
		SourcePromotionReason:                                                sourceSwitch.SourcePromotionReason,
		SourceWeightedAdmissionResonanceGraftAdmissionDecisionID:             sourceSwitch.SourceWeightedAdmissionResonanceGraftAdmissionDecisionID,
		SourceWeightedAdmissionResonanceGraftAdmissionDecisionReady:          sourceSwitch.SourceWeightedAdmissionResonanceGraftAdmissionDecisionReady,
		SourceWeightedAdmissionResonanceGraftAdmissionProofPreconditionID:    sourceSwitch.SourceWeightedAdmissionResonanceGraftAdmissionProofPreconditionID,
		SourceWeightedAdmissionResonanceGraftAdmissionProofPreconditionReady: sourceSwitch.SourceWeightedAdmissionResonanceGraftAdmissionProofPreconditionReady,
		SourceWeightedAdmissionResonanceGraftAdmissionProofID:                sourceSwitch.SourceWeightedAdmissionResonanceGraftAdmissionProofID,
		SourceWeightedAdmissionResonanceGraftAdmissionProofReady:             sourceSwitch.SourceWeightedAdmissionResonanceGraftAdmissionProofReady,
		SourceWeightedAdmissionResonanceGraftCandidateStoreReaderID:          sourceSwitch.SourceWeightedAdmissionResonanceGraftCandidateStoreReaderID,
		SourceWeightedAdmissionResonanceGraftCandidateStoreReaderReady:       sourceSwitch.SourceWeightedAdmissionResonanceGraftCandidateStoreReaderReady,
		SourceWeightedAdmissionResonanceGraftCandidateStoreID:                sourceSwitch.SourceWeightedAdmissionResonanceGraftCandidateStoreID,
		SourceWeightedAdmissionResonanceGraftCandidateStoreReady:             sourceSwitch.SourceWeightedAdmissionResonanceGraftCandidateStoreReady,
		SourceWeightedAdmissionResonanceGraftCandidateID:                     sourceSwitch.SourceWeightedAdmissionResonanceGraftCandidateID,
		SourceWeightedAdmissionResonanceGraftCandidateReady:                  sourceSwitch.SourceWeightedAdmissionResonanceGraftCandidateReady,
		SourceWeightedAdmissionResonanceGraftGateID:                          sourceSwitch.SourceWeightedAdmissionResonanceGraftGateID,
		SourceWeightedAdmissionResonanceGraftGateReady:                       sourceSwitch.SourceWeightedAdmissionResonanceGraftGateReady,
		SourceWeightedAdmissionResonanceGraftPreflightID:                     sourceSwitch.SourceWeightedAdmissionResonanceGraftPreflightID,
		SourceWeightedAdmissionResonanceGraftPreflightReady:                  sourceSwitch.SourceWeightedAdmissionResonanceGraftPreflightReady,
		SourceWeightedAdmissionResonanceGraftBoundaryID:                      sourceSwitch.SourceWeightedAdmissionResonanceGraftBoundaryID,
		SourceWeightedAdmissionResonanceGraftBoundaryReady:                   sourceSwitch.SourceWeightedAdmissionResonanceGraftBoundaryReady,
		SourceWeightedAdmissionResonanceObservationID:                        sourceSwitch.SourceWeightedAdmissionResonanceObservationID,
		SourceWeightedAdmissionResonanceObservationReady:                     sourceSwitch.SourceWeightedAdmissionResonanceObservationReady,
		SourceWeightedAdmissionResonanceReceiverID:                           sourceSwitch.SourceWeightedAdmissionResonanceReceiverID,
		SourceWeightedAdmissionResonanceReceiverReady:                        sourceSwitch.SourceWeightedAdmissionResonanceReceiverReady,
		SourceWeightedAdmissionResonanceIntentReady:                          sourceSwitch.SourceWeightedAdmissionResonanceIntentReady,
		SourceWeightedAdmissionFinalGateReady:                                sourceSwitch.SourceWeightedAdmissionFinalGateReady,
		SourceWeightedAdmissionSealReady:                                     sourceSwitch.SourceWeightedAdmissionSealReady,
		SourceWeightedAdmissionPermitReady:                                   sourceSwitch.SourceWeightedAdmissionPermitReady,
		SourceWeightedAdmissionAuthorityConsumed:                             sourceSwitch.SourceWeightedAdmissionAuthorityConsumed,
		SourceWeightedAdmissionAuthorityRequired:                             sourceSwitch.SourceWeightedAdmissionAuthorityRequired,
		BodySmokeWeighted:                                                    sourceSwitch.BodySmokeWeighted,
		NanoDirectRunner:                                                     sourceSwitch.NanoDirectRunner,
		NanoDirectFinalGate:                                                  sourceSwitch.NanoDirectFinalGate,
		ResonanceGraftAdmissionProof:                                         sourceSwitch.ResonanceGraftAdmissionProof,
		BoundaryReportFullChain:                                              sourceSwitch.BoundaryReportFullChain,
		SourceAuthorityGranted:                                               sourceSwitch.SourceAuthorityGranted,
		AuthorityGranted:                                                     false,
		ContractsReady:                                                       false,
		WriteAllowed:                                                         false,
		AdmissionAllowed:                                                     false,
		LiveAdmissionEnabled:                                                 false,
		MutatesState:                                                         false,
		BodyTarget:                                                           "none",
		Passed:                                                               true,
		Reason:                                                               "weighted resonance shadow graft admission enable gate closed; operator key absent and mutation refused",
	}
	gate.CausalID = admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionEnableGateCausalID(gate)
	gate.EnableGateHash = admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionEnableGateHash(gate)
	gate.ReadBackHash = admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionEnableGateReadBackHash(gate)
	gate.WeightedAdmissionResonanceGraftAdmissionEnableGateID = admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionEnableGateID(gate)
	if gate.CausalID == "" ||
		gate.EnableGateHash == "" ||
		gate.ReadBackHash == "" ||
		gate.WeightedAdmissionResonanceGraftAdmissionEnableGateID == "" ||
		gate.EnableGateHash == gate.ReadBackHash {
		return fmt.Errorf("weighted admission resonance graft admission enable gate read-back proof failed")
	}
	raw, err := json.MarshalIndent(gate, "", "  ")
	if err != nil {
		return fmt.Errorf("weighted admission resonance graft admission enable gate marshal failed: %w", err)
	}
	raw = append(raw, '\n')
	if err := os.WriteFile(outputPath, raw, 0600); err != nil {
		return fmt.Errorf("weighted admission resonance graft admission enable gate write failed: %w", err)
	}
	fmt.Printf("[admission-live-route-weighted-admission-resonance-graft-admission-enable-gate] pass: resonance_graft_admission_enable_gate_report=%s resonance_graft_admission_switch_report=%s\n", outputPath, switchPath)
	return nil
}

func runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionEnableGateAssert(args []string) error {
	if len(args) != 1 {
		return fmt.Errorf("usage: --admission-live-route-weighted-admission-resonance-graft-admission-enable-gate-assert REPORT")
	}
	report, root, err := readAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionEnableGateReportForAssert(args[0])
	if err != nil {
		return err
	}
	return admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionEnableGateReportError(report, root)
}

func admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionEnableGateReportError(report admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionEnableGateReport, root map[string]json.RawMessage) error {
	if _, ok := root["schema"]; !ok {
		return fmt.Errorf("weighted admission resonance graft admission enable gate schema missing")
	}
	if report.Schema != admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionEnableGateSchema {
		return fmt.Errorf("weighted admission resonance graft admission enable gate schema mismatch: got %q want %q", report.Schema, admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionEnableGateSchema)
	}
	if report.Status != "shadow_graft_admission_enable_gate_disabled_dry_run" {
		return fmt.Errorf("weighted admission resonance graft admission enable gate status mismatch: got %q want %q", report.Status, "shadow_graft_admission_enable_gate_disabled_dry_run")
	}
	if report.Target != "live_route_admission_next_step" {
		return fmt.Errorf("weighted admission resonance graft admission enable gate target mismatch: got %q want %q", report.Target, "live_route_admission_next_step")
	}
	if report.TargetKind != "weighted_internal_world_shadow_graft_admission_enable_gate" {
		return fmt.Errorf("weighted admission resonance graft admission enable gate target_kind mismatch: got %q want %q", report.TargetKind, "weighted_internal_world_shadow_graft_admission_enable_gate")
	}
	if report.TargetMode != "closed_enable_gate_dry_run" {
		return fmt.Errorf("weighted admission resonance graft admission enable gate target_mode mismatch: got %q want %q", report.TargetMode, "closed_enable_gate_dry_run")
	}
	if report.Action != "hold_weighted_resonance_shadow_graft_admission_switch_disabled_dry_run" {
		return fmt.Errorf("weighted admission resonance graft admission enable gate action mismatch: got %q want %q", report.Action, "hold_weighted_resonance_shadow_graft_admission_switch_disabled_dry_run")
	}
	if report.EnableState != "disabled" {
		return fmt.Errorf("weighted admission resonance graft admission enable gate enable_state mismatch: got %q want %q", report.EnableState, "disabled")
	}
	if report.EnableAction != "require_operator_key" {
		return fmt.Errorf("weighted admission resonance graft admission enable gate enable_action mismatch: got %q want %q", report.EnableAction, "require_operator_key")
	}
	if report.SwitchState != "disabled" {
		return fmt.Errorf("weighted admission resonance graft admission enable gate switch_state mismatch: got %q want %q", report.SwitchState, "disabled")
	}
	if report.SwitchAction != "hold_pending_live_admission" {
		return fmt.Errorf("weighted admission resonance graft admission enable gate switch_action mismatch: got %q want %q", report.SwitchAction, "hold_pending_live_admission")
	}
	if report.Promotion != "pending_live_admission" {
		return fmt.Errorf("weighted admission resonance graft admission enable gate promotion mismatch: got %q want %q", report.Promotion, "pending_live_admission")
	}
	if report.ReceiptShape != "weighted_resonance_shadow_graft_admission_enable_gate_receipt" {
		return fmt.Errorf("weighted admission resonance graft admission enable gate receipt_shape mismatch: got %q want %q", report.ReceiptShape, "weighted_resonance_shadow_graft_admission_enable_gate_receipt")
	}
	if report.EnableGateKind != "shadow_graft_admission_enable_gate" ||
		report.EnableGateMode != "closed_switch_enable_guard" ||
		report.EnableGateStage != "pre_live_graft_admission_enable_gate" {
		return fmt.Errorf("weighted admission resonance graft admission enable gate shape mismatch")
	}
	for _, required := range []struct {
		name  string
		value bool
	}{
		{"weighted_admission_resonance_graft_admission_enable_gate_ready", report.WeightedAdmissionResonanceGraftAdmissionEnableGateReady},
		{"weighted_admission_resonance_graft_admission_switch_consumed", report.WeightedAdmissionResonanceGraftAdmissionSwitchConsumed},
		{"weighted_admission_resonance_graft_admission_switch_required", report.WeightedAdmissionResonanceGraftAdmissionSwitchRequired},
		{"next_step_blocked_without_resonance_graft_admission_enable_gate", report.NextStepBlockedWithoutResonanceGraftAdmissionEnableGate},
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
		{"rollback_required", report.RollbackRequired},
		{"read_only", report.ReadOnly},
		{"replay_only", report.ReplayOnly},
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
			return fmt.Errorf("weighted admission resonance graft admission enable gate %s not ready", required.name)
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
			return fmt.Errorf("weighted admission resonance graft admission enable gate opened %s", closed.name)
		}
	}
	for _, requiredString := range []struct {
		name  string
		value string
	}{
		{"weighted_admission_resonance_graft_admission_enable_gate_id", report.WeightedAdmissionResonanceGraftAdmissionEnableGateID},
		{"causal_id", report.CausalID},
		{"enable_gate_hash", report.EnableGateHash},
		{"read_back_hash", report.ReadBackHash},
		{"source_report", report.SourceReport},
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
			return fmt.Errorf("weighted admission resonance graft admission enable gate %s missing", requiredString.name)
		}
	}
	if report.SourceSchema != admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionSwitchSchema {
		return fmt.Errorf("weighted admission resonance graft admission enable gate source_schema mismatch: got %q want %q", report.SourceSchema, admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionSwitchSchema)
	}
	if report.SourceStatus != "shadow_graft_admission_switch_disabled_dry_run" {
		return fmt.Errorf("weighted admission resonance graft admission enable gate source_status mismatch: got %q want %q", report.SourceStatus, "shadow_graft_admission_switch_disabled_dry_run")
	}
	if report.SourceTarget != "live_route_admission_next_step" {
		return fmt.Errorf("weighted admission resonance graft admission enable gate source_target mismatch: got %q want %q", report.SourceTarget, "live_route_admission_next_step")
	}
	if report.SourceSwitchState != "disabled" {
		return fmt.Errorf("weighted admission resonance graft admission enable gate source_switch_state mismatch: got %q want %q", report.SourceSwitchState, "disabled")
	}
	if report.SourceSwitchAction != "hold_pending_live_admission" {
		return fmt.Errorf("weighted admission resonance graft admission enable gate source_switch_action mismatch: got %q want %q", report.SourceSwitchAction, "hold_pending_live_admission")
	}
	if report.SwitchState != report.SourceSwitchState || report.SwitchAction != report.SourceSwitchAction {
		return fmt.Errorf("weighted admission resonance graft admission enable gate source switch state/action not carried")
	}
	if report.SourceSwitchReceiptShape != "weighted_resonance_shadow_graft_admission_switch_receipt" ||
		report.SourceSwitchKind != "shadow_graft_admission_switch" ||
		report.SourceSwitchMode != "closed_promotion_switch_guard" ||
		report.SourceSwitchStage != "pre_live_graft_admission_switch" {
		return fmt.Errorf("weighted admission resonance graft admission enable gate source switch shape mismatch")
	}
	if report.SourceSwitchBodyTarget != "none" {
		return fmt.Errorf("weighted admission resonance graft admission enable gate source_switch_body_target mismatch: got %q want %q", report.SourceSwitchBodyTarget, "none")
	}
	if report.SourceSwitchReason != "weighted resonance shadow graft admission promotion held at disabled switch without mutation" {
		return fmt.Errorf("weighted admission resonance graft admission enable gate source_switch_reason mismatch: got %q", report.SourceSwitchReason)
	}
	if report.SourcePromotion != "pending_live_admission" {
		return fmt.Errorf("weighted admission resonance graft admission enable gate source_promotion mismatch: got %q want %q", report.SourcePromotion, "pending_live_admission")
	}
	if report.SourcePromotionAction != "promote_weighted_resonance_shadow_graft_admission_dry_run" {
		return fmt.Errorf("weighted admission resonance graft admission enable gate source_promotion_action mismatch: got %q want %q", report.SourcePromotionAction, "promote_weighted_resonance_shadow_graft_admission_dry_run")
	}
	if report.SourcePromotionReceiptShape != "weighted_resonance_shadow_graft_admission_promotion_receipt" ||
		report.SourcePromotionKind != "shadow_graft_admission_promotion" ||
		report.SourcePromotionMode != "closed_decision_promotion" ||
		report.SourcePromotionStage != "pre_live_graft_admission_promotion" {
		return fmt.Errorf("weighted admission resonance graft admission enable gate source promotion shape mismatch")
	}
	if report.SourcePromotionBodyTarget != "none" {
		return fmt.Errorf("weighted admission resonance graft admission enable gate source_promotion_body_target mismatch: got %q want %q", report.SourcePromotionBodyTarget, "none")
	}
	if report.BodyTarget != "none" {
		return fmt.Errorf("weighted admission resonance graft admission enable gate body_target mismatch: got %q want %q", report.BodyTarget, "none")
	}
	if !strings.HasPrefix(report.WeightedAdmissionResonanceGraftAdmissionEnableGateID, "weighted-resonance-graft-admission-enable-gate-id-") {
		return fmt.Errorf("weighted admission resonance graft admission enable gate id prefix mismatch")
	}
	if !strings.HasPrefix(report.CausalID, "weighted-resonance-graft-admission-enable-gate-causal-") {
		return fmt.Errorf("weighted admission resonance graft admission enable gate causal prefix mismatch")
	}
	if !strings.HasPrefix(report.EnableGateHash, "weighted-resonance-graft-admission-enable-gate-") {
		return fmt.Errorf("weighted admission resonance graft admission enable gate hash prefix mismatch")
	}
	if !strings.HasPrefix(report.ReadBackHash, "weighted-resonance-graft-admission-enable-gate-read-") ||
		report.EnableGateHash == report.ReadBackHash {
		return fmt.Errorf("weighted admission resonance graft admission enable gate read-back mismatch")
	}
	if !strings.HasPrefix(report.SourceWeightedAdmissionResonanceGraftAdmissionSwitchID, "weighted-resonance-graft-admission-switch-id-") ||
		!strings.HasPrefix(report.SourceWeightedAdmissionResonanceGraftAdmissionSwitchCausalID, "weighted-resonance-graft-admission-switch-causal-") ||
		!strings.HasPrefix(report.SourceWeightedAdmissionResonanceGraftAdmissionSwitchHash, "weighted-resonance-graft-admission-switch-") ||
		!strings.HasPrefix(report.SourceWeightedAdmissionResonanceGraftAdmissionSwitchReadBack, "weighted-resonance-graft-admission-switch-read-") {
		return fmt.Errorf("weighted admission resonance graft admission enable gate source switch mismatch")
	}
	if !strings.HasPrefix(report.SourceWeightedAdmissionResonanceGraftAdmissionPromotionID, "weighted-resonance-graft-admission-promotion-id-") ||
		!strings.HasPrefix(report.SourceWeightedAdmissionResonanceGraftAdmissionPromotionCausalID, "weighted-resonance-graft-admission-promotion-causal-") ||
		!strings.HasPrefix(report.SourceWeightedAdmissionResonanceGraftAdmissionPromotionHash, "weighted-resonance-graft-admission-promotion-") ||
		!strings.HasPrefix(report.SourceWeightedAdmissionResonanceGraftAdmissionPromotionReadBack, "weighted-resonance-graft-admission-promotion-read-") {
		return fmt.Errorf("weighted admission resonance graft admission enable gate source promotion mismatch")
	}
	if !strings.HasPrefix(report.SourceWeightedAdmissionResonanceGraftAdmissionDecisionID, "weighted-resonance-graft-admission-decision-id-") {
		return fmt.Errorf("weighted admission resonance graft admission enable gate source decision id prefix mismatch")
	}
	if !strings.HasPrefix(report.SourceWeightedAdmissionResonanceGraftAdmissionProofPreconditionID, "weighted-resonance-graft-admission-proof-precondition-id-") {
		return fmt.Errorf("weighted admission resonance graft admission enable gate source precondition id prefix mismatch")
	}
	if !strings.HasPrefix(report.SourceWeightedAdmissionResonanceGraftAdmissionProofID, "weighted-resonance-graft-admission-proof-id-") {
		return fmt.Errorf("weighted admission resonance graft admission enable gate source proof id prefix mismatch")
	}
	if !strings.HasPrefix(report.SourceWeightedAdmissionResonanceGraftCandidateStoreReaderID, "weighted-resonance-graft-candidate-store-reader-id-") {
		return fmt.Errorf("weighted admission resonance graft admission enable gate source reader id prefix mismatch")
	}
	if !strings.HasPrefix(report.SourceWeightedAdmissionResonanceGraftCandidateStoreID, "weighted-resonance-graft-candidate-store-id-") {
		return fmt.Errorf("weighted admission resonance graft admission enable gate source store id prefix mismatch")
	}
	if !strings.HasPrefix(report.SourceWeightedAdmissionResonanceGraftCandidateID, "weighted-resonance-graft-candidate-id-") {
		return fmt.Errorf("weighted admission resonance graft admission enable gate source candidate id prefix mismatch")
	}
	if !strings.HasPrefix(report.SourceWeightedAdmissionResonanceGraftGateID, "weighted-resonance-graft-gate-id-") {
		return fmt.Errorf("weighted admission resonance graft admission enable gate source gate id prefix mismatch")
	}
	if !strings.HasPrefix(report.SourceWeightedAdmissionResonanceGraftPreflightID, "weighted-resonance-graft-preflight-id-") {
		return fmt.Errorf("weighted admission resonance graft admission enable gate source preflight id prefix mismatch")
	}
	if !strings.HasPrefix(report.SourceWeightedAdmissionResonanceGraftBoundaryID, "weighted-resonance-graft-boundary-id-") {
		return fmt.Errorf("weighted admission resonance graft admission enable gate source boundary id prefix mismatch")
	}
	if !strings.HasPrefix(report.SourceWeightedAdmissionResonanceObservationID, "weighted-resonance-observation-") {
		return fmt.Errorf("weighted admission resonance graft admission enable gate source observation id prefix mismatch")
	}
	if !strings.HasPrefix(report.SourceWeightedAdmissionResonanceReceiverID, "weighted-resonance-receiver-") {
		return fmt.Errorf("weighted admission resonance graft admission enable gate source receiver id prefix mismatch")
	}
	if report.CausalID == "" || report.CausalID != admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionEnableGateCausalID(report) {
		return fmt.Errorf("weighted admission resonance graft admission enable gate causal_id mismatch")
	}
	if report.EnableGateHash == "" || report.EnableGateHash != admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionEnableGateHash(report) {
		return fmt.Errorf("weighted admission resonance graft admission enable gate enable_gate_hash mismatch")
	}
	if report.ReadBackHash == "" || report.ReadBackHash != admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionEnableGateReadBackHash(report) {
		return fmt.Errorf("weighted admission resonance graft admission enable gate read_back_hash mismatch")
	}
	if report.EnableGateHash == report.ReadBackHash {
		return fmt.Errorf("weighted admission resonance graft admission enable gate read-back proof collapsed")
	}
	if report.WeightedAdmissionResonanceGraftAdmissionEnableGateID == "" || report.WeightedAdmissionResonanceGraftAdmissionEnableGateID != admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionEnableGateID(report) {
		return fmt.Errorf("weighted admission resonance graft admission enable gate id mismatch")
	}
	if report.Reason != "weighted resonance shadow graft admission enable gate closed; operator key absent and mutation refused" {
		return fmt.Errorf("weighted admission resonance graft admission enable gate reason mismatch: got %q", report.Reason)
	}
	return nil
}

func admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionEnableGateCausalID(sw admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionEnableGateReport) string {
	h := hashJSON(struct {
		SourceSwitchID    string `json:"source_switch_id"`
		SourceSwitchRead  string `json:"source_switch_read_back_hash"`
		SourcePromotionID string `json:"source_promotion_id"`
		Target            string `json:"target"`
		EnableGateKind    string `json:"enable_gate_kind"`
		EnableGateStage   string `json:"enable_gate_stage"`
	}{
		SourceSwitchID:    sw.SourceWeightedAdmissionResonanceGraftAdmissionSwitchID,
		SourceSwitchRead:  sw.SourceWeightedAdmissionResonanceGraftAdmissionSwitchReadBack,
		SourcePromotionID: sw.SourceWeightedAdmissionResonanceGraftAdmissionPromotionID,
		Target:            sw.Target,
		EnableGateKind:    sw.EnableGateKind,
		EnableGateStage:   sw.EnableGateStage,
	})
	if h == "" {
		return ""
	}
	return "weighted-resonance-graft-admission-enable-gate-causal-" + h
}

func admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionEnableGateHash(sw admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionEnableGateReport) string {
	h := hashJSON(struct {
		CausalID          string `json:"causal_id"`
		SourceSwitchID    string `json:"source_switch_id"`
		SourceSwitchHash  string `json:"source_switch_hash"`
		SourceSwitchRead  string `json:"source_switch_read_back_hash"`
		EnableState       string `json:"enable_state"`
		EnableAction      string `json:"enable_action"`
		SwitchState       string `json:"switch_state"`
		SwitchAction      string `json:"switch_action"`
		Promotion         string `json:"promotion"`
		Action            string `json:"action"`
		ReceiptShape      string `json:"receipt_shape"`
		EnableGateMode    string `json:"enable_gate_mode"`
		SwitchVerified    bool   `json:"switch_verified"`
		ReadOnly          bool   `json:"read_only"`
		ReplayOnly        bool   `json:"replay_only"`
		AdmissionRequired bool   `json:"admission_required"`
		ShadowOnly        bool   `json:"shadow_only"`
		DryRunOnly        bool   `json:"dry_run_only"`
		GraftAllowed      bool   `json:"graft_allowed"`
		BodyMutation      bool   `json:"body_mutation_allowed"`
		LiveAdmission     bool   `json:"live_admission_enabled"`
	}{
		CausalID:          sw.CausalID,
		SourceSwitchID:    sw.SourceWeightedAdmissionResonanceGraftAdmissionSwitchID,
		SourceSwitchHash:  sw.SourceWeightedAdmissionResonanceGraftAdmissionSwitchHash,
		SourceSwitchRead:  sw.SourceWeightedAdmissionResonanceGraftAdmissionSwitchReadBack,
		EnableState:       sw.EnableState,
		EnableAction:      sw.EnableAction,
		SwitchState:       sw.SwitchState,
		SwitchAction:      sw.SwitchAction,
		Promotion:         sw.Promotion,
		Action:            sw.Action,
		ReceiptShape:      sw.ReceiptShape,
		EnableGateMode:    sw.EnableGateMode,
		SwitchVerified:    sw.SwitchVerified,
		ReadOnly:          sw.ReadOnly,
		ReplayOnly:        sw.ReplayOnly,
		AdmissionRequired: sw.AdmissionRequired,
		ShadowOnly:        sw.ShadowOnly,
		DryRunOnly:        sw.DryRunOnly,
		GraftAllowed:      sw.GraftAllowed,
		BodyMutation:      sw.BodyMutationAllowed,
		LiveAdmission:     sw.LiveAdmissionEnabled,
	})
	if h == "" {
		return ""
	}
	return "weighted-resonance-graft-admission-enable-gate-" + h
}

func admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionEnableGateReadBackHash(sw admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionEnableGateReport) string {
	h := hashJSON(struct {
		EnableGateHash   string `json:"enable_gate_hash"`
		SourceSwitchID   string `json:"source_switch_id"`
		SourceSwitchRead string `json:"source_switch_read_back_hash"`
		EnableGateKind   string `json:"enable_gate_kind"`
		EnableGateReady  bool   `json:"enable_gate_ready"`
		SwitchConsumed   bool   `json:"switch_consumed"`
		LiveReady        bool   `json:"live_ready"`
		BodyMutation     bool   `json:"body_mutation"`
		LiveAdmission    bool   `json:"live_admission"`
		WriteAllowed     bool   `json:"write_allowed"`
		AdmissionAllowed bool   `json:"admission_allowed"`
	}{
		EnableGateHash:   sw.EnableGateHash,
		SourceSwitchID:   sw.SourceWeightedAdmissionResonanceGraftAdmissionSwitchID,
		SourceSwitchRead: sw.SourceWeightedAdmissionResonanceGraftAdmissionSwitchReadBack,
		EnableGateKind:   sw.EnableGateKind,
		EnableGateReady:  sw.WeightedAdmissionResonanceGraftAdmissionEnableGateReady,
		SwitchConsumed:   sw.WeightedAdmissionResonanceGraftAdmissionSwitchConsumed,
		LiveReady:        sw.LiveReady,
		BodyMutation:     sw.BodyMutationAllowed,
		LiveAdmission:    sw.LiveAdmissionEnabled,
		WriteAllowed:     sw.WriteAllowed,
		AdmissionAllowed: sw.AdmissionAllowed,
	})
	if h == "" {
		return ""
	}
	return "weighted-resonance-graft-admission-enable-gate-read-" + h
}

func admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionEnableGateID(sw admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionEnableGateReport) string {
	h := hashJSON(struct {
		Schema                  string `json:"schema"`
		Status                  string `json:"status"`
		Action                  string `json:"action"`
		EnableState             string `json:"enable_state"`
		EnableAction            string `json:"enable_action"`
		SwitchState             string `json:"switch_state"`
		SwitchAction            string `json:"switch_action"`
		Promotion               string `json:"promotion"`
		SourceReport            string `json:"source_report"`
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
		EnableGateHash          string `json:"enable_gate_hash"`
		ReadBackHash            string `json:"read_back_hash"`
		Ready                   bool   `json:"ready"`
		ReceiptShape            string `json:"receipt_shape"`
		EnableGateKind          string `json:"enable_gate_kind"`
		EnableGateMode          string `json:"enable_gate_mode"`
		EnableGateStage         string `json:"enable_gate_stage"`
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
		RollbackRequired        bool   `json:"rollback_required"`
		ReadOnly                bool   `json:"read_only"`
		ReplayOnly              bool   `json:"replay_only"`
		LiveReady               bool   `json:"live_ready"`
		ContractsReady          bool   `json:"contracts_ready"`
		BodyTarget              string `json:"body_target"`
		WriteAllowed            bool   `json:"write_allowed"`
		AdmissionAllowed        bool   `json:"admission_allowed"`
		LiveAdmissionEnabled    bool   `json:"live_admission_enabled"`
		MutatesState            bool   `json:"mutates_state"`
		NextStepBlockedWithout  bool   `json:"next_step_blocked_without_resonance_graft_admission_enable_gate"`
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
		EnableState:             sw.EnableState,
		EnableAction:            sw.EnableAction,
		SwitchState:             sw.SwitchState,
		SwitchAction:            sw.SwitchAction,
		Promotion:               sw.Promotion,
		SourceReport:            sw.SourceReport,
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
		EnableGateHash:          sw.EnableGateHash,
		ReadBackHash:            sw.ReadBackHash,
		Ready:                   sw.WeightedAdmissionResonanceGraftAdmissionEnableGateReady,
		ReceiptShape:            sw.ReceiptShape,
		EnableGateKind:          sw.EnableGateKind,
		EnableGateMode:          sw.EnableGateMode,
		EnableGateStage:         sw.EnableGateStage,
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
		RollbackRequired:        sw.RollbackRequired,
		ReadOnly:                sw.ReadOnly,
		ReplayOnly:              sw.ReplayOnly,
		LiveReady:               sw.LiveReady,
		ContractsReady:          sw.ContractsReady,
		BodyTarget:              sw.BodyTarget,
		WriteAllowed:            sw.WriteAllowed,
		AdmissionAllowed:        sw.AdmissionAllowed,
		LiveAdmissionEnabled:    sw.LiveAdmissionEnabled,
		MutatesState:            sw.MutatesState,
		NextStepBlockedWithout:  sw.NextStepBlockedWithoutResonanceGraftAdmissionEnableGate,
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
	return "weighted-resonance-graft-admission-enable-gate-id-" + h
}

func readAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionEnableGateReportForAssert(path string) (admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionEnableGateReport, map[string]json.RawMessage, error) {
	var report admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionEnableGateReport
	if strings.TrimSpace(path) == "" {
		return report, nil, fmt.Errorf("weighted admission resonance graft admission enable gate path missing")
	}
	raw, err := os.ReadFile(path)
	if err != nil {
		if os.IsNotExist(err) {
			return report, nil, fmt.Errorf("weighted admission resonance graft admission enable gate not written")
		}
		return report, nil, err
	}
	if len(raw) == 0 {
		return report, nil, fmt.Errorf("weighted admission resonance graft admission enable gate not written")
	}
	var root map[string]json.RawMessage
	if err := json.Unmarshal(raw, &root); err != nil {
		return report, nil, fmt.Errorf("weighted admission resonance graft admission enable gate JSON invalid: %w", err)
	}
	if err := json.Unmarshal(raw, &report); err != nil {
		return report, nil, fmt.Errorf("weighted admission resonance graft admission enable gate decode failed: %w", err)
	}
	return report, root, nil
}
