package main

import (
	"encoding/json"
	"fmt"
	"os"
	"strings"
)

const admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReceiverObservationBoundaryPreflightGateCandidateSchema = "arianna.live_route_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_final_gate_receiver_observation_boundary_preflight_gate_candidate.v1"

type admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReceiverObservationBoundaryPreflightGateCandidateReport struct {
	admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReceiverObservationBoundaryPreflightGateReport

	AdmissionFinalGateObservationBoundaryPreflightGateCandidateState                                                                                                                                                                string `json:"admission_final_gate_observation_boundary_preflight_gate_candidate_state"`
	AdmissionFinalGateObservationBoundaryPreflightGateCandidateAction                                                                                                                                                               string `json:"admission_final_gate_observation_boundary_preflight_gate_candidate_action"`
	AdmissionFinalGateObservationBoundaryPreflightGateCandidateTarget                                                                                                                                                               string `json:"admission_final_gate_observation_boundary_preflight_gate_candidate_target"`
	AdmissionFinalGateObservationBoundaryPreflightGateCandidateTargetKind                                                                                                                                                           string `json:"admission_final_gate_observation_boundary_preflight_gate_candidate_target_kind"`
	AdmissionFinalGateObservationBoundaryPreflightGateCandidateTargetMode                                                                                                                                                           string `json:"admission_final_gate_observation_boundary_preflight_gate_candidate_target_mode"`
	AdmissionFinalGateObservationBoundaryPreflightGateCandidateDryRunOnly                                                                                                                                                           bool   `json:"admission_final_gate_observation_boundary_preflight_gate_candidate_dry_run_only"`
	AdmissionFinalGateObservationBoundaryPreflightGateCandidateGateVerified                                                                                                                                                         bool   `json:"admission_final_gate_observation_boundary_preflight_gate_candidate_gate_verified"`
	AdmissionFinalGateObservationBoundaryPreflightGateCandidatePreflightVerified                                                                                                                                                    bool   `json:"admission_final_gate_observation_boundary_preflight_gate_candidate_preflight_verified"`
	AdmissionFinalGateObservationBoundaryPreflightGateCandidateBoundaryVerified                                                                                                                                                     bool   `json:"admission_final_gate_observation_boundary_preflight_gate_candidate_boundary_verified"`
	AdmissionFinalGateObservationBoundaryPreflightGateCandidateObservationVerified                                                                                                                                                  bool   `json:"admission_final_gate_observation_boundary_preflight_gate_candidate_observation_verified"`
	AdmissionFinalGateObservationBoundaryPreflightGateCandidateReadBackVerified                                                                                                                                                     bool   `json:"admission_final_gate_observation_boundary_preflight_gate_candidate_read_back_verified"`
	AdmissionFinalGateObservationBoundaryPreflightGateCandidateReady                                                                                                                                                                bool   `json:"admission_final_gate_observation_boundary_preflight_gate_candidate_ready"`
	FinalGateObservationBoundaryPreflightGateCandidateKind                                                                                                                                                                          string `json:"final_gate_observation_boundary_preflight_gate_candidate_kind"`
	FinalGateObservationBoundaryPreflightGateCandidateMode                                                                                                                                                                          string `json:"final_gate_observation_boundary_preflight_gate_candidate_mode"`
	FinalGateObservationBoundaryPreflightGateCandidateStage                                                                                                                                                                         string `json:"final_gate_observation_boundary_preflight_gate_candidate_stage"`
	FinalGateObservationBoundaryPreflightGateCandidateRawDreamTextObserved                                                                                                                                                          bool   `json:"final_gate_observation_boundary_preflight_gate_candidate_raw_dream_text_observed"`
	FinalGateObservationBoundaryPreflightGateCandidateRawDreamTextForwarded                                                                                                                                                         bool   `json:"final_gate_observation_boundary_preflight_gate_candidate_raw_dream_text_forwarded"`
	FinalGateObservationBoundaryPreflightGateCandidateRawDreamTextAllowed                                                                                                                                                           bool   `json:"final_gate_observation_boundary_preflight_gate_candidate_raw_dream_text_allowed"`
	FinalGateObservationBoundaryPreflightGateCandidateJanusSurfaceAllowed                                                                                                                                                           bool   `json:"final_gate_observation_boundary_preflight_gate_candidate_janus_surface_allowed"`
	FinalGateObservationBoundaryPreflightGateCandidateCoocLearningAllowed                                                                                                                                                           bool   `json:"final_gate_observation_boundary_preflight_gate_candidate_cooc_learning_allowed"`
	FinalGateObservationBoundaryPreflightGateCandidateDeltaHarvestAllowed                                                                                                                                                           bool   `json:"final_gate_observation_boundary_preflight_gate_candidate_delta_harvest_allowed"`
	FinalGateObservationBoundaryPreflightGateCandidateBodyMutationAllowed                                                                                                                                                           bool   `json:"final_gate_observation_boundary_preflight_gate_candidate_body_mutation_allowed"`
	FinalGateObservationBoundaryPreflightGateCandidatePreStateHashRequired                                                                                                                                                          bool   `json:"final_gate_observation_boundary_preflight_gate_candidate_pre_state_hash_required"`
	FinalGateObservationBoundaryPreflightGateCandidatePostStateHashRequired                                                                                                                                                         bool   `json:"final_gate_observation_boundary_preflight_gate_candidate_post_state_hash_required"`
	WeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReceiverObservationBoundaryPreflightGateCandidateReady bool   `json:"weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_final_gate_receiver_observation_boundary_preflight_gate_candidate_ready"`
	WeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReceiverObservationBoundaryPreflightGateConsumed       bool   `json:"weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_final_gate_receiver_observation_boundary_preflight_gate_consumed"`
	WeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReceiverObservationBoundaryPreflightGateRequired       bool   `json:"weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_final_gate_receiver_observation_boundary_preflight_gate_required"`
	NextStepBlockedWithoutResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidate                                                                                                                                 bool   `json:"next_step_blocked_without_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate"`
	WeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReceiverObservationBoundaryPreflightGateCandidateID    string `json:"weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_final_gate_receiver_observation_boundary_preflight_gate_candidate_id"`
	AdmissionFinalGateObservationBoundaryPreflightGateCandidateHash                                                                                                                                                                 string `json:"admission_final_gate_observation_boundary_preflight_gate_candidate_hash"`
	AdmissionFinalGateObservationBoundaryPreflightGateCandidateReadBackHash                                                                                                                                                         string `json:"admission_final_gate_observation_boundary_preflight_gate_candidate_read_back_hash"`
	SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReceiverObservationBoundaryPreflightGateID       string `json:"source_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_final_gate_receiver_observation_boundary_preflight_gate_id"`
	SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReceiverObservationBoundaryPreflightGateReady    bool   `json:"source_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_final_gate_receiver_observation_boundary_preflight_gate_ready"`
	SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReceiverObservationBoundaryPreflightGateCausal   string `json:"source_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_final_gate_receiver_observation_boundary_preflight_gate_causal_id"`
	SourceAdmissionFinalGateObservationBoundaryPreflightGateHash                                                                                                                                                                    string `json:"source_admission_final_gate_observation_boundary_preflight_gate_hash"`
	SourceAdmissionFinalGateObservationBoundaryPreflightGateReadBackHash                                                                                                                                                            string `json:"source_admission_final_gate_observation_boundary_preflight_gate_read_back_hash"`
	SourceAdmissionFinalGateObservationBoundaryPreflightGateReceiptShape                                                                                                                                                            string `json:"source_admission_final_gate_observation_boundary_preflight_gate_receipt_shape"`
	SourceAdmissionFinalGateObservationBoundaryPreflightGateState                                                                                                                                                                   string `json:"source_admission_final_gate_observation_boundary_preflight_gate_state"`
	SourceAdmissionFinalGateObservationBoundaryPreflightGateAction                                                                                                                                                                  string `json:"source_admission_final_gate_observation_boundary_preflight_gate_action"`
	SourceAdmissionFinalGateObservationBoundaryPreflightGateTarget                                                                                                                                                                  string `json:"source_admission_final_gate_observation_boundary_preflight_gate_target"`
	SourceAdmissionFinalGateObservationBoundaryPreflightGateTargetKind                                                                                                                                                              string `json:"source_admission_final_gate_observation_boundary_preflight_gate_target_kind"`
	SourceAdmissionFinalGateObservationBoundaryPreflightGateTargetMode                                                                                                                                                              string `json:"source_admission_final_gate_observation_boundary_preflight_gate_target_mode"`
	SourceAdmissionFinalGateObservationBoundaryPreflightGateDryRunOnly                                                                                                                                                              bool   `json:"source_admission_final_gate_observation_boundary_preflight_gate_dry_run_only"`
	SourceAdmissionFinalGateObservationBoundaryPreflightGatePreflightVerified                                                                                                                                                       bool   `json:"source_admission_final_gate_observation_boundary_preflight_gate_preflight_verified"`
	SourceAdmissionFinalGateObservationBoundaryPreflightGateBoundaryVerified                                                                                                                                                        bool   `json:"source_admission_final_gate_observation_boundary_preflight_gate_boundary_verified"`
	SourceAdmissionFinalGateObservationBoundaryPreflightGateObservationVerified                                                                                                                                                     bool   `json:"source_admission_final_gate_observation_boundary_preflight_gate_observation_verified"`
	SourceAdmissionFinalGateObservationBoundaryPreflightGateReadBackVerified                                                                                                                                                        bool   `json:"source_admission_final_gate_observation_boundary_preflight_gate_read_back_verified"`
	SourceAdmissionFinalGateObservationBoundaryPreflightGateReady                                                                                                                                                                   bool   `json:"source_admission_final_gate_observation_boundary_preflight_gate_ready"`
	SourceFinalGateObservationBoundaryPreflightGateKind                                                                                                                                                                             string `json:"source_final_gate_observation_boundary_preflight_gate_kind"`
	SourceFinalGateObservationBoundaryPreflightGateMode                                                                                                                                                                             string `json:"source_final_gate_observation_boundary_preflight_gate_mode"`
	SourceFinalGateObservationBoundaryPreflightGateStage                                                                                                                                                                            string `json:"source_final_gate_observation_boundary_preflight_gate_stage"`
	SourceFinalGateObservationBoundaryPreflightGateRawDreamTextObserved                                                                                                                                                             bool   `json:"source_final_gate_observation_boundary_preflight_gate_raw_dream_text_observed"`
	SourceFinalGateObservationBoundaryPreflightGateRawDreamTextForwarded                                                                                                                                                            bool   `json:"source_final_gate_observation_boundary_preflight_gate_raw_dream_text_forwarded"`
	SourceFinalGateObservationBoundaryPreflightGateRawDreamTextAllowed                                                                                                                                                              bool   `json:"source_final_gate_observation_boundary_preflight_gate_raw_dream_text_allowed"`
	SourceFinalGateObservationBoundaryPreflightGateJanusSurfaceAllowed                                                                                                                                                              bool   `json:"source_final_gate_observation_boundary_preflight_gate_janus_surface_allowed"`
	SourceFinalGateObservationBoundaryPreflightGateCoocLearningAllowed                                                                                                                                                              bool   `json:"source_final_gate_observation_boundary_preflight_gate_cooc_learning_allowed"`
	SourceFinalGateObservationBoundaryPreflightGateDeltaHarvestAllowed                                                                                                                                                              bool   `json:"source_final_gate_observation_boundary_preflight_gate_delta_harvest_allowed"`
	SourceFinalGateObservationBoundaryPreflightGateBodyMutationAllowed                                                                                                                                                              bool   `json:"source_final_gate_observation_boundary_preflight_gate_body_mutation_allowed"`
	SourceFinalGateObservationBoundaryPreflightGatePreStateHashRequired                                                                                                                                                             bool   `json:"source_final_gate_observation_boundary_preflight_gate_pre_state_hash_required"`
	SourceFinalGateObservationBoundaryPreflightGatePostStateHashRequired                                                                                                                                                            bool   `json:"source_final_gate_observation_boundary_preflight_gate_post_state_hash_required"`
	SourceAdmissionFinalGateObservationBoundaryPreflightGateReason                                                                                                                                                                  string `json:"source_admission_final_gate_observation_boundary_preflight_gate_reason"`
}

func runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReceiverObservationBoundaryPreflightGateCandidate(args []string) error {
	if len(args) != 2 {
		return fmt.Errorf("usage: --admission-live-route-weighted-admission-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-candidate-store-reader-proof-precondition-decision-promotion-switch-enable-gate-live-stage-final-gate-receiver-observation-boundary-preflight-gate-candidate RESONANCE_GRAFT_ADMISSION_FINAL_GATE_OBSERVATION_BOUNDARY_PREFLIGHT_GATE_REPORT RESONANCE_GRAFT_ADMISSION_FINAL_GATE_OBSERVATION_BOUNDARY_PREFLIGHT_GATE_CANDIDATE_REPORT")
	}
	gatePath := args[0]
	outputPath := args[1]
	if strings.TrimSpace(outputPath) == "" {
		return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage final gate receiver observation boundary preflight gate candidate output path missing")
	}
	sourceGate, root, err := readAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReceiverObservationBoundaryPreflightGateReportForAssert(gatePath)
	if err != nil {
		return err
	}
	if err := admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReceiverObservationBoundaryPreflightGateReportError(sourceGate, root); err != nil {
		return err
	}
	candidate := admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReceiverObservationBoundaryPreflightGateCandidateReport{
		admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReceiverObservationBoundaryPreflightGateReport: sourceGate,
		AdmissionFinalGateObservationBoundaryPreflightGateCandidateState:               "blocked",
		AdmissionFinalGateObservationBoundaryPreflightGateCandidateAction:              "draft_blocked_final_gate_observation_boundary_preflight_gate_candidate",
		AdmissionFinalGateObservationBoundaryPreflightGateCandidateTarget:              "resonance",
		AdmissionFinalGateObservationBoundaryPreflightGateCandidateTargetKind:          "weighted_internal_world_shadow_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_final_gate_receiver_observation_boundary_preflight_gate",
		AdmissionFinalGateObservationBoundaryPreflightGateCandidateTargetMode:          "closed_preflight_gate_candidate_dry_run",
		AdmissionFinalGateObservationBoundaryPreflightGateCandidateDryRunOnly:          true,
		AdmissionFinalGateObservationBoundaryPreflightGateCandidateGateVerified:        true,
		AdmissionFinalGateObservationBoundaryPreflightGateCandidatePreflightVerified:   sourceGate.AdmissionFinalGateObservationBoundaryPreflightGatePreflightVerified,
		AdmissionFinalGateObservationBoundaryPreflightGateCandidateBoundaryVerified:    sourceGate.AdmissionFinalGateObservationBoundaryPreflightGateBoundaryVerified,
		AdmissionFinalGateObservationBoundaryPreflightGateCandidateObservationVerified: sourceGate.AdmissionFinalGateObservationBoundaryPreflightGateObservationVerified,
		AdmissionFinalGateObservationBoundaryPreflightGateCandidateReadBackVerified:    true,
		AdmissionFinalGateObservationBoundaryPreflightGateCandidateReady:               false,
		FinalGateObservationBoundaryPreflightGateCandidateKind:                         "blocked_final_gate_observation_boundary_preflight_gate_candidate",
		FinalGateObservationBoundaryPreflightGateCandidateMode:                         "no_mutation_preflight_gate_candidate",
		FinalGateObservationBoundaryPreflightGateCandidateStage:                        "post_preflight_gate_pre_live_admission",
		FinalGateObservationBoundaryPreflightGateCandidateRawDreamTextObserved:         false,
		FinalGateObservationBoundaryPreflightGateCandidateRawDreamTextForwarded:        false,
		FinalGateObservationBoundaryPreflightGateCandidateRawDreamTextAllowed:          false,
		FinalGateObservationBoundaryPreflightGateCandidateJanusSurfaceAllowed:          false,
		FinalGateObservationBoundaryPreflightGateCandidateCoocLearningAllowed:          false,
		FinalGateObservationBoundaryPreflightGateCandidateDeltaHarvestAllowed:          false,
		FinalGateObservationBoundaryPreflightGateCandidateBodyMutationAllowed:          false,
		FinalGateObservationBoundaryPreflightGateCandidatePreStateHashRequired:         true,
		FinalGateObservationBoundaryPreflightGateCandidatePostStateHashRequired:        true,
		WeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReceiverObservationBoundaryPreflightGateCandidateReady: true,
		WeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReceiverObservationBoundaryPreflightGateConsumed:       true,
		WeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReceiverObservationBoundaryPreflightGateRequired:       true,
		NextStepBlockedWithoutResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidate:                                                                                                                                 true,
		SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReceiverObservationBoundaryPreflightGateID:       sourceGate.WeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReceiverObservationBoundaryPreflightGateID,
		SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReceiverObservationBoundaryPreflightGateReady:    sourceGate.WeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReceiverObservationBoundaryPreflightGateReady,
		SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReceiverObservationBoundaryPreflightGateCausal:   sourceGate.CausalID,
		SourceAdmissionFinalGateObservationBoundaryPreflightGateHash:                sourceGate.AdmissionFinalGateObservationBoundaryPreflightGateHash,
		SourceAdmissionFinalGateObservationBoundaryPreflightGateReadBackHash:        sourceGate.AdmissionFinalGateObservationBoundaryPreflightGateReadBackHash,
		SourceAdmissionFinalGateObservationBoundaryPreflightGateReceiptShape:        sourceGate.ReceiptShape,
		SourceAdmissionFinalGateObservationBoundaryPreflightGateState:               sourceGate.AdmissionFinalGateObservationBoundaryPreflightGateState,
		SourceAdmissionFinalGateObservationBoundaryPreflightGateAction:              sourceGate.AdmissionFinalGateObservationBoundaryPreflightGateAction,
		SourceAdmissionFinalGateObservationBoundaryPreflightGateTarget:              sourceGate.AdmissionFinalGateObservationBoundaryPreflightGateTarget,
		SourceAdmissionFinalGateObservationBoundaryPreflightGateTargetKind:          sourceGate.AdmissionFinalGateObservationBoundaryPreflightGateTargetKind,
		SourceAdmissionFinalGateObservationBoundaryPreflightGateTargetMode:          sourceGate.AdmissionFinalGateObservationBoundaryPreflightGateTargetMode,
		SourceAdmissionFinalGateObservationBoundaryPreflightGateDryRunOnly:          sourceGate.AdmissionFinalGateObservationBoundaryPreflightGateDryRunOnly,
		SourceAdmissionFinalGateObservationBoundaryPreflightGatePreflightVerified:   sourceGate.AdmissionFinalGateObservationBoundaryPreflightGatePreflightVerified,
		SourceAdmissionFinalGateObservationBoundaryPreflightGateBoundaryVerified:    sourceGate.AdmissionFinalGateObservationBoundaryPreflightGateBoundaryVerified,
		SourceAdmissionFinalGateObservationBoundaryPreflightGateObservationVerified: sourceGate.AdmissionFinalGateObservationBoundaryPreflightGateObservationVerified,
		SourceAdmissionFinalGateObservationBoundaryPreflightGateReadBackVerified:    sourceGate.AdmissionFinalGateObservationBoundaryPreflightGateReadBackVerified,
		SourceAdmissionFinalGateObservationBoundaryPreflightGateReady:               sourceGate.AdmissionFinalGateObservationBoundaryPreflightGateReady,
		SourceFinalGateObservationBoundaryPreflightGateKind:                         sourceGate.FinalGateObservationBoundaryPreflightGateKind,
		SourceFinalGateObservationBoundaryPreflightGateMode:                         sourceGate.FinalGateObservationBoundaryPreflightGateMode,
		SourceFinalGateObservationBoundaryPreflightGateStage:                        sourceGate.FinalGateObservationBoundaryPreflightGateStage,
		SourceFinalGateObservationBoundaryPreflightGateRawDreamTextObserved:         sourceGate.FinalGateObservationBoundaryPreflightGateRawDreamTextObserved,
		SourceFinalGateObservationBoundaryPreflightGateRawDreamTextForwarded:        sourceGate.FinalGateObservationBoundaryPreflightGateRawDreamTextForwarded,
		SourceFinalGateObservationBoundaryPreflightGateRawDreamTextAllowed:          sourceGate.FinalGateObservationBoundaryPreflightGateRawDreamTextAllowed,
		SourceFinalGateObservationBoundaryPreflightGateJanusSurfaceAllowed:          sourceGate.FinalGateObservationBoundaryPreflightGateJanusSurfaceAllowed,
		SourceFinalGateObservationBoundaryPreflightGateCoocLearningAllowed:          sourceGate.FinalGateObservationBoundaryPreflightGateCoocLearningAllowed,
		SourceFinalGateObservationBoundaryPreflightGateDeltaHarvestAllowed:          sourceGate.FinalGateObservationBoundaryPreflightGateDeltaHarvestAllowed,
		SourceFinalGateObservationBoundaryPreflightGateBodyMutationAllowed:          sourceGate.FinalGateObservationBoundaryPreflightGateBodyMutationAllowed,
		SourceFinalGateObservationBoundaryPreflightGatePreStateHashRequired:         sourceGate.FinalGateObservationBoundaryPreflightGatePreStateHashRequired,
		SourceFinalGateObservationBoundaryPreflightGatePostStateHashRequired:        sourceGate.FinalGateObservationBoundaryPreflightGatePostStateHashRequired,
		SourceAdmissionFinalGateObservationBoundaryPreflightGateReason:              sourceGate.Reason,
	}
	candidate.Schema = admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReceiverObservationBoundaryPreflightGateCandidateSchema
	candidate.Status = "shadow_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_final_gate_receiver_observation_boundary_preflight_gate_candidate_blocked_dry_run"
	candidate.Target = "live_route_admission_next_step"
	candidate.TargetKind = "weighted_internal_world_shadow_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_final_gate_receiver_observation_boundary_preflight_gate_candidate"
	candidate.TargetMode = "closed_preflight_gate_candidate_dry_run"
	candidate.Action = "draft_weighted_resonance_shadow_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_final_gate_receiver_observation_boundary_preflight_gate_candidate_dry_run"
	candidate.WriterAction = "reject_blocked_admission_final_gate_observation_boundary_preflight_gate_candidate"
	candidate.RollbackAction = "reject_blocked_admission_final_gate_observation_boundary_preflight_gate_candidate"
	candidate.LedgerState = "blocked"
	candidate.LedgerAction = "reject_blocked_admission_final_gate_observation_boundary_preflight_gate_candidate"
	candidate.LedgerContract = "none"
	candidate.LedgerEntrypoint = "none"
	candidate.LedgerReceiptShape = "none"
	candidate.LedgerWriteScope = "none"
	candidate.LedgerReady = false
	candidate.LedgerAppendAllowed = false
	candidate.ReceiptShape = "weighted_resonance_shadow_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_final_gate_receiver_observation_boundary_preflight_gate_candidate_receipt"
	candidate.SourceSchema = sourceGate.Schema
	candidate.SourceStatus = sourceGate.Status
	candidate.SourceTarget = sourceGate.Target
	candidate.SourceReport = gatePath
	candidate.AuthorityGranted = false
	candidate.Reason = "weighted resonance shadow graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage final gate receiver observation boundary preflight gate candidate drafted from blocked gate; live admission remains closed"
	candidate.CausalID = admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReceiverObservationBoundaryPreflightGateCandidateCausalID(candidate)
	candidate.AdmissionFinalGateObservationBoundaryPreflightGateCandidateHash = admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReceiverObservationBoundaryPreflightGateCandidateHash(candidate)
	candidate.AdmissionFinalGateObservationBoundaryPreflightGateCandidateReadBackHash = admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReceiverObservationBoundaryPreflightGateCandidateReadBackHash(candidate)
	candidate.WeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReceiverObservationBoundaryPreflightGateCandidateID = admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReceiverObservationBoundaryPreflightGateCandidateID(candidate)
	if candidate.CausalID == "" ||
		candidate.AdmissionFinalGateObservationBoundaryPreflightGateCandidateHash == "" ||
		candidate.AdmissionFinalGateObservationBoundaryPreflightGateCandidateReadBackHash == "" ||
		candidate.WeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReceiverObservationBoundaryPreflightGateCandidateID == "" ||
		candidate.AdmissionFinalGateObservationBoundaryPreflightGateCandidateHash == candidate.AdmissionFinalGateObservationBoundaryPreflightGateCandidateReadBackHash {
		return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage final gate receiver observation boundary preflight gate candidate read-back proof failed")
	}
	raw, err := json.MarshalIndent(candidate, "", "  ")
	if err != nil {
		return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage final gate receiver observation boundary preflight gate candidate marshal failed: %w", err)
	}
	raw = append(raw, '\n')
	if err := os.WriteFile(outputPath, raw, 0600); err != nil {
		return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage final gate receiver observation boundary preflight gate candidate write failed: %w", err)
	}
	fmt.Printf("[admission-live-route-weighted-admission-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-candidate-store-reader-proof-precondition-decision-promotion-switch-enable-gate-live-stage-final-gate-receiver-observation-boundary-preflight-gate-candidate] pass: resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_report=%s resonance_graft_admission_final_gate_observation_boundary_preflight_gate_report=%s\n", outputPath, gatePath)
	return nil
}

func runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReceiverObservationBoundaryPreflightGateCandidateAssert(args []string) error {
	if len(args) != 1 {
		return fmt.Errorf("usage: --admission-live-route-weighted-admission-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-candidate-store-reader-proof-precondition-decision-promotion-switch-enable-gate-live-stage-final-gate-receiver-observation-boundary-preflight-gate-candidate-assert REPORT")
	}
	report, root, err := readAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReceiverObservationBoundaryPreflightGateCandidateReportForAssert(args[0])
	if err != nil {
		return err
	}
	return admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReceiverObservationBoundaryPreflightGateCandidateReportError(report, root)
}

func admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReceiverObservationBoundaryPreflightGateCandidateReportError(report admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReceiverObservationBoundaryPreflightGateCandidateReport, root map[string]json.RawMessage) error {
	if _, ok := root["schema"]; !ok {
		return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage final gate receiver observation boundary preflight gate candidate schema missing")
	}
	if report.Schema != admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReceiverObservationBoundaryPreflightGateCandidateSchema {
		return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage final gate receiver observation boundary preflight gate candidate schema mismatch: got %q want %q", report.Schema, admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReceiverObservationBoundaryPreflightGateCandidateSchema)
	}
	if report.Status != "shadow_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_final_gate_receiver_observation_boundary_preflight_gate_candidate_blocked_dry_run" {
		return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage final gate receiver observation boundary preflight gate candidate status mismatch: got %q want %q", report.Status, "shadow_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_final_gate_receiver_observation_boundary_preflight_gate_candidate_blocked_dry_run")
	}
	if report.Target != "live_route_admission_next_step" {
		return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage final gate receiver observation boundary preflight gate candidate target mismatch: got %q want %q", report.Target, "live_route_admission_next_step")
	}
	if report.TargetKind != "weighted_internal_world_shadow_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_final_gate_receiver_observation_boundary_preflight_gate_candidate" {
		return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage final gate receiver observation boundary preflight gate candidate target_kind mismatch: got %q want %q", report.TargetKind, "weighted_internal_world_shadow_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_final_gate_receiver_observation_boundary_preflight_gate_candidate")
	}
	if report.TargetMode != "closed_preflight_gate_candidate_dry_run" {
		return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage final gate receiver observation boundary preflight gate candidate target_mode mismatch: got %q want %q", report.TargetMode, "closed_preflight_gate_candidate_dry_run")
	}
	if report.Action != "draft_weighted_resonance_shadow_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_final_gate_receiver_observation_boundary_preflight_gate_candidate_dry_run" {
		return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage final gate receiver observation boundary preflight gate candidate action mismatch: got %q want %q", report.Action, "draft_weighted_resonance_shadow_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_final_gate_receiver_observation_boundary_preflight_gate_candidate_dry_run")
	}
	if report.WriterAction != "reject_blocked_admission_final_gate_observation_boundary_preflight_gate_candidate" || report.RollbackAction != "reject_blocked_admission_final_gate_observation_boundary_preflight_gate_candidate" {
		return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage final gate receiver observation boundary preflight gate candidate writer/rollback action mismatch")
	}
	if report.LedgerState != "blocked" ||
		report.LedgerAction != "reject_blocked_admission_final_gate_observation_boundary_preflight_gate_candidate" ||
		report.LedgerContract != "none" ||
		report.LedgerEntrypoint != "none" ||
		report.LedgerReceiptShape != "none" ||
		report.LedgerWriteScope != "none" {
		return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage final gate receiver observation boundary preflight gate candidate ledger guard mismatch")
	}
	if report.AdmissionFinalGateObservationBoundaryPreflightGateCandidateState != "blocked" ||
		report.AdmissionFinalGateObservationBoundaryPreflightGateCandidateAction != "draft_blocked_final_gate_observation_boundary_preflight_gate_candidate" ||
		report.AdmissionFinalGateObservationBoundaryPreflightGateCandidateTarget != "resonance" ||
		report.AdmissionFinalGateObservationBoundaryPreflightGateCandidateTargetKind != "weighted_internal_world_shadow_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_final_gate_receiver_observation_boundary_preflight_gate" ||
		report.AdmissionFinalGateObservationBoundaryPreflightGateCandidateTargetMode != "closed_preflight_gate_candidate_dry_run" {
		return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage final gate receiver observation boundary preflight gate candidate shape mismatch")
	}
	if report.ReceiptShape != "weighted_resonance_shadow_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_final_gate_receiver_observation_boundary_preflight_gate_candidate_receipt" {
		return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage final gate receiver observation boundary preflight gate candidate receipt_shape mismatch: got %q want %q", report.ReceiptShape, "weighted_resonance_shadow_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_final_gate_receiver_observation_boundary_preflight_gate_candidate_receipt")
	}
	for _, required := range []struct {
		name  string
		value bool
	}{
		{"admission_final_gate_observation_boundary_preflight_gate_candidate_dry_run_only", report.AdmissionFinalGateObservationBoundaryPreflightGateCandidateDryRunOnly},
		{"admission_final_gate_observation_boundary_preflight_gate_candidate_gate_verified", report.AdmissionFinalGateObservationBoundaryPreflightGateCandidateGateVerified},
		{"admission_final_gate_observation_boundary_preflight_gate_candidate_preflight_verified", report.AdmissionFinalGateObservationBoundaryPreflightGateCandidatePreflightVerified},
		{"admission_final_gate_observation_boundary_preflight_gate_candidate_boundary_verified", report.AdmissionFinalGateObservationBoundaryPreflightGateCandidateBoundaryVerified},
		{"admission_final_gate_observation_boundary_preflight_gate_candidate_observation_verified", report.AdmissionFinalGateObservationBoundaryPreflightGateCandidateObservationVerified},
		{"admission_final_gate_observation_boundary_preflight_gate_candidate_read_back_verified", report.AdmissionFinalGateObservationBoundaryPreflightGateCandidateReadBackVerified},
		{"final_gate_observation_boundary_preflight_gate_candidate_pre_state_hash_required", report.FinalGateObservationBoundaryPreflightGateCandidatePreStateHashRequired},
		{"final_gate_observation_boundary_preflight_gate_candidate_post_state_hash_required", report.FinalGateObservationBoundaryPreflightGateCandidatePostStateHashRequired},
		{"weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_final_gate_receiver_observation_boundary_preflight_gate_candidate_ready", report.WeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReceiverObservationBoundaryPreflightGateCandidateReady},
		{"weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_final_gate_receiver_observation_boundary_preflight_gate_consumed", report.WeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReceiverObservationBoundaryPreflightGateConsumed},
		{"weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_final_gate_receiver_observation_boundary_preflight_gate_required", report.WeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReceiverObservationBoundaryPreflightGateRequired},
		{"next_step_blocked_without_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate", report.NextStepBlockedWithoutResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidate},
		{"source_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_final_gate_receiver_observation_boundary_preflight_gate_ready", report.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReceiverObservationBoundaryPreflightGateReady},
		{"source_admission_final_gate_observation_boundary_preflight_gate_dry_run_only", report.SourceAdmissionFinalGateObservationBoundaryPreflightGateDryRunOnly},
		{"source_admission_final_gate_observation_boundary_preflight_gate_preflight_verified", report.SourceAdmissionFinalGateObservationBoundaryPreflightGatePreflightVerified},
		{"source_admission_final_gate_observation_boundary_preflight_gate_boundary_verified", report.SourceAdmissionFinalGateObservationBoundaryPreflightGateBoundaryVerified},
		{"source_admission_final_gate_observation_boundary_preflight_gate_observation_verified", report.SourceAdmissionFinalGateObservationBoundaryPreflightGateObservationVerified},
		{"source_admission_final_gate_observation_boundary_preflight_gate_read_back_verified", report.SourceAdmissionFinalGateObservationBoundaryPreflightGateReadBackVerified},
		{"source_final_gate_observation_boundary_preflight_gate_pre_state_hash_required", report.SourceFinalGateObservationBoundaryPreflightGatePreStateHashRequired},
		{"source_final_gate_observation_boundary_preflight_gate_post_state_hash_required", report.SourceFinalGateObservationBoundaryPreflightGatePostStateHashRequired},
		{"weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_final_gate_receiver_observation_boundary_preflight_gate_candidate_ready", report.WeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReceiverObservationBoundaryPreflightGateCandidateReady},
		{"weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_final_gate_receiver_observation_boundary_preflight_ready", report.WeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReceiverObservationBoundaryPreflightReady},
		{"weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_final_gate_receiver_observation_boundary_ready", report.WeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReceiverObservationBoundaryReady},
		{"weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_final_gate_receiver_observation_ready", report.WeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReceiverObservationReady},
		{"weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_final_gate_receiver_ready", report.WeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReceiverReady},
		{"weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_final_gate_intent_ready", report.WeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateIntentReady},
		{"weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_final_gate_ready", report.WeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReady},
		{"weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_seal_ready", report.WeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageSealReady},
		{"weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_authority_ready", report.WeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageAuthorityReady},
		{"weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_permit_ready", report.WeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStagePermitReady},
		{"weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_readiness_ready", report.WeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageReadinessReady},
		{"writer_inventory_verified", report.WriterInventoryVerified},
		{"writer_preflight_verified", report.WriterPreflightVerified},
		{"live_ready", report.LiveReady},
		{"admission_required", report.AdmissionRequired},
		{"shadow_only", report.ShadowOnly},
		{"dry_run_only", report.DryRunOnly},
		{"requires_writer", report.RequiresWriter},
		{"rollback_required", report.RollbackRequired},
		{"requires_rollback", report.RequiresRollback},
		{"read_only", report.ReadOnly},
		{"replay_only", report.ReplayOnly},
		{"passed", report.Passed},
	} {
		if !required.value {
			return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage final gate receiver observation boundary preflight gate candidate %s not ready", required.name)
		}
	}
	for _, closed := range []struct {
		name  string
		value bool
	}{
		{"admission_final_gate_observation_boundary_preflight_gate_candidate_ready", report.AdmissionFinalGateObservationBoundaryPreflightGateCandidateReady},
		{"final_gate_observation_boundary_preflight_gate_candidate_raw_dream_text_observed", report.FinalGateObservationBoundaryPreflightGateCandidateRawDreamTextObserved},
		{"final_gate_observation_boundary_preflight_gate_candidate_raw_dream_text_forwarded", report.FinalGateObservationBoundaryPreflightGateCandidateRawDreamTextForwarded},
		{"final_gate_observation_boundary_preflight_gate_candidate_raw_dream_text_allowed", report.FinalGateObservationBoundaryPreflightGateCandidateRawDreamTextAllowed},
		{"final_gate_observation_boundary_preflight_gate_candidate_janus_surface_allowed", report.FinalGateObservationBoundaryPreflightGateCandidateJanusSurfaceAllowed},
		{"final_gate_observation_boundary_preflight_gate_candidate_cooc_learning_allowed", report.FinalGateObservationBoundaryPreflightGateCandidateCoocLearningAllowed},
		{"final_gate_observation_boundary_preflight_gate_candidate_delta_harvest_allowed", report.FinalGateObservationBoundaryPreflightGateCandidateDeltaHarvestAllowed},
		{"final_gate_observation_boundary_preflight_gate_candidate_body_mutation_allowed", report.FinalGateObservationBoundaryPreflightGateCandidateBodyMutationAllowed},
		{"source_admission_final_gate_observation_boundary_preflight_gate_ready", report.SourceAdmissionFinalGateObservationBoundaryPreflightGateReady},
		{"source_final_gate_observation_boundary_preflight_gate_raw_dream_text_observed", report.SourceFinalGateObservationBoundaryPreflightGateRawDreamTextObserved},
		{"source_final_gate_observation_boundary_preflight_gate_raw_dream_text_forwarded", report.SourceFinalGateObservationBoundaryPreflightGateRawDreamTextForwarded},
		{"source_final_gate_observation_boundary_preflight_gate_raw_dream_text_allowed", report.SourceFinalGateObservationBoundaryPreflightGateRawDreamTextAllowed},
		{"source_final_gate_observation_boundary_preflight_gate_janus_surface_allowed", report.SourceFinalGateObservationBoundaryPreflightGateJanusSurfaceAllowed},
		{"source_final_gate_observation_boundary_preflight_gate_cooc_learning_allowed", report.SourceFinalGateObservationBoundaryPreflightGateCoocLearningAllowed},
		{"source_final_gate_observation_boundary_preflight_gate_delta_harvest_allowed", report.SourceFinalGateObservationBoundaryPreflightGateDeltaHarvestAllowed},
		{"source_final_gate_observation_boundary_preflight_gate_body_mutation_allowed", report.SourceFinalGateObservationBoundaryPreflightGateBodyMutationAllowed},
		{"admission_final_gate_observation_boundary_preflight_gate_ready", report.AdmissionFinalGateObservationBoundaryPreflightGateReady},
		{"final_gate_observation_boundary_preflight_gate_raw_dream_text_observed", report.FinalGateObservationBoundaryPreflightGateRawDreamTextObserved},
		{"final_gate_observation_boundary_preflight_gate_raw_dream_text_forwarded", report.FinalGateObservationBoundaryPreflightGateRawDreamTextForwarded},
		{"final_gate_observation_boundary_preflight_gate_raw_dream_text_allowed", report.FinalGateObservationBoundaryPreflightGateRawDreamTextAllowed},
		{"final_gate_observation_boundary_preflight_gate_janus_surface_allowed", report.FinalGateObservationBoundaryPreflightGateJanusSurfaceAllowed},
		{"final_gate_observation_boundary_preflight_gate_cooc_learning_allowed", report.FinalGateObservationBoundaryPreflightGateCoocLearningAllowed},
		{"final_gate_observation_boundary_preflight_gate_delta_harvest_allowed", report.FinalGateObservationBoundaryPreflightGateDeltaHarvestAllowed},
		{"final_gate_observation_boundary_preflight_gate_body_mutation_allowed", report.FinalGateObservationBoundaryPreflightGateBodyMutationAllowed},
		{"admission_final_gate_observation_boundary_preflight_ready", report.AdmissionFinalGateObservationBoundaryPreflightReady},
		{"admission_final_gate_observation_boundary_ready", report.AdmissionFinalGateObservationBoundaryReady},
		{"admission_final_gate_observation_ready", report.AdmissionFinalGateObservationReady},
		{"admission_final_gate_receiver_ready", report.AdmissionFinalGateReceiverReady},
		{"admission_final_gate_intent_ready", report.AdmissionFinalGateIntentReady},
		{"admission_final_gate_ready", report.AdmissionFinalGateReady},
		{"ledger_ready", report.LedgerReady},
		{"ledger_append_allowed", report.LedgerAppendAllowed},
		{"contracts_ready", report.ContractsReady},
		{"write_allowed", report.WriteAllowed},
		{"admission_allowed", report.AdmissionAllowed},
		{"live_admission_enabled", report.LiveAdmissionEnabled},
		{"mutates_state", report.MutatesState},
		{"body_mutation_allowed", report.BodyMutationAllowed},
		{"authority_granted", report.AuthorityGranted},
	} {
		if closed.value {
			return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage final gate receiver observation boundary preflight gate candidate opened %s", closed.name)
		}
	}
	if report.FinalGateObservationBoundaryPreflightGateCandidateKind != "blocked_final_gate_observation_boundary_preflight_gate_candidate" {
		return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage final gate receiver observation boundary preflight gate candidate candidate_kind mismatch: got %q want %q", report.FinalGateObservationBoundaryPreflightGateCandidateKind, "blocked_final_gate_observation_boundary_preflight_gate_candidate")
	}
	if report.FinalGateObservationBoundaryPreflightGateCandidateMode != "no_mutation_preflight_gate_candidate" {
		return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage final gate receiver observation boundary preflight gate candidate candidate_mode mismatch: got %q want %q", report.FinalGateObservationBoundaryPreflightGateCandidateMode, "no_mutation_preflight_gate_candidate")
	}
	if report.FinalGateObservationBoundaryPreflightGateCandidateStage != "post_preflight_gate_pre_live_admission" {
		return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage final gate receiver observation boundary preflight gate candidate candidate_stage mismatch: got %q want %q", report.FinalGateObservationBoundaryPreflightGateCandidateStage, "post_preflight_gate_pre_live_admission")
	}
	for _, requiredString := range []struct {
		name  string
		value string
	}{
		{"weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_final_gate_receiver_observation_boundary_preflight_gate_candidate_id", report.WeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReceiverObservationBoundaryPreflightGateCandidateID},
		{"causal_id", report.CausalID},
		{"admission_final_gate_observation_boundary_preflight_gate_candidate_hash", report.AdmissionFinalGateObservationBoundaryPreflightGateCandidateHash},
		{"admission_final_gate_observation_boundary_preflight_gate_candidate_read_back_hash", report.AdmissionFinalGateObservationBoundaryPreflightGateCandidateReadBackHash},
		{"source_report", report.SourceReport},
		{"source_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_final_gate_receiver_observation_boundary_preflight_gate_id", report.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReceiverObservationBoundaryPreflightGateID},
		{"source_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_final_gate_receiver_observation_boundary_preflight_gate_causal_id", report.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReceiverObservationBoundaryPreflightGateCausal},
		{"source_admission_final_gate_observation_boundary_preflight_gate_hash", report.SourceAdmissionFinalGateObservationBoundaryPreflightGateHash},
		{"source_admission_final_gate_observation_boundary_preflight_gate_read_back_hash", report.SourceAdmissionFinalGateObservationBoundaryPreflightGateReadBackHash},
		{"source_admission_final_gate_observation_boundary_preflight_gate_reason", report.SourceAdmissionFinalGateObservationBoundaryPreflightGateReason},
		{"source_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_id", report.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightID},
		{"source_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_final_gate_receiver_observation_boundary_id", report.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReceiverObservationBoundaryID},
		{"source_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_final_gate_receiver_observation_id", report.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReceiverObservationID},
	} {
		if strings.TrimSpace(requiredString.value) == "" {
			return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage final gate receiver observation boundary preflight gate candidate %s missing", requiredString.name)
		}
	}
	if report.SourceSchema != admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReceiverObservationBoundaryPreflightGateSchema {
		return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage final gate receiver observation boundary preflight gate candidate source_schema mismatch: got %q want %q", report.SourceSchema, admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReceiverObservationBoundaryPreflightGateSchema)
	}
	if report.SourceStatus != "shadow_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_final_gate_receiver_observation_boundary_preflight_gate_blocked_dry_run" {
		return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage final gate receiver observation boundary preflight gate candidate source_status mismatch: got %q want %q", report.SourceStatus, "shadow_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_final_gate_receiver_observation_boundary_preflight_gate_blocked_dry_run")
	}
	if report.SourceTarget != "live_route_admission_next_step" {
		return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage final gate receiver observation boundary preflight gate candidate source_target mismatch: got %q want %q", report.SourceTarget, "live_route_admission_next_step")
	}
	if report.SourceAdmissionFinalGateObservationBoundaryPreflightGateReceiptShape != "weighted_resonance_shadow_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_final_gate_receiver_observation_boundary_preflight_gate_receipt" ||
		report.SourceAdmissionFinalGateObservationBoundaryPreflightGateState != "blocked" ||
		report.SourceAdmissionFinalGateObservationBoundaryPreflightGateAction != "gate_blocked_final_gate_observation_boundary_preflight" ||
		report.SourceAdmissionFinalGateObservationBoundaryPreflightGateTarget != "resonance" ||
		report.SourceAdmissionFinalGateObservationBoundaryPreflightGateTargetKind != "weighted_internal_world_shadow_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_final_gate_receiver_observation_boundary_preflight" ||
		report.SourceAdmissionFinalGateObservationBoundaryPreflightGateTargetMode != "closed_preflight_gate_guard_dry_run" {
		return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage final gate receiver observation boundary preflight gate candidate source admission final gate observation boundary preflight gate shape mismatch")
	}
	if report.SourceFinalGateObservationBoundaryPreflightGateKind != "blocked_final_gate_observation_boundary_preflight_gate" ||
		report.SourceFinalGateObservationBoundaryPreflightGateMode != "no_mutation_preflight_gate" ||
		report.SourceFinalGateObservationBoundaryPreflightGateStage != "post_boundary_preflight_pre_live_admission" {
		return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage final gate receiver observation boundary preflight gate candidate source final gate observation boundary preflight gate mismatch")
	}
	if report.SourceAdmissionFinalGateObservationBoundaryPreflightGateReason != "weighted resonance shadow graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage final gate receiver observation boundary preflight gate checked from blocked preflight; live admission remains closed" {
		return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage final gate receiver observation boundary preflight gate candidate source_admission_final_gate_observation_boundary_preflight_gate_reason mismatch: got %q", report.SourceAdmissionFinalGateObservationBoundaryPreflightGateReason)
	}
	if !strings.HasPrefix(report.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReceiverObservationBoundaryPreflightGateID, "weighted-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-candidate-store-reader-proof-precondition-decision-promotion-switch-enable-gate-live-stage-final-gate-receiver-observation-boundary-preflight-gate-id-") ||
		!strings.HasPrefix(report.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReceiverObservationBoundaryPreflightGateCausal, "weighted-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-candidate-store-reader-proof-precondition-decision-promotion-switch-enable-gate-live-stage-final-gate-receiver-observation-boundary-preflight-gate-causal-") ||
		!strings.HasPrefix(report.SourceAdmissionFinalGateObservationBoundaryPreflightGateHash, "weighted-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-candidate-store-reader-proof-precondition-decision-promotion-switch-enable-gate-live-stage-final-gate-receiver-observation-boundary-preflight-gate-") ||
		!strings.HasPrefix(report.SourceAdmissionFinalGateObservationBoundaryPreflightGateReadBackHash, "weighted-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-candidate-store-reader-proof-precondition-decision-promotion-switch-enable-gate-live-stage-final-gate-receiver-observation-boundary-preflight-gate-read-") ||
		report.SourceAdmissionFinalGateObservationBoundaryPreflightGateHash == report.SourceAdmissionFinalGateObservationBoundaryPreflightGateReadBackHash {
		return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage final gate receiver observation boundary preflight gate candidate source gate proof mismatch")
	}
	if report.BodyTarget != "none" {
		return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage final gate receiver observation boundary preflight gate candidate body_target mismatch: got %q want %q", report.BodyTarget, "none")
	}
	if report.CausalID == "" || report.CausalID != admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReceiverObservationBoundaryPreflightGateCandidateCausalID(report) {
		return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage final gate receiver observation boundary preflight gate candidate causal_id mismatch")
	}
	if report.AdmissionFinalGateObservationBoundaryPreflightGateCandidateHash == "" || report.AdmissionFinalGateObservationBoundaryPreflightGateCandidateHash != admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReceiverObservationBoundaryPreflightGateCandidateHash(report) {
		return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage final gate receiver observation boundary preflight gate candidate candidate_hash mismatch")
	}
	if report.AdmissionFinalGateObservationBoundaryPreflightGateCandidateReadBackHash == "" || report.AdmissionFinalGateObservationBoundaryPreflightGateCandidateReadBackHash != admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReceiverObservationBoundaryPreflightGateCandidateReadBackHash(report) {
		return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage final gate receiver observation boundary preflight gate candidate read_back_hash mismatch")
	}
	if report.AdmissionFinalGateObservationBoundaryPreflightGateCandidateHash == report.AdmissionFinalGateObservationBoundaryPreflightGateCandidateReadBackHash {
		return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage final gate receiver observation boundary preflight gate candidate read-back proof collapsed")
	}
	if report.WeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReceiverObservationBoundaryPreflightGateCandidateID == "" || report.WeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReceiverObservationBoundaryPreflightGateCandidateID != admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReceiverObservationBoundaryPreflightGateCandidateID(report) {
		return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage final gate receiver observation boundary preflight gate candidate id mismatch")
	}
	if report.Reason != "weighted resonance shadow graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage final gate receiver observation boundary preflight gate candidate drafted from blocked gate; live admission remains closed" {
		return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage final gate receiver observation boundary preflight gate candidate reason mismatch: got %q", report.Reason)
	}
	return nil
}

func admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReceiverObservationBoundaryPreflightGateCandidateCausalID(report admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReceiverObservationBoundaryPreflightGateCandidateReport) string {
	h := hashJSON(struct {
		SourceGateID   string `json:"source_admission_final_gate_observation_boundary_preflight_gate_id"`
		SourceGateRead string `json:"source_admission_final_gate_observation_boundary_preflight_gate_read_back_hash"`
		SourceReport   string `json:"source_report"`
		CandidateKind  string `json:"candidate_kind"`
		CandidateStage string `json:"candidate_stage"`
	}{
		SourceGateID:   report.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReceiverObservationBoundaryPreflightGateID,
		SourceGateRead: report.SourceAdmissionFinalGateObservationBoundaryPreflightGateReadBackHash,
		SourceReport:   report.SourceReport,
		CandidateKind:  report.FinalGateObservationBoundaryPreflightGateCandidateKind,
		CandidateStage: report.FinalGateObservationBoundaryPreflightGateCandidateStage,
	})
	if h == "" {
		return ""
	}
	return "weighted-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-candidate-store-reader-proof-precondition-decision-promotion-switch-enable-gate-live-stage-final-gate-receiver-observation-boundary-preflight-gate-candidate-causal-" + h
}

func admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReceiverObservationBoundaryPreflightGateCandidateHash(report admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReceiverObservationBoundaryPreflightGateCandidateReport) string {
	h := hashJSON(struct {
		CausalID            string `json:"causal_id"`
		SourceGateID        string `json:"source_admission_final_gate_observation_boundary_preflight_gate_id"`
		SourceGateHash      string `json:"source_admission_final_gate_observation_boundary_preflight_gate_hash"`
		SourceGateRead      string `json:"source_admission_final_gate_observation_boundary_preflight_gate_read_back_hash"`
		CandidateMode       string `json:"candidate_mode"`
		GateVerified        bool   `json:"gate_verified"`
		PreflightVerified   bool   `json:"preflight_verified"`
		BoundaryVerified    bool   `json:"boundary_verified"`
		ObservationVerified bool   `json:"observation_verified"`
		DryRunOnly          bool   `json:"dry_run_only"`
		RawDreamTextVisible bool   `json:"raw_dream_text_visible"`
		BodyMutationAllowed bool   `json:"body_mutation_allowed"`
	}{
		CausalID:            report.CausalID,
		SourceGateID:        report.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReceiverObservationBoundaryPreflightGateID,
		SourceGateHash:      report.SourceAdmissionFinalGateObservationBoundaryPreflightGateHash,
		SourceGateRead:      report.SourceAdmissionFinalGateObservationBoundaryPreflightGateReadBackHash,
		CandidateMode:       report.FinalGateObservationBoundaryPreflightGateCandidateMode,
		GateVerified:        report.AdmissionFinalGateObservationBoundaryPreflightGateCandidateGateVerified,
		PreflightVerified:   report.AdmissionFinalGateObservationBoundaryPreflightGateCandidatePreflightVerified,
		BoundaryVerified:    report.AdmissionFinalGateObservationBoundaryPreflightGateCandidateBoundaryVerified,
		ObservationVerified: report.AdmissionFinalGateObservationBoundaryPreflightGateCandidateObservationVerified,
		DryRunOnly:          report.AdmissionFinalGateObservationBoundaryPreflightGateCandidateDryRunOnly,
		RawDreamTextVisible: report.FinalGateObservationBoundaryPreflightGateCandidateRawDreamTextObserved || report.FinalGateObservationBoundaryPreflightGateCandidateRawDreamTextForwarded,
		BodyMutationAllowed: report.FinalGateObservationBoundaryPreflightGateCandidateBodyMutationAllowed,
	})
	if h == "" {
		return ""
	}
	return "weighted-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-candidate-store-reader-proof-precondition-decision-promotion-switch-enable-gate-live-stage-final-gate-receiver-observation-boundary-preflight-gate-candidate-" + h
}

func admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReceiverObservationBoundaryPreflightGateCandidateReadBackHash(report admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReceiverObservationBoundaryPreflightGateCandidateReport) string {
	h := hashJSON(struct {
		CandidateHash  string `json:"candidate_hash"`
		SourceGate     string `json:"source_admission_final_gate_observation_boundary_preflight_gate_id"`
		CandidateKind  string `json:"candidate_kind"`
		ReadBack       bool   `json:"read_back_verified"`
		CandidateReady bool   `json:"candidate_ready"`
		AdmissionOpen  bool   `json:"admission_opened"`
	}{
		CandidateHash:  report.AdmissionFinalGateObservationBoundaryPreflightGateCandidateHash,
		SourceGate:     report.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReceiverObservationBoundaryPreflightGateID,
		CandidateKind:  report.FinalGateObservationBoundaryPreflightGateCandidateKind,
		ReadBack:       report.AdmissionFinalGateObservationBoundaryPreflightGateCandidateReadBackVerified,
		CandidateReady: report.AdmissionFinalGateObservationBoundaryPreflightGateCandidateReady,
		AdmissionOpen:  report.LiveAdmissionEnabled,
	})
	if h == "" {
		return ""
	}
	return "weighted-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-candidate-store-reader-proof-precondition-decision-promotion-switch-enable-gate-live-stage-final-gate-receiver-observation-boundary-preflight-gate-candidate-read-" + h
}

func admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReceiverObservationBoundaryPreflightGateCandidateID(report admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReceiverObservationBoundaryPreflightGateCandidateReport) string {
	h := hashJSON(struct {
		Schema                 string `json:"schema"`
		Status                 string `json:"status"`
		Action                 string `json:"action"`
		SourceGateID           string `json:"source_admission_final_gate_observation_boundary_preflight_gate_id"`
		CausalID               string `json:"causal_id"`
		CandidateHash          string `json:"candidate_hash"`
		ReadBackHash           string `json:"read_back_hash"`
		Ready                  bool   `json:"weighted_candidate_ready"`
		CandidateReady         bool   `json:"admission_final_gate_observation_boundary_preflight_gate_candidate_ready"`
		GateVerified           bool   `json:"gate_verified"`
		PreflightVerified      bool   `json:"preflight_verified"`
		BoundaryVerified       bool   `json:"boundary_verified"`
		ObservationVerified    bool   `json:"observation_verified"`
		DryRunOnly             bool   `json:"dry_run_only"`
		BodyMutationAllowed    bool   `json:"body_mutation_allowed"`
		WriteAllowed           bool   `json:"write_allowed"`
		AdmissionAllowed       bool   `json:"admission_allowed"`
		LiveAdmissionEnabled   bool   `json:"live_admission_enabled"`
		MutatesState           bool   `json:"mutates_state"`
		NextStepBlockedWithout bool   `json:"next_step_blocked_without_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate"`
		SourceGateReady        bool   `json:"source_weighted_gate_ready"`
		SourceGateClosed       bool   `json:"source_gate_closed"`
	}{
		Schema:                 report.Schema,
		Status:                 report.Status,
		Action:                 report.Action,
		SourceGateID:           report.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReceiverObservationBoundaryPreflightGateID,
		CausalID:               report.CausalID,
		CandidateHash:          report.AdmissionFinalGateObservationBoundaryPreflightGateCandidateHash,
		ReadBackHash:           report.AdmissionFinalGateObservationBoundaryPreflightGateCandidateReadBackHash,
		Ready:                  report.WeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReceiverObservationBoundaryPreflightGateCandidateReady,
		CandidateReady:         report.AdmissionFinalGateObservationBoundaryPreflightGateCandidateReady,
		GateVerified:           report.AdmissionFinalGateObservationBoundaryPreflightGateCandidateGateVerified,
		PreflightVerified:      report.AdmissionFinalGateObservationBoundaryPreflightGateCandidatePreflightVerified,
		BoundaryVerified:       report.AdmissionFinalGateObservationBoundaryPreflightGateCandidateBoundaryVerified,
		ObservationVerified:    report.AdmissionFinalGateObservationBoundaryPreflightGateCandidateObservationVerified,
		DryRunOnly:             report.AdmissionFinalGateObservationBoundaryPreflightGateCandidateDryRunOnly,
		BodyMutationAllowed:    report.FinalGateObservationBoundaryPreflightGateCandidateBodyMutationAllowed,
		WriteAllowed:           report.WriteAllowed,
		AdmissionAllowed:       report.AdmissionAllowed,
		LiveAdmissionEnabled:   report.LiveAdmissionEnabled,
		MutatesState:           report.MutatesState,
		NextStepBlockedWithout: report.NextStepBlockedWithoutResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidate,
		SourceGateReady:        report.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReceiverObservationBoundaryPreflightGateReady,
		SourceGateClosed:       !report.SourceAdmissionFinalGateObservationBoundaryPreflightGateReady && !report.SourceFinalGateObservationBoundaryPreflightGateBodyMutationAllowed,
	})
	if h == "" {
		return ""
	}
	return "weighted-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-candidate-store-reader-proof-precondition-decision-promotion-switch-enable-gate-live-stage-final-gate-receiver-observation-boundary-preflight-gate-candidate-id-" + h
}

func readAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReceiverObservationBoundaryPreflightGateCandidateReportForAssert(path string) (admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReceiverObservationBoundaryPreflightGateCandidateReport, map[string]json.RawMessage, error) {
	var report admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReceiverObservationBoundaryPreflightGateCandidateReport
	if strings.TrimSpace(path) == "" {
		return report, nil, fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage final gate receiver observation boundary preflight gate candidate path missing")
	}
	raw, err := os.ReadFile(path)
	if err != nil {
		if os.IsNotExist(err) {
			return report, nil, fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage final gate receiver observation boundary preflight gate candidate not written")
		}
		return report, nil, err
	}
	if len(raw) == 0 {
		return report, nil, fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage final gate receiver observation boundary preflight gate candidate not written")
	}
	var root map[string]json.RawMessage
	if err := json.Unmarshal(raw, &root); err != nil {
		return report, nil, fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage final gate receiver observation boundary preflight gate candidate JSON invalid: %w", err)
	}
	if err := json.Unmarshal(raw, &report); err != nil {
		return report, nil, fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage final gate receiver observation boundary preflight gate candidate decode failed: %w", err)
	}
	return report, root, nil
}
