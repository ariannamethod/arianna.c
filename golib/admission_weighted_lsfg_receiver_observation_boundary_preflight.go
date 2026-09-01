package main

import (
	"encoding/json"
	"fmt"
	"os"
	"strings"
)

const admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReceiverObservationBoundaryPreflightSchema = "arianna.live_route_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_final_gate_receiver_observation_boundary_preflight.v1"

type admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReceiverObservationBoundaryPreflightReport struct {
	admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReceiverObservationBoundaryReport

	AdmissionFinalGateObservationBoundaryPreflightState                                                                                                                                                                string `json:"admission_final_gate_observation_boundary_preflight_state"`
	AdmissionFinalGateObservationBoundaryPreflightAction                                                                                                                                                               string `json:"admission_final_gate_observation_boundary_preflight_action"`
	AdmissionFinalGateObservationBoundaryPreflightTarget                                                                                                                                                               string `json:"admission_final_gate_observation_boundary_preflight_target"`
	AdmissionFinalGateObservationBoundaryPreflightTargetKind                                                                                                                                                           string `json:"admission_final_gate_observation_boundary_preflight_target_kind"`
	AdmissionFinalGateObservationBoundaryPreflightTargetMode                                                                                                                                                           string `json:"admission_final_gate_observation_boundary_preflight_target_mode"`
	AdmissionFinalGateObservationBoundaryPreflightDryRunOnly                                                                                                                                                           bool   `json:"admission_final_gate_observation_boundary_preflight_dry_run_only"`
	AdmissionFinalGateObservationBoundaryPreflightBoundaryVerified                                                                                                                                                     bool   `json:"admission_final_gate_observation_boundary_preflight_boundary_verified"`
	AdmissionFinalGateObservationBoundaryPreflightObservationVerified                                                                                                                                                  bool   `json:"admission_final_gate_observation_boundary_preflight_observation_verified"`
	AdmissionFinalGateObservationBoundaryPreflightReadBackVerified                                                                                                                                                     bool   `json:"admission_final_gate_observation_boundary_preflight_read_back_verified"`
	AdmissionFinalGateObservationBoundaryPreflightReady                                                                                                                                                                bool   `json:"admission_final_gate_observation_boundary_preflight_ready"`
	FinalGateObservationBoundaryPreflightKind                                                                                                                                                                          string `json:"final_gate_observation_boundary_preflight_kind"`
	FinalGateObservationBoundaryPreflightMode                                                                                                                                                                          string `json:"final_gate_observation_boundary_preflight_mode"`
	FinalGateObservationBoundaryPreflightStage                                                                                                                                                                         string `json:"final_gate_observation_boundary_preflight_stage"`
	FinalGateObservationBoundaryPreflightRawDreamTextObserved                                                                                                                                                          bool   `json:"final_gate_observation_boundary_preflight_raw_dream_text_observed"`
	FinalGateObservationBoundaryPreflightRawDreamTextForwarded                                                                                                                                                         bool   `json:"final_gate_observation_boundary_preflight_raw_dream_text_forwarded"`
	FinalGateObservationBoundaryPreflightRawDreamTextAllowed                                                                                                                                                           bool   `json:"final_gate_observation_boundary_preflight_raw_dream_text_allowed"`
	FinalGateObservationBoundaryPreflightJanusSurfaceAllowed                                                                                                                                                           bool   `json:"final_gate_observation_boundary_preflight_janus_surface_allowed"`
	FinalGateObservationBoundaryPreflightCoocLearningAllowed                                                                                                                                                           bool   `json:"final_gate_observation_boundary_preflight_cooc_learning_allowed"`
	FinalGateObservationBoundaryPreflightDeltaHarvestAllowed                                                                                                                                                           bool   `json:"final_gate_observation_boundary_preflight_delta_harvest_allowed"`
	FinalGateObservationBoundaryPreflightBodyMutationAllowed                                                                                                                                                           bool   `json:"final_gate_observation_boundary_preflight_body_mutation_allowed"`
	FinalGateObservationBoundaryPreflightPreStateHashRequired                                                                                                                                                          bool   `json:"final_gate_observation_boundary_preflight_pre_state_hash_required"`
	FinalGateObservationBoundaryPreflightPostStateHashRequired                                                                                                                                                         bool   `json:"final_gate_observation_boundary_preflight_post_state_hash_required"`
	WeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReceiverObservationBoundaryPreflightReady bool   `json:"weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_final_gate_receiver_observation_boundary_preflight_ready"`
	WeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReceiverObservationBoundaryConsumed       bool   `json:"weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_final_gate_receiver_observation_boundary_consumed"`
	WeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReceiverObservationBoundaryRequired       bool   `json:"weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_final_gate_receiver_observation_boundary_required"`
	NextStepBlockedWithoutResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReceiverObservationBoundaryPreflight bool   `json:"next_step_blocked_without_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_final_gate_receiver_observation_boundary_preflight"`
	WeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReceiverObservationBoundaryPreflightID    string `json:"weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_final_gate_receiver_observation_boundary_preflight_id"`
	AdmissionFinalGateObservationBoundaryPreflightHash                                                                                                                                                                 string `json:"admission_final_gate_observation_boundary_preflight_hash"`
	AdmissionFinalGateObservationBoundaryPreflightReadBackHash                                                                                                                                                         string `json:"admission_final_gate_observation_boundary_preflight_read_back_hash"`
	SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReceiverObservationBoundaryID       string `json:"source_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_final_gate_receiver_observation_boundary_id"`
	SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReceiverObservationBoundaryReady    bool   `json:"source_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_final_gate_receiver_observation_boundary_ready"`
	SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReceiverObservationBoundaryCausal   string `json:"source_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_final_gate_receiver_observation_boundary_causal_id"`
	SourceAdmissionFinalGateObservationBoundaryHash                                                                                                                                                                    string `json:"source_admission_final_gate_observation_boundary_hash"`
	SourceAdmissionFinalGateObservationBoundaryReadBackHash                                                                                                                                                            string `json:"source_admission_final_gate_observation_boundary_read_back_hash"`
	SourceAdmissionFinalGateObservationBoundaryReceiptShape                                                                                                                                                            string `json:"source_admission_final_gate_observation_boundary_receipt_shape"`
	SourceAdmissionFinalGateObservationBoundaryState                                                                                                                                                                   string `json:"source_admission_final_gate_observation_boundary_state"`
	SourceAdmissionFinalGateObservationBoundaryAction                                                                                                                                                                  string `json:"source_admission_final_gate_observation_boundary_action"`
	SourceAdmissionFinalGateObservationBoundaryTarget                                                                                                                                                                  string `json:"source_admission_final_gate_observation_boundary_target"`
	SourceAdmissionFinalGateObservationBoundaryTargetKind                                                                                                                                                              string `json:"source_admission_final_gate_observation_boundary_target_kind"`
	SourceAdmissionFinalGateObservationBoundaryTargetMode                                                                                                                                                              string `json:"source_admission_final_gate_observation_boundary_target_mode"`
	SourceAdmissionFinalGateObservationBoundaryDryRunOnly                                                                                                                                                              bool   `json:"source_admission_final_gate_observation_boundary_dry_run_only"`
	SourceAdmissionFinalGateObservationBoundaryObservationVerified                                                                                                                                                     bool   `json:"source_admission_final_gate_observation_boundary_observation_verified"`
	SourceAdmissionFinalGateObservationBoundaryReadBackVerified                                                                                                                                                        bool   `json:"source_admission_final_gate_observation_boundary_read_back_verified"`
	SourceAdmissionFinalGateObservationBoundaryReady                                                                                                                                                                   bool   `json:"source_admission_final_gate_observation_boundary_ready"`
	SourceFinalGateObservationBoundaryKind                                                                                                                                                                             string `json:"source_final_gate_observation_boundary_kind"`
	SourceFinalGateObservationBoundaryMode                                                                                                                                                                             string `json:"source_final_gate_observation_boundary_mode"`
	SourceFinalGateObservationBoundaryStage                                                                                                                                                                            string `json:"source_final_gate_observation_boundary_stage"`
	SourceFinalGateObservationBoundaryRawDreamTextObserved                                                                                                                                                             bool   `json:"source_final_gate_observation_boundary_raw_dream_text_observed"`
	SourceFinalGateObservationBoundaryRawDreamTextForwarded                                                                                                                                                            bool   `json:"source_final_gate_observation_boundary_raw_dream_text_forwarded"`
	SourceFinalGateObservationBoundaryRawDreamTextAllowed                                                                                                                                                              bool   `json:"source_final_gate_observation_boundary_raw_dream_text_allowed"`
	SourceFinalGateObservationBoundaryJanusSurfaceAllowed                                                                                                                                                              bool   `json:"source_final_gate_observation_boundary_janus_surface_allowed"`
	SourceFinalGateObservationBoundaryCoocLearningAllowed                                                                                                                                                              bool   `json:"source_final_gate_observation_boundary_cooc_learning_allowed"`
	SourceFinalGateObservationBoundaryDeltaHarvestAllowed                                                                                                                                                              bool   `json:"source_final_gate_observation_boundary_delta_harvest_allowed"`
	SourceFinalGateObservationBoundaryBodyMutationAllowed                                                                                                                                                              bool   `json:"source_final_gate_observation_boundary_body_mutation_allowed"`
	SourceFinalGateObservationBoundaryPreStateHashRequired                                                                                                                                                             bool   `json:"source_final_gate_observation_boundary_pre_state_hash_required"`
	SourceFinalGateObservationBoundaryPostStateHashRequired                                                                                                                                                            bool   `json:"source_final_gate_observation_boundary_post_state_hash_required"`
	SourceAdmissionFinalGateObservationBoundaryReason                                                                                                                                                                  string `json:"source_admission_final_gate_observation_boundary_reason"`
}

func runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReceiverObservationBoundaryPreflight(args []string) error {
	if len(args) != 2 {
		return fmt.Errorf("usage: --admission-live-route-weighted-admission-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-candidate-store-reader-proof-precondition-decision-promotion-switch-enable-gate-live-stage-final-gate-receiver-observation-boundary-preflight RESONANCE_GRAFT_ADMISSION_FINAL_GATE_OBSERVATION_BOUNDARY_PREFLIGHT_GATE_CANDIDATE_STORE_READER_PROOF_PRECONDITION_DECISION_PROMOTION_SWITCH_ENABLE_GATE_LIVE_STAGE_FINAL_GATE_RECEIVER_OBSERVATION_BOUNDARY_REPORT RESONANCE_GRAFT_ADMISSION_FINAL_GATE_OBSERVATION_BOUNDARY_PREFLIGHT_GATE_CANDIDATE_STORE_READER_PROOF_PRECONDITION_DECISION_PROMOTION_SWITCH_ENABLE_GATE_LIVE_STAGE_FINAL_GATE_RECEIVER_OBSERVATION_BOUNDARY_PREFLIGHT_REPORT")
	}
	boundaryPath := args[0]
	outputPath := args[1]
	if strings.TrimSpace(outputPath) == "" {
		return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage final gate receiver observation boundary preflight output path missing")
	}
	sourceBoundary, root, err := readAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReceiverObservationBoundaryReportForAssert(boundaryPath)
	if err != nil {
		return err
	}
	if err := admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReceiverObservationBoundaryReportError(sourceBoundary, root); err != nil {
		return err
	}
	preflight := admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReceiverObservationBoundaryPreflightReport{
		admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReceiverObservationBoundaryReport: sourceBoundary,
		AdmissionFinalGateObservationBoundaryPreflightState:               "blocked",
		AdmissionFinalGateObservationBoundaryPreflightAction:              "check_blocked_final_gate_observation_boundary_preflight",
		AdmissionFinalGateObservationBoundaryPreflightTarget:              "resonance",
		AdmissionFinalGateObservationBoundaryPreflightTargetKind:          "weighted_internal_world_shadow_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_final_gate_receiver_observation_boundary",
		AdmissionFinalGateObservationBoundaryPreflightTargetMode:          "closed_preflight_guard_dry_run",
		AdmissionFinalGateObservationBoundaryPreflightDryRunOnly:          true,
		AdmissionFinalGateObservationBoundaryPreflightBoundaryVerified:    true,
		AdmissionFinalGateObservationBoundaryPreflightObservationVerified: sourceBoundary.WeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReceiverObservationBoundaryReady && sourceBoundary.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReceiverObservationReady,
		AdmissionFinalGateObservationBoundaryPreflightReadBackVerified:    true,
		AdmissionFinalGateObservationBoundaryPreflightReady:               false,
		FinalGateObservationBoundaryPreflightKind:                         "blocked_final_gate_observation_boundary_preflight",
		FinalGateObservationBoundaryPreflightMode:                         "no_mutation_preflight",
		FinalGateObservationBoundaryPreflightStage:                        "post_observation_boundary_pre_live_admission",
		FinalGateObservationBoundaryPreflightRawDreamTextObserved:         false,
		FinalGateObservationBoundaryPreflightRawDreamTextForwarded:        false,
		FinalGateObservationBoundaryPreflightRawDreamTextAllowed:          false,
		FinalGateObservationBoundaryPreflightJanusSurfaceAllowed:          false,
		FinalGateObservationBoundaryPreflightCoocLearningAllowed:          false,
		FinalGateObservationBoundaryPreflightDeltaHarvestAllowed:          false,
		FinalGateObservationBoundaryPreflightBodyMutationAllowed:          false,
		FinalGateObservationBoundaryPreflightPreStateHashRequired:         true,
		FinalGateObservationBoundaryPreflightPostStateHashRequired:        true,
		WeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReceiverObservationBoundaryPreflightReady: true,
		WeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReceiverObservationBoundaryConsumed:       true,
		WeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReceiverObservationBoundaryRequired:       true,
		NextStepBlockedWithoutResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReceiverObservationBoundaryPreflight: true,
		SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReceiverObservationBoundaryID:       sourceBoundary.WeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReceiverObservationBoundaryID,
		SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReceiverObservationBoundaryReady:    sourceBoundary.WeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReceiverObservationBoundaryReady,
		SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReceiverObservationBoundaryCausal:   sourceBoundary.CausalID,
		SourceAdmissionFinalGateObservationBoundaryHash:                sourceBoundary.AdmissionFinalGateObservationBoundaryHash,
		SourceAdmissionFinalGateObservationBoundaryReadBackHash:        sourceBoundary.AdmissionFinalGateObservationBoundaryReadBackHash,
		SourceAdmissionFinalGateObservationBoundaryReceiptShape:        sourceBoundary.ReceiptShape,
		SourceAdmissionFinalGateObservationBoundaryState:               sourceBoundary.AdmissionFinalGateObservationBoundaryState,
		SourceAdmissionFinalGateObservationBoundaryAction:              sourceBoundary.AdmissionFinalGateObservationBoundaryAction,
		SourceAdmissionFinalGateObservationBoundaryTarget:              sourceBoundary.AdmissionFinalGateObservationBoundaryTarget,
		SourceAdmissionFinalGateObservationBoundaryTargetKind:          sourceBoundary.AdmissionFinalGateObservationBoundaryTargetKind,
		SourceAdmissionFinalGateObservationBoundaryTargetMode:          sourceBoundary.AdmissionFinalGateObservationBoundaryTargetMode,
		SourceAdmissionFinalGateObservationBoundaryDryRunOnly:          sourceBoundary.AdmissionFinalGateObservationBoundaryDryRunOnly,
		SourceAdmissionFinalGateObservationBoundaryObservationVerified: sourceBoundary.AdmissionFinalGateObservationBoundaryObservationVerified,
		SourceAdmissionFinalGateObservationBoundaryReadBackVerified:    sourceBoundary.AdmissionFinalGateObservationBoundaryReadBackVerified,
		SourceAdmissionFinalGateObservationBoundaryReady:               sourceBoundary.AdmissionFinalGateObservationBoundaryReady,
		SourceFinalGateObservationBoundaryKind:                         sourceBoundary.FinalGateObservationBoundaryKind,
		SourceFinalGateObservationBoundaryMode:                         sourceBoundary.FinalGateObservationBoundaryMode,
		SourceFinalGateObservationBoundaryStage:                        sourceBoundary.FinalGateObservationBoundaryStage,
		SourceFinalGateObservationBoundaryRawDreamTextObserved:         sourceBoundary.FinalGateObservationBoundaryRawDreamTextObserved,
		SourceFinalGateObservationBoundaryRawDreamTextForwarded:        sourceBoundary.FinalGateObservationBoundaryRawDreamTextForwarded,
		SourceFinalGateObservationBoundaryRawDreamTextAllowed:          sourceBoundary.FinalGateObservationBoundaryRawDreamTextAllowed,
		SourceFinalGateObservationBoundaryJanusSurfaceAllowed:          sourceBoundary.FinalGateObservationBoundaryJanusSurfaceAllowed,
		SourceFinalGateObservationBoundaryCoocLearningAllowed:          sourceBoundary.FinalGateObservationBoundaryCoocLearningAllowed,
		SourceFinalGateObservationBoundaryDeltaHarvestAllowed:          sourceBoundary.FinalGateObservationBoundaryDeltaHarvestAllowed,
		SourceFinalGateObservationBoundaryBodyMutationAllowed:          sourceBoundary.FinalGateObservationBoundaryBodyMutationAllowed,
		SourceFinalGateObservationBoundaryPreStateHashRequired:         sourceBoundary.FinalGateObservationBoundaryPreStateHashRequired,
		SourceFinalGateObservationBoundaryPostStateHashRequired:        sourceBoundary.FinalGateObservationBoundaryPostStateHashRequired,
		SourceAdmissionFinalGateObservationBoundaryReason:              sourceBoundary.Reason,
	}
	preflight.Schema = admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReceiverObservationBoundaryPreflightSchema
	preflight.Status = "shadow_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_final_gate_receiver_observation_boundary_preflight_blocked_dry_run"
	preflight.Target = "live_route_admission_next_step"
	preflight.TargetKind = "weighted_internal_world_shadow_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_final_gate_receiver_observation_boundary_preflight"
	preflight.TargetMode = "closed_preflight_guard_dry_run"
	preflight.Action = "check_weighted_resonance_shadow_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_final_gate_receiver_observation_boundary_preflight_dry_run"
	preflight.WriterAction = "reject_blocked_admission_final_gate_observation_boundary_preflight"
	preflight.RollbackAction = "reject_blocked_admission_final_gate_observation_boundary_preflight"
	preflight.LedgerState = "blocked"
	preflight.LedgerAction = "reject_blocked_admission_final_gate_observation_boundary_preflight"
	preflight.LedgerContract = "none"
	preflight.LedgerEntrypoint = "none"
	preflight.LedgerReceiptShape = "none"
	preflight.LedgerWriteScope = "none"
	preflight.LedgerReady = false
	preflight.LedgerAppendAllowed = false
	preflight.ReceiptShape = "weighted_resonance_shadow_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_final_gate_receiver_observation_boundary_preflight_receipt"
	preflight.SourceSchema = sourceBoundary.Schema
	preflight.SourceStatus = sourceBoundary.Status
	preflight.SourceTarget = sourceBoundary.Target
	preflight.SourceReport = boundaryPath
	preflight.AuthorityGranted = false
	preflight.Reason = "weighted resonance shadow graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage final gate receiver observation boundary preflight checked from blocked boundary; live admission remains closed"
	preflight.CausalID = admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReceiverObservationBoundaryPreflightCausalID(preflight)
	preflight.AdmissionFinalGateObservationBoundaryPreflightHash = admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReceiverObservationBoundaryPreflightHash(preflight)
	preflight.AdmissionFinalGateObservationBoundaryPreflightReadBackHash = admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReceiverObservationBoundaryPreflightReadBackHash(preflight)
	preflight.WeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReceiverObservationBoundaryPreflightID = admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReceiverObservationBoundaryPreflightID(preflight)
	if preflight.CausalID == "" ||
		preflight.AdmissionFinalGateObservationBoundaryPreflightHash == "" ||
		preflight.AdmissionFinalGateObservationBoundaryPreflightReadBackHash == "" ||
		preflight.WeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReceiverObservationBoundaryPreflightID == "" ||
		preflight.AdmissionFinalGateObservationBoundaryPreflightHash == preflight.AdmissionFinalGateObservationBoundaryPreflightReadBackHash {
		return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage final gate receiver observation boundary preflight read-back proof failed")
	}
	raw, err := json.MarshalIndent(preflight, "", "  ")
	if err != nil {
		return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage final gate receiver observation boundary preflight marshal failed: %w", err)
	}
	raw = append(raw, '\n')
	if err := os.WriteFile(outputPath, raw, 0600); err != nil {
		return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage final gate receiver observation boundary preflight write failed: %w", err)
	}
	fmt.Printf("[admission-live-route-weighted-admission-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-candidate-store-reader-proof-precondition-decision-promotion-switch-enable-gate-live-stage-final-gate-receiver-observation-boundary-preflight] pass: resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_final_gate_receiver_observation_boundary_preflight_report=%s resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_final_gate_receiver_observation_boundary_report=%s\n", outputPath, boundaryPath)
	return nil
}

func runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReceiverObservationBoundaryPreflightAssert(args []string) error {
	if len(args) != 1 {
		return fmt.Errorf("usage: --admission-live-route-weighted-admission-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-candidate-store-reader-proof-precondition-decision-promotion-switch-enable-gate-live-stage-final-gate-receiver-observation-boundary-preflight-assert REPORT")
	}
	report, root, err := readAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReceiverObservationBoundaryPreflightReportForAssert(args[0])
	if err != nil {
		return err
	}
	return admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReceiverObservationBoundaryPreflightReportError(report, root)
}

func admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReceiverObservationBoundaryPreflightReportError(report admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReceiverObservationBoundaryPreflightReport, root map[string]json.RawMessage) error {
	if _, ok := root["schema"]; !ok {
		return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage final gate receiver observation boundary preflight schema missing")
	}
	if report.Schema != admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReceiverObservationBoundaryPreflightSchema {
		return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage final gate receiver observation boundary preflight schema mismatch: got %q want %q", report.Schema, admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReceiverObservationBoundaryPreflightSchema)
	}
	if report.Status != "shadow_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_final_gate_receiver_observation_boundary_preflight_blocked_dry_run" {
		return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage final gate receiver observation boundary preflight status mismatch: got %q want %q", report.Status, "shadow_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_final_gate_receiver_observation_boundary_preflight_blocked_dry_run")
	}
	if report.Target != "live_route_admission_next_step" {
		return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage final gate receiver observation boundary preflight target mismatch: got %q want %q", report.Target, "live_route_admission_next_step")
	}
	if report.TargetKind != "weighted_internal_world_shadow_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_final_gate_receiver_observation_boundary_preflight" {
		return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage final gate receiver observation boundary preflight target_kind mismatch: got %q want %q", report.TargetKind, "weighted_internal_world_shadow_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_final_gate_receiver_observation_boundary_preflight")
	}
	if report.TargetMode != "closed_preflight_guard_dry_run" {
		return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage final gate receiver observation boundary preflight target_mode mismatch: got %q want %q", report.TargetMode, "closed_preflight_guard_dry_run")
	}
	if report.Action != "check_weighted_resonance_shadow_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_final_gate_receiver_observation_boundary_preflight_dry_run" {
		return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage final gate receiver observation boundary preflight action mismatch: got %q want %q", report.Action, "check_weighted_resonance_shadow_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_final_gate_receiver_observation_boundary_preflight_dry_run")
	}
	if report.WriterAction != "reject_blocked_admission_final_gate_observation_boundary_preflight" || report.RollbackAction != "reject_blocked_admission_final_gate_observation_boundary_preflight" {
		return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage final gate receiver observation boundary preflight writer/rollback action mismatch")
	}
	if report.LedgerState != "blocked" ||
		report.LedgerAction != "reject_blocked_admission_final_gate_observation_boundary_preflight" ||
		report.LedgerContract != "none" ||
		report.LedgerEntrypoint != "none" ||
		report.LedgerReceiptShape != "none" ||
		report.LedgerWriteScope != "none" {
		return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage final gate receiver observation boundary preflight ledger guard mismatch")
	}
	if report.AdmissionFinalGateObservationBoundaryPreflightState != "blocked" ||
		report.AdmissionFinalGateObservationBoundaryPreflightAction != "check_blocked_final_gate_observation_boundary_preflight" ||
		report.AdmissionFinalGateObservationBoundaryPreflightTarget != "resonance" ||
		report.AdmissionFinalGateObservationBoundaryPreflightTargetKind != "weighted_internal_world_shadow_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_final_gate_receiver_observation_boundary" ||
		report.AdmissionFinalGateObservationBoundaryPreflightTargetMode != "closed_preflight_guard_dry_run" {
		return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage final gate receiver observation boundary preflight shape mismatch")
	}
	if report.ReceiptShape != "weighted_resonance_shadow_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_final_gate_receiver_observation_boundary_preflight_receipt" {
		return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage final gate receiver observation boundary preflight receipt_shape mismatch: got %q want %q", report.ReceiptShape, "weighted_resonance_shadow_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_final_gate_receiver_observation_boundary_preflight_receipt")
	}
	for _, required := range []struct {
		name  string
		value bool
	}{
		{"admission_final_gate_observation_boundary_preflight_dry_run_only", report.AdmissionFinalGateObservationBoundaryPreflightDryRunOnly},
		{"admission_final_gate_observation_boundary_preflight_boundary_verified", report.AdmissionFinalGateObservationBoundaryPreflightBoundaryVerified},
		{"admission_final_gate_observation_boundary_preflight_observation_verified", report.AdmissionFinalGateObservationBoundaryPreflightObservationVerified},
		{"admission_final_gate_observation_boundary_preflight_read_back_verified", report.AdmissionFinalGateObservationBoundaryPreflightReadBackVerified},
		{"final_gate_observation_boundary_preflight_pre_state_hash_required", report.FinalGateObservationBoundaryPreflightPreStateHashRequired},
		{"final_gate_observation_boundary_preflight_post_state_hash_required", report.FinalGateObservationBoundaryPreflightPostStateHashRequired},
		{"weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_final_gate_receiver_observation_boundary_preflight_ready", report.WeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReceiverObservationBoundaryPreflightReady},
		{"weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_final_gate_receiver_observation_boundary_consumed", report.WeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReceiverObservationBoundaryConsumed},
		{"weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_final_gate_receiver_observation_boundary_required", report.WeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReceiverObservationBoundaryRequired},
		{"next_step_blocked_without_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_final_gate_receiver_observation_boundary_preflight", report.NextStepBlockedWithoutResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReceiverObservationBoundaryPreflight},
		{"source_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_final_gate_receiver_observation_boundary_ready", report.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReceiverObservationBoundaryReady},
		{"source_admission_final_gate_observation_boundary_dry_run_only", report.SourceAdmissionFinalGateObservationBoundaryDryRunOnly},
		{"source_admission_final_gate_observation_boundary_observation_verified", report.SourceAdmissionFinalGateObservationBoundaryObservationVerified},
		{"source_admission_final_gate_observation_boundary_read_back_verified", report.SourceAdmissionFinalGateObservationBoundaryReadBackVerified},
		{"source_final_gate_observation_boundary_pre_state_hash_required", report.SourceFinalGateObservationBoundaryPreStateHashRequired},
		{"source_final_gate_observation_boundary_post_state_hash_required", report.SourceFinalGateObservationBoundaryPostStateHashRequired},
		{"weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_final_gate_receiver_observation_boundary_ready", report.WeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReceiverObservationBoundaryReady},
		{"weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_final_gate_receiver_observation_consumed", report.WeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReceiverObservationConsumed},
		{"weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_final_gate_receiver_observation_required", report.WeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReceiverObservationRequired},
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
			return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage final gate receiver observation boundary preflight %s not ready", required.name)
		}
	}
	for _, closed := range []struct {
		name  string
		value bool
	}{
		{"admission_final_gate_observation_boundary_preflight_ready", report.AdmissionFinalGateObservationBoundaryPreflightReady},
		{"final_gate_observation_boundary_preflight_raw_dream_text_observed", report.FinalGateObservationBoundaryPreflightRawDreamTextObserved},
		{"final_gate_observation_boundary_preflight_raw_dream_text_forwarded", report.FinalGateObservationBoundaryPreflightRawDreamTextForwarded},
		{"final_gate_observation_boundary_preflight_raw_dream_text_allowed", report.FinalGateObservationBoundaryPreflightRawDreamTextAllowed},
		{"final_gate_observation_boundary_preflight_janus_surface_allowed", report.FinalGateObservationBoundaryPreflightJanusSurfaceAllowed},
		{"final_gate_observation_boundary_preflight_cooc_learning_allowed", report.FinalGateObservationBoundaryPreflightCoocLearningAllowed},
		{"final_gate_observation_boundary_preflight_delta_harvest_allowed", report.FinalGateObservationBoundaryPreflightDeltaHarvestAllowed},
		{"final_gate_observation_boundary_preflight_body_mutation_allowed", report.FinalGateObservationBoundaryPreflightBodyMutationAllowed},
		{"source_admission_final_gate_observation_boundary_ready", report.SourceAdmissionFinalGateObservationBoundaryReady},
		{"source_final_gate_observation_boundary_raw_dream_text_observed", report.SourceFinalGateObservationBoundaryRawDreamTextObserved},
		{"source_final_gate_observation_boundary_raw_dream_text_forwarded", report.SourceFinalGateObservationBoundaryRawDreamTextForwarded},
		{"source_final_gate_observation_boundary_raw_dream_text_allowed", report.SourceFinalGateObservationBoundaryRawDreamTextAllowed},
		{"source_final_gate_observation_boundary_janus_surface_allowed", report.SourceFinalGateObservationBoundaryJanusSurfaceAllowed},
		{"source_final_gate_observation_boundary_cooc_learning_allowed", report.SourceFinalGateObservationBoundaryCoocLearningAllowed},
		{"source_final_gate_observation_boundary_delta_harvest_allowed", report.SourceFinalGateObservationBoundaryDeltaHarvestAllowed},
		{"source_final_gate_observation_boundary_body_mutation_allowed", report.SourceFinalGateObservationBoundaryBodyMutationAllowed},
		{"admission_final_gate_observation_boundary_ready", report.AdmissionFinalGateObservationBoundaryReady},
		{"final_gate_observation_boundary_raw_dream_text_observed", report.FinalGateObservationBoundaryRawDreamTextObserved},
		{"final_gate_observation_boundary_raw_dream_text_forwarded", report.FinalGateObservationBoundaryRawDreamTextForwarded},
		{"final_gate_observation_boundary_raw_dream_text_allowed", report.FinalGateObservationBoundaryRawDreamTextAllowed},
		{"final_gate_observation_boundary_janus_surface_allowed", report.FinalGateObservationBoundaryJanusSurfaceAllowed},
		{"final_gate_observation_boundary_cooc_learning_allowed", report.FinalGateObservationBoundaryCoocLearningAllowed},
		{"final_gate_observation_boundary_delta_harvest_allowed", report.FinalGateObservationBoundaryDeltaHarvestAllowed},
		{"final_gate_observation_boundary_body_mutation_allowed", report.FinalGateObservationBoundaryBodyMutationAllowed},
		{"admission_final_gate_observation_receiver_verified", report.AdmissionFinalGateObservationReceiverVerified},
		{"admission_final_gate_observation_ready", report.AdmissionFinalGateObservationReady},
		{"admission_final_gate_receiver_ready", report.AdmissionFinalGateReceiverReady},
		{"admission_final_gate_intent_ready", report.AdmissionFinalGateIntentReady},
		{"admission_final_gate_ready", report.AdmissionFinalGateReady},
		{"raw_dream_text_allowed", report.RawDreamTextAllowed},
		{"janus_surface_allowed", report.JanusSurfaceAllowed},
		{"cooc_learning_allowed", report.CoocLearningAllowed},
		{"delta_harvest_allowed", report.DeltaHarvestAllowed},
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
			return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage final gate receiver observation boundary preflight opened %s", closed.name)
		}
	}
	if report.FinalGateObservationBoundaryPreflightKind != "blocked_final_gate_observation_boundary_preflight" {
		return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage final gate receiver observation boundary preflight preflight_kind mismatch: got %q want %q", report.FinalGateObservationBoundaryPreflightKind, "blocked_final_gate_observation_boundary_preflight")
	}
	if report.FinalGateObservationBoundaryPreflightMode != "no_mutation_preflight" {
		return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage final gate receiver observation boundary preflight preflight_mode mismatch: got %q want %q", report.FinalGateObservationBoundaryPreflightMode, "no_mutation_preflight")
	}
	if report.FinalGateObservationBoundaryPreflightStage != "post_observation_boundary_pre_live_admission" {
		return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage final gate receiver observation boundary preflight preflight_stage mismatch: got %q want %q", report.FinalGateObservationBoundaryPreflightStage, "post_observation_boundary_pre_live_admission")
	}
	for _, requiredString := range []struct {
		name  string
		value string
	}{
		{"weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_final_gate_receiver_observation_boundary_preflight_id", report.WeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReceiverObservationBoundaryPreflightID},
		{"causal_id", report.CausalID},
		{"admission_final_gate_observation_boundary_preflight_hash", report.AdmissionFinalGateObservationBoundaryPreflightHash},
		{"admission_final_gate_observation_boundary_preflight_read_back_hash", report.AdmissionFinalGateObservationBoundaryPreflightReadBackHash},
		{"source_report", report.SourceReport},
		{"source_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_final_gate_receiver_observation_boundary_id", report.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReceiverObservationBoundaryID},
		{"source_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_final_gate_receiver_observation_boundary_causal_id", report.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReceiverObservationBoundaryCausal},
		{"source_admission_final_gate_observation_boundary_hash", report.SourceAdmissionFinalGateObservationBoundaryHash},
		{"source_admission_final_gate_observation_boundary_read_back_hash", report.SourceAdmissionFinalGateObservationBoundaryReadBackHash},
		{"source_admission_final_gate_observation_boundary_reason", report.SourceAdmissionFinalGateObservationBoundaryReason},
		{"source_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_final_gate_receiver_observation_id", report.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReceiverObservationID},
		{"source_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_final_gate_receiver_id", report.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReceiverID},
		{"source_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_final_gate_intent_id", report.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateIntentID},
	} {
		if strings.TrimSpace(requiredString.value) == "" {
			return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage final gate receiver observation boundary preflight %s missing", requiredString.name)
		}
	}
	if report.SourceSchema != admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReceiverObservationBoundarySchema {
		return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage final gate receiver observation boundary preflight source_schema mismatch: got %q want %q", report.SourceSchema, admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReceiverObservationBoundarySchema)
	}
	if report.SourceStatus != "shadow_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_final_gate_receiver_observation_boundary_declared_dry_run" {
		return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage final gate receiver observation boundary preflight source_status mismatch: got %q want %q", report.SourceStatus, "shadow_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_final_gate_receiver_observation_boundary_declared_dry_run")
	}
	if report.SourceTarget != "live_route_admission_next_step" {
		return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage final gate receiver observation boundary preflight source_target mismatch: got %q want %q", report.SourceTarget, "live_route_admission_next_step")
	}
	if report.SourceAdmissionFinalGateObservationBoundaryReceiptShape != "weighted_resonance_shadow_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_final_gate_receiver_observation_boundary_receipt" ||
		report.SourceAdmissionFinalGateObservationBoundaryState != "declared" ||
		report.SourceAdmissionFinalGateObservationBoundaryAction != "declare_blocked_final_gate_observation_boundary" ||
		report.SourceAdmissionFinalGateObservationBoundaryTarget != "resonance" ||
		report.SourceAdmissionFinalGateObservationBoundaryTargetKind != "weighted_internal_world_shadow_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_final_gate_receiver_observation" ||
		report.SourceAdmissionFinalGateObservationBoundaryTargetMode != "receipt_only_closed_dry_run" {
		return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage final gate receiver observation boundary preflight source admission final gate observation boundary shape mismatch")
	}
	if report.SourceFinalGateObservationBoundaryKind != "blocked_final_gate_observation_boundary" ||
		report.SourceFinalGateObservationBoundaryMode != "no_mutation_closed_boundary_receipt" ||
		report.SourceFinalGateObservationBoundaryStage != "post_observation_pre_live_admission" {
		return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage final gate receiver observation boundary preflight source final gate observation boundary mismatch")
	}
	if report.SourceAdmissionFinalGateObservationBoundaryReason != "weighted resonance shadow graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage final gate receiver observation boundary declared from recorded observation; live admission remains closed" {
		return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage final gate receiver observation boundary preflight source_admission_final_gate_observation_boundary_reason mismatch: got %q", report.SourceAdmissionFinalGateObservationBoundaryReason)
	}
	if !strings.HasPrefix(report.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReceiverObservationBoundaryID, "weighted-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-candidate-store-reader-proof-precondition-decision-promotion-switch-enable-gate-live-stage-final-gate-receiver-observation-boundary-id-") ||
		!strings.HasPrefix(report.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReceiverObservationBoundaryCausal, "weighted-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-candidate-store-reader-proof-precondition-decision-promotion-switch-enable-gate-live-stage-final-gate-receiver-observation-boundary-causal-") ||
		!strings.HasPrefix(report.SourceAdmissionFinalGateObservationBoundaryHash, "weighted-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-candidate-store-reader-proof-precondition-decision-promotion-switch-enable-gate-live-stage-final-gate-receiver-observation-boundary-") ||
		!strings.HasPrefix(report.SourceAdmissionFinalGateObservationBoundaryReadBackHash, "weighted-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-candidate-store-reader-proof-precondition-decision-promotion-switch-enable-gate-live-stage-final-gate-receiver-observation-boundary-read-") ||
		report.SourceAdmissionFinalGateObservationBoundaryHash == report.SourceAdmissionFinalGateObservationBoundaryReadBackHash {
		return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage final gate receiver observation boundary preflight source boundary proof mismatch")
	}
	if report.BodyTarget != "none" {
		return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage final gate receiver observation boundary preflight body_target mismatch: got %q want %q", report.BodyTarget, "none")
	}
	if report.CausalID == "" || report.CausalID != admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReceiverObservationBoundaryPreflightCausalID(report) {
		return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage final gate receiver observation boundary preflight causal_id mismatch")
	}
	if report.AdmissionFinalGateObservationBoundaryPreflightHash == "" || report.AdmissionFinalGateObservationBoundaryPreflightHash != admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReceiverObservationBoundaryPreflightHash(report) {
		return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage final gate receiver observation boundary preflight preflight_hash mismatch")
	}
	if report.AdmissionFinalGateObservationBoundaryPreflightReadBackHash == "" || report.AdmissionFinalGateObservationBoundaryPreflightReadBackHash != admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReceiverObservationBoundaryPreflightReadBackHash(report) {
		return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage final gate receiver observation boundary preflight read_back_hash mismatch")
	}
	if report.AdmissionFinalGateObservationBoundaryPreflightHash == report.AdmissionFinalGateObservationBoundaryPreflightReadBackHash {
		return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage final gate receiver observation boundary preflight read-back proof collapsed")
	}
	if report.WeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReceiverObservationBoundaryPreflightID == "" || report.WeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReceiverObservationBoundaryPreflightID != admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReceiverObservationBoundaryPreflightID(report) {
		return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage final gate receiver observation boundary preflight id mismatch")
	}
	if report.Reason != "weighted resonance shadow graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage final gate receiver observation boundary preflight checked from blocked boundary; live admission remains closed" {
		return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage final gate receiver observation boundary preflight reason mismatch: got %q", report.Reason)
	}
	return nil
}

func admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReceiverObservationBoundaryPreflightCausalID(report admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReceiverObservationBoundaryPreflightReport) string {
	h := hashJSON(struct {
		SourceBoundaryID   string `json:"source_admission_final_gate_observation_boundary_id"`
		SourceBoundaryRead string `json:"source_admission_final_gate_observation_boundary_read_back_hash"`
		SourceReport       string `json:"source_report"`
		PreflightKind      string `json:"preflight_kind"`
		PreflightStage     string `json:"preflight_stage"`
	}{
		SourceBoundaryID:   report.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReceiverObservationBoundaryID,
		SourceBoundaryRead: report.SourceAdmissionFinalGateObservationBoundaryReadBackHash,
		SourceReport:       report.SourceReport,
		PreflightKind:      report.FinalGateObservationBoundaryPreflightKind,
		PreflightStage:     report.FinalGateObservationBoundaryPreflightStage,
	})
	if h == "" {
		return ""
	}
	return "weighted-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-candidate-store-reader-proof-precondition-decision-promotion-switch-enable-gate-live-stage-final-gate-receiver-observation-boundary-preflight-causal-" + h
}

func admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReceiverObservationBoundaryPreflightHash(report admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReceiverObservationBoundaryPreflightReport) string {
	h := hashJSON(struct {
		CausalID            string `json:"causal_id"`
		SourceBoundaryID    string `json:"source_admission_final_gate_observation_boundary_id"`
		SourceBoundaryHash  string `json:"source_admission_final_gate_observation_boundary_hash"`
		SourceBoundaryRead  string `json:"source_admission_final_gate_observation_boundary_read_back_hash"`
		PreflightMode       string `json:"preflight_mode"`
		BoundaryVerified    bool   `json:"boundary_verified"`
		ObservationVerified bool   `json:"observation_verified"`
		DryRunOnly          bool   `json:"dry_run_only"`
		RawDreamTextVisible bool   `json:"raw_dream_text_visible"`
		BodyMutationAllowed bool   `json:"body_mutation_allowed"`
	}{
		CausalID:            report.CausalID,
		SourceBoundaryID:    report.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReceiverObservationBoundaryID,
		SourceBoundaryHash:  report.SourceAdmissionFinalGateObservationBoundaryHash,
		SourceBoundaryRead:  report.SourceAdmissionFinalGateObservationBoundaryReadBackHash,
		PreflightMode:       report.FinalGateObservationBoundaryPreflightMode,
		BoundaryVerified:    report.AdmissionFinalGateObservationBoundaryPreflightBoundaryVerified,
		ObservationVerified: report.AdmissionFinalGateObservationBoundaryPreflightObservationVerified,
		DryRunOnly:          report.AdmissionFinalGateObservationBoundaryPreflightDryRunOnly,
		RawDreamTextVisible: report.FinalGateObservationBoundaryPreflightRawDreamTextObserved || report.FinalGateObservationBoundaryPreflightRawDreamTextForwarded,
		BodyMutationAllowed: report.FinalGateObservationBoundaryPreflightBodyMutationAllowed,
	})
	if h == "" {
		return ""
	}
	return "weighted-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-candidate-store-reader-proof-precondition-decision-promotion-switch-enable-gate-live-stage-final-gate-receiver-observation-boundary-preflight-" + h
}

func admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReceiverObservationBoundaryPreflightReadBackHash(report admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReceiverObservationBoundaryPreflightReport) string {
	h := hashJSON(struct {
		PreflightHash   string `json:"preflight_hash"`
		SourceBoundary  string `json:"source_admission_final_gate_observation_boundary_id"`
		PreflightKind   string `json:"preflight_kind"`
		ReadBack        bool   `json:"read_back_verified"`
		PreflightReady  bool   `json:"preflight_ready"`
		AdmissionOpened bool   `json:"admission_opened"`
	}{
		PreflightHash:   report.AdmissionFinalGateObservationBoundaryPreflightHash,
		SourceBoundary:  report.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReceiverObservationBoundaryID,
		PreflightKind:   report.FinalGateObservationBoundaryPreflightKind,
		ReadBack:        report.AdmissionFinalGateObservationBoundaryPreflightReadBackVerified,
		PreflightReady:  report.AdmissionFinalGateObservationBoundaryPreflightReady,
		AdmissionOpened: report.LiveAdmissionEnabled,
	})
	if h == "" {
		return ""
	}
	return "weighted-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-candidate-store-reader-proof-precondition-decision-promotion-switch-enable-gate-live-stage-final-gate-receiver-observation-boundary-preflight-read-" + h
}

func admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReceiverObservationBoundaryPreflightID(report admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReceiverObservationBoundaryPreflightReport) string {
	h := hashJSON(struct {
		Schema                 string `json:"schema"`
		Status                 string `json:"status"`
		Action                 string `json:"action"`
		SourceBoundaryID       string `json:"source_admission_final_gate_observation_boundary_id"`
		CausalID               string `json:"causal_id"`
		PreflightHash          string `json:"preflight_hash"`
		ReadBackHash           string `json:"read_back_hash"`
		Ready                  bool   `json:"weighted_preflight_ready"`
		PreflightReady         bool   `json:"admission_final_gate_observation_boundary_preflight_ready"`
		BoundaryVerified       bool   `json:"boundary_verified"`
		ObservationVerified    bool   `json:"observation_verified"`
		DryRunOnly             bool   `json:"dry_run_only"`
		BodyMutationAllowed    bool   `json:"body_mutation_allowed"`
		WriteAllowed           bool   `json:"write_allowed"`
		AdmissionAllowed       bool   `json:"admission_allowed"`
		LiveAdmissionEnabled   bool   `json:"live_admission_enabled"`
		MutatesState           bool   `json:"mutates_state"`
		NextStepBlockedWithout bool   `json:"next_step_blocked_without_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_final_gate_receiver_observation_boundary_preflight"`
		SourceBoundaryReady    bool   `json:"source_weighted_boundary_ready"`
		SourceObservationReady bool   `json:"source_weighted_observation_ready"`
		SourceBoundaryClosed   bool   `json:"source_boundary_closed"`
	}{
		Schema:                 report.Schema,
		Status:                 report.Status,
		Action:                 report.Action,
		SourceBoundaryID:       report.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReceiverObservationBoundaryID,
		CausalID:               report.CausalID,
		PreflightHash:          report.AdmissionFinalGateObservationBoundaryPreflightHash,
		ReadBackHash:           report.AdmissionFinalGateObservationBoundaryPreflightReadBackHash,
		Ready:                  report.WeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReceiverObservationBoundaryPreflightReady,
		PreflightReady:         report.AdmissionFinalGateObservationBoundaryPreflightReady,
		BoundaryVerified:       report.AdmissionFinalGateObservationBoundaryPreflightBoundaryVerified,
		ObservationVerified:    report.AdmissionFinalGateObservationBoundaryPreflightObservationVerified,
		DryRunOnly:             report.AdmissionFinalGateObservationBoundaryPreflightDryRunOnly,
		BodyMutationAllowed:    report.FinalGateObservationBoundaryPreflightBodyMutationAllowed,
		WriteAllowed:           report.WriteAllowed,
		AdmissionAllowed:       report.AdmissionAllowed,
		LiveAdmissionEnabled:   report.LiveAdmissionEnabled,
		MutatesState:           report.MutatesState,
		NextStepBlockedWithout: report.NextStepBlockedWithoutResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReceiverObservationBoundaryPreflight,
		SourceBoundaryReady:    report.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReceiverObservationBoundaryReady,
		SourceObservationReady: report.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReceiverObservationReady,
		SourceBoundaryClosed:   !report.SourceAdmissionFinalGateObservationBoundaryReady && !report.SourceFinalGateObservationBoundaryBodyMutationAllowed,
	})
	if h == "" {
		return ""
	}
	return "weighted-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-candidate-store-reader-proof-precondition-decision-promotion-switch-enable-gate-live-stage-final-gate-receiver-observation-boundary-preflight-id-" + h
}

func readAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReceiverObservationBoundaryPreflightReportForAssert(path string) (admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReceiverObservationBoundaryPreflightReport, map[string]json.RawMessage, error) {
	var report admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReceiverObservationBoundaryPreflightReport
	if strings.TrimSpace(path) == "" {
		return report, nil, fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage final gate receiver observation boundary preflight path missing")
	}
	raw, err := os.ReadFile(path)
	if err != nil {
		if os.IsNotExist(err) {
			return report, nil, fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage final gate receiver observation boundary preflight not written")
		}
		return report, nil, err
	}
	if len(raw) == 0 {
		return report, nil, fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage final gate receiver observation boundary preflight not written")
	}
	var root map[string]json.RawMessage
	if err := json.Unmarshal(raw, &root); err != nil {
		return report, nil, fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage final gate receiver observation boundary preflight JSON invalid: %w", err)
	}
	if err := json.Unmarshal(raw, &report); err != nil {
		return report, nil, fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage final gate receiver observation boundary preflight decode failed: %w", err)
	}
	return report, root, nil
}
