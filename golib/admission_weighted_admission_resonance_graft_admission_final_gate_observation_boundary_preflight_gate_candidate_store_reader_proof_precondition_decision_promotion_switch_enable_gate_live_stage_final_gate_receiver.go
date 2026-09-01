package main

import (
	"encoding/json"
	"fmt"
	"os"
	"strings"
)

const admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReceiverSchema = "arianna.live_route_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_final_gate_receiver.v1"

type admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReceiverReport struct {
	admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateIntentReport

	AdmissionFinalGateReceiverState                                                                                                                                                             string  `json:"admission_final_gate_receiver_state"`
	AdmissionFinalGateReceiverAction                                                                                                                                                            string  `json:"admission_final_gate_receiver_action"`
	AdmissionFinalGateReceiverTarget                                                                                                                                                            string  `json:"admission_final_gate_receiver_target"`
	AdmissionFinalGateReceiverTargetKind                                                                                                                                                        string  `json:"admission_final_gate_receiver_target_kind"`
	AdmissionFinalGateReceiverTargetMode                                                                                                                                                        string  `json:"admission_final_gate_receiver_target_mode"`
	AdmissionFinalGateReceiverDryRunOnly                                                                                                                                                        bool    `json:"admission_final_gate_receiver_dry_run_only"`
	AdmissionFinalGateReceiverIntentVerified                                                                                                                                                    bool    `json:"admission_final_gate_receiver_intent_verified"`
	AdmissionFinalGateReceiverFinalGateVerified                                                                                                                                                 bool    `json:"admission_final_gate_receiver_final_gate_verified"`
	AdmissionFinalGateReceiverReady                                                                                                                                                             bool    `json:"admission_final_gate_receiver_ready"`
	FinalGateReceiver                                                                                                                                                                           string  `json:"final_gate_receiver"`
	FinalGateReceiverKind                                                                                                                                                                       string  `json:"final_gate_receiver_kind"`
	FinalGateReceiverInfluenceKind                                                                                                                                                              string  `json:"final_gate_receiver_influence_kind"`
	FinalGateReceiverMaxInfluence                                                                                                                                                               float64 `json:"final_gate_receiver_max_influence"`
	FinalGateReceiverTTLTurns                                                                                                                                                                   int     `json:"final_gate_receiver_ttl_turns"`
	FinalGateReceiverStateHashMode                                                                                                                                                              string  `json:"final_gate_receiver_state_hash_mode"`
	FinalGateReceiverRawDreamTextObserved                                                                                                                                                       bool    `json:"final_gate_receiver_raw_dream_text_observed"`
	FinalGateReceiverRawDreamTextForwarded                                                                                                                                                      bool    `json:"final_gate_receiver_raw_dream_text_forwarded"`
	FinalGateReceiverRawDreamTextAllowed                                                                                                                                                        bool    `json:"final_gate_receiver_raw_dream_text_allowed"`
	FinalGateReceiverJanusSurfaceAllowed                                                                                                                                                        bool    `json:"final_gate_receiver_janus_surface_allowed"`
	FinalGateReceiverCoocLearningAllowed                                                                                                                                                        bool    `json:"final_gate_receiver_cooc_learning_allowed"`
	FinalGateReceiverDeltaHarvestAllowed                                                                                                                                                        bool    `json:"final_gate_receiver_delta_harvest_allowed"`
	FinalGateReceiverBodyMutationAllowed                                                                                                                                                        bool    `json:"final_gate_receiver_body_mutation_allowed"`
	FinalGateReceiverPreStateHashRequired                                                                                                                                                       bool    `json:"final_gate_receiver_pre_state_hash_required"`
	FinalGateReceiverPostStateHashRequired                                                                                                                                                      bool    `json:"final_gate_receiver_post_state_hash_required"`
	WeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReceiverReady      bool    `json:"weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_final_gate_receiver_ready"`
	WeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateIntentConsumed     bool    `json:"weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_final_gate_intent_consumed"`
	WeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateIntentRequired     bool    `json:"weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_final_gate_intent_required"`
	NextStepBlockedWithoutResonanceGraftAdmissionFinalGateReceiver                                                                                                                              bool    `json:"next_step_blocked_without_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_final_gate_receiver"`
	WeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReceiverID         string  `json:"weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_final_gate_receiver_id"`
	AdmissionFinalGateReceiverPreStateHash                                                                                                                                                      string  `json:"admission_final_gate_receiver_pre_state_hash"`
	AdmissionFinalGateReceiverPostStateHash                                                                                                                                                     string  `json:"admission_final_gate_receiver_post_state_hash"`
	AdmissionFinalGateReceiverStateDeltaHash                                                                                                                                                    string  `json:"admission_final_gate_receiver_state_delta_hash"`
	SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateIntentID     string  `json:"source_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_final_gate_intent_id"`
	SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateIntentReady  bool    `json:"source_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_final_gate_intent_ready"`
	SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateIntentCausal string  `json:"source_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_final_gate_intent_causal_id"`
	SourceAdmissionFinalGateIntentHash                                                                                                                                                          string  `json:"source_admission_final_gate_intent_hash"`
	SourceAdmissionFinalGateIntentReadBack                                                                                                                                                      string  `json:"source_admission_final_gate_intent_read_back_hash"`
	SourceAdmissionFinalGateIntentReceiptShape                                                                                                                                                  string  `json:"source_admission_final_gate_intent_receipt_shape"`
	SourceAdmissionFinalGateIntentState                                                                                                                                                         string  `json:"source_admission_final_gate_intent_state"`
	SourceAdmissionFinalGateIntentAction                                                                                                                                                        string  `json:"source_admission_final_gate_intent_action"`
	SourceAdmissionFinalGateIntentTarget                                                                                                                                                        string  `json:"source_admission_final_gate_intent_target"`
	SourceAdmissionFinalGateIntentTargetKind                                                                                                                                                    string  `json:"source_admission_final_gate_intent_target_kind"`
	SourceAdmissionFinalGateIntentTargetMode                                                                                                                                                    string  `json:"source_admission_final_gate_intent_target_mode"`
	SourceAdmissionFinalGateIntentDryRunOnly                                                                                                                                                    bool    `json:"source_admission_final_gate_intent_dry_run_only"`
	SourceAdmissionFinalGateIntentFinalGateVerified                                                                                                                                             bool    `json:"source_admission_final_gate_intent_final_gate_verified"`
	SourceAdmissionFinalGateIntentSealVerified                                                                                                                                                  bool    `json:"source_admission_final_gate_intent_seal_verified"`
	SourceAdmissionFinalGateIntentReady                                                                                                                                                         bool    `json:"source_admission_final_gate_intent_ready"`
	SourceFinalGateIntentReceiver                                                                                                                                                               string  `json:"source_final_gate_intent_receiver"`
	SourceFinalGateIntentReceiverKind                                                                                                                                                           string  `json:"source_final_gate_intent_receiver_kind"`
	SourceFinalGateIntentInfluenceKind                                                                                                                                                          string  `json:"source_final_gate_intent_influence_kind"`
	SourceFinalGateIntentRawDreamTextAllowed                                                                                                                                                    bool    `json:"source_final_gate_intent_raw_dream_text_allowed"`
	SourceFinalGateIntentJanusSurfaceAllowed                                                                                                                                                    bool    `json:"source_final_gate_intent_janus_surface_allowed"`
	SourceFinalGateIntentCoocLearningAllowed                                                                                                                                                    bool    `json:"source_final_gate_intent_cooc_learning_allowed"`
	SourceFinalGateIntentDeltaHarvestAllowed                                                                                                                                                    bool    `json:"source_final_gate_intent_delta_harvest_allowed"`
	SourceFinalGateIntentPreStateHashRequired                                                                                                                                                   bool    `json:"source_final_gate_intent_pre_state_hash_required"`
	SourceFinalGateIntentPostStateHashRequired                                                                                                                                                  bool    `json:"source_final_gate_intent_post_state_hash_required"`
	SourceAdmissionFinalGateIntentReason                                                                                                                                                        string  `json:"source_admission_final_gate_intent_reason"`
}

func runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReceiver(args []string) error {
	if len(args) != 2 {
		return fmt.Errorf("usage: --admission-live-route-weighted-admission-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-candidate-store-reader-proof-precondition-decision-promotion-switch-enable-gate-live-stage-final-gate-receiver RESONANCE_GRAFT_ADMISSION_FINAL_GATE_OBSERVATION_BOUNDARY_PREFLIGHT_GATE_CANDIDATE_STORE_READER_PROOF_PRECONDITION_DECISION_PROMOTION_SWITCH_ENABLE_GATE_LIVE_STAGE_FINAL_GATE_INTENT_REPORT RESONANCE_GRAFT_ADMISSION_FINAL_GATE_OBSERVATION_BOUNDARY_PREFLIGHT_GATE_CANDIDATE_STORE_READER_PROOF_PRECONDITION_DECISION_PROMOTION_SWITCH_ENABLE_GATE_LIVE_STAGE_FINAL_GATE_RECEIVER_REPORT")
	}
	intentPath := args[0]
	outputPath := args[1]
	if strings.TrimSpace(outputPath) == "" {
		return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage final gate receiver output path missing")
	}
	sourceIntent, root, err := readAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateIntentReportForAssert(intentPath)
	if err != nil {
		return err
	}
	if err := admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateIntentReportError(sourceIntent, root); err != nil {
		return err
	}
	receiver := admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReceiverReport{
		admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateIntentReport: sourceIntent,
		AdmissionFinalGateReceiverState:             "previewed",
		AdmissionFinalGateReceiverAction:            "preview_blocked_final_gate_receiver",
		AdmissionFinalGateReceiverTarget:            "resonance",
		AdmissionFinalGateReceiverTargetKind:        "weighted_internal_world_shadow_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_final_gate_intent",
		AdmissionFinalGateReceiverTargetMode:        "bounded_receiver_preview_dry_run",
		AdmissionFinalGateReceiverDryRunOnly:        true,
		AdmissionFinalGateReceiverIntentVerified:    false,
		AdmissionFinalGateReceiverFinalGateVerified: false,
		AdmissionFinalGateReceiverReady:             false,
		FinalGateReceiver:                           sourceIntent.FinalGateIntentReceiver,
		FinalGateReceiverKind:                       sourceIntent.FinalGateIntentReceiverKind,
		FinalGateReceiverInfluenceKind:              sourceIntent.FinalGateIntentInfluenceKind,
		FinalGateReceiverMaxInfluence:               sourceIntent.FinalGateIntentMaxInfluence,
		FinalGateReceiverTTLTurns:                   sourceIntent.FinalGateIntentTTLTurns,
		FinalGateReceiverStateHashMode:              "blocked_intent_receiver_preview",
		FinalGateReceiverRawDreamTextObserved:       false,
		FinalGateReceiverRawDreamTextForwarded:      false,
		FinalGateReceiverRawDreamTextAllowed:        false,
		FinalGateReceiverJanusSurfaceAllowed:        false,
		FinalGateReceiverCoocLearningAllowed:        false,
		FinalGateReceiverDeltaHarvestAllowed:        false,
		FinalGateReceiverBodyMutationAllowed:        false,
		FinalGateReceiverPreStateHashRequired:       true,
		FinalGateReceiverPostStateHashRequired:      true,
		WeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReceiverReady:  true,
		WeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateIntentConsumed: true,
		WeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateIntentRequired: true,
		NextStepBlockedWithoutResonanceGraftAdmissionFinalGateReceiver: true,
		SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateIntentID:     sourceIntent.WeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateIntentID,
		SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateIntentReady:  sourceIntent.WeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateIntentReady,
		SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateIntentCausal: sourceIntent.CausalID,
		SourceAdmissionFinalGateIntentHash:              sourceIntent.AdmissionFinalGateIntentHash,
		SourceAdmissionFinalGateIntentReadBack:          sourceIntent.AdmissionFinalGateIntentReadBackHash,
		SourceAdmissionFinalGateIntentReceiptShape:      sourceIntent.ReceiptShape,
		SourceAdmissionFinalGateIntentState:             sourceIntent.AdmissionFinalGateIntentState,
		SourceAdmissionFinalGateIntentAction:            sourceIntent.AdmissionFinalGateIntentAction,
		SourceAdmissionFinalGateIntentTarget:            sourceIntent.AdmissionFinalGateIntentTarget,
		SourceAdmissionFinalGateIntentTargetKind:        sourceIntent.AdmissionFinalGateIntentTargetKind,
		SourceAdmissionFinalGateIntentTargetMode:        sourceIntent.AdmissionFinalGateIntentTargetMode,
		SourceAdmissionFinalGateIntentDryRunOnly:        sourceIntent.AdmissionFinalGateIntentDryRunOnly,
		SourceAdmissionFinalGateIntentFinalGateVerified: sourceIntent.AdmissionFinalGateIntentFinalGateVerified,
		SourceAdmissionFinalGateIntentSealVerified:      sourceIntent.AdmissionFinalGateIntentSealVerified,
		SourceAdmissionFinalGateIntentReady:             sourceIntent.AdmissionFinalGateIntentReady,
		SourceFinalGateIntentReceiver:                   sourceIntent.FinalGateIntentReceiver,
		SourceFinalGateIntentReceiverKind:               sourceIntent.FinalGateIntentReceiverKind,
		SourceFinalGateIntentInfluenceKind:              sourceIntent.FinalGateIntentInfluenceKind,
		SourceFinalGateIntentRawDreamTextAllowed:        sourceIntent.FinalGateIntentRawDreamTextAllowed,
		SourceFinalGateIntentJanusSurfaceAllowed:        sourceIntent.FinalGateIntentJanusSurfaceAllowed,
		SourceFinalGateIntentCoocLearningAllowed:        sourceIntent.FinalGateIntentCoocLearningAllowed,
		SourceFinalGateIntentDeltaHarvestAllowed:        sourceIntent.FinalGateIntentDeltaHarvestAllowed,
		SourceFinalGateIntentPreStateHashRequired:       sourceIntent.FinalGateIntentPreStateHashRequired,
		SourceFinalGateIntentPostStateHashRequired:      sourceIntent.FinalGateIntentPostStateHashRequired,
		SourceAdmissionFinalGateIntentReason:            sourceIntent.Reason,
	}
	receiver.Schema = admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReceiverSchema
	receiver.Status = "shadow_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_final_gate_receiver_previewed_dry_run"
	receiver.Target = "live_route_admission_next_step"
	receiver.TargetKind = "weighted_internal_world_shadow_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_final_gate_receiver"
	receiver.TargetMode = "bounded_receiver_preview_dry_run"
	receiver.Action = "preview_weighted_resonance_shadow_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_final_gate_receiver_dry_run"
	receiver.WriterAction = "reject_blocked_admission_final_gate_receiver"
	receiver.RollbackAction = "reject_blocked_admission_final_gate_receiver"
	receiver.LedgerState = "blocked"
	receiver.LedgerAction = "reject_blocked_admission_final_gate_receiver"
	receiver.LedgerContract = "none"
	receiver.LedgerEntrypoint = "none"
	receiver.LedgerReceiptShape = "none"
	receiver.LedgerWriteScope = "none"
	receiver.LedgerReady = false
	receiver.LedgerAppendAllowed = false
	receiver.ReceiptShape = "weighted_resonance_shadow_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_final_gate_receiver_receipt"
	receiver.SourceSchema = sourceIntent.Schema
	receiver.SourceStatus = sourceIntent.Status
	receiver.SourceTarget = sourceIntent.Target
	receiver.SourceReport = intentPath
	receiver.AuthorityGranted = false
	receiver.Reason = "weighted resonance shadow graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage final gate receiver previewed from blocked final gate intent; live admission remains closed"
	receiver.CausalID = admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReceiverCausalID(receiver)
	receiver.AdmissionFinalGateReceiverPreStateHash = admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReceiverPreStateHash(receiver)
	receiver.AdmissionFinalGateReceiverPostStateHash = admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReceiverPostStateHash(receiver)
	receiver.AdmissionFinalGateReceiverStateDeltaHash = admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReceiverStateDeltaHash(receiver)
	receiver.WeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReceiverID = admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReceiverID(receiver)
	if receiver.CausalID == "" ||
		receiver.AdmissionFinalGateReceiverPreStateHash == "" ||
		receiver.AdmissionFinalGateReceiverPostStateHash == "" ||
		receiver.AdmissionFinalGateReceiverStateDeltaHash == "" ||
		receiver.WeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReceiverID == "" ||
		receiver.AdmissionFinalGateReceiverPreStateHash == receiver.AdmissionFinalGateReceiverPostStateHash {
		return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage final gate receiver state proof failed")
	}
	raw, err := json.MarshalIndent(receiver, "", "  ")
	if err != nil {
		return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage final gate receiver marshal failed: %w", err)
	}
	raw = append(raw, '\n')
	if err := os.WriteFile(outputPath, raw, 0600); err != nil {
		return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage final gate receiver write failed: %w", err)
	}
	fmt.Printf("[admission-live-route-weighted-admission-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-candidate-store-reader-proof-precondition-decision-promotion-switch-enable-gate-live-stage-final-gate-receiver] pass: resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_final_gate_receiver_report=%s resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_final_gate_intent_report=%s\n", outputPath, intentPath)
	return nil
}

func runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReceiverAssert(args []string) error {
	if len(args) != 1 {
		return fmt.Errorf("usage: --admission-live-route-weighted-admission-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-candidate-store-reader-proof-precondition-decision-promotion-switch-enable-gate-live-stage-final-gate-receiver-assert REPORT")
	}
	report, root, err := readAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReceiverReportForAssert(args[0])
	if err != nil {
		return err
	}
	return admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReceiverReportError(report, root)
}

func admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReceiverReportError(report admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReceiverReport, root map[string]json.RawMessage) error {
	if _, ok := root["schema"]; !ok {
		return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage final gate receiver schema missing")
	}
	if report.Schema != admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReceiverSchema {
		return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage final gate receiver schema mismatch: got %q want %q", report.Schema, admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReceiverSchema)
	}
	if report.Status != "shadow_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_final_gate_receiver_previewed_dry_run" {
		return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage final gate receiver status mismatch: got %q want %q", report.Status, "shadow_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_final_gate_receiver_previewed_dry_run")
	}
	if report.Target != "live_route_admission_next_step" {
		return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage final gate receiver target mismatch: got %q want %q", report.Target, "live_route_admission_next_step")
	}
	if report.TargetKind != "weighted_internal_world_shadow_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_final_gate_receiver" {
		return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage final gate receiver target_kind mismatch: got %q want %q", report.TargetKind, "weighted_internal_world_shadow_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_final_gate_receiver")
	}
	if report.TargetMode != "bounded_receiver_preview_dry_run" {
		return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage final gate receiver target_mode mismatch: got %q want %q", report.TargetMode, "bounded_receiver_preview_dry_run")
	}
	if report.Action != "preview_weighted_resonance_shadow_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_final_gate_receiver_dry_run" {
		return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage final gate receiver action mismatch: got %q want %q", report.Action, "preview_weighted_resonance_shadow_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_final_gate_receiver_dry_run")
	}
	if report.WriterAction != "reject_blocked_admission_final_gate_receiver" || report.RollbackAction != "reject_blocked_admission_final_gate_receiver" {
		return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage final gate receiver writer/rollback action mismatch")
	}
	if report.LedgerState != "blocked" ||
		report.LedgerAction != "reject_blocked_admission_final_gate_receiver" ||
		report.LedgerContract != "none" ||
		report.LedgerEntrypoint != "none" ||
		report.LedgerReceiptShape != "none" ||
		report.LedgerWriteScope != "none" {
		return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage final gate receiver ledger guard mismatch")
	}
	if report.AdmissionFinalGateReceiverState != "previewed" ||
		report.AdmissionFinalGateReceiverAction != "preview_blocked_final_gate_receiver" ||
		report.AdmissionFinalGateReceiverTarget != "resonance" ||
		report.AdmissionFinalGateReceiverTargetKind != "weighted_internal_world_shadow_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_final_gate_intent" ||
		report.AdmissionFinalGateReceiverTargetMode != "bounded_receiver_preview_dry_run" {
		return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage final gate receiver shape mismatch")
	}
	if report.ReceiptShape != "weighted_resonance_shadow_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_final_gate_receiver_receipt" {
		return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage final gate receiver receipt_shape mismatch: got %q want %q", report.ReceiptShape, "weighted_resonance_shadow_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_final_gate_receiver_receipt")
	}
	for _, required := range []struct {
		name  string
		value bool
	}{
		{"admission_final_gate_receiver_dry_run_only", report.AdmissionFinalGateReceiverDryRunOnly},
		{"final_gate_receiver_pre_state_hash_required", report.FinalGateReceiverPreStateHashRequired},
		{"final_gate_receiver_post_state_hash_required", report.FinalGateReceiverPostStateHashRequired},
		{"weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_final_gate_receiver_ready", report.WeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReceiverReady},
		{"weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_final_gate_intent_consumed", report.WeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateIntentConsumed},
		{"weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_final_gate_intent_required", report.WeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateIntentRequired},
		{"next_step_blocked_without_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_final_gate_receiver", report.NextStepBlockedWithoutResonanceGraftAdmissionFinalGateReceiver},
		{"source_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_final_gate_intent_ready", report.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateIntentReady},
		{"source_admission_final_gate_intent_dry_run_only", report.SourceAdmissionFinalGateIntentDryRunOnly},
		{"source_final_gate_intent_pre_state_hash_required", report.SourceFinalGateIntentPreStateHashRequired},
		{"source_final_gate_intent_post_state_hash_required", report.SourceFinalGateIntentPostStateHashRequired},
		{"weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_final_gate_intent_ready", report.WeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateIntentReady},
		{"weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_final_gate_consumed", report.WeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateConsumed},
		{"weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_final_gate_required", report.WeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateRequired},
		{"weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_final_gate_ready", report.WeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReady},
		{"weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_seal_consumed", report.WeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageSealConsumed},
		{"weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_seal_required", report.WeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageSealRequired},
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
			return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage final gate receiver %s not ready", required.name)
		}
	}
	for _, closed := range []struct {
		name  string
		value bool
	}{
		{"admission_final_gate_receiver_intent_verified", report.AdmissionFinalGateReceiverIntentVerified},
		{"admission_final_gate_receiver_final_gate_verified", report.AdmissionFinalGateReceiverFinalGateVerified},
		{"admission_final_gate_receiver_ready", report.AdmissionFinalGateReceiverReady},
		{"final_gate_receiver_raw_dream_text_observed", report.FinalGateReceiverRawDreamTextObserved},
		{"final_gate_receiver_raw_dream_text_forwarded", report.FinalGateReceiverRawDreamTextForwarded},
		{"final_gate_receiver_raw_dream_text_allowed", report.FinalGateReceiverRawDreamTextAllowed},
		{"final_gate_receiver_janus_surface_allowed", report.FinalGateReceiverJanusSurfaceAllowed},
		{"final_gate_receiver_cooc_learning_allowed", report.FinalGateReceiverCoocLearningAllowed},
		{"final_gate_receiver_delta_harvest_allowed", report.FinalGateReceiverDeltaHarvestAllowed},
		{"final_gate_receiver_body_mutation_allowed", report.FinalGateReceiverBodyMutationAllowed},
		{"source_admission_final_gate_intent_final_gate_verified", report.SourceAdmissionFinalGateIntentFinalGateVerified},
		{"source_admission_final_gate_intent_seal_verified", report.SourceAdmissionFinalGateIntentSealVerified},
		{"source_admission_final_gate_intent_ready", report.SourceAdmissionFinalGateIntentReady},
		{"source_final_gate_intent_raw_dream_text_allowed", report.SourceFinalGateIntentRawDreamTextAllowed},
		{"source_final_gate_intent_janus_surface_allowed", report.SourceFinalGateIntentJanusSurfaceAllowed},
		{"source_final_gate_intent_cooc_learning_allowed", report.SourceFinalGateIntentCoocLearningAllowed},
		{"source_final_gate_intent_delta_harvest_allowed", report.SourceFinalGateIntentDeltaHarvestAllowed},
		{"admission_final_gate_intent_final_gate_verified", report.AdmissionFinalGateIntentFinalGateVerified},
		{"admission_final_gate_intent_seal_verified", report.AdmissionFinalGateIntentSealVerified},
		{"admission_final_gate_intent_ready", report.AdmissionFinalGateIntentReady},
		{"final_gate_intent_raw_dream_text_allowed", report.FinalGateIntentRawDreamTextAllowed},
		{"final_gate_intent_janus_surface_allowed", report.FinalGateIntentJanusSurfaceAllowed},
		{"final_gate_intent_cooc_learning_allowed", report.FinalGateIntentCoocLearningAllowed},
		{"final_gate_intent_delta_harvest_allowed", report.FinalGateIntentDeltaHarvestAllowed},
		{"source_admission_final_gate_ready", report.SourceAdmissionFinalGateReady},
		{"admission_final_gate_ready", report.AdmissionFinalGateReady},
		{"raw_dream_text_allowed", report.RawDreamTextAllowed},
		{"janus_surface_allowed", report.JanusSurfaceAllowed},
		{"cooc_learning_allowed", report.CoocLearningAllowed},
		{"delta_harvest_allowed", report.DeltaHarvestAllowed},
		{"admission_seal_ready", report.AdmissionSealReady},
		{"admission_authority_granted", report.AdmissionAuthorityGranted},
		{"manual_permit_requested", report.ManualPermitRequested},
		{"permit_key_matched", report.PermitKeyMatched},
		{"admission_permit_ready", report.AdmissionPermitReady},
		{"admission_readiness_ready", report.AdmissionReadinessReady},
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
			return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage final gate receiver opened %s", closed.name)
		}
	}
	if report.FinalGateReceiver != "resonance" {
		return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage final gate receiver receiver mismatch: got %q want %q", report.FinalGateReceiver, "resonance")
	}
	if report.FinalGateReceiverKind != "internal_world" {
		return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage final gate receiver receiver_kind mismatch: got %q want %q", report.FinalGateReceiverKind, "internal_world")
	}
	if report.FinalGateReceiverInfluenceKind != "bounded_direction" {
		return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage final gate receiver influence_kind mismatch: got %q want %q", report.FinalGateReceiverInfluenceKind, "bounded_direction")
	}
	if report.FinalGateReceiverMaxInfluence != admissionLiveRouteTurnCandidateAdmissionResonanceIntentMaxGain {
		return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage final gate receiver max_influence mismatch: got %.6f want %.6f", report.FinalGateReceiverMaxInfluence, admissionLiveRouteTurnCandidateAdmissionResonanceIntentMaxGain)
	}
	if report.FinalGateReceiverTTLTurns != admissionLiveRouteTurnCandidateAdmissionResonanceIntentTTL {
		return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage final gate receiver ttl_turns mismatch: got %d want %d", report.FinalGateReceiverTTLTurns, admissionLiveRouteTurnCandidateAdmissionResonanceIntentTTL)
	}
	if report.FinalGateReceiverStateHashMode != "blocked_intent_receiver_preview" {
		return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage final gate receiver state_hash_mode mismatch: got %q want %q", report.FinalGateReceiverStateHashMode, "blocked_intent_receiver_preview")
	}
	for _, requiredString := range []struct {
		name  string
		value string
	}{
		{"weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_final_gate_receiver_id", report.WeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReceiverID},
		{"causal_id", report.CausalID},
		{"admission_final_gate_receiver_pre_state_hash", report.AdmissionFinalGateReceiverPreStateHash},
		{"admission_final_gate_receiver_post_state_hash", report.AdmissionFinalGateReceiverPostStateHash},
		{"admission_final_gate_receiver_state_delta_hash", report.AdmissionFinalGateReceiverStateDeltaHash},
		{"source_report", report.SourceReport},
		{"source_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_final_gate_intent_id", report.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateIntentID},
		{"source_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_final_gate_intent_causal_id", report.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateIntentCausal},
		{"source_admission_final_gate_intent_hash", report.SourceAdmissionFinalGateIntentHash},
		{"source_admission_final_gate_intent_read_back_hash", report.SourceAdmissionFinalGateIntentReadBack},
		{"source_admission_final_gate_intent_reason", report.SourceAdmissionFinalGateIntentReason},
		{"source_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_final_gate_id", report.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateID},
		{"source_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_seal_id", report.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageSealID},
		{"source_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_authority_id", report.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageAuthorityID},
	} {
		if strings.TrimSpace(requiredString.value) == "" {
			return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage final gate receiver %s missing", requiredString.name)
		}
	}
	if report.SourceSchema != admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateIntentSchema {
		return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage final gate receiver source_schema mismatch: got %q want %q", report.SourceSchema, admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateIntentSchema)
	}
	if report.SourceStatus != "shadow_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_final_gate_intent_blocked_dry_run" {
		return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage final gate receiver source_status mismatch: got %q want %q", report.SourceStatus, "shadow_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_final_gate_intent_blocked_dry_run")
	}
	if report.SourceTarget != "live_route_admission_next_step" {
		return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage final gate receiver source_target mismatch: got %q want %q", report.SourceTarget, "live_route_admission_next_step")
	}
	if report.SourceAdmissionFinalGateIntentReceiptShape != "weighted_resonance_shadow_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_final_gate_intent_receipt" ||
		report.SourceAdmissionFinalGateIntentState != "blocked" ||
		report.SourceAdmissionFinalGateIntentAction != "draft_blocked_final_gate_intent" ||
		report.SourceAdmissionFinalGateIntentTarget != "resonance" ||
		report.SourceAdmissionFinalGateIntentTargetKind != "weighted_internal_world_shadow_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_final_gate" ||
		report.SourceAdmissionFinalGateIntentTargetMode != "bounded_intent_guard_dry_run" {
		return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage final gate receiver source admission final gate intent shape mismatch")
	}
	if report.SourceFinalGateIntentReceiver != "resonance" ||
		report.SourceFinalGateIntentReceiverKind != "internal_world" ||
		report.SourceFinalGateIntentInfluenceKind != "bounded_direction" {
		return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage final gate receiver source final gate intent receiver mismatch")
	}
	if report.SourceAdmissionFinalGateIntentReason != "weighted resonance shadow graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage final gate intent drafted from blocked final gate; live admission remains closed" {
		return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage final gate receiver source_admission_final_gate_intent_reason mismatch: got %q", report.SourceAdmissionFinalGateIntentReason)
	}
	if !strings.HasPrefix(report.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateIntentID, "weighted-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-candidate-store-reader-proof-precondition-decision-promotion-switch-enable-gate-live-stage-final-gate-intent-id-") ||
		!strings.HasPrefix(report.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateIntentCausal, "weighted-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-candidate-store-reader-proof-precondition-decision-promotion-switch-enable-gate-live-stage-final-gate-intent-causal-") ||
		!strings.HasPrefix(report.SourceAdmissionFinalGateIntentHash, "weighted-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-candidate-store-reader-proof-precondition-decision-promotion-switch-enable-gate-live-stage-final-gate-intent-") ||
		!strings.HasPrefix(report.SourceAdmissionFinalGateIntentReadBack, "weighted-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-candidate-store-reader-proof-precondition-decision-promotion-switch-enable-gate-live-stage-final-gate-intent-read-") {
		return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage final gate receiver source final gate intent mismatch")
	}
	if report.BodyTarget != "none" {
		return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage final gate receiver body_target mismatch: got %q want %q", report.BodyTarget, "none")
	}
	if report.CausalID == "" || report.CausalID != admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReceiverCausalID(report) {
		return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage final gate receiver causal_id mismatch")
	}
	if report.AdmissionFinalGateReceiverPreStateHash == "" || report.AdmissionFinalGateReceiverPreStateHash != admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReceiverPreStateHash(report) {
		return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage final gate receiver pre_state_hash mismatch")
	}
	if report.AdmissionFinalGateReceiverPostStateHash == "" || report.AdmissionFinalGateReceiverPostStateHash != admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReceiverPostStateHash(report) {
		return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage final gate receiver post_state_hash mismatch")
	}
	if report.AdmissionFinalGateReceiverStateDeltaHash == "" || report.AdmissionFinalGateReceiverStateDeltaHash != admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReceiverStateDeltaHash(report) {
		return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage final gate receiver state_delta_hash mismatch")
	}
	if report.AdmissionFinalGateReceiverPreStateHash == report.AdmissionFinalGateReceiverPostStateHash {
		return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage final gate receiver state proof collapsed")
	}
	if report.WeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReceiverID == "" || report.WeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReceiverID != admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReceiverID(report) {
		return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage final gate receiver id mismatch")
	}
	if report.Reason != "weighted resonance shadow graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage final gate receiver previewed from blocked final gate intent; live admission remains closed" {
		return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage final gate receiver reason mismatch: got %q", report.Reason)
	}
	return nil
}

func admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReceiverCausalID(report admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReceiverReport) string {
	h := hashJSON(struct {
		SourceIntentID   string  `json:"source_admission_final_gate_intent_id"`
		SourceIntentRead string  `json:"source_admission_final_gate_intent_read_back_hash"`
		SourceFinalGate  string  `json:"source_admission_final_gate_id"`
		Receiver         string  `json:"receiver"`
		ReceiverKind     string  `json:"receiver_kind"`
		InfluenceKind    string  `json:"influence_kind"`
		MaxInfluence     float64 `json:"max_influence"`
		TTLTurns         int     `json:"ttl_turns"`
		State            string  `json:"admission_final_gate_receiver_state"`
		Action           string  `json:"admission_final_gate_receiver_action"`
	}{
		SourceIntentID:   report.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateIntentID,
		SourceIntentRead: report.SourceAdmissionFinalGateIntentReadBack,
		SourceFinalGate:  report.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateID,
		Receiver:         report.FinalGateReceiver,
		ReceiverKind:     report.FinalGateReceiverKind,
		InfluenceKind:    report.FinalGateReceiverInfluenceKind,
		MaxInfluence:     report.FinalGateReceiverMaxInfluence,
		TTLTurns:         report.FinalGateReceiverTTLTurns,
		State:            report.AdmissionFinalGateReceiverState,
		Action:           report.AdmissionFinalGateReceiverAction,
	})
	if h == "" {
		return ""
	}
	return "weighted-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-candidate-store-reader-proof-precondition-decision-promotion-switch-enable-gate-live-stage-final-gate-receiver-causal-" + h
}

func admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReceiverPreStateHash(report admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReceiverReport) string {
	h := hashJSON(struct {
		SourceIntentID   string `json:"source_admission_final_gate_intent_id"`
		SourceIntentHash string `json:"source_admission_final_gate_intent_hash"`
		SourceIntentRead string `json:"source_admission_final_gate_intent_read_back_hash"`
		SourceFinalGate  string `json:"source_admission_final_gate_id"`
		SourceFinalRead  string `json:"source_admission_final_gate_read_back_hash"`
		StateHashMode    string `json:"state_hash_mode"`
		Receiver         string `json:"receiver"`
		ReceiverKind     string `json:"receiver_kind"`
		IntentConsumed   bool   `json:"intent_consumed"`
		IntentRequired   bool   `json:"intent_required"`
		ReceiverReady    bool   `json:"receiver_ready"`
	}{
		SourceIntentID:   report.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateIntentID,
		SourceIntentHash: report.SourceAdmissionFinalGateIntentHash,
		SourceIntentRead: report.SourceAdmissionFinalGateIntentReadBack,
		SourceFinalGate:  report.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateID,
		SourceFinalRead:  report.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReadBack,
		StateHashMode:    report.FinalGateReceiverStateHashMode,
		Receiver:         report.FinalGateReceiver,
		ReceiverKind:     report.FinalGateReceiverKind,
		IntentConsumed:   report.WeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateIntentConsumed,
		IntentRequired:   report.WeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateIntentRequired,
		ReceiverReady:    report.WeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReceiverReady,
	})
	if h == "" {
		return ""
	}
	return "weighted-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-candidate-store-reader-proof-precondition-decision-promotion-switch-enable-gate-live-stage-final-gate-receiver-pre-" + h
}

func admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReceiverPostStateHash(report admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReceiverReport) string {
	h := hashJSON(struct {
		PreStateHash          string  `json:"pre_state_hash"`
		CausalID              string  `json:"causal_id"`
		Receiver              string  `json:"receiver"`
		ReceiverKind          string  `json:"receiver_kind"`
		InfluenceKind         string  `json:"influence_kind"`
		MaxInfluence          float64 `json:"max_influence"`
		TTLTurns              int     `json:"ttl_turns"`
		StateHashMode         string  `json:"state_hash_mode"`
		RawDreamTextObserved  bool    `json:"raw_dream_text_observed"`
		RawDreamTextForwarded bool    `json:"raw_dream_text_forwarded"`
		BodyMutationAllowed   bool    `json:"body_mutation_allowed"`
	}{
		PreStateHash:          report.AdmissionFinalGateReceiverPreStateHash,
		CausalID:              report.CausalID,
		Receiver:              report.FinalGateReceiver,
		ReceiverKind:          report.FinalGateReceiverKind,
		InfluenceKind:         report.FinalGateReceiverInfluenceKind,
		MaxInfluence:          report.FinalGateReceiverMaxInfluence,
		TTLTurns:              report.FinalGateReceiverTTLTurns,
		StateHashMode:         report.FinalGateReceiverStateHashMode,
		RawDreamTextObserved:  report.FinalGateReceiverRawDreamTextObserved,
		RawDreamTextForwarded: report.FinalGateReceiverRawDreamTextForwarded,
		BodyMutationAllowed:   report.FinalGateReceiverBodyMutationAllowed,
	})
	if h == "" {
		return ""
	}
	return "weighted-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-candidate-store-reader-proof-precondition-decision-promotion-switch-enable-gate-live-stage-final-gate-receiver-post-" + h
}

func admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReceiverStateDeltaHash(report admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReceiverReport) string {
	h := hashJSON(struct {
		PreStateHash        string `json:"pre_state_hash"`
		PostStateHash       string `json:"post_state_hash"`
		CausalID            string `json:"causal_id"`
		RawTextObserved     bool   `json:"raw_text_observed"`
		RawTextForwarded    bool   `json:"raw_text_forwarded"`
		JanusSurfaceAllowed bool   `json:"janus_surface_allowed"`
		CoocLearningAllowed bool   `json:"cooc_learning_allowed"`
		DeltaHarvestAllowed bool   `json:"delta_harvest_allowed"`
		BodyMutationAllowed bool   `json:"body_mutation_allowed"`
		ReceiverReady       bool   `json:"admission_final_gate_receiver_ready"`
	}{
		PreStateHash:        report.AdmissionFinalGateReceiverPreStateHash,
		PostStateHash:       report.AdmissionFinalGateReceiverPostStateHash,
		CausalID:            report.CausalID,
		RawTextObserved:     report.FinalGateReceiverRawDreamTextObserved,
		RawTextForwarded:    report.FinalGateReceiverRawDreamTextForwarded,
		JanusSurfaceAllowed: report.FinalGateReceiverJanusSurfaceAllowed,
		CoocLearningAllowed: report.FinalGateReceiverCoocLearningAllowed,
		DeltaHarvestAllowed: report.FinalGateReceiverDeltaHarvestAllowed,
		BodyMutationAllowed: report.FinalGateReceiverBodyMutationAllowed,
		ReceiverReady:       report.AdmissionFinalGateReceiverReady,
	})
	if h == "" {
		return ""
	}
	return "weighted-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-candidate-store-reader-proof-precondition-decision-promotion-switch-enable-gate-live-stage-final-gate-receiver-delta-" + h
}

func admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReceiverID(report admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReceiverReport) string {
	h := hashJSON(struct {
		Schema                 string `json:"schema"`
		Status                 string `json:"status"`
		Action                 string `json:"action"`
		SourceIntentID         string `json:"source_admission_final_gate_intent_id"`
		SourceIntentHash       string `json:"source_admission_final_gate_intent_hash"`
		SourceIntentRead       string `json:"source_admission_final_gate_intent_read_back_hash"`
		CausalID               string `json:"causal_id"`
		PreHash                string `json:"pre_state_hash"`
		PostHash               string `json:"post_state_hash"`
		DeltaHash              string `json:"state_delta_hash"`
		State                  string `json:"admission_final_gate_receiver_state"`
		ActionReceiver         string `json:"admission_final_gate_receiver_action"`
		Ready                  bool   `json:"weighted_receiver_ready"`
		ReceiverReady          bool   `json:"admission_final_gate_receiver_ready"`
		IntentVerified         bool   `json:"admission_final_gate_receiver_intent_verified"`
		WriteAllowed           bool   `json:"write_allowed"`
		AdmissionAllowed       bool   `json:"admission_allowed"`
		LiveAdmissionEnabled   bool   `json:"live_admission_enabled"`
		MutatesState           bool   `json:"mutates_state"`
		NextStepBlockedWithout bool   `json:"next_step_blocked_without_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_final_gate_receiver"`
	}{
		Schema:                 report.Schema,
		Status:                 report.Status,
		Action:                 report.Action,
		SourceIntentID:         report.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateIntentID,
		SourceIntentHash:       report.SourceAdmissionFinalGateIntentHash,
		SourceIntentRead:       report.SourceAdmissionFinalGateIntentReadBack,
		CausalID:               report.CausalID,
		PreHash:                report.AdmissionFinalGateReceiverPreStateHash,
		PostHash:               report.AdmissionFinalGateReceiverPostStateHash,
		DeltaHash:              report.AdmissionFinalGateReceiverStateDeltaHash,
		State:                  report.AdmissionFinalGateReceiverState,
		ActionReceiver:         report.AdmissionFinalGateReceiverAction,
		Ready:                  report.WeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReceiverReady,
		ReceiverReady:          report.AdmissionFinalGateReceiverReady,
		IntentVerified:         report.AdmissionFinalGateReceiverIntentVerified,
		WriteAllowed:           report.WriteAllowed,
		AdmissionAllowed:       report.AdmissionAllowed,
		LiveAdmissionEnabled:   report.LiveAdmissionEnabled,
		MutatesState:           report.MutatesState,
		NextStepBlockedWithout: report.NextStepBlockedWithoutResonanceGraftAdmissionFinalGateReceiver,
	})
	if h == "" {
		return ""
	}
	return "weighted-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-candidate-store-reader-proof-precondition-decision-promotion-switch-enable-gate-live-stage-final-gate-receiver-id-" + h
}

func readAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReceiverReportForAssert(path string) (admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReceiverReport, map[string]json.RawMessage, error) {
	var report admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReceiverReport
	if strings.TrimSpace(path) == "" {
		return report, nil, fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage final gate receiver path missing")
	}
	raw, err := os.ReadFile(path)
	if err != nil {
		if os.IsNotExist(err) {
			return report, nil, fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage final gate receiver not written")
		}
		return report, nil, err
	}
	if len(raw) == 0 {
		return report, nil, fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage final gate receiver not written")
	}
	var root map[string]json.RawMessage
	if err := json.Unmarshal(raw, &root); err != nil {
		return report, nil, fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage final gate receiver JSON invalid: %w", err)
	}
	if err := json.Unmarshal(raw, &report); err != nil {
		return report, nil, fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage final gate receiver decode failed: %w", err)
	}
	return report, root, nil
}
