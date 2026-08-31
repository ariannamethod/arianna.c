package main

import (
	"encoding/json"
	"fmt"
	"os"
	"strings"
)

const admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateSchema = "arianna.live_route_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_final_gate.v1"

type admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReport struct {
	admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageSealReport

	AdmissionFinalGateState                                                                                                                                                            string `json:"admission_final_gate_state"`
	AdmissionFinalGateAction                                                                                                                                                           string `json:"admission_final_gate_action"`
	AdmissionFinalGateTarget                                                                                                                                                           string `json:"admission_final_gate_target"`
	AdmissionFinalGateTargetKind                                                                                                                                                       string `json:"admission_final_gate_target_kind"`
	AdmissionFinalGateTargetMode                                                                                                                                                       string `json:"admission_final_gate_target_mode"`
	AdmissionFinalGateDryRunOnly                                                                                                                                                       bool   `json:"admission_final_gate_dry_run_only"`
	AdmissionFinalGateSealVerified                                                                                                                                                     bool   `json:"admission_final_gate_seal_verified"`
	AdmissionFinalGateAuthorityVerified                                                                                                                                                bool   `json:"admission_final_gate_authority_verified"`
	AdmissionFinalGatePermitVerified                                                                                                                                                   bool   `json:"admission_final_gate_permit_verified"`
	AdmissionFinalGateLedgerVerified                                                                                                                                                   bool   `json:"admission_final_gate_ledger_verified"`
	AdmissionFinalGateReady                                                                                                                                                            bool   `json:"admission_final_gate_ready"`
	WeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReady     bool   `json:"weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_final_gate_ready"`
	WeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageSealConsumed       bool   `json:"weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_seal_consumed"`
	WeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageSealRequired       bool   `json:"weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_seal_required"`
	NextStepBlockedWithoutResonanceGraftAdmissionFinalGate                                                                                                                             bool   `json:"next_step_blocked_without_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_final_gate"`
	WeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateID        string `json:"weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_final_gate_id"`
	AdmissionFinalGateHash                                                                                                                                                             string `json:"admission_final_gate_hash"`
	AdmissionFinalGateReadBackHash                                                                                                                                                     string `json:"admission_final_gate_read_back_hash"`
	SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageSealID       string `json:"source_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_seal_id"`
	SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageSealReady    bool   `json:"source_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_seal_ready"`
	SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageSealCausalID string `json:"source_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_seal_causal_id"`
	SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageSealHash     string `json:"source_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_seal_hash"`
	SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageSealReadBack string `json:"source_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_seal_read_back_hash"`
	SourceAdmissionSealReportReceiptShape                                                                                                                                              string `json:"source_admission_seal_report_receipt_shape"`
	SourceAdmissionSealState                                                                                                                                                           string `json:"source_admission_seal_state"`
	SourceAdmissionSealAction                                                                                                                                                          string `json:"source_admission_seal_action"`
	SourceAdmissionSealTarget                                                                                                                                                          string `json:"source_admission_seal_target"`
	SourceAdmissionSealTargetKind                                                                                                                                                      string `json:"source_admission_seal_target_kind"`
	SourceAdmissionSealTargetMode                                                                                                                                                      string `json:"source_admission_seal_target_mode"`
	SourceAdmissionSealDryRunOnly                                                                                                                                                      bool   `json:"source_admission_seal_dry_run_only"`
	SourceAdmissionSealAuthorityVerified                                                                                                                                               bool   `json:"source_admission_seal_authority_verified"`
	SourceAdmissionSealPermitVerified                                                                                                                                                  bool   `json:"source_admission_seal_permit_verified"`
	SourceAdmissionSealLedgerVerified                                                                                                                                                  bool   `json:"source_admission_seal_ledger_verified"`
	SourceAdmissionSealReady                                                                                                                                                           bool   `json:"source_admission_seal_ready"`
	SourceAdmissionSealImmutableReceipt                                                                                                                                                bool   `json:"source_admission_seal_immutable_receipt"`
	SourceAdmissionSealReason                                                                                                                                                          string `json:"source_admission_seal_reason"`
	SourceAdmissionAuthoritySchema                                                                                                                                                     string `json:"source_admission_authority_schema"`
}

func runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGate(args []string) error {
	if len(args) != 2 {
		return fmt.Errorf("usage: --admission-live-route-weighted-admission-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-candidate-store-reader-proof-precondition-decision-promotion-switch-enable-gate-live-stage-final-gate RESONANCE_GRAFT_ADMISSION_FINAL_GATE_OBSERVATION_BOUNDARY_PREFLIGHT_GATE_CANDIDATE_STORE_READER_PROOF_PRECONDITION_DECISION_PROMOTION_SWITCH_ENABLE_GATE_LIVE_STAGE_SEAL_REPORT RESONANCE_GRAFT_ADMISSION_FINAL_GATE_OBSERVATION_BOUNDARY_PREFLIGHT_GATE_CANDIDATE_STORE_READER_PROOF_PRECONDITION_DECISION_PROMOTION_SWITCH_ENABLE_GATE_LIVE_STAGE_FINAL_GATE_REPORT")
	}
	sealPath := args[0]
	outputPath := args[1]
	if strings.TrimSpace(outputPath) == "" {
		return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage final gate output path missing")
	}
	sourceSeal, root, err := readAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageSealReportForAssert(sealPath)
	if err != nil {
		return err
	}
	if err := admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageSealReportError(sourceSeal, root); err != nil {
		return err
	}
	finalGate := admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReport{
		admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageSealReport: sourceSeal,
		AdmissionFinalGateState:             "blocked",
		AdmissionFinalGateAction:            "reject_blocked_admission_seal",
		AdmissionFinalGateTarget:            "live_admission_final_gate",
		AdmissionFinalGateTargetKind:        "weighted_internal_world_shadow_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_seal",
		AdmissionFinalGateTargetMode:        "closed_final_gate_guard_dry_run",
		AdmissionFinalGateDryRunOnly:        true,
		AdmissionFinalGateSealVerified:      false,
		AdmissionFinalGateAuthorityVerified: false,
		AdmissionFinalGatePermitVerified:    false,
		AdmissionFinalGateLedgerVerified:    false,
		AdmissionFinalGateReady:             false,
		WeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReady: true,
		WeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageSealConsumed:   true,
		WeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageSealRequired:   true,
		NextStepBlockedWithoutResonanceGraftAdmissionFinalGate: true,
		SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageSealID:       sourceSeal.WeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageSealID,
		SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageSealReady:    sourceSeal.WeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageSealReady,
		SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageSealCausalID: sourceSeal.CausalID,
		SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageSealHash:     sourceSeal.AdmissionSealHash,
		SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageSealReadBack: sourceSeal.AdmissionSealReadBackHash,
		SourceAdmissionSealReportReceiptShape: sourceSeal.ReceiptShape,
		SourceAdmissionSealState:              sourceSeal.AdmissionSealState,
		SourceAdmissionSealAction:             sourceSeal.AdmissionSealAction,
		SourceAdmissionSealTarget:             sourceSeal.AdmissionSealTarget,
		SourceAdmissionSealTargetKind:         sourceSeal.AdmissionSealTargetKind,
		SourceAdmissionSealTargetMode:         sourceSeal.AdmissionSealTargetMode,
		SourceAdmissionSealDryRunOnly:         sourceSeal.AdmissionSealDryRunOnly,
		SourceAdmissionSealAuthorityVerified:  sourceSeal.AdmissionSealAuthorityVerified,
		SourceAdmissionSealPermitVerified:     sourceSeal.AdmissionSealPermitVerified,
		SourceAdmissionSealLedgerVerified:     sourceSeal.AdmissionSealLedgerVerified,
		SourceAdmissionSealReady:              sourceSeal.AdmissionSealReady,
		SourceAdmissionSealImmutableReceipt:   sourceSeal.AdmissionSealImmutableReceipt,
		SourceAdmissionSealReason:             sourceSeal.Reason,
		SourceAdmissionAuthoritySchema:        sourceSeal.SourceSchema,
	}
	finalGate.Schema = admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateSchema
	finalGate.Status = "shadow_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_final_gate_blocked_dry_run"
	finalGate.TargetKind = "weighted_internal_world_shadow_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_final_gate"
	finalGate.TargetMode = "closed_final_gate_guard_dry_run"
	finalGate.Action = "block_weighted_resonance_shadow_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_seal_blocked_dry_run"
	finalGate.WriterAction = "reject_blocked_admission_seal"
	finalGate.RollbackAction = "reject_blocked_admission_seal"
	finalGate.LedgerState = "blocked"
	finalGate.LedgerAction = "reject_blocked_admission_seal"
	finalGate.LedgerContract = "none"
	finalGate.LedgerEntrypoint = "none"
	finalGate.LedgerReceiptShape = "none"
	finalGate.LedgerWriteScope = "none"
	finalGate.LedgerReady = false
	finalGate.LedgerAppendAllowed = false
	finalGate.ReceiptShape = "weighted_resonance_shadow_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_final_gate_receipt"
	finalGate.SourceSchema = sourceSeal.Schema
	finalGate.SourceStatus = sourceSeal.Status
	finalGate.SourceTarget = sourceSeal.Target
	finalGate.SourceReport = sealPath
	finalGate.AuthorityGranted = false
	finalGate.Reason = "weighted resonance shadow graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage final gate blocked by blocked seal; final admission remains closed"
	finalGate.CausalID = admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateCausalID(finalGate)
	finalGate.AdmissionFinalGateHash = admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateHash(finalGate)
	finalGate.AdmissionFinalGateReadBackHash = admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReadBackHash(finalGate)
	finalGate.WeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateID = admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateID(finalGate)
	if finalGate.CausalID == "" ||
		finalGate.AdmissionFinalGateHash == "" ||
		finalGate.AdmissionFinalGateReadBackHash == "" ||
		finalGate.WeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateID == "" ||
		finalGate.AdmissionFinalGateHash == finalGate.AdmissionFinalGateReadBackHash {
		return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage final gate read-back proof failed")
	}
	raw, err := json.MarshalIndent(finalGate, "", "  ")
	if err != nil {
		return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage final gate marshal failed: %w", err)
	}
	raw = append(raw, '\n')
	if err := os.WriteFile(outputPath, raw, 0600); err != nil {
		return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage final gate write failed: %w", err)
	}
	fmt.Printf("[admission-live-route-weighted-admission-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-candidate-store-reader-proof-precondition-decision-promotion-switch-enable-gate-live-stage-final-gate] pass: resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_final_gate_report=%s resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_seal_report=%s\n", outputPath, sealPath)
	return nil
}

func runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateAssert(args []string) error {
	if len(args) != 1 {
		return fmt.Errorf("usage: --admission-live-route-weighted-admission-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-candidate-store-reader-proof-precondition-decision-promotion-switch-enable-gate-live-stage-final-gate-assert REPORT")
	}
	report, root, err := readAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReportForAssert(args[0])
	if err != nil {
		return err
	}
	return admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReportError(report, root)
}

func admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReportError(report admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReport, root map[string]json.RawMessage) error {
	if _, ok := root["schema"]; !ok {
		return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage final gate schema missing")
	}
	if report.Schema != admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateSchema {
		return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage final gate schema mismatch: got %q want %q", report.Schema, admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateSchema)
	}
	if report.Status != "shadow_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_final_gate_blocked_dry_run" {
		return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage final gate status mismatch: got %q want %q", report.Status, "shadow_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_final_gate_blocked_dry_run")
	}
	if report.Target != "live_route_admission_next_step" {
		return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage final gate target mismatch: got %q want %q", report.Target, "live_route_admission_next_step")
	}
	if report.TargetKind != "weighted_internal_world_shadow_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_final_gate" {
		return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage final gate target_kind mismatch: got %q want %q", report.TargetKind, "weighted_internal_world_shadow_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_final_gate")
	}
	if report.TargetMode != "closed_final_gate_guard_dry_run" {
		return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage final gate target_mode mismatch: got %q want %q", report.TargetMode, "closed_final_gate_guard_dry_run")
	}
	if report.Action != "block_weighted_resonance_shadow_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_seal_blocked_dry_run" {
		return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage final gate action mismatch: got %q want %q", report.Action, "block_weighted_resonance_shadow_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_seal_blocked_dry_run")
	}
	if report.WriterAction != "reject_blocked_admission_seal" || report.RollbackAction != "reject_blocked_admission_seal" {
		return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage final gate writer/rollback action mismatch")
	}
	if report.LedgerState != "blocked" ||
		report.LedgerAction != "reject_blocked_admission_seal" ||
		report.LedgerContract != "none" ||
		report.LedgerEntrypoint != "none" ||
		report.LedgerReceiptShape != "none" ||
		report.LedgerWriteScope != "none" {
		return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage final gate ledger guard mismatch")
	}
	if report.AdmissionFinalGateState != "blocked" ||
		report.AdmissionFinalGateAction != "reject_blocked_admission_seal" ||
		report.AdmissionFinalGateTarget != "live_admission_final_gate" ||
		report.AdmissionFinalGateTargetKind != "weighted_internal_world_shadow_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_seal" ||
		report.AdmissionFinalGateTargetMode != "closed_final_gate_guard_dry_run" {
		return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage final gate shape mismatch")
	}
	if report.ReceiptShape != "weighted_resonance_shadow_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_final_gate_receipt" {
		return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage final gate receipt_shape mismatch: got %q want %q", report.ReceiptShape, "weighted_resonance_shadow_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_final_gate_receipt")
	}
	for _, required := range []struct {
		name  string
		value bool
	}{
		{"weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_final_gate_ready", report.WeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReady},
		{"weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_seal_consumed", report.WeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageSealConsumed},
		{"weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_seal_required", report.WeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageSealRequired},
		{"next_step_blocked_without_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_final_gate", report.NextStepBlockedWithoutResonanceGraftAdmissionFinalGate},
		{"source_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_seal_ready", report.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageSealReady},
		{"weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_seal_ready", report.WeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageSealReady},
		{"weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_authority_consumed", report.WeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageAuthorityConsumed},
		{"weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_authority_required", report.WeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageAuthorityRequired},
		{"weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_authority_ready", report.WeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageAuthorityReady},
		{"weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_permit_consumed", report.WeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStagePermitConsumed},
		{"weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_permit_required", report.WeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStagePermitRequired},
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
			return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage final gate %s not ready", required.name)
		}
	}
	for _, closed := range []struct {
		name  string
		value bool
	}{
		{"admission_final_gate_seal_verified", report.AdmissionFinalGateSealVerified},
		{"admission_final_gate_authority_verified", report.AdmissionFinalGateAuthorityVerified},
		{"admission_final_gate_permit_verified", report.AdmissionFinalGatePermitVerified},
		{"admission_final_gate_ledger_verified", report.AdmissionFinalGateLedgerVerified},
		{"admission_final_gate_ready", report.AdmissionFinalGateReady},
		{"source_admission_seal_authority_verified", report.SourceAdmissionSealAuthorityVerified},
		{"source_admission_seal_permit_verified", report.SourceAdmissionSealPermitVerified},
		{"source_admission_seal_ledger_verified", report.SourceAdmissionSealLedgerVerified},
		{"source_admission_seal_ready", report.SourceAdmissionSealReady},
		{"admission_seal_authority_verified", report.AdmissionSealAuthorityVerified},
		{"admission_seal_permit_verified", report.AdmissionSealPermitVerified},
		{"admission_seal_ledger_verified", report.AdmissionSealLedgerVerified},
		{"admission_seal_ready", report.AdmissionSealReady},
		{"source_admission_authority_permit_verified", report.SourceAdmissionAuthorityPermitVerified},
		{"source_admission_authority_ledger_verified", report.SourceAdmissionAuthorityLedgerVerified},
		{"source_admission_authority_writer_ready", report.SourceAdmissionAuthorityWriterReady},
		{"source_admission_authority_rollback_ready", report.SourceAdmissionAuthorityRollbackReady},
		{"source_admission_authority_ready", report.SourceAdmissionAuthorityReady},
		{"source_admission_authority_granted", report.SourceAdmissionAuthorityGranted},
		{"admission_authority_permit_verified", report.AdmissionAuthorityPermitVerified},
		{"admission_authority_ledger_verified", report.AdmissionAuthorityLedgerVerified},
		{"admission_authority_writer_ready", report.AdmissionAuthorityWriterReady},
		{"admission_authority_rollback_ready", report.AdmissionAuthorityRollbackReady},
		{"admission_authority_ready", report.AdmissionAuthorityReady},
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
			return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage final gate opened %s", closed.name)
		}
	}
	if !report.AdmissionFinalGateDryRunOnly || !report.AdmissionSealDryRunOnly || !report.SourceAdmissionSealDryRunOnly {
		return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage final gate dry-run flag mismatch")
	}
	for _, requiredString := range []struct {
		name  string
		value string
	}{
		{"weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_final_gate_id", report.WeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateID},
		{"causal_id", report.CausalID},
		{"admission_final_gate_hash", report.AdmissionFinalGateHash},
		{"admission_final_gate_read_back_hash", report.AdmissionFinalGateReadBackHash},
		{"source_report", report.SourceReport},
		{"source_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_seal_id", report.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageSealID},
		{"source_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_seal_causal_id", report.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageSealCausalID},
		{"source_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_seal_hash", report.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageSealHash},
		{"source_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_seal_read_back_hash", report.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageSealReadBack},
		{"source_admission_seal_reason", report.SourceAdmissionSealReason},
		{"source_admission_authority_schema", report.SourceAdmissionAuthoritySchema},
		{"source_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_authority_id", report.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageAuthorityID},
		{"source_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_permit_id", report.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStagePermitID},
	} {
		if strings.TrimSpace(requiredString.value) == "" {
			return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage final gate %s missing", requiredString.name)
		}
	}
	if report.SourceSchema != admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageSealSchema {
		return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage final gate source_schema mismatch: got %q want %q", report.SourceSchema, admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageSealSchema)
	}
	if report.SourceStatus != "shadow_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_seal_blocked_dry_run" {
		return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage final gate source_status mismatch: got %q want %q", report.SourceStatus, "shadow_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_seal_blocked_dry_run")
	}
	if report.SourceTarget != "live_route_admission_next_step" {
		return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage final gate source_target mismatch: got %q want %q", report.SourceTarget, "live_route_admission_next_step")
	}
	if report.SourceAdmissionAuthoritySchema != admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageAuthoritySchema {
		return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage final gate source_admission_authority_schema mismatch: got %q want %q", report.SourceAdmissionAuthoritySchema, admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageAuthoritySchema)
	}
	if report.SourceAdmissionSealReportReceiptShape != "weighted_resonance_shadow_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_seal_receipt" ||
		report.SourceAdmissionSealState != "sealed" ||
		report.SourceAdmissionSealAction != "seal_blocked_admission_authority" ||
		report.SourceAdmissionSealTarget != "live_admission_authority" ||
		report.SourceAdmissionSealTargetKind != "weighted_internal_world_shadow_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_authority" ||
		report.SourceAdmissionSealTargetMode != "closed_seal_guard_dry_run" ||
		!report.SourceAdmissionSealImmutableReceipt {
		return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage final gate source admission seal shape mismatch")
	}
	if report.SourceAdmissionSealReason != "weighted resonance shadow graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage seal fixed blocked authority provenance; live authority remains closed" {
		return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage final gate source_admission_seal_reason mismatch: got %q", report.SourceAdmissionSealReason)
	}
	if !strings.HasPrefix(report.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageSealID, "weighted-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-candidate-store-reader-proof-precondition-decision-promotion-switch-enable-gate-live-stage-seal-id-") ||
		!strings.HasPrefix(report.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageSealCausalID, "weighted-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-candidate-store-reader-proof-precondition-decision-promotion-switch-enable-gate-live-stage-seal-causal-") ||
		!strings.HasPrefix(report.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageSealHash, "weighted-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-candidate-store-reader-proof-precondition-decision-promotion-switch-enable-gate-live-stage-seal-") ||
		!strings.HasPrefix(report.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageSealReadBack, "weighted-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-candidate-store-reader-proof-precondition-decision-promotion-switch-enable-gate-live-stage-seal-read-") {
		return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage final gate source seal mismatch")
	}
	if report.BodyTarget != "none" {
		return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage final gate body_target mismatch: got %q want %q", report.BodyTarget, "none")
	}
	if report.CausalID == "" || report.CausalID != admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateCausalID(report) {
		return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage final gate causal_id mismatch")
	}
	if report.AdmissionFinalGateHash == "" || report.AdmissionFinalGateHash != admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateHash(report) {
		return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage final gate admission_final_gate_hash mismatch")
	}
	if report.AdmissionFinalGateReadBackHash == "" || report.AdmissionFinalGateReadBackHash != admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReadBackHash(report) {
		return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage final gate admission_final_gate_read_back_hash mismatch")
	}
	if report.AdmissionFinalGateHash == report.AdmissionFinalGateReadBackHash {
		return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage final gate read-back proof collapsed")
	}
	if report.WeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateID == "" || report.WeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateID != admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateID(report) {
		return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage final gate id mismatch")
	}
	if report.Reason != "weighted resonance shadow graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage final gate blocked by blocked seal; final admission remains closed" {
		return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage final gate reason mismatch: got %q", report.Reason)
	}
	return nil
}

func admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateCausalID(report admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReport) string {
	h := hashJSON(struct {
		SourceSealID      string `json:"source_admission_seal_id"`
		SourceSealRead    string `json:"source_admission_seal_read_back_hash"`
		SourceAuthorityID string `json:"source_admission_authority_id"`
		Target            string `json:"target"`
		State             string `json:"admission_final_gate_state"`
		Action            string `json:"admission_final_gate_action"`
	}{
		SourceSealID:      report.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageSealID,
		SourceSealRead:    report.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageSealReadBack,
		SourceAuthorityID: report.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageAuthorityID,
		Target:            report.Target,
		State:             report.AdmissionFinalGateState,
		Action:            report.AdmissionFinalGateAction,
	})
	if h == "" {
		return ""
	}
	return "weighted-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-candidate-store-reader-proof-precondition-decision-promotion-switch-enable-gate-live-stage-final-gate-causal-" + h
}

func admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateHash(report admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReport) string {
	h := hashJSON(struct {
		CausalID               string `json:"causal_id"`
		SourceSealID           string `json:"source_admission_seal_id"`
		SourceSealHash         string `json:"source_admission_seal_hash"`
		SourceSealRead         string `json:"source_admission_seal_read_back_hash"`
		State                  string `json:"admission_final_gate_state"`
		Action                 string `json:"admission_final_gate_action"`
		Target                 string `json:"admission_final_gate_target"`
		TargetKind             string `json:"admission_final_gate_target_kind"`
		TargetMode             string `json:"admission_final_gate_target_mode"`
		DryRunOnly             bool   `json:"admission_final_gate_dry_run_only"`
		SealVerified           bool   `json:"admission_final_gate_seal_verified"`
		AuthorityVerified      bool   `json:"admission_final_gate_authority_verified"`
		PermitVerified         bool   `json:"admission_final_gate_permit_verified"`
		LedgerVerified         bool   `json:"admission_final_gate_ledger_verified"`
		Ready                  bool   `json:"admission_final_gate_ready"`
		WeightedReady          bool   `json:"weighted_final_gate_ready"`
		SourceWeightedReady    bool   `json:"source_weighted_seal_ready"`
		SourceAdmissionSeal    bool   `json:"source_admission_seal_ready"`
		SourceSealAuthority    bool   `json:"source_admission_seal_authority_verified"`
		WriteAllowed           bool   `json:"write_allowed"`
		AdmissionAllowed       bool   `json:"admission_allowed"`
		LiveAdmissionEnabled   bool   `json:"live_admission_enabled"`
		MutatesState           bool   `json:"mutates_state"`
		BodyMutationAllowed    bool   `json:"body_mutation_allowed"`
		NextStepBlockedWithout bool   `json:"next_step_blocked_without_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_final_gate"`
	}{
		CausalID:               report.CausalID,
		SourceSealID:           report.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageSealID,
		SourceSealHash:         report.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageSealHash,
		SourceSealRead:         report.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageSealReadBack,
		State:                  report.AdmissionFinalGateState,
		Action:                 report.AdmissionFinalGateAction,
		Target:                 report.AdmissionFinalGateTarget,
		TargetKind:             report.AdmissionFinalGateTargetKind,
		TargetMode:             report.AdmissionFinalGateTargetMode,
		DryRunOnly:             report.AdmissionFinalGateDryRunOnly,
		SealVerified:           report.AdmissionFinalGateSealVerified,
		AuthorityVerified:      report.AdmissionFinalGateAuthorityVerified,
		PermitVerified:         report.AdmissionFinalGatePermitVerified,
		LedgerVerified:         report.AdmissionFinalGateLedgerVerified,
		Ready:                  report.AdmissionFinalGateReady,
		WeightedReady:          report.WeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReady,
		SourceWeightedReady:    report.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageSealReady,
		SourceAdmissionSeal:    report.SourceAdmissionSealReady,
		SourceSealAuthority:    report.SourceAdmissionSealAuthorityVerified,
		WriteAllowed:           report.WriteAllowed,
		AdmissionAllowed:       report.AdmissionAllowed,
		LiveAdmissionEnabled:   report.LiveAdmissionEnabled,
		MutatesState:           report.MutatesState,
		BodyMutationAllowed:    report.BodyMutationAllowed,
		NextStepBlockedWithout: report.NextStepBlockedWithoutResonanceGraftAdmissionFinalGate,
	})
	if h == "" {
		return ""
	}
	return "weighted-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-candidate-store-reader-proof-precondition-decision-promotion-switch-enable-gate-live-stage-final-gate-" + h
}

func admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReadBackHash(report admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReport) string {
	h := hashJSON(struct {
		AdmissionFinalGateHash string `json:"admission_final_gate_hash"`
		SourceSealID           string `json:"source_admission_seal_id"`
		SourceSealRead         string `json:"source_admission_seal_read_back_hash"`
		WeightedReady          bool   `json:"weighted_final_gate_ready"`
		SealConsumed           bool   `json:"seal_consumed"`
		SealRequired           bool   `json:"seal_required"`
		FinalGateReady         bool   `json:"admission_final_gate_ready"`
		SealVerified           bool   `json:"admission_final_gate_seal_verified"`
		WriteAllowed           bool   `json:"write_allowed"`
		AdmissionAllowed       bool   `json:"admission_allowed"`
		LiveEnabled            bool   `json:"live_admission_enabled"`
		MutatesState           bool   `json:"mutates_state"`
	}{
		AdmissionFinalGateHash: report.AdmissionFinalGateHash,
		SourceSealID:           report.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageSealID,
		SourceSealRead:         report.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageSealReadBack,
		WeightedReady:          report.WeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReady,
		SealConsumed:           report.WeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageSealConsumed,
		SealRequired:           report.WeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageSealRequired,
		FinalGateReady:         report.AdmissionFinalGateReady,
		SealVerified:           report.AdmissionFinalGateSealVerified,
		WriteAllowed:           report.WriteAllowed,
		AdmissionAllowed:       report.AdmissionAllowed,
		LiveEnabled:            report.LiveAdmissionEnabled,
		MutatesState:           report.MutatesState,
	})
	if h == "" {
		return ""
	}
	return "weighted-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-candidate-store-reader-proof-precondition-decision-promotion-switch-enable-gate-live-stage-final-gate-read-" + h
}

func admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateID(report admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReport) string {
	h := hashJSON(struct {
		Schema                 string `json:"schema"`
		Status                 string `json:"status"`
		Action                 string `json:"action"`
		SourceSealID           string `json:"source_admission_seal_id"`
		SourceSealHash         string `json:"source_admission_seal_hash"`
		SourceSealRead         string `json:"source_admission_seal_read_back_hash"`
		CausalID               string `json:"causal_id"`
		AdmissionFinalGateHash string `json:"admission_final_gate_hash"`
		AdmissionFinalGateRead string `json:"admission_final_gate_read_back_hash"`
		State                  string `json:"admission_final_gate_state"`
		ActionGate             string `json:"admission_final_gate_action"`
		Ready                  bool   `json:"weighted_final_gate_ready"`
		FinalGateReady         bool   `json:"admission_final_gate_ready"`
		SealVerified           bool   `json:"admission_final_gate_seal_verified"`
		WriteAllowed           bool   `json:"write_allowed"`
		AdmissionAllowed       bool   `json:"admission_allowed"`
		LiveAdmissionEnabled   bool   `json:"live_admission_enabled"`
		MutatesState           bool   `json:"mutates_state"`
		NextStepBlockedWithout bool   `json:"next_step_blocked_without_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_final_gate"`
	}{
		Schema:                 report.Schema,
		Status:                 report.Status,
		Action:                 report.Action,
		SourceSealID:           report.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageSealID,
		SourceSealHash:         report.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageSealHash,
		SourceSealRead:         report.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageSealReadBack,
		CausalID:               report.CausalID,
		AdmissionFinalGateHash: report.AdmissionFinalGateHash,
		AdmissionFinalGateRead: report.AdmissionFinalGateReadBackHash,
		State:                  report.AdmissionFinalGateState,
		ActionGate:             report.AdmissionFinalGateAction,
		Ready:                  report.WeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReady,
		FinalGateReady:         report.AdmissionFinalGateReady,
		SealVerified:           report.AdmissionFinalGateSealVerified,
		WriteAllowed:           report.WriteAllowed,
		AdmissionAllowed:       report.AdmissionAllowed,
		LiveAdmissionEnabled:   report.LiveAdmissionEnabled,
		MutatesState:           report.MutatesState,
		NextStepBlockedWithout: report.NextStepBlockedWithoutResonanceGraftAdmissionFinalGate,
	})
	if h == "" {
		return ""
	}
	return "weighted-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-candidate-store-reader-proof-precondition-decision-promotion-switch-enable-gate-live-stage-final-gate-id-" + h
}

func readAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReportForAssert(path string) (admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReport, map[string]json.RawMessage, error) {
	var report admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReport
	if strings.TrimSpace(path) == "" {
		return report, nil, fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage final gate path missing")
	}
	raw, err := os.ReadFile(path)
	if err != nil {
		if os.IsNotExist(err) {
			return report, nil, fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage final gate not written")
		}
		return report, nil, err
	}
	if len(raw) == 0 {
		return report, nil, fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage final gate not written")
	}
	var root map[string]json.RawMessage
	if err := json.Unmarshal(raw, &root); err != nil {
		return report, nil, fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage final gate JSON invalid: %w", err)
	}
	if err := json.Unmarshal(raw, &report); err != nil {
		return report, nil, fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage final gate decode failed: %w", err)
	}
	return report, root, nil
}
