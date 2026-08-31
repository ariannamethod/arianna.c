package main

import (
	"encoding/json"
	"fmt"
	"os"
	"strings"
)

const admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageSealSchema = "arianna.live_route_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_seal.v1"

type admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageSealReport struct {
	admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageAuthorityReport

	AdmissionSealState                                                                                                                                                                string `json:"admission_seal_state"`
	AdmissionSealAction                                                                                                                                                               string `json:"admission_seal_action"`
	AdmissionSealTarget                                                                                                                                                               string `json:"admission_seal_target"`
	AdmissionSealTargetKind                                                                                                                                                           string `json:"admission_seal_target_kind"`
	AdmissionSealTargetMode                                                                                                                                                           string `json:"admission_seal_target_mode"`
	AdmissionSealDryRunOnly                                                                                                                                                           bool   `json:"admission_seal_dry_run_only"`
	AdmissionSealAuthorityVerified                                                                                                                                                    bool   `json:"admission_seal_authority_verified"`
	AdmissionSealPermitVerified                                                                                                                                                       bool   `json:"admission_seal_permit_verified"`
	AdmissionSealLedgerVerified                                                                                                                                                       bool   `json:"admission_seal_ledger_verified"`
	AdmissionSealReady                                                                                                                                                                bool   `json:"admission_seal_ready"`
	AdmissionSealImmutableReceipt                                                                                                                                                     bool   `json:"admission_seal_immutable_receipt"`
	WeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageSealReady         bool   `json:"weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_seal_ready"`
	WeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageAuthorityConsumed bool   `json:"weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_authority_consumed"`
	WeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageAuthorityRequired bool   `json:"weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_authority_required"`
	NextStepBlockedWithoutResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageSeal         bool   `json:"next_step_blocked_without_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_seal"`
	WeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageSealID            string `json:"weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_seal_id"`
	AdmissionSealHash                                                                                                                                                                 string `json:"admission_seal_hash"`
	AdmissionSealReadBackHash                                                                                                                                                         string `json:"admission_seal_read_back_hash"`

	SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageAuthorityID       string `json:"source_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_authority_id"`
	SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageAuthorityReady    bool   `json:"source_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_authority_ready"`
	SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageAuthorityCausalID string `json:"source_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_authority_causal_id"`
	SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageAuthorityHash     string `json:"source_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_authority_hash"`
	SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageAuthorityReadBack string `json:"source_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_authority_read_back_hash"`
	SourceAdmissionAuthorityReportReceiptShape                                                                                                                                              string `json:"source_admission_authority_report_receipt_shape"`
	SourceAdmissionAuthorityState                                                                                                                                                           string `json:"source_admission_authority_state"`
	SourceAdmissionAuthorityAction                                                                                                                                                          string `json:"source_admission_authority_action"`
	SourceAdmissionAuthorityTarget                                                                                                                                                          string `json:"source_admission_authority_target"`
	SourceAdmissionAuthorityTargetKind                                                                                                                                                      string `json:"source_admission_authority_target_kind"`
	SourceAdmissionAuthorityTargetMode                                                                                                                                                      string `json:"source_admission_authority_target_mode"`
	SourceAdmissionAuthorityDryRunOnly                                                                                                                                                      bool   `json:"source_admission_authority_dry_run_only"`
	SourceAdmissionAuthorityPermitVerified                                                                                                                                                  bool   `json:"source_admission_authority_permit_verified"`
	SourceAdmissionAuthorityLedgerVerified                                                                                                                                                  bool   `json:"source_admission_authority_ledger_verified"`
	SourceAdmissionAuthorityWriterReady                                                                                                                                                     bool   `json:"source_admission_authority_writer_ready"`
	SourceAdmissionAuthorityRollbackReady                                                                                                                                                   bool   `json:"source_admission_authority_rollback_ready"`
	SourceAdmissionAuthorityReady                                                                                                                                                           bool   `json:"source_admission_authority_ready"`
	SourceAdmissionAuthorityGranted                                                                                                                                                         bool   `json:"source_admission_authority_granted"`
	SourceAdmissionAuthorityReason                                                                                                                                                          string `json:"source_admission_authority_reason"`
	SourceAdmissionPermitSchema                                                                                                                                                             string `json:"source_admission_permit_schema"`
}

func runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageSeal(args []string) error {
	if len(args) != 2 {
		return fmt.Errorf("usage: --admission-live-route-weighted-admission-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-candidate-store-reader-proof-precondition-decision-promotion-switch-enable-gate-live-stage-seal RESONANCE_GRAFT_ADMISSION_AUTHORITY_REPORT RESONANCE_GRAFT_ADMISSION_SEAL_REPORT")
	}
	authorityPath := args[0]
	outputPath := args[1]
	if strings.TrimSpace(outputPath) == "" {
		return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage seal output path missing")
	}
	sourceAuthority, root, err := readAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageAuthorityReportForAssert(authorityPath)
	if err != nil {
		return err
	}
	if err := admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageAuthorityReportError(sourceAuthority, root); err != nil {
		return err
	}
	seal := admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageSealReport{
		admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageAuthorityReport: sourceAuthority,
		AdmissionSealState:             "sealed",
		AdmissionSealAction:            "seal_blocked_admission_authority",
		AdmissionSealTarget:            "live_admission_authority",
		AdmissionSealTargetKind:        "weighted_internal_world_shadow_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_authority",
		AdmissionSealTargetMode:        "closed_seal_guard_dry_run",
		AdmissionSealDryRunOnly:        true,
		AdmissionSealAuthorityVerified: false,
		AdmissionSealPermitVerified:    false,
		AdmissionSealLedgerVerified:    false,
		AdmissionSealReady:             false,
		AdmissionSealImmutableReceipt:  true,
		WeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageSealReady:               true,
		WeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageAuthorityConsumed:       true,
		WeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageAuthorityRequired:       true,
		NextStepBlockedWithoutResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageSeal:               true,
		SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageAuthorityID:       sourceAuthority.WeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageAuthorityID,
		SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageAuthorityReady:    sourceAuthority.WeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageAuthorityReady,
		SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageAuthorityCausalID: sourceAuthority.CausalID,
		SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageAuthorityHash:     sourceAuthority.AdmissionAuthorityHash,
		SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageAuthorityReadBack: sourceAuthority.AdmissionAuthorityReadBackHash,
		SourceAdmissionAuthorityReportReceiptShape: sourceAuthority.ReceiptShape,
		SourceAdmissionAuthorityState:              sourceAuthority.AdmissionAuthorityState,
		SourceAdmissionAuthorityAction:             sourceAuthority.AdmissionAuthorityAction,
		SourceAdmissionAuthorityTarget:             sourceAuthority.AdmissionAuthorityTarget,
		SourceAdmissionAuthorityTargetKind:         sourceAuthority.AdmissionAuthorityTargetKind,
		SourceAdmissionAuthorityTargetMode:         sourceAuthority.AdmissionAuthorityTargetMode,
		SourceAdmissionAuthorityDryRunOnly:         sourceAuthority.AdmissionAuthorityDryRunOnly,
		SourceAdmissionAuthorityPermitVerified:     sourceAuthority.AdmissionAuthorityPermitVerified,
		SourceAdmissionAuthorityLedgerVerified:     sourceAuthority.AdmissionAuthorityLedgerVerified,
		SourceAdmissionAuthorityWriterReady:        sourceAuthority.AdmissionAuthorityWriterReady,
		SourceAdmissionAuthorityRollbackReady:      sourceAuthority.AdmissionAuthorityRollbackReady,
		SourceAdmissionAuthorityReady:              sourceAuthority.AdmissionAuthorityReady,
		SourceAdmissionAuthorityGranted:            sourceAuthority.AdmissionAuthorityGranted,
		SourceAdmissionAuthorityReason:             sourceAuthority.Reason,
		SourceAdmissionPermitSchema:                sourceAuthority.SourceSchema,
	}
	seal.Schema = admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageSealSchema
	seal.Status = "shadow_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_seal_blocked_dry_run"
	seal.TargetKind = "weighted_internal_world_shadow_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_seal"
	seal.TargetMode = "closed_seal_guard_dry_run"
	seal.Action = "seal_weighted_resonance_shadow_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_authority_blocked_dry_run"
	seal.WriterAction = "reject_blocked_admission_authority"
	seal.RollbackAction = "reject_blocked_admission_authority"
	seal.LedgerState = "blocked"
	seal.LedgerAction = "reject_blocked_admission_authority"
	seal.LedgerContract = "none"
	seal.LedgerEntrypoint = "none"
	seal.LedgerReceiptShape = "none"
	seal.LedgerWriteScope = "none"
	seal.LedgerReady = false
	seal.LedgerAppendAllowed = false
	seal.ReceiptShape = "weighted_resonance_shadow_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_seal_receipt"
	seal.SourceSchema = sourceAuthority.Schema
	seal.SourceStatus = sourceAuthority.Status
	seal.SourceTarget = sourceAuthority.Target
	seal.SourceReport = authorityPath
	seal.AuthorityGranted = false
	seal.Reason = "weighted resonance shadow graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage seal fixed blocked authority provenance; live authority remains closed"
	seal.CausalID = admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageSealCausalID(seal)
	seal.AdmissionSealHash = admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageSealHash(seal)
	seal.AdmissionSealReadBackHash = admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageSealReadBackHash(seal)
	seal.WeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageSealID = admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageSealID(seal)
	if seal.CausalID == "" ||
		seal.AdmissionSealHash == "" ||
		seal.AdmissionSealReadBackHash == "" ||
		seal.WeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageSealID == "" ||
		seal.AdmissionSealHash == seal.AdmissionSealReadBackHash {
		return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage seal read-back proof failed")
	}
	raw, err := json.MarshalIndent(seal, "", "  ")
	if err != nil {
		return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage seal marshal failed: %w", err)
	}
	raw = append(raw, '\n')
	if err := os.WriteFile(outputPath, raw, 0600); err != nil {
		return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage seal write failed: %w", err)
	}
	fmt.Printf("[admission-live-route-weighted-admission-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-candidate-store-reader-proof-precondition-decision-promotion-switch-enable-gate-live-stage-seal] pass: resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_seal_report=%s resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_authority_report=%s\n", outputPath, authorityPath)
	return nil
}

func runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageSealAssert(args []string) error {
	if len(args) != 1 {
		return fmt.Errorf("usage: --admission-live-route-weighted-admission-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-candidate-store-reader-proof-precondition-decision-promotion-switch-enable-gate-live-stage-seal-assert REPORT")
	}
	report, root, err := readAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageSealReportForAssert(args[0])
	if err != nil {
		return err
	}
	return admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageSealReportError(report, root)
}

func admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageSealReportError(report admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageSealReport, root map[string]json.RawMessage) error {
	if _, ok := root["schema"]; !ok {
		return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage seal schema missing")
	}
	if report.Schema != admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageSealSchema {
		return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage seal schema mismatch: got %q want %q", report.Schema, admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageSealSchema)
	}
	if report.Status != "shadow_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_seal_blocked_dry_run" {
		return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage seal status mismatch: got %q want %q", report.Status, "shadow_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_seal_blocked_dry_run")
	}
	if report.Target != "live_route_admission_next_step" {
		return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage seal target mismatch: got %q want %q", report.Target, "live_route_admission_next_step")
	}
	if report.TargetKind != "weighted_internal_world_shadow_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_seal" {
		return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage seal target_kind mismatch: got %q want %q", report.TargetKind, "weighted_internal_world_shadow_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_seal")
	}
	if report.TargetMode != "closed_seal_guard_dry_run" {
		return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage seal target_mode mismatch: got %q want %q", report.TargetMode, "closed_seal_guard_dry_run")
	}
	if report.Action != "seal_weighted_resonance_shadow_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_authority_blocked_dry_run" {
		return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage seal action mismatch: got %q want %q", report.Action, "seal_weighted_resonance_shadow_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_authority_blocked_dry_run")
	}
	if report.WriterAction != "reject_blocked_admission_authority" || report.RollbackAction != "reject_blocked_admission_authority" {
		return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage seal writer/rollback action mismatch")
	}
	if report.LedgerState != "blocked" ||
		report.LedgerAction != "reject_blocked_admission_authority" ||
		report.LedgerContract != "none" ||
		report.LedgerEntrypoint != "none" ||
		report.LedgerReceiptShape != "none" ||
		report.LedgerWriteScope != "none" {
		return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage seal ledger guard mismatch")
	}
	if report.AdmissionSealState != "sealed" ||
		report.AdmissionSealAction != "seal_blocked_admission_authority" ||
		report.AdmissionSealTarget != "live_admission_authority" ||
		report.AdmissionSealTargetKind != "weighted_internal_world_shadow_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_authority" ||
		report.AdmissionSealTargetMode != "closed_seal_guard_dry_run" {
		return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage seal shape mismatch")
	}
	if report.ReceiptShape != "weighted_resonance_shadow_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_seal_receipt" {
		return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage seal receipt_shape mismatch: got %q want %q", report.ReceiptShape, "weighted_resonance_shadow_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_seal_receipt")
	}
	for _, required := range []struct {
		name  string
		value bool
	}{
		{"admission_seal_immutable_receipt", report.AdmissionSealImmutableReceipt},
		{"weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_seal_ready", report.WeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageSealReady},
		{"weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_authority_consumed", report.WeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageAuthorityConsumed},
		{"weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_authority_required", report.WeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageAuthorityRequired},
		{"next_step_blocked_without_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_seal", report.NextStepBlockedWithoutResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageSeal},
		{"source_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_authority_ready", report.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageAuthorityReady},
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
			return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage seal %s not ready", required.name)
		}
	}
	for _, closed := range []struct {
		name  string
		value bool
	}{
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
			return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage seal opened %s", closed.name)
		}
	}
	if !report.AdmissionSealDryRunOnly || !report.AdmissionAuthorityDryRunOnly || !report.SourceAdmissionAuthorityDryRunOnly {
		return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage seal dry-run flag mismatch")
	}
	for _, requiredString := range []struct {
		name  string
		value string
	}{
		{"weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_seal_id", report.WeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageSealID},
		{"causal_id", report.CausalID},
		{"admission_seal_hash", report.AdmissionSealHash},
		{"admission_seal_read_back_hash", report.AdmissionSealReadBackHash},
		{"source_report", report.SourceReport},
		{"source_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_authority_id", report.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageAuthorityID},
		{"source_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_authority_causal_id", report.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageAuthorityCausalID},
		{"source_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_authority_hash", report.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageAuthorityHash},
		{"source_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_authority_read_back_hash", report.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageAuthorityReadBack},
		{"source_admission_authority_reason", report.SourceAdmissionAuthorityReason},
		{"source_admission_permit_schema", report.SourceAdmissionPermitSchema},
		{"source_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_permit_id", report.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStagePermitID},
		{"source_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_readiness_id", report.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageReadinessID},
	} {
		if strings.TrimSpace(requiredString.value) == "" {
			return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage seal %s missing", requiredString.name)
		}
	}
	if report.SourceSchema != admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageAuthoritySchema {
		return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage seal source_schema mismatch: got %q want %q", report.SourceSchema, admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageAuthoritySchema)
	}
	if report.SourceStatus != "shadow_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_authority_blocked_dry_run" {
		return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage seal source_status mismatch: got %q want %q", report.SourceStatus, "shadow_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_authority_blocked_dry_run")
	}
	if report.SourceTarget != "live_route_admission_next_step" {
		return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage seal source_target mismatch: got %q want %q", report.SourceTarget, "live_route_admission_next_step")
	}
	if report.SourceAdmissionPermitSchema != admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStagePermitSchema {
		return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage seal source_admission_permit_schema mismatch: got %q want %q", report.SourceAdmissionPermitSchema, admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStagePermitSchema)
	}
	if report.SourceAdmissionAuthorityReportReceiptShape != "weighted_resonance_shadow_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_authority_receipt" ||
		report.SourceAdmissionAuthorityState != "blocked" ||
		report.SourceAdmissionAuthorityAction != "reject_blocked_admission_permit" ||
		report.SourceAdmissionAuthorityTarget != "live_admission_authority" ||
		report.SourceAdmissionAuthorityTargetKind != "weighted_internal_world_shadow_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_permit" ||
		report.SourceAdmissionAuthorityTargetMode != "closed_authority_guard_dry_run" {
		return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage seal source admission authority shape mismatch")
	}
	if report.SourceAdmissionAuthorityReason != "weighted resonance shadow graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage authority blocked by blocked permit; live authority remains closed" {
		return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage seal source_admission_authority_reason mismatch: got %q", report.SourceAdmissionAuthorityReason)
	}
	if !strings.HasPrefix(report.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageAuthorityID, "weighted-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-candidate-store-reader-proof-precondition-decision-promotion-switch-enable-gate-live-stage-authority-id-") ||
		!strings.HasPrefix(report.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageAuthorityCausalID, "weighted-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-candidate-store-reader-proof-precondition-decision-promotion-switch-enable-gate-live-stage-authority-causal-") ||
		!strings.HasPrefix(report.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageAuthorityHash, "weighted-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-candidate-store-reader-proof-precondition-decision-promotion-switch-enable-gate-live-stage-authority-") ||
		!strings.HasPrefix(report.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageAuthorityReadBack, "weighted-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-candidate-store-reader-proof-precondition-decision-promotion-switch-enable-gate-live-stage-authority-read-") {
		return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage seal source authority mismatch")
	}
	if report.BodyTarget != "none" {
		return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage seal body_target mismatch: got %q want %q", report.BodyTarget, "none")
	}
	if report.CausalID == "" || report.CausalID != admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageSealCausalID(report) {
		return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage seal causal_id mismatch")
	}
	if report.AdmissionSealHash == "" || report.AdmissionSealHash != admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageSealHash(report) {
		return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage seal admission_seal_hash mismatch")
	}
	if report.AdmissionSealReadBackHash == "" || report.AdmissionSealReadBackHash != admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageSealReadBackHash(report) {
		return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage seal admission_seal_read_back_hash mismatch")
	}
	if report.AdmissionSealHash == report.AdmissionSealReadBackHash {
		return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage seal read-back proof collapsed")
	}
	if report.WeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageSealID == "" || report.WeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageSealID != admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageSealID(report) {
		return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage seal id mismatch")
	}
	if report.Reason != "weighted resonance shadow graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage seal fixed blocked authority provenance; live authority remains closed" {
		return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage seal reason mismatch: got %q", report.Reason)
	}
	return nil
}

func admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageSealCausalID(report admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageSealReport) string {
	h := hashJSON(struct {
		SourceAuthorityID   string `json:"source_admission_authority_id"`
		SourceAuthorityRead string `json:"source_admission_authority_read_back_hash"`
		SourcePermitID      string `json:"source_admission_permit_id"`
		Target              string `json:"target"`
		State               string `json:"admission_seal_state"`
		Action              string `json:"admission_seal_action"`
	}{
		SourceAuthorityID:   report.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageAuthorityID,
		SourceAuthorityRead: report.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageAuthorityReadBack,
		SourcePermitID:      report.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStagePermitID,
		Target:              report.Target,
		State:               report.AdmissionSealState,
		Action:              report.AdmissionSealAction,
	})
	if h == "" {
		return ""
	}
	return "weighted-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-candidate-store-reader-proof-precondition-decision-promotion-switch-enable-gate-live-stage-seal-causal-" + h
}

func admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageSealHash(report admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageSealReport) string {
	h := hashJSON(struct {
		CausalID               string `json:"causal_id"`
		SourceAuthorityID      string `json:"source_admission_authority_id"`
		SourceAuthorityHash    string `json:"source_admission_authority_hash"`
		SourceAuthorityRead    string `json:"source_admission_authority_read_back_hash"`
		State                  string `json:"admission_seal_state"`
		Action                 string `json:"admission_seal_action"`
		Target                 string `json:"admission_seal_target"`
		TargetKind             string `json:"admission_seal_target_kind"`
		TargetMode             string `json:"admission_seal_target_mode"`
		DryRunOnly             bool   `json:"admission_seal_dry_run_only"`
		AuthorityVerified      bool   `json:"admission_seal_authority_verified"`
		PermitVerified         bool   `json:"admission_seal_permit_verified"`
		LedgerVerified         bool   `json:"admission_seal_ledger_verified"`
		Ready                  bool   `json:"admission_seal_ready"`
		ImmutableReceipt       bool   `json:"admission_seal_immutable_receipt"`
		WeightedReady          bool   `json:"weighted_seal_ready"`
		SourceWeightedReady    bool   `json:"source_weighted_authority_ready"`
		SourceAuthorityReady   bool   `json:"source_admission_authority_ready"`
		SourceAuthorityGranted bool   `json:"source_admission_authority_granted"`
		WriteAllowed           bool   `json:"write_allowed"`
		AdmissionAllowed       bool   `json:"admission_allowed"`
		LiveAdmissionEnabled   bool   `json:"live_admission_enabled"`
		MutatesState           bool   `json:"mutates_state"`
		BodyMutationAllowed    bool   `json:"body_mutation_allowed"`
		NextStepBlockedWithout bool   `json:"next_step_blocked_without_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_seal"`
	}{
		CausalID:               report.CausalID,
		SourceAuthorityID:      report.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageAuthorityID,
		SourceAuthorityHash:    report.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageAuthorityHash,
		SourceAuthorityRead:    report.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageAuthorityReadBack,
		State:                  report.AdmissionSealState,
		Action:                 report.AdmissionSealAction,
		Target:                 report.AdmissionSealTarget,
		TargetKind:             report.AdmissionSealTargetKind,
		TargetMode:             report.AdmissionSealTargetMode,
		DryRunOnly:             report.AdmissionSealDryRunOnly,
		AuthorityVerified:      report.AdmissionSealAuthorityVerified,
		PermitVerified:         report.AdmissionSealPermitVerified,
		LedgerVerified:         report.AdmissionSealLedgerVerified,
		Ready:                  report.AdmissionSealReady,
		ImmutableReceipt:       report.AdmissionSealImmutableReceipt,
		WeightedReady:          report.WeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageSealReady,
		SourceWeightedReady:    report.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageAuthorityReady,
		SourceAuthorityReady:   report.SourceAdmissionAuthorityReady,
		SourceAuthorityGranted: report.SourceAdmissionAuthorityGranted,
		WriteAllowed:           report.WriteAllowed,
		AdmissionAllowed:       report.AdmissionAllowed,
		LiveAdmissionEnabled:   report.LiveAdmissionEnabled,
		MutatesState:           report.MutatesState,
		BodyMutationAllowed:    report.BodyMutationAllowed,
		NextStepBlockedWithout: report.NextStepBlockedWithoutResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageSeal,
	})
	if h == "" {
		return ""
	}
	return "weighted-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-candidate-store-reader-proof-precondition-decision-promotion-switch-enable-gate-live-stage-seal-" + h
}

func admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageSealReadBackHash(report admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageSealReport) string {
	h := hashJSON(struct {
		AdmissionSealHash   string `json:"admission_seal_hash"`
		SourceAuthorityID   string `json:"source_admission_authority_id"`
		SourceAuthorityRead string `json:"source_admission_authority_read_back_hash"`
		WeightedReady       bool   `json:"weighted_seal_ready"`
		AuthorityConsumed   bool   `json:"authority_consumed"`
		AuthorityRequired   bool   `json:"authority_required"`
		SealReady           bool   `json:"admission_seal_ready"`
		ImmutableReceipt    bool   `json:"admission_seal_immutable_receipt"`
		AuthorityVerified   bool   `json:"admission_seal_authority_verified"`
		WriteAllowed        bool   `json:"write_allowed"`
		AdmissionAllowed    bool   `json:"admission_allowed"`
		LiveEnabled         bool   `json:"live_admission_enabled"`
		MutatesState        bool   `json:"mutates_state"`
	}{
		AdmissionSealHash:   report.AdmissionSealHash,
		SourceAuthorityID:   report.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageAuthorityID,
		SourceAuthorityRead: report.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageAuthorityReadBack,
		WeightedReady:       report.WeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageSealReady,
		AuthorityConsumed:   report.WeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageAuthorityConsumed,
		AuthorityRequired:   report.WeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageAuthorityRequired,
		SealReady:           report.AdmissionSealReady,
		ImmutableReceipt:    report.AdmissionSealImmutableReceipt,
		AuthorityVerified:   report.AdmissionSealAuthorityVerified,
		WriteAllowed:        report.WriteAllowed,
		AdmissionAllowed:    report.AdmissionAllowed,
		LiveEnabled:         report.LiveAdmissionEnabled,
		MutatesState:        report.MutatesState,
	})
	if h == "" {
		return ""
	}
	return "weighted-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-candidate-store-reader-proof-precondition-decision-promotion-switch-enable-gate-live-stage-seal-read-" + h
}

func admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageSealID(report admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageSealReport) string {
	h := hashJSON(struct {
		Schema                 string `json:"schema"`
		Status                 string `json:"status"`
		Action                 string `json:"action"`
		SourceAuthorityID      string `json:"source_admission_authority_id"`
		SourceAuthorityHash    string `json:"source_admission_authority_hash"`
		SourceAuthorityRead    string `json:"source_admission_authority_read_back_hash"`
		CausalID               string `json:"causal_id"`
		AdmissionSealHash      string `json:"admission_seal_hash"`
		AdmissionSealRead      string `json:"admission_seal_read_back_hash"`
		State                  string `json:"admission_seal_state"`
		ActionSeal             string `json:"admission_seal_action"`
		Ready                  bool   `json:"weighted_seal_ready"`
		SealReady              bool   `json:"admission_seal_ready"`
		ImmutableReceipt       bool   `json:"admission_seal_immutable_receipt"`
		AuthorityVerified      bool   `json:"admission_seal_authority_verified"`
		WriteAllowed           bool   `json:"write_allowed"`
		AdmissionAllowed       bool   `json:"admission_allowed"`
		LiveAdmissionEnabled   bool   `json:"live_admission_enabled"`
		MutatesState           bool   `json:"mutates_state"`
		NextStepBlockedWithout bool   `json:"next_step_blocked_without_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_seal"`
	}{
		Schema:                 report.Schema,
		Status:                 report.Status,
		Action:                 report.Action,
		SourceAuthorityID:      report.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageAuthorityID,
		SourceAuthorityHash:    report.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageAuthorityHash,
		SourceAuthorityRead:    report.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageAuthorityReadBack,
		CausalID:               report.CausalID,
		AdmissionSealHash:      report.AdmissionSealHash,
		AdmissionSealRead:      report.AdmissionSealReadBackHash,
		State:                  report.AdmissionSealState,
		ActionSeal:             report.AdmissionSealAction,
		Ready:                  report.WeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageSealReady,
		SealReady:              report.AdmissionSealReady,
		ImmutableReceipt:       report.AdmissionSealImmutableReceipt,
		AuthorityVerified:      report.AdmissionSealAuthorityVerified,
		WriteAllowed:           report.WriteAllowed,
		AdmissionAllowed:       report.AdmissionAllowed,
		LiveAdmissionEnabled:   report.LiveAdmissionEnabled,
		MutatesState:           report.MutatesState,
		NextStepBlockedWithout: report.NextStepBlockedWithoutResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageSeal,
	})
	if h == "" {
		return ""
	}
	return "weighted-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-candidate-store-reader-proof-precondition-decision-promotion-switch-enable-gate-live-stage-seal-id-" + h
}

func readAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageSealReportForAssert(path string) (admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageSealReport, map[string]json.RawMessage, error) {
	var report admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageSealReport
	if strings.TrimSpace(path) == "" {
		return report, nil, fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage seal path missing")
	}
	raw, err := os.ReadFile(path)
	if err != nil {
		if os.IsNotExist(err) {
			return report, nil, fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage seal not written")
		}
		return report, nil, err
	}
	if len(raw) == 0 {
		return report, nil, fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage seal not written")
	}
	var root map[string]json.RawMessage
	if err := json.Unmarshal(raw, &root); err != nil {
		return report, nil, fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage seal JSON invalid: %w", err)
	}
	if err := json.Unmarshal(raw, &report); err != nil {
		return report, nil, fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage seal decode failed: %w", err)
	}
	return report, root, nil
}
