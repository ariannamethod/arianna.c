package main

import (
	"encoding/json"
	"fmt"
	"os"
	"strings"
)

const admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageLedgerVerificationSchema = "arianna.live_route_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_ledger_verification.v1"

type admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageLedgerVerificationReport struct {
	admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageLedgerPersistenceReport

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

	WeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageLedgerVerificationReady         bool   `json:"weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_ledger_verification_ready"`
	WeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageLedgerPersistenceConsumed       bool   `json:"weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_ledger_persistence_consumed"`
	WeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageLedgerPersistenceRequired       bool   `json:"weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_ledger_persistence_required"`
	NextStepBlockedWithoutResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageLedgerVerification         bool   `json:"next_step_blocked_without_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_ledger_verification"`
	WeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageLedgerVerificationID            string `json:"weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_ledger_verification_id"`
	LedgerVerificationHash                                                                                                                                                                          string `json:"ledger_verification_hash"`
	LedgerVerificationReadBackHash                                                                                                                                                                  string `json:"ledger_verification_read_back_hash"`
	SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageLedgerPersistenceID       string `json:"source_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_ledger_persistence_id"`
	SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageLedgerPersistenceReady    bool   `json:"source_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_ledger_persistence_ready"`
	SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageLedgerPersistenceCausalID string `json:"source_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_ledger_persistence_causal_id"`
	SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageLedgerPersistenceHash     string `json:"source_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_ledger_persistence_hash"`
	SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageLedgerPersistenceReadBack string `json:"source_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_ledger_persistence_read_back_hash"`
	SourceLedgerPersistenceReportReceiptShape                                                                                                                                                       string `json:"source_ledger_persistence_report_receipt_shape"`
	SourceLedgerPersistenceState                                                                                                                                                                    string `json:"source_ledger_persistence_state"`
	SourceLedgerPersistenceAction                                                                                                                                                                   string `json:"source_ledger_persistence_action"`
	SourceLedgerPersistenceTarget                                                                                                                                                                   string `json:"source_ledger_persistence_target"`
	SourceLedgerPersistenceTargetKind                                                                                                                                                               string `json:"source_ledger_persistence_target_kind"`
	SourceLedgerPersistenceTargetMode                                                                                                                                                               string `json:"source_ledger_persistence_target_mode"`
	SourceLedgerPersistenceReceiptShape                                                                                                                                                             string `json:"source_ledger_persistence_receipt_shape"`
	SourceLedgerPersistenceWriteScope                                                                                                                                                               string `json:"source_ledger_persistence_write_scope"`
	SourceLedgerPersistenceAppendOnly                                                                                                                                                               bool   `json:"source_ledger_persistence_append_only"`
	SourceLedgerPersistenceDryRunOnly                                                                                                                                                               bool   `json:"source_ledger_persistence_dry_run_only"`
	SourceLedgerPersistenceReceiptPersisted                                                                                                                                                         bool   `json:"source_ledger_persistence_receipt_persisted"`
	SourceLedgerPersistenceReady                                                                                                                                                                    bool   `json:"source_ledger_persistence_ready"`
	SourceLedgerPersistenceReason                                                                                                                                                                   string `json:"source_ledger_persistence_reason"`
	SourceLedgerImplementationSchema                                                                                                                                                                string `json:"source_ledger_implementation_schema"`
}

func runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageLedgerVerification(args []string) error {
	if len(args) != 2 {
		return fmt.Errorf("usage: --admission-live-route-weighted-admission-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-candidate-store-reader-proof-precondition-decision-promotion-switch-enable-gate-live-stage-ledger-verification RESONANCE_GRAFT_ADMISSION_FINAL_GATE_OBSERVATION_BOUNDARY_PREFLIGHT_GATE_CANDIDATE_STORE_READER_PROOF_PRECONDITION_DECISION_PROMOTION_SWITCH_ENABLE_GATE_LIVE_STAGE_LEDGER_PERSISTENCE_REPORT RESONANCE_GRAFT_ADMISSION_FINAL_GATE_OBSERVATION_BOUNDARY_PREFLIGHT_GATE_CANDIDATE_STORE_READER_PROOF_PRECONDITION_DECISION_PROMOTION_SWITCH_ENABLE_GATE_LIVE_STAGE_LEDGER_VERIFICATION_REPORT")
	}
	persistencePath := args[0]
	outputPath := args[1]
	if strings.TrimSpace(outputPath) == "" {
		return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage ledger verification output path missing")
	}
	sourcePersistence, root, err := readAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageLedgerPersistenceReportForAssert(persistencePath)
	if err != nil {
		return err
	}
	if err := admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageLedgerPersistenceReportError(sourcePersistence, root); err != nil {
		return err
	}
	verification := admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageLedgerVerificationReport{
		admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageLedgerPersistenceReport: sourcePersistence,
		LedgerVerificationState:           "blocked",
		LedgerVerificationAction:          "reject_blocked_ledger_persistence",
		LedgerVerificationTarget:          "admission_ledger_receipt",
		LedgerVerificationTargetKind:      "weighted_internal_world_shadow_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_ledger_persistence",
		LedgerVerificationTargetMode:      "closed_read_back_guard_dry_run",
		LedgerVerificationReceiptShape:    "none",
		LedgerVerificationAppendOnly:      false,
		LedgerVerificationDryRunOnly:      true,
		LedgerVerificationReceiptReadBack: false,
		LedgerVerificationReceiptVerified: false,
		LedgerVerificationReady:           false,
		WeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageLedgerVerificationReady:         true,
		WeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageLedgerPersistenceConsumed:       true,
		WeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageLedgerPersistenceRequired:       true,
		NextStepBlockedWithoutResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageLedgerVerification:         true,
		SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageLedgerPersistenceID:       sourcePersistence.WeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageLedgerPersistenceID,
		SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageLedgerPersistenceReady:    sourcePersistence.WeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageLedgerPersistenceReady,
		SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageLedgerPersistenceCausalID: sourcePersistence.CausalID,
		SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageLedgerPersistenceHash:     sourcePersistence.LedgerPersistenceHash,
		SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageLedgerPersistenceReadBack: sourcePersistence.LedgerPersistenceReadBackHash,
		SourceLedgerPersistenceReportReceiptShape: sourcePersistence.ReceiptShape,
		SourceLedgerPersistenceState:              sourcePersistence.LedgerPersistenceState,
		SourceLedgerPersistenceAction:             sourcePersistence.LedgerPersistenceAction,
		SourceLedgerPersistenceTarget:             sourcePersistence.LedgerPersistenceTarget,
		SourceLedgerPersistenceTargetKind:         sourcePersistence.LedgerPersistenceTargetKind,
		SourceLedgerPersistenceTargetMode:         sourcePersistence.LedgerPersistenceTargetMode,
		SourceLedgerPersistenceReceiptShape:       sourcePersistence.LedgerPersistenceReceiptShape,
		SourceLedgerPersistenceWriteScope:         sourcePersistence.LedgerPersistenceWriteScope,
		SourceLedgerPersistenceAppendOnly:         sourcePersistence.LedgerPersistenceAppendOnly,
		SourceLedgerPersistenceDryRunOnly:         sourcePersistence.LedgerPersistenceDryRunOnly,
		SourceLedgerPersistenceReceiptPersisted:   sourcePersistence.LedgerPersistenceReceiptPersisted,
		SourceLedgerPersistenceReady:              sourcePersistence.LedgerPersistenceReady,
		SourceLedgerPersistenceReason:             sourcePersistence.Reason,
		SourceLedgerImplementationSchema:          sourcePersistence.SourceSchema,
	}
	verification.Schema = admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageLedgerVerificationSchema
	verification.Status = "shadow_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_ledger_verification_blocked_dry_run"
	verification.TargetKind = "weighted_internal_world_shadow_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_ledger_verification"
	verification.TargetMode = "closed_ledger_verification_guard_dry_run"
	verification.Action = "block_weighted_resonance_shadow_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_ledger_persistence_blocked_dry_run"
	verification.WriterAction = "reject_blocked_ledger_persistence"
	verification.RollbackAction = "reject_blocked_ledger_persistence"
	verification.LedgerState = "blocked"
	verification.LedgerAction = "reject_blocked_ledger_persistence"
	verification.LedgerContract = "none"
	verification.LedgerEntrypoint = "none"
	verification.LedgerReceiptShape = "none"
	verification.LedgerWriteScope = "none"
	verification.LedgerReady = false
	verification.LedgerAppendAllowed = false
	verification.ReceiptShape = "weighted_resonance_shadow_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_ledger_verification_receipt"
	verification.SourceSchema = sourcePersistence.Schema
	verification.SourceStatus = sourcePersistence.Status
	verification.SourceTarget = sourcePersistence.Target
	verification.SourceReport = persistencePath
	verification.Reason = "weighted resonance shadow graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage ledger verification blocked by blocked ledger persistence; receipt read-back remains closed"
	verification.CausalID = admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageLedgerVerificationCausalID(verification)
	verification.LedgerVerificationHash = admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageLedgerVerificationHash(verification)
	verification.LedgerVerificationReadBackHash = admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageLedgerVerificationReadBackHash(verification)
	verification.WeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageLedgerVerificationID = admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageLedgerVerificationID(verification)
	if verification.CausalID == "" ||
		verification.LedgerVerificationHash == "" ||
		verification.LedgerVerificationReadBackHash == "" ||
		verification.WeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageLedgerVerificationID == "" ||
		verification.LedgerVerificationHash == verification.LedgerVerificationReadBackHash {
		return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage ledger verification read-back proof failed")
	}
	raw, err := json.MarshalIndent(verification, "", "  ")
	if err != nil {
		return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage ledger verification marshal failed: %w", err)
	}
	raw = append(raw, '\n')
	if err := os.WriteFile(outputPath, raw, 0600); err != nil {
		return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage ledger verification write failed: %w", err)
	}
	fmt.Printf("[admission-live-route-weighted-admission-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-candidate-store-reader-proof-precondition-decision-promotion-switch-enable-gate-live-stage-ledger-verification] pass: resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_ledger_verification_report=%s resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_ledger_persistence_report=%s\n", outputPath, persistencePath)
	return nil
}

func runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageLedgerVerificationAssert(args []string) error {
	if len(args) != 1 {
		return fmt.Errorf("usage: --admission-live-route-weighted-admission-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-candidate-store-reader-proof-precondition-decision-promotion-switch-enable-gate-live-stage-ledger-verification-assert REPORT")
	}
	report, root, err := readAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageLedgerVerificationReportForAssert(args[0])
	if err != nil {
		return err
	}
	return admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageLedgerVerificationReportError(report, root)
}

func admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageLedgerVerificationReportError(report admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageLedgerVerificationReport, root map[string]json.RawMessage) error {
	if _, ok := root["schema"]; !ok {
		return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage ledger verification schema missing")
	}
	if report.Schema != admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageLedgerVerificationSchema {
		return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage ledger verification schema mismatch: got %q want %q", report.Schema, admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageLedgerVerificationSchema)
	}
	if report.Status != "shadow_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_ledger_verification_blocked_dry_run" {
		return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage ledger verification status mismatch: got %q want %q", report.Status, "shadow_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_ledger_verification_blocked_dry_run")
	}
	if report.TargetKind != "weighted_internal_world_shadow_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_ledger_verification" {
		return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage ledger verification target_kind mismatch: got %q want %q", report.TargetKind, "weighted_internal_world_shadow_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_ledger_verification")
	}
	if report.TargetMode != "closed_ledger_verification_guard_dry_run" {
		return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage ledger verification target_mode mismatch: got %q want %q", report.TargetMode, "closed_ledger_verification_guard_dry_run")
	}
	if report.Action != "block_weighted_resonance_shadow_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_ledger_persistence_blocked_dry_run" {
		return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage ledger verification action mismatch: got %q want %q", report.Action, "block_weighted_resonance_shadow_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_ledger_persistence_blocked_dry_run")
	}
	if report.WriterAction != "reject_blocked_ledger_persistence" || report.RollbackAction != "reject_blocked_ledger_persistence" {
		return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage ledger verification writer/rollback action mismatch")
	}
	if report.LedgerState != "blocked" ||
		report.LedgerAction != "reject_blocked_ledger_persistence" ||
		report.LedgerContract != "none" ||
		report.LedgerEntrypoint != "none" ||
		report.LedgerReceiptShape != "none" ||
		report.LedgerWriteScope != "none" {
		return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage ledger verification ledger guard mismatch")
	}
	if report.LedgerVerificationState != "blocked" ||
		report.LedgerVerificationAction != "reject_blocked_ledger_persistence" ||
		report.LedgerVerificationTarget != "admission_ledger_receipt" ||
		report.LedgerVerificationTargetKind != "weighted_internal_world_shadow_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_ledger_persistence" ||
		report.LedgerVerificationTargetMode != "closed_read_back_guard_dry_run" ||
		report.LedgerVerificationReceiptShape != "none" {
		return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage ledger verification shape mismatch")
	}
	if report.ReceiptShape != "weighted_resonance_shadow_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_ledger_verification_receipt" {
		return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage ledger verification receipt_shape mismatch: got %q want %q", report.ReceiptShape, "weighted_resonance_shadow_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_ledger_verification_receipt")
	}
	for _, required := range []struct {
		name  string
		value bool
	}{
		{"weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_ledger_verification_ready", report.WeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageLedgerVerificationReady},
		{"weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_ledger_persistence_consumed", report.WeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageLedgerPersistenceConsumed},
		{"weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_ledger_persistence_required", report.WeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageLedgerPersistenceRequired},
		{"next_step_blocked_without_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_ledger_verification", report.NextStepBlockedWithoutResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageLedgerVerification},
		{"source_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_ledger_persistence_ready", report.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageLedgerPersistenceReady},
		{"weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_ledger_persistence_ready", report.WeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageLedgerPersistenceReady},
		{"weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_ledger_implementation_consumed", report.WeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageLedgerImplementationConsumed},
		{"weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_ledger_implementation_required", report.WeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageLedgerImplementationRequired},
		{"weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_ledger_implementation_ready", report.WeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageLedgerImplementationReady},
		{"weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_ledger_consumed", report.WeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageLedgerConsumed},
		{"weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_ledger_required", report.WeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageLedgerRequired},
		{"source_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_ledger_ready", report.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageLedgerReady},
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
			return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage ledger verification %s not ready", required.name)
		}
	}
	for _, closed := range []struct {
		name  string
		value bool
	}{
		{"ledger_ready", report.LedgerReady},
		{"ledger_append_allowed", report.LedgerAppendAllowed},
		{"ledger_implementation_append_only", report.LedgerImplementationAppendOnly},
		{"ledger_implementation_receipt_persisted", report.LedgerImplementationReceiptPersisted},
		{"ledger_implementation_ready", report.LedgerImplementationReady},
		{"ledger_persistence_append_only", report.LedgerPersistenceAppendOnly},
		{"ledger_persistence_receipt_persisted", report.LedgerPersistenceReceiptPersisted},
		{"ledger_persistence_ready", report.LedgerPersistenceReady},
		{"ledger_verification_append_only", report.LedgerVerificationAppendOnly},
		{"ledger_verification_receipt_read_back", report.LedgerVerificationReceiptReadBack},
		{"ledger_verification_receipt_verified", report.LedgerVerificationReceiptVerified},
		{"ledger_verification_ready", report.LedgerVerificationReady},
		{"source_ledger_persistence_append_only", report.SourceLedgerPersistenceAppendOnly},
		{"source_ledger_persistence_receipt_persisted", report.SourceLedgerPersistenceReceiptPersisted},
		{"source_ledger_persistence_ready", report.SourceLedgerPersistenceReady},
		{"source_ledger_implementation_append_only", report.SourceLedgerImplementationAppendOnly},
		{"source_ledger_implementation_receipt_persisted", report.SourceLedgerImplementationReceiptPersisted},
		{"source_ledger_implementation_ready", report.SourceLedgerImplementationReady},
		{"writer_ready", report.WriterReady},
		{"rollback_ready", report.RollbackReady},
		{"writer_contract_present", report.WriterContractPresent},
		{"rollback_contract_present", report.RollbackContractPresent},
		{"ledger_contract_present", report.LedgerContractPresent},
		{"source_admission_ledger_ledger_ready", report.SourceAdmissionLedgerLedgerReady},
		{"source_admission_ledger_ledger_append_allowed", report.SourceAdmissionLedgerLedgerAppendAllowed},
		{"source_writer_contract_contracts_ready", report.SourceWriterContractContractsReady},
		{"contracts_ready", report.ContractsReady},
		{"write_allowed", report.WriteAllowed},
		{"admission_allowed", report.AdmissionAllowed},
		{"live_admission_enabled", report.LiveAdmissionEnabled},
		{"mutates_state", report.MutatesState},
		{"body_mutation_allowed", report.BodyMutationAllowed},
		{"authority_granted", report.AuthorityGranted},
	} {
		if closed.value {
			return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage ledger verification opened %s", closed.name)
		}
	}
	if !report.LedgerVerificationDryRunOnly {
		return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage ledger verification ledger_verification_dry_run_only not ready")
	}
	if !report.SourceLedgerPersistenceDryRunOnly {
		return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage ledger verification source_ledger_persistence_dry_run_only not ready")
	}
	for _, requiredString := range []struct {
		name  string
		value string
	}{
		{"weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_ledger_verification_id", report.WeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageLedgerVerificationID},
		{"causal_id", report.CausalID},
		{"ledger_verification_hash", report.LedgerVerificationHash},
		{"ledger_verification_read_back_hash", report.LedgerVerificationReadBackHash},
		{"source_report", report.SourceReport},
		{"source_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_ledger_persistence_id", report.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageLedgerPersistenceID},
		{"source_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_ledger_persistence_causal_id", report.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageLedgerPersistenceCausalID},
		{"source_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_ledger_persistence_hash", report.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageLedgerPersistenceHash},
		{"source_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_ledger_persistence_read_back_hash", report.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageLedgerPersistenceReadBack},
		{"source_ledger_persistence_reason", report.SourceLedgerPersistenceReason},
		{"source_ledger_implementation_schema", report.SourceLedgerImplementationSchema},
		{"source_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_ledger_implementation_id", report.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageLedgerImplementationID},
		{"source_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_ledger_id", report.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageLedgerID},
	} {
		if strings.TrimSpace(requiredString.value) == "" {
			return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage ledger verification %s missing", requiredString.name)
		}
	}
	if report.SourceSchema != admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageLedgerPersistenceSchema {
		return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage ledger verification source_schema mismatch: got %q want %q", report.SourceSchema, admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageLedgerPersistenceSchema)
	}
	if report.SourceStatus != "shadow_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_ledger_persistence_blocked_dry_run" {
		return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage ledger verification source_status mismatch: got %q want %q", report.SourceStatus, "shadow_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_ledger_persistence_blocked_dry_run")
	}
	if report.SourceLedgerImplementationSchema != admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageLedgerImplementationSchema {
		return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage ledger verification source_ledger_implementation_schema mismatch: got %q want %q", report.SourceLedgerImplementationSchema, admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageLedgerImplementationSchema)
	}
	if report.SourceLedgerPersistenceReportReceiptShape != "weighted_resonance_shadow_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_ledger_persistence_receipt" ||
		report.SourceLedgerPersistenceState != "blocked" ||
		report.SourceLedgerPersistenceAction != "reject_blocked_ledger_implementation" ||
		report.SourceLedgerPersistenceTarget != "admission_ledger_receipt" ||
		report.SourceLedgerPersistenceTargetKind != "weighted_internal_world_shadow_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_ledger_implementation" ||
		report.SourceLedgerPersistenceTargetMode != "closed_persistence_guard_dry_run" ||
		report.SourceLedgerPersistenceReceiptShape != "none" ||
		report.SourceLedgerPersistenceWriteScope != "none" {
		return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage ledger verification source ledger persistence shape mismatch")
	}
	if report.SourceLedgerPersistenceReason != "weighted resonance shadow graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage ledger persistence blocked by blocked ledger implementation; ledger receipt persistence remains closed" {
		return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage ledger verification source_ledger_persistence_reason mismatch: got %q", report.SourceLedgerPersistenceReason)
	}
	if !strings.HasPrefix(report.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageLedgerPersistenceID, "weighted-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-candidate-store-reader-proof-precondition-decision-promotion-switch-enable-gate-live-stage-ledger-persistence-id-") ||
		!strings.HasPrefix(report.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageLedgerPersistenceCausalID, "weighted-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-candidate-store-reader-proof-precondition-decision-promotion-switch-enable-gate-live-stage-ledger-persistence-causal-") ||
		!strings.HasPrefix(report.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageLedgerPersistenceHash, "weighted-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-candidate-store-reader-proof-precondition-decision-promotion-switch-enable-gate-live-stage-ledger-persistence-") ||
		!strings.HasPrefix(report.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageLedgerPersistenceReadBack, "weighted-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-candidate-store-reader-proof-precondition-decision-promotion-switch-enable-gate-live-stage-ledger-persistence-read-") {
		return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage ledger verification source ledger persistence mismatch")
	}
	if report.BodyTarget != "none" {
		return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage ledger verification body_target mismatch: got %q want %q", report.BodyTarget, "none")
	}
	if report.CausalID == "" || report.CausalID != admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageLedgerVerificationCausalID(report) {
		return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage ledger verification causal_id mismatch")
	}
	if report.LedgerVerificationHash == "" || report.LedgerVerificationHash != admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageLedgerVerificationHash(report) {
		return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage ledger verification ledger_verification_hash mismatch")
	}
	if report.LedgerVerificationReadBackHash == "" || report.LedgerVerificationReadBackHash != admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageLedgerVerificationReadBackHash(report) {
		return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage ledger verification ledger_verification_read_back_hash mismatch")
	}
	if report.LedgerVerificationHash == report.LedgerVerificationReadBackHash {
		return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage ledger verification read-back proof collapsed")
	}
	if report.WeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageLedgerVerificationID == "" || report.WeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageLedgerVerificationID != admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageLedgerVerificationID(report) {
		return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage ledger verification id mismatch")
	}
	if report.Reason != "weighted resonance shadow graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage ledger verification blocked by blocked ledger persistence; receipt read-back remains closed" {
		return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage ledger verification reason mismatch: got %q", report.Reason)
	}
	return nil
}

func admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageLedgerVerificationCausalID(report admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageLedgerVerificationReport) string {
	h := hashJSON(struct {
		SourcePersistenceID   string `json:"source_ledger_persistence_id"`
		SourcePersistenceRead string `json:"source_ledger_persistence_read_back_hash"`
		SourceImplementation  string `json:"source_ledger_implementation_id"`
		Target                string `json:"target"`
		State                 string `json:"ledger_verification_state"`
		Action                string `json:"ledger_verification_action"`
	}{
		SourcePersistenceID:   report.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageLedgerPersistenceID,
		SourcePersistenceRead: report.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageLedgerPersistenceReadBack,
		SourceImplementation:  report.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageLedgerImplementationID,
		Target:                report.Target,
		State:                 report.LedgerVerificationState,
		Action:                report.LedgerVerificationAction,
	})
	if h == "" {
		return ""
	}
	return "weighted-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-candidate-store-reader-proof-precondition-decision-promotion-switch-enable-gate-live-stage-ledger-verification-causal-" + h
}

func admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageLedgerVerificationHash(report admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageLedgerVerificationReport) string {
	h := hashJSON(struct {
		CausalID               string `json:"causal_id"`
		SourcePersistenceID    string `json:"source_ledger_persistence_id"`
		SourcePersistenceHash  string `json:"source_ledger_persistence_hash"`
		SourcePersistenceRead  string `json:"source_ledger_persistence_read_back_hash"`
		SourceImplementationID string `json:"source_ledger_implementation_id"`
		State                  string `json:"ledger_verification_state"`
		Action                 string `json:"ledger_verification_action"`
		Target                 string `json:"ledger_verification_target"`
		TargetKind             string `json:"ledger_verification_target_kind"`
		TargetMode             string `json:"ledger_verification_target_mode"`
		ReceiptShape           string `json:"ledger_verification_receipt_shape"`
		AppendOnly             bool   `json:"ledger_verification_append_only"`
		DryRunOnly             bool   `json:"ledger_verification_dry_run_only"`
		ReceiptReadBack        bool   `json:"ledger_verification_receipt_read_back"`
		ReceiptVerified        bool   `json:"ledger_verification_receipt_verified"`
		Ready                  bool   `json:"ledger_verification_ready"`
		WeightedReady          bool   `json:"weighted_ledger_verification_ready"`
		SourcePersistenceReady bool   `json:"source_ledger_persistence_ready"`
		LedgerAppendAllowed    bool   `json:"ledger_append_allowed"`
		WriteAllowed           bool   `json:"write_allowed"`
		AdmissionAllowed       bool   `json:"admission_allowed"`
		LiveAdmissionEnabled   bool   `json:"live_admission_enabled"`
		MutatesState           bool   `json:"mutates_state"`
		BodyMutationAllowed    bool   `json:"body_mutation_allowed"`
		ContractsReady         bool   `json:"contracts_ready"`
		NextStepBlockedWithout bool   `json:"next_step_blocked_without_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_ledger_verification"`
	}{
		CausalID:               report.CausalID,
		SourcePersistenceID:    report.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageLedgerPersistenceID,
		SourcePersistenceHash:  report.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageLedgerPersistenceHash,
		SourcePersistenceRead:  report.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageLedgerPersistenceReadBack,
		SourceImplementationID: report.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageLedgerImplementationID,
		State:                  report.LedgerVerificationState,
		Action:                 report.LedgerVerificationAction,
		Target:                 report.LedgerVerificationTarget,
		TargetKind:             report.LedgerVerificationTargetKind,
		TargetMode:             report.LedgerVerificationTargetMode,
		ReceiptShape:           report.LedgerVerificationReceiptShape,
		AppendOnly:             report.LedgerVerificationAppendOnly,
		DryRunOnly:             report.LedgerVerificationDryRunOnly,
		ReceiptReadBack:        report.LedgerVerificationReceiptReadBack,
		ReceiptVerified:        report.LedgerVerificationReceiptVerified,
		Ready:                  report.LedgerVerificationReady,
		WeightedReady:          report.WeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageLedgerVerificationReady,
		SourcePersistenceReady: report.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageLedgerPersistenceReady,
		LedgerAppendAllowed:    report.LedgerAppendAllowed,
		WriteAllowed:           report.WriteAllowed,
		AdmissionAllowed:       report.AdmissionAllowed,
		LiveAdmissionEnabled:   report.LiveAdmissionEnabled,
		MutatesState:           report.MutatesState,
		BodyMutationAllowed:    report.BodyMutationAllowed,
		ContractsReady:         report.ContractsReady,
		NextStepBlockedWithout: report.NextStepBlockedWithoutResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageLedgerVerification,
	})
	if h == "" {
		return ""
	}
	return "weighted-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-candidate-store-reader-proof-precondition-decision-promotion-switch-enable-gate-live-stage-ledger-verification-" + h
}

func admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageLedgerVerificationReadBackHash(report admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageLedgerVerificationReport) string {
	h := hashJSON(struct {
		LedgerVerificationHash string `json:"ledger_verification_hash"`
		SourcePersistenceID    string `json:"source_ledger_persistence_id"`
		SourcePersistenceRead  string `json:"source_ledger_persistence_read_back_hash"`
		WeightedReady          bool   `json:"weighted_ledger_verification_ready"`
		PersistenceConsumed    bool   `json:"ledger_persistence_consumed"`
		PersistenceRequired    bool   `json:"ledger_persistence_required"`
		VerificationReady      bool   `json:"ledger_verification_ready"`
		ReceiptVerified        bool   `json:"ledger_verification_receipt_verified"`
		ContractsReady         bool   `json:"contracts_ready"`
		WriteAllowed           bool   `json:"write_allowed"`
		AdmissionAllowed       bool   `json:"admission_allowed"`
		LiveAdmissionEnabled   bool   `json:"live_admission_enabled"`
		MutatesState           bool   `json:"mutates_state"`
	}{
		LedgerVerificationHash: report.LedgerVerificationHash,
		SourcePersistenceID:    report.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageLedgerPersistenceID,
		SourcePersistenceRead:  report.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageLedgerPersistenceReadBack,
		WeightedReady:          report.WeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageLedgerVerificationReady,
		PersistenceConsumed:    report.WeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageLedgerPersistenceConsumed,
		PersistenceRequired:    report.WeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageLedgerPersistenceRequired,
		VerificationReady:      report.LedgerVerificationReady,
		ReceiptVerified:        report.LedgerVerificationReceiptVerified,
		ContractsReady:         report.ContractsReady,
		WriteAllowed:           report.WriteAllowed,
		AdmissionAllowed:       report.AdmissionAllowed,
		LiveAdmissionEnabled:   report.LiveAdmissionEnabled,
		MutatesState:           report.MutatesState,
	})
	if h == "" {
		return ""
	}
	return "weighted-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-candidate-store-reader-proof-precondition-decision-promotion-switch-enable-gate-live-stage-ledger-verification-read-" + h
}

func admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageLedgerVerificationID(report admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageLedgerVerificationReport) string {
	h := hashJSON(struct {
		Schema                     string `json:"schema"`
		Status                     string `json:"status"`
		Action                     string `json:"action"`
		SourcePersistenceID        string `json:"source_ledger_persistence_id"`
		SourcePersistenceHash      string `json:"source_ledger_persistence_hash"`
		SourcePersistenceRead      string `json:"source_ledger_persistence_read_back_hash"`
		SourceImplementationID     string `json:"source_ledger_implementation_id"`
		CausalID                   string `json:"causal_id"`
		LedgerVerificationHash     string `json:"ledger_verification_hash"`
		LedgerVerificationReadBack string `json:"ledger_verification_read_back_hash"`
		State                      string `json:"ledger_verification_state"`
		ActionVerification         string `json:"ledger_verification_action"`
		Ready                      bool   `json:"weighted_ledger_verification_ready"`
		ReceiptVerified            bool   `json:"ledger_verification_receipt_verified"`
		WriteAllowed               bool   `json:"write_allowed"`
		AdmissionAllowed           bool   `json:"admission_allowed"`
		LiveAdmissionEnabled       bool   `json:"live_admission_enabled"`
		MutatesState               bool   `json:"mutates_state"`
		NextStepBlockedWithout     bool   `json:"next_step_blocked_without_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_ledger_verification"`
	}{
		Schema:                     report.Schema,
		Status:                     report.Status,
		Action:                     report.Action,
		SourcePersistenceID:        report.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageLedgerPersistenceID,
		SourcePersistenceHash:      report.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageLedgerPersistenceHash,
		SourcePersistenceRead:      report.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageLedgerPersistenceReadBack,
		SourceImplementationID:     report.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageLedgerImplementationID,
		CausalID:                   report.CausalID,
		LedgerVerificationHash:     report.LedgerVerificationHash,
		LedgerVerificationReadBack: report.LedgerVerificationReadBackHash,
		State:                      report.LedgerVerificationState,
		ActionVerification:         report.LedgerVerificationAction,
		Ready:                      report.WeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageLedgerVerificationReady,
		ReceiptVerified:            report.LedgerVerificationReceiptVerified,
		WriteAllowed:               report.WriteAllowed,
		AdmissionAllowed:           report.AdmissionAllowed,
		LiveAdmissionEnabled:       report.LiveAdmissionEnabled,
		MutatesState:               report.MutatesState,
		NextStepBlockedWithout:     report.NextStepBlockedWithoutResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageLedgerVerification,
	})
	if h == "" {
		return ""
	}
	return "weighted-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-candidate-store-reader-proof-precondition-decision-promotion-switch-enable-gate-live-stage-ledger-verification-id-" + h
}

func readAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageLedgerVerificationReportForAssert(path string) (admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageLedgerVerificationReport, map[string]json.RawMessage, error) {
	var report admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageLedgerVerificationReport
	if strings.TrimSpace(path) == "" {
		return report, nil, fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage ledger verification path missing")
	}
	raw, err := os.ReadFile(path)
	if err != nil {
		if os.IsNotExist(err) {
			return report, nil, fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage ledger verification not written")
		}
		return report, nil, err
	}
	if len(raw) == 0 {
		return report, nil, fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage ledger verification not written")
	}
	var root map[string]json.RawMessage
	if err := json.Unmarshal(raw, &root); err != nil {
		return report, nil, fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage ledger verification JSON invalid: %w", err)
	}
	if err := json.Unmarshal(raw, &report); err != nil {
		return report, nil, fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage ledger verification decode failed: %w", err)
	}
	return report, root, nil
}
