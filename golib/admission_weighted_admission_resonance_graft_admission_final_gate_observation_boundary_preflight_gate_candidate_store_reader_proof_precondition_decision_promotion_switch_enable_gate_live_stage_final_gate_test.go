package main

import (
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGate(t *testing.T) {
	dir := t.TempDir()

	usage := "usage: --admission-live-route-weighted-admission-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-candidate-store-reader-proof-precondition-decision-promotion-switch-enable-gate-live-stage-final-gate RESONANCE_GRAFT_ADMISSION_FINAL_GATE_OBSERVATION_BOUNDARY_PREFLIGHT_GATE_CANDIDATE_STORE_READER_PROOF_PRECONDITION_DECISION_PROMOTION_SWITCH_ENABLE_GATE_LIVE_STAGE_SEAL_REPORT RESONANCE_GRAFT_ADMISSION_FINAL_GATE_OBSERVATION_BOUNDARY_PREFLIGHT_GATE_CANDIDATE_STORE_READER_PROOF_PRECONDITION_DECISION_PROMOTION_SWITCH_ENABLE_GATE_LIVE_STAGE_FINAL_GATE_REPORT"
	requireBoundaryAssertError(t, runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGate(nil), usage)
	requireBoundaryAssertError(t, runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGate([]string{"seal.json"}), usage)
	requireBoundaryAssertError(t, runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGate([]string{"seal.json", "final_gate.json", "extra"}), usage)
	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGate([]string{"  ", filepath.Join(dir, "final_gate.json")}),
		"weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage seal path missing",
	)

	sealPath := filepath.Join(dir, "seal.json")
	writeWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageSealFixture(t, sealPath)
	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGate([]string{sealPath, "  "}),
		"weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage final gate output path missing",
	)

	finalGatePath := filepath.Join(dir, "final_gate.json")
	if err := runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGate([]string{sealPath, finalGatePath}); err != nil {
		t.Fatalf("valid weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage final gate rejected: %v", err)
	}
	raw, err := os.ReadFile(finalGatePath)
	if err != nil {
		t.Fatalf("read weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage final gate: %v", err)
	}
	var finalGate admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReport
	if err := json.Unmarshal(raw, &finalGate); err != nil {
		t.Fatalf("decode weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage final gate: %v", err)
	}
	sourceRaw, err := os.ReadFile(sealPath)
	if err != nil {
		t.Fatalf("read weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage seal: %v", err)
	}
	var sourceSeal admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageSealReport
	if err := json.Unmarshal(sourceRaw, &sourceSeal); err != nil {
		t.Fatalf("decode weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage seal: %v", err)
	}
	if finalGate.Schema != admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateSchema ||
		finalGate.Status != "shadow_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_final_gate_blocked_dry_run" ||
		finalGate.Target != "live_route_admission_next_step" ||
		finalGate.TargetKind != "weighted_internal_world_shadow_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_final_gate" ||
		finalGate.TargetMode != "closed_final_gate_guard_dry_run" ||
		finalGate.Action != "block_weighted_resonance_shadow_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_seal_blocked_dry_run" ||
		finalGate.WriterAction != "reject_blocked_admission_seal" ||
		finalGate.RollbackAction != "reject_blocked_admission_seal" ||
		finalGate.LedgerState != "blocked" ||
		finalGate.LedgerAction != "reject_blocked_admission_seal" ||
		finalGate.LedgerContract != "none" ||
		finalGate.LedgerEntrypoint != "none" ||
		finalGate.LedgerReceiptShape != "none" ||
		finalGate.LedgerWriteScope != "none" ||
		finalGate.ReceiptShape != "weighted_resonance_shadow_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_final_gate_receipt" ||
		finalGate.AdmissionFinalGateState != "blocked" ||
		finalGate.AdmissionFinalGateAction != "reject_blocked_admission_seal" ||
		finalGate.AdmissionFinalGateTarget != "live_admission_final_gate" ||
		finalGate.AdmissionFinalGateTargetKind != "weighted_internal_world_shadow_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_seal" ||
		finalGate.AdmissionFinalGateTargetMode != "closed_final_gate_guard_dry_run" ||
		!finalGate.AdmissionFinalGateDryRunOnly ||
		finalGate.AdmissionFinalGateSealVerified ||
		finalGate.AdmissionFinalGateAuthorityVerified ||
		finalGate.AdmissionFinalGatePermitVerified ||
		finalGate.AdmissionFinalGateLedgerVerified ||
		finalGate.AdmissionFinalGateReady ||
		!finalGate.WeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReady ||
		!finalGate.WeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageSealConsumed ||
		!finalGate.WeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageSealRequired ||
		!finalGate.NextStepBlockedWithoutResonanceGraftAdmissionFinalGate ||
		finalGate.SourceSchema != admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageSealSchema ||
		finalGate.SourceStatus != "shadow_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_seal_blocked_dry_run" ||
		finalGate.SourceTarget != "live_route_admission_next_step" ||
		finalGate.SourceReport != sealPath ||
		finalGate.SourceAdmissionAuthoritySchema != admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageAuthoritySchema ||
		finalGate.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageSealID != sourceSeal.WeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageSealID ||
		finalGate.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageSealHash != sourceSeal.AdmissionSealHash ||
		finalGate.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageSealReadBack != sourceSeal.AdmissionSealReadBackHash ||
		finalGate.SourceAdmissionSealReportReceiptShape != sourceSeal.ReceiptShape ||
		finalGate.SourceAdmissionSealAction != sourceSeal.AdmissionSealAction ||
		!finalGate.SourceAdmissionSealDryRunOnly ||
		finalGate.SourceAdmissionSealAuthorityVerified ||
		finalGate.SourceAdmissionSealPermitVerified ||
		finalGate.SourceAdmissionSealLedgerVerified ||
		finalGate.SourceAdmissionSealReady ||
		!finalGate.SourceAdmissionSealImmutableReceipt ||
		finalGate.CausalID != admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateCausalID(finalGate) ||
		finalGate.AdmissionFinalGateHash != admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateHash(finalGate) ||
		finalGate.AdmissionFinalGateReadBackHash != admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReadBackHash(finalGate) ||
		finalGate.AdmissionFinalGateHash == finalGate.AdmissionFinalGateReadBackHash ||
		finalGate.WeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateID != admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateID(finalGate) ||
		finalGate.LedgerReady ||
		finalGate.LedgerAppendAllowed ||
		finalGate.WriteAllowed ||
		finalGate.AdmissionAllowed ||
		finalGate.LiveAdmissionEnabled ||
		finalGate.MutatesState ||
		finalGate.BodyMutationAllowed ||
		finalGate.AuthorityGranted ||
		finalGate.BodyTarget != "none" ||
		!finalGate.Passed ||
		finalGate.Reason != "weighted resonance shadow graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage final gate blocked by blocked seal; final admission remains closed" {
		t.Fatalf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage final gate lost contract: %+v", finalGate)
	}

	notReadyPath := filepath.Join(dir, "not_ready_seal.json")
	writeWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageSealFixture(t, notReadyPath)
	writeWeightedReadinessFixture(t, notReadyPath, stringsReplaceFirst(readText(t, notReadyPath), `"weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_seal_ready": true`, `"weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_seal_ready": false`))
	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGate([]string{notReadyPath, filepath.Join(dir, "not_ready_final_gate.json")}),
		"weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage seal weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_seal_ready not ready",
	)

	openedSealPath := filepath.Join(dir, "opened_seal.json")
	writeWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageSealFixture(t, openedSealPath)
	writeWeightedReadinessFixture(t, openedSealPath, stringsReplaceFirst(readText(t, openedSealPath), `"admission_seal_ready": false`, `"admission_seal_ready": true`))
	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGate([]string{openedSealPath, filepath.Join(dir, "opened_final_gate.json")}),
		"weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage seal opened admission_seal_ready",
	)

	badSchemaPath := filepath.Join(dir, "bad_schema.json")
	writeWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageSealFixture(t, badSchemaPath)
	writeWeightedReadinessFixture(t, badSchemaPath, stringsReplaceFirst(readText(t, badSchemaPath), `"schema": "arianna.live_route_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_seal.v1"`, `"schema": "arianna.live_route_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_seal.v0"`))
	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGate([]string{badSchemaPath, filepath.Join(dir, "bad_schema_final_gate.json")}),
		`weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage seal schema mismatch: got "arianna.live_route_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_seal.v0" want "`+admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageSealSchema+`"`,
	)

	badHashPath := filepath.Join(dir, "bad_hash.json")
	writeWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageSealFixture(t, badHashPath)
	writeWeightedReadinessFixture(t, badHashPath, stringsReplaceFirst(readText(t, badHashPath), `"admission_seal_hash": "weighted-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-candidate-store-reader-proof-precondition-decision-promotion-switch-enable-gate-live-stage-seal-`, `"admission_seal_hash": "weighted-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-candidate-store-reader-proof-precondition-decision-promotion-switch-enable-gate-live-stage-seal-bad`))
	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGate([]string{badHashPath, filepath.Join(dir, "bad_hash_final_gate.json")}),
		"weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage seal admission_seal_hash mismatch",
	)

	if err := runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGate([]string{sealPath, filepath.Join(dir, "missing", "final_gate.json")}); err == nil ||
		!strings.HasPrefix(err.Error(), "weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage final gate write failed:") {
		t.Fatalf("expected weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage final gate write failure, got %v", err)
	}
}
