package main

import (
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageLedgerPersistence(t *testing.T) {
	dir := t.TempDir()

	usage := "usage: --admission-live-route-weighted-admission-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-candidate-store-reader-proof-precondition-decision-promotion-switch-enable-gate-live-stage-ledger-persistence RESONANCE_GRAFT_ADMISSION_FINAL_GATE_OBSERVATION_BOUNDARY_PREFLIGHT_GATE_CANDIDATE_STORE_READER_PROOF_PRECONDITION_DECISION_PROMOTION_SWITCH_ENABLE_GATE_LIVE_STAGE_LEDGER_IMPLEMENTATION_REPORT RESONANCE_GRAFT_ADMISSION_FINAL_GATE_OBSERVATION_BOUNDARY_PREFLIGHT_GATE_CANDIDATE_STORE_READER_PROOF_PRECONDITION_DECISION_PROMOTION_SWITCH_ENABLE_GATE_LIVE_STAGE_LEDGER_PERSISTENCE_REPORT"
	requireBoundaryAssertError(t, runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageLedgerPersistence(nil), usage)
	requireBoundaryAssertError(t, runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageLedgerPersistence([]string{"ledger_impl.json"}), usage)
	requireBoundaryAssertError(t, runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageLedgerPersistence([]string{"ledger_impl.json", "ledger_persist.json", "extra"}), usage)
	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageLedgerPersistence([]string{"  ", filepath.Join(dir, "ledger_persist.json")}),
		"weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage ledger implementation path missing",
	)

	implPath := filepath.Join(dir, "ledger_impl.json")
	writeWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageLedgerImplementationFixture(t, implPath)
	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageLedgerPersistence([]string{implPath, "  "}),
		"weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage ledger persistence output path missing",
	)

	persistPath := filepath.Join(dir, "ledger_persist.json")
	if err := runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageLedgerPersistence([]string{implPath, persistPath}); err != nil {
		t.Fatalf("valid weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage ledger persistence rejected: %v", err)
	}
	raw, err := os.ReadFile(persistPath)
	if err != nil {
		t.Fatalf("read weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage ledger persistence: %v", err)
	}
	var persist admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageLedgerPersistenceReport
	if err := json.Unmarshal(raw, &persist); err != nil {
		t.Fatalf("decode weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage ledger persistence: %v", err)
	}
	sourceRaw, err := os.ReadFile(implPath)
	if err != nil {
		t.Fatalf("read weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage ledger implementation: %v", err)
	}
	var sourceImpl admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageLedgerImplementationReport
	if err := json.Unmarshal(sourceRaw, &sourceImpl); err != nil {
		t.Fatalf("decode weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage ledger implementation: %v", err)
	}
	if persist.Schema != admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageLedgerPersistenceSchema ||
		persist.Status != "shadow_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_ledger_persistence_blocked_dry_run" ||
		persist.TargetKind != "weighted_internal_world_shadow_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_ledger_persistence" ||
		persist.TargetMode != "closed_ledger_persistence_guard_dry_run" ||
		persist.Action != "block_weighted_resonance_shadow_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_ledger_implementation_blocked_dry_run" ||
		persist.WriterAction != "reject_blocked_ledger_implementation" ||
		persist.RollbackAction != "reject_blocked_ledger_implementation" ||
		persist.LedgerPersistenceState != "blocked" ||
		persist.LedgerPersistenceAction != "reject_blocked_ledger_implementation" ||
		persist.LedgerPersistenceTarget != "admission_ledger_receipt" ||
		persist.LedgerPersistenceTargetKind != "weighted_internal_world_shadow_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_ledger_implementation" ||
		persist.LedgerPersistenceTargetMode != "closed_persistence_guard_dry_run" ||
		persist.LedgerPersistenceReceiptShape != "none" ||
		persist.LedgerPersistenceWriteScope != "none" ||
		persist.LedgerPersistenceAppendOnly ||
		!persist.LedgerPersistenceDryRunOnly ||
		persist.LedgerPersistenceReceiptPersisted ||
		persist.LedgerPersistenceReady ||
		!persist.WeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageLedgerPersistenceReady ||
		!persist.WeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageLedgerImplementationConsumed ||
		!persist.WeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageLedgerImplementationRequired ||
		!persist.NextStepBlockedWithoutResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageLedgerPersistence ||
		persist.SourceSchema != admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageLedgerImplementationSchema ||
		persist.SourceStatus != "shadow_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_ledger_implementation_blocked_dry_run" ||
		persist.SourceReport != implPath ||
		persist.SourceAdmissionLedgerSchema != admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageLedgerSchema ||
		persist.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageLedgerImplementationID != sourceImpl.WeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageLedgerImplementationID ||
		persist.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageLedgerImplementationHash != sourceImpl.LedgerImplementationHash ||
		persist.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageLedgerImplementationReadBack != sourceImpl.LedgerImplementationReadBackHash ||
		persist.SourceLedgerImplementationReportReceiptShape != sourceImpl.ReceiptShape ||
		persist.SourceLedgerImplementationAction != sourceImpl.LedgerImplementationAction ||
		persist.SourceLedgerImplementationAppendOnly ||
		!persist.SourceLedgerImplementationDryRunOnly ||
		persist.SourceLedgerImplementationReceiptPersisted ||
		persist.SourceLedgerImplementationReady ||
		persist.CausalID != admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageLedgerPersistenceCausalID(persist) ||
		persist.LedgerPersistenceHash != admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageLedgerPersistenceHash(persist) ||
		persist.LedgerPersistenceReadBackHash != admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageLedgerPersistenceReadBackHash(persist) ||
		persist.LedgerPersistenceHash == persist.LedgerPersistenceReadBackHash ||
		persist.WeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageLedgerPersistenceID != admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageLedgerPersistenceID(persist) ||
		persist.LedgerAppendAllowed ||
		persist.WriteAllowed ||
		persist.AdmissionAllowed ||
		persist.LiveAdmissionEnabled ||
		persist.MutatesState ||
		persist.BodyMutationAllowed ||
		persist.BodyTarget != "none" ||
		!persist.Passed ||
		persist.Reason != "weighted resonance shadow graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage ledger persistence blocked by blocked ledger implementation; ledger receipt persistence remains closed" {
		t.Fatalf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage ledger persistence lost contract: %+v", persist)
	}

	openedImplPath := filepath.Join(dir, "open_impl.json")
	writeWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageLedgerImplementationFixture(t, openedImplPath)
	writeWeightedReadinessFixture(t, openedImplPath, stringsReplaceFirst(readText(t, openedImplPath), `"ledger_implementation_ready": false`, `"ledger_implementation_ready": true`))
	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageLedgerPersistence([]string{openedImplPath, filepath.Join(dir, "opened_ledger_persist.json")}),
		"weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage ledger implementation opened ledger_implementation_ready",
	)

	badSchemaPath := filepath.Join(dir, "bad_schema.json")
	writeWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageLedgerImplementationFixture(t, badSchemaPath)
	writeWeightedReadinessFixture(t, badSchemaPath, stringsReplaceFirst(readText(t, badSchemaPath), `"schema": "arianna.live_route_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_ledger_implementation.v1"`, `"schema": "arianna.live_route_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_ledger_implementation.v0"`))
	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageLedgerPersistence([]string{badSchemaPath, filepath.Join(dir, "bad_schema_ledger_persist.json")}),
		`weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage ledger implementation schema mismatch: got "arianna.live_route_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_ledger_implementation.v0" want "`+admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageLedgerImplementationSchema+`"`,
	)

	badHashPath := filepath.Join(dir, "bad_hash.json")
	writeWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageLedgerImplementationFixture(t, badHashPath)
	writeWeightedReadinessFixture(t, badHashPath, stringsReplaceFirst(readText(t, badHashPath), `"ledger_implementation_hash": "weighted-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-candidate-store-reader-proof-precondition-decision-promotion-switch-enable-gate-live-stage-ledger-implementation-`, `"ledger_implementation_hash": "weighted-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-candidate-store-reader-proof-precondition-decision-promotion-switch-enable-gate-live-stage-ledger-implementation-bad`))
	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageLedgerPersistence([]string{badHashPath, filepath.Join(dir, "bad_hash_ledger_persist.json")}),
		"weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage ledger implementation ledger_implementation_hash mismatch",
	)

	if err := runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageLedgerPersistence([]string{implPath, filepath.Join(dir, "missing", "ledger_persist.json")}); err == nil ||
		!strings.HasPrefix(err.Error(), "weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage ledger persistence write failed:") {
		t.Fatalf("expected weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage ledger persistence write failure, got %v", err)
	}
}
