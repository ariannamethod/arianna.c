package main

import (
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageLedger(t *testing.T) {
	dir := t.TempDir()

	usage := "usage: --admission-live-route-weighted-admission-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-candidate-store-reader-proof-precondition-decision-promotion-switch-enable-gate-live-stage-ledger RESONANCE_GRAFT_ADMISSION_FINAL_GATE_OBSERVATION_BOUNDARY_PREFLIGHT_GATE_CANDIDATE_STORE_READER_PROOF_PRECONDITION_DECISION_PROMOTION_SWITCH_ENABLE_GATE_LIVE_STAGE_WRITER_CONTRACT_REPORT RESONANCE_GRAFT_ADMISSION_FINAL_GATE_OBSERVATION_BOUNDARY_PREFLIGHT_GATE_CANDIDATE_STORE_READER_PROOF_PRECONDITION_DECISION_PROMOTION_SWITCH_ENABLE_GATE_LIVE_STAGE_LEDGER_REPORT"
	requireBoundaryAssertError(t, runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageLedger(nil), usage)
	requireBoundaryAssertError(t, runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageLedger([]string{"writer_contract.json"}), usage)
	requireBoundaryAssertError(t, runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageLedger([]string{"writer_contract.json", "ledger.json", "extra"}), usage)
	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageLedger([]string{"  ", filepath.Join(dir, "ledger.json")}),
		"weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage writer contract path missing",
	)

	writerContractPath := filepath.Join(dir, "writer_contract.json")
	writeWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageWriterContractFixture(t, writerContractPath)
	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageLedger([]string{writerContractPath, "  "}),
		"weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage ledger output path missing",
	)

	ledgerPath := filepath.Join(dir, "ledger.json")
	if err := runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageLedger([]string{writerContractPath, ledgerPath}); err != nil {
		t.Fatalf("valid weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage ledger rejected: %v", err)
	}
	raw, err := os.ReadFile(ledgerPath)
	if err != nil {
		t.Fatalf("read weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage ledger: %v", err)
	}
	var ledger admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageLedgerReport
	if err := json.Unmarshal(raw, &ledger); err != nil {
		t.Fatalf("decode weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage ledger: %v", err)
	}
	sourceRaw, err := os.ReadFile(writerContractPath)
	if err != nil {
		t.Fatalf("read weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage writer contract: %v", err)
	}
	var sourceContract admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageWriterContractReport
	if err := json.Unmarshal(sourceRaw, &sourceContract); err != nil {
		t.Fatalf("decode weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage writer contract: %v", err)
	}
	if ledger.Schema != admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageLedgerSchema ||
		ledger.Status != "shadow_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_ledger_blocked_dry_run" ||
		ledger.TargetKind != "weighted_internal_world_shadow_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_ledger" ||
		ledger.TargetMode != "closed_admission_ledger_guard_dry_run" ||
		ledger.Action != "block_weighted_resonance_shadow_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_writer_contract_blocked_dry_run" ||
		ledger.WriterState != "blocked" ||
		ledger.WriterAction != "reject_blocked_writer_contract" ||
		ledger.RollbackState != "blocked" ||
		ledger.RollbackAction != "reject_blocked_writer_contract" ||
		ledger.InventoryState != "blocked" ||
		ledger.InventoryAction != "reject_blocked_writer_preflight" ||
		ledger.ContractState != "blocked" ||
		ledger.ContractAction != "reject_blocked_writer_inventory" ||
		ledger.LedgerState != "blocked" ||
		ledger.LedgerAction != "reject_blocked_writer_contract" ||
		ledger.WriterContract != "none" ||
		ledger.RollbackContract != "none" ||
		ledger.AdmissionLedgerContract != "none" ||
		ledger.WriterContractShape != "none" ||
		ledger.RollbackContractShape != "none" ||
		ledger.LedgerContractShape != "none" ||
		ledger.LedgerContract != "none" ||
		ledger.LedgerEntrypoint != "none" ||
		ledger.LedgerReceiptShape != "none" ||
		ledger.LedgerWriteScope != "none" ||
		ledger.WriteScope != "none" ||
		ledger.RollbackScope != "none" ||
		ledger.LedgerMode != "none" ||
		ledger.WriterContractPresent ||
		ledger.RollbackContractPresent ||
		ledger.LedgerContractPresent ||
		ledger.ContractsReady ||
		ledger.LedgerReady ||
		ledger.LedgerAppendAllowed ||
		ledger.AdmissionLedgerMode != "closed_writer_contract_ledger_guard" ||
		ledger.AdmissionLedgerStage != "post_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_writer_contract_pre_ledger_append" ||
		!ledger.WeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageLedgerReady ||
		!ledger.WeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageWriterContractConsumed ||
		!ledger.WeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageWriterContractRequired ||
		!ledger.NextStepBlockedWithoutResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageLedger ||
		ledger.CausalID != admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageLedgerCausalID(ledger) ||
		ledger.AdmissionLedgerHash != admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageLedgerHash(ledger) ||
		ledger.ReadBackHash != admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageLedgerReadBackHash(ledger) ||
		ledger.AdmissionLedgerHash == ledger.ReadBackHash ||
		ledger.WeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageLedgerID != admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageLedgerID(ledger) ||
		ledger.SourceSchema != admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageWriterContractSchema ||
		ledger.SourceStatus != "shadow_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_writer_contract_blocked_dry_run" ||
		ledger.SourceReport != writerContractPath ||
		ledger.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageWriterContractID != sourceContract.WeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageWriterContractID ||
		ledger.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageWriterContractHash != sourceContract.WriterContractHash ||
		ledger.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageWriterContractReadBack != sourceContract.ReadBackHash ||
		ledger.SourceWriterContractReceiptShape != sourceContract.ReceiptShape ||
		ledger.SourceWriterContractContractAction != sourceContract.ContractAction ||
		ledger.SourceWriterContractWriterAction != sourceContract.WriterAction ||
		ledger.SourceWriterContractRollbackAction != sourceContract.RollbackAction ||
		ledger.SourceWriterContractWriterContract != "none" ||
		ledger.SourceWriterContractContractsReady ||
		ledger.SourceWriterContractBodyTarget != "none" ||
		ledger.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageWriterInventoryID != sourceContract.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageWriterInventoryID ||
		ledger.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageWriterInventoryHash != sourceContract.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageWriterInventoryHash ||
		ledger.SourceWriterInventoryWriterAction != sourceContract.SourceWriterInventoryWriterAction ||
		ledger.SourceWriterInventoryWriterContract != "none" ||
		ledger.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageWriterPreflightID != sourceContract.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageWriterPreflightID ||
		ledger.SourceWriterPreflightWriterAction != sourceContract.SourceWriterPreflightWriterAction ||
		ledger.SourceWriterPreflightBodyTarget != "none" ||
		!ledger.SourceWriterPreflightPassed ||
		ledger.BodyMutationAllowed ||
		ledger.WriterReady ||
		ledger.RollbackReady ||
		ledger.WriteAllowed ||
		ledger.AdmissionAllowed ||
		ledger.LiveAdmissionEnabled ||
		ledger.MutatesState ||
		ledger.BodyTarget != "none" ||
		!ledger.Passed ||
		ledger.Reason != "weighted resonance shadow graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage ledger blocked by blocked writer contract; ledger receipt append remains closed" {
		t.Fatalf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage ledger lost contract: %+v", ledger)
	}

	openedContractPath := filepath.Join(dir, "open_contract.json")
	writeWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageWriterContractFixture(t, openedContractPath)
	writeWeightedReadinessFixture(t, openedContractPath, stringsReplaceFirst(readText(t, openedContractPath), `"live_admission_enabled": false`, `"live_admission_enabled": true`))
	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageLedger([]string{openedContractPath, filepath.Join(dir, "opened_ledger.json")}),
		"weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage writer contract opened live_admission_enabled",
	)

	badSchemaPath := filepath.Join(dir, "bad_schema.json")
	writeWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageWriterContractFixture(t, badSchemaPath)
	writeWeightedReadinessFixture(t, badSchemaPath, stringsReplaceFirst(readText(t, badSchemaPath), `"schema": "arianna.live_route_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_writer_contract.v1"`, `"schema": "arianna.live_route_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_writer_contract.v0"`))
	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageLedger([]string{badSchemaPath, filepath.Join(dir, "bad_schema_ledger.json")}),
		`weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage writer contract schema mismatch: got "arianna.live_route_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_writer_contract.v0" want "`+admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageWriterContractSchema+`"`,
	)

	badHashPath := filepath.Join(dir, "bad_hash.json")
	writeWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageWriterContractFixture(t, badHashPath)
	writeWeightedReadinessFixture(t, badHashPath, stringsReplaceFirst(readText(t, badHashPath), `"writer_contract_hash": "weighted-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-candidate-store-reader-proof-precondition-decision-promotion-switch-enable-gate-live-stage-writer-contract-`, `"writer_contract_hash": "weighted-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-candidate-store-reader-proof-precondition-decision-promotion-switch-enable-gate-live-stage-writer-contract-bad`))
	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageLedger([]string{badHashPath, filepath.Join(dir, "bad_hash_ledger.json")}),
		"weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage writer contract writer_contract_hash mismatch",
	)

	if err := runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageLedger([]string{writerContractPath, filepath.Join(dir, "missing", "ledger.json")}); err == nil ||
		!strings.HasPrefix(err.Error(), "weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage ledger write failed:") {
		t.Fatalf("expected weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage ledger write failure, got %v", err)
	}
}
