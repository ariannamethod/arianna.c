package main

import (
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionLedger(t *testing.T) {
	dir := t.TempDir()

	usage := "usage: --admission-live-route-weighted-admission-resonance-graft-admission-ledger RESONANCE_GRAFT_ADMISSION_WRITER_CONTRACT_REPORT RESONANCE_GRAFT_ADMISSION_LEDGER_REPORT"
	requireBoundaryAssertError(t, runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionLedger(nil), usage)
	requireBoundaryAssertError(t, runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionLedger([]string{"writer_contract.json"}), usage)
	requireBoundaryAssertError(t, runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionLedger([]string{"writer_contract.json", "ledger.json", "extra"}), usage)
	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionLedger([]string{"  ", filepath.Join(dir, "ledger.json")}),
		"weighted admission resonance graft admission writer contract path missing",
	)

	writerContractPath := filepath.Join(dir, "writer_contract.json")
	writeWeightedAdmissionResonanceGraftAdmissionWriterContractFixture(t, writerContractPath)
	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionLedger([]string{writerContractPath, "  "}),
		"weighted admission resonance graft admission ledger output path missing",
	)

	ledgerPath := filepath.Join(dir, "ledger.json")
	if err := runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionLedger([]string{writerContractPath, ledgerPath}); err != nil {
		t.Fatalf("valid weighted admission resonance graft admission ledger rejected: %v", err)
	}
	raw, err := os.ReadFile(ledgerPath)
	if err != nil {
		t.Fatalf("read weighted admission resonance graft admission ledger: %v", err)
	}
	var ledger admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionLedgerReport
	if err := json.Unmarshal(raw, &ledger); err != nil {
		t.Fatalf("decode weighted admission resonance graft admission ledger: %v", err)
	}
	sourceRaw, err := os.ReadFile(writerContractPath)
	if err != nil {
		t.Fatalf("read weighted admission resonance graft admission writer contract: %v", err)
	}
	var sourceContract admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionWriterContractReport
	if err := json.Unmarshal(sourceRaw, &sourceContract); err != nil {
		t.Fatalf("decode weighted admission resonance graft admission writer contract: %v", err)
	}
	if ledger.Schema != admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionLedgerSchema ||
		ledger.Status != "shadow_graft_admission_ledger_blocked_dry_run" ||
		ledger.TargetKind != "weighted_internal_world_shadow_graft_admission_ledger" ||
		ledger.TargetMode != "closed_admission_ledger_guard_dry_run" ||
		ledger.Action != "block_weighted_resonance_shadow_graft_admission_writer_contract_blocked_dry_run" ||
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
		ledger.AdmissionLedgerStage != "pre_ledger_append_graft_admission_ledger" ||
		!ledger.WeightedAdmissionResonanceGraftAdmissionLedgerReady ||
		!ledger.WeightedAdmissionResonanceGraftAdmissionWriterContractConsumed ||
		!ledger.WeightedAdmissionResonanceGraftAdmissionWriterContractRequired ||
		!ledger.NextStepBlockedWithoutResonanceGraftAdmissionLedger ||
		ledger.CausalID != admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionLedgerCausalID(ledger) ||
		ledger.AdmissionLedgerHash != admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionLedgerHash(ledger) ||
		ledger.ReadBackHash != admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionLedgerReadBackHash(ledger) ||
		ledger.AdmissionLedgerHash == ledger.ReadBackHash ||
		ledger.WeightedAdmissionResonanceGraftAdmissionLedgerID != admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionLedgerID(ledger) ||
		ledger.SourceSchema != admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionWriterContractSchema ||
		ledger.SourceStatus != "shadow_graft_admission_writer_contract_blocked_dry_run" ||
		ledger.SourceReport != writerContractPath ||
		ledger.SourceWeightedAdmissionResonanceGraftAdmissionWriterContractID != sourceContract.WeightedAdmissionResonanceGraftAdmissionWriterContractID ||
		ledger.SourceWeightedAdmissionResonanceGraftAdmissionWriterContractHash != sourceContract.WriterContractHash ||
		ledger.SourceWeightedAdmissionResonanceGraftAdmissionWriterContractReadBack != sourceContract.ReadBackHash ||
		ledger.SourceWriterContractReceiptShape != sourceContract.ReceiptShape ||
		ledger.SourceWriterContractContractAction != sourceContract.ContractAction ||
		ledger.SourceWriterContractWriterAction != sourceContract.WriterAction ||
		ledger.SourceWriterContractRollbackAction != sourceContract.RollbackAction ||
		ledger.SourceWriterContractWriterContract != "none" ||
		ledger.SourceWriterContractContractsReady ||
		ledger.SourceWriterContractBodyTarget != "none" ||
		ledger.SourceWeightedAdmissionResonanceGraftAdmissionWriterInventoryID != sourceContract.SourceWeightedAdmissionResonanceGraftAdmissionWriterInventoryID ||
		ledger.SourceWeightedAdmissionResonanceGraftAdmissionWriterInventoryHash != sourceContract.SourceWeightedAdmissionResonanceGraftAdmissionWriterInventoryHash ||
		ledger.SourceWriterInventoryWriterAction != sourceContract.SourceWriterInventoryWriterAction ||
		ledger.SourceWriterInventoryWriterContract != "none" ||
		ledger.SourceWeightedAdmissionResonanceGraftAdmissionWriterPreflightID != sourceContract.SourceWeightedAdmissionResonanceGraftAdmissionWriterPreflightID ||
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
		ledger.Reason != "weighted resonance shadow graft admission ledger blocked by blocked writer contract; ledger receipt append remains closed" {
		t.Fatalf("weighted admission resonance graft admission ledger lost contract: %+v", ledger)
	}

	openedContractPath := filepath.Join(dir, "open_contract.json")
	writeWeightedAdmissionResonanceGraftAdmissionWriterContractFixture(t, openedContractPath)
	writeWeightedReadinessFixture(t, openedContractPath, stringsReplaceFirst(readText(t, openedContractPath), `"live_admission_enabled": false`, `"live_admission_enabled": true`))
	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionLedger([]string{openedContractPath, filepath.Join(dir, "opened_ledger.json")}),
		"weighted admission resonance graft admission writer contract opened live_admission_enabled",
	)

	badSchemaPath := filepath.Join(dir, "bad_schema.json")
	writeWeightedAdmissionResonanceGraftAdmissionWriterContractFixture(t, badSchemaPath)
	writeWeightedReadinessFixture(t, badSchemaPath, stringsReplaceFirst(readText(t, badSchemaPath), `"schema": "arianna.live_route_weighted_admission_resonance_graft_admission_writer_contract.v1"`, `"schema": "arianna.live_route_weighted_admission_resonance_graft_admission_writer_contract.v0"`))
	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionLedger([]string{badSchemaPath, filepath.Join(dir, "bad_schema_ledger.json")}),
		`weighted admission resonance graft admission writer contract schema mismatch: got "arianna.live_route_weighted_admission_resonance_graft_admission_writer_contract.v0" want "`+admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionWriterContractSchema+`"`,
	)

	badHashPath := filepath.Join(dir, "bad_hash.json")
	writeWeightedAdmissionResonanceGraftAdmissionWriterContractFixture(t, badHashPath)
	writeWeightedReadinessFixture(t, badHashPath, stringsReplaceFirst(readText(t, badHashPath), `"writer_contract_hash": "weighted-resonance-graft-admission-writer-contract-`, `"writer_contract_hash": "weighted-resonance-graft-admission-writer-contract-bad`))
	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionLedger([]string{badHashPath, filepath.Join(dir, "bad_hash_ledger.json")}),
		"weighted admission resonance graft admission writer contract writer_contract_hash mismatch",
	)

	if err := runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionLedger([]string{writerContractPath, filepath.Join(dir, "missing", "ledger.json")}); err == nil ||
		!strings.HasPrefix(err.Error(), "weighted admission resonance graft admission ledger write failed:") {
		t.Fatalf("expected weighted admission resonance graft admission ledger write failure, got %v", err)
	}
}
