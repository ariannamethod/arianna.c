package main

import (
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionWriterContract(t *testing.T) {
	dir := t.TempDir()

	usage := "usage: --admission-live-route-weighted-admission-resonance-graft-admission-writer-contract RESONANCE_GRAFT_ADMISSION_WRITER_INVENTORY_REPORT RESONANCE_GRAFT_ADMISSION_WRITER_CONTRACT_REPORT"
	requireBoundaryAssertError(t, runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionWriterContract(nil), usage)
	requireBoundaryAssertError(t, runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionWriterContract([]string{"writer_inventory.json"}), usage)
	requireBoundaryAssertError(t, runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionWriterContract([]string{"writer_inventory.json", "writer_contract.json", "extra"}), usage)
	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionWriterContract([]string{"  ", filepath.Join(dir, "writer_contract.json")}),
		"weighted admission resonance graft admission writer inventory path missing",
	)

	writerInventoryPath := filepath.Join(dir, "writer_inventory.json")
	writeWeightedAdmissionResonanceGraftAdmissionWriterInventoryFixture(t, writerInventoryPath)
	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionWriterContract([]string{writerInventoryPath, "  "}),
		"weighted admission resonance graft admission writer contract output path missing",
	)

	writerContractPath := filepath.Join(dir, "writer_contract.json")
	if err := runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionWriterContract([]string{writerInventoryPath, writerContractPath}); err != nil {
		t.Fatalf("valid weighted admission resonance graft admission writer contract rejected: %v", err)
	}
	raw, err := os.ReadFile(writerContractPath)
	if err != nil {
		t.Fatalf("read weighted admission resonance graft admission writer contract: %v", err)
	}
	var contract admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionWriterContractReport
	if err := json.Unmarshal(raw, &contract); err != nil {
		t.Fatalf("decode weighted admission resonance graft admission writer contract: %v", err)
	}
	sourceRaw, err := os.ReadFile(writerInventoryPath)
	if err != nil {
		t.Fatalf("read weighted admission resonance graft admission writer inventory: %v", err)
	}
	var sourceInventory admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionWriterInventoryReport
	if err := json.Unmarshal(sourceRaw, &sourceInventory); err != nil {
		t.Fatalf("decode weighted admission resonance graft admission writer inventory: %v", err)
	}
	if contract.Schema != admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionWriterContractSchema ||
		contract.Status != "shadow_graft_admission_writer_contract_blocked_dry_run" ||
		contract.TargetKind != "weighted_internal_world_shadow_graft_admission_writer_contract" ||
		contract.TargetMode != "closed_writer_contract_guard_dry_run" ||
		contract.Action != "block_weighted_resonance_shadow_graft_admission_writer_inventory_blocked_dry_run" ||
		contract.WriterState != "blocked" ||
		contract.WriterAction != "reject_blocked_writer_inventory" ||
		contract.RollbackState != "blocked" ||
		contract.RollbackAction != "reject_blocked_writer_inventory" ||
		contract.InventoryState != "blocked" ||
		contract.InventoryAction != "reject_blocked_writer_preflight" ||
		contract.ContractState != "blocked" ||
		contract.ContractAction != "reject_blocked_writer_inventory" ||
		contract.WriterContract != "none" ||
		contract.RollbackContract != "none" ||
		contract.AdmissionLedgerContract != "none" ||
		contract.WriterContractShape != "none" ||
		contract.RollbackContractShape != "none" ||
		contract.LedgerContractShape != "none" ||
		contract.WriteScope != "none" ||
		contract.RollbackScope != "none" ||
		contract.LedgerMode != "none" ||
		contract.WriterContractPresent ||
		contract.RollbackContractPresent ||
		contract.LedgerContractPresent ||
		contract.ContractsReady ||
		contract.WriterContractMode != "closed_writer_inventory_contract_guard" ||
		contract.WriterContractStage != "pre_admission_ledger_graft_admission_writer_contract" ||
		!contract.WeightedAdmissionResonanceGraftAdmissionWriterContractReady ||
		!contract.WeightedAdmissionResonanceGraftAdmissionWriterInventoryConsumed ||
		!contract.WeightedAdmissionResonanceGraftAdmissionWriterInventoryRequired ||
		!contract.NextStepBlockedWithoutResonanceGraftAdmissionWriterContract ||
		!contract.WriterInventoryVerified ||
		!contract.WriterInventoryHashVerified ||
		!contract.WriterInventoryReadBackVerified ||
		!contract.WriterPreflightVerified ||
		!contract.WriterPreflightHashVerified ||
		!contract.WriterPreflightReadBackVerified ||
		contract.CausalID != admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionWriterContractCausalID(contract) ||
		contract.WriterContractHash != admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionWriterContractHash(contract) ||
		contract.ReadBackHash != admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionWriterContractReadBackHash(contract) ||
		contract.WriterContractHash == contract.ReadBackHash ||
		contract.WeightedAdmissionResonanceGraftAdmissionWriterContractID != admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionWriterContractID(contract) ||
		contract.SourceSchema != admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionWriterInventorySchema ||
		contract.SourceStatus != "shadow_graft_admission_writer_inventory_blocked_dry_run" ||
		contract.SourceReport != writerInventoryPath ||
		contract.SourceWeightedAdmissionResonanceGraftAdmissionWriterInventoryID != sourceInventory.WeightedAdmissionResonanceGraftAdmissionWriterInventoryID ||
		contract.SourceWeightedAdmissionResonanceGraftAdmissionWriterInventoryHash != sourceInventory.WriterInventoryHash ||
		contract.SourceWeightedAdmissionResonanceGraftAdmissionWriterInventoryReadBack != sourceInventory.ReadBackHash ||
		contract.SourceWriterInventoryWriterAction != sourceInventory.WriterAction ||
		contract.SourceWriterInventoryRollbackAction != sourceInventory.RollbackAction ||
		contract.SourceWriterInventoryWriterContract != "none" ||
		contract.SourceWriterInventoryContractsReady ||
		contract.SourceWriterInventoryBodyTarget != "none" ||
		contract.SourceWeightedAdmissionResonanceGraftAdmissionWriterPreflightID != sourceInventory.SourceWeightedAdmissionResonanceGraftAdmissionWriterPreflightID ||
		contract.SourceWeightedAdmissionResonanceGraftAdmissionWriterPreflightHash != sourceInventory.SourceWeightedAdmissionResonanceGraftAdmissionWriterPreflightHash ||
		contract.SourceWeightedAdmissionResonanceGraftAdmissionWriterPreflightReadBack != sourceInventory.SourceWeightedAdmissionResonanceGraftAdmissionWriterPreflightReadBack ||
		contract.SourceWriterPreflightWriterAction != sourceInventory.SourceWriterPreflightWriterAction ||
		contract.SourceWriterPreflightRollbackAction != sourceInventory.SourceWriterPreflightRollbackAction ||
		contract.SourceWriterPreflightBodyTarget != "none" ||
		!contract.SourceWriterPreflightPassed ||
		contract.SourceWeightedAdmissionResonanceGraftAdmissionLiveStageID != sourceInventory.SourceWeightedAdmissionResonanceGraftAdmissionLiveStageID ||
		contract.SourceWeightedAdmissionResonanceGraftAdmissionEnableGateID != sourceInventory.SourceWeightedAdmissionResonanceGraftAdmissionEnableGateID ||
		contract.BodyMutationAllowed ||
		contract.WriterReady ||
		contract.RollbackReady ||
		contract.WriteAllowed ||
		contract.AdmissionAllowed ||
		contract.LiveAdmissionEnabled ||
		contract.MutatesState ||
		contract.BodyTarget != "none" ||
		!contract.Passed ||
		contract.Reason != "weighted resonance shadow graft admission writer contract blocked by blocked writer inventory; writer, rollback, and ledger contract shapes remain absent" {
		t.Fatalf("weighted admission resonance graft admission writer contract lost contract: %+v", contract)
	}

	openedInventoryPath := filepath.Join(dir, "open_inv.json")
	writeWeightedAdmissionResonanceGraftAdmissionWriterInventoryFixture(t, openedInventoryPath)
	writeWeightedReadinessFixture(t, openedInventoryPath, stringsReplaceFirst(readText(t, openedInventoryPath), `"live_admission_enabled": false`, `"live_admission_enabled": true`))
	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionWriterContract([]string{openedInventoryPath, filepath.Join(dir, "opened_writer_contract.json")}),
		"weighted admission resonance graft admission writer inventory opened live_admission_enabled",
	)

	badSchemaPath := filepath.Join(dir, "bad_schema.json")
	writeWeightedAdmissionResonanceGraftAdmissionWriterInventoryFixture(t, badSchemaPath)
	writeWeightedReadinessFixture(t, badSchemaPath, stringsReplaceFirst(readText(t, badSchemaPath), `"schema": "arianna.live_route_weighted_admission_resonance_graft_admission_writer_inventory.v1"`, `"schema": "arianna.live_route_weighted_admission_resonance_graft_admission_writer_inventory.v0"`))
	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionWriterContract([]string{badSchemaPath, filepath.Join(dir, "bad_schema_writer_contract.json")}),
		`weighted admission resonance graft admission writer inventory schema mismatch: got "arianna.live_route_weighted_admission_resonance_graft_admission_writer_inventory.v0" want "`+admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionWriterInventorySchema+`"`,
	)

	badHashPath := filepath.Join(dir, "bad_hash.json")
	writeWeightedAdmissionResonanceGraftAdmissionWriterInventoryFixture(t, badHashPath)
	writeWeightedReadinessFixture(t, badHashPath, stringsReplaceFirst(readText(t, badHashPath), `"writer_inventory_hash": "weighted-resonance-graft-admission-writer-inventory-`, `"writer_inventory_hash": "weighted-resonance-graft-admission-writer-inventory-bad`))
	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionWriterContract([]string{badHashPath, filepath.Join(dir, "bad_hash_writer_contract.json")}),
		"weighted admission resonance graft admission writer inventory writer_inventory_hash mismatch",
	)

	if err := runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionWriterContract([]string{writerInventoryPath, filepath.Join(dir, "missing", "writer_contract.json")}); err == nil ||
		!strings.HasPrefix(err.Error(), "weighted admission resonance graft admission writer contract write failed:") {
		t.Fatalf("expected weighted admission resonance graft admission writer contract write failure, got %v", err)
	}
}
