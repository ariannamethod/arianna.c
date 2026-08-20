package main

import (
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionWriterInventory(t *testing.T) {
	dir := t.TempDir()

	usage := "usage: --admission-live-route-weighted-admission-resonance-graft-admission-writer-inventory RESONANCE_GRAFT_ADMISSION_WRITER_PREFLIGHT_REPORT RESONANCE_GRAFT_ADMISSION_WRITER_INVENTORY_REPORT"
	requireBoundaryAssertError(t, runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionWriterInventory(nil), usage)
	requireBoundaryAssertError(t, runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionWriterInventory([]string{"writer_preflight.json"}), usage)
	requireBoundaryAssertError(t, runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionWriterInventory([]string{"writer_preflight.json", "writer_inventory.json", "extra"}), usage)
	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionWriterInventory([]string{"  ", filepath.Join(dir, "writer_inventory.json")}),
		"weighted admission resonance graft admission writer preflight path missing",
	)

	writerPreflightPath := filepath.Join(dir, "writer_preflight.json")
	writeWeightedAdmissionResonanceGraftAdmissionWriterPreflightFixture(t, writerPreflightPath)
	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionWriterInventory([]string{writerPreflightPath, "  "}),
		"weighted admission resonance graft admission writer inventory output path missing",
	)

	writerInventoryPath := filepath.Join(dir, "writer_inventory.json")
	if err := runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionWriterInventory([]string{writerPreflightPath, writerInventoryPath}); err != nil {
		t.Fatalf("valid weighted admission resonance graft admission writer inventory rejected: %v", err)
	}
	raw, err := os.ReadFile(writerInventoryPath)
	if err != nil {
		t.Fatalf("read weighted admission resonance graft admission writer inventory: %v", err)
	}
	var inventory admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionWriterInventoryReport
	if err := json.Unmarshal(raw, &inventory); err != nil {
		t.Fatalf("decode weighted admission resonance graft admission writer inventory: %v", err)
	}
	sourceRaw, err := os.ReadFile(writerPreflightPath)
	if err != nil {
		t.Fatalf("read weighted admission resonance graft admission writer preflight: %v", err)
	}
	var sourcePreflight admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionWriterPreflightReport
	if err := json.Unmarshal(sourceRaw, &sourcePreflight); err != nil {
		t.Fatalf("decode weighted admission resonance graft admission writer preflight: %v", err)
	}
	if inventory.Schema != admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionWriterInventorySchema ||
		inventory.Status != "shadow_graft_admission_writer_inventory_blocked_dry_run" ||
		inventory.TargetKind != "weighted_internal_world_shadow_graft_admission_writer_inventory" ||
		inventory.TargetMode != "closed_writer_inventory_guard_dry_run" ||
		inventory.Action != "block_weighted_resonance_shadow_graft_admission_writer_preflight_blocked_dry_run" ||
		inventory.WriterState != "blocked" ||
		inventory.WriterAction != "reject_blocked_writer_preflight" ||
		inventory.RollbackState != "blocked" ||
		inventory.RollbackAction != "reject_blocked_writer_preflight" ||
		inventory.InventoryState != "blocked" ||
		inventory.InventoryAction != "reject_blocked_writer_preflight" ||
		inventory.WriterContract != "none" ||
		inventory.RollbackContract != "none" ||
		inventory.AdmissionLedgerContract != "none" ||
		inventory.WriterContractPresent ||
		inventory.RollbackContractPresent ||
		inventory.LedgerContractPresent ||
		inventory.ContractsReady ||
		inventory.WriterInventoryMode != "closed_writer_preflight_inventory_guard" ||
		inventory.WriterInventoryStage != "pre_writer_contract_graft_admission_writer_inventory" ||
		!inventory.WeightedAdmissionResonanceGraftAdmissionWriterInventoryReady ||
		!inventory.WeightedAdmissionResonanceGraftAdmissionWriterPreflightConsumed ||
		!inventory.WeightedAdmissionResonanceGraftAdmissionWriterPreflightRequired ||
		!inventory.NextStepBlockedWithoutResonanceGraftAdmissionWriterInventory ||
		!inventory.WriterPreflightVerified ||
		!inventory.WriterPreflightHashVerified ||
		!inventory.WriterPreflightReadBackVerified ||
		inventory.CausalID != admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionWriterInventoryCausalID(inventory) ||
		inventory.WriterInventoryHash != admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionWriterInventoryHash(inventory) ||
		inventory.ReadBackHash != admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionWriterInventoryReadBackHash(inventory) ||
		inventory.WriterInventoryHash == inventory.ReadBackHash ||
		inventory.WeightedAdmissionResonanceGraftAdmissionWriterInventoryID != admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionWriterInventoryID(inventory) ||
		inventory.SourceSchema != admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionWriterPreflightSchema ||
		inventory.SourceStatus != "shadow_graft_admission_writer_preflight_blocked_dry_run" ||
		inventory.SourceReport != writerPreflightPath ||
		inventory.SourceWeightedAdmissionResonanceGraftAdmissionWriterPreflightID != sourcePreflight.WeightedAdmissionResonanceGraftAdmissionWriterPreflightID ||
		inventory.SourceWeightedAdmissionResonanceGraftAdmissionWriterPreflightHash != sourcePreflight.WriterPreflightHash ||
		inventory.SourceWeightedAdmissionResonanceGraftAdmissionWriterPreflightReadBack != sourcePreflight.ReadBackHash ||
		inventory.SourceWriterPreflightWriterAction != sourcePreflight.WriterAction ||
		inventory.SourceWriterPreflightRollbackAction != sourcePreflight.RollbackAction ||
		inventory.SourceWriterPreflightBodyTarget != "none" ||
		!inventory.SourceWriterPreflightPassed ||
		inventory.SourceWeightedAdmissionResonanceGraftAdmissionLiveStageID != sourcePreflight.SourceWeightedAdmissionResonanceGraftAdmissionLiveStageID ||
		inventory.SourceWeightedAdmissionResonanceGraftAdmissionEnableGateID != sourcePreflight.SourceWeightedAdmissionResonanceGraftAdmissionEnableGateID ||
		inventory.BodyMutationAllowed ||
		inventory.WriterReady ||
		inventory.RollbackReady ||
		inventory.WriteAllowed ||
		inventory.AdmissionAllowed ||
		inventory.LiveAdmissionEnabled ||
		inventory.MutatesState ||
		inventory.BodyTarget != "none" ||
		!inventory.Passed ||
		inventory.Reason != "weighted resonance shadow graft admission writer inventory blocked by blocked writer preflight; writer, rollback, and ledger contracts remain absent" {
		t.Fatalf("weighted admission resonance graft admission writer inventory lost contract: %+v", inventory)
	}

	openedPreflightPath := filepath.Join(dir, "opened_writer_preflight.json")
	writeWeightedAdmissionResonanceGraftAdmissionWriterPreflightFixture(t, openedPreflightPath)
	writeWeightedReadinessFixture(t, openedPreflightPath, stringsReplaceFirst(readText(t, openedPreflightPath), `"live_admission_enabled": false`, `"live_admission_enabled": true`))
	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionWriterInventory([]string{openedPreflightPath, filepath.Join(dir, "opened_writer_inventory.json")}),
		"weighted admission resonance graft admission writer preflight opened live_admission_enabled",
	)

	badSchemaPath := filepath.Join(dir, "bad_schema_writer_preflight.json")
	writeWeightedAdmissionResonanceGraftAdmissionWriterPreflightFixture(t, badSchemaPath)
	writeWeightedReadinessFixture(t, badSchemaPath, stringsReplaceFirst(readText(t, badSchemaPath), `"schema": "arianna.live_route_weighted_admission_resonance_graft_admission_writer_preflight.v1"`, `"schema": "arianna.live_route_weighted_admission_resonance_graft_admission_writer_preflight.v0"`))
	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionWriterInventory([]string{badSchemaPath, filepath.Join(dir, "bad_schema_writer_inventory.json")}),
		`weighted admission resonance graft admission writer preflight schema mismatch: got "arianna.live_route_weighted_admission_resonance_graft_admission_writer_preflight.v0" want "`+admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionWriterPreflightSchema+`"`,
	)

	badHashPath := filepath.Join(dir, "bad_hash_writer_preflight.json")
	writeWeightedAdmissionResonanceGraftAdmissionWriterPreflightFixture(t, badHashPath)
	writeWeightedReadinessFixture(t, badHashPath, stringsReplaceFirst(readText(t, badHashPath), `"writer_preflight_hash": "weighted-resonance-graft-admission-writer-preflight-`, `"writer_preflight_hash": "weighted-resonance-graft-admission-writer-preflight-bad`))
	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionWriterInventory([]string{badHashPath, filepath.Join(dir, "bad_hash_writer_inventory.json")}),
		"weighted admission resonance graft admission writer preflight writer_preflight_hash mismatch",
	)

	if err := runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionWriterInventory([]string{writerPreflightPath, filepath.Join(dir, "missing", "writer_inventory.json")}); err == nil ||
		!strings.HasPrefix(err.Error(), "weighted admission resonance graft admission writer inventory write failed:") {
		t.Fatalf("expected weighted admission resonance graft admission writer inventory write failure, got %v", err)
	}
}
