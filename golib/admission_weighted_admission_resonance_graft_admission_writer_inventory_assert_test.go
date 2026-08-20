package main

import (
	"path/filepath"
	"strings"
	"testing"
)

func TestAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionWriterInventoryAssert(t *testing.T) {
	dir := t.TempDir()

	requireBoundaryAssertError(t, runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionWriterInventoryAssert(nil), "usage: --admission-live-route-weighted-admission-resonance-graft-admission-writer-inventory-assert REPORT")
	requireBoundaryAssertError(t, runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionWriterInventoryAssert([]string{"writer_inventory.json", "extra"}), "usage: --admission-live-route-weighted-admission-resonance-graft-admission-writer-inventory-assert REPORT")
	requireBoundaryAssertError(t, runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionWriterInventoryAssert([]string{"  "}), "weighted admission resonance graft admission writer inventory path missing")
	requireBoundaryAssertError(t, runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionWriterInventoryAssert([]string{filepath.Join(dir, "missing.json")}), "weighted admission resonance graft admission writer inventory not written")

	emptyPath := filepath.Join(dir, "empty.json")
	writeWeightedReadinessFixture(t, emptyPath, "")
	requireBoundaryAssertError(t, runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionWriterInventoryAssert([]string{emptyPath}), "weighted admission resonance graft admission writer inventory not written")

	invalidPath := filepath.Join(dir, "invalid.json")
	writeWeightedReadinessFixture(t, invalidPath, "{")
	err := runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionWriterInventoryAssert([]string{invalidPath})
	if err == nil || !strings.HasPrefix(err.Error(), "weighted admission resonance graft admission writer inventory JSON invalid:") {
		t.Fatalf("expected invalid JSON error, got %v", err)
	}

	missingSchemaPath := filepath.Join(dir, "missing_schema.json")
	writeWeightedAdmissionResonanceGraftAdmissionWriterInventoryFixture(t, missingSchemaPath)
	writeWeightedReadinessFixture(t, missingSchemaPath, stringsReplaceFirst(readText(t, missingSchemaPath), `"schema": "arianna.live_route_weighted_admission_resonance_graft_admission_writer_inventory.v1",`, ""))
	requireBoundaryAssertError(t, runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionWriterInventoryAssert([]string{missingSchemaPath}), "weighted admission resonance graft admission writer inventory schema missing")

	badSchemaPath := filepath.Join(dir, "bad_schema.json")
	writeWeightedAdmissionResonanceGraftAdmissionWriterInventoryFixture(t, badSchemaPath)
	writeWeightedReadinessFixture(t, badSchemaPath, stringsReplaceFirst(readText(t, badSchemaPath), `"schema": "arianna.live_route_weighted_admission_resonance_graft_admission_writer_inventory.v1"`, `"schema": "arianna.live_route_weighted_admission_resonance_graft_admission_writer_inventory.v0"`))
	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionWriterInventoryAssert([]string{badSchemaPath}),
		`weighted admission resonance graft admission writer inventory schema mismatch: got "arianna.live_route_weighted_admission_resonance_graft_admission_writer_inventory.v0" want "`+admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionWriterInventorySchema+`"`,
	)

	validPath := filepath.Join(dir, "valid.json")
	writeWeightedAdmissionResonanceGraftAdmissionWriterInventoryFixture(t, validPath)
	if err := runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionWriterInventoryAssert([]string{validPath}); err != nil {
		t.Fatalf("valid weighted admission resonance graft admission writer inventory rejected: %v", err)
	}

	badStatusPath := filepath.Join(dir, "bad_status.json")
	writeWeightedAdmissionResonanceGraftAdmissionWriterInventoryFixture(t, badStatusPath)
	writeWeightedReadinessFixture(t, badStatusPath, stringsReplaceFirst(readText(t, badStatusPath), `"status": "shadow_graft_admission_writer_inventory_blocked_dry_run"`, `"status": "open"`))
	requireBoundaryAssertError(t, runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionWriterInventoryAssert([]string{badStatusPath}), `weighted admission resonance graft admission writer inventory status mismatch: got "open" want "shadow_graft_admission_writer_inventory_blocked_dry_run"`)

	notReadyPath := filepath.Join(dir, "not_ready.json")
	writeWeightedAdmissionResonanceGraftAdmissionWriterInventoryFixture(t, notReadyPath)
	writeWeightedReadinessFixture(t, notReadyPath, stringsReplaceFirst(readText(t, notReadyPath), `"weighted_admission_resonance_graft_admission_writer_inventory_ready": true`, `"weighted_admission_resonance_graft_admission_writer_inventory_ready": false`))
	requireBoundaryAssertError(t, runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionWriterInventoryAssert([]string{notReadyPath}), "weighted admission resonance graft admission writer inventory weighted_admission_resonance_graft_admission_writer_inventory_ready not ready")

	badInventoryActionPath := filepath.Join(dir, "bad_inventory_action.json")
	writeWeightedAdmissionResonanceGraftAdmissionWriterInventoryFixture(t, badInventoryActionPath)
	writeWeightedReadinessFixture(t, badInventoryActionPath, stringsReplaceFirst(readText(t, badInventoryActionPath), `"inventory_action": "reject_blocked_writer_preflight"`, `"inventory_action": "name_required_contracts"`))
	requireBoundaryAssertError(t, runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionWriterInventoryAssert([]string{badInventoryActionPath}), `weighted admission resonance graft admission writer inventory inventory_action mismatch: got "name_required_contracts" want "reject_blocked_writer_preflight"`)

	namedContractPath := filepath.Join(dir, "named_contract.json")
	writeWeightedAdmissionResonanceGraftAdmissionWriterInventoryFixture(t, namedContractPath)
	writeWeightedReadinessFixture(t, namedContractPath, stringsReplaceFirst(readText(t, namedContractPath), `"writer_contract": "none"`, `"writer_contract": "live_admission_writer.v1"`))
	requireBoundaryAssertError(t, runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionWriterInventoryAssert([]string{namedContractPath}), "weighted admission resonance graft admission writer inventory contracts unexpectedly named")

	openedPath := filepath.Join(dir, "opened.json")
	writeWeightedAdmissionResonanceGraftAdmissionWriterInventoryFixture(t, openedPath)
	writeWeightedReadinessFixture(t, openedPath, stringsReplaceFirst(readText(t, openedPath), `"live_admission_enabled": false`, `"live_admission_enabled": true`))
	requireBoundaryAssertError(t, runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionWriterInventoryAssert([]string{openedPath}), "weighted admission resonance graft admission writer inventory opened live_admission_enabled")

	openedSourcePreflightPath := filepath.Join(dir, "opened_source_preflight.json")
	writeWeightedAdmissionResonanceGraftAdmissionWriterInventoryFixture(t, openedSourcePreflightPath)
	writeWeightedReadinessFixture(t, openedSourcePreflightPath, stringsReplaceFirst(readText(t, openedSourcePreflightPath), `"source_writer_preflight_live_admission_enabled": false`, `"source_writer_preflight_live_admission_enabled": true`))
	requireBoundaryAssertError(t, runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionWriterInventoryAssert([]string{openedSourcePreflightPath}), "weighted admission resonance graft admission writer inventory opened source_writer_preflight_live_admission_enabled")

	missingPathField := filepath.Join(dir, "missing_path.json")
	writeWeightedAdmissionResonanceGraftAdmissionWriterInventoryFixture(t, missingPathField)
	writerPreflightReport := filepath.Join(dir, "writer_preflight-"+filepath.Base(missingPathField))
	writeWeightedReadinessFixture(t, missingPathField, stringsReplaceFirst(readText(t, missingPathField), `"source_report": "`+writerPreflightReport+`"`, `"source_report": " "`))
	requireBoundaryAssertError(t, runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionWriterInventoryAssert([]string{missingPathField}), "weighted admission resonance graft admission writer inventory source_report missing")

	badSourcePath := filepath.Join(dir, "bad_source.json")
	writeWeightedAdmissionResonanceGraftAdmissionWriterInventoryFixture(t, badSourcePath)
	writeWeightedReadinessFixture(t, badSourcePath, stringsReplaceFirst(readText(t, badSourcePath), `"source_schema": "arianna.live_route_weighted_admission_resonance_graft_admission_writer_preflight.v1"`, `"source_schema": "arianna.live_route_weighted_admission_resonance_graft_admission_writer_preflight.v0"`))
	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionWriterInventoryAssert([]string{badSourcePath}),
		`weighted admission resonance graft admission writer inventory source_schema mismatch: got "arianna.live_route_weighted_admission_resonance_graft_admission_writer_preflight.v0" want "`+admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionWriterPreflightSchema+`"`,
	)

	badSourcePreflightPath := filepath.Join(dir, "bad_source_preflight.json")
	writeWeightedAdmissionResonanceGraftAdmissionWriterInventoryFixture(t, badSourcePreflightPath)
	writeWeightedReadinessFixture(t, badSourcePreflightPath, stringsReplaceFirst(readText(t, badSourcePreflightPath), `"source_writer_preflight_kind": "shadow_graft_admission_writer_preflight"`, `"source_writer_preflight_kind": "writer_preflight"`))
	requireBoundaryAssertError(t, runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionWriterInventoryAssert([]string{badSourcePreflightPath}), "weighted admission resonance graft admission writer inventory source writer preflight shape mismatch")

	badWriterInventoryHashPath := filepath.Join(dir, "bad_writer_inventory_hash.json")
	writeWeightedAdmissionResonanceGraftAdmissionWriterInventoryFixture(t, badWriterInventoryHashPath)
	writeWeightedReadinessFixture(t, badWriterInventoryHashPath, stringsReplaceFirst(readText(t, badWriterInventoryHashPath), `"writer_inventory_hash": "weighted-resonance-graft-admission-writer-inventory-`, `"writer_inventory_hash": "weighted-resonance-graft-admission-writer-inventory-bad`))
	requireBoundaryAssertError(t, runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionWriterInventoryAssert([]string{badWriterInventoryHashPath}), "weighted admission resonance graft admission writer inventory writer_inventory_hash mismatch")

	badBodyTargetPath := filepath.Join(dir, "bad_body_target.json")
	writeWeightedAdmissionResonanceGraftAdmissionWriterInventoryFixture(t, badBodyTargetPath)
	writeWeightedReadinessFixture(t, badBodyTargetPath, stringsReplaceFirst(readText(t, badBodyTargetPath), `"body_target": "none"`, `"body_target": "live"`))
	requireBoundaryAssertError(t, runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionWriterInventoryAssert([]string{badBodyTargetPath}), `weighted admission resonance graft admission writer inventory body_target mismatch: got "live" want "none"`)
}

func writeWeightedAdmissionResonanceGraftAdmissionWriterInventoryFixture(t *testing.T, writerInventoryPath string) {
	t.Helper()
	dir := filepath.Dir(writerInventoryPath)
	writerPreflightPath := filepath.Join(dir, "writer_preflight-"+filepath.Base(writerInventoryPath))
	writeWeightedAdmissionResonanceGraftAdmissionWriterPreflightFixture(t, writerPreflightPath)
	if err := runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionWriterInventory([]string{writerPreflightPath, writerInventoryPath}); err != nil {
		t.Fatalf("write weighted admission resonance graft admission writer inventory fixture: %v", err)
	}
}
