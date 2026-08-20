package main

import (
	"path/filepath"
	"strings"
	"testing"
)

func TestAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionWriterContractAssert(t *testing.T) {
	dir := t.TempDir()

	requireBoundaryAssertError(t, runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionWriterContractAssert(nil), "usage: --admission-live-route-weighted-admission-resonance-graft-admission-writer-contract-assert REPORT")
	requireBoundaryAssertError(t, runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionWriterContractAssert([]string{"writer_contract.json", "extra"}), "usage: --admission-live-route-weighted-admission-resonance-graft-admission-writer-contract-assert REPORT")
	requireBoundaryAssertError(t, runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionWriterContractAssert([]string{"  "}), "weighted admission resonance graft admission writer contract path missing")
	requireBoundaryAssertError(t, runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionWriterContractAssert([]string{filepath.Join(dir, "missing.json")}), "weighted admission resonance graft admission writer contract not written")

	emptyPath := filepath.Join(dir, "empty.json")
	writeWeightedReadinessFixture(t, emptyPath, "")
	requireBoundaryAssertError(t, runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionWriterContractAssert([]string{emptyPath}), "weighted admission resonance graft admission writer contract not written")

	invalidPath := filepath.Join(dir, "invalid.json")
	writeWeightedReadinessFixture(t, invalidPath, "{")
	err := runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionWriterContractAssert([]string{invalidPath})
	if err == nil || !strings.HasPrefix(err.Error(), "weighted admission resonance graft admission writer contract JSON invalid:") {
		t.Fatalf("expected invalid JSON error, got %v", err)
	}

	missingSchemaPath := filepath.Join(dir, "missing_schema.json")
	writeWeightedAdmissionResonanceGraftAdmissionWriterContractFixture(t, missingSchemaPath)
	writeWeightedReadinessFixture(t, missingSchemaPath, stringsReplaceFirst(readText(t, missingSchemaPath), `"schema": "arianna.live_route_weighted_admission_resonance_graft_admission_writer_contract.v1",`, ""))
	requireBoundaryAssertError(t, runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionWriterContractAssert([]string{missingSchemaPath}), "weighted admission resonance graft admission writer contract schema missing")

	badSchemaPath := filepath.Join(dir, "bad_schema.json")
	writeWeightedAdmissionResonanceGraftAdmissionWriterContractFixture(t, badSchemaPath)
	writeWeightedReadinessFixture(t, badSchemaPath, stringsReplaceFirst(readText(t, badSchemaPath), `"schema": "arianna.live_route_weighted_admission_resonance_graft_admission_writer_contract.v1"`, `"schema": "arianna.live_route_weighted_admission_resonance_graft_admission_writer_contract.v0"`))
	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionWriterContractAssert([]string{badSchemaPath}),
		`weighted admission resonance graft admission writer contract schema mismatch: got "arianna.live_route_weighted_admission_resonance_graft_admission_writer_contract.v0" want "`+admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionWriterContractSchema+`"`,
	)

	validPath := filepath.Join(dir, "valid.json")
	writeWeightedAdmissionResonanceGraftAdmissionWriterContractFixture(t, validPath)
	if err := runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionWriterContractAssert([]string{validPath}); err != nil {
		t.Fatalf("valid weighted admission resonance graft admission writer contract rejected: %v", err)
	}

	badStatusPath := filepath.Join(dir, "bad_status.json")
	writeWeightedAdmissionResonanceGraftAdmissionWriterContractFixture(t, badStatusPath)
	writeWeightedReadinessFixture(t, badStatusPath, stringsReplaceFirst(readText(t, badStatusPath), `"status": "shadow_graft_admission_writer_contract_blocked_dry_run"`, `"status": "open"`))
	requireBoundaryAssertError(t, runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionWriterContractAssert([]string{badStatusPath}), `weighted admission resonance graft admission writer contract status mismatch: got "open" want "shadow_graft_admission_writer_contract_blocked_dry_run"`)

	notReadyPath := filepath.Join(dir, "not_ready.json")
	writeWeightedAdmissionResonanceGraftAdmissionWriterContractFixture(t, notReadyPath)
	writeWeightedReadinessFixture(t, notReadyPath, stringsReplaceFirst(readText(t, notReadyPath), `"weighted_admission_resonance_graft_admission_writer_contract_ready": true`, `"weighted_admission_resonance_graft_admission_writer_contract_ready": false`))
	requireBoundaryAssertError(t, runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionWriterContractAssert([]string{notReadyPath}), "weighted admission resonance graft admission writer contract weighted_admission_resonance_graft_admission_writer_contract_ready not ready")

	badInventoryActionPath := filepath.Join(dir, "bad_inventory_action.json")
	writeWeightedAdmissionResonanceGraftAdmissionWriterContractFixture(t, badInventoryActionPath)
	writeWeightedReadinessFixture(t, badInventoryActionPath, stringsReplaceFirst(readText(t, badInventoryActionPath), `"inventory_action": "reject_blocked_writer_preflight"`, `"inventory_action": "name_required_contracts"`))
	requireBoundaryAssertError(t, runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionWriterContractAssert([]string{badInventoryActionPath}), `weighted admission resonance graft admission writer contract inventory_action mismatch: got "name_required_contracts" want "reject_blocked_writer_preflight"`)

	badContractActionPath := filepath.Join(dir, "bad_contract_action.json")
	writeWeightedAdmissionResonanceGraftAdmissionWriterContractFixture(t, badContractActionPath)
	writeWeightedReadinessFixture(t, badContractActionPath, stringsReplaceFirst(readText(t, badContractActionPath), `"contract_action": "reject_blocked_writer_inventory"`, `"contract_action": "name_required_contracts"`))
	requireBoundaryAssertError(t, runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionWriterContractAssert([]string{badContractActionPath}), `weighted admission resonance graft admission writer contract contract_action mismatch: got "name_required_contracts" want "reject_blocked_writer_inventory"`)

	namedContractPath := filepath.Join(dir, "named_contract.json")
	writeWeightedAdmissionResonanceGraftAdmissionWriterContractFixture(t, namedContractPath)
	writeWeightedReadinessFixture(t, namedContractPath, stringsReplaceFirst(readText(t, namedContractPath), `"writer_contract": "none"`, `"writer_contract": "live_admission_writer.v1"`))
	requireBoundaryAssertError(t, runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionWriterContractAssert([]string{namedContractPath}), "weighted admission resonance graft admission writer contract contracts unexpectedly named")

	namedContractShapePath := filepath.Join(dir, "named_contract_shape.json")
	writeWeightedAdmissionResonanceGraftAdmissionWriterContractFixture(t, namedContractShapePath)
	writeWeightedReadinessFixture(t, namedContractShapePath, stringsReplaceFirst(readText(t, namedContractShapePath), `"writer_contract_shape": "none"`, `"writer_contract_shape": "live_admission_writer_shape.v1"`))
	requireBoundaryAssertError(t, runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionWriterContractAssert([]string{namedContractShapePath}), "weighted admission resonance graft admission writer contract contract shapes unexpectedly named")

	openedPath := filepath.Join(dir, "opened.json")
	writeWeightedAdmissionResonanceGraftAdmissionWriterContractFixture(t, openedPath)
	writeWeightedReadinessFixture(t, openedPath, stringsReplaceFirst(readText(t, openedPath), `"live_admission_enabled": false`, `"live_admission_enabled": true`))
	requireBoundaryAssertError(t, runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionWriterContractAssert([]string{openedPath}), "weighted admission resonance graft admission writer contract opened live_admission_enabled")

	openedSourcePreflightPath := filepath.Join(dir, "opened_source_preflight.json")
	writeWeightedAdmissionResonanceGraftAdmissionWriterContractFixture(t, openedSourcePreflightPath)
	writeWeightedReadinessFixture(t, openedSourcePreflightPath, stringsReplaceFirst(readText(t, openedSourcePreflightPath), `"source_writer_preflight_live_admission_enabled": false`, `"source_writer_preflight_live_admission_enabled": true`))
	requireBoundaryAssertError(t, runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionWriterContractAssert([]string{openedSourcePreflightPath}), "weighted admission resonance graft admission writer contract opened source_writer_preflight_live_admission_enabled")

	missingPathField := filepath.Join(dir, "missing_path.json")
	writeWeightedAdmissionResonanceGraftAdmissionWriterContractFixture(t, missingPathField)
	writerInventoryReport := filepath.Join(dir, "srcinv.json")
	writeWeightedReadinessFixture(t, missingPathField, stringsReplaceFirst(readText(t, missingPathField), `"source_report": "`+writerInventoryReport+`"`, `"source_report": " "`))
	requireBoundaryAssertError(t, runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionWriterContractAssert([]string{missingPathField}), "weighted admission resonance graft admission writer contract source_report missing")

	badSourcePath := filepath.Join(dir, "bad_source.json")
	writeWeightedAdmissionResonanceGraftAdmissionWriterContractFixture(t, badSourcePath)
	writeWeightedReadinessFixture(t, badSourcePath, stringsReplaceFirst(readText(t, badSourcePath), `"source_schema": "arianna.live_route_weighted_admission_resonance_graft_admission_writer_inventory.v1"`, `"source_schema": "arianna.live_route_weighted_admission_resonance_graft_admission_writer_inventory.v0"`))
	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionWriterContractAssert([]string{badSourcePath}),
		`weighted admission resonance graft admission writer contract source_schema mismatch: got "arianna.live_route_weighted_admission_resonance_graft_admission_writer_inventory.v0" want "`+admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionWriterInventorySchema+`"`,
	)

	badSourceInventoryPath := filepath.Join(dir, "bad_source_inventory.json")
	writeWeightedAdmissionResonanceGraftAdmissionWriterContractFixture(t, badSourceInventoryPath)
	writeWeightedReadinessFixture(t, badSourceInventoryPath, stringsReplaceFirst(readText(t, badSourceInventoryPath), `"source_writer_inventory_kind": "shadow_graft_admission_writer_inventory"`, `"source_writer_inventory_kind": "writer_inventory"`))
	requireBoundaryAssertError(t, runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionWriterContractAssert([]string{badSourceInventoryPath}), "weighted admission resonance graft admission writer contract source writer inventory shape mismatch")

	badSourcePreflightPath := filepath.Join(dir, "bad_source_preflight.json")
	writeWeightedAdmissionResonanceGraftAdmissionWriterContractFixture(t, badSourcePreflightPath)
	writeWeightedReadinessFixture(t, badSourcePreflightPath, stringsReplaceFirst(readText(t, badSourcePreflightPath), `"source_writer_preflight_kind": "shadow_graft_admission_writer_preflight"`, `"source_writer_preflight_kind": "writer_preflight"`))
	requireBoundaryAssertError(t, runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionWriterContractAssert([]string{badSourcePreflightPath}), "weighted admission resonance graft admission writer contract source writer preflight shape mismatch")

	badWriterContractHashPath := filepath.Join(dir, "bad_writer_contract_hash.json")
	writeWeightedAdmissionResonanceGraftAdmissionWriterContractFixture(t, badWriterContractHashPath)
	writeWeightedReadinessFixture(t, badWriterContractHashPath, stringsReplaceFirst(readText(t, badWriterContractHashPath), `"writer_contract_hash": "weighted-resonance-graft-admission-writer-contract-`, `"writer_contract_hash": "weighted-resonance-graft-admission-writer-contract-bad`))
	requireBoundaryAssertError(t, runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionWriterContractAssert([]string{badWriterContractHashPath}), "weighted admission resonance graft admission writer contract writer_contract_hash mismatch")

	badBodyTargetPath := filepath.Join(dir, "bad_body_target.json")
	writeWeightedAdmissionResonanceGraftAdmissionWriterContractFixture(t, badBodyTargetPath)
	writeWeightedReadinessFixture(t, badBodyTargetPath, stringsReplaceFirst(readText(t, badBodyTargetPath), `"body_target": "none"`, `"body_target": "live"`))
	requireBoundaryAssertError(t, runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionWriterContractAssert([]string{badBodyTargetPath}), `weighted admission resonance graft admission writer contract body_target mismatch: got "live" want "none"`)
}

func writeWeightedAdmissionResonanceGraftAdmissionWriterContractFixture(t *testing.T, writerInventoryPath string) {
	t.Helper()
	dir := filepath.Dir(writerInventoryPath)
	sourceInventoryPath := filepath.Join(dir, "srcinv.json")
	writeWeightedAdmissionResonanceGraftAdmissionWriterInventoryFixture(t, sourceInventoryPath)
	if err := runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionWriterContract([]string{sourceInventoryPath, writerInventoryPath}); err != nil {
		t.Fatalf("write weighted admission resonance graft admission writer contract fixture: %v", err)
	}
}
