package main

import (
	"path/filepath"
	"strings"
	"testing"
)

func TestAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionLedgerAssert(t *testing.T) {
	dir := t.TempDir()

	requireBoundaryAssertError(t, runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionLedgerAssert(nil), "usage: --admission-live-route-weighted-admission-resonance-graft-admission-ledger-assert REPORT")
	requireBoundaryAssertError(t, runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionLedgerAssert([]string{"ledger.json", "extra"}), "usage: --admission-live-route-weighted-admission-resonance-graft-admission-ledger-assert REPORT")
	requireBoundaryAssertError(t, runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionLedgerAssert([]string{"  "}), "weighted admission resonance graft admission ledger path missing")
	requireBoundaryAssertError(t, runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionLedgerAssert([]string{filepath.Join(dir, "missing.json")}), "weighted admission resonance graft admission ledger not written")

	emptyPath := filepath.Join(dir, "empty.json")
	writeWeightedReadinessFixture(t, emptyPath, "")
	requireBoundaryAssertError(t, runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionLedgerAssert([]string{emptyPath}), "weighted admission resonance graft admission ledger not written")

	invalidPath := filepath.Join(dir, "invalid.json")
	writeWeightedReadinessFixture(t, invalidPath, "{")
	err := runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionLedgerAssert([]string{invalidPath})
	if err == nil || !strings.HasPrefix(err.Error(), "weighted admission resonance graft admission ledger JSON invalid:") {
		t.Fatalf("expected invalid JSON error, got %v", err)
	}

	missingSchemaPath := filepath.Join(dir, "missing_schema.json")
	writeWeightedAdmissionResonanceGraftAdmissionLedgerFixture(t, missingSchemaPath)
	writeWeightedReadinessFixture(t, missingSchemaPath, stringsReplaceFirst(readText(t, missingSchemaPath), `"schema": "arianna.live_route_weighted_admission_resonance_graft_admission_ledger.v1",`, ""))
	requireBoundaryAssertError(t, runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionLedgerAssert([]string{missingSchemaPath}), "weighted admission resonance graft admission ledger schema missing")

	badSchemaPath := filepath.Join(dir, "bad_schema.json")
	writeWeightedAdmissionResonanceGraftAdmissionLedgerFixture(t, badSchemaPath)
	writeWeightedReadinessFixture(t, badSchemaPath, stringsReplaceFirst(readText(t, badSchemaPath), `"schema": "arianna.live_route_weighted_admission_resonance_graft_admission_ledger.v1"`, `"schema": "arianna.live_route_weighted_admission_resonance_graft_admission_ledger.v0"`))
	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionLedgerAssert([]string{badSchemaPath}),
		`weighted admission resonance graft admission ledger schema mismatch: got "arianna.live_route_weighted_admission_resonance_graft_admission_ledger.v0" want "`+admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionLedgerSchema+`"`,
	)

	validPath := filepath.Join(dir, "valid.json")
	writeWeightedAdmissionResonanceGraftAdmissionLedgerFixture(t, validPath)
	if err := runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionLedgerAssert([]string{validPath}); err != nil {
		t.Fatalf("valid weighted admission resonance graft admission ledger rejected: %v", err)
	}

	badStatusPath := filepath.Join(dir, "bad_status.json")
	writeWeightedAdmissionResonanceGraftAdmissionLedgerFixture(t, badStatusPath)
	writeWeightedReadinessFixture(t, badStatusPath, stringsReplaceFirst(readText(t, badStatusPath), `"status": "shadow_graft_admission_ledger_blocked_dry_run"`, `"status": "open"`))
	requireBoundaryAssertError(t, runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionLedgerAssert([]string{badStatusPath}), `weighted admission resonance graft admission ledger status mismatch: got "open" want "shadow_graft_admission_ledger_blocked_dry_run"`)

	notReadyPath := filepath.Join(dir, "not_ready.json")
	writeWeightedAdmissionResonanceGraftAdmissionLedgerFixture(t, notReadyPath)
	writeWeightedReadinessFixture(t, notReadyPath, stringsReplaceFirst(readText(t, notReadyPath), `"weighted_admission_resonance_graft_admission_ledger_ready": true`, `"weighted_admission_resonance_graft_admission_ledger_ready": false`))
	requireBoundaryAssertError(t, runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionLedgerAssert([]string{notReadyPath}), "weighted admission resonance graft admission ledger weighted_admission_resonance_graft_admission_ledger_ready not ready")

	badActionPath := filepath.Join(dir, "bad_action.json")
	writeWeightedAdmissionResonanceGraftAdmissionLedgerFixture(t, badActionPath)
	writeWeightedReadinessFixture(t, badActionPath, stringsReplaceFirst(readText(t, badActionPath), `"action": "block_weighted_resonance_shadow_graft_admission_writer_contract_blocked_dry_run"`, `"action": "append_ledger"`))
	requireBoundaryAssertError(t, runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionLedgerAssert([]string{badActionPath}), `weighted admission resonance graft admission ledger action mismatch: got "append_ledger" want "block_weighted_resonance_shadow_graft_admission_writer_contract_blocked_dry_run"`)

	badLedgerActionPath := filepath.Join(dir, "bad_ledger_action.json")
	writeWeightedAdmissionResonanceGraftAdmissionLedgerFixture(t, badLedgerActionPath)
	writeWeightedReadinessFixture(t, badLedgerActionPath, stringsReplaceFirst(readText(t, badLedgerActionPath), `"ledger_action": "reject_blocked_writer_contract"`, `"ledger_action": "append_admission_ledger_receipt"`))
	requireBoundaryAssertError(t, runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionLedgerAssert([]string{badLedgerActionPath}), "weighted admission resonance graft admission ledger ledger state/action mismatch")

	namedContractPath := filepath.Join(dir, "named_contract.json")
	writeWeightedAdmissionResonanceGraftAdmissionLedgerFixture(t, namedContractPath)
	writeWeightedReadinessFixture(t, namedContractPath, stringsReplaceFirst(readText(t, namedContractPath), `"writer_contract": "none"`, `"writer_contract": "live_admission_writer.v1"`))
	requireBoundaryAssertError(t, runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionLedgerAssert([]string{namedContractPath}), "weighted admission resonance graft admission ledger contracts unexpectedly named")

	namedContractShapePath := filepath.Join(dir, "named_contract_shape.json")
	writeWeightedAdmissionResonanceGraftAdmissionLedgerFixture(t, namedContractShapePath)
	writeWeightedReadinessFixture(t, namedContractShapePath, stringsReplaceFirst(readText(t, namedContractShapePath), `"writer_contract_shape": "none"`, `"writer_contract_shape": "live_admission_writer_shape.v1"`))
	requireBoundaryAssertError(t, runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionLedgerAssert([]string{namedContractShapePath}), "weighted admission resonance graft admission ledger contract shapes unexpectedly named")

	openedLedgerAppendPath := filepath.Join(dir, "opened_ledger_append.json")
	writeWeightedAdmissionResonanceGraftAdmissionLedgerFixture(t, openedLedgerAppendPath)
	writeWeightedReadinessFixture(t, openedLedgerAppendPath, stringsReplaceFirst(readText(t, openedLedgerAppendPath), `"ledger_append_allowed": false`, `"ledger_append_allowed": true`))
	requireBoundaryAssertError(t, runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionLedgerAssert([]string{openedLedgerAppendPath}), "weighted admission resonance graft admission ledger opened ledger_append_allowed")

	openedPath := filepath.Join(dir, "opened.json")
	writeWeightedAdmissionResonanceGraftAdmissionLedgerFixture(t, openedPath)
	writeWeightedReadinessFixture(t, openedPath, stringsReplaceFirst(readText(t, openedPath), `"live_admission_enabled": false`, `"live_admission_enabled": true`))
	requireBoundaryAssertError(t, runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionLedgerAssert([]string{openedPath}), "weighted admission resonance graft admission ledger opened live_admission_enabled")

	openedSourceContractPath := filepath.Join(dir, "opened_source_contract.json")
	writeWeightedAdmissionResonanceGraftAdmissionLedgerFixture(t, openedSourceContractPath)
	writeWeightedReadinessFixture(t, openedSourceContractPath, stringsReplaceFirst(readText(t, openedSourceContractPath), `"source_writer_contract_contracts_ready": false`, `"source_writer_contract_contracts_ready": true`))
	requireBoundaryAssertError(t, runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionLedgerAssert([]string{openedSourceContractPath}), "weighted admission resonance graft admission ledger opened source_writer_contract_contracts_ready")

	missingPathField := filepath.Join(dir, "missing_path.json")
	writeWeightedAdmissionResonanceGraftAdmissionLedgerFixture(t, missingPathField)
	writerContractReport := filepath.Join(dir, "srcwc.json")
	writeWeightedReadinessFixture(t, missingPathField, stringsReplaceFirst(readText(t, missingPathField), `"source_report": "`+writerContractReport+`"`, `"source_report": " "`))
	requireBoundaryAssertError(t, runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionLedgerAssert([]string{missingPathField}), "weighted admission resonance graft admission ledger source_report missing")

	badSourcePath := filepath.Join(dir, "bad_source.json")
	writeWeightedAdmissionResonanceGraftAdmissionLedgerFixture(t, badSourcePath)
	writeWeightedReadinessFixture(t, badSourcePath, stringsReplaceFirst(readText(t, badSourcePath), `"source_schema": "arianna.live_route_weighted_admission_resonance_graft_admission_writer_contract.v1"`, `"source_schema": "arianna.live_route_weighted_admission_resonance_graft_admission_writer_contract.v0"`))
	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionLedgerAssert([]string{badSourcePath}),
		`weighted admission resonance graft admission ledger source_schema mismatch: got "arianna.live_route_weighted_admission_resonance_graft_admission_writer_contract.v0" want "`+admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionWriterContractSchema+`"`,
	)

	badSourceContractPath := filepath.Join(dir, "bad_source_contract.json")
	writeWeightedAdmissionResonanceGraftAdmissionLedgerFixture(t, badSourceContractPath)
	writeWeightedReadinessFixture(t, badSourceContractPath, stringsReplaceFirst(readText(t, badSourceContractPath), `"source_writer_contract_kind": "shadow_graft_admission_writer_contract"`, `"source_writer_contract_kind": "writer_contract"`))
	requireBoundaryAssertError(t, runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionLedgerAssert([]string{badSourceContractPath}), "weighted admission resonance graft admission ledger source writer contract shape mismatch")

	badSourceInventoryPath := filepath.Join(dir, "bad_source_inventory.json")
	writeWeightedAdmissionResonanceGraftAdmissionLedgerFixture(t, badSourceInventoryPath)
	writeWeightedReadinessFixture(t, badSourceInventoryPath, stringsReplaceFirst(readText(t, badSourceInventoryPath), `"source_writer_inventory_kind": "shadow_graft_admission_writer_inventory"`, `"source_writer_inventory_kind": "writer_inventory"`))
	requireBoundaryAssertError(t, runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionLedgerAssert([]string{badSourceInventoryPath}), "weighted admission resonance graft admission ledger source writer inventory shape mismatch")

	badAdmissionLedgerHashPath := filepath.Join(dir, "bad_admission_ledger_hash.json")
	writeWeightedAdmissionResonanceGraftAdmissionLedgerFixture(t, badAdmissionLedgerHashPath)
	writeWeightedReadinessFixture(t, badAdmissionLedgerHashPath, stringsReplaceFirst(readText(t, badAdmissionLedgerHashPath), `"admission_ledger_hash": "weighted-resonance-graft-admission-ledger-`, `"admission_ledger_hash": "weighted-resonance-graft-admission-ledger-bad`))
	requireBoundaryAssertError(t, runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionLedgerAssert([]string{badAdmissionLedgerHashPath}), "weighted admission resonance graft admission ledger admission_ledger_hash mismatch")

	badSourceWriterContractHashPath := filepath.Join(dir, "bad_source_writer_contract_hash.json")
	writeWeightedAdmissionResonanceGraftAdmissionLedgerFixture(t, badSourceWriterContractHashPath)
	writeWeightedReadinessFixture(t, badSourceWriterContractHashPath, stringsReplaceFirst(readText(t, badSourceWriterContractHashPath), `"source_weighted_admission_resonance_graft_admission_writer_contract_hash": "weighted-resonance-graft-admission-writer-contract-`, `"source_weighted_admission_resonance_graft_admission_writer_contract_hash": "weighted-resonance-graft-admission-writer-contract-bad`))
	requireBoundaryAssertError(t, runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionLedgerAssert([]string{badSourceWriterContractHashPath}), "weighted admission resonance graft admission ledger admission_ledger_hash mismatch")

	badBodyTargetPath := filepath.Join(dir, "bad_body_target.json")
	writeWeightedAdmissionResonanceGraftAdmissionLedgerFixture(t, badBodyTargetPath)
	writeWeightedReadinessFixture(t, badBodyTargetPath, stringsReplaceFirst(readText(t, badBodyTargetPath), `"body_target": "none"`, `"body_target": "live"`))
	requireBoundaryAssertError(t, runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionLedgerAssert([]string{badBodyTargetPath}), `weighted admission resonance graft admission ledger body_target mismatch: got "live" want "none"`)
}

func writeWeightedAdmissionResonanceGraftAdmissionLedgerFixture(t *testing.T, ledgerPath string) {
	t.Helper()
	dir := filepath.Dir(ledgerPath)
	sourceWriterContractPath := filepath.Join(dir, "srcwc.json")
	writeWeightedAdmissionResonanceGraftAdmissionWriterContractFixture(t, sourceWriterContractPath)
	if err := runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionLedger([]string{sourceWriterContractPath, ledgerPath}); err != nil {
		t.Fatalf("write weighted admission resonance graft admission ledger fixture: %v", err)
	}
}
