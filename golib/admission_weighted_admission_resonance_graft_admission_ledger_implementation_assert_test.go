package main

import (
	"path/filepath"
	"strings"
	"testing"
)

func TestAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionLedgerImplementationAssert(t *testing.T) {
	dir := t.TempDir()

	requireBoundaryAssertError(t, runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionLedgerImplementationAssert(nil), "usage: --admission-live-route-weighted-admission-resonance-graft-admission-ledger-implementation-assert REPORT")
	requireBoundaryAssertError(t, runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionLedgerImplementationAssert([]string{"ledger_impl.json", "extra"}), "usage: --admission-live-route-weighted-admission-resonance-graft-admission-ledger-implementation-assert REPORT")
	requireBoundaryAssertError(t, runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionLedgerImplementationAssert([]string{"  "}), "weighted admission resonance graft admission ledger implementation path missing")
	requireBoundaryAssertError(t, runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionLedgerImplementationAssert([]string{filepath.Join(dir, "missing.json")}), "weighted admission resonance graft admission ledger implementation not written")

	emptyPath := filepath.Join(dir, "empty.json")
	writeWeightedReadinessFixture(t, emptyPath, "")
	requireBoundaryAssertError(t, runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionLedgerImplementationAssert([]string{emptyPath}), "weighted admission resonance graft admission ledger implementation not written")

	invalidPath := filepath.Join(dir, "invalid.json")
	writeWeightedReadinessFixture(t, invalidPath, "{")
	err := runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionLedgerImplementationAssert([]string{invalidPath})
	if err == nil || !strings.HasPrefix(err.Error(), "weighted admission resonance graft admission ledger implementation JSON invalid:") {
		t.Fatalf("expected invalid JSON error, got %v", err)
	}

	missingSchemaPath := filepath.Join(dir, "missing_schema.json")
	writeWeightedAdmissionResonanceGraftAdmissionLedgerImplementationFixture(t, missingSchemaPath)
	writeWeightedReadinessFixture(t, missingSchemaPath, stringsReplaceFirst(readText(t, missingSchemaPath), `"schema": "arianna.live_route_weighted_admission_resonance_graft_admission_ledger_implementation.v1",`, ""))
	requireBoundaryAssertError(t, runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionLedgerImplementationAssert([]string{missingSchemaPath}), "weighted admission resonance graft admission ledger implementation schema missing")

	badSchemaPath := filepath.Join(dir, "bad_schema.json")
	writeWeightedAdmissionResonanceGraftAdmissionLedgerImplementationFixture(t, badSchemaPath)
	writeWeightedReadinessFixture(t, badSchemaPath, stringsReplaceFirst(readText(t, badSchemaPath), `"schema": "arianna.live_route_weighted_admission_resonance_graft_admission_ledger_implementation.v1"`, `"schema": "arianna.live_route_weighted_admission_resonance_graft_admission_ledger_implementation.v0"`))
	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionLedgerImplementationAssert([]string{badSchemaPath}),
		`weighted admission resonance graft admission ledger implementation schema mismatch: got "arianna.live_route_weighted_admission_resonance_graft_admission_ledger_implementation.v0" want "`+admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionLedgerImplementationSchema+`"`,
	)

	validPath := filepath.Join(dir, "valid.json")
	writeWeightedAdmissionResonanceGraftAdmissionLedgerImplementationFixture(t, validPath)
	if err := runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionLedgerImplementationAssert([]string{validPath}); err != nil {
		t.Fatalf("valid weighted admission resonance graft admission ledger implementation rejected: %v", err)
	}

	badStatusPath := filepath.Join(dir, "bad_status.json")
	writeWeightedAdmissionResonanceGraftAdmissionLedgerImplementationFixture(t, badStatusPath)
	writeWeightedReadinessFixture(t, badStatusPath, stringsReplaceFirst(readText(t, badStatusPath), `"status": "shadow_graft_admission_ledger_implementation_blocked_dry_run"`, `"status": "open"`))
	requireBoundaryAssertError(t, runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionLedgerImplementationAssert([]string{badStatusPath}), `weighted admission resonance graft admission ledger implementation status mismatch: got "open" want "shadow_graft_admission_ledger_implementation_blocked_dry_run"`)

	notReadyPath := filepath.Join(dir, "not_ready.json")
	writeWeightedAdmissionResonanceGraftAdmissionLedgerImplementationFixture(t, notReadyPath)
	writeWeightedReadinessFixture(t, notReadyPath, stringsReplaceFirst(readText(t, notReadyPath), `"weighted_admission_resonance_graft_admission_ledger_implementation_ready": true`, `"weighted_admission_resonance_graft_admission_ledger_implementation_ready": false`))
	requireBoundaryAssertError(t, runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionLedgerImplementationAssert([]string{notReadyPath}), "weighted admission resonance graft admission ledger implementation weighted_admission_resonance_graft_admission_ledger_implementation_ready not ready")

	badActionPath := filepath.Join(dir, "bad_action.json")
	writeWeightedAdmissionResonanceGraftAdmissionLedgerImplementationFixture(t, badActionPath)
	writeWeightedReadinessFixture(t, badActionPath, stringsReplaceFirst(readText(t, badActionPath), `"action": "block_weighted_resonance_shadow_graft_admission_ledger_blocked_dry_run"`, `"action": "open"`))
	requireBoundaryAssertError(t, runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionLedgerImplementationAssert([]string{badActionPath}), `weighted admission resonance graft admission ledger implementation action mismatch: got "open" want "block_weighted_resonance_shadow_graft_admission_ledger_blocked_dry_run"`)

	badImplActionPath := filepath.Join(dir, "bad_impl_action.json")
	writeWeightedAdmissionResonanceGraftAdmissionLedgerImplementationFixture(t, badImplActionPath)
	writeWeightedReadinessFixture(t, badImplActionPath, stringsReplaceFirst(readText(t, badImplActionPath), `"ledger_implementation_action": "reject_blocked_admission_ledger"`, `"ledger_implementation_action": "append_admission_ledger_receipt"`))
	requireBoundaryAssertError(t, runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionLedgerImplementationAssert([]string{badImplActionPath}), "weighted admission resonance graft admission ledger implementation shape mismatch")

	openedAppendOnlyPath := filepath.Join(dir, "opened_append_only.json")
	writeWeightedAdmissionResonanceGraftAdmissionLedgerImplementationFixture(t, openedAppendOnlyPath)
	writeWeightedReadinessFixture(t, openedAppendOnlyPath, stringsReplaceFirst(readText(t, openedAppendOnlyPath), `"ledger_implementation_append_only": false`, `"ledger_implementation_append_only": true`))
	requireBoundaryAssertError(t, runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionLedgerImplementationAssert([]string{openedAppendOnlyPath}), "weighted admission resonance graft admission ledger implementation opened ledger_implementation_append_only")

	openedPath := filepath.Join(dir, "opened.json")
	writeWeightedAdmissionResonanceGraftAdmissionLedgerImplementationFixture(t, openedPath)
	writeWeightedReadinessFixture(t, openedPath, stringsReplaceFirst(readText(t, openedPath), `"live_admission_enabled": false`, `"live_admission_enabled": true`))
	requireBoundaryAssertError(t, runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionLedgerImplementationAssert([]string{openedPath}), "weighted admission resonance graft admission ledger implementation opened live_admission_enabled")

	badSourcePath := filepath.Join(dir, "bad_source.json")
	writeWeightedAdmissionResonanceGraftAdmissionLedgerImplementationFixture(t, badSourcePath)
	writeWeightedReadinessFixture(t, badSourcePath, stringsReplaceFirst(readText(t, badSourcePath), `"source_schema": "arianna.live_route_weighted_admission_resonance_graft_admission_ledger.v1"`, `"source_schema": "arianna.live_route_weighted_admission_resonance_graft_admission_ledger.v0"`))
	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionLedgerImplementationAssert([]string{badSourcePath}),
		`weighted admission resonance graft admission ledger implementation source_schema mismatch: got "arianna.live_route_weighted_admission_resonance_graft_admission_ledger.v0" want "`+admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionLedgerSchema+`"`,
	)

	badSourceLedgerShapePath := filepath.Join(dir, "bad_source_ledger_shape.json")
	writeWeightedAdmissionResonanceGraftAdmissionLedgerImplementationFixture(t, badSourceLedgerShapePath)
	writeWeightedReadinessFixture(t, badSourceLedgerShapePath, stringsReplaceFirst(readText(t, badSourceLedgerShapePath), `"source_admission_ledger_kind": "shadow_graft_admission_ledger"`, `"source_admission_ledger_kind": "ledger"`))
	requireBoundaryAssertError(t, runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionLedgerImplementationAssert([]string{badSourceLedgerShapePath}), "weighted admission resonance graft admission ledger implementation source admission ledger shape mismatch")

	badLedgerImplementationHashPath := filepath.Join(dir, "bad_ledger_implementation_hash.json")
	writeWeightedAdmissionResonanceGraftAdmissionLedgerImplementationFixture(t, badLedgerImplementationHashPath)
	writeWeightedReadinessFixture(t, badLedgerImplementationHashPath, stringsReplaceFirst(readText(t, badLedgerImplementationHashPath), `"ledger_implementation_hash": "weighted-resonance-graft-admission-ledger-implementation-`, `"ledger_implementation_hash": "weighted-resonance-graft-admission-ledger-implementation-bad`))
	requireBoundaryAssertError(t, runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionLedgerImplementationAssert([]string{badLedgerImplementationHashPath}), "weighted admission resonance graft admission ledger implementation ledger_implementation_hash mismatch")

	badBodyTargetPath := filepath.Join(dir, "bad_body_target.json")
	writeWeightedAdmissionResonanceGraftAdmissionLedgerImplementationFixture(t, badBodyTargetPath)
	writeWeightedReadinessFixture(t, badBodyTargetPath, stringsReplaceFirst(readText(t, badBodyTargetPath), `"body_target": "none"`, `"body_target": "live"`))
	requireBoundaryAssertError(t, runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionLedgerImplementationAssert([]string{badBodyTargetPath}), `weighted admission resonance graft admission ledger implementation body_target mismatch: got "live" want "none"`)
}

func writeWeightedAdmissionResonanceGraftAdmissionLedgerImplementationFixture(t *testing.T, ledgerImplementationPath string) {
	t.Helper()
	dir := filepath.Dir(ledgerImplementationPath)
	sourceLedgerPath := filepath.Join(dir, "srcledger.json")
	writeWeightedAdmissionResonanceGraftAdmissionLedgerFixture(t, sourceLedgerPath)
	if err := runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionLedgerImplementation([]string{sourceLedgerPath, ledgerImplementationPath}); err != nil {
		t.Fatalf("write weighted admission resonance graft admission ledger implementation fixture: %v", err)
	}
}
