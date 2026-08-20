package main

import (
	"path/filepath"
	"strings"
	"testing"
)

func TestAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionLedgerPersistenceAssert(t *testing.T) {
	dir := t.TempDir()

	requireBoundaryAssertError(t, runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionLedgerPersistenceAssert(nil), "usage: --admission-live-route-weighted-admission-resonance-graft-admission-ledger-persistence-assert REPORT")
	requireBoundaryAssertError(t, runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionLedgerPersistenceAssert([]string{"ledger_persist.json", "extra"}), "usage: --admission-live-route-weighted-admission-resonance-graft-admission-ledger-persistence-assert REPORT")
	requireBoundaryAssertError(t, runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionLedgerPersistenceAssert([]string{"  "}), "weighted admission resonance graft admission ledger persistence path missing")
	requireBoundaryAssertError(t, runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionLedgerPersistenceAssert([]string{filepath.Join(dir, "missing.json")}), "weighted admission resonance graft admission ledger persistence not written")

	emptyPath := filepath.Join(dir, "empty.json")
	writeWeightedReadinessFixture(t, emptyPath, "")
	requireBoundaryAssertError(t, runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionLedgerPersistenceAssert([]string{emptyPath}), "weighted admission resonance graft admission ledger persistence not written")

	invalidPath := filepath.Join(dir, "invalid.json")
	writeWeightedReadinessFixture(t, invalidPath, "{")
	err := runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionLedgerPersistenceAssert([]string{invalidPath})
	if err == nil || !strings.HasPrefix(err.Error(), "weighted admission resonance graft admission ledger persistence JSON invalid:") {
		t.Fatalf("expected invalid JSON error, got %v", err)
	}

	missingSchemaPath := filepath.Join(dir, "missing_schema.json")
	writeWeightedAdmissionResonanceGraftAdmissionLedgerPersistenceFixture(t, missingSchemaPath)
	writeWeightedReadinessFixture(t, missingSchemaPath, stringsReplaceFirst(readText(t, missingSchemaPath), `"schema": "arianna.live_route_weighted_admission_resonance_graft_admission_ledger_persistence.v1",`, ""))
	requireBoundaryAssertError(t, runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionLedgerPersistenceAssert([]string{missingSchemaPath}), "weighted admission resonance graft admission ledger persistence schema missing")

	badSchemaPath := filepath.Join(dir, "bad_schema.json")
	writeWeightedAdmissionResonanceGraftAdmissionLedgerPersistenceFixture(t, badSchemaPath)
	writeWeightedReadinessFixture(t, badSchemaPath, stringsReplaceFirst(readText(t, badSchemaPath), `"schema": "arianna.live_route_weighted_admission_resonance_graft_admission_ledger_persistence.v1"`, `"schema": "arianna.live_route_weighted_admission_resonance_graft_admission_ledger_persistence.v0"`))
	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionLedgerPersistenceAssert([]string{badSchemaPath}),
		`weighted admission resonance graft admission ledger persistence schema mismatch: got "arianna.live_route_weighted_admission_resonance_graft_admission_ledger_persistence.v0" want "`+admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionLedgerPersistenceSchema+`"`,
	)

	validPath := filepath.Join(dir, "valid.json")
	writeWeightedAdmissionResonanceGraftAdmissionLedgerPersistenceFixture(t, validPath)
	if err := runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionLedgerPersistenceAssert([]string{validPath}); err != nil {
		t.Fatalf("valid weighted admission resonance graft admission ledger persistence rejected: %v", err)
	}

	badStatusPath := filepath.Join(dir, "bad_status.json")
	writeWeightedAdmissionResonanceGraftAdmissionLedgerPersistenceFixture(t, badStatusPath)
	writeWeightedReadinessFixture(t, badStatusPath, stringsReplaceFirst(readText(t, badStatusPath), `"status": "shadow_graft_admission_ledger_persistence_blocked_dry_run"`, `"status": "open"`))
	requireBoundaryAssertError(t, runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionLedgerPersistenceAssert([]string{badStatusPath}), `weighted admission resonance graft admission ledger persistence status mismatch: got "open" want "shadow_graft_admission_ledger_persistence_blocked_dry_run"`)

	notReadyPath := filepath.Join(dir, "not_ready.json")
	writeWeightedAdmissionResonanceGraftAdmissionLedgerPersistenceFixture(t, notReadyPath)
	writeWeightedReadinessFixture(t, notReadyPath, stringsReplaceFirst(readText(t, notReadyPath), `"weighted_admission_resonance_graft_admission_ledger_persistence_ready": true`, `"weighted_admission_resonance_graft_admission_ledger_persistence_ready": false`))
	requireBoundaryAssertError(t, runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionLedgerPersistenceAssert([]string{notReadyPath}), "weighted admission resonance graft admission ledger persistence weighted_admission_resonance_graft_admission_ledger_persistence_ready not ready")

	badActionPath := filepath.Join(dir, "bad_action.json")
	writeWeightedAdmissionResonanceGraftAdmissionLedgerPersistenceFixture(t, badActionPath)
	writeWeightedReadinessFixture(t, badActionPath, stringsReplaceFirst(readText(t, badActionPath), `"action": "block_weighted_resonance_shadow_graft_admission_ledger_implementation_blocked_dry_run"`, `"action": "open"`))
	requireBoundaryAssertError(t, runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionLedgerPersistenceAssert([]string{badActionPath}), `weighted admission resonance graft admission ledger persistence action mismatch: got "open" want "block_weighted_resonance_shadow_graft_admission_ledger_implementation_blocked_dry_run"`)

	badPersistenceActionPath := filepath.Join(dir, "bad_persistence_action.json")
	writeWeightedAdmissionResonanceGraftAdmissionLedgerPersistenceFixture(t, badPersistenceActionPath)
	writeWeightedReadinessFixture(t, badPersistenceActionPath, stringsReplaceFirst(readText(t, badPersistenceActionPath), `"ledger_persistence_action": "reject_blocked_ledger_implementation"`, `"ledger_persistence_action": "append_admission_ledger_receipt"`))
	requireBoundaryAssertError(t, runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionLedgerPersistenceAssert([]string{badPersistenceActionPath}), "weighted admission resonance graft admission ledger persistence shape mismatch")

	openedPersistedPath := filepath.Join(dir, "opened_persisted.json")
	writeWeightedAdmissionResonanceGraftAdmissionLedgerPersistenceFixture(t, openedPersistedPath)
	writeWeightedReadinessFixture(t, openedPersistedPath, stringsReplaceFirst(readText(t, openedPersistedPath), `"ledger_persistence_receipt_persisted": false`, `"ledger_persistence_receipt_persisted": true`))
	requireBoundaryAssertError(t, runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionLedgerPersistenceAssert([]string{openedPersistedPath}), "weighted admission resonance graft admission ledger persistence opened ledger_persistence_receipt_persisted")

	openedPath := filepath.Join(dir, "opened.json")
	writeWeightedAdmissionResonanceGraftAdmissionLedgerPersistenceFixture(t, openedPath)
	writeWeightedReadinessFixture(t, openedPath, stringsReplaceFirst(readText(t, openedPath), `"live_admission_enabled": false`, `"live_admission_enabled": true`))
	requireBoundaryAssertError(t, runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionLedgerPersistenceAssert([]string{openedPath}), "weighted admission resonance graft admission ledger persistence opened live_admission_enabled")

	badSourcePath := filepath.Join(dir, "bad_source.json")
	writeWeightedAdmissionResonanceGraftAdmissionLedgerPersistenceFixture(t, badSourcePath)
	writeWeightedReadinessFixture(t, badSourcePath, stringsReplaceFirst(readText(t, badSourcePath), `"source_schema": "arianna.live_route_weighted_admission_resonance_graft_admission_ledger_implementation.v1"`, `"source_schema": "arianna.live_route_weighted_admission_resonance_graft_admission_ledger_implementation.v0"`))
	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionLedgerPersistenceAssert([]string{badSourcePath}),
		`weighted admission resonance graft admission ledger persistence source_schema mismatch: got "arianna.live_route_weighted_admission_resonance_graft_admission_ledger_implementation.v0" want "`+admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionLedgerImplementationSchema+`"`,
	)

	badSourceImplementationShapePath := filepath.Join(dir, "bad_source_implementation_shape.json")
	writeWeightedAdmissionResonanceGraftAdmissionLedgerPersistenceFixture(t, badSourceImplementationShapePath)
	writeWeightedReadinessFixture(t, badSourceImplementationShapePath, stringsReplaceFirst(readText(t, badSourceImplementationShapePath), `"source_ledger_implementation_action": "reject_blocked_admission_ledger"`, `"source_ledger_implementation_action": "append"`))
	requireBoundaryAssertError(t, runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionLedgerPersistenceAssert([]string{badSourceImplementationShapePath}), "weighted admission resonance graft admission ledger persistence source ledger implementation shape mismatch")

	badLedgerPersistenceHashPath := filepath.Join(dir, "bad_ledger_persistence_hash.json")
	writeWeightedAdmissionResonanceGraftAdmissionLedgerPersistenceFixture(t, badLedgerPersistenceHashPath)
	writeWeightedReadinessFixture(t, badLedgerPersistenceHashPath, stringsReplaceFirst(readText(t, badLedgerPersistenceHashPath), `"ledger_persistence_hash": "weighted-resonance-graft-admission-ledger-persistence-`, `"ledger_persistence_hash": "weighted-resonance-graft-admission-ledger-persistence-bad`))
	requireBoundaryAssertError(t, runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionLedgerPersistenceAssert([]string{badLedgerPersistenceHashPath}), "weighted admission resonance graft admission ledger persistence ledger_persistence_hash mismatch")

	badBodyTargetPath := filepath.Join(dir, "bad_body_target.json")
	writeWeightedAdmissionResonanceGraftAdmissionLedgerPersistenceFixture(t, badBodyTargetPath)
	writeWeightedReadinessFixture(t, badBodyTargetPath, stringsReplaceFirst(readText(t, badBodyTargetPath), `"body_target": "none"`, `"body_target": "live"`))
	requireBoundaryAssertError(t, runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionLedgerPersistenceAssert([]string{badBodyTargetPath}), `weighted admission resonance graft admission ledger persistence body_target mismatch: got "live" want "none"`)
}

func writeWeightedAdmissionResonanceGraftAdmissionLedgerPersistenceFixture(t *testing.T, ledgerPersistencePath string) {
	t.Helper()
	dir := filepath.Dir(ledgerPersistencePath)
	sourceImplementationPath := filepath.Join(dir, "srcledgerimpl.json")
	writeWeightedAdmissionResonanceGraftAdmissionLedgerImplementationFixture(t, sourceImplementationPath)
	if err := runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionLedgerPersistence([]string{sourceImplementationPath, ledgerPersistencePath}); err != nil {
		t.Fatalf("write weighted admission resonance graft admission ledger persistence fixture: %v", err)
	}
}
