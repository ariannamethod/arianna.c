package main

import (
	"path/filepath"
	"strings"
	"testing"
)

func TestAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionLedgerVerificationAssert(t *testing.T) {
	dir := t.TempDir()

	requireBoundaryAssertError(t, runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionLedgerVerificationAssert(nil), "usage: --admission-live-route-weighted-admission-resonance-graft-admission-ledger-verification-assert REPORT")
	requireBoundaryAssertError(t, runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionLedgerVerificationAssert([]string{"ledger_verify.json", "extra"}), "usage: --admission-live-route-weighted-admission-resonance-graft-admission-ledger-verification-assert REPORT")
	requireBoundaryAssertError(t, runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionLedgerVerificationAssert([]string{"  "}), "weighted admission resonance graft admission ledger verification path missing")
	requireBoundaryAssertError(t, runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionLedgerVerificationAssert([]string{filepath.Join(dir, "missing.json")}), "weighted admission resonance graft admission ledger verification not written")

	emptyPath := filepath.Join(dir, "empty.json")
	writeWeightedReadinessFixture(t, emptyPath, "")
	requireBoundaryAssertError(t, runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionLedgerVerificationAssert([]string{emptyPath}), "weighted admission resonance graft admission ledger verification not written")

	invalidPath := filepath.Join(dir, "invalid.json")
	writeWeightedReadinessFixture(t, invalidPath, "{")
	err := runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionLedgerVerificationAssert([]string{invalidPath})
	if err == nil || !strings.HasPrefix(err.Error(), "weighted admission resonance graft admission ledger verification JSON invalid:") {
		t.Fatalf("expected invalid JSON error, got %v", err)
	}

	missingSchemaPath := filepath.Join(dir, "missing_schema.json")
	writeWeightedAdmissionResonanceGraftAdmissionLedgerVerificationFixture(t, missingSchemaPath)
	writeWeightedReadinessFixture(t, missingSchemaPath, stringsReplaceFirst(readText(t, missingSchemaPath), `"schema": "arianna.live_route_weighted_admission_resonance_graft_admission_ledger_verification.v1",`, ""))
	requireBoundaryAssertError(t, runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionLedgerVerificationAssert([]string{missingSchemaPath}), "weighted admission resonance graft admission ledger verification schema missing")

	badSchemaPath := filepath.Join(dir, "bad_schema.json")
	writeWeightedAdmissionResonanceGraftAdmissionLedgerVerificationFixture(t, badSchemaPath)
	writeWeightedReadinessFixture(t, badSchemaPath, stringsReplaceFirst(readText(t, badSchemaPath), `"schema": "arianna.live_route_weighted_admission_resonance_graft_admission_ledger_verification.v1"`, `"schema": "arianna.live_route_weighted_admission_resonance_graft_admission_ledger_verification.v0"`))
	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionLedgerVerificationAssert([]string{badSchemaPath}),
		`weighted admission resonance graft admission ledger verification schema mismatch: got "arianna.live_route_weighted_admission_resonance_graft_admission_ledger_verification.v0" want "`+admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionLedgerVerificationSchema+`"`,
	)

	validPath := filepath.Join(dir, "valid.json")
	writeWeightedAdmissionResonanceGraftAdmissionLedgerVerificationFixture(t, validPath)
	if err := runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionLedgerVerificationAssert([]string{validPath}); err != nil {
		t.Fatalf("valid weighted admission resonance graft admission ledger verification rejected: %v", err)
	}

	badStatusPath := filepath.Join(dir, "bad_status.json")
	writeWeightedAdmissionResonanceGraftAdmissionLedgerVerificationFixture(t, badStatusPath)
	writeWeightedReadinessFixture(t, badStatusPath, stringsReplaceFirst(readText(t, badStatusPath), `"status": "shadow_graft_admission_ledger_verification_blocked_dry_run"`, `"status": "open"`))
	requireBoundaryAssertError(t, runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionLedgerVerificationAssert([]string{badStatusPath}), `weighted admission resonance graft admission ledger verification status mismatch: got "open" want "shadow_graft_admission_ledger_verification_blocked_dry_run"`)

	notReadyPath := filepath.Join(dir, "not_ready.json")
	writeWeightedAdmissionResonanceGraftAdmissionLedgerVerificationFixture(t, notReadyPath)
	writeWeightedReadinessFixture(t, notReadyPath, stringsReplaceFirst(readText(t, notReadyPath), `"weighted_admission_resonance_graft_admission_ledger_verification_ready": true`, `"weighted_admission_resonance_graft_admission_ledger_verification_ready": false`))
	requireBoundaryAssertError(t, runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionLedgerVerificationAssert([]string{notReadyPath}), "weighted admission resonance graft admission ledger verification weighted_admission_resonance_graft_admission_ledger_verification_ready not ready")

	badActionPath := filepath.Join(dir, "bad_action.json")
	writeWeightedAdmissionResonanceGraftAdmissionLedgerVerificationFixture(t, badActionPath)
	writeWeightedReadinessFixture(t, badActionPath, stringsReplaceFirst(readText(t, badActionPath), `"action": "block_weighted_resonance_shadow_graft_admission_ledger_persistence_blocked_dry_run"`, `"action": "open"`))
	requireBoundaryAssertError(t, runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionLedgerVerificationAssert([]string{badActionPath}), `weighted admission resonance graft admission ledger verification action mismatch: got "open" want "block_weighted_resonance_shadow_graft_admission_ledger_persistence_blocked_dry_run"`)

	badVerificationActionPath := filepath.Join(dir, "bad_verification_action.json")
	writeWeightedAdmissionResonanceGraftAdmissionLedgerVerificationFixture(t, badVerificationActionPath)
	writeWeightedReadinessFixture(t, badVerificationActionPath, stringsReplaceFirst(readText(t, badVerificationActionPath), `"ledger_verification_action": "reject_blocked_ledger_persistence"`, `"ledger_verification_action": "verify_persisted_admission_ledger_receipt"`))
	requireBoundaryAssertError(t, runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionLedgerVerificationAssert([]string{badVerificationActionPath}), "weighted admission resonance graft admission ledger verification shape mismatch")

	openedVerifiedPath := filepath.Join(dir, "opened_verified.json")
	writeWeightedAdmissionResonanceGraftAdmissionLedgerVerificationFixture(t, openedVerifiedPath)
	writeWeightedReadinessFixture(t, openedVerifiedPath, stringsReplaceFirst(readText(t, openedVerifiedPath), `"ledger_verification_receipt_verified": false`, `"ledger_verification_receipt_verified": true`))
	requireBoundaryAssertError(t, runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionLedgerVerificationAssert([]string{openedVerifiedPath}), "weighted admission resonance graft admission ledger verification opened ledger_verification_receipt_verified")

	openedPath := filepath.Join(dir, "opened.json")
	writeWeightedAdmissionResonanceGraftAdmissionLedgerVerificationFixture(t, openedPath)
	writeWeightedReadinessFixture(t, openedPath, stringsReplaceFirst(readText(t, openedPath), `"live_admission_enabled": false`, `"live_admission_enabled": true`))
	requireBoundaryAssertError(t, runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionLedgerVerificationAssert([]string{openedPath}), "weighted admission resonance graft admission ledger verification opened live_admission_enabled")

	badSourcePath := filepath.Join(dir, "bad_source.json")
	writeWeightedAdmissionResonanceGraftAdmissionLedgerVerificationFixture(t, badSourcePath)
	writeWeightedReadinessFixture(t, badSourcePath, stringsReplaceFirst(readText(t, badSourcePath), `"source_schema": "arianna.live_route_weighted_admission_resonance_graft_admission_ledger_persistence.v1"`, `"source_schema": "arianna.live_route_weighted_admission_resonance_graft_admission_ledger_persistence.v0"`))
	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionLedgerVerificationAssert([]string{badSourcePath}),
		`weighted admission resonance graft admission ledger verification source_schema mismatch: got "arianna.live_route_weighted_admission_resonance_graft_admission_ledger_persistence.v0" want "`+admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionLedgerPersistenceSchema+`"`,
	)

	badSourcePersistenceShapePath := filepath.Join(dir, "bad_source_persistence_shape.json")
	writeWeightedAdmissionResonanceGraftAdmissionLedgerVerificationFixture(t, badSourcePersistenceShapePath)
	writeWeightedReadinessFixture(t, badSourcePersistenceShapePath, stringsReplaceFirst(readText(t, badSourcePersistenceShapePath), `"source_ledger_persistence_action": "reject_blocked_ledger_implementation"`, `"source_ledger_persistence_action": "append"`))
	requireBoundaryAssertError(t, runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionLedgerVerificationAssert([]string{badSourcePersistenceShapePath}), "weighted admission resonance graft admission ledger verification source ledger persistence shape mismatch")

	badLedgerVerificationHashPath := filepath.Join(dir, "bad_ledger_verification_hash.json")
	writeWeightedAdmissionResonanceGraftAdmissionLedgerVerificationFixture(t, badLedgerVerificationHashPath)
	writeWeightedReadinessFixture(t, badLedgerVerificationHashPath, stringsReplaceFirst(readText(t, badLedgerVerificationHashPath), `"ledger_verification_hash": "weighted-resonance-graft-admission-ledger-verification-`, `"ledger_verification_hash": "weighted-resonance-graft-admission-ledger-verification-bad`))
	requireBoundaryAssertError(t, runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionLedgerVerificationAssert([]string{badLedgerVerificationHashPath}), "weighted admission resonance graft admission ledger verification ledger_verification_hash mismatch")

	badBodyTargetPath := filepath.Join(dir, "bad_body_target.json")
	writeWeightedAdmissionResonanceGraftAdmissionLedgerVerificationFixture(t, badBodyTargetPath)
	writeWeightedReadinessFixture(t, badBodyTargetPath, stringsReplaceFirst(readText(t, badBodyTargetPath), `"body_target": "none"`, `"body_target": "live"`))
	requireBoundaryAssertError(t, runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionLedgerVerificationAssert([]string{badBodyTargetPath}), `weighted admission resonance graft admission ledger verification body_target mismatch: got "live" want "none"`)
}

func writeWeightedAdmissionResonanceGraftAdmissionLedgerVerificationFixture(t *testing.T, ledgerVerificationPath string) {
	t.Helper()
	dir := filepath.Dir(ledgerVerificationPath)
	sourcePersistencePath := filepath.Join(dir, "srcledgerpersist.json")
	writeWeightedAdmissionResonanceGraftAdmissionLedgerPersistenceFixture(t, sourcePersistencePath)
	if err := runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionLedgerVerification([]string{sourcePersistencePath, ledgerVerificationPath}); err != nil {
		t.Fatalf("write weighted admission resonance graft admission ledger verification fixture: %v", err)
	}
}
