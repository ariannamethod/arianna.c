package main

import (
	"path/filepath"
	"strings"
	"testing"
)

func TestAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionReadinessAssert(t *testing.T) {
	dir := t.TempDir()

	requireBoundaryAssertError(t, runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionReadinessAssert(nil), "usage: --admission-live-route-weighted-admission-resonance-graft-admission-readiness-assert REPORT")
	requireBoundaryAssertError(t, runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionReadinessAssert([]string{"readiness.json", "extra"}), "usage: --admission-live-route-weighted-admission-resonance-graft-admission-readiness-assert REPORT")
	requireBoundaryAssertError(t, runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionReadinessAssert([]string{"  "}), "weighted admission resonance graft admission readiness path missing")
	requireBoundaryAssertError(t, runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionReadinessAssert([]string{filepath.Join(dir, "missing.json")}), "weighted admission resonance graft admission readiness not written")

	emptyPath := filepath.Join(dir, "empty.json")
	writeWeightedReadinessFixture(t, emptyPath, "")
	requireBoundaryAssertError(t, runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionReadinessAssert([]string{emptyPath}), "weighted admission resonance graft admission readiness not written")

	invalidPath := filepath.Join(dir, "invalid.json")
	writeWeightedReadinessFixture(t, invalidPath, "{")
	err := runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionReadinessAssert([]string{invalidPath})
	if err == nil || !strings.HasPrefix(err.Error(), "weighted admission resonance graft admission readiness JSON invalid:") {
		t.Fatalf("expected invalid JSON error, got %v", err)
	}

	missingSchemaPath := filepath.Join(dir, "missing_schema.json")
	writeWeightedAdmissionResonanceGraftAdmissionReadinessFixture(t, missingSchemaPath)
	writeWeightedReadinessFixture(t, missingSchemaPath, stringsReplaceFirst(readText(t, missingSchemaPath), `"schema": "arianna.live_route_weighted_admission_resonance_graft_admission_readiness.v1",`, ""))
	requireBoundaryAssertError(t, runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionReadinessAssert([]string{missingSchemaPath}), "weighted admission resonance graft admission readiness schema missing")

	badSchemaPath := filepath.Join(dir, "bad_schema.json")
	writeWeightedAdmissionResonanceGraftAdmissionReadinessFixture(t, badSchemaPath)
	writeWeightedReadinessFixture(t, badSchemaPath, stringsReplaceFirst(readText(t, badSchemaPath), `"schema": "arianna.live_route_weighted_admission_resonance_graft_admission_readiness.v1"`, `"schema": "arianna.live_route_weighted_admission_resonance_graft_admission_readiness.v0"`))
	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionReadinessAssert([]string{badSchemaPath}),
		`weighted admission resonance graft admission readiness schema mismatch: got "arianna.live_route_weighted_admission_resonance_graft_admission_readiness.v0" want "`+admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionReadinessSchema+`"`,
	)

	validPath := filepath.Join(dir, "valid.json")
	writeWeightedAdmissionResonanceGraftAdmissionReadinessFixture(t, validPath)
	if err := runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionReadinessAssert([]string{validPath}); err != nil {
		t.Fatalf("valid weighted admission resonance graft admission readiness rejected: %v", err)
	}

	badStatusPath := filepath.Join(dir, "bad_status.json")
	writeWeightedAdmissionResonanceGraftAdmissionReadinessFixture(t, badStatusPath)
	writeWeightedReadinessFixture(t, badStatusPath, stringsReplaceFirst(readText(t, badStatusPath), `"status": "shadow_graft_admission_readiness_blocked_dry_run"`, `"status": "open"`))
	requireBoundaryAssertError(t, runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionReadinessAssert([]string{badStatusPath}), `weighted admission resonance graft admission readiness status mismatch: got "open" want "shadow_graft_admission_readiness_blocked_dry_run"`)

	notReadyPath := filepath.Join(dir, "not_ready.json")
	writeWeightedAdmissionResonanceGraftAdmissionReadinessFixture(t, notReadyPath)
	writeWeightedReadinessFixture(t, notReadyPath, stringsReplaceFirst(readText(t, notReadyPath), `"weighted_admission_resonance_graft_admission_readiness_ready": true`, `"weighted_admission_resonance_graft_admission_readiness_ready": false`))
	requireBoundaryAssertError(t, runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionReadinessAssert([]string{notReadyPath}), "weighted admission resonance graft admission readiness weighted_admission_resonance_graft_admission_readiness_ready not ready")

	badActionPath := filepath.Join(dir, "bad_action.json")
	writeWeightedAdmissionResonanceGraftAdmissionReadinessFixture(t, badActionPath)
	writeWeightedReadinessFixture(t, badActionPath, stringsReplaceFirst(readText(t, badActionPath), `"action": "block_weighted_resonance_shadow_graft_admission_ledger_verification_blocked_dry_run"`, `"action": "open"`))
	requireBoundaryAssertError(t, runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionReadinessAssert([]string{badActionPath}), `weighted admission resonance graft admission readiness action mismatch: got "open" want "block_weighted_resonance_shadow_graft_admission_ledger_verification_blocked_dry_run"`)

	badReadinessActionPath := filepath.Join(dir, "bad_readiness_action.json")
	writeWeightedAdmissionResonanceGraftAdmissionReadinessFixture(t, badReadinessActionPath)
	writeWeightedReadinessFixture(t, badReadinessActionPath, stringsReplaceFirst(readText(t, badReadinessActionPath), `"admission_readiness_action": "reject_blocked_ledger_verification"`, `"admission_readiness_action": "declare_verified_live_admission_readiness_dry_run"`))
	requireBoundaryAssertError(t, runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionReadinessAssert([]string{badReadinessActionPath}), "weighted admission resonance graft admission readiness shape mismatch")

	openedReadyPath := filepath.Join(dir, "opened_ready.json")
	writeWeightedAdmissionResonanceGraftAdmissionReadinessFixture(t, openedReadyPath)
	writeWeightedReadinessFixture(t, openedReadyPath, stringsReplaceFirst(readText(t, openedReadyPath), `"admission_readiness_ready": false`, `"admission_readiness_ready": true`))
	requireBoundaryAssertError(t, runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionReadinessAssert([]string{openedReadyPath}), "weighted admission resonance graft admission readiness opened admission_readiness_ready")

	openedPath := filepath.Join(dir, "opened.json")
	writeWeightedAdmissionResonanceGraftAdmissionReadinessFixture(t, openedPath)
	writeWeightedReadinessFixture(t, openedPath, stringsReplaceFirst(readText(t, openedPath), `"live_admission_enabled": false`, `"live_admission_enabled": true`))
	requireBoundaryAssertError(t, runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionReadinessAssert([]string{openedPath}), "weighted admission resonance graft admission readiness opened live_admission_enabled")

	badSourcePath := filepath.Join(dir, "bad_source.json")
	writeWeightedAdmissionResonanceGraftAdmissionReadinessFixture(t, badSourcePath)
	writeWeightedReadinessFixture(t, badSourcePath, stringsReplaceFirst(readText(t, badSourcePath), `"source_schema": "arianna.live_route_weighted_admission_resonance_graft_admission_ledger_verification.v1"`, `"source_schema": "arianna.live_route_weighted_admission_resonance_graft_admission_ledger_verification.v0"`))
	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionReadinessAssert([]string{badSourcePath}),
		`weighted admission resonance graft admission readiness source_schema mismatch: got "arianna.live_route_weighted_admission_resonance_graft_admission_ledger_verification.v0" want "`+admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionLedgerVerificationSchema+`"`,
	)

	badSourceVerificationShapePath := filepath.Join(dir, "bad_source_verification_shape.json")
	writeWeightedAdmissionResonanceGraftAdmissionReadinessFixture(t, badSourceVerificationShapePath)
	writeWeightedReadinessFixture(t, badSourceVerificationShapePath, stringsReplaceFirst(readText(t, badSourceVerificationShapePath), `"source_ledger_verification_action": "reject_blocked_ledger_persistence"`, `"source_ledger_verification_action": "verify"`))
	requireBoundaryAssertError(t, runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionReadinessAssert([]string{badSourceVerificationShapePath}), "weighted admission resonance graft admission readiness source ledger verification shape mismatch")

	badReadinessHashPath := filepath.Join(dir, "bad_readiness_hash.json")
	writeWeightedAdmissionResonanceGraftAdmissionReadinessFixture(t, badReadinessHashPath)
	writeWeightedReadinessFixture(t, badReadinessHashPath, stringsReplaceFirst(readText(t, badReadinessHashPath), `"admission_readiness_hash": "weighted-resonance-graft-admission-readiness-`, `"admission_readiness_hash": "weighted-resonance-graft-admission-readiness-bad`))
	requireBoundaryAssertError(t, runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionReadinessAssert([]string{badReadinessHashPath}), "weighted admission resonance graft admission readiness admission_readiness_hash mismatch")

	badBodyTargetPath := filepath.Join(dir, "bad_body_target.json")
	writeWeightedAdmissionResonanceGraftAdmissionReadinessFixture(t, badBodyTargetPath)
	writeWeightedReadinessFixture(t, badBodyTargetPath, stringsReplaceFirst(readText(t, badBodyTargetPath), `"body_target": "none"`, `"body_target": "live"`))
	requireBoundaryAssertError(t, runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionReadinessAssert([]string{badBodyTargetPath}), `weighted admission resonance graft admission readiness body_target mismatch: got "live" want "none"`)
}

func writeWeightedAdmissionResonanceGraftAdmissionReadinessFixture(t *testing.T, readinessPath string) {
	t.Helper()
	dir := filepath.Dir(readinessPath)
	sourceVerificationPath := filepath.Join(dir, "srcledgerverify.json")
	writeWeightedAdmissionResonanceGraftAdmissionLedgerVerificationFixture(t, sourceVerificationPath)
	if err := runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionReadiness([]string{sourceVerificationPath, readinessPath}); err != nil {
		t.Fatalf("write weighted admission resonance graft admission readiness fixture: %v", err)
	}
}
