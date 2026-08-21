package main

import (
	"path/filepath"
	"strings"
	"testing"
)

func TestAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionSealAssert(t *testing.T) {
	dir := t.TempDir()

	requireBoundaryAssertError(t, runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionSealAssert(nil), "usage: --admission-live-route-weighted-admission-resonance-graft-admission-seal-assert REPORT")
	requireBoundaryAssertError(t, runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionSealAssert([]string{"seal.json", "extra"}), "usage: --admission-live-route-weighted-admission-resonance-graft-admission-seal-assert REPORT")
	requireBoundaryAssertError(t, runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionSealAssert([]string{"  "}), "weighted admission resonance graft admission seal path missing")
	requireBoundaryAssertError(t, runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionSealAssert([]string{filepath.Join(dir, "missing.json")}), "weighted admission resonance graft admission seal not written")

	emptyPath := filepath.Join(dir, "empty.json")
	writeWeightedReadinessFixture(t, emptyPath, "")
	requireBoundaryAssertError(t, runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionSealAssert([]string{emptyPath}), "weighted admission resonance graft admission seal not written")

	invalidPath := filepath.Join(dir, "invalid.json")
	writeWeightedReadinessFixture(t, invalidPath, "{")
	err := runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionSealAssert([]string{invalidPath})
	if err == nil || !strings.HasPrefix(err.Error(), "weighted admission resonance graft admission seal JSON invalid:") {
		t.Fatalf("expected invalid JSON error, got %v", err)
	}

	missingSchemaPath := filepath.Join(dir, "missing_schema.json")
	writeWeightedAdmissionResonanceGraftAdmissionSealFixture(t, missingSchemaPath)
	writeWeightedReadinessFixture(t, missingSchemaPath, stringsReplaceFirst(readText(t, missingSchemaPath), `"schema": "arianna.live_route_weighted_admission_resonance_graft_admission_seal.v1",`, ""))
	requireBoundaryAssertError(t, runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionSealAssert([]string{missingSchemaPath}), "weighted admission resonance graft admission seal schema missing")

	badSchemaPath := filepath.Join(dir, "bad_schema.json")
	writeWeightedAdmissionResonanceGraftAdmissionSealFixture(t, badSchemaPath)
	writeWeightedReadinessFixture(t, badSchemaPath, stringsReplaceFirst(readText(t, badSchemaPath), `"schema": "arianna.live_route_weighted_admission_resonance_graft_admission_seal.v1"`, `"schema": "arianna.live_route_weighted_admission_resonance_graft_admission_seal.v0"`))
	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionSealAssert([]string{badSchemaPath}),
		`weighted admission resonance graft admission seal schema mismatch: got "arianna.live_route_weighted_admission_resonance_graft_admission_seal.v0" want "`+admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionSealSchema+`"`,
	)

	validPath := filepath.Join(dir, "valid.json")
	writeWeightedAdmissionResonanceGraftAdmissionSealFixture(t, validPath)
	if err := runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionSealAssert([]string{validPath}); err != nil {
		t.Fatalf("valid weighted admission resonance graft admission seal rejected: %v", err)
	}

	badStatusPath := filepath.Join(dir, "bad_status.json")
	writeWeightedAdmissionResonanceGraftAdmissionSealFixture(t, badStatusPath)
	writeWeightedReadinessFixture(t, badStatusPath, stringsReplaceFirst(readText(t, badStatusPath), `"status": "shadow_graft_admission_seal_blocked_dry_run"`, `"status": "open"`))
	requireBoundaryAssertError(t, runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionSealAssert([]string{badStatusPath}), `weighted admission resonance graft admission seal status mismatch: got "open" want "shadow_graft_admission_seal_blocked_dry_run"`)

	notReadyPath := filepath.Join(dir, "not_ready.json")
	writeWeightedAdmissionResonanceGraftAdmissionSealFixture(t, notReadyPath)
	writeWeightedReadinessFixture(t, notReadyPath, stringsReplaceFirst(readText(t, notReadyPath), `"weighted_admission_resonance_graft_admission_seal_ready": true`, `"weighted_admission_resonance_graft_admission_seal_ready": false`))
	requireBoundaryAssertError(t, runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionSealAssert([]string{notReadyPath}), "weighted admission resonance graft admission seal weighted_admission_resonance_graft_admission_seal_ready not ready")

	badActionPath := filepath.Join(dir, "bad_action.json")
	writeWeightedAdmissionResonanceGraftAdmissionSealFixture(t, badActionPath)
	writeWeightedReadinessFixture(t, badActionPath, stringsReplaceFirst(readText(t, badActionPath), `"action": "seal_weighted_resonance_shadow_graft_admission_authority_blocked_dry_run"`, `"action": "open"`))
	requireBoundaryAssertError(t, runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionSealAssert([]string{badActionPath}), `weighted admission resonance graft admission seal action mismatch: got "open" want "seal_weighted_resonance_shadow_graft_admission_authority_blocked_dry_run"`)

	badSealActionPath := filepath.Join(dir, "bad_seal_action.json")
	writeWeightedAdmissionResonanceGraftAdmissionSealFixture(t, badSealActionPath)
	writeWeightedReadinessFixture(t, badSealActionPath, stringsReplaceFirst(readText(t, badSealActionPath), `"admission_seal_action": "seal_blocked_admission_authority"`, `"admission_seal_action": "open"`))
	requireBoundaryAssertError(t, runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionSealAssert([]string{badSealActionPath}), "weighted admission resonance graft admission seal shape mismatch")

	openedSealPath := filepath.Join(dir, "opened_seal.json")
	writeWeightedAdmissionResonanceGraftAdmissionSealFixture(t, openedSealPath)
	writeWeightedReadinessFixture(t, openedSealPath, stringsReplaceFirst(readText(t, openedSealPath), `"admission_seal_ready": false`, `"admission_seal_ready": true`))
	requireBoundaryAssertError(t, runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionSealAssert([]string{openedSealPath}), "weighted admission resonance graft admission seal opened admission_seal_ready")

	openedAuthorityPath := filepath.Join(dir, "opened_authority.json")
	writeWeightedAdmissionResonanceGraftAdmissionSealFixture(t, openedAuthorityPath)
	writeWeightedReadinessFixture(t, openedAuthorityPath, stringsReplaceFirst(readText(t, openedAuthorityPath), `"source_admission_authority_granted": false`, `"source_admission_authority_granted": true`))
	requireBoundaryAssertError(t, runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionSealAssert([]string{openedAuthorityPath}), "weighted admission resonance graft admission seal opened source_admission_authority_granted")

	openedPath := filepath.Join(dir, "opened.json")
	writeWeightedAdmissionResonanceGraftAdmissionSealFixture(t, openedPath)
	writeWeightedReadinessFixture(t, openedPath, stringsReplaceFirst(readText(t, openedPath), `"live_admission_enabled": false`, `"live_admission_enabled": true`))
	requireBoundaryAssertError(t, runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionSealAssert([]string{openedPath}), "weighted admission resonance graft admission seal opened live_admission_enabled")

	badSourcePath := filepath.Join(dir, "bad_source.json")
	writeWeightedAdmissionResonanceGraftAdmissionSealFixture(t, badSourcePath)
	writeWeightedReadinessFixture(t, badSourcePath, stringsReplaceFirst(readText(t, badSourcePath), `"source_schema": "arianna.live_route_weighted_admission_resonance_graft_admission_authority.v1"`, `"source_schema": "arianna.live_route_weighted_admission_resonance_graft_admission_authority.v0"`))
	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionSealAssert([]string{badSourcePath}),
		`weighted admission resonance graft admission seal source_schema mismatch: got "arianna.live_route_weighted_admission_resonance_graft_admission_authority.v0" want "`+admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionAuthoritySchema+`"`,
	)

	badSourceAuthorityShapePath := filepath.Join(dir, "bad_source_authority_shape.json")
	writeWeightedAdmissionResonanceGraftAdmissionSealFixture(t, badSourceAuthorityShapePath)
	writeWeightedReadinessFixture(t, badSourceAuthorityShapePath, stringsReplaceFirst(readText(t, badSourceAuthorityShapePath), `"source_admission_authority_action": "reject_blocked_admission_permit"`, `"source_admission_authority_action": "ready"`))
	requireBoundaryAssertError(t, runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionSealAssert([]string{badSourceAuthorityShapePath}), "weighted admission resonance graft admission seal source admission authority shape mismatch")

	badSealHashPath := filepath.Join(dir, "bad_seal_hash.json")
	writeWeightedAdmissionResonanceGraftAdmissionSealFixture(t, badSealHashPath)
	writeWeightedReadinessFixture(t, badSealHashPath, stringsReplaceFirst(readText(t, badSealHashPath), `"admission_seal_hash": "weighted-resonance-graft-admission-seal-`, `"admission_seal_hash": "weighted-resonance-graft-admission-seal-bad`))
	requireBoundaryAssertError(t, runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionSealAssert([]string{badSealHashPath}), "weighted admission resonance graft admission seal admission_seal_hash mismatch")

	badBodyTargetPath := filepath.Join(dir, "bad_body_target.json")
	writeWeightedAdmissionResonanceGraftAdmissionSealFixture(t, badBodyTargetPath)
	writeWeightedReadinessFixture(t, badBodyTargetPath, stringsReplaceFirst(readText(t, badBodyTargetPath), `"body_target": "none"`, `"body_target": "live"`))
	requireBoundaryAssertError(t, runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionSealAssert([]string{badBodyTargetPath}), `weighted admission resonance graft admission seal body_target mismatch: got "live" want "none"`)
}

func writeWeightedAdmissionResonanceGraftAdmissionSealFixture(t *testing.T, sealPath string) {
	t.Helper()
	dir := filepath.Dir(sealPath)
	sourceAuthorityPath := filepath.Join(dir, "srcauthority.json")
	writeWeightedAdmissionResonanceGraftAdmissionAuthorityFixture(t, sourceAuthorityPath)
	if err := runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionSeal([]string{sourceAuthorityPath, sealPath}); err != nil {
		t.Fatalf("write weighted admission resonance graft admission seal fixture: %v", err)
	}
}
