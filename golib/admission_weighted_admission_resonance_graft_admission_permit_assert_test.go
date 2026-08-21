package main

import (
	"path/filepath"
	"strings"
	"testing"
)

func TestAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionPermitAssert(t *testing.T) {
	dir := t.TempDir()

	requireBoundaryAssertError(t, runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionPermitAssert(nil), "usage: --admission-live-route-weighted-admission-resonance-graft-admission-permit-assert REPORT")
	requireBoundaryAssertError(t, runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionPermitAssert([]string{"permit.json", "extra"}), "usage: --admission-live-route-weighted-admission-resonance-graft-admission-permit-assert REPORT")
	requireBoundaryAssertError(t, runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionPermitAssert([]string{"  "}), "weighted admission resonance graft admission permit path missing")
	requireBoundaryAssertError(t, runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionPermitAssert([]string{filepath.Join(dir, "missing.json")}), "weighted admission resonance graft admission permit not written")

	emptyPath := filepath.Join(dir, "empty.json")
	writeWeightedReadinessFixture(t, emptyPath, "")
	requireBoundaryAssertError(t, runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionPermitAssert([]string{emptyPath}), "weighted admission resonance graft admission permit not written")

	invalidPath := filepath.Join(dir, "invalid.json")
	writeWeightedReadinessFixture(t, invalidPath, "{")
	err := runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionPermitAssert([]string{invalidPath})
	if err == nil || !strings.HasPrefix(err.Error(), "weighted admission resonance graft admission permit JSON invalid:") {
		t.Fatalf("expected invalid JSON error, got %v", err)
	}

	missingSchemaPath := filepath.Join(dir, "missing_schema.json")
	writeWeightedAdmissionResonanceGraftAdmissionPermitFixture(t, missingSchemaPath)
	writeWeightedReadinessFixture(t, missingSchemaPath, stringsReplaceFirst(readText(t, missingSchemaPath), `"schema": "arianna.live_route_weighted_admission_resonance_graft_admission_permit.v1",`, ""))
	requireBoundaryAssertError(t, runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionPermitAssert([]string{missingSchemaPath}), "weighted admission resonance graft admission permit schema missing")

	badSchemaPath := filepath.Join(dir, "bad_schema.json")
	writeWeightedAdmissionResonanceGraftAdmissionPermitFixture(t, badSchemaPath)
	writeWeightedReadinessFixture(t, badSchemaPath, stringsReplaceFirst(readText(t, badSchemaPath), `"schema": "arianna.live_route_weighted_admission_resonance_graft_admission_permit.v1"`, `"schema": "arianna.live_route_weighted_admission_resonance_graft_admission_permit.v0"`))
	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionPermitAssert([]string{badSchemaPath}),
		`weighted admission resonance graft admission permit schema mismatch: got "arianna.live_route_weighted_admission_resonance_graft_admission_permit.v0" want "`+admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionPermitSchema+`"`,
	)

	validPath := filepath.Join(dir, "valid.json")
	writeWeightedAdmissionResonanceGraftAdmissionPermitFixture(t, validPath)
	if err := runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionPermitAssert([]string{validPath}); err != nil {
		t.Fatalf("valid weighted admission resonance graft admission permit rejected: %v", err)
	}

	badStatusPath := filepath.Join(dir, "bad_status.json")
	writeWeightedAdmissionResonanceGraftAdmissionPermitFixture(t, badStatusPath)
	writeWeightedReadinessFixture(t, badStatusPath, stringsReplaceFirst(readText(t, badStatusPath), `"status": "shadow_graft_admission_permit_blocked_dry_run"`, `"status": "open"`))
	requireBoundaryAssertError(t, runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionPermitAssert([]string{badStatusPath}), `weighted admission resonance graft admission permit status mismatch: got "open" want "shadow_graft_admission_permit_blocked_dry_run"`)

	notReadyPath := filepath.Join(dir, "not_ready.json")
	writeWeightedAdmissionResonanceGraftAdmissionPermitFixture(t, notReadyPath)
	writeWeightedReadinessFixture(t, notReadyPath, stringsReplaceFirst(readText(t, notReadyPath), `"weighted_admission_resonance_graft_admission_permit_ready": true`, `"weighted_admission_resonance_graft_admission_permit_ready": false`))
	requireBoundaryAssertError(t, runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionPermitAssert([]string{notReadyPath}), "weighted admission resonance graft admission permit weighted_admission_resonance_graft_admission_permit_ready not ready")

	badActionPath := filepath.Join(dir, "bad_action.json")
	writeWeightedAdmissionResonanceGraftAdmissionPermitFixture(t, badActionPath)
	writeWeightedReadinessFixture(t, badActionPath, stringsReplaceFirst(readText(t, badActionPath), `"action": "block_weighted_resonance_shadow_graft_admission_readiness_blocked_dry_run"`, `"action": "open"`))
	requireBoundaryAssertError(t, runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionPermitAssert([]string{badActionPath}), `weighted admission resonance graft admission permit action mismatch: got "open" want "block_weighted_resonance_shadow_graft_admission_readiness_blocked_dry_run"`)

	badPermitActionPath := filepath.Join(dir, "bad_permit_action.json")
	writeWeightedAdmissionResonanceGraftAdmissionPermitFixture(t, badPermitActionPath)
	writeWeightedReadinessFixture(t, badPermitActionPath, stringsReplaceFirst(readText(t, badPermitActionPath), `"admission_permit_action": "reject_blocked_admission_readiness"`, `"admission_permit_action": "acknowledge_verified_live_admission_readiness_dry_run"`))
	requireBoundaryAssertError(t, runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionPermitAssert([]string{badPermitActionPath}), "weighted admission resonance graft admission permit shape mismatch")

	openedPermitPath := filepath.Join(dir, "opened_permit.json")
	writeWeightedAdmissionResonanceGraftAdmissionPermitFixture(t, openedPermitPath)
	writeWeightedReadinessFixture(t, openedPermitPath, stringsReplaceFirst(readText(t, openedPermitPath), `"admission_permit_ready": false`, `"admission_permit_ready": true`))
	requireBoundaryAssertError(t, runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionPermitAssert([]string{openedPermitPath}), "weighted admission resonance graft admission permit opened admission_permit_ready")

	openedManualPath := filepath.Join(dir, "opened_manual.json")
	writeWeightedAdmissionResonanceGraftAdmissionPermitFixture(t, openedManualPath)
	writeWeightedReadinessFixture(t, openedManualPath, stringsReplaceFirst(readText(t, openedManualPath), `"manual_permit_requested": false`, `"manual_permit_requested": true`))
	requireBoundaryAssertError(t, runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionPermitAssert([]string{openedManualPath}), "weighted admission resonance graft admission permit opened manual_permit_requested")

	openedPath := filepath.Join(dir, "opened.json")
	writeWeightedAdmissionResonanceGraftAdmissionPermitFixture(t, openedPath)
	writeWeightedReadinessFixture(t, openedPath, stringsReplaceFirst(readText(t, openedPath), `"live_admission_enabled": false`, `"live_admission_enabled": true`))
	requireBoundaryAssertError(t, runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionPermitAssert([]string{openedPath}), "weighted admission resonance graft admission permit opened live_admission_enabled")

	badSourcePath := filepath.Join(dir, "bad_source.json")
	writeWeightedAdmissionResonanceGraftAdmissionPermitFixture(t, badSourcePath)
	writeWeightedReadinessFixture(t, badSourcePath, stringsReplaceFirst(readText(t, badSourcePath), `"source_schema": "arianna.live_route_weighted_admission_resonance_graft_admission_readiness.v1"`, `"source_schema": "arianna.live_route_weighted_admission_resonance_graft_admission_readiness.v0"`))
	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionPermitAssert([]string{badSourcePath}),
		`weighted admission resonance graft admission permit source_schema mismatch: got "arianna.live_route_weighted_admission_resonance_graft_admission_readiness.v0" want "`+admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionReadinessSchema+`"`,
	)

	badSourceReadinessShapePath := filepath.Join(dir, "bad_source_readiness_shape.json")
	writeWeightedAdmissionResonanceGraftAdmissionPermitFixture(t, badSourceReadinessShapePath)
	writeWeightedReadinessFixture(t, badSourceReadinessShapePath, stringsReplaceFirst(readText(t, badSourceReadinessShapePath), `"source_admission_readiness_action": "reject_blocked_ledger_verification"`, `"source_admission_readiness_action": "ready"`))
	requireBoundaryAssertError(t, runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionPermitAssert([]string{badSourceReadinessShapePath}), "weighted admission resonance graft admission permit source admission readiness shape mismatch")

	badPermitHashPath := filepath.Join(dir, "bad_permit_hash.json")
	writeWeightedAdmissionResonanceGraftAdmissionPermitFixture(t, badPermitHashPath)
	writeWeightedReadinessFixture(t, badPermitHashPath, stringsReplaceFirst(readText(t, badPermitHashPath), `"admission_permit_hash": "weighted-resonance-graft-admission-permit-`, `"admission_permit_hash": "weighted-resonance-graft-admission-permit-bad`))
	requireBoundaryAssertError(t, runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionPermitAssert([]string{badPermitHashPath}), "weighted admission resonance graft admission permit admission_permit_hash mismatch")

	badBodyTargetPath := filepath.Join(dir, "bad_body_target.json")
	writeWeightedAdmissionResonanceGraftAdmissionPermitFixture(t, badBodyTargetPath)
	writeWeightedReadinessFixture(t, badBodyTargetPath, stringsReplaceFirst(readText(t, badBodyTargetPath), `"body_target": "none"`, `"body_target": "live"`))
	requireBoundaryAssertError(t, runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionPermitAssert([]string{badBodyTargetPath}), `weighted admission resonance graft admission permit body_target mismatch: got "live" want "none"`)
}

func writeWeightedAdmissionResonanceGraftAdmissionPermitFixture(t *testing.T, permitPath string) {
	t.Helper()
	dir := filepath.Dir(permitPath)
	sourceReadinessPath := filepath.Join(dir, "srcreadiness.json")
	writeWeightedAdmissionResonanceGraftAdmissionReadinessFixture(t, sourceReadinessPath)
	if err := runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionPermit([]string{sourceReadinessPath, permitPath}); err != nil {
		t.Fatalf("write weighted admission resonance graft admission permit fixture: %v", err)
	}
}
