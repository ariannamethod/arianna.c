package main

import (
	"path/filepath"
	"strings"
	"testing"
)

func TestAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionAuthorityAssert(t *testing.T) {
	dir := t.TempDir()

	requireBoundaryAssertError(t, runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionAuthorityAssert(nil), "usage: --admission-live-route-weighted-admission-resonance-graft-admission-authority-assert REPORT")
	requireBoundaryAssertError(t, runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionAuthorityAssert([]string{"authority.json", "extra"}), "usage: --admission-live-route-weighted-admission-resonance-graft-admission-authority-assert REPORT")
	requireBoundaryAssertError(t, runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionAuthorityAssert([]string{"  "}), "weighted admission resonance graft admission authority path missing")
	requireBoundaryAssertError(t, runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionAuthorityAssert([]string{filepath.Join(dir, "missing.json")}), "weighted admission resonance graft admission authority not written")

	emptyPath := filepath.Join(dir, "empty.json")
	writeWeightedReadinessFixture(t, emptyPath, "")
	requireBoundaryAssertError(t, runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionAuthorityAssert([]string{emptyPath}), "weighted admission resonance graft admission authority not written")

	invalidPath := filepath.Join(dir, "invalid.json")
	writeWeightedReadinessFixture(t, invalidPath, "{")
	err := runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionAuthorityAssert([]string{invalidPath})
	if err == nil || !strings.HasPrefix(err.Error(), "weighted admission resonance graft admission authority JSON invalid:") {
		t.Fatalf("expected invalid JSON error, got %v", err)
	}

	missingSchemaPath := filepath.Join(dir, "missing_schema.json")
	writeWeightedAdmissionResonanceGraftAdmissionAuthorityFixture(t, missingSchemaPath)
	writeWeightedReadinessFixture(t, missingSchemaPath, stringsReplaceFirst(readText(t, missingSchemaPath), `"schema": "arianna.live_route_weighted_admission_resonance_graft_admission_authority.v1",`, ""))
	requireBoundaryAssertError(t, runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionAuthorityAssert([]string{missingSchemaPath}), "weighted admission resonance graft admission authority schema missing")

	badSchemaPath := filepath.Join(dir, "bad_schema.json")
	writeWeightedAdmissionResonanceGraftAdmissionAuthorityFixture(t, badSchemaPath)
	writeWeightedReadinessFixture(t, badSchemaPath, stringsReplaceFirst(readText(t, badSchemaPath), `"schema": "arianna.live_route_weighted_admission_resonance_graft_admission_authority.v1"`, `"schema": "arianna.live_route_weighted_admission_resonance_graft_admission_authority.v0"`))
	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionAuthorityAssert([]string{badSchemaPath}),
		`weighted admission resonance graft admission authority schema mismatch: got "arianna.live_route_weighted_admission_resonance_graft_admission_authority.v0" want "`+admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionAuthoritySchema+`"`,
	)

	validPath := filepath.Join(dir, "valid.json")
	writeWeightedAdmissionResonanceGraftAdmissionAuthorityFixture(t, validPath)
	if err := runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionAuthorityAssert([]string{validPath}); err != nil {
		t.Fatalf("valid weighted admission resonance graft admission authority rejected: %v", err)
	}

	badStatusPath := filepath.Join(dir, "bad_status.json")
	writeWeightedAdmissionResonanceGraftAdmissionAuthorityFixture(t, badStatusPath)
	writeWeightedReadinessFixture(t, badStatusPath, stringsReplaceFirst(readText(t, badStatusPath), `"status": "shadow_graft_admission_authority_blocked_dry_run"`, `"status": "open"`))
	requireBoundaryAssertError(t, runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionAuthorityAssert([]string{badStatusPath}), `weighted admission resonance graft admission authority status mismatch: got "open" want "shadow_graft_admission_authority_blocked_dry_run"`)

	notReadyPath := filepath.Join(dir, "not_ready.json")
	writeWeightedAdmissionResonanceGraftAdmissionAuthorityFixture(t, notReadyPath)
	writeWeightedReadinessFixture(t, notReadyPath, stringsReplaceFirst(readText(t, notReadyPath), `"weighted_admission_resonance_graft_admission_authority_ready": true`, `"weighted_admission_resonance_graft_admission_authority_ready": false`))
	requireBoundaryAssertError(t, runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionAuthorityAssert([]string{notReadyPath}), "weighted admission resonance graft admission authority weighted_admission_resonance_graft_admission_authority_ready not ready")

	badActionPath := filepath.Join(dir, "bad_action.json")
	writeWeightedAdmissionResonanceGraftAdmissionAuthorityFixture(t, badActionPath)
	writeWeightedReadinessFixture(t, badActionPath, stringsReplaceFirst(readText(t, badActionPath), `"action": "block_weighted_resonance_shadow_graft_admission_permit_blocked_dry_run"`, `"action": "open"`))
	requireBoundaryAssertError(t, runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionAuthorityAssert([]string{badActionPath}), `weighted admission resonance graft admission authority action mismatch: got "open" want "block_weighted_resonance_shadow_graft_admission_permit_blocked_dry_run"`)

	badAuthorityActionPath := filepath.Join(dir, "bad_authority_action.json")
	writeWeightedAdmissionResonanceGraftAdmissionAuthorityFixture(t, badAuthorityActionPath)
	writeWeightedReadinessFixture(t, badAuthorityActionPath, stringsReplaceFirst(readText(t, badAuthorityActionPath), `"admission_authority_action": "reject_blocked_admission_permit"`, `"admission_authority_action": "grant_live_authority"`))
	requireBoundaryAssertError(t, runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionAuthorityAssert([]string{badAuthorityActionPath}), "weighted admission resonance graft admission authority shape mismatch")

	openedAuthorityPath := filepath.Join(dir, "opened_authority.json")
	writeWeightedAdmissionResonanceGraftAdmissionAuthorityFixture(t, openedAuthorityPath)
	writeWeightedReadinessFixture(t, openedAuthorityPath, stringsReplaceFirst(readText(t, openedAuthorityPath), `"admission_authority_granted": false`, `"admission_authority_granted": true`))
	requireBoundaryAssertError(t, runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionAuthorityAssert([]string{openedAuthorityPath}), "weighted admission resonance graft admission authority opened admission_authority_granted")

	openedManualPath := filepath.Join(dir, "opened_manual.json")
	writeWeightedAdmissionResonanceGraftAdmissionAuthorityFixture(t, openedManualPath)
	writeWeightedReadinessFixture(t, openedManualPath, stringsReplaceFirst(readText(t, openedManualPath), `"source_manual_permit_requested": false`, `"source_manual_permit_requested": true`))
	requireBoundaryAssertError(t, runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionAuthorityAssert([]string{openedManualPath}), "weighted admission resonance graft admission authority opened source_manual_permit_requested")

	openedPath := filepath.Join(dir, "opened.json")
	writeWeightedAdmissionResonanceGraftAdmissionAuthorityFixture(t, openedPath)
	writeWeightedReadinessFixture(t, openedPath, stringsReplaceFirst(readText(t, openedPath), `"live_admission_enabled": false`, `"live_admission_enabled": true`))
	requireBoundaryAssertError(t, runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionAuthorityAssert([]string{openedPath}), "weighted admission resonance graft admission authority opened live_admission_enabled")

	badSourcePath := filepath.Join(dir, "bad_source.json")
	writeWeightedAdmissionResonanceGraftAdmissionAuthorityFixture(t, badSourcePath)
	writeWeightedReadinessFixture(t, badSourcePath, stringsReplaceFirst(readText(t, badSourcePath), `"source_schema": "arianna.live_route_weighted_admission_resonance_graft_admission_permit.v1"`, `"source_schema": "arianna.live_route_weighted_admission_resonance_graft_admission_permit.v0"`))
	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionAuthorityAssert([]string{badSourcePath}),
		`weighted admission resonance graft admission authority source_schema mismatch: got "arianna.live_route_weighted_admission_resonance_graft_admission_permit.v0" want "`+admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionPermitSchema+`"`,
	)

	badSourcePermitShapePath := filepath.Join(dir, "bad_source_permit_shape.json")
	writeWeightedAdmissionResonanceGraftAdmissionAuthorityFixture(t, badSourcePermitShapePath)
	writeWeightedReadinessFixture(t, badSourcePermitShapePath, stringsReplaceFirst(readText(t, badSourcePermitShapePath), `"source_admission_permit_action": "reject_blocked_admission_readiness"`, `"source_admission_permit_action": "ready"`))
	requireBoundaryAssertError(t, runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionAuthorityAssert([]string{badSourcePermitShapePath}), "weighted admission resonance graft admission authority source admission permit shape mismatch")

	badAuthorityHashPath := filepath.Join(dir, "bad_authority_hash.json")
	writeWeightedAdmissionResonanceGraftAdmissionAuthorityFixture(t, badAuthorityHashPath)
	writeWeightedReadinessFixture(t, badAuthorityHashPath, stringsReplaceFirst(readText(t, badAuthorityHashPath), `"admission_authority_hash": "weighted-resonance-graft-admission-authority-`, `"admission_authority_hash": "weighted-resonance-graft-admission-authority-bad`))
	requireBoundaryAssertError(t, runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionAuthorityAssert([]string{badAuthorityHashPath}), "weighted admission resonance graft admission authority admission_authority_hash mismatch")

	badBodyTargetPath := filepath.Join(dir, "bad_body_target.json")
	writeWeightedAdmissionResonanceGraftAdmissionAuthorityFixture(t, badBodyTargetPath)
	writeWeightedReadinessFixture(t, badBodyTargetPath, stringsReplaceFirst(readText(t, badBodyTargetPath), `"body_target": "none"`, `"body_target": "live"`))
	requireBoundaryAssertError(t, runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionAuthorityAssert([]string{badBodyTargetPath}), `weighted admission resonance graft admission authority body_target mismatch: got "live" want "none"`)
}

func writeWeightedAdmissionResonanceGraftAdmissionAuthorityFixture(t *testing.T, authorityPath string) {
	t.Helper()
	dir := filepath.Dir(authorityPath)
	sourcePermitPath := filepath.Join(dir, "srcpermit.json")
	writeWeightedAdmissionResonanceGraftAdmissionPermitFixture(t, sourcePermitPath)
	if err := runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionAuthority([]string{sourcePermitPath, authorityPath}); err != nil {
		t.Fatalf("write weighted admission resonance graft admission authority fixture: %v", err)
	}
}
