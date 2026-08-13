package main

import (
	"path/filepath"
	"strings"
	"testing"
)

func TestAdmissionLiveRouteWeightedAdmissionResonanceGraftPreflightAssert(t *testing.T) {
	dir := t.TempDir()

	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionResonanceGraftPreflightAssert(nil),
		"usage: --admission-live-route-weighted-admission-resonance-graft-preflight-assert REPORT",
	)
	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionResonanceGraftPreflightAssert([]string{"preflight.json", "extra"}),
		"usage: --admission-live-route-weighted-admission-resonance-graft-preflight-assert REPORT",
	)
	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionResonanceGraftPreflightAssert([]string{"  "}),
		"weighted admission resonance graft preflight path missing",
	)
	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionResonanceGraftPreflightAssert([]string{filepath.Join(dir, "missing.json")}),
		"weighted admission resonance graft preflight not written",
	)

	emptyPath := filepath.Join(dir, "empty.json")
	writeWeightedReadinessFixture(t, emptyPath, "")
	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionResonanceGraftPreflightAssert([]string{emptyPath}),
		"weighted admission resonance graft preflight not written",
	)

	invalidPath := filepath.Join(dir, "invalid.json")
	writeWeightedReadinessFixture(t, invalidPath, "{")
	err := runAdmissionLiveRouteWeightedAdmissionResonanceGraftPreflightAssert([]string{invalidPath})
	if err == nil || !strings.HasPrefix(err.Error(), "weighted admission resonance graft preflight JSON invalid:") {
		t.Fatalf("expected invalid JSON error, got %v", err)
	}

	missingSchemaPath := filepath.Join(dir, "missing_schema.json")
	writeWeightedAdmissionResonanceGraftPreflightFixture(t, missingSchemaPath)
	preflightText := readText(t, missingSchemaPath)
	writeWeightedReadinessFixture(t, missingSchemaPath, stringsReplaceFirst(preflightText, `"schema": "arianna.live_route_weighted_admission_resonance_graft_preflight.v1",`, ""))
	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionResonanceGraftPreflightAssert([]string{missingSchemaPath}),
		"weighted admission resonance graft preflight schema missing",
	)

	badSchemaPath := filepath.Join(dir, "bad_schema.json")
	writeWeightedAdmissionResonanceGraftPreflightFixture(t, badSchemaPath)
	writeWeightedReadinessFixture(t, badSchemaPath, stringsReplaceFirst(readText(t, badSchemaPath), `"schema": "arianna.live_route_weighted_admission_resonance_graft_preflight.v1"`, `"schema": "arianna.live_route_weighted_admission_resonance_graft_preflight.v0"`))
	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionResonanceGraftPreflightAssert([]string{badSchemaPath}),
		`weighted admission resonance graft preflight schema mismatch: got "arianna.live_route_weighted_admission_resonance_graft_preflight.v0" want "`+admissionLiveRouteWeightedAdmissionResonanceGraftPreflightSchema+`"`,
	)

	validPath := filepath.Join(dir, "valid.json")
	writeWeightedAdmissionResonanceGraftPreflightFixture(t, validPath)
	if err := runAdmissionLiveRouteWeightedAdmissionResonanceGraftPreflightAssert([]string{validPath}); err != nil {
		t.Fatalf("valid weighted admission resonance graft preflight rejected: %v", err)
	}

	badStatusPath := filepath.Join(dir, "bad_status.json")
	writeWeightedAdmissionResonanceGraftPreflightFixture(t, badStatusPath)
	writeWeightedReadinessFixture(t, badStatusPath, stringsReplaceFirst(readText(t, badStatusPath), `"status": "shadow_graft_preflight_ready_dry_run"`, `"status": "open"`))
	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionResonanceGraftPreflightAssert([]string{badStatusPath}),
		`weighted admission resonance graft preflight status mismatch: got "open" want "shadow_graft_preflight_ready_dry_run"`,
	)

	notReadyPath := filepath.Join(dir, "not_ready.json")
	writeWeightedAdmissionResonanceGraftPreflightFixture(t, notReadyPath)
	writeWeightedReadinessFixture(t, notReadyPath, stringsReplaceFirst(readText(t, notReadyPath), `"weighted_admission_resonance_graft_preflight_ready": true`, `"weighted_admission_resonance_graft_preflight_ready": false`))
	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionResonanceGraftPreflightAssert([]string{notReadyPath}),
		"weighted admission resonance graft preflight weighted_admission_resonance_graft_preflight_ready not ready",
	)

	openedPath := filepath.Join(dir, "opened.json")
	writeWeightedAdmissionResonanceGraftPreflightFixture(t, openedPath)
	writeWeightedReadinessFixture(t, openedPath, stringsReplaceFirst(readText(t, openedPath), `"graft_allowed": false`, `"graft_allowed": true`))
	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionResonanceGraftPreflightAssert([]string{openedPath}),
		"weighted admission resonance graft preflight opened graft_allowed",
	)

	openedLivePath := filepath.Join(dir, "opened_live.json")
	writeWeightedAdmissionResonanceGraftPreflightFixture(t, openedLivePath)
	writeWeightedReadinessFixture(t, openedLivePath, stringsReplaceFirst(readText(t, openedLivePath), `"live_admission_enabled": false`, `"live_admission_enabled": true`))
	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionResonanceGraftPreflightAssert([]string{openedLivePath}),
		"weighted admission resonance graft preflight opened live_admission_enabled",
	)

	missingPathField := filepath.Join(dir, "missing_path.json")
	writeWeightedAdmissionResonanceGraftPreflightFixture(t, missingPathField)
	boundaryReport := filepath.Join(dir, "boundary-"+filepath.Base(missingPathField))
	writeWeightedReadinessFixture(t, missingPathField, stringsReplaceFirst(readText(t, missingPathField), `"source_report": "`+boundaryReport+`"`, `"source_report": " "`))
	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionResonanceGraftPreflightAssert([]string{missingPathField}),
		"weighted admission resonance graft preflight source_report missing",
	)

	badSourcePath := filepath.Join(dir, "bad_source.json")
	writeWeightedAdmissionResonanceGraftPreflightFixture(t, badSourcePath)
	writeWeightedReadinessFixture(t, badSourcePath, stringsReplaceFirst(readText(t, badSourcePath), `"source_schema": "arianna.live_route_weighted_admission_resonance_graft_boundary.v1"`, `"source_schema": "arianna.live_route_weighted_admission_resonance_graft_boundary.v0"`))
	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionResonanceGraftPreflightAssert([]string{badSourcePath}),
		`weighted admission resonance graft preflight source_schema mismatch: got "arianna.live_route_weighted_admission_resonance_graft_boundary.v0" want "`+admissionLiveRouteWeightedAdmissionResonanceGraftBoundarySchema+`"`,
	)

	badSourceBoundaryKindPath := filepath.Join(dir, "bad_source_boundary_kind.json")
	writeWeightedAdmissionResonanceGraftPreflightFixture(t, badSourceBoundaryKindPath)
	writeWeightedReadinessFixture(t, badSourceBoundaryKindPath, stringsReplaceFirst(readText(t, badSourceBoundaryKindPath), `"source_boundary_kind": "shadow_graft_boundary"`, `"source_boundary_kind": "live_graft"`))
	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionResonanceGraftPreflightAssert([]string{badSourceBoundaryKindPath}),
		"weighted admission resonance graft preflight source boundary shape mismatch",
	)

	badPreflightHashPath := filepath.Join(dir, "bad_preflight_hash.json")
	writeWeightedAdmissionResonanceGraftPreflightFixture(t, badPreflightHashPath)
	writeWeightedReadinessFixture(t, badPreflightHashPath, stringsReplaceFirst(readText(t, badPreflightHashPath), `"preflight_hash": "weighted-resonance-graft-preflight-`, `"preflight_hash": "weighted-resonance-graft-preflight-bad`))
	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionResonanceGraftPreflightAssert([]string{badPreflightHashPath}),
		"weighted admission resonance graft preflight preflight_hash mismatch",
	)

	badSourceGuardPath := filepath.Join(dir, "bad_source_guard.json")
	writeWeightedAdmissionResonanceGraftPreflightFixture(t, badSourceGuardPath)
	writeWeightedReadinessFixture(t, badSourceGuardPath, stringsReplaceFirst(readText(t, badSourceGuardPath), `"source_raw_dream_text_allowed": false`, `"source_raw_dream_text_allowed": true`))
	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionResonanceGraftPreflightAssert([]string{badSourceGuardPath}),
		"weighted admission resonance graft preflight opened source_raw_dream_text_allowed",
	)

	badBoundaryGuardPath := filepath.Join(dir, "bad_boundary_guard.json")
	writeWeightedAdmissionResonanceGraftPreflightFixture(t, badBoundaryGuardPath)
	writeWeightedReadinessFixture(t, badBoundaryGuardPath, stringsReplaceFirst(readText(t, badBoundaryGuardPath), `"source_boundary_graft_allowed": false`, `"source_boundary_graft_allowed": true`))
	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionResonanceGraftPreflightAssert([]string{badBoundaryGuardPath}),
		"weighted admission resonance graft preflight opened source_boundary_graft_allowed",
	)

	badBodyTargetPath := filepath.Join(dir, "bad_body_target.json")
	writeWeightedAdmissionResonanceGraftPreflightFixture(t, badBodyTargetPath)
	writeWeightedReadinessFixture(t, badBodyTargetPath, stringsReplaceFirst(readText(t, badBodyTargetPath), `"body_target": "none"`, `"body_target": "live"`))
	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionResonanceGraftPreflightAssert([]string{badBodyTargetPath}),
		`weighted admission resonance graft preflight body_target mismatch: got "live" want "none"`,
	)
}

func writeWeightedAdmissionResonanceGraftPreflightFixture(t *testing.T, preflightPath string) {
	t.Helper()
	dir := filepath.Dir(preflightPath)
	boundaryPath := filepath.Join(dir, "boundary-"+filepath.Base(preflightPath))
	writeWeightedAdmissionResonanceGraftBoundaryFixture(t, boundaryPath)
	if err := runAdmissionLiveRouteWeightedAdmissionResonanceGraftPreflight([]string{boundaryPath, preflightPath}); err != nil {
		t.Fatalf("write weighted admission resonance graft preflight fixture: %v", err)
	}
}
