package main

import (
	"path/filepath"
	"strings"
	"testing"
)

func TestAdmissionLiveRouteWeightedAdmissionResonanceGraftBoundaryAssert(t *testing.T) {
	dir := t.TempDir()

	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionResonanceGraftBoundaryAssert(nil),
		"usage: --admission-live-route-weighted-admission-resonance-graft-boundary-assert REPORT",
	)
	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionResonanceGraftBoundaryAssert([]string{"boundary.json", "extra"}),
		"usage: --admission-live-route-weighted-admission-resonance-graft-boundary-assert REPORT",
	)
	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionResonanceGraftBoundaryAssert([]string{"  "}),
		"weighted admission resonance graft boundary path missing",
	)
	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionResonanceGraftBoundaryAssert([]string{filepath.Join(dir, "missing.json")}),
		"weighted admission resonance graft boundary not written",
	)

	emptyPath := filepath.Join(dir, "empty.json")
	writeWeightedReadinessFixture(t, emptyPath, "")
	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionResonanceGraftBoundaryAssert([]string{emptyPath}),
		"weighted admission resonance graft boundary not written",
	)

	invalidPath := filepath.Join(dir, "invalid.json")
	writeWeightedReadinessFixture(t, invalidPath, "{")
	err := runAdmissionLiveRouteWeightedAdmissionResonanceGraftBoundaryAssert([]string{invalidPath})
	if err == nil || !strings.HasPrefix(err.Error(), "weighted admission resonance graft boundary JSON invalid:") {
		t.Fatalf("expected invalid JSON error, got %v", err)
	}

	missingSchemaPath := filepath.Join(dir, "missing_schema.json")
	writeWeightedAdmissionResonanceGraftBoundaryFixture(t, missingSchemaPath)
	boundaryText := readText(t, missingSchemaPath)
	writeWeightedReadinessFixture(t, missingSchemaPath, stringsReplaceFirst(boundaryText, `"schema": "arianna.live_route_weighted_admission_resonance_graft_boundary.v1",`, ""))
	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionResonanceGraftBoundaryAssert([]string{missingSchemaPath}),
		"weighted admission resonance graft boundary schema missing",
	)

	badSchemaPath := filepath.Join(dir, "bad_schema.json")
	writeWeightedAdmissionResonanceGraftBoundaryFixture(t, badSchemaPath)
	writeWeightedReadinessFixture(t, badSchemaPath, stringsReplaceFirst(readText(t, badSchemaPath), `"schema": "arianna.live_route_weighted_admission_resonance_graft_boundary.v1"`, `"schema": "arianna.live_route_weighted_admission_resonance_graft_boundary.v0"`))
	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionResonanceGraftBoundaryAssert([]string{badSchemaPath}),
		`weighted admission resonance graft boundary schema mismatch: got "arianna.live_route_weighted_admission_resonance_graft_boundary.v0" want "`+admissionLiveRouteWeightedAdmissionResonanceGraftBoundarySchema+`"`,
	)

	validPath := filepath.Join(dir, "valid.json")
	writeWeightedAdmissionResonanceGraftBoundaryFixture(t, validPath)
	if err := runAdmissionLiveRouteWeightedAdmissionResonanceGraftBoundaryAssert([]string{validPath}); err != nil {
		t.Fatalf("valid weighted admission resonance graft boundary rejected: %v", err)
	}

	badStatusPath := filepath.Join(dir, "bad_status.json")
	writeWeightedAdmissionResonanceGraftBoundaryFixture(t, badStatusPath)
	writeWeightedReadinessFixture(t, badStatusPath, stringsReplaceFirst(readText(t, badStatusPath), `"status": "shadow_graft_boundary_declared_dry_run"`, `"status": "open"`))
	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionResonanceGraftBoundaryAssert([]string{badStatusPath}),
		`weighted admission resonance graft boundary status mismatch: got "open" want "shadow_graft_boundary_declared_dry_run"`,
	)

	notReadyPath := filepath.Join(dir, "not_ready.json")
	writeWeightedAdmissionResonanceGraftBoundaryFixture(t, notReadyPath)
	writeWeightedReadinessFixture(t, notReadyPath, stringsReplaceFirst(readText(t, notReadyPath), `"weighted_admission_resonance_graft_boundary_ready": true`, `"weighted_admission_resonance_graft_boundary_ready": false`))
	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionResonanceGraftBoundaryAssert([]string{notReadyPath}),
		"weighted admission resonance graft boundary weighted_admission_resonance_graft_boundary_ready not ready",
	)

	openedPath := filepath.Join(dir, "opened.json")
	writeWeightedAdmissionResonanceGraftBoundaryFixture(t, openedPath)
	writeWeightedReadinessFixture(t, openedPath, stringsReplaceFirst(readText(t, openedPath), `"graft_allowed": false`, `"graft_allowed": true`))
	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionResonanceGraftBoundaryAssert([]string{openedPath}),
		"weighted admission resonance graft boundary opened graft_allowed",
	)

	openedLivePath := filepath.Join(dir, "opened_live.json")
	writeWeightedAdmissionResonanceGraftBoundaryFixture(t, openedLivePath)
	writeWeightedReadinessFixture(t, openedLivePath, stringsReplaceFirst(readText(t, openedLivePath), `"live_admission_enabled": false`, `"live_admission_enabled": true`))
	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionResonanceGraftBoundaryAssert([]string{openedLivePath}),
		"weighted admission resonance graft boundary opened live_admission_enabled",
	)

	missingPathField := filepath.Join(dir, "missing_path.json")
	writeWeightedAdmissionResonanceGraftBoundaryFixture(t, missingPathField)
	observationReport := filepath.Join(dir, "observation-"+filepath.Base(missingPathField))
	writeWeightedReadinessFixture(t, missingPathField, stringsReplaceFirst(readText(t, missingPathField), `"source_report": "`+observationReport+`"`, `"source_report": " "`))
	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionResonanceGraftBoundaryAssert([]string{missingPathField}),
		"weighted admission resonance graft boundary source_report missing",
	)

	badSourcePath := filepath.Join(dir, "bad_source.json")
	writeWeightedAdmissionResonanceGraftBoundaryFixture(t, badSourcePath)
	writeWeightedReadinessFixture(t, badSourcePath, stringsReplaceFirst(readText(t, badSourcePath), `"source_schema": "arianna.live_route_weighted_admission_resonance_observation.v1"`, `"source_schema": "arianna.live_route_weighted_admission_resonance_observation.v0"`))
	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionResonanceGraftBoundaryAssert([]string{badSourcePath}),
		`weighted admission resonance graft boundary source_schema mismatch: got "arianna.live_route_weighted_admission_resonance_observation.v0" want "`+admissionLiveRouteWeightedAdmissionResonanceObservationSchema+`"`,
	)

	badObserverPath := filepath.Join(dir, "bad_observer.json")
	writeWeightedAdmissionResonanceGraftBoundaryFixture(t, badObserverPath)
	writeWeightedReadinessFixture(t, badObserverPath, stringsReplaceFirst(readText(t, badObserverPath), `"source_observer": "resonance"`, `"source_observer": "janus"`))
	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionResonanceGraftBoundaryAssert([]string{badObserverPath}),
		`weighted admission resonance graft boundary source_observer mismatch: got "janus" want "resonance"`,
	)

	badHashPath := filepath.Join(dir, "bad_hash.json")
	writeWeightedAdmissionResonanceGraftBoundaryFixture(t, badHashPath)
	writeWeightedReadinessFixture(t, badHashPath, stringsReplaceFirst(readText(t, badHashPath), `"boundary_hash": "weighted-resonance-graft-boundary-`, `"boundary_hash": "weighted-resonance-graft-boundary-bad`))
	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionResonanceGraftBoundaryAssert([]string{badHashPath}),
		"weighted admission resonance graft boundary boundary_hash mismatch",
	)

	badSourceGuardPath := filepath.Join(dir, "bad_source_guard.json")
	writeWeightedAdmissionResonanceGraftBoundaryFixture(t, badSourceGuardPath)
	writeWeightedReadinessFixture(t, badSourceGuardPath, stringsReplaceFirst(readText(t, badSourceGuardPath), `"source_raw_dream_text_allowed": false`, `"source_raw_dream_text_allowed": true`))
	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionResonanceGraftBoundaryAssert([]string{badSourceGuardPath}),
		"weighted admission resonance graft boundary opened source_raw_dream_text_allowed",
	)

	badBodyTargetPath := filepath.Join(dir, "bad_body_target.json")
	writeWeightedAdmissionResonanceGraftBoundaryFixture(t, badBodyTargetPath)
	writeWeightedReadinessFixture(t, badBodyTargetPath, stringsReplaceFirst(readText(t, badBodyTargetPath), `"body_target": "none"`, `"body_target": "live"`))
	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionResonanceGraftBoundaryAssert([]string{badBodyTargetPath}),
		`weighted admission resonance graft boundary body_target mismatch: got "live" want "none"`,
	)
}

func writeWeightedAdmissionResonanceGraftBoundaryFixture(t *testing.T, boundaryPath string) {
	t.Helper()
	dir := filepath.Dir(boundaryPath)
	observationPath := filepath.Join(dir, "observation-"+filepath.Base(boundaryPath))
	writeWeightedAdmissionResonanceObservationFixture(t, observationPath)
	if err := runAdmissionLiveRouteWeightedAdmissionResonanceGraftBoundary([]string{observationPath, boundaryPath}); err != nil {
		t.Fatalf("write weighted admission resonance graft boundary fixture: %v", err)
	}
}
