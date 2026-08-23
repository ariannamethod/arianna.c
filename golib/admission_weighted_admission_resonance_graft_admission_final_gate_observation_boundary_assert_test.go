package main

import (
	"path/filepath"
	"strings"
	"testing"
)

func TestAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryAssert(t *testing.T) {
	dir := t.TempDir()

	requireBoundaryAssertError(t, runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryAssert(nil), "usage: --admission-live-route-weighted-admission-resonance-graft-admission-final-gate-observation-boundary-assert REPORT")
	requireBoundaryAssertError(t, runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryAssert([]string{"boundary.json", "extra"}), "usage: --admission-live-route-weighted-admission-resonance-graft-admission-final-gate-observation-boundary-assert REPORT")
	requireBoundaryAssertError(t, runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryAssert([]string{"  "}), "weighted admission resonance graft admission final gate observation boundary path missing")
	requireBoundaryAssertError(t, runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryAssert([]string{filepath.Join(dir, "missing.json")}), "weighted admission resonance graft admission final gate observation boundary not written")

	emptyPath := filepath.Join(dir, "empty.json")
	writeWeightedReadinessFixture(t, emptyPath, "")
	requireBoundaryAssertError(t, runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryAssert([]string{emptyPath}), "weighted admission resonance graft admission final gate observation boundary not written")

	invalidPath := filepath.Join(dir, "invalid.json")
	writeWeightedReadinessFixture(t, invalidPath, "{")
	err := runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryAssert([]string{invalidPath})
	if err == nil || !strings.HasPrefix(err.Error(), "weighted admission resonance graft admission final gate observation boundary JSON invalid:") {
		t.Fatalf("expected invalid JSON error, got %v", err)
	}

	missingSchemaPath := filepath.Join(dir, "missing_schema.json")
	writeWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryFixture(t, missingSchemaPath)
	writeWeightedReadinessFixture(t, missingSchemaPath, stringsReplaceFirst(readText(t, missingSchemaPath), `"schema": "arianna.live_route_weighted_admission_resonance_graft_admission_final_gate_observation_boundary.v1",`, ""))
	requireBoundaryAssertError(t, runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryAssert([]string{missingSchemaPath}), "weighted admission resonance graft admission final gate observation boundary schema missing")

	badSchemaPath := filepath.Join(dir, "bad_schema.json")
	writeWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryFixture(t, badSchemaPath)
	writeWeightedReadinessFixture(t, badSchemaPath, stringsReplaceFirst(readText(t, badSchemaPath), `"schema": "arianna.live_route_weighted_admission_resonance_graft_admission_final_gate_observation_boundary.v1"`, `"schema": "arianna.live_route_weighted_admission_resonance_graft_admission_final_gate_observation_boundary.v0"`))
	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryAssert([]string{badSchemaPath}),
		`weighted admission resonance graft admission final gate observation boundary schema mismatch: got "arianna.live_route_weighted_admission_resonance_graft_admission_final_gate_observation_boundary.v0" want "`+admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundarySchema+`"`,
	)

	validPath := filepath.Join(dir, "valid.json")
	writeWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryFixture(t, validPath)
	if err := runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryAssert([]string{validPath}); err != nil {
		t.Fatalf("valid weighted admission resonance graft admission final gate observation boundary rejected: %v", err)
	}

	badStatusPath := filepath.Join(dir, "bad_status.json")
	writeWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryFixture(t, badStatusPath)
	writeWeightedReadinessFixture(t, badStatusPath, stringsReplaceFirst(readText(t, badStatusPath), `"status": "shadow_graft_admission_final_gate_observation_boundary_declared_dry_run"`, `"status": "open"`))
	requireBoundaryAssertError(t, runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryAssert([]string{badStatusPath}), `weighted admission resonance graft admission final gate observation boundary status mismatch: got "open" want "shadow_graft_admission_final_gate_observation_boundary_declared_dry_run"`)

	notReadyPath := filepath.Join(dir, "not_ready.json")
	writeWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryFixture(t, notReadyPath)
	writeWeightedReadinessFixture(t, notReadyPath, stringsReplaceFirst(readText(t, notReadyPath), `"weighted_admission_resonance_graft_admission_final_gate_observation_boundary_ready": true`, `"weighted_admission_resonance_graft_admission_final_gate_observation_boundary_ready": false`))
	requireBoundaryAssertError(t, runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryAssert([]string{notReadyPath}), "weighted admission resonance graft admission final gate observation boundary weighted_admission_resonance_graft_admission_final_gate_observation_boundary_ready not ready")

	badActionPath := filepath.Join(dir, "bad_action.json")
	writeWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryFixture(t, badActionPath)
	writeWeightedReadinessFixture(t, badActionPath, stringsReplaceFirst(readText(t, badActionPath), `"action": "declare_weighted_resonance_shadow_graft_admission_final_gate_observation_boundary_dry_run"`, `"action": "open"`))
	requireBoundaryAssertError(t, runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryAssert([]string{badActionPath}), `weighted admission resonance graft admission final gate observation boundary action mismatch: got "open" want "declare_weighted_resonance_shadow_graft_admission_final_gate_observation_boundary_dry_run"`)

	badBoundaryActionPath := filepath.Join(dir, "bad_boundary_action.json")
	writeWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryFixture(t, badBoundaryActionPath)
	writeWeightedReadinessFixture(t, badBoundaryActionPath, stringsReplaceFirst(readText(t, badBoundaryActionPath), `"admission_final_gate_observation_boundary_action": "declare_blocked_final_gate_observation_boundary"`, `"admission_final_gate_observation_boundary_action": "open"`))
	requireBoundaryAssertError(t, runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryAssert([]string{badBoundaryActionPath}), "weighted admission resonance graft admission final gate observation boundary shape mismatch")

	openedBoundaryPath := filepath.Join(dir, "opened_boundary.json")
	writeWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryFixture(t, openedBoundaryPath)
	writeWeightedReadinessFixture(t, openedBoundaryPath, stringsReplaceFirst(readText(t, openedBoundaryPath), `"admission_final_gate_observation_boundary_ready": false`, `"admission_final_gate_observation_boundary_ready": true`))
	requireBoundaryAssertError(t, runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryAssert([]string{openedBoundaryPath}), "weighted admission resonance graft admission final gate observation boundary opened admission_final_gate_observation_boundary_ready")

	openedSourceObservationPath := filepath.Join(dir, "opened_source_observation.json")
	writeWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryFixture(t, openedSourceObservationPath)
	writeWeightedReadinessFixture(t, openedSourceObservationPath, stringsReplaceFirst(readText(t, openedSourceObservationPath), `"source_admission_final_gate_observation_ready": false`, `"source_admission_final_gate_observation_ready": true`))
	requireBoundaryAssertError(t, runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryAssert([]string{openedSourceObservationPath}), "weighted admission resonance graft admission final gate observation boundary opened source_admission_final_gate_observation_ready")

	openedPath := filepath.Join(dir, "opened.json")
	writeWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryFixture(t, openedPath)
	writeWeightedReadinessFixture(t, openedPath, stringsReplaceFirst(readText(t, openedPath), `"live_admission_enabled": false`, `"live_admission_enabled": true`))
	requireBoundaryAssertError(t, runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryAssert([]string{openedPath}), "weighted admission resonance graft admission final gate observation boundary opened live_admission_enabled")

	badSourcePath := filepath.Join(dir, "bad_source.json")
	writeWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryFixture(t, badSourcePath)
	writeWeightedReadinessFixture(t, badSourcePath, stringsReplaceFirst(readText(t, badSourcePath), `"source_schema": "arianna.live_route_weighted_admission_resonance_graft_admission_final_gate_observation.v1"`, `"source_schema": "arianna.live_route_weighted_admission_resonance_graft_admission_final_gate_observation.v0"`))
	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryAssert([]string{badSourcePath}),
		`weighted admission resonance graft admission final gate observation boundary source_schema mismatch: got "arianna.live_route_weighted_admission_resonance_graft_admission_final_gate_observation.v0" want "`+admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationSchema+`"`,
	)

	badSourceShapePath := filepath.Join(dir, "bad_source_shape.json")
	writeWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryFixture(t, badSourceShapePath)
	writeWeightedReadinessFixture(t, badSourceShapePath, stringsReplaceFirst(readText(t, badSourceShapePath), `"source_admission_final_gate_observation_action": "record_blocked_final_gate_receiver_observation"`, `"source_admission_final_gate_observation_action": "ready"`))
	requireBoundaryAssertError(t, runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryAssert([]string{badSourceShapePath}), "weighted admission resonance graft admission final gate observation boundary source admission final gate observation shape mismatch")

	badBoundaryKindPath := filepath.Join(dir, "bad_boundary_kind.json")
	writeWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryFixture(t, badBoundaryKindPath)
	writeWeightedReadinessFixture(t, badBoundaryKindPath, stringsReplaceFirst(readText(t, badBoundaryKindPath), `"final_gate_observation_boundary_kind": "blocked_final_gate_observation_boundary"`, `"final_gate_observation_boundary_kind": "open"`))
	requireBoundaryAssertError(t, runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryAssert([]string{badBoundaryKindPath}), `weighted admission resonance graft admission final gate observation boundary boundary_kind mismatch: got "open" want "blocked_final_gate_observation_boundary"`)

	badBoundaryHashPath := filepath.Join(dir, "bad_boundary_hash.json")
	writeWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryFixture(t, badBoundaryHashPath)
	writeWeightedReadinessFixture(t, badBoundaryHashPath, stringsReplaceFirst(readText(t, badBoundaryHashPath), `"admission_final_gate_observation_boundary_hash": "weighted-resonance-graft-admission-final-gate-observation-boundary-`, `"admission_final_gate_observation_boundary_hash": "weighted-resonance-graft-admission-final-gate-observation-boundary-bad`))
	requireBoundaryAssertError(t, runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryAssert([]string{badBoundaryHashPath}), "weighted admission resonance graft admission final gate observation boundary boundary_hash mismatch")

	badBodyTargetPath := filepath.Join(dir, "bad_body_target.json")
	writeWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryFixture(t, badBodyTargetPath)
	writeWeightedReadinessFixture(t, badBodyTargetPath, stringsReplaceFirst(readText(t, badBodyTargetPath), `"body_target": "none"`, `"body_target": "live"`))
	requireBoundaryAssertError(t, runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryAssert([]string{badBodyTargetPath}), `weighted admission resonance graft admission final gate observation boundary body_target mismatch: got "live" want "none"`)
}

func writeWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryFixture(t *testing.T, boundaryPath string) {
	t.Helper()
	dir := filepath.Dir(boundaryPath)
	observationPath := filepath.Join(dir, "srcobservation.json")
	writeWeightedAdmissionResonanceGraftAdmissionFinalGateObservationFixture(t, observationPath)
	if err := runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundary([]string{observationPath, boundaryPath}); err != nil {
		t.Fatalf("write weighted admission resonance graft admission final gate observation boundary fixture: %v", err)
	}
}
