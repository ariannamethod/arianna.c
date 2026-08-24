package main

import (
	"path/filepath"
	"strings"
	"testing"
)

func TestAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightAssert(t *testing.T) {
	dir := t.TempDir()

	requireBoundaryAssertError(t, runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightAssert(nil), "usage: --admission-live-route-weighted-admission-resonance-graft-admission-final-gate-observation-boundary-preflight-assert REPORT")
	requireBoundaryAssertError(t, runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightAssert([]string{"preflight.json", "extra"}), "usage: --admission-live-route-weighted-admission-resonance-graft-admission-final-gate-observation-boundary-preflight-assert REPORT")
	requireBoundaryAssertError(t, runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightAssert([]string{"  "}), "weighted admission resonance graft admission final gate observation boundary preflight path missing")
	requireBoundaryAssertError(t, runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightAssert([]string{filepath.Join(dir, "missing.json")}), "weighted admission resonance graft admission final gate observation boundary preflight not written")

	emptyPath := filepath.Join(dir, "empty.json")
	writeWeightedReadinessFixture(t, emptyPath, "")
	requireBoundaryAssertError(t, runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightAssert([]string{emptyPath}), "weighted admission resonance graft admission final gate observation boundary preflight not written")

	invalidPath := filepath.Join(dir, "invalid.json")
	writeWeightedReadinessFixture(t, invalidPath, "{")
	err := runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightAssert([]string{invalidPath})
	if err == nil || !strings.HasPrefix(err.Error(), "weighted admission resonance graft admission final gate observation boundary preflight JSON invalid:") {
		t.Fatalf("expected invalid JSON error, got %v", err)
	}

	missingSchemaPath := filepath.Join(dir, "missing_schema.json")
	writeWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightFixture(t, missingSchemaPath)
	writeWeightedReadinessFixture(t, missingSchemaPath, stringsReplaceFirst(readText(t, missingSchemaPath), `"schema": "arianna.live_route_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight.v1",`, ""))
	requireBoundaryAssertError(t, runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightAssert([]string{missingSchemaPath}), "weighted admission resonance graft admission final gate observation boundary preflight schema missing")

	badSchemaPath := filepath.Join(dir, "bad_schema.json")
	writeWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightFixture(t, badSchemaPath)
	writeWeightedReadinessFixture(t, badSchemaPath, stringsReplaceFirst(readText(t, badSchemaPath), `"schema": "arianna.live_route_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight.v1"`, `"schema": "arianna.live_route_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight.v0"`))
	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightAssert([]string{badSchemaPath}),
		`weighted admission resonance graft admission final gate observation boundary preflight schema mismatch: got "arianna.live_route_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight.v0" want "`+admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightSchema+`"`,
	)

	validPath := filepath.Join(dir, "valid.json")
	writeWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightFixture(t, validPath)
	if err := runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightAssert([]string{validPath}); err != nil {
		t.Fatalf("valid weighted admission resonance graft admission final gate observation boundary preflight rejected: %v", err)
	}

	badStatusPath := filepath.Join(dir, "bad_status.json")
	writeWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightFixture(t, badStatusPath)
	writeWeightedReadinessFixture(t, badStatusPath, stringsReplaceFirst(readText(t, badStatusPath), `"status": "shadow_graft_admission_final_gate_observation_boundary_preflight_blocked_dry_run"`, `"status": "open"`))
	requireBoundaryAssertError(t, runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightAssert([]string{badStatusPath}), `weighted admission resonance graft admission final gate observation boundary preflight status mismatch: got "open" want "shadow_graft_admission_final_gate_observation_boundary_preflight_blocked_dry_run"`)

	notReadyPath := filepath.Join(dir, "not_ready.json")
	writeWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightFixture(t, notReadyPath)
	writeWeightedReadinessFixture(t, notReadyPath, stringsReplaceFirst(readText(t, notReadyPath), `"weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_ready": true`, `"weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_ready": false`))
	requireBoundaryAssertError(t, runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightAssert([]string{notReadyPath}), "weighted admission resonance graft admission final gate observation boundary preflight weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_ready not ready")

	badActionPath := filepath.Join(dir, "bad_action.json")
	writeWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightFixture(t, badActionPath)
	writeWeightedReadinessFixture(t, badActionPath, stringsReplaceFirst(readText(t, badActionPath), `"action": "check_weighted_resonance_shadow_graft_admission_final_gate_observation_boundary_preflight_dry_run"`, `"action": "open"`))
	requireBoundaryAssertError(t, runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightAssert([]string{badActionPath}), `weighted admission resonance graft admission final gate observation boundary preflight action mismatch: got "open" want "check_weighted_resonance_shadow_graft_admission_final_gate_observation_boundary_preflight_dry_run"`)

	badPreflightActionPath := filepath.Join(dir, "bad_preflight_action.json")
	writeWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightFixture(t, badPreflightActionPath)
	writeWeightedReadinessFixture(t, badPreflightActionPath, stringsReplaceFirst(readText(t, badPreflightActionPath), `"admission_final_gate_observation_boundary_preflight_action": "check_blocked_final_gate_observation_boundary_preflight"`, `"admission_final_gate_observation_boundary_preflight_action": "open"`))
	requireBoundaryAssertError(t, runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightAssert([]string{badPreflightActionPath}), "weighted admission resonance graft admission final gate observation boundary preflight shape mismatch")

	openedPreflightPath := filepath.Join(dir, "opened_preflight.json")
	writeWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightFixture(t, openedPreflightPath)
	writeWeightedReadinessFixture(t, openedPreflightPath, stringsReplaceFirst(readText(t, openedPreflightPath), `"admission_final_gate_observation_boundary_preflight_ready": false`, `"admission_final_gate_observation_boundary_preflight_ready": true`))
	requireBoundaryAssertError(t, runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightAssert([]string{openedPreflightPath}), "weighted admission resonance graft admission final gate observation boundary preflight opened admission_final_gate_observation_boundary_preflight_ready")

	openedSourceBoundaryPath := filepath.Join(dir, "opened_source_boundary.json")
	writeWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightFixture(t, openedSourceBoundaryPath)
	writeWeightedReadinessFixture(t, openedSourceBoundaryPath, stringsReplaceFirst(readText(t, openedSourceBoundaryPath), `"source_admission_final_gate_observation_boundary_ready": false`, `"source_admission_final_gate_observation_boundary_ready": true`))
	requireBoundaryAssertError(t, runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightAssert([]string{openedSourceBoundaryPath}), "weighted admission resonance graft admission final gate observation boundary preflight opened source_admission_final_gate_observation_boundary_ready")

	openedPath := filepath.Join(dir, "opened.json")
	writeWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightFixture(t, openedPath)
	writeWeightedReadinessFixture(t, openedPath, stringsReplaceFirst(readText(t, openedPath), `"live_admission_enabled": false`, `"live_admission_enabled": true`))
	requireBoundaryAssertError(t, runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightAssert([]string{openedPath}), "weighted admission resonance graft admission final gate observation boundary preflight opened live_admission_enabled")

	badSourcePath := filepath.Join(dir, "bad_source.json")
	writeWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightFixture(t, badSourcePath)
	writeWeightedReadinessFixture(t, badSourcePath, stringsReplaceFirst(readText(t, badSourcePath), `"source_schema": "arianna.live_route_weighted_admission_resonance_graft_admission_final_gate_observation_boundary.v1"`, `"source_schema": "arianna.live_route_weighted_admission_resonance_graft_admission_final_gate_observation_boundary.v0"`))
	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightAssert([]string{badSourcePath}),
		`weighted admission resonance graft admission final gate observation boundary preflight source_schema mismatch: got "arianna.live_route_weighted_admission_resonance_graft_admission_final_gate_observation_boundary.v0" want "`+admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundarySchema+`"`,
	)

	badSourceShapePath := filepath.Join(dir, "bad_source_shape.json")
	writeWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightFixture(t, badSourceShapePath)
	writeWeightedReadinessFixture(t, badSourceShapePath, stringsReplaceFirst(readText(t, badSourceShapePath), `"source_admission_final_gate_observation_boundary_action": "declare_blocked_final_gate_observation_boundary"`, `"source_admission_final_gate_observation_boundary_action": "ready"`))
	requireBoundaryAssertError(t, runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightAssert([]string{badSourceShapePath}), "weighted admission resonance graft admission final gate observation boundary preflight source admission final gate observation boundary shape mismatch")

	badPreflightKindPath := filepath.Join(dir, "bad_preflight_kind.json")
	writeWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightFixture(t, badPreflightKindPath)
	writeWeightedReadinessFixture(t, badPreflightKindPath, stringsReplaceFirst(readText(t, badPreflightKindPath), `"final_gate_observation_boundary_preflight_kind": "blocked_final_gate_observation_boundary_preflight"`, `"final_gate_observation_boundary_preflight_kind": "open"`))
	requireBoundaryAssertError(t, runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightAssert([]string{badPreflightKindPath}), `weighted admission resonance graft admission final gate observation boundary preflight preflight_kind mismatch: got "open" want "blocked_final_gate_observation_boundary_preflight"`)

	badPreflightHashPath := filepath.Join(dir, "bad_preflight_hash.json")
	writeWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightFixture(t, badPreflightHashPath)
	writeWeightedReadinessFixture(t, badPreflightHashPath, stringsReplaceFirst(readText(t, badPreflightHashPath), `"admission_final_gate_observation_boundary_preflight_hash": "weighted-resonance-graft-admission-final-gate-observation-boundary-preflight-`, `"admission_final_gate_observation_boundary_preflight_hash": "weighted-resonance-graft-admission-final-gate-observation-boundary-preflight-bad`))
	requireBoundaryAssertError(t, runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightAssert([]string{badPreflightHashPath}), "weighted admission resonance graft admission final gate observation boundary preflight preflight_hash mismatch")

	badBodyTargetPath := filepath.Join(dir, "bad_body_target.json")
	writeWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightFixture(t, badBodyTargetPath)
	writeWeightedReadinessFixture(t, badBodyTargetPath, stringsReplaceFirst(readText(t, badBodyTargetPath), `"body_target": "none"`, `"body_target": "live"`))
	requireBoundaryAssertError(t, runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightAssert([]string{badBodyTargetPath}), `weighted admission resonance graft admission final gate observation boundary preflight body_target mismatch: got "live" want "none"`)
}

func writeWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightFixture(t *testing.T, preflightPath string) {
	t.Helper()
	dir := filepath.Dir(preflightPath)
	boundaryPath := filepath.Join(dir, "srcboundary.json")
	writeWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryFixture(t, boundaryPath)
	if err := runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflight([]string{boundaryPath, preflightPath}); err != nil {
		t.Fatalf("write weighted admission resonance graft admission final gate observation boundary preflight fixture: %v", err)
	}
}
