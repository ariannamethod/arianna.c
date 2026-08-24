package main

import (
	"path/filepath"
	"strings"
	"testing"
)

func TestAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateAssert(t *testing.T) {
	dir := t.TempDir()

	requireBoundaryAssertError(t, runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateAssert(nil), "usage: --admission-live-route-weighted-admission-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-assert REPORT")
	requireBoundaryAssertError(t, runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateAssert([]string{"gate.json", "extra"}), "usage: --admission-live-route-weighted-admission-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-assert REPORT")
	requireBoundaryAssertError(t, runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateAssert([]string{"  "}), "weighted admission resonance graft admission final gate observation boundary preflight gate path missing")
	requireBoundaryAssertError(t, runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateAssert([]string{filepath.Join(dir, "missing.json")}), "weighted admission resonance graft admission final gate observation boundary preflight gate not written")

	emptyPath := filepath.Join(dir, "empty.json")
	writeWeightedReadinessFixture(t, emptyPath, "")
	requireBoundaryAssertError(t, runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateAssert([]string{emptyPath}), "weighted admission resonance graft admission final gate observation boundary preflight gate not written")

	invalidPath := filepath.Join(dir, "invalid.json")
	writeWeightedReadinessFixture(t, invalidPath, "{")
	err := runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateAssert([]string{invalidPath})
	if err == nil || !strings.HasPrefix(err.Error(), "weighted admission resonance graft admission final gate observation boundary preflight gate JSON invalid:") {
		t.Fatalf("expected invalid JSON error, got %v", err)
	}

	missingSchemaPath := filepath.Join(dir, "missing_schema.json")
	writeWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateFixture(t, missingSchemaPath)
	writeWeightedReadinessFixture(t, missingSchemaPath, stringsReplaceFirst(readText(t, missingSchemaPath), `"schema": "arianna.live_route_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate.v1",`, ""))
	requireBoundaryAssertError(t, runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateAssert([]string{missingSchemaPath}), "weighted admission resonance graft admission final gate observation boundary preflight gate schema missing")

	badSchemaPath := filepath.Join(dir, "bad_schema.json")
	writeWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateFixture(t, badSchemaPath)
	writeWeightedReadinessFixture(t, badSchemaPath, stringsReplaceFirst(readText(t, badSchemaPath), `"schema": "arianna.live_route_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate.v1"`, `"schema": "arianna.live_route_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate.v0"`))
	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateAssert([]string{badSchemaPath}),
		`weighted admission resonance graft admission final gate observation boundary preflight gate schema mismatch: got "arianna.live_route_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate.v0" want "`+admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateSchema+`"`,
	)

	validPath := filepath.Join(dir, "valid.json")
	writeWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateFixture(t, validPath)
	if err := runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateAssert([]string{validPath}); err != nil {
		t.Fatalf("valid weighted admission resonance graft admission final gate observation boundary preflight gate rejected: %v", err)
	}

	badStatusPath := filepath.Join(dir, "bad_status.json")
	writeWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateFixture(t, badStatusPath)
	writeWeightedReadinessFixture(t, badStatusPath, stringsReplaceFirst(readText(t, badStatusPath), `"status": "shadow_graft_admission_final_gate_observation_boundary_preflight_gate_blocked_dry_run"`, `"status": "open"`))
	requireBoundaryAssertError(t, runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateAssert([]string{badStatusPath}), `weighted admission resonance graft admission final gate observation boundary preflight gate status mismatch: got "open" want "shadow_graft_admission_final_gate_observation_boundary_preflight_gate_blocked_dry_run"`)

	notReadyPath := filepath.Join(dir, "not_ready.json")
	writeWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateFixture(t, notReadyPath)
	writeWeightedReadinessFixture(t, notReadyPath, stringsReplaceFirst(readText(t, notReadyPath), `"weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_ready": true`, `"weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_ready": false`))
	requireBoundaryAssertError(t, runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateAssert([]string{notReadyPath}), "weighted admission resonance graft admission final gate observation boundary preflight gate weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_ready not ready")

	badActionPath := filepath.Join(dir, "bad_action.json")
	writeWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateFixture(t, badActionPath)
	writeWeightedReadinessFixture(t, badActionPath, stringsReplaceFirst(readText(t, badActionPath), `"action": "gate_weighted_resonance_shadow_graft_admission_final_gate_observation_boundary_preflight_dry_run"`, `"action": "open"`))
	requireBoundaryAssertError(t, runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateAssert([]string{badActionPath}), `weighted admission resonance graft admission final gate observation boundary preflight gate action mismatch: got "open" want "gate_weighted_resonance_shadow_graft_admission_final_gate_observation_boundary_preflight_dry_run"`)

	badGateActionPath := filepath.Join(dir, "bad_gate_action.json")
	writeWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateFixture(t, badGateActionPath)
	writeWeightedReadinessFixture(t, badGateActionPath, stringsReplaceFirst(readText(t, badGateActionPath), `"admission_final_gate_observation_boundary_preflight_gate_action": "gate_blocked_final_gate_observation_boundary_preflight"`, `"admission_final_gate_observation_boundary_preflight_gate_action": "open"`))
	requireBoundaryAssertError(t, runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateAssert([]string{badGateActionPath}), "weighted admission resonance graft admission final gate observation boundary preflight gate shape mismatch")

	openedGatePath := filepath.Join(dir, "opened_gate.json")
	writeWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateFixture(t, openedGatePath)
	writeWeightedReadinessFixture(t, openedGatePath, stringsReplaceFirst(readText(t, openedGatePath), `"admission_final_gate_observation_boundary_preflight_gate_ready": false`, `"admission_final_gate_observation_boundary_preflight_gate_ready": true`))
	requireBoundaryAssertError(t, runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateAssert([]string{openedGatePath}), "weighted admission resonance graft admission final gate observation boundary preflight gate opened admission_final_gate_observation_boundary_preflight_gate_ready")

	openedSourcePreflightPath := filepath.Join(dir, "opened_source_preflight.json")
	writeWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateFixture(t, openedSourcePreflightPath)
	writeWeightedReadinessFixture(t, openedSourcePreflightPath, stringsReplaceFirst(readText(t, openedSourcePreflightPath), `"source_admission_final_gate_observation_boundary_preflight_ready": false`, `"source_admission_final_gate_observation_boundary_preflight_ready": true`))
	requireBoundaryAssertError(t, runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateAssert([]string{openedSourcePreflightPath}), "weighted admission resonance graft admission final gate observation boundary preflight gate opened source_admission_final_gate_observation_boundary_preflight_ready")

	openedPath := filepath.Join(dir, "opened.json")
	writeWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateFixture(t, openedPath)
	writeWeightedReadinessFixture(t, openedPath, stringsReplaceFirst(readText(t, openedPath), `"live_admission_enabled": false`, `"live_admission_enabled": true`))
	requireBoundaryAssertError(t, runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateAssert([]string{openedPath}), "weighted admission resonance graft admission final gate observation boundary preflight gate opened live_admission_enabled")

	badSourcePath := filepath.Join(dir, "bad_source.json")
	writeWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateFixture(t, badSourcePath)
	writeWeightedReadinessFixture(t, badSourcePath, stringsReplaceFirst(readText(t, badSourcePath), `"source_schema": "arianna.live_route_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight.v1"`, `"source_schema": "arianna.live_route_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight.v0"`))
	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateAssert([]string{badSourcePath}),
		`weighted admission resonance graft admission final gate observation boundary preflight gate source_schema mismatch: got "arianna.live_route_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight.v0" want "`+admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightSchema+`"`,
	)

	badSourceShapePath := filepath.Join(dir, "bad_source_shape.json")
	writeWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateFixture(t, badSourceShapePath)
	writeWeightedReadinessFixture(t, badSourceShapePath, stringsReplaceFirst(readText(t, badSourceShapePath), `"source_admission_final_gate_observation_boundary_preflight_action": "check_blocked_final_gate_observation_boundary_preflight"`, `"source_admission_final_gate_observation_boundary_preflight_action": "ready"`))
	requireBoundaryAssertError(t, runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateAssert([]string{badSourceShapePath}), "weighted admission resonance graft admission final gate observation boundary preflight gate source admission final gate observation boundary preflight shape mismatch")

	badGateKindPath := filepath.Join(dir, "bad_gate_kind.json")
	writeWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateFixture(t, badGateKindPath)
	writeWeightedReadinessFixture(t, badGateKindPath, stringsReplaceFirst(readText(t, badGateKindPath), `"final_gate_observation_boundary_preflight_gate_kind": "blocked_final_gate_observation_boundary_preflight_gate"`, `"final_gate_observation_boundary_preflight_gate_kind": "open"`))
	requireBoundaryAssertError(t, runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateAssert([]string{badGateKindPath}), `weighted admission resonance graft admission final gate observation boundary preflight gate gate_kind mismatch: got "open" want "blocked_final_gate_observation_boundary_preflight_gate"`)

	badGateHashPath := filepath.Join(dir, "bad_gate_hash.json")
	writeWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateFixture(t, badGateHashPath)
	writeWeightedReadinessFixture(t, badGateHashPath, stringsReplaceFirst(readText(t, badGateHashPath), `"admission_final_gate_observation_boundary_preflight_gate_hash": "weighted-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-`, `"admission_final_gate_observation_boundary_preflight_gate_hash": "weighted-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-bad`))
	requireBoundaryAssertError(t, runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateAssert([]string{badGateHashPath}), "weighted admission resonance graft admission final gate observation boundary preflight gate gate_hash mismatch")

	badBodyTargetPath := filepath.Join(dir, "bad_body_target.json")
	writeWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateFixture(t, badBodyTargetPath)
	writeWeightedReadinessFixture(t, badBodyTargetPath, stringsReplaceFirst(readText(t, badBodyTargetPath), `"body_target": "none"`, `"body_target": "live"`))
	requireBoundaryAssertError(t, runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateAssert([]string{badBodyTargetPath}), `weighted admission resonance graft admission final gate observation boundary preflight gate body_target mismatch: got "live" want "none"`)
}

func writeWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateFixture(t *testing.T, gatePath string) {
	t.Helper()
	dir := filepath.Dir(gatePath)
	preflightPath := filepath.Join(dir, "srcpreflight.json")
	writeWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightFixture(t, preflightPath)
	if err := runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGate([]string{preflightPath, gatePath}); err != nil {
		t.Fatalf("write weighted admission resonance graft admission final gate observation boundary preflight gate fixture: %v", err)
	}
}
