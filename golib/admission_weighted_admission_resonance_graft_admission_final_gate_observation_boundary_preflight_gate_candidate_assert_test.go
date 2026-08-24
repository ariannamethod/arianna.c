package main

import (
	"path/filepath"
	"strings"
	"testing"
)

func TestAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateAssert(t *testing.T) {
	dir := t.TempDir()

	requireBoundaryAssertError(t, runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateAssert(nil), "usage: --admission-live-route-weighted-admission-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-candidate-assert REPORT")
	requireBoundaryAssertError(t, runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateAssert([]string{"candidate.json", "extra"}), "usage: --admission-live-route-weighted-admission-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-candidate-assert REPORT")
	requireBoundaryAssertError(t, runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateAssert([]string{"  "}), "weighted admission resonance graft admission final gate observation boundary preflight gate candidate path missing")
	requireBoundaryAssertError(t, runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateAssert([]string{filepath.Join(dir, "missing.json")}), "weighted admission resonance graft admission final gate observation boundary preflight gate candidate not written")

	emptyPath := filepath.Join(dir, "empty.json")
	writeWeightedReadinessFixture(t, emptyPath, "")
	requireBoundaryAssertError(t, runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateAssert([]string{emptyPath}), "weighted admission resonance graft admission final gate observation boundary preflight gate candidate not written")

	invalidPath := filepath.Join(dir, "invalid.json")
	writeWeightedReadinessFixture(t, invalidPath, "{")
	err := runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateAssert([]string{invalidPath})
	if err == nil || !strings.HasPrefix(err.Error(), "weighted admission resonance graft admission final gate observation boundary preflight gate candidate JSON invalid:") {
		t.Fatalf("expected invalid JSON error, got %v", err)
	}

	missingSchemaPath := filepath.Join(dir, "missing_schema.json")
	writeWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateFixture(t, missingSchemaPath)
	writeWeightedReadinessFixture(t, missingSchemaPath, stringsReplaceFirst(readText(t, missingSchemaPath), `"schema": "arianna.live_route_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate.v1",`, ""))
	requireBoundaryAssertError(t, runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateAssert([]string{missingSchemaPath}), "weighted admission resonance graft admission final gate observation boundary preflight gate candidate schema missing")

	badSchemaPath := filepath.Join(dir, "bad_schema.json")
	writeWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateFixture(t, badSchemaPath)
	writeWeightedReadinessFixture(t, badSchemaPath, stringsReplaceFirst(readText(t, badSchemaPath), `"schema": "arianna.live_route_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate.v1"`, `"schema": "arianna.live_route_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate.v0"`))
	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateAssert([]string{badSchemaPath}),
		`weighted admission resonance graft admission final gate observation boundary preflight gate candidate schema mismatch: got "arianna.live_route_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate.v0" want "`+admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateSchema+`"`,
	)

	validPath := filepath.Join(dir, "valid.json")
	writeWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateFixture(t, validPath)
	if err := runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateAssert([]string{validPath}); err != nil {
		t.Fatalf("valid weighted admission resonance graft admission final gate observation boundary preflight gate candidate rejected: %v", err)
	}

	badStatusPath := filepath.Join(dir, "bad_status.json")
	writeWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateFixture(t, badStatusPath)
	writeWeightedReadinessFixture(t, badStatusPath, stringsReplaceFirst(readText(t, badStatusPath), `"status": "shadow_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_blocked_dry_run"`, `"status": "open"`))
	requireBoundaryAssertError(t, runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateAssert([]string{badStatusPath}), `weighted admission resonance graft admission final gate observation boundary preflight gate candidate status mismatch: got "open" want "shadow_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_blocked_dry_run"`)

	notReadyPath := filepath.Join(dir, "not_ready.json")
	writeWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateFixture(t, notReadyPath)
	writeWeightedReadinessFixture(t, notReadyPath, stringsReplaceFirst(readText(t, notReadyPath), `"weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_ready": true`, `"weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_ready": false`))
	requireBoundaryAssertError(t, runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateAssert([]string{notReadyPath}), "weighted admission resonance graft admission final gate observation boundary preflight gate candidate weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_ready not ready")

	badActionPath := filepath.Join(dir, "bad_action.json")
	writeWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateFixture(t, badActionPath)
	writeWeightedReadinessFixture(t, badActionPath, stringsReplaceFirst(readText(t, badActionPath), `"action": "draft_weighted_resonance_shadow_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_dry_run"`, `"action": "open"`))
	requireBoundaryAssertError(t, runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateAssert([]string{badActionPath}), `weighted admission resonance graft admission final gate observation boundary preflight gate candidate action mismatch: got "open" want "draft_weighted_resonance_shadow_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_dry_run"`)

	badCandidateActionPath := filepath.Join(dir, "bad_candidate_action.json")
	writeWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateFixture(t, badCandidateActionPath)
	writeWeightedReadinessFixture(t, badCandidateActionPath, stringsReplaceFirst(readText(t, badCandidateActionPath), `"admission_final_gate_observation_boundary_preflight_gate_candidate_action": "draft_blocked_final_gate_observation_boundary_preflight_gate_candidate"`, `"admission_final_gate_observation_boundary_preflight_gate_candidate_action": "open"`))
	requireBoundaryAssertError(t, runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateAssert([]string{badCandidateActionPath}), "weighted admission resonance graft admission final gate observation boundary preflight gate candidate shape mismatch")

	openedCandidatePath := filepath.Join(dir, "opened_candidate.json")
	writeWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateFixture(t, openedCandidatePath)
	writeWeightedReadinessFixture(t, openedCandidatePath, stringsReplaceFirst(readText(t, openedCandidatePath), `"admission_final_gate_observation_boundary_preflight_gate_candidate_ready": false`, `"admission_final_gate_observation_boundary_preflight_gate_candidate_ready": true`))
	requireBoundaryAssertError(t, runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateAssert([]string{openedCandidatePath}), "weighted admission resonance graft admission final gate observation boundary preflight gate candidate opened admission_final_gate_observation_boundary_preflight_gate_candidate_ready")

	openedSourceGatePath := filepath.Join(dir, "opened_source_gate.json")
	writeWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateFixture(t, openedSourceGatePath)
	writeWeightedReadinessFixture(t, openedSourceGatePath, stringsReplaceFirst(readText(t, openedSourceGatePath), `"source_admission_final_gate_observation_boundary_preflight_gate_ready": false`, `"source_admission_final_gate_observation_boundary_preflight_gate_ready": true`))
	requireBoundaryAssertError(t, runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateAssert([]string{openedSourceGatePath}), "weighted admission resonance graft admission final gate observation boundary preflight gate candidate opened source_admission_final_gate_observation_boundary_preflight_gate_ready")

	openedPath := filepath.Join(dir, "opened.json")
	writeWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateFixture(t, openedPath)
	writeWeightedReadinessFixture(t, openedPath, stringsReplaceFirst(readText(t, openedPath), `"live_admission_enabled": false`, `"live_admission_enabled": true`))
	requireBoundaryAssertError(t, runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateAssert([]string{openedPath}), "weighted admission resonance graft admission final gate observation boundary preflight gate candidate opened live_admission_enabled")

	badSourcePath := filepath.Join(dir, "bad_source.json")
	writeWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateFixture(t, badSourcePath)
	writeWeightedReadinessFixture(t, badSourcePath, stringsReplaceFirst(readText(t, badSourcePath), `"source_schema": "arianna.live_route_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate.v1"`, `"source_schema": "arianna.live_route_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate.v0"`))
	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateAssert([]string{badSourcePath}),
		`weighted admission resonance graft admission final gate observation boundary preflight gate candidate source_schema mismatch: got "arianna.live_route_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate.v0" want "`+admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateSchema+`"`,
	)

	badSourceShapePath := filepath.Join(dir, "bad_source_shape.json")
	writeWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateFixture(t, badSourceShapePath)
	writeWeightedReadinessFixture(t, badSourceShapePath, stringsReplaceFirst(readText(t, badSourceShapePath), `"source_admission_final_gate_observation_boundary_preflight_gate_action": "gate_blocked_final_gate_observation_boundary_preflight"`, `"source_admission_final_gate_observation_boundary_preflight_gate_action": "ready"`))
	requireBoundaryAssertError(t, runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateAssert([]string{badSourceShapePath}), "weighted admission resonance graft admission final gate observation boundary preflight gate candidate source admission final gate observation boundary preflight gate shape mismatch")

	badCandidateKindPath := filepath.Join(dir, "bad_candidate_kind.json")
	writeWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateFixture(t, badCandidateKindPath)
	writeWeightedReadinessFixture(t, badCandidateKindPath, stringsReplaceFirst(readText(t, badCandidateKindPath), `"final_gate_observation_boundary_preflight_gate_candidate_kind": "blocked_final_gate_observation_boundary_preflight_gate_candidate"`, `"final_gate_observation_boundary_preflight_gate_candidate_kind": "open"`))
	requireBoundaryAssertError(t, runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateAssert([]string{badCandidateKindPath}), `weighted admission resonance graft admission final gate observation boundary preflight gate candidate candidate_kind mismatch: got "open" want "blocked_final_gate_observation_boundary_preflight_gate_candidate"`)

	badCandidateHashPath := filepath.Join(dir, "bad_candidate_hash.json")
	writeWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateFixture(t, badCandidateHashPath)
	writeWeightedReadinessFixture(t, badCandidateHashPath, stringsReplaceFirst(readText(t, badCandidateHashPath), `"admission_final_gate_observation_boundary_preflight_gate_candidate_hash": "weighted-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-candidate-`, `"admission_final_gate_observation_boundary_preflight_gate_candidate_hash": "weighted-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-candidate-bad`))
	requireBoundaryAssertError(t, runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateAssert([]string{badCandidateHashPath}), "weighted admission resonance graft admission final gate observation boundary preflight gate candidate candidate_hash mismatch")

	badBodyTargetPath := filepath.Join(dir, "bad_body_target.json")
	writeWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateFixture(t, badBodyTargetPath)
	writeWeightedReadinessFixture(t, badBodyTargetPath, stringsReplaceFirst(readText(t, badBodyTargetPath), `"body_target": "none"`, `"body_target": "live"`))
	requireBoundaryAssertError(t, runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateAssert([]string{badBodyTargetPath}), `weighted admission resonance graft admission final gate observation boundary preflight gate candidate body_target mismatch: got "live" want "none"`)
}

func writeWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateFixture(t *testing.T, candidatePath string) {
	t.Helper()
	dir := filepath.Dir(candidatePath)
	gatePath := filepath.Join(dir, "srcgate.json")
	writeWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateFixture(t, gatePath)
	if err := runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidate([]string{gatePath, candidatePath}); err != nil {
		t.Fatalf("write weighted admission resonance graft admission final gate observation boundary preflight gate candidate fixture: %v", err)
	}
}
