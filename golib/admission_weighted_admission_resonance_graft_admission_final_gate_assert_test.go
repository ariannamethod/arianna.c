package main

import (
	"path/filepath"
	"strings"
	"testing"
)

func TestAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateAssert(t *testing.T) {
	dir := t.TempDir()

	requireBoundaryAssertError(t, runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateAssert(nil), "usage: --admission-live-route-weighted-admission-resonance-graft-admission-final-gate-assert REPORT")
	requireBoundaryAssertError(t, runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateAssert([]string{"final_gate.json", "extra"}), "usage: --admission-live-route-weighted-admission-resonance-graft-admission-final-gate-assert REPORT")
	requireBoundaryAssertError(t, runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateAssert([]string{"  "}), "weighted admission resonance graft admission final gate path missing")
	requireBoundaryAssertError(t, runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateAssert([]string{filepath.Join(dir, "missing.json")}), "weighted admission resonance graft admission final gate not written")

	emptyPath := filepath.Join(dir, "empty.json")
	writeWeightedReadinessFixture(t, emptyPath, "")
	requireBoundaryAssertError(t, runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateAssert([]string{emptyPath}), "weighted admission resonance graft admission final gate not written")

	invalidPath := filepath.Join(dir, "invalid.json")
	writeWeightedReadinessFixture(t, invalidPath, "{")
	err := runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateAssert([]string{invalidPath})
	if err == nil || !strings.HasPrefix(err.Error(), "weighted admission resonance graft admission final gate JSON invalid:") {
		t.Fatalf("expected invalid JSON error, got %v", err)
	}

	missingSchemaPath := filepath.Join(dir, "missing_schema.json")
	writeWeightedAdmissionResonanceGraftAdmissionFinalGateFixture(t, missingSchemaPath)
	writeWeightedReadinessFixture(t, missingSchemaPath, stringsReplaceFirst(readText(t, missingSchemaPath), `"schema": "arianna.live_route_weighted_admission_resonance_graft_admission_final_gate.v1",`, ""))
	requireBoundaryAssertError(t, runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateAssert([]string{missingSchemaPath}), "weighted admission resonance graft admission final gate schema missing")

	badSchemaPath := filepath.Join(dir, "bad_schema.json")
	writeWeightedAdmissionResonanceGraftAdmissionFinalGateFixture(t, badSchemaPath)
	writeWeightedReadinessFixture(t, badSchemaPath, stringsReplaceFirst(readText(t, badSchemaPath), `"schema": "arianna.live_route_weighted_admission_resonance_graft_admission_final_gate.v1"`, `"schema": "arianna.live_route_weighted_admission_resonance_graft_admission_final_gate.v0"`))
	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateAssert([]string{badSchemaPath}),
		`weighted admission resonance graft admission final gate schema mismatch: got "arianna.live_route_weighted_admission_resonance_graft_admission_final_gate.v0" want "`+admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateSchema+`"`,
	)

	validPath := filepath.Join(dir, "valid.json")
	writeWeightedAdmissionResonanceGraftAdmissionFinalGateFixture(t, validPath)
	if err := runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateAssert([]string{validPath}); err != nil {
		t.Fatalf("valid weighted admission resonance graft admission final gate rejected: %v", err)
	}

	badStatusPath := filepath.Join(dir, "bad_status.json")
	writeWeightedAdmissionResonanceGraftAdmissionFinalGateFixture(t, badStatusPath)
	writeWeightedReadinessFixture(t, badStatusPath, stringsReplaceFirst(readText(t, badStatusPath), `"status": "shadow_graft_admission_final_gate_blocked_dry_run"`, `"status": "open"`))
	requireBoundaryAssertError(t, runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateAssert([]string{badStatusPath}), `weighted admission resonance graft admission final gate status mismatch: got "open" want "shadow_graft_admission_final_gate_blocked_dry_run"`)

	notReadyPath := filepath.Join(dir, "not_ready.json")
	writeWeightedAdmissionResonanceGraftAdmissionFinalGateFixture(t, notReadyPath)
	writeWeightedReadinessFixture(t, notReadyPath, stringsReplaceFirst(readText(t, notReadyPath), `"weighted_admission_resonance_graft_admission_final_gate_ready": true`, `"weighted_admission_resonance_graft_admission_final_gate_ready": false`))
	requireBoundaryAssertError(t, runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateAssert([]string{notReadyPath}), "weighted admission resonance graft admission final gate weighted_admission_resonance_graft_admission_final_gate_ready not ready")

	badActionPath := filepath.Join(dir, "bad_action.json")
	writeWeightedAdmissionResonanceGraftAdmissionFinalGateFixture(t, badActionPath)
	writeWeightedReadinessFixture(t, badActionPath, stringsReplaceFirst(readText(t, badActionPath), `"action": "block_weighted_resonance_shadow_graft_admission_seal_blocked_dry_run"`, `"action": "open"`))
	requireBoundaryAssertError(t, runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateAssert([]string{badActionPath}), `weighted admission resonance graft admission final gate action mismatch: got "open" want "block_weighted_resonance_shadow_graft_admission_seal_blocked_dry_run"`)

	badFinalGateActionPath := filepath.Join(dir, "bad_final_gate_action.json")
	writeWeightedAdmissionResonanceGraftAdmissionFinalGateFixture(t, badFinalGateActionPath)
	writeWeightedReadinessFixture(t, badFinalGateActionPath, stringsReplaceFirst(readText(t, badFinalGateActionPath), `"admission_final_gate_action": "reject_blocked_admission_seal"`, `"admission_final_gate_action": "open"`))
	requireBoundaryAssertError(t, runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateAssert([]string{badFinalGateActionPath}), "weighted admission resonance graft admission final gate shape mismatch")

	openedFinalGatePath := filepath.Join(dir, "opened_final_gate.json")
	writeWeightedAdmissionResonanceGraftAdmissionFinalGateFixture(t, openedFinalGatePath)
	writeWeightedReadinessFixture(t, openedFinalGatePath, stringsReplaceFirst(readText(t, openedFinalGatePath), `"admission_final_gate_ready": false`, `"admission_final_gate_ready": true`))
	requireBoundaryAssertError(t, runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateAssert([]string{openedFinalGatePath}), "weighted admission resonance graft admission final gate opened admission_final_gate_ready")

	openedSourceSealPath := filepath.Join(dir, "opened_source_seal.json")
	writeWeightedAdmissionResonanceGraftAdmissionFinalGateFixture(t, openedSourceSealPath)
	writeWeightedReadinessFixture(t, openedSourceSealPath, stringsReplaceFirst(readText(t, openedSourceSealPath), `"source_admission_seal_ready": false`, `"source_admission_seal_ready": true`))
	requireBoundaryAssertError(t, runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateAssert([]string{openedSourceSealPath}), "weighted admission resonance graft admission final gate opened source_admission_seal_ready")

	openedPath := filepath.Join(dir, "opened.json")
	writeWeightedAdmissionResonanceGraftAdmissionFinalGateFixture(t, openedPath)
	writeWeightedReadinessFixture(t, openedPath, stringsReplaceFirst(readText(t, openedPath), `"live_admission_enabled": false`, `"live_admission_enabled": true`))
	requireBoundaryAssertError(t, runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateAssert([]string{openedPath}), "weighted admission resonance graft admission final gate opened live_admission_enabled")

	badSourcePath := filepath.Join(dir, "bad_source.json")
	writeWeightedAdmissionResonanceGraftAdmissionFinalGateFixture(t, badSourcePath)
	writeWeightedReadinessFixture(t, badSourcePath, stringsReplaceFirst(readText(t, badSourcePath), `"source_schema": "arianna.live_route_weighted_admission_resonance_graft_admission_seal.v1"`, `"source_schema": "arianna.live_route_weighted_admission_resonance_graft_admission_seal.v0"`))
	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateAssert([]string{badSourcePath}),
		`weighted admission resonance graft admission final gate source_schema mismatch: got "arianna.live_route_weighted_admission_resonance_graft_admission_seal.v0" want "`+admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionSealSchema+`"`,
	)

	badSourceSealShapePath := filepath.Join(dir, "bad_source_seal_shape.json")
	writeWeightedAdmissionResonanceGraftAdmissionFinalGateFixture(t, badSourceSealShapePath)
	writeWeightedReadinessFixture(t, badSourceSealShapePath, stringsReplaceFirst(readText(t, badSourceSealShapePath), `"source_admission_seal_action": "seal_blocked_admission_authority"`, `"source_admission_seal_action": "ready"`))
	requireBoundaryAssertError(t, runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateAssert([]string{badSourceSealShapePath}), "weighted admission resonance graft admission final gate source admission seal shape mismatch")

	badFinalGateHashPath := filepath.Join(dir, "bad_final_gate_hash.json")
	writeWeightedAdmissionResonanceGraftAdmissionFinalGateFixture(t, badFinalGateHashPath)
	writeWeightedReadinessFixture(t, badFinalGateHashPath, stringsReplaceFirst(readText(t, badFinalGateHashPath), `"admission_final_gate_hash": "weighted-resonance-graft-admission-final-gate-`, `"admission_final_gate_hash": "weighted-resonance-graft-admission-final-gate-bad`))
	requireBoundaryAssertError(t, runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateAssert([]string{badFinalGateHashPath}), "weighted admission resonance graft admission final gate admission_final_gate_hash mismatch")

	badBodyTargetPath := filepath.Join(dir, "bad_body_target.json")
	writeWeightedAdmissionResonanceGraftAdmissionFinalGateFixture(t, badBodyTargetPath)
	writeWeightedReadinessFixture(t, badBodyTargetPath, stringsReplaceFirst(readText(t, badBodyTargetPath), `"body_target": "none"`, `"body_target": "live"`))
	requireBoundaryAssertError(t, runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateAssert([]string{badBodyTargetPath}), `weighted admission resonance graft admission final gate body_target mismatch: got "live" want "none"`)
}

func writeWeightedAdmissionResonanceGraftAdmissionFinalGateFixture(t *testing.T, finalGatePath string) {
	t.Helper()
	dir := filepath.Dir(finalGatePath)
	sealPath := filepath.Join(dir, "srcseal.json")
	writeWeightedAdmissionResonanceGraftAdmissionSealFixture(t, sealPath)
	if err := runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGate([]string{sealPath, finalGatePath}); err != nil {
		t.Fatalf("write weighted admission resonance graft admission final gate fixture: %v", err)
	}
}
