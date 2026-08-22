package main

import (
	"path/filepath"
	"strings"
	"testing"
)

func TestAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateIntentAssert(t *testing.T) {
	dir := t.TempDir()

	requireBoundaryAssertError(t, runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateIntentAssert(nil), "usage: --admission-live-route-weighted-admission-resonance-graft-admission-final-gate-intent-assert REPORT")
	requireBoundaryAssertError(t, runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateIntentAssert([]string{"intent.json", "extra"}), "usage: --admission-live-route-weighted-admission-resonance-graft-admission-final-gate-intent-assert REPORT")
	requireBoundaryAssertError(t, runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateIntentAssert([]string{"  "}), "weighted admission resonance graft admission final gate intent path missing")
	requireBoundaryAssertError(t, runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateIntentAssert([]string{filepath.Join(dir, "missing.json")}), "weighted admission resonance graft admission final gate intent not written")

	emptyPath := filepath.Join(dir, "empty.json")
	writeWeightedReadinessFixture(t, emptyPath, "")
	requireBoundaryAssertError(t, runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateIntentAssert([]string{emptyPath}), "weighted admission resonance graft admission final gate intent not written")

	invalidPath := filepath.Join(dir, "invalid.json")
	writeWeightedReadinessFixture(t, invalidPath, "{")
	err := runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateIntentAssert([]string{invalidPath})
	if err == nil || !strings.HasPrefix(err.Error(), "weighted admission resonance graft admission final gate intent JSON invalid:") {
		t.Fatalf("expected invalid JSON error, got %v", err)
	}

	missingSchemaPath := filepath.Join(dir, "missing_schema.json")
	writeWeightedAdmissionResonanceGraftAdmissionFinalGateIntentFixture(t, missingSchemaPath)
	writeWeightedReadinessFixture(t, missingSchemaPath, stringsReplaceFirst(readText(t, missingSchemaPath), `"schema": "arianna.live_route_weighted_admission_resonance_graft_admission_final_gate_intent.v1",`, ""))
	requireBoundaryAssertError(t, runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateIntentAssert([]string{missingSchemaPath}), "weighted admission resonance graft admission final gate intent schema missing")

	badSchemaPath := filepath.Join(dir, "bad_schema.json")
	writeWeightedAdmissionResonanceGraftAdmissionFinalGateIntentFixture(t, badSchemaPath)
	writeWeightedReadinessFixture(t, badSchemaPath, stringsReplaceFirst(readText(t, badSchemaPath), `"schema": "arianna.live_route_weighted_admission_resonance_graft_admission_final_gate_intent.v1"`, `"schema": "arianna.live_route_weighted_admission_resonance_graft_admission_final_gate_intent.v0"`))
	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateIntentAssert([]string{badSchemaPath}),
		`weighted admission resonance graft admission final gate intent schema mismatch: got "arianna.live_route_weighted_admission_resonance_graft_admission_final_gate_intent.v0" want "`+admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateIntentSchema+`"`,
	)

	validPath := filepath.Join(dir, "valid.json")
	writeWeightedAdmissionResonanceGraftAdmissionFinalGateIntentFixture(t, validPath)
	if err := runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateIntentAssert([]string{validPath}); err != nil {
		t.Fatalf("valid weighted admission resonance graft admission final gate intent rejected: %v", err)
	}

	badStatusPath := filepath.Join(dir, "bad_status.json")
	writeWeightedAdmissionResonanceGraftAdmissionFinalGateIntentFixture(t, badStatusPath)
	writeWeightedReadinessFixture(t, badStatusPath, stringsReplaceFirst(readText(t, badStatusPath), `"status": "shadow_graft_admission_final_gate_intent_blocked_dry_run"`, `"status": "open"`))
	requireBoundaryAssertError(t, runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateIntentAssert([]string{badStatusPath}), `weighted admission resonance graft admission final gate intent status mismatch: got "open" want "shadow_graft_admission_final_gate_intent_blocked_dry_run"`)

	notReadyPath := filepath.Join(dir, "not_ready.json")
	writeWeightedAdmissionResonanceGraftAdmissionFinalGateIntentFixture(t, notReadyPath)
	writeWeightedReadinessFixture(t, notReadyPath, stringsReplaceFirst(readText(t, notReadyPath), `"weighted_admission_resonance_graft_admission_final_gate_intent_ready": true`, `"weighted_admission_resonance_graft_admission_final_gate_intent_ready": false`))
	requireBoundaryAssertError(t, runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateIntentAssert([]string{notReadyPath}), "weighted admission resonance graft admission final gate intent weighted_admission_resonance_graft_admission_final_gate_intent_ready not ready")

	badActionPath := filepath.Join(dir, "bad_action.json")
	writeWeightedAdmissionResonanceGraftAdmissionFinalGateIntentFixture(t, badActionPath)
	writeWeightedReadinessFixture(t, badActionPath, stringsReplaceFirst(readText(t, badActionPath), `"action": "draft_weighted_resonance_shadow_graft_admission_final_gate_intent_dry_run"`, `"action": "open"`))
	requireBoundaryAssertError(t, runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateIntentAssert([]string{badActionPath}), `weighted admission resonance graft admission final gate intent action mismatch: got "open" want "draft_weighted_resonance_shadow_graft_admission_final_gate_intent_dry_run"`)

	badIntentActionPath := filepath.Join(dir, "bad_intent_action.json")
	writeWeightedAdmissionResonanceGraftAdmissionFinalGateIntentFixture(t, badIntentActionPath)
	writeWeightedReadinessFixture(t, badIntentActionPath, stringsReplaceFirst(readText(t, badIntentActionPath), `"admission_final_gate_intent_action": "draft_blocked_final_gate_intent"`, `"admission_final_gate_intent_action": "open"`))
	requireBoundaryAssertError(t, runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateIntentAssert([]string{badIntentActionPath}), "weighted admission resonance graft admission final gate intent shape mismatch")

	openedIntentPath := filepath.Join(dir, "opened_intent.json")
	writeWeightedAdmissionResonanceGraftAdmissionFinalGateIntentFixture(t, openedIntentPath)
	writeWeightedReadinessFixture(t, openedIntentPath, stringsReplaceFirst(readText(t, openedIntentPath), `"admission_final_gate_intent_ready": false`, `"admission_final_gate_intent_ready": true`))
	requireBoundaryAssertError(t, runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateIntentAssert([]string{openedIntentPath}), "weighted admission resonance graft admission final gate intent opened admission_final_gate_intent_ready")

	openedSourceFinalGatePath := filepath.Join(dir, "opened_source_final_gate.json")
	writeWeightedAdmissionResonanceGraftAdmissionFinalGateIntentFixture(t, openedSourceFinalGatePath)
	writeWeightedReadinessFixture(t, openedSourceFinalGatePath, stringsReplaceFirst(readText(t, openedSourceFinalGatePath), `"source_admission_final_gate_ready": false`, `"source_admission_final_gate_ready": true`))
	requireBoundaryAssertError(t, runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateIntentAssert([]string{openedSourceFinalGatePath}), "weighted admission resonance graft admission final gate intent opened source_admission_final_gate_ready")

	openedPath := filepath.Join(dir, "opened.json")
	writeWeightedAdmissionResonanceGraftAdmissionFinalGateIntentFixture(t, openedPath)
	writeWeightedReadinessFixture(t, openedPath, stringsReplaceFirst(readText(t, openedPath), `"live_admission_enabled": false`, `"live_admission_enabled": true`))
	requireBoundaryAssertError(t, runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateIntentAssert([]string{openedPath}), "weighted admission resonance graft admission final gate intent opened live_admission_enabled")

	badSourcePath := filepath.Join(dir, "bad_source.json")
	writeWeightedAdmissionResonanceGraftAdmissionFinalGateIntentFixture(t, badSourcePath)
	writeWeightedReadinessFixture(t, badSourcePath, stringsReplaceFirst(readText(t, badSourcePath), `"source_schema": "arianna.live_route_weighted_admission_resonance_graft_admission_final_gate.v1"`, `"source_schema": "arianna.live_route_weighted_admission_resonance_graft_admission_final_gate.v0"`))
	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateIntentAssert([]string{badSourcePath}),
		`weighted admission resonance graft admission final gate intent source_schema mismatch: got "arianna.live_route_weighted_admission_resonance_graft_admission_final_gate.v0" want "`+admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateSchema+`"`,
	)

	badSourceShapePath := filepath.Join(dir, "bad_source_shape.json")
	writeWeightedAdmissionResonanceGraftAdmissionFinalGateIntentFixture(t, badSourceShapePath)
	writeWeightedReadinessFixture(t, badSourceShapePath, stringsReplaceFirst(readText(t, badSourceShapePath), `"source_admission_final_gate_action": "reject_blocked_admission_seal"`, `"source_admission_final_gate_action": "ready"`))
	requireBoundaryAssertError(t, runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateIntentAssert([]string{badSourceShapePath}), "weighted admission resonance graft admission final gate intent source admission final gate shape mismatch")

	badReceiverPath := filepath.Join(dir, "bad_receiver.json")
	writeWeightedAdmissionResonanceGraftAdmissionFinalGateIntentFixture(t, badReceiverPath)
	writeWeightedReadinessFixture(t, badReceiverPath, stringsReplaceFirst(readText(t, badReceiverPath), `"final_gate_intent_receiver": "resonance"`, `"final_gate_intent_receiver": "janus"`))
	requireBoundaryAssertError(t, runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateIntentAssert([]string{badReceiverPath}), `weighted admission resonance graft admission final gate intent receiver mismatch: got "janus" want "resonance"`)

	badIntentHashPath := filepath.Join(dir, "bad_intent_hash.json")
	writeWeightedAdmissionResonanceGraftAdmissionFinalGateIntentFixture(t, badIntentHashPath)
	writeWeightedReadinessFixture(t, badIntentHashPath, stringsReplaceFirst(readText(t, badIntentHashPath), `"admission_final_gate_intent_hash": "weighted-resonance-graft-admission-final-gate-intent-`, `"admission_final_gate_intent_hash": "weighted-resonance-graft-admission-final-gate-intent-bad`))
	requireBoundaryAssertError(t, runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateIntentAssert([]string{badIntentHashPath}), "weighted admission resonance graft admission final gate intent admission_final_gate_intent_hash mismatch")

	badBodyTargetPath := filepath.Join(dir, "bad_body_target.json")
	writeWeightedAdmissionResonanceGraftAdmissionFinalGateIntentFixture(t, badBodyTargetPath)
	writeWeightedReadinessFixture(t, badBodyTargetPath, stringsReplaceFirst(readText(t, badBodyTargetPath), `"body_target": "none"`, `"body_target": "live"`))
	requireBoundaryAssertError(t, runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateIntentAssert([]string{badBodyTargetPath}), `weighted admission resonance graft admission final gate intent body_target mismatch: got "live" want "none"`)
}

func writeWeightedAdmissionResonanceGraftAdmissionFinalGateIntentFixture(t *testing.T, intentPath string) {
	t.Helper()
	dir := filepath.Dir(intentPath)
	finalGatePath := filepath.Join(dir, "srcfinalgate.json")
	writeWeightedAdmissionResonanceGraftAdmissionFinalGateFixture(t, finalGatePath)
	if err := runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateIntent([]string{finalGatePath, intentPath}); err != nil {
		t.Fatalf("write weighted admission resonance graft admission final gate intent fixture: %v", err)
	}
}
