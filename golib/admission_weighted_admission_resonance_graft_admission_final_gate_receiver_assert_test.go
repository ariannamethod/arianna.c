package main

import (
	"path/filepath"
	"strings"
	"testing"
)

func TestAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateReceiverAssert(t *testing.T) {
	dir := t.TempDir()

	requireBoundaryAssertError(t, runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateReceiverAssert(nil), "usage: --admission-live-route-weighted-admission-resonance-graft-admission-final-gate-receiver-assert REPORT")
	requireBoundaryAssertError(t, runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateReceiverAssert([]string{"receiver.json", "extra"}), "usage: --admission-live-route-weighted-admission-resonance-graft-admission-final-gate-receiver-assert REPORT")
	requireBoundaryAssertError(t, runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateReceiverAssert([]string{"  "}), "weighted admission resonance graft admission final gate receiver path missing")
	requireBoundaryAssertError(t, runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateReceiverAssert([]string{filepath.Join(dir, "missing.json")}), "weighted admission resonance graft admission final gate receiver not written")

	emptyPath := filepath.Join(dir, "empty.json")
	writeWeightedReadinessFixture(t, emptyPath, "")
	requireBoundaryAssertError(t, runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateReceiverAssert([]string{emptyPath}), "weighted admission resonance graft admission final gate receiver not written")

	invalidPath := filepath.Join(dir, "invalid.json")
	writeWeightedReadinessFixture(t, invalidPath, "{")
	err := runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateReceiverAssert([]string{invalidPath})
	if err == nil || !strings.HasPrefix(err.Error(), "weighted admission resonance graft admission final gate receiver JSON invalid:") {
		t.Fatalf("expected invalid JSON error, got %v", err)
	}

	missingSchemaPath := filepath.Join(dir, "missing_schema.json")
	writeWeightedAdmissionResonanceGraftAdmissionFinalGateReceiverFixture(t, missingSchemaPath)
	writeWeightedReadinessFixture(t, missingSchemaPath, stringsReplaceFirst(readText(t, missingSchemaPath), `"schema": "arianna.live_route_weighted_admission_resonance_graft_admission_final_gate_receiver.v1",`, ""))
	requireBoundaryAssertError(t, runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateReceiverAssert([]string{missingSchemaPath}), "weighted admission resonance graft admission final gate receiver schema missing")

	badSchemaPath := filepath.Join(dir, "bad_schema.json")
	writeWeightedAdmissionResonanceGraftAdmissionFinalGateReceiverFixture(t, badSchemaPath)
	writeWeightedReadinessFixture(t, badSchemaPath, stringsReplaceFirst(readText(t, badSchemaPath), `"schema": "arianna.live_route_weighted_admission_resonance_graft_admission_final_gate_receiver.v1"`, `"schema": "arianna.live_route_weighted_admission_resonance_graft_admission_final_gate_receiver.v0"`))
	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateReceiverAssert([]string{badSchemaPath}),
		`weighted admission resonance graft admission final gate receiver schema mismatch: got "arianna.live_route_weighted_admission_resonance_graft_admission_final_gate_receiver.v0" want "`+admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateReceiverSchema+`"`,
	)

	validPath := filepath.Join(dir, "valid.json")
	writeWeightedAdmissionResonanceGraftAdmissionFinalGateReceiverFixture(t, validPath)
	if err := runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateReceiverAssert([]string{validPath}); err != nil {
		t.Fatalf("valid weighted admission resonance graft admission final gate receiver rejected: %v", err)
	}

	badStatusPath := filepath.Join(dir, "bad_status.json")
	writeWeightedAdmissionResonanceGraftAdmissionFinalGateReceiverFixture(t, badStatusPath)
	writeWeightedReadinessFixture(t, badStatusPath, stringsReplaceFirst(readText(t, badStatusPath), `"status": "shadow_graft_admission_final_gate_receiver_previewed_dry_run"`, `"status": "open"`))
	requireBoundaryAssertError(t, runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateReceiverAssert([]string{badStatusPath}), `weighted admission resonance graft admission final gate receiver status mismatch: got "open" want "shadow_graft_admission_final_gate_receiver_previewed_dry_run"`)

	notReadyPath := filepath.Join(dir, "not_ready.json")
	writeWeightedAdmissionResonanceGraftAdmissionFinalGateReceiverFixture(t, notReadyPath)
	writeWeightedReadinessFixture(t, notReadyPath, stringsReplaceFirst(readText(t, notReadyPath), `"weighted_admission_resonance_graft_admission_final_gate_receiver_ready": true`, `"weighted_admission_resonance_graft_admission_final_gate_receiver_ready": false`))
	requireBoundaryAssertError(t, runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateReceiverAssert([]string{notReadyPath}), "weighted admission resonance graft admission final gate receiver weighted_admission_resonance_graft_admission_final_gate_receiver_ready not ready")

	badActionPath := filepath.Join(dir, "bad_action.json")
	writeWeightedAdmissionResonanceGraftAdmissionFinalGateReceiverFixture(t, badActionPath)
	writeWeightedReadinessFixture(t, badActionPath, stringsReplaceFirst(readText(t, badActionPath), `"action": "preview_weighted_resonance_shadow_graft_admission_final_gate_receiver_dry_run"`, `"action": "open"`))
	requireBoundaryAssertError(t, runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateReceiverAssert([]string{badActionPath}), `weighted admission resonance graft admission final gate receiver action mismatch: got "open" want "preview_weighted_resonance_shadow_graft_admission_final_gate_receiver_dry_run"`)

	badReceiverActionPath := filepath.Join(dir, "bad_receiver_action.json")
	writeWeightedAdmissionResonanceGraftAdmissionFinalGateReceiverFixture(t, badReceiverActionPath)
	writeWeightedReadinessFixture(t, badReceiverActionPath, stringsReplaceFirst(readText(t, badReceiverActionPath), `"admission_final_gate_receiver_action": "preview_blocked_final_gate_receiver"`, `"admission_final_gate_receiver_action": "open"`))
	requireBoundaryAssertError(t, runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateReceiverAssert([]string{badReceiverActionPath}), "weighted admission resonance graft admission final gate receiver shape mismatch")

	openedReceiverPath := filepath.Join(dir, "opened_receiver.json")
	writeWeightedAdmissionResonanceGraftAdmissionFinalGateReceiverFixture(t, openedReceiverPath)
	writeWeightedReadinessFixture(t, openedReceiverPath, stringsReplaceFirst(readText(t, openedReceiverPath), `"admission_final_gate_receiver_ready": false`, `"admission_final_gate_receiver_ready": true`))
	requireBoundaryAssertError(t, runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateReceiverAssert([]string{openedReceiverPath}), "weighted admission resonance graft admission final gate receiver opened admission_final_gate_receiver_ready")

	openedSourceIntentPath := filepath.Join(dir, "opened_source_intent.json")
	writeWeightedAdmissionResonanceGraftAdmissionFinalGateReceiverFixture(t, openedSourceIntentPath)
	writeWeightedReadinessFixture(t, openedSourceIntentPath, stringsReplaceFirst(readText(t, openedSourceIntentPath), `"source_admission_final_gate_intent_ready": false`, `"source_admission_final_gate_intent_ready": true`))
	requireBoundaryAssertError(t, runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateReceiverAssert([]string{openedSourceIntentPath}), "weighted admission resonance graft admission final gate receiver opened source_admission_final_gate_intent_ready")

	openedPath := filepath.Join(dir, "opened.json")
	writeWeightedAdmissionResonanceGraftAdmissionFinalGateReceiverFixture(t, openedPath)
	writeWeightedReadinessFixture(t, openedPath, stringsReplaceFirst(readText(t, openedPath), `"live_admission_enabled": false`, `"live_admission_enabled": true`))
	requireBoundaryAssertError(t, runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateReceiverAssert([]string{openedPath}), "weighted admission resonance graft admission final gate receiver opened live_admission_enabled")

	badSourcePath := filepath.Join(dir, "bad_source.json")
	writeWeightedAdmissionResonanceGraftAdmissionFinalGateReceiverFixture(t, badSourcePath)
	writeWeightedReadinessFixture(t, badSourcePath, stringsReplaceFirst(readText(t, badSourcePath), `"source_schema": "arianna.live_route_weighted_admission_resonance_graft_admission_final_gate_intent.v1"`, `"source_schema": "arianna.live_route_weighted_admission_resonance_graft_admission_final_gate_intent.v0"`))
	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateReceiverAssert([]string{badSourcePath}),
		`weighted admission resonance graft admission final gate receiver source_schema mismatch: got "arianna.live_route_weighted_admission_resonance_graft_admission_final_gate_intent.v0" want "`+admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateIntentSchema+`"`,
	)

	badSourceShapePath := filepath.Join(dir, "bad_source_shape.json")
	writeWeightedAdmissionResonanceGraftAdmissionFinalGateReceiverFixture(t, badSourceShapePath)
	writeWeightedReadinessFixture(t, badSourceShapePath, stringsReplaceFirst(readText(t, badSourceShapePath), `"source_admission_final_gate_intent_action": "draft_blocked_final_gate_intent"`, `"source_admission_final_gate_intent_action": "ready"`))
	requireBoundaryAssertError(t, runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateReceiverAssert([]string{badSourceShapePath}), "weighted admission resonance graft admission final gate receiver source admission final gate intent shape mismatch")

	badReceiverPath := filepath.Join(dir, "bad_receiver.json")
	writeWeightedAdmissionResonanceGraftAdmissionFinalGateReceiverFixture(t, badReceiverPath)
	writeWeightedReadinessFixture(t, badReceiverPath, stringsReplaceFirst(readText(t, badReceiverPath), `"final_gate_receiver": "resonance"`, `"final_gate_receiver": "janus"`))
	requireBoundaryAssertError(t, runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateReceiverAssert([]string{badReceiverPath}), `weighted admission resonance graft admission final gate receiver receiver mismatch: got "janus" want "resonance"`)

	badPreHashPath := filepath.Join(dir, "bad_pre_hash.json")
	writeWeightedAdmissionResonanceGraftAdmissionFinalGateReceiverFixture(t, badPreHashPath)
	writeWeightedReadinessFixture(t, badPreHashPath, stringsReplaceFirst(readText(t, badPreHashPath), `"admission_final_gate_receiver_pre_state_hash": "weighted-resonance-graft-admission-final-gate-receiver-pre-`, `"admission_final_gate_receiver_pre_state_hash": "weighted-resonance-graft-admission-final-gate-receiver-pre-bad`))
	requireBoundaryAssertError(t, runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateReceiverAssert([]string{badPreHashPath}), "weighted admission resonance graft admission final gate receiver pre_state_hash mismatch")

	badBodyTargetPath := filepath.Join(dir, "bad_body_target.json")
	writeWeightedAdmissionResonanceGraftAdmissionFinalGateReceiverFixture(t, badBodyTargetPath)
	writeWeightedReadinessFixture(t, badBodyTargetPath, stringsReplaceFirst(readText(t, badBodyTargetPath), `"body_target": "none"`, `"body_target": "live"`))
	requireBoundaryAssertError(t, runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateReceiverAssert([]string{badBodyTargetPath}), `weighted admission resonance graft admission final gate receiver body_target mismatch: got "live" want "none"`)
}

func writeWeightedAdmissionResonanceGraftAdmissionFinalGateReceiverFixture(t *testing.T, receiverPath string) {
	t.Helper()
	dir := filepath.Dir(receiverPath)
	intentPath := filepath.Join(dir, "srcintent.json")
	writeWeightedAdmissionResonanceGraftAdmissionFinalGateIntentFixture(t, intentPath)
	if err := runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateReceiver([]string{intentPath, receiverPath}); err != nil {
		t.Fatalf("write weighted admission resonance graft admission final gate receiver fixture: %v", err)
	}
}
