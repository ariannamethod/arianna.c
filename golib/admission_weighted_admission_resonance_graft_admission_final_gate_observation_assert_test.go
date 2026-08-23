package main

import (
	"path/filepath"
	"strings"
	"testing"
)

func TestAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationAssert(t *testing.T) {
	dir := t.TempDir()

	requireBoundaryAssertError(t, runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationAssert(nil), "usage: --admission-live-route-weighted-admission-resonance-graft-admission-final-gate-observation-assert REPORT")
	requireBoundaryAssertError(t, runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationAssert([]string{"observation.json", "extra"}), "usage: --admission-live-route-weighted-admission-resonance-graft-admission-final-gate-observation-assert REPORT")
	requireBoundaryAssertError(t, runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationAssert([]string{"  "}), "weighted admission resonance graft admission final gate observation path missing")
	requireBoundaryAssertError(t, runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationAssert([]string{filepath.Join(dir, "missing.json")}), "weighted admission resonance graft admission final gate observation not written")

	emptyPath := filepath.Join(dir, "empty.json")
	writeWeightedReadinessFixture(t, emptyPath, "")
	requireBoundaryAssertError(t, runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationAssert([]string{emptyPath}), "weighted admission resonance graft admission final gate observation not written")

	invalidPath := filepath.Join(dir, "invalid.json")
	writeWeightedReadinessFixture(t, invalidPath, "{")
	err := runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationAssert([]string{invalidPath})
	if err == nil || !strings.HasPrefix(err.Error(), "weighted admission resonance graft admission final gate observation JSON invalid:") {
		t.Fatalf("expected invalid JSON error, got %v", err)
	}

	missingSchemaPath := filepath.Join(dir, "missing_schema.json")
	writeWeightedAdmissionResonanceGraftAdmissionFinalGateObservationFixture(t, missingSchemaPath)
	writeWeightedReadinessFixture(t, missingSchemaPath, stringsReplaceFirst(readText(t, missingSchemaPath), `"schema": "arianna.live_route_weighted_admission_resonance_graft_admission_final_gate_observation.v1",`, ""))
	requireBoundaryAssertError(t, runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationAssert([]string{missingSchemaPath}), "weighted admission resonance graft admission final gate observation schema missing")

	badSchemaPath := filepath.Join(dir, "bad_schema.json")
	writeWeightedAdmissionResonanceGraftAdmissionFinalGateObservationFixture(t, badSchemaPath)
	writeWeightedReadinessFixture(t, badSchemaPath, stringsReplaceFirst(readText(t, badSchemaPath), `"schema": "arianna.live_route_weighted_admission_resonance_graft_admission_final_gate_observation.v1"`, `"schema": "arianna.live_route_weighted_admission_resonance_graft_admission_final_gate_observation.v0"`))
	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationAssert([]string{badSchemaPath}),
		`weighted admission resonance graft admission final gate observation schema mismatch: got "arianna.live_route_weighted_admission_resonance_graft_admission_final_gate_observation.v0" want "`+admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationSchema+`"`,
	)

	validPath := filepath.Join(dir, "valid.json")
	writeWeightedAdmissionResonanceGraftAdmissionFinalGateObservationFixture(t, validPath)
	if err := runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationAssert([]string{validPath}); err != nil {
		t.Fatalf("valid weighted admission resonance graft admission final gate observation rejected: %v", err)
	}

	badStatusPath := filepath.Join(dir, "bad_status.json")
	writeWeightedAdmissionResonanceGraftAdmissionFinalGateObservationFixture(t, badStatusPath)
	writeWeightedReadinessFixture(t, badStatusPath, stringsReplaceFirst(readText(t, badStatusPath), `"status": "shadow_graft_admission_final_gate_observation_recorded_dry_run"`, `"status": "open"`))
	requireBoundaryAssertError(t, runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationAssert([]string{badStatusPath}), `weighted admission resonance graft admission final gate observation status mismatch: got "open" want "shadow_graft_admission_final_gate_observation_recorded_dry_run"`)

	notReadyPath := filepath.Join(dir, "not_ready.json")
	writeWeightedAdmissionResonanceGraftAdmissionFinalGateObservationFixture(t, notReadyPath)
	writeWeightedReadinessFixture(t, notReadyPath, stringsReplaceFirst(readText(t, notReadyPath), `"weighted_admission_resonance_graft_admission_final_gate_observation_ready": true`, `"weighted_admission_resonance_graft_admission_final_gate_observation_ready": false`))
	requireBoundaryAssertError(t, runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationAssert([]string{notReadyPath}), "weighted admission resonance graft admission final gate observation weighted_admission_resonance_graft_admission_final_gate_observation_ready not ready")

	badActionPath := filepath.Join(dir, "bad_action.json")
	writeWeightedAdmissionResonanceGraftAdmissionFinalGateObservationFixture(t, badActionPath)
	writeWeightedReadinessFixture(t, badActionPath, stringsReplaceFirst(readText(t, badActionPath), `"action": "record_weighted_resonance_shadow_graft_admission_final_gate_observation_dry_run"`, `"action": "open"`))
	requireBoundaryAssertError(t, runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationAssert([]string{badActionPath}), `weighted admission resonance graft admission final gate observation action mismatch: got "open" want "record_weighted_resonance_shadow_graft_admission_final_gate_observation_dry_run"`)

	badObservationActionPath := filepath.Join(dir, "bad_observation_action.json")
	writeWeightedAdmissionResonanceGraftAdmissionFinalGateObservationFixture(t, badObservationActionPath)
	writeWeightedReadinessFixture(t, badObservationActionPath, stringsReplaceFirst(readText(t, badObservationActionPath), `"admission_final_gate_observation_action": "record_blocked_final_gate_receiver_observation"`, `"admission_final_gate_observation_action": "open"`))
	requireBoundaryAssertError(t, runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationAssert([]string{badObservationActionPath}), "weighted admission resonance graft admission final gate observation shape mismatch")

	openedObservationPath := filepath.Join(dir, "opened_observation.json")
	writeWeightedAdmissionResonanceGraftAdmissionFinalGateObservationFixture(t, openedObservationPath)
	writeWeightedReadinessFixture(t, openedObservationPath, stringsReplaceFirst(readText(t, openedObservationPath), `"admission_final_gate_observation_ready": false`, `"admission_final_gate_observation_ready": true`))
	requireBoundaryAssertError(t, runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationAssert([]string{openedObservationPath}), "weighted admission resonance graft admission final gate observation opened admission_final_gate_observation_ready")

	openedSourceReceiverPath := filepath.Join(dir, "opened_source_receiver.json")
	writeWeightedAdmissionResonanceGraftAdmissionFinalGateObservationFixture(t, openedSourceReceiverPath)
	writeWeightedReadinessFixture(t, openedSourceReceiverPath, stringsReplaceFirst(readText(t, openedSourceReceiverPath), `"source_admission_final_gate_receiver_ready": false`, `"source_admission_final_gate_receiver_ready": true`))
	requireBoundaryAssertError(t, runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationAssert([]string{openedSourceReceiverPath}), "weighted admission resonance graft admission final gate observation opened source_admission_final_gate_receiver_ready")

	openedPath := filepath.Join(dir, "opened.json")
	writeWeightedAdmissionResonanceGraftAdmissionFinalGateObservationFixture(t, openedPath)
	writeWeightedReadinessFixture(t, openedPath, stringsReplaceFirst(readText(t, openedPath), `"live_admission_enabled": false`, `"live_admission_enabled": true`))
	requireBoundaryAssertError(t, runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationAssert([]string{openedPath}), "weighted admission resonance graft admission final gate observation opened live_admission_enabled")

	badSourcePath := filepath.Join(dir, "bad_source.json")
	writeWeightedAdmissionResonanceGraftAdmissionFinalGateObservationFixture(t, badSourcePath)
	writeWeightedReadinessFixture(t, badSourcePath, stringsReplaceFirst(readText(t, badSourcePath), `"source_schema": "arianna.live_route_weighted_admission_resonance_graft_admission_final_gate_receiver.v1"`, `"source_schema": "arianna.live_route_weighted_admission_resonance_graft_admission_final_gate_receiver.v0"`))
	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationAssert([]string{badSourcePath}),
		`weighted admission resonance graft admission final gate observation source_schema mismatch: got "arianna.live_route_weighted_admission_resonance_graft_admission_final_gate_receiver.v0" want "`+admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateReceiverSchema+`"`,
	)

	badSourceShapePath := filepath.Join(dir, "bad_source_shape.json")
	writeWeightedAdmissionResonanceGraftAdmissionFinalGateObservationFixture(t, badSourceShapePath)
	writeWeightedReadinessFixture(t, badSourceShapePath, stringsReplaceFirst(readText(t, badSourceShapePath), `"source_admission_final_gate_receiver_action": "preview_blocked_final_gate_receiver"`, `"source_admission_final_gate_receiver_action": "ready"`))
	requireBoundaryAssertError(t, runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationAssert([]string{badSourceShapePath}), "weighted admission resonance graft admission final gate observation source admission final gate receiver shape mismatch")

	badObserverPath := filepath.Join(dir, "bad_observer.json")
	writeWeightedAdmissionResonanceGraftAdmissionFinalGateObservationFixture(t, badObserverPath)
	writeWeightedReadinessFixture(t, badObserverPath, stringsReplaceFirst(readText(t, badObserverPath), `"final_gate_observation_observer": "resonance"`, `"final_gate_observation_observer": "janus"`))
	requireBoundaryAssertError(t, runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationAssert([]string{badObserverPath}), `weighted admission resonance graft admission final gate observation observer mismatch: got "janus" want "resonance"`)

	badAppendHashPath := filepath.Join(dir, "bad_append_hash.json")
	writeWeightedAdmissionResonanceGraftAdmissionFinalGateObservationFixture(t, badAppendHashPath)
	writeWeightedReadinessFixture(t, badAppendHashPath, stringsReplaceFirst(readText(t, badAppendHashPath), `"admission_final_gate_observation_append_hash": "weighted-resonance-graft-admission-final-gate-observation-append-`, `"admission_final_gate_observation_append_hash": "weighted-resonance-graft-admission-final-gate-observation-append-bad`))
	requireBoundaryAssertError(t, runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationAssert([]string{badAppendHashPath}), "weighted admission resonance graft admission final gate observation append_hash mismatch")

	badBodyTargetPath := filepath.Join(dir, "bad_body_target.json")
	writeWeightedAdmissionResonanceGraftAdmissionFinalGateObservationFixture(t, badBodyTargetPath)
	writeWeightedReadinessFixture(t, badBodyTargetPath, stringsReplaceFirst(readText(t, badBodyTargetPath), `"body_target": "none"`, `"body_target": "live"`))
	requireBoundaryAssertError(t, runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationAssert([]string{badBodyTargetPath}), `weighted admission resonance graft admission final gate observation body_target mismatch: got "live" want "none"`)
}

func writeWeightedAdmissionResonanceGraftAdmissionFinalGateObservationFixture(t *testing.T, observationPath string) {
	t.Helper()
	dir := filepath.Dir(observationPath)
	receiverPath := filepath.Join(dir, "srcreceiver.json")
	writeWeightedAdmissionResonanceGraftAdmissionFinalGateReceiverFixture(t, receiverPath)
	if err := runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservation([]string{receiverPath, observationPath}); err != nil {
		t.Fatalf("write weighted admission resonance graft admission final gate observation fixture: %v", err)
	}
}
