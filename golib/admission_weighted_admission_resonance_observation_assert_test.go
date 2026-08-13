package main

import (
	"path/filepath"
	"strings"
	"testing"
)

func TestAdmissionLiveRouteWeightedAdmissionResonanceObservationAssert(t *testing.T) {
	dir := t.TempDir()

	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionResonanceObservationAssert(nil),
		"usage: --admission-live-route-weighted-admission-resonance-observation-assert REPORT",
	)
	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionResonanceObservationAssert([]string{"observation.json", "extra"}),
		"usage: --admission-live-route-weighted-admission-resonance-observation-assert REPORT",
	)
	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionResonanceObservationAssert([]string{"  "}),
		"weighted admission resonance observation path missing",
	)
	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionResonanceObservationAssert([]string{filepath.Join(dir, "missing.json")}),
		"weighted admission resonance observation not written",
	)

	emptyPath := filepath.Join(dir, "empty.json")
	writeWeightedReadinessFixture(t, emptyPath, "")
	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionResonanceObservationAssert([]string{emptyPath}),
		"weighted admission resonance observation not written",
	)

	invalidPath := filepath.Join(dir, "invalid.json")
	writeWeightedReadinessFixture(t, invalidPath, "{")
	err := runAdmissionLiveRouteWeightedAdmissionResonanceObservationAssert([]string{invalidPath})
	if err == nil || !strings.HasPrefix(err.Error(), "weighted admission resonance observation JSON invalid:") {
		t.Fatalf("expected invalid JSON error, got %v", err)
	}

	missingSchemaPath := filepath.Join(dir, "missing_schema.json")
	writeWeightedAdmissionResonanceObservationFixture(t, missingSchemaPath)
	observationText := readText(t, missingSchemaPath)
	writeWeightedReadinessFixture(t, missingSchemaPath, stringsReplaceFirst(observationText, `"schema": "arianna.live_route_weighted_admission_resonance_observation.v1",`, ""))
	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionResonanceObservationAssert([]string{missingSchemaPath}),
		"weighted admission resonance observation schema missing",
	)

	badSchemaPath := filepath.Join(dir, "bad_schema.json")
	writeWeightedAdmissionResonanceObservationFixture(t, badSchemaPath)
	writeWeightedReadinessFixture(t, badSchemaPath, stringsReplaceFirst(readText(t, badSchemaPath), `"schema": "arianna.live_route_weighted_admission_resonance_observation.v1"`, `"schema": "arianna.live_route_weighted_admission_resonance_observation.v0"`))
	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionResonanceObservationAssert([]string{badSchemaPath}),
		`weighted admission resonance observation schema mismatch: got "arianna.live_route_weighted_admission_resonance_observation.v0" want "`+admissionLiveRouteWeightedAdmissionResonanceObservationSchema+`"`,
	)

	validPath := filepath.Join(dir, "valid.json")
	writeWeightedAdmissionResonanceObservationFixture(t, validPath)
	if err := runAdmissionLiveRouteWeightedAdmissionResonanceObservationAssert([]string{validPath}); err != nil {
		t.Fatalf("valid weighted admission resonance observation rejected: %v", err)
	}

	badStatusPath := filepath.Join(dir, "bad_status.json")
	writeWeightedAdmissionResonanceObservationFixture(t, badStatusPath)
	writeWeightedReadinessFixture(t, badStatusPath, stringsReplaceFirst(readText(t, badStatusPath), `"status": "observation_recorded_dry_run"`, `"status": "open"`))
	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionResonanceObservationAssert([]string{badStatusPath}),
		`weighted admission resonance observation status mismatch: got "open" want "observation_recorded_dry_run"`,
	)

	notReadyPath := filepath.Join(dir, "not_ready.json")
	writeWeightedAdmissionResonanceObservationFixture(t, notReadyPath)
	writeWeightedReadinessFixture(t, notReadyPath, stringsReplaceFirst(readText(t, notReadyPath), `"weighted_admission_resonance_observation_ready": true`, `"weighted_admission_resonance_observation_ready": false`))
	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionResonanceObservationAssert([]string{notReadyPath}),
		"weighted admission resonance observation weighted_admission_resonance_observation_ready not ready",
	)

	openedPath := filepath.Join(dir, "opened.json")
	writeWeightedAdmissionResonanceObservationFixture(t, openedPath)
	writeWeightedReadinessFixture(t, openedPath, stringsReplaceFirst(readText(t, openedPath), `"live_admission_enabled": false`, `"live_admission_enabled": true`))
	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionResonanceObservationAssert([]string{openedPath}),
		"weighted admission resonance observation opened live_admission_enabled",
	)

	missingPathField := filepath.Join(dir, "missing_path.json")
	writeWeightedAdmissionResonanceObservationFixture(t, missingPathField)
	receiverReport := filepath.Join(dir, "receiver-"+filepath.Base(missingPathField))
	writeWeightedReadinessFixture(t, missingPathField, stringsReplaceFirst(readText(t, missingPathField), `"source_report": "`+receiverReport+`"`, `"source_report": " "`))
	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionResonanceObservationAssert([]string{missingPathField}),
		"weighted admission resonance observation source_report missing",
	)

	badSourcePath := filepath.Join(dir, "bad_source.json")
	writeWeightedAdmissionResonanceObservationFixture(t, badSourcePath)
	writeWeightedReadinessFixture(t, badSourcePath, stringsReplaceFirst(readText(t, badSourcePath), `"source_schema": "arianna.live_route_weighted_admission_resonance_receiver.v1"`, `"source_schema": "arianna.live_route_weighted_admission_resonance_receiver.v0"`))
	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionResonanceObservationAssert([]string{badSourcePath}),
		`weighted admission resonance observation source_schema mismatch: got "arianna.live_route_weighted_admission_resonance_receiver.v0" want "`+admissionLiveRouteWeightedAdmissionResonanceReceiverSchema+`"`,
	)

	badObserverPath := filepath.Join(dir, "bad_observer.json")
	writeWeightedAdmissionResonanceObservationFixture(t, badObserverPath)
	writeWeightedReadinessFixture(t, badObserverPath, stringsReplaceFirst(readText(t, badObserverPath), `"observer": "resonance"`, `"observer": "janus"`))
	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionResonanceObservationAssert([]string{badObserverPath}),
		`weighted admission resonance observation observer mismatch: got "janus" want "resonance"`,
	)

	badHashPath := filepath.Join(dir, "bad_hash.json")
	writeWeightedAdmissionResonanceObservationFixture(t, badHashPath)
	writeWeightedReadinessFixture(t, badHashPath, stringsReplaceFirst(readText(t, badHashPath), `"append_hash": "weighted-resonance-observation-append-`, `"append_hash": "weighted-resonance-observation-append-bad`))
	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionResonanceObservationAssert([]string{badHashPath}),
		"weighted admission resonance observation append_hash mismatch",
	)

	badSourceGuardPath := filepath.Join(dir, "bad_source_guard.json")
	writeWeightedAdmissionResonanceObservationFixture(t, badSourceGuardPath)
	writeWeightedReadinessFixture(t, badSourceGuardPath, stringsReplaceFirst(readText(t, badSourceGuardPath), `"source_raw_dream_text_allowed": false`, `"source_raw_dream_text_allowed": true`))
	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionResonanceObservationAssert([]string{badSourceGuardPath}),
		"weighted admission resonance observation opened source_raw_dream_text_allowed",
	)

	badBodyTargetPath := filepath.Join(dir, "bad_body_target.json")
	writeWeightedAdmissionResonanceObservationFixture(t, badBodyTargetPath)
	writeWeightedReadinessFixture(t, badBodyTargetPath, stringsReplaceFirst(readText(t, badBodyTargetPath), `"body_target": "none"`, `"body_target": "live"`))
	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionResonanceObservationAssert([]string{badBodyTargetPath}),
		`weighted admission resonance observation body_target mismatch: got "live" want "none"`,
	)
}

func writeWeightedAdmissionResonanceObservationFixture(t *testing.T, observationPath string) {
	t.Helper()
	dir := filepath.Dir(observationPath)
	receiverPath := filepath.Join(dir, "receiver-"+filepath.Base(observationPath))
	writeWeightedAdmissionResonanceReceiverFixture(t, receiverPath)
	if err := runAdmissionLiveRouteWeightedAdmissionResonanceObservation([]string{receiverPath, observationPath}); err != nil {
		t.Fatalf("write weighted admission resonance observation fixture: %v", err)
	}
}
