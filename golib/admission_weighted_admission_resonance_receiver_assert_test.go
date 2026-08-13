package main

import (
	"path/filepath"
	"strings"
	"testing"
)

func TestAdmissionLiveRouteWeightedAdmissionResonanceReceiverAssert(t *testing.T) {
	dir := t.TempDir()

	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionResonanceReceiverAssert(nil),
		"usage: --admission-live-route-weighted-admission-resonance-receiver-assert REPORT",
	)
	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionResonanceReceiverAssert([]string{"receiver.json", "extra"}),
		"usage: --admission-live-route-weighted-admission-resonance-receiver-assert REPORT",
	)
	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionResonanceReceiverAssert([]string{"  "}),
		"weighted admission resonance receiver path missing",
	)
	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionResonanceReceiverAssert([]string{filepath.Join(dir, "missing.json")}),
		"weighted admission resonance receiver not written",
	)

	emptyPath := filepath.Join(dir, "empty.json")
	writeWeightedReadinessFixture(t, emptyPath, "")
	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionResonanceReceiverAssert([]string{emptyPath}),
		"weighted admission resonance receiver not written",
	)

	invalidPath := filepath.Join(dir, "invalid.json")
	writeWeightedReadinessFixture(t, invalidPath, "{")
	err := runAdmissionLiveRouteWeightedAdmissionResonanceReceiverAssert([]string{invalidPath})
	if err == nil || !strings.HasPrefix(err.Error(), "weighted admission resonance receiver JSON invalid:") {
		t.Fatalf("expected invalid JSON error, got %v", err)
	}

	missingSchemaPath := filepath.Join(dir, "missing_schema.json")
	writeWeightedAdmissionResonanceReceiverFixture(t, missingSchemaPath)
	receiverText := readText(t, missingSchemaPath)
	writeWeightedReadinessFixture(t, missingSchemaPath, stringsReplaceFirst(receiverText, `"schema": "arianna.live_route_weighted_admission_resonance_receiver.v1",`, ""))
	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionResonanceReceiverAssert([]string{missingSchemaPath}),
		"weighted admission resonance receiver schema missing",
	)

	badSchemaPath := filepath.Join(dir, "bad_schema.json")
	writeWeightedAdmissionResonanceReceiverFixture(t, badSchemaPath)
	writeWeightedReadinessFixture(t, badSchemaPath, stringsReplaceFirst(readText(t, badSchemaPath), `"schema": "arianna.live_route_weighted_admission_resonance_receiver.v1"`, `"schema": "arianna.live_route_weighted_admission_resonance_receiver.v0"`))
	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionResonanceReceiverAssert([]string{badSchemaPath}),
		`weighted admission resonance receiver schema mismatch: got "arianna.live_route_weighted_admission_resonance_receiver.v0" want "`+admissionLiveRouteWeightedAdmissionResonanceReceiverSchema+`"`,
	)

	validPath := filepath.Join(dir, "valid.json")
	writeWeightedAdmissionResonanceReceiverFixture(t, validPath)
	if err := runAdmissionLiveRouteWeightedAdmissionResonanceReceiverAssert([]string{validPath}); err != nil {
		t.Fatalf("valid weighted admission resonance receiver rejected: %v", err)
	}

	badStatusPath := filepath.Join(dir, "bad_status.json")
	writeWeightedAdmissionResonanceReceiverFixture(t, badStatusPath)
	writeWeightedReadinessFixture(t, badStatusPath, stringsReplaceFirst(readText(t, badStatusPath), `"status": "receiver_previewed_dry_run"`, `"status": "open"`))
	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionResonanceReceiverAssert([]string{badStatusPath}),
		`weighted admission resonance receiver status mismatch: got "open" want "receiver_previewed_dry_run"`,
	)

	notReadyPath := filepath.Join(dir, "not_ready.json")
	writeWeightedAdmissionResonanceReceiverFixture(t, notReadyPath)
	writeWeightedReadinessFixture(t, notReadyPath, stringsReplaceFirst(readText(t, notReadyPath), `"weighted_admission_resonance_receiver_ready": true`, `"weighted_admission_resonance_receiver_ready": false`))
	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionResonanceReceiverAssert([]string{notReadyPath}),
		"weighted admission resonance receiver weighted_admission_resonance_receiver_ready not ready",
	)

	openedPath := filepath.Join(dir, "opened.json")
	writeWeightedAdmissionResonanceReceiverFixture(t, openedPath)
	writeWeightedReadinessFixture(t, openedPath, stringsReplaceFirst(readText(t, openedPath), `"live_admission_enabled": false`, `"live_admission_enabled": true`))
	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionResonanceReceiverAssert([]string{openedPath}),
		"weighted admission resonance receiver opened live_admission_enabled",
	)

	missingPathField := filepath.Join(dir, "missing_path.json")
	writeWeightedAdmissionResonanceReceiverFixture(t, missingPathField)
	intentReport := filepath.Join(dir, "intent-"+filepath.Base(missingPathField))
	writeWeightedReadinessFixture(t, missingPathField, stringsReplaceFirst(readText(t, missingPathField), `"source_report": "`+intentReport+`"`, `"source_report": " "`))
	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionResonanceReceiverAssert([]string{missingPathField}),
		"weighted admission resonance receiver source_report missing",
	)

	badSourcePath := filepath.Join(dir, "bad_source.json")
	writeWeightedAdmissionResonanceReceiverFixture(t, badSourcePath)
	writeWeightedReadinessFixture(t, badSourcePath, stringsReplaceFirst(readText(t, badSourcePath), `"source_schema": "arianna.live_route_weighted_admission_resonance_intent.v1"`, `"source_schema": "arianna.live_route_weighted_admission_resonance_intent.v0"`))
	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionResonanceReceiverAssert([]string{badSourcePath}),
		`weighted admission resonance receiver source_schema mismatch: got "arianna.live_route_weighted_admission_resonance_intent.v0" want "`+admissionLiveRouteWeightedAdmissionResonanceIntentSchema+`"`,
	)

	badReceiverPath := filepath.Join(dir, "bad_receiver.json")
	writeWeightedAdmissionResonanceReceiverFixture(t, badReceiverPath)
	writeWeightedReadinessFixture(t, badReceiverPath, stringsReplaceFirst(readText(t, badReceiverPath), `"receiver": "resonance"`, `"receiver": "janus"`))
	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionResonanceReceiverAssert([]string{badReceiverPath}),
		`weighted admission resonance receiver receiver mismatch: got "janus" want "resonance"`,
	)

	badHashPath := filepath.Join(dir, "bad_hash.json")
	writeWeightedAdmissionResonanceReceiverFixture(t, badHashPath)
	writeWeightedReadinessFixture(t, badHashPath, stringsReplaceFirst(readText(t, badHashPath), `"state_delta_hash": "weighted-resonance-receiver-delta-`, `"state_delta_hash": "weighted-resonance-receiver-delta-bad`))
	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionResonanceReceiverAssert([]string{badHashPath}),
		"weighted admission resonance receiver state_delta_hash mismatch",
	)

	badSourceGuardPath := filepath.Join(dir, "bad_source_guard.json")
	writeWeightedAdmissionResonanceReceiverFixture(t, badSourceGuardPath)
	writeWeightedReadinessFixture(t, badSourceGuardPath, stringsReplaceFirst(readText(t, badSourceGuardPath), `"source_raw_dream_text_allowed": false`, `"source_raw_dream_text_allowed": true`))
	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionResonanceReceiverAssert([]string{badSourceGuardPath}),
		"weighted admission resonance receiver opened source_raw_dream_text_allowed",
	)
}

func writeWeightedAdmissionResonanceReceiverFixture(t *testing.T, receiverPath string) {
	t.Helper()
	dir := filepath.Dir(receiverPath)
	intentPath := filepath.Join(dir, "intent-"+filepath.Base(receiverPath))
	writeWeightedAdmissionResonanceIntentFixture(t, intentPath)
	if err := runAdmissionLiveRouteWeightedAdmissionResonanceReceiver([]string{intentPath, receiverPath}); err != nil {
		t.Fatalf("write weighted admission resonance receiver fixture: %v", err)
	}
}
