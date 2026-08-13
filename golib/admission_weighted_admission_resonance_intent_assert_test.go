package main

import (
	"path/filepath"
	"strings"
	"testing"
)

func TestAdmissionLiveRouteWeightedAdmissionResonanceIntentAssert(t *testing.T) {
	dir := t.TempDir()

	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionResonanceIntentAssert(nil),
		"usage: --admission-live-route-weighted-admission-resonance-intent-assert REPORT",
	)
	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionResonanceIntentAssert([]string{"resonance_intent.json", "extra"}),
		"usage: --admission-live-route-weighted-admission-resonance-intent-assert REPORT",
	)
	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionResonanceIntentAssert([]string{"  "}),
		"weighted admission resonance intent path missing",
	)
	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionResonanceIntentAssert([]string{filepath.Join(dir, "missing.json")}),
		"weighted admission resonance intent not written",
	)

	emptyPath := filepath.Join(dir, "empty.json")
	writeWeightedReadinessFixture(t, emptyPath, "")
	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionResonanceIntentAssert([]string{emptyPath}),
		"weighted admission resonance intent not written",
	)

	invalidPath := filepath.Join(dir, "invalid.json")
	writeWeightedReadinessFixture(t, invalidPath, "{")
	err := runAdmissionLiveRouteWeightedAdmissionResonanceIntentAssert([]string{invalidPath})
	if err == nil || !strings.HasPrefix(err.Error(), "weighted admission resonance intent JSON invalid:") {
		t.Fatalf("expected invalid JSON error, got %v", err)
	}

	missingSchemaPath := filepath.Join(dir, "missing_schema.json")
	writeWeightedAdmissionResonanceIntentFixture(t, missingSchemaPath)
	intentText := readText(t, missingSchemaPath)
	writeWeightedReadinessFixture(t, missingSchemaPath, stringsReplaceFirst(intentText, `"schema": "arianna.live_route_weighted_admission_resonance_intent.v1",`, ""))
	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionResonanceIntentAssert([]string{missingSchemaPath}),
		"weighted admission resonance intent schema missing",
	)

	badSchemaPath := filepath.Join(dir, "bad_schema.json")
	writeWeightedAdmissionResonanceIntentFixture(t, badSchemaPath)
	writeWeightedReadinessFixture(t, badSchemaPath, stringsReplaceFirst(readText(t, badSchemaPath), `"schema": "arianna.live_route_weighted_admission_resonance_intent.v1"`, `"schema": "arianna.live_route_weighted_admission_resonance_intent.v0"`))
	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionResonanceIntentAssert([]string{badSchemaPath}),
		`weighted admission resonance intent schema mismatch: got "arianna.live_route_weighted_admission_resonance_intent.v0" want "`+admissionLiveRouteWeightedAdmissionResonanceIntentSchema+`"`,
	)

	validPath := filepath.Join(dir, "valid.json")
	writeWeightedAdmissionResonanceIntentFixture(t, validPath)
	if err := runAdmissionLiveRouteWeightedAdmissionResonanceIntentAssert([]string{validPath}); err != nil {
		t.Fatalf("valid weighted admission resonance intent rejected: %v", err)
	}

	badStatusPath := filepath.Join(dir, "bad_status.json")
	writeWeightedAdmissionResonanceIntentFixture(t, badStatusPath)
	writeWeightedReadinessFixture(t, badStatusPath, stringsReplaceFirst(readText(t, badStatusPath), `"status": "resonance_intent_drafted_dry_run"`, `"status": "open"`))
	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionResonanceIntentAssert([]string{badStatusPath}),
		`weighted admission resonance intent status mismatch: got "open" want "resonance_intent_drafted_dry_run"`,
	)

	notReadyPath := filepath.Join(dir, "not_ready.json")
	writeWeightedAdmissionResonanceIntentFixture(t, notReadyPath)
	writeWeightedReadinessFixture(t, notReadyPath, stringsReplaceFirst(readText(t, notReadyPath), `"weighted_admission_resonance_intent_ready": true`, `"weighted_admission_resonance_intent_ready": false`))
	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionResonanceIntentAssert([]string{notReadyPath}),
		"weighted admission resonance intent weighted_admission_resonance_intent_ready not ready",
	)

	openedPath := filepath.Join(dir, "opened.json")
	writeWeightedAdmissionResonanceIntentFixture(t, openedPath)
	writeWeightedReadinessFixture(t, openedPath, stringsReplaceFirst(readText(t, openedPath), `"live_admission_enabled": false`, `"live_admission_enabled": true`))
	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionResonanceIntentAssert([]string{openedPath}),
		"weighted admission resonance intent opened live_admission_enabled",
	)

	missingPathField := filepath.Join(dir, "missing_path.json")
	writeWeightedAdmissionResonanceIntentFixture(t, missingPathField)
	finalGateReport := filepath.Join(dir, "final-gate-"+filepath.Base(missingPathField))
	writeWeightedReadinessFixture(t, missingPathField, stringsReplaceFirst(readText(t, missingPathField), `"source_report": "`+finalGateReport+`"`, `"source_report": " "`))
	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionResonanceIntentAssert([]string{missingPathField}),
		"weighted admission resonance intent source_report missing",
	)

	badSourcePath := filepath.Join(dir, "bad_source.json")
	writeWeightedAdmissionResonanceIntentFixture(t, badSourcePath)
	writeWeightedReadinessFixture(t, badSourcePath, stringsReplaceFirst(readText(t, badSourcePath), `"source_schema": "arianna.live_route_weighted_admission_final_gate.v1"`, `"source_schema": "arianna.live_route_weighted_admission_final_gate.v0"`))
	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionResonanceIntentAssert([]string{badSourcePath}),
		`weighted admission resonance intent source_schema mismatch: got "arianna.live_route_weighted_admission_final_gate.v0" want "`+admissionLiveRouteWeightedAdmissionFinalGateSchema+`"`,
	)

	badReceiverPath := filepath.Join(dir, "bad_receiver.json")
	writeWeightedAdmissionResonanceIntentFixture(t, badReceiverPath)
	writeWeightedReadinessFixture(t, badReceiverPath, stringsReplaceFirst(readText(t, badReceiverPath), `"receiver": "resonance"`, `"receiver": "janus"`))
	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionResonanceIntentAssert([]string{badReceiverPath}),
		`weighted admission resonance intent receiver mismatch: got "janus" want "resonance"`,
	)
}

func writeWeightedAdmissionResonanceIntentFixture(t *testing.T, intentPath string) {
	t.Helper()
	dir := filepath.Dir(intentPath)
	finalGatePath := filepath.Join(dir, "final-gate-"+filepath.Base(intentPath))
	writeWeightedAdmissionFinalGateFixture(t, finalGatePath)
	if err := runAdmissionLiveRouteWeightedAdmissionResonanceIntent([]string{finalGatePath, intentPath}); err != nil {
		t.Fatalf("write weighted admission resonance intent fixture: %v", err)
	}
}
