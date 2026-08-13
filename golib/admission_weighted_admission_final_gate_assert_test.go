package main

import (
	"path/filepath"
	"strings"
	"testing"
)

func TestAdmissionLiveRouteWeightedAdmissionFinalGateAssert(t *testing.T) {
	dir := t.TempDir()

	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionFinalGateAssert(nil),
		"usage: --admission-live-route-weighted-admission-final-gate-assert REPORT",
	)
	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionFinalGateAssert([]string{"final_gate.json", "extra"}),
		"usage: --admission-live-route-weighted-admission-final-gate-assert REPORT",
	)
	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionFinalGateAssert([]string{"  "}),
		"weighted admission final gate path missing",
	)
	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionFinalGateAssert([]string{filepath.Join(dir, "missing.json")}),
		"weighted admission final gate not written",
	)

	emptyPath := filepath.Join(dir, "empty.json")
	writeWeightedReadinessFixture(t, emptyPath, "")
	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionFinalGateAssert([]string{emptyPath}),
		"weighted admission final gate not written",
	)

	invalidPath := filepath.Join(dir, "invalid.json")
	writeWeightedReadinessFixture(t, invalidPath, "{")
	err := runAdmissionLiveRouteWeightedAdmissionFinalGateAssert([]string{invalidPath})
	if err == nil || !strings.HasPrefix(err.Error(), "weighted admission final gate JSON invalid:") {
		t.Fatalf("expected invalid JSON error, got %v", err)
	}

	missingSchemaPath := filepath.Join(dir, "missing_schema.json")
	writeWeightedAdmissionFinalGateFixture(t, missingSchemaPath)
	finalGateText := readText(t, missingSchemaPath)
	writeWeightedReadinessFixture(t, missingSchemaPath, stringsReplaceFirst(finalGateText, `"schema": "arianna.live_route_weighted_admission_final_gate.v1",`, ""))
	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionFinalGateAssert([]string{missingSchemaPath}),
		"weighted admission final gate schema missing",
	)

	badSchemaPath := filepath.Join(dir, "bad_schema.json")
	writeWeightedAdmissionFinalGateFixture(t, badSchemaPath)
	writeWeightedReadinessFixture(t, badSchemaPath, stringsReplaceFirst(readText(t, badSchemaPath), `"schema": "arianna.live_route_weighted_admission_final_gate.v1"`, `"schema": "arianna.live_route_weighted_admission_final_gate.v0"`))
	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionFinalGateAssert([]string{badSchemaPath}),
		`weighted admission final gate schema mismatch: got "arianna.live_route_weighted_admission_final_gate.v0" want "`+admissionLiveRouteWeightedAdmissionFinalGateSchema+`"`,
	)

	validPath := filepath.Join(dir, "valid.json")
	writeWeightedAdmissionFinalGateFixture(t, validPath)
	if err := runAdmissionLiveRouteWeightedAdmissionFinalGateAssert([]string{validPath}); err != nil {
		t.Fatalf("valid weighted admission final gate rejected: %v", err)
	}

	badStatusPath := filepath.Join(dir, "bad_status.json")
	writeWeightedAdmissionFinalGateFixture(t, badStatusPath)
	writeWeightedReadinessFixture(t, badStatusPath, stringsReplaceFirst(readText(t, badStatusPath), `"status": "ready_closed_dry_run"`, `"status": "open"`))
	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionFinalGateAssert([]string{badStatusPath}),
		`weighted admission final gate status mismatch: got "open" want "ready_closed_dry_run"`,
	)

	notReadyPath := filepath.Join(dir, "not_ready.json")
	writeWeightedAdmissionFinalGateFixture(t, notReadyPath)
	writeWeightedReadinessFixture(t, notReadyPath, stringsReplaceFirst(readText(t, notReadyPath), `"weighted_admission_final_gate_ready": true`, `"weighted_admission_final_gate_ready": false`))
	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionFinalGateAssert([]string{notReadyPath}),
		"weighted admission final gate weighted_admission_final_gate_ready not ready",
	)

	openedPath := filepath.Join(dir, "opened.json")
	writeWeightedAdmissionFinalGateFixture(t, openedPath)
	writeWeightedReadinessFixture(t, openedPath, stringsReplaceFirst(readText(t, openedPath), `"live_admission_enabled": false`, `"live_admission_enabled": true`))
	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionFinalGateAssert([]string{openedPath}),
		"weighted admission final gate opened live_admission_enabled",
	)

	missingPathField := filepath.Join(dir, "missing_path.json")
	writeWeightedAdmissionFinalGateFixture(t, missingPathField)
	sealReport := filepath.Join(dir, "seal-"+filepath.Base(missingPathField))
	writeWeightedReadinessFixture(t, missingPathField, stringsReplaceFirst(readText(t, missingPathField), `"source_report": "`+sealReport+`"`, `"source_report": " "`))
	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionFinalGateAssert([]string{missingPathField}),
		"weighted admission final gate source_report missing",
	)

	badSourcePath := filepath.Join(dir, "bad_source.json")
	writeWeightedAdmissionFinalGateFixture(t, badSourcePath)
	writeWeightedReadinessFixture(t, badSourcePath, stringsReplaceFirst(readText(t, badSourcePath), `"source_schema": "arianna.live_route_weighted_admission_seal.v1"`, `"source_schema": "arianna.live_route_weighted_admission_seal.v0"`))
	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionFinalGateAssert([]string{badSourcePath}),
		`weighted admission final gate source_schema mismatch: got "arianna.live_route_weighted_admission_seal.v0" want "`+admissionLiveRouteWeightedAdmissionSealSchema+`"`,
	)
}

func writeWeightedAdmissionFinalGateFixture(t *testing.T, finalGatePath string) {
	t.Helper()
	dir := filepath.Dir(finalGatePath)
	sealPath := filepath.Join(dir, "seal-"+filepath.Base(finalGatePath))
	writeWeightedAdmissionSealFixture(t, sealPath)
	if err := runAdmissionLiveRouteWeightedAdmissionFinalGate([]string{sealPath, finalGatePath}); err != nil {
		t.Fatalf("write weighted admission final gate fixture: %v", err)
	}
}
