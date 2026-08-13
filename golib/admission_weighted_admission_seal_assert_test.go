package main

import (
	"path/filepath"
	"strings"
	"testing"
)

func TestAdmissionLiveRouteWeightedAdmissionSealAssert(t *testing.T) {
	dir := t.TempDir()

	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionSealAssert(nil),
		"usage: --admission-live-route-weighted-admission-seal-assert REPORT",
	)
	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionSealAssert([]string{"seal.json", "extra"}),
		"usage: --admission-live-route-weighted-admission-seal-assert REPORT",
	)
	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionSealAssert([]string{"  "}),
		"weighted admission seal path missing",
	)
	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionSealAssert([]string{filepath.Join(dir, "missing.json")}),
		"weighted admission seal not written",
	)

	emptyPath := filepath.Join(dir, "empty.json")
	writeWeightedReadinessFixture(t, emptyPath, "")
	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionSealAssert([]string{emptyPath}),
		"weighted admission seal not written",
	)

	invalidPath := filepath.Join(dir, "invalid.json")
	writeWeightedReadinessFixture(t, invalidPath, "{")
	err := runAdmissionLiveRouteWeightedAdmissionSealAssert([]string{invalidPath})
	if err == nil || !strings.HasPrefix(err.Error(), "weighted admission seal JSON invalid:") {
		t.Fatalf("expected invalid JSON error, got %v", err)
	}

	missingSchemaPath := filepath.Join(dir, "missing_schema.json")
	writeWeightedAdmissionSealFixture(t, missingSchemaPath)
	sealText := readText(t, missingSchemaPath)
	writeWeightedReadinessFixture(t, missingSchemaPath, stringsReplaceFirst(sealText, `"schema": "arianna.live_route_weighted_admission_seal.v1",`, ""))
	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionSealAssert([]string{missingSchemaPath}),
		"weighted admission seal schema missing",
	)

	badSchemaPath := filepath.Join(dir, "bad_schema.json")
	writeWeightedAdmissionSealFixture(t, badSchemaPath)
	writeWeightedReadinessFixture(t, badSchemaPath, stringsReplaceFirst(readText(t, badSchemaPath), `"schema": "arianna.live_route_weighted_admission_seal.v1"`, `"schema": "arianna.live_route_weighted_admission_seal.v0"`))
	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionSealAssert([]string{badSchemaPath}),
		`weighted admission seal schema mismatch: got "arianna.live_route_weighted_admission_seal.v0" want "`+admissionLiveRouteWeightedAdmissionSealSchema+`"`,
	)

	validPath := filepath.Join(dir, "valid.json")
	writeWeightedAdmissionSealFixture(t, validPath)
	if err := runAdmissionLiveRouteWeightedAdmissionSealAssert([]string{validPath}); err != nil {
		t.Fatalf("valid weighted admission seal rejected: %v", err)
	}

	badStatusPath := filepath.Join(dir, "bad_status.json")
	writeWeightedAdmissionSealFixture(t, badStatusPath)
	writeWeightedReadinessFixture(t, badStatusPath, stringsReplaceFirst(readText(t, badStatusPath), `"status": "sealed_closed_dry_run"`, `"status": "open"`))
	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionSealAssert([]string{badStatusPath}),
		`weighted admission seal status mismatch: got "open" want "sealed_closed_dry_run"`,
	)

	notReadyPath := filepath.Join(dir, "not_ready.json")
	writeWeightedAdmissionSealFixture(t, notReadyPath)
	writeWeightedReadinessFixture(t, notReadyPath, stringsReplaceFirst(readText(t, notReadyPath), `"weighted_admission_seal_ready": true`, `"weighted_admission_seal_ready": false`))
	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionSealAssert([]string{notReadyPath}),
		"weighted admission seal weighted_admission_seal_ready not ready",
	)

	openedPath := filepath.Join(dir, "opened.json")
	writeWeightedAdmissionSealFixture(t, openedPath)
	writeWeightedReadinessFixture(t, openedPath, stringsReplaceFirst(readText(t, openedPath), `"live_admission_enabled": false`, `"live_admission_enabled": true`))
	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionSealAssert([]string{openedPath}),
		"weighted admission seal opened live_admission_enabled",
	)

	missingPathField := filepath.Join(dir, "missing_path.json")
	writeWeightedAdmissionSealFixture(t, missingPathField)
	permitReport := filepath.Join(dir, "permit-"+filepath.Base(missingPathField))
	writeWeightedReadinessFixture(t, missingPathField, stringsReplaceFirst(readText(t, missingPathField), `"source_report": "`+permitReport+`"`, `"source_report": " "`))
	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionSealAssert([]string{missingPathField}),
		"weighted admission seal source_report missing",
	)

	badSourcePath := filepath.Join(dir, "bad_source.json")
	writeWeightedAdmissionSealFixture(t, badSourcePath)
	writeWeightedReadinessFixture(t, badSourcePath, stringsReplaceFirst(readText(t, badSourcePath), `"source_schema": "arianna.live_route_weighted_admission_permit.v1"`, `"source_schema": "arianna.live_route_weighted_admission_permit.v0"`))
	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionSealAssert([]string{badSourcePath}),
		`weighted admission seal source_schema mismatch: got "arianna.live_route_weighted_admission_permit.v0" want "`+admissionLiveRouteWeightedAdmissionPermitSchema+`"`,
	)
}

func writeWeightedAdmissionSealFixture(t *testing.T, sealPath string) {
	t.Helper()
	dir := filepath.Dir(sealPath)
	permitPath := filepath.Join(dir, "permit-"+filepath.Base(sealPath))
	writeWeightedAdmissionPermitFixture(t, permitPath)
	if err := runAdmissionLiveRouteWeightedAdmissionSeal([]string{permitPath, sealPath}); err != nil {
		t.Fatalf("write weighted admission seal fixture: %v", err)
	}
}
