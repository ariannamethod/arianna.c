package main

import (
	"path/filepath"
	"strings"
	"testing"
)

func TestAdmissionLiveRouteWeightedAdmissionPermitAssert(t *testing.T) {
	dir := t.TempDir()

	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionPermitAssert(nil),
		"usage: --admission-live-route-weighted-admission-permit-assert REPORT",
	)
	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionPermitAssert([]string{"permit.json", "extra"}),
		"usage: --admission-live-route-weighted-admission-permit-assert REPORT",
	)
	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionPermitAssert([]string{"  "}),
		"weighted admission permit path missing",
	)
	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionPermitAssert([]string{filepath.Join(dir, "missing.json")}),
		"weighted admission permit not written",
	)

	emptyPath := filepath.Join(dir, "empty.json")
	writeWeightedReadinessFixture(t, emptyPath, "")
	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionPermitAssert([]string{emptyPath}),
		"weighted admission permit not written",
	)

	invalidPath := filepath.Join(dir, "invalid.json")
	writeWeightedReadinessFixture(t, invalidPath, "{")
	err := runAdmissionLiveRouteWeightedAdmissionPermitAssert([]string{invalidPath})
	if err == nil || !strings.HasPrefix(err.Error(), "weighted admission permit JSON invalid:") {
		t.Fatalf("expected invalid JSON error, got %v", err)
	}

	missingSchemaPath := filepath.Join(dir, "missing_schema.json")
	writeWeightedAdmissionPermitFixture(t, missingSchemaPath)
	permitText := readText(t, missingSchemaPath)
	writeWeightedReadinessFixture(t, missingSchemaPath, stringsReplaceFirst(permitText, `"schema": "arianna.live_route_weighted_admission_permit.v1",`, ""))
	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionPermitAssert([]string{missingSchemaPath}),
		"weighted admission permit schema missing",
	)

	badSchemaPath := filepath.Join(dir, "bad_schema.json")
	writeWeightedAdmissionPermitFixture(t, badSchemaPath)
	writeWeightedReadinessFixture(t, badSchemaPath, stringsReplaceFirst(readText(t, badSchemaPath), `"schema": "arianna.live_route_weighted_admission_permit.v1"`, `"schema": "arianna.live_route_weighted_admission_permit.v0"`))
	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionPermitAssert([]string{badSchemaPath}),
		`weighted admission permit schema mismatch: got "arianna.live_route_weighted_admission_permit.v0" want "`+admissionLiveRouteWeightedAdmissionPermitSchema+`"`,
	)

	validPath := filepath.Join(dir, "valid.json")
	writeWeightedAdmissionPermitFixture(t, validPath)
	if err := runAdmissionLiveRouteWeightedAdmissionPermitAssert([]string{validPath}); err != nil {
		t.Fatalf("valid weighted admission permit rejected: %v", err)
	}

	badStatusPath := filepath.Join(dir, "bad_status.json")
	writeWeightedAdmissionPermitFixture(t, badStatusPath)
	writeWeightedReadinessFixture(t, badStatusPath, stringsReplaceFirst(readText(t, badStatusPath), `"status": "operator_permitted_closed_dry_run"`, `"status": "open"`))
	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionPermitAssert([]string{badStatusPath}),
		`weighted admission permit status mismatch: got "open" want "operator_permitted_closed_dry_run"`,
	)

	notReadyPath := filepath.Join(dir, "not_ready.json")
	writeWeightedAdmissionPermitFixture(t, notReadyPath)
	writeWeightedReadinessFixture(t, notReadyPath, stringsReplaceFirst(readText(t, notReadyPath), `"weighted_admission_permit_ready": true`, `"weighted_admission_permit_ready": false`))
	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionPermitAssert([]string{notReadyPath}),
		"weighted admission permit weighted_admission_permit_ready not ready",
	)

	openedPath := filepath.Join(dir, "opened.json")
	writeWeightedAdmissionPermitFixture(t, openedPath)
	writeWeightedReadinessFixture(t, openedPath, stringsReplaceFirst(readText(t, openedPath), `"admission_allowed": false`, `"admission_allowed": true`))
	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionPermitAssert([]string{openedPath}),
		"weighted admission permit opened admission_allowed",
	)

	missingPathField := filepath.Join(dir, "missing_path.json")
	writeWeightedAdmissionPermitFixture(t, missingPathField)
	contractReport := filepath.Join(dir, "contract-authority-"+filepath.Base(missingPathField))
	writeWeightedReadinessFixture(t, missingPathField, stringsReplaceFirst(readText(t, missingPathField), `"source_contract_report": "`+contractReport+`"`, `"source_contract_report": " "`))
	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionPermitAssert([]string{missingPathField}),
		"weighted admission permit source_contract_report missing",
	)

	badSourcePath := filepath.Join(dir, "bad_source.json")
	writeWeightedAdmissionPermitFixture(t, badSourcePath)
	writeWeightedReadinessFixture(t, badSourcePath, stringsReplaceFirst(readText(t, badSourcePath), `"source_schema": "arianna.live_route_weighted_admission_authority.v1"`, `"source_schema": "arianna.live_route_weighted_admission_authority.v0"`))
	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionPermitAssert([]string{badSourcePath}),
		`weighted admission permit source_schema mismatch: got "arianna.live_route_weighted_admission_authority.v0" want "`+admissionLiveRouteWeightedAdmissionAuthoritySchema+`"`,
	)
}

func writeWeightedAdmissionPermitFixture(t *testing.T, permitPath string) {
	t.Helper()
	dir := filepath.Dir(permitPath)
	authorityPath := filepath.Join(dir, "authority-"+filepath.Base(permitPath))
	writeWeightedAdmissionAuthorityFixture(t, authorityPath)
	t.Setenv("A2A_WEIGHTED_ADMISSION_PERMIT_KEY", weightedAdmissionPermitKey)
	if err := runAdmissionLiveRouteWeightedAdmissionPermit([]string{authorityPath, permitPath}); err != nil {
		t.Fatalf("write weighted admission permit fixture: %v", err)
	}
}
