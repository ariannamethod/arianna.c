package main

import (
	"path/filepath"
	"strings"
	"testing"
)

func TestAdmissionLiveRouteWeightedAdmissionAuthorityAssert(t *testing.T) {
	dir := t.TempDir()

	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionAuthorityAssert(nil),
		"usage: --admission-live-route-weighted-admission-authority-assert REPORT",
	)
	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionAuthorityAssert([]string{"authority.json", "extra"}),
		"usage: --admission-live-route-weighted-admission-authority-assert REPORT",
	)
	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionAuthorityAssert([]string{"  "}),
		"weighted admission authority path missing",
	)
	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionAuthorityAssert([]string{filepath.Join(dir, "missing.json")}),
		"weighted admission authority not written",
	)

	emptyPath := filepath.Join(dir, "empty.json")
	writeWeightedReadinessFixture(t, emptyPath, "")
	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionAuthorityAssert([]string{emptyPath}),
		"weighted admission authority not written",
	)

	invalidPath := filepath.Join(dir, "invalid.json")
	writeWeightedReadinessFixture(t, invalidPath, "{")
	err := runAdmissionLiveRouteWeightedAdmissionAuthorityAssert([]string{invalidPath})
	if err == nil || !strings.HasPrefix(err.Error(), "weighted admission authority JSON invalid:") {
		t.Fatalf("expected invalid JSON error, got %v", err)
	}

	missingSchemaPath := filepath.Join(dir, "missing_schema.json")
	writeWeightedAdmissionAuthorityFixture(t, missingSchemaPath)
	authorityText := readText(t, missingSchemaPath)
	writeWeightedReadinessFixture(t, missingSchemaPath, stringsReplaceFirst(authorityText, `"schema": "arianna.live_route_weighted_admission_authority.v1",`, ""))
	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionAuthorityAssert([]string{missingSchemaPath}),
		"weighted admission authority schema missing",
	)

	badSchemaPath := filepath.Join(dir, "bad_schema.json")
	writeWeightedAdmissionAuthorityFixture(t, badSchemaPath)
	writeWeightedReadinessFixture(t, badSchemaPath, stringsReplaceFirst(readText(t, badSchemaPath), `"schema": "arianna.live_route_weighted_admission_authority.v1"`, `"schema": "arianna.live_route_weighted_admission_authority.v0"`))
	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionAuthorityAssert([]string{badSchemaPath}),
		`weighted admission authority schema mismatch: got "arianna.live_route_weighted_admission_authority.v0" want "`+admissionLiveRouteWeightedAdmissionAuthoritySchema+`"`,
	)

	validPath := filepath.Join(dir, "valid.json")
	writeWeightedAdmissionAuthorityFixture(t, validPath)
	if err := runAdmissionLiveRouteWeightedAdmissionAuthorityAssert([]string{validPath}); err != nil {
		t.Fatalf("valid weighted admission authority rejected: %v", err)
	}

	badStatusPath := filepath.Join(dir, "bad_status.json")
	writeWeightedAdmissionAuthorityFixture(t, badStatusPath)
	writeWeightedReadinessFixture(t, badStatusPath, stringsReplaceFirst(readText(t, badStatusPath), `"status": "authority_receipt_closed_dry_run"`, `"status": "open"`))
	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionAuthorityAssert([]string{badStatusPath}),
		`weighted admission authority status mismatch: got "open" want "authority_receipt_closed_dry_run"`,
	)

	notReadyPath := filepath.Join(dir, "not_ready.json")
	writeWeightedAdmissionAuthorityFixture(t, notReadyPath)
	writeWeightedReadinessFixture(t, notReadyPath, stringsReplaceFirst(readText(t, notReadyPath), `"weighted_admission_authority_receipt_ready": true`, `"weighted_admission_authority_receipt_ready": false`))
	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionAuthorityAssert([]string{notReadyPath}),
		"weighted admission authority weighted_admission_authority_receipt_ready not ready",
	)

	openedPath := filepath.Join(dir, "opened.json")
	writeWeightedAdmissionAuthorityFixture(t, openedPath)
	writeWeightedReadinessFixture(t, openedPath, stringsReplaceFirst(readText(t, openedPath), `"authority_granted": false`, `"authority_granted": true`))
	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionAuthorityAssert([]string{openedPath}),
		"weighted admission authority opened authority_granted",
	)

	missingPathField := filepath.Join(dir, "missing_path.json")
	writeWeightedAdmissionAuthorityFixture(t, missingPathField)
	preconditionReport := filepath.Join(dir, "precondition-contract-"+filepath.Base(missingPathField))
	writeWeightedReadinessFixture(t, missingPathField, stringsReplaceFirst(readText(t, missingPathField), `"source_precondition_report": "`+preconditionReport+`"`, `"source_precondition_report": " "`))
	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionAuthorityAssert([]string{missingPathField}),
		"weighted admission authority source_precondition_report missing",
	)

	badSourcePath := filepath.Join(dir, "bad_source.json")
	writeWeightedAdmissionAuthorityFixture(t, badSourcePath)
	writeWeightedReadinessFixture(t, badSourcePath, stringsReplaceFirst(readText(t, badSourcePath), `"source_schema": "arianna.live_route_weighted_admission_contract.v1"`, `"source_schema": "arianna.live_route_weighted_admission_contract.v0"`))
	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionAuthorityAssert([]string{badSourcePath}),
		`weighted admission authority source_schema mismatch: got "arianna.live_route_weighted_admission_contract.v0" want "`+admissionLiveRouteWeightedAdmissionContractSchema+`"`,
	)
}

func writeWeightedAdmissionAuthorityFixture(t *testing.T, authorityPath string) {
	t.Helper()
	dir := filepath.Dir(authorityPath)
	contractPath := filepath.Join(dir, "contract-"+filepath.Base(authorityPath))
	writeWeightedAdmissionContractFixture(t, contractPath)
	if err := runAdmissionLiveRouteWeightedAdmissionAuthority([]string{contractPath, authorityPath}); err != nil {
		t.Fatalf("write weighted admission authority fixture: %v", err)
	}
}
