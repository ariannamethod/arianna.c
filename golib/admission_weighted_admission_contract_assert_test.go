package main

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestAdmissionLiveRouteWeightedAdmissionContractAssert(t *testing.T) {
	dir := t.TempDir()

	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionContractAssert(nil),
		"usage: --admission-live-route-weighted-admission-contract-assert REPORT",
	)
	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionContractAssert([]string{"contract.json", "extra"}),
		"usage: --admission-live-route-weighted-admission-contract-assert REPORT",
	)
	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionContractAssert([]string{"  "}),
		"weighted admission contract path missing",
	)
	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionContractAssert([]string{filepath.Join(dir, "missing.json")}),
		"weighted admission contract not written",
	)

	emptyPath := filepath.Join(dir, "empty.json")
	writeWeightedReadinessFixture(t, emptyPath, "")
	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionContractAssert([]string{emptyPath}),
		"weighted admission contract not written",
	)

	invalidPath := filepath.Join(dir, "invalid.json")
	writeWeightedReadinessFixture(t, invalidPath, "{")
	err := runAdmissionLiveRouteWeightedAdmissionContractAssert([]string{invalidPath})
	if err == nil || !strings.HasPrefix(err.Error(), "weighted admission contract JSON invalid:") {
		t.Fatalf("expected invalid JSON error, got %v", err)
	}

	missingSchemaPath := filepath.Join(dir, "missing_schema.json")
	writeWeightedAdmissionContractFixture(t, missingSchemaPath)
	contractText := readText(t, missingSchemaPath)
	writeWeightedReadinessFixture(t, missingSchemaPath, stringsReplaceFirst(contractText, `"schema": "arianna.live_route_weighted_admission_contract.v1",`, ""))
	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionContractAssert([]string{missingSchemaPath}),
		"weighted admission contract schema missing",
	)

	badSchemaPath := filepath.Join(dir, "bad_schema.json")
	writeWeightedAdmissionContractFixture(t, badSchemaPath)
	writeWeightedReadinessFixture(t, badSchemaPath, stringsReplaceFirst(readText(t, badSchemaPath), `"schema": "arianna.live_route_weighted_admission_contract.v1"`, `"schema": "arianna.live_route_weighted_admission_contract.v0"`))
	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionContractAssert([]string{badSchemaPath}),
		`weighted admission contract schema mismatch: got "arianna.live_route_weighted_admission_contract.v0" want "`+admissionLiveRouteWeightedAdmissionContractSchema+`"`,
	)

	validPath := filepath.Join(dir, "valid.json")
	writeWeightedAdmissionContractFixture(t, validPath)
	if err := runAdmissionLiveRouteWeightedAdmissionContractAssert([]string{validPath}); err != nil {
		t.Fatalf("valid weighted admission contract rejected: %v", err)
	}

	badStatusPath := filepath.Join(dir, "bad_status.json")
	writeWeightedAdmissionContractFixture(t, badStatusPath)
	writeWeightedReadinessFixture(t, badStatusPath, stringsReplaceFirst(readText(t, badStatusPath), `"status": "contract_ready_closed_dry_run"`, `"status": "open"`))
	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionContractAssert([]string{badStatusPath}),
		`weighted admission contract status mismatch: got "open" want "contract_ready_closed_dry_run"`,
	)

	notReadyPath := filepath.Join(dir, "not_ready.json")
	writeWeightedAdmissionContractFixture(t, notReadyPath)
	writeWeightedReadinessFixture(t, notReadyPath, stringsReplaceFirst(readText(t, notReadyPath), `"weighted_admission_contract_ready": true`, `"weighted_admission_contract_ready": false`))
	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionContractAssert([]string{notReadyPath}),
		"weighted admission contract weighted_admission_contract_ready not ready",
	)

	openedPath := filepath.Join(dir, "opened.json")
	writeWeightedAdmissionContractFixture(t, openedPath)
	writeWeightedReadinessFixture(t, openedPath, stringsReplaceFirst(readText(t, openedPath), `"live_admission_enabled": false`, `"live_admission_enabled": true`))
	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionContractAssert([]string{openedPath}),
		"weighted admission contract opened live_admission_enabled",
	)

	missingPathField := filepath.Join(dir, "missing_path.json")
	writeWeightedAdmissionContractFixture(t, missingPathField)
	writeWeightedReadinessFixture(t, missingPathField, stringsReplaceFirst(readText(t, missingPathField), `"source_proof_log": "/tmp/proof.jsonl"`, `"source_proof_log": " "`))
	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionContractAssert([]string{missingPathField}),
		"weighted admission contract source_proof_log missing",
	)

	badSourcePath := filepath.Join(dir, "bad_source.json")
	writeWeightedAdmissionContractFixture(t, badSourcePath)
	writeWeightedReadinessFixture(t, badSourcePath, stringsReplaceFirst(readText(t, badSourcePath), `"source_schema": "arianna.live_route_weighted_readiness_precondition.v1"`, `"source_schema": "arianna.live_route_weighted_readiness_precondition.v0"`))
	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionContractAssert([]string{badSourcePath}),
		`weighted admission contract source_schema mismatch: got "arianna.live_route_weighted_readiness_precondition.v0" want "`+admissionLiveRouteWeightedReadinessPreconditionSchema+`"`,
	)
}

func writeWeightedAdmissionContractFixture(t *testing.T, contractPath string) {
	t.Helper()
	dir := filepath.Dir(contractPath)
	preconditionPath := filepath.Join(dir, "precondition-"+filepath.Base(contractPath))
	writeWeightedReadinessPreconditionReportFixture(t, preconditionPath)
	if err := runAdmissionLiveRouteWeightedAdmissionContract([]string{preconditionPath, contractPath}); err != nil {
		t.Fatalf("write weighted admission contract fixture: %v", err)
	}
}

func readText(t *testing.T, path string) string {
	t.Helper()
	raw, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("read %s: %v", path, err)
	}
	return string(raw)
}
