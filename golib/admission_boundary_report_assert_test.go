package main

import (
	"os"
	"path/filepath"
	"testing"
)

func TestAdmissionLiveRouteBoundaryReportAssertErrorContract(t *testing.T) {
	dir := t.TempDir()

	requireBoundaryAssertError(t,
		runAdmissionLiveRouteBoundaryReportAssert([]string{"report-only"}),
		"usage: --admission-live-route-boundary-report-assert REPORT EXPECTED_RECEIPTS [STAGE...]",
	)
	requireBoundaryAssertError(t,
		runAdmissionLiveRouteBoundaryReportAssert([]string{"  ", "1"}),
		"boundary report path missing",
	)

	requireBoundaryAssertError(t,
		runAdmissionLiveRouteBoundaryReportAssert([]string{filepath.Join(dir, "missing.json"), "1", "final_gate"}),
		"boundary report not written",
	)

	emptyPath := filepath.Join(dir, "empty.json")
	writeBoundaryAssertFixture(t, emptyPath, "")
	requireBoundaryAssertError(t,
		runAdmissionLiveRouteBoundaryReportAssert([]string{emptyPath, "1", "final_gate"}),
		"boundary report not written",
	)

	missingSchemaPath := filepath.Join(dir, "missing_schema.json")
	writeBoundaryAssertFixture(t, missingSchemaPath, `{"passed":true,"receipts_checked":1,"boundary":{},"stages":[{"name":"final_gate","passed":true}]}`)
	requireBoundaryAssertError(t,
		runAdmissionLiveRouteBoundaryReportAssert([]string{missingSchemaPath, "1", "final_gate"}),
		"boundary report schema missing",
	)

	badSchemaPath := filepath.Join(dir, "bad_schema.json")
	writeBoundaryAssertFixture(t, badSchemaPath, `{"schema":"arianna.live_route_boundary_report.v0","passed":true,"receipts_checked":1,"boundary":{},"stages":[{"name":"final_gate","passed":true}]}`)
	requireBoundaryAssertError(t,
		runAdmissionLiveRouteBoundaryReportAssert([]string{badSchemaPath, "1", "final_gate"}),
		`boundary report schema mismatch: got "arianna.live_route_boundary_report.v0" want "`+admissionLiveRouteBoundaryReportSchema+`"`,
	)

	missingBoundaryPath := filepath.Join(dir, "missing_boundary.json")
	writeBoundaryAssertFixture(t, missingBoundaryPath, `{"schema":"`+admissionLiveRouteBoundaryReportSchema+`","passed":true,"receipts_checked":1,"boundary":null,"stages":[{"name":"final_gate","passed":true}]}`)
	requireBoundaryAssertError(t,
		runAdmissionLiveRouteBoundaryReportAssert([]string{missingBoundaryPath, "1", "final_gate"}),
		"boundary report projection missing",
	)

	validCompactPath := filepath.Join(dir, "valid_compact.json")
	writeBoundaryAssertFixture(t, validCompactPath, `{"schema":"`+admissionLiveRouteBoundaryReportSchema+`","passed":true,"receipts_checked":1,"boundary":{},"stages":[{"name":"final_gate","passed":true}]}`)
	if err := runAdmissionLiveRouteBoundaryReportAssert([]string{validCompactPath, "1", "final_gate"}); err != nil {
		t.Fatalf("valid compact boundary report rejected: %v", err)
	}

	requireBoundaryAssertError(t,
		runAdmissionLiveRouteBoundaryReportAssert([]string{validCompactPath, "one", "final_gate"}),
		"expected receipt count must be numeric",
	)
	requireBoundaryAssertError(t,
		runAdmissionLiveRouteBoundaryReportAssert([]string{validCompactPath, "2", "final_gate"}),
		"boundary report receipt count mismatch",
	)
	requireBoundaryAssertError(t,
		runAdmissionLiveRouteBoundaryReportAssert([]string{validCompactPath, "1", "  "}),
		"empty boundary report stage name",
	)
	requireBoundaryAssertError(t,
		runAdmissionLiveRouteBoundaryReportAssert([]string{validCompactPath, "1", "missing_stage"}),
		"boundary report stage missing: missing_stage",
	)

	didNotPassPath := filepath.Join(dir, "did_not_pass.json")
	writeBoundaryAssertFixture(t, didNotPassPath, `{"schema":"`+admissionLiveRouteBoundaryReportSchema+`","passed":false,"receipts_checked":1,"boundary":{},"stages":[{"name":"final_gate","passed":true}]}`)
	requireBoundaryAssertError(t,
		runAdmissionLiveRouteBoundaryReportAssert([]string{didNotPassPath, "1", "final_gate"}),
		"boundary report did not pass",
	)

	duplicatedStagePath := filepath.Join(dir, "duplicated_stage.json")
	writeBoundaryAssertFixture(t, duplicatedStagePath, `{"schema":"`+admissionLiveRouteBoundaryReportSchema+`","passed":true,"receipts_checked":2,"boundary":{},"stages":[{"name":"final_gate","passed":true},{"name":"final_gate","passed":true}]}`)
	requireBoundaryAssertError(t,
		runAdmissionLiveRouteBoundaryReportAssert([]string{duplicatedStagePath, "2", "final_gate"}),
		"boundary report stage duplicated: final_gate",
	)

	failedStagePath := filepath.Join(dir, "failed_stage.json")
	writeBoundaryAssertFixture(t, failedStagePath, `{"schema":"`+admissionLiveRouteBoundaryReportSchema+`","passed":true,"receipts_checked":1,"boundary":{},"stages":[{"name":"final_gate","passed":false}]}`)
	requireBoundaryAssertError(t,
		runAdmissionLiveRouteBoundaryReportAssert([]string{failedStagePath, "1", "final_gate"}),
		"boundary report stage did not pass: final_gate",
	)
}

func TestAdmissionLiveRouteBoundaryReportFailedDiagnosticsAssertErrorContract(t *testing.T) {
	dir := t.TempDir()

	requireBoundaryAssertError(t,
		runAdmissionLiveRouteBoundaryReportFailedDiagnosticsAssert([]string{"report-only", "writer_receipt"}),
		"usage: --admission-live-route-boundary-report-failed-diagnostics-assert REPORT STAGE EXPECTED_MISMATCH [EXPECTED_MISMATCH...]",
	)

	requireBoundaryAssertError(t,
		runAdmissionLiveRouteBoundaryReportFailedDiagnosticsAssert([]string{filepath.Join(dir, "missing_failed.json"), "writer_receipt", "route_missing_organs"}),
		"boundary report not written",
	)

	badSchemaPath := filepath.Join(dir, "bad_failed_schema.json")
	writeBoundaryAssertFixture(t, badSchemaPath, `{"schema":"arianna.live_route_boundary_report.v0","passed":false,"receipts_checked":1,"boundary":{},"stages":[{"name":"writer_receipt","passed":false,"mismatches":["route_missing_organs"]}],"reasons":["boundary_mismatch:writer_receipt"]}`)
	requireBoundaryAssertError(t,
		runAdmissionLiveRouteBoundaryReportFailedDiagnosticsAssert([]string{badSchemaPath, "writer_receipt", "route_missing_organs"}),
		`boundary report schema mismatch: got "arianna.live_route_boundary_report.v0" want "`+admissionLiveRouteBoundaryReportSchema+`"`,
	)

	missingMismatchPath := filepath.Join(dir, "missing_failed_mismatch.json")
	writeBoundaryAssertFixture(t, missingMismatchPath, `{"schema":"`+admissionLiveRouteBoundaryReportSchema+`","passed":false,"receipts_checked":1,"boundary":{},"stages":[{"name":"writer_receipt","passed":false,"mismatches":["body_inventory_status"]}],"reasons":["boundary_mismatch:writer_receipt"]}`)
	requireBoundaryAssertError(t,
		runAdmissionLiveRouteBoundaryReportFailedDiagnosticsAssert([]string{missingMismatchPath, "writer_receipt", "route_missing_organs"}),
		"boundary report stage mismatch missing: writer_receipt/route_missing_organs",
	)

	validFailedPath := filepath.Join(dir, "valid_failed_compact.json")
	writeBoundaryAssertFixture(t, validFailedPath, `{"schema":"`+admissionLiveRouteBoundaryReportSchema+`","passed":false,"receipts_checked":1,"boundary":{},"stages":[{"name":"writer_receipt","passed":false,"mismatches":["body_inventory_status","route_missing_organs"]}],"reasons":["boundary_mismatch:writer_receipt"]}`)
	if err := runAdmissionLiveRouteBoundaryReportFailedDiagnosticsAssert([]string{validFailedPath, "writer_receipt", "body_inventory_status", "route_missing_organs"}); err != nil {
		t.Fatalf("valid compact failed boundary report rejected: %v", err)
	}

	requireBoundaryAssertError(t,
		runAdmissionLiveRouteBoundaryReportFailedDiagnosticsAssert([]string{validFailedPath, "  ", "route_missing_organs"}),
		"boundary report stage name missing",
	)
	requireBoundaryAssertError(t,
		runAdmissionLiveRouteBoundaryReportFailedDiagnosticsAssert([]string{validFailedPath, "writer_receipt", "  "}),
		"empty boundary mismatch name",
	)
	requireBoundaryAssertError(t,
		runAdmissionLiveRouteBoundaryReportFailedDiagnosticsAssert([]string{validFailedPath, "missing_stage", "route_missing_organs"}),
		"boundary mismatch reason missing: missing_stage",
	)

	missingStagePath := filepath.Join(dir, "missing_stage_failed.json")
	writeBoundaryAssertFixture(t, missingStagePath, `{"schema":"`+admissionLiveRouteBoundaryReportSchema+`","passed":false,"receipts_checked":1,"boundary":{},"stages":[{"name":"other_stage","passed":false,"mismatches":["route_missing_organs"]}],"reasons":["boundary_mismatch:writer_receipt"]}`)
	requireBoundaryAssertError(t,
		runAdmissionLiveRouteBoundaryReportFailedDiagnosticsAssert([]string{missingStagePath, "writer_receipt", "route_missing_organs"}),
		"boundary report stage missing: writer_receipt",
	)

	didNotFailPath := filepath.Join(dir, "did_not_fail.json")
	writeBoundaryAssertFixture(t, didNotFailPath, `{"schema":"`+admissionLiveRouteBoundaryReportSchema+`","passed":true,"receipts_checked":1,"boundary":{},"stages":[{"name":"writer_receipt","passed":false,"mismatches":["route_missing_organs"]}],"reasons":["boundary_mismatch:writer_receipt"]}`)
	requireBoundaryAssertError(t,
		runAdmissionLiveRouteBoundaryReportFailedDiagnosticsAssert([]string{didNotFailPath, "writer_receipt", "route_missing_organs"}),
		"boundary report did not fail",
	)

	missingReasonPath := filepath.Join(dir, "missing_reason.json")
	writeBoundaryAssertFixture(t, missingReasonPath, `{"schema":"`+admissionLiveRouteBoundaryReportSchema+`","passed":false,"receipts_checked":1,"boundary":{},"stages":[{"name":"writer_receipt","passed":false,"mismatches":["route_missing_organs"]}],"reasons":["other_failure"]}`)
	requireBoundaryAssertError(t,
		runAdmissionLiveRouteBoundaryReportFailedDiagnosticsAssert([]string{missingReasonPath, "writer_receipt", "route_missing_organs"}),
		"boundary mismatch reason missing: writer_receipt",
	)

	duplicatedStagePath := filepath.Join(dir, "duplicated_failed_stage.json")
	writeBoundaryAssertFixture(t, duplicatedStagePath, `{"schema":"`+admissionLiveRouteBoundaryReportSchema+`","passed":false,"receipts_checked":2,"boundary":{},"stages":[{"name":"writer_receipt","passed":false,"mismatches":["route_missing_organs"]},{"name":"writer_receipt","passed":false,"mismatches":["route_missing_organs"]}],"reasons":["boundary_mismatch:writer_receipt"]}`)
	requireBoundaryAssertError(t,
		runAdmissionLiveRouteBoundaryReportFailedDiagnosticsAssert([]string{duplicatedStagePath, "writer_receipt", "route_missing_organs"}),
		"boundary report stage duplicated: writer_receipt",
	)

	stageDidNotFailPath := filepath.Join(dir, "stage_did_not_fail.json")
	writeBoundaryAssertFixture(t, stageDidNotFailPath, `{"schema":"`+admissionLiveRouteBoundaryReportSchema+`","passed":false,"receipts_checked":1,"boundary":{},"stages":[{"name":"writer_receipt","passed":true,"mismatches":["route_missing_organs"]}],"reasons":["boundary_mismatch:writer_receipt"]}`)
	requireBoundaryAssertError(t,
		runAdmissionLiveRouteBoundaryReportFailedDiagnosticsAssert([]string{stageDidNotFailPath, "writer_receipt", "route_missing_organs"}),
		"boundary report stage did not fail: writer_receipt",
	)

	stageMismatchesMissingPath := filepath.Join(dir, "stage_mismatches_missing.json")
	writeBoundaryAssertFixture(t, stageMismatchesMissingPath, `{"schema":"`+admissionLiveRouteBoundaryReportSchema+`","passed":false,"receipts_checked":1,"boundary":{},"stages":[{"name":"writer_receipt","passed":false}],"reasons":["boundary_mismatch:writer_receipt"]}`)
	requireBoundaryAssertError(t,
		runAdmissionLiveRouteBoundaryReportFailedDiagnosticsAssert([]string{stageMismatchesMissingPath, "writer_receipt", "route_missing_organs"}),
		"boundary report stage mismatches missing: writer_receipt",
	)
}

func writeBoundaryAssertFixture(t *testing.T, path string, content string) {
	t.Helper()
	if err := os.WriteFile(path, []byte(content), 0600); err != nil {
		t.Fatalf("write boundary assertion fixture %s: %v", path, err)
	}
}

func requireBoundaryAssertError(t *testing.T, err error, want string) {
	t.Helper()
	if err == nil {
		t.Fatalf("expected error %q, got nil", want)
	}
	if err.Error() != want {
		t.Fatalf("wrong error:\n got: %q\nwant: %q", err.Error(), want)
	}
}
