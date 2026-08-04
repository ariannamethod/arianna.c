package main

import (
	"os"
	"path/filepath"
	"testing"
)

func TestAdmissionLiveRouteBoundaryReportAssertErrorContract(t *testing.T) {
	dir := t.TempDir()

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
}

func TestAdmissionLiveRouteBoundaryReportFailedDiagnosticsAssertErrorContract(t *testing.T) {
	dir := t.TempDir()

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
