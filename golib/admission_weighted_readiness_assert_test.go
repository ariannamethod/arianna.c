package main

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestAdmissionLiveRouteWeightedReadinessAssertErrorContract(t *testing.T) {
	dir := t.TempDir()

	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedReadinessAssert(nil),
		"usage: --admission-live-route-weighted-readiness-assert REPORT",
	)
	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedReadinessAssert([]string{"report.json", "extra"}),
		"usage: --admission-live-route-weighted-readiness-assert REPORT",
	)
	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedReadinessAssert([]string{"  "}),
		"weighted readiness report path missing",
	)
	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedReadinessAssert([]string{filepath.Join(dir, "missing.json")}),
		"weighted readiness report not written",
	)

	emptyPath := filepath.Join(dir, "empty.json")
	writeWeightedReadinessFixture(t, emptyPath, "")
	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedReadinessAssert([]string{emptyPath}),
		"weighted readiness report not written",
	)

	invalidPath := filepath.Join(dir, "invalid.json")
	writeWeightedReadinessFixture(t, invalidPath, "{")
	err := runAdmissionLiveRouteWeightedReadinessAssert([]string{invalidPath})
	if err == nil || !strings.HasPrefix(err.Error(), "weighted readiness report JSON invalid:") {
		t.Fatalf("expected invalid JSON error, got %v", err)
	}

	missingSchemaPath := filepath.Join(dir, "missing_schema.json")
	writeWeightedReadinessFixture(t, missingSchemaPath, weightedReadinessFixture(""))
	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedReadinessAssert([]string{missingSchemaPath}),
		"weighted readiness report schema missing",
	)

	badSchemaPath := filepath.Join(dir, "bad_schema.json")
	writeWeightedReadinessFixture(t, badSchemaPath, weightedReadinessFixture(`"schema":"arianna.live_route_weighted_readiness.v0",`))
	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedReadinessAssert([]string{badSchemaPath}),
		`weighted readiness report schema mismatch: got "arianna.live_route_weighted_readiness.v0" want "`+admissionLiveRouteWeightedReadinessSchema+`"`,
	)

	validPath := filepath.Join(dir, "valid.json")
	writeWeightedReadinessFixture(t, validPath, weightedReadinessFixture(`"schema":"`+admissionLiveRouteWeightedReadinessSchema+`",`))
	if err := runAdmissionLiveRouteWeightedReadinessAssert([]string{validPath}); err != nil {
		t.Fatalf("valid weighted readiness report rejected: %v", err)
	}

	badStatusPath := filepath.Join(dir, "bad_status.json")
	writeWeightedReadinessFixture(t, badStatusPath, stringsReplaceFirst(weightedReadinessFixture(`"schema":"`+admissionLiveRouteWeightedReadinessSchema+`",`), `"status":"ready_closed_dry_run"`, `"status":"open"`))
	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedReadinessAssert([]string{badStatusPath}),
		`weighted readiness report status mismatch: got "open" want "ready_closed_dry_run"`,
	)

	notReadyPath := filepath.Join(dir, "not_ready.json")
	writeWeightedReadinessFixture(t, notReadyPath, stringsReplaceFirst(weightedReadinessFixture(`"schema":"`+admissionLiveRouteWeightedReadinessSchema+`",`), `"nano_direct_final_gate":true`, `"nano_direct_final_gate":false`))
	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedReadinessAssert([]string{notReadyPath}),
		"weighted readiness report nano_direct_final_gate not ready",
	)

	openedPath := filepath.Join(dir, "opened.json")
	writeWeightedReadinessFixture(t, openedPath, stringsReplaceFirst(weightedReadinessFixture(`"schema":"`+admissionLiveRouteWeightedReadinessSchema+`",`), `"live_admission_enabled":false`, `"live_admission_enabled":true`))
	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedReadinessAssert([]string{openedPath}),
		"weighted readiness report opened live_admission_enabled",
	)

	missingPathField := filepath.Join(dir, "missing_path.json")
	writeWeightedReadinessFixture(t, missingPathField, stringsReplaceFirst(weightedReadinessFixture(`"schema":"`+admissionLiveRouteWeightedReadinessSchema+`",`), `"proof_log":"/tmp/proof.jsonl"`, `"proof_log":" "`))
	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedReadinessAssert([]string{missingPathField}),
		"weighted readiness report proof_log missing",
	)
}

func writeWeightedReadinessFixture(t *testing.T, path string, content string) {
	t.Helper()
	if err := os.WriteFile(path, []byte(content), 0600); err != nil {
		t.Fatalf("write weighted readiness fixture %s: %v", path, err)
	}
}

func weightedReadinessFixture(schemaFragment string) string {
	return `{
` + schemaFragment + `
"status":"ready_closed_dry_run",
"target":"live_admission",
"body_smoke_weighted":true,
"nano_direct_runner":true,
"nano_direct_final_gate":true,
"resonance_graft_admission_proof":true,
"boundary_report_full_chain":true,
"contracts_ready":false,
"write_allowed":false,
"admission_allowed":false,
"live_admission_enabled":false,
"mutates_state":false,
"body_workdir":"/tmp/body",
"boundary_report":"/tmp/boundary.json",
"proof_log":"/tmp/proof.jsonl",
"final_gate_log":"/tmp/final_gate.jsonl"
}`
}

func stringsReplaceFirst(s string, old string, new string) string {
	idx := strings.Index(s, old)
	if idx < 0 {
		return s
	}
	return s[:idx] + new + s[idx+len(old):]
}
