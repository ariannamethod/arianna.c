package main

import (
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestAdmissionLiveRouteWeightedReadinessPreconditionContract(t *testing.T) {
	dir := t.TempDir()

	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedReadinessPrecondition(nil),
		"usage: --admission-live-route-weighted-readiness-precondition READINESS_REPORT PRECONDITION_REPORT",
	)
	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedReadinessPrecondition([]string{"readiness.json"}),
		"usage: --admission-live-route-weighted-readiness-precondition READINESS_REPORT PRECONDITION_REPORT",
	)
	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedReadinessPrecondition([]string{"readiness.json", "out.json", "extra"}),
		"usage: --admission-live-route-weighted-readiness-precondition READINESS_REPORT PRECONDITION_REPORT",
	)
	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedReadinessPrecondition([]string{"  ", filepath.Join(dir, "precondition.json")}),
		"weighted readiness report path missing",
	)

	readinessPath := filepath.Join(dir, "readiness.json")
	writeWeightedReadinessFixture(t, readinessPath, weightedReadinessFixture(`"schema":"`+admissionLiveRouteWeightedReadinessSchema+`",`))

	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedReadinessPrecondition([]string{readinessPath, "  "}),
		"weighted readiness precondition output path missing",
	)

	preconditionPath := filepath.Join(dir, "precondition.json")
	if err := runAdmissionLiveRouteWeightedReadinessPrecondition([]string{readinessPath, preconditionPath}); err != nil {
		t.Fatalf("valid weighted readiness precondition rejected: %v", err)
	}
	raw, err := os.ReadFile(preconditionPath)
	if err != nil {
		t.Fatalf("read precondition report: %v", err)
	}
	var precondition admissionLiveRouteWeightedReadinessPreconditionReport
	if err := json.Unmarshal(raw, &precondition); err != nil {
		t.Fatalf("decode precondition report: %v", err)
	}
	if precondition.Schema != admissionLiveRouteWeightedReadinessPreconditionSchema ||
		precondition.Status != "precondition_satisfied_closed_dry_run" ||
		precondition.Target != "live_route_admission_next_step" ||
		precondition.TargetKind != "weighted_pre_live_admission_precondition" ||
		precondition.TargetMode != "closed_dry_run" ||
		precondition.Action != "consume_weighted_readiness_before_live_route_admission" ||
		!precondition.WeightedReadinessConsumed ||
		!precondition.WeightedReadinessRequired ||
		!precondition.NextStepBlockedWithoutReadiness ||
		precondition.SourceSchema != admissionLiveRouteWeightedReadinessSchema ||
		precondition.SourceStatus != "ready_closed_dry_run" ||
		precondition.SourceTarget != "live_admission" ||
		precondition.SourceReport != readinessPath ||
		!precondition.BodySmokeWeighted ||
		!precondition.NanoDirectRunner ||
		!precondition.NanoDirectFinalGate ||
		!precondition.ResonanceGraftAdmissionProof ||
		!precondition.BoundaryReportFullChain ||
		precondition.ContractsReady ||
		precondition.WriteAllowed ||
		precondition.AdmissionAllowed ||
		precondition.LiveAdmissionEnabled ||
		precondition.MutatesState ||
		!precondition.Passed ||
		precondition.Reason != "weighted readiness consumed; live route admission remains disabled" {
		t.Fatalf("weighted readiness precondition report lost contract: %+v", precondition)
	}

	openedPath := filepath.Join(dir, "opened_readiness.json")
	writeWeightedReadinessFixture(t, openedPath, stringsReplaceFirst(weightedReadinessFixture(`"schema":"`+admissionLiveRouteWeightedReadinessSchema+`",`), `"write_allowed":false`, `"write_allowed":true`))
	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedReadinessPrecondition([]string{openedPath, filepath.Join(dir, "opened_precondition.json")}),
		"weighted readiness report opened write_allowed",
	)

	badSchemaPath := filepath.Join(dir, "bad_schema_readiness.json")
	writeWeightedReadinessFixture(t, badSchemaPath, weightedReadinessFixture(`"schema":"arianna.live_route_weighted_readiness.v0",`))
	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedReadinessPrecondition([]string{badSchemaPath, filepath.Join(dir, "bad_schema_precondition.json")}),
		`weighted readiness report schema mismatch: got "arianna.live_route_weighted_readiness.v0" want "`+admissionLiveRouteWeightedReadinessSchema+`"`,
	)

	if err := runAdmissionLiveRouteWeightedReadinessPrecondition([]string{readinessPath, filepath.Join(dir, "missing", "precondition.json")}); err == nil ||
		!strings.HasPrefix(err.Error(), "weighted readiness precondition write failed:") {
		t.Fatalf("expected precondition write failure, got %v", err)
	}
}
