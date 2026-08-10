package main

import (
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestAdmissionLiveRouteWeightedAdmissionContract(t *testing.T) {
	dir := t.TempDir()

	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionContract(nil),
		"usage: --admission-live-route-weighted-admission-contract PRECONDITION_REPORT CONTRACT_REPORT",
	)
	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionContract([]string{"precondition.json"}),
		"usage: --admission-live-route-weighted-admission-contract PRECONDITION_REPORT CONTRACT_REPORT",
	)
	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionContract([]string{"precondition.json", "contract.json", "extra"}),
		"usage: --admission-live-route-weighted-admission-contract PRECONDITION_REPORT CONTRACT_REPORT",
	)
	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionContract([]string{"  ", filepath.Join(dir, "contract.json")}),
		"weighted readiness precondition path missing",
	)

	preconditionPath := filepath.Join(dir, "precondition.json")
	writeWeightedReadinessPreconditionReportFixture(t, preconditionPath)

	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionContract([]string{preconditionPath, "  "}),
		"weighted admission contract output path missing",
	)

	contractPath := filepath.Join(dir, "contract.json")
	if err := runAdmissionLiveRouteWeightedAdmissionContract([]string{preconditionPath, contractPath}); err != nil {
		t.Fatalf("valid weighted admission contract rejected: %v", err)
	}
	raw, err := os.ReadFile(contractPath)
	if err != nil {
		t.Fatalf("read weighted admission contract: %v", err)
	}
	var contract admissionLiveRouteWeightedAdmissionContractReport
	if err := json.Unmarshal(raw, &contract); err != nil {
		t.Fatalf("decode weighted admission contract: %v", err)
	}
	if contract.Schema != admissionLiveRouteWeightedAdmissionContractSchema ||
		contract.Status != "contract_ready_closed_dry_run" ||
		contract.Target != "live_route_admission" ||
		contract.TargetKind != "weighted_live_route_admission_contract" ||
		contract.TargetMode != "closed_contract_dry_run" ||
		contract.Action != "bind_weighted_precondition_before_live_route_admission" ||
		!contract.WeightedAdmissionContractReady ||
		!contract.WeightedReadinessPreconditionConsumed ||
		!contract.WeightedReadinessPreconditionRequired ||
		!contract.NextStepBlockedWithoutPrecondition ||
		contract.SourceSchema != admissionLiveRouteWeightedReadinessPreconditionSchema ||
		contract.SourceStatus != "precondition_satisfied_closed_dry_run" ||
		contract.SourceTarget != "live_route_admission_next_step" ||
		contract.SourceReport != preconditionPath ||
		contract.SourceReadinessReport == "" ||
		!contract.BodySmokeWeighted ||
		!contract.NanoDirectRunner ||
		!contract.NanoDirectFinalGate ||
		!contract.ResonanceGraftAdmissionProof ||
		!contract.BoundaryReportFullChain ||
		contract.ContractsReady ||
		contract.WriteAllowed ||
		contract.AdmissionAllowed ||
		contract.LiveAdmissionEnabled ||
		contract.MutatesState ||
		!contract.Passed ||
		contract.Reason != "weighted readiness precondition bound; live route admission remains disabled" {
		t.Fatalf("weighted admission contract lost contract: %+v", contract)
	}

	openedPath := filepath.Join(dir, "opened_precondition.json")
	writeWeightedReadinessPreconditionReportFixture(t, openedPath)
	rawOpened, err := os.ReadFile(openedPath)
	if err != nil {
		t.Fatalf("read opened precondition fixture: %v", err)
	}
	writeWeightedReadinessFixture(t, openedPath, stringsReplaceFirst(string(rawOpened), `"write_allowed": false`, `"write_allowed": true`))
	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionContract([]string{openedPath, filepath.Join(dir, "opened_contract.json")}),
		"weighted readiness precondition opened write_allowed",
	)

	badSchemaPath := filepath.Join(dir, "bad_schema_precondition.json")
	writeWeightedReadinessPreconditionReportFixture(t, badSchemaPath)
	rawBadSchema, err := os.ReadFile(badSchemaPath)
	if err != nil {
		t.Fatalf("read bad schema precondition fixture: %v", err)
	}
	writeWeightedReadinessFixture(t, badSchemaPath, stringsReplaceFirst(string(rawBadSchema), `"schema": "arianna.live_route_weighted_readiness_precondition.v1"`, `"schema": "arianna.live_route_weighted_readiness_precondition.v0"`))
	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionContract([]string{badSchemaPath, filepath.Join(dir, "bad_schema_contract.json")}),
		`weighted readiness precondition schema mismatch: got "arianna.live_route_weighted_readiness_precondition.v0" want "`+admissionLiveRouteWeightedReadinessPreconditionSchema+`"`,
	)

	notConsumedPath := filepath.Join(dir, "not_consumed_precondition.json")
	writeWeightedReadinessPreconditionReportFixture(t, notConsumedPath)
	rawNotConsumed, err := os.ReadFile(notConsumedPath)
	if err != nil {
		t.Fatalf("read not-consumed precondition fixture: %v", err)
	}
	writeWeightedReadinessFixture(t, notConsumedPath, stringsReplaceFirst(string(rawNotConsumed), `"weighted_readiness_consumed": true`, `"weighted_readiness_consumed": false`))
	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionContract([]string{notConsumedPath, filepath.Join(dir, "not_consumed_contract.json")}),
		"weighted readiness precondition weighted_readiness_consumed not ready",
	)

	if err := runAdmissionLiveRouteWeightedAdmissionContract([]string{preconditionPath, filepath.Join(dir, "missing", "contract.json")}); err == nil ||
		!strings.HasPrefix(err.Error(), "weighted admission contract write failed:") {
		t.Fatalf("expected weighted admission contract write failure, got %v", err)
	}
}

func writeWeightedReadinessPreconditionReportFixture(t *testing.T, preconditionPath string) {
	t.Helper()
	dir := filepath.Dir(preconditionPath)
	readinessPath := filepath.Join(dir, "readiness-"+filepath.Base(preconditionPath))
	writeWeightedReadinessFixture(t, readinessPath, weightedReadinessFixture(`"schema":"`+admissionLiveRouteWeightedReadinessSchema+`",`))
	if err := runAdmissionLiveRouteWeightedReadinessPrecondition([]string{readinessPath, preconditionPath}); err != nil {
		t.Fatalf("write weighted readiness precondition fixture: %v", err)
	}
}
