package main

import (
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestAdmissionLiveRouteWeightedAdmissionFinalGate(t *testing.T) {
	dir := t.TempDir()

	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionFinalGate(nil),
		"usage: --admission-live-route-weighted-admission-final-gate SEAL_REPORT FINAL_GATE_REPORT",
	)
	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionFinalGate([]string{"seal.json"}),
		"usage: --admission-live-route-weighted-admission-final-gate SEAL_REPORT FINAL_GATE_REPORT",
	)
	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionFinalGate([]string{"seal.json", "final_gate.json", "extra"}),
		"usage: --admission-live-route-weighted-admission-final-gate SEAL_REPORT FINAL_GATE_REPORT",
	)
	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionFinalGate([]string{"  ", filepath.Join(dir, "final_gate.json")}),
		"weighted admission seal path missing",
	)

	sealPath := filepath.Join(dir, "seal.json")
	writeWeightedAdmissionSealFixture(t, sealPath)

	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionFinalGate([]string{sealPath, "  "}),
		"weighted admission final gate output path missing",
	)

	finalGatePath := filepath.Join(dir, "final_gate.json")
	if err := runAdmissionLiveRouteWeightedAdmissionFinalGate([]string{sealPath, finalGatePath}); err != nil {
		t.Fatalf("valid weighted admission final gate rejected: %v", err)
	}
	raw, err := os.ReadFile(finalGatePath)
	if err != nil {
		t.Fatalf("read weighted admission final gate: %v", err)
	}
	var finalGate admissionLiveRouteWeightedAdmissionFinalGateReport
	if err := json.Unmarshal(raw, &finalGate); err != nil {
		t.Fatalf("decode weighted admission final gate: %v", err)
	}
	if finalGate.Schema != admissionLiveRouteWeightedAdmissionFinalGateSchema ||
		finalGate.Status != "ready_closed_dry_run" ||
		finalGate.Target != "live_route_admission_final_gate" ||
		finalGate.TargetKind != "weighted_live_route_admission_final_gate" ||
		finalGate.TargetMode != "final_gate_closed_dry_run" ||
		finalGate.Action != "verify_weighted_admission_seal_provenance_dry_run" ||
		!finalGate.WeightedAdmissionFinalGateReady ||
		!finalGate.WeightedAdmissionSealConsumed ||
		!finalGate.WeightedAdmissionSealRequired ||
		!finalGate.NextStepBlockedWithoutFinalGate ||
		finalGate.SourceSchema != admissionLiveRouteWeightedAdmissionSealSchema ||
		finalGate.SourceStatus != "sealed_closed_dry_run" ||
		finalGate.SourceTarget != "live_route_admission_seal" ||
		finalGate.SourceReport != sealPath ||
		finalGate.SourcePermitReport == "" ||
		finalGate.SourceAuthorityReport == "" ||
		finalGate.SourceContractReport == "" ||
		finalGate.SourcePreconditionReport == "" ||
		finalGate.SourceReadinessReport == "" ||
		finalGate.SourceBodyWorkdir == "" ||
		finalGate.SourceBoundaryReport == "" ||
		finalGate.SourceProofLog == "" ||
		finalGate.SourceFinalGateLog == "" ||
		!finalGate.SourceWeightedAdmissionSealReady ||
		!finalGate.SourceWeightedAdmissionPermitConsumed ||
		!finalGate.SourceWeightedAdmissionPermitRequired ||
		!finalGate.SourceWeightedAdmissionPermitReady ||
		!finalGate.SourceWeightedAdmissionAuthorityConsumed ||
		!finalGate.SourceWeightedAdmissionAuthorityRequired ||
		!finalGate.SourceManualPermitRequested ||
		!finalGate.SourcePermitKeyMatched ||
		!finalGate.BodySmokeWeighted ||
		!finalGate.NanoDirectRunner ||
		!finalGate.NanoDirectFinalGate ||
		!finalGate.ResonanceGraftAdmissionProof ||
		!finalGate.BoundaryReportFullChain ||
		finalGate.SourceAuthorityGranted ||
		finalGate.AuthorityGranted ||
		finalGate.ContractsReady ||
		finalGate.WriteAllowed ||
		finalGate.AdmissionAllowed ||
		finalGate.LiveAdmissionEnabled ||
		finalGate.MutatesState ||
		!finalGate.Passed ||
		finalGate.Reason != "weighted admission seal cleared final gate; live admission remains disabled" {
		t.Fatalf("weighted admission final gate lost contract: %+v", finalGate)
	}

	openedPath := filepath.Join(dir, "opened_seal.json")
	writeWeightedAdmissionSealFixture(t, openedPath)
	rawOpened, err := os.ReadFile(openedPath)
	if err != nil {
		t.Fatalf("read opened seal fixture: %v", err)
	}
	writeWeightedReadinessFixture(t, openedPath, stringsReplaceFirst(string(rawOpened), `"admission_allowed": false`, `"admission_allowed": true`))
	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionFinalGate([]string{openedPath, filepath.Join(dir, "opened_final_gate.json")}),
		"weighted admission seal opened admission_allowed",
	)

	badSchemaPath := filepath.Join(dir, "bad_schema_seal.json")
	writeWeightedAdmissionSealFixture(t, badSchemaPath)
	rawBadSchema, err := os.ReadFile(badSchemaPath)
	if err != nil {
		t.Fatalf("read bad schema seal fixture: %v", err)
	}
	writeWeightedReadinessFixture(t, badSchemaPath, stringsReplaceFirst(string(rawBadSchema), `"schema": "arianna.live_route_weighted_admission_seal.v1"`, `"schema": "arianna.live_route_weighted_admission_seal.v0"`))
	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionFinalGate([]string{badSchemaPath, filepath.Join(dir, "bad_schema_final_gate.json")}),
		`weighted admission seal schema mismatch: got "arianna.live_route_weighted_admission_seal.v0" want "`+admissionLiveRouteWeightedAdmissionSealSchema+`"`,
	)

	notReadyPath := filepath.Join(dir, "not_ready_seal.json")
	writeWeightedAdmissionSealFixture(t, notReadyPath)
	rawNotReady, err := os.ReadFile(notReadyPath)
	if err != nil {
		t.Fatalf("read not-ready seal fixture: %v", err)
	}
	writeWeightedReadinessFixture(t, notReadyPath, stringsReplaceFirst(string(rawNotReady), `"weighted_admission_seal_ready": true`, `"weighted_admission_seal_ready": false`))
	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionFinalGate([]string{notReadyPath, filepath.Join(dir, "not_ready_final_gate.json")}),
		"weighted admission seal weighted_admission_seal_ready not ready",
	)

	if err := runAdmissionLiveRouteWeightedAdmissionFinalGate([]string{sealPath, filepath.Join(dir, "missing", "final_gate.json")}); err == nil ||
		!strings.HasPrefix(err.Error(), "weighted admission final gate write failed:") {
		t.Fatalf("expected weighted admission final gate write failure, got %v", err)
	}
}
