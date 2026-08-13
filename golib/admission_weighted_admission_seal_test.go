package main

import (
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestAdmissionLiveRouteWeightedAdmissionSeal(t *testing.T) {
	dir := t.TempDir()

	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionSeal(nil),
		"usage: --admission-live-route-weighted-admission-seal PERMIT_REPORT SEAL_REPORT",
	)
	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionSeal([]string{"permit.json"}),
		"usage: --admission-live-route-weighted-admission-seal PERMIT_REPORT SEAL_REPORT",
	)
	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionSeal([]string{"permit.json", "seal.json", "extra"}),
		"usage: --admission-live-route-weighted-admission-seal PERMIT_REPORT SEAL_REPORT",
	)
	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionSeal([]string{"  ", filepath.Join(dir, "seal.json")}),
		"weighted admission permit path missing",
	)

	permitPath := filepath.Join(dir, "permit.json")
	writeWeightedAdmissionPermitFixture(t, permitPath)

	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionSeal([]string{permitPath, "  "}),
		"weighted admission seal output path missing",
	)

	sealPath := filepath.Join(dir, "seal.json")
	if err := runAdmissionLiveRouteWeightedAdmissionSeal([]string{permitPath, sealPath}); err != nil {
		t.Fatalf("valid weighted admission seal rejected: %v", err)
	}
	raw, err := os.ReadFile(sealPath)
	if err != nil {
		t.Fatalf("read weighted admission seal: %v", err)
	}
	var seal admissionLiveRouteWeightedAdmissionSealReport
	if err := json.Unmarshal(raw, &seal); err != nil {
		t.Fatalf("decode weighted admission seal: %v", err)
	}
	if seal.Schema != admissionLiveRouteWeightedAdmissionSealSchema ||
		seal.Status != "sealed_closed_dry_run" ||
		seal.Target != "live_route_admission_seal" ||
		seal.TargetKind != "weighted_live_route_admission_seal" ||
		seal.TargetMode != "sealed_closed_dry_run" ||
		seal.Action != "seal_weighted_admission_permit_provenance_dry_run" ||
		!seal.WeightedAdmissionSealReady ||
		!seal.WeightedAdmissionPermitConsumed ||
		!seal.WeightedAdmissionPermitRequired ||
		!seal.NextStepBlockedWithoutSeal ||
		seal.SourceSchema != admissionLiveRouteWeightedAdmissionPermitSchema ||
		seal.SourceStatus != "operator_permitted_closed_dry_run" ||
		seal.SourceTarget != "live_route_admission_permit" ||
		seal.SourceReport != permitPath ||
		seal.SourceAuthorityReport == "" ||
		seal.SourceContractReport == "" ||
		seal.SourcePreconditionReport == "" ||
		seal.SourceReadinessReport == "" ||
		!seal.SourceWeightedAdmissionPermitReady ||
		!seal.SourceWeightedAdmissionAuthorityConsumed ||
		!seal.SourceWeightedAdmissionAuthorityRequired ||
		!seal.SourceManualPermitRequested ||
		!seal.SourcePermitKeyMatched ||
		!seal.BodySmokeWeighted ||
		!seal.NanoDirectRunner ||
		!seal.NanoDirectFinalGate ||
		!seal.ResonanceGraftAdmissionProof ||
		!seal.BoundaryReportFullChain ||
		seal.SourceAuthorityGranted ||
		seal.AuthorityGranted ||
		seal.ContractsReady ||
		seal.WriteAllowed ||
		seal.AdmissionAllowed ||
		seal.LiveAdmissionEnabled ||
		seal.MutatesState ||
		!seal.Passed ||
		seal.Reason != "weighted admission permit sealed as immutable dry-run receipt; live admission remains disabled" {
		t.Fatalf("weighted admission seal lost contract: %+v", seal)
	}

	openedPath := filepath.Join(dir, "opened_permit.json")
	writeWeightedAdmissionPermitFixture(t, openedPath)
	rawOpened, err := os.ReadFile(openedPath)
	if err != nil {
		t.Fatalf("read opened permit fixture: %v", err)
	}
	writeWeightedReadinessFixture(t, openedPath, stringsReplaceFirst(string(rawOpened), `"admission_allowed": false`, `"admission_allowed": true`))
	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionSeal([]string{openedPath, filepath.Join(dir, "opened_seal.json")}),
		"weighted admission permit opened admission_allowed",
	)

	badSchemaPath := filepath.Join(dir, "bad_schema_permit.json")
	writeWeightedAdmissionPermitFixture(t, badSchemaPath)
	rawBadSchema, err := os.ReadFile(badSchemaPath)
	if err != nil {
		t.Fatalf("read bad schema permit fixture: %v", err)
	}
	writeWeightedReadinessFixture(t, badSchemaPath, stringsReplaceFirst(string(rawBadSchema), `"schema": "arianna.live_route_weighted_admission_permit.v1"`, `"schema": "arianna.live_route_weighted_admission_permit.v0"`))
	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionSeal([]string{badSchemaPath, filepath.Join(dir, "bad_schema_seal.json")}),
		`weighted admission permit schema mismatch: got "arianna.live_route_weighted_admission_permit.v0" want "`+admissionLiveRouteWeightedAdmissionPermitSchema+`"`,
	)

	notReadyPath := filepath.Join(dir, "not_ready_permit.json")
	writeWeightedAdmissionPermitFixture(t, notReadyPath)
	rawNotReady, err := os.ReadFile(notReadyPath)
	if err != nil {
		t.Fatalf("read not-ready permit fixture: %v", err)
	}
	writeWeightedReadinessFixture(t, notReadyPath, stringsReplaceFirst(string(rawNotReady), `"weighted_admission_permit_ready": true`, `"weighted_admission_permit_ready": false`))
	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionSeal([]string{notReadyPath, filepath.Join(dir, "not_ready_seal.json")}),
		"weighted admission permit weighted_admission_permit_ready not ready",
	)

	if err := runAdmissionLiveRouteWeightedAdmissionSeal([]string{permitPath, filepath.Join(dir, "missing", "seal.json")}); err == nil ||
		!strings.HasPrefix(err.Error(), "weighted admission seal write failed:") {
		t.Fatalf("expected weighted admission seal write failure, got %v", err)
	}
}
