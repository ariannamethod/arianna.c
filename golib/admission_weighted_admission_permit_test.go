package main

import (
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestAdmissionLiveRouteWeightedAdmissionPermit(t *testing.T) {
	dir := t.TempDir()

	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionPermit(nil),
		"usage: --admission-live-route-weighted-admission-permit AUTHORITY_REPORT PERMIT_REPORT",
	)
	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionPermit([]string{"authority.json"}),
		"usage: --admission-live-route-weighted-admission-permit AUTHORITY_REPORT PERMIT_REPORT",
	)
	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionPermit([]string{"authority.json", "permit.json", "extra"}),
		"usage: --admission-live-route-weighted-admission-permit AUTHORITY_REPORT PERMIT_REPORT",
	)
	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionPermit([]string{"  ", filepath.Join(dir, "permit.json")}),
		"weighted admission authority path missing",
	)

	authorityPath := filepath.Join(dir, "authority.json")
	writeWeightedAdmissionAuthorityFixture(t, authorityPath)

	t.Setenv("A2A_WEIGHTED_ADMISSION_PERMIT_KEY", "wrong")
	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionPermit([]string{authorityPath, filepath.Join(dir, "wrong_key_permit.json")}),
		"A2A_WEIGHTED_ADMISSION_PERMIT_KEY must match dry-run confirmation for weighted admission permit",
	)

	t.Setenv("A2A_WEIGHTED_ADMISSION_PERMIT_KEY", weightedAdmissionPermitKey)
	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionPermit([]string{authorityPath, "  "}),
		"weighted admission permit output path missing",
	)

	permitPath := filepath.Join(dir, "permit.json")
	if err := runAdmissionLiveRouteWeightedAdmissionPermit([]string{authorityPath, permitPath}); err != nil {
		t.Fatalf("valid weighted admission permit rejected: %v", err)
	}
	raw, err := os.ReadFile(permitPath)
	if err != nil {
		t.Fatalf("read weighted admission permit: %v", err)
	}
	var permit admissionLiveRouteWeightedAdmissionPermitReport
	if err := json.Unmarshal(raw, &permit); err != nil {
		t.Fatalf("decode weighted admission permit: %v", err)
	}
	if permit.Schema != admissionLiveRouteWeightedAdmissionPermitSchema ||
		permit.Status != "operator_permitted_closed_dry_run" ||
		permit.Target != "live_route_admission_permit" ||
		permit.TargetKind != "weighted_live_route_admission_permit" ||
		permit.TargetMode != "permit_closed_dry_run" ||
		permit.Action != "acknowledge_closed_weighted_admission_authority_dry_run" ||
		!permit.WeightedAdmissionPermitReady ||
		!permit.WeightedAdmissionAuthorityConsumed ||
		!permit.WeightedAdmissionAuthorityRequired ||
		!permit.ManualPermitRequested ||
		!permit.PermitKeyMatched ||
		!permit.NextStepBlockedWithoutPermit ||
		permit.SourceSchema != admissionLiveRouteWeightedAdmissionAuthoritySchema ||
		permit.SourceStatus != "authority_receipt_closed_dry_run" ||
		permit.SourceTarget != "live_route_admission_authority" ||
		permit.SourceReport != authorityPath ||
		permit.SourceContractReport == "" ||
		permit.SourcePreconditionReport == "" ||
		permit.SourceReadinessReport == "" ||
		!permit.BodySmokeWeighted ||
		!permit.NanoDirectRunner ||
		!permit.NanoDirectFinalGate ||
		!permit.ResonanceGraftAdmissionProof ||
		!permit.BoundaryReportFullChain ||
		permit.SourceAuthorityGranted ||
		permit.AuthorityGranted ||
		permit.ContractsReady ||
		permit.WriteAllowed ||
		permit.AdmissionAllowed ||
		permit.LiveAdmissionEnabled ||
		permit.MutatesState ||
		!permit.Passed ||
		permit.Reason != "operator permit accepted for closed weighted authority; live admission remains disabled" {
		t.Fatalf("weighted admission permit lost contract: %+v", permit)
	}

	openedPath := filepath.Join(dir, "opened_authority.json")
	writeWeightedAdmissionAuthorityFixture(t, openedPath)
	rawOpened, err := os.ReadFile(openedPath)
	if err != nil {
		t.Fatalf("read opened authority fixture: %v", err)
	}
	writeWeightedReadinessFixture(t, openedPath, stringsReplaceFirst(string(rawOpened), `"authority_granted": false`, `"authority_granted": true`))
	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionPermit([]string{openedPath, filepath.Join(dir, "opened_permit.json")}),
		"weighted admission authority opened authority_granted",
	)

	badSchemaPath := filepath.Join(dir, "bad_schema_authority.json")
	writeWeightedAdmissionAuthorityFixture(t, badSchemaPath)
	rawBadSchema, err := os.ReadFile(badSchemaPath)
	if err != nil {
		t.Fatalf("read bad schema authority fixture: %v", err)
	}
	writeWeightedReadinessFixture(t, badSchemaPath, stringsReplaceFirst(string(rawBadSchema), `"schema": "arianna.live_route_weighted_admission_authority.v1"`, `"schema": "arianna.live_route_weighted_admission_authority.v0"`))
	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionPermit([]string{badSchemaPath, filepath.Join(dir, "bad_schema_permit.json")}),
		`weighted admission authority schema mismatch: got "arianna.live_route_weighted_admission_authority.v0" want "`+admissionLiveRouteWeightedAdmissionAuthoritySchema+`"`,
	)

	notReadyPath := filepath.Join(dir, "not_ready_authority.json")
	writeWeightedAdmissionAuthorityFixture(t, notReadyPath)
	rawNotReady, err := os.ReadFile(notReadyPath)
	if err != nil {
		t.Fatalf("read not-ready authority fixture: %v", err)
	}
	writeWeightedReadinessFixture(t, notReadyPath, stringsReplaceFirst(string(rawNotReady), `"weighted_admission_authority_receipt_ready": true`, `"weighted_admission_authority_receipt_ready": false`))
	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionPermit([]string{notReadyPath, filepath.Join(dir, "not_ready_permit.json")}),
		"weighted admission authority weighted_admission_authority_receipt_ready not ready",
	)

	if err := runAdmissionLiveRouteWeightedAdmissionPermit([]string{authorityPath, filepath.Join(dir, "missing", "permit.json")}); err == nil ||
		!strings.HasPrefix(err.Error(), "weighted admission permit write failed:") {
		t.Fatalf("expected weighted admission permit write failure, got %v", err)
	}
}
