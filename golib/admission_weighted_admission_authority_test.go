package main

import (
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestAdmissionLiveRouteWeightedAdmissionAuthority(t *testing.T) {
	dir := t.TempDir()

	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionAuthority(nil),
		"usage: --admission-live-route-weighted-admission-authority CONTRACT_REPORT AUTHORITY_REPORT",
	)
	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionAuthority([]string{"contract.json"}),
		"usage: --admission-live-route-weighted-admission-authority CONTRACT_REPORT AUTHORITY_REPORT",
	)
	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionAuthority([]string{"contract.json", "authority.json", "extra"}),
		"usage: --admission-live-route-weighted-admission-authority CONTRACT_REPORT AUTHORITY_REPORT",
	)
	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionAuthority([]string{"  ", filepath.Join(dir, "authority.json")}),
		"weighted admission contract path missing",
	)

	contractPath := filepath.Join(dir, "contract.json")
	writeWeightedAdmissionContractFixture(t, contractPath)

	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionAuthority([]string{contractPath, "  "}),
		"weighted admission authority output path missing",
	)

	authorityPath := filepath.Join(dir, "authority.json")
	if err := runAdmissionLiveRouteWeightedAdmissionAuthority([]string{contractPath, authorityPath}); err != nil {
		t.Fatalf("valid weighted admission authority rejected: %v", err)
	}
	raw, err := os.ReadFile(authorityPath)
	if err != nil {
		t.Fatalf("read weighted admission authority: %v", err)
	}
	var authority admissionLiveRouteWeightedAdmissionAuthorityReport
	if err := json.Unmarshal(raw, &authority); err != nil {
		t.Fatalf("decode weighted admission authority: %v", err)
	}
	if authority.Schema != admissionLiveRouteWeightedAdmissionAuthoritySchema ||
		authority.Status != "authority_receipt_closed_dry_run" ||
		authority.Target != "live_route_admission_authority" ||
		authority.TargetKind != "weighted_live_route_admission_authority" ||
		authority.TargetMode != "closed_authority_dry_run" ||
		authority.Action != "consume_weighted_admission_contract_before_live_authority" ||
		!authority.WeightedAdmissionAuthorityReceiptReady ||
		!authority.WeightedAdmissionContractConsumed ||
		!authority.WeightedAdmissionContractRequired ||
		!authority.NextStepBlockedWithoutAuthority ||
		authority.SourceSchema != admissionLiveRouteWeightedAdmissionContractSchema ||
		authority.SourceStatus != "contract_ready_closed_dry_run" ||
		authority.SourceTarget != "live_route_admission" ||
		authority.SourceReport != contractPath ||
		authority.SourcePreconditionReport == "" ||
		authority.SourceReadinessReport == "" ||
		!authority.BodySmokeWeighted ||
		!authority.NanoDirectRunner ||
		!authority.NanoDirectFinalGate ||
		!authority.ResonanceGraftAdmissionProof ||
		!authority.BoundaryReportFullChain ||
		authority.AuthorityGranted ||
		authority.ContractsReady ||
		authority.WriteAllowed ||
		authority.AdmissionAllowed ||
		authority.LiveAdmissionEnabled ||
		authority.MutatesState ||
		!authority.Passed ||
		authority.Reason != "weighted admission contract consumed; live authority remains disabled" {
		t.Fatalf("weighted admission authority lost contract: %+v", authority)
	}

	openedPath := filepath.Join(dir, "opened_contract.json")
	writeWeightedAdmissionContractFixture(t, openedPath)
	rawOpened, err := os.ReadFile(openedPath)
	if err != nil {
		t.Fatalf("read opened contract fixture: %v", err)
	}
	writeWeightedReadinessFixture(t, openedPath, stringsReplaceFirst(string(rawOpened), `"admission_allowed": false`, `"admission_allowed": true`))
	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionAuthority([]string{openedPath, filepath.Join(dir, "opened_authority.json")}),
		"weighted admission contract opened admission_allowed",
	)

	badSchemaPath := filepath.Join(dir, "bad_schema_contract.json")
	writeWeightedAdmissionContractFixture(t, badSchemaPath)
	rawBadSchema, err := os.ReadFile(badSchemaPath)
	if err != nil {
		t.Fatalf("read bad schema contract fixture: %v", err)
	}
	writeWeightedReadinessFixture(t, badSchemaPath, stringsReplaceFirst(string(rawBadSchema), `"schema": "arianna.live_route_weighted_admission_contract.v1"`, `"schema": "arianna.live_route_weighted_admission_contract.v0"`))
	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionAuthority([]string{badSchemaPath, filepath.Join(dir, "bad_schema_authority.json")}),
		`weighted admission contract schema mismatch: got "arianna.live_route_weighted_admission_contract.v0" want "`+admissionLiveRouteWeightedAdmissionContractSchema+`"`,
	)

	notReadyPath := filepath.Join(dir, "not_ready_contract.json")
	writeWeightedAdmissionContractFixture(t, notReadyPath)
	rawNotReady, err := os.ReadFile(notReadyPath)
	if err != nil {
		t.Fatalf("read not-ready contract fixture: %v", err)
	}
	writeWeightedReadinessFixture(t, notReadyPath, stringsReplaceFirst(string(rawNotReady), `"weighted_admission_contract_ready": true`, `"weighted_admission_contract_ready": false`))
	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionAuthority([]string{notReadyPath, filepath.Join(dir, "not_ready_authority.json")}),
		"weighted admission contract weighted_admission_contract_ready not ready",
	)

	if err := runAdmissionLiveRouteWeightedAdmissionAuthority([]string{contractPath, filepath.Join(dir, "missing", "authority.json")}); err == nil ||
		!strings.HasPrefix(err.Error(), "weighted admission authority write failed:") {
		t.Fatalf("expected weighted admission authority write failure, got %v", err)
	}
}
