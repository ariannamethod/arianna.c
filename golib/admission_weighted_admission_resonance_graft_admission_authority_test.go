package main

import (
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionAuthority(t *testing.T) {
	dir := t.TempDir()

	usage := "usage: --admission-live-route-weighted-admission-resonance-graft-admission-authority RESONANCE_GRAFT_ADMISSION_PERMIT_REPORT RESONANCE_GRAFT_ADMISSION_AUTHORITY_REPORT"
	requireBoundaryAssertError(t, runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionAuthority(nil), usage)
	requireBoundaryAssertError(t, runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionAuthority([]string{"permit.json"}), usage)
	requireBoundaryAssertError(t, runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionAuthority([]string{"permit.json", "authority.json", "extra"}), usage)
	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionAuthority([]string{"  ", filepath.Join(dir, "authority.json")}),
		"weighted admission resonance graft admission permit path missing",
	)

	permitPath := filepath.Join(dir, "permit.json")
	writeWeightedAdmissionResonanceGraftAdmissionPermitFixture(t, permitPath)
	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionAuthority([]string{permitPath, "  "}),
		"weighted admission resonance graft admission authority output path missing",
	)

	authorityPath := filepath.Join(dir, "authority.json")
	if err := runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionAuthority([]string{permitPath, authorityPath}); err != nil {
		t.Fatalf("valid weighted admission resonance graft admission authority rejected: %v", err)
	}
	raw, err := os.ReadFile(authorityPath)
	if err != nil {
		t.Fatalf("read weighted admission resonance graft admission authority: %v", err)
	}
	var authority admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionAuthorityReport
	if err := json.Unmarshal(raw, &authority); err != nil {
		t.Fatalf("decode weighted admission resonance graft admission authority: %v", err)
	}
	sourceRaw, err := os.ReadFile(permitPath)
	if err != nil {
		t.Fatalf("read weighted admission resonance graft admission permit: %v", err)
	}
	var sourcePermit admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionPermitReport
	if err := json.Unmarshal(sourceRaw, &sourcePermit); err != nil {
		t.Fatalf("decode weighted admission resonance graft admission permit: %v", err)
	}
	if authority.Schema != admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionAuthoritySchema ||
		authority.Status != "shadow_graft_admission_authority_blocked_dry_run" ||
		authority.Target != "live_route_admission_next_step" ||
		authority.TargetKind != "weighted_internal_world_shadow_graft_admission_authority" ||
		authority.TargetMode != "closed_authority_guard_dry_run" ||
		authority.Action != "block_weighted_resonance_shadow_graft_admission_permit_blocked_dry_run" ||
		authority.WriterAction != "reject_blocked_admission_permit" ||
		authority.RollbackAction != "reject_blocked_admission_permit" ||
		authority.AdmissionAuthorityState != "blocked" ||
		authority.AdmissionAuthorityAction != "reject_blocked_admission_permit" ||
		authority.AdmissionAuthorityTarget != "live_admission_authority" ||
		authority.AdmissionAuthorityTargetKind != "weighted_internal_world_shadow_graft_admission_permit" ||
		authority.AdmissionAuthorityTargetMode != "closed_authority_guard_dry_run" ||
		!authority.AdmissionAuthorityDryRunOnly ||
		authority.AdmissionAuthorityPermitVerified ||
		authority.AdmissionAuthorityLedgerVerified ||
		authority.AdmissionAuthorityWriterReady ||
		authority.AdmissionAuthorityRollbackReady ||
		authority.AdmissionAuthorityReady ||
		authority.AdmissionAuthorityGranted ||
		!authority.WeightedAdmissionResonanceGraftAdmissionAuthorityReady ||
		!authority.WeightedAdmissionResonanceGraftAdmissionPermitConsumed ||
		!authority.WeightedAdmissionResonanceGraftAdmissionPermitRequired ||
		!authority.NextStepBlockedWithoutResonanceGraftAdmissionAuthority ||
		authority.SourceSchema != admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionPermitSchema ||
		authority.SourceStatus != "shadow_graft_admission_permit_blocked_dry_run" ||
		authority.SourceTarget != "live_route_admission_next_step" ||
		authority.SourceReport != permitPath ||
		authority.SourceAdmissionReadinessSchema != admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionReadinessSchema ||
		authority.SourceWeightedAdmissionResonanceGraftAdmissionPermitID != sourcePermit.WeightedAdmissionResonanceGraftAdmissionPermitID ||
		authority.SourceWeightedAdmissionResonanceGraftAdmissionPermitHash != sourcePermit.AdmissionPermitHash ||
		authority.SourceWeightedAdmissionResonanceGraftAdmissionPermitReadBack != sourcePermit.AdmissionPermitReadBackHash ||
		authority.SourceAdmissionPermitReportReceiptShape != sourcePermit.ReceiptShape ||
		authority.SourceAdmissionPermitAction != sourcePermit.AdmissionPermitAction ||
		!authority.SourceAdmissionPermitDryRunOnly ||
		authority.SourceAdmissionPermitReadinessVerified ||
		authority.SourceAdmissionPermitLedgerVerified ||
		authority.SourceAdmissionPermitWriterReady ||
		authority.SourceAdmissionPermitRollbackReady ||
		authority.SourceAdmissionPermitLedgerReady ||
		authority.SourceAdmissionPermitReady ||
		authority.SourceManualPermitRequested ||
		authority.SourcePermitKeyMatched ||
		authority.CausalID != admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionAuthorityCausalID(authority) ||
		authority.AdmissionAuthorityHash != admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionAuthorityHash(authority) ||
		authority.AdmissionAuthorityReadBackHash != admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionAuthorityReadBackHash(authority) ||
		authority.AdmissionAuthorityHash == authority.AdmissionAuthorityReadBackHash ||
		authority.WeightedAdmissionResonanceGraftAdmissionAuthorityID != admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionAuthorityID(authority) ||
		authority.LedgerAppendAllowed ||
		authority.WriteAllowed ||
		authority.AdmissionAllowed ||
		authority.LiveAdmissionEnabled ||
		authority.MutatesState ||
		authority.BodyMutationAllowed ||
		authority.AuthorityGranted ||
		authority.BodyTarget != "none" ||
		!authority.Passed ||
		authority.Reason != "weighted resonance shadow graft admission authority blocked by blocked permit; live authority remains closed" {
		t.Fatalf("weighted admission resonance graft admission authority lost contract: %+v", authority)
	}

	notReadyPath := filepath.Join(dir, "not_ready_permit.json")
	writeWeightedAdmissionResonanceGraftAdmissionPermitFixture(t, notReadyPath)
	writeWeightedReadinessFixture(t, notReadyPath, stringsReplaceFirst(readText(t, notReadyPath), `"weighted_admission_resonance_graft_admission_permit_ready": true`, `"weighted_admission_resonance_graft_admission_permit_ready": false`))
	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionAuthority([]string{notReadyPath, filepath.Join(dir, "not_ready_authority.json")}),
		"weighted admission resonance graft admission permit weighted_admission_resonance_graft_admission_permit_ready not ready",
	)

	openedPermitPath := filepath.Join(dir, "opened_permit.json")
	writeWeightedAdmissionResonanceGraftAdmissionPermitFixture(t, openedPermitPath)
	writeWeightedReadinessFixture(t, openedPermitPath, stringsReplaceFirst(readText(t, openedPermitPath), `"admission_permit_ready": false`, `"admission_permit_ready": true`))
	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionAuthority([]string{openedPermitPath, filepath.Join(dir, "opened_authority.json")}),
		"weighted admission resonance graft admission permit opened admission_permit_ready",
	)

	badSchemaPath := filepath.Join(dir, "bad_schema.json")
	writeWeightedAdmissionResonanceGraftAdmissionPermitFixture(t, badSchemaPath)
	writeWeightedReadinessFixture(t, badSchemaPath, stringsReplaceFirst(readText(t, badSchemaPath), `"schema": "arianna.live_route_weighted_admission_resonance_graft_admission_permit.v1"`, `"schema": "arianna.live_route_weighted_admission_resonance_graft_admission_permit.v0"`))
	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionAuthority([]string{badSchemaPath, filepath.Join(dir, "bad_schema_authority.json")}),
		`weighted admission resonance graft admission permit schema mismatch: got "arianna.live_route_weighted_admission_resonance_graft_admission_permit.v0" want "`+admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionPermitSchema+`"`,
	)

	badHashPath := filepath.Join(dir, "bad_hash.json")
	writeWeightedAdmissionResonanceGraftAdmissionPermitFixture(t, badHashPath)
	writeWeightedReadinessFixture(t, badHashPath, stringsReplaceFirst(readText(t, badHashPath), `"admission_permit_hash": "weighted-resonance-graft-admission-permit-`, `"admission_permit_hash": "weighted-resonance-graft-admission-permit-bad`))
	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionAuthority([]string{badHashPath, filepath.Join(dir, "bad_hash_authority.json")}),
		"weighted admission resonance graft admission permit admission_permit_hash mismatch",
	)

	if err := runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionAuthority([]string{permitPath, filepath.Join(dir, "missing", "authority.json")}); err == nil ||
		!strings.HasPrefix(err.Error(), "weighted admission resonance graft admission authority write failed:") {
		t.Fatalf("expected weighted admission resonance graft admission authority write failure, got %v", err)
	}
}
