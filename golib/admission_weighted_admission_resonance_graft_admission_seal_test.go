package main

import (
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionSeal(t *testing.T) {
	dir := t.TempDir()

	usage := "usage: --admission-live-route-weighted-admission-resonance-graft-admission-seal RESONANCE_GRAFT_ADMISSION_AUTHORITY_REPORT RESONANCE_GRAFT_ADMISSION_SEAL_REPORT"
	requireBoundaryAssertError(t, runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionSeal(nil), usage)
	requireBoundaryAssertError(t, runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionSeal([]string{"authority.json"}), usage)
	requireBoundaryAssertError(t, runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionSeal([]string{"authority.json", "seal.json", "extra"}), usage)
	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionSeal([]string{"  ", filepath.Join(dir, "seal.json")}),
		"weighted admission resonance graft admission authority path missing",
	)

	authorityPath := filepath.Join(dir, "authority.json")
	writeWeightedAdmissionResonanceGraftAdmissionAuthorityFixture(t, authorityPath)
	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionSeal([]string{authorityPath, "  "}),
		"weighted admission resonance graft admission seal output path missing",
	)

	sealPath := filepath.Join(dir, "seal.json")
	if err := runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionSeal([]string{authorityPath, sealPath}); err != nil {
		t.Fatalf("valid weighted admission resonance graft admission seal rejected: %v", err)
	}
	raw, err := os.ReadFile(sealPath)
	if err != nil {
		t.Fatalf("read weighted admission resonance graft admission seal: %v", err)
	}
	var seal admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionSealReport
	if err := json.Unmarshal(raw, &seal); err != nil {
		t.Fatalf("decode weighted admission resonance graft admission seal: %v", err)
	}
	sourceRaw, err := os.ReadFile(authorityPath)
	if err != nil {
		t.Fatalf("read weighted admission resonance graft admission authority: %v", err)
	}
	var sourceAuthority admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionAuthorityReport
	if err := json.Unmarshal(sourceRaw, &sourceAuthority); err != nil {
		t.Fatalf("decode weighted admission resonance graft admission authority: %v", err)
	}
	if seal.Schema != admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionSealSchema ||
		seal.Status != "shadow_graft_admission_seal_blocked_dry_run" ||
		seal.Target != "live_route_admission_next_step" ||
		seal.TargetKind != "weighted_internal_world_shadow_graft_admission_seal" ||
		seal.TargetMode != "closed_seal_guard_dry_run" ||
		seal.Action != "seal_weighted_resonance_shadow_graft_admission_authority_blocked_dry_run" ||
		seal.WriterAction != "reject_blocked_admission_authority" ||
		seal.RollbackAction != "reject_blocked_admission_authority" ||
		seal.AdmissionSealState != "sealed" ||
		seal.AdmissionSealAction != "seal_blocked_admission_authority" ||
		seal.AdmissionSealTarget != "live_admission_authority" ||
		seal.AdmissionSealTargetKind != "weighted_internal_world_shadow_graft_admission_authority" ||
		seal.AdmissionSealTargetMode != "closed_seal_guard_dry_run" ||
		!seal.AdmissionSealDryRunOnly ||
		seal.AdmissionSealAuthorityVerified ||
		seal.AdmissionSealPermitVerified ||
		seal.AdmissionSealLedgerVerified ||
		seal.AdmissionSealReady ||
		!seal.AdmissionSealImmutableReceipt ||
		!seal.WeightedAdmissionResonanceGraftAdmissionSealReady ||
		!seal.WeightedAdmissionResonanceGraftAdmissionAuthorityConsumed ||
		!seal.WeightedAdmissionResonanceGraftAdmissionAuthorityRequired ||
		!seal.NextStepBlockedWithoutResonanceGraftAdmissionSeal ||
		seal.SourceSchema != admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionAuthoritySchema ||
		seal.SourceStatus != "shadow_graft_admission_authority_blocked_dry_run" ||
		seal.SourceTarget != "live_route_admission_next_step" ||
		seal.SourceReport != authorityPath ||
		seal.SourceAdmissionPermitSchema != admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionPermitSchema ||
		seal.SourceWeightedAdmissionResonanceGraftAdmissionAuthorityID != sourceAuthority.WeightedAdmissionResonanceGraftAdmissionAuthorityID ||
		seal.SourceWeightedAdmissionResonanceGraftAdmissionAuthorityHash != sourceAuthority.AdmissionAuthorityHash ||
		seal.SourceWeightedAdmissionResonanceGraftAdmissionAuthorityReadBack != sourceAuthority.AdmissionAuthorityReadBackHash ||
		seal.SourceAdmissionAuthorityReportReceiptShape != sourceAuthority.ReceiptShape ||
		seal.SourceAdmissionAuthorityAction != sourceAuthority.AdmissionAuthorityAction ||
		!seal.SourceAdmissionAuthorityDryRunOnly ||
		seal.SourceAdmissionAuthorityPermitVerified ||
		seal.SourceAdmissionAuthorityLedgerVerified ||
		seal.SourceAdmissionAuthorityWriterReady ||
		seal.SourceAdmissionAuthorityRollbackReady ||
		seal.SourceAdmissionAuthorityReady ||
		seal.SourceAdmissionAuthorityGranted ||
		seal.CausalID != admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionSealCausalID(seal) ||
		seal.AdmissionSealHash != admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionSealHash(seal) ||
		seal.AdmissionSealReadBackHash != admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionSealReadBackHash(seal) ||
		seal.AdmissionSealHash == seal.AdmissionSealReadBackHash ||
		seal.WeightedAdmissionResonanceGraftAdmissionSealID != admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionSealID(seal) ||
		seal.LedgerAppendAllowed ||
		seal.WriteAllowed ||
		seal.AdmissionAllowed ||
		seal.LiveAdmissionEnabled ||
		seal.MutatesState ||
		seal.BodyMutationAllowed ||
		seal.AuthorityGranted ||
		seal.BodyTarget != "none" ||
		!seal.Passed ||
		seal.Reason != "weighted resonance shadow graft admission seal fixed blocked authority provenance; live authority remains closed" {
		t.Fatalf("weighted admission resonance graft admission seal lost contract: %+v", seal)
	}

	notReadyPath := filepath.Join(dir, "not_ready_authority.json")
	writeWeightedAdmissionResonanceGraftAdmissionAuthorityFixture(t, notReadyPath)
	writeWeightedReadinessFixture(t, notReadyPath, stringsReplaceFirst(readText(t, notReadyPath), `"weighted_admission_resonance_graft_admission_authority_ready": true`, `"weighted_admission_resonance_graft_admission_authority_ready": false`))
	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionSeal([]string{notReadyPath, filepath.Join(dir, "not_ready_seal.json")}),
		"weighted admission resonance graft admission authority weighted_admission_resonance_graft_admission_authority_ready not ready",
	)

	openedAuthorityPath := filepath.Join(dir, "opened_authority.json")
	writeWeightedAdmissionResonanceGraftAdmissionAuthorityFixture(t, openedAuthorityPath)
	writeWeightedReadinessFixture(t, openedAuthorityPath, stringsReplaceFirst(readText(t, openedAuthorityPath), `"admission_authority_granted": false`, `"admission_authority_granted": true`))
	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionSeal([]string{openedAuthorityPath, filepath.Join(dir, "opened_seal.json")}),
		"weighted admission resonance graft admission authority opened admission_authority_granted",
	)

	badSchemaPath := filepath.Join(dir, "bad_schema.json")
	writeWeightedAdmissionResonanceGraftAdmissionAuthorityFixture(t, badSchemaPath)
	writeWeightedReadinessFixture(t, badSchemaPath, stringsReplaceFirst(readText(t, badSchemaPath), `"schema": "arianna.live_route_weighted_admission_resonance_graft_admission_authority.v1"`, `"schema": "arianna.live_route_weighted_admission_resonance_graft_admission_authority.v0"`))
	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionSeal([]string{badSchemaPath, filepath.Join(dir, "bad_schema_seal.json")}),
		`weighted admission resonance graft admission authority schema mismatch: got "arianna.live_route_weighted_admission_resonance_graft_admission_authority.v0" want "`+admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionAuthoritySchema+`"`,
	)

	badHashPath := filepath.Join(dir, "bad_hash.json")
	writeWeightedAdmissionResonanceGraftAdmissionAuthorityFixture(t, badHashPath)
	writeWeightedReadinessFixture(t, badHashPath, stringsReplaceFirst(readText(t, badHashPath), `"admission_authority_hash": "weighted-resonance-graft-admission-authority-`, `"admission_authority_hash": "weighted-resonance-graft-admission-authority-bad`))
	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionSeal([]string{badHashPath, filepath.Join(dir, "bad_hash_seal.json")}),
		"weighted admission resonance graft admission authority admission_authority_hash mismatch",
	)

	if err := runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionSeal([]string{authorityPath, filepath.Join(dir, "missing", "seal.json")}); err == nil ||
		!strings.HasPrefix(err.Error(), "weighted admission resonance graft admission seal write failed:") {
		t.Fatalf("expected weighted admission resonance graft admission seal write failure, got %v", err)
	}
}
