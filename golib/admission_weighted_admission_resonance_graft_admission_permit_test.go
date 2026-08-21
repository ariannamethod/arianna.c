package main

import (
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionPermit(t *testing.T) {
	dir := t.TempDir()

	usage := "usage: --admission-live-route-weighted-admission-resonance-graft-admission-permit RESONANCE_GRAFT_ADMISSION_READINESS_REPORT RESONANCE_GRAFT_ADMISSION_PERMIT_REPORT"
	requireBoundaryAssertError(t, runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionPermit(nil), usage)
	requireBoundaryAssertError(t, runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionPermit([]string{"readiness.json"}), usage)
	requireBoundaryAssertError(t, runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionPermit([]string{"readiness.json", "permit.json", "extra"}), usage)
	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionPermit([]string{"  ", filepath.Join(dir, "permit.json")}),
		"weighted admission resonance graft admission readiness path missing",
	)

	readinessPath := filepath.Join(dir, "readiness.json")
	writeWeightedAdmissionResonanceGraftAdmissionReadinessFixture(t, readinessPath)
	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionPermit([]string{readinessPath, "  "}),
		"weighted admission resonance graft admission permit output path missing",
	)

	permitPath := filepath.Join(dir, "permit.json")
	if err := runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionPermit([]string{readinessPath, permitPath}); err != nil {
		t.Fatalf("valid weighted admission resonance graft admission permit rejected: %v", err)
	}
	raw, err := os.ReadFile(permitPath)
	if err != nil {
		t.Fatalf("read weighted admission resonance graft admission permit: %v", err)
	}
	var permit admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionPermitReport
	if err := json.Unmarshal(raw, &permit); err != nil {
		t.Fatalf("decode weighted admission resonance graft admission permit: %v", err)
	}
	sourceRaw, err := os.ReadFile(readinessPath)
	if err != nil {
		t.Fatalf("read weighted admission resonance graft admission readiness: %v", err)
	}
	var sourceReadiness admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionReadinessReport
	if err := json.Unmarshal(sourceRaw, &sourceReadiness); err != nil {
		t.Fatalf("decode weighted admission resonance graft admission readiness: %v", err)
	}
	if permit.Schema != admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionPermitSchema ||
		permit.Status != "shadow_graft_admission_permit_blocked_dry_run" ||
		permit.TargetKind != "weighted_internal_world_shadow_graft_admission_permit" ||
		permit.TargetMode != "closed_permit_guard_dry_run" ||
		permit.Action != "block_weighted_resonance_shadow_graft_admission_readiness_blocked_dry_run" ||
		permit.WriterAction != "reject_blocked_admission_readiness" ||
		permit.RollbackAction != "reject_blocked_admission_readiness" ||
		permit.AdmissionPermitState != "blocked" ||
		permit.AdmissionPermitAction != "reject_blocked_admission_readiness" ||
		permit.AdmissionPermitTarget != "live_admission" ||
		permit.AdmissionPermitTargetKind != "weighted_internal_world_shadow_graft_admission_readiness" ||
		permit.AdmissionPermitTargetMode != "closed_permit_guard_dry_run" ||
		!permit.AdmissionPermitDryRunOnly ||
		permit.AdmissionPermitReadinessVerified ||
		permit.AdmissionPermitLedgerVerified ||
		permit.AdmissionPermitWriterReady ||
		permit.AdmissionPermitRollbackReady ||
		permit.AdmissionPermitLedgerReady ||
		permit.AdmissionPermitReady ||
		permit.ManualPermitRequested ||
		permit.PermitKeyMatched ||
		!permit.WeightedAdmissionResonanceGraftAdmissionPermitReady ||
		!permit.WeightedAdmissionResonanceGraftAdmissionReadinessConsumed ||
		!permit.WeightedAdmissionResonanceGraftAdmissionReadinessRequired ||
		!permit.NextStepBlockedWithoutResonanceGraftAdmissionPermit ||
		permit.SourceSchema != admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionReadinessSchema ||
		permit.SourceStatus != "shadow_graft_admission_readiness_blocked_dry_run" ||
		permit.SourceReport != readinessPath ||
		permit.SourceLedgerVerificationSchema != admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionLedgerVerificationSchema ||
		permit.SourceWeightedAdmissionResonanceGraftAdmissionReadinessID != sourceReadiness.WeightedAdmissionResonanceGraftAdmissionReadinessID ||
		permit.SourceWeightedAdmissionResonanceGraftAdmissionReadinessHash != sourceReadiness.AdmissionReadinessHash ||
		permit.SourceWeightedAdmissionResonanceGraftAdmissionReadinessReadBack != sourceReadiness.AdmissionReadinessReadBackHash ||
		permit.SourceAdmissionReadinessReportReceiptShape != sourceReadiness.ReceiptShape ||
		permit.SourceAdmissionReadinessAction != sourceReadiness.AdmissionReadinessAction ||
		!permit.SourceAdmissionReadinessDryRunOnly ||
		permit.SourceAdmissionReadinessLedgerVerified ||
		permit.SourceAdmissionReadinessWriterReady ||
		permit.SourceAdmissionReadinessRollbackReady ||
		permit.SourceAdmissionReadinessLedgerReady ||
		permit.SourceAdmissionReadinessReady ||
		permit.CausalID != admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionPermitCausalID(permit) ||
		permit.AdmissionPermitHash != admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionPermitHash(permit) ||
		permit.AdmissionPermitReadBackHash != admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionPermitReadBackHash(permit) ||
		permit.AdmissionPermitHash == permit.AdmissionPermitReadBackHash ||
		permit.WeightedAdmissionResonanceGraftAdmissionPermitID != admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionPermitID(permit) ||
		permit.LedgerAppendAllowed ||
		permit.WriteAllowed ||
		permit.AdmissionAllowed ||
		permit.LiveAdmissionEnabled ||
		permit.MutatesState ||
		permit.BodyMutationAllowed ||
		permit.BodyTarget != "none" ||
		!permit.Passed ||
		permit.Reason != "weighted resonance shadow graft admission permit blocked by blocked readiness; manual permit remains closed" {
		t.Fatalf("weighted admission resonance graft admission permit lost contract: %+v", permit)
	}

	notReadyPath := filepath.Join(dir, "not_ready_readiness.json")
	writeWeightedAdmissionResonanceGraftAdmissionReadinessFixture(t, notReadyPath)
	writeWeightedReadinessFixture(t, notReadyPath, stringsReplaceFirst(readText(t, notReadyPath), `"weighted_admission_resonance_graft_admission_readiness_ready": true`, `"weighted_admission_resonance_graft_admission_readiness_ready": false`))
	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionPermit([]string{notReadyPath, filepath.Join(dir, "not_ready_permit.json")}),
		"weighted admission resonance graft admission readiness weighted_admission_resonance_graft_admission_readiness_ready not ready",
	)

	openedReadinessPath := filepath.Join(dir, "opened_readiness.json")
	writeWeightedAdmissionResonanceGraftAdmissionReadinessFixture(t, openedReadinessPath)
	writeWeightedReadinessFixture(t, openedReadinessPath, stringsReplaceFirst(readText(t, openedReadinessPath), `"admission_readiness_ready": false`, `"admission_readiness_ready": true`))
	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionPermit([]string{openedReadinessPath, filepath.Join(dir, "opened_permit.json")}),
		"weighted admission resonance graft admission readiness opened admission_readiness_ready",
	)

	badSchemaPath := filepath.Join(dir, "bad_schema.json")
	writeWeightedAdmissionResonanceGraftAdmissionReadinessFixture(t, badSchemaPath)
	writeWeightedReadinessFixture(t, badSchemaPath, stringsReplaceFirst(readText(t, badSchemaPath), `"schema": "arianna.live_route_weighted_admission_resonance_graft_admission_readiness.v1"`, `"schema": "arianna.live_route_weighted_admission_resonance_graft_admission_readiness.v0"`))
	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionPermit([]string{badSchemaPath, filepath.Join(dir, "bad_schema_permit.json")}),
		`weighted admission resonance graft admission readiness schema mismatch: got "arianna.live_route_weighted_admission_resonance_graft_admission_readiness.v0" want "`+admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionReadinessSchema+`"`,
	)

	badHashPath := filepath.Join(dir, "bad_hash.json")
	writeWeightedAdmissionResonanceGraftAdmissionReadinessFixture(t, badHashPath)
	writeWeightedReadinessFixture(t, badHashPath, stringsReplaceFirst(readText(t, badHashPath), `"admission_readiness_hash": "weighted-resonance-graft-admission-readiness-`, `"admission_readiness_hash": "weighted-resonance-graft-admission-readiness-bad`))
	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionPermit([]string{badHashPath, filepath.Join(dir, "bad_hash_permit.json")}),
		"weighted admission resonance graft admission readiness admission_readiness_hash mismatch",
	)

	if err := runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionPermit([]string{readinessPath, filepath.Join(dir, "missing", "permit.json")}); err == nil ||
		!strings.HasPrefix(err.Error(), "weighted admission resonance graft admission permit write failed:") {
		t.Fatalf("expected weighted admission resonance graft admission permit write failure, got %v", err)
	}
}
