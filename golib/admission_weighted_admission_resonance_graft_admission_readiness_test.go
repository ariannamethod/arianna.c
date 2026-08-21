package main

import (
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionReadiness(t *testing.T) {
	dir := t.TempDir()

	usage := "usage: --admission-live-route-weighted-admission-resonance-graft-admission-readiness RESONANCE_GRAFT_ADMISSION_LEDGER_VERIFICATION_REPORT RESONANCE_GRAFT_ADMISSION_READINESS_REPORT"
	requireBoundaryAssertError(t, runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionReadiness(nil), usage)
	requireBoundaryAssertError(t, runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionReadiness([]string{"ledger_verify.json"}), usage)
	requireBoundaryAssertError(t, runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionReadiness([]string{"ledger_verify.json", "readiness.json", "extra"}), usage)
	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionReadiness([]string{"  ", filepath.Join(dir, "readiness.json")}),
		"weighted admission resonance graft admission ledger verification path missing",
	)

	verifyPath := filepath.Join(dir, "ledger_verify.json")
	writeWeightedAdmissionResonanceGraftAdmissionLedgerVerificationFixture(t, verifyPath)
	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionReadiness([]string{verifyPath, "  "}),
		"weighted admission resonance graft admission readiness output path missing",
	)

	readinessPath := filepath.Join(dir, "readiness.json")
	if err := runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionReadiness([]string{verifyPath, readinessPath}); err != nil {
		t.Fatalf("valid weighted admission resonance graft admission readiness rejected: %v", err)
	}
	raw, err := os.ReadFile(readinessPath)
	if err != nil {
		t.Fatalf("read weighted admission resonance graft admission readiness: %v", err)
	}
	var readiness admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionReadinessReport
	if err := json.Unmarshal(raw, &readiness); err != nil {
		t.Fatalf("decode weighted admission resonance graft admission readiness: %v", err)
	}
	sourceRaw, err := os.ReadFile(verifyPath)
	if err != nil {
		t.Fatalf("read weighted admission resonance graft admission ledger verification: %v", err)
	}
	var sourceVerify admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionLedgerVerificationReport
	if err := json.Unmarshal(sourceRaw, &sourceVerify); err != nil {
		t.Fatalf("decode weighted admission resonance graft admission ledger verification: %v", err)
	}
	if readiness.Schema != admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionReadinessSchema ||
		readiness.Status != "shadow_graft_admission_readiness_blocked_dry_run" ||
		readiness.TargetKind != "weighted_internal_world_shadow_graft_admission_readiness" ||
		readiness.TargetMode != "closed_readiness_guard_dry_run" ||
		readiness.Action != "block_weighted_resonance_shadow_graft_admission_ledger_verification_blocked_dry_run" ||
		readiness.WriterAction != "reject_blocked_ledger_verification" ||
		readiness.RollbackAction != "reject_blocked_ledger_verification" ||
		readiness.AdmissionReadinessState != "blocked" ||
		readiness.AdmissionReadinessAction != "reject_blocked_ledger_verification" ||
		readiness.AdmissionReadinessTarget != "live_admission" ||
		readiness.AdmissionReadinessTargetKind != "weighted_internal_world_shadow_graft_admission_ledger_verification" ||
		readiness.AdmissionReadinessTargetMode != "closed_readiness_guard_dry_run" ||
		!readiness.AdmissionReadinessDryRunOnly ||
		readiness.AdmissionReadinessLedgerVerified ||
		readiness.AdmissionReadinessWriterReady ||
		readiness.AdmissionReadinessRollbackReady ||
		readiness.AdmissionReadinessLedgerReady ||
		readiness.AdmissionReadinessReady ||
		!readiness.WeightedAdmissionResonanceGraftAdmissionReadinessReady ||
		!readiness.WeightedAdmissionResonanceGraftAdmissionLedgerVerificationConsumed ||
		!readiness.WeightedAdmissionResonanceGraftAdmissionLedgerVerificationRequired ||
		!readiness.NextStepBlockedWithoutResonanceGraftAdmissionReadiness ||
		readiness.SourceSchema != admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionLedgerVerificationSchema ||
		readiness.SourceStatus != "shadow_graft_admission_ledger_verification_blocked_dry_run" ||
		readiness.SourceReport != verifyPath ||
		readiness.SourceLedgerPersistenceSchema != admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionLedgerPersistenceSchema ||
		readiness.SourceWeightedAdmissionResonanceGraftAdmissionLedgerVerificationID != sourceVerify.WeightedAdmissionResonanceGraftAdmissionLedgerVerificationID ||
		readiness.SourceWeightedAdmissionResonanceGraftAdmissionLedgerVerificationHash != sourceVerify.LedgerVerificationHash ||
		readiness.SourceWeightedAdmissionResonanceGraftAdmissionLedgerVerificationReadBack != sourceVerify.LedgerVerificationReadBackHash ||
		readiness.SourceLedgerVerificationReportReceiptShape != sourceVerify.ReceiptShape ||
		readiness.SourceLedgerVerificationAction != sourceVerify.LedgerVerificationAction ||
		readiness.SourceLedgerVerificationAppendOnly ||
		!readiness.SourceLedgerVerificationDryRunOnly ||
		readiness.SourceLedgerVerificationReceiptReadBack ||
		readiness.SourceLedgerVerificationReceiptVerified ||
		readiness.SourceLedgerVerificationReady ||
		readiness.CausalID != admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionReadinessCausalID(readiness) ||
		readiness.AdmissionReadinessHash != admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionReadinessHash(readiness) ||
		readiness.AdmissionReadinessReadBackHash != admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionReadinessReadBackHash(readiness) ||
		readiness.AdmissionReadinessHash == readiness.AdmissionReadinessReadBackHash ||
		readiness.WeightedAdmissionResonanceGraftAdmissionReadinessID != admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionReadinessID(readiness) ||
		readiness.LedgerAppendAllowed ||
		readiness.WriteAllowed ||
		readiness.AdmissionAllowed ||
		readiness.LiveAdmissionEnabled ||
		readiness.MutatesState ||
		readiness.BodyMutationAllowed ||
		readiness.BodyTarget != "none" ||
		!readiness.Passed ||
		readiness.Reason != "weighted resonance shadow graft admission readiness blocked by blocked ledger verification; live admission readiness remains closed" {
		t.Fatalf("weighted admission resonance graft admission readiness lost contract: %+v", readiness)
	}

	notReadyPath := filepath.Join(dir, "not_ready_verify.json")
	writeWeightedAdmissionResonanceGraftAdmissionLedgerVerificationFixture(t, notReadyPath)
	writeWeightedReadinessFixture(t, notReadyPath, stringsReplaceFirst(readText(t, notReadyPath), `"weighted_admission_resonance_graft_admission_ledger_verification_ready": true`, `"weighted_admission_resonance_graft_admission_ledger_verification_ready": false`))
	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionReadiness([]string{notReadyPath, filepath.Join(dir, "not_ready_readiness.json")}),
		"weighted admission resonance graft admission ledger verification weighted_admission_resonance_graft_admission_ledger_verification_ready not ready",
	)

	openedVerifiedPath := filepath.Join(dir, "opened_verified.json")
	writeWeightedAdmissionResonanceGraftAdmissionLedgerVerificationFixture(t, openedVerifiedPath)
	writeWeightedReadinessFixture(t, openedVerifiedPath, stringsReplaceFirst(readText(t, openedVerifiedPath), `"ledger_verification_receipt_verified": false`, `"ledger_verification_receipt_verified": true`))
	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionReadiness([]string{openedVerifiedPath, filepath.Join(dir, "opened_readiness.json")}),
		"weighted admission resonance graft admission ledger verification opened ledger_verification_receipt_verified",
	)

	badSchemaPath := filepath.Join(dir, "bad_schema.json")
	writeWeightedAdmissionResonanceGraftAdmissionLedgerVerificationFixture(t, badSchemaPath)
	writeWeightedReadinessFixture(t, badSchemaPath, stringsReplaceFirst(readText(t, badSchemaPath), `"schema": "arianna.live_route_weighted_admission_resonance_graft_admission_ledger_verification.v1"`, `"schema": "arianna.live_route_weighted_admission_resonance_graft_admission_ledger_verification.v0"`))
	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionReadiness([]string{badSchemaPath, filepath.Join(dir, "bad_schema_readiness.json")}),
		`weighted admission resonance graft admission ledger verification schema mismatch: got "arianna.live_route_weighted_admission_resonance_graft_admission_ledger_verification.v0" want "`+admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionLedgerVerificationSchema+`"`,
	)

	badHashPath := filepath.Join(dir, "bad_hash.json")
	writeWeightedAdmissionResonanceGraftAdmissionLedgerVerificationFixture(t, badHashPath)
	writeWeightedReadinessFixture(t, badHashPath, stringsReplaceFirst(readText(t, badHashPath), `"ledger_verification_hash": "weighted-resonance-graft-admission-ledger-verification-`, `"ledger_verification_hash": "weighted-resonance-graft-admission-ledger-verification-bad`))
	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionReadiness([]string{badHashPath, filepath.Join(dir, "bad_hash_readiness.json")}),
		"weighted admission resonance graft admission ledger verification ledger_verification_hash mismatch",
	)

	if err := runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionReadiness([]string{verifyPath, filepath.Join(dir, "missing", "readiness.json")}); err == nil ||
		!strings.HasPrefix(err.Error(), "weighted admission resonance graft admission readiness write failed:") {
		t.Fatalf("expected weighted admission resonance graft admission readiness write failure, got %v", err)
	}
}
