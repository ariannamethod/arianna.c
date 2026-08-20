package main

import (
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionLedgerVerification(t *testing.T) {
	dir := t.TempDir()

	usage := "usage: --admission-live-route-weighted-admission-resonance-graft-admission-ledger-verification RESONANCE_GRAFT_ADMISSION_LEDGER_PERSISTENCE_REPORT RESONANCE_GRAFT_ADMISSION_LEDGER_VERIFICATION_REPORT"
	requireBoundaryAssertError(t, runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionLedgerVerification(nil), usage)
	requireBoundaryAssertError(t, runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionLedgerVerification([]string{"ledger_persist.json"}), usage)
	requireBoundaryAssertError(t, runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionLedgerVerification([]string{"ledger_persist.json", "ledger_verify.json", "extra"}), usage)
	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionLedgerVerification([]string{"  ", filepath.Join(dir, "ledger_verify.json")}),
		"weighted admission resonance graft admission ledger persistence path missing",
	)

	persistPath := filepath.Join(dir, "ledger_persist.json")
	writeWeightedAdmissionResonanceGraftAdmissionLedgerPersistenceFixture(t, persistPath)
	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionLedgerVerification([]string{persistPath, "  "}),
		"weighted admission resonance graft admission ledger verification output path missing",
	)

	verifyPath := filepath.Join(dir, "ledger_verify.json")
	if err := runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionLedgerVerification([]string{persistPath, verifyPath}); err != nil {
		t.Fatalf("valid weighted admission resonance graft admission ledger verification rejected: %v", err)
	}
	raw, err := os.ReadFile(verifyPath)
	if err != nil {
		t.Fatalf("read weighted admission resonance graft admission ledger verification: %v", err)
	}
	var verify admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionLedgerVerificationReport
	if err := json.Unmarshal(raw, &verify); err != nil {
		t.Fatalf("decode weighted admission resonance graft admission ledger verification: %v", err)
	}
	sourceRaw, err := os.ReadFile(persistPath)
	if err != nil {
		t.Fatalf("read weighted admission resonance graft admission ledger persistence: %v", err)
	}
	var sourcePersist admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionLedgerPersistenceReport
	if err := json.Unmarshal(sourceRaw, &sourcePersist); err != nil {
		t.Fatalf("decode weighted admission resonance graft admission ledger persistence: %v", err)
	}
	if verify.Schema != admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionLedgerVerificationSchema ||
		verify.Status != "shadow_graft_admission_ledger_verification_blocked_dry_run" ||
		verify.TargetKind != "weighted_internal_world_shadow_graft_admission_ledger_verification" ||
		verify.TargetMode != "closed_ledger_verification_guard_dry_run" ||
		verify.Action != "block_weighted_resonance_shadow_graft_admission_ledger_persistence_blocked_dry_run" ||
		verify.WriterAction != "reject_blocked_ledger_persistence" ||
		verify.RollbackAction != "reject_blocked_ledger_persistence" ||
		verify.LedgerVerificationState != "blocked" ||
		verify.LedgerVerificationAction != "reject_blocked_ledger_persistence" ||
		verify.LedgerVerificationTarget != "admission_ledger_receipt" ||
		verify.LedgerVerificationTargetKind != "weighted_internal_world_shadow_graft_admission_ledger_persistence" ||
		verify.LedgerVerificationTargetMode != "closed_read_back_guard_dry_run" ||
		verify.LedgerVerificationReceiptShape != "none" ||
		verify.LedgerVerificationAppendOnly ||
		!verify.LedgerVerificationDryRunOnly ||
		verify.LedgerVerificationReceiptReadBack ||
		verify.LedgerVerificationReceiptVerified ||
		verify.LedgerVerificationReady ||
		!verify.WeightedAdmissionResonanceGraftAdmissionLedgerVerificationReady ||
		!verify.WeightedAdmissionResonanceGraftAdmissionLedgerPersistenceConsumed ||
		!verify.WeightedAdmissionResonanceGraftAdmissionLedgerPersistenceRequired ||
		!verify.NextStepBlockedWithoutResonanceGraftAdmissionLedgerVerification ||
		verify.SourceSchema != admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionLedgerPersistenceSchema ||
		verify.SourceStatus != "shadow_graft_admission_ledger_persistence_blocked_dry_run" ||
		verify.SourceReport != persistPath ||
		verify.SourceLedgerImplementationSchema != admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionLedgerImplementationSchema ||
		verify.SourceWeightedAdmissionResonanceGraftAdmissionLedgerPersistenceID != sourcePersist.WeightedAdmissionResonanceGraftAdmissionLedgerPersistenceID ||
		verify.SourceWeightedAdmissionResonanceGraftAdmissionLedgerPersistenceHash != sourcePersist.LedgerPersistenceHash ||
		verify.SourceWeightedAdmissionResonanceGraftAdmissionLedgerPersistenceReadBack != sourcePersist.LedgerPersistenceReadBackHash ||
		verify.SourceLedgerPersistenceReportReceiptShape != sourcePersist.ReceiptShape ||
		verify.SourceLedgerPersistenceAction != sourcePersist.LedgerPersistenceAction ||
		verify.SourceLedgerPersistenceAppendOnly ||
		!verify.SourceLedgerPersistenceDryRunOnly ||
		verify.SourceLedgerPersistenceReceiptPersisted ||
		verify.SourceLedgerPersistenceReady ||
		verify.CausalID != admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionLedgerVerificationCausalID(verify) ||
		verify.LedgerVerificationHash != admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionLedgerVerificationHash(verify) ||
		verify.LedgerVerificationReadBackHash != admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionLedgerVerificationReadBackHash(verify) ||
		verify.LedgerVerificationHash == verify.LedgerVerificationReadBackHash ||
		verify.WeightedAdmissionResonanceGraftAdmissionLedgerVerificationID != admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionLedgerVerificationID(verify) ||
		verify.LedgerAppendAllowed ||
		verify.WriteAllowed ||
		verify.AdmissionAllowed ||
		verify.LiveAdmissionEnabled ||
		verify.MutatesState ||
		verify.BodyMutationAllowed ||
		verify.BodyTarget != "none" ||
		!verify.Passed ||
		verify.Reason != "weighted resonance shadow graft admission ledger verification blocked by blocked ledger persistence; receipt read-back remains closed" {
		t.Fatalf("weighted admission resonance graft admission ledger verification lost contract: %+v", verify)
	}

	openedPersistPath := filepath.Join(dir, "open_persist.json")
	writeWeightedAdmissionResonanceGraftAdmissionLedgerPersistenceFixture(t, openedPersistPath)
	writeWeightedReadinessFixture(t, openedPersistPath, stringsReplaceFirst(readText(t, openedPersistPath), `"ledger_persistence_ready": false`, `"ledger_persistence_ready": true`))
	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionLedgerVerification([]string{openedPersistPath, filepath.Join(dir, "opened_ledger_verify.json")}),
		"weighted admission resonance graft admission ledger persistence opened ledger_persistence_ready",
	)

	badSchemaPath := filepath.Join(dir, "bad_schema.json")
	writeWeightedAdmissionResonanceGraftAdmissionLedgerPersistenceFixture(t, badSchemaPath)
	writeWeightedReadinessFixture(t, badSchemaPath, stringsReplaceFirst(readText(t, badSchemaPath), `"schema": "arianna.live_route_weighted_admission_resonance_graft_admission_ledger_persistence.v1"`, `"schema": "arianna.live_route_weighted_admission_resonance_graft_admission_ledger_persistence.v0"`))
	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionLedgerVerification([]string{badSchemaPath, filepath.Join(dir, "bad_schema_ledger_verify.json")}),
		`weighted admission resonance graft admission ledger persistence schema mismatch: got "arianna.live_route_weighted_admission_resonance_graft_admission_ledger_persistence.v0" want "`+admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionLedgerPersistenceSchema+`"`,
	)

	badHashPath := filepath.Join(dir, "bad_hash.json")
	writeWeightedAdmissionResonanceGraftAdmissionLedgerPersistenceFixture(t, badHashPath)
	writeWeightedReadinessFixture(t, badHashPath, stringsReplaceFirst(readText(t, badHashPath), `"ledger_persistence_hash": "weighted-resonance-graft-admission-ledger-persistence-`, `"ledger_persistence_hash": "weighted-resonance-graft-admission-ledger-persistence-bad`))
	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionLedgerVerification([]string{badHashPath, filepath.Join(dir, "bad_hash_ledger_verify.json")}),
		"weighted admission resonance graft admission ledger persistence ledger_persistence_hash mismatch",
	)

	if err := runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionLedgerVerification([]string{persistPath, filepath.Join(dir, "missing", "ledger_verify.json")}); err == nil ||
		!strings.HasPrefix(err.Error(), "weighted admission resonance graft admission ledger verification write failed:") {
		t.Fatalf("expected weighted admission resonance graft admission ledger verification write failure, got %v", err)
	}
}
