package main

import (
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionLedgerPersistence(t *testing.T) {
	dir := t.TempDir()

	usage := "usage: --admission-live-route-weighted-admission-resonance-graft-admission-ledger-persistence RESONANCE_GRAFT_ADMISSION_LEDGER_IMPLEMENTATION_REPORT RESONANCE_GRAFT_ADMISSION_LEDGER_PERSISTENCE_REPORT"
	requireBoundaryAssertError(t, runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionLedgerPersistence(nil), usage)
	requireBoundaryAssertError(t, runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionLedgerPersistence([]string{"ledger_impl.json"}), usage)
	requireBoundaryAssertError(t, runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionLedgerPersistence([]string{"ledger_impl.json", "ledger_persist.json", "extra"}), usage)
	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionLedgerPersistence([]string{"  ", filepath.Join(dir, "ledger_persist.json")}),
		"weighted admission resonance graft admission ledger implementation path missing",
	)

	implPath := filepath.Join(dir, "ledger_impl.json")
	writeWeightedAdmissionResonanceGraftAdmissionLedgerImplementationFixture(t, implPath)
	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionLedgerPersistence([]string{implPath, "  "}),
		"weighted admission resonance graft admission ledger persistence output path missing",
	)

	persistPath := filepath.Join(dir, "ledger_persist.json")
	if err := runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionLedgerPersistence([]string{implPath, persistPath}); err != nil {
		t.Fatalf("valid weighted admission resonance graft admission ledger persistence rejected: %v", err)
	}
	raw, err := os.ReadFile(persistPath)
	if err != nil {
		t.Fatalf("read weighted admission resonance graft admission ledger persistence: %v", err)
	}
	var persist admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionLedgerPersistenceReport
	if err := json.Unmarshal(raw, &persist); err != nil {
		t.Fatalf("decode weighted admission resonance graft admission ledger persistence: %v", err)
	}
	sourceRaw, err := os.ReadFile(implPath)
	if err != nil {
		t.Fatalf("read weighted admission resonance graft admission ledger implementation: %v", err)
	}
	var sourceImpl admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionLedgerImplementationReport
	if err := json.Unmarshal(sourceRaw, &sourceImpl); err != nil {
		t.Fatalf("decode weighted admission resonance graft admission ledger implementation: %v", err)
	}
	if persist.Schema != admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionLedgerPersistenceSchema ||
		persist.Status != "shadow_graft_admission_ledger_persistence_blocked_dry_run" ||
		persist.TargetKind != "weighted_internal_world_shadow_graft_admission_ledger_persistence" ||
		persist.TargetMode != "closed_ledger_persistence_guard_dry_run" ||
		persist.Action != "block_weighted_resonance_shadow_graft_admission_ledger_implementation_blocked_dry_run" ||
		persist.WriterAction != "reject_blocked_ledger_implementation" ||
		persist.RollbackAction != "reject_blocked_ledger_implementation" ||
		persist.LedgerPersistenceState != "blocked" ||
		persist.LedgerPersistenceAction != "reject_blocked_ledger_implementation" ||
		persist.LedgerPersistenceTarget != "admission_ledger_receipt" ||
		persist.LedgerPersistenceTargetKind != "weighted_internal_world_shadow_graft_admission_ledger_implementation" ||
		persist.LedgerPersistenceTargetMode != "closed_persistence_guard_dry_run" ||
		persist.LedgerPersistenceReceiptShape != "none" ||
		persist.LedgerPersistenceWriteScope != "none" ||
		persist.LedgerPersistenceAppendOnly ||
		!persist.LedgerPersistenceDryRunOnly ||
		persist.LedgerPersistenceReceiptPersisted ||
		persist.LedgerPersistenceReady ||
		!persist.WeightedAdmissionResonanceGraftAdmissionLedgerPersistenceReady ||
		!persist.WeightedAdmissionResonanceGraftAdmissionLedgerImplementationConsumed ||
		!persist.WeightedAdmissionResonanceGraftAdmissionLedgerImplementationRequired ||
		!persist.NextStepBlockedWithoutResonanceGraftAdmissionLedgerPersistence ||
		persist.SourceSchema != admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionLedgerImplementationSchema ||
		persist.SourceStatus != "shadow_graft_admission_ledger_implementation_blocked_dry_run" ||
		persist.SourceReport != implPath ||
		persist.SourceAdmissionLedgerSchema != admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionLedgerSchema ||
		persist.SourceWeightedAdmissionResonanceGraftAdmissionLedgerImplementationID != sourceImpl.WeightedAdmissionResonanceGraftAdmissionLedgerImplementationID ||
		persist.SourceWeightedAdmissionResonanceGraftAdmissionLedgerImplementationHash != sourceImpl.LedgerImplementationHash ||
		persist.SourceWeightedAdmissionResonanceGraftAdmissionLedgerImplementationReadBack != sourceImpl.LedgerImplementationReadBackHash ||
		persist.SourceLedgerImplementationReportReceiptShape != sourceImpl.ReceiptShape ||
		persist.SourceLedgerImplementationAction != sourceImpl.LedgerImplementationAction ||
		persist.SourceLedgerImplementationAppendOnly ||
		!persist.SourceLedgerImplementationDryRunOnly ||
		persist.SourceLedgerImplementationReceiptPersisted ||
		persist.SourceLedgerImplementationReady ||
		persist.CausalID != admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionLedgerPersistenceCausalID(persist) ||
		persist.LedgerPersistenceHash != admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionLedgerPersistenceHash(persist) ||
		persist.LedgerPersistenceReadBackHash != admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionLedgerPersistenceReadBackHash(persist) ||
		persist.LedgerPersistenceHash == persist.LedgerPersistenceReadBackHash ||
		persist.WeightedAdmissionResonanceGraftAdmissionLedgerPersistenceID != admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionLedgerPersistenceID(persist) ||
		persist.LedgerAppendAllowed ||
		persist.WriteAllowed ||
		persist.AdmissionAllowed ||
		persist.LiveAdmissionEnabled ||
		persist.MutatesState ||
		persist.BodyMutationAllowed ||
		persist.BodyTarget != "none" ||
		!persist.Passed ||
		persist.Reason != "weighted resonance shadow graft admission ledger persistence blocked by blocked ledger implementation; ledger receipt persistence remains closed" {
		t.Fatalf("weighted admission resonance graft admission ledger persistence lost contract: %+v", persist)
	}

	openedImplPath := filepath.Join(dir, "open_impl.json")
	writeWeightedAdmissionResonanceGraftAdmissionLedgerImplementationFixture(t, openedImplPath)
	writeWeightedReadinessFixture(t, openedImplPath, stringsReplaceFirst(readText(t, openedImplPath), `"ledger_implementation_ready": false`, `"ledger_implementation_ready": true`))
	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionLedgerPersistence([]string{openedImplPath, filepath.Join(dir, "opened_ledger_persist.json")}),
		"weighted admission resonance graft admission ledger implementation opened ledger_implementation_ready",
	)

	badSchemaPath := filepath.Join(dir, "bad_schema.json")
	writeWeightedAdmissionResonanceGraftAdmissionLedgerImplementationFixture(t, badSchemaPath)
	writeWeightedReadinessFixture(t, badSchemaPath, stringsReplaceFirst(readText(t, badSchemaPath), `"schema": "arianna.live_route_weighted_admission_resonance_graft_admission_ledger_implementation.v1"`, `"schema": "arianna.live_route_weighted_admission_resonance_graft_admission_ledger_implementation.v0"`))
	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionLedgerPersistence([]string{badSchemaPath, filepath.Join(dir, "bad_schema_ledger_persist.json")}),
		`weighted admission resonance graft admission ledger implementation schema mismatch: got "arianna.live_route_weighted_admission_resonance_graft_admission_ledger_implementation.v0" want "`+admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionLedgerImplementationSchema+`"`,
	)

	badHashPath := filepath.Join(dir, "bad_hash.json")
	writeWeightedAdmissionResonanceGraftAdmissionLedgerImplementationFixture(t, badHashPath)
	writeWeightedReadinessFixture(t, badHashPath, stringsReplaceFirst(readText(t, badHashPath), `"ledger_implementation_hash": "weighted-resonance-graft-admission-ledger-implementation-`, `"ledger_implementation_hash": "weighted-resonance-graft-admission-ledger-implementation-bad`))
	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionLedgerPersistence([]string{badHashPath, filepath.Join(dir, "bad_hash_ledger_persist.json")}),
		"weighted admission resonance graft admission ledger implementation ledger_implementation_hash mismatch",
	)

	if err := runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionLedgerPersistence([]string{implPath, filepath.Join(dir, "missing", "ledger_persist.json")}); err == nil ||
		!strings.HasPrefix(err.Error(), "weighted admission resonance graft admission ledger persistence write failed:") {
		t.Fatalf("expected weighted admission resonance graft admission ledger persistence write failure, got %v", err)
	}
}
