package main

import (
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionLedgerImplementation(t *testing.T) {
	dir := t.TempDir()

	usage := "usage: --admission-live-route-weighted-admission-resonance-graft-admission-ledger-implementation RESONANCE_GRAFT_ADMISSION_LEDGER_REPORT RESONANCE_GRAFT_ADMISSION_LEDGER_IMPLEMENTATION_REPORT"
	requireBoundaryAssertError(t, runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionLedgerImplementation(nil), usage)
	requireBoundaryAssertError(t, runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionLedgerImplementation([]string{"ledger.json"}), usage)
	requireBoundaryAssertError(t, runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionLedgerImplementation([]string{"ledger.json", "ledger_impl.json", "extra"}), usage)
	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionLedgerImplementation([]string{"  ", filepath.Join(dir, "ledger_impl.json")}),
		"weighted admission resonance graft admission ledger path missing",
	)

	ledgerPath := filepath.Join(dir, "ledger.json")
	writeWeightedAdmissionResonanceGraftAdmissionLedgerFixture(t, ledgerPath)
	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionLedgerImplementation([]string{ledgerPath, "  "}),
		"weighted admission resonance graft admission ledger implementation output path missing",
	)

	implPath := filepath.Join(dir, "ledger_impl.json")
	if err := runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionLedgerImplementation([]string{ledgerPath, implPath}); err != nil {
		t.Fatalf("valid weighted admission resonance graft admission ledger implementation rejected: %v", err)
	}
	raw, err := os.ReadFile(implPath)
	if err != nil {
		t.Fatalf("read weighted admission resonance graft admission ledger implementation: %v", err)
	}
	var impl admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionLedgerImplementationReport
	if err := json.Unmarshal(raw, &impl); err != nil {
		t.Fatalf("decode weighted admission resonance graft admission ledger implementation: %v", err)
	}
	sourceRaw, err := os.ReadFile(ledgerPath)
	if err != nil {
		t.Fatalf("read weighted admission resonance graft admission ledger: %v", err)
	}
	var sourceLedger admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionLedgerReport
	if err := json.Unmarshal(sourceRaw, &sourceLedger); err != nil {
		t.Fatalf("decode weighted admission resonance graft admission ledger: %v", err)
	}
	if impl.Schema != admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionLedgerImplementationSchema ||
		impl.Status != "shadow_graft_admission_ledger_implementation_blocked_dry_run" ||
		impl.TargetKind != "weighted_internal_world_shadow_graft_admission_ledger_implementation" ||
		impl.Action != "block_weighted_resonance_shadow_graft_admission_ledger_blocked_dry_run" ||
		impl.WriterAction != "reject_blocked_admission_ledger" ||
		impl.RollbackAction != "reject_blocked_admission_ledger" ||
		impl.LedgerImplementationState != "blocked" ||
		impl.LedgerImplementationAction != "reject_blocked_admission_ledger" ||
		impl.LedgerImplementationTarget != "admission_ledger" ||
		impl.LedgerImplementationEntrypoint != "none" ||
		impl.LedgerImplementationReceiptShape != "none" ||
		impl.LedgerImplementationWriteScope != "none" ||
		impl.LedgerImplementationAppendOnly ||
		!impl.LedgerImplementationDryRunOnly ||
		impl.LedgerImplementationReceiptPersisted ||
		impl.LedgerImplementationReady ||
		!impl.WeightedAdmissionResonanceGraftAdmissionLedgerImplementationReady ||
		!impl.WeightedAdmissionResonanceGraftAdmissionLedgerConsumed ||
		!impl.WeightedAdmissionResonanceGraftAdmissionLedgerRequired ||
		!impl.NextStepBlockedWithoutResonanceGraftAdmissionLedgerImplementation ||
		impl.SourceSchema != admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionLedgerSchema ||
		impl.SourceStatus != "shadow_graft_admission_ledger_blocked_dry_run" ||
		impl.SourceReport != ledgerPath ||
		impl.SourceWeightedAdmissionResonanceGraftAdmissionLedgerID != sourceLedger.WeightedAdmissionResonanceGraftAdmissionLedgerID ||
		impl.SourceWeightedAdmissionResonanceGraftAdmissionLedgerHash != sourceLedger.AdmissionLedgerHash ||
		impl.SourceWeightedAdmissionResonanceGraftAdmissionLedgerReadBack != sourceLedger.ReadBackHash ||
		impl.SourceAdmissionLedgerReceiptShape != sourceLedger.ReceiptShape ||
		impl.SourceAdmissionLedgerLedgerAction != sourceLedger.LedgerAction ||
		impl.SourceAdmissionLedgerLedgerAppendAllowed ||
		impl.SourceWriterContractSchema != admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionWriterContractSchema ||
		impl.SourceWeightedAdmissionResonanceGraftAdmissionWriterContractID != sourceLedger.SourceWeightedAdmissionResonanceGraftAdmissionWriterContractID ||
		impl.CausalID != admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionLedgerImplementationCausalID(impl) ||
		impl.LedgerImplementationHash != admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionLedgerImplementationHash(impl) ||
		impl.LedgerImplementationReadBackHash != admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionLedgerImplementationReadBackHash(impl) ||
		impl.LedgerImplementationHash == impl.LedgerImplementationReadBackHash ||
		impl.WeightedAdmissionResonanceGraftAdmissionLedgerImplementationID != admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionLedgerImplementationID(impl) ||
		impl.LedgerAppendAllowed ||
		impl.WriteAllowed ||
		impl.AdmissionAllowed ||
		impl.LiveAdmissionEnabled ||
		impl.MutatesState ||
		impl.BodyMutationAllowed ||
		impl.BodyTarget != "none" ||
		!impl.Passed ||
		impl.Reason != "weighted resonance shadow graft admission ledger implementation blocked by blocked admission ledger; implementation append contract remains closed" {
		t.Fatalf("weighted admission resonance graft admission ledger implementation lost contract: %+v", impl)
	}

	openedLedgerPath := filepath.Join(dir, "open_ledger.json")
	writeWeightedAdmissionResonanceGraftAdmissionLedgerFixture(t, openedLedgerPath)
	writeWeightedReadinessFixture(t, openedLedgerPath, stringsReplaceFirst(readText(t, openedLedgerPath), `"ledger_append_allowed": false`, `"ledger_append_allowed": true`))
	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionLedgerImplementation([]string{openedLedgerPath, filepath.Join(dir, "opened_ledger_impl.json")}),
		"weighted admission resonance graft admission ledger opened ledger_append_allowed",
	)

	badSchemaPath := filepath.Join(dir, "bad_schema.json")
	writeWeightedAdmissionResonanceGraftAdmissionLedgerFixture(t, badSchemaPath)
	writeWeightedReadinessFixture(t, badSchemaPath, stringsReplaceFirst(readText(t, badSchemaPath), `"schema": "arianna.live_route_weighted_admission_resonance_graft_admission_ledger.v1"`, `"schema": "arianna.live_route_weighted_admission_resonance_graft_admission_ledger.v0"`))
	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionLedgerImplementation([]string{badSchemaPath, filepath.Join(dir, "bad_schema_ledger_impl.json")}),
		`weighted admission resonance graft admission ledger schema mismatch: got "arianna.live_route_weighted_admission_resonance_graft_admission_ledger.v0" want "`+admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionLedgerSchema+`"`,
	)

	badHashPath := filepath.Join(dir, "bad_hash.json")
	writeWeightedAdmissionResonanceGraftAdmissionLedgerFixture(t, badHashPath)
	writeWeightedReadinessFixture(t, badHashPath, stringsReplaceFirst(readText(t, badHashPath), `"admission_ledger_hash": "weighted-resonance-graft-admission-ledger-`, `"admission_ledger_hash": "weighted-resonance-graft-admission-ledger-bad`))
	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionLedgerImplementation([]string{badHashPath, filepath.Join(dir, "bad_hash_ledger_impl.json")}),
		"weighted admission resonance graft admission ledger admission_ledger_hash mismatch",
	)

	if err := runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionLedgerImplementation([]string{ledgerPath, filepath.Join(dir, "missing", "ledger_impl.json")}); err == nil ||
		!strings.HasPrefix(err.Error(), "weighted admission resonance graft admission ledger implementation write failed:") {
		t.Fatalf("expected weighted admission resonance graft admission ledger implementation write failure, got %v", err)
	}
}
