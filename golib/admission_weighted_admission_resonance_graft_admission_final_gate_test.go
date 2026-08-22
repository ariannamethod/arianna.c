package main

import (
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGate(t *testing.T) {
	dir := t.TempDir()

	usage := "usage: --admission-live-route-weighted-admission-resonance-graft-admission-final-gate RESONANCE_GRAFT_ADMISSION_SEAL_REPORT RESONANCE_GRAFT_ADMISSION_FINAL_GATE_REPORT"
	requireBoundaryAssertError(t, runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGate(nil), usage)
	requireBoundaryAssertError(t, runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGate([]string{"seal.json"}), usage)
	requireBoundaryAssertError(t, runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGate([]string{"seal.json", "final_gate.json", "extra"}), usage)
	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGate([]string{"  ", filepath.Join(dir, "final_gate.json")}),
		"weighted admission resonance graft admission seal path missing",
	)

	sealPath := filepath.Join(dir, "seal.json")
	writeWeightedAdmissionResonanceGraftAdmissionSealFixture(t, sealPath)
	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGate([]string{sealPath, "  "}),
		"weighted admission resonance graft admission final gate output path missing",
	)

	finalGatePath := filepath.Join(dir, "final_gate.json")
	if err := runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGate([]string{sealPath, finalGatePath}); err != nil {
		t.Fatalf("valid weighted admission resonance graft admission final gate rejected: %v", err)
	}
	raw, err := os.ReadFile(finalGatePath)
	if err != nil {
		t.Fatalf("read weighted admission resonance graft admission final gate: %v", err)
	}
	var finalGate admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateReport
	if err := json.Unmarshal(raw, &finalGate); err != nil {
		t.Fatalf("decode weighted admission resonance graft admission final gate: %v", err)
	}
	sourceRaw, err := os.ReadFile(sealPath)
	if err != nil {
		t.Fatalf("read weighted admission resonance graft admission seal: %v", err)
	}
	var sourceSeal admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionSealReport
	if err := json.Unmarshal(sourceRaw, &sourceSeal); err != nil {
		t.Fatalf("decode weighted admission resonance graft admission seal: %v", err)
	}
	if finalGate.Schema != admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateSchema ||
		finalGate.Status != "shadow_graft_admission_final_gate_blocked_dry_run" ||
		finalGate.Target != "live_route_admission_next_step" ||
		finalGate.TargetKind != "weighted_internal_world_shadow_graft_admission_final_gate" ||
		finalGate.TargetMode != "closed_final_gate_guard_dry_run" ||
		finalGate.Action != "block_weighted_resonance_shadow_graft_admission_seal_blocked_dry_run" ||
		finalGate.WriterAction != "reject_blocked_admission_seal" ||
		finalGate.RollbackAction != "reject_blocked_admission_seal" ||
		finalGate.LedgerState != "blocked" ||
		finalGate.LedgerAction != "reject_blocked_admission_seal" ||
		finalGate.LedgerContract != "none" ||
		finalGate.LedgerEntrypoint != "none" ||
		finalGate.LedgerReceiptShape != "none" ||
		finalGate.LedgerWriteScope != "none" ||
		finalGate.ReceiptShape != "weighted_resonance_shadow_graft_admission_final_gate_receipt" ||
		finalGate.AdmissionFinalGateState != "blocked" ||
		finalGate.AdmissionFinalGateAction != "reject_blocked_admission_seal" ||
		finalGate.AdmissionFinalGateTarget != "live_admission_final_gate" ||
		finalGate.AdmissionFinalGateTargetKind != "weighted_internal_world_shadow_graft_admission_seal" ||
		finalGate.AdmissionFinalGateTargetMode != "closed_final_gate_guard_dry_run" ||
		!finalGate.AdmissionFinalGateDryRunOnly ||
		finalGate.AdmissionFinalGateSealVerified ||
		finalGate.AdmissionFinalGateAuthorityVerified ||
		finalGate.AdmissionFinalGatePermitVerified ||
		finalGate.AdmissionFinalGateLedgerVerified ||
		finalGate.AdmissionFinalGateReady ||
		!finalGate.WeightedAdmissionResonanceGraftAdmissionFinalGateReady ||
		!finalGate.WeightedAdmissionResonanceGraftAdmissionSealConsumed ||
		!finalGate.WeightedAdmissionResonanceGraftAdmissionSealRequired ||
		!finalGate.NextStepBlockedWithoutResonanceGraftAdmissionFinalGate ||
		finalGate.SourceSchema != admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionSealSchema ||
		finalGate.SourceStatus != "shadow_graft_admission_seal_blocked_dry_run" ||
		finalGate.SourceTarget != "live_route_admission_next_step" ||
		finalGate.SourceReport != sealPath ||
		finalGate.SourceAdmissionAuthoritySchema != admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionAuthoritySchema ||
		finalGate.SourceWeightedAdmissionResonanceGraftAdmissionSealID != sourceSeal.WeightedAdmissionResonanceGraftAdmissionSealID ||
		finalGate.SourceWeightedAdmissionResonanceGraftAdmissionSealHash != sourceSeal.AdmissionSealHash ||
		finalGate.SourceWeightedAdmissionResonanceGraftAdmissionSealReadBack != sourceSeal.AdmissionSealReadBackHash ||
		finalGate.SourceAdmissionSealReportReceiptShape != sourceSeal.ReceiptShape ||
		finalGate.SourceAdmissionSealAction != sourceSeal.AdmissionSealAction ||
		!finalGate.SourceAdmissionSealDryRunOnly ||
		finalGate.SourceAdmissionSealAuthorityVerified ||
		finalGate.SourceAdmissionSealPermitVerified ||
		finalGate.SourceAdmissionSealLedgerVerified ||
		finalGate.SourceAdmissionSealReady ||
		!finalGate.SourceAdmissionSealImmutableReceipt ||
		finalGate.CausalID != admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateCausalID(finalGate) ||
		finalGate.AdmissionFinalGateHash != admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateHash(finalGate) ||
		finalGate.AdmissionFinalGateReadBackHash != admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateReadBackHash(finalGate) ||
		finalGate.AdmissionFinalGateHash == finalGate.AdmissionFinalGateReadBackHash ||
		finalGate.WeightedAdmissionResonanceGraftAdmissionFinalGateID != admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateID(finalGate) ||
		finalGate.LedgerReady ||
		finalGate.LedgerAppendAllowed ||
		finalGate.WriteAllowed ||
		finalGate.AdmissionAllowed ||
		finalGate.LiveAdmissionEnabled ||
		finalGate.MutatesState ||
		finalGate.BodyMutationAllowed ||
		finalGate.AuthorityGranted ||
		finalGate.BodyTarget != "none" ||
		!finalGate.Passed ||
		finalGate.Reason != "weighted resonance shadow graft admission final gate blocked by blocked seal; final admission remains closed" {
		t.Fatalf("weighted admission resonance graft admission final gate lost contract: %+v", finalGate)
	}

	notReadyPath := filepath.Join(dir, "not_ready_seal.json")
	writeWeightedAdmissionResonanceGraftAdmissionSealFixture(t, notReadyPath)
	writeWeightedReadinessFixture(t, notReadyPath, stringsReplaceFirst(readText(t, notReadyPath), `"weighted_admission_resonance_graft_admission_seal_ready": true`, `"weighted_admission_resonance_graft_admission_seal_ready": false`))
	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGate([]string{notReadyPath, filepath.Join(dir, "not_ready_final_gate.json")}),
		"weighted admission resonance graft admission seal weighted_admission_resonance_graft_admission_seal_ready not ready",
	)

	openedSealPath := filepath.Join(dir, "opened_seal.json")
	writeWeightedAdmissionResonanceGraftAdmissionSealFixture(t, openedSealPath)
	writeWeightedReadinessFixture(t, openedSealPath, stringsReplaceFirst(readText(t, openedSealPath), `"admission_seal_ready": false`, `"admission_seal_ready": true`))
	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGate([]string{openedSealPath, filepath.Join(dir, "opened_final_gate.json")}),
		"weighted admission resonance graft admission seal opened admission_seal_ready",
	)

	badSchemaPath := filepath.Join(dir, "bad_schema.json")
	writeWeightedAdmissionResonanceGraftAdmissionSealFixture(t, badSchemaPath)
	writeWeightedReadinessFixture(t, badSchemaPath, stringsReplaceFirst(readText(t, badSchemaPath), `"schema": "arianna.live_route_weighted_admission_resonance_graft_admission_seal.v1"`, `"schema": "arianna.live_route_weighted_admission_resonance_graft_admission_seal.v0"`))
	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGate([]string{badSchemaPath, filepath.Join(dir, "bad_schema_final_gate.json")}),
		`weighted admission resonance graft admission seal schema mismatch: got "arianna.live_route_weighted_admission_resonance_graft_admission_seal.v0" want "`+admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionSealSchema+`"`,
	)

	badHashPath := filepath.Join(dir, "bad_hash.json")
	writeWeightedAdmissionResonanceGraftAdmissionSealFixture(t, badHashPath)
	writeWeightedReadinessFixture(t, badHashPath, stringsReplaceFirst(readText(t, badHashPath), `"admission_seal_hash": "weighted-resonance-graft-admission-seal-`, `"admission_seal_hash": "weighted-resonance-graft-admission-seal-bad`))
	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGate([]string{badHashPath, filepath.Join(dir, "bad_hash_final_gate.json")}),
		"weighted admission resonance graft admission seal admission_seal_hash mismatch",
	)

	if err := runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGate([]string{sealPath, filepath.Join(dir, "missing", "final_gate.json")}); err == nil ||
		!strings.HasPrefix(err.Error(), "weighted admission resonance graft admission final gate write failed:") {
		t.Fatalf("expected weighted admission resonance graft admission final gate write failure, got %v", err)
	}
}
