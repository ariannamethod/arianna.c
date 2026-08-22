package main

import (
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateIntent(t *testing.T) {
	dir := t.TempDir()

	usage := "usage: --admission-live-route-weighted-admission-resonance-graft-admission-final-gate-intent RESONANCE_GRAFT_ADMISSION_FINAL_GATE_REPORT RESONANCE_GRAFT_ADMISSION_FINAL_GATE_INTENT_REPORT"
	requireBoundaryAssertError(t, runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateIntent(nil), usage)
	requireBoundaryAssertError(t, runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateIntent([]string{"final_gate.json"}), usage)
	requireBoundaryAssertError(t, runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateIntent([]string{"final_gate.json", "intent.json", "extra"}), usage)
	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateIntent([]string{"  ", filepath.Join(dir, "intent.json")}),
		"weighted admission resonance graft admission final gate path missing",
	)

	finalGatePath := filepath.Join(dir, "final_gate.json")
	writeWeightedAdmissionResonanceGraftAdmissionFinalGateFixture(t, finalGatePath)
	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateIntent([]string{finalGatePath, "  "}),
		"weighted admission resonance graft admission final gate intent output path missing",
	)

	intentPath := filepath.Join(dir, "intent.json")
	if err := runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateIntent([]string{finalGatePath, intentPath}); err != nil {
		t.Fatalf("valid weighted admission resonance graft admission final gate intent rejected: %v", err)
	}
	raw, err := os.ReadFile(intentPath)
	if err != nil {
		t.Fatalf("read weighted admission resonance graft admission final gate intent: %v", err)
	}
	var intent admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateIntentReport
	if err := json.Unmarshal(raw, &intent); err != nil {
		t.Fatalf("decode weighted admission resonance graft admission final gate intent: %v", err)
	}
	sourceRaw, err := os.ReadFile(finalGatePath)
	if err != nil {
		t.Fatalf("read weighted admission resonance graft admission final gate: %v", err)
	}
	var sourceFinalGate admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateReport
	if err := json.Unmarshal(sourceRaw, &sourceFinalGate); err != nil {
		t.Fatalf("decode weighted admission resonance graft admission final gate: %v", err)
	}
	if intent.Schema != admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateIntentSchema ||
		intent.Status != "shadow_graft_admission_final_gate_intent_blocked_dry_run" ||
		intent.Target != "live_route_admission_next_step" ||
		intent.TargetKind != "weighted_internal_world_shadow_graft_admission_final_gate_intent" ||
		intent.TargetMode != "bounded_intent_guard_dry_run" ||
		intent.Action != "draft_weighted_resonance_shadow_graft_admission_final_gate_intent_dry_run" ||
		intent.WriterAction != "reject_blocked_admission_final_gate_intent" ||
		intent.RollbackAction != "reject_blocked_admission_final_gate_intent" ||
		intent.LedgerState != "blocked" ||
		intent.LedgerAction != "reject_blocked_admission_final_gate_intent" ||
		intent.LedgerContract != "none" ||
		intent.LedgerEntrypoint != "none" ||
		intent.LedgerReceiptShape != "none" ||
		intent.LedgerWriteScope != "none" ||
		intent.ReceiptShape != "weighted_resonance_shadow_graft_admission_final_gate_intent_receipt" ||
		intent.AdmissionFinalGateIntentState != "blocked" ||
		intent.AdmissionFinalGateIntentAction != "draft_blocked_final_gate_intent" ||
		intent.AdmissionFinalGateIntentTarget != "resonance" ||
		intent.AdmissionFinalGateIntentTargetKind != "weighted_internal_world_shadow_graft_admission_final_gate" ||
		intent.AdmissionFinalGateIntentTargetMode != "bounded_intent_guard_dry_run" ||
		!intent.AdmissionFinalGateIntentDryRunOnly ||
		intent.AdmissionFinalGateIntentFinalGateVerified ||
		intent.AdmissionFinalGateIntentSealVerified ||
		intent.AdmissionFinalGateIntentReady ||
		intent.FinalGateIntentReceiver != "resonance" ||
		intent.FinalGateIntentReceiverKind != "internal_world" ||
		intent.FinalGateIntentInfluenceKind != "bounded_direction" ||
		intent.FinalGateIntentMaxInfluence != admissionLiveRouteTurnCandidateAdmissionResonanceIntentMaxGain ||
		intent.FinalGateIntentTTLTurns != admissionLiveRouteTurnCandidateAdmissionResonanceIntentTTL ||
		intent.FinalGateIntentRawDreamTextAllowed ||
		intent.FinalGateIntentJanusSurfaceAllowed ||
		intent.FinalGateIntentCoocLearningAllowed ||
		intent.FinalGateIntentDeltaHarvestAllowed ||
		!intent.FinalGateIntentPreStateHashRequired ||
		!intent.FinalGateIntentPostStateHashRequired ||
		!intent.WeightedAdmissionResonanceGraftAdmissionFinalGateIntentReady ||
		!intent.WeightedAdmissionResonanceGraftAdmissionFinalGateConsumed ||
		!intent.WeightedAdmissionResonanceGraftAdmissionFinalGateRequired ||
		!intent.NextStepBlockedWithoutResonanceGraftAdmissionFinalGateIntent ||
		intent.SourceSchema != admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateSchema ||
		intent.SourceStatus != "shadow_graft_admission_final_gate_blocked_dry_run" ||
		intent.SourceTarget != "live_route_admission_next_step" ||
		intent.SourceReport != finalGatePath ||
		intent.SourceAdmissionSealSchema != admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionSealSchema ||
		intent.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateID != sourceFinalGate.WeightedAdmissionResonanceGraftAdmissionFinalGateID ||
		intent.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateHash != sourceFinalGate.AdmissionFinalGateHash ||
		intent.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateReadBack != sourceFinalGate.AdmissionFinalGateReadBackHash ||
		intent.SourceAdmissionFinalGateReportReceiptShape != sourceFinalGate.ReceiptShape ||
		intent.SourceAdmissionFinalGateAction != sourceFinalGate.AdmissionFinalGateAction ||
		!intent.SourceAdmissionFinalGateDryRunOnly ||
		intent.SourceAdmissionFinalGateSealVerified ||
		intent.SourceAdmissionFinalGateAuthorityVerified ||
		intent.SourceAdmissionFinalGatePermitVerified ||
		intent.SourceAdmissionFinalGateLedgerVerified ||
		intent.SourceAdmissionFinalGateReady ||
		intent.CausalID != admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateIntentCausalID(intent) ||
		intent.AdmissionFinalGateIntentHash != admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateIntentHash(intent) ||
		intent.AdmissionFinalGateIntentReadBackHash != admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateIntentReadBackHash(intent) ||
		intent.AdmissionFinalGateIntentHash == intent.AdmissionFinalGateIntentReadBackHash ||
		intent.WeightedAdmissionResonanceGraftAdmissionFinalGateIntentID != admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateIntentID(intent) ||
		intent.LedgerReady ||
		intent.LedgerAppendAllowed ||
		intent.WriteAllowed ||
		intent.AdmissionAllowed ||
		intent.LiveAdmissionEnabled ||
		intent.MutatesState ||
		intent.BodyMutationAllowed ||
		intent.AuthorityGranted ||
		intent.BodyTarget != "none" ||
		!intent.Passed ||
		intent.Reason != "weighted resonance shadow graft admission final gate intent drafted from blocked final gate; live admission remains closed" {
		t.Fatalf("weighted admission resonance graft admission final gate intent lost contract: %+v", intent)
	}

	notReadyPath := filepath.Join(dir, "not_ready_final_gate.json")
	writeWeightedAdmissionResonanceGraftAdmissionFinalGateFixture(t, notReadyPath)
	writeWeightedReadinessFixture(t, notReadyPath, stringsReplaceFirst(readText(t, notReadyPath), `"weighted_admission_resonance_graft_admission_final_gate_ready": true`, `"weighted_admission_resonance_graft_admission_final_gate_ready": false`))
	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateIntent([]string{notReadyPath, filepath.Join(dir, "not_ready_intent.json")}),
		"weighted admission resonance graft admission final gate weighted_admission_resonance_graft_admission_final_gate_ready not ready",
	)

	openedFinalGatePath := filepath.Join(dir, "opened_final_gate.json")
	writeWeightedAdmissionResonanceGraftAdmissionFinalGateFixture(t, openedFinalGatePath)
	writeWeightedReadinessFixture(t, openedFinalGatePath, stringsReplaceFirst(readText(t, openedFinalGatePath), `"admission_final_gate_ready": false`, `"admission_final_gate_ready": true`))
	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateIntent([]string{openedFinalGatePath, filepath.Join(dir, "opened_intent.json")}),
		"weighted admission resonance graft admission final gate opened admission_final_gate_ready",
	)

	badSchemaPath := filepath.Join(dir, "bad_schema.json")
	writeWeightedAdmissionResonanceGraftAdmissionFinalGateFixture(t, badSchemaPath)
	writeWeightedReadinessFixture(t, badSchemaPath, stringsReplaceFirst(readText(t, badSchemaPath), `"schema": "arianna.live_route_weighted_admission_resonance_graft_admission_final_gate.v1"`, `"schema": "arianna.live_route_weighted_admission_resonance_graft_admission_final_gate.v0"`))
	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateIntent([]string{badSchemaPath, filepath.Join(dir, "bad_schema_intent.json")}),
		`weighted admission resonance graft admission final gate schema mismatch: got "arianna.live_route_weighted_admission_resonance_graft_admission_final_gate.v0" want "`+admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateSchema+`"`,
	)

	badHashPath := filepath.Join(dir, "bad_hash.json")
	writeWeightedAdmissionResonanceGraftAdmissionFinalGateFixture(t, badHashPath)
	writeWeightedReadinessFixture(t, badHashPath, stringsReplaceFirst(readText(t, badHashPath), `"admission_final_gate_hash": "weighted-resonance-graft-admission-final-gate-`, `"admission_final_gate_hash": "weighted-resonance-graft-admission-final-gate-bad`))
	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateIntent([]string{badHashPath, filepath.Join(dir, "bad_hash_intent.json")}),
		"weighted admission resonance graft admission final gate admission_final_gate_hash mismatch",
	)

	if err := runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateIntent([]string{finalGatePath, filepath.Join(dir, "missing", "intent.json")}); err == nil ||
		!strings.HasPrefix(err.Error(), "weighted admission resonance graft admission final gate intent write failed:") {
		t.Fatalf("expected weighted admission resonance graft admission final gate intent write failure, got %v", err)
	}
}
