package main

import (
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateReceiver(t *testing.T) {
	dir := t.TempDir()

	usage := "usage: --admission-live-route-weighted-admission-resonance-graft-admission-final-gate-receiver RESONANCE_GRAFT_ADMISSION_FINAL_GATE_INTENT_REPORT RESONANCE_GRAFT_ADMISSION_FINAL_GATE_RECEIVER_REPORT"
	requireBoundaryAssertError(t, runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateReceiver(nil), usage)
	requireBoundaryAssertError(t, runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateReceiver([]string{"intent.json"}), usage)
	requireBoundaryAssertError(t, runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateReceiver([]string{"intent.json", "receiver.json", "extra"}), usage)
	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateReceiver([]string{"  ", filepath.Join(dir, "receiver.json")}),
		"weighted admission resonance graft admission final gate intent path missing",
	)

	intentPath := filepath.Join(dir, "intent.json")
	writeWeightedAdmissionResonanceGraftAdmissionFinalGateIntentFixture(t, intentPath)
	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateReceiver([]string{intentPath, "  "}),
		"weighted admission resonance graft admission final gate receiver output path missing",
	)

	receiverPath := filepath.Join(dir, "receiver.json")
	if err := runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateReceiver([]string{intentPath, receiverPath}); err != nil {
		t.Fatalf("valid weighted admission resonance graft admission final gate receiver rejected: %v", err)
	}
	raw, err := os.ReadFile(receiverPath)
	if err != nil {
		t.Fatalf("read weighted admission resonance graft admission final gate receiver: %v", err)
	}
	var receiver admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateReceiverReport
	if err := json.Unmarshal(raw, &receiver); err != nil {
		t.Fatalf("decode weighted admission resonance graft admission final gate receiver: %v", err)
	}
	sourceRaw, err := os.ReadFile(intentPath)
	if err != nil {
		t.Fatalf("read weighted admission resonance graft admission final gate intent: %v", err)
	}
	var sourceIntent admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateIntentReport
	if err := json.Unmarshal(sourceRaw, &sourceIntent); err != nil {
		t.Fatalf("decode weighted admission resonance graft admission final gate intent: %v", err)
	}
	if receiver.Schema != admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateReceiverSchema ||
		receiver.Status != "shadow_graft_admission_final_gate_receiver_previewed_dry_run" ||
		receiver.Target != "live_route_admission_next_step" ||
		receiver.TargetKind != "weighted_internal_world_shadow_graft_admission_final_gate_receiver" ||
		receiver.TargetMode != "bounded_receiver_preview_dry_run" ||
		receiver.Action != "preview_weighted_resonance_shadow_graft_admission_final_gate_receiver_dry_run" ||
		receiver.WriterAction != "reject_blocked_admission_final_gate_receiver" ||
		receiver.RollbackAction != "reject_blocked_admission_final_gate_receiver" ||
		receiver.LedgerState != "blocked" ||
		receiver.LedgerAction != "reject_blocked_admission_final_gate_receiver" ||
		receiver.LedgerContract != "none" ||
		receiver.LedgerEntrypoint != "none" ||
		receiver.LedgerReceiptShape != "none" ||
		receiver.LedgerWriteScope != "none" ||
		receiver.ReceiptShape != "weighted_resonance_shadow_graft_admission_final_gate_receiver_receipt" ||
		receiver.AdmissionFinalGateReceiverState != "previewed" ||
		receiver.AdmissionFinalGateReceiverAction != "preview_blocked_final_gate_receiver" ||
		receiver.AdmissionFinalGateReceiverTarget != "resonance" ||
		receiver.AdmissionFinalGateReceiverTargetKind != "weighted_internal_world_shadow_graft_admission_final_gate_intent" ||
		receiver.AdmissionFinalGateReceiverTargetMode != "bounded_receiver_preview_dry_run" ||
		!receiver.AdmissionFinalGateReceiverDryRunOnly ||
		receiver.AdmissionFinalGateReceiverIntentVerified ||
		receiver.AdmissionFinalGateReceiverFinalGateVerified ||
		receiver.AdmissionFinalGateReceiverReady ||
		receiver.FinalGateReceiver != "resonance" ||
		receiver.FinalGateReceiverKind != "internal_world" ||
		receiver.FinalGateReceiverInfluenceKind != "bounded_direction" ||
		receiver.FinalGateReceiverMaxInfluence != admissionLiveRouteTurnCandidateAdmissionResonanceIntentMaxGain ||
		receiver.FinalGateReceiverTTLTurns != admissionLiveRouteTurnCandidateAdmissionResonanceIntentTTL ||
		receiver.FinalGateReceiverStateHashMode != "blocked_intent_receiver_preview" ||
		receiver.FinalGateReceiverRawDreamTextObserved ||
		receiver.FinalGateReceiverRawDreamTextForwarded ||
		receiver.FinalGateReceiverRawDreamTextAllowed ||
		receiver.FinalGateReceiverJanusSurfaceAllowed ||
		receiver.FinalGateReceiverCoocLearningAllowed ||
		receiver.FinalGateReceiverDeltaHarvestAllowed ||
		receiver.FinalGateReceiverBodyMutationAllowed ||
		!receiver.FinalGateReceiverPreStateHashRequired ||
		!receiver.FinalGateReceiverPostStateHashRequired ||
		!receiver.WeightedAdmissionResonanceGraftAdmissionFinalGateReceiverReady ||
		!receiver.WeightedAdmissionResonanceGraftAdmissionFinalGateIntentConsumed ||
		!receiver.WeightedAdmissionResonanceGraftAdmissionFinalGateIntentRequired ||
		!receiver.NextStepBlockedWithoutResonanceGraftAdmissionFinalGateReceiver ||
		receiver.SourceSchema != admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateIntentSchema ||
		receiver.SourceStatus != "shadow_graft_admission_final_gate_intent_blocked_dry_run" ||
		receiver.SourceTarget != "live_route_admission_next_step" ||
		receiver.SourceReport != intentPath ||
		receiver.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateIntentID != sourceIntent.WeightedAdmissionResonanceGraftAdmissionFinalGateIntentID ||
		receiver.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateIntentCausal != sourceIntent.CausalID ||
		receiver.SourceAdmissionFinalGateIntentHash != sourceIntent.AdmissionFinalGateIntentHash ||
		receiver.SourceAdmissionFinalGateIntentReadBack != sourceIntent.AdmissionFinalGateIntentReadBackHash ||
		receiver.SourceAdmissionFinalGateIntentReceiptShape != sourceIntent.ReceiptShape ||
		receiver.SourceAdmissionFinalGateIntentAction != sourceIntent.AdmissionFinalGateIntentAction ||
		!receiver.SourceAdmissionFinalGateIntentDryRunOnly ||
		receiver.SourceAdmissionFinalGateIntentFinalGateVerified ||
		receiver.SourceAdmissionFinalGateIntentSealVerified ||
		receiver.SourceAdmissionFinalGateIntentReady ||
		receiver.SourceFinalGateIntentReceiver != sourceIntent.FinalGateIntentReceiver ||
		receiver.SourceFinalGateIntentReceiverKind != sourceIntent.FinalGateIntentReceiverKind ||
		receiver.SourceFinalGateIntentInfluenceKind != sourceIntent.FinalGateIntentInfluenceKind ||
		receiver.SourceFinalGateIntentRawDreamTextAllowed ||
		receiver.SourceFinalGateIntentJanusSurfaceAllowed ||
		receiver.SourceFinalGateIntentCoocLearningAllowed ||
		receiver.SourceFinalGateIntentDeltaHarvestAllowed ||
		!receiver.SourceFinalGateIntentPreStateHashRequired ||
		!receiver.SourceFinalGateIntentPostStateHashRequired ||
		receiver.CausalID != admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateReceiverCausalID(receiver) ||
		receiver.AdmissionFinalGateReceiverPreStateHash != admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateReceiverPreStateHash(receiver) ||
		receiver.AdmissionFinalGateReceiverPostStateHash != admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateReceiverPostStateHash(receiver) ||
		receiver.AdmissionFinalGateReceiverStateDeltaHash != admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateReceiverStateDeltaHash(receiver) ||
		receiver.AdmissionFinalGateReceiverPreStateHash == receiver.AdmissionFinalGateReceiverPostStateHash ||
		receiver.WeightedAdmissionResonanceGraftAdmissionFinalGateReceiverID != admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateReceiverID(receiver) ||
		receiver.LedgerReady ||
		receiver.LedgerAppendAllowed ||
		receiver.WriteAllowed ||
		receiver.AdmissionAllowed ||
		receiver.LiveAdmissionEnabled ||
		receiver.MutatesState ||
		receiver.BodyMutationAllowed ||
		receiver.AuthorityGranted ||
		receiver.BodyTarget != "none" ||
		!receiver.Passed ||
		receiver.Reason != "weighted resonance shadow graft admission final gate receiver previewed from blocked final gate intent; live admission remains closed" {
		t.Fatalf("weighted admission resonance graft admission final gate receiver lost contract: %+v", receiver)
	}

	notReadyPath := filepath.Join(dir, "not_ready_intent.json")
	writeWeightedAdmissionResonanceGraftAdmissionFinalGateIntentFixture(t, notReadyPath)
	writeWeightedReadinessFixture(t, notReadyPath, stringsReplaceFirst(readText(t, notReadyPath), `"weighted_admission_resonance_graft_admission_final_gate_intent_ready": true`, `"weighted_admission_resonance_graft_admission_final_gate_intent_ready": false`))
	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateReceiver([]string{notReadyPath, filepath.Join(dir, "not_ready_receiver.json")}),
		"weighted admission resonance graft admission final gate intent weighted_admission_resonance_graft_admission_final_gate_intent_ready not ready",
	)

	openedIntentPath := filepath.Join(dir, "opened_intent.json")
	writeWeightedAdmissionResonanceGraftAdmissionFinalGateIntentFixture(t, openedIntentPath)
	writeWeightedReadinessFixture(t, openedIntentPath, stringsReplaceFirst(readText(t, openedIntentPath), `"admission_final_gate_intent_ready": false`, `"admission_final_gate_intent_ready": true`))
	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateReceiver([]string{openedIntentPath, filepath.Join(dir, "opened_receiver.json")}),
		"weighted admission resonance graft admission final gate intent opened admission_final_gate_intent_ready",
	)

	badSchemaPath := filepath.Join(dir, "bad_schema.json")
	writeWeightedAdmissionResonanceGraftAdmissionFinalGateIntentFixture(t, badSchemaPath)
	writeWeightedReadinessFixture(t, badSchemaPath, stringsReplaceFirst(readText(t, badSchemaPath), `"schema": "arianna.live_route_weighted_admission_resonance_graft_admission_final_gate_intent.v1"`, `"schema": "arianna.live_route_weighted_admission_resonance_graft_admission_final_gate_intent.v0"`))
	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateReceiver([]string{badSchemaPath, filepath.Join(dir, "bad_schema_receiver.json")}),
		`weighted admission resonance graft admission final gate intent schema mismatch: got "arianna.live_route_weighted_admission_resonance_graft_admission_final_gate_intent.v0" want "`+admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateIntentSchema+`"`,
	)

	badHashPath := filepath.Join(dir, "bad_hash.json")
	writeWeightedAdmissionResonanceGraftAdmissionFinalGateIntentFixture(t, badHashPath)
	writeWeightedReadinessFixture(t, badHashPath, stringsReplaceFirst(readText(t, badHashPath), `"admission_final_gate_intent_hash": "weighted-resonance-graft-admission-final-gate-intent-`, `"admission_final_gate_intent_hash": "weighted-resonance-graft-admission-final-gate-intent-bad`))
	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateReceiver([]string{badHashPath, filepath.Join(dir, "bad_hash_receiver.json")}),
		"weighted admission resonance graft admission final gate intent admission_final_gate_intent_hash mismatch",
	)

	if err := runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateReceiver([]string{intentPath, filepath.Join(dir, "missing", "receiver.json")}); err == nil ||
		!strings.HasPrefix(err.Error(), "weighted admission resonance graft admission final gate receiver write failed:") {
		t.Fatalf("expected weighted admission resonance graft admission final gate receiver write failure, got %v", err)
	}
}
