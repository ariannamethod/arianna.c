package main

import (
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservation(t *testing.T) {
	dir := t.TempDir()

	usage := "usage: --admission-live-route-weighted-admission-resonance-graft-admission-final-gate-observation RESONANCE_GRAFT_ADMISSION_FINAL_GATE_RECEIVER_REPORT RESONANCE_GRAFT_ADMISSION_FINAL_GATE_OBSERVATION_REPORT"
	requireBoundaryAssertError(t, runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservation(nil), usage)
	requireBoundaryAssertError(t, runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservation([]string{"receiver.json"}), usage)
	requireBoundaryAssertError(t, runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservation([]string{"receiver.json", "observation.json", "extra"}), usage)
	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservation([]string{"  ", filepath.Join(dir, "observation.json")}),
		"weighted admission resonance graft admission final gate receiver path missing",
	)

	receiverPath := filepath.Join(dir, "receiver.json")
	writeWeightedAdmissionResonanceGraftAdmissionFinalGateReceiverFixture(t, receiverPath)
	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservation([]string{receiverPath, "  "}),
		"weighted admission resonance graft admission final gate observation output path missing",
	)

	observationPath := filepath.Join(dir, "observation.json")
	if err := runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservation([]string{receiverPath, observationPath}); err != nil {
		t.Fatalf("valid weighted admission resonance graft admission final gate observation rejected: %v", err)
	}
	raw, err := os.ReadFile(observationPath)
	if err != nil {
		t.Fatalf("read weighted admission resonance graft admission final gate observation: %v", err)
	}
	var observation admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationReport
	if err := json.Unmarshal(raw, &observation); err != nil {
		t.Fatalf("decode weighted admission resonance graft admission final gate observation: %v", err)
	}
	sourceRaw, err := os.ReadFile(receiverPath)
	if err != nil {
		t.Fatalf("read weighted admission resonance graft admission final gate receiver: %v", err)
	}
	var sourceReceiver admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateReceiverReport
	if err := json.Unmarshal(sourceRaw, &sourceReceiver); err != nil {
		t.Fatalf("decode weighted admission resonance graft admission final gate receiver: %v", err)
	}
	if observation.Schema != admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationSchema ||
		observation.Status != "shadow_graft_admission_final_gate_observation_recorded_dry_run" ||
		observation.Target != "live_route_admission_next_step" ||
		observation.TargetKind != "weighted_internal_world_shadow_graft_admission_final_gate_observation" ||
		observation.TargetMode != "append_only_read_back_dry_run" ||
		observation.Action != "record_weighted_resonance_shadow_graft_admission_final_gate_observation_dry_run" ||
		observation.WriterAction != "reject_blocked_admission_final_gate_observation" ||
		observation.RollbackAction != "reject_blocked_admission_final_gate_observation" ||
		observation.LedgerState != "blocked" ||
		observation.LedgerAction != "reject_blocked_admission_final_gate_observation" ||
		observation.ReceiptShape != "weighted_resonance_shadow_graft_admission_final_gate_observation_receipt" ||
		observation.AdmissionFinalGateObservationState != "recorded" ||
		observation.AdmissionFinalGateObservationAction != "record_blocked_final_gate_receiver_observation" ||
		observation.AdmissionFinalGateObservationTarget != "resonance" ||
		observation.AdmissionFinalGateObservationTargetKind != "weighted_internal_world_shadow_graft_admission_final_gate_receiver" ||
		observation.AdmissionFinalGateObservationTargetMode != "append_only_read_back_dry_run" ||
		!observation.AdmissionFinalGateObservationDryRunOnly ||
		!observation.AdmissionFinalGateObservationAppendOnly ||
		!observation.AdmissionFinalGateObservationReadBack ||
		!observation.AdmissionFinalGateObservationReceiptVerified ||
		observation.AdmissionFinalGateObservationReceiverVerified ||
		observation.AdmissionFinalGateObservationReady ||
		observation.FinalGateObservationObserver != "resonance" ||
		observation.FinalGateObservationObserverKind != "internal_world" ||
		observation.FinalGateObservationKind != "blocked_final_gate_receiver_state_proof" ||
		observation.FinalGateObservationMode != "sealed_metadata_observation" ||
		observation.FinalGateObservationRawDreamTextObserved ||
		observation.FinalGateObservationRawDreamTextForwarded ||
		observation.FinalGateObservationRawDreamTextAllowed ||
		observation.FinalGateObservationJanusSurfaceAllowed ||
		observation.FinalGateObservationCoocLearningAllowed ||
		observation.FinalGateObservationDeltaHarvestAllowed ||
		observation.FinalGateObservationBodyMutationAllowed ||
		!observation.FinalGateObservationPreStateHashRequired ||
		!observation.FinalGateObservationPostStateHashRequired ||
		!observation.WeightedAdmissionResonanceGraftAdmissionFinalGateObservationReady ||
		!observation.WeightedAdmissionResonanceGraftAdmissionFinalGateReceiverConsumed ||
		!observation.WeightedAdmissionResonanceGraftAdmissionFinalGateReceiverRequired ||
		!observation.NextStepBlockedWithoutResonanceGraftAdmissionFinalGateObservation ||
		observation.SourceSchema != admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateReceiverSchema ||
		observation.SourceStatus != "shadow_graft_admission_final_gate_receiver_previewed_dry_run" ||
		observation.SourceTarget != "live_route_admission_next_step" ||
		observation.SourceReport != receiverPath ||
		observation.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateReceiverID != sourceReceiver.WeightedAdmissionResonanceGraftAdmissionFinalGateReceiverID ||
		observation.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateReceiverCausal != sourceReceiver.CausalID ||
		observation.SourceAdmissionFinalGateReceiverPreStateHash != sourceReceiver.AdmissionFinalGateReceiverPreStateHash ||
		observation.SourceAdmissionFinalGateReceiverPostStateHash != sourceReceiver.AdmissionFinalGateReceiverPostStateHash ||
		observation.SourceAdmissionFinalGateReceiverStateDeltaHash != sourceReceiver.AdmissionFinalGateReceiverStateDeltaHash ||
		observation.SourceAdmissionFinalGateReceiverReceiptShape != sourceReceiver.ReceiptShape ||
		observation.SourceAdmissionFinalGateReceiverAction != sourceReceiver.AdmissionFinalGateReceiverAction ||
		!observation.SourceAdmissionFinalGateReceiverDryRunOnly ||
		observation.SourceAdmissionFinalGateReceiverReady ||
		observation.SourceFinalGateReceiver != sourceReceiver.FinalGateReceiver ||
		observation.CausalID != admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationCausalID(observation) ||
		observation.AdmissionFinalGateObservationAppendHash != admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationAppendHash(observation) ||
		observation.AdmissionFinalGateObservationReadBackHash != admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationReadBackHash(observation) ||
		observation.AdmissionFinalGateObservationAppendHash == observation.AdmissionFinalGateObservationReadBackHash ||
		observation.WeightedAdmissionResonanceGraftAdmissionFinalGateObservationID != admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationID(observation) ||
		observation.LedgerReady ||
		observation.LedgerAppendAllowed ||
		observation.WriteAllowed ||
		observation.AdmissionAllowed ||
		observation.LiveAdmissionEnabled ||
		observation.MutatesState ||
		observation.BodyMutationAllowed ||
		observation.AuthorityGranted ||
		observation.BodyTarget != "none" ||
		!observation.Passed ||
		observation.Reason != "weighted resonance shadow graft admission final gate observation recorded from blocked receiver; live admission remains closed" {
		t.Fatalf("weighted admission resonance graft admission final gate observation lost contract: %+v", observation)
	}

	notReadyPath := filepath.Join(dir, "not_ready_receiver.json")
	writeWeightedAdmissionResonanceGraftAdmissionFinalGateReceiverFixture(t, notReadyPath)
	writeWeightedReadinessFixture(t, notReadyPath, stringsReplaceFirst(readText(t, notReadyPath), `"weighted_admission_resonance_graft_admission_final_gate_receiver_ready": true`, `"weighted_admission_resonance_graft_admission_final_gate_receiver_ready": false`))
	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservation([]string{notReadyPath, filepath.Join(dir, "not_ready_observation.json")}),
		"weighted admission resonance graft admission final gate receiver weighted_admission_resonance_graft_admission_final_gate_receiver_ready not ready",
	)

	openedReceiverPath := filepath.Join(dir, "opened_receiver.json")
	writeWeightedAdmissionResonanceGraftAdmissionFinalGateReceiverFixture(t, openedReceiverPath)
	writeWeightedReadinessFixture(t, openedReceiverPath, stringsReplaceFirst(readText(t, openedReceiverPath), `"admission_final_gate_receiver_ready": false`, `"admission_final_gate_receiver_ready": true`))
	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservation([]string{openedReceiverPath, filepath.Join(dir, "opened_observation.json")}),
		"weighted admission resonance graft admission final gate receiver opened admission_final_gate_receiver_ready",
	)

	badHashPath := filepath.Join(dir, "bad_hash.json")
	writeWeightedAdmissionResonanceGraftAdmissionFinalGateReceiverFixture(t, badHashPath)
	writeWeightedReadinessFixture(t, badHashPath, stringsReplaceFirst(readText(t, badHashPath), `"admission_final_gate_receiver_pre_state_hash": "weighted-resonance-graft-admission-final-gate-receiver-pre-`, `"admission_final_gate_receiver_pre_state_hash": "weighted-resonance-graft-admission-final-gate-receiver-pre-bad`))
	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservation([]string{badHashPath, filepath.Join(dir, "bad_hash_observation.json")}),
		"weighted admission resonance graft admission final gate receiver pre_state_hash mismatch",
	)

	if err := runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservation([]string{receiverPath, filepath.Join(dir, "missing", "observation.json")}); err == nil ||
		!strings.HasPrefix(err.Error(), "weighted admission resonance graft admission final gate observation write failed:") {
		t.Fatalf("expected weighted admission resonance graft admission final gate observation write failure, got %v", err)
	}
}
