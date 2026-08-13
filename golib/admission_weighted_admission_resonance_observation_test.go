package main

import (
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestAdmissionLiveRouteWeightedAdmissionResonanceObservation(t *testing.T) {
	dir := t.TempDir()

	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionResonanceObservation(nil),
		"usage: --admission-live-route-weighted-admission-resonance-observation RESONANCE_RECEIVER_REPORT RESONANCE_OBSERVATION_REPORT",
	)
	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionResonanceObservation([]string{"receiver.json"}),
		"usage: --admission-live-route-weighted-admission-resonance-observation RESONANCE_RECEIVER_REPORT RESONANCE_OBSERVATION_REPORT",
	)
	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionResonanceObservation([]string{"receiver.json", "observation.json", "extra"}),
		"usage: --admission-live-route-weighted-admission-resonance-observation RESONANCE_RECEIVER_REPORT RESONANCE_OBSERVATION_REPORT",
	)
	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionResonanceObservation([]string{"  ", filepath.Join(dir, "observation.json")}),
		"weighted admission resonance receiver path missing",
	)

	receiverPath := filepath.Join(dir, "receiver.json")
	writeWeightedAdmissionResonanceReceiverFixture(t, receiverPath)

	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionResonanceObservation([]string{receiverPath, "  "}),
		"weighted admission resonance observation output path missing",
	)

	observationPath := filepath.Join(dir, "observation.json")
	if err := runAdmissionLiveRouteWeightedAdmissionResonanceObservation([]string{receiverPath, observationPath}); err != nil {
		t.Fatalf("valid weighted admission resonance observation rejected: %v", err)
	}
	raw, err := os.ReadFile(observationPath)
	if err != nil {
		t.Fatalf("read weighted admission resonance observation: %v", err)
	}
	var observation admissionLiveRouteWeightedAdmissionResonanceObservationReport
	if err := json.Unmarshal(raw, &observation); err != nil {
		t.Fatalf("decode weighted admission resonance observation: %v", err)
	}
	if observation.Schema != admissionLiveRouteWeightedAdmissionResonanceObservationSchema ||
		observation.Status != "observation_recorded_dry_run" ||
		observation.Target != "resonance" ||
		observation.TargetKind != "weighted_internal_world_observation" ||
		observation.TargetMode != "append_only_read_back_dry_run" ||
		observation.Action != "record_weighted_resonance_receiver_observation_dry_run" ||
		!observation.WeightedAdmissionResonanceObservationReady ||
		!observation.WeightedAdmissionResonanceReceiverConsumed ||
		!observation.WeightedAdmissionResonanceReceiverRequired ||
		!observation.NextStepBlockedWithoutResonanceObservation ||
		observation.WeightedAdmissionResonanceObservationID == "" ||
		observation.WeightedAdmissionResonanceObservationID != admissionLiveRouteWeightedAdmissionResonanceObservationID(observation) ||
		observation.Observer != "resonance" ||
		observation.ObserverKind != "internal_world" ||
		observation.ObservationKind != "weighted_receiver_state_proof" ||
		observation.ObservationMode != "sealed_metadata_observation" ||
		observation.CausalID == "" ||
		observation.CausalID != admissionLiveRouteWeightedAdmissionResonanceObservationCausalID(observation) ||
		observation.AppendHash == "" ||
		observation.AppendHash != admissionLiveRouteWeightedAdmissionResonanceObservationAppendHash(observation) ||
		observation.ReadBackHash == "" ||
		observation.ReadBackHash != admissionLiveRouteWeightedAdmissionResonanceObservationReadBackHash(observation) ||
		observation.AppendHash == observation.ReadBackHash ||
		!observation.AppendOnly ||
		!observation.ReadBack ||
		!observation.ReceiptVerified ||
		!observation.DryRunOnly ||
		observation.RawDreamTextObserved ||
		observation.RawDreamTextForwarded ||
		observation.JanusSurfaceAllowed ||
		observation.CoocLearningAllowed ||
		observation.DeltaHarvestAllowed ||
		observation.BodyMutationAllowed ||
		!observation.RollbackRequired ||
		observation.SourceSchema != admissionLiveRouteWeightedAdmissionResonanceReceiverSchema ||
		observation.SourceStatus != "receiver_previewed_dry_run" ||
		observation.SourceTarget != "resonance" ||
		observation.SourceReport != receiverPath ||
		observation.SourceResonanceIntentReport == "" ||
		observation.SourceFinalGateReport == "" ||
		observation.SourceSealReport == "" ||
		observation.SourcePermitReport == "" ||
		observation.SourceAuthorityReport == "" ||
		observation.SourceContractReport == "" ||
		observation.SourcePreconditionReport == "" ||
		observation.SourceReadinessReport == "" ||
		observation.SourceBodyWorkdir == "" ||
		observation.SourceBoundaryReport == "" ||
		observation.SourceProofLog == "" ||
		observation.SourceFinalGateLog == "" ||
		!strings.HasPrefix(observation.SourceWeightedAdmissionResonanceReceiverID, "weighted-resonance-receiver-") ||
		!observation.SourceWeightedAdmissionResonanceReceiverReady ||
		!strings.HasPrefix(observation.SourceWeightedAdmissionResonanceReceiverCausal, "weighted-resonance-receiver-causal-") ||
		!strings.HasPrefix(observation.SourceReceiverPreStateHash, "weighted-resonance-receiver-pre-") ||
		!strings.HasPrefix(observation.SourceReceiverPostStateHash, "weighted-resonance-receiver-post-") ||
		!strings.HasPrefix(observation.SourceReceiverStateDeltaHash, "weighted-resonance-receiver-delta-") ||
		observation.SourceReceiverPreStateHash == observation.SourceReceiverPostStateHash ||
		!observation.SourceWeightedAdmissionResonanceIntentConsumed ||
		!observation.SourceWeightedAdmissionResonanceIntentRequired ||
		!observation.SourceWeightedAdmissionResonanceIntentReady ||
		!observation.SourceWeightedAdmissionFinalGateConsumed ||
		!observation.SourceWeightedAdmissionFinalGateRequired ||
		!observation.SourceWeightedAdmissionFinalGateReady ||
		!observation.SourceWeightedAdmissionSealConsumed ||
		!observation.SourceWeightedAdmissionSealRequired ||
		!observation.SourceWeightedAdmissionSealReady ||
		!observation.SourceWeightedAdmissionPermitConsumed ||
		!observation.SourceWeightedAdmissionPermitRequired ||
		!observation.SourceWeightedAdmissionPermitReady ||
		!observation.SourceWeightedAdmissionAuthorityConsumed ||
		!observation.SourceWeightedAdmissionAuthorityRequired ||
		!observation.SourceManualPermitRequested ||
		!observation.SourcePermitKeyMatched ||
		observation.SourceRawDreamTextAllowed ||
		observation.SourceRawDreamTextObserved ||
		observation.SourceRawDreamTextForwarded ||
		observation.SourceJanusSurfaceAllowed ||
		observation.SourceCoocLearningAllowed ||
		observation.SourceDeltaHarvestAllowed ||
		observation.SourceBodyMutationAllowed ||
		!observation.SourceRollbackRequired ||
		!observation.SourcePreStateHashRequired ||
		!observation.SourcePostStateHashRequired ||
		!observation.BodySmokeWeighted ||
		!observation.NanoDirectRunner ||
		!observation.NanoDirectFinalGate ||
		!observation.ResonanceGraftAdmissionProof ||
		!observation.BoundaryReportFullChain ||
		observation.SourceAuthorityGranted ||
		observation.AuthorityGranted ||
		observation.ContractsReady ||
		observation.WriteAllowed ||
		observation.AdmissionAllowed ||
		observation.LiveAdmissionEnabled ||
		observation.MutatesState ||
		observation.BodyTarget != "none" ||
		!observation.Passed ||
		observation.Reason != "weighted resonance observation recorded and read back without body mutation" {
		t.Fatalf("weighted admission resonance observation lost contract: %+v", observation)
	}

	openedPath := filepath.Join(dir, "opened_receiver.json")
	writeWeightedAdmissionResonanceReceiverFixture(t, openedPath)
	rawOpened, err := os.ReadFile(openedPath)
	if err != nil {
		t.Fatalf("read opened receiver fixture: %v", err)
	}
	writeWeightedReadinessFixture(t, openedPath, stringsReplaceFirst(string(rawOpened), `"live_admission_enabled": false`, `"live_admission_enabled": true`))
	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionResonanceObservation([]string{openedPath, filepath.Join(dir, "opened_observation.json")}),
		"weighted admission resonance receiver opened live_admission_enabled",
	)

	badSchemaPath := filepath.Join(dir, "bad_schema_receiver.json")
	writeWeightedAdmissionResonanceReceiverFixture(t, badSchemaPath)
	rawBadSchema, err := os.ReadFile(badSchemaPath)
	if err != nil {
		t.Fatalf("read bad schema receiver fixture: %v", err)
	}
	writeWeightedReadinessFixture(t, badSchemaPath, stringsReplaceFirst(string(rawBadSchema), `"schema": "arianna.live_route_weighted_admission_resonance_receiver.v1"`, `"schema": "arianna.live_route_weighted_admission_resonance_receiver.v0"`))
	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionResonanceObservation([]string{badSchemaPath, filepath.Join(dir, "bad_schema_observation.json")}),
		`weighted admission resonance receiver schema mismatch: got "arianna.live_route_weighted_admission_resonance_receiver.v0" want "`+admissionLiveRouteWeightedAdmissionResonanceReceiverSchema+`"`,
	)

	notReadyPath := filepath.Join(dir, "not_ready_receiver.json")
	writeWeightedAdmissionResonanceReceiverFixture(t, notReadyPath)
	rawNotReady, err := os.ReadFile(notReadyPath)
	if err != nil {
		t.Fatalf("read not-ready receiver fixture: %v", err)
	}
	writeWeightedReadinessFixture(t, notReadyPath, stringsReplaceFirst(string(rawNotReady), `"weighted_admission_resonance_receiver_ready": true`, `"weighted_admission_resonance_receiver_ready": false`))
	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionResonanceObservation([]string{notReadyPath, filepath.Join(dir, "not_ready_observation.json")}),
		"weighted admission resonance receiver weighted_admission_resonance_receiver_ready not ready",
	)

	badHashPath := filepath.Join(dir, "bad_hash_receiver.json")
	writeWeightedAdmissionResonanceReceiverFixture(t, badHashPath)
	rawBadHash, err := os.ReadFile(badHashPath)
	if err != nil {
		t.Fatalf("read bad hash receiver fixture: %v", err)
	}
	writeWeightedReadinessFixture(t, badHashPath, stringsReplaceFirst(string(rawBadHash), `"state_delta_hash": "weighted-resonance-receiver-delta-`, `"state_delta_hash": "weighted-resonance-receiver-delta-bad`))
	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionResonanceObservation([]string{badHashPath, filepath.Join(dir, "bad_hash_observation.json")}),
		"weighted admission resonance receiver state_delta_hash mismatch",
	)

	if err := runAdmissionLiveRouteWeightedAdmissionResonanceObservation([]string{receiverPath, filepath.Join(dir, "missing", "observation.json")}); err == nil ||
		!strings.HasPrefix(err.Error(), "weighted admission resonance observation write failed:") {
		t.Fatalf("expected weighted admission resonance observation write failure, got %v", err)
	}
}
