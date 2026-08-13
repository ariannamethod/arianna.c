package main

import (
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestAdmissionLiveRouteWeightedAdmissionResonanceReceiver(t *testing.T) {
	dir := t.TempDir()

	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionResonanceReceiver(nil),
		"usage: --admission-live-route-weighted-admission-resonance-receiver RESONANCE_INTENT_REPORT RESONANCE_RECEIVER_REPORT",
	)
	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionResonanceReceiver([]string{"intent.json"}),
		"usage: --admission-live-route-weighted-admission-resonance-receiver RESONANCE_INTENT_REPORT RESONANCE_RECEIVER_REPORT",
	)
	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionResonanceReceiver([]string{"intent.json", "receiver.json", "extra"}),
		"usage: --admission-live-route-weighted-admission-resonance-receiver RESONANCE_INTENT_REPORT RESONANCE_RECEIVER_REPORT",
	)
	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionResonanceReceiver([]string{"  ", filepath.Join(dir, "receiver.json")}),
		"weighted admission resonance intent path missing",
	)

	intentPath := filepath.Join(dir, "intent.json")
	writeWeightedAdmissionResonanceIntentFixture(t, intentPath)

	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionResonanceReceiver([]string{intentPath, "  "}),
		"weighted admission resonance receiver output path missing",
	)

	receiverPath := filepath.Join(dir, "receiver.json")
	if err := runAdmissionLiveRouteWeightedAdmissionResonanceReceiver([]string{intentPath, receiverPath}); err != nil {
		t.Fatalf("valid weighted admission resonance receiver rejected: %v", err)
	}
	raw, err := os.ReadFile(receiverPath)
	if err != nil {
		t.Fatalf("read weighted admission resonance receiver: %v", err)
	}
	var receiver admissionLiveRouteWeightedAdmissionResonanceReceiverReport
	if err := json.Unmarshal(raw, &receiver); err != nil {
		t.Fatalf("decode weighted admission resonance receiver: %v", err)
	}
	if receiver.Schema != admissionLiveRouteWeightedAdmissionResonanceReceiverSchema ||
		receiver.Status != "receiver_previewed_dry_run" ||
		receiver.Target != "resonance" ||
		receiver.TargetKind != "weighted_live_route_first_receiver" ||
		receiver.TargetMode != "bounded_direction_preview_dry_run" ||
		receiver.Action != "preview_weighted_resonance_receive_dry_run" ||
		!receiver.WeightedAdmissionResonanceReceiverReady ||
		!receiver.WeightedAdmissionResonanceIntentConsumed ||
		!receiver.WeightedAdmissionResonanceIntentRequired ||
		!receiver.NextStepBlockedWithoutResonanceReceiver ||
		receiver.WeightedAdmissionResonanceReceiverID == "" ||
		receiver.WeightedAdmissionResonanceReceiverID != admissionLiveRouteWeightedAdmissionResonanceReceiverID(receiver) ||
		receiver.Receiver != "resonance" ||
		receiver.ReceiverKind != "internal_world" ||
		receiver.InfluenceKind != "bounded_direction" ||
		receiver.MaxInfluence != admissionLiveRouteTurnCandidateAdmissionResonanceIntentMaxGain ||
		receiver.TTLTurns != admissionLiveRouteTurnCandidateAdmissionResonanceIntentTTL ||
		receiver.CausalID == "" ||
		receiver.CausalID != admissionLiveRouteWeightedAdmissionResonanceReceiverCausalID(receiver) ||
		receiver.PreStateHash == "" ||
		receiver.PreStateHash != admissionLiveRouteWeightedAdmissionResonanceReceiverPreStateHash(receiver) ||
		receiver.PostStateHash == "" ||
		receiver.PostStateHash != admissionLiveRouteWeightedAdmissionResonanceReceiverPostStateHash(receiver) ||
		receiver.StateDeltaHash == "" ||
		receiver.StateDeltaHash != admissionLiveRouteWeightedAdmissionResonanceReceiverStateDeltaHash(receiver) ||
		receiver.PreStateHash == receiver.PostStateHash ||
		receiver.StateHashMode != "sealed_metadata_preview" ||
		!receiver.DryRunOnly ||
		receiver.RawDreamTextObserved ||
		receiver.RawDreamTextForwarded ||
		receiver.JanusSurfaceAllowed ||
		receiver.CoocLearningAllowed ||
		receiver.DeltaHarvestAllowed ||
		receiver.BodyMutationAllowed ||
		!receiver.RollbackRequired ||
		receiver.SourceSchema != admissionLiveRouteWeightedAdmissionResonanceIntentSchema ||
		receiver.SourceStatus != "resonance_intent_drafted_dry_run" ||
		receiver.SourceTarget != "resonance" ||
		receiver.SourceReport != intentPath ||
		receiver.SourceFinalGateReport == "" ||
		receiver.SourceSealReport == "" ||
		receiver.SourcePermitReport == "" ||
		receiver.SourceAuthorityReport == "" ||
		receiver.SourceContractReport == "" ||
		receiver.SourcePreconditionReport == "" ||
		receiver.SourceReadinessReport == "" ||
		receiver.SourceBodyWorkdir == "" ||
		receiver.SourceBoundaryReport == "" ||
		receiver.SourceProofLog == "" ||
		receiver.SourceFinalGateLog == "" ||
		!receiver.SourceWeightedAdmissionResonanceIntentReady ||
		!receiver.SourceWeightedAdmissionFinalGateConsumed ||
		!receiver.SourceWeightedAdmissionFinalGateRequired ||
		!receiver.SourceWeightedAdmissionFinalGateReady ||
		!receiver.SourceWeightedAdmissionSealConsumed ||
		!receiver.SourceWeightedAdmissionSealRequired ||
		!receiver.SourceWeightedAdmissionSealReady ||
		!receiver.SourceWeightedAdmissionPermitConsumed ||
		!receiver.SourceWeightedAdmissionPermitRequired ||
		!receiver.SourceWeightedAdmissionPermitReady ||
		!receiver.SourceWeightedAdmissionAuthorityConsumed ||
		!receiver.SourceWeightedAdmissionAuthorityRequired ||
		!receiver.SourceManualPermitRequested ||
		!receiver.SourcePermitKeyMatched ||
		receiver.SourceRawDreamTextAllowed ||
		receiver.SourceJanusSurfaceAllowed ||
		receiver.SourceCoocLearningAllowed ||
		receiver.SourceDeltaHarvestAllowed ||
		!receiver.SourceRollbackRequired ||
		!receiver.SourcePreStateHashRequired ||
		!receiver.SourcePostStateHashRequired ||
		!receiver.BodySmokeWeighted ||
		!receiver.NanoDirectRunner ||
		!receiver.NanoDirectFinalGate ||
		!receiver.ResonanceGraftAdmissionProof ||
		!receiver.BoundaryReportFullChain ||
		receiver.SourceAuthorityGranted ||
		receiver.AuthorityGranted ||
		receiver.ContractsReady ||
		receiver.WriteAllowed ||
		receiver.AdmissionAllowed ||
		receiver.LiveAdmissionEnabled ||
		receiver.MutatesState ||
		!receiver.Passed ||
		receiver.Reason != "weighted resonance receiver previewed sealed intent without body mutation" {
		t.Fatalf("weighted admission resonance receiver lost contract: %+v", receiver)
	}

	openedPath := filepath.Join(dir, "opened_intent.json")
	writeWeightedAdmissionResonanceIntentFixture(t, openedPath)
	rawOpened, err := os.ReadFile(openedPath)
	if err != nil {
		t.Fatalf("read opened intent fixture: %v", err)
	}
	writeWeightedReadinessFixture(t, openedPath, stringsReplaceFirst(string(rawOpened), `"live_admission_enabled": false`, `"live_admission_enabled": true`))
	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionResonanceReceiver([]string{openedPath, filepath.Join(dir, "opened_receiver.json")}),
		"weighted admission resonance intent opened live_admission_enabled",
	)

	badSchemaPath := filepath.Join(dir, "bad_schema_intent.json")
	writeWeightedAdmissionResonanceIntentFixture(t, badSchemaPath)
	rawBadSchema, err := os.ReadFile(badSchemaPath)
	if err != nil {
		t.Fatalf("read bad schema intent fixture: %v", err)
	}
	writeWeightedReadinessFixture(t, badSchemaPath, stringsReplaceFirst(string(rawBadSchema), `"schema": "arianna.live_route_weighted_admission_resonance_intent.v1"`, `"schema": "arianna.live_route_weighted_admission_resonance_intent.v0"`))
	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionResonanceReceiver([]string{badSchemaPath, filepath.Join(dir, "bad_schema_receiver.json")}),
		`weighted admission resonance intent schema mismatch: got "arianna.live_route_weighted_admission_resonance_intent.v0" want "`+admissionLiveRouteWeightedAdmissionResonanceIntentSchema+`"`,
	)

	notReadyPath := filepath.Join(dir, "not_ready_intent.json")
	writeWeightedAdmissionResonanceIntentFixture(t, notReadyPath)
	rawNotReady, err := os.ReadFile(notReadyPath)
	if err != nil {
		t.Fatalf("read not-ready intent fixture: %v", err)
	}
	writeWeightedReadinessFixture(t, notReadyPath, stringsReplaceFirst(string(rawNotReady), `"weighted_admission_resonance_intent_ready": true`, `"weighted_admission_resonance_intent_ready": false`))
	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionResonanceReceiver([]string{notReadyPath, filepath.Join(dir, "not_ready_receiver.json")}),
		"weighted admission resonance intent weighted_admission_resonance_intent_ready not ready",
	)

	badReceiverPath := filepath.Join(dir, "bad_receiver_intent.json")
	writeWeightedAdmissionResonanceIntentFixture(t, badReceiverPath)
	rawBadReceiver, err := os.ReadFile(badReceiverPath)
	if err != nil {
		t.Fatalf("read bad receiver intent fixture: %v", err)
	}
	writeWeightedReadinessFixture(t, badReceiverPath, stringsReplaceFirst(string(rawBadReceiver), `"receiver": "resonance"`, `"receiver": "janus"`))
	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionResonanceReceiver([]string{badReceiverPath, filepath.Join(dir, "bad_receiver.json")}),
		`weighted admission resonance intent receiver mismatch: got "janus" want "resonance"`,
	)

	if err := runAdmissionLiveRouteWeightedAdmissionResonanceReceiver([]string{intentPath, filepath.Join(dir, "missing", "receiver.json")}); err == nil ||
		!strings.HasPrefix(err.Error(), "weighted admission resonance receiver write failed:") {
		t.Fatalf("expected weighted admission resonance receiver write failure, got %v", err)
	}
}
