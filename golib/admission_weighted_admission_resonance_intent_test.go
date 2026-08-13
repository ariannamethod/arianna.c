package main

import (
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestAdmissionLiveRouteWeightedAdmissionResonanceIntent(t *testing.T) {
	dir := t.TempDir()

	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionResonanceIntent(nil),
		"usage: --admission-live-route-weighted-admission-resonance-intent FINAL_GATE_REPORT RESONANCE_INTENT_REPORT",
	)
	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionResonanceIntent([]string{"final_gate.json"}),
		"usage: --admission-live-route-weighted-admission-resonance-intent FINAL_GATE_REPORT RESONANCE_INTENT_REPORT",
	)
	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionResonanceIntent([]string{"final_gate.json", "resonance_intent.json", "extra"}),
		"usage: --admission-live-route-weighted-admission-resonance-intent FINAL_GATE_REPORT RESONANCE_INTENT_REPORT",
	)
	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionResonanceIntent([]string{"  ", filepath.Join(dir, "resonance_intent.json")}),
		"weighted admission final gate path missing",
	)

	finalGatePath := filepath.Join(dir, "final_gate.json")
	writeWeightedAdmissionFinalGateFixture(t, finalGatePath)

	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionResonanceIntent([]string{finalGatePath, "  "}),
		"weighted admission resonance intent output path missing",
	)

	intentPath := filepath.Join(dir, "resonance_intent.json")
	if err := runAdmissionLiveRouteWeightedAdmissionResonanceIntent([]string{finalGatePath, intentPath}); err != nil {
		t.Fatalf("valid weighted admission resonance intent rejected: %v", err)
	}
	raw, err := os.ReadFile(intentPath)
	if err != nil {
		t.Fatalf("read weighted admission resonance intent: %v", err)
	}
	var intent admissionLiveRouteWeightedAdmissionResonanceIntentReport
	if err := json.Unmarshal(raw, &intent); err != nil {
		t.Fatalf("decode weighted admission resonance intent: %v", err)
	}
	if intent.Schema != admissionLiveRouteWeightedAdmissionResonanceIntentSchema ||
		intent.Status != "resonance_intent_drafted_dry_run" ||
		intent.Target != "resonance" ||
		intent.TargetKind != "weighted_live_route_first_receiver" ||
		intent.TargetMode != "bounded_direction_dry_run" ||
		intent.Action != "draft_weighted_resonance_direction_intent_dry_run" ||
		!intent.WeightedAdmissionResonanceIntentReady ||
		!intent.WeightedAdmissionFinalGateConsumed ||
		!intent.WeightedAdmissionFinalGateRequired ||
		!intent.NextStepBlockedWithoutResonanceIntent ||
		intent.Receiver != "resonance" ||
		intent.ReceiverKind != "internal_world" ||
		intent.InfluenceKind != "bounded_direction" ||
		intent.MaxInfluence != admissionLiveRouteTurnCandidateAdmissionResonanceIntentMaxGain ||
		intent.TTLTurns != admissionLiveRouteTurnCandidateAdmissionResonanceIntentTTL ||
		intent.RawDreamTextAllowed ||
		intent.JanusSurfaceAllowed ||
		intent.CoocLearningAllowed ||
		intent.DeltaHarvestAllowed ||
		!intent.RollbackRequired ||
		!intent.PreStateHashRequired ||
		!intent.PostStateHashRequired ||
		intent.SourceSchema != admissionLiveRouteWeightedAdmissionFinalGateSchema ||
		intent.SourceStatus != "ready_closed_dry_run" ||
		intent.SourceTarget != "live_route_admission_final_gate" ||
		intent.SourceReport != finalGatePath ||
		intent.SourceSealReport == "" ||
		intent.SourcePermitReport == "" ||
		intent.SourceAuthorityReport == "" ||
		intent.SourceContractReport == "" ||
		intent.SourcePreconditionReport == "" ||
		intent.SourceReadinessReport == "" ||
		intent.SourceBodyWorkdir == "" ||
		intent.SourceBoundaryReport == "" ||
		intent.SourceProofLog == "" ||
		intent.SourceFinalGateLog == "" ||
		!intent.SourceWeightedAdmissionFinalGateReady ||
		!intent.SourceWeightedAdmissionSealConsumed ||
		!intent.SourceWeightedAdmissionSealRequired ||
		!intent.SourceWeightedAdmissionSealReady ||
		!intent.SourceWeightedAdmissionPermitConsumed ||
		!intent.SourceWeightedAdmissionPermitRequired ||
		!intent.SourceWeightedAdmissionPermitReady ||
		!intent.SourceWeightedAdmissionAuthorityConsumed ||
		!intent.SourceWeightedAdmissionAuthorityRequired ||
		!intent.SourceManualPermitRequested ||
		!intent.SourcePermitKeyMatched ||
		!intent.BodySmokeWeighted ||
		!intent.NanoDirectRunner ||
		!intent.NanoDirectFinalGate ||
		!intent.ResonanceGraftAdmissionProof ||
		!intent.BoundaryReportFullChain ||
		intent.SourceAuthorityGranted ||
		intent.AuthorityGranted ||
		intent.ContractsReady ||
		intent.WriteAllowed ||
		intent.AdmissionAllowed ||
		intent.LiveAdmissionEnabled ||
		intent.MutatesState ||
		!intent.Passed ||
		intent.Reason != "weighted resonance intent drafted from final gate; live admission remains disabled" {
		t.Fatalf("weighted admission resonance intent lost contract: %+v", intent)
	}

	openedPath := filepath.Join(dir, "opened_final_gate.json")
	writeWeightedAdmissionFinalGateFixture(t, openedPath)
	rawOpened, err := os.ReadFile(openedPath)
	if err != nil {
		t.Fatalf("read opened final gate fixture: %v", err)
	}
	writeWeightedReadinessFixture(t, openedPath, stringsReplaceFirst(string(rawOpened), `"admission_allowed": false`, `"admission_allowed": true`))
	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionResonanceIntent([]string{openedPath, filepath.Join(dir, "opened_intent.json")}),
		"weighted admission final gate opened admission_allowed",
	)

	badSchemaPath := filepath.Join(dir, "bad_schema_final_gate.json")
	writeWeightedAdmissionFinalGateFixture(t, badSchemaPath)
	rawBadSchema, err := os.ReadFile(badSchemaPath)
	if err != nil {
		t.Fatalf("read bad schema final gate fixture: %v", err)
	}
	writeWeightedReadinessFixture(t, badSchemaPath, stringsReplaceFirst(string(rawBadSchema), `"schema": "arianna.live_route_weighted_admission_final_gate.v1"`, `"schema": "arianna.live_route_weighted_admission_final_gate.v0"`))
	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionResonanceIntent([]string{badSchemaPath, filepath.Join(dir, "bad_schema_intent.json")}),
		`weighted admission final gate schema mismatch: got "arianna.live_route_weighted_admission_final_gate.v0" want "`+admissionLiveRouteWeightedAdmissionFinalGateSchema+`"`,
	)

	notReadyPath := filepath.Join(dir, "not_ready_final_gate.json")
	writeWeightedAdmissionFinalGateFixture(t, notReadyPath)
	rawNotReady, err := os.ReadFile(notReadyPath)
	if err != nil {
		t.Fatalf("read not-ready final gate fixture: %v", err)
	}
	writeWeightedReadinessFixture(t, notReadyPath, stringsReplaceFirst(string(rawNotReady), `"weighted_admission_final_gate_ready": true`, `"weighted_admission_final_gate_ready": false`))
	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionResonanceIntent([]string{notReadyPath, filepath.Join(dir, "not_ready_intent.json")}),
		"weighted admission final gate weighted_admission_final_gate_ready not ready",
	)

	if err := runAdmissionLiveRouteWeightedAdmissionResonanceIntent([]string{finalGatePath, filepath.Join(dir, "missing", "intent.json")}); err == nil ||
		!strings.HasPrefix(err.Error(), "weighted admission resonance intent write failed:") {
		t.Fatalf("expected weighted admission resonance intent write failure, got %v", err)
	}
}
