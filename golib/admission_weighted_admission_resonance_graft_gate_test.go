package main

import (
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestAdmissionLiveRouteWeightedAdmissionResonanceGraftGate(t *testing.T) {
	dir := t.TempDir()

	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionResonanceGraftGate(nil),
		"usage: --admission-live-route-weighted-admission-resonance-graft-gate RESONANCE_GRAFT_PREFLIGHT_REPORT RESONANCE_GRAFT_GATE_REPORT",
	)
	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionResonanceGraftGate([]string{"preflight.json"}),
		"usage: --admission-live-route-weighted-admission-resonance-graft-gate RESONANCE_GRAFT_PREFLIGHT_REPORT RESONANCE_GRAFT_GATE_REPORT",
	)
	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionResonanceGraftGate([]string{"preflight.json", "gate.json", "extra"}),
		"usage: --admission-live-route-weighted-admission-resonance-graft-gate RESONANCE_GRAFT_PREFLIGHT_REPORT RESONANCE_GRAFT_GATE_REPORT",
	)
	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionResonanceGraftGate([]string{"  ", filepath.Join(dir, "gate.json")}),
		"weighted admission resonance graft preflight path missing",
	)

	preflightPath := filepath.Join(dir, "preflight.json")
	writeWeightedAdmissionResonanceGraftPreflightFixture(t, preflightPath)

	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionResonanceGraftGate([]string{preflightPath, "  "}),
		"weighted admission resonance graft gate output path missing",
	)

	gatePath := filepath.Join(dir, "gate.json")
	if err := runAdmissionLiveRouteWeightedAdmissionResonanceGraftGate([]string{preflightPath, gatePath}); err != nil {
		t.Fatalf("valid weighted admission resonance graft gate rejected: %v", err)
	}
	raw, err := os.ReadFile(gatePath)
	if err != nil {
		t.Fatalf("read weighted admission resonance graft gate: %v", err)
	}
	var gate admissionLiveRouteWeightedAdmissionResonanceGraftGateReport
	if err := json.Unmarshal(raw, &gate); err != nil {
		t.Fatalf("decode weighted admission resonance graft gate: %v", err)
	}
	if gate.Schema != admissionLiveRouteWeightedAdmissionResonanceGraftGateSchema ||
		gate.Status != "shadow_graft_gate_ready_dry_run" ||
		gate.Target != "resonance" ||
		gate.TargetKind != "weighted_internal_world_shadow_graft_gate" ||
		gate.TargetMode != "receipt_only_closed_gate_dry_run" ||
		gate.Action != "gate_weighted_resonance_shadow_graft_dry_run" ||
		!gate.WeightedAdmissionResonanceGraftGateReady ||
		!gate.WeightedAdmissionResonanceGraftPreflightConsumed ||
		!gate.WeightedAdmissionResonanceGraftPreflightRequired ||
		!gate.NextStepBlockedWithoutResonanceGraftGate ||
		gate.WeightedAdmissionResonanceGraftGateID == "" ||
		gate.WeightedAdmissionResonanceGraftGateID != admissionLiveRouteWeightedAdmissionResonanceGraftGateID(gate) ||
		gate.ReceiptShape != "weighted_resonance_shadow_graft_gate_contract" ||
		gate.GateKind != "shadow_graft_gate" ||
		gate.GateMode != "no_mutation_gate" ||
		gate.GateStage != "pre_live_graft_gate" ||
		gate.CausalID == "" ||
		gate.CausalID != admissionLiveRouteWeightedAdmissionResonanceGraftGateCausalID(gate) ||
		gate.GateHash == "" ||
		gate.GateHash != admissionLiveRouteWeightedAdmissionResonanceGraftGateHash(gate) ||
		gate.ReadBackHash == "" ||
		gate.ReadBackHash != admissionLiveRouteWeightedAdmissionResonanceGraftGateReadBackHash(gate) ||
		gate.GateHash == gate.ReadBackHash ||
		!gate.PreflightVerified ||
		!gate.BoundaryVerified ||
		!gate.ObservationVerified ||
		!gate.ReceiverVerified ||
		!gate.IntentVerified ||
		!gate.FinalGateVerified ||
		!gate.SealVerified ||
		!gate.PermitVerified ||
		!gate.AuthorityVerified ||
		!gate.AdmissionRequired ||
		!gate.ShadowOnly ||
		gate.GraftAllowed ||
		!gate.DryRunOnly ||
		!gate.LiveReady ||
		gate.RawDreamTextAllowed ||
		gate.RawDreamTextObserved ||
		gate.RawDreamTextForwarded ||
		gate.JanusSurfaceAllowed ||
		gate.CoocLearningAllowed ||
		gate.DeltaHarvestAllowed ||
		gate.BodyMutationAllowed ||
		!gate.RollbackRequired ||
		gate.SourceSchema != admissionLiveRouteWeightedAdmissionResonanceGraftPreflightSchema ||
		gate.SourceStatus != "shadow_graft_preflight_ready_dry_run" ||
		gate.SourceTarget != "resonance" ||
		gate.SourceReport != preflightPath ||
		!strings.HasPrefix(gate.SourceWeightedAdmissionResonanceGraftPreflightID, "weighted-resonance-graft-preflight-id-") ||
		!gate.SourceWeightedAdmissionResonanceGraftPreflightReady ||
		!strings.HasPrefix(gate.SourceWeightedAdmissionResonanceGraftPreflightCausal, "weighted-resonance-graft-preflight-causal-") ||
		!strings.HasPrefix(gate.SourceWeightedAdmissionResonanceGraftPreflightHash, "weighted-resonance-graft-preflight-") ||
		!strings.HasPrefix(gate.SourceWeightedAdmissionResonanceGraftPreflightRead, "weighted-resonance-graft-preflight-read-") ||
		gate.SourceWeightedAdmissionResonanceGraftPreflightHash == gate.SourceWeightedAdmissionResonanceGraftPreflightRead ||
		gate.SourcePreflightAction != "prepare_weighted_resonance_shadow_graft_preflight_dry_run" ||
		gate.SourcePreflightReceiptShape != "weighted_resonance_shadow_graft_preflight_contract" ||
		gate.SourcePreflightKind != "shadow_graft_preflight" ||
		gate.SourcePreflightMode != "no_mutation_preflight" ||
		gate.SourcePreflightStage != "pre_live_graft_admission" ||
		!gate.SourcePreflightShadowOnly ||
		gate.SourcePreflightGraftAllowed ||
		!gate.SourcePreflightDryRunOnly ||
		!gate.SourcePreflightLiveReady ||
		gate.SourcePreflightRawDreamTextAllowed ||
		gate.SourcePreflightRawDreamTextObserved ||
		gate.SourcePreflightRawDreamTextForwarded ||
		gate.SourcePreflightJanusSurfaceAllowed ||
		gate.SourcePreflightCoocLearningAllowed ||
		gate.SourcePreflightDeltaHarvestAllowed ||
		gate.SourcePreflightBodyMutationAllowed ||
		!gate.SourcePreflightRollbackRequired ||
		gate.SourceGraftBoundarySchema != admissionLiveRouteWeightedAdmissionResonanceGraftBoundarySchema ||
		gate.SourceGraftBoundaryStatus != "shadow_graft_boundary_declared_dry_run" ||
		gate.SourceGraftBoundaryTarget != "resonance" ||
		gate.SourceGraftBoundaryReport == "" ||
		!strings.HasPrefix(gate.SourceWeightedAdmissionResonanceGraftBoundaryID, "weighted-resonance-graft-boundary-id-") ||
		!gate.SourceWeightedAdmissionResonanceGraftBoundaryReady ||
		!strings.HasPrefix(gate.SourceWeightedAdmissionResonanceGraftBoundaryCausal, "weighted-resonance-graft-boundary-causal-") ||
		!strings.HasPrefix(gate.SourceWeightedAdmissionResonanceGraftBoundaryHash, "weighted-resonance-graft-boundary-") ||
		!strings.HasPrefix(gate.SourceWeightedAdmissionResonanceGraftBoundaryRead, "weighted-resonance-graft-boundary-read-") ||
		gate.SourceBoundaryAction != "declare_weighted_resonance_shadow_graft_boundary_dry_run" ||
		gate.SourceBoundaryReceiptShape != "weighted_resonance_observation_shadow_graft_boundary" ||
		gate.SourceBoundaryKind != "shadow_graft_boundary" ||
		gate.SourceBoundaryMode != "no_mutation_receipt" ||
		gate.SourceBoundaryStage != "pre_live_graft" ||
		!gate.SourceBoundaryShadowOnly ||
		gate.SourceBoundaryGraftAllowed ||
		!gate.SourceBoundaryDryRunOnly ||
		!gate.SourceBoundaryLiveReady ||
		gate.SourceBoundaryRawDreamTextAllowed ||
		gate.SourceBoundaryRawDreamTextObserved ||
		gate.SourceBoundaryRawDreamTextForwarded ||
		gate.SourceBoundaryJanusSurfaceAllowed ||
		gate.SourceBoundaryCoocLearningAllowed ||
		gate.SourceBoundaryDeltaHarvestAllowed ||
		gate.SourceBoundaryBodyMutationAllowed ||
		!gate.SourceBoundaryRollbackRequired ||
		gate.SourceObservationSchema != admissionLiveRouteWeightedAdmissionResonanceObservationSchema ||
		gate.SourceObservationStatus != "observation_recorded_dry_run" ||
		gate.SourceObservationTarget != "resonance" ||
		gate.SourceObservationReport == "" ||
		!strings.HasPrefix(gate.SourceWeightedAdmissionResonanceObservationID, "weighted-resonance-observation-") ||
		!gate.SourceWeightedAdmissionResonanceObservationReady ||
		!strings.HasPrefix(gate.SourceWeightedAdmissionResonanceObservationCausal, "weighted-resonance-observation-causal-") ||
		!strings.HasPrefix(gate.SourceWeightedAdmissionResonanceObservationAppend, "weighted-resonance-observation-append-") ||
		!strings.HasPrefix(gate.SourceWeightedAdmissionResonanceObservationRead, "weighted-resonance-observation-read-") ||
		gate.SourceObserver != "resonance" ||
		gate.SourceObserverKind != "internal_world" ||
		gate.SourceObservationKind != "weighted_receiver_state_proof" ||
		gate.SourceObservationMode != "sealed_metadata_observation" ||
		!gate.SourceAppendOnly ||
		!gate.SourceReadBack ||
		!gate.SourceReceiptVerified ||
		!gate.SourceDryRunOnly ||
		gate.SourceObservationRawDreamTextObserved ||
		gate.SourceObservationRawDreamTextForwarded ||
		gate.SourceObservationJanusSurfaceAllowed ||
		gate.SourceObservationCoocLearningAllowed ||
		gate.SourceObservationDeltaHarvestAllowed ||
		gate.SourceObservationBodyMutationAllowed ||
		!gate.SourceObservationRollbackRequired ||
		gate.SourceResonanceReceiverReport == "" ||
		gate.SourceResonanceIntentReport == "" ||
		gate.SourceFinalGateReport == "" ||
		gate.SourceSealReport == "" ||
		gate.SourcePermitReport == "" ||
		gate.SourceAuthorityReport == "" ||
		gate.SourceContractReport == "" ||
		gate.SourcePreconditionReport == "" ||
		gate.SourceReadinessReport == "" ||
		gate.SourceBodyWorkdir == "" ||
		gate.SourceBoundaryReport == "" ||
		gate.SourceProofLog == "" ||
		gate.SourceFinalGateLog == "" ||
		!strings.HasPrefix(gate.SourceWeightedAdmissionResonanceReceiverID, "weighted-resonance-receiver-") ||
		!gate.SourceWeightedAdmissionResonanceReceiverReady ||
		!strings.HasPrefix(gate.SourceWeightedAdmissionResonanceReceiverCausal, "weighted-resonance-receiver-causal-") ||
		!strings.HasPrefix(gate.SourceReceiverPreStateHash, "weighted-resonance-receiver-pre-") ||
		!strings.HasPrefix(gate.SourceReceiverPostStateHash, "weighted-resonance-receiver-post-") ||
		!strings.HasPrefix(gate.SourceReceiverStateDeltaHash, "weighted-resonance-receiver-delta-") ||
		gate.SourceReceiverPreStateHash == gate.SourceReceiverPostStateHash ||
		!gate.SourceWeightedAdmissionResonanceIntentConsumed ||
		!gate.SourceWeightedAdmissionResonanceIntentRequired ||
		!gate.SourceWeightedAdmissionResonanceIntentReady ||
		!gate.SourceWeightedAdmissionFinalGateConsumed ||
		!gate.SourceWeightedAdmissionFinalGateRequired ||
		!gate.SourceWeightedAdmissionFinalGateReady ||
		!gate.SourceWeightedAdmissionSealConsumed ||
		!gate.SourceWeightedAdmissionSealRequired ||
		!gate.SourceWeightedAdmissionSealReady ||
		!gate.SourceWeightedAdmissionPermitConsumed ||
		!gate.SourceWeightedAdmissionPermitRequired ||
		!gate.SourceWeightedAdmissionPermitReady ||
		!gate.SourceWeightedAdmissionAuthorityConsumed ||
		!gate.SourceWeightedAdmissionAuthorityRequired ||
		!gate.SourceManualPermitRequested ||
		!gate.SourcePermitKeyMatched ||
		gate.SourceRawDreamTextAllowed ||
		gate.SourceRawDreamTextObserved ||
		gate.SourceRawDreamTextForwarded ||
		gate.SourceJanusSurfaceAllowed ||
		gate.SourceCoocLearningAllowed ||
		gate.SourceDeltaHarvestAllowed ||
		gate.SourceBodyMutationAllowed ||
		!gate.SourceRollbackRequired ||
		!gate.SourcePreStateHashRequired ||
		!gate.SourcePostStateHashRequired ||
		!gate.BodySmokeWeighted ||
		!gate.NanoDirectRunner ||
		!gate.NanoDirectFinalGate ||
		!gate.ResonanceGraftAdmissionProof ||
		!gate.BoundaryReportFullChain ||
		gate.SourceAuthorityGranted ||
		gate.AuthorityGranted ||
		gate.ContractsReady ||
		gate.WriteAllowed ||
		gate.AdmissionAllowed ||
		gate.LiveAdmissionEnabled ||
		gate.MutatesState ||
		gate.BodyTarget != "none" ||
		!gate.Passed ||
		gate.Reason != "weighted resonance shadow graft gate prepared without body mutation" {
		t.Fatalf("weighted admission resonance graft gate lost contract: %+v", gate)
	}

	openedPath := filepath.Join(dir, "opened_preflight.json")
	writeWeightedAdmissionResonanceGraftPreflightFixture(t, openedPath)
	writeWeightedReadinessFixture(t, openedPath, stringsReplaceFirst(readText(t, openedPath), `"graft_allowed": false`, `"graft_allowed": true`))
	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionResonanceGraftGate([]string{openedPath, filepath.Join(dir, "opened_gate.json")}),
		"weighted admission resonance graft preflight opened graft_allowed",
	)

	badSchemaPath := filepath.Join(dir, "bad_schema_preflight.json")
	writeWeightedAdmissionResonanceGraftPreflightFixture(t, badSchemaPath)
	writeWeightedReadinessFixture(t, badSchemaPath, stringsReplaceFirst(readText(t, badSchemaPath), `"schema": "arianna.live_route_weighted_admission_resonance_graft_preflight.v1"`, `"schema": "arianna.live_route_weighted_admission_resonance_graft_preflight.v0"`))
	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionResonanceGraftGate([]string{badSchemaPath, filepath.Join(dir, "bad_schema_gate.json")}),
		`weighted admission resonance graft preflight schema mismatch: got "arianna.live_route_weighted_admission_resonance_graft_preflight.v0" want "`+admissionLiveRouteWeightedAdmissionResonanceGraftPreflightSchema+`"`,
	)

	notReadyPath := filepath.Join(dir, "not_ready_preflight.json")
	writeWeightedAdmissionResonanceGraftPreflightFixture(t, notReadyPath)
	writeWeightedReadinessFixture(t, notReadyPath, stringsReplaceFirst(readText(t, notReadyPath), `"weighted_admission_resonance_graft_preflight_ready": true`, `"weighted_admission_resonance_graft_preflight_ready": false`))
	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionResonanceGraftGate([]string{notReadyPath, filepath.Join(dir, "not_ready_gate.json")}),
		"weighted admission resonance graft preflight weighted_admission_resonance_graft_preflight_ready not ready",
	)

	badHashPath := filepath.Join(dir, "bad_hash_preflight.json")
	writeWeightedAdmissionResonanceGraftPreflightFixture(t, badHashPath)
	writeWeightedReadinessFixture(t, badHashPath, stringsReplaceFirst(readText(t, badHashPath), `"preflight_hash": "weighted-resonance-graft-preflight-`, `"preflight_hash": "weighted-resonance-graft-preflight-bad`))
	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionResonanceGraftGate([]string{badHashPath, filepath.Join(dir, "bad_hash_gate.json")}),
		"weighted admission resonance graft preflight preflight_hash mismatch",
	)

	if err := runAdmissionLiveRouteWeightedAdmissionResonanceGraftGate([]string{preflightPath, filepath.Join(dir, "missing", "gate.json")}); err == nil ||
		!strings.HasPrefix(err.Error(), "weighted admission resonance graft gate write failed:") {
		t.Fatalf("expected weighted admission resonance graft gate write failure, got %v", err)
	}
}
