package main

import (
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestAdmissionLiveRouteWeightedAdmissionResonanceGraftBoundary(t *testing.T) {
	dir := t.TempDir()

	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionResonanceGraftBoundary(nil),
		"usage: --admission-live-route-weighted-admission-resonance-graft-boundary RESONANCE_OBSERVATION_REPORT RESONANCE_GRAFT_BOUNDARY_REPORT",
	)
	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionResonanceGraftBoundary([]string{"observation.json"}),
		"usage: --admission-live-route-weighted-admission-resonance-graft-boundary RESONANCE_OBSERVATION_REPORT RESONANCE_GRAFT_BOUNDARY_REPORT",
	)
	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionResonanceGraftBoundary([]string{"observation.json", "boundary.json", "extra"}),
		"usage: --admission-live-route-weighted-admission-resonance-graft-boundary RESONANCE_OBSERVATION_REPORT RESONANCE_GRAFT_BOUNDARY_REPORT",
	)
	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionResonanceGraftBoundary([]string{"  ", filepath.Join(dir, "boundary.json")}),
		"weighted admission resonance observation path missing",
	)

	observationPath := filepath.Join(dir, "observation.json")
	writeWeightedAdmissionResonanceObservationFixture(t, observationPath)

	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionResonanceGraftBoundary([]string{observationPath, "  "}),
		"weighted admission resonance graft boundary output path missing",
	)

	boundaryPath := filepath.Join(dir, "boundary.json")
	if err := runAdmissionLiveRouteWeightedAdmissionResonanceGraftBoundary([]string{observationPath, boundaryPath}); err != nil {
		t.Fatalf("valid weighted admission resonance graft boundary rejected: %v", err)
	}
	raw, err := os.ReadFile(boundaryPath)
	if err != nil {
		t.Fatalf("read weighted admission resonance graft boundary: %v", err)
	}
	var boundary admissionLiveRouteWeightedAdmissionResonanceGraftBoundaryReport
	if err := json.Unmarshal(raw, &boundary); err != nil {
		t.Fatalf("decode weighted admission resonance graft boundary: %v", err)
	}
	if boundary.Schema != admissionLiveRouteWeightedAdmissionResonanceGraftBoundarySchema ||
		boundary.Status != "shadow_graft_boundary_declared_dry_run" ||
		boundary.Target != "resonance" ||
		boundary.TargetKind != "weighted_internal_world_shadow_graft" ||
		boundary.TargetMode != "receipt_only_closed_dry_run" ||
		boundary.Action != "declare_weighted_resonance_shadow_graft_boundary_dry_run" ||
		!boundary.WeightedAdmissionResonanceGraftBoundaryReady ||
		!boundary.WeightedAdmissionResonanceObservationConsumed ||
		!boundary.WeightedAdmissionResonanceObservationRequired ||
		!boundary.NextStepBlockedWithoutResonanceGraftBoundary ||
		boundary.WeightedAdmissionResonanceGraftBoundaryID == "" ||
		boundary.WeightedAdmissionResonanceGraftBoundaryID != admissionLiveRouteWeightedAdmissionResonanceGraftBoundaryID(boundary) ||
		boundary.ReceiptShape != "weighted_resonance_observation_shadow_graft_boundary" ||
		boundary.BoundaryKind != "shadow_graft_boundary" ||
		boundary.BoundaryMode != "no_mutation_receipt" ||
		boundary.BoundaryStage != "pre_live_graft" ||
		boundary.CausalID == "" ||
		boundary.CausalID != admissionLiveRouteWeightedAdmissionResonanceGraftBoundaryCausalID(boundary) ||
		boundary.BoundaryHash == "" ||
		boundary.BoundaryHash != admissionLiveRouteWeightedAdmissionResonanceGraftBoundaryHash(boundary) ||
		boundary.ReadBackHash == "" ||
		boundary.ReadBackHash != admissionLiveRouteWeightedAdmissionResonanceGraftBoundaryReadBackHash(boundary) ||
		boundary.BoundaryHash == boundary.ReadBackHash ||
		!boundary.ShadowOnly ||
		boundary.GraftAllowed ||
		!boundary.DryRunOnly ||
		!boundary.LiveReady ||
		boundary.RawDreamTextAllowed ||
		boundary.RawDreamTextObserved ||
		boundary.RawDreamTextForwarded ||
		boundary.JanusSurfaceAllowed ||
		boundary.CoocLearningAllowed ||
		boundary.DeltaHarvestAllowed ||
		boundary.BodyMutationAllowed ||
		!boundary.RollbackRequired ||
		boundary.SourceSchema != admissionLiveRouteWeightedAdmissionResonanceObservationSchema ||
		boundary.SourceStatus != "observation_recorded_dry_run" ||
		boundary.SourceTarget != "resonance" ||
		boundary.SourceReport != observationPath ||
		!strings.HasPrefix(boundary.SourceWeightedAdmissionResonanceObservationID, "weighted-resonance-observation-") ||
		!boundary.SourceWeightedAdmissionResonanceObservationReady ||
		!strings.HasPrefix(boundary.SourceWeightedAdmissionResonanceObservationCausal, "weighted-resonance-observation-causal-") ||
		!strings.HasPrefix(boundary.SourceWeightedAdmissionResonanceObservationAppend, "weighted-resonance-observation-append-") ||
		!strings.HasPrefix(boundary.SourceWeightedAdmissionResonanceObservationRead, "weighted-resonance-observation-read-") ||
		boundary.SourceObserver != "resonance" ||
		boundary.SourceObserverKind != "internal_world" ||
		boundary.SourceObservationKind != "weighted_receiver_state_proof" ||
		boundary.SourceObservationMode != "sealed_metadata_observation" ||
		!boundary.SourceAppendOnly ||
		!boundary.SourceReadBack ||
		!boundary.SourceReceiptVerified ||
		!boundary.SourceDryRunOnly ||
		boundary.SourceObservationRawDreamTextObserved ||
		boundary.SourceObservationRawDreamTextForwarded ||
		boundary.SourceObservationJanusSurfaceAllowed ||
		boundary.SourceObservationCoocLearningAllowed ||
		boundary.SourceObservationDeltaHarvestAllowed ||
		boundary.SourceObservationBodyMutationAllowed ||
		!boundary.SourceObservationRollbackRequired ||
		boundary.SourceResonanceReceiverReport == "" ||
		boundary.SourceResonanceIntentReport == "" ||
		boundary.SourceFinalGateReport == "" ||
		boundary.SourceSealReport == "" ||
		boundary.SourcePermitReport == "" ||
		boundary.SourceAuthorityReport == "" ||
		boundary.SourceContractReport == "" ||
		boundary.SourcePreconditionReport == "" ||
		boundary.SourceReadinessReport == "" ||
		boundary.SourceBodyWorkdir == "" ||
		boundary.SourceBoundaryReport == "" ||
		boundary.SourceProofLog == "" ||
		boundary.SourceFinalGateLog == "" ||
		!strings.HasPrefix(boundary.SourceWeightedAdmissionResonanceReceiverID, "weighted-resonance-receiver-") ||
		!boundary.SourceWeightedAdmissionResonanceReceiverReady ||
		!strings.HasPrefix(boundary.SourceWeightedAdmissionResonanceReceiverCausal, "weighted-resonance-receiver-causal-") ||
		!strings.HasPrefix(boundary.SourceReceiverPreStateHash, "weighted-resonance-receiver-pre-") ||
		!strings.HasPrefix(boundary.SourceReceiverPostStateHash, "weighted-resonance-receiver-post-") ||
		!strings.HasPrefix(boundary.SourceReceiverStateDeltaHash, "weighted-resonance-receiver-delta-") ||
		boundary.SourceReceiverPreStateHash == boundary.SourceReceiverPostStateHash ||
		!boundary.SourceWeightedAdmissionResonanceIntentConsumed ||
		!boundary.SourceWeightedAdmissionResonanceIntentRequired ||
		!boundary.SourceWeightedAdmissionResonanceIntentReady ||
		!boundary.SourceWeightedAdmissionFinalGateConsumed ||
		!boundary.SourceWeightedAdmissionFinalGateRequired ||
		!boundary.SourceWeightedAdmissionFinalGateReady ||
		!boundary.SourceWeightedAdmissionSealConsumed ||
		!boundary.SourceWeightedAdmissionSealRequired ||
		!boundary.SourceWeightedAdmissionSealReady ||
		!boundary.SourceWeightedAdmissionPermitConsumed ||
		!boundary.SourceWeightedAdmissionPermitRequired ||
		!boundary.SourceWeightedAdmissionPermitReady ||
		!boundary.SourceWeightedAdmissionAuthorityConsumed ||
		!boundary.SourceWeightedAdmissionAuthorityRequired ||
		!boundary.SourceManualPermitRequested ||
		!boundary.SourcePermitKeyMatched ||
		boundary.SourceRawDreamTextAllowed ||
		boundary.SourceRawDreamTextObserved ||
		boundary.SourceRawDreamTextForwarded ||
		boundary.SourceJanusSurfaceAllowed ||
		boundary.SourceCoocLearningAllowed ||
		boundary.SourceDeltaHarvestAllowed ||
		boundary.SourceBodyMutationAllowed ||
		!boundary.SourceRollbackRequired ||
		!boundary.SourcePreStateHashRequired ||
		!boundary.SourcePostStateHashRequired ||
		!boundary.BodySmokeWeighted ||
		!boundary.NanoDirectRunner ||
		!boundary.NanoDirectFinalGate ||
		!boundary.ResonanceGraftAdmissionProof ||
		!boundary.BoundaryReportFullChain ||
		boundary.SourceAuthorityGranted ||
		boundary.AuthorityGranted ||
		boundary.ContractsReady ||
		boundary.WriteAllowed ||
		boundary.AdmissionAllowed ||
		boundary.LiveAdmissionEnabled ||
		boundary.MutatesState ||
		boundary.BodyTarget != "none" ||
		!boundary.Passed ||
		boundary.Reason != "weighted resonance shadow graft boundary declared without body mutation" {
		t.Fatalf("weighted admission resonance graft boundary lost contract: %+v", boundary)
	}

	openedPath := filepath.Join(dir, "opened_observation.json")
	writeWeightedAdmissionResonanceObservationFixture(t, openedPath)
	writeWeightedReadinessFixture(t, openedPath, stringsReplaceFirst(readText(t, openedPath), `"live_admission_enabled": false`, `"live_admission_enabled": true`))
	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionResonanceGraftBoundary([]string{openedPath, filepath.Join(dir, "opened_boundary.json")}),
		"weighted admission resonance observation opened live_admission_enabled",
	)

	badSchemaPath := filepath.Join(dir, "bad_schema_observation.json")
	writeWeightedAdmissionResonanceObservationFixture(t, badSchemaPath)
	writeWeightedReadinessFixture(t, badSchemaPath, stringsReplaceFirst(readText(t, badSchemaPath), `"schema": "arianna.live_route_weighted_admission_resonance_observation.v1"`, `"schema": "arianna.live_route_weighted_admission_resonance_observation.v0"`))
	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionResonanceGraftBoundary([]string{badSchemaPath, filepath.Join(dir, "bad_schema_boundary.json")}),
		`weighted admission resonance observation schema mismatch: got "arianna.live_route_weighted_admission_resonance_observation.v0" want "`+admissionLiveRouteWeightedAdmissionResonanceObservationSchema+`"`,
	)

	notReadyPath := filepath.Join(dir, "not_ready_observation.json")
	writeWeightedAdmissionResonanceObservationFixture(t, notReadyPath)
	writeWeightedReadinessFixture(t, notReadyPath, stringsReplaceFirst(readText(t, notReadyPath), `"weighted_admission_resonance_observation_ready": true`, `"weighted_admission_resonance_observation_ready": false`))
	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionResonanceGraftBoundary([]string{notReadyPath, filepath.Join(dir, "not_ready_boundary.json")}),
		"weighted admission resonance observation weighted_admission_resonance_observation_ready not ready",
	)

	badHashPath := filepath.Join(dir, "bad_hash_observation.json")
	writeWeightedAdmissionResonanceObservationFixture(t, badHashPath)
	writeWeightedReadinessFixture(t, badHashPath, stringsReplaceFirst(readText(t, badHashPath), `"append_hash": "weighted-resonance-observation-append-`, `"append_hash": "weighted-resonance-observation-append-bad`))
	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionResonanceGraftBoundary([]string{badHashPath, filepath.Join(dir, "bad_hash_boundary.json")}),
		"weighted admission resonance observation append_hash mismatch",
	)

	if err := runAdmissionLiveRouteWeightedAdmissionResonanceGraftBoundary([]string{observationPath, filepath.Join(dir, "missing", "boundary.json")}); err == nil ||
		!strings.HasPrefix(err.Error(), "weighted admission resonance graft boundary write failed:") {
		t.Fatalf("expected weighted admission resonance graft boundary write failure, got %v", err)
	}
}
