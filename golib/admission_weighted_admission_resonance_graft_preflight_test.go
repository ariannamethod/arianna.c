package main

import (
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestAdmissionLiveRouteWeightedAdmissionResonanceGraftPreflight(t *testing.T) {
	dir := t.TempDir()

	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionResonanceGraftPreflight(nil),
		"usage: --admission-live-route-weighted-admission-resonance-graft-preflight RESONANCE_GRAFT_BOUNDARY_REPORT RESONANCE_GRAFT_PREFLIGHT_REPORT",
	)
	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionResonanceGraftPreflight([]string{"boundary.json"}),
		"usage: --admission-live-route-weighted-admission-resonance-graft-preflight RESONANCE_GRAFT_BOUNDARY_REPORT RESONANCE_GRAFT_PREFLIGHT_REPORT",
	)
	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionResonanceGraftPreflight([]string{"boundary.json", "preflight.json", "extra"}),
		"usage: --admission-live-route-weighted-admission-resonance-graft-preflight RESONANCE_GRAFT_BOUNDARY_REPORT RESONANCE_GRAFT_PREFLIGHT_REPORT",
	)
	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionResonanceGraftPreflight([]string{"  ", filepath.Join(dir, "preflight.json")}),
		"weighted admission resonance graft boundary path missing",
	)

	boundaryPath := filepath.Join(dir, "boundary.json")
	writeWeightedAdmissionResonanceGraftBoundaryFixture(t, boundaryPath)

	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionResonanceGraftPreflight([]string{boundaryPath, "  "}),
		"weighted admission resonance graft preflight output path missing",
	)

	preflightPath := filepath.Join(dir, "preflight.json")
	if err := runAdmissionLiveRouteWeightedAdmissionResonanceGraftPreflight([]string{boundaryPath, preflightPath}); err != nil {
		t.Fatalf("valid weighted admission resonance graft preflight rejected: %v", err)
	}
	raw, err := os.ReadFile(preflightPath)
	if err != nil {
		t.Fatalf("read weighted admission resonance graft preflight: %v", err)
	}
	var preflight admissionLiveRouteWeightedAdmissionResonanceGraftPreflightReport
	if err := json.Unmarshal(raw, &preflight); err != nil {
		t.Fatalf("decode weighted admission resonance graft preflight: %v", err)
	}
	if preflight.Schema != admissionLiveRouteWeightedAdmissionResonanceGraftPreflightSchema ||
		preflight.Status != "shadow_graft_preflight_ready_dry_run" ||
		preflight.Target != "resonance" ||
		preflight.TargetKind != "weighted_internal_world_shadow_graft_preflight" ||
		preflight.TargetMode != "receipt_only_closed_preflight_dry_run" ||
		preflight.Action != "prepare_weighted_resonance_shadow_graft_preflight_dry_run" ||
		!preflight.WeightedAdmissionResonanceGraftPreflightReady ||
		!preflight.WeightedAdmissionResonanceGraftBoundaryConsumed ||
		!preflight.WeightedAdmissionResonanceGraftBoundaryRequired ||
		!preflight.NextStepBlockedWithoutResonanceGraftPreflight ||
		preflight.WeightedAdmissionResonanceGraftPreflightID == "" ||
		preflight.WeightedAdmissionResonanceGraftPreflightID != admissionLiveRouteWeightedAdmissionResonanceGraftPreflightID(preflight) ||
		preflight.ReceiptShape != "weighted_resonance_shadow_graft_preflight_contract" ||
		preflight.PreflightKind != "shadow_graft_preflight" ||
		preflight.PreflightMode != "no_mutation_preflight" ||
		preflight.PreflightStage != "pre_live_graft_admission" ||
		preflight.CausalID == "" ||
		preflight.CausalID != admissionLiveRouteWeightedAdmissionResonanceGraftPreflightCausalID(preflight) ||
		preflight.PreflightHash == "" ||
		preflight.PreflightHash != admissionLiveRouteWeightedAdmissionResonanceGraftPreflightHash(preflight) ||
		preflight.ReadBackHash == "" ||
		preflight.ReadBackHash != admissionLiveRouteWeightedAdmissionResonanceGraftPreflightReadBackHash(preflight) ||
		preflight.PreflightHash == preflight.ReadBackHash ||
		!preflight.BoundaryVerified ||
		!preflight.ObservationVerified ||
		!preflight.ReceiverVerified ||
		!preflight.IntentVerified ||
		!preflight.FinalGateVerified ||
		!preflight.SealVerified ||
		!preflight.PermitVerified ||
		!preflight.AuthorityVerified ||
		!preflight.AdmissionRequired ||
		!preflight.ShadowOnly ||
		preflight.GraftAllowed ||
		!preflight.DryRunOnly ||
		!preflight.LiveReady ||
		preflight.RawDreamTextAllowed ||
		preflight.RawDreamTextObserved ||
		preflight.RawDreamTextForwarded ||
		preflight.JanusSurfaceAllowed ||
		preflight.CoocLearningAllowed ||
		preflight.DeltaHarvestAllowed ||
		preflight.BodyMutationAllowed ||
		!preflight.RollbackRequired ||
		preflight.SourceSchema != admissionLiveRouteWeightedAdmissionResonanceGraftBoundarySchema ||
		preflight.SourceStatus != "shadow_graft_boundary_declared_dry_run" ||
		preflight.SourceTarget != "resonance" ||
		preflight.SourceReport != boundaryPath ||
		!strings.HasPrefix(preflight.SourceWeightedAdmissionResonanceGraftBoundaryID, "weighted-resonance-graft-boundary-id-") ||
		!preflight.SourceWeightedAdmissionResonanceGraftBoundaryReady ||
		!strings.HasPrefix(preflight.SourceWeightedAdmissionResonanceGraftBoundaryCausal, "weighted-resonance-graft-boundary-causal-") ||
		!strings.HasPrefix(preflight.SourceWeightedAdmissionResonanceGraftBoundaryHash, "weighted-resonance-graft-boundary-") ||
		!strings.HasPrefix(preflight.SourceWeightedAdmissionResonanceGraftBoundaryRead, "weighted-resonance-graft-boundary-read-") ||
		preflight.SourceBoundaryAction != "declare_weighted_resonance_shadow_graft_boundary_dry_run" ||
		preflight.SourceBoundaryReceiptShape != "weighted_resonance_observation_shadow_graft_boundary" ||
		preflight.SourceBoundaryKind != "shadow_graft_boundary" ||
		preflight.SourceBoundaryMode != "no_mutation_receipt" ||
		preflight.SourceBoundaryStage != "pre_live_graft" ||
		!preflight.SourceBoundaryShadowOnly ||
		preflight.SourceBoundaryGraftAllowed ||
		!preflight.SourceBoundaryDryRunOnly ||
		!preflight.SourceBoundaryLiveReady ||
		preflight.SourceBoundaryRawDreamTextAllowed ||
		preflight.SourceBoundaryRawDreamTextObserved ||
		preflight.SourceBoundaryRawDreamTextForwarded ||
		preflight.SourceBoundaryJanusSurfaceAllowed ||
		preflight.SourceBoundaryCoocLearningAllowed ||
		preflight.SourceBoundaryDeltaHarvestAllowed ||
		preflight.SourceBoundaryBodyMutationAllowed ||
		!preflight.SourceBoundaryRollbackRequired ||
		preflight.SourceObservationSchema != admissionLiveRouteWeightedAdmissionResonanceObservationSchema ||
		preflight.SourceObservationStatus != "observation_recorded_dry_run" ||
		preflight.SourceObservationTarget != "resonance" ||
		preflight.SourceObservationReport == "" ||
		!strings.HasPrefix(preflight.SourceWeightedAdmissionResonanceObservationID, "weighted-resonance-observation-") ||
		!preflight.SourceWeightedAdmissionResonanceObservationReady ||
		!strings.HasPrefix(preflight.SourceWeightedAdmissionResonanceObservationCausal, "weighted-resonance-observation-causal-") ||
		!strings.HasPrefix(preflight.SourceWeightedAdmissionResonanceObservationAppend, "weighted-resonance-observation-append-") ||
		!strings.HasPrefix(preflight.SourceWeightedAdmissionResonanceObservationRead, "weighted-resonance-observation-read-") ||
		preflight.SourceObserver != "resonance" ||
		preflight.SourceObserverKind != "internal_world" ||
		preflight.SourceObservationKind != "weighted_receiver_state_proof" ||
		preflight.SourceObservationMode != "sealed_metadata_observation" ||
		!preflight.SourceAppendOnly ||
		!preflight.SourceReadBack ||
		!preflight.SourceReceiptVerified ||
		!preflight.SourceDryRunOnly ||
		preflight.SourceObservationRawDreamTextObserved ||
		preflight.SourceObservationRawDreamTextForwarded ||
		preflight.SourceObservationJanusSurfaceAllowed ||
		preflight.SourceObservationCoocLearningAllowed ||
		preflight.SourceObservationDeltaHarvestAllowed ||
		preflight.SourceObservationBodyMutationAllowed ||
		!preflight.SourceObservationRollbackRequired ||
		preflight.SourceResonanceReceiverReport == "" ||
		preflight.SourceResonanceIntentReport == "" ||
		preflight.SourceFinalGateReport == "" ||
		preflight.SourceSealReport == "" ||
		preflight.SourcePermitReport == "" ||
		preflight.SourceAuthorityReport == "" ||
		preflight.SourceContractReport == "" ||
		preflight.SourcePreconditionReport == "" ||
		preflight.SourceReadinessReport == "" ||
		preflight.SourceBodyWorkdir == "" ||
		preflight.SourceBoundaryReport == "" ||
		preflight.SourceProofLog == "" ||
		preflight.SourceFinalGateLog == "" ||
		!strings.HasPrefix(preflight.SourceWeightedAdmissionResonanceReceiverID, "weighted-resonance-receiver-") ||
		!preflight.SourceWeightedAdmissionResonanceReceiverReady ||
		!strings.HasPrefix(preflight.SourceWeightedAdmissionResonanceReceiverCausal, "weighted-resonance-receiver-causal-") ||
		!strings.HasPrefix(preflight.SourceReceiverPreStateHash, "weighted-resonance-receiver-pre-") ||
		!strings.HasPrefix(preflight.SourceReceiverPostStateHash, "weighted-resonance-receiver-post-") ||
		!strings.HasPrefix(preflight.SourceReceiverStateDeltaHash, "weighted-resonance-receiver-delta-") ||
		preflight.SourceReceiverPreStateHash == preflight.SourceReceiverPostStateHash ||
		!preflight.SourceWeightedAdmissionResonanceIntentConsumed ||
		!preflight.SourceWeightedAdmissionResonanceIntentRequired ||
		!preflight.SourceWeightedAdmissionResonanceIntentReady ||
		!preflight.SourceWeightedAdmissionFinalGateConsumed ||
		!preflight.SourceWeightedAdmissionFinalGateRequired ||
		!preflight.SourceWeightedAdmissionFinalGateReady ||
		!preflight.SourceWeightedAdmissionSealConsumed ||
		!preflight.SourceWeightedAdmissionSealRequired ||
		!preflight.SourceWeightedAdmissionSealReady ||
		!preflight.SourceWeightedAdmissionPermitConsumed ||
		!preflight.SourceWeightedAdmissionPermitRequired ||
		!preflight.SourceWeightedAdmissionPermitReady ||
		!preflight.SourceWeightedAdmissionAuthorityConsumed ||
		!preflight.SourceWeightedAdmissionAuthorityRequired ||
		!preflight.SourceManualPermitRequested ||
		!preflight.SourcePermitKeyMatched ||
		preflight.SourceRawDreamTextAllowed ||
		preflight.SourceRawDreamTextObserved ||
		preflight.SourceRawDreamTextForwarded ||
		preflight.SourceJanusSurfaceAllowed ||
		preflight.SourceCoocLearningAllowed ||
		preflight.SourceDeltaHarvestAllowed ||
		preflight.SourceBodyMutationAllowed ||
		!preflight.SourceRollbackRequired ||
		!preflight.SourcePreStateHashRequired ||
		!preflight.SourcePostStateHashRequired ||
		!preflight.BodySmokeWeighted ||
		!preflight.NanoDirectRunner ||
		!preflight.NanoDirectFinalGate ||
		!preflight.ResonanceGraftAdmissionProof ||
		!preflight.BoundaryReportFullChain ||
		preflight.SourceAuthorityGranted ||
		preflight.AuthorityGranted ||
		preflight.ContractsReady ||
		preflight.WriteAllowed ||
		preflight.AdmissionAllowed ||
		preflight.LiveAdmissionEnabled ||
		preflight.MutatesState ||
		preflight.BodyTarget != "none" ||
		!preflight.Passed ||
		preflight.Reason != "weighted resonance shadow graft preflight prepared without body mutation" {
		t.Fatalf("weighted admission resonance graft preflight lost contract: %+v", preflight)
	}

	openedPath := filepath.Join(dir, "opened_boundary.json")
	writeWeightedAdmissionResonanceGraftBoundaryFixture(t, openedPath)
	writeWeightedReadinessFixture(t, openedPath, stringsReplaceFirst(readText(t, openedPath), `"graft_allowed": false`, `"graft_allowed": true`))
	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionResonanceGraftPreflight([]string{openedPath, filepath.Join(dir, "opened_preflight.json")}),
		"weighted admission resonance graft boundary opened graft_allowed",
	)

	badSchemaPath := filepath.Join(dir, "bad_schema_boundary.json")
	writeWeightedAdmissionResonanceGraftBoundaryFixture(t, badSchemaPath)
	writeWeightedReadinessFixture(t, badSchemaPath, stringsReplaceFirst(readText(t, badSchemaPath), `"schema": "arianna.live_route_weighted_admission_resonance_graft_boundary.v1"`, `"schema": "arianna.live_route_weighted_admission_resonance_graft_boundary.v0"`))
	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionResonanceGraftPreflight([]string{badSchemaPath, filepath.Join(dir, "bad_schema_preflight.json")}),
		`weighted admission resonance graft boundary schema mismatch: got "arianna.live_route_weighted_admission_resonance_graft_boundary.v0" want "`+admissionLiveRouteWeightedAdmissionResonanceGraftBoundarySchema+`"`,
	)

	notReadyPath := filepath.Join(dir, "not_ready_boundary.json")
	writeWeightedAdmissionResonanceGraftBoundaryFixture(t, notReadyPath)
	writeWeightedReadinessFixture(t, notReadyPath, stringsReplaceFirst(readText(t, notReadyPath), `"weighted_admission_resonance_graft_boundary_ready": true`, `"weighted_admission_resonance_graft_boundary_ready": false`))
	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionResonanceGraftPreflight([]string{notReadyPath, filepath.Join(dir, "not_ready_preflight.json")}),
		"weighted admission resonance graft boundary weighted_admission_resonance_graft_boundary_ready not ready",
	)

	badHashPath := filepath.Join(dir, "bad_hash_boundary.json")
	writeWeightedAdmissionResonanceGraftBoundaryFixture(t, badHashPath)
	writeWeightedReadinessFixture(t, badHashPath, stringsReplaceFirst(readText(t, badHashPath), `"boundary_hash": "weighted-resonance-graft-boundary-`, `"boundary_hash": "weighted-resonance-graft-boundary-bad`))
	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionResonanceGraftPreflight([]string{badHashPath, filepath.Join(dir, "bad_hash_preflight.json")}),
		"weighted admission resonance graft boundary boundary_hash mismatch",
	)

	if err := runAdmissionLiveRouteWeightedAdmissionResonanceGraftPreflight([]string{boundaryPath, filepath.Join(dir, "missing", "preflight.json")}); err == nil ||
		!strings.HasPrefix(err.Error(), "weighted admission resonance graft preflight write failed:") {
		t.Fatalf("expected weighted admission resonance graft preflight write failure, got %v", err)
	}
}
