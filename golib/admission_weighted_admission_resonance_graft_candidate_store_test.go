package main

import (
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestAdmissionLiveRouteWeightedAdmissionResonanceGraftCandidateStore(t *testing.T) {
	dir := t.TempDir()

	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionResonanceGraftCandidateStore(nil),
		"usage: --admission-live-route-weighted-admission-resonance-graft-candidate-store RESONANCE_GRAFT_CANDIDATE_REPORT RESONANCE_GRAFT_CANDIDATE_STORE_REPORT",
	)
	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionResonanceGraftCandidateStore([]string{"candidate.json"}),
		"usage: --admission-live-route-weighted-admission-resonance-graft-candidate-store RESONANCE_GRAFT_CANDIDATE_REPORT RESONANCE_GRAFT_CANDIDATE_STORE_REPORT",
	)
	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionResonanceGraftCandidateStore([]string{"candidate.json", "store.json", "extra"}),
		"usage: --admission-live-route-weighted-admission-resonance-graft-candidate-store RESONANCE_GRAFT_CANDIDATE_REPORT RESONANCE_GRAFT_CANDIDATE_STORE_REPORT",
	)
	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionResonanceGraftCandidateStore([]string{"  ", filepath.Join(dir, "store.json")}),
		"weighted admission resonance graft candidate path missing",
	)

	candidatePath := filepath.Join(dir, "candidate.json")
	writeWeightedAdmissionResonanceGraftCandidateFixture(t, candidatePath)

	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionResonanceGraftCandidateStore([]string{candidatePath, "  "}),
		"weighted admission resonance graft candidate store output path missing",
	)

	storePath := filepath.Join(dir, "store.json")
	if err := runAdmissionLiveRouteWeightedAdmissionResonanceGraftCandidateStore([]string{candidatePath, storePath}); err != nil {
		t.Fatalf("valid weighted admission resonance graft candidate store rejected: %v", err)
	}
	raw, err := os.ReadFile(storePath)
	if err != nil {
		t.Fatalf("read weighted admission resonance graft candidate store: %v", err)
	}
	var store admissionLiveRouteWeightedAdmissionResonanceGraftCandidateStoreReport
	if err := json.Unmarshal(raw, &store); err != nil {
		t.Fatalf("decode weighted admission resonance graft candidate store: %v", err)
	}
	candidateRaw, err := os.ReadFile(candidatePath)
	if err != nil {
		t.Fatalf("read weighted admission resonance graft candidate: %v", err)
	}
	var candidate admissionLiveRouteWeightedAdmissionResonanceGraftCandidateReport
	if err := json.Unmarshal(candidateRaw, &candidate); err != nil {
		t.Fatalf("decode weighted admission resonance graft candidate: %v", err)
	}
	if store.Schema != admissionLiveRouteWeightedAdmissionResonanceGraftCandidateStoreSchema ||
		store.Status != "shadow_graft_candidate_stored_dry_run" ||
		store.Target != "resonance" ||
		store.TargetKind != "weighted_internal_world_shadow_graft_candidate_store" ||
		store.TargetMode != "append_only_read_back_store_dry_run" ||
		store.Action != "store_weighted_resonance_shadow_graft_candidate_dry_run" ||
		!store.WeightedAdmissionResonanceGraftCandidateStoreReady ||
		!store.WeightedAdmissionResonanceGraftCandidateConsumed ||
		!store.WeightedAdmissionResonanceGraftCandidateRequired ||
		!store.NextStepBlockedWithoutResonanceGraftCandidateStore ||
		store.ReceiptShape != "weighted_resonance_shadow_graft_candidate_store_receipt" ||
		store.StoreKind != "shadow_graft_candidate_store" ||
		store.StoreMode != "append_only_read_back_store" ||
		store.StoreStage != "pre_live_graft_candidate_store" ||
		store.CausalID != admissionLiveRouteWeightedAdmissionResonanceGraftCandidateStoreCausalID(store) ||
		store.StoreHash != admissionLiveRouteWeightedAdmissionResonanceGraftCandidateStoreHash(store) ||
		store.ReadBackHash != admissionLiveRouteWeightedAdmissionResonanceGraftCandidateStoreReadBackHash(store) ||
		store.StoreHash == store.ReadBackHash ||
		store.WeightedAdmissionResonanceGraftCandidateStoreID != admissionLiveRouteWeightedAdmissionResonanceGraftCandidateStoreID(store) ||
		!store.CandidateVerified ||
		!store.GateVerified ||
		!store.PreflightVerified ||
		!store.BoundaryVerified ||
		!store.ObservationVerified ||
		!store.ReceiverVerified ||
		!store.IntentVerified ||
		!store.FinalGateVerified ||
		!store.SealVerified ||
		!store.PermitVerified ||
		!store.AuthorityVerified ||
		!store.AdmissionRequired ||
		!store.ShadowOnly ||
		store.GraftAllowed ||
		!store.DryRunOnly ||
		!store.LiveReady ||
		store.RawDreamTextAllowed ||
		store.RawDreamTextObserved ||
		store.RawDreamTextForwarded ||
		store.JanusSurfaceAllowed ||
		store.CoocLearningAllowed ||
		store.DeltaHarvestAllowed ||
		store.BodyMutationAllowed ||
		!store.RollbackRequired ||
		!store.AppendOnly ||
		!store.ReadBack ||
		!store.ReceiptPersisted ||
		!store.ReceiptVerified ||
		store.SourceSchema != admissionLiveRouteWeightedAdmissionResonanceGraftCandidateSchema ||
		store.SourceStatus != "shadow_graft_candidate_ready_dry_run" ||
		store.SourceTarget != "resonance" ||
		store.SourceReport != candidatePath ||
		store.SourceWeightedAdmissionResonanceGraftCandidateID != candidate.WeightedAdmissionResonanceGraftCandidateID ||
		!store.SourceWeightedAdmissionResonanceGraftCandidateReady ||
		store.SourceWeightedAdmissionResonanceGraftCandidateCausalID != candidate.CausalID ||
		store.SourceWeightedAdmissionResonanceGraftCandidateHash != candidate.CandidateHash ||
		store.SourceWeightedAdmissionResonanceGraftCandidateReadBackHash != candidate.ReadBackHash ||
		store.SourceCandidateAction != "draft_weighted_resonance_shadow_graft_candidate_dry_run" ||
		store.SourceCandidateReceiptShape != "weighted_resonance_shadow_graft_candidate_contract" ||
		store.SourceCandidateKind != "shadow_graft_candidate" ||
		store.SourceCandidateMode != "no_mutation_candidate" ||
		store.SourceCandidateStage != "pre_live_graft_candidate" ||
		!store.SourceCandidateShadowOnly ||
		store.SourceCandidateGraftAllowed ||
		!store.SourceCandidateDryRunOnly ||
		!store.SourceCandidateLiveReady ||
		store.SourceCandidateRawDreamTextAllowed ||
		store.SourceCandidateRawDreamTextObserved ||
		store.SourceCandidateRawDreamTextForwarded ||
		store.SourceCandidateJanusSurfaceAllowed ||
		store.SourceCandidateCoocLearningAllowed ||
		store.SourceCandidateDeltaHarvestAllowed ||
		store.SourceCandidateBodyMutationAllowed ||
		!store.SourceCandidateRollbackRequired ||
		store.SourceWeightedAdmissionResonanceGraftGateID != candidate.SourceWeightedAdmissionResonanceGraftGateID ||
		!store.SourceWeightedAdmissionResonanceGraftGateReady ||
		store.SourceWeightedAdmissionResonanceGraftGateCausalID != candidate.SourceWeightedAdmissionResonanceGraftGateCausal ||
		store.SourceWeightedAdmissionResonanceGraftGateHash != candidate.SourceWeightedAdmissionResonanceGraftGateHash ||
		store.SourceWeightedAdmissionResonanceGraftGateReadBackHash != candidate.SourceWeightedAdmissionResonanceGraftGateRead ||
		store.SourceWeightedAdmissionResonanceGraftPreflightID != candidate.SourceWeightedAdmissionResonanceGraftPreflightID ||
		store.SourceWeightedAdmissionResonanceGraftBoundaryID != candidate.SourceWeightedAdmissionResonanceGraftBoundaryID ||
		store.SourceWeightedAdmissionResonanceObservationID != candidate.SourceWeightedAdmissionResonanceObservationID ||
		store.SourceWeightedAdmissionResonanceReceiverID != candidate.SourceWeightedAdmissionResonanceReceiverID ||
		!store.SourceWeightedAdmissionResonanceIntentReady ||
		!store.SourceWeightedAdmissionFinalGateReady ||
		!store.SourceWeightedAdmissionSealReady ||
		!store.SourceWeightedAdmissionPermitReady ||
		!store.SourceWeightedAdmissionAuthorityConsumed ||
		!store.SourceWeightedAdmissionAuthorityRequired ||
		!store.BodySmokeWeighted ||
		!store.NanoDirectRunner ||
		!store.NanoDirectFinalGate ||
		!store.ResonanceGraftAdmissionProof ||
		!store.BoundaryReportFullChain ||
		store.SourceAuthorityGranted ||
		store.AuthorityGranted ||
		store.ContractsReady ||
		store.WriteAllowed ||
		store.AdmissionAllowed ||
		store.LiveAdmissionEnabled ||
		store.MutatesState ||
		store.BodyTarget != "none" ||
		!store.Passed ||
		store.Reason != "weighted resonance shadow graft candidate stored without body mutation" {
		t.Fatalf("weighted admission resonance graft candidate store lost contract: %+v", store)
	}

	openedPath := filepath.Join(dir, "opened_candidate.json")
	writeWeightedAdmissionResonanceGraftCandidateFixture(t, openedPath)
	writeWeightedReadinessFixture(t, openedPath, stringsReplaceFirst(readText(t, openedPath), `"graft_allowed": false`, `"graft_allowed": true`))
	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionResonanceGraftCandidateStore([]string{openedPath, filepath.Join(dir, "opened_store.json")}),
		"weighted admission resonance graft candidate opened graft_allowed",
	)

	badSchemaPath := filepath.Join(dir, "bad_schema_candidate.json")
	writeWeightedAdmissionResonanceGraftCandidateFixture(t, badSchemaPath)
	writeWeightedReadinessFixture(t, badSchemaPath, stringsReplaceFirst(readText(t, badSchemaPath), `"schema": "arianna.live_route_weighted_admission_resonance_graft_candidate.v1"`, `"schema": "arianna.live_route_weighted_admission_resonance_graft_candidate.v0"`))
	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionResonanceGraftCandidateStore([]string{badSchemaPath, filepath.Join(dir, "bad_schema_store.json")}),
		`weighted admission resonance graft candidate schema mismatch: got "arianna.live_route_weighted_admission_resonance_graft_candidate.v0" want "`+admissionLiveRouteWeightedAdmissionResonanceGraftCandidateSchema+`"`,
	)

	badHashPath := filepath.Join(dir, "bad_hash_candidate.json")
	writeWeightedAdmissionResonanceGraftCandidateFixture(t, badHashPath)
	writeWeightedReadinessFixture(t, badHashPath, stringsReplaceFirst(readText(t, badHashPath), `"candidate_hash": "weighted-resonance-graft-candidate-`, `"candidate_hash": "weighted-resonance-graft-candidate-bad`))
	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionResonanceGraftCandidateStore([]string{badHashPath, filepath.Join(dir, "bad_hash_store.json")}),
		"weighted admission resonance graft candidate candidate_hash mismatch",
	)

	if err := runAdmissionLiveRouteWeightedAdmissionResonanceGraftCandidateStore([]string{candidatePath, filepath.Join(dir, "missing", "store.json")}); err == nil ||
		!strings.HasPrefix(err.Error(), "weighted admission resonance graft candidate store write failed:") {
		t.Fatalf("expected weighted admission resonance graft candidate store write failure, got %v", err)
	}
}
