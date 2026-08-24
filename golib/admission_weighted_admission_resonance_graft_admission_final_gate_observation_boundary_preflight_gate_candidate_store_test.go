package main

import (
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStore(t *testing.T) {
	dir := t.TempDir()

	usage := "usage: --admission-live-route-weighted-admission-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-candidate-store RESONANCE_GRAFT_ADMISSION_FINAL_GATE_OBSERVATION_BOUNDARY_PREFLIGHT_GATE_CANDIDATE_REPORT RESONANCE_GRAFT_ADMISSION_FINAL_GATE_OBSERVATION_BOUNDARY_PREFLIGHT_GATE_CANDIDATE_STORE_REPORT"
	requireBoundaryAssertError(t, runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStore(nil), usage)
	requireBoundaryAssertError(t, runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStore([]string{"candidate.json"}), usage)
	requireBoundaryAssertError(t, runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStore([]string{"candidate.json", "store.json", "extra"}), usage)
	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStore([]string{"  ", filepath.Join(dir, "store.json")}),
		"weighted admission resonance graft admission final gate observation boundary preflight gate candidate path missing",
	)

	candidatePath := filepath.Join(dir, "candidate.json")
	writeWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateFixture(t, candidatePath)
	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStore([]string{candidatePath, "  "}),
		"weighted admission resonance graft admission final gate observation boundary preflight gate candidate store output path missing",
	)

	storePath := filepath.Join(dir, "store.json")
	if err := runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStore([]string{candidatePath, storePath}); err != nil {
		t.Fatalf("valid weighted admission resonance graft admission final gate observation boundary preflight gate candidate store rejected: %v", err)
	}
	raw, err := os.ReadFile(storePath)
	if err != nil {
		t.Fatalf("read weighted admission resonance graft admission final gate observation boundary preflight gate candidate store: %v", err)
	}
	var store admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReport
	if err := json.Unmarshal(raw, &store); err != nil {
		t.Fatalf("decode weighted admission resonance graft admission final gate observation boundary preflight gate candidate store: %v", err)
	}
	sourceRaw, err := os.ReadFile(candidatePath)
	if err != nil {
		t.Fatalf("read weighted admission resonance graft admission final gate observation boundary preflight gate candidate: %v", err)
	}
	var sourceCandidate admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateReport
	if err := json.Unmarshal(sourceRaw, &sourceCandidate); err != nil {
		t.Fatalf("decode weighted admission resonance graft admission final gate observation boundary preflight gate candidate: %v", err)
	}
	if store.Schema != admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreSchema ||
		store.Status != "shadow_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_stored_dry_run" ||
		store.Target != "live_route_admission_next_step" ||
		store.TargetKind != "weighted_internal_world_shadow_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store" ||
		store.TargetMode != "append_only_read_back_store_dry_run" ||
		store.Action != "store_weighted_resonance_shadow_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_dry_run" ||
		store.LedgerState != "blocked" ||
		store.LedgerAction != "reject_blocked_admission_final_gate_observation_boundary_preflight_gate_candidate_store_ledger_append" ||
		store.LedgerContract != "none" ||
		store.LedgerEntrypoint != "none" ||
		store.LedgerReceiptShape != "none" ||
		store.LedgerWriteScope != "none" ||
		store.LedgerReady ||
		store.LedgerAppendAllowed ||
		!store.WeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReady ||
		!store.WeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateConsumed ||
		!store.WeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateRequired ||
		!store.NextStepBlockedWithoutResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStore ||
		store.ReceiptShape != "weighted_resonance_shadow_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_receipt" ||
		store.StoreKind != "shadow_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store" ||
		store.StoreMode != "append_only_read_back_store" ||
		store.StoreStage != "post_preflight_gate_candidate_pre_live_admission_store" ||
		!store.CandidateVerified ||
		!store.GateVerified ||
		!store.PreflightVerified ||
		!store.BoundaryVerified ||
		!store.ObservationVerified ||
		!store.FinalGateVerified ||
		!store.SealVerified ||
		!store.PermitVerified ||
		!store.AuthorityVerified ||
		!store.AdmissionRequired ||
		!store.ShadowOnly ||
		!store.DryRunOnly ||
		!store.LiveReady ||
		!store.RollbackRequired ||
		!store.AppendOnly ||
		!store.ReadBack ||
		!store.ReceiptPersisted ||
		!store.ReceiptVerified ||
		store.RawDreamTextAllowed ||
		store.RawDreamTextObserved ||
		store.RawDreamTextForwarded ||
		store.JanusSurfaceAllowed ||
		store.CoocLearningAllowed ||
		store.DeltaHarvestAllowed ||
		store.BodyMutationAllowed ||
		store.AuthorityGranted ||
		store.ContractsReady ||
		store.WriteAllowed ||
		store.AdmissionAllowed ||
		store.LiveAdmissionEnabled ||
		store.MutatesState ||
		store.BodyTarget != "none" ||
		!store.Passed ||
		store.SourceSchema != admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateSchema ||
		store.SourceStatus != "shadow_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_blocked_dry_run" ||
		store.SourceTarget != "live_route_admission_next_step" ||
		store.SourceReport != candidatePath ||
		store.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateID != sourceCandidate.WeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateID ||
		store.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateCausalID != sourceCandidate.CausalID ||
		store.SourceAdmissionFinalGateObservationBoundaryPreflightGateCandidateHash != sourceCandidate.AdmissionFinalGateObservationBoundaryPreflightGateCandidateHash ||
		store.SourceAdmissionFinalGateObservationBoundaryPreflightGateCandidateReadBackHash != sourceCandidate.AdmissionFinalGateObservationBoundaryPreflightGateCandidateReadBackHash ||
		store.SourceCandidateReceiptShape != sourceCandidate.ReceiptShape ||
		store.SourceCandidateState != sourceCandidate.AdmissionFinalGateObservationBoundaryPreflightGateCandidateState ||
		store.SourceCandidateAction != sourceCandidate.AdmissionFinalGateObservationBoundaryPreflightGateCandidateAction ||
		store.SourceCandidateKind != sourceCandidate.FinalGateObservationBoundaryPreflightGateCandidateKind ||
		store.SourceCandidateMode != sourceCandidate.FinalGateObservationBoundaryPreflightGateCandidateMode ||
		store.SourceCandidateStage != sourceCandidate.FinalGateObservationBoundaryPreflightGateCandidateStage ||
		!store.SourceCandidateDryRunOnly ||
		!store.SourceCandidateGateVerified ||
		!store.SourceCandidatePreflightVerified ||
		!store.SourceCandidateBoundaryVerified ||
		!store.SourceCandidateObservationVerified ||
		!store.SourceCandidateReadBackVerified ||
		store.SourceCandidateOpened ||
		store.SourceCandidateRawDreamTextAllowed ||
		store.SourceCandidateRawDreamTextObserved ||
		store.SourceCandidateRawDreamTextForwarded ||
		store.SourceCandidateBodyMutationAllowed ||
		store.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateID != sourceCandidate.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateID ||
		store.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCausalID != sourceCandidate.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCausal ||
		store.SourceAdmissionFinalGateObservationBoundaryPreflightGateHash != sourceCandidate.SourceAdmissionFinalGateObservationBoundaryPreflightGateHash ||
		store.SourceAdmissionFinalGateObservationBoundaryPreflightGateReadBackHash != sourceCandidate.SourceAdmissionFinalGateObservationBoundaryPreflightGateReadBackHash ||
		store.SourceAdmissionFinalGateObservationBoundaryPreflightGateReady ||
		!store.SourceAdmissionFinalGateObservationBoundaryPreflightGateDryRunOnly ||
		!store.SourceAdmissionFinalGateObservationBoundaryPreflightGatePreflightVerified ||
		!store.SourceAdmissionFinalGateObservationBoundaryPreflightGateBoundaryVerified ||
		!store.SourceAdmissionFinalGateObservationBoundaryPreflightGateObservationVerified ||
		!store.SourceAdmissionFinalGateObservationBoundaryPreflightGateReadBackVerified ||
		store.SourceFinalGateObservationBoundaryPreflightGateBodyMutationAllowed ||
		!store.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightReady ||
		!store.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryReady ||
		!store.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationReady ||
		!store.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateReceiverReady ||
		!store.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateIntentReady ||
		!store.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateReady ||
		!store.SourceWeightedAdmissionResonanceGraftAdmissionSealReady ||
		!store.SourceWeightedAdmissionResonanceGraftAdmissionAuthorityReady ||
		!store.SourceWeightedAdmissionResonanceGraftAdmissionPermitReady ||
		!store.SourceWeightedAdmissionResonanceGraftAdmissionReadinessReady ||
		!store.SourceWriterInventoryVerified ||
		!store.SourceWriterPreflightVerified ||
		!store.SourceAdmissionRequired ||
		!store.SourceShadowOnly ||
		!store.SourceDryRunOnly ||
		!store.SourceRequiresWriter ||
		!store.SourceRollbackRequired ||
		!store.SourceRequiresRollback ||
		!store.SourceReadOnly ||
		!store.SourceReplayOnly ||
		store.SourceLedgerReady ||
		store.SourceLedgerAppendAllowed ||
		store.SourceAuthorityGranted ||
		store.SourceContractsReady ||
		store.SourceWriteAllowed ||
		store.SourceAdmissionAllowed ||
		store.SourceLiveAdmissionEnabled ||
		store.SourceMutatesState ||
		store.SourceBodyMutationAllowed ||
		store.SourceBodyTarget != "none" ||
		store.CausalID != admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreCausalID(store) ||
		store.StoreHash != admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreHash(store) ||
		store.ReadBackHash != admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReadBackHash(store) ||
		store.StoreHash == store.ReadBackHash ||
		store.WeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreID != admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreID(store) ||
		store.Reason != "weighted resonance shadow graft admission final gate observation boundary preflight gate candidate stored without ledger append or body mutation" {
		t.Fatalf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store lost contract: %+v", store)
	}

	notReadyPath := filepath.Join(dir, "not_ready_candidate.json")
	writeWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateFixture(t, notReadyPath)
	writeWeightedReadinessFixture(t, notReadyPath, stringsReplaceFirst(readText(t, notReadyPath), `"weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_ready": true`, `"weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_ready": false`))
	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStore([]string{notReadyPath, filepath.Join(dir, "not_ready_store.json")}),
		"weighted admission resonance graft admission final gate observation boundary preflight gate candidate weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_ready not ready",
	)

	openedPath := filepath.Join(dir, "opened_candidate.json")
	writeWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateFixture(t, openedPath)
	writeWeightedReadinessFixture(t, openedPath, stringsReplaceFirst(readText(t, openedPath), `"admission_final_gate_observation_boundary_preflight_gate_candidate_ready": false`, `"admission_final_gate_observation_boundary_preflight_gate_candidate_ready": true`))
	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStore([]string{openedPath, filepath.Join(dir, "opened_store.json")}),
		"weighted admission resonance graft admission final gate observation boundary preflight gate candidate opened admission_final_gate_observation_boundary_preflight_gate_candidate_ready",
	)

	badHashPath := filepath.Join(dir, "bad_hash_candidate.json")
	writeWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateFixture(t, badHashPath)
	writeWeightedReadinessFixture(t, badHashPath, stringsReplaceFirst(readText(t, badHashPath), `"admission_final_gate_observation_boundary_preflight_gate_candidate_hash": "weighted-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-candidate-`, `"admission_final_gate_observation_boundary_preflight_gate_candidate_hash": "weighted-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-candidate-bad`))
	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStore([]string{badHashPath, filepath.Join(dir, "bad_hash_store.json")}),
		"weighted admission resonance graft admission final gate observation boundary preflight gate candidate candidate_hash mismatch",
	)

	if err := runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStore([]string{candidatePath, filepath.Join(dir, "missing", "store.json")}); err == nil ||
		!strings.HasPrefix(err.Error(), "weighted admission resonance graft admission final gate observation boundary preflight gate candidate store write failed:") {
		t.Fatalf("expected weighted admission resonance graft admission final gate observation boundary preflight gate candidate store write failure, got %v", err)
	}
}
