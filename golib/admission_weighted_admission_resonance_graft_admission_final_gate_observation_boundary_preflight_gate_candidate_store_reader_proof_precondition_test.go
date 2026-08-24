package main

import (
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPrecondition(t *testing.T) {
	dir := t.TempDir()

	usage := "usage: --admission-live-route-weighted-admission-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-candidate-store-reader-proof-precondition RESONANCE_GRAFT_ADMISSION_FINAL_GATE_OBSERVATION_BOUNDARY_PREFLIGHT_GATE_CANDIDATE_STORE_READER_PROOF_REPORT RESONANCE_GRAFT_ADMISSION_FINAL_GATE_OBSERVATION_BOUNDARY_PREFLIGHT_GATE_CANDIDATE_STORE_READER_PROOF_PRECONDITION_REPORT"
	requireBoundaryAssertError(t, runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPrecondition(nil), usage)
	requireBoundaryAssertError(t, runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPrecondition([]string{"proof.json"}), usage)
	requireBoundaryAssertError(t, runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPrecondition([]string{"proof.json", "precondition.json", "extra"}), usage)
	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPrecondition([]string{"  ", filepath.Join(dir, "precondition.json")}),
		"weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof path missing",
	)

	proofPath := filepath.Join(dir, "proof.json")
	writeWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofFixture(t, proofPath)
	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPrecondition([]string{proofPath, "  "}),
		"weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition output path missing",
	)

	preconditionPath := filepath.Join(dir, "precondition.json")
	if err := runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPrecondition([]string{proofPath, preconditionPath}); err != nil {
		t.Fatalf("valid weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition rejected: %v", err)
	}
	raw, err := os.ReadFile(preconditionPath)
	if err != nil {
		t.Fatalf("read weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition: %v", err)
	}
	var precondition admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionReport
	if err := json.Unmarshal(raw, &precondition); err != nil {
		t.Fatalf("decode weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition: %v", err)
	}
	proofRaw, err := os.ReadFile(proofPath)
	if err != nil {
		t.Fatalf("read weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof: %v", err)
	}
	var proof admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofReport
	if err := json.Unmarshal(proofRaw, &proof); err != nil {
		t.Fatalf("decode weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof: %v", err)
	}
	if precondition.Schema != admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionSchema ||
		precondition.Status != "shadow_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_satisfied_dry_run" ||
		precondition.Target != "live_route_admission_next_step" ||
		precondition.TargetKind != "weighted_internal_world_shadow_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition" ||
		precondition.TargetMode != "closed_receipt_precondition_dry_run" ||
		precondition.Action != "consume_weighted_resonance_shadow_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_before_live_route_admission" ||
		precondition.LedgerState != "blocked" ||
		precondition.LedgerReady ||
		precondition.LedgerAppendAllowed ||
		!precondition.WeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionReady ||
		!precondition.WeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofConsumed ||
		!precondition.WeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofRequired ||
		!precondition.NextStepBlockedWithoutResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPrecondition ||
		precondition.ReceiptShape != "weighted_resonance_shadow_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_receipt" ||
		precondition.PreconditionKind != "shadow_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition" ||
		precondition.PreconditionMode != "closed_receipt_consumption" ||
		precondition.PreconditionStage != "post_preflight_gate_candidate_store_reader_proof_pre_live_admission_precondition" ||
		precondition.CausalID != admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionCausalID(precondition) ||
		precondition.PreconditionHash != admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionHash(precondition) ||
		precondition.ReadBackHash != admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionReadBackHash(precondition) ||
		precondition.PreconditionHash == precondition.ReadBackHash ||
		precondition.WeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionID != admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionID(precondition) ||
		!precondition.ProofVerified ||
		!precondition.ProofHashVerified ||
		!precondition.ProofReadBackVerified ||
		!precondition.StoreReaderVerified ||
		!precondition.StoreVerified ||
		!precondition.CandidateVerified ||
		!precondition.GateVerified ||
		!precondition.PreflightVerified ||
		!precondition.BoundaryVerified ||
		!precondition.ObservationVerified ||
		!precondition.ReceiverVerified ||
		!precondition.IntentVerified ||
		!precondition.FinalGateVerified ||
		!precondition.SealVerified ||
		!precondition.PermitVerified ||
		!precondition.AuthorityVerified ||
		!precondition.ReaderHashVerified ||
		!precondition.ReaderReplayVerified ||
		!precondition.ReaderReadBackVerified ||
		!precondition.StoreHashVerified ||
		!precondition.StoreReadBackVerified ||
		!precondition.AdmissionRequired ||
		!precondition.ShadowOnly ||
		precondition.GraftAllowed ||
		!precondition.DryRunOnly ||
		!precondition.LiveReady ||
		precondition.RawDreamTextAllowed ||
		precondition.RawDreamTextObserved ||
		precondition.RawDreamTextForwarded ||
		precondition.JanusSurfaceAllowed ||
		precondition.CoocLearningAllowed ||
		precondition.DeltaHarvestAllowed ||
		precondition.BodyMutationAllowed ||
		!precondition.RollbackRequired ||
		!precondition.ReadOnly ||
		!precondition.ReplayOnly ||
		precondition.AuthorityGranted ||
		precondition.ContractsReady ||
		precondition.WriteAllowed ||
		precondition.AdmissionAllowed ||
		precondition.LiveAdmissionEnabled ||
		precondition.MutatesState ||
		precondition.BodyTarget != "none" ||
		!precondition.Passed ||
		precondition.Reason != "weighted resonance shadow graft admission final gate observation boundary preflight gate candidate store reader proof consumed as closed precondition" ||
		precondition.SourceSchema != admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofSchema ||
		precondition.SourceStatus != "shadow_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_ready_dry_run" ||
		precondition.SourceTarget != "live_route_admission_next_step" ||
		precondition.SourceReport != proofPath ||
		precondition.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofID != proof.WeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofID ||
		!precondition.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofReady ||
		precondition.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofCausalID != proof.CausalID ||
		precondition.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofHash != proof.ProofHash ||
		precondition.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofReadBack != proof.ReadBackHash ||
		precondition.SourceProofAction != "prove_weighted_resonance_shadow_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_dry_run" ||
		precondition.SourceProofReceiptShape != "weighted_resonance_shadow_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_receipt" ||
		precondition.SourceProofKind != "shadow_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof" ||
		precondition.SourceProofMode != "closed_read_back_reader_proof" ||
		precondition.SourceProofStage != "post_preflight_gate_candidate_store_reader_pre_live_admission_proof" ||
		precondition.SourceProofLedgerReady ||
		precondition.SourceProofLedgerAppendAllowed ||
		!precondition.SourceProofAdmissionRequired ||
		!precondition.SourceProofShadowOnly ||
		precondition.SourceProofGraftAllowed ||
		!precondition.SourceProofDryRunOnly ||
		!precondition.SourceProofLiveReady ||
		precondition.SourceProofRawDreamTextAllowed ||
		precondition.SourceProofRawDreamTextObserved ||
		precondition.SourceProofRawDreamTextForwarded ||
		precondition.SourceProofJanusSurfaceAllowed ||
		precondition.SourceProofCoocLearningAllowed ||
		precondition.SourceProofDeltaHarvestAllowed ||
		precondition.SourceProofBodyMutationAllowed ||
		!precondition.SourceProofRollbackRequired ||
		!precondition.SourceProofReadOnly ||
		!precondition.SourceProofReplayOnly ||
		precondition.SourceProofAuthorityGranted ||
		precondition.SourceProofContractsReady ||
		precondition.SourceProofWriteAllowed ||
		precondition.SourceProofAdmissionAllowed ||
		precondition.SourceProofLiveAdmissionEnabled ||
		precondition.SourceProofMutatesState ||
		precondition.SourceProofBodyTarget != "none" ||
		!precondition.SourceProofPassed ||
		precondition.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderID != proof.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderID ||
		precondition.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreID != proof.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreID ||
		precondition.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateID != proof.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateID ||
		precondition.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateID != proof.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateID ||
		precondition.SourceAdmissionFinalGateObservationBoundaryPreflightGateReady ||
		!precondition.SourceWriterInventoryVerified ||
		!precondition.SourceWriterPreflightVerified ||
		!precondition.SourceAdmissionRequired ||
		!precondition.SourceShadowOnly ||
		!precondition.SourceDryRunOnly ||
		!precondition.SourceRequiresWriter ||
		!precondition.SourceRollbackRequired ||
		!precondition.SourceRequiresRollback ||
		!precondition.SourceReadOnly ||
		!precondition.SourceReplayOnly {
		t.Fatalf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition lost contract: %+v", precondition)
	}

	badSchemaPath := filepath.Join(dir, "bad_schema_proof.json")
	writeWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofFixture(t, badSchemaPath)
	writeWeightedReadinessFixture(t, badSchemaPath, stringsReplaceFirst(readText(t, badSchemaPath), `"schema": "arianna.live_route_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof.v1"`, `"schema": "arianna.live_route_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof.v0"`))
	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPrecondition([]string{badSchemaPath, filepath.Join(dir, "bad_schema_precondition.json")}),
		`weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof schema mismatch: got "arianna.live_route_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof.v0" want "`+admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofSchema+`"`,
	)

	openedProofPath := filepath.Join(dir, "opened_proof.json")
	writeWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofFixture(t, openedProofPath)
	writeWeightedReadinessFixture(t, openedProofPath, stringsReplaceFirst(readText(t, openedProofPath), `"live_admission_enabled": false`, `"live_admission_enabled": true`))
	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPrecondition([]string{openedProofPath, filepath.Join(dir, "opened_precondition.json")}),
		"weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof opened live_admission_enabled",
	)

	badHashPath := filepath.Join(dir, "bad_hash_proof.json")
	writeWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofFixture(t, badHashPath)
	writeWeightedReadinessFixture(t, badHashPath, stringsReplaceFirst(readText(t, badHashPath), `"proof_hash": "weighted-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-candidate-store-reader-proof-`, `"proof_hash": "weighted-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-candidate-store-reader-proof-bad`))
	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPrecondition([]string{badHashPath, filepath.Join(dir, "bad_hash_precondition.json")}),
		"weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof proof_hash mismatch",
	)

	if err := runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPrecondition([]string{proofPath, filepath.Join(dir, "missing", "precondition.json")}); err == nil ||
		!strings.HasPrefix(err.Error(), "weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition write failed:") {
		t.Fatalf("expected weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition write failure, got %v", err)
	}
}
