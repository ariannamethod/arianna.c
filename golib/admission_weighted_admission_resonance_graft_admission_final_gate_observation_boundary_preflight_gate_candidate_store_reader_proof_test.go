package main

import (
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProof(t *testing.T) {
	dir := t.TempDir()

	usage := "usage: --admission-live-route-weighted-admission-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-candidate-store-reader-proof RESONANCE_GRAFT_ADMISSION_FINAL_GATE_OBSERVATION_BOUNDARY_PREFLIGHT_GATE_CANDIDATE_STORE_READER_REPORT RESONANCE_GRAFT_ADMISSION_FINAL_GATE_OBSERVATION_BOUNDARY_PREFLIGHT_GATE_CANDIDATE_STORE_READER_PROOF_REPORT"
	requireBoundaryAssertError(t, runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProof(nil), usage)
	requireBoundaryAssertError(t, runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProof([]string{"reader.json"}), usage)
	requireBoundaryAssertError(t, runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProof([]string{"reader.json", "proof.json", "extra"}), usage)
	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProof([]string{"  ", filepath.Join(dir, "proof.json")}),
		"weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader path missing",
	)

	readerPath := filepath.Join(dir, "reader.json")
	writeWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderFixture(t, readerPath)
	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProof([]string{readerPath, "  "}),
		"weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof output path missing",
	)

	proofPath := filepath.Join(dir, "proof.json")
	if err := runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProof([]string{readerPath, proofPath}); err != nil {
		t.Fatalf("valid weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof rejected: %v", err)
	}
	raw, err := os.ReadFile(proofPath)
	if err != nil {
		t.Fatalf("read weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof: %v", err)
	}
	var proof admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofReport
	if err := json.Unmarshal(raw, &proof); err != nil {
		t.Fatalf("decode weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof: %v", err)
	}
	readerRaw, err := os.ReadFile(readerPath)
	if err != nil {
		t.Fatalf("read weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader: %v", err)
	}
	var reader admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderReport
	if err := json.Unmarshal(readerRaw, &reader); err != nil {
		t.Fatalf("decode weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader: %v", err)
	}
	if proof.Schema != admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofSchema ||
		proof.Status != "shadow_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_ready_dry_run" ||
		proof.Target != "live_route_admission_next_step" ||
		proof.TargetKind != "weighted_internal_world_shadow_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof" ||
		proof.TargetMode != "receipt_only_closed_reader_proof_dry_run" ||
		proof.Action != "prove_weighted_resonance_shadow_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_dry_run" ||
		proof.LedgerReady ||
		proof.LedgerAppendAllowed ||
		!proof.WeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofReady ||
		!proof.WeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderConsumed ||
		!proof.WeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderRequired ||
		!proof.NextStepBlockedWithoutResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProof ||
		proof.ReceiptShape != "weighted_resonance_shadow_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_receipt" ||
		proof.ProofKind != "shadow_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof" ||
		proof.ProofMode != "closed_read_back_reader_proof" ||
		proof.ProofStage != "post_preflight_gate_candidate_store_reader_pre_live_admission_proof" ||
		proof.CausalID != admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofCausalID(proof) ||
		proof.ProofHash != admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofHash(proof) ||
		proof.ReadBackHash != admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofReadBackHash(proof) ||
		proof.ProofHash == proof.ReadBackHash ||
		proof.WeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofID != admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofID(proof) ||
		!proof.StoreReaderVerified ||
		!proof.StoreVerified ||
		!proof.CandidateVerified ||
		!proof.GateVerified ||
		!proof.PreflightVerified ||
		!proof.BoundaryVerified ||
		!proof.ObservationVerified ||
		!proof.ReceiverVerified ||
		!proof.IntentVerified ||
		!proof.FinalGateVerified ||
		!proof.SealVerified ||
		!proof.PermitVerified ||
		!proof.AuthorityVerified ||
		!proof.ReaderHashVerified ||
		!proof.ReaderReplayVerified ||
		!proof.ReaderReadBackVerified ||
		!proof.StoreHashVerified ||
		!proof.StoreReadBackVerified ||
		!proof.AdmissionRequired ||
		!proof.ShadowOnly ||
		proof.GraftAllowed ||
		!proof.DryRunOnly ||
		!proof.LiveReady ||
		proof.RawDreamTextAllowed ||
		proof.RawDreamTextObserved ||
		proof.RawDreamTextForwarded ||
		proof.BodyMutationAllowed ||
		!proof.RollbackRequired ||
		!proof.ReadOnly ||
		!proof.ReplayOnly ||
		proof.AuthorityGranted ||
		proof.ContractsReady ||
		proof.WriteAllowed ||
		proof.AdmissionAllowed ||
		proof.LiveAdmissionEnabled ||
		proof.MutatesState ||
		proof.BodyTarget != "none" ||
		!proof.Passed ||
		proof.SourceSchema != admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderSchema ||
		proof.SourceStatus != "shadow_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_read_back_dry_run" ||
		proof.SourceTarget != "live_route_admission_next_step" ||
		proof.SourceReport != readerPath ||
		proof.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderID != reader.WeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderID ||
		proof.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderCausalID != reader.CausalID ||
		proof.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderHash != reader.ReaderHash ||
		proof.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderReplayHash != reader.ReplayHash ||
		proof.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderReadBackHash != reader.ReadBackHash ||
		proof.SourceReaderAction != reader.Action ||
		proof.SourceReaderReceiptShape != reader.ReceiptShape ||
		proof.SourceReaderKind != reader.ReaderKind ||
		proof.SourceReaderMode != reader.ReaderMode ||
		proof.SourceReaderStage != reader.ReaderStage ||
		proof.SourceReaderLedgerReady ||
		proof.SourceReaderLedgerAppendAllowed ||
		proof.SourceReaderBodyMutationAllowed ||
		proof.SourceReaderAuthorityGranted ||
		proof.SourceReaderWriteAllowed ||
		proof.SourceReaderLiveAdmissionEnabled ||
		proof.SourceReaderMutatesState ||
		proof.SourceReaderBodyTarget != "none" ||
		!proof.SourceReaderPassed ||
		proof.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreID != reader.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreID ||
		proof.SourceAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreHash != reader.SourceAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreHash ||
		proof.SourceAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReadBackHash != reader.SourceAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReadBackHash ||
		proof.SourceStoreLedgerAppendAllowed ||
		proof.SourceStoreBodyMutationAllowed ||
		proof.SourceStoreBodyTarget != "none" ||
		proof.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateID != reader.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateID ||
		proof.SourceAdmissionFinalGateObservationBoundaryPreflightGateCandidateHash != reader.SourceAdmissionFinalGateObservationBoundaryPreflightGateCandidateHash ||
		proof.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateID != reader.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateID ||
		proof.SourceAdmissionFinalGateObservationBoundaryPreflightGateReady ||
		!proof.SourceAdmissionFinalGateObservationBoundaryPreflightGateDryRunOnly ||
		proof.SourceCandidateOpened ||
		proof.SourceCandidateBodyMutationAllowed ||
		!proof.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightReady ||
		!proof.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryReady ||
		!proof.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationReady ||
		!proof.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateReceiverReady ||
		!proof.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateReady ||
		!proof.SourceWriterInventoryVerified ||
		!proof.SourceWriterPreflightVerified ||
		!proof.SourceAdmissionRequired ||
		!proof.SourceShadowOnly ||
		!proof.SourceDryRunOnly ||
		!proof.SourceReadOnly ||
		!proof.SourceReplayOnly ||
		proof.Reason != "weighted resonance shadow graft admission final gate observation boundary preflight gate candidate store reader proof sealed without ledger append or body mutation" {
		t.Fatalf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof lost contract: %+v", proof)
	}

	openedPath := filepath.Join(dir, "opened_reader.json")
	writeWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderFixture(t, openedPath)
	writeWeightedReadinessFixture(t, openedPath, stringsReplaceFirst(readText(t, openedPath), `"ledger_append_allowed": false`, `"ledger_append_allowed": true`))
	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProof([]string{openedPath, filepath.Join(dir, "opened_proof.json")}),
		"weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader opened ledger_append_allowed",
	)

	badSchemaPath := filepath.Join(dir, "bad_schema_reader.json")
	writeWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderFixture(t, badSchemaPath)
	writeWeightedReadinessFixture(t, badSchemaPath, stringsReplaceFirst(readText(t, badSchemaPath), `"schema": "arianna.live_route_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader.v1"`, `"schema": "arianna.live_route_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader.v0"`))
	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProof([]string{badSchemaPath, filepath.Join(dir, "bad_schema_proof.json")}),
		`weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader schema mismatch: got "arianna.live_route_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader.v0" want "`+admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderSchema+`"`,
	)

	badHashPath := filepath.Join(dir, "bad_hash_reader.json")
	writeWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderFixture(t, badHashPath)
	writeWeightedReadinessFixture(t, badHashPath, stringsReplaceFirst(readText(t, badHashPath), `"reader_hash": "weighted-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-candidate-store-reader-`, `"reader_hash": "weighted-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-candidate-store-reader-bad`))
	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProof([]string{badHashPath, filepath.Join(dir, "bad_hash_proof.json")}),
		"weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader reader_hash mismatch",
	)

	if err := runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProof([]string{readerPath, filepath.Join(dir, "missing", "proof.json")}); err == nil ||
		!strings.HasPrefix(err.Error(), "weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof write failed:") {
		t.Fatalf("expected weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof write failure, got %v", err)
	}
}
