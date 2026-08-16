package main

import (
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionProof(t *testing.T) {
	dir := t.TempDir()

	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionProof(nil),
		"usage: --admission-live-route-weighted-admission-resonance-graft-admission-proof RESONANCE_GRAFT_CANDIDATE_STORE_READER_REPORT RESONANCE_GRAFT_ADMISSION_PROOF_REPORT",
	)
	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionProof([]string{"reader.json"}),
		"usage: --admission-live-route-weighted-admission-resonance-graft-admission-proof RESONANCE_GRAFT_CANDIDATE_STORE_READER_REPORT RESONANCE_GRAFT_ADMISSION_PROOF_REPORT",
	)
	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionProof([]string{"reader.json", "proof.json", "extra"}),
		"usage: --admission-live-route-weighted-admission-resonance-graft-admission-proof RESONANCE_GRAFT_CANDIDATE_STORE_READER_REPORT RESONANCE_GRAFT_ADMISSION_PROOF_REPORT",
	)
	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionProof([]string{"  ", filepath.Join(dir, "proof.json")}),
		"weighted admission resonance graft candidate store reader path missing",
	)

	readerPath := filepath.Join(dir, "reader.json")
	writeWeightedAdmissionResonanceGraftCandidateStoreReaderFixture(t, readerPath)

	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionProof([]string{readerPath, "  "}),
		"weighted admission resonance graft admission proof output path missing",
	)

	proofPath := filepath.Join(dir, "proof.json")
	if err := runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionProof([]string{readerPath, proofPath}); err != nil {
		t.Fatalf("valid weighted admission resonance graft admission proof rejected: %v", err)
	}
	raw, err := os.ReadFile(proofPath)
	if err != nil {
		t.Fatalf("read weighted admission resonance graft admission proof: %v", err)
	}
	var proof admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionProofReport
	if err := json.Unmarshal(raw, &proof); err != nil {
		t.Fatalf("decode weighted admission resonance graft admission proof: %v", err)
	}
	readerRaw, err := os.ReadFile(readerPath)
	if err != nil {
		t.Fatalf("read weighted admission resonance graft candidate store reader: %v", err)
	}
	var reader admissionLiveRouteWeightedAdmissionResonanceGraftCandidateStoreReaderReport
	if err := json.Unmarshal(readerRaw, &reader); err != nil {
		t.Fatalf("decode weighted admission resonance graft candidate store reader: %v", err)
	}
	if proof.Schema != admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionProofSchema ||
		proof.Status != "shadow_graft_admission_proof_ready_dry_run" ||
		proof.Target != "resonance" ||
		proof.TargetKind != "weighted_internal_world_shadow_graft_admission_proof" ||
		proof.TargetMode != "receipt_only_closed_admission_proof_dry_run" ||
		proof.Action != "prove_weighted_resonance_shadow_graft_admission_dry_run" ||
		!proof.WeightedAdmissionResonanceGraftAdmissionProofReady ||
		!proof.WeightedAdmissionResonanceGraftCandidateStoreReaderConsumed ||
		!proof.WeightedAdmissionResonanceGraftCandidateStoreReaderRequired ||
		!proof.NextStepBlockedWithoutResonanceGraftAdmissionProof ||
		proof.ReceiptShape != "weighted_resonance_shadow_graft_admission_proof_receipt" ||
		proof.ProofKind != "shadow_graft_admission_proof" ||
		proof.ProofMode != "closed_read_back_admission_proof" ||
		proof.ProofStage != "pre_live_graft_admission_proof" ||
		proof.CausalID != admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionProofCausalID(proof) ||
		proof.ProofHash != admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionProofHash(proof) ||
		proof.ReadBackHash != admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionProofReadBackHash(proof) ||
		proof.ProofHash == proof.ReadBackHash ||
		proof.WeightedAdmissionResonanceGraftAdmissionProofID != admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionProofID(proof) ||
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
		!proof.AdmissionRequired ||
		!proof.ShadowOnly ||
		proof.GraftAllowed ||
		!proof.DryRunOnly ||
		!proof.LiveReady ||
		proof.RawDreamTextAllowed ||
		proof.RawDreamTextObserved ||
		proof.RawDreamTextForwarded ||
		proof.JanusSurfaceAllowed ||
		proof.CoocLearningAllowed ||
		proof.DeltaHarvestAllowed ||
		proof.BodyMutationAllowed ||
		!proof.RollbackRequired ||
		!proof.ReadOnly ||
		!proof.ReplayOnly ||
		proof.SourceSchema != admissionLiveRouteWeightedAdmissionResonanceGraftCandidateStoreReaderSchema ||
		proof.SourceStatus != "shadow_graft_candidate_store_read_back_dry_run" ||
		proof.SourceTarget != "resonance" ||
		proof.SourceReport != readerPath ||
		proof.SourceWeightedAdmissionResonanceGraftCandidateStoreReaderID != reader.WeightedAdmissionResonanceGraftCandidateStoreReaderID ||
		!proof.SourceWeightedAdmissionResonanceGraftCandidateStoreReaderReady ||
		proof.SourceWeightedAdmissionResonanceGraftCandidateStoreReaderCausalID != reader.CausalID ||
		proof.SourceWeightedAdmissionResonanceGraftCandidateStoreReaderHash != reader.ReaderHash ||
		proof.SourceWeightedAdmissionResonanceGraftCandidateStoreReaderReplayHash != reader.ReplayHash ||
		proof.SourceWeightedAdmissionResonanceGraftCandidateStoreReaderReadBack != reader.ReadBackHash ||
		proof.SourceReaderAction != "read_weighted_resonance_shadow_graft_candidate_store_dry_run" ||
		proof.SourceReaderReceiptShape != "weighted_resonance_shadow_graft_candidate_store_reader_receipt" ||
		proof.SourceReaderKind != "shadow_graft_candidate_store_reader" ||
		proof.SourceReaderMode != "read_only_replay" ||
		proof.SourceReaderStage != "pre_live_graft_candidate_store_reader" ||
		!proof.SourceReaderReadOnly ||
		!proof.SourceReaderReplayOnly ||
		!proof.SourceReaderStoreVerified ||
		!proof.SourceReaderCandidateVerified ||
		!proof.SourceReaderHashVerified ||
		!proof.SourceReaderReplayVerified ||
		!proof.SourceReaderReadBackVerified ||
		!proof.SourceReaderAdmissionRequired ||
		!proof.SourceReaderShadowOnly ||
		proof.SourceReaderGraftAllowed ||
		!proof.SourceReaderDryRunOnly ||
		!proof.SourceReaderLiveReady ||
		proof.SourceReaderRawDreamTextAllowed ||
		proof.SourceReaderRawDreamTextObserved ||
		proof.SourceReaderRawDreamTextForwarded ||
		proof.SourceReaderJanusSurfaceAllowed ||
		proof.SourceReaderCoocLearningAllowed ||
		proof.SourceReaderDeltaHarvestAllowed ||
		proof.SourceReaderBodyMutationAllowed ||
		!proof.SourceReaderRollbackRequired ||
		proof.SourceReaderAuthorityGranted ||
		proof.SourceReaderContractsReady ||
		proof.SourceReaderWriteAllowed ||
		proof.SourceReaderAdmissionAllowed ||
		proof.SourceReaderLiveAdmissionEnabled ||
		proof.SourceReaderMutatesState ||
		proof.SourceReaderBodyTarget != "none" ||
		!proof.SourceReaderPassed ||
		proof.SourceWeightedAdmissionResonanceGraftCandidateStoreID != reader.SourceWeightedAdmissionResonanceGraftCandidateStoreID ||
		proof.SourceWeightedAdmissionResonanceGraftCandidateStoreHash != reader.SourceWeightedAdmissionResonanceGraftCandidateStoreHash ||
		proof.SourceWeightedAdmissionResonanceGraftCandidateStoreReadBackHash != reader.SourceWeightedAdmissionResonanceGraftCandidateStoreReadBackHash ||
		proof.SourceStoreAction != "store_weighted_resonance_shadow_graft_candidate_dry_run" ||
		proof.SourceStoreReceiptShape != "weighted_resonance_shadow_graft_candidate_store_receipt" ||
		proof.SourceStoreKind != "shadow_graft_candidate_store" ||
		proof.SourceStoreMode != "append_only_read_back_store" ||
		proof.SourceStoreStage != "pre_live_graft_candidate_store" ||
		!proof.SourceStoreAppendOnly ||
		!proof.SourceStoreReadBack ||
		!proof.SourceStoreReceiptPersisted ||
		!proof.SourceStoreReceiptVerified ||
		proof.SourceStoreGraftAllowed ||
		proof.SourceStoreRawDreamTextAllowed ||
		proof.SourceStoreJanusSurfaceAllowed ||
		proof.SourceStoreCoocLearningAllowed ||
		proof.SourceStoreDeltaHarvestAllowed ||
		proof.SourceStoreBodyMutationAllowed ||
		proof.SourceWeightedAdmissionResonanceGraftCandidateID != reader.SourceWeightedAdmissionResonanceGraftCandidateID ||
		proof.SourceWeightedAdmissionResonanceGraftCandidateHash != reader.SourceWeightedAdmissionResonanceGraftCandidateHash ||
		proof.SourceWeightedAdmissionResonanceGraftGateID != reader.SourceWeightedAdmissionResonanceGraftGateID ||
		proof.SourceWeightedAdmissionResonanceGraftPreflightID != reader.SourceWeightedAdmissionResonanceGraftPreflightID ||
		proof.SourceWeightedAdmissionResonanceGraftBoundaryID != reader.SourceWeightedAdmissionResonanceGraftBoundaryID ||
		proof.SourceWeightedAdmissionResonanceObservationID != reader.SourceWeightedAdmissionResonanceObservationID ||
		proof.SourceWeightedAdmissionResonanceReceiverID != reader.SourceWeightedAdmissionResonanceReceiverID ||
		!proof.BodySmokeWeighted ||
		!proof.NanoDirectRunner ||
		!proof.NanoDirectFinalGate ||
		!proof.ResonanceGraftAdmissionProof ||
		!proof.BoundaryReportFullChain ||
		proof.SourceAuthorityGranted ||
		proof.AuthorityGranted ||
		proof.ContractsReady ||
		proof.WriteAllowed ||
		proof.AdmissionAllowed ||
		proof.LiveAdmissionEnabled ||
		proof.MutatesState ||
		proof.BodyTarget != "none" ||
		!proof.Passed ||
		proof.Reason != "weighted resonance shadow graft admission proof sealed without body mutation" {
		t.Fatalf("weighted admission resonance graft admission proof lost contract: %+v", proof)
	}

	openedPath := filepath.Join(dir, "opened_reader.json")
	writeWeightedAdmissionResonanceGraftCandidateStoreReaderFixture(t, openedPath)
	writeWeightedReadinessFixture(t, openedPath, stringsReplaceFirst(readText(t, openedPath), `"graft_allowed": false`, `"graft_allowed": true`))
	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionProof([]string{openedPath, filepath.Join(dir, "opened_proof.json")}),
		"weighted admission resonance graft candidate store reader opened graft_allowed",
	)

	badSchemaPath := filepath.Join(dir, "bad_schema_reader.json")
	writeWeightedAdmissionResonanceGraftCandidateStoreReaderFixture(t, badSchemaPath)
	writeWeightedReadinessFixture(t, badSchemaPath, stringsReplaceFirst(readText(t, badSchemaPath), `"schema": "arianna.live_route_weighted_admission_resonance_graft_candidate_store_reader.v1"`, `"schema": "arianna.live_route_weighted_admission_resonance_graft_candidate_store_reader.v0"`))
	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionProof([]string{badSchemaPath, filepath.Join(dir, "bad_schema_proof.json")}),
		`weighted admission resonance graft candidate store reader schema mismatch: got "arianna.live_route_weighted_admission_resonance_graft_candidate_store_reader.v0" want "`+admissionLiveRouteWeightedAdmissionResonanceGraftCandidateStoreReaderSchema+`"`,
	)

	badHashPath := filepath.Join(dir, "bad_hash_reader.json")
	writeWeightedAdmissionResonanceGraftCandidateStoreReaderFixture(t, badHashPath)
	writeWeightedReadinessFixture(t, badHashPath, stringsReplaceFirst(readText(t, badHashPath), `"reader_hash": "weighted-resonance-graft-candidate-store-reader-`, `"reader_hash": "weighted-resonance-graft-candidate-store-reader-bad`))
	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionProof([]string{badHashPath, filepath.Join(dir, "bad_hash_proof.json")}),
		"weighted admission resonance graft candidate store reader reader_hash mismatch",
	)

	if err := runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionProof([]string{readerPath, filepath.Join(dir, "missing", "proof.json")}); err == nil ||
		!strings.HasPrefix(err.Error(), "weighted admission resonance graft admission proof write failed:") {
		t.Fatalf("expected weighted admission resonance graft admission proof write failure, got %v", err)
	}
}
