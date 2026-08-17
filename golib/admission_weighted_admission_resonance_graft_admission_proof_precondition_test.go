package main

import (
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionProofPrecondition(t *testing.T) {
	dir := t.TempDir()

	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionProofPrecondition(nil),
		"usage: --admission-live-route-weighted-admission-resonance-graft-admission-proof-precondition RESONANCE_GRAFT_ADMISSION_PROOF_REPORT RESONANCE_GRAFT_ADMISSION_PROOF_PRECONDITION_REPORT",
	)
	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionProofPrecondition([]string{"proof.json"}),
		"usage: --admission-live-route-weighted-admission-resonance-graft-admission-proof-precondition RESONANCE_GRAFT_ADMISSION_PROOF_REPORT RESONANCE_GRAFT_ADMISSION_PROOF_PRECONDITION_REPORT",
	)
	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionProofPrecondition([]string{"proof.json", "precondition.json", "extra"}),
		"usage: --admission-live-route-weighted-admission-resonance-graft-admission-proof-precondition RESONANCE_GRAFT_ADMISSION_PROOF_REPORT RESONANCE_GRAFT_ADMISSION_PROOF_PRECONDITION_REPORT",
	)
	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionProofPrecondition([]string{"  ", filepath.Join(dir, "precondition.json")}),
		"weighted admission resonance graft admission proof path missing",
	)

	proofPath := filepath.Join(dir, "proof.json")
	writeWeightedAdmissionResonanceGraftAdmissionProofFixture(t, proofPath)

	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionProofPrecondition([]string{proofPath, "  "}),
		"weighted admission resonance graft admission proof precondition output path missing",
	)

	preconditionPath := filepath.Join(dir, "precondition.json")
	if err := runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionProofPrecondition([]string{proofPath, preconditionPath}); err != nil {
		t.Fatalf("valid weighted admission resonance graft admission proof precondition rejected: %v", err)
	}
	raw, err := os.ReadFile(preconditionPath)
	if err != nil {
		t.Fatalf("read weighted admission resonance graft admission proof precondition: %v", err)
	}
	var precondition admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionProofPreconditionReport
	if err := json.Unmarshal(raw, &precondition); err != nil {
		t.Fatalf("decode weighted admission resonance graft admission proof precondition: %v", err)
	}
	proofRaw, err := os.ReadFile(proofPath)
	if err != nil {
		t.Fatalf("read weighted admission resonance graft admission proof: %v", err)
	}
	var proof admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionProofReport
	if err := json.Unmarshal(proofRaw, &proof); err != nil {
		t.Fatalf("decode weighted admission resonance graft admission proof: %v", err)
	}
	if precondition.Schema != admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionProofPreconditionSchema ||
		precondition.Status != "shadow_graft_admission_proof_precondition_satisfied_dry_run" ||
		precondition.Target != "live_route_admission_next_step" ||
		precondition.TargetKind != "weighted_internal_world_shadow_graft_admission_proof_precondition" ||
		precondition.TargetMode != "closed_receipt_precondition_dry_run" ||
		precondition.Action != "consume_weighted_resonance_shadow_graft_admission_proof_before_live_route_admission" ||
		!precondition.WeightedAdmissionResonanceGraftAdmissionProofPreconditionReady ||
		!precondition.WeightedAdmissionResonanceGraftAdmissionProofConsumed ||
		!precondition.WeightedAdmissionResonanceGraftAdmissionProofRequired ||
		!precondition.NextStepBlockedWithoutResonanceGraftAdmissionProof ||
		precondition.ReceiptShape != "weighted_resonance_shadow_graft_admission_proof_precondition_receipt" ||
		precondition.PreconditionKind != "shadow_graft_admission_proof_precondition" ||
		precondition.PreconditionMode != "closed_receipt_consumption" ||
		precondition.PreconditionStage != "pre_live_graft_admission_proof_precondition" ||
		precondition.CausalID != admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionProofPreconditionCausalID(precondition) ||
		precondition.PreconditionHash != admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionProofPreconditionHash(precondition) ||
		precondition.ReadBackHash != admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionProofPreconditionReadBackHash(precondition) ||
		precondition.PreconditionHash == precondition.ReadBackHash ||
		precondition.WeightedAdmissionResonanceGraftAdmissionProofPreconditionID != admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionProofPreconditionID(precondition) ||
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
		precondition.SourceSchema != admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionProofSchema ||
		precondition.SourceStatus != "shadow_graft_admission_proof_ready_dry_run" ||
		precondition.SourceTarget != "resonance" ||
		precondition.SourceReport != proofPath ||
		precondition.SourceWeightedAdmissionResonanceGraftAdmissionProofID != proof.WeightedAdmissionResonanceGraftAdmissionProofID ||
		!precondition.SourceWeightedAdmissionResonanceGraftAdmissionProofReady ||
		precondition.SourceWeightedAdmissionResonanceGraftAdmissionProofCausalID != proof.CausalID ||
		precondition.SourceWeightedAdmissionResonanceGraftAdmissionProofHash != proof.ProofHash ||
		precondition.SourceWeightedAdmissionResonanceGraftAdmissionProofReadBack != proof.ReadBackHash ||
		precondition.SourceProofAction != "prove_weighted_resonance_shadow_graft_admission_dry_run" ||
		precondition.SourceProofReceiptShape != "weighted_resonance_shadow_graft_admission_proof_receipt" ||
		precondition.SourceProofKind != "shadow_graft_admission_proof" ||
		precondition.SourceProofMode != "closed_read_back_admission_proof" ||
		precondition.SourceProofStage != "pre_live_graft_admission_proof" ||
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
		precondition.SourceWeightedAdmissionResonanceGraftCandidateStoreReaderID != proof.SourceWeightedAdmissionResonanceGraftCandidateStoreReaderID ||
		precondition.SourceWeightedAdmissionResonanceGraftCandidateStoreID != proof.SourceWeightedAdmissionResonanceGraftCandidateStoreID ||
		precondition.SourceWeightedAdmissionResonanceGraftCandidateID != proof.SourceWeightedAdmissionResonanceGraftCandidateID ||
		precondition.SourceWeightedAdmissionResonanceGraftGateID != proof.SourceWeightedAdmissionResonanceGraftGateID ||
		precondition.SourceWeightedAdmissionResonanceGraftPreflightID != proof.SourceWeightedAdmissionResonanceGraftPreflightID ||
		precondition.SourceWeightedAdmissionResonanceGraftBoundaryID != proof.SourceWeightedAdmissionResonanceGraftBoundaryID ||
		precondition.SourceWeightedAdmissionResonanceObservationID != proof.SourceWeightedAdmissionResonanceObservationID ||
		precondition.SourceWeightedAdmissionResonanceReceiverID != proof.SourceWeightedAdmissionResonanceReceiverID ||
		!precondition.BodySmokeWeighted ||
		!precondition.NanoDirectRunner ||
		!precondition.NanoDirectFinalGate ||
		!precondition.ResonanceGraftAdmissionProof ||
		!precondition.BoundaryReportFullChain ||
		precondition.SourceAuthorityGranted ||
		precondition.AuthorityGranted ||
		precondition.ContractsReady ||
		precondition.WriteAllowed ||
		precondition.AdmissionAllowed ||
		precondition.LiveAdmissionEnabled ||
		precondition.MutatesState ||
		precondition.BodyTarget != "none" ||
		!precondition.Passed ||
		precondition.Reason != "weighted resonance shadow graft admission proof consumed as closed precondition" {
		t.Fatalf("weighted admission resonance graft admission proof precondition lost contract: %+v", precondition)
	}

	openedPath := filepath.Join(dir, "opened_proof.json")
	writeWeightedAdmissionResonanceGraftAdmissionProofFixture(t, openedPath)
	writeWeightedReadinessFixture(t, openedPath, stringsReplaceFirst(readText(t, openedPath), `"graft_allowed": false`, `"graft_allowed": true`))
	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionProofPrecondition([]string{openedPath, filepath.Join(dir, "opened_precondition.json")}),
		"weighted admission resonance graft admission proof opened graft_allowed",
	)

	badSchemaPath := filepath.Join(dir, "bad_schema_proof.json")
	writeWeightedAdmissionResonanceGraftAdmissionProofFixture(t, badSchemaPath)
	writeWeightedReadinessFixture(t, badSchemaPath, stringsReplaceFirst(readText(t, badSchemaPath), `"schema": "arianna.live_route_weighted_admission_resonance_graft_admission_proof.v1"`, `"schema": "arianna.live_route_weighted_admission_resonance_graft_admission_proof.v0"`))
	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionProofPrecondition([]string{badSchemaPath, filepath.Join(dir, "bad_schema_precondition.json")}),
		`weighted admission resonance graft admission proof schema mismatch: got "arianna.live_route_weighted_admission_resonance_graft_admission_proof.v0" want "`+admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionProofSchema+`"`,
	)

	badHashPath := filepath.Join(dir, "bad_hash_proof.json")
	writeWeightedAdmissionResonanceGraftAdmissionProofFixture(t, badHashPath)
	writeWeightedReadinessFixture(t, badHashPath, stringsReplaceFirst(readText(t, badHashPath), `"proof_hash": "weighted-resonance-graft-admission-proof-`, `"proof_hash": "weighted-resonance-graft-admission-proof-bad`))
	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionProofPrecondition([]string{badHashPath, filepath.Join(dir, "bad_hash_precondition.json")}),
		"weighted admission resonance graft admission proof proof_hash mismatch",
	)

	if err := runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionProofPrecondition([]string{proofPath, filepath.Join(dir, "missing", "precondition.json")}); err == nil ||
		!strings.HasPrefix(err.Error(), "weighted admission resonance graft admission proof precondition write failed:") {
		t.Fatalf("expected weighted admission resonance graft admission proof precondition write failure, got %v", err)
	}
}
