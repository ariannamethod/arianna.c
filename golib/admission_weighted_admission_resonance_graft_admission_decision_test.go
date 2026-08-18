package main

import (
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionDecision(t *testing.T) {
	dir := t.TempDir()

	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionDecision(nil),
		"usage: --admission-live-route-weighted-admission-resonance-graft-admission-decision RESONANCE_GRAFT_ADMISSION_PROOF_PRECONDITION_REPORT RESONANCE_GRAFT_ADMISSION_DECISION_REPORT",
	)
	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionDecision([]string{"precondition.json"}),
		"usage: --admission-live-route-weighted-admission-resonance-graft-admission-decision RESONANCE_GRAFT_ADMISSION_PROOF_PRECONDITION_REPORT RESONANCE_GRAFT_ADMISSION_DECISION_REPORT",
	)
	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionDecision([]string{"precondition.json", "decision.json", "extra"}),
		"usage: --admission-live-route-weighted-admission-resonance-graft-admission-decision RESONANCE_GRAFT_ADMISSION_PROOF_PRECONDITION_REPORT RESONANCE_GRAFT_ADMISSION_DECISION_REPORT",
	)
	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionDecision([]string{"  ", filepath.Join(dir, "decision.json")}),
		"weighted admission resonance graft admission proof precondition path missing",
	)

	preconditionPath := filepath.Join(dir, "precondition.json")
	writeWeightedAdmissionResonanceGraftAdmissionProofPreconditionFixture(t, preconditionPath)

	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionDecision([]string{preconditionPath, "  "}),
		"weighted admission resonance graft admission decision output path missing",
	)

	decisionPath := filepath.Join(dir, "decision.json")
	if err := runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionDecision([]string{preconditionPath, decisionPath}); err != nil {
		t.Fatalf("valid weighted admission resonance graft admission decision rejected: %v", err)
	}
	raw, err := os.ReadFile(decisionPath)
	if err != nil {
		t.Fatalf("read weighted admission resonance graft admission decision: %v", err)
	}
	var decision admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionDecisionReport
	if err := json.Unmarshal(raw, &decision); err != nil {
		t.Fatalf("decode weighted admission resonance graft admission decision: %v", err)
	}
	preconditionRaw, err := os.ReadFile(preconditionPath)
	if err != nil {
		t.Fatalf("read weighted admission resonance graft admission proof precondition: %v", err)
	}
	var precondition admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionProofPreconditionReport
	if err := json.Unmarshal(preconditionRaw, &precondition); err != nil {
		t.Fatalf("decode weighted admission resonance graft admission proof precondition: %v", err)
	}
	if decision.Schema != admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionDecisionSchema ||
		decision.Status != "shadow_graft_admission_decision_ready_dry_run" ||
		decision.Target != "live_route_admission_next_step" ||
		decision.TargetKind != "weighted_internal_world_shadow_graft_admission_decision" ||
		decision.TargetMode != "closed_decision_receipt_dry_run" ||
		decision.Action != "decide_weighted_resonance_shadow_graft_admission_dry_run" ||
		decision.Decision != "shadow_ready" ||
		!decision.WeightedAdmissionResonanceGraftAdmissionDecisionReady ||
		!decision.WeightedAdmissionResonanceGraftAdmissionProofPreconditionConsumed ||
		!decision.WeightedAdmissionResonanceGraftAdmissionProofPreconditionRequired ||
		!decision.NextStepBlockedWithoutResonanceGraftAdmissionDecision ||
		decision.ReceiptShape != "weighted_resonance_shadow_graft_admission_decision_receipt" ||
		decision.DecisionKind != "shadow_graft_admission_decision" ||
		decision.DecisionMode != "closed_precondition_decision" ||
		decision.DecisionStage != "pre_live_graft_admission_decision" ||
		decision.CausalID != admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionDecisionCausalID(decision) ||
		decision.DecisionHash != admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionDecisionHash(decision) ||
		decision.ReadBackHash != admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionDecisionReadBackHash(decision) ||
		decision.DecisionHash == decision.ReadBackHash ||
		decision.WeightedAdmissionResonanceGraftAdmissionDecisionID != admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionDecisionID(decision) ||
		!decision.ProofPreconditionVerified ||
		!decision.PreconditionHashVerified ||
		!decision.PreconditionReadBackVerified ||
		!decision.ProofVerified ||
		!decision.ProofHashVerified ||
		!decision.ProofReadBackVerified ||
		!decision.StoreReaderVerified ||
		!decision.StoreVerified ||
		!decision.CandidateVerified ||
		!decision.GateVerified ||
		!decision.PreflightVerified ||
		!decision.BoundaryVerified ||
		!decision.ObservationVerified ||
		!decision.ReceiverVerified ||
		!decision.IntentVerified ||
		!decision.FinalGateVerified ||
		!decision.SealVerified ||
		!decision.PermitVerified ||
		!decision.AuthorityVerified ||
		!decision.AdmissionRequired ||
		!decision.ShadowOnly ||
		decision.GraftAllowed ||
		!decision.DryRunOnly ||
		!decision.LiveReady ||
		decision.RawDreamTextAllowed ||
		decision.RawDreamTextObserved ||
		decision.RawDreamTextForwarded ||
		decision.JanusSurfaceAllowed ||
		decision.CoocLearningAllowed ||
		decision.DeltaHarvestAllowed ||
		decision.BodyMutationAllowed ||
		!decision.RollbackRequired ||
		!decision.ReadOnly ||
		!decision.ReplayOnly ||
		decision.SourceSchema != admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionProofPreconditionSchema ||
		decision.SourceStatus != "shadow_graft_admission_proof_precondition_satisfied_dry_run" ||
		decision.SourceTarget != "live_route_admission_next_step" ||
		decision.SourceReport != preconditionPath ||
		decision.SourceWeightedAdmissionResonanceGraftAdmissionProofPreconditionID != precondition.WeightedAdmissionResonanceGraftAdmissionProofPreconditionID ||
		!decision.SourceWeightedAdmissionResonanceGraftAdmissionProofPreconditionReady ||
		decision.SourceWeightedAdmissionResonanceGraftAdmissionProofPreconditionCausalID != precondition.CausalID ||
		decision.SourceWeightedAdmissionResonanceGraftAdmissionProofPreconditionHash != precondition.PreconditionHash ||
		decision.SourceWeightedAdmissionResonanceGraftAdmissionProofPreconditionReadBack != precondition.ReadBackHash ||
		decision.SourcePreconditionAction != "consume_weighted_resonance_shadow_graft_admission_proof_before_live_route_admission" ||
		decision.SourcePreconditionReceiptShape != "weighted_resonance_shadow_graft_admission_proof_precondition_receipt" ||
		decision.SourcePreconditionKind != "shadow_graft_admission_proof_precondition" ||
		decision.SourcePreconditionMode != "closed_receipt_consumption" ||
		decision.SourcePreconditionStage != "pre_live_graft_admission_proof_precondition" ||
		!decision.SourcePreconditionAdmissionRequired ||
		!decision.SourcePreconditionShadowOnly ||
		decision.SourcePreconditionGraftAllowed ||
		!decision.SourcePreconditionDryRunOnly ||
		!decision.SourcePreconditionLiveReady ||
		decision.SourcePreconditionRawDreamTextAllowed ||
		decision.SourcePreconditionBodyMutationAllowed ||
		!decision.SourcePreconditionRollbackRequired ||
		!decision.SourcePreconditionReadOnly ||
		!decision.SourcePreconditionReplayOnly ||
		decision.SourcePreconditionBodyTarget != "none" ||
		!decision.SourcePreconditionPassed ||
		decision.SourceWeightedAdmissionResonanceGraftAdmissionProofID != precondition.SourceWeightedAdmissionResonanceGraftAdmissionProofID ||
		!decision.SourceWeightedAdmissionResonanceGraftAdmissionProofReady ||
		decision.SourceWeightedAdmissionResonanceGraftAdmissionProofHash != precondition.SourceWeightedAdmissionResonanceGraftAdmissionProofHash ||
		decision.SourceWeightedAdmissionResonanceGraftAdmissionProofReadBack != precondition.SourceWeightedAdmissionResonanceGraftAdmissionProofReadBack ||
		decision.SourceProofAction != "prove_weighted_resonance_shadow_graft_admission_dry_run" ||
		decision.SourceProofReceiptShape != "weighted_resonance_shadow_graft_admission_proof_receipt" ||
		decision.SourceProofKind != "shadow_graft_admission_proof" ||
		decision.SourceProofMode != "closed_read_back_admission_proof" ||
		decision.SourceProofStage != "pre_live_graft_admission_proof" ||
		decision.SourceProofGraftAllowed ||
		decision.SourceProofLiveAdmissionEnabled ||
		decision.SourceProofBodyMutationAllowed ||
		decision.SourceProofBodyTarget != "none" ||
		!decision.SourceProofPassed ||
		decision.SourceWeightedAdmissionResonanceGraftCandidateStoreReaderID != precondition.SourceWeightedAdmissionResonanceGraftCandidateStoreReaderID ||
		decision.SourceWeightedAdmissionResonanceGraftCandidateStoreID != precondition.SourceWeightedAdmissionResonanceGraftCandidateStoreID ||
		decision.SourceWeightedAdmissionResonanceGraftCandidateID != precondition.SourceWeightedAdmissionResonanceGraftCandidateID ||
		decision.SourceWeightedAdmissionResonanceGraftGateID != precondition.SourceWeightedAdmissionResonanceGraftGateID ||
		decision.SourceWeightedAdmissionResonanceGraftPreflightID != precondition.SourceWeightedAdmissionResonanceGraftPreflightID ||
		decision.SourceWeightedAdmissionResonanceGraftBoundaryID != precondition.SourceWeightedAdmissionResonanceGraftBoundaryID ||
		decision.SourceWeightedAdmissionResonanceObservationID != precondition.SourceWeightedAdmissionResonanceObservationID ||
		decision.SourceWeightedAdmissionResonanceReceiverID != precondition.SourceWeightedAdmissionResonanceReceiverID ||
		!decision.BodySmokeWeighted ||
		!decision.NanoDirectRunner ||
		!decision.NanoDirectFinalGate ||
		!decision.ResonanceGraftAdmissionProof ||
		!decision.BoundaryReportFullChain ||
		decision.SourceAuthorityGranted ||
		decision.AuthorityGranted ||
		decision.ContractsReady ||
		decision.WriteAllowed ||
		decision.AdmissionAllowed ||
		decision.LiveAdmissionEnabled ||
		decision.MutatesState ||
		decision.BodyTarget != "none" ||
		!decision.Passed ||
		decision.Reason != "weighted resonance shadow graft admission decision accepted precondition as closed shadow-ready receipt" {
		t.Fatalf("weighted admission resonance graft admission decision lost contract: %+v", decision)
	}

	openedPath := filepath.Join(dir, "opened_precondition.json")
	writeWeightedAdmissionResonanceGraftAdmissionProofPreconditionFixture(t, openedPath)
	writeWeightedReadinessFixture(t, openedPath, stringsReplaceFirst(readText(t, openedPath), `"graft_allowed": false`, `"graft_allowed": true`))
	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionDecision([]string{openedPath, filepath.Join(dir, "opened_decision.json")}),
		"weighted admission resonance graft admission proof precondition opened graft_allowed",
	)

	badSchemaPath := filepath.Join(dir, "bad_schema_precondition.json")
	writeWeightedAdmissionResonanceGraftAdmissionProofPreconditionFixture(t, badSchemaPath)
	writeWeightedReadinessFixture(t, badSchemaPath, stringsReplaceFirst(readText(t, badSchemaPath), `"schema": "arianna.live_route_weighted_admission_resonance_graft_admission_proof_precondition.v1"`, `"schema": "arianna.live_route_weighted_admission_resonance_graft_admission_proof_precondition.v0"`))
	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionDecision([]string{badSchemaPath, filepath.Join(dir, "bad_schema_decision.json")}),
		`weighted admission resonance graft admission proof precondition schema mismatch: got "arianna.live_route_weighted_admission_resonance_graft_admission_proof_precondition.v0" want "`+admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionProofPreconditionSchema+`"`,
	)

	badHashPath := filepath.Join(dir, "bad_hash_precondition.json")
	writeWeightedAdmissionResonanceGraftAdmissionProofPreconditionFixture(t, badHashPath)
	writeWeightedReadinessFixture(t, badHashPath, stringsReplaceFirst(readText(t, badHashPath), `"precondition_hash": "weighted-resonance-graft-admission-proof-precondition-`, `"precondition_hash": "weighted-resonance-graft-admission-proof-precondition-bad`))
	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionDecision([]string{badHashPath, filepath.Join(dir, "bad_hash_decision.json")}),
		"weighted admission resonance graft admission proof precondition precondition_hash mismatch",
	)

	if err := runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionDecision([]string{preconditionPath, filepath.Join(dir, "missing", "decision.json")}); err == nil ||
		!strings.HasPrefix(err.Error(), "weighted admission resonance graft admission decision write failed:") {
		t.Fatalf("expected weighted admission resonance graft admission decision write failure, got %v", err)
	}
}
