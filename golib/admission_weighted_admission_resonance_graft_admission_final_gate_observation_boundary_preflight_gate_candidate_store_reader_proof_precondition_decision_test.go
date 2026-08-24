package main

import (
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecision(t *testing.T) {
	dir := t.TempDir()

	usage := "usage: --admission-live-route-weighted-admission-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-candidate-store-reader-proof-precondition-decision RESONANCE_GRAFT_ADMISSION_FINAL_GATE_OBSERVATION_BOUNDARY_PREFLIGHT_GATE_CANDIDATE_STORE_READER_PROOF_PRECONDITION_REPORT RESONANCE_GRAFT_ADMISSION_FINAL_GATE_OBSERVATION_BOUNDARY_PREFLIGHT_GATE_CANDIDATE_STORE_READER_PROOF_PRECONDITION_DECISION_REPORT"
	requireBoundaryAssertError(t, runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecision(nil), usage)
	requireBoundaryAssertError(t, runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecision([]string{"precondition.json"}), usage)
	requireBoundaryAssertError(t, runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecision([]string{"precondition.json", "decision.json", "extra"}), usage)
	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecision([]string{"  ", filepath.Join(dir, "decision.json")}),
		"weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition path missing",
	)

	preconditionPath := filepath.Join(dir, "precondition.json")
	writeWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionFixture(t, preconditionPath)
	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecision([]string{preconditionPath, "  "}),
		"weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision output path missing",
	)

	decisionPath := filepath.Join(dir, "decision.json")
	if err := runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecision([]string{preconditionPath, decisionPath}); err != nil {
		t.Fatalf("valid weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision rejected: %v", err)
	}
	raw, err := os.ReadFile(decisionPath)
	if err != nil {
		t.Fatalf("read weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision: %v", err)
	}
	var decision admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionReport
	if err := json.Unmarshal(raw, &decision); err != nil {
		t.Fatalf("decode weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision: %v", err)
	}
	preconditionRaw, err := os.ReadFile(preconditionPath)
	if err != nil {
		t.Fatalf("read weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition: %v", err)
	}
	var precondition admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionReport
	if err := json.Unmarshal(preconditionRaw, &precondition); err != nil {
		t.Fatalf("decode weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition: %v", err)
	}
	if decision.Schema != admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionSchema ||
		decision.Status != "shadow_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_ready_dry_run" ||
		decision.Target != "live_route_admission_next_step" ||
		decision.TargetKind != "weighted_internal_world_shadow_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision" ||
		decision.TargetMode != "closed_decision_receipt_dry_run" ||
		decision.Action != "decide_weighted_resonance_shadow_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_dry_run" ||
		decision.Decision != "shadow_ready" ||
		decision.LedgerReady ||
		decision.LedgerAppendAllowed ||
		!decision.WeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionReady ||
		!decision.WeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionConsumed ||
		!decision.WeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionRequired ||
		!decision.NextStepBlockedWithoutResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecision ||
		decision.ReceiptShape != "weighted_resonance_shadow_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_receipt" ||
		decision.DecisionKind != "shadow_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision" ||
		decision.DecisionMode != "closed_proof_precondition_decision" ||
		decision.DecisionStage != "post_preflight_gate_candidate_store_reader_proof_precondition_pre_live_admission_decision" ||
		decision.CausalID != admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionCausalID(decision) ||
		decision.DecisionHash != admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionHash(decision) ||
		decision.ReadBackHash != admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionReadBackHash(decision) ||
		decision.DecisionHash == decision.ReadBackHash ||
		decision.WeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionID != admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionID(decision) ||
		!decision.ProofPreconditionVerified ||
		!decision.PreconditionHashVerified ||
		!decision.PreconditionReadBackVerified ||
		!decision.ProofVerified ||
		!decision.ProofHashVerified ||
		!decision.ProofReadBackVerified ||
		!decision.ReaderHashVerified ||
		!decision.ReaderReplayVerified ||
		!decision.ReaderReadBackVerified ||
		!decision.StoreHashVerified ||
		!decision.StoreReadBackVerified ||
		!decision.AdmissionRequired ||
		!decision.ShadowOnly ||
		decision.GraftAllowed ||
		!decision.DryRunOnly ||
		!decision.LiveReady ||
		decision.RawDreamTextAllowed ||
		decision.BodyMutationAllowed ||
		!decision.RollbackRequired ||
		!decision.ReadOnly ||
		!decision.ReplayOnly ||
		decision.AuthorityGranted ||
		decision.ContractsReady ||
		decision.WriteAllowed ||
		decision.AdmissionAllowed ||
		decision.LiveAdmissionEnabled ||
		decision.MutatesState ||
		decision.BodyTarget != "none" ||
		!decision.Passed ||
		decision.Reason != "weighted resonance shadow graft admission final gate observation boundary preflight gate candidate store reader proof precondition accepted as closed shadow-ready decision receipt" ||
		decision.SourceSchema != admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionSchema ||
		decision.SourceStatus != "shadow_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_satisfied_dry_run" ||
		decision.SourceReport != preconditionPath ||
		decision.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionID != precondition.WeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionID ||
		decision.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionHash != precondition.PreconditionHash ||
		decision.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionReadBack != precondition.ReadBackHash ||
		decision.SourcePreconditionLedgerAppendAllowed ||
		decision.SourcePreconditionGraftAllowed ||
		decision.SourcePreconditionLiveAdmissionEnabled ||
		decision.SourcePreconditionBodyMutationAllowed ||
		decision.SourcePreconditionBodyTarget != "none" ||
		!decision.SourcePreconditionPassed ||
		decision.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofID != precondition.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofID ||
		decision.SourceProofLedgerAppendAllowed ||
		decision.SourceProofGraftAllowed ||
		decision.SourceProofLiveAdmissionEnabled ||
		decision.SourceProofBodyMutationAllowed ||
		decision.SourceProofBodyTarget != "none" ||
		!decision.SourceProofPassed ||
		decision.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderID != precondition.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderID ||
		decision.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreID != precondition.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreID ||
		decision.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateID != precondition.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateID ||
		decision.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateID != precondition.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateID ||
		decision.SourceAdmissionFinalGateObservationBoundaryPreflightGateReady {
		t.Fatalf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision lost contract: %+v", decision)
	}

	badSchemaPath := filepath.Join(dir, "bad_schema_precondition.json")
	writeWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionFixture(t, badSchemaPath)
	writeWeightedReadinessFixture(t, badSchemaPath, stringsReplaceFirst(readText(t, badSchemaPath), `"schema": "arianna.live_route_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition.v1"`, `"schema": "arianna.live_route_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition.v0"`))
	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecision([]string{badSchemaPath, filepath.Join(dir, "bad_schema_decision.json")}),
		`weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition schema mismatch: got "arianna.live_route_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition.v0" want "`+admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionSchema+`"`,
	)

	openedPath := filepath.Join(dir, "opened_precondition.json")
	writeWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionFixture(t, openedPath)
	writeWeightedReadinessFixture(t, openedPath, stringsReplaceFirst(readText(t, openedPath), `"live_admission_enabled": false`, `"live_admission_enabled": true`))
	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecision([]string{openedPath, filepath.Join(dir, "opened_decision.json")}),
		"weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition opened live_admission_enabled",
	)

	badHashPath := filepath.Join(dir, "bad_hash_precondition.json")
	writeWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionFixture(t, badHashPath)
	writeWeightedReadinessFixture(t, badHashPath, stringsReplaceFirst(readText(t, badHashPath), `"precondition_hash": "weighted-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-candidate-store-reader-proof-precondition-`, `"precondition_hash": "weighted-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-candidate-store-reader-proof-precondition-bad`))
	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecision([]string{badHashPath, filepath.Join(dir, "bad_hash_decision.json")}),
		"weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition precondition_hash mismatch",
	)

	if err := runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecision([]string{preconditionPath, filepath.Join(dir, "missing", "decision.json")}); err == nil ||
		!strings.HasPrefix(err.Error(), "weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision write failed:") {
		t.Fatalf("expected weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision write failure, got %v", err)
	}
}
