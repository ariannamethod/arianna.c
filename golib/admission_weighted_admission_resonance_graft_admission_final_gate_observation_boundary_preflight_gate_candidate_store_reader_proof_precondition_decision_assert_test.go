package main

import (
	"path/filepath"
	"strings"
	"testing"
)

func TestAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionAssert(t *testing.T) {
	dir := t.TempDir()

	requireBoundaryAssertError(t, runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionAssert(nil), "usage: --admission-live-route-weighted-admission-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-candidate-store-reader-proof-precondition-decision-assert REPORT")
	requireBoundaryAssertError(t, runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionAssert([]string{"decision.json", "extra"}), "usage: --admission-live-route-weighted-admission-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-candidate-store-reader-proof-precondition-decision-assert REPORT")
	requireBoundaryAssertError(t, runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionAssert([]string{"  "}), "weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision path missing")
	requireBoundaryAssertError(t, runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionAssert([]string{filepath.Join(dir, "missing.json")}), "weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision not written")

	emptyPath := filepath.Join(dir, "empty.json")
	writeWeightedReadinessFixture(t, emptyPath, "")
	requireBoundaryAssertError(t, runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionAssert([]string{emptyPath}), "weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision not written")

	invalidPath := filepath.Join(dir, "invalid.json")
	writeWeightedReadinessFixture(t, invalidPath, "{")
	err := runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionAssert([]string{invalidPath})
	if err == nil || !strings.HasPrefix(err.Error(), "weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision JSON invalid:") {
		t.Fatalf("expected invalid JSON error, got %v", err)
	}

	missingSchemaPath := filepath.Join(dir, "missing_schema.json")
	writeWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionFixture(t, missingSchemaPath)
	writeWeightedReadinessFixture(t, missingSchemaPath, stringsReplaceFirst(readText(t, missingSchemaPath), `"schema": "arianna.live_route_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision.v1",`, ""))
	requireBoundaryAssertError(t, runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionAssert([]string{missingSchemaPath}), "weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision schema missing")

	badSchemaPath := filepath.Join(dir, "bad_schema.json")
	writeWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionFixture(t, badSchemaPath)
	writeWeightedReadinessFixture(t, badSchemaPath, stringsReplaceFirst(readText(t, badSchemaPath), `"schema": "arianna.live_route_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision.v1"`, `"schema": "arianna.live_route_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision.v0"`))
	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionAssert([]string{badSchemaPath}),
		`weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision schema mismatch: got "arianna.live_route_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision.v0" want "`+admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionSchema+`"`,
	)

	validPath := filepath.Join(dir, "valid.json")
	writeWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionFixture(t, validPath)
	if err := runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionAssert([]string{validPath}); err != nil {
		t.Fatalf("valid weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision rejected: %v", err)
	}

	badStatusPath := filepath.Join(dir, "bad_status.json")
	writeWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionFixture(t, badStatusPath)
	writeWeightedReadinessFixture(t, badStatusPath, stringsReplaceFirst(readText(t, badStatusPath), `"status": "shadow_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_ready_dry_run"`, `"status": "open"`))
	requireBoundaryAssertError(t, runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionAssert([]string{badStatusPath}), `weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision status mismatch: got "open" want "shadow_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_ready_dry_run"`)

	notReadyPath := filepath.Join(dir, "not_ready.json")
	writeWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionFixture(t, notReadyPath)
	writeWeightedReadinessFixture(t, notReadyPath, stringsReplaceFirst(readText(t, notReadyPath), `"weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_ready": true`, `"weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_ready": false`))
	requireBoundaryAssertError(t, runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionAssert([]string{notReadyPath}), "weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_ready not ready")

	badDecisionPath := filepath.Join(dir, "bad_decision.json")
	writeWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionFixture(t, badDecisionPath)
	writeWeightedReadinessFixture(t, badDecisionPath, stringsReplaceFirst(readText(t, badDecisionPath), `"decision": "shadow_ready"`, `"decision": "reject"`))
	requireBoundaryAssertError(t, runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionAssert([]string{badDecisionPath}), `weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision decision mismatch: got "reject" want "shadow_ready"`)

	openedPath := filepath.Join(dir, "opened.json")
	writeWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionFixture(t, openedPath)
	writeWeightedReadinessFixture(t, openedPath, stringsReplaceFirst(readText(t, openedPath), `"live_admission_enabled": false`, `"live_admission_enabled": true`))
	requireBoundaryAssertError(t, runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionAssert([]string{openedPath}), "weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision opened live_admission_enabled")

	missingPathField := filepath.Join(dir, "missing_path.json")
	writeWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionFixture(t, missingPathField)
	preconditionReport := filepath.Join(dir, "precondition-"+filepath.Base(missingPathField))
	writeWeightedReadinessFixture(t, missingPathField, stringsReplaceFirst(readText(t, missingPathField), `"source_report": "`+preconditionReport+`"`, `"source_report": " "`))
	requireBoundaryAssertError(t, runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionAssert([]string{missingPathField}), "weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision source_report missing")

	badSourcePath := filepath.Join(dir, "bad_source.json")
	writeWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionFixture(t, badSourcePath)
	writeWeightedReadinessFixture(t, badSourcePath, stringsReplaceFirst(readText(t, badSourcePath), `"source_schema": "arianna.live_route_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition.v1"`, `"source_schema": "arianna.live_route_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition.v0"`))
	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionAssert([]string{badSourcePath}),
		`weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision source_schema mismatch: got "arianna.live_route_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition.v0" want "`+admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionSchema+`"`,
	)

	badSourcePreconditionKindPath := filepath.Join(dir, "bad_source_precondition_kind.json")
	writeWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionFixture(t, badSourcePreconditionKindPath)
	writeWeightedReadinessFixture(t, badSourcePreconditionKindPath, stringsReplaceFirst(readText(t, badSourcePreconditionKindPath), `"source_precondition_kind": "shadow_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition"`, `"source_precondition_kind": "live_precondition"`))
	requireBoundaryAssertError(t, runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionAssert([]string{badSourcePreconditionKindPath}), "weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision source precondition shape mismatch")

	badSourceProofGuardPath := filepath.Join(dir, "bad_source_proof_guard.json")
	writeWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionFixture(t, badSourceProofGuardPath)
	writeWeightedReadinessFixture(t, badSourceProofGuardPath, stringsReplaceFirst(readText(t, badSourceProofGuardPath), `"source_proof_ledger_append_allowed": false`, `"source_proof_ledger_append_allowed": true`))
	requireBoundaryAssertError(t, runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionAssert([]string{badSourceProofGuardPath}), "weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision opened source_proof_ledger_append_allowed")

	badDecisionHashPath := filepath.Join(dir, "bad_decision_hash.json")
	writeWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionFixture(t, badDecisionHashPath)
	writeWeightedReadinessFixture(t, badDecisionHashPath, stringsReplaceFirst(readText(t, badDecisionHashPath), `"decision_hash": "weighted-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-candidate-store-reader-proof-precondition-decision-`, `"decision_hash": "weighted-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-candidate-store-reader-proof-precondition-decision-bad`))
	requireBoundaryAssertError(t, runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionAssert([]string{badDecisionHashPath}), "weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision decision_hash mismatch")

	badBodyTargetPath := filepath.Join(dir, "bad_body_target.json")
	writeWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionFixture(t, badBodyTargetPath)
	writeWeightedReadinessFixture(t, badBodyTargetPath, stringsReplaceFirst(readText(t, badBodyTargetPath), `"body_target": "none"`, `"body_target": "live"`))
	requireBoundaryAssertError(t, runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionAssert([]string{badBodyTargetPath}), "weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision body target mismatch")
}

func writeWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionFixture(t *testing.T, decisionPath string) {
	t.Helper()
	dir := filepath.Dir(decisionPath)
	preconditionPath := filepath.Join(dir, "precondition-"+filepath.Base(decisionPath))
	writeWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionFixture(t, preconditionPath)
	if err := runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecision([]string{preconditionPath, decisionPath}); err != nil {
		t.Fatalf("write weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision fixture: %v", err)
	}
}
