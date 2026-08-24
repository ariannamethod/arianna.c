package main

import (
	"path/filepath"
	"strings"
	"testing"
)

func TestAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofAssert(t *testing.T) {
	dir := t.TempDir()

	requireBoundaryAssertError(t, runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofAssert(nil), "usage: --admission-live-route-weighted-admission-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-candidate-store-reader-proof-assert REPORT")
	requireBoundaryAssertError(t, runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofAssert([]string{"proof.json", "extra"}), "usage: --admission-live-route-weighted-admission-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-candidate-store-reader-proof-assert REPORT")
	requireBoundaryAssertError(t, runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofAssert([]string{"  "}), "weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof path missing")
	requireBoundaryAssertError(t, runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofAssert([]string{filepath.Join(dir, "missing.json")}), "weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof not written")

	emptyPath := filepath.Join(dir, "empty.json")
	writeWeightedReadinessFixture(t, emptyPath, "")
	requireBoundaryAssertError(t, runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofAssert([]string{emptyPath}), "weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof not written")

	invalidPath := filepath.Join(dir, "invalid.json")
	writeWeightedReadinessFixture(t, invalidPath, "{")
	err := runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofAssert([]string{invalidPath})
	if err == nil || !strings.HasPrefix(err.Error(), "weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof JSON invalid:") {
		t.Fatalf("expected invalid JSON error, got %v", err)
	}

	missingSchemaPath := filepath.Join(dir, "missing_schema.json")
	writeWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofFixture(t, missingSchemaPath)
	writeWeightedReadinessFixture(t, missingSchemaPath, stringsReplaceFirst(readText(t, missingSchemaPath), `"schema": "arianna.live_route_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof.v1",`, ""))
	requireBoundaryAssertError(t, runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofAssert([]string{missingSchemaPath}), "weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof schema missing")

	badSchemaPath := filepath.Join(dir, "bad_schema.json")
	writeWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofFixture(t, badSchemaPath)
	writeWeightedReadinessFixture(t, badSchemaPath, stringsReplaceFirst(readText(t, badSchemaPath), `"schema": "arianna.live_route_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof.v1"`, `"schema": "arianna.live_route_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof.v0"`))
	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofAssert([]string{badSchemaPath}),
		`weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof schema mismatch: got "arianna.live_route_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof.v0" want "`+admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofSchema+`"`,
	)

	validPath := filepath.Join(dir, "valid.json")
	writeWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofFixture(t, validPath)
	if err := runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofAssert([]string{validPath}); err != nil {
		t.Fatalf("valid weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof rejected: %v", err)
	}

	badStatusPath := filepath.Join(dir, "bad_status.json")
	writeWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofFixture(t, badStatusPath)
	writeWeightedReadinessFixture(t, badStatusPath, stringsReplaceFirst(readText(t, badStatusPath), `"status": "shadow_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_ready_dry_run"`, `"status": "open"`))
	requireBoundaryAssertError(t, runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofAssert([]string{badStatusPath}), `weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof status mismatch: got "open" want "shadow_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_ready_dry_run"`)

	notReadyPath := filepath.Join(dir, "not_ready.json")
	writeWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofFixture(t, notReadyPath)
	writeWeightedReadinessFixture(t, notReadyPath, stringsReplaceFirst(readText(t, notReadyPath), `"weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_ready": true`, `"weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_ready": false`))
	requireBoundaryAssertError(t, runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofAssert([]string{notReadyPath}), "weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_ready not ready")

	openedLivePath := filepath.Join(dir, "opened_live.json")
	writeWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofFixture(t, openedLivePath)
	writeWeightedReadinessFixture(t, openedLivePath, stringsReplaceFirst(readText(t, openedLivePath), `"live_admission_enabled": false`, `"live_admission_enabled": true`))
	requireBoundaryAssertError(t, runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofAssert([]string{openedLivePath}), "weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof opened live_admission_enabled")

	openedSourcePath := filepath.Join(dir, "opened_source.json")
	writeWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofFixture(t, openedSourcePath)
	writeWeightedReadinessFixture(t, openedSourcePath, stringsReplaceFirst(readText(t, openedSourcePath), `"source_reader_ledger_append_allowed": false`, `"source_reader_ledger_append_allowed": true`))
	requireBoundaryAssertError(t, runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofAssert([]string{openedSourcePath}), "weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof opened source_reader_ledger_append_allowed")

	badSourcePath := filepath.Join(dir, "bad_source.json")
	writeWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofFixture(t, badSourcePath)
	writeWeightedReadinessFixture(t, badSourcePath, stringsReplaceFirst(readText(t, badSourcePath), `"source_schema": "arianna.live_route_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader.v1"`, `"source_schema": "arianna.live_route_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader.v0"`))
	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofAssert([]string{badSourcePath}),
		`weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof source_schema mismatch: got "arianna.live_route_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader.v0" want "`+admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderSchema+`"`,
	)

	badProofHashPath := filepath.Join(dir, "bad_proof_hash.json")
	writeWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofFixture(t, badProofHashPath)
	writeWeightedReadinessFixture(t, badProofHashPath, stringsReplaceFirst(readText(t, badProofHashPath), `"proof_hash": "weighted-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-candidate-store-reader-proof-`, `"proof_hash": "weighted-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-candidate-store-reader-proof-bad`))
	requireBoundaryAssertError(t, runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofAssert([]string{badProofHashPath}), "weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof proof_hash mismatch")

	badBodyTargetPath := filepath.Join(dir, "bad_body_target.json")
	writeWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofFixture(t, badBodyTargetPath)
	writeWeightedReadinessFixture(t, badBodyTargetPath, stringsReplaceFirst(readText(t, badBodyTargetPath), `"body_target": "none"`, `"body_target": "live"`))
	requireBoundaryAssertError(t, runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofAssert([]string{badBodyTargetPath}), "weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof body target mismatch")
}

func writeWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofFixture(t *testing.T, proofPath string) {
	t.Helper()
	dir := filepath.Dir(proofPath)
	readerPath := filepath.Join(dir, "reader-"+filepath.Base(proofPath))
	writeWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderFixture(t, readerPath)
	if err := runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProof([]string{readerPath, proofPath}); err != nil {
		t.Fatalf("write weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof fixture: %v", err)
	}
}
