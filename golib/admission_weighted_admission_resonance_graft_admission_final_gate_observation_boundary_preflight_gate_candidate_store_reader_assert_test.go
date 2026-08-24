package main

import (
	"path/filepath"
	"strings"
	"testing"
)

func TestAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderAssert(t *testing.T) {
	dir := t.TempDir()

	requireBoundaryAssertError(t, runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderAssert(nil), "usage: --admission-live-route-weighted-admission-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-candidate-store-reader-assert REPORT")
	requireBoundaryAssertError(t, runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderAssert([]string{"reader.json", "extra"}), "usage: --admission-live-route-weighted-admission-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-candidate-store-reader-assert REPORT")
	requireBoundaryAssertError(t, runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderAssert([]string{"  "}), "weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader path missing")
	requireBoundaryAssertError(t, runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderAssert([]string{filepath.Join(dir, "missing.json")}), "weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader not written")

	emptyPath := filepath.Join(dir, "empty.json")
	writeWeightedReadinessFixture(t, emptyPath, "")
	requireBoundaryAssertError(t, runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderAssert([]string{emptyPath}), "weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader not written")

	invalidPath := filepath.Join(dir, "invalid.json")
	writeWeightedReadinessFixture(t, invalidPath, "{")
	err := runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderAssert([]string{invalidPath})
	if err == nil || !strings.HasPrefix(err.Error(), "weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader JSON invalid:") {
		t.Fatalf("expected invalid JSON error, got %v", err)
	}

	missingSchemaPath := filepath.Join(dir, "missing_schema.json")
	writeWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderFixture(t, missingSchemaPath)
	writeWeightedReadinessFixture(t, missingSchemaPath, stringsReplaceFirst(readText(t, missingSchemaPath), `"schema": "arianna.live_route_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader.v1",`, ""))
	requireBoundaryAssertError(t, runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderAssert([]string{missingSchemaPath}), "weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader schema missing")

	badSchemaPath := filepath.Join(dir, "bad_schema.json")
	writeWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderFixture(t, badSchemaPath)
	writeWeightedReadinessFixture(t, badSchemaPath, stringsReplaceFirst(readText(t, badSchemaPath), `"schema": "arianna.live_route_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader.v1"`, `"schema": "arianna.live_route_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader.v0"`))
	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderAssert([]string{badSchemaPath}),
		`weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader schema mismatch: got "arianna.live_route_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader.v0" want "`+admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderSchema+`"`,
	)

	validPath := filepath.Join(dir, "valid.json")
	writeWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderFixture(t, validPath)
	if err := runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderAssert([]string{validPath}); err != nil {
		t.Fatalf("valid weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader rejected: %v", err)
	}

	badStatusPath := filepath.Join(dir, "bad_status.json")
	writeWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderFixture(t, badStatusPath)
	writeWeightedReadinessFixture(t, badStatusPath, stringsReplaceFirst(readText(t, badStatusPath), `"status": "shadow_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_read_back_dry_run"`, `"status": "open"`))
	requireBoundaryAssertError(t, runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderAssert([]string{badStatusPath}), `weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader status mismatch: got "open" want "shadow_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_read_back_dry_run"`)

	notReadyPath := filepath.Join(dir, "not_ready.json")
	writeWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderFixture(t, notReadyPath)
	writeWeightedReadinessFixture(t, notReadyPath, stringsReplaceFirst(readText(t, notReadyPath), `"weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_ready": true`, `"weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_ready": false`))
	requireBoundaryAssertError(t, runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderAssert([]string{notReadyPath}), "weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_ready not ready")

	openedLivePath := filepath.Join(dir, "opened_live.json")
	writeWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderFixture(t, openedLivePath)
	writeWeightedReadinessFixture(t, openedLivePath, stringsReplaceFirst(readText(t, openedLivePath), `"live_admission_enabled": false`, `"live_admission_enabled": true`))
	requireBoundaryAssertError(t, runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderAssert([]string{openedLivePath}), "weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader opened live_admission_enabled")

	openedLedgerPath := filepath.Join(dir, "opened_ledger.json")
	writeWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderFixture(t, openedLedgerPath)
	writeWeightedReadinessFixture(t, openedLedgerPath, stringsReplaceFirst(readText(t, openedLedgerPath), `"ledger_append_allowed": false`, `"ledger_append_allowed": true`))
	requireBoundaryAssertError(t, runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderAssert([]string{openedLedgerPath}), "weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader opened ledger_append_allowed")

	badSourcePath := filepath.Join(dir, "bad_source.json")
	writeWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderFixture(t, badSourcePath)
	writeWeightedReadinessFixture(t, badSourcePath, stringsReplaceFirst(readText(t, badSourcePath), `"source_schema": "arianna.live_route_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store.v1"`, `"source_schema": "arianna.live_route_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store.v0"`))
	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderAssert([]string{badSourcePath}),
		`weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader source_schema mismatch: got "arianna.live_route_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store.v0" want "`+admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreSchema+`"`,
	)

	badSourceShapePath := filepath.Join(dir, "bad_source_shape.json")
	writeWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderFixture(t, badSourceShapePath)
	writeWeightedReadinessFixture(t, badSourceShapePath, stringsReplaceFirst(readText(t, badSourceShapePath), `"source_store_kind": "shadow_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store"`, `"source_store_kind": "open"`))
	requireBoundaryAssertError(t, runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderAssert([]string{badSourceShapePath}), "weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader source store shape mismatch")

	openedSourcePath := filepath.Join(dir, "opened_source.json")
	writeWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderFixture(t, openedSourcePath)
	writeWeightedReadinessFixture(t, openedSourcePath, stringsReplaceFirst(readText(t, openedSourcePath), `"source_store_ledger_append_allowed": false`, `"source_store_ledger_append_allowed": true`))
	requireBoundaryAssertError(t, runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderAssert([]string{openedSourcePath}), "weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader opened source_store_ledger_append_allowed")

	badReaderHashPath := filepath.Join(dir, "bad_reader_hash.json")
	writeWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderFixture(t, badReaderHashPath)
	writeWeightedReadinessFixture(t, badReaderHashPath, stringsReplaceFirst(readText(t, badReaderHashPath), `"reader_hash": "weighted-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-candidate-store-reader-`, `"reader_hash": "weighted-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-candidate-store-reader-bad`))
	requireBoundaryAssertError(t, runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderAssert([]string{badReaderHashPath}), "weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader reader_hash mismatch")

	badBodyTargetPath := filepath.Join(dir, "bad_body_target.json")
	writeWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderFixture(t, badBodyTargetPath)
	writeWeightedReadinessFixture(t, badBodyTargetPath, stringsReplaceFirst(readText(t, badBodyTargetPath), `"body_target": "none"`, `"body_target": "live"`))
	requireBoundaryAssertError(t, runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderAssert([]string{badBodyTargetPath}), `weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader body_target mismatch: got "live" want "none"`)
}

func writeWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderFixture(t *testing.T, readerPath string) {
	t.Helper()
	dir := filepath.Dir(readerPath)
	storePath := filepath.Join(dir, "store-"+filepath.Base(readerPath))
	writeWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreFixture(t, storePath)
	if err := runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReader([]string{storePath, readerPath}); err != nil {
		t.Fatalf("write weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader fixture: %v", err)
	}
}
