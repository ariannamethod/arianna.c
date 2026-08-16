package main

import (
	"path/filepath"
	"strings"
	"testing"
)

func TestAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionProofAssert(t *testing.T) {
	dir := t.TempDir()

	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionProofAssert(nil),
		"usage: --admission-live-route-weighted-admission-resonance-graft-admission-proof-assert REPORT",
	)
	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionProofAssert([]string{"proof.json", "extra"}),
		"usage: --admission-live-route-weighted-admission-resonance-graft-admission-proof-assert REPORT",
	)
	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionProofAssert([]string{"  "}),
		"weighted admission resonance graft admission proof path missing",
	)
	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionProofAssert([]string{filepath.Join(dir, "missing.json")}),
		"weighted admission resonance graft admission proof not written",
	)

	emptyPath := filepath.Join(dir, "empty.json")
	writeWeightedReadinessFixture(t, emptyPath, "")
	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionProofAssert([]string{emptyPath}),
		"weighted admission resonance graft admission proof not written",
	)

	invalidPath := filepath.Join(dir, "invalid.json")
	writeWeightedReadinessFixture(t, invalidPath, "{")
	err := runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionProofAssert([]string{invalidPath})
	if err == nil || !strings.HasPrefix(err.Error(), "weighted admission resonance graft admission proof JSON invalid:") {
		t.Fatalf("expected invalid JSON error, got %v", err)
	}

	missingSchemaPath := filepath.Join(dir, "missing_schema.json")
	writeWeightedAdmissionResonanceGraftAdmissionProofFixture(t, missingSchemaPath)
	proofText := readText(t, missingSchemaPath)
	writeWeightedReadinessFixture(t, missingSchemaPath, stringsReplaceFirst(proofText, `"schema": "arianna.live_route_weighted_admission_resonance_graft_admission_proof.v1",`, ""))
	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionProofAssert([]string{missingSchemaPath}),
		"weighted admission resonance graft admission proof schema missing",
	)

	badSchemaPath := filepath.Join(dir, "bad_schema.json")
	writeWeightedAdmissionResonanceGraftAdmissionProofFixture(t, badSchemaPath)
	writeWeightedReadinessFixture(t, badSchemaPath, stringsReplaceFirst(readText(t, badSchemaPath), `"schema": "arianna.live_route_weighted_admission_resonance_graft_admission_proof.v1"`, `"schema": "arianna.live_route_weighted_admission_resonance_graft_admission_proof.v0"`))
	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionProofAssert([]string{badSchemaPath}),
		`weighted admission resonance graft admission proof schema mismatch: got "arianna.live_route_weighted_admission_resonance_graft_admission_proof.v0" want "`+admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionProofSchema+`"`,
	)

	validPath := filepath.Join(dir, "valid.json")
	writeWeightedAdmissionResonanceGraftAdmissionProofFixture(t, validPath)
	if err := runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionProofAssert([]string{validPath}); err != nil {
		t.Fatalf("valid weighted admission resonance graft admission proof rejected: %v", err)
	}

	badStatusPath := filepath.Join(dir, "bad_status.json")
	writeWeightedAdmissionResonanceGraftAdmissionProofFixture(t, badStatusPath)
	writeWeightedReadinessFixture(t, badStatusPath, stringsReplaceFirst(readText(t, badStatusPath), `"status": "shadow_graft_admission_proof_ready_dry_run"`, `"status": "open"`))
	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionProofAssert([]string{badStatusPath}),
		`weighted admission resonance graft admission proof status mismatch: got "open" want "shadow_graft_admission_proof_ready_dry_run"`,
	)

	notReadyPath := filepath.Join(dir, "not_ready.json")
	writeWeightedAdmissionResonanceGraftAdmissionProofFixture(t, notReadyPath)
	writeWeightedReadinessFixture(t, notReadyPath, stringsReplaceFirst(readText(t, notReadyPath), `"weighted_admission_resonance_graft_admission_proof_ready": true`, `"weighted_admission_resonance_graft_admission_proof_ready": false`))
	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionProofAssert([]string{notReadyPath}),
		"weighted admission resonance graft admission proof weighted_admission_resonance_graft_admission_proof_ready not ready",
	)

	openedPath := filepath.Join(dir, "opened.json")
	writeWeightedAdmissionResonanceGraftAdmissionProofFixture(t, openedPath)
	writeWeightedReadinessFixture(t, openedPath, stringsReplaceFirst(readText(t, openedPath), `"graft_allowed": false`, `"graft_allowed": true`))
	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionProofAssert([]string{openedPath}),
		"weighted admission resonance graft admission proof opened graft_allowed",
	)

	openedLivePath := filepath.Join(dir, "opened_live.json")
	writeWeightedAdmissionResonanceGraftAdmissionProofFixture(t, openedLivePath)
	writeWeightedReadinessFixture(t, openedLivePath, stringsReplaceFirst(readText(t, openedLivePath), `"live_admission_enabled": false`, `"live_admission_enabled": true`))
	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionProofAssert([]string{openedLivePath}),
		"weighted admission resonance graft admission proof opened live_admission_enabled",
	)

	missingPathField := filepath.Join(dir, "missing_path.json")
	writeWeightedAdmissionResonanceGraftAdmissionProofFixture(t, missingPathField)
	readerReport := filepath.Join(dir, "reader-"+filepath.Base(missingPathField))
	writeWeightedReadinessFixture(t, missingPathField, stringsReplaceFirst(readText(t, missingPathField), `"source_report": "`+readerReport+`"`, `"source_report": " "`))
	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionProofAssert([]string{missingPathField}),
		"weighted admission resonance graft admission proof source_report missing",
	)

	badSourcePath := filepath.Join(dir, "bad_source.json")
	writeWeightedAdmissionResonanceGraftAdmissionProofFixture(t, badSourcePath)
	writeWeightedReadinessFixture(t, badSourcePath, stringsReplaceFirst(readText(t, badSourcePath), `"source_schema": "arianna.live_route_weighted_admission_resonance_graft_candidate_store_reader.v1"`, `"source_schema": "arianna.live_route_weighted_admission_resonance_graft_candidate_store_reader.v0"`))
	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionProofAssert([]string{badSourcePath}),
		`weighted admission resonance graft admission proof source_schema mismatch: got "arianna.live_route_weighted_admission_resonance_graft_candidate_store_reader.v0" want "`+admissionLiveRouteWeightedAdmissionResonanceGraftCandidateStoreReaderSchema+`"`,
	)

	badSourceReaderKindPath := filepath.Join(dir, "bad_source_reader_kind.json")
	writeWeightedAdmissionResonanceGraftAdmissionProofFixture(t, badSourceReaderKindPath)
	writeWeightedReadinessFixture(t, badSourceReaderKindPath, stringsReplaceFirst(readText(t, badSourceReaderKindPath), `"source_reader_kind": "shadow_graft_candidate_store_reader"`, `"source_reader_kind": "live_reader"`))
	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionProofAssert([]string{badSourceReaderKindPath}),
		"weighted admission resonance graft admission proof source reader shape mismatch",
	)

	badSourceReaderGuardPath := filepath.Join(dir, "bad_source_reader_guard.json")
	writeWeightedAdmissionResonanceGraftAdmissionProofFixture(t, badSourceReaderGuardPath)
	writeWeightedReadinessFixture(t, badSourceReaderGuardPath, stringsReplaceFirst(readText(t, badSourceReaderGuardPath), `"source_reader_graft_allowed": false`, `"source_reader_graft_allowed": true`))
	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionProofAssert([]string{badSourceReaderGuardPath}),
		"weighted admission resonance graft admission proof opened source_reader_graft_allowed",
	)

	badSourceGuardPath := filepath.Join(dir, "bad_source_guard.json")
	writeWeightedAdmissionResonanceGraftAdmissionProofFixture(t, badSourceGuardPath)
	writeWeightedReadinessFixture(t, badSourceGuardPath, stringsReplaceFirst(readText(t, badSourceGuardPath), `"source_reader_raw_dream_text_allowed": false`, `"source_reader_raw_dream_text_allowed": true`))
	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionProofAssert([]string{badSourceGuardPath}),
		"weighted admission resonance graft admission proof opened source_reader_raw_dream_text_allowed",
	)

	badProofHashPath := filepath.Join(dir, "bad_proof_hash.json")
	writeWeightedAdmissionResonanceGraftAdmissionProofFixture(t, badProofHashPath)
	writeWeightedReadinessFixture(t, badProofHashPath, stringsReplaceFirst(readText(t, badProofHashPath), `"proof_hash": "weighted-resonance-graft-admission-proof-`, `"proof_hash": "weighted-resonance-graft-admission-proof-bad`))
	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionProofAssert([]string{badProofHashPath}),
		"weighted admission resonance graft admission proof proof_hash mismatch",
	)

	badBodyTargetPath := filepath.Join(dir, "bad_body_target.json")
	writeWeightedAdmissionResonanceGraftAdmissionProofFixture(t, badBodyTargetPath)
	writeWeightedReadinessFixture(t, badBodyTargetPath, stringsReplaceFirst(readText(t, badBodyTargetPath), `"body_target": "none"`, `"body_target": "live"`))
	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionProofAssert([]string{badBodyTargetPath}),
		`weighted admission resonance graft admission proof body_target mismatch: got "live" want "none"`,
	)
}

func writeWeightedAdmissionResonanceGraftAdmissionProofFixture(t *testing.T, proofPath string) {
	t.Helper()
	dir := filepath.Dir(proofPath)
	readerPath := filepath.Join(dir, "reader-"+filepath.Base(proofPath))
	writeWeightedAdmissionResonanceGraftCandidateStoreReaderFixture(t, readerPath)
	if err := runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionProof([]string{readerPath, proofPath}); err != nil {
		t.Fatalf("write weighted admission resonance graft admission proof fixture: %v", err)
	}
}
