package main

import (
	"path/filepath"
	"strings"
	"testing"
)

func TestAdmissionLiveRouteWeightedAdmissionResonanceGraftCandidateStoreReaderAssert(t *testing.T) {
	dir := t.TempDir()

	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionResonanceGraftCandidateStoreReaderAssert(nil),
		"usage: --admission-live-route-weighted-admission-resonance-graft-candidate-store-reader-assert REPORT",
	)
	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionResonanceGraftCandidateStoreReaderAssert([]string{"reader.json", "extra"}),
		"usage: --admission-live-route-weighted-admission-resonance-graft-candidate-store-reader-assert REPORT",
	)
	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionResonanceGraftCandidateStoreReaderAssert([]string{"  "}),
		"weighted admission resonance graft candidate store reader path missing",
	)
	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionResonanceGraftCandidateStoreReaderAssert([]string{filepath.Join(dir, "missing.json")}),
		"weighted admission resonance graft candidate store reader not written",
	)

	emptyPath := filepath.Join(dir, "empty.json")
	writeWeightedReadinessFixture(t, emptyPath, "")
	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionResonanceGraftCandidateStoreReaderAssert([]string{emptyPath}),
		"weighted admission resonance graft candidate store reader not written",
	)

	invalidPath := filepath.Join(dir, "invalid.json")
	writeWeightedReadinessFixture(t, invalidPath, "{")
	err := runAdmissionLiveRouteWeightedAdmissionResonanceGraftCandidateStoreReaderAssert([]string{invalidPath})
	if err == nil || !strings.HasPrefix(err.Error(), "weighted admission resonance graft candidate store reader JSON invalid:") {
		t.Fatalf("expected invalid JSON error, got %v", err)
	}

	missingSchemaPath := filepath.Join(dir, "missing_schema.json")
	writeWeightedAdmissionResonanceGraftCandidateStoreReaderFixture(t, missingSchemaPath)
	readerText := readText(t, missingSchemaPath)
	writeWeightedReadinessFixture(t, missingSchemaPath, stringsReplaceFirst(readerText, `"schema": "arianna.live_route_weighted_admission_resonance_graft_candidate_store_reader.v1",`, ""))
	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionResonanceGraftCandidateStoreReaderAssert([]string{missingSchemaPath}),
		"weighted admission resonance graft candidate store reader schema missing",
	)

	badSchemaPath := filepath.Join(dir, "bad_schema.json")
	writeWeightedAdmissionResonanceGraftCandidateStoreReaderFixture(t, badSchemaPath)
	writeWeightedReadinessFixture(t, badSchemaPath, stringsReplaceFirst(readText(t, badSchemaPath), `"schema": "arianna.live_route_weighted_admission_resonance_graft_candidate_store_reader.v1"`, `"schema": "arianna.live_route_weighted_admission_resonance_graft_candidate_store_reader.v0"`))
	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionResonanceGraftCandidateStoreReaderAssert([]string{badSchemaPath}),
		`weighted admission resonance graft candidate store reader schema mismatch: got "arianna.live_route_weighted_admission_resonance_graft_candidate_store_reader.v0" want "`+admissionLiveRouteWeightedAdmissionResonanceGraftCandidateStoreReaderSchema+`"`,
	)

	validPath := filepath.Join(dir, "valid.json")
	writeWeightedAdmissionResonanceGraftCandidateStoreReaderFixture(t, validPath)
	if err := runAdmissionLiveRouteWeightedAdmissionResonanceGraftCandidateStoreReaderAssert([]string{validPath}); err != nil {
		t.Fatalf("valid weighted admission resonance graft candidate store reader rejected: %v", err)
	}

	badStatusPath := filepath.Join(dir, "bad_status.json")
	writeWeightedAdmissionResonanceGraftCandidateStoreReaderFixture(t, badStatusPath)
	writeWeightedReadinessFixture(t, badStatusPath, stringsReplaceFirst(readText(t, badStatusPath), `"status": "shadow_graft_candidate_store_read_back_dry_run"`, `"status": "open"`))
	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionResonanceGraftCandidateStoreReaderAssert([]string{badStatusPath}),
		`weighted admission resonance graft candidate store reader status mismatch: got "open" want "shadow_graft_candidate_store_read_back_dry_run"`,
	)

	notReadyPath := filepath.Join(dir, "not_ready.json")
	writeWeightedAdmissionResonanceGraftCandidateStoreReaderFixture(t, notReadyPath)
	writeWeightedReadinessFixture(t, notReadyPath, stringsReplaceFirst(readText(t, notReadyPath), `"weighted_admission_resonance_graft_candidate_store_reader_ready": true`, `"weighted_admission_resonance_graft_candidate_store_reader_ready": false`))
	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionResonanceGraftCandidateStoreReaderAssert([]string{notReadyPath}),
		"weighted admission resonance graft candidate store reader weighted_admission_resonance_graft_candidate_store_reader_ready not ready",
	)

	openedPath := filepath.Join(dir, "opened.json")
	writeWeightedAdmissionResonanceGraftCandidateStoreReaderFixture(t, openedPath)
	writeWeightedReadinessFixture(t, openedPath, stringsReplaceFirst(readText(t, openedPath), `"graft_allowed": false`, `"graft_allowed": true`))
	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionResonanceGraftCandidateStoreReaderAssert([]string{openedPath}),
		"weighted admission resonance graft candidate store reader opened graft_allowed",
	)

	openedLivePath := filepath.Join(dir, "opened_live.json")
	writeWeightedAdmissionResonanceGraftCandidateStoreReaderFixture(t, openedLivePath)
	writeWeightedReadinessFixture(t, openedLivePath, stringsReplaceFirst(readText(t, openedLivePath), `"live_admission_enabled": false`, `"live_admission_enabled": true`))
	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionResonanceGraftCandidateStoreReaderAssert([]string{openedLivePath}),
		"weighted admission resonance graft candidate store reader opened live_admission_enabled",
	)

	missingPathField := filepath.Join(dir, "missing_path.json")
	writeWeightedAdmissionResonanceGraftCandidateStoreReaderFixture(t, missingPathField)
	storeReport := filepath.Join(dir, "store-"+filepath.Base(missingPathField))
	writeWeightedReadinessFixture(t, missingPathField, stringsReplaceFirst(readText(t, missingPathField), `"source_report": "`+storeReport+`"`, `"source_report": " "`))
	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionResonanceGraftCandidateStoreReaderAssert([]string{missingPathField}),
		"weighted admission resonance graft candidate store reader source_report missing",
	)

	badSourcePath := filepath.Join(dir, "bad_source.json")
	writeWeightedAdmissionResonanceGraftCandidateStoreReaderFixture(t, badSourcePath)
	writeWeightedReadinessFixture(t, badSourcePath, stringsReplaceFirst(readText(t, badSourcePath), `"source_schema": "arianna.live_route_weighted_admission_resonance_graft_candidate_store.v1"`, `"source_schema": "arianna.live_route_weighted_admission_resonance_graft_candidate_store.v0"`))
	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionResonanceGraftCandidateStoreReaderAssert([]string{badSourcePath}),
		`weighted admission resonance graft candidate store reader source_schema mismatch: got "arianna.live_route_weighted_admission_resonance_graft_candidate_store.v0" want "`+admissionLiveRouteWeightedAdmissionResonanceGraftCandidateStoreSchema+`"`,
	)

	badSourceStoreKindPath := filepath.Join(dir, "bad_source_store_kind.json")
	writeWeightedAdmissionResonanceGraftCandidateStoreReaderFixture(t, badSourceStoreKindPath)
	writeWeightedReadinessFixture(t, badSourceStoreKindPath, stringsReplaceFirst(readText(t, badSourceStoreKindPath), `"source_store_kind": "shadow_graft_candidate_store"`, `"source_store_kind": "live_store"`))
	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionResonanceGraftCandidateStoreReaderAssert([]string{badSourceStoreKindPath}),
		"weighted admission resonance graft candidate store reader source store shape mismatch",
	)

	badSourceStoreGuardPath := filepath.Join(dir, "bad_source_store_guard.json")
	writeWeightedAdmissionResonanceGraftCandidateStoreReaderFixture(t, badSourceStoreGuardPath)
	writeWeightedReadinessFixture(t, badSourceStoreGuardPath, stringsReplaceFirst(readText(t, badSourceStoreGuardPath), `"source_store_graft_allowed": false`, `"source_store_graft_allowed": true`))
	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionResonanceGraftCandidateStoreReaderAssert([]string{badSourceStoreGuardPath}),
		"weighted admission resonance graft candidate store reader opened source_store_graft_allowed",
	)

	badSourceGuardPath := filepath.Join(dir, "bad_source_guard.json")
	writeWeightedAdmissionResonanceGraftCandidateStoreReaderFixture(t, badSourceGuardPath)
	writeWeightedReadinessFixture(t, badSourceGuardPath, stringsReplaceFirst(readText(t, badSourceGuardPath), `"source_store_raw_dream_text_allowed": false`, `"source_store_raw_dream_text_allowed": true`))
	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionResonanceGraftCandidateStoreReaderAssert([]string{badSourceGuardPath}),
		"weighted admission resonance graft candidate store reader opened source_store_raw_dream_text_allowed",
	)

	badReaderHashPath := filepath.Join(dir, "bad_reader_hash.json")
	writeWeightedAdmissionResonanceGraftCandidateStoreReaderFixture(t, badReaderHashPath)
	writeWeightedReadinessFixture(t, badReaderHashPath, stringsReplaceFirst(readText(t, badReaderHashPath), `"reader_hash": "weighted-resonance-graft-candidate-store-reader-`, `"reader_hash": "weighted-resonance-graft-candidate-store-reader-bad`))
	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionResonanceGraftCandidateStoreReaderAssert([]string{badReaderHashPath}),
		"weighted admission resonance graft candidate store reader reader_hash mismatch",
	)

	badBodyTargetPath := filepath.Join(dir, "bad_body_target.json")
	writeWeightedAdmissionResonanceGraftCandidateStoreReaderFixture(t, badBodyTargetPath)
	writeWeightedReadinessFixture(t, badBodyTargetPath, stringsReplaceFirst(readText(t, badBodyTargetPath), `"body_target": "none"`, `"body_target": "live"`))
	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionResonanceGraftCandidateStoreReaderAssert([]string{badBodyTargetPath}),
		`weighted admission resonance graft candidate store reader body_target mismatch: got "live" want "none"`,
	)
}

func writeWeightedAdmissionResonanceGraftCandidateStoreReaderFixture(t *testing.T, readerPath string) {
	t.Helper()
	dir := filepath.Dir(readerPath)
	storePath := filepath.Join(dir, "store-"+filepath.Base(readerPath))
	writeWeightedAdmissionResonanceGraftCandidateStoreFixture(t, storePath)
	if err := runAdmissionLiveRouteWeightedAdmissionResonanceGraftCandidateStoreReader([]string{storePath, readerPath}); err != nil {
		t.Fatalf("write weighted admission resonance graft candidate store reader fixture: %v", err)
	}
}
