package main

import (
	"path/filepath"
	"strings"
	"testing"
)

func TestAdmissionLiveRouteWeightedAdmissionResonanceGraftCandidateStoreAssert(t *testing.T) {
	dir := t.TempDir()

	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionResonanceGraftCandidateStoreAssert(nil),
		"usage: --admission-live-route-weighted-admission-resonance-graft-candidate-store-assert REPORT",
	)
	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionResonanceGraftCandidateStoreAssert([]string{"store.json", "extra"}),
		"usage: --admission-live-route-weighted-admission-resonance-graft-candidate-store-assert REPORT",
	)
	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionResonanceGraftCandidateStoreAssert([]string{"  "}),
		"weighted admission resonance graft candidate store path missing",
	)
	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionResonanceGraftCandidateStoreAssert([]string{filepath.Join(dir, "missing.json")}),
		"weighted admission resonance graft candidate store not written",
	)

	emptyPath := filepath.Join(dir, "empty.json")
	writeWeightedReadinessFixture(t, emptyPath, "")
	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionResonanceGraftCandidateStoreAssert([]string{emptyPath}),
		"weighted admission resonance graft candidate store not written",
	)

	invalidPath := filepath.Join(dir, "invalid.json")
	writeWeightedReadinessFixture(t, invalidPath, "{")
	err := runAdmissionLiveRouteWeightedAdmissionResonanceGraftCandidateStoreAssert([]string{invalidPath})
	if err == nil || !strings.HasPrefix(err.Error(), "weighted admission resonance graft candidate store JSON invalid:") {
		t.Fatalf("expected invalid JSON error, got %v", err)
	}

	missingSchemaPath := filepath.Join(dir, "missing_schema.json")
	writeWeightedAdmissionResonanceGraftCandidateStoreFixture(t, missingSchemaPath)
	storeText := readText(t, missingSchemaPath)
	writeWeightedReadinessFixture(t, missingSchemaPath, stringsReplaceFirst(storeText, `"schema": "arianna.live_route_weighted_admission_resonance_graft_candidate_store.v1",`, ""))
	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionResonanceGraftCandidateStoreAssert([]string{missingSchemaPath}),
		"weighted admission resonance graft candidate store schema missing",
	)

	badSchemaPath := filepath.Join(dir, "bad_schema.json")
	writeWeightedAdmissionResonanceGraftCandidateStoreFixture(t, badSchemaPath)
	writeWeightedReadinessFixture(t, badSchemaPath, stringsReplaceFirst(readText(t, badSchemaPath), `"schema": "arianna.live_route_weighted_admission_resonance_graft_candidate_store.v1"`, `"schema": "arianna.live_route_weighted_admission_resonance_graft_candidate_store.v0"`))
	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionResonanceGraftCandidateStoreAssert([]string{badSchemaPath}),
		`weighted admission resonance graft candidate store schema mismatch: got "arianna.live_route_weighted_admission_resonance_graft_candidate_store.v0" want "`+admissionLiveRouteWeightedAdmissionResonanceGraftCandidateStoreSchema+`"`,
	)

	validPath := filepath.Join(dir, "valid.json")
	writeWeightedAdmissionResonanceGraftCandidateStoreFixture(t, validPath)
	if err := runAdmissionLiveRouteWeightedAdmissionResonanceGraftCandidateStoreAssert([]string{validPath}); err != nil {
		t.Fatalf("valid weighted admission resonance graft candidate store rejected: %v", err)
	}

	badStatusPath := filepath.Join(dir, "bad_status.json")
	writeWeightedAdmissionResonanceGraftCandidateStoreFixture(t, badStatusPath)
	writeWeightedReadinessFixture(t, badStatusPath, stringsReplaceFirst(readText(t, badStatusPath), `"status": "shadow_graft_candidate_stored_dry_run"`, `"status": "open"`))
	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionResonanceGraftCandidateStoreAssert([]string{badStatusPath}),
		`weighted admission resonance graft candidate store status mismatch: got "open" want "shadow_graft_candidate_stored_dry_run"`,
	)

	notReadyPath := filepath.Join(dir, "not_ready.json")
	writeWeightedAdmissionResonanceGraftCandidateStoreFixture(t, notReadyPath)
	writeWeightedReadinessFixture(t, notReadyPath, stringsReplaceFirst(readText(t, notReadyPath), `"weighted_admission_resonance_graft_candidate_store_ready": true`, `"weighted_admission_resonance_graft_candidate_store_ready": false`))
	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionResonanceGraftCandidateStoreAssert([]string{notReadyPath}),
		"weighted admission resonance graft candidate store weighted_admission_resonance_graft_candidate_store_ready not ready",
	)

	openedPath := filepath.Join(dir, "opened.json")
	writeWeightedAdmissionResonanceGraftCandidateStoreFixture(t, openedPath)
	writeWeightedReadinessFixture(t, openedPath, stringsReplaceFirst(readText(t, openedPath), `"graft_allowed": false`, `"graft_allowed": true`))
	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionResonanceGraftCandidateStoreAssert([]string{openedPath}),
		"weighted admission resonance graft candidate store opened graft_allowed",
	)

	openedLivePath := filepath.Join(dir, "opened_live.json")
	writeWeightedAdmissionResonanceGraftCandidateStoreFixture(t, openedLivePath)
	writeWeightedReadinessFixture(t, openedLivePath, stringsReplaceFirst(readText(t, openedLivePath), `"live_admission_enabled": false`, `"live_admission_enabled": true`))
	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionResonanceGraftCandidateStoreAssert([]string{openedLivePath}),
		"weighted admission resonance graft candidate store opened live_admission_enabled",
	)

	missingPathField := filepath.Join(dir, "missing_path.json")
	writeWeightedAdmissionResonanceGraftCandidateStoreFixture(t, missingPathField)
	candidateReport := filepath.Join(dir, "candidate-"+filepath.Base(missingPathField))
	writeWeightedReadinessFixture(t, missingPathField, stringsReplaceFirst(readText(t, missingPathField), `"source_report": "`+candidateReport+`"`, `"source_report": " "`))
	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionResonanceGraftCandidateStoreAssert([]string{missingPathField}),
		"weighted admission resonance graft candidate store source_report missing",
	)

	badSourcePath := filepath.Join(dir, "bad_source.json")
	writeWeightedAdmissionResonanceGraftCandidateStoreFixture(t, badSourcePath)
	writeWeightedReadinessFixture(t, badSourcePath, stringsReplaceFirst(readText(t, badSourcePath), `"source_schema": "arianna.live_route_weighted_admission_resonance_graft_candidate.v1"`, `"source_schema": "arianna.live_route_weighted_admission_resonance_graft_candidate.v0"`))
	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionResonanceGraftCandidateStoreAssert([]string{badSourcePath}),
		`weighted admission resonance graft candidate store source_schema mismatch: got "arianna.live_route_weighted_admission_resonance_graft_candidate.v0" want "`+admissionLiveRouteWeightedAdmissionResonanceGraftCandidateSchema+`"`,
	)

	badSourceCandidateKindPath := filepath.Join(dir, "bad_source_candidate_kind.json")
	writeWeightedAdmissionResonanceGraftCandidateStoreFixture(t, badSourceCandidateKindPath)
	writeWeightedReadinessFixture(t, badSourceCandidateKindPath, stringsReplaceFirst(readText(t, badSourceCandidateKindPath), `"source_candidate_kind": "shadow_graft_candidate"`, `"source_candidate_kind": "live_graft"`))
	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionResonanceGraftCandidateStoreAssert([]string{badSourceCandidateKindPath}),
		"weighted admission resonance graft candidate store source candidate shape mismatch",
	)

	badSourceCandidateGuardPath := filepath.Join(dir, "bad_source_candidate_guard.json")
	writeWeightedAdmissionResonanceGraftCandidateStoreFixture(t, badSourceCandidateGuardPath)
	writeWeightedReadinessFixture(t, badSourceCandidateGuardPath, stringsReplaceFirst(readText(t, badSourceCandidateGuardPath), `"source_candidate_graft_allowed": false`, `"source_candidate_graft_allowed": true`))
	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionResonanceGraftCandidateStoreAssert([]string{badSourceCandidateGuardPath}),
		"weighted admission resonance graft candidate store opened source_candidate_graft_allowed",
	)

	badSourceGuardPath := filepath.Join(dir, "bad_source_guard.json")
	writeWeightedAdmissionResonanceGraftCandidateStoreFixture(t, badSourceGuardPath)
	writeWeightedReadinessFixture(t, badSourceGuardPath, stringsReplaceFirst(readText(t, badSourceGuardPath), `"source_candidate_raw_dream_text_allowed": false`, `"source_candidate_raw_dream_text_allowed": true`))
	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionResonanceGraftCandidateStoreAssert([]string{badSourceGuardPath}),
		"weighted admission resonance graft candidate store opened source_candidate_raw_dream_text_allowed",
	)

	badStoreHashPath := filepath.Join(dir, "bad_store_hash.json")
	writeWeightedAdmissionResonanceGraftCandidateStoreFixture(t, badStoreHashPath)
	writeWeightedReadinessFixture(t, badStoreHashPath, stringsReplaceFirst(readText(t, badStoreHashPath), `"store_hash": "weighted-resonance-graft-candidate-store-`, `"store_hash": "weighted-resonance-graft-candidate-store-bad`))
	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionResonanceGraftCandidateStoreAssert([]string{badStoreHashPath}),
		"weighted admission resonance graft candidate store store_hash mismatch",
	)

	badBodyTargetPath := filepath.Join(dir, "bad_body_target.json")
	writeWeightedAdmissionResonanceGraftCandidateStoreFixture(t, badBodyTargetPath)
	writeWeightedReadinessFixture(t, badBodyTargetPath, stringsReplaceFirst(readText(t, badBodyTargetPath), `"body_target": "none"`, `"body_target": "live"`))
	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionResonanceGraftCandidateStoreAssert([]string{badBodyTargetPath}),
		`weighted admission resonance graft candidate store body_target mismatch: got "live" want "none"`,
	)
}

func writeWeightedAdmissionResonanceGraftCandidateStoreFixture(t *testing.T, storePath string) {
	t.Helper()
	dir := filepath.Dir(storePath)
	candidatePath := filepath.Join(dir, "candidate-"+filepath.Base(storePath))
	writeWeightedAdmissionResonanceGraftCandidateFixture(t, candidatePath)
	if err := runAdmissionLiveRouteWeightedAdmissionResonanceGraftCandidateStore([]string{candidatePath, storePath}); err != nil {
		t.Fatalf("write weighted admission resonance graft candidate store fixture: %v", err)
	}
}
