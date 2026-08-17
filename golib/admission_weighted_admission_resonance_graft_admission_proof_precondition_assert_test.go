package main

import (
	"path/filepath"
	"strings"
	"testing"
)

func TestAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionProofPreconditionAssert(t *testing.T) {
	dir := t.TempDir()

	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionProofPreconditionAssert(nil),
		"usage: --admission-live-route-weighted-admission-resonance-graft-admission-proof-precondition-assert REPORT",
	)
	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionProofPreconditionAssert([]string{"precondition.json", "extra"}),
		"usage: --admission-live-route-weighted-admission-resonance-graft-admission-proof-precondition-assert REPORT",
	)
	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionProofPreconditionAssert([]string{"  "}),
		"weighted admission resonance graft admission proof precondition path missing",
	)
	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionProofPreconditionAssert([]string{filepath.Join(dir, "missing.json")}),
		"weighted admission resonance graft admission proof precondition not written",
	)

	emptyPath := filepath.Join(dir, "empty.json")
	writeWeightedReadinessFixture(t, emptyPath, "")
	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionProofPreconditionAssert([]string{emptyPath}),
		"weighted admission resonance graft admission proof precondition not written",
	)

	invalidPath := filepath.Join(dir, "invalid.json")
	writeWeightedReadinessFixture(t, invalidPath, "{")
	err := runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionProofPreconditionAssert([]string{invalidPath})
	if err == nil || !strings.HasPrefix(err.Error(), "weighted admission resonance graft admission proof precondition JSON invalid:") {
		t.Fatalf("expected invalid JSON error, got %v", err)
	}

	missingSchemaPath := filepath.Join(dir, "missing_schema.json")
	writeWeightedAdmissionResonanceGraftAdmissionProofPreconditionFixture(t, missingSchemaPath)
	preconditionText := readText(t, missingSchemaPath)
	writeWeightedReadinessFixture(t, missingSchemaPath, stringsReplaceFirst(preconditionText, `"schema": "arianna.live_route_weighted_admission_resonance_graft_admission_proof_precondition.v1",`, ""))
	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionProofPreconditionAssert([]string{missingSchemaPath}),
		"weighted admission resonance graft admission proof precondition schema missing",
	)

	badSchemaPath := filepath.Join(dir, "bad_schema.json")
	writeWeightedAdmissionResonanceGraftAdmissionProofPreconditionFixture(t, badSchemaPath)
	writeWeightedReadinessFixture(t, badSchemaPath, stringsReplaceFirst(readText(t, badSchemaPath), `"schema": "arianna.live_route_weighted_admission_resonance_graft_admission_proof_precondition.v1"`, `"schema": "arianna.live_route_weighted_admission_resonance_graft_admission_proof_precondition.v0"`))
	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionProofPreconditionAssert([]string{badSchemaPath}),
		`weighted admission resonance graft admission proof precondition schema mismatch: got "arianna.live_route_weighted_admission_resonance_graft_admission_proof_precondition.v0" want "`+admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionProofPreconditionSchema+`"`,
	)

	validPath := filepath.Join(dir, "valid.json")
	writeWeightedAdmissionResonanceGraftAdmissionProofPreconditionFixture(t, validPath)
	if err := runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionProofPreconditionAssert([]string{validPath}); err != nil {
		t.Fatalf("valid weighted admission resonance graft admission proof precondition rejected: %v", err)
	}

	badStatusPath := filepath.Join(dir, "bad_status.json")
	writeWeightedAdmissionResonanceGraftAdmissionProofPreconditionFixture(t, badStatusPath)
	writeWeightedReadinessFixture(t, badStatusPath, stringsReplaceFirst(readText(t, badStatusPath), `"status": "shadow_graft_admission_proof_precondition_satisfied_dry_run"`, `"status": "open"`))
	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionProofPreconditionAssert([]string{badStatusPath}),
		`weighted admission resonance graft admission proof precondition status mismatch: got "open" want "shadow_graft_admission_proof_precondition_satisfied_dry_run"`,
	)

	notReadyPath := filepath.Join(dir, "not_ready.json")
	writeWeightedAdmissionResonanceGraftAdmissionProofPreconditionFixture(t, notReadyPath)
	writeWeightedReadinessFixture(t, notReadyPath, stringsReplaceFirst(readText(t, notReadyPath), `"weighted_admission_resonance_graft_admission_proof_precondition_ready": true`, `"weighted_admission_resonance_graft_admission_proof_precondition_ready": false`))
	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionProofPreconditionAssert([]string{notReadyPath}),
		"weighted admission resonance graft admission proof precondition weighted_admission_resonance_graft_admission_proof_precondition_ready not ready",
	)

	openedPath := filepath.Join(dir, "opened.json")
	writeWeightedAdmissionResonanceGraftAdmissionProofPreconditionFixture(t, openedPath)
	writeWeightedReadinessFixture(t, openedPath, stringsReplaceFirst(readText(t, openedPath), `"graft_allowed": false`, `"graft_allowed": true`))
	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionProofPreconditionAssert([]string{openedPath}),
		"weighted admission resonance graft admission proof precondition opened graft_allowed",
	)

	openedLivePath := filepath.Join(dir, "opened_live.json")
	writeWeightedAdmissionResonanceGraftAdmissionProofPreconditionFixture(t, openedLivePath)
	writeWeightedReadinessFixture(t, openedLivePath, stringsReplaceFirst(readText(t, openedLivePath), `"live_admission_enabled": false`, `"live_admission_enabled": true`))
	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionProofPreconditionAssert([]string{openedLivePath}),
		"weighted admission resonance graft admission proof precondition opened live_admission_enabled",
	)

	missingPathField := filepath.Join(dir, "missing_path.json")
	writeWeightedAdmissionResonanceGraftAdmissionProofPreconditionFixture(t, missingPathField)
	proofReport := filepath.Join(dir, "proof-"+filepath.Base(missingPathField))
	writeWeightedReadinessFixture(t, missingPathField, stringsReplaceFirst(readText(t, missingPathField), `"source_report": "`+proofReport+`"`, `"source_report": " "`))
	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionProofPreconditionAssert([]string{missingPathField}),
		"weighted admission resonance graft admission proof precondition source_report missing",
	)

	badSourcePath := filepath.Join(dir, "bad_source.json")
	writeWeightedAdmissionResonanceGraftAdmissionProofPreconditionFixture(t, badSourcePath)
	writeWeightedReadinessFixture(t, badSourcePath, stringsReplaceFirst(readText(t, badSourcePath), `"source_schema": "arianna.live_route_weighted_admission_resonance_graft_admission_proof.v1"`, `"source_schema": "arianna.live_route_weighted_admission_resonance_graft_admission_proof.v0"`))
	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionProofPreconditionAssert([]string{badSourcePath}),
		`weighted admission resonance graft admission proof precondition source_schema mismatch: got "arianna.live_route_weighted_admission_resonance_graft_admission_proof.v0" want "`+admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionProofSchema+`"`,
	)

	badSourceProofKindPath := filepath.Join(dir, "bad_source_proof_kind.json")
	writeWeightedAdmissionResonanceGraftAdmissionProofPreconditionFixture(t, badSourceProofKindPath)
	writeWeightedReadinessFixture(t, badSourceProofKindPath, stringsReplaceFirst(readText(t, badSourceProofKindPath), `"source_proof_kind": "shadow_graft_admission_proof"`, `"source_proof_kind": "live_proof"`))
	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionProofPreconditionAssert([]string{badSourceProofKindPath}),
		"weighted admission resonance graft admission proof precondition source proof shape mismatch",
	)

	badSourceProofGuardPath := filepath.Join(dir, "bad_source_proof_guard.json")
	writeWeightedAdmissionResonanceGraftAdmissionProofPreconditionFixture(t, badSourceProofGuardPath)
	writeWeightedReadinessFixture(t, badSourceProofGuardPath, stringsReplaceFirst(readText(t, badSourceProofGuardPath), `"source_proof_graft_allowed": false`, `"source_proof_graft_allowed": true`))
	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionProofPreconditionAssert([]string{badSourceProofGuardPath}),
		"weighted admission resonance graft admission proof precondition opened source_proof_graft_allowed",
	)

	badSourceGuardPath := filepath.Join(dir, "bad_source_guard.json")
	writeWeightedAdmissionResonanceGraftAdmissionProofPreconditionFixture(t, badSourceGuardPath)
	writeWeightedReadinessFixture(t, badSourceGuardPath, stringsReplaceFirst(readText(t, badSourceGuardPath), `"source_proof_raw_dream_text_allowed": false`, `"source_proof_raw_dream_text_allowed": true`))
	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionProofPreconditionAssert([]string{badSourceGuardPath}),
		"weighted admission resonance graft admission proof precondition opened source_proof_raw_dream_text_allowed",
	)

	badPreconditionHashPath := filepath.Join(dir, "bad_precondition_hash.json")
	writeWeightedAdmissionResonanceGraftAdmissionProofPreconditionFixture(t, badPreconditionHashPath)
	writeWeightedReadinessFixture(t, badPreconditionHashPath, stringsReplaceFirst(readText(t, badPreconditionHashPath), `"precondition_hash": "weighted-resonance-graft-admission-proof-precondition-`, `"precondition_hash": "weighted-resonance-graft-admission-proof-precondition-bad`))
	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionProofPreconditionAssert([]string{badPreconditionHashPath}),
		"weighted admission resonance graft admission proof precondition precondition_hash mismatch",
	)

	badBodyTargetPath := filepath.Join(dir, "bad_body_target.json")
	writeWeightedAdmissionResonanceGraftAdmissionProofPreconditionFixture(t, badBodyTargetPath)
	writeWeightedReadinessFixture(t, badBodyTargetPath, stringsReplaceFirst(readText(t, badBodyTargetPath), `"body_target": "none"`, `"body_target": "live"`))
	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionProofPreconditionAssert([]string{badBodyTargetPath}),
		`weighted admission resonance graft admission proof precondition body_target mismatch: got "live" want "none"`,
	)
}

func writeWeightedAdmissionResonanceGraftAdmissionProofPreconditionFixture(t *testing.T, preconditionPath string) {
	t.Helper()
	dir := filepath.Dir(preconditionPath)
	proofPath := filepath.Join(dir, "proof-"+filepath.Base(preconditionPath))
	writeWeightedAdmissionResonanceGraftAdmissionProofFixture(t, proofPath)
	if err := runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionProofPrecondition([]string{proofPath, preconditionPath}); err != nil {
		t.Fatalf("write weighted admission resonance graft admission proof precondition fixture: %v", err)
	}
}
