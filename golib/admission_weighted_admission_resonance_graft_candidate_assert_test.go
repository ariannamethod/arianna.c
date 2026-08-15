package main

import (
	"path/filepath"
	"strings"
	"testing"
)

func TestAdmissionLiveRouteWeightedAdmissionResonanceGraftCandidateAssert(t *testing.T) {
	dir := t.TempDir()

	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionResonanceGraftCandidateAssert(nil),
		"usage: --admission-live-route-weighted-admission-resonance-graft-candidate-assert REPORT",
	)
	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionResonanceGraftCandidateAssert([]string{"candidate.json", "extra"}),
		"usage: --admission-live-route-weighted-admission-resonance-graft-candidate-assert REPORT",
	)
	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionResonanceGraftCandidateAssert([]string{"  "}),
		"weighted admission resonance graft candidate path missing",
	)
	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionResonanceGraftCandidateAssert([]string{filepath.Join(dir, "missing.json")}),
		"weighted admission resonance graft candidate not written",
	)

	emptyPath := filepath.Join(dir, "empty.json")
	writeWeightedReadinessFixture(t, emptyPath, "")
	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionResonanceGraftCandidateAssert([]string{emptyPath}),
		"weighted admission resonance graft candidate not written",
	)

	invalidPath := filepath.Join(dir, "invalid.json")
	writeWeightedReadinessFixture(t, invalidPath, "{")
	err := runAdmissionLiveRouteWeightedAdmissionResonanceGraftCandidateAssert([]string{invalidPath})
	if err == nil || !strings.HasPrefix(err.Error(), "weighted admission resonance graft candidate JSON invalid:") {
		t.Fatalf("expected invalid JSON error, got %v", err)
	}

	missingSchemaPath := filepath.Join(dir, "missing_schema.json")
	writeWeightedAdmissionResonanceGraftCandidateFixture(t, missingSchemaPath)
	candidateText := readText(t, missingSchemaPath)
	writeWeightedReadinessFixture(t, missingSchemaPath, stringsReplaceFirst(candidateText, `"schema": "arianna.live_route_weighted_admission_resonance_graft_candidate.v1",`, ""))
	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionResonanceGraftCandidateAssert([]string{missingSchemaPath}),
		"weighted admission resonance graft candidate schema missing",
	)

	badSchemaPath := filepath.Join(dir, "bad_schema.json")
	writeWeightedAdmissionResonanceGraftCandidateFixture(t, badSchemaPath)
	writeWeightedReadinessFixture(t, badSchemaPath, stringsReplaceFirst(readText(t, badSchemaPath), `"schema": "arianna.live_route_weighted_admission_resonance_graft_candidate.v1"`, `"schema": "arianna.live_route_weighted_admission_resonance_graft_candidate.v0"`))
	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionResonanceGraftCandidateAssert([]string{badSchemaPath}),
		`weighted admission resonance graft candidate schema mismatch: got "arianna.live_route_weighted_admission_resonance_graft_candidate.v0" want "`+admissionLiveRouteWeightedAdmissionResonanceGraftCandidateSchema+`"`,
	)

	validPath := filepath.Join(dir, "valid.json")
	writeWeightedAdmissionResonanceGraftCandidateFixture(t, validPath)
	if err := runAdmissionLiveRouteWeightedAdmissionResonanceGraftCandidateAssert([]string{validPath}); err != nil {
		t.Fatalf("valid weighted admission resonance graft candidate rejected: %v", err)
	}

	badStatusPath := filepath.Join(dir, "bad_status.json")
	writeWeightedAdmissionResonanceGraftCandidateFixture(t, badStatusPath)
	writeWeightedReadinessFixture(t, badStatusPath, stringsReplaceFirst(readText(t, badStatusPath), `"status": "shadow_graft_candidate_ready_dry_run"`, `"status": "open"`))
	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionResonanceGraftCandidateAssert([]string{badStatusPath}),
		`weighted admission resonance graft candidate status mismatch: got "open" want "shadow_graft_candidate_ready_dry_run"`,
	)

	notReadyPath := filepath.Join(dir, "not_ready.json")
	writeWeightedAdmissionResonanceGraftCandidateFixture(t, notReadyPath)
	writeWeightedReadinessFixture(t, notReadyPath, stringsReplaceFirst(readText(t, notReadyPath), `"weighted_admission_resonance_graft_candidate_ready": true`, `"weighted_admission_resonance_graft_candidate_ready": false`))
	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionResonanceGraftCandidateAssert([]string{notReadyPath}),
		"weighted admission resonance graft candidate weighted_admission_resonance_graft_candidate_ready not ready",
	)

	openedPath := filepath.Join(dir, "opened.json")
	writeWeightedAdmissionResonanceGraftCandidateFixture(t, openedPath)
	writeWeightedReadinessFixture(t, openedPath, stringsReplaceFirst(readText(t, openedPath), `"graft_allowed": false`, `"graft_allowed": true`))
	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionResonanceGraftCandidateAssert([]string{openedPath}),
		"weighted admission resonance graft candidate opened graft_allowed",
	)

	openedLivePath := filepath.Join(dir, "opened_live.json")
	writeWeightedAdmissionResonanceGraftCandidateFixture(t, openedLivePath)
	writeWeightedReadinessFixture(t, openedLivePath, stringsReplaceFirst(readText(t, openedLivePath), `"live_admission_enabled": false`, `"live_admission_enabled": true`))
	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionResonanceGraftCandidateAssert([]string{openedLivePath}),
		"weighted admission resonance graft candidate opened live_admission_enabled",
	)

	missingPathField := filepath.Join(dir, "missing_path.json")
	writeWeightedAdmissionResonanceGraftCandidateFixture(t, missingPathField)
	gateReport := filepath.Join(dir, "gate-"+filepath.Base(missingPathField))
	writeWeightedReadinessFixture(t, missingPathField, stringsReplaceFirst(readText(t, missingPathField), `"source_report": "`+gateReport+`"`, `"source_report": " "`))
	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionResonanceGraftCandidateAssert([]string{missingPathField}),
		"weighted admission resonance graft candidate source_report missing",
	)

	badSourcePath := filepath.Join(dir, "bad_source.json")
	writeWeightedAdmissionResonanceGraftCandidateFixture(t, badSourcePath)
	writeWeightedReadinessFixture(t, badSourcePath, stringsReplaceFirst(readText(t, badSourcePath), `"source_schema": "arianna.live_route_weighted_admission_resonance_graft_gate.v1"`, `"source_schema": "arianna.live_route_weighted_admission_resonance_graft_gate.v0"`))
	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionResonanceGraftCandidateAssert([]string{badSourcePath}),
		`weighted admission resonance graft candidate source_schema mismatch: got "arianna.live_route_weighted_admission_resonance_graft_gate.v0" want "`+admissionLiveRouteWeightedAdmissionResonanceGraftGateSchema+`"`,
	)

	badSourceGateKindPath := filepath.Join(dir, "bad_source_gate_kind.json")
	writeWeightedAdmissionResonanceGraftCandidateFixture(t, badSourceGateKindPath)
	writeWeightedReadinessFixture(t, badSourceGateKindPath, stringsReplaceFirst(readText(t, badSourceGateKindPath), `"source_gate_kind": "shadow_graft_gate"`, `"source_gate_kind": "live_graft"`))
	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionResonanceGraftCandidateAssert([]string{badSourceGateKindPath}),
		"weighted admission resonance graft candidate source gate shape mismatch",
	)

	badSourcePreflightKindPath := filepath.Join(dir, "bad_source_preflight_kind.json")
	writeWeightedAdmissionResonanceGraftCandidateFixture(t, badSourcePreflightKindPath)
	writeWeightedReadinessFixture(t, badSourcePreflightKindPath, stringsReplaceFirst(readText(t, badSourcePreflightKindPath), `"source_preflight_kind": "shadow_graft_preflight"`, `"source_preflight_kind": "live_graft"`))
	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionResonanceGraftCandidateAssert([]string{badSourcePreflightKindPath}),
		"weighted admission resonance graft candidate source preflight shape mismatch",
	)

	badCandidateHashPath := filepath.Join(dir, "bad_candidate_hash.json")
	writeWeightedAdmissionResonanceGraftCandidateFixture(t, badCandidateHashPath)
	writeWeightedReadinessFixture(t, badCandidateHashPath, stringsReplaceFirst(readText(t, badCandidateHashPath), `"candidate_hash": "weighted-resonance-graft-candidate-`, `"candidate_hash": "weighted-resonance-graft-candidate-bad`))
	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionResonanceGraftCandidateAssert([]string{badCandidateHashPath}),
		"weighted admission resonance graft candidate candidate_hash mismatch",
	)

	badSourceGateGuardPath := filepath.Join(dir, "bad_source_gate_guard.json")
	writeWeightedAdmissionResonanceGraftCandidateFixture(t, badSourceGateGuardPath)
	writeWeightedReadinessFixture(t, badSourceGateGuardPath, stringsReplaceFirst(readText(t, badSourceGateGuardPath), `"source_gate_graft_allowed": false`, `"source_gate_graft_allowed": true`))
	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionResonanceGraftCandidateAssert([]string{badSourceGateGuardPath}),
		"weighted admission resonance graft candidate opened source_gate_graft_allowed",
	)

	badSourceGuardPath := filepath.Join(dir, "bad_source_guard.json")
	writeWeightedAdmissionResonanceGraftCandidateFixture(t, badSourceGuardPath)
	writeWeightedReadinessFixture(t, badSourceGuardPath, stringsReplaceFirst(readText(t, badSourceGuardPath), `"source_raw_dream_text_allowed": false`, `"source_raw_dream_text_allowed": true`))
	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionResonanceGraftCandidateAssert([]string{badSourceGuardPath}),
		"weighted admission resonance graft candidate opened source_raw_dream_text_allowed",
	)

	badBodyTargetPath := filepath.Join(dir, "bad_body_target.json")
	writeWeightedAdmissionResonanceGraftCandidateFixture(t, badBodyTargetPath)
	writeWeightedReadinessFixture(t, badBodyTargetPath, stringsReplaceFirst(readText(t, badBodyTargetPath), `"body_target": "none"`, `"body_target": "live"`))
	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionResonanceGraftCandidateAssert([]string{badBodyTargetPath}),
		`weighted admission resonance graft candidate body_target mismatch: got "live" want "none"`,
	)
}

func writeWeightedAdmissionResonanceGraftCandidateFixture(t *testing.T, candidatePath string) {
	t.Helper()
	dir := filepath.Dir(candidatePath)
	gatePath := filepath.Join(dir, "gate-"+filepath.Base(candidatePath))
	writeWeightedAdmissionResonanceGraftGateFixture(t, gatePath)
	if err := runAdmissionLiveRouteWeightedAdmissionResonanceGraftCandidate([]string{gatePath, candidatePath}); err != nil {
		t.Fatalf("write weighted admission resonance graft candidate fixture: %v", err)
	}
}
