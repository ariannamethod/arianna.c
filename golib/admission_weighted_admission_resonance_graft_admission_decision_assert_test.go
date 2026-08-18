package main

import (
	"path/filepath"
	"strings"
	"testing"
)

func TestAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionDecisionAssert(t *testing.T) {
	dir := t.TempDir()

	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionDecisionAssert(nil),
		"usage: --admission-live-route-weighted-admission-resonance-graft-admission-decision-assert REPORT",
	)
	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionDecisionAssert([]string{"decision.json", "extra"}),
		"usage: --admission-live-route-weighted-admission-resonance-graft-admission-decision-assert REPORT",
	)
	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionDecisionAssert([]string{"  "}),
		"weighted admission resonance graft admission decision path missing",
	)
	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionDecisionAssert([]string{filepath.Join(dir, "missing.json")}),
		"weighted admission resonance graft admission decision not written",
	)

	emptyPath := filepath.Join(dir, "empty.json")
	writeWeightedReadinessFixture(t, emptyPath, "")
	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionDecisionAssert([]string{emptyPath}),
		"weighted admission resonance graft admission decision not written",
	)

	invalidPath := filepath.Join(dir, "invalid.json")
	writeWeightedReadinessFixture(t, invalidPath, "{")
	err := runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionDecisionAssert([]string{invalidPath})
	if err == nil || !strings.HasPrefix(err.Error(), "weighted admission resonance graft admission decision JSON invalid:") {
		t.Fatalf("expected invalid JSON error, got %v", err)
	}

	missingSchemaPath := filepath.Join(dir, "missing_schema.json")
	writeWeightedAdmissionResonanceGraftAdmissionDecisionFixture(t, missingSchemaPath)
	decisionText := readText(t, missingSchemaPath)
	writeWeightedReadinessFixture(t, missingSchemaPath, stringsReplaceFirst(decisionText, `"schema": "arianna.live_route_weighted_admission_resonance_graft_admission_decision.v1",`, ""))
	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionDecisionAssert([]string{missingSchemaPath}),
		"weighted admission resonance graft admission decision schema missing",
	)

	badSchemaPath := filepath.Join(dir, "bad_schema.json")
	writeWeightedAdmissionResonanceGraftAdmissionDecisionFixture(t, badSchemaPath)
	writeWeightedReadinessFixture(t, badSchemaPath, stringsReplaceFirst(readText(t, badSchemaPath), `"schema": "arianna.live_route_weighted_admission_resonance_graft_admission_decision.v1"`, `"schema": "arianna.live_route_weighted_admission_resonance_graft_admission_decision.v0"`))
	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionDecisionAssert([]string{badSchemaPath}),
		`weighted admission resonance graft admission decision schema mismatch: got "arianna.live_route_weighted_admission_resonance_graft_admission_decision.v0" want "`+admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionDecisionSchema+`"`,
	)

	validPath := filepath.Join(dir, "valid.json")
	writeWeightedAdmissionResonanceGraftAdmissionDecisionFixture(t, validPath)
	if err := runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionDecisionAssert([]string{validPath}); err != nil {
		t.Fatalf("valid weighted admission resonance graft admission decision rejected: %v", err)
	}

	badStatusPath := filepath.Join(dir, "bad_status.json")
	writeWeightedAdmissionResonanceGraftAdmissionDecisionFixture(t, badStatusPath)
	writeWeightedReadinessFixture(t, badStatusPath, stringsReplaceFirst(readText(t, badStatusPath), `"status": "shadow_graft_admission_decision_ready_dry_run"`, `"status": "open"`))
	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionDecisionAssert([]string{badStatusPath}),
		`weighted admission resonance graft admission decision status mismatch: got "open" want "shadow_graft_admission_decision_ready_dry_run"`,
	)

	notReadyPath := filepath.Join(dir, "not_ready.json")
	writeWeightedAdmissionResonanceGraftAdmissionDecisionFixture(t, notReadyPath)
	writeWeightedReadinessFixture(t, notReadyPath, stringsReplaceFirst(readText(t, notReadyPath), `"weighted_admission_resonance_graft_admission_decision_ready": true`, `"weighted_admission_resonance_graft_admission_decision_ready": false`))
	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionDecisionAssert([]string{notReadyPath}),
		"weighted admission resonance graft admission decision weighted_admission_resonance_graft_admission_decision_ready not ready",
	)

	badDecisionPath := filepath.Join(dir, "bad_decision.json")
	writeWeightedAdmissionResonanceGraftAdmissionDecisionFixture(t, badDecisionPath)
	writeWeightedReadinessFixture(t, badDecisionPath, stringsReplaceFirst(readText(t, badDecisionPath), `"decision": "shadow_ready"`, `"decision": "reject"`))
	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionDecisionAssert([]string{badDecisionPath}),
		`weighted admission resonance graft admission decision decision mismatch: got "reject" want "shadow_ready"`,
	)

	openedPath := filepath.Join(dir, "opened.json")
	writeWeightedAdmissionResonanceGraftAdmissionDecisionFixture(t, openedPath)
	writeWeightedReadinessFixture(t, openedPath, stringsReplaceFirst(readText(t, openedPath), `"live_admission_enabled": false`, `"live_admission_enabled": true`))
	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionDecisionAssert([]string{openedPath}),
		"weighted admission resonance graft admission decision opened live_admission_enabled",
	)

	missingPathField := filepath.Join(dir, "missing_path.json")
	writeWeightedAdmissionResonanceGraftAdmissionDecisionFixture(t, missingPathField)
	preconditionReport := filepath.Join(dir, "precondition-"+filepath.Base(missingPathField))
	writeWeightedReadinessFixture(t, missingPathField, stringsReplaceFirst(readText(t, missingPathField), `"source_report": "`+preconditionReport+`"`, `"source_report": " "`))
	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionDecisionAssert([]string{missingPathField}),
		"weighted admission resonance graft admission decision source_report missing",
	)

	badSourcePath := filepath.Join(dir, "bad_source.json")
	writeWeightedAdmissionResonanceGraftAdmissionDecisionFixture(t, badSourcePath)
	writeWeightedReadinessFixture(t, badSourcePath, stringsReplaceFirst(readText(t, badSourcePath), `"source_schema": "arianna.live_route_weighted_admission_resonance_graft_admission_proof_precondition.v1"`, `"source_schema": "arianna.live_route_weighted_admission_resonance_graft_admission_proof_precondition.v0"`))
	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionDecisionAssert([]string{badSourcePath}),
		`weighted admission resonance graft admission decision source_schema mismatch: got "arianna.live_route_weighted_admission_resonance_graft_admission_proof_precondition.v0" want "`+admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionProofPreconditionSchema+`"`,
	)

	badSourcePreconditionKindPath := filepath.Join(dir, "bad_source_precondition_kind.json")
	writeWeightedAdmissionResonanceGraftAdmissionDecisionFixture(t, badSourcePreconditionKindPath)
	writeWeightedReadinessFixture(t, badSourcePreconditionKindPath, stringsReplaceFirst(readText(t, badSourcePreconditionKindPath), `"source_precondition_kind": "shadow_graft_admission_proof_precondition"`, `"source_precondition_kind": "live_precondition"`))
	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionDecisionAssert([]string{badSourcePreconditionKindPath}),
		"weighted admission resonance graft admission decision source precondition shape mismatch",
	)

	badSourceProofKindPath := filepath.Join(dir, "bad_source_proof_kind.json")
	writeWeightedAdmissionResonanceGraftAdmissionDecisionFixture(t, badSourceProofKindPath)
	writeWeightedReadinessFixture(t, badSourceProofKindPath, stringsReplaceFirst(readText(t, badSourceProofKindPath), `"source_proof_kind": "shadow_graft_admission_proof"`, `"source_proof_kind": "live_proof"`))
	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionDecisionAssert([]string{badSourceProofKindPath}),
		"weighted admission resonance graft admission decision source proof shape mismatch",
	)

	badDecisionHashPath := filepath.Join(dir, "bad_decision_hash.json")
	writeWeightedAdmissionResonanceGraftAdmissionDecisionFixture(t, badDecisionHashPath)
	writeWeightedReadinessFixture(t, badDecisionHashPath, stringsReplaceFirst(readText(t, badDecisionHashPath), `"decision_hash": "weighted-resonance-graft-admission-decision-`, `"decision_hash": "weighted-resonance-graft-admission-decision-bad`))
	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionDecisionAssert([]string{badDecisionHashPath}),
		"weighted admission resonance graft admission decision decision_hash mismatch",
	)

	badBodyTargetPath := filepath.Join(dir, "bad_body_target.json")
	writeWeightedAdmissionResonanceGraftAdmissionDecisionFixture(t, badBodyTargetPath)
	writeWeightedReadinessFixture(t, badBodyTargetPath, stringsReplaceFirst(readText(t, badBodyTargetPath), `"body_target": "none"`, `"body_target": "live"`))
	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionDecisionAssert([]string{badBodyTargetPath}),
		`weighted admission resonance graft admission decision body_target mismatch: got "live" want "none"`,
	)
}

func writeWeightedAdmissionResonanceGraftAdmissionDecisionFixture(t *testing.T, decisionPath string) {
	t.Helper()
	dir := filepath.Dir(decisionPath)
	preconditionPath := filepath.Join(dir, "precondition-"+filepath.Base(decisionPath))
	writeWeightedAdmissionResonanceGraftAdmissionProofPreconditionFixture(t, preconditionPath)
	if err := runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionDecision([]string{preconditionPath, decisionPath}); err != nil {
		t.Fatalf("write weighted admission resonance graft admission decision fixture: %v", err)
	}
}
