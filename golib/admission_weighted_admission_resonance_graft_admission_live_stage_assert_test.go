package main

import (
	"path/filepath"
	"strings"
	"testing"
)

func TestAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionLiveStageAssert(t *testing.T) {
	dir := t.TempDir()

	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionLiveStageAssert(nil),
		"usage: --admission-live-route-weighted-admission-resonance-graft-admission-live-stage-assert REPORT",
	)
	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionLiveStageAssert([]string{"live_stage.json", "extra"}),
		"usage: --admission-live-route-weighted-admission-resonance-graft-admission-live-stage-assert REPORT",
	)
	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionLiveStageAssert([]string{"  "}),
		"weighted admission resonance graft admission live stage path missing",
	)
	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionLiveStageAssert([]string{filepath.Join(dir, "missing.json")}),
		"weighted admission resonance graft admission live stage not written",
	)

	emptyPath := filepath.Join(dir, "empty.json")
	writeWeightedReadinessFixture(t, emptyPath, "")
	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionLiveStageAssert([]string{emptyPath}),
		"weighted admission resonance graft admission live stage not written",
	)

	invalidPath := filepath.Join(dir, "invalid.json")
	writeWeightedReadinessFixture(t, invalidPath, "{")
	err := runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionLiveStageAssert([]string{invalidPath})
	if err == nil || !strings.HasPrefix(err.Error(), "weighted admission resonance graft admission live stage JSON invalid:") {
		t.Fatalf("expected invalid JSON error, got %v", err)
	}

	missingSchemaPath := filepath.Join(dir, "missing_schema.json")
	writeWeightedAdmissionResonanceGraftAdmissionLiveStageFixture(t, missingSchemaPath)
	stageText := readText(t, missingSchemaPath)
	writeWeightedReadinessFixture(t, missingSchemaPath, stringsReplaceFirst(stageText, `"schema": "arianna.live_route_weighted_admission_resonance_graft_admission_live_stage.v1",`, ""))
	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionLiveStageAssert([]string{missingSchemaPath}),
		"weighted admission resonance graft admission live stage schema missing",
	)

	badSchemaPath := filepath.Join(dir, "bad_schema.json")
	writeWeightedAdmissionResonanceGraftAdmissionLiveStageFixture(t, badSchemaPath)
	writeWeightedReadinessFixture(t, badSchemaPath, stringsReplaceFirst(readText(t, badSchemaPath), `"schema": "arianna.live_route_weighted_admission_resonance_graft_admission_live_stage.v1"`, `"schema": "arianna.live_route_weighted_admission_resonance_graft_admission_live_stage.v0"`))
	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionLiveStageAssert([]string{badSchemaPath}),
		`weighted admission resonance graft admission live stage schema mismatch: got "arianna.live_route_weighted_admission_resonance_graft_admission_live_stage.v0" want "`+admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionLiveStageSchema+`"`,
	)

	validPath := filepath.Join(dir, "valid.json")
	writeWeightedAdmissionResonanceGraftAdmissionLiveStageFixture(t, validPath)
	if err := runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionLiveStageAssert([]string{validPath}); err != nil {
		t.Fatalf("valid weighted admission resonance graft admission live stage rejected: %v", err)
	}

	badStatusPath := filepath.Join(dir, "bad_status.json")
	writeWeightedAdmissionResonanceGraftAdmissionLiveStageFixture(t, badStatusPath)
	writeWeightedReadinessFixture(t, badStatusPath, stringsReplaceFirst(readText(t, badStatusPath), `"status": "shadow_graft_admission_live_stage_blocked_dry_run"`, `"status": "open"`))
	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionLiveStageAssert([]string{badStatusPath}),
		`weighted admission resonance graft admission live stage status mismatch: got "open" want "shadow_graft_admission_live_stage_blocked_dry_run"`,
	)

	notReadyPath := filepath.Join(dir, "not_ready.json")
	writeWeightedAdmissionResonanceGraftAdmissionLiveStageFixture(t, notReadyPath)
	writeWeightedReadinessFixture(t, notReadyPath, stringsReplaceFirst(readText(t, notReadyPath), `"weighted_admission_resonance_graft_admission_live_stage_ready": true`, `"weighted_admission_resonance_graft_admission_live_stage_ready": false`))
	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionLiveStageAssert([]string{notReadyPath}),
		"weighted admission resonance graft admission live stage weighted_admission_resonance_graft_admission_live_stage_ready not ready",
	)

	badStageStatePath := filepath.Join(dir, "bad_stage_state.json")
	writeWeightedAdmissionResonanceGraftAdmissionLiveStageFixture(t, badStageStatePath)
	writeWeightedReadinessFixture(t, badStageStatePath, stringsReplaceFirst(readText(t, badStageStatePath), `"stage_state": "blocked"`, `"stage_state": "staged"`))
	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionLiveStageAssert([]string{badStageStatePath}),
		`weighted admission resonance graft admission live stage stage_state mismatch: got "staged" want "blocked"`,
	)

	badStageActionPath := filepath.Join(dir, "bad_stage_action.json")
	writeWeightedAdmissionResonanceGraftAdmissionLiveStageFixture(t, badStageActionPath)
	writeWeightedReadinessFixture(t, badStageActionPath, stringsReplaceFirst(readText(t, badStageActionPath), `"stage_action": "reject_disabled_enable_gate"`, `"stage_action": "stage_live_admission"`))
	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionLiveStageAssert([]string{badStageActionPath}),
		`weighted admission resonance graft admission live stage stage_action mismatch: got "stage_live_admission" want "reject_disabled_enable_gate"`,
	)

	badEnableActionPath := filepath.Join(dir, "bad_enable_action.json")
	writeWeightedAdmissionResonanceGraftAdmissionLiveStageFixture(t, badEnableActionPath)
	writeWeightedReadinessFixture(t, badEnableActionPath, stringsReplaceFirst(readText(t, badEnableActionPath), `"enable_action": "require_operator_key"`, `"enable_action": "would_enable_live_admission_dry_run"`))
	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionLiveStageAssert([]string{badEnableActionPath}),
		`weighted admission resonance graft admission live stage enable_action mismatch: got "would_enable_live_admission_dry_run" want "require_operator_key"`,
	)

	openedPath := filepath.Join(dir, "opened.json")
	writeWeightedAdmissionResonanceGraftAdmissionLiveStageFixture(t, openedPath)
	writeWeightedReadinessFixture(t, openedPath, stringsReplaceFirst(readText(t, openedPath), `"live_admission_enabled": false`, `"live_admission_enabled": true`))
	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionLiveStageAssert([]string{openedPath}),
		"weighted admission resonance graft admission live stage opened live_admission_enabled",
	)

	openedWriterPath := filepath.Join(dir, "opened_writer.json")
	writeWeightedAdmissionResonanceGraftAdmissionLiveStageFixture(t, openedWriterPath)
	writeWeightedReadinessFixture(t, openedWriterPath, stringsReplaceFirst(readText(t, openedWriterPath), `"writer_ready": false`, `"writer_ready": true`))
	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionLiveStageAssert([]string{openedWriterPath}),
		"weighted admission resonance graft admission live stage opened writer_ready",
	)

	openedSourceGatePath := filepath.Join(dir, "opened_source_gate.json")
	writeWeightedAdmissionResonanceGraftAdmissionLiveStageFixture(t, openedSourceGatePath)
	writeWeightedReadinessFixture(t, openedSourceGatePath, stringsReplaceFirst(readText(t, openedSourceGatePath), `"source_enable_gate_live_admission_enabled": false`, `"source_enable_gate_live_admission_enabled": true`))
	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionLiveStageAssert([]string{openedSourceGatePath}),
		"weighted admission resonance graft admission live stage opened source_enable_gate_live_admission_enabled",
	)

	missingPathField := filepath.Join(dir, "missing_path.json")
	writeWeightedAdmissionResonanceGraftAdmissionLiveStageFixture(t, missingPathField)
	enableGateReport := filepath.Join(dir, "enable_gate-"+filepath.Base(missingPathField))
	writeWeightedReadinessFixture(t, missingPathField, stringsReplaceFirst(readText(t, missingPathField), `"source_report": "`+enableGateReport+`"`, `"source_report": " "`))
	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionLiveStageAssert([]string{missingPathField}),
		"weighted admission resonance graft admission live stage source_report missing",
	)

	badSourcePath := filepath.Join(dir, "bad_source.json")
	writeWeightedAdmissionResonanceGraftAdmissionLiveStageFixture(t, badSourcePath)
	writeWeightedReadinessFixture(t, badSourcePath, stringsReplaceFirst(readText(t, badSourcePath), `"source_schema": "arianna.live_route_weighted_admission_resonance_graft_admission_enable_gate.v1"`, `"source_schema": "arianna.live_route_weighted_admission_resonance_graft_admission_enable_gate.v0"`))
	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionLiveStageAssert([]string{badSourcePath}),
		`weighted admission resonance graft admission live stage source_schema mismatch: got "arianna.live_route_weighted_admission_resonance_graft_admission_enable_gate.v0" want "`+admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionEnableGateSchema+`"`,
	)

	badSourceGatePath := filepath.Join(dir, "bad_source_gate.json")
	writeWeightedAdmissionResonanceGraftAdmissionLiveStageFixture(t, badSourceGatePath)
	writeWeightedReadinessFixture(t, badSourceGatePath, stringsReplaceFirst(readText(t, badSourceGatePath), `"source_enable_gate_kind": "shadow_graft_admission_enable_gate"`, `"source_enable_gate_kind": "live_enable_gate"`))
	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionLiveStageAssert([]string{badSourceGatePath}),
		"weighted admission resonance graft admission live stage source enable gate shape mismatch",
	)

	badSourcePromotionPath := filepath.Join(dir, "bad_source_promotion.json")
	writeWeightedAdmissionResonanceGraftAdmissionLiveStageFixture(t, badSourcePromotionPath)
	writeWeightedReadinessFixture(t, badSourcePromotionPath, stringsReplaceFirst(readText(t, badSourcePromotionPath), `"source_promotion": "pending_live_admission"`, `"source_promotion": "blocked"`))
	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionLiveStageAssert([]string{badSourcePromotionPath}),
		`weighted admission resonance graft admission live stage source_promotion mismatch: got "blocked" want "pending_live_admission"`,
	)

	badLiveStageHashPath := filepath.Join(dir, "bad_live_stage_hash.json")
	writeWeightedAdmissionResonanceGraftAdmissionLiveStageFixture(t, badLiveStageHashPath)
	writeWeightedReadinessFixture(t, badLiveStageHashPath, stringsReplaceFirst(readText(t, badLiveStageHashPath), `"live_stage_hash": "weighted-resonance-graft-admission-live-stage-`, `"live_stage_hash": "weighted-resonance-graft-admission-live-stage-bad`))
	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionLiveStageAssert([]string{badLiveStageHashPath}),
		"weighted admission resonance graft admission live stage live_stage_hash mismatch",
	)

	badBodyTargetPath := filepath.Join(dir, "bad_body_target.json")
	writeWeightedAdmissionResonanceGraftAdmissionLiveStageFixture(t, badBodyTargetPath)
	writeWeightedReadinessFixture(t, badBodyTargetPath, stringsReplaceFirst(readText(t, badBodyTargetPath), `"body_target": "none"`, `"body_target": "live"`))
	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionLiveStageAssert([]string{badBodyTargetPath}),
		`weighted admission resonance graft admission live stage body_target mismatch: got "live" want "none"`,
	)
}

func writeWeightedAdmissionResonanceGraftAdmissionLiveStageFixture(t *testing.T, liveStagePath string) {
	t.Helper()
	dir := filepath.Dir(liveStagePath)
	enableGatePath := filepath.Join(dir, "enable_gate-"+filepath.Base(liveStagePath))
	writeWeightedAdmissionResonanceGraftAdmissionEnableGateFixture(t, enableGatePath)
	if err := runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionLiveStage([]string{enableGatePath, liveStagePath}); err != nil {
		t.Fatalf("write weighted admission resonance graft admission live stage fixture: %v", err)
	}
}
