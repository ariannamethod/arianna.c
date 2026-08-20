package main

import (
	"path/filepath"
	"strings"
	"testing"
)

func TestAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionWriterPreflightAssert(t *testing.T) {
	dir := t.TempDir()

	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionWriterPreflightAssert(nil),
		"usage: --admission-live-route-weighted-admission-resonance-graft-admission-writer-preflight-assert REPORT",
	)
	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionWriterPreflightAssert([]string{"writer_preflight.json", "extra"}),
		"usage: --admission-live-route-weighted-admission-resonance-graft-admission-writer-preflight-assert REPORT",
	)
	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionWriterPreflightAssert([]string{"  "}),
		"weighted admission resonance graft admission writer preflight path missing",
	)
	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionWriterPreflightAssert([]string{filepath.Join(dir, "missing.json")}),
		"weighted admission resonance graft admission writer preflight not written",
	)

	emptyPath := filepath.Join(dir, "empty.json")
	writeWeightedReadinessFixture(t, emptyPath, "")
	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionWriterPreflightAssert([]string{emptyPath}),
		"weighted admission resonance graft admission writer preflight not written",
	)

	invalidPath := filepath.Join(dir, "invalid.json")
	writeWeightedReadinessFixture(t, invalidPath, "{")
	err := runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionWriterPreflightAssert([]string{invalidPath})
	if err == nil || !strings.HasPrefix(err.Error(), "weighted admission resonance graft admission writer preflight JSON invalid:") {
		t.Fatalf("expected invalid JSON error, got %v", err)
	}

	missingSchemaPath := filepath.Join(dir, "missing_schema.json")
	writeWeightedAdmissionResonanceGraftAdmissionWriterPreflightFixture(t, missingSchemaPath)
	preflightText := readText(t, missingSchemaPath)
	writeWeightedReadinessFixture(t, missingSchemaPath, stringsReplaceFirst(preflightText, `"schema": "arianna.live_route_weighted_admission_resonance_graft_admission_writer_preflight.v1",`, ""))
	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionWriterPreflightAssert([]string{missingSchemaPath}),
		"weighted admission resonance graft admission writer preflight schema missing",
	)

	badSchemaPath := filepath.Join(dir, "bad_schema.json")
	writeWeightedAdmissionResonanceGraftAdmissionWriterPreflightFixture(t, badSchemaPath)
	writeWeightedReadinessFixture(t, badSchemaPath, stringsReplaceFirst(readText(t, badSchemaPath), `"schema": "arianna.live_route_weighted_admission_resonance_graft_admission_writer_preflight.v1"`, `"schema": "arianna.live_route_weighted_admission_resonance_graft_admission_writer_preflight.v0"`))
	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionWriterPreflightAssert([]string{badSchemaPath}),
		`weighted admission resonance graft admission writer preflight schema mismatch: got "arianna.live_route_weighted_admission_resonance_graft_admission_writer_preflight.v0" want "`+admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionWriterPreflightSchema+`"`,
	)

	validPath := filepath.Join(dir, "valid.json")
	writeWeightedAdmissionResonanceGraftAdmissionWriterPreflightFixture(t, validPath)
	if err := runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionWriterPreflightAssert([]string{validPath}); err != nil {
		t.Fatalf("valid weighted admission resonance graft admission writer preflight rejected: %v", err)
	}

	badStatusPath := filepath.Join(dir, "bad_status.json")
	writeWeightedAdmissionResonanceGraftAdmissionWriterPreflightFixture(t, badStatusPath)
	writeWeightedReadinessFixture(t, badStatusPath, stringsReplaceFirst(readText(t, badStatusPath), `"status": "shadow_graft_admission_writer_preflight_blocked_dry_run"`, `"status": "open"`))
	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionWriterPreflightAssert([]string{badStatusPath}),
		`weighted admission resonance graft admission writer preflight status mismatch: got "open" want "shadow_graft_admission_writer_preflight_blocked_dry_run"`,
	)

	notReadyPath := filepath.Join(dir, "not_ready.json")
	writeWeightedAdmissionResonanceGraftAdmissionWriterPreflightFixture(t, notReadyPath)
	writeWeightedReadinessFixture(t, notReadyPath, stringsReplaceFirst(readText(t, notReadyPath), `"weighted_admission_resonance_graft_admission_writer_preflight_ready": true`, `"weighted_admission_resonance_graft_admission_writer_preflight_ready": false`))
	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionWriterPreflightAssert([]string{notReadyPath}),
		"weighted admission resonance graft admission writer preflight weighted_admission_resonance_graft_admission_writer_preflight_ready not ready",
	)

	badWriterStatePath := filepath.Join(dir, "bad_writer_state.json")
	writeWeightedAdmissionResonanceGraftAdmissionWriterPreflightFixture(t, badWriterStatePath)
	writeWeightedReadinessFixture(t, badWriterStatePath, stringsReplaceFirst(readText(t, badWriterStatePath), `"writer_state": "blocked"`, `"writer_state": "absent"`))
	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionWriterPreflightAssert([]string{badWriterStatePath}),
		`weighted admission resonance graft admission writer preflight writer_state mismatch: got "absent" want "blocked"`,
	)

	badWriterActionPath := filepath.Join(dir, "bad_writer_action.json")
	writeWeightedAdmissionResonanceGraftAdmissionWriterPreflightFixture(t, badWriterActionPath)
	writeWeightedReadinessFixture(t, badWriterActionPath, stringsReplaceFirst(readText(t, badWriterActionPath), `"writer_action": "reject_blocked_live_stage"`, `"writer_action": "require_writer_contract"`))
	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionWriterPreflightAssert([]string{badWriterActionPath}),
		`weighted admission resonance graft admission writer preflight writer_action mismatch: got "require_writer_contract" want "reject_blocked_live_stage"`,
	)

	badStageActionPath := filepath.Join(dir, "bad_stage_action.json")
	writeWeightedAdmissionResonanceGraftAdmissionWriterPreflightFixture(t, badStageActionPath)
	writeWeightedReadinessFixture(t, badStageActionPath, stringsReplaceFirst(readText(t, badStageActionPath), `"stage_action": "reject_disabled_enable_gate"`, `"stage_action": "stage_live_admission"`))
	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionWriterPreflightAssert([]string{badStageActionPath}),
		`weighted admission resonance graft admission writer preflight stage_action mismatch: got "stage_live_admission" want "reject_disabled_enable_gate"`,
	)

	openedPath := filepath.Join(dir, "opened.json")
	writeWeightedAdmissionResonanceGraftAdmissionWriterPreflightFixture(t, openedPath)
	writeWeightedReadinessFixture(t, openedPath, stringsReplaceFirst(readText(t, openedPath), `"live_admission_enabled": false`, `"live_admission_enabled": true`))
	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionWriterPreflightAssert([]string{openedPath}),
		"weighted admission resonance graft admission writer preflight opened live_admission_enabled",
	)

	openedWriterPath := filepath.Join(dir, "opened_writer.json")
	writeWeightedAdmissionResonanceGraftAdmissionWriterPreflightFixture(t, openedWriterPath)
	writeWeightedReadinessFixture(t, openedWriterPath, stringsReplaceFirst(readText(t, openedWriterPath), `"writer_ready": false`, `"writer_ready": true`))
	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionWriterPreflightAssert([]string{openedWriterPath}),
		"weighted admission resonance graft admission writer preflight opened writer_ready",
	)

	openedSourceStagePath := filepath.Join(dir, "opened_source_stage.json")
	writeWeightedAdmissionResonanceGraftAdmissionWriterPreflightFixture(t, openedSourceStagePath)
	writeWeightedReadinessFixture(t, openedSourceStagePath, stringsReplaceFirst(readText(t, openedSourceStagePath), `"source_live_stage_live_admission_enabled": false`, `"source_live_stage_live_admission_enabled": true`))
	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionWriterPreflightAssert([]string{openedSourceStagePath}),
		"weighted admission resonance graft admission writer preflight opened source_live_stage_live_admission_enabled",
	)

	missingPathField := filepath.Join(dir, "missing_path.json")
	writeWeightedAdmissionResonanceGraftAdmissionWriterPreflightFixture(t, missingPathField)
	liveStageReport := filepath.Join(dir, "live_stage-"+filepath.Base(missingPathField))
	writeWeightedReadinessFixture(t, missingPathField, stringsReplaceFirst(readText(t, missingPathField), `"source_report": "`+liveStageReport+`"`, `"source_report": " "`))
	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionWriterPreflightAssert([]string{missingPathField}),
		"weighted admission resonance graft admission writer preflight source_report missing",
	)

	badSourcePath := filepath.Join(dir, "bad_source.json")
	writeWeightedAdmissionResonanceGraftAdmissionWriterPreflightFixture(t, badSourcePath)
	writeWeightedReadinessFixture(t, badSourcePath, stringsReplaceFirst(readText(t, badSourcePath), `"source_schema": "arianna.live_route_weighted_admission_resonance_graft_admission_live_stage.v1"`, `"source_schema": "arianna.live_route_weighted_admission_resonance_graft_admission_live_stage.v0"`))
	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionWriterPreflightAssert([]string{badSourcePath}),
		`weighted admission resonance graft admission writer preflight source_schema mismatch: got "arianna.live_route_weighted_admission_resonance_graft_admission_live_stage.v0" want "`+admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionLiveStageSchema+`"`,
	)

	badSourceStagePath := filepath.Join(dir, "bad_source_stage.json")
	writeWeightedAdmissionResonanceGraftAdmissionWriterPreflightFixture(t, badSourceStagePath)
	writeWeightedReadinessFixture(t, badSourceStagePath, stringsReplaceFirst(readText(t, badSourceStagePath), `"source_live_stage_kind": "shadow_graft_admission_live_stage"`, `"source_live_stage_kind": "live_stage"`))
	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionWriterPreflightAssert([]string{badSourceStagePath}),
		"weighted admission resonance graft admission writer preflight source live stage shape mismatch",
	)

	badSourcePromotionPath := filepath.Join(dir, "bad_source_promotion.json")
	writeWeightedAdmissionResonanceGraftAdmissionWriterPreflightFixture(t, badSourcePromotionPath)
	writeWeightedReadinessFixture(t, badSourcePromotionPath, stringsReplaceFirst(readText(t, badSourcePromotionPath), `"source_promotion": "pending_live_admission"`, `"source_promotion": "blocked"`))
	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionWriterPreflightAssert([]string{badSourcePromotionPath}),
		`weighted admission resonance graft admission writer preflight source_promotion mismatch: got "blocked" want "pending_live_admission"`,
	)

	badWriterPreflightHashPath := filepath.Join(dir, "bad_writer_preflight_hash.json")
	writeWeightedAdmissionResonanceGraftAdmissionWriterPreflightFixture(t, badWriterPreflightHashPath)
	writeWeightedReadinessFixture(t, badWriterPreflightHashPath, stringsReplaceFirst(readText(t, badWriterPreflightHashPath), `"writer_preflight_hash": "weighted-resonance-graft-admission-writer-preflight-`, `"writer_preflight_hash": "weighted-resonance-graft-admission-writer-preflight-bad`))
	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionWriterPreflightAssert([]string{badWriterPreflightHashPath}),
		"weighted admission resonance graft admission writer preflight writer_preflight_hash mismatch",
	)

	badBodyTargetPath := filepath.Join(dir, "bad_body_target.json")
	writeWeightedAdmissionResonanceGraftAdmissionWriterPreflightFixture(t, badBodyTargetPath)
	writeWeightedReadinessFixture(t, badBodyTargetPath, stringsReplaceFirst(readText(t, badBodyTargetPath), `"body_target": "none"`, `"body_target": "live"`))
	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionWriterPreflightAssert([]string{badBodyTargetPath}),
		`weighted admission resonance graft admission writer preflight body_target mismatch: got "live" want "none"`,
	)
}

func writeWeightedAdmissionResonanceGraftAdmissionWriterPreflightFixture(t *testing.T, writerPreflightPath string) {
	t.Helper()
	dir := filepath.Dir(writerPreflightPath)
	liveStagePath := filepath.Join(dir, "live_stage-"+filepath.Base(writerPreflightPath))
	writeWeightedAdmissionResonanceGraftAdmissionLiveStageFixture(t, liveStagePath)
	if err := runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionWriterPreflight([]string{liveStagePath, writerPreflightPath}); err != nil {
		t.Fatalf("write weighted admission resonance graft admission writer preflight fixture: %v", err)
	}
}
