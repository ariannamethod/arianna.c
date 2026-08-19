package main

import (
	"path/filepath"
	"strings"
	"testing"
)

func TestAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionEnableGateAssert(t *testing.T) {
	dir := t.TempDir()

	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionEnableGateAssert(nil),
		"usage: --admission-live-route-weighted-admission-resonance-graft-admission-enable-gate-assert REPORT",
	)
	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionEnableGateAssert([]string{"enable_gate.json", "extra"}),
		"usage: --admission-live-route-weighted-admission-resonance-graft-admission-enable-gate-assert REPORT",
	)
	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionEnableGateAssert([]string{"  "}),
		"weighted admission resonance graft admission enable gate path missing",
	)
	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionEnableGateAssert([]string{filepath.Join(dir, "missing.json")}),
		"weighted admission resonance graft admission enable gate not written",
	)

	emptyPath := filepath.Join(dir, "empty.json")
	writeWeightedReadinessFixture(t, emptyPath, "")
	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionEnableGateAssert([]string{emptyPath}),
		"weighted admission resonance graft admission enable gate not written",
	)

	invalidPath := filepath.Join(dir, "invalid.json")
	writeWeightedReadinessFixture(t, invalidPath, "{")
	err := runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionEnableGateAssert([]string{invalidPath})
	if err == nil || !strings.HasPrefix(err.Error(), "weighted admission resonance graft admission enable gate JSON invalid:") {
		t.Fatalf("expected invalid JSON error, got %v", err)
	}

	missingSchemaPath := filepath.Join(dir, "missing_schema.json")
	writeWeightedAdmissionResonanceGraftAdmissionEnableGateFixture(t, missingSchemaPath)
	gateText := readText(t, missingSchemaPath)
	writeWeightedReadinessFixture(t, missingSchemaPath, stringsReplaceFirst(gateText, `"schema": "arianna.live_route_weighted_admission_resonance_graft_admission_enable_gate.v1",`, ""))
	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionEnableGateAssert([]string{missingSchemaPath}),
		"weighted admission resonance graft admission enable gate schema missing",
	)

	badSchemaPath := filepath.Join(dir, "bad_schema.json")
	writeWeightedAdmissionResonanceGraftAdmissionEnableGateFixture(t, badSchemaPath)
	writeWeightedReadinessFixture(t, badSchemaPath, stringsReplaceFirst(readText(t, badSchemaPath), `"schema": "arianna.live_route_weighted_admission_resonance_graft_admission_enable_gate.v1"`, `"schema": "arianna.live_route_weighted_admission_resonance_graft_admission_enable_gate.v0"`))
	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionEnableGateAssert([]string{badSchemaPath}),
		`weighted admission resonance graft admission enable gate schema mismatch: got "arianna.live_route_weighted_admission_resonance_graft_admission_enable_gate.v0" want "`+admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionEnableGateSchema+`"`,
	)

	validPath := filepath.Join(dir, "valid.json")
	writeWeightedAdmissionResonanceGraftAdmissionEnableGateFixture(t, validPath)
	if err := runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionEnableGateAssert([]string{validPath}); err != nil {
		t.Fatalf("valid weighted admission resonance graft admission enable gate rejected: %v", err)
	}

	badStatusPath := filepath.Join(dir, "bad_status.json")
	writeWeightedAdmissionResonanceGraftAdmissionEnableGateFixture(t, badStatusPath)
	writeWeightedReadinessFixture(t, badStatusPath, stringsReplaceFirst(readText(t, badStatusPath), `"status": "shadow_graft_admission_enable_gate_disabled_dry_run"`, `"status": "open"`))
	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionEnableGateAssert([]string{badStatusPath}),
		`weighted admission resonance graft admission enable gate status mismatch: got "open" want "shadow_graft_admission_enable_gate_disabled_dry_run"`,
	)

	notReadyPath := filepath.Join(dir, "not_ready.json")
	writeWeightedAdmissionResonanceGraftAdmissionEnableGateFixture(t, notReadyPath)
	writeWeightedReadinessFixture(t, notReadyPath, stringsReplaceFirst(readText(t, notReadyPath), `"weighted_admission_resonance_graft_admission_enable_gate_ready": true`, `"weighted_admission_resonance_graft_admission_enable_gate_ready": false`))
	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionEnableGateAssert([]string{notReadyPath}),
		"weighted admission resonance graft admission enable gate weighted_admission_resonance_graft_admission_enable_gate_ready not ready",
	)

	badEnableStatePath := filepath.Join(dir, "bad_enable_state.json")
	writeWeightedAdmissionResonanceGraftAdmissionEnableGateFixture(t, badEnableStatePath)
	writeWeightedReadinessFixture(t, badEnableStatePath, stringsReplaceFirst(readText(t, badEnableStatePath), `"enable_state": "disabled"`, `"enable_state": "armed_dry_run"`))
	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionEnableGateAssert([]string{badEnableStatePath}),
		`weighted admission resonance graft admission enable gate enable_state mismatch: got "armed_dry_run" want "disabled"`,
	)

	badEnableActionPath := filepath.Join(dir, "bad_enable_action.json")
	writeWeightedAdmissionResonanceGraftAdmissionEnableGateFixture(t, badEnableActionPath)
	writeWeightedReadinessFixture(t, badEnableActionPath, stringsReplaceFirst(readText(t, badEnableActionPath), `"enable_action": "require_operator_key"`, `"enable_action": "would_enable_live_admission_dry_run"`))
	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionEnableGateAssert([]string{badEnableActionPath}),
		`weighted admission resonance graft admission enable gate enable_action mismatch: got "would_enable_live_admission_dry_run" want "require_operator_key"`,
	)

	badSwitchStatePath := filepath.Join(dir, "bad_switch_state.json")
	writeWeightedAdmissionResonanceGraftAdmissionEnableGateFixture(t, badSwitchStatePath)
	writeWeightedReadinessFixture(t, badSwitchStatePath, stringsReplaceFirst(readText(t, badSwitchStatePath), `"switch_state": "disabled"`, `"switch_state": "enabled"`))
	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionEnableGateAssert([]string{badSwitchStatePath}),
		`weighted admission resonance graft admission enable gate switch_state mismatch: got "enabled" want "disabled"`,
	)

	openedPath := filepath.Join(dir, "opened.json")
	writeWeightedAdmissionResonanceGraftAdmissionEnableGateFixture(t, openedPath)
	writeWeightedReadinessFixture(t, openedPath, stringsReplaceFirst(readText(t, openedPath), `"live_admission_enabled": false`, `"live_admission_enabled": true`))
	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionEnableGateAssert([]string{openedPath}),
		"weighted admission resonance graft admission enable gate opened live_admission_enabled",
	)

	openedSourceSwitchPath := filepath.Join(dir, "opened_source_switch.json")
	writeWeightedAdmissionResonanceGraftAdmissionEnableGateFixture(t, openedSourceSwitchPath)
	writeWeightedReadinessFixture(t, openedSourceSwitchPath, stringsReplaceFirst(readText(t, openedSourceSwitchPath), `"source_switch_live_admission_enabled": false`, `"source_switch_live_admission_enabled": true`))
	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionEnableGateAssert([]string{openedSourceSwitchPath}),
		"weighted admission resonance graft admission enable gate opened source_switch_live_admission_enabled",
	)

	missingPathField := filepath.Join(dir, "missing_path.json")
	writeWeightedAdmissionResonanceGraftAdmissionEnableGateFixture(t, missingPathField)
	switchReport := filepath.Join(dir, "switch-"+filepath.Base(missingPathField))
	writeWeightedReadinessFixture(t, missingPathField, stringsReplaceFirst(readText(t, missingPathField), `"source_report": "`+switchReport+`"`, `"source_report": " "`))
	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionEnableGateAssert([]string{missingPathField}),
		"weighted admission resonance graft admission enable gate source_report missing",
	)

	badSourcePath := filepath.Join(dir, "bad_source.json")
	writeWeightedAdmissionResonanceGraftAdmissionEnableGateFixture(t, badSourcePath)
	writeWeightedReadinessFixture(t, badSourcePath, stringsReplaceFirst(readText(t, badSourcePath), `"source_schema": "arianna.live_route_weighted_admission_resonance_graft_admission_switch.v1"`, `"source_schema": "arianna.live_route_weighted_admission_resonance_graft_admission_switch.v0"`))
	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionEnableGateAssert([]string{badSourcePath}),
		`weighted admission resonance graft admission enable gate source_schema mismatch: got "arianna.live_route_weighted_admission_resonance_graft_admission_switch.v0" want "`+admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionSwitchSchema+`"`,
	)

	badSourceSwitchPath := filepath.Join(dir, "bad_source_switch.json")
	writeWeightedAdmissionResonanceGraftAdmissionEnableGateFixture(t, badSourceSwitchPath)
	writeWeightedReadinessFixture(t, badSourceSwitchPath, stringsReplaceFirst(readText(t, badSourceSwitchPath), `"source_switch_kind": "shadow_graft_admission_switch"`, `"source_switch_kind": "live_switch"`))
	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionEnableGateAssert([]string{badSourceSwitchPath}),
		"weighted admission resonance graft admission enable gate source switch shape mismatch",
	)

	badSourcePromotionPath := filepath.Join(dir, "bad_source_promotion.json")
	writeWeightedAdmissionResonanceGraftAdmissionEnableGateFixture(t, badSourcePromotionPath)
	writeWeightedReadinessFixture(t, badSourcePromotionPath, stringsReplaceFirst(readText(t, badSourcePromotionPath), `"source_promotion": "pending_live_admission"`, `"source_promotion": "blocked"`))
	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionEnableGateAssert([]string{badSourcePromotionPath}),
		`weighted admission resonance graft admission enable gate source_promotion mismatch: got "blocked" want "pending_live_admission"`,
	)

	badEnableGateHashPath := filepath.Join(dir, "bad_enable_gate_hash.json")
	writeWeightedAdmissionResonanceGraftAdmissionEnableGateFixture(t, badEnableGateHashPath)
	writeWeightedReadinessFixture(t, badEnableGateHashPath, stringsReplaceFirst(readText(t, badEnableGateHashPath), `"enable_gate_hash": "weighted-resonance-graft-admission-enable-gate-`, `"enable_gate_hash": "weighted-resonance-graft-admission-enable-gate-bad`))
	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionEnableGateAssert([]string{badEnableGateHashPath}),
		"weighted admission resonance graft admission enable gate enable_gate_hash mismatch",
	)

	badBodyTargetPath := filepath.Join(dir, "bad_body_target.json")
	writeWeightedAdmissionResonanceGraftAdmissionEnableGateFixture(t, badBodyTargetPath)
	writeWeightedReadinessFixture(t, badBodyTargetPath, stringsReplaceFirst(readText(t, badBodyTargetPath), `"body_target": "none"`, `"body_target": "live"`))
	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionEnableGateAssert([]string{badBodyTargetPath}),
		`weighted admission resonance graft admission enable gate body_target mismatch: got "live" want "none"`,
	)
}

func writeWeightedAdmissionResonanceGraftAdmissionEnableGateFixture(t *testing.T, enableGatePath string) {
	t.Helper()
	dir := filepath.Dir(enableGatePath)
	switchPath := filepath.Join(dir, "switch-"+filepath.Base(enableGatePath))
	writeWeightedAdmissionResonanceGraftAdmissionSwitchFixture(t, switchPath)
	if err := runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionEnableGate([]string{switchPath, enableGatePath}); err != nil {
		t.Fatalf("write weighted admission resonance graft admission enable gate fixture: %v", err)
	}
}
