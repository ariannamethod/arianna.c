package main

import (
	"path/filepath"
	"strings"
	"testing"
)

func TestAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionSwitchAssert(t *testing.T) {
	dir := t.TempDir()

	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionSwitchAssert(nil),
		"usage: --admission-live-route-weighted-admission-resonance-graft-admission-switch-assert REPORT",
	)
	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionSwitchAssert([]string{"switch.json", "extra"}),
		"usage: --admission-live-route-weighted-admission-resonance-graft-admission-switch-assert REPORT",
	)
	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionSwitchAssert([]string{"  "}),
		"weighted admission resonance graft admission switch path missing",
	)
	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionSwitchAssert([]string{filepath.Join(dir, "missing.json")}),
		"weighted admission resonance graft admission switch not written",
	)

	emptyPath := filepath.Join(dir, "empty.json")
	writeWeightedReadinessFixture(t, emptyPath, "")
	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionSwitchAssert([]string{emptyPath}),
		"weighted admission resonance graft admission switch not written",
	)

	invalidPath := filepath.Join(dir, "invalid.json")
	writeWeightedReadinessFixture(t, invalidPath, "{")
	err := runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionSwitchAssert([]string{invalidPath})
	if err == nil || !strings.HasPrefix(err.Error(), "weighted admission resonance graft admission switch JSON invalid:") {
		t.Fatalf("expected invalid JSON error, got %v", err)
	}

	missingSchemaPath := filepath.Join(dir, "missing_schema.json")
	writeWeightedAdmissionResonanceGraftAdmissionSwitchFixture(t, missingSchemaPath)
	switchText := readText(t, missingSchemaPath)
	writeWeightedReadinessFixture(t, missingSchemaPath, stringsReplaceFirst(switchText, `"schema": "arianna.live_route_weighted_admission_resonance_graft_admission_switch.v1",`, ""))
	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionSwitchAssert([]string{missingSchemaPath}),
		"weighted admission resonance graft admission switch schema missing",
	)

	badSchemaPath := filepath.Join(dir, "bad_schema.json")
	writeWeightedAdmissionResonanceGraftAdmissionSwitchFixture(t, badSchemaPath)
	writeWeightedReadinessFixture(t, badSchemaPath, stringsReplaceFirst(readText(t, badSchemaPath), `"schema": "arianna.live_route_weighted_admission_resonance_graft_admission_switch.v1"`, `"schema": "arianna.live_route_weighted_admission_resonance_graft_admission_switch.v0"`))
	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionSwitchAssert([]string{badSchemaPath}),
		`weighted admission resonance graft admission switch schema mismatch: got "arianna.live_route_weighted_admission_resonance_graft_admission_switch.v0" want "`+admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionSwitchSchema+`"`,
	)

	validPath := filepath.Join(dir, "valid.json")
	writeWeightedAdmissionResonanceGraftAdmissionSwitchFixture(t, validPath)
	if err := runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionSwitchAssert([]string{validPath}); err != nil {
		t.Fatalf("valid weighted admission resonance graft admission switch rejected: %v", err)
	}

	badStatusPath := filepath.Join(dir, "bad_status.json")
	writeWeightedAdmissionResonanceGraftAdmissionSwitchFixture(t, badStatusPath)
	writeWeightedReadinessFixture(t, badStatusPath, stringsReplaceFirst(readText(t, badStatusPath), `"status": "shadow_graft_admission_switch_disabled_dry_run"`, `"status": "open"`))
	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionSwitchAssert([]string{badStatusPath}),
		`weighted admission resonance graft admission switch status mismatch: got "open" want "shadow_graft_admission_switch_disabled_dry_run"`,
	)

	notReadyPath := filepath.Join(dir, "not_ready.json")
	writeWeightedAdmissionResonanceGraftAdmissionSwitchFixture(t, notReadyPath)
	writeWeightedReadinessFixture(t, notReadyPath, stringsReplaceFirst(readText(t, notReadyPath), `"weighted_admission_resonance_graft_admission_switch_ready": true`, `"weighted_admission_resonance_graft_admission_switch_ready": false`))
	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionSwitchAssert([]string{notReadyPath}),
		"weighted admission resonance graft admission switch weighted_admission_resonance_graft_admission_switch_ready not ready",
	)

	badSwitchStatePath := filepath.Join(dir, "bad_switch_state.json")
	writeWeightedAdmissionResonanceGraftAdmissionSwitchFixture(t, badSwitchStatePath)
	writeWeightedReadinessFixture(t, badSwitchStatePath, stringsReplaceFirst(readText(t, badSwitchStatePath), `"switch_state": "disabled"`, `"switch_state": "enabled"`))
	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionSwitchAssert([]string{badSwitchStatePath}),
		`weighted admission resonance graft admission switch switch_state mismatch: got "enabled" want "disabled"`,
	)

	badSwitchActionPath := filepath.Join(dir, "bad_switch_action.json")
	writeWeightedAdmissionResonanceGraftAdmissionSwitchFixture(t, badSwitchActionPath)
	writeWeightedReadinessFixture(t, badSwitchActionPath, stringsReplaceFirst(readText(t, badSwitchActionPath), `"switch_action": "hold_pending_live_admission"`, `"switch_action": "enable_live_admission"`))
	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionSwitchAssert([]string{badSwitchActionPath}),
		`weighted admission resonance graft admission switch switch_action mismatch: got "enable_live_admission" want "hold_pending_live_admission"`,
	)

	openedPath := filepath.Join(dir, "opened.json")
	writeWeightedAdmissionResonanceGraftAdmissionSwitchFixture(t, openedPath)
	writeWeightedReadinessFixture(t, openedPath, stringsReplaceFirst(readText(t, openedPath), `"live_admission_enabled": false`, `"live_admission_enabled": true`))
	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionSwitchAssert([]string{openedPath}),
		"weighted admission resonance graft admission switch opened live_admission_enabled",
	)

	missingPathField := filepath.Join(dir, "missing_path.json")
	writeWeightedAdmissionResonanceGraftAdmissionSwitchFixture(t, missingPathField)
	promotionReport := filepath.Join(dir, "promotion-"+filepath.Base(missingPathField))
	writeWeightedReadinessFixture(t, missingPathField, stringsReplaceFirst(readText(t, missingPathField), `"source_report": "`+promotionReport+`"`, `"source_report": " "`))
	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionSwitchAssert([]string{missingPathField}),
		"weighted admission resonance graft admission switch source_report missing",
	)

	badSourcePath := filepath.Join(dir, "bad_source.json")
	writeWeightedAdmissionResonanceGraftAdmissionSwitchFixture(t, badSourcePath)
	writeWeightedReadinessFixture(t, badSourcePath, stringsReplaceFirst(readText(t, badSourcePath), `"source_schema": "arianna.live_route_weighted_admission_resonance_graft_admission_promotion.v1"`, `"source_schema": "arianna.live_route_weighted_admission_resonance_graft_admission_promotion.v0"`))
	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionSwitchAssert([]string{badSourcePath}),
		`weighted admission resonance graft admission switch source_schema mismatch: got "arianna.live_route_weighted_admission_resonance_graft_admission_promotion.v0" want "`+admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionPromotionSchema+`"`,
	)

	badSourcePromotionKindPath := filepath.Join(dir, "bad_source_promotion_kind.json")
	writeWeightedAdmissionResonanceGraftAdmissionSwitchFixture(t, badSourcePromotionKindPath)
	writeWeightedReadinessFixture(t, badSourcePromotionKindPath, stringsReplaceFirst(readText(t, badSourcePromotionKindPath), `"source_promotion_kind": "shadow_graft_admission_promotion"`, `"source_promotion_kind": "live_promotion"`))
	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionSwitchAssert([]string{badSourcePromotionKindPath}),
		"weighted admission resonance graft admission switch source promotion shape mismatch",
	)

	badSourcePromotionPath := filepath.Join(dir, "bad_source_promotion.json")
	writeWeightedAdmissionResonanceGraftAdmissionSwitchFixture(t, badSourcePromotionPath)
	writeWeightedReadinessFixture(t, badSourcePromotionPath, stringsReplaceFirst(readText(t, badSourcePromotionPath), `"source_promotion": "pending_live_admission"`, `"source_promotion": "blocked"`))
	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionSwitchAssert([]string{badSourcePromotionPath}),
		`weighted admission resonance graft admission switch source_promotion mismatch: got "blocked" want "pending_live_admission"`,
	)

	badSwitchHashPath := filepath.Join(dir, "bad_switch_hash.json")
	writeWeightedAdmissionResonanceGraftAdmissionSwitchFixture(t, badSwitchHashPath)
	writeWeightedReadinessFixture(t, badSwitchHashPath, stringsReplaceFirst(readText(t, badSwitchHashPath), `"switch_hash": "weighted-resonance-graft-admission-switch-`, `"switch_hash": "weighted-resonance-graft-admission-switch-bad`))
	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionSwitchAssert([]string{badSwitchHashPath}),
		"weighted admission resonance graft admission switch switch_hash mismatch",
	)

	badBodyTargetPath := filepath.Join(dir, "bad_body_target.json")
	writeWeightedAdmissionResonanceGraftAdmissionSwitchFixture(t, badBodyTargetPath)
	writeWeightedReadinessFixture(t, badBodyTargetPath, stringsReplaceFirst(readText(t, badBodyTargetPath), `"body_target": "none"`, `"body_target": "live"`))
	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionSwitchAssert([]string{badBodyTargetPath}),
		`weighted admission resonance graft admission switch body_target mismatch: got "live" want "none"`,
	)
}

func writeWeightedAdmissionResonanceGraftAdmissionSwitchFixture(t *testing.T, switchPath string) {
	t.Helper()
	dir := filepath.Dir(switchPath)
	promotionPath := filepath.Join(dir, "promotion-"+filepath.Base(switchPath))
	writeWeightedAdmissionResonanceGraftAdmissionPromotionFixture(t, promotionPath)
	if err := runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionSwitch([]string{promotionPath, switchPath}); err != nil {
		t.Fatalf("write weighted admission resonance graft admission switch fixture: %v", err)
	}
}
