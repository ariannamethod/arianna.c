package main

import (
	"path/filepath"
	"strings"
	"testing"
)

func TestAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionPromotionAssert(t *testing.T) {
	dir := t.TempDir()

	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionPromotionAssert(nil),
		"usage: --admission-live-route-weighted-admission-resonance-graft-admission-promotion-assert REPORT",
	)
	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionPromotionAssert([]string{"promotion.json", "extra"}),
		"usage: --admission-live-route-weighted-admission-resonance-graft-admission-promotion-assert REPORT",
	)
	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionPromotionAssert([]string{"  "}),
		"weighted admission resonance graft admission promotion path missing",
	)
	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionPromotionAssert([]string{filepath.Join(dir, "missing.json")}),
		"weighted admission resonance graft admission promotion not written",
	)

	emptyPath := filepath.Join(dir, "empty.json")
	writeWeightedReadinessFixture(t, emptyPath, "")
	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionPromotionAssert([]string{emptyPath}),
		"weighted admission resonance graft admission promotion not written",
	)

	invalidPath := filepath.Join(dir, "invalid.json")
	writeWeightedReadinessFixture(t, invalidPath, "{")
	err := runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionPromotionAssert([]string{invalidPath})
	if err == nil || !strings.HasPrefix(err.Error(), "weighted admission resonance graft admission promotion JSON invalid:") {
		t.Fatalf("expected invalid JSON error, got %v", err)
	}

	missingSchemaPath := filepath.Join(dir, "missing_schema.json")
	writeWeightedAdmissionResonanceGraftAdmissionPromotionFixture(t, missingSchemaPath)
	promotionText := readText(t, missingSchemaPath)
	writeWeightedReadinessFixture(t, missingSchemaPath, stringsReplaceFirst(promotionText, `"schema": "arianna.live_route_weighted_admission_resonance_graft_admission_promotion.v1",`, ""))
	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionPromotionAssert([]string{missingSchemaPath}),
		"weighted admission resonance graft admission promotion schema missing",
	)

	badSchemaPath := filepath.Join(dir, "bad_schema.json")
	writeWeightedAdmissionResonanceGraftAdmissionPromotionFixture(t, badSchemaPath)
	writeWeightedReadinessFixture(t, badSchemaPath, stringsReplaceFirst(readText(t, badSchemaPath), `"schema": "arianna.live_route_weighted_admission_resonance_graft_admission_promotion.v1"`, `"schema": "arianna.live_route_weighted_admission_resonance_graft_admission_promotion.v0"`))
	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionPromotionAssert([]string{badSchemaPath}),
		`weighted admission resonance graft admission promotion schema mismatch: got "arianna.live_route_weighted_admission_resonance_graft_admission_promotion.v0" want "`+admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionPromotionSchema+`"`,
	)

	validPath := filepath.Join(dir, "valid.json")
	writeWeightedAdmissionResonanceGraftAdmissionPromotionFixture(t, validPath)
	if err := runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionPromotionAssert([]string{validPath}); err != nil {
		t.Fatalf("valid weighted admission resonance graft admission promotion rejected: %v", err)
	}

	badStatusPath := filepath.Join(dir, "bad_status.json")
	writeWeightedAdmissionResonanceGraftAdmissionPromotionFixture(t, badStatusPath)
	writeWeightedReadinessFixture(t, badStatusPath, stringsReplaceFirst(readText(t, badStatusPath), `"status": "shadow_graft_admission_promotion_ready_dry_run"`, `"status": "open"`))
	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionPromotionAssert([]string{badStatusPath}),
		`weighted admission resonance graft admission promotion status mismatch: got "open" want "shadow_graft_admission_promotion_ready_dry_run"`,
	)

	notReadyPath := filepath.Join(dir, "not_ready.json")
	writeWeightedAdmissionResonanceGraftAdmissionPromotionFixture(t, notReadyPath)
	writeWeightedReadinessFixture(t, notReadyPath, stringsReplaceFirst(readText(t, notReadyPath), `"weighted_admission_resonance_graft_admission_promotion_ready": true`, `"weighted_admission_resonance_graft_admission_promotion_ready": false`))
	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionPromotionAssert([]string{notReadyPath}),
		"weighted admission resonance graft admission promotion weighted_admission_resonance_graft_admission_promotion_ready not ready",
	)

	badPromotionPath := filepath.Join(dir, "bad_promotion.json")
	writeWeightedAdmissionResonanceGraftAdmissionPromotionFixture(t, badPromotionPath)
	writeWeightedReadinessFixture(t, badPromotionPath, stringsReplaceFirst(readText(t, badPromotionPath), `"promotion": "pending_live_admission"`, `"promotion": "blocked"`))
	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionPromotionAssert([]string{badPromotionPath}),
		`weighted admission resonance graft admission promotion promotion mismatch: got "blocked" want "pending_live_admission"`,
	)

	openedPath := filepath.Join(dir, "opened.json")
	writeWeightedAdmissionResonanceGraftAdmissionPromotionFixture(t, openedPath)
	writeWeightedReadinessFixture(t, openedPath, stringsReplaceFirst(readText(t, openedPath), `"live_admission_enabled": false`, `"live_admission_enabled": true`))
	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionPromotionAssert([]string{openedPath}),
		"weighted admission resonance graft admission promotion opened live_admission_enabled",
	)

	missingPathField := filepath.Join(dir, "missing_path.json")
	writeWeightedAdmissionResonanceGraftAdmissionPromotionFixture(t, missingPathField)
	decisionReport := filepath.Join(dir, "decision-"+filepath.Base(missingPathField))
	writeWeightedReadinessFixture(t, missingPathField, stringsReplaceFirst(readText(t, missingPathField), `"source_report": "`+decisionReport+`"`, `"source_report": " "`))
	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionPromotionAssert([]string{missingPathField}),
		"weighted admission resonance graft admission promotion source_report missing",
	)

	badSourcePath := filepath.Join(dir, "bad_source.json")
	writeWeightedAdmissionResonanceGraftAdmissionPromotionFixture(t, badSourcePath)
	writeWeightedReadinessFixture(t, badSourcePath, stringsReplaceFirst(readText(t, badSourcePath), `"source_schema": "arianna.live_route_weighted_admission_resonance_graft_admission_decision.v1"`, `"source_schema": "arianna.live_route_weighted_admission_resonance_graft_admission_decision.v0"`))
	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionPromotionAssert([]string{badSourcePath}),
		`weighted admission resonance graft admission promotion source_schema mismatch: got "arianna.live_route_weighted_admission_resonance_graft_admission_decision.v0" want "`+admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionDecisionSchema+`"`,
	)

	badSourceDecisionKindPath := filepath.Join(dir, "bad_source_decision_kind.json")
	writeWeightedAdmissionResonanceGraftAdmissionPromotionFixture(t, badSourceDecisionKindPath)
	writeWeightedReadinessFixture(t, badSourceDecisionKindPath, stringsReplaceFirst(readText(t, badSourceDecisionKindPath), `"source_decision_kind": "shadow_graft_admission_decision"`, `"source_decision_kind": "live_decision"`))
	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionPromotionAssert([]string{badSourceDecisionKindPath}),
		"weighted admission resonance graft admission promotion source decision shape mismatch",
	)

	badSourceDecisionPath := filepath.Join(dir, "bad_source_decision.json")
	writeWeightedAdmissionResonanceGraftAdmissionPromotionFixture(t, badSourceDecisionPath)
	writeWeightedReadinessFixture(t, badSourceDecisionPath, stringsReplaceFirst(readText(t, badSourceDecisionPath), `"source_decision": "shadow_ready"`, `"source_decision": "reject"`))
	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionPromotionAssert([]string{badSourceDecisionPath}),
		`weighted admission resonance graft admission promotion source_decision mismatch: got "reject" want "shadow_ready"`,
	)

	badPromotionHashPath := filepath.Join(dir, "bad_promotion_hash.json")
	writeWeightedAdmissionResonanceGraftAdmissionPromotionFixture(t, badPromotionHashPath)
	writeWeightedReadinessFixture(t, badPromotionHashPath, stringsReplaceFirst(readText(t, badPromotionHashPath), `"promotion_hash": "weighted-resonance-graft-admission-promotion-`, `"promotion_hash": "weighted-resonance-graft-admission-promotion-bad`))
	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionPromotionAssert([]string{badPromotionHashPath}),
		"weighted admission resonance graft admission promotion promotion_hash mismatch",
	)

	badBodyTargetPath := filepath.Join(dir, "bad_body_target.json")
	writeWeightedAdmissionResonanceGraftAdmissionPromotionFixture(t, badBodyTargetPath)
	writeWeightedReadinessFixture(t, badBodyTargetPath, stringsReplaceFirst(readText(t, badBodyTargetPath), `"body_target": "none"`, `"body_target": "live"`))
	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionPromotionAssert([]string{badBodyTargetPath}),
		`weighted admission resonance graft admission promotion body_target mismatch: got "live" want "none"`,
	)
}

func writeWeightedAdmissionResonanceGraftAdmissionPromotionFixture(t *testing.T, promotionPath string) {
	t.Helper()
	dir := filepath.Dir(promotionPath)
	decisionPath := filepath.Join(dir, "decision-"+filepath.Base(promotionPath))
	writeWeightedAdmissionResonanceGraftAdmissionDecisionFixture(t, decisionPath)
	if err := runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionPromotion([]string{decisionPath, promotionPath}); err != nil {
		t.Fatalf("write weighted admission resonance graft admission promotion fixture: %v", err)
	}
}
