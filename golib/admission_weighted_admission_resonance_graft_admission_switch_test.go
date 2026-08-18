package main

import (
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionSwitch(t *testing.T) {
	dir := t.TempDir()

	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionSwitch(nil),
		"usage: --admission-live-route-weighted-admission-resonance-graft-admission-switch RESONANCE_GRAFT_ADMISSION_PROMOTION_REPORT RESONANCE_GRAFT_ADMISSION_SWITCH_REPORT",
	)
	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionSwitch([]string{"promotion.json"}),
		"usage: --admission-live-route-weighted-admission-resonance-graft-admission-switch RESONANCE_GRAFT_ADMISSION_PROMOTION_REPORT RESONANCE_GRAFT_ADMISSION_SWITCH_REPORT",
	)
	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionSwitch([]string{"promotion.json", "switch.json", "extra"}),
		"usage: --admission-live-route-weighted-admission-resonance-graft-admission-switch RESONANCE_GRAFT_ADMISSION_PROMOTION_REPORT RESONANCE_GRAFT_ADMISSION_SWITCH_REPORT",
	)
	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionSwitch([]string{"  ", filepath.Join(dir, "switch.json")}),
		"weighted admission resonance graft admission promotion path missing",
	)

	promotionPath := filepath.Join(dir, "promotion.json")
	writeWeightedAdmissionResonanceGraftAdmissionPromotionFixture(t, promotionPath)

	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionSwitch([]string{promotionPath, "  "}),
		"weighted admission resonance graft admission switch output path missing",
	)

	switchPath := filepath.Join(dir, "switch.json")
	if err := runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionSwitch([]string{promotionPath, switchPath}); err != nil {
		t.Fatalf("valid weighted admission resonance graft admission switch rejected: %v", err)
	}
	raw, err := os.ReadFile(switchPath)
	if err != nil {
		t.Fatalf("read weighted admission resonance graft admission switch: %v", err)
	}
	var sw admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionSwitchReport
	if err := json.Unmarshal(raw, &sw); err != nil {
		t.Fatalf("decode weighted admission resonance graft admission switch: %v", err)
	}
	promotionRaw, err := os.ReadFile(promotionPath)
	if err != nil {
		t.Fatalf("read weighted admission resonance graft admission promotion: %v", err)
	}
	var promotion admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionPromotionReport
	if err := json.Unmarshal(promotionRaw, &promotion); err != nil {
		t.Fatalf("decode weighted admission resonance graft admission promotion: %v", err)
	}
	if sw.Schema != admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionSwitchSchema ||
		sw.Status != "shadow_graft_admission_switch_disabled_dry_run" ||
		sw.Target != "live_route_admission_next_step" ||
		sw.TargetKind != "weighted_internal_world_shadow_graft_admission_switch" ||
		sw.TargetMode != "closed_switch_guard_dry_run" ||
		sw.Action != "hold_weighted_resonance_shadow_graft_admission_promotion_dry_run" ||
		sw.SwitchState != "disabled" ||
		sw.SwitchAction != "hold_pending_live_admission" ||
		sw.Promotion != "pending_live_admission" ||
		!sw.WeightedAdmissionResonanceGraftAdmissionSwitchReady ||
		!sw.WeightedAdmissionResonanceGraftAdmissionPromotionConsumed ||
		!sw.WeightedAdmissionResonanceGraftAdmissionPromotionRequired ||
		!sw.NextStepBlockedWithoutResonanceGraftAdmissionSwitch ||
		sw.ReceiptShape != "weighted_resonance_shadow_graft_admission_switch_receipt" ||
		sw.SwitchKind != "shadow_graft_admission_switch" ||
		sw.SwitchMode != "closed_promotion_switch_guard" ||
		sw.SwitchStage != "pre_live_graft_admission_switch" ||
		sw.CausalID != admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionSwitchCausalID(sw) ||
		sw.SwitchHash != admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionSwitchHash(sw) ||
		sw.ReadBackHash != admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionSwitchReadBackHash(sw) ||
		sw.SwitchHash == sw.ReadBackHash ||
		sw.WeightedAdmissionResonanceGraftAdmissionSwitchID != admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionSwitchID(sw) ||
		!sw.PromotionVerified ||
		!sw.PromotionHashVerified ||
		!sw.PromotionReadBackVerified ||
		!sw.DecisionVerified ||
		!sw.ProofVerified ||
		!sw.StoreReaderVerified ||
		!sw.CandidateVerified ||
		!sw.AuthorityVerified ||
		!sw.AdmissionRequired ||
		!sw.ShadowOnly ||
		sw.GraftAllowed ||
		!sw.DryRunOnly ||
		!sw.LiveReady ||
		sw.RawDreamTextAllowed ||
		sw.JanusSurfaceAllowed ||
		sw.CoocLearningAllowed ||
		sw.DeltaHarvestAllowed ||
		sw.BodyMutationAllowed ||
		!sw.RollbackRequired ||
		!sw.ReadOnly ||
		!sw.ReplayOnly ||
		sw.SourceSchema != admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionPromotionSchema ||
		sw.SourceStatus != "shadow_graft_admission_promotion_ready_dry_run" ||
		sw.SourceTarget != "live_route_admission_next_step" ||
		sw.SourceReport != promotionPath ||
		sw.SourceWeightedAdmissionResonanceGraftAdmissionPromotionID != promotion.WeightedAdmissionResonanceGraftAdmissionPromotionID ||
		!sw.SourceWeightedAdmissionResonanceGraftAdmissionPromotionReady ||
		sw.SourceWeightedAdmissionResonanceGraftAdmissionPromotionCausalID != promotion.CausalID ||
		sw.SourceWeightedAdmissionResonanceGraftAdmissionPromotionHash != promotion.PromotionHash ||
		sw.SourceWeightedAdmissionResonanceGraftAdmissionPromotionReadBack != promotion.ReadBackHash ||
		sw.SourcePromotion != "pending_live_admission" ||
		sw.SourcePromotionAction != "promote_weighted_resonance_shadow_graft_admission_dry_run" ||
		sw.SourcePromotionReceiptShape != "weighted_resonance_shadow_graft_admission_promotion_receipt" ||
		sw.SourcePromotionKind != "shadow_graft_admission_promotion" ||
		sw.SourcePromotionMode != "closed_decision_promotion" ||
		sw.SourcePromotionStage != "pre_live_graft_admission_promotion" ||
		!sw.SourcePromotionAdmissionRequired ||
		!sw.SourcePromotionShadowOnly ||
		sw.SourcePromotionGraftAllowed ||
		!sw.SourcePromotionDryRunOnly ||
		!sw.SourcePromotionLiveReady ||
		sw.SourcePromotionRawDreamTextAllowed ||
		sw.SourcePromotionBodyMutationAllowed ||
		!sw.SourcePromotionRollbackRequired ||
		!sw.SourcePromotionReadOnly ||
		!sw.SourcePromotionReplayOnly ||
		sw.SourcePromotionWriteAllowed ||
		sw.SourcePromotionAdmissionAllowed ||
		sw.SourcePromotionLiveAdmissionEnabled ||
		sw.SourcePromotionMutatesState ||
		sw.SourcePromotionBodyTarget != "none" ||
		!sw.SourcePromotionPassed ||
		sw.SourceWeightedAdmissionResonanceGraftAdmissionDecisionID != promotion.SourceWeightedAdmissionResonanceGraftAdmissionDecisionID ||
		sw.SourceWeightedAdmissionResonanceGraftAdmissionProofPreconditionID != promotion.SourceWeightedAdmissionResonanceGraftAdmissionProofPreconditionID ||
		sw.SourceWeightedAdmissionResonanceGraftAdmissionProofID != promotion.SourceWeightedAdmissionResonanceGraftAdmissionProofID ||
		sw.SourceWeightedAdmissionResonanceGraftCandidateStoreReaderID != promotion.SourceWeightedAdmissionResonanceGraftCandidateStoreReaderID ||
		sw.SourceWeightedAdmissionResonanceGraftCandidateStoreID != promotion.SourceWeightedAdmissionResonanceGraftCandidateStoreID ||
		sw.SourceWeightedAdmissionResonanceGraftCandidateID != promotion.SourceWeightedAdmissionResonanceGraftCandidateID ||
		sw.SourceWeightedAdmissionResonanceGraftGateID != promotion.SourceWeightedAdmissionResonanceGraftGateID ||
		sw.SourceWeightedAdmissionResonanceGraftPreflightID != promotion.SourceWeightedAdmissionResonanceGraftPreflightID ||
		sw.SourceWeightedAdmissionResonanceGraftBoundaryID != promotion.SourceWeightedAdmissionResonanceGraftBoundaryID ||
		sw.SourceWeightedAdmissionResonanceObservationID != promotion.SourceWeightedAdmissionResonanceObservationID ||
		sw.SourceWeightedAdmissionResonanceReceiverID != promotion.SourceWeightedAdmissionResonanceReceiverID ||
		!sw.BodySmokeWeighted ||
		!sw.NanoDirectRunner ||
		!sw.NanoDirectFinalGate ||
		!sw.ResonanceGraftAdmissionProof ||
		!sw.BoundaryReportFullChain ||
		sw.SourceAuthorityGranted ||
		sw.AuthorityGranted ||
		sw.ContractsReady ||
		sw.WriteAllowed ||
		sw.AdmissionAllowed ||
		sw.LiveAdmissionEnabled ||
		sw.MutatesState ||
		sw.BodyTarget != "none" ||
		!sw.Passed ||
		sw.Reason != "weighted resonance shadow graft admission promotion held at disabled switch without mutation" {
		t.Fatalf("weighted admission resonance graft admission switch lost contract: %+v", sw)
	}

	openedPath := filepath.Join(dir, "opened_promotion.json")
	writeWeightedAdmissionResonanceGraftAdmissionPromotionFixture(t, openedPath)
	writeWeightedReadinessFixture(t, openedPath, stringsReplaceFirst(readText(t, openedPath), `"live_admission_enabled": false`, `"live_admission_enabled": true`))
	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionSwitch([]string{openedPath, filepath.Join(dir, "opened_switch.json")}),
		"weighted admission resonance graft admission promotion opened live_admission_enabled",
	)

	badSchemaPath := filepath.Join(dir, "bad_schema_promotion.json")
	writeWeightedAdmissionResonanceGraftAdmissionPromotionFixture(t, badSchemaPath)
	writeWeightedReadinessFixture(t, badSchemaPath, stringsReplaceFirst(readText(t, badSchemaPath), `"schema": "arianna.live_route_weighted_admission_resonance_graft_admission_promotion.v1"`, `"schema": "arianna.live_route_weighted_admission_resonance_graft_admission_promotion.v0"`))
	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionSwitch([]string{badSchemaPath, filepath.Join(dir, "bad_schema_switch.json")}),
		`weighted admission resonance graft admission promotion schema mismatch: got "arianna.live_route_weighted_admission_resonance_graft_admission_promotion.v0" want "`+admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionPromotionSchema+`"`,
	)

	badHashPath := filepath.Join(dir, "bad_hash_promotion.json")
	writeWeightedAdmissionResonanceGraftAdmissionPromotionFixture(t, badHashPath)
	writeWeightedReadinessFixture(t, badHashPath, stringsReplaceFirst(readText(t, badHashPath), `"promotion_hash": "weighted-resonance-graft-admission-promotion-`, `"promotion_hash": "weighted-resonance-graft-admission-promotion-bad`))
	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionSwitch([]string{badHashPath, filepath.Join(dir, "bad_hash_switch.json")}),
		"weighted admission resonance graft admission promotion promotion_hash mismatch",
	)

	if err := runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionSwitch([]string{promotionPath, filepath.Join(dir, "missing", "switch.json")}); err == nil ||
		!strings.HasPrefix(err.Error(), "weighted admission resonance graft admission switch write failed:") {
		t.Fatalf("expected weighted admission resonance graft admission switch write failure, got %v", err)
	}
}
