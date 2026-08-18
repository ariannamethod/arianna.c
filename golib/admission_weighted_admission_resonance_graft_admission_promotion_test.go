package main

import (
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionPromotion(t *testing.T) {
	dir := t.TempDir()

	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionPromotion(nil),
		"usage: --admission-live-route-weighted-admission-resonance-graft-admission-promotion RESONANCE_GRAFT_ADMISSION_DECISION_REPORT RESONANCE_GRAFT_ADMISSION_PROMOTION_REPORT",
	)
	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionPromotion([]string{"decision.json"}),
		"usage: --admission-live-route-weighted-admission-resonance-graft-admission-promotion RESONANCE_GRAFT_ADMISSION_DECISION_REPORT RESONANCE_GRAFT_ADMISSION_PROMOTION_REPORT",
	)
	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionPromotion([]string{"decision.json", "promotion.json", "extra"}),
		"usage: --admission-live-route-weighted-admission-resonance-graft-admission-promotion RESONANCE_GRAFT_ADMISSION_DECISION_REPORT RESONANCE_GRAFT_ADMISSION_PROMOTION_REPORT",
	)
	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionPromotion([]string{"  ", filepath.Join(dir, "promotion.json")}),
		"weighted admission resonance graft admission decision path missing",
	)

	decisionPath := filepath.Join(dir, "decision.json")
	writeWeightedAdmissionResonanceGraftAdmissionDecisionFixture(t, decisionPath)

	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionPromotion([]string{decisionPath, "  "}),
		"weighted admission resonance graft admission promotion output path missing",
	)

	promotionPath := filepath.Join(dir, "promotion.json")
	if err := runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionPromotion([]string{decisionPath, promotionPath}); err != nil {
		t.Fatalf("valid weighted admission resonance graft admission promotion rejected: %v", err)
	}
	raw, err := os.ReadFile(promotionPath)
	if err != nil {
		t.Fatalf("read weighted admission resonance graft admission promotion: %v", err)
	}
	var promotion admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionPromotionReport
	if err := json.Unmarshal(raw, &promotion); err != nil {
		t.Fatalf("decode weighted admission resonance graft admission promotion: %v", err)
	}
	decisionRaw, err := os.ReadFile(decisionPath)
	if err != nil {
		t.Fatalf("read weighted admission resonance graft admission decision: %v", err)
	}
	var decision admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionDecisionReport
	if err := json.Unmarshal(decisionRaw, &decision); err != nil {
		t.Fatalf("decode weighted admission resonance graft admission decision: %v", err)
	}
	if promotion.Schema != admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionPromotionSchema ||
		promotion.Status != "shadow_graft_admission_promotion_ready_dry_run" ||
		promotion.Target != "live_route_admission_next_step" ||
		promotion.TargetKind != "weighted_internal_world_shadow_graft_admission_promotion" ||
		promotion.TargetMode != "closed_promotion_receipt_dry_run" ||
		promotion.Action != "promote_weighted_resonance_shadow_graft_admission_dry_run" ||
		promotion.Promotion != "pending_live_admission" ||
		!promotion.WeightedAdmissionResonanceGraftAdmissionPromotionReady ||
		!promotion.WeightedAdmissionResonanceGraftAdmissionDecisionConsumed ||
		!promotion.WeightedAdmissionResonanceGraftAdmissionDecisionRequired ||
		!promotion.NextStepBlockedWithoutResonanceGraftAdmissionPromotion ||
		promotion.ReceiptShape != "weighted_resonance_shadow_graft_admission_promotion_receipt" ||
		promotion.PromotionKind != "shadow_graft_admission_promotion" ||
		promotion.PromotionMode != "closed_decision_promotion" ||
		promotion.PromotionStage != "pre_live_graft_admission_promotion" ||
		promotion.CausalID != admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionPromotionCausalID(promotion) ||
		promotion.PromotionHash != admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionPromotionHash(promotion) ||
		promotion.ReadBackHash != admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionPromotionReadBackHash(promotion) ||
		promotion.PromotionHash == promotion.ReadBackHash ||
		promotion.WeightedAdmissionResonanceGraftAdmissionPromotionID != admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionPromotionID(promotion) ||
		!promotion.DecisionVerified ||
		!promotion.DecisionHashVerified ||
		!promotion.DecisionReadBackVerified ||
		!promotion.ProofPreconditionVerified ||
		!promotion.ProofVerified ||
		!promotion.StoreReaderVerified ||
		!promotion.CandidateVerified ||
		!promotion.AuthorityVerified ||
		!promotion.AdmissionRequired ||
		!promotion.ShadowOnly ||
		promotion.GraftAllowed ||
		!promotion.DryRunOnly ||
		!promotion.LiveReady ||
		promotion.RawDreamTextAllowed ||
		promotion.JanusSurfaceAllowed ||
		promotion.CoocLearningAllowed ||
		promotion.DeltaHarvestAllowed ||
		promotion.BodyMutationAllowed ||
		!promotion.RollbackRequired ||
		!promotion.ReadOnly ||
		!promotion.ReplayOnly ||
		promotion.SourceSchema != admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionDecisionSchema ||
		promotion.SourceStatus != "shadow_graft_admission_decision_ready_dry_run" ||
		promotion.SourceTarget != "live_route_admission_next_step" ||
		promotion.SourceReport != decisionPath ||
		promotion.SourceWeightedAdmissionResonanceGraftAdmissionDecisionID != decision.WeightedAdmissionResonanceGraftAdmissionDecisionID ||
		!promotion.SourceWeightedAdmissionResonanceGraftAdmissionDecisionReady ||
		promotion.SourceWeightedAdmissionResonanceGraftAdmissionDecisionCausalID != decision.CausalID ||
		promotion.SourceWeightedAdmissionResonanceGraftAdmissionDecisionHash != decision.DecisionHash ||
		promotion.SourceWeightedAdmissionResonanceGraftAdmissionDecisionReadBack != decision.ReadBackHash ||
		promotion.SourceDecision != "shadow_ready" ||
		promotion.SourceDecisionAction != "decide_weighted_resonance_shadow_graft_admission_dry_run" ||
		promotion.SourceDecisionReceiptShape != "weighted_resonance_shadow_graft_admission_decision_receipt" ||
		promotion.SourceDecisionKind != "shadow_graft_admission_decision" ||
		promotion.SourceDecisionMode != "closed_precondition_decision" ||
		promotion.SourceDecisionStage != "pre_live_graft_admission_decision" ||
		!promotion.SourceDecisionAdmissionRequired ||
		!promotion.SourceDecisionShadowOnly ||
		promotion.SourceDecisionGraftAllowed ||
		!promotion.SourceDecisionDryRunOnly ||
		!promotion.SourceDecisionLiveReady ||
		promotion.SourceDecisionRawDreamTextAllowed ||
		promotion.SourceDecisionBodyMutationAllowed ||
		!promotion.SourceDecisionRollbackRequired ||
		!promotion.SourceDecisionReadOnly ||
		!promotion.SourceDecisionReplayOnly ||
		promotion.SourceDecisionWriteAllowed ||
		promotion.SourceDecisionAdmissionAllowed ||
		promotion.SourceDecisionLiveAdmissionEnabled ||
		promotion.SourceDecisionMutatesState ||
		promotion.SourceDecisionBodyTarget != "none" ||
		!promotion.SourceDecisionPassed ||
		promotion.SourceWeightedAdmissionResonanceGraftAdmissionProofPreconditionID != decision.SourceWeightedAdmissionResonanceGraftAdmissionProofPreconditionID ||
		promotion.SourceWeightedAdmissionResonanceGraftAdmissionProofID != decision.SourceWeightedAdmissionResonanceGraftAdmissionProofID ||
		promotion.SourceWeightedAdmissionResonanceGraftCandidateStoreReaderID != decision.SourceWeightedAdmissionResonanceGraftCandidateStoreReaderID ||
		promotion.SourceWeightedAdmissionResonanceGraftCandidateStoreID != decision.SourceWeightedAdmissionResonanceGraftCandidateStoreID ||
		promotion.SourceWeightedAdmissionResonanceGraftCandidateID != decision.SourceWeightedAdmissionResonanceGraftCandidateID ||
		promotion.SourceWeightedAdmissionResonanceGraftGateID != decision.SourceWeightedAdmissionResonanceGraftGateID ||
		promotion.SourceWeightedAdmissionResonanceGraftPreflightID != decision.SourceWeightedAdmissionResonanceGraftPreflightID ||
		promotion.SourceWeightedAdmissionResonanceGraftBoundaryID != decision.SourceWeightedAdmissionResonanceGraftBoundaryID ||
		promotion.SourceWeightedAdmissionResonanceObservationID != decision.SourceWeightedAdmissionResonanceObservationID ||
		promotion.SourceWeightedAdmissionResonanceReceiverID != decision.SourceWeightedAdmissionResonanceReceiverID ||
		!promotion.BodySmokeWeighted ||
		!promotion.NanoDirectRunner ||
		!promotion.NanoDirectFinalGate ||
		!promotion.ResonanceGraftAdmissionProof ||
		!promotion.BoundaryReportFullChain ||
		promotion.SourceAuthorityGranted ||
		promotion.AuthorityGranted ||
		promotion.ContractsReady ||
		promotion.WriteAllowed ||
		promotion.AdmissionAllowed ||
		promotion.LiveAdmissionEnabled ||
		promotion.MutatesState ||
		promotion.BodyTarget != "none" ||
		!promotion.Passed ||
		promotion.Reason != "weighted resonance shadow graft admission decision promoted as pending live admission while closed" {
		t.Fatalf("weighted admission resonance graft admission promotion lost contract: %+v", promotion)
	}

	openedPath := filepath.Join(dir, "opened_decision.json")
	writeWeightedAdmissionResonanceGraftAdmissionDecisionFixture(t, openedPath)
	writeWeightedReadinessFixture(t, openedPath, stringsReplaceFirst(readText(t, openedPath), `"live_admission_enabled": false`, `"live_admission_enabled": true`))
	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionPromotion([]string{openedPath, filepath.Join(dir, "opened_promotion.json")}),
		"weighted admission resonance graft admission decision opened live_admission_enabled",
	)

	badSchemaPath := filepath.Join(dir, "bad_schema_decision.json")
	writeWeightedAdmissionResonanceGraftAdmissionDecisionFixture(t, badSchemaPath)
	writeWeightedReadinessFixture(t, badSchemaPath, stringsReplaceFirst(readText(t, badSchemaPath), `"schema": "arianna.live_route_weighted_admission_resonance_graft_admission_decision.v1"`, `"schema": "arianna.live_route_weighted_admission_resonance_graft_admission_decision.v0"`))
	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionPromotion([]string{badSchemaPath, filepath.Join(dir, "bad_schema_promotion.json")}),
		`weighted admission resonance graft admission decision schema mismatch: got "arianna.live_route_weighted_admission_resonance_graft_admission_decision.v0" want "`+admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionDecisionSchema+`"`,
	)

	badHashPath := filepath.Join(dir, "bad_hash_decision.json")
	writeWeightedAdmissionResonanceGraftAdmissionDecisionFixture(t, badHashPath)
	writeWeightedReadinessFixture(t, badHashPath, stringsReplaceFirst(readText(t, badHashPath), `"decision_hash": "weighted-resonance-graft-admission-decision-`, `"decision_hash": "weighted-resonance-graft-admission-decision-bad`))
	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionPromotion([]string{badHashPath, filepath.Join(dir, "bad_hash_promotion.json")}),
		"weighted admission resonance graft admission decision decision_hash mismatch",
	)

	if err := runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionPromotion([]string{decisionPath, filepath.Join(dir, "missing", "promotion.json")}); err == nil ||
		!strings.HasPrefix(err.Error(), "weighted admission resonance graft admission promotion write failed:") {
		t.Fatalf("expected weighted admission resonance graft admission promotion write failure, got %v", err)
	}
}
