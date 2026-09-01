package main

import (
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateIntent(t *testing.T) {
	dir := t.TempDir()

	usage := "usage: --admission-live-route-weighted-admission-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-candidate-store-reader-proof-precondition-decision-promotion-switch-enable-gate-live-stage-final-gate-intent RESONANCE_GRAFT_ADMISSION_FINAL_GATE_OBSERVATION_BOUNDARY_PREFLIGHT_GATE_CANDIDATE_STORE_READER_PROOF_PRECONDITION_DECISION_PROMOTION_SWITCH_ENABLE_GATE_LIVE_STAGE_FINAL_GATE_REPORT RESONANCE_GRAFT_ADMISSION_FINAL_GATE_OBSERVATION_BOUNDARY_PREFLIGHT_GATE_CANDIDATE_STORE_READER_PROOF_PRECONDITION_DECISION_PROMOTION_SWITCH_ENABLE_GATE_LIVE_STAGE_FINAL_GATE_INTENT_REPORT"
	requireBoundaryAssertError(t, runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateIntent(nil), usage)
	requireBoundaryAssertError(t, runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateIntent([]string{"final_gate.json"}), usage)
	requireBoundaryAssertError(t, runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateIntent([]string{"final_gate.json", "intent.json", "extra"}), usage)
	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateIntent([]string{"  ", filepath.Join(dir, "intent.json")}),
		"weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage final gate path missing",
	)

	finalGatePath := filepath.Join(dir, "final_gate.json")
	writeWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateFixture(t, finalGatePath)
	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateIntent([]string{finalGatePath, "  "}),
		"weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage final gate intent output path missing",
	)

	intentPath := filepath.Join(dir, "intent.json")
	if err := runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateIntent([]string{finalGatePath, intentPath}); err != nil {
		t.Fatalf("valid weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage final gate intent rejected: %v", err)
	}
	raw, err := os.ReadFile(intentPath)
	if err != nil {
		t.Fatalf("read weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage final gate intent: %v", err)
	}
	var intent admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateIntentReport
	if err := json.Unmarshal(raw, &intent); err != nil {
		t.Fatalf("decode weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage final gate intent: %v", err)
	}
	sourceRaw, err := os.ReadFile(finalGatePath)
	if err != nil {
		t.Fatalf("read weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage final gate: %v", err)
	}
	var sourceFinalGate admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReport
	if err := json.Unmarshal(sourceRaw, &sourceFinalGate); err != nil {
		t.Fatalf("decode weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage final gate: %v", err)
	}
	if intent.Schema != admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateIntentSchema ||
		intent.Status != "shadow_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_final_gate_intent_blocked_dry_run" ||
		intent.Target != "live_route_admission_next_step" ||
		intent.TargetKind != "weighted_internal_world_shadow_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_final_gate_intent" ||
		intent.TargetMode != "bounded_intent_guard_dry_run" ||
		intent.Action != "draft_weighted_resonance_shadow_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_final_gate_intent_dry_run" ||
		intent.WriterAction != "reject_blocked_admission_final_gate_intent" ||
		intent.RollbackAction != "reject_blocked_admission_final_gate_intent" ||
		intent.LedgerState != "blocked" ||
		intent.LedgerAction != "reject_blocked_admission_final_gate_intent" ||
		intent.LedgerContract != "none" ||
		intent.LedgerEntrypoint != "none" ||
		intent.LedgerReceiptShape != "none" ||
		intent.LedgerWriteScope != "none" ||
		intent.ReceiptShape != "weighted_resonance_shadow_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_final_gate_intent_receipt" ||
		intent.AdmissionFinalGateIntentState != "blocked" ||
		intent.AdmissionFinalGateIntentAction != "draft_blocked_final_gate_intent" ||
		intent.AdmissionFinalGateIntentTarget != "resonance" ||
		intent.AdmissionFinalGateIntentTargetKind != "weighted_internal_world_shadow_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_final_gate" ||
		intent.AdmissionFinalGateIntentTargetMode != "bounded_intent_guard_dry_run" ||
		!intent.AdmissionFinalGateIntentDryRunOnly ||
		intent.AdmissionFinalGateIntentFinalGateVerified ||
		intent.AdmissionFinalGateIntentSealVerified ||
		intent.AdmissionFinalGateIntentReady ||
		intent.FinalGateIntentReceiver != "resonance" ||
		intent.FinalGateIntentReceiverKind != "internal_world" ||
		intent.FinalGateIntentInfluenceKind != "bounded_direction" ||
		intent.FinalGateIntentMaxInfluence != admissionLiveRouteTurnCandidateAdmissionResonanceIntentMaxGain ||
		intent.FinalGateIntentTTLTurns != admissionLiveRouteTurnCandidateAdmissionResonanceIntentTTL ||
		intent.FinalGateIntentRawDreamTextAllowed ||
		intent.FinalGateIntentJanusSurfaceAllowed ||
		intent.FinalGateIntentCoocLearningAllowed ||
		intent.FinalGateIntentDeltaHarvestAllowed ||
		!intent.FinalGateIntentPreStateHashRequired ||
		!intent.FinalGateIntentPostStateHashRequired ||
		!intent.WeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateIntentReady ||
		!intent.WeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateConsumed ||
		!intent.WeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateRequired ||
		!intent.NextStepBlockedWithoutResonanceGraftAdmissionFinalGateIntent ||
		intent.SourceSchema != admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateSchema ||
		intent.SourceStatus != "shadow_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_final_gate_blocked_dry_run" ||
		intent.SourceTarget != "live_route_admission_next_step" ||
		intent.SourceReport != finalGatePath ||
		intent.SourceAdmissionSealSchema != admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageSealSchema ||
		intent.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateID != sourceFinalGate.WeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateID ||
		intent.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateHash != sourceFinalGate.AdmissionFinalGateHash ||
		intent.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReadBack != sourceFinalGate.AdmissionFinalGateReadBackHash ||
		intent.SourceAdmissionFinalGateReportReceiptShape != sourceFinalGate.ReceiptShape ||
		intent.SourceAdmissionFinalGateAction != sourceFinalGate.AdmissionFinalGateAction ||
		!intent.SourceAdmissionFinalGateDryRunOnly ||
		intent.SourceAdmissionFinalGateSealVerified ||
		intent.SourceAdmissionFinalGateAuthorityVerified ||
		intent.SourceAdmissionFinalGatePermitVerified ||
		intent.SourceAdmissionFinalGateLedgerVerified ||
		intent.SourceAdmissionFinalGateReady ||
		intent.CausalID != admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateIntentCausalID(intent) ||
		intent.AdmissionFinalGateIntentHash != admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateIntentHash(intent) ||
		intent.AdmissionFinalGateIntentReadBackHash != admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateIntentReadBackHash(intent) ||
		intent.AdmissionFinalGateIntentHash == intent.AdmissionFinalGateIntentReadBackHash ||
		intent.WeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateIntentID != admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateIntentID(intent) ||
		intent.LedgerReady ||
		intent.LedgerAppendAllowed ||
		intent.WriteAllowed ||
		intent.AdmissionAllowed ||
		intent.LiveAdmissionEnabled ||
		intent.MutatesState ||
		intent.BodyMutationAllowed ||
		intent.AuthorityGranted ||
		intent.BodyTarget != "none" ||
		!intent.Passed ||
		intent.Reason != "weighted resonance shadow graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage final gate intent drafted from blocked final gate; live admission remains closed" {
		t.Fatalf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage final gate intent lost contract: %+v", intent)
	}

	notReadyPath := filepath.Join(dir, "not_ready_final_gate.json")
	writeWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateFixture(t, notReadyPath)
	writeWeightedReadinessFixture(t, notReadyPath, stringsReplaceFirst(readText(t, notReadyPath), `"weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_final_gate_ready": true`, `"weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_final_gate_ready": false`))
	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateIntent([]string{notReadyPath, filepath.Join(dir, "not_ready_intent.json")}),
		"weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage final gate weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_final_gate_ready not ready",
	)

	openedFinalGatePath := filepath.Join(dir, "opened_final_gate.json")
	writeWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateFixture(t, openedFinalGatePath)
	writeWeightedReadinessFixture(t, openedFinalGatePath, stringsReplaceFirst(readText(t, openedFinalGatePath), `"admission_final_gate_ready": false`, `"admission_final_gate_ready": true`))
	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateIntent([]string{openedFinalGatePath, filepath.Join(dir, "opened_intent.json")}),
		"weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage final gate opened admission_final_gate_ready",
	)

	badSchemaPath := filepath.Join(dir, "bad_schema.json")
	writeWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateFixture(t, badSchemaPath)
	writeWeightedReadinessFixture(t, badSchemaPath, stringsReplaceFirst(readText(t, badSchemaPath), `"schema": "arianna.live_route_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_final_gate.v1"`, `"schema": "arianna.live_route_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_final_gate.v0"`))
	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateIntent([]string{badSchemaPath, filepath.Join(dir, "bad_schema_intent.json")}),
		`weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage final gate schema mismatch: got "arianna.live_route_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_final_gate.v0" want "`+admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateSchema+`"`,
	)

	badHashPath := filepath.Join(dir, "bad_hash.json")
	writeWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateFixture(t, badHashPath)
	writeWeightedReadinessFixture(t, badHashPath, stringsReplaceFirst(readText(t, badHashPath), `"admission_final_gate_hash": "weighted-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-candidate-store-reader-proof-precondition-decision-promotion-switch-enable-gate-live-stage-final-gate-`, `"admission_final_gate_hash": "weighted-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-candidate-store-reader-proof-precondition-decision-promotion-switch-enable-gate-live-stage-final-gate-bad`))
	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateIntent([]string{badHashPath, filepath.Join(dir, "bad_hash_intent.json")}),
		"weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage final gate admission_final_gate_hash mismatch",
	)

	if err := runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateIntent([]string{finalGatePath, filepath.Join(dir, "missing", "intent.json")}); err == nil ||
		!strings.HasPrefix(err.Error(), "weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage final gate intent write failed:") {
		t.Fatalf("expected weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage final gate intent write failure, got %v", err)
	}
}
