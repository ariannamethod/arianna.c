package main

import (
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReceiver(t *testing.T) {
	dir := t.TempDir()

	usage := "usage: --admission-live-route-weighted-admission-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-candidate-store-reader-proof-precondition-decision-promotion-switch-enable-gate-live-stage-final-gate-receiver RESONANCE_GRAFT_ADMISSION_FINAL_GATE_OBSERVATION_BOUNDARY_PREFLIGHT_GATE_CANDIDATE_STORE_READER_PROOF_PRECONDITION_DECISION_PROMOTION_SWITCH_ENABLE_GATE_LIVE_STAGE_FINAL_GATE_INTENT_REPORT RESONANCE_GRAFT_ADMISSION_FINAL_GATE_OBSERVATION_BOUNDARY_PREFLIGHT_GATE_CANDIDATE_STORE_READER_PROOF_PRECONDITION_DECISION_PROMOTION_SWITCH_ENABLE_GATE_LIVE_STAGE_FINAL_GATE_RECEIVER_REPORT"
	requireBoundaryAssertError(t, runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReceiver(nil), usage)
	requireBoundaryAssertError(t, runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReceiver([]string{"intent.json"}), usage)
	requireBoundaryAssertError(t, runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReceiver([]string{"intent.json", "receiver.json", "extra"}), usage)
	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReceiver([]string{"  ", filepath.Join(dir, "receiver.json")}),
		"weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage final gate intent path missing",
	)

	intentPath := filepath.Join(dir, "intent.json")
	writeWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateIntentFixture(t, intentPath)
	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReceiver([]string{intentPath, "  "}),
		"weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage final gate receiver output path missing",
	)

	receiverPath := filepath.Join(dir, "receiver.json")
	if err := runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReceiver([]string{intentPath, receiverPath}); err != nil {
		t.Fatalf("valid weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage final gate receiver rejected: %v", err)
	}
	raw, err := os.ReadFile(receiverPath)
	if err != nil {
		t.Fatalf("read weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage final gate receiver: %v", err)
	}
	var receiver admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReceiverReport
	if err := json.Unmarshal(raw, &receiver); err != nil {
		t.Fatalf("decode weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage final gate receiver: %v", err)
	}
	sourceRaw, err := os.ReadFile(intentPath)
	if err != nil {
		t.Fatalf("read weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage final gate intent: %v", err)
	}
	var sourceIntent admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateIntentReport
	if err := json.Unmarshal(sourceRaw, &sourceIntent); err != nil {
		t.Fatalf("decode weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage final gate intent: %v", err)
	}
	if receiver.Schema != admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReceiverSchema ||
		receiver.Status != "shadow_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_final_gate_receiver_previewed_dry_run" ||
		receiver.Target != "live_route_admission_next_step" ||
		receiver.TargetKind != "weighted_internal_world_shadow_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_final_gate_receiver" ||
		receiver.TargetMode != "bounded_receiver_preview_dry_run" ||
		receiver.Action != "preview_weighted_resonance_shadow_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_final_gate_receiver_dry_run" ||
		receiver.WriterAction != "reject_blocked_admission_final_gate_receiver" ||
		receiver.RollbackAction != "reject_blocked_admission_final_gate_receiver" ||
		receiver.LedgerState != "blocked" ||
		receiver.LedgerAction != "reject_blocked_admission_final_gate_receiver" ||
		receiver.LedgerContract != "none" ||
		receiver.LedgerEntrypoint != "none" ||
		receiver.LedgerReceiptShape != "none" ||
		receiver.LedgerWriteScope != "none" ||
		receiver.ReceiptShape != "weighted_resonance_shadow_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_final_gate_receiver_receipt" ||
		receiver.AdmissionFinalGateReceiverState != "previewed" ||
		receiver.AdmissionFinalGateReceiverAction != "preview_blocked_final_gate_receiver" ||
		receiver.AdmissionFinalGateReceiverTarget != "resonance" ||
		receiver.AdmissionFinalGateReceiverTargetKind != "weighted_internal_world_shadow_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_final_gate_intent" ||
		receiver.AdmissionFinalGateReceiverTargetMode != "bounded_receiver_preview_dry_run" ||
		!receiver.AdmissionFinalGateReceiverDryRunOnly ||
		receiver.AdmissionFinalGateReceiverIntentVerified ||
		receiver.AdmissionFinalGateReceiverFinalGateVerified ||
		receiver.AdmissionFinalGateReceiverReady ||
		receiver.FinalGateReceiver != "resonance" ||
		receiver.FinalGateReceiverKind != "internal_world" ||
		receiver.FinalGateReceiverInfluenceKind != "bounded_direction" ||
		receiver.FinalGateReceiverMaxInfluence != admissionLiveRouteTurnCandidateAdmissionResonanceIntentMaxGain ||
		receiver.FinalGateReceiverTTLTurns != admissionLiveRouteTurnCandidateAdmissionResonanceIntentTTL ||
		receiver.FinalGateReceiverStateHashMode != "blocked_intent_receiver_preview" ||
		receiver.FinalGateReceiverRawDreamTextObserved ||
		receiver.FinalGateReceiverRawDreamTextForwarded ||
		receiver.FinalGateReceiverRawDreamTextAllowed ||
		receiver.FinalGateReceiverJanusSurfaceAllowed ||
		receiver.FinalGateReceiverCoocLearningAllowed ||
		receiver.FinalGateReceiverDeltaHarvestAllowed ||
		receiver.FinalGateReceiverBodyMutationAllowed ||
		!receiver.FinalGateReceiverPreStateHashRequired ||
		!receiver.FinalGateReceiverPostStateHashRequired ||
		!receiver.WeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReceiverReady ||
		!receiver.WeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateIntentConsumed ||
		!receiver.WeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateIntentRequired ||
		!receiver.NextStepBlockedWithoutResonanceGraftAdmissionFinalGateReceiver ||
		receiver.SourceSchema != admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateIntentSchema ||
		receiver.SourceStatus != "shadow_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_final_gate_intent_blocked_dry_run" ||
		receiver.SourceTarget != "live_route_admission_next_step" ||
		receiver.SourceReport != intentPath ||
		receiver.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateIntentID != sourceIntent.WeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateIntentID ||
		receiver.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateIntentCausal != sourceIntent.CausalID ||
		receiver.SourceAdmissionFinalGateIntentHash != sourceIntent.AdmissionFinalGateIntentHash ||
		receiver.SourceAdmissionFinalGateIntentReadBack != sourceIntent.AdmissionFinalGateIntentReadBackHash ||
		receiver.SourceAdmissionFinalGateIntentReceiptShape != sourceIntent.ReceiptShape ||
		receiver.SourceAdmissionFinalGateIntentAction != sourceIntent.AdmissionFinalGateIntentAction ||
		!receiver.SourceAdmissionFinalGateIntentDryRunOnly ||
		receiver.SourceAdmissionFinalGateIntentFinalGateVerified ||
		receiver.SourceAdmissionFinalGateIntentSealVerified ||
		receiver.SourceAdmissionFinalGateIntentReady ||
		receiver.SourceFinalGateIntentReceiver != sourceIntent.FinalGateIntentReceiver ||
		receiver.SourceFinalGateIntentReceiverKind != sourceIntent.FinalGateIntentReceiverKind ||
		receiver.SourceFinalGateIntentInfluenceKind != sourceIntent.FinalGateIntentInfluenceKind ||
		receiver.SourceFinalGateIntentRawDreamTextAllowed ||
		receiver.SourceFinalGateIntentJanusSurfaceAllowed ||
		receiver.SourceFinalGateIntentCoocLearningAllowed ||
		receiver.SourceFinalGateIntentDeltaHarvestAllowed ||
		!receiver.SourceFinalGateIntentPreStateHashRequired ||
		!receiver.SourceFinalGateIntentPostStateHashRequired ||
		receiver.CausalID != admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReceiverCausalID(receiver) ||
		receiver.AdmissionFinalGateReceiverPreStateHash != admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReceiverPreStateHash(receiver) ||
		receiver.AdmissionFinalGateReceiverPostStateHash != admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReceiverPostStateHash(receiver) ||
		receiver.AdmissionFinalGateReceiverStateDeltaHash != admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReceiverStateDeltaHash(receiver) ||
		receiver.AdmissionFinalGateReceiverPreStateHash == receiver.AdmissionFinalGateReceiverPostStateHash ||
		receiver.WeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReceiverID != admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReceiverID(receiver) ||
		receiver.LedgerReady ||
		receiver.LedgerAppendAllowed ||
		receiver.WriteAllowed ||
		receiver.AdmissionAllowed ||
		receiver.LiveAdmissionEnabled ||
		receiver.MutatesState ||
		receiver.BodyMutationAllowed ||
		receiver.AuthorityGranted ||
		receiver.BodyTarget != "none" ||
		!receiver.Passed ||
		receiver.Reason != "weighted resonance shadow graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage final gate receiver previewed from blocked final gate intent; live admission remains closed" {
		t.Fatalf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage final gate receiver lost contract: %+v", receiver)
	}

	notReadyPath := filepath.Join(dir, "not_ready_intent.json")
	writeWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateIntentFixture(t, notReadyPath)
	writeWeightedReadinessFixture(t, notReadyPath, stringsReplaceFirst(readText(t, notReadyPath), `"weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_final_gate_intent_ready": true`, `"weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_final_gate_intent_ready": false`))
	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReceiver([]string{notReadyPath, filepath.Join(dir, "not_ready_receiver.json")}),
		"weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage final gate intent weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_final_gate_intent_ready not ready",
	)

	openedIntentPath := filepath.Join(dir, "opened_intent.json")
	writeWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateIntentFixture(t, openedIntentPath)
	writeWeightedReadinessFixture(t, openedIntentPath, stringsReplaceFirst(readText(t, openedIntentPath), `"admission_final_gate_intent_ready": false`, `"admission_final_gate_intent_ready": true`))
	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReceiver([]string{openedIntentPath, filepath.Join(dir, "opened_receiver.json")}),
		"weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage final gate intent opened admission_final_gate_intent_ready",
	)

	badSchemaPath := filepath.Join(dir, "bad_schema.json")
	writeWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateIntentFixture(t, badSchemaPath)
	writeWeightedReadinessFixture(t, badSchemaPath, stringsReplaceFirst(readText(t, badSchemaPath), `"schema": "arianna.live_route_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_final_gate_intent.v1"`, `"schema": "arianna.live_route_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_final_gate_intent.v0"`))
	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReceiver([]string{badSchemaPath, filepath.Join(dir, "bad_schema_receiver.json")}),
		`weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage final gate intent schema mismatch: got "arianna.live_route_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_final_gate_intent.v0" want "`+admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateIntentSchema+`"`,
	)

	badHashPath := filepath.Join(dir, "bad_hash.json")
	writeWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateIntentFixture(t, badHashPath)
	writeWeightedReadinessFixture(t, badHashPath, stringsReplaceFirst(readText(t, badHashPath), `"admission_final_gate_intent_hash": "weighted-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-candidate-store-reader-proof-precondition-decision-promotion-switch-enable-gate-live-stage-final-gate-intent-`, `"admission_final_gate_intent_hash": "weighted-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-candidate-store-reader-proof-precondition-decision-promotion-switch-enable-gate-live-stage-final-gate-intent-bad`))
	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReceiver([]string{badHashPath, filepath.Join(dir, "bad_hash_receiver.json")}),
		"weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage final gate intent admission_final_gate_intent_hash mismatch",
	)

	if err := runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReceiver([]string{intentPath, filepath.Join(dir, "missing", "receiver.json")}); err == nil ||
		!strings.HasPrefix(err.Error(), "weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage final gate receiver write failed:") {
		t.Fatalf("expected weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage final gate receiver write failure, got %v", err)
	}
}
