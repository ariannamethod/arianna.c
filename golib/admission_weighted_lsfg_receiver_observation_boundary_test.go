package main

import (
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReceiverObservationBoundary(t *testing.T) {
	dir := t.TempDir()

	usage := "usage: --admission-live-route-weighted-admission-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-candidate-store-reader-proof-precondition-decision-promotion-switch-enable-gate-live-stage-final-gate-receiver-observation-boundary RESONANCE_GRAFT_ADMISSION_FINAL_GATE_OBSERVATION_BOUNDARY_PREFLIGHT_GATE_CANDIDATE_STORE_READER_PROOF_PRECONDITION_DECISION_PROMOTION_SWITCH_ENABLE_GATE_LIVE_STAGE_FINAL_GATE_RECEIVER_OBSERVATION_REPORT RESONANCE_GRAFT_ADMISSION_FINAL_GATE_OBSERVATION_BOUNDARY_PREFLIGHT_GATE_CANDIDATE_STORE_READER_PROOF_PRECONDITION_DECISION_PROMOTION_SWITCH_ENABLE_GATE_LIVE_STAGE_FINAL_GATE_RECEIVER_OBSERVATION_BOUNDARY_REPORT"
	requireBoundaryAssertError(t, runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReceiverObservationBoundary(nil), usage)
	requireBoundaryAssertError(t, runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReceiverObservationBoundary([]string{"observation.json"}), usage)
	requireBoundaryAssertError(t, runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReceiverObservationBoundary([]string{"observation.json", "boundary.json", "extra"}), usage)
	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReceiverObservationBoundary([]string{"  ", filepath.Join(dir, "boundary.json")}),
		"weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage final gate receiver observation path missing",
	)

	observationPath := filepath.Join(dir, "observation.json")
	writeWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReceiverObservationFixture(t, observationPath)
	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReceiverObservationBoundary([]string{observationPath, "  "}),
		"weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage final gate receiver observation boundary output path missing",
	)

	boundaryPath := filepath.Join(dir, "boundary.json")
	if err := runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReceiverObservationBoundary([]string{observationPath, boundaryPath}); err != nil {
		t.Fatalf("valid weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage final gate receiver observation boundary rejected: %v", err)
	}
	raw, err := os.ReadFile(boundaryPath)
	if err != nil {
		t.Fatalf("read weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage final gate receiver observation boundary: %v", err)
	}
	var boundary admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReceiverObservationBoundaryReport
	if err := json.Unmarshal(raw, &boundary); err != nil {
		t.Fatalf("decode weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage final gate receiver observation boundary: %v", err)
	}
	sourceRaw, err := os.ReadFile(observationPath)
	if err != nil {
		t.Fatalf("read weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage final gate receiver observation: %v", err)
	}
	var sourceObservation admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReceiverObservationReport
	if err := json.Unmarshal(sourceRaw, &sourceObservation); err != nil {
		t.Fatalf("decode weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage final gate receiver observation: %v", err)
	}
	if boundary.Schema != admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReceiverObservationBoundarySchema ||
		boundary.Status != "shadow_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_final_gate_receiver_observation_boundary_declared_dry_run" ||
		boundary.Target != "live_route_admission_next_step" ||
		boundary.TargetKind != "weighted_internal_world_shadow_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_final_gate_receiver_observation_boundary" ||
		boundary.TargetMode != "receipt_only_closed_dry_run" ||
		boundary.Action != "declare_weighted_resonance_shadow_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_final_gate_receiver_observation_boundary_dry_run" ||
		boundary.WriterAction != "reject_blocked_admission_final_gate_observation_boundary" ||
		boundary.RollbackAction != "reject_blocked_admission_final_gate_observation_boundary" ||
		boundary.LedgerState != "blocked" ||
		boundary.LedgerAction != "reject_blocked_admission_final_gate_observation_boundary" ||
		boundary.ReceiptShape != "weighted_resonance_shadow_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_final_gate_receiver_observation_boundary_receipt" ||
		boundary.AdmissionFinalGateObservationBoundaryState != "declared" ||
		boundary.AdmissionFinalGateObservationBoundaryAction != "declare_blocked_final_gate_observation_boundary" ||
		boundary.AdmissionFinalGateObservationBoundaryTarget != "resonance" ||
		boundary.AdmissionFinalGateObservationBoundaryTargetKind != "weighted_internal_world_shadow_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_final_gate_receiver_observation" ||
		boundary.AdmissionFinalGateObservationBoundaryTargetMode != "receipt_only_closed_dry_run" ||
		!boundary.AdmissionFinalGateObservationBoundaryDryRunOnly ||
		!boundary.AdmissionFinalGateObservationBoundaryObservationVerified ||
		!boundary.AdmissionFinalGateObservationBoundaryReadBackVerified ||
		boundary.AdmissionFinalGateObservationBoundaryReady ||
		boundary.FinalGateObservationBoundaryKind != "blocked_final_gate_observation_boundary" ||
		boundary.FinalGateObservationBoundaryMode != "no_mutation_closed_boundary_receipt" ||
		boundary.FinalGateObservationBoundaryStage != "post_observation_pre_live_admission" ||
		boundary.FinalGateObservationBoundaryRawDreamTextObserved ||
		boundary.FinalGateObservationBoundaryRawDreamTextForwarded ||
		boundary.FinalGateObservationBoundaryRawDreamTextAllowed ||
		boundary.FinalGateObservationBoundaryJanusSurfaceAllowed ||
		boundary.FinalGateObservationBoundaryCoocLearningAllowed ||
		boundary.FinalGateObservationBoundaryDeltaHarvestAllowed ||
		boundary.FinalGateObservationBoundaryBodyMutationAllowed ||
		!boundary.FinalGateObservationBoundaryPreStateHashRequired ||
		!boundary.FinalGateObservationBoundaryPostStateHashRequired ||
		!boundary.WeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReceiverObservationBoundaryReady ||
		!boundary.WeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReceiverObservationConsumed ||
		!boundary.WeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReceiverObservationRequired ||
		!boundary.NextStepBlockedWithoutResonanceGraftAdmissionFinalGateObservationBoundary ||
		boundary.SourceSchema != admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReceiverObservationSchema ||
		boundary.SourceStatus != "shadow_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_final_gate_receiver_observation_recorded_dry_run" ||
		boundary.SourceTarget != "live_route_admission_next_step" ||
		boundary.SourceReport != observationPath ||
		boundary.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReceiverObservationID != sourceObservation.WeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReceiverObservationID ||
		boundary.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReceiverObservationCausal != sourceObservation.CausalID ||
		boundary.SourceAdmissionFinalGateObservationAppendHash != sourceObservation.AdmissionFinalGateObservationAppendHash ||
		boundary.SourceAdmissionFinalGateObservationReadBackHash != sourceObservation.AdmissionFinalGateObservationReadBackHash ||
		boundary.SourceAdmissionFinalGateObservationReceiptShape != sourceObservation.ReceiptShape ||
		boundary.SourceAdmissionFinalGateObservationAction != sourceObservation.AdmissionFinalGateObservationAction ||
		!boundary.SourceAdmissionFinalGateObservationDryRunOnly ||
		!boundary.SourceAdmissionFinalGateObservationAppendOnly ||
		!boundary.SourceAdmissionFinalGateObservationReadBack ||
		!boundary.SourceAdmissionFinalGateObservationReceiptVerified ||
		boundary.SourceAdmissionFinalGateObservationReady ||
		boundary.SourceFinalGateObservationObserver != sourceObservation.FinalGateObservationObserver ||
		boundary.CausalID != admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReceiverObservationBoundaryCausalID(boundary) ||
		boundary.AdmissionFinalGateObservationBoundaryHash != admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReceiverObservationBoundaryHash(boundary) ||
		boundary.AdmissionFinalGateObservationBoundaryReadBackHash != admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReceiverObservationBoundaryReadBackHash(boundary) ||
		boundary.AdmissionFinalGateObservationBoundaryHash == boundary.AdmissionFinalGateObservationBoundaryReadBackHash ||
		boundary.WeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReceiverObservationBoundaryID != admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReceiverObservationBoundaryID(boundary) ||
		boundary.LedgerReady ||
		boundary.LedgerAppendAllowed ||
		boundary.WriteAllowed ||
		boundary.AdmissionAllowed ||
		boundary.LiveAdmissionEnabled ||
		boundary.MutatesState ||
		boundary.BodyMutationAllowed ||
		boundary.AuthorityGranted ||
		boundary.BodyTarget != "none" ||
		!boundary.Passed ||
		boundary.Reason != "weighted resonance shadow graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage final gate receiver observation boundary declared from recorded observation; live admission remains closed" {
		t.Fatalf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage final gate receiver observation boundary lost contract: %+v", boundary)
	}

	notReadyPath := filepath.Join(dir, "not_ready_observation.json")
	writeWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReceiverObservationFixture(t, notReadyPath)
	writeWeightedReadinessFixture(t, notReadyPath, stringsReplaceFirst(readText(t, notReadyPath), `"weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_final_gate_receiver_observation_ready": true`, `"weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_final_gate_receiver_observation_ready": false`))
	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReceiverObservationBoundary([]string{notReadyPath, filepath.Join(dir, "not_ready_boundary.json")}),
		"weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage final gate receiver observation weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_final_gate_receiver_observation_ready not ready",
	)

	openedObservationPath := filepath.Join(dir, "opened_observation.json")
	writeWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReceiverObservationFixture(t, openedObservationPath)
	writeWeightedReadinessFixture(t, openedObservationPath, stringsReplaceFirst(readText(t, openedObservationPath), `"admission_final_gate_observation_ready": false`, `"admission_final_gate_observation_ready": true`))
	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReceiverObservationBoundary([]string{openedObservationPath, filepath.Join(dir, "opened_boundary.json")}),
		"weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage final gate receiver observation opened admission_final_gate_observation_ready",
	)

	badHashPath := filepath.Join(dir, "bad_hash.json")
	writeWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReceiverObservationFixture(t, badHashPath)
	writeWeightedReadinessFixture(t, badHashPath, stringsReplaceFirst(readText(t, badHashPath), `"admission_final_gate_observation_append_hash": "weighted-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-candidate-store-reader-proof-precondition-decision-promotion-switch-enable-gate-live-stage-final-gate-receiver-observation-append-`, `"admission_final_gate_observation_append_hash": "weighted-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-candidate-store-reader-proof-precondition-decision-promotion-switch-enable-gate-live-stage-final-gate-receiver-observation-append-bad`))
	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReceiverObservationBoundary([]string{badHashPath, filepath.Join(dir, "bad_hash_boundary.json")}),
		"weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage final gate receiver observation append_hash mismatch",
	)

	if err := runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReceiverObservationBoundary([]string{observationPath, filepath.Join(dir, "missing", "boundary.json")}); err == nil ||
		!strings.HasPrefix(err.Error(), "weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage final gate receiver observation boundary write failed:") {
		t.Fatalf("expected weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage final gate receiver observation boundary write failure, got %v", err)
	}
}
