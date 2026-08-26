package main

import (
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageWriterInventory(t *testing.T) {
	dir := t.TempDir()

	usage := "usage: --admission-live-route-weighted-admission-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-candidate-store-reader-proof-precondition-decision-promotion-switch-enable-gate-live-stage-writer-inventory RESONANCE_GRAFT_ADMISSION_WRITER_PREFLIGHT_REPORT RESONANCE_GRAFT_ADMISSION_WRITER_INVENTORY_REPORT"
	requireBoundaryAssertError(t, runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageWriterInventory(nil), usage)
	requireBoundaryAssertError(t, runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageWriterInventory([]string{"writer_preflight.json"}), usage)
	requireBoundaryAssertError(t, runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageWriterInventory([]string{"writer_preflight.json", "writer_inventory.json", "extra"}), usage)
	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageWriterInventory([]string{"  ", filepath.Join(dir, "writer_inventory.json")}),
		"weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage writer preflight path missing",
	)

	writerPreflightPath := filepath.Join(dir, "writer_preflight.json")
	writeWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageWriterPreflightFixture(t, writerPreflightPath)
	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageWriterInventory([]string{writerPreflightPath, "  "}),
		"weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage writer inventory output path missing",
	)

	writerInventoryPath := filepath.Join(dir, "writer_inventory.json")
	if err := runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageWriterInventory([]string{writerPreflightPath, writerInventoryPath}); err != nil {
		t.Fatalf("valid weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage writer inventory rejected: %v", err)
	}
	raw, err := os.ReadFile(writerInventoryPath)
	if err != nil {
		t.Fatalf("read weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage writer inventory: %v", err)
	}
	var inventory admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageWriterInventoryReport
	if err := json.Unmarshal(raw, &inventory); err != nil {
		t.Fatalf("decode weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage writer inventory: %v", err)
	}
	sourceRaw, err := os.ReadFile(writerPreflightPath)
	if err != nil {
		t.Fatalf("read weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage writer preflight: %v", err)
	}
	var sourcePreflight admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageWriterPreflightReport
	if err := json.Unmarshal(sourceRaw, &sourcePreflight); err != nil {
		t.Fatalf("decode weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage writer preflight: %v", err)
	}
	if inventory.Schema != admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageWriterInventorySchema ||
		inventory.Status != "shadow_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_writer_inventory_blocked_dry_run" ||
		inventory.Target != "live_route_admission_next_step" ||
		inventory.TargetKind != "weighted_internal_world_shadow_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_writer_inventory" ||
		inventory.TargetMode != "closed_writer_inventory_guard_dry_run" ||
		inventory.Action != "block_weighted_resonance_shadow_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_writer_preflight_blocked_dry_run" ||
		inventory.WriterState != "blocked" ||
		inventory.WriterAction != "reject_blocked_writer_preflight" ||
		inventory.RollbackState != "blocked" ||
		inventory.RollbackAction != "reject_blocked_writer_preflight" ||
		inventory.StageState != "blocked" ||
		inventory.StageAction != "reject_disabled_enable_gate" ||
		inventory.EnableState != "disabled" ||
		inventory.EnableAction != "require_operator_key" ||
		inventory.SwitchState != "disabled" ||
		inventory.SwitchAction != "hold_pending_live_admission" ||
		inventory.Promotion != "pending_live_admission" ||
		inventory.InventoryState != "blocked" ||
		inventory.InventoryAction != "reject_blocked_writer_preflight" ||
		inventory.WriterContract != "none" ||
		inventory.RollbackContract != "none" ||
		inventory.AdmissionLedgerContract != "none" ||
		inventory.WriterContractPresent ||
		inventory.RollbackContractPresent ||
		inventory.LedgerContractPresent ||
		inventory.ContractsReady ||
		inventory.WriterInventoryMode != "closed_writer_preflight_inventory_guard" ||
		inventory.WriterInventoryStage != "post_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_writer_preflight_pre_writer_contract_inventory" ||
		!inventory.WeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageWriterInventoryReady ||
		!inventory.WeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageWriterPreflightConsumed ||
		!inventory.WeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageWriterPreflightRequired ||
		!inventory.NextStepBlockedWithoutResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageWriterInventory ||
		!inventory.WriterPreflightVerified ||
		!inventory.WriterPreflightHashVerified ||
		!inventory.WriterPreflightReadBackVerified ||
		!inventory.LiveStageVerified ||
		!inventory.EnableGateVerified ||
		!inventory.SwitchVerified ||
		!inventory.PromotionVerified ||
		!inventory.DecisionVerified ||
		!inventory.ProofPreconditionVerified ||
		!inventory.ProofVerified ||
		!inventory.StoreReaderVerified ||
		!inventory.CandidateVerified ||
		!inventory.AdmissionRequired ||
		!inventory.ShadowOnly ||
		inventory.GraftAllowed ||
		!inventory.DryRunOnly ||
		!inventory.LiveReady ||
		inventory.BodyMutationAllowed ||
		!inventory.RequiresWriter ||
		inventory.WriterReady ||
		!inventory.RollbackRequired ||
		!inventory.RequiresRollback ||
		inventory.RollbackReady ||
		!inventory.ReadOnly ||
		!inventory.ReplayOnly ||
		inventory.WriteAllowed ||
		inventory.AdmissionAllowed ||
		inventory.LiveAdmissionEnabled ||
		inventory.MutatesState ||
		inventory.BodyTarget != "none" ||
		!inventory.Passed ||
		inventory.Reason != "weighted resonance shadow graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage writer inventory blocked by blocked writer preflight; writer, rollback, and ledger contracts remain absent" ||
		inventory.CausalID != admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageWriterInventoryCausalID(inventory) ||
		inventory.WriterInventoryHash != admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageWriterInventoryHash(inventory) ||
		inventory.ReadBackHash != admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageWriterInventoryReadBackHash(inventory) ||
		inventory.WriterInventoryHash == inventory.ReadBackHash ||
		inventory.WeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageWriterInventoryID != admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageWriterInventoryID(inventory) ||
		inventory.SourceSchema != admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageWriterPreflightSchema ||
		inventory.SourceStatus != "shadow_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_writer_preflight_blocked_dry_run" ||
		inventory.SourceReport != writerPreflightPath ||
		inventory.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageWriterPreflightID != sourcePreflight.WeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageWriterPreflightID ||
		inventory.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageWriterPreflightHash != sourcePreflight.WriterPreflightHash ||
		inventory.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageWriterPreflightReadBack != sourcePreflight.ReadBackHash ||
		inventory.SourceWriterPreflightKind != sourcePreflight.WriterPreflightKind ||
		inventory.SourceWriterPreflightWriterAction != sourcePreflight.WriterAction ||
		inventory.SourceWriterPreflightRollbackAction != sourcePreflight.RollbackAction ||
		inventory.SourceWriterPreflightLiveAdmissionEnabled ||
		inventory.SourceWriterPreflightBodyTarget != "none" ||
		!inventory.SourceWriterPreflightPassed ||
		inventory.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageID != sourcePreflight.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageID ||
		inventory.SourceLiveStageKind != sourcePreflight.SourceLiveStageKind ||
		inventory.SourceLiveStageLiveAdmissionEnabled ||
		inventory.SourceLiveStageBodyTarget != "none" ||
		!inventory.SourceLiveStagePassed ||
		inventory.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateID != sourcePreflight.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateID ||
		inventory.SourceEnableState != "disabled" ||
		inventory.SourceEnableAction != "require_operator_key" ||
		inventory.SourceEnableGateLiveAdmissionEnabled ||
		inventory.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchID != sourcePreflight.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchID ||
		inventory.SourceSwitchState != "disabled" ||
		inventory.SourceSwitchAction != "hold_pending_live_admission" ||
		inventory.SourceSwitchLiveAdmissionEnabled ||
		inventory.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionID != sourcePreflight.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionID ||
		inventory.SourcePromotion != "pending_live_admission" ||
		inventory.SourcePromotionLiveAdmissionEnabled {
		t.Fatalf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage writer inventory lost contract: %+v", inventory)
	}

	openedPreflightPath := filepath.Join(dir, "opened_writer_preflight.json")
	writeWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageWriterPreflightFixture(t, openedPreflightPath)
	writeWeightedReadinessFixture(t, openedPreflightPath, stringsReplaceFirst(readText(t, openedPreflightPath), `"live_admission_enabled": false`, `"live_admission_enabled": true`))
	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageWriterInventory([]string{openedPreflightPath, filepath.Join(dir, "opened_writer_inventory.json")}),
		"weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage writer preflight opened live_admission_enabled",
	)

	badSchemaPath := filepath.Join(dir, "bad_schema_writer_preflight.json")
	writeWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageWriterPreflightFixture(t, badSchemaPath)
	writeWeightedReadinessFixture(t, badSchemaPath, stringsReplaceFirst(readText(t, badSchemaPath), `"schema": "arianna.live_route_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_writer_preflight.v1"`, `"schema": "arianna.live_route_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_writer_preflight.v0"`))
	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageWriterInventory([]string{badSchemaPath, filepath.Join(dir, "bad_schema_writer_inventory.json")}),
		`weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage writer preflight schema mismatch: got "arianna.live_route_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_writer_preflight.v0" want "`+admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageWriterPreflightSchema+`"`,
	)

	badHashPath := filepath.Join(dir, "bad_hash_writer_preflight.json")
	writeWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageWriterPreflightFixture(t, badHashPath)
	writeWeightedReadinessFixture(t, badHashPath, stringsReplaceFirst(readText(t, badHashPath), `"writer_preflight_hash": "weighted-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-candidate-store-reader-proof-precondition-decision-promotion-switch-enable-gate-live-stage-writer-preflight-`, `"writer_preflight_hash": "weighted-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-candidate-store-reader-proof-precondition-decision-promotion-switch-enable-gate-live-stage-writer-preflight-bad`))
	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageWriterInventory([]string{badHashPath, filepath.Join(dir, "bad_hash_writer_inventory.json")}),
		"weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage writer preflight writer_preflight_hash mismatch",
	)

	if err := runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageWriterInventory([]string{writerPreflightPath, filepath.Join(dir, "missing", "writer_inventory.json")}); err == nil ||
		!strings.HasPrefix(err.Error(), "weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage writer inventory write failed:") {
		t.Fatalf("expected weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage writer inventory write failure, got %v", err)
	}
}
