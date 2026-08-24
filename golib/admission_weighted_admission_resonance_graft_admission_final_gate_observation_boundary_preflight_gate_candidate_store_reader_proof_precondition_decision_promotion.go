package main

import (
	"encoding/json"
	"fmt"
	"os"
	"strings"
)

const admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSchema = "arianna.live_route_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion.v1"

type admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionReport struct {
	Schema                                                                                                                                       string `json:"schema"`
	Status                                                                                                                                       string `json:"status"`
	Target                                                                                                                                       string `json:"target"`
	TargetKind                                                                                                                                   string `json:"target_kind"`
	TargetMode                                                                                                                                   string `json:"target_mode"`
	Action                                                                                                                                       string `json:"action"`
	Promotion                                                                                                                                    string `json:"promotion"`
	LedgerState                                                                                                                                  string `json:"ledger_state"`
	LedgerAction                                                                                                                                 string `json:"ledger_action"`
	LedgerContract                                                                                                                               string `json:"ledger_contract"`
	LedgerEntrypoint                                                                                                                             string `json:"ledger_entrypoint"`
	LedgerReceiptShape                                                                                                                           string `json:"ledger_receipt_shape"`
	LedgerWriteScope                                                                                                                             string `json:"ledger_write_scope"`
	LedgerReady                                                                                                                                  bool   `json:"ledger_ready"`
	LedgerAppendAllowed                                                                                                                          bool   `json:"ledger_append_allowed"`
	WeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionReady bool   `json:"weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_ready"`
	WeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionConsumed       bool   `json:"weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_consumed"`
	WeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionRequired       bool   `json:"weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_required"`
	NextStepBlockedWithoutResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotion bool   `json:"next_step_blocked_without_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion"`
	WeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionID    string `json:"weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_id"`
	ReceiptShape                                                                                                                                 string `json:"receipt_shape"`
	PromotionKind                                                                                                                                string `json:"promotion_kind"`
	PromotionMode                                                                                                                                string `json:"promotion_mode"`
	PromotionStage                                                                                                                               string `json:"promotion_stage"`
	CausalID                                                                                                                                     string `json:"causal_id"`
	PromotionHash                                                                                                                                string `json:"promotion_hash"`
	ReadBackHash                                                                                                                                 string `json:"read_back_hash"`
	DecisionVerified                                                                                                                             bool   `json:"decision_verified"`
	DecisionHashVerified                                                                                                                         bool   `json:"decision_hash_verified"`
	DecisionReadBackVerified                                                                                                                     bool   `json:"decision_read_back_verified"`
	ProofPreconditionVerified                                                                                                                    bool   `json:"proof_precondition_verified"`
	PreconditionHashVerified                                                                                                                     bool   `json:"precondition_hash_verified"`
	PreconditionReadBackVerified                                                                                                                 bool   `json:"precondition_read_back_verified"`
	ProofVerified                                                                                                                                bool   `json:"proof_verified"`
	ProofHashVerified                                                                                                                            bool   `json:"proof_hash_verified"`
	ProofReadBackVerified                                                                                                                        bool   `json:"proof_read_back_verified"`
	StoreReaderVerified                                                                                                                          bool   `json:"store_reader_verified"`
	StoreVerified                                                                                                                                bool   `json:"store_verified"`
	CandidateVerified                                                                                                                            bool   `json:"candidate_verified"`
	GateVerified                                                                                                                                 bool   `json:"gate_verified"`
	PreflightVerified                                                                                                                            bool   `json:"preflight_verified"`
	BoundaryVerified                                                                                                                             bool   `json:"boundary_verified"`
	ObservationVerified                                                                                                                          bool   `json:"observation_verified"`
	ReceiverVerified                                                                                                                             bool   `json:"receiver_verified"`
	IntentVerified                                                                                                                               bool   `json:"intent_verified"`
	FinalGateVerified                                                                                                                            bool   `json:"final_gate_verified"`
	SealVerified                                                                                                                                 bool   `json:"seal_verified"`
	PermitVerified                                                                                                                               bool   `json:"permit_verified"`
	AuthorityVerified                                                                                                                            bool   `json:"authority_verified"`
	ReaderHashVerified                                                                                                                           bool   `json:"reader_hash_verified"`
	ReaderReplayVerified                                                                                                                         bool   `json:"reader_replay_verified"`
	ReaderReadBackVerified                                                                                                                       bool   `json:"reader_read_back_verified"`
	StoreHashVerified                                                                                                                            bool   `json:"store_hash_verified"`
	StoreReadBackVerified                                                                                                                        bool   `json:"store_read_back_verified"`
	AdmissionRequired                                                                                                                            bool   `json:"admission_required"`
	ShadowOnly                                                                                                                                   bool   `json:"shadow_only"`
	GraftAllowed                                                                                                                                 bool   `json:"graft_allowed"`
	DryRunOnly                                                                                                                                   bool   `json:"dry_run_only"`
	LiveReady                                                                                                                                    bool   `json:"live_ready"`
	RawDreamTextAllowed                                                                                                                          bool   `json:"raw_dream_text_allowed"`
	RawDreamTextObserved                                                                                                                         bool   `json:"raw_dream_text_observed"`
	RawDreamTextForwarded                                                                                                                        bool   `json:"raw_dream_text_forwarded"`
	JanusSurfaceAllowed                                                                                                                          bool   `json:"janus_surface_allowed"`
	CoocLearningAllowed                                                                                                                          bool   `json:"cooc_learning_allowed"`
	DeltaHarvestAllowed                                                                                                                          bool   `json:"delta_harvest_allowed"`
	BodyMutationAllowed                                                                                                                          bool   `json:"body_mutation_allowed"`
	RollbackRequired                                                                                                                             bool   `json:"rollback_required"`
	ReadOnly                                                                                                                                     bool   `json:"read_only"`
	ReplayOnly                                                                                                                                   bool   `json:"replay_only"`
	AuthorityGranted                                                                                                                             bool   `json:"authority_granted"`
	ContractsReady                                                                                                                               bool   `json:"contracts_ready"`
	WriteAllowed                                                                                                                                 bool   `json:"write_allowed"`
	AdmissionAllowed                                                                                                                             bool   `json:"admission_allowed"`
	LiveAdmissionEnabled                                                                                                                         bool   `json:"live_admission_enabled"`
	MutatesState                                                                                                                                 bool   `json:"mutates_state"`
	BodyTarget                                                                                                                                   string `json:"body_target"`
	Passed                                                                                                                                       bool   `json:"passed"`
	Reason                                                                                                                                       string `json:"reason"`

	SourceSchema                                                                                                                                 string `json:"source_schema"`
	SourceStatus                                                                                                                                 string `json:"source_status"`
	SourceTarget                                                                                                                                 string `json:"source_target"`
	SourceReport                                                                                                                                 string `json:"source_report"`
	SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionID       string `json:"source_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_id"`
	SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionReady    bool   `json:"source_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_ready"`
	SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionCausalID string `json:"source_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_causal_id"`
	SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionHash     string `json:"source_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_hash"`
	SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionReadBack string `json:"source_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_read_back_hash"`
	SourceDecision                                                                                                                               string `json:"source_decision"`
	SourceDecisionAction                                                                                                                         string `json:"source_decision_action"`
	SourceDecisionReceiptShape                                                                                                                   string `json:"source_decision_receipt_shape"`
	SourceDecisionKind                                                                                                                           string `json:"source_decision_kind"`
	SourceDecisionMode                                                                                                                           string `json:"source_decision_mode"`
	SourceDecisionStage                                                                                                                          string `json:"source_decision_stage"`
	SourceDecisionLedgerReady                                                                                                                    bool   `json:"source_decision_ledger_ready"`
	SourceDecisionLedgerAppendAllowed                                                                                                            bool   `json:"source_decision_ledger_append_allowed"`
	SourceDecisionAdmissionRequired                                                                                                              bool   `json:"source_decision_admission_required"`
	SourceDecisionShadowOnly                                                                                                                     bool   `json:"source_decision_shadow_only"`
	SourceDecisionGraftAllowed                                                                                                                   bool   `json:"source_decision_graft_allowed"`
	SourceDecisionDryRunOnly                                                                                                                     bool   `json:"source_decision_dry_run_only"`
	SourceDecisionLiveReady                                                                                                                      bool   `json:"source_decision_live_ready"`
	SourceDecisionRawDreamTextAllowed                                                                                                            bool   `json:"source_decision_raw_dream_text_allowed"`
	SourceDecisionRawDreamTextObserved                                                                                                           bool   `json:"source_decision_raw_dream_text_observed"`
	SourceDecisionRawDreamTextForwarded                                                                                                          bool   `json:"source_decision_raw_dream_text_forwarded"`
	SourceDecisionJanusSurfaceAllowed                                                                                                            bool   `json:"source_decision_janus_surface_allowed"`
	SourceDecisionCoocLearningAllowed                                                                                                            bool   `json:"source_decision_cooc_learning_allowed"`
	SourceDecisionDeltaHarvestAllowed                                                                                                            bool   `json:"source_decision_delta_harvest_allowed"`
	SourceDecisionBodyMutationAllowed                                                                                                            bool   `json:"source_decision_body_mutation_allowed"`
	SourceDecisionRollbackRequired                                                                                                               bool   `json:"source_decision_rollback_required"`
	SourceDecisionReadOnly                                                                                                                       bool   `json:"source_decision_read_only"`
	SourceDecisionReplayOnly                                                                                                                     bool   `json:"source_decision_replay_only"`
	SourceDecisionAuthorityGranted                                                                                                               bool   `json:"source_decision_authority_granted"`
	SourceDecisionContractsReady                                                                                                                 bool   `json:"source_decision_contracts_ready"`
	SourceDecisionWriteAllowed                                                                                                                   bool   `json:"source_decision_write_allowed"`
	SourceDecisionAdmissionAllowed                                                                                                               bool   `json:"source_decision_admission_allowed"`
	SourceDecisionLiveAdmissionEnabled                                                                                                           bool   `json:"source_decision_live_admission_enabled"`
	SourceDecisionMutatesState                                                                                                                   bool   `json:"source_decision_mutates_state"`
	SourceDecisionBodyTarget                                                                                                                     string `json:"source_decision_body_target"`
	SourceDecisionPassed                                                                                                                         bool   `json:"source_decision_passed"`
	SourceDecisionReason                                                                                                                         string `json:"source_decision_reason"`

	SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionID    string `json:"source_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_id"`
	SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionReady bool   `json:"source_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_ready"`
	SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofID                string `json:"source_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_id"`
	SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofReady             bool   `json:"source_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_ready"`
	SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderID                     string `json:"source_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_id"`
	SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderReady                  bool   `json:"source_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_ready"`
	SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreID                           string `json:"source_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_id"`
	SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReady                        bool   `json:"source_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_ready"`
	SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateID                                string `json:"source_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_id"`
	SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateReady                             bool   `json:"source_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_ready"`
	SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateID                                         string `json:"source_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_id"`
	SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateReady                                      bool   `json:"source_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_ready"`
	SourceAdmissionFinalGateObservationBoundaryPreflightGateReady                                                                     bool   `json:"source_admission_final_gate_observation_boundary_preflight_gate_ready"`
	SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightID                                             string `json:"source_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_id"`
	SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightReady                                          bool   `json:"source_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_ready"`
	SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryID                                                      string `json:"source_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_id"`
	SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryReady                                                   bool   `json:"source_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_ready"`
	SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationID                                                              string `json:"source_weighted_admission_resonance_graft_admission_final_gate_observation_id"`
	SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationReady                                                           bool   `json:"source_weighted_admission_resonance_graft_admission_final_gate_observation_ready"`
	SourceWeightedAdmissionResonanceGraftAdmissionFinalGateReceiverID                                                                 string `json:"source_weighted_admission_resonance_graft_admission_final_gate_receiver_id"`
	SourceWeightedAdmissionResonanceGraftAdmissionFinalGateReceiverReady                                                              bool   `json:"source_weighted_admission_resonance_graft_admission_final_gate_receiver_ready"`
	SourceWeightedAdmissionResonanceGraftAdmissionFinalGateIntentReady                                                                bool   `json:"source_weighted_admission_resonance_graft_admission_final_gate_intent_ready"`
	SourceWeightedAdmissionResonanceGraftAdmissionFinalGateReady                                                                      bool   `json:"source_weighted_admission_resonance_graft_admission_final_gate_ready"`
	SourceWeightedAdmissionResonanceGraftAdmissionSealReady                                                                           bool   `json:"source_weighted_admission_resonance_graft_admission_seal_ready"`
	SourceWeightedAdmissionResonanceGraftAdmissionAuthorityReady                                                                      bool   `json:"source_weighted_admission_resonance_graft_admission_authority_ready"`
	SourceWeightedAdmissionResonanceGraftAdmissionPermitReady                                                                         bool   `json:"source_weighted_admission_resonance_graft_admission_permit_ready"`
	SourceWeightedAdmissionResonanceGraftAdmissionReadinessReady                                                                      bool   `json:"source_weighted_admission_resonance_graft_admission_readiness_ready"`
	SourceWriterInventoryVerified                                                                                                     bool   `json:"source_writer_inventory_verified"`
	SourceWriterPreflightVerified                                                                                                     bool   `json:"source_writer_preflight_verified"`
	SourceAdmissionRequired                                                                                                           bool   `json:"source_admission_required"`
	SourceShadowOnly                                                                                                                  bool   `json:"source_shadow_only"`
	SourceDryRunOnly                                                                                                                  bool   `json:"source_dry_run_only"`
	SourceRequiresWriter                                                                                                              bool   `json:"source_requires_writer"`
	SourceRollbackRequired                                                                                                            bool   `json:"source_rollback_required"`
	SourceRequiresRollback                                                                                                            bool   `json:"source_requires_rollback"`
	SourceReadOnly                                                                                                                    bool   `json:"source_read_only"`
	SourceReplayOnly                                                                                                                  bool   `json:"source_replay_only"`
}

func runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotion(args []string) error {
	if len(args) != 2 {
		return fmt.Errorf("usage: --admission-live-route-weighted-admission-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-candidate-store-reader-proof-precondition-decision-promotion RESONANCE_GRAFT_ADMISSION_FINAL_GATE_OBSERVATION_BOUNDARY_PREFLIGHT_GATE_CANDIDATE_STORE_READER_PROOF_PRECONDITION_DECISION_REPORT RESONANCE_GRAFT_ADMISSION_FINAL_GATE_OBSERVATION_BOUNDARY_PREFLIGHT_GATE_CANDIDATE_STORE_READER_PROOF_PRECONDITION_DECISION_PROMOTION_REPORT")
	}
	decisionPath := args[0]
	outputPath := args[1]
	if strings.TrimSpace(outputPath) == "" {
		return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion output path missing")
	}
	decision, root, err := readAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionReportForAssert(decisionPath)
	if err != nil {
		return err
	}
	if err := admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionReportError(decision, root); err != nil {
		return err
	}
	promotion := admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionReport{
		Schema:         admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSchema,
		Status:         "shadow_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_ready_dry_run",
		Target:         "live_route_admission_next_step",
		TargetKind:     "weighted_internal_world_shadow_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion",
		TargetMode:     "closed_promotion_receipt_dry_run",
		Action:         "promote_weighted_resonance_shadow_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_dry_run",
		Promotion:      "pending_live_admission",
		LedgerState:    "blocked",
		LedgerAction:   "reject_blocked_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_ledger_append",
		LedgerContract: "none", LedgerEntrypoint: "none", LedgerReceiptShape: "none", LedgerWriteScope: "none",
		LedgerReady: false, LedgerAppendAllowed: false,
		WeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionReady: true,
		WeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionConsumed:       true,
		WeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionRequired:       true,
		NextStepBlockedWithoutResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotion: true,
		ReceiptShape:                 "weighted_resonance_shadow_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_receipt",
		PromotionKind:                "shadow_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion",
		PromotionMode:                "closed_decision_promotion",
		PromotionStage:               "post_preflight_gate_candidate_store_reader_proof_precondition_decision_pre_live_admission_promotion",
		DecisionVerified:             true,
		DecisionHashVerified:         true,
		DecisionReadBackVerified:     true,
		ProofPreconditionVerified:    decision.ProofPreconditionVerified,
		PreconditionHashVerified:     decision.PreconditionHashVerified,
		PreconditionReadBackVerified: decision.PreconditionReadBackVerified,
		ProofVerified:                decision.ProofVerified,
		ProofHashVerified:            decision.ProofHashVerified,
		ProofReadBackVerified:        decision.ProofReadBackVerified,
		StoreReaderVerified:          decision.StoreReaderVerified,
		StoreVerified:                decision.StoreVerified,
		CandidateVerified:            decision.CandidateVerified,
		GateVerified:                 decision.GateVerified,
		PreflightVerified:            decision.PreflightVerified,
		BoundaryVerified:             decision.BoundaryVerified,
		ObservationVerified:          decision.ObservationVerified,
		ReceiverVerified:             decision.ReceiverVerified,
		IntentVerified:               decision.IntentVerified,
		FinalGateVerified:            decision.FinalGateVerified,
		SealVerified:                 decision.SealVerified,
		PermitVerified:               decision.PermitVerified,
		AuthorityVerified:            decision.AuthorityVerified,
		ReaderHashVerified:           decision.ReaderHashVerified,
		ReaderReplayVerified:         decision.ReaderReplayVerified,
		ReaderReadBackVerified:       decision.ReaderReadBackVerified,
		StoreHashVerified:            decision.StoreHashVerified,
		StoreReadBackVerified:        decision.StoreReadBackVerified,
		AdmissionRequired:            true,
		ShadowOnly:                   true,
		GraftAllowed:                 false,
		DryRunOnly:                   true,
		LiveReady:                    true,
		RawDreamTextAllowed:          false,
		RawDreamTextObserved:         false,
		RawDreamTextForwarded:        false,
		JanusSurfaceAllowed:          false,
		CoocLearningAllowed:          false,
		DeltaHarvestAllowed:          false,
		BodyMutationAllowed:          false,
		RollbackRequired:             true,
		ReadOnly:                     true,
		ReplayOnly:                   true,
		AuthorityGranted:             false,
		ContractsReady:               false,
		WriteAllowed:                 false,
		AdmissionAllowed:             false,
		LiveAdmissionEnabled:         false,
		MutatesState:                 false,
		BodyTarget:                   "none",
		Passed:                       true,
		Reason:                       "weighted resonance shadow graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promoted as pending live admission while closed",

		SourceSchema: decision.Schema, SourceStatus: decision.Status, SourceTarget: decision.Target, SourceReport: decisionPath,
		SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionID:       decision.WeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionID,
		SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionReady:    decision.WeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionReady,
		SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionCausalID: decision.CausalID,
		SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionHash:     decision.DecisionHash,
		SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionReadBack: decision.ReadBackHash,
		SourceDecision: decision.Decision, SourceDecisionAction: decision.Action, SourceDecisionReceiptShape: decision.ReceiptShape, SourceDecisionKind: decision.DecisionKind, SourceDecisionMode: decision.DecisionMode, SourceDecisionStage: decision.DecisionStage,
		SourceDecisionLedgerReady: decision.LedgerReady, SourceDecisionLedgerAppendAllowed: decision.LedgerAppendAllowed,
		SourceDecisionAdmissionRequired: decision.AdmissionRequired, SourceDecisionShadowOnly: decision.ShadowOnly, SourceDecisionGraftAllowed: decision.GraftAllowed, SourceDecisionDryRunOnly: decision.DryRunOnly, SourceDecisionLiveReady: decision.LiveReady,
		SourceDecisionRawDreamTextAllowed: decision.RawDreamTextAllowed, SourceDecisionRawDreamTextObserved: decision.RawDreamTextObserved, SourceDecisionRawDreamTextForwarded: decision.RawDreamTextForwarded, SourceDecisionJanusSurfaceAllowed: decision.JanusSurfaceAllowed, SourceDecisionCoocLearningAllowed: decision.CoocLearningAllowed, SourceDecisionDeltaHarvestAllowed: decision.DeltaHarvestAllowed, SourceDecisionBodyMutationAllowed: decision.BodyMutationAllowed,
		SourceDecisionRollbackRequired: decision.RollbackRequired, SourceDecisionReadOnly: decision.ReadOnly, SourceDecisionReplayOnly: decision.ReplayOnly, SourceDecisionAuthorityGranted: decision.AuthorityGranted, SourceDecisionContractsReady: decision.ContractsReady, SourceDecisionWriteAllowed: decision.WriteAllowed, SourceDecisionAdmissionAllowed: decision.AdmissionAllowed, SourceDecisionLiveAdmissionEnabled: decision.LiveAdmissionEnabled, SourceDecisionMutatesState: decision.MutatesState, SourceDecisionBodyTarget: decision.BodyTarget, SourceDecisionPassed: decision.Passed, SourceDecisionReason: decision.Reason,

		SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionID:    decision.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionID,
		SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionReady: decision.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionReady,
		SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofID:                decision.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofID,
		SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofReady:             decision.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofReady,
		SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderID:                     decision.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderID,
		SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderReady:                  decision.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderReady,
		SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreID:                           decision.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreID,
		SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReady:                        decision.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReady,
		SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateID:                                decision.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateID,
		SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateReady:                             decision.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateReady,
		SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateID:                                         decision.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateID,
		SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateReady:                                      decision.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateReady,
		SourceAdmissionFinalGateObservationBoundaryPreflightGateReady:                                                                     decision.SourceAdmissionFinalGateObservationBoundaryPreflightGateReady,
		SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightID:                                             decision.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightID,
		SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightReady:                                          decision.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightReady,
		SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryID:                                                      decision.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryID,
		SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryReady:                                                   decision.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryReady,
		SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationID:                                                              decision.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationID,
		SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationReady:                                                           decision.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationReady,
		SourceWeightedAdmissionResonanceGraftAdmissionFinalGateReceiverID:                                                                 decision.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateReceiverID,
		SourceWeightedAdmissionResonanceGraftAdmissionFinalGateReceiverReady:                                                              decision.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateReceiverReady,
		SourceWeightedAdmissionResonanceGraftAdmissionFinalGateIntentReady:                                                                decision.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateIntentReady,
		SourceWeightedAdmissionResonanceGraftAdmissionFinalGateReady:                                                                      decision.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateReady,
		SourceWeightedAdmissionResonanceGraftAdmissionSealReady:                                                                           decision.SourceWeightedAdmissionResonanceGraftAdmissionSealReady,
		SourceWeightedAdmissionResonanceGraftAdmissionAuthorityReady:                                                                      decision.SourceWeightedAdmissionResonanceGraftAdmissionAuthorityReady,
		SourceWeightedAdmissionResonanceGraftAdmissionPermitReady:                                                                         decision.SourceWeightedAdmissionResonanceGraftAdmissionPermitReady,
		SourceWeightedAdmissionResonanceGraftAdmissionReadinessReady:                                                                      decision.SourceWeightedAdmissionResonanceGraftAdmissionReadinessReady,
		SourceWriterInventoryVerified: decision.SourceWriterInventoryVerified, SourceWriterPreflightVerified: decision.SourceWriterPreflightVerified, SourceAdmissionRequired: decision.SourceAdmissionRequired, SourceShadowOnly: decision.SourceShadowOnly, SourceDryRunOnly: decision.SourceDryRunOnly, SourceRequiresWriter: decision.SourceRequiresWriter, SourceRollbackRequired: decision.SourceRollbackRequired, SourceRequiresRollback: decision.SourceRequiresRollback, SourceReadOnly: decision.SourceReadOnly, SourceReplayOnly: decision.SourceReplayOnly,
	}
	promotion.CausalID = admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionCausalID(promotion)
	promotion.PromotionHash = admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionHash(promotion)
	promotion.ReadBackHash = admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionReadBackHash(promotion)
	promotion.WeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionID = admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionID(promotion)
	if promotion.CausalID == "" || promotion.PromotionHash == "" || promotion.ReadBackHash == "" || promotion.WeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionID == "" || promotion.PromotionHash == promotion.ReadBackHash {
		return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion read-back proof failed")
	}
	raw, err := json.MarshalIndent(promotion, "", "  ")
	if err != nil {
		return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion marshal failed: %w", err)
	}
	raw = append(raw, '\n')
	if err := os.WriteFile(outputPath, raw, 0600); err != nil {
		return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion write failed: %w", err)
	}
	fmt.Printf("[admission-live-route-weighted-admission-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-candidate-store-reader-proof-precondition-decision-promotion] pass: resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_report=%s resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_report=%s\n", outputPath, decisionPath)
	return nil
}

func runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionAssert(args []string) error {
	if len(args) != 1 {
		return fmt.Errorf("usage: --admission-live-route-weighted-admission-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-candidate-store-reader-proof-precondition-decision-promotion-assert REPORT")
	}
	report, root, err := readAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionReportForAssert(args[0])
	if err != nil {
		return err
	}
	return admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionReportError(report, root)
}

func admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionReportError(report admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionReport, root map[string]json.RawMessage) error {
	if _, ok := root["schema"]; !ok {
		return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion schema missing")
	}
	if report.Schema != admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSchema {
		return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion schema mismatch: got %q want %q", report.Schema, admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSchema)
	}
	if report.Status != "shadow_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_ready_dry_run" {
		return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion status mismatch: got %q want %q", report.Status, "shadow_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_ready_dry_run")
	}
	if report.Target != "live_route_admission_next_step" || report.TargetKind != "weighted_internal_world_shadow_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion" || report.TargetMode != "closed_promotion_receipt_dry_run" || report.Action != "promote_weighted_resonance_shadow_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_dry_run" {
		return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion route shape mismatch")
	}
	if report.Promotion != "pending_live_admission" {
		return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion promotion mismatch: got %q want %q", report.Promotion, "pending_live_admission")
	}
	if report.LedgerState != "blocked" || report.LedgerAction != "reject_blocked_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_ledger_append" || report.LedgerContract != "none" || report.LedgerEntrypoint != "none" || report.LedgerReceiptShape != "none" || report.LedgerWriteScope != "none" {
		return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion ledger guard mismatch")
	}
	if report.ReceiptShape != "weighted_resonance_shadow_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_receipt" || report.PromotionKind != "shadow_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion" || report.PromotionMode != "closed_decision_promotion" || report.PromotionStage != "post_preflight_gate_candidate_store_reader_proof_precondition_decision_pre_live_admission_promotion" {
		return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion shape mismatch")
	}
	for _, required := range []struct {
		name  string
		value bool
	}{
		{"weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_ready", report.WeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionReady}, {"weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_consumed", report.WeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionConsumed}, {"weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_required", report.WeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionRequired}, {"next_step_blocked_without_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion", report.NextStepBlockedWithoutResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotion},
		{"decision_verified", report.DecisionVerified}, {"decision_hash_verified", report.DecisionHashVerified}, {"decision_read_back_verified", report.DecisionReadBackVerified}, {"proof_precondition_verified", report.ProofPreconditionVerified}, {"precondition_hash_verified", report.PreconditionHashVerified}, {"precondition_read_back_verified", report.PreconditionReadBackVerified}, {"proof_verified", report.ProofVerified}, {"proof_hash_verified", report.ProofHashVerified}, {"proof_read_back_verified", report.ProofReadBackVerified}, {"store_reader_verified", report.StoreReaderVerified}, {"store_verified", report.StoreVerified}, {"candidate_verified", report.CandidateVerified}, {"gate_verified", report.GateVerified}, {"preflight_verified", report.PreflightVerified}, {"boundary_verified", report.BoundaryVerified}, {"observation_verified", report.ObservationVerified}, {"receiver_verified", report.ReceiverVerified}, {"intent_verified", report.IntentVerified}, {"final_gate_verified", report.FinalGateVerified}, {"seal_verified", report.SealVerified}, {"permit_verified", report.PermitVerified}, {"authority_verified", report.AuthorityVerified}, {"reader_hash_verified", report.ReaderHashVerified}, {"reader_replay_verified", report.ReaderReplayVerified}, {"reader_read_back_verified", report.ReaderReadBackVerified}, {"store_hash_verified", report.StoreHashVerified}, {"store_read_back_verified", report.StoreReadBackVerified}, {"admission_required", report.AdmissionRequired}, {"shadow_only", report.ShadowOnly}, {"dry_run_only", report.DryRunOnly}, {"live_ready", report.LiveReady}, {"rollback_required", report.RollbackRequired}, {"read_only", report.ReadOnly}, {"replay_only", report.ReplayOnly},
		{"source_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_ready", report.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionReady}, {"source_decision_admission_required", report.SourceDecisionAdmissionRequired}, {"source_decision_shadow_only", report.SourceDecisionShadowOnly}, {"source_decision_dry_run_only", report.SourceDecisionDryRunOnly}, {"source_decision_live_ready", report.SourceDecisionLiveReady}, {"source_decision_rollback_required", report.SourceDecisionRollbackRequired}, {"source_decision_read_only", report.SourceDecisionReadOnly}, {"source_decision_replay_only", report.SourceDecisionReplayOnly}, {"source_decision_passed", report.SourceDecisionPassed},
		{"source_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_ready", report.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionReady}, {"source_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_ready", report.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofReady}, {"source_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_ready", report.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderReady}, {"source_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_ready", report.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReady}, {"source_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_ready", report.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateReady}, {"source_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_ready", report.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateReady}, {"source_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_ready", report.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightReady}, {"source_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_ready", report.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryReady}, {"source_weighted_admission_resonance_graft_admission_final_gate_observation_ready", report.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationReady}, {"source_weighted_admission_resonance_graft_admission_final_gate_receiver_ready", report.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateReceiverReady}, {"source_weighted_admission_resonance_graft_admission_final_gate_intent_ready", report.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateIntentReady}, {"source_weighted_admission_resonance_graft_admission_final_gate_ready", report.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateReady}, {"source_weighted_admission_resonance_graft_admission_seal_ready", report.SourceWeightedAdmissionResonanceGraftAdmissionSealReady}, {"source_weighted_admission_resonance_graft_admission_authority_ready", report.SourceWeightedAdmissionResonanceGraftAdmissionAuthorityReady}, {"source_weighted_admission_resonance_graft_admission_permit_ready", report.SourceWeightedAdmissionResonanceGraftAdmissionPermitReady}, {"source_weighted_admission_resonance_graft_admission_readiness_ready", report.SourceWeightedAdmissionResonanceGraftAdmissionReadinessReady}, {"source_writer_inventory_verified", report.SourceWriterInventoryVerified}, {"source_writer_preflight_verified", report.SourceWriterPreflightVerified}, {"source_admission_required", report.SourceAdmissionRequired}, {"source_shadow_only", report.SourceShadowOnly}, {"source_dry_run_only", report.SourceDryRunOnly}, {"source_requires_writer", report.SourceRequiresWriter}, {"source_rollback_required", report.SourceRollbackRequired}, {"source_requires_rollback", report.SourceRequiresRollback}, {"source_read_only", report.SourceReadOnly}, {"source_replay_only", report.SourceReplayOnly}, {"passed", report.Passed},
	} {
		if !required.value {
			return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion %s not ready", required.name)
		}
	}
	for _, closed := range []struct {
		name  string
		value bool
	}{
		{"ledger_ready", report.LedgerReady}, {"ledger_append_allowed", report.LedgerAppendAllowed}, {"graft_allowed", report.GraftAllowed}, {"raw_dream_text_allowed", report.RawDreamTextAllowed}, {"raw_dream_text_observed", report.RawDreamTextObserved}, {"raw_dream_text_forwarded", report.RawDreamTextForwarded}, {"janus_surface_allowed", report.JanusSurfaceAllowed}, {"cooc_learning_allowed", report.CoocLearningAllowed}, {"delta_harvest_allowed", report.DeltaHarvestAllowed}, {"body_mutation_allowed", report.BodyMutationAllowed}, {"authority_granted", report.AuthorityGranted}, {"contracts_ready", report.ContractsReady}, {"write_allowed", report.WriteAllowed}, {"admission_allowed", report.AdmissionAllowed}, {"live_admission_enabled", report.LiveAdmissionEnabled}, {"mutates_state", report.MutatesState},
		{"source_decision_ledger_ready", report.SourceDecisionLedgerReady}, {"source_decision_ledger_append_allowed", report.SourceDecisionLedgerAppendAllowed}, {"source_decision_graft_allowed", report.SourceDecisionGraftAllowed}, {"source_decision_raw_dream_text_allowed", report.SourceDecisionRawDreamTextAllowed}, {"source_decision_raw_dream_text_observed", report.SourceDecisionRawDreamTextObserved}, {"source_decision_raw_dream_text_forwarded", report.SourceDecisionRawDreamTextForwarded}, {"source_decision_janus_surface_allowed", report.SourceDecisionJanusSurfaceAllowed}, {"source_decision_cooc_learning_allowed", report.SourceDecisionCoocLearningAllowed}, {"source_decision_delta_harvest_allowed", report.SourceDecisionDeltaHarvestAllowed}, {"source_decision_body_mutation_allowed", report.SourceDecisionBodyMutationAllowed}, {"source_decision_authority_granted", report.SourceDecisionAuthorityGranted}, {"source_decision_contracts_ready", report.SourceDecisionContractsReady}, {"source_decision_write_allowed", report.SourceDecisionWriteAllowed}, {"source_decision_admission_allowed", report.SourceDecisionAdmissionAllowed}, {"source_decision_live_admission_enabled", report.SourceDecisionLiveAdmissionEnabled}, {"source_decision_mutates_state", report.SourceDecisionMutatesState}, {"source_admission_final_gate_observation_boundary_preflight_gate_ready", report.SourceAdmissionFinalGateObservationBoundaryPreflightGateReady},
	} {
		if closed.value {
			return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion opened %s", closed.name)
		}
	}
	for _, requiredString := range []struct{ name, value string }{
		{"weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_id", report.WeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionID}, {"causal_id", report.CausalID}, {"promotion_hash", report.PromotionHash}, {"read_back_hash", report.ReadBackHash}, {"source_report", report.SourceReport}, {"source_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_id", report.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionID}, {"source_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_hash", report.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionHash}, {"source_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_read_back_hash", report.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionReadBack}, {"source_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_id", report.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionID}, {"source_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_id", report.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofID}, {"source_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_id", report.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderID}, {"source_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_id", report.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreID}, {"source_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_id", report.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateID}, {"source_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_id", report.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateID},
	} {
		if strings.TrimSpace(requiredString.value) == "" {
			return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion %s missing", requiredString.name)
		}
	}
	if report.SourceSchema != admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionSchema {
		return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion source_schema mismatch: got %q want %q", report.SourceSchema, admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionSchema)
	}
	if report.SourceStatus != "shadow_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_ready_dry_run" || report.SourceTarget != "live_route_admission_next_step" || report.SourceDecision != "shadow_ready" {
		return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion source decision route mismatch")
	}
	if report.SourceDecisionReceiptShape != "weighted_resonance_shadow_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_receipt" || report.SourceDecisionKind != "shadow_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision" || report.SourceDecisionMode != "closed_proof_precondition_decision" || report.SourceDecisionStage != "post_preflight_gate_candidate_store_reader_proof_precondition_pre_live_admission_decision" {
		return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion source decision shape mismatch")
	}
	if report.BodyTarget != "none" || report.SourceDecisionBodyTarget != "none" {
		return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion body target mismatch")
	}
	if !strings.HasPrefix(report.WeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionID, "weighted-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-candidate-store-reader-proof-precondition-decision-promotion-id-") || !strings.HasPrefix(report.CausalID, "weighted-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-candidate-store-reader-proof-precondition-decision-promotion-causal-") || !strings.HasPrefix(report.PromotionHash, "weighted-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-candidate-store-reader-proof-precondition-decision-promotion-") || !strings.HasPrefix(report.ReadBackHash, "weighted-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-candidate-store-reader-proof-precondition-decision-promotion-read-") || report.PromotionHash == report.ReadBackHash {
		return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion prefix mismatch")
	}
	if !strings.HasPrefix(report.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionID, "weighted-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-candidate-store-reader-proof-precondition-decision-id-") || !strings.HasPrefix(report.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionID, "weighted-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-candidate-store-reader-proof-precondition-id-") || !strings.HasPrefix(report.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofID, "weighted-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-candidate-store-reader-proof-id-") || !strings.HasPrefix(report.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderID, "weighted-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-candidate-store-reader-id-") || !strings.HasPrefix(report.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreID, "weighted-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-candidate-store-id-") || !strings.HasPrefix(report.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateID, "weighted-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-candidate-id-") || !strings.HasPrefix(report.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateID, "weighted-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-id-") {
		return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion source chain prefix mismatch")
	}
	if report.CausalID == "" || report.CausalID != admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionCausalID(report) {
		return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion causal_id mismatch")
	}
	if report.PromotionHash == "" || report.PromotionHash != admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionHash(report) {
		return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion promotion_hash mismatch")
	}
	if report.ReadBackHash == "" || report.ReadBackHash != admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionReadBackHash(report) {
		return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion read_back_hash mismatch")
	}
	if report.WeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionID == "" || report.WeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionID != admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionID(report) {
		return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion id mismatch")
	}
	if report.Reason != "weighted resonance shadow graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promoted as pending live admission while closed" {
		return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion reason mismatch: got %q", report.Reason)
	}
	return nil
}

func admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionCausalID(promotion admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionReport) string {
	h := hashJSON(map[string]interface{}{"source_decision_id": promotion.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionID, "source_decision_read_back_hash": promotion.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionReadBack, "source_precondition_id": promotion.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionID, "source_proof_id": promotion.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofID, "source_reader_id": promotion.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderID, "target": promotion.Target, "promotion_kind": promotion.PromotionKind, "promotion_stage": promotion.PromotionStage})
	if h == "" {
		return ""
	}
	return "weighted-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-candidate-store-reader-proof-precondition-decision-promotion-causal-" + h
}

func admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionHash(promotion admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionReport) string {
	h := hashJSON(map[string]interface{}{"causal_id": promotion.CausalID, "source_decision_id": promotion.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionID, "source_decision_hash": promotion.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionHash, "source_decision_read_back_hash": promotion.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionReadBack, "promotion": promotion.Promotion, "promotion_mode": promotion.PromotionMode, "receipt_shape": promotion.ReceiptShape, "decision_verified": promotion.DecisionVerified, "decision_hash_verified": promotion.DecisionHashVerified, "decision_read_back_verified": promotion.DecisionReadBackVerified, "read_only": promotion.ReadOnly, "replay_only": promotion.ReplayOnly, "admission_required": promotion.AdmissionRequired, "shadow_only": promotion.ShadowOnly, "dry_run_only": promotion.DryRunOnly, "graft_allowed": promotion.GraftAllowed, "ledger_append_allowed": promotion.LedgerAppendAllowed})
	if h == "" {
		return ""
	}
	return "weighted-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-candidate-store-reader-proof-precondition-decision-promotion-" + h
}

func admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionReadBackHash(promotion admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionReport) string {
	h := hashJSON(map[string]interface{}{"promotion_hash": promotion.PromotionHash, "source_decision_id": promotion.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionID, "source_precondition_id": promotion.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionID, "promotion_kind": promotion.PromotionKind, "promotion_ready": promotion.WeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionReady, "body_mutation": promotion.BodyMutationAllowed, "live_admission": promotion.LiveAdmissionEnabled, "write_allowed": promotion.WriteAllowed, "admission_allowed": promotion.AdmissionAllowed, "ledger_append_allowed": promotion.LedgerAppendAllowed})
	if h == "" {
		return ""
	}
	return "weighted-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-candidate-store-reader-proof-precondition-decision-promotion-read-" + h
}

func admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionID(promotion admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionReport) string {
	h := hashJSON(map[string]interface{}{"schema": promotion.Schema, "status": promotion.Status, "action": promotion.Action, "promotion": promotion.Promotion, "source_report": promotion.SourceReport, "source_decision_id": promotion.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionID, "source_precondition_id": promotion.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionID, "source_proof_id": promotion.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofID, "source_reader_id": promotion.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderID, "causal_id": promotion.CausalID, "promotion_hash": promotion.PromotionHash, "read_back_hash": promotion.ReadBackHash, "receipt_shape": promotion.ReceiptShape, "promotion_kind": promotion.PromotionKind, "promotion_mode": promotion.PromotionMode, "promotion_stage": promotion.PromotionStage, "body_target": promotion.BodyTarget, "ready": promotion.WeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionReady, "decision_verified": promotion.DecisionVerified, "decision_hash_verified": promotion.DecisionHashVerified, "decision_read_back_verified": promotion.DecisionReadBackVerified, "admission_required": promotion.AdmissionRequired, "shadow_only": promotion.ShadowOnly, "graft_allowed": promotion.GraftAllowed, "dry_run_only": promotion.DryRunOnly, "read_only": promotion.ReadOnly, "replay_only": promotion.ReplayOnly, "live_ready": promotion.LiveReady, "contracts_ready": promotion.ContractsReady, "write_allowed": promotion.WriteAllowed, "admission_allowed": promotion.AdmissionAllowed, "live_admission_enabled": promotion.LiveAdmissionEnabled, "mutates_state": promotion.MutatesState, "ledger_append_allowed": promotion.LedgerAppendAllowed, "next_step_blocked_without": promotion.NextStepBlockedWithoutResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotion})
	if h == "" {
		return ""
	}
	return "weighted-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-candidate-store-reader-proof-precondition-decision-promotion-id-" + h
}

func readAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionReportForAssert(path string) (admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionReport, map[string]json.RawMessage, error) {
	var report admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionReport
	if strings.TrimSpace(path) == "" {
		return report, nil, fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion path missing")
	}
	raw, err := os.ReadFile(path)
	if err != nil {
		if os.IsNotExist(err) {
			return report, nil, fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion not written")
		}
		return report, nil, err
	}
	if len(raw) == 0 {
		return report, nil, fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion not written")
	}
	var root map[string]json.RawMessage
	if err := json.Unmarshal(raw, &root); err != nil {
		return report, nil, fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion JSON invalid: %w", err)
	}
	if err := json.Unmarshal(raw, &report); err != nil {
		return report, nil, fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion decode failed: %w", err)
	}
	return report, root, nil
}
