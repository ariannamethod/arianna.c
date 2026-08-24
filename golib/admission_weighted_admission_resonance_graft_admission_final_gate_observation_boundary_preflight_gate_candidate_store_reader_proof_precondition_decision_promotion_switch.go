package main

import (
	"encoding/json"
	"fmt"
	"os"
	"strings"
)

const admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchSchema = "arianna.live_route_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch.v1"

type admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchReport struct {
	Schema                                                                                                                                             string `json:"schema"`
	Status                                                                                                                                             string `json:"status"`
	Target                                                                                                                                             string `json:"target"`
	TargetKind                                                                                                                                         string `json:"target_kind"`
	TargetMode                                                                                                                                         string `json:"target_mode"`
	Action                                                                                                                                             string `json:"action"`
	SwitchState                                                                                                                                        string `json:"switch_state"`
	SwitchAction                                                                                                                                       string `json:"switch_action"`
	Promotion                                                                                                                                          string `json:"promotion"`
	LedgerState                                                                                                                                        string `json:"ledger_state"`
	LedgerAction                                                                                                                                       string `json:"ledger_action"`
	LedgerContract                                                                                                                                     string `json:"ledger_contract"`
	LedgerEntrypoint                                                                                                                                   string `json:"ledger_entrypoint"`
	LedgerReceiptShape                                                                                                                                 string `json:"ledger_receipt_shape"`
	LedgerWriteScope                                                                                                                                   string `json:"ledger_write_scope"`
	LedgerReady                                                                                                                                        bool   `json:"ledger_ready"`
	LedgerAppendAllowed                                                                                                                                bool   `json:"ledger_append_allowed"`
	WeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchReady bool   `json:"weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_ready"`
	WeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionConsumed    bool   `json:"weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_consumed"`
	WeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionRequired    bool   `json:"weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_required"`
	NextStepBlockedWithoutResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitch bool   `json:"next_step_blocked_without_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch"`
	WeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchID    string `json:"weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_id"`
	ReceiptShape                                                                                                                                       string `json:"receipt_shape"`
	SwitchKind                                                                                                                                         string `json:"switch_kind"`
	SwitchMode                                                                                                                                         string `json:"switch_mode"`
	SwitchStage                                                                                                                                        string `json:"switch_stage"`
	CausalID                                                                                                                                           string `json:"causal_id"`
	SwitchHash                                                                                                                                         string `json:"switch_hash"`
	ReadBackHash                                                                                                                                       string `json:"read_back_hash"`
	PromotionVerified                                                                                                                                  bool   `json:"promotion_verified"`
	PromotionHashVerified                                                                                                                              bool   `json:"promotion_hash_verified"`
	PromotionReadBackVerified                                                                                                                          bool   `json:"promotion_read_back_verified"`
	DecisionVerified                                                                                                                                   bool   `json:"decision_verified"`
	DecisionHashVerified                                                                                                                               bool   `json:"decision_hash_verified"`
	DecisionReadBackVerified                                                                                                                           bool   `json:"decision_read_back_verified"`
	ProofPreconditionVerified                                                                                                                          bool   `json:"proof_precondition_verified"`
	PreconditionHashVerified                                                                                                                           bool   `json:"precondition_hash_verified"`
	PreconditionReadBackVerified                                                                                                                       bool   `json:"precondition_read_back_verified"`
	ProofVerified                                                                                                                                      bool   `json:"proof_verified"`
	ProofHashVerified                                                                                                                                  bool   `json:"proof_hash_verified"`
	ProofReadBackVerified                                                                                                                              bool   `json:"proof_read_back_verified"`
	StoreReaderVerified                                                                                                                                bool   `json:"store_reader_verified"`
	StoreVerified                                                                                                                                      bool   `json:"store_verified"`
	CandidateVerified                                                                                                                                  bool   `json:"candidate_verified"`
	GateVerified                                                                                                                                       bool   `json:"gate_verified"`
	PreflightVerified                                                                                                                                  bool   `json:"preflight_verified"`
	BoundaryVerified                                                                                                                                   bool   `json:"boundary_verified"`
	ObservationVerified                                                                                                                                bool   `json:"observation_verified"`
	ReceiverVerified                                                                                                                                   bool   `json:"receiver_verified"`
	IntentVerified                                                                                                                                     bool   `json:"intent_verified"`
	FinalGateVerified                                                                                                                                  bool   `json:"final_gate_verified"`
	SealVerified                                                                                                                                       bool   `json:"seal_verified"`
	PermitVerified                                                                                                                                     bool   `json:"permit_verified"`
	AuthorityVerified                                                                                                                                  bool   `json:"authority_verified"`
	ReaderHashVerified                                                                                                                                 bool   `json:"reader_hash_verified"`
	ReaderReplayVerified                                                                                                                               bool   `json:"reader_replay_verified"`
	ReaderReadBackVerified                                                                                                                             bool   `json:"reader_read_back_verified"`
	StoreHashVerified                                                                                                                                  bool   `json:"store_hash_verified"`
	StoreReadBackVerified                                                                                                                              bool   `json:"store_read_back_verified"`
	AdmissionRequired                                                                                                                                  bool   `json:"admission_required"`
	ShadowOnly                                                                                                                                         bool   `json:"shadow_only"`
	GraftAllowed                                                                                                                                       bool   `json:"graft_allowed"`
	DryRunOnly                                                                                                                                         bool   `json:"dry_run_only"`
	LiveReady                                                                                                                                          bool   `json:"live_ready"`
	RawDreamTextAllowed                                                                                                                                bool   `json:"raw_dream_text_allowed"`
	RawDreamTextObserved                                                                                                                               bool   `json:"raw_dream_text_observed"`
	RawDreamTextForwarded                                                                                                                              bool   `json:"raw_dream_text_forwarded"`
	JanusSurfaceAllowed                                                                                                                                bool   `json:"janus_surface_allowed"`
	CoocLearningAllowed                                                                                                                                bool   `json:"cooc_learning_allowed"`
	DeltaHarvestAllowed                                                                                                                                bool   `json:"delta_harvest_allowed"`
	BodyMutationAllowed                                                                                                                                bool   `json:"body_mutation_allowed"`
	RollbackRequired                                                                                                                                   bool   `json:"rollback_required"`
	ReadOnly                                                                                                                                           bool   `json:"read_only"`
	ReplayOnly                                                                                                                                         bool   `json:"replay_only"`
	AuthorityGranted                                                                                                                                   bool   `json:"authority_granted"`
	ContractsReady                                                                                                                                     bool   `json:"contracts_ready"`
	WriteAllowed                                                                                                                                       bool   `json:"write_allowed"`
	AdmissionAllowed                                                                                                                                   bool   `json:"admission_allowed"`
	LiveAdmissionEnabled                                                                                                                               bool   `json:"live_admission_enabled"`
	MutatesState                                                                                                                                       bool   `json:"mutates_state"`
	BodyTarget                                                                                                                                         string `json:"body_target"`
	Passed                                                                                                                                             bool   `json:"passed"`
	Reason                                                                                                                                             string `json:"reason"`

	SourceSchema                                                                                                                                          string `json:"source_schema"`
	SourceStatus                                                                                                                                          string `json:"source_status"`
	SourceTarget                                                                                                                                          string `json:"source_target"`
	SourceReport                                                                                                                                          string `json:"source_report"`
	SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionID       string `json:"source_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_id"`
	SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionReady    bool   `json:"source_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_ready"`
	SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionCausalID string `json:"source_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_causal_id"`
	SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionHash     string `json:"source_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_hash"`
	SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionReadBack string `json:"source_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_read_back_hash"`
	SourcePromotion                                                                                                                                       string `json:"source_promotion"`
	SourcePromotionAction                                                                                                                                 string `json:"source_promotion_action"`
	SourcePromotionReceiptShape                                                                                                                           string `json:"source_promotion_receipt_shape"`
	SourcePromotionKind                                                                                                                                   string `json:"source_promotion_kind"`
	SourcePromotionMode                                                                                                                                   string `json:"source_promotion_mode"`
	SourcePromotionStage                                                                                                                                  string `json:"source_promotion_stage"`
	SourcePromotionLedgerReady                                                                                                                            bool   `json:"source_promotion_ledger_ready"`
	SourcePromotionLedgerAppendAllowed                                                                                                                    bool   `json:"source_promotion_ledger_append_allowed"`
	SourcePromotionAdmissionRequired                                                                                                                      bool   `json:"source_promotion_admission_required"`
	SourcePromotionShadowOnly                                                                                                                             bool   `json:"source_promotion_shadow_only"`
	SourcePromotionGraftAllowed                                                                                                                           bool   `json:"source_promotion_graft_allowed"`
	SourcePromotionDryRunOnly                                                                                                                             bool   `json:"source_promotion_dry_run_only"`
	SourcePromotionLiveReady                                                                                                                              bool   `json:"source_promotion_live_ready"`
	SourcePromotionRawDreamTextAllowed                                                                                                                    bool   `json:"source_promotion_raw_dream_text_allowed"`
	SourcePromotionRawDreamTextObserved                                                                                                                   bool   `json:"source_promotion_raw_dream_text_observed"`
	SourcePromotionRawDreamTextForwarded                                                                                                                  bool   `json:"source_promotion_raw_dream_text_forwarded"`
	SourcePromotionJanusSurfaceAllowed                                                                                                                    bool   `json:"source_promotion_janus_surface_allowed"`
	SourcePromotionCoocLearningAllowed                                                                                                                    bool   `json:"source_promotion_cooc_learning_allowed"`
	SourcePromotionDeltaHarvestAllowed                                                                                                                    bool   `json:"source_promotion_delta_harvest_allowed"`
	SourcePromotionBodyMutationAllowed                                                                                                                    bool   `json:"source_promotion_body_mutation_allowed"`
	SourcePromotionRollbackRequired                                                                                                                       bool   `json:"source_promotion_rollback_required"`
	SourcePromotionReadOnly                                                                                                                               bool   `json:"source_promotion_read_only"`
	SourcePromotionReplayOnly                                                                                                                             bool   `json:"source_promotion_replay_only"`
	SourcePromotionAuthorityGranted                                                                                                                       bool   `json:"source_promotion_authority_granted"`
	SourcePromotionContractsReady                                                                                                                         bool   `json:"source_promotion_contracts_ready"`
	SourcePromotionWriteAllowed                                                                                                                           bool   `json:"source_promotion_write_allowed"`
	SourcePromotionAdmissionAllowed                                                                                                                       bool   `json:"source_promotion_admission_allowed"`
	SourcePromotionLiveAdmissionEnabled                                                                                                                   bool   `json:"source_promotion_live_admission_enabled"`
	SourcePromotionMutatesState                                                                                                                           bool   `json:"source_promotion_mutates_state"`
	SourcePromotionBodyTarget                                                                                                                             string `json:"source_promotion_body_target"`
	SourcePromotionPassed                                                                                                                                 bool   `json:"source_promotion_passed"`
	SourcePromotionReason                                                                                                                                 string `json:"source_promotion_reason"`
	SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionID                string `json:"source_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_id"`
	SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionReady             bool   `json:"source_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_ready"`
	SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionCausalID          string `json:"source_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_causal_id"`
	SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionHash              string `json:"source_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_hash"`
	SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionReadBack          string `json:"source_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_read_back_hash"`
	SourceDecision                                                                                                                                        string `json:"source_decision"`
	SourceDecisionAction                                                                                                                                  string `json:"source_decision_action"`
	SourceDecisionReceiptShape                                                                                                                            string `json:"source_decision_receipt_shape"`
	SourceDecisionKind                                                                                                                                    string `json:"source_decision_kind"`
	SourceDecisionMode                                                                                                                                    string `json:"source_decision_mode"`
	SourceDecisionStage                                                                                                                                   string `json:"source_decision_stage"`
	SourceDecisionLedgerReady                                                                                                                             bool   `json:"source_decision_ledger_ready"`
	SourceDecisionLedgerAppendAllowed                                                                                                                     bool   `json:"source_decision_ledger_append_allowed"`
	SourceDecisionAdmissionRequired                                                                                                                       bool   `json:"source_decision_admission_required"`
	SourceDecisionShadowOnly                                                                                                                              bool   `json:"source_decision_shadow_only"`
	SourceDecisionGraftAllowed                                                                                                                            bool   `json:"source_decision_graft_allowed"`
	SourceDecisionDryRunOnly                                                                                                                              bool   `json:"source_decision_dry_run_only"`
	SourceDecisionLiveReady                                                                                                                               bool   `json:"source_decision_live_ready"`
	SourceDecisionRawDreamTextAllowed                                                                                                                     bool   `json:"source_decision_raw_dream_text_allowed"`
	SourceDecisionRawDreamTextObserved                                                                                                                    bool   `json:"source_decision_raw_dream_text_observed"`
	SourceDecisionRawDreamTextForwarded                                                                                                                   bool   `json:"source_decision_raw_dream_text_forwarded"`
	SourceDecisionJanusSurfaceAllowed                                                                                                                     bool   `json:"source_decision_janus_surface_allowed"`
	SourceDecisionCoocLearningAllowed                                                                                                                     bool   `json:"source_decision_cooc_learning_allowed"`
	SourceDecisionDeltaHarvestAllowed                                                                                                                     bool   `json:"source_decision_delta_harvest_allowed"`
	SourceDecisionBodyMutationAllowed                                                                                                                     bool   `json:"source_decision_body_mutation_allowed"`
	SourceDecisionRollbackRequired                                                                                                                        bool   `json:"source_decision_rollback_required"`
	SourceDecisionReadOnly                                                                                                                                bool   `json:"source_decision_read_only"`
	SourceDecisionReplayOnly                                                                                                                              bool   `json:"source_decision_replay_only"`
	SourceDecisionAuthorityGranted                                                                                                                        bool   `json:"source_decision_authority_granted"`
	SourceDecisionContractsReady                                                                                                                          bool   `json:"source_decision_contracts_ready"`
	SourceDecisionWriteAllowed                                                                                                                            bool   `json:"source_decision_write_allowed"`
	SourceDecisionAdmissionAllowed                                                                                                                        bool   `json:"source_decision_admission_allowed"`
	SourceDecisionLiveAdmissionEnabled                                                                                                                    bool   `json:"source_decision_live_admission_enabled"`
	SourceDecisionMutatesState                                                                                                                            bool   `json:"source_decision_mutates_state"`
	SourceDecisionBodyTarget                                                                                                                              string `json:"source_decision_body_target"`
	SourceDecisionPassed                                                                                                                                  bool   `json:"source_decision_passed"`
	SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionID                        string `json:"source_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_id"`
	SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionReady                     bool   `json:"source_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_ready"`
	SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofID                                    string `json:"source_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_id"`
	SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofReady                                 bool   `json:"source_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_ready"`
	SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderID                                         string `json:"source_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_id"`
	SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderReady                                      bool   `json:"source_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_ready"`
	SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreID                                               string `json:"source_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_id"`
	SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReady                                            bool   `json:"source_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_ready"`
	SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateID                                                    string `json:"source_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_id"`
	SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateReady                                                 bool   `json:"source_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_ready"`
	SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateID                                                             string `json:"source_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_id"`
	SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateReady                                                          bool   `json:"source_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_ready"`
	SourceAdmissionFinalGateObservationBoundaryPreflightGateReady                                                                                         bool   `json:"source_admission_final_gate_observation_boundary_preflight_gate_ready"`
	SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightID                                                                 string `json:"source_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_id"`
	SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightReady                                                              bool   `json:"source_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_ready"`
	SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryID                                                                          string `json:"source_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_id"`
	SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryReady                                                                       bool   `json:"source_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_ready"`
	SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationID                                                                                  string `json:"source_weighted_admission_resonance_graft_admission_final_gate_observation_id"`
	SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationReady                                                                               bool   `json:"source_weighted_admission_resonance_graft_admission_final_gate_observation_ready"`
	SourceWeightedAdmissionResonanceGraftAdmissionFinalGateReceiverID                                                                                     string `json:"source_weighted_admission_resonance_graft_admission_final_gate_receiver_id"`
	SourceWeightedAdmissionResonanceGraftAdmissionFinalGateReceiverReady                                                                                  bool   `json:"source_weighted_admission_resonance_graft_admission_final_gate_receiver_ready"`
	SourceWeightedAdmissionResonanceGraftAdmissionFinalGateIntentReady                                                                                    bool   `json:"source_weighted_admission_resonance_graft_admission_final_gate_intent_ready"`
	SourceWeightedAdmissionResonanceGraftAdmissionFinalGateReady                                                                                          bool   `json:"source_weighted_admission_resonance_graft_admission_final_gate_ready"`
	SourceWeightedAdmissionResonanceGraftAdmissionSealReady                                                                                               bool   `json:"source_weighted_admission_resonance_graft_admission_seal_ready"`
	SourceWeightedAdmissionResonanceGraftAdmissionAuthorityReady                                                                                          bool   `json:"source_weighted_admission_resonance_graft_admission_authority_ready"`
	SourceWeightedAdmissionResonanceGraftAdmissionPermitReady                                                                                             bool   `json:"source_weighted_admission_resonance_graft_admission_permit_ready"`
	SourceWeightedAdmissionResonanceGraftAdmissionReadinessReady                                                                                          bool   `json:"source_weighted_admission_resonance_graft_admission_readiness_ready"`
	SourceWriterInventoryVerified                                                                                                                         bool   `json:"source_writer_inventory_verified"`
	SourceWriterPreflightVerified                                                                                                                         bool   `json:"source_writer_preflight_verified"`
	SourceAdmissionRequired                                                                                                                               bool   `json:"source_admission_required"`
	SourceShadowOnly                                                                                                                                      bool   `json:"source_shadow_only"`
	SourceDryRunOnly                                                                                                                                      bool   `json:"source_dry_run_only"`
	SourceRequiresWriter                                                                                                                                  bool   `json:"source_requires_writer"`
	SourceRollbackRequired                                                                                                                                bool   `json:"source_rollback_required"`
	SourceRequiresRollback                                                                                                                                bool   `json:"source_requires_rollback"`
	SourceReadOnly                                                                                                                                        bool   `json:"source_read_only"`
	SourceReplayOnly                                                                                                                                      bool   `json:"source_replay_only"`
}

func runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitch(args []string) error {
	if len(args) != 2 {
		return fmt.Errorf("usage: --admission-live-route-weighted-admission-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-candidate-store-reader-proof-precondition-decision-promotion-switch RESONANCE_GRAFT_ADMISSION_FINAL_GATE_OBSERVATION_BOUNDARY_PREFLIGHT_GATE_CANDIDATE_STORE_READER_PROOF_PRECONDITION_DECISION_PROMOTION_REPORT RESONANCE_GRAFT_ADMISSION_FINAL_GATE_OBSERVATION_BOUNDARY_PREFLIGHT_GATE_CANDIDATE_STORE_READER_PROOF_PRECONDITION_DECISION_PROMOTION_SWITCH_REPORT")
	}
	promotionPath := args[0]
	outputPath := args[1]
	if strings.TrimSpace(outputPath) == "" {
		return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch output path missing")
	}
	promotion, root, err := readAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionReportForAssert(promotionPath)
	if err != nil {
		return err
	}
	if err := admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionReportError(promotion, root); err != nil {
		return err
	}
	sw := admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchReport{
		Schema:              admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchSchema,
		Status:              "shadow_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_disabled_dry_run",
		Target:              "live_route_admission_next_step",
		TargetKind:          "weighted_internal_world_shadow_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch",
		TargetMode:          "closed_switch_guard_dry_run",
		Action:              "hold_weighted_resonance_shadow_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_dry_run",
		SwitchState:         "disabled",
		SwitchAction:        "hold_pending_live_admission",
		Promotion:           "pending_live_admission",
		LedgerState:         "blocked",
		LedgerAction:        "reject_blocked_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_ledger_append",
		LedgerContract:      "none",
		LedgerEntrypoint:    "none",
		LedgerReceiptShape:  "none",
		LedgerWriteScope:    "none",
		LedgerReady:         false,
		LedgerAppendAllowed: false,
		WeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchReady: true,
		WeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionConsumed:    true,
		WeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionRequired:    true,
		NextStepBlockedWithoutResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitch: true,
		ReceiptShape:                 "weighted_resonance_shadow_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_receipt",
		SwitchKind:                   "shadow_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch",
		SwitchMode:                   "closed_promotion_switch_guard",
		SwitchStage:                  "post_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_pre_live_admission_switch",
		PromotionVerified:            true,
		PromotionHashVerified:        true,
		PromotionReadBackVerified:    true,
		DecisionVerified:             promotion.DecisionVerified,
		DecisionHashVerified:         promotion.DecisionHashVerified,
		DecisionReadBackVerified:     promotion.DecisionReadBackVerified,
		ProofPreconditionVerified:    promotion.ProofPreconditionVerified,
		PreconditionHashVerified:     promotion.PreconditionHashVerified,
		PreconditionReadBackVerified: promotion.PreconditionReadBackVerified,
		ProofVerified:                promotion.ProofVerified,
		ProofHashVerified:            promotion.ProofHashVerified,
		ProofReadBackVerified:        promotion.ProofReadBackVerified,
		StoreReaderVerified:          promotion.StoreReaderVerified,
		StoreVerified:                promotion.StoreVerified,
		CandidateVerified:            promotion.CandidateVerified,
		GateVerified:                 promotion.GateVerified,
		PreflightVerified:            promotion.PreflightVerified,
		BoundaryVerified:             promotion.BoundaryVerified,
		ObservationVerified:          promotion.ObservationVerified,
		ReceiverVerified:             promotion.ReceiverVerified,
		IntentVerified:               promotion.IntentVerified,
		FinalGateVerified:            promotion.FinalGateVerified,
		SealVerified:                 promotion.SealVerified,
		PermitVerified:               promotion.PermitVerified,
		AuthorityVerified:            promotion.AuthorityVerified,
		ReaderHashVerified:           promotion.ReaderHashVerified,
		ReaderReplayVerified:         promotion.ReaderReplayVerified,
		ReaderReadBackVerified:       promotion.ReaderReadBackVerified,
		StoreHashVerified:            promotion.StoreHashVerified,
		StoreReadBackVerified:        promotion.StoreReadBackVerified,
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
		Reason:                       "weighted resonance shadow graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion held at disabled switch without mutation",

		SourceSchema: promotion.Schema, SourceStatus: promotion.Status, SourceTarget: promotion.Target, SourceReport: promotionPath,
		SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionID:       promotion.WeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionID,
		SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionReady:    promotion.WeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionReady,
		SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionCausalID: promotion.CausalID,
		SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionHash:     promotion.PromotionHash,
		SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionReadBack: promotion.ReadBackHash,
		SourcePromotion: promotion.Promotion, SourcePromotionAction: promotion.Action, SourcePromotionReceiptShape: promotion.ReceiptShape, SourcePromotionKind: promotion.PromotionKind, SourcePromotionMode: promotion.PromotionMode, SourcePromotionStage: promotion.PromotionStage,
		SourcePromotionLedgerReady: promotion.LedgerReady, SourcePromotionLedgerAppendAllowed: promotion.LedgerAppendAllowed,
		SourcePromotionAdmissionRequired: promotion.AdmissionRequired, SourcePromotionShadowOnly: promotion.ShadowOnly, SourcePromotionGraftAllowed: promotion.GraftAllowed, SourcePromotionDryRunOnly: promotion.DryRunOnly, SourcePromotionLiveReady: promotion.LiveReady,
		SourcePromotionRawDreamTextAllowed: promotion.RawDreamTextAllowed, SourcePromotionRawDreamTextObserved: promotion.RawDreamTextObserved, SourcePromotionRawDreamTextForwarded: promotion.RawDreamTextForwarded, SourcePromotionJanusSurfaceAllowed: promotion.JanusSurfaceAllowed, SourcePromotionCoocLearningAllowed: promotion.CoocLearningAllowed, SourcePromotionDeltaHarvestAllowed: promotion.DeltaHarvestAllowed, SourcePromotionBodyMutationAllowed: promotion.BodyMutationAllowed,
		SourcePromotionRollbackRequired: promotion.RollbackRequired, SourcePromotionReadOnly: promotion.ReadOnly, SourcePromotionReplayOnly: promotion.ReplayOnly, SourcePromotionAuthorityGranted: promotion.AuthorityGranted, SourcePromotionContractsReady: promotion.ContractsReady, SourcePromotionWriteAllowed: promotion.WriteAllowed, SourcePromotionAdmissionAllowed: promotion.AdmissionAllowed, SourcePromotionLiveAdmissionEnabled: promotion.LiveAdmissionEnabled, SourcePromotionMutatesState: promotion.MutatesState, SourcePromotionBodyTarget: promotion.BodyTarget, SourcePromotionPassed: promotion.Passed, SourcePromotionReason: promotion.Reason,

		SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionID:       promotion.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionID,
		SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionReady:    promotion.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionReady,
		SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionCausalID: promotion.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionCausalID,
		SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionHash:     promotion.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionHash,
		SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionReadBack: promotion.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionReadBack,
		SourceDecision: promotion.SourceDecision, SourceDecisionAction: promotion.SourceDecisionAction, SourceDecisionReceiptShape: promotion.SourceDecisionReceiptShape, SourceDecisionKind: promotion.SourceDecisionKind, SourceDecisionMode: promotion.SourceDecisionMode, SourceDecisionStage: promotion.SourceDecisionStage,
		SourceDecisionLedgerReady: promotion.SourceDecisionLedgerReady, SourceDecisionLedgerAppendAllowed: promotion.SourceDecisionLedgerAppendAllowed,
		SourceDecisionAdmissionRequired: promotion.SourceDecisionAdmissionRequired, SourceDecisionShadowOnly: promotion.SourceDecisionShadowOnly, SourceDecisionGraftAllowed: promotion.SourceDecisionGraftAllowed, SourceDecisionDryRunOnly: promotion.SourceDecisionDryRunOnly, SourceDecisionLiveReady: promotion.SourceDecisionLiveReady,
		SourceDecisionRawDreamTextAllowed: promotion.SourceDecisionRawDreamTextAllowed, SourceDecisionRawDreamTextObserved: promotion.SourceDecisionRawDreamTextObserved, SourceDecisionRawDreamTextForwarded: promotion.SourceDecisionRawDreamTextForwarded, SourceDecisionJanusSurfaceAllowed: promotion.SourceDecisionJanusSurfaceAllowed, SourceDecisionCoocLearningAllowed: promotion.SourceDecisionCoocLearningAllowed, SourceDecisionDeltaHarvestAllowed: promotion.SourceDecisionDeltaHarvestAllowed, SourceDecisionBodyMutationAllowed: promotion.SourceDecisionBodyMutationAllowed,
		SourceDecisionRollbackRequired: promotion.SourceDecisionRollbackRequired, SourceDecisionReadOnly: promotion.SourceDecisionReadOnly, SourceDecisionReplayOnly: promotion.SourceDecisionReplayOnly, SourceDecisionAuthorityGranted: promotion.SourceDecisionAuthorityGranted, SourceDecisionContractsReady: promotion.SourceDecisionContractsReady, SourceDecisionWriteAllowed: promotion.SourceDecisionWriteAllowed, SourceDecisionAdmissionAllowed: promotion.SourceDecisionAdmissionAllowed, SourceDecisionLiveAdmissionEnabled: promotion.SourceDecisionLiveAdmissionEnabled, SourceDecisionMutatesState: promotion.SourceDecisionMutatesState, SourceDecisionBodyTarget: promotion.SourceDecisionBodyTarget, SourceDecisionPassed: promotion.SourceDecisionPassed,

		SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionID:    promotion.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionID,
		SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionReady: promotion.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionReady,
		SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofID:                promotion.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofID,
		SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofReady:             promotion.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofReady,
		SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderID:                     promotion.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderID,
		SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderReady:                  promotion.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderReady,
		SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreID:                           promotion.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreID,
		SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReady:                        promotion.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReady,
		SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateID:                                promotion.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateID,
		SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateReady:                             promotion.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateReady,
		SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateID:                                         promotion.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateID,
		SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateReady:                                      promotion.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateReady,
		SourceAdmissionFinalGateObservationBoundaryPreflightGateReady:                                                                     promotion.SourceAdmissionFinalGateObservationBoundaryPreflightGateReady,
		SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightID:                                             promotion.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightID,
		SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightReady:                                          promotion.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightReady,
		SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryID:                                                      promotion.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryID,
		SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryReady:                                                   promotion.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryReady,
		SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationID:                                                              promotion.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationID,
		SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationReady:                                                           promotion.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationReady,
		SourceWeightedAdmissionResonanceGraftAdmissionFinalGateReceiverID:                                                                 promotion.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateReceiverID,
		SourceWeightedAdmissionResonanceGraftAdmissionFinalGateReceiverReady:                                                              promotion.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateReceiverReady,
		SourceWeightedAdmissionResonanceGraftAdmissionFinalGateIntentReady:                                                                promotion.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateIntentReady,
		SourceWeightedAdmissionResonanceGraftAdmissionFinalGateReady:                                                                      promotion.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateReady,
		SourceWeightedAdmissionResonanceGraftAdmissionSealReady:                                                                           promotion.SourceWeightedAdmissionResonanceGraftAdmissionSealReady,
		SourceWeightedAdmissionResonanceGraftAdmissionAuthorityReady:                                                                      promotion.SourceWeightedAdmissionResonanceGraftAdmissionAuthorityReady,
		SourceWeightedAdmissionResonanceGraftAdmissionPermitReady:                                                                         promotion.SourceWeightedAdmissionResonanceGraftAdmissionPermitReady,
		SourceWeightedAdmissionResonanceGraftAdmissionReadinessReady:                                                                      promotion.SourceWeightedAdmissionResonanceGraftAdmissionReadinessReady,
		SourceWriterInventoryVerified: promotion.SourceWriterInventoryVerified, SourceWriterPreflightVerified: promotion.SourceWriterPreflightVerified, SourceAdmissionRequired: promotion.SourceAdmissionRequired, SourceShadowOnly: promotion.SourceShadowOnly, SourceDryRunOnly: promotion.SourceDryRunOnly, SourceRequiresWriter: promotion.SourceRequiresWriter, SourceRollbackRequired: promotion.SourceRollbackRequired, SourceRequiresRollback: promotion.SourceRequiresRollback, SourceReadOnly: promotion.SourceReadOnly, SourceReplayOnly: promotion.SourceReplayOnly,
	}
	sw.CausalID = admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchCausalID(sw)
	sw.SwitchHash = admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchHash(sw)
	sw.ReadBackHash = admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchReadBackHash(sw)
	sw.WeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchID = admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchID(sw)
	if sw.CausalID == "" || sw.SwitchHash == "" || sw.ReadBackHash == "" || sw.WeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchID == "" || sw.SwitchHash == sw.ReadBackHash {
		return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch read-back proof failed")
	}
	raw, err := json.MarshalIndent(sw, "", "  ")
	if err != nil {
		return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch marshal failed: %w", err)
	}
	raw = append(raw, '\n')
	if err := os.WriteFile(outputPath, raw, 0600); err != nil {
		return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch write failed: %w", err)
	}
	fmt.Printf("[admission-live-route-weighted-admission-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-candidate-store-reader-proof-precondition-decision-promotion-switch] pass: resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_report=%s resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_report=%s\n", outputPath, promotionPath)
	return nil
}

func runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchAssert(args []string) error {
	if len(args) != 1 {
		return fmt.Errorf("usage: --admission-live-route-weighted-admission-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-candidate-store-reader-proof-precondition-decision-promotion-switch-assert REPORT")
	}
	report, root, err := readAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchReportForAssert(args[0])
	if err != nil {
		return err
	}
	return admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchReportError(report, root)
}

func admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchReportError(report admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchReport, root map[string]json.RawMessage) error {
	if _, ok := root["schema"]; !ok {
		return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch schema missing")
	}
	if report.Schema != admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchSchema {
		return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch schema mismatch: got %q want %q", report.Schema, admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchSchema)
	}
	if report.Status != "shadow_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_disabled_dry_run" {
		return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch status mismatch: got %q want %q", report.Status, "shadow_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_disabled_dry_run")
	}
	if report.Target != "live_route_admission_next_step" || report.TargetKind != "weighted_internal_world_shadow_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch" || report.TargetMode != "closed_switch_guard_dry_run" || report.Action != "hold_weighted_resonance_shadow_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_dry_run" {
		return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch route shape mismatch")
	}
	if report.SwitchState != "disabled" {
		return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch switch_state mismatch: got %q want %q", report.SwitchState, "disabled")
	}
	if report.SwitchAction != "hold_pending_live_admission" {
		return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch switch_action mismatch: got %q want %q", report.SwitchAction, "hold_pending_live_admission")
	}
	if report.Promotion != "pending_live_admission" {
		return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch promotion mismatch: got %q want %q", report.Promotion, "pending_live_admission")
	}
	if report.LedgerState != "blocked" || report.LedgerAction != "reject_blocked_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_ledger_append" || report.LedgerContract != "none" || report.LedgerEntrypoint != "none" || report.LedgerReceiptShape != "none" || report.LedgerWriteScope != "none" {
		return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch ledger guard mismatch")
	}
	if report.ReceiptShape != "weighted_resonance_shadow_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_receipt" || report.SwitchKind != "shadow_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch" || report.SwitchMode != "closed_promotion_switch_guard" || report.SwitchStage != "post_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_pre_live_admission_switch" {
		return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch shape mismatch")
	}
	for _, required := range []struct {
		name  string
		value bool
	}{
		{"weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_ready", report.WeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchReady}, {"weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_consumed", report.WeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionConsumed}, {"weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_required", report.WeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionRequired}, {"next_step_blocked_without_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch", report.NextStepBlockedWithoutResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitch},
		{"promotion_verified", report.PromotionVerified}, {"promotion_hash_verified", report.PromotionHashVerified}, {"promotion_read_back_verified", report.PromotionReadBackVerified}, {"decision_verified", report.DecisionVerified}, {"decision_hash_verified", report.DecisionHashVerified}, {"decision_read_back_verified", report.DecisionReadBackVerified}, {"proof_precondition_verified", report.ProofPreconditionVerified}, {"precondition_hash_verified", report.PreconditionHashVerified}, {"precondition_read_back_verified", report.PreconditionReadBackVerified}, {"proof_verified", report.ProofVerified}, {"proof_hash_verified", report.ProofHashVerified}, {"proof_read_back_verified", report.ProofReadBackVerified}, {"store_reader_verified", report.StoreReaderVerified}, {"store_verified", report.StoreVerified}, {"candidate_verified", report.CandidateVerified}, {"gate_verified", report.GateVerified}, {"preflight_verified", report.PreflightVerified}, {"boundary_verified", report.BoundaryVerified}, {"observation_verified", report.ObservationVerified}, {"receiver_verified", report.ReceiverVerified}, {"intent_verified", report.IntentVerified}, {"final_gate_verified", report.FinalGateVerified}, {"seal_verified", report.SealVerified}, {"permit_verified", report.PermitVerified}, {"authority_verified", report.AuthorityVerified}, {"reader_hash_verified", report.ReaderHashVerified}, {"reader_replay_verified", report.ReaderReplayVerified}, {"reader_read_back_verified", report.ReaderReadBackVerified}, {"store_hash_verified", report.StoreHashVerified}, {"store_read_back_verified", report.StoreReadBackVerified}, {"admission_required", report.AdmissionRequired}, {"shadow_only", report.ShadowOnly}, {"dry_run_only", report.DryRunOnly}, {"live_ready", report.LiveReady}, {"rollback_required", report.RollbackRequired}, {"read_only", report.ReadOnly}, {"replay_only", report.ReplayOnly},
		{"source_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_ready", report.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionReady}, {"source_promotion_admission_required", report.SourcePromotionAdmissionRequired}, {"source_promotion_shadow_only", report.SourcePromotionShadowOnly}, {"source_promotion_dry_run_only", report.SourcePromotionDryRunOnly}, {"source_promotion_live_ready", report.SourcePromotionLiveReady}, {"source_promotion_rollback_required", report.SourcePromotionRollbackRequired}, {"source_promotion_read_only", report.SourcePromotionReadOnly}, {"source_promotion_replay_only", report.SourcePromotionReplayOnly}, {"source_promotion_passed", report.SourcePromotionPassed},
		{"source_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_ready", report.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionReady}, {"source_decision_admission_required", report.SourceDecisionAdmissionRequired}, {"source_decision_shadow_only", report.SourceDecisionShadowOnly}, {"source_decision_dry_run_only", report.SourceDecisionDryRunOnly}, {"source_decision_live_ready", report.SourceDecisionLiveReady}, {"source_decision_rollback_required", report.SourceDecisionRollbackRequired}, {"source_decision_read_only", report.SourceDecisionReadOnly}, {"source_decision_replay_only", report.SourceDecisionReplayOnly}, {"source_decision_passed", report.SourceDecisionPassed},
		{"source_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_ready", report.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionReady}, {"source_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_ready", report.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofReady}, {"source_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_ready", report.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderReady}, {"source_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_ready", report.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReady}, {"source_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_ready", report.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateReady}, {"source_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_ready", report.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateReady}, {"source_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_ready", report.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightReady}, {"source_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_ready", report.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryReady}, {"source_weighted_admission_resonance_graft_admission_final_gate_observation_ready", report.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationReady}, {"source_weighted_admission_resonance_graft_admission_final_gate_receiver_ready", report.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateReceiverReady}, {"source_weighted_admission_resonance_graft_admission_final_gate_intent_ready", report.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateIntentReady}, {"source_weighted_admission_resonance_graft_admission_final_gate_ready", report.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateReady}, {"source_weighted_admission_resonance_graft_admission_seal_ready", report.SourceWeightedAdmissionResonanceGraftAdmissionSealReady}, {"source_weighted_admission_resonance_graft_admission_authority_ready", report.SourceWeightedAdmissionResonanceGraftAdmissionAuthorityReady}, {"source_weighted_admission_resonance_graft_admission_permit_ready", report.SourceWeightedAdmissionResonanceGraftAdmissionPermitReady}, {"source_weighted_admission_resonance_graft_admission_readiness_ready", report.SourceWeightedAdmissionResonanceGraftAdmissionReadinessReady}, {"source_writer_inventory_verified", report.SourceWriterInventoryVerified}, {"source_writer_preflight_verified", report.SourceWriterPreflightVerified}, {"source_admission_required", report.SourceAdmissionRequired}, {"source_shadow_only", report.SourceShadowOnly}, {"source_dry_run_only", report.SourceDryRunOnly}, {"source_requires_writer", report.SourceRequiresWriter}, {"source_rollback_required", report.SourceRollbackRequired}, {"source_requires_rollback", report.SourceRequiresRollback}, {"source_read_only", report.SourceReadOnly}, {"source_replay_only", report.SourceReplayOnly}, {"passed", report.Passed},
	} {
		if !required.value {
			return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch %s not ready", required.name)
		}
	}
	for _, closed := range []struct {
		name  string
		value bool
	}{
		{"ledger_ready", report.LedgerReady}, {"ledger_append_allowed", report.LedgerAppendAllowed}, {"graft_allowed", report.GraftAllowed}, {"raw_dream_text_allowed", report.RawDreamTextAllowed}, {"raw_dream_text_observed", report.RawDreamTextObserved}, {"raw_dream_text_forwarded", report.RawDreamTextForwarded}, {"janus_surface_allowed", report.JanusSurfaceAllowed}, {"cooc_learning_allowed", report.CoocLearningAllowed}, {"delta_harvest_allowed", report.DeltaHarvestAllowed}, {"body_mutation_allowed", report.BodyMutationAllowed}, {"authority_granted", report.AuthorityGranted}, {"contracts_ready", report.ContractsReady}, {"write_allowed", report.WriteAllowed}, {"admission_allowed", report.AdmissionAllowed}, {"live_admission_enabled", report.LiveAdmissionEnabled}, {"mutates_state", report.MutatesState},
		{"source_promotion_ledger_ready", report.SourcePromotionLedgerReady}, {"source_promotion_ledger_append_allowed", report.SourcePromotionLedgerAppendAllowed}, {"source_promotion_graft_allowed", report.SourcePromotionGraftAllowed}, {"source_promotion_raw_dream_text_allowed", report.SourcePromotionRawDreamTextAllowed}, {"source_promotion_raw_dream_text_observed", report.SourcePromotionRawDreamTextObserved}, {"source_promotion_raw_dream_text_forwarded", report.SourcePromotionRawDreamTextForwarded}, {"source_promotion_janus_surface_allowed", report.SourcePromotionJanusSurfaceAllowed}, {"source_promotion_cooc_learning_allowed", report.SourcePromotionCoocLearningAllowed}, {"source_promotion_delta_harvest_allowed", report.SourcePromotionDeltaHarvestAllowed}, {"source_promotion_body_mutation_allowed", report.SourcePromotionBodyMutationAllowed}, {"source_promotion_authority_granted", report.SourcePromotionAuthorityGranted}, {"source_promotion_contracts_ready", report.SourcePromotionContractsReady}, {"source_promotion_write_allowed", report.SourcePromotionWriteAllowed}, {"source_promotion_admission_allowed", report.SourcePromotionAdmissionAllowed}, {"source_promotion_live_admission_enabled", report.SourcePromotionLiveAdmissionEnabled}, {"source_promotion_mutates_state", report.SourcePromotionMutatesState},
		{"source_decision_ledger_ready", report.SourceDecisionLedgerReady}, {"source_decision_ledger_append_allowed", report.SourceDecisionLedgerAppendAllowed}, {"source_decision_graft_allowed", report.SourceDecisionGraftAllowed}, {"source_decision_raw_dream_text_allowed", report.SourceDecisionRawDreamTextAllowed}, {"source_decision_raw_dream_text_observed", report.SourceDecisionRawDreamTextObserved}, {"source_decision_raw_dream_text_forwarded", report.SourceDecisionRawDreamTextForwarded}, {"source_decision_janus_surface_allowed", report.SourceDecisionJanusSurfaceAllowed}, {"source_decision_cooc_learning_allowed", report.SourceDecisionCoocLearningAllowed}, {"source_decision_delta_harvest_allowed", report.SourceDecisionDeltaHarvestAllowed}, {"source_decision_body_mutation_allowed", report.SourceDecisionBodyMutationAllowed}, {"source_decision_authority_granted", report.SourceDecisionAuthorityGranted}, {"source_decision_contracts_ready", report.SourceDecisionContractsReady}, {"source_decision_write_allowed", report.SourceDecisionWriteAllowed}, {"source_decision_admission_allowed", report.SourceDecisionAdmissionAllowed}, {"source_decision_live_admission_enabled", report.SourceDecisionLiveAdmissionEnabled}, {"source_decision_mutates_state", report.SourceDecisionMutatesState}, {"source_admission_final_gate_observation_boundary_preflight_gate_ready", report.SourceAdmissionFinalGateObservationBoundaryPreflightGateReady},
	} {
		if closed.value {
			return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch opened %s", closed.name)
		}
	}
	for _, requiredString := range []struct{ name, value string }{
		{"weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_id", report.WeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchID}, {"causal_id", report.CausalID}, {"switch_hash", report.SwitchHash}, {"read_back_hash", report.ReadBackHash}, {"source_report", report.SourceReport}, {"source_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_id", report.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionID}, {"source_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_hash", report.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionHash}, {"source_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_read_back_hash", report.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionReadBack}, {"source_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_id", report.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionID}, {"source_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_hash", report.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionHash}, {"source_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_read_back_hash", report.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionReadBack}, {"source_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_id", report.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionID}, {"source_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_id", report.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofID}, {"source_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_id", report.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderID}, {"source_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_id", report.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreID}, {"source_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_id", report.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateID}, {"source_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_id", report.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateID},
	} {
		if strings.TrimSpace(requiredString.value) == "" {
			return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch %s missing", requiredString.name)
		}
	}
	if report.SourceSchema != admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSchema {
		return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch source_schema mismatch: got %q want %q", report.SourceSchema, admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSchema)
	}
	if report.SourceStatus != "shadow_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_ready_dry_run" || report.SourceTarget != "live_route_admission_next_step" || report.SourcePromotion != "pending_live_admission" {
		return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch source promotion route mismatch")
	}
	if report.SourcePromotionReceiptShape != "weighted_resonance_shadow_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_receipt" || report.SourcePromotionKind != "shadow_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion" || report.SourcePromotionMode != "closed_decision_promotion" || report.SourcePromotionStage != "post_preflight_gate_candidate_store_reader_proof_precondition_decision_pre_live_admission_promotion" {
		return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch source promotion shape mismatch")
	}
	if report.SourceDecision != "shadow_ready" || report.SourceDecisionReceiptShape != "weighted_resonance_shadow_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_receipt" || report.SourceDecisionKind != "shadow_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision" || report.SourceDecisionMode != "closed_proof_precondition_decision" || report.SourceDecisionStage != "post_preflight_gate_candidate_store_reader_proof_precondition_pre_live_admission_decision" {
		return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch source decision shape mismatch")
	}
	if report.BodyTarget != "none" || report.SourcePromotionBodyTarget != "none" || report.SourceDecisionBodyTarget != "none" {
		return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch body target mismatch")
	}
	if !strings.HasPrefix(report.WeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchID, "weighted-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-candidate-store-reader-proof-precondition-decision-promotion-switch-id-") || !strings.HasPrefix(report.CausalID, "weighted-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-candidate-store-reader-proof-precondition-decision-promotion-switch-causal-") || !strings.HasPrefix(report.SwitchHash, "weighted-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-candidate-store-reader-proof-precondition-decision-promotion-switch-") || !strings.HasPrefix(report.ReadBackHash, "weighted-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-candidate-store-reader-proof-precondition-decision-promotion-switch-read-") || report.SwitchHash == report.ReadBackHash {
		return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch prefix mismatch")
	}
	if !strings.HasPrefix(report.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionID, "weighted-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-candidate-store-reader-proof-precondition-decision-promotion-id-") || !strings.HasPrefix(report.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionID, "weighted-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-candidate-store-reader-proof-precondition-decision-id-") || !strings.HasPrefix(report.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionID, "weighted-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-candidate-store-reader-proof-precondition-id-") || !strings.HasPrefix(report.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofID, "weighted-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-candidate-store-reader-proof-id-") || !strings.HasPrefix(report.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderID, "weighted-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-candidate-store-reader-id-") || !strings.HasPrefix(report.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreID, "weighted-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-candidate-store-id-") || !strings.HasPrefix(report.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateID, "weighted-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-candidate-id-") || !strings.HasPrefix(report.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateID, "weighted-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-id-") {
		return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch source chain prefix mismatch")
	}
	if report.CausalID == "" || report.CausalID != admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchCausalID(report) {
		return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch causal_id mismatch")
	}
	if report.SwitchHash == "" || report.SwitchHash != admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchHash(report) {
		return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch switch_hash mismatch")
	}
	if report.ReadBackHash == "" || report.ReadBackHash != admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchReadBackHash(report) {
		return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch read_back_hash mismatch")
	}
	if report.WeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchID == "" || report.WeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchID != admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchID(report) {
		return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch id mismatch")
	}
	if report.Reason != "weighted resonance shadow graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion held at disabled switch without mutation" {
		return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch reason mismatch: got %q", report.Reason)
	}
	return nil
}

func admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchCausalID(sw admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchReport) string {
	h := hashJSON(map[string]interface{}{"source_promotion_id": sw.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionID, "source_promotion_read_back_hash": sw.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionReadBack, "source_decision_id": sw.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionID, "source_precondition_id": sw.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionID, "target": sw.Target, "switch_kind": sw.SwitchKind, "switch_stage": sw.SwitchStage})
	if h == "" {
		return ""
	}
	return "weighted-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-candidate-store-reader-proof-precondition-decision-promotion-switch-causal-" + h
}

func admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchHash(sw admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchReport) string {
	h := hashJSON(map[string]interface{}{"causal_id": sw.CausalID, "source_promotion_id": sw.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionID, "source_promotion_hash": sw.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionHash, "source_promotion_read_back_hash": sw.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionReadBack, "switch_state": sw.SwitchState, "switch_action": sw.SwitchAction, "promotion": sw.Promotion, "switch_mode": sw.SwitchMode, "receipt_shape": sw.ReceiptShape, "promotion_verified": sw.PromotionVerified, "promotion_hash_verified": sw.PromotionHashVerified, "promotion_read_back_verified": sw.PromotionReadBackVerified, "read_only": sw.ReadOnly, "replay_only": sw.ReplayOnly, "admission_required": sw.AdmissionRequired, "shadow_only": sw.ShadowOnly, "dry_run_only": sw.DryRunOnly, "graft_allowed": sw.GraftAllowed, "ledger_append_allowed": sw.LedgerAppendAllowed})
	if h == "" {
		return ""
	}
	return "weighted-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-candidate-store-reader-proof-precondition-decision-promotion-switch-" + h
}

func admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchReadBackHash(sw admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchReport) string {
	h := hashJSON(map[string]interface{}{"switch_hash": sw.SwitchHash, "source_promotion_id": sw.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionID, "source_decision_id": sw.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionID, "switch_state": sw.SwitchState, "switch_action": sw.SwitchAction, "switch_ready": sw.WeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchReady, "body_mutation": sw.BodyMutationAllowed, "live_admission": sw.LiveAdmissionEnabled, "write_allowed": sw.WriteAllowed, "admission_allowed": sw.AdmissionAllowed, "ledger_append_allowed": sw.LedgerAppendAllowed})
	if h == "" {
		return ""
	}
	return "weighted-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-candidate-store-reader-proof-precondition-decision-promotion-switch-read-" + h
}

func admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchID(sw admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchReport) string {
	h := hashJSON(map[string]interface{}{"schema": sw.Schema, "status": sw.Status, "action": sw.Action, "switch_state": sw.SwitchState, "switch_action": sw.SwitchAction, "promotion": sw.Promotion, "source_report": sw.SourceReport, "source_promotion_id": sw.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionID, "source_decision_id": sw.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionID, "source_precondition_id": sw.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionID, "source_proof_id": sw.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofID, "source_reader_id": sw.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderID, "causal_id": sw.CausalID, "switch_hash": sw.SwitchHash, "read_back_hash": sw.ReadBackHash, "receipt_shape": sw.ReceiptShape, "switch_kind": sw.SwitchKind, "switch_mode": sw.SwitchMode, "switch_stage": sw.SwitchStage, "body_target": sw.BodyTarget, "ready": sw.WeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchReady, "promotion_verified": sw.PromotionVerified, "promotion_hash_verified": sw.PromotionHashVerified, "promotion_read_back_verified": sw.PromotionReadBackVerified, "admission_required": sw.AdmissionRequired, "shadow_only": sw.ShadowOnly, "graft_allowed": sw.GraftAllowed, "dry_run_only": sw.DryRunOnly, "read_only": sw.ReadOnly, "replay_only": sw.ReplayOnly, "live_ready": sw.LiveReady, "contracts_ready": sw.ContractsReady, "write_allowed": sw.WriteAllowed, "admission_allowed": sw.AdmissionAllowed, "live_admission_enabled": sw.LiveAdmissionEnabled, "mutates_state": sw.MutatesState, "ledger_append_allowed": sw.LedgerAppendAllowed, "next_step_blocked_without": sw.NextStepBlockedWithoutResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitch})
	if h == "" {
		return ""
	}
	return "weighted-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-candidate-store-reader-proof-precondition-decision-promotion-switch-id-" + h
}

func readAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchReportForAssert(path string) (admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchReport, map[string]json.RawMessage, error) {
	var report admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchReport
	if strings.TrimSpace(path) == "" {
		return report, nil, fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch path missing")
	}
	raw, err := os.ReadFile(path)
	if err != nil {
		if os.IsNotExist(err) {
			return report, nil, fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch not written")
		}
		return report, nil, err
	}
	if len(raw) == 0 {
		return report, nil, fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch not written")
	}
	var root map[string]json.RawMessage
	if err := json.Unmarshal(raw, &root); err != nil {
		return report, nil, fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch JSON invalid: %w", err)
	}
	if err := json.Unmarshal(raw, &report); err != nil {
		return report, nil, fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch decode failed: %w", err)
	}
	return report, root, nil
}
