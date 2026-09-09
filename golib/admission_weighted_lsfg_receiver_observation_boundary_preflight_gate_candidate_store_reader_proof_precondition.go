package main

import (
	"encoding/json"
	"fmt"
	"os"
	"strings"
)

const admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReceiverObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionSchema = "arianna.live_route_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_final_gate_receiver_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition.v1"

type admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReceiverObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionReport struct {
	Schema                                                                                                                                                                                                                                                      string `json:"schema"`
	Status                                                                                                                                                                                                                                                      string `json:"status"`
	Target                                                                                                                                                                                                                                                      string `json:"target"`
	TargetKind                                                                                                                                                                                                                                                  string `json:"target_kind"`
	TargetMode                                                                                                                                                                                                                                                  string `json:"target_mode"`
	Action                                                                                                                                                                                                                                                      string `json:"action"`
	LedgerState                                                                                                                                                                                                                                                 string `json:"ledger_state"`
	LedgerAction                                                                                                                                                                                                                                                string `json:"ledger_action"`
	LedgerContract                                                                                                                                                                                                                                              string `json:"ledger_contract"`
	LedgerEntrypoint                                                                                                                                                                                                                                            string `json:"ledger_entrypoint"`
	LedgerReceiptShape                                                                                                                                                                                                                                          string `json:"ledger_receipt_shape"`
	LedgerWriteScope                                                                                                                                                                                                                                            string `json:"ledger_write_scope"`
	LedgerReady                                                                                                                                                                                                                                                 bool   `json:"ledger_ready"`
	LedgerAppendAllowed                                                                                                                                                                                                                                         bool   `json:"ledger_append_allowed"`
	WeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReceiverObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionReady bool   `json:"weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_final_gate_receiver_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_ready"`
	WeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReceiverObservationBoundaryPreflightGateCandidateStoreReaderProofConsumed          bool   `json:"weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_final_gate_receiver_observation_boundary_preflight_gate_candidate_store_reader_proof_consumed"`
	WeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReceiverObservationBoundaryPreflightGateCandidateStoreReaderProofRequired          bool   `json:"weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_final_gate_receiver_observation_boundary_preflight_gate_candidate_store_reader_proof_required"`
	NextStepBlockedWithoutResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPrecondition                                                                                                                                 bool   `json:"next_step_blocked_without_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition"`
	WeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReceiverObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionID    string `json:"weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_final_gate_receiver_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_id"`
	ReceiptShape                                                                                                                                                                                                                                                string `json:"receipt_shape"`
	PreconditionKind                                                                                                                                                                                                                                            string `json:"precondition_kind"`
	PreconditionMode                                                                                                                                                                                                                                            string `json:"precondition_mode"`
	PreconditionStage                                                                                                                                                                                                                                           string `json:"precondition_stage"`
	CausalID                                                                                                                                                                                                                                                    string `json:"causal_id"`
	PreconditionHash                                                                                                                                                                                                                                            string `json:"precondition_hash"`
	ReadBackHash                                                                                                                                                                                                                                                string `json:"read_back_hash"`
	ProofVerified                                                                                                                                                                                                                                               bool   `json:"proof_verified"`
	ProofHashVerified                                                                                                                                                                                                                                           bool   `json:"proof_hash_verified"`
	ProofReadBackVerified                                                                                                                                                                                                                                       bool   `json:"proof_read_back_verified"`
	StoreReaderVerified                                                                                                                                                                                                                                         bool   `json:"store_reader_verified"`
	StoreVerified                                                                                                                                                                                                                                               bool   `json:"store_verified"`
	CandidateVerified                                                                                                                                                                                                                                           bool   `json:"candidate_verified"`
	GateVerified                                                                                                                                                                                                                                                bool   `json:"gate_verified"`
	PreflightVerified                                                                                                                                                                                                                                           bool   `json:"preflight_verified"`
	BoundaryVerified                                                                                                                                                                                                                                            bool   `json:"boundary_verified"`
	ObservationVerified                                                                                                                                                                                                                                         bool   `json:"observation_verified"`
	ReceiverVerified                                                                                                                                                                                                                                            bool   `json:"receiver_verified"`
	IntentVerified                                                                                                                                                                                                                                              bool   `json:"intent_verified"`
	FinalGateVerified                                                                                                                                                                                                                                           bool   `json:"final_gate_verified"`
	SealVerified                                                                                                                                                                                                                                                bool   `json:"seal_verified"`
	PermitVerified                                                                                                                                                                                                                                              bool   `json:"permit_verified"`
	AuthorityVerified                                                                                                                                                                                                                                           bool   `json:"authority_verified"`
	ReaderHashVerified                                                                                                                                                                                                                                          bool   `json:"reader_hash_verified"`
	ReaderReplayVerified                                                                                                                                                                                                                                        bool   `json:"reader_replay_verified"`
	ReaderReadBackVerified                                                                                                                                                                                                                                      bool   `json:"reader_read_back_verified"`
	StoreHashVerified                                                                                                                                                                                                                                           bool   `json:"store_hash_verified"`
	StoreReadBackVerified                                                                                                                                                                                                                                       bool   `json:"store_read_back_verified"`
	AdmissionRequired                                                                                                                                                                                                                                           bool   `json:"admission_required"`
	ShadowOnly                                                                                                                                                                                                                                                  bool   `json:"shadow_only"`
	GraftAllowed                                                                                                                                                                                                                                                bool   `json:"graft_allowed"`
	DryRunOnly                                                                                                                                                                                                                                                  bool   `json:"dry_run_only"`
	LiveReady                                                                                                                                                                                                                                                   bool   `json:"live_ready"`
	RawDreamTextAllowed                                                                                                                                                                                                                                         bool   `json:"raw_dream_text_allowed"`
	RawDreamTextObserved                                                                                                                                                                                                                                        bool   `json:"raw_dream_text_observed"`
	RawDreamTextForwarded                                                                                                                                                                                                                                       bool   `json:"raw_dream_text_forwarded"`
	JanusSurfaceAllowed                                                                                                                                                                                                                                         bool   `json:"janus_surface_allowed"`
	CoocLearningAllowed                                                                                                                                                                                                                                         bool   `json:"cooc_learning_allowed"`
	DeltaHarvestAllowed                                                                                                                                                                                                                                         bool   `json:"delta_harvest_allowed"`
	BodyMutationAllowed                                                                                                                                                                                                                                         bool   `json:"body_mutation_allowed"`
	RollbackRequired                                                                                                                                                                                                                                            bool   `json:"rollback_required"`
	ReadOnly                                                                                                                                                                                                                                                    bool   `json:"read_only"`
	ReplayOnly                                                                                                                                                                                                                                                  bool   `json:"replay_only"`
	AuthorityGranted                                                                                                                                                                                                                                            bool   `json:"authority_granted"`
	ContractsReady                                                                                                                                                                                                                                              bool   `json:"contracts_ready"`
	WriteAllowed                                                                                                                                                                                                                                                bool   `json:"write_allowed"`
	AdmissionAllowed                                                                                                                                                                                                                                            bool   `json:"admission_allowed"`
	LiveAdmissionEnabled                                                                                                                                                                                                                                        bool   `json:"live_admission_enabled"`
	MutatesState                                                                                                                                                                                                                                                bool   `json:"mutates_state"`
	BodyTarget                                                                                                                                                                                                                                                  string `json:"body_target"`
	Passed                                                                                                                                                                                                                                                      bool   `json:"passed"`
	Reason                                                                                                                                                                                                                                                      string `json:"reason"`

	SourceSchema                                                                                                                                                                                                                                             string `json:"source_schema"`
	SourceStatus                                                                                                                                                                                                                                             string `json:"source_status"`
	SourceTarget                                                                                                                                                                                                                                             string `json:"source_target"`
	SourceReport                                                                                                                                                                                                                                             string `json:"source_report"`
	SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReceiverObservationBoundaryPreflightGateCandidateStoreReaderProofID       string `json:"source_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_final_gate_receiver_observation_boundary_preflight_gate_candidate_store_reader_proof_id"`
	SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReceiverObservationBoundaryPreflightGateCandidateStoreReaderProofReady    bool   `json:"source_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_final_gate_receiver_observation_boundary_preflight_gate_candidate_store_reader_proof_ready"`
	SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReceiverObservationBoundaryPreflightGateCandidateStoreReaderProofCausalID string `json:"source_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_final_gate_receiver_observation_boundary_preflight_gate_candidate_store_reader_proof_causal_id"`
	SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReceiverObservationBoundaryPreflightGateCandidateStoreReaderProofHash     string `json:"source_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_final_gate_receiver_observation_boundary_preflight_gate_candidate_store_reader_proof_hash"`
	SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReceiverObservationBoundaryPreflightGateCandidateStoreReaderProofReadBack string `json:"source_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_final_gate_receiver_observation_boundary_preflight_gate_candidate_store_reader_proof_read_back_hash"`
	SourceProofAction                                                                                                                                                                                                                                        string `json:"source_proof_action"`
	SourceProofReceiptShape                                                                                                                                                                                                                                  string `json:"source_proof_receipt_shape"`
	SourceProofKind                                                                                                                                                                                                                                          string `json:"source_proof_kind"`
	SourceProofMode                                                                                                                                                                                                                                          string `json:"source_proof_mode"`
	SourceProofStage                                                                                                                                                                                                                                         string `json:"source_proof_stage"`
	SourceProofLedgerReady                                                                                                                                                                                                                                   bool   `json:"source_proof_ledger_ready"`
	SourceProofLedgerAppendAllowed                                                                                                                                                                                                                           bool   `json:"source_proof_ledger_append_allowed"`
	SourceProofAdmissionRequired                                                                                                                                                                                                                             bool   `json:"source_proof_admission_required"`
	SourceProofShadowOnly                                                                                                                                                                                                                                    bool   `json:"source_proof_shadow_only"`
	SourceProofGraftAllowed                                                                                                                                                                                                                                  bool   `json:"source_proof_graft_allowed"`
	SourceProofDryRunOnly                                                                                                                                                                                                                                    bool   `json:"source_proof_dry_run_only"`
	SourceProofLiveReady                                                                                                                                                                                                                                     bool   `json:"source_proof_live_ready"`
	SourceProofRawDreamTextAllowed                                                                                                                                                                                                                           bool   `json:"source_proof_raw_dream_text_allowed"`
	SourceProofRawDreamTextObserved                                                                                                                                                                                                                          bool   `json:"source_proof_raw_dream_text_observed"`
	SourceProofRawDreamTextForwarded                                                                                                                                                                                                                         bool   `json:"source_proof_raw_dream_text_forwarded"`
	SourceProofJanusSurfaceAllowed                                                                                                                                                                                                                           bool   `json:"source_proof_janus_surface_allowed"`
	SourceProofCoocLearningAllowed                                                                                                                                                                                                                           bool   `json:"source_proof_cooc_learning_allowed"`
	SourceProofDeltaHarvestAllowed                                                                                                                                                                                                                           bool   `json:"source_proof_delta_harvest_allowed"`
	SourceProofBodyMutationAllowed                                                                                                                                                                                                                           bool   `json:"source_proof_body_mutation_allowed"`
	SourceProofRollbackRequired                                                                                                                                                                                                                              bool   `json:"source_proof_rollback_required"`
	SourceProofReadOnly                                                                                                                                                                                                                                      bool   `json:"source_proof_read_only"`
	SourceProofReplayOnly                                                                                                                                                                                                                                    bool   `json:"source_proof_replay_only"`
	SourceProofAuthorityGranted                                                                                                                                                                                                                              bool   `json:"source_proof_authority_granted"`
	SourceProofContractsReady                                                                                                                                                                                                                                bool   `json:"source_proof_contracts_ready"`
	SourceProofWriteAllowed                                                                                                                                                                                                                                  bool   `json:"source_proof_write_allowed"`
	SourceProofAdmissionAllowed                                                                                                                                                                                                                              bool   `json:"source_proof_admission_allowed"`
	SourceProofLiveAdmissionEnabled                                                                                                                                                                                                                          bool   `json:"source_proof_live_admission_enabled"`
	SourceProofMutatesState                                                                                                                                                                                                                                  bool   `json:"source_proof_mutates_state"`
	SourceProofBodyTarget                                                                                                                                                                                                                                    string `json:"source_proof_body_target"`
	SourceProofPassed                                                                                                                                                                                                                                        bool   `json:"source_proof_passed"`
	SourceProofReason                                                                                                                                                                                                                                        string `json:"source_proof_reason"`

	SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReceiverObservationBoundaryPreflightGateCandidateStoreReaderID       string `json:"source_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_final_gate_receiver_observation_boundary_preflight_gate_candidate_store_reader_id"`
	SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReceiverObservationBoundaryPreflightGateCandidateStoreReaderReady    bool   `json:"source_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_final_gate_receiver_observation_boundary_preflight_gate_candidate_store_reader_ready"`
	SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReceiverObservationBoundaryPreflightGateCandidateStoreReaderHash     string `json:"source_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_final_gate_receiver_observation_boundary_preflight_gate_candidate_store_reader_hash"`
	SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReceiverObservationBoundaryPreflightGateCandidateStoreReaderReadBack string `json:"source_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_final_gate_receiver_observation_boundary_preflight_gate_candidate_store_reader_read_back_hash"`
	SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReceiverObservationBoundaryPreflightGateCandidateStoreID             string `json:"source_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_final_gate_receiver_observation_boundary_preflight_gate_candidate_store_id"`
	SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReceiverObservationBoundaryPreflightGateCandidateStoreReady          bool   `json:"source_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_final_gate_receiver_observation_boundary_preflight_gate_candidate_store_ready"`
	SourceAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreHash                                                                                                                                                                          string `json:"source_admission_final_gate_observation_boundary_preflight_gate_candidate_store_hash"`
	SourceAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReadBackHash                                                                                                                                                                  string `json:"source_admission_final_gate_observation_boundary_preflight_gate_candidate_store_read_back_hash"`
	SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReceiverObservationBoundaryPreflightGateCandidateID                  string `json:"source_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_final_gate_receiver_observation_boundary_preflight_gate_candidate_id"`
	SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReceiverObservationBoundaryPreflightGateCandidateReady               bool   `json:"source_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_final_gate_receiver_observation_boundary_preflight_gate_candidate_ready"`
	SourceAdmissionFinalGateObservationBoundaryPreflightGateCandidateHash                                                                                                                                                                               string `json:"source_admission_final_gate_observation_boundary_preflight_gate_candidate_hash"`
	SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReceiverObservationBoundaryPreflightGateID                           string `json:"source_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_final_gate_receiver_observation_boundary_preflight_gate_id"`
	SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReceiverObservationBoundaryPreflightGateReady                        bool   `json:"source_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_final_gate_receiver_observation_boundary_preflight_gate_ready"`
	SourceAdmissionFinalGateObservationBoundaryPreflightGateHash                                                                                                                                                                                        string `json:"source_admission_final_gate_observation_boundary_preflight_gate_hash"`
	SourceAdmissionFinalGateObservationBoundaryPreflightGateReady                                                                                                                                                                                       bool   `json:"source_admission_final_gate_observation_boundary_preflight_gate_ready"`
	SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReceiverObservationBoundaryPreflightID                               string `json:"source_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_final_gate_receiver_observation_boundary_preflight_id"`
	SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReceiverObservationBoundaryPreflightReady                            bool   `json:"source_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_final_gate_receiver_observation_boundary_preflight_ready"`
	SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReceiverObservationBoundaryID                                        string `json:"source_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_final_gate_receiver_observation_boundary_id"`
	SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReceiverObservationBoundaryReady                                     bool   `json:"source_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_final_gate_receiver_observation_boundary_ready"`
	SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReceiverObservationID                                                string `json:"source_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_final_gate_receiver_observation_id"`
	SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReceiverObservationReady                                             bool   `json:"source_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_final_gate_receiver_observation_ready"`
	SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReceiverID                                                           string `json:"source_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_final_gate_receiver_id"`
	SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReceiverReady                                                        bool   `json:"source_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_final_gate_receiver_ready"`
	SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateIntentReady                                                          bool   `json:"source_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_final_gate_intent_ready"`
	SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReady                                                                bool   `json:"source_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_final_gate_ready"`
	SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageSealReady                                                                     bool   `json:"source_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_seal_ready"`
	SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageAuthorityReady                                                                bool   `json:"source_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_authority_ready"`
	SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStagePermitReady                                                                   bool   `json:"source_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_permit_ready"`
	SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageReadinessReady                                                                bool   `json:"source_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_readiness_ready"`
	SourceWriterInventoryVerified                                                                                                                                                                                                                       bool   `json:"source_writer_inventory_verified"`
	SourceWriterPreflightVerified                                                                                                                                                                                                                       bool   `json:"source_writer_preflight_verified"`
	SourceAdmissionRequired                                                                                                                                                                                                                             bool   `json:"source_admission_required"`
	SourceShadowOnly                                                                                                                                                                                                                                    bool   `json:"source_shadow_only"`
	SourceDryRunOnly                                                                                                                                                                                                                                    bool   `json:"source_dry_run_only"`
	SourceRequiresWriter                                                                                                                                                                                                                                bool   `json:"source_requires_writer"`
	SourceRollbackRequired                                                                                                                                                                                                                              bool   `json:"source_rollback_required"`
	SourceRequiresRollback                                                                                                                                                                                                                              bool   `json:"source_requires_rollback"`
	SourceReadOnly                                                                                                                                                                                                                                      bool   `json:"source_read_only"`
	SourceReplayOnly                                                                                                                                                                                                                                    bool   `json:"source_replay_only"`
}

func runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReceiverObservationBoundaryPreflightGateCandidateStoreReaderProofPrecondition(args []string) error {
	if len(args) != 2 {
		return fmt.Errorf("usage: --admission-live-route-weighted-admission-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-candidate-store-reader-proof-precondition-decision-promotion-switch-enable-gate-live-stage-final-gate-receiver-observation-boundary-preflight-gate-candidate-store-reader-proof-precondition RESONANCE_GRAFT_ADMISSION_FINAL_GATE_OBSERVATION_BOUNDARY_PREFLIGHT_GATE_CANDIDATE_STORE_READER_PROOF_REPORT RESONANCE_GRAFT_ADMISSION_FINAL_GATE_OBSERVATION_BOUNDARY_PREFLIGHT_GATE_CANDIDATE_STORE_READER_PROOF_PRECONDITION_REPORT")
	}
	proofPath := args[0]
	outputPath := args[1]
	if strings.TrimSpace(outputPath) == "" {
		return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage final gate receiver observation boundary preflight gate candidate store reader proof precondition output path missing")
	}
	proof, root, err := readAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReceiverObservationBoundaryPreflightGateCandidateStoreReaderProofReportForAssert(proofPath)
	if err != nil {
		return err
	}
	if err := admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReceiverObservationBoundaryPreflightGateCandidateStoreReaderProofReportError(proof, root); err != nil {
		return err
	}
	precondition := admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReceiverObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionReport{
		Schema:              admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReceiverObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionSchema,
		Status:              "shadow_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_satisfied_dry_run",
		Target:              "live_route_admission_next_step",
		TargetKind:          "weighted_internal_world_shadow_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition",
		TargetMode:          "closed_receipt_precondition_dry_run",
		Action:              "consume_weighted_resonance_shadow_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_before_live_route_admission",
		LedgerState:         "blocked",
		LedgerAction:        "reject_blocked_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_ledger_append",
		LedgerContract:      "none",
		LedgerEntrypoint:    "none",
		LedgerReceiptShape:  "none",
		LedgerWriteScope:    "none",
		LedgerReady:         false,
		LedgerAppendAllowed: false,
		WeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReceiverObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionReady: true,
		WeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReceiverObservationBoundaryPreflightGateCandidateStoreReaderProofConsumed:          true,
		WeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReceiverObservationBoundaryPreflightGateCandidateStoreReaderProofRequired:          true,
		NextStepBlockedWithoutResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPrecondition:                                                                                                                                 true,
		ReceiptShape:           "weighted_resonance_shadow_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_receipt",
		PreconditionKind:       "shadow_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition",
		PreconditionMode:       "closed_receipt_consumption",
		PreconditionStage:      "post_preflight_gate_candidate_store_reader_proof_pre_live_admission_precondition",
		ProofVerified:          true,
		ProofHashVerified:      true,
		ProofReadBackVerified:  true,
		StoreReaderVerified:    proof.StoreReaderVerified,
		StoreVerified:          proof.StoreVerified,
		CandidateVerified:      proof.CandidateVerified,
		GateVerified:           proof.GateVerified,
		PreflightVerified:      proof.PreflightVerified,
		BoundaryVerified:       proof.BoundaryVerified,
		ObservationVerified:    proof.ObservationVerified,
		ReceiverVerified:       proof.ReceiverVerified,
		IntentVerified:         proof.IntentVerified,
		FinalGateVerified:      proof.FinalGateVerified,
		SealVerified:           proof.SealVerified,
		PermitVerified:         proof.PermitVerified,
		AuthorityVerified:      proof.AuthorityVerified,
		ReaderHashVerified:     proof.ReaderHashVerified,
		ReaderReplayVerified:   proof.ReaderReplayVerified,
		ReaderReadBackVerified: proof.ReaderReadBackVerified,
		StoreHashVerified:      proof.StoreHashVerified,
		StoreReadBackVerified:  proof.StoreReadBackVerified,
		AdmissionRequired:      true,
		ShadowOnly:             true,
		GraftAllowed:           false,
		DryRunOnly:             true,
		LiveReady:              proof.LiveReady,
		RawDreamTextAllowed:    false,
		RawDreamTextObserved:   false,
		RawDreamTextForwarded:  false,
		JanusSurfaceAllowed:    false,
		CoocLearningAllowed:    false,
		DeltaHarvestAllowed:    false,
		BodyMutationAllowed:    false,
		RollbackRequired:       true,
		ReadOnly:               true,
		ReplayOnly:             true,
		AuthorityGranted:       false,
		ContractsReady:         false,
		WriteAllowed:           false,
		AdmissionAllowed:       false,
		LiveAdmissionEnabled:   false,
		MutatesState:           false,
		BodyTarget:             "none",
		Passed:                 true,
		Reason:                 "weighted resonance shadow graft admission final gate observation boundary preflight gate candidate store reader proof consumed as closed precondition",

		SourceSchema: proof.Schema,
		SourceStatus: proof.Status,
		SourceTarget: proof.Target,
		SourceReport: proofPath,
		SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReceiverObservationBoundaryPreflightGateCandidateStoreReaderProofID:       proof.WeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReceiverObservationBoundaryPreflightGateCandidateStoreReaderProofID,
		SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReceiverObservationBoundaryPreflightGateCandidateStoreReaderProofReady:    proof.WeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReceiverObservationBoundaryPreflightGateCandidateStoreReaderProofReady,
		SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReceiverObservationBoundaryPreflightGateCandidateStoreReaderProofCausalID: proof.CausalID,
		SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReceiverObservationBoundaryPreflightGateCandidateStoreReaderProofHash:     proof.ProofHash,
		SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReceiverObservationBoundaryPreflightGateCandidateStoreReaderProofReadBack: proof.ReadBackHash,
		SourceProofAction:                proof.Action,
		SourceProofReceiptShape:          proof.ReceiptShape,
		SourceProofKind:                  proof.ProofKind,
		SourceProofMode:                  proof.ProofMode,
		SourceProofStage:                 proof.ProofStage,
		SourceProofLedgerReady:           proof.LedgerReady,
		SourceProofLedgerAppendAllowed:   proof.LedgerAppendAllowed,
		SourceProofAdmissionRequired:     proof.AdmissionRequired,
		SourceProofShadowOnly:            proof.ShadowOnly,
		SourceProofGraftAllowed:          proof.GraftAllowed,
		SourceProofDryRunOnly:            proof.DryRunOnly,
		SourceProofLiveReady:             proof.LiveReady,
		SourceProofRawDreamTextAllowed:   proof.RawDreamTextAllowed,
		SourceProofRawDreamTextObserved:  proof.RawDreamTextObserved,
		SourceProofRawDreamTextForwarded: proof.RawDreamTextForwarded,
		SourceProofJanusSurfaceAllowed:   proof.JanusSurfaceAllowed,
		SourceProofCoocLearningAllowed:   proof.CoocLearningAllowed,
		SourceProofDeltaHarvestAllowed:   proof.DeltaHarvestAllowed,
		SourceProofBodyMutationAllowed:   proof.BodyMutationAllowed,
		SourceProofRollbackRequired:      proof.RollbackRequired,
		SourceProofReadOnly:              proof.ReadOnly,
		SourceProofReplayOnly:            proof.ReplayOnly,
		SourceProofAuthorityGranted:      proof.AuthorityGranted,
		SourceProofContractsReady:        proof.ContractsReady,
		SourceProofWriteAllowed:          proof.WriteAllowed,
		SourceProofAdmissionAllowed:      proof.AdmissionAllowed,
		SourceProofLiveAdmissionEnabled:  proof.LiveAdmissionEnabled,
		SourceProofMutatesState:          proof.MutatesState,
		SourceProofBodyTarget:            proof.BodyTarget,
		SourceProofPassed:                proof.Passed,
		SourceProofReason:                proof.Reason,

		SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReceiverObservationBoundaryPreflightGateCandidateStoreReaderID:       proof.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReceiverObservationBoundaryPreflightGateCandidateStoreReaderID,
		SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReceiverObservationBoundaryPreflightGateCandidateStoreReaderReady:    proof.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReceiverObservationBoundaryPreflightGateCandidateStoreReaderReady,
		SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReceiverObservationBoundaryPreflightGateCandidateStoreReaderHash:     proof.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReceiverObservationBoundaryPreflightGateCandidateStoreReaderHash,
		SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReceiverObservationBoundaryPreflightGateCandidateStoreReaderReadBack: proof.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReceiverObservationBoundaryPreflightGateCandidateStoreReaderReadBackHash,
		SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReceiverObservationBoundaryPreflightGateCandidateStoreID:             proof.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReceiverObservationBoundaryPreflightGateCandidateStoreID,
		SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReceiverObservationBoundaryPreflightGateCandidateStoreReady:          proof.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReceiverObservationBoundaryPreflightGateCandidateStoreReady,
		SourceAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreHash:         proof.SourceAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreHash,
		SourceAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReadBackHash: proof.SourceAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReadBackHash,
		SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReceiverObservationBoundaryPreflightGateCandidateID:    proof.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReceiverObservationBoundaryPreflightGateCandidateID,
		SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReceiverObservationBoundaryPreflightGateCandidateReady: proof.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReceiverObservationBoundaryPreflightGateCandidateReady,
		SourceAdmissionFinalGateObservationBoundaryPreflightGateCandidateHash: proof.SourceAdmissionFinalGateObservationBoundaryPreflightGateCandidateHash,
		SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReceiverObservationBoundaryPreflightGateID:    proof.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReceiverObservationBoundaryPreflightGateID,
		SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReceiverObservationBoundaryPreflightGateReady: proof.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReceiverObservationBoundaryPreflightGateReady,
		SourceAdmissionFinalGateObservationBoundaryPreflightGateHash:  proof.SourceAdmissionFinalGateObservationBoundaryPreflightGateHash,
		SourceAdmissionFinalGateObservationBoundaryPreflightGateReady: proof.SourceAdmissionFinalGateObservationBoundaryPreflightGateReady,
		SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReceiverObservationBoundaryPreflightID:    proof.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReceiverObservationBoundaryPreflightID,
		SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReceiverObservationBoundaryPreflightReady: proof.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReceiverObservationBoundaryPreflightReady,
		SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReceiverObservationBoundaryID:             proof.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReceiverObservationBoundaryID,
		SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReceiverObservationBoundaryReady:          proof.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReceiverObservationBoundaryReady,
		SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReceiverObservationID:                     proof.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReceiverObservationID,
		SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReceiverObservationReady:                  proof.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReceiverObservationReady,
		SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReceiverID:                                proof.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReceiverID,
		SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReceiverReady:                             proof.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReceiverReady,
		SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateIntentReady:                               proof.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateIntentReady,
		SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReady:                                     proof.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReady,
		SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageSealReady:                                          proof.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageSealReady,
		SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageAuthorityReady:                                     proof.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageAuthorityReady,
		SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStagePermitReady:                                        proof.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStagePermitReady,
		SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageReadinessReady:                                     proof.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageReadinessReady,
		SourceWriterInventoryVerified: proof.SourceWriterInventoryVerified,
		SourceWriterPreflightVerified: proof.SourceWriterPreflightVerified,
		SourceAdmissionRequired:       proof.SourceAdmissionRequired,
		SourceShadowOnly:              proof.SourceShadowOnly,
		SourceDryRunOnly:              proof.SourceDryRunOnly,
		SourceRequiresWriter:          proof.SourceRequiresWriter,
		SourceRollbackRequired:        proof.SourceRollbackRequired,
		SourceRequiresRollback:        proof.SourceRequiresRollback,
		SourceReadOnly:                proof.SourceReadOnly,
		SourceReplayOnly:              proof.SourceReplayOnly,
	}
	precondition.CausalID = admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReceiverObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionCausalID(precondition)
	precondition.PreconditionHash = admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReceiverObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionHash(precondition)
	precondition.ReadBackHash = admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReceiverObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionReadBackHash(precondition)
	precondition.WeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReceiverObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionID = admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReceiverObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionID(precondition)
	if precondition.CausalID == "" ||
		precondition.PreconditionHash == "" ||
		precondition.ReadBackHash == "" ||
		precondition.WeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReceiverObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionID == "" ||
		precondition.PreconditionHash == precondition.ReadBackHash {
		return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage final gate receiver observation boundary preflight gate candidate store reader proof precondition read-back proof failed")
	}
	raw, err := json.MarshalIndent(precondition, "", "  ")
	if err != nil {
		return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage final gate receiver observation boundary preflight gate candidate store reader proof precondition marshal failed: %w", err)
	}
	raw = append(raw, '\n')
	if err := os.WriteFile(outputPath, raw, 0600); err != nil {
		return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage final gate receiver observation boundary preflight gate candidate store reader proof precondition write failed: %w", err)
	}
	fmt.Printf("[admission-live-route-weighted-admission-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-candidate-store-reader-proof-precondition-decision-promotion-switch-enable-gate-live-stage-final-gate-receiver-observation-boundary-preflight-gate-candidate-store-reader-proof-precondition] pass: resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_report=%s resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_report=%s\n", outputPath, proofPath)
	return nil
}

func runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReceiverObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionAssert(args []string) error {
	if len(args) != 1 {
		return fmt.Errorf("usage: --admission-live-route-weighted-admission-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-candidate-store-reader-proof-precondition-decision-promotion-switch-enable-gate-live-stage-final-gate-receiver-observation-boundary-preflight-gate-candidate-store-reader-proof-precondition-assert REPORT")
	}
	report, root, err := readAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReceiverObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionReportForAssert(args[0])
	if err != nil {
		return err
	}
	return admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReceiverObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionReportError(report, root)
}

func admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReceiverObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionReportError(report admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReceiverObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionReport, root map[string]json.RawMessage) error {
	if _, ok := root["schema"]; !ok {
		return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage final gate receiver observation boundary preflight gate candidate store reader proof precondition schema missing")
	}
	if report.Schema != admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReceiverObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionSchema {
		return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage final gate receiver observation boundary preflight gate candidate store reader proof precondition schema mismatch: got %q want %q", report.Schema, admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReceiverObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionSchema)
	}
	if report.Status != "shadow_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_satisfied_dry_run" {
		return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage final gate receiver observation boundary preflight gate candidate store reader proof precondition status mismatch: got %q want %q", report.Status, "shadow_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_satisfied_dry_run")
	}
	if report.Target != "live_route_admission_next_step" ||
		report.TargetKind != "weighted_internal_world_shadow_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition" ||
		report.TargetMode != "closed_receipt_precondition_dry_run" ||
		report.Action != "consume_weighted_resonance_shadow_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_before_live_route_admission" {
		return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage final gate receiver observation boundary preflight gate candidate store reader proof precondition route shape mismatch")
	}
	if report.LedgerState != "blocked" ||
		report.LedgerAction != "reject_blocked_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_ledger_append" ||
		report.LedgerContract != "none" ||
		report.LedgerEntrypoint != "none" ||
		report.LedgerReceiptShape != "none" ||
		report.LedgerWriteScope != "none" {
		return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage final gate receiver observation boundary preflight gate candidate store reader proof precondition ledger guard mismatch")
	}
	if report.ReceiptShape != "weighted_resonance_shadow_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_receipt" ||
		report.PreconditionKind != "shadow_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition" ||
		report.PreconditionMode != "closed_receipt_consumption" ||
		report.PreconditionStage != "post_preflight_gate_candidate_store_reader_proof_pre_live_admission_precondition" {
		return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage final gate receiver observation boundary preflight gate candidate store reader proof precondition shape mismatch")
	}
	for _, required := range []struct {
		name  string
		value bool
	}{
		{"weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_final_gate_receiver_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_ready", report.WeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReceiverObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionReady},
		{"weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_final_gate_receiver_observation_boundary_preflight_gate_candidate_store_reader_proof_consumed", report.WeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReceiverObservationBoundaryPreflightGateCandidateStoreReaderProofConsumed},
		{"weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_final_gate_receiver_observation_boundary_preflight_gate_candidate_store_reader_proof_required", report.WeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReceiverObservationBoundaryPreflightGateCandidateStoreReaderProofRequired},
		{"next_step_blocked_without_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition", report.NextStepBlockedWithoutResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPrecondition},
		{"proof_verified", report.ProofVerified},
		{"proof_hash_verified", report.ProofHashVerified},
		{"proof_read_back_verified", report.ProofReadBackVerified},
		{"store_reader_verified", report.StoreReaderVerified},
		{"store_verified", report.StoreVerified},
		{"candidate_verified", report.CandidateVerified},
		{"gate_verified", report.GateVerified},
		{"preflight_verified", report.PreflightVerified},
		{"boundary_verified", report.BoundaryVerified},
		{"observation_verified", report.ObservationVerified},
		{"receiver_verified", report.ReceiverVerified},
		{"intent_verified", report.IntentVerified},
		{"final_gate_verified", report.FinalGateVerified},
		{"seal_verified", report.SealVerified},
		{"permit_verified", report.PermitVerified},
		{"authority_verified", report.AuthorityVerified},
		{"reader_hash_verified", report.ReaderHashVerified},
		{"reader_replay_verified", report.ReaderReplayVerified},
		{"reader_read_back_verified", report.ReaderReadBackVerified},
		{"store_hash_verified", report.StoreHashVerified},
		{"store_read_back_verified", report.StoreReadBackVerified},
		{"admission_required", report.AdmissionRequired},
		{"shadow_only", report.ShadowOnly},
		{"dry_run_only", report.DryRunOnly},
		{"live_ready", report.LiveReady},
		{"rollback_required", report.RollbackRequired},
		{"read_only", report.ReadOnly},
		{"replay_only", report.ReplayOnly},
		{"source_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_final_gate_receiver_observation_boundary_preflight_gate_candidate_store_reader_proof_ready", report.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReceiverObservationBoundaryPreflightGateCandidateStoreReaderProofReady},
		{"source_proof_admission_required", report.SourceProofAdmissionRequired},
		{"source_proof_shadow_only", report.SourceProofShadowOnly},
		{"source_proof_dry_run_only", report.SourceProofDryRunOnly},
		{"source_proof_live_ready", report.SourceProofLiveReady},
		{"source_proof_rollback_required", report.SourceProofRollbackRequired},
		{"source_proof_read_only", report.SourceProofReadOnly},
		{"source_proof_replay_only", report.SourceProofReplayOnly},
		{"source_proof_passed", report.SourceProofPassed},
		{"source_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_final_gate_receiver_observation_boundary_preflight_gate_candidate_store_reader_ready", report.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReceiverObservationBoundaryPreflightGateCandidateStoreReaderReady},
		{"source_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_final_gate_receiver_observation_boundary_preflight_gate_candidate_store_ready", report.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReceiverObservationBoundaryPreflightGateCandidateStoreReady},
		{"source_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_final_gate_receiver_observation_boundary_preflight_gate_candidate_ready", report.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReceiverObservationBoundaryPreflightGateCandidateReady},
		{"source_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_final_gate_receiver_observation_boundary_preflight_gate_ready", report.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReceiverObservationBoundaryPreflightGateReady},
		{"source_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_final_gate_receiver_observation_boundary_preflight_ready", report.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReceiverObservationBoundaryPreflightReady},
		{"source_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_final_gate_receiver_observation_boundary_ready", report.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReceiverObservationBoundaryReady},
		{"source_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_final_gate_receiver_observation_ready", report.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReceiverObservationReady},
		{"source_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_final_gate_receiver_ready", report.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReceiverReady},
		{"source_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_final_gate_intent_ready", report.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateIntentReady},
		{"source_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_final_gate_ready", report.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReady},
		{"source_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_seal_ready", report.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageSealReady},
		{"source_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_authority_ready", report.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageAuthorityReady},
		{"source_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_permit_ready", report.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStagePermitReady},
		{"source_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_readiness_ready", report.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageReadinessReady},
		{"source_writer_inventory_verified", report.SourceWriterInventoryVerified},
		{"source_writer_preflight_verified", report.SourceWriterPreflightVerified},
		{"source_admission_required", report.SourceAdmissionRequired},
		{"source_shadow_only", report.SourceShadowOnly},
		{"source_dry_run_only", report.SourceDryRunOnly},
		{"source_requires_writer", report.SourceRequiresWriter},
		{"source_rollback_required", report.SourceRollbackRequired},
		{"source_requires_rollback", report.SourceRequiresRollback},
		{"source_read_only", report.SourceReadOnly},
		{"source_replay_only", report.SourceReplayOnly},
		{"passed", report.Passed},
	} {
		if !required.value {
			return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage final gate receiver observation boundary preflight gate candidate store reader proof precondition %s not ready", required.name)
		}
	}
	for _, closed := range []struct {
		name  string
		value bool
	}{
		{"ledger_ready", report.LedgerReady},
		{"ledger_append_allowed", report.LedgerAppendAllowed},
		{"graft_allowed", report.GraftAllowed},
		{"raw_dream_text_allowed", report.RawDreamTextAllowed},
		{"raw_dream_text_observed", report.RawDreamTextObserved},
		{"raw_dream_text_forwarded", report.RawDreamTextForwarded},
		{"janus_surface_allowed", report.JanusSurfaceAllowed},
		{"cooc_learning_allowed", report.CoocLearningAllowed},
		{"delta_harvest_allowed", report.DeltaHarvestAllowed},
		{"body_mutation_allowed", report.BodyMutationAllowed},
		{"authority_granted", report.AuthorityGranted},
		{"contracts_ready", report.ContractsReady},
		{"write_allowed", report.WriteAllowed},
		{"admission_allowed", report.AdmissionAllowed},
		{"live_admission_enabled", report.LiveAdmissionEnabled},
		{"mutates_state", report.MutatesState},
		{"source_proof_ledger_ready", report.SourceProofLedgerReady},
		{"source_proof_ledger_append_allowed", report.SourceProofLedgerAppendAllowed},
		{"source_proof_graft_allowed", report.SourceProofGraftAllowed},
		{"source_proof_raw_dream_text_allowed", report.SourceProofRawDreamTextAllowed},
		{"source_proof_raw_dream_text_observed", report.SourceProofRawDreamTextObserved},
		{"source_proof_raw_dream_text_forwarded", report.SourceProofRawDreamTextForwarded},
		{"source_proof_janus_surface_allowed", report.SourceProofJanusSurfaceAllowed},
		{"source_proof_cooc_learning_allowed", report.SourceProofCoocLearningAllowed},
		{"source_proof_delta_harvest_allowed", report.SourceProofDeltaHarvestAllowed},
		{"source_proof_body_mutation_allowed", report.SourceProofBodyMutationAllowed},
		{"source_proof_authority_granted", report.SourceProofAuthorityGranted},
		{"source_proof_contracts_ready", report.SourceProofContractsReady},
		{"source_proof_write_allowed", report.SourceProofWriteAllowed},
		{"source_proof_admission_allowed", report.SourceProofAdmissionAllowed},
		{"source_proof_live_admission_enabled", report.SourceProofLiveAdmissionEnabled},
		{"source_proof_mutates_state", report.SourceProofMutatesState},
		{"source_admission_final_gate_observation_boundary_preflight_gate_ready", report.SourceAdmissionFinalGateObservationBoundaryPreflightGateReady},
	} {
		if closed.value {
			return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage final gate receiver observation boundary preflight gate candidate store reader proof precondition opened %s", closed.name)
		}
	}
	for _, requiredString := range []struct {
		name  string
		value string
	}{
		{"weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_final_gate_receiver_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_id", report.WeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReceiverObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionID},
		{"causal_id", report.CausalID},
		{"precondition_hash", report.PreconditionHash},
		{"read_back_hash", report.ReadBackHash},
		{"source_report", report.SourceReport},
		{"source_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_final_gate_receiver_observation_boundary_preflight_gate_candidate_store_reader_proof_id", report.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReceiverObservationBoundaryPreflightGateCandidateStoreReaderProofID},
		{"source_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_final_gate_receiver_observation_boundary_preflight_gate_candidate_store_reader_proof_causal_id", report.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReceiverObservationBoundaryPreflightGateCandidateStoreReaderProofCausalID},
		{"source_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_final_gate_receiver_observation_boundary_preflight_gate_candidate_store_reader_proof_hash", report.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReceiverObservationBoundaryPreflightGateCandidateStoreReaderProofHash},
		{"source_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_final_gate_receiver_observation_boundary_preflight_gate_candidate_store_reader_proof_read_back_hash", report.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReceiverObservationBoundaryPreflightGateCandidateStoreReaderProofReadBack},
		{"source_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_final_gate_receiver_observation_boundary_preflight_gate_candidate_store_reader_id", report.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReceiverObservationBoundaryPreflightGateCandidateStoreReaderID},
		{"source_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_final_gate_receiver_observation_boundary_preflight_gate_candidate_store_id", report.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReceiverObservationBoundaryPreflightGateCandidateStoreID},
		{"source_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_final_gate_receiver_observation_boundary_preflight_gate_candidate_id", report.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReceiverObservationBoundaryPreflightGateCandidateID},
		{"source_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_final_gate_receiver_observation_boundary_preflight_gate_id", report.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReceiverObservationBoundaryPreflightGateID},
		{"source_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_final_gate_receiver_observation_boundary_preflight_id", report.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReceiverObservationBoundaryPreflightID},
		{"source_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_final_gate_receiver_observation_boundary_id", report.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReceiverObservationBoundaryID},
		{"source_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_final_gate_receiver_observation_id", report.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReceiverObservationID},
		{"source_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_final_gate_receiver_id", report.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReceiverID},
	} {
		if strings.TrimSpace(requiredString.value) == "" {
			return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage final gate receiver observation boundary preflight gate candidate store reader proof precondition %s missing", requiredString.name)
		}
	}
	if report.SourceSchema != admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReceiverObservationBoundaryPreflightGateCandidateStoreReaderProofSchema {
		return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage final gate receiver observation boundary preflight gate candidate store reader proof precondition source_schema mismatch: got %q want %q", report.SourceSchema, admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReceiverObservationBoundaryPreflightGateCandidateStoreReaderProofSchema)
	}
	if report.SourceStatus != "shadow_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_ready_dry_run" ||
		report.SourceTarget != "live_route_admission_next_step" ||
		report.SourceProofAction != "prove_weighted_resonance_shadow_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_dry_run" {
		return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage final gate receiver observation boundary preflight gate candidate store reader proof precondition source proof route mismatch")
	}
	if report.SourceProofReceiptShape != "weighted_resonance_shadow_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_receipt" ||
		report.SourceProofKind != "shadow_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof" ||
		report.SourceProofMode != "closed_read_back_reader_proof" ||
		report.SourceProofStage != "post_preflight_gate_candidate_store_reader_pre_live_admission_proof" {
		return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage final gate receiver observation boundary preflight gate candidate store reader proof precondition source proof shape mismatch")
	}
	if report.BodyTarget != "none" || report.SourceProofBodyTarget != "none" {
		return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage final gate receiver observation boundary preflight gate candidate store reader proof precondition body target mismatch")
	}
	if !strings.HasPrefix(report.WeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReceiverObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionID, "weighted-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-candidate-store-reader-proof-precondition-decision-promotion-switch-enable-gate-live-stage-final-gate-receiver-observation-boundary-preflight-gate-candidate-store-reader-proof-precondition-id-") ||
		!strings.HasPrefix(report.CausalID, "weighted-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-candidate-store-reader-proof-precondition-decision-promotion-switch-enable-gate-live-stage-final-gate-receiver-observation-boundary-preflight-gate-candidate-store-reader-proof-precondition-causal-") ||
		!strings.HasPrefix(report.PreconditionHash, "weighted-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-candidate-store-reader-proof-precondition-decision-promotion-switch-enable-gate-live-stage-final-gate-receiver-observation-boundary-preflight-gate-candidate-store-reader-proof-precondition-") ||
		!strings.HasPrefix(report.ReadBackHash, "weighted-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-candidate-store-reader-proof-precondition-decision-promotion-switch-enable-gate-live-stage-final-gate-receiver-observation-boundary-preflight-gate-candidate-store-reader-proof-precondition-read-") ||
		report.PreconditionHash == report.ReadBackHash {
		return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage final gate receiver observation boundary preflight gate candidate store reader proof precondition prefix mismatch")
	}
	if !strings.HasPrefix(report.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReceiverObservationBoundaryPreflightGateCandidateStoreReaderProofID, "weighted-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-candidate-store-reader-proof-precondition-decision-promotion-switch-enable-gate-live-stage-final-gate-receiver-observation-boundary-preflight-gate-candidate-store-reader-proof-id-") ||
		!strings.HasPrefix(report.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReceiverObservationBoundaryPreflightGateCandidateStoreReaderProofCausalID, "weighted-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-candidate-store-reader-proof-precondition-decision-promotion-switch-enable-gate-live-stage-final-gate-receiver-observation-boundary-preflight-gate-candidate-store-reader-proof-causal-") ||
		!strings.HasPrefix(report.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReceiverObservationBoundaryPreflightGateCandidateStoreReaderProofHash, "weighted-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-candidate-store-reader-proof-precondition-decision-promotion-switch-enable-gate-live-stage-final-gate-receiver-observation-boundary-preflight-gate-candidate-store-reader-proof-") ||
		!strings.HasPrefix(report.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReceiverObservationBoundaryPreflightGateCandidateStoreReaderProofReadBack, "weighted-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-candidate-store-reader-proof-precondition-decision-promotion-switch-enable-gate-live-stage-final-gate-receiver-observation-boundary-preflight-gate-candidate-store-reader-proof-read-") {
		return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage final gate receiver observation boundary preflight gate candidate store reader proof precondition source proof mismatch")
	}
	if !strings.HasPrefix(report.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReceiverObservationBoundaryPreflightGateCandidateStoreReaderID, "weighted-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-candidate-store-reader-proof-precondition-decision-promotion-switch-enable-gate-live-stage-final-gate-receiver-observation-boundary-preflight-gate-candidate-store-reader-id-") ||
		!strings.HasPrefix(report.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReceiverObservationBoundaryPreflightGateCandidateStoreID, "weighted-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-candidate-store-reader-proof-precondition-decision-promotion-switch-enable-gate-live-stage-final-gate-receiver-observation-boundary-preflight-gate-candidate-store-id-") ||
		!strings.HasPrefix(report.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReceiverObservationBoundaryPreflightGateCandidateID, "weighted-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-candidate-store-reader-proof-precondition-decision-promotion-switch-enable-gate-live-stage-final-gate-receiver-observation-boundary-preflight-gate-candidate-id-") ||
		!strings.HasPrefix(report.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReceiverObservationBoundaryPreflightGateID, "weighted-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-candidate-store-reader-proof-precondition-decision-promotion-switch-enable-gate-live-stage-final-gate-receiver-observation-boundary-preflight-gate-id-") {
		return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage final gate receiver observation boundary preflight gate candidate store reader proof precondition source chain prefix mismatch")
	}
	if report.CausalID == "" || report.CausalID != admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReceiverObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionCausalID(report) {
		return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage final gate receiver observation boundary preflight gate candidate store reader proof precondition causal_id mismatch")
	}
	if report.PreconditionHash == "" || report.PreconditionHash != admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReceiverObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionHash(report) {
		return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage final gate receiver observation boundary preflight gate candidate store reader proof precondition precondition_hash mismatch")
	}
	if report.ReadBackHash == "" || report.ReadBackHash != admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReceiverObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionReadBackHash(report) {
		return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage final gate receiver observation boundary preflight gate candidate store reader proof precondition read_back_hash mismatch")
	}
	if report.PreconditionHash == report.ReadBackHash {
		return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage final gate receiver observation boundary preflight gate candidate store reader proof precondition read-back proof collapsed")
	}
	if report.WeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReceiverObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionID == "" ||
		report.WeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReceiverObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionID != admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReceiverObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionID(report) {
		return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage final gate receiver observation boundary preflight gate candidate store reader proof precondition id mismatch")
	}
	if report.Reason != "weighted resonance shadow graft admission final gate observation boundary preflight gate candidate store reader proof consumed as closed precondition" {
		return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage final gate receiver observation boundary preflight gate candidate store reader proof precondition reason mismatch: got %q", report.Reason)
	}
	return nil
}

func admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReceiverObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionCausalID(precondition admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReceiverObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionReport) string {
	h := hashJSON(struct {
		SourceProofID       string `json:"source_proof_id"`
		SourceProofReadBack string `json:"source_proof_read_back_hash"`
		SourceReaderID      string `json:"source_reader_id"`
		SourceStoreID       string `json:"source_store_id"`
		SourceCandidateID   string `json:"source_candidate_id"`
		SourceGateID        string `json:"source_gate_id"`
		Target              string `json:"target"`
		PreconditionKind    string `json:"precondition_kind"`
		PreconditionStage   string `json:"precondition_stage"`
	}{
		SourceProofID:       precondition.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReceiverObservationBoundaryPreflightGateCandidateStoreReaderProofID,
		SourceProofReadBack: precondition.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReceiverObservationBoundaryPreflightGateCandidateStoreReaderProofReadBack,
		SourceReaderID:      precondition.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReceiverObservationBoundaryPreflightGateCandidateStoreReaderID,
		SourceStoreID:       precondition.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReceiverObservationBoundaryPreflightGateCandidateStoreID,
		SourceCandidateID:   precondition.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReceiverObservationBoundaryPreflightGateCandidateID,
		SourceGateID:        precondition.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReceiverObservationBoundaryPreflightGateID,
		Target:              precondition.Target,
		PreconditionKind:    precondition.PreconditionKind,
		PreconditionStage:   precondition.PreconditionStage,
	})
	if h == "" {
		return ""
	}
	return "weighted-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-candidate-store-reader-proof-precondition-decision-promotion-switch-enable-gate-live-stage-final-gate-receiver-observation-boundary-preflight-gate-candidate-store-reader-proof-precondition-causal-" + h
}

func admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReceiverObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionHash(precondition admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReceiverObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionReport) string {
	h := hashJSON(struct {
		CausalID              string `json:"causal_id"`
		SourceProofID         string `json:"source_proof_id"`
		SourceProofHash       string `json:"source_proof_hash"`
		SourceProofReadBack   string `json:"source_proof_read_back_hash"`
		PreconditionMode      string `json:"precondition_mode"`
		ReceiptShape          string `json:"receipt_shape"`
		ProofVerified         bool   `json:"proof_verified"`
		ProofHashVerified     bool   `json:"proof_hash_verified"`
		ProofReadBackVerified bool   `json:"proof_read_back_verified"`
		ReadOnly              bool   `json:"read_only"`
		ReplayOnly            bool   `json:"replay_only"`
		AdmissionRequired     bool   `json:"admission_required"`
		ShadowOnly            bool   `json:"shadow_only"`
		DryRunOnly            bool   `json:"dry_run_only"`
		GraftAllowed          bool   `json:"graft_allowed"`
		LedgerAppendAllowed   bool   `json:"ledger_append_allowed"`
	}{
		CausalID:              precondition.CausalID,
		SourceProofID:         precondition.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReceiverObservationBoundaryPreflightGateCandidateStoreReaderProofID,
		SourceProofHash:       precondition.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReceiverObservationBoundaryPreflightGateCandidateStoreReaderProofHash,
		SourceProofReadBack:   precondition.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReceiverObservationBoundaryPreflightGateCandidateStoreReaderProofReadBack,
		PreconditionMode:      precondition.PreconditionMode,
		ReceiptShape:          precondition.ReceiptShape,
		ProofVerified:         precondition.ProofVerified,
		ProofHashVerified:     precondition.ProofHashVerified,
		ProofReadBackVerified: precondition.ProofReadBackVerified,
		ReadOnly:              precondition.ReadOnly,
		ReplayOnly:            precondition.ReplayOnly,
		AdmissionRequired:     precondition.AdmissionRequired,
		ShadowOnly:            precondition.ShadowOnly,
		DryRunOnly:            precondition.DryRunOnly,
		GraftAllowed:          precondition.GraftAllowed,
		LedgerAppendAllowed:   precondition.LedgerAppendAllowed,
	})
	if h == "" {
		return ""
	}
	return "weighted-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-candidate-store-reader-proof-precondition-decision-promotion-switch-enable-gate-live-stage-final-gate-receiver-observation-boundary-preflight-gate-candidate-store-reader-proof-precondition-" + h
}

func admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReceiverObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionReadBackHash(precondition admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReceiverObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionReport) string {
	h := hashJSON(struct {
		PreconditionHash  string `json:"precondition_hash"`
		SourceProofID     string `json:"source_proof_id"`
		SourceReaderID    string `json:"source_reader_id"`
		PreconditionKind  string `json:"precondition_kind"`
		PreconditionReady bool   `json:"precondition_ready"`
		BodyMutation      bool   `json:"body_mutation"`
		LiveAdmission     bool   `json:"live_admission"`
		WriteAllowed      bool   `json:"write_allowed"`
		AdmissionAllowed  bool   `json:"admission_allowed"`
		LedgerAppend      bool   `json:"ledger_append_allowed"`
	}{
		PreconditionHash:  precondition.PreconditionHash,
		SourceProofID:     precondition.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReceiverObservationBoundaryPreflightGateCandidateStoreReaderProofID,
		SourceReaderID:    precondition.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReceiverObservationBoundaryPreflightGateCandidateStoreReaderID,
		PreconditionKind:  precondition.PreconditionKind,
		PreconditionReady: precondition.WeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReceiverObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionReady,
		BodyMutation:      precondition.BodyMutationAllowed,
		LiveAdmission:     precondition.LiveAdmissionEnabled,
		WriteAllowed:      precondition.WriteAllowed,
		AdmissionAllowed:  precondition.AdmissionAllowed,
		LedgerAppend:      precondition.LedgerAppendAllowed,
	})
	if h == "" {
		return ""
	}
	return "weighted-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-candidate-store-reader-proof-precondition-decision-promotion-switch-enable-gate-live-stage-final-gate-receiver-observation-boundary-preflight-gate-candidate-store-reader-proof-precondition-read-" + h
}

func admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReceiverObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionID(precondition admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReceiverObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionReport) string {
	h := hashJSON(struct {
		Schema                 string `json:"schema"`
		Status                 string `json:"status"`
		Action                 string `json:"action"`
		SourceReport           string `json:"source_report"`
		SourceProofID          string `json:"source_proof_id"`
		SourceReaderID         string `json:"source_reader_id"`
		SourceStoreID          string `json:"source_store_id"`
		SourceCandidateID      string `json:"source_candidate_id"`
		SourceGateID           string `json:"source_gate_id"`
		CausalID               string `json:"causal_id"`
		PreconditionHash       string `json:"precondition_hash"`
		ReadBackHash           string `json:"read_back_hash"`
		Ready                  bool   `json:"ready"`
		ReceiptShape           string `json:"receipt_shape"`
		PreconditionKind       string `json:"precondition_kind"`
		PreconditionMode       string `json:"precondition_mode"`
		PreconditionStage      string `json:"precondition_stage"`
		ProofVerified          bool   `json:"proof_verified"`
		ProofHashVerified      bool   `json:"proof_hash_verified"`
		ProofReadBackVerified  bool   `json:"proof_read_back_verified"`
		AdmissionRequired      bool   `json:"admission_required"`
		ShadowOnly             bool   `json:"shadow_only"`
		GraftAllowed           bool   `json:"graft_allowed"`
		DryRunOnly             bool   `json:"dry_run_only"`
		ReadOnly               bool   `json:"read_only"`
		ReplayOnly             bool   `json:"replay_only"`
		LiveReady              bool   `json:"live_ready"`
		ContractsReady         bool   `json:"contracts_ready"`
		BodyTarget             string `json:"body_target"`
		WriteAllowed           bool   `json:"write_allowed"`
		AdmissionAllowed       bool   `json:"admission_allowed"`
		LiveAdmissionEnabled   bool   `json:"live_admission_enabled"`
		MutatesState           bool   `json:"mutates_state"`
		LedgerAppendAllowed    bool   `json:"ledger_append_allowed"`
		NextStepBlockedWithout bool   `json:"next_step_blocked_without_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition"`
	}{
		Schema:                 precondition.Schema,
		Status:                 precondition.Status,
		Action:                 precondition.Action,
		SourceReport:           precondition.SourceReport,
		SourceProofID:          precondition.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReceiverObservationBoundaryPreflightGateCandidateStoreReaderProofID,
		SourceReaderID:         precondition.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReceiverObservationBoundaryPreflightGateCandidateStoreReaderID,
		SourceStoreID:          precondition.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReceiverObservationBoundaryPreflightGateCandidateStoreID,
		SourceCandidateID:      precondition.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReceiverObservationBoundaryPreflightGateCandidateID,
		SourceGateID:           precondition.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReceiverObservationBoundaryPreflightGateID,
		CausalID:               precondition.CausalID,
		PreconditionHash:       precondition.PreconditionHash,
		ReadBackHash:           precondition.ReadBackHash,
		Ready:                  precondition.WeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReceiverObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionReady,
		ReceiptShape:           precondition.ReceiptShape,
		PreconditionKind:       precondition.PreconditionKind,
		PreconditionMode:       precondition.PreconditionMode,
		PreconditionStage:      precondition.PreconditionStage,
		ProofVerified:          precondition.ProofVerified,
		ProofHashVerified:      precondition.ProofHashVerified,
		ProofReadBackVerified:  precondition.ProofReadBackVerified,
		AdmissionRequired:      precondition.AdmissionRequired,
		ShadowOnly:             precondition.ShadowOnly,
		GraftAllowed:           precondition.GraftAllowed,
		DryRunOnly:             precondition.DryRunOnly,
		ReadOnly:               precondition.ReadOnly,
		ReplayOnly:             precondition.ReplayOnly,
		LiveReady:              precondition.LiveReady,
		ContractsReady:         precondition.ContractsReady,
		BodyTarget:             precondition.BodyTarget,
		WriteAllowed:           precondition.WriteAllowed,
		AdmissionAllowed:       precondition.AdmissionAllowed,
		LiveAdmissionEnabled:   precondition.LiveAdmissionEnabled,
		MutatesState:           precondition.MutatesState,
		LedgerAppendAllowed:    precondition.LedgerAppendAllowed,
		NextStepBlockedWithout: precondition.NextStepBlockedWithoutResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPrecondition,
	})
	if h == "" {
		return ""
	}
	return "weighted-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-candidate-store-reader-proof-precondition-decision-promotion-switch-enable-gate-live-stage-final-gate-receiver-observation-boundary-preflight-gate-candidate-store-reader-proof-precondition-id-" + h
}

func readAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReceiverObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionReportForAssert(path string) (admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReceiverObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionReport, map[string]json.RawMessage, error) {
	var report admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReceiverObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionReport
	if strings.TrimSpace(path) == "" {
		return report, nil, fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage final gate receiver observation boundary preflight gate candidate store reader proof precondition path missing")
	}
	raw, err := os.ReadFile(path)
	if err != nil {
		if os.IsNotExist(err) {
			return report, nil, fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage final gate receiver observation boundary preflight gate candidate store reader proof precondition not written")
		}
		return report, nil, err
	}
	if len(raw) == 0 {
		return report, nil, fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage final gate receiver observation boundary preflight gate candidate store reader proof precondition not written")
	}
	var root map[string]json.RawMessage
	if err := json.Unmarshal(raw, &root); err != nil {
		return report, nil, fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage final gate receiver observation boundary preflight gate candidate store reader proof precondition JSON invalid: %w", err)
	}
	if err := json.Unmarshal(raw, &report); err != nil {
		return report, nil, fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage final gate receiver observation boundary preflight gate candidate store reader proof precondition decode failed: %w", err)
	}
	return report, root, nil
}
