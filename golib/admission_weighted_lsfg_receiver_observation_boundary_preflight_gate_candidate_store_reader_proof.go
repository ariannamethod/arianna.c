package main

import (
	"encoding/json"
	"fmt"
	"os"
	"strings"
)

const admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReceiverObservationBoundaryPreflightGateCandidateStoreReaderProofSchema = "arianna.live_route_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_final_gate_receiver_observation_boundary_preflight_gate_candidate_store_reader_proof.v1"

type admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReceiverObservationBoundaryPreflightGateCandidateStoreReaderProofReport struct {
	Schema                                                                                                                                                                                                                                          string `json:"schema"`
	Status                                                                                                                                                                                                                                          string `json:"status"`
	Target                                                                                                                                                                                                                                          string `json:"target"`
	TargetKind                                                                                                                                                                                                                                      string `json:"target_kind"`
	TargetMode                                                                                                                                                                                                                                      string `json:"target_mode"`
	Action                                                                                                                                                                                                                                          string `json:"action"`
	LedgerState                                                                                                                                                                                                                                     string `json:"ledger_state"`
	LedgerAction                                                                                                                                                                                                                                    string `json:"ledger_action"`
	LedgerContract                                                                                                                                                                                                                                  string `json:"ledger_contract"`
	LedgerEntrypoint                                                                                                                                                                                                                                string `json:"ledger_entrypoint"`
	LedgerReceiptShape                                                                                                                                                                                                                              string `json:"ledger_receipt_shape"`
	LedgerWriteScope                                                                                                                                                                                                                                string `json:"ledger_write_scope"`
	LedgerReady                                                                                                                                                                                                                                     bool   `json:"ledger_ready"`
	LedgerAppendAllowed                                                                                                                                                                                                                             bool   `json:"ledger_append_allowed"`
	WeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReceiverObservationBoundaryPreflightGateCandidateStoreReaderProofReady bool   `json:"weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_final_gate_receiver_observation_boundary_preflight_gate_candidate_store_reader_proof_ready"`
	WeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReceiverObservationBoundaryPreflightGateCandidateStoreReaderConsumed   bool   `json:"weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_final_gate_receiver_observation_boundary_preflight_gate_candidate_store_reader_consumed"`
	WeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReceiverObservationBoundaryPreflightGateCandidateStoreReaderRequired   bool   `json:"weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_final_gate_receiver_observation_boundary_preflight_gate_candidate_store_reader_required"`
	NextStepBlockedWithoutResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProof                                                                                                                                 bool   `json:"next_step_blocked_without_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof"`
	WeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReceiverObservationBoundaryPreflightGateCandidateStoreReaderProofID    string `json:"weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_final_gate_receiver_observation_boundary_preflight_gate_candidate_store_reader_proof_id"`
	ReceiptShape                                                                                                                                                                                                                                    string `json:"receipt_shape"`
	ProofKind                                                                                                                                                                                                                                       string `json:"proof_kind"`
	ProofMode                                                                                                                                                                                                                                       string `json:"proof_mode"`
	ProofStage                                                                                                                                                                                                                                      string `json:"proof_stage"`
	CausalID                                                                                                                                                                                                                                        string `json:"causal_id"`
	ProofHash                                                                                                                                                                                                                                       string `json:"proof_hash"`
	ReadBackHash                                                                                                                                                                                                                                    string `json:"read_back_hash"`
	StoreReaderVerified                                                                                                                                                                                                                             bool   `json:"store_reader_verified"`
	StoreVerified                                                                                                                                                                                                                                   bool   `json:"store_verified"`
	CandidateVerified                                                                                                                                                                                                                               bool   `json:"candidate_verified"`
	GateVerified                                                                                                                                                                                                                                    bool   `json:"gate_verified"`
	PreflightVerified                                                                                                                                                                                                                               bool   `json:"preflight_verified"`
	BoundaryVerified                                                                                                                                                                                                                                bool   `json:"boundary_verified"`
	ObservationVerified                                                                                                                                                                                                                             bool   `json:"observation_verified"`
	ReceiverVerified                                                                                                                                                                                                                                bool   `json:"receiver_verified"`
	IntentVerified                                                                                                                                                                                                                                  bool   `json:"intent_verified"`
	FinalGateVerified                                                                                                                                                                                                                               bool   `json:"final_gate_verified"`
	SealVerified                                                                                                                                                                                                                                    bool   `json:"seal_verified"`
	PermitVerified                                                                                                                                                                                                                                  bool   `json:"permit_verified"`
	AuthorityVerified                                                                                                                                                                                                                               bool   `json:"authority_verified"`
	ReaderHashVerified                                                                                                                                                                                                                              bool   `json:"reader_hash_verified"`
	ReaderReplayVerified                                                                                                                                                                                                                            bool   `json:"reader_replay_verified"`
	ReaderReadBackVerified                                                                                                                                                                                                                          bool   `json:"reader_read_back_verified"`
	StoreHashVerified                                                                                                                                                                                                                               bool   `json:"store_hash_verified"`
	StoreReadBackVerified                                                                                                                                                                                                                           bool   `json:"store_read_back_verified"`
	AdmissionRequired                                                                                                                                                                                                                               bool   `json:"admission_required"`
	ShadowOnly                                                                                                                                                                                                                                      bool   `json:"shadow_only"`
	GraftAllowed                                                                                                                                                                                                                                    bool   `json:"graft_allowed"`
	DryRunOnly                                                                                                                                                                                                                                      bool   `json:"dry_run_only"`
	LiveReady                                                                                                                                                                                                                                       bool   `json:"live_ready"`
	RawDreamTextAllowed                                                                                                                                                                                                                             bool   `json:"raw_dream_text_allowed"`
	RawDreamTextObserved                                                                                                                                                                                                                            bool   `json:"raw_dream_text_observed"`
	RawDreamTextForwarded                                                                                                                                                                                                                           bool   `json:"raw_dream_text_forwarded"`
	JanusSurfaceAllowed                                                                                                                                                                                                                             bool   `json:"janus_surface_allowed"`
	CoocLearningAllowed                                                                                                                                                                                                                             bool   `json:"cooc_learning_allowed"`
	DeltaHarvestAllowed                                                                                                                                                                                                                             bool   `json:"delta_harvest_allowed"`
	BodyMutationAllowed                                                                                                                                                                                                                             bool   `json:"body_mutation_allowed"`
	RollbackRequired                                                                                                                                                                                                                                bool   `json:"rollback_required"`
	ReadOnly                                                                                                                                                                                                                                        bool   `json:"read_only"`
	ReplayOnly                                                                                                                                                                                                                                      bool   `json:"replay_only"`
	AuthorityGranted                                                                                                                                                                                                                                bool   `json:"authority_granted"`
	ContractsReady                                                                                                                                                                                                                                  bool   `json:"contracts_ready"`
	WriteAllowed                                                                                                                                                                                                                                    bool   `json:"write_allowed"`
	AdmissionAllowed                                                                                                                                                                                                                                bool   `json:"admission_allowed"`
	LiveAdmissionEnabled                                                                                                                                                                                                                            bool   `json:"live_admission_enabled"`
	MutatesState                                                                                                                                                                                                                                    bool   `json:"mutates_state"`
	BodyTarget                                                                                                                                                                                                                                      string `json:"body_target"`
	Passed                                                                                                                                                                                                                                          bool   `json:"passed"`
	Reason                                                                                                                                                                                                                                          string `json:"reason"`

	SourceSchema                                                                                                                                                                                                                                            string `json:"source_schema"`
	SourceStatus                                                                                                                                                                                                                                            string `json:"source_status"`
	SourceTarget                                                                                                                                                                                                                                            string `json:"source_target"`
	SourceReport                                                                                                                                                                                                                                            string `json:"source_report"`
	SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReceiverObservationBoundaryPreflightGateCandidateStoreReaderID           string `json:"source_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_final_gate_receiver_observation_boundary_preflight_gate_candidate_store_reader_id"`
	SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReceiverObservationBoundaryPreflightGateCandidateStoreReaderReady        bool   `json:"source_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_final_gate_receiver_observation_boundary_preflight_gate_candidate_store_reader_ready"`
	SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReceiverObservationBoundaryPreflightGateCandidateStoreReaderCausalID     string `json:"source_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_final_gate_receiver_observation_boundary_preflight_gate_candidate_store_reader_causal_id"`
	SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReceiverObservationBoundaryPreflightGateCandidateStoreReaderHash         string `json:"source_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_final_gate_receiver_observation_boundary_preflight_gate_candidate_store_reader_hash"`
	SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReceiverObservationBoundaryPreflightGateCandidateStoreReaderReplayHash   string `json:"source_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_final_gate_receiver_observation_boundary_preflight_gate_candidate_store_reader_replay_hash"`
	SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReceiverObservationBoundaryPreflightGateCandidateStoreReaderReadBackHash string `json:"source_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_final_gate_receiver_observation_boundary_preflight_gate_candidate_store_reader_read_back_hash"`
	SourceReaderAction                                                                                                                                                                                                                                      string `json:"source_reader_action"`
	SourceReaderReceiptShape                                                                                                                                                                                                                                string `json:"source_reader_receipt_shape"`
	SourceReaderKind                                                                                                                                                                                                                                        string `json:"source_reader_kind"`
	SourceReaderMode                                                                                                                                                                                                                                        string `json:"source_reader_mode"`
	SourceReaderStage                                                                                                                                                                                                                                       string `json:"source_reader_stage"`
	SourceReaderReadOnly                                                                                                                                                                                                                                    bool   `json:"source_reader_read_only"`
	SourceReaderReplayOnly                                                                                                                                                                                                                                  bool   `json:"source_reader_replay_only"`
	SourceReaderStoreVerified                                                                                                                                                                                                                               bool   `json:"source_reader_store_verified"`
	SourceReaderCandidateVerified                                                                                                                                                                                                                           bool   `json:"source_reader_candidate_verified"`
	SourceReaderGateVerified                                                                                                                                                                                                                                bool   `json:"source_reader_gate_verified"`
	SourceReaderPreflightVerified                                                                                                                                                                                                                           bool   `json:"source_reader_preflight_verified"`
	SourceReaderBoundaryVerified                                                                                                                                                                                                                            bool   `json:"source_reader_boundary_verified"`
	SourceReaderObservationVerified                                                                                                                                                                                                                         bool   `json:"source_reader_observation_verified"`
	SourceReaderFinalGateVerified                                                                                                                                                                                                                           bool   `json:"source_reader_final_gate_verified"`
	SourceReaderSealVerified                                                                                                                                                                                                                                bool   `json:"source_reader_seal_verified"`
	SourceReaderPermitVerified                                                                                                                                                                                                                              bool   `json:"source_reader_permit_verified"`
	SourceReaderAuthorityVerified                                                                                                                                                                                                                           bool   `json:"source_reader_authority_verified"`
	SourceReaderStoreHashVerified                                                                                                                                                                                                                           bool   `json:"source_reader_store_hash_verified"`
	SourceReaderStoreReadBackVerified                                                                                                                                                                                                                       bool   `json:"source_reader_store_read_back_verified"`
	SourceReaderAdmissionRequired                                                                                                                                                                                                                           bool   `json:"source_reader_admission_required"`
	SourceReaderShadowOnly                                                                                                                                                                                                                                  bool   `json:"source_reader_shadow_only"`
	SourceReaderDryRunOnly                                                                                                                                                                                                                                  bool   `json:"source_reader_dry_run_only"`
	SourceReaderLiveReady                                                                                                                                                                                                                                   bool   `json:"source_reader_live_ready"`
	SourceReaderRollbackRequired                                                                                                                                                                                                                            bool   `json:"source_reader_rollback_required"`
	SourceReaderLedgerReady                                                                                                                                                                                                                                 bool   `json:"source_reader_ledger_ready"`
	SourceReaderLedgerAppendAllowed                                                                                                                                                                                                                         bool   `json:"source_reader_ledger_append_allowed"`
	SourceReaderRawDreamTextAllowed                                                                                                                                                                                                                         bool   `json:"source_reader_raw_dream_text_allowed"`
	SourceReaderRawDreamTextObserved                                                                                                                                                                                                                        bool   `json:"source_reader_raw_dream_text_observed"`
	SourceReaderRawDreamTextForwarded                                                                                                                                                                                                                       bool   `json:"source_reader_raw_dream_text_forwarded"`
	SourceReaderJanusSurfaceAllowed                                                                                                                                                                                                                         bool   `json:"source_reader_janus_surface_allowed"`
	SourceReaderCoocLearningAllowed                                                                                                                                                                                                                         bool   `json:"source_reader_cooc_learning_allowed"`
	SourceReaderDeltaHarvestAllowed                                                                                                                                                                                                                         bool   `json:"source_reader_delta_harvest_allowed"`
	SourceReaderBodyMutationAllowed                                                                                                                                                                                                                         bool   `json:"source_reader_body_mutation_allowed"`
	SourceReaderAuthorityGranted                                                                                                                                                                                                                            bool   `json:"source_reader_authority_granted"`
	SourceReaderContractsReady                                                                                                                                                                                                                              bool   `json:"source_reader_contracts_ready"`
	SourceReaderWriteAllowed                                                                                                                                                                                                                                bool   `json:"source_reader_write_allowed"`
	SourceReaderAdmissionAllowed                                                                                                                                                                                                                            bool   `json:"source_reader_admission_allowed"`
	SourceReaderLiveAdmissionEnabled                                                                                                                                                                                                                        bool   `json:"source_reader_live_admission_enabled"`
	SourceReaderMutatesState                                                                                                                                                                                                                                bool   `json:"source_reader_mutates_state"`
	SourceReaderBodyTarget                                                                                                                                                                                                                                  string `json:"source_reader_body_target"`
	SourceReaderPassed                                                                                                                                                                                                                                      bool   `json:"source_reader_passed"`
	SourceReaderReason                                                                                                                                                                                                                                      string `json:"source_reader_reason"`

	SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReceiverObservationBoundaryPreflightGateCandidateStoreID       string `json:"source_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_final_gate_receiver_observation_boundary_preflight_gate_candidate_store_id"`
	SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReceiverObservationBoundaryPreflightGateCandidateStoreReady    bool   `json:"source_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_final_gate_receiver_observation_boundary_preflight_gate_candidate_store_ready"`
	SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReceiverObservationBoundaryPreflightGateCandidateStoreCausalID string `json:"source_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_final_gate_receiver_observation_boundary_preflight_gate_candidate_store_causal_id"`
	SourceAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreHash                                                                                                                                                                    string `json:"source_admission_final_gate_observation_boundary_preflight_gate_candidate_store_hash"`
	SourceAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReadBackHash                                                                                                                                                            string `json:"source_admission_final_gate_observation_boundary_preflight_gate_candidate_store_read_back_hash"`
	SourceStoreReceiptShape                                                                                                                                                                                                                       string `json:"source_store_receipt_shape"`
	SourceStoreKind                                                                                                                                                                                                                               string `json:"source_store_kind"`
	SourceStoreMode                                                                                                                                                                                                                               string `json:"source_store_mode"`
	SourceStoreStage                                                                                                                                                                                                                              string `json:"source_store_stage"`
	SourceStoreAppendOnly                                                                                                                                                                                                                         bool   `json:"source_store_append_only"`
	SourceStoreReadBack                                                                                                                                                                                                                           bool   `json:"source_store_read_back"`
	SourceStoreReceiptPersisted                                                                                                                                                                                                                   bool   `json:"source_store_receipt_persisted"`
	SourceStoreReceiptVerified                                                                                                                                                                                                                    bool   `json:"source_store_receipt_verified"`
	SourceStoreLedgerReady                                                                                                                                                                                                                        bool   `json:"source_store_ledger_ready"`
	SourceStoreLedgerAppendAllowed                                                                                                                                                                                                                bool   `json:"source_store_ledger_append_allowed"`
	SourceStoreRawDreamTextAllowed                                                                                                                                                                                                                bool   `json:"source_store_raw_dream_text_allowed"`
	SourceStoreRawDreamTextObserved                                                                                                                                                                                                               bool   `json:"source_store_raw_dream_text_observed"`
	SourceStoreRawDreamTextForwarded                                                                                                                                                                                                              bool   `json:"source_store_raw_dream_text_forwarded"`
	SourceStoreJanusSurfaceAllowed                                                                                                                                                                                                                bool   `json:"source_store_janus_surface_allowed"`
	SourceStoreCoocLearningAllowed                                                                                                                                                                                                                bool   `json:"source_store_cooc_learning_allowed"`
	SourceStoreDeltaHarvestAllowed                                                                                                                                                                                                                bool   `json:"source_store_delta_harvest_allowed"`
	SourceStoreBodyMutationAllowed                                                                                                                                                                                                                bool   `json:"source_store_body_mutation_allowed"`
	SourceStoreAuthorityGranted                                                                                                                                                                                                                   bool   `json:"source_store_authority_granted"`
	SourceStoreContractsReady                                                                                                                                                                                                                     bool   `json:"source_store_contracts_ready"`
	SourceStoreWriteAllowed                                                                                                                                                                                                                       bool   `json:"source_store_write_allowed"`
	SourceStoreAdmissionAllowed                                                                                                                                                                                                                   bool   `json:"source_store_admission_allowed"`
	SourceStoreLiveAdmissionEnabled                                                                                                                                                                                                               bool   `json:"source_store_live_admission_enabled"`
	SourceStoreMutatesState                                                                                                                                                                                                                       bool   `json:"source_store_mutates_state"`
	SourceStoreBodyTarget                                                                                                                                                                                                                         string `json:"source_store_body_target"`

	SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReceiverObservationBoundaryPreflightGateCandidateID       string `json:"source_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_final_gate_receiver_observation_boundary_preflight_gate_candidate_id"`
	SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReceiverObservationBoundaryPreflightGateCandidateReady    bool   `json:"source_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_final_gate_receiver_observation_boundary_preflight_gate_candidate_ready"`
	SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReceiverObservationBoundaryPreflightGateCandidateCausalID string `json:"source_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_final_gate_receiver_observation_boundary_preflight_gate_candidate_causal_id"`
	SourceAdmissionFinalGateObservationBoundaryPreflightGateCandidateHash                                                                                                                                                                    string `json:"source_admission_final_gate_observation_boundary_preflight_gate_candidate_hash"`
	SourceAdmissionFinalGateObservationBoundaryPreflightGateCandidateReadBackHash                                                                                                                                                            string `json:"source_admission_final_gate_observation_boundary_preflight_gate_candidate_read_back_hash"`
	SourceCandidateReceiptShape                                                                                                                                                                                                              string `json:"source_candidate_receipt_shape"`
	SourceCandidateState                                                                                                                                                                                                                     string `json:"source_candidate_state"`
	SourceCandidateKind                                                                                                                                                                                                                      string `json:"source_candidate_kind"`
	SourceCandidateMode                                                                                                                                                                                                                      string `json:"source_candidate_mode"`
	SourceCandidateStage                                                                                                                                                                                                                     string `json:"source_candidate_stage"`
	SourceCandidateDryRunOnly                                                                                                                                                                                                                bool   `json:"source_candidate_dry_run_only"`
	SourceCandidateGateVerified                                                                                                                                                                                                              bool   `json:"source_candidate_gate_verified"`
	SourceCandidatePreflightVerified                                                                                                                                                                                                         bool   `json:"source_candidate_preflight_verified"`
	SourceCandidateBoundaryVerified                                                                                                                                                                                                          bool   `json:"source_candidate_boundary_verified"`
	SourceCandidateObservationVerified                                                                                                                                                                                                       bool   `json:"source_candidate_observation_verified"`
	SourceCandidateReadBackVerified                                                                                                                                                                                                          bool   `json:"source_candidate_read_back_verified"`
	SourceCandidateOpened                                                                                                                                                                                                                    bool   `json:"source_candidate_opened"`
	SourceCandidateRawDreamTextObserved                                                                                                                                                                                                      bool   `json:"source_candidate_raw_dream_text_observed"`
	SourceCandidateRawDreamTextForwarded                                                                                                                                                                                                     bool   `json:"source_candidate_raw_dream_text_forwarded"`
	SourceCandidateRawDreamTextAllowed                                                                                                                                                                                                       bool   `json:"source_candidate_raw_dream_text_allowed"`
	SourceCandidateBodyMutationAllowed                                                                                                                                                                                                       bool   `json:"source_candidate_body_mutation_allowed"`

	SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReceiverObservationBoundaryPreflightGateID       string `json:"source_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_final_gate_receiver_observation_boundary_preflight_gate_id"`
	SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReceiverObservationBoundaryPreflightGateReady    bool   `json:"source_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_final_gate_receiver_observation_boundary_preflight_gate_ready"`
	SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReceiverObservationBoundaryPreflightGateCausalID string `json:"source_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_final_gate_receiver_observation_boundary_preflight_gate_causal_id"`
	SourceAdmissionFinalGateObservationBoundaryPreflightGateHash                                                                                                                                                                    string `json:"source_admission_final_gate_observation_boundary_preflight_gate_hash"`
	SourceAdmissionFinalGateObservationBoundaryPreflightGateReadBackHash                                                                                                                                                            string `json:"source_admission_final_gate_observation_boundary_preflight_gate_read_back_hash"`
	SourceAdmissionFinalGateObservationBoundaryPreflightGateReady                                                                                                                                                                   bool   `json:"source_admission_final_gate_observation_boundary_preflight_gate_ready"`
	SourceAdmissionFinalGateObservationBoundaryPreflightGateDryRunOnly                                                                                                                                                              bool   `json:"source_admission_final_gate_observation_boundary_preflight_gate_dry_run_only"`
	SourceAdmissionFinalGateObservationBoundaryPreflightGateReadBackVerified                                                                                                                                                        bool   `json:"source_admission_final_gate_observation_boundary_preflight_gate_read_back_verified"`
	SourceFinalGateObservationBoundaryPreflightGateRawDreamTextObserved                                                                                                                                                             bool   `json:"source_final_gate_observation_boundary_preflight_gate_raw_dream_text_observed"`
	SourceFinalGateObservationBoundaryPreflightGateRawDreamTextForwarded                                                                                                                                                            bool   `json:"source_final_gate_observation_boundary_preflight_gate_raw_dream_text_forwarded"`
	SourceFinalGateObservationBoundaryPreflightGateRawDreamTextAllowed                                                                                                                                                              bool   `json:"source_final_gate_observation_boundary_preflight_gate_raw_dream_text_allowed"`
	SourceFinalGateObservationBoundaryPreflightGateBodyMutationAllowed                                                                                                                                                              bool   `json:"source_final_gate_observation_boundary_preflight_gate_body_mutation_allowed"`

	SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReceiverObservationBoundaryPreflightID    string `json:"source_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_final_gate_receiver_observation_boundary_preflight_id"`
	SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReceiverObservationBoundaryPreflightReady bool   `json:"source_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_final_gate_receiver_observation_boundary_preflight_ready"`
	SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReceiverObservationBoundaryID             string `json:"source_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_final_gate_receiver_observation_boundary_id"`
	SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReceiverObservationBoundaryReady          bool   `json:"source_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_final_gate_receiver_observation_boundary_ready"`
	SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReceiverObservationID                     string `json:"source_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_final_gate_receiver_observation_id"`
	SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReceiverObservationReady                  bool   `json:"source_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_final_gate_receiver_observation_ready"`
	SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReceiverID                                string `json:"source_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_final_gate_receiver_id"`
	SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReceiverReady                             bool   `json:"source_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_final_gate_receiver_ready"`
	SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateIntentReady                               bool   `json:"source_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_final_gate_intent_ready"`
	SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReady                                     bool   `json:"source_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_final_gate_ready"`
	SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageSealReady                                          bool   `json:"source_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_seal_ready"`
	SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageAuthorityReady                                     bool   `json:"source_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_authority_ready"`
	SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStagePermitReady                                        bool   `json:"source_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_permit_ready"`
	SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageReadinessReady                                     bool   `json:"source_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_readiness_ready"`
	SourceWriterInventoryVerified                                                                                                                                                                                            bool   `json:"source_writer_inventory_verified"`
	SourceWriterPreflightVerified                                                                                                                                                                                            bool   `json:"source_writer_preflight_verified"`
	SourceAdmissionRequired                                                                                                                                                                                                  bool   `json:"source_admission_required"`
	SourceShadowOnly                                                                                                                                                                                                         bool   `json:"source_shadow_only"`
	SourceDryRunOnly                                                                                                                                                                                                         bool   `json:"source_dry_run_only"`
	SourceRequiresWriter                                                                                                                                                                                                     bool   `json:"source_requires_writer"`
	SourceRollbackRequired                                                                                                                                                                                                   bool   `json:"source_rollback_required"`
	SourceRequiresRollback                                                                                                                                                                                                   bool   `json:"source_requires_rollback"`
	SourceReadOnly                                                                                                                                                                                                           bool   `json:"source_read_only"`
	SourceReplayOnly                                                                                                                                                                                                         bool   `json:"source_replay_only"`
}

func runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReceiverObservationBoundaryPreflightGateCandidateStoreReaderProof(args []string) error {
	if len(args) != 2 {
		return fmt.Errorf("usage: --admission-live-route-weighted-admission-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-candidate-store-reader-proof-precondition-decision-promotion-switch-enable-gate-live-stage-final-gate-receiver-observation-boundary-preflight-gate-candidate-store-reader-proof RESONANCE_GRAFT_ADMISSION_FINAL_GATE_OBSERVATION_BOUNDARY_PREFLIGHT_GATE_CANDIDATE_STORE_READER_REPORT RESONANCE_GRAFT_ADMISSION_FINAL_GATE_OBSERVATION_BOUNDARY_PREFLIGHT_GATE_CANDIDATE_STORE_READER_PROOF_REPORT")
	}
	readerPath := args[0]
	outputPath := args[1]
	if strings.TrimSpace(outputPath) == "" {
		return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage final gate receiver observation boundary preflight gate candidate store reader proof output path missing")
	}
	reader, root, err := readAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReceiverObservationBoundaryPreflightGateCandidateStoreReaderReportForAssert(readerPath)
	if err != nil {
		return err
	}
	if err := admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReceiverObservationBoundaryPreflightGateCandidateStoreReaderReportError(reader, root); err != nil {
		return err
	}
	proof := admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReceiverObservationBoundaryPreflightGateCandidateStoreReaderProofReport{
		Schema:              admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReceiverObservationBoundaryPreflightGateCandidateStoreReaderProofSchema,
		Status:              "shadow_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_ready_dry_run",
		Target:              "live_route_admission_next_step",
		TargetKind:          "weighted_internal_world_shadow_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof",
		TargetMode:          "receipt_only_closed_reader_proof_dry_run",
		Action:              "prove_weighted_resonance_shadow_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_dry_run",
		LedgerState:         "blocked",
		LedgerAction:        "reject_blocked_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_ledger_append",
		LedgerContract:      "none",
		LedgerEntrypoint:    "none",
		LedgerReceiptShape:  "none",
		LedgerWriteScope:    "none",
		LedgerReady:         false,
		LedgerAppendAllowed: false,
		WeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReceiverObservationBoundaryPreflightGateCandidateStoreReaderProofReady: true,
		WeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReceiverObservationBoundaryPreflightGateCandidateStoreReaderConsumed:   true,
		WeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReceiverObservationBoundaryPreflightGateCandidateStoreReaderRequired:   true,
		NextStepBlockedWithoutResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProof:                                                                                                                                 true,
		ReceiptShape:           "weighted_resonance_shadow_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_receipt",
		ProofKind:              "shadow_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof",
		ProofMode:              "closed_read_back_reader_proof",
		ProofStage:             "post_preflight_gate_candidate_store_reader_pre_live_admission_proof",
		StoreReaderVerified:    true,
		StoreVerified:          reader.StoreVerified,
		CandidateVerified:      reader.CandidateVerified,
		GateVerified:           reader.GateVerified,
		PreflightVerified:      reader.PreflightVerified,
		BoundaryVerified:       reader.BoundaryVerified,
		ObservationVerified:    reader.ObservationVerified,
		ReceiverVerified:       reader.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReceiverReady,
		IntentVerified:         reader.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateIntentReady,
		FinalGateVerified:      reader.FinalGateVerified,
		SealVerified:           reader.SealVerified,
		PermitVerified:         reader.PermitVerified,
		AuthorityVerified:      reader.AuthorityVerified,
		ReaderHashVerified:     true,
		ReaderReplayVerified:   true,
		ReaderReadBackVerified: true,
		StoreHashVerified:      reader.StoreHashVerified,
		StoreReadBackVerified:  reader.StoreReadBackVerified,
		AdmissionRequired:      true,
		ShadowOnly:             true,
		GraftAllowed:           false,
		DryRunOnly:             true,
		LiveReady:              reader.LiveReady,
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
		Reason:                 "weighted resonance shadow graft admission final gate observation boundary preflight gate candidate store reader proof sealed without ledger append or body mutation",

		SourceSchema: reader.Schema,
		SourceStatus: reader.Status,
		SourceTarget: reader.Target,
		SourceReport: readerPath,
		SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReceiverObservationBoundaryPreflightGateCandidateStoreReaderID:           reader.WeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReceiverObservationBoundaryPreflightGateCandidateStoreReaderID,
		SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReceiverObservationBoundaryPreflightGateCandidateStoreReaderReady:        reader.WeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReceiverObservationBoundaryPreflightGateCandidateStoreReaderReady,
		SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReceiverObservationBoundaryPreflightGateCandidateStoreReaderCausalID:     reader.CausalID,
		SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReceiverObservationBoundaryPreflightGateCandidateStoreReaderHash:         reader.ReaderHash,
		SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReceiverObservationBoundaryPreflightGateCandidateStoreReaderReplayHash:   reader.ReplayHash,
		SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReceiverObservationBoundaryPreflightGateCandidateStoreReaderReadBackHash: reader.ReadBackHash,
		SourceReaderAction:                reader.Action,
		SourceReaderReceiptShape:          reader.ReceiptShape,
		SourceReaderKind:                  reader.ReaderKind,
		SourceReaderMode:                  reader.ReaderMode,
		SourceReaderStage:                 reader.ReaderStage,
		SourceReaderReadOnly:              reader.ReadOnly,
		SourceReaderReplayOnly:            reader.ReplayOnly,
		SourceReaderStoreVerified:         reader.StoreVerified,
		SourceReaderCandidateVerified:     reader.CandidateVerified,
		SourceReaderGateVerified:          reader.GateVerified,
		SourceReaderPreflightVerified:     reader.PreflightVerified,
		SourceReaderBoundaryVerified:      reader.BoundaryVerified,
		SourceReaderObservationVerified:   reader.ObservationVerified,
		SourceReaderFinalGateVerified:     reader.FinalGateVerified,
		SourceReaderSealVerified:          reader.SealVerified,
		SourceReaderPermitVerified:        reader.PermitVerified,
		SourceReaderAuthorityVerified:     reader.AuthorityVerified,
		SourceReaderStoreHashVerified:     reader.StoreHashVerified,
		SourceReaderStoreReadBackVerified: reader.StoreReadBackVerified,
		SourceReaderAdmissionRequired:     reader.AdmissionRequired,
		SourceReaderShadowOnly:            reader.ShadowOnly,
		SourceReaderDryRunOnly:            reader.DryRunOnly,
		SourceReaderLiveReady:             reader.LiveReady,
		SourceReaderRollbackRequired:      reader.RollbackRequired,
		SourceReaderLedgerReady:           reader.LedgerReady,
		SourceReaderLedgerAppendAllowed:   reader.LedgerAppendAllowed,
		SourceReaderRawDreamTextAllowed:   reader.RawDreamTextAllowed,
		SourceReaderRawDreamTextObserved:  reader.RawDreamTextObserved,
		SourceReaderRawDreamTextForwarded: reader.RawDreamTextForwarded,
		SourceReaderJanusSurfaceAllowed:   reader.JanusSurfaceAllowed,
		SourceReaderCoocLearningAllowed:   reader.CoocLearningAllowed,
		SourceReaderDeltaHarvestAllowed:   reader.DeltaHarvestAllowed,
		SourceReaderBodyMutationAllowed:   reader.BodyMutationAllowed,
		SourceReaderAuthorityGranted:      reader.AuthorityGranted,
		SourceReaderContractsReady:        reader.ContractsReady,
		SourceReaderWriteAllowed:          reader.WriteAllowed,
		SourceReaderAdmissionAllowed:      reader.AdmissionAllowed,
		SourceReaderLiveAdmissionEnabled:  reader.LiveAdmissionEnabled,
		SourceReaderMutatesState:          reader.MutatesState,
		SourceReaderBodyTarget:            reader.BodyTarget,
		SourceReaderPassed:                reader.Passed,
		SourceReaderReason:                reader.Reason,

		SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReceiverObservationBoundaryPreflightGateCandidateStoreID:       reader.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReceiverObservationBoundaryPreflightGateCandidateStoreID,
		SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReceiverObservationBoundaryPreflightGateCandidateStoreReady:    reader.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReceiverObservationBoundaryPreflightGateCandidateStoreReady,
		SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReceiverObservationBoundaryPreflightGateCandidateStoreCausalID: reader.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReceiverObservationBoundaryPreflightGateCandidateStoreCausal,
		SourceAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreHash:         reader.SourceAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreHash,
		SourceAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReadBackHash: reader.SourceAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReadBackHash,
		SourceStoreReceiptShape:          reader.SourceStoreReceiptShape,
		SourceStoreKind:                  reader.SourceStoreKind,
		SourceStoreMode:                  reader.SourceStoreMode,
		SourceStoreStage:                 reader.SourceStoreStage,
		SourceStoreAppendOnly:            reader.SourceStoreAppendOnly,
		SourceStoreReadBack:              reader.SourceStoreReadBack,
		SourceStoreReceiptPersisted:      reader.SourceStoreReceiptPersisted,
		SourceStoreReceiptVerified:       reader.SourceStoreReceiptVerified,
		SourceStoreLedgerReady:           reader.SourceStoreLedgerReady,
		SourceStoreLedgerAppendAllowed:   reader.SourceStoreLedgerAppendAllowed,
		SourceStoreRawDreamTextAllowed:   reader.SourceStoreRawDreamTextAllowed,
		SourceStoreRawDreamTextObserved:  reader.SourceStoreRawDreamTextObserved,
		SourceStoreRawDreamTextForwarded: reader.SourceStoreRawDreamTextForwarded,
		SourceStoreJanusSurfaceAllowed:   reader.SourceStoreJanusSurfaceAllowed,
		SourceStoreCoocLearningAllowed:   reader.SourceStoreCoocLearningAllowed,
		SourceStoreDeltaHarvestAllowed:   reader.SourceStoreDeltaHarvestAllowed,
		SourceStoreBodyMutationAllowed:   reader.SourceStoreBodyMutationAllowed,
		SourceStoreAuthorityGranted:      reader.SourceStoreAuthorityGranted,
		SourceStoreContractsReady:        reader.SourceStoreContractsReady,
		SourceStoreWriteAllowed:          reader.SourceStoreWriteAllowed,
		SourceStoreAdmissionAllowed:      reader.SourceStoreAdmissionAllowed,
		SourceStoreLiveAdmissionEnabled:  reader.SourceStoreLiveAdmissionEnabled,
		SourceStoreMutatesState:          reader.SourceStoreMutatesState,
		SourceStoreBodyTarget:            reader.SourceStoreBodyTarget,

		SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReceiverObservationBoundaryPreflightGateCandidateID:       reader.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReceiverObservationBoundaryPreflightGateCandidateID,
		SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReceiverObservationBoundaryPreflightGateCandidateReady:    reader.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReceiverObservationBoundaryPreflightGateCandidateReady,
		SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReceiverObservationBoundaryPreflightGateCandidateCausalID: reader.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReceiverObservationBoundaryPreflightGateCandidateCausalID,
		SourceAdmissionFinalGateObservationBoundaryPreflightGateCandidateHash:         reader.SourceAdmissionFinalGateObservationBoundaryPreflightGateCandidateHash,
		SourceAdmissionFinalGateObservationBoundaryPreflightGateCandidateReadBackHash: reader.SourceAdmissionFinalGateObservationBoundaryPreflightGateCandidateReadBackHash,
		SourceCandidateReceiptShape:          reader.SourceCandidateReceiptShape,
		SourceCandidateState:                 reader.SourceCandidateState,
		SourceCandidateKind:                  reader.SourceCandidateKind,
		SourceCandidateMode:                  reader.SourceCandidateMode,
		SourceCandidateStage:                 reader.SourceCandidateStage,
		SourceCandidateDryRunOnly:            reader.SourceCandidateDryRunOnly,
		SourceCandidateGateVerified:          reader.SourceCandidateGateVerified,
		SourceCandidatePreflightVerified:     reader.SourceCandidatePreflightVerified,
		SourceCandidateBoundaryVerified:      reader.SourceCandidateBoundaryVerified,
		SourceCandidateObservationVerified:   reader.SourceCandidateObservationVerified,
		SourceCandidateReadBackVerified:      reader.SourceCandidateReadBackVerified,
		SourceCandidateOpened:                reader.SourceCandidateOpened,
		SourceCandidateRawDreamTextObserved:  reader.SourceCandidateRawDreamTextObserved,
		SourceCandidateRawDreamTextForwarded: reader.SourceCandidateRawDreamTextForwarded,
		SourceCandidateRawDreamTextAllowed:   reader.SourceCandidateRawDreamTextAllowed,
		SourceCandidateBodyMutationAllowed:   reader.SourceCandidateBodyMutationAllowed,

		SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReceiverObservationBoundaryPreflightGateID:       reader.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReceiverObservationBoundaryPreflightGateID,
		SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReceiverObservationBoundaryPreflightGateReady:    reader.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReceiverObservationBoundaryPreflightGateReady,
		SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReceiverObservationBoundaryPreflightGateCausalID: reader.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReceiverObservationBoundaryPreflightGateCausalID,
		SourceAdmissionFinalGateObservationBoundaryPreflightGateHash:             reader.SourceAdmissionFinalGateObservationBoundaryPreflightGateHash,
		SourceAdmissionFinalGateObservationBoundaryPreflightGateReadBackHash:     reader.SourceAdmissionFinalGateObservationBoundaryPreflightGateReadBackHash,
		SourceAdmissionFinalGateObservationBoundaryPreflightGateReady:            reader.SourceAdmissionFinalGateObservationBoundaryPreflightGateReady,
		SourceAdmissionFinalGateObservationBoundaryPreflightGateDryRunOnly:       reader.SourceAdmissionFinalGateObservationBoundaryPreflightGateDryRunOnly,
		SourceAdmissionFinalGateObservationBoundaryPreflightGateReadBackVerified: reader.SourceAdmissionFinalGateObservationBoundaryPreflightGateReadBackVerified,
		SourceFinalGateObservationBoundaryPreflightGateRawDreamTextObserved:      reader.SourceFinalGateObservationBoundaryPreflightGateRawDreamTextObserved,
		SourceFinalGateObservationBoundaryPreflightGateRawDreamTextForwarded:     reader.SourceFinalGateObservationBoundaryPreflightGateRawDreamTextForwarded,
		SourceFinalGateObservationBoundaryPreflightGateRawDreamTextAllowed:       reader.SourceFinalGateObservationBoundaryPreflightGateRawDreamTextAllowed,
		SourceFinalGateObservationBoundaryPreflightGateBodyMutationAllowed:       reader.SourceFinalGateObservationBoundaryPreflightGateBodyMutationAllowed,

		SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReceiverObservationBoundaryPreflightID:    reader.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReceiverObservationBoundaryPreflightID,
		SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReceiverObservationBoundaryPreflightReady: reader.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReceiverObservationBoundaryPreflightReady,
		SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReceiverObservationBoundaryID:             reader.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReceiverObservationBoundaryID,
		SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReceiverObservationBoundaryReady:          reader.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReceiverObservationBoundaryReady,
		SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReceiverObservationID:                     reader.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReceiverObservationID,
		SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReceiverObservationReady:                  reader.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReceiverObservationReady,
		SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReceiverID:                                reader.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReceiverID,
		SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReceiverReady:                             reader.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReceiverReady,
		SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateIntentReady:                               reader.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateIntentReady,
		SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReady:                                     reader.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReady,
		SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageSealReady:                                          reader.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageSealReady,
		SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageAuthorityReady:                                     reader.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageAuthorityReady,
		SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStagePermitReady:                                        reader.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStagePermitReady,
		SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageReadinessReady:                                     reader.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageReadinessReady,
		SourceWriterInventoryVerified: reader.SourceWriterInventoryVerified,
		SourceWriterPreflightVerified: reader.SourceWriterPreflightVerified,
		SourceAdmissionRequired:       reader.SourceAdmissionRequired,
		SourceShadowOnly:              reader.SourceShadowOnly,
		SourceDryRunOnly:              reader.SourceDryRunOnly,
		SourceRequiresWriter:          reader.SourceRequiresWriter,
		SourceRollbackRequired:        reader.SourceRollbackRequired,
		SourceRequiresRollback:        reader.SourceRequiresRollback,
		SourceReadOnly:                reader.SourceReadOnly,
		SourceReplayOnly:              reader.SourceReplayOnly,
	}
	proof.CausalID = admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReceiverObservationBoundaryPreflightGateCandidateStoreReaderProofCausalID(proof)
	proof.ProofHash = admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReceiverObservationBoundaryPreflightGateCandidateStoreReaderProofHash(proof)
	proof.ReadBackHash = admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReceiverObservationBoundaryPreflightGateCandidateStoreReaderProofReadBackHash(proof)
	proof.WeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReceiverObservationBoundaryPreflightGateCandidateStoreReaderProofID = admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReceiverObservationBoundaryPreflightGateCandidateStoreReaderProofID(proof)
	if proof.CausalID == "" ||
		proof.ProofHash == "" ||
		proof.ReadBackHash == "" ||
		proof.WeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReceiverObservationBoundaryPreflightGateCandidateStoreReaderProofID == "" ||
		proof.ProofHash == proof.ReadBackHash {
		return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage final gate receiver observation boundary preflight gate candidate store reader proof read-back proof failed")
	}
	raw, err := json.MarshalIndent(proof, "", "  ")
	if err != nil {
		return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage final gate receiver observation boundary preflight gate candidate store reader proof marshal failed: %w", err)
	}
	raw = append(raw, '\n')
	if err := os.WriteFile(outputPath, raw, 0600); err != nil {
		return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage final gate receiver observation boundary preflight gate candidate store reader proof write failed: %w", err)
	}
	fmt.Printf("[admission-live-route-weighted-admission-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-candidate-store-reader-proof-precondition-decision-promotion-switch-enable-gate-live-stage-final-gate-receiver-observation-boundary-preflight-gate-candidate-store-reader-proof] pass: resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_report=%s resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_report=%s\n", outputPath, readerPath)
	return nil
}

func runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReceiverObservationBoundaryPreflightGateCandidateStoreReaderProofAssert(args []string) error {
	if len(args) != 1 {
		return fmt.Errorf("usage: --admission-live-route-weighted-admission-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-candidate-store-reader-proof-precondition-decision-promotion-switch-enable-gate-live-stage-final-gate-receiver-observation-boundary-preflight-gate-candidate-store-reader-proof-assert REPORT")
	}
	report, root, err := readAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReceiverObservationBoundaryPreflightGateCandidateStoreReaderProofReportForAssert(args[0])
	if err != nil {
		return err
	}
	return admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReceiverObservationBoundaryPreflightGateCandidateStoreReaderProofReportError(report, root)
}

func admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReceiverObservationBoundaryPreflightGateCandidateStoreReaderProofReportError(report admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReceiverObservationBoundaryPreflightGateCandidateStoreReaderProofReport, root map[string]json.RawMessage) error {
	if _, ok := root["schema"]; !ok {
		return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage final gate receiver observation boundary preflight gate candidate store reader proof schema missing")
	}
	if report.Schema != admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReceiverObservationBoundaryPreflightGateCandidateStoreReaderProofSchema {
		return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage final gate receiver observation boundary preflight gate candidate store reader proof schema mismatch: got %q want %q", report.Schema, admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReceiverObservationBoundaryPreflightGateCandidateStoreReaderProofSchema)
	}
	if report.Status != "shadow_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_ready_dry_run" {
		return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage final gate receiver observation boundary preflight gate candidate store reader proof status mismatch: got %q want %q", report.Status, "shadow_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_ready_dry_run")
	}
	if report.Target != "live_route_admission_next_step" ||
		report.TargetKind != "weighted_internal_world_shadow_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof" ||
		report.TargetMode != "receipt_only_closed_reader_proof_dry_run" ||
		report.Action != "prove_weighted_resonance_shadow_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_dry_run" {
		return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage final gate receiver observation boundary preflight gate candidate store reader proof route shape mismatch")
	}
	if report.LedgerState != "blocked" ||
		report.LedgerAction != "reject_blocked_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_ledger_append" ||
		report.LedgerContract != "none" ||
		report.LedgerEntrypoint != "none" ||
		report.LedgerReceiptShape != "none" ||
		report.LedgerWriteScope != "none" {
		return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage final gate receiver observation boundary preflight gate candidate store reader proof ledger guard mismatch")
	}
	if report.ReceiptShape != "weighted_resonance_shadow_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_receipt" ||
		report.ProofKind != "shadow_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof" ||
		report.ProofMode != "closed_read_back_reader_proof" ||
		report.ProofStage != "post_preflight_gate_candidate_store_reader_pre_live_admission_proof" {
		return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage final gate receiver observation boundary preflight gate candidate store reader proof shape mismatch")
	}
	for _, required := range []struct {
		name  string
		value bool
	}{
		{"weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_final_gate_receiver_observation_boundary_preflight_gate_candidate_store_reader_proof_ready", report.WeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReceiverObservationBoundaryPreflightGateCandidateStoreReaderProofReady},
		{"weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_final_gate_receiver_observation_boundary_preflight_gate_candidate_store_reader_consumed", report.WeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReceiverObservationBoundaryPreflightGateCandidateStoreReaderConsumed},
		{"weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_final_gate_receiver_observation_boundary_preflight_gate_candidate_store_reader_required", report.WeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReceiverObservationBoundaryPreflightGateCandidateStoreReaderRequired},
		{"next_step_blocked_without_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof", report.NextStepBlockedWithoutResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProof},
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
		{"source_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_final_gate_receiver_observation_boundary_preflight_gate_candidate_store_reader_ready", report.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReceiverObservationBoundaryPreflightGateCandidateStoreReaderReady},
		{"source_reader_read_only", report.SourceReaderReadOnly},
		{"source_reader_replay_only", report.SourceReaderReplayOnly},
		{"source_reader_store_verified", report.SourceReaderStoreVerified},
		{"source_reader_candidate_verified", report.SourceReaderCandidateVerified},
		{"source_reader_gate_verified", report.SourceReaderGateVerified},
		{"source_reader_preflight_verified", report.SourceReaderPreflightVerified},
		{"source_reader_boundary_verified", report.SourceReaderBoundaryVerified},
		{"source_reader_observation_verified", report.SourceReaderObservationVerified},
		{"source_reader_final_gate_verified", report.SourceReaderFinalGateVerified},
		{"source_reader_seal_verified", report.SourceReaderSealVerified},
		{"source_reader_permit_verified", report.SourceReaderPermitVerified},
		{"source_reader_authority_verified", report.SourceReaderAuthorityVerified},
		{"source_reader_store_hash_verified", report.SourceReaderStoreHashVerified},
		{"source_reader_store_read_back_verified", report.SourceReaderStoreReadBackVerified},
		{"source_reader_admission_required", report.SourceReaderAdmissionRequired},
		{"source_reader_shadow_only", report.SourceReaderShadowOnly},
		{"source_reader_dry_run_only", report.SourceReaderDryRunOnly},
		{"source_reader_live_ready", report.SourceReaderLiveReady},
		{"source_reader_rollback_required", report.SourceReaderRollbackRequired},
		{"source_reader_passed", report.SourceReaderPassed},
		{"source_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_final_gate_receiver_observation_boundary_preflight_gate_candidate_store_ready", report.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReceiverObservationBoundaryPreflightGateCandidateStoreReady},
		{"source_store_append_only", report.SourceStoreAppendOnly},
		{"source_store_read_back", report.SourceStoreReadBack},
		{"source_store_receipt_persisted", report.SourceStoreReceiptPersisted},
		{"source_store_receipt_verified", report.SourceStoreReceiptVerified},
		{"source_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_final_gate_receiver_observation_boundary_preflight_gate_candidate_ready", report.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReceiverObservationBoundaryPreflightGateCandidateReady},
		{"source_candidate_dry_run_only", report.SourceCandidateDryRunOnly},
		{"source_candidate_gate_verified", report.SourceCandidateGateVerified},
		{"source_candidate_preflight_verified", report.SourceCandidatePreflightVerified},
		{"source_candidate_boundary_verified", report.SourceCandidateBoundaryVerified},
		{"source_candidate_observation_verified", report.SourceCandidateObservationVerified},
		{"source_candidate_read_back_verified", report.SourceCandidateReadBackVerified},
		{"source_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_final_gate_receiver_observation_boundary_preflight_gate_ready", report.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReceiverObservationBoundaryPreflightGateReady},
		{"source_admission_final_gate_observation_boundary_preflight_gate_dry_run_only", report.SourceAdmissionFinalGateObservationBoundaryPreflightGateDryRunOnly},
		{"source_admission_final_gate_observation_boundary_preflight_gate_read_back_verified", report.SourceAdmissionFinalGateObservationBoundaryPreflightGateReadBackVerified},
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
			return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage final gate receiver observation boundary preflight gate candidate store reader proof %s not ready", required.name)
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
		{"source_reader_ledger_ready", report.SourceReaderLedgerReady},
		{"source_reader_ledger_append_allowed", report.SourceReaderLedgerAppendAllowed},
		{"source_reader_raw_dream_text_allowed", report.SourceReaderRawDreamTextAllowed},
		{"source_reader_raw_dream_text_observed", report.SourceReaderRawDreamTextObserved},
		{"source_reader_raw_dream_text_forwarded", report.SourceReaderRawDreamTextForwarded},
		{"source_reader_janus_surface_allowed", report.SourceReaderJanusSurfaceAllowed},
		{"source_reader_cooc_learning_allowed", report.SourceReaderCoocLearningAllowed},
		{"source_reader_delta_harvest_allowed", report.SourceReaderDeltaHarvestAllowed},
		{"source_reader_body_mutation_allowed", report.SourceReaderBodyMutationAllowed},
		{"source_reader_authority_granted", report.SourceReaderAuthorityGranted},
		{"source_reader_contracts_ready", report.SourceReaderContractsReady},
		{"source_reader_write_allowed", report.SourceReaderWriteAllowed},
		{"source_reader_admission_allowed", report.SourceReaderAdmissionAllowed},
		{"source_reader_live_admission_enabled", report.SourceReaderLiveAdmissionEnabled},
		{"source_reader_mutates_state", report.SourceReaderMutatesState},
		{"source_store_ledger_ready", report.SourceStoreLedgerReady},
		{"source_store_ledger_append_allowed", report.SourceStoreLedgerAppendAllowed},
		{"source_store_raw_dream_text_allowed", report.SourceStoreRawDreamTextAllowed},
		{"source_store_raw_dream_text_observed", report.SourceStoreRawDreamTextObserved},
		{"source_store_raw_dream_text_forwarded", report.SourceStoreRawDreamTextForwarded},
		{"source_store_janus_surface_allowed", report.SourceStoreJanusSurfaceAllowed},
		{"source_store_cooc_learning_allowed", report.SourceStoreCoocLearningAllowed},
		{"source_store_delta_harvest_allowed", report.SourceStoreDeltaHarvestAllowed},
		{"source_store_body_mutation_allowed", report.SourceStoreBodyMutationAllowed},
		{"source_store_authority_granted", report.SourceStoreAuthorityGranted},
		{"source_store_contracts_ready", report.SourceStoreContractsReady},
		{"source_store_write_allowed", report.SourceStoreWriteAllowed},
		{"source_store_admission_allowed", report.SourceStoreAdmissionAllowed},
		{"source_store_live_admission_enabled", report.SourceStoreLiveAdmissionEnabled},
		{"source_store_mutates_state", report.SourceStoreMutatesState},
		{"source_candidate_opened", report.SourceCandidateOpened},
		{"source_candidate_raw_dream_text_observed", report.SourceCandidateRawDreamTextObserved},
		{"source_candidate_raw_dream_text_forwarded", report.SourceCandidateRawDreamTextForwarded},
		{"source_candidate_raw_dream_text_allowed", report.SourceCandidateRawDreamTextAllowed},
		{"source_candidate_body_mutation_allowed", report.SourceCandidateBodyMutationAllowed},
		{"source_admission_final_gate_observation_boundary_preflight_gate_ready", report.SourceAdmissionFinalGateObservationBoundaryPreflightGateReady},
		{"source_final_gate_observation_boundary_preflight_gate_raw_dream_text_observed", report.SourceFinalGateObservationBoundaryPreflightGateRawDreamTextObserved},
		{"source_final_gate_observation_boundary_preflight_gate_raw_dream_text_forwarded", report.SourceFinalGateObservationBoundaryPreflightGateRawDreamTextForwarded},
		{"source_final_gate_observation_boundary_preflight_gate_raw_dream_text_allowed", report.SourceFinalGateObservationBoundaryPreflightGateRawDreamTextAllowed},
		{"source_final_gate_observation_boundary_preflight_gate_body_mutation_allowed", report.SourceFinalGateObservationBoundaryPreflightGateBodyMutationAllowed},
	} {
		if closed.value {
			return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage final gate receiver observation boundary preflight gate candidate store reader proof opened %s", closed.name)
		}
	}
	for _, requiredString := range []struct {
		name  string
		value string
	}{
		{"weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_final_gate_receiver_observation_boundary_preflight_gate_candidate_store_reader_proof_id", report.WeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReceiverObservationBoundaryPreflightGateCandidateStoreReaderProofID},
		{"causal_id", report.CausalID},
		{"proof_hash", report.ProofHash},
		{"read_back_hash", report.ReadBackHash},
		{"source_report", report.SourceReport},
		{"source_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_final_gate_receiver_observation_boundary_preflight_gate_candidate_store_reader_id", report.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReceiverObservationBoundaryPreflightGateCandidateStoreReaderID},
		{"source_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_final_gate_receiver_observation_boundary_preflight_gate_candidate_store_reader_causal_id", report.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReceiverObservationBoundaryPreflightGateCandidateStoreReaderCausalID},
		{"source_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_final_gate_receiver_observation_boundary_preflight_gate_candidate_store_reader_hash", report.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReceiverObservationBoundaryPreflightGateCandidateStoreReaderHash},
		{"source_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_final_gate_receiver_observation_boundary_preflight_gate_candidate_store_reader_replay_hash", report.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReceiverObservationBoundaryPreflightGateCandidateStoreReaderReplayHash},
		{"source_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_final_gate_receiver_observation_boundary_preflight_gate_candidate_store_reader_read_back_hash", report.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReceiverObservationBoundaryPreflightGateCandidateStoreReaderReadBackHash},
		{"source_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_final_gate_receiver_observation_boundary_preflight_gate_candidate_store_id", report.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReceiverObservationBoundaryPreflightGateCandidateStoreID},
		{"source_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_final_gate_receiver_observation_boundary_preflight_gate_candidate_store_causal_id", report.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReceiverObservationBoundaryPreflightGateCandidateStoreCausalID},
		{"source_admission_final_gate_observation_boundary_preflight_gate_candidate_store_hash", report.SourceAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreHash},
		{"source_admission_final_gate_observation_boundary_preflight_gate_candidate_store_read_back_hash", report.SourceAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReadBackHash},
		{"source_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_final_gate_receiver_observation_boundary_preflight_gate_candidate_id", report.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReceiverObservationBoundaryPreflightGateCandidateID},
		{"source_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_final_gate_receiver_observation_boundary_preflight_gate_candidate_causal_id", report.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReceiverObservationBoundaryPreflightGateCandidateCausalID},
		{"source_admission_final_gate_observation_boundary_preflight_gate_candidate_hash", report.SourceAdmissionFinalGateObservationBoundaryPreflightGateCandidateHash},
		{"source_admission_final_gate_observation_boundary_preflight_gate_candidate_read_back_hash", report.SourceAdmissionFinalGateObservationBoundaryPreflightGateCandidateReadBackHash},
		{"source_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_final_gate_receiver_observation_boundary_preflight_gate_id", report.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReceiverObservationBoundaryPreflightGateID},
		{"source_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_final_gate_receiver_observation_boundary_preflight_gate_causal_id", report.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReceiverObservationBoundaryPreflightGateCausalID},
		{"source_admission_final_gate_observation_boundary_preflight_gate_hash", report.SourceAdmissionFinalGateObservationBoundaryPreflightGateHash},
		{"source_admission_final_gate_observation_boundary_preflight_gate_read_back_hash", report.SourceAdmissionFinalGateObservationBoundaryPreflightGateReadBackHash},
		{"source_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_final_gate_receiver_observation_boundary_preflight_id", report.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReceiverObservationBoundaryPreflightID},
		{"source_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_final_gate_receiver_observation_boundary_id", report.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReceiverObservationBoundaryID},
		{"source_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_final_gate_receiver_observation_id", report.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReceiverObservationID},
		{"source_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_final_gate_receiver_id", report.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReceiverID},
	} {
		if strings.TrimSpace(requiredString.value) == "" {
			return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage final gate receiver observation boundary preflight gate candidate store reader proof %s missing", requiredString.name)
		}
	}
	if report.SourceSchema != admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReceiverObservationBoundaryPreflightGateCandidateStoreReaderSchema {
		return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage final gate receiver observation boundary preflight gate candidate store reader proof source_schema mismatch: got %q want %q", report.SourceSchema, admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReceiverObservationBoundaryPreflightGateCandidateStoreReaderSchema)
	}
	if report.SourceStatus != "shadow_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_read_back_dry_run" ||
		report.SourceTarget != "live_route_admission_next_step" ||
		report.SourceReaderAction != "read_weighted_resonance_shadow_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_dry_run" {
		return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage final gate receiver observation boundary preflight gate candidate store reader proof source reader route mismatch")
	}
	if report.SourceReaderReceiptShape != "weighted_resonance_shadow_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_receipt" ||
		report.SourceReaderKind != "shadow_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader" ||
		report.SourceReaderMode != "read_only_replay" ||
		report.SourceReaderStage != "post_preflight_gate_candidate_store_pre_live_admission_reader" {
		return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage final gate receiver observation boundary preflight gate candidate store reader proof source reader shape mismatch")
	}
	if report.SourceStoreReceiptShape != "weighted_resonance_shadow_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_receipt" ||
		report.SourceStoreKind != "shadow_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store" ||
		report.SourceStoreMode != "append_only_read_back_store" ||
		report.SourceStoreStage != "post_preflight_gate_candidate_pre_live_admission_store" {
		return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage final gate receiver observation boundary preflight gate candidate store reader proof source store shape mismatch")
	}
	if report.SourceCandidateReceiptShape != "weighted_resonance_shadow_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_final_gate_receiver_observation_boundary_preflight_gate_candidate_receipt" ||
		report.SourceCandidateState != "blocked" ||
		report.SourceCandidateKind != "blocked_final_gate_observation_boundary_preflight_gate_candidate" ||
		report.SourceCandidateMode != "no_mutation_preflight_gate_candidate" ||
		report.SourceCandidateStage != "post_preflight_gate_pre_live_admission" {
		return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage final gate receiver observation boundary preflight gate candidate store reader proof source candidate shape mismatch")
	}
	if report.BodyTarget != "none" || report.SourceReaderBodyTarget != "none" || report.SourceStoreBodyTarget != "none" {
		return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage final gate receiver observation boundary preflight gate candidate store reader proof body target mismatch")
	}
	if !strings.HasPrefix(report.WeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReceiverObservationBoundaryPreflightGateCandidateStoreReaderProofID, "weighted-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-candidate-store-reader-proof-precondition-decision-promotion-switch-enable-gate-live-stage-final-gate-receiver-observation-boundary-preflight-gate-candidate-store-reader-proof-id-") ||
		!strings.HasPrefix(report.CausalID, "weighted-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-candidate-store-reader-proof-precondition-decision-promotion-switch-enable-gate-live-stage-final-gate-receiver-observation-boundary-preflight-gate-candidate-store-reader-proof-causal-") ||
		!strings.HasPrefix(report.ProofHash, "weighted-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-candidate-store-reader-proof-precondition-decision-promotion-switch-enable-gate-live-stage-final-gate-receiver-observation-boundary-preflight-gate-candidate-store-reader-proof-") ||
		!strings.HasPrefix(report.ReadBackHash, "weighted-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-candidate-store-reader-proof-precondition-decision-promotion-switch-enable-gate-live-stage-final-gate-receiver-observation-boundary-preflight-gate-candidate-store-reader-proof-read-") ||
		report.ProofHash == report.ReadBackHash {
		return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage final gate receiver observation boundary preflight gate candidate store reader proof prefix mismatch")
	}
	if !strings.HasPrefix(report.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReceiverObservationBoundaryPreflightGateCandidateStoreReaderID, "weighted-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-candidate-store-reader-proof-precondition-decision-promotion-switch-enable-gate-live-stage-final-gate-receiver-observation-boundary-preflight-gate-candidate-store-reader-id-") ||
		!strings.HasPrefix(report.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReceiverObservationBoundaryPreflightGateCandidateStoreReaderCausalID, "weighted-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-candidate-store-reader-proof-precondition-decision-promotion-switch-enable-gate-live-stage-final-gate-receiver-observation-boundary-preflight-gate-candidate-store-reader-causal-") ||
		!strings.HasPrefix(report.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReceiverObservationBoundaryPreflightGateCandidateStoreReaderHash, "weighted-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-candidate-store-reader-proof-precondition-decision-promotion-switch-enable-gate-live-stage-final-gate-receiver-observation-boundary-preflight-gate-candidate-store-reader-") ||
		!strings.HasPrefix(report.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReceiverObservationBoundaryPreflightGateCandidateStoreReaderReplayHash, "weighted-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-candidate-store-reader-proof-precondition-decision-promotion-switch-enable-gate-live-stage-final-gate-receiver-observation-boundary-preflight-gate-candidate-store-reader-replay-") ||
		!strings.HasPrefix(report.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReceiverObservationBoundaryPreflightGateCandidateStoreReaderReadBackHash, "weighted-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-candidate-store-reader-proof-precondition-decision-promotion-switch-enable-gate-live-stage-final-gate-receiver-observation-boundary-preflight-gate-candidate-store-reader-read-") {
		return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage final gate receiver observation boundary preflight gate candidate store reader proof source reader proof mismatch")
	}
	if !strings.HasPrefix(report.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReceiverObservationBoundaryPreflightGateCandidateStoreID, "weighted-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-candidate-store-reader-proof-precondition-decision-promotion-switch-enable-gate-live-stage-final-gate-receiver-observation-boundary-preflight-gate-candidate-store-id-") ||
		!strings.HasPrefix(report.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReceiverObservationBoundaryPreflightGateCandidateStoreCausalID, "weighted-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-candidate-store-reader-proof-precondition-decision-promotion-switch-enable-gate-live-stage-final-gate-receiver-observation-boundary-preflight-gate-candidate-store-causal-") ||
		!strings.HasPrefix(report.SourceAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreHash, "weighted-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-candidate-store-reader-proof-precondition-decision-promotion-switch-enable-gate-live-stage-final-gate-receiver-observation-boundary-preflight-gate-candidate-store-") ||
		!strings.HasPrefix(report.SourceAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReadBackHash, "weighted-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-candidate-store-reader-proof-precondition-decision-promotion-switch-enable-gate-live-stage-final-gate-receiver-observation-boundary-preflight-gate-candidate-store-read-") {
		return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage final gate receiver observation boundary preflight gate candidate store reader proof source store proof mismatch")
	}
	if !strings.HasPrefix(report.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReceiverObservationBoundaryPreflightGateCandidateID, "weighted-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-candidate-store-reader-proof-precondition-decision-promotion-switch-enable-gate-live-stage-final-gate-receiver-observation-boundary-preflight-gate-candidate-id-") ||
		!strings.HasPrefix(report.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReceiverObservationBoundaryPreflightGateCandidateCausalID, "weighted-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-candidate-store-reader-proof-precondition-decision-promotion-switch-enable-gate-live-stage-final-gate-receiver-observation-boundary-preflight-gate-candidate-causal-") ||
		!strings.HasPrefix(report.SourceAdmissionFinalGateObservationBoundaryPreflightGateCandidateHash, "weighted-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-candidate-store-reader-proof-precondition-decision-promotion-switch-enable-gate-live-stage-final-gate-receiver-observation-boundary-preflight-gate-candidate-") ||
		!strings.HasPrefix(report.SourceAdmissionFinalGateObservationBoundaryPreflightGateCandidateReadBackHash, "weighted-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-candidate-store-reader-proof-precondition-decision-promotion-switch-enable-gate-live-stage-final-gate-receiver-observation-boundary-preflight-gate-candidate-read-") {
		return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage final gate receiver observation boundary preflight gate candidate store reader proof source candidate proof mismatch")
	}
	if !strings.HasPrefix(report.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReceiverObservationBoundaryPreflightGateID, "weighted-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-candidate-store-reader-proof-precondition-decision-promotion-switch-enable-gate-live-stage-final-gate-receiver-observation-boundary-preflight-gate-id-") ||
		!strings.HasPrefix(report.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReceiverObservationBoundaryPreflightGateCausalID, "weighted-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-candidate-store-reader-proof-precondition-decision-promotion-switch-enable-gate-live-stage-final-gate-receiver-observation-boundary-preflight-gate-causal-") ||
		!strings.HasPrefix(report.SourceAdmissionFinalGateObservationBoundaryPreflightGateHash, "weighted-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-candidate-store-reader-proof-precondition-decision-promotion-switch-enable-gate-live-stage-final-gate-receiver-observation-boundary-preflight-gate-") ||
		!strings.HasPrefix(report.SourceAdmissionFinalGateObservationBoundaryPreflightGateReadBackHash, "weighted-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-candidate-store-reader-proof-precondition-decision-promotion-switch-enable-gate-live-stage-final-gate-receiver-observation-boundary-preflight-gate-read-") {
		return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage final gate receiver observation boundary preflight gate candidate store reader proof source gate proof mismatch")
	}
	if !strings.HasPrefix(report.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReceiverObservationBoundaryPreflightID, "weighted-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-candidate-store-reader-proof-precondition-decision-promotion-switch-enable-gate-live-stage-final-gate-receiver-observation-boundary-preflight-id-") ||
		!strings.HasPrefix(report.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReceiverObservationBoundaryID, "weighted-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-candidate-store-reader-proof-precondition-decision-promotion-switch-enable-gate-live-stage-final-gate-receiver-observation-boundary-id-") ||
		!strings.HasPrefix(report.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReceiverObservationID, "weighted-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-candidate-store-reader-proof-precondition-decision-promotion-switch-enable-gate-live-stage-final-gate-receiver-observation-id-") ||
		!strings.HasPrefix(report.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReceiverID, "weighted-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-candidate-store-reader-proof-precondition-decision-promotion-switch-enable-gate-live-stage-final-gate-receiver-id-") {
		return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage final gate receiver observation boundary preflight gate candidate store reader proof source chain prefix mismatch")
	}
	if report.CausalID == "" || report.CausalID != admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReceiverObservationBoundaryPreflightGateCandidateStoreReaderProofCausalID(report) {
		return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage final gate receiver observation boundary preflight gate candidate store reader proof causal_id mismatch")
	}
	if report.ProofHash == "" || report.ProofHash != admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReceiverObservationBoundaryPreflightGateCandidateStoreReaderProofHash(report) {
		return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage final gate receiver observation boundary preflight gate candidate store reader proof proof_hash mismatch")
	}
	if report.ReadBackHash == "" || report.ReadBackHash != admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReceiverObservationBoundaryPreflightGateCandidateStoreReaderProofReadBackHash(report) {
		return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage final gate receiver observation boundary preflight gate candidate store reader proof read_back_hash mismatch")
	}
	if report.ProofHash == report.ReadBackHash {
		return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage final gate receiver observation boundary preflight gate candidate store reader proof read-back proof collapsed")
	}
	if report.WeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReceiverObservationBoundaryPreflightGateCandidateStoreReaderProofID == "" ||
		report.WeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReceiverObservationBoundaryPreflightGateCandidateStoreReaderProofID != admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReceiverObservationBoundaryPreflightGateCandidateStoreReaderProofID(report) {
		return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage final gate receiver observation boundary preflight gate candidate store reader proof id mismatch")
	}
	if report.Reason != "weighted resonance shadow graft admission final gate observation boundary preflight gate candidate store reader proof sealed without ledger append or body mutation" {
		return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage final gate receiver observation boundary preflight gate candidate store reader proof reason mismatch: got %q", report.Reason)
	}
	return nil
}

func admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReceiverObservationBoundaryPreflightGateCandidateStoreReaderProofCausalID(proof admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReceiverObservationBoundaryPreflightGateCandidateStoreReaderProofReport) string {
	h := hashJSON(struct {
		SourceReaderID       string `json:"source_reader_id"`
		SourceReaderReadBack string `json:"source_reader_read_back_hash"`
		SourceStoreID        string `json:"source_store_id"`
		SourceStoreReadBack  string `json:"source_store_read_back_hash"`
		SourceCandidateID    string `json:"source_candidate_id"`
		SourceGateID         string `json:"source_gate_id"`
		SourcePreflightID    string `json:"source_preflight_id"`
		SourceBoundaryID     string `json:"source_boundary_id"`
		SourceObservationID  string `json:"source_observation_id"`
		Target               string `json:"target"`
		ProofKind            string `json:"proof_kind"`
		ProofStage           string `json:"proof_stage"`
	}{
		SourceReaderID:       proof.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReceiverObservationBoundaryPreflightGateCandidateStoreReaderID,
		SourceReaderReadBack: proof.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReceiverObservationBoundaryPreflightGateCandidateStoreReaderReadBackHash,
		SourceStoreID:        proof.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReceiverObservationBoundaryPreflightGateCandidateStoreID,
		SourceStoreReadBack:  proof.SourceAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReadBackHash,
		SourceCandidateID:    proof.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReceiverObservationBoundaryPreflightGateCandidateID,
		SourceGateID:         proof.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReceiverObservationBoundaryPreflightGateID,
		SourcePreflightID:    proof.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReceiverObservationBoundaryPreflightID,
		SourceBoundaryID:     proof.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReceiverObservationBoundaryID,
		SourceObservationID:  proof.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReceiverObservationID,
		Target:               proof.Target,
		ProofKind:            proof.ProofKind,
		ProofStage:           proof.ProofStage,
	})
	if h == "" {
		return ""
	}
	return "weighted-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-candidate-store-reader-proof-precondition-decision-promotion-switch-enable-gate-live-stage-final-gate-receiver-observation-boundary-preflight-gate-candidate-store-reader-proof-causal-" + h
}

func admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReceiverObservationBoundaryPreflightGateCandidateStoreReaderProofHash(proof admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReceiverObservationBoundaryPreflightGateCandidateStoreReaderProofReport) string {
	h := hashJSON(struct {
		CausalID               string `json:"causal_id"`
		SourceReaderID         string `json:"source_reader_id"`
		SourceReaderHash       string `json:"source_reader_hash"`
		SourceReaderReplayHash string `json:"source_reader_replay_hash"`
		SourceReaderReadBack   string `json:"source_reader_read_back_hash"`
		SourceStoreID          string `json:"source_store_id"`
		SourceStoreHash        string `json:"source_store_hash"`
		SourceStoreReadBack    string `json:"source_store_read_back_hash"`
		ProofMode              string `json:"proof_mode"`
		ReceiptShape           string `json:"receipt_shape"`
		StoreReaderVerified    bool   `json:"store_reader_verified"`
		StoreVerified          bool   `json:"store_verified"`
		CandidateVerified      bool   `json:"candidate_verified"`
		ReaderHashVerified     bool   `json:"reader_hash_verified"`
		ReaderReplayVerified   bool   `json:"reader_replay_verified"`
		ReaderReadBackVerified bool   `json:"reader_read_back_verified"`
		StoreHashVerified      bool   `json:"store_hash_verified"`
		StoreReadBackVerified  bool   `json:"store_read_back_verified"`
		ReadOnly               bool   `json:"read_only"`
		ReplayOnly             bool   `json:"replay_only"`
		AdmissionRequired      bool   `json:"admission_required"`
		ShadowOnly             bool   `json:"shadow_only"`
		DryRunOnly             bool   `json:"dry_run_only"`
		GraftAllowed           bool   `json:"graft_allowed"`
		LedgerAppendAllowed    bool   `json:"ledger_append_allowed"`
	}{
		CausalID:               proof.CausalID,
		SourceReaderID:         proof.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReceiverObservationBoundaryPreflightGateCandidateStoreReaderID,
		SourceReaderHash:       proof.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReceiverObservationBoundaryPreflightGateCandidateStoreReaderHash,
		SourceReaderReplayHash: proof.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReceiverObservationBoundaryPreflightGateCandidateStoreReaderReplayHash,
		SourceReaderReadBack:   proof.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReceiverObservationBoundaryPreflightGateCandidateStoreReaderReadBackHash,
		SourceStoreID:          proof.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReceiverObservationBoundaryPreflightGateCandidateStoreID,
		SourceStoreHash:        proof.SourceAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreHash,
		SourceStoreReadBack:    proof.SourceAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReadBackHash,
		ProofMode:              proof.ProofMode,
		ReceiptShape:           proof.ReceiptShape,
		StoreReaderVerified:    proof.StoreReaderVerified,
		StoreVerified:          proof.StoreVerified,
		CandidateVerified:      proof.CandidateVerified,
		ReaderHashVerified:     proof.ReaderHashVerified,
		ReaderReplayVerified:   proof.ReaderReplayVerified,
		ReaderReadBackVerified: proof.ReaderReadBackVerified,
		StoreHashVerified:      proof.StoreHashVerified,
		StoreReadBackVerified:  proof.StoreReadBackVerified,
		ReadOnly:               proof.ReadOnly,
		ReplayOnly:             proof.ReplayOnly,
		AdmissionRequired:      proof.AdmissionRequired,
		ShadowOnly:             proof.ShadowOnly,
		DryRunOnly:             proof.DryRunOnly,
		GraftAllowed:           proof.GraftAllowed,
		LedgerAppendAllowed:    proof.LedgerAppendAllowed,
	})
	if h == "" {
		return ""
	}
	return "weighted-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-candidate-store-reader-proof-precondition-decision-promotion-switch-enable-gate-live-stage-final-gate-receiver-observation-boundary-preflight-gate-candidate-store-reader-proof-" + h
}

func admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReceiverObservationBoundaryPreflightGateCandidateStoreReaderProofReadBackHash(proof admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReceiverObservationBoundaryPreflightGateCandidateStoreReaderProofReport) string {
	h := hashJSON(struct {
		ProofHash           string `json:"proof_hash"`
		SourceReaderID      string `json:"source_reader_id"`
		SourceStoreID       string `json:"source_store_id"`
		SourceCandidate     string `json:"source_candidate_id"`
		ProofKind           string `json:"proof_kind"`
		ProofReady          bool   `json:"proof_ready"`
		BodyMutation        bool   `json:"body_mutation"`
		LiveAdmission       bool   `json:"live_admission"`
		WriteAllowed        bool   `json:"write_allowed"`
		AdmissionAllowed    bool   `json:"admission_allowed"`
		LedgerAppendAllowed bool   `json:"ledger_append_allowed"`
	}{
		ProofHash:           proof.ProofHash,
		SourceReaderID:      proof.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReceiverObservationBoundaryPreflightGateCandidateStoreReaderID,
		SourceStoreID:       proof.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReceiverObservationBoundaryPreflightGateCandidateStoreID,
		SourceCandidate:     proof.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReceiverObservationBoundaryPreflightGateCandidateID,
		ProofKind:           proof.ProofKind,
		ProofReady:          proof.WeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReceiverObservationBoundaryPreflightGateCandidateStoreReaderProofReady,
		BodyMutation:        proof.BodyMutationAllowed,
		LiveAdmission:       proof.LiveAdmissionEnabled,
		WriteAllowed:        proof.WriteAllowed,
		AdmissionAllowed:    proof.AdmissionAllowed,
		LedgerAppendAllowed: proof.LedgerAppendAllowed,
	})
	if h == "" {
		return ""
	}
	return "weighted-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-candidate-store-reader-proof-precondition-decision-promotion-switch-enable-gate-live-stage-final-gate-receiver-observation-boundary-preflight-gate-candidate-store-reader-proof-read-" + h
}

func admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReceiverObservationBoundaryPreflightGateCandidateStoreReaderProofID(proof admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReceiverObservationBoundaryPreflightGateCandidateStoreReaderProofReport) string {
	h := hashJSON(struct {
		Schema                 string `json:"schema"`
		Status                 string `json:"status"`
		Action                 string `json:"action"`
		SourceReport           string `json:"source_report"`
		SourceReaderID         string `json:"source_reader_id"`
		SourceStoreID          string `json:"source_store_id"`
		SourceCandidateID      string `json:"source_candidate_id"`
		SourceGateID           string `json:"source_gate_id"`
		SourcePreflightID      string `json:"source_preflight_id"`
		SourceBoundaryID       string `json:"source_boundary_id"`
		SourceObservationID    string `json:"source_observation_id"`
		SourceReceiverID       string `json:"source_receiver_id"`
		CausalID               string `json:"causal_id"`
		ProofHash              string `json:"proof_hash"`
		ReadBackHash           string `json:"read_back_hash"`
		Ready                  bool   `json:"ready"`
		ReceiptShape           string `json:"receipt_shape"`
		ProofKind              string `json:"proof_kind"`
		ProofMode              string `json:"proof_mode"`
		ProofStage             string `json:"proof_stage"`
		StoreReaderVerified    bool   `json:"store_reader_verified"`
		StoreVerified          bool   `json:"store_verified"`
		CandidateVerified      bool   `json:"candidate_verified"`
		GateVerified           bool   `json:"gate_verified"`
		PreflightVerified      bool   `json:"preflight_verified"`
		BoundaryVerified       bool   `json:"boundary_verified"`
		ObservationVerified    bool   `json:"observation_verified"`
		ReceiverVerified       bool   `json:"receiver_verified"`
		IntentVerified         bool   `json:"intent_verified"`
		FinalGateVerified      bool   `json:"final_gate_verified"`
		SealVerified           bool   `json:"seal_verified"`
		PermitVerified         bool   `json:"permit_verified"`
		AuthorityVerified      bool   `json:"authority_verified"`
		ReaderHashVerified     bool   `json:"reader_hash_verified"`
		ReaderReplayVerified   bool   `json:"reader_replay_verified"`
		ReaderReadBackVerified bool   `json:"reader_read_back_verified"`
		StoreHashVerified      bool   `json:"store_hash_verified"`
		StoreReadBackVerified  bool   `json:"store_read_back_verified"`
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
		NextStepBlockedWithout bool   `json:"next_step_blocked_without_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof"`
		SourceReaderReady      bool   `json:"source_reader_ready"`
		SourceStoreReady       bool   `json:"source_store_ready"`
		SourceCandidateReady   bool   `json:"source_candidate_ready"`
		SourceGateReady        bool   `json:"source_gate_ready"`
		SourcePreflightReady   bool   `json:"source_preflight_ready"`
		SourceBoundaryReady    bool   `json:"source_boundary_ready"`
		SourceObservationReady bool   `json:"source_observation_ready"`
		SourceReceiverReady    bool   `json:"source_receiver_ready"`
		SourceIntentReady      bool   `json:"source_intent_ready"`
		SourceFinalGateReady   bool   `json:"source_final_gate_ready"`
		SourceSealReady        bool   `json:"source_seal_ready"`
		SourceAuthorityReady   bool   `json:"source_authority_ready"`
		SourcePermitReady      bool   `json:"source_permit_ready"`
	}{
		Schema:                 proof.Schema,
		Status:                 proof.Status,
		Action:                 proof.Action,
		SourceReport:           proof.SourceReport,
		SourceReaderID:         proof.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReceiverObservationBoundaryPreflightGateCandidateStoreReaderID,
		SourceStoreID:          proof.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReceiverObservationBoundaryPreflightGateCandidateStoreID,
		SourceCandidateID:      proof.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReceiverObservationBoundaryPreflightGateCandidateID,
		SourceGateID:           proof.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReceiverObservationBoundaryPreflightGateID,
		SourcePreflightID:      proof.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReceiverObservationBoundaryPreflightID,
		SourceBoundaryID:       proof.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReceiverObservationBoundaryID,
		SourceObservationID:    proof.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReceiverObservationID,
		SourceReceiverID:       proof.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReceiverID,
		CausalID:               proof.CausalID,
		ProofHash:              proof.ProofHash,
		ReadBackHash:           proof.ReadBackHash,
		Ready:                  proof.WeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReceiverObservationBoundaryPreflightGateCandidateStoreReaderProofReady,
		ReceiptShape:           proof.ReceiptShape,
		ProofKind:              proof.ProofKind,
		ProofMode:              proof.ProofMode,
		ProofStage:             proof.ProofStage,
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
		AdmissionRequired:      proof.AdmissionRequired,
		ShadowOnly:             proof.ShadowOnly,
		GraftAllowed:           proof.GraftAllowed,
		DryRunOnly:             proof.DryRunOnly,
		ReadOnly:               proof.ReadOnly,
		ReplayOnly:             proof.ReplayOnly,
		LiveReady:              proof.LiveReady,
		ContractsReady:         proof.ContractsReady,
		BodyTarget:             proof.BodyTarget,
		WriteAllowed:           proof.WriteAllowed,
		AdmissionAllowed:       proof.AdmissionAllowed,
		LiveAdmissionEnabled:   proof.LiveAdmissionEnabled,
		MutatesState:           proof.MutatesState,
		LedgerAppendAllowed:    proof.LedgerAppendAllowed,
		NextStepBlockedWithout: proof.NextStepBlockedWithoutResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProof,
		SourceReaderReady:      proof.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReceiverObservationBoundaryPreflightGateCandidateStoreReaderReady,
		SourceStoreReady:       proof.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReceiverObservationBoundaryPreflightGateCandidateStoreReady,
		SourceCandidateReady:   proof.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReceiverObservationBoundaryPreflightGateCandidateReady,
		SourceGateReady:        proof.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReceiverObservationBoundaryPreflightGateReady,
		SourcePreflightReady:   proof.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReceiverObservationBoundaryPreflightReady,
		SourceBoundaryReady:    proof.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReceiverObservationBoundaryReady,
		SourceObservationReady: proof.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReceiverObservationReady,
		SourceReceiverReady:    proof.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReceiverReady,
		SourceIntentReady:      proof.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateIntentReady,
		SourceFinalGateReady:   proof.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReady,
		SourceSealReady:        proof.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageSealReady,
		SourceAuthorityReady:   proof.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageAuthorityReady,
		SourcePermitReady:      proof.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStagePermitReady,
	})
	if h == "" {
		return ""
	}
	return "weighted-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-candidate-store-reader-proof-precondition-decision-promotion-switch-enable-gate-live-stage-final-gate-receiver-observation-boundary-preflight-gate-candidate-store-reader-proof-id-" + h
}

func readAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReceiverObservationBoundaryPreflightGateCandidateStoreReaderProofReportForAssert(path string) (admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReceiverObservationBoundaryPreflightGateCandidateStoreReaderProofReport, map[string]json.RawMessage, error) {
	var report admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReceiverObservationBoundaryPreflightGateCandidateStoreReaderProofReport
	if strings.TrimSpace(path) == "" {
		return report, nil, fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage final gate receiver observation boundary preflight gate candidate store reader proof path missing")
	}
	raw, err := os.ReadFile(path)
	if err != nil {
		if os.IsNotExist(err) {
			return report, nil, fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage final gate receiver observation boundary preflight gate candidate store reader proof not written")
		}
		return report, nil, err
	}
	if len(raw) == 0 {
		return report, nil, fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage final gate receiver observation boundary preflight gate candidate store reader proof not written")
	}
	var root map[string]json.RawMessage
	if err := json.Unmarshal(raw, &root); err != nil {
		return report, nil, fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage final gate receiver observation boundary preflight gate candidate store reader proof JSON invalid: %w", err)
	}
	if err := json.Unmarshal(raw, &report); err != nil {
		return report, nil, fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage final gate receiver observation boundary preflight gate candidate store reader proof decode failed: %w", err)
	}
	return report, root, nil
}
